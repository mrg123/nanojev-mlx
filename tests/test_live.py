"""Live demo server tests.

Starts the real HTTP server on an ephemeral loopback port with a stub engine, so the
actual request path is exercised -- not a mock of it. Needs no checkpoint and no MLX,
so it runs in CI.

    python -m unittest discover -s tests -p "test_live.py" -v
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.live import SnakeDemo, handler_class  # noqa: E402


class StubEngine:
    """Always picks the first offered candidate. Deterministic and free."""

    def __init__(self):
        self.calls = 0

    def predict(self, payload, batch_questions=0, temperature=1.0):
        self.calls += 1
        states = []
        for row in payload["states"]:
            candidates = list(row["questions"]["action"]["criteria"])
            share = 1.0 / len(candidates)
            states.append(
                {
                    "id": row["id"],
                    "answers": {"action": {"type": "choice", "probabilities": {a: share for a in candidates}}},
                }
            )
        return {"states": states, "execution": {"forward_passes": 0, "network_model_calls": 0}}


class LiveServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = StubEngine()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class(SnakeDemo(cls.engine)))
        cls.server.daemon_threads = True
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def call(self, path, body=None, method="POST"):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            url, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read())
            finally:
                exc.close()

    # -- page and health --------------------------------------------------
    def test_serves_the_page(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=30) as response:
            page = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("<html", page)
        self.assertIn("/api/snake/step", page)
        # Self-contained: no CDN, no external asset.
        self.assertNotIn("https://", page)

    def test_health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/health", timeout=30) as response:
            body = json.loads(response.read())
        self.assertTrue(body["ready"])
        self.assertEqual(body["runtime"], "mlx")

    def test_an_abandoned_keepalive_connection_does_not_freeze_the_server(self):
        """Regression: a browser drops keep-alive connections without warning.

        On reload, navigation or tab close the client just goes away. With a
        single-threaded HTTPServer talking HTTP/1.1, the handler then blocked forever on
        `rfile.readline()` waiting for a next request that never came -- and since there
        was only one thread, every later request hung too. That froze the whole app.

        The socket is deliberately left OPEN while the second request is made: that is
        precisely the state that used to deadlock the server.
        """
        abandoned = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            abandoned.sendall(b"GET /api/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            self.assertIn(b"200", abandoned.recv(64))
            # The server is now parked waiting for another request on this connection.
            request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/health")
            with urllib.request.urlopen(request, timeout=15) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(json.loads(response.read())["ready"])
        finally:
            abandoned.close()

    def test_concurrent_requests_are_served(self):
        """Threading must not lose the serialisation of model work, nor deadlock it."""
        _, game = self.call("/api/snake/new", {"size": 8, "seed": 6, "max_steps": 30})
        key = game["session"]
        results, errors = [], []

        def worker():
            try:
                results.append(self.call("/api/snake/step", {"session": key})[0])
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [], f"concurrent requests raised: {errors}")
        self.assertEqual(results, [200] * 4)

    # -- session protocol -------------------------------------------------
    def test_new_game_returns_an_initial_position(self):
        status, body = self.call("/api/snake/new", {"size": 8, "seed": 3, "max_steps": 30})
        self.assertEqual(status, 200)
        self.assertEqual(body["size"], 8)
        self.assertEqual(body["seed"], 3)
        self.assertFalse(body["done"])
        self.assertEqual(body["steps"], 0)
        self.assertEqual(len(body["state"]["body"]), 3)

    def test_step_advances_one_move(self):
        _, game = self.call("/api/snake/new", {"size": 8, "seed": 3, "max_steps": 30})
        status, body = self.call("/api/snake/step", {"session": game["session"]})
        self.assertEqual(status, 200)
        self.assertEqual(body["steps"], 1)
        self.assertIsNotNone(body["step"])
        self.assertIn(body["step"]["action"], body["step"]["candidate_order"])

    def test_a_whole_episode_runs_to_the_horizon(self):
        _, game = self.call("/api/snake/new", {"size": 8, "seed": 4, "max_steps": 25})
        key, steps = game["session"], 0
        for _ in range(200):
            _, body = self.call("/api/snake/step", {"session": key})
            steps += 1
            if body["done"]:
                break
        self.assertTrue(body["done"], "episode never finished")
        self.assertLessEqual(steps, 25, "played past the horizon")
        self.assertNotIn(body["outcome"], {"wall_collision", "self_collision"})

    def test_unknown_session_is_rejected(self):
        status, body = self.call("/api/snake/step", {"session": "nope"})
        self.assertEqual(status, 400)
        self.assertIn("session", body["error"])

    # -- human takeover ---------------------------------------------------
    def test_override_is_honoured_and_counted(self):
        _, game = self.call("/api/snake/new", {"size": 10, "seed": 12, "max_steps": 60})
        key = game["session"]
        for _ in range(60):
            _, body = self.call("/api/snake/step", {"session": key})
            if body["done"]:
                self.skipTest("episode ended before a two-candidate decision appeared")
            step = body["step"]
            if step is None or step["actor"] != "model_tiebreak":
                continue
            pick = step["candidate_order"][-1]  # deliberately not necessarily the argmax
            _, body2 = self.call("/api/snake/step", {"session": key, "action": pick})
            if body2["step"] is None:
                self.skipTest("episode became trapped right after the decision")
            self.assertEqual(body2["step"]["action"], pick)
            self.assertEqual(body2["step"]["actor"], "human_override")
            self.assertEqual(body2["human_overrides"], 1)
            # The model was still consulted, so its probabilities are on the record.
            self.assertEqual(set(body2["step"]["candidate_order"]), set(step["candidate_order"]))
            return
        self.fail("no model decision appeared in 60 steps")

    def test_a_move_outside_the_candidates_is_rejected(self):
        """Overriding is confined to the planner's candidates -- you cannot bypass it."""
        _, game = self.call("/api/snake/new", {"size": 10, "seed": 12, "max_steps": 60})
        key = game["session"]
        _, body = self.call("/api/snake/step", {"session": key})
        step = body["step"]
        blocked = sorted({"north", "east", "south", "west"} - set(step["candidate_order"]))
        for bad in ["sideways"] + blocked:
            status, response = self.call("/api/snake/step", {"session": key, "action": bad})
            self.assertEqual(status, 400, f"{bad!r} should have been rejected")
            self.assertIn(bad, response["error"])

    def test_an_all_override_episode_never_collides(self):
        """Every accepted override must be a move the planner already cleared."""
        _, game = self.call("/api/snake/new", {"size": 10, "seed": 21, "max_steps": 80})
        key = game["session"]
        overrides = 0
        for _ in range(200):
            _, body = self.call("/api/snake/state", {"session": key})
            if body["done"]:
                break
            # Ask the model for the candidates by stepping normally, then undo is not
            # possible -- so instead override whenever we know the candidates.
            _, body = self.call("/api/snake/step", {"session": key})
            step = body["step"]
            if body["done"]:
                break
            if step and step["actor"] == "model_tiebreak":
                pick = step["candidate_order"][-1]
                _, body = self.call("/api/snake/step", {"session": key, "action": pick})
                overrides += 1
                if body["done"]:
                    break
        self.assertGreater(overrides, 0, "no override was exercised")
        _, final = self.call("/api/snake/state", {"session": key})
        self.assertNotIn(final["outcome"], {"wall_collision", "self_collision"})
        self.assertEqual(final["human_overrides"], overrides)


if __name__ == "__main__":
    unittest.main()
