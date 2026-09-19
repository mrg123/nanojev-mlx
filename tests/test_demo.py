"""Demo tests: Snake environment, code planner, and episode loop.

These need **no checkpoint and no MLX** -- the engine is a stub -- so they run in
CI alongside the protocol tests. That is deliberate: the demo is the most visible
part of the repository, so it should not be the only untested part.

    python -m unittest discover -s tests -p "test_demo.py" -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo import snake_game as snake  # noqa: E402
from demo.controller import (  # noqa: E402
    choose_action,
    plan_candidates,
    render_composed_request,
    run_episode,
    validate_choice,
)


class FirstChoiceEngine:
    """Picks the first offered candidate with uniform probabilities.

    Mirrors UniformChoice from the original rollout: a valid engine with no model
    behind it, so the controller can be exercised deterministically.
    """

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


class SnakeEnvironmentTest(unittest.TestCase):
    def test_initial_state_is_valid_and_deterministic(self):
        first, second = snake.make_snake(8, 42), snake.make_snake(8, 42)
        self.assertEqual(first, second)
        snake.validate_state(first)
        self.assertEqual(len(first["body"]), 3)

    def test_different_seeds_give_different_food(self):
        boards = {tuple(snake.make_snake(8, s)["food"]) for s in range(12)}
        self.assertGreater(len(boards), 1)

    def test_step_into_wall_is_terminal(self):
        state = snake.make_snake(4, 7)
        # head-neck orientation must agree with `direction`: head (0,2) minus neck
        # (0,1) is (0, 1) == east.
        state["body"] = [[0, 2], [0, 1], [0, 0]]
        state["direction"] = "east"
        state["food"] = [2, 2]
        snake.validate_state(state)
        after = snake.step(state, "north")
        self.assertTrue(after["done"])
        self.assertEqual(after["outcome"], "wall_collision")


class PlannerTest(unittest.TestCase):
    def test_offered_actions_are_always_immediately_safe(self):
        for size in (6, 8, 12):
            for seed in range(6):
                state = snake.make_snake(size, seed)
                for _ in range(40):
                    if state["done"]:
                        break
                    plan = plan_candidates(state)
                    for action in plan["offered_actions"]:
                        self.assertIn(action, plan["safe_actions"])
                        self.assertTrue(
                            snake.one_step_safe(state, action),
                            f"{action} was offered but collides (size={size} seed={seed})",
                        )
                    if not plan["offered_actions"]:
                        # A trapped snake is a legitimate outcome: every legal move
                        # collides, so the planner correctly offers nothing.
                        self.assertEqual(plan["safe_actions"], [])
                        break
                    state = snake.step(state, plan["offered_actions"][0])

    def test_planner_never_offers_a_reverse_move(self):
        state = snake.make_snake(8, 3)
        plan = plan_candidates(state)
        self.assertNotIn(snake.REVERSE[state["direction"]], plan["offered_actions"])

    def test_request_shape_matches_what_the_model_accepts(self):
        # A model Choice needs at least two candidates, so find a step that has them
        # rather than assuming the opening position does.
        state = next(
            s
            for seed in range(40)
            for s in [snake.make_snake(12, seed)]
            if len(plan_candidates(s)["offered_actions"]) >= 2
        )
        plan = plan_candidates(state)
        request = render_composed_request(state, plan["offered_actions"])
        self.assertEqual(set(request), {"state", "questions"})
        question = request["questions"]["action"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(list(question["criteria"]), plan["offered_actions"])

    def test_controller_request_passes_the_ports_own_validator(self):
        """The demo and the model must agree on the wire format -- checked directly."""
        from nanojev_mlx.text import validate_request

        for seed in range(12):
            state = snake.make_snake(12, seed)
            plan = plan_candidates(state)
            if len(plan["offered_actions"]) < 2:
                continue
            request = {
                "id": f"snake:{seed}:step:0",
                **render_composed_request(state, plan["offered_actions"]),
            }
            states = validate_request({"states": [request]})
            self.assertEqual(len(states), 1)
            self.assertEqual(set(states[0]), {"id", "state", "questions"})
            self.assertEqual(states[0]["questions"]["action"]["type"], "choice")

    def test_render_rejects_a_single_candidate(self):
        state = snake.make_snake(8, 5)
        with self.assertRaises(ValueError):
            render_composed_request(state, ["north"])


class ChoiceValidationTest(unittest.TestCase):
    def test_rejects_probabilities_over_the_wrong_candidates(self):
        with self.assertRaises(ValueError):
            validate_choice({"type": "choice", "probabilities": {"north": 1.0}}, ["north", "south"])

    def test_rejects_unnormalized(self):
        with self.assertRaises(ValueError):
            validate_choice({"type": "choice", "probabilities": {"north": 0.4, "south": 0.4}}, ["north", "south"])

    def test_greedy_picks_the_argmax(self):
        action, draw = choose_action({"north": 0.1, "south": 0.9}, "greedy", None)
        self.assertEqual(action, "south")
        self.assertIsNone(draw)

    def test_greedy_breaks_ties_by_candidate_order(self):
        action, _ = choose_action({"north": 0.5, "south": 0.5}, "greedy", None)
        self.assertEqual(action, "north")


class EpisodeTest(unittest.TestCase):
    def test_episode_never_ends_in_a_collision(self):
        # The whole contract of the composed controller: code removes colliding
        # moves, so a collision would mean the planner admitted an unsafe action.
        for seed in range(8):
            engine = FirstChoiceEngine()
            episode = run_episode(engine, size=10, seed=seed, max_steps=60)
            self.assertNotIn(episode["outcome"], {"wall_collision", "self_collision"})
            self.assertEqual(episode["outcome"], "horizon_survived")

    def test_every_step_is_either_model_or_forced(self):
        episode = run_episode(FirstChoiceEngine(), size=10, seed=1, max_steps=40)
        actors = [row["actor"] for row in episode["steps"]]
        self.assertEqual(set(actors) <= {"model_tiebreak", "forced_move"}, True)
        self.assertEqual(episode["model_decisions"] + episode["forced_moves"], episode["survival_steps"])

    def test_model_is_not_called_for_forced_moves(self):
        engine = FirstChoiceEngine()
        episode = run_episode(engine, size=10, seed=1, max_steps=40)
        # One predict call per step that actually needed a decision.
        self.assertEqual(engine.calls, episode["model_decisions"])

    def test_snake_survives_and_that_is_observable(self):
        episode = run_episode(FirstChoiceEngine(), size=10, seed=2, max_steps=60)
        self.assertEqual(episode["steps"][-1]["next_state"]["steps"], episode["survival_steps"])
        for row in episode["steps"]:
            self.assertIn(row["action"], row["candidate_order"])
            totals = sum(row["probabilities"].values())
            self.assertAlmostEqual(totals, 1.0, places=6)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            run_episode(FirstChoiceEngine(), controller="random")
        with self.assertRaises(ValueError):
            run_episode(FirstChoiceEngine(), max_steps=0)


if __name__ == "__main__":
    unittest.main()
