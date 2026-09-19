"""Common safety/food planner with model Choice tie-breaking in Snake.

Ported from NanoJev (MIT, Copyright (c) 2026 OpenJev contributors):
  https://github.com/TianyuCodings/NanoJev/blob/main/scripts/evaluate_composed_snake.py

The planner and prompt strings are reproduced verbatim. That matters: the model's
decision is a function of the prompt text, so paraphrasing any of it would change
the trajectory and break comparability with the reference results.

The split of labour is the point of the architecture, not an implementation detail:

    code   filters immediately colliding moves and keeps the moves on a shortest
           static path to the currently visible food
    model  breaks ties between the surviving candidates (never called at all when
           only one candidate survives)

So this is a code planner with a model tie-breaker, not a learned Snake controller.
"""

from __future__ import annotations

import copy
import random
from collections import deque

from . import snake_game as snake


def food_distance(next_body, food, size):
    """Static BFS; the next tail stays blocked, a conservative approximation."""
    start, target = tuple(next_body[0]), tuple(food)
    if start == target:
        return 0
    blocked = set(map(tuple, next_body[1:]))
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        position, distance = queue.popleft()
        for dr, dc in snake.DIRECTIONS.values():
            cell = position[0] + dr, position[1] + dc
            if cell in seen or cell in blocked or not all(0 <= x < size for x in cell):
                continue
            if cell == target:
                return distance + 1
            seen.add(cell)
            queue.append((cell, distance + 1))
    return None


def plan_candidates(state):
    """Planning uses current food, the actual next body, and immediate collision only."""
    snake.validate_state(state)
    safe, distances = [], {}
    for action in snake.valid_actions(state):
        after = snake.step(state, action)
        if after["done"] and after["outcome"] != "win":
            continue
        safe.append(action)
        # Never inspect after['food'] or after['rng_state']: they would be future information.
        remaining = food_distance(after["body"], state["food"], state["size"])
        distances[action] = None if remaining is None else remaining + 1
    reachable = [action for action in safe if distances[action] is not None]
    offered = (
        [action for action in reachable if distances[action] == min(distances[a] for a in reachable)]
        if reachable
        else list(safe)
    )
    return {
        "safe_actions": safe,
        "offered_actions": offered,
        "static_total_distances": distances,
        "mode": "shortest_static_food_path" if reachable else "all_safe_no_static_food_path",
    }


def render_composed_request(state, offered_actions):
    """Stable model input: physical state and one dynamic Choice; no planner scores."""
    base = snake.render_request(state)
    candidates = base["questions"]["action"]["criteria"]
    if (
        len(offered_actions) < 2
        or len(set(offered_actions)) != len(offered_actions)
        or not set(offered_actions) <= set(candidates)
    ):
        raise ValueError("A model Choice requires at least two distinct non-reverse candidates")
    question = {
        "type": "choice",
        "instructions": "Choose the offered move that best preserves space for subsequent movement while collecting the currently visible food. "
        "A common code planner has selected the offered candidates. Use only the current physical state; tied moves are equivalent.",
        "criteria": {action: candidates[action] for action in offered_actions},
    }
    return {"state": base["state"], "questions": {"action": question}}


def physical_state(state):
    return {
        key: copy.deepcopy(state[key])
        for key in ("game", "size", "body", "direction", "food", "done", "outcome", "score", "steps")
    }


def validate_choice(answer, candidates):
    import math

    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("Expected a Choice answer")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(candidates):
        raise ValueError("Choice probabilities must cover exactly the offered candidates")
    if any(
        isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise ValueError("Invalid Choice probability")
    if abs(sum(probabilities.values()) - 1) > 1e-5:
        raise ValueError("Choice probabilities must sum to one")
    return {action: probabilities[action] for action in candidates}


def choose_action(probabilities, controller, rng):
    actions = list(probabilities)
    if controller == "greedy":
        return max(actions, key=probabilities.__getitem__), None
    draw, cumulative = rng.random(), 0.0
    total = sum(probabilities.values())
    for action in actions:
        cumulative += probabilities[action] / total
        if draw < cumulative:
            return action, draw
    return actions[-1], draw


class MlxEngine:
    """Adapts the MLX predictor to the engine interface the original controller expects.

    The original rollout calls `engine.predict({"states": [...]}, batch_questions=..., temperature=1.0)`
    and reads `response["states"]` / `response["execution"]`. nanojev-mlx already returns exactly
    that shape, so this adapter is the entire integration surface.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, payload, batch_questions=0, temperature=1.0):
        from nanojev_mlx import run_prediction

        return run_prediction(
            self.model, payload, temperature=temperature, batch_questions=batch_questions
        )


class EpisodeSession:
    """One Snake episode, advanced one decision at a time.

    `run_episode` drives this in a loop; the live server drives it from HTTP requests.
    Both therefore execute exactly the same code, so a live game cannot drift from a
    recorded one -- and the tests that cover the loop cover both.

    Pass `override_action` to `step()` to play one move yourself. The model is still
    queried first, so you can see what it thought before you overrode it, and the
    override is validated against the planner's candidates -- a human cannot play a
    move the planner would have filtered out as colliding.
    """

    def __init__(self, engine, size=12, seed=61005, controller="greedy", max_steps=256):
        if controller not in ("greedy", "sample"):
            raise ValueError("controller must be 'greedy' or 'sample'")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.engine = engine
        self.size = size
        self.seed = seed
        self.controller = controller
        self.max_steps = max_steps
        self.state = snake.make_snake(size, seed)
        self.initial_state = copy.deepcopy(self.state)
        self.rng = random.Random(seed)
        self.trace = []
        self.status = None

    @property
    def done(self) -> bool:
        return self.status is not None

    def _settle(self):
        """Set the terminal status if the episode is over as of now."""
        if self.state["done"]:
            self.status = self.state["outcome"]
        elif len(self.trace) >= self.max_steps:
            self.status = "horizon_survived"

    def step(self, override_action=None):
        """Advance one decision. Returns the step record, or None if the snake is
        trapped -- no legal move left, so the episode ends without one."""
        if self.done:
            raise ValueError("episode already finished")
        self._settle()
        if self.done:
            return None

        plan = plan_candidates(self.state)
        actions = plan["offered_actions"]
        if not actions:
            self.status = "trapped"
            return None

        before = self.state
        if len(actions) == 1:
            probabilities, actor = {actions[0]: 1.0}, "forced_move"
        else:
            request = {
                "id": f"snake:{self.seed}:step:{len(self.trace)}",
                **render_composed_request(before, actions),
            }
            response = self.engine.predict({"states": [copy.deepcopy(request)]}, temperature=1.0)
            returned = response.get("states", [])
            if len(returned) != 1 or returned[0].get("id") != request["id"]:
                raise ValueError("Choice response ID mismatch")
            answer = returned[0].get("answers", {}).get("action")
            probabilities, actor = validate_choice(answer, actions), "model_tiebreak"

        if override_action is not None:
            if override_action not in actions:
                raise ValueError(f"{override_action!r} is not among the offered moves {actions}")
            chosen, draw, actor = override_action, None, "human_override"
        elif actor == "forced_move":
            chosen, draw = next(iter(probabilities)), None
        else:
            chosen, draw = choose_action(probabilities, self.controller, self.rng)

        after = snake.step(before, chosen)
        if after["done"] and after["outcome"] != "win":
            raise RuntimeError("Common planner admitted an immediately colliding action")

        total = sum(probabilities.values())
        record = {
            "step_index": len(self.trace),
            "state": physical_state(before),
            "action": chosen,
            "actor": actor,
            "probabilities": probabilities,
            "sampling_probabilities": {k: v / total for k, v in probabilities.items()},
            "controller_draw": draw,
            "candidate_order": list(probabilities),
            "planner": plan,
            "ate_food": after["score"] > before["score"],
            "next_state": physical_state(after),
        }
        self.trace.append(record)
        self.state = after
        self._settle()
        return record

    def snapshot(self) -> dict:
        """Everything a client needs to draw the current position."""
        return {
            "state": physical_state(self.state),
            "done": self.done,
            "outcome": self.status,
            "steps": len(self.trace),
            "food_collected": self.state["score"] - self.initial_state["score"],
            "model_decisions": sum(r["actor"] == "model_tiebreak" for r in self.trace),
            "forced_moves": sum(r["actor"] == "forced_move" for r in self.trace),
            "human_overrides": sum(r["actor"] == "human_override" for r in self.trace),
        }

    def result(self) -> dict:
        if not self.done:
            raise ValueError("episode is still running")
        final = physical_state(self.state)
        trace = self.trace
        return {
            "schema": "nanojev-mlx-snake-demo-v1",
            "size": self.size,
            "seed": self.seed,
            "controller": self.controller,
            "horizon": self.max_steps,
            "initial_state": physical_state(self.initial_state),
            "final_state": final,
            "outcome": self.status,
            "full_board_win": self.status == "win",
            "horizon_survived": self.status == "horizon_survived",
            "survival_steps": len(trace),
            "food_collected": final["score"] - self.initial_state["score"],
            "forced_moves": sum(row["actor"] == "forced_move" for row in trace),
            "model_decisions": sum(row["actor"] == "model_tiebreak" for row in trace),
            "human_overrides": sum(row["actor"] == "human_override" for row in trace),
            "steps": trace,
        }


def run_episode(engine, size=12, seed=61005, controller="greedy", max_steps=256):
    """Play one episode to completion. Returns the recorded trace plus statistics."""
    session = EpisodeSession(engine, size=size, seed=seed, controller=controller, max_steps=max_steps)
    while not session.done:
        session.step()
    return session.result()
