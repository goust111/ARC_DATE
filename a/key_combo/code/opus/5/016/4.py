from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, in_bounds, neighbors4
except Exception:  # pragma: no cover - local fallback for isolated loading
    class Color:
        BLACK = 0
        TEAL = 8

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])

    def neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1


BG = int(Color.BLACK)
CROSS = int(Color.TEAL)
DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _add(pos: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return int(pos[0] + direction[0] * steps), int(pos[1] + direction[1] * steps)


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS_URDL[(DIRS_URDL.index(direction) + 1) % 4]


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    return DIRS_URDL[(DIRS_URDL.index(direction) - 1) % 4]


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    return int(r) == 0 or int(c) == 0 or int(r) == int(shape[0]) - 1 or int(c) == int(shape[1]) - 1


def _extract_cross(grid: np.ndarray) -> tuple[tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    teal = np.argwhere(g == CROSS)
    assert teal.size > 0, "expected a teal cross"

    centers: list[tuple[int, int]] = []
    for r, c in teal:
        rr = int(r)
        cc = int(c)
        if all(in_bounds(g.shape, rr + dr, cc + dc) and g[rr + dr, cc + dc] == CROSS for dr, dc in DIRS_URDL):
            centers.append((rr, cc))
    assert len(centers) == 1, "expected exactly one cross center"
    cr, cc = centers[0]

    coords = {(int(r), int(c)) for r, c in teal}
    assert all(r == cr or c == cc for r, c in coords), "foreground must be a 1-pixel-thick cross"

    arms: list[dict] = []
    seen = {(cr, cc)}
    for outward in DIRS_URDL:
        dr, dc = outward
        cells: list[tuple[int, int]] = []
        r, c = cr, cc
        while in_bounds(g.shape, r + dr, c + dc) and g[r + dr, c + dc] == CROSS:
            r += dr
            c += dc
            cells.append((int(r), int(c)))
            seen.add((int(r), int(c)))
        assert cells, "center must have four non-empty arms"
        arms.append(
            {
                "outward": outward,
                "length": int(len(cells)),
                "tip": cells[-1],
                "cells": cells,
            }
        )

    lengths = [int(arm["length"]) for arm in arms]
    assert len(set(lengths)) == 4, "arm lengths must be distinct"
    assert seen == coords, "cross must contain only center + four straight arms"
    assert any(length % 2 == 0 for length in lengths) and any(length % 2 == 1 for length in lengths)
    return (cr, cc), arms


def _spacing_ok(out: np.ndarray, cell: tuple[int, int], prev: tuple[int, int]) -> bool:
    r, c = cell
    if out[r, c] != BG:
        return False
    for nr, nc in neighbors4(r, c):
        if in_bounds(out.shape, nr, nc) and out[nr, nc] != BG and (int(nr), int(nc)) != prev:
            return False
    return True


def _can_walk(out: np.ndarray, cur: tuple[int, int], direction: tuple[int, int], length: int) -> bool:
    prev = cur
    for step in range(1, int(length) + 1):
        nxt = _add(cur, direction, step)
        if not in_bounds(out.shape, *nxt):
            return False
        if not _spacing_ok(out, nxt, prev):
            return False
        prev = nxt
    return True


def _do_walk(agent: dict, out: np.ndarray, direction: tuple[int, int], length: int) -> None:
    start = agent["cur"]
    for step in range(1, int(length) + 1):
        rr, cc = _add(start, direction, step)
        out[rr, cc] = CROSS
    agent["cur"] = _add(start, direction, length)
    agent["added"] += int(length)
    agent["moves"] += 1


def _step_spiral(agent: dict, out: np.ndarray) -> bool:
    if agent["done"]:
        return False

    cur = agent["cur"]
    options = [_right(agent["heading"]), agent["heading"]]
    if not _is_border(out.shape, *cur):
        options.append(_left(agent["heading"]))

    for direction in options:
        turned = direction != agent["heading"]
        step_len = 1 if turned else (1 if agent["post_turn"] else 2)
        if _can_walk(out, cur, direction, step_len):
            _do_walk(agent, out, direction, step_len)
            if turned:
                agent["turns"] += 1
            agent["post_turn"] = 1 if (agent["post_turn"] == 0 and turned) else 0
            agent["heading"] = direction
            return True

    agent["done"] = True
    return False


def _step_line(agent: dict, out: np.ndarray) -> bool:
    if agent["done"]:
        return False

    if _can_walk(out, agent["cur"], agent["heading"], 1):
        _do_walk(agent, out, agent["heading"], 1)
        return True

    agent["done"] = True
    return False


def _simulate(
    grid: np.ndarray,
    *,
    behavior_override: dict[int, str] | None = None,
    active_indices: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    center, arms = _extract_cross(g)
    active_set = set(range(4)) if active_indices is None else {int(i) for i in active_indices}

    out = g.copy()
    agents: list[dict] = []
    for idx, arm in enumerate(arms):
        if idx not in active_set:
            continue
        kind = "spiral" if int(arm["length"]) % 2 == 1 else "line"
        if behavior_override is not None and idx in behavior_override:
            kind = str(behavior_override[idx])
        heading = _right(arm["outward"]) if kind == "spiral" else arm["outward"]
        agents.append(
            {
                "index": int(idx),
                "kind": kind,
                "length": int(arm["length"]),
                "outward": arm["outward"],
                "tip": arm["tip"],
                "cur": arm["tip"],
                "heading": heading,
                "post_turn": 0,
                "done": False,
                "added": 0,
                "turns": 0,
                "moves": 0,
            }
        )

    guard = 0
    limit = int(g.shape[0] * g.shape[1] * 8)
    changed = True
    while changed and guard < limit:
        changed = False
        guard += 1
        for agent in agents:
            moved = _step_spiral(agent, out) if agent["kind"] == "spiral" else _step_line(agent, out)
            changed = changed or moved
        if all(agent["done"] for agent in agents):
            break

    summary = {
        "center": center,
        "arms": [
            {
                "index": int(agent["index"]),
                "kind": agent["kind"],
                "length": int(agent["length"]),
                "added": int(agent["added"]),
                "turns": int(agent["turns"]),
                "moves": int(agent["moves"]),
            }
            for agent in agents
        ],
    }
    return out, summary


def _independent_union(grid: np.ndarray) -> np.ndarray:
    pieces = [np.asarray(grid, dtype=int)]
    for idx in range(4):
        solo, _ = _simulate(grid, active_indices=[idx])
        pieces.append(solo)
    return np.maximum.reduce(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each arm of an asymmetric teal cross by a parity-selected path rule.

    Input structure:
    - Black background with exactly one 1-pixel-thick teal cross whose center pixel is present.
    - The four cardinal arms have distinct positive lengths measured from the center to each tip.
    - The cross is placed away from the border so extra growth can spread into open space.

    Rule:
    - Identify the center and the four arm tips in up, right, down, left order.
    - Odd-length arms launch spiral growth from their tips: initial heading is the outward direction rotated 90 degrees clockwise, and each turn decision prefers right > straight > left, with left turns allowed only when the current tip is not on the border.
    - Even-length arms launch straight growth from their tips along the outward direction.
    - Spiral moves use the original step schedule: a turning move uses 1 cell, ordinary straight motion uses 2 cells, and the move immediately after a turn also uses 1 cell.
    - All new teal cells must stay orthogonally separated from every existing non-background cell except the immediate predecessor, and growth is resolved round-robin in the fixed order up, right, down, left until every arm is blocked.

    Key insight:
    - Arm-length parity does not label a pixel; it selects the whole growth dynamics of that arm, so mixed spiral and straight walkers compete for the same free space and later walkers inherit the barriers created earlier in the round-robin process.
    """
    out, _summary = _simulate(grid)
    return out


GRID_MIN = 21
GRID_MAX = 30
BORDER_GAP = 6
MAX_RETRIES = 500


def _build_cross(
    shape: tuple[int, int],
    center: tuple[int, int],
    lengths: list[int],
) -> np.ndarray:
    h, w = map(int, shape)
    cr, cc = map(int, center)
    g = np.zeros((h, w), dtype=int)
    g[cr, cc] = CROSS
    for arm_idx, (dr, dc) in enumerate(DIRS_URDL):
        for step in range(1, int(lengths[arm_idx]) + 1):
            rr = int(cr + dr * step)
            cc2 = int(cc + dc * step)
            g[rr, cc2] = CROSS
    return g


def _sample_lengths() -> list[int]:
    while True:
        lengths = random.sample(range(1, 8), 4)
        if any(v % 2 == 0 for v in lengths) and any(v % 2 == 1 for v in lengths):
            random.shuffle(lengths)
            return [int(v) for v in lengths]


def _choose_shape_and_center(lengths: list[int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    up, right, down, left = map(int, lengths)
    min_h = int(up + down + 1 + 2 * BORDER_GAP)
    min_w = int(left + right + 1 + 2 * BORDER_GAP)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    tight = random.random() < 0.65
    if tight:
        h_hi = min(GRID_MAX, min_h + 3)
        w_hi = min(GRID_MAX, min_w + 3)
    else:
        h_hi = GRID_MAX
        w_hi = GRID_MAX

    h_lo = max(GRID_MIN, min_h)
    w_lo = max(GRID_MIN, min_w)
    h_hi = max(h_lo, h_hi)
    w_hi = max(w_lo, w_hi)

    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)

    row_lo = int(up + BORDER_GAP)
    row_hi = int(h - 1 - BORDER_GAP - down)
    col_lo = int(left + BORDER_GAP)
    col_hi = int(w - 1 - BORDER_GAP - right)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    cr = random.randint(row_lo, row_hi)
    cc = random.randint(col_lo, col_hi)
    return (h, w), (cr, cc)


def _generator_checks(grid: np.ndarray) -> bool:
    out, summary = _simulate(grid)
    actual_agents = summary["arms"]

    added_total = int(np.sum(out == CROSS) - np.sum(grid == CROSS))
    if added_total < 10:
        return False

    fill = float(np.mean(out == CROSS))
    if not (0.10 <= fill <= 0.55):
        return False

    odd_agents = [agent for agent in actual_agents if agent["kind"] == "spiral"]
    even_agents = [agent for agent in actual_agents if agent["kind"] == "line"]
    if not odd_agents or not even_agents:
        return False
    if not any(agent["added"] >= 4 for agent in odd_agents):
        return False
    if not any(agent["turns"] >= 1 for agent in odd_agents):
        return False
    if not any(agent["added"] >= 3 for agent in even_agents):
        return False
    if sum(agent["added"] > 0 for agent in actual_agents) < 3:
        return False

    all_spiral, _ = _simulate(grid, behavior_override={idx: "spiral" for idx in range(4)})
    all_line, _ = _simulate(grid, behavior_override={idx: "line" for idx in range(4)})
    if np.array_equal(out, all_spiral) or np.array_equal(out, all_line):
        return False

    union = _independent_union(grid)
    if np.array_equal(out, union):
        return False

    parity_flipped = {}
    for agent in actual_agents:
        parity_flipped[int(agent["index"])] = "line" if agent["kind"] == "spiral" else "spiral"
    flipped_out, _ = _simulate(grid, behavior_override=parity_flipped)
    if np.array_equal(out, flipped_out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        lengths = _sample_lengths()
        placement = _choose_shape_and_center(lengths)
        if placement is None:
            continue
        shape, center = placement
        grid = _build_cross(shape, center, lengths)
        if _generator_checks(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
