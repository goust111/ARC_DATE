"""ARC task: a marker-distorted, marker-colored clockwise spiral."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _inside(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _run_spiral(
    shape: tuple[int, int], markers: dict[tuple[int, int], int]
) -> tuple[np.ndarray, list[tuple[int, int]], list[dict], list[tuple[int, tuple[int, int]]]]:
    """Simulate once and retain a trace used by the generator's semantic checks."""
    out = np.zeros(shape, dtype=int)
    r = c = direction = 0
    color = GREY
    path = [(r, c)]
    carved = {(r, c)}
    out[r, c] = color
    turns: list[dict] = []
    opportunities: list[tuple[int, tuple[int, int]]] = []

    # Four consecutive turns without a move are enough to prove there is no exit.
    turns_without_move = 0
    for _ in range(shape[0] * shape[1] * 8):
        dr, dc = DIRECTIONS[direction]
        nxt = (r + dr, c + dc)
        ahead = (r + 2 * dr, c + 2 * dc)

        if not _inside(shape, nxt):
            turns.append({"at": (r, c), "reason": "boundary", "blocker": None})
            direction = (direction + 1) % 4
            turns_without_move += 1
        else:
            nxt_blocked = nxt in carved or nxt in markers
            if nxt_blocked:
                break

            ahead_kind = None
            if _inside(shape, ahead):
                if ahead in markers:
                    ahead_kind = "marker"
                elif ahead in carved:
                    ahead_kind = "track"
                else:
                    opportunities.append((len(path) - 1, ahead))

            if ahead_kind is not None:
                turns.append({"at": (r, c), "reason": ahead_kind, "blocker": ahead})
                if ahead_kind == "marker":
                    color = markers[ahead]
                    # The turning cell belongs to the newly colored segment.
                    out[r, c] = color
                direction = (direction + 1) % 4
                turns_without_move += 1
            else:
                r, c = nxt
                carved.add((r, c))
                path.append((r, c))
                out[r, c] = color
                turns_without_move = 0

        if turns_without_move >= 4:
            break
    else:
        raise RuntimeError("spiral simulation did not terminate")

    for cell, marker_color in markers.items():
        out[cell] = marker_color
    return out, path, turns, opportunities


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a clockwise spiral whose isolated markers bend and recolor its path.

    Input structure:
    - The grid is black except for several isolated, differently colored single-pixel
      markers; marker colors are non-black and are never grey.
    - Coordinates use the whole input canvas, with the start fixed at its top-left cell.

    Rule:
    - Start at the top-left, heading right, and carve in grey. Turn clockwise when
      the next cell is outside the canvas or the cell two steps ahead is existing
      track or a marker; stop before entering any track or marker cell.
    - A turn caused by a marker changes the path color, including the turning cell,
      to that marker's color until another marker-caused turn changes it again.
    - Preserve every marker pixel itself.

    Key insight:
    - A marker is simultaneously a ghost wall and a color signal: its early turn
      changes the later spiral geometry and starts a new color segment at that bend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(g[0, 0]) == BLACK, "the fixed start must be empty"
    values = set(map(int, np.unique(g)))
    assert values <= {BLACK, *MARKER_COLORS}, "input may contain only black and non-grey markers"

    marker_cells = np.argwhere(g != BLACK)
    markers = {(int(r), int(c)): int(g[r, c]) for r, c in marker_cells}
    assert len(set(markers.values())) == len(markers), "marker colors must be unique"
    out, _path, _turns, _opportunities = _run_spiral(g.shape, markers)
    return out


def _marker_turns(turns: list[dict]) -> list[dict]:
    return [event for event in turns if event["reason"] == "marker"]


def _isolated(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in others)


def _valid_scene(shape: tuple[int, int], markers: dict[tuple[int, int], int]) -> bool:
    """Check visible structure, unique causality, and counterfactual dependence."""
    if not (2 <= len(markers) <= 4):
        return False
    if len(set(markers.values())) != len(markers):
        return False
    if any(not _isolated(cell, set(markers) - {cell}) for cell in markers):
        return False

    out, path, turns, _ = _run_spiral(shape, markers)
    events = _marker_turns(turns)
    counts = Counter(event["blocker"] for event in events)
    # Every visible marker has exactly one role, so no unused/doubly-interpreted clue exists.
    if set(counts) != set(markers) or any(counts[cell] != 1 for cell in markers):
        return False
    if not any(event["reason"] == "boundary" for event in turns):
        return False
    if not any(event["reason"] == "track" for event in turns):
        return False

    # Each signal must yield a legible segment, not merely recolor a terminal pixel.
    for color in markers.values():
        if int(np.count_nonzero(out == color)) < 3:  # marker plus at least two path cells
            return False

    baseline, base_path, _, _ = _run_spiral(shape, {})
    if set(path) == set(base_path) or np.array_equal(out, baseline):
        return False

    # Counterexamples against ignoring any marker: each removal changes geometry,
    # and hence proves that all collision steps are necessary rather than decorative.
    for cell in markers:
        reduced = dict(markers)
        reduced.pop(cell)
        reduced_out, reduced_path, _, _ = _run_spiral(shape, reduced)
        if set(reduced_path) == set(path) or np.array_equal(reduced_out, out):
            return False
    return True


def _attempt_scene(shape: tuple[int, int], count: int) -> dict[tuple[int, int], int] | None:
    markers: dict[tuple[int, int], int] = {}
    colors = random.sample(MARKER_COLORS, count)

    for color in colors:
        _out, _path, turns, opportunities = _run_spiral(shape, markers)
        if markers:
            triggered = {event["blocker"] for event in _marker_turns(turns)}
            if triggered != set(markers):
                return None

        # Prefer the middle of the still-unfolding path: it leaves visible colored
        # path on both sides and avoids a pile-up near the final spiral center.
        random.shuffle(opportunities)
        opportunities.sort(key=lambda item: abs(item[0] - len(_path) * random.uniform(.38, .72)))
        candidates = [cell for _, cell in opportunities if cell != (0, 0)]
        for cell in candidates[:80]:
            if cell in markers or not _isolated(cell, set(markers)):
                continue
            trial = dict(markers)
            trial[cell] = color
            _, _, trial_turns, _ = _run_spiral(shape, trial)
            triggered = {event["blocker"] for event in _marker_turns(trial_turns)}
            if triggered == set(trial):
                markers = trial
                break
        else:
            return None
    return markers


def generate_input() -> np.ndarray:
    """Generate diverse non-degenerate scenes and reject semantic shortcuts."""
    for _ in range(500):
        shape = (random.randint(10, 21), random.randint(10, 22))
        count = random.choices((2, 3, 4), weights=(3, 5, 2), k=1)[0]
        markers = _attempt_scene(shape, count)
        if markers is None or not _valid_scene(shape, markers):
            continue
        grid = np.zeros(shape, dtype=int)
        for (r, c), color in markers.items():
            grid[r, c] = color
        # A final independent interface check catches dtype/shape and parsing drift.
        solved = main(grid)
        expected, _, _, _ = _run_spiral(shape, markers)
        if solved.shape == grid.shape and np.array_equal(solved, expected):
            return grid
    raise RuntimeError("failed to generate a non-degenerate spiral scene")
