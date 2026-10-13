from __future__ import annotations

import random
from collections import Counter

import numpy as np


SEED_COLOR = 1
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_MAX_ATTEMPTS = 600
_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGNATURES: set[tuple[int, ...]] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _spiral_trace(seed: tuple[int, int], shape: tuple[int, int]) -> list[dict[str, int | bool]]:
    h, w = shape
    r, c = seed
    max_delta = max(r, h - 1 - r, c, w - 1 - c)
    max_leg_length = 2 * (max_delta + 1)

    trace: list[dict[str, int | bool]] = []
    direction_index = 0
    leg_length = 2
    step_index = 0
    seen_offgrid = False

    while leg_length <= max_leg_length:
        dr, dc = _DIRS[direction_index]
        for _ in range(leg_length):
            r += dr
            c += dc
            inside = _in_bounds(shape, r, c)
            if not inside:
                seen_offgrid = True
            trace.append(
                {
                    "step": step_index,
                    "r": r,
                    "c": c,
                    "inside": inside,
                    "after_offgrid": seen_offgrid,
                }
            )
            step_index += 1

        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_length += 2

    return trace


def _spiral_path(seed: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    return [
        (int(step["r"]), int(step["c"]))
        for step in _spiral_trace(seed, shape)
        if bool(step["inside"])
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect the colors of scatter pixels hit by an invisible square spiral and output them as a row.

    Input structure:
    - The dominant background color fills most of the grid.
    - Exactly one fixed-color seed pixel marks the spiral origin.
    - Every other foreground object is a single scatter pixel, while the spiral itself is absent.

    Rule:
    - Starting from the seed, trace an unbounded square spiral that goes up, right, down, left with leg lengths 2, 2, 4, 4, 6, 6, ...
    - Out-of-bounds steps still count; continue tracing until the spiral has fully wrapped around the whole grid.
    - Whenever the traced path first passes through an in-bounds scatter pixel, append that pixel's color to the output row.
      Scatter pixels never visited by the spiral are decoys and are ignored.

    Key insight:
    - The hidden spiral, not geometric proximity, is the only ordering cue: some nearby pixels sit forever in the gaps between spiral rings and therefore never enter the summary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    seed_coords = np.argwhere(g == SEED_COLOR)
    assert seed_coords.shape == (1, 2), "expected exactly one fixed-color seed pixel"
    seed = (int(seed_coords[0, 0]), int(seed_coords[0, 1]))

    bg = _get_bg(g)
    collected: list[int] = []
    visited_scatter_positions: set[tuple[int, int]] = set()

    for r, c in _spiral_path(seed, g.shape):
        color = int(g[r, c])
        if color in (bg, SEED_COLOR):
            continue
        pos = (r, c)
        if pos not in visited_scatter_positions:
            collected.append(color)
            visited_scatter_positions.add(pos)

    assert collected, "expected at least one spiral-hit scatter pixel"
    return np.array([collected], dtype=int)


def _seed_candidates(shape: tuple[int, int], want_edge: bool) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            min_edge = min(r, c, h - 1 - r, w - 1 - c)
            if want_edge:
                if min_edge > 2:
                    continue
            else:
                if min_edge < 3:
                    continue
            trace = _spiral_trace((r, c), shape)
            inbounds = [step for step in trace if bool(step["inside"])]
            if len(inbounds) < 12:
                continue
            first_off = next((i for i, step in enumerate(trace) if not bool(step["inside"])), None)
            first_reentry = None
            if first_off is not None:
                first_reentry = next(
                    (i for i in range(first_off + 1, len(trace)) if bool(trace[i]["inside"])),
                    None,
                )
            if want_edge and first_reentry is None:
                continue
            candidates.append((r, c))
    return candidates


def _first_reentry_step(trace: list[dict[str, int | bool]]) -> tuple[int | None, int | None]:
    first_off = next((i for i, step in enumerate(trace) if not bool(step["inside"])), None)
    if first_off is None:
        return None, None
    first_reentry = next(
        (i for i in range(first_off + 1, len(trace)) if bool(trace[i]["inside"])),
        None,
    )
    return first_off, first_reentry


def _pick_distinct_cells(
    candidates: list[tuple[int, int]],
    blocked: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    picks: list[tuple[int, int]] = []
    for cell in random.sample(candidates, k=len(candidates)):
        if any(_chebyshev(cell, other) < 2 for other in blocked):
            continue
        if any(_chebyshev(cell, other) < 2 for other in picks):
            continue
        picks.append(cell)
        if len(picks) == count:
            return picks
    return None


def _actual_order_ok(seed: tuple[int, int], ordered_hits: list[tuple[int, int]]) -> bool:
    actual = tuple(ordered_hits)
    heuristics = [
        tuple(sorted(actual, key=lambda p: (_manhattan(seed, p), p[0], p[1]))),
        tuple(sorted(actual, key=lambda p: (_chebyshev(seed, p), p[0], p[1]))),
        tuple(sorted(actual, key=lambda p: ((p[0] - seed[0]) ** 2 + (p[1] - seed[1]) ** 2, p[0], p[1]))),
        tuple(sorted(actual)),
        tuple(sorted(actual, key=lambda p: (p[1], p[0]))),
    ]
    return all(order != actual for order in heuristics)


def _decoy_counterexample_ok(
    seed: tuple[int, int],
    hits: list[tuple[int, int]],
    decoys: list[tuple[int, int]],
) -> bool:
    if not decoys:
        return False
    hit_distances = [_manhattan(seed, cell) for cell in hits]
    decoy_distances = [_manhattan(seed, cell) for cell in decoys]
    return min(decoy_distances) < max(hit_distances)


def _choose_hits_and_decoys(
    seed: tuple[int, int],
    shape: tuple[int, int],
    want_edge: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    trace = _spiral_trace(seed, shape)
    path_entries = [step for step in trace if bool(step["inside"])]
    path_cells = [(int(step["r"]), int(step["c"])) for step in path_entries]
    path_set = set(path_cells)

    first_off, first_reentry = _first_reentry_step(trace)
    before_off = []
    after_reentry = []
    for step in path_entries:
        pos = (int(step["r"]), int(step["c"]))
        if _chebyshev(pos, seed) < 2:
            continue
        if first_off is not None and int(step["step"]) < first_off:
            before_off.append(pos)
        if first_reentry is not None and int(step["step"]) >= first_reentry:
            after_reentry.append(pos)

    n_hits = random.randint(3, 5)
    blocked = [seed]
    chosen: list[tuple[int, int]] = []

    if want_edge:
        early = _pick_distinct_cells(before_off, blocked, 1)
        late = _pick_distinct_cells(after_reentry, blocked + (early or []), 1)
        if early is None or late is None:
            return None
        chosen.extend(early)
        chosen.extend(late)

    remaining_needed = n_hits - len(chosen)
    more = _pick_distinct_cells(path_cells, blocked + chosen, remaining_needed)
    if more is None:
        return None
    chosen.extend(more)

    path_index = {cell: idx for idx, cell in enumerate(path_cells)}
    chosen = sorted(dict.fromkeys(chosen), key=path_index.__getitem__)
    if len(chosen) != n_hits:
        return None
    if not _actual_order_ok(seed, chosen):
        return None

    non_path_cells = [
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
        if (r, c) not in path_set and _chebyshev((r, c), seed) >= 2
    ]
    preferred_decoys = [
        cell for cell in non_path_cells if _manhattan(seed, cell) <= max(_manhattan(seed, h) for h in chosen)
    ]
    n_decoys = random.randint(1, min(3, 8 - n_hits))
    decoys = _pick_distinct_cells(preferred_decoys, blocked + chosen, n_decoys)
    if decoys is None:
        decoys = _pick_distinct_cells(non_path_cells, blocked + chosen, n_decoys)
    if decoys is None:
        return None
    if not _decoy_counterexample_ok(seed, chosen, decoys):
        return None
    return chosen, decoys


def _build_grid(
    shape: tuple[int, int],
    bg: int,
    seed: tuple[int, int],
    hits: list[tuple[int, int]],
    decoys: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    grid[seed] = SEED_COLOR

    scatter_palette = [color for color in range(10) if color not in (bg, SEED_COLOR)]
    random.shuffle(scatter_palette)
    colors = scatter_palette[: len(hits) + len(decoys)]

    for cell, color in zip(hits + decoys, colors):
        grid[cell] = color
    return grid


def _singletons_only(grid: np.ndarray, bg: int) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, SEED_COLOR):
                continue
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) != (r, c) and int(grid[rr, cc]) not in (bg, SEED_COLOR):
                        return False
    return True


def _validate_generated(
    grid: np.ndarray,
    seed: tuple[int, int],
    hits: list[tuple[int, int]],
    decoys: list[tuple[int, int]],
    want_edge: bool,
) -> bool:
    bg = _get_bg(grid)
    if not _singletons_only(grid, bg):
        return False

    out = main(grid)
    expected = tuple(int(grid[cell]) for cell in hits)
    if out.shape != (1, len(hits)):
        return False
    if tuple(map(int, out[0])) != expected:
        return False

    trace = _spiral_trace(seed, grid.shape)
    if want_edge:
        first_off, first_reentry = _first_reentry_step(trace)
        if first_off is None or first_reentry is None:
            return False
        hit_step_by_cell = {
            (int(step["r"]), int(step["c"])): int(step["step"])
            for step in trace
            if bool(step["inside"])
        }
        if not any(hit_step_by_cell[cell] >= first_reentry for cell in hits):
            return False
        if not any(hit_step_by_cell[cell] < first_off for cell in hits):
            return False

    if not _actual_order_ok(seed, hits):
        return False
    if not _decoy_counterexample_ok(seed, hits, decoys):
        return False
    return True


def _output_signature(out: np.ndarray) -> tuple[int, ...]:
    return (int(out.shape[1]),) + tuple(map(int, out[0]))


def _gen_attempt(want_edge: bool) -> np.ndarray | None:
    h = random.randint(9, 17)
    w = random.randint(9, 17)
    shape = (h, w)

    seed_candidates = _seed_candidates(shape, want_edge)
    if not seed_candidates:
        return None
    seed = random.choice(seed_candidates)

    choice = _choose_hits_and_decoys(seed, shape, want_edge)
    if choice is None:
        return None
    hits, decoys = choice

    bg = random.choice([color for color in range(10) if color != SEED_COLOR])
    grid = _build_grid(shape, bg, seed, hits, decoys)
    if not _validate_generated(grid, seed, hits, decoys, want_edge):
        return None

    signature = _output_signature(main(grid))
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(_MAX_ATTEMPTS):
        want_edge = (_GEN_CALL_COUNT % 2 == 0)
        grid = _gen_attempt(want_edge=want_edge)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid spiral-summary input")
