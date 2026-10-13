from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR_TO_INDEX = {step: idx for idx, step in enumerate(DIRS)}
BG = 0
GEN_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    """Reference arc_utils semantics: return the most common grid value."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    h, w = mask.shape
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            out.append((nr, nc))
    return out


def _traverse_path(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev: tuple[int, int] | None = None
    curr = start
    total = int(np.count_nonzero(mask))
    while True:
        nxts = [coord for coord in _neighbors4(mask, curr[0], curr[1]) if coord != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, curr = curr, nxts[0]
        order.append(curr)
    assert len(order) == total, "foreground must be a single simple path"
    return order


def _direction_runs(order: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    dirs: list[int] = []
    for (r0, c0), (r1, c1) in zip(order, order[1:]):
        step = (r1 - r0, c1 - c0)
        assert step in DIR_TO_INDEX, "path must move orthogonally one cell at a time"
        dirs.append(DIR_TO_INDEX[step])
    assert dirs, "need at least two path cells"

    run_dirs = [dirs[0]]
    run_lens = [1]
    for direction in dirs[1:]:
        if direction == run_dirs[-1]:
            run_lens[-1] += 1
        else:
            run_dirs.append(direction)
            run_lens.append(1)
    return run_dirs, run_lens


def _analyze_order(order: list[tuple[int, int]]) -> dict | None:
    if len(order) < 3:
        return None
    run_dirs, run_lens = _direction_runs(order)
    expected = list(range(2, 2 + len(run_lens)))
    if run_lens != expected:
        return None
    if len(run_dirs) < 2:
        return None

    turn_diffs = [int((b - a) % 4) for a, b in zip(run_dirs, run_dirs[1:])]
    if any(diff not in (1, 3) for diff in turn_diffs):
        return None
    if len(set(turn_diffs)) != 1:
        return None

    turn_step = 1 if turn_diffs[0] == 1 else -1
    segment_count = len(run_lens) + 1
    return {
        "order": order,
        "run_dirs": run_dirs,
        "run_lens": run_lens,
        "turn_step": turn_step,
        "segment_count": segment_count,
        "last_direction": run_dirs[-1],
        "end": order[-1],
    }


def _extract_path(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mask = g != bg
    assert np.any(mask), "expected a foreground path"

    degrees = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        deg = len(_neighbors4(mask, int(r), int(c)))
        assert deg in (1, 2), "foreground must be a one-cell-wide unbranched path"
        degrees[(int(r), int(c))] = deg
        if deg == 1:
            endpoints.append((int(r), int(c)))
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    candidates = []
    for endpoint in endpoints:
        order = _traverse_path(mask, endpoint)
        analyzed = _analyze_order(order)
        if analyzed is not None:
            candidates.append(analyzed)
    assert len(candidates) == 1, "path direction must be uniquely determined"
    solved = candidates[0]
    solved["bg"] = bg
    solved["colors"] = [int(g[r, c]) for r, c in solved["order"]]
    return solved


def _minimal_period(seq: list[int]) -> int:
    for period in range(1, len(seq)):
        if all(seq[idx] == seq[idx - period] for idx in range(period, len(seq))):
            return period
    return len(seq)


def _extend_segment(
    end: tuple[int, int],
    direction: int,
    seg_len: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    r, c = end
    cells: list[tuple[int, int]] = []
    dr, dc = DIRS[direction]
    for _ in range(seg_len):
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return None
        cells.append((r, c))
    return cells


def _paint_path(shape: tuple[int, int], coords: Iterable[tuple[int, int]], cycle: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for idx, (r, c) in enumerate(coords):
        grid[r, c] = int(cycle[idx % len(cycle)])
    return grid


def _segment_ranges(segment_count: int) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for seg_len in range(1, segment_count + 1):
        end = start + seg_len
        ranges.append((start, end))
        start = end
    return ranges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the visible spiral by whole increasing segments while advancing its per-cell color cycle.

    Input structure:
    - The grid has a dominant background and one connected one-cell-wide foreground path.
    - Starting from one endpoint, the path is a spiral polyline whose segment cell counts are
      1, 2, 3, ...; consecutive segments are perpendicular and every turn goes to the same side.
    - Reading colors along that ordered path reveals a short repeating cycle that continues
      across corners without resetting at segment boundaries.

    Rule:
    - Identify the unique endpoint that makes the segment lengths increase as 1, 2, 3, ...
      and infer whether the path always turns left or always turns right.
    - Read the foreground colors in that order, extract the shortest repeating color cycle,
      and keep the current cycle phase at the visible endpoint.
    - Append new segments one by one: each next segment is one cell longer, rotates 90 degrees
      to the same side, and colors advance one cell at a time through the same cycle.
    - Stop before the first segment that would leave the grid; never draw a partial segment.

    Key insight:
    - The geometry determines the only valid traversal direction, but the colors repeat by cell
      rather than by segment, so the extension is correct only when segment growth, turn side,
      and cycle phase stay synchronized.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    solved = _extract_path(g)

    colors = solved["colors"]
    period = _minimal_period(colors)
    assert period < len(colors), "input must expose a repeating color cycle"
    cycle = colors[:period]

    out = g.copy()
    end = solved["end"]
    direction = solved["last_direction"]
    turn_step = solved["turn_step"]
    next_seg_len = solved["segment_count"] + 1
    occupied = set(solved["order"])
    color_index = len(colors)

    while True:
        next_dir = (direction + turn_step) % 4
        segment = _extend_segment(end, next_dir, next_seg_len, g.shape)
        if segment is None:
            break
        assert not any(cell in occupied for cell in segment), "spiral extension must stay disjoint"
        for cell in segment:
            out[cell] = cycle[color_index % period]
            occupied.add(cell)
            color_index += 1
        end = segment[-1]
        direction = next_dir
        next_seg_len += 1
    return out


def _build_maximal_spiral(
    shape: tuple[int, int],
    start: tuple[int, int],
    initial_direction: int,
    turn_step: int,
) -> tuple[list[tuple[int, int]], int]:
    coords = [start]
    end = start
    direction = initial_direction
    seg_len = 2
    while True:
        segment = _extend_segment(end, direction, seg_len, shape)
        if segment is None:
            break
        coords.extend(segment)
        end = segment[-1]
        direction = (direction + turn_step) % 4
        seg_len += 1
    return coords, seg_len - 1


def _make_cycle(rng: np.random.Generator) -> list[int]:
    use_tail = bool(rng.integers(0, 2))
    color_count = 4 if use_tail else 3
    palette = list(map(int, rng.choice(np.arange(1, 10), size=color_count, replace=False)))
    cycle = [palette[0], palette[1], palette[1], palette[2], palette[2], palette[2]]
    if use_tail:
        cycle.append(palette[3])
    return cycle


def _validate_instance(inp: np.ndarray, out: np.ndarray, cycle: list[int], visible_segments: int, full_segments: int) -> bool:
    if np.array_equal(inp, out):
        return False
    try:
        analyzed = _extract_path(inp)
        if _minimal_period(analyzed["colors"]) != len(cycle):
            return False
        if analyzed["segment_count"] != visible_segments:
            return False
        if not np.array_equal(main(inp), out):
            return False
        solved_out = _extract_path(out)
        if solved_out["segment_count"] != full_segments:
            return False
        if _minimal_period(solved_out["colors"]) != len(cycle):
            return False
    except AssertionError:
        return False

    order = analyzed["order"]
    colors = analyzed["colors"]
    ranges = _segment_ranges(visible_segments)
    if visible_segments < 5 or full_segments <= visible_segments:
        return False
    if len(set(colors[slice(*ranges[0])])) != 1:
        return False
    if len(set(colors[slice(*ranges[1])])) != 1:
        return False
    if len(set(colors[slice(*ranges[2])])) != 1:
        return False
    if not any(len(set(colors[slice(*segment_range)])) > 1 for segment_range in ranges[3:]):
        return False
    if len(order) <= 2 * len(cycle):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(14, 31))
    width = int(rng.integers(14, 31))
    shape = (height, width)
    turn_step = 1 if bool(rng.integers(0, 2)) else -1
    initial_direction = int(rng.integers(0, 4))
    start = (
        int(rng.integers(max(1, height // 4), min(height - 1, (3 * height) // 4))),
        int(rng.integers(max(1, width // 4), min(width - 1, (3 * width) // 4))),
    )

    coords, full_segments = _build_maximal_spiral(shape, start, initial_direction, turn_step)
    if not (6 <= full_segments <= 11):
        return None

    cycle = _make_cycle(rng)
    min_visible = max(5, len(cycle) - 1)
    if full_segments <= min_visible:
        return None
    visible_segments = int(rng.integers(min_visible, full_segments))

    visible_cells = visible_segments * (visible_segments + 1) // 2
    full_cells = full_segments * (full_segments + 1) // 2
    inp = _paint_path(shape, coords[:visible_cells], cycle)
    out = _paint_path(shape, coords[:full_cells], cycle)
    if not _validate_instance(inp, out, cycle, visible_segments, full_segments):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-cycle input")
