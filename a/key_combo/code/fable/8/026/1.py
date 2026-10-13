from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
YELLOW = 4
CYAN = 8
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _build_spiral(
    start: tuple[int, int],
    initial_dir: int,
    turn: int,
    segment_count: int,
) -> tuple[list[tuple[int, int]], list[dict[str, int]]]:
    path = [tuple(start)]
    segments: list[dict[str, int]] = []
    r, c = start
    direction = int(initial_dir)

    for seg_idx in range(int(segment_count)):
        start_idx = len(path)
        dr, dc = DIRS[direction]
        seg_len = seg_idx + 2
        for _ in range(seg_len):
            r += dr
            c += dc
            path.append((r, c))
        segments.append(
            {
                "direction": direction,
                "step_start": start_idx,
                "step_end": len(path) - 1,
            }
        )
        direction = (direction + turn) % 4
    return path, segments


def _trace_candidate(
    shape: tuple[int, int],
    start: tuple[int, int],
    yellow: tuple[int, int],
    clues: set[tuple[int, int]],
    initial_dir: int,
    turn: int,
) -> dict[str, object] | None:
    h, w = shape
    r, c = start
    bbox_r0 = bbox_r1 = r
    bbox_c0 = bbox_c1 = c
    direction = int(initial_dir)
    path = [tuple(start)]
    visited: dict[tuple[int, int], int] = {}
    if start in clues:
        visited[start] = 0
    yellow_step: int | None = 0 if start == yellow else None

    for seg_idx in range(1, 80):
        dr, dc = DIRS[direction]
        for _ in range(seg_idx + 1):
            r += dr
            c += dc
            coord = (r, c)
            path.append(coord)
            if yellow_step is None and not _in_bounds(shape, r, c):
                return None
            if not _in_bounds(shape, r, c) and yellow_step is not None:
                pass
            if coord in clues and coord not in visited:
                visited[coord] = len(path) - 1
            if coord == yellow and yellow_step is None:
                yellow_step = len(path) - 1
            bbox_r0 = min(bbox_r0, r)
            bbox_r1 = max(bbox_r1, r)
            bbox_c0 = min(bbox_c0, c)
            bbox_c1 = max(bbox_c1, c)

        if bbox_r0 < 0 and bbox_c0 < 0 and bbox_r1 >= h and bbox_c1 >= w:
            break
        direction = (direction + turn) % 4

    if yellow_step is None:
        return None
    if not clues.issubset(visited):
        return None

    return {
        "initial_dir": int(initial_dir),
        "turn": int(turn),
        "path": path,
        "yellow_step": int(yellow_step),
        "visited": visited,
    }


def _recover_solution(grid: np.ndarray) -> dict[str, object]:
    start_coords = np.argwhere(grid == CYAN)
    yellow_coords = np.argwhere(grid == YELLOW)
    assert start_coords.shape == (1, 2), "expected exactly one cyan start marker"
    assert yellow_coords.shape == (1, 2), "expected exactly one yellow stop marker"

    start = tuple(map(int, start_coords[0]))
    yellow = tuple(map(int, yellow_coords[0]))
    clues = {tuple(map(int, rc)) for rc in np.argwhere((grid == RED) | (grid == YELLOW))}
    candidates = []
    for initial_dir in range(4):
        for turn in (-1, 1):
            candidate = _trace_candidate(grid.shape, start, yellow, clues, initial_dir, turn)
            if candidate is not None:
                candidates.append(candidate)
    assert len(candidates) == 1, "spiral parameters must be uniquely determined"
    return candidates[0]


def _render_output(shape: tuple[int, int], start: tuple[int, int], yellow: tuple[int, int], path: list[tuple[int, int]], yellow_step: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in path[1:yellow_step]:
        assert _in_bounds(shape, r, c)
        out[r, c] = RED
    out[start] = CYAN
    out[yellow] = YELLOW
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the unique growing square spiral implied by scattered relics and stop at the yellow marker.

    Input structure:
    - A black grid contains exactly one cyan start pixel, one yellow stop pixel,
      and several red relic pixels.
    - The cyan pixel is the start of an unseen square spiral whose segment lengths
      are 2, 3, 4, ... and whose direction turns by 90 degrees after each segment.
    - Every red relic and the yellow marker lie on that same spiral path.

    Rule:
    - Enumerate the four possible initial directions and the two possible turn
      orientations, and keep the unique spiral that passes through all red relics
      and the yellow stop marker.
    - Walk that spiral from the cyan start until the first time the yellow marker
      is reached, and paint every traversed intermediate cell red.
    - Preserve the cyan start and yellow stop colors; leave all other cells black.

    Key insight:
    - The red pixels are not endpoints to connect directly; they are sparse traces
      of a hidden arithmetic-length spiral, and the yellow marker truncates the
      otherwise continuing spiral.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    solution = _recover_solution(grid)
    start = tuple(map(int, np.argwhere(grid == CYAN)[0]))
    yellow = tuple(map(int, np.argwhere(grid == YELLOW)[0]))
    return _render_output(grid.shape, start, yellow, solution["path"], int(solution["yellow_step"]))


def _choice_from_segment(segments: list[dict[str, int]], seg_idx: int, stop_idx: int | None = None, after: bool = False) -> int | None:
    if not (0 <= seg_idx < len(segments)):
        return None
    start = int(segments[seg_idx]["step_start"])
    end = int(segments[seg_idx]["step_end"])
    values = list(range(start, end + 1))
    if stop_idx is not None:
        if after:
            values = [idx for idx in values if idx > stop_idx]
        else:
            values = [idx for idx in values if idx < stop_idx]
    if not values:
        return None
    options = [values[0], values[len(values) // 2], values[-1]]
    options = sorted(set(options))
    return random.choice(options)


def _make_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    initial_dir = random.randrange(4)
    turn = random.choice([-1, 1])
    stop_segment = random.randint(3, 9)
    extra_segments = random.randint(1, 4)
    path, segments = _build_spiral((0, 0), initial_dir, turn, stop_segment + 1 + extra_segments)
    stop_start = int(segments[stop_segment]["step_start"])
    stop_end = int(segments[stop_segment]["step_end"])
    stop_idx = random.randint(stop_start, stop_end)

    prefix = path[: stop_idx + 1]
    after_indices = list(range(stop_idx + 1, len(path)))
    if not after_indices:
        raise AssertionError("expected trailing spiral cells after the stop marker")

    mandatory = set()
    for seg_idx in (0, 1, max(2, stop_segment - 1)):
        choice = _choice_from_segment(segments, seg_idx, stop_idx=stop_idx, after=False)
        if choice is not None:
            mandatory.add(choice)
    first_after = _choice_from_segment(segments, stop_segment, stop_idx=stop_idx, after=True)
    if first_after is not None:
        mandatory.add(first_after)
    later_after_seg = min(len(segments) - 1, stop_segment + random.randint(1, extra_segments))
    choice = _choice_from_segment(segments, later_after_seg, stop_idx=stop_idx, after=True)
    if choice is not None:
        mandatory.add(choice)

    pool = [idx for idx in range(1, len(path)) if idx != stop_idx]
    random.shuffle(pool)
    target_total = random.randint(4, min(8, len(pool)))
    clue_indices = set(mandatory)
    for idx in pool:
        if len(clue_indices) >= target_total:
            break
        clue_indices.add(idx)

    if not any(idx > stop_idx for idx in clue_indices):
        clue_indices.add(random.choice(after_indices))

    clue_coords = [path[idx] for idx in sorted(clue_indices)]
    all_coords = prefix + clue_coords
    r0, c0, r1, c1 = _bbox(all_coords)
    base_h = r1 - r0 + 1
    base_w = c1 - c0 + 1
    if base_h > MAX_SIDE or base_w > MAX_SIDE:
        raise AssertionError("base case exceeds ARC side limit")

    margin_top = random.randint(0, MAX_SIDE - base_h)
    margin_left = random.randint(0, MAX_SIDE - base_w)
    margin_bottom = random.randint(0, MAX_SIDE - base_h - margin_top)
    margin_right = random.randint(0, MAX_SIDE - base_w - margin_left)
    h = base_h + margin_top + margin_bottom
    w = base_w + margin_left + margin_right

    def shift(coord: tuple[int, int]) -> tuple[int, int]:
        return coord[0] - r0 + margin_top, coord[1] - c0 + margin_left

    grid = np.full((h, w), BG, dtype=int)
    shifted_start = shift((0, 0))
    shifted_yellow = shift(path[stop_idx])
    grid[shifted_start] = CYAN
    grid[shifted_yellow] = YELLOW
    for idx in clue_indices:
        rr, cc = shift(path[idx])
        if (rr, cc) != shifted_yellow:
            grid[rr, cc] = RED

    shifted_prefix = [shift(coord) for coord in prefix]
    expected = _render_output(grid.shape, shifted_start, shifted_yellow, shifted_prefix, stop_idx)

    solution = _recover_solution(grid)
    out = main(grid)
    assert np.array_equal(out, expected)
    assert int(solution["initial_dir"]) == initial_dir
    assert int(solution["turn"]) == turn
    assert int(solution["yellow_step"]) == stop_idx
    assert any(step > stop_idx for step in solution["visited"].values() if step != stop_idx)

    return grid, expected, {
        "initial_dir": initial_dir,
        "turn": turn,
        "yellow_step": stop_idx,
        "height": h,
        "width": w,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, _expected, _meta = _make_case()
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid spiral reconstruction input")
