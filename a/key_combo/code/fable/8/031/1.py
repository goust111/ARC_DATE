from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Generator bounds.
MIN_RUN_LEN = 2
MAX_RUN_LEN = 6
TOP_ONLY_RANGE = (1, 2)
LEFT_ONLY_RANGE = (1, 2)
SLACK_RANGE = (2, 8)
GENERATOR_RETRIES = 400


RunMap = Dict[int, Tuple[int, int]]


def _extract_edge_runs(line: np.ndarray) -> RunMap:
    """Return {color: (start, end)} for nonzero contiguous runs in one edge line."""
    arr = np.asarray(line, dtype=int)
    runs: RunMap = {}
    prev_end = -2
    i = 1
    while i < arr.size:
        color = int(arr[i])
        if color == BG:
            i += 1
            continue
        start = i
        while i + 1 < arr.size and int(arr[i + 1]) == color:
            i += 1
        end = i
        assert start > prev_end + 1, "edge runs must be separated by at least one background cell"
        assert color not in runs, "each color may define at most one run per edge"
        runs[color] = (int(start), int(end))
        prev_end = end
        i += 1
    return runs


def _compose_rectangles(shape: Tuple[int, int], rectangles: Iterable[Tuple[int, int, int, int, int]]) -> np.ndarray:
    """
    Paint same-color rectangles and erase cells covered by two rectangles.

    Each rectangle is `(color, r0, r1, c0, c1)` with inclusive bounds.
    """
    masks: List[Tuple[int, np.ndarray]] = []
    coverage = np.zeros(shape, dtype=int)
    for color, r0, r1, c0, c1 in rectangles:
        mask = np.zeros(shape, dtype=bool)
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
        masks.append((int(color), mask))
        coverage += mask.astype(int)

    out = np.zeros(shape, dtype=int)
    single = coverage == 1
    for color, mask in masks:
        out[single & mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild rectangles from same-color top and left projection runs, erasing double-covered cells to background.

    Input structure:
    - The grid has black background and an otherwise empty interior.
    - The top row contains separated colored horizontal runs, and the left column
      contains separated colored vertical runs; neither uses the corner cell.
    - Some colors appear once on both edges, while other colors appear on only
      one edge as inert distractor runs.

    Rule:
    - For every color that appears on both edges, pair its left-edge row span
      with its top-edge column span to define one filled rectangle.
    - Paint cells covered by exactly one such rectangle with that rectangle's
      color, and keep cells covered by two rectangles black.
    - Colors present on only one edge do not create any interior shape.
    - Preserve the original edge runs in the output.

    Key insight:
    - The border runs are rectangle projections, not rays: first reconstruct the
      full same-color rectangles from the two orthogonal spans, then combine them
      with an xor-like annihilation rule on overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert int(g[0, 0]) == BG, "the corner cell must stay background"
    assert np.all(g[1:, 1:] == BG), "the interior must be blank in the input"

    top_runs = _extract_edge_runs(g[0, :])
    left_runs = _extract_edge_runs(g[:, 0])

    rectangles = []
    for color in sorted(set(top_runs) & set(left_runs)):
        c0, c1 = top_runs[color]
        r0, r1 = left_runs[color]
        rectangles.append((int(color), int(r0), int(r1), int(c0), int(c1)))

    out = g.copy()
    fill = _compose_rectangles(g.shape, rectangles)
    out[1:, 1:] = fill[1:, 1:]
    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def _random_composition(total: int, parts: int) -> List[int]:
    """Split a non-negative integer into `parts` non-negative pieces."""
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out: List[int] = []
    for cut in cuts + [total + parts - 1]:
        out.append(int(cut - prev - 1))
        prev = cut
    assert sum(out) == total
    return out


def _sample_nonadjacent_intervals(edge_len: int, items: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
    """Place `(color, length)` runs on positions 1..edge_len with at least one blank gap."""
    items = items[:]
    random.shuffle(items)
    required = sum(length for _color, length in items) + max(0, len(items) - 1)
    if required > edge_len:
        raise ValueError("runs do not fit on edge")

    slack = edge_len - required
    extra = _random_composition(slack, len(items) + 1)

    intervals: Dict[int, Tuple[int, int]] = {}
    cursor = 1 + extra[0]
    for idx, (color, length) in enumerate(items):
        start = cursor
        end = start + length - 1
        intervals[int(color)] = (int(start), int(end))
        cursor = end + 1
        if idx < len(items) - 1:
            cursor += 1 + extra[idx + 1]
    return intervals


def _run_length(run: Tuple[int, int]) -> int:
    return int(run[1] - run[0] + 1)


def _sample_lengths() -> Tuple[List[int], List[int], List[int], List[int]]:
    paired_top = [random.randint(MIN_RUN_LEN, MAX_RUN_LEN) for _ in range(2)]
    paired_left = [random.randint(MIN_RUN_LEN, MAX_RUN_LEN) for _ in range(2)]

    top_only_count = random.randint(*TOP_ONLY_RANGE)
    left_only_count = random.randint(*LEFT_ONLY_RANGE)

    top_only = [random.choice(paired_top)]
    left_only = [random.choice(paired_left)]
    while len(top_only) < top_only_count:
        top_only.append(random.randint(MIN_RUN_LEN, MAX_RUN_LEN))
    while len(left_only) < left_only_count:
        left_only.append(random.randint(MIN_RUN_LEN, MAX_RUN_LEN))
    return paired_top, paired_left, top_only, left_only


def _required_edge_len(lengths: List[int]) -> int:
    return int(sum(lengths) + max(0, len(lengths) - 1))


def _pick_canvas_size(lengths: List[int]) -> int:
    base = _required_edge_len(lengths)
    slack = random.randint(*SLACK_RANGE)
    size = base + slack + 1  # +1 for the corner cell that cannot be used by runs
    return min(30, size)


def _render_input(
    shape: Tuple[int, int],
    top_runs: Dict[int, Tuple[int, int]],
    left_runs: Dict[int, Tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, (c0, c1) in top_runs.items():
        grid[0, int(c0) : int(c1) + 1] = int(color)
    for color, (r0, r1) in left_runs.items():
        grid[int(r0) : int(r1) + 1, 0] = int(color)
    return grid


def _validate_generated_grid(
    grid: np.ndarray,
    paired_colors: List[int],
    top_only_colors: List[int],
    left_only_colors: List[int],
) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30
    assert int(grid[0, 0]) == BG
    assert np.all(grid[1:, 1:] == BG)

    top_runs = _extract_edge_runs(grid[0, :])
    left_runs = _extract_edge_runs(grid[:, 0])
    assert set(top_runs) == set(paired_colors) | set(top_only_colors)
    assert set(left_runs) == set(paired_colors) | set(left_only_colors)
    assert set(top_only_colors).isdisjoint(left_runs)
    assert set(left_only_colors).isdisjoint(top_runs)
    assert len(set(paired_colors)) == 2
    assert len(top_only_colors) >= 1 and len(left_only_colors) >= 1

    paired_top_lengths = {_run_length(top_runs[color]) for color in paired_colors}
    paired_left_lengths = {_run_length(left_runs[color]) for color in paired_colors}
    assert any(_run_length(top_runs[color]) in paired_top_lengths for color in top_only_colors)
    assert any(_run_length(left_runs[color]) in paired_left_lengths for color in left_only_colors)

    out = main(grid)
    assert np.array_equal(out[0, :], grid[0, :])
    assert np.array_equal(out[:, 0], grid[:, 0])
    interior_colors = set(map(int, np.unique(out[1:, 1:]))) - {BG}
    assert interior_colors == set(paired_colors), "both paired colors must appear in the interior output"


def _attempt_generate() -> np.ndarray | None:
    paired_top, paired_left, top_only_lens, left_only_lens = _sample_lengths()
    top_lengths = paired_top + top_only_lens
    left_lengths = paired_left + left_only_lens

    width = _pick_canvas_size(top_lengths)
    height = _pick_canvas_size(left_lengths)
    if width < _required_edge_len(top_lengths) + 1 or height < _required_edge_len(left_lengths) + 1:
        return None

    top_only_count = len(top_only_lens)
    left_only_count = len(left_only_lens)
    total_colors = 2 + top_only_count + left_only_count
    if total_colors > len(PALETTE):
        return None

    colors = random.sample(PALETTE, total_colors)
    paired_colors = colors[:2]
    top_only_colors = colors[2 : 2 + top_only_count]
    left_only_colors = colors[2 + top_only_count :]

    top_items = list(zip(paired_colors, paired_top)) + list(zip(top_only_colors, top_only_lens))
    left_items = list(zip(paired_colors, paired_left)) + list(zip(left_only_colors, left_only_lens))

    top_runs = _sample_nonadjacent_intervals(width - 1, top_items)
    left_runs = _sample_nonadjacent_intervals(height - 1, left_items)
    grid = _render_input((height, width), top_runs, left_runs)

    _validate_generated_grid(grid, paired_colors, top_only_colors, left_only_colors)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-projection sample")
