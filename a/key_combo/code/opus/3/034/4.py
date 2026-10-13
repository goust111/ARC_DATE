from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import find_components


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (18, 24)
GRID_WIDTH_RANGE = (16, 24)
RECT_COUNT_RANGE = (3, 5)
RECT_HEIGHT_RANGE = (3, 6)
RECT_WIDTH_RANGE = (3, 9)

MIN_OVERLAP_PAIRS = 2
TOP_MARGIN = 1
INTER_GAP = 1
BOTTOM_CLEAR_ROWS = 2
GEN_RETRIES = 400


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch == color):
            raise AssertionError("expected every foreground component to be a solid rectangle")
        height, width = patch.shape
        if height < 3 or width < 3:
            raise AssertionError("expected rectangles of at least 3x3")
        rectangles.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "height": int(height),
                "width": int(width),
                "inner_area": int((height - 2) * (width - 2)),
            }
        )
    rectangles.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["color"]))
    return rectangles


def _winner(rectangles: list[dict], col: int) -> int:
    covering = [rect for rect in rectangles if rect["c0"] <= col <= rect["c1"]]
    if not covering:
        return BG
    best = max(
        covering,
        key=lambda rect: (
            rect["inner_area"],
            rect["height"] * rect["width"],
            rect["width"],
            rect["height"],
            -rect["r0"],
            -rect["c0"],
        ),
    )
    return int(best["color"])


def _summary_row(rectangles: list[dict], width: int) -> np.ndarray:
    return np.array([_winner(rectangles, col) for col in range(width)], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """Hollow each solid rectangle and compress the per-column winner into a bottom summary row.

    Input structure:
    - A black background contains several non-black monochromatic axis-aligned solid rectangles.
    - Rectangles do not physically overlap because their row spans are separated, but their column spans may overlap.
    - Rectangle colors are all distinct, and the output keeps the same canvas size.

    Rule:
    - Detect every solid rectangle and erase only its strict interior, keeping a one-pixel border frame.
    - Compute each rectangle's internal area as (width - 2) x (height - 2).
    - For each grid column, consider all rectangle frames whose horizontal span covers that column and choose the one with the largest internal area; paint the bottom-row cell with that rectangle's color, or black if no rectangle covers the column.
    - Leave all other cells as the hollowed frames.

    Key insight:
    - The rectangle bbox determines both the visible hollow frame and the priority weight, so the 2D arrangement can be summarized as a 1D ownership strip across columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _extract_rectangles(g)
    assert rectangles, "expected at least one rectangle"

    out = g.copy()
    for rect in rectangles:
        if rect["height"] > 2 and rect["width"] > 2:
            out[rect["r0"] + 1 : rect["r1"], rect["c0"] + 1 : rect["c1"]] = BG

    out[-1, :] = _summary_row(rectangles, g.shape[1])
    return out


def _sample_heights(height: int, count: int) -> list[int] | None:
    max_sum = height - count - BOTTOM_CLEAR_ROWS
    for _ in range(80):
        heights = [random.randint(*RECT_HEIGHT_RANGE) for _ in range(count)]
        if sum(heights) <= max_sum:
            return heights
    return None


def _sample_row_starts(height: int, heights: list[int]) -> list[int]:
    count = len(heights)
    used = sum(heights) + TOP_MARGIN + (count - 1) * INTER_GAP + 1
    slack = (height - 1) - used
    gaps = [0] * (count + 1)
    for _ in range(slack):
        gaps[random.randrange(count + 1)] += 1

    starts = []
    cursor = TOP_MARGIN + gaps[0]
    for idx, rect_h in enumerate(heights):
        starts.append(cursor)
        cursor += rect_h
        if idx < count - 1:
            cursor += INTER_GAP + gaps[idx + 1]
    assert cursor <= height - BOTTOM_CLEAR_ROWS
    return starts


def _sample_intervals(widths: list[int], canvas_width: int) -> list[tuple[int, int]] | None:
    starts = [0]
    ends = [widths[0] - 1]
    overlap_pairs = 0

    for idx in range(1, len(widths)):
        prev_start, prev_end = starts[-1], ends[-1]
        prev_width = widths[idx - 1]
        width = widths[idx]
        must_overlap = idx < 3 or random.random() < 0.6

        if must_overlap:
            max_shared = min(prev_width - 1, width - 1)
            if max_shared < 1:
                return None
            shared = random.randint(1, max_shared)
            start = prev_end - shared + 1
            overlap_pairs += 1
        else:
            gap = random.randint(1, 3)
            start = prev_end + gap + 1
        starts.append(start)
        ends.append(start + width - 1)

    if overlap_pairs < MIN_OVERLAP_PAIRS:
        return None

    span = ends[-1] - starts[0] + 1
    free = canvas_width - span
    if free < 1:
        return None

    shift = random.randint(0, free)
    intervals = [(start + shift, end + shift) for start, end in zip(starts, ends)]
    if span == canvas_width:
        return None
    return intervals


def _overlap_pairs(intervals: list[tuple[int, int]]) -> int:
    pairs = 0
    for idx, (a0, a1) in enumerate(intervals):
        for b0, b1 in intervals[idx + 1 :]:
            if max(a0, b0) <= min(a1, b1):
                pairs += 1
    return pairs


def _coverage_counts(intervals: list[tuple[int, int]], width: int) -> list[int]:
    return [sum(c0 <= col <= c1 for c0, c1 in intervals) for col in range(width)]


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    max_rects = min(RECT_COUNT_RANGE[1], (height - 2) // 4)
    if max_rects < RECT_COUNT_RANGE[0]:
        return None
    count = random.randint(RECT_COUNT_RANGE[0], max_rects)

    heights = _sample_heights(height, count)
    if heights is None:
        return None
    row_starts = _sample_row_starts(height, heights)

    max_width = min(RECT_WIDTH_RANGE[1], width - 2)
    if max_width < RECT_WIDTH_RANGE[0]:
        return None

    for _ in range(120):
        widths = [random.randint(RECT_WIDTH_RANGE[0], max_width) for _ in range(count)]
        intervals = _sample_intervals(widths, width)
        if intervals is None:
            continue

        packages = list(zip(widths, intervals))
        random.shuffle(packages)
        widths = [pkg[0] for pkg in packages]
        intervals = [pkg[1] for pkg in packages]

        inner_areas = [(h - 2) * (w - 2) for h, w in zip(heights, widths)]
        if len(set(inner_areas)) != count:
            continue

        if _overlap_pairs(intervals) < MIN_OVERLAP_PAIRS:
            continue

        coverage = _coverage_counts(intervals, width)
        if max(coverage) < 2:
            continue

        colors = random.sample(PALETTE, count)
        grid = np.zeros((height, width), dtype=int)
        rectangles = []
        for color, r0, rect_h, rect_w, (c0, c1), inner_area in zip(
            colors,
            row_starts,
            heights,
            widths,
            intervals,
            inner_areas,
        ):
            r1 = r0 + rect_h - 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            rectangles.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "height": rect_h,
                    "width": rect_w,
                    "inner_area": inner_area,
                }
            )

        summary = _summary_row(rectangles, width)
        if BG not in summary:
            continue
        if set(np.unique(summary)) - {BG} != set(colors):
            continue
        if np.count_nonzero(np.array(coverage) >= 2) < 2:
            continue

        solved = main(grid)
        if not np.array_equal(solved[-1], summary):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
