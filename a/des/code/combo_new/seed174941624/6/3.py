from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MIN_GAP = 4
MAX_GENERATE_ATTEMPTS = 400


@dataclass(frozen=True)
class Segment:
    color: int
    axis: str
    fixed: int
    start: int
    end: int


def _extract_segments(grid: np.ndarray) -> tuple[list[Segment], list[Segment]]:
    segments: list[Segment] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        coords = np.argwhere(grid == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if rows.size == 1 and cols.size >= 2:
            segments.append(
                Segment(
                    color=color,
                    axis="h",
                    fixed=int(rows[0]),
                    start=int(cols.min()),
                    end=int(cols.max()),
                )
            )
        elif cols.size == 1 and rows.size >= 2:
            segments.append(
                Segment(
                    color=color,
                    axis="v",
                    fixed=int(cols[0]),
                    start=int(rows.min()),
                    end=int(rows.max()),
                )
            )
        else:
            raise AssertionError("each segment color must occupy exactly one row or one column")

    horizontals = [seg for seg in segments if seg.axis == "h"]
    verticals = [seg for seg in segments if seg.axis == "v"]
    assert horizontals and verticals, "expected at least one horizontal and one vertical segment"
    return horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Highlight each line crossing with a 3x3 square colored by the visible top segment.

    Input structure:
    - The background is black.
    - Each non-black color belongs to exactly one single-pixel-wide horizontal or vertical line segment.
    - Horizontal and vertical segments may cross; at each crossing only the top segment's color is visible in the shared cell.
    - Aside from those hidden crossing cells, every segment stays straight and continuous along its own row or column.

    Rule:
    - Reconstruct every segment's orientation and full span from its visible same-color cells.
    - A logical crossing occurs wherever a horizontal span and a vertical span cover the same cell.
    - The color already visible at that crossing cell is the top segment color.
    - For every crossing, paint the centered 3x3 neighborhood with that visible top color, clipped to the grid boundaries.
    - Leave all non-crossing line pixels and background cells unchanged.

    Key insight:
    - The visible color at a crossing is not enough by itself; the segment direction inferred from surrounding continuity reveals which horizontal/vertical spans truly intersect, and that same visible color determines the 3x3 highlight.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    height, width = g.shape

    horizontals, verticals = _extract_segments(g)

    for h_seg in horizontals:
        row = h_seg.fixed
        for v_seg in verticals:
            col = v_seg.fixed
            if h_seg.start <= col <= h_seg.end and v_seg.start <= row <= v_seg.end:
                top_color = int(g[row, col])
                assert top_color in {h_seg.color, v_seg.color}, "crossing cell must show one of the two segments"
                r0 = max(0, row - 1)
                r1 = min(height, row + 2)
                c0 = max(0, col - 1)
                c1 = min(width, col + 2)
                out[r0:r1, c0:c1] = top_color

    return out


def _pick_spaced_positions(
    rng: random.Random,
    size: int,
    count: int,
    *,
    min_gap: int = MIN_GAP,
) -> list[int] | None:
    if count <= 0:
        return []
    candidates = list(range(1, size - 1))
    if len(candidates) < count:
        return None
    for _ in range(200):
        picked = sorted(rng.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _choose_span(rng: random.Random, limit: int, marks: list[int]) -> tuple[int, int]:
    left_mark = marks[0]
    right_mark = marks[-1]
    start = rng.randint(0, left_mark - 1)
    end = rng.randint(right_mark + 1, limit - 1)
    return start, end


def _render(
    shape: tuple[int, int],
    horizontals: list[Segment],
    verticals: list[Segment],
    top_by_crossing: dict[tuple[int, int], str],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for seg in horizontals:
        grid[seg.fixed, seg.start : seg.end + 1] = seg.color
    for seg in verticals:
        grid[seg.start : seg.end + 1, seg.fixed] = seg.color
    for (h_idx, v_idx), top_axis in top_by_crossing.items():
        h_seg = horizontals[h_idx]
        v_seg = verticals[v_idx]
        color = h_seg.color if top_axis == "h" else v_seg.color
        grid[h_seg.fixed, v_seg.fixed] = color
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    counts = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
    num_h, num_v = rng.choice(counts)
    height = rng.randint(max(12, 4 * num_h + 4), 26)
    width = rng.randint(max(12, 4 * num_v + 4), 26)

    rows = _pick_spaced_positions(rng, height, num_h)
    cols = _pick_spaced_positions(rng, width, num_v)
    if rows is None or cols is None:
        return None

    colors = rng.sample(PALETTE, num_h + num_v)
    h_colors = colors[:num_h]
    v_colors = colors[num_h:]

    horizontals: list[Segment] = []
    for row, color in zip(rows, h_colors):
        left_idx = rng.randrange(num_v)
        right_idx = rng.randrange(left_idx, num_v)
        start, end = _choose_span(rng, width, cols[left_idx : right_idx + 1])
        horizontals.append(Segment(color=color, axis="h", fixed=row, start=start, end=end))

    verticals: list[Segment] = []
    for col, color in zip(cols, v_colors):
        top_idx = rng.randrange(num_h)
        bottom_idx = rng.randrange(top_idx, num_h)
        start, end = _choose_span(rng, height, rows[top_idx : bottom_idx + 1])
        verticals.append(Segment(color=color, axis="v", fixed=col, start=start, end=end))

    crossings: list[tuple[int, int]] = []
    for h_idx, h_seg in enumerate(horizontals):
        for v_idx, v_seg in enumerate(verticals):
            if h_seg.start <= v_seg.fixed <= h_seg.end and v_seg.start <= h_seg.fixed <= v_seg.end:
                crossings.append((h_idx, v_idx))
    if not crossings:
        return None

    top_by_crossing: dict[tuple[int, int], str] = {}
    if len(crossings) >= 2 and rng.random() < 0.5:
        shuffled = crossings[:]
        rng.shuffle(shuffled)
        top_by_crossing[shuffled[0]] = "h"
        top_by_crossing[shuffled[1]] = "v"
    for crossing in crossings:
        top_by_crossing.setdefault(crossing, rng.choice(["h", "v"]))

    grid = _render((height, width), horizontals, verticals, top_by_crossing)

    for color in colors:
        if int(np.count_nonzero(grid == color)) < 2:
            return None

    try:
        parsed_h, parsed_v = _extract_segments(grid)
        if len(parsed_h) != num_h or len(parsed_v) != num_v:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-lines input")
