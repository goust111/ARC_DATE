from __future__ import annotations

from collections import Counter
import random

import numpy as np


COLORS = (1, 2, 3, 4)
MARKER = 0

GRID_MIN = 12
GRID_MAX = 20
ROW_COUNT_CHOICES = (2, 3)
COL_COUNT_CHOICES = (2, 3)
PART_MIN = 4
PART_MAX = 8
GEN_ATTEMPTS = 400


def _partition_cuts(grid: np.ndarray, axis: int) -> list[int]:
    length = int(grid.shape[axis])
    cross = int(grid.shape[1 - axis])
    cuts: list[int] = []
    threshold = max(2, int(np.ceil(0.75 * cross)))

    for idx in range(length - 1):
        if axis == 0:
            score = int(np.count_nonzero(grid[idx, :] != grid[idx + 1, :]))
        else:
            score = int(np.count_nonzero(grid[:, idx] != grid[:, idx + 1]))
        if score >= threshold:
            cuts.append(idx + 1)
    return cuts


def _segments(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    bounds = [0, *cuts, int(length)]
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


def _dominant_color(region: np.ndarray) -> int:
    values = [int(v) for v in region.ravel() if int(v) != MARKER]
    counts = Counter(values).most_common()
    assert counts, "expected a non-empty non-black region"
    top_count = counts[0][1]
    tied = [color for color, count in counts if count == top_count]
    assert len(tied) == 1, "expected a unique majority color per region"
    return int(counts[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a divider through each black-marked rectangle using that rectangle's majority color.

    Input structure:
    - The full grid is tiled by axis-aligned non-square rectangular regions.
    - Each region is mostly one color from a small palette, with a few minority-color noise pixels.
    - Some regions contain a single black marker pixel inside the rectangle.

    Rule:
    - Recover the rectangular partition from strong color-change boundaries between adjacent rows and columns.
    - For every region containing a black marker, count all non-black pixels in that region and take the unique most frequent color as the line color.
    - If the region is wider than tall, recolor the full marker column inside that region; if taller than wide, recolor the full marker row inside that region.

    Key insight:
    - The marker only selects position; the rectangle geometry chooses line direction, and the region's statistical majority color chooses the line color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_segments = _segments(g.shape[0], _partition_cuts(g, axis=0))
    col_segments = _segments(g.shape[1], _partition_cuts(g, axis=1))
    assert row_segments and col_segments, "expected at least one region"

    out = g.copy()
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            region = g[r0:r1, c0:c1]
            height = int(r1 - r0)
            width = int(c1 - c0)
            assert height != width, "expected non-square regions"

            markers = np.argwhere(region == MARKER)
            assert markers.shape[0] <= 1, "expected at most one marker per region"
            if markers.shape[0] == 0:
                continue

            rr, cc = map(int, markers[0])
            line_color = _dominant_color(region)
            if width > height:
                out[r0:r1, c0 + cc] = line_color
            else:
                out[r0 + rr, c0:c1] = line_color
    return out


def _random_parts(count: int) -> list[int]:
    return [random.randint(PART_MIN, PART_MAX) for _ in range(int(count))]


def _choose_dimensions() -> tuple[list[int], list[int]]:
    for _ in range(200):
        row_count = random.choice(ROW_COUNT_CHOICES)
        col_count = random.choice(COL_COUNT_CHOICES)
        row_heights = _random_parts(row_count)
        col_widths = _random_parts(col_count)
        if not (GRID_MIN <= sum(row_heights) <= GRID_MAX):
            continue
        if not (GRID_MIN <= sum(col_widths) <= GRID_MAX):
            continue
        if any(h == w for h in row_heights for w in col_widths):
            continue
        if not (min(row_heights) < max(col_widths) and max(row_heights) > min(col_widths)):
            continue
        return row_heights, col_widths
    raise RuntimeError("failed to sample non-square mixed-orientation dimensions")


def _color_layout(n_rows: int, n_cols: int) -> list[list[int]]:
    layout = [[-1] * n_cols for _ in range(n_rows)]

    def backtrack(pos: int) -> bool:
        if pos == n_rows * n_cols:
            return True
        r, c = divmod(pos, n_cols)
        choices = list(COLORS)
        random.shuffle(choices)
        for color in choices:
            if r > 0 and layout[r - 1][c] == color:
                continue
            if c > 0 and layout[r][c - 1] == color:
                continue
            layout[r][c] = color
            if backtrack(pos + 1):
                return True
        layout[r][c] = -1
        return False

    ok = backtrack(0)
    assert ok, "failed to assign region colors"
    return layout


def _region_boxes(row_heights: list[int], col_widths: list[int]) -> list[dict]:
    row_starts = np.cumsum([0, *row_heights[:-1]]).tolist()
    col_starts = np.cumsum([0, *col_widths[:-1]]).tolist()
    layout = _color_layout(len(row_heights), len(col_widths))

    boxes: list[dict] = []
    for ir, (r0, h) in enumerate(zip(row_starts, row_heights)):
        for ic, (c0, w) in enumerate(zip(col_starts, col_widths)):
            boxes.append(
                {
                    "ir": ir,
                    "ic": ic,
                    "r0": int(r0),
                    "r1": int(r0 + h),
                    "c0": int(c0),
                    "c1": int(c0 + w),
                    "height": int(h),
                    "width": int(w),
                    "color": int(layout[ir][ic]),
                }
            )
    return boxes


def _interior_positions(box: dict) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(int(box["r0"]) + 1, int(box["r1"]) - 1)
        for c in range(int(box["c0"]) + 1, int(box["c1"]) - 1)
    ]


def _line_positions(box: dict, marker: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = marker
    if int(box["width"]) > int(box["height"]):
        return [(rr, c) for rr in range(int(box["r0"]), int(box["r1"]))]
    return [(r, cc) for cc in range(int(box["c0"]), int(box["c1"]))]


def _sample_marked_boxes(boxes: list[dict]) -> list[int]:
    wide = [i for i, box in enumerate(boxes) if int(box["width"]) > int(box["height"])]
    tall = [i for i, box in enumerate(boxes) if int(box["height"]) > int(box["width"])]
    assert wide and tall, "expected both orientations to exist"

    chosen = {random.choice(wide), random.choice(tall)}
    target = random.randint(2, min(5, len(boxes)))
    remaining = [i for i in range(len(boxes)) if i not in chosen]
    random.shuffle(remaining)
    for idx in remaining[: max(0, target - len(chosen))]:
        chosen.add(idx)
    return sorted(chosen)


def _paint_noise(
    grid: np.ndarray,
    box: dict,
    marker: tuple[int, int] | None,
) -> None:
    majority = int(box["color"])
    candidates = _interior_positions(box)
    assert candidates, "expected each region to have an interior"

    line_cells: list[tuple[int, int]] = []
    if marker is not None:
        line_cells = [
            pos
            for pos in _line_positions(box, marker)
            if box["r0"] < pos[0] < box["r1"] - 1 and box["c0"] < pos[1] < box["c1"] - 1
        ]
        line_cells = [pos for pos in line_cells if pos != marker]

    colors = [c for c in COLORS if c != majority]
    n_noise_colors = random.randint(1, min(3, len(colors)))
    noise_colors = random.sample(colors, n_noise_colors)
    max_noise = min(len(candidates) - (1 if marker is not None else 0), max(3, int(box["height"] + box["width"]) // 2))
    noise_total = random.randint(2 if marker is None else 3, max_noise)

    chosen_positions: list[tuple[int, int]] = []
    if marker is not None and line_cells:
        on_line = min(len(line_cells), random.randint(1, min(3, len(line_cells))))
        chosen_positions.extend(random.sample(line_cells, on_line))

    remaining = [pos for pos in candidates if pos != marker and pos not in chosen_positions]
    random.shuffle(remaining)
    chosen_positions.extend(remaining[: max(0, noise_total - len(chosen_positions))])

    for i, pos in enumerate(chosen_positions):
        grid[pos] = noise_colors[i % len(noise_colors)]


def _gen_attempt() -> np.ndarray | None:
    row_heights, col_widths = _choose_dimensions()
    H, W = sum(row_heights), sum(col_widths)
    grid = np.zeros((H, W), dtype=int)
    boxes = _region_boxes(row_heights, col_widths)
    marked_indices = set(_sample_marked_boxes(boxes))

    for idx, box in enumerate(boxes):
        r0, r1, c0, c1 = int(box["r0"]), int(box["r1"]), int(box["c0"]), int(box["c1"])
        grid[r0:r1, c0:c1] = int(box["color"])

        marker = None
        interior = _interior_positions(box)
        if idx in marked_indices:
            marker = random.choice(interior)
        _paint_noise(grid, box, marker)
        if marker is not None:
            grid[marker] = MARKER

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != grid)) < len(marked_indices) * 2:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 11")
