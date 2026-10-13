from __future__ import annotations

import random

import numpy as np


MAROON = 9
YELLOW = 4
ALL_COLORS = tuple(range(10))

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (20, 30)
BAND_COUNT_OPTIONS = (2, 3, 4)
BAND_WIDTH_RANGE = (2, 5)
HEIGHT_RANGE = (1, 6)
GEN_ATTEMPTS = 400
ROW_ATTEMPTS = 200


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if bool(value) and start is None:
            start = i
        if start is not None and (not bool(value) or i == len(mask) - 1):
            end = i if bool(value) else i - 1
            runs.append((start, end))
            start = None
    return runs


def _extract_bands(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    bands: list[dict[str, int]] = []
    for r in range(h):
        height = 0
        while height < w and int(grid[r, height]) == MAROON:
            height += 1
        if height == 0:
            continue

        seed_mask = np.array(
            [int(v) not in (bg, MAROON, YELLOW) for v in grid[r]],
            dtype=bool,
        )
        seed_runs = _runs(seed_mask)
        assert len(seed_runs) == 1, "each maroon ruler row must contain exactly one seed band"
        c0, c1 = seed_runs[0]
        color = int(grid[r, c0])
        assert np.all(grid[r, c0 : c1 + 1] == color), "seed band must be monochrome"
        bands.append(
            {
                "row": int(r),
                "col0": int(c0),
                "col1": int(c1),
                "height": int(height),
                "color": color,
            }
        )
    assert len(bands) >= 2, "expected at least two seed bands"
    return bands


def _grow_bands(grid: np.ndarray, bands: list[dict[str, int]], bg: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.array(grid, copy=True)
    active = np.zeros(grid.shape, dtype=bool)

    for band in bands:
        r = band["row"]
        c0 = band["col0"]
        c1 = band["col1"]
        color = band["color"]
        height = band["height"]

        out[r, c0 : c1 + 1] = color
        active[r, c0 : c1 + 1] = True
        for step in range(1, height + 1):
            rr = r - step
            if rr < 0:
                break
            if step % 2 == 1:
                out[rr, c0 : c1 + 1] = color
                active[rr, c0 : c1 + 1] = True
            else:
                out[rr, c0 : c1 + 1] = bg

    return out, active


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed band upward by the maroon count on the same row,
    alternating seed color and background, then bridge adjacent pillars through
    yellow anchor columns whenever both neighboring stripes are active.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    bands = _extract_bands(g, bg)

    out, active = _grow_bands(g, bands, bg)
    anchor_coords = [(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)]

    fills: list[tuple[int, int, int]] = []
    for c in sorted({c for _r, c in anchor_coords}):
        if c == 0 or c == g.shape[1] - 1:
            continue
        for r in range(g.shape[0]):
            if active[r, c - 1] and active[r, c + 1]:
                fills.append((int(r), int(c), int(out[r, c - 1])))

    for r, c, color in fills:
        out[r, c] = color

    for r, c in anchor_coords:
        out[r, c] = bg

    return out


def _active_rows(row: int, height: int) -> set[int]:
    rows = {int(row)}
    for step in range(1, height + 1, 2):
        rows.add(int(row - step))
    return rows


def _pair_overlaps(row_a: int, height_a: int, row_b: int, height_b: int) -> bool:
    return bool(_active_rows(row_a, height_a) & _active_rows(row_b, height_b))


def _sample_rows(h: int, heights: list[int], target: list[bool]) -> list[int] | None:
    low = max(max(heights) + 1, h // 2)
    high = h - 2
    if high - low + 1 < len(heights):
        return None
    candidates = list(range(low, high + 1))

    for _ in range(ROW_ATTEMPTS):
        rows = random.sample(candidates, len(heights))
        actual = [
            _pair_overlaps(rows[i], heights[i], rows[i + 1], heights[i + 1])
            for i in range(len(rows) - 1)
        ]
        if actual == target:
            return rows
    return None


def _seed_layout(w: int, widths: list[int], max_height: int) -> list[tuple[int, int]] | None:
    total_span = sum(widths) + (len(widths) - 1)
    left_pad = max_height + random.randint(2, 4)
    right_pad = random.randint(1, 3)
    start_hi = w - right_pad - total_span
    if start_hi < left_pad:
        return None

    start = random.randint(left_pad, start_hi)
    spans: list[tuple[int, int]] = []
    c = start
    for width in widths:
        spans.append((int(c), int(c + width - 1)))
        c += width + 1
    return spans


def _build_scene(
    h: int,
    w: int,
    bg: int,
    heights: list[int],
    rows: list[int],
    colors: list[int],
    spans: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)

    for row, height, color, (c0, c1) in zip(rows, heights, colors, spans):
        grid[row, :height] = MAROON
        grid[row, c0 : c1 + 1] = color

    anchor_row = h - 1
    for (_, left_c1), (right_c0, _) in zip(spans, spans[1:]):
        assert right_c0 - left_c1 == 2, "adjacent seed bands must leave exactly one anchor column"
        grid[anchor_row, left_c1 + 1] = YELLOW

    return grid


def _valid_scene(grid: np.ndarray, target_overlap: list[bool]) -> bool:
    bg = _bg_color(grid)
    bands = sorted(_extract_bands(grid, bg), key=lambda band: band["col0"])
    anchor_cols = sorted(int(c) for _r, c in np.argwhere(grid == YELLOW))
    if len(anchor_cols) != len(bands) - 1:
        return False

    actual_overlap = [
        _pair_overlaps(
            bands[i]["row"],
            bands[i]["height"],
            bands[i + 1]["row"],
            bands[i + 1]["height"],
        )
        for i in range(len(bands) - 1)
    ]
    if actual_overlap != target_overlap:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    if max(band["height"] for band in bands) < 2:
        return False

    changed_anchor_cols = 0
    for c in anchor_cols:
        col_in = grid[:, c]
        col_out = out[:, c]
        if np.any((col_out != bg) & (col_in != YELLOW)):
            changed_anchor_cols += 1

    if any(target_overlap) and changed_anchor_cols == 0:
        return False
    if not any(target_overlap) and changed_anchor_cols != 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        band_count = random.choice(BAND_COUNT_OPTIONS)

        heights = [random.randint(*HEIGHT_RANGE) for _ in range(band_count)]
        if max(heights) < 2:
            continue

        target_overlap = [bool(random.getrandbits(1)) for _ in range(band_count - 1)]
        if band_count >= 3 and (all(target_overlap) or not any(target_overlap)):
            target_overlap[random.randrange(len(target_overlap))] = not target_overlap[0]

        rows = _sample_rows(h, heights, target_overlap)
        if rows is None:
            continue

        widths = [random.randint(*BAND_WIDTH_RANGE) for _ in range(band_count)]
        spans = _seed_layout(w, widths, max(heights))
        if spans is None:
            continue

        palette = [color for color in ALL_COLORS if color not in (MAROON, YELLOW)]
        bg = random.choice(palette)
        seed_choices = [color for color in palette if color != bg]
        if len(seed_choices) < band_count:
            continue
        colors = random.sample(seed_choices, band_count)

        grid = _build_scene(h, w, bg, heights, rows, colors, spans)
        if _valid_scene(grid, target_overlap):
            return grid

    raise RuntimeError("failed to generate a valid ARC scene")
