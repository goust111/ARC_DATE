from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DIVIDER_COLOR = 5
PALETTE = tuple(color for color in range(1, 10) if color != DIVIDER_COLOR)

MIN_SEGMENTS = 2
MAX_SEGMENTS = 5
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200


def _legend_colors(grid: np.ndarray, bg: int) -> list[int]:
    cols = np.flatnonzero(grid[-1] != int(bg))
    assert cols.size > 0, "expected a bottom legend row"
    assert np.all(np.diff(cols) == 1), "legend cells must form one contiguous strip"
    colors = [int(grid[-1, c]) for c in cols]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return colors


def _divider_color(main_area: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in np.unique(main_area):
        color = int(color)
        if color == int(bg):
            continue
        has_full_row = bool(np.any(np.all(main_area == color, axis=1)))
        has_full_col = bool(np.any(np.all(main_area == color, axis=0)))
        if has_full_row or has_full_col:
            candidates.append(color)
    assert len(candidates) == 1, f"expected one divider color, got {candidates}"
    return candidates[0]


def _count_components(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each colored line segment by the legend color indexed by how many grey-separated regions it spans.

    Input structure:
    - Black background fills the unused cells.
    - The bottom row contains a contiguous left-to-right legend of distinct colors; its 1st, 2nd, 3rd, ... cells encode the output colors for crossing 1, 2, 3, ... regions.
    - Above the legend, one grey color forms full horizontal and/or vertical divider lines that partition the main area into rectangles.
    - Every other color in the main area belongs to exactly one horizontal or vertical segment, and grey divider pixels interrupt that segment at crossing points.

    Rule:
    - Keep the grid size, divider lines, and legend row unchanged.
    - For each non-grey segment color in the main area, count how many 4-connected pieces that color has; each piece is one partition-crossing fragment of the original straight segment.
    - Let that count be K, then recolor every pixel of that segment to the Kth legend color.

    Key insight:
    - The input segment colors are distractors; the real invariant is the number of region pieces created when grey partition lines cut the straight segment, and the bottom legend maps that piece count to the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a legend row"

    bg = BACKGROUND
    legend = _legend_colors(g, bg)
    main_area = g[:-1]
    divider = _divider_color(main_area, bg)

    out = g.copy()
    out_main = out[:-1]
    for color in np.unique(main_area):
        color = int(color)
        if color in (int(bg), int(divider)):
            continue
        mask = main_area == color
        pieces = _count_components(mask)
        assert 1 <= pieces <= len(legend), "segment piece count must map into the legend"
        out_main[mask] = legend[pieces - 1]
    return out


def _band_layout(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    lines: list[int] = []
    pos = 0
    for i, size in enumerate(sizes):
        start = pos
        end = pos + int(size) - 1
        bands.append((start, end))
        pos = end + 1
        if i + 1 < len(sizes):
            lines.append(pos)
            pos += 1
    return bands, lines


def _all_positions(bands: list[tuple[int, int]]) -> list[int]:
    return [coord for start, end in bands for coord in range(start, end + 1)]


def _sample_layout() -> tuple[list[tuple[int, int]], list[int], list[tuple[int, int]], list[int], int, int] | None:
    row_band_count = random.randint(1, 4)
    col_band_count = random.randint(1, 5)
    if row_band_count == 1 and col_band_count == 1:
        return None

    row_sizes = [random.randint(3, 5) for _ in range(row_band_count)]
    col_sizes = [random.randint(3, 5) for _ in range(col_band_count)]
    row_bands, row_lines = _band_layout(row_sizes)
    col_bands, col_lines = _band_layout(col_sizes)
    main_h = row_bands[-1][1] + 1
    main_w = col_bands[-1][1] + 1
    if main_h + 1 > 30 or main_w > 30:
        return None
    legend_len = max(len(row_bands), len(col_bands))
    if main_w < legend_len:
        return None
    return row_bands, row_lines, col_bands, col_lines, main_h, main_w


def _sample_endpoints(
    bands: list[tuple[int, int]],
    start_band: int,
    end_band: int,
    *,
    avoid_full_span: bool,
) -> tuple[int, int] | None:
    if start_band == end_band:
        lo, hi = bands[start_band]
        if hi - lo + 1 < 2:
            return None
        for _ in range(40):
            start = random.randint(lo, hi - 1)
            end = random.randint(start + 1, hi)
            if avoid_full_span and start == lo and end == hi:
                continue
            return start, end
        return None

    start = random.randint(bands[start_band][0], bands[start_band][1])
    end = random.randint(bands[end_band][0], bands[end_band][1])
    return start, end


def _horizontal_cells(
    row: int,
    col_bands: list[tuple[int, int]],
    start_band: int,
    end_band: int,
    start_col: int,
    end_col: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for band in range(start_band, end_band + 1):
        lo, hi = col_bands[band]
        c0 = start_col if band == start_band else lo
        c1 = end_col if band == end_band else hi
        cells.extend((row, c) for c in range(c0, c1 + 1))
    return cells


def _vertical_cells(
    col: int,
    row_bands: list[tuple[int, int]],
    start_band: int,
    end_band: int,
    start_row: int,
    end_row: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for band in range(start_band, end_band + 1):
        lo, hi = row_bands[band]
        r0 = start_row if band == start_band else lo
        r1 = end_row if band == end_band else hi
        cells.extend((r, col) for r in range(r0, r1 + 1))
    return cells


def _place_segment(
    grid: np.ndarray,
    occupied: np.ndarray,
    *,
    color: int,
    orientation: str,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    used_rows: set[int],
    used_cols: set[int],
    force_k: int | None,
) -> int | None:
    rows = _all_positions(row_bands)
    cols = _all_positions(col_bands)

    for _ in range(PLACEMENT_RETRIES):
        if orientation == "horizontal":
            row_choices = [r for r in rows if r not in used_rows]
            if not row_choices:
                return None
            row = random.choice(row_choices)
            max_k = len(col_bands)
            k = force_k if force_k is not None else random.randint(1, max_k)
            if not (1 <= k <= max_k):
                continue
            start_band = random.randint(0, len(col_bands) - k)
            end_band = start_band + k - 1
            endpoints = _sample_endpoints(
                col_bands,
                start_band,
                end_band,
                avoid_full_span=len(col_bands) == 1,
            )
            if endpoints is None:
                continue
            start_col, end_col = endpoints
            cells = _horizontal_cells(row, col_bands, start_band, end_band, start_col, end_col)
            used_rows_candidate = row
        else:
            col_choices = [c for c in cols if c not in used_cols]
            if not col_choices:
                return None
            col = random.choice(col_choices)
            max_k = len(row_bands)
            k = force_k if force_k is not None else random.randint(1, max_k)
            if not (1 <= k <= max_k):
                continue
            start_band = random.randint(0, len(row_bands) - k)
            end_band = start_band + k - 1
            endpoints = _sample_endpoints(
                row_bands,
                start_band,
                end_band,
                avoid_full_span=len(row_bands) == 1,
            )
            if endpoints is None:
                continue
            start_row, end_row = endpoints
            cells = _vertical_cells(col, row_bands, start_band, end_band, start_row, end_row)
            used_rows_candidate = None

        if len(cells) < 2:
            continue
        if any(occupied[r, c] for r, c in cells):
            continue

        for r, c in cells:
            grid[r, c] = int(color)
            occupied[r, c] = True

        if orientation == "horizontal":
            used_rows.add(int(used_rows_candidate))
        else:
            used_cols.add(int(col))
        return int(k)

    return None


def _orientation_plan(row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> list[str] | None:
    modes = ["mixed"]
    if len(col_bands) > 1:
        modes.append("horizontal_only")
    if len(row_bands) > 1:
        modes.append("vertical_only")
    mode = random.choice(modes)

    if mode == "horizontal_only":
        max_segments = min(MAX_SEGMENTS, len(_all_positions(row_bands)), len(PALETTE))
        if max_segments < MIN_SEGMENTS:
            return None
        return ["horizontal"] * random.randint(MIN_SEGMENTS, max_segments)

    if mode == "vertical_only":
        max_segments = min(MAX_SEGMENTS, len(_all_positions(col_bands)), len(PALETTE))
        if max_segments < MIN_SEGMENTS:
            return None
        return ["vertical"] * random.randint(MIN_SEGMENTS, max_segments)

    max_segments = min(
        MAX_SEGMENTS,
        len(_all_positions(row_bands)) + len(_all_positions(col_bands)),
        len(PALETTE),
    )
    if max_segments < MIN_SEGMENTS:
        return None
    num_segments = random.randint(MIN_SEGMENTS, max_segments)
    h_low = max(1, num_segments - len(_all_positions(col_bands)))
    h_high = min(len(_all_positions(row_bands)), num_segments - 1)
    if h_low > h_high:
        return None
    h_count = random.randint(h_low, h_high)
    plan = ["horizontal"] * h_count + ["vertical"] * (num_segments - h_count)
    random.shuffle(plan)
    return plan


def _paint_dividers(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> None:
    if row_lines:
        grid[row_lines, :] = DIVIDER_COLOR
    if col_lines:
        grid[:-1, col_lines] = DIVIDER_COLOR


def _paint_legend(grid: np.ndarray, legend_colors: list[int]) -> None:
    start = random.randint(0, grid.shape[1] - len(legend_colors))
    grid[-1, start : start + len(legend_colors)] = np.array(legend_colors, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    row_bands, row_lines, col_bands, col_lines, main_h, main_w = layout

    legend_len = max(len(row_bands), len(col_bands))
    legend_colors = random.sample(list(PALETTE), legend_len)

    grid = np.full((main_h + 1, main_w), BACKGROUND, dtype=int)
    _paint_dividers(grid, row_lines, col_lines)
    _paint_legend(grid, legend_colors)

    plan = _orientation_plan(row_bands, col_bands)
    if plan is None:
        return None

    segment_colors = random.sample(list(PALETTE), len(plan))
    occupied = np.zeros((main_h, main_w), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    eligible_cross = [
        i
        for i, orientation in enumerate(plan)
        if (len(col_bands) if orientation == "horizontal" else len(row_bands)) > 1
    ]
    if not eligible_cross:
        return None

    forced_cross = random.choice(eligible_cross)
    forced_single_choices = [i for i in range(len(plan)) if i != forced_cross]
    forced_single = random.choice(forced_single_choices) if forced_single_choices else None

    piece_counts: list[int] = []
    for i, (orientation, color) in enumerate(zip(plan, segment_colors)):
        max_k = len(col_bands) if orientation == "horizontal" else len(row_bands)
        force_k = None
        if i == forced_cross:
            force_k = random.randint(2, max_k)
        elif i == forced_single:
            force_k = 1

        piece_count = _place_segment(
            grid,
            occupied,
            color=int(color),
            orientation=orientation,
            row_bands=row_bands,
            col_bands=col_bands,
            used_rows=used_rows,
            used_cols=used_cols,
            force_k=force_k,
        )
        if piece_count is None:
            return None
        piece_counts.append(piece_count)

    if not any(count > 1 for count in piece_counts):
        return None
    if len(set(piece_counts)) == 1 and len(plan) > 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
