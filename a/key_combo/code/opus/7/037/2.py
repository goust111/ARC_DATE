import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

BAR_COLORS = (1, 5, 6, 8, 9)
LENGTHS = (3, 5, 7)
KIND_ROW = "row"
KIND_COL = "col"
KIND_NONE = "none"

MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bar_cells(center: tuple[int, int], length: int, orient: str) -> list[tuple[int, int]]:
    cr, cc = map(int, center)
    half = int(length) // 2
    if orient == "H":
        return [(cr, c) for c in range(cc - half, cc + half + 1)]
    return [(r, cc) for r in range(cr - half, cr + half + 1)]


def _rotated_orient(orient: str) -> str:
    return "V" if orient == "H" else "H"


def _ring_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, center)
    return {
        (r0 + dr, c0 + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    }


def _neighbors8(cell: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, cell)
    return {(r0 + dr, c0 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _classify_crossing(
    cells: list[tuple[int, int]],
    green_row: int,
    green_col: int,
) -> tuple[str, tuple[int, int] | None, int | None]:
    row_hits = [(r, c) for r, c in cells if r == green_row]
    col_hits = [(r, c) for r, c in cells if c == green_col]
    if row_hits and col_hits:
        return "both", (int(green_row), int(green_col)), RED
    if row_hits:
        assert len(row_hits) == 1
        return KIND_ROW, row_hits[0], YELLOW
    if col_hits:
        assert len(col_hits) == 1
        return KIND_COL, col_hits[0], ORANGE
    return KIND_NONE, None, None


def _extract_bar(center_color_grid: np.ndarray, color: int) -> tuple[tuple[int, int], int, str]:
    coords = np.argwhere(center_color_grid == color)
    assert len(coords) >= 3
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    length = max(height, width)
    assert (height == 1) != (width == 1)
    assert len(coords) == length
    assert length % 2 == 1
    center = ((int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2)
    orient = "H" if height == 1 else "V"
    return center, length, orient


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each colored odd bar, then mark new green-line crossings with a type-coded 3x3 ring.

    Input structure:
    - Black is the dominant background.
    - One full green row and one full green column cross away from the border.
    - Several other colors each form one odd-length 1-cell-thick horizontal or vertical bar that does not overlap the green guides.

    Rule:
    - Keep the green full row and full column fixed.
    - Rotate every non-green bar 90 degrees around its midpoint, preserving its color, center, and length.
    - Around each rotated crossing with only the green row, paint a yellow 3x3 ring; with only the green column, paint an orange ring; with the row-column crossing itself, paint a red ring.
    - Preserve every crossing center as the bar color, and let rotated bar pixels overwrite rings while rings overwrite the green guides.

    Key insight:
    - The decisive step is the in-place 90-degree bar rotation: only after swapping each bar's orientation can you test which guide line it newly intersects and choose the ring color from that crossing type.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    green_rows = np.flatnonzero(np.all(g == GREEN, axis=1))
    green_cols = np.flatnonzero(np.all(g == GREEN, axis=0))
    assert len(green_rows) == 1
    assert len(green_cols) == 1
    green_row = int(green_rows[0])
    green_col = int(green_cols[0])
    assert 0 < green_row < h - 1
    assert 0 < green_col < w - 1

    bg = _get_bg(g)
    bar_colors = [int(c) for c in np.unique(g) if c not in (bg, GREEN)]
    out = np.full_like(g, bg)
    out[green_row, :] = GREEN
    out[:, green_col] = GREEN

    rotated_bars: list[tuple[int, list[tuple[int, int]]]] = []
    ring_specs: list[tuple[tuple[int, int], int]] = []

    for color in bar_colors:
        center, length, orient = _extract_bar(g, color)
        rotated = _bar_cells(center, length, _rotated_orient(orient))
        rotated_bars.append((color, rotated))
        kind, cross_center, ring_color = _classify_crossing(rotated, green_row, green_col)
        if kind != KIND_NONE:
            assert cross_center is not None and ring_color is not None
            rr, cc = cross_center
            assert 0 < rr < h - 1
            assert 0 < cc < w - 1
            ring_specs.append((cross_center, ring_color))

    for center, ring_color in ring_specs:
        for r, c in _ring_cells(center):
            out[r, c] = int(ring_color)

    for color, cells in rotated_bars:
        for r, c in cells:
            out[r, c] = int(color)

    return out


def _candidate_specs(
    height: int,
    width: int,
    green_row: int,
    green_col: int,
    length: int,
    kind: str,
) -> list[tuple[tuple[int, int], str]]:
    half = length // 2
    centers: list[tuple[tuple[int, int], str]] = []
    for cr in range(half, height - half):
        for cc in range(half, width - half):
            if kind == KIND_ROW:
                final_orient = "V"
            elif kind == KIND_COL:
                final_orient = "H"
            else:
                final_orient = random.choice(("H", "V"))

            input_orient = _rotated_orient(final_orient)
            input_cells = _bar_cells((cr, cc), length, input_orient)
            if any(r == green_row or c == green_col for r, c in input_cells):
                continue

            output_cells = _bar_cells((cr, cc), length, final_orient)
            actual_kind, cross_center, _ring_color = _classify_crossing(output_cells, green_row, green_col)
            if kind != actual_kind:
                continue
            if cross_center is not None:
                rr, cc2 = cross_center
                if not (0 < rr < height - 1 and 0 < cc2 < width - 1):
                    continue
            centers.append(((cr, cc), final_orient))
    random.shuffle(centers)
    return centers


def _render_input(
    height: int,
    width: int,
    green_row: int,
    green_col: int,
    bars: list[dict],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[green_row, :] = GREEN
    grid[:, green_col] = GREEN
    for bar in bars:
        for r, c in bar["input_cells"]:
            grid[r, c] = int(bar["color"])
    return grid


def _expected_output(
    height: int,
    width: int,
    green_row: int,
    green_col: int,
    bars: list[dict],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    out[green_row, :] = GREEN
    out[:, green_col] = GREEN
    for bar in bars:
        if bar["cross_center"] is not None:
            for r, c in _ring_cells(bar["cross_center"]):
                out[r, c] = int(bar["ring_color"])
    for bar in bars:
        for r, c in bar["output_cells"]:
            out[r, c] = int(bar["color"])
    return out


def _choose_kinds(count: int) -> list[str]:
    kinds = [KIND_ROW, KIND_COL]
    extras = random.choices([KIND_ROW, KIND_COL, KIND_NONE], weights=[3, 3, 4], k=count - 2)
    kinds.extend(extras)
    if count >= 4 and KIND_NONE not in kinds and random.random() < 0.7:
        kinds[-1] = KIND_NONE
    random.shuffle(kinds)
    return kinds


def _place_bars(
    height: int,
    width: int,
    green_row: int,
    green_col: int,
    kinds: list[str],
    colors: list[int],
    lengths: list[int],
) -> list[dict] | None:
    bars: list[dict] = []
    input_forbidden: set[tuple[int, int]] = set()
    output_pixels: set[tuple[int, int]] = set()
    ring_pixels: set[tuple[int, int]] = set()

    items = list(zip(kinds, colors, lengths))
    items.sort(key=lambda item: (item[0] == KIND_NONE, -item[2]))

    for kind, color, length in items:
        candidates = _candidate_specs(height, width, green_row, green_col, length, kind)
        chosen = None
        for center, final_orient in candidates:
            input_cells = _bar_cells(center, length, _rotated_orient(final_orient))
            output_cells = _bar_cells(center, length, final_orient)
            actual_kind, cross_center, ring_color = _classify_crossing(output_cells, green_row, green_col)
            assert actual_kind == kind

            input_set = set(input_cells)
            output_set = set(output_cells)
            if input_set & input_forbidden:
                continue
            if output_set & output_pixels:
                continue
            if output_set & ring_pixels:
                continue

            new_ring = set()
            if cross_center is not None:
                new_ring = _ring_cells(cross_center)
                if new_ring & output_pixels:
                    continue
                if new_ring & ring_pixels:
                    continue

            chosen = {
                "color": int(color),
                "center": center,
                "length": int(length),
                "input_cells": input_cells,
                "output_cells": output_cells,
                "cross_center": cross_center,
                "ring_color": ring_color,
            }
            break

        if chosen is None:
            return None

        bars.append(chosen)
        output_pixels.update(chosen["output_cells"])
        if chosen["cross_center"] is not None:
            ring_pixels.update(_ring_cells(chosen["cross_center"]))
        for cell in chosen["input_cells"]:
            input_forbidden.update(_neighbors8(cell))

    return bars


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 19)
    width = random.randint(13, 19)
    green_row = random.randint(2, height - 3)
    green_col = random.randint(2, width - 3)

    count = random.randint(3, 5)
    colors = random.sample(BAR_COLORS, count)
    lengths = random.choices(LENGTHS, weights=(5, 4, 2), k=count)
    kinds = _choose_kinds(count)

    bars = _place_bars(height, width, green_row, green_col, kinds, colors, lengths)
    if bars is None:
        return None

    crossing_count = sum(bar["cross_center"] is not None for bar in bars)
    if crossing_count < 2:
        return None
    if not any(bar["ring_color"] == YELLOW for bar in bars):
        return None
    if not any(bar["ring_color"] == ORANGE for bar in bars):
        return None

    inp = _render_input(height, width, green_row, green_col, bars)
    expected = _expected_output(height, width, green_row, green_col, bars)
    if _get_bg(inp) != BLACK:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
