from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

AREA_TO_COLOR = {
    3: RED,
    4: GREEN,
    5: BLUE,
    6: YELLOW,
    7: ORANGE,
}

GLYPH_DISPLAY_COLORS = (PINK, MAROON)
GEN_ATTEMPTS = 400
MAX_DIM = 26


def _true_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(flags.tolist()):
        if value and start is None:
            start = int(idx)
        elif not value and start is not None:
            intervals.append((start, int(idx) - 1))
            start = None
    if start is not None:
        intervals.append((start, int(len(flags)) - 1))
    return intervals


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_separator_column(grid: np.ndarray) -> int:
    for c in range(int(grid.shape[1])):
        if np.all(grid[:, c] == GREY):
            return int(c)
    raise AssertionError("expected one full-height grey separator column")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_glyph_areas(right: np.ndarray) -> list[int]:
    comps = _components(right != BLACK)
    glyphs: list[tuple[int, int, int]] = []
    for coords in comps:
        vals = {int(right[r, c]) for r, c in coords}
        assert len(vals) == 1, "each glyph must be monochromatic"
        area = int(coords.shape[0])
        assert 3 <= area <= 7, "glyph area must be in 3..7"
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 <= 3 and c1 - c0 + 1 <= 3, "glyph must fit in 3x3"
        glyphs.append((int(r0), int(c0), area))
    glyphs.sort(key=lambda item: (item[0], item[1]))
    return [area for _r0, _c0, area in glyphs]


def _lattice_segments(left: np.ndarray) -> list[list[tuple[int, int]]]:
    block_mask = left == TEAL
    row_bands = _true_intervals(block_mask.any(axis=1))
    col_bands = _true_intervals(block_mask.any(axis=0))

    assert len(row_bands) >= 2 and len(col_bands) >= 2, "need a rectangular block lattice"
    assert all(e - s + 1 == 2 for s, e in row_bands), "every block row band must be 2 cells tall"
    assert all(e - s + 1 == 2 for s, e in col_bands), "every block col band must be 2 cells wide"
    assert all(next_s == prev_e + 2 for (_ps, prev_e), (next_s, _ne) in zip(row_bands, row_bands[1:])), (
        "row gutter width must be 1"
    )
    assert all(next_s == prev_e + 2 for (_ps, prev_e), (next_s, _ne) in zip(col_bands, col_bands[1:])), (
        "col gutter width must be 1"
    )

    row_band_mask = np.zeros(left.shape[0], dtype=bool)
    col_band_mask = np.zeros(left.shape[1], dtype=bool)
    for s, e in row_bands:
        row_band_mask[s : e + 1] = True
    for s, e in col_bands:
        col_band_mask[s : e + 1] = True
    assert np.array_equal(block_mask, row_band_mask[:, None] & col_band_mask[None, :]), (
        "teal cells must form a complete rectangular array of 2x2 blocks"
    )

    r0, r1 = int(row_bands[0][0]), int(row_bands[-1][1])
    c0, c1 = int(col_bands[0][0]), int(col_bands[-1][1])
    segments: list[list[tuple[int, int]]] = []

    for (_prev_s, prev_e), (_next_s, _next_e) in zip(row_bands, row_bands[1:]):
        gutter_r = int(prev_e + 1)
        coords = [(gutter_r, c) for c in range(c0, c1 + 1) if int(left[gutter_r, c]) == BLACK]
        assert coords, "horizontal gutter segment must be non-empty"
        segments.append(coords)

    for (_prev_s, prev_e), (_next_s, _next_e) in zip(col_bands, col_bands[1:]):
        gutter_c = int(prev_e + 1)
        coords = [(r, gutter_c) for r in range(r0, r1 + 1) if int(left[r, gutter_c]) == BLACK]
        assert coords, "vertical gutter segment must be non-empty"
        segments.append(coords)

    return segments


def _paint_segments(left: np.ndarray, colors: list[int], segments: list[list[tuple[int, int]]]) -> np.ndarray:
    out = np.array(left, dtype=int, copy=True)
    assert len(colors) == len(segments)
    for color, coords in zip(colors, segments):
        for r, c in coords:
            if int(out[r, c]) == BLACK:
                out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode right-side glyph areas into colors and paint the left lattice gutters in enumeration order.

    Input structure:
    - The background is black, and one full-height grey column splits the grid into a left lattice and right glyph panel.
    - Left of the divider, teal 2x2 blocks form a complete rectangular array with exactly 1-cell black gutters between adjacent block rows and columns.
    - Right of the divider, several monochromatic 4-connected glyphs appear in separate 3x3 boxes stacked from top to bottom, and each glyph area is between 3 and 7.

    Rule:
    - Remove the grey divider and everything to its right, keeping the left panel at its original height and width.
    - Read the right-side glyphs from top to bottom and map each glyph area using 3->red, 4->green, 5->blue, 6->yellow, 7->orange.
    - Enumerate left-side gutter segments by taking every horizontal gutter row first (top to bottom), then every vertical gutter column (left to right).
    - Paint the currently black cells of segment i with the color decoded from glyph i, leaving teal blocks unchanged.

    Key insight:
    - The glyph shape and glyph color are distractions; only the pixel count matters, and the final intersections depend on the specified horizontal-then-vertical painting order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    areas = _decode_glyph_areas(right)
    decoded_colors = [AREA_TO_COLOR[int(area)] for area in areas]
    segments = _lattice_segments(left)
    assert len(decoded_colors) == len(segments), "glyph count must match gutter segment count"
    return _paint_segments(left, decoded_colors, segments)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _random_connected_glyph(area: int, rng: np.random.Generator) -> np.ndarray:
    assert 3 <= area <= 7
    all_cells = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(200):
        chosen = {all_cells[_randint(rng, 0, len(all_cells) - 1)]}
        while len(chosen) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in chosen:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in chosen:
                        frontier.append((nr, nc))
            if not frontier:
                break
            chosen.add(frontier[_randint(rng, 0, len(frontier) - 1)])
        if len(chosen) != area:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to sample a connected 3x3 glyph")


def _sample_decoded_areas(
    h_count: int,
    v_count: int,
    rng: np.random.Generator,
) -> list[int] | None:
    total = h_count + v_count
    for _ in range(200):
        areas = [_randint(rng, 3, 7) for _ in range(total)]
        colors = [AREA_TO_COLOR[a] for a in areas]
        horiz = colors[:h_count]
        vert = colors[h_count:]

        if len(set(colors)) < 3:
            continue
        if not any(h != v for h in horiz for v in vert):
            continue

        reverse_sensitive = False
        if h_count >= 2 and horiz != list(reversed(horiz)):
            reverse_sensitive = True
        if v_count >= 2 and vert != list(reversed(vert)):
            reverse_sensitive = True
        if not reverse_sensitive:
            continue

        return areas
    return None


def _sample_glyph_display_colors(
    areas: list[int],
    rng: np.random.Generator,
) -> list[int] | None:
    for _ in range(100):
        colors = [GLYPH_DISPLAY_COLORS[_randint(rng, 0, len(GLYPH_DISPLAY_COLORS) - 1)] for _ in areas]
        if len(areas) >= 2 and len(set(colors)) < 2:
            continue
        if any(colors[i] == colors[j] and areas[i] != areas[j] for i in range(len(areas)) for j in range(i + 1, len(areas))):
            return colors
    return None


def _expected_from_layout(
    inp: np.ndarray,
    sep: int,
    row_count: int,
    col_count: int,
    top: int,
    left_margin: int,
    areas: list[int],
) -> np.ndarray:
    left = np.array(inp[:, :sep], dtype=int, copy=True)
    colors = [AREA_TO_COLOR[a] for a in areas]

    bbox_r0 = int(top)
    bbox_r1 = int(top + (3 * row_count - 1) - 1)
    bbox_c0 = int(left_margin)
    bbox_c1 = int(left_margin + (3 * col_count - 1) - 1)

    idx = 0
    for row_gap in range(row_count - 1):
        rr = int(top + 2 + 3 * row_gap)
        for c in range(bbox_c0, bbox_c1 + 1):
            if int(left[rr, c]) == BLACK:
                left[rr, c] = colors[idx]
        idx += 1

    for col_gap in range(col_count - 1):
        cc = int(left_margin + 2 + 3 * col_gap)
        for r in range(bbox_r0, bbox_r1 + 1):
            if int(left[r, cc]) == BLACK:
                left[r, cc] = colors[idx]
        idx += 1

    return left


def _alt_vertical_first(left: np.ndarray, h_count: int, colors: list[int]) -> np.ndarray:
    segments = _lattice_segments(left)
    reordered_segments = segments[h_count:] + segments[:h_count]
    return _paint_segments(left, colors, reordered_segments)


def _alt_reversed_axis(left: np.ndarray, h_count: int, v_count: int, colors: list[int]) -> np.ndarray | None:
    segments = _lattice_segments(left)
    horiz_segments = segments[:h_count]
    vert_segments = segments[h_count:]

    if h_count >= 2 and colors[:h_count] != list(reversed(colors[:h_count])):
        return _paint_segments(left, colors, list(reversed(horiz_segments)) + vert_segments)
    if v_count >= 2 and colors[h_count:] != list(reversed(colors[h_count:])):
        return _paint_segments(left, colors, horiz_segments + list(reversed(vert_segments)))
    return None


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    layouts = [
        (2, 3),
        (2, 4),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    row_count, col_count = layouts[_randint(rng, 0, len(layouts) - 1)]

    h_count = row_count - 1
    v_count = col_count - 1
    areas = _sample_decoded_areas(h_count, v_count, rng)
    if areas is None:
        return None
    glyph_colors = _sample_glyph_display_colors(areas, rng)
    if glyph_colors is None:
        return None

    max_gap = 2 if 3 * len(areas) + 2 * (len(areas) - 1) + 2 <= MAX_DIM else 1
    gap = _randint(rng, 1, max_gap)
    glyph_stack_h = 3 * len(areas) + gap * (len(areas) - 1)
    lattice_h = 3 * row_count - 1
    lattice_w = 3 * col_count - 1

    left_pad = _randint(rng, 0, 2)
    left_right_pad = _randint(rng, 0, 2)
    right_left_pad = _randint(rng, 0, 2)
    right_right_pad = _randint(rng, 0, 2)
    left_width = left_pad + lattice_w + left_right_pad
    right_width = right_left_pad + 3 + right_right_pad
    width = left_width + 1 + right_width
    if width > MAX_DIM:
        return None

    max_top = min(2, MAX_DIM - max(lattice_h, glyph_stack_h))
    if max_top < 0:
        return None
    left_top = _randint(rng, 0, max_top)
    right_top = _randint(rng, 0, max_top)
    base_height = max(left_top + lattice_h, right_top + glyph_stack_h)
    max_bottom = min(2, MAX_DIM - base_height)
    if max_bottom < 0:
        return None
    height = base_height + _randint(rng, 0, max_bottom)

    grid = np.zeros((height, width), dtype=int)
    sep = left_width
    grid[:, sep] = GREY

    for rr in range(row_count):
        for cc in range(col_count):
            r0 = int(left_top + 3 * rr)
            c0 = int(left_pad + 3 * cc)
            grid[r0 : r0 + 2, c0 : c0 + 2] = TEAL

    glyph_c0 = int(sep + 1 + right_left_pad)
    for idx, area in enumerate(areas):
        mask = _random_connected_glyph(area, rng)
        r0 = int(right_top + idx * (3 + gap))
        block = grid[r0 : r0 + 3, glyph_c0 : glyph_c0 + 3]
        block[mask] = int(glyph_colors[idx])

    expected = _expected_from_layout(grid, sep, row_count, col_count, left_top, left_pad, areas)
    left = grid[:, :sep]
    decoded_colors = [AREA_TO_COLOR[a] for a in areas]

    if np.array_equal(expected, _alt_vertical_first(left, h_count, decoded_colors)):
        return None
    alt_axis = _alt_reversed_axis(left, h_count, v_count, decoded_colors)
    if alt_axis is not None and np.array_equal(expected, alt_axis):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        built = _build_attempt(rng)
        if built is None:
            continue
        grid, expected = built
        if grid.ndim != 2 or expected.ndim != 2:
            continue
        if max(grid.shape) > MAX_DIM or max(expected.shape) > MAX_DIM:
            continue
        return grid
    raise RuntimeError("failed to generate a valid gutter-color decoding puzzle")
