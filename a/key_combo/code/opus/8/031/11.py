from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_RANGE = (17, 23)
RADIATION_MARGIN = 3
RECT_SPAN_MIN = 7
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != bg)


def _glyph_coords(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _run_cells(center: tuple[int, int], dr: int, dc: int, length: int, offset: int) -> list[tuple[int, int]]:
    r, c = center
    if dr != 0:
        rr = r + dr * offset
        half = length // 2
        return [(rr, cc) for cc in range(c - half, c + half + 1)]
    cc = c + dc * offset
    half = length // 2
    return [(rr, cc) for rr in range(r - half, r + half + 1)]


def _dashed_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    n = len(cells)
    return [cell for i, cell in enumerate(cells) if min(i, n - 1 - i) % 2 == 0]


def _parse_markers(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    bg = BG
    assert int(np.min(g)) >= 0
    assert int(np.max(g)) <= 9
    assert int(np.count_nonzero(g == bg)) > 0

    coords = np.argwhere(g != bg)
    assert coords.shape == (4, 2), "expected exactly four markers"
    colors = _colors_present(g, bg)
    assert len(colors) == 2, "expected exactly two marker colors"
    for color in colors:
        assert int(np.sum(g == color)) == 2, "each color must appear twice"

    rows = sorted(int(r) for r in np.unique(coords[:, 0]))
    cols = sorted(int(c) for c in np.unique(coords[:, 1]))
    assert len(rows) == 2 and len(cols) == 2, "markers must form a rectangle"
    top, bottom = rows
    left, right = cols
    assert bottom > top and right > left
    assert bottom - top >= 5 and right - left >= 5

    corners = {
        "tl": (top, left),
        "tr": (top, right),
        "bl": (bottom, left),
        "br": (bottom, right),
    }
    for pos in corners.values():
        assert int(g[pos]) in colors, "all rectangle corners must be occupied"

    assert int(g[corners["tl"]]) == int(g[corners["br"]]), "same color must occupy one diagonal"
    assert int(g[corners["tr"]]) == int(g[corners["bl"]]), "same color must occupy the other diagonal"
    assert int(g[corners["tl"]]) != int(g[corners["tr"]]), "the two diagonals must use different colors"

    return {
        "grid": g,
        "bg": bg,
        "colors": colors,
        "corners": corners,
    }


def _render(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_markers(grid)
    g = parsed["grid"]
    bg = parsed["bg"]
    color_a, color_b = parsed["colors"]
    corners = parsed["corners"]

    out = np.full_like(g, bg)
    glyph_mask = np.zeros(g.shape, dtype=bool)
    corner_list = [corners["tl"], corners["tr"], corners["bl"], corners["br"]]

    for r, c in corner_list:
        glyph_mask[r - 1 : r + 2, c - 1 : c + 2] = True

    for r, c in corner_list:
        center_color = int(g[r, c])
        ring_color = color_b if center_color == color_a else color_a
        out[r - 1 : r + 2, c - 1 : c + 2] = ring_color
        out[r, c] = center_color

    for r, c in corner_list:
        center_color = int(g[r, c])
        ring_color = color_b if center_color == color_a else color_a
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for length, offset in ((3, 2), (1, 3)):
                for rr, cc in _dashed_cells(_run_cells((r, c), dr, dc, length, offset)):
                    if _in_bounds(out.shape, rr, cc) and not glyph_mask[rr, cc]:
                        out[rr, cc] = ring_color
    return out


def _collect_radiation_cells(grid: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    parsed = _parse_markers(grid)
    cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for center in parsed["corners"].values():
        emitted: set[tuple[int, int]] = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for length, offset in ((3, 2), (1, 3)):
                emitted.update(_dashed_cells(_run_cells(center, dr, dc, length, offset)))
        cells[center] = emitted
    return cells


def _validate_input(grid: np.ndarray) -> bool:
    parsed = _parse_markers(grid)
    g = parsed["grid"]
    corners = parsed["corners"]
    h, w = g.shape

    glyph_cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
    all_glyph: set[tuple[int, int]] = set()
    for center in corners.values():
        r, c = center
        assert RADIATION_MARGIN <= r < h - RADIATION_MARGIN
        assert RADIATION_MARGIN <= c < w - RADIATION_MARGIN
        cells = set(_glyph_coords(r, c))
        assert all(_in_bounds(g.shape, rr, cc) for rr, cc in cells)
        glyph_cells[center] = cells
        assert all_glyph.isdisjoint(cells), "glyphs must be disjoint"
        all_glyph |= cells

    emitted_by_center = _collect_radiation_cells(g)
    all_emitted: set[tuple[int, int]] = set()
    for center, emitted in emitted_by_center.items():
        assert all(_in_bounds(g.shape, rr, cc) for rr, cc in emitted), "radiation must fully fit"
        assert emitted.isdisjoint(all_glyph), "radiation cannot enter glyph area"
        assert all_emitted.isdisjoint(emitted), "radiations should remain visually separate"
        all_emitted |= emitted

    out = _render(g)
    assert out.shape == g.shape
    assert np.any(out != g), "output must change the grid"
    assert len(_colors_present(out, BG)) == 2, "output should still use exactly the two structural colors"
    assert int(np.sum(out != BG)) == len(all_glyph) + len(all_emitted), "pixel accounting must match glyphs plus radiation"
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand diagonal marker pairs into opposite-color corner glyphs and emit shrinking dashed radiation.

    Input structure:
    - The background is black, with exactly four non-black marker pixels.
    - The four markers sit at the corners of one axis-aligned rectangle.
    - Exactly two colors are present, each appearing twice on opposite corners of the rectangle.

    Rule:
    - Replace each marker with a centered 3x3 glyph whose center keeps the marker color and whose other eight cells use the opposite diagonal color.
    - From each glyph, emit four outward orthogonal runs: a length-3 layer one cell beyond the glyph edge, then a centered length-1 layer one step farther.
    - In every emitted run, color only the cells whose distance to the nearest run end is even; glyph cells keep priority if a radiation cell would overlap them.

    Key insight:
    - The rectangle corners identify both the four glyph centers and the two complementary colors, and each glyph then radiates a fixed shrinking dashed pattern in the four cardinal directions.
    """
    return _render(np.asarray(grid, dtype=int))


def _choose_spans(max_h_span: int, max_w_span: int) -> tuple[int, int]:
    branch = random.choice(["square", "tall", "wide"])
    if branch == "square":
        base = random.randint(RECT_SPAN_MIN, min(max_h_span, max_w_span))
        h_span = base
        w_span = random.randint(max(RECT_SPAN_MIN, base - 2), min(max_w_span, base + 2))
        return h_span, w_span
    if branch == "tall":
        h_span = random.randint(RECT_SPAN_MIN + 1, max_h_span)
        w_hi = min(max_w_span, max(RECT_SPAN_MIN, h_span - 1))
        w_span = random.randint(RECT_SPAN_MIN, w_hi)
        return h_span, w_span
    w_span = random.randint(RECT_SPAN_MIN + 1, max_w_span)
    h_hi = min(max_h_span, max(RECT_SPAN_MIN, w_span - 1))
    h_span = random.randint(RECT_SPAN_MIN, h_hi)
    return h_span, w_span


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    max_h_span = h - 1 - 2 * RADIATION_MARGIN
    max_w_span = w - 1 - 2 * RADIATION_MARGIN
    if max_h_span < RECT_SPAN_MIN or max_w_span < RECT_SPAN_MIN:
        return None

    h_span, w_span = _choose_spans(max_h_span, max_w_span)
    top = random.randint(RADIATION_MARGIN, h - 1 - RADIATION_MARGIN - h_span)
    left = random.randint(RADIATION_MARGIN, w - 1 - RADIATION_MARGIN - w_span)
    bottom = top + h_span
    right = left + w_span
    color_a, color_b = random.sample(PALETTE, 2)

    grid = np.zeros((h, w), dtype=int)
    grid[top, left] = color_a
    grid[bottom, right] = color_a
    grid[top, right] = color_b
    grid[bottom, left] = color_b

    try:
        if _validate_input(grid):
            expected = _render(grid)
            actual = main(grid)
            if np.array_equal(actual, expected):
                return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
