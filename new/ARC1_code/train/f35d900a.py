import numpy as np

from arc_utils import Color, colors_present, get_bg


def _paint_dash_gap(out: np.ndarray, fixed: int, start: int, end: int, axis: str) -> None:
    assert start <= end
    assert axis in {"h", "v"}
    n = end - start + 1
    for i, pos in enumerate(range(start, end + 1)):
        if min(i, n - 1 - i) % 2 == 0:
            if axis == "h":
                out[fixed, pos] = Color.GREY
            else:
                out[pos, fixed] = Color.GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand four diagonal color markers into corner glyphs and add dashed grey sides between them.

    Input structure
    - The input is a background grid with exactly four foreground pixels.
    - The foreground has two colors, each appearing twice, on opposite corners of one
      axis-aligned rectangle.

    Rule
    - Each marker becomes a centered 3x3 glyph: the center keeps its input color and
      the eight surrounding cells use the other marker color.
    - In the open gap between adjacent glyphs on each rectangle side, paint grey dashes.
      A gap cell is grey when its nearest side-end distance is even.

    Key insight
    - The four pixels are rectangle corners; their diagonal color pairing defines the
      corner glyphs, and the sides are dashed symmetrically inward from both ends.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg != Color.GREY

    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2
    assert Color.GREY not in colors

    marker_coords = np.argwhere(g != bg)
    assert marker_coords.shape == (4, 2)
    for color in colors:
        assert int(np.sum(g == color)) == 2

    rows = sorted({int(r) for r in marker_coords[:, 0]})
    cols = sorted({int(c) for c in marker_coords[:, 1]})
    assert len(rows) == 2
    assert len(cols) == 2
    top, bottom = rows
    left, right = cols
    assert bottom - top >= 4
    assert right - left >= 4

    for r in rows:
        for c in cols:
            assert int(g[r, c]) in colors
            assert 0 < r < g.shape[0] - 1
            assert 0 < c < g.shape[1] - 1

    assert int(g[top, left]) == int(g[bottom, right])
    assert int(g[top, right]) == int(g[bottom, left])
    assert int(g[top, left]) != int(g[top, right])

    out = np.full_like(g, bg)
    for r, c in marker_coords:
        color = int(g[r, c])
        other = colors[0] if colors[1] == color else colors[1]
        out[r - 1 : r + 2, c - 1 : c + 2] = other
        out[r, c] = color

    _paint_dash_gap(out, top, left + 2, right - 2, "h")
    _paint_dash_gap(out, bottom, left + 2, right - 2, "h")
    _paint_dash_gap(out, left, top + 2, bottom - 2, "v")
    _paint_dash_gap(out, right, top + 2, bottom - 2, "v")

    return out

# --------------- generator (o5x) ---------------
# Canvas stops are exclusive; rectangle span max is inclusive.
_GEN_HEIGHT_RANGE = (14, 18)
_GEN_WIDTH_RANGE = (14, 19)
_GEN_RECT_SPAN_RANGE = (5, 11)
_GEN_EDGE_MARGIN = 1

# Palette, rendering geometry, and retry budget.
_GEN_BACKGROUND_COLOR = 0
_GEN_FOREGROUND_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_DASH_COLOR = 5
_GEN_GLYPH_RADIUS = 1
_GEN_DASH_INSET = 2
_GEN_MAX_ATTEMPTS = 300


def _gen_paint_dash_gap(out: np.ndarray, fixed: int, start: int, end: int, axis: str) -> None:
    offsets = np.arange(end - start + 1)
    positions = start + offsets[np.minimum(offsets, offsets[-1] - offsets) % 2 == 0]
    if axis == "h":
        out[fixed, positions] = _GEN_DASH_COLOR
    else:
        out[positions, fixed] = _GEN_DASH_COLOR


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    coords = np.argwhere(inp != 0)
    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})
    top, bottom = rows
    left, right = cols
    color_a = int(inp[top, left])
    color_b = int(inp[top, right])

    out = np.full_like(inp, _GEN_BACKGROUND_COLOR)
    rr = _GEN_GLYPH_RADIUS
    for r, c in coords:
        color = int(inp[r, c])
        other = color_b if color == color_a else color_a
        out[r - rr : r + rr + 1, c - rr : c + rr + 1] = other
        out[r, c] = color

    gap = _GEN_DASH_INSET
    _gen_paint_dash_gap(out, top, left + gap, right - gap, "h")
    _gen_paint_dash_gap(out, bottom, left + gap, right - gap, "h")
    _gen_paint_dash_gap(out, left, top + gap, bottom - gap, "v")
    _gen_paint_dash_gap(out, right, top + gap, bottom - gap, "v")
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*_GEN_HEIGHT_RANGE))
    width = int(np.random.randint(*_GEN_WIDTH_RANGE))
    rect_min, rect_max = _GEN_RECT_SPAN_RANGE
    margin_cost = 2 * _GEN_EDGE_MARGIN + 1
    rect_h = int(np.random.randint(rect_min, min(rect_max, height - margin_cost) + 1))
    rect_w = int(np.random.randint(rect_min, min(rect_max, width - margin_cost) + 1))
    top = int(np.random.randint(_GEN_EDGE_MARGIN, height - rect_h - _GEN_EDGE_MARGIN))
    left = int(np.random.randint(_GEN_EDGE_MARGIN, width - rect_w - _GEN_EDGE_MARGIN))
    bottom = top + rect_h
    right = left + rect_w
    color_a, color_b = [int(c) for c in np.random.choice(_GEN_FOREGROUND_COLORS, 2, replace=False)]

    inp = np.full((height, width), _GEN_BACKGROUND_COLOR, dtype=int)
    inp[top, left] = color_a
    inp[top, right] = color_b
    inp[bottom, left] = color_b
    inp[bottom, right] = color_a

    expected = _gen_expected_output(inp)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid f35d900a input")
