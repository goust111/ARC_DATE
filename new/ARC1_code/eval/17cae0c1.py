import numpy as np

from arc_utils import Color


_SYMBOL_TO_COLOR = {
    ((1, 1, 1), (1, 0, 1), (1, 1, 1)): Color.GREEN,
    ((0, 0, 0), (0, 1, 0), (0, 0, 0)): Color.YELLOW,
    ((0, 0, 1), (0, 1, 0), (1, 0, 0)): Color.MAROON,
    ((0, 0, 0), (0, 0, 0), (1, 1, 1)): Color.BLUE,
    ((1, 1, 1), (0, 0, 0), (0, 0, 0)): Color.PINK,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each 3x3 grey symbol into its color and fill that symbol's whole 3-column panel.

    Input structure
    - The grid is three rows high and is partitioned into adjacent 3x3 panels.
    - Each panel is a black/grey symbol mask.
    - Every symbol is one of the learned task glyphs.

    Rule
    - Convert grey pixels in each 3x3 panel to a binary mask.
    - Decode the mask with the task's symbol-to-color codebook.
    - Replace the entire 3x3 panel by the decoded color.

    Key insight
    - The output preserves the panel positions but discards the internal black/grey drawing; each drawing is a compact symbol for a solid color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == 3, "expected three-row symbol panels"
    assert width > 0, "expected at least one symbol panel"
    assert width % 3 == 0, "expected an integer number of 3-column panels"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.GREY}, "expected only black and grey input pixels"

    out = np.empty_like(g)
    for col0 in range(0, width, 3):
        block = g[:, col0 : col0 + 3]
        assert block.shape == (3, 3), "expected each panel to be 3x3"
        symbol = tuple(tuple(map(int, row)) for row in (block == Color.GREY))
        assert symbol in _SYMBOL_TO_COLOR, "encountered an unknown grey symbol"
        out[:, col0 : col0 + 3] = _SYMBOL_TO_COLOR[symbol]

    return out

# --------------- generator (o5x) ---------------

# Panel geometry.
_GEN_PANEL_SIZE_17CAE0C1 = 3
_GEN_PANEL_COUNT_17CAE0C1 = 3
_GEN_GRID_SHAPE_17CAE0C1 = (
    _GEN_PANEL_SIZE_17CAE0C1,
    _GEN_PANEL_SIZE_17CAE0C1 * _GEN_PANEL_COUNT_17CAE0C1,
)

# Rendering colors.
_GEN_BACKGROUND_17CAE0C1 = Color.BLACK
_GEN_INK_17CAE0C1 = Color.GREY

# Sampling and validation.
_GEN_RETRY_LIMIT_17CAE0C1 = 300
_GEN_SAMPLE_REPLACE_17CAE0C1 = False

# Codebook masks, in the same order as decoded output colors.
_GEN_SYMBOLS_17CAE0C1 = np.array(
    [
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
    ],
    dtype=int,
)

_GEN_SYMBOL_COLORS_17CAE0C1 = (
    Color.GREEN,
    Color.YELLOW,
    Color.MAROON,
    Color.BLUE,
    Color.PINK,
)


def _gen_panel_slice_17cae0c1(panel_i: int) -> slice:
    c0 = _GEN_PANEL_SIZE_17CAE0C1 * int(panel_i)
    return slice(c0, c0 + _GEN_PANEL_SIZE_17CAE0C1)


def _gen_decode_symbol_17cae0c1(symbol: np.ndarray) -> int | None:
    """Return the decoded color for a generated mask, or None for unknown masks."""
    for template, color in zip(_GEN_SYMBOLS_17CAE0C1, _GEN_SYMBOL_COLORS_17CAE0C1):
        if np.array_equal(symbol, template):
            return int(color)
    return None


def _gen_expected_output_17cae0c1(inp: np.ndarray) -> np.ndarray | None:
    """Decode generated panels to the oracle output, rejecting unknown masks."""
    expected = np.empty_like(inp)
    for panel_i, c0 in enumerate(range(0, inp.shape[1], _GEN_PANEL_SIZE_17CAE0C1)):
        cols = _gen_panel_slice_17cae0c1(panel_i)
        symbol = (inp[:, cols] == _GEN_INK_17CAE0C1).astype(int)
        color = _gen_decode_symbol_17cae0c1(symbol)
        if color is None:
            return None
        expected[:, c0 : c0 + _GEN_PANEL_SIZE_17CAE0C1] = color
    return expected


def _gen_sample_symbol_idxs_17cae0c1() -> np.ndarray:
    return np.random.choice(
        len(_GEN_SYMBOLS_17CAE0C1),
        _GEN_PANEL_COUNT_17CAE0C1,
        replace=_GEN_SAMPLE_REPLACE_17CAE0C1,
    )


def _gen_render_symbols_17cae0c1(symbol_idxs: np.ndarray) -> np.ndarray:
    """Render selected codebook masks as black/grey panels in order."""
    grid = np.full(_GEN_GRID_SHAPE_17CAE0C1, _GEN_BACKGROUND_17CAE0C1, dtype=int)
    for i, symbol_i in enumerate(symbol_idxs):
        cols = _gen_panel_slice_17cae0c1(i)
        symbol = _GEN_SYMBOLS_17CAE0C1[int(symbol_i)]
        grid[:, cols] = np.where(symbol, _GEN_INK_17CAE0C1, _GEN_BACKGROUND_17CAE0C1)

    return grid


def _gen_attempt() -> np.ndarray | None:
    symbol_idxs = _gen_sample_symbol_idxs_17cae0c1()
    return _gen_render_symbols_17cae0c1(symbol_idxs)


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT_17CAE0C1):
        inp = _gen_attempt()
        if inp is None:
            continue
        out = _gen_expected_output_17cae0c1(inp)
        if out is not None and np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid 17cae0c1 input")
