import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay two stacked foreground masks and recolor their union yellow.

    Input structure
    - The input is a 2D grid with black background.
    - The grid has even height and splits into two equal-size panels.
    - The top panel contains a green mask and the bottom panel contains a grey mask.
    Rule
    - Align the two panels cell by cell.
    - Output one panel-sized black grid.
    - Paint a yellow pixel wherever either aligned input cell is foreground.
    Key insight
    - The output is the union of the two foreground masks after vertical alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0, "expected two equal-height panels"
    assert height > 0 and width > 0, "expected nonempty panels"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    half = height // 2
    top = g[:half, :]
    bottom = g[half:, :]
    assert top.shape == bottom.shape, "expected aligned top and bottom panels"

    top_colors = set(colors_present(top, exclude={bg}))
    bottom_colors = set(colors_present(bottom, exclude={bg}))
    assert top_colors == {Color.GREEN}, "expected a green mask in the top panel"
    assert bottom_colors == {Color.GREY}, "expected a grey mask in the bottom panel"

    out = np.full(top.shape, Color.BLACK, dtype=int)
    out[(top != bg) | (bottom != bg)] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------

# Panel geometry, colors, count ranges, and oracle retry budget.
_GEN_PANEL_SHAPE = (6, 4)
_GEN_PANEL_CELLS = _GEN_PANEL_SHAPE[0] * _GEN_PANEL_SHAPE[1]
_GEN_BACKGROUND_COLOR = 0
_GEN_TOP_COLOR = 3
_GEN_BOTTOM_COLOR = 5
_GEN_OUTPUT_COLOR = 4
_GEN_MASK_COUNT_RANGE = range(8, 15)
_GEN_UNION_COUNT_RANGE = range(12, 22)
_GEN_RETRY_LIMIT = 300

_GEN_RNG = np.random.default_rng()
_GEN_VALID_COUNT_TRIPLES = tuple(
    (top_n, bottom_n, union_n)
    for top_n in _GEN_MASK_COUNT_RANGE
    for bottom_n in _GEN_MASK_COUNT_RANGE
    for union_n in _GEN_UNION_COUNT_RANGE
    if max(top_n, bottom_n) <= union_n <= top_n + bottom_n
)


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    half = inp.shape[0] // 2
    top, bottom = inp[:half], inp[half:]
    out = np.zeros(top.shape, dtype=int)
    out[(top != _GEN_BACKGROUND_COLOR) | (bottom != _GEN_BACKGROUND_COLOR)] = _GEN_OUTPUT_COLOR
    return out


def _sample_mask_cells(
    top_n: int, bottom_n: int, union_n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat indices for two masks with the requested union size."""
    overlap_n = top_n + bottom_n - union_n
    union_cells = _GEN_RNG.choice(_GEN_PANEL_CELLS, size=union_n, replace=False)
    union_cells = _GEN_RNG.permutation(union_cells)
    overlap = union_cells[:overlap_n]
    top_cells = np.concatenate([overlap, union_cells[overlap_n:top_n]])
    bottom_cells = np.concatenate([overlap, union_cells[top_n:]])
    return top_cells, bottom_cells


def _gen_attempt() -> np.ndarray:
    top_n, bottom_n, union_n = _GEN_VALID_COUNT_TRIPLES[
        int(_GEN_RNG.integers(len(_GEN_VALID_COUNT_TRIPLES)))
    ]
    top_cells, bottom_cells = _sample_mask_cells(top_n, bottom_n, union_n)
    top = np.zeros(_GEN_PANEL_SHAPE, dtype=int)
    bottom = np.zeros(_GEN_PANEL_SHAPE, dtype=int)
    top.flat[top_cells] = _GEN_TOP_COLOR
    bottom.flat[bottom_cells] = _GEN_BOTTOM_COLOR
    return np.vstack([top, bottom])


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        expected = _gen_expected_output(inp)
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid d19f7514 input")
