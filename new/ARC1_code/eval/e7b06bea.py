import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse the right-edge palette into one vertical coded bar using the grey marker height as the per-color run length.

    Input structure
    The grid has a dominant background, one top-aligned vertical grey marker
    segment, and a contiguous set of full-height non-background palette columns
    at the right edge. The column immediately to the left of the palette is
    background.

    Rule
    Read the palette columns left-to-right. Erase those palette columns, then
    fill the adjacent background column with the palette colors from top to
    bottom, repeating each palette color for as many rows as the grey marker
    segment is tall and cycling through the palette until the grid ends.

    Key insight
    The grey segment is not copied; its height is the run-length instruction
    for expanding the ordered right-edge palette into the new vertical bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape

    bg = get_bg(g)

    marker_mask = g == Color.GREY
    marker_coords = np.argwhere(marker_mask)
    assert marker_coords.size > 0, "expected a grey marker segment"
    marker_cols = np.unique(marker_coords[:, 1])
    assert len(marker_cols) == 1, "grey marker must occupy one column"
    marker_col = int(marker_cols[0])

    marker_rows = np.sort(np.unique(marker_coords[:, 0]))
    run_len = len(marker_rows)
    assert run_len == int(marker_mask.sum()), "grey marker must be one pixel wide"
    assert np.array_equal(marker_rows, np.arange(run_len)), "grey marker must be top-aligned"
    assert np.all(g[run_len:, marker_col] == bg), "marker column below marker must be background"

    palette_cols = []
    palette = []
    for c in range(W):
        values = np.unique(g[:, c])
        if len(values) != 1:
            continue
        color = int(values[0])
        if color != bg and color != Color.GREY:
            palette_cols.append(c)
            palette.append(color)

    assert palette_cols, "expected full-height palette columns"
    assert palette_cols == list(range(palette_cols[0], W)), "palette must be contiguous at right edge"

    target_col = palette_cols[0] - 1
    assert target_col >= 0, "palette must have a background column to its left"
    assert np.all(g[:, target_col] == bg), "target column must start as background"

    structural_mask = (g == bg) | marker_mask
    structural_mask[:, palette_cols] = True
    assert np.all(structural_mask), "unexpected foreground outside marker and palette"

    out = g.copy()
    out[:, palette_cols] = bg
    out[:, target_col] = [palette[(r // run_len) % len(palette)] for r in range(H)]
    return out

# --------------- generator (o5x) ---------------
# Sampling bounds and layout widths.
HEIGHT_RANGE, RUN_LEN_RANGE, PALETTE_COUNT_RANGE = (5, 14), (1, 4), (2, 3)
MIN_GAP, MAX_WIDTH, FIXED_LAYOUT_COLS = 1, 9, 2
# Color choices and retry budget.
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
BG_COLOR, MARKER_COLOR = 0, 5
GEN_RETRY_LIMIT = 300


def _gen_expected_output(inp: np.ndarray, palette: list[int], run_len: int, target_col: int, first_palette_col: int) -> np.ndarray:
    out = inp.copy()
    out[:, first_palette_col:] = BG_COLOR
    out[:, target_col] = [palette[(r // run_len) % len(palette)] for r in range(inp.shape[0])]
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*HEIGHT_RANGE)
    P = random.randint(*PALETTE_COUNT_RANGE)
    run_len = random.randint(*RUN_LEN_RANGE)
    gap = random.randint(MIN_GAP, MAX_WIDTH - P - FIXED_LAYOUT_COLS)
    W = FIXED_LAYOUT_COLS + gap + P

    inp = np.full((H, W), BG_COLOR, dtype=int)
    inp[:run_len, 0] = MARKER_COLOR

    palette = random.sample(PALETTE_COLORS, P)
    first_palette_col = W - P
    for i, color in enumerate(palette):
        inp[:, first_palette_col + i] = color

    target_col = first_palette_col - 1
    expected = _gen_expected_output(inp, palette, run_len, target_col, first_palette_col)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid e7b06bea input")
