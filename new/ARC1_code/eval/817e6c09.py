import numpy as np

from arc_utils import Color, bbox_h, bbox_w, colors_present, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor alternating 2x2 red blocks teal, starting with the rightmost block.

    Input structure
    A black background contains one foreground color: disjoint 2x2 red square
    objects whose left edges have distinct columns.

    Rule
    Sort the red squares from right to left. Recolor ranks 0, 2, 4, ... in that
    order to teal, and leave the intervening squares red.

    Key insight
    Row position is a distractor; the visible sequence is ordered horizontally,
    and the selected parity is anchored at the rightmost square.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK

    fg_colors = colors_present(g, exclude={bg})
    assert fg_colors == [Color.RED]

    blocks = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.RED],
    )
    assert blocks

    left_cols = []
    for block in blocks:
        assert block["area"] == 4
        assert bbox_h(block["bbox"]) == 2
        assert bbox_w(block["bbox"]) == 2
        left_cols.append(block["bbox"][1])
    assert len(set(left_cols)) == len(left_cols)

    out = g.copy()
    for rank, block in enumerate(sorted(blocks, key=lambda b: b["bbox"][1], reverse=True)):
        if rank % 2 == 0:
            coords = block["coords"]
            out[coords[:, 0], coords[:, 1]] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

# CFG: canvas and block geometry.
ROW_COUNT, MIN_WIDTH, MAX_WIDTH, BLOCK_SIZE = 7, 5, 17, 2
# CFG: color roles.
BG_COLOR, INPUT_BLOCK_COLOR, OUTPUT_BLOCK_COLOR = Color.BLACK, Color.RED, Color.TEAL
# CFG: block and layout sampling.
MIN_BLOCKS, MAX_BLOCKS = 2, 9
GAP_VALUES = (1, 2, 3)
MIN_GAP, MAX_GAP = min(GAP_VALUES), max(GAP_VALUES)
START_COL_ONE_PROB = 0.15
MIN_WIDTH_DENSITY_NUM, MIN_WIDTH_DENSITY_DEN, MIN_WIDTH_DENSITY_PAD = 8, 7, 1
# CFG: generation retry budget.
GENERATION_RETRIES = 300


def _gen_gap_sequence(n_gaps, start_col, n_blocks):
    min_width = max(MIN_WIDTH, (MIN_WIDTH_DENSITY_NUM * n_blocks) // MIN_WIDTH_DENSITY_DEN + MIN_WIDTH_DENSITY_PAD)
    min_total = max(MIN_GAP * n_gaps, min_width - start_col - BLOCK_SIZE)
    max_total = min(MAX_GAP * n_gaps, MAX_WIDTH - start_col - BLOCK_SIZE)
    if min_total > max_total:
        return None

    target = int(np.random.randint(min_total, max_total + 1))
    gaps = []
    remaining = target
    for i in range(n_gaps):
        slots = n_gaps - i - 1
        valid = [gap for gap in GAP_VALUES if MIN_GAP * slots <= remaining - gap <= MAX_GAP * slots]
        if not valid:
            return None
        gap = int(np.random.choice(valid))
        gaps.append(gap)
        remaining -= gap
    return gaps


def _build_left_cols(start_col, gaps):
    left_cols = [start_col]
    for gap in gaps:
        left_cols.append(left_cols[-1] + gap)
    return left_cols


def _can_place_row(left_cols, rows, idx, top_row):
    col = left_cols[idx]
    for j in range(idx):
        delta = col - left_cols[j]
        if delta > BLOCK_SIZE:
            continue
        if delta == BLOCK_SIZE - 1 and abs(top_row - rows[j]) < BLOCK_SIZE + 1:
            return False
        if delta == BLOCK_SIZE and abs(top_row - rows[j]) < BLOCK_SIZE:
            return False
    return True


def _gen_rows_for_left_cols(left_cols):
    rows = [-1] * len(left_cols)

    def place(idx):
        if idx == len(left_cols):
            return True
        for top_row in np.random.permutation(ROW_COUNT - BLOCK_SIZE + 1):
            top_row = int(top_row)
            if _can_place_row(left_cols, rows, idx, top_row):
                rows[idx] = top_row
                if place(idx + 1):
                    return True
                rows[idx] = -1
        return False

    return rows if place(0) else None


def _gen_attempt():
    n_blocks = int(np.random.randint(MIN_BLOCKS, MAX_BLOCKS + 1))
    start_col = 1 if np.random.random() < START_COL_ONE_PROB else 0
    gaps = _gen_gap_sequence(n_blocks - 1, start_col, n_blocks)
    if gaps is None:
        return None

    left_cols = _build_left_cols(start_col, gaps)
    width = left_cols[-1] + BLOCK_SIZE
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        return None

    top_rows = _gen_rows_for_left_cols(left_cols)
    if top_rows is None:
        return None

    inp = np.full((ROW_COUNT, width), BG_COLOR, dtype=int)
    for top_row, left_col in zip(top_rows, left_cols):
        inp[top_row : top_row + BLOCK_SIZE, left_col : left_col + BLOCK_SIZE] = INPUT_BLOCK_COLOR

    expected = inp.copy()
    for rank, idx in enumerate(np.argsort(left_cols)[::-1]):
        if rank % 2 == 0:
            top_row = top_rows[int(idx)]
            left_col = left_cols[int(idx)]
            expected[top_row : top_row + BLOCK_SIZE, left_col : left_col + BLOCK_SIZE] = OUTPUT_BLOCK_COLOR

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 817e6c09 input")
