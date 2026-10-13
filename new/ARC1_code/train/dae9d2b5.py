import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the two side-by-side foreground masks and render their union in pink.

    Input structure
    The input is two equal square panels placed horizontally on a black
    background. Each panel contains exactly one non-background foreground color,
    and the two panel colors are different. Pink is absent from the input.

    Rule
    Split the grid at the vertical midpoint. For each position in one panel,
    output pink if the corresponding cell is foreground in either input panel;
    otherwise output black.

    Key insight
    The yellow and green cells are aligned masks, not colors to preserve; the
    output is the boolean union of the two masks recolored to a single color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    h, w = g.shape
    assert w % 2 == 0, "input width must split into two equal panels"
    panel_w = w // 2
    assert h == panel_w, "the two panels must be square"

    bg = get_bg(g)
    assert bg == Color.BLACK, "black must be the panel background"
    assert not np.any(g == Color.PINK), "pink must be reserved for the output mask"

    left = g[:, :panel_w]
    right = g[:, panel_w:]
    left_colors = set(colors_present(left, exclude={bg}))
    right_colors = set(colors_present(right, exclude={bg}))
    assert len(left_colors) == 1, "left panel must contain exactly one foreground color"
    assert len(right_colors) == 1, "right panel must contain exactly one foreground color"
    assert left_colors.isdisjoint(right_colors), "panel foreground colors must be distinct"

    mask = (left != bg) | (right != bg)
    out = np.full((h, panel_w), bg, dtype=int)
    out[mask] = Color.PINK
    return out

# --------------- generator (o5x) ---------------

# Geometry and fixed task colors.
PANEL_SIZE = 3
PANEL_CELLS = PANEL_SIZE * PANEL_SIZE
PANEL_SHAPE, INPUT_SHAPE = (PANEL_SIZE, PANEL_SIZE), (PANEL_SIZE, PANEL_SIZE * 2)
BG_COLOR, LEFT_COLOR = Color.BLACK, Color.YELLOW
RIGHT_COLOR, OUTPUT_COLOR = Color.GREEN, Color.PINK

# Panel fill counts, overlap limits, and retry budget.
PANEL_FILL_MIN, PANEL_FILL_MAX_EXCLUSIVE = 2, 6
MAX_OVERLAP_CELLS, UNION_CELL_MIN, UNION_CELL_MAX = 3, 4, 8
MAX_GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    left_count = int(np.random.randint(PANEL_FILL_MIN, PANEL_FILL_MAX_EXCLUSIVE))
    right_count = int(np.random.randint(PANEL_FILL_MIN, PANEL_FILL_MAX_EXCLUSIVE))
    valid_overlaps = [
        k
        for k in range(min(left_count, right_count))
        if k <= MAX_OVERLAP_CELLS
        and UNION_CELL_MIN <= left_count + right_count - k <= UNION_CELL_MAX
    ]
    if not valid_overlaps:
        return None

    overlap_count = int(np.random.choice(valid_overlaps))
    left_unique_count = left_count - overlap_count
    right_unique_count = right_count - overlap_count

    order = np.random.permutation(PANEL_CELLS)
    shared = order[:overlap_count]
    left_unique = order[overlap_count:left_count]
    right_unique = order[left_count : left_count + right_unique_count]

    left_mask = np.zeros(PANEL_SHAPE, dtype=bool)
    right_mask = np.zeros(PANEL_SHAPE, dtype=bool)
    left_mask.flat[shared] = True
    right_mask.flat[shared] = True
    left_mask.flat[left_unique] = True
    right_mask.flat[right_unique] = True

    inp = np.full(INPUT_SHAPE, BG_COLOR, dtype=int)
    left_panel = inp[:, :PANEL_SIZE]
    right_panel = inp[:, PANEL_SIZE:]
    left_panel[left_mask] = LEFT_COLOR
    right_panel[right_mask] = RIGHT_COLOR

    out = np.full(PANEL_SHAPE, BG_COLOR, dtype=int)
    out[left_mask | right_mask] = OUTPUT_COLOR
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid dae9d2b5 input")
