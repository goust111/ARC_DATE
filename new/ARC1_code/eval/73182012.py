import numpy as np

from arc_utils import bbox, get_bg, paste as _gen_paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the original quadrant from a four-way mirrored rectangular panel.

    Input structure
    - The input is a larger background grid containing one foreground panel.
    - The panel has even height and width.
    - Its four quadrants are reflections of the same source pattern.

    Rule
    - Crop to the foreground panel's bounding box.
    - Read the upper-left quadrant as the source pattern.
    - Verify that the other quadrants are its left-right, up-down, and 180-degree reflections.

    Key insight
    - The apparent 12x12 drawing is not a pixel-scale zoom; it is a reflected 2x2 mosaic whose upper-left quadrant is the requested 4x4 image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    box = bbox(g != bg)
    assert box is not None
    r0, c0, r1, c1 = box
    panel = g[r0 : r1 + 1, c0 : c1 + 1]

    h, w = panel.shape
    assert h % 2 == 0 and w % 2 == 0
    mh, mw = h // 2, w // 2

    source = panel[:mh, :mw]
    assert np.array_equal(panel[:mh, mw:], source[:, ::-1])
    assert np.array_equal(panel[mh:, :mw], source[::-1, :])
    assert np.array_equal(panel[mh:, mw:], source[::-1, ::-1])

    return source.copy()

# --------------- generator (o5x) ---------------
# Canvas and mirrored panel geometry.
QUADRANT_SIZE = 4
PANEL_SIZE = 2 * QUADRANT_SIZE
CANVAS_SIZE = 12
PLACEMENT_RANGE = (0, CANVAS_SIZE - PANEL_SIZE)

# Quadrant sampling constraints and retry policy.
COLOR_COUNT_RANGE = (3, 6)
ZERO_TARGET_RANGE = (4, 7)
NONZERO_COLORS = range(1, 10)
MAX_GENERATION_ATTEMPTS = 300

# Cached coordinate groups for the transpose-symmetric source quadrant.
DIAG_CELLS = [(i, i) for i in range(QUADRANT_SIZE)]
OFFDIAG_CELLS = [(r, c) for r in range(QUADRANT_SIZE) for c in range(r + 1, QUADRANT_SIZE)]
TOP_EDGE_CELLS = [(0, c) for c in range(QUADRANT_SIZE)]
UPPER_TRIANGLE_CELLS = [(r, c) for r in range(QUADRANT_SIZE) for c in range(r, QUADRANT_SIZE)]


def _gen_zero_positions(z_target: int, color_count: int) -> set[tuple[int, int]]:
    choices = []
    max_zero_upper_cells = len(UPPER_TRIANGLE_CELLS) - color_count
    for forced_nonzero in TOP_EDGE_CELLS:
        diag_pool = [cell for cell in DIAG_CELLS if cell != forced_nonzero]
        offdiag_pool = [cell for cell in OFFDIAG_CELLS if cell != forced_nonzero]
        for diag_count in range(len(diag_pool) + 1):
            rem = z_target - diag_count
            if rem % 2:
                continue
            offdiag_count = rem // 2
            upper_zero_count = diag_count + offdiag_count
            if 0 <= offdiag_count <= len(offdiag_pool) and upper_zero_count <= max_zero_upper_cells:
                choices.append((diag_pool, offdiag_pool, diag_count, offdiag_count))

    diag_pool, offdiag_pool, diag_count, offdiag_count = random.choice(choices)
    zeros = set(random.sample(diag_pool, diag_count))
    zeros.update(random.sample(offdiag_pool, offdiag_count))
    return zeros


def _gen_make_quadrant() -> np.ndarray:
    color_count = random.randint(*COLOR_COUNT_RANGE)
    z_target = random.randint(*ZERO_TARGET_RANGE)
    zero_positions = _gen_zero_positions(z_target, color_count)
    nonzero_cells = [cell for cell in UPPER_TRIANGLE_CELLS if cell not in zero_positions]

    colors = random.sample(NONZERO_COLORS, color_count)
    assignments = colors + [random.choice(colors) for _ in range(len(nonzero_cells) - color_count)]
    random.shuffle(assignments)

    out = np.zeros((QUADRANT_SIZE, QUADRANT_SIZE), dtype=int)
    for (r, c), color in zip(nonzero_cells, assignments):
        out[r, c] = color
        out[c, r] = color
    return out


def _gen_make_panel(source: np.ndarray) -> np.ndarray:
    top = np.concatenate([source, source[:, ::-1]], axis=1)
    bottom = np.concatenate([source[::-1, :], source[::-1, ::-1]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _gen_attempt() -> np.ndarray | None:
    source = _gen_make_quadrant()
    panel = _gen_make_panel(source)
    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    r0 = random.randint(*PLACEMENT_RANGE)
    c0 = random.randint(*PLACEMENT_RANGE)
    _gen_paste(inp, panel, r0, c0)

    try:
        if np.array_equal(main(inp), source):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
