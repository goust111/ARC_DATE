import numpy as np

from arc_utils import bbox, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground cell of the compact motif with a full copy of that motif in the corresponding output block.

    Input structure
    - The grid contains a single non-background color forming a compact square-bbox motif.
    - The full grid dimensions are the motif dimensions squared.

    Rule
    - Crop the foreground motif from its bounding box.
    - For each foreground pixel in the motif at local position (r, c), paste the whole motif into output block (r, c).
    - Leave all other blocks as background.

    Key insight
    - The sparse input is a self-expansion: the motif is both the block-placement mask and the stamp copied into every selected block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    color = fg_colors[0]

    fg = g == color
    fg_bbox = bbox(fg)
    assert fg_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = fg_bbox
    motif = g[r0 : r1 + 1, c0 : c1 + 1]
    motif_mask = motif == color
    motif_h, motif_w = motif.shape

    assert motif_h == motif_w, "expected a square motif bbox"
    assert g.shape == (motif_h * motif_h, motif_w * motif_w), "expected grid shape to be motif dimensions squared"

    stamp = np.where(motif_mask, color, bg)
    out = np.full_like(g, bg)
    for block_r, block_c in np.argwhere(motif_mask):
        rs = int(block_r) * motif_h
        cs = int(block_c) * motif_w
        out[rs : rs + motif_h, cs : cs + motif_w] = stamp

    return out

# --------------- generator (o5x) ---------------

# Geometry and color configuration.
MOTIF_SIZE = 3
GRID_SIZE = MOTIF_SIZE * MOTIF_SIZE
BACKGROUND_COLOR = 0
FOREGROUND_COLOR_RANGE = (1, 10)
# Motif density and retry budget.
FOREGROUND_COUNT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 300

def _gen_full_bbox_mask() -> np.ndarray:
    """Sample one random cell per row/column, then optional extra cells."""
    mask = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=bool)
    cols = np.random.permutation(MOTIF_SIZE)
    for r, c in enumerate(cols):
        mask[r, int(c)] = True

    k = int(np.random.randint(*FOREGROUND_COUNT_RANGE))
    if k > FOREGROUND_COUNT_RANGE[0]:
        remaining = np.argwhere(~mask)
        chosen = np.random.choice(len(remaining), size=k - FOREGROUND_COUNT_RANGE[0], replace=False)
        mask[remaining[chosen, 0], remaining[chosen, 1]] = True

    return mask


def _gen_attempt() -> np.ndarray | None:
    from arc_utils import paste

    color = int(np.random.randint(*FOREGROUND_COLOR_RANGE))
    mask = _gen_full_bbox_mask()
    motif = np.where(mask, color, BACKGROUND_COLOR).astype(int)

    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for br, bc in np.argwhere(mask):
        paste(out, motif, int(br) * MOTIF_SIZE, int(bc) * MOTIF_SIZE, transparent=None)

    fg_blocks = np.argwhere(mask)
    br, bc = fg_blocks[int(np.random.randint(len(fg_blocks)))]
    inp = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    paste(inp, motif, int(br) * MOTIF_SIZE, int(bc) * MOTIF_SIZE, transparent=None)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
