import numpy as np

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the input sprite in a periodic right-and-down pattern where every third horizontal copy is pink.

    Input structure
    - A black/dominant background contains exactly one 8-connected foreground sprite.
    - The sprite has a tight bounding box; its mask, height, and width define the repeated tile.
    - Repeated copies are separated by one background row or column.

    Rule
    - Starting at the sprite's original upper-left bbox corner, place slots every sprite_width+1 columns.
    - On the original sprite row, fill the horizontal slots with the sprite mask; slots 0,1,3,4,... keep the source color, while slots 2,5,8,... become pink.
    - Starting at each pink horizontal slot, repeat pink copies downward every sprite_height+1 rows.
    - Clip any copy that extends beyond the grid edge.

    Key insight
    - The first row establishes a repeating two-source-color then one-pink rhythm, and only the pink columns continue vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    assert len(components) == 1, "expected exactly one foreground sprite"

    comp = components[0]
    source_color = int(comp["color"])
    r0, c0, _r1, _c1 = map(int, comp["bbox"])
    mask = crop_bbox(g == source_color, comp["bbox"])
    assert mask.any(), "sprite mask must be nonempty"
    sprite_h, sprite_w = map(int, mask.shape)
    assert sprite_h > 0 and sprite_w > 0, "sprite bbox must be nonempty"

    out = np.full(g.shape, int(bg), dtype=int)
    for row_slot, rr in enumerate(range(r0, g.shape[0], sprite_h + 1)):
        for col_slot, cc in enumerate(range(c0, g.shape[1], sprite_w + 1)):
            if col_slot % 3 == 2:
                color = Color.PINK
            elif row_slot == 0:
                color = source_color
            else:
                continue

            patch = np.full(mask.shape, int(bg), dtype=int)
            patch[mask] = int(color)
            paste(out, patch, rr, cc, transparent=bg, clip=True)

    return out

# --------------- generator (o5x) ---------------
# Canvas, colors, and placement bounds.
GRID_SHAPE, BACKGROUND, SPRITE_COLOR = (23, 28), 0, 7
START_ROW_RANGE, START_COL_RANGE = (4, 8), (2, 6)
# Sprite mask sampling.
SPRITE_WIDTH, SPRITE_HEIGHTS = 3, (3, 5)
ROW_KIND_COUNT, SIDE_ONLY_KIND, CENTER_ONLY_KIND, FULL_ROW_KIND = 3, 0, 1, 2
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300

def _gen_sprite_mask(height: int) -> np.ndarray:
    """Return a 3-wide LR/UD-symmetric mask that uses both side and center cells."""
    row_count = (height + 1) // 2
    row_kinds = np.random.randint(0, ROW_KIND_COUNT, size=row_count)
    if not np.any(row_kinds != SIDE_ONLY_KIND):
        row_kinds[np.random.randint(row_count)] = FULL_ROW_KIND
    if not np.any(row_kinds != CENTER_ONLY_KIND):
        row_kinds[np.random.randint(row_count)] = FULL_ROW_KIND
    mask = np.zeros((height, SPRITE_WIDTH), dtype=bool)
    for rr, kind in enumerate(row_kinds):
        side = kind != SIDE_ONLY_KIND
        center = kind != CENTER_ONLY_KIND
        row = np.array([side, center, side], dtype=bool)
        mask[rr] = row
        mask[height - 1 - rr] = row
    return mask


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.choice(SPRITE_HEIGHTS))
    mask = _gen_sprite_mask(height)
    r0 = int(np.random.randint(*START_ROW_RANGE))
    c0 = int(np.random.randint(*START_COL_RANGE))
    grid = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = SPRITE_COLOR
    paste(grid, patch, r0, c0, transparent=BACKGROUND)

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid d304284e input")
