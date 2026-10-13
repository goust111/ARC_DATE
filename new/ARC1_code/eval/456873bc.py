import numpy as np

from arc_utils import Color, bbox, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Complete a self-similar lattice of red sprites and mark each copy's lattice coordinate in teal.

    Input structure
    - The grid is a square partitioned into a k by k lattice of k by k slots, with
      uniform background separators between slots.
    - One foreground color forms a solid rectangle that hides one whole slot row
      or one whole slot column; the other foreground color forms repeated copies
      of a single k by k sprite in the visible slots.

    Rule
    - Use the sprite's own foreground mask as the coarse lattice mask: a slot
      contains a copy exactly when the corresponding cell in the sprite is filled.
    - Erase the occluder, redraw every required sprite copy, and recolor in each
      copy the cell whose local coordinate equals that copy's lattice coordinate.

    Key insight
    - The repeated sprite is a miniature map of the complete coarse arrangement,
      so the hidden row or column is recovered by reading that miniature map.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H == W

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2

    solid_rects = []
    for color in fg_colors:
        mask = g == color
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if int(mask.sum()) == area:
            solid_rects.append((color, b))
    assert len(solid_rects) == 1

    occluder_color, (r0, c0, r1, c1) = solid_rects[0]
    sprite_colors = [color for color in fg_colors if color != occluder_color]
    assert len(sprite_colors) == 1
    sprite_color = sprite_colors[0]

    hides_slot_row = c0 == 0 and c1 == W - 1
    hides_slot_col = r0 == 0 and r1 == H - 1
    assert hides_slot_row != hides_slot_col

    k = (r1 - r0 + 1) if hides_slot_row else (c1 - c0 + 1)
    assert k > 1
    assert (H - k * k) % (k - 1) == 0
    gap = (H - k * k) // (k - 1)
    assert gap >= 1

    starts = [i * (k + gap) for i in range(k)]
    assert starts[-1] + k == H
    assert (r0 if hides_slot_row else c0) in starts

    visible_sprite_masks = []
    for rs in starts:
        for cs in starts:
            block = g[rs : rs + k, cs : cs + k]
            if np.any(block == occluder_color):
                continue
            assert np.all((block == bg) | (block == sprite_color))
            sprite_mask = block == sprite_color
            if np.any(sprite_mask):
                visible_sprite_masks.append(sprite_mask)
    assert visible_sprite_masks

    sprite_mask = visible_sprite_masks[0]
    for mask in visible_sprite_masks:
        assert np.array_equal(mask, sprite_mask)

    empty_mask = np.zeros((k, k), dtype=bool)
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            block = g[rs : rs + k, cs : cs + k]
            if np.any(block == occluder_color):
                continue
            expected = sprite_mask if sprite_mask[br, bc] else empty_mask
            assert np.array_equal(block == sprite_color, expected)

    out = np.full_like(g, bg)
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            if not sprite_mask[br, bc]:
                continue
            block = out[rs : rs + k, cs : cs + k]
            block[sprite_mask] = sprite_color
            block[br, bc] = Color.TEAL

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry settings.
K_CHOICES, GAP, MAX_GENERATION_ATTEMPTS = [3, 4], 1, 300
BACKGROUND_COLOR, SPRITE_COLOR, MARK_COLOR, OCCLUDER_COLOR = Color.BLACK, Color.RED, Color.TEAL, Color.GREEN
# Sprite sampling bounds.
MIN_FILLED_CELLS, FILL_RATIO_MIN, FILL_RATIO_MAX = 2, 0.50, 0.65
# Occluder axis choices.
HORIZONTAL_AXIS, AXIS_COUNT = 0, 2


def _gen_sprite_mask(k: int) -> np.ndarray:
    total = k * k
    counts = [
        n for n in range(MIN_FILLED_CELLS, total + 1)
        if FILL_RATIO_MIN <= n / total <= FILL_RATIO_MAX
    ]
    count = counts[int(np.random.randint(len(counts)))]
    flat = np.zeros(total, dtype=bool)
    flat[np.random.choice(total, size=count, replace=False)] = True
    return flat.reshape(k, k)

def _gen_build_output(sprite_mask: np.ndarray, starts: list[int]) -> np.ndarray:
    k = sprite_mask.shape[0]
    size = k * k + (k - 1) * GAP
    out = np.full((size, size), BACKGROUND_COLOR, dtype=int)
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            if not sprite_mask[br, bc]:
                continue
            block = out[rs:rs + k, cs:cs + k]
            block[sprite_mask] = SPRITE_COLOR
            block[br, bc] = MARK_COLOR
    return out

def _choose_hidden_slot(sprite_mask: np.ndarray, axis: int) -> int:
    valid = np.flatnonzero(sprite_mask.any(axis=1 if axis == HORIZONTAL_AXIS else 0))
    return int(valid[np.random.randint(len(valid))])

def _apply_occluder(inp: np.ndarray, sprite_mask: np.ndarray, starts: list[int], axis: int) -> None:
    k = sprite_mask.shape[0]
    start = starts[_choose_hidden_slot(sprite_mask, axis)]
    if axis == HORIZONTAL_AXIS:
        inp[start:start + k, :] = OCCLUDER_COLOR
    else:
        inp[:, start:start + k] = OCCLUDER_COLOR

def _gen_attempt() -> np.ndarray | None:
    k = int(np.random.choice(K_CHOICES))
    sprite_mask = _gen_sprite_mask(k)
    starts = [i * (k + GAP) for i in range(k)]
    out = _gen_build_output(sprite_mask, starts)

    inp = out.copy()
    inp[inp == MARK_COLOR] = SPRITE_COLOR

    axis = int(np.random.randint(AXIS_COUNT))
    _apply_occluder(inp, sprite_mask, starts, axis)

    if np.array_equal(main(inp), out):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
