from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, components_mask, crop_bbox, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vertically align all repeated colored sprites to the teal sprite's top row.

    Input structure
    The grid has a uniform background and several foreground colors. Each
    foreground color forms exactly one 4-connected sprite, and all sprites have
    the same normalized mask. The teal sprite is the vertical alignment anchor.

    Rule
    Clear the canvas to background. For every colored sprite, keep its original
    left column and shape, but paste it with its top row equal to the teal
    sprite's original top row.

    Key insight
    The sprites are already horizontally separated copies of the same mask; the
    only missing structure is their common vertical alignment to the teal copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert Color.TEAL in colors, "expected one teal anchor sprite"

    bboxes = {}
    sprites = {}
    anchor_sprite = None
    anchor_bbox = None

    for color in colors:
        mask = g == color
        comps = components_mask(mask, connectivity=4)
        assert len(comps) == 1, "each color must form one sprite"

        b = bbox(mask)
        assert b is not None, "sprite mask must be non-empty"
        sprite = crop_bbox(mask, b)
        bboxes[color] = b
        sprites[color] = sprite

        if color == Color.TEAL:
            anchor_sprite = sprite
            anchor_bbox = b

    assert anchor_sprite is not None and anchor_bbox is not None, "missing teal anchor"
    for color, sprite in sprites.items():
        assert sprite.shape == anchor_sprite.shape, "all sprites must have same bbox size"
        assert np.array_equal(sprite, anchor_sprite), "all sprites must have same mask"

    top = anchor_bbox[0]
    out = np.full_like(g, bg)
    for color in colors:
        _r0, c0, _r1, _c1 = bboxes[color]
        sprite = sprites[color]
        patch = np.full(sprite.shape, bg, dtype=int)
        patch[sprite] = color

        target = out[top : top + sprite.shape[0], c0 : c0 + sprite.shape[1]]
        assert target.shape == sprite.shape, "aligned sprite must fit in grid"
        assert np.all(target[sprite] == bg), "aligned sprites must not overlap"
        paste(out, patch, top, c0, transparent=bg)

    return out

# --------------- generator (o5x) ---------------

# Canvas/object ranges are inclusive.
GEN_BG = Color.BLACK
GRID_ROWS = (13, 17)
GRID_COLS = (23, 30)
SPRITE_COUNT = (3, 5)

# Shared sprite geometry/density.
SPRITE_H = (4, 5)
SPRITE_W = (4, 6)
SPRITE_DENSITY = (0.60, 0.80)

# Placement and retry limits.
VERTICAL_SPREAD = (6, 10)
ANCHOR_COLOR = Color.TEAL
NON_ANCHOR_COLORS = [c for c in Color.NOT_BLACK if c != ANCHOR_COLOR]
GEN_ATTEMPTS = 300

def _gen_randint(bounds: tuple[int, int]) -> int:
    return int(np.random.randint(bounds[0], bounds[1] + 1))

def _gen_orbit_cells(w: int, r: int, c: int) -> list[tuple[int, int]]:
    if w % 2 == 1 and c == w // 2:
        return [(r, c)]
    return [(r, c), (r, w - 1 - c)]

def _gen_orbit_delta(mask: np.ndarray, r: int, c: int) -> int:
    _h, w = mask.shape
    return sum(not bool(mask[rr, cc]) for rr, cc in _gen_orbit_cells(w, r, c))

def _gen_orbit_adjacent(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    cells = set(_gen_orbit_cells(w, r, c))
    for rr, cc in cells:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and mask[nr, nc]:
                return True
    return False

def _gen_add_orbit(mask: np.ndarray, r: int, c: int) -> int:
    before = int(mask.sum())
    _h, w = mask.shape
    for rr, cc in _gen_orbit_cells(w, r, c):
        mask[rr, cc] = True
    return int(mask.sum()) - before

def _gen_fill_mask(mask: np.ndarray, target_area: int) -> np.ndarray | None:
    """Grow a connected LR-symmetric mask to the chosen exact area."""
    h, w = mask.shape
    area = int(mask.sum())
    while area < target_area:
        remaining = target_area - area
        candidates = []
        for r in range(h):
            for c in range((w + 1) // 2):
                delta = _gen_orbit_delta(mask, r, c)
                if 0 < delta <= remaining and _gen_orbit_adjacent(mask, r, c):
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = candidates[int(np.random.randint(len(candidates)))]
        area += _gen_add_orbit(mask, r, c)
    return mask

def _gen_lr_symmetric_mask(h: int, w: int) -> np.ndarray | None:
    min_density, max_density = SPRITE_DENSITY
    min_area = int(np.ceil(min_density * h * w))
    max_area = int(np.floor(max_density * h * w))
    areas = [a for a in range(min_area, max_area + 1) if w % 2 == 1 or a % 2 == 0]
    if not areas:
        return None

    target_area = int(np.random.choice(areas))
    bridge_r = int(np.random.randint(h))
    path_cols = list(np.random.permutation((w + 1) // 2))

    for path_c in path_cols:
        mask = np.zeros((h, w), dtype=bool)
        mask[bridge_r, :] = True
        for r in range(h):
            _gen_add_orbit(mask, r, int(path_c))
        if int(mask.sum()) > target_area:
            continue

        filled = _gen_fill_mask(mask, target_area)
        if filled is not None:
            return filled
    return None

def _gen_sprite_size(cols: int, count: int) -> tuple[int, int] | None:
    possible_widths = [w for w in range(SPRITE_W[0], SPRITE_W[1] + 1) if count * w <= cols]
    if not possible_widths:
        return None
    sprite_h = _gen_randint(SPRITE_H)
    sprite_w = int(np.random.choice(possible_widths))
    return sprite_h, sprite_w

def _gen_spaced_lefts(cols: int, count: int, sprite_w: int) -> list[int]:
    slack = cols - count * sprite_w
    extras = np.random.multinomial(slack, np.full(count + 1, 1 / (count + 1)))
    lefts = []
    c = int(extras[0])
    for i in range(count):
        lefts.append(c)
        c += sprite_w + int(extras[i + 1])
    return lefts

def _gen_top_rows(rows: int, sprite_h: int, count: int) -> tuple[int, list[int]] | None:
    """Choose the teal anchor row and non-anchor rows with exact spread."""
    spread_min, spread_max = VERTICAL_SPREAD
    max_top = rows - sprite_h
    max_spread = min(spread_max, max_top)
    if max_spread < spread_min:
        return None

    spread = int(np.random.randint(spread_min, max_spread + 1))
    lo = int(np.random.randint(0, max_top - spread + 1))
    hi = lo + spread
    positions = list(range(lo, hi + 1))
    anchor_top = int(np.random.choice(positions))

    required = [p for p in (lo, hi) if p != anchor_top]
    available = [p for p in positions if p != anchor_top and p not in required]
    need = count - 1 - len(required)
    if need < 0 or need > len(available):
        return None

    extra = list(np.random.choice(available, size=need, replace=False)) if need else []
    displaced = [int(p) for p in required + extra]
    displaced = [displaced[i] for i in np.random.permutation(len(displaced))]
    return anchor_top, displaced

def _gen_colors(count: int) -> list[int]:
    colors = [ANCHOR_COLOR] + [
        int(c) for c in np.random.choice(NON_ANCHOR_COLORS, size=count - 1, replace=False)
    ]
    return [colors[i] for i in np.random.permutation(count)]

def _gen_render_grids(rows, cols, mask, colors, lefts, anchor_top, displaced_tops):
    """Render the aligned output and vertically displaced input."""
    sprite_h, sprite_w = mask.shape
    top_by_color = {ANCHOR_COLOR: anchor_top}
    for color, top in zip([c for c in colors if c != ANCHOR_COLOR], displaced_tops):
        top_by_color[int(color)] = int(top)

    out = np.full((rows, cols), GEN_BG, dtype=int)
    inp = np.full((rows, cols), GEN_BG, dtype=int)
    for color, left in zip(colors, lefts):
        patch = np.full((sprite_h, sprite_w), GEN_BG, dtype=int)
        patch[mask] = int(color)
        paste(out, patch, anchor_top, int(left), transparent=GEN_BG)
        paste(inp, patch, top_by_color[int(color)], int(left), transparent=GEN_BG)
    return inp, out

def _gen_attempt() -> np.ndarray | None:
    rows = _gen_randint(GRID_ROWS)
    cols = _gen_randint(GRID_COLS)
    count = _gen_randint(SPRITE_COUNT)

    sprite_size = _gen_sprite_size(cols, count)
    if sprite_size is None:
        return None
    sprite_h, sprite_w = sprite_size

    mask = _gen_lr_symmetric_mask(sprite_h, sprite_w)
    if mask is None:
        return None

    lefts = _gen_spaced_lefts(cols, count, sprite_w)
    top_choice = _gen_top_rows(rows, sprite_h, count)
    if top_choice is None:
        return None
    anchor_top, displaced_tops = top_choice

    colors = _gen_colors(count)
    inp, out = _gen_render_grids(rows, cols, mask, colors, lefts, anchor_top, displaced_tops)

    if np.array_equal(main(inp), out):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e41c6fd3 input")
