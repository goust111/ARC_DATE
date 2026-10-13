from __future__ import annotations

import numpy as np

from arc_utils import components_mask, get_bg, paste
import random


def _component_order(coords: np.ndarray, direction: int) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == -1:
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    assert direction == 1, "horizontal slide direction must be left or right"
    return (-int(cols.max()), int(rows.min()), -int(cols.min()), int(rows.max()))


def _settled_components(g: np.ndarray, color: int, side_col: int, direction: int) -> list[np.ndarray]:
    h, w = g.shape
    movable = (g == color)
    movable[:, side_col] = False

    comps = components_mask(movable, connectivity=8)
    assert comps, "expected at least one object away from the matching wall"
    comps.sort(key=lambda coords: _component_order(coords, direction))

    settled = np.zeros((h, w), dtype=bool)
    settled[:, side_col] = True
    out_coords = []

    for coords in comps:
        shift = 0
        while True:
            candidate = coords + np.array([0, shift + direction])
            cols = candidate[:, 1]
            if int(cols.min()) < 0 or int(cols.max()) >= w:
                break
            if np.any(settled[candidate[:, 0], candidate[:, 1]]):
                break
            shift += direction

        placed = coords + np.array([0, shift])
        assert not np.any(settled[placed[:, 0], placed[:, 1]]), "settled objects may touch but not overlap"
        settled[placed[:, 0], placed[:, 1]] = True
        out_coords.append(placed)

    return out_coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each inner colored object horizontally into the same-colored vertical wall until it touches an obstacle.

    Input structure
    - The background is the dominant color.
    - The only foreground colors are two non-background colors.
    - One foreground color fills the left column and the other fills the right column.
    - Extra 8-connected objects of each wall color appear away from that color's wall.

    Rule
    - Clear the extra objects from their original positions.
    - For each wall color, process its 8-connected extra objects from nearest to farthest from the matching wall.
    - Slide each object horizontally toward its matching wall as far as possible without overlapping the wall or an already settled object of the same color.
    - Keep the two vertical wall columns fixed.

    Key insight
    - The side columns act like color-specific gravity walls: same-colored objects compact sideways against their own wall while preserving each object's shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2, "expected a non-empty 2D grid with two side columns"

    bg = get_bg(g)
    w = g.shape[1]
    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])

    assert left_color != right_color, "left and right walls must have different colors"
    assert bg not in {left_color, right_color}, "walls must be foreground colors"
    assert np.all(g[:, 0] == left_color), "left wall must fill the full left column"
    assert np.all(g[:, w - 1] == right_color), "right wall must fill the full right column"
    assert set(map(int, np.unique(g))) == {int(bg), left_color, right_color}, "expected exactly background plus two wall colors"

    out = np.full_like(g, int(bg))
    out[:, 0] = left_color
    out[:, w - 1] = right_color

    placements: list[tuple[int, np.ndarray]] = []
    for color, side_col, direction in ((left_color, 0, -1), (right_color, w - 1, 1)):
        for coords in _settled_components(g, color, side_col, direction):
            placements.append((color, coords))

    occupied = (out != bg)
    for color, coords in placements:
        assert not np.any(occupied[coords[:, 0], coords[:, 1]]), "different colors must not settle on the same pixel"
        out[coords[:, 0], coords[:, 1]] = color
        occupied[coords[:, 0], coords[:, 1]] = True

    assert int(np.sum(out == left_color)) == int(np.sum(g == left_color)), "left color count must be preserved"
    assert int(np.sum(out == right_color)) == int(np.sum(g == right_color)), "right color count must be preserved"
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GRID_HEIGHT_BOUNDS = (9, 10)
GRID_WIDTH_BOUNDS = (9, 10)
BACKGROUND_COLOR = 7
WALL_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
# Object counts and pixel budgets.
COMPONENT_COUNT_BOUNDS = (2, 3)
INNER_FOREGROUND_BOUNDS = (15, 23)
SPRITE_PIXEL_BOUNDS = (1, 7)
# Sprite geometry limits.
SPRITE_HEIGHT_BOUNDS = (1, 5)
SPRITE_WIDTH_BOUNDS = (1, 3)
MIN_SPRITE_DENSITY = 0.56
# Input scatter and retry limits.
SCATTER_OFFSET_BOUNDS = (1, 5)
GENERATION_ATTEMPTS = 300

Placement = tuple[int, np.ndarray, int, int, bool]


def _gen_bounded_composition(total: int, parts: int, lo: int, hi: int) -> list[int]:
    values = []
    remaining = int(total)
    for i in range(int(parts)):
        slots = int(parts) - i - 1
        low = max(int(lo), remaining - int(hi) * slots)
        high = min(int(hi), remaining - int(lo) * slots)
        value = random.randint(low, high)
        values.append(value)
        remaining -= value
    random.shuffle(values)
    return values


def _gen_sprite_mask(size: int) -> np.ndarray:
    size = int(size)
    dims = [
        (h, w)
        for h in range(SPRITE_HEIGHT_BOUNDS[0], SPRITE_HEIGHT_BOUNDS[1] + 1)
        for w in range(SPRITE_WIDTH_BOUNDS[0], SPRITE_WIDTH_BOUNDS[1] + 1)
        if max(h, w) <= size <= h * w and size / (h * w) >= MIN_SPRITE_DENSITY
    ]
    h, w = random.choice(dims)

    coords: set[tuple[int, int]] = set()
    length = max(h, w)
    if length == 1:
        coords.add((0, 0))
    else:
        for i in range(length):
            r = (i * (h - 1) + (length - 1) // 2) // (length - 1)
            c = (i * (w - 1) + (length - 1) // 2) // (length - 1)
            coords.add((int(r), int(c)))

    while len(coords) < size:
        frontier = []
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.append((nr, nc))
        coords.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    if random.choice([False, True]):
        mask = np.flipud(mask)
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    return mask.copy()


def _gen_mask_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(r0), int(c0)])


def _gen_base_grid(shape: tuple[int, int], bg: int, left_color: int, right_color: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    grid[:, 0] = int(left_color)
    grid[:, shape[1] - 1] = int(right_color)
    return grid


def _gen_place_output_side(
    out: np.ndarray, color: int, masks: list[np.ndarray], *, left_side: bool, bg: int
) -> list[Placement]:
    h, w = out.shape
    placements = []
    for mask in sorted(masks, key=lambda m: int(m.sum()), reverse=True):
        mh, mw = mask.shape
        c0 = 1 if left_side else w - 1 - mw
        rows = list(range(0, h - mh + 1))
        random.shuffle(rows)
        for r0 in rows:
            coords = _gen_mask_coords(mask, r0, c0)
            if np.any(out[coords[:, 0], coords[:, 1]] != bg):
                continue
            patch = np.where(mask, int(color), int(bg))
            paste(out, patch, r0, c0, transparent=bg)
            placements.append((int(color), mask, int(r0), int(c0), bool(left_side)))
            break
        else:
            return []
    return placements


def _gen_touches_existing(coords: np.ndarray, existing: list[np.ndarray]) -> bool:
    for other in existing:
        dr = np.abs(coords[:, None, 0] - other[None, :, 0])
        dc = np.abs(coords[:, None, 1] - other[None, :, 1])
        if bool(np.any((dr <= 1) & (dc <= 1))):
            return True
    return False


def _gen_scatter_input(
    shape: tuple[int, int], bg: int, left_color: int, right_color: int, placements: list[Placement]
) -> np.ndarray | None:
    h, w = shape
    inp = _gen_base_grid((h, w), bg, left_color, right_color)
    by_color: dict[int, list[np.ndarray]] = {int(left_color): [], int(right_color): []}

    shuffled = list(placements)
    random.shuffle(shuffled)
    for color, mask, r0, c_out, left_side in shuffled:
        mh, mw = mask.shape
        max_offset = (w - 2 - (c_out + mw - 1)) if left_side else (c_out - 1)
        offsets = list(range(SCATTER_OFFSET_BOUNDS[0], min(SCATTER_OFFSET_BOUNDS[1], int(max_offset)) + 1))
        random.shuffle(offsets)
        for offset in offsets:
            c_in = c_out + offset if left_side else c_out - offset
            coords = _gen_mask_coords(mask, r0, c_in)
            if np.any(inp[coords[:, 0], coords[:, 1]] != bg):
                continue
            if _gen_touches_existing(coords, by_color[int(color)]):
                continue
            patch = np.where(mask, int(color), int(bg))
            paste(inp, patch, r0, c_in, transparent=bg)
            by_color[int(color)].append(coords)
            break
        else:
            return None
    return inp


def _gen_output_layout(
    shape: tuple[int, int], bg: int, left_color: int, right_color: int,
    left_masks: list[np.ndarray], right_masks: list[np.ndarray],
) -> tuple[np.ndarray, list[Placement]] | None:
    out = _gen_base_grid(shape, bg, left_color, right_color)
    left_placements = _gen_place_output_side(out, left_color, left_masks, left_side=True, bg=bg)
    if len(left_placements) != len(left_masks):
        return None
    right_placements = _gen_place_output_side(out, right_color, right_masks, left_side=False, bg=bg)
    if len(right_placements) != len(right_masks):
        return None
    return out, left_placements + right_placements


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_BOUNDS)
    w = random.randint(*GRID_WIDTH_BOUNDS)
    bg = BACKGROUND_COLOR
    left_color, right_color = random.sample(WALL_COLORS, 2)

    left_count = random.randint(*COMPONENT_COUNT_BOUNDS)
    right_count = random.randint(*COMPONENT_COUNT_BOUNDS)
    total_inner = random.randint(*INNER_FOREGROUND_BOUNDS)
    sizes = _gen_bounded_composition(total_inner, left_count + right_count, *SPRITE_PIXEL_BOUNDS)
    left_masks = [_gen_sprite_mask(size) for size in sizes[:left_count]]
    right_masks = [_gen_sprite_mask(size) for size in sizes[left_count:]]

    layout = _gen_output_layout((h, w), bg, left_color, right_color, left_masks, right_masks)
    if layout is None:
        return None
    out, placements = layout

    inp = _gen_scatter_input((h, w), bg, left_color, right_color, placements)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f0100645 input")
