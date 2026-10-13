import random
from collections import deque
from math import gcd

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TARGET_ORDERS = (1, 2, 4)
LEGEND_SPAN = 7
RESERVED_BOTTOM_ROWS = 2
GEN_RETRIES = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
SHAPE_SIZES = (5, 7, 9)
PLACEMENT_TRIALS = 250
SHAPE_RETRIES = 120


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each ring center by looking up the ring's rotational symmetry order in the bottom legend.

    Input structure:
    - The background is black.
    - Several one-pixel-wide colored closed loops appear above the bottom legend, and each loop contains one grey pixel at its geometric center.
    - The bottommost non-background row is a legend row whose occupied slots encode the colors for symmetry orders 1, 2, and 4.

    Rule:
    - Ignore the legend row and treat each remaining non-background, non-grey connected loop as one object.
    - For each loop, use its central grey pixel as the rotation center and test whether the loop's pixel set is invariant under 90-degree and 180-degree rotation.
    - Map the highest valid order k in {1, 2, 4} to the legend color stored in slot k, then replace the loop's grey center with that color.

    Key insight:
    - The loop color is irrelevant; the needed attribute is the loop's rotational symmetry order, and the bottom legend converts that geometric property into the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend_row = _find_legend_row(g, bg)
    legend = _parse_legend(g[legend_row], bg)

    grey_points = {
        (int(r), int(c))
        for r, c in np.argwhere(g == GREY)
        if int(r) < legend_row
    }
    assert grey_points

    ring_mask = (g != bg) & (g != GREY)
    ring_mask[legend_row, :] = False
    components = _components_mask(ring_mask, connectivity=8)
    assert components

    out = g.copy()
    seen_centers = set()
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0
        assert center in grey_points
        assert r0 < center[0] < r1 and c0 < center[1] < c1
        seen_centers.add(center)
        order = _rotation_order(coords, center)
        out[center] = legend[order]

    assert seen_centers == grey_points
    return out


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_legend_row(grid: np.ndarray, bg: int) -> int:
    rows = np.flatnonzero(np.any(grid != bg, axis=1))
    assert len(rows) >= 2
    legend_row = int(rows[-1])
    assert legend_row == grid.shape[0] - 1
    assert not np.any(grid[legend_row] == GREY)
    return legend_row


def _parse_legend(row: np.ndarray, bg: int) -> dict[int, int]:
    positions = [int(c) for c in np.flatnonzero(row != bg)]
    assert len(positions) == 3
    base = positions[0]
    step = gcd(positions[1] - positions[0], positions[2] - positions[1])
    assert step == 2
    slot_by_col = {int(c): 1 + (int(c) - base) // step for c in positions}
    assert sorted(slot_by_col.values()) == [1, 2, 4]
    for c in range(base, positions[-1] + 1):
        if c not in slot_by_col:
            assert int(row[c]) == bg
    return {slot: int(row[c]) for c, slot in slot_by_col.items()}


def _components_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotation_order(coords: list[tuple[int, int]], center: tuple[int, int]) -> int:
    cr, cc = center
    rel = {(r - cr, c - cc) for r, c in coords}
    rot90 = {(-dc, dr) for dr, dc in rel}
    if rot90 == rel:
        return 4
    rot180 = {(-dr, -dc) for dr, dc in rel}
    if rot180 == rel:
        return 2
    return 1


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(2, 5)
    legend_colors = rng.sample(PALETTE, 3)
    ring_palette = [color for color in PALETTE if color not in legend_colors]
    ring_colors = rng.sample(ring_palette, object_count)
    orders = _sample_orders(object_count, rng)

    sprites = []
    for color, order in zip(ring_colors, orders):
        mask = _make_ring_mask(order, rng)
        sprite = np.full(mask.shape, BG, dtype=int)
        sprite[mask] = color
        sprite[mask.shape[0] // 2, mask.shape[1] // 2] = GREY
        sprites.append(_random_transform(sprite, rng))

    required_area = sum((sprite.shape[0] + 2) * (sprite.shape[1] + 2) for sprite in sprites)
    max_h = max(sprite.shape[0] for sprite in sprites)
    max_w = max(sprite.shape[1] for sprite in sprites)

    grid_w = None
    usable_h = None
    min_h = max(12, max_h + 1)
    min_w = max(10, max(max_w + 1, LEGEND_SPAN))
    for _ in range(60):
        candidate_h = rng.randint(min_h, 28)
        candidate_w = rng.randint(min_w, 30)
        if candidate_h * candidate_w >= required_area:
            usable_h = candidate_h
            grid_w = candidate_w
            break
    if usable_h is None or grid_w is None:
        return None

    grid = np.full((usable_h + RESERVED_BOTTOM_ROWS, grid_w), BG, dtype=int)
    if not _place_sprites(grid[:usable_h], sprites, rng):
        return None

    legend_start = rng.randint(0, grid_w - LEGEND_SPAN)
    grid[-1, legend_start] = legend_colors[0]
    grid[-1, legend_start + 2] = legend_colors[1]
    grid[-1, legend_start + 6] = legend_colors[2]

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def _sample_orders(object_count: int, rng: random.Random) -> list[int]:
    if object_count == 2:
        orders = list(rng.sample(TARGET_ORDERS, 2))
        rng.shuffle(orders)
        return orders

    orders = list(TARGET_ORDERS)
    while len(orders) < object_count:
        orders.append(rng.choice(TARGET_ORDERS))
    rng.shuffle(orders)
    return orders


def _place_sprites(canvas: np.ndarray, sprites: list[np.ndarray], rng: random.Random) -> bool:
    placements: list[tuple[int, int, int, int]] = []
    for sprite in sorted(sprites, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        sh, sw = sprite.shape
        placed = False
        for _ in range(PLACEMENT_TRIALS):
            r = rng.randint(0, canvas.shape[0] - sh)
            c = rng.randint(0, canvas.shape[1] - sw)
            bbox = (r, c, r + sh - 1, c + sw - 1)
            if any(_boxes_too_close(bbox, other) for other in placements):
                continue
            patch = canvas[r : r + sh, c : c + sw]
            mask = sprite != BG
            if np.any(patch[mask] != BG):
                continue
            patch[mask] = sprite[mask]
            placements.append(bbox)
            placed = True
            break
        if not placed:
            return False
    return True


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + 2 <= br0 or br1 + 2 <= ar0 or ac1 + 2 <= bc0 or bc1 + 2 <= ac0
    return not separated


def _make_ring_mask(target_order: int, rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        if target_order == 4:
            size = rng.choice(SHAPE_SIZES)
            notch = rng.randint(0, (size - 3) // 2)
            solid = _solid_from_notches(size, size, notch, notch, notch, notch)
        elif target_order == 2:
            h = rng.choice(SHAPE_SIZES)
            w = rng.choice(SHAPE_SIZES)
            notch_cap = min((h - 3) // 2, (w - 3) // 2)
            tl = rng.randint(0, notch_cap)
            tr = rng.randint(0, notch_cap)
            solid = _solid_from_notches(h, w, tl, tr, tr, tl)
        else:
            h = rng.choice(SHAPE_SIZES)
            w = rng.choice(SHAPE_SIZES)
            notch_cap = min((h - 3) // 2, (w - 3) // 2)
            tl = rng.randint(0, notch_cap)
            tr = rng.randint(0, notch_cap)
            bl = rng.randint(0, notch_cap)
            br = rng.randint(0, notch_cap)
            solid = _solid_from_notches(h, w, tl, tr, bl, br)

        center = (solid.shape[0] // 2, solid.shape[1] // 2)
        if not _has_open_center(solid, center):
            continue

        mask = _boundary_from_solid(solid)
        if _rotation_order(mask_nonzero(mask), center) != target_order:
            continue
        if len(_components_mask(mask, connectivity=8)) != 1:
            continue
        if not _center_is_enclosed(mask, center):
            continue
        return mask

    raise RuntimeError(f"failed to generate order-{target_order} ring")


def _solid_from_notches(
    height: int,
    width: int,
    top_left: int,
    top_right: int,
    bottom_left: int,
    bottom_right: int,
) -> np.ndarray:
    solid = np.ones((height, width), dtype=bool)
    if top_left:
        solid[:top_left, :top_left] = False
    if top_right:
        solid[:top_right, width - top_right :] = False
    if bottom_left:
        solid[height - bottom_left :, :bottom_left] = False
    if bottom_right:
        solid[height - bottom_right :, width - bottom_right :] = False
    return solid


def _has_open_center(solid: np.ndarray, center: tuple[int, int]) -> bool:
    r, c = center
    if not solid[r, c]:
        return False
    if r == 0 or r == solid.shape[0] - 1 or c == 0 or c == solid.shape[1] - 1:
        return False
    return all(solid[r + dr, c + dc] for dr, dc in NEIGHBORS4)


def _boundary_from_solid(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    boundary = np.zeros_like(solid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not solid[rr, cc]:
                    boundary[r, c] = True
                    break
    return boundary


def mask_nonzero(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _center_is_enclosed(mask: np.ndarray, center: tuple[int, int]) -> bool:
    padded = np.pad(mask, 1, constant_values=False)
    seen = np.zeros_like(padded, dtype=bool)
    queue = deque([(0, 0)])
    seen[0, 0] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < padded.shape[0] and 0 <= cc < padded.shape[1]:
                if padded[rr, cc] or seen[rr, cc]:
                    continue
                seen[rr, cc] = True
                queue.append((rr, cc))

    cr, cc = center
    return not seen[cr + 1, cc + 1]


def _random_transform(sprite: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(sprite, copy=True)
    out = np.rot90(out, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, copy=True)
