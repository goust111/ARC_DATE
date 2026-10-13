from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

MAX_ATTEMPTS = 500
PADDING = 2


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _boundary_reachable_black(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    black = grid == BLACK
    reachable = np.zeros_like(black, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return reachable


def _interiors_by_component(grid: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    blue_mask = grid == BLUE
    blue_comps = _components(blue_mask)
    comp_id = np.full(grid.shape, -1, dtype=int)
    for idx, coords in enumerate(blue_comps):
        comp_id[coords[:, 0], coords[:, 1]] = idx

    enclosed_black = (grid == BLACK) & ~_boundary_reachable_black(grid)
    enclosed_comps = _components(enclosed_black)
    interiors = [np.zeros_like(blue_mask, dtype=bool) for _ in blue_comps]

    for region in enclosed_comps:
        adjacent = set()
        for r, c in region:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    cid = int(comp_id[nr, nc])
                    if cid >= 0:
                        adjacent.add(cid)
        if len(adjacent) == 1:
            cid = next(iter(adjacent))
            interiors[cid][region[:, 0], region[:, 1]] = True
    return blue_comps, [np.argwhere(mask) for mask in interiors]


def _paint_tiling(
    out: np.ndarray,
    region_coords: np.ndarray,
    template_mask: np.ndarray,
) -> None:
    if region_coords.size == 0:
        return
    region_mask = np.zeros(out.shape, dtype=bool)
    region_mask[region_coords[:, 0], region_coords[:, 1]] = True
    r0, c0, r1, c1 = _bbox(region_mask)
    th, tw = template_mask.shape
    for base_r in range(r0, r1 + 1, th):
        for base_c in range(c0, c1 + 1, tw):
            for tr in range(th):
                for tc in range(tw):
                    if not template_mask[tr, tc]:
                        continue
                    rr = base_r + tr
                    cc = base_c + tc
                    if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                        if region_mask[rr, cc] and out[rr, cc] == BLACK:
                            out[rr, cc] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the red sprite and tile it inside the interiors of blue closed contours.

    Input structure:
    - Black background with several blue one-cell-wide 4-connected contour components.
    - Some blue components are closed loops that trap black interior cells; others have gaps and stay open.
    - One separate red sprite provides the tiling template after cropping to its tight bounding box.

    Rule:
    - Crop the red sprite to its tight bounding box and treat the red support inside that box as the template.
    - Find each blue 4-connected component and identify the black cells enclosed by that component, if any.
    - Erase the original red sprite, then for every closed blue component tile the cropped template over the enclosed region,
      anchored at the enclosed region's bounding-box top-left and stepping by the template height and width; only template pixels
      landing on enclosed black cells are painted red. Open blue components remain unchanged.

    Key insight:
    - The red object does not mark where to paint, only what pattern to repeat; the blue topology decides which regions are eligible,
      and the tiling origin is tied to each enclosed interior rather than to the whole canvas.
    """
    g = np.asarray(grid, dtype=int)
    red_mask = g == RED
    red_box = _bbox(red_mask)
    assert red_box is not None, "expected one red sprite"
    template = _crop(np.where(red_mask, RED, BLACK), red_box)
    template_mask = template == RED

    out = g.copy()
    out[red_mask] = BLACK

    _blue_comps, interiors = _interiors_by_component(g)
    for region in interiors:
        if region.size == 0:
            continue
        _paint_tiling(out, region, template_mask)
    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _coords_to_array(coords: list[tuple[int, int]], color: int) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    arr = np.full((h, w), BLACK, dtype=int)
    for r, c in coords:
        arr[r, c] = color
    return arr


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    for _ in range(random.randrange(4)):
        h = max(r for r, _ in pts) + 1
        pts = {(c, h - 1 - r) for r, c in pts}
    if random.random() < 0.5:
        w = max(c for _, c in pts) + 1
        pts = {(r, w - 1 - c) for r, c in pts}
    if random.random() < 0.5:
        h = max(r for r, _ in pts) + 1
        pts = {(h - 1 - r, c) for r, c in pts}
    return _normalize_coords(pts)


def _sprite_shapes() -> list[list[tuple[int, int]]]:
    base = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
    ]
    shapes = []
    seen = set()
    variants = (
        lambda pts: pts,
        lambda pts: {(r, -c) for r, c in pts},
        lambda pts: {(-r, c) for r, c in pts},
        lambda pts: {(-r, -c) for r, c in pts},
        lambda pts: {(c, r) for r, c in pts},
        lambda pts: {(c, -r) for r, c in pts},
        lambda pts: {(-c, r) for r, c in pts},
        lambda pts: {(-c, -r) for r, c in pts},
    )
    for coords in base:
        pts = set(coords)
        for transform in variants:
            t = tuple(_normalize_coords(transform(pts)))
            box = (max(r for r, _ in t) + 1, max(c for _, c in t) + 1)
            if len(t) in (3, 4) and box in {(2, 2), (2, 3), (3, 2)} and box[0] <= 3 and box[1] <= 3:
                if np.count_nonzero(_coords_to_array(list(t), RED) == BLACK) > 0 and t not in seen:
                    seen.add(t)
                    shapes.append(list(t))
    return shapes


SPRITE_SHAPES = _sprite_shapes()


def _rectangle_frame(inner_h: int, inner_w: int) -> np.ndarray:
    arr = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    arr[0, :] = BLUE
    arr[-1, :] = BLUE
    arr[:, 0] = BLUE
    arr[:, -1] = BLUE
    return arr


def _notched_loop(outer_h: int, outer_w: int) -> np.ndarray:
    notch_top = random.randint(2, outer_h - 5)
    notch_bottom = random.randint(notch_top + 2, outer_h - 3)
    indent_col = random.randint(2, outer_w - 3)
    coords: set[tuple[int, int]] = set()
    for c in range(outer_w):
        coords.add((0, c))
        coords.add((outer_h - 1, c))
    for r in range(outer_h):
        coords.add((r, 0))
    for r in range(1, notch_top + 1):
        coords.add((r, outer_w - 1))
    for c in range(indent_col, outer_w):
        coords.add((notch_top, c))
        coords.add((notch_bottom, c))
    for r in range(notch_top, notch_bottom + 1):
        coords.add((r, indent_col))
    for r in range(notch_bottom, outer_h):
        coords.add((r, outer_w - 1))
    return _coords_to_array(_transform_coords(sorted(coords)), BLUE)


def _remove_gap(shape: np.ndarray) -> np.ndarray:
    coords = np.argwhere(shape == BLUE)
    candidates = []
    for r, c in coords:
        neighbors = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < shape.shape[0] and 0 <= nc < shape.shape[1] and shape[nr, nc] == BLUE:
                neighbors += 1
        if neighbors == 2:
            candidates.append((int(r), int(c)))
    gap_r, gap_c = random.choice(candidates)
    out = shape.copy()
    out[gap_r, gap_c] = BLACK
    return out


def _shape_with_min_interior(template_box: tuple[int, int], closed: bool, irregular: bool) -> np.ndarray | None:
    min_h = max(4, 2 * template_box[0])
    min_w = max(4, 2 * template_box[1])
    for _ in range(80):
        if irregular:
            outer_h = random.randint(max(7, min_h + 2), min(12, min_h + 5))
            outer_w = random.randint(max(6, min_w + 2), min(12, min_w + 5))
            shape = _notched_loop(outer_h, outer_w)
        else:
            inner_h = random.randint(min_h, min(9, min_h + 4))
            inner_w = random.randint(min_w, min(10, min_w + 4))
            shape = _rectangle_frame(inner_h, inner_w)
        if not closed:
            shape = _remove_gap(shape)
        probe = shape.copy()
        blue_comps, interiors = _interiors_by_component(probe)
        assert len(blue_comps) == 1
        region = interiors[0]
        if closed:
            if region.size == 0:
                continue
            mask = np.zeros(shape.shape, dtype=bool)
            mask[region[:, 0], region[:, 1]] = True
            box = _bbox(mask)
            if box is None:
                continue
            r0, c0, r1, c1 = box
            if (r1 - r0 + 1) < min_h or (c1 - c0 + 1) < min_w:
                continue
        else:
            if region.size != 0:
                continue
        return shape
    return None


def _can_place(occupied: np.ndarray, shape: np.ndarray, top: int, left: int, pad: int) -> bool:
    h, w = shape.shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _place_random(occupied: np.ndarray, shape: np.ndarray, pad: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = shape.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        if _can_place(occupied, shape, r, c, pad):
            occupied[r : r + h, c : c + w] |= shape != BLACK
            return r, c
    return None


def _place_canonical(occupied: np.ndarray, shape: np.ndarray, pad: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    h, w = shape.shape
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            if _can_place(occupied, shape, r, c, pad):
                occupied[r : r + h, c : c + w] |= shape != BLACK
                return r, c
    return None


def _paste(dst: np.ndarray, shape: np.ndarray, top: int, left: int) -> None:
    h, w = shape.shape
    mask = shape != BLACK
    dst[top : top + h, left : left + w][mask] = shape[mask]


def _naive_bbox_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    red_box = _bbox(g == RED)
    assert red_box is not None
    template = _crop(np.where(g == RED, RED, BLACK), red_box) == RED
    out = g.copy()
    out[g == RED] = BLACK
    comps = _components(g == BLUE)
    for comp in comps:
        comp_grid = np.full(g.shape, BLACK, dtype=int)
        comp_grid[comp[:, 0], comp[:, 1]] = BLUE
        if _interiors_by_component(comp_grid)[1][0].size == 0:
            continue
        r0, c0 = comp.min(axis=0)
        r1, c1 = comp.max(axis=0)
        region_mask = np.zeros(g.shape, dtype=bool)
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            region_mask[r0 + 1 : r1, c0 + 1 : c1] = True
        region = np.argwhere(region_mask)
        _paint_tiling(out, region, template)
    return out


def _analyze_example(grid: np.ndarray) -> dict[str, object] | None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return None
    red_comps = _components(grid == RED)
    if len(red_comps) != 1:
        return None
    red_coords = red_comps[0]
    if red_coords.shape[0] not in (3, 4):
        return None
    red_mask = np.zeros(grid.shape, dtype=bool)
    red_mask[red_coords[:, 0], red_coords[:, 1]] = True
    red_box = _bbox(red_mask)
    assert red_box is not None
    sprite = _crop(np.where(red_mask, RED, BLACK), red_box)
    if sprite.shape not in {(2, 2), (2, 3), (3, 2)}:
        return None
    if not np.any(sprite == BLACK):
        return None

    blue_comps, interiors = _interiors_by_component(grid)
    if len(blue_comps) < 3:
        return None
    closed = [i for i, reg in enumerate(interiors) if reg.size > 0]
    open_ = [i for i, reg in enumerate(interiors) if reg.size == 0]
    if len(closed) < 2 or len(open_) < 1:
        return None

    irregular_closed = False
    multi_fill = 0
    out = main(grid)
    out_red = out == RED
    for idx in closed:
        mask = np.zeros(grid.shape, dtype=bool)
        region = interiors[idx]
        mask[region[:, 0], region[:, 1]] = True
        box = _bbox(mask)
        if box is None:
            return None
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 4:
            return None
        bbox_mask = np.zeros(grid.shape, dtype=bool)
        bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        if np.any(mask ^ bbox_mask):
            irregular_closed = True
        if np.any(out_red & mask):
            multi_fill += 1
    if not irregular_closed:
        return None
    if multi_fill < 2:
        return None
    if np.array_equal(_naive_bbox_fill(grid), out):
        return None
    return {"out": out, "closed": len(closed), "open": len(open_)}


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(21, 29)
    W = random.randint(21, 29)
    grid = np.full((H, W), BLACK, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    sprite_coords = random.choice(SPRITE_SHAPES)
    sprite = _coords_to_array(sprite_coords, RED)
    template_box = sprite.shape

    closed_count = random.randint(2, 3)
    open_count = random.randint(1, 2)
    blue_shapes: list[np.ndarray] = []

    first_irregular = _shape_with_min_interior(template_box, closed=True, irregular=True)
    if first_irregular is None:
        return None
    blue_shapes.append(first_irregular)

    for _ in range(closed_count - 1):
        shape = _shape_with_min_interior(
            template_box,
            closed=True,
            irregular=random.random() < 0.45,
        )
        if shape is None:
            return None
        blue_shapes.append(shape)

    for _ in range(open_count):
        shape = _shape_with_min_interior(
            template_box,
            closed=False,
            irregular=random.random() < 0.5,
        )
        if shape is None:
            return None
        blue_shapes.append(shape)

    blue_shapes.sort(key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True)
    for shape in blue_shapes:
        pos = _place_random(occupied, shape, PADDING)
        if pos is None:
            return None
        _paste(grid, shape, pos[0], pos[1])

    sprite_pos = _place_canonical(occupied, sprite, PADDING + 1)
    if sprite_pos is None:
        return None
    _paste(grid, sprite, sprite_pos[0], sprite_pos[1])

    return grid if _analyze_example(grid) is not None else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
