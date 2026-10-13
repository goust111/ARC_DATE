from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    raise ValueError(f"unsupported connectivity: {connectivity}")


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    nbrs = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in nbrs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for coords in _components(grid == FRAME, connectivity=4):
        if _is_frame_component(coords):
            frames.append(_bbox(coords))
    frames.sort(key=lambda box: (box[0], box[1]))
    return frames


def _layer_sprites(grid: np.ndarray, obj_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, np.ndarray]]:
    r0, c0, r1, c1 = obj_bbox
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = crop != BG
    colors = [int(color) for color in np.unique(crop[mask])]
    assert 2 <= len(colors) <= 4

    layers: list[tuple[int, int, int, np.ndarray]] = []
    for color in colors:
        pts = np.argwhere(crop == color)
        assert pts.size > 0
        lr0, lc0 = pts.min(axis=0)
        lr1, lc1 = pts.max(axis=0)
        sprite = np.full((int(lr1 - lr0 + 1), int(lc1 - lc0 + 1)), BG, dtype=int)
        rel = pts - np.array([lr0, lc0], dtype=int)
        sprite[rel[:, 0], rel[:, 1]] = color
        layers.append((int(len(pts)), int(lr0), int(lc0), sprite))

    layers.sort(key=lambda item: (-item[0], item[1], item[2], item[3].shape[0], item[3].shape[1]))
    return layers


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _find_frames(g)
    assert frames

    obj_mask = (g != BG) & (g != FRAME)
    objects = [_bbox(coords) for coords in _components(obj_mask, connectivity=8)]
    objects.sort(key=lambda box: (box[0], box[1]))
    assert len(objects) == len(frames)

    out = g.copy()
    for obj_bbox, frame_bbox in zip(objects, frames):
        fr0, fc0, fr1, fc1 = frame_bbox
        inner_r0, inner_c0 = fr0 + 1, fc0 + 1
        inner_r1, inner_c1 = fr1 - 1, fc1 - 1
        assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1
        inner_center_r = (inner_r0 + inner_r1) // 2
        inner_center_c = (inner_c0 + inner_c1) // 2

        out[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = BG
        for _area, _lr0, _lc0, sprite in _layer_sprites(g, obj_bbox):
            sh, sw = sprite.shape
            top = inner_center_r - sh // 2
            left = inner_center_c - sw // 2
            bottom = top + sh - 1
            right = left + sw - 1
            assert inner_r0 <= top <= bottom <= inner_r1
            assert inner_c0 <= left <= right <= inner_c1
            patch = out[top : top + sh, left : left + sw]
            mask = sprite != BG
            patch[mask] = sprite[mask]
            out[top : top + sh, left : left + sw] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restack each multicolor object's color layers inside its matching frame, ordered from largest layer to smallest.

    Input structure:
    - bg = black. The left side contains several multicolor 8-connected objects.
    - The right side contains the same number of empty grey one-pixel rectangular frames.
    - Objects and frames are paired by their top-to-bottom order.

    Rule:
    - Split each multicolor object into one layer per visible color, keeping that color's exact pixel shape.
    - Sort the layers by pixel count from largest to smallest.
    - For the matching grey frame, clear its interior to black, then place the layers there in that order.
    - Each layer is centered by aligning the center of its own bounding box with the center of the frame interior; later layers overwrite earlier ones.

    Key insight:
    - The visible mixed object is reinterpreted as separate single-color shapes, and the answer comes from sorting those shapes by area before re-centering and stacking them.
    """
    return _solve(np.asarray(grid, dtype=int))


# --------------- generator ---------------

GEN_RETRIES = 400
OBJECT_COUNT_OPTIONS = (2, 3)
LAYER_COUNT_OPTIONS = (2, 3, 4)
GRID_HEIGHT_RANGE = (14, 30)
ROW_GAP_RANGE = (1, 3)
MID_GAP_RANGE = (3, 5)
LAYER_AREA_POOL = {
    2: (4, 5, 6, 7, 8, 9),
    3: (3, 4, 5, 6, 7, 8),
    4: (3, 4, 5, 6, 7),
}


def _masked_neighbors(mask: np.ndarray, r: int, c: int) -> Iterable[tuple[int, int]]:
    h, w = mask.shape
    for dr, dc in _neighbors(8):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            yield nr, nc


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    if not (1 <= area <= h * w):
        return None
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        weights: list[int] = []
        for r, c in np.argwhere(mask):
            for dr, dc in _neighbors(8):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    cell = (nr, nc)
                    if cell not in frontier:
                        frontier.append(cell)
                        contact = sum(1 for _ in _masked_neighbors(mask, nr, nc))
                        weights.append(1 + contact)
        if not frontier:
            return None
        pick = random.choices(frontier, weights=weights, k=1)[0]
        mask[pick] = True

    cropped = _crop_mask(mask)
    if min(cropped.shape) < 2:
        return None
    return cropped


def _random_sprite(area: int) -> np.ndarray | None:
    shapes = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if area <= h * w and 2 * area >= h * w
    ]
    random.shuffle(shapes)
    for h, w in shapes:
        for _ in range(40):
            mask = _grow_connected_mask(h, w, area)
            if mask is None:
                continue
            if int(mask.sum()) != area:
                continue
            if min(mask.shape) < 2 or max(mask.shape) > 5:
                continue
            return mask
    return None


def _touches_union(union: np.ndarray, sprite: np.ndarray, top: int, left: int) -> int:
    count = 0
    for dr, dc in np.argwhere(sprite):
        rr = int(top + dr)
        cc = int(left + dc)
        for nr, nc in _masked_neighbors(union, rr, cc):
            _ = (nr, nc)
            count += 1
    return count


def _place_sprites(layer_masks: list[np.ndarray], layer_colors: list[int]) -> np.ndarray | None:
    canvas_h = canvas_w = 10
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    union = np.zeros((canvas_h, canvas_w), dtype=bool)
    order = sorted(range(len(layer_masks)), key=lambda idx: (-int(layer_masks[idx].sum()), random.random()))

    first = order[0]
    first_mask = layer_masks[first]
    fh, fw = first_mask.shape
    first_positions = [
        (r, c)
        for r in range(1, canvas_h - fh)
        for c in range(1, canvas_w - fw)
    ]
    random.shuffle(first_positions)
    start_r, start_c = first_positions[0]
    canvas[start_r : start_r + fh, start_c : start_c + fw][first_mask] = layer_colors[first]
    union[start_r : start_r + fh, start_c : start_c + fw] |= first_mask

    for idx in order[1:]:
        sprite = layer_masks[idx]
        sh, sw = sprite.shape
        candidates: list[tuple[int, int, int, int, int]] = []
        occupied_pts = np.argwhere(union)
        ur0, uc0 = occupied_pts.min(axis=0)
        ur1, uc1 = occupied_pts.max(axis=0)
        for top in range(canvas_h - sh + 1):
            for left in range(canvas_w - sw + 1):
                region = union[top : top + sh, left : left + sw]
                if bool(np.any(region & sprite)):
                    continue
                contact = _touches_union(union, sprite, top, left)
                if contact == 0:
                    continue
                pts = np.argwhere(sprite) + np.array([top, left], dtype=int)
                pr0, pc0 = pts.min(axis=0)
                pr1, pc1 = pts.max(axis=0)
                nr0 = min(int(ur0), int(pr0))
                nc0 = min(int(uc0), int(pc0))
                nr1 = max(int(ur1), int(pr1))
                nc1 = max(int(uc1), int(pc1))
                new_h = nr1 - nr0 + 1
                new_w = nc1 - nc0 + 1
                if new_h > 8 or new_w > 8:
                    continue
                candidates.append((new_h * new_w, -contact, top, left, idx))
        if not candidates:
            return None
        candidates.sort()
        top_area = candidates[: min(8, len(candidates))]
        _bbox_area, _neg_contact, top, left, _idx = random.choice(top_area)
        canvas[top : top + sh, left : left + sw][sprite] = layer_colors[idx]
        union[top : top + sh, left : left + sw] |= sprite

    obj_pts = np.argwhere(union)
    r0, c0 = obj_pts.min(axis=0)
    r1, c1 = obj_pts.max(axis=0)
    obj = canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    if min(obj.shape) < 4 or max(obj.shape) > 8:
        return None
    if len(_components(obj != BG, connectivity=8)) != 1:
        return None
    if len(np.unique(obj[obj != BG])) != len(layer_masks):
        return None
    return obj


def _make_object() -> np.ndarray | None:
    n_layers = random.choice(LAYER_COUNT_OPTIONS)
    areas = list(random.sample(LAYER_AREA_POOL[n_layers], n_layers))
    random.shuffle(areas)
    masks: list[np.ndarray] = []
    for area in areas:
        sprite = _random_sprite(area)
        if sprite is None:
            return None
        masks.append(sprite)
    colors = random.sample(PALETTE, n_layers)
    obj = _place_sprites(masks, colors)
    if obj is None:
        return None
    obj_colors = [int(color) for color in np.unique(obj[obj != BG])]
    layer_areas = [int(np.count_nonzero(obj == color)) for color in obj_colors]
    if len(set(layer_areas)) != n_layers:
        return None
    return obj


def _frame_box(obj: np.ndarray) -> np.ndarray:
    oh, ow = obj.shape
    inner_h = oh + random.randint(0, 2)
    inner_w = ow + random.randint(0, 2)
    frame = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    frame[0, :] = FRAME
    frame[-1, :] = FRAME
    frame[:, 0] = FRAME
    frame[:, -1] = FRAME
    return frame


def _assemble_grid(objects: list[np.ndarray], frames: list[np.ndarray]) -> np.ndarray | None:
    n = len(objects)
    band_heights = [max(obj.shape[0], frm.shape[0]) + random.randint(*ROW_GAP_RANGE) for obj, frm in zip(objects, frames)]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    grid_h = top_margin + sum(band_heights) + bottom_margin
    if not (GRID_HEIGHT_RANGE[0] <= grid_h <= GRID_HEIGHT_RANGE[1]):
        return None

    left_region = max(obj.shape[1] for obj in objects) + random.randint(3, 5)
    right_region = max(frm.shape[1] for frm in frames) + random.randint(3, 5)
    middle_gap = random.randint(*MID_GAP_RANGE)
    grid_w = left_region + middle_gap + right_region
    if grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    row = top_margin
    object_boxes: list[tuple[int, int, int, int]] = []
    frame_boxes: list[tuple[int, int, int, int]] = []
    for obj, frm, band_h in zip(objects, frames, band_heights):
        oh, ow = obj.shape
        fh, fw = frm.shape
        obj_top = row + random.randint(0, band_h - oh)
        frm_top = row + random.randint(0, band_h - fh)
        obj_left_max = left_region - ow - 1
        frm_left_min = left_region + middle_gap + 1
        frm_left_max = grid_w - fw - 1
        if obj_left_max < 1 or frm_left_max < frm_left_min:
            return None
        obj_left = random.randint(1, obj_left_max)
        frm_left = random.randint(frm_left_min, frm_left_max)
        grid[obj_top : obj_top + oh, obj_left : obj_left + ow][obj != BG] = obj[obj != BG]
        grid[frm_top : frm_top + fh, frm_left : frm_left + fw][frm != BG] = frm[frm != BG]
        object_boxes.append((obj_top, obj_left, obj_top + oh - 1, obj_left + ow - 1))
        frame_boxes.append((frm_top, frm_left, frm_top + fh - 1, frm_left + fw - 1))
        row += band_h

    if len(_components((grid != BG) & (grid != FRAME), connectivity=8)) != n:
        return None
    if len(_find_frames(grid)) != n:
        return None

    if [box[0] for box in sorted(object_boxes)] != sorted(box[0] for box in object_boxes):
        return None
    if [box[0] for box in sorted(frame_boxes)] != sorted(box[0] for box in frame_boxes):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_OPTIONS)
    objects = []
    for _ in range(n_objects):
        obj = _make_object()
        if obj is None:
            return None
        objects.append(obj)

    frames = [_frame_box(obj) for obj in objects]
    grid = _assemble_grid(objects, frames)
    if grid is None:
        return None

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
