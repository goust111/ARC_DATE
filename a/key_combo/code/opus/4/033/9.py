from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRANSFORM_NAMES = ("id", "rot90", "rot180", "rot270", "flip", "flip_rot90", "flip_rot180", "flip_rot270")
MAX_GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0].copy()
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _tight_shape_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    tight = _crop_mask(mask)
    coords = tuple(map(tuple, np.argwhere(tight)))
    return tight.shape, coords


def _apply_transform(mask: np.ndarray, idx: int) -> np.ndarray:
    idx = int(idx)
    if idx < 4:
        out = np.rot90(mask, idx)
    else:
        out = np.fliplr(np.rot90(mask, idx - 4))
    return _crop_mask(out)


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    return [_apply_transform(mask, idx) for idx in range(8)]


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return min(_shape_key(variant) for variant in _all_transforms(mask))


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _is_hollow_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    local[rel[:, 0], rel[:, 1]] = True
    return np.array_equal(local, _frame_border_mask(h, w))


def _sorted_by_bbox(comps: list[np.ndarray]) -> list[np.ndarray]:
    return sorted(comps, key=lambda pts: _bbox_coords(pts)[:2])


def _extract_references(grid: np.ndarray, frame_comps: list[np.ndarray]) -> dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int]:
    mapping: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {}
    for frame in frame_comps:
        r0, c0, r1, c1 = _bbox_coords(frame)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        sprite_mask = (interior != BG) & (interior != GRAY)
        comps = _components_from_mask(sprite_mask)
        assert len(comps) == 1, "each frame must contain exactly one reference sprite"
        sprite_coords = comps[0] + np.array([r0 + 1, c0 + 1], dtype=int)
        colors = {int(grid[r, c]) for r, c in sprite_coords}
        assert len(colors) == 1, "reference sprite must be monochrome"
        key = _canonical_shape_key(_tight_shape_mask(sprite_coords))
        assert key not in mapping, "reference shapes must be unique up to rotation/mirror"
        mapping[key] = colors.pop()
    return mapping


def _paint_coords(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor each outside gray sprite by matching it to a framed reference sprite up to rotation or mirror, then erase the legend frames.

    Input structure:
    - The background is black.
    - Several gray hollow rectangular frames sit along one side of the grid.
    - Each frame encloses one small monochrome reference sprite of a distinct color.
    - Outside the frames are gray 4-connected query sprites with matching shapes.

    Rule:
    - Identify every gray component and separate hollow rectangular frame borders from non-frame gray query sprites.
    - For each frame, extract the enclosed colored sprite and build a lookup table from its shape to its color, treating rotations and mirror variants as equivalent.
    - Process the outside gray query sprites in row-major bbox order, recoloring each one with the color of the uniquely matching reference shape.
    - Erase all frame borders and all enclosed reference sprites, leaving only the recolored query sprites on black.

    Key insight:
    - The framed sprites are not output objects; they form a shape-to-color dictionary, and the outside gray sprites must be matched against that dictionary under the full rotation/mirror equivalence class.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2

    gray_comps = _components_from_mask(g == GRAY)
    assert gray_comps, "expected gray components"

    frame_comps: list[np.ndarray] = []
    query_comps: list[np.ndarray] = []
    for comp in gray_comps:
        if _is_hollow_frame(comp):
            frame_comps.append(comp)
        else:
            query_comps.append(comp)

    assert 3 <= len(frame_comps) <= 4
    assert len(query_comps) == len(frame_comps)

    reference_map = _extract_references(g, _sorted_by_bbox(frame_comps))

    out = np.zeros_like(g)
    for comp in _sorted_by_bbox(query_comps):
        key = _canonical_shape_key(_tight_shape_mask(comp))
        assert key in reference_map, "query sprite must match a reference sprite"
        _paint_coords(out, comp, reference_map[key])
    return out


def _generate_shape(area: int) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if -3 <= nr <= 3 and -3 <= nc <= 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        coords = np.asarray(sorted(cells), dtype=int)
        mask = _tight_shape_mask(coords)
        if min(mask.shape) < 2:
            continue
        keys = {_shape_key(variant) for variant in _all_transforms(mask)}
        if len(keys) != 8:
            continue
        return mask
    return None


def _generate_shape_set(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    keys: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for _ in range(300):
        shapes.clear()
        keys.clear()
        areas = [random.randint(4, 6) for _ in range(count)]
        ok = True
        for area in areas:
            shape = _generate_shape(area)
            if shape is None:
                ok = False
                break
            key = _canonical_shape_key(shape)
            if key in keys:
                ok = False
                break
            keys.add(key)
            shapes.append(shape)
        if ok and len(shapes) == count:
            return shapes.copy()
    return None


def _make_frame_patch(shape: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    sh, sw = shape.shape
    inner_h = sh + random.randint(1, 3)
    inner_w = sw + random.randint(1, 3)
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = GRAY
    patch[-1, :] = GRAY
    patch[:, 0] = GRAY
    patch[:, -1] = GRAY
    off_r = random.randint(0, inner_h - sh)
    off_c = random.randint(0, inner_w - sw)
    patch[1 + off_r : 1 + off_r + sh, 1 + off_c : 1 + off_c + sw][shape] = int(color)
    return patch, (1 + off_r, 1 + off_c)


def _iter_patch_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(patch != BG):
        cells.append((top + int(r), left + int(c), int(patch[r, c])))
    return cells


def _cells_clear(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _occupy(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _transform_plan(count: int) -> list[int]:
    plan = [random.randint(0, 7) for _ in range(count)]
    rot_slot = random.randrange(count)
    mirror_slot = random.randrange(count)
    while mirror_slot == rot_slot:
        mirror_slot = random.randrange(count)
    plan[rot_slot] = random.choice((1, 2, 3))
    plan[mirror_slot] = random.choice((4, 5, 6, 7))
    return plan


def _place_frames(
    grid: np.ndarray,
    occupied: np.ndarray,
    patches: list[np.ndarray],
    side: str,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    h, w = grid.shape
    if side == "left":
        gap = random.randint(1, 2)
        total_h = sum(p.shape[0] for p in patches) + gap * (len(patches) - 1)
        strip_w = max(p.shape[1] for p in patches)
        if total_h > h - 2 or strip_w > w - 3:
            return None
        max_top = h - total_h - 1
        max_left = w - strip_w - 1
        if max_top < 1 or max_left < 1:
            return None
        r = random.randint(1, max_top)
        base_c = random.randint(1, min(2, max_left))
        for patch in patches:
            jitter = strip_w - patch.shape[1]
            c = base_c + random.randint(0, jitter)
            positions.append((r, c))
            r += patch.shape[0] + gap
    else:
        gap = random.randint(1, 2)
        total_w = sum(p.shape[1] for p in patches) + gap * (len(patches) - 1)
        strip_h = max(p.shape[0] for p in patches)
        if total_w > w - 2 or strip_h > h - 3:
            return None
        max_left = w - total_w - 1
        max_top = h - strip_h - 1
        if max_left < 1 or max_top < 1:
            return None
        c = random.randint(1, max_left)
        base_r = random.randint(1, min(2, max_top))
        for patch in patches:
            jitter = strip_h - patch.shape[0]
            r = base_r + random.randint(0, jitter)
            positions.append((r, c))
            c += patch.shape[1] + gap

    for patch, (r0, c0) in zip(patches, positions):
        for rr, cc, color in _iter_patch_cells(patch, r0, c0):
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            grid[rr, cc] = color
        _occupy(occupied, [(rr, cc) for rr, cc, _ in _iter_patch_cells(patch, r0, c0)])
    return positions


def _query_region(side: str, frame_positions: list[tuple[int, int]], patches: list[np.ndarray], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    if side == "left":
        max_c = max(c + patch.shape[1] for (r, c), patch in zip(frame_positions, patches))
        return 1, max_c + 2, h - 1, w - 1
    max_r = max(r + patch.shape[0] for (r, c), patch in zip(frame_positions, patches))
    return max_r + 2, 1, h - 1, w - 1


def _place_queries(
    grid: np.ndarray,
    occupied: np.ndarray,
    region: tuple[int, int, int, int],
    shapes: list[np.ndarray],
    colors: list[int],
    transforms: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    r_lo, c_lo, r_hi, c_hi = region
    expected = np.zeros_like(grid)
    meta: list[dict] = []
    order = list(range(len(shapes)))
    random.shuffle(order)
    for idx in order:
        patch_mask = _apply_transform(shapes[idx], transforms[idx])
        ph, pw = patch_mask.shape
        placements = []
        for r0 in range(r_lo, r_hi - ph + 1):
            for c0 in range(c_lo, c_hi - pw + 1):
                cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(patch_mask)]
                if _cells_clear(occupied, cells):
                    placements.append((r0, c0, cells))
        if not placements:
            return None
        r0, c0, cells = random.choice(placements)
        for rr, cc in cells:
            grid[rr, cc] = GRAY
            expected[rr, cc] = colors[idx]
        _occupy(occupied, cells)
        meta.append(
            {
                "index": idx,
                "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
                "transform": int(transforms[idx]),
                "mask": patch_mask,
                "cells": cells,
                "color": int(colors[idx]),
            }
        )
    return expected, meta


def _direct_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return _shape_key(_crop_mask(mask))


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    shapes: list[np.ndarray],
    colors: list[int],
    transforms: list[int],
    query_meta: list[dict],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if len({int(c) for c in colors}) != len(colors):
        return False
    ref_keys = [_canonical_shape_key(shape) for shape in shapes]
    if len(set(ref_keys)) != len(ref_keys):
        return False
    direct_ref = {_direct_shape_key(shape) for shape in shapes}
    if not any(t in {1, 2, 3} for t in transforms):
        return False
    if not any(t in {4, 5, 6, 7} for t in transforms):
        return False
    if not any(_direct_shape_key(meta["mask"]) not in direct_ref for meta in query_meta):
        return False
    if set(np.unique(expected)) - {BG} != set(colors):
        return False
    gray_components = _components_from_mask(grid == GRAY)
    frame_count = sum(_is_hollow_frame(comp) for comp in gray_components)
    if frame_count != len(shapes):
        return False
    query_count = len(gray_components) - frame_count
    if query_count != len(shapes):
        return False
    return True


def _build_instance() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int], list[int], list[dict]] | None:
    count = random.randint(3, 4)
    shapes = _generate_shape_set(count)
    if shapes is None:
        return None
    colors = random.sample(PALETTE, count)
    frame_patches = []
    for shape, color in zip(shapes, colors):
        patch, _ = _make_frame_patch(shape, color)
        frame_patches.append(patch)

    side = random.choice(("left", "top"))
    if side == "left":
        gap = 2
        total_h = 2 + sum(p.shape[0] for p in frame_patches) + (count - 1) * random.randint(1, 2)
        query_w = random.randint(10, 15)
        h = max(random.randint(18, 24), total_h + 1)
        w = max(max(p.shape[1] for p in frame_patches) + gap + query_w, 18)
    else:
        gap = 2
        total_w = 2 + sum(p.shape[1] for p in frame_patches) + (count - 1) * random.randint(1, 2)
        query_h = random.randint(10, 15)
        h = max(max(p.shape[0] for p in frame_patches) + gap + query_h, 18)
        w = max(random.randint(18, 24), total_w + 1)

    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    patches = frame_patches.copy()
    random.shuffle(patches)
    patch_to_key = {id(patch): idx for idx, patch in enumerate(frame_patches)}

    frame_positions = _place_frames(grid, occupied, patches, side)
    if frame_positions is None:
        return None
    region = _query_region(side, frame_positions, patches, grid.shape)
    if region[0] >= region[2] or region[1] >= region[3]:
        return None

    ordered_shapes = [shapes[patch_to_key[id(patch)]] for patch in patches]
    ordered_colors = [colors[patch_to_key[id(patch)]] for patch in patches]
    transforms = _transform_plan(count)
    query_result = _place_queries(grid, occupied, region, ordered_shapes, ordered_colors, transforms)
    if query_result is None:
        return None
    expected, query_meta = query_result
    return grid, expected, ordered_shapes, ordered_colors, transforms, query_meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, expected, shapes, colors, transforms, query_meta = built
        if _validate_instance(grid, expected, shapes, colors, transforms, query_meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
