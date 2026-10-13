from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
YELLOW = 4
GRAY = 5
TEAL = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    comps.sort(key=lambda pts: tuple(pts.min(axis=0)))
    return comps


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out, (int(r0), int(c0), int(r1), int(c1))


def _shape_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return mask.shape[0], mask.shape[1], tuple(mask.astype(int).ravel())


def _canonical_mirror_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    flipped = mask[:, ::-1]
    return min(_shape_key(mask), _shape_key(flipped))


def _find_teal_rectangle(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _find_components(grid == TEAL)
    assert len(comps) == 1, "expected one teal rectangle"
    coords = comps[0]
    r0, c0, r1, c1 = coords.min(axis=0).tolist() + coords.max(axis=0).tolist()
    rect = (int(r0), int(c0), int(r1), int(c1))
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TEAL), "teal object must be a solid rectangle"
    return rect


def _gray_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for coords in _find_components(grid == GRAY):
        mask, bbox = _coords_to_mask(coords)
        objects.append({"coords": coords, "mask": mask, "bbox": bbox, "key": _canonical_mirror_key(mask)})
    return objects


def _cast_ray(out: np.ndarray, wall_mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> None:
    h, w = out.shape
    rr, cc = int(r), int(c)
    while 0 <= rr < h and 0 <= cc < w:
        if wall_mask[rr, cc]:
            break
        out[rr, cc] = YELLOW
        rr += dr
        cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend yellow rays from the teal rectangle while only mirror-paired gray shapes block them.

    Input structure:
    - A black background contains one solid teal rectangle and several separate gray objects.
    - Some gray objects come in left-right mirror pairs, while others have no mirrored partner anywhere in the grid.

    Rule:
    - Classify gray objects by their cropped shape up to horizontal reflection.
    - Any gray object that has at least one mirrored partner acts as a wall; singleton shapes are phantoms.
    - From every row of the teal rectangle cast rays left and right, and from every column cast rays up and down.
    - Paint every traversed cell yellow, including phantom gray pixels, but stop immediately before wall pixels.

    Key insight:
    - Blocking depends on global mirror-pairing between gray object shapes rather than on local position or color alone.
    """
    g = np.asarray(grid, dtype=int)
    rect = _find_teal_rectangle(g)
    gray_objects = _gray_objects(g)

    groups: dict[tuple[int, int, tuple[int, ...]], list[dict]] = defaultdict(list)
    for obj in gray_objects:
        groups[obj["key"]].append(obj)

    wall_mask = np.zeros_like(g, dtype=bool)
    for objs in groups.values():
        if len(objs) >= 2:
            for obj in objs:
                wall_mask[obj["coords"][:, 0], obj["coords"][:, 1]] = True

    out = g.copy()
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        _cast_ray(out, wall_mask, r, c0 - 1, 0, -1)
        _cast_ray(out, wall_mask, r, c1 + 1, 0, 1)
    for c in range(c0, c1 + 1):
        _cast_ray(out, wall_mask, r0 - 1, c, -1, 0)
        _cast_ray(out, wall_mask, r1 + 1, c, 1, 0)
    return out


def _random_connected_shape(
    rng: np.random.Generator,
    used_keys: set[tuple[int, int, tuple[int, ...]]],
    *,
    require_asymmetric: bool,
) -> np.ndarray:
    for _ in range(400):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        target_area = int(rng.integers(2, min(h * w, 7)))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(h)), int(rng.integers(w)))
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < target_area and frontier:
            base_r, base_c = random.choice(tuple(frontier))
            candidates = []
            for dr, dc in DIRS4:
                nr, nc = base_r + dr, base_c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    candidates.append((nr, nc))
            if not candidates:
                frontier.remove((base_r, base_c))
                continue
            chosen = candidates[int(rng.integers(len(candidates)))]
            mask[chosen] = True
            frontier.add(chosen)
        coords = np.argwhere(mask)
        cropped, _bbox = _coords_to_mask(coords)
        if int(cropped.sum()) < 2:
            continue
        if require_asymmetric and np.array_equal(cropped, cropped[:, ::-1]):
            continue
        key = _canonical_mirror_key(cropped)
        if key in used_keys:
            continue
        return cropped
    raise RuntimeError("failed to sample a unique shape")


def _global_cells(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    cells = np.argwhere(mask)
    return [(int(r0 + r), int(c0 + c)) for r, c in cells]


def _can_place(mask: np.ndarray, occupied: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for rr, cc in _global_cells(mask, r0, c0):
        r_lo = max(0, rr - 1)
        r_hi = min(H, rr + 2)
        c_lo = max(0, cc - 1)
        c_hi = min(W, cc + 2)
        if occupied[r_lo:r_hi, c_lo:c_hi].any():
            return False
    return True


def _paint_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    cells = _global_cells(mask, r0, c0)
    for rr, cc in cells:
        grid[rr, cc] = GRAY
        occupied[rr, cc] = True
    return cells


def _touches_rect_rays(mask: np.ndarray, r0: int, c0: int, rect: tuple[int, int, int, int]) -> bool:
    rr0, cc0, rr1, cc1 = rect
    for rr, cc in _global_cells(mask, r0, c0):
        if rr0 <= rr <= rr1 or cc0 <= cc <= cc1:
            return True
    return False


def _place_on_row(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    target_row: int,
    side: str,
) -> dict:
    H, W = grid.shape
    h, w = mask.shape
    rect_r0, rect_c0, rect_r1, rect_c1 = rect
    assert rect_r0 <= target_row <= rect_r1

    if side == "left":
        c_start, c_stop = 1, rect_c0 - w - 2
    else:
        c_start, c_stop = rect_c1 + 3, W - w - 1
    assert c_start <= c_stop

    local_cells = [tuple(cell) for cell in np.argwhere(mask)]
    rng.shuffle(local_cells)
    columns = list(range(c_start, c_stop + 1))
    for local_r, local_c in local_cells:
        r_base = target_row - int(local_r)
        if not (0 <= r_base <= H - h):
            continue
        rng.shuffle(columns)
        for c_base in columns:
            if _can_place(mask, occupied, r_base, c_base):
                cells = _paint_mask(grid, occupied, mask, r_base, c_base)
                ray_cells = [(rr, cc) for rr, cc in cells if rr == target_row]
                if not ray_cells:
                    continue
                if side == "left":
                    edge = min(cc for _rr, cc in ray_cells)
                    beyond = (target_row, edge - 1)
                else:
                    edge = max(cc for _rr, cc in ray_cells)
                    beyond = (target_row, edge + 1)
                return {"cells": cells, "ray_cells": ray_cells, "beyond": beyond}
    raise RuntimeError("failed to place row-aligned object")


def _place_on_col(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    target_col: int,
    side: str,
) -> dict:
    H, W = grid.shape
    h, w = mask.shape
    rect_r0, rect_c0, rect_r1, rect_c1 = rect
    assert rect_c0 <= target_col <= rect_c1

    if side == "up":
        r_start, r_stop = 1, rect_r0 - h - 2
    else:
        r_start, r_stop = rect_r1 + 3, H - h - 1
    assert r_start <= r_stop

    local_cells = [tuple(cell) for cell in np.argwhere(mask)]
    rng.shuffle(local_cells)
    rows = list(range(r_start, r_stop + 1))
    for local_r, local_c in local_cells:
        c_base = target_col - int(local_c)
        if not (0 <= c_base <= W - w):
            continue
        rng.shuffle(rows)
        for r_base in rows:
            if _can_place(mask, occupied, r_base, c_base):
                cells = _paint_mask(grid, occupied, mask, r_base, c_base)
                ray_cells = [(rr, cc) for rr, cc in cells if cc == target_col]
                if not ray_cells:
                    continue
                if side == "up":
                    edge = min(rr for rr, _cc in ray_cells)
                    beyond = (edge - 1, target_col)
                else:
                    edge = max(rr for rr, _cc in ray_cells)
                    beyond = (edge + 1, target_col)
                return {"cells": cells, "ray_cells": ray_cells, "beyond": beyond}
    raise RuntimeError("failed to place col-aligned object")


def _place_random_off_ray(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    H, W = grid.shape
    h, w = mask.shape
    for _ in range(600):
        r0 = int(rng.integers(0, H - h + 1))
        c0 = int(rng.integers(0, W - w + 1))
        if _touches_rect_rays(mask, r0, c0, rect):
            continue
        if _can_place(mask, occupied, r0, c0):
            return _paint_mask(grid, occupied, mask, r0, c0)
    raise RuntimeError("failed to place off-ray object")


def _build_attempt(rng: np.random.Generator) -> np.ndarray:
    H = int(rng.integers(20, 30))
    W = int(rng.integers(20, 30))
    rect_h = int(rng.integers(4, 7))
    rect_w = int(rng.integers(4, 7))
    r0 = int(rng.integers(6, H - rect_h - 5))
    c0 = int(rng.integers(6, W - rect_w - 5))
    rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = TEAL
    occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

    rect_rows = list(range(rect[0], rect[2] + 1))
    rect_cols = list(range(rect[1], rect[3] + 1))
    rng.shuffle(rect_rows)
    rng.shuffle(rect_cols)
    row_wall, row_single = rect_rows[:2]
    col_wall, col_single = rect_cols[:2]
    row_wall_side = random.choice(["left", "right"])
    row_single_side = "right" if row_wall_side == "left" else "left"
    col_wall_side = random.choice(["up", "down"])
    col_single_side = "down" if col_wall_side == "up" else "up"

    used_keys: set[tuple[int, int, tuple[int, ...]]] = set()

    wall_shape_row = _random_connected_shape(rng, used_keys, require_asymmetric=True)
    used_keys.add(_canonical_mirror_key(wall_shape_row))
    row_wall_info = _place_on_row(rng, grid, occupied, wall_shape_row, rect, row_wall, row_wall_side)
    _place_random_off_ray(rng, grid, occupied, wall_shape_row[:, ::-1], rect)

    wall_shape_col = _random_connected_shape(rng, used_keys, require_asymmetric=True)
    used_keys.add(_canonical_mirror_key(wall_shape_col))
    col_wall_info = _place_on_col(rng, grid, occupied, wall_shape_col, rect, col_wall, col_wall_side)
    _place_random_off_ray(rng, grid, occupied, wall_shape_col[:, ::-1], rect)

    row_single_shape = _random_connected_shape(rng, used_keys, require_asymmetric=False)
    used_keys.add(_canonical_mirror_key(row_single_shape))
    row_single_info = _place_on_row(rng, grid, occupied, row_single_shape, rect, row_single, row_single_side)

    col_single_shape = _random_connected_shape(rng, used_keys, require_asymmetric=False)
    used_keys.add(_canonical_mirror_key(col_single_shape))
    col_single_info = _place_on_col(rng, grid, occupied, col_single_shape, rect, col_single, col_single_side)

    for _ in range(int(rng.integers(0, 2))):
        extra_pair_shape = _random_connected_shape(rng, used_keys, require_asymmetric=bool(rng.integers(0, 2)))
        used_keys.add(_canonical_mirror_key(extra_pair_shape))
        _place_random_off_ray(rng, grid, occupied, extra_pair_shape, rect)
        _place_random_off_ray(rng, grid, occupied, extra_pair_shape[:, ::-1], rect)

    for _ in range(int(rng.integers(1, 4))):
        extra_single_shape = _random_connected_shape(rng, used_keys, require_asymmetric=False)
        used_keys.add(_canonical_mirror_key(extra_single_shape))
        _place_random_off_ray(rng, grid, occupied, extra_single_shape, rect)

    out = main(grid)
    if not np.any((grid == GRAY) & (out == YELLOW)):
        raise RuntimeError("no singleton was repainted")
    if np.count_nonzero((grid == BG) & (out == YELLOW)) < rect_h + rect_w:
        raise RuntimeError("rays are not visibly extending")
    for info in (row_wall_info, col_wall_info):
        br, bc = info["beyond"]
        if out[br, bc] != BG:
            raise RuntimeError("wall did not leave a blocked cell behind it")
    for info in (row_single_info, col_single_info):
        rr, cc = info["ray_cells"][0]
        if out[rr, cc] != YELLOW:
            raise RuntimeError("singleton on a ray was not recolored")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = np.random.default_rng()
        try:
            return _build_attempt(rng)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
