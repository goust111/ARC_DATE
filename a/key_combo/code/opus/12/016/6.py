from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAROON = 9

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")
ROTATION_K = {"up": 0, "right": 3, "down": 2, "left": 1}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _scale2(grid: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(grid, dtype=int), 2, axis=0), 2, axis=1)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return patch


def _mask_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _is_valid_shape(mask: np.ndarray) -> bool:
    if int(mask.sum()) != 4:
        return False
    l_masks = [
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 1]], dtype=bool),
    ]
    t_masks = [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
        np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    ]
    for ref in l_masks + t_masks:
        if _mask_equal(mask, ref):
            return True
    return False


def _marker_direction(patch: np.ndarray) -> str:
    reds = np.argwhere(patch == RED)
    if reds.shape != (1, 2):
        raise ValueError("each component must contain exactly one red marker")
    mr, mc = map(float, reds[0])
    h, w = patch.shape
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    dr = mr - cr
    dc = mc - cc
    if abs(dr) < 1e-9:
        return "right" if dc > 0 else "left"
    if abs(dc) < 1e-9:
        return "down" if dr > 0 else "up"
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _anchor_local(mask: np.ndarray) -> tuple[int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cr = (mask.shape[0] - 1) / 2.0
    cc = (mask.shape[1] - 1) / 2.0
    best = None
    for r, c in coords:
        degree = sum((nr, nc) in coords for nr, nc in _neighbors4(r, c))
        dist2 = (r - cr) ** 2 + (c - cc) ** 2
        cand = (-degree, dist2, r, c)
        if best is None or cand < best[0]:
            best = (cand, (r, c))
    assert best is not None
    return best[1]


def _rotated_anchor_top_left(mask: np.ndarray, k: int) -> tuple[int, int]:
    ar, ac = _anchor_local(mask)
    anchor = np.zeros(mask.shape, dtype=int)
    anchor[ar, ac] = 1
    anchor = _scale2(anchor)
    anchor = np.rot90(anchor, k)
    coords = np.argwhere(anchor == 1)
    r0, c0 = coords.min(axis=0)
    return int(r0), int(c0)


def _component_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    patch = _crop_component(grid, coords)
    mask = patch != BLACK
    if not _is_valid_shape(mask):
        raise ValueError("component must be a 4-cell L or T shape")
    marker_dir = _marker_direction(patch)
    k = ROTATION_K[marker_dir]
    r0, c0, r1, c1 = _bbox(coords)
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
        "marker_dir": marker_dir,
        "rotation_k": k,
        "anchor_local": _anchor_local(mask),
    }


def _find_structures(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue
            block = g[r - 1 : r + 2, c - 1 : c + 2]
            if block.shape != (3, 3):
                continue
            ring = block.copy()
            ring[1, 1] = GREY
            if np.all(ring == GREY):
                out.append({"center": (r, c), "bbox": (r - 1, c - 1, r + 1, c + 1)})
    return out


def _find_components(grid: np.ndarray) -> list[dict]:
    comps = _components(np.isin(grid, [BLUE, RED]))
    out = []
    for coords in comps:
        out.append(_component_info(grid, coords))
    return out


def _build_scaled_component_id(grid: np.ndarray, comps: list[dict]) -> np.ndarray:
    h, w = grid.shape
    out = np.full((2 * h, 2 * w), -1, dtype=int)
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            rr = int(r) * 2
            cc = int(c) * 2
            out[rr : rr + 2, cc : cc + 2] = idx
    return out


def _ray_hits_from_structure(
    scaled_grid: np.ndarray, scaled_ids: np.ndarray, center: tuple[int, int]
) -> dict[tuple[int, str], int]:
    h, w = scaled_grid.shape
    r, c = center
    hits: dict[tuple[int, str], int] = {}
    row_lines = [2 * r, 2 * r + 1]
    col_lines = [2 * c, 2 * c + 1]

    for rr in row_lines:
        cc = 2 * c + 4
        dist = 0
        while 0 <= rr < h and 0 <= cc < w:
            if scaled_ids[rr, cc] >= 0:
                key = (int(scaled_ids[rr, cc]), "right")
                hits[key] = min(hits.get(key, 10**9), dist)
                break
            if int(scaled_grid[rr, cc]) != BLACK:
                break
            cc += 1
            dist += 1

        cc = 2 * c - 3
        dist = 0
        while 0 <= rr < h and 0 <= cc < w:
            if scaled_ids[rr, cc] >= 0:
                key = (int(scaled_ids[rr, cc]), "left")
                hits[key] = min(hits.get(key, 10**9), dist)
                break
            if int(scaled_grid[rr, cc]) != BLACK:
                break
            cc -= 1
            dist += 1

    for cc in col_lines:
        rr = 2 * r - 3
        dist = 0
        while 0 <= rr < h and 0 <= cc < w:
            if scaled_ids[rr, cc] >= 0:
                key = (int(scaled_ids[rr, cc]), "up")
                hits[key] = min(hits.get(key, 10**9), dist)
                break
            if int(scaled_grid[rr, cc]) != BLACK:
                break
            rr -= 1
            dist += 1

        rr = 2 * r + 4
        dist = 0
        while 0 <= rr < h and 0 <= cc < w:
            if scaled_ids[rr, cc] >= 0:
                key = (int(scaled_ids[rr, cc]), "down")
                hits[key] = min(hits.get(key, 10**9), dist)
                break
            if int(scaled_grid[rr, cc]) != BLACK:
                break
            rr += 1
            dist += 1

    return hits


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    structures = _find_structures(g)
    components = _find_components(g)
    scaled_grid = _scale2(g)
    scaled_ids = _build_scaled_component_id(g, components)

    proposals = []
    for s_idx, structure in enumerate(structures):
        hits = _ray_hits_from_structure(scaled_grid, scaled_ids, structure["center"])
        if not hits:
            continue
        best = min((dist, comp_idx, direction) for (comp_idx, direction), dist in hits.items())
        proposals.append((best[0], s_idx, best[1], best[2]))

    proposals.sort()
    assigned_structs = set()
    assigned_comps = set()
    assignments = []
    for dist, s_idx, comp_idx, direction in proposals:
        if s_idx in assigned_structs or comp_idx in assigned_comps:
            continue
        assigned_structs.add(s_idx)
        assigned_comps.add(comp_idx)
        assignments.append(
            {
                "structure_idx": s_idx,
                "component_idx": comp_idx,
                "ray_dir": direction,
                "distance": int(dist),
            }
        )

    return {
        "structures": structures,
        "components": components,
        "assignments": assignments,
    }


def _move_vector_for_ray(ray_dir: str) -> tuple[int, int]:
    dr, dc = DIRS[ray_dir]
    return -dr, -dc


def _transform_component_for_target(comp: dict, target_center: tuple[int, int]) -> dict:
    r0, c0, r1, c1 = comp["bbox"]
    patch = comp["patch"]
    h, w = patch.shape
    k = int(comp["rotation_k"])
    scaled_patch = _scale2(patch)
    rotated_patch = np.rot90(scaled_patch, k)
    rot_r0 = 2 * r0 + (h - w if k % 2 == 1 else 0)
    rot_c0 = 2 * c0 + (w - h if k % 2 == 1 else 0)
    anchor_local = _rotated_anchor_top_left(comp["mask"], k)
    anchor_world = (rot_r0 + anchor_local[0], rot_c0 + anchor_local[1])
    cavity_top_left = (2 * target_center[0], 2 * target_center[1])
    delta = (cavity_top_left[0] - anchor_world[0], cavity_top_left[1] - anchor_world[1])
    return {
        "scaled_patch": scaled_patch,
        "rotated_patch": rotated_patch,
        "rotated_top_left": (int(rot_r0), int(rot_c0)),
        "anchor_world": (int(anchor_world[0]), int(anchor_world[1])),
        "delta": (int(delta[0]), int(delta[1])),
        "final_top_left": (int(rot_r0 + delta[0]), int(rot_c0 + delta[1])),
        "cavity_top_left": cavity_top_left,
    }


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top_left: tuple[int, int], color: int | None = None) -> None:
    r0, c0 = map(int, top_left)
    h, w = patch.shape
    mask = patch != BLACK
    rr0, rr1 = r0, r0 + h
    cc0, cc1 = c0, c0 + w
    if not (0 <= rr0 and 0 <= cc0 and rr1 <= dst.shape[0] and cc1 <= dst.shape[1]):
        raise ValueError("patch out of bounds")
    if color is None:
        dst[rr0:rr1, cc0:cc1][mask] = patch[mask]
    else:
        dst[rr0:rr1, cc0:cc1][mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Double the grid, ray-match each marked tetromino to a boxed cavity, rotate it from its marker, and dock it into the cavity.

    Input structure:
    - The background is black.
    - There are 2-3 grey 3x3 box structures whose center cell is a black cavity.
    - There are 2-3 movable blue components, each a 4-connected 4-cell L or T tetromino with exactly one red marker pixel.

    Rule:
    - First upscale the whole grid by a factor of two so every input pixel becomes a 2x2 block.
    - For each structure, cast axis-aligned rays outward from the scaled cavity lines; the first hit movable component is that structure's partner and the hit side identifies the docking side.
    - Read the component's marker position inside its original tight bbox: above -> 0 degrees, right -> 90 clockwise, below -> 180 degrees, left -> 270 clockwise; rotate the upscaled component accordingly around its bbox center.
    - Move the rotated component inward from its hit side until its central 2x2 anchor block lands exactly on the structure's 2x2 cavity, erase the component's original position to black, and paint the traversed path maroon.
    - Components that are never hit by any structure ray stay in place.

    Key insight:
    - The 2x upscale is not cosmetic: it creates a 2x2 cavity that the rotated component can dock into, while the single red marker independently encodes how much that component must rotate before sliding.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    analysis = _analyze_scene(g)
    out = _scale2(g)

    for assignment in analysis["assignments"]:
        comp = analysis["components"][assignment["component_idx"]]
        structure = analysis["structures"][assignment["structure_idx"]]
        tf = _transform_component_for_target(comp, structure["center"])

        for r, c in comp["coords"]:
            rr = int(r) * 2
            cc = int(c) * 2
            out[rr : rr + 2, cc : cc + 2] = BLACK

        delta_r, delta_c = tf["delta"]
        steps = max(abs(delta_r), abs(delta_c))
        step_r = 0 if delta_r == 0 else (1 if delta_r > 0 else -1)
        step_c = 0 if delta_c == 0 else (1 if delta_c > 0 else -1)
        for t in range(steps):
            pos = (
                tf["rotated_top_left"][0] + step_r * t,
                tf["rotated_top_left"][1] + step_c * t,
            )
            _paint_patch(out, tf["rotated_patch"], pos, color=MAROON)

        _paint_patch(out, tf["rotated_patch"], tf["final_top_left"])

    return out


T_TEMPLATES = [
    np.array([[BLUE, RED, BLUE], [BLACK, BLUE, BLACK]], dtype=int),
    np.array([[BLUE, BLUE, RED], [BLACK, BLUE, BLACK]], dtype=int),
    np.array([[BLUE, BLUE, BLUE], [BLACK, RED, BLACK]], dtype=int),
    np.array([[RED, BLUE, BLUE], [BLACK, BLUE, BLACK]], dtype=int),
    np.array([[BLACK, BLUE], [RED, BLUE], [BLACK, BLUE]], dtype=int),
    np.array([[BLACK, RED], [BLUE, BLUE], [BLACK, BLUE]], dtype=int),
    np.array([[BLACK, BLUE], [BLUE, BLUE], [BLACK, RED]], dtype=int),
    np.array([[BLACK, BLUE], [BLUE, BLUE], [BLACK, RED]], dtype=int),
]

L_TEMPLATES = [
    np.array([[RED, BLACK], [BLUE, BLACK], [BLUE, BLUE]], dtype=int),
    np.array([[BLUE, BLACK], [RED, BLACK], [BLUE, BLUE]], dtype=int),
    np.array([[BLUE, BLACK], [BLUE, BLACK], [RED, BLUE]], dtype=int),
    np.array([[BLUE, BLUE, RED], [BLUE, BLACK, BLACK]], dtype=int),
    np.array([[BLACK, RED], [BLACK, BLUE], [BLUE, BLUE]], dtype=int),
    np.array([[RED, BLUE, BLUE], [BLACK, BLACK, BLUE]], dtype=int),
]

ALL_TEMPLATES = {"T": T_TEMPLATES, "L": L_TEMPLATES}

GRID_RANGE = (11, 15)
GEN_RETRIES = 500


def _place_structure(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = GREY
    grid[r, c] = BLACK


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    dr = 0
    if a[2] < b[0]:
        dr = b[0] - a[2] - 1
    elif b[2] < a[0]:
        dr = a[0] - b[2] - 1
    dc = 0
    if a[3] < b[1]:
        dc = b[1] - a[3] - 1
    elif b[3] < a[1]:
        dc = a[1] - b[3] - 1
    return max(dr, dc) if dr == 0 or dc == 0 else min(dr, dc)


def _choose_structure_centers(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    rows = list(range(2, h - 2))
    cols = list(range(2, w - 2))
    random.shuffle(rows)
    random.shuffle(cols)
    centers = []
    for r in rows:
        for c in cols:
            if any(r == rr or c == cc for rr, cc in centers):
                continue
            bb = (r - 1, c - 1, r + 1, c + 1)
            if any(_bbox_gap(bb, (rr - 1, cc - 1, rr + 1, cc + 1)) < 2 for rr, cc in centers):
                continue
            centers.append((r, c))
            if len(centers) == count:
                return centers
    return None


def _fits_on_black(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> bool:
    r0, c0 = top_left
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    return bool(np.all(sub[patch != BLACK] == BLACK))


def _component_isolated(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> bool:
    coords = {tuple(map(int, rc)) for rc in _component_abs_coords(patch, top_left)}
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords:
                continue
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) in (BLUE, RED):
                return False
    return True


def _component_abs_coords(patch: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    r0, c0 = top_left
    coords = np.argwhere(patch != BLACK)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _clear_ray(grid: np.ndarray, center: tuple[int, int], patch: np.ndarray, top_left: tuple[int, int], ray_dir: str) -> bool:
    r, c = center
    coords = _component_abs_coords(patch, top_left)
    pts = [tuple(map(int, rc)) for rc in coords]
    if ray_dir == "right":
        cols = [cc for rr, cc in pts if rr == r and cc > c + 1]
        if not cols:
            return False
        nearest = min(cols)
        return bool(np.all(grid[r, c + 2 : nearest] == BLACK))
    if ray_dir == "left":
        cols = [cc for rr, cc in pts if rr == r and cc < c - 1]
        if not cols:
            return False
        nearest = max(cols)
        return bool(np.all(grid[r, nearest + 1 : c - 1] == BLACK))
    if ray_dir == "down":
        rows = [rr for rr, cc in pts if cc == c and rr > r + 1]
        if not rows:
            return False
        nearest = min(rows)
        return bool(np.all(grid[r + 2 : nearest, c] == BLACK))
    rows = [rr for rr, cc in pts if cc == c and rr < r - 1]
    if not rows:
        return False
    nearest = max(rows)
    return bool(np.all(grid[nearest + 1 : r - 1, c] == BLACK))


def _scaled_cells(patch: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    out = set()
    for r, c in np.argwhere(patch != BLACK):
        rr = int(r0 + r)
        cc = int(c0 + c)
        out.add((rr, cc))
    return out


def _scaled_occupied_from_grid(grid: np.ndarray) -> set[tuple[int, int]]:
    out = set()
    for r, c in np.argwhere(grid != BLACK):
        rr = int(r) * 2
        cc = int(c) * 2
        for dr in range(2):
            for dc in range(2):
                out.add((rr + dr, cc + dc))
    return out


def _placement_candidates(
    grid: np.ndarray,
    structures: list[tuple[int, int]],
    target_idx: int,
    patch: np.ndarray,
    ray_dir: str,
    reserved_final: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    target_center = structures[target_idx]
    target_r, target_c = target_center
    target_bbox_scaled = {
        (rr, cc)
        for rr in range(2 * (target_r - 1), 2 * (target_r + 2))
        for cc in range(2 * (target_c - 1), 2 * (target_c + 2))
    }
    other_struct_scaled = set()
    for i, (sr, sc) in enumerate(structures):
        if i == target_idx:
            continue
        for rr in range(2 * (sr - 1), 2 * (sr + 2)):
            for cc in range(2 * (sc - 1), 2 * (sc + 2)):
                other_struct_scaled.add((rr, cc))

    forbidden_now = _scaled_occupied_from_grid(grid)

    out = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if ray_dir == "right" and c0 <= target_c + 1:
                continue
            if ray_dir == "left" and c0 + pw - 1 >= target_c - 1:
                continue
            if ray_dir == "down" and r0 <= target_r + 1:
                continue
            if ray_dir == "up" and r0 + ph - 1 >= target_r - 1:
                continue
            pos = (r0, c0)
            if not _fits_on_black(grid, patch, pos):
                continue
            if not _component_isolated(grid, patch, pos):
                continue
            if not _clear_ray(grid, target_center, patch, pos, ray_dir):
                continue

            coords = _component_abs_coords(patch, pos)
            comp = _component_info(
                np.where(patch == BLACK, BLACK, patch),
                np.argwhere(patch != BLACK),
            )
            comp["bbox"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            comp["coords"] = coords
            tf = _transform_component_for_target(comp, target_center)
            delta_r, delta_c = tf["delta"]
            move_r, move_c = _move_vector_for_ray(ray_dir)
            if move_r == 0 and not (delta_r == 0 and delta_c * move_c > 0):
                continue
            if move_c == 0 and not (delta_c == 0 and delta_r * move_r > 0):
                continue

            try:
                final_cells = _scaled_cells(tf["rotated_patch"], tf["final_top_left"])
            except Exception:
                continue
            if any(
                rr < 0 or cc < 0 or rr >= 2 * h or cc >= 2 * w
                for rr, cc in final_cells
            ):
                continue
            if final_cells & other_struct_scaled:
                continue
            if final_cells & reserved_final:
                continue

            steps = max(abs(delta_r), abs(delta_c))
            step_r = 0 if delta_r == 0 else (1 if delta_r > 0 else -1)
            step_c = 0 if delta_c == 0 else (1 if delta_c > 0 else -1)
            ok = True
            for t in range(steps):
                pos2 = (
                    tf["rotated_top_left"][0] + step_r * t,
                    tf["rotated_top_left"][1] + step_c * t,
                )
                path_cells = _scaled_cells(tf["rotated_patch"], pos2)
                if any(rr < 0 or cc < 0 or rr >= 2 * h or cc >= 2 * w for rr, cc in path_cells):
                    ok = False
                    break
                if path_cells & other_struct_scaled:
                    ok = False
                    break
                if (path_cells & reserved_final) - target_bbox_scaled:
                    ok = False
                    break
                if (path_cells & forbidden_now) - target_bbox_scaled:
                    ok = False
                    break
            if ok:
                out.append(pos)
    return out


def _paint_original_patch(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[patch != BLACK] = patch[patch != BLACK]


def _unmatched_candidate_positions(
    grid: np.ndarray, structures: list[tuple[int, int]], patch: np.ndarray
) -> list[tuple[int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    rows_forbidden = {r for r, _ in structures}
    cols_forbidden = {c for _, c in structures}
    out = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if not _fits_on_black(grid, patch, (r0, c0)):
                continue
            if not _component_isolated(grid, patch, (r0, c0)):
                continue
            coords = _component_abs_coords(patch, (r0, c0))
            if any(int(r) in rows_forbidden or int(c) in cols_forbidden for r, c in coords):
                continue
            out.append((r0, c0))
    return out


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        analysis = _analyze_scene(inp)
        out = main(inp)
    except Exception:
        return False

    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if out.shape != (inp.shape[0] * 2, inp.shape[1] * 2):
        return False
    if len(analysis["structures"]) not in (2, 3):
        return False
    if len(analysis["components"]) not in (2, 3):
        return False
    if len(analysis["assignments"]) < 2:
        return False

    comp_use = {a["component_idx"] for a in analysis["assignments"]}
    if len(comp_use) != len(analysis["assignments"]):
        return False

    for a in analysis["assignments"]:
        comp = analysis["components"][a["component_idx"]]
        structure = analysis["structures"][a["structure_idx"]]
        tf = _transform_component_for_target(comp, structure["center"])
        delta_r, delta_c = tf["delta"]
        move_r, move_c = _move_vector_for_ray(a["ray_dir"])
        if move_r == 0 and not (delta_r == 0 and delta_c * move_c > 0):
            return False
        if move_c == 0 and not (delta_c == 0 and delta_r * move_r > 0):
            return False

    if np.array_equal(out, _scale2(inp)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    if 2 * max(h, w) > 30:
        return None

    structure_count = 3 if random.random() < 0.8 else 2
    centers = _choose_structure_centers(h, w, structure_count)
    if centers is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for center in centers:
        _place_structure(grid, center)

    total_components = 2 if structure_count == 2 else (3 if random.random() < 0.85 else 2)
    matched_count = min(total_components, structure_count)
    if structure_count == 3 and total_components == 3 and random.random() < 0.65:
        matched_count = 2
    extra_count = total_components - matched_count

    target_indices = random.sample(range(structure_count), matched_count)
    reserved_final: set[tuple[int, int]] = set()

    for target_idx in target_indices:
        success = False
        shape_types = ["L", "T"]
        random.shuffle(shape_types)
        for shape_type in shape_types:
            templates = ALL_TEMPLATES[shape_type][:]
            random.shuffle(templates)
            dir_options = list(DIR_ORDER)
            random.shuffle(dir_options)
            for patch in templates:
                for ray_dir in dir_options:
                    candidates = _placement_candidates(grid, centers, target_idx, patch, ray_dir, reserved_final)
                    if not candidates:
                        continue
                    pos = random.choice(candidates)
                    _paint_original_patch(grid, patch, pos)
                    analysis = _analyze_scene(grid)
                    if len(analysis["assignments"]) < 1:
                        grid[pos[0] : pos[0] + patch.shape[0], pos[1] : pos[1] + patch.shape[1]][patch != BLACK] = BLACK
                        continue
                    latest = None
                    for a in analysis["assignments"]:
                        if a["structure_idx"] == target_idx:
                            latest = a
                    if latest is None:
                        grid[pos[0] : pos[0] + patch.shape[0], pos[1] : pos[1] + patch.shape[1]][patch != BLACK] = BLACK
                        continue
                    comp = analysis["components"][latest["component_idx"]]
                    tf = _transform_component_for_target(comp, centers[target_idx])
                    reserved_final |= _scaled_cells(tf["rotated_patch"], tf["final_top_left"])
                    success = True
                    break
                if success:
                    break
            if success:
                break
        if not success:
            return None

    for _ in range(extra_count):
        success = False
        shape_types = ["L", "T"]
        random.shuffle(shape_types)
        for shape_type in shape_types:
            templates = ALL_TEMPLATES[shape_type][:]
            random.shuffle(templates)
            for patch in templates:
                candidates = _unmatched_candidate_positions(grid, centers, patch)
                if not candidates:
                    continue
                pos = random.choice(candidates)
                _paint_original_patch(grid, patch, pos)
                success = True
                break
            if success:
                break
        if not success:
            return None

    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
