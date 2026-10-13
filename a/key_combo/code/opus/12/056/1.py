from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = (2, 1, 3, 7, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                holes.append(np.array(cells, dtype=int))
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    return len(_connected_components(mask)) == 1


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            holes_local = _hole_components(local)
            holes_global = [
                hole + np.array([r0, c0], dtype=int) for hole in holes_local
            ]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                    "holes": holes_global,
                    "hollow": bool(holes_global),
                }
            )
    return objects


def _transform_details(grid: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _analyze_objects(g)
    out = np.zeros_like(g)
    fg_mask = g != BG

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    centers = np.array([obj["center"] for obj in objects], dtype=int)
    dist_sq = (
        (rows[None, :, :] - centers[:, 0][:, None, None]) ** 2
        + (cols[None, :, :] - centers[:, 1][:, None, None]) ** 2
    )
    best = dist_sq.min(axis=0)
    tie_mask = (dist_sq == best[None, :, :]).sum(axis=0) > 1
    owner = dist_sq.argmin(axis=0).astype(int)
    owner[tie_mask] = -1
    owner[fg_mask] = -2

    for idx, obj in enumerate(objects):
        cr, cc = obj["center"]
        d = np.maximum(np.abs(rows - cr), np.abs(cols - cc))
        phase = 0 if obj["hollow"] else 1
        fill = (owner == idx) & ((d % 2) == phase)
        out[fill] = int(obj["color"])

    out[fg_mask] = g[fg_mask]
    return out, objects, owner, tie_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint center-based concentric ring territories whose parity flips for hollow objects.

    Input structure:
    - A black background contains 3-5 separated 4-connected monochrome foreground objects.
    - Objects use distinct visible colors, have non-overlapping bounding boxes, and some contain enclosed black holes inside their own bbox.
    - The only reference points are the objects themselves: each object contributes its bbox center and its hollow/solid topology.

    Rule:
    - Detect every foreground object and decide whether it is hollow by checking for a 4-connected black component inside the object's bbox that does not touch that bbox edge.
    - Compute each object's bbox center with floor division on both coordinates.
    - For every non-object cell, compare squared Euclidean distance to all object centers; ties stay black, otherwise the cell belongs to the unique nearest object.
    - Inside each owned territory, color Chebyshev-distance rings around the owning center: hollow objects color even-distance rings, solid objects color odd-distance rings.
    - Restore the original object pixels unchanged on top of the ring field.

    Key insight:
    - The same Voronoi-clipped ring pattern appears around every object, but the ring phase is inverted only by the object's hole topology, so solving requires combining hole detection, center finding, and nearest-owner clipping.
    """
    return _transform_details(np.asarray(grid, dtype=int))[0]


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            key = (variant.shape, variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _make_hollow_shape() -> np.ndarray:
    base_dims = random.choice([(3, 3), (3, 4), (4, 3), (4, 4)])
    for _ in range(120):
        h, w = base_dims
        coords = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        }

        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        if random.random() < 0.75:
            coords.discard(random.choice(corners))

        if random.random() < 0.55:
            edge_choices = [
                (-1, random.randrange(w)),
                (h, random.randrange(w)),
                (random.randrange(h), -1),
                (random.randrange(h), w),
            ]
            tr, tc = random.choice(edge_choices)
            attach = []
            for dr, dc in DIR4:
                nr, nc = tr + dr, tc + dc
                if (nr, nc) in coords:
                    attach.append((nr, nc))
            if attach:
                coords.add((tr, tc))

        mask = _coords_to_mask(coords)
        mask = _normalize_mask(mask)
        holes = _hole_components(mask)
        if not _is_connected(mask):
            continue
        if not holes:
            continue
        area = int(mask.sum())
        if not (5 <= area <= 12):
            continue
        hole_sizes = [len(hole) for hole in holes]
        if any(size < 1 or size > 4 for size in hole_sizes):
            continue
        return random.choice(_dihedral_variants(mask))
    raise RuntimeError("failed to build hollow shape")


def _make_solid_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        max_area = min(12, h * w - 1)
        if max_area < 5:
            continue
        target = random.randint(5, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(tuple(frontier)))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _normalize_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == mask.shape[0] * mask.shape[1]:
            continue
        if _hole_components(mask):
            continue
        if not _is_connected(mask):
            continue
        return random.choice(_dihedral_variants(mask))
    raise RuntimeError("failed to build solid shape")


def _translate_shape(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([top, left], dtype=int)


def _min_chebyshev(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    best = 10**9
    for ra, ca in coords_a:
        d = np.max(np.abs(coords_b - np.array([ra, ca], dtype=int)), axis=1).min()
        best = min(best, int(d))
    return best


def _place_shapes(shapes: list[np.ndarray], grid_shape: tuple[int, int]) -> list[np.ndarray] | None:
    h, w = grid_shape
    order = sorted(range(len(shapes)), key=lambda i: shapes[i].shape[0] * shapes[i].shape[1], reverse=True)
    placed: list[np.ndarray | None] = [None] * len(shapes)
    placed_coords: list[np.ndarray] = []

    for idx in order:
        mask = shapes[idx]
        sh, sw = mask.shape
        if sh + 2 > h or sw + 2 > w:
            return None
        ok = False
        for _ in range(PLACE_ATTEMPTS):
            top = random.randint(1, h - sh - 1)
            left = random.randint(1, w - sw - 1)
            coords = _translate_shape(mask, top, left)
            if any(_min_chebyshev(coords, other) < 3 for other in placed_coords):
                continue
            placed[idx] = coords
            placed_coords.append(coords)
            ok = True
            break
        if not ok:
            return None

    return [coords for coords in placed if coords is not None]


def _build_grid(coords_list: list[np.ndarray], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for coords, color in zip(coords_list, colors):
        grid[coords[:, 0], coords[:, 1]] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    objects = _analyze_objects(grid)
    if not (3 <= len(objects) <= 5):
        return False
    if len({obj["color"] for obj in objects}) != len(objects):
        return False
    hollows = [obj for obj in objects if obj["hollow"]]
    solids = [obj for obj in objects if not obj["hollow"]]
    if not hollows or not solids:
        return False
    for obj in objects:
        area = int(len(obj["coords"]))
        if not (5 <= area <= 12):
            return False
    for i, obj_a in enumerate(objects):
        ar0, ac0, ar1, ac1 = obj_a["bbox"]
        if ar0 < 0 or ac0 < 0 or ar1 >= grid.shape[0] or ac1 >= grid.shape[1]:
            return False
        for obj_b in objects[i + 1 :]:
            br0, bc0, br1, bc1 = obj_b["bbox"]
            if not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0):
                return False
            if _min_chebyshev(obj_a["coords"], obj_b["coords"]) < 3:
                return False

    out, details, owner, tie_mask = _transform_details(grid)
    if len(details) != len(objects):
        return False
    non_fg = grid == BG
    if not np.any(tie_mask & non_fg):
        return False
    added = (out != BG) & non_fg
    if not np.any(added):
        return False
    if len({int(v) for v in out[added]}) < 2:
        return False

    rows = np.arange(grid.shape[0])[:, None]
    cols = np.arange(grid.shape[1])[None, :]
    for idx, obj in enumerate(details):
        cr, cc = obj["center"]
        d = np.maximum(np.abs(rows - cr), np.abs(cols - cc))
        owned = owner == idx
        colored_owned = owned & (out == int(obj["color"])) & non_fg
        if not np.any(colored_owned):
            return False
        if obj["hollow"]:
            hole_mask = np.zeros(grid.shape, dtype=bool)
            for hole in obj["holes"]:
                hole_mask[hole[:, 0], hole[:, 1]] = True
            if not np.any(hole_mask):
                return False
            if not np.any((out == int(obj["color"])) & hole_mask):
                return False
            if not np.any(colored_owned & ((d % 2) == 0)):
                return False
        else:
            if not np.any(colored_owned & ((d % 2) == 1)):
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(17, 22)
        w = random.randint(17, 22)
        count = random.randint(3, 5)
        hollow_count = random.randint(1, count - 1)
        colors = random.sample(PALETTE, count)

        shapes = [_make_hollow_shape() for _ in range(hollow_count)]
        shapes.extend(_make_solid_shape() for _ in range(count - hollow_count))
        random.shuffle(shapes)

        coords_list = _place_shapes(shapes, (h, w))
        if coords_list is None:
            continue
        grid = _build_grid(coords_list, colors, (h, w))
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
