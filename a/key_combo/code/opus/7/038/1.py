from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500
SHAPE_RETRIES = 400
PLACEMENT_RETRIES = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    br0, bc0, br1, bc1 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _has_role_symmetry(mask: np.ndarray) -> bool:
    return bool(
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, rng.randrange(4))
    return _crop_mask(out)


def _make_random_connected_mask(rng: random.Random, want_symmetric: bool) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        area = rng.randint(3, min(6, h * w))
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if not (3 <= int(mask.sum()) <= 6):
            continue
        if _has_role_symmetry(mask) != want_symmetric:
            continue
        return _random_transform(mask, rng)
    raise RuntimeError("failed to sample a valid shape")


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[dict] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            bbox = _bbox_from_coords(coords)
            mask = np.zeros((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), dtype=bool)
            mask[coords[:, 0] - bbox[0], coords[:, 1] - bbox[1]] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _index_grid(shape: tuple[int, int], comps: list[dict]) -> np.ndarray:
    idx = np.full(shape, -1, dtype=int)
    for i, comp in enumerate(comps):
        coords = comp["coords"]
        idx[coords[:, 0], coords[:, 1]] = i
    return idx


def _corner_rays(bbox: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return [
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    ]


def _ray_cells(
    shape: tuple[int, int], occupied: np.ndarray, bbox: tuple[int, int, int, int]
) -> list[dict]:
    rays = []
    for corner_id, (corner, direction) in enumerate(_corner_rays(bbox)):
        dr, dc = direction
        r, c = corner[0] + dr, corner[1] + dc
        cells = []
        while _in_bounds(shape, r, c) and not occupied[r, c]:
            cells.append((r, c))
            r += dr
            c += dc
        rays.append(
            {
                "corner_id": corner_id,
                "direction": direction,
                "cells": cells,
            }
        )
    return rays


def _hit_sources(comps: list[dict], shape: tuple[int, int], grid: np.ndarray) -> dict[int, set[int]]:
    idx_grid = _index_grid(shape, comps)
    sources = [i for i, comp in enumerate(comps) if _has_role_symmetry(comp["mask"])]
    targets = [i for i in range(len(comps)) if i not in sources]
    hits = {i: set() for i in targets}
    for src in sources:
        for corner, direction in _corner_rays(comps[src]["bbox"]):
            dr, dc = direction
            r, c = corner[0] + dr, corner[1] + dc
            while _in_bounds(shape, r, c):
                if int(grid[r, c]) == BACKGROUND:
                    r += dr
                    c += dc
                    continue
                dst = int(idx_grid[r, c])
                if dst in hits:
                    hits[dst].add(src)
                break
    return hits


def _distance_key(src_bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
    sr = src_bbox[0] + src_bbox[2]
    sc = src_bbox[1] + src_bbox[3]
    tr = dst_bbox[0] + dst_bbox[2]
    tc = dst_bbox[1] + dst_bbox[3]
    dist = abs(sr - tr) + abs(sc - tc)
    return dist, src_bbox[0], src_bbox[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every non-symmetric sprite that lies on a diagonal corner ray from a symmetric sprite.

    Input structure:
    - The grid has black background and several small monochromatic 4-connected sprites.
    - Exactly the sprites whose cropped shape is vertically symmetric, horizontally symmetric, or 180-degree symmetric act as radiation sources.
    - All other sprites are receiving targets; sprites do not touch the grid edge.

    Rule:
    - Detect each sprite and test symmetry inside its bounding box mask.
    - From each source sprite's four bounding-box corners, trace one outward diagonal ray until the first non-background cell or the grid boundary.
    - If the first blocked cell on a ray belongs to a target sprite, recolor that whole target with the closest hitting source's color, using source-center Manhattan distance and then source top-left row/col as tie-breakers.

    Key insight:
    - Symmetry is not the output filter; it assigns sprite roles, which then determines who emits rays and who inherits color from the nearest hitting source.
    """
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g)
    hits = _hit_sources(comps, g.shape, g)
    out = g.copy()
    for dst, srcs in hits.items():
        if not srcs:
            continue
        best = min(srcs, key=lambda src: _distance_key(comps[src]["bbox"], comps[dst]["bbox"]))
        coords = comps[dst]["coords"]
        out[coords[:, 0], coords[:, 1]] = comps[best]["color"]
    return out


def _can_place(
    occupied: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    mask: np.ndarray,
    top: int,
    left: int,
    forbidden: set[tuple[int, int]],
    allowed_hits: set[tuple[int, int]],
) -> bool:
    h, w = mask.shape
    H, W = occupied.shape
    if top < 1 or left < 1 or top + h > H - 1 or left + w > W - 1:
        return False
    bbox = (top, left, top + h - 1, left + w - 1)
    for other in bboxes:
        if _boxes_too_close(bbox, other, pad=1):
            return False
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        if occupied[r, c]:
            return False
        if (r, c) in forbidden and (r, c) not in allowed_hits:
            return False
    return True


def _place_mask(
    grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    local = np.argwhere(mask)
    coords = local + np.array([top, left], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color
    bbox = (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1)
    return coords, bbox


def _available_rays(
    grid: np.ndarray, source_bboxes: list[tuple[int, int, int, int]], used: set[tuple[int, int]]
) -> list[dict]:
    occupied = grid != BACKGROUND
    rays = []
    for source_id, bbox in enumerate(source_bboxes):
        for info in _ray_cells(grid.shape, occupied, bbox):
            ray_id = (source_id, info["corner_id"])
            if ray_id in used or not info["cells"]:
                continue
            rays.append(
                {
                    "ray_id": ray_id,
                    "source_id": source_id,
                    "corner_id": info["corner_id"],
                    "direction": info["direction"],
                    "cells": info["cells"],
                }
            )
    return rays


def _try_place_single_target(
    rng: random.Random,
    grid: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
    rays: list[dict],
    mask: np.ndarray,
    preferred_sources: set[int] | None = None,
) -> dict | None:
    local_cells = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    ray_order = rays[:]
    rng.shuffle(ray_order)
    if preferred_sources:
        ray_order.sort(key=lambda item: item["source_id"] not in preferred_sources)
    for ray in ray_order:
        hit_cells = ray["cells"][:]
        rng.shuffle(hit_cells)
        for hit in hit_cells:
            for anchor in rng.sample(local_cells, len(local_cells)):
                top = hit[0] - anchor[0]
                left = hit[1] - anchor[1]
                allowed = {hit}
                if not _can_place(grid != BACKGROUND, bboxes, mask, top, left, forbidden, allowed):
                    continue
                coords, bbox = _place_mask(grid, mask, -1, top, left)
                grid[coords[:, 0], coords[:, 1]] = BACKGROUND
                hit_index = ray["cells"].index(hit)
                return {
                    "bbox": bbox,
                    "coords": coords,
                    "used_rays": [(ray["ray_id"], hit_index, hit)],
                    "sources": {ray["source_id"]},
                }
    return None


def _try_place_double_target(
    rng: random.Random,
    grid: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
    rays: list[dict],
    mask: np.ndarray,
) -> dict | None:
    local_cells = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
    by_source: dict[int, list[dict]] = {}
    for ray in rays:
        by_source.setdefault(ray["source_id"], []).append(ray)
    if len(by_source) < 2:
        return None
    source_ids = sorted(by_source)
    pairs = [(a, b) for a in by_source[source_ids[0]] for b in by_source[source_ids[1]]]
    rng.shuffle(pairs)
    for ray_a, ray_b in pairs:
        hits_a = ray_a["cells"][:]
        hits_b = ray_b["cells"][:]
        rng.shuffle(hits_a)
        rng.shuffle(hits_b)
        for hit_a in hits_a:
            for hit_b in hits_b:
                for anchor_a in local_cells:
                    for anchor_b in local_cells:
                        if anchor_a == anchor_b:
                            continue
                        top = hit_a[0] - anchor_a[0]
                        left = hit_a[1] - anchor_a[1]
                        if (top, left) != (hit_b[0] - anchor_b[0], hit_b[1] - anchor_b[1]):
                            continue
                        allowed = {hit_a, hit_b}
                        if not _can_place(grid != BACKGROUND, bboxes, mask, top, left, forbidden, allowed):
                            continue
                        coords, bbox = _place_mask(grid, mask, -1, top, left)
                        grid[coords[:, 0], coords[:, 1]] = BACKGROUND
                        return {
                            "bbox": bbox,
                            "coords": coords,
                            "used_rays": [
                                (ray_a["ray_id"], ray_a["cells"].index(hit_a), hit_a),
                                (ray_b["ray_id"], ray_b["cells"].index(hit_b), hit_b),
                            ],
                            "sources": {ray_a["source_id"], ray_b["source_id"]},
                        }
    return None


def _validate_generated(grid: np.ndarray, require_double: bool) -> bool:
    comps = _find_components(grid)
    if len(comps) not in (5, 6):
        return False
    if len({comp["color"] for comp in comps}) != len(comps):
        return False
    if any(comp["area"] < 3 or comp["area"] > 6 for comp in comps):
        return False
    if any(
        comp["bbox"][0] == 0
        or comp["bbox"][1] == 0
        or comp["bbox"][2] == grid.shape[0] - 1
        or comp["bbox"][3] == grid.shape[1] - 1
        for comp in comps
    ):
        return False
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _boxes_too_close(comps[i]["bbox"], comps[j]["bbox"], pad=1):
                return False
    sources = [i for i, comp in enumerate(comps) if _has_role_symmetry(comp["mask"])]
    if len(sources) != 2:
        return False
    hits = _hit_sources(comps, grid.shape, grid)
    if not hits or any(not srcs for srcs in hits.values()):
        return False
    if len({src for srcs in hits.values() for src in srcs}) != 2:
        return False
    if require_double and not any(len(srcs) >= 2 for srcs in hits.values()):
        return False
    return not np.array_equal(main(grid), grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        H = rng.randint(14, 18)
        W = rng.randint(14, 18)
        target_count = rng.choice((3, 4))
        require_double = target_count >= 3 and rng.random() < 0.45
        grid = np.zeros((H, W), dtype=int)
        colors = rng.sample(PALETTE, 2 + target_count)

        source_masks = [_make_random_connected_mask(rng, True) for _ in range(2)]
        source_bboxes: list[tuple[int, int, int, int]] = []
        occupied = np.zeros((H, W), dtype=bool)
        ok = True
        for mask, color in zip(source_masks, colors[:2]):
            placed = False
            for _ in range(PLACEMENT_RETRIES):
                top = rng.randint(1, H - mask.shape[0] - 1)
                left = rng.randint(1, W - mask.shape[1] - 1)
                if not _can_place(occupied, source_bboxes, mask, top, left, set(), set()):
                    continue
                _place_mask(grid, mask, color, top, left)
                source_bboxes.append((top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1))
                occupied = grid != BACKGROUND
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        if len([ray for ray in _available_rays(grid, source_bboxes, set()) if ray["cells"]]) < target_count:
            continue

        bboxes = source_bboxes[:]
        forbidden: set[tuple[int, int]] = set()
        used_rays: set[tuple[int, int]] = set()
        source_hit_coverage = {0: 0, 1: 0}
        made_double = False

        for idx in range(target_count):
            mask = _make_random_connected_mask(rng, False)
            rays = _available_rays(grid, source_bboxes, used_rays)
            if not rays:
                ok = False
                break

            placement = None
            if require_double and not made_double:
                placement = _try_place_double_target(rng, grid, bboxes, forbidden, rays, mask)
                if placement is None:
                    ok = False
                    break
                made_double = True
            else:
                preferred = {src for src, count in source_hit_coverage.items() if count == 0}
                if target_count - idx <= len(preferred):
                    placement = _try_place_single_target(rng, grid, bboxes, forbidden, rays, mask, preferred)
                if placement is None:
                    placement = _try_place_single_target(rng, grid, bboxes, forbidden, rays, mask)
                if placement is None:
                    ok = False
                    break

            coords = placement["coords"]
            grid[coords[:, 0], coords[:, 1]] = colors[2 + idx]
            bboxes.append(placement["bbox"])
            for ray_id, hit_index, _ in placement["used_rays"]:
                used_rays.add(ray_id)
                ray = next(ray for ray in rays if ray["ray_id"] == ray_id)
                forbidden.update(ray["cells"][:hit_index])
            for src in placement["sources"]:
                source_hit_coverage[src] += 1

        if not ok:
            continue
        if require_double and not made_double:
            continue
        if not _validate_generated(grid, require_double):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
