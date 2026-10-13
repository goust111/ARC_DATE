from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(cells, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.asarray(coords, dtype=int)
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, bbox


def _hole_components_from_coords(coords: np.ndarray) -> list[np.ndarray]:
    mask, bbox = _coords_to_mask(coords)
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    holes_mask = (~mask) & (~outside)
    r0, c0, _, _ = bbox
    holes: list[np.ndarray] = []
    for part in _mask_components(holes_mask):
        holes.append(part + np.array([r0, c0], dtype=int))
    return holes


def _find_monochromatic_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            holes = _hole_components_from_coords(coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                    "holes": holes,
                    "hole_area": int(sum(hole.shape[0] for hole in holes)),
                }
            )
    return objects


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    objects = _find_monochromatic_objects(grid)
    rings = [obj for obj in objects if obj["hole_area"] > 0]
    blocks = [obj for obj in objects if obj["hole_area"] == 0]
    return {"objects": objects, "rings": rings, "blocks": blocks}


def _solve_from_analysis(shape: tuple[int, int], analysis: dict[str, object]) -> np.ndarray:
    rings = list(analysis["rings"])
    blocks = list(analysis["blocks"])
    assert rings and len(rings) == len(blocks), "expected equally many rings and paint blocks"

    ring_map: dict[int, dict[str, object]] = {}
    for ring in rings:
        border_area = int(ring["area"])
        assert border_area not in ring_map, "ring border areas must be unique"
        ring_map[border_area] = ring

    block_map: dict[int, dict[str, object]] = {}
    for block in blocks:
        area = int(block["area"])
        assert area not in block_map, "paint block areas must be unique"
        block_map[area] = block

    assert set(ring_map) == set(block_map), "paint amount must match ring border length exactly"

    out = np.zeros(shape, dtype=int)
    for area, ring in ring_map.items():
        new_color = int(block_map[area]["color"])
        coords = np.asarray(ring["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = new_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each hollow ring's border with the uniquely area-matched paint block color and erase the spent block.

    Input structure:
    - Black background contains several monochromatic 4-connected foreground objects of two kinds.
    - Hollow rings are closed single-color border components with one enclosed black hole that does not touch the grid edge.
    - Paint-can blocks are solid single-color components with no enclosed hole, and object colors are all distinct.

    Rule:
    - For every solid block, count its cells; for every hollow ring, count only the colored border cells, not the hole capacity.
    - Pair blocks to rings by exact equality between block area and ring border-cell count; the pairing is one-to-one.
    - Recolor each ring border to its matched block color, keep every hole black, and remove all paint blocks.

    Key insight:
    - The conserved quantity is paint used on the ring border, so equal hole capacity is a decoy and never gets filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    analysis = _analyze_grid(g)
    return _solve_from_analysis(g.shape, analysis)


def _random_connected_shape(
    area: int,
    *,
    min_dim: int,
    max_dim: int,
    max_attempts: int = 120,
) -> np.ndarray | None:
    candidates = [
        (h, w)
        for h in range(min_dim, max_dim + 1)
        for w in range(min_dim, max_dim + 1)
        if h * w >= area
    ]
    random.shuffle(candidates)
    for h, w in candidates:
        for _ in range(max_attempts):
            start = (random.randrange(h), random.randrange(w))
            cells = {start}
            frontier: set[tuple[int, int]] = set()
            for nr, nc in _neighbors4(*start):
                if 0 <= nr < h and 0 <= nc < w:
                    frontier.add((nr, nc))
            while len(cells) < area and frontier:
                cell = random.choice(tuple(frontier))
                frontier.remove(cell)
                cells.add(cell)
                for nr, nc in _neighbors4(*cell):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if len(cells) != area:
                continue
            coords = _normalize(np.asarray(sorted(cells), dtype=int))
            if _hole_components_from_coords(coords):
                continue
            return coords
    return None


def _render_patch(coords: np.ndarray, color: int) -> np.ndarray:
    coords = _normalize(coords)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[coords[:, 0], coords[:, 1]] = int(color)
    return patch


def _make_ring(color: int, hole_area: int, *, compact: bool) -> dict[str, object] | None:
    max_dim = 5 if hole_area <= 8 else 6
    for _ in range(180):
        inner = _random_connected_shape(hole_area, min_dim=2, max_dim=max_dim)
        if inner is None:
            continue
        ih = int(inner[:, 0].max()) + 1
        iw = int(inner[:, 1].max()) + 1

        outer = {(r, c) for r in range(ih + 2) for c in range(iw + 2)}
        inner_shifted = {(int(r) + 1, int(c) + 1) for r, c in inner}

        extra_min = 1
        extra_max = 3 if compact else 6
        for _extra in range(random.randint(extra_min, extra_max)):
            frontier = []
            for r, c in outer:
                for dr, dc in DIR4:
                    cand = (r + dr, c + dc)
                    if cand not in outer:
                        frontier.append(cand)
            if not frontier:
                break
            outer.add(random.choice(frontier))

        outer_arr = np.asarray(sorted(outer), dtype=int)
        shift = outer_arr.min(axis=0)
        outer_arr -= shift
        inner_arr = np.asarray(sorted(inner_shifted), dtype=int) - shift

        oh = int(outer_arr[:, 0].max()) + 1
        ow = int(outer_arr[:, 1].max()) + 1
        if max(oh, ow) > (7 if compact else 9):
            continue

        outer_mask = np.zeros((oh, ow), dtype=bool)
        outer_mask[outer_arr[:, 0], outer_arr[:, 1]] = True
        inner_mask = np.zeros((oh, ow), dtype=bool)
        inner_mask[inner_arr[:, 0], inner_arr[:, 1]] = True
        ring_mask = outer_mask & (~inner_mask)
        ring_coords = np.argwhere(ring_mask)
        if len(_mask_components(ring_mask)) != 1:
            continue

        holes = _hole_components_from_coords(ring_coords)
        if len(holes) != 1 or int(holes[0].shape[0]) != hole_area:
            continue

        border_area = int(ring_coords.shape[0])
        if compact and not (10 <= border_area <= 18):
            continue
        if not compact and not (border_area <= 28):
            continue

        patch = np.zeros((oh, ow), dtype=int)
        patch[ring_mask] = int(color)
        return {
            "kind": "ring",
            "color": int(color),
            "patch": patch,
            "footprint": outer_mask,
            "ring_coords": ring_coords,
            "area": border_area,
            "hole_area": hole_area,
            "center": tuple(np.mean(ring_coords, axis=0)),
        }
    return None


def _make_block(color: int, area: int) -> dict[str, object] | None:
    max_dim = 6 if area <= 18 else 7
    for _ in range(180):
        coords = _random_connected_shape(area, min_dim=1, max_dim=max_dim)
        if coords is None:
            continue
        if _hole_components_from_coords(coords):
            continue
        patch = _render_patch(coords, color)
        if max(patch.shape) > 8:
            continue
        return {
            "kind": "block",
            "color": int(color),
            "patch": patch,
            "footprint": patch != BG,
            "area": int(area),
            "center": tuple(np.mean(coords, axis=0)),
        }
    return None


def _can_place(reserved: np.ndarray, footprint: np.ndarray, top: int, left: int) -> bool:
    h, w = reserved.shape
    ph, pw = footprint.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    rr, cc = np.nonzero(footprint)
    return not np.any(reserved[rr + top, cc + left])


def _place_patch(canvas: np.ndarray, reserved: np.ndarray, obj: dict[str, object], top: int, left: int) -> None:
    patch = np.asarray(obj["patch"], dtype=int)
    footprint = np.asarray(obj["footprint"], dtype=bool)
    ph, pw = patch.shape
    sub = canvas[top : top + ph, left : left + pw]
    mask = patch != BG
    sub[mask] = patch[mask]

    rr, cc = np.nonzero(footprint)
    for r, c in zip(rr + top, cc + left):
        for dr, dc in DIR8 + [(0, 0)]:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1]:
                reserved[nr, nc] = True

    obj["top_left"] = (int(top), int(left))


def _placed_center(obj: dict[str, object]) -> tuple[float, float]:
    top, left = obj["top_left"]
    patch = np.asarray(obj["patch"], dtype=int)
    rr, cc = np.nonzero(patch != BG)
    return float(top + rr.mean()), float(left + cc.mean())


def _try_place_objects(objects: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    canvas = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    ordered = sorted(objects, key=lambda obj: -int(np.count_nonzero(obj["footprint"])))
    for obj in ordered:
        footprint = np.asarray(obj["footprint"], dtype=bool)
        ph, pw = footprint.shape
        candidates = [
            (r, c)
            for r in range(h - ph + 1)
            for c in range(w - pw + 1)
            if _can_place(reserved, footprint, r, c)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        _place_patch(canvas, reserved, obj, top, left)
    return canvas


def _expected_output_from_generated(objects: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    blocks_by_area = {int(obj["area"]): obj for obj in objects if obj["kind"] == "block"}
    for ring in [obj for obj in objects if obj["kind"] == "ring"]:
        top, left = ring["top_left"]
        patch = np.asarray(ring["patch"], dtype=int)
        rr, cc = np.nonzero(patch != BG)
        color = int(blocks_by_area[int(ring["area"])]["color"])
        out[rr + top, cc + left] = color
    return out


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    dr = a[0] - b[0]
    dc = a[1] - b[1]
    return dr * dr + dc * dc


def _semantic_checks(grid: np.ndarray, objects: list[dict[str, object]], expected: np.ndarray) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    analysis = _analyze_grid(grid)
    rings = list(analysis["rings"])
    blocks = list(analysis["blocks"])
    if len(rings) < 2 or len(rings) != len(blocks):
        return False
    if len(rings) != len(objects) // 2:
        return False

    colors = [int(obj["color"]) for obj in analysis["objects"]]
    if len(set(colors)) != len(colors):
        return False

    ring_areas = [int(ring["area"]) for ring in rings]
    block_areas = [int(block["area"]) for block in blocks]
    if len(set(ring_areas)) != len(ring_areas):
        return False
    if len(set(block_areas)) != len(block_areas):
        return False
    if set(ring_areas) != set(block_areas):
        return False

    if any(len(ring["holes"]) != 1 for ring in rings):
        return False
    if any(block["hole_area"] != 0 for block in blocks):
        return False

    area_to_ring = {int(ring["area"]): ring for ring in rings}
    decoy_pairs = [
        (area, ring)
        for area in ring_areas
        for ring in rings
        if int(ring["hole_area"]) == area and area_to_ring[area] is not ring
    ]
    if not decoy_pairs:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    for area, ring in decoy_pairs:
        hole = np.asarray(ring["holes"][0], dtype=int)
        if np.any(actual[hole[:, 0], hole[:, 1]] != BG):
            return False

    ring_centers = {int(ring["area"]): tuple(np.mean(np.asarray(ring["coords"]), axis=0)) for ring in rings}
    block_centers = {int(block["area"]): tuple(np.mean(np.asarray(block["coords"]), axis=0)) for block in blocks}
    nearest_matches = 0
    for area, b_center in block_centers.items():
        nearest_area = min(ring_centers, key=lambda ra: _distance_sq(b_center, ring_centers[ra]))
        nearest_matches += int(nearest_area == area)
    if nearest_matches == len(rings):
        return False

    rings_left = [int(r["area"]) for r in sorted(rings, key=lambda r: (int(r["bbox"][1]), int(r["bbox"][0])))]
    blocks_left = [int(b["area"]) for b in sorted(blocks, key=lambda b: (int(b["bbox"][1]), int(b["bbox"][0])))]
    if rings_left == blocks_left:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    ring_count = random.choice([2, 3, 3, 4])
    colors = random.sample(PALETTE, ring_count * 2)
    ring_colors = colors[:ring_count]
    block_colors = colors[ring_count:]

    rings: list[dict[str, object]] = []
    used_border_areas: set[int] = set()

    for idx in range(ring_count - 1):
        made = None
        for _ in range(150):
            hole_area = random.randint(2, 7 if ring_count >= 3 else 8)
            candidate = _make_ring(ring_colors[idx], hole_area, compact=True)
            if candidate is None:
                continue
            border_area = int(candidate["area"])
            if border_area in used_border_areas:
                continue
            used_border_areas.add(border_area)
            made = candidate
            break
        if made is None:
            return None
        rings.append(made)

    decoy_area = random.choice([int(ring["area"]) for ring in rings])
    final_ring = None
    for _ in range(200):
        candidate = _make_ring(ring_colors[-1], decoy_area, compact=False)
        if candidate is None:
            continue
        border_area = int(candidate["area"])
        if border_area in used_border_areas:
            continue
        final_ring = candidate
        used_border_areas.add(border_area)
        break
    if final_ring is None:
        return None
    rings.append(final_ring)
    random.shuffle(rings)

    blocks: list[dict[str, object]] = []
    for color, ring in zip(block_colors, rings):
        block = _make_block(color, int(ring["area"]))
        if block is None:
            return None
        blocks.append(block)
    random.shuffle(blocks)

    objects = rings + blocks
    random.shuffle(objects)

    for _ in range(80):
        shape = (random.randint(20, 30), random.randint(20, 30))
        placed = [dict(obj) for obj in objects]
        grid = _try_place_objects(placed, shape)
        if grid is None:
            continue
        expected = _expected_output_from_generated(placed, shape)
        if _semantic_checks(grid, placed, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paint-ring ARC input")
