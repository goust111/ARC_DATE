from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
HOST_COLORS = (1, 3)
CANDIDATE_COLORS = (2, 4, 5, 6)
AREA_HOST_MIN = 12
AREA_CANDIDATE_MAX = 6
GRID_MIN = 18
GRID_MAX = 24
GEN_ATTEMPTS = 400


HOLE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
]

CANDIDATE_LIBRARY = HOLE_SHAPES + [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    tight = _trim_mask(mask)
    return tuple(tuple(int(v) for v in row) for row in tight.astype(int))


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rotated = _trim_mask(np.rot90(mask, k))
        sig = _mask_signature(rotated)
        if sig not in seen:
            seen.add(sig)
            variants.append(rotated)
    return variants


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_mask_signature(rotated) for rotated in _rotations(mask))


def _same_up_to_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    sig = _mask_signature(_trim_mask(a))
    return any(_mask_signature(rotated) == sig for rotated in _rotations(b))


def _shape_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
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
            components.append(np.array(cells, dtype=int))
    return components


def _find_monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return components


def _extract_holes(grid: np.ndarray, host: dict) -> list[dict]:
    r0, c0, r1, c1 = host["bbox"]
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    black = window == BG
    holes = []
    for coords in _components_of_mask(black):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 1] == 0):
            continue
        if np.any(coords[:, 0] == black.shape[0] - 1) or np.any(coords[:, 1] == black.shape[1] - 1):
            continue
        abs_coords = coords + np.array([r0, c0])
        mask = np.zeros(black.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        holes.append(
            {
                "host_color": host["color"],
                "host_bbox": host["bbox"],
                "coords": abs_coords,
                "bbox": _bbox_coords(abs_coords),
                "shape": _trim_mask(mask),
            }
        )
    holes.sort(key=lambda hole: (hole["host_bbox"][0], hole["host_bbox"][1], hole["bbox"][0], hole["bbox"][1]))
    return holes


def _analyze_grid(grid: np.ndarray) -> dict:
    components = _find_monochrome_components(grid, bg=BG)
    hosts = [comp for comp in components if comp["area"] >= AREA_HOST_MIN]
    candidates = [comp for comp in components if comp["area"] <= AREA_CANDIDATE_MAX]
    middling = [comp for comp in components if AREA_CANDIDATE_MAX < comp["area"] < AREA_HOST_MIN]
    holes: list[dict] = []
    for host in hosts:
        holes.extend(_extract_holes(grid, host))
    for candidate in candidates:
        coords = candidate["coords"]
        rel = coords - coords.min(axis=0)
        mask = np.zeros((rel[:, 0].max() + 1, rel[:, 1].max() + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        candidate["shape"] = _trim_mask(mask)
        matches = []
        for hole_idx, hole in enumerate(holes):
            if _shape_area(hole["shape"]) != candidate["area"]:
                continue
            if any(np.array_equal(rot, hole["shape"]) for rot in _rotations(candidate["shape"])):
                matches.append(hole_idx)
        candidate["possible_holes"] = matches
    candidates.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return {"components": components, "hosts": hosts, "candidates": candidates, "holes": holes, "middling": middling}


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Swap each small candidate object's color with the color of the large host cavity it rotationally fits.

    Input structure:
    - The grid uses black as background and contains several monochromatic 4-connected foreground components.
    - Large components (area at least 12) are host bodies; small components (area at most 6) are candidate objects.
    - Each host body may contain one or more enclosed black 4-connected hole regions inside its own bounding box.

    Rule:
    - Detect every monochromatic foreground component and classify it by area threshold alone.
    - For each large host, collect black components inside its bounding box that do not touch that box border; these are the holes.
    - For each small candidate, try the four right-angle rotations and match it to the first still-unused hole whose shape is exactly identical.
    - When a match is found, recolor the candidate's original pixels to the host body's color and recolor the hole pixels to the candidate's original color.

    Key insight:
    - Area decides which objects are movable candidates and which objects are cavity-bearing hosts, and the successful operation is a color swap between the candidate's original location and the matched hole.
    """
    grid = np.asarray(input_grid, dtype=int)
    analysis = _analyze_grid(grid)
    out = grid.copy()
    used_holes: set[int] = set()
    for candidate in analysis["candidates"]:
        chosen_hole = None
        for hole_idx in candidate["possible_holes"]:
            if hole_idx not in used_holes:
                chosen_hole = hole_idx
                break
        if chosen_hole is None:
            continue
        hole = analysis["holes"][chosen_hole]
        used_holes.add(chosen_hole)
        out[candidate["coords"][:, 0], candidate["coords"][:, 1]] = int(hole["host_color"])
        out[hole["coords"][:, 0], hole["coords"][:, 1]] = int(candidate["color"])
    return out


def _random_hole_pair() -> tuple[np.ndarray, np.ndarray]:
    by_area: dict[int, list[np.ndarray]] = {}
    for shape in HOLE_SHAPES:
        by_area.setdefault(_shape_area(shape), []).append(shape)

    prefer_same_area = random.random() < 0.55
    if prefer_same_area:
        same_area_groups = []
        for group in by_area.values():
            unique = []
            seen = set()
            for shape in group:
                sig = _canonical_shape(shape)
                if sig not in seen:
                    seen.add(sig)
                    unique.append(shape)
            if len(unique) >= 2:
                same_area_groups.append(unique)
        if same_area_groups:
            group = random.choice(same_area_groups)
            a, b = random.sample(group, 2)
            if not _same_up_to_rotation(a, b):
                return a.copy(), b.copy()

    while True:
        a, b = random.sample(HOLE_SHAPES, 2)
        if not _same_up_to_rotation(a, b):
            return a.copy(), b.copy()


def _build_host_patch(hole_mask: np.ndarray, color: int) -> dict | None:
    hole_mask = _trim_mask(hole_mask)
    hh, hw = hole_mask.shape
    hole_cells = {(r + 1, c + 1) for r, c in np.argwhere(hole_mask)}
    object_cells = {(r, c) for r in range(hh + 2) for c in range(hw + 2)} - hole_cells
    base_area = len(object_cells)
    target_min = max(15, base_area + 1)
    target_max = min(25, base_area + 4)
    if target_min > target_max:
        return None
    target_area = random.randint(target_min, target_max)
    coords = set(object_cells)

    while len(coords) < target_area:
        frontier = set()
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in coords or (nr, nc) in hole_cells:
                    continue
                rows = [cell[0] for cell in coords] + [nr]
                cols = [cell[1] for cell in coords] + [nc]
                if max(rows) - min(rows) + 1 <= 7 and max(cols) - min(cols) + 1 <= 7:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        coords.add(random.choice(sorted(frontier)))

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, min_c = min(rows), min(cols)
    norm_cells = {(r - min_r, c - min_c) for r, c in coords}
    norm_hole = {(r - min_r, c - min_c) for r, c in hole_cells}
    h = max(r for r, _ in norm_cells) + 1
    w = max(c for _, c in norm_cells) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm_cells:
        patch[r, c] = color
    return {"patch": patch, "hole_cells": norm_hole, "shape": hole_mask.copy()}


def _rotate_shape(mask: np.ndarray, k: int) -> np.ndarray:
    return _trim_mask(np.rot90(mask, k))


def _candidate_options(hole_shapes: list[np.ndarray]) -> list[np.ndarray]:
    hole_sigs = {_canonical_shape(shape) for shape in hole_shapes}
    preferred_areas = {_shape_area(shape) for shape in hole_shapes}
    preferred = [
        shape.copy()
        for shape in CANDIDATE_LIBRARY
        if _shape_area(shape) in preferred_areas and _canonical_shape(shape) not in hole_sigs
    ]
    fallback = [shape.copy() for shape in CANDIDATE_LIBRARY if _canonical_shape(shape) not in hole_sigs]
    return preferred or fallback


def _shape_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _can_place(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    gh, gw = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    return not np.any(blocked[r0 : r0 + h, c0 : c0 + w])


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    r1 = min(blocked.shape[0], r0 + h + 1)
    c1 = min(blocked.shape[1], c0 + w + 1)
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    blocked[rr0:r1, cc0:c1] = True


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _place_patches(grid: np.ndarray, items: list[dict]) -> bool:
    blocked = np.zeros(grid.shape, dtype=bool)
    ordered = sorted(items, key=lambda item: (item["patch"].shape[0] * item["patch"].shape[1]), reverse=True)
    for item in ordered:
        patch = item["patch"]
        h, w = patch.shape
        positions = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1) if _can_place(blocked, patch, r, c)]
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        _paste(grid, patch, r0, c0)
        _mark_blocked(blocked, patch, r0, c0)
        item["pos"] = (r0, c0)
    return True


def _expected_output_from_analysis(grid: np.ndarray, analysis: dict) -> np.ndarray:
    expected = grid.copy()
    used_holes: set[int] = set()
    for candidate in analysis["candidates"]:
        chosen_hole = None
        for hole_idx in candidate["possible_holes"]:
            if hole_idx not in used_holes:
                chosen_hole = hole_idx
                break
        if chosen_hole is None:
            continue
        hole = analysis["holes"][chosen_hole]
        used_holes.add(chosen_hole)
        expected[candidate["coords"][:, 0], candidate["coords"][:, 1]] = int(hole["host_color"])
        expected[hole["coords"][:, 0], hole["coords"][:, 1]] = int(candidate["color"])
    return expected


def _validate_instance(grid: np.ndarray) -> bool:
    analysis = _analyze_grid(grid)
    if len(analysis["hosts"]) != 2 or len(analysis["holes"]) != 2:
        return False
    if len(analysis["candidates"]) not in (3, 4):
        return False
    if analysis["middling"]:
        return False
    if {host["color"] for host in analysis["hosts"]} != set(HOST_COLORS):
        return False

    hole_shapes = [_canonical_shape(hole["shape"]) for hole in analysis["holes"]]
    if hole_shapes[0] == hole_shapes[1]:
        return False

    matched = [cand for cand in analysis["candidates"] if cand["possible_holes"]]
    unmatched = [cand for cand in analysis["candidates"] if not cand["possible_holes"]]
    if len(matched) != 2 or len(unmatched) == 0:
        return False
    if any(len(cand["possible_holes"]) != 1 for cand in matched):
        return False
    if len({cand["possible_holes"][0] for cand in matched}) != 2:
        return False

    out = main(grid)
    expected = _expected_output_from_analysis(grid, analysis)
    if not np.array_equal(out, expected):
        return False

    for cand in unmatched:
        vals = out[cand["coords"][:, 0], cand["coords"][:, 1]]
        if not np.all(vals == cand["color"]):
            return False

    for cand in matched:
        hole = analysis["holes"][cand["possible_holes"][0]]
        cand_vals = out[cand["coords"][:, 0], cand["coords"][:, 1]]
        hole_vals = out[hole["coords"][:, 0], hole["coords"][:, 1]]
        if not np.all(cand_vals == hole["host_color"]):
            return False
        if not np.all(hole_vals == cand["color"]):
            return False
        if np.array_equal(cand_vals, np.full(cand_vals.shape, cand["color"], dtype=int)):
            return False
    return True


def _build_instance() -> np.ndarray | None:
    grid = np.zeros((random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX)), dtype=int)
    hole_a, hole_b = _random_hole_pair()
    host_defs = []
    for shape, color in zip((hole_a, hole_b), HOST_COLORS):
        host = _build_host_patch(shape, color)
        if host is None:
            return None
        host_defs.append(host)

    candidate_count = random.randint(3, 4)
    candidate_colors = random.sample(list(CANDIDATE_COLORS), candidate_count)
    items: list[dict] = []
    for host in host_defs:
        items.append({"patch": host["patch"], "role": "host"})

    matched_masks = [_rotate_shape(hole_a, random.randint(0, 3)), _rotate_shape(hole_b, random.randint(0, 3))]
    random.shuffle(matched_masks)
    for idx, mask in enumerate(matched_masks):
        items.append({"patch": _shape_to_patch(mask, candidate_colors[idx]), "role": "matched"})

    nonmatch_count = candidate_count - 2
    options = _candidate_options([hole_a, hole_b])
    if len(options) < nonmatch_count:
        return None
    random.shuffle(options)
    for i in range(nonmatch_count):
        mask = _rotate_shape(options[i], random.randint(0, 3))
        items.append({"patch": _shape_to_patch(mask, candidate_colors[2 + i]), "role": "unmatched"})

    if not _place_patches(grid, items):
        return None
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
