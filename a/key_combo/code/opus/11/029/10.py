from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
OBJECT_COLOR = 5
ODD_OUTLINE_COLOR = 2
EVEN_OUTLINE_COLOR = 1
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 18
GRID_MAX = 26
MAX_HOLES = 3
MAX_GENERATION_ATTEMPTS = 400
OBJECT_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200
WORK_CANVAS = 15
TARGET_AREA_MIN = 15
TARGET_AREA_MAX = 30


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0].copy()
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _background_components(mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    exterior = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(cells, dtype=int)
            comps.append(arr)
            if touches_border:
                exterior[arr[:, 0], arr[:, 1]] = True
    return comps, exterior


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    pad = np.pad(mask.astype(bool), 1, constant_values=False)
    comps, exterior = _background_components(pad)
    holes: list[np.ndarray] = []
    for comp in comps:
        if exterior[comp[0, 0], comp[0, 1]]:
            continue
        shifted = comp - 1
        keep = (
            (0 <= shifted[:, 0])
            & (shifted[:, 0] < mask.shape[0])
            & (0 <= shifted[:, 1])
            & (shifted[:, 1] < mask.shape[1])
        )
        holes.append(shifted[keep])
    return holes


def _hole_count(mask: np.ndarray) -> int:
    return len(_hole_components(mask))


def _exterior_background(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask.astype(bool), 1, constant_values=False)
    _comps, exterior = _background_components(pad)
    return exterior[1:-1, 1:-1]


def _outline_distance_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = _exterior_background(mask)
    dist = np.full((h, w), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if not exterior[r, c]:
                continue
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    dist[r, c] = 1
                    queue.append((r, c))
                    break

    while queue:
        r, c = queue.popleft()
        nd = dist[r, c] + 1
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and exterior[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                queue.append((nr, nc))

    return dist


def _analyze_object(mask: np.ndarray) -> dict:
    comps = _components(mask)
    assert len(comps) == 1
    coords = comps[0]
    holes = _hole_components(mask)
    return {
        "coords": coords,
        "bbox": _bbox_from_coords(coords),
        "area": int(mask.sum()),
        "hole_count": len(holes),
        "hole_sizes": sorted(int(len(hole)) for hole in holes),
    }


def _solve_with_options(
    grid: np.ndarray,
    *,
    fixed_thickness: int | None = None,
    constant_color: int | None = None,
    include_holes: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    object_mask = g == OBJECT_COLOR
    components = _components(object_mask)

    for coords in components:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        hole_count = _hole_count(object_mask[r0 : r1 + 1, c0 : c1 + 1])
        thickness = hole_count if fixed_thickness is None else fixed_thickness
        color = (
            ODD_OUTLINE_COLOR
            if hole_count % 2 == 1
            else EVEN_OUTLINE_COLOR
        )
        if constant_color is not None:
            color = constant_color
        if thickness <= 0:
            continue

        patch_h = (r1 - r0 + 1) + 2 * (thickness + 1)
        patch_w = (c1 - c0 + 1) + 2 * (thickness + 1)
        patch = np.zeros((patch_h, patch_w), dtype=bool)
        roff = thickness + 1
        coff = thickness + 1
        for rr, cc in coords:
            patch[roff + rr - r0, coff + cc - c0] = True

        if include_holes:
            bg_mask = ~patch
            dist = np.full(patch.shape, -1, dtype=int)
            queue: deque[tuple[int, int]] = deque()
            for rr in range(patch.shape[0]):
                for cc in range(patch.shape[1]):
                    if not bg_mask[rr, cc]:
                        continue
                    for dr, dc in DIR4:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and patch[nr, nc]:
                            dist[rr, cc] = 1
                            queue.append((rr, cc))
                            break
            while queue:
                rr, cc = queue.popleft()
                nd = dist[rr, cc] + 1
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1] and bg_mask[nr, nc] and dist[nr, nc] < 0:
                        dist[nr, nc] = nd
                        queue.append((nr, nc))
        else:
            dist = _outline_distance_map(patch)

        paint = (dist >= 1) & (dist <= thickness)
        for rr, cc in np.argwhere(paint):
            gr = r0 + (rr - roff)
            gc = c0 + (cc - coff)
            if 0 <= gr < g.shape[0] and 0 <= gc < g.shape[1] and out[gr, gc] == BACKGROUND:
                out[gr, gc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each gray object's enclosed holes, then draw an outward outline whose thickness equals that count while preserving the holes.

    Input structure:
    - The grid is a black background containing 2 or 3 separated gray 4-connected objects.
    - Each gray object is a closed irregular curve of thickness 1 or 2 that encloses 1 to 3 black holes.
    - Different objects may enclose different numbers of holes, and the objects are spaced so their outward outlines do not interfere.

    Rule:
    - Keep every original gray object pixel unchanged.
    - For each gray 4-connected component, count how many enclosed black 4-connected holes it contains.
    - Outline only the exterior of that object by exactly that many 4-neighbor layers; leave internal hole pixels black.
    - Use red for odd hole counts and blue for even hole counts.

    Key insight:
    - The same topological property controls two visible effects at once: hole count sets the outline thickness, and hole-count parity sets the outline color.
    """
    return _solve_with_options(np.asarray(grid, dtype=int))


def _random_polyomino(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}

    if random.random() < 0.5:
        max_r = max(r for r, _ in norm)
        norm = {(max_r - r, c) for r, c in norm}
    if random.random() < 0.5:
        max_c = max(c for _, c in norm)
        norm = {(r, max_c - c) for r, c in norm}
    if random.random() < 0.5:
        norm = {(c, r) for r, c in norm}

    min_r = min(r for r, _ in norm)
    min_c = min(c for _, c in norm)
    return {(r - min_r, c - min_c) for r, c in norm}


def _build_base_mask(hole_count: int) -> np.ndarray:
    canvas = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=bool)
    poly = _random_polyomino(hole_count)
    max_r = max(r for r, _ in poly)
    max_c = max(c for _, c in poly)
    start_r = (WORK_CANVAS - (2 * max_r + 3)) // 2
    start_c = (WORK_CANVAS - (2 * max_c + 3)) // 2
    for pr, pc in poly:
        cr = start_r + 1 + 2 * pr
        cc = start_c + 1 + 2 * pc
        canvas[cr - 1 : cr + 2, cc - 1 : cc + 2] = True
        canvas[cr, cc] = False
    return canvas


def _exterior_neighbor_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    exterior = _exterior_background(mask)
    candidates: set[tuple[int, int]] = set()
    for r, c in np.argwhere(exterior):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                candidates.add((rr, cc))
                break
    return list(candidates)


def _candidate_removals(mask: np.ndarray) -> list[tuple[int, int]]:
    exterior = _exterior_background(mask)
    removable: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if any(
            0 <= rr + dr < mask.shape[0]
            and 0 <= cc + dc < mask.shape[1]
            and exterior[rr + dr, cc + dc]
            for dr, dc in DIR4
        ):
            removable.append((rr, cc))
    return removable


def _try_add(mask: np.ndarray, hole_count: int) -> bool:
    candidates = _exterior_neighbor_candidates(mask)
    random.shuffle(candidates)
    for r, c in candidates:
        trial = mask.copy()
        trial[r, c] = True
        if trial.sum() > TARGET_AREA_MAX:
            continue
        if len(_components(trial)) != 1:
            continue
        if _hole_count(trial) != hole_count:
            continue
        mask[:, :] = trial
        return True
    return False


def _try_remove(mask: np.ndarray, hole_count: int) -> bool:
    candidates = _candidate_removals(mask)
    random.shuffle(candidates)
    for r, c in candidates:
        trial = mask.copy()
        trial[r, c] = False
        if trial.sum() < TARGET_AREA_MIN:
            continue
        if len(_components(trial)) != 1:
            continue
        if _hole_count(trial) != hole_count:
            continue
        mask[:, :] = trial
        return True
    return False


def _make_object(hole_count: int, target_area: int) -> np.ndarray:
    for _ in range(OBJECT_ATTEMPTS):
        mask = _build_base_mask(hole_count)
        if len(_components(mask)) != 1 or _hole_count(mask) != hole_count:
            continue

        steps = random.randint(10, 30)
        for _step in range(steps):
            area = int(mask.sum())
            if area < target_area:
                if not _try_add(mask, hole_count):
                    break
            elif area > target_area:
                if not _try_remove(mask, hole_count):
                    break
            else:
                if random.random() < 0.65:
                    _try_add(mask, hole_count)
                else:
                    _try_remove(mask, hole_count)

        while int(mask.sum()) < TARGET_AREA_MIN:
            if not _try_add(mask, hole_count):
                break
        while int(mask.sum()) > TARGET_AREA_MAX:
            if not _try_remove(mask, hole_count):
                break

        area = int(mask.sum())
        trimmed = _trim(mask)
        if not (TARGET_AREA_MIN <= area <= TARGET_AREA_MAX):
            continue
        if len(_components(trimmed)) != 1:
            continue
        if _hole_count(trimmed) != hole_count:
            continue
        if trimmed.shape[0] > 12 or trimmed.shape[1] > 12:
            continue
        if not np.any(trimmed.sum(axis=0) >= 3) or not np.any(trimmed.sum(axis=1) >= 3):
            continue
        return trimmed.astype(int)
    raise RuntimeError(f"failed to build object with {hole_count} holes")


def _place_objects(
    grid_shape: tuple[int, int],
    objects: list[np.ndarray],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    placements: list[tuple[int, int]] = []
    padded_boxes: list[tuple[int, int, int, int]] = []
    for obj in objects:
        oh, ow = obj.shape
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, h - oh)
            c0 = random.randint(0, w - ow)
            box = (
                max(0, r0 - MAX_HOLES),
                max(0, c0 - MAX_HOLES),
                min(h - 1, r0 + oh - 1 + MAX_HOLES),
                min(w - 1, c0 + ow - 1 + MAX_HOLES),
            )
            overlaps = False
            for br0, bc0, br1, bc1 in padded_boxes:
                if not (box[2] < br0 or br1 < box[0] or box[3] < bc0 or bc1 < box[1]):
                    overlaps = True
                    break
            if overlaps:
                continue
            placements.append((r0, c0))
            padded_boxes.append(box)
            break
        else:
            raise RuntimeError("failed to place objects without outline overlap")
    return placements


def _object_stats(grid: np.ndarray) -> list[dict]:
    mask = grid == OBJECT_COLOR
    stats = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        submask = mask[r0 : r1 + 1, c0 : c1 + 1]
        stats.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": int(submask.sum()),
                "hole_count": _hole_count(submask),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )
    return stats


def _nontrivial_area_relation(stats: list[dict]) -> bool:
    for i, a in enumerate(stats):
        for j, b in enumerate(stats):
            if i == j:
                continue
            if a["hole_count"] < b["hole_count"] and a["area"] >= b["area"]:
                return True
            if a["hole_count"] < b["hole_count"] and a["bbox_area"] >= b["bbox_area"]:
                return True
    return False


def _validate_sample(grid: np.ndarray) -> None:
    stats = sorted(_object_stats(grid), key=lambda item: item["hole_count"])
    hole_counts = [item["hole_count"] for item in stats]
    assert len(stats) in (2, 3)
    assert all(1 <= h <= 3 for h in hole_counts)
    assert 2 in hole_counts
    assert len(set(hole_counts)) == len(hole_counts)
    assert _nontrivial_area_relation(stats)

    out = main(grid)
    assert np.any(out == ODD_OUTLINE_COLOR)
    assert np.any(out == EVEN_OUTLINE_COLOR)

    fixed = _solve_with_options(grid, fixed_thickness=2)
    constant = _solve_with_options(grid, constant_color=ODD_OUTLINE_COLOR)
    fill_holes = _solve_with_options(grid, include_holes=True)
    assert not np.array_equal(out, fixed)
    assert not np.array_equal(out, constant)
    assert not np.array_equal(out, fill_holes)

    for item in stats:
        r0, c0, r1, c1 = item["bbox"]
        sub_in = grid[r0 : r1 + 1, c0 : c1 + 1]
        sub_out = out[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(sub_in == OBJECT_COLOR, sub_out == OBJECT_COLOR)
        holes = _hole_components(sub_in == OBJECT_COLOR)
        assert holes
        for hole in holes:
            assert np.all(sub_out[hole[:, 0], hole[:, 1]] == BACKGROUND)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_objects = 3 if random.random() < 0.6 else 2
        if n_objects == 3:
            hole_counts = [1, 2, 3]
        else:
            hole_counts = [1, 2] if random.random() < 0.5 else [2, 3]
        random.shuffle(hole_counts)

        while True:
            target_areas = random.sample(range(TARGET_AREA_MIN, TARGET_AREA_MAX + 1), len(hole_counts))
            paired = sorted(zip(hole_counts, target_areas))
            if any(a0 < a1 and t0 >= t1 for (a0, t0), (a1, t1) in zip(paired, paired[1:])):
                break

        try:
            objects = [_make_object(hole_count, target_area) for hole_count, target_area in zip(hole_counts, target_areas)]
        except RuntimeError:
            continue

        grid_h = random.randint(GRID_MIN, GRID_MAX)
        grid_w = random.randint(GRID_MIN, GRID_MAX)
        if n_objects == 3:
            grid_h = max(grid_h, 21)
            grid_w = max(grid_w, 21)

        try:
            placements = _place_objects((grid_h, grid_w), objects)
        except RuntimeError:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        for obj, (r0, c0) in zip(objects, placements):
            oh, ow = obj.shape
            grid[r0 : r0 + oh, c0 : c0 + ow][obj.astype(bool)] = OBJECT_COLOR

        try:
            _validate_sample(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input")

