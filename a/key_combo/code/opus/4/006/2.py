import random
from collections import deque

import numpy as np


BG = 0
GREY = 5

# The prompt asks for 4-6 objects but only names four marker colors.
# The generator extends the visible marker palette so every object can still
# keep a unique internal marker when 5-6 objects are sampled.
MARKER_COLORS = (1, 2, 3, 4, 6, 8)

GRID_SIZE_RANGE = (14, 22)
OBJECT_COUNT_RANGE = (4, 6)
OBJECT_AREA_RANGE = (3, 12)
MAX_SHAPE_SPAN = 6
GEN_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_bbox(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_perimeter(shape: tuple[tuple[int, int], ...]) -> int:
    cells = set(shape)
    exposed = 0
    for r, c in cells:
        for nbr in _neighbors4(r, c):
            if nbr not in cells:
                exposed += 1
    return exposed


def _all_transforms(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    coords = list(shape)
    out = []
    for _ in range(4):
        out.append(_normalize(coords))
        out.append(_normalize([(r, -c) for r, c in coords]))
        coords = [(c, -r) for r, c in coords]
    unique = []
    seen = set()
    for candidate in out:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _canonical_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_all_transforms(shape))


def _random_connected_shape(area: int) -> tuple[tuple[int, int], ...]:
    limit = MAX_SHAPE_SPAN
    for _ in range(200):
        cells = {(limit // 2, limit // 2)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < limit and 0 <= nc < limit and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            if random.random() < 0.35:
                frontier.sort(key=lambda p: abs(p[0] - limit // 2) + abs(p[1] - limit // 2), reverse=True)
                chosen = frontier[0]
            else:
                chosen = random.choice(frontier)
            cells.add(chosen)
        if len(cells) != area:
            continue
        shape = _normalize(cells)
        h, w = _shape_dims(shape)
        if max(h, w) > MAX_SHAPE_SPAN:
            continue
        return shape
    raise RuntimeError("shape generation failed")


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects = []
    for comp_id, coords_arr in enumerate(_components(g != BG)):
        coords = [tuple(map(int, p)) for p in coords_arr]
        marker_cells = [(r, c) for r, c in coords if int(g[r, c]) not in (BG, GREY)]
        assert len(marker_cells) == 1, "every object must contain exactly one marker pixel"
        marker_pos = marker_cells[0]
        marker_color = int(g[marker_pos])
        grey_coords = [(r, c) for r, c in coords if int(g[r, c]) == GREY]
        assert len(grey_coords) + 1 == len(coords), "objects should contain only grey cells plus one marker"

        shape = _normalize(coords)
        r0, c0, r1, c1 = _coords_bbox(coords)
        center_r = sum(r for r, _ in coords) / len(coords)
        center_c = sum(c for _, c in coords) / len(coords)
        objects.append(
            {
                "id": comp_id,
                "coords": coords,
                "grey_coords": grey_coords,
                "marker_pos": marker_pos,
                "marker_color": marker_color,
                "area": len(coords),
                "shape": shape,
                "canonical": _canonical_shape(shape),
                "perimeter": _shape_perimeter(shape),
                "bbox": (r0, c0, r1, c1),
                "bbox_perimeter": 2 * ((_shape_dims(shape)[0]) + (_shape_dims(shape)[1])),
                "top_left": min(coords),
                "center": (center_r, center_c),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey object bodies from a global perimeter ranking and neighbor marker borrowing.

    Input structure:
    - The grid has a black background containing 4-connected foreground objects separated by background gaps.
    - Each object is mostly grey and contains exactly one non-black, non-grey marker pixel.
    - Objects may differ in size and shape, and the full object includes both the grey cells and its marker pixel.

    Rule:
    - Compute each object's 4-neighbor perimeter by counting every object edge exposed to the background or the grid boundary.
    - Sort the objects by increasing perimeter to get ranks 1, 2, 3, ...
    - Recolor grey cells of odd-ranked objects to their own marker color, and recolor grey cells of even-ranked objects to the marker color of the object ranked immediately above them.
    - Keep every marker pixel unchanged.

    Key insight:
    - The output color of each object is determined by a global geometric order, not by local position or by the object's own color alone.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _analyze_objects(g)
    perimeters = [obj["perimeter"] for obj in objects]
    assert len(perimeters) == len(set(perimeters)), "object perimeters must be unique"

    ranked = sorted(objects, key=lambda obj: obj["perimeter"])
    for rank, obj in enumerate(ranked, start=1):
        fill_color = obj["marker_color"] if rank % 2 == 1 else ranked[rank - 2]["marker_color"]
        for r, c in obj["grey_coords"]:
            out[r, c] = fill_color
    return out


def _choose_object_specs(count: int) -> list[dict]:
    for _ in range(400):
        specs = []
        used_canonicals = set()
        used_perimeters = set()
        attempts = 0
        while len(specs) < count and attempts < 4000:
            attempts += 1
            target_area = random.randint(*OBJECT_AREA_RANGE)
            shape = _random_connected_shape(target_area)
            canonical = _canonical_shape(shape)
            perimeter = _shape_perimeter(shape)
            if canonical in used_canonicals or perimeter in used_perimeters:
                continue
            specs.append(
                {
                    "base_shape": shape,
                    "canonical": canonical,
                    "area": len(shape),
                    "perimeter": perimeter,
                }
            )
            used_canonicals.add(canonical)
            used_perimeters.add(perimeter)
        if len(specs) != count:
            continue

        rank_order = [spec["canonical"] for spec in sorted(specs, key=lambda spec: spec["perimeter"])]
        area_order = [spec["canonical"] for spec in sorted(specs, key=lambda spec: (spec["area"], spec["perimeter"]))]
        bbox_order = [
            spec["canonical"]
            for spec in sorted(
                specs,
                key=lambda spec: (2 * sum(_shape_dims(spec["base_shape"])), spec["area"], spec["perimeter"]),
            )
        ]
        if rank_order == area_order or rank_order == list(reversed(area_order)):
            continue
        if rank_order == bbox_order or rank_order == list(reversed(bbox_order)):
            continue
        return specs
    raise RuntimeError("failed to choose object specs")


def _valid_offsets(shape: tuple[tuple[int, int], ...], blocked: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    sh, sw = _shape_dims(shape)
    candidates = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            placed = [(r0 + r, c0 + c) for r, c in shape]
            if all((r, c) not in blocked for r, c in placed):
                candidates.append((r0, c0))
    return candidates


def _place_objects(specs: list[dict], h: int, w: int) -> list[dict] | None:
    blocked = set()
    placed = [None] * len(specs)
    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        transforms = _all_transforms(specs[idx]["base_shape"])
        random.shuffle(transforms)
        chosen = None
        for shape in transforms:
            offsets = _valid_offsets(shape, blocked, h, w)
            if not offsets:
                continue
            r0, c0 = random.choice(offsets)
            placed_cells = [(r0 + r, c0 + c) for r, c in shape]
            chosen = (shape, placed_cells, r0, c0)
            break
        if chosen is None:
            return None
        shape, placed_cells, r0, c0 = chosen
        marker_local = random.choice(list(shape))
        marker_abs = (r0 + marker_local[0], c0 + marker_local[1])
        for r, c in placed_cells:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    blocked.add((rr, cc))
        placed[idx] = {
            "shape": shape,
            "cells": placed_cells,
            "marker_pos": marker_abs,
            "area": len(shape),
            "perimeter": specs[idx]["perimeter"],
            "canonical": specs[idx]["canonical"],
        }
    return placed


def _manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _has_pairwise_gap(objects: list[dict]) -> bool:
    cell_sets = [set(obj["coords"]) for obj in objects]
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            for r1, c1 in cell_sets[i]:
                for r2, c2 in cell_sets[j]:
                    if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                        return False
    return True


def _semantic_input_checks(grid: np.ndarray) -> bool:
    try:
        objects = _analyze_objects(grid)
    except AssertionError:
        return False
    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return False
    if len({obj["marker_color"] for obj in objects}) != len(objects):
        return False
    if any(obj["marker_color"] in (BG, GREY) for obj in objects):
        return False
    if any(not (OBJECT_AREA_RANGE[0] <= obj["area"] <= OBJECT_AREA_RANGE[1]) for obj in objects):
        return False
    if len({obj["perimeter"] for obj in objects}) != len(objects):
        return False
    if len({obj["canonical"] for obj in objects}) != len(objects):
        return False
    if not _has_pairwise_gap(objects):
        return False

    rank_order = [obj["id"] for obj in sorted(objects, key=lambda obj: obj["perimeter"])]
    area_order = [obj["id"] for obj in sorted(objects, key=lambda obj: (obj["area"], obj["perimeter"]))]
    bbox_order = [obj["id"] for obj in sorted(objects, key=lambda obj: (obj["bbox_perimeter"], obj["area"], obj["perimeter"]))]
    row_order = [obj["id"] for obj in sorted(objects, key=lambda obj: obj["top_left"])]
    col_order = [obj["id"] for obj in sorted(objects, key=lambda obj: (obj["top_left"][1], obj["top_left"][0]))]

    if rank_order == area_order or rank_order == list(reversed(area_order)):
        return False
    if rank_order == bbox_order or rank_order == list(reversed(bbox_order)):
        return False
    if rank_order == row_order or rank_order == list(reversed(row_order)):
        return False
    if rank_order == col_order or rank_order == list(reversed(col_order)):
        return False

    ranked = sorted(objects, key=lambda obj: obj["perimeter"])
    has_nonlocal_borrow = False
    for rank in range(1, len(ranked), 2):
        borrower = ranked[rank]
        donor = ranked[rank - 1]
        dists = sorted(
            (
                _manhattan(borrower["center"], other["center"]),
                other["id"],
            )
            for other in objects
            if other["id"] != borrower["id"]
        )
        if len(dists) >= 2 and dists[0][0] < dists[1][0] and dists[0][1] != donor["id"]:
            has_nonlocal_borrow = True
            break
    return has_nonlocal_borrow


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        obj_count = random.randint(*OBJECT_COUNT_RANGE)
        grid_h = random.randint(*GRID_SIZE_RANGE)
        grid_w = random.randint(*GRID_SIZE_RANGE)
        specs = _choose_object_specs(obj_count)
        placed = _place_objects(specs, grid_h, grid_w)
        if placed is None:
            continue

        marker_colors = random.sample(MARKER_COLORS, obj_count)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        for obj, color in zip(placed, marker_colors):
            marker_pos = obj["marker_pos"]
            for r, c in obj["cells"]:
                grid[r, c] = GREY
            grid[marker_pos] = color

        if _semantic_input_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
