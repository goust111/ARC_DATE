from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
CORNER_PALETTE = (1, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 500
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _w = _shape_dims(shape)
    return _normalize_shape({(c, h - 1 - r) for r, c in shape})


def _flip(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _h, w = _shape_dims(shape)
    return _normalize_shape({(r, w - 1 - c) for r, c in shape})


def _parse_shape(spec: str) -> tuple[tuple[int, int], ...]:
    rows = spec.split("/")
    cells = {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"}
    return _normalize_shape(cells)


_BASE_SHAPES = {
    1: ["#"],
    2: ["##"],
    3: ["##/#.", "###"],
    4: ["##/##", "###/#..", "###/.#.", "##/.##"],
    5: [".#./###/.#.", "###/.#./.#.", "##/##/#.", "####/#...", "##/.##/..#"],
    6: ["###/###", "####/#.../#...", "###/.#./.#./.#.", "##/.##/..##", "###/##./#.."],
}


def _shape_variants(area: int) -> list[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for spec in _BASE_SHAPES[area]:
        shape = _parse_shape(spec)
        for _ in range(4):
            variants.add(shape)
            variants.add(_flip(shape))
            shape = _rotate(shape)
    return list(variants)


_SHAPE_VARIANTS = {area: _shape_variants(area) for area in sorted(_BASE_SHAPES)}
_AREA_SETS = [
    tuple(sorted(combo, reverse=True))
    for size in (2, 3, 4)
    for combo in itertools.combinations(range(1, 7), size)
]


def find_corner_regions(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, GRAY)):
        coords = np.argwhere(g == color)
        if coords.shape[0] != 4:
            continue
        rows = sorted({int(r) for r, _ in coords})
        cols = sorted({int(c) for _, c in coords})
        if len(rows) != 2 or len(cols) != 2:
            continue
        if rows[1] - rows[0] < 2 or cols[1] - cols[0] < 2:
            continue
        expected = {
            (rows[0], cols[0]),
            (rows[0], cols[1]),
            (rows[1], cols[0]),
            (rows[1], cols[1]),
        }
        actual = {tuple(map(int, rc)) for rc in coords}
        if actual != expected:
            continue
        regions.append({"color": color, "r0": rows[0], "c0": cols[0], "r1": rows[1], "c1": cols[1]})
    regions.sort(key=lambda item: (item["r0"], item["c0"], item["color"]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor ranked gray components inside corner-marked rectangles and erase everything else.

    Input structure:
    - A black background contains 2-3 non-overlapping rectangles marked only by four
      same-colored corner pixels.
    - Inside each marked rectangle, excluding the corner rows and columns, there are
      2-4 gray 4-connected components with distinct areas.
    - Different rectangles use different corner colors, and gray objects from one
      rectangle never interact with another.

    Rule:
    - Detect each corner-marker rectangle from its four same-colored corner pixels.
    - Within that rectangle's interior, find all gray 4-connected components and rank
      them by area from largest to smallest.
    - Recolor the largest component with the rectangle's corner color, recolor the
      second largest component red, remove every lower-ranked gray component, and also
      erase the corner markers themselves.

    Key insight:
    - The corner pixels define independent local arenas, so component area matters only
      relative to other gray components inside the same marked rectangle.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    for region in find_corner_regions(g):
        r0, c0, r1, c1 = region["r0"], region["c0"], region["r1"], region["c1"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        comps = _components4(interior == GRAY)
        ranked = sorted(comps, key=lambda comp: (-len(comp), min(comp)))
        for rank, comp in enumerate(ranked):
            if rank == 0:
                color = region["color"]
            elif rank == 1:
                color = RED
            else:
                continue
            for rr, cc in comp:
                out[r0 + 1 + rr, c0 + 1 + cc] = color
    return out


def _choose_area_sets(region_count: int) -> list[tuple[int, ...]]:
    for _ in range(400):
        sets = random.sample(_AREA_SETS, region_count)
        if not any(len(area_set) >= 3 for area_set in sets):
            continue
        ranks_by_area: dict[int, set[int]] = {}
        for area_set in sets:
            for rank, area in enumerate(area_set, start=1):
                ranks_by_area.setdefault(area, set()).add(rank)
        if any(len(ranks) >= 2 for ranks in ranks_by_area.values()):
            return sets
    raise RuntimeError("failed to choose non-degenerate area sets")


def _valid_component_cells(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], inner_h: int, inner_w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < inner_h and 0 <= c < inner_w):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) in occupied:
                return False
    return True


def _place_shapes(inner_h: int, inner_w: int, areas: tuple[int, ...]) -> list[dict[str, object]] | None:
    ordered_areas = sorted(areas, reverse=True)

    def search(idx: int, occupied: set[tuple[int, int]], placed: list[dict[str, object]]):
        if idx == len(ordered_areas):
            return [dict(item) for item in placed]
        area = ordered_areas[idx]
        variants = _SHAPE_VARIANTS[area][:]
        random.shuffle(variants)
        for shape in variants:
            h, w = _shape_dims(shape)
            placements: list[set[tuple[int, int]]] = []
            for r0 in range(inner_h - h + 1):
                for c0 in range(inner_w - w + 1):
                    cells = {(r0 + r, c0 + c) for r, c in shape}
                    if _valid_component_cells(cells, occupied, inner_h, inner_w):
                        placements.append(cells)
            random.shuffle(placements)
            for cells in placements:
                placed.append({"area": area, "cells": tuple(sorted(cells))})
                result = search(idx + 1, occupied | cells, placed)
                if result is not None:
                    return result
                placed.pop()
        return None

    return search(0, set(), [])


def _build_region(marker_color: int, areas: tuple[int, ...]) -> dict[str, object] | None:
    dims = [(h, w) for h in range(3, 6) for w in range(3, 6)]
    dims.sort(key=lambda hw: (-(hw[0] * hw[1]), random.random()))
    for inner_h, inner_w in dims:
        placed = _place_shapes(inner_h, inner_w, areas)
        if placed is None:
            continue
        patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
        patch[0, 0] = marker_color
        patch[0, -1] = marker_color
        patch[-1, 0] = marker_color
        patch[-1, -1] = marker_color
        for comp in placed:
            for r, c in comp["cells"]:
                patch[r + 1, c + 1] = GRAY
        return {
            "color": marker_color,
            "areas": tuple(sorted((int(comp["area"]) for comp in placed), reverse=True)),
            "patch": patch,
            "components": placed,
            "outer_h": inner_h + 2,
            "outer_w": inner_w + 2,
        }
    return None


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _place_regions_on_canvas(regions: list[dict[str, object]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    for _ in range(250):
        region_count = len(regions)
        min_h = max(15, max(int(region["outer_h"]) for region in regions) + 2)
        min_w = max(15, max(int(region["outer_w"]) for region in regions) + 2)
        max_side = 19 if region_count == 2 else 22
        grid_h = random.randint(min_h, max_side)
        grid_w = random.randint(min_w, max_side)

        indexed = sorted(
            enumerate(regions),
            key=lambda item: -(int(item[1]["outer_h"]) * int(item[1]["outer_w"])),
        )
        positions: list[tuple[int, int] | None] = [None] * region_count
        boxes: list[tuple[int, int, int, int]] = []

        def search(idx: int) -> bool:
            if idx == len(indexed):
                return True
            original_idx, region = indexed[idx]
            h = int(region["outer_h"])
            w = int(region["outer_w"])
            candidates = [
                (r0, c0)
                for r0 in range(1, grid_h - h)
                for c0 in range(1, grid_w - w)
            ]
            random.shuffle(candidates)
            for r0, c0 in candidates:
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_boxes_conflict(box, other) for other in boxes):
                    continue
                boxes.append(box)
                positions[original_idx] = (r0, c0)
                if search(idx + 1):
                    return True
                boxes.pop()
                positions[original_idx] = None
            return False

        if search(0):
            return grid_h, grid_w, [pos for pos in positions if pos is not None]
    return None


def _render_expected(grid_shape: tuple[int, int], regions: list[dict[str, object]], positions: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for region, (r0, c0) in zip(regions, positions):
        ranked = sorted(region["components"], key=lambda comp: (-int(comp["area"]), min(comp["cells"])))
        for rank, comp in enumerate(ranked):
            if rank == 0:
                color = int(region["color"])
            elif rank == 1:
                color = RED
            else:
                continue
            for rr, cc in comp["cells"]:
                out[r0 + 1 + rr, c0 + 1 + cc] = color
    return out


def _shared_area_with_different_rank(regions: list[dict[str, object]]) -> bool:
    ranks_by_area: dict[int, set[int]] = {}
    for region in regions:
        for rank, area in enumerate(region["areas"], start=1):
            ranks_by_area.setdefault(int(area), set()).add(rank)
    return any(len(ranks) >= 2 for ranks in ranks_by_area.values())


def _validate_instance(grid: np.ndarray, expected: np.ndarray, region_specs: list[dict[str, object]]) -> bool:
    detected = find_corner_regions(grid)
    if len(detected) != len(region_specs):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if not any(len(region["areas"]) >= 3 for region in region_specs):
        return False
    if not _shared_area_with_different_rank(region_specs):
        return False
    output_colors = set(int(v) for v in np.unique(expected) if int(v) != BLACK)
    if RED not in output_colors:
        return False
    for region, spec in zip(detected, sorted(region_specs, key=lambda item: (item["position"][0], item["position"][1]))):
        if region["color"] != spec["color"]:
            return False
        if (region["r0"], region["c0"]) != spec["position"]:
            return False
        interior = grid[region["r0"] + 1 : region["r1"], region["c0"] + 1 : region["c1"]]
        areas = sorted((len(comp) for comp in _components4(interior == GRAY)), reverse=True)
        if tuple(areas) != tuple(spec["areas"]):
            return False
        if len(set(areas)) != len(areas) or len(areas) < 2:
            return False
        if expected[region["r0"], region["c0"]] != BLACK:
            return False
        if expected[region["r0"], region["c1"]] != BLACK:
            return False
        if expected[region["r1"], region["c0"]] != BLACK:
            return False
        if expected[region["r1"], region["c1"]] != BLACK:
            return False
    return not np.array_equal(grid, expected)


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    region_count = random.choice([2, 2, 3])
    area_sets = _choose_area_sets(region_count)
    marker_colors = random.sample(CORNER_PALETTE, region_count)
    regions: list[dict[str, object]] = []
    for color, areas in zip(marker_colors, area_sets):
        region = _build_region(color, areas)
        if region is None:
            raise RuntimeError("failed to build region")
        regions.append(region)

    placement = _place_regions_on_canvas(regions)
    if placement is None:
        raise RuntimeError("failed to place regions")
    grid_h, grid_w, positions = placement

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for region, (r0, c0) in zip(regions, positions):
        patch = region["patch"]
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        region["position"] = (r0, c0)

    expected = _render_expected((grid_h, grid_w), regions, positions)
    if not _validate_instance(grid, expected, regions):
        raise RuntimeError("generated instance failed semantic validation")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid, _expected = _generate_instance()
            return grid
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate valid input for puzzle 5")
