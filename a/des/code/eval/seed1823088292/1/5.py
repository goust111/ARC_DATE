from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
RED = 2
GREY = 5
OBJECT_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
SCENARIOS = ("overlap", "isolated", "mixed")
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nbr in _neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _is_orthogonally_convex(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    for r in range(height):
        cols = sorted(c for rr, c in cells if rr == r)
        if not cols:
            return False
        if cols[-1] - cols[0] + 1 != len(cols):
            return False
    for c in range(width):
        rows = sorted(r for r, cc in cells if cc == c)
        if not rows:
            return False
        if rows[-1] - rows[0] + 1 != len(rows):
            return False
    return True


def _edge_midpoints(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    mid_r = r0 + (height - 1) // 2
    mid_c = c0 + (width - 1) // 2
    return {
        "up": (r0, mid_c),
        "down": (r1, mid_c),
        "left": (mid_r, c0),
        "right": (mid_r, c1),
    }


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components = _component_coords(g != BLACK)
    objects: list[dict] = []
    for coords in components:
        counts = Counter(int(g[r, c]) for r, c in coords)
        assert len(counts) == 2, f"expected exactly two colors per object, got {counts}"

        base_color, base_count = counts.most_common(1)[0]
        marker_candidates = [color for color, count in counts.items() if count == 1 and color != base_color]
        assert len(marker_candidates) == 1, f"expected one singleton marker color, got {counts}"
        marker_color = marker_candidates[0]
        assert base_count >= 2, "object body must contain at least two pixels"

        marker_positions = [(r, c) for r, c in coords if int(g[r, c]) == marker_color]
        assert len(marker_positions) == 1
        marker = marker_positions[0]

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        midpoints = _edge_midpoints(r0, c0, r1, c1)
        matching = [name for name, point in midpoints.items() if point == marker]
        assert len(matching) == 1, f"marker must sit on a unique bbox midpoint edge, got {marker} in {midpoints}"
        direction = matching[0]

        objects.append(
            {
                "coords": sorted(coords),
                "bbox": (r0, c0, r1, c1),
                "area": len(coords),
                "base_color": base_color,
                "marker_color": marker_color,
                "marker": marker,
                "direction": direction,
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _shadow_cells(grid: np.ndarray, obj: dict) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = obj["bbox"]
    length = int(obj["area"])
    direction = obj["direction"]
    cells: set[tuple[int, int]] = set()

    if direction == "up":
        row_range = range(r0 - length, r0)
        col_range = range(c0, c1 + 1)
    elif direction == "down":
        row_range = range(r1 + 1, r1 + length + 1)
        col_range = range(c0, c1 + 1)
    elif direction == "left":
        row_range = range(r0, r1 + 1)
        col_range = range(c0 - length, c0)
    elif direction == "right":
        row_range = range(r0, r1 + 1)
        col_range = range(c1 + 1, c1 + length + 1)
    else:
        raise AssertionError(f"unknown direction {direction}")

    for r in row_range:
        for c in col_range:
            if 0 <= r < h and 0 <= c < w and int(g[r, c]) == BLACK:
                cells.add((r, c))
    return cells


def _shadow_overlap_data(grid: np.ndarray, objects: list[dict]) -> tuple[set[int], set[tuple[int, int]], list[set[tuple[int, int]]]]:
    shadow_sets = [_shadow_cells(grid, obj) for obj in objects]
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, cells in enumerate(shadow_sets):
        for cell in cells:
            owners[cell].append(idx)

    keep: set[int] = set()
    overlap_cells: set[tuple[int, int]] = set()
    for cell, ids in owners.items():
        if len(ids) >= 2:
            keep.update(ids)
            overlap_cells.add(cell)
    return keep, overlap_cells, shadow_sets


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    keep, overlap_cells, shadow_sets = _shadow_overlap_data(g, objects)

    out = np.full_like(g, BLACK)
    for idx, obj in enumerate(objects):
        if idx not in keep:
            continue
        for r, c in obj["coords"]:
            out[r, c] = int(obj["base_color"])

    for idx, cells in enumerate(shadow_sets):
        if idx not in keep:
            continue
        for cell in cells:
            out[cell] = RED if cell in overlap_cells else GREY

    return out


def _make_shape(direction: str, base_color: int, marker_color: int) -> dict | None:
    for _ in range(100):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        if height + width - 1 > 9:
            continue

        midpoint = _edge_midpoints(0, 0, height - 1, width - 1)[direction]
        anchor = (midpoint[0], midpoint[1])
        target_area = random.randint(max(4, height + width - 1), min(9, height * width))

        cells = {(r, c) for r in range(height) for c in range(width)}
        while len(cells) > target_area:
            removable = []
            for cell in cells:
                if cell == anchor:
                    continue
                reduced = cells - {cell}
                if not _is_connected(reduced):
                    continue
                if not _is_orthogonally_convex(reduced, height, width):
                    continue
                removable.append(cell)
            if not removable:
                break
            cells.remove(random.choice(removable))

        if len(cells) != target_area:
            continue

        patch = np.zeros((height, width), dtype=int)
        for r, c in cells:
            patch[r, c] = base_color
        patch[anchor] = marker_color
        return {
            "patch": patch,
            "cells": sorted(cells),
            "marker_local": anchor,
            "height": height,
            "width": width,
            "area": len(cells),
            "direction": direction,
            "base_color": base_color,
            "marker_color": marker_color,
        }
    return None


def _placed_bbox(obj: dict) -> tuple[int, int, int, int]:
    return (obj["r0"], obj["c0"], obj["r0"] + obj["height"] - 1, obj["c0"] + obj["width"] - 1)


def _boxes_far_enough(a: dict, b: dict, gap: int = 3) -> bool:
    ar0, ac0, ar1, ac1 = _placed_bbox(a)
    br0, bc0, br1, bc1 = _placed_bbox(b)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _fits_with_shadow(shape: dict, r0: int, c0: int, height: int, width: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + shape["height"] > height or c0 + shape["width"] > width:
        return False

    direction = shape["direction"]
    area = shape["area"]
    if direction == "up":
        return r0 - area >= 0
    if direction == "down":
        return r0 + shape["height"] + area <= height
    if direction == "left":
        return c0 - area >= 0
    if direction == "right":
        return c0 + shape["width"] + area <= width
    return False


def _render_grid(placed: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for obj in placed:
        r0, c0 = obj["r0"], obj["c0"]
        for rr, cc in obj["cells"]:
            value = obj["marker_color"] if (rr, cc) == obj["marker_local"] else obj["base_color"]
            grid[r0 + rr, c0 + cc] = value
    return grid


def _random_place(shape: dict, placed: list[dict], height: int, width: int) -> dict | None:
    for _ in range(200):
        r0 = random.randint(0, height - shape["height"])
        c0 = random.randint(0, width - shape["width"])
        if not _fits_with_shadow(shape, r0, c0, height, width):
            continue
        candidate = dict(shape)
        candidate["r0"] = r0
        candidate["c0"] = c0
        if all(_boxes_far_enough(candidate, other) for other in placed):
            return candidate
    return None


def _overlap_pair_directions() -> tuple[str, str]:
    return random.choice(
        [
            ("right", "down"),
            ("right", "up"),
            ("left", "down"),
            ("left", "up"),
            ("down", "right"),
            ("down", "left"),
            ("up", "right"),
            ("up", "left"),
        ]
    )


def _place_overlap_pair(shape_a: dict, shape_b: dict, height: int, width: int) -> tuple[dict, dict] | None:
    for _ in range(200):
        horiz, vert = shape_a, shape_b
        overlap_r = random.randint(6, height - 7)
        overlap_c = random.randint(6, width - 7)

        if horiz["direction"] in ("left", "right"):
            if vert["direction"] not in ("up", "down"):
                continue
        else:
            horiz, vert = vert, horiz
            if horiz["direction"] not in ("left", "right") or vert["direction"] not in ("up", "down"):
                continue

        if vert["direction"] == "down":
            hr0 = overlap_r
        else:
            hr0 = overlap_r - horiz["height"] + 1
        if horiz["direction"] == "right":
            vc0 = overlap_c
        else:
            vc0 = overlap_c - vert["width"] + 1

        if not (0 <= hr0 <= height - horiz["height"] and 0 <= vc0 <= width - vert["width"]):
            continue

        if horiz["direction"] == "right":
            if horiz["area"] < 4:
                continue
            dh = random.randint(4, horiz["area"])
            hc0 = overlap_c - dh - horiz["width"] + 1
        else:
            if horiz["area"] < 4:
                continue
            dh = random.randint(4, horiz["area"])
            hc0 = overlap_c + dh

        if vert["direction"] == "down":
            if vert["area"] < 4:
                continue
            dv = random.randint(4, vert["area"])
            vr0 = overlap_r - dv - vert["height"] + 1
        else:
            if vert["area"] < 4:
                continue
            dv = random.randint(4, vert["area"])
            vr0 = overlap_r + dv

        if not _fits_with_shadow(horiz, hr0, hc0, height, width):
            continue
        if not _fits_with_shadow(vert, vr0, vc0, height, width):
            continue

        horiz_obj = dict(horiz)
        vert_obj = dict(vert)
        horiz_obj["r0"] = hr0
        horiz_obj["c0"] = hc0
        vert_obj["r0"] = vr0
        vert_obj["c0"] = vc0
        if _boxes_far_enough(horiz_obj, vert_obj):
            return horiz_obj, vert_obj
    return None


def _build_shapes(num_objects: int, forced_directions: list[str] | None = None) -> list[dict] | None:
    base_colors = random.sample(OBJECT_COLORS, num_objects)
    shapes: list[dict] = []
    for idx in range(num_objects):
        direction = forced_directions[idx] if forced_directions is not None else random.choice(DIRECTIONS)
        marker_pool = [color for color in OBJECT_COLORS if color != base_colors[idx]]
        marker_color = random.choice(marker_pool)
        shape = _make_shape(direction, base_colors[idx], marker_color)
        if shape is None:
            return None
        shapes.append(shape)
    return shapes


def _scenario_ok(grid: np.ndarray, scenario: str, expected_objects: int) -> bool:
    try:
        objects = _extract_objects(grid)
    except AssertionError:
        return False
    if len(objects) != expected_objects:
        return False

    keep, overlap_cells, _shadow_sets = _shadow_overlap_data(grid, objects)
    out = main(grid)
    if scenario == "isolated":
        return len(keep) == 0 and not np.any(out)
    if scenario == "overlap":
        return len(keep) == expected_objects and len(overlap_cells) >= 1 and np.any(out == RED)
    if scenario == "mixed":
        return 1 < len(keep) < expected_objects and len(overlap_cells) >= 1 and np.any(out == RED)
    return False


def _gen_attempt(scenario: str) -> np.ndarray | None:
    height = random.randint(20, 26)
    width = random.randint(20, 26)

    if scenario == "overlap":
        dirs = list(_overlap_pair_directions())
        shapes = _build_shapes(2, forced_directions=dirs)
        if shapes is None:
            return None
        pair = _place_overlap_pair(shapes[0], shapes[1], height, width)
        if pair is None:
            return None
        placed = [pair[0], pair[1]]
    elif scenario == "isolated":
        num_objects = random.randint(2, 4)
        shapes = _build_shapes(num_objects)
        if shapes is None:
            return None
        placed = []
        for shape in shapes:
            obj = _random_place(shape, placed, height, width)
            if obj is None:
                return None
            placed.append(obj)
    else:
        extra_count = random.randint(1, 2)
        dirs = list(_overlap_pair_directions())
        shapes = _build_shapes(2 + extra_count, forced_directions=dirs + [random.choice(DIRECTIONS) for _ in range(extra_count)])
        if shapes is None:
            return None
        pair = _place_overlap_pair(shapes[0], shapes[1], height, width)
        if pair is None:
            return None
        placed = [pair[0], pair[1]]
        for shape in shapes[2:]:
            obj = _random_place(shape, placed, height, width)
            if obj is None:
                return None
            placed.append(obj)

    grid = _render_grid(placed, height, width)
    if _scenario_ok(grid, scenario, len(placed)):
        return grid
    return None


def generate_input() -> np.ndarray:
    scenario = random.choice(SCENARIOS)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
