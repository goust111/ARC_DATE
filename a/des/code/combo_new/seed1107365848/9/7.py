from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
ALL_DIMS = [(h, w) for h in range(1, 5) for w in range(1, 5)]
DIMS_BY_AREA: dict[int, list[tuple[int, int]]] = {}
for _h, _w in ALL_DIMS:
    DIMS_BY_AREA.setdefault(_h * _w, []).append((_h, _w))

MIN_PAIRS = 2
MAX_PAIRS = 4
GRID_H_RANGE = (12, 24)
GRID_W_MAX = 24
OBJECT_PADDING = 1
LEGEND_GAP = 1
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _parse_legend(row: np.ndarray) -> list[tuple[int, int]]:
    row = np.asarray(row, dtype=int)
    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    c = 0
    while c < row.size:
        if int(row[c]) == BG:
            c += 1
            continue
        assert c + 1 < row.size and int(row[c + 1]) != BG, "legend pairs must be adjacent colored pixels"
        assert c == 0 or int(row[c - 1]) == BG, "legend entries must start after black"
        assert c + 2 >= row.size or int(row[c + 2]) == BG, "legend entries must have length exactly two"
        left = int(row[c])
        right = int(row[c + 1])
        assert left != right, "paired colors must be distinct"
        assert left not in seen and right not in seen, "each legend color must appear exactly once"
        seen.add(left)
        seen.add(right)
        pairs.append((left, right))
        c += 3
    assert pairs, "legend row must contain at least one color pair"
    return pairs


def _extract_objects(g: np.ndarray) -> dict[int, dict[str, object]]:
    top = g[:-1, :]
    colors = sorted(set(map(int, np.unique(top))) - {BG})
    assert colors, "expected at least one object above the legend row"
    objects: dict[int, dict[str, object]] = {}
    for color in colors:
        coords = np.argwhere(top == color)
        assert coords.size > 0
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(coords.shape[0])
        assert area == h * w, "each color object must be a solid rectangle"
        assert np.all(top[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle interior must be filled"
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        objects[color] = {
            "color": color,
            "coords": coords.copy(),
            "cells": {tuple(map(int, rc)) for rc in coords},
            "bbox": (r0, c0, r1, c1),
            "area": area,
            "center": center,
            "top": r0,
            "left": c0,
        }
    return objects


def _source_target(
    first: dict[str, object], second: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    key_first = (int(first["area"]), int(first["top"]), int(first["left"]))
    key_second = (int(second["area"]), int(second["top"]), int(second["left"]))
    if key_first <= key_second:
        return first, second
    return second, first


def _path_cells(source: dict[str, object], target: dict[str, object]) -> set[tuple[int, int]]:
    sr, sc = source["center"]
    tr, tc = target["center"]
    cells = {(r, sc) for r in range(min(sr, tr), max(sr, tr) + 1)}
    cells.update((tr, c) for c in range(min(sc, tc), max(sc, tc) + 1))
    return cells


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 3 <= h <= 30 and 1 <= w <= 30
    assert int(g.min()) >= 0 and int(g.max()) <= 9

    legend_pairs = _parse_legend(g[-1, :])
    objects = _extract_objects(g)
    legend_colors = {color for pair in legend_pairs for color in pair}
    assert set(objects) == legend_colors, "legend colors must match the object colors above the legend"

    out = g.copy()
    out[-1, :] = BG
    for color_a, color_b in legend_pairs:
        source, target = _source_target(objects[color_a], objects[color_b])
        for r, c in _path_cells(source, target):
            out[r, c] = int(source["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each legend-paired color pair by an L-shaped line whose color comes from the smaller rectangle.

    Input structure:
    - The background is black.
    - The bottom row is a color-pair legend: each entry is two adjacent colored pixels, and neighboring entries are separated by black.
    - Above the legend, each non-black color forms exactly one solid axis-aligned rectangle, with one rectangle per legend color.

    Rule:
    - Read the bottom-row legend as unordered color pairs.
    - For each pair, compare the two matching rectangles: the smaller-area rectangle is the source and the larger-area rectangle is the target; if the areas tie, the higher rectangle is the source.
    - Draw an L-shaped path in the source color from the source center pixel vertically to the target center row, then horizontally to the target center pixel, overwriting anything on that path.
    - Erase the legend row to black in the output.

    Key insight:
    - The legend only says which colors belong together; the line color and direction come from a separate comparison of the paired rectangles' sizes and positions.
    """
    return _solve(grid)


def _sample_pair_shapes(equal_area: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    if equal_area:
        area = random.choice(sorted(DIMS_BY_AREA))
        first = random.choice(DIMS_BY_AREA[area])
        if random.random() < 0.75:
            second = first
        else:
            second = random.choice(DIMS_BY_AREA[area])
        return first, second

    while True:
        first = random.choice(ALL_DIMS)
        second = random.choice(ALL_DIMS)
        if first[0] * first[1] != second[0] * second[1]:
            return first, second


def _legend_width(pair_count: int) -> int:
    return 2 * pair_count + LEGEND_GAP * (pair_count - 1)


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - OBJECT_PADDING)
    cc0 = max(0, c0 - OBJECT_PADDING)
    rr1 = min(mask.shape[0], r0 + h + OBJECT_PADDING)
    cc1 = min(mask.shape[1], c0 + w + OBJECT_PADDING)
    mask[rr0:rr1, cc0:cc1] = True


def _try_place_objects(
    height: int, width: int, specs: list[dict[str, int]]
) -> dict[int, dict[str, object]] | None:
    reserve = np.zeros((height - 1, width), dtype=bool)
    placed: dict[int, dict[str, object]] = {}
    order = sorted(specs, key=lambda spec: spec["h"] * spec["w"], reverse=True)
    for spec in order:
        h = int(spec["h"])
        w = int(spec["w"])
        positions = [(r, c) for r in range(height - h) for c in range(width - w + 1)]
        random.shuffle(positions)
        for r0, c0 in positions[:PLACEMENT_RETRIES]:
            if reserve[r0 : r0 + h, c0 : c0 + w].any():
                continue
            placed[int(spec["color"])] = {
                "color": int(spec["color"]),
                "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
            }
            _reserve(reserve, r0, c0, h, w)
            break
        else:
            return None
    return placed


def _materialize_objects(placed: dict[int, dict[str, object]]) -> dict[int, dict[str, object]]:
    objects: dict[int, dict[str, object]] = {}
    for color, data in placed.items():
        r0, c0, r1, c1 = data["bbox"]
        coords = np.array(
            [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)],
            dtype=int,
        )
        objects[color] = {
            "color": color,
            "coords": coords,
            "cells": {tuple(map(int, rc)) for rc in coords},
            "bbox": (r0, c0, r1, c1),
            "area": int(coords.shape[0]),
            "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            "top": r0,
            "left": c0,
        }
    return objects


def _valid_paths(
    objects: dict[int, dict[str, object]], legend_pairs: list[tuple[int, int]]
) -> bool:
    seen_path_cells: set[tuple[int, int]] = set()
    all_object_cells = {
        color: set(obj["cells"])
        for color, obj in objects.items()
    }
    for color_a, color_b in legend_pairs:
        first = objects[color_a]
        second = objects[color_b]
        source, target = _source_target(first, second)
        if int(first["area"]) == int(second["area"]) and int(first["top"]) == int(second["top"]):
            return False
        sr, sc = source["center"]
        tr, tc = target["center"]
        if sr == tr or sc == tc:
            return False
        path = _path_cells(source, target)
        allowed = all_object_cells[color_a] | all_object_cells[color_b]
        for other_color, cells in all_object_cells.items():
            if other_color in {color_a, color_b}:
                continue
            if path & cells:
                return False
        if path & seen_path_cells:
            return False
        seen_path_cells |= path
        if len(path) < 3:
            return False
    return True


def _build_input(
    height: int,
    width: int,
    objects: dict[int, dict[str, object]],
    legend_pairs: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for color, obj in objects.items():
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    offset = random.randint(0, width - _legend_width(len(legend_pairs)))
    c = offset
    for index, (left, right) in enumerate(legend_pairs):
        grid[-1, c] = left
        grid[-1, c + 1] = right
        c += 2
        if index < len(legend_pairs) - 1:
            c += LEGEND_GAP
    return grid


def _pair_branch_flags(pair_count: int) -> list[bool]:
    flags = [bool(random.getrandbits(1)) for _ in range(pair_count)]
    if pair_count >= 2 and (all(flags) or not any(flags)):
        flags[random.randrange(pair_count)] = not flags[0]
    return flags


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(MIN_PAIRS, MAX_PAIRS)
    colors = random.sample(PALETTE, 2 * pair_count)
    pair_colors = [(colors[i], colors[i + 1]) for i in range(0, len(colors), 2)]
    random.shuffle(pair_colors)

    branch_flags = _pair_branch_flags(pair_count)
    specs: list[dict[str, int]] = []
    legend_pairs: list[tuple[int, int]] = []
    for equal_area, (color_a, color_b) in zip(branch_flags, pair_colors):
        (h1, w1), (h2, w2) = _sample_pair_shapes(equal_area)
        if random.random() < 0.5:
            legend_pairs.append((color_a, color_b))
            specs.append({"color": color_a, "h": h1, "w": w1})
            specs.append({"color": color_b, "h": h2, "w": w2})
        else:
            legend_pairs.append((color_b, color_a))
            specs.append({"color": color_a, "h": h1, "w": w1})
            specs.append({"color": color_b, "h": h2, "w": w2})

    legend_w = _legend_width(pair_count)
    width_min = max(legend_w, 12)
    width = random.randint(width_min, GRID_W_MAX)
    height = random.randint(*GRID_H_RANGE)

    placed = _try_place_objects(height, width, specs)
    if placed is None:
        return None
    objects = _materialize_objects(placed)
    if not _valid_paths(objects, legend_pairs):
        return None

    grid = _build_input(height, width, objects, legend_pairs)
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
