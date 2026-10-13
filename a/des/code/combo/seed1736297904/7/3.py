from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MAGENTA = 6
PALETTE = [1, 2, 3, 4, 5, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component_from(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) != int(color):
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return seen


def _first_hit(
    grid: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[int, tuple[int, int], int] | None:
    h, w = grid.shape
    rr, cc = r + dr, c + dc
    dist = 1
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value == MAGENTA:
            rr += dr
            cc += dc
            dist += 1
            continue
        if value != BG:
            return dist, (rr, cc), value
        rr += dr
        cc += dc
        dist += 1
    return None


def _parse_object_from_marker(grid: np.ndarray, marker: tuple[int, int]) -> dict:
    mr, mc = marker
    candidates = []
    for idx, (dr, dc) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
        hit = _first_hit(grid, mr, mc, dr, dc)
        if hit is None:
            continue
        dist, start, color = hit
        candidates.append((int(dist), int(idx), int(dr), int(dc), start, int(color)))

    assert candidates, "each marker must align with an object"
    candidates.sort()
    assert len(candidates) == 1 or candidates[0][0] < candidates[1][0], (
        "marker alignment must be unambiguous"
    )

    _dist, _idx, dr, dc, near_start, color = candidates[0]
    near = _component_from(grid, near_start, color)
    h, w = grid.shape

    if dr == 0:
        row = mr
        row_cols = [c for r, c in near if r == row]
        assert row_cols, "aligned marker row must cross the near half"
        if dc == 1:
            edge = max(row_cols)
            crack_c = edge + 1
            far_start = (row, crack_c + 1)
        else:
            edge = min(row_cols)
            crack_c = edge - 1
            far_start = (row, crack_c - 1)
        assert 0 <= crack_c < w
        assert int(grid[row, crack_c]) == BG
        assert 0 <= far_start[1] < w
        assert int(grid[far_start]) == color
        far = _component_from(grid, far_start, color)
        all_rows = [r for r, _c in near] + [r for r, _c in far]
        r0, r1 = min(all_rows), max(all_rows)
        crack = {(r, crack_c) for r in range(r0, r1 + 1)}
        assert all(int(grid[r, crack_c]) == BG for r in range(r0, r1 + 1))
        move = (0, dc * (r1 - r0 + 1))
    else:
        col = mc
        col_rows = [r for r, c in near if c == col]
        assert col_rows, "aligned marker column must cross the near half"
        if dr == 1:
            edge = max(col_rows)
            crack_r = edge + 1
            far_start = (crack_r + 1, col)
        else:
            edge = min(col_rows)
            crack_r = edge - 1
            far_start = (crack_r - 1, col)
        assert 0 <= crack_r < h
        assert int(grid[crack_r, col]) == BG
        assert 0 <= far_start[0] < h
        assert int(grid[far_start]) == color
        far = _component_from(grid, far_start, color)
        all_cols = [c for _r, c in near] + [c for _r, c in far]
        c0, c1 = min(all_cols), max(all_cols)
        crack = {(crack_r, c) for c in range(c0, c1 + 1)}
        assert all(int(grid[crack_r, c]) == BG for c in range(c0, c1 + 1))
        move = (dr * (c1 - c0 + 1), 0)

    assert near.isdisjoint(far), "crack must separate the two halves"
    assert len(near) != len(far), "the crack must split the object into unequal parts"
    return {
        "marker": marker,
        "color": color,
        "near": near,
        "far": far,
        "move": move,
        "crack": crack,
    }


def _objects_from_grid(grid: np.ndarray) -> list[dict]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MAGENTA)]
    if not markers:
        return []

    objects: list[dict] = []
    covered: set[tuple[int, int]] = set()
    for marker in markers:
        obj = _parse_object_from_marker(grid, marker)
        cells = set(obj["near"]) | set(obj["far"])
        assert covered.isdisjoint(cells), "each object must have exactly one marker"
        covered |= cells
        objects.append(obj)

    colored = {
        tuple(map(int, rc))
        for rc in np.argwhere((grid != BG) & (grid != MAGENTA))
    }
    assert covered == colored, "all colored pixels must belong to marked cracked objects"
    return objects


def _apply_transform(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[out == MAGENTA] = BG

    for obj in objects:
        for r, c in obj["far"]:
            out[r, c] = BG

    occupied = set()
    for obj in objects:
        occupied.update(obj["near"])

    moved_sets: list[tuple[int, set[tuple[int, int]]]] = []
    h, w = out.shape
    for obj in objects:
        dr, dc = obj["move"]
        moved = {(r + dr, c + dc) for r, c in obj["far"]}
        for r, c in moved:
            assert 0 <= r < h and 0 <= c < w, "translated half must stay inside the grid"
        assert moved.isdisjoint(occupied), "translated halves must not overlap"
        moved_sets.append((int(obj["color"]), moved))
        occupied |= moved

    for color, moved in moved_sets:
        for r, c in moved:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each cracked monochrome object into two halves and push the far half away from the nearest magenta marker.
    """
    g = np.asarray(grid, dtype=int)
    if not np.any(g == MAGENTA):
        return g.copy()
    objects = _objects_from_grid(g)
    return _apply_transform(g, objects)


def _expanded(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h - 1, r + radius) + 1):
            for cc in range(max(0, c - radius), min(w - 1, c + radius) + 1):
                out.add((rr, cc))
    return out


def _rect_cells(r0: int, c0: int, hh: int, ww: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)}


def _random_unequal_pair(lo: int, hi: int) -> tuple[int, int]:
    a = random.randint(lo, hi)
    b = random.randint(lo, hi)
    while a == b:
        b = random.randint(lo, hi)
    return a, b


def _make_spec(color: int, orientation: str) -> dict:
    if orientation == "vertical":
        height = random.randint(3, 6)
        left_w, right_w = _random_unequal_pair(1, 4)
        side = random.choice(["left", "right"])
        return {
            "color": color,
            "orientation": orientation,
            "height": height,
            "left_w": left_w,
            "right_w": right_w,
            "marker_gap": random.randint(1, 4),
            "align": random.randint(0, height - 1),
            "side": side,
            "footprint": height * (left_w + right_w + 1 + height),
        }

    width = random.randint(3, 6)
    top_h, bottom_h = _random_unequal_pair(1, 4)
    side = random.choice(["up", "down"])
    return {
        "color": color,
        "orientation": orientation,
        "width": width,
        "top_h": top_h,
        "bottom_h": bottom_h,
        "marker_gap": random.randint(1, 4),
        "align": random.randint(0, width - 1),
        "side": side,
        "footprint": width * (top_h + bottom_h + 1 + width),
    }


def _materialize_spec(spec: dict, r0: int, c0: int, h: int, w: int) -> dict | None:
    color = int(spec["color"])

    if spec["orientation"] == "vertical":
        height = int(spec["height"])
        left_w = int(spec["left_w"])
        right_w = int(spec["right_w"])
        total_w = left_w + 1 + right_w
        if not (0 <= r0 and r0 + height <= h and 0 <= c0 and c0 + total_w <= w):
            return None

        crack_c = c0 + left_w
        left = _rect_cells(r0, c0, height, left_w)
        right = _rect_cells(r0, crack_c + 1, height, right_w)
        if spec["side"] == "left":
            marker = (r0 + int(spec["align"]), c0 - int(spec["marker_gap"]))
            near = left
            far = right
            move = (0, height)
        else:
            marker = (r0 + int(spec["align"]), c0 + total_w - 1 + int(spec["marker_gap"]))
            near = right
            far = left
            move = (0, -height)

    else:
        width = int(spec["width"])
        top_h = int(spec["top_h"])
        bottom_h = int(spec["bottom_h"])
        total_h = top_h + 1 + bottom_h
        if not (0 <= r0 and r0 + total_h <= h and 0 <= c0 and c0 + width <= w):
            return None

        crack_r = r0 + top_h
        top = _rect_cells(r0, c0, top_h, width)
        bottom = _rect_cells(crack_r + 1, c0, bottom_h, width)
        if spec["side"] == "up":
            marker = (r0 - int(spec["marker_gap"]), c0 + int(spec["align"]))
            near = top
            far = bottom
            move = (width, 0)
        else:
            marker = (r0 + total_h - 1 + int(spec["marker_gap"]), c0 + int(spec["align"]))
            near = bottom
            far = top
            move = (-width, 0)

    all_input = set(near) | set(far) | {marker}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in all_input):
        return None

    moved = {(r + move[0], c + move[1]) for r, c in far}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in moved):
        return None

    input_reserve = _expanded(set(near) | set(far), h, w, radius=1) | _expanded({marker}, h, w, radius=1)
    output_cells = set(near) | moved
    output_reserve = _expanded(output_cells, h, w, radius=1)
    return {
        "color": color,
        "near": near,
        "far": far,
        "marker": marker,
        "move": move,
        "input_cells": set(near) | set(far),
        "output_cells": output_cells,
        "input_reserve": input_reserve,
        "output_reserve": output_reserve,
    }


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        objects = _objects_from_grid(grid)
        if len(objects) < 2:
            return False
        out = _apply_transform(grid, objects)
    except AssertionError:
        return False

    if not isinstance(out, np.ndarray):
        return False
    if out.shape != grid.shape:
        return False
    if np.any(out == MAGENTA):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    n_objects = random.randint(2, 4)
    colors = random.sample(PALETTE, n_objects)

    orientations = [random.choice(["vertical", "horizontal"]) for _ in range(n_objects)]
    if n_objects >= 2 and len(set(orientations)) == 1 and random.random() < 0.75:
        orientations[-1] = "horizontal" if orientations[0] == "vertical" else "vertical"

    specs = [_make_spec(color, orientation) for color, orientation in zip(colors, orientations)]
    specs.sort(key=lambda spec: int(spec["footprint"]), reverse=True)

    placed = []
    input_reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()

    for spec in specs:
        layout = None
        for _ in range(200):
            if spec["orientation"] == "vertical":
                height = int(spec["height"])
                total_w = int(spec["left_w"]) + 1 + int(spec["right_w"])
                if spec["side"] == "left":
                    min_c0 = int(spec["marker_gap"])
                    max_c0 = w - total_w - height
                else:
                    min_c0 = height
                    max_c0 = w - total_w - int(spec["marker_gap"])
                min_r0 = 0
                max_r0 = h - height
            else:
                width = int(spec["width"])
                total_h = int(spec["top_h"]) + 1 + int(spec["bottom_h"])
                if spec["side"] == "up":
                    min_r0 = int(spec["marker_gap"])
                    max_r0 = h - total_h - width
                else:
                    min_r0 = width
                    max_r0 = h - total_h - int(spec["marker_gap"])
                min_c0 = 0
                max_c0 = w - width

            if min_r0 > max_r0 or min_c0 > max_c0:
                return None

            r0 = random.randint(min_r0, max_r0)
            c0 = random.randint(min_c0, max_c0)
            candidate = _materialize_spec(spec, r0, c0, h, w)
            if candidate is None:
                continue
            if not input_reserved.isdisjoint(candidate["input_reserve"]):
                continue
            if not output_reserved.isdisjoint(candidate["output_reserve"]):
                continue
            layout = candidate
            break

        if layout is None:
            return None

        placed.append(layout)
        input_reserved |= layout["input_reserve"]
        output_reserved |= layout["output_reserve"]

    grid = np.full((h, w), BG, dtype=int)
    for obj in placed:
        for r, c in obj["input_cells"]:
            grid[r, c] = int(obj["color"])
        mr, mc = obj["marker"]
        grid[mr, mc] = MAGENTA

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cracked-object scene")
