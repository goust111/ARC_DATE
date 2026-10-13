from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
GREY = 5
BLACK = 0
REGION_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _color_components(grid: np.ndarray, excluded: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in pts]
            cols = [cc for _, cc in pts]
            comps.append(
                {
                    "color": color,
                    "cells": pts,
                    "set": set(pts),
                    "area": len(pts),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _orientation_and_depth(cells: set[tuple[int, int]]) -> tuple[str, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    if width >= height:
        return "horizontal", max(1, width // 2)
    return "vertical", max(1, height // 2)


def _shadow_direction(marker: tuple[int, int], region: set[tuple[int, int]]) -> tuple[int, int]:
    mr, mc = marker
    adj = [(rr, cc) for rr, cc in ((mr - 1, mc), (mr + 1, mc), (mr, mc - 1), (mr, mc + 1)) if (rr, cc) in region]
    assert len(adj) == 1, "marker must touch exactly one region pixel"
    rr, cc = adj[0]
    return _sign(mr - rr), _sign(mc - cc)


def _edge_pixels(region: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(r, c) for r, c in region if (r + dr, c + dc) not in region]


def _apply_shadow(
    out: np.ndarray,
    region: set[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    depth: int,
) -> None:
    h, w = out.shape
    dr, dc = direction
    for r, c in _edge_pixels(region, direction):
        for dist in range(1, depth + 1):
            nr = r + dr * dist
            nc = c + dc * dist
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color if dist % 2 == 1 else GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each elongated colored region, infer its main axis from the bbox,
    then cast a striped shadow from the side marked by the adjacent grey pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    regions = [comp for comp in _color_components(g, {bg, GREY, BLACK}) if comp["area"] > 1]
    region_by_cell = {cell: idx for idx, comp in enumerate(regions) for cell in comp["cells"]}

    markers: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(g == GREY):
        rr, cc = int(r), int(c)
        touched = {(nr, nc) for nr, nc in _neighbors4(rr, cc, *g.shape) if (nr, nc) in region_by_cell}
        if len(touched) != 1:
            continue
        idx = region_by_cell[next(iter(touched))]
        markers[idx] = (rr, cc)

    out = g.copy()
    out[out == GREY] = bg

    ordered = sorted(
        enumerate(regions),
        key=lambda item: (
            int(item[1]["area"]),
            int(item[1]["bbox"][0]),
            int(item[1]["bbox"][1]),
            int(item[1]["color"]),
        ),
    )
    for idx, comp in ordered:
        if idx not in markers:
            continue
        direction = _shadow_direction(markers[idx], comp["set"])
        _orientation, depth = _orientation_and_depth(comp["set"])
        _apply_shadow(out, comp["set"], int(comp["color"]), direction, depth)

    return out


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _column_strip(box_h: int, box_w: int) -> set[tuple[int, int]] | None:
    intervals = []
    for length in (2, 3):
        if length > box_h:
            continue
        for top in range(box_h - length + 1):
            intervals.append((top, top + length - 1))
    if not intervals:
        return None

    for _ in range(100):
        seq = [random.choice(intervals)]
        for _col in range(1, box_w):
            prev = seq[-1]
            choices = [iv for iv in intervals if _overlap(prev, iv)]
            drifted = [iv for iv in choices if abs(iv[0] - prev[0]) <= 1]
            seq.append(random.choice(drifted or choices))
        if min(top for top, _ in seq) != 0:
            continue
        if max(bot for _, bot in seq) != box_h - 1:
            continue
        if box_h >= 3 and all(top == 0 and bot == box_h - 1 for top, bot in seq):
            continue
        cells = {(r, c) for c, (top, bot) in enumerate(seq) for r in range(top, bot + 1)}
        if box_h > 3 and len({top for top, _ in seq}) < 2:
            continue
        return cells
    return None


def _transpose(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, r) for r, c in cells}


def _make_region(kind: str) -> set[tuple[int, int]] | None:
    if kind == "horizontal":
        box_h = random.choice((2, 3))
        box_w = random.randint(5, 8)
        return _column_strip(box_h, box_w)
    if kind == "vertical":
        box_h = random.choice((2, 3))
        box_w = random.randint(5, 8)
        base = _column_strip(box_h, box_w)
        return None if base is None else _transpose(base)
    if kind == "square":
        size = random.randint(5, 7)
        return _column_strip(size, size)
    raise ValueError(kind)


def _marker_candidates(region: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    cands = []
    for r, c in _edge_pixels(region, direction):
        mr, mc = r + dr, c + dc
        adj = sum((nr, nc) in region for nr, nc in ((mr - 1, mc), (mr + 1, mc), (mr, mc - 1), (mr, mc + 1)))
        if adj == 1:
            cands.append((mr, mc))
    return cands


def _shadow_cells(region: set[tuple[int, int]], direction: tuple[int, int], depth: int) -> set[tuple[int, int]]:
    dr, dc = direction
    cells = set()
    for r, c in _edge_pixels(region, direction):
        for dist in range(1, depth + 1):
            cells.add((r + dr * dist, c + dc * dist))
    return cells


def _translate(cells: set[tuple[int, int]], off_r: int, off_c: int) -> set[tuple[int, int]]:
    return {(r + off_r, c + off_c) for r, c in cells}


def _expanded(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _footprint_bounds(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _build_object(direction: tuple[int, int], kind: str, color: int) -> dict | None:
    for _ in range(120):
        region = _make_region(kind)
        if region is None:
            continue
        orient, depth = _orientation_and_depth(region)
        if direction in (UP, DOWN) and orient != "horizontal":
            continue
        if direction in (LEFT, RIGHT) and orient != "vertical":
            continue
        markers = _marker_candidates(region, direction)
        if not markers:
            continue
        marker = random.choice(markers)
        shadow = _shadow_cells(region, direction, depth)
        footprint = set(region)
        footprint.add(marker)
        footprint |= shadow
        return {
            "region": region,
            "direction": direction,
            "marker": marker,
            "shadow": shadow,
            "footprint": footprint,
            "color": color,
            "area": len(region),
            "kind": kind,
        }
    return None


def _demo_positions(corner: str, h: int, w: int) -> dict[tuple[int, int], int]:
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, w - 2
    elif corner == "BL":
        r0, c0 = h - 2, 0
    elif corner == "BR":
        r0, c0 = h - 2, w - 2
    else:
        raise ValueError(corner)
    return {(r0, c0): BLACK, (r0, c0 + 1): 1, (r0 + 1, c0): 1, (r0 + 1, c0 + 1): BLACK}


def _place_object(spec: dict, occupied: set[tuple[int, int]], h: int, w: int) -> dict | None:
    r0, c0, r1, c1 = _footprint_bounds(spec["footprint"])
    row_lo = -r0
    row_hi = h - 1 - r1
    col_lo = -c0
    col_hi = w - 1 - c1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(positions)
    for off_r, off_c in positions:
        placed_footprint = _translate(spec["footprint"], off_r, off_c)
        reserve = _expanded(placed_footprint, h, w, pad=1)
        if reserve & occupied:
            continue
        placed = {
            "region": _translate(spec["region"], off_r, off_c),
            "marker": next(iter(_translate({spec["marker"]}, off_r, off_c))),
            "direction": spec["direction"],
            "color": spec["color"],
            "area": spec["area"],
            "kind": spec["kind"],
            "reserve": reserve,
        }
        return placed
    return None


def _render_expected(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    out[out == GREY] = BG
    ordered = sorted(objects, key=lambda obj: (int(obj["area"]), int(min(r for r, _ in obj["region"])), int(min(c for _, c in obj["region"]))))
    for obj in ordered:
        _orient, depth = _orientation_and_depth(obj["region"])
        _apply_shadow(out, obj["region"], int(obj["color"]), obj["direction"], depth)
    return out


def _validate_input(grid: np.ndarray, expected_regions: int) -> bool:
    bg = _most_common_color(grid)
    regions = [comp for comp in _color_components(grid, {bg, GREY, BLACK}) if comp["area"] > 1]
    if len(regions) != expected_regions:
        return False
    if len({comp["area"] for comp in regions}) != len(regions):
        return False
    region_by_cell = {cell: idx for idx, comp in enumerate(regions) for cell in comp["cells"]}
    markers = list(map(tuple, np.argwhere(grid == GREY)))
    if len(markers) != expected_regions:
        return False
    seen_regions = set()
    for r, c in markers:
        touched = {(nr, nc) for nr, nc in _neighbors4(int(r), int(c), *grid.shape) if (nr, nc) in region_by_cell}
        if len(touched) != 1:
            return False
        idx = region_by_cell[next(iter(touched))]
        if idx in seen_regions:
            return False
        seen_regions.add(idx)
    return len(seen_regions) == expected_regions


def _generate_attempt() -> np.ndarray | None:
    n_objects = 4 if random.random() < 0.55 else 5
    h = random.randint(22 if n_objects == 4 else 24, 28 if n_objects == 4 else 30)
    w = random.randint(22 if n_objects == 4 else 24, 28 if n_objects == 4 else 30)

    colors = list(random.sample(REGION_COLORS, n_objects + 1))
    demo_color = colors.pop()
    directions = [UP, DOWN, LEFT, RIGHT]
    if n_objects == 5:
        directions.append(random.choice((UP, DOWN, LEFT, RIGHT)))

    square_dir = random.choice((UP, DOWN)) if random.random() < 0.5 else None
    specs = []
    used_areas = set()
    for direction, color in zip(directions, colors):
        kind = "vertical"
        if direction in (UP, DOWN):
            kind = "square" if direction == square_dir else "horizontal"
        spec = _build_object(direction, kind, color)
        if spec is None or spec["area"] in used_areas:
            return None
        used_areas.add(spec["area"])
        specs.append(spec)
    specs.sort(key=lambda obj: len(obj["footprint"]), reverse=True)

    grid = np.full((h, w), BG, dtype=int)
    corner = random.choice(("TL", "TR", "BL", "BR"))
    demo = _demo_positions(corner, h, w)
    for cell, value in demo.items():
        grid[cell] = demo_color if value == 1 else value
    occupied = _expanded(set(demo.keys()), h, w, pad=1)

    placed_objects = []
    for spec in specs:
        placed = _place_object(spec, occupied, h, w)
        if placed is None:
            return None
        occupied |= placed["reserve"]
        placed_objects.append(placed)
        for r, c in placed["region"]:
            grid[r, c] = int(placed["color"])
        mr, mc = placed["marker"]
        grid[mr, mc] = GREY

    if not _validate_input(grid, n_objects):
        return None

    expected = _render_expected(grid, placed_objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a random puzzle instance with four guaranteed shadow directions,
    plus an optional fifth region for extra variety.
    """
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
