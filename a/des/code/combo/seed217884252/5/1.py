import random
from collections import defaultdict

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
PAIR_RETRIES = 250


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _component_info(grid: np.ndarray, pts: list[tuple[int, int]]) -> dict:
    h, w = grid.shape
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    cr, cc = pts[0]
    color = int(grid[cr, cc])
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    is_rect = len(pts) == area and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    return {
        "cells": pts,
        "cell": pts[0],
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "touches_border": r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1,
        "is_rect": is_rect,
    }


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    markers: list[dict] = []
    rects_by_color: dict[int, list[dict]] = defaultdict(list)
    for pts in _components(g != BG):
        info = _component_info(g, pts)
        if info["touches_border"]:
            if len(info["cells"]) == 1:
                markers.append(info)
            continue
        if info["is_rect"]:
            rects_by_color[info["color"]].append(info)

    pairs: list[dict] = []
    for color, rects in rects_by_color.items():
        if len(rects) != 2:
            continue

        a, b = rects
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]

        if ar0 == br0 and ar1 == br1:
            left, right = sorted(rects, key=lambda item: item["bbox"][1])
            _, lc0, _, lc1 = left["bbox"]
            rr0, rc0, rr1, _ = right["bbox"]
            assert lc1 + 1 <= rc0 - 1, "horizontal rectangle pair must have a positive gap"
            aligned_markers = [
                m
                for m in markers
                if m["cell"][1] in (0, w - 1) and 0 < m["cell"][0] < h - 1 and rr0 <= m["cell"][0] <= rr1
            ]
            assert len(aligned_markers) == 1, "horizontal pair must have exactly one aligned side marker"
            pairs.append(
                {
                    "rect_color": color,
                    "orientation": "h",
                    "fill_color": aligned_markers[0]["color"],
                    "gap": (rr0, lc1 + 1, rr1, rc0 - 1),
                }
            )
            continue

        if ac0 == bc0 and ac1 == bc1:
            top, bottom = sorted(rects, key=lambda item: item["bbox"][0])
            tr0, tc0, tr1, tc1 = top["bbox"]
            br0, _, _, _ = bottom["bbox"]
            assert tr1 + 1 <= br0 - 1, "vertical rectangle pair must have a positive gap"
            aligned_markers = [
                m
                for m in markers
                if m["cell"][0] in (0, h - 1) and 0 < m["cell"][1] < w - 1 and tc0 <= m["cell"][1] <= tc1
            ]
            assert len(aligned_markers) == 1, "vertical pair must have exactly one aligned top/bottom marker"
            pairs.append(
                {
                    "rect_color": color,
                    "orientation": "v",
                    "fill_color": aligned_markers[0]["color"],
                    "gap": (tr1 + 1, tc0, br0 - 1, tc1),
                }
            )

    assert pairs, "expected at least one rectangle pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the gap between each aligned same-color rectangle pair using the color
    of its aligned edge marker.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for pair in _extract_pairs(g):
        r0, c0, r1, c1 = pair["gap"]
        gap = out[r0 : r1 + 1, c0 : c1 + 1]
        gap[gap == BG] = pair["fill_color"]
    return out


def _interval_overlaps(existing: list[tuple[int, int]], start: int, stop: int) -> bool:
    return any(not (stop < a0 or a1 < start) for a0, a1 in existing)


def _reserve_bounds(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> tuple[int, int, int, int]:
    h, w = shape
    return max(1, r0 - 1), max(1, c0 - 1), min(h - 2, r1 + 1), min(w - 2, c1 + 1)


def _paint_box(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _place_horizontal(
    grid: np.ndarray,
    reserved: np.ndarray,
    used_row_bands: list[tuple[int, int]],
    rect_color: int,
    fill_color: int,
    rng: random.Random,
) -> dict | None:
    h, w = grid.shape
    for _ in range(PAIR_RETRIES):
        rect_h = rng.randint(2, min(5, h - 4))
        rect_w1 = rng.randint(2, min(5, w - 4))
        rect_w2 = rng.randint(2, min(5, w - 4))
        gap_w_room = w - 4 - rect_w1 - rect_w2
        if gap_w_room < 1:
            continue
        gap_w = rng.randint(1, min(4, gap_w_room))
        total_w = rect_w1 + gap_w + rect_w2

        row_start_max = h - 2 - rect_h
        col_start_max = w - 2 - total_w
        if row_start_max < 2 or col_start_max < 2:
            continue

        r0 = rng.randint(2, row_start_max)
        c0 = rng.randint(2, col_start_max)
        r1 = r0 + rect_h - 1
        c1 = c0 + total_w - 1

        if _interval_overlaps(used_row_bands, r0, r1):
            continue

        er0, ec0, er1, ec1 = _reserve_bounds(grid.shape, r0, c0, r1, c1)
        if reserved[er0 : er1 + 1, ec0 : ec1 + 1].any():
            continue

        marker_row = rng.randint(r0, r1)
        marker_col = rng.choice((0, w - 1))
        if grid[marker_row, marker_col] != BG:
            continue

        left_c1 = c0 + rect_w1 - 1
        right_c0 = left_c1 + gap_w + 1
        _paint_box(grid, r0, c0, r1, left_c1, rect_color)
        _paint_box(grid, r0, right_c0, r1, right_c0 + rect_w2 - 1, rect_color)
        grid[marker_row, marker_col] = fill_color
        reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True
        used_row_bands.append((r0, r1))
        return {
            "orientation": "h",
            "rect_color": rect_color,
            "fill_color": fill_color,
            "gap": (r0, left_c1 + 1, r1, right_c0 - 1),
        }
    return None


def _place_vertical(
    grid: np.ndarray,
    reserved: np.ndarray,
    used_col_bands: list[tuple[int, int]],
    rect_color: int,
    fill_color: int,
    rng: random.Random,
) -> dict | None:
    h, w = grid.shape
    for _ in range(PAIR_RETRIES):
        rect_w = rng.randint(2, min(5, w - 4))
        rect_h1 = rng.randint(2, min(5, h - 4))
        rect_h2 = rng.randint(2, min(5, h - 4))
        gap_h_room = h - 4 - rect_h1 - rect_h2
        if gap_h_room < 1:
            continue
        gap_h = rng.randint(1, min(4, gap_h_room))
        total_h = rect_h1 + gap_h + rect_h2

        row_start_max = h - 2 - total_h
        col_start_max = w - 2 - rect_w
        if row_start_max < 2 or col_start_max < 2:
            continue

        r0 = rng.randint(2, row_start_max)
        c0 = rng.randint(2, col_start_max)
        r1 = r0 + total_h - 1
        c1 = c0 + rect_w - 1

        if _interval_overlaps(used_col_bands, c0, c1):
            continue

        er0, ec0, er1, ec1 = _reserve_bounds(grid.shape, r0, c0, r1, c1)
        if reserved[er0 : er1 + 1, ec0 : ec1 + 1].any():
            continue

        marker_col = rng.randint(c0, c1)
        marker_row = rng.choice((0, h - 1))
        if grid[marker_row, marker_col] != BG:
            continue

        top_r1 = r0 + rect_h1 - 1
        bottom_r0 = top_r1 + gap_h + 1
        _paint_box(grid, r0, c0, top_r1, c1, rect_color)
        _paint_box(grid, bottom_r0, c0, bottom_r0 + rect_h2 - 1, c1, rect_color)
        grid[marker_row, marker_col] = fill_color
        reserved[er0 : er1 + 1, ec0 : ec1 + 1] = True
        used_col_bands.append((c0, c1))
        return {
            "orientation": "v",
            "rect_color": rect_color,
            "fill_color": fill_color,
            "gap": (top_r1 + 1, c0, bottom_r0 - 1, c1),
        }
    return None


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["gap"]
        out[r0 : r1 + 1, c0 : c1 + 1] = spec["fill_color"]
    return out


def _direction_plan(pair_count: int, rng: random.Random) -> list[str]:
    if pair_count == 1:
        return [rng.choice(("h", "v"))]
    mode = rng.choice(("h", "v", "mixed"))
    if mode == "h":
        return ["h"] * pair_count
    if mode == "v":
        return ["v"] * pair_count
    dirs = ["h", "v"] + [rng.choice(("h", "v")) for _ in range(pair_count - 2)]
    rng.shuffle(dirs)
    return dirs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 18)
    w = rng.randint(10, 18)
    max_pairs = 1 + int(h >= 12 and w >= 12) + int(h >= 15 and w >= 15)
    pair_count = rng.randint(1, min(3, max_pairs))

    directions = _direction_plan(pair_count, rng)
    colors = list(COLORS)
    rng.shuffle(colors)
    rect_colors = colors[:pair_count]
    fill_colors = colors[pair_count : 2 * pair_count]

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    used_row_bands: list[tuple[int, int]] = []
    used_col_bands: list[tuple[int, int]] = []
    specs: list[dict] = []

    items = list(zip(directions, rect_colors, fill_colors))
    rng.shuffle(items)
    for direction, rect_color, fill_color in items:
        if direction == "h":
            spec = _place_horizontal(grid, reserved, used_row_bands, rect_color, fill_color, rng)
        else:
            spec = _place_vertical(grid, reserved, used_col_bands, rect_color, fill_color, rng)
        if spec is None:
            return None
        specs.append(spec)

    try:
        pairs = _extract_pairs(grid)
    except AssertionError:
        return None
    if len(pairs) != pair_count:
        return None

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate random grids with 1-3 rectangle pairs, mixing horizontal and
    vertical alignments across samples while keeping every pair unambiguous.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
