import random
from typing import Iterable

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))

RECT_COUNT_OPTIONS = (1, 2, 2, 3)
GRID_MIN_SIZE = 12
GRID_MAX_SIZE = 30
RECT_MIN_SIDE = 5
RECT_MAX_SIDE = 12
PLACEMENT_TRIES = 160
GENERATION_TRIES = 400

CORNER_OFFSETS = {
    "TL": ((0, 0), (0, 1), (1, 0)),
    "TR": ((0, 0), (0, -1), (1, 0)),
    "BL": ((0, 0), (0, 1), (-1, 0)),
    "BR": ((0, 0), (0, -1), (-1, 0)),
}


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _l_marker_from_component(comp: list[tuple[int, int]]) -> dict | None:
    if len(comp) != 3:
        return None
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    if bottom - top != 1 or right - left != 1:
        return None

    block = {(top, left), (top, right), (bottom, left), (bottom, right)}
    cells = set(comp)
    missing = block - cells
    if len(missing) != 1:
        return None
    miss = next(iter(missing))

    if miss == (bottom, right):
        corner_type = "TL"
        corner = (top, left)
    elif miss == (bottom, left):
        corner_type = "TR"
        corner = (top, right)
    elif miss == (top, right):
        corner_type = "BL"
        corner = (bottom, left)
    else:
        corner_type = "BR"
        corner = (bottom, right)

    return {
        "corner_type": corner_type,
        "corner": corner,
        "cells": tuple(sorted(comp)),
    }


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        comps = _components(g == color)
        markers = []
        for comp in comps:
            marker = _l_marker_from_component(comp)
            assert marker is not None, f"color {color} must consist of 3-pixel L markers"
            markers.append(marker)
        assert len(markers) == 2, f"color {color} must appear as exactly two L markers"

        by_corner = {m["corner_type"]: m["corner"] for m in markers}
        corner_types = set(by_corner)
        rows = [corner[0] for corner in by_corner.values()]
        cols = [corner[1] for corner in by_corner.values()]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)

        if corner_types == {"TL", "BR"}:
            assert by_corner["TL"] == (top, left)
            assert by_corner["BR"] == (bottom, right)
            orientation = "desc"
        elif corner_types == {"TR", "BL"}:
            assert by_corner["TR"] == (top, right)
            assert by_corner["BL"] == (bottom, left)
            orientation = "asc"
        else:
            raise AssertionError(f"color {color} must mark opposite rectangle corners")

        rects.append(
            {
                "color": color,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "height": bottom - top + 1,
                "width": right - left + 1,
                "orientation": orientation,
            }
        )

    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            overlap = not (
                a["bottom"] < b["top"]
                or b["bottom"] < a["top"]
                or a["right"] < b["left"]
                or b["right"] < a["left"]
            )
            assert not overlap, "rectangle regions must not overlap"

    return rects


def _stripe_cell(dr: int, dc: int, width: int, orientation: str) -> bool:
    if orientation == "desc":
        normal = dr - dc
    else:
        normal = dr + dc - (width - 1)
    return ((normal // 2) % 2) == 0


def _paint_l(grid: np.ndarray, color: int, corner_type: str, corner_r: int, corner_c: int) -> None:
    for dr, dc in CORNER_OFFSETS[corner_type]:
        grid[corner_r + dr, corner_c + dc] = color


def _draw_input_rectangles(grid: np.ndarray, rects: list[dict]) -> None:
    for rect in rects:
        color = int(rect["color"])
        top = int(rect["top"])
        bottom = int(rect["bottom"])
        left = int(rect["left"])
        right = int(rect["right"])
        if rect["orientation"] == "desc":
            _paint_l(grid, color, "TL", top, left)
            _paint_l(grid, color, "BR", bottom, right)
        else:
            _paint_l(grid, color, "TR", top, right)
            _paint_l(grid, color, "BL", bottom, left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marker-defined rectangle with diagonal stripes whose slope is encoded by paired L markers.

    Input structure:
    - The background is black.
    - Each non-black color appears as exactly two 3-pixel L markers.
    - Every L marker is a 2x2 block with one missing cell, and its corner sits on a corner of an implied rectangle.
    - Same-colored marker pairs occupy opposite corners of non-overlapping rectangles, with both L openings facing inward.

    Rule:
    - For each color, detect the two L markers and recover which rectangle corners they occupy.
    - The two marker corners define the axis-aligned rectangle to transform.
    - If the pair is top-left plus bottom-right, draw `\\`-slanting diagonal stripes inside the rectangle.
    - If the pair is top-right plus bottom-left, draw `/`-slanting diagonal stripes inside the rectangle.
    - Preserve the original L marker pixels and fill only background cells of the rectangle.

    Key insight:
    - The missing cell of each 2x2 L reveals which rectangle corner that marker occupies, so the pair jointly encodes both the rectangle extent and the stripe slope.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for rect in _extract_rectangles(g):
        top = int(rect["top"])
        bottom = int(rect["bottom"])
        left = int(rect["left"])
        right = int(rect["right"])
        width = int(rect["width"])
        color = int(rect["color"])
        orientation = str(rect["orientation"])
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if int(out[r, c]) == BG and _stripe_cell(r - top, c - left, width, orientation):
                    out[r, c] = color
    return out


def _random_rectangles(rng: random.Random, height: int, width: int) -> list[dict] | None:
    count_options = list(RECT_COUNT_OPTIONS)
    if min(height, width) < 16:
        count_options = [1, 2, 2]
    count = rng.choice(count_options)

    rects: list[dict] = []
    reserved = np.zeros((height, width), dtype=bool)

    colors = rng.sample(list(ALL_COLORS), count)
    orientations = [rng.choice(["desc", "asc"]) for _ in range(count)]
    if count >= 2 and rng.random() < 0.5:
        orientations[0] = "desc"
        orientations[1] = "asc"
        rng.shuffle(orientations)

    for color, orientation in zip(colors, orientations):
        placed = False
        side_cap = min(RECT_MAX_SIDE, height, width)
        if count == 3:
            side_cap = min(side_cap, 10)
        for _ in range(PLACEMENT_TRIES):
            rect_h = rng.randint(RECT_MIN_SIDE, side_cap)
            rect_w = rng.randint(RECT_MIN_SIDE, side_cap)
            top = rng.randint(0, height - rect_h)
            left = rng.randint(0, width - rect_w)
            bottom = top + rect_h - 1
            right = left + rect_w - 1

            pad = 1
            rs = slice(max(0, top - pad), min(height, bottom + pad + 1))
            cs = slice(max(0, left - pad), min(width, right + pad + 1))
            if bool(reserved[rs, cs].any()):
                continue

            rect = {
                "color": int(color),
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "orientation": orientation,
            }
            rects.append(rect)
            reserved[rs, cs] = True
            placed = True
            break
        if not placed:
            return None

    return rects


def _valid_generated_grid(grid: np.ndarray, expected_count: int) -> bool:
    try:
        rects = _extract_rectangles(grid)
        out = main(grid)
    except AssertionError:
        return False

    if len(rects) != expected_count:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(grid, out):
        return False

    for rect in rects:
        region_in = grid[rect["top"] : rect["bottom"] + 1, rect["left"] : rect["right"] + 1]
        region_out = out[rect["top"] : rect["bottom"] + 1, rect["left"] : rect["right"] + 1]
        if int(np.count_nonzero(region_out == rect["color"])) <= int(np.count_nonzero(region_in == rect["color"])):
            return False

    orientations = {rect["orientation"] for rect in rects}
    return bool(orientations)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_TRIES):
        height = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
        width = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
        rects = _random_rectangles(rng, height, width)
        if rects is None:
            continue

        grid = np.full((height, width), BG, dtype=int)
        _draw_input_rectangles(grid, rects)

        if _valid_generated_grid(grid, len(rects)):
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
