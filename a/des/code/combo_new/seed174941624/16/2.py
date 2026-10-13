import random
from collections import defaultdict

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
CANVAS_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400
PAIR_RETRIES = 300
PLACEMENT_RETRIES = 400
OUTPUT_GAP = 1
PAIR_COUNT_RANGE = (2, 5)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _pair_record(parts: list[dict[str, object]]) -> dict[str, int]:
    assert len(parts) == 2
    a, b = parts
    bbox_a = a["bbox"]
    bbox_b = b["bbox"]
    h1, w1 = _bbox_dims(bbox_a)
    h2, w2 = _bbox_dims(bbox_b)
    same_h = h1 == h2
    same_w = w1 == w2

    if same_h and not same_w:
        rect_h, rect_w = h1, w1 + w2
    elif same_w and not same_h:
        rect_h, rect_w = h1 + h2, w1
    elif same_h and same_w:
        ar0, ac0, ar1, ac1 = bbox_a
        br0, bc0, br1, bc1 = bbox_b
        row_gap = abs((ar0 + ar1) - (br0 + br1))
        col_gap = abs((ac0 + ac1) - (bc0 + bc1))
        if col_gap >= row_gap:
            rect_h, rect_w = h1, w1 + w2
        else:
            rect_h, rect_w = h1 + h2, w1
    else:
        raise AssertionError("paired fragments must share exactly one bbox dimension")

    anchor_r = min(int(bbox_a[0]), int(bbox_b[0]))
    anchor_c = min(int(bbox_a[1]), int(bbox_b[1]))
    return {
        "color": int(a["color"]),
        "rect_h": int(rect_h),
        "rect_w": int(rect_w),
        "area": int(rect_h * rect_w),
        "anchor_r": int(anchor_r),
        "anchor_c": int(anchor_c),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each color's two fragments, complete their implied rectangle, and pack the rectangles by area.

    Input structure:
    - The background is black.
    - Each non-black color appears in exactly two separate 4-connected fragments.
    - The two fragments of one color have bounding boxes that share one dimension, indicating whether they recombine horizontally or vertically.

    Rule:
    - For each color, find its two fragments and compute their minimal bounding boxes.
    - If the boxes have the same height, complete that color into a solid rectangle of height h and width w1+w2; if they have the same width, complete it into a solid rectangle of height h1+h2 and width w.
    - Sort the completed rectangles by area from small to large, place them left-to-right with one black column between neighbors, and bottom-align all rectangles in the output.

    Key insight:
    - The fragments are only clues for recovering each color's final rectangle size; the output ignores their original positions and shows the completed rectangles in area order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in _find_components(g):
        by_color[int(comp["color"])].append(comp)

    assert by_color
    pairs = []
    for parts in by_color.values():
        pairs.append(_pair_record(parts))

    pairs.sort(
        key=lambda item: (
            int(item["area"]),
            int(item["anchor_r"]),
            int(item["anchor_c"]),
            int(item["rect_h"]),
            int(item["rect_w"]),
        )
    )

    out_h = max(int(item["rect_h"]) for item in pairs)
    out_w = sum(int(item["rect_w"]) for item in pairs) + OUTPUT_GAP * (len(pairs) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for item in pairs:
        rect_h = int(item["rect_h"])
        rect_w = int(item["rect_w"])
        color = int(item["color"])
        out[out_h - rect_h : out_h, c : c + rect_w] = color
        c += rect_w + OUTPUT_GAP

    return out


def _sample_orientations(n_pairs: int) -> list[str]:
    if n_pairs == 1:
        return [random.choice(["h", "v"])]

    mode = random.random()
    if mode < 0.5:
        base = ["h", "v"] + [random.choice(["h", "v"]) for _ in range(n_pairs - 2)]
        random.shuffle(base)
        return base
    if mode < 0.75:
        return ["h"] * n_pairs
    return ["v"] * n_pairs


def _sample_pair_specs() -> list[dict[str, object]] | None:
    for _ in range(PAIR_RETRIES):
        n_pairs = random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
        if n_pairs > len(COLOR_POOL):
            continue

        colors = random.sample(COLOR_POOL, n_pairs)
        orientations = _sample_orientations(n_pairs)
        areas: set[int] = set()
        specs: list[dict[str, object]] = []
        total_out_w = 0
        max_out_h = 0

        ok = True
        for color, orient in zip(colors, orientations):
            chosen = None
            for _ in range(80):
                if orient == "h":
                    shared_h = random.randint(1, 6)
                    w1, w2 = random.sample(range(1, 7), 2)
                    rect_h, rect_w = shared_h, w1 + w2
                    pieces = [(shared_h, w1), (shared_h, w2)]
                else:
                    shared_w = random.randint(1, 6)
                    h1, h2 = random.sample(range(1, 7), 2)
                    rect_h, rect_w = h1 + h2, shared_w
                    pieces = [(h1, shared_w), (h2, shared_w)]

                area = rect_h * rect_w
                if area in areas:
                    continue
                chosen = {
                    "color": int(color),
                    "orient": orient,
                    "pieces": pieces,
                    "rect_h": int(rect_h),
                    "rect_w": int(rect_w),
                    "area": int(area),
                }
                break

            if chosen is None:
                ok = False
                break

            specs.append(chosen)
            areas.add(int(chosen["area"]))
            total_out_w += int(chosen["rect_w"])
            max_out_h = max(max_out_h, int(chosen["rect_h"]))

        total_out_w += OUTPUT_GAP * (n_pairs - 1)
        if ok and total_out_w <= 30 and max_out_h <= 30:
            return specs

    return None


def _region_is_free(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(int(occupied.shape[0]), r + h + 1)
    c1 = min(int(occupied.shape[1]), c + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_rect(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    max_r = int(grid.shape[0] - h - 1)
    max_c = int(grid.shape[1] - w - 1)
    if max_r < 1 or max_c < 1:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(1, max_r)
        c = random.randint(1, max_c)
        if not _region_is_free(occupied, r, c, h, w):
            continue
        grid[r : r + h, c : c + w] = int(color)
        occupied[r : r + h, c : c + w] = True
        return r, c

    return None


def _render_expected(specs: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: int(item["area"]))
    out_h = max(int(item["rect_h"]) for item in ordered)
    out_w = sum(int(item["rect_w"]) for item in ordered) + OUTPUT_GAP * (len(ordered) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for item in ordered:
        rect_h = int(item["rect_h"])
        rect_w = int(item["rect_w"])
        color = int(item["color"])
        out[out_h - rect_h : out_h, c : c + rect_w] = color
        c += rect_w + OUTPUT_GAP

    return out


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_room = int(r0)
    left_room = int(c0)
    bottom_room = int(grid.shape[0] - 1 - r1)
    right_room = int(grid.shape[1] - 1 - c1)

    top = random.randint(1, min(4, top_room))
    left = random.randint(1, min(4, left_room))
    bottom = random.randint(1, min(4, bottom_room))
    right = random.randint(1, min(4, right_room))

    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _background_is_clear(grid: np.ndarray) -> bool:
    fg = int(np.count_nonzero(grid != BG))
    bg = int(grid.size - fg)
    return bg > fg


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_pair_specs()
    if specs is None:
        return None

    grid = np.full((CANVAS_SIDE, CANVAS_SIDE), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    for spec in specs:
        color = int(spec["color"])
        pieces = list(spec["pieces"])
        random.shuffle(pieces)
        for h, w in pieces:
            placed = _place_rect(grid, occupied, color, int(h), int(w))
            if placed is None:
                return None

    cropped = _crop_with_margin(grid)
    if max(cropped.shape) > 30 or not _background_is_clear(cropped):
        return None

    expected = _render_expected(specs)
    if not np.array_equal(main(cropped), expected):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
