from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 500
TOP_LEVEL_PAD = 2
SIBLING_PAD = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rect_h(rect: tuple[int, int, int, int]) -> int:
    return rect[2] - rect[0] + 1


def _rect_w(rect: tuple[int, int, int, int]) -> int:
    return rect[3] - rect[1] + 1


def _expand_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _draw_rect(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    side_colors: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = rect
    top, right, bottom, left = side_colors
    grid[r0, c0:c1 + 1] = top
    grid[r1, c0:c1 + 1] = bottom
    if r1 - r0 >= 2:
        grid[r0 + 1:r1, c0] = left
        grid[r0 + 1:r1, c1] = right


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != BG:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == BG:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            if touches_border:
                continue

            boundary = set()
            for cr, cc in cells:
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != BG:
                        boundary.add((nr, nc))

            if not boundary:
                continue

            counts: Counter[int] = Counter()
            first_pos: dict[int, int] = {}
            for br, bc in sorted(boundary):
                color = int(g[br, bc])
                counts[color] += 1
                first_pos.setdefault(color, br * w + bc)

            distinct = len(counts)
            target_count = max(counts.values()) if distinct % 2 == 1 else min(counts.values())
            candidates = [color for color, count in counts.items() if count == target_count]
            fill_color = min(candidates, key=lambda color: first_pos[color])

            regions.append(
                {
                    "cells": cells,
                    "boundary": boundary,
                    "counts": dict(counts),
                    "distinct": distinct,
                    "fill_color": fill_color,
                    "decisive": len(candidates) == 1,
                }
            )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black region with a color chosen from the colors on its surrounding line segments.

    Input structure:
    - The background is black.
    - One-pixel-wide colored horizontal and vertical line segments join at their endpoints.
    - Those segments form one or more closed orthogonal boundaries, possibly with nesting.
    - The interiors of the closed regions are black 4-connected background cells.

    Rule:
    - Split the black background into 4-connected regions using every colored line pixel as a barrier.
    - Ignore the border-connected outside background and process every remaining enclosed region.
    - For one region, collect every colored boundary pixel that is 4-neighbor-adjacent to the region and count how many boundary pixels belong to each color.
    - If the region's boundary uses an odd number of different colors, fill the region with the most frequent boundary color; if it uses an even number of different colors, fill it with the rarest boundary color.

    Key insight:
    - The decision is made per enclosed background component, not per rectangle or per side: a nested region can be bounded by multiple separate loops, and all adjacent boundary colors must be counted together.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _analyze_regions(g):
        fill_color = int(region["fill_color"])
        for r, c in region["cells"]:
            out[r, c] = fill_color

    return out


def _place_top_level_rect(
    used: np.ndarray,
    rng: random.Random,
    h_range: tuple[int, int],
    w_range: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    h_lim, w_lim = used.shape
    lo_h = max(4, h_range[0])
    hi_h = min(h_range[1], h_lim - 2)
    lo_w = max(4, w_range[0])
    hi_w = min(w_range[1], w_lim - 2)
    if lo_h > hi_h or lo_w > hi_w:
        return None

    for _ in range(300):
        rect_h = rng.randint(lo_h, hi_h)
        rect_w = rng.randint(lo_w, hi_w)
        r0 = rng.randint(1, h_lim - rect_h - 1)
        c0 = rng.randint(1, w_lim - rect_w - 1)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        rr0 = max(0, r0 - TOP_LEVEL_PAD)
        cc0 = max(0, c0 - TOP_LEVEL_PAD)
        rr1 = min(h_lim - 1, rect[2] + TOP_LEVEL_PAD)
        cc1 = min(w_lim - 1, rect[3] + TOP_LEVEL_PAD)
        if not used[rr0:rr1 + 1, cc0:cc1 + 1].any():
            used[rr0:rr1 + 1, cc0:cc1 + 1] = True
            return rect
    return None


def _place_child_rect(
    parent: tuple[int, int, int, int],
    siblings: list[tuple[int, int, int, int]],
    rng: random.Random,
    h_range: tuple[int, int],
    w_range: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    max_h = _rect_h(parent) - 4
    max_w = _rect_w(parent) - 4
    lo_h = max(4, h_range[0])
    hi_h = min(h_range[1], max_h)
    lo_w = max(4, w_range[0])
    hi_w = min(w_range[1], max_w)
    if lo_h > hi_h or lo_w > hi_w:
        return None

    for _ in range(300):
        rect_h = rng.randint(lo_h, hi_h)
        rect_w = rng.randint(lo_w, hi_w)
        r0_min = parent[0] + 2
        r0_max = parent[2] - rect_h - 1
        c0_min = parent[1] + 2
        c0_max = parent[3] - rect_w - 1
        if r0_min > r0_max or c0_min > c0_max:
            continue
        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        inflated = _expand_rect(rect, SIBLING_PAD)
        if any(_overlap(inflated, sib) for sib in siblings):
            continue
        return rect
    return None


def _odd_leaf_pattern(
    rect: tuple[int, int, int, int],
    palette: list[int],
    rng: random.Random,
) -> tuple[int, int, int, int]:
    a, b, c = rng.sample(palette, 3)
    if _rect_w(rect) >= _rect_h(rect):
        return a, b, a, c
    return b, a, c, a


def _even_leaf_pattern(palette: list[int], rng: random.Random) -> tuple[int, int, int, int]:
    dominant, rare = rng.sample(palette, 2)
    side_colors = [dominant, dominant, dominant, dominant]
    side_colors[rng.randrange(4)] = rare
    return tuple(side_colors)


def _random_pattern(palette: list[int], rng: random.Random) -> tuple[int, int, int, int]:
    distinct = rng.randint(1, 4)
    colors = rng.sample(palette, distinct)
    sides = colors[:]
    while len(sides) < 4:
        sides.append(rng.choice(colors))
    rng.shuffle(sides)
    return tuple(sides)


def _make_nested_cluster(
    used: np.ndarray,
    palette: list[int],
    rng: random.Random,
) -> list[dict] | None:
    root = _place_top_level_rect(used, rng, (10, 15), (10, 15))
    if root is None:
        return None

    nodes = [{"rect": root, "colors": _random_pattern(palette, rng), "children": []}]
    child_rects: list[tuple[int, int, int, int]] = []

    want_grandchild = _rect_h(root) >= 12 and _rect_w(root) >= 12 and rng.random() < 0.6
    host_child = None
    if want_grandchild:
        host_child = _place_child_rect(root, child_rects, rng, (8, 10), (8, 10))
        if host_child is not None:
            child_rects.append(host_child)
            nodes[0]["children"].append({"rect": host_child, "colors": _random_pattern(palette, rng), "children": []})

    extra_children = 1 if host_child is None else 0
    if rng.random() < 0.55:
        extra_children += 1

    for _ in range(extra_children):
        child = _place_child_rect(root, child_rects, rng, (4, 7), (4, 7))
        if child is None:
            continue
        child_rects.append(child)
        nodes[0]["children"].append({"rect": child, "colors": _random_pattern(palette, rng), "children": []})

    if host_child is not None:
        grandchild = _place_child_rect(host_child, [], rng, (4, 6), (4, 6))
        if grandchild is not None:
            nodes[0]["children"][0]["children"].append(
                {"rect": grandchild, "colors": _random_pattern(palette, rng), "children": []}
            )

    return nodes


def _draw_nodes(grid: np.ndarray, nodes: list[dict]) -> None:
    for node in nodes:
        _draw_rect(grid, node["rect"], node["colors"])
        _draw_nodes(grid, node["children"])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    grid = np.zeros((h, w), dtype=int)
    used = np.zeros((h, w), dtype=bool)
    palette = rng.sample(PALETTE, rng.randint(5, len(PALETTE)))

    if rng.random() < 0.5:
        odd_rect = _place_top_level_rect(used, rng, (4, 7), (6, 10))
    else:
        odd_rect = _place_top_level_rect(used, rng, (6, 10), (4, 7))
    even_rect = _place_top_level_rect(used, rng, (4, 9), (4, 9))
    if odd_rect is None or even_rect is None:
        return None

    nodes = [
        {"rect": odd_rect, "colors": _odd_leaf_pattern(odd_rect, palette, rng), "children": []},
        {"rect": even_rect, "colors": _even_leaf_pattern(palette, rng), "children": []},
    ]

    if rng.random() < 0.7:
        nested = _make_nested_cluster(used, palette, rng)
        if nested is not None:
            nodes.extend(nested)

    _draw_nodes(grid, nodes)

    regions = _analyze_regions(grid)
    if len(regions) < 2:
        return None
    if any(not region["decisive"] for region in regions):
        return None
    parities = {region["distinct"] % 2 for region in regions}
    if parities != {0, 1}:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
