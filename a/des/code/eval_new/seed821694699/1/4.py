from __future__ import annotations

from collections import deque
import random
from typing import Dict, List, Tuple

import numpy as np

Point = Tuple[int, int]
BG = 0
DIRECTIONS: Tuple[Point, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _connected_component(grid: np.ndarray, start: Point, color: int, mask: np.ndarray) -> List[Point]:
    h, w = grid.shape
    q = deque([start])
    component: List[Point] = []
    mask[start[0], start[1]] = True

    while q:
        r, c = q.popleft()
        component.append((r, c))
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if mask[nr, nc] or grid[nr, nc] != color:
                continue
            mask[nr, nc] = True
            q.append((nr, nc))
    return component


def _find_separator_rows(grid: np.ndarray) -> Tuple[int, int, int]:
    h, w = grid.shape
    candidates: List[Tuple[int, int, int]] = []

    for r in range(h):
        row = grid[r]
        nz = row[row != BG]
        if nz.size == 0:
            continue
        vals, counts = np.unique(nz, return_counts=True)
        if len(vals) != 1:
            continue
        color = int(vals[0])
        count = int(counts[0])
        if count >= max(1, w - 2):
            candidates.append((r, color, count))

    if not candidates:
        # Soft fallback for uncommon formatting.
        for r in range(h):
            row = grid[r]
            nz = row[row != BG]
            if nz.size == 0:
                continue
            vals, counts = np.unique(nz, return_counts=True)
            k = int(counts.max())
            color = int(vals[int(np.argmax(counts))])
            if k >= max(1, int(w * 0.9)):
                candidates.append((r, color, k))

    if not candidates:
        mid = h // 2
        color = 10 if 10 != BG else 11
        return mid, mid, color

    row, color, _ = max(candidates, key=lambda t: t[2])
    sep_rows = {row}

    def _is_sep_row(ridx: int, sep_color: int) -> bool:
        rr = grid[ridx]
        nz = rr[rr != BG]
        if nz.size == 0:
            return False
        if np.any(nz != sep_color):
            return False
        return nz.size >= max(1, w - 2)

    up = row - 1
    while up >= 0 and _is_sep_row(up, color):
        sep_rows.add(up)
        up -= 1

    down = row + 1
    while down < h and _is_sep_row(down, color):
        sep_rows.add(down)
        down += 1

    lo, hi = min(sep_rows), max(sep_rows)
    return lo, hi, color


def _components_by_color_top(grid: np.ndarray, max_r: int, banned_color: int) -> List[Dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Dict] = []

    for r in range(max_r):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or color == banned_color or visited[r, c]:
                continue

            comp = _connected_component(grid, (r, c), color, visited)
            r0 = min(pt[0] for pt in comp)
            r1 = max(pt[0] for pt in comp)
            c0 = min(pt[1] for pt in comp)
            c1 = max(pt[1] for pt in comp)
            hbox = r1 - r0 + 1
            wbox = c1 - c0 + 1
            area = hbox * wbox

            if hbox < 3 or wbox < 3:
                continue
            if area < len(comp) + 1:
                continue

            fill_ratio = len(comp) / area
            if fill_ratio < 0.45 or fill_ratio >= 0.98:
                continue

            touches_top = any(rr == r0 for rr, _ in comp)
            touches_bottom = any(rr == r1 for rr, _ in comp)
            touches_left = any(cc == c0 for _, cc in comp)
            touches_right = any(cc == c1 for _, cc in comp)
            if not (touches_top and touches_bottom and touches_left and touches_right):
                continue

            comps.append(
                {
                    "color": color,
                    "cells": comp,
                    "bbox": (r0, r1, c0, c1),
                    "center": ((c0 + c1) / 2.0),
                    "area": area,
                    "count": len(comp),
                }
            )

    # Deduplicate accidental overlaps if duplicate color components appear touching in a chain.
    comps.sort(key=lambda x: (x["bbox"][0], x["bbox"][2], -x["count"]))
    return comps


def _holes_sorted_desc(grid: np.ndarray, template: Dict) -> List[List[Point]]:
    r0, r1, c0, c1 = template["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    visited = np.zeros((h, w), dtype=bool)
    holes: List[List[Point]] = []

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if grid[rr, cc] != BG:
                continue

            lr, lc = rr - r0, cc - c0
            if visited[lr, lc]:
                continue

            q = deque([(rr, cc)])
            visited[lr, lc] = True
            comp: List[Point] = []
            touches_border = False

            while q:
                cr, cc2 = q.popleft()
                comp.append((cr, cc2))
                if cr == r0 or cr == r1 or cc2 == c0 or cc2 == c1:
                    touches_border = True

                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc2 + dc
                    if nr < r0 or nr > r1 or nc < c0 or nc > c1:
                        continue
                    lr2, lc2 = nr - r0, nc - c0
                    if visited[lr2, lc2] or grid[nr, nc] != BG:
                        continue
                    visited[lr2, lc2] = True
                    q.append((nr, nc))

            if not touches_border:
                holes.append(comp)

    holes.sort(key=len, reverse=True)
    return holes


def _collect_legend_bars(grid: np.ndarray, y0: int, sep_color: int) -> List[Dict]:
    h, w = grid.shape
    bars: List[Dict] = []

    for r in range(y0, h):
        row = grid[r]
        cols = np.where((row != BG) & (row != sep_color))[0]
        if cols.size < 1:
            continue

        vals = row[cols]
        if len(cols) != len(np.unique(cols)):
            continue
        if len(vals) != len(np.unique(vals)):
            continue
        if cols.size > 1 and np.any(np.diff(cols) == 1):
            continue

        bars.append(
            {
                "row": r,
                "cols": cols.astype(int).tolist(),
                "colors": [int(v) for v in vals.tolist()],
                "center": float(cols.mean()),
            }
        )

    bars.sort(key=lambda b: b["center"])
    return bars


def _pair_by_center(templates: List[Dict], bars: List[Dict]) -> List[Tuple[Dict, Dict]]:
    if not templates or not bars:
        return []

    sorted_templates = sorted(templates, key=lambda t: t["center"])
    sorted_bars = sorted(bars, key=lambda b: b["center"])

    if len(sorted_templates) == len(sorted_bars):
        return list(zip(sorted_templates, sorted_bars))

    used = set()
    pairs: List[Tuple[Dict, Dict]] = []

    for bar in sorted_bars:
        best_idx = -1
        best_dist = 10**9
        for i, tpl in enumerate(sorted_templates):
            if i in used:
                continue
            dist = abs(tpl["center"] - bar["center"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx == -1:
            continue
        used.add(best_idx)
        pairs.append((sorted_templates[best_idx], bar))

        if len(used) == len(sorted_templates):
            break

    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Solve the repair-and-paint puzzle.

1) Parse the scene and recover structural regions
   - detect the horizontal grey separator line,
   - detect top-area template rectangles with background holes,
   - detect bottom legend bars.
2) Pair each legend bar with its template by horizontal-center proximity.
3) For each paired template, sort holes by area descending and assign legend colors
   in left-to-right order to those holes.
4) Fill hole cells with assigned colors and erase legend pixels, preserving everything else.

Parameters:
    grid (np.ndarray): ARC-style integer color matrix (BG=0 background).

Returns:
    np.ndarray: solved grid with only the holes recolored and legend bar cells cleared.

Algorithm:
    Single-pass connected-component extraction is used for template candidates and holes,
    with O(H*W) complexity for each major region scan.

Limitations:
    Assumes input follows the described format: one (or few-row) separator strip, templates
    only in the upper region, and separated legend bars in the lower region.
    """
    arr = np.array(grid, dtype=int, copy=True)
    h, w = arr.shape

    sep_top, sep_bottom, sep_color = _find_separator_rows(arr)
    top_h = sep_top
    bottom_start = sep_bottom + 1

    templates = _components_by_color_top(arr, top_h, sep_color)
    if not templates:
        return arr

    bars = _collect_legend_bars(arr, bottom_start, sep_color)
    if not bars:
        return arr

    pairs = _pair_by_center(templates, bars)

    for template, bar in pairs:
        holes = _holes_sorted_desc(arr, template)
        if not holes:
            continue

        colors = bar["colors"]
        assign_n = min(len(holes), len(colors))
        for i in range(assign_n):
            fill_color = colors[i]
            for r, c in holes[i]:
                arr[r, c] = fill_color

    for bar in bars:
        y = bar["row"]
        for x in bar["cols"]:
            arr[y, int(x)] = BG

    return arr


def _pick_random_connected_blob(
    height: int, width: int, target: int, rng: random.Random, forbidden: np.ndarray
) -> List[Point]:
    free: List[Point] = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if not forbidden[r, c]
    ]
    if not free:
        return []

    start = rng.choice(free)
    forbidden[start[0], start[1]] = True
    comp = [start]
    frontier = [start]

    while len(comp) < target and frontier:
        cr, cc = rng.choice(frontier)
        neigh: List[Point] = []
        for dr, dc in DIRECTIONS:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if forbidden[nr, nc]:
                continue
            neigh.append((nr, nc))

        if not neigh:
            frontier.remove((cr, cc))
            continue

        nr, nc = rng.choice(neigh)
        forbidden[nr, nc] = True
        comp.append((nr, nc))
        frontier.append((nr, nc))

    return comp


def _find_holes_for_template(grid: np.ndarray, rect: Tuple[int, int, int, int]) -> List[List[Point]]:
    fake_tpl = {"bbox": rect}
    return _holes_sorted_desc(grid, fake_tpl)


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(14, 30)
    w = rng.randint(14, 30)

    sep_top = rng.randint(5, h - 9)
    sep_thick = rng.randint(1, 2)
    sep_bottom = min(h - 5, sep_top + sep_thick - 1)
    sep_color = rng.choice([7, 8, 9, 11, 12, 13])

    arr = np.zeros((h, w), dtype=int)
    for r in range(sep_top, sep_bottom + 1):
        arr[r, :] = sep_color

    top_h = sep_top
    max_templates = max(1, min(4, max(1, (top_h - 1) // 4)))
    num_templates = rng.randint(1, max_templates)

    forbidden_top = np.zeros((top_h, w), dtype=bool)
    used_colors = {BG, sep_color}
    templates: List[Dict] = []

    for _ in range(num_templates):
        for _ in range(120):
            th = rng.randint(4, min(9, max(4, top_h - 1)))
            tw = rng.randint(4, min(10, max(4, w - 1)))
            if top_h - th <= 0 or w - tw <= 0:
                continue

            tr = rng.randint(0, top_h - th)
            tc = rng.randint(0, w - tw)
            if np.any(forbidden_top[tr : tr + th, tc : tc + tw]):
                continue

            template_color_pool = [c for c in range(20, 240) if c not in used_colors]
            if not template_color_pool:
                template_color_pool = list(range(20, 250))
            t_color = rng.choice(template_color_pool)
            used_colors.add(t_color)

            arr[tr : tr + th, tc : tc + tw] = t_color
            forbidden_top[tr : tr + th, tc : tc + tw] = True

            inner_h = th - 2
            inner_w = tw - 2
            if inner_h > 0 and inner_w > 0:
                num_holes = rng.randint(1, 3)
                hole_cells: List[Point] = []
                forbidden_hole = np.zeros((inner_h, inner_w), dtype=bool)

                max_holes = min(num_holes, inner_h * inner_w)
                for _h in range(max_holes):
                    area = rng.randint(1, max(1, inner_h * inner_w))
                    hole = _pick_random_connected_blob(inner_h, inner_w, area, rng, forbidden_hole)
                    if not hole:
                        continue
                    for rr, cc in hole:
                        gr = tr + 1 + rr
                        gc = tc + 1 + cc
                        arr[gr, gc] = BG
                        hole_cells.append((gr, gc))

                    if len(hole_cells) >= inner_h * inner_w:
                        break

                if not hole_cells:
                    # Guarantee at least one defect for puzzle compliance.
                    rr = tr + 1 + rng.randint(0, max(0, inner_h - 1))
                    cc = tc + 1 + rng.randint(0, max(0, inner_w - 1))
                    arr[rr, cc] = BG
                    hole_cells.append((rr, cc))
            else:
                rr = tr + rng.randint(0, th - 1)
                cc = tc + rng.randint(0, tw - 1)
                arr[rr, cc] = BG
                hole_cells = [(rr, cc)]

            holes_by_size = _find_holes_for_template(arr, (tr, tr + th - 1, tc, tc + tw - 1))
            templates.append(
                {
                    "bbox": (tr, tr + th - 1, tc, tc + tw - 1),
                    "holes": holes_by_size,
                    "color": t_color,
                }
            )
            break

    bottom_start = sep_bottom + 1
    available_rows = [r for r in range(bottom_start, h)]
    if not available_rows:
        return arr

    row_choices: List[int]
    if len(available_rows) >= len(templates):
        row_choices = rng.sample(available_rows, len(templates))
    else:
        row_choices = [rng.choice(available_rows) for _ in templates]

    for (tpl, bar_row) in zip(templates, row_choices):
        holes = _find_holes_for_template(arr, tpl["bbox"])
        if not holes:
            continue

        hole_count = len(holes)
        palette = [c for c in range(50, 250) if c not in used_colors]
        if len(palette) < hole_count:
            palette = [c for c in range(2, 250) if c not in used_colors]
        colors = rng.sample(palette, hole_count)
        for c in colors:
            used_colors.add(c)

        run = 2 * hole_count - 1
        if run > w:
            hole_count = (w + 1) // 2
            colors = colors[:hole_count]
            run = 2 * hole_count - 1

        start = rng.randint(0, w - run)
        for i, fill in enumerate(colors[:hole_count]):
            arr[bar_row, start + 2 * i] = fill

    return arr


if __name__ == "__main__":
    sample = generate_input()
    solved = main(sample)
    print(sample.shape, "->", solved.shape)
