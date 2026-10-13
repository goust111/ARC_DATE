from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7]


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r0 for r0, _ in cells]
            cols = [c0 for _, c0 in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _arrow_from_component(comp: dict) -> dict:
    cells = comp["cells"]
    assert len(cells) == 3, "each arrow must have exactly three cells"
    r0, c0, r1, c1 = comp["bbox"]
    assert r1 == r0 + 1 and c1 == c0 + 1, "each arrow must occupy a 2x2 box"
    cell_set = set(cells)
    missing: tuple[int, int] | None = None
    for dr in range(2):
        for dc in range(2):
            rc = (r0 + dr, c0 + dc)
            if rc not in cell_set:
                assert missing is None, "expected exactly one missing corner"
                missing = rc
    assert missing is not None
    mr, mc = missing
    return {
        "color": int(comp["color"]),
        "cells": cells,
        "bbox": comp["bbox"],
        "missing": missing,
        "local_missing": (mr - r0, mc - c0),
    }


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _mutual_pairs(arrows: list[dict]) -> list[tuple[str, int, int]]:
    by_id = {id(arrow): i for i, arrow in enumerate(arrows)}
    right_target: dict[int, int] = {}
    left_target: dict[int, int] = {}
    down_target: dict[int, int] = {}
    up_target: dict[int, int] = {}

    for i, a in enumerate(arrows):
        ar, ac = a["missing"]
        amr, amc = a["local_missing"]
        if amc == 1:
            candidates = [(j, b["missing"][1]) for j, b in enumerate(arrows) if j != i and b["missing"][0] == ar and b["missing"][1] > ac]
            if candidates:
                right_target[i] = min(candidates, key=lambda x: x[1])[0]
        else:
            candidates = [(j, b["missing"][1]) for j, b in enumerate(arrows) if j != i and b["missing"][0] == ar and b["missing"][1] < ac]
            if candidates:
                left_target[i] = max(candidates, key=lambda x: x[1])[0]
        if amr == 1:
            candidates = [(j, b["missing"][0]) for j, b in enumerate(arrows) if j != i and b["missing"][1] == ac and b["missing"][0] > ar]
            if candidates:
                down_target[i] = min(candidates, key=lambda x: x[1])[0]
        else:
            candidates = [(j, b["missing"][0]) for j, b in enumerate(arrows) if j != i and b["missing"][1] == ac and b["missing"][0] < ar]
            if candidates:
                up_target[i] = max(candidates, key=lambda x: x[1])[0]

    pairs: list[tuple[str, int, int]] = []
    used: set[frozenset[int]] = set()
    for i, j in right_target.items():
        if left_target.get(j) == i and arrows[i]["color"] != arrows[j]["color"]:
            key = frozenset((i, j))
            if key not in used:
                used.add(key)
                pairs.append(("H", i, j))
    for i, j in down_target.items():
        if up_target.get(j) == i and arrows[i]["color"] != arrows[j]["color"]:
            key = frozenset((i, j))
            if key not in used:
                used.add(key)
                pairs.append(("V", i, j))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair mutually pointing L-shaped arrows, then draw a maroon strip in the region they point into together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    arrows = [_arrow_from_component(comp) for comp in _components(g, bg)]
    assert arrows, "expected at least one arrow"

    out = g.copy()
    for axis, i, j in _mutual_pairs(arrows):
        a = arrows[i]
        b = arrows[j]
        if axis == "H":
            row = a["missing"][0]
            left = min(a["missing"][1], b["missing"][1])
            right = max(a["missing"][1], b["missing"][1])
            if right - left > 1:
                out[row, left + 1 : right] = MAROON
        else:
            col = a["missing"][1]
            top = min(a["missing"][0], b["missing"][0])
            bottom = max(a["missing"][0], b["missing"][0])
            if bottom - top > 1:
                out[top + 1 : bottom, col] = MAROON
    return out


def _arrow_cells(r0: int, c0: int, local_missing: tuple[int, int]) -> list[tuple[int, int]]:
    miss_r, miss_c = local_missing
    cells = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != (miss_r, miss_c):
                cells.append((r0 + dr, c0 + dc))
    return cells


def _dilate(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _pair_cells(
    axis: str,
    miss_a: tuple[int, int],
    miss_b: tuple[int, int],
    local_a: tuple[int, int],
    local_b: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    ar, ac = miss_a
    br, bc = miss_b
    a_cells = _arrow_cells(ar - local_a[0], ac - local_a[1], local_a)
    b_cells = _arrow_cells(br - local_b[0], bc - local_b[1], local_b)
    if axis == "H":
        fill = [(ar, c) for c in range(min(ac, bc) + 1, max(ac, bc))]
    else:
        fill = [(r, ac) for r in range(min(ar, br) + 1, max(ar, br))]
    return a_cells, b_cells, fill


def _try_place_pair(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    axis: str,
    colors: tuple[int, int],
) -> bool:
    h, w = grid.shape
    for _ in range(200):
        if axis == "H":
            row = random.randint(1, h - 2)
            gap = random.randint(2, max(2, min(8, w - 5)))
            left_missing = random.randint(1, w - gap - 3)
            right_missing = left_missing + gap
            local_a = (random.randint(0, 1), 1)
            local_b = (random.randint(0, 1), 0)
            a_cells, b_cells, fill = _pair_cells("H", (row, left_missing), (row, right_missing), local_a, local_b)
        else:
            col = random.randint(1, w - 2)
            gap = random.randint(2, max(2, min(8, h - 5)))
            top_missing = random.randint(1, h - gap - 3)
            bottom_missing = top_missing + gap
            local_a = (1, random.randint(0, 1))
            local_b = (0, random.randint(0, 1))
            a_cells, b_cells, fill = _pair_cells("V", (top_missing, col), (bottom_missing, col), local_a, local_b)

        all_cells = set(a_cells) | set(b_cells) | set(fill)
        if not all(0 <= r < h and 0 <= c < w for r, c in all_cells):
            continue
        if any((r, c) in reserved for r, c in all_cells):
            continue
        if any(int(grid[r, c]) != BG for r, c in set(a_cells) | set(b_cells)):
            continue

        for r, c in a_cells:
            grid[r, c] = colors[0]
        for r, c in b_cells:
            grid[r, c] = colors[1]
        reserved.update(_dilate(all_cells, h, w, radius=1))
        return True
    return False


def _try_place_unpaired(grid: np.ndarray, reserved: set[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for _ in range(200):
        local_missing = (random.randint(0, 1), random.randint(0, 1))
        r0 = random.randint(0, h - 2)
        c0 = random.randint(0, w - 2)
        cells = set(_arrow_cells(r0, c0, local_missing))
        if any((r, c) in reserved for r, c in cells):
            continue
        if any(int(grid[r, c]) != BG for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
        reserved.update(_dilate(cells, h, w, radius=1))
        return True
    return False


def _has_branch_coverage(inp: np.ndarray, out: np.ndarray) -> bool:
    delta = out != inp
    if not np.any(delta):
        return True
    rows = np.any(delta, axis=1)
    cols = np.any(delta, axis=0)
    return bool(np.any(np.sum(delta, axis=1) > 1) or np.any(np.sum(delta, axis=0) > 1))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 22)
    w = random.randint(10, 22)
    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()

    layout = random.choice(["H", "V", "HV", "MIX"])
    if layout == "H":
        h_pairs, v_pairs = 1, 0
    elif layout == "V":
        h_pairs, v_pairs = 0, 1
    elif layout == "HV":
        h_pairs, v_pairs = 1, 1
    else:
        h_pairs = random.randint(0, 2)
        v_pairs = random.randint(0, 2)
        if h_pairs + v_pairs == 0:
            h_pairs = 1

    colors = random.sample(PALETTE, min(len(PALETTE), 2 * (h_pairs + v_pairs) + 4))
    color_idx = 0
    for _ in range(h_pairs):
        if color_idx + 2 > len(colors) or not _try_place_pair(grid, reserved, "H", (colors[color_idx], colors[color_idx + 1])):
            return None
        color_idx += 2
    for _ in range(v_pairs):
        if color_idx + 2 > len(colors) or not _try_place_pair(grid, reserved, "V", (colors[color_idx], colors[color_idx + 1])):
            return None
        color_idx += 2

    unpaired_target = random.randint(0, 4)
    remaining = [c for c in PALETTE if c not in colors[:color_idx]]
    random.shuffle(remaining)
    for i in range(unpaired_target):
        color = remaining[i % len(remaining)]
        _try_place_unpaired(grid, reserved, color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    arrows = [_arrow_from_component(comp) for comp in _components(grid, BG)]
    pairs = _mutual_pairs(arrows)
    if len(pairs) != h_pairs + v_pairs:
        return None
    if any(int(grid[r, c]) == MAROON for r, c in np.argwhere(grid == MAROON)):
        return None
    if np.any((out != grid) & (out != MAROON)):
        return None
    if not _has_branch_coverage(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
