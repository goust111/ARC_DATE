from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

import os
import sys
from pathlib import Path


_arc_utils_path = None
for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    for _cand_dir in (_p / "new" / "solvers", _p / "solvers"):
        _cand = _cand_dir / "arc_utils.py"
        if _cand.exists():
            _arc_utils_path = str(_cand_dir)
            break
    if _arc_utils_path is not None:
        break
if _arc_utils_path and _arc_utils_path not in sys.path:
    sys.path.insert(0, _arc_utils_path)

from arc_utils import find_components, get_bg



_BAR_COLOR = 5


def _detect_bar_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows = []
    for r, row in enumerate(grid):
        first = int(row[0])
        if first == bg:
            continue
        if np.all(row == first):
            rows.append(r)
    return rows


def _regions_from_bars(H: int, bar_rows: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(bar_rows) + [H]
    regions = []
    for top, bottom in zip(bounds[:-1], bounds[1:]):
        start = top + 1
        end = bottom - 1
        if start <= end:
            regions.append((start, end))
    return regions


def _random_polyomino(area: int) -> np.ndarray:
    area = int(area)
    assert 2 <= area <= 20

    cells = [(0, 0)]
    frontier: list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    frontier_set = set(frontier)

    while len(cells) < area and frontier:
        r, c = random.choice(frontier)
        frontier.remove((r, c))
        frontier_set.discard((r, c))

        if (r, c) in cells:
            continue
        cells.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            cand = (nr, nc)
            if cand in cells or cand in frontier_set:
                continue
            if len(cells) < area:
                frontier.append(cand)
                frontier_set.add(cand)

    if len(cells) < area:
        while len(cells) < area:
            cells.append((cells[-1][0], cells[-1][1]))

    cells = np.array(cells, dtype=int)
    r0, c0 = cells.min(axis=0)
    cells[:, 0] -= r0
    cells[:, 1] -= c0
    return cells


def _can_place(
    grid: np.ndarray,
    bg: int,
    patch_coords: np.ndarray,
    top: int,
    left: int,
) -> bool:
    H, W = grid.shape
    for r, c in patch_coords:
        rr = top + int(r)
        cc = left + int(c)
        if rr < 0 or rr >= H or cc < 0 or cc >= W:
            return False
        if grid[rr, cc] != bg:
            return False
    return True


def _place_patch(
    grid: np.ndarray, color: int, patch: np.ndarray, top: int, left: int
) -> None:
    for r, c in patch:
        grid[top + int(r), left + int(c)] = color


def _settle_object(coords: np.ndarray, bg: int, bar_rows: list[int], H: int) -> dict:
    del bg
    minr, minc = int(coords[:, 0].min()), int(coords[:, 1].min())
    maxr = int(coords[:, 0].max())
    below = [r for r in bar_rows if r > maxr]
    support = int(min(below)) if below else H
    drop = support - maxr - 1
    moved = coords + np.array([drop, 0], dtype=int)
    return {
        "coords": moved,
        "support": support,
        "bbox": (
            int(moved[:, 0].min()),
            int(moved[:, 1].min()),
            int(moved[:, 0].max()),
            int(moved[:, 1].max()),
        ),
    }


def _solve_region(objects: list[dict], H: int, W: int) -> list[tuple[int, np.ndarray]]:
    sorted_objects = sorted(objects, key=lambda item: (item["area"], item["color"], item.get("seed", 0)))
    cursor = 0
    out = []
    for obj in sorted_objects:
        coords = obj["settled"]
        color = obj["color"]
        minr, minc, maxr, maxc = obj["bbox"]
        rel = coords.copy()
        rel[:, 0] -= int(minr)
        rel[:, 1] -= int(minc)
        height = int(maxr - minr + 1)
        target_top = int(obj["support"] - height)
        width = int(maxc - minc + 1)

        if cursor + width > W:
            return out

        for rr, cc in rel:
            out.append((target_top + int(rr), cursor + int(cc), color))
        cursor += width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    H, W = g.shape

    bg = int(get_bg(g))
    bar_rows = _detect_bar_rows(g, bg)
    if len(bar_rows) < 2:
        return g.copy()

    objects = []
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=None,
        return_format="dict",
    )

    for comp in comps:
        color = int(comp["color"])
        if color in (bg, _BAR_COLOR):
            continue
        coords = np.array(comp["coords"], dtype=int)
        minr = int(coords[:, 0].min())
        maxr = int(coords[:, 0].max())
        minc = int(coords[:, 1].min())
        maxc = int(coords[:, 1].max())
        below = [r for r in bar_rows if r > maxr]
        support = int(min(below)) if below else H
        drop = support - maxr - 1
        settled = coords + np.array([drop, 0], dtype=int)
        objects.append(
            {
                "color": color,
                "area": int(comp["area"]),
                "support": support,
                "settled": settled,
                "bbox": (
                    int(settled[:, 0].min()),
                    int(settled[:, 1].min()),
                    int(settled[:, 0].max()),
                    int(settled[:, 1].max()),
                ),
            }
        )

    if not objects:
        return g.copy()

    by_support: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        by_support[obj["support"]].append(obj)

    out = g.copy()
    movable = (out != bg) & (out != _BAR_COLOR)
    out[movable] = bg

    for support in sorted(by_support.keys()):
        group = sorted(by_support[support], key=lambda item: item["area"])
        for row, col, color in _solve_region(group, H, W):
            if 0 <= row < H and 0 <= col < W:
                out[row, col] = color

    return out


def _choose_bar_rows(H: int) -> list[int]:
    candidates = list(range(1, H - 1))
    for _ in range(200):
        k = random.randint(2, 3)
        rows = sorted(random.sample(candidates, k))
        ok = True
        for a, b in zip(rows, rows[1:]):
            if b - a < 3:
                ok = False
                break
        if ok:
            return rows
    return sorted(random.sample(candidates, 2))


def _generate_one() -> np.ndarray | None:
    H = random.randint(13, 17)
    W = random.randint(13, 17)
    bg = 0

    grid = np.full((H, W), bg, dtype=int)
    bar_rows = _choose_bar_rows(H)
    for r in bar_rows:
        grid[r, :] = _BAR_COLOR

    regions = _regions_from_bars(H, bar_rows)
    if not regions:
        return None

    obj_count = random.randint(4, 6)
    colors = random.sample([1, 2, 3, 4, 6, 7], obj_count)
    areas = random.sample(range(3, 9), obj_count)

    objects = []
    for color, area in zip(colors, areas):
        patch = _random_polyomino(area)
        ph, pw = int(patch[:, 0].max()) + 1, int(patch[:, 1].max()) + 1

        placed = False
        for _ in range(400):
            r0, r1 = random.choice(regions)
            if r0 + ph - 1 > r1:
                continue
            top = random.randint(r0, r1 - ph + 1)
            left = random.randint(0, W - pw)

            if not _can_place(grid, bg, patch, top, left):
                continue
            _place_patch(grid, color, patch, top, left)
            objects.append((color, patch))
            placed = True
            break

        if not placed:
            return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle-1 input")
