from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

GEN_MAX_ATTEMPTS = 400
GROUP_MAX_ATTEMPTS = 250
BAND_H = 7
BAND_GAP = 1
MIN_H = 10
MAX_H = 28
MIN_W = 18
MAX_W = 30
GROUP_KINDS = ("solo", "pair_partial", "pair_full", "triple")


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    cells = list(cells)
    if not cells:
        return None
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _block_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, RED)]
    for color in colors:
        for coords in _components_of_color(grid, color):
            box = _bbox(coords)
            assert box is not None
            r0, c0, r1, c1 = box
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            assert side_h == side_w and side_h in (2, 3), "non-arrow objects must be 2x2 or 3x3 solid squares"
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(patch == color), "square blocks must be solid"
            objects.append({"color": color, "cells": set(coords)})
    return objects


def _parse_arrow(coords: list[tuple[int, int]]) -> dict:
    assert len(coords) == 4, "arrow must have exactly four red cells"
    cells = set(coords)

    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in coords:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    for row, cols in by_row.items():
        cols = sorted(cols)
        if len(cols) != 3 or cols[2] != cols[0] + 2:
            continue
        shaft = {(row, cols[0]), (row, cols[1]), (row, cols[2])}
        extra = next(iter(cells - shaft))
        left_end = (row, cols[0])
        right_end = (row, cols[2])
        if abs(extra[0] - left_end[0]) + abs(extra[1] - left_end[1]) == 1:
            return {
                "dir": "L",
                "axis": "h",
                "shaft": row,
                "front": min(c for _, c in coords),
                "sort_key": (min(r for r, _ in coords), min(c for _, c in coords)),
            }
        if abs(extra[0] - right_end[0]) + abs(extra[1] - right_end[1]) == 1:
            return {
                "dir": "R",
                "axis": "h",
                "shaft": row,
                "front": max(c for _, c in coords),
                "sort_key": (min(r for r, _ in coords), min(c for _, c in coords)),
            }

    for col, rows in by_col.items():
        rows = sorted(rows)
        if len(rows) != 3 or rows[2] != rows[0] + 2:
            continue
        shaft = {(rows[0], col), (rows[1], col), (rows[2], col)}
        extra = next(iter(cells - shaft))
        top_end = (rows[0], col)
        bottom_end = (rows[2], col)
        if abs(extra[0] - top_end[0]) + abs(extra[1] - top_end[1]) == 1:
            return {
                "dir": "U",
                "axis": "v",
                "shaft": col,
                "front": min(r for r, _ in coords),
                "sort_key": (min(r for r, _ in coords), min(c for _, c in coords)),
            }
        if abs(extra[0] - bottom_end[0]) + abs(extra[1] - bottom_end[1]) == 1:
            return {
                "dir": "D",
                "axis": "v",
                "shaft": col,
                "front": max(r for r, _ in coords),
                "sort_key": (min(r for r, _ in coords), min(c for _, c in coords)),
            }

    raise AssertionError("red component is not a valid arrow")


def _distance_to_edge(cells: set[tuple[int, int]], direction: str, h: int, w: int) -> int:
    assert cells
    if direction == "R":
        return (w - 1) - max(c for _, c in cells)
    if direction == "L":
        return min(c for _, c in cells)
    if direction == "D":
        return (h - 1) - max(r for r, _ in cells)
    if direction == "U":
        return min(r for r, _ in cells)
    raise AssertionError("unknown direction")


def _find_target_index(objects: list[dict], arrow: dict) -> int:
    best_idx = -1
    best_dist: int | None = None
    direction = arrow["dir"]

    for idx, obj in enumerate(objects):
        if not obj["cells"]:
            continue
        box = _bbox(obj["cells"])
        assert box is not None
        r0, c0, r1, c1 = box

        if arrow["axis"] == "h":
            row = int(arrow["shaft"])
            if not (r0 <= row <= r1):
                continue
            if direction == "R":
                if c0 <= arrow["front"]:
                    continue
                dist = c0 - int(arrow["front"])
            else:
                if c1 >= arrow["front"]:
                    continue
                dist = int(arrow["front"]) - c1
        else:
            col = int(arrow["shaft"])
            if not (c0 <= col <= c1):
                continue
            if direction == "D":
                if r0 <= arrow["front"]:
                    continue
                dist = r0 - int(arrow["front"])
            else:
                if r1 >= arrow["front"]:
                    continue
                dist = int(arrow["front"]) - r1

        if best_dist is None or dist < best_dist:
            best_idx = idx
            best_dist = dist

    assert best_idx >= 0, "each arrow must have a target block in front of it"
    return best_idx


def _push_objects(objects: list[dict], target_idx: int, direction: str, h: int, w: int) -> None:
    dr, dc = DIRS[direction]
    steps = _distance_to_edge(objects[target_idx]["cells"], direction, h, w)
    active = {target_idx}

    for _ in range(steps):
        stationary: dict[tuple[int, int], int] = {}
        for idx, obj in enumerate(objects):
            if idx in active:
                continue
            for cell in obj["cells"]:
                stationary[cell] = idx

        changed = True
        while changed:
            changed = False
            for idx in tuple(active):
                for r, c in objects[idx]["cells"]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    hit = stationary.get((nr, nc))
                    if hit is not None and hit not in active:
                        active.add(hit)
                        changed = True

        for idx in active:
            shifted = {
                (r + dr, c + dc)
                for r, c in objects[idx]["cells"]
                if 0 <= r + dr < h and 0 <= c + dc < w
            }
            objects[idx]["cells"] = shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each aligned square block toward the edge indicated by its red arrow, carrying any contacted blocks along the same direction.

    Input structure:
    - The background is black.
    - Non-red foreground objects are solid 2x2 or 3x3 monochrome square blocks.
    - Red foreground components are four-cell arrows: a 3-cell straight shaft plus one offset head cell, pointing up, down, left, or right.
    - Each arrow has exactly one square block in front of it on the shaft row or shaft column.

    Rule:
    - Detect every red arrow, infer its direction, and erase all arrow pixels.
    - For each arrow, find the unique square block that lies ahead of the arrow on the arrow's shaft row or shaft column.
    - Move that target block one cell at a time toward the indicated edge until the target block's own front edge reaches the grid boundary.
    - If the moving block runs into another block, that block starts moving in the same direction as part of the same push chain; cells pushed beyond the grid are cropped away.
    - All vacated cells become black.

    Key insight:
    - The arrow's geometry identifies both its direction and the single aligned target block, but the true transformation is a stepwise chain push: later blocks do not jump directly to the edge, they begin moving only when the pushed block reaches them, so front blocks can be partially or fully clipped away.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    arrows = [_parse_arrow(coords) for coords in _components_of_color(g, RED)]
    objects = _block_objects(g)

    for arrow in sorted(arrows, key=lambda item: item["sort_key"]):
        target_idx = _find_target_index(objects, arrow)
        _push_objects(objects, target_idx, arrow["dir"], h, w)

    out = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            out[r, c] = int(obj["color"])
    return out


def _clone_objects(objects: list[dict]) -> list[dict]:
    return [{"color": int(obj["color"]), "cells": set(obj["cells"])} for obj in objects]


def _group_rows(band_top: int, sizes: list[int], sign: int, head_down: bool) -> tuple[list[int], int] | None:
    band_bottom = band_top + BAND_H - 1
    size0 = sizes[0]

    if sign == 1:
        extent_bottom = sum(sizes) - len(sizes)
        min_t0 = band_top + (0 if head_down else 1)
        max_t0 = band_bottom - extent_bottom
        if head_down:
            max_t0 = min(max_t0, band_bottom - 1)
        if min_t0 > max_t0:
            return None
        t0 = random.randint(min_t0, max_t0)
        tops = [t0]
        for prev_size in sizes[:-1]:
            tops.append(tops[-1] + prev_size - 1)
        shaft_row = t0
        return tops, shaft_row

    up_extent = sum(sizes[1:]) - (len(sizes) - 1)
    min_t0 = band_top + up_extent
    max_t0 = band_bottom - size0 + 1
    if head_down:
        max_t0 = min(max_t0, band_bottom - size0)
    if min_t0 > max_t0:
        return None
    t0 = random.randint(min_t0, max_t0)
    tops = [t0]
    for size in sizes[1:]:
        tops.append(tops[-1] - (size - 1))
    shaft_row = t0 + size0 - 1
    return tops, shaft_row


def _group_succeeds(kind: str, before: list[dict], after: list[dict]) -> bool:
    if kind == "solo":
        return before[0]["cells"] != after[0]["cells"]
    if kind == "pair_partial":
        return 0 < len(after[1]["cells"]) < len(before[1]["cells"])
    if kind == "pair_full":
        return len(after[1]["cells"]) == 0
    if kind == "triple":
        moved1 = before[1]["cells"] != after[1]["cells"]
        moved2 = before[2]["cells"] != after[2]["cells"]
        clipped = len(after[1]["cells"]) < len(before[1]["cells"]) or len(after[2]["cells"]) < len(before[2]["cells"])
        return moved1 and moved2 and clipped
    raise AssertionError("unknown group kind")


def _make_base_group(w: int, band_top: int) -> dict | None:
    for _ in range(GROUP_MAX_ATTEMPTS):
        kind = random.choice(GROUP_KINDS)
        n_blocks = {"solo": 1, "pair_partial": 2, "pair_full": 2, "triple": 3}[kind]
        sizes = [random.choice((2, 3)) for _ in range(n_blocks)]
        sign = random.choice((-1, 1))
        head_down = random.choice((False, True))
        row_info = _group_rows(band_top, sizes, sign, head_down)
        if row_info is None:
            continue
        tops, shaft_row = row_info

        gaps = [random.randint(1, 3) for _ in range(n_blocks - 1)]
        total_width = sum(sizes) + sum(gaps)
        gap_arrow = random.randint(1, 3)
        min_left = gap_arrow + 3
        max_left = w - total_width
        if min_left > max_left:
            continue

        candidate_lefts = list(range(min_left, max_left + 1))
        random.shuffle(candidate_lefts)

        for target_left in candidate_lefts:
            lefts = [target_left]
            for idx in range(1, n_blocks):
                lefts.append(lefts[-1] + sizes[idx - 1] + gaps[idx - 1])

            local_h = band_top + BAND_H
            before = []
            for idx, size in enumerate(sizes):
                cells = {
                    (r, c)
                    for r in range(tops[idx], tops[idx] + size)
                    for c in range(lefts[idx], lefts[idx] + size)
                }
                before.append({"color": COLOR_POOL[idx], "cells": cells})

            after = _clone_objects(before)
            _push_objects(after, 0, "R", local_h, w)
            if not _group_succeeds(kind, before, after):
                continue

            colors = random.sample(COLOR_POOL, n_blocks)
            objects = []
            for idx in range(n_blocks):
                objects.append({"color": colors[idx], "cells": set(before[idx]["cells"])})

            arrow_c0 = target_left - gap_arrow - 3
            arrow_cells = {
                (shaft_row, arrow_c0),
                (shaft_row, arrow_c0 + 1),
                (shaft_row, arrow_c0 + 2),
                (shaft_row + (1 if head_down else -1), arrow_c0 + 2),
            }
            return {"objects": objects, "arrow": arrow_cells, "kind": kind}
    return None


def _orientation_transform(grid: np.ndarray) -> np.ndarray:
    mode = random.choice(("R", "L", "D", "U"))
    if mode == "R":
        return grid
    if mode == "L":
        return np.fliplr(grid)
    if mode == "D":
        return grid.T
    return np.flipud(grid.T)


def _gen_base_input() -> np.ndarray | None:
    h = random.randint(MIN_H, MAX_H)
    w = random.randint(MIN_W, MAX_W)
    lane_tops = list(range(0, h - BAND_H + 1, BAND_H + BAND_GAP))
    if not lane_tops:
        return None

    n_groups = random.randint(1, len(lane_tops))
    selected_tops = random.sample(lane_tops, n_groups)
    kinds = [random.choice(GROUP_KINDS) for _ in range(n_groups)]
    if all(kind == "solo" for kind in kinds):
        kinds[random.randrange(n_groups)] = random.choice(("pair_partial", "pair_full", "triple"))

    grid = np.full((h, w), BG, dtype=int)
    for band_top, forced_kind in zip(sorted(selected_tops), kinds):
        group = None
        for _ in range(GROUP_MAX_ATTEMPTS):
            candidate = _make_base_group(w, band_top)
            if candidate is not None and candidate["kind"] == forced_kind:
                group = candidate
                break
        if group is None:
            return None

        for obj in group["objects"]:
            for r, c in obj["cells"]:
                assert grid[r, c] == BG
                grid[r, c] = int(obj["color"])
        for r, c in group["arrow"]:
            assert grid[r, c] == BG
            grid[r, c] = RED

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        base = _gen_base_input()
        if base is None:
            continue
        grid = _orientation_transform(base)
        out = main(grid)
        if grid.shape != out.shape:
            continue
        if np.count_nonzero(grid == RED) < 4:
            continue
        return grid.astype(int)
    raise RuntimeError("failed to generate valid arrow-push ARC input")
