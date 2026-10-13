from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GRID_RANGE = (23, 34)
SLOT_MIN = 8
OUTER_MARGIN = 1
SLOT_GAP = 3
GEN_RETRIES = 400
SLOT_RETRIES = 80
SHAPE_RETRIES = 80


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(h: int, w: int, r: int, c: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(h, w, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _repair_loops(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    colors = [int(v) for v in np.unique(g) if int(v) != BG]

    changed = True
    while changed:
        changed = False
        for color in colors:
            targets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
            for r, c in np.argwhere(g == int(color)):
                r = int(r)
                c = int(c)
                same = [(nr, nc) for nr, nc in _neighbors4(h, w, r, c) if int(g[nr, nc]) == int(color)]
                if len(same) != 1:
                    continue
                nr, nc = same[0]
                tr, tc = 2 * r - nr, 2 * c - nc
                if _in_bounds(h, w, tr, tc) and int(g[tr, tc]) == BG:
                    targets[(tr, tc)].append((r, c))
            for (tr, tc), endpoints in targets.items():
                if len(endpoints) == 2:
                    g[tr, tc] = int(color)
                    changed = True

    return g


def _collect_enclosed_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict[str, object]] = []

    for cells in _components(g == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        border_colors = {
            int(g[nr, nc])
            for r, c in cells
            for nr, nc in _neighbors4(h, w, r, c)
            if int(g[nr, nc]) != BG
        }
        if len(border_colors) != 1:
            continue
        anchor = min(cells)
        regions.append(
            {
                "cells": cells,
                "area": len(cells),
                "border_color": next(iter(border_colors)),
                "anchor": anchor,
            }
        )

    regions.sort(key=lambda item: (-int(item["area"]), tuple(item["anchor"])))
    return regions


def _fill_ranked_regions(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    regions = _collect_enclosed_regions(out)
    prev_color: int | None = None
    for region in regions:
        border_color = int(region["border_color"])
        fill_color = border_color if prev_color is None else prev_color
        for r, c in region["cells"]:
            out[r, c] = fill_color
        prev_color = border_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair one-pixel gaps between same-color orthogonal segments, close each loop,
    then fill loop interiors by descending area with a one-step color shift.
    """
    repaired = _repair_loops(np.asarray(grid, dtype=int))
    return _fill_ranked_regions(repaired)


def _trace_boundary(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:] + vertices[:1], strict=False):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0, c1 + step, step):
                boundary.add((r0, c))
        else:
            step = 1 if r1 > r0 else -1
            for r in range(r0, r1 + step, step):
                boundary.add((r, c0))
    return boundary


def _local_area(height: int, width: int, boundary: set[tuple[int, int]]) -> int:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in boundary:
        mask[r, c] = True

    areas = [
        len(comp)
        for comp in _components(~mask)
        if not any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in comp)
    ]
    assert len(areas) == 1
    return int(areas[0])


def _rectangle_shape(max_h: int, max_w: int) -> dict[str, object] | None:
    if max_h < 4 or max_w < 4:
        return None
    h = random.randint(4, min(max_h, 11))
    w = random.randint(4, min(max_w, 11))
    vertices = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)]
    boundary = _trace_boundary(vertices)
    return {
        "height": h,
        "width": w,
        "boundary": boundary,
        "turns": set(vertices),
        "area": (h - 2) * (w - 2),
    }


def _notched_shape(max_h: int, max_w: int) -> dict[str, object] | None:
    if max_h < 8 or max_w < 8:
        return None

    valid_sides: list[str] = []
    if max_h >= 8 and max_w >= 10:
        valid_sides.extend(["top", "bottom"])
    if max_h >= 10 and max_w >= 8:
        valid_sides.extend(["left", "right"])
    if not valid_sides:
        return None

    side = random.choice(valid_sides)
    if side in {"top", "bottom"}:
        h = random.randint(8, min(max_h, 13))
        w = random.randint(10, min(max_w, 14))
        if w < 10 or h < 8:
            return None
        left = random.randint(3, w - 7)
        span = random.randint(3, w - left - 4)
        depth = random.randint(3, h - 4)
        if side == "top":
            vertices = [
                (0, 0),
                (0, left),
                (depth, left),
                (depth, left + span),
                (0, left + span),
                (0, w - 1),
                (h - 1, w - 1),
                (h - 1, 0),
            ]
        else:
            vertices = [
                (0, 0),
                (0, w - 1),
                (h - 1, w - 1),
                (h - 1, left + span),
                (h - 1 - depth, left + span),
                (h - 1 - depth, left),
                (h - 1, left),
                (h - 1, 0),
            ]
    else:
        h = random.randint(10, min(max_h, 14))
        w = random.randint(8, min(max_w, 13))
        if h < 10 or w < 8:
            return None
        top = random.randint(3, h - 7)
        span = random.randint(3, h - top - 4)
        depth = random.randint(3, w - 4)
        if side == "left":
            vertices = [
                (0, 0),
                (0, w - 1),
                (h - 1, w - 1),
                (h - 1, 0),
                (top + span, 0),
                (top + span, depth),
                (top, depth),
                (top, 0),
            ]
        else:
            vertices = [
                (0, 0),
                (0, w - 1),
                (top, w - 1),
                (top, w - 1 - depth),
                (top + span, w - 1 - depth),
                (top + span, w - 1),
                (h - 1, w - 1),
                (h - 1, 0),
            ]

    boundary = _trace_boundary(vertices)
    return {
        "height": h,
        "width": w,
        "boundary": boundary,
        "turns": set(vertices),
        "area": _local_area(h, w, boundary),
    }


def _sample_shape(max_h: int, max_w: int) -> dict[str, object] | None:
    builders = []
    if max_h >= 4 and max_w >= 4:
        builders.append(_rectangle_shape)
    if max_h >= 8 and max_w >= 8:
        builders.append(_notched_shape)
    if not builders:
        return None

    for _ in range(SHAPE_RETRIES):
        builder = random.choice(builders)
        shape = builder(max_h, max_w)
        if shape is not None:
            return shape
    return None


def _slot_ranges(h: int, w: int) -> list[tuple[int, int, int, int]]:
    top_h = random.randint(SLOT_MIN, h - SLOT_MIN - OUTER_MARGIN * 2 - SLOT_GAP)
    left_w = random.randint(SLOT_MIN, w - SLOT_MIN - OUTER_MARGIN * 2 - SLOT_GAP)
    bottom_r0 = OUTER_MARGIN + top_h + SLOT_GAP
    right_c0 = OUTER_MARGIN + left_w + SLOT_GAP
    return [
        (OUTER_MARGIN, OUTER_MARGIN + top_h - 1, OUTER_MARGIN, OUTER_MARGIN + left_w - 1),
        (OUTER_MARGIN, OUTER_MARGIN + top_h - 1, right_c0, w - OUTER_MARGIN - 1),
        (bottom_r0, h - OUTER_MARGIN - 1, OUTER_MARGIN, OUTER_MARGIN + left_w - 1),
        (bottom_r0, h - OUTER_MARGIN - 1, right_c0, w - OUTER_MARGIN - 1),
    ]


def _paint_shape(grid: np.ndarray, shape: dict[str, object], r0: int, c0: int, color: int) -> None:
    for r, c in shape["boundary"]:
        grid[r0 + r, c0 + c] = int(color)


def _break_turns(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    broken = np.array(grid, dtype=int, copy=True)
    for placed in placements:
        r0 = int(placed["r0"])
        c0 = int(placed["c0"])
        for tr, tc in placed["turns"]:
            broken[r0 + tr, c0 + tc] = BG
    return broken


def _is_single_cycle(grid: np.ndarray, color: int) -> bool:
    mask = grid == int(color)
    comps = _components(mask)
    if len(comps) != 1:
        return False
    h, w = grid.shape
    return all(
        sum(1 for nr, nc in _neighbors4(h, w, r, c) if int(grid[nr, nc]) == int(color)) == 2
        for r, c in comps[0]
    )


def _generate_case() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    slots = _slot_ranges(h, w)
    n_loops = random.choice([3, 4])
    chosen_slots = random.sample(range(4), n_loops)
    colors = random.sample(range(1, 10), n_loops)

    closed = np.zeros((h, w), dtype=int)
    placements: list[dict[str, object]] = []
    used_areas: set[int] = set()

    for color, slot_idx in zip(colors, chosen_slots, strict=False):
        sr0, sr1, sc0, sc1 = slots[slot_idx]
        slot_h = sr1 - sr0 + 1
        slot_w = sc1 - sc0 + 1
        placed = None
        for _ in range(SLOT_RETRIES):
            shape = _sample_shape(slot_h, slot_w)
            if shape is None:
                continue
            area = int(shape["area"])
            if area in used_areas:
                continue
            max_r0 = sr1 - int(shape["height"]) + 1
            max_c0 = sc1 - int(shape["width"]) + 1
            if max_r0 < sr0 or max_c0 < sc0:
                continue
            r0 = random.randint(sr0, max_r0)
            c0 = random.randint(sc0, max_c0)
            _paint_shape(closed, shape, r0, c0, int(color))
            placed = {
                "r0": r0,
                "c0": c0,
                "turns": shape["turns"],
                "area": area,
                "color": int(color),
            }
            used_areas.add(area)
            placements.append(placed)
            break
        if placed is None:
            return None

    broken = _break_turns(closed, placements)
    repaired = _repair_loops(broken)

    if any(not _is_single_cycle(repaired, int(color)) for color in colors):
        return None

    regions = _collect_enclosed_regions(repaired)
    if len(regions) != n_loops:
        return None
    if len({int(region["area"]) for region in regions}) != n_loops:
        return None

    expected = _fill_ranked_regions(closed)
    got = main(broken)
    if not np.array_equal(got, expected):
        return None

    return broken


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("generate_input failed after repeated retries")
