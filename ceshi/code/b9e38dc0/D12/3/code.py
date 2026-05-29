import importlib.util
import random
from pathlib import Path

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    return [set((int(r), int(c)) for r, c in pts) for pts in components_mask(mask, connectivity=8)]


def _bbox_from_comp(comp: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _find_fill_color(g: np.ndarray, bg: int, wall: int, comps: list[set[tuple[int, int]]]) -> int:
    other_colors = [int(v) for v in np.unique(g) if int(v) not in (bg, wall)]
    assert other_colors, "expected at least one non-wall color"
    wall_points = [p for comp in comps for p in comp]

    def min_dist(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best:
                    best = d
                    if best == 0:
                        return 0, len(pts)
        return best, len(pts)

    return min(other_colors, key=min_dist)


def _detect_openings(comp_mask: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int]]:
    pts = np.argwhere(comp_mask)
    assert pts.size, "expected non-empty wall component"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)

    openings: list[dict] = []

    def add_runs(side: str, cells: list[tuple[int, int]]) -> None:
        run: list[tuple[int, int]] = []
        for r, c in cells:
            if not comp_mask[r, c]:
                run.append((r, c))
            elif run:
                openings.append(make_opening(side, run))
                run = []
        if run:
            openings.append(make_opening(side, run))

    def make_opening(side: str, run: list[tuple[int, int]]) -> dict:
        if side == "top":
            direction = (-1, 0)
        elif side == "bottom":
            direction = (1, 0)
        elif side == "left":
            direction = (0, -1)
        else:
            direction = (0, 1)
        return {"side": side, "cells": run[:], "dir": direction}

    add_runs("top", [(r0, c) for c in range(c0 + 1, c1)])
    add_runs("bottom", [(r1, c) for c in range(c0 + 1, c1)])
    add_runs("left", [(r, c0) for r in range(r0 + 1, r1)])
    add_runs("right", [(r, c1) for r in range(r0 + 1, r1)])
    return openings, (r0, c0, r1, c1)


def _flood(passable: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    seen = np.zeros_like(passable, dtype=bool)
    q = [p for p in seeds if passable[p[0], p[1]]]
    for r, c in q:
        seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _solve_component(
    g: np.ndarray,
    wall_comp: set[tuple[int, int]],
    bg: int,
    wall: int,
    fill: int,
) -> tuple[np.ndarray, int]:
    h, w = g.shape
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True
    openings, (r0, c0, r1, c1) = _detect_openings(comp_mask)
    assert openings, "expected at least one opening"

    room_mask = np.zeros_like(comp_mask, dtype=bool)
    if r1 - r0 > 1 and c1 - c0 > 1:
        room_mask[r0 + 1:r1, c0 + 1:c1] = True
    passable_room = room_mask & (~comp_mask) & ((g == bg) | (g == fill))

    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    seen = _flood(passable_room, fill_pos)

    out = np.array(g, copy=True)
    out[seen & (g == bg)] = fill

    other_walls = (g == wall) & (~comp_mask)
    obstacles = (g != bg) & (g != fill) & (g != wall)
    for opening in openings:
        dr, dc = opening["dir"]
        for r, c in opening["cells"]:
            rr, cc = r, c
            while 0 <= rr < h and 0 <= cc < w:
                if other_walls[rr, cc] or obstacles[rr, cc]:
                    break
                if out[rr, cc] == bg or out[rr, cc] == fill:
                    out[rr, cc] = fill
                rr += dr
                cc += dc

    added = int(((out == fill) & (g != fill)).sum())
    return out, added


def _solve_nested(grid: np.ndarray, record: bool = False):
    g = np.array(grid, copy=True)
    bg = get_bg(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    comps = _component_sets(g == wall)
    assert comps, "expected at least one wall component"
    fill = _find_fill_color(g, bg, wall, comps)

    remaining = comps[:]
    current = g
    stage_info = []
    while remaining:
        fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(current == fill))]
        if not fill_pos:
            break

        def comp_score(comp: set[tuple[int, int]]) -> tuple[int, int]:
            best = 10**9
            for r, c in fill_pos:
                for wr, wc in comp:
                    d = abs(r - wr) + abs(c - wc)
                    if d < best:
                        best = d
                        if best == 0:
                            break
                if best == 0:
                    break
            r0, c0, r1, c1 = _bbox_from_comp(comp)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            return best, area

        wall_comp = min(remaining, key=comp_score)
        current, added = _solve_component(current, wall_comp, bg, wall, fill)
        remaining.remove(wall_comp)
        if record:
            stage_info.append({"component_size": len(wall_comp), "added": added})

    if record:
        return current, stage_info
    return current


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve nested open-frame rooms from the innermost level outward.

    The current fill first expands inside the nearest room, then leaks through that room's gap
    until it reaches the next enclosing room. Repeating that stage over all same-colored wall
    components turns nested rooms into a sequential inner-to-outer expansion task.
    """
    return _solve_nested(np.array(grid, copy=True), record=False)


GRID_H = (15, 19)
GRID_W = (15, 19)
NEST_LEVELS = [2, 2, 2, 3]
OUTER_SIZE = (10, 14)
GAP_LEN = (1, 2)
LEVEL_INSET = 2
FRAME_MARGIN = 2
SEED_SIZE_WEIGHTS = [1, 1, 2, 2, 3, 4]
MIN_NEW_FILL = 20
GEN_RETRIES = 400


def _rect_border(h: int, w: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(w):
        pts.add((0, c))
        pts.add((h - 1, c))
    for r in range(h):
        pts.add((r, 0))
        pts.add((r, w - 1))
    return pts


def _choose_gap(side: str, h: int, w: int) -> tuple[int, int]:
    span = w if side in ("top", "bottom") else h
    gap_len = random.randint(*GAP_LEN)
    lo = 1
    hi = span - gap_len - 2
    if hi < lo:
        raise ValueError("span too small for non-corner gap")
    start = random.randint(lo, hi)
    return start, gap_len


def _build_frame_with_gap(h: int, w: int, side: str) -> tuple[set[tuple[int, int]], dict]:
    wall = _rect_border(h, w)
    start, gap_len = _choose_gap(side, h, w)
    for off in range(gap_len):
        if side == "top":
            wall.remove((0, start + off))
        elif side == "bottom":
            wall.remove((h - 1, start + off))
        elif side == "left":
            wall.remove((start + off, 0))
        else:
            wall.remove((start + off, w - 1))
    return wall, {"side": side, "start": start, "length": gap_len, "h": h, "w": w}


def _shift_points(points: set[tuple[int, int]], off: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = off
    return {(r + dr, c + dc) for r, c in points}


def _interior_points(top: int, left: int, h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(top + 1, top + h - 1) for c in range(left + 1, left + w - 1)]


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    frontier = [seed]
    target = random.choice(SEED_SIZE_WEIGHTS)
    while len(blob) < target and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target:
                    break
    return blob


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


_BASE_MAIN = None


def _get_base_main():
    global _BASE_MAIN
    if _BASE_MAIN is None:
        root = Path(__file__).resolve().parents[5]
        base_path = root / "arc2_solver_select" / "b9e38dc0.py"
        spec = importlib.util.spec_from_file_location("b9e38dc0_base", base_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _BASE_MAIN = mod.main
    return _BASE_MAIN


def analyze_nested_structure(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    bg = get_bg(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    if not non_bg:
        return {"levels": 0, "stage_adds": [], "new_fill": 0}
    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    comps = _component_sets(g == wall)
    out, stage_info = _solve_nested(g, record=True)
    fill = _find_fill_color(g, bg, wall, comps)
    return {
        "levels": len(comps),
        "stage_adds": [info["added"] for info in stage_info],
        "new_fill": int(((out == fill) & (g == bg)).sum()),
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    levels = random.choice(NEST_LEVELS)

    min_outer = 6 + 2 * LEVEL_INSET * (levels - 1)
    max_outer_h = min(OUTER_SIZE[1], h - 2 * FRAME_MARGIN)
    max_outer_w = min(OUTER_SIZE[1], w - 2 * FRAME_MARGIN)
    if max_outer_h < min_outer or max_outer_w < min_outer:
        return None

    outer_h = random.randint(max(min_outer, OUTER_SIZE[0]), max_outer_h)
    outer_w = random.randint(max(min_outer, OUTER_SIZE[0]), max_outer_w)
    top = random.randint(FRAME_MARGIN, h - outer_h - FRAME_MARGIN)
    left = random.randint(FRAME_MARGIN, w - outer_w - FRAME_MARGIN)

    side_cycle = random.sample(["top", "right", "bottom", "left"], 4)
    level_sides = side_cycle[:levels]

    frames = []
    for i in range(levels):
        fh = outer_h - 2 * LEVEL_INSET * i
        fw = outer_w - 2 * LEVEL_INSET * i
        if fh < 6 or fw < 6:
            return None
        wall, gap = _build_frame_with_gap(fh, fw, level_sides[i])
        off = (top + LEVEL_INSET * i, left + LEVEL_INSET * i)
        frames.append({"wall": _shift_points(wall, off), "gap": gap, "box": (off[0], off[1], fh, fw)})

    palette = random.sample(range(10), 3)
    bg, wall_color, fill_color = palette
    grid = np.full((h, w), bg, dtype=int)
    for frame in frames:
        for r, c in frame["wall"]:
            if grid[r, c] != bg:
                return None
            grid[r, c] = wall_color

    deepest_top, deepest_left, deepest_h, deepest_w = frames[-1]["box"]
    interior = _interior_points(deepest_top, deepest_left, deepest_h, deepest_w)
    wall_set = frames[-1]["wall"]
    seed_pool = [
        p for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for r, c in blob:
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        out, stage_info = _solve_nested(grid, record=True)
    except Exception:
        return None
    if len(stage_info) != levels or any(info["added"] <= 0 for info in stage_info):
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    try:
        base_out = _get_base_main()(grid)
        if np.array_equal(out, base_out):
            return None
    except Exception:
        pass

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid nested b9e38dc0 input")
