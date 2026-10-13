from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


Coord = tuple[int, int]


def _corner_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _segment(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar != br and ac != bc:
        raise AssertionError("segment endpoints must share a row or column")
    out = [a]
    if ar == br:
        step = 1 if bc >= ac else -1
        for c in range(ac + step, bc + step, step):
            out.append((ar, c))
    else:
        step = 1 if br >= ar else -1
        for r in range(ar + step, br + step, step):
            out.append((r, ac))
    return out


def _polyline(points: list[Coord]) -> list[Coord]:
    path: list[Coord] = []
    for i in range(len(points) - 1):
        seg = _segment(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        path.extend(seg)
    if not path:
        return [points[0]]
    return path


def _terminal_cells(side: str, start: int, h: int, w: int) -> list[Coord]:
    if side == "top":
        return [(0, start), (0, start + 1), (1, start), (1, start + 1)]
    if side == "bottom":
        return [(h - 2, start), (h - 2, start + 1), (h - 1, start), (h - 1, start + 1)]
    if side == "left":
        return [(start, 0), (start, 1), (start + 1, 0), (start + 1, 1)]
    if side == "right":
        return [(start, w - 2), (start, w - 1), (start + 1, w - 2), (start + 1, w - 1)]
    raise AssertionError("unknown side")


def _port_options(side: str, start: int, h: int, w: int) -> list[Coord]:
    if side == "top":
        return [(2, start), (2, start + 1)]
    if side == "bottom":
        return [(h - 3, start), (h - 3, start + 1)]
    if side == "left":
        return [(start, 2), (start + 1, 2)]
    if side == "right":
        return [(start, w - 3), (start + 1, w - 3)]
    raise AssertionError("unknown side")


def _starts_for_side(side: str, h: int, w: int) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(2, w - 3))
    return list(range(2, h - 3))


def _make_terminal(side: str, start: int, h: int, w: int) -> dict:
    return {
        "side": side,
        "start": int(start),
        "cells": _terminal_cells(side, start, h, w),
        "ports": _port_options(side, start, h, w),
    }


def _route_candidates(a: dict, b: dict, h: int, w: int, blocked: set[Coord], rng: random.Random) -> list[dict]:
    rows = list(range(2, h - 2))
    cols = list(range(2, w - 2))
    rng.shuffle(rows)
    rng.shuffle(cols)
    row_lanes = rows[: min(3, len(rows))]
    col_lanes = cols[: min(3, len(cols))]

    candidates: list[dict] = []
    seen: set[tuple[Coord, ...]] = set()
    for pa in a["ports"]:
        for pb in b["ports"]:
            point_sets: list[tuple[str, list[Coord]]] = [
                ("l_row_then_col", [pa, (pa[0], pb[1]), pb]),
                ("l_col_then_row", [pa, (pb[0], pa[1]), pb]),
            ]
            for mid_r in row_lanes:
                point_sets.append(("row_bus", [pa, (mid_r, pa[1]), (mid_r, pb[1]), pb]))
            for mid_c in col_lanes:
                point_sets.append(("col_bus", [pa, (pa[0], mid_c), (pb[0], mid_c), pb]))

            for style, pts in point_sets:
                try:
                    coords = _polyline(pts)
                except AssertionError:
                    continue
                if any((r, c) in blocked for r, c in coords):
                    continue
                key = tuple(coords)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"style": style, "coords": coords})
    return candidates


def _has_bad_overlap(routes: list[list[Coord]]) -> tuple[bool, bool]:
    route_sets = [set(route) for route in routes]
    cell_use: Counter[Coord] = Counter()
    has_cross = False
    for route_set in route_sets:
        for cell in route_set:
            cell_use[cell] += 1
    if any(count > 2 for count in cell_use.values()):
        return True, False

    for i in range(len(route_sets)):
        for j in range(i + 1, len(route_sets)):
            inter = route_sets[i] & route_sets[j]
            if not inter:
                continue
            has_cross = True
            if len(inter) > 2:
                return True, False
            for r, c in inter:
                if (r + 1, c) in inter or (r - 1, c) in inter or (r, c + 1) in inter or (r, c - 1) in inter:
                    return True, False
    return False, has_cross


def _draw_half(h: int, w: int, paths: list[dict], routes: list[list[Coord]], draw_order: list[int]) -> tuple[np.ndarray, np.ndarray]:
    half = np.full((h, w), BLACK, dtype=int)
    terminal_mask = np.zeros((h, w), dtype=bool)
    for path in paths:
        color = int(path["color"])
        for r, c in path["a"]["cells"] + path["b"]["cells"]:
            half[r, c] = color
            terminal_mask[r, c] = True
    for idx in draw_order:
        color = int(paths[idx]["color"])
        for r, c in routes[idx]:
            half[r, c] = color
    return half, terminal_mask


def _erase_visible_slices(right: np.ndarray, paths: list[dict], routes: list[list[Coord]], terminal_mask: np.ndarray, rng: random.Random) -> int:
    erased = 0
    for idx, path in enumerate(paths):
        color = int(path["color"])
        visible = [(r, c) for r, c in routes[idx] if int(right[r, c]) == color and not bool(terminal_mask[r, c])]
        if len(visible) < 3:
            continue
        keep_left = 1 if len(visible) < 6 else 2
        keep_right = 1 if len(visible) < 6 else 2
        lo = keep_left
        hi = len(visible) - keep_right
        if lo >= hi:
            continue
        max_gap_len = min(3, hi - lo)
        gap_count = 1 if len(visible) < 9 else rng.randint(1, 2)
        blocked_starts: set[int] = set()
        for _ in range(gap_count):
            starts = [i for i in range(lo, hi) if i not in blocked_starts]
            if not starts:
                break
            start = int(rng.choice(starts))
            gap_len = int(rng.randint(1, min(max_gap_len, hi - start) + 1))
            for j in range(start, start + gap_len):
                r, c = visible[j]
                if int(right[r, c]) != BLACK:
                    right[r, c] = BLACK
                    erased += 1
            for j in range(max(lo, start - 1), min(hi, start + gap_len + 1)):
                blocked_starts.add(j)
    return erased


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the broken circuit drawing on the right by copying the complete left-hand layout across the separator.

    Input structure:
    - The background is black.
    - One full-height grey column splits the grid into two equal-width halves.
    - The left half contains complete colored 2x2 terminals and 1-cell-wide orthogonal wire paths.
    - The right half contains the same arrangement, but some wire pixels have been erased to black.

    Rule:
    - Locate the vertical grey separator and align the left and right halves by relative position.
    - Treat the left half as the complete template for the right half.
    - For every black pixel on the right whose aligned pixel on the left is a colored terminal or wire pixel, copy that left-hand color into the right-hand pixel.
    - Leave the left half, the separator column, and already-correct right-hand pixels unchanged.

    Key insight:
    - The right side is not a different puzzle instance; it is the same circuit as the left side with only local deletions, so aligned pixel-by-pixel transfer restores every missing segment, including crossing colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w % 2 == 1, "expected a single middle separator column"

    bg = _corner_bg(g)
    sep = w // 2
    sep_color = int(g[0, sep])
    assert sep_color != bg and bool(np.all(g[:, sep] == sep_color)), "middle column must be the grey separator"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "halves must be equal sized"

    out = g.copy()
    right_out = out[:, sep + 1 :]
    fill_mask = (right == bg) & (left != bg)
    right_out[fill_mask] = left[fill_mask]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = int(rng.randint(9, 16))
    w = int(rng.randint(9, 14))
    force_cross = bool(rng.random() < 0.5)
    path_count = int(rng.randint(3, 4) if force_cross else rng.randint(2, 3))
    colors = list(rng.sample(COLOR_POOL, path_count))

    side_pairs: list[tuple[str, str]] = []
    if force_cross and path_count >= 2:
        side_pairs.extend([("top", "bottom"), ("left", "right")])
    while len(side_pairs) < path_count:
        a, b = rng.sample(SIDES, 2)
        side_pairs.append((a, b))

    terminal_occupied: set[Coord] = set()
    paths: list[dict] = []
    for color, (side_a, side_b) in zip(colors, side_pairs, strict=False):
        placed = None
        starts_a = _starts_for_side(side_a, h, w)
        starts_b = _starts_for_side(side_b, h, w)
        rng.shuffle(starts_a)
        rng.shuffle(starts_b)
        for start_a in starts_a:
            term_a = _make_terminal(side_a, start_a, h, w)
            if any(cell in terminal_occupied for cell in term_a["cells"]):
                continue
            for start_b in starts_b:
                term_b = _make_terminal(side_b, start_b, h, w)
                if any(cell in terminal_occupied for cell in term_b["cells"]):
                    continue
                if set(term_a["cells"]) & set(term_b["cells"]):
                    continue
                placed = {"color": int(color), "a": term_a, "b": term_b}
                break
            if placed is not None:
                break
        if placed is None:
            return None
        terminal_occupied.update(placed["a"]["cells"])
        terminal_occupied.update(placed["b"]["cells"])
        paths.append(placed)

    blocked = set(terminal_occupied)
    candidate_routes: list[list[dict]] = []
    for path in paths:
        cands = _route_candidates(path["a"], path["b"], h, w, blocked, rng)
        if not cands:
            return None
        candidate_routes.append(cands)

    routes: list[list[Coord]] | None = None
    for _ in range(300):
        chosen: list[list[Coord]] = []
        for idx, cands in enumerate(candidate_routes):
            if force_cross and idx == 0:
                pref = [cand for cand in cands if cand["style"] == "row_bus"]
                chosen.append(list(rng.choice(pref or cands)["coords"]))
            elif force_cross and idx == 1:
                pref = [cand for cand in cands if cand["style"] == "col_bus"]
                chosen.append(list(rng.choice(pref or cands)["coords"]))
            else:
                chosen.append(list(rng.choice(cands)["coords"]))
        bad, has_cross = _has_bad_overlap(chosen)
        if bad or has_cross != force_cross:
            continue
        routes = chosen
        break
    if routes is None:
        return None

    draw_order = list(range(path_count))
    rng.shuffle(draw_order)
    half, terminal_mask = _draw_half(h, w, paths, routes, draw_order)

    visible_counts = []
    for idx, path in enumerate(paths):
        color = int(path["color"])
        count = sum(1 for r, c in routes[idx] if int(half[r, c]) == color and not bool(terminal_mask[r, c]))
        visible_counts.append(count)
    if min(visible_counts) < 2:
        return None

    full = np.full((h, 2 * w + 1), BLACK, dtype=int)
    full[:, :w] = half
    full[:, w] = GREY
    full[:, w + 1 :] = half

    right = half.copy()
    erased = _erase_visible_slices(right, paths, routes, terminal_mask, rng)
    if erased < max(2, path_count):
        return None

    inp = full.copy()
    inp[:, w + 1 :] = right
    if np.array_equal(inp, full):
        return None
    if not np.array_equal(main(inp), full):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid circuit-repair input")
