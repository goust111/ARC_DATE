import random

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
BRANCH_TYPES = ("both", "h_only", "v_only")
MAX_ATTEMPTS = 600
PAIR_SEARCH_WIDTH = 240


def _perimeter_index(pos: tuple[int, int], h: int, w: int) -> int:
    r, c = pos
    if r == 0:
        return c - 1
    if c == w - 1:
        return (w - 2) + (r - 1)
    if r == h - 1:
        return (w - 2) + (h - 2) + (w - 2 - c)
    if c == 0:
        return (w - 2) + (h - 2) + (w - 2) + (h - 2 - r)
    raise AssertionError("position is not on the non-corner border")


def _inward_point(pos: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = pos
    if r == 0:
        return (1, c)
    if r == h - 1:
        return (h - 2, c)
    if c == 0:
        return (r, 1)
    if c == w - 1:
        return (r, w - 2)
    raise AssertionError("position is not on the border")


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return {(r1, c) for c in range(lo, hi + 1)}
    if c1 == c2:
        lo, hi = sorted((r1, r2))
        return {(r, c1) for r in range(lo, hi + 1)}
    raise AssertionError("segment must be axis-aligned")


def _path_info(
    p1: tuple[int, int], p2: tuple[int, int], h: int, w: int
) -> dict[str, object] | None:
    i1 = _perimeter_index(p1, h, w)
    i2 = _perimeter_index(p2, h, w)
    start, end = (p1, p2) if i1 <= i2 else (p2, p1)
    start_idx = min(i1, i2)
    s = _inward_point(start, h, w)
    t = _inward_point(end, h, w)
    if s[0] == t[0] or s[1] == t[1]:
        return None
    if abs(s[0] - t[0]) < 2 or abs(s[1] - t[1]) < 2:
        return None

    bend_h = (s[0], t[1])
    bend_v = (t[0], s[1])
    cand_h = _line_cells(s, bend_h) | _line_cells(bend_h, t)
    cand_v = _line_cells(s, bend_v) | _line_cells(bend_v, t)
    return {
        "anchors": (start, end),
        "start_idx": start_idx,
        "cand_h": frozenset(cand_h),
        "cand_v": frozenset(cand_v),
    }


def _legend_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}


def _paint_legend(grid: np.ndarray, top_left: tuple[int, int], outer: int, inner: int) -> None:
    r0, c0 = top_left
    grid[r0 : r0 + 3, c0 : c0 + 3] = outer
    grid[r0 + 1, c0 + 1] = inner


def _detect_legends(grid: np.ndarray) -> dict[int, dict[str, object]]:
    h, w = grid.shape
    legends: dict[int, dict[str, object]] = {}
    used = np.zeros_like(grid, dtype=bool)
    for r0 in range(1, h - 3):
        for c0 in range(1, w - 3):
            if used[r0 : r0 + 3, c0 : c0 + 3].any():
                continue
            sub = grid[r0 : r0 + 3, c0 : c0 + 3]
            outer = int(sub[0, 0])
            inner = int(sub[1, 1])
            if outer == BG or inner == BG or inner == outer:
                continue
            ring = [int(sub[r, c]) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
            if any(v != outer for v in ring):
                continue
            used[r0 : r0 + 3, c0 : c0 + 3] = True
            assert outer not in legends, "legend outer colors must be unique"
            legends[outer] = {
                "inner": inner,
                "cells": _legend_cells((r0, c0)),
            }
    return legends


def _detect_anchor_pairs(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    border: dict[int, list[tuple[int, int]]] = {}

    for c in range(1, w - 1):
        v = int(grid[0, c])
        if v != BG:
            border.setdefault(v, []).append((0, c))
    for r in range(1, h - 1):
        v = int(grid[r, w - 1])
        if v != BG:
            border.setdefault(v, []).append((r, w - 1))
    for c in range(w - 2, 0, -1):
        v = int(grid[h - 1, c])
        if v != BG:
            border.setdefault(v, []).append((h - 1, c))
    for r in range(h - 2, 0, -1):
        v = int(grid[r, 0])
        if v != BG:
            border.setdefault(v, []).append((r, 0))

    pairs: list[dict[str, object]] = []
    for color, pts in border.items():
        assert len(pts) == 2, "each anchor color must appear exactly twice on the border"
        info = _path_info(pts[0], pts[1], h, w)
        assert info is not None, "anchor pair must define two L-shaped candidates"
        info["color"] = color
        pairs.append(info)

    pairs.sort(key=lambda item: int(item["start_idx"]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode 3x3 color legends and connect matching border anchor pairs with recolored L paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    legends = _detect_legends(g)
    pairs = _detect_anchor_pairs(g)
    assert legends and len(legends) == len(pairs)

    static_blocked = {
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if int(g[r, c]) != BG
    }

    out = g.copy()
    blocked = set(static_blocked)
    for pair in pairs:
        outer = int(pair["color"])
        assert outer in legends, "every anchor color must have a matching legend"
        draw_color = int(legends[outer]["inner"])
        cand_h = set(pair["cand_h"])
        cand_v = set(pair["cand_v"])
        clear_h = cand_h.isdisjoint(blocked)
        clear_v = cand_v.isdisjoint(blocked)
        assert clear_h or clear_v, "at least one L-path must stay clear"
        chosen = cand_h if clear_h else cand_v
        for r, c in chosen:
            out[r, c] = draw_color
        blocked.update(chosen)
    return out


def _random_derangement(values: list[int]) -> list[int]:
    while True:
        perm = values[:]
        random.shuffle(perm)
        if all(a != b for a, b in zip(values, perm, strict=False)):
            return perm


def _grid_bounds_for_pairs(pair_count: int) -> tuple[int, int]:
    min_size = {3: 20, 4: 22, 5: 24}[pair_count]
    return min_size, 25


def _border_positions(h: int, w: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(0, c) for c in range(1, w - 1)],
        "right": [(r, w - 1) for r in range(1, h - 1)],
        "bottom": [(h - 1, c) for c in range(1, w - 1)],
        "left": [(r, 0) for r in range(1, h - 1)],
    }


def _pair_pool(h: int, w: int) -> list[dict[str, object]]:
    positions = _border_positions(h, w)
    pool: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for i, side_a in enumerate(SIDES):
        for side_b in SIDES[i + 1 :]:
            for p1 in positions[side_a]:
                for p2 in positions[side_b]:
                    key = tuple(sorted((p1, p2)))
                    if key in seen:
                        continue
                    seen.add(key)
                    info = _path_info(p1, p2, h, w)
                    if info is None:
                        continue
                    pool.append(info)
    random.shuffle(pool)
    return pool


def _pick_pairs(
    h: int, w: int, branches: list[str]
) -> list[dict[str, object]] | None:
    pool = _pair_pool(h, w)
    chosen: list[dict[str, object]] = []
    used_anchors: set[tuple[int, int]] = set()
    search_branches = branches[:]
    random.shuffle(search_branches)
    search_branches.sort(key=lambda branch: 1 if branch == "both" else 0)

    def compatible(candidate: dict[str, object], selected: frozenset[tuple[int, int]]) -> bool:
        anchors = set(candidate["anchors"])
        if anchors & used_anchors:
            return False
        cand_h = candidate["cand_h"]
        cand_v = candidate["cand_v"]
        for old in chosen:
            old_selected = old["selected"]
            if not set(cand_h).isdisjoint(old_selected):
                return False
            if not set(cand_v).isdisjoint(old_selected):
                return False
            if not set(selected).isdisjoint(old["cand_h"]):
                return False
            if not set(selected).isdisjoint(old["cand_v"]):
                return False
        return True

    def backtrack(depth: int) -> bool:
        if depth == len(search_branches):
            return True
        branch = search_branches[depth]
        order = random.sample(range(len(pool)), k=min(len(pool), PAIR_SEARCH_WIDTH))
        for idx in order:
            base = pool[idx]
            selected = base["cand_v"] if branch == "v_only" else base["cand_h"]
            if not compatible(base, selected):
                continue
            item = {
                "anchors": base["anchors"],
                "start_idx": base["start_idx"],
                "cand_h": base["cand_h"],
                "cand_v": base["cand_v"],
                "branch": branch,
                "selected": selected,
            }
            chosen.append(item)
            used_anchors.update(item["anchors"])
            if any(not _legend_options(i, chosen, h, w) for i in range(len(chosen))):
                chosen.pop()
                used_anchors.difference_update(item["anchors"])
                continue
            if backtrack(depth + 1):
                return True
            chosen.pop()
            used_anchors.difference_update(item["anchors"])
        return False

    if not backtrack(0):
        return None
    chosen.sort(key=lambda item: int(item["start_idx"]))
    return chosen


def _legend_options(
    pair_index: int, pairs: list[dict[str, object]], h: int, w: int
) -> list[tuple[int, int, set[tuple[int, int]]]]:
    pair = pairs[pair_index]
    forbidden: set[tuple[int, int]] = set()
    required: set[tuple[int, int]] = set()

    for idx, other in enumerate(pairs):
        cand_h = set(other["cand_h"])
        cand_v = set(other["cand_v"])
        if idx != pair_index:
            forbidden.update(cand_h)
            forbidden.update(cand_v)
            continue
        branch = str(pair["branch"])
        if branch == "both":
            forbidden.update(cand_h)
            forbidden.update(cand_v)
        elif branch == "h_only":
            forbidden.update(cand_h)
            required = cand_v
        else:
            forbidden.update(cand_v)
            required = cand_h

    options: list[tuple[int, int, set[tuple[int, int]]]] = []
    for r0 in range(1, h - 3):
        for c0 in range(1, w - 3):
            cells = _legend_cells((r0, c0))
            if cells & forbidden:
                continue
            if required and not (cells & required):
                continue
            options.append((r0, c0, cells))
    random.shuffle(options)
    return options


def _place_legends(
    pairs: list[dict[str, object]], h: int, w: int
) -> list[tuple[int, int]] | None:
    options = [_legend_options(i, pairs, h, w) for i in range(len(pairs))]
    if any(not opts for opts in options):
        return None

    order = sorted(range(len(pairs)), key=lambda idx: len(options[idx]))
    placements: list[tuple[int, int] | None] = [None] * len(pairs)
    used_cells: set[tuple[int, int]] = set()

    def backtrack(depth: int) -> bool:
        if depth == len(order):
            return True
        pair_idx = order[depth]
        candidates = options[pair_idx][:]
        random.shuffle(candidates)
        for r0, c0, cells in candidates:
            if cells & used_cells:
                continue
            placements[pair_idx] = (r0, c0)
            used_cells.update(cells)
            if backtrack(depth + 1):
                return True
            placements[pair_idx] = None
            used_cells.difference_update(cells)
        return False

    if not backtrack(0):
        return None
    return [pos for pos in placements if pos is not None]


def _build_input_and_output() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choices([3, 4, 5], weights=[4, 4, 2], k=1)[0]
    lo, hi = _grid_bounds_for_pairs(pair_count)
    h = random.randint(lo, hi)
    w = random.randint(lo, hi)

    outer_colors = random.sample(list(range(1, 10)), pair_count)
    inner_colors = _random_derangement(outer_colors)

    branches = [BRANCH_TYPES[i % len(BRANCH_TYPES)] for i in range(pair_count)]
    random.shuffle(branches)

    pairs = _pick_pairs(h, w, branches)
    if pairs is None:
        return None

    legend_positions = _place_legends(pairs, h, w)
    if legend_positions is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()

    for idx, pair in enumerate(pairs):
        outer = outer_colors[idx]
        inner = inner_colors[idx]
        pair["color"] = outer
        pair["draw_color"] = inner
        a1, a2 = pair["anchors"]
        grid[a1] = outer
        grid[a2] = outer
        expected[a1] = outer
        expected[a2] = outer

    for idx, pair in enumerate(pairs):
        pos = legend_positions[idx]
        pair["legend_top_left"] = pos
        _paint_legend(grid, pos, int(pair["color"]), int(pair["draw_color"]))
        _paint_legend(expected, pos, int(pair["color"]), int(pair["draw_color"]))

    for pair in pairs:
        for r, c in pair["selected"]:
            expected[r, c] = int(pair["draw_color"])

    legends = _detect_legends(grid)
    parsed_pairs = _detect_anchor_pairs(grid)
    if set(legends) != set(outer_colors) or len(parsed_pairs) != pair_count:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_input_and_output()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input grid")
