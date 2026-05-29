# Variant of 28a6681f
# Dimension: D2 — 对角滑动规则 (path)
# Change: 当正下方被阻挡时，水像素检查左下和右下两个对角方向——要求侧面和对角位置均为背景，且必须恰好只有一个方向可行（无歧义）... → 禁止对角滑动

from __future__ import annotations

import itertools
import random

import numpy as np


# Design analysis
# ----------------
# variant:
#   core_visual_motif:
#     - The grid stays 10x10 with blue water (1) and 3 solid colors.
#     - The solids form a right-high staircase profile: a low basin on one side, a tall
#       plateau on the source side, and a one-cell edge chute beside that plateau.
#     - The input water is a small supported 2x2 source block resting on top of the tall side.
#     - Under the original task, that source would prefer the adjacent edge chute and drain
#       into the narrow side column. In this variant, adjacent corner-style exits are dead:
#       water must use a farther drop that is reached by sliding across at least one supported
#       shelf cell, so it ends in the low basin instead.
#   behavior_trace:
#     original:
#       - A water pixel resting on the plateau sees a drop one column away at the outer edge.
#       - The old solver treats that one-step corner as a legal continuation and spills outward.
#     variant:
#       - The same pixel can still fall vertically, but horizontal sliding only targets a drop
#         that lies beyond at least one supported shelf cell.
#       - The adjacent edge chute is rejected, so the nearest legal route is a farther interior
#         drop that leads into the low basin.
#     consequence:
#       - Output water clusters on the low side instead of the outer chute. This is visible even
#         with only 4 water pixels because the final water block moves several columns away.
#   complexity_calibration:
#     - The generator uses 3 prevalidated staircase profiles. They share one rule but differ in
#       low-basin depth and shelf height, so examples are recognizably related without collapsing
#       to a single frozen bitmap.
#     - Carryover is limited to at most 2 already-stable basin pixels; this keeps the task
#       readable and avoids excessive source-shape branching.
#   implicit_assumption_audit:
#     - In the original code, any legal diagonal step is observationally subsumed by the
#       horizontal branch because the adjacent diagonal landing cell is also a nearest drop
#       column at distance 1.
#     - If diagonal logic were removed without touching horizontal targeting, D2 would be
#       invisible on generated samples.
#     - Therefore the variant redefines horizontal reachability narrowly: the drop column must
#       be at least two columns away, and every intermediate shelf cell must be empty and
#       supported. This is the minimum adaptation needed to make "no corner-cutting" visible.
#   placement_legality:
#     - Every sampled profile is bottom-anchored and leaves source cells inside the grid.
#     - Input water is always supported from below and never has background on both sides.
#     - Carryover cells are copied only from a solved output, so they are guaranteed stable.
#   observability_and_cleanliness:
#     - The grids use a single stepped landform with a compact source block, not many unrelated
#       objects, so the route choice is the only moving part.
#     - The decisive difference is spatial, not color-driven: no sorting or mapping uses color id.


PROFILES: tuple[tuple[int, ...], ...] = (
    (1, 1, 3, 5, 5, 7, 7, 7, 7, 0),
    (2, 2, 4, 4, 4, 7, 7, 7, 7, 0),
    (2, 2, 4, 6, 6, 7, 7, 7, 7, 0),
)

SOURCE_POSITIONS: tuple[tuple[int, int], ...] = (
    (2, 7),
    (2, 8),
    (1, 7),
    (1, 8),
)


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling in stepped basins when corner-cutting is forbidden.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Other colors form fixed bottom-anchored staircases, shelves, and basin walls.
    - Water starts as a supported source block on the tall side, optionally with a tiny amount
      already resting in the destination basin.

    Rule:
    - Each water pixel falls straight down through bg cells.
    - When supported, it may slide only across a supported horizontal shelf to a farther column
      where a vertical fall is possible.
    - A one-step corner opening is not a legal target: the drop column must be at least two
      columns away, so diagonal-style stair bypasses are removed.
    - Process repeats until no water pixel moves.

    Key insight: Without diagonal or corner-equivalent escapes, water can use only true vertical
    shafts plus longer flat shelves, so narrow step chutes no longer connect basins.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
        remaining = set(points)
        if use_diagonal:
            neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        else:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                r, c = stack.pop()
                for dr, dc in neighbors:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt in remaining:
                        remaining.remove(nxt)
                        stack.append(nxt)
        return count

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert component_count(pts, False) == component_count(pts, True), "guide must be 4-connected"

    def shelf_clear(r: int, a: int, b: int) -> bool:
        step = 1 if b > a else -1
        for x in range(int(a + step), int(b), int(step)):
            if int(g[r, x]) != int(bg) or int(g[r + 1, x]) == int(bg):
                return False
        return True

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                drops = []
                for x in range(W):
                    if abs(int(x - cc)) < 2:
                        continue
                    if int(g[rr, x]) != int(bg) or int(g[rr + 1, x]) != int(bg):
                        continue
                    if shelf_clear(int(rr), int(cc), int(x)):
                        drops.append(int(x))

                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                cc = nearest[0]
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(water)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(bg)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(bg)

    return g


def _build_structure(profile: tuple[int, ...], colors: tuple[int, int, int]) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    for c, height in enumerate(profile):
        color = colors[0] if c <= 2 else colors[1] if c <= 5 else colors[2]
        for r in range(10 - int(height), 10):
            g[r, c] = int(color)
    return g


def _find_valid_input(structure: np.ndarray) -> np.ndarray:
    target_input = np.array(structure, dtype=int).copy()
    for r, c in SOURCE_POSITIONS:
        target_input[r, c] = 1
    target_output = main(target_input)

    left_basin = sorted(
        [(int(r), int(c)) for r, c in np.argwhere(target_output == 1) if int(c) <= 1],
        reverse=True,
    )
    plans: list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
    for carry_size in range(min(2, len(left_basin)), -1, -1):
        for carry in itertools.combinations(left_basin, carry_size):
            for keep in itertools.combinations(SOURCE_POSITIONS, len(SOURCE_POSITIONS) - carry_size):
                plans.append((carry, keep))
    random.shuffle(plans)

    for carry, keep in plans:
        candidate = np.array(structure, dtype=int).copy()
        for r, c in carry:
            candidate[r, c] = 1
        for r, c in keep:
            candidate[r, c] = 1
        try:
            if np.array_equal(main(candidate), target_output):
                return candidate
        except AssertionError:
            continue

    raise AssertionError("failed to realize a validated input")


def generate_input() -> np.ndarray:
    for _ in range(100):
        profile = random.choice(PROFILES)
        colors = tuple(random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3))
        structure = _build_structure(profile, colors)
        try:
            return _find_valid_input(structure)
        except AssertionError:
            continue
    raise AssertionError("generator could not build a valid sample")
