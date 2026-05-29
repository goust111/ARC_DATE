# Variant of 28a6681f
# Dimension: D20 — 输出中结构的保留/变换 (transform)
# Change: 所有盆壁结构在输出中完全保留不变（invariant: structure pixels unchanged）... → 未使用的结构被移除

from __future__ import annotations

import random
import numpy as np


# Design analysis for the D20 variant
# a) Concretization and trace:
#    - The original task keeps every basin wall pixel after the water settles.
#    - In this variant, the water still follows the exact same physics first; only after that do we
#      inspect the bottom-anchored structure. A basin pixel survives iff it is 4-neighbor adjacent
#      to at least one final water pixel. Dry bottom-anchored structure is erased to bg.
#    - Small trace on sample-1 style geometry:
#      original: water spills from the right source into the narrow central basin; the untouched left
#      basin wall still stays visible.
#      variant: the same spill simulation happens first, but then the untouched left basin has no
#      final contact with water, so its pixels are recolored to 0, while the wetted central wall/floor
#      remains.
# b) Complexity calibration:
#    - Deletion is pixel-level, not whole-component-level, so the visual effect is stronger than just
#      removing an entire unused cup.
#    - The generator keeps 4 original-style structure families, plus mirroring and recoloring, so the
#      solver still has to reason about full water motion and cannot memorize one fixed silhouette.
# c) Hidden assumptions review against the original main():
#    - Input water support checks still hold unchanged: deletion happens only after settling.
#    - The settle() logic assumes static obstacles while simulating a single output. That remains true,
#      because structure deletion is postponed until all water stops.
#    - The old return path used the settled grid directly. That is no longer safe: returning early
#      would preserve dry structure and miss the D20 change. Therefore main() now has an extra
#      deletion pass after simulation.
#    - Output stability checks must run on the post-deletion grid, not on the pre-deletion one.
# d) Placement legality:
#    - The generator uses hand-vetted 10x10 structure families derived from the original task idiom.
#    - Only horizontal mirroring and color remapping are applied, so no overlap or out-of-bounds risk
#      is introduced.
#    - Optional water trimming is monotone from the top of a supported source block, so support and
#      one-sidedness constraints remain valid.
# e) Visual observability:
#    - The effect is visible in a few examples because empty basins or dry upper wall segments vanish.
#    - The deleted/retained structure depends on the final water destination, so the human solver must
#      still mentally run the water before inferring which walls stay.
# f) Visual cleanliness:
#    - Only bottom-anchored structure is eligible for deletion. Upper guide pixels stay unchanged,
#      which keeps the scene readable instead of erasing multiple unrelated layers at once.
# g) Color inference:
#    - No hidden ordering by color id is used. Recoloring is a pure permutation among visible solid
#      colors; deletion depends only on geometric contact with final water.
#    - Reference-task note: 8b9c3697 also performs a "run the main interaction, then delete the
#      unmatched leftovers" pattern. This variant keeps that high-level conditional-deletion shape,
#      but the condition here is final water contact with basin structure.


BG = 0
WATER = 1
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = ORTHO_DIRS
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


def _check_input_constraints(g: np.ndarray) -> None:
    H, W = g.shape
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    top_colors = set(map(int, g[:7].flatten())) - {int(BG), int(WATER)} - bottom_colors
    for color in sorted(top_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _check_output_stability(g: np.ndarray) -> None:
    H, W = g.shape
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
        assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)


def _simulate_water(grid: np.ndarray, check_input: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        _check_input_constraints(g)

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    return g


def _bottom_anchored_structure(g: np.ndarray) -> set[tuple[int, int]]:
    H, W = g.shape
    solid = (g != int(BG)) & (g != int(WATER))
    stack = [(H - 1, int(c)) for c in range(W) if bool(solid[H - 1, c])]
    anchored: set[tuple[int, int]] = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in anchored:
            continue
        anchored.add((int(r), int(c)))
        for dr, dc in ORTHO_DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and bool(solid[rr, cc]) and (rr, cc) not in anchored:
                stack.append((rr, cc))
    return anchored


def _delete_unused_basin_pixels(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    anchored = _bottom_anchored_structure(g)
    used: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        for dr, dc in ORTHO_DIRS:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) in anchored:
                used.add((nr, nc))

    out = np.array(g, dtype=int).copy()
    for r, c in anchored:
        if (int(r), int(c)) not in used:
            out[int(r), int(c)] = int(BG)
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Settle blue water into bottom basins, then erase the dry parts of those basins.

    Input structure:
    - bg = 0 and water = 1.
    - Non-water solids form bottom-anchored basins and spillways; some inputs also include an upper guide.
    - Water begins as a supported side reservoir or guide-fed source, not yet in its final resting basin.

    Rule:
    - First run the original local water simulation: fall vertically, slip through a unique open diagonal, and slide horizontally toward a unique reachable drop while supported.
    - After the water stops, inspect only bottom-anchored structure pixels.
    - Keep a structure pixel only if it is 4-neighbor adjacent to final water; erase every other bottom-anchored structure pixel to background.

    Key insight: Solving still requires the full water-flow simulation, because the surviving walls are determined only after the final wet contact pattern is known.
    """
    settled = _simulate_water(np.array(grid, dtype=int), check_input=check_input)
    out = _delete_unused_basin_pixels(settled)
    if check_output:
        _check_output_stability(out)
    return out


_TEMPLATE_1 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
        [0, 0, 0, 0, 0, 3, 0, 3, 1, 1],
        [0, 0, 0, 0, 0, 3, 0, 3, 1, 1],
        [0, 0, 0, 0, 2, 3, 0, 3, 1, 1],
        [0, 0, 0, 0, 2, 3, 0, 3, 1, 1],
        [0, 2, 2, 2, 2, 3, 3, 3, 1, 1],
    ],
    dtype=int,
)

_TEMPLATE_2 = np.array(
    [
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 3, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 3, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 3, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 3, 0, 0, 3, 1, 1, 1, 1],
        [0, 0, 3, 0, 0, 3, 5, 1, 1, 5],
        [0, 0, 3, 1, 1, 3, 5, 1, 1, 5],
        [0, 0, 3, 1, 1, 3, 5, 1, 1, 5],
        [0, 0, 3, 3, 3, 3, 5, 5, 5, 5],
    ],
    dtype=int,
)

_TEMPLATE_3 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9, 9, 1, 1, 0, 0, 0, 0, 9, 9],
        [0, 9, 9, 1, 0, 0, 0, 9, 9, 0],
        [0, 0, 9, 9, 0, 0, 0, 9, 0, 0],
        [0, 0, 0, 9, 0, 0, 9, 9, 0, 0],
        [0, 0, 0, 0, 0, 9, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [2, 0, 0, 4, 0, 4, 0, 0, 0, 0],
        [2, 0, 2, 4, 0, 4, 0, 0, 0, 0],
        [2, 2, 2, 4, 4, 4, 0, 0, 0, 0],
    ],
    dtype=int,
)

_TEMPLATE_4 = np.array(
    [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [9, 9, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 9, 9, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 9, 9, 9, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 9, 9, 9, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 5, 0, 5],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [3, 3, 0, 0, 0, 0, 5, 5, 0, 5],
        [3, 3, 3, 0, 0, 5, 5, 5, 0, 5],
        [3, 3, 3, 3, 5, 5, 5, 5, 0, 5],
    ],
    dtype=int,
)

_TEMPLATES = [
    {"grid": _TEMPLATE_1, "trim": "right_block"},
    {"grid": _TEMPLATE_2, "trim": "none"},
    {"grid": _TEMPLATE_3, "trim": "none"},
    {"grid": _TEMPLATE_4, "trim": "none"},
]


def _remap_structure_colors(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    structure_colors = sorted(set(map(int, np.unique(g))) - {int(BG), int(WATER)})
    bottom_colors = sorted(set(map(int, g[-1])) - {int(BG), int(WATER)})
    guide_colors = [int(c) for c in structure_colors if int(c) not in set(bottom_colors)]

    palette = [2, 3, 4, 5, 6, 7, 8, 9]
    mapping: dict[int, int] = {}

    basin_targets = random.sample(palette, len(bottom_colors))
    for old, new in zip(bottom_colors, basin_targets):
        mapping[int(old)] = int(new)

    remaining = [int(c) for c in palette if int(c) not in set(mapping.values())]
    guide_targets = random.sample(remaining, len(guide_colors))
    for old, new in zip(guide_colors, guide_targets):
        mapping[int(old)] = int(new)

    out = g.copy()
    for old, new in mapping.items():
        out[g == int(old)] = int(new)
    return out


def _trim_template_water(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    if mode == "right_block":
        clear_rows = random.randint(0, 3)
        if clear_rows:
            g[:clear_rows, 8:10] = np.where(g[:clear_rows, 8:10] == int(WATER), int(BG), g[:clear_rows, 8:10])
    return g


def _variant_metrics(inp: np.ndarray) -> tuple[int, int]:
    settled = _simulate_water(inp, check_input=True)
    out = _delete_unused_basin_pixels(settled)
    anchored = _bottom_anchored_structure(settled)
    deleted = sum(1 for r, c in anchored if int(settled[r, c]) != int(BG) and int(out[r, c]) == int(BG))

    bottom_colors = sorted(set(int(settled[r, c]) for r, c in anchored))
    vanished_colors = 0
    for color in bottom_colors:
        if all(int(out[r, c]) != int(color) for r, c in anchored if int(settled[r, c]) == int(color)):
            vanished_colors += 1
    return int(deleted), int(vanished_colors)


def generate_input() -> np.ndarray:
    for _ in range(200):
        template = random.choice(_TEMPLATES)
        g = _trim_template_water(template["grid"], template["trim"])
        g = _remap_structure_colors(g)
        if random.random() < 0.5:
            g = np.fliplr(g)

        solved = main(g)
        deleted, vanished_colors = _variant_metrics(g)
        if deleted >= 2 and (vanished_colors >= 1 or deleted >= 4):
            assert np.array_equal(solved, main(g))
            return np.array(g, dtype=int)

    raise AssertionError("failed to generate a non-degenerate D20 sample")
