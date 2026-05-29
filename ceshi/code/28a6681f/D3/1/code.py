# Variant of 28a6681f
# Dimension: D3 — 水平滑动规则 (path)
# Change: 支撑在非背景像素上时，水像素在当前行的连续背景段内找到最近的可落点（下方为背景的列），然后逐步水平移动过去... → 最远可落点滑动

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np

# Design analysis for the D3 variant
# ----------------------------------
# concretization:
# - The generator keeps the original 10x10 water-and-basin setting, but it now
#   forces a long supported shelf on row 4. Water starts as a supported edge
#   reservoir pressed against that shelf.
# - Under the shelf there are three vertical escape options inside one continuous
#   background run: a near shaft, a mid shaft, and a far edge shaft. The new
#   rule must choose the farthest drop first, so the output visibly fills the
#   remote edge shaft before the nearer openings.
# - Trace on the canonical left-source template:
#   input water starts at columns 0-1 on row 4 / above it.
#   original nearest rule: drop candidates under row 5 are columns 2, 6, and 9;
#   from column 0 the nearest is 2, so water falls into the left shaft first.
#   new farthest rule: the farthest is 9, so the pixel must deliberately slide
#   past columns 2 and 6 and only then fall at column 9. Once that shaft is full,
#   later water switches to the remaining farther candidate.
#   The visible difference is therefore not subtle: the filled basin order flips.
#
# complexity calibration:
# - A single shelf with 3 drop candidates is the minimum structure that makes
#   "nearest" versus "farthest" obvious in 2-3 examples.
# - Mirror symmetry is sampled, so the rule is not overfit to "always move right".
# - Water amount varies through a 1-2 column supported reservoir, producing both
#   pure remote-fill cases and overflow-to-nearer-shaft cases.
#
# implicit assumption audit:
# - In the original code, "pick nearest drop and step one cell" is safe because a
#   nearer chosen target cannot hide another even-nearer drop along the way.
# - That assumption breaks for "pick farthest": a naive replacement would let the
#   pixel step toward the far target, land on an intermediate drop column, and
#   immediately fall there on the next loop.
# - The solver therefore adds a horizontal slide target state. Once a farthest
#   drop column is chosen on a supported row, the pixel keeps stepping toward that
#   chosen column before gravity is reconsidered. All other movement rules remain
#   unchanged.
#
# placement legality:
# - The template uses only explicit occupied cells, so nothing overlaps.
# - Every input water pixel is supported by either a solid cell or another water
#   pixel below it.
# - The source is pressed against a grid edge or another water column, so no input
#   water pixel has background on both horizontal sides.
# - All solids are bottom-connected colors already present on the last row, so the
#   original guide-validation logic remains legal even though this generator does
#   not place a separate upper guide object.
#
# visual observability and cleanliness:
# - Outputs differ by basin-filling order, not by invisible metadata. The far edge
#   shaft filling first is easy to read directly from the blue pixel layout.
# - Only three solid colors are used, and their role is purely structural rather
#   than symbolic; no color sorting or hidden color-based branching is used.


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
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


def _simulate(
    grid: np.ndarray,
    *,
    prefer_farthest: bool,
    check_input: bool,
    check_output: bool,
    collect_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1
    stats = {
        "horizontal_decisions": 0,
        "multi_drop_decisions": 0,
        "different_extreme_decisions": 0,
    }

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        slide_target: int | None = None

        for _ in range(H * W * 2):
            if slide_target is not None:
                if cc == slide_target:
                    slide_target = None
                else:
                    step = -1 if slide_target < cc else 1
                    nx = int(cc + step)
                    assert int(g[rr, nx]) == int(bg)
                    cc = nx
                    continue

            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                slide_target = None
                continue

            opts = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(bg)
                and int(g[rr + 1, cc - 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(bg)
                and int(g[rr + 1, cc + 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                slide_target = None
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                L = int(cc)
                while L - 1 >= 0 and int(g[rr, L - 1]) == int(bg):
                    L -= 1
                R = int(cc)
                while R + 1 < W and int(g[rr, R + 1]) == int(bg):
                    R += 1

                drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
                if not drops:
                    break

                stats["horizontal_decisions"] += 1
                if len(drops) >= 2:
                    stats["multi_drop_decisions"] += 1

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"

                max_dist = max(abs(int(x - cc)) for x in drops)
                farthest = [int(x) for x in drops if abs(int(x - cc)) == int(max_dist)]
                assert len(farthest) == 1, f"ambiguous farthest drop columns: {farthest}"

                nearest_best = nearest[0]
                farthest_best = farthest[0]
                if nearest_best != farthest_best:
                    stats["different_extreme_decisions"] += 1

                best = farthest_best if prefer_farthest else nearest_best
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(bg)
                cc = nx
                slide_target = int(best) if prefer_farthest else None
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

    if collect_stats:
        return g, stats
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate supported blue water flowing through bottom basin structures.

    Input structure:
    - Background is 0 and water is color 1.
    - The remaining colors form fixed shelves, walls, and narrow reservoirs anchored to the lower half.
    - Water begins as a supported edge-packed source column or ragged two-column reservoir.

    Rule:
    - Each water pixel falls straight down through background.
    - If it is blocked below, it may slide diagonally only through the unique open side path.
    - When it is resting on support, it scans the full contiguous background run in its row, chooses the farthest drop column in that run, and then keeps sliding step by step until it reaches that chosen drop.
    - Repeat until no water pixel can move any further.

    Key insight: The structure is unchanged, but replacing nearest-drop sliding with farthest-drop sliding makes water bypass closer shafts and fill remote basins first.
    """
    return _simulate(
        grid,
        prefer_farthest=True,
        check_input=check_input,
        check_output=check_output,
    )


def _build_template(
    *,
    mirror: bool,
    side_height: int,
    colors: tuple[int, int, int],
) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    left_wall, center_mass, source_mass = colors

    # Lower structures: a near vertical shaft at column 2, a mid basin near 6-7,
    # and an edge shaft at the far side. The shelf row sits above them at row 4.
    g[5:, 0] = int(source_mass)
    g[5:, 1] = int(left_wall)
    g[5:, 3] = int(left_wall)
    g[5:, 4] = int(center_mass)
    g[5:, 5] = int(center_mass)
    g[5:, 8] = int(left_wall)
    g[5, 7] = int(left_wall)
    g[9, 2] = int(left_wall)
    g[9, 6:8] = int(center_mass)

    # Supported source reservoir pressed against one edge.
    g[0:5, 0] = 1
    g[5 - side_height : 5, 1] = 1

    if mirror:
        g = np.fliplr(g)
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    palette = [2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(200):
        colors = tuple(rng.sample(palette, 3))
        side_height = rng.randint(1, 4)
        mirror = bool(rng.randint(0, 1))
        inp = _build_template(mirror=mirror, side_height=side_height, colors=colors)

        out_new, stats = _simulate(
            inp,
            prefer_farthest=True,
            check_input=True,
            check_output=True,
            collect_stats=True,
        )
        out_old = _simulate(
            inp,
            prefer_farthest=False,
            check_input=True,
            check_output=False,
        )

        if stats["different_extreme_decisions"] < 1:
            continue
        if np.array_equal(out_new, out_old):
            continue
        return inp

    raise RuntimeError("failed to generate a non-degenerate D3 variant sample")
