# Variant of 28a6681f
# Dimension: D3 — 水平滑动与寻找落点机制 (path)
# Change: 水像素在支撑面上沿水平方向滑向同一连续背景段中最近的落点（下方为背景的列），且落点必须唯一... → 滑向最远落点而非最近

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) Concrete behavior and trace:
#    The variant is easiest to expose with one supported source shelf and two drop shafts below it.
#    The source water sits on a 2-cell-wide supported ledge and is visibly closer to one shaft.
#    In the original task, supported water drifts toward the nearer unique shaft and drops there.
#    In this variant, the same shelf contact should commit to the farther unique shaft instead.
#    A direct trace on the canonical layout is:
#    - Source columns are 3-4, left shaft is at column 1, right shaft is at column 8.
#    - Original logic: water at column 3 sees distances 2 and 5, so it walks left and drops into
#      the left shaft first.
#    - Variant logic: the same water should bypass the closer left shaft and slide to column 8,
#      then fall through the right shaft first.
#    - The two outputs therefore differ in which vertical basin gets the topmost water cells.
# b) Complexity calibration:
#    Two shafts are enough because the source is visibly closer to one and farther from the other,
#    so the path inversion is easy to notice. To avoid visual underfitting, the generator varies
#    mirror direction, basin capacities, and solid colors while keeping the path decision repeated
#    across six water pixels in every sample.
# c) Hidden-assumption audit:
#    The original solver recomputes the nearest drop after every 1-cell horizontal step. That is
#    harmless for the nearest-drop rule because the target remains monotone once chosen. After
#    inversion, the same 1-step recomputation causes left/right oscillation on shelves with drops
#    on both sides. To keep D3 logically viable, horizontal motion must commit to the unique
#    farthest drop for the current shelf contact and slide to that column before resuming gravity.
#    This is the minimal compatibility fix required by the new rule.
# d) Placement legality:
#    The generator uses a fully controlled 10x10 template. All solid pixels are bottom-anchored,
#    the only open roof cells are the two shaft mouths, and the source water is supported by a
#    solid roof row. Mirroring is done after construction, so the geometry stays in-bounds.
# e) Visual observability:
#    Every accepted sample is checked against an internal "original" simulator. We only keep cases
#    where the far-drop output and near-drop output differ by at least four cells and the favored
#    shaft changes by at least two water cells, so the inversion is visible in 2-3 examples.
# f) Visual cleanliness:
#    Water, shafts, and solids occupy disjoint simple regions: top source rectangle, two vertical
#    shafts, and one bottom-anchored mass of 3 solid colors. There are no overlapping markers or
#    ambiguous decorations.
# g) Color inference:
#    Colors are static obstacle colors only. No color-dependent ordering is used; all ordering is
#    geometric and visible from position alone.
# Solver/generator assumption review:
# - `main()` assumes any non-bottom solid color used in the upper rows also appears on the bottom
#   row, so the original guide checks remain valid.
# - `generate_input()` guarantees a 2-cell-wide source on the shelf, preventing the input-side
#   "bg on both horizontal sides" assertion from firing.
# - The generator fixes the source row and shaft mouths so the far-drop choice is always unique.


def _simulate(
    grid: np.ndarray,
    *,
    prefer_farthest: bool,
    check_input: bool = True,
    check_output: bool = True,
) -> np.ndarray:
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

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
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

                if prefer_farthest:
                    best_dist = max(abs(int(x - cc)) for x in drops)
                    best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(best_dist)]
                    assert len(best_cols) == 1, f"ambiguous farthest drop columns: {best_cols}"
                    best = int(best_cols[0])
                    assert best != int(cc)
                    for x in range(min(int(cc), best) + 1, max(int(cc), best) + 1):
                        assert int(g[rr, int(x)]) == int(bg)
                    cc = int(best)
                else:
                    best_dist = min(abs(int(x - cc)) for x in drops)
                    best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(best_dist)]
                    assert len(best_cols) == 1, f"ambiguous nearest drop columns: {best_cols}"
                    best = int(best_cols[0])
                    assert best != int(cc)
                    step = -1 if best < int(cc) else 1
                    nx = int(cc + step)
                    assert int(g[rr, nx]) == int(bg)
                    cc = nx
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


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling into coupled vertical basins under gravity, but with reversed shelf targeting.

    Input structure:
    - Background is 0 and water is color 1.
    - All other colors are fixed solids that form supported shelves, shafts, and bottom-anchored basins.
    - Input water begins in supported source stacks above the basin area, not yet at its final resting place.

    Rule:
    - Water falls straight down through background cells whenever possible.
    - If blocked below, it may slide diagonally down through the unique open side corridor.
    - When resting on a supported horizontal background run, it selects the unique farthest drop column
      in that run and slides across the shelf to that drop before resuming gravity.
    - Repeat this local motion until no water pixel can move.

    Key insight: The decisive action is not simple gravity but long horizontal commitment on supported shelves:
    water skips the nearer shaft and heads for the most distant available fall, so remote basins fill first.
    """
    return _simulate(
        grid,
        prefer_farthest=True,
        check_input=check_input,
        check_output=check_output,
    )


def _build_case(
    *,
    left_color: int,
    center_color: int,
    right_color: int,
    left_capacity: int,
    right_capacity: int,
    mirror: bool,
) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    shelf_row = 2
    roof_row = shelf_row + 1

    g[roof_row:, :] = int(center_color)
    g[roof_row:, :4] = int(left_color)
    g[roof_row:, 6:] = int(right_color)

    for r in range(roof_row, roof_row + int(left_capacity)):
        g[int(r), 1] = 0
    for r in range(roof_row, roof_row + int(right_capacity)):
        g[int(r), 8] = 0

    g[: shelf_row + 1, 3:5] = 1

    if mirror:
        g = np.fliplr(g)

    return g


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    near = _simulate(grid, prefer_farthest=False, check_input=True, check_output=True)
    far = main(grid, check_input=True, check_output=True)

    source_cols = np.argwhere(np.array(grid, dtype=int) == 1)[:, 1]
    source_on_left = int(source_cols.mean()) < 5
    near_col = 1 if source_on_left else 8
    far_col = 8 if source_on_left else 1

    return {
        "hamming_diff": int(np.sum(near != far)),
        "near_policy_near_shaft": int(np.sum(near[:, near_col] == 1)),
        "near_policy_far_shaft": int(np.sum(near[:, far_col] == 1)),
        "far_policy_near_shaft": int(np.sum(far[:, near_col] == 1)),
        "far_policy_far_shaft": int(np.sum(far[:, far_col] == 1)),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    palettes = [tuple(rng.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)) for _ in range(4)]
    capacity_options = [(4, 6), (4, 5), (3, 6)]

    for _ in range(128):
        left_color, center_color, right_color = rng.choice(palettes)
        left_capacity, right_capacity = rng.choice(capacity_options)
        mirror = bool(rng.getrandbits(1))
        grid = _build_case(
            left_color=int(left_color),
            center_color=int(center_color),
            right_color=int(right_color),
            left_capacity=int(left_capacity),
            right_capacity=int(right_capacity),
            mirror=mirror,
        )

        metrics = _behavior_metrics(grid)
        if (
            metrics["hamming_diff"] >= 4
            and metrics["far_policy_far_shaft"] >= metrics["near_policy_far_shaft"] + 2
            and metrics["near_policy_near_shaft"] >= metrics["far_policy_near_shaft"] + 2
        ):
            return grid

    raise AssertionError("generator could not build a visible far-drop variant")
