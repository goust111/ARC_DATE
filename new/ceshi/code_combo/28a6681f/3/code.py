# Combo variant of 28a6681f (index 3)
# combo_spec: D2/0+D3/0
# Name: 宽松对角线+最远落点

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ----------------
# a) Concrete behavior and trace:
#    We use one trapped source column on the high side, one corner-cutting exit, and two drop
#    shafts on the same supported shelf. A representative non-mirrored case is:
#    - Source water is a 5-cell vertical column at column 8, rows 0-4.
#    - The cell at (4, 8) is supported by a solid at (5, 8). Its left neighbor (4, 7) is solid,
#      but the diagonal target (5, 7) is background.
#    - The shelf at row 5 is background from columns 1-7. Below it, column 5 is a nearby shaft
#      and column 1 is a farther shaft.
#    Original rule:
#    - The bottom source cell cannot move diagonally because the side cell (4, 7) is blocked.
#    - Therefore the whole source stays trapped on the high side; the far-drop rule never gets a
#      chance to act.
#    D2-only:
#    - The same cell may now move to (5, 7) because only the diagonal target matters.
#    - Once on the shelf, nearest-drop logic sends it first toward column 5.
#    Combo:
#    - The cell still cuts the blocked corner to reach (5, 7).
#    - On that shelf, it now commits to the unique farthest legal drop, column 1, bypassing the
#      nearer shaft at column 5.
#    - Repeating this on all source cells moves the whole source into the remote basin instead of
#      the nearer one.
# b) Complexity calibration:
#    The layout keeps the original 10x10 scale, 1 water color, and 3 solid colors, but reduces
#    the scene to one decisive motif. We vary only mirror direction, shelf height, and the two
#    shaft depths. That is enough to avoid frozen duplicates while keeping the rule legible across
#    2-4 examples.
# c) Hidden-assumption audit:
#    - The original diagonal code assumes "open side corridor" is part of legality. D2/0 breaks
#      that assumption, so diagonal checks must consult only the diagonal landing cell.
#    - The original horizontal code recomputes the nearest drop after each 1-step slide. That is
#      safe for nearest-drop motion but incompatible with D3/0 because a farthest target can
#      switch mid-slide or oscillate. The combo solver therefore chooses the unique farthest drop
#      for the current shelf contact and commits directly to that column before gravity resumes.
#    - Ambiguous diagonals and ambiguous farthest drops still remain illegal. The generator builds
#      a single corner exit and two same-direction shafts so every accepted sample has uniqueness.
# d) Placement legality:
#    All solids are bottom-anchored. The source wall, shelf, shafts, and source water are built by
#    fixed in-bounds coordinates, then optionally mirrored. The shelf and shaft carving never
#    overlap the source support cell, so the corner exit and both drop columns remain distinct.
# e) Visual observability:
#    A human can see the decisive evidence directly: the input water starts in a trapped column on
#    one side, but the output settles in a farther basin that is not adjacent to the source. The
#    generator rejects cases where combo output matches the original or either single-variant
#    behavior, so the training examples consistently expose both rule changes.
# f) Visual cleanliness:
#    The grid contains one source column, one horizontal shelf, and two clean shafts cut into one
#    bottom mass. There are no overlapping markers or color-coded control signals; all reasoning
#    is geometric.
# g) Change inferability:
#    Colors do not encode any ordering. All choices are visible from position alone: a blocked
#    corner next to the source and two different shaft distances on the same shelf.
# Solver/generator assumption review:
# - `main()` still assumes input water is supported and never has background on both horizontal
#   sides; the generator enforces that with a wall on one side of the source column.
# - The guide-consistency check from the original solver is preserved. Upper-row solids reuse
#   colors that also appear on the bottom row, so those checks never misclassify the source wall
#   as a separate guide object.
# - Horizontal target uniqueness depends on the shelf exposing exactly two drop columns on one
#   side of the source entry point; `_build_case()` guarantees that geometry before validation.


CASE_SPECS: tuple[tuple[int, int, int], ...] = (
    (4, 3, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 4, 3),
    (5, 3, 3),
    (5, 4, 2),
    (5, 4, 3),
)


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
    relaxed_diagonal: bool,
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
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            opts = []
            if rr + 1 < H and cc - 1 >= 0 and int(g[rr + 1, cc - 1]) == int(bg):
                if relaxed_diagonal or int(g[rr, cc - 1]) == int(bg):
                    opts.append((int(rr + 1), int(cc - 1)))
            if rr + 1 < H and cc + 1 < W and int(g[rr + 1, cc + 1]) == int(bg):
                if relaxed_diagonal or int(g[rr, cc + 1]) == int(bg):
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
    Simulate water settling in coupled basins when blocked-corner diagonals are legal and shelves target the farthest drop.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors are static bottom-anchored solids that form a trapped source pocket, a supported shelf,
      and multiple basin shafts below that shelf.
    - Input water begins as supported source columns or runs pressed against a wall, not yet at its final basin.

    Rule:
    - Water falls straight down through background whenever possible.
    - If blocked below, it may move diagonally down-left or down-right whenever the diagonal landing cell is
      background; the side-adjacent cell no longer needs to be open, but the diagonal choice must still be unique.
    - When resting on a supported horizontal background run, it selects the unique farthest drop column in that run,
      slides all the way to that column, and then resumes gravity.
    - Repeat this local motion until no water pixel can move.

    Key insight: The decisive route is a two-stage detour. Water first cuts through a blocked corner that the
    original task would forbid, then commits to the most distant reachable shaft on the shelf, so a remote basin
    fills before a nearer one.
    """
    return _simulate(
        grid,
        relaxed_diagonal=True,
        prefer_farthest=True,
        check_input=check_input,
        check_output=check_output,
    )


def _build_case(
    *,
    left_color: int,
    center_color: int,
    right_color: int,
    shelf_row: int,
    far_depth: int,
    near_depth: int,
    mirror: bool,
) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    support_row = int(shelf_row + 1)

    g[support_row:, :] = int(center_color)
    g[support_row:, :3] = int(left_color)
    g[support_row:, 7:] = int(right_color)

    g[int(shelf_row), 1:8] = 0
    g[int(shelf_row), 8:] = int(right_color)
    g[: int(shelf_row), 7] = int(right_color)

    for r in range(int(shelf_row)):
        g[int(r), 8] = 1

    for r in range(int(support_row), min(10, int(support_row + far_depth))):
        g[int(r), 1] = 0
    for r in range(int(support_row), min(10, int(support_row + near_depth))):
        g[int(r), 5] = 0

    if mirror:
        g = np.fliplr(g)

    return g


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    original = _simulate(
        grid,
        relaxed_diagonal=False,
        prefer_farthest=False,
        check_input=True,
        check_output=False,
    )
    diag_only = _simulate(
        grid,
        relaxed_diagonal=True,
        prefer_farthest=False,
        check_input=True,
        check_output=False,
    )
    far_only = _simulate(
        grid,
        relaxed_diagonal=False,
        prefer_farthest=True,
        check_input=True,
        check_output=False,
    )
    combo = main(grid, check_input=True, check_output=True)

    source_on_left = float(np.argwhere(np.array(grid, dtype=int) == 1)[:, 1].mean()) < 5.0
    near_col = 4 if source_on_left else 5
    far_col = 8 if source_on_left else 1

    return {
        "combo_vs_original": int(np.sum(combo != original)),
        "combo_vs_diag_only": int(np.sum(combo != diag_only)),
        "combo_vs_far_only": int(np.sum(combo != far_only)),
        "combo_far_basin": int(np.sum(combo[:, far_col] == 1)),
        "diag_far_basin": int(np.sum(diag_only[:, far_col] == 1)),
        "combo_near_basin": int(np.sum(combo[:, near_col] == 1)),
        "diag_near_basin": int(np.sum(diag_only[:, near_col] == 1)),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    palettes = [tuple(rng.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)) for _ in range(6)]

    for _ in range(128):
        shelf_row, far_depth, near_depth = rng.choice(CASE_SPECS)
        left_color, center_color, right_color = rng.choice(palettes)
        mirror = bool(rng.getrandbits(1))
        grid = _build_case(
            left_color=int(left_color),
            center_color=int(center_color),
            right_color=int(right_color),
            shelf_row=int(shelf_row),
            far_depth=int(far_depth),
            near_depth=int(near_depth),
            mirror=mirror,
        )

        metrics = _behavior_metrics(grid)
        if (
            metrics["combo_vs_original"] >= 8
            and metrics["combo_vs_diag_only"] >= 2
            and metrics["combo_vs_far_only"] >= 8
            and metrics["combo_far_basin"] >= metrics["diag_far_basin"] + 1
            and metrics["combo_near_basin"] <= metrics["diag_near_basin"] - 1
        ):
            return grid

    raise AssertionError("generator could not build a visible combo sample")
