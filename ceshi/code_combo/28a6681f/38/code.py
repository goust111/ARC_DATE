# Combo variant of 28a6681f (index 38)
# combo_spec: D2/0+D3/0+D7/2+D18/0
# Name: 宽松对角线+最远落点+带孔壁面+消歧

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ----------------
# a) Concrete behavior and trace
#    The fused task keeps the 10x10 gravity puzzle, but the terrain is now a chained
#    three-basin scene with two perforated dividers and a top edge-entering source ledge.
#    A compact source block rests on that ledge. One extra suspended peg sits across a gap,
#    so a draining source pixel reaches a state with:
#    - below blocked,
#    - both diagonal targets open,
#    - the same-row side cells occupied by source water.
#    Original rule on the same geometry:
#    - strict diagonal needs an open side cell, so that pixel cannot corner-cut through the
#      packed source block;
#    - supported shelf motion then prefers the nearest drop;
#    - ambiguous two-way choices would assert.
#    Fused rule:
#    - D2 allows the pixel to move diagonally even though the side cell is blocked;
#    - D18 resolves the resulting two-way diagonal tie toward the guide/source side;
#    - once on a supported leak row, D3 commits to the farthest available drop, often on the
#      far side of one or both divider holes rather than the near local drop;
#    - D7 makes those holes visible side passages, so the water can reach a remote basin while
#      the final waterline still stays below the divider tops.
#    The visual difference is large: under the fused rule, water jumps through blocked corners,
#    crosses wall gaps, and ends in the far basin; under the original path logic, it stays much
#    closer to the source side.
#
# b) Complexity calibration
#    The generator uses a small family of six prevalidated templates rather than a single frozen
#    bitmap. The repeated invariants are:
#    - source block width is 3 or 4, always as a 2-row supported rectangle;
#    - divider holes are at one shared row (4 or 5), so the leak mechanism repeats visibly;
#    - one suspended peg creates repeated relaxed-diagonal events;
#    - the basin chain always uses the same coarse columns (left / middle / remote), so humans
#      can compare examples without relearning the whole layout.
#    This keeps the four fused rules legible while still giving multiple examples.
#
# c) Implicit-assumption audit for main()
#    - The original `len(opts) == 1` assumption on diagonal motion is no longer valid.
#      We must keep all open diagonal targets and resolve ties deterministically.
#    - The original diagonal side-cell check is no longer sound. We now distinguish:
#      `diag_all` = any open diagonal target; `diag_strict` = old side-open subset. Metrics count
#      when a move exists only because blocked-side diagonal is allowed.
#    - The original nearest-drop, one-step slide is not compatible with D3 inversion. To avoid
#      oscillation, the fused solver commits directly to the selected farthest target column.
#    - Hole traversal does not need a bespoke motion primitive: a wall hole simply enlarges the
#      same supported horizontal run. The generator therefore guarantees hole rows line up with
#      supported water shelves.
#    - D18 adds a new hidden input assumption: tie preference must be inferable from the input.
#      This solver reads the upper source side from the topmost water band and reuses that fixed
#      preference for all diagonal and horizontal ties.
#
# d) Placement legality
#    The generator keeps all geometry inside the 10x10 grid and uses fixed wall columns so the
#    basin chain never overlaps the source ledge. Source water is always supported from below and
#    forms a compact rectangle, so the inherited input assertions still hold.
#
# e) Visual observability
#    Each accepted sample must satisfy all of the following:
#    - at least one relaxed-only diagonal event happened;
#    - at least one ambiguity was resolved by the source-side tie rule;
#    - at least two farthest-drop commitments happened;
#    - the remote basin contains water even though its top water cell remains below the hole row,
#      proving side leakage rather than only crest overflow;
#    - the fused output differs from both `strict+far` and `relaxed+near` ablations.
#    This gives direct evidence for all four fused changes in 2-4 examples.
#
# f) Visual cleanliness
#    The grids use only water, background, and three solid colors. The scene is organized into
#    three horizontal bands: source ledge at the top, two leak dividers in the middle, and stable
#    basins at the bottom. No extra markers or symbolic overlays are used.
#
# g) Color inference and new assumptions review
#    Colors do not encode ordering or any hidden numeric rule. All choices are geometric:
#    source side, wall continuity, hole row, and drop distance. The only new solver assumption is
#    that the topmost water band represents one dominant source side; the generator guarantees that
#    by always using a single compact source block and no pre-placed carryover water.

BG = 0
WATER = 1
WALL_COLUMNS = (3, 6)
OBSTACLE_COLORS = tuple(range(2, 10))

_PROFILES = (
    {"src_left": 2, "src_width": 3, "peg_col": 6, "hole_row": 4, "left_wall_top": 4, "mid_wall_top": 3, "mirror": False},
    {"src_left": 2, "src_width": 3, "peg_col": 6, "hole_row": 5, "left_wall_top": 4, "mid_wall_top": 5, "mirror": False},
    {"src_left": 1, "src_width": 4, "peg_col": 6, "hole_row": 4, "left_wall_top": 4, "mid_wall_top": 3, "mirror": False},
    {"src_left": 1, "src_width": 4, "peg_col": 7, "hole_row": 4, "left_wall_top": 4, "mid_wall_top": 3, "mirror": False},
    {"src_left": 1, "src_width": 4, "peg_col": 6, "hole_row": 4, "left_wall_top": 5, "mid_wall_top": 5, "mirror": False},
    {"src_left": 2, "src_width": 3, "peg_col": 6, "hole_row": 4, "left_wall_top": 5, "mid_wall_top": 5, "mirror": False},
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


def _prefer_left_from_source(grid: np.ndarray) -> bool:
    pts = np.argwhere(np.array(grid, dtype=int) == int(WATER))
    assert len(pts) > 0, "grid must contain source water"
    top_row = int(pts[:, 0].min())
    top_band = pts[pts[:, 0] <= top_row + 1]
    center = (grid.shape[1] - 1) / 2.0
    return bool(float(top_band[:, 1].mean()) <= center)


def _simulate(
    grid: np.ndarray,
    *,
    strict_diag_side: bool,
    prefer_farthest: bool,
    check_input: bool = True,
    check_output: bool = True,
    collect_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    prefer_left = _prefer_left_from_source(g)
    metrics = {
        "relaxed_only_diag": 0,
        "diag_ties": 0,
        "horiz_ties": 0,
        "farthest_commits": 0,
        "hole_crossings": 0,
        "moved_pixels": 0,
        "tie_pref_left": int(prefer_left),
        "tie_pref_right": int(not prefer_left),
    }

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def choose_column(columns: list[int]) -> int:
        return int(min(columns) if prefer_left else max(columns))

    def choose_position(options: list[tuple[int, int]]) -> tuple[int, int]:
        target_col = choose_column([int(c) for _r, c in options])
        for rr, cc in options:
            if int(cc) == int(target_col):
                return int(rr), int(cc)
        raise AssertionError("failed to choose tied position")

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            diag_all: list[tuple[int, int]] = []
            diag_strict: list[tuple[int, int]] = []
            for dc in (-1, 1):
                nc = int(cc + dc)
                if rr + 1 < H and 0 <= nc < W and int(g[rr + 1, nc]) == int(BG):
                    diag_all.append((int(rr + 1), int(nc)))
                    if int(g[rr, nc]) == int(BG):
                        diag_strict.append((int(rr + 1), int(nc)))

            diag_opts = diag_strict if strict_diag_side else diag_all
            if diag_opts:
                if not strict_diag_side and diag_all and not diag_strict:
                    metrics["relaxed_only_diag"] += 1
                if len(diag_opts) > 1:
                    metrics["diag_ties"] += 1
                rr, cc = choose_position(diag_opts)
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                L = int(cc)
                while L - 1 >= 0 and int(g[rr, L - 1]) == int(BG):
                    L -= 1
                R = int(cc)
                while R + 1 < W and int(g[rr, R + 1]) == int(BG):
                    R += 1

                drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                best_dist = (
                    max(abs(int(x - cc)) for x in drops)
                    if prefer_farthest
                    else min(abs(int(x - cc)) for x in drops)
                )
                best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(best_dist)]
                if len(best_cols) > 1:
                    metrics["horiz_ties"] += 1
                target_col = choose_column(best_cols)
                if int(target_col) == int(cc):
                    break

                if prefer_farthest:
                    if abs(int(target_col - cc)) > 1:
                        metrics["farthest_commits"] += 1
                    metrics["hole_crossings"] += sum(
                        1
                        for wall in WALL_COLUMNS
                        if (int(cc) < int(wall) < int(target_col)) or (int(target_col) < int(wall) < int(cc))
                    )
                    cc = int(target_col)
                else:
                    cc += -1 if int(target_col) < int(cc) else 1
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            if (int(rr), int(cc)) != (int(r), int(c)):
                metrics["moved_pixels"] += 1
                moved = True
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    if collect_metrics:
        return g, metrics
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate coupled basin-filling water with blocked-side diagonals, farthest-drop shelves, wall holes, and deterministic tie-breaking.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors form a bottom-anchored basin chain plus an upper edge-entering source ledge.
    - Divider walls may contain one-cell holes, and a small suspended peg can create pinched corner escapes above the basins.

    Rule:
    - Water falls straight down through background cells whenever possible.
    - If blocked below, it may move diagonally to any open down-left or down-right target even when the same-row side cell is occupied.
    - When resting on a supported horizontal background run, it chooses the farthest reachable drop column in that run and commits directly to it.
    - If multiple diagonal targets or multiple equally far drop columns are tied, choose toward the visible source side: left for a left-side source block, right for a right-side source block.

    Key insight: Reachability expands in three ways at once: blocked corners no longer stop diagonal motion, wall holes join separate basins into one leak row, and shelf motion skips nearby drops in favor of the farthest one. A fixed side-priority rule then resolves the many ambiguities this wider reach creates.
    """
    result = _simulate(
        grid,
        strict_diag_side=False,
        prefer_farthest=True,
        check_input=check_input,
        check_output=check_output,
        collect_metrics=False,
    )
    assert isinstance(result, np.ndarray)
    return result


def _build_template(profile: dict[str, int | bool]) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)

    src_left = int(profile["src_left"])
    src_width = int(profile["src_width"])
    peg_col = int(profile["peg_col"])
    hole_row = int(profile["hole_row"])
    left_wall_top = int(profile["left_wall_top"])
    mid_wall_top = int(profile["mid_wall_top"])

    g[1:3, src_left : src_left + src_width] = int(WATER)
    g[3, : src_left + src_width] = 2
    g[3, peg_col] = 3

    for r in range(left_wall_top, 9):
        if r != hole_row:
            g[r, 3] = 2

    g[4, 5] = 3

    for r in range(mid_wall_top, 9):
        if r != hole_row:
            g[r, 6] = 3

    for r in range(4, 9):
        g[r, 9] = 4

    g[9, :4] = 2
    g[9, 4:7] = 3
    g[9, 7:] = 4

    if bool(profile["mirror"]):
        g = np.fliplr(g).copy()

    return g


def _colorize(template: np.ndarray, colors: tuple[int, int, int]) -> np.ndarray:
    out = np.array(template, dtype=int).copy()
    for code, color in zip((2, 3, 4), colors):
        out[out == int(code)] = int(color)
    return out


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.array(a, dtype=int) != np.array(b, dtype=int)))


def _leak_visibility(output: np.ndarray, hole_row: int, prefer_left: bool) -> bool:
    remote_cols = (7, 8) if prefer_left else (1, 2)
    remote = np.argwhere(output[:, remote_cols] == int(WATER))
    middle = np.argwhere(output[:, 4:6] == int(WATER))
    if len(remote) == 0 or len(middle) == 0:
        return False
    top_remote = int(remote[:, 0].min())
    holes_open = int(output[hole_row, 3]) == int(BG) and int(output[hole_row, 6]) == int(BG)
    return bool(top_remote > int(hole_row) and holes_open)


def _behavior_report(grid: np.ndarray, hole_row: int) -> tuple[np.ndarray, dict[str, int]]:
    combo, combo_metrics = _simulate(
        grid,
        strict_diag_side=False,
        prefer_farthest=True,
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    assert isinstance(combo, np.ndarray)
    assert isinstance(combo_metrics, dict)

    strict_near = _simulate(
        grid,
        strict_diag_side=True,
        prefer_farthest=False,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    strict_far = _simulate(
        grid,
        strict_diag_side=True,
        prefer_farthest=True,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    relaxed_near = _simulate(
        grid,
        strict_diag_side=False,
        prefer_farthest=False,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    assert isinstance(strict_near, np.ndarray)
    assert isinstance(strict_far, np.ndarray)
    assert isinstance(relaxed_near, np.ndarray)

    report = dict(combo_metrics)
    report["diff_from_originalized"] = _hamming(combo, strict_near)
    report["diff_from_strict_far"] = _hamming(combo, strict_far)
    report["diff_from_relaxed_near"] = _hamming(combo, relaxed_near)
    report["visible_leak"] = int(
        _leak_visibility(combo, hole_row=hole_row, prefer_left=bool(report["tie_pref_left"]))
    )
    return combo, report


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        profile = dict(rng.choice(_PROFILES))
        colors = tuple(rng.sample(OBSTACLE_COLORS, 3))
        grid = _colorize(_build_template(profile), colors)

        combo, report = _behavior_report(grid, hole_row=int(profile["hole_row"]))
        solved = main(grid, check_input=True, check_output=True)

        if not np.array_equal(combo, solved):
            continue
        if np.array_equal(grid, combo):
            continue
        if report["relaxed_only_diag"] < 1:
            continue
        if report["farthest_commits"] < 2:
            continue
        if report["diag_ties"] + report["horiz_ties"] < 1:
            continue
        if report["hole_crossings"] < 2:
            continue
        if report["moved_pixels"] < 6:
            continue
        if report["visible_leak"] != 1:
            continue
        if report["diff_from_originalized"] < 4:
            continue
        if report["diff_from_strict_far"] < 2:
            continue
        if report["diff_from_relaxed_near"] < 6:
            continue
        return grid

    raise AssertionError("failed to sample a valid combo-38 input")
