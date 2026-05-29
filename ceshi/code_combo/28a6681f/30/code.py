# Combo variant of 28a6681f (index 30)
# combo_spec: D2/0+D3/0+D18/0
# Name: 宽松对角线+最远落点+消歧规则

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ----------------
# a) Concrete behavior and trace:
#    The canonical layout uses a 3x2 supported source block on columns 4-5 or 5-6, resting above
#    three visible drop mouths on one shelf. For `src_left=4`, the decisive pixel eventually sits
#    at (2,4) and sees drops at columns 1, 6, and 7.
#    - Original rule chain:
#      The nearest drop is column 6, so water heads right. Even if it ever reaches the left shaft,
#      the cell at (5,1) cannot move diagonally because both same-row side cells are blocked.
#    - Combo rule chain:
#      The farthest drops are columns 1 and 7, so the shelf decision is ambiguous by distance and
#      is resolved left-first, sending water to column 1. There it lands above a blocker whose two
#      diagonal targets are open at (6,0) and (6,2), but both same-row side cells remain solid.
#      Loose diagonal motion allows crossing that side wall, and the diagonal tie is again resolved
#      left-first. The water therefore escapes into the hidden left pocket.
#    The resulting output differs visibly: combo samples split water between the left pocket and the
#    right shafts, while the original/near-drop physics keeps the moving water in the right basin.
# b) Complexity calibration:
#    The generator uses a small family of eight prevalidated geometries:
#    `src_left in {4,5}`, `left_shaft_depth in {3,4}`, and `right_shaft_depth in {5,6}`.
#    These changes are enough to vary the offset and final stack heights without changing the core
#    inference pattern. Each accepted sample repeats the fused behavior three times with six moving
#    source pixels, plus one already-stable carryover pixel in the far-right shaft, so the rule is
#    visible rather than a one-off accident and the final twin-shaft stack stays output-stable.
# c) Hidden assumptions audit:
#    - Original diagonal logic required both the side cell and the diagonal target to be background.
#      D2 invalidates that assumption, so the new diagonal predicate checks only the target cell and
#      records when the chosen move crossed a blocked side.
#    - Original horizontal motion could safely move one step and recompute because nearest-drop
#      targeting is monotone. After D3 inversion, that causes oscillation on shelves with drops on
#      both sides, so the fused solver commits directly to the chosen farthest drop column.
#    - Original code asserted uniqueness for diagonal and horizontal choices. D18 removes that
#      assumption and replaces it with one explicit geometric tie-breaker: lower column wins.
#    - Original upper-guide connectivity assertions were about optional decorative guide objects
#      rather than water physics. This combo family uses no such guides, so retaining only support
#      and final-stability checks avoids accidental rejection of otherwise valid fused samples.
# d) Placement legality:
#    Every structure is bottom-anchored inside a 10x10 grid. Water occupies only source cells in
#    the top three rows; solids and carved background pockets never overlap those cells. The left
#    hidden pocket is opened only when `left_drop-1 >= 0`, which the chosen offsets guarantee.
# e) Visual observability:
#    The decisive controls are all geometric and human-visible:
#    - three shelf drops show that the chosen drop is not the nearest one,
#    - the two farthest drops are symmetric in distance, so left-priority is inferable,
#    - the left shaft ends in a blocked-corner fork, making the loose diagonal rule visible.
#    A baseline nearest/open-side simulation produces a different bitmap on every generated sample,
#    and the generator checks this quantitatively before accepting the grid.
# f) Visual cleanliness:
#    The scene has only four readable regions: the top source block, the left hidden pocket, the
#    center roof/spine, and the right twin shafts. Solids are colored by broad column bands, so the
#    open channels remain easy to parse without extra markers or overlapping annotations.
# g) Color-rule audit:
#    Colors do not encode ordering. The solver's choices depend only on relative positions and
#    reachability. The generator merely assigns three distinct solid colors to left/center/right
#    structure bands to keep the landform legible.
# h) New solver/generator assumption review:
#    - The generator keeps the nearest drop unique while making the farthest pair tied, so the
#      observed ambiguity comes from the fused rule rather than an arbitrary baseline choice.
#    - The source is always a 3x2 block, which guarantees the input support assertions and also
#      ensures that the left source column repeatedly faces the same shelf decision after the right
#      source column drains away. One extra carryover water cell is parked at the bottom of the
#      far-right shaft so adjacent right-side water stacks end in a fully supported stable block.
#    - The lower fork always places solid blockers directly beside the resting cell above it, so a
#      blocked-side diagonal event is guaranteed rather than incidental.

BG = 0
WATER = 1
GRID_SHAPE = (10, 10)
SOURCE_HEIGHT = 3
SOLID_PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)
CASE_OPTIONS: tuple[tuple[int, int, int], ...] = (
    (4, 3, 5),
    (4, 3, 6),
    (4, 4, 5),
    (4, 4, 6),
    (5, 3, 5),
    (5, 3, 6),
    (5, 4, 5),
    (5, 4, 6),
)


def _validate_input(g: np.ndarray) -> None:
    H, W = g.shape
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"


def _validate_output(g: np.ndarray) -> None:
    H, W = g.shape
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
        assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)


def _simulate(
    grid: np.ndarray,
    *,
    allow_blocked_side_diagonal: bool,
    prefer_farthest_drop: bool,
    tie_break_left: bool,
    check_input: bool = True,
    check_output: bool = True,
    collect_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    metrics = {
        "blocked_side_diagonals": 0,
        "diag_ties": 0,
        "horizontal_ties": 0,
        "farthest_slides": 0,
        "moved_pixels": 0,
    }

    if check_input:
        _validate_input(g)

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            diag_options: list[tuple[int, int, bool]] = []
            for dc in (-1, 1):
                nc = int(cc + dc)
                if rr + 1 >= H or not (0 <= nc < W):
                    continue
                if int(g[rr + 1, nc]) != int(BG):
                    continue
                side_blocked = int(g[rr, nc]) != int(BG)
                if side_blocked and not allow_blocked_side_diagonal:
                    continue
                diag_options.append((int(rr + 1), nc, bool(side_blocked)))

            if diag_options:
                if len(diag_options) > 1:
                    metrics["diag_ties"] += 1
                chosen = min(diag_options, key=lambda item: int(item[1])) if tie_break_left else diag_options[0]
                if chosen[2]:
                    metrics["blocked_side_diagonals"] += 1
                rr, cc = int(chosen[0]), int(chosen[1])
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

                if prefer_farthest_drop:
                    best_distance = max(abs(int(x - cc)) for x in drops)
                    best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(best_distance)]
                    if len(best_cols) > 1:
                        metrics["horizontal_ties"] += 1
                    best = min(best_cols) if tie_break_left else best_cols[0]
                    if best == int(cc):
                        break
                    metrics["farthest_slides"] += 1
                    cc = int(best)
                    continue

                best_distance = min(abs(int(x - cc)) for x in drops)
                best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(best_distance)]
                if len(best_cols) > 1:
                    metrics["horizontal_ties"] += 1
                best = min(best_cols) if tie_break_left else best_cols[0]
                if best == int(cc):
                    break
                cc = int(cc + (-1 if best < int(cc) else 1))
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _ in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            if (int(rr), int(cc)) != (int(r), int(c)):
                moved = True
                metrics["moved_pixels"] += 1
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        _validate_output(g)

    if collect_metrics:
        return g, metrics
    return g


def _build_case(
    *,
    src_left: int,
    left_shaft_depth: int,
    right_shaft_depth: int,
    colors: tuple[int, int, int],
) -> np.ndarray:
    left_color, center_color, right_color = colors
    g = np.zeros(GRID_SHAPE, dtype=int)
    H, W = g.shape

    for r in range(3, H):
        for c in range(W):
            if c <= int(src_left - 1):
                g[r, c] = int(left_color)
            elif c <= int(src_left + 1):
                g[r, c] = int(center_color)
            else:
                g[r, c] = int(right_color)

    g[:SOURCE_HEIGHT, src_left : src_left + 2] = int(WATER)

    left_drop = int(src_left - 3)
    near_drop = int(src_left + 2)
    far_drop = int(src_left + 3)
    chamber_row = int(3 + left_shaft_depth)

    for r in range(3, 3 + int(left_shaft_depth)):
        g[int(r), int(left_drop)] = int(BG)
    for r in range(3, 3 + int(right_shaft_depth)):
        g[int(r), int(near_drop)] = int(BG)
        g[int(r), int(far_drop)] = int(BG)

    g[int(chamber_row), int(left_drop - 1)] = int(BG)
    g[int(chamber_row), int(left_drop + 1)] = int(BG)
    for r in range(int(chamber_row + 1), 9):
        g[int(r), int(left_drop - 1)] = int(BG)
        g[int(r), int(left_drop + 1)] = int(BG)

    # Keep one stable carryover cell in the far-right shaft so the final adjacent shaft stack has
    # no exposed horizontal face while leaving the fused route choice unchanged.
    g[int(3 + right_shaft_depth - 1), int(far_drop)] = int(WATER)

    return g


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    combo_out, combo_metrics = _simulate(
        grid,
        allow_blocked_side_diagonal=True,
        prefer_farthest_drop=True,
        tie_break_left=True,
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    assert isinstance(combo_out, np.ndarray)
    assert isinstance(combo_metrics, dict)
    baseline = _simulate(
        grid,
        allow_blocked_side_diagonal=False,
        prefer_farthest_drop=False,
        tie_break_left=True,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    assert isinstance(baseline, np.ndarray)

    metrics = dict(combo_metrics)
    metrics["combo_vs_baseline_diff"] = int(np.sum(combo_out != baseline))
    metrics["water_count"] = int(np.sum(grid == int(WATER)))
    return metrics


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling in coupled basins with corner-cutting diagonals and long-range shelf choices.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors are static bottom-anchored solids arranged as a roof shelf, three visible drop
      mouths, and a lower forked basin with hidden side pockets.
    - Water starts as supported source blocks above the shelf, with a tiny stable carryover allowed
      at the base of the far-right shaft.

    Rule:
    - Water falls straight down through background whenever a cell below is open.
    - If blocked below, it may move diagonally down-left or down-right whenever the diagonal target
      is background, even if the same-row side cell is blocked.
    - If both diagonals are legal, choose the left one.
    - When resting on a supported horizontal run, choose the farthest reachable drop column in that
      run; if multiple farthest drops tie, choose the leftmost one and commit to that column before
      falling again.
    - Repeat until no water pixel can move.

    Key insight: The output is determined by a three-stage fused policy: water can cut blocked
    corners, it prefers the most distant fall rather than the nearest one, and every left/right
    ambiguity is resolved consistently to the left.
    """
    result = _simulate(
        grid,
        allow_blocked_side_diagonal=True,
        prefer_farthest_drop=True,
        tie_break_left=True,
        check_input=check_input,
        check_output=check_output,
        collect_metrics=False,
    )
    assert isinstance(result, np.ndarray)
    return result


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        src_left, left_shaft_depth, right_shaft_depth = rng.choice(CASE_OPTIONS)
        colors = tuple(rng.sample(list(SOLID_PALETTE), 3))
        grid = _build_case(
            src_left=int(src_left),
            left_shaft_depth=int(left_shaft_depth),
            right_shaft_depth=int(right_shaft_depth),
            colors=(int(colors[0]), int(colors[1]), int(colors[2])),
        )

        metrics = _behavior_metrics(grid)
        if metrics["water_count"] != 7:
            continue
        if metrics["blocked_side_diagonals"] < 1:
            continue
        if metrics["diag_ties"] < 1:
            continue
        if metrics["horizontal_ties"] < 1:
            continue
        if metrics["farthest_slides"] < 1:
            continue
        if metrics["moved_pixels"] < 4:
            continue
        if metrics["combo_vs_baseline_diff"] < 4:
            continue
        return grid

    raise AssertionError("generator could not build a visible fused sample")
