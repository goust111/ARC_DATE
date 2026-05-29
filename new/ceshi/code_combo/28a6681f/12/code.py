# Combo variant of 28a6681f (index 12)
# combo_spec: D1/2+D9/2
# Name: 反向重力+单向溢流

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# a) Concrete behavior:
#    The original task had bottom basins and downward gravity. This combo vertically flips the
#    whole physical picture and then adds a global side bias for overflow. The generated scene is
#    a 10x10 canvas with three top-anchored inverted basins:
#    - left cup between the outer wall at col 0 and the short divider at col 3,
#    - source cup between col 3 and the longer divider at col 6,
#    - right cup between col 6 and the outer wall at col 9.
#    Water starts in a bottom reservoir hanging under a short roof that touches exactly one side
#    edge. In the canonical left-to-right case that roof is row 6, cols 0..3 and the reservoir
#    fills rows 7..9, cols 0..3. A concrete trace:
#    - Original: a source pixel drops into a lower basin, then may overflow over the lower wall.
#    - Combo: the same pixel rises, enters the middle inverted basin, and once blocked above it
#      can only keep sliding toward the globally allowed side. The left divider is visibly shorter
#      than the right divider, so a bidirectional reverse-gravity rule would spill left first.
#      The combo instead ignores that easier left escape and keeps feeding the right cup only.
#    - Difference: the solver must combine a vertical inversion with a non-local directional cue.
#
# b) Complexity calibration:
#    The layout family is intentionally narrow: three cups, one lower source roof, and only five
#    validated (divider-height, water-count) profiles. This keeps the rule legible while still
#    showing both 10/11/12-pixel fills and two right-divider depths. Horizontal mirroring is the
#    only orientation variation, so humans see repeated evidence that the active overflow side is
#    tied to the lower edge-entering source structure rather than to absolute left/right.
#
# c) Implicit-assumption audit against the old main():
#    - Old `settle()` assumed monotone downward motion. Every vertical branch must be mirrored to
#      upward motion.
#    - Old diagonal movement required a unique open side. After adding one-way overflow, some
#      mirrored overflow states can expose both up-left and up-right diagonals. The combo resolves
#      that ambiguity using the same global overflow direction; without that, valid generated
#      samples would crash.
#    - Old horizontal sliding picked the nearest drop on either side. That is the key broken
#      assumption: in this combo the forbidden side is ignored even when it offers the easier rise.
#    - Old guide detection looked for diagonal-rich upper components distinct from bottom basins.
#      That no longer holds because the upper structures are the real containers and the lower cue
#      is just a short launch roof, so the guide heuristic is removed.
#    - Old output checks required left/below/right enclosure for every water pixel. That is invalid
#      here because one-way overflow can leave a water edge exposed on the forbidden side while
#      still being a true fixed point. The new output check explicitly verifies "remove one pixel,
#      resettle it, and it returns to the same cell".
#    New solver assumptions:
#    - A non-water solid marker exists on exactly one lower side edge (row >= 6), revealing the
#      allowed overflow direction.
#    - Generated inputs use only the validated cup family, so the directional tie-break always
#      resolves to a single target cell.
#
# d) Placement legality:
#    All structures stay inside the 10x10 frame. The top basins occupy rows 1..5 or 1..6, the
#    lower source roof is fixed at row 6, and source water is packed strictly below it. No input
#    water overlaps solids; support is guaranteed because the reservoir is filled top-down, so each
#    lower source cell has either the roof or another water cell directly above it.
#
# e) Visual observability:
#    Every accepted sample is checked against a bidirectional reverse-gravity baseline. The combo is
#    kept only if:
#    - the allowed-side cup receives water,
#    - the forbidden-side cup stays empty,
#    - and the baseline would have filled the forbidden-side cup instead.
#    This makes the fusion human-visible from the geometry alone: the shorter wrong-way divider is
#    ignored, so the active cue must be the lower edge-entering source direction.
#
# f) Visual cleanliness:
#    Only four semantic layers exist: background, water, the lower launch roof, and two top-basin
#    color regions. There are no overlapping symbols, no decorative guides, and a blank bottom band
#    remains after solving, so the rising motion is easy to read.
#
# g) Color inferability:
#    Colors do not encode motion priority. They only separate the left/source/right solid pieces.
#    The direction rule is structural: infer it from which lower side edge contains the source roof,
#    then watch overflow propagate only away from that entry side.


BG = 0
WATER = 1
SIZE = 10

LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = -1

TOP_ROOF_ROW = 1
SOURCE_ROOF_ROW = 6
LEFT_WALL_COL = 0
LEFT_DIVIDER_COL = 3
RIGHT_DIVIDER_COL = 6
RIGHT_WALL_COL = 9

PALETTE = tuple(range(2, 10))

PROFILES: tuple[dict[str, int], ...] = (
    {"left_outer": 5, "left_divider": 4, "right_divider": 5, "right_outer": 5, "water_count": 10},
    {"left_outer": 5, "left_divider": 4, "right_divider": 5, "right_outer": 5, "water_count": 11},
    {"left_outer": 5, "left_divider": 4, "right_divider": 5, "right_outer": 5, "water_count": 12},
    {"left_outer": 5, "left_divider": 4, "right_divider": 6, "right_outer": 5, "water_count": 11},
    {"left_outer": 5, "left_divider": 4, "right_divider": 6, "right_outer": 5, "water_count": 12},
)


def _mirror(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid).copy()


def _detect_overflow_direction(grid: np.ndarray) -> int:
    left_signal = sum(
        int(grid[r, 0] not in (BG, WATER))
        for r in range(SOURCE_ROOF_ROW, SIZE)
    )
    right_signal = sum(
        int(grid[r, SIZE - 1] not in (BG, WATER))
        for r in range(SOURCE_ROOF_ROW, SIZE)
    )
    if left_signal > 0 and right_signal == 0:
        return LEFT_TO_RIGHT
    if right_signal > 0 and left_signal == 0:
        return RIGHT_TO_LEFT
    raise AssertionError("lower source roof must touch exactly one side edge")


def _source_fill_order(direction: int) -> list[tuple[int, int]]:
    base = [(r, c) for r in range(SOURCE_ROOF_ROW + 1, SIZE) for c in range(0, 4)]
    if direction == LEFT_TO_RIGHT:
        return base
    return [(int(r), int(SIZE - 1 - c)) for r, c in base]


def _canonicalize(grid: np.ndarray, direction: int) -> np.ndarray:
    return np.array(grid, dtype=int).copy() if direction == LEFT_TO_RIGHT else _mirror(grid)


def _build_canonical_layout(colors: tuple[int, int, int], profile: dict[str, int]) -> np.ndarray:
    left_color, source_color, right_color = colors
    g = np.zeros((SIZE, SIZE), dtype=int)

    g[TOP_ROOF_ROW, 0:4] = left_color
    g[TOP_ROOF_ROW, 3:7] = source_color
    g[TOP_ROOF_ROW, 6:10] = right_color

    g[TOP_ROOF_ROW : int(profile["left_outer"]) + 1, LEFT_WALL_COL] = left_color
    g[TOP_ROOF_ROW : int(profile["left_divider"]) + 1, LEFT_DIVIDER_COL] = left_color
    g[TOP_ROOF_ROW : int(profile["right_divider"]) + 1, RIGHT_DIVIDER_COL] = right_color
    g[TOP_ROOF_ROW : int(profile["right_outer"]) + 1, RIGHT_WALL_COL] = right_color

    g[SOURCE_ROOF_ROW, 0:4] = source_color
    return g


def _build_input(
    colors: tuple[int, int, int],
    profile: dict[str, int],
    direction: int,
) -> np.ndarray:
    g = _build_canonical_layout(colors, profile)
    for r, c in _source_fill_order(LEFT_TO_RIGHT)[: int(profile["water_count"])]:
        g[r, c] = WATER
    if direction == RIGHT_TO_LEFT:
        g = _mirror(g)
    return g


def _simulate(
    grid: np.ndarray,
    *,
    overflow_dir: int | None,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    initial_water = int(np.sum(g == WATER))

    if check_input:
        assert g.shape == (SIZE, SIZE), "generator and solver assume a 10x10 canvas"
        assert initial_water > 0, "input must contain water"
        if overflow_dir is not None:
            assert overflow_dir in (LEFT_TO_RIGHT, RIGHT_TO_LEFT)
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            assert rr == 0 or int(g[rr - 1, cc]) != int(BG), "input water must be supported from above"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr - 1 >= 0 and int(g[rr - 1, cc]) == int(BG):
                rr -= 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr - 1 >= 0
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr - 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr - 1), int(cc - 1)))
            if (
                rr - 1 >= 0
                and cc + 1 < w
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr - 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr - 1), int(cc + 1)))
            if opts:
                if len(opts) == 2 and overflow_dir is not None:
                    filtered = [pos for pos in opts if int((pos[1] - cc) * overflow_dir) > 0]
                    if len(filtered) == 1:
                        opts = filtered
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr - 1 >= 0 and int(g[rr - 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == int(BG):
                    right += 1

                rises = [x for x in range(left, right + 1) if int(g[rr - 1, x]) == int(BG)]
                if not rises:
                    break

                pool = list(rises)
                if overflow_dir is not None:
                    pool = [x for x in rises if int((x - cc) * overflow_dir) > 0]
                    if not pool:
                        break

                min_dist = min(abs(int(x - cc)) for x in pool)
                nearest = [int(x) for x in pool if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous rise columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                cc += -1 if best < cc else 1
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(h * w):
        moved = False
        pts = sorted((int(r), int(c)) for r, c in np.argwhere(g == WATER))
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        assert int(np.sum(g == WATER)) == initial_water, "water count must be conserved"
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            g[rr, cc] = int(BG)
            nxt = settle(rr, cc)
            g[rr, cc] = int(WATER)
            assert nxt == (rr, cc), "output water must already be a fixed point"

    return g


def _behavior_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    direction = _detect_overflow_direction(inp)
    baseline = _simulate(inp, overflow_dir=None, check_input=False, check_output=False)
    norm_out = _canonicalize(out, direction)
    norm_base = _canonicalize(baseline, direction)

    return {
        "direction": int(direction),
        "water_count": int(np.sum(inp == WATER)),
        "source_cup": int(np.sum(norm_out[:, 4:6] == WATER)),
        "allowed_cup": int(np.sum(norm_out[:, 7:9] == WATER)),
        "forbidden_cup": int(np.sum(norm_out[:, 1:3] == WATER)),
        "baseline_allowed_cup": int(np.sum(norm_base[:, 7:9] == WATER)),
        "baseline_forbidden_cup": int(np.sum(norm_base[:, 1:3] == WATER)),
        "diff_cells": int(np.sum(out != baseline)),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate upward-flowing water that fills inverted ceiling basins and can overflow only one way.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors form top-anchored inverted basins plus one lower launch roof touching exactly
      one side edge; that lower side cue reveals whether overflow is left-to-right or right-to-left.
    - Water begins packed in a supported bottom reservoir under that launch roof, and may also
      include already-stable water hanging inside the inverted basin chain.

    Rule:
    - Each water pixel rises straight upward through background cells.
    - If blocked above, it may move diagonally upward through an open side gap; when both mirrored
      diagonals are available, the global overflow direction selects the allowed one.
    - When hanging from a roof or another water cell, it slides horizontally only toward rise
      openings on the allowed overflow side, even if the opposite side offers the easier escape.
    - Repeated global passes continue until every water pixel is at a fixed point.

    Key insight: This is not just the original basin puzzle flipped vertically. The solver must
    also infer a one-way spill rule from the lower source orientation, so an apparently easier
    wrong-way overflow path is intentionally ignored.
    """
    overflow_dir = _detect_overflow_direction(np.array(grid, dtype=int))
    return _simulate(
        grid,
        overflow_dir=overflow_dir,
        check_input=check_input,
        check_output=check_output,
    )


def _make_example(
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    profile = dict(rng.choice(PROFILES))
    direction = rng.choice((LEFT_TO_RIGHT, RIGHT_TO_LEFT))
    colors = tuple(rng.sample(list(PALETTE), 3))

    inp = _build_input(colors, profile, direction)
    out = main(inp)
    metrics = _behavior_metrics(inp, out)

    assert np.array_equal(main(out), out), "solver output must already be stable"
    assert metrics["allowed_cup"] > 0, "accepted samples must visibly overflow toward the allowed side"
    assert metrics["forbidden_cup"] == 0, "accepted samples must keep the forbidden side empty"
    assert metrics["baseline_forbidden_cup"] > 0, "baseline must visibly disagree on overflow side"
    assert metrics["baseline_allowed_cup"] == 0, "baseline should prefer the shorter wrong-way divider"
    assert metrics["diff_cells"] >= 8, "fusion must differ substantially from pure reverse gravity"

    return inp, out, metrics


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        inp, out, metrics = _make_example(rng)
        if not np.array_equal(main(inp), out):
            continue
        if metrics["allowed_cup"] <= 0 or metrics["forbidden_cup"] != 0:
            continue
        return inp
    raise RuntimeError("failed to generate a valid reverse-gravity one-way-overflow instance")
