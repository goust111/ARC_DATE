# Variant of 28a6681f
# Dimension: D6 — 终止条件/稳定判据 (dst)
# Change: 水沉入盆地最低可达位置——输出中每个水像素的左、下、右都为非背景或网格边缘... → 水位高度限制

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D6 variant
# ----------------------------------
# a) Concrete form and behavior trace
#    - The base scene stays close to the source task: a 10x10 grid, blue water (1),
#      bottom-anchored solid walls, and an upper supported source that pours water into
#      adjacent basins.
#    - The new visible control is a pair of marker pixels in color 8 placed on the two
#      outer basin walls at the same row. That shared row is the waterline.
#    - Geometry is a two-basin chain: outer left wall at column 2, divider at column 4,
#      outer right wall at column 7. The divider can be short enough to allow overflow
#      or tall enough to keep both basins separate. This preserves the original
#      “connected basin / possible spill” flavor.
#    - Source water sits as a supported block on a left or right shelf. Under the
#      original rule, once lower cells fill, extra water would keep stacking into basin
#      cells above the marker row. In the variant, a moved water pixel is not allowed to
#      come to rest above the marker row, so that extra water stays on the source shelf.
#    - Small trace:
#         source on the right, waterline row = 6, divider short enough to overflow.
#         Original logic: source water drips in, fills rows 8 and 7, then continues to
#         occupy row 5 inside the basin chain.
#         Variant logic: the same water can still fall and slide, but if its best resting
#         place would be row 5 (< 6), it is rejected and stays at its original supported
#         source cell. The basin therefore stops at the marked height.
#
# b) Complexity calibration
#    - Only one new symbol is introduced: marker color 8.
#    - All marker pixels share one row, so the task does not require hidden ordering or
#      basin-specific color mapping.
#    - The basin network stays small: a narrow basin beside a wider basin, separated by
#      one central wall whose top controls visible overflow.
#      This is enough to make the waterline visible while keeping the rule inferable from
#      a few examples.
#
# c) Original-code assumptions that no longer hold
#    - Original settle() assumed any supported background cell could be a valid terminal
#      resting place. That is false now: supported cells above the waterline are illegal
#      new resting places.
#    - Original check_output asserted “left/down/right blocked” for every output water
#      pixel. That also fails now because excess water may remain stably on the source
#      shelf with open air on one side. The new invariant is instead a fixed point:
#      removing any output water pixel and re-settling it under the new rule must return
#      it to the same cell.
#    - Input-side support and non-ambiguous local motion remain useful and are preserved.
#
# d) Placement legality
#    - The bottom structure uses a fixed 2/4/7 wall pattern. This asymmetric layout is
#      deliberate: it avoids double-diagonal ambiguity while still allowing spill from one
#      basin into the other when the center wall is short.
#    - The template family is pre-validated so the source block, shelf, and walls remain
#      readable after optional mirroring.
#    - The marker row is always below the upper source and inside both outer walls.
#    - The generator retries until the produced input passes main() and the limited and
#      unlimited simulations differ.
#
# e) Visual observability
#    - The user can see the same basin family across samples, but the marker row changes.
#      In some cases the line is low, so much water stays on the source shelf; in others
#      the line is higher and the basins fill further or overflow.
#    - The marker color is constant and structurally attached to the basin walls, so the
#      “fill only up to this height” pattern is visible without any hidden metadata.
#
# f) Visual cleanliness
#    - There are only four functional layers: background, solids, blue water, and the
#      marker color. The marker reuses wall cells rather than floating in open air, so it
#      reads as a level mark instead of a new obstacle type.
#
# g) Color inferability
#    - Water is always 1 and the waterline is always 8. Solid wall colors vary among a
#      small palette, but they do not encode behavior. No color-id sorting or hidden
#      numeric rule is used.
#
# h) Generator validation / non-degeneration metric
#    - A sample is accepted only if:
#         1. main(input) changes the grid,
#         2. the limited and unlimited simulations differ, and
#         3. the limited simulation leaves strictly more water inside the source box than
#            the unlimited one.
#      This directly measures whether the waterline is actively constraining filling.
#
# i) New implicit assumptions and how generate_input() guarantees them
#    - Assumption: all marker pixels lie on one row. Guaranteed by placing exactly two
#      markers at (level, 2) and (level, 7), then optionally mirroring the whole layout.
#    - Assumption: source cells are already supported, so rejecting a move by returning a
#      pixel to its start position still leaves a legal output. Guaranteed by building the
#      source as a compact block on a shelf at the grid edge.
#    - Assumption: the shelf opens toward a unique basin entrance. Guaranteed by fixed
#      shelf endpoints and a single open side.


BG = 0
WATER = 1
MARKER = 8
H = 10
W = 10


def _marker_row(grid: np.ndarray) -> int | None:
    rows = {int(r) for r, _c in np.argwhere(grid == MARKER)}
    if not rows:
        return None
    if len(rows) != 1:
        raise AssertionError("waterline markers must share a single row")
    return next(iter(rows))


def _simulate(grid: np.ndarray, limit_row: int | None, check_output: bool) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()

    def settle(r: int, c: int) -> tuple[int, int]:
        start = (int(r), int(c))
        rr, cc = start
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == BG:
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == BG
                and int(g[rr + 1, cc - 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == BG
                and int(g[rr + 1, cc + 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != BG:
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == BG:
                    right += 1

                drops = [x for x in range(left, right + 1) if int(g[rr + 1, x]) == BG]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == BG
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")

        if limit_row is not None and rr < limit_row and (rr, cc) != start:
            return start
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == WATER)], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != WATER:
                continue
            g[r, c] = BG
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = WATER
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == WATER)]
        for r, c in pts:
            g[r, c] = BG
            rr, cc = settle(r, c)
            g[r, c] = WATER
            assert (rr, cc) == (r, c), "output water is not stable under the waterline rule"

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-filling water that obeys gravity but stops at a visible wall marker height.

    Input structure:
    - Background is 0 and water is 1.
    - Other nonzero colors are fixed solids; color 8 is a special waterline marker painted on wall cells.
    - Water starts as supported source blocks on an upper shelf and/or as already settled basin water.

    Rule:
    - Water still falls straight down, then uses the same diagonal slip and supported horizontal search as in the source task.
    - The marker row gives a maximum fill height for the basin system: a water pixel may move through higher cells while falling, but it may not finish in a new resting place above the marker row.
    - If the best resting place would be above the marked level, that pixel stays at its original supported source cell instead.

    Key insight:
    - The motion physics are unchanged; the only new mechanism is that the colored wall markers convert “deepest reachable cell” into “deepest reachable cell that does not raise the water above the visible line”.
    """
    g = np.array(grid, dtype=int).copy()
    limit_row = _marker_row(g)

    if check_input:
        assert g.shape == (H, W), "grid must be 10x10"
        assert limit_row is not None, "input must contain a waterline marker"
        markers = [(int(r), int(c)) for r, c in np.argwhere(g == MARKER)]
        assert len(markers) >= 2, "waterline should be visible on at least two wall cells"
        for r, c in markers:
            adjacent_bg = (
                (r > 0 and int(g[r - 1, c]) == BG)
                or (r + 1 < H and int(g[r + 1, c]) == BG)
                or (c > 0 and int(g[r, c - 1]) == BG)
                or (c + 1 < W and int(g[r, c + 1]) == BG)
            )
            assert adjacent_bg, "marker should touch open space"

        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != BG, "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == BG
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    return _simulate(g, limit_row=limit_row, check_output=check_output)


def _source_box(side: str, floor_row: int, height: int) -> tuple[range, range]:
    rows = range(floor_row - height, floor_row)
    if side == "left":
        return rows, range(0, 3)
    return rows, range(7, 10)


def _count_water(grid: np.ndarray, rows: range, cols: range) -> int:
    return sum(int(grid[r, c] == WATER) for r in rows for c in cols)


def _build_candidate(rng: random.Random, want_overflow: bool) -> tuple[np.ndarray, dict[str, int | str]]:
    g = np.zeros((H, W), dtype=int)
    wall_colors = rng.sample([2, 3, 4, 5, 6, 7, 9], 3)
    left_color, divider_color, right_color = wall_colors

    left_wall = 2
    divider = 4
    right_wall = 7
    source_side = rng.choice(["left", "right"])
    if want_overflow:
        template = rng.choice(
            [
                {"shelf_row": 3, "source_height": 3, "level_row": 7, "divider_top": 8},
                {"shelf_row": 4, "source_height": 3, "level_row": 7, "divider_top": 8},
                {"shelf_row": 4, "source_height": 4, "level_row": 6, "divider_top": 7},
                {"shelf_row": 4, "source_height": 4, "level_row": 7, "divider_top": 8},
            ]
        )
    else:
        template = rng.choice(
            [
                {"shelf_row": 3, "source_height": 2, "level_row": 5, "divider_top": 3},
                {"shelf_row": 3, "source_height": 2, "level_row": 6, "divider_top": 4},
                {"shelf_row": 4, "source_height": 3, "level_row": 6, "divider_top": 4},
                {"shelf_row": 4, "source_height": 4, "level_row": 7, "divider_top": 6},
            ]
        )

    shelf_row = int(template["shelf_row"])
    source_height = int(template["source_height"])
    level_row = int(template["level_row"])
    divider_top = int(template["divider_top"])
    left_top = rng.choice([2, 3, 4])
    right_top = rng.choice([2, 3, 4])

    g[9, left_wall : divider + 1] = left_color
    g[9, divider : right_wall + 1] = right_color
    g[left_top:10, left_wall] = left_color
    g[divider_top:10, divider] = divider_color
    g[right_top:10, right_wall] = right_color

    g[level_row, left_wall] = MARKER
    g[level_row, right_wall] = MARKER

    g[shelf_row, 0:3] = divider_color
    g[shelf_row - source_height : shelf_row, 0:3] = WATER

    if source_side == "right":
        g = np.fliplr(g)

    meta: dict[str, int | str] = {
        "source_side": source_side,
        "shelf_row": shelf_row,
        "source_height": source_height,
        "level_row": level_row,
        "overflow": int(want_overflow),
    }
    return g, meta


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        want_overflow = bool(rng.getrandbits(1))
        inp, meta = _build_candidate(rng, want_overflow=want_overflow)
        try:
            out = main(inp, check_input=True, check_output=True)
            base_out = _simulate(inp, limit_row=None, check_output=True)
        except AssertionError:
            continue

        if np.array_equal(out, inp):
            continue
        if np.array_equal(out, base_out):
            continue

        rows, cols = _source_box(
            str(meta["source_side"]),
            int(meta["shelf_row"]),
            int(meta["source_height"]),
        )
        limited_source = _count_water(out, rows, cols)
        unlimited_source = _count_water(base_out, rows, cols)
        if limited_source <= unlimited_source:
            continue

        has_left_basin = bool(np.any(out[5:, 3:5] == WATER))
        has_right_basin = bool(np.any(out[5:, 5:7] == WATER))
        if want_overflow != (has_left_basin and has_right_basin):
            continue

        return inp

    raise RuntimeError("failed to generate a non-degenerate waterline sample")
