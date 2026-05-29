# Combo variant of 28a6681f (index 27)
# combo_spec: D11/0+D17/0
# Name: 从上往下处理+全局同步模拟
#
# Design analysis
# ---------------
# a) Concrete fused behavior and traced example:
# - I restrict generation to a clean two-basin family: two width-2 basins sit at the
#   bottom, separated by a shorter middle wall, and a supported 3-cell side shelf holds
#   a compact 3x3 source block above the left or right edge.
# - Original task behavior: water is processed one pixel at a time from bottom to top, so
#   the source side basin settles deeply before later pixels react; with 8 source pixels
#   on the left, the original solver finishes as basin counts (8, 0).
# - Fused behavior here: each round has ordered global phases (down, then diagonal, then
#   horizontal), and within each phase rows are processed from top to bottom while all
#   water in the same row moves simultaneously. Because the row group sees the grid after
#   higher rows already moved in that phase, the motion is layered rather than fully
#   settled. On the same 8-pixel source, the fused solver yields (5, 3): part of the
#   upper front reaches the lowered divider before the near basin has compacted.
# - This is the intended visible signature: instead of “fill the near basin completely,
#   then overflow later”, the output shows premature spill caused by row-synchronous
#   layering.
#
# b) Complexity calibration:
# - The geometry is deliberately narrow: always two adjacent basins, always a shorter
#   divider, always one side source, and water counts only in {5, 7, 8}. This keeps the
#   rule readable from 2-4 ARC examples.
# - I still vary the source side, outer wall heights (4 or 5), and structure colors so
#   the task does not collapse to a single hard-coded picture.
# - Width-2 basins make the difference legible: a few pixels spilling into the neighbor
#   basin are obvious by eye, whereas on wide basins the same rule would be harder to
#   notice.
#
# c) Implicit-assumption audit against the original solver:
# - Original `settle()` assumes one pixel can run all the way to its final resting place
#   before the next pixel is considered. D11/0 explicitly breaks that assumption, so the
#   settle-to-fixed-point helper must be replaced by a round-based simulator.
# - Original outer-loop ordering is bottom-up reverse sort. D17/0 inverts that within each
#   simulation phase, so lower water can now be blocked by water that started above it.
# - Original output validation assumed every water cell ended with non-background on the
#   left, below, and right. That is too strong after global synchronization: a bottom-row
#   water cell may legally keep background on one side if no supported horizontal move
#   remains. The new validation therefore checks “no fused round can move anything”
#   instead of the old local side-contact condition.
# - Original code asserted on per-pixel diagonal/horizontal ambiguity. In the fused
#   simulator, symmetric options can arise dynamically even from safe inputs. I resolve
#   those by not moving that pixel in that phase; otherwise the variant would reject many
#   valid layered states.
#
# d) Placement legality:
# - The generator only uses wall columns (1,4,7) or its mirror (2,5,8), with tops
#   (4 or 5, 6, 4 or 5). These are bottom-anchored, non-overlapping, and stay inside the
#   10x10 grid.
# - The side shelf is fixed to row 3, so it never overlaps the basin walls. Source water
#   occupies a bottom-up prefix of the 3x3 block above that shelf, which guarantees every
#   input water pixel is supported and no source pixel has background on both horizontal
#   sides.
#
# e) Visual observability:
# - In accepted samples, the legacy solver fills only the source-side basin, while the
#   fused solver activates both basins. This makes the combo behavior visible without any
#   hidden metadata or color coding.
# - Left-source and right-source mirrored cases show the same structural idea, helping a
#   human infer that the governing rule is about simulation order rather than about a
#   particular color or absolute position.
#
# f) Visual cleanliness:
# - There are only three object types on screen: static walls/shelf, blue water, and black
#   background. No guides, markers, or overlapping annotations are added.
# - The upper source area and lower basin area are separated by empty rows, so the input
#   remains easy to parse.
#
# g) Change inferability:
# - The variant does not rely on invisible properties such as numeric color ordering.
#   The only changed mechanism is how identical blue pixels move relative to one another.
# - Because the middle wall is visibly shorter, the resulting early spill into the second
#   basin is a human-observable consequence of the new dynamics.
#
# h) Non-degeneration metrics used by the generator:
# - `gravity_gap`: pixel count where fused output differs from the original bottom-up
#   settle solver. Accepted samples require `gravity_gap >= 4`.
# - `basin_l1_gap`: L1 difference between fused and legacy basin water-count vectors.
#   Accepted samples require `basin_l1_gap >= 2`.
# - `active_basin_shift`: the legacy solver must activate exactly one basin while the fused
#   solver activates exactly two. This enforces the intended “layered premature overflow”
#   signature instead of a cosmetic rearrangement.
#
# i) New solver assumptions and generator guarantees:
# - `main()` assumes a standard 10x10 ARC grid with water color 1 and background 0.
# - `main()` further assumes only that non-water cells are rigid obstacles; it does not
#   need the generator’s exact wall palette.
# - `generate_input()` guarantees the source-water support condition, the lowered-divider
#   basin geometry, and the non-degeneration metrics above, so the fused behavior is
#   present rather than accidentally collapsing back to the source task.

from __future__ import annotations

import random

import numpy as np


BG = 0
WATER = 1
H = 10
W = 10
OUTER_TOP_CHOICES = (4, 5)
MIDDLE_TOP = 6
SHELF_ROW = 3
SHELF_WIDTH = 3
WATER_COUNT_CHOICES = (5, 7, 8)
SOLID_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
PHASES = ("down", "diag", "horiz")


def _source_columns(source_side: str) -> list[int]:
    return [0, 1, 2] if source_side == "left" else [7, 8, 9]


def _source_cells(source_side: str) -> list[tuple[int, int]]:
    cols = [0, 1, 2] if source_side == "left" else [9, 8, 7]
    cells: list[tuple[int, int]] = []
    for r in range(SHELF_ROW - 1, -1, -1):
        for c in cols:
            cells.append((int(r), int(c)))
    return cells


def _wall_columns(source_side: str) -> tuple[int, int, int]:
    return (1, 4, 7) if source_side == "left" else (2, 5, 8)


def _phase_target(grid: np.ndarray, r: int, c: int, phase: str) -> tuple[int, int] | None:
    rr = int(r)
    cc = int(c)

    if phase == "down":
        if rr + 1 < H and int(grid[rr + 1, cc]) == int(BG):
            return int(rr + 1), int(cc)
        return None

    if phase == "diag":
        if rr + 1 >= H or int(grid[rr + 1, cc]) == int(BG):
            return None
        opts: list[tuple[int, int]] = []
        if (
            cc - 1 >= 0
            and int(grid[rr, cc - 1]) == int(BG)
            and int(grid[rr + 1, cc - 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc - 1)))
        if (
            cc + 1 < W
            and int(grid[rr, cc + 1]) == int(BG)
            and int(grid[rr + 1, cc + 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc + 1)))
        return opts[0] if len(opts) == 1 else None

    if phase == "horiz":
        if rr + 1 >= H or int(grid[rr + 1, cc]) == int(BG):
            return None

        left = int(cc)
        while left - 1 >= 0 and int(grid[rr, left - 1]) == int(BG):
            left -= 1
        right = int(cc)
        while right + 1 < W and int(grid[rr, right + 1]) == int(BG):
            right += 1

        drops = [int(x) for x in range(int(left), int(right) + 1) if int(grid[rr + 1, int(x)]) == int(BG)]
        if not drops:
            return None

        min_dist = min(abs(int(x - cc)) for x in drops)
        nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
        if len(nearest) != 1:
            return None

        best = int(nearest[0])
        if best == int(cc):
            return None
        step = -1 if best < int(cc) else 1
        nx = int(cc + step)
        if int(grid[rr, nx]) != int(BG):
            return None
        return int(rr), int(nx)

    raise ValueError(f"unknown phase: {phase}")


def _apply_phase(grid: np.ndarray, phase: str) -> bool:
    moved = False
    row_groups: dict[int, list[tuple[int, int]]] = {int(r): [] for r in range(H)}
    for r, c in np.argwhere(grid == int(WATER)):
        row_groups[int(r)].append((int(r), int(c)))

    for r in range(H):
        coords = sorted(row_groups[int(r)])
        if not coords:
            continue

        snapshot = np.array(grid, dtype=int).copy()
        claims: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for rr, cc in coords:
            tgt = _phase_target(snapshot, int(rr), int(cc), phase)
            if tgt is not None:
                claims.setdefault((int(tgt[0]), int(tgt[1])), []).append((int(rr), int(cc)))

        valid_moves = [
            ((int(srcs[0][0]), int(srcs[0][1])), (int(tgt[0]), int(tgt[1])))
            for tgt, srcs in claims.items()
            if len(srcs) == 1
        ]
        if not valid_moves:
            continue

        moved = True
        for (sr, sc), _ in valid_moves:
            grid[int(sr), int(sc)] = int(BG)
        for _, (tr, tc) in valid_moves:
            grid[int(tr), int(tc)] = int(WATER)

    return moved


def _run_round(grid: np.ndarray) -> tuple[np.ndarray, bool]:
    g = np.array(grid, dtype=int).copy()
    moved = False
    for phase in PHASES:
        moved = bool(_apply_phase(g, phase) or moved)
    return g, bool(moved)


def _legacy_main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()

    if check_input:
        assert g.shape == (H, W)
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

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
                best = int(nearest[0])
                assert best != int(cc)
                cc += -1 if best < int(cc) else 1
                continue

            break
        else:
            raise AssertionError("legacy water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("legacy water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g


def _basin_counts(grid: np.ndarray) -> tuple[int, int]:
    walls = [int(c) for c in range(W) if int(grid[H - 1, c]) not in (int(BG), int(WATER))]
    assert walls in ([1, 4, 7], [2, 5, 8]), f"unexpected wall columns: {walls}"
    counts = [int(np.sum(grid[:, int(left) + 1 : int(right)] == int(WATER))) for left, right in zip(walls, walls[1:])]
    assert len(counts) == 2
    return int(counts[0]), int(counts[1])


def _active_basin_count(counts: tuple[int, int]) -> int:
    return int(sum(int(x > 0) for x in counts))


def _draw_case(
    rng: random.Random,
    source_side: str,
    outer_top_left: int,
    outer_top_right: int,
    water_count: int,
) -> np.ndarray:
    wall_cols = _wall_columns(source_side)
    wall_tops = (int(outer_top_left), int(MIDDLE_TOP), int(outer_top_right))
    outer_color, middle_color, shelf_color = rng.sample(list(SOLID_POOL), 3)

    g = np.zeros((H, W), dtype=int)
    wall_colors = (int(outer_color), int(middle_color), int(outer_color))
    for c, top, color in zip(wall_cols, wall_tops, wall_colors):
        g[int(top) :, int(c)] = int(color)

    for c in _source_columns(source_side):
        g[int(SHELF_ROW), int(c)] = int(shelf_color)

    cells = _source_cells(source_side)
    assert 1 <= int(water_count) <= len(cells)
    for r, c in cells[: int(water_count)]:
        assert int(g[int(r), int(c)]) == int(BG)
        g[int(r), int(c)] = int(WATER)

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-filling water with row-synchronous global rounds instead of letting each pixel settle to completion on its own.

    Input structure:
    - Background is 0 and moving water is color 1.
    - Other colors are rigid obstacles: bottom-anchored basin walls plus an upper support shelf for the source water.
    - Water starts in supported edge reservoirs above the basin chain; it is not already settled inside the lower basins.

    Rule:
    - Repeat global rounds until no water moves.
    - Each round has three ordered phases: one-cell downward motion, then one-cell diagonal slipping, then one-cell supported horizontal sliding toward a unique reachable drop.
    - Inside each phase, collect the water pixels currently present in that phase, process rows from top to bottom, and move all pixels from the same row simultaneously; if two same-row pixels claim the same destination, neither one moves there in that phase.
    - Because higher rows act first in each phase and the grid is updated between rows, upper water can occupy or block cells before lower water gets its turn.

    Key insight:
    - The fusion is not just “top-down” or just “global” in isolation. Water now evolves as layered fronts: every round updates many pixels together, but the update itself has a visible row priority, which changes when overflow into the neighboring basin begins.
    """
    g = np.array(grid, dtype=int).copy()
    assert g.shape == (H, W), f"expected a {H}x{W} grid"

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    structure = np.array(g, dtype=int).copy()
    structure[structure == int(WATER)] = int(BG)
    input_water = int(np.count_nonzero(g == int(WATER)))

    for _round in range(H * W * 4):
        g, moved = _run_round(g)
        if not moved:
            break
    else:
        raise AssertionError("fused water simulation did not converge")

    if check_output:
        out_structure = np.array(g, dtype=int).copy()
        out_structure[out_structure == int(WATER)] = int(BG)
        assert np.array_equal(out_structure, structure), "structure pixels must stay unchanged"
        assert int(np.count_nonzero(g == int(WATER))) == int(input_water), "water count must be conserved"
        _probe, probe_moved = _run_round(g)
        assert not probe_moved, "output must be a fused-simulation fixed point"

    return g


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = rng or random.Random()

    for _ in range(200):
        source_side = rng.choice(["left", "right"])
        outer_top_left = rng.choice(OUTER_TOP_CHOICES)
        outer_top_right = rng.choice(OUTER_TOP_CHOICES)
        water_count = rng.choice(WATER_COUNT_CHOICES)

        grid = _draw_case(
            rng,
            source_side=source_side,
            outer_top_left=int(outer_top_left),
            outer_top_right=int(outer_top_right),
            water_count=int(water_count),
        )

        try:
            fused = main(grid)
            legacy = _legacy_main(grid, check_output=False)
        except AssertionError:
            continue

        gravity_gap = int(np.count_nonzero(fused != legacy))
        fused_counts = _basin_counts(fused)
        legacy_counts = _basin_counts(legacy)
        basin_l1_gap = int(sum(abs(int(a) - int(b)) for a, b in zip(fused_counts, legacy_counts)))
        fused_active = int(_active_basin_count(fused_counts))
        legacy_active = int(_active_basin_count(legacy_counts))

        if gravity_gap < 4:
            continue
        if basin_l1_gap < 2:
            continue
        if legacy_active != 1 or fused_active != 2:
            continue

        return grid

    raise AssertionError("failed to generate a non-degenerate D11/0+D17/0 sample")
