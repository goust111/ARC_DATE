# Combo variant of 28a6681f (index 23)
# combo_spec: D5/2+D14/2
# Name: 控制像素水量+水龙头释放

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D5/2 + D14/2
# ---------------------------------------
# a) Concretization + behavior trace:
#    - The grid stays 10x10 with background 0, water 1, and rigid basin walls in 2-5.
#      Input water is completely hidden. Every sample instead shows:
#        * one fixed 2-cell faucet bar in color 8 on the top of an upper ledge;
#        * two edge control pixels in colors 6/7/9 on the far side of the top edge.
#    - The faucet bar gives only the outlet location: after deleting the bar, water is emitted
#      from its unique open-end cell. The control pixels give only the total quantity via the
#      fixed map 6 -> 3, 7 -> 5, 9 -> 7, so two controls encode totals 10/12/14.
#    - Concrete trace on the right-shelf two-basin template:
#        input: controls [7,9] on the far-left top edge, faucet bar 88 on the far-right top edge.
#        original task:
#          * sees no visible blue source at all, so it would leave the grid unchanged.
#        D14-only style:
#          * might release only 2 drops because the bar length is 2, badly underfilling the basins.
#        D5-only style:
#          * might treat the edge markers themselves as source columns, so the water would drain
#            from the wrong side of the map and settle in a different reservoir.
#        combo:
#          * decode [7,9] -> 12 hidden drops;
#          * decode the bar geometry -> release from the right shelf outlet;
#          * settle those 12 drops with the original fall / unique diagonal / horizontal-search
#            physics until the near basin fills and spills into the neighbor.
#      The quantity and the release position are therefore both indirect and both necessary.
#
# b) Complexity calibration:
#    - The quantity code is intentionally tiny: only three control colors and only totals
#      10/12/14 in generated examples. That gives repeated support for the same color semantics.
#    - The faucet bar length is fixed to 2 in every example so humans cannot confuse bar length
#      with quantity. Its only job is to mark the outlet side.
#    - Guide-shaped templates were intentionally dropped from the generator: in those layouts many
#      different top-edge release cells collapse to the same path, which makes the faucet position
#      visually redundant. Restricting the family to no-guide basin shelves keeps the fusion signal
#      observable instead of accidentally degenerating into a quantity-only task.
#
# c) Hidden-assumption audit of the original main():
#    - The original input checks assumed movable blue source pixels already existed. That is false
#      here, so input validation must move to "controls are legal" and "the faucet geometry is legal".
#    - The original settling loop iterated over `g == water` immediately. Here that would produce an
#      empty set and return the input unchanged, so the solver must first reconstruct hidden water.
#    - The original task could scan upper colored objects as possible guides. That assumption is no
#      longer useful because the upper colored markers now include a faucet-support shelf and the tap.
#      The combo solver therefore keeps only the water physics and removes guide-shape validation.
#    - The original generator also relied on a stronger output-shape invariant: every final water
#      cell had non-background to its left, below, and right. Faucet release on upper shelves can
#      create wider terrace fixed points that are still immobile under the same physics, so the
#      combo solver weakens output checking to "rerunning the dynamics makes no further change".
#    - Water conservation changes from "visible input water count is preserved" to
#      "final water count equals the decoded control total", not the faucet length.
#
# d) Placement legality:
#    - Each template is a hand-validated 10x10 basin layout with an extra top shelf that supports
#      the faucet bar. Controls occupy pre-cleared top-edge cells far from the faucet.
#    - Mirroring is applied to the whole template, so tap cells, alternative-control-side source
#      cells, and control slots all stay in-bounds and non-overlapping.
#    - The generator validates every candidate by running the new solver; blocked source mouths,
#      ambiguous slips, or illegal marker overlaps are rejected.
#
# e) Visual observability:
#    - Inputs show two distinct marker systems:
#        * same-color faucet bars always sit on a shelf and disappear;
#        * edge control pixels stay visible and correlate with the total amount of blue output.
#    - The generator rejects any sample whose combo output matches either of two ablations:
#        * `tap_only`: emit only 2 drops from the faucet as if bar length were the quantity;
#        * `control_side`: emit the full decoded amount from the control side instead of the faucet.
#      This guarantees both hidden dimensions matter in accepted samples.
#
# f) Visual cleanliness:
#    - Controls and faucet never overlap and occupy separate ends of the same top row.
#    - Structure colors are limited to 2-5, controls to 6/7/9, faucet to 8, and water to 1.
#      That keeps semantic roles visually distinct without overloading the palette.
#
# g) Change inferability / color discipline:
#    - No rule depends on numeric color ordering. Control colors are fixed semantic labels with
#      repeated cross-example support, and all other colors remain purely structural.
#    - Because the faucet length never changes while output water totals do change, humans can infer
#      that quantity comes from the edge controls rather than from the faucet itself.
#
# h) Fusion metrics used by the generator:
#    - `decoded_total`: sum of control amounts; must be one of {10, 12, 14}.
#    - `delta_vs_tap_only`: combo core output must differ from emitting only 2 drops from the faucet.
#    - `delta_vs_control_side`: combo core output must differ from emitting the same decoded total
#      from a top-edge source on the control side instead of from the faucet outlet.
#
# i) New solver assumptions guaranteed by generate_input():
#    - Exactly one horizontal 2-cell faucet bar exists.
#    - There are exactly two control pixels, both on the top edge and disjoint from the faucet.
#    - Clearing the controls and the faucet leaves an empty outlet cell before every injection.
#    - The control-side ablation source cell is also safe, so the generator can measure whether the
#      faucet position truly changes the result.


BG = 0
WATER = 1
TAP = 8
CONTROL_TO_AMOUNT = {6: 3, 7: 5, 9: 7}
CONTROL_COLORS = tuple(sorted(CONTROL_TO_AMOUNT))
STRUCTURE_COLORS = (2, 3, 4, 5)
GRID_SHAPE = (10, 10)
TAP_LENGTH = 2
MAX_GENERATION_ATTEMPTS = 400


def _grid_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _simulate_visible_water(
    grid: np.ndarray,
    *,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"
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

            opts: list[tuple[int, int]] = []
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
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
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

    return g


def _parse_controls(grid: np.ndarray, *, check_input: bool) -> list[tuple[int, int, int]]:
    g = np.array(grid, dtype=int)
    controls: list[tuple[int, int, int]] = []

    for color in CONTROL_COLORS:
        for r, c in np.argwhere(g == int(color)):
            controls.append((int(r), int(c), int(color)))

    controls.sort(key=lambda x: (x[0], x[1]))

    if check_input:
        assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"
        assert not np.any(g == int(WATER)), "variant input must not contain visible water"
        assert 1 <= len(controls) <= 2, "variant expects one or two control pixels"
        for r, c, color in controls:
            assert r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1), "control must sit on the grid edge"
            assert int(g[r, c]) == int(color)
        for color in CONTROL_COLORS:
            for r, c in np.argwhere(g == int(color)):
                rr = int(r)
                cc = int(c)
                assert rr in (0, g.shape[0] - 1) or cc in (0, g.shape[1] - 1), "control color may only appear on edges"

    return controls


def _parse_faucet(grid: np.ndarray, *, check_input: bool) -> tuple[int, int]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    tap_pts = np.argwhere(g == int(TAP))

    if check_input:
        assert len(tap_pts) > 0, "input must contain a faucet bar"

    rows = sorted({int(r) for r, _c in tap_pts})
    assert len(rows) == 1, "faucet must be a single horizontal row"
    row = rows[0]
    cols = sorted(int(c) for _r, c in tap_pts)
    left = cols[0]
    right = cols[-1]
    assert cols == list(range(int(left), int(right) + 1)), "faucet cells must form one contiguous run"
    assert len(cols) == TAP_LENGTH, f"faucet length must be {TAP_LENGTH}"

    if check_input:
        for c in range(int(left), int(right) + 1):
            assert row + 1 < H, "faucet cannot sit on the bottom row"
            below = int(g[row + 1, int(c)])
            assert below not in {int(BG), int(WATER), int(TAP)} | set(CONTROL_COLORS), "faucet must sit on structure"
            assert row == 0 or int(g[row - 1, int(c)]) == int(BG), "faucet must be exposed from above"

    left_open = left > 0 and int(g[row, left - 1]) == int(BG)
    right_open = right + 1 < W and int(g[row, right + 1]) == int(BG)
    assert left_open != right_open, "faucet must have exactly one open side"
    emit_c = int(left if left_open else right)
    return int(row), int(emit_c)


def _decoded_total(controls: list[tuple[int, int, int]]) -> int:
    return int(sum(int(CONTROL_TO_AMOUNT[color]) for _r, _c, color in controls))


def _cleared_sim_grid(
    grid: np.ndarray,
    controls: list[tuple[int, int, int]],
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    for r, c, _color in controls:
        g[int(r), int(c)] = int(BG)
    g[g == int(TAP)] = int(BG)
    return g


def _pour_hidden_water(
    cleared_grid: np.ndarray,
    *,
    source_cell: tuple[int, int],
    amount: int,
) -> np.ndarray:
    g = np.array(cleared_grid, dtype=int).copy()
    src_r, src_c = source_cell

    for _ in range(int(amount)):
        assert int(g[int(src_r), int(src_c)]) == int(BG), "source mouth became blocked"
        g[int(src_r), int(src_c)] = int(WATER)
        g = _simulate_visible_water(g, check_input=False, check_output=False)

    return g


def _core_output(
    grid: np.ndarray,
    *,
    controls: list[tuple[int, int, int]],
    source_cell: tuple[int, int],
    amount: int,
) -> np.ndarray:
    cleared = _cleared_sim_grid(grid, controls)
    return _pour_hidden_water(cleared, source_cell=source_cell, amount=int(amount))


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Decode edge quantity markers and a faucet outlet, then pour that hidden water through the basin network.

    Input structure:
    - Background is 0 and visible water is absent from the input.
    - One or two edge control pixels in colors 6/7/9 encode hidden water amounts 3/5/7.
    - A single 2-cell faucet bar in color 8 sits on top of an upper structure ledge and points to a unique open side.
    - Other non-zero colors are fixed basin walls, shelves, and dividers.

    Rule:
    - Read all control pixels, sum their encoded amounts, and read the faucet geometry to locate the release cell.
    - Remove the controls and the faucet from the simulated grid, since neither is solid output structure.
    - Inject the decoded number of water pixels one by one from the faucet outlet.
    - After each injection, reuse the original water physics: fall vertically, take the unique open diagonal when blocked, and slide horizontally toward the nearest unique drop.
    - When all hidden water has settled, restore only the control pixels in the final output.

    Key insight:
    - The puzzle now hides both source dimensions at once: the border markers tell you how much unseen water exists, while the faucet bar tells you where that unseen water enters.
    """
    g = np.array(grid, dtype=int).copy()
    assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"

    controls = _parse_controls(g, check_input=bool(check_input))
    faucet_cell = _parse_faucet(g, check_input=bool(check_input))
    total = _decoded_total(controls)

    sim = _core_output(g, controls=controls, source_cell=faucet_cell, amount=int(total))

    if check_output:
        stable = _simulate_visible_water(sim, check_input=False, check_output=False)
        assert np.array_equal(stable, sim), "decoded water must already be stable"
        assert int(np.count_nonzero(sim == int(WATER))) == int(total), "decoded water count must be conserved"
        assert not np.any(sim == int(TAP)), "faucet must disappear in the output"

    out = np.array(sim, dtype=int).copy()
    for r, c, color in controls:
        assert int(out[int(r), int(c)]) == int(BG), "control cell cannot be occupied by settled water"
        out[int(r), int(c)] = int(color)
    return out


def _mirror_template(template: dict[str, object]) -> dict[str, object]:
    mirrored = dict(template)
    mirrored["name"] = f"{template['name']}_mirrored"
    mirrored["grid"] = np.fliplr(np.array(template["grid"], dtype=int))
    mirrored["tap_cells"] = tuple(sorted((int(r), int(GRID_SHAPE[1] - 1 - c)) for r, c in template["tap_cells"]))
    mirrored["emit"] = (int(template["emit"][0]), int(GRID_SHAPE[1] - 1 - int(template["emit"][1])))
    mirrored["alt_emit"] = (int(template["alt_emit"][0]), int(GRID_SHAPE[1] - 1 - int(template["alt_emit"][1])))
    mirrored["control_slots"] = tuple(sorted(int(GRID_SHAPE[1] - 1 - c) for c in template["control_slots"]))
    return mirrored


BASE_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "name": "two_basin_shelf",
        "grid": _grid_from_rows(
            (
                "0000000000",
                "0000000033",
                "0000000300",
                "0000000300",
                "0000000300",
                "0000030300",
                "0000030300",
                "0000230300",
                "0000230300",
                "0222233300",
            )
        ),
        "tap_cells": ((0, 8), (0, 9)),
        "emit": (0, 8),
        "alt_emit": (0, 0),
        "control_slots": (0, 1, 2),
    },
    {
        "name": "stacked_basin_shelf",
        "grid": _grid_from_rows(
            (
                "0000000000",
                "0000000033",
                "0020000000",
                "0020000000",
                "0020000000",
                "0020020030",
                "0020023003",
                "0020023003",
                "0020023003",
                "0022223333",
            )
        ),
        "tap_cells": ((0, 8), (0, 9)),
        "emit": (0, 8),
        "alt_emit": (0, 0),
        "control_slots": (0, 1, 2),
    },
)

ALL_TEMPLATES = BASE_TEMPLATES + tuple(_mirror_template(template) for template in BASE_TEMPLATES)

CONTROL_RECIPES: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((0, 1), (6, 9)),
    ((0, 1), (7, 7)),
    ((0, 2), (7, 9)),
    ((1, 2), (6, 9)),
    ((1, 2), (7, 7)),
    ((1, 2), (7, 9)),
    ((0, 2), (9, 9)),
)


def _recolor_structure(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    source_colors = sorted(int(v) for v in np.unique(g) if int(v) != int(BG))
    target_colors = rng.sample(list(STRUCTURE_COLORS), k=len(source_colors))
    color_map = {int(src): int(dst) for src, dst in zip(source_colors, target_colors)}
    for src, dst in color_map.items():
        g[g == int(src)] = int(dst)
    return g


def _choose_controls(template: dict[str, object], rng: random.Random) -> list[tuple[int, int, int]]:
    slot_indices, colors = rng.choice(CONTROL_RECIPES)
    slots = [int(template["control_slots"][idx]) for idx in slot_indices]
    return [(0, int(c), int(color)) for c, color in zip(slots, colors)]


def _candidate_input(template: dict[str, object], rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    g = _recolor_structure(np.array(template["grid"], dtype=int), rng)
    for r, c in template["tap_cells"]:
        assert int(g[int(r), int(c)]) == int(BG)
        g[int(r), int(c)] = int(TAP)
    controls = _choose_controls(template, rng)
    for r, c, color in controls:
        assert int(g[int(r), int(c)]) == int(BG)
        g[int(r), int(c)] = int(color)
    return g, controls


def _generator_metrics(
    grid: np.ndarray,
    *,
    controls: list[tuple[int, int, int]],
    faucet_cell: tuple[int, int],
    alt_emit: tuple[int, int],
) -> dict[str, object]:
    total = _decoded_total(controls)
    combo_core = _core_output(grid, controls=controls, source_cell=faucet_cell, amount=int(total))
    tap_only_core = _core_output(grid, controls=controls, source_cell=faucet_cell, amount=int(TAP_LENGTH))
    control_side_core = _core_output(grid, controls=controls, source_cell=alt_emit, amount=int(total))
    return {
        "decoded_total": int(total),
        "combo_core": combo_core,
        "delta_vs_tap_only": not np.array_equal(combo_core, tap_only_core),
        "delta_vs_control_side": not np.array_equal(combo_core, control_side_core),
    }


def _accept_candidate(
    template: dict[str, object],
    grid: np.ndarray,
    controls: list[tuple[int, int, int]],
) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    faucet_cell = _parse_faucet(grid, check_input=False)
    metrics = _generator_metrics(
        grid,
        controls=controls,
        faucet_cell=faucet_cell,
        alt_emit=tuple(template["alt_emit"]),
    )

    decoded_total = int(metrics["decoded_total"])
    combo_core = np.array(metrics["combo_core"], dtype=int)
    water_rows = [int(r) for r, _c in np.argwhere(combo_core == int(WATER))]
    water_cols = {int(c) for _r, c in np.argwhere(combo_core == int(WATER))}

    if int(np.count_nonzero(combo_core == int(WATER))) != int(decoded_total):
        return False
    if not bool(metrics["delta_vs_tap_only"]):
        return False
    if not bool(metrics["delta_vs_control_side"]):
        return False
    if not water_rows or min(water_rows) < 5:
        return False
    if len(water_cols) < 4:
        return False
    if np.any(out == int(TAP)):
        return False
    if int(np.count_nonzero(out == int(WATER))) != int(decoded_total):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        template = dict(rng.choice(ALL_TEMPLATES))
        grid, controls = _candidate_input(template, rng)
        if _accept_candidate(template, grid, controls):
            return grid
    raise RuntimeError("failed to generate a valid D5/2+D14/2 combo instance")
