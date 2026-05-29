# Variant of 28a6681f
# Dimension: D14 — 输入中水的约束条件 (src)
# Change: 每个水像素必须被下方支撑（不悬空），且不能两侧都是背景... → 水像素由结构顶部释放

from __future__ import annotations

import random

import numpy as np


"""
variant_design:
  reference_note:
    original_task: supported blue source water is already drawn in the input
    source_task_inspiration: borrow the "control marker" idea from 4a21e3da, but keep the original basin physics

  faucet_encoding:
    tap_color: 8
    water_color: 1
    shape: each faucet is one horizontal run of tap pixels placed on top of a visible structure ledge
    quantity_rule: faucet run length == released water count
    outlet_rule: the faucet emits from the unique open end of the run, i.e. the side with adjacent background
    lifecycle: tap pixels disappear before the simulation; only released blue water remains in the output

  behavior_trace:
    original:
      - a supported blue run already sits on a ledge
      - the solver immediately relaxes those blue pixels into the lower basins
    variant:
      - the same ledge carries a tap bar such as 888 instead of blue water
      - the solver reads that as "release 3 drops from the open end of this bar"
      - after each emitted drop, all existing water is relaxed with the original gravity / diagonal / horizontal rules
    difference:
      - the changing part of the input is no longer blue water placement but the tap-bar count and placement
      - the changing part of the solver is no longer "validate existing water" but "decode faucet bars, emit water, then reuse the old settling physics"

  complexity_calibration:
    reasons_for_bar_encoding:
      - a single control pixel would hide quantity in an invisible convention
      - a horizontal run makes quantity directly countable in 2-4 ARC examples
      - repeated 2+2, 2+3, and 3+3 bar combinations provide visible support for the counting rule
    reasons_for_two_faucets:
      - one short bar is easy to miss
      - two bars make the "marker count -> water amount" relation much clearer without adding many cases

  implicit_assumptions_from_original_main:
    - original input checks assumed every movable source pixel was already blue water
    - the original diagonal ambiguity checks were harmless because the generator only placed supported water
    - in the variant, those assumptions move to faucet outlets instead: each bar must have exactly one open side and a solid ledge below
    - the original guide-color scan would misread tap pixels as guide fragments, so that logic is intentionally removed
    - original water conservation becomes "output water count equals total tap pixels"

  placement_validity:
    - the generator uses only prevalidated 10x10 templates
    - every template has 2 faucet ledges, each separated vertically so taps never sit on top of taps
    - each ledge is flush to one edge, so the tap bar has one blocked side and one open side
    - lower structures are bottom-anchored and already verified against the solver

  visual_observability:
    - input contains no blue water at all
    - bright tap bars appear in the upper structure area and vanish in the output
    - blue water appears only in the lower basins, making the release behavior easy to notice
    - mirroring left/right and remapping structure colors preserve the same rule while keeping examples non-identical

  neatness_and_colors:
    - tap color is fixed so the human does not need to infer a hidden color map
    - structure colors are remapped only among non-water, non-tap colors
    - no ordering depends on color ids; only visible run length and visible open-side geometry matter
"""


BG = 0
WATER = 1
TAP = 8
STRUCTURE_PALETTE = (2, 3, 4, 5, 6, 7, 9)


TEMPLATES: tuple[dict[str, object], ...] = (
    {"surface": (10, 6, 5, 6, 9, 7, 6, 9, 9, 4), "shelves": ((1, 3), (4, 3))},
    {"surface": (10, 6, 7, 8, 6, 5, 8, 8, 6, 4), "shelves": ((1, 3), (4, 3))},
    {"surface": (10, 9, 9, 9, 5, 5, 9, 5, 8, 6), "shelves": ((1, 3), (4, 2))},
    {"surface": (10, 8, 4, 4, 8, 6, 9, 7, 9, 8), "shelves": ((1, 3), (4, 2))},
    {"surface": (10, 6, 6, 9, 8, 7, 9, 4, 4, 6), "shelves": ((1, 2), (4, 2))},
    {"surface": (10, 8, 9, 9, 7, 4, 5, 7, 8, 6), "shelves": ((1, 2), (4, 2))},
)


def _assert_stable_output(g: np.ndarray) -> None:
    H, W = g.shape
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
        assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)


def _relax_water(grid: np.ndarray, check_output: bool = False) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

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
        _assert_stable_output(g)
    return g


def _parse_faucets(grid: np.ndarray, check_input: bool) -> list[tuple[int, int, int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    tap_pts = np.argwhere(g == int(TAP))
    if check_input:
        assert g.shape == (10, 10), "grid must stay 10x10"
        assert not np.any(g == int(WATER)), "variant input must not contain pre-placed water"
        assert len(tap_pts) > 0, "input must contain faucet markers"

    faucets: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    for r, c in tap_pts:
        rr = int(r)
        cc = int(c)
        if (rr, cc) in seen:
            continue

        left = int(cc)
        while left - 1 >= 0 and int(g[rr, left - 1]) == int(TAP):
            left -= 1
        right = int(cc)
        while right + 1 < W and int(g[rr, right + 1]) == int(TAP):
            right += 1

        for x in range(int(left), int(right) + 1):
            seen.add((int(rr), int(x)))
            if check_input:
                assert rr + 1 < H, "faucet cannot be on bottom row"
                assert int(g[rr + 1, int(x)]) not in (int(BG), int(TAP)), "faucet must sit on structure top"
                assert rr == 0 or int(g[rr - 1, int(x)]) == int(BG), "faucet must be exposed from above"
                assert rr == 0 or int(g[rr - 1, int(x)]) != int(TAP), "faucet bars must be single-row"
                assert rr + 1 == H or int(g[rr + 1, int(x)]) != int(TAP), "faucet bars must be single-row"

        left_open = left - 1 >= 0 and int(g[rr, left - 1]) == int(BG)
        right_open = right + 1 < W and int(g[rr, right + 1]) == int(BG)
        assert left_open != right_open, "faucet run must have exactly one open side"

        emit_c = int(left if left_open else right)
        faucets.append((int(rr), int(emit_c), int(right - left + 1)))

    faucets.sort()
    return faucets


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Release counted water from faucet markers and let it settle into the basin structures.

    Input structure:
    - bg = 0. Water is absent from the input; instead color 8 marks faucet bars on upper ledges.
    - Each maximal horizontal faucet bar sits on top of a structure and points toward a unique open side.
    - Other non-zero colors are static solid walls, shelves, and basin structures.

    Rule:
    - Read each faucet bar as a source: its length is the number of water drops to emit.
    - Emit those drops one at a time from the bar's open-end cell, removing the bar markers first.
    - After each emitted drop, reuse the original gravity, diagonal slide, and horizontal slide settling until stable.

    Key insight: The new task is not about locating movable blue pixels in the input; it is about decoding control bars into water quantity and outlet position before running the old basin simulation.
    """
    g = np.array(grid, dtype=int).copy()
    faucets = _parse_faucets(g, check_input=check_input)

    if not faucets:
        return _relax_water(g, check_output=check_output)

    g[g == int(TAP)] = int(BG)

    for r, c, qty in faucets:
        for _ in range(int(qty)):
            assert int(g[int(r), int(c)]) == int(BG), "faucet outlet must be empty before emission"
            g[int(r), int(c)] = int(WATER)
            g = _relax_water(g, check_output=False)

    if check_output:
        _assert_stable_output(g)
    return g


def _build_template(template: dict[str, object]) -> np.ndarray:
    left_color, right_color, shelf_color = random.sample(STRUCTURE_PALETTE, 3)
    if random.random() < 0.35:
        shelf_color = random.choice((left_color, right_color))

    g = np.zeros((10, 10), dtype=int)
    surface = tuple(int(x) for x in template["surface"])
    shelves = tuple((int(r), int(length)) for r, length in template["shelves"])

    for c, top in enumerate(surface):
        if int(top) < 10:
            fill_color = int(left_color if c <= 4 else right_color)
            g[int(top) :, int(c)] = int(fill_color)

    for shelf_row, length in shelves:
        g[int(shelf_row), 0 : int(length)] = int(shelf_color)
        g[int(shelf_row) - 1, 0 : int(length)] = int(TAP)

    if random.random() < 0.5:
        g = np.fliplr(g)
    return g


def generate_input() -> np.ndarray:
    for _ in range(32):
        template = random.choice(TEMPLATES)
        g = _build_template(template)
        out = main(g)

        tap_count = int(np.count_nonzero(g == int(TAP)))
        water_count = int(np.count_nonzero(out == int(WATER)))
        water_rows = [int(r) for r, _c in np.argwhere(out == int(WATER))]
        water_cols = {int(c) for _r, c in np.argwhere(out == int(WATER))}

        assert int(np.count_nonzero(g == int(WATER))) == 0
        if tap_count < 4:
            continue
        if water_count != tap_count:
            continue
        if not water_rows or min(water_rows) < 4:
            continue
        if len(water_cols) < 2:
            continue
        return g
    raise AssertionError("failed to build a valid faucet-source sample")
