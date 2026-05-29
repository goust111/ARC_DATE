# Combo variant of 28a6681f (index 14)
# combo_spec: D4/2+D15/1
# Name: 大量预填水+深度分层颜色
#
# Design analysis
# ---------------
# spec:
#   concretization:
#     The scene stays a 10x10 bottom-basin water task, but the liquid now has two input roles:
#       - old prefilled water: deep-blue color 1, already stable inside the basin network;
#       - fresh added water: light-blue color 8, still sitting in the source region.
#     Solver-side, only color 8 moves. After it settles, every water cell is recolored from
#     scratch by final depth: the exposed top layer in each vertical water stack becomes light
#     blue 8, and every cell with water directly above becomes deep blue 1.
#   behavior_trace:
#     Small example: suppose a basin already contains a dark stack
#       rows 9-7: [1, 1, 1]
#     and two new light pixels 8 are placed on the source shelf above it.
#     Original task: if all water were just movable blue 1, there would be no distinction between
#     existing and new water, and the output would still be a single-color pool.
#     Fused rule:
#       1. The existing 1-cells do not move; they act as occupied support cells.
#       2. The two 8-cells continue the original gravity / diagonal-slip / horizontal-search
#          process until they find the next reachable cells above or beside that old water level.
#       3. Once all water is stable, color no longer means old-vs-new. Instead, every surface
#          water cell becomes 8, and submerged cells become 1.
#     Difference from the base task: old water changes the reachable depth for the fresh water,
#     and then the final palette reflects depth rather than source identity.
#   complexity_calibration:
#     Only two water colors are used, with fixed semantics repeated across all samples:
#       - input 1 = old/stable water,
#       - input 8 = fresh/movable water,
#       - output 1/8 = deep/shallow water.
#     The depth mapping is intentionally binary (surface vs below-surface), not many-level, so
#     2-4 ARC examples are enough to infer it. Generator search also enforces that both colors
#     appear in the output and that old water is visibly recolored at the surface.
#   implicit_assumptions_review:
#     The original `water = 1` assumption breaks in two ways:
#       - not every liquid-colored cell moves anymore;
#       - output colors no longer preserve input roles.
#     Therefore main() is split into two phases:
#       - simulate only color 8 through background cells while treating color 1 as fixed support;
#       - recolor all final liquid by depth.
#     The original guide validation also has to exclude both liquid colors, not just one.
#     The original output checker scanned only one water color; here it must check the union of
#     deep and shallow final water.
#   placement_legality:
#     Generator starts from official 28a6681f source layouts, mirrors and recolors only solids,
#     and never edits the source mask geometry. Old water is created by first simulating a legal
#     smaller source-water subset to a stable state; fresh water is then placed only in source-mask
#     cells that are empty, supported, and not open on both horizontal sides.
#   visual_observability:
#     Every accepted sample shows three separate visual facts:
#       - a large dark prefilled body already occupies basin cells in the input;
#       - a small light-blue fresh addition remains in the source area;
#       - in the output, the top exposed water layer becomes light blue even when those cells were
#         dark in the input, revealing that final color depends on depth, not on age.
#   cleanliness:
#     No legends or extra markers are added. The scene stays "basins + water", and the only added
#     complexity is semantic: fixed old water plus depth recoloring. That keeps the picture close
#     to the base task and avoids layered annotation clutter.
#   color_reasoning:
#     No hidden color ordering is used. Humans can infer the mapping from geometry alone:
#     output surface water is always 8, deeper water beneath it is 1. The same mapping repeats in
#     every sample, and old input water often changes from 1 to 8 at exposed surface cells.
#   fusion_metrics:
#     Generator rejects candidates unless both fused halves are active:
#       - prefill_effect: removing old water either invalidates the fresh source support or changes
#         the fresh water's settled positions;
#       - depth_effect: the recolored output differs from the pre-recolor settled state, with at
#         least two old-water input cells turning into shallow output cells.
#   solver_assumptions:
#     New assumptions guaranteed by generate_input():
#       - structure colors never use 1 or 8;
#       - every color-1 cell in the input is already stable water;
#       - every color-8 cell in the input is supported fresh water in the source region;
#       - the source-mask cells used for fresh water belong to the original task's legal water
#         source area, so the original motion physics stays unambiguous.

from __future__ import annotations

import random

import numpy as np


BG = 0
DEEP_WATER = 1
SHALLOW_WATER = 8
LIQUID_COLORS = (DEEP_WATER, SHALLOW_WATER)
SOLID_COLORS = (2, 3, 4, 5, 6, 7, 9)
GRID_SHAPE = (10, 10)
MAX_GENERATION_ATTEMPTS = 500


_BASE_INPUTS: tuple[np.ndarray, ...] = (
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 3, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 3, 0, 3, 1, 1],
            [0, 0, 0, 0, 2, 3, 0, 3, 1, 1],
            [0, 0, 0, 0, 2, 3, 0, 3, 1, 1],
            [0, 2, 2, 2, 2, 3, 3, 3, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 3, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 3, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 3, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 3, 0, 0, 3, 1, 1, 1, 1],
            [0, 0, 3, 0, 0, 3, 5, 1, 1, 5],
            [0, 0, 3, 1, 1, 3, 5, 1, 1, 5],
            [0, 0, 3, 1, 1, 3, 5, 1, 1, 5],
            [0, 0, 3, 3, 3, 3, 5, 5, 5, 5],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 9, 1, 1, 0, 0, 0, 0, 9, 9],
            [0, 9, 9, 1, 0, 0, 0, 9, 9, 0],
            [0, 0, 9, 9, 0, 0, 0, 9, 0, 0],
            [0, 0, 0, 9, 0, 0, 9, 9, 0, 0],
            [0, 0, 0, 0, 0, 9, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
            [2, 0, 0, 4, 0, 4, 0, 0, 0, 0],
            [2, 0, 2, 4, 0, 4, 0, 0, 0, 0],
            [2, 2, 2, 4, 4, 4, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 9, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 9, 9, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 9, 9, 9, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 9, 9, 9, 0, 0, 5],
            [0, 0, 0, 0, 0, 0, 0, 5, 0, 5],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 5],
            [3, 3, 0, 0, 0, 0, 5, 5, 0, 5],
            [3, 3, 3, 0, 0, 5, 5, 5, 0, 5],
            [3, 3, 3, 3, 5, 5, 5, 5, 0, 5],
        ],
        dtype=int,
    ),
)


def _positions_of_value(grid: np.ndarray, value: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(np.asarray(grid, dtype=int) == int(value))]


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


def _assert_supported_input_cells(grid: np.ndarray, water_color: int) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for r, c in np.argwhere(g == int(water_color)):
        rr = int(r)
        cc = int(c)
        assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"


def _assert_stable_cells(grid: np.ndarray, colors: tuple[int, ...]) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = np.isin(g, colors)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or bool(mask[rr, cc - 1]) or int(g[rr, cc - 1]) != int(BG), "water must be left-supported"
        assert rr == h - 1 or bool(mask[rr + 1, cc]) or int(g[rr + 1, cc]) != int(BG), "water must be floor-supported"
        assert cc == w - 1 or bool(mask[rr, cc + 1]) or int(g[rr, cc + 1]) != int(BG), "water must be right-supported"


def _validate_guides(grid: np.ndarray, excluded_colors: set[int]) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bottom_colors = set(map(int, g[h - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - excluded_colors - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _simulate_movable_color(
    grid: np.ndarray,
    movable_color: int,
    fixed_stable_colors: tuple[int, ...] = (),
    *,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    if check_input:
        _assert_supported_input_cells(g, int(movable_color))
        if fixed_stable_colors:
            _assert_stable_cells(g, tuple(int(c) for c in fixed_stable_colors))
        _validate_guides(g, {BG, int(movable_color), *map(int, fixed_stable_colors)})

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < h
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < h
                and cc + 1 < w
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == int(BG):
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

    for _pass in range(h * w):
        moved = False
        pts = sorted(_positions_of_value(g, int(movable_color)), reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(movable_color):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(movable_color)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        _assert_stable_cells(g, (int(movable_color),))
    return g


def _recolor_by_depth(settled: np.ndarray) -> np.ndarray:
    g = np.array(settled, dtype=int).copy()
    water = np.isin(g, LIQUID_COLORS)
    out = g.copy()
    for r, c in np.argwhere(water):
        rr = int(r)
        cc = int(c)
        if rr > 0 and bool(water[rr - 1, cc]):
            out[rr, cc] = int(DEEP_WATER)
        else:
            out[rr, cc] = int(SHALLOW_WATER)
    return out


def _simulate_variant(
    grid: np.ndarray,
    *,
    check_input: bool,
    check_output: bool,
) -> tuple[np.ndarray, np.ndarray]:
    settled = _simulate_movable_color(
        grid,
        SHALLOW_WATER,
        (DEEP_WATER,),
        check_input=bool(check_input),
        check_output=False,
    )
    out = _recolor_by_depth(settled)
    if check_output:
        _assert_stable_cells(out, LIQUID_COLORS)
    return settled, out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Continue filling a basin network that already contains stable water, then recolor the result by depth.

    Input structure:
    - bg = 0. Deep-blue color 1 marks old prefilled water that is already stable inside the basins.
    - Light-blue color 8 marks the small amount of fresh water that still needs to flow from the source region.
    - All other nonzero colors are static basin walls, floors, or optional guides copied from the original task family.

    Rule:
    - Only the fresh 8-cells move, and they keep the original 28a6681f movement priorities: fall straight down through background, otherwise use the unique open diagonal, otherwise slide along a supported row toward the nearest reachable drop.
    - Old 1-cells do not move; they behave as already-filled occupied liquid support and can raise the effective water level enough to cause earlier spill or shallower settling.
    - After the fresh water finishes moving, all water is recolored from scratch by final depth: exposed surface cells become light blue 8, and cells with water directly above them become deep blue 1.

    Key insight: The task is no longer "move one blue fluid". The solver must first treat old and new water differently during motion, then forget that distinction and infer the final colors from surface depth instead.
    """
    _settled, out = _simulate_variant(
        np.asarray(grid, dtype=int),
        check_input=bool(check_input),
        check_output=bool(check_output),
    )
    return out


def _remap_structure(base: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.array(base, dtype=int).copy()
    if random.random() < 0.5:
        g = np.fliplr(g)
    source_positions = [(int(r), int(c)) for r, c in np.argwhere(g == int(DEEP_WATER))]
    solid_values = sorted(set(map(int, g.flatten())) - {BG, DEEP_WATER})
    palette = random.sample(list(SOLID_COLORS), len(solid_values))
    remapped = g.copy()
    for old, new in zip(solid_values, palette):
        remapped[g == int(old)] = int(new)
    remapped[g == int(DEEP_WATER)] = int(BG)
    source_positions.sort(reverse=True)
    return remapped, source_positions


def _seed_prefilled_state(structure: np.ndarray, source_positions: list[tuple[int, int]], old_seed_count: int) -> np.ndarray:
    seed = np.array(structure, dtype=int).copy()
    for r, c in source_positions[:old_seed_count]:
        seed[int(r), int(c)] = int(DEEP_WATER)
    return _simulate_movable_color(
        seed,
        DEEP_WATER,
        (),
        check_input=True,
        check_output=True,
    )


def _pick_fresh_source_cells(
    source_positions: list[tuple[int, int]],
    occupied_grid: np.ndarray,
    fresh_count: int,
) -> list[tuple[int, int]] | None:
    work = np.array(occupied_grid, dtype=int).copy()
    chosen: list[tuple[int, int]] = []
    h, w = work.shape
    for r, c in source_positions:
        rr = int(r)
        cc = int(c)
        if int(work[rr, cc]) != int(BG):
            continue
        if rr != h - 1 and int(work[rr + 1, cc]) == int(BG):
            continue
        left_bg = cc > 0 and int(work[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < w and int(work[rr, cc + 1]) == int(BG)
        if left_bg and right_bg:
            continue
        chosen.append((rr, cc))
        work[rr, cc] = int(SHALLOW_WATER)
        if len(chosen) == fresh_count:
            return chosen
    return None


def _positions_as_set(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _variant_metrics(grid: np.ndarray) -> dict[str, int | bool | str]:
    settled, out = _simulate_variant(grid, check_input=True, check_output=True)
    fresh_input = np.asarray(grid, dtype=int) == int(SHALLOW_WATER)
    old_input = np.asarray(grid, dtype=int) == int(DEEP_WATER)
    fresh_final = np.asarray(settled, dtype=int) == int(SHALLOW_WATER)

    fresh_only = np.array(grid, dtype=int).copy()
    fresh_only[fresh_only == int(DEEP_WATER)] = int(BG)

    try:
        settled_without_old = _simulate_movable_color(
            fresh_only,
            SHALLOW_WATER,
            (),
            check_input=True,
            check_output=True,
        )
        prefill_effect = _positions_as_set(fresh_final) != _positions_as_set(settled_without_old == int(SHALLOW_WATER))
        prefill_mode = "changed" if prefill_effect else "same"
    except AssertionError:
        prefill_effect = True
        prefill_mode = "invalid_without_old"

    old_to_shallow = int(np.count_nonzero(old_input & (out == int(SHALLOW_WATER))))
    deep_fresh = int(np.count_nonzero(fresh_final & (out == int(DEEP_WATER))))
    moved_fresh = int(len(_positions_as_set(fresh_input) - _positions_as_set(fresh_final)))
    recolor_effect = bool(not np.array_equal(settled, out))
    final_has_both = bool(np.any(out == int(DEEP_WATER)) and np.any(out == int(SHALLOW_WATER)))

    return {
        "prefill_effect": bool(prefill_effect),
        "prefill_mode": prefill_mode,
        "depth_effect": bool(recolor_effect),
        "old_to_shallow": int(old_to_shallow),
        "deep_fresh": int(deep_fresh),
        "moved_fresh": int(moved_fresh),
        "final_has_both": bool(final_has_both),
        "old_count": int(np.count_nonzero(old_input)),
        "fresh_count": int(np.count_nonzero(fresh_input)),
    }


def _build_candidate() -> np.ndarray | None:
    base = random.choice(_BASE_INPUTS)
    structure, source_positions = _remap_structure(base)
    total_source = len(source_positions)
    if total_source < 4:
        return None

    max_fresh = min(4, total_source - 1)
    fresh_count = random.randint(2, max_fresh)
    old_seed_min = max(fresh_count * 2, total_source // 2, 4)
    if old_seed_min > total_source:
        return None
    old_seed_count = random.randint(old_seed_min, total_source)

    try:
        old_state = _seed_prefilled_state(structure, source_positions, old_seed_count)
    except AssertionError:
        return None

    fresh_cells = _pick_fresh_source_cells(source_positions, old_state, fresh_count)
    if fresh_cells is None:
        return None

    candidate = np.array(old_state, dtype=int).copy()
    for r, c in fresh_cells:
        candidate[int(r), int(c)] = int(SHALLOW_WATER)
    return candidate


def generate_input() -> np.ndarray:
    """
    Generate a legal fused-variant input.

    Strategy:
    - Start from an official 28a6681f source layout and recolor only the solids.
    - Simulate a large monochrome prefill episode to obtain stable old water.
    - Add a small fresh source patch back into empty source-mask cells.
    - Keep only cases where old water materially affects fresh-water behavior and final depth
      recoloring is visibly active.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is None:
            continue
        try:
            metrics = _variant_metrics(candidate)
        except AssertionError:
            continue
        if (
            metrics["prefill_effect"]
            and metrics["depth_effect"]
            and metrics["final_has_both"]
            and int(metrics["old_to_shallow"]) >= 2
            and int(metrics["moved_fresh"]) >= 1
            and int(metrics["old_count"]) >= int(metrics["fresh_count"]) + 3
        ):
            return candidate
    raise RuntimeError("generate_input failed to produce a visible D4/2 + D15/1 sample")
