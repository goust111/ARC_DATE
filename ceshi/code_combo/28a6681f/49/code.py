# Combo variant of 28a6681f (index 49)
# combo_spec: D6/1+D10/3
# Name: 悬浮障碍物+源结构擦除
#
# Design analysis
# spec:
#   concretization:
#     The grid remains 10x10 with background 0 and water 1. The lower half still contains a
#     connected basin network that is anchored to the bottom, but the upper half now always has
#     two detached obstacle components:
#       - a floating source shelf/reservoir component that directly supports the initial water;
#       - a second floating baffle/funnel component that stays in the output and changes the route.
#     All source water begins on the floating source component, not inside the bottom basins.
#     After water has drained away, that entire source component is erased to background.
#   behavior_trace:
#     Canonical unmirrored layout:
#       1. A 3x3 water block sits above a floating L-shaped source shelf. The right edge of the
#          shelf is open, so water can only escape by sliding right and dropping from that edge.
#       2. The water falls into the lower basin system. The separate floating baffle above the
#          right basin blocks the naive straight descent and forces a different mid-air route.
#       3. After the water stabilizes in the lower basins, the whole source shelf component is
#          removed, so the upper-left structure collapses to background in the output.
#     Compared with the original task, suspended geometry matters during routing and the structure
#     itself changes afterward, so the input/output difference is not just "same walls, moved water".
#   complexity_calibration:
#     The family is intentionally narrow: one basin template, one erasable source component, one
#     persistent floating baffle, optional extra source-column water, and optional mirroring.
#     This keeps the rule legible while still making both modifications visually unavoidable.
#   implicit_assumptions_review:
#     The original solver assumed upper non-bottom solids with diagonal links were edge-entering
#     guides and validated them specially. That assumption breaks here because upper solids are now
#     arbitrary suspended obstacles, so that validation must be removed.
#     The gravity / diagonal-slip / horizontal-search physics can remain unchanged.
#     New assumption: there is exactly one floating solid component that directly supports initial
#     water from below. The generator guarantees this, and the solver uses that component as the
#     erasable source structure.
#   placement_legality:
#     The generator uses a fixed safe template with bounded optional variations. Floating components
#     are separated from anchored walls by at least one empty row, components never overlap, and all
#     initial water cells are supported with at most one horizontal background side.
#   visual_observability:
#     In every sample the source water is visibly perched on a detached shelf, and that exact shelf
#     disappears in the output. A second suspended obstacle remains, so the solver cannot explain
#     the examples by "erase every upper structure" or by ordinary bottom-cup settling alone.
#   cleanliness:
#     Source shelf, persistent floating baffle, and anchored basins occupy distinct bands. The eye
#     can follow one route: upper reservoir -> leak edge -> lower basins, then notice the missing
#     source shelf in the output.
#   color_reasoning:
#     Colors only denote rigid solids. No rule depends on color identity or ordering.
#   fusion_metrics:
#     The generator rejects candidates unless both signals are present:
#       - erase_diff: output differs from plain settling because source-structure cells vanish;
#       - route_diff: removing the persistent floating baffle changes or breaks the resulting flow.
#   solver_assumptions:
#     Each loop/branch in main() assumes:
#       - `_source_component_mask()` can find a unique non-anchored support component;
#       - `_simulate()` still sees unambiguous diagonal and horizontal moves;
#       - erasing the source component happens after settling, so no final water cell occupies it.
#     The generator constructs examples to satisfy all three constraints.

from __future__ import annotations

import random

import numpy as np


def _paint(g: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    H, W = g.shape
    for r, c in cells:
        assert 0 <= int(r) < H and 0 <= int(c) < W
        assert int(g[int(r), int(c)]) == 0
    for r, c in cells:
        g[int(r), int(c)] = int(color)


def _solid_mask(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    return (g != 0) & (g != 1)


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if bool(seen[rr, cc]):
            continue
        comp = np.zeros_like(mask, dtype=bool)
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp[rr, cc] = True
        while stack:
            cr, cc = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(cr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    comp[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _anchored_mask(grid: np.ndarray) -> np.ndarray:
    solid = _solid_mask(grid)
    H, W = solid.shape
    anchored = np.zeros_like(solid, dtype=bool)
    stack = [(int(H - 1), int(c)) for c in range(W) if bool(solid[int(H - 1), int(c)])]
    for r, c in stack:
        anchored[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and bool(solid[nr, nc]) and not bool(anchored[nr, nc]):
                anchored[nr, nc] = True
                stack.append((nr, nc))
    return anchored


def _source_component_mask(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    solid = _solid_mask(g)
    anchored = _anchored_mask(g)

    supports = np.zeros_like(solid, dtype=bool)
    for r, c in np.argwhere(g == 1):
        rr = int(r)
        cc = int(c)
        if rr + 1 < H and bool(solid[rr + 1, cc]):
            supports[rr + 1, cc] = True

    candidates: list[tuple[int, int, np.ndarray]] = []
    for comp in _component_masks(solid):
        if bool(np.any(comp & anchored)):
            continue
        support_hits = int(np.count_nonzero(comp & supports))
        if support_hits > 0:
            top_row = min(int(r) for r, _c in np.argwhere(comp))
            candidates.append((support_hits, -top_row, comp))

    assert candidates, "missing floating source component"
    best_hits = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_hits]
    assert len(best) == 1, "ambiguous floating source component"
    return np.array(best[0][2], dtype=bool)


def _simulate(
    grid: np.ndarray,
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
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(bg):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(bg):
                    right += 1

                drops = [x for x in range(int(left), int(right) + 1) if int(g[rr + 1, x]) == int(bg)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
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
            if int(g[r, c]) != int(water):
                continue
            g[r, c] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(water)
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


def _apply_source_erase(settled: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    out = np.array(settled, dtype=int).copy()
    out[np.array(source_mask, dtype=bool)] = 0
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water draining from a floating source shelf through suspended obstacles into lower basins.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors are rigid solids: a bottom-anchored basin network plus at least two detached upper
      components, one acting as the water source shelf and another acting as a persistent floating baffle.
    - The initial water run is fully supported by the source shelf component rather than by the floor.

    Rule:
    - Water still follows local gravity: fall straight down through background, slip diagonally only
      through a unique open side channel, and slide horizontally along a supported ledge toward the
      nearest unique drop.
    - The suspended baffle remains solid and can redirect the descent before the water reaches the basins.
    - After all water has settled, erase the entire floating solid component that directly supported the
      source water, turning that source structure into background.

    Key insight: The output is determined by two coupled effects: suspended mid-air obstacles control the
    route while the specific floating component that served as the source disappears once the water leaves it.
    """
    source_mask = _source_component_mask(grid)
    settled = _simulate(grid, check_input=check_input, check_output=False)
    out = _apply_source_erase(settled, source_mask)

    if check_output:
        for r, c in np.argwhere(out == 1):
            rr = int(r)
            cc = int(c)
            assert rr == out.shape[0] - 1 or int(out[rr + 1, cc]) != 0
        assert not bool(np.any((_solid_mask(grid) & source_mask) & (out != 0))), "source structure not erased"

    return out


def _build_candidate(rng: random.Random) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    palette = rng.sample(list(range(2, 10)), k=3)
    c_source = int(palette[0])
    c_baffle = int(palette[1])
    c_left = int(palette[2])
    c_mid = int(palette[(2 + 1) % len(palette)])
    c_right = int(palette[(2 + 2) % len(palette)])

    source_cells = [(3, c) for c in range(0, 5)] + [(1, 0), (2, 0)]
    baffle_cells = [(3, c) for c in range(6, 9)] + [(4, 5), (4, 6), (5, 5)]

    basin_left = [(9, c) for c in range(1, 4)] + [(r, 1) for r in range(6, 9)]
    basin_mid = [(9, c) for c in range(4, 7)] + [(7, 4), (8, 4), (8, 5)]
    basin_right = [(9, c) for c in range(7, 9)] + [(r, 7) for r in range(6, 9)]

    _paint(g, source_cells, c_source)
    _paint(g, baffle_cells, c_baffle)
    _paint(g, basin_left, c_left)
    _paint(g, basin_mid, c_mid)
    _paint(g, basin_right, c_right)

    water_cells = [(r, c) for r in range(0, 3) for c in range(1, 4)]
    extra_height = rng.choice((0, 1, 2, 3))
    for r in range(3 - extra_height, 3):
        water_cells.append((r, 4))

    for r, c in water_cells:
        assert int(g[r, c]) == 0
        g[r, c] = 1

    if rng.random() < 0.5:
        g = np.fliplr(g).copy()

    return g


def _floating_mask(grid: np.ndarray) -> np.ndarray:
    solid = _solid_mask(grid)
    return solid & (~_anchored_mask(grid))


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    source = _source_component_mask(inp)
    source_cells = int(np.count_nonzero(source))
    settled_only = _simulate(inp, check_input=False, check_output=False)
    erase_diff = int(np.count_nonzero(out != settled_only))

    persistent_float = _floating_mask(inp) & (~source)
    persistent_cells = int(np.count_nonzero(persistent_float))
    stripped = np.array(inp, dtype=int).copy()
    stripped[persistent_float] = 0
    try:
        stripped_out = main(stripped, check_input=False, check_output=False)
        route_diff = int(np.count_nonzero(stripped_out != out))
    except AssertionError:
        route_diff = int(inp.size)

    return {
        "source_cells": source_cells,
        "erase_diff": erase_diff,
        "persistent_cells": persistent_cells,
        "route_diff": route_diff,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inp = _build_candidate(rng)
    try:
        out = main(inp)
    except AssertionError:
        return None

    metrics = _variant_metrics(inp, out)
    if metrics["source_cells"] < 6:
        return None
    if metrics["erase_diff"] < metrics["source_cells"]:
        return None
    if metrics["persistent_cells"] < 5:
        return None
    if metrics["route_diff"] < 5:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
