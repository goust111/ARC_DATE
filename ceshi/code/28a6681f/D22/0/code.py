# Variant of 28a6681f
# Dimension: D22 — 水源与盆地的空间关系 (meta)
# Change: 水源位于盆地上方或侧面，通过guide或直接的路径连接到盆地区域... → 水源位于盆地内部（需排出）

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D22 inversion
# -------------------------------------
# spec:
#   grid: 10x10
#   colors:
#     water: 1
#     solids: exactly 2 non-water colors, one for the source chamber and one for the receiver basin
#   concrete_form:
#     - The source is no longer an exterior pour or edge reservoir.
#     - Input water starts on a supported shelf inside a small upper chamber.
#     - The chamber has one unique side outlet: a short shaft immediately outside the shelf.
#     - Below that shaft sits a lower receiving basin, so the same gravity rules now look like drainage.
#   behavior_trace:
#     - Original task at the key step:
#       exterior water rests on a ledge, slides toward the nearest drop, then falls into a basin.
#     - Variant at the same key step:
#       interior water rests on an elevated inner shelf, slides toward the chamber's only outlet, then falls into a lower basin.
#     - Difference:
#       the local movement law is unchanged; only the source/receiver geometry is inverted.
#   complexity_calibration:
#     - One outlet per sample avoids ambiguous horizontal or diagonal choices.
#     - Source width is restricted to 2-3 cells; this is large enough to look like a chamber, but small enough to drain reliably.
#     - Layouts are mirrored left/right so the rule is not tied to a fixed direction.
#   implicit_assumptions_review:
#     - The original `input water cannot have bg on both horizontal sides` assumption encoded an exterior source run.
#       That is not valid for an internal chamber, so it is removed.
#     - The original guide-validation block depends on upper edge-entering ramps.
#       This variant has no guide object, so that check is dropped.
#     - The settling logic itself still matches the variant: gravity, diagonal slip, and horizontal search for a unique drop
#       are exactly the mechanics that let the chamber empty through its outlet.
#     - The output-stability check must be adapted.
#       In the drain variant, a pixel can stably rest on an outlet ledge with a horizontal opening that does not lead to any drop,
#       so the original left/below/right enclosure test is too strict.
#       Instead we verify that every output pixel is a fixed point of the same settle rule.
#   placement_legality:
#     - Generator templates are hand-constrained so walls never overlap water, remain in-bounds, and always leave exactly one usable outlet.
#     - The bottom receiver floor and side walls are placed before validation; every sample is re-run through `main()` and rejected if the shelf fails to clear.
#   visual_observability:
#     - In every sample, all blue pixels begin inside the upper chamber and end lower in a neighboring basin.
#     - This makes the inversion visible from 2-3 examples without relying on color ids or hidden ordering rules.
#   visual_cleanliness:
#     - Only two solid colors are used, and the active geometry occupies a compact region.
#     - The source chamber, outlet shaft, and receiver basin remain visually separable.
#   color_inference:
#     - Colors are decorative only; there is no color-dependent behavior.
#     - Randomizing the two solid colors helps show that position/shape, not color identity, controls the transformation.
#   solver_assumptions_and_generator_guarantees:
#     - `main()` assumes supported input water and a unique reachable drop whenever a move exists.
#     - `generate_input()` guarantees this by sampling only from a vetted template catalog and validating with `main()`.


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water draining out of an internal chamber into a lower basin using local gravity rules.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Other colors are fixed walls, shelves, and receiver basins anchored near the bottom.
    - Water starts inside a supported upper chamber rather than above the basin system.

    Rule:
    - Each water pixel still falls straight down through bg cells.
    - When blocked below, it may slip diagonally only through the unique open side.
    - When resting on a shelf, it slides horizontally toward the nearest reachable drop, which now acts as the chamber outlet.
    - Repeat until no water pixel can move.

    Key insight: The physics is the same as the original task, but the geometry is inverted: the important step is recognizing
    that supported water inside a chamber will empty itself through the only lower outlet into the receiving basin.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1

    if check_input:
        assert g.ndim == 2
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
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

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
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
            g[rr, cc] = int(bg)
            sr, sc = settle(int(rr), int(cc))
            g[rr, cc] = int(water)
            assert (int(sr), int(sc)) == (int(rr), int(cc)), "output water must be stable"

    return g


_SAFE_LAYOUTS: tuple[tuple[int, int, int, int], ...] = (
    (1, 7, 2, 2),
    (1, 7, 3, 2),
    (1, 7, 3, 3),
    (1, 8, 2, 2),
    (1, 8, 3, 2),
    (1, 8, 3, 3),
    (2, 7, 2, 2),
    (2, 8, 2, 2),
    (2, 8, 3, 2),
    (2, 8, 3, 3),
)


def _build_case(
    source_color: int,
    target_color: int,
    x_left: int,
    x_right: int,
    source_width: int,
    water_count: int,
    mirror: bool,
) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)
    src_row = 6
    outlet_col = int(x_left + source_width + 1)

    g[src_row - 1, x_left] = int(source_color)
    g[src_row - 1, outlet_col] = int(source_color)

    g[src_row, x_left] = int(source_color)
    g[src_row, outlet_col - water_count : outlet_col] = 1
    g[src_row, x_right] = int(target_color)

    g[src_row + 1, x_left:outlet_col] = int(source_color)
    g[src_row + 1, x_right] = int(target_color)

    g[src_row + 2, x_left : outlet_col - 1] = int(source_color)
    g[src_row + 2, outlet_col] = int(target_color)
    g[src_row + 2, x_right] = int(target_color)

    g[9, x_left : outlet_col - 1] = int(source_color)
    g[9, outlet_col - 1 : x_right + 1] = int(target_color)

    if mirror:
        g = np.fliplr(g)
    return g


def _layout_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    input_positions = {(int(r), int(c)) for r, c in np.argwhere(inp == 1)}
    cleared_source_cells = sum(1 for r, c in input_positions if int(out[r, c]) != 1)
    input_row_sum = sum(int(r) for r, _c in input_positions)
    output_positions = {(int(r), int(c)) for r, c in np.argwhere(out == 1)}
    output_row_sum = sum(int(r) for r, _c in output_positions)
    receiver_pixels = sum(1 for r, c in output_positions if (r, c) not in input_positions)
    return {
        "input_water": len(input_positions),
        "cleared_source_cells": int(cleared_source_cells),
        "row_sum_gain": int(output_row_sum - input_row_sum),
        "receiver_pixels": int(receiver_pixels),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    solid_colors = rng.sample([2, 3, 4, 5, 6, 7, 8, 9], 2)

    for _ in range(200):
        x_left, x_right, source_width, water_count = rng.choice(_SAFE_LAYOUTS)
        mirror = bool(rng.randrange(2))
        inp = _build_case(
            source_color=int(solid_colors[0]),
            target_color=int(solid_colors[1]),
            x_left=int(x_left),
            x_right=int(x_right),
            source_width=int(source_width),
            water_count=int(water_count),
            mirror=mirror,
        )

        out = main(inp)
        metrics = _layout_metrics(inp, out)
        if (
            metrics["input_water"] >= 2
            and metrics["cleared_source_cells"] == metrics["input_water"]
            and metrics["row_sum_gain"] >= metrics["input_water"]
            and metrics["receiver_pixels"] == metrics["input_water"]
        ):
            return inp

    raise RuntimeError("failed to generate a draining D22 sample")
