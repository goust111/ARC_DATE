# Combo variant of 28a6681f (index 24)
# combo_spec: D1/2+D22/0
# Name: 盆地内水外排+反向重力

from __future__ import annotations

import random

import numpy as np


BG = 0
WATER = 1


# Design analysis for the combo variant
# a) Concrete behavior:
#    The combined task keeps the original local flow law but flips both the global objective and
#    the vertical direction. A useful way to reason about it is to imagine a safe forward-gravity
#    template first, then flip it vertically.
#    - Forward frame used for construction:
#      a compact source chamber sits near the top, water drains out through one shelf outlet,
#      and the packet falls into a lower basin.
#    - Actual ARC sample after vertical flip:
#      the same chamber is now near the bottom, the lower basin becomes a top-anchored inverted
#      receiver, and the water must leave the lower chamber by rising upward.
#    A concrete trace for one pixel:
#    - Original task: water above a basin drops, maybe slides diagonally downward, then settles low.
#    - This combo: water starts trapped inside the lower chamber, hangs from a ceiling, slides toward
#      the unique ceiling gap, rises through that shaft, and finally packs into the upper inverted basin.
#    - Difference: both "where the water starts" and "which vertical direction is preferred" are reversed.
# b) Complexity calibration:
#    Full multi-basin overflow plus anti-gravity drainage is easy to make ambiguous under the
#    original unique-drop rule. To keep the rule inferable and the generator reliable, each sample
#    uses one dominant receiver basin with two controlled shape families:
#    - a plain inverted U receiver,
#    - or, for the narrower layout, the same receiver with one inner step.
#    Mirror symmetry, two source widths, and several water counts still provide repeated evidence
#    that the key transformation is "internal lower source -> upward evacuation into a top receiver".
# c) Implicit assumption audit against the old main():
#    - Old input support check (`cell below is non-bg`) is invalid. Under reverse gravity, stable
#      input water must instead be supported from above.
#    - Old output enclosure test (`left/below/right blocked`) is also invalid because anti-gravity
#      stability depends on the mirrored settle rule, not on literal support below.
#    - The old settle() logic itself is still correct after vertical mirroring. Rather than duplicate
#      two subtly different physics engines, the new solver flips the grid vertically, runs the same
#      downward solver internally, then flips back.
#    - The old guide-validation logic is removed entirely because this combo has no edge-entering guide.
# d) Placement legality:
#    The generator only samples from hand-checked parameter families. Source walls, source water,
#    and receiver walls occupy disjoint coordinates; all placements stay within the 10x10 canvas.
#    Mirror is applied only after the full forward template is built, so no extra overlap cases are introduced.
# e) Visual observability:
#    In every accepted sample, all blue cells begin in the lower chamber and end in the upper receiver.
#    The chamber roof, the vertical shaft, and the top-anchored inverted basin remain visible after the move,
#    so a human can infer both reversals from 2-4 examples without hidden conventions.
# f) Visual cleanliness:
#    Only two solid colors are used: one for the source chamber and one for the receiver structure.
#    The active geometry is separated vertically, leaving a clear middle transit zone where upward motion is visible.
# g) Color inferability and solver assumptions:
#    Colors are decorative only. Blue always denotes water; the two solid colors can swap freely.
#    The solver assumes exactly the geometric properties the generator guarantees:
#    - all input water is supported from above,
#    - the source chamber has one usable upward outlet after mirroring,
#    - the chosen receiver shapes never create ambiguous nearest-drop or diagonal choices.


def _settle_down(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = g.shape
    rr = int(r)
    cc = int(c)
    for _ in range(h * w):
        if rr + 1 < h and int(g[rr + 1, cc]) == BG:
            rr += 1
            continue

        opts: list[tuple[int, int]] = []
        if rr + 1 < h and cc - 1 >= 0 and int(g[rr, cc - 1]) == BG and int(g[rr + 1, cc - 1]) == BG:
            opts.append((int(rr + 1), int(cc - 1)))
        if rr + 1 < h and cc + 1 < w and int(g[rr, cc + 1]) == BG and int(g[rr + 1, cc + 1]) == BG:
            opts.append((int(rr + 1), int(cc + 1)))
        if opts:
            assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
            rr, cc = opts[0]
            continue

        if rr + 1 < h and int(g[rr + 1, cc]) != BG:
            left = int(cc)
            while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                left -= 1
            right = int(cc)
            while right + 1 < w and int(g[rr, right + 1]) == BG:
                right += 1

            drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == BG]
            if not drops:
                break

            min_dist = min(abs(int(x - cc)) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            best = nearest[0]
            assert best != int(cc)
            cc += -1 if best < int(cc) else 1
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")
    return int(rr), int(cc)


def _solve_down(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    if check_input:
        assert g.shape == (10, 10), "generator and solver both assume the original 10x10 canvas"
        water_cells = [(int(r), int(c)) for r, c in np.argwhere(g == WATER)]
        assert water_cells, "input must contain water"
        for r, c in water_cells:
            assert r == h - 1 or int(g[r + 1, c]) != BG, "downward-frame input water must be supported"

    for _pass in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == WATER)], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != WATER:
                continue
            g[r, c] = BG
            rr, cc = _settle_down(g, int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = WATER
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            g[rr, cc] = BG
            sr, sc = _settle_down(g, int(rr), int(cc))
            g[rr, cc] = WATER
            assert (int(sr), int(sc)) == (int(rr), int(cc)), "downward-frame output water must be stable"

    return g


def _settle_up(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    flipped = np.flipud(g)
    fr = int(g.shape[0] - 1 - int(r))
    sr, sc = _settle_down(flipped, fr, int(c))
    return int(g.shape[0] - 1 - int(sr)), int(sc)


def _source_fill_order(outlet: int, width: int) -> list[tuple[int, int]]:
    cells = [(2, int(c)) for c in range(int(outlet - 1), int(outlet - width - 1), -1)]
    cells += [(1, int(c)) for c in range(int(outlet - 1), int(outlet - width - 1), -1)]
    return cells


def _build_forward_case(
    source_color: int,
    basin_color: int,
    width: int,
    water_count: int,
    mirror: bool,
    stepped_receiver: bool,
) -> np.ndarray:
    assert width in (3, 4)
    assert width + 1 <= water_count <= 2 * width
    assert not stepped_receiver or width == 3

    g = np.zeros((10, 10), dtype=int)
    source_left = 1
    outlet = 5 if width == 3 else 6
    basin_left = 4 if width == 3 else 5

    for r in (0, 1, 2):
        g[r, source_left] = int(source_color)
    for r in (0, 1):
        g[r, outlet] = int(source_color)
    for c in range(source_left, outlet):
        g[3, c] = int(source_color)

    for r, c in _source_fill_order(int(outlet), int(width))[:water_count]:
        g[r, c] = WATER

    for c in range(basin_left, 10):
        g[9, c] = int(basin_color)
    for r in range(6, 10):
        g[r, basin_left] = int(basin_color)
        g[r, 9] = int(basin_color)

    if stepped_receiver:
        g[8, 7] = int(basin_color)
        g[9, 7] = int(basin_color)

    if mirror:
        g = np.fliplr(g)

    return g


def _behavior_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    input_positions = {(int(r), int(c)) for r, c in np.argwhere(inp == WATER)}
    output_positions = {(int(r), int(c)) for r, c in np.argwhere(out == WATER)}
    input_row_sum = sum(int(r) for r, _c in input_positions)
    output_row_sum = sum(int(r) for r, _c in output_positions)
    return {
        "input_water": int(len(input_positions)),
        "input_bottom_water": int(sum(1 for r, _c in input_positions if int(r) >= 7)),
        "output_top_water": int(sum(1 for r, _c in output_positions if int(r) <= 3)),
        "source_cleared": int(sum(1 for r, c in input_positions if (int(r), int(c)) not in output_positions)),
        "upward_row_gain": int(input_row_sum - output_row_sum),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Drain water upward out of a lower internal basin into a top-anchored inverted receiver.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Two other colors form fixed source-chamber walls near the bottom and fixed inverted basin walls near the top.
    - Water starts packed inside the lower chamber, supported from above rather than resting on a floor.

    Rule:
    - Reverse gravity relative to the original task: each water pixel tries to rise straight upward through bg cells.
    - When blocked above, it may slip diagonally upward only through the unique open side.
    - When hanging from a ceiling, it slides horizontally toward the nearest column where it can rise again.
    - Repeat until no water pixel moves. This is implemented exactly by vertically flipping the grid, running the original downward settle logic, then flipping back.

    Key insight: Both the task direction and the physics direction are inverted. The blue cells are not being poured into a basin;
    they are escaping from a lower basin, and the stable destination is an upside-down container attached to the top of the grid.
    """
    g = np.array(grid, dtype=int).copy()

    if check_input:
        assert g.shape == (10, 10), "generator and solver both assume the original 10x10 canvas"
        water_cells = [(int(r), int(c)) for r, c in np.argwhere(g == WATER)]
        assert water_cells, "input must contain water"
        for r, c in water_cells:
            assert r == 0 or int(g[r - 1, c]) != BG, "reverse-gravity input water must be supported from above"

    out = np.flipud(_solve_down(np.flipud(g), check_input=False, check_output=False))

    if check_output:
        assert int(np.sum(out == WATER)) == int(np.sum(g == WATER)), "water count must be conserved"
        for r, c in np.argwhere(out == WATER):
            rr = int(r)
            cc = int(c)
            out[rr, cc] = BG
            sr, sc = _settle_up(out, int(rr), int(cc))
            out[rr, cc] = WATER
            assert (int(sr), int(sc)) == (int(rr), int(cc)), "reverse-gravity output water must be stable"

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    palette = [2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(200):
        source_color, basin_color = rng.sample(palette, 2)
        width = rng.choice([3, 3, 4])
        water_count = rng.randint(width + 1, 2 * width)
        stepped_receiver = bool(width == 3 and rng.randrange(2))
        mirror = bool(rng.randrange(2))

        forward_inp = _build_forward_case(
            source_color=int(source_color),
            basin_color=int(basin_color),
            width=int(width),
            water_count=int(water_count),
            mirror=mirror,
            stepped_receiver=stepped_receiver,
        )
        actual_inp = np.flipud(forward_inp)
        actual_out = main(actual_inp)
        downward_out = _solve_down(actual_inp, check_input=False, check_output=False)
        metrics = _behavior_metrics(actual_inp, actual_out)

        if (
            metrics["input_water"] >= 4
            and metrics["input_bottom_water"] == metrics["input_water"]
            and metrics["output_top_water"] == metrics["input_water"]
            and metrics["source_cleared"] == metrics["input_water"]
            and metrics["upward_row_gain"] >= metrics["input_water"] * 4
            and not np.array_equal(actual_out, downward_out)
        ):
            return actual_inp

    raise RuntimeError("failed to generate a valid upward-draining combo sample")
