# Combo variant of 28a6681f (index 43)
# combo_spec: D3/0+D5/2
# Name: 无水平滑动+碰撞特定目标停止

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D3/0 + D5/2
# --------------------------------------
# a) Concretization + behavior trace:
#    - The scene stays a 10x10 basin puzzle with water color 1 and a dedicated target color 8.
#      The generated landform is a high right-side shelf beside a narrow vertical chute; a
#      single target pixel is embedded in the chute wall above the eventual floor.
#    - Small concrete trace on template A:
#        input top rows: a 2x3 source straddles the chute mouth and the shelf roof.
#        original task:
#          * chute-aligned water falls first;
#          * shelf water later scans horizontally to the nearest drop and also drains into the
#            chute, so the upper shelf empties.
#        D3/0 only:
#          * the same chute water still descends;
#          * shelf water loses horizontal routing and remains trapped high in rows 1-2.
#        D5/2 only:
#          * horizontal routing still exists, so the shelf still empties;
#          * one of the descending pixels halts as soon as it touches the target instead of
#            sinking to the lower resting band.
#        combo:
#          * shelf water remains stranded because horizontal movement is gone;
#          * chute water stops early when it first becomes target-adjacent.
#      The output therefore contains both a visible upper residue and a higher-than-usual
#      stopped pixel near the target, so the two changes reinforce each other rather than acting
#      on disjoint examples.
#
# b) Complexity calibration:
#    - Only two validated geometry families are used, plus structure recoloring. That is enough
#      variety to avoid a frozen bitmap, but not enough to create ambiguous route choices.
#    - The target color is fixed to 8 across all samples. Keeping exactly one semantic marker
#      color makes the stopping cue easy to infer from 2-4 examples.
#    - The source shape is always a compact 2x3 block. Repeating the same source motif helps a
#      human notice that only the chute-aligned part can still reach the lower area.
#
# c) Hidden-assumption audit of the original main():
#    - The original solver assumed that a supported water pixel may scan its whole row for the
#      nearest drop. That branch is invalid here and must be optionalized away.
#    - The original output invariant "left, below, and right all blocked" is no longer correct:
#      target contact can freeze a water cell while there is still empty space below or beside it.
#      The new invariant is: each final water cell either touches the target or is a genuine
#      no-horizontal dead end with no downward or diagonal exit.
#    - The original input checks required all source water to be supported from below. These
#      combo templates deliberately straddle a chute mouth, so some source cells start above
#      empty space. The input contract is therefore weakened to structural sanity only.
#    - Target cells behave like solids for pathing, so movement tests still traverse only
#      background cells. No additional occupancy state is needed.
#
# d) Placement legality:
#    - Every template is a hand-validated 10x10 bitmap, so all solids, water, and the target are
#      in-bounds and non-overlapping before recoloring.
#    - Structure recoloring only remaps non-zero solid colors and never touches water or target,
#      so semantics cannot drift.
#
# e) Visual observability:
#    - The combo leaves two repeated visual cues in the same sample:
#      * at least two water pixels remain high in the top band under the no-horizontal rule;
#      * at least one lower water pixel visibly stops adjacent to the lone target marker.
#    - Internal generator metrics reject any sample where the D3 cue disappears against the
#      D5-only ablation, or where the D5 cue disappears against the D3-only ablation.
#    - No rule depends on hidden numeric ordering; all cues come from visible geometry and the
#      fixed target marker.
#
# f) Visual cleanliness:
#    - Each sample has one compact source zone, one chute-and-shelf landform, and one target
#      marker. There are no extra annotations layered on top of the basin.
#    - The target sits on an exposed wall face rather than inside a multicolor cluster, so the
#      stopping cue stays legible.
#
# g) Change inferability / color discipline:
#    - Colors serve only three roles: background 0, water 1, target 8, and 2-3 recolored solids.
#      The solid recolor preserves the original ARC-style variety without introducing extra
#      semantic categories.
#    - Because the same target color repeats while solid colors vary, a human can infer that the
#      target's role is structural rather than palette-order based.
#
# 9) New solver assumptions guaranteed by generate_input():
#    - The grid contains exactly one target pixel of color 8.
#    - Source water occupies only template-approved cells; these layouts have a unique diagonal
#      choice whenever diagonal motion is available, so the solver never sees ambiguous splits.
#    - The top band always starts with a mixed source: some pixels are chute-aligned, others sit
#      on the high shelf. That guarantees the two combo dimensions both have something to act on.
#    - At least one water pixel can touch the target under the combo solver, and at least two
#      water pixels remain in the top three rows under the combo solver but not under the D5-only
#      ablation. These generator-side checks prevent regression to either single-variant behavior.


BG = 0
WATER = 1
TARGET = 8
GRID_SHAPE = (10, 10)
TOP_BAND_ROWS = 3
SOLID_PALETTE = (2, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 200


TEMPLATES = (
    {
        "rows": (
            "0000000000",
            "0000011100",
            "0000011100",
            "0000006666",
            "0000006666",
            "0000008666",
            "0000600666",
            "0000600666",
            "0000600000",
            "2222222222",
        ),
    },
    {
        "rows": (
            "0000000000",
            "0000111000",
            "0000111000",
            "0000066660",
            "0000066660",
            "0000086660",
            "0000500660",
            "0000500660",
            "0000500000",
            "2222222222",
        ),
    },
)


def _parse_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _touches_target(g: np.ndarray, r: int, c: int) -> bool:
    H, W = g.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(TARGET):
            return True
    return False


def _diagonal_options(g: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    H, W = g.shape
    opts: list[tuple[int, int]] = []
    if (
        r + 1 < H
        and c - 1 >= 0
        and int(g[r, c - 1]) == int(BG)
        and int(g[r + 1, c - 1]) == int(BG)
    ):
        opts.append((int(r + 1), int(c - 1)))
    if (
        r + 1 < H
        and c + 1 < W
        and int(g[r, c + 1]) == int(BG)
        and int(g[r + 1, c + 1]) == int(BG)
    ):
        opts.append((int(r + 1), int(c + 1)))
    return opts


def _is_no_horizontal_dead_end(g: np.ndarray, r: int, c: int) -> bool:
    H, _W = g.shape
    below_blocked = bool(r == H - 1 or int(g[r + 1, c]) != int(BG))
    return bool(below_blocked and not _diagonal_options(g, int(r), int(c)))


def _simulate(
    grid: np.ndarray,
    *,
    allow_horizontal: bool,
    stop_on_target: bool,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        assert g.shape == GRID_SHAPE, f"expected grid shape {GRID_SHAPE}, got {g.shape}"
        assert np.count_nonzero(g == int(TARGET)) == 1, "variant input must contain exactly one target"
        assert np.all((g >= 0) & (g <= 9)), "grid colors must stay within ARC palette"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if stop_on_target and _touches_target(g, int(rr), int(cc)):
                return int(rr), int(cc)

            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = _diagonal_options(g, int(rr), int(cc))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if allow_horizontal and rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if drops:
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

            return int(rr), int(cc)

        raise AssertionError("water pixel did not settle")

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            assert int(g[int(rr), int(cc)]) == int(BG), "water paths should not overlap"
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert _touches_target(g, int(rr), int(cc)) or _is_no_horizontal_dead_end(
                g, int(rr), int(cc)
            ), "output water must either touch target or be a no-horizontal dead end"

    return g


def _main_d3_only(grid: np.ndarray) -> np.ndarray:
    return _simulate(
        grid,
        allow_horizontal=False,
        stop_on_target=False,
        check_input=False,
        check_output=False,
    )


def _main_d5_only(grid: np.ndarray) -> np.ndarray:
    return _simulate(
        grid,
        allow_horizontal=True,
        stop_on_target=True,
        check_input=False,
        check_output=False,
    )


def _main_original(grid: np.ndarray) -> np.ndarray:
    return _simulate(
        grid,
        allow_horizontal=True,
        stop_on_target=False,
        check_input=False,
        check_output=False,
    )


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin water that may only drop or slip diagonally, and that freezes on target contact.

    Input structure:
    - bg = 0. Moving water is color 1 and the dedicated stop marker is color 8.
    - Other colors are fixed bottom-anchored shelves, walls, and chute boundaries.
    - Source water can straddle both a supported roof and the mouth of a chute, so some pixels
      begin on solid support while others begin directly above empty space.

    Rule:
    - A water pixel stops immediately once its current cell is 4-neighbor-adjacent to the target.
    - Otherwise it falls straight down through background cells, or if blocked below it may take
      the unique open diagonal descent.
    - Horizontal shelf sliding is disabled completely; if neither downward nor diagonal motion is
      available, that pixel stays where it is.
    - Repeatedly re-settle all water pixels until a full pass causes no movement.

    Key insight: Removing horizontal rerouting strands shelf water high above the basin, while the
    target truncates the chute path early. Only chute-aligned water can ever reach the target.
    """
    return _simulate(
        grid,
        allow_horizontal=False,
        stop_on_target=True,
        check_input=bool(check_input),
        check_output=bool(check_output),
    )


def _apply_structure_recolor(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    colors = sorted(set(map(int, out.flatten())) - {int(BG), int(WATER), int(TARGET)})
    new_colors = random.sample(SOLID_PALETTE, k=len(colors))
    mapping = {int(src): int(dst) for src, dst in zip(colors, new_colors)}
    for src, dst in mapping.items():
        out[out == int(src)] = int(-dst)
    out[out < 0] *= -1
    return out


def _count_top_band_water(g: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(g, dtype=int)[:TOP_BAND_ROWS] == int(WATER)))


def _touching_water_count(g: np.ndarray) -> int:
    total = 0
    for r, c in np.argwhere(np.asarray(g, dtype=int) == int(WATER)):
        total += int(_touches_target(np.asarray(g, dtype=int), int(r), int(c)))
    return int(total)


def _descent_front_row(g: np.ndarray) -> int:
    rows = [int(r) for r, _c in np.argwhere(np.asarray(g, dtype=int) == int(WATER)) if int(r) >= TOP_BAND_ROWS]
    assert rows, "expected at least one lower water cell"
    return int(min(rows))


def _quantify_combo_behavior(candidate: np.ndarray) -> dict[str, int]:
    combo = main(candidate, check_input=False, check_output=False)
    d3_only = _main_d3_only(candidate)
    d5_only = _main_d5_only(candidate)
    original = _main_original(candidate)
    return {
        "combo_top_band": _count_top_band_water(combo),
        "d5_top_band": _count_top_band_water(d5_only),
        "combo_touching": _touching_water_count(combo),
        "combo_front_row": _descent_front_row(combo),
        "d3_front_row": _descent_front_row(d3_only),
        "delta_vs_d3": int(np.count_nonzero(combo != d3_only)),
        "delta_vs_d5": int(np.count_nonzero(combo != d5_only)),
        "delta_vs_original": int(np.count_nonzero(combo != original)),
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        spec = random.choice(TEMPLATES)
        g = _parse_rows(spec["rows"])

        g = _apply_structure_recolor(g)

        metrics = _quantify_combo_behavior(g)
        if metrics["combo_top_band"] < 2:
            continue
        if metrics["d5_top_band"] != 0:
            continue
        if metrics["combo_touching"] < 1:
            continue
        if metrics["combo_front_row"] >= metrics["d3_front_row"]:
            continue
        if metrics["delta_vs_d3"] == 0 or metrics["delta_vs_d5"] == 0 or metrics["delta_vs_original"] == 0:
            continue
        assert np.count_nonzero(g == int(TARGET)) == 1
        return g

    raise AssertionError("failed to generate a valid combo input")
