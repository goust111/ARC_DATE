# Combo variant of 28a6681f (index 40)
# combo_spec: D17/1+D19/2
# Name: 图例区域定义水色+预填水为不同颜色

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D17/1 + D19/2
# ----------------------------------------
# a) Concretization + behavior trace:
#    - Base task: color 1 is always the only water color. Every water pixel follows the
#      same gravity/diagonal/horizontal-settling rule until all water reaches stable basin
#      cells.
#    - Combo input: a 2x2 legend is pasted into one top corner. The corner cell gives the
#      fresh moving-water color, the edge-adjacent cell below it gives the old carried-over
#      water color, and the inward column repeats the structure color. Outside the legend,
#      fresh water starts in source stacks while old water may already occupy deep basin
#      cells.
#    - Combo output: only the fresh-water color moves. Old water stays where it is and acts
#      like already-filled liquid support. Example in a top-left legend:
#        [fresh, wall]
#        [old  , wall]
#      If a basin already contains two `old` cells at the floor, the incoming `fresh` cells
#      settle above or beside them rather than recoloring or displacing them.
#
# b) Complexity calibration:
#    - The legend is kept to one simple 2x2 layout in a top corner so humans can infer it
#      from 2-4 examples. Making the legend roam freely or encode more roles would create
#      unnecessary ambiguity.
#    - All solids use one repeated structure color per sample. This keeps the color system
#      small: background, structure, fresh water, old water.
#    - Old water is always present in multiple cells, not just one marker pixel, so the
#      D19/2 change is visually obvious and repeatedly supported within each sample.
#
# c) Hidden-assumption audit of the original main():
#    - `water = 1` no longer holds. The solver must decode the moving-water color from the
#      legend before any simulation begins.
#    - Every pixel of the water color used to be movable. That is false now because one
#      legend cell shares the fresh-water color but must remain fixed, so legend cells are
#      explicitly masked out from the simulation.
#    - Previously all non-background non-water colors were just solids. Now the old-water
#      color is neither background nor structure, but it must still block motion and remain
#      unchanged. The movement code therefore treats any non-background cell as occupied,
#      while only fresh-water cells outside the legend are iterated and moved.
#    - Output checks can no longer scan every fresh-color pixel globally because the legend
#      corner fresh-color key is intentionally unstable-looking. Stability checks therefore
#      exclude the legend mask.
#
# d) Placement legality:
#    - The generator uses fixed 10x10 templates derived from valid base-task layouts, so the
#      basin geometry already respects bounds and non-overlap constraints.
#    - Only templates with a free 2x2 top corner are used, ensuring the legend never covers
#      structure or source water.
#    - Old-water carryover cells are selected from a solved basin state and then pasted back
#      into background basin cells only, so they are guaranteed to be in-bounds and stable.
#
# e) Visual observability:
#    - The repeated structure-color column inside the legend visually links the legend to the
#      basin walls. The outer column visibly contains the two water-role colors.
#    - Because fresh water changes position between input and output while old water does not,
#      the role distinction is inferable from examples without hidden numeric color ordering.
#    - Fresh water is never color 1 in generated samples, so solving by memorizing blue=water
#      visibly fails.
#
# f) Visual cleanliness:
#    - The scene has three clean regions: a small corner legend, one basin/wall family, and
#      liquid pixels. No extra separators or overlapping annotations are introduced.
#    - The legend occupies a free 2x2 top-corner patch, so it remains a compact, visually
#      separate cue rather than being woven into the basin walls.
#
# g) Change inferability / color discipline:
#    - Only three non-background colors appear, all repeated with clear roles.
#    - The legend provides the structural clue for color remapping, while multiple prefilled
#      old-water cells ensure the second water role is not a one-off accident.
#    - A degeneration guard rejects inputs where no fresh-water pixel would move, because that
#      would weaken the evidence that the legend color actually controls the simulation.
#
# 9) New solver assumptions guaranteed by generate_input():
#    - Exactly one top corner contains the 2x2 legend pattern, either top-left or top-right.
#    - The legend uses three distinct non-zero colors: fresh water, old water, and structure.
#    - Fresh-water source pixels outside the legend are all supported from below and never
#      have background on both horizontal sides, matching the original simulation precondition.
#    - Old-water carryover pixels outside the legend are already stable basin cells.
#    - Fresh-water colors come from {2..9} excluding 1, so the combo behavior does not
#      collapse back to the original hard-coded color-1 solver.


BG = 0
GRID_SHAPE = (10, 10)
COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


BASE_TEMPLATE_A = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 2, 1, 1],
        [0, 0, 0, 0, 0, 2, 0, 2, 1, 1],
        [0, 0, 0, 0, 0, 2, 0, 2, 1, 1],
        [0, 0, 0, 0, 2, 2, 0, 2, 1, 1],
        [0, 0, 0, 0, 2, 2, 0, 2, 1, 1],
        [0, 2, 2, 2, 2, 2, 2, 2, 1, 1],
    ],
    dtype=int,
)

BASE_TEMPLATE_B = np.array(
    [
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 2, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 2, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 2, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 2, 0, 0, 2, 1, 1, 1, 1],
        [0, 0, 2, 0, 0, 2, 2, 1, 1, 2],
        [0, 0, 2, 1, 1, 2, 2, 1, 1, 2],
        [0, 0, 2, 1, 1, 2, 2, 1, 1, 2],
        [0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    ],
    dtype=int,
)

TEMPLATES = (
    (BASE_TEMPLATE_A, "tl"),
    (np.fliplr(BASE_TEMPLATE_A).copy(), "tr"),
    (BASE_TEMPLATE_B, "tl"),
    (np.fliplr(BASE_TEMPLATE_B).copy(), "tr"),
)


def _legend_layout(shape: tuple[int, int], corner: str) -> dict[str, object]:
    h, w = map(int, shape)
    if corner == "tl":
        new_pos = (0, 0)
        old_pos = (1, 0)
        struct_positions = ((0, 1), (1, 1))
        padding: tuple[tuple[int, int], ...] = ()
    elif corner == "tr":
        new_pos = (0, w - 1)
        old_pos = (1, w - 1)
        struct_positions = ((0, w - 2), (1, w - 2))
        padding = ()
    else:
        raise AssertionError(f"unsupported legend corner: {corner}")

    mask = np.zeros(shape, dtype=bool)
    for r, c in (new_pos, old_pos, *struct_positions):
        mask[int(r), int(c)] = True
    return {
        "new_pos": new_pos,
        "old_pos": old_pos,
        "struct_positions": struct_positions,
        "padding": padding,
        "mask": mask,
    }


def _decode_legend(grid: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for corner in ("tl", "tr"):
        layout = _legend_layout(g.shape, corner)
        nr, nc = layout["new_pos"]
        orow, oc = layout["old_pos"]
        (s1r, s1c), (s2r, s2c) = layout["struct_positions"]
        fresh = int(g[int(nr), int(nc)])
        old = int(g[int(orow), int(oc)])
        struct_a = int(g[int(s1r), int(s1c)])
        struct_b = int(g[int(s2r), int(s2c)])
        if (
            fresh != BG
            and old != BG
            and struct_a == struct_b
            and struct_a != BG
            and len({fresh, old, struct_a}) == 3
            and all(int(g[int(r), int(c)]) == BG for r, c in layout["padding"])
        ):
            candidates.append((fresh, old, struct_a, layout["mask"]))
    assert len(candidates) == 1, "expected exactly one valid top-corner legend"
    return candidates[0]


def _fresh_mask(grid: np.ndarray, fresh: int, legend_mask: np.ndarray) -> np.ndarray:
    return (np.asarray(grid, dtype=int) == int(fresh)) & (~legend_mask)


def _assert_fresh_input(grid: np.ndarray, fresh: int, legend_mask: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    for r, c in np.argwhere(_fresh_mask(g, fresh, legend_mask)):
        rr = int(r)
        cc = int(c)
        assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "fresh input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
        right_bg = cc + 1 < w and int(g[rr, cc + 1]) == BG
        assert not (left_bg and right_bg), "fresh input water cannot have bg on both sides"


def _assert_old_stable(grid: np.ndarray, old: int, legend_mask: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    for r, c in np.argwhere((g == int(old)) & (~legend_mask)):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != BG, "old water must already be side-supported"
        assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "old water must already be floor-supported"
        assert cc == w - 1 or int(g[rr, cc + 1]) != BG, "old water must already be side-supported"


def _settle_fresh(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = map(int, g.shape)
    rr = int(r)
    cc = int(c)
    for _ in range(h * w):
        if rr + 1 < h and int(g[rr + 1, cc]) == BG:
            rr += 1
            continue

        options: list[tuple[int, int]] = []
        if rr + 1 < h and cc - 1 >= 0 and int(g[rr, cc - 1]) == BG and int(g[rr + 1, cc - 1]) == BG:
            options.append((rr + 1, cc - 1))
        if rr + 1 < h and cc + 1 < w and int(g[rr, cc + 1]) == BG and int(g[rr + 1, cc + 1]) == BG:
            options.append((rr + 1, cc + 1))
        if options:
            assert len(options) == 1, f"ambiguous diagonal moves: {options}"
            rr, cc = options[0]
            continue

        if rr + 1 < h and int(g[rr + 1, cc]) != BG:
            left = int(cc)
            while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                left -= 1
            right = int(cc)
            while right + 1 < w and int(g[rr, right + 1]) == BG:
                right += 1
            drops = [x for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == BG]
            if not drops:
                break
            min_dist = min(abs(int(x - cc)) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            target = int(nearest[0])
            assert target != int(cc)
            step = -1 if target < cc else 1
            nx = int(cc + step)
            assert int(g[rr, nx]) == BG
            cc = nx
            continue

        break
    else:
        raise AssertionError("fresh water pixel did not settle")
    return rr, cc


def _simulate(grid: np.ndarray, fresh: int, legend_mask: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = map(int, g.shape)
    for _pass in range(h * w):
        moved = False
        pts = sorted((int(r), int(c)) for r, c in np.argwhere(_fresh_mask(g, fresh, legend_mask)))
        pts.reverse()
        for r, c in pts:
            if int(g[r, c]) != int(fresh):
                continue
            g[r, c] = BG
            rr, cc = _settle_fresh(g, r, c)
            moved = moved or (rr, cc) != (r, c)
            g[rr, cc] = int(fresh)
        if not moved:
            return g
    raise AssertionError("fresh water simulation did not converge")


def _assert_fresh_output(grid: np.ndarray, fresh: int, legend_mask: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    for r, c in np.argwhere(_fresh_mask(g, fresh, legend_mask)):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != BG, "fresh output water must be left-supported"
        assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "fresh output water must be floor-supported"
        assert cc == w - 1 or int(g[rr, cc + 1]) != BG, "fresh output water must be right-supported"


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin filling after decoding the moving-water color from a corner legend.

    Input structure:
    - Background is 0 on a 10x10 grid.
    - One top-corner 2x2 legend identifies three roles: the corner cell is fresh moving
      water, the cell below it is old prefilled water, and the inward column is the solid
      basin color.
    - Outside the legend, fresh water starts in supported source stacks while old water may
      already occupy stable basin cells.

    Rule:
    - Read the fresh-water color from the legend, then simulate only those fresh-water pixels.
    - Fresh water falls through background, can slide diagonally through a unique open side,
      and can slide horizontally toward a unique reachable drop exactly as in the base task.
    - Old water and structures never move; they simply remain occupied cells that can support
      incoming fresh water. The legend itself also stays fixed.

    Key insight: The task has two coupled steps: first decode which color is the active water
    from the corner legend, then run the original settling physics while treating the other
    water color as already-stable liquid that must stay in place.
    """
    g = np.array(grid, dtype=int).copy()
    assert g.shape == GRID_SHAPE, "expected a 10x10 grid"
    fresh, old, _structure, legend_mask = _decode_legend(g)

    if check_input:
        _assert_fresh_input(g, fresh, legend_mask)
        _assert_old_stable(g, old, legend_mask)

    out = _simulate(g, fresh, legend_mask)

    if check_output:
        _assert_fresh_output(out, fresh, legend_mask)
        _assert_old_stable(out, old, legend_mask)

    return out


def _paint_legend(grid: np.ndarray, corner: str, fresh: int, old: int, structure: int) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    layout = _legend_layout(g.shape, corner)
    nr, nc = layout["new_pos"]
    orow, oc = layout["old_pos"]
    (s1r, s1c), (s2r, s2c) = layout["struct_positions"]
    for r, c in (layout["new_pos"], layout["old_pos"], *layout["struct_positions"]):
        assert int(g[int(r), int(c)]) == BG, "legend must be placed on a free corner"
    for r, c in layout["padding"]:
        assert int(g[int(r), int(c)]) == BG, "legend padding must remain background"
    g[int(nr), int(nc)] = int(fresh)
    g[int(orow), int(oc)] = int(old)
    g[int(s1r), int(s1c)] = int(structure)
    g[int(s2r), int(s2c)] = int(structure)
    return g


def _render_template(template: np.ndarray, fresh: int, structure: int) -> np.ndarray:
    g = np.full(template.shape, BG, dtype=int)
    g[np.asarray(template, dtype=int) == 1] = int(fresh)
    g[np.asarray(template, dtype=int) == 2] = int(structure)
    return g


def _carry_candidates(solved: np.ndarray, base_input: np.ndarray, fresh: int, legend_mask: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(_fresh_mask(solved, fresh, legend_mask)):
        rr = int(r)
        cc = int(c)
        if int(base_input[rr, cc]) != BG:
            continue
        if rr < 5:
            continue
        candidates.append((rr, cc))
    candidates.sort(key=lambda rc: (-rc[0], abs(rc[1] - 4), rc[1]))
    return candidates


def _is_valid_fresh_layout(grid: np.ndarray, fresh: int, legend_mask: np.ndarray) -> bool:
    try:
        _assert_fresh_input(grid, fresh, legend_mask)
    except AssertionError:
        return False
    return True


def _select_source_removals(grid: np.ndarray, fresh: int, legend_mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    working = np.array(grid, dtype=int).copy()
    removed: list[tuple[int, int]] = []
    while len(removed) < int(count):
        fresh_points = sorted((int(r), int(c)) for r, c in np.argwhere(_fresh_mask(working, fresh, legend_mask)))
        chosen = None
        for r, c in fresh_points:
            trial = working.copy()
            trial[r, c] = BG
            if not _is_valid_fresh_layout(trial, fresh, legend_mask):
                continue
            chosen = (r, c)
            working = trial
            removed.append(chosen)
            break
        if chosen is None:
            return None
    return removed


def _try_generate_input() -> np.ndarray | None:
    template, corner = random.choice(TEMPLATES)
    fresh, old, structure = random.sample(COLOR_POOL, 3)
    base = _render_template(template, fresh, structure)
    base = _paint_legend(base, corner, fresh, old, structure)
    fresh0, old0, _structure0, legend_mask = _decode_legend(base)
    assert fresh0 == fresh and old0 == old

    solved = main(base, check_input=True, check_output=True)
    carry_candidates = _carry_candidates(solved, base, fresh, legend_mask)
    source_count = int(np.count_nonzero(_fresh_mask(base, fresh, legend_mask)))
    max_carry = min(4, len(carry_candidates), source_count - 1)
    if max_carry < 2:
        return None

    carry_count = random.randint(2, max_carry)
    removals = _select_source_removals(base, fresh, legend_mask, carry_count)
    if removals is None:
        return None

    inp = base.copy()
    for r, c in removals:
        inp[int(r), int(c)] = BG
    for r, c in carry_candidates[:carry_count]:
        inp[int(r), int(c)] = int(old)

    try:
        out = main(inp, check_input=True, check_output=True)
    except AssertionError:
        return None

    if not np.any((inp == int(old)) & (~legend_mask)):
        return None
    if not np.any((out == int(fresh)) & (~legend_mask)):
        return None
    if not np.any((out == int(old)) & (~legend_mask)):
        return None
    if not np.any(inp != out):
        return None

    input_fresh = sorted((int(r), int(c)) for r, c in np.argwhere(_fresh_mask(inp, fresh, legend_mask)))
    output_fresh = sorted((int(r), int(c)) for r, c in np.argwhere(_fresh_mask(out, fresh, legend_mask)))
    if input_fresh == output_fresh:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _try_generate_input()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid combo sample")
