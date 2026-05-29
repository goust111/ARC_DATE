# Combo variant of 28a6681f (index 25)
# combo_spec: D7/2+D23/1
# Name: 斜面反射+带孔壁面

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) Concrete fused behavior and traced example
#    This family keeps the original 10x10 basin chain from 28a6681f in the lower half,
#    but adds one upper staircase guide that enters from the left or right edge. The lower
#    half still contains three bottom-anchored connected basins:
#    - the upstream basin sits on the guide side;
#    - the middle basin is separated by a taller divider that contains one one-cell hole;
#    - the far basin is separated by a shorter solid wall that only allows top overflow.
#    The fused path is intentionally serial:
#    1. input water is packed above the staircase guide rather than already inside the basin;
#    2. the guide orientation determines which side the source water is dispatched into;
#    3. once that water enters the lower basin system, it follows the original fall/slide
#       settling law, so it can leak through the divider hole into the neighboring basin and
#       later overflow the lower crest into the far basin.
#    Concrete left-guide trace:
#    - the guide touches the left edge and slopes down-right into the basin area;
#    - 16 source pixels are stacked in the upper-left source reservoir above the guide;
#    - under the old full-grid local solver, the upper source arrangement is not enough to
#      choose a unique route and the sample either fails or behaves differently;
#    - under the fused rule, those 16 pixels first slide along the left-entering staircase,
#      so they are injected into the left upstream basin one by one;
#    - the lower solver then reproduces the D7 behavior: the waterline reaches the divider
#      hole, some pixels cross that hole into the middle basin, and further volume can spill
#      over the shorter downstream wall into the far basin.
#
# b) Complexity calibration
#    The family is deliberately narrow:
#    - one fixed staircase-guide template, mirrored left/right;
#    - one fixed three-basin layout skeleton with a single holed divider and one solid overflow wall;
#    - a small set of validated wall-height and water-count profiles inherited from the original task.
#    This keeps the two cues repeated across examples:
#    - the guide side really matters because it chooses the source-entry side;
#    - the divider hole really matters because the middle basin can fill before the taller wall top.
#    Freedom is limited to profile choice, mirror, carryover depth, right-wall height, and colors,
#    which is enough visual variation without making the family noisy.
#
# c) Implicit-assumption audit for main()
#    - The original solver assumed every water pixel is already inside the main play area and can be
#      handled by one uniform local settle loop. That breaks here because the upper staircase is a
#      routing device: before normal basin settling starts, water above that guide must first be
#      interpreted as sliding to the guide side.
#    - The original hole behavior is still valid once water is in the lower basin chain, because a
#      one-cell wall opening is still just a background gap inside a supported row. No new lower-half
#      mechanics are needed there.
#    - The new solver therefore splits the task into two stages:
#      1. detect the unique upper staircase component and count the source water stacked above it;
#      2. run the original lower settle law on a guide-free lower grid while injecting that many
#         source pixels from the guide side edge.
#    - This introduces a new assumption: all moving source water lives above the guide base row, while
#      any lower water in the input is already stable carryover. The generator guarantees exactly that.
#
# d) Placement legality
#    The generator never places source water on top of basin-wall pixels or guide pixels. Source slots
#    are computed only from background cells in rows 0..4 on the guide side, and they are filled
#    bottom-up so every input water pixel is supported by either guide structure or another water cell.
#    The lower basin terrain is still fully inside the 10x10 grid, with the same legal wall columns
#    and hole rows as the D7 single-variant reference.
#
# e) Visual observability
#    A human solver can infer the fusion from a few examples because both cues recur directly:
#    - the input always shows a staircase guide attached to one edge and a large source pile resting
#      above it, while the output removes that upper water entirely and leaves only lower basin water;
#    - the lower divider visibly contains a one-cell hole, and the output often shows middle-basin
#      water even when the final top row stays below the divider crest, so side leakage is required.
#    The far basin can also receive water, but only after the shorter downstream wall is overtopped,
#    which keeps hole leakage and top overflow visually distinct.
#
# f) Visual cleanliness
#    The image stays readable because the upper guide/source area and lower basin area occupy separate
#    bands. There are only background, blue water, and three structure colors: one for the upstream
#    basin wall, one for the downstream walls, and one dedicated guide color in the upper band.
#
# g) Inferability of the change
#    No hidden color ordering is introduced. Direction comes from the visible staircase orientation,
#    and leakage comes from the visible hole location. The same lower geometry appears in both mirror
#    directions, so the learnable rule is not "left means X color" but "water above the staircase
#    first follows the staircase side, then the holed divider can leak into the next basin."
#
# 9) New solver assumptions audited against the generator
#    - There is exactly one upper staircase guide component that touches one side edge and does not
#      reach the bottom row. The generator always builds exactly one such component.
#    - All source water is stored above the guide base row, on the same side as the guide entry.
#      The generator computes source slots from that region and validates support.
#    - The lower basin system is still an ordinary 28a6681f-style terrain after removing the guide,
#      so the original fall/diagonal/horizontal settle loop remains valid there.
#    - Every accepted sample is re-run through both the fused solver and a legacy no-dispatch solver;
#      samples are kept only when the fused solver uses guide dispatch and hole crossing, and the
#      legacy behavior differs or fails.


H = 10
W = 10
BG = 0
WATER = 1
GUIDE_TEMPLATE = (
    (2, 0),
    (2, 1),
    (3, 1),
    (3, 2),
    (4, 2),
    (5, 0),
    (5, 1),
    (5, 2),
)
PROFILES = (
    {"w1_top": 3, "hole_row": 5, "w2_top": 6, "water_choices": (17, 19), "retain_choices": (2,)},
    {"w1_top": 4, "hole_row": 5, "w2_top": 6, "water_choices": (17, 19), "retain_choices": (2,)},
)
RIGHT_WALL_TOPS = (3, 4, 5)
OBSTACLE_COLORS = tuple(range(2, 10))
MAX_RETRIES = 300


def _solid_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if bool(seen[rr, cc]):
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((int(cr), int(cc2)))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(cr + dr)
                nc = int(cc2 + dc)
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _guide_cells(mirror: bool) -> list[tuple[int, int]]:
    if mirror:
        return [(int(r), int(W - 1 - c)) for r, c in GUIDE_TEMPLATE]
    return [(int(r), int(c)) for r, c in GUIDE_TEMPLATE]


def _find_guide_component(base: np.ndarray) -> dict[str, int | list[tuple[int, int]]] | None:
    bottom_colors = set(map(int, base[H - 1]))
    candidates = []
    for color in sorted(set(map(int, base.flatten())) - {int(BG), int(WATER)} - bottom_colors):
        comp = [(int(r), int(c)) for r, c in np.argwhere(base == int(color))]
        if not comp:
            continue
        rows = [int(r) for r, _c in comp]
        cols = [int(c) for _r, c in comp]
        if any(r == H - 1 for r in rows):
            continue
        touches_left = any(c == 0 for c in cols)
        touches_right = any(c == W - 1 for c in cols)
        if not (touches_left or touches_right):
            continue
        comp_set = set(comp)
        diag_links = sum(1 for r, c in comp if (r + 1, c - 1) in comp_set or (r + 1, c + 1) in comp_set)
        if diag_links < 3:
            continue
        side = -1 if touches_left else 1
        candidates.append(
            {
                "cells": [(int(r), int(c)) for r, c in comp],
                "side": int(side),
                "bottom": int(max(rows)),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: len(item["cells"]), reverse=True)
    return candidates[0]


def _strip_water(grid: np.ndarray) -> np.ndarray:
    base = np.array(grid, dtype=int).copy()
    base[base == int(WATER)] = int(BG)
    return base


def _remove_guide(base: np.ndarray, guide: dict[str, int | list[tuple[int, int]]] | None) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    if guide is None:
        return out
    for r, c in guide["cells"]:
        out[int(r), int(c)] = int(BG)
    return out


def _find_holes(base: np.ndarray) -> set[tuple[int, int]]:
    holes: set[tuple[int, int]] = set()
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(base[r, c]) != int(BG):
                continue
            if int(base[r - 1, c]) == int(BG) or int(base[r + 1, c]) == int(BG):
                continue
            if int(base[H - 1, c]) == int(BG):
                continue
            holes.add((int(r), int(c)))
    return holes


def _lower_settle(
    g: np.ndarray,
    r: int,
    c: int,
    hole_cells: set[tuple[int, int]],
    metrics: dict[str, int] | None,
) -> tuple[int, int]:
    rr = int(r)
    cc = int(c)
    for _ in range(H * W):
        if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
            rr += 1
            if metrics is not None:
                metrics["falls"] += 1
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
            if metrics is not None:
                metrics["diag_steps"] += 1
            continue

        if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
            left = int(cc)
            while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                left -= 1
            right = int(cc)
            while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                right += 1

            drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == int(BG)]
            if not drops:
                break

            min_dist = min(abs(int(x) - cc) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x) - cc) == int(min_dist)]
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            best = nearest[0]
            assert best != int(cc)
            step = -1 if best < int(cc) else 1
            nx = int(cc + step)
            assert int(g[rr, nx]) == int(BG)
            if metrics is not None:
                metrics["horiz_steps"] += 1
                if (int(rr), int(nx)) in hole_cells:
                    metrics["hole_crossings"] += 1
            cc = nx
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")
    return int(rr), int(cc)


def _simulate_lower(
    grid: np.ndarray,
    check_input: bool,
    check_output: bool,
    track_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    hole_cells = _find_holes(np.where(g == int(WATER), int(BG), g))
    metrics = {"falls": 0, "diag_steps": 0, "horiz_steps": 0, "hole_crossings": 0}

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = _lower_settle(g, int(r), int(c), hole_cells, metrics if track_metrics else None)
            moved = bool(moved or ((rr, cc) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
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

    if track_metrics:
        return g, metrics
    return g


def _simulate_combo(
    grid: np.ndarray,
    check_input: bool,
    check_output: bool,
    track_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    base = _strip_water(g)
    guide = _find_guide_component(base)
    assert guide is not None, "expected one upper staircase guide"
    guide_bottom = int(guide["bottom"])
    guide_side = int(guide["side"])
    source_cols = range(0, 5) if guide_side < 0 else range(5, W)

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"
            if rr < guide_bottom:
                assert cc in source_cols, "upper source water must sit on the guide side"

    guide_free_base = _remove_guide(base, guide)
    lower = np.array(guide_free_base, dtype=int).copy()
    source_count = 0
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        if rr < guide_bottom and cc in source_cols:
            source_count += 1
        else:
            lower[rr, cc] = int(WATER)

    total_water = int(np.count_nonzero(g == int(WATER)))
    metrics = {"guide_dispatches": int(source_count), "source_water": int(source_count), "hole_crossings": 0, "falls": 0, "diag_steps": 0, "horiz_steps": 0}
    injection = (0, 0) if guide_side < 0 else (0, W - 1)
    for _ in range(int(source_count)):
        lower[injection] = int(WATER)
        if track_metrics:
            lower, step_metrics = _simulate_lower(lower, check_input=False, check_output=False, track_metrics=True)
            for key in ("hole_crossings", "falls", "diag_steps", "horiz_steps"):
                metrics[key] += int(step_metrics[key])
        else:
            lower = _simulate_lower(lower, check_input=False, check_output=False, track_metrics=False)

    out = np.array(lower, dtype=int).copy()
    for r, c in guide["cells"]:
        out[int(r), int(c)] = int(base[int(r), int(c)])

    if check_output:
        assert int(np.count_nonzero(out == int(WATER))) == int(total_water), "water count must be conserved"
        assert not np.any(out[:guide_bottom, list(source_cols)] == int(WATER)), "all source water must leave the upper guide area"
        lower_out = _remove_guide(out, guide)
        stable_lower = _simulate_lower(lower_out, check_input=True, check_output=True, track_metrics=False)
        assert np.array_equal(stable_lower, lower_out), "lower basin water must already be stable"
        solved_again = _simulate_combo(out, check_input=True, check_output=False, track_metrics=False)
        assert np.array_equal(solved_again, out), "output must be a fixed point of the fused solver"

    if track_metrics:
        return out, metrics
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Route upper source water along a staircase wall, then settle it through leaky chained basins.

    Input structure:
    - Background is 0 and water is color 1.
    - The lower half contains the usual bottom-anchored basin chain, but the first divider has a
      visible one-pixel hole and the next divider is a shorter solid overflow wall.
    - The upper half contains one staircase guide attached to the left or right edge; the moving
      source water is packed above that guide, while the lower half may keep a little stable carryover.

    Rule:
    - First interpret every water pixel above the guide base row as source water that slides along
      the staircase direction into the matching side of the lower basin system.
    - After that handoff, use the original gravity law in the lower half: fall straight down through
      background, take a unique open diagonal if needed, and slide along a supported row toward the
      unique reachable drop. The wall hole is just another background gap inside that supported row.
    - Inject the source water one pixel at a time from the guide side edge, repeat the lower settling
      after each injection, and stop when the whole lower configuration is stable.

    Key insight: The guide and the hole change different stages of the same path. The staircase decides
    which basin receives the incoming source water, and only then can the holed divider leak that water
    into the neighboring basin before any later crest overflow.
    """
    return _simulate_combo(
        grid=np.array(grid, dtype=int),
        check_input=check_input,
        check_output=check_output,
        track_metrics=False,
    )


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _simulate_lower(np.array(grid, dtype=int), check_input=True, check_output=True, track_metrics=False)


def _build_lower_base(
    profile: dict[str, int | tuple[int, ...]],
    right_wall_top: int,
    colors: tuple[int, int, int],
    mirror: bool,
) -> np.ndarray:
    left_color, right_color, _guide_color = map(int, colors)
    g = np.zeros((H, W), dtype=int)
    for c in range(0, 4):
        g[9, c] = left_color
    for c in range(4, 7):
        g[9, c] = right_color
    for c in range(7, 10):
        g[9, c] = right_color

    for r in range(int(profile["w1_top"]), 9):
        if r != int(profile["hole_row"]):
            g[r, 3] = left_color
    for r in range(int(profile["w2_top"]), 9):
        g[r, 6] = right_color
    for r in range(int(right_wall_top), 9):
        g[r, 9] = right_color

    if mirror:
        g = np.fliplr(g)
    return g


def _guide_color(colors: tuple[int, int, int], mirror: bool) -> int:
    return int(colors[2])


def _paste_guide(grid: np.ndarray, mirror: bool, color: int) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    for r, c in _guide_cells(mirror):
        g[int(r), int(c)] = int(color)
    return g


def _source_slots(lower_base: np.ndarray, mirror: bool) -> list[tuple[int, int]]:
    blocked = {(int(r), int(c)) for r, c in np.argwhere(lower_base != int(BG))}
    blocked.update(_guide_cells(mirror))
    slots: list[tuple[int, int]] = []
    occupied = set(blocked)
    cols = list(range(5, W)) if mirror else list(range(0, 5))
    if mirror:
        cols = list(reversed(cols))
    for r in range(4, -1, -1):
        for c in cols:
            cell = (int(r), int(c))
            if cell in occupied:
                continue
            below = (int(r + 1), int(c))
            if below not in occupied:
                continue
            left_ok = c == 0 or (int(r), int(c - 1)) in occupied
            right_ok = c == W - 1 or (int(r), int(c + 1)) in occupied
            if not (left_ok or right_ok):
                continue
            slots.append(cell)
            occupied.add(cell)
    return slots


def _pour_lower(lower_base: np.ndarray, source: tuple[int, int], water_count: int) -> np.ndarray:
    out = np.array(lower_base, dtype=int).copy()
    sr = int(source[0])
    sc = int(source[1])
    for _ in range(int(water_count)):
        assert int(out[sr, sc]) == int(BG)
        out[sr, sc] = int(WATER)
        out = _simulate_lower(out, check_input=False, check_output=False, track_metrics=False)
    return out


def _variant_visible(output: np.ndarray, hole_row: int, mirror: bool) -> bool:
    water_pts = np.argwhere(output == int(WATER))
    if water_pts.size == 0:
        return False
    top_water_row = int(water_pts[:, 0].min())
    middle_has = bool(np.any(output[:, 4:6] == int(WATER)))
    far_has = bool(np.any(output[:, 1:3] == int(WATER))) if mirror else bool(np.any(output[:, 7:9] == int(WATER)))
    hole_col = 6 if mirror else 3
    hole_open = int(output[int(hole_row), int(hole_col)]) == int(BG)
    return bool(middle_has and far_has and hole_open and top_water_row > int(hole_row))


def _build_candidate(
    profile: dict[str, int | tuple[int, ...]],
    right_wall_top: int,
    water_count: int,
    retain_rows: int,
    mirror: bool,
    colors: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    lower_base = _build_lower_base(profile, right_wall_top, colors, mirror)
    slots = _source_slots(lower_base, mirror)
    guide_color = _guide_color(colors, mirror)
    source = (0, W - 1) if mirror else (0, 0)

    try:
        output_lower = _pour_lower(lower_base, source, water_count)
    except AssertionError:
        return None
    output = _paste_guide(output_lower, mirror, guide_color)
    if not _variant_visible(output, int(profile["hole_row"]), mirror):
        return None

    input_grid = np.array(output, dtype=int).copy()
    carry_cols = range(7, 10) if mirror else range(0, 3)
    retain = {
        (int(r), int(c))
        for r, c in np.argwhere(output_lower == int(WATER))
        if int(r) >= 9 - int(retain_rows) and int(c) in carry_cols
    }

    moved: list[tuple[int, int]] = []
    for r, c in np.argwhere(output_lower == int(WATER)):
        cell = (int(r), int(c))
        if cell in retain:
            continue
        moved.append(cell)
        input_grid[cell] = int(BG)

    if len(moved) > len(slots):
        return None
    for r, c in slots[: len(moved)]:
        input_grid[int(r), int(c)] = int(WATER)

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        profile = random.choice(PROFILES)
        right_wall_top = int(random.choice(RIGHT_WALL_TOPS))
        water_count = int(random.choice(profile["water_choices"]))
        retain_rows = int(random.choice(profile["retain_choices"]))
        mirror = bool(random.choice((False, True)))
        colors = tuple(int(x) for x in random.sample(OBSTACLE_COLORS, 3))

        built = _build_candidate(
            profile=profile,
            right_wall_top=right_wall_top,
            water_count=water_count,
            retain_rows=retain_rows,
            mirror=mirror,
            colors=colors,
        )
        if built is None:
            continue
        input_grid, output = built

        try:
            solved, metrics = _simulate_combo(input_grid, check_input=True, check_output=True, track_metrics=True)
        except AssertionError:
            continue
        if not np.array_equal(solved, output):
            continue
        if np.array_equal(input_grid, output):
            continue
        if int(metrics["guide_dispatches"]) <= 0:
            continue
        if int(metrics["hole_crossings"]) <= 0:
            continue

        same_as_legacy = False
        try:
            legacy_out = _legacy_main(input_grid)
            same_as_legacy = bool(np.array_equal(legacy_out, output))
        except Exception:
            same_as_legacy = False
        if same_as_legacy:
            continue

        return input_grid

    raise AssertionError("failed to generate a valid fused guide-and-hole example")
