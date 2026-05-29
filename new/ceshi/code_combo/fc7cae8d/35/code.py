from __future__ import annotations

# Combo variant of fc7cae8d (index 35)
# combo_spec: D1/1+D3/1+D9/0
# Name: 颜色长度+非实心控制+图例
#
# Design analysis
# a) Concrete behavior and trace
#    - Original task: two solid frame-control edges advertise a horizontal control color and a
#      vertical control color. The inner rectangle carries two full guide strips whose colors
#      directly equal those control colors. Solver only aligns by dihedral transform and crops.
#    - Fused task here:
#      1. A tiny 2x2 legend sits in the inner top-left area. Each column is a guide->control
#         mapping: top cell = guide color, bottom cell = its control color.
#      2. Both selected frame-control sides now use the same control color, but only at 80-90%
#         density. Their measurable difference is the longest contiguous run of that dominant
#         control color on each side.
#      3. The object keeps two auxiliary outer guide strips, but each strip is only a partial run.
#         The two guide colors are different, yet the legend maps both to the same control color.
#      4. Therefore the solver must: decode legend first, then detect which border sides are the
#         dominant non-solid controls, then measure each side's longest dominant-color run, then
#         find the unique transform whose guide sides match both mapped color and run length.
#    - Concrete mini example:
#      legend = [[2, 3],
#                [7, 7]]
#      frame top: mostly 7 with gaps, longest 5; frame left: mostly 7 with gaps, longest 8.
#      object in one orientation shows top guide color 3 length 5 and left guide color 2 length 8.
#      Direct original logic fails because 2/3 != 7. Legend-decoded color-only logic says both
#      guides can target either selected side because both decode to 7. Length resolves the swap:
#      the len-5 guide must go to the len-5 control side and the len-8 guide to the len-8 side.
# b) Complexity calibration
#    - If the two controls kept different colors, length would be decorative. Mapping both guide
#      colors to one shared control color makes the length test necessary instead of cosmetic.
#    - If controls stayed solid, the D3 change would be invisible. The generator forces 80-90%
#      fill with a few background holes so the solver must identify the dominant color and then
#      measure a segment inside that broken edge.
#    - The legend has only two columns, enough to be human-readable. More rows would clutter the
#      already busy frame scene; fewer symbols would hide the mapping rule.
# c) Implicit-assumption audit against the original solver
#    - Original `detect_controls_frame()` assumed nearly solid full edges and direct colors.
#      Broken by both D3 and D9. New code measures dominant-color ratio plus longest run.
#    - Original `guide_edges()` assumed guide rows/cols are almost fully colored.
#      Broken by D1. New code looks for a single contiguous run on an otherwise background strip.
#    - Original matching assumed `guide_color == control_color`. Broken by D9 and deliberately
#      made ambiguous by mapping both guide colors to the same control color. New code decodes
#      legend first and then also checks the run length.
#    - Original `extract_center_object()` assumed the largest internal component is the object.
#      This still holds because the generator adds only one extra inner component: the tiny legend.
# d) Placement legality
#    - The legend is a fixed 2x2 component at an inner corner, never touching the border.
#    - The object is placed with `random_free_location(..., border=1, padding=1)` after the
#      legend has already been pasted, so it cannot overlap or touch the legend and cannot touch
#      the frame border.
#    - Guide runs stay away from object corners automatically because the guide rows/cols sit on
#      auxiliary strips offset from the perpendicular strip by one cell.
# e) Visual observability
#    - Across examples, humans can see: a corner legend, two same-colored but broken control edges,
#      and two differently colored guide runs on the object. The output is still the cropped core,
#      but the chosen orientation visibly depends on both the legend and the unequal run lengths.
# f) Visual cleanliness
#    - The legend is tiny and isolated in the inner top-left area.
#    - Non-selected frame sides carry only isolated grey noise cells, so they read as distractors
#      rather than extra controls.
#    - Control edges use one dominant color with a few holes, which is visually clean enough to
#      read as "same border signal, but broken".
# g) Color-change inferability
#    - Only three semantic color roles matter outside the core pattern: one shared control color,
#      two guide colors, and the fixed grey distractor color. The 2x2 legend explicitly exposes
#      the guide->control relation, so the color indirection is structurally signaled rather than
#      hidden in arbitrary recoloring.
#
# New solver assumptions introduced here, and how the generator guarantees them:
# - There is exactly one 2x2 internal legend component besides the main object.
# - Exactly two border sides have dominant non-bg ratio >= 0.8 and dominant-run length >= 4.
# - Those two selected sides are adjacent and share the same dominant control color.
# - The object has exactly two guide sides, each with one contiguous run on an otherwise bg strip.
# - The two guide lengths are distinct, and color-only matching after legend decoding is ambiguous
#   on purpose (typically >=2 transforms), while full color+length matching is unique.

from pathlib import Path
import math
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = {"top", "bottom"}
VERTICAL_SIDES = {"left", "right"}
NOISE_COLOR = 5
LEGEND_TOP = 2
LEGEND_LEFT = 2


def _side_values(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, 1:-1]
    if side == "bottom":
        return arr[-1, 1:-1]
    if side == "left":
        return arr[1:-1, 0]
    return arr[1:-1, -1]


def _object_edge_values(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, :]
    if side == "bottom":
        return arr[-1, :]
    if side == "left":
        return arr[:, 0]
    return arr[:, -1]


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _runs(values: list[int], bg: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    start: int | None = None
    color: int | None = None
    for idx, val in enumerate(values + [bg]):
        val = int(val)
        if val != int(bg):
            if start is None:
                start = int(idx)
                color = int(val)
            elif int(val) != int(color):
                out.append((int(color), int(start), int(idx - start)))
                start = int(idx)
                color = int(val)
            continue
        if start is not None:
            out.append((int(color), int(start), int(idx - start)))
            start = None
            color = None
    return out


def _longest_run_of_color(values: list[int], color: int) -> int:
    best = 0
    cur = 0
    for val in values:
        if int(val) == int(color):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _dominant_border_feature(values: np.ndarray, bg: int) -> tuple[int, int, int, float] | None:
    vals = list(map(int, values))
    non_bg = [v for v in vals if int(v) != int(bg)]
    if not non_bg:
        return None
    counts: dict[int, int] = {}
    for v in non_bg:
        counts[int(v)] = counts.get(int(v), 0) + 1
    color, count = max(counts.items(), key=lambda item: (int(item[1]), int(item[0])))
    longest = _longest_run_of_color(vals, int(color))
    ratio = float(count) / float(len(vals))
    return int(color), int(count), int(longest), float(ratio)


def _detect_controls_frame(arr: np.ndarray, bg: int) -> dict[str, tuple[int, int]]:
    candidates: list[tuple[float, int, int, str, int]] = []
    for side in SIDES:
        feature = _dominant_border_feature(_side_values(arr, side), int(bg))
        if feature is None:
            continue
        color, count, longest, ratio = feature
        if ratio < 0.78 or longest < 4:
            continue
        candidates.append((float(ratio), int(count), int(longest), side, int(color)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[4]), reverse=True)
    desired: dict[str, tuple[int, int]] = {}
    for _ratio, _count, longest, side, color in candidates[:2]:
        desired[side] = (int(color), int(longest))
    return desired


def _internal_components(arr: np.ndarray, bg: int) -> list[dict[str, object]]:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    out: list[dict[str, object]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        out.append(comp)
    return out


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    comps = _internal_components(arr, int(bg))
    if not comps:
        return np.array(arr, copy=True)
    best = max(comps, key=lambda comp: int(comp["area"]))
    return crop_bbox(arr, tuple(map(int, best["bbox"])))


def _extract_legend_map(arr: np.ndarray, bg: int) -> dict[int, int]:
    comps = _internal_components(arr, int(bg))
    if not comps:
        return {}
    object_comp = max(comps, key=lambda comp: int(comp["area"]))
    legend_candidates: list[tuple[int, int, dict[str, object]]] = []
    for comp in comps:
        if comp is object_comp:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if int(comp["area"]) != 4:
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        legend_candidates.append((int(r0), int(c0), comp))
    if not legend_candidates:
        return {}
    _r0, _c0, legend_comp = min(legend_candidates, key=lambda item: (item[0] + item[1], item[0], item[1]))
    patch = crop_bbox(arr, tuple(map(int, legend_comp["bbox"])))
    return {
        int(patch[0, 0]): int(patch[1, 0]),
        int(patch[0, 1]): int(patch[1, 1]),
    }


def _guide_features(arr: np.ndarray, bg: int) -> dict[str, tuple[int, int]]:
    H, W = arr.shape
    out: dict[str, tuple[int, int]] = {}
    for side in SIDES:
        line = list(map(int, _object_edge_values(arr, side)))
        non_bg = [idx for idx, val in enumerate(line) if int(val) != int(bg)]
        if not non_bg or len(non_bg) == len(line):
            continue
        color = int(line[non_bg[0]])
        if any(int(line[idx]) != int(color) for idx in non_bg):
            continue
        if non_bg != list(range(non_bg[0], non_bg[-1] + 1)):
            continue
        if len(non_bg) < 2:
            continue
        if side == "top":
            if H < 2 or any(int(arr[1, idx]) == int(bg) for idx in non_bg):
                continue
        elif side == "bottom":
            if H < 2 or any(int(arr[-2, idx]) == int(bg) for idx in non_bg):
                continue
        elif side == "left":
            if W < 2 or any(int(arr[idx, 1]) == int(bg) for idx in non_bg):
                continue
        else:
            if W < 2 or any(int(arr[idx, -2]) == int(bg) for idx in non_bg):
                continue
        out[side] = (int(color), int(len(non_bg)))
    return out


def _crop_aligned_object(arr: np.ndarray, desired_sides: set[str]) -> np.ndarray:
    top = 1 if "top" in desired_sides else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in desired_sides else 0)
    left = 1 if "left" in desired_sides else 0
    right = int(arr.shape[1]) - (1 if "right" in desired_sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend, read the broken frame controls by dominant color + longest run,
    align the object by mapped color + length, then crop off the guide strips.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend_map = _extract_legend_map(g, int(bg))
    desired = _detect_controls_frame(g, int(bg))
    obj = _extract_center_object(g, int(bg))

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            ok = True
            for side, (ctrl_color, ctrl_len) in desired.items():
                guide_color, guide_len = features[side]
                if legend_map.get(int(guide_color)) != int(ctrl_color):
                    ok = False
                    break
                if int(guide_len) != int(ctrl_len):
                    ok = False
                    break
            if ok:
                return _crop_aligned_object(t, set(desired))
    return obj


def _is_visibly_asymmetric(arr: np.ndarray) -> bool:
    for k in range(4):
        for flip in (False, True):
            if k == 0 and not flip:
                continue
            t = _transform(arr, int(k), bool(flip))
            if t.shape == arr.shape and np.array_equal(t, arr):
                return False
    return True


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    area = int(h * w)
    max_nonfill = max(len(pattern_colors) + 6, int(area * 0.35))
    min_nonfill = max(len(pattern_colors) + 4, int(area * 0.12))
    for _ in range(160):
        core = np.full((h, w), int(fill), dtype=int)
        target_nonfill = random.randint(min_nonfill, max_nonfill)
        painted = 0

        for color in pattern_colors:
            r = random.randrange(h)
            c = random.randrange(w)
            if int(core[r, c]) == int(fill):
                painted += 1
            core[r, c] = int(color)

        while painted < target_nonfill:
            color = random.choice(pattern_colors)
            shape_kind = random.choice(["dot", "line", "rect"])
            if shape_kind == "dot":
                cells = [(random.randrange(h), random.randrange(w))]
            elif shape_kind == "line":
                line_len = random.randint(2, min(5, max(h, w)))
                if random.random() < 0.5:
                    rr = random.randrange(h)
                    c0 = random.randrange(max(1, w - line_len + 1))
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(max(1, h - line_len + 1))
                    cc = random.randrange(w)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            else:
                mh = random.randint(1, min(3, h))
                mw = random.randint(1, min(3, w))
                r0 = random.randrange(h - mh + 1)
                c0 = random.randrange(w - mw + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

            for r, c in cells:
                if painted >= target_nonfill:
                    break
                if int(core[r, c]) == int(fill):
                    core[r, c] = int(color)
                    painted += 1

        marker_color = random.choice(pattern_colors)
        corner = random.choice(
            [
                (0, 0),
                (0, max(0, w - 2)),
                (max(0, h - 2), 0),
                (max(0, h - 2), max(0, w - 2)),
            ]
        )
        mr, mc = corner
        for r, c in [(mr, mc), (min(h - 1, mr + 1), mc), (min(h - 1, mr + 1), min(w - 1, mc + 1))]:
            core[r, c] = int(marker_color)

        if _is_visibly_asymmetric(core):
            return core
    raise RuntimeError("failed to build asymmetric core")


def _attach_guides(
    core: np.ndarray,
    horiz_side: str,
    vert_side: str,
    guide_h_color: int,
    guide_v_color: int,
    horiz_len: int,
    vert_len: int,
) -> np.ndarray:
    h, w = core.shape
    has_top = horiz_side == "top"
    has_bottom = horiz_side == "bottom"
    has_left = vert_side == "left"
    has_right = vert_side == "right"
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        start = c0 + random.randint(0, w - horiz_len)
        out[0, int(start) : int(start + horiz_len)] = int(guide_h_color)
    if has_bottom:
        start = c0 + random.randint(0, w - horiz_len)
        out[-1, int(start) : int(start + horiz_len)] = int(guide_h_color)
    if has_left:
        start = r0 + random.randint(0, h - vert_len)
        out[int(start) : int(start + vert_len), 0] = int(guide_v_color)
    if has_right:
        start = r0 + random.randint(0, h - vert_len)
        out[int(start) : int(start + vert_len), -1] = int(guide_v_color)

    return out


def _partition_sum(total: int, parts: int, cap: int) -> list[int] | None:
    if parts <= 0:
        return [] if total == 0 else None
    if total < parts or total > parts * cap:
        return None
    out = [1] * parts
    remain = int(total - parts)
    slots = list(range(parts))
    while remain > 0:
        random.shuffle(slots)
        advanced = False
        for idx in slots:
            room = int(cap - out[idx])
            if room <= 0:
                continue
            add = random.randint(1, min(room, remain))
            out[idx] += int(add)
            remain -= int(add)
            advanced = True
            if remain <= 0:
                break
        if not advanced:
            return None
    return out


def _broken_control_line(length: int, color: int, longest_run: int) -> np.ndarray | None:
    min_count = int(math.ceil(0.80 * length))
    max_count = int(math.floor(0.90 * length))
    if max_count < min_count or longest_run < 4:
        return None
    for _ in range(200):
        target_count = random.randint(min_count, max_count)
        holes = int(length - target_count)
        if holes < 1:
            continue
        max_runs = min(4, holes + 1)
        run_count_choices = [rc for rc in range(2, max_runs + 1)]
        random.shuffle(run_count_choices)
        for run_count in run_count_choices:
            other_total = int(target_count - longest_run)
            others = _partition_sum(other_total, run_count - 1, max(longest_run - 1, 1))
            if others is None:
                continue
            segments = [int(longest_run)] + [int(x) for x in others]
            random.shuffle(segments)
            base_internal = [1] * (run_count - 1)
            gap_extra = int(holes - (run_count - 1))
            extra_bins = [0] * (run_count + 1)
            for _j in range(gap_extra):
                extra_bins[random.randrange(run_count + 1)] += 1
            gaps = [extra_bins[0]]
            for idx in range(run_count - 1):
                gaps.append(int(base_internal[idx] + extra_bins[idx + 1]))
            gaps.append(extra_bins[-1])
            line: list[int] = []
            for idx, seg in enumerate(segments):
                line.extend([0] * int(gaps[idx]))
                line.extend([int(color)] * int(seg))
            line.extend([0] * int(gaps[-1]))
            if len(line) != int(length):
                continue
            if _longest_run_of_color(line, int(color)) != int(longest_run):
                continue
            if sum(1 for v in line if int(v) == int(color)) != int(target_count):
                continue
            return np.array(line, dtype=int)
    return None


def _place_isolated_noise(line: np.ndarray, noise_color: int) -> None:
    if line.size <= 0:
        return
    available = list(range(int(line.size)))
    random.shuffle(available)
    count = random.randint(max(1, line.size // 6), max(1, line.size // 3))
    chosen: list[int] = []
    for idx in available:
        if any(abs(int(idx) - int(prev)) <= 1 for prev in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= count:
            break
    for idx in chosen:
        line[int(idx)] = int(noise_color)


def _paint_frame(
    grid: np.ndarray,
    horiz_side: str,
    vert_side: str,
    ctrl_color: int,
    horiz_len: int,
    vert_len: int,
) -> bool:
    n = int(grid.shape[0])
    horiz = _broken_control_line(n - 2, int(ctrl_color), int(horiz_len))
    vert = _broken_control_line(n - 2, int(ctrl_color), int(vert_len))
    if horiz is None or vert is None:
        return False

    if horiz_side == "top":
        grid[0, 1:-1] = horiz
    else:
        grid[-1, 1:-1] = horiz

    if vert_side == "left":
        grid[1:-1, 0] = vert
    else:
        grid[1:-1, -1] = vert

    for side in SIDES:
        if side in (horiz_side, vert_side):
            continue
        if side == "top":
            _place_isolated_noise(grid[0, 1:-1], NOISE_COLOR)
        elif side == "bottom":
            _place_isolated_noise(grid[-1, 1:-1], NOISE_COLOR)
        elif side == "left":
            _place_isolated_noise(grid[1:-1, 0], NOISE_COLOR)
        else:
            _place_isolated_noise(grid[1:-1, -1], NOISE_COLOR)

    grid[0, 0] = 0
    grid[0, -1] = 0
    grid[-1, 0] = 0
    grid[-1, -1] = 0
    return True


def _legend_patch(guide_h: int, guide_v: int, ctrl_color: int) -> np.ndarray:
    return np.array([[int(guide_h), int(guide_v)], [int(ctrl_color), int(ctrl_color)]], dtype=int)


def _desired_features_from_grid(grid: np.ndarray) -> dict[str, tuple[int, int]]:
    return _detect_controls_frame(np.array(grid, dtype=int), int(get_bg(grid)))


def _count_color_only_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend = _extract_legend_map(g, int(bg))
    desired = _desired_features_from_grid(g)
    obj = _extract_center_object(g, int(bg))
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(legend.get(int(features[side][0])) == int(desired[side][0]) for side in desired):
                count += 1
    return int(count)


def _count_direct_color_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _desired_features_from_grid(g)
    obj = _extract_center_object(g, int(bg))
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(int(features[side][0]) == int(desired[side][0]) for side in desired):
                count += 1
    return int(count)


def _count_full_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend = _extract_legend_map(g, int(bg))
    desired = _desired_features_from_grid(g)
    obj = _extract_center_object(g, int(bg))
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            good = True
            for side, (ctrl_color, ctrl_len) in desired.items():
                guide_color, guide_len = features[side]
                if legend.get(int(guide_color)) != int(ctrl_color) or int(guide_len) != int(ctrl_len):
                    good = False
                    break
            if good:
                count += 1
    return int(count)


def variant_metrics(grid: np.ndarray) -> dict[str, float]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _desired_features_from_grid(g)
    legend = _extract_legend_map(g, int(bg))
    obj = _extract_center_object(g, int(bg))
    guide_lengths = sorted(int(length) for _side, (_color, length) in _guide_features(obj, int(bg)).items())
    ctrl_lengths = sorted(int(length) for _side, (_color, length) in desired.items())
    fill_ratios = []
    for side in desired:
        feature = _dominant_border_feature(_side_values(g, side), int(bg))
        if feature is None:
            continue
        _color, _count, _longest, ratio = feature
        fill_ratios.append(float(ratio))
    mapped_targets = sorted(legend.values())
    return {
        "legend_size": float(len(legend)),
        "direct_color_matches": float(_count_direct_color_matches(g)),
        "mapped_color_only_matches": float(_count_color_only_matches(g)),
        "full_rule_matches": float(_count_full_matches(g)),
        "guide_length_gap": float(abs(guide_lengths[0] - guide_lengths[1])) if len(guide_lengths) == 2 else 0.0,
        "ctrl_length_gap": float(abs(ctrl_lengths[0] - ctrl_lengths[1])) if len(ctrl_lengths) == 2 else 0.0,
        "min_ctrl_fill_ratio": float(min(fill_ratios)) if fill_ratios else 0.0,
        "max_ctrl_fill_ratio": float(max(fill_ratios)) if fill_ratios else 0.0,
        "shared_control_color": float(int(len(set(mapped_targets)) == 1 and len(mapped_targets) == 2)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(28, 34)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    semantic_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    ctrl_color = random.choice(semantic_colors)
    remaining = [c for c in semantic_colors if c != int(ctrl_color)]
    guide_h, guide_v = random.sample(remaining, 2)
    remaining = [c for c in remaining if c not in {int(guide_h), int(guide_v)}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != int(fill)]
    pattern_colors = random.sample(remaining, random.randint(2, min(3, len(remaining))))

    max_core_h = min(16, n - 8)
    max_core_w = min(16, n - 8)
    if max_core_h < 8 or max_core_w < 8:
        return None

    for _ in range(50):
        core_h = random.randint(8, max_core_h)
        core_w = random.randint(8, max_core_w)
        if core_h != core_w:
            break
    else:
        return None

    max_horiz_len = min(core_w - 1, 12)
    max_vert_len = min(core_h - 1, 12)
    if max_horiz_len < 4 or max_vert_len < 4:
        return None

    horiz_len = random.randint(max(4, max_horiz_len - 4), max_horiz_len)
    vert_len = random.randint(max(4, max_vert_len - 4), max_vert_len)
    if horiz_len == vert_len:
        if vert_len < max_vert_len:
            vert_len += 1
        elif horiz_len > 4:
            horiz_len -= 1
        else:
            return None
    if horiz_len == vert_len:
        return None

    core = _build_core(core_h, core_w, int(fill), list(map(int, pattern_colors)))
    aligned_object = _attach_guides(
        core,
        horiz_side,
        vert_side,
        int(guide_h),
        int(guide_v),
        int(horiz_len),
        int(vert_len),
    )

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, int(k), bool(flip))

    grid = np.zeros((n, n), dtype=int)
    if not _paint_frame(grid, horiz_side, vert_side, int(ctrl_color), int(horiz_len), int(vert_len)):
        return None

    legend = _legend_patch(int(guide_h), int(guide_v), int(ctrl_color))
    paste(grid, legend, LEGEND_TOP, LEGEND_LEFT, transparent=0)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1, padding=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    metrics = variant_metrics(grid)
    if int(metrics["legend_size"]) != 2:
        return None
    if int(metrics["shared_control_color"]) != 1:
        return None
    if int(metrics["direct_color_matches"]) != 0:
        return None
    if int(metrics["mapped_color_only_matches"]) < 2:
        return None
    if int(metrics["full_rule_matches"]) != 1:
        return None
    if metrics["guide_length_gap"] <= 0 or metrics["ctrl_length_gap"] <= 0:
        return None
    if metrics["min_ctrl_fill_ratio"] < 0.80 or metrics["max_ctrl_fill_ratio"] > 0.90:
        return None
    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 35")
