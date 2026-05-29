from __future__ import annotations

# Combo variant of fc7cae8d (index 4)
# combo_spec: D1/1+D3/1
# Name: 颜色长度匹配+非实心控制边
#
# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Original task: two adjacent frame borders were solid control edges, and the inner object
#      had two full guide strips whose colors directly identified which sides to keep after a
#      dihedral transform.
#    - This combo changes both ends of that pipeline. The chosen frame sides are still adjacent,
#      but each is only 80-90% filled by one dominant control color with several bg gaps. The
#      side signal is therefore no longer "solid edge", but "edge with overwhelmingly dominant
#      color". On each such side, the measurable feature is the longest contiguous run of that
#      dominant color.
#    - The inner object still uses removable outer guide rows/cols, but each guide is only one
#      partial contiguous run rather than a full side. Both guides share the same color, so color
#      alone only reveals the control corner, not which guide belongs to which side. The correct
#      transform must place the guide run of length Lh onto the frame side whose dominant-color
#      longest run is Lh, and likewise for Lv.
#    - Example: suppose the top border has color 3 in 85% of cells and its longest uninterrupted
#      color-3 run has length 6; the left border is also mostly color 3 but its longest run is 9.
#      The object contains one top/left guide pair of color 3 with run lengths 6 and 9. A main-
#      diagonal reflection still keeps the guide colors on the same two sides, but swaps the 6 and
#      9, so color-only logic would accept it while color+length logic rejects it.
# b) Complexity calibration:
#    - Reusing one control color on both selected sides is necessary; otherwise the original color
#      cue already fixes the mapping and the length feature becomes decorative.
#    - Using gappy control edges instead of partial frame segments keeps D3/1 visually distinct:
#      the two selected sides are still "full borders", but they must be recognized statistically.
#    - The guide runs stay partial and the core is forced non-square, so the wrong color-only
#      reflection usually yields a transposed output and the extra constraint is observable.
# c) Implicit assumptions that break, and the replacement logic:
#    - Original `detect_controls_frame()` assumed almost-solid sides. That is invalid once the
#      control edges contain several bg gaps. The new solver ranks all four frame sides by dominant
#      non-bg fill ratio and extracts the longest run of that dominant color.
#    - Original `guide_edges()` assumed a guide nearly filled the whole object side. That also
#      breaks: guides are now sparse outer strips with a single contiguous run, so the solver must
#      explicitly detect "one run on an otherwise bg edge strip".
#    - Original matching only compared colors. Here the match key is `(guide_color, guide_length)`.
#    - The new solver assumes the two real control sides have much higher dominant-color occupancy
#      than the two noise sides, and that the object's two guides are the only valid edge runs.
#      The generator enforces both conditions.
# d) Placement legality:
#    - The object is still placed with `random_free_location(..., border=1)`, so it never touches
#      the frame.
#    - Guide runs avoid the object-guide corner cells, which keeps the auxiliary border strips
#      visually separable and removable by cropping one full row/col after alignment.
#    - Frame gaps are inserted only on the selected border lines, and noise is confined to the two
#      unselected sides, so there is no overlap between elements.
# e) Visual observability:
#    - Across examples, a human sees two adjacent borders that are mostly one color but broken by a
#      few holes, plus an inner object whose two same-colored guide marks have different lengths.
#      This supports the rule "find the dominant border color, measure the longest segment, then
#      orient the object so the guide segment lengths match the corresponding borders".
# f) Visual cleanliness:
#    - Control borders use one repeated color with sparse bg holes; the other borders use isolated
#      grey noise cells. The object remains a compact filled rectangle with only two auxiliary edge
#      strips, so the frame cue, guide cue, and core pattern stay visually separated.
# g) Color-change inferability:
#    - No arbitrary recoloring happens. The output is still the core cropped from the aligned
#      object. Color is only used as a matching label, and the important structural signal is the
#      shared control color together with differing run lengths.
#
# New solver assumptions introduced here, and generator guarantees:
# - Exactly two frame sides have dominant non-bg ratio in the "high occupancy" regime; generator
#   paints them at 80-90% occupancy and keeps the other two sides sparse.
# - The longest dominant-color run lengths of the two selected sides are distinct; generator
#   constructs the gappy border lines to satisfy chosen lengths exactly.
# - The object has exactly two guide sides, both same-colored, each represented by one contiguous
#   run on an otherwise bg outer strip; generator attaches guides in that format only.
# - The core is non-square, so the color-only diagonal-reflection ambiguity changes the cropped
#   output and the length condition materially matters.

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_ARC_UTILS_DIR = _ROOT / "arc2_solver_select"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDES = ("top", "bottom", "left", "right")
CORNERS = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))
NOISE_COLOR = 5


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


def _runs(values: list[int], bg: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    start = None
    color = None
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


def _dominant_border_feature(values: np.ndarray, bg: int) -> tuple[int, int, int, float] | None:
    line = list(map(int, values))
    non_bg = [val for val in line if int(val) != int(bg)]
    if not non_bg:
        return None
    counts: dict[int, int] = {}
    for val in non_bg:
        counts[int(val)] = counts.get(int(val), 0) + 1
    color, count = max(counts.items(), key=lambda item: (int(item[1]), int(item[0])))
    longest = 0
    for run_color, _start, length in _runs(line, int(bg)):
        if int(run_color) == int(color):
            longest = max(int(longest), int(length))
    ratio = float(count) / float(len(line))
    return int(color), int(count), int(longest), float(ratio)


def _detect_frame_controls(arr: np.ndarray, bg: int) -> dict[str, tuple[int, int]]:
    candidates: list[tuple[float, int, int, str, int]] = []
    for side in SIDES:
        feat = _dominant_border_feature(_side_values(arr, side), int(bg))
        if feat is None:
            continue
        color, count, longest, ratio = feat
        if int(longest) < 2:
            continue
        candidates.append((float(ratio), int(count), int(longest), side, int(color)))
    candidates.sort(key=lambda item: (float(item[0]), int(item[1]), int(item[2]), int(item[4])), reverse=True)
    return {side: (int(color), int(longest)) for ratio, count, longest, side, color in candidates[:2]}


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


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


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the two gappy control borders by dominant-color occupancy, then align the central object
    so its same-colored guide runs match both the target sides and the target run lengths.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    desired = _detect_frame_controls(g, int(bg))
    obj = _extract_center_object(g, int(bg))
    if len(desired) != 2:
        return obj

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(tuple(map(int, features[side])) == tuple(map(int, desired[side])) for side in desired):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                return t[int(top) : int(bottom), int(left) : int(right)]

    return obj


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors) + 2, int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors) + 1, area // 8), max_nonfill)
    painted = 0

    for color in pattern_colors:
        r = random.randrange(h)
        c = random.randrange(w)
        core[r, c] = int(color)
        painted += 1

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

    return core


def _choose_segment_offset(room: int) -> int:
    return random.randint(0, int(room))


def _attach_guides(
    core: np.ndarray,
    horiz_side: str,
    vert_side: str,
    guide_color: int,
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
        start = c0 + _choose_segment_offset(w - horiz_len)
        out[0, int(start) : int(start + horiz_len)] = int(guide_color)
    if has_bottom:
        start = c0 + _choose_segment_offset(w - horiz_len)
        out[-1, int(start) : int(start + horiz_len)] = int(guide_color)
    if has_left:
        start = r0 + _choose_segment_offset(h - vert_len)
        out[int(start) : int(start + vert_len), 0] = int(guide_color)
    if has_right:
        start = r0 + _choose_segment_offset(h - vert_len)
        out[int(start) : int(start + vert_len), -1] = int(guide_color)

    return out


def _distribute_with_caps(total: int, caps: list[int]) -> list[int] | None:
    if total < 0 or total > sum(int(cap) for cap in caps):
        return None
    remaining = int(total)
    extras = [0] * len(caps)
    order = list(range(len(caps)))
    random.shuffle(order)
    for pos, idx in enumerate(order):
        cap = int(caps[idx])
        future_cap = sum(int(caps[order[j]]) for j in range(pos + 1, len(order)))
        low = max(0, int(remaining) - int(future_cap))
        high = min(int(cap), int(remaining))
        if low > high:
            return None
        take = random.randint(int(low), int(high))
        extras[idx] = int(take)
        remaining -= int(take)
    if remaining != 0:
        return None
    return extras


def _make_control_line(line_len: int, color: int, target_run: int, bg: int = 0) -> np.ndarray | None:
    min_fill = max(int(np.ceil(0.80 * line_len)), int(target_run) + 1)
    max_fill = int(np.floor(0.90 * line_len))
    if min_fill > max_fill:
        return None

    configs: list[tuple[int, int, int, list[int], list[int]]] = []
    for fill_count in range(min_fill, max_fill + 1):
        gap_count = int(line_len - fill_count)
        if gap_count < 1:
            continue
        for anchor in range(gap_count + 1):
            base = [0] * (gap_count + 1)
            base[anchor] = int(target_run)
            for idx in range(gap_count + 1):
                if idx == anchor:
                    continue
                if 0 < idx < gap_count:
                    base[idx] = 1
            base_sum = sum(int(x) for x in base)
            if base_sum > int(fill_count):
                continue
            caps: list[int] = []
            ok = True
            for idx, base_val in enumerate(base):
                if idx == anchor:
                    caps.append(0)
                    continue
                cap = int(target_run) - 1 - int(base_val)
                if cap < 0:
                    ok = False
                    break
                caps.append(int(cap))
            if not ok:
                continue
            if int(fill_count) - int(base_sum) <= sum(int(x) for x in caps):
                configs.append((int(fill_count), int(gap_count), int(anchor), list(base), caps))

    random.shuffle(configs)
    for fill_count, gap_count, anchor, base, caps in configs:
        extras = _distribute_with_caps(int(fill_count) - sum(base), caps)
        if extras is None:
            continue
        segs = [int(base[idx] + extras[idx]) for idx in range(gap_count + 1)]
        line: list[int] = []
        for idx, seg_len in enumerate(segs):
            line.extend([int(color)] * int(seg_len))
            if idx < gap_count:
                line.append(int(bg))
        if len(line) != int(line_len):
            continue
        longest = 0
        current = 0
        count = 0
        for val in line:
            if int(val) == int(color):
                current += 1
                count += 1
                longest = max(int(longest), int(current))
            else:
                current = 0
        ratio = float(count) / float(line_len)
        if int(longest) == int(target_run) and 0.80 <= ratio <= 0.90:
            return np.array(line, dtype=int)
    return None


def _paint_noise_side(line: np.ndarray, noise_color: int) -> None:
    size = int(line.size)
    if size <= 0:
        return
    min_count = max(1, int(np.ceil(0.18 * size)))
    max_count = min(int(np.floor(0.35 * size)), (size + 1) // 2)
    count = random.randint(min_count, max_count)
    chosen: list[int] = []
    for idx in random.sample(list(range(size)), size):
        if any(abs(int(idx) - int(prev)) <= 1 for prev in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= int(count):
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
    bg: int = 0,
) -> bool:
    line_len = int(grid.shape[0] - 2)
    horiz_line = _make_control_line(line_len, int(ctrl_color), int(horiz_len), bg=int(bg))
    vert_line = _make_control_line(line_len, int(ctrl_color), int(vert_len), bg=int(bg))
    if horiz_line is None or vert_line is None:
        return False

    if horiz_side == "top":
        grid[0, 1:-1] = horiz_line
    else:
        grid[-1, 1:-1] = horiz_line

    if vert_side == "left":
        grid[1:-1, 0] = vert_line
    else:
        grid[1:-1, -1] = vert_line

    for side in SIDES:
        if side in (horiz_side, vert_side):
            continue
        if side == "top":
            _paint_noise_side(grid[0, 1:-1], NOISE_COLOR)
        elif side == "bottom":
            _paint_noise_side(grid[-1, 1:-1], NOISE_COLOR)
        elif side == "left":
            _paint_noise_side(grid[1:-1, 0], NOISE_COLOR)
        else:
            _paint_noise_side(grid[1:-1, -1], NOISE_COLOR)

    grid[0, 0] = int(bg)
    grid[0, -1] = int(bg)
    grid[-1, 0] = int(bg)
    grid[-1, -1] = int(bg)
    return True


def _frame_metrics(grid: np.ndarray) -> dict[str, tuple[int, int, int, float]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out: dict[str, tuple[int, int, int, float]] = {}
    for side in SIDES:
        feat = _dominant_border_feature(_side_values(g, side), int(bg))
        if feat is not None:
            out[side] = feat
    return out


def _count_color_only_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_frame_controls(g, int(bg))
    desired_colors = {side: int(color) for side, (color, _length) in desired.items()}
    obj = _extract_center_object(g, int(bg))
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired_colors):
                continue
            if all(int(features[side][0]) == int(desired_colors[side]) for side in desired_colors):
                count += 1
    return int(count)


def _fusion_metric(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    desired = _detect_frame_controls(np.array(grid, dtype=int), int(get_bg(np.array(grid, dtype=int))))
    color_only = _count_color_only_matches(grid)
    sides = sorted(desired)
    if len(sides) != 2:
        return (color_only, 0, 0, 0, 0)
    lengths = sorted(int(desired[side][1]) for side in sides)
    metrics = _frame_metrics(grid)
    ratios = sorted(int(round(metrics[side][3] * 100)) for side in sides if side in metrics)
    if len(ratios) != 2:
        ratios = [0, 0]
    return (color_only, int(lengths[0]), int(lengths[1]), int(ratios[0]), int(ratios[1]))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(25, 31)
    horiz_side, vert_side = random.choice(CORNERS)

    all_colors = list(range(1, 10))
    ctrl_color = random.choice([c for c in all_colors if c != NOISE_COLOR])
    remaining = [c for c in all_colors if c not in {ctrl_color, NOISE_COLOR}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_core_h = min(18, n - 5)
    max_core_w = min(18, n - 5)
    if max_core_h < 8 or max_core_w < 8:
        return None

    core = None
    core_h = core_w = 0
    for _ in range(40):
        core_h = random.randint(8, max_core_h)
        core_w = random.randint(8, max_core_w)
        if core_h != core_w:
            core = _build_core(core_h, core_w, fill, pattern_colors)
            break
    if core is None:
        return None

    max_horiz_len = min(core_w - 2, n - 6)
    max_vert_len = min(core_h - 2, n - 6)
    if max_horiz_len < 4 or max_vert_len < 4:
        return None

    for _ in range(80):
        horiz_len = random.randint(4, max_horiz_len)
        vert_len = random.randint(4, max_vert_len)
        if horiz_len == vert_len:
            continue
        if _make_control_line(n - 2, ctrl_color, horiz_len) is None:
            continue
        if _make_control_line(n - 2, ctrl_color, vert_len) is None:
            continue
        break
    else:
        return None

    aligned_object = _attach_guides(core, horiz_side, vert_side, ctrl_color, horiz_len, vert_len)
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    if not _paint_frame(grid, horiz_side, vert_side, ctrl_color, horiz_len, vert_len):
        return None

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), core):
        return None
    if _count_color_only_matches(grid) < 2:
        return None

    desired = _detect_frame_controls(grid, int(get_bg(grid)))
    if len(desired) != 2:
        return None
    lengths = sorted(int(feat[1]) for feat in desired.values())
    if lengths[0] == lengths[1]:
        return None

    metrics = _frame_metrics(grid)
    selected_ratios = [metrics[side][3] for side in desired if side in metrics]
    other_ratios = [metrics[side][3] for side in SIDES if side not in desired and side in metrics]
    if len(selected_ratios) != 2 or any(not (0.80 <= ratio <= 0.90) for ratio in selected_ratios):
        return None
    if any(ratio >= 0.45 for ratio in other_ratios):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D1/1+D3/1")
