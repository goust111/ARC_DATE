from __future__ import annotations

# Variant of fc7cae8d
# Dimension: D1 — 匹配条件：边颜色对齐 (match_cond)
# Change: 引导边颜色 == 框架控制边颜色（edge_color_match），通过旋转/翻转实现对齐... → 颜色+长度双重匹配
#
# Design analysis for this variant
# a) Concrete behavior and trace:
#    - Original task: the frame advertised two target sides with two solid border colors, and the
#      inner object had two full-length guide strips with matching colors. Solver only had to find
#      a dihedral transform whose guide-strip colors land on the same sides as the frame colors.
#    - Variant here: both control sides use the same control color, so color alone only says
#      "these two sides belong to the control pair". The distinguishing signal is the length of
#      the contiguous control-color segment on each border side. The object likewise carries two
#      1px guide strips made of the same color, but each strip is only a partial contiguous run.
#      The run lengths must match the corresponding frame-side run lengths after rotation/flip.
#    - Concrete example: if the frame has a top segment of color 2 with length 4 and a left
#      segment of color 2 with length 7, then the aligned object must expose a top guide of
#      color 2 length 4 and a left guide of color 2 length 7. A transpose-like transform can
#      still place color-2 guides on top and left, but it swaps the 4 and 7, so color-only logic
#      accepts it while color+length logic rejects it.
# b) Complexity calibration:
#    - Keeping distinct guide colors would leave the original puzzle essentially unchanged because
#      the side mapping would already be unique. Using the same control color on both selected
#      sides makes the extra numeric constraint necessary instead of decorative.
#    - The guide strips are partial runs rather than full sides, so the solver must measure exact
#      contiguous-run lengths instead of checking "edge is almost solid".
#    - The core is forced non-square so the color-only ambiguous transform would usually change the
#      output shape/orientation, making the new rule visually observable to a human.
# c) Original implicit assumptions that break under the variant:
#    - `detect_controls_frame()` assumed each control side was almost fully colored. That fails
#      because selected sides now contain only a partial same-color run on an otherwise blank side.
#    - `guide_edges()` assumed a guide edge was almost the whole side of the cropped object. That
#      fails because guide strips are now sparse outer rows/cols with one contiguous run and the
#      rest background.
#    - Matching `desired[side] = color` was sufficient before; now it is insufficient because two
#      sides intentionally share the same color and must be separated by run length.
# d) Placement legality:
#    - The object is still placed strictly inside the frame using `random_free_location(..., border=1)`.
#    - Guide segments never touch object corners, so the outer guide row/col remains a removable
#      auxiliary strip. Cropping one full row/col on matched sides still returns the core exactly.
#    - Frame control segments stay away from the frame corners and non-selected sides only get
#      isolated noise cells, so the intended two longest border runs remain unambiguous.
# e) Visual observability:
#    - Across examples, the frame visibly shows two same-colored border segments with different
#      lengths, and the object visibly shows two same-colored guide runs with different lengths.
#      A human can infer that both side and run length matter, because a simple color match no
#      longer tells which guide should go to which frame side.
#
# New solver assumptions introduced by the variant, and how the generator guarantees them:
# - There are exactly two meaningful border runs longer than 1 cell; generator enforces this by
#   painting only the selected control segments as long runs and keeping all other border noise
#   isolated (run length 1).
# - The object has exactly two guide sides, both represented by a single contiguous same-color run
#   on an otherwise background outer strip; generator constructs guides in that exact format.
# - The two guide lengths differ; generator samples distinct lengths and rejects accidental ties.
# - The inner core contains no background holes and is non-square; generator builds a filled
#   rectangle with non-background decorations and forces `core_h != core_w`.

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_ARC_UTILS_DIR = _ROOT / "arc2_solver_select"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDES = ("top", "bottom", "left", "right")


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


def _best_border_run(values: np.ndarray, bg: int) -> tuple[int, int] | None:
    runs = _runs(list(map(int, values)), int(bg))
    if not runs:
        return None
    color, _start, length = max(runs, key=lambda item: (int(item[2]), int(item[0])))
    return int(color), int(length)


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
    Align the central object using border-side color+length controls, then remove the guide strips.

    Variant rule:
    - Two frame sides contain same-color control runs, but with different lengths.
    - The object has two same-color guide runs on auxiliary outer strips.
    - A dihedral transform is correct only if each guide run matches both the target side and
      that side's control-run length.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    candidates: list[tuple[int, str, int]] = []
    for side in SIDES:
        best = _best_border_run(_side_values(g, side), int(bg))
        if best is None:
            continue
        color, length = best
        if int(length) < 2:
            continue
        candidates.append((int(length), side, int(color)))
    candidates.sort(key=lambda item: (int(item[0]), int(item[2])), reverse=True)
    desired = {side: (int(color), int(length)) for length, side, color in candidates[:2]}

    obj = _extract_center_object(g, int(bg))

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


def _place_isolated_noise(line: np.ndarray, noise_color: int) -> None:
    if line.size <= 0:
        return
    available = list(range(line.size))
    random.shuffle(available)
    count = random.randint(1, max(1, line.size // 5))
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
    noise_color: int,
) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        c0 = random.randint(1, n - 1 - horiz_len)
        grid[0, c0 : c0 + horiz_len] = int(ctrl_color)
    else:
        c0 = random.randint(1, n - 1 - horiz_len)
        grid[-1, c0 : c0 + horiz_len] = int(ctrl_color)

    if vert_side == "left":
        r0 = random.randint(1, n - 1 - vert_len)
        grid[r0 : r0 + vert_len, 0] = int(ctrl_color)
    else:
        r0 = random.randint(1, n - 1 - vert_len)
        grid[r0 : r0 + vert_len, -1] = int(ctrl_color)

    for side in SIDES:
        if side in (horiz_side, vert_side):
            continue
        if side == "top":
            _place_isolated_noise(grid[0, 1:-1], int(noise_color))
        elif side == "bottom":
            _place_isolated_noise(grid[-1, 1:-1], int(noise_color))
        elif side == "left":
            _place_isolated_noise(grid[1:-1, 0], int(noise_color))
        else:
            _place_isolated_noise(grid[1:-1, -1], int(noise_color))

    grid[0, 0] = 0
    grid[0, -1] = 0
    grid[-1, 0] = 0
    grid[-1, -1] = 0


def _desired_features_from_grid(grid: np.ndarray) -> dict[str, tuple[int, int]]:
    bg = int(get_bg(grid))
    candidates: list[tuple[int, str, int]] = []
    for side in SIDES:
        best = _best_border_run(_side_values(grid, side), int(bg))
        if best is None:
            continue
        color, length = best
        if int(length) < 2:
            continue
        candidates.append((int(length), side, int(color)))
    candidates.sort(key=lambda item: (int(item[0]), int(item[2])), reverse=True)
    return {side: (int(color), int(length)) for length, side, color in candidates[:2]}


def _count_color_only_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired_colors = {side: color for side, (color, _length) in _desired_features_from_grid(g).items()}
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


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_color, noise_color = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {ctrl_color, noise_color}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(18, n - 5)
    max_core_w = min(18, n - 5)
    if max_core_h < 6 or max_core_w < 6:
        return None

    for _ in range(40):
        core_h = random.randint(6, max_core_h)
        core_w = random.randint(6, max_core_w)
        if core_h != core_w:
            break
    else:
        return None

    max_horiz_len = min(core_w - 2, n - 4)
    max_vert_len = min(core_h - 2, n - 4)
    if max_horiz_len < 3 or max_vert_len < 3:
        return None

    horiz_len = random.randint(3, max_horiz_len)
    vert_len = random.randint(3, max_vert_len)
    if horiz_len == vert_len:
        if max_vert_len > 3 and vert_len == max_vert_len:
            vert_len -= 1
        elif max_vert_len >= vert_len + 1:
            vert_len += 1
        elif max_horiz_len > 3:
            horiz_len -= 1
        else:
            return None
    if horiz_len == vert_len:
        return None

    core = _build_core(core_h, core_w, fill, pattern_colors)
    aligned_object = _attach_guides(core, horiz_side, vert_side, ctrl_color, horiz_len, vert_len)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_color, horiz_len, vert_len, noise_color)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), core):
        return None
    if _count_color_only_matches(grid) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D1 variant")
