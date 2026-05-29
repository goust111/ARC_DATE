# Combo variant of fc7cae8d (index 38)
# combo_spec: D1/1+D4/1+D11/2
# Name: 颜色长度+互换+正方形核心

from __future__ import annotations

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Original task: frame controls were almost-solid full edges with two different colors; the
#      inner object had two full guide strips with those colors; solver only needed a D4 transform
#      that aligned guide-side colors to frame-side colors, then cropped those sides away.
#    - Fused task: the frame instead shows two adjacent partial border runs of the SAME color, but
#      with different lengths, for example top=4 cells and left=7 cells. The inner object is a
#      square core wrapped by two auxiliary guide strips on adjacent sides, also using that same
#      color with lengths 4 and 7. Because the core is square, rotating the object preserves its
#      shape; the only way to know which orientation is correct is to match both side and run
#      length. After alignment, the solver crops the two auxiliary strips and then swaps the two
#      non-fill pattern colors inside the square core.
#    - Small trace: suppose the aligned object has a top guide run length 4 and a left guide run
#      length 7, and the cropped core uses fill=1 plus pattern colors {2, 3}. A transpose-like
#      transform can still put same-color guides on top and left, but swaps which side gets length
#      4 versus 7. Color-only logic cannot distinguish those states; color+length logic can. Once
#      the correct aligned crop is found, all 2s become 3s and all 3s become 2s.
# b) Complexity calibration:
#    - Both control sides intentionally share one color; otherwise the length signal would become
#      decorative because color already fixes the orientation.
#    - The guide marks are partial runs rather than full strips, so the solver must measure exact
#      contiguous lengths instead of checking whether an edge is almost solid.
#    - The core is forced square to remove the original rectangle-shape cue for 90-degree turns,
#      but the interior is forced D4-asymmetric so the puzzle does not collapse into trivial
#      rotational symmetry or a swap-invariant output.
#    - Only two pattern colors are used, and both are repeated multiple times, so the post-crop
#      color swap is visually obvious rather than looking like random recoloring.
# c) Implicit-assumption review of the original main():
#    - Original `detect_controls_frame()` assumed near-solid border edges. That breaks here because
#      selected borders are sparse partial runs on otherwise blank borders, so control detection now
#      scans contiguous non-background runs and takes the two longest runs (>1 cell).
#    - Original `guide_edges()` assumed the guide side was almost the whole object edge. That also
#      breaks here; the object edge now contains one contiguous same-color run embedded in
#      background. The new guide parser validates that the run is contiguous, shorter than the full
#      edge, and backed by non-background core cells on the inner side.
#    - Original matching only compared colors. That is insufficient here because both guide sides
#      share one color on purpose; the solver must compare `(color, length)` per side.
#    - Original crop result was final output. That no longer holds because the fused rule requires
#      a second stage: crop first, then swap the two non-fill colors.
# d) Placement legality:
#    - The generator first builds the aligned square object, applies an inverse D4 transform, then
#      uses `random_free_location(..., border=1)` before pasting. This keeps the object strictly
#      inside the outer frame, non-overlapping, and in bounds.
#    - Guide runs never touch the outer object corners, so removing one whole outer row/column on
#      the matched sides returns the core exactly.
#    - Frame control runs also stay away from frame corners; all non-selected border noise is
#      isolated single cells, so the intended two longest runs remain unambiguous.
# e) Visual observability:
#    - Across examples, a human sees two same-colored frame segments of different lengths and two
#      same-colored guide segments of different lengths on the inner object, so "measure the run"
#      is externally visible.
#    - The core is always square, so orientation cannot be inferred from rectangle aspect ratio.
#    - The output keeps the aligned square shape but visibly exchanges the two repeated pattern
#      colors, making the post-crop transform inferable from a few examples.
# f) Visual cleanliness:
#    - The frame only contains one control color plus sparse single-cell noise on the unused sides,
#      avoiding clutter on the border.
#    - The object is one compact square-with-guides component; there are no extra floating markers
#      or overlapping annotations inside the same region.
#    - Guide strips stay outside the core, so the core pattern itself remains readable.
# g) Color-change inferability:
#    - The generator uses exactly two non-fill pattern colors, both appearing several times.
#      Because the output preserves every shape position and only exchanges those two colors, a
#      human can infer a clean two-color swap instead of an arbitrary color remapping.
#
# New solver assumptions introduced by the fusion, and how the generator guarantees them:
# - Exactly two meaningful border control runs exist, both longer than 1 cell and both same color.
#   Generator paints only two long same-color border runs and keeps all other border noise isolated.
# - The guided object is the largest non-border component. Generator places only one interior
#   object and keeps the border annotations attached to the frame.
# - Each guide side is represented by one contiguous same-color run on an otherwise background
#   outer strip. Generator constructs the guide strips in exactly that format.
# - The two guide lengths differ. Generator samples distinct lengths and rejects ties.
# - The cropped core contains exactly two non-fill colors and the fill color is most frequent.
#   Generator builds the core under those count constraints and rejects degenerate samples.
# - Color-only matching should be ambiguous while color+length matching is unique. Generator checks
#   transform counts and rejects samples unless at least two color-only matches but exactly one
#   exact match exist.

from collections import Counter
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
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _guide_features(arr: np.ndarray, bg: int) -> dict[str, tuple[int, int]]:
    h, w = arr.shape
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
            if h < 2 or any(int(arr[1, idx]) == int(bg) for idx in non_bg):
                continue
        elif side == "bottom":
            if h < 2 or any(int(arr[-2, idx]) == int(bg) for idx in non_bg):
                continue
        elif side == "left":
            if w < 2 or any(int(arr[idx, 1]) == int(bg) for idx in non_bg):
                continue
        else:
            if w < 2 or any(int(arr[idx, -2]) == int(bg) for idx in non_bg):
                continue
        out[side] = (int(color), int(len(non_bg)))
    return out


def _desired_features_from_grid(grid: np.ndarray, bg: int) -> dict[str, tuple[int, int]]:
    candidates: list[tuple[int, str, int]] = []
    for side in SIDES:
        best = _best_border_run(_side_values(grid, side), int(bg))
        if best is None:
            continue
        color, length = best
        if int(length) < 2:
            continue
        candidates.append((int(length), side, int(color)))
    candidates.sort(key=lambda item: (int(item[0]), int(item[2]), item[1]), reverse=True)
    return {side: (int(color), int(length)) for length, side, color in candidates[:2]}


def _crop_guided_core(arr: np.ndarray, desired_sides: set[str]) -> np.ndarray:
    top = 1 if "top" in desired_sides else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in desired_sides else 0)
    left = 1 if "left" in desired_sides else 0
    right = int(arr.shape[1]) - (1 if "right" in desired_sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def _swap_pattern_colors(arr: np.ndarray) -> np.ndarray:
    counts = Counter(map(int, arr.flatten()))
    fill = int(counts.most_common(1)[0][0])
    pattern_colors = sorted(int(color) for color in counts if int(color) != int(fill))
    if len(pattern_colors) != 2:
        return arr.copy()
    a, b = map(int, pattern_colors)
    out = arr.copy()
    mask_a = arr == a
    mask_b = arr == b
    out[mask_a] = b
    out[mask_b] = a
    return out


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the guided square object using frame-side color+length controls.
    Then crop away the matched guide strips and swap the two pattern colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _desired_features_from_grid(g, int(bg))
    obj = _extract_center_object(g, int(bg))

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(tuple(map(int, features[side])) == tuple(map(int, desired[side])) for side in desired):
                core = _crop_guided_core(t, set(desired))
                return _swap_pattern_colors(core)

    return obj


def _d4_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        base = np.rot90(arr, int(k))
        variants.append(base)
        variants.append(np.fliplr(base))
    return variants


def _is_d4_distinct(arr: np.ndarray) -> bool:
    seen: list[np.ndarray] = []
    for variant in _d4_variants(arr):
        if any(np.array_equal(variant, prev) for prev in seen):
            return False
        seen.append(variant)
    return True


def _build_core(side: int, fill: int, pattern_colors: tuple[int, int]) -> np.ndarray | None:
    a, b = map(int, pattern_colors)
    core = np.full((side, side), int(fill), dtype=int)
    area = side * side
    min_per_color = max(4, area // 16)
    max_total = max(2 * min_per_color + 2, int(area * 0.34))
    total_nonfill = random.randint(2 * min_per_color, max_total)
    targets = [min_per_color, min_per_color]
    extra = total_nonfill - 2 * min_per_color
    if extra > 0:
        give_a = random.randint(0, extra)
        targets[0] += give_a
        targets[1] += extra - give_a

    for color, target in zip((a, b), targets):
        painted = 0
        attempts = 0
        while painted < target and attempts < area * 12:
            attempts += 1
            shape_kind = random.choice(["dot", "line", "rect", "el"])
            if shape_kind == "dot":
                cells = [(random.randrange(side), random.randrange(side))]
            elif shape_kind == "line":
                line_len = random.randint(2, min(6, side))
                if random.random() < 0.5:
                    rr = random.randrange(side)
                    c0 = random.randrange(side - line_len + 1)
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(side - line_len + 1)
                    cc = random.randrange(side)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            elif shape_kind == "rect":
                rh = random.randint(1, min(4, side))
                rw = random.randint(1, min(4, side))
                r0 = random.randrange(side - rh + 1)
                c0 = random.randrange(side - rw + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)]
            else:
                rh = random.randint(2, min(5, side))
                rw = random.randint(2, min(5, side))
                r0 = random.randrange(side - rh + 1)
                c0 = random.randrange(side - rw + 1)
                cells = [(r0 + dr, c0) for dr in range(rh)] + [(r0 + rh - 1, c0 + dc) for dc in range(rw)]

            random.shuffle(cells)
            for r, c in cells:
                if core[r, c] != fill:
                    continue
                core[r, c] = int(color)
                painted += 1
                if painted >= target:
                    break
        if painted < target:
            return None

    counts = {int(color): int(count) for color, count in Counter(map(int, core.flatten())).items()}
    if counts.get(int(fill), 0) <= max(counts.get(a, 0), counts.get(b, 0)):
        return None
    if counts.get(a, 0) < min_per_color or counts.get(b, 0) < min_per_color:
        return None
    if not _is_d4_distinct(core):
        return None
    if np.array_equal(core, _swap_pattern_colors(core)):
        return None
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
    count = random.randint(1, max(1, line.size // 6))
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


def _count_transform_matches(grid: np.ndarray) -> tuple[int, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _desired_features_from_grid(g, int(bg))
    desired_colors = {side: int(color) for side, (color, _length) in desired.items()}
    obj = _extract_center_object(g, int(bg))
    color_only = 0
    exact = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(int(features[side][0]) == int(desired_colors[side]) for side in desired_colors):
                color_only += 1
            if all(tuple(map(int, features[side])) == tuple(map(int, desired[side])) for side in desired):
                exact += 1
    return int(color_only), int(exact)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _desired_features_from_grid(g, int(bg))
    unswapped = None
    obj = _extract_center_object(g, int(bg))
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(tuple(map(int, features[side])) == tuple(map(int, desired[side])) for side in desired):
                unswapped = _crop_guided_core(t, set(desired))
                break
        if unswapped is not None:
            break
    color_only, exact = _count_transform_matches(g)
    swapped = main(g)
    changed = 0 if unswapped is None else int(not np.array_equal(swapped, unswapped))
    desired_lengths = sorted(int(length) for _side, (_color, length) in desired.items())
    return {
        "color_only_matches": int(color_only),
        "exact_matches": int(exact),
        "swap_changes_output": int(changed),
        "min_control_len": int(desired_lengths[0]) if desired_lengths else 0,
        "max_control_len": int(desired_lengths[-1]) if desired_lengths else 0,
        "core_side": int(swapped.shape[0]),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_color, noise_color = random.sample(all_colors, 2)
    remaining = [color for color in all_colors if color not in {ctrl_color, noise_color}]
    fill = random.choice(remaining)
    remaining = [color for color in remaining if color != fill]
    pattern_colors = tuple(random.sample(remaining, 2))

    max_core_side = min(18, n - 6)
    if max_core_side < 7:
        return None
    core_side = random.randint(7, max_core_side)

    max_horiz_len = core_side - 2
    max_vert_len = core_side - 2
    if max_horiz_len < 3 or max_vert_len < 3:
        return None

    horiz_len = random.randint(3, max_horiz_len)
    vert_len = random.randint(3, max_vert_len)
    if horiz_len == vert_len:
        if vert_len < max_vert_len:
            vert_len += 1
        elif horiz_len > 3:
            horiz_len -= 1
        else:
            return None
    if horiz_len == vert_len:
        return None

    core = _build_core(core_side, fill, pattern_colors)
    if core is None:
        return None
    expected = _swap_pattern_colors(core)

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

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if actual.shape[0] != actual.shape[1]:
        return None

    color_only, exact = _count_transform_matches(grid)
    if color_only < 2 or exact != 1:
        return None

    metrics = _fusion_metrics(grid)
    if metrics["swap_changes_output"] != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 38")
