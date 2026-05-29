from __future__ import annotations

# Combo variant of fc7cae8d (index 2)
# combo_spec: D1/1+D2/0
# Name: 颜色长度匹配+可变引导宽度
#
# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Original task: the frame gave two target sides by color, the inner object had two 1px guide
#      edges with matching colors, and the solver only had to find the dihedral transform whose
#      guide colors landed on the same sides, then crop 1 row/col.
#    - Combo task: the two frame controls now use the same color and are only partial contiguous
#      runs, so side identity is recovered by run length instead of by color alone. Meanwhile the
#      object's guide borders are widened to 2px or 3px. Each widened guide is a solid rectangle
#      of size width x length or length x width, with the shared guide color, and the two guides
#      meet only through the core because the width x width corner block stays background.
#    - Example: if the frame has a top run of color 2 length 5 and a left run of color 2 length 8,
#      then the aligned object must show a top guide block color 2, width 3, length 5 and a left
#      guide block color 2, width 3, length 8. A transform that puts the same-color guides on top
#      and left but swaps the lengths is invalid; a transform that gets the lengths right but then
#      crops only 1 row/col is also invalid because guide residue remains.
# b) Complexity calibration:
#    - Width is restricted to {2, 3}; this is enough to force width inference without turning the
#      puzzle into noisy parameter search.
#    - Both frame controls deliberately share one color, so length is a necessary discriminator.
#    - Guide lengths are partial runs, not full edges, so the length signal is visible and does not
#      collapse back to the original full-edge color matching.
# c) Original implicit assumptions broken by the combo:
#    - `detect_controls_frame()` expected almost-solid full border edges. That fails because frame
#      controls are now partial runs on otherwise blank sides.
#    - `guide_edges()` expected each guide to be a 1px almost-solid edge. That fails because guides
#      are wider strips and only occupy a contiguous sub-interval along that strip.
#    - Cropping exactly 1 row/col is no longer valid; the solver must infer the common guide width
#      from the object before cropping.
# d) Placement legality:
#    - The generator keeps a single connected inner object and places it with
#      `random_free_location(..., border=1)`, so it never overlaps the frame or exits bounds.
#    - The corner overlap zone of the two guide strips is left as background, which keeps the two
#      widened guides visually separable and prevents accidental corner merging.
# e) Visual observability:
#    - A human sees two same-colored frame runs of different lengths and two same-colored wide guide
#      strips of matching lengths. Across samples, width alternates between 2 and 3, so both the
#      need to measure width and the need to use length as a tie-breaker are visible.
# f) Visual cleanliness:
#    - Only three structural layers are used: sparse border controls/noise, one centered object,
#      and one guide color. The generator avoids multi-object overlap and keeps border noise
#      isolated to singleton cells.
# g) Color inferability:
#    - No hidden color remapping is introduced. The shared guide/control color is structurally
#      signaled by repeated use on exactly the informative runs; other colors live only inside the
#      core pattern or as isolated border noise.
#
# New solver assumptions introduced here, and how the generator guarantees them:
# - Exactly two border runs have length >= 3; the generator paints only the selected frame sides as
#   long runs and keeps all other border marks isolated.
# - The inner object has exactly two detectable guide strips, and they share one width in {2, 3};
#   the generator constructs both guides from the same sampled width.
# - The two guide lengths are distinct, so color-only alignment remains ambiguous while
#   color+length alignment is unique; the generator rejects accidental ties.
# - The core is a filled non-square rectangle with no background holes, so once the correct guide
#   width is removed the output is unambiguous and visually different from a length-swapped match.

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


def _guide_band_feature(arr: np.ndarray, side: str, bg: int) -> tuple[int, int, int] | None:
    H, W = arr.shape
    max_width = min(3, (H - 1) if side in ("top", "bottom") else (W - 1))
    if max_width < 2:
        return None

    for width in range(max_width, 1, -1):
        if side == "top":
            band = arr[:width, :]
            mask = np.any(band != int(bg), axis=0)
            total_len = W
            attach_ok = lambda s, e: bool(H > width) and bool(np.all(arr[width, s : e + 1] != int(bg)))
            block = lambda s, e: band[:, s : e + 1]
        elif side == "bottom":
            band = arr[H - width :, :]
            mask = np.any(band != int(bg), axis=0)
            total_len = W
            attach_ok = lambda s, e: bool(H > width) and bool(np.all(arr[H - width - 1, s : e + 1] != int(bg)))
            block = lambda s, e: band[:, s : e + 1]
        elif side == "left":
            band = arr[:, :width]
            mask = np.any(band != int(bg), axis=1)
            total_len = H
            attach_ok = lambda s, e: bool(W > width) and bool(np.all(arr[s : e + 1, width] != int(bg)))
            block = lambda s, e: band[s : e + 1, :]
        else:
            band = arr[:, W - width :]
            mask = np.any(band != int(bg), axis=1)
            total_len = H
            attach_ok = lambda s, e: bool(W > width) and bool(np.all(arr[s : e + 1, W - width - 1] != int(bg)))
            block = lambda s, e: band[s : e + 1, :]

        idx = np.flatnonzero(mask)
        if idx.size == 0 or idx.size == total_len:
            continue
        start = int(idx[0])
        end = int(idx[-1])
        if list(map(int, idx)) != list(range(start, end + 1)):
            continue
        length = int(end - start + 1)
        if length < 3:
            continue
        band_block = block(start, end)
        colors = set(map(int, band_block.flatten())) - {int(bg)}
        if len(colors) != 1 or np.any(band_block == int(bg)):
            continue
        if not attach_ok(start, end):
            continue
        color = int(next(iter(colors)))
        return color, length, int(width)
    return None


def _guide_features(arr: np.ndarray, bg: int) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for side in SIDES:
        feat = _guide_band_feature(arr, side, int(bg))
        if feat is not None:
            out[side] = feat
    return out


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the center object by matching frame-side control run color+length, then crop the full
    inferred guide-strip width from the matched sides.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    candidates: list[tuple[int, str, int]] = []
    for side in SIDES:
        best = _best_border_run(_side_values(g, side), int(bg))
        if best is None:
            continue
        color, length = best
        if int(length) < 3:
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
            widths = {int(width) for _color, _length, width in features.values()}
            if len(widths) != 1:
                continue
            if not all(
                int(features[side][0]) == int(desired[side][0]) and int(features[side][1]) == int(desired[side][1])
                for side in desired
            ):
                continue
            guide_width = int(next(iter(widths)))
            top = guide_width if "top" in desired else 0
            bottom = int(t.shape[0]) - (guide_width if "bottom" in desired else 0)
            left = guide_width if "left" in desired else 0
            right = int(t.shape[1]) - (guide_width if "right" in desired else 0)
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
    guide_width: int,
) -> np.ndarray:
    h, w = core.shape
    has_top = horiz_side == "top"
    has_bottom = horiz_side == "bottom"
    has_left = vert_side == "left"
    has_right = vert_side == "right"
    out = np.zeros(
        (h + guide_width * (has_top + has_bottom), w + guide_width * (has_left + has_right)),
        dtype=int,
    )
    r0 = guide_width if has_top else 0
    c0 = guide_width if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        start = c0 + _choose_segment_offset(w - horiz_len)
        out[:guide_width, int(start) : int(start + horiz_len)] = int(guide_color)
    if has_bottom:
        start = c0 + _choose_segment_offset(w - horiz_len)
        out[-guide_width:, int(start) : int(start + horiz_len)] = int(guide_color)
    if has_left:
        start = r0 + _choose_segment_offset(h - vert_len)
        out[int(start) : int(start + vert_len), :guide_width] = int(guide_color)
    if has_right:
        start = r0 + _choose_segment_offset(h - vert_len)
        out[int(start) : int(start + vert_len), -guide_width:] = int(guide_color)

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
        if int(length) < 3:
            continue
        candidates.append((int(length), side, int(color)))
    candidates.sort(key=lambda item: (int(item[0]), int(item[2])), reverse=True)
    return {side: (int(color), int(length)) for length, side, color in candidates[:2]}


def _count_color_width_only_matches(grid: np.ndarray) -> int:
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
            widths = {int(width) for _color, _length, width in features.values()}
            if len(widths) != 1:
                continue
            if all(int(features[side][0]) == int(desired[side][0]) for side in desired):
                count += 1
    return int(count)


def _sample_metrics(grid: np.ndarray) -> tuple[int, int]:
    bg = int(get_bg(grid))
    obj = _extract_center_object(grid, int(bg))
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            desired = _desired_features_from_grid(grid)
            if set(features) != set(desired):
                continue
            widths = {int(width) for _color, _length, width in features.values()}
            if len(widths) != 1:
                continue
            if all(
                int(features[side][0]) == int(desired[side][0]) and int(features[side][1]) == int(desired[side][1])
                for side in desired
            ):
                return int(next(iter(widths))), int(_count_color_width_only_matches(grid))
    raise ValueError("no aligned metrics found")


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(26, 32)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_color, noise_color = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {ctrl_color, noise_color}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    guide_width = random.choice([2, 3])
    max_core_h = min(18, n - 2 * guide_width - 3)
    max_core_w = min(18, n - 2 * guide_width - 3)
    if max_core_h < 7 or max_core_w < 7:
        return None

    for _ in range(40):
        core_h = random.randint(7, max_core_h)
        core_w = random.randint(7, max_core_w)
        if core_h != core_w:
            break
    else:
        return None

    max_horiz_len = core_w - 2
    max_vert_len = core_h - 2
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

    core = _build_core(core_h, core_w, fill, pattern_colors)
    aligned_object = _attach_guides(
        core,
        horiz_side,
        vert_side,
        ctrl_color,
        horiz_len,
        vert_len,
        guide_width,
    )

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
    if _count_color_width_only_matches(grid) < 2:
        return None
    width_metric, _ambiguous_matches = _sample_metrics(grid)
    if width_metric not in {2, 3}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")
