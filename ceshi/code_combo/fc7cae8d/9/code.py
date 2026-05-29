# Combo variant of fc7cae8d (index 9)
# combo_spec: D1/1+D9/0
# Name: 颜色长度匹配+颜色图例
from __future__ import annotations

# Design analysis for this combo
# a) Concrete behavior and trace:
#    - Original task: the frame showed two target sides by direct border colors, and the inner
#      object had full guide strips of the same colors. The solver only needed to find a dihedral
#      transform whose guide-strip colors landed on the same sides, then crop those strips away.
#    - D1/1 already changed this into "same control color + different run lengths": the frame now
#      has two partial control runs of one shared color, and the object has two partial guide runs.
#      Color alone only says "both belong to the control pair"; exact run length tells which guide
#      must end up on which side.
#    - This combo adds D9/0 by inserting a tiny corner legend. Each legend row is a 2-cell mapping
#      `guide_color -> control_color`. Both legend rows point to the same control color, so the
#      solver must first decode the legend, then compare mapped colors, then still use run length
#      to decide which transformed guide belongs to which control side.
#    - Concrete example: legend rows are `[6, 2]` and `[8, 2]`, so guide colors 6 and 8 both map
#      to frame color 2. Suppose the frame has a top run `(color 2, length 4)` and a left run
#      `(color 2, length 7)`. If a transformed object shows top guide `(6,4)` and left guide
#      `(8,7)`, it is correct because `6 -> 2` and `8 -> 2`. A transpose-like transform can still
#      put the same two mapped colors on top/left, but it swaps the 4 and 7, so only the full
#      "legend + color + length" rule accepts the right orientation.
# b) Complexity calibration:
#    - If guide colors directly matched the frame color, the legend would be decorative; therefore
#      generator forces both guide colors to differ from the frame control color.
#    - If the two guide runs had equal lengths, length matching would be decorative; therefore the
#      two run lengths are always distinct.
#    - If the core were square, the color-only wrong transform could output the same shape; the
#      core is therefore forced non-square so the length-resolved orientation remains visible.
#    - The legend uses only two rows and one repeated target color, which keeps the symbol system
#      small enough for ARC-style induction while still making the lookup indispensable.
# c) Implicit assumptions broken from the original solver:
#    - Original `detect_controls_frame()` assumed almost-solid full border edges. That fails here:
#      selected sides are sparse partial runs, so solver must scan contiguous runs instead.
#    - Original `guide_edges()` assumed a guide edge was almost the whole cropped side. That also
#      fails: each guide is now only one contiguous run on an otherwise background auxiliary strip.
#    - Original matching assumed guide color == control color. That is false after D9/0, so solver
#      must decode the corner legend before comparing.
#    - Combining D1/1 with D9/0 creates a new interaction: after lookup, both guide colors collapse
#      to the same control color, so even the post-lookup color signal is insufficient without the
#      run-length signal.
# d) Placement legality:
#    - The legend occupies a reserved 4x4 corner box whose inner 2x2 cells hold the lookup table.
#      Generator places it in the corner opposite the control corner so it never touches control runs.
#    - The guide-bearing object is placed strictly inside the frame and outside the reserved legend
#      box, so it does not overlap the legend or the border controls and stays as one connected
#      component separate from the legend component.
#    - Control runs stay away from the outer corners, so the reserved legend corner remains blank
#      except for the legend itself and can be recognized reliably.
# e) Visual observability:
#    - A human sees: two same-colored border runs of different lengths, two different guide colors
#      on the object, and a tiny corner lookup table converting those guide colors into the single
#      frame control color. Across examples, the only stable explanation is "read the legend, then
#      align by side while also matching the exact run lengths."
# f) Visual cleanliness:
#    - The legend is isolated in an otherwise empty 4x4 corner block, so it is not mixed with the
#      central object or border controls.
#    - Unused frame sides stay blank rather than noisy, keeping attention on the two meaningful
#      control runs and the small legend.
#    - The object remains one large rectangular component while the legend is a tiny corner block,
#      making the two functional regions easy to separate both for humans and for the solver.
# g) Color-change inferability:
#    - Only two guide colors participate in the lookup, and both map to one repeated control color.
#      This is a very small palette transformation with explicit structural support from the legend.
#      The mapping is not hidden in arbitrary recoloring; it is spatially localized and repeated.
#
# New solver assumptions introduced by the combo, and how generator guarantees them:
# - Exactly one corner contains a clean 4x4 reserved block whose inner 2x2 cells are the legend;
#   generator always paints that exact pattern and keeps the rest of the 4x4 block background.
# - There are exactly two meaningful border runs of length >= 2; generator paints only the selected
#   horizontal and vertical control runs and leaves other sides blank.
# - The legend maps both guide colors to the same control color; generator enforces this so length
#   remains necessary after legend lookup.
# - The object has exactly two guide sides and each guide side contains one contiguous run on an
#   otherwise background auxiliary strip; generator constructs guides in that exact format.
# - The two guide lengths differ and the core is non-square; generator rejects accidental ties and
#   forces `core_h != core_w`.
# - The largest non-border connected component is the object, not the legend; generator keeps the
#   legend tiny and the object much larger.

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_ARC_UTILS_DIR = _ROOT / "arc2_solver_select"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


SIDES = ("top", "bottom", "left", "right")
OPPOSITE_LEGEND_CORNER = {
    ("top", "left"): "bottom_right",
    ("top", "right"): "bottom_left",
    ("bottom", "left"): "top_right",
    ("bottom", "right"): "top_left",
}


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
    start: int | None = None
    color: int | None = None
    for idx, val in enumerate(values + [int(bg)]):
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


def _legend_origins(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    H, W = map(int, shape)
    return {
        "top_left": (0, 0),
        "top_right": (0, W - 4),
        "bottom_left": (H - 4, 0),
        "bottom_right": (H - 4, W - 4),
    }


def _decode_legend(arr: np.ndarray, bg: int) -> dict[int, int]:
    expected = np.zeros((4, 4), dtype=bool)
    expected[1:3, 1:3] = True
    for _corner, (r0, c0) in _legend_origins(arr.shape).items():
        block = arr[int(r0) : int(r0) + 4, int(c0) : int(c0) + 4]
        if block.shape != (4, 4):
            continue
        mask = block != int(bg)
        if not np.array_equal(mask, expected):
            continue
        legend = block[1:3, 1:3]
        mapping: dict[int, int] = {}
        ok = True
        for rr in range(2):
            src = int(legend[rr, 0])
            dst = int(legend[rr, 1])
            if src == int(bg) or dst == int(bg) or src in mapping:
                ok = False
                break
            mapping[int(src)] = int(dst)
        if ok:
            return mapping
    return {}


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


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the corner legend, align the object by mapped color + exact run length, then crop guides.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend = _decode_legend(g, int(bg))
    desired = _desired_features_from_grid(g)
    obj = _extract_center_object(g, int(bg))

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            ok = True
            for side, (guide_color, guide_len) in features.items():
                mapped = legend.get(int(guide_color))
                target_color, target_len = desired[side]
                if mapped != int(target_color) or int(guide_len) != int(target_len):
                    ok = False
                    break
            if ok:
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

    return core


def _attach_guides(
    core: np.ndarray,
    horiz_side: str,
    vert_side: str,
    horiz_guide_color: int,
    vert_guide_color: int,
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
        out[0, int(start) : int(start + horiz_len)] = int(horiz_guide_color)
    if has_bottom:
        start = c0 + random.randint(0, w - horiz_len)
        out[-1, int(start) : int(start + horiz_len)] = int(horiz_guide_color)
    if has_left:
        start = r0 + random.randint(0, h - vert_len)
        out[int(start) : int(start + vert_len), 0] = int(vert_guide_color)
    if has_right:
        start = r0 + random.randint(0, h - vert_len)
        out[int(start) : int(start + vert_len), -1] = int(vert_guide_color)

    return out


def _paint_frame(
    grid: np.ndarray,
    horiz_side: str,
    vert_side: str,
    ctrl_color: int,
    horiz_len: int,
    vert_len: int,
) -> None:
    n = grid.shape[0]
    c0 = random.randint(1, n - 1 - horiz_len)
    if horiz_side == "top":
        grid[0, int(c0) : int(c0 + horiz_len)] = int(ctrl_color)
    else:
        grid[-1, int(c0) : int(c0 + horiz_len)] = int(ctrl_color)

    r0 = random.randint(1, n - 1 - vert_len)
    if vert_side == "left":
        grid[int(r0) : int(r0 + vert_len), 0] = int(ctrl_color)
    else:
        grid[int(r0) : int(r0 + vert_len), -1] = int(ctrl_color)

    grid[0, 0] = 0
    grid[0, -1] = 0
    grid[-1, 0] = 0
    grid[-1, -1] = 0


def _legend_bbox(shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    H, W = map(int, shape)
    if corner == "top_left":
        return 0, 0, 3, 3
    if corner == "top_right":
        return 0, W - 4, 3, W - 1
    if corner == "bottom_left":
        return H - 4, 0, H - 1, 3
    return H - 4, W - 4, H - 1, W - 1


def _paint_legend(grid: np.ndarray, corner: str, guide_colors: tuple[int, int], ctrl_color: int) -> None:
    H, W = map(int, grid.shape)
    if corner == "top_left":
        r0, c0 = 1, 1
    elif corner == "top_right":
        r0, c0 = 1, W - 3
    elif corner == "bottom_left":
        r0, c0 = H - 3, 1
    else:
        r0, c0 = H - 3, W - 3
    grid[int(r0), int(c0)] = int(guide_colors[0])
    grid[int(r0), int(c0) + 1] = int(ctrl_color)
    grid[int(r0) + 1, int(c0)] = int(guide_colors[1])
    grid[int(r0) + 1, int(c0) + 1] = int(ctrl_color)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_object_location(
    grid_shape: tuple[int, int],
    obj_shape: tuple[int, int],
    forbidden: list[tuple[int, int, int, int]],
    border: int = 1,
) -> tuple[int, int]:
    H, W = map(int, grid_shape)
    h, w = map(int, obj_shape)
    r_min = int(border)
    c_min = int(border)
    r_max = H - int(border) - h
    c_max = W - int(border) - w
    if r_max < r_min or c_max < c_min:
        raise ValueError("no legal placement window")
    for _ in range(400):
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        box = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        if any(_boxes_intersect(box, bad) for bad in forbidden):
            continue
        return int(r0), int(c0)
    raise ValueError("failed to place object")


def _count_direct_color_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = {side: color for side, (color, _length) in _desired_features_from_grid(g).items()}
    obj = _extract_center_object(g, int(bg))
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(int(features[side][0]) == int(desired[side]) for side in desired):
                count += 1
    return int(count)


def _count_mapped_color_only_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend = _decode_legend(g, int(bg))
    desired = {side: color for side, (color, _length) in _desired_features_from_grid(g).items()}
    obj = _extract_center_object(g, int(bg))
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            features = _guide_features(t, int(bg))
            if set(features) != set(desired):
                continue
            if all(int(legend.get(int(features[side][0]), -1)) == int(desired[side]) for side in desired):
                count += 1
    return int(count)


def _count_full_matches(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    legend = _decode_legend(g, int(bg))
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
            ok = True
            for side, (guide_color, guide_len) in features.items():
                mapped = legend.get(int(guide_color))
                target_color, target_len = desired[side]
                if mapped != int(target_color) or int(guide_len) != int(target_len):
                    ok = False
                    break
            if ok:
                count += 1
    return int(count)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 31)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )
    legend_corner = OPPOSITE_LEGEND_CORNER[(horiz_side, vert_side)]

    all_colors = list(range(1, 10))
    ctrl_color = random.choice(all_colors)
    remaining = [c for c in all_colors if c != ctrl_color]
    guide_colors = tuple(random.sample(remaining, 2))
    remaining = [c for c in remaining if c not in set(guide_colors)]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(18, n - 8)
    max_core_w = min(18, n - 8)
    if max_core_h < 6 or max_core_w < 6:
        return None

    for _ in range(50):
        core_h = random.randint(6, max_core_h)
        core_w = random.randint(6, max_core_w)
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
        if max_vert_len >= vert_len + 1:
            vert_len += 1
        elif max_vert_len > 3:
            vert_len -= 1
        elif max_horiz_len >= horiz_len + 1:
            horiz_len += 1
        elif max_horiz_len > 3:
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
        int(guide_colors[0]),
        int(guide_colors[1]),
        int(horiz_len),
        int(vert_len),
    )

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, int(ctrl_color), int(horiz_len), int(vert_len))
    _paint_legend(grid, legend_corner, guide_colors, int(ctrl_color))

    try:
        r0, c0 = _random_object_location(
            grid.shape,
            input_object.shape,
            [_legend_bbox(grid.shape, legend_corner)],
            border=1,
        )
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), core):
        return None
    if _count_direct_color_matches(grid) != 0:
        return None
    if _count_mapped_color_only_matches(grid) < 2:
        return None
    if _count_full_matches(grid) != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo 9")
