# Variant of fc7cae8d
# Dimension: D4 — 变换类型 (transform)
# Change: 对齐后裁掉引导边（crop），输出纯核心矩形... → 裁剪后擦除填充色
#
# Design analysis for this variant
# - Reference-task note:
#   The requested source task `58f5dbd5` is present in the evaluation data, but its
#   solver source is not available in this workspace. I therefore used its official
#   I/O pairs as the baseline for the "erase" mechanism: keep the motif/border
#   colors, but replace the dominant support/fill color with background 0.
# - Concrete behavior trace:
#   Suppose alignment yields a cropped core
#       [[4, 4, 1],
#        [4, 3, 4],
#        [4, 4, 4]]
#   Original fc7cae8d outputs that 3x3 block unchanged.
#   This D4 variant performs one extra step after the same crop:
#   dominant fill color = 4, so output becomes
#       [[0, 0, 1],
#        [0, 3, 0],
#        [0, 0, 0]]
#   The input construction and D4 matching stay the same; only the post-crop
#   transform changes.
# - Complexity calibration:
#   If the core were almost empty or perfectly symmetric, the new behavior would
#   be visually weak. The generator therefore keeps the fill clearly dominant
#   (so the solver can identify it), but also paints multi-color patterns that
#   touch all four sides of the core. After erasing the fill, the output still
#   spans the full cropped rectangle and looks very different from the original
#   dense block.
# - Implicit assumptions review:
#   `detect_controls_frame` assumes exactly two near-solid control edges on the
#   outer frame; the generator still builds one horizontal and one vertical
#   control side and only adds noise to the other two sides.
#   `extract_center_object` assumes the interior object is the largest non-frame
#   component; the generator places exactly one guided object strictly inside the
#   frame.
#   `guide_edges` assumes the object's dominant color is the core fill, not a
#   guide or pattern color. The generator enforces fill-majority by capping the
#   pattern density below 40 percent.
#   The crop indices assume the aligned object has guides on exactly the two
#   desired sides. `_attach_guides` preserves that invariant.
#   The new `_erase_fill` step assumes the cropped core still has one dominant
#   support color; generator construction guarantees that.
# - Placement legality:
#   The object is transformed first, then inserted with `random_free_location`
#   using `transparent=0` and `border=1`, so it cannot overlap the colored frame
#   and always stays fully interior.
# - Observability:
#   Across examples, the original task outputs a solid rectangle; this variant
#   outputs the same aligned/cropped rectangle size but with most cells turned
#   black. That sparse-vs-dense contrast is directly visible from a few examples.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.insert(0, ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the central rectangle to two colored frame-edge controls via a D4 transform,
    crop away the matching 1px guide edges, then erase the dominant core fill to 0.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def detect_controls_frame(arr: np.ndarray) -> list[tuple[str, int]]:
        H, W = arr.shape
        sides = {
            "top": arr[0, 1:-1],
            "bottom": arr[-1, 1:-1],
            "left": arr[1:-1, 0],
            "right": arr[1:-1, -1],
        }
        cand: list[tuple[str, int, int, int]] = []
        for side, v in sides.items():
            v = list(map(int, v))
            v_nb = [x for x in v if x != bg]
            if not v_nb:
                continue
            col, count = Counter(v_nb).most_common(1)[0]
            if int(count) >= len(v) - 1:
                cand.append((side, int(col), int(count), int(len(v))))
        cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
        return [(side, col) for side, col, _count, _length in cand[:2]]

    def extract_center_object(arr: np.ndarray) -> np.ndarray:
        H, W = arr.shape
        comps = find_components(arr, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
        best = None
        for comp in comps:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
                continue
            area = int(comp["area"])
            if best is None or area > best[0]:
                best = (area, (r0, c0, r1, c1))
        if best is None:
            return arr.copy()
        return crop_bbox(arr, best[1])

    def guide_edges(arr: np.ndarray) -> dict[str, int]:
        fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
        edges = {
            "top": arr[0, :],
            "bottom": arr[-1, :],
            "left": arr[:, 0],
            "right": arr[:, -1],
        }
        out: dict[str, int] = {}
        for side, v in edges.items():
            v = list(map(int, v))
            v_nf = [x for x in v if x != fill]
            if not v_nf:
                continue
            col, count = Counter(v_nf).most_common(1)[0]
            if int(count) >= len(v) - 1:
                out[side] = int(col)
        return out

    def crop_guides(arr: np.ndarray, sides: dict[str, int]) -> np.ndarray:
        top = 1 if "top" in sides else 0
        bottom = int(arr.shape[0]) - (1 if "bottom" in sides else 0)
        left = 1 if "left" in sides else 0
        right = int(arr.shape[1]) - (1 if "right" in sides else 0)
        return arr[int(top) : int(bottom), int(left) : int(right)]

    def erase_fill(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.copy()
        fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
        out = np.array(arr, copy=True)
        out[out == fill] = bg
        return out

    desired = {side: col for side, col in detect_controls_frame(g)}
    obj = extract_center_object(g)

    for k in range(4):
        base = np.rot90(obj, k)
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            gedges = guide_edges(transformed)
            if all(side in gedges and gedges[side] == col for side, col in desired.items()):
                return erase_fill(crop_guides(transformed, desired))

    fallback_edges = guide_edges(obj)
    return erase_fill(crop_guides(obj, fallback_edges))


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _erase_fill(core: np.ndarray, fill: int, bg: int = 0) -> np.ndarray:
    out = np.array(core, copy=True)
    out[out == int(fill)] = int(bg)
    return out


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    min_nonfill = max(len(pattern_colors) + 4, int(np.ceil(area * 0.22)))
    max_nonfill = max(min_nonfill, int(np.floor(area * 0.38)))
    target_nonfill = random.randint(min_nonfill, max_nonfill)

    seeds: list[tuple[int, int, int]] = []
    for color in pattern_colors:
        seeds.append((random.randrange(h), random.randrange(w), int(color)))
    seeds.extend(
        [
            (0, random.randrange(w), int(random.choice(pattern_colors))),
            (h - 1, random.randrange(w), int(random.choice(pattern_colors))),
            (random.randrange(h), 0, int(random.choice(pattern_colors))),
            (random.randrange(h), w - 1, int(random.choice(pattern_colors))),
        ]
    )
    for r, c, color in seeds:
        core[r, c] = color

    painted = int(np.count_nonzero(core != fill))
    attempts = 0
    while painted < target_nonfill and attempts < area * 12:
        attempts += 1
        color = int(random.choice(pattern_colors))
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            horizontal = random.random() < 0.5
            if horizontal:
                line_len = random.randint(2, min(5, w))
                rr = random.randrange(h)
                c0 = random.randrange(w - line_len + 1)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                line_len = random.randint(2, min(5, h))
                r0 = random.randrange(h - line_len + 1)
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
            if core[r, c] != fill:
                continue
            core[r, c] = color
            painted += 1

    if painted < target_nonfill:
        fill_cells = list(zip(*np.where(core == fill)))
        for r, c in random.sample(fill_cells, min(target_nonfill - painted, len(fill_cells))):
            core[r, c] = int(random.choice(pattern_colors))

    for coords in {
        "top": [(0, c) for c in range(w)],
        "bottom": [(h - 1, c) for c in range(w)],
        "left": [(r, 0) for r in range(h)],
        "right": [(r, w - 1) for r in range(h)],
    }.values():
        if all(core[r, c] == fill for r, c in coords):
            r, c = random.choice(coords)
            core[r, c] = int(random.choice(pattern_colors))

    return core


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])

    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(20, n - 3)
    max_core_w = min(20, n - 3)
    if max_core_h < 5 or max_core_w < 5:
        return None

    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)
    expected = _erase_fill(core, fill)

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D4/3 variant")
