# Combo variant of fc7cae8d (index 42)
# combo_spec: D3/1+D9/2+D10/1
# Name: 非实心控制+控制像素+多色噪声

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDES = ("top", "bottom", "left", "right")
CORNER_PAIRS = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))


def _side_values(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, 1:-1]
    if side == "bottom":
        return arr[-1, 1:-1]
    if side == "left":
        return arr[1:-1, 0]
    if side == "right":
        return arr[1:-1, -1]
    raise ValueError(f"unknown side: {side}")


def _side_coords(n: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, n - 1)]
    if side == "bottom":
        return [(n - 1, c) for c in range(1, n - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, n - 1)]
    if side == "right":
        return [(r, n - 1) for r in range(1, n - 1)]
    raise ValueError(f"unknown side: {side}")


def _summarize_side(arr: np.ndarray, side: str, bg: int) -> dict[str, object]:
    vals = list(map(int, _side_values(arr, side)))
    counts = Counter(v for v in vals if v != int(bg))
    dominant = None
    dom_count = 0
    if counts:
        dominant, dom_count = counts.most_common(1)[0]
    payload = [(idx, val) for idx, val in enumerate(vals) if val not in {int(bg), int(dominant) if dominant is not None else -1}]
    return {
        "side": side,
        "vals": vals,
        "dominant": None if dominant is None else int(dominant),
        "dom_count": int(dom_count),
        "dom_ratio": float(dom_count / len(vals)) if vals else 0.0,
        "nonbg_ratio": float(sum(v != int(bg) for v in vals) / len(vals)) if vals else 0.0,
        "payload": [(int(idx), int(val)) for idx, val in payload],
    }


def _detect_controls_frame(arr: np.ndarray, bg: int) -> dict[str, int]:
    infos = {side: _summarize_side(arr, side, bg) for side in SIDES}

    best_pair = None
    best_score = -1.0
    for s1, s2 in CORNER_PAIRS:
        i1 = infos[s1]
        i2 = infos[s2]
        if i1["dominant"] is None or i2["dominant"] is None:
            continue
        if int(i1["dominant"]) != int(i2["dominant"]):
            continue
        if float(i1["dom_ratio"]) < 0.72 or float(i2["dom_ratio"]) < 0.72:
            continue
        if len(i1["payload"]) != 1 or len(i2["payload"]) != 1:
            continue
        score = float(i1["dom_ratio"]) + float(i2["dom_ratio"]) + 0.1 * (
            float(i1["nonbg_ratio"]) + float(i2["nonbg_ratio"])
        )
        if score > best_score:
            best_score = score
            best_pair = (s1, s2)

    if best_pair is None:
        # Generator guarantees a clean pair; this fallback keeps main total.
        ranked = sorted(infos.values(), key=lambda info: float(info["dom_ratio"]), reverse=True)
        candidate_sides = [str(info["side"]) for info in ranked[:2]]
        best_pair = None
        for s1, s2 in CORNER_PAIRS:
            if {s1, s2} == set(candidate_sides):
                best_pair = (s1, s2)
                break
        if best_pair is None:
            best_pair = CORNER_PAIRS[0]

    desired: dict[str, int] = {}
    for side in best_pair:
        payload = infos[side]["payload"]
        if payload:
            desired[side] = int(payload[0][1])
    return desired


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, vals in edges.items():
        vec = list(map(int, vals))
        off_fill = [v for v in vec if v != int(fill)]
        if not off_fill:
            continue
        col, count = Counter(off_fill).most_common(1)[0]
        if int(count) >= len(vec) - 1:
            out[side] = int(col)
    return out


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion design notes
    -------------------
    a) Concrete behavior:
       - Original task: two adjacent border edges were solid control colors, and the object's guide edges used those same colors.
       - New task: the chosen corner still determines the output orientation, but the border signal is split in two layers:
         1. Two adjacent border sides are dense "carrier" sides: the same carrier color fills about 80-90% of each side, with a
            few background gaps.
         2. Each dense side contains exactly one non-carrier control pixel. The side position says which destination edge it
            controls; the control-pixel color says which object guide color must face that side.
         3. The other two border sides are sparse multi-color noise containing carrier and guide colors too, so color uniqueness
            is no longer reliable.
       - Example sketch:
         * top dense side: mostly 8, one pixel 3 -> top wants guide color 3
         * left dense side: mostly 8, one pixel 6 -> left wants guide color 6
         * bottom/right: scattered {8,3,6,5}
         * object currently has guides {bottom: 3, right: 6}; solver must rotate/reflect until {top: 3, left: 6}.

    b) Complexity calibration:
       - The carrier side uses one shared dominant color on both active sides, giving a strong but not trivial visual cue.
       - Noise sides are capped well below the carrier density, so there is one clear corner pair instead of many ambiguous ones.
       - Control pixels remain singletons so the extra decoding step is visible but still learnable from a few examples.

    c) Which original assumptions broke:
       - Original `detect_controls_frame` assumed each active side was almost entirely one of the final guide colors.
         That fails because dense sides now share a carrier color instead.
       - Original solver could identify control sides from color identity alone; that fails because noise sides also reuse carrier
         and guide colors.
       - The object-side logic still assumes guide strips are nearly full edges after crop and transform. Generator preserves that.

    d) Placement legality:
       - Generator keeps the object at least one cell away from the border, so border noise never merges into the object component.
       - Border decorations use only border cells; corners are ignored by solver and remain non-essential.

    e/f/g) Human observability and visual cleanliness:
       - A human can first spot the two unusually dense adjacent sides, then notice the single odd-colored pixel on each.
       - Noise is multi-color but restricted to the opposite sides, so the frame stays legible instead of becoming uniformly messy.
       - Only two guide colors are mapped, and each is anchored by both an object guide strip and one border control pixel.

    New solver assumptions introduced here and guaranteed by the generator:
       - Exactly one adjacent side-pair shares a high-density dominant carrier color.
       - Each active side has exactly one non-background, non-carrier pixel.
       - The central object is the largest non-border connected component.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    desired = _detect_controls_frame(g, bg)
    obj = _extract_center_object(g, bg)
    if len(desired) != 2:
        return obj

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            edges = _guide_edges(t)
            if all(side in edges and int(edges[side]) == int(color) for side, color in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                return t[int(top) : int(bottom), int(left) : int(right)]

    return obj


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors), int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors), area // 10), max_nonfill)
    painted = 0

    for color in pattern_colors:
        r = random.randrange(h)
        c = random.randrange(w)
        if core[r, c] == int(fill):
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

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        new_cells = [(r, c) for r, c in cells if core[r, c] == int(fill)]
        if len(new_cells) > remaining:
            new_cells = random.sample(new_cells, remaining)
        for r, c in new_cells:
            core[r, c] = int(color)
        painted += len(new_cells)

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


def _paint_control_side(grid: np.ndarray, side: str, carrier: int, signal_color: int) -> None:
    coords = _side_coords(grid.shape[0], side)
    length = len(coords)
    low = max(1, math.ceil(0.80 * length) - 1)
    high = min(length - 2, math.floor(0.88 * length) - 1)
    if high < low:
        high = low
    carrier_count = random.randint(low, high)

    if length >= 7:
        signal_idx = random.randint(2, length - 3)
    else:
        signal_idx = random.randrange(length)
    carrier_pool = [idx for idx in range(length) if idx != signal_idx]
    carrier_positions = set(random.sample(carrier_pool, carrier_count))

    for idx, (r, c) in enumerate(coords):
        if idx in carrier_positions:
            grid[r, c] = int(carrier)
    sr, sc = coords[signal_idx]
    grid[sr, sc] = int(signal_color)


def _paint_noise_side(grid: np.ndarray, side: str, palette: list[int]) -> None:
    coords = _side_coords(grid.shape[0], side)
    length = len(coords)
    count = random.randint(math.ceil(0.35 * length), math.floor(0.55 * length))

    for _ in range(100):
        idxs = random.sample(range(length), count)
        random.shuffle(idxs)
        assignments: dict[int, int] = {}
        anchors = [palette[0], palette[1], palette[2]]
        for idx, color in zip(idxs[:3], anchors):
            assignments[int(idx)] = int(color)
        for idx in idxs[3:]:
            assignments[int(idx)] = int(random.choice(palette))
        counts = Counter(assignments.values())
        if len(counts) >= 3 and max(counts.values()) <= math.floor(0.45 * length):
            for idx, color in assignments.items():
                r, c = coords[int(idx)]
                grid[r, c] = int(color)
            return

    chosen = random.sample(range(length), count)
    for idx in chosen:
        r, c = coords[idx]
        grid[r, c] = int(random.choice(palette))


def _frame_signature(grid: np.ndarray) -> dict[str, dict[str, object]]:
    bg = int(get_bg(grid))
    return {side: _summarize_side(grid, side, bg) for side in SIDES}


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(CORNER_PAIRS)

    all_colors = list(range(1, 10))
    guide_h, guide_v, carrier, fill = random.sample(all_colors, 4)
    remaining = [c for c in all_colors if c not in {guide_h, guide_v, carrier, fill}]
    pattern_colors = random.sample(remaining, random.randint(2, 3))
    noise_extras = random.sample([c for c in remaining if c not in pattern_colors], 2)

    max_core_h = min(20, n - 3)
    max_core_w = min(20, n - 3)
    if max_core_h < 5 or max_core_w < 5:
        return None
    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    guide_colors = {horiz_side: guide_h, vert_side: guide_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_control_side(grid, horiz_side, carrier, guide_h)
    _paint_control_side(grid, vert_side, carrier, guide_v)

    noise_palette = [carrier, guide_h, guide_v, noise_extras[0], noise_extras[1]]
    for side in SIDES:
        if side in (horiz_side, vert_side):
            continue
        _paint_noise_side(grid, side, noise_palette)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    sig = _frame_signature(grid)
    active = [sig[horiz_side], sig[vert_side]]
    inactive = [sig[side] for side in SIDES if side not in (horiz_side, vert_side)]
    if any(len(info["payload"]) != 1 for info in active):
        return None
    if any(int(info["dominant"]) != int(carrier) for info in active):
        return None
    if any(float(info["dom_ratio"]) < 0.75 for info in active):
        return None
    if any(float(info["dom_ratio"]) > 0.62 for info in inactive):
        return None

    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")
