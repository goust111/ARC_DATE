# Combo variant of fc7cae8d (index 32)
# combo_spec: D3/1+D9/2
# Name: 非实心控制边+控制像素

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

# Design notes for the fused variant:
# - Pixel-level behavior:
#   Original fc7cae8d used two solid frame edges whose colors directly matched the
#   object's two guide edges. Here the frame edges become "carrier edges": each
#   control side is only 80-90% filled with a dominant carrier color and contains
#   exactly one off-color signal pixel. The signal pixel's color, not the carrier
#   color, tells which guide color should face that side. The signal pixel is
#   readable only after the solver first identifies which broken border edges are
#   the true controls.
# - Small concrete example:
#   Suppose the top border is mostly 7 with one 2 and a few 0 gaps, while the left
#   border is mostly 8 with one 4 and a few 0 gaps. The inner object has a 2-guide
#   and a 4-guide on two adjacent sides. Original logic would match guide colors to
#   border-majority colors 7/8 and fail; fused logic must read "top wants 2" and
#   "left wants 4" from the embedded signal pixels.
# - Complexity calibration:
#   Control sides use a dominant-ratio band around 0.75-0.90, while the two
#   non-control sides use sparse grey noise around 0.30-0.55. This keeps the
#   carrier-vs-noise contrast visible to humans while forcing statistical detection.
#   Signal pixels are unique per control side and carrier colors are chosen distinct
#   from guide colors, so the variant cannot collapse back to direct color matching.
# - Hidden assumptions audited against the original solver:
#   1. "Frame controls are solid" is broken, so border detection now relies on
#      dominant color ratio plus a unique off-color signal.
#   2. "Frame color equals guide color" is broken, so matching uses signal pixels.
#   3. "Largest non-border component is the object" still holds; generator keeps a
#      single interior object and all border markings on the frame.
#   4. "Guide edges are near-solid strips" still holds; generator keeps original
#      guide-edge construction so dihedral search remains human-legible.
# - Placement legality / visual cleanliness:
#   The generator still places one object with a 1-cell safety border using
#   random_free_location(..., border=1). Border markings never overlap the object,
#   corners are left empty, and carrier sides never share colors with guide sides.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the core rectangle after aligning guide edges to broken carrier edges.

    The two adjacent control sides on the outer frame are not solid anymore:
    each is a mostly-filled carrier edge with one signal pixel embedded in it.
    The containing side gives the destination side; the signal pixel color tells
    which object guide color must face that side.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def detect_frame_instructions(arr: np.ndarray) -> dict[str, int]:
        H, W = arr.shape
        sides = {
            "top": arr[0, 1 : W - 1],
            "bottom": arr[H - 1, 1 : W - 1],
            "left": arr[1 : H - 1, 0],
            "right": arr[1 : H - 1, W - 1],
        }
        candidates: dict[str, dict[str, int | float]] = {}

        for side, values_arr in sides.items():
            values = [int(x) for x in values_arr]
            nonbg = [x for x in values if x != bg]
            if not nonbg:
                continue
            dominant, dominant_count = Counter(nonbg).most_common(1)[0]
            dominant = int(dominant)
            dominant_count = int(dominant_count)
            if dominant == 5:
                continue

            dominant_ratio = dominant_count / float(len(values))
            signal_cells = [(idx, val) for idx, val in enumerate(values) if val not in {bg, dominant}]
            if not (0.72 <= dominant_ratio <= 0.92):
                continue
            if len(signal_cells) != 1:
                continue

            signal_idx, signal_color = signal_cells[0]
            candidates[side] = {
                "signal_color": int(signal_color),
                "dominant_ratio": float(dominant_ratio),
                "signal_idx": int(signal_idx),
            }

        best_pair: tuple[str, str] | None = None
        best_score = -1.0
        for pair in (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")):
            if pair[0] not in candidates or pair[1] not in candidates:
                continue
            score = float(candidates[pair[0]]["dominant_ratio"]) + float(candidates[pair[1]]["dominant_ratio"])
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            return {}
        return {side: int(candidates[side]["signal_color"]) for side in best_pair}

    def extract_center_object(arr: np.ndarray) -> np.ndarray:
        H, W = arr.shape
        comps = find_components(arr, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
        best: tuple[int, tuple[int, int, int, int]] | None = None
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
        if arr.shape[0] >= 3 and arr.shape[1] >= 3:
            fill = int(Counter(map(int, arr[1:-1, 1:-1].flatten())).most_common(1)[0][0])
        else:
            fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
        edges = {
            "top": arr[0, :],
            "bottom": arr[-1, :],
            "left": arr[:, 0],
            "right": arr[:, -1],
        }
        out: dict[str, int] = {}
        for side, values_arr in edges.items():
            values = [int(x) for x in values_arr]
            nonfill = [x for x in values if x != fill]
            if not nonfill:
                continue
            color, count = Counter(nonfill).most_common(1)[0]
            if int(count) >= len(values) - 1:
                out[side] = int(color)
        return out

    desired = detect_frame_instructions(g)
    obj = extract_center_object(g)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = guide_edges(transformed)
            if all(side in guides and int(guides[side]) == int(color) for side, color in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = transformed.shape[0] - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = transformed.shape[1] - (1 if "right" in desired else 0)
                return transformed[int(top) : int(bottom), int(left) : int(right)]

    return obj


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors), int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors), area // 10), max_nonfill)
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

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        fill_cells = [(r, c) for r, c in cells if core[r, c] == fill]
        if len(fill_cells) > remaining:
            fill_cells = random.sample(fill_cells, remaining)
        for r, c in fill_cells:
            core[r, c] = int(color)
        painted += len(fill_cells)
        for r, c in cells:
            if core[r, c] != fill:
                continue
            if painted >= target_nonfill:
                break
            core[r, c] = int(color)
            painted += 1

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


def _side_coords(n: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, n - 1)]
    if side == "bottom":
        return [(n - 1, c) for c in range(1, n - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, n - 1)]
    return [(r, n - 1) for r in range(1, n - 1)]


def _paint_control_side(grid: np.ndarray, side: str, carrier_color: int, signal_color: int) -> None:
    coords = _side_coords(grid.shape[0], side)
    length = len(coords)
    filled_total = random.randint(int(np.ceil(0.82 * length)), int(np.floor(0.90 * length)))
    signal_band = list(range(2, length - 2)) if length >= 6 else list(range(length))
    signal_idx = random.choice(signal_band)
    carrier_slots = [idx for idx in range(length) if idx != signal_idx]
    carrier_count = filled_total - 1
    carrier_indices = set(random.sample(carrier_slots, carrier_count))

    for idx in carrier_indices:
        r, c = coords[idx]
        grid[r, c] = int(carrier_color)
    sr, sc = coords[signal_idx]
    grid[sr, sc] = int(signal_color)


def _paint_noise_side(grid: np.ndarray, side: str) -> None:
    coords = _side_coords(grid.shape[0], side)
    min_count = int(np.ceil(0.30 * len(coords)))
    max_count = int(np.floor(0.55 * len(coords)))
    noise_count = random.randint(min_count, max_count)
    for r, c in random.sample(coords, noise_count):
        grid[r, c] = 5


def _paint_frame(
    grid: np.ndarray,
    horiz_side: str,
    vert_side: str,
    horiz_carrier: int,
    vert_carrier: int,
    horiz_signal: int,
    vert_signal: int,
) -> None:
    _paint_control_side(grid, horiz_side, horiz_carrier, horiz_signal)
    _paint_control_side(grid, vert_side, vert_carrier, vert_signal)
    for side in ("top", "bottom", "left", "right"):
        if side in {horiz_side, vert_side}:
            continue
        _paint_noise_side(grid, side)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    usable_colors = [c for c in range(1, 10) if c != 5]
    guide_h, guide_v, carrier_h, carrier_v = random.sample(usable_colors, 4)
    remaining = [c for c in range(1, 10) if c not in {5, guide_h, guide_v, carrier_h, carrier_v}]
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

    guide_colors = {horiz_side: guide_h, vert_side: guide_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, carrier_h, carrier_v, guide_h, guide_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate fused input for fc7cae8d")
