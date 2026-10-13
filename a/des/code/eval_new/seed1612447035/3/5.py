from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
GROUP_COUNT_CHOICES = (1, 2, 2, 3)
RING_COUNT_CHOICES = (2, 3)  # 4 layers cannot fit the exact pixel-count line in <=30x30.
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400
RESERVE_MARGIN = 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components4_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _ring_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = border & ~mask
    if int(missing.sum()) != 1:
        return None

    gap_r, gap_c = map(int, np.argwhere(missing)[0])
    if gap_r == 0 and 0 < gap_c < w - 1:
        side = "top"
        if gap_c not in {w // 2, (w - 1) // 2}:
            return None
    elif gap_r == h - 1 and 0 < gap_c < w - 1:
        side = "bottom"
        if gap_c not in {w // 2, (w - 1) // 2}:
            return None
    elif gap_c == 0 and 0 < gap_r < h - 1:
        side = "left"
        if gap_r not in {h // 2, (h - 1) // 2}:
            return None
    elif gap_c == w - 1 and 0 < gap_r < h - 1:
        side = "right"
        if gap_r not in {h // 2, (h - 1) // 2}:
            return None
    else:
        return None

    expected_area = int(border.sum()) - 1
    if int(comp["area"]) != expected_area:
        return None

    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "area": int(comp["area"]),
        "side": side,
        "center2": (int(r0 + r1), int(c0 + c1)),
        "center": (int((r0 + r1) // 2), int((c0 + c1) // 2)),
    }


def _detect_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    groups_by_key: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for comp in _components4_by_color(g, bg):
        info = _ring_info(comp)
        if info is not None:
            key = (info["center2"][0], info["center2"][1], info["side"])
            groups_by_key[key].append(info)

    groups: list[dict] = []
    for items in groups_by_key.values():
        rings = sorted(items, key=lambda item: item["area"])
        nested = True
        for inner, outer in zip(rings, rings[1:]):
            ir0, ic0, ir1, ic1 = inner["bbox"]
            or0, oc0, or1, oc1 = outer["bbox"]
            if not (or0 <= ir0 and oc0 <= ic0 and or1 >= ir1 and oc1 >= ic1):
                nested = False
                break
        if not nested:
            continue
        groups.append(
            {
                "side": rings[0]["side"],
                "center": rings[0]["center"],
                "rings": rings,
            }
        )
    return groups


def _draw_group_segments(out: np.ndarray, group: dict, bg: int) -> None:
    h, w = out.shape
    side = group["side"]
    cr, cc = group["center"]
    for idx, ring in enumerate(group["rings"]):
        color = int(ring["color"])
        length = int(ring["area"])
        half_lo = (length - 1) // 2
        half_hi = length // 2
        if side == "top":
            row = cr + 2 * idx
            c0, c1 = cc - half_lo, cc + half_hi
            assert 0 <= row < h and 0 <= c0 and c1 < w
            assert np.all((out[row, c0 : c1 + 1] == bg) | (out[row, c0 : c1 + 1] == color))
            out[row, c0 : c1 + 1] = color
        elif side == "bottom":
            row = cr - 2 * idx
            c0, c1 = cc - half_lo, cc + half_hi
            assert 0 <= row < h and 0 <= c0 and c1 < w
            assert np.all((out[row, c0 : c1 + 1] == bg) | (out[row, c0 : c1 + 1] == color))
            out[row, c0 : c1 + 1] = color
        elif side == "left":
            col = cc + 2 * idx
            r0, r1 = cr - half_lo, cr + half_hi
            assert 0 <= col < w and 0 <= r0 and r1 < h
            assert np.all((out[r0 : r1 + 1, col] == bg) | (out[r0 : r1 + 1, col] == color))
            out[r0 : r1 + 1, col] = color
        else:
            col = cc - 2 * idx
            r0, r1 = cr - half_lo, cr + half_hi
            assert 0 <= col < w and 0 <= r0 and r1 < h
            assert np.all((out[r0 : r1 + 1, col] == bg) | (out[r0 : r1 + 1, col] == color))
            out[r0 : r1 + 1, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold each concentric open rectangular-ring group into parallel centered line segments.

    Input structure:
    - bg = the most common color.
    - The foreground is made of several groups of nested one-pixel-thick rectangular rings.
    - Within one group, each ring has a different color and exactly one one-cell gap on the same side midpoint.

    Rule:
    - Detect each ring as a rectangular border missing one midpoint cell on its top, bottom, left, or right side, then group rings that share a center and gap side.
    - Remove the original rings. If the gap is on the top or bottom, replace the group by horizontal segments; if the gap is on the left or right, replace it by vertical segments.
    - Each segment keeps its ring color and has length equal to that ring's total colored-pixel count. The innermost ring becomes the segment closest to the center, and outer rings stay parallel one blank cell farther in the direction opposite the gap.

    Key insight:
    - The missing midpoint marks both which side of the ring was opened and which perpendicular axis the straightened form must use; the ring's pixel count is preserved exactly as the new segment length.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    groups = _detect_groups(g)
    if not groups:
        return g.copy()

    out = g.copy()
    for group in groups:
        for ring in group["rings"]:
            coords = ring["coords"]
            out[coords[:, 0], coords[:, 1]] = bg
    for group in groups:
        _draw_group_segments(out, group, bg)
    return out


def _ring_boxes(spec: dict) -> list[tuple[int, int, int, int]]:
    top = int(spec["top"])
    left = int(spec["left"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    rings = int(spec["rings"])
    return [
        (top + depth, left + depth, outer_h - 2 * depth, outer_w - 2 * depth)
        for depth in range(rings)
    ]


def _ring_length(height: int, width: int) -> int:
    return int(2 * (height + width) - 5)


def _group_center(spec: dict) -> tuple[int, int]:
    return int(spec["top"] + spec["outer_h"] // 2), int(spec["left"] + spec["outer_w"] // 2)


def _draw_open_ring(grid: np.ndarray, top: int, left: int, height: int, width: int, side: str, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _paint_group_input(grid: np.ndarray, spec: dict, bg: int) -> None:
    side = str(spec["side"])
    colors = list(spec["colors_outer_to_inner"])
    for (top, left, height, width), color in zip(_ring_boxes(spec), colors):
        _draw_open_ring(grid, top, left, height, width, side, int(color))
        bottom = top + height - 1
        right = left + width - 1
        mid_r = top + height // 2
        mid_c = left + width // 2
        if side == "top":
            grid[top, mid_c] = bg
        elif side == "bottom":
            grid[bottom, mid_c] = bg
        elif side == "left":
            grid[mid_r, left] = bg
        else:
            grid[mid_r, right] = bg


def _paint_group_output(grid: np.ndarray, spec: dict) -> None:
    side = str(spec["side"])
    cr, cc = _group_center(spec)
    boxes = _ring_boxes(spec)
    for idx, ((top, left, height, width), color) in enumerate(zip(reversed(boxes), reversed(spec["colors_outer_to_inner"]))):
        _ = (top, left)  # bbox order still determines inner-to-outer lengths.
        length = _ring_length(height, width)
        half = length // 2
        if side == "top":
            row = cr + 2 * idx
            grid[row, cc - half : cc + half + 1] = int(color)
        elif side == "bottom":
            row = cr - 2 * idx
            grid[row, cc - half : cc + half + 1] = int(color)
        elif side == "left":
            col = cc + 2 * idx
            grid[cr - half : cr + half + 1, col] = int(color)
        else:
            col = cc - 2 * idx
            grid[cr - half : cr + half + 1, col] = int(color)


def _group_union_bbox(spec: dict) -> tuple[int, int, int, int]:
    top = int(spec["top"])
    left = int(spec["left"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    rings = int(spec["rings"])
    side = str(spec["side"])
    cr, cc = _group_center(spec)
    input_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
    max_len = _ring_length(outer_h, outer_w)
    half = max_len // 2

    if side == "top":
        output_bbox = (cr, cc - half, cr + 2 * (rings - 1), cc + half)
    elif side == "bottom":
        output_bbox = (cr - 2 * (rings - 1), cc - half, cr, cc + half)
    elif side == "left":
        output_bbox = (cr - half, cc, cr + half, cc + 2 * (rings - 1))
    else:
        output_bbox = (cr - half, cc - 2 * (rings - 1), cr + half, cc)

    return (
        min(input_bbox[0], output_bbox[0]),
        min(input_bbox[1], output_bbox[1]),
        max(input_bbox[2], output_bbox[2]),
        max(input_bbox[3], output_bbox[3]),
    )


def _expand_bbox(bbox: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _bbox_fits(bbox: tuple[int, int, int, int], height: int, width: int) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 and 0 <= c0 and r1 < height and c1 < width


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _feasible_outer_dims(rings: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    min_side = 2 * rings + 1
    for outer_h in range(min_side, 16, 2):
        for outer_w in range(min_side, 16, 2):
            if _ring_length(outer_h, outer_w) <= 29:
                dims.append((outer_h, outer_w))
    return dims


FEASIBLE_DIMS = {rings: _feasible_outer_dims(rings) for rings in RING_COUNT_CHOICES}


def _sample_group_specs(height: int, width: int, bg: int) -> list[dict] | None:
    n_groups = int(random.choice(GROUP_COUNT_CHOICES))
    sides = list(SIDES)
    random.shuffle(sides)
    reserved: list[tuple[int, int, int, int]] = []
    specs: list[dict] = []

    for idx in range(n_groups):
        preferred = sides[idx % len(sides)]
        side_pool = [preferred] + [side for side in SIDES if side != preferred]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rings = int(random.choice(RING_COUNT_CHOICES))
            outer_h, outer_w = random.choice(FEASIBLE_DIMS[rings])
            side = random.choice(side_pool[:2]) if random.random() < 0.7 else random.choice(SIDES)
            colors = random.sample([color for color in PALETTE if color != bg], rings)
            placements = []
            for top in range(height - outer_h + 1):
                for left in range(width - outer_w + 1):
                    spec = {
                        "top": top,
                        "left": left,
                        "outer_h": outer_h,
                        "outer_w": outer_w,
                        "rings": rings,
                        "side": side,
                        "colors_outer_to_inner": colors,
                    }
                    bbox = _expand_bbox(_group_union_bbox(spec), RESERVE_MARGIN)
                    if not _bbox_fits(bbox, height, width):
                        continue
                    if any(_boxes_overlap(bbox, other) for other in reserved):
                        continue
                    placements.append((top, left, bbox))
            if not placements:
                continue
            top, left, bbox = random.choice(placements)
            specs.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "outer_h": int(outer_h),
                    "outer_w": int(outer_w),
                    "rings": int(rings),
                    "side": side,
                    "colors_outer_to_inner": list(map(int, colors)),
                }
            )
            reserved.append(bbox)
            placed = True
            break
        if not placed:
            return None
    return specs


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bg = random.choice(PALETTE)
    specs = _sample_group_specs(height, width, bg)
    if specs is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    for spec in specs:
        _paint_group_input(inp, spec, bg)

    expected = np.full_like(inp, bg)
    for spec in specs:
        _paint_group_output(expected, spec)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: exhausted attempts")
