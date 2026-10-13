import random
from collections import deque

import numpy as np


BG = 8
MARKER = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATE_RETRIES = 400
PLACEMENT_TRIES = 180


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _pair_rectangles(spec: dict) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    top = spec["top"]
    left = spec["left"]
    lower_h = spec["lower_h"]
    lower_w = spec["lower_w"]
    overlap_h = spec["overlap_h"]
    overlap_w = spec["overlap_w"]
    extra_h = spec["extra_h"]
    extra_w = spec["extra_w"]
    upper_h = overlap_h + extra_h
    upper_w = overlap_w + extra_w
    corner = spec["corner"]

    if corner == "tl":
        lower = (top + extra_h, left + extra_w, top + extra_h + lower_h - 1, left + extra_w + lower_w - 1)
        upper = (top, left, top + upper_h - 1, left + upper_w - 1)
    elif corner == "tr":
        lower = (top + extra_h, left, top + extra_h + lower_h - 1, left + lower_w - 1)
        upper = (top, left + lower_w - overlap_w, top + upper_h - 1, left + lower_w - overlap_w + upper_w - 1)
    elif corner == "bl":
        lower = (top, left + extra_w, top + lower_h - 1, left + extra_w + lower_w - 1)
        upper = (top + lower_h - overlap_h, left, top + lower_h - overlap_h + upper_h - 1, left + upper_w - 1)
    else:
        lower = (top, left, top + lower_h - 1, left + lower_w - 1)
        upper = (
            top + lower_h - overlap_h,
            left + lower_w - overlap_w,
            top + lower_h - overlap_h + upper_h - 1,
            left + lower_w - overlap_w + upper_w - 1,
        )
    return lower, upper


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find overlapping rectangle pairs, infer which rectangle is on top, and stamp the overlap's point reflection inside the lower rectangle.

    Input structure:
    - The background is azure.
    - The grid contains several isolated pairs of partially overlapping solid rectangles, with two different colors per pair.
    - In each pair, the overlap is visible only in the upper rectangle's color, so one color forms a full rectangle while the other appears as an L-shape or a rectangle with a missing corner.
    - A single grey marker pixel sits immediately above the horizontal center of each pair's combined bounding box.

    Rule:
    - Split the non-background, non-grey pixels into connected pair-components.
    - Within each component, the color whose visible cells form a complete axis-aligned rectangle is the upper layer; the other color belongs to the lower rectangle.
    - Recover the hidden overlap as the intersection of the upper rectangle bbox and the lower rectangle bbox.
    - Reflect every overlap cell by 180 degrees around the center of the lower rectangle and repaint those reflected cells with the upper color, clipping to the lower rectangle if needed.
    - Leave the original upper rectangle, the grey markers, and the azure background unchanged.

    Key insight:
    - Occlusion changes only one rectangle's visible shape: the intact rectangle reveals the upper layer, and the lower rectangle's missing-corner bbox gives the symmetry center for where the upper color must be stamped.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    pair_mask = (g != BG) & (g != MARKER)
    for comp in _components(pair_mask):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 2:
            continue

        color_info = []
        for color in colors:
            coords = [(r, c) for r, c in comp if int(g[r, c]) == color]
            bbox = _bbox(coords)
            color_info.append((color, coords, bbox, len(coords) == _bbox_area(bbox)))

        upper_candidates = [item for item in color_info if item[3]]
        if len(upper_candidates) != 1:
            continue
        upper_color, _, upper_bbox, _ = upper_candidates[0]
        _, _, lower_bbox, _ = next(item for item in color_info if item[0] != upper_color)

        overlap = _intersect(upper_bbox, lower_bbox)
        if overlap is None:
            continue

        lr0, lc0, lr1, lc1 = lower_bbox
        or0, oc0, or1, oc1 = overlap
        for r in range(or0, or1 + 1):
            for c in range(oc0, oc1 + 1):
                mr = lr0 + lr1 - r
                mc = lc0 + lc1 - c
                if lr0 <= mr <= lr1 and lc0 <= mc <= lc1:
                    out[mr, mc] = upper_color

    return out


def _sample_pair_spec(idx: int) -> dict:
    lower_h = random.randint(4, 7)
    lower_w = random.randint(4, 7)
    overlap_h = random.randint(1, lower_h - 1)
    overlap_w = random.randint(1, lower_w - 1)
    extra_h = random.randint(1, min(3, 8 - overlap_h))
    extra_w = random.randint(1, min(3, 8 - overlap_w))
    return {
        "idx": idx,
        "corner": random.choice(("tl", "tr", "bl", "br")),
        "lower_h": lower_h,
        "lower_w": lower_w,
        "overlap_h": overlap_h,
        "overlap_w": overlap_w,
        "extra_h": extra_h,
        "extra_w": extra_w,
        "pair_h": lower_h + extra_h,
        "pair_w": lower_w + extra_w,
    }


def _choose_grid_shape(specs: list[dict]) -> tuple[int, int] | None:
    min_h = max(spec["pair_h"] for spec in specs) + 3
    min_w = max(spec["pair_w"] for spec in specs) + 3
    reserve_area = sum((spec["pair_h"] + 3) * (spec["pair_w"] + 3) for spec in specs)
    for _ in range(80):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        if h * w >= reserve_area:
            return h, w
    return None


def _place_specs(specs: list[dict], h: int, w: int) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    for spec in sorted(specs, key=lambda item: item["pair_h"] * item["pair_w"], reverse=True):
        top_lo = 2
        top_hi = h - spec["pair_h"] - 1
        left_lo = 1
        left_hi = w - spec["pair_w"] - 1
        if top_lo > top_hi or left_lo > left_hi:
            return False

        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(top_lo, top_hi)
            left = random.randint(left_lo, left_hi)
            rr0 = top - 2
            rr1 = top + spec["pair_h"]
            cc0 = left - 1
            cc1 = left + spec["pair_w"]
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            spec["top"] = top
            spec["left"] = left
            placed = True
            break
        if not placed:
            return False
    return True


def _render_pair(grid: np.ndarray, spec: dict) -> None:
    lower, upper = _pair_rectangles(spec)
    _paint_rect(grid, lower, spec["lower_color"])
    _paint_rect(grid, upper, spec["upper_color"])

    marker_col = spec["left"] + (spec["pair_w"] - 1) // 2
    grid[spec["top"] - 1, marker_col] = MARKER


def _apply_expected_pair(out: np.ndarray, spec: dict) -> None:
    lower, upper = _pair_rectangles(spec)
    upper_color = spec["upper_color"]

    overlap = _intersect(lower, upper)
    assert overlap is not None
    or0, oc0, or1, oc1 = overlap
    lr0, lc0, lr1, lc1 = lower
    for r in range(or0, or1 + 1):
        for c in range(oc0, oc1 + 1):
            mr = lr0 + lr1 - r
            mc = lc0 + lc1 - c
            if lr0 <= mr <= lr1 and lc0 <= mc <= lc1:
                out[mr, mc] = upper_color


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.randint(1, 3)
    specs = [_sample_pair_spec(i) for i in range(pair_count)]

    shape = _choose_grid_shape(specs)
    if shape is None:
        return None
    h, w = shape
    if not _place_specs(specs, h, w):
        return None

    palette = list(OBJECT_COLORS)
    random.shuffle(palette)
    while len(palette) < 2 * pair_count:
        palette.extend(random.sample(OBJECT_COLORS, len(OBJECT_COLORS)))

    grid = np.full((h, w), BG, dtype=int)
    for i, spec in enumerate(specs):
        lower_color, upper_color = palette[2 * i], palette[2 * i + 1]
        if lower_color == upper_color:
            return None
        spec["lower_color"] = lower_color
        spec["upper_color"] = upper_color
        _render_pair(grid, spec)

    out = grid.copy()
    for spec in specs:
        _apply_expected_pair(out, spec)

    if not np.array_equal(main(grid), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        built = _build_example()
        if built is None:
            continue
        grid, _ = built
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
