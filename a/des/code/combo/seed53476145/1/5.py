import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
REGION_COUNT_CHOICES = (1, 2, 3)
REGION_COUNT_WEIGHTS = (3, 4, 2)
OUTER_HEIGHT_RANGE = (8, 13)
OUTER_WIDTH_RANGE = (8, 13)
REGION_GAP = 2
PLACEMENT_RETRIES = 80
EXTRA_POINT_MAX = 6
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _marker_color(line: np.ndarray, wall_color: int) -> int:
    marker_positions = [i for i, v in enumerate(line.tolist()) if int(v) != wall_color]
    assert marker_positions, "each side must contain one colored marker segment"
    lo = min(marker_positions)
    hi = max(marker_positions)
    assert marker_positions == list(range(lo, hi + 1)), "marker segment must be contiguous"
    colors = {int(line[i]) for i in marker_positions}
    assert len(colors) == 1, "each side marker must be monochrome"
    return next(iter(colors))


def _region_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, int, int, int, dict[str, int]]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3, "region border must enclose an interior"

    expected_area = 2 * (h + w) - 4
    assert len(coords) == expected_area, "border component must be a one-cell-thick rectangle"

    for r, c in coords:
        assert r in (r0, r1) or c in (c0, c1), "border pixels must lie on the bbox perimeter"

    border = grid[r0 : r1 + 1, c0 : c1 + 1]
    wall_color = int(border[0, 0])
    assert int(border[0, -1]) == wall_color
    assert int(border[-1, 0]) == wall_color
    assert int(border[-1, -1]) == wall_color

    top = border[0, :]
    bottom = border[-1, :]
    left = border[:, 0]
    right = border[:, -1]
    assert np.all(top != BLACK) and np.all(bottom != BLACK)
    assert np.all(left != BLACK) and np.all(right != BLACK)

    side_colors = {
        "top": _marker_color(top[1:-1], wall_color),
        "right": _marker_color(right[1:-1], wall_color),
        "bottom": _marker_color(bottom[1:-1], wall_color),
        "left": _marker_color(left[1:-1], wall_color),
    }
    assert len(set(side_colors.values())) == 4, "the four side markers must use different colors"
    return r0, c0, r1, c1, side_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each maroon point inside a marked rectangular region using the nearest side's marker color."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    border_mask = (g != BLACK) & (g != MAROON)
    for coords in _components(border_mask):
        r0, c0, r1, c1, side_colors = _region_info(g, coords)
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(g[r, c]) != MAROON:
                    continue
                choices = [
                    (r - r0, side_colors["top"]),
                    (c1 - c, side_colors["right"]),
                    (r1 - r, side_colors["bottom"]),
                    (c - c0, side_colors["left"]),
                ]
                out[r, c] = min(choices, key=lambda item: item[0])[1]
    return out


def _expanded_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _place_boxes(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            box_h = random.randint(*OUTER_HEIGHT_RANGE)
            box_w = random.randint(*OUTER_WIDTH_RANGE)
            if box_h >= height - 1 or box_w >= width - 1:
                continue
            r0 = random.randint(0, height - box_h)
            c0 = random.randint(0, width - box_w)
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            padded = _expanded_box(box, REGION_GAP, (height, width))
            if any(_boxes_overlap(padded, _expanded_box(prev, REGION_GAP, (height, width))) for prev in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _segment_bounds(side_len: int) -> tuple[int, int]:
    seg_len = random.randint(1, side_len - 2)
    start = random.randint(1, side_len - seg_len - 1)
    return start, seg_len


def _paint_region(grid: np.ndarray, box: tuple[int, int, int, int], side_colors: dict[str, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    inner_w = c1 - c0 - 1
    inner_h = r1 - r0 - 1

    start, seg_len = _segment_bounds(inner_w)
    grid[r0, c0 + start : c0 + start + seg_len] = side_colors["top"]
    start, seg_len = _segment_bounds(inner_w)
    grid[r1, c0 + start : c0 + start + seg_len] = side_colors["bottom"]
    start, seg_len = _segment_bounds(inner_h)
    grid[r0 + start : r0 + start + seg_len, c0] = side_colors["left"]
    start, seg_len = _segment_bounds(inner_h)
    grid[r0 + start : r0 + start + seg_len, c1] = side_colors["right"]


def _strict_points(inner_h: int, inner_w: int) -> dict[str, tuple[int, int]]:
    mid_r = inner_h // 2 + 1
    mid_c = inner_w // 2 + 1
    return {
        "top": (1, mid_c),
        "right": (mid_r, inner_w),
        "bottom": (inner_h, mid_c),
        "left": (mid_r, 1),
    }


def _tie_candidates(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    return [(1, 1), (1, inner_w), (inner_h, inner_w), (inner_h, 1)]


def _can_add(point: tuple[int, int], chosen: set[tuple[int, int]]) -> bool:
    return all(abs(point[0] - rr) + abs(point[1] - cc) > 1 for rr, cc in chosen)


def _choose_points(inner_h: int, inner_w: int) -> set[tuple[int, int]]:
    chosen: set[tuple[int, int]] = set(_strict_points(inner_h, inner_w).values())

    tie_options = [pt for pt in _tie_candidates(inner_h, inner_w) if pt not in chosen]
    tie_point = random.choice(tie_options)
    chosen.add(tie_point)

    extra_budget = min(EXTRA_POINT_MAX, inner_h * inner_w - len(chosen))
    extra_count = random.randint(0, extra_budget)
    candidates = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]
    random.shuffle(candidates)
    for point in candidates:
        if len(chosen) >= 5 + extra_count:
            break
        if point in chosen or not _can_add(point, chosen):
            continue
        chosen.add(point)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.choices(REGION_COUNT_CHOICES, weights=REGION_COUNT_WEIGHTS, k=1)[0]
    boxes = _place_boxes(height, width, count)
    if boxes is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for box in boxes:
        side_palette = random.sample(MARKER_COLORS, 4)
        side_colors = dict(zip(("top", "right", "bottom", "left"), side_palette))
        _paint_region(grid, box, side_colors)

        r0, c0, r1, c1 = box
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        points = _choose_points(inner_h, inner_w)
        for pr, pc in points:
            grid[r0 + pr, c0 + pc] = MAROON

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
