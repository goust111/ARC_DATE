from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (10, 22)
GRID_W_RANGE = (10, 22)

MODE_CHOICES = ("disjoint", "nested", "cross", "mixed")

GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 200
MIN_BG_FRACTION = 0.12
MIN_COLORED_FRACTION = 0.10

Rect = tuple[int, int, int, int]


def _rect_h(rect: Rect) -> int:
    return int(rect[2] - rect[0] + 1)


def _rect_w(rect: Rect) -> int:
    return int(rect[3] - rect[1] + 1)


def _rect_area(rect: Rect) -> int:
    return int(_rect_h(rect) * _rect_w(rect))


def _rect_overlap_area(a: Rect, b: Rect) -> int:
    h = min(int(a[2]), int(b[2])) - max(int(a[0]), int(b[0])) + 1
    w = min(int(a[3]), int(b[3])) - max(int(a[1]), int(b[1])) + 1
    return int(max(0, h) * max(0, w))


def _rect_intersection(a: Rect, b: Rect) -> Rect | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _touch_or_overlap(a: Rect, b: Rect, pad: int = 0) -> bool:
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _diag_endpoints(rect: Rect, diag: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = map(int, rect)
    if int(diag) == 0:
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _random_small_dims(h: int, w: int) -> tuple[int, int]:
    max_h = min(7, int(h))
    max_w = min(7, int(w))
    if max_h < 1 or max_w < 1:
        raise ValueError("grid too small for rectangle placement")

    if random.random() < 0.30:
        if random.random() < 0.5:
            return 1, random.randint(2, max(2, max_w))
        return random.randint(2, max(2, max_h)), 1

    lo_h = 2 if max_h >= 2 else 1
    lo_w = 2 if max_w >= 2 else 1
    hh = random.randint(lo_h, max_h)
    ww = random.randint(lo_w, max_w)
    if hh == 1 and ww == 1:
        return 1, 2
    return int(hh), int(ww)


def _place_disjoint_rect(existing: list[Rect], h: int, w: int) -> Rect | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        rh, rw = _random_small_dims(h, w)
        if rh > h or rw > w:
            continue
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        rect = (int(r0), int(c0), int(r0 + rh - 1), int(c0 + rw - 1))
        if any(_touch_or_overlap(rect, other, pad=1) for other in existing):
            continue
        return rect
    return None


def _build_disjoint(h: int, w: int) -> list[Rect] | None:
    max_count = min(6, 3 + (h * w >= 160) + (h * w >= 240) + (h * w >= 320))
    count = random.randint(3, max_count)
    rects: list[Rect] = []
    for _ in range(count):
        rect = _place_disjoint_rect(rects, h, w)
        if rect is None:
            return None
        rects.append(rect)
    return rects


def _build_nested(h: int, w: int) -> list[Rect] | None:
    if h < 6 or w < 6:
        return None

    outer_h_lo = max(4, h // 3)
    outer_w_lo = max(4, w // 3)
    outer_h = random.randint(outer_h_lo, max(outer_h_lo, h - 2))
    outer_w = random.randint(outer_w_lo, max(outer_w_lo, w - 2))
    r0 = random.randint(0, h - outer_h)
    c0 = random.randint(0, w - outer_w)
    outer = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))

    rects = [outer]
    current = outer
    chain_len = random.randint(2, 3)

    for _ in range(chain_len - 1):
        placed = None
        for _attempt in range(PLACEMENT_ATTEMPTS):
            cur_h = _rect_h(current)
            cur_w = _rect_w(current)
            new_h = random.randint(1, cur_h)
            new_w = random.randint(1, cur_w)
            if new_h == cur_h and new_w == cur_w:
                continue
            if new_h == 1 and new_w == 1:
                continue
            nr0 = random.randint(int(current[0]), int(current[2]) - new_h + 1)
            nc0 = random.randint(int(current[1]), int(current[3]) - new_w + 1)
            placed = (int(nr0), int(nc0), int(nr0 + new_h - 1), int(nc0 + new_w - 1))
            break
        if placed is None:
            return None
        rects.append(placed)
        current = placed

    extra_count = random.randint(0, 2 if h * w >= 180 else 1)
    for _ in range(extra_count):
        rect = _place_disjoint_rect(rects, h, w)
        if rect is not None:
            rects.append(rect)
    return rects


def _build_cross_core(h: int, w: int) -> list[Rect] | None:
    if h < 8 or w < 8:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        h1_hi = min(3, h - 4)
        w1_hi = min(10, w - 2)
        h2_hi = min(10, h - 2)
        if h1_hi < 1 or w1_hi < 4 or h2_hi < 4:
            return None

        h1 = random.randint(1, h1_hi)
        w1 = random.randint(4, w1_hi)
        w2 = random.randint(1, min(3, w1 - 2))
        h2 = random.randint(4, h2_hi)
        if h1 * w1 == h2 * w2:
            continue

        r0h = random.randint(1, h - h1 - 1)
        c0h = random.randint(1, w - w1 - 1)

        c0v_lo = c0h + 1
        c0v_hi = c0h + w1 - w2 - 1
        if c0v_lo > c0v_hi:
            continue
        c0v = random.randint(c0v_lo, c0v_hi)

        r0v_lo = max(0, r0h + h1 - h2 + 1)
        r0v_hi = min(r0h - 1, h - h2)
        if r0v_lo > r0v_hi:
            continue
        r0v = random.randint(r0v_lo, r0v_hi)

        horiz = (int(r0h), int(c0h), int(r0h + h1 - 1), int(c0h + w1 - 1))
        vert = (int(r0v), int(c0v), int(r0v + h2 - 1), int(c0v + w2 - 1))
        if _rect_overlap_area(horiz, vert) > 0:
            return [horiz, vert]

    return None


def _build_cross(h: int, w: int) -> list[Rect] | None:
    rects = _build_cross_core(h, w)
    if rects is None:
        return None

    extra_count = random.randint(0, 2 if h * w >= 180 else 1)
    for _ in range(extra_count):
        rect = _place_disjoint_rect(rects, h, w)
        if rect is not None:
            rects.append(rect)
    return rects


def _build_mixed(h: int, w: int) -> list[Rect] | None:
    if h < 10 or w < 10:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        side = random.choice(("right", "bottom"))

        outer_h_lo = max(4, h // 3)
        outer_w_lo = max(4, w // 3)
        outer_h = random.randint(outer_h_lo, max(outer_h_lo, h - 2))
        outer_w = random.randint(outer_w_lo, max(outer_w_lo, w - 2))

        if side == "right":
            if outer_w >= w:
                continue
            r0 = random.randint(0, h - outer_h)
            c0 = random.randint(0, w - outer_w - 1)
            outer = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))

            max_extra = min(4, w - int(outer[3]) - 1)
            if max_extra < 1 or outer_w < 3:
                continue
            overlap_w = random.randint(2, outer_w - 1)
            extra_w = random.randint(1, max_extra)
            arm_h = random.randint(2, outer_h)
            arm_r0 = random.randint(int(outer[0]), int(outer[2]) - arm_h + 1)
            arm_c0 = int(outer[3]) - overlap_w + 1
            arm = (
                int(arm_r0),
                int(arm_c0),
                int(arm_r0 + arm_h - 1),
                int(outer[3] + extra_w),
            )
        else:
            if outer_h >= h:
                continue
            r0 = random.randint(0, h - outer_h - 1)
            c0 = random.randint(0, w - outer_w)
            outer = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))

            max_extra = min(4, h - int(outer[2]) - 1)
            if max_extra < 1 or outer_h < 3:
                continue
            overlap_h = random.randint(2, outer_h - 1)
            extra_h = random.randint(1, max_extra)
            arm_w = random.randint(2, outer_w)
            arm_c0 = random.randint(int(outer[1]), int(outer[3]) - arm_w + 1)
            arm_r0 = int(outer[2]) - overlap_h + 1
            arm = (
                int(arm_r0),
                int(arm_c0),
                int(outer[2] + extra_h),
                int(arm_c0 + arm_w - 1),
            )

        if _rect_area(outer) == _rect_area(arm):
            continue

        inter = _rect_intersection(outer, arm)
        if inter is None:
            continue
        inter_h = _rect_h(inter)
        inter_w = _rect_w(inter)
        if inter_h == 1 and inter_w == 1:
            continue

        inner = None
        for _attempt in range(PLACEMENT_ATTEMPTS):
            inner_h = random.randint(1, inter_h)
            inner_w = random.randint(1, inter_w)
            if inner_h == 1 and inner_w == 1:
                continue
            ir0 = random.randint(int(inter[0]), int(inter[2]) - inner_h + 1)
            ic0 = random.randint(int(inter[1]), int(inter[3]) - inner_w + 1)
            inner = (int(ir0), int(ic0), int(ir0 + inner_h - 1), int(ic0 + inner_w - 1))
            break
        if inner is None:
            continue

        rects = [outer, arm, inner]
        extra_count = random.randint(0, 1)
        for _ in range(extra_count):
            rect = _place_disjoint_rect(rects, h, w)
            if rect is not None:
                rects.append(rect)
        return rects

    return None


def _assign_diagonals(rects: list[Rect]) -> list[int] | None:
    order = list(range(len(rects)))
    random.shuffle(order)
    diags: list[int | None] = [None] * len(rects)
    used: set[tuple[int, int]] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        options = [0, 1]
        random.shuffle(options)
        for diag in options:
            p0, p1 = _diag_endpoints(rects[idx], diag)
            if p0 == p1 or p0 in used or p1 in used:
                continue
            used.add(p0)
            used.add(p1)
            diags[idx] = int(diag)
            if dfs(pos + 1):
                return True
            used.remove(p0)
            used.remove(p1)
            diags[idx] = None
        return False

    if not dfs(0):
        return None
    return [int(diag) for diag in diags]


def _render_input(h: int, w: int, rects: list[Rect], diags: list[int], colors: list[int]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for rect, diag, color in zip(rects, diags, colors, strict=False):
        p0, p1 = _diag_endpoints(rect, diag)
        grid[p0] = int(color)
        grid[p1] = int(color)
    return grid


def _candidate_grid(mode: str) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    if mode == "disjoint":
        rects = _build_disjoint(h, w)
    elif mode == "nested":
        rects = _build_nested(h, w)
    elif mode == "cross":
        rects = _build_cross(h, w)
    else:
        rects = _build_mixed(h, w)

    if rects is None:
        return None
    if len(rects) < 2 or len(rects) > len(PALETTE):
        return None
    if max(_rect_area(rect) for rect in rects) < 6:
        return None

    overlap_pairs = 0
    for i, rect_a in enumerate(rects):
        if _rect_area(rect_a) < 2:
            return None
        for rect_b in rects[i + 1 :]:
            overlap = _rect_overlap_area(rect_a, rect_b)
            if overlap == 0:
                continue
            overlap_pairs += 1
            if _rect_area(rect_a) == _rect_area(rect_b):
                return None

    if mode == "disjoint" and overlap_pairs != 0:
        return None
    if mode != "disjoint" and overlap_pairs == 0:
        return None

    diags = _assign_diagonals(rects)
    if diags is None:
        return None

    colors = random.sample(list(PALETTE), len(rects))
    grid = _render_input(h, w, rects, diags, colors)

    if any(int((grid == color).sum()) != 2 for color in colors):
        return None

    out = main(grid)
    total = int(h * w)
    colored = int((out != BG).sum())
    bg_cells = total - colored
    if colored < max(8, int(total * MIN_COLORED_FRACTION)):
        return None
    if bg_cells < max(8, int(total * MIN_BG_FRACTION)):
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each same-color pixel pair into its axis-aligned rectangle.

    When rectangles overlap, smaller-area rectangles are on top, so the
    output is painted from larger area to smaller area.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    rects: list[tuple[int, int, int, int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        pts = np.argwhere(g == color)
        assert int(len(pts)) == 2, "each foreground color must appear exactly twice"
        (r0, c0), (r1, c1) = pts.tolist()
        rr0, rr1 = sorted((int(r0), int(r1)))
        cc0, cc1 = sorted((int(c0), int(c1)))
        area = int((rr1 - rr0 + 1) * (cc1 - cc0 + 1))
        rects.append((area, rr0, cc0, rr1, cc1, color))

    rects.sort(key=lambda item: (-int(item[0]), int(item[1]), int(item[2]), int(item[5])))
    for _area, r0, c0, r1, c1, color in rects:
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(color)

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        mode = random.choice(MODE_CHOICES)
        grid = _candidate_grid(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid grid after {GENERATION_ATTEMPTS} attempts")
