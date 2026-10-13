import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rect_from_component(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert coords.shape[0] == h * w, "component must be a solid rectangle"
    return r0, c0, r1, c1


def _square_for_pair(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> tuple[int, int, int]:
    a = tuple(map(int, rect_a))
    b = tuple(map(int, rect_b))
    ah = a[0], a[2]
    aw = a[1], a[3]
    bh = b[0], b[2]
    bw = b[1], b[3]
    assert (a[2] - a[0], a[3] - a[1]) == (b[2] - b[0], b[3] - b[1]), "rectangles must match in size"

    if ah == bh:
        left, right = (a, b) if a[1] < b[1] else (b, a)
        gap = right[1] - left[3] - 1
        assert gap > 0, "horizontal pair must have a positive gap"
        top_twice = left[0] + left[2] - gap + 1
        assert top_twice % 2 == 0, "square must be vertically centerable"
        top = top_twice // 2
        col = left[3] + 1
        return top, col, gap

    if aw == bw:
        top_rect, bottom_rect = (a, b) if a[0] < b[0] else (b, a)
        gap = bottom_rect[0] - top_rect[2] - 1
        assert gap > 0, "vertical pair must have a positive gap"
        left_twice = top_rect[1] + top_rect[3] - gap + 1
        assert left_twice % 2 == 0, "square must be horizontally centerable"
        row = top_rect[2] + 1
        left = left_twice // 2
        return row, left, gap

    raise AssertionError("pair must be aligned horizontally or vertically")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    gaps = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each foreground color must form exactly two rectangles"
        rects = [_rect_from_component(comp) for comp in comps]
        top, left, size = _square_for_pair(rects[0], rects[1])
        assert 0 <= top and 0 <= left and top + size <= g.shape[0] and left + size <= g.shape[1]
        gaps.append(size)
        out[top : top + size, left : left + size] = color

    assert len(gaps) == len(set(gaps)), "pair gaps must be distinct"
    return out


def _sample_even_or_odd(min_value: int, max_value: int, parity: int) -> int | None:
    choices = [v for v in range(min_value, max_value + 1) if v % 2 == parity]
    if not choices:
        return None
    return random.choice(choices)


def _can_place(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = reserved.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return not reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _fill_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _place_horizontal(grid: np.ndarray, reserved: np.ndarray, color: int, gap: int) -> bool:
    h_total, w_total = grid.shape
    parity = gap % 2
    rect_h = _sample_even_or_odd(gap, min(8, h_total - 2), parity)
    if rect_h is None:
        return False
    rect_w = random.randint(2, min(6, max(2, (w_total - gap - 4) // 2)))
    span_w = rect_w * 2 + gap
    if span_w > w_total - 2:
        return False
    for _ in range(120):
        r0 = random.randint(1, h_total - rect_h - 1)
        c0 = random.randint(1, w_total - span_w - 1)
        if not _can_place(reserved, r0, c0, r0 + rect_h - 1, c0 + span_w - 1):
            continue
        _fill_rect(grid, r0, c0, rect_h, rect_w, color)
        _fill_rect(grid, r0, c0 + rect_w + gap, rect_h, rect_w, color)
        reserved[r0 : r0 + rect_h, c0 : c0 + span_w] = True
        return True
    return False


def _place_vertical(grid: np.ndarray, reserved: np.ndarray, color: int, gap: int) -> bool:
    h_total, w_total = grid.shape
    parity = gap % 2
    rect_w = _sample_even_or_odd(gap, min(8, w_total - 2), parity)
    if rect_w is None:
        return False
    rect_h = random.randint(2, min(6, max(2, (h_total - gap - 4) // 2)))
    span_h = rect_h * 2 + gap
    if span_h > h_total - 2:
        return False
    for _ in range(120):
        r0 = random.randint(1, h_total - span_h - 1)
        c0 = random.randint(1, w_total - rect_w - 1)
        if not _can_place(reserved, r0, c0, r0 + span_h - 1, c0 + rect_w - 1):
            continue
        _fill_rect(grid, r0, c0, rect_h, rect_w, color)
        _fill_rect(grid, r0 + rect_h + gap, c0, rect_h, rect_w, color)
        reserved[r0 : r0 + span_h, c0 : c0 + rect_w] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    num_pairs = random.randint(2, 4)
    max_gap = min(6, min(h, w) // 3)
    gap_choices = list(range(2, max_gap + 1))
    if len(gap_choices) < num_pairs:
        return None
    gaps = random.sample(gap_choices, num_pairs)
    colors = random.sample(PALETTE, num_pairs)

    orientations = [random.randint(0, 1) for _ in range(num_pairs)]
    if num_pairs >= 2 and len(set(orientations)) == 1 and random.random() < 0.6:
        orientations[0] = 1 - orientations[0]

    for color, gap, orient in zip(colors, gaps, orientations):
        ok = _place_horizontal(grid, reserved, color, gap) if orient == 0 else _place_vertical(grid, reserved, color, gap)
        if not ok:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    counts = Counter(map(int, grid.flat))
    if counts[BG] <= max(count for color, count in counts.items() if color != BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
