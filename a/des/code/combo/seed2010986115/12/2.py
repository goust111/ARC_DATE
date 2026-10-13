from __future__ import annotations

import random

import numpy as np


GREY = 5
BLACK = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN_BY_COUNT = {2: 19, 3: 23, 4: 25}
GRID_MAX_SIDE = 29
MAX_HALF_SIDE_BY_COUNT = {2: 5, 3: 5, 4: 4}
PLACEMENT_PADDING = 1
CENTER_PADDING = 2
GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    black_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == BLACK)}
    assert len(black_cells) == 5, "expected exactly one 5-pixel black cross"
    for r, c in black_cells:
        if all((r + dr, c + dc) in black_cells for dr, dc in NEIGHBORS4):
            expected = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
            assert black_cells == expected, "black pixels must form a single-pixel cross"
            return r, c
    raise AssertionError("failed to locate the cross center")


def _color_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (GREY, BLACK):
            continue
        mask = grid == color
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert not bool(np.any((patch != GREY) & (patch != color))), "each color group must occupy its own bbox"
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rects.append((h * w, h, w, color))
    areas = [area for area, _h, _w, _color in rects]
    assert len(set(areas)) == len(areas), "assembled rectangle areas must be unique"
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group colored fragments by color, complete each group's bounding rectangle,
    then stack the completed rectangles concentrically at the grid center.

    Input structure
    - The background is grey.
    - A unique black 5-pixel cross marks the stacking center.
    - Each non-grey, non-black color appears as several 4-connected fragments.
    - Filling that color's minimal bounding box reconstructs one solid rectangle.

    Rule
    - For each foreground color, recover the rectangle given by its full bbox.
    - Sort the recovered rectangles by area from large to small.
    - Clear the canvas to grey, center each rectangle on the cross center, and
      draw from largest to smallest so smaller rectangles overwrite larger ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    center_r, center_c = _find_cross_center(g)
    rects = sorted(_color_rectangles(g), reverse=True)

    out = np.full(g.shape, GREY, dtype=int)
    height, width = g.shape
    for _area, h, w, color in rects:
        r0 = center_r - h // 2
        c0 = center_c - w // 2
        assert 0 <= r0 and r0 + h <= height
        assert 0 <= c0 and c0 + w <= width
        out[r0 : r0 + h, c0 : c0 + w] = color
    return out


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == 1]


def _sample_rectangles(height: int, width: int, count: int) -> list[tuple[int, int, int]]:
    max_half_h = min(MAX_HALF_SIDE_BY_COUNT[count], height // 2 - 1)
    max_half_w = min(MAX_HALF_SIDE_BY_COUNT[count], width // 2 - 1)
    if min(max_half_h, max_half_w) < count:
        raise ValueError("grid too small for the requested number of nested rectangles")

    half_heights = sorted(random.sample(range(1, max_half_h + 1), count))
    half_widths = sorted(random.sample(range(1, max_half_w + 1), count))
    colors = random.sample(COLOR_POOL, count)

    rects = []
    for color, hh, hw in zip(colors, half_heights, half_widths):
        rects.append((2 * hh + 1, 2 * hw + 1, color))
    return rects


def _fragment_mask(height: int, width: int) -> np.ndarray:
    builders = []

    def vertical_gap() -> np.ndarray:
        gap_col = random.randint(1, width - 2)
        mask = np.ones((height, width), dtype=bool)
        mask[:, gap_col] = False
        return mask

    def horizontal_gap() -> np.ndarray:
        gap_row = random.randint(1, height - 2)
        mask = np.ones((height, width), dtype=bool)
        mask[gap_row, :] = False
        return mask

    def cross_gap() -> np.ndarray:
        gap_row = random.randint(1, height - 2)
        gap_col = random.randint(1, width - 2)
        mask = np.ones((height, width), dtype=bool)
        mask[gap_row, :] = False
        mask[:, gap_col] = False
        return mask

    def t_gap(split_top: bool) -> np.ndarray:
        gap_row = random.randint(1, height - 2)
        gap_col = random.randint(1, width - 2)
        mask = np.ones((height, width), dtype=bool)
        mask[gap_row, :] = False
        if split_top:
            mask[:gap_row, gap_col] = False
        else:
            mask[gap_row + 1 :, gap_col] = False
        return mask

    if width >= 3:
        builders.append(vertical_gap)
    if height >= 3:
        builders.append(horizontal_gap)
    if height >= 3 and width >= 3:
        builders.extend([cross_gap, lambda: t_gap(True), lambda: t_gap(False)])

    for _ in range(40):
        mask = random.choice(builders)()
        comps = _components(mask)
        if not (2 <= len(comps) <= 4):
            continue
        if _bbox_from_mask(mask) != (0, 0, height - 1, width - 1):
            continue
        return mask
    raise RuntimeError("failed to sample a fragmented rectangle mask")


def _reserve(blocked: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + height + padding)
    cc1 = min(w, c0 + width + padding)
    blocked[rr0:rr1, cc0:cc1] = True


def _can_place(blocked: np.ndarray, r0: int, c0: int, height: int, width: int, padding: int) -> bool:
    h, w = blocked.shape
    if r0 < 0 or c0 < 0 or r0 + height > h or c0 + width > w:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + height + padding)
    cc1 = min(w, c0 + width + padding)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _paint_patch(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _build_expected(shape: tuple[int, int], rects: list[tuple[int, int, int]]) -> np.ndarray:
    height, width = shape
    center_r = height // 2
    center_c = width // 2
    out = np.full(shape, GREY, dtype=int)
    for h, w, color in sorted(rects, key=lambda item: item[0] * item[1], reverse=True):
        r0 = center_r - h // 2
        c0 = center_c - w // 2
        out[r0 : r0 + h, c0 : c0 + w] = color
    return out


def _place_fragments(shape: tuple[int, int], rects: list[tuple[int, int, int]]) -> np.ndarray | None:
    height, width = shape
    grid = np.full(shape, GREY, dtype=int)
    center_r = height // 2
    center_c = width // 2
    grid[center_r, center_c] = BLACK
    grid[center_r - 1, center_c] = BLACK
    grid[center_r + 1, center_c] = BLACK
    grid[center_r, center_c - 1] = BLACK
    grid[center_r, center_c + 1] = BLACK

    blocked = np.zeros(shape, dtype=bool)
    _reserve(
        blocked,
        center_r - 1,
        center_c - 1,
        3,
        3,
        padding=CENTER_PADDING,
    )

    specs = sorted(rects, key=lambda item: item[0] * item[1], reverse=True)
    for h, w, color in specs:
        mask = _fragment_mask(h, w)
        candidates = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            if not _can_place(blocked, r0, c0, h, w, padding=PLACEMENT_PADDING):
                continue
            _paint_patch(grid, mask, color, r0, c0)
            _reserve(blocked, r0, c0, h, w, padding=PLACEMENT_PADDING)
            placed = True
            break
        if not placed:
            return None

    return grid


def _background_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = freq.get(GREY, 0)
    return all(bg_count > count for value, count in freq.items() if value != GREY)


def _valid_fragments(grid: np.ndarray, rects: list[tuple[int, int, int]]) -> bool:
    expected_by_color = {color: (h, w) for h, w, color in rects}
    for color, (expected_h, expected_w) in expected_by_color.items():
        mask = grid == color
        comps = _components(mask)
        if not (2 <= len(comps) <= 4):
            return False
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        if (r1 - r0 + 1, c1 - c0 + 1) != (expected_h, expected_w):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    min_side = GRID_MIN_BY_COUNT[count]
    height = random.choice(_odd_choices(min_side, GRID_MAX_SIDE))
    width = random.choice(_odd_choices(min_side, GRID_MAX_SIDE))

    try:
        rects = _sample_rectangles(height, width, count)
    except ValueError:
        return None

    inp = _place_fragments((height, width), rects)
    if inp is None:
        return None
    if not _background_is_dominant(inp):
        return None
    if not _valid_fragments(inp, rects):
        return None

    expected = _build_expected((height, width), rects)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
