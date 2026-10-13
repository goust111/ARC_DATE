from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _has_no_holes(mask: np.ndarray) -> bool:
    bg = np.pad(~mask, 1, constant_values=True)
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(bg[nr, nc]) and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return not bool(np.any(bg[1:-1, 1:-1] & ~seen[1:-1, 1:-1]))


def _uses_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_color_bar(g: np.ndarray) -> tuple[list[int], list[tuple[int, int]]]:
    all_colors = {int(v) for v in g.reshape(-1) if int(v) != BG}
    assert len(all_colors) >= 2, "expected multiple object colors"

    best: tuple[tuple[int, int, int], list[int], list[tuple[int, int]]] | None = None
    h, w = g.shape
    for c in range(w):
        r = 0
        while r < h:
            if int(g[r, c]) == BG:
                r += 1
                continue
            start = r
            vals: list[int] = []
            cells: list[tuple[int, int]] = []
            while r < h and int(g[r, c]) != BG:
                vals.append(int(g[r, c]))
                cells.append((int(r), int(c)))
                r += 1
            end = r - 1
            if len(vals) == len(all_colors) and len(set(vals)) == len(vals) and set(vals) == all_colors:
                score = (int(end), int(c), -int(start))
                if best is None or score > best[0]:
                    best = (score, vals, cells)

    assert best is not None, "right-side color bar was not found"
    return best[1], best[2]


def _relative_shape(coords: np.ndarray) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return rel, (int(r1 - r0 + 1), int(c1 - c0 + 1))


def _mask_from_rel(rel: frozenset[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in rel:
        out[int(r), int(c)] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack identical colored objects at the base object's position using the color bar order.

    Input structure:
    - The background is black.
    - Several separated monochrome objects have distinct colors and the same normalized no-hole shape.
    - Near the lower-right corner, a one-column color bar contains one pixel of every object color from top to bottom.

    Rule:
    - Read the lower-right vertical color bar as the stacking order, from bottom layer at the top of the bar to top layer at the bottom.
    - Remove the color bar, group the remaining pixels by color, and verify that all grouped objects have the same translated shape.
    - Use the first color in the bar as the base position and crop to that object's bounding box.
    - Translate each object shape to the base crop and paint them in bar order, so later colors cover earlier colors.

    Key insight:
    - The objects only become comparable after translation normalization, and the small color bar is the visible key that selects both the base object and the occlusion order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    order, bar_cells = _find_color_bar(g)
    work = g.copy()
    for r, c in bar_cells:
        work[int(r), int(c)] = BG

    assert {int(v) for v in work.reshape(-1) if int(v) != BG} == set(order), (
        "all object colors must be listed by the color bar"
    )

    shapes: dict[int, tuple[frozenset[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]] = {}
    for color in order:
        coords = np.argwhere(work == int(color))
        assert coords.size > 0, "every bar color must have an object"
        bbox = _bbox(coords)
        rel, shape = _relative_shape(coords)
        mask = _mask_from_rel(rel, shape)
        assert _is_connected(mask), "each object must be connected"
        shapes[int(color)] = (rel, shape, bbox)

    base_rel, base_shape, _base_bbox = shapes[int(order[0])]
    for color in order[1:]:
        rel, shape, _bbox_value = shapes[int(color)]
        assert shape == base_shape and rel == base_rel, "objects must match after translation"

    out = np.full(base_shape, BG, dtype=int)
    for color in order:
        rel, _shape, _bbox_value = shapes[int(color)]
        for r, c in rel:
            out[int(r), int(c)] = int(color)
    return out


# --------------- generator ---------------


def _random_shape() -> np.ndarray:
    for _ in range(120):
        h = random.randint(3, 8)
        w = random.randint(3, 8)
        mask = np.ones((h, w), dtype=bool)
        min_area = max(5, (h * w) // 2)
        target = random.randint(min_area, h * w - 1)

        for _step in range(h * w * 3):
            if int(mask.sum()) <= target:
                break
            cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
            random.shuffle(cells)
            removed = False
            for r, c in cells:
                trial = mask.copy()
                trial[r, c] = False
                if (
                    int(trial.sum()) >= min_area
                    and _uses_full_bbox(trial)
                    and _is_connected(trial)
                    and _has_no_holes(trial)
                ):
                    mask = trial
                    removed = True
                    break
            if not removed:
                break

        if (
            int(mask.sum()) < h * w
            and _uses_full_bbox(mask)
            and _is_connected(mask)
            and _has_no_holes(mask)
        ):
            return _random_transform(mask)

    fallback = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    )
    return _random_transform(fallback)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.choice([False, True]):
        out = np.fliplr(out)
    if random.choice([False, True]):
        out = np.flipud(out)
    return np.array(out, dtype=bool)


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(reserved.shape[0], int(r0) + int(h) + int(pad))
    cc1 = min(reserved.shape[1], int(c0) + int(w) + int(pad))
    reserved[rr0:rr1, cc0:cc1] = True


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > reserved.shape[0] or c0 + w > reserved.shape[1]:
        return False
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(reserved.shape[0], int(r0) + int(h) + 1)
    cc1 = min(reserved.shape[1], int(c0) + int(w) + 1)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _paint_shape(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    view = grid[int(r0) : int(r0) + mask.shape[0], int(c0) : int(c0) + mask.shape[1]]
    view[mask] = int(color)


def _expected_output(mask: np.ndarray, order: list[int]) -> np.ndarray:
    out = np.full(mask.shape, BG, dtype=int)
    out[mask] = int(order[-1])
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    colors = random.sample(COLORS, count)
    order = random.sample(colors, count)
    mask = _random_shape()
    mh, mw = map(int, mask.shape)

    min_h = max(12, mh + 5, count + 4)
    min_w = max(12, mw + 5)
    if min_h > 30 or min_w > 30:
        return None
    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)

    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    bar_col = width - 1 - random.randint(0, min(1, width - 1))
    bar_bottom = height - 1 - random.randint(0, min(1, height - count))
    bar_top = bar_bottom - count + 1
    if bar_top < 0:
        return None
    for i, color in enumerate(order):
        grid[bar_top + i, bar_col] = int(color)
    _reserve(reserved, bar_top, bar_col, count, 1, pad=1)

    placements: dict[int, tuple[int, int]] = {}
    for color in random.sample(colors, count):
        placed = False
        for _ in range(250):
            row = random.randint(1, height - mh - 1)
            col = random.randint(1, width - mw - 1)
            if not _can_place(reserved, row, col, mh, mw):
                continue
            _paint_shape(grid, mask, row, col, int(color))
            _reserve(reserved, row, col, mh, mw, pad=1)
            placements[int(color)] = (int(row), int(col))
            placed = True
            break
        if not placed:
            return None

    assert set(placements) == set(colors)
    expected = _expected_output(mask, order)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacking input")
