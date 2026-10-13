from __future__ import annotations

import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_marker(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if coords.shape != (2, 2):
            continue

        same_row = int(coords[0, 0]) == int(coords[1, 0])
        same_col = int(coords[0, 1]) == int(coords[1, 1])
        if same_row == same_col:
            continue

        if same_row:
            if w % 2 != 0:
                continue
            half = w // 2
            cols = coords[:, 1]
            in_left = bool(np.all(cols < half))
            in_right = bool(np.all(cols >= half))
            if in_left or in_right:
                candidates.append((color, 1, 0 if in_left else 1))
        else:
            if h % 2 != 0:
                continue
            half = h // 2
            rows = coords[:, 0]
            in_top = bool(np.all(rows < half))
            in_bottom = bool(np.all(rows >= half))
            if in_top or in_bottom:
                candidates.append((color, 0, 0 if in_top else 1))

    assert len(candidates) == 1, "expected a unique aligned two-pixel marker color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker, axis, side = _find_marker(g, bg)
    h, w = g.shape

    if axis == 1:
        assert w % 2 == 0, "row-aligned markers require even width"
        half = w // 2
        out = g[:, side * half : (side + 1) * half].copy()
    else:
        assert h % 2 == 0, "column-aligned markers require even height"
        half = h // 2
        out = g[side * half : (side + 1) * half, :].copy()

    out[out == marker] = bg
    return out


PALETTE = tuple(range(10))
MAX_ATTEMPTS = 400
MIN_OBJECTS = 1
MAX_OBJECTS = 4


def _paint_rect(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top : top + height, left : left + width] = color


def _paint_plus(canvas: np.ndarray, top: int, left: int, arm: int, color: int) -> None:
    center_r = top + arm
    center_c = left + arm
    canvas[top : top + 2 * arm + 1, center_c] = color
    canvas[center_r, left : left + 2 * arm + 1] = color


def _paint_l(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top : top + height, left] = color
    canvas[top + height - 1, left : left + width] = color


def _add_object(half: np.ndarray, occupied: np.ndarray, color: int) -> bool:
    h, w = half.shape
    kind = random.choice(("rect", "plus", "l"))

    for _ in range(60):
        mask = np.zeros_like(half, dtype=bool)
        if kind == "rect":
            obj_h = random.randint(1, min(4, h))
            obj_w = random.randint(1, min(4, w))
            if obj_h * obj_w < 3:
                continue
            top = random.randint(0, h - obj_h)
            left = random.randint(0, w - obj_w)
            mask[top : top + obj_h, left : left + obj_w] = True
        elif kind == "plus":
            if min(h, w) < 3:
                continue
            arm = random.randint(1, min(2, (min(h, w) - 1) // 2))
            size = 2 * arm + 1
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
            center_r = top + arm
            center_c = left + arm
            mask[top : top + size, center_c] = True
            mask[center_r, left : left + size] = True
        else:
            obj_h = random.randint(2, min(4, h))
            obj_w = random.randint(2, min(4, w))
            top = random.randint(0, h - obj_h)
            left = random.randint(0, w - obj_w)
            mask[top : top + obj_h, left] = True
            mask[top + obj_h - 1, left : left + obj_w] = True

        if np.count_nonzero(mask) < 3:
            continue
        if np.any(occupied & mask):
            continue

        occupied |= mask
        half[mask] = color
        return True

    return False


def _build_half(shape: tuple[int, int], bg: int, colors: list[int]) -> np.ndarray | None:
    half = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    object_count = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    random.shuffle(colors)
    used = 0

    for color in colors:
        if used >= object_count:
            break
        if _add_object(half, occupied, color):
            used += 1

    if used == 0:
        return None
    return half


def _merge_halves(axis: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if axis == 1:
        return np.concatenate([first, second], axis=1)
    return np.concatenate([first, second], axis=0)


def _marker_positions(grid: np.ndarray, axis: int, side: int, bg: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if axis == 1:
        half_w = w // 2
        c0 = 0 if side == 0 else half_w
        c1 = half_w - 1 if side == 0 else w - 1
        for r in range(h):
            bg_cols = [c for c in range(c0, c1 + 1) if grid[r, c] == bg]
            for i in range(len(bg_cols)):
                for j in range(i + 1, len(bg_cols)):
                    pairs.append(((r, bg_cols[i]), (r, bg_cols[j])))
    else:
        half_h = h // 2
        r0 = 0 if side == 0 else half_h
        r1 = half_h - 1 if side == 0 else h - 1
        for c in range(w):
            bg_rows = [r for r in range(r0, r1 + 1) if grid[r, c] == bg]
            for i in range(len(bg_rows)):
                for j in range(i + 1, len(bg_rows)):
                    pairs.append(((bg_rows[i], c), (bg_rows[j], c)))

    random.shuffle(pairs)
    return pairs


def _gen_attempt() -> np.ndarray | None:
    axis = random.randint(0, 1)
    side = random.randint(0, 1)

    bg = random.choice(PALETTE)
    remaining = [c for c in PALETTE if c != bg]
    marker = random.choice(remaining)
    content_colors = [c for c in remaining if c != marker]
    random.shuffle(content_colors)

    if axis == 1:
        h = random.randint(5, 12)
        half_w = random.randint(4, 10)
        half_shape = (h, half_w)
    else:
        half_h = random.randint(4, 10)
        w = random.randint(5, 12)
        half_shape = (half_h, w)

    kept = _build_half(half_shape, bg, content_colors[:])
    other = _build_half(half_shape, bg, content_colors[:])
    if kept is None or other is None:
        return None

    full = _merge_halves(axis, kept if side == 0 else other, other if side == 0 else kept)
    expected = kept.copy()

    pairs = _marker_positions(full, axis, side, bg)
    if not pairs:
        return None

    for a, b in pairs:
        candidate = full.copy()
        candidate[a] = marker
        candidate[b] = marker
        try:
            out = main(candidate)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return candidate

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
