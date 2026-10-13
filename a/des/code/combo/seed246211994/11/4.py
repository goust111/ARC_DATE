from __future__ import annotations

import random

import numpy as np


GEN_PAYLOAD_SIZE_RANGE = (4, 10)
GEN_OBJECT_COUNT_RANGE = (3, 7)
GEN_ZERO_ZERO_PROB = 0.18
GEN_MAX_ATTEMPTS = 400
GEN_SHAPE_TRIES = 80
GEN_PLACE_TRIES = 80


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).reshape(-1)
    colors, counts = np.unique(flat, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _unique_bg_index(line: np.ndarray, bg: int) -> int:
    idx = np.flatnonzero(np.asarray(line, dtype=int) == int(bg))
    assert len(idx) == 1, "control line must contain exactly one background marker"
    return int(idx[0])


def _shift_payload(payload: np.ndarray, down: int, right: int) -> np.ndarray:
    out = np.asarray(payload, dtype=int).copy()
    out = np.roll(out, int(down), axis=0)
    out = np.roll(out, int(right), axis=1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the control row/column, then cyclically shift the payload by the
    marker positions encoded in the top row and left column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = _mode_color(g)
    down = _unique_bg_index(g[:, 0], bg)
    right = _unique_bg_index(g[0], bg)

    payload = g[1:, 1:]
    return _shift_payload(payload, down, right)


def _build_rect(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _build_hollow_rect(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _build_l_shape(h: int, w: int, orientation: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if orientation == 0:
        mask[0, :] = True
        mask[:, 0] = True
    elif orientation == 1:
        mask[0, :] = True
        mask[:, -1] = True
    elif orientation == 2:
        mask[-1, :] = True
        mask[:, 0] = True
    else:
        mask[-1, :] = True
        mask[:, -1] = True
    return mask


def _build_plus(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mid = size // 2
    mask[mid, :] = True
    mask[:, mid] = True
    return mask


def _random_shape(side: int, max_cells: int, rng: random.Random) -> np.ndarray | None:
    kinds = ["rect", "line", "hollow", "lshape", "plus"]
    for _ in range(GEN_SHAPE_TRIES):
        kind = rng.choice(kinds)
        if kind == "rect":
            h = rng.randint(1, min(4, side))
            w = rng.randint(1, min(4, side))
            mask = _build_rect(h, w)
        elif kind == "line":
            length = rng.randint(2, min(5, side))
            if rng.random() < 0.5:
                mask = _build_rect(1, length)
            else:
                mask = _build_rect(length, 1)
        elif kind == "hollow":
            if side < 3:
                continue
            h = rng.randint(3, min(6, side))
            w = rng.randint(3, min(6, side))
            mask = _build_hollow_rect(h, w)
        elif kind == "lshape":
            if side < 2:
                continue
            h = rng.randint(2, min(5, side))
            w = rng.randint(2, min(5, side))
            mask = _build_l_shape(h, w, rng.randrange(4))
        else:
            sizes = [size for size in (3, 5) if size <= side]
            if not sizes:
                continue
            mask = _build_plus(rng.choice(sizes))

        area = int(mask.sum())
        if 0 < area <= int(max_cells):
            return mask
    return None


def _place_mask(payload: np.ndarray, mask: np.ndarray, color: int, bg: int, rng: random.Random) -> bool:
    side = int(payload.shape[0])
    h, w = map(int, mask.shape)
    for _ in range(GEN_PLACE_TRIES):
        r0 = rng.randint(0, side - h)
        c0 = rng.randint(0, side - w)
        region = payload[r0 : r0 + h, c0 : c0 + w]
        if np.any(region[mask] != int(bg)):
            continue
        region[mask] = int(color)
        return True
    return False


def _payload_active_axes(payload: np.ndarray, bg: int) -> tuple[int, int]:
    mask = np.asarray(payload, dtype=int) != int(bg)
    return int(mask.any(axis=1).sum()), int(mask.any(axis=0).sum())


def _build_payload(side: int, bg: int, colors: list[int], rng: random.Random, max_non_bg: int) -> np.ndarray | None:
    payload = np.full((side, side), int(bg), dtype=int)
    target_objects = rng.randint(*GEN_OBJECT_COUNT_RANGE)
    target_objects = min(target_objects, int(max_non_bg))

    painted = 0
    used_colors: set[int] = set()
    for _ in range(target_objects):
        remaining = int(max_non_bg - painted)
        if remaining <= 0:
            break
        mask = _random_shape(side, remaining, rng)
        if mask is None:
            continue
        color = rng.choice(colors)
        if _place_mask(payload, mask, color, bg, rng):
            painted += int(mask.sum())
            used_colors.add(int(color))

    extra_points = rng.randint(1, min(3, max_non_bg - painted) if max_non_bg - painted > 0 else 1)
    for _ in range(extra_points):
        free = np.argwhere(payload == int(bg))
        if free.size == 0:
            break
        r, c = map(int, free[rng.randrange(len(free))])
        color = rng.choice(colors)
        payload[r, c] = int(color)
        painted += 1
        used_colors.add(int(color))
        if painted >= int(max_non_bg):
            break

    if painted < max(4, side):
        return None
    if len(used_colors) < 2:
        return None
    active_rows, active_cols = _payload_active_axes(payload, bg)
    if active_rows < 2 or active_cols < 2:
        return None
    return payload


def _unique_mode_is_bg(grid: np.ndarray, bg: int) -> bool:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    order = np.argsort(counts)[::-1]
    best_color = int(colors[int(order[0])])
    best_count = int(counts[int(order[0])])
    second_count = int(counts[int(order[1])]) if len(order) > 1 else -1
    return best_color == int(bg) and best_count > second_count


def _pick_markers(side: int, rng: random.Random) -> tuple[int, int]:
    if rng.random() < GEN_ZERO_ZERO_PROB:
        return 0, 0
    return rng.randint(1, side), rng.randint(1, side)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.randint(*GEN_PAYLOAD_SIZE_RANGE)
    bg = rng.randrange(10)
    down, right = _pick_markers(side, rng)
    shared_top_left_bg = down == 0 and right == 0

    palette = [color for color in range(10) if color != bg]
    if shared_top_left_bg and rng.random() < 0.4:
        row_color, col_color = rng.sample(palette, 2)
    else:
        row_color = rng.choice(palette)
        col_color = row_color

    object_colors = [color for color in range(10) if color not in {bg, row_color, col_color}]
    color_count = rng.randint(2, min(4, len(object_colors)))
    object_colors = rng.sample(object_colors, color_count)

    unique_markers = 1 if shared_top_left_bg else 2
    control_non_bg = 2 * side if shared_top_left_bg else 2 * side - 1
    min_payload_bg = control_non_bg - unique_markers + 1
    max_non_bg = side * side - min_payload_bg
    if max_non_bg < max(4, side):
        return None

    payload = _build_payload(side, bg, object_colors, rng, max_non_bg)
    if payload is None:
        return None

    expected = _shift_payload(payload, down, right)
    if (down % side != 0 or right % side != 0) and np.array_equal(expected, payload):
        return None

    grid = np.full((side + 1, side + 1), int(bg), dtype=int)
    grid[1:, 1:] = payload
    grid[0, 1:] = int(row_color)
    grid[1:, 0] = int(col_color)
    if not shared_top_left_bg:
        grid[0, 0] = int(row_color)
    grid[down, 0] = int(bg)
    grid[0, right] = int(bg)

    if int((grid[0] == int(bg)).sum()) != 1:
        return None
    if int((grid[:, 0] == int(bg)).sum()) != 1:
        return None
    if not _unique_mode_is_bg(grid, bg):
        return None

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
