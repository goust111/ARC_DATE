from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
NON_FIXED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_GENERATION_ATTEMPTS = 400
STRIP_COUNT_RANGE = (3, 6)
STRIP_WIDTH_RANGE = (2, 4)
IMAGE_HEIGHT_RANGE = (3, 8)
MARGIN_RANGE = (1, 3)


def _constant_step(cols: np.ndarray) -> int | None:
    if cols.size < 2:
        return None
    diffs = np.diff(cols)
    if np.all(diffs == diffs[0]):
        return int(diffs[0])
    return None


def _locate_layout(grid: np.ndarray) -> tuple[int, int, int, int, list[int]]:
    h, w = map(int, grid.shape)
    layouts: list[tuple[int, int, int, int, list[int]]] = []

    for tick_row in range(h):
        row = grid[tick_row]
        tick_cols = np.flatnonzero(row == GREY)
        strip_w = _constant_step(tick_cols)
        if strip_w is None:
            continue
        if np.any((row != BLACK) & (row != GREY)):
            continue

        strip_count = int(tick_cols.size)
        left = int(tick_cols[0])
        width = strip_count * strip_w
        if left + width > w:
            continue

        for order_row in range(tick_row + 3, h):
            order_cols = np.flatnonzero(grid[order_row] != BLACK)
            if order_cols.size != strip_count:
                continue
            order_colors = [int(grid[order_row, c]) for c in order_cols]
            if GREY in order_colors or len(set(order_colors)) != strip_count:
                continue
            if int(order_cols[0]) < left or int(order_cols[-1]) >= left + width:
                continue

            image = grid[tick_row + 1 : order_row, left : left + width]
            if image.shape[0] < 2:
                continue
            if np.any((image == BLACK) | (image == GREY)):
                continue
            if np.any(grid[tick_row + 1 : order_row, :left] != BLACK):
                continue
            if np.any(grid[tick_row + 1 : order_row, left + width :] != BLACK):
                continue

            layouts.append((tick_row, order_row, left, strip_w, order_colors))

    assert len(layouts) == 1, "expected exactly one valid shredded-image layout"
    return layouts[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a vertically shredded image by using grey cut ticks and bottom color order labels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tick_row, order_row, left, strip_w, order_colors = _locate_layout(g)
    strip_count = len(order_colors)
    width = strip_count * strip_w
    image = g[tick_row + 1 : order_row, left : left + width].copy()

    label_set = set(order_colors)
    strips_by_label: dict[int, np.ndarray] = {}
    for idx in range(strip_count):
        c0 = idx * strip_w
        strip = image[:, c0 : c0 + strip_w].copy()
        label_pos = np.flatnonzero(np.isin(strip[0], order_colors))
        assert label_pos.size == 1, "each strip must have exactly one top-row label pixel"
        label_col = int(label_pos[0])
        label_color = int(strip[0, label_col])
        assert label_color in label_set and label_color not in strips_by_label
        strip[0, label_col] = strip[1, label_col]
        strips_by_label[label_color] = strip

    assert set(strips_by_label) == label_set, "all order colors must match one strip label"
    return np.hstack([strips_by_label[color] for color in order_colors])


def _make_pattern(height: int, width: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    mode = random.randrange(3)
    img = np.empty((height, width), dtype=int)

    if mode == 0:
        row_band = random.randint(1, max(1, height // 2))
        col_band = random.randint(1, max(1, width // 3))
        phase = random.randrange(k)
        for r in range(height):
            for c in range(width):
                idx = (r // row_band + 2 * (c // col_band) + phase) % k
                img[r, c] = palette[idx]
    elif mode == 1:
        a = random.randint(1, 3)
        b = random.randint(1, 4)
        phase = random.randrange(k)
        for r in range(height):
            for c in range(width):
                idx = (a * r + b * c + (c // max(1, width // k)) + phase) % k
                img[r, c] = palette[idx]
    else:
        img.fill(palette[0])
        for _ in range(random.randint(4, 8)):
            color = random.choice(palette[1:] or palette)
            shape = random.randrange(3)
            if shape == 0:
                r0 = random.randint(0, height - 1)
                r1 = random.randint(r0, height - 1)
                img[r0 : r1 + 1, :] = color
            elif shape == 1:
                c0 = random.randint(0, width - 1)
                c1 = random.randint(c0, width - 1)
                img[:, c0 : c1 + 1] = color
            else:
                r0 = random.randint(0, height - 1)
                c0 = random.randint(0, width - 1)
                r1 = random.randint(r0, height - 1)
                c1 = random.randint(c0, width - 1)
                img[r0 : r1 + 1, c0 : c1 + 1] = color

    if np.unique(img).size < 2:
        split = random.randint(1, width - 1)
        img[:, split:] = palette[-1]
    return img


def _marker_columns(left: int, width: int, strip_count: int, strip_w: int) -> list[int]:
    if random.random() < 0.5:
        return [
            left + i * strip_w + random.randint(0, strip_w - 1)
            for i in range(strip_count)
        ]
    return sorted(random.sample(range(left, left + width), strip_count))


def _shuffle_indices(n: int) -> list[int]:
    order = list(range(n))
    while True:
        random.shuffle(order)
        if any(i != order[i] for i in range(n)):
            return order


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    strip_count = random.randint(*STRIP_COUNT_RANGE)
    strip_w = random.randint(*STRIP_WIDTH_RANGE)
    image_h = random.randint(*IMAGE_HEIGHT_RANGE)
    width = strip_count * strip_w

    label_colors = random.sample(NON_FIXED_COLORS, strip_count)
    image_color_pool = [c for c in NON_FIXED_COLORS if c not in label_colors]
    if len(image_color_pool) < 2:
        return None
    image_palette = random.sample(
        image_color_pool,
        random.randint(2, min(4, len(image_color_pool))),
    )

    target = _make_pattern(image_h, width, image_palette)
    label_offsets = [random.randint(0, strip_w - 1) for _ in range(strip_count)]
    for idx, offset in enumerate(label_offsets):
        col = idx * strip_w + offset
        target[0, col] = target[1, col]

    strips: list[np.ndarray] = []
    for idx, color in enumerate(label_colors):
        c0 = idx * strip_w
        strip = target[:, c0 : c0 + strip_w].copy()
        strip[0, label_offsets[idx]] = color
        strips.append(strip)

    shuffle_order = _shuffle_indices(strip_count)
    shredded = np.hstack([strips[idx] for idx in shuffle_order])

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    total_h = top_margin + 1 + image_h + 1 + bottom_margin
    total_w = left_margin + width + right_margin
    if total_h > 30 or total_w > 30:
        return None

    tick_row = top_margin
    image_row = tick_row + 1
    order_row = image_row + image_h
    grid = np.full((total_h, total_w), BLACK, dtype=int)

    for idx in range(strip_count):
        grid[tick_row, left_margin + idx * strip_w] = GREY
    grid[image_row : image_row + image_h, left_margin : left_margin + width] = shredded

    for col, color in zip(_marker_columns(left_margin, width, strip_count, strip_w), label_colors):
        grid[order_row, col] = color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, target):
        return None
    return grid, target


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid shredded-image example")
