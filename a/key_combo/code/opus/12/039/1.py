from __future__ import annotations

import random

import numpy as np


CONTROL_FILL = 5
CONTROL_MARK = 7
NESTED_COLOR_POOL = (1, 2, 3, 4, 6)
PAYLOAD_COLOR_POOL = (0, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    parts = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    assert boundary.size > 0, "expected a non-empty nested boundary"
    colors = np.unique(boundary)
    assert colors.size == 1, "each peeled boundary must be monochromatic"
    return int(colors[0])


def _layer_order(rect: np.ndarray) -> list[int]:
    top, left = 0, 0
    bottom, right = map(int, rect.shape)
    order: list[int] = []

    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        order.append(color)
        previous = (top, bottom, left, right)

        while top < bottom and np.all(rect[top, left:right] == color):
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            right -= 1

        assert (top, bottom, left, right) != previous, "peeling must remove at least one stripe"

    assert len(order) >= 3, "expected 3-5 nested layers"
    assert len(order) == len(set(order)), "nested layer colors must be unique"
    return order


def _find_control_column(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        col = grid[:, c]
        marker_rows = np.flatnonzero(col == CONTROL_MARK)
        if marker_rows.size != 1:
            continue
        if np.count_nonzero(col == CONTROL_FILL) != h - 1:
            continue
        if np.any((col != CONTROL_FILL) & (col != CONTROL_MARK)):
            continue
        if c == 0 or c == w - 1:
            continue
        candidates.append((c, int(marker_rows[0])))
    assert len(candidates) == 1, "expected exactly one grey control column with one orange marker"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the control column, roll the right payload by the encoded offsets, and reverse the left nested layer colors.

    Input structure:
    - The grid is split into a left rectangular nested-color block and a right full-height payload block.
    - Exactly one grey control column lies between them; it is filled with grey except for one orange marker pixel.
    - The left block is a solid concentric nested rectangle whose outermost layer touches all four edges of that block.

    Rule:
    - Find the control column and read the orange marker's 0-based row index as the vertical roll amount.
    - Count the left nested rectangle's color layers by repeatedly peeling uniform outer boundaries; this count is the horizontal roll amount.
    - Remove the control column, reverse the left block's layer-color order, and cyclically roll the right block down by the marker row and right by the layer count.

    Key insight:
    - The nested rectangle has a dual role: its depth encodes the payload's horizontal shift while the same layers must also be recolored in reverse order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    control_col, row_shift = _find_control_column(g)
    left = g[:, :control_col]
    right = g[:, control_col + 1 :]
    assert left.size > 0 and right.size > 0, "control column must separate two non-empty regions"

    layers = _layer_order(left)
    col_shift = len(layers)

    reversed_left = left.copy()
    for src, dst in zip(layers, reversed(layers)):
        reversed_left[left == src] = int(dst)

    rolled_right = np.roll(right, shift=(row_shift, col_shift), axis=(0, 1))
    return np.concatenate([reversed_left, rolled_right], axis=1)


def _sample_thicknesses(height: int, width: int, n_layers: int) -> list[int] | None:
    thicks: list[int] = []
    h, w = int(height), int(width)
    for i in range(n_layers - 1):
        remaining_layers = n_layers - i - 1
        min_after = 2 * (remaining_layers - 1) + 1
        max_t = min(2, (min(h, w) - min_after) // 2)
        if max_t < 1:
            return None
        thicks.append(random.randint(1, max_t))
        h -= 2 * thicks[-1]
        w -= 2 * thicks[-1]
    return thicks


def _build_nested_rect(height: int, width: int, n_layers: int) -> tuple[np.ndarray, np.ndarray] | None:
    thicks = _sample_thicknesses(height, width, n_layers)
    if thicks is None:
        return None

    colors = random.sample(list(NESTED_COLOR_POOL), n_layers)
    rect = np.zeros((height, width), dtype=int)
    out = np.zeros_like(rect)
    top, left = 0, 0
    bottom, right = int(height), int(width)

    for i, color in enumerate(colors):
        rect[top:bottom, left:right] = int(color)
        out[top:bottom, left:right] = int(colors[-1 - i])
        if i < len(thicks):
            t = thicks[i]
            top += t
            left += t
            bottom -= t
            right -= t

    return rect, out


def _target_counts(area: int, color_count: int) -> list[int]:
    base = [2] * color_count
    remaining = area - sum(base)
    assert remaining >= 0
    for _ in range(remaining):
        base[random.randrange(color_count)] += 1
    random.shuffle(base)
    return base


def _frontier(labels: np.ndarray, label: int) -> list[tuple[int, int]]:
    h, w = labels.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(labels == label):
        rr = int(r)
        cc = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                cells.add((nr, nc))
    return list(cells)


def _grow_payload(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    color_count = len(colors)
    area = height * width
    labels = np.full((height, width), -1, dtype=int)
    seeds = random.sample(range(area), color_count)
    targets = _target_counts(area, color_count)
    have = [1] * color_count

    for label, pos in enumerate(seeds):
        labels[pos // width, pos % width] = label

    while np.any(labels == -1):
        active = [label for label in range(color_count) if have[label] < targets[label]]
        if not active:
            return None

        chosen = None
        random.shuffle(active)
        for label in active:
            frontier = _frontier(labels, label)
            if frontier:
                r, c = random.choice(frontier)
                chosen = (label, r, c)
                break
        if chosen is None:
            return None

        label, r, c = chosen
        labels[r, c] = label
        have[label] += 1

    payload = np.array(colors, dtype=int)[labels]
    return payload


def _sample_marker_row(height: int) -> int:
    branch = random.choice(("top", "bottom", "any"))
    if branch == "top":
        return 0
    if branch == "bottom":
        return height - 1
    return random.randint(0, height - 1)


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.randint(3, 5)
    min_side = 2 * (n_layers - 1) + 1

    height = random.randint(max(10, min_side), 18)
    left_width = random.randint(max(7, min_side), 12)
    right_min = max(6, n_layers)
    right_max = min(12, 29 - left_width)
    if right_max < right_min:
        return None
    right_width = random.randint(right_min, right_max)

    nested_built = _build_nested_rect(height, left_width, n_layers)
    if nested_built is None:
        return None
    nested, nested_out = nested_built

    payload_color_count = random.choice((2, 3))
    payload_colors = random.sample(list(PAYLOAD_COLOR_POOL), payload_color_count)
    payload = _grow_payload(height, right_width, payload_colors)
    if payload is None:
        return None

    marker_row = _sample_marker_row(height)
    rolled_payload = np.roll(payload, shift=(marker_row, n_layers), axis=(0, 1))
    if np.array_equal(payload, rolled_payload):
        return None

    control = np.full((height, 1), CONTROL_FILL, dtype=int)
    control[marker_row, 0] = CONTROL_MARK
    grid = np.concatenate([nested, control, payload], axis=1)
    expected = np.concatenate([nested_out, rolled_payload], axis=1)

    if not np.array_equal(main(grid), expected):
        return None

    if len(_layer_order(nested)) != n_layers:
        return None

    if np.array_equal(payload, np.roll(payload, shift=(0, n_layers), axis=(0, 1))):
        return None
    if marker_row > 0 and np.array_equal(payload, np.roll(payload, shift=(marker_row, 0), axis=(0, 1))):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth payload-shift example")
