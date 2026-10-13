from __future__ import annotations

import random

import numpy as np


BAR_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 300


def _find_separator_color(grid: np.ndarray) -> int:
    full_column_colors: list[int] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != 0 and np.all(col == col[0]):
            full_column_colors.append(int(col[0]))
    assert full_column_colors, "missing full-height separator columns"

    best_color = full_column_colors[0]
    best_count = 0
    for color in dict.fromkeys(full_column_colors):
        count = full_column_colors.count(color)
        if count > best_count:
            best_color = color
            best_count = count
    return best_color


def _window_spans(grid: np.ndarray, separator_color: int) -> list[tuple[int, int]]:
    is_separator = np.all(grid == int(separator_color), axis=0)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for c, flag in enumerate(is_separator):
        if flag:
            if start is not None:
                spans.append((start, c))
                start = None
        elif start is None:
            start = c
    if start is not None:
        spans.append((start, grid.shape[1]))
    assert spans, "expected at least one non-separator window"
    return spans


def _full_row_map(window: np.ndarray) -> dict[int, list[int]]:
    row_map: dict[int, list[int]] = {}
    for r, row in enumerate(window):
        if row.size and int(row[0]) != 0 and np.all(row == row[0]):
            row_map.setdefault(int(row[0]), []).append(int(r))
    return row_map


def _tight_crop(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != 0)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align separator-sliced sprite strips by a shared ridge row, then restore missing windows by 180-degree rotational symmetry.

    Input structure:
    - A black-background grid is split into left-to-right windows by full-height grey separator columns.
    - Each non-empty window contains one vertical slice of the same hidden multicolor sprite, but independently shifted up or down.
    - The sprite contains one single-color ridge row that spans its full width, while one or two windows may be entirely black placeholders.

    Rule:
    - Find the separator color from the full-height divider columns and split the grid into ordered windows.
    - Identify the ridge color as the non-black color that forms a complete row inside every non-empty window, then vertically realign windows so those ridge rows coincide.
    - Remove separator columns, concatenate the windows in order while preserving empty-window widths, use 180-degree rotational symmetry to fill missing black columns, and return the tight non-black bounding box.

    Key insight:
    - The windows preserve horizontal slice order, the shared ridge row fixes each slice's vertical offset, and rotational symmetry supplies the contents of windows that were replaced by black placeholders.
    """
    g = np.asarray(grid, dtype=int)
    separator_color = _find_separator_color(g)
    spans = _window_spans(g, separator_color)
    windows = [g[:, c0:c1] for c0, c1 in spans]

    non_empty_windows = [window for window in windows if np.any(window != 0)]
    assert non_empty_windows, "expected at least one visible slice"

    row_maps = [_full_row_map(window) for window in non_empty_windows]
    common_colors = set(row_maps[0])
    for row_map in row_maps[1:]:
        common_colors &= set(row_map)
    assert common_colors, "failed to locate a shared ridge color"

    first_window_order = [color for color in _full_row_map(non_empty_windows[0]) if color in common_colors]
    ridge_color = min(
        first_window_order,
        key=lambda color: sum(len(row_map[color]) for row_map in row_maps),
    )

    ridge_rows: list[int | None] = []
    half_height = 0
    for window in windows:
        if not np.any(window != 0):
            ridge_rows.append(None)
            continue

        row_map = _full_row_map(window)
        assert ridge_color in row_map, "visible window is missing the ridge row"
        rows = row_map[ridge_color]
        ridge_row = rows[len(rows) // 2]
        ridge_rows.append(ridge_row)

        coords = np.argwhere(window != 0)
        top = int(coords[:, 0].min())
        bottom = int(coords[:, 0].max())
        half_height = max(half_height, ridge_row - top, bottom - ridge_row)

    stitched = np.zeros((2 * half_height + 1, sum(window.shape[1] for window in windows)), dtype=int)
    x = 0
    for window, ridge_row in zip(windows, ridge_rows):
        width = window.shape[1]
        if ridge_row is not None:
            delta = half_height - ridge_row
            dst_r0 = max(0, delta)
            src_r0 = max(0, -delta)
            dst_r1 = min(stitched.shape[0], delta + window.shape[0])
            src_r1 = src_r0 + (dst_r1 - dst_r0)
            visible = window[src_r0:src_r1]
            target = stitched[dst_r0:dst_r1, x : x + width]
            mask = visible != 0
            target[mask] = visible[mask]
        x += width

    rotated = np.rot90(stitched, 2)
    completed = np.where(stitched != 0, stitched, rotated)
    return _tight_crop(completed)


def _make_mirrored_widths(n_windows: int) -> list[int]:
    pair_count = n_windows // 2
    left = [random.randint(2, 4) for _ in range(pair_count)]
    if len(set(left)) == 1 and pair_count > 1:
        idx = random.randrange(pair_count)
        left[idx] = random.choice([width for width in (2, 3, 4) if width != left[idx]])
    return left + left[::-1]


def _build_sprite(widths: list[int]) -> np.ndarray:
    half_height = random.randint(2, 4)
    height = 2 * half_height + 1
    width = sum(widths)

    colors = random.sample(PALETTE, random.randint(3, 5))
    ridge_color = colors[0]
    body_colors = colors[1:]

    sprite = np.zeros((height, width), dtype=int)
    sprite[half_height, :] = ridge_color

    for r in range(half_height):
        x = 0
        for window_width in widths:
            fill_count = random.randint(1, window_width - 1)
            filled_cols = random.sample(range(x, x + window_width), fill_count)
            for c in filled_cols:
                sprite[r, c] = random.choice(body_colors)
            x += window_width

    for r in range(half_height):
        for c in range(width):
            color = int(sprite[r, c])
            if color != 0:
                sprite[height - 1 - r, width - 1 - c] = color

    return sprite


def _choose_missing_windows(n_windows: int) -> set[int]:
    pair_count = n_windows // 2
    missing_count = random.randint(1, min(2, pair_count))
    missing_pairs = random.sample(range(pair_count), missing_count)
    missing: set[int] = set()
    for pair_idx in missing_pairs:
        left = pair_idx
        right = n_windows - 1 - pair_idx
        missing.add(random.choice((left, right)))
    return missing


def _render_input(sprite: np.ndarray, widths: list[int], missing: set[int]) -> np.ndarray | None:
    sprite_h = sprite.shape[0]
    extra_rows = random.randint(2, 6)
    grid_h = sprite_h + extra_rows
    grid_w = sprite.shape[1] + len(widths) - 1

    visible_windows = [idx for idx in range(len(widths)) if idx not in missing]
    offsets: dict[int, int] = {}
    for _ in range(60):
        for idx in visible_windows:
            offsets[idx] = random.randint(0, extra_rows)
        if len({offsets[idx] for idx in visible_windows}) >= min(2, len(visible_windows)):
            break
    else:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    x_in = 0
    x_sprite = 0
    for idx, window_width in enumerate(widths):
        if idx not in missing:
            top = offsets[idx]
            grid[top : top + sprite_h, x_in : x_in + window_width] = sprite[
                :, x_sprite : x_sprite + window_width
            ]
        x_in += window_width
        x_sprite += window_width
        if idx < len(widths) - 1:
            grid[:, x_in] = BAR_COLOR
            x_in += 1

    return grid


def _generate_attempt() -> np.ndarray | None:
    n_windows = random.choice((4, 6))
    widths = _make_mirrored_widths(n_windows)
    sprite = _build_sprite(widths)
    missing = _choose_missing_windows(n_windows)
    grid = _render_input(sprite, widths, missing)
    if grid is None:
        return None

    expected = _tight_crop(sprite)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ridge-alignment puzzle input")
