import random
from collections import Counter

import numpy as np


MARK_COLOR = 2
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _sprite_windows(grid: np.ndarray, bg: int) -> tuple[list[dict], list[tuple[int, int]], list[tuple[int, int]]]:
    occupied = grid != bg
    row_runs = _contiguous_runs(np.flatnonzero(np.any(occupied, axis=1)))
    col_runs = _contiguous_runs(np.flatnonzero(np.any(occupied, axis=0)))
    assert row_runs and col_runs, "expected at least one sprite band"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1 and len(col_widths) == 1, "sprite bands must share one size"

    windows: list[dict] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = [int(color) for color in np.unique(patch) if int(color) != bg]
            assert len(colors) == 1, "each sprite window must contain exactly one foreground color"
            color = colors[0]
            windows.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "mask": patch == color,
                }
            )
    return windows, row_runs, col_runs


def main(grid: np.ndarray) -> np.ndarray:
    """Mark, in red, every pixel where a same-sized copy differs from the uniquely colored prototype."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    windows, row_runs, col_runs = _sprite_windows(g, bg)
    assert len(windows) >= 3, "expected one prototype and multiple copies"
    assert len(row_runs) * len(col_runs) == len(windows)

    color_counts = Counter(window["color"] for window in windows)
    assert len(color_counts) == 2, "expected one prototype color and one copy color"

    prototype_colors = [color for color, count in color_counts.items() if count == 1]
    copy_colors = [color for color, count in color_counts.items() if count >= 2]
    assert len(prototype_colors) == 1 and len(copy_colors) == 1

    prototype_color = prototype_colors[0]
    copy_color = copy_colors[0]
    prototype = next(window for window in windows if window["color"] == prototype_color)
    prototype_mask = prototype["mask"]

    out = g.copy()
    for window in windows:
        if window["color"] != copy_color:
            continue
        diff = np.logical_xor(window["mask"], prototype_mask)
        diff_count = int(diff.sum())
        assert 1 <= diff_count <= 3, "copy must differ from prototype in 1..3 pixels"
        r0, c0, r1, c1 = window["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1][diff] = MARK_COLOR

    return out


def _random_dimensions() -> tuple[int, int, int, int]:
    choices = [(1, 3), (1, 4), (2, 2), (2, 3), (3, 1), (3, 2)]
    rows, cols = random.choice(choices)
    sprite_h = random.randint(5, 7)
    sprite_w = random.randint(5, 7)
    return rows, cols, sprite_h, sprite_w


def _prototype_mask(height: int, width: int) -> np.ndarray:
    inner_h = height - 2
    inner_w = width - 2
    total_inner = inner_h * inner_w
    assert total_inner >= 9

    for _ in range(200):
        fill_count = random.randint(3, total_inner - 3)
        interior = np.zeros((inner_h, inner_w), dtype=bool)
        chosen = random.sample(range(total_inner), fill_count)
        interior.flat[chosen] = True

        mask = np.ones((height, width), dtype=bool)
        mask[1:-1, 1:-1] = interior
        if int(interior.sum()) >= 3 and int((~interior).sum()) >= 3:
            return mask

    raise RuntimeError("failed to build prototype mask")


def _choose_diff_mask(prototype_mask: np.ndarray, forced_mode: str | None = None) -> np.ndarray:
    result = prototype_mask.copy()
    interior = result[1:-1, 1:-1]
    remove_positions = [tuple(pos) for pos in np.argwhere(interior)]
    add_positions = [tuple(pos) for pos in np.argwhere(~interior)]
    diff_count = random.randint(1, 3)

    options: list[tuple[str, int]] = []
    if len(remove_positions) >= diff_count:
        options.append(("remove", diff_count))
    if len(add_positions) >= diff_count:
        options.append(("add", diff_count))
    if diff_count >= 2:
        for remove_count in range(1, diff_count):
            add_count = diff_count - remove_count
            if len(remove_positions) >= remove_count and len(add_positions) >= add_count:
                options.append(("mixed", remove_count))

    if forced_mode is not None:
        filtered = [option for option in options if option[0] == forced_mode]
        if filtered:
            options = filtered
    assert options, "prototype must support at least one visible difference mode"

    mode, remove_count = random.choice(options)
    if mode == "remove":
        for r, c in random.sample(remove_positions, diff_count):
            result[r + 1, c + 1] = False
    elif mode == "add":
        for r, c in random.sample(add_positions, diff_count):
            result[r + 1, c + 1] = True
    else:
        add_count = diff_count - remove_count
        for r, c in random.sample(remove_positions, remove_count):
            result[r + 1, c + 1] = False
        for r, c in random.sample(add_positions, add_count):
            result[r + 1, c + 1] = True
    return result


def _build_grid() -> np.ndarray:
    rows, cols, sprite_h, sprite_w = _random_dimensions()
    gap_r = random.randint(1, 4)
    gap_c = random.randint(1, 4)
    margin_top = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_right = random.randint(1, 3)

    height = margin_top + rows * sprite_h + (rows - 1) * gap_r + margin_bottom
    width = margin_left + cols * sprite_w + (cols - 1) * gap_c + margin_right

    bg, prototype_color, copy_color = random.sample([color for color in PALETTE if color != MARK_COLOR], 3)
    grid = np.full((height, width), bg, dtype=int)

    prototype_mask = _prototype_mask(sprite_h, sprite_w)
    slot_count = rows * cols
    prototype_index = random.randrange(slot_count)
    forced_modes = [None] * slot_count
    copy_slots = [idx for idx in range(slot_count) if idx != prototype_index]
    if len(copy_slots) >= 2:
        forced_modes[copy_slots[0]] = "remove"
        forced_modes[copy_slots[1]] = "add"

    for idx in range(slot_count):
        row_idx = idx // cols
        col_idx = idx % cols
        r0 = margin_top + row_idx * (sprite_h + gap_r)
        c0 = margin_left + col_idx * (sprite_w + gap_c)
        patch = grid[r0 : r0 + sprite_h, c0 : c0 + sprite_w]

        if idx == prototype_index:
            mask = prototype_mask
            color = prototype_color
        else:
            mask = _choose_diff_mask(prototype_mask, forced_mode=forced_modes[idx])
            color = copy_color
        patch[mask] = color

    return grid


def _is_valid_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= max(count for color, count in enumerate(counts) if color != bg):
        return False

    try:
        out = main(grid)
        windows, _, _ = _sprite_windows(grid, bg)
    except Exception:
        return False

    prototype_window = None
    color_counts = Counter(window["color"] for window in windows)
    for window in windows:
        if color_counts[window["color"]] == 1:
            prototype_window = window
            break
    if prototype_window is None:
        return False

    prototype_mask = prototype_window["mask"]
    for window in windows:
        if window is prototype_window:
            continue
        diff = np.logical_xor(window["mask"], prototype_mask)
        if not (1 <= int(diff.sum()) <= 3):
            return False
        r0, c0, r1, c1 = window["bbox"]
        patch_out = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch_out[diff] == MARK_COLOR):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_grid()
        if _is_valid_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
