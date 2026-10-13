import random

import numpy as np


BG = 0
MAX_GRID_SIZE = 30
MAX_KEY_WIDTH = 4
GEN_RETRIES = 400


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    solid_rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BG and np.all(grid[r, :] == grid[r, 0])
    ]
    assert len(solid_rows) == 1, "expected exactly one solid non-black separator row"
    row = int(solid_rows[0])
    return row, int(grid[row, 0])


def _label_columns(grid: np.ndarray, separator_row: int, separator_color: int) -> dict[int, int]:
    labels: dict[int, int] = {}
    for c, value in enumerate(grid[0]):
        color = int(value)
        if color in (BG, separator_color):
            continue
        assert color not in labels, "label colors must be unique"
        labels[color] = int(c)
    assert labels, "expected at least one column label"
    return labels


def _extract_key_ops(lower: np.ndarray, label_colors: set[int]) -> list[tuple[int, int, int]]:
    height, width = map(int, lower.shape)
    ops: list[tuple[int, int, int]] = []
    for r in range(height):
        left_color = int(lower[r, 0])
        left_width = 0
        if left_color in label_colors:
            while left_width < width and int(lower[r, left_width]) == left_color:
                left_width += 1
            ops.append((int(r), int(left_color), int(left_width)))

        right_color = int(lower[r, width - 1])
        if right_color not in label_colors:
            continue

        right_width = 0
        while right_width < width and int(lower[r, width - 1 - right_width]) == right_color:
            right_width += 1

        # Avoid double-counting one run if a malformed row is label-colored across the full width.
        if left_width and width - right_width < left_width:
            continue
        ops.append((int(r), int(right_color), int(right_width)))

    ops.sort(key=lambda item: item[0])
    return ops


def _apply_column_shifts(
    lower: np.ndarray,
    ops: list[tuple[int, int, int]],
    color_to_col: dict[int, int],
) -> np.ndarray:
    out = np.array(lower, dtype=int, copy=True)
    height = int(out.shape[0])
    if height == 0:
        return out

    for _row, color, steps in ops:
        if color not in color_to_col:
            continue
        shift = int(steps) % height
        if shift == 0:
            continue
        col = int(color_to_col[color])
        out[:, col] = np.roll(out[:, col], shift)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Unlock labeled columns by using each edge key block's color as the column selector and its width as the downward cyclic shift.

    Input structure:
    - The background is black.
    - A single solid grey row splits the grid into an upper label area and a lower work area.
    - The first row of the upper area contains isolated one-pixel color labels, each naming one column.
    - In the lower area, a multicolor pattern shares the canvas with one-row key blocks placed flush against the left or right edge; key colors match upper labels.

    Rule:
    - Read the top-row label colors to map each key color to its target column.
    - For every lower-area edge key block, in top-to-bottom order, rotate the entire target column downward within the lower area.
    - The block width is the rotation amount, and rotations are cyclic so pixels falling off the bottom reappear at the top.
    - Return only the transformed lower area, leaving untargeted columns unchanged.

    Key insight:
    - The key block color does not directly act on its own position; it must first be translated through the upper label row to identify which lower-area column to move.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row, separator_color = _find_separator_row(g)
    color_to_col = _label_columns(g, separator_row, separator_color)
    lower = g[separator_row + 1 :, :]
    ops = _extract_key_ops(lower, set(color_to_col))
    return _apply_column_shifts(lower, ops, color_to_col)


def _try_place_patch(
    canvas: np.ndarray,
    occupied: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    left_margin: int,
    right_margin: int,
) -> bool:
    height, width = map(int, canvas.shape)
    if h <= 0 or w <= 0:
        return False
    if not (0 <= r0 and 0 <= c0 and r0 + h <= height and c0 + w <= width):
        return False
    if c0 < left_margin or c0 + w > width - right_margin:
        return False
    if occupied[r0 : r0 + h, c0 : c0 + w].any():
        return False
    canvas[r0 : r0 + h, c0 : c0 + w] = int(color)
    occupied[r0 : r0 + h, c0 : c0 + w] = True
    return True


def _place_anchor_shape(
    canvas: np.ndarray,
    occupied: np.ndarray,
    anchor_col: int,
    palette: list[int],
    left_margin: int,
    right_margin: int,
) -> bool:
    height, width = map(int, canvas.shape)
    for _ in range(40):
        color = random.choice(palette)
        kind = random.choice(["dot", "hbar", "vbar", "rect"])
        anchor_row = random.randrange(height)

        if kind == "dot":
            if _try_place_patch(
                canvas, occupied, anchor_row, anchor_col, 1, 1, color, left_margin, right_margin
            ):
                return True
            continue

        if kind == "hbar":
            patch_w = random.randint(2, 4)
            patch_h = 1
            lo = max(left_margin, anchor_col - patch_w + 1)
            hi = min(anchor_col, width - right_margin - patch_w)
            if lo > hi:
                continue
            c0 = random.randint(lo, hi)
            if _try_place_patch(
                canvas, occupied, anchor_row, c0, patch_h, patch_w, color, left_margin, right_margin
            ):
                return True
            continue

        if kind == "vbar":
            patch_h = random.randint(2, min(4, height))
            patch_w = 1
            lo = max(0, anchor_row - patch_h + 1)
            hi = min(anchor_row, height - patch_h)
            if lo > hi:
                continue
            r0 = random.randint(lo, hi)
            if _try_place_patch(
                canvas, occupied, r0, anchor_col, patch_h, patch_w, color, left_margin, right_margin
            ):
                return True
            continue

        patch_h = random.randint(1, min(3, height))
        patch_w = random.randint(2, 4)
        row_lo = max(0, anchor_row - patch_h + 1)
        row_hi = min(anchor_row, height - patch_h)
        col_lo = max(left_margin, anchor_col - patch_w + 1)
        col_hi = min(anchor_col, width - right_margin - patch_w)
        if row_lo > row_hi or col_lo > col_hi:
            continue
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        if _try_place_patch(
            canvas, occupied, r0, c0, patch_h, patch_w, color, left_margin, right_margin
        ):
            return True

    return False


def _paint_pattern(
    lower: np.ndarray,
    label_cols: list[int],
    palette: list[int],
    left_margin: int,
    right_margin: int,
) -> bool:
    occupied = np.zeros_like(lower, dtype=bool)
    used_colors: set[int] = set()

    for col in label_cols:
        placed_for_col = 0
        target_shapes = random.randint(2, 4)
        for _ in range(target_shapes):
            if _place_anchor_shape(lower, occupied, int(col), palette, left_margin, right_margin):
                placed_for_col += 1
        if placed_for_col == 0:
            return False
        used_colors.update(int(v) for v in lower[:, col] if int(v) != BG)

    extra_shapes = random.randint(3, 7)
    height, width = map(int, lower.shape)
    for _ in range(extra_shapes):
        for _ in range(20):
            patch_h = random.randint(1, min(3, height))
            patch_w = random.randint(1, 4)
            r0 = random.randint(0, height - patch_h)
            c0 = random.randint(left_margin, width - right_margin - patch_w)
            color = random.choice(palette)
            if _try_place_patch(
                lower, occupied, r0, c0, patch_h, patch_w, color, left_margin, right_margin
            ):
                used_colors.add(int(color))
                break

    if len(used_colors) < 2:
        for color in palette:
            for r in range(lower.shape[0]):
                for c in range(left_margin, lower.shape[1] - right_margin):
                    if not occupied[r, c]:
                        lower[r, c] = int(color)
                        occupied[r, c] = True
                        used_colors.add(int(color))
                        if len(used_colors) >= 2:
                            break
                if len(used_colors) >= 2:
                    break
            if len(used_colors) >= 2:
                break

    non_bg = int(np.count_nonzero(lower != BG))
    return len(used_colors) >= 2 and non_bg >= len(label_cols) * 2


def _sample_label_columns(width: int, count: int, left_margin: int, right_margin: int) -> list[int] | None:
    interior = list(range(left_margin, width - right_margin))
    if len(interior) < count:
        return None
    cols = sorted(random.sample(interior, count))
    return [int(c) for c in cols]


def _sample_key_colors(label_colors: list[int], key_count: int, force_repeat: bool) -> list[int]:
    if force_repeat and key_count >= 3:
        repeated = random.choice(label_colors)
        colors = [int(repeated), int(repeated)]
        while len(colors) < key_count:
            colors.append(int(random.choice(label_colors)))
        random.shuffle(colors)
        return colors

    if key_count <= len(label_colors):
        return [int(c) for c in random.sample(label_colors, key_count)]

    return [int(random.choice(label_colors)) for _ in range(key_count)]


def _sample_key_sides(key_count: int) -> list[str]:
    mode = random.choice(["left", "right", "mixed"])
    if mode == "left":
        return ["left"] * key_count
    if mode == "right":
        return ["right"] * key_count

    sides = ["left", "right"]
    while len(sides) < key_count:
        sides.append(random.choice(["left", "right"]))
    random.shuffle(sides)
    return sides


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(16, 26)
    upper_blank_rows = random.randint(0, 3)
    lower_height = random.randint(7, 15)
    total_height = 1 + upper_blank_rows + 1 + lower_height
    if total_height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
        return None

    left_margin = right_margin = MAX_KEY_WIDTH
    label_count = random.randint(3, min(5, width - left_margin - right_margin))
    label_cols = _sample_label_columns(width, label_count, left_margin, right_margin)
    if label_cols is None:
        return None

    separator_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    color_pool = [c for c in range(10) if c not in (BG, separator_color)]
    random.shuffle(color_pool)
    label_colors = [int(c) for c in color_pool[:label_count]]
    pattern_palette = [int(c) for c in color_pool[label_count : label_count + random.randint(2, 3)]]
    if len(pattern_palette) < 2:
        return None

    lower = np.full((lower_height, width), BG, dtype=int)
    if not _paint_pattern(lower, label_cols, pattern_palette, left_margin, right_margin):
        return None

    key_count = random.randint(2, min(6, lower_height))
    key_rows = sorted(random.sample(range(lower_height), key_count))
    key_colors = _sample_key_colors(label_colors, key_count, force_repeat=random.random() < 0.5)
    key_sides = _sample_key_sides(key_count)
    key_widths = [random.randint(1, MAX_KEY_WIDTH) for _ in range(key_count)]

    ops: list[tuple[int, int, int]] = []
    for row, color, side, block_width in zip(key_rows, key_colors, key_sides, key_widths):
        if side == "left":
            lower[row, :block_width] = int(color)
        else:
            lower[row, width - block_width :] = int(color)
        ops.append((int(row), int(color), int(block_width)))

    grid = np.full((total_height, width), BG, dtype=int)
    for col, color in zip(label_cols, label_colors):
        grid[0, int(col)] = int(color)
    separator_row = 1 + upper_blank_rows
    grid[separator_row, :] = int(separator_color)
    grid[separator_row + 1 :, :] = lower

    color_to_col = {int(color): int(col) for color, col in zip(label_colors, label_cols)}
    expected = _apply_column_shifts(lower, sorted(ops, key=lambda item: item[0]), color_to_col)

    if np.array_equal(expected, lower):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
