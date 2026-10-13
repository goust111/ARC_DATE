from __future__ import annotations

import random

import numpy as np


RED = 2
BLUE = 1
BACKGROUND = 0
BODY_COLORS = [BACKGROUND, 3, 4, 5, 6, 7, 8, 9]
FOREGROUND_COLORS = [3, 4, 5, 6, 7, 8, 9]

PICTURE_HEIGHT_RANGE = (6, 10)
PICTURE_WIDTH_RANGE = (6, 12)
RUN_COUNT_RANGE = (3, 6)
ROW_EDIT_TRIES = 40
GENERATION_RETRIES = 400


def _restore_full_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=int)
    width = int(row.shape[0])
    red_idx = np.flatnonzero(row == RED)
    blue_idx = np.flatnonzero(row == BLUE)
    assert red_idx.size == 1 and blue_idx.size == 1, "each row must contain exactly one red and one blue anchor"

    red = int(red_idx[0])
    blue = int(blue_idx[0])
    is_normal = int(row[(blue + 1) % width]) == RED
    is_flipped = int(row[(red + 1) % width]) == BLUE
    assert is_normal ^ is_flipped, "red and blue must be cyclic neighbors with a unique orientation"

    oriented = row if is_normal else row[::-1]
    red = int(np.flatnonzero(oriented == RED)[0])
    restored = np.roll(oriented, shift=-red)
    assert int(restored[0]) == RED and int(restored[-1]) == BLUE, "anchors must become the two border cells"
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Undo each row's possible mirror and cyclic shift, then trim the red-blue anchor columns.

    Input structure:
    - The true picture originally had a solid red left border and a solid blue right border.
    - Red and blue do not appear inside the picture body; each input row therefore contains exactly one red and one blue anchor.
    - Each row was independently cyclically shifted, and some rows were mirrored left-right before that shift, so the anchors stay cyclically adjacent.

    Rule:
    - For each row, inspect the cyclic direction of the adjacent red-blue pair.
    - If blue's right neighbor is red, the row is already in the normal orientation; if red's right neighbor is blue, first mirror the whole row back.
    - Then cyclically roll the row so red lands in the leftmost column and blue lands in the rightmost column.
    - Delete those two anchor columns and output the recovered middle picture.

    Key insight:
    - The red and blue cells are per-row endpoint anchors: their cyclic order reveals whether that row was mirrored, and their red position then reveals how far to roll the row back.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 1 and width >= 3, "grid must contain at least one body column between the anchors"

    restored_rows = [_restore_full_row(g[r]) for r in range(height)]
    restored = np.stack(restored_rows, axis=0)
    body = restored[:, 1:-1]
    assert not np.any(body == RED) and not np.any(body == BLUE), "anchors must not appear inside the body"
    return body


def _row_is_informative(row: np.ndarray) -> bool:
    return len(np.unique(row)) >= 2 and not np.array_equal(row, row[::-1])


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    points = [0, *cuts, total]
    return [points[i + 1] - points[i] for i in range(parts)]


def _sample_base_row(width: int, palette: list[int]) -> np.ndarray:
    runs = random.randint(RUN_COUNT_RANGE[0], min(RUN_COUNT_RANGE[1], width))
    lengths = _random_composition(width, runs)
    colors = [random.choice(palette)]
    while len(colors) < runs:
        options = [color for color in palette if color != colors[-1]]
        colors.append(random.choice(options))

    row = np.empty(width, dtype=int)
    cursor = 0
    for length, color in zip(lengths, colors):
        row[cursor : cursor + length] = color
        cursor += length
    return row


def _sample_mutated_row(source: np.ndarray, palette: list[int]) -> np.ndarray:
    width = int(source.shape[0])
    row = source.copy()
    edits = random.randint(1, max(2, width // 3))
    for _ in range(edits):
        c0 = random.randint(0, width - 1)
        c1 = random.randint(c0, width - 1)
        options = [color for color in palette if not np.all(row[c0 : c1 + 1] == color)]
        row[c0 : c1 + 1] = random.choice(options)
    return row


def _overlay_blocks(picture: np.ndarray, fg_colors: list[int]) -> None:
    height, width = map(int, picture.shape)
    for _ in range(random.randint(1, 3)):
        r0 = random.randint(0, height - 2)
        r1 = random.randint(r0 + 1, height - 1)
        c0 = random.randint(0, width - 2)
        c1 = random.randint(c0 + 1, width - 1)
        picture[r0 : r1 + 1, c0 : c1 + 1] = random.choice(fg_colors)

    for _ in range(random.randint(1, 3)):
        if random.choice([False, True]):
            r = random.randrange(height)
            c0 = random.randint(0, width - 2)
            c1 = random.randint(c0 + 1, width - 1)
            picture[r, c0 : c1 + 1] = random.choice(fg_colors)
        else:
            c = random.randrange(width)
            r0 = random.randint(0, height - 2)
            r1 = random.randint(r0 + 1, height - 1)
            picture[r0 : r1 + 1, c] = random.choice(fg_colors)


def _picture_ok(picture: np.ndarray) -> bool:
    height, width = map(int, picture.shape)
    informative_rows = sum(_row_is_informative(row) for row in picture)
    if informative_rows < max(4, height // 2 + 1):
        return False
    if len(np.unique(picture)) < 4:
        return False
    if len({tuple(map(int, row)) for row in picture}) < max(4, height // 2):
        return False
    if not np.any(picture[:, : width // 2] != BACKGROUND):
        return False
    if not np.any(picture[:, width // 2 :] != BACKGROUND):
        return False
    return True


def _build_picture(height: int, width: int) -> np.ndarray | None:
    fg_colors = random.sample(FOREGROUND_COLORS, random.randint(3, 5))
    palette = [BACKGROUND, *fg_colors]
    picture = np.empty((height, width), dtype=int)

    for _ in range(ROW_EDIT_TRIES):
        row = _sample_base_row(width, palette)
        if _row_is_informative(row):
            picture[0] = row
            break
    else:
        return None

    for r in range(1, height):
        for _ in range(ROW_EDIT_TRIES):
            source = picture[r - 1] if random.random() < 0.7 else picture[random.randrange(r)]
            row = _sample_mutated_row(source, palette)
            if _row_is_informative(row) and not np.array_equal(row, picture[r - 1]):
                picture[r] = row
                break
        else:
            return None

    _overlay_blocks(picture, fg_colors)
    return picture if _picture_ok(picture) else None


def _decorate_row(body_row: np.ndarray) -> np.ndarray:
    return np.concatenate([np.array([RED], dtype=int), body_row.astype(int), np.array([BLUE], dtype=int)])


def _scramble_row(body_row: np.ndarray, flipped: bool, shift: int) -> np.ndarray:
    row = _decorate_row(body_row)
    if flipped:
        row = row[::-1]
    return np.roll(row, shift=shift)


def _solve_without_flip(grid: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(grid, dtype=int):
        red = int(np.flatnonzero(row == RED)[0])
        restored = np.roll(row, shift=-red)
        rows.append(restored[1:-1])
    return np.stack(rows, axis=0)


def _solve_with_all_rows_flipped(grid: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(grid, dtype=int):
        flipped = row[::-1]
        red = int(np.flatnonzero(flipped == RED)[0])
        restored = np.roll(flipped, shift=-red)
        rows.append(restored[1:-1])
    return np.stack(rows, axis=0)


def _solve_with_linear_adjacency_only(grid: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(grid, dtype=int):
        flip = False
        if np.any((row[:-1] == RED) & (row[1:] == BLUE)):
            flip = True
        elif np.any((row[:-1] == BLUE) & (row[1:] == RED)):
            flip = False
        candidate = row[::-1] if flip else row
        red = int(np.flatnonzero(candidate == RED)[0])
        restored = np.roll(candidate, shift=-red)
        rows.append(restored[1:-1])
    return np.stack(rows, axis=0)


def _semantic_checks(
    grid: np.ndarray,
    picture: np.ndarray,
    flipped_rows: list[bool],
    shifts: list[int],
) -> bool:
    if not np.array_equal(main(grid), picture):
        return False

    if not any(flipped_rows) or all(flipped_rows):
        return False
    if 0 not in shifts or len(set(shifts)) < 3:
        return False

    informative = [_row_is_informative(row) for row in picture]
    if not any(f and s == 0 and info for f, s, info in zip(flipped_rows, shifts, informative)):
        return False
    if not any((not f) and s != 0 and info for f, s, info in zip(flipped_rows, shifts, informative)):
        return False

    full_width = int(grid.shape[1])
    has_internal_anchor_pair = False
    for row in grid:
        red = int(np.flatnonzero(row == RED)[0])
        blue = int(np.flatnonzero(row == BLUE)[0])
        if abs(red - blue) == 1:
            has_internal_anchor_pair = True
        if int(row[(red + 1) % full_width]) == BLUE and int(row[(blue + 1) % full_width]) == RED:
            return False
    if not has_internal_anchor_pair:
        return False

    if np.array_equal(_solve_without_flip(grid), picture):
        return False
    if np.array_equal(_solve_with_all_rows_flipped(grid), picture):
        return False
    if np.array_equal(_solve_with_linear_adjacency_only(grid), picture):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*PICTURE_HEIGHT_RANGE)
    width = random.randint(*PICTURE_WIDTH_RANGE)
    picture = _build_picture(height, width)
    if picture is None:
        return None

    row_width = width + 2
    informative_rows = [r for r in range(height) if _row_is_informative(picture[r])]
    if len(informative_rows) < 4:
        return None

    flipped_rows = [random.choice([False, True]) for _ in range(height)]
    shifts = [random.randrange(row_width) for _ in range(height)]

    forced_flip_boundary = random.choice(informative_rows)
    forced_normal_internal = random.choice([r for r in informative_rows if r != forced_flip_boundary])
    forced_flip_internal = random.choice([r for r in informative_rows if r not in {forced_flip_boundary, forced_normal_internal}])

    flipped_rows[forced_flip_boundary] = True
    shifts[forced_flip_boundary] = 0
    flipped_rows[forced_normal_internal] = False
    shifts[forced_normal_internal] = random.randint(1, row_width - 1)
    flipped_rows[forced_flip_internal] = True
    shifts[forced_flip_internal] = random.randint(1, row_width - 1)

    if all(flipped_rows) or not any(flipped_rows):
        flipped_rows[random.randrange(height)] = not flipped_rows[random.randrange(height)]

    if 0 not in shifts:
        shifts[random.randrange(height)] = 0
    if len(set(shifts)) < 3:
        shifts[random.randrange(height)] = (shifts[random.randrange(height)] + 1) % row_width
        shifts[random.randrange(height)] = (shifts[random.randrange(height)] + 2) % row_width

    grid = np.stack(
        [_scramble_row(picture[r], flipped_rows[r], shifts[r]) for r in range(height)],
        axis=0,
    )
    return grid if _semantic_checks(grid, picture, flipped_rows, shifts) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
