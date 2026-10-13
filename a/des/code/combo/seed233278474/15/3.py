from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
GREEN = 3
MARKER_COLORS = (BLUE, GREEN)
PATTERN_COLORS = (2, 4, 5, 6, 7, 8, 9)

# Generator configuration.
PATTERN_HEIGHT_RANGE = (7, 15)
PATTERN_WIDTH_RANGE = (7, 15)
VERTICAL_FOLD_RANGE = (1, 3)
HORIZONTAL_FOLD_RANGE = (1, 3)
MIN_FINAL_SIDE = 2
OBJECT_COUNT_RANGE = (5, 10)
SHAPE_PALETTE_SIZE_RANGE = (2, 4)
MAX_PATTERN_BUILD_ATTEMPTS = 120
MAX_INSTRUCTION_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 500


def _transform_position(pos: int, axis: int, direction: int, length: int) -> int | None:
    if pos == axis:
        return None

    if direction == BLUE:
        new_pos = pos if pos < axis else 2 * axis - pos
        new_length = axis
    else:
        new_pos = axis - pos - 1 if pos < axis else pos - axis - 1
        new_length = length - axis - 1

    if not 0 <= new_pos < new_length:
        return None
    return int(new_pos)


def _simulate_instruction_positions(
    length: int,
    instructions: list[tuple[int, int]],
) -> tuple[list[int], int] | None:
    current_positions = [int(pos) for pos, _direction in instructions]
    current_length = int(length)
    executed_axes: list[int] = []

    for idx, (_orig_pos, direction) in enumerate(instructions):
        axis = current_positions[idx]
        if not 0 < axis < current_length - 1:
            return None
        executed_axes.append(axis)

        for future_idx in range(idx + 1, len(instructions)):
            transformed = _transform_position(
                current_positions[future_idx],
                axis,
                direction,
                current_length,
            )
            if transformed is None:
                return None
            current_positions[future_idx] = transformed

        current_length = axis if direction == BLUE else current_length - axis - 1
        if current_length < 1:
            return None

    return executed_axes, current_length


def _merge_source_onto_target(target: np.ndarray, source: np.ndarray) -> int:
    fill_mask = (target == BACKGROUND) & (source != BACKGROUND)
    target[fill_mask] = source[fill_mask]
    return int(fill_mask.sum())


def _fold_vertical(pattern: np.ndarray, axis: int, direction: int) -> tuple[np.ndarray, int, int]:
    height, width = pattern.shape
    assert 0 < axis < width - 1, "vertical fold axis must have cells on both sides"

    if direction == BLUE:
        target = pattern[:, :axis].copy()
        source = pattern[:, axis + 1 :]
        overlap = min(axis, width - axis - 1)
        inserted = 0
        if overlap > 0:
            mirrored = source[:, :overlap][:, ::-1]
            inserted = _merge_source_onto_target(target[:, axis - overlap : axis], mirrored)
        return target, int(np.count_nonzero(source)), inserted

    assert direction == GREEN, "vertical fold marker must be blue or green"
    target = pattern[:, axis + 1 :].copy()
    source = pattern[:, :axis]
    overlap = min(axis, width - axis - 1)
    inserted = 0
    if overlap > 0:
        mirrored = source[:, axis - overlap : axis][:, ::-1]
        inserted = _merge_source_onto_target(target[:, :overlap], mirrored)
    return target, int(np.count_nonzero(source)), inserted


def _fold_horizontal(pattern: np.ndarray, axis: int, direction: int) -> tuple[np.ndarray, int, int]:
    height, width = pattern.shape
    assert 0 < axis < height - 1, "horizontal fold axis must have cells on both sides"

    if direction == BLUE:
        target = pattern[:axis, :].copy()
        source = pattern[axis + 1 :, :]
        overlap = min(axis, height - axis - 1)
        inserted = 0
        if overlap > 0:
            mirrored = source[:overlap, :][::-1, :]
            inserted = _merge_source_onto_target(target[axis - overlap : axis, :], mirrored)
        return target, int(np.count_nonzero(source)), inserted

    assert direction == GREEN, "horizontal fold marker must be blue or green"
    target = pattern[axis + 1 :, :].copy()
    source = pattern[:axis, :]
    overlap = min(axis, height - axis - 1)
    inserted = 0
    if overlap > 0:
        mirrored = source[axis - overlap : axis, :][::-1, :]
        inserted = _merge_source_onto_target(target[:overlap, :], mirrored)
    return target, int(np.count_nonzero(source)), inserted


def _apply_instruction_sequence(
    pattern: np.ndarray,
    instructions: list[tuple[int, int]],
    vertical: bool,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    current = np.asarray(pattern, dtype=int)
    current_positions = [int(pos) for pos, _direction in instructions]
    current_length = current.shape[1] if vertical else current.shape[0]
    activity: list[tuple[int, int]] = []

    for idx, (_orig_pos, direction) in enumerate(instructions):
        axis = current_positions[idx]
        if not 0 < axis < current_length - 1:
            raise AssertionError("fold axis became invalid after earlier folds")
        old_length = current_length

        if vertical:
            current, source_pixels, inserted_pixels = _fold_vertical(current, axis, direction)
            current_length = current.shape[1]
        else:
            current, source_pixels, inserted_pixels = _fold_horizontal(current, axis, direction)
            current_length = current.shape[0]
        activity.append((source_pixels, inserted_pixels))

        for future_idx in range(idx + 1, len(instructions)):
            transformed = _transform_position(
                current_positions[future_idx],
                axis,
                direction,
                old_length,
            )
            if transformed is None:
                raise AssertionError("later fold instruction was folded out of range")
            current_positions[future_idx] = transformed

        if current_length < 1:
            raise AssertionError("fold sequence collapsed the pattern")

    return current, activity


def _parse_instructions(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3, "expected a grid with a marker border"
    assert int(g[0, 0]) == BACKGROUND, "top-left marker corner should remain background"

    top = g[0, 1:]
    left = g[1:, 0]
    assert np.all(np.isin(top, (BACKGROUND, BLUE, GREEN))), "top marker row may only contain blue/green markers"
    assert np.all(np.isin(left, (BACKGROUND, BLUE, GREEN))), "left marker column may only contain blue/green markers"

    vertical = [(int(c), int(top[c])) for c in np.flatnonzero(np.isin(top, MARKER_COLORS))]
    horizontal = [(int(r), int(left[r])) for r in np.flatnonzero(np.isin(left, MARKER_COLORS))]
    assert vertical or horizontal, "expected at least one fold instruction"
    return vertical, horizontal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the central pattern using blue/green markers on the top row and left column.

    The marker border itself is not part of the foldable canvas. Top-row markers define
    vertical folds and are executed left-to-right; left-column markers define horizontal
    folds and are executed top-to-bottom. Blue folds the far side back toward the origin
    (right onto left or bottom onto top). Green folds the near side away from the origin
    (left onto right or top onto bottom). When mirrored pixels land on occupied cells,
    the existing target color wins.
    """
    g = np.asarray(grid, dtype=int)
    vertical, horizontal = _parse_instructions(g)
    pattern = g[1:, 1:].copy()
    assert np.any(pattern != BACKGROUND), "central pattern must contain foreground pixels"

    pattern, _vertical_activity = _apply_instruction_sequence(pattern, vertical, vertical=True)
    pattern, _horizontal_activity = _apply_instruction_sequence(pattern, horizontal, vertical=False)
    return pattern


def _random_blob_shape(height: int, width: int, area: int) -> np.ndarray:
    cells = {(random.randrange(height), random.randrange(width))}
    frontier = set(cells)
    while len(cells) < area:
        r, c = random.choice(tuple(frontier))
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier.remove((r, c))
            if not frontier:
                break
            continue
        choice = random.choice(neighbors)
        cells.add(choice)
        frontier.add(choice)
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_shape() -> np.ndarray:
    kind = random.choice(("rect", "line", "l", "blob"))
    if kind == "rect":
        return np.ones((random.randint(1, 3), random.randint(1, 4)), dtype=bool)
    if kind == "line":
        if random.random() < 0.5:
            return np.ones((1, random.randint(2, 5)), dtype=bool)
        return np.ones((random.randint(2, 5), 1), dtype=bool)
    if kind == "l":
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        mask = np.zeros((height, width), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask

    height = random.randint(2, 4)
    width = random.randint(2, 4)
    area = random.randint(2, min(6, height * width))
    return _random_blob_shape(height, width, area)


def _can_place(mask: np.ndarray, occupied: np.ndarray, top: int, left: int) -> bool:
    height, width = mask.shape
    row0 = max(0, top - 1)
    row1 = min(occupied.shape[0], top + height + 1)
    col0 = max(0, left - 1)
    col1 = min(occupied.shape[1], left + width + 1)
    return not bool(np.any(occupied[row0:row1, col0:col1]))


def _build_pattern(height: int, width: int) -> np.ndarray | None:
    palette_size = random.randint(*SHAPE_PALETTE_SIZE_RANGE)
    palette = random.sample(PATTERN_COLORS, k=palette_size)
    object_count = random.randint(*OBJECT_COUNT_RANGE)

    pattern = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    placed = 0

    for _ in range(MAX_PATTERN_BUILD_ATTEMPTS):
        if placed >= object_count:
            break
        shape = _random_shape()
        sh, sw = shape.shape
        if sh > height or sw > width:
            continue
        top = random.randint(0, height - sh)
        left = random.randint(0, width - sw)
        if not _can_place(shape, occupied, top, left):
            continue

        color = random.choice(palette)
        region = pattern[top : top + sh, left : left + sw]
        region[shape] = color
        occupied[top : top + sh, left : left + sw] |= shape
        placed += 1

    colors_used = {int(v) for v in pattern.flat if int(v) != BACKGROUND}
    if placed < max(4, object_count - 2) or len(colors_used) < 2:
        return None
    return pattern


def _sample_instructions(length: int, count_range: tuple[int, int]) -> list[tuple[int, int]] | None:
    max_count = min(count_range[1], max(1, length - 2))
    if max_count < count_range[0]:
        return None

    for _ in range(MAX_INSTRUCTION_ATTEMPTS):
        count = random.randint(count_range[0], max_count)
        positions = sorted(random.sample(range(1, length - 1), k=count))
        directions = [random.choice(MARKER_COLORS) for _ in range(count)]
        if count >= 2 and random.random() < 0.6 and len(set(directions)) == 1:
            flip_idx = random.randrange(count)
            directions[flip_idx] = BLUE if directions[flip_idx] == GREEN else GREEN
        instructions = list(zip(positions, directions))
        simulated = _simulate_instruction_positions(length, instructions)
        if simulated is None:
            continue
        _axes, final_length = simulated
        if final_length >= MIN_FINAL_SIDE:
            return instructions
    return None


def _is_effective(activity: list[tuple[int, int]]) -> bool:
    return all(source_pixels > 0 and inserted_pixels > 0 for source_pixels, inserted_pixels in activity)


def _build_input_grid(
    pattern: np.ndarray,
    vertical: list[tuple[int, int]],
    horizontal: list[tuple[int, int]],
) -> np.ndarray:
    height, width = pattern.shape
    grid = np.zeros((height + 1, width + 1), dtype=int)
    grid[1:, 1:] = pattern
    for axis, color in vertical:
        grid[0, axis + 1] = color
    for axis, color in horizontal:
        grid[axis + 1, 0] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*PATTERN_HEIGHT_RANGE)
    width = random.randint(*PATTERN_WIDTH_RANGE)

    vertical = _sample_instructions(width, VERTICAL_FOLD_RANGE)
    horizontal = _sample_instructions(height, HORIZONTAL_FOLD_RANGE)
    if vertical is None or horizontal is None:
        return None

    pattern = _build_pattern(height, width)
    if pattern is None:
        return None

    after_vertical, vertical_activity = _apply_instruction_sequence(pattern, vertical, vertical=True)
    after_all, horizontal_activity = _apply_instruction_sequence(after_vertical, horizontal, vertical=False)
    if not _is_effective(vertical_activity) or not _is_effective(horizontal_activity):
        return None
    if min(after_all.shape) < MIN_FINAL_SIDE or not np.any(after_all != BACKGROUND):
        return None

    grid = _build_input_grid(pattern, vertical, horizontal)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(solved, after_all) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding example")
