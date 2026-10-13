from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 7
MARKER_COLOR = 2
FRAME_SIZE = 7
INNER_SIZE = 5
BLOCK_H = 13  # source 5 + gap 1 + frame 7
BLOCK_W = 7
SOURCE_COLORS = (1, 3, 4, 5, 6, 8, 9)
MAX_GEN_ATTEMPTS = 400
PATTERN_RETRIES = 200

INSTRUCTIONS = ("tl", "tr", "bl", "br")
MARKER_OFFSETS = {
    "tl": (0, 0),
    "tr": (0, 3),
    "bl": (3, 0),
    "br": (3, 3),
}


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_border_mask() -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


FRAME_BORDER_MASK = _frame_border_mask()


def _apply_transform(patch: np.ndarray, instruction: str) -> np.ndarray:
    if instruction == "tl":
        return patch.copy()
    if instruction == "tr":
        return np.fliplr(patch)
    if instruction == "bl":
        return np.flipud(patch)
    if instruction == "br":
        return np.rot90(patch, 2)
    raise AssertionError(f"unknown instruction: {instruction}")


def _marker_instruction(inner: np.ndarray) -> str:
    assert inner.shape == (INNER_SIZE, INNER_SIZE)
    marker_mask = inner == MARKER_COLOR
    matches: list[str] = []
    for instruction, (r0, c0) in MARKER_OFFSETS.items():
        expected = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        expected[r0 : r0 + 2, c0 : c0 + 2] = True
        if np.array_equal(marker_mask, expected):
            matches.append(instruction)
    assert len(matches) == 1, "frame interior must contain exactly one 2x2 corner marker"
    return matches[0]


def _looks_like_frame(grid: np.ndarray, top: int, left: int, bg: int) -> bool:
    patch = grid[top : top + FRAME_SIZE, left : left + FRAME_SIZE]
    if patch.shape != (FRAME_SIZE, FRAME_SIZE):
        return False
    if not np.all(patch[FRAME_BORDER_MASK] == FRAME_COLOR):
        return False
    inner = patch[1:-1, 1:-1]
    allowed = (inner == bg) | (inner == MARKER_COLOR)
    if not np.all(allowed):
        return False
    try:
        _marker_instruction(inner)
    except AssertionError:
        return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, str]]:
    h, w = grid.shape
    frames: list[tuple[int, int, str]] = []
    for top in range(h - FRAME_SIZE + 1):
        for left in range(w - FRAME_SIZE + 1):
            if not _looks_like_frame(grid, top, left, bg):
                continue
            instruction = _marker_instruction(grid[top + 1 : top + 6, left + 1 : left + 6])
            frames.append((top, left, instruction))
    assert frames, "expected at least one instruction frame"
    return frames


def _neighbors4(r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < INNER_SIZE:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < INNER_SIZE:
        yield r, c + 1


def _random_pattern() -> np.ndarray:
    for _ in range(PATTERN_RETRIES):
        target_area = random.randint(6, 10)
        start = (random.randrange(INNER_SIZE), random.randrange(INNER_SIZE))
        cells = {start}
        frontier = set(_neighbors4(*start))

        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nxt in _neighbors4(*cell):
                if nxt not in cells:
                    frontier.add(nxt)

        if len(cells) != target_area:
            continue

        coords = np.asarray(sorted(cells), dtype=int)
        row_span = int(coords[:, 0].max() - coords[:, 0].min() + 1)
        col_span = int(coords[:, 1].max() - coords[:, 1].min() + 1)
        if row_span < 3 or col_span < 3:
            continue

        color_count = 2 if len(cells) < 8 else random.choice((2, 3))
        colors = random.sample(SOURCE_COLORS, color_count)
        ordered = list(map(tuple, coords))
        random.shuffle(ordered)

        pattern = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
        for idx, (r, c) in enumerate(ordered):
            if idx < color_count:
                pattern[r, c] = colors[idx]
            else:
                pattern[r, c] = random.choice(colors)

        if np.array_equal(pattern, np.fliplr(pattern)):
            continue
        if np.array_equal(pattern, np.flipud(pattern)):
            continue
        if np.array_equal(pattern, np.rot90(pattern, 2)):
            continue
        return pattern

    raise RuntimeError("failed to generate a non-symmetric 5x5 source pattern")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each frame's corner marker, transform the 5x5 source above it, and place the result inside the frame.

    Input structure:
    - The background is black, and the grid contains one or more orange 7x7 hollow square frames with 5x5 interiors.
    - Inside each frame, exactly one 2x2 red block sits in one interior corner and acts as an instruction marker.
    - Directly above each frame, separated by one background row, there is a 5x5 source pattern drawn with non-black, non-orange, non-red colors.

    Rule:
    - Decode the marker corner as one of four transforms on the source patch: top-left = unchanged, top-right = horizontal flip, bottom-left = vertical flip, bottom-right = 180-degree rotation.
    - Apply that transform to the full 5x5 source patch, including its black cells.
    - Replace the frame interior with the transformed patch, remove the red marker, and clear the 5x5 source area above the frame back to background.
    - Leave every orange frame border unchanged.

    Key insight:
    - The red 2x2 block is not content to copy; its corner position is a symbolic instruction that selects which geometric transform to apply to the separate source object above the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _mode_color(g)

    frames = _find_frames(g, bg)
    out = g.copy()

    for top, left, instruction in frames:
        source_top = top - 6
        source_left = left + 1
        assert source_top >= 0, "each frame must have a 5x5 source patch directly above it"

        source = g[source_top : source_top + INNER_SIZE, source_left : source_left + INNER_SIZE]
        assert source.shape == (INNER_SIZE, INNER_SIZE)
        assert np.any(source != bg), "source patch must contain a non-background pattern"
        assert not np.any(source == FRAME_COLOR), "source patch must not use the frame color"
        assert not np.any(source == MARKER_COLOR), "source patch must not use the marker color"

        transformed = _apply_transform(source, instruction)
        out[source_top : source_top + INNER_SIZE, source_left : source_left + INNER_SIZE] = bg
        out[top + 1 : top + 1 + INNER_SIZE, left + 1 : left + 1 + INNER_SIZE] = transformed

    return out


def _row_counts(frame_count: int) -> list[int]:
    if frame_count == 4:
        return list(random.choice(((2, 2), (3, 1), (1, 3))))
    return list(random.choice(((3, 2), (2, 3))))


def _row_width(count: int, gaps: list[int]) -> int:
    if count == 0:
        return 0
    return count * BLOCK_W + sum(gaps)


def _layout(frame_count: int) -> tuple[int, int, list[tuple[int, int]]]:
    row_counts = _row_counts(frame_count)
    row_gap = random.randint(1, 2)
    row_gaps = [[random.randint(1, 3) for _ in range(count - 1)] for count in row_counts]
    widths = [_row_width(count, gaps) for count, gaps in zip(row_counts, row_gaps)]

    base_w = max(widths)
    base_h = len(row_counts) * BLOCK_H + (len(row_counts) - 1) * row_gap
    assert base_w <= 30 and base_h <= 30

    extra_w = 30 - base_w
    extra_h = 30 - base_h
    left_margin = random.randint(0, min(3, extra_w))
    right_margin = random.randint(0, min(3, extra_w - left_margin))
    top_margin = random.randint(0, min(2, extra_h))
    bottom_margin = random.randint(0, min(2, extra_h - top_margin))

    width = base_w + left_margin + right_margin
    height = base_h + top_margin + bottom_margin

    positions: list[tuple[int, int]] = []
    y = top_margin
    for count, gaps, row_width in zip(row_counts, row_gaps, widths):
        slack = base_w - row_width
        row_left = left_margin + random.randint(0, slack)
        x = row_left
        for idx in range(count):
            positions.append((y, x))
            x += BLOCK_W
            if idx + 1 < count:
                x += gaps[idx]
        y += BLOCK_H + row_gap

    random.shuffle(positions)
    return height, width, positions


def _paint_frame(grid: np.ndarray, source_top: int, frame_left: int, pattern: np.ndarray, instruction: str) -> None:
    grid[source_top : source_top + INNER_SIZE, frame_left + 1 : frame_left + 1 + INNER_SIZE] = pattern

    frame_top = source_top + 6
    frame = grid[frame_top : frame_top + FRAME_SIZE, frame_left : frame_left + FRAME_SIZE]
    frame[FRAME_BORDER_MASK] = FRAME_COLOR

    inner_top = frame_top + 1
    inner_left = frame_left + 1
    marker_r, marker_c = MARKER_OFFSETS[instruction]
    grid[inner_top + marker_r : inner_top + marker_r + 2, inner_left + marker_c : inner_left + marker_c + 2] = MARKER_COLOR


def _build_candidate() -> np.ndarray:
    frame_count = random.choice((4, 5))
    height, width, positions = _layout(frame_count)
    grid = np.full((height, width), BG, dtype=int)

    instructions = list(INSTRUCTIONS)
    if frame_count == 5:
        instructions.append(random.choice(INSTRUCTIONS))
    random.shuffle(instructions)

    for (source_top, frame_left), instruction in zip(positions, instructions):
        pattern = _random_pattern()
        _paint_frame(grid, source_top, frame_left, pattern, instruction)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _build_candidate()
            out = main(grid)
            assert grid.ndim == 2 and out.ndim == 2
            assert max(grid.shape) <= 30 and max(out.shape) <= 30
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate valid input")
