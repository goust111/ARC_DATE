import random
from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PINK = 6
MAROON = 9

SQUARE_SIZE = 3
SQUARE_COUNT = 4
SQUARE_GAP = 2
STACK_WIDTH = 3

_SEEN_OUTPUT_SIGNATURES: set[tuple[int, ...]] = set()


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_gray_squares(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    squares: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] != GREY:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == GREY:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if len(coords) == 9 and r1 - r0 == 2 and c1 - c0 == 2:
                if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
                    squares.append((r0, c0, r1, c1))

    squares.sort(key=lambda box: box[1])
    return squares


def _pattern_from_square(grid: np.ndarray, square: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, _r1, _c1 = square
    pattern = np.full((SQUARE_SIZE, SQUARE_SIZE), GREY, dtype=int)
    found_marker = False

    for local_c in range(SQUARE_SIZE):
        top_r, top_c = r0 - 1, c0 + local_c
        if top_r >= 0 and grid[top_r, top_c] in (PINK, MAROON):
            found_marker = True
            marker_color = int(grid[top_r, top_c])
            target_r = 0 if marker_color == MAROON else 1
            pattern[target_r, local_c] = YELLOW

        bottom_r, bottom_c = r0 + SQUARE_SIZE, c0 + local_c
        if bottom_r < grid.shape[0] and grid[bottom_r, bottom_c] in (PINK, MAROON):
            found_marker = True
            marker_color = int(grid[bottom_r, bottom_c])
            target_r = 2 if marker_color == MAROON else 1
            pattern[target_r, local_c] = YELLOW

    if not found_marker:
        raise ValueError("each square must have at least one adjacent marker")
    return pattern


def _merge_rows(row_a: np.ndarray, row_b: np.ndarray) -> np.ndarray:
    merged = np.full_like(row_a, GREY)
    mask_a = row_a != GREY
    mask_b = row_b != GREY
    merged[mask_a] = row_a[mask_a]
    merged[~mask_a & mask_b] = row_b[~mask_a & mask_b]
    return merged


def _stack_patterns(patterns: list[np.ndarray]) -> np.ndarray:
    stacked = patterns[0].copy()
    for idx, pattern in enumerate(patterns[1:], start=1):
        next_pattern = np.flipud(pattern) if idx % 2 == 1 else pattern
        overlap = _merge_rows(stacked[-1], next_pattern[0])
        stacked = np.vstack([stacked[:-1], overlap[np.newaxis, :], next_pattern[1:]])
    return stacked


def main(grid: np.ndarray) -> np.ndarray:
    """Project top/bottom markers into four gray 3x3 squares, then stack the resulting square patterns with alternating vertical flips.

    Input structure:
    - The grid has a black background and exactly four solid gray 3x3 squares arranged left-to-right.
    - Each square has one or two magenta or maroon marker pixels on the cell directly above or below one of its columns.
    - Markers are outside the squares, 4-neighbor adjacent to the touched edge, and different markers of one square use different columns.

    Rule:
    - For each square, inspect the marker pixels touching its top or bottom edge.
    - A maroon marker projects a yellow cell onto the near-edge row of the square in the same column, while a magenta marker projects onto the middle row.
    - Keep all other square cells gray, discard the original markers, then stack the four 3x3 results from left to right into a 3-column output.
    - Before stacking, vertically flip the patterns at odd indices 1 and 3; adjacent patterns overlap by one row, and each overlap cell keeps yellow if either side is yellow, otherwise gray.

    Key insight:
    - The visible output is controlled by three coupled cues: marker side chooses the projection edge, marker color chooses projection depth, and square order chooses where flipped local patterns land in the final overlapped stack.
    """
    grid = np.asarray(grid, dtype=int)
    squares = _find_gray_squares(grid)
    if len(squares) != SQUARE_COUNT:
        raise ValueError("expected exactly four gray 3x3 squares")

    patterns = [_pattern_from_square(grid, square) for square in squares]
    return _stack_patterns(patterns)


def _pattern_signature(patterns: list[list[tuple[str, int, int]]]) -> tuple[tuple[tuple[str, int, int], ...], ...]:
    return tuple(tuple(sorted(block)) for block in patterns)


def _output_signature(grid: np.ndarray) -> tuple[int, ...]:
    out = main(grid)
    return tuple(int(x) for x in out.reshape(-1))


def _logical_layout_hash(patterns: list[list[tuple[str, int, int]]]) -> int:
    score = 0
    for block_idx, markers in enumerate(patterns):
        for marker_idx, (side, color, col) in enumerate(sorted(markers)):
            side_code = 0 if side == "top" else 1
            color_code = 0 if color == PINK else 1
            score = score * 19 + (block_idx + 1) * 5 + marker_idx * 3 + side_code * 7 + color_code * 11 + col
    return score


def _has_overlap_collision(patterns: list[list[tuple[str, int, int]]]) -> bool:
    relevant_pairs = [
        (0, 1, "bottom"),
        (1, 2, "top"),
        (2, 3, "bottom"),
    ]
    for left_idx, right_idx, side in relevant_pairs:
        left_cols = {col for marker_side, color, col in patterns[left_idx] if marker_side == side and color == MAROON}
        right_cols = {col for marker_side, color, col in patterns[right_idx] if marker_side == side and color == MAROON}
        if left_cols & right_cols:
            return True
    return False


def _valid_patterns(patterns: list[list[tuple[str, int, int]]]) -> bool:
    flat = [(block_idx, side, color, col) for block_idx, block in enumerate(patterns) for side, color, col in block]
    colors = {color for _block_idx, _side, color, _col in flat}
    sides = {side for _block_idx, side, _color, _col in flat}
    counts = {len(block) for block in patterns}

    if colors != {PINK, MAROON}:
        return False
    if sides != {"top", "bottom"}:
        return False
    if counts == {1} or counts == {2}:
        return False
    if not any(block_idx in (1, 3) and color == MAROON for block_idx, _side, color, _col in flat):
        return False
    if _has_overlap_collision(patterns):
        return False

    projected = []
    for block in patterns:
        local = np.full((3, 3), GREY, dtype=int)
        for side, color, col in block:
            row = 1 if color == PINK else (0 if side == "top" else 2)
            local[row, col] = YELLOW
        projected.append(tuple(int(x) for x in local.reshape(-1)))
    if len(set(projected)) < 3:
        return False
    return True


def _make_marker_patterns(rng: random.Random) -> list[list[tuple[str, int, int]]]:
    while True:
        patterns: list[list[tuple[str, int, int]]] = []
        counts = [rng.randint(1, 2) for _ in range(SQUARE_COUNT)]
        if len(set(counts)) == 1:
            counts[rng.randrange(SQUARE_COUNT)] = 3 - counts[0]

        for count in counts:
            cols = rng.sample([0, 1, 2], count)
            block = []
            for col in cols:
                side = rng.choice(["top", "bottom"])
                color = rng.choice([PINK, MAROON])
                block.append((side, color, col))
            patterns.append(block)

        if _valid_patterns(patterns):
            return patterns


def _render_input_from_patterns(patterns: list[list[tuple[str, int, int]]]) -> np.ndarray:
    base_w = SQUARE_COUNT * SQUARE_SIZE + (SQUARE_COUNT - 1) * SQUARE_GAP
    layout_hash = _logical_layout_hash(patterns)
    height = 5 + layout_hash % 4
    width = base_w + 1 + (layout_hash // 4) % 8
    left_margin = (layout_hash // 32) % (width - base_w + 1)
    square_row = 1 + (layout_hash // 256) % (height - 4)

    grid = np.zeros((height, width), dtype=int)
    for idx in range(SQUARE_COUNT):
        c0 = left_margin + idx * (SQUARE_SIZE + SQUARE_GAP)
        grid[square_row : square_row + SQUARE_SIZE, c0 : c0 + SQUARE_SIZE] = GREY
        for side, color, local_c in patterns[idx]:
            marker_r = square_row - 1 if side == "top" else square_row + SQUARE_SIZE
            marker_c = c0 + local_c
            grid[marker_r, marker_c] = color
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        patterns = _make_marker_patterns(rng)
        grid = _render_input_from_patterns(patterns)
        out_sig = _output_signature(grid)
        if out_sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        if grid.ndim != 2 or max(grid.shape) > 30:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(out_sig)
        return grid
    raise RuntimeError("failed to generate a fresh valid input")
