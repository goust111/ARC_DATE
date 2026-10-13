from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GRAY = 5

BLOCK = 4
INNER = slice(1, 3)
FRAME_MASK = np.ones((BLOCK, BLOCK), dtype=bool)
FRAME_MASK[INNER, INNER] = False

# A 2x2 "cross" approximation: full right column + full bottom row.
CROSS_STAMP = np.array([[BLACK, RED], [RED, RED]], dtype=int)

IRREGULAR_HOLES = (
    np.array([[BLACK, BLACK], [BLACK, GRAY]], dtype=int),
    np.array([[BLACK, BLACK], [GRAY, BLACK]], dtype=int),
    np.array([[BLACK, GRAY], [BLACK, BLACK]], dtype=int),
    np.array([[GRAY, BLACK], [BLACK, BLACK]], dtype=int),
    np.array([[BLACK, BLACK], [GRAY, GRAY]], dtype=int),
    np.array([[GRAY, GRAY], [BLACK, BLACK]], dtype=int),
    np.array([[BLACK, GRAY], [BLACK, GRAY]], dtype=int),
    np.array([[GRAY, BLACK], [GRAY, BLACK]], dtype=int),
)

GEN_ATTEMPTS = 400


def _frame_with_inner(inner: np.ndarray) -> np.ndarray:
    frame = np.full((BLOCK, BLOCK), GRAY, dtype=int)
    frame[INNER, INNER] = np.asarray(inner, dtype=int)
    return frame


def _sample_spaced_starts(low: int, high: int, count: int, gap: int = 5) -> list[int] | None:
    starts = list(range(low, high + 1))
    random.shuffle(starts)
    chosen: list[int] = []
    for start in starts:
        if all(abs(start - prev) >= gap for prev in chosen):
            chosen.append(start)
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _overlaps(occupied: np.ndarray, r0: int, c0: int) -> bool:
    return bool(occupied[r0 : r0 + BLOCK, c0 : c0 + BLOCK].any())


def _place_frame(grid: np.ndarray, occupied: np.ndarray, r0: int, c0: int, inner: np.ndarray) -> None:
    grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = _frame_with_inner(inner)
    occupied[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = True


def _expected_output(grid: np.ndarray, row_starts: list[int], col_starts: list[int]) -> np.ndarray:
    out = grid.copy()
    for r0 in row_starts:
        for c0 in col_starts:
            inner = out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
            if np.all(inner == BLACK):
                out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = CROSS_STAMP
            else:
                inner[inner == BLACK] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp selected 4x4 gray frames according to whether their 2x2 hole is a full square.

    Input structure:
    - A black background contains yellow markers on the top row and on the right edge.
    - Each selected row/column pair points to the top-left corner of a 4x4 gray frame.
    - Inside each frame, the 2x2 center is either a full black square hole or a partial black irregular hole.

    Rule:
    - Read yellow markers from the top row as selected columns and from the rightmost column as selected rows.
    - Take the Cartesian product of those marked rows and columns to find the candidate 4x4 frames.
    - For each candidate, inspect only its inner 2x2 cells: a full 2x2 black hole becomes a red cross stamp with one corner left black; otherwise every existing black inner cell becomes blue and gray inner cells stay gray.

    Key insight:
    - The marked rows and columns only choose which frames matter; the fill pattern then depends on the geometry of the chosen frame's own hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid is too small for markers plus 4x4 frames"

    row_starts = [int(r) for r in np.flatnonzero(g[:, w - 1] == YELLOW) if r + BLOCK <= h]
    col_starts = [int(c) for c in np.flatnonzero(g[0, : w - 1] == YELLOW) if c + BLOCK <= w - 1]
    assert row_starts and col_starts, "expected both top-row and right-edge markers"

    out = g.copy()
    for r0 in row_starts:
        for c0 in col_starts:
            window = g[r0 : r0 + BLOCK, c0 : c0 + BLOCK]
            assert window.shape == (BLOCK, BLOCK), "candidate frame runs out of bounds"
            assert np.all(window[FRAME_MASK] == GRAY), "candidate window must be a gray frame"

            inner = window[INNER, INNER]
            if np.all(inner == BLACK):
                out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = CROSS_STAMP
            else:
                patch = out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
                patch[patch == BLACK] = BLUE

    return out


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    row_count = random.randint(2, 3)
    col_count = random.randint(2, 3)

    min_h = max(14, 5 * row_count)
    min_w = max(14, 5 * col_count)
    h = random.randint(min_h, min(20, max(min_h, 20)))
    w = random.randint(min_w, min(20, max(min_w, 20)))

    row_starts = _sample_spaced_starts(1, h - BLOCK, row_count)
    col_starts = _sample_spaced_starts(0, w - BLOCK - 1, col_count)
    if row_starts is None or col_starts is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, col_starts] = YELLOW
    grid[row_starts, w - 1] = YELLOW

    occupied = np.zeros((h, w), dtype=bool)
    candidate_positions = [(r0, c0) for r0 in row_starts for c0 in col_starts]

    square_count = random.randint(1, len(candidate_positions) - 1)
    square_flags = [True] * square_count + [False] * (len(candidate_positions) - square_count)
    random.shuffle(square_flags)

    for (r0, c0), is_square in zip(candidate_positions, square_flags):
        if _overlaps(occupied, r0, c0):
            return None
        inner = np.full((2, 2), BLACK, dtype=int) if is_square else random.choice(IRREGULAR_HOLES)
        _place_frame(grid, occupied, r0, c0, inner)

    distractor_candidates: list[tuple[int, int, int]] = []
    candidate_set = set(candidate_positions)
    for r0 in range(1, h - BLOCK + 1):
        for c0 in range(0, w - BLOCK):
            if (r0, c0) in candidate_set or _overlaps(occupied, r0, c0):
                continue
            priority = 0
            if (r0 in row_starts) ^ (c0 in col_starts):
                priority = 2
            elif r0 in row_starts or c0 in col_starts:
                priority = 1
            distractor_candidates.append((priority, r0, c0))

    distractor_count = random.randint(0, min(3, len(distractor_candidates)))
    if distractor_count:
        random.shuffle(distractor_candidates)
        distractor_candidates.sort(key=lambda item: item[0], reverse=True)
        placed = 0
        for _priority, r0, c0 in distractor_candidates:
            if _overlaps(occupied, r0, c0):
                continue
            if random.random() < 0.35 and placed + 1 < distractor_count:
                continue
            inner = np.full((2, 2), BLACK, dtype=int) if random.random() < 0.4 else random.choice(IRREGULAR_HOLES)
            _place_frame(grid, occupied, r0, c0, inner)
            placed += 1
            if placed == distractor_count:
                break

    expected = _expected_output(grid, row_starts, col_starts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
