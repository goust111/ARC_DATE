from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
CELL = 3
SYMBOL_COLORS = (2, 3, 4, 6, 7)
SIDES = ("top", "right", "bottom", "left")

MIN_COARSE = 4
MAX_COARSE = 6
MIN_KEEP_COLORS = 3
MAX_KEEP_COLORS = 4
MIN_EMPTY_CELLS = 2
MAX_GENERATION_ATTEMPTS = 500

RING_PATCHES = {
    color: np.array(
        (
            (color, color, color),
            (color, BG, color),
            (color, color, color),
        ),
        dtype=int,
    )
    for color in SYMBOL_COLORS
}


def _uniform_edge_candidates(grid: np.ndarray, bg: int = BG) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = []
    edges = (
        ("top", grid[0, :]),
        ("right", grid[:, -1]),
        ("bottom", grid[-1, :]),
        ("left", grid[:, 0]),
    )
    for side, strip in edges:
        values = np.unique(strip)
        if len(values) == 1 and int(values[0]) != bg:
            candidates.append((side, int(values[0])))
    return candidates


def _marker_side_and_color(grid: np.ndarray, bg: int = BG) -> tuple[str, int]:
    candidates = _uniform_edge_candidates(grid, bg)
    assert len(candidates) == 1, f"expected exactly one full-edge marker line, got {candidates}"
    return candidates[0]


def _symbol_region(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        region = grid[1:, :]
    elif side == "right":
        region = grid[:, :-1]
    elif side == "bottom":
        region = grid[:-1, :]
    elif side == "left":
        region = grid[:, 1:]
    else:
        raise AssertionError(side)

    assert region.shape[0] % CELL == 0 and region.shape[1] % CELL == 0, (
        f"symbol region must decompose into 3x3 cells, got {region.shape}"
    )
    return region


def _extract_symbolic_grid(grid: np.ndarray, bg: int = BG) -> tuple[np.ndarray, str, int]:
    side, marker_color = _marker_side_and_color(grid, bg)
    region = _symbol_region(grid, side)
    rows = region.shape[0] // CELL
    cols = region.shape[1] // CELL
    symbolic = np.full((rows, cols), bg, dtype=int)

    for r in range(rows):
        for c in range(cols):
            patch = region[r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL]
            colors = [int(v) for v in np.unique(patch) if int(v) != bg]
            if not colors:
                continue
            assert len(colors) == 1, f"mixed-color symbol patch at {(r, c)}"
            color = colors[0]
            assert color in RING_PATCHES, f"unexpected symbol color {color}"
            assert np.array_equal(patch, RING_PATCHES[color]), f"invalid hollow symbol at {(r, c)}"
            symbolic[r, c] = color

    return symbolic, side, marker_color


def _ordered_kept_counts(symbolic: np.ndarray, distractor: int, bg: int = BG) -> list[tuple[int, int]]:
    counts = Counter(
        int(v)
        for v in symbolic.ravel()
        if int(v) != bg and int(v) != distractor
    )
    assert len(counts) >= 3, "expected at least three non-distractor colors"
    values = list(counts.values())
    assert len(values) == len(set(values)), "non-distractor frequencies must be unique"
    return sorted(counts.items(), key=lambda item: item[1])


def _render_histogram(ordered: list[tuple[int, int]], side: str, bg: int = BG) -> np.ndarray:
    n_colors = len(ordered)
    max_count = max(count for _color, count in ordered)

    if side in ("top", "bottom"):
        out = np.full((n_colors, max_count), bg, dtype=int)
        for i, (color, count) in enumerate(ordered):
            row = i if side == "top" else n_colors - 1 - i
            if side == "top":
                out[row, :count] = color
            else:
                out[row, max_count - count :] = color
    else:
        out = np.full((max_count, n_colors), bg, dtype=int)
        for i, (color, count) in enumerate(ordered):
            col = n_colors - 1 - i if side == "right" else i
            if side == "right":
                out[:count, col] = color
            else:
                out[max_count - count :, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress hollow 3x3 symbols, ignore the marker-colored distractors, and summarize the remaining colors as a directed histogram.

    Input structure:
    - The grid has black background, one complete edge marker line, and a coarse 3x3-cell lattice of monochrome hollow symbols.
    - Each occupied coarse cell is exactly one 3x3 ring whose center is black; empty coarse cells are fully black.
    - The marker line's color matches at least one symbol color.

    Rule:
    - Find the unique full-edge marker line and read both its side and its color.
    - Treat every hollow symbol of that same color as a distractor and ignore it; compress every remaining hollow symbol to one pixel at its coarse-cell position.
    - Count the remaining symbols by color, sort colors by increasing frequency, and draw a histogram whose bar direction is determined by the marker side: top -> bars grow left-to-right, right -> top-to-bottom, bottom -> right-to-left, left -> bottom-to-top.

    Key insight:
    - The single edge marker encodes two independent controls at once: its color tells which symbols are distractors, while its position tells how the frequency histogram must be oriented.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    symbolic, side, distractor = _extract_symbolic_grid(g, BG)
    ordered = _ordered_kept_counts(symbolic, distractor, BG)
    return _render_histogram(ordered, side, BG)


def _sample_color_counts(capacity: int, n_keep: int) -> tuple[list[int], int]:
    max_count_cap = min(8, capacity - MIN_EMPTY_CELLS - 1)
    assert max_count_cap >= n_keep
    for _ in range(200):
        keep_counts = random.sample(range(1, max_count_cap + 1), n_keep)
        total_keep = sum(keep_counts)
        remaining = capacity - MIN_EMPTY_CELLS - total_keep
        if remaining < 1:
            continue
        distractor_count = random.randint(1, min(7, remaining))
        return keep_counts, distractor_count
    raise AssertionError("failed to sample valid color counts")


def _build_symbolic_case() -> tuple[np.ndarray, str, int, list[tuple[int, int]]]:
    rows = random.randint(MIN_COARSE, MAX_COARSE)
    cols = random.randint(MIN_COARSE, MAX_COARSE)
    side = random.choice(SIDES)

    distractor = random.choice(SYMBOL_COLORS)
    n_keep = random.randint(MIN_KEEP_COLORS, min(MAX_KEEP_COLORS, len(SYMBOL_COLORS) - 1))
    keep_colors = random.sample([c for c in SYMBOL_COLORS if c != distractor], n_keep)
    keep_counts, distractor_count = _sample_color_counts(rows * cols, n_keep)

    occupied = sum(keep_counts) + distractor_count
    assert occupied <= rows * cols - MIN_EMPTY_CELLS

    symbolic = np.full((rows, cols), BG, dtype=int)
    coords = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(coords)
    cursor = 0

    distractor_cells = coords[cursor : cursor + distractor_count]
    cursor += distractor_count
    for r, c in distractor_cells:
        symbolic[r, c] = distractor

    ordered_pairs = list(zip(keep_colors, keep_counts))
    random.shuffle(ordered_pairs)
    for color, count in ordered_pairs:
        for r, c in coords[cursor : cursor + count]:
            symbolic[r, c] = color
        cursor += count

    expected = sorted(((color, count) for color, count in ordered_pairs), key=lambda item: item[1])
    return symbolic, side, distractor, expected


def _expand_symbolic(symbolic: np.ndarray, side: str, marker_color: int) -> np.ndarray:
    rows, cols = symbolic.shape
    height = rows * CELL + (1 if side in ("top", "bottom") else 0)
    width = cols * CELL + (1 if side in ("left", "right") else 0)
    grid = np.full((height, width), BG, dtype=int)

    row_offset = 1 if side == "top" else 0
    col_offset = 1 if side == "left" else 0

    if side == "top":
        grid[0, :] = marker_color
    elif side == "right":
        grid[:, -1] = marker_color
    elif side == "bottom":
        grid[-1, :] = marker_color
    else:
        grid[:, 0] = marker_color

    for r in range(rows):
        for c in range(cols):
            color = int(symbolic[r, c])
            if color == BG:
                continue
            rr = row_offset + r * CELL
            cc = col_offset + c * CELL
            grid[rr : rr + CELL, cc : cc + CELL] = RING_PATCHES[color]
    return grid


def _validate_case(
    grid: np.ndarray,
    symbolic: np.ndarray,
    side: str,
    distractor: int,
    expected_ordered: list[tuple[int, int]],
) -> None:
    detected_symbolic, detected_side, detected_distractor = _extract_symbolic_grid(grid, BG)
    assert detected_side == side
    assert detected_distractor == distractor
    assert np.array_equal(detected_symbolic, symbolic)

    assert len(_uniform_edge_candidates(grid, BG)) == 1
    assert np.count_nonzero(symbolic == BG) >= MIN_EMPTY_CELLS
    assert np.count_nonzero(symbolic == distractor) >= 1

    expected = _render_histogram(expected_ordered, side, BG)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    for alt_side in SIDES:
        if alt_side == side:
            continue
        alt = _render_histogram(expected_ordered, alt_side, BG)
        assert not np.array_equal(expected, alt), (
            f"direction cue is degenerate for {side} vs {alt_side}"
        )

    wrong_counts = Counter(int(v) for v in symbolic.ravel() if int(v) != BG)
    wrong_ordered = sorted(wrong_counts.items(), key=lambda item: (item[1], item[0]))
    wrong = _render_histogram(wrong_ordered, side, BG)
    assert not np.array_equal(expected, wrong), "distractor filtering must matter"

    recovered = _ordered_kept_counts(symbolic, distractor, BG)
    assert recovered == expected_ordered


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        symbolic, side, distractor, expected_ordered = _build_symbolic_case()
        grid = _expand_symbolic(symbolic, side, distractor)
        try:
            _validate_case(grid, symbolic, side, distractor, expected_ordered)
        except AssertionError:
            continue
        return grid
    raise AssertionError("failed to generate a valid input")
