from __future__ import annotations

import random

import numpy as np


# ARC color ids used as visible motion instructions.
BLUE = 1
RED = 2
ORANGE = 7
MAROON = 9

OFFSET_BY_COLOR: dict[int, tuple[int, int]] = {
    ORANGE: (-2, 0),
    RED: (0, -2),
    MAROON: (2, 0),
    BLUE: (0, 1),
}

INSTRUCTION_COLORS = tuple(OFFSET_BY_COLOR)
BACKGROUND_CHOICES = (0, 3, 4, 5, 6, 8)

MIN_GRID_SIZE = 12
MAX_GRID_SIZE = 18
MIN_PIXELS_PER_COLOR = 1
MAX_PIXELS_PER_COLOR = 4
MIN_SOURCE_CHEB_GAP = 2
MIN_DEST_CHEB_GAP = 2
GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    best = int(np.max(counts))
    winners = [int(v) for v, n in zip(values, counts) if int(n) == best]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return winners[0]


def _shifted_cell(shape: tuple[int, int], r: int, c: int, dr: int, dc: int) -> tuple[int, int]:
    rr = int(r + dr)
    cc = int(c + dc)
    h, w = shape
    assert 0 <= rr < h and 0 <= cc < w, "shifted pixel must stay in bounds"
    return rr, cc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored marker pixel by a fixed per-color offset and draw only the shifted pixels.

    Input structure
    - The background is the dominant color.
    - The foreground is a sparse set of colored singleton pixels.
    - Each foreground color is a visible motion instruction with a fixed learned offset.

    Rule
    - Clear the canvas to the background.
    - Shift every foreground pixel by the fixed offset associated with its color.
    - Draw the shifted pixel in the same color.

    Key insight
    - Color specifies motion; the source position matters only as the anchor for the translated landing cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    out = np.full_like(g, bg)

    for r, c in np.argwhere(g != bg):
        color = int(g[r, c])
        assert color in OFFSET_BY_COLOR, f"unexpected instruction color: {color}"
        dr, dc = OFFSET_BY_COLOR[color]
        rr, cc = _shifted_cell(g.shape, int(r), int(c), dr, dc)
        prior = int(out[rr, cc])
        assert prior in (bg, color), "different translated colors cannot collide"
        out[rr, cc] = color

    return out


def _chebyshev_far_enough(cell: tuple[int, int], others: set[tuple[int, int]], min_gap: int) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) >= min_gap for rr, cc in others)


def _candidate_sources(
    shape: tuple[int, int],
    dr: int,
    dc: int,
    src_cells: set[tuple[int, int]],
    dst_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    row_start = max(0, -dr)
    row_stop = h - max(0, dr)
    col_start = max(0, -dc)
    col_stop = w - max(0, dc)

    candidates: list[tuple[int, int]] = []
    for r in range(row_start, row_stop):
        for c in range(col_start, col_stop):
            src = (r, c)
            dst = (r + dr, c + dc)
            if src in src_cells or src in dst_cells or dst in src_cells or dst in dst_cells:
                continue
            if not _chebyshev_far_enough(src, src_cells, MIN_SOURCE_CHEB_GAP):
                continue
            if not _chebyshev_far_enough(dst, dst_cells, MIN_DEST_CHEB_GAP):
                continue
            candidates.append(src)
    random.shuffle(candidates)
    return candidates


def _counts_for_shape(h: int, w: int) -> dict[int, int]:
    area = h * w
    max_per_color = 4 if area >= 180 else 3
    return {
        color: random.randint(MIN_PIXELS_PER_COLOR, max_per_color)
        for color in INSTRUCTION_COLORS
    }


def _validate_generated(inp: np.ndarray) -> None:
    out = main(inp)
    bg = _dominant_color(inp)
    in_colors = [int(v) for v in inp.ravel() if int(v) != bg]
    out_colors = [int(v) for v in out.ravel() if int(v) != bg]
    assert len(in_colors) == len(out_colors) >= len(INSTRUCTION_COLORS)
    assert sorted(in_colors) == sorted(out_colors)
    assert not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        bg = random.choice(BACKGROUND_CHOICES)
        grid = np.full((h, w), bg, dtype=int)

        src_cells: set[tuple[int, int]] = set()
        dst_cells: set[tuple[int, int]] = set()
        counts = _counts_for_shape(h, w)
        colors = list(INSTRUCTION_COLORS)
        random.shuffle(colors)

        ok = True
        for color in colors:
            dr, dc = OFFSET_BY_COLOR[color]
            for _ in range(counts[color]):
                candidates = _candidate_sources((h, w), dr, dc, src_cells, dst_cells)
                if not candidates:
                    ok = False
                    break
                r, c = random.choice(candidates)
                rr, cc = r + dr, c + dc
                grid[r, c] = color
                src_cells.add((r, c))
                dst_cells.add((rr, cc))
            if not ok:
                break

        if not ok:
            continue

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
