from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PALETTE_LEN = 5
PALETTE_POOL = [3, 4, 5, 6, 7, 8, 9]

GEN_RETRIES = 400
ENCLOSED_LAYOUTS = [
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 3),
]
EDGE_BAND_RANGE = (1, 4)
INNER_BAND_RANGE = (3, 5)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _palette_candidates(grid: np.ndarray) -> list[tuple[int, int, list[int]]]:
    h, _ = grid.shape
    rows = [0] if h == 1 else [0, h - 1]
    candidates: list[tuple[int, int, list[int]]] = []
    for r in rows:
        row = grid[r]
        nz = np.flatnonzero(row != BG)
        if nz.size != PALETTE_LEN:
            continue
        start = int(nz[0])
        if not np.array_equal(nz, np.arange(start, start + PALETTE_LEN)):
            continue
        colors = [int(v) for v in row[start : start + PALETTE_LEN]]
        if len(set(colors)) != PALETTE_LEN:
            continue
        if any(color in (BG, BLUE, RED) for color in colors):
            continue
        candidates.append((r, start, colors))
    return candidates


def _detect_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    candidates = _palette_candidates(grid)
    assert len(candidates) == 1, "expected exactly one palette row on the top or bottom edge"
    row, _start, colors = candidates[0]
    return int(row), colors


def _field_view(grid: np.ndarray, palette_row: int) -> tuple[np.ndarray, int]:
    if palette_row == 0:
        return grid[1:, :], 1
    return grid[:-1, :], 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed red-bounded region by indexing into the edge palette with its blue-marker count.

    Input structure:
    - The background is black.
    - One continuous red wall network runs through the main play area and partitions it into regions.
    - Some enclosed regions contain blue marker pixels; the rest of the play area contains only black background.
    - The very top row or very bottom row contains a 5-color ordered palette of adjacent single cells, using colors distinct from black, red, and blue.

    Rule:
    - Detect the palette row and read its colors from left to right as entries 1 through 5.
    - In the remaining rows, treat red pixels as walls and find every 4-connected non-red region.
    - Ignore regions that touch the outer border of the play area, because they are not enclosed.
    - Count the blue markers inside each enclosed region: if the count is 0, leave that region black; otherwise fill the whole region, including the blue markers, with palette color number N.

    Key insight:
    - Only the number of blue markers in each enclosed region matters; their positions do not. The palette is an ordered index, not a color-matching key.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette_row, palette = _detect_palette(g)
    field, row_offset = _field_view(g, palette_row)
    fh, fw = field.shape

    out = g.copy()
    comps = _components(field != RED)
    for comp in comps:
        if any(r in (0, fh - 1) or c in (0, fw - 1) for r, c in comp):
            continue
        blue_count = sum(1 for r, c in comp if int(field[r, c]) == BLUE)
        assert 0 <= blue_count <= PALETTE_LEN
        if blue_count == 0:
            continue
        fill = int(palette[blue_count - 1])
        for r, c in comp:
            out[row_offset + r, c] = fill
    return out


def _make_lines(enclosed_count: int, rng: random.Random) -> tuple[list[int], int]:
    edge_before = rng.randint(*EDGE_BAND_RANGE)
    edge_after = rng.randint(*EDGE_BAND_RANGE)
    inner_widths = [rng.randint(*INNER_BAND_RANGE) for _ in range(enclosed_count)]

    positions = [edge_before]
    cursor = edge_before
    for width in inner_widths:
        cursor += 1 + width
        positions.append(cursor)

    total = edge_before + edge_after + sum(inner_widths) + len(positions)
    return positions, total


def _enclosed_regions(h_lines: list[int], v_lines: list[int]) -> list[tuple[slice, slice]]:
    regions: list[tuple[slice, slice]] = []
    for r0, r1 in zip(h_lines, h_lines[1:]):
        for c0, c1 in zip(v_lines, v_lines[1:]):
            regions.append((slice(r0 + 1, r1), slice(c0 + 1, c1)))
    return regions


def _marker_counts(n_regions: int, rng: random.Random) -> list[int]:
    assert n_regions >= 2
    counts = [0, rng.randint(1, PALETTE_LEN)]
    cycle = list(range(PALETTE_LEN + 1))
    rng.shuffle(cycle)
    while len(counts) < n_regions:
        if not cycle:
            cycle = list(range(PALETTE_LEN + 1))
            rng.shuffle(cycle)
        counts.append(cycle.pop())
    rng.shuffle(counts)
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    enclosed_cols, enclosed_rows = rng.choice(ENCLOSED_LAYOUTS)

    v_lines, width = _make_lines(enclosed_cols, rng)
    h_lines, field_height = _make_lines(enclosed_rows, rng)
    total_height = field_height + 1
    if width > 30 or total_height > 30:
        return None

    palette_top = bool(rng.randrange(2))
    grid = np.full((total_height, width), BG, dtype=int)
    palette_row = 0 if palette_top else total_height - 1
    field, _ = _field_view(grid, palette_row)

    field[h_lines, :] = RED
    field[:, v_lines] = RED

    palette_colors = rng.sample(PALETTE_POOL, PALETTE_LEN)
    palette_start = rng.randint(0, width - PALETTE_LEN)
    grid[palette_row, palette_start : palette_start + PALETTE_LEN] = palette_colors

    regions = _enclosed_regions(h_lines, v_lines)
    counts = _marker_counts(len(regions), rng)
    for (row_slice, col_slice), count in zip(regions, counts):
        if count == 0:
            continue
        cells = [
            (r, c)
            for r in range(row_slice.start, row_slice.stop)
            for c in range(col_slice.start, col_slice.stop)
        ]
        for r, c in rng.sample(cells, count):
            field[r, c] = BLUE

    if len(_palette_candidates(grid)) != 1:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
