import random
from collections import deque

import numpy as np


DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NON_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GEN_TRIES = 400


def _extract_structure(grid: np.ndarray) -> tuple[list[int], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 4

    palette = [int(g[r, w - 1]) for r in range(h) if int(g[r, w - 1]) != 0]
    assert 2 <= len(palette) <= 4, "expected 2-4 palette pixels in the rightmost column"

    coords = np.argwhere(g[:, : w - 1] != 0)
    assert 4 <= len(coords) <= 8, "expected one horizontal line of length 4-8"
    rows = {int(r) for r, _c in coords}
    assert len(rows) == 1, "foreground outside palette must be a single row"
    row = rows.pop()

    cols = sorted(int(c) for _r, c in coords)
    assert cols == list(range(cols[0], cols[-1] + 1)), "line must be contiguous"
    line = [(row, c, int(g[row, c])) for c in cols]
    assert len({color for _r, _c, color in line}) >= 3, "line must use 3-5 distinct colors"
    return palette, line


def _cast_rays(shape: tuple[int, int], line: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    canvas = np.zeros((h, w), dtype=int)
    for r, c, color in line:
        canvas[r, c] = int(color)
        for dr, dc in DIAG_DIRS:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] == 0:
                canvas[nr, nc] = int(color)
                nr += dr
                nc += dc
    return canvas


def _background_regions(canvas: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = canvas.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if canvas[r, c] != 0 or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append(cells)
    return regions


def _solve_playfield(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, tuple[int, int], list[tuple[int, int]]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    palette, line = _extract_structure(g)
    canvas = _cast_rays((h, w - 1), line)
    regions = _background_regions(canvas)
    ranked = sorted(
        ((len(cells), min(cells), cells) for cells in regions),
        key=lambda item: (item[0], item[1][0], item[1][1]),
    )

    out = np.zeros_like(g)
    out[:, : w - 1] = canvas
    for idx, (_area, _anchor, cells) in enumerate(ranked):
        color = palette[idx % len(palette)]
        for r, c in cells:
            out[r, c] = int(color)
    return out, ranked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast diagonal rays from a multicolor line, then fill the ray-partitioned regions by ascending area using a cycling right-edge palette.

    Input structure:
    - Black background grid with exactly two foreground structures.
    - A single horizontal multicolor line of length 4-8 lies away from the right edge.
    - The rightmost column contains 2-4 palette pixels ordered from top to bottom; all other cells are black.

    Rule:
    - Ignore the palette column as geometry and work on a blank canvas covering the remaining columns.
    - Process the line pixels from left to right; each pixel paints itself and emits four 45-degree rays (NW, NE, SW, SE) that stop before the first already-painted cell or the grid boundary.
    - Treat the remaining black cells as 4-connected regions, sort those regions by area ascending, and break equal-area ties by the region's topmost-leftmost cell.
    - Fill the sorted regions with the palette colors in top-to-bottom palette order, cycling when there are more regions than palette entries; the palette column itself becomes black in the output.

    Key insight:
    - The left-to-right ray order turns one horizontal line into irregular partitions, and area sorting converts those partitions into a stable sequence that the palette can color cyclically.
    """
    out, _ranked = _solve_playfield(np.asarray(grid, dtype=int))
    return out


def _pick_spaced_rows(rng: random.Random, h: int, count: int) -> list[int]:
    min_gap = 2 if h >= 10 else 1
    for _ in range(80):
        rows = sorted(rng.sample(range(h), count))
        if all(rows[i + 1] - rows[i] >= min_gap for i in range(count - 1)):
            return rows
    return sorted(rng.sample(range(h), count))


def _build_line_colors(rng: random.Random, length: int, colors: list[int]) -> list[int]:
    min_distinct = max(3, (length + 1) // 2)
    max_distinct = min(5, length)
    distinct = rng.randint(min_distinct, max_distinct)
    base = colors[:distinct]
    counts = {color: 1 for color in base}
    extras = length - distinct
    for color in rng.sample(base, extras):
        counts[color] += 1

    sequence: list[int] = []
    prev = None
    while len(sequence) < length:
        options = [color for color, count in counts.items() if count > 0 and color != prev]
        if not options:
            options = [color for color, count in counts.items() if count > 0]
        max_count = max(counts[color] for color in options)
        best = [color for color in options if counts[color] == max_count]
        color = rng.choice(best)
        sequence.append(color)
        counts[color] -= 1
        prev = color
    return sequence


def _make_candidate_grid(
    rng: random.Random,
    h: int,
    w: int,
    row: int,
    start_col: int,
    length: int,
    palette_count: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    colors = NON_BLACK[:]
    rng.shuffle(colors)
    palette_colors = colors[:palette_count]
    line_pool = colors[palette_count:]

    line_colors = _build_line_colors(rng, length, line_pool)
    for offset, color in enumerate(line_colors):
        grid[row, start_col + offset] = int(color)

    for r, color in zip(_pick_spaced_rows(rng, h, palette_count), palette_colors):
        grid[r, w - 1] = int(color)
    return grid


def _valid_geometry(h: int, w: int, row: int, start_col: int, length: int) -> bool:
    if not (0 <= row < h):
        return False
    if start_col < 1:
        return False
    if start_col + length - 1 > w - 4:
        return False
    return True


def _geometry_pool(h: int, w: int) -> list[tuple[int, int, int]]:
    rows = [r for r in range(3, min(6, h - 4))]
    starts = [1, 2, 3, 4]
    lengths = [4, 5, 6, 7, 8]
    pool = []
    for row in rows:
        for start_col in starts:
            for length in lengths:
                if _valid_geometry(h, w, row, start_col, length):
                    pool.append((row, start_col, length))
    return pool


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_TRIES):
        h = rng.randint(13, 18)
        w = rng.randint(13, 19)
        geometries = _geometry_pool(h, w)
        if not geometries:
            continue

        want_tie = rng.random() < 0.35
        if want_tie and (4, 6) not in {(start, length) for _row, start, length in geometries}:
            want_tie = False

        if want_tie:
            tie_rows = [row for row, start, length in geometries if start == 4 and length == 6]
            row = rng.choice(tie_rows)
            start_col = 4
            length = 6
        else:
            row, start_col, length = rng.choice(geometries)

        palette_count = rng.randint(2, 4)
        grid = _make_candidate_grid(rng, h, w, row, start_col, length, palette_count)
        out, ranked = _solve_playfield(grid)
        areas = [area for area, _anchor, _cells in ranked]

        if len(ranked) < 4:
            continue
        if want_tie != (len(areas) != len(set(areas))):
            continue
        if np.any(out[:, -1] != 0):
            continue
        if len({int(color) for color in grid[:, -1] if int(color) != 0}) != palette_count:
            continue
        if len({int(grid[row, c]) for c in range(start_col, start_col + length)}) < 3:
            continue
        return grid
    raise RuntimeError("failed to generate a valid sample")
