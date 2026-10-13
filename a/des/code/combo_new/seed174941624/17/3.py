import random
from collections import Counter, deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
AZURE = 8

CELL_SIZE = 7
PITCH = CELL_SIZE + 1
MAIN_COLS = 2
MAIN_ROW_OPTIONS = (2, 3)
LEGEND_BANDS = 4
CANVAS_H = 30
GEN_RETRIES = 200


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _full_span_rows(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> list[int]:
    rows = []
    for r in range(r0, r1 + 1):
        if np.all(grid[r, c0 : c1 + 1] == color):
            rows.append(r)
    return rows


def _full_span_cols(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> list[int]:
    cols = []
    for c in range(c0, c1 + 1):
        if np.all(grid[r0 : r1 + 1, c] == color):
            cols.append(c)
    return cols


def _is_regular(lines: list[int]) -> bool:
    if len(lines) < 2:
        return False
    diffs = [b - a for a, b in zip(lines, lines[1:])]
    return len(set(diffs)) == 1


def _find_main_lattice(grid: np.ndarray) -> tuple[list[int], list[int]]:
    azure_mask = grid == AZURE
    candidates: list[tuple[int, list[int], list[int]]] = []
    for comp in _components4(azure_mask):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        row_lines = _full_span_rows(grid, AZURE, r0, c0, r1, c1)
        col_lines = _full_span_cols(grid, AZURE, r0, c0, r1, c1)
        if len(row_lines) < 3 or len(col_lines) < 2:
            continue
        if not _is_regular(row_lines) or not _is_regular(col_lines):
            continue
        row_pitch = row_lines[1] - row_lines[0]
        col_pitch = col_lines[1] - col_lines[0]
        if row_pitch != col_pitch:
            continue
        score = len(row_lines) * len(col_lines) * len(comp)
        candidates.append((score, row_lines, col_lines))
    assert candidates, "main lattice not found"
    _, row_lines, col_lines = max(candidates, key=lambda item: item[0])
    return row_lines, col_lines


def _draw_nested_frames(patch: np.ndarray, layers: int) -> None:
    h, w = patch.shape
    patch[:, :] = BG
    for inset in range(layers):
        r0 = inset
        r1 = h - 1 - inset
        c0 = inset
        c1 = w - 1 - inset
        if r0 > r1 or c0 > c1:
            break
        patch[r0, c0 : c1 + 1] = GREEN
        patch[r1, c0 : c1 + 1] = GREEN
        patch[r0 : r1 + 1, c0] = GREEN
        patch[r0 : r1 + 1, c1] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced cell with a nested frame whose depth matches its red-dot count.

    Input structure:
    - The background is black, and the main area is partitioned by equally spaced azure fence lines into square cells.
    - Each main cell contains zero to four red marker pixels at arbitrary positions.
    - A legend region sits to the right of the main area behind a double azure separator and shows example green nested frames.

    Rule:
    - Detect the regular azure lattice that forms the main cell grid.
    - For each cell interior, count its red pixels.
    - If the count is zero, clear the cell interior to black; otherwise draw that many concentric green rectangular frames, shrinking inward by one pixel per layer.
    - Remove the original red markers while leaving the fences and legend unchanged.

    Key insight:
    - The red markers are only a count signal: their quantity, not their placement, determines the nesting depth of the green frame pattern inside each cell.
    """
    g = np.asarray(grid, dtype=int)
    row_lines, col_lines = _find_main_lattice(g)

    out = g.copy()
    for r0, r1 in zip(row_lines, row_lines[1:]):
        for c0, c1 in zip(col_lines, col_lines[1:]):
            cell = g[r0 + 1 : r1, c0 + 1 : c1]
            layers = int(np.count_nonzero(cell == RED))
            patch = out[r0 + 1 : r1, c0 + 1 : c1]
            _draw_nested_frames(patch, layers)
    return out


def _draw_main_grid(grid: np.ndarray, top: int, left: int, rows: int, cols: int) -> None:
    h = rows * CELL_SIZE + rows + 1
    w = cols * CELL_SIZE + cols + 1
    for r in range(top, top + h, PITCH):
        grid[r, left : left + w] = AZURE
    for c in range(left, left + w, PITCH):
        grid[top : top + h, c] = AZURE


def _legend_geometry(main_left: int, main_w: int, rng: random.Random) -> tuple[int, int, int]:
    gap = rng.choice((1, 2))
    sep0 = main_left + main_w + gap
    sep1 = sep0 + 1
    legend_left = sep1 + 1
    right_border = legend_left + CELL_SIZE
    return sep0, legend_left, right_border


def _draw_legend(grid: np.ndarray, sep0: int, legend_left: int, right_border: int) -> None:
    sep1 = sep0 + 1
    grid[:, sep0] = AZURE
    grid[:, sep1] = AZURE
    grid[0, sep0 : right_border + 1] = AZURE
    grid[-1, sep0 : right_border + 1] = AZURE
    grid[:, right_border] = AZURE

    for idx in range(LEGEND_BANDS):
        r0 = 1 + idx * CELL_SIZE
        r1 = r0 + CELL_SIZE
        patch = grid[r0:r1, legend_left:right_border]
        _draw_nested_frames(patch, idx + 1)
        patch[0, : idx + 1] = RED


def _random_marker_positions(rng: random.Random, count: int) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE)]
    return rng.sample(coords, count)


def _cell_counts(rng: random.Random, rows: int, cols: int) -> list[int]:
    n = rows * cols
    if n >= 5:
        counts = [0, 1, 2, 3, 4]
        while len(counts) < n:
            counts.append(rng.randint(0, 4))
    else:
        counts = rng.sample([0, 1, 2, 3, 4], n)
    rng.shuffle(counts)
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.choice(MAIN_ROW_OPTIONS)
    cols = MAIN_COLS
    main_h = rows * CELL_SIZE + rows + 1
    main_w = cols * CELL_SIZE + cols + 1

    top = rng.randint(0, CANVAS_H - main_h)
    left = rng.randint(0, 1)
    sep0, legend_left, right_border = _legend_geometry(left, main_w, rng)
    canvas_w = right_border + 1 + rng.randint(0, 1)
    if canvas_w > 30:
        return None

    grid = np.full((CANVAS_H, canvas_w), BG, dtype=int)
    _draw_main_grid(grid, top, left, rows, cols)
    _draw_legend(grid, sep0, legend_left, right_border)

    counts = _cell_counts(rng, rows, cols)
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            count = counts[idx]
            idx += 1
            inner_r0 = top + 1 + rr * PITCH
            inner_c0 = left + 1 + cc * PITCH
            for dr, dc in _random_marker_positions(rng, count):
                grid[inner_r0 + dr, inner_c0 + dc] = RED

    row_lines, col_lines = _find_main_lattice(grid)
    if len(row_lines) != rows + 1 or len(col_lines) != cols + 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
