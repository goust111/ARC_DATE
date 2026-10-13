from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEED = 8
LINE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GEN_RETRIES = 300


def _components4(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[list[tuple[int, int]], bool]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            touches_border = False

            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True

                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            out.append((pts, touches_border))

    return out


def _rectangular_regions(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    passable = (grid == BG) | (grid == SEED)
    regions: list[tuple[int, int, int, int]] = []

    for pts, touches_border in _components4(passable):
        if touches_border:
            continue

        arr = np.array(pts, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area != len(pts):
            continue

        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((window == BG) | (window == SEED)):
            continue

        regions.append((int(r0), int(c0), int(r1), int(c1)))

    regions.sort()
    return regions


def _side_color(values: np.ndarray) -> int:
    colors = {int(v) for v in np.asarray(values).ravel() if int(v) != BG}
    assert len(colors) == 1, "each rectangle side should have one boundary color"
    return int(next(iter(colors)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow inward concentric frames inside each seeded rectangular region, using the region's own side colors for the corresponding frame sides.

    Input structure:
    - bg = black.
    - Non-bg 1-pixel lines form closed rectangular regions; neighboring regions may share boundary segments.
    - Some region interiors contain 0-3 scattered azure seed pixels; other interior cells are bg.

    Rule:
    - Detect every enclosed rectangular interior and count its seed pixels.
    - For a region with N seeds, clear the seeds and draw N inward concentric rectangular frame layers, starting flush against the inner wall.
    - On every layer, the top/bottom/left/right frame sides inherit the colors of that region's corresponding outer boundary sides; corner pixels use the horizontal side color.

    Key insight:
    - The seed count encodes how many nested layers to grow, while the four side colors must be copied direction-wise rather than treating each layer as monochrome.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1 in _rectangular_regions(g):
        interior = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_count = int(np.count_nonzero(interior == SEED))

        cleared = out[r0 : r1 + 1, c0 : c1 + 1]
        cleared[interior == SEED] = BG

        if seed_count == 0:
            continue

        top_color = _side_color(g[r0 - 1, c0 : c1 + 1])
        bottom_color = _side_color(g[r1 + 1, c0 : c1 + 1])
        left_color = _side_color(g[r0 : r1 + 1, c0 - 1])
        right_color = _side_color(g[r0 : r1 + 1, c1 + 1])

        for layer in range(seed_count):
            tr = int(r0 + layer)
            br = int(r1 - layer)
            lc = int(c0 + layer)
            rc = int(c1 - layer)
            if tr > br or lc > rc:
                break

            out[tr, lc : rc + 1] = top_color
            if br != tr:
                out[br, lc : rc + 1] = bottom_color

            if tr + 1 <= br - 1:
                out[tr + 1 : br, lc] = left_color
                if rc != lc:
                    out[tr + 1 : br, rc] = right_color

    return out


def _sample_counts(rows: int, cols: int) -> np.ndarray:
    counts = np.array(
        [[random.randint(0, 3) for _ in range(cols)] for _ in range(rows)],
        dtype=int,
    )

    if not np.any(counts > 0):
        rr = random.randrange(rows)
        cc = random.randrange(cols)
        counts[rr, cc] = random.randint(1, 3)

    if rows * cols > 1 and not np.any(counts == 0):
        rr = random.randrange(rows)
        cc = random.randrange(cols)
        counts[rr, cc] = 0

    return counts


def _sized_axis(requirements: list[int]) -> tuple[list[int], int, int]:
    board_size = sum(requirements) + len(requirements) + 1
    min_total = board_size + 2
    if min_total > 30:
        raise ValueError("board is too large for ARC canvas")

    target_total = random.randint(min_total, 30)
    extra = target_total - board_size

    sizes = requirements[:]
    outer_a = 1
    outer_b = 1
    extra -= 2

    slots = list(range(len(sizes) + 2))
    while extra > 0:
        idx = random.choice(slots)
        if idx < len(sizes):
            sizes[idx] += 1
        elif idx == len(sizes):
            outer_a += 1
        else:
            outer_b += 1
        extra -= 1

    return sizes, outer_a, outer_b


def _axis_boundaries(sizes: list[int]) -> list[int]:
    bounds = [0]
    pos = 0
    for size in sizes:
        pos += size + 1
        bounds.append(pos)
    return bounds


def _seed_positions(r0: int, r1: int, c0: int, c1: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(cells)
    picked: list[tuple[int, int]] = []

    for cell in cells:
        if all(max(abs(cell[0] - pr), abs(cell[1] - pc)) >= 2 for pr, pc in picked):
            picked.append(cell)
            if len(picked) == count:
                return picked

    return random.sample(cells, count)


def _cell_side_colors(
    h_colors: np.ndarray,
    v_colors: np.ndarray,
    r: int,
    c: int,
) -> tuple[int, int, int, int]:
    top = int(h_colors[r, c])
    bottom = int(h_colors[r + 1, c])
    left = int(v_colors[r, c])
    right = int(v_colors[r, c + 1])
    return top, bottom, left, right


def _draw_board(
    grid: np.ndarray,
    top_margin: int,
    left_margin: int,
    row_sizes: list[int],
    col_sizes: list[int],
    h_colors: np.ndarray,
    v_colors: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    yb = _axis_boundaries(row_sizes)
    xb = _axis_boundaries(col_sizes)
    cells: list[tuple[int, int, int, int]] = []

    for hr in range(len(row_sizes) + 1):
        rr = top_margin + yb[hr]
        for hc in range(len(col_sizes)):
            c0 = left_margin + xb[hc] + 1
            c1 = left_margin + xb[hc + 1] - 1
            grid[rr, c0 : c1 + 1] = int(h_colors[hr, hc])

    for vc in range(len(col_sizes) + 1):
        cc = left_margin + xb[vc]
        for vr in range(len(row_sizes)):
            r0 = top_margin + yb[vr] + 1
            r1 = top_margin + yb[vr + 1] - 1
            grid[r0 : r1 + 1, cc] = int(v_colors[vr, vc])

    for ir in range(len(row_sizes) + 1):
        rr = top_margin + yb[ir]
        for ic in range(len(col_sizes) + 1):
            choices: list[int] = []
            if ic > 0:
                choices.append(int(h_colors[ir, ic - 1]))
            if ic < len(col_sizes):
                choices.append(int(h_colors[ir, ic]))
            if ir > 0:
                choices.append(int(v_colors[ir - 1, ic]))
            if ir < len(row_sizes):
                choices.append(int(v_colors[ir, ic]))
            grid[rr, left_margin + xb[ic]] = random.choice(choices)

    for r in range(len(row_sizes)):
        for c in range(len(col_sizes)):
            r0 = top_margin + yb[r] + 1
            r1 = top_margin + yb[r + 1] - 1
            c0 = left_margin + xb[c] + 1
            c1 = left_margin + xb[c + 1] - 1
            cells.append((r0, c0, r1, c1))

    return cells


def _interesting_seeded_cell(counts: np.ndarray, h_colors: np.ndarray, v_colors: np.ndarray) -> bool:
    rows, cols = counts.shape
    for r in range(rows):
        for c in range(cols):
            if int(counts[r, c]) == 0:
                continue
            side_set = set(_cell_side_colors(h_colors, v_colors, r, c))
            if len(side_set) >= 2:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(1, 3)
    cols = random.randint(1, 3)
    counts = _sample_counts(rows, cols)

    row_requirements = [max(3, 2 * int(counts[r, :].max()) + 1) for r in range(rows)]
    col_requirements = [max(3, 2 * int(counts[:, c].max()) + 1) for c in range(cols)]

    try:
        row_sizes, top_margin, bottom_margin = _sized_axis(row_requirements)
        col_sizes, left_margin, right_margin = _sized_axis(col_requirements)
    except ValueError:
        return None

    height = top_margin + sum(row_sizes) + rows + 1 + bottom_margin
    width = left_margin + sum(col_sizes) + cols + 1 + right_margin
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)

    h_colors = np.array(
        [[random.choice(LINE_COLORS) for _ in range(cols)] for _ in range(rows + 1)],
        dtype=int,
    )
    v_colors = np.array(
        [[random.choice(LINE_COLORS) for _ in range(cols + 1)] for _ in range(rows)],
        dtype=int,
    )

    if not _interesting_seeded_cell(counts, h_colors, v_colors):
        return None

    cells = _draw_board(grid, top_margin, left_margin, row_sizes, col_sizes, h_colors, v_colors)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            r0, c0, r1, c1 = cells[idx]
            idx += 1
            seed_count = int(counts[r, c])
            if seed_count == 0:
                continue
            for sr, sc in _seed_positions(r0, r1, c0, c1, seed_count):
                grid[sr, sc] = SEED

    if len(_rectangular_regions(grid)) != rows * cols:
        return None

    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
