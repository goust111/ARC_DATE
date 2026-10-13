from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL = 5
HIT_FILL = 3
MISS_FILL = 4
ARROW_PALETTE = (1, 2, 6, 7, 8, 9)
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GENERATION_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    """ARC-style background detector: most common color in the grid."""
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_monochrome_components(
    grid: np.ndarray,
    *,
    forbidden: set[int],
) -> list[dict]:
    """Minimal 4-connected monochrome component finder using arc_utils conventions."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in forbidden:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not in_bounds(g.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def detect_frame_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if bool((row == row[0]).all()) and int(row[0]) != int(bg):
            candidates.append(int(row[0]))
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if bool((col == col[0]).all()) and int(col[0]) != int(bg):
            candidates.append(int(col[0]))
    if candidates:
        return int(Counter(candidates).most_common(1)[0][0])
    return WALL


def detect_cells(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int]]:
    row_lines = [r for r in range(grid.shape[0]) if bool((grid[r, :] == int(wall)).all())]
    col_lines = [c for c in range(grid.shape[1]) if bool((grid[:, c] == int(wall)).all())]
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "expected full lattice lines"

    row_spans = [(row_lines[i] + 1, row_lines[i + 1] - 1) for i in range(len(row_lines) - 1) if row_lines[i + 1] - row_lines[i] > 1]
    col_spans = [(col_lines[i] + 1, col_lines[i + 1] - 1) for i in range(len(col_lines) - 1) if col_lines[i + 1] - col_lines[i] > 1]
    return [(r0, r1, c0, c1) for r0, r1 in row_spans for c0, c1 in col_spans]


def arrow_direction(comp: dict) -> tuple[int, int]:
    coords = np.asarray(comp["coords"], dtype=float)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    center = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0], dtype=float)
    centroid = coords.mean(axis=0)
    dr, dc = np.sign(centroid - center).astype(int)
    assert abs(int(dr)) == 1 and abs(int(dc)) == 1, "arrow direction must be diagonal"
    return int(dr), int(dc)


def trace_ray(
    out: np.ndarray,
    ray_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    color: int,
    *,
    wall: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    dr, dc = map(int, direction)
    r = (r1 if dr > 0 else r0) + dr
    c = (c1 if dc > 0 else c0) + dc

    while in_bounds(out.shape, r, c) and int(out[r, c]) != int(wall):
        if int(out[r, c]) == int(bg):
            out[r, c] = int(color)
            ray_mask[r, c] = True
        r += dr
        c += dc


def render_arrow(color: int, direction: tuple[int, int]) -> np.ndarray:
    sprite = np.full((2, 2), int(color), dtype=int)
    dr, dc = map(int, direction)
    missing_r = 0 if dr > 0 else 1
    missing_c = 0 if dc > 0 else 1
    sprite[missing_r, missing_c] = BG
    return sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire diagonal rays from non-dominant arrow colors and flood each lattice cell by whether a ray crossed it.

    Input structure:
    - The background is black, with a complete gray lattice made of full rows and full columns.
    - Some lattice cells contain a single 3-pixel monochrome arrowhead: a 2x2 block missing one corner.
    - Arrow colors appear with a unique most-frequent color, and arrows never touch the gray walls.

    Rule:
    - Count arrowheads by color and deactivate every arrow of the unique most-frequent color.
    - For each remaining arrow, compare its 3-pixel centroid to its 2x2 bbox center to infer one of the four diagonal directions.
    - Start just outside the arrow bbox on that corner and extend a same-colored 45-degree ray until the next step would hit a gray wall or leave the grid.
    - After all rays are drawn, fill every cell's remaining black pixels green if that cell contains any new ray pixel; otherwise fill those black pixels yellow.

    Key insight:
    - The missing corner gives a diagonal direction, but only colors that are not the global frequency winner are allowed to emit rays, and those rays define the later cell-wise flood fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    wall = detect_frame_color(g, bg)
    arrows = find_monochrome_components(g, forbidden={bg, wall})
    assert arrows, "expected arrow components"

    freq = Counter(int(comp["color"]) for comp in arrows)
    most_common = freq.most_common()
    assert len(most_common) == 1 or int(most_common[0][1]) > int(most_common[1][1]), "most frequent arrow color must be unique"
    dormant_color = int(most_common[0][0])

    out = g.copy()
    ray_mask = np.zeros_like(g, dtype=bool)

    for comp in arrows:
        assert int(comp["area"]) == 3, "each arrow must have area 3"
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each arrow must fit a 2x2 bbox"
        if int(comp["color"]) == dormant_color:
            continue
        direction = arrow_direction(comp)
        trace_ray(out, ray_mask, comp["bbox"], direction, int(comp["color"]), wall=wall, bg=bg)

    for r0, r1, c0, c1 in detect_cells(g, wall):
        fill = HIT_FILL if bool(ray_mask[r0 : r1 + 1, c0 : c1 + 1].any()) else MISS_FILL
        cell = out[r0 : r1 + 1, c0 : c1 + 1]
        cell[cell == int(bg)] = int(fill)

    return out


def choose_lattice() -> tuple[int, int, int, int]:
    for _ in range(200):
        n_rows = random.randint(3, 5)
        n_cols = random.randint(3, 5)
        cell_h = random.randint(4, 5)
        cell_w = random.randint(4, 5)
        height = n_rows * cell_h + (n_rows + 1)
        width = n_cols * cell_w + (n_cols + 1)
        if height <= 30 and width <= 30:
            return n_rows, n_cols, cell_h, cell_w
    raise RuntimeError("failed to choose lattice within ARC bounds")


def choose_color_counts(total_arrows: int, num_colors: int) -> list[int]:
    while True:
        dominant = random.randint(2, total_arrows - num_colors + 1)
        remaining = total_arrows - dominant
        if remaining < 2 or remaining < num_colors - 1:
            continue
        others = [1] * (num_colors - 1)
        for _ in range(remaining - (num_colors - 1)):
            others[random.randrange(num_colors - 1)] += 1
        if dominant > max(others):
            return [dominant, *others]


def cell_bounds(n_rows: int, n_cols: int, cell_h: int, cell_w: int) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]]:
    row_lines = [k * (cell_h + 1) for k in range(n_rows + 1)]
    col_lines = [k * (cell_w + 1) for k in range(n_cols + 1)]
    cells = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0 = row_lines[rr] + 1
            r1 = row_lines[rr + 1] - 1
            c0 = col_lines[cc] + 1
            c1 = col_lines[cc + 1] - 1
            cells.append((r0, r1, c0, c1))
    return row_lines, col_lines, cells


def place_arrow(
    grid: np.ndarray,
    cell: tuple[int, int, int, int],
    color: int,
    direction: tuple[int, int],
) -> None:
    r0, r1, c0, c1 = map(int, cell)
    top = random.randint(r0 + 1, r1 - 2)
    left = random.randint(c0 + 1, c1 - 2)
    sprite = render_arrow(color, direction)
    grid[top : top + 2, left : left + 2] = np.where(sprite != BG, sprite, grid[top : top + 2, left : left + 2])


def generation_attempt() -> np.ndarray | None:
    n_rows, n_cols, cell_h, cell_w = choose_lattice()
    row_lines, col_lines, cells = cell_bounds(n_rows, n_cols, cell_h, cell_w)

    grid = np.full((n_rows * cell_h + (n_rows + 1), n_cols * cell_w + (n_cols + 1)), BG, dtype=int)
    grid[row_lines, :] = WALL
    grid[:, col_lines] = WALL

    total_cells = len(cells)
    total_arrows = random.randint(5, min(8, total_cells))
    num_colors = random.choice([2, 3])
    if total_arrows < num_colors + 2:
        total_arrows = num_colors + 2

    colors = random.sample(ARROW_PALETTE, num_colors)
    counts = choose_color_counts(total_arrows, num_colors)
    dormant_color = colors[0]
    arrow_colors = [color for color, count in zip(colors, counts) for _ in range(count)]
    random.shuffle(arrow_colors)

    directions = [random.choice(DIAGONALS) for _ in range(total_arrows)]
    active_indices = [i for i, color in enumerate(arrow_colors) if int(color) != int(dormant_color)]
    if len(active_indices) >= 2 and len({directions[i] for i in active_indices}) == 1:
        pivot = active_indices[-1]
        first_dir = directions[active_indices[0]]
        directions[pivot] = random.choice([d for d in DIAGONALS if d != first_dir])

    chosen_cells = random.sample(cells, total_arrows)
    for cell, color, direction in zip(chosen_cells, arrow_colors, directions):
        place_arrow(grid, cell, int(color), direction)

    freq = Counter(arrow_colors)
    if len(freq) < 2 or int(freq[dormant_color]) <= max(v for k, v in freq.items() if k != dormant_color):
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if not bool((out == HIT_FILL).any()) or not bool((out == MISS_FILL).any()):
        return None
    if len({tuple(direction) for direction in directions}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frequency-gated arrow lattice input")
