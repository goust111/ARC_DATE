import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
COLOR_OFFSETS = {
    1: (0, 1),   # blue -> right
    2: (0, -1),  # red -> left
    7: (-1, 0),  # orange -> up
    9: (1, 0),   # maroon -> down
}
DRAW_ORDER = [1, 2, 7, 9]
GEN_RETRIES = 400


def _rect_edge_cells(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    rows = sorted({int(r) for r, _ in points})
    cols = sorted({int(c) for _, c in points})
    assert len(rows) == 2 and len(cols) == 2, "shifted markers must form rectangle corners"

    r0, r1 = rows
    c0, c1 = cols
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _render_lattice(
    tile_rows: int,
    tile_cols: int,
    tile_size: int,
    gap: int,
    margin: int,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    height = 2 * margin + tile_rows * tile_size + (tile_rows - 1) * gap
    width = 2 * margin + tile_cols * tile_size + (tile_cols - 1) * gap
    grid = np.full((height, width), BG, dtype=int)
    centers: dict[tuple[int, int], tuple[int, int]] = {}
    center_offset = tile_size // 2

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            r0 = margin + tr * (tile_size + gap)
            c0 = margin + tc * (tile_size + gap)
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = FRAME
            if tile_size > 2:
                grid[r0 + 1 : r0 + tile_size - 1, c0 + 1 : c0 + tile_size - 1] = BG
            centers[(tr, tc)] = (r0 + center_offset, c0 + center_offset)

    return grid, centers


def _translated_markers(
    groups: dict[int, list[tuple[int, int]]]
) -> dict[int, list[tuple[int, int]]]:
    moved: dict[int, list[tuple[int, int]]] = {}
    for color, points in groups.items():
        dr, dc = COLOR_OFFSETS[color]
        moved[color] = [(r + dr, c + dc) for r, c in points]
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift motion-colored tile markers, then draw each color's rectangle through the lattice channels.

    Input structure:
    - A black background contains a regular array of grey framed tiles separated by black channels.
    - Some tile centers contain single-pixel markers in the motion colors blue, red, orange, or maroon.
    - Every present marker color appears exactly four times.

    Rule:
    - Remove each marker from its original cell and move it one cell according to its color: blue right, red left, orange up, maroon down.
    - For each color, treat the four shifted markers as the corners of one axis-aligned rectangle.
    - Draw that rectangle's four sides in the same color, covering black channel cells or grey frame cells as needed.
    - If a later rectangle crosses an earlier line, the later color overwrites it, but translated marker pixels of other colors are preserved.

    Key insight:
    - The marker color serves two roles at once: it chooses the translation direction and also defines which four shifted pixels belong to the same rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    groups: dict[int, list[tuple[int, int]]] = {}
    for color in DRAW_ORDER:
        coords = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        if coords:
            assert len(coords) == 4, f"color {color} must appear exactly four times"
            groups[color] = coords

    out = g.copy()
    for points in groups.values():
        for r, c in points:
            out[r, c] = BG

    moved = _translated_markers(groups)
    marker_owner: dict[tuple[int, int], int] = {}
    for color, points in moved.items():
        for r, c in points:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "translated marker out of bounds"
            assert (r, c) not in marker_owner, "translated markers must stay distinct"
            marker_owner[(r, c)] = color
            out[r, c] = color

    for color in DRAW_ORDER:
        if color not in moved:
            continue
        for r, c in _rect_edge_cells(moved[color]):
            owner = marker_owner.get((r, c))
            if owner is not None and owner != color:
                continue
            out[r, c] = color

    return out


def _pick_rectangles(
    tile_rows: int,
    tile_cols: int,
    colors: list[int],
) -> dict[int, list[tuple[int, int]]] | None:
    rectangles = [
        (r0, r1, c0, c1)
        for r0 in range(tile_rows)
        for r1 in range(r0 + 1, tile_rows)
        for c0 in range(tile_cols)
        for c1 in range(c0 + 1, tile_cols)
    ]
    random.shuffle(rectangles)

    groups: dict[int, list[tuple[int, int]]] = {}
    used_tiles: set[tuple[int, int]] = set()
    for color in colors:
        placed = False
        for r0, r1, c0, c1 in rectangles:
            corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
            if any(corner in used_tiles for corner in corners):
                continue
            groups[color] = corners
            used_tiles.update(corners)
            placed = True
            break
        if not placed:
            return None
    return groups


def _build_candidate() -> np.ndarray | None:
    roomy = random.random() < 0.65
    if roomy:
        tile_size = 5
        gap = random.randint(1, 2)
    else:
        tile_size = 3
        gap = random.randint(1, 2)

    tile_rows = random.randint(3, 4)
    tile_cols = random.randint(3, 4)
    margin = random.randint(1, 2)

    base, centers = _render_lattice(tile_rows, tile_cols, tile_size, gap, margin)
    max_colors = min(len(DRAW_ORDER), (tile_rows * tile_cols) // 4)
    color_count = random.randint(2, max_colors)
    colors = random.sample(DRAW_ORDER, color_count)

    tile_groups = _pick_rectangles(tile_rows, tile_cols, colors)
    if tile_groups is None:
        return None

    groups: dict[int, list[tuple[int, int]]] = {
        color: [centers[tile] for tile in tiles]
        for color, tiles in tile_groups.items()
    }
    moved = _translated_markers(groups)

    all_moved = [point for points in moved.values() for point in points]
    if len(all_moved) != len(set(all_moved)):
        return None

    if any(not (0 <= r < base.shape[0] and 0 <= c < base.shape[1]) for r, c in all_moved):
        return None

    edges_by_color = {color: _rect_edge_cells(points) for color, points in moved.items()}
    source_points = {point for points in groups.values() for point in points}
    moved_points = {point: color for color, points in moved.items() for point in points}

    for color, edge in edges_by_color.items():
        for point, other_color in moved_points.items():
            if other_color != color and point in edge:
                return None
        for point in source_points:
            if point in edge and point not in set(groups[color]):
                return None

    overlap_cells: set[tuple[int, int]] = set()
    colors_list = list(edges_by_color)
    for i, color_a in enumerate(colors_list):
        for color_b in colors_list[i + 1 :]:
            overlap_cells |= edges_by_color[color_a] & edges_by_color[color_b]

    allow_cross = color_count >= 3 and random.random() < 0.55
    if allow_cross:
        if not overlap_cells or len(overlap_cells) > 6:
            return None
    elif overlap_cells:
        return None

    if len(set().union(*edges_by_color.values())) > int(base.size * 0.34):
        return None

    grid = base.copy()
    for color, points in groups.items():
        for r, c in points:
            grid[r, c] = color

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[counts.argmax()]) != BG:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    changed = int(np.count_nonzero(out != grid))
    if changed < 12:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 10")
