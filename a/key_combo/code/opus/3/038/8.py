from __future__ import annotations

import importlib.util
import random

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_038_8", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords


GEN_ATTEMPTS = 400
GRID_MIN = 13
GRID_MAX = 22
MIN_ROOM_SIZE = 3
BLOCK_ROOM_MIN = 4
LINE_COUNT_OPTIONS = (
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
)
MARKER_COLORS = tuple(color for color in Color.NOT_BLACK if color != Color.GRAY)


def _boundary_dots(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    dots: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        if int(grid[0, c]) == Color.GRAY:
            dots.append((0, c))
        if int(grid[h - 1, c]) == Color.GRAY:
            dots.append((h - 1, c))
    for r in range(1, h - 1):
        if int(grid[r, 0]) == Color.GRAY:
            dots.append((r, 0))
        if int(grid[r, w - 1]) == Color.GRAY:
            dots.append((r, w - 1))
    return dots


def _ray_positions(dots: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[list[int], list[int]]:
    h, w = shape
    horizontal_rows = sorted({int(r) for r, c in dots if c in {0, w - 1}})
    vertical_cols = sorted({int(c) for r, c in dots if r in {0, h - 1}})
    return horizontal_rows, vertical_cols


def _marker_block(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    marker_mask = (grid != Color.BLACK) & (grid != Color.GRAY)
    coords = np.argwhere(marker_mask)
    assert coords.shape[0] == 4, "expected exactly four non-black non-gray marker cells"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert r1 == r0 + 1 and c1 == c0 + 1, "marker cells must form a 2x2 block"
    assert bool(marker_mask[r0 : r1 + 1, c0 : c1 + 1].all()), "marker bbox must be full"
    block = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = [int(block[0, 0]), int(block[0, 1]), int(block[1, 0]), int(block[1, 1])]
    assert len(set(colors)) == 4, "marker block colors must be distinct"
    assert all(color not in {Color.BLACK, Color.GRAY} for color in colors), "marker block uses reserved colors"
    return int(r0), int(c0), block


def _room_slices(limit: int, separators: list[int]) -> list[tuple[int, int]]:
    walls = [0, *separators, limit - 1]
    spans = [(walls[i] + 1, walls[i + 1]) for i in range(len(walls) - 1)]
    assert all(start < end for start, end in spans), "every region must have positive area"
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend gray edge-dot rays and color each ray-bounded room from a repeating 2x2 marker block.

    Input structure:
    - The grid has a black background.
    - Gray pixels appear only as non-corner boundary dots on the outer frame.
    - Exactly four non-black, non-gray pixels form one interior 2x2 marker block with four distinct colors.

    Rule:
    - Each top-edge or bottom-edge gray dot casts a vertical gray ray through the interior of its column, and each left-edge or right-edge gray dot casts a horizontal gray ray through the interior of its row.
    - The rays together with the border partition the interior into rectangular rooms indexed top-to-bottom and left-to-right as (i, j).
    - Read the marker block as TL, TR, BL, BR; then fill every black cell in room (i, j) with the block color at position (i mod 2, j mod 2).
    - Keep the interior rays and the marker block unchanged, but erase the boundary dots themselves back to black.

    Key insight:
    - The edge dots define the room topology, while the 2x2 marker block acts as a wallpaper tile whose parity pattern assigns a color to every room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid is too small for this task"

    dots = _boundary_dots(g)
    assert 3 <= len(dots), "expected at least three boundary dots"
    horizontal_rows, vertical_cols = _ray_positions(dots, g.shape)
    assert horizontal_rows and vertical_cols, "need both horizontal and vertical ray families"

    marker_r, marker_c, block = _marker_block(g)
    assert marker_r not in horizontal_rows and marker_r + 1 not in horizontal_rows, "marker block overlaps a ray row"
    assert marker_c not in vertical_cols and marker_c + 1 not in vertical_cols, "marker block overlaps a ray column"

    out = g.copy()
    for c in vertical_cols:
        out[1 : h - 1, c] = Color.GRAY
    for r in horizontal_rows:
        out[r, 1 : w - 1] = Color.GRAY
    for r, c in dots:
        out[r, c] = Color.BLACK

    row_rooms = _room_slices(h, horizontal_rows)
    col_rooms = _room_slices(w, vertical_cols)
    for i, (r0, r1) in enumerate(row_rooms):
        for j, (c0, c1) in enumerate(col_rooms):
            room = out[r0:r1, c0:c1]
            room_color = int(block[i % 2, j % 2])
            room[room == Color.BLACK] = room_color

    return out


def _centerish_index(count: int) -> int:
    order = sorted(range(count), key=lambda idx: (abs(idx - (count - 1) / 2), random.random()))
    return random.choice(order[: min(2, count)])


def _sample_room_sizes(total: int, parts: int, block_index: int) -> list[int] | None:
    sizes = [MIN_ROOM_SIZE] * parts
    sizes[block_index] = BLOCK_ROOM_MIN
    extra = total - sum(sizes)
    if extra < 0:
        return None
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1
    return sizes


def _line_positions(room_sizes: list[int]) -> list[int]:
    lines: list[int] = []
    cursor = 0
    for size in room_sizes[:-1]:
        cursor += size + 1
        lines.append(cursor)
    return lines


def _assign_edge_sides(count: int, side_a: str, side_b: str) -> list[str]:
    min_a = max(0, count - 2)
    max_a = min(2, count)
    count_a = random.randint(min_a, max_a)
    sides = [side_a] * count_a + [side_b] * (count - count_a)
    random.shuffle(sides)
    return sides


def _place_marker_block(grid: np.ndarray, row_slice: tuple[int, int], col_slice: tuple[int, int]) -> None:
    r0, r1 = row_slice
    c0, c1 = col_slice
    block_r = random.randint(r0 + 1, r1 - 3)
    block_c = random.randint(c0 + 1, c1 - 3)
    colors = random.sample(MARKER_COLORS, 4)
    block = np.array(colors, dtype=int).reshape(2, 2)
    grid[block_r : block_r + 2, block_c : block_c + 2] = block


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and max(g.shape) <= 30

    dots = _boundary_dots(g)
    assert 3 <= len(dots) <= 5, "boundary dot count must stay within the prompt bounds"
    edge_counts = {
        "top": int(np.count_nonzero(g[0, 1:-1] == Color.GRAY)),
        "bottom": int(np.count_nonzero(g[h - 1, 1:-1] == Color.GRAY)),
        "left": int(np.count_nonzero(g[1:-1, 0] == Color.GRAY)),
        "right": int(np.count_nonzero(g[1:-1, w - 1] == Color.GRAY)),
    }
    assert all(count <= 2 for count in edge_counts.values()), "each edge may hold at most two boundary dots"

    horizontal_rows, vertical_cols = _ray_positions(dots, g.shape)
    assert horizontal_rows and vertical_cols, "need at least one ray in each orientation"

    marker_r, marker_c, _block = _marker_block(g)
    assert marker_r not in horizontal_rows and marker_r + 1 not in horizontal_rows
    assert marker_c not in vertical_cols and marker_c + 1 not in vertical_cols

    row_rooms = _room_slices(h, horizontal_rows)
    col_rooms = _room_slices(w, vertical_cols)
    room_found = False
    for r0, r1 in row_rooms:
        for c0, c1 in col_rooms:
            if r0 <= marker_r and marker_r + 1 < r1 and c0 <= marker_c and marker_c + 1 < c1:
                room_found = True
    assert room_found, "marker block must lie inside exactly one room"

    out = main(g)
    assert out.shape == g.shape
    assert np.all(out[0, :] == Color.BLACK)
    assert np.all(out[h - 1, :] == Color.BLACK)
    assert np.all(out[:, 0] == Color.BLACK)
    assert np.all(out[:, w - 1] == Color.BLACK)
    assert not np.any(out[1 : h - 1, 1 : w - 1] == Color.BLACK), "all interior background cells must be filled"


def _gen_attempt() -> np.ndarray | None:
    n_h, n_v = random.choice(LINE_COUNT_OPTIONS)
    h_min = max(GRID_MIN, n_h + 2 + MIN_ROOM_SIZE * (n_h + 1) + 1)
    w_min = max(GRID_MIN, n_v + 2 + MIN_ROOM_SIZE * (n_v + 1) + 1)
    if h_min > GRID_MAX or w_min > GRID_MAX:
        return None

    h = random.randint(h_min, GRID_MAX)
    w = random.randint(w_min, GRID_MAX)
    grid = np.full((h, w), Color.BLACK, dtype=int)

    block_row_index = _centerish_index(n_h + 1)
    block_col_index = _centerish_index(n_v + 1)
    row_sizes = _sample_room_sizes(h - 2 - n_h, n_h + 1, block_row_index)
    col_sizes = _sample_room_sizes(w - 2 - n_v, n_v + 1, block_col_index)
    if row_sizes is None or col_sizes is None:
        return None

    horizontal_rows = _line_positions(row_sizes)
    vertical_cols = _line_positions(col_sizes)
    row_rooms = _room_slices(h, horizontal_rows)
    col_rooms = _room_slices(w, vertical_cols)

    for c, side in zip(vertical_cols, _assign_edge_sides(len(vertical_cols), "top", "bottom")):
        grid[0 if side == "top" else h - 1, c] = Color.GRAY
    for r, side in zip(horizontal_rows, _assign_edge_sides(len(horizontal_rows), "left", "right")):
        grid[r, 0 if side == "left" else w - 1] = Color.GRAY

    _place_marker_block(grid, row_rooms[block_row_index], col_rooms[block_col_index])
    _validate_input(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 038/8")
