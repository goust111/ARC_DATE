from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 5
WALL = 9
PATTERN_COLORS = tuple(c for c in range(10) if c not in (BG, WALL))
MAX_GRID_SIDE = 30
GEN_RETRIES = 400


def _horizontal_wall_rows(grid: np.ndarray) -> list[int]:
    width = int(grid.shape[1])
    threshold = max(3, width // 2)
    return [int(r) for r in range(grid.shape[0]) if int(np.count_nonzero(grid[r] == WALL)) >= threshold]


def _vertical_wall_cols(grid: np.ndarray) -> list[int]:
    height = int(grid.shape[0])
    threshold = max(3, height // 2)
    return [int(c) for c in range(grid.shape[1]) if int(np.count_nonzero(grid[:, c] == WALL)) >= threshold]


def _room_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h_lines = _horizontal_wall_rows(grid)
    v_lines = _vertical_wall_cols(grid)
    assert len(h_lines) >= 2 and len(v_lines) >= 2, "could not identify the room lattice"

    boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in zip(h_lines, h_lines[1:], strict=False):
        if r1 - r0 <= 1:
            continue
        for c0, c1 in zip(v_lines, v_lines[1:], strict=False):
            if c1 - c0 <= 1:
                continue
            boxes.append((int(r0), int(r1), int(c0), int(c1)))
    assert boxes, "no rooms found"
    return boxes


def _find_gap(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r0, r1, c0, c1 = box
    candidates = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }

    found: list[tuple[str, tuple[int, int]]] = []
    for side, cells in candidates.items():
        gaps = [pos for pos in cells if int(grid[pos]) != WALL]
        if gaps:
            assert len(gaps) == 1, "each side may contain at most one gap"
            found.append((side, gaps[0]))

    assert len(found) == 1, "each room must have exactly one gap"
    return found[0]


def _pattern_pixels(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    r0, r1, c0, c1 = box
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    rel = np.argwhere((inner != BG) & (inner != WALL))
    assert rel.size, "room must contain a pattern"
    rows = rel[:, 0].astype(int) + int(r0 + 1)
    cols = rel[:, 1].astype(int) + int(c0 + 1)
    colors = [int(grid[int(r), int(c)]) for r, c in zip(rows, cols, strict=False)]
    return rows, cols, colors


def _marker_cells(side: str, gap: tuple[int, int], length: int) -> Iterable[tuple[int, int]]:
    r, c = gap
    if side == "top":
        for dr in range(length):
            yield int(r + dr), int(c)
    elif side == "bottom":
        for dr in range(length):
            yield int(r - dr), int(c)
    elif side == "left":
        for dc in range(length):
            yield int(r), int(c + dc)
    else:
        for dc in range(length):
            yield int(r), int(c - dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each room pattern away from its wall gap and encode color-count with a maroon marker bar.

    Input structure:
    - Grey background.
    - Maroon 1-pixel wall lines partition the canvas into rectangular rooms.
    - Every room contains one multi-pixel pattern using 1-3 non-grey, non-maroon colors.
    - Each room boundary has exactly one 1-pixel gap on one side.

    Rule:
    - Find the side containing the gap for each room.
    - Translate the whole room pattern toward the side opposite the gap until it touches that wall.
    - Let N be the number of distinct colors used in the pattern.
    - Draw a maroon bar of length N from the gap inward along the same axis.
    """
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    for r0, r1, c0, c1 in _room_boxes(grid):
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

    for box in _room_boxes(grid):
        r0, r1, c0, c1 = box
        side, gap = _find_gap(grid, box)
        rows, cols, colors = _pattern_pixels(grid, box)
        color_count = len(set(colors))
        assert 1 <= color_count <= 3

        if side == "top":
            dr = int((r1 - 1) - rows.max())
            dc = 0
        elif side == "bottom":
            dr = int((r0 + 1) - rows.min())
            dc = 0
        elif side == "left":
            dr = 0
            dc = int((c1 - 1) - cols.max())
        else:
            dr = 0
            dc = int((c0 + 1) - cols.min())

        for r, c, color in zip(rows, cols, colors, strict=False):
            out[int(r + dr), int(c + dc)] = int(color)

        for rr, cc in _marker_cells(side, gap, color_count):
            assert r0 <= rr <= r1 and c0 <= cc <= c1
            out[int(rr), int(cc)] = WALL

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _grow_connected_shape(height: int, width: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}

    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, height, width):
                if (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _partition_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    lengths: list[int] = []
    for cut in cuts + [total]:
        lengths.append(int(cut - prev))
        prev = cut
    return lengths


def _colorize_shape(cells: set[tuple[int, int]], n_colors: int) -> np.ndarray:
    height = max(r for r, _c in cells) + 1
    width = max(c for _r, c in cells) + 1
    patch = np.full((height, width), BG, dtype=int)
    colors = random.sample(PATTERN_COLORS, n_colors)

    if n_colors == 1:
        for r, c in cells:
            patch[r, c] = colors[0]
        return patch

    key_mode = random.choice(("row", "col", "diag", "anti"))
    if key_mode == "row":
        ordered = sorted(cells, key=lambda rc: (rc[0], rc[1]))
    elif key_mode == "col":
        ordered = sorted(cells, key=lambda rc: (rc[1], rc[0]))
    elif key_mode == "diag":
        ordered = sorted(cells, key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
    else:
        ordered = sorted(cells, key=lambda rc: (rc[0] - rc[1], rc[0], rc[1]))

    lengths = _partition_positive(len(ordered), n_colors)
    idx = 0
    for color, length in zip(colors, lengths, strict=False):
        for r, c in ordered[idx : idx + length]:
            patch[r, c] = int(color)
        idx += length
    return patch


def _make_pattern(room_h: int, room_w: int, side: str) -> np.ndarray | None:
    axis_vertical = side in ("top", "bottom")
    n_colors = random.randint(1, 3)

    max_h = room_h - 2
    max_w = room_w - 2
    if axis_vertical:
        max_h = min(max_h, room_h - n_colors)
    else:
        max_w = min(max_w, room_w - n_colors)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(80):
        box_h = random.randint(2, max_h)
        box_w = random.randint(2, max_w)
        max_area = box_h * box_w
        min_area = max(n_colors, (max_area + 2) // 3)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        cells = _grow_connected_shape(box_h, box_w, area)
        if cells is None:
            continue

        actual_h = max(r for r, _c in cells) + 1
        actual_w = max(c for _r, c in cells) + 1
        if actual_h < 2 or actual_w < 2:
            continue
        if actual_h > room_h - 2 or actual_w > room_w - 2:
            continue
        if axis_vertical and actual_h > room_h - n_colors:
            continue
        if (not axis_vertical) and actual_w > room_w - n_colors:
            continue

        return _colorize_shape(cells, n_colors)

    return None


def _sample_layout() -> tuple[list[int], list[int]] | None:
    mode = random.choice(("single_row", "double_row", "single_col", "double_col"))
    if mode == "single_row":
        n_rows, n_cols = 1, random.randint(2, 4)
    elif mode == "double_row":
        n_rows, n_cols = 2, random.randint(2, 4)
    elif mode == "single_col":
        n_rows, n_cols = random.randint(2, 4), 1
    else:
        n_rows, n_cols = random.randint(2, 4), 2

    heights = [random.randint(6, 9) for _ in range(n_rows)]
    widths = [random.randint(6, 9) for _ in range(n_cols)]
    total_h = sum(heights) + n_rows + 1
    total_w = sum(widths) + n_cols + 1
    if total_h > MAX_GRID_SIDE or total_w > MAX_GRID_SIDE:
        return None
    return heights, widths


def _line_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cur = 0
    for size in sizes:
        cur += int(size) + 1
        positions.append(int(cur))
    return positions


def _allowed_sides(row: int, col: int, n_rows: int, n_cols: int) -> list[str]:
    sides: list[str] = []
    if row == 0:
        sides.append("top")
    if row == n_rows - 1:
        sides.append("bottom")
    if col == 0:
        sides.append("left")
    if col == n_cols - 1:
        sides.append("right")
    return sides


def _place_gap(grid: np.ndarray, box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, r1, c0, c1 = box
    if side == "top":
        gap = (r0, random.randint(c0 + 1, c1 - 1))
    elif side == "bottom":
        gap = (r1, random.randint(c0 + 1, c1 - 1))
    elif side == "left":
        gap = (random.randint(r0 + 1, r1 - 1), c0)
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c1)
    grid[gap] = BG
    return gap


def _embed_patch(grid: np.ndarray, room_top: int, room_left: int, room_h: int, room_w: int, patch: np.ndarray) -> None:
    patch_h, patch_w = map(int, patch.shape)
    top = random.randint(1, room_h - patch_h - 1)
    left = random.randint(1, room_w - patch_w - 1)
    rr = room_top + top
    cc = room_left + left
    mask = patch != BG
    grid[rr : rr + patch_h, cc : cc + patch_w][mask] = patch[mask]


def _build_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    heights, widths = layout
    n_rows, n_cols = len(heights), len(widths)
    h_lines = _line_positions(heights)
    v_lines = _line_positions(widths)

    grid = np.full((h_lines[-1] + 1, v_lines[-1] + 1), BG, dtype=int)
    grid[h_lines, :] = WALL
    grid[:, v_lines] = WALL

    for ri in range(n_rows):
        for ci in range(n_cols):
            r0, r1 = h_lines[ri], h_lines[ri + 1]
            c0, c1 = v_lines[ci], v_lines[ci + 1]
            room_h = r1 - r0 - 1
            room_w = c1 - c0 - 1
            side = random.choice(_allowed_sides(ri, ci, n_rows, n_cols))
            patch = _make_pattern(room_h, room_w, side)
            if patch is None:
                return None
            _place_gap(grid, (r0, r1, c0, c1), side)
            _embed_patch(grid, r0, c0, room_h, room_w, patch)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
