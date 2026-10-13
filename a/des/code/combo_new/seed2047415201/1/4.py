from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


AZURE = 8
GREY = 5
LEGEND_COLORS = (0, 1, 2, 3, 4, 6, 7, 9)
MAX_GRID = 30
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _legend_colors(g: np.ndarray) -> list[int]:
    """Read non-wall, non-background same-color legend components top-to-bottom."""
    h, w = g.shape
    mask = (g != AZURE) & (g != GREY)
    seen = np.zeros((h, w), dtype=bool)
    entries: list[tuple[int, int, int, int]] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        color = int(g[sr, sc])
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c, h, w):
                if not seen[nr, nc] and int(g[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        entries.append((min(rows), min(cols), max(rows), color))

    entries.sort(key=lambda x: (x[0], x[1], x[2]))
    return [color for _, _, _, color in entries]


def _wall_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    wall = g == GREY
    h, w = g.shape
    if w > 1:
        row_pairs = np.sum(wall[:, :-1] & wall[:, 1:], axis=1)
    else:
        row_pairs = np.zeros(h, dtype=int)
    if h > 1:
        col_pairs = np.sum(wall[:-1, :] & wall[1:, :], axis=0)
    else:
        col_pairs = np.zeros(w, dtype=int)
    rows = [int(r) for r in np.flatnonzero(row_pairs > 0)]
    cols = [int(c) for c in np.flatnonzero(col_pairs > 0)]
    return rows, cols


def _horizontal_doors(g: np.ndarray, r: int, c0: int, c1: int) -> list[tuple[int, int]] | None:
    if int(g[r, c0]) != GREY or int(g[r, c1]) != GREY:
        return None
    doors: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        v = int(g[r, c])
        if v == GREY:
            continue
        if v == AZURE and int(g[r, c - 1]) == GREY and int(g[r, c + 1]) == GREY:
            doors.append((r, c))
            continue
        return None
    return doors


def _vertical_doors(g: np.ndarray, c: int, r0: int, r1: int) -> list[tuple[int, int]] | None:
    if int(g[r0, c]) != GREY or int(g[r1, c]) != GREY:
        return None
    doors: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        v = int(g[r, c])
        if v == GREY:
            continue
        if v == AZURE and int(g[r - 1, c]) == GREY and int(g[r + 1, c]) == GREY:
            doors.append((r, c))
            continue
        return None
    return doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled rectangular room with the legend color indexed by its number of doors.

    Input structure:
    - The background is Azure and the one-pixel orthogonal wall network is Grey.
    - Grey wall rows and columns bound several rectangular rooms; single Azure gaps inside those wall segments are doors.
    - A lower-right vertical legend of colored squares gives the fill color for door counts 1, 2, 3, ... from top to bottom.

    Rule:
    - Reconstruct the wall-line grid and examine every rectangle between adjacent Grey wall rows and columns.
    - A room side may contain Grey wall pixels or single Azure door gaps whose two neighbors along the wall direction are Grey.
    - Count the distinct door pixels on the room boundary; use the N-th legend color for N doors and fill only the room interior Azure pixels.
    - Keep Grey walls, Azure door gaps, the outside background, and the legend unchanged.

    Key insight:
    Door count is a topological feature of a room boundary, and that count indexes the visible color legend.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    palette = _legend_colors(g)
    assert palette, "expected a lower-right color legend"

    wall_rows, wall_cols = _wall_lines(g)
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected rectangular wall lines"

    for ri in range(len(wall_rows) - 1):
        r0, r1 = wall_rows[ri], wall_rows[ri + 1]
        if r1 <= r0 + 1:
            continue
        for ci in range(len(wall_cols) - 1):
            c0, c1 = wall_cols[ci], wall_cols[ci + 1]
            if c1 <= c0 + 1:
                continue
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or not bool(np.all(interior == AZURE)):
                continue

            door_pixels: set[tuple[int, int]] = set()
            top = _horizontal_doors(g, r0, c0, c1)
            bottom = _horizontal_doors(g, r1, c0, c1)
            left = _vertical_doors(g, c0, r0, r1)
            right = _vertical_doors(g, c1, r0, r1)
            if top is None or bottom is None or left is None or right is None:
                continue
            door_pixels.update(top)
            door_pixels.update(bottom)
            door_pixels.update(left)
            door_pixels.update(right)

            door_count = len(door_pixels)
            if 1 <= door_count <= len(palette):
                region = out[r0 + 1 : r1, c0 + 1 : c1]
                region[interior == AZURE] = int(palette[door_count - 1])

    return out


# --------------- generator ---------------


def _line_positions(start: int, sizes: list[int]) -> list[int]:
    lines = [int(start)]
    pos = int(start)
    for size in sizes:
        pos += int(size) + 1
        lines.append(pos)
    return lines


def _room_side_keys(room_r: int, room_c: int) -> list[tuple[str, int, int]]:
    return [
        ("h", room_r, room_c),
        ("h", room_r + 1, room_c),
        ("v", room_r, room_c),
        ("v", room_r, room_c + 1),
    ]


def _is_internal_segment(key: tuple[str, int, int], n_rows: int, n_cols: int) -> bool:
    kind, a, _b = key
    if kind == "h":
        return 0 < a < n_rows
    return 0 < _b < n_cols


def _segment_length(key: tuple[str, int, int], row_lines: list[int], col_lines: list[int]) -> int:
    kind, a, b = key
    if kind == "h":
        return int(col_lines[b + 1] - col_lines[b] - 1)
    return int(row_lines[a + 1] - row_lines[a] - 1)


def _valid_door_offsets(length: int, count: int) -> list[tuple[int, ...]]:
    offsets = range(1, int(length) + 1)
    valid: list[tuple[int, ...]] = []
    for combo in combinations(offsets, int(count)):
        door_set = set(combo)
        if any((x - 1 in door_set) or (x + 1 in door_set) for x in combo):
            continue
        has_grey_pair = any(x not in door_set and x + 1 not in door_set for x in range(0, int(length) + 1))
        if has_grey_pair:
            valid.append(tuple(int(x) for x in combo))
    return valid


def _door_cells_for_segment(
    key: tuple[str, int, int], row_lines: list[int], col_lines: list[int], count: int
) -> list[tuple[int, int]] | None:
    length = _segment_length(key, row_lines, col_lines)
    choices = _valid_door_offsets(length, count)
    if not choices:
        return None
    offsets = random.choice(choices)
    kind, a, b = key
    cells: list[tuple[int, int]] = []
    if kind == "h":
        r = row_lines[a]
        c0 = col_lines[b]
        cells = [(int(r), int(c0 + off)) for off in offsets]
    else:
        r0 = row_lines[a]
        c = col_lines[b]
        cells = [(int(r0 + off), int(c)) for off in offsets]
    return cells


def _room_door_counts(
    n_rows: int, n_cols: int, segment_doors: dict[tuple[str, int, int], list[tuple[int, int]]]
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for rr in range(n_rows):
        for cc in range(n_cols):
            total = 0
            for key in _room_side_keys(rr, cc):
                total += len(segment_doors.get(key, []))
            counts[(rr, cc)] = int(total)
    return counts


def _add_one_door(
    key: tuple[str, int, int],
    row_lines: list[int],
    col_lines: list[int],
    segment_doors: dict[tuple[str, int, int], list[tuple[int, int]]],
) -> bool:
    if key not in segment_doors:
        cells = _door_cells_for_segment(key, row_lines, col_lines, 1)
        if cells is None:
            return False
        segment_doors[key] = cells
        return True

    if len(segment_doors[key]) == 1:
        cells = _door_cells_for_segment(key, row_lines, col_lines, 2)
        if cells is None:
            return False
        segment_doors[key] = cells
        return True
    return False


def _make_doors(
    n_rows: int, n_cols: int, row_lines: list[int], col_lines: list[int]
) -> dict[tuple[str, int, int], list[tuple[int, int]]] | None:
    internal: list[tuple[str, int, int]] = []
    for wall_r in range(1, n_rows):
        for room_c in range(n_cols):
            internal.append(("h", wall_r, room_c))
    for room_r in range(n_rows):
        for wall_c in range(1, n_cols):
            internal.append(("v", room_r, wall_c))

    segment_doors: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    p = random.uniform(0.22, 0.58)
    random.shuffle(internal)
    for key in internal:
        if random.random() >= p:
            continue
        count = 2 if random.random() < 0.18 else 1
        cells = _door_cells_for_segment(key, row_lines, col_lines, count)
        if cells is None:
            cells = _door_cells_for_segment(key, row_lines, col_lines, 1)
        if cells is not None:
            segment_doors[key] = cells

    for _ in range(20):
        counts = _room_door_counts(n_rows, n_cols, segment_doors)
        zeros = [room for room, count in counts.items() if count == 0]
        if not zeros:
            break
        random.shuffle(zeros)
        for room in zeros:
            choices = [key for key in _room_side_keys(*room) if _is_internal_segment(key, n_rows, n_cols)]
            random.shuffle(choices)
            if not any(_add_one_door(key, row_lines, col_lines, segment_doors) for key in choices):
                return None

    counts = _room_door_counts(n_rows, n_cols, segment_doors)
    if min(counts.values()) < 1:
        return None

    if len(set(counts.values())) < 2:
        candidates = internal[:]
        random.shuffle(candidates)
        if not any(_add_one_door(key, row_lines, col_lines, segment_doors) for key in candidates):
            return None
        counts = _room_door_counts(n_rows, n_cols, segment_doors)

    if min(counts.values()) < 1 or len(set(counts.values())) < 2 or max(counts.values()) > 6:
        return None
    return segment_doors


def _draw_walls(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> None:
    r0, r1 = row_lines[0], row_lines[-1]
    c0, c1 = col_lines[0], col_lines[-1]
    for r in row_lines:
        grid[r, c0 : c1 + 1] = GREY
    for c in col_lines:
        grid[r0 : r1 + 1, c] = GREY


def _draw_doors(
    grid: np.ndarray, segment_doors: dict[tuple[str, int, int], list[tuple[int, int]]]
) -> None:
    for cells in segment_doors.values():
        for r, c in cells:
            grid[r, c] = AZURE


def _draw_legend(grid: np.ndarray, colors: list[int], square: int, gap: int) -> None:
    h, w = grid.shape
    total_h = len(colors) * square + (len(colors) - 1) * gap
    r0 = h - total_h
    c0 = w - square
    for i, color in enumerate(colors):
        rr = r0 + i * (square + gap)
        grid[rr : rr + square, c0 : c0 + square] = int(color)


def _expected_output(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    n_rows: int,
    n_cols: int,
    segment_doors: dict[tuple[str, int, int], list[tuple[int, int]]],
    legend: list[int],
) -> np.ndarray:
    out = grid.copy()
    counts = _room_door_counts(n_rows, n_cols, segment_doors)
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0, r1 = row_lines[rr], row_lines[rr + 1]
            c0, c1 = col_lines[cc], col_lines[cc + 1]
            color = int(legend[counts[(rr, cc)] - 1])
            out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.choice([2, 3, 3])
    n_cols = random.choice([2, 3, 3, 4])
    room_heights = [random.randint(3, 6) for _ in range(n_rows)]
    room_widths = [random.randint(3, 6) for _ in range(n_cols)]

    network_h = sum(room_heights) + n_rows + 1
    network_w = sum(room_widths) + n_cols + 1
    top = random.randint(1, 3)
    left = random.randint(1, 3)

    row_lines = _line_positions(top, room_heights)
    col_lines = _line_positions(left, room_widths)
    segment_doors = _make_doors(n_rows, n_cols, row_lines, col_lines)
    if segment_doors is None:
        return None

    counts = _room_door_counts(n_rows, n_cols, segment_doors)
    max_count = max(counts.values())
    if max_count > len(LEGEND_COLORS):
        return None

    legend_n = random.randint(max_count, min(len(LEGEND_COLORS), max_count + random.choice([0, 1, 2])))
    legend_colors = random.sample(list(LEGEND_COLORS), legend_n)
    square = 1 if legend_n > 5 else random.choice([1, 2])
    gap = random.choice([0, 1])
    legend_h = legend_n * square + (legend_n - 1) * gap

    right_margin = random.randint(square + 2, square + 6)
    bottom_margin = random.randint(1, 4)
    height = max(top + network_h + bottom_margin, legend_h + random.randint(1, 4))
    width = left + network_w + right_margin
    if height > MAX_GRID or width > MAX_GRID:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    _draw_walls(grid, row_lines, col_lines)
    _draw_doors(grid, segment_doors)
    _draw_legend(grid, legend_colors, square, gap)

    expected = _expected_output(
        grid, row_lines, col_lines, n_rows, n_cols, segment_doors, legend_colors
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
