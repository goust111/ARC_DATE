import random
from collections import defaultdict

import numpy as np


BLACK = 0
MAROON = 9
PALETTE_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 300


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r1 < r0 or c1 < c0:
        return 0
    total = int(prefix[r1, c1])
    if r0:
        total -= int(prefix[r0 - 1, c1])
    if c0:
        total -= int(prefix[r1, c0 - 1])
    if r0 and c0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _row_sum(prefix: np.ndarray, r: int, c0: int, c1: int) -> int:
    total = int(prefix[r, c1])
    if c0:
        total -= int(prefix[r, c0 - 1])
    return total


def _col_sum(prefix: np.ndarray, c: int, r0: int, r1: int) -> int:
    total = int(prefix[r1, c])
    if r0:
        total -= int(prefix[r0 - 1, c])
    return total


def _palette_info(grid: np.ndarray) -> tuple[int, int, list[int]]:
    h, _ = grid.shape
    palette_h = 0
    for r in range(h - 1, -1, -1):
        row = grid[r]
        if np.any((row != BLACK) & (row != MAROON)):
            palette_h += 1
        else:
            break
    assert 1 <= palette_h <= 2, "expected a 1-row or 2-row palette band at the bottom"

    board_h = h - palette_h
    ref = grid[board_h]
    mask = (ref != BLACK) & (ref != MAROON)
    cols = np.flatnonzero(mask)
    assert cols.size > 0, "palette row must contain at least one swatch"
    c0, c1 = int(cols[0]), int(cols[-1])

    palette = [int(v) for v in ref[c0 : c1 + 1] if v not in (BLACK, MAROON)]
    assert palette, "palette must be non-empty"
    assert len(palette) == len(set(palette)), "palette swatches must be distinct"

    for r in range(board_h, h):
        assert not np.any(grid[r] == MAROON), "palette band must not contain maroon walls"
        row_palette = [int(v) for v in grid[r, c0 : c1 + 1] if v not in (BLACK, MAROON)]
        assert row_palette == palette, "palette rows must agree"

    return board_h, palette_h, palette


def _find_rooms(board: np.ndarray, palette_len: int) -> list[tuple[int, int, int, int, int]]:
    maroon = board == MAROON
    h, w = board.shape
    area_ps = maroon.cumsum(axis=0).cumsum(axis=1)
    row_ps = maroon.cumsum(axis=1)
    col_ps = maroon.cumsum(axis=0)

    rooms: list[tuple[int, int, int, int, int]] = []
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            if not maroon[r0, c0]:
                continue
            for r1 in range(r0 + 3, h):
                if not maroon[r1, c0]:
                    continue
                for c1 in range(c0 + 3, w):
                    if not maroon[r0, c1] or not maroon[r1, c1]:
                        continue
                    if _rect_sum(area_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue

                    top = _row_sum(row_ps, r0, c0, c1)
                    bottom = _row_sum(row_ps, r1, c0, c1)
                    left = _col_sum(col_ps, c0, r0, r1)
                    right = _col_sum(col_ps, c1, r0, r1)

                    border_cells = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
                    maroon_border = top + bottom + left + right - 4
                    windows = border_cells - maroon_border
                    if 1 <= windows <= palette_len:
                        rooms.append((r0, c0, r1, c1, windows))

    assert rooms, "expected at least one room"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each maroon-walled rectangular room with the palette color indexed by its window count.

    Input structure:
    - The background is black.
    - One-cell-thick maroon walls outline one or more axis-aligned rectangular rooms.
    - Some wall cells are missing and appear as black single-cell windows.
    - The bottom one or two rows form a palette band whose swatches, read left-to-right, encode the fill color for 1 window, 2 windows, and so on.

    Rule:
    - Detect every rectangular room whose border is made of maroon cells with possible black gaps.
    - Count the black gap cells on that room's four border segments, excluding the four corners.
    - Use that count as a 1-based index into the palette band.
    - Fill the room's interior black cells with the indexed palette color, while leaving maroon walls, black windows, and the palette band unchanged.

    Key insight:
    - The black gaps on a room's border are not ordinary background; they are the hidden numeric key that selects the room's fill color from the bottom palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    board_h, _palette_h, palette = _palette_info(g)
    board = g[:board_h]
    assert np.all((board == BLACK) | (board == MAROON)), "board area must use only black and maroon"

    out = g.copy()
    rooms = _find_rooms(board, len(palette))
    for r0, c0, r1, c1, windows in rooms:
        color = int(palette[windows - 1])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = color

    return out


def _boundary_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    positions = []
    positions.extend((r0, c) for c in range(c0 + 1, c1))
    positions.extend((r1, c) for c in range(c0 + 1, c1))
    positions.extend((r, c0) for r in range(r0 + 1, r1))
    positions.extend((r, c1) for r in range(r0 + 1, r1))
    return positions


def _non_adjacent_pick(
    rng: random.Random, positions: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    shuffled = positions[:]
    rng.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for pos in shuffled:
        if all(abs(pos[0] - rr) + abs(pos[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return chosen
    return None


def _room_counts_from_windows(
    room_bboxes: list[tuple[int, int, int, int]], windows: set[tuple[int, int]]
) -> list[int]:
    counts = []
    for bbox in room_bboxes:
        counts.append(sum(pos in windows for pos in _boundary_positions(bbox)))
    return counts


def _try_disjoint_layout(
    rng: random.Random, palette: list[int], palette_h: int
) -> np.ndarray | None:
    palette_len = len(palette)
    room_count = rng.randint(2, 5)
    board_h = rng.randint(12, 24)
    board_w = rng.randint(max(12, palette_len + 2), 28)
    if board_h + palette_h > 30:
        return None

    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(room_count):
        placed = False
        for _ in range(80):
            height = rng.randint(4, min(9, board_h - 2))
            width = rng.randint(4, min(9, board_w - 2))
            if height <= 3 or width <= 3:
                continue
            max_windows_here = min(palette_len, len(_boundary_positions((0, 0, height - 1, width - 1))) // 2)
            if max_windows_here < 1:
                continue
            r0 = rng.randint(0, board_h - height)
            c0 = rng.randint(0, board_w - width)
            r1, c1 = r0 + height - 1, c0 + width - 1

            ok = True
            for or0, oc0, or1, oc1 in bboxes:
                if not (r1 + 2 < or0 or or1 + 2 < r0 or c1 + 2 < oc0 or oc1 + 2 < c0):
                    ok = False
                    break
            if ok:
                bboxes.append((r0, c0, r1, c1))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((board_h + palette_h, board_w), BLACK, dtype=int)
    windows: set[tuple[int, int]] = set()
    counts: list[int] = []

    for bbox in bboxes:
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = MAROON
        grid[r1, c0 : c1 + 1] = MAROON
        grid[r0 : r1 + 1, c0] = MAROON
        grid[r0 : r1 + 1, c1] = MAROON

        boundary = _boundary_positions(bbox)
        max_count = min(palette_len, len(boundary) // 2)
        count = rng.randint(1, max_count)
        chosen = _non_adjacent_pick(rng, boundary, count)
        if chosen is None:
            return None
        windows.update(chosen)
        counts.append(count)

    for r, c in windows:
        grid[r, c] = BLACK

    if len(set(counts)) == 1 and palette_len > 1:
        return None

    palette_row = np.full(board_w, BLACK, dtype=int)
    offset = rng.randint(0, board_w - palette_len)
    palette_row[offset : offset + palette_len] = palette
    grid[board_h] = palette_row
    if palette_h == 2:
        grid[board_h + 1] = palette_row

    return grid


def _candidate_window_map(
    room_bboxes: list[tuple[int, int, int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, ...]], list[list[int]]]:
    cell_to_rooms: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, bbox in enumerate(room_bboxes):
        for cell in _boundary_positions(bbox):
            cell_to_rooms[cell].add(idx)

    cells = sorted(cell_to_rooms)
    room_sets = [tuple(sorted(cell_to_rooms[cell])) for cell in cells]
    by_room: list[list[int]] = [[] for _ in room_bboxes]
    for idx, rooms in enumerate(room_sets):
        for room in rooms:
            by_room[room].append(idx)
    return cells, room_sets, by_room


def _try_partition_layout(
    rng: random.Random, palette: list[int], palette_h: int
) -> np.ndarray | None:
    palette_len = len(palette)
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    cell_heights = [rng.randint(2, 5) for _ in range(rows)]
    cell_widths = [rng.randint(2, 5) for _ in range(cols)]
    board_h = sum(cell_heights) + rows + 1
    board_w = sum(cell_widths) + cols + 1
    if board_h + palette_h > 30 or board_w > 30:
        return None

    wall_rows = [0]
    for height in cell_heights:
        wall_rows.append(wall_rows[-1] + height + 1)
    wall_cols = [0]
    for width in cell_widths:
        wall_cols.append(wall_cols[-1] + width + 1)

    grid = np.full((board_h + palette_h, board_w), BLACK, dtype=int)
    for r in wall_rows:
        grid[r, :] = MAROON
    for c in wall_cols:
        grid[:board_h, c] = MAROON

    room_bboxes = [
        (wall_rows[i], wall_cols[j], wall_rows[i + 1], wall_cols[j + 1])
        for i in range(rows)
        for j in range(cols)
    ]
    cells, room_sets, by_room = _candidate_window_map(room_bboxes)
    counts = [0] * len(room_bboxes)
    windows: set[int] = set()

    def can_open(cell_idx: int) -> bool:
        if cell_idx in windows:
            return False
        return all(counts[room] < palette_len for room in room_sets[cell_idx])

    def open_cell(cell_idx: int) -> None:
        windows.add(cell_idx)
        for room in room_sets[cell_idx]:
            counts[room] += 1

    room_order = list(range(len(room_bboxes)))
    rng.shuffle(room_order)
    for room in room_order:
        choices = [idx for idx in by_room[room] if can_open(idx)]
        if not choices:
            return None
        open_cell(rng.choice(choices))

    extra_steps = rng.randint(len(room_bboxes) // 2, len(room_bboxes) * 2)
    extra_order = list(range(len(cells)))
    rng.shuffle(extra_order)
    for cell_idx in extra_order:
        if extra_steps <= 0:
            break
        if rng.random() < 0.5 and can_open(cell_idx):
            open_cell(cell_idx)
            extra_steps -= 1

    if max(counts) > palette_len or min(counts) < 1:
        return None
    if len(set(counts)) == 1 and palette_len > 1:
        return None

    for cell_idx in windows:
        r, c = cells[cell_idx]
        grid[r, c] = BLACK

    palette_row = np.full(board_w, BLACK, dtype=int)
    offset = rng.randint(0, board_w - palette_len)
    palette_row[offset : offset + palette_len] = palette
    grid[board_h] = palette_row
    if palette_h == 2:
        grid[board_h + 1] = palette_row

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_h = rng.choice([1, 2])
    palette_len = rng.randint(3, 5)
    palette = rng.sample(PALETTE_POOL, palette_len)

    if rng.random() < 0.5:
        grid = _try_partition_layout(rng, palette, palette_h)
    else:
        grid = _try_disjoint_layout(rng, palette, palette_h)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-window puzzle")
