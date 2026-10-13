from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
REGION_COLORS = (1, 2, 3, 4, 6, 7, 8)
SIDE_PAIRS = (
    ("top", "bottom"),
    ("top", "right"),
    ("left", "bottom"),
    ("left", "right"),
)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _concat_paths(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        if not part:
            continue
        if out and out[-1] == part[0]:
            out.extend(part[1:])
        else:
            out.extend(part)
    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _connect_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a == b:
        return [a]
    if a[0] == b[0] or a[1] == b[1]:
        return _line_cells(a, b)

    bends = [(a[0], b[1]), (b[0], a[1])]
    random.shuffle(bends)
    bend = bends[0]
    return _concat_paths(_line_cells(a, bend), _line_cells(bend, b))


def _detect_board_and_bounds(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    board_mask = (grid != BLACK) & (grid != MAROON)
    coords = np.argwhere(board_mask)
    assert coords.size > 0, "expected a non-black partition board"

    br0, bc0 = map(int, coords.min(axis=0))
    br1, bc1 = map(int, coords.max(axis=0))
    board = grid[br0 : br1 + 1, bc0 : bc1 + 1]

    row_bounds: list[int] = []
    for r in range(board.shape[0]):
        row = board[r]
        if np.all(np.isin(row, (GREY, MAROON))):
            grey = int(np.count_nonzero(row == GREY))
            maroon = int(np.count_nonzero(row == MAROON))
            if grey > maroon:
                row_bounds.append(br0 + r)

    col_bounds: list[int] = []
    for c in range(board.shape[1]):
        col = board[:, c]
        if np.all(np.isin(col, (GREY, MAROON))):
            grey = int(np.count_nonzero(col == GREY))
            maroon = int(np.count_nonzero(col == MAROON))
            if grey > maroon:
                col_bounds.append(bc0 + c)

    assert len(row_bounds) >= 2 and len(col_bounds) >= 2, "expected framed partitions"
    return (br0, br1, bc0, bc1), row_bounds, col_bounds


def _extract_regions(
    grid: np.ndarray, row_bounds: list[int], col_bounds: list[int]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]]:
    region_map = np.full(grid.shape, -1, dtype=int)
    rects: list[tuple[int, int, int, int]] = []
    colors: list[int] = []
    region_id = 0

    for top, bottom in zip(row_bounds, row_bounds[1:]):
        for left, right in zip(col_bounds, col_bounds[1:]):
            r0, r1 = top + 1, bottom - 1
            c0, c1 = left + 1, right - 1
            if r0 > r1 or c0 > c1:
                continue

            region_map[r0 : r1 + 1, c0 : c1 + 1] = region_id
            vals = [
                int(v)
                for v in grid[r0 : r1 + 1, c0 : c1 + 1].ravel()
                if v not in (BLACK, GREY, MAROON)
            ]
            assert vals, "path must not cover an entire region"
            uniq, counts = np.unique(vals, return_counts=True)
            rects.append((r0, r1, c0, c1))
            colors.append(int(uniq[int(np.argmax(counts))]))
            region_id += 1

    return region_map, rects, colors


def _path_neighbors(
    path_cells: set[tuple[int, int]], cell: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if (nr, nc) in path_cells:
            out.append((nr, nc))
    return out


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    path_cells = set(map(tuple, np.argwhere(grid == MAROON)))
    assert path_cells, "expected a maroon path"

    endpoints = [
        cell for cell in path_cells if len(_path_neighbors(path_cells, cell)) == 1
    ]
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    starts = [cell for cell in endpoints if cell[0] == 0 or cell[1] == 0]
    start = starts[0] if len(starts) == 1 else min(endpoints)

    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [cell for cell in _path_neighbors(path_cells, cur) if cell != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must not branch"
        prev, cur = cur, nxt[0]
        order.append(cur)

    assert len(order) == len(path_cells), "path must be connected"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cyclically shift the colors of the rectangular partitions visited by the maroon path.

    Input structure:
    - Black background surrounds a grey-framed rectangular tiling split by full grey
      horizontal and vertical separators into colored rectangular partitions.
    - Each partition is uniformly filled with one non-black, non-grey color, though
      different partitions may share a color.
    - A single maroon orthogonal path starts on the top or left grid edge, enters the
      framed board, crosses separator lines between partitions, and exits on the bottom
      or right grid edge.

    Rule:
    - Trace the maroon path from its edge entry to its edge exit and record the visited
      partition order.
    - Read the original colors of those visited partitions.
    - Recolor each visited partition with the next visited partition's original color,
      wrapping the last partition back to the first color.
    - Remove the path: cells outside the frame return to black, cells on grey separator
      lines return to grey, and cells inside partitions take the partition's new color.

    Key insight:
    - The maroon path is an ordered tour through the partition graph, so the real
      transformation is a cyclic recoloring of whole partitions rather than a repaint
      of the path pixels themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (br0, br1, bc0, bc1), row_bounds, col_bounds = _detect_board_and_bounds(g)
    region_map, rects, colors = _extract_regions(g, row_bounds, col_bounds)
    path_order = _trace_path(g)

    region_sequence: list[int] = []
    seen_regions: set[int] = set()
    for cell in path_order:
        region_id = int(region_map[cell])
        if region_id >= 0 and (not region_sequence or region_sequence[-1] != region_id):
            assert region_id not in seen_regions, "path may visit each region at most once"
            region_sequence.append(region_id)
            seen_regions.add(region_id)

    assert len(region_sequence) >= 2, "path must traverse multiple partitions"
    shifted = {
        region_sequence[i]: colors[region_sequence[(i + 1) % len(region_sequence)]]
        for i in range(len(region_sequence))
    }

    out = np.full_like(g, BLACK)
    for r in row_bounds:
        out[r, bc0 : bc1 + 1] = GREY
    for c in col_bounds:
        out[br0 : br1 + 1, c] = GREY
    for region_id, (r0, r1, c0, c1) in enumerate(rects):
        out[r0 : r1 + 1, c0 : c1 + 1] = shifted.get(region_id, colors[region_id])

    return out


def _touches_side(cell: tuple[int, int], n_rows: int, n_cols: int, side: str) -> bool:
    r, c = cell
    if side == "top":
        return r == 0
    if side == "bottom":
        return r == n_rows - 1
    if side == "left":
        return c == 0
    if side == "right":
        return c == n_cols - 1
    raise AssertionError("unknown side")


def _region_neighbors(
    cell: tuple[int, int], n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            out.append((nr, nc))
    random.shuffle(out)
    return out


def _sample_region_path(
    n_rows: int, n_cols: int, start_side: str, end_side: str
) -> list[tuple[int, int]] | None:
    total = n_rows * n_cols
    lengths = list(range(min(8, total), 2, -1))
    random.shuffle(lengths)

    starts = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if _touches_side((r, c), n_rows, n_cols, start_side)
    ]
    random.shuffle(starts)

    for target_len in lengths:
        for start in starts:

            def dfs(
                path: list[tuple[int, int]], seen: set[tuple[int, int]]
            ) -> list[tuple[int, int]] | None:
                cur = path[-1]
                if len(path) == target_len:
                    if _touches_side(cur, n_rows, n_cols, end_side) and cur != start:
                        return path
                    return None

                for nxt in _region_neighbors(cur, n_rows, n_cols):
                    if nxt in seen:
                        continue
                    res = dfs(path + [nxt], seen | {nxt})
                    if res is not None:
                        return res
                return None

            result = dfs([start], {start})
            if result is not None:
                return result

    return None


def _outer_port(
    rect: tuple[int, int, int, int], side: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, r1, c0, c1 = rect
    if side == "top":
        c = random.randint(c0, c1)
        return (r0 - 1, c), (r0 - 2, c)
    if side == "bottom":
        c = random.randint(c0, c1)
        return (r1 + 1, c), (r1 + 2, c)
    if side == "left":
        r = random.randint(r0, r1)
        return (r, c0 - 1), (r, c0 - 2)
    if side == "right":
        r = random.randint(r0, r1)
        return (r, c1 + 1), (r, c1 + 2)
    raise AssertionError("unknown side")


def _shared_port(
    rects: dict[tuple[int, int], tuple[int, int, int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[tuple[int, int], tuple[str, str]]:
    ar0, ar1, ac0, ac1 = rects[a]
    ar, ac = a
    br, bc = b

    if ar == br and bc == ac + 1:
        r = random.randint(ar0, ar1)
        return (r, ac1 + 1), ("right", "left")
    if ar == br and bc == ac - 1:
        r = random.randint(ar0, ar1)
        return (r, ac0 - 1), ("left", "right")
    if ac == bc and br == ar + 1:
        c = random.randint(ac0, ac1)
        return (ar1 + 1, c), ("bottom", "top")
    if ac == bc and br == ar - 1:
        c = random.randint(ac0, ac1)
        return (ar0 - 1, c), ("top", "bottom")
    raise AssertionError("regions must be adjacent")


def _inside_neighbor(
    rect: tuple[int, int, int, int], side: str, boundary_cell: tuple[int, int]
) -> tuple[int, int]:
    r0, r1, c0, c1 = rect
    r, c = boundary_cell
    if side == "top":
        return (r0, c)
    if side == "bottom":
        return (r1, c)
    if side == "left":
        return (r, c0)
    if side == "right":
        return (r, c1)
    raise AssertionError("unknown side")


def _region_segment(
    rect: tuple[int, int, int, int],
    entry_side: str,
    entry_cell: tuple[int, int],
    exit_side: str,
    exit_cell: tuple[int, int],
) -> list[tuple[int, int]]:
    start = _inside_neighbor(rect, entry_side, entry_cell)
    end = _inside_neighbor(rect, exit_side, exit_cell)
    return _concat_paths([entry_cell], _connect_points(start, end), [exit_cell])


def _gen_attempt() -> np.ndarray | None:
    n_region_rows = random.randint(2, 4)
    n_region_cols = random.randint(2, 4)
    heights = [random.randint(2, 5) for _ in range(n_region_rows)]
    widths = [random.randint(2, 5) for _ in range(n_region_cols)]

    board_h = sum(heights) + n_region_rows + 1
    board_w = sum(widths) + n_region_cols + 1
    H, W = board_h + 2, board_w + 2
    if H > 30 or W > 30:
        return None

    row_bounds = [1]
    for h in heights:
        row_bounds.append(row_bounds[-1] + h + 1)
    col_bounds = [1]
    for w in widths:
        col_bounds.append(col_bounds[-1] + w + 1)

    rects: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(n_region_rows):
        for c in range(n_region_cols):
            rects[(r, c)] = (
                row_bounds[r] + 1,
                row_bounds[r + 1] - 1,
                col_bounds[c] + 1,
                col_bounds[c + 1] - 1,
            )

    start_side, end_side = random.choice(SIDE_PAIRS)
    region_path = _sample_region_path(
        n_region_rows, n_region_cols, start_side, end_side
    )
    if region_path is None:
        return None

    colors = {cell: random.choice(REGION_COLORS) for cell in rects}
    visited_colors = [colors[cell] for cell in region_path]
    if len(set(visited_colors)) < 2:
        choices = [c for c in REGION_COLORS if c != colors[region_path[0]]]
        colors[region_path[-1]] = random.choice(choices)

    grid = np.full((H, W), BLACK, dtype=int)
    board_r0, board_r1 = 1, H - 2
    board_c0, board_c1 = 1, W - 2
    for r in row_bounds:
        grid[r, board_c0 : board_c1 + 1] = GREY
    for c in col_bounds:
        grid[board_r0 : board_r1 + 1, c] = GREY
    for cell, rect in rects.items():
        r0, r1, c0, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = colors[cell]

    entry: dict[tuple[int, int], tuple[str, tuple[int, int]]] = {}
    exit_: dict[tuple[int, int], tuple[str, tuple[int, int]]] = {}
    start_frame, start_edge = _outer_port(rects[region_path[0]], start_side)
    end_frame, end_edge = _outer_port(rects[region_path[-1]], end_side)
    entry[region_path[0]] = (start_side, start_frame)
    exit_[region_path[-1]] = (end_side, end_frame)

    for a, b in zip(region_path, region_path[1:]):
        cross_cell, (side_a, side_b) = _shared_port(rects, a, b)
        exit_[a] = (side_a, cross_cell)
        entry[b] = (side_b, cross_cell)

    path_cells = [start_edge]
    for cell in region_path:
        seg = _region_segment(
            rects[cell],
            entry[cell][0],
            entry[cell][1],
            exit_[cell][0],
            exit_[cell][1],
        )
        path_cells = _concat_paths(path_cells, seg)

        r0, r1, c0, c1 = rects[cell]
        covered = {(r, c) for r, c in seg if r0 <= r <= r1 and c0 <= c <= c1}
        if len(covered) >= (r1 - r0 + 1) * (c1 - c0 + 1):
            return None

    path_cells.append(end_edge)
    if len(set(path_cells)) != len(path_cells):
        return None

    for r, c in path_cells:
        grid[r, c] = MAROON

    bbox, found_rows, found_cols = _detect_board_and_bounds(grid)
    if bbox != (board_r0, board_r1, board_c0, board_c1):
        return None
    if found_rows != row_bounds or found_cols != col_bounds:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
