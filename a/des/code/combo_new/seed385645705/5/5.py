from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np


class Partition(NamedTuple):
    shape: tuple[int, int]
    sep_color: int
    sep_rows: tuple[int, ...]
    sep_cols: tuple[int, ...]
    row_spans: tuple[tuple[int, int], ...]
    col_spans: tuple[tuple[int, int], ...]
    cell_h: int
    cell_w: int


BG = 0
SEP = 9
INTERFACE_POOL = (1, 2, 3, 4, 5)
PATTERN_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_RETRIES = 400


def _axis_partition(size: int, separators: list[int]) -> tuple[tuple[tuple[int, int], ...], int] | None:
    separators = sorted(set(int(x) for x in separators))
    if not separators:
        return None

    # Bordered grids: separator lines frame the whole tiling.
    if len(separators) >= 3 and separators[0] == 0 and separators[-1] == size - 1:
        diffs = [int(separators[i + 1] - separators[i]) for i in range(len(separators) - 1)]
        if diffs and len(set(diffs)) == 1:
            cell_size = int(diffs[0] - 1)
            spans = tuple((int(separators[i] + 1), int(separators[i + 1])) for i in range(len(separators) - 1))
            if len(spans) >= 2 and cell_size >= 3 and cell_size % 2 == 1:
                return spans, cell_size

    # Borderless grids: use virtual outer boundaries.
    augmented = [-1] + separators + [size]
    diffs = [int(augmented[i + 1] - augmented[i]) for i in range(len(augmented) - 1)]
    if diffs and len(set(diffs)) == 1:
        cell_size = int(diffs[0] - 1)
        spans = tuple((int(augmented[i] + 1), int(augmented[i + 1])) for i in range(len(augmented) - 1))
        if len(spans) >= 2 and cell_size >= 3 and cell_size % 2 == 1:
            return spans, cell_size

    return None


def _infer_partition(grid: np.ndarray) -> Partition:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, Partition]] = []

    for color in map(int, np.unique(g)):
        sep_rows = [r for r in range(h) if bool(np.all(g[r, :] == color))]
        sep_cols = [c for c in range(w) if bool(np.all(g[:, c] == color))]
        row_part = _axis_partition(h, sep_rows)
        col_part = _axis_partition(w, sep_cols)
        if row_part is None or col_part is None:
            continue

        row_spans, cell_h = row_part
        col_spans, cell_w = col_part
        part = Partition(
            shape=(h, w),
            sep_color=int(color),
            sep_rows=tuple(int(x) for x in sep_rows),
            sep_cols=tuple(int(x) for x in sep_cols),
            row_spans=row_spans,
            col_spans=col_spans,
            cell_h=int(cell_h),
            cell_w=int(cell_w),
        )
        cells = int(len(row_spans) * len(col_spans))
        score = int(len(sep_rows) + len(sep_cols))
        candidates.append((cells, score, part))

    assert candidates, "failed to infer the separator lattice"
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _infer_bg(grid: np.ndarray, sep_color: int) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int)[np.asarray(grid, dtype=int) != int(sep_color)], return_counts=True)
    assert len(vals) > 0
    return int(vals[int(np.argmax(counts))])


def _extract_tiles(grid: np.ndarray, part: Partition, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mid_r = int(part.cell_h // 2)
    mid_c = int(part.cell_w // 2)
    tiles: list[dict] = []

    for r0, r1 in part.row_spans:
        for c0, c1 in part.col_spans:
            cell = g[int(r0) : int(r1), int(c0) : int(c1)].copy()
            edges = (
                int(cell[0, mid_c]),
                int(cell[mid_r, part.cell_w - 1]),
                int(cell[part.cell_h - 1, mid_c]),
                int(cell[mid_r, 0]),
            )
            assert all(color not in (int(bg), int(part.sep_color)) for color in edges), "missing edge interface"
            inner = cell[1:-1, 1:-1]
            tiles.append(
                {
                    "arr": cell,
                    "edges": edges,
                    "blank": not bool(np.any(inner != int(bg))),
                }
            )

    assert len(tiles) == len(part.row_spans) * len(part.col_spans)
    assert sum(int(tile["blank"]) for tile in tiles) == 1, "expected exactly one blank tile"
    return tiles


def _solve_orders(tiles: list[dict], rows: int, cols: int, max_solutions: int = 2) -> list[tuple[int, ...]]:
    n = len(tiles)
    assert n == rows * cols
    all_ids = tuple(range(n))
    by_top: dict[int, list[int]] = {}
    by_left: dict[int, list[int]] = {}
    by_top_left: dict[tuple[int, int], list[int]] = {}
    for idx, tile in enumerate(tiles):
        top, right, bottom, left = tile["edges"]
        by_top.setdefault(int(top), []).append(int(idx))
        by_left.setdefault(int(left), []).append(int(idx))
        by_top_left.setdefault((int(top), int(left)), []).append(int(idx))

    order = [-1] * n
    used = [False] * n
    solutions: list[tuple[int, ...]] = []

    def dfs(pos: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if pos == n:
            solutions.append(tuple(int(x) for x in order))
            return

        r, c = divmod(pos, cols)
        if r > 0 and c > 0:
            need_top = int(tiles[order[pos - cols]]["edges"][2])
            need_left = int(tiles[order[pos - 1]]["edges"][1])
            candidates = by_top_left.get((need_top, need_left), ())
        elif r > 0:
            need_top = int(tiles[order[pos - cols]]["edges"][2])
            candidates = by_top.get(need_top, ())
        elif c > 0:
            need_left = int(tiles[order[pos - 1]]["edges"][1])
            candidates = by_left.get(need_left, ())
        else:
            candidates = all_ids

        for idx in candidates:
            if used[int(idx)]:
                continue
            used[int(idx)] = True
            order[pos] = int(idx)
            dfs(pos + 1)
            order[pos] = -1
            used[int(idx)] = False
            if len(solutions) >= max_solutions:
                return

    dfs(0)
    return solutions


def _render_grid(part: Partition, tiles: list[dict], order: tuple[int, ...] | list[int], bg: int) -> np.ndarray:
    out = np.full(part.shape, int(bg), dtype=int)
    if part.sep_rows:
        out[np.asarray(part.sep_rows, dtype=int), :] = int(part.sep_color)
    if part.sep_cols:
        out[:, np.asarray(part.sep_cols, dtype=int)] = int(part.sep_color)

    idx = 0
    for r0, r1 in part.row_spans:
        for c0, c1 in part.col_spans:
            out[int(r0) : int(r1), int(c0) : int(c1)] = tiles[int(order[idx])]["arr"]
            idx += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the shuffled tile grid so adjacent edge-interface colors match.

    Input structure:
    - Black background surrounds a lattice of equal rectangular cells separated by full single-pixel maroon rows and columns.
    - Each cell contains a small interior motif that does not touch the cell boundary.
    - The midpoint of each cell edge contains one non-black interface pixel just inside the separator line.
    - Exactly one cell is blank inside, but that blank tile still has the four interface pixels.

    Rule:
    - Treat each cell contents as one fixed-orientation tile.
    - Two neighboring output tiles are compatible exactly when the interface colors on their shared side are equal.
    - Permute the tiles, without rotating or flipping them, until every horizontal and vertical adjacency satisfies that color-match rule.
    - Keep the separator lattice unchanged and move the blank tile together with the other tiles.

    Key insight:
    - The visible motifs are only cargo; the real jigsaw code is the four midpoint interface pixels on each tile edge, and the whole grid must be solved globally rather than greedily.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    part = _infer_partition(g)
    bg = _infer_bg(g, part.sep_color)
    tiles = _extract_tiles(g, part, bg)
    rows = int(len(part.row_spans))
    cols = int(len(part.col_spans))

    solutions = _solve_orders(tiles, rows, cols, max_solutions=2)
    assert len(solutions) == 1, "expected a unique tiling arrangement"
    return _render_grid(part, tiles, solutions[0], bg)


def _build_partition(rows: int, cols: int, cell_h: int, cell_w: int, sep_color: int) -> Partition:
    sep_rows = tuple(int(i * (cell_h + 1)) for i in range(rows + 1))
    sep_cols = tuple(int(i * (cell_w + 1)) for i in range(cols + 1))
    row_spans = tuple((int(sep_rows[i] + 1), int(sep_rows[i + 1])) for i in range(rows))
    col_spans = tuple((int(sep_cols[i] + 1), int(sep_cols[i + 1])) for i in range(cols))
    return Partition(
        shape=(int(rows * (cell_h + 1) + 1), int(cols * (cell_w + 1) + 1)),
        sep_color=int(sep_color),
        sep_rows=sep_rows,
        sep_cols=sep_cols,
        row_spans=row_spans,
        col_spans=col_spans,
        cell_h=int(cell_h),
        cell_w=int(cell_w),
    )


def _layout_options() -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for cell_h in (5, 7):
        for cell_w in (5, 7):
            max_rows = (30 - 1) // (cell_h + 1)
            max_cols = (30 - 1) // (cell_w + 1)
            for rows in range(2, max_rows + 1):
                for cols in range(2, max_cols + 1):
                    cells = rows * cols
                    if 4 <= cells <= 9:
                        options.append((rows, cols, cell_h, cell_w))
    return options


LAYOUT_OPTIONS = _layout_options()


def _random_connected_cells(h: int, w: int, area: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        seen = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return sorted(cells)


def _paint_pattern(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    inner_h = h - 2
    inner_w = w - 2
    max_area = min(6, inner_h * inner_w)
    area = random.randint(3, max_area)
    cells = _random_connected_cells(inner_h, inner_w, area)
    assert cells is not None

    n_colors = random.randint(2, min(3, area))
    colors = random.sample(PATTERN_POOL, n_colors)
    assignments = colors.copy()
    while len(assignments) < area:
        assignments.append(random.choice(colors))
    random.shuffle(assignments)

    for (r, c), color in zip(cells, assignments):
        arr[int(r + 1), int(c + 1)] = int(color)
    assert np.count_nonzero(arr[1:-1, 1:-1] != int(bg)) >= 3
    assert len(set(int(x) for x in arr[1:-1, 1:-1].ravel() if int(x) != int(bg))) >= 2
    return arr


def _make_tile(cell_h: int, cell_w: int, edges: tuple[int, int, int, int], *, blank: bool) -> np.ndarray:
    tile = np.full((cell_h, cell_w), BG, dtype=int)
    mid_r = cell_h // 2
    mid_c = cell_w // 2
    tile[0, mid_c] = int(edges[0])
    tile[mid_r, cell_w - 1] = int(edges[1])
    tile[cell_h - 1, mid_c] = int(edges[2])
    tile[mid_r, 0] = int(edges[3])
    if not blank:
        tile = _paint_pattern(tile, BG)
    return tile


def _edge_assignment(rows: int, cols: int, interface_colors: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    top = [[random.choice(interface_colors) for _ in range(cols)] for _ in range(rows)]
    right = [[random.choice(interface_colors) for _ in range(cols)] for _ in range(rows)]
    bottom = [[random.choice(interface_colors) for _ in range(cols)] for _ in range(rows)]
    left = [[random.choice(interface_colors) for _ in range(cols)] for _ in range(rows)]

    for r in range(rows - 1):
        for c in range(cols):
            color = int(random.choice(interface_colors))
            bottom[r][c] = color
            top[r + 1][c] = color
    for r in range(rows):
        for c in range(cols - 1):
            color = int(random.choice(interface_colors))
            right[r][c] = color
            left[r][c + 1] = color

    edges: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            edges.append((int(top[r][c]), int(right[r][c]), int(bottom[r][c]), int(left[r][c])))
    return edges


def _mismatch_count(order: list[int] | tuple[int, ...], tiles: list[dict], rows: int, cols: int) -> int:
    mismatches = 0
    for r in range(rows):
        for c in range(cols - 1):
            a = tiles[int(order[r * cols + c])]["edges"][1]
            b = tiles[int(order[r * cols + c + 1])]["edges"][3]
            mismatches += int(a != b)
    for r in range(rows - 1):
        for c in range(cols):
            a = tiles[int(order[r * cols + c])]["edges"][2]
            b = tiles[int(order[(r + 1) * cols + c])]["edges"][0]
            mismatches += int(a != b)
    return int(mismatches)


def _generate_tiles(rows: int, cols: int, cell_h: int, cell_w: int) -> list[dict] | None:
    interface_count = random.randint(3, 5)
    interface_colors = tuple(random.sample(INTERFACE_POOL, interface_count))
    edges = _edge_assignment(rows, cols, interface_colors)
    blank_idx = random.randrange(rows * cols)

    seen_signatures: set[bytes] = set()
    tiles: list[dict] = []
    for idx, edge_tuple in enumerate(edges):
        for _ in range(50):
            arr = _make_tile(cell_h, cell_w, edge_tuple, blank=idx == blank_idx)
            signature = arr.tobytes()
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            tiles.append(
                {
                    "arr": arr,
                    "edges": edge_tuple,
                    "blank": idx == blank_idx,
                }
            )
            break
        else:
            return None
    return tiles


def _gen_attempt() -> np.ndarray | None:
    rows, cols, cell_h, cell_w = random.choice(LAYOUT_OPTIONS)
    part = _build_partition(rows, cols, cell_h, cell_w, SEP)
    tiles = _generate_tiles(rows, cols, cell_h, cell_w)
    if tiles is None:
        return None

    solved_order = tuple(range(rows * cols))
    solutions = _solve_orders(tiles, rows, cols, max_solutions=2)
    if len(solutions) != 1 or solutions[0] != solved_order:
        return None

    shuffled = list(solved_order)
    for _ in range(50):
        random.shuffle(shuffled)
        if tuple(shuffled) == solved_order:
            continue
        if _mismatch_count(shuffled, tiles, rows, cols) == 0:
            continue
        break
    else:
        return None

    inp = _render_grid(part, tiles, shuffled, BG)
    out = _render_grid(part, tiles, solved_order, BG)
    if np.array_equal(inp, out):
        return None

    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shuffled interface-tiling puzzle")
