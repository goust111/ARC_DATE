from __future__ import annotations

from collections import Counter, deque
from importlib.machinery import SourceFileLoader
import random

import numpy as np


BACKGROUND = 0
DIVIDER = 5
START = 6
SPRITE_COLORS = (1, 2, 3, 4, 7, 8, 9)

GEN_RETRIES = 400
BLOCK_STARTS_4 = ((0, 0), (0, 2), (2, 0), (2, 2))
BLOCK_STARTS_5 = ((0, 0), (0, 2), (2, 0), (2, 2))
ORTHO_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


try:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    get_bg = _ARC_UTILS.get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _ranges_between(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    points = [-1] + list(dividers) + [int(size)]
    return [
        (points[i] + 1, points[i + 1] - 1)
        for i in range(len(points) - 1)
        if points[i] + 1 <= points[i + 1] - 1
    ]


def _detect_dividers(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    divider_rows = [r for r in range(h) if int(np.count_nonzero(grid[r] == DIVIDER)) >= w // 2]
    divider_cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == DIVIDER)) >= h // 2]
    assert len(divider_rows) in (3, 4) and len(divider_cols) in (3, 4)
    return divider_rows, divider_cols


def _locate_interval(x: int, ranges: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(ranges):
        if lo <= x <= hi:
            return idx
    raise AssertionError(f"coordinate {x} must lie inside one cell range")


def _cell_nonstart_color(grid: np.ndarray, row_range: tuple[int, int], col_range: tuple[int, int]) -> int:
    r0, r1 = row_range
    c0, c1 = col_range
    vals = set(map(int, grid[r0 : r1 + 1, c0 : c1 + 1].ravel())) - {BACKGROUND, DIVIDER, START}
    assert len(vals) <= 1, "each occupied cell must contain exactly one sprite color"
    return -1 if not vals else vals.pop()


def _cell_has_start(grid: np.ndarray, row_range: tuple[int, int], col_range: tuple[int, int]) -> bool:
    r0, r1 = row_range
    c0, c1 = col_range
    return bool(np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == START))


def _cell_graph(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    divider_rows: list[int],
    divider_cols: list[int],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    rows = len(row_ranges)
    cols = len(col_ranges)
    graph = {(r, c): set() for r in range(rows) for c in range(cols)}

    for r in range(rows - 1):
        divider_row = divider_rows[r]
        for c, (c0, c1) in enumerate(col_ranges):
            if np.any(grid[divider_row, c0 : c1 + 1] == BACKGROUND):
                graph[(r, c)].add((r + 1, c))
                graph[(r + 1, c)].add((r, c))

    for r, (r0, r1) in enumerate(row_ranges):
        for c in range(cols - 1):
            divider_col = divider_cols[c]
            if np.any(grid[r0 : r1 + 1, divider_col] == BACKGROUND):
                graph[(r, c)].add((r, c + 1))
                graph[(r, c + 1)].add((r, c))

    return graph


def _reachable_cells(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start_cell: tuple[int, int],
) -> set[tuple[int, int]]:
    seen = {start_cell}
    queue: deque[tuple[int, int]] = deque([start_cell])
    while queue:
        cell = queue.popleft()
        for nxt in graph[cell]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _predict_missing_mask(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    have: list[tuple[int, int]],
    missing: tuple[int, int],
    color: int,
) -> np.ndarray:
    mr, mc = missing
    predictions = []
    for r, c in have:
        r0, r1 = row_ranges[r]
        c0, c1 = col_ranges[c]
        pred = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        if r != mr:
            pred = pred[::-1, :]
        if c != mc:
            pred = pred[:, ::-1]
        predictions.append(pred)

    first = predictions[0]
    assert all(np.array_equal(first, pred) for pred in predictions[1:]), (
        "the three visible sprites must imply one mirrored missing sprite"
    )
    return first.copy()


def _inspect_puzzle(
    grid: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], set[tuple[int, int]]],
    tuple[int, int],
    set[tuple[int, int]],
]:
    divider_rows, divider_cols = _detect_dividers(grid)
    row_ranges = _ranges_between(grid.shape[0], divider_rows)
    col_ranges = _ranges_between(grid.shape[1], divider_cols)
    assert len(row_ranges) == len(col_ranges) and len(row_ranges) in (4, 5)

    sizes_r = {r1 - r0 + 1 for r0, r1 in row_ranges}
    sizes_c = {c1 - c0 + 1 for c0, c1 in col_ranges}
    assert len(sizes_r) == 1 and len(sizes_c) == 1

    start_pixels = np.argwhere(grid == START)
    assert start_pixels.shape[0] == 1, "expected exactly one start marker"
    start_r, start_c = map(int, start_pixels[0])
    start_cell = (
        _locate_interval(start_r, row_ranges),
        _locate_interval(start_c, col_ranges),
    )

    graph = _cell_graph(grid, row_ranges, col_ranges, divider_rows, divider_cols)
    reachable = _reachable_cells(graph, start_cell)

    cell_colors = np.full((len(row_ranges), len(col_ranges)), -2, dtype=int)
    has_start = np.zeros((len(row_ranges), len(col_ranges)), dtype=bool)
    for r, row_range in enumerate(row_ranges):
        for c, col_range in enumerate(col_ranges):
            cell_colors[r, c] = _cell_nonstart_color(grid, row_range, col_range)
            has_start[r, c] = _cell_has_start(grid, row_range, col_range)

    return row_ranges, col_ranges, cell_colors, has_start, graph, start_cell, reachable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete only the reachable missing mirrored sprite in each divider-defined 2x2 cell block.

    Input structure:
    - Black is the background and gray 1-pixel divider lines partition the canvas into a 4x4 or 5x5 grid of equal-sized cells.
    - Some divider segments contain 1-pixel black gaps that act as doors between adjacent cells.
    - A few 2x2 cell blocks contain one monochrome sprite color in exactly three cells, one empty fourth cell, and one separate cell contains a single magenta start marker.

    Rule:
    - Detect the divider rows and columns, then build a cell-level graph where a black gap on a shared divider segment connects the two neighboring cells.
    - Starting from the cell containing the magenta marker, mark every cell reachable through those divider-door connections.
    - For each 2x2 block with three same-color mirrored sprites and one empty cell, fill the empty cell only when all four cells of that block are reachable; infer the missing sprite by reflecting the visible sprites toward the missing corner.
    - Erase the magenta start marker, and leave all divider lines, door gaps, unreachable empty cells, and existing sprites unchanged.

    Key insight:
    - The gray lattice has a double role: it both defines the local symmetry cells and simultaneously acts as the maze wall system whose gaps determine whether a seemingly completable 2x2 quartet is allowed to be completed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BACKGROUND

    row_ranges, col_ranges, cell_colors, has_start, _graph, _start_cell, reachable = _inspect_puzzle(g)
    out = g.copy()
    out[out == START] = BACKGROUND

    for r in range(len(row_ranges) - 1):
        for c in range(len(col_ranges) - 1):
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if any(has_start[cr, cc] for cr, cc in cells):
                continue

            vals = [int(cell_colors[cr, cc]) for cr, cc in cells]
            sprite_colors = sorted({val for val in vals if val != -1})
            if len(sprite_colors) != 1:
                continue
            color = sprite_colors[0]

            have = [cell for cell, val in zip(cells, vals) if val == color]
            missing = [cell for cell, val in zip(cells, vals) if val == -1]
            if len(have) != 3 or len(missing) != 1:
                continue
            if not all(cell in reachable for cell in cells):
                continue

            target = missing[0]
            pred = _predict_missing_mask(g, row_ranges, col_ranges, have, target, color)
            r0, r1 = row_ranges[target[0]]
            c0, c1 = col_ranges[target[1]]
            dst = out[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(dst == BACKGROUND)
            dst[pred] = color

    return out


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHO_STEPS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _random_sprite_mask(cell_size: int) -> np.ndarray:
    for _ in range(300):
        target_area = random.randint(2, min(4, cell_size * cell_size - 1))
        cells = {(random.randrange(cell_size), random.randrange(cell_size))}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in ORTHO_STEPS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((cell_size, cell_size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if len(_mask_components(mask)) != 1:
            continue
        if int(np.count_nonzero(np.any(mask, axis=1))) < 2 or int(np.count_nonzero(np.any(mask, axis=0))) < 2:
            continue
        if np.array_equal(mask, mask[::-1, :]) or np.array_equal(mask, mask[:, ::-1]):
            continue
        return mask

    raise RuntimeError("failed to sample a visibly asymmetric connected sprite")


def _block_cells(block_start: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = block_start
    return {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}


def _cell_neighbors(cell: tuple[int, int], cell_count: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ORTHO_STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < cell_count and 0 <= nc < cell_count:
            out.append((nr, nc))
    return out


def _bfs_path_between_sets(
    cell_count: int,
    sources: set[tuple[int, int]],
    targets: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue: deque[tuple[int, int]] = deque(sources)
    parent = {cell: None for cell in sources}

    while queue:
        cell = queue.popleft()
        if cell in targets:
            path = []
            cur = cell
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]
        for nxt in _cell_neighbors(cell, cell_count):
            if nxt in blocked or nxt in parent:
                continue
            parent[nxt] = cell
            queue.append(nxt)
    return None


def _block_tree_edges(block_start: tuple[int, int]) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    r, c = block_start
    return {
        tuple(sorted(((r, c), (r, c + 1)))),
        tuple(sorted(((r, c), (r + 1, c)))),
        tuple(sorted(((r + 1, c), (r + 1, c + 1)))),
    }


def _build_layout() -> dict | None:
    cell_count = random.choice((4, 5))
    cell_size = random.choice((3, 4))
    candidates = list(BLOCK_STARTS_4 if cell_count == 4 else BLOCK_STARTS_5)

    if cell_count == 4:
        total_blocks = 3
        reachable_block_count = 2
    else:
        total_blocks = random.choice((3, 4))
        reachable_block_count = 2 if total_blocks == 3 else random.choice((2, 3))

    blocks = random.sample(candidates, total_blocks)
    reachable_blocks = set(random.sample(blocks, reachable_block_count))
    unreachable_blocks = set(blocks) - reachable_blocks

    all_block_cells = set().union(*(_block_cells(block) for block in blocks))
    blocked_cells = set().union(*(_block_cells(block) for block in unreachable_blocks))
    free_cells = [(r, c) for r in range(cell_count) for c in range(cell_count) if (r, c) not in all_block_cells]
    if not free_cells:
        return None

    start_cell = random.choice(free_cells)
    reachable_region = {start_cell}
    reachable_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    order = list(reachable_blocks)
    random.shuffle(order)
    for block in order:
        path = _bfs_path_between_sets(cell_count, reachable_region, _block_cells(block), blocked_cells)
        if path is None:
            return None
        for a, b in zip(path, path[1:]):
            reachable_edges.add(tuple(sorted((a, b))))
        reachable_region.update(path)
        reachable_region.update(_block_cells(block))
        reachable_edges |= _block_tree_edges(block)

    extras = [
        cell
        for cell in [(r, c) for r in range(cell_count) for c in range(cell_count)]
        if cell not in reachable_region
        and cell not in blocked_cells
        and any(nbr in reachable_region for nbr in _cell_neighbors(cell, cell_count))
    ]
    random.shuffle(extras)
    for cell in extras[: random.randint(0, min(3, len(extras)))]:
        neighbors = [nbr for nbr in _cell_neighbors(cell, cell_count) if nbr in reachable_region]
        reachable_edges.add(tuple(sorted((cell, random.choice(neighbors)))))
        reachable_region.add(cell)

    unreachable_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for block in unreachable_blocks:
        unreachable_edges |= _block_tree_edges(block)

    return {
        "cell_count": cell_count,
        "cell_size": cell_size,
        "blocks": blocks,
        "reachable_blocks": reachable_blocks,
        "unreachable_blocks": unreachable_blocks,
        "start_cell": start_cell,
        "reachable_edges": reachable_edges,
        "unreachable_edges": unreachable_edges,
    }


def _cell_ranges(cell_count: int, cell_size: int) -> tuple[list[tuple[int, int]], list[int], int]:
    ranges = []
    dividers = []
    pos = 0
    for idx in range(cell_count):
        ranges.append((pos, pos + cell_size - 1))
        pos += cell_size
        if idx != cell_count - 1:
            dividers.append(pos)
            pos += 1
    return ranges, dividers, pos


def _carve_door(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    divider_rows: list[int],
    divider_cols: list[int],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        cell_col = min(c1, c2)
        divider_col = divider_cols[cell_col]
        r0, r1b = row_ranges[r1]
        grid[random.randint(r0, r1b), divider_col] = BACKGROUND
        return

    cell_row = min(r1, r2)
    divider_row = divider_rows[cell_row]
    c0, c1b = col_ranges[c1]
    grid[divider_row, random.randint(c0, c1b)] = BACKGROUND


def _render_layout(layout: dict) -> tuple[np.ndarray, np.ndarray]:
    cell_count = int(layout["cell_count"])
    cell_size = int(layout["cell_size"])
    row_ranges, divider_rows, height = _cell_ranges(cell_count, cell_size)
    col_ranges, divider_cols, width = _cell_ranges(cell_count, cell_size)
    assert height == width and height <= 30

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[divider_rows, :] = DIVIDER
    grid[:, divider_cols] = DIVIDER

    for edge in sorted(layout["reachable_edges"] | layout["unreachable_edges"]):
        _carve_door(grid, row_ranges, col_ranges, divider_rows, divider_cols, edge[0], edge[1])

    start_cell = tuple(layout["start_cell"])
    sr0, sr1 = row_ranges[start_cell[0]]
    sc0, sc1 = col_ranges[start_cell[1]]
    start_pixel = (random.randint(sr0, sr1), random.randint(sc0, sc1))
    grid[start_pixel] = START

    oracle = grid.copy()
    oracle[oracle == START] = BACKGROUND
    block_specs = []
    for block, color in zip(layout["blocks"], random.sample(SPRITE_COLORS, len(layout["blocks"]))):
        mask = _random_sprite_mask(cell_size)
        cells = sorted(_block_cells(block))
        missing = random.choice(cells)
        block_specs.append({"block": block, "color": color, "mask": mask, "missing": missing})
        for cell in cells:
            if cell == missing:
                continue
            placed = mask.copy()
            if cell[0] != missing[0]:
                placed = placed[::-1, :]
            if cell[1] != missing[1]:
                placed = placed[:, ::-1]
            r0, r1 = row_ranges[cell[0]]
            c0, c1 = col_ranges[cell[1]]
            grid[r0 : r1 + 1, c0 : c1 + 1][placed] = color
            oracle[r0 : r1 + 1, c0 : c1 + 1][placed] = color
        if block in layout["reachable_blocks"]:
            r0, r1 = row_ranges[missing[0]]
            c0, c1 = col_ranges[missing[1]]
            oracle[r0 : r1 + 1, c0 : c1 + 1][mask] = color

    layout["block_specs"] = block_specs
    return grid, oracle


def _validate_instance(grid: np.ndarray, oracle: np.ndarray, layout: dict) -> bool:
    try:
        out = main(grid)
        if not np.array_equal(out, oracle):
            return False

        row_ranges, col_ranges, cell_colors, has_start, _graph, _start_cell, reachable = _inspect_puzzle(grid)
        if int(np.count_nonzero(grid == START)) != 1 or int(np.count_nonzero(out == START)) != 0:
            return False
        if len(layout["reachable_blocks"]) not in (2, 3):
            return False
        if len(layout["unreachable_blocks"]) not in (1, 2):
            return False
        if len(layout["blocks"]) != len(layout["block_specs"]):
            return False
        if len({spec["color"] for spec in layout["block_specs"]}) != len(layout["block_specs"]):
            return False

        seen_blocks = []
        for r in range(len(row_ranges) - 1):
            for c in range(len(col_ranges) - 1):
                cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                if any(has_start[cr, cc] for cr, cc in cells):
                    continue
                vals = [int(cell_colors[cr, cc]) for cr, cc in cells]
                sprite_colors = {val for val in vals if val != -1}
                if len(sprite_colors) != 1:
                    continue
                color = next(iter(sprite_colors))
                if vals.count(-1) == 1 and vals.count(color) == 3:
                    seen_blocks.append((r, c))
        if set(seen_blocks) != set(layout["blocks"]):
            return False

        for spec in layout["block_specs"]:
            block = tuple(spec["block"])
            missing = tuple(spec["missing"])
            cells = _block_cells(block)
            is_reachable = all(cell in reachable for cell in cells)
            if is_reachable != (block in layout["reachable_blocks"]):
                return False

            r0, r1 = row_ranges[missing[0]]
            c0, c1 = col_ranges[missing[1]]
            inp_patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(inp_patch != BACKGROUND):
                return False
            if block in layout["reachable_blocks"]:
                expected = np.zeros_like(out_patch)
                expected[spec["mask"]] = int(spec["color"])
                if not np.array_equal(out_patch, expected):
                    return False
            else:
                if np.any(out_patch != BACKGROUND):
                    return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        layout = _build_layout()
        if layout is None:
            continue
        grid, oracle = _render_layout(layout)
        if _validate_instance(grid, oracle, layout):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
