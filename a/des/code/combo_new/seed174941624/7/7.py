from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
EDGE_DIRECTIONS = ("up", "down", "left", "right")

GRID_MARGIN_CHOICES = (1, 2)
REGION_COUNT_MIN = 2
REGION_COUNT_MAX = 4
REGION_SIZE_OPTIONS = (3, 5, 7)
MAIN_CHAIN_MAX = 5
EXTRA_CHAIN_MAX = 4
GEN_RETRIES = 300
CHAIN_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            pts.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _region_infos(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    mask = (grid != BLACK) & (grid != GREY)
    comps = _components(mask)
    assert comps, "expected at least one colored region"

    region_of = np.full(grid.shape, -1, dtype=int)
    infos: list[dict[str, object]] = []
    for idx, pts in enumerate(comps):
        for r, c in pts:
            region_of[r, c] = idx

        r0, c0, r1, c1 = _bbox(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(pts) == area, "each region must be a solid rectangle"

        values = [int(grid[r, c]) for r, c in pts]
        counts = Counter(values)
        fill_color = int(counts.most_common(1)[0][0])
        pointer_cells = [(r, c) for r, c in pts if int(grid[r, c]) != fill_color]
        assert len(pointer_cells) <= 1, "each region may contain at most one pointer"

        direction = None
        pointer = None
        if pointer_cells:
            pointer = pointer_cells[0]
            sides = []
            if pointer[0] == r0:
                sides.append("up")
            if pointer[0] == r1:
                sides.append("down")
            if pointer[1] == c0:
                sides.append("left")
            if pointer[1] == c1:
                sides.append("right")
            assert len(sides) == 1, "pointer must sit on exactly one region edge"
            direction = sides[0]

        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "fill": fill_color,
                "pointer": pointer,
                "direction": direction,
                "target": None,
            }
        )

    h, w = grid.shape
    for info in infos:
        if info["direction"] is None:
            continue
        r0, c0, r1, c1 = info["bbox"]
        pr, pc = info["pointer"]
        if info["direction"] == "up":
            assert r0 - 1 >= 0 and int(grid[r0 - 1, pc]) == GREY
            tr, tc = r0 - 2, pc
        elif info["direction"] == "down":
            assert r1 + 1 < h and int(grid[r1 + 1, pc]) == GREY
            tr, tc = r1 + 2, pc
        elif info["direction"] == "left":
            assert c0 - 1 >= 0 and int(grid[pr, c0 - 1]) == GREY
            tr, tc = pr, c0 - 2
        else:
            assert c1 + 1 < w and int(grid[pr, c1 + 1]) == GREY
            tr, tc = pr, c1 + 2

        if 0 <= tr < h and 0 <= tc < w and region_of[tr, tc] != -1:
            info["target"] = int(region_of[tr, tc])

    return infos, region_of


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cascade each pointed region's fill color into the adjacent region indicated by its edge marker.

    Input structure:
    - The background is black.
    - Grey one-pixel divider lines partition the grid into solid rectangular regions.
    - Each region is uniformly filled with one non-black non-grey color, and some regions contain one single-pixel pointer of another color on exactly one edge midpoint.

    Rule:
    - Recover every rectangular region and use the unique off-fill pixel, when present, to decode a direction from the edge it touches.
    - In that direction, cross the grey divider to the adjacent region and replace that region's fill color with the current region's fill color.
    - If the overwritten neighbor also has a pointer, continue the propagation from that neighbor using its new color, so only source regions start chains.
    - Remove all pointer pixels by repainting each region uniformly with its final fill color while keeping the grey divider lines unchanged.

    Key insight:
    - The pointer color itself is irrelevant; only its position on a region edge matters.
    - Propagation is chain-based, so intermediate pointed regions must pass on the color they just received rather than their original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30, "ARC grids must stay within 30x30"

    infos, _region_of = _region_infos(g)
    n_regions = len(infos)

    indegree = [0] * n_regions
    for info in infos:
        target = info["target"]
        if target is not None:
            indegree[int(target)] += 1
    assert max(indegree) <= 1, "generated tasks use non-merging propagation chains"

    final_fills = [int(info["fill"]) for info in infos]
    queue = deque(
        idx for idx, info in enumerate(infos) if info["direction"] is not None and indegree[idx] == 0
    )
    processed: set[int] = set()

    while queue:
        idx = int(queue.popleft())
        if idx in processed:
            continue
        processed.add(idx)
        target = infos[idx]["target"]
        if target is None:
            continue
        target = int(target)
        final_fills[target] = final_fills[idx]
        if infos[target]["direction"] is not None:
            queue.append(target)

    out = g.copy()
    for idx, info in enumerate(infos):
        r0, c0, r1, c1 = info["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = final_fills[idx]
    return out


def _sample_axis_sizes(count: int, budget: int, rng: random.Random) -> list[int] | None:
    sizes: list[int] = []
    remaining = budget
    for i in range(count):
        cells_left = count - i - 1
        max_here = min(max(REGION_SIZE_OPTIONS), remaining - 3 * cells_left)
        options = [size for size in REGION_SIZE_OPTIONS if size <= max_here]
        if not options:
            return None
        size = rng.choice(options)
        sizes.append(size)
        remaining -= size
    return sizes


def _separator_positions(margin: int, sizes: list[int]) -> list[int]:
    positions = [margin]
    cur = margin
    for size in sizes:
        cur += size + 1
        positions.append(cur)
    return positions


def _cell_bbox(
    row_idx: int,
    col_idx: int,
    row_lines: list[int],
    col_lines: list[int],
) -> tuple[int, int, int, int]:
    return (
        row_lines[row_idx] + 1,
        col_lines[col_idx] + 1,
        row_lines[row_idx + 1] - 1,
        col_lines[col_idx + 1] - 1,
    )


def _grid_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _dist_to_border(cell: tuple[int, int], rows: int, cols: int) -> int:
    r, c = cell
    return min(r, c, rows - 1 - r, cols - 1 - c)


def _border_exit_dirs(cell: tuple[int, int], rows: int, cols: int) -> list[str]:
    r, c = cell
    dirs = []
    if r == 0:
        dirs.append("up")
    if r == rows - 1:
        dirs.append("down")
    if c == 0:
        dirs.append("left")
    if c == cols - 1:
        dirs.append("right")
    return dirs


def _find_path(
    available: set[tuple[int, int]],
    rows: int,
    cols: int,
    length: int,
    require_border_end: bool,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    starts = list(available)
    rng.shuffle(starts)
    if require_border_end:
        starts.sort(key=lambda cell: (_dist_to_border(cell, rows, cols), cell[0], cell[1]), reverse=True)

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        cur = path[-1]
        remaining = length - len(path)
        if require_border_end and _dist_to_border(cur, rows, cols) > remaining:
            return None
        if len(path) == length:
            if not require_border_end or _border_exit_dirs(cur, rows, cols):
                return path[:]
            return None

        nbrs = [nbr for nbr in _grid_neighbors(cur, rows, cols) if nbr in available and nbr not in used]
        rng.shuffle(nbrs)
        if require_border_end:
            nbrs.sort(key=lambda cell: _dist_to_border(cell, rows, cols))
        for nbr in nbrs:
            used.add(nbr)
            path.append(nbr)
            result = dfs(path, used)
            if result is not None:
                return result
            path.pop()
            used.remove(nbr)
        return None

    for start in starts:
        result = dfs([start], {start})
        if result is not None:
            return result
    return None


def _direction_from_step(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if (dr, dc) == (-1, 0):
        return "up"
    if (dr, dc) == (1, 0):
        return "down"
    if (dr, dc) == (0, -1):
        return "left"
    if (dr, dc) == (0, 1):
        return "right"
    raise ValueError(f"non-adjacent cells: {a} -> {b}")


def _build_chains(rows: int, cols: int, rng: random.Random) -> list[dict[str, object]] | None:
    total = rows * cols
    available = {(r, c) for r in range(rows) for c in range(cols)}
    chains: list[dict[str, object]] = []

    main_min = 3 if total >= 4 else 2
    main_max = min(MAIN_CHAIN_MAX, total - 1)
    if main_max < main_min:
        return None

    main_lengths = list(range(main_min, main_max + 1))
    rng.shuffle(main_lengths)
    main_path = None
    for length in main_lengths:
        main_path = _find_path(available, rows, cols, length, False, rng)
        if main_path is not None:
            break
    if main_path is None:
        return None

    chains.append({"cells": main_path, "term": "sink"})
    available -= set(main_path)

    extra_target = 0
    if total >= 6 and rng.random() < 0.7:
        extra_target += 1
    if total >= 10 and rng.random() < 0.45:
        extra_target += 1

    for extra_idx in range(extra_target):
        if len(available) < 2:
            break
        preferred_term = "boundary" if extra_idx == 0 and rng.random() < 0.65 else rng.choice(("sink", "boundary"))
        terms = [preferred_term, "sink" if preferred_term == "boundary" else "boundary"]
        made = False
        for term in terms:
            lengths = list(range(2, min(EXTRA_CHAIN_MAX, len(available)) + 1))
            rng.shuffle(lengths)
            for length in lengths:
                path = _find_path(available, rows, cols, length, term == "boundary", rng)
                if path is None:
                    continue
                chains.append({"cells": path, "term": term})
                available -= set(path)
                made = True
                break
            if made:
                break

    return chains


def _final_colors(
    fills: dict[tuple[int, int], int],
    chains: list[dict[str, object]],
) -> dict[tuple[int, int], int]:
    out = dict(fills)
    for chain in chains:
        cells = chain["cells"]
        source_color = int(fills[cells[0]])
        for cell in cells[1:]:
            out[cell] = source_color
    return out


def _pointer_cell(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "up":
        return r0, mid_c
    if direction == "down":
        return r1, mid_c
    if direction == "left":
        return mid_r, c0
    if direction == "right":
        return mid_r, c1
    raise ValueError(f"unknown direction: {direction}")


def _build_case(
    margin: int,
    row_sizes: list[int],
    col_sizes: list[int],
    chains: list[dict[str, object]],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    rows = len(row_sizes)
    cols = len(col_sizes)
    height = 2 * margin + (rows + 1) + sum(row_sizes)
    width = 2 * margin + (cols + 1) + sum(col_sizes)
    inp = np.full((height, width), BLACK, dtype=int)
    out = np.full((height, width), BLACK, dtype=int)

    row_lines = _separator_positions(margin, row_sizes)
    col_lines = _separator_positions(margin, col_sizes)
    top = row_lines[0]
    bottom = row_lines[-1]
    left = col_lines[0]
    right = col_lines[-1]

    for r in row_lines:
        inp[r, left : right + 1] = GREY
        out[r, left : right + 1] = GREY
    for c in col_lines:
        inp[top : bottom + 1, c] = GREY
        out[top : bottom + 1, c] = GREY

    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    fills = {cell: rng.choice(FILL_COLORS) for cell in all_cells}

    source_colors = list(FILL_COLORS)
    rng.shuffle(source_colors)
    for idx, chain in enumerate(chains):
        source = chain["cells"][0]
        source_color = source_colors[idx]
        fills[source] = source_color
        for cell in chain["cells"][1:]:
            fills[cell] = rng.choice([color for color in FILL_COLORS if color != source_color])

    final = _final_colors(fills, chains)

    pointer_dirs: dict[tuple[int, int], str] = {}
    for chain in chains:
        cells = chain["cells"]
        for a, b in zip(cells, cells[1:]):
            pointer_dirs[a] = _direction_from_step(a, b)
        if chain["term"] == "boundary":
            exits = _border_exit_dirs(cells[-1], rows, cols)
            assert exits, "boundary chain must end on the outer border"
            pointer_dirs[cells[-1]] = rng.choice(exits)

    for cell in all_cells:
        bbox = _cell_bbox(cell[0], cell[1], row_lines, col_lines)
        r0, c0, r1, c1 = bbox
        inp[r0 : r1 + 1, c0 : c1 + 1] = fills[cell]
        out[r0 : r1 + 1, c0 : c1 + 1] = final[cell]
        if cell in pointer_dirs:
            pr, pc = _pointer_cell(bbox, pointer_dirs[cell])
            inp[pr, pc] = rng.choice([color for color in FILL_COLORS if color != fills[cell]])

    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    margin = rng.choice(GRID_MARGIN_CHOICES)
    rows = rng.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)
    cols = rng.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)

    row_budget = 30 - 2 * margin - (rows + 1)
    col_budget = 30 - 2 * margin - (cols + 1)
    if row_budget < 3 * rows or col_budget < 3 * cols:
        return None

    row_sizes = _sample_axis_sizes(rows, row_budget, rng)
    col_sizes = _sample_axis_sizes(cols, col_budget, rng)
    if row_sizes is None or col_sizes is None:
        return None

    chains = _build_chains(rows, cols, rng)
    if chains is None:
        return None

    inp, expected = _build_case(margin, row_sizes, col_sizes, chains, rng)
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-region ARC input")
