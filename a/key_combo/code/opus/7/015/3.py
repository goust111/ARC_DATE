from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_gap_position(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BLACK:
        return False
    horizontal_gap = 0 < c < w - 1 and int(grid[r, c - 1]) == GREEN and int(grid[r, c + 1]) == GREEN
    vertical_gap = 0 < r < h - 1 and int(grid[r - 1, c]) == GREEN and int(grid[r + 1, c]) == GREEN
    return horizontal_gap or vertical_gap


def _markers_on_top_row(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) in (BLUE, RED)]


def _settled_grid(grid: np.ndarray, *, seal_red_gaps: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    markers = _markers_on_top_row(out)
    for c, _color in markers:
        out[0, c] = BLACK

    h = out.shape[0]
    for c, color in markers:
        if color == BLUE:
            r = 0
            while r + 1 < h and int(out[r + 1, c]) != GREEN:
                r += 1
            out[r, c] = BLUE
        else:
            if seal_red_gaps:
                for r in range(1, h):
                    if _is_gap_position(out, r, c):
                        out[r, c] = GREEN
            out[h - 1, c] = RED
    return out


def _classify_black_regions(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for coords in _components_mask(out == BLACK):
        r0, c0, r1, c1 = _bbox_coords(coords)
        fill = YELLOW if len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1) else ORANGE
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop red and blue top markers through a holed green lattice, then recolor black regions by whether the post-repair cell area is intact or broken.

    Input structure:
    - The grid has a black background and a one-pixel-wide green lattice that partitions
      the canvas into equal rectangular cells.
    - Some green line segments contain one-cell black gaps that connect neighboring cells.
    - The top row contains several single-pixel blue or red markers in distinct columns.

    Rule:
    - Erase the original top-row markers and process each marker by its column.
    - A blue marker falls straight down until the cell below it is green, then stops there;
      it does not change the lattice.
    - A red marker ignores green bars, lands on the bottom row, and turns every traversed
      black lattice gap in its column into green, thereby sealing those openings.
    - After all drops, find every 4-connected black region. If a region exactly fills its
      tight bounding box, recolor it yellow; otherwise recolor it orange. Keep the green
      lattice and the settled red/blue pixels unchanged.

    Key insight:
    - The red markers are not just falling objects: they repair specific lattice gaps on
      their way down, so the final region colors depend on simulating the drop before
      testing which black areas are still single intact cells versus broken multi-cell leaks.
    """
    settled = _settled_grid(grid, seal_red_gaps=True)
    return _classify_black_regions(settled)


def _graph_component_sizes(
    nrows: int,
    ncols: int,
    edges: set[tuple[str, int, int]],
) -> dict[tuple[int, int], int]:
    adj = {(r, c): set() for r in range(nrows) for c in range(ncols)}
    for orient, r, c in edges:
        if orient == "h":
            a = (r, c)
            b = (r + 1, c)
        else:
            a = (r, c)
            b = (r, c + 1)
        adj[a].add(b)
        adj[b].add(a)

    sizes: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()
    for cell in adj:
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        for item in comp:
            sizes[item] = len(comp)
    return sizes


def _cell_box(
    line_rows: list[int],
    line_cols: list[int],
    cell_r: int,
    cell_c: int,
) -> tuple[int, int, int, int]:
    return (
        line_rows[cell_r] + 1,
        line_cols[cell_c] + 1,
        line_rows[cell_r + 1] - 1,
        line_cols[cell_c + 1] - 1,
    )


def _horizontal_gap_pixel(
    line_rows: list[int],
    line_cols: list[int],
    edge_r: int,
    edge_c: int,
    col: int,
) -> tuple[int, int]:
    return line_rows[edge_r + 1], int(col)


def _vertical_gap_pixel(
    line_rows: list[int],
    line_cols: list[int],
    edge_r: int,
    edge_c: int,
    row: int,
) -> tuple[int, int]:
    return int(row), line_cols[edge_c + 1]


def _build_lattice(
    nrows: int,
    ncols: int,
    cell_size: int,
    top_offset: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    line_rows = [top_offset + k * (cell_size + 1) for k in range(nrows + 1)]
    line_cols = [k * (cell_size + 1) for k in range(ncols + 1)]
    height = line_rows[-1] + 2
    width = line_cols[-1] + 1
    grid = np.zeros((height, width), dtype=int)
    grid[line_rows, :] = GREEN
    grid[:, line_cols] = GREEN
    return grid, line_rows, line_cols


def _random_interior_col(line_cols: list[int], cell_c: int) -> int:
    return random.randint(line_cols[cell_c] + 1, line_cols[cell_c + 1] - 1)


def _random_interior_row(line_rows: list[int], cell_r: int) -> int:
    return random.randint(line_rows[cell_r] + 1, line_rows[cell_r + 1] - 1)


def _candidate_blue_columns(width: int, red_cols: set[int]) -> list[int]:
    cols = [c for c in range(width) if c not in red_cols]
    random.shuffle(cols)
    return cols


def _generate_case() -> tuple[np.ndarray, dict]:
    nrows = random.randint(3, 4)
    ncols = random.randint(3, 4)
    cell_size = random.randint(3, 4)
    top_offset = random.randint(1, 2)
    grid, line_rows, line_cols = _build_lattice(nrows, ncols, cell_size, top_offset)

    red_cell_cols = random.sample(range(ncols), random.randint(1, min(2, ncols - 1)))
    red_drop_cols = {cell_c: _random_interior_col(line_cols, cell_c) for cell_c in red_cell_cols}

    target_col = random.choice(red_cell_cols)
    target_edge_r = random.randrange(nrows - 1)
    target_edge = ("h", target_edge_r, target_col)
    target_cell = random.choice([(target_edge_r, target_col), (target_edge_r + 1, target_col)])

    sealed_edges: set[tuple[str, int, int]] = {target_edge}
    for cell_c in red_cell_cols:
        extra_rows = list(range(nrows - 1))
        random.shuffle(extra_rows)
        want = 1 if cell_c == target_col else random.choice([1, 1, 2])
        count = 0
        for edge_r in extra_rows:
            edge = ("h", edge_r, cell_c)
            if edge == target_edge:
                continue
            if count >= want - 1:
                break
            if random.random() < 0.55:
                sealed_edges.add(edge)
                count += 1

    persistent_edges: set[tuple[str, int, int]] = set()

    broken_col_choices = [c for c in range(ncols) if c not in red_cell_cols]
    broken_col = random.choice(broken_col_choices)
    broken_len = random.randint(2, min(nrows, 3))
    broken_start = random.randint(0, nrows - broken_len)
    for edge_r in range(broken_start, broken_start + broken_len - 1):
        persistent_edges.add(("h", edge_r, broken_col))

    if random.random() < 0.6:
        row_choices = [r for r in range(nrows) if r != target_cell[0]]
        if row_choices:
            vr = random.choice(row_choices)
            vc_choices = [c for c in range(ncols - 1) if (vr, c) != target_cell and (vr, c + 1) != target_cell]
            if vc_choices:
                persistent_edges.add(("v", vr, random.choice(vc_choices)))

    if random.random() < 0.35 and len(broken_col_choices) > 1:
        other_col = random.choice([c for c in broken_col_choices if c != broken_col])
        edge_r = random.randrange(nrows - 1)
        persistent_edges.add(("h", edge_r, other_col))

    persistent_edges = {edge for edge in persistent_edges if target_cell not in _edge_cells(edge)}
    if not any(edge[0] == "h" for edge in persistent_edges):
        fallback_col = broken_col_choices[0]
        fallback_r = 0 if target_cell != (0, fallback_col) else 1
        if fallback_r < nrows - 1:
            persistent_edges.add(("h", fallback_r, fallback_col))

    gap_pixels: dict[tuple[str, int, int], tuple[int, int]] = {}
    for orient, edge_r, edge_c in sealed_edges | persistent_edges:
        if orient == "h":
            if edge_c in red_drop_cols and (orient, edge_r, edge_c) in sealed_edges:
                gap_pixels[(orient, edge_r, edge_c)] = _horizontal_gap_pixel(
                    line_rows, line_cols, edge_r, edge_c, red_drop_cols[edge_c]
                )
            else:
                col = _random_interior_col(line_cols, edge_c)
                if edge_c in red_drop_cols and col == red_drop_cols[edge_c]:
                    choices = [cc for cc in range(line_cols[edge_c] + 1, line_cols[edge_c + 1]) if cc != red_drop_cols[edge_c]]
                    if choices:
                        col = random.choice(choices)
                gap_pixels[(orient, edge_r, edge_c)] = _horizontal_gap_pixel(line_rows, line_cols, edge_r, edge_c, col)
        else:
            gap_pixels[(orient, edge_r, edge_c)] = _vertical_gap_pixel(
                line_rows, line_cols, edge_r, edge_c, _random_interior_row(line_rows, edge_r)
            )

    for r, c in gap_pixels.values():
        grid[r, c] = BLACK

    marker_count = random.randint(3, 5)
    blue_count = marker_count - len(red_drop_cols)
    blue_cols = _candidate_blue_columns(grid.shape[1], set(red_drop_cols.values()))[:blue_count]
    if len(blue_cols) < blue_count:
        raise RuntimeError("not enough columns for markers")
    for c in red_drop_cols.values():
        grid[0, c] = RED
    for c in blue_cols:
        grid[0, c] = BLUE

    meta = {
        "nrows": nrows,
        "ncols": ncols,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "target_cell": target_cell,
        "persistent_edges": persistent_edges,
        "sealed_edges": sealed_edges,
        "initial_sizes": _graph_component_sizes(nrows, ncols, persistent_edges | sealed_edges),
        "final_sizes": _graph_component_sizes(nrows, ncols, persistent_edges),
    }
    return grid, meta


def _edge_cells(edge: tuple[str, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    orient, r, c = edge
    if orient == "h":
        return (r, c), (r + 1, c)
    return (r, c), (r, c + 1)


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    markers = _markers_on_top_row(grid)
    if not (3 <= len(markers) <= 5):
        return False
    if not any(color == RED for _c, color in markers) or not any(color == BLUE for _c, color in markers):
        return False

    line_rows = meta["line_rows"]
    line_cols = meta["line_cols"]

    for c, color in markers:
        if color == RED:
            if not any(_is_gap_position(grid, r, c) for r in range(1, grid.shape[0])):
                return False
        else:
            if not any(int(grid[r, c]) == GREEN for r in line_rows):
                return False

    out = main(grid)
    no_seal_out = _classify_black_regions(_settled_grid(grid, seal_red_gaps=False))
    if np.array_equal(out, no_seal_out):
        return False

    target_cell = meta["target_cell"]
    if meta["initial_sizes"][target_cell] <= 1 or meta["final_sizes"][target_cell] != 1:
        return False
    if not any(size > 1 for size in meta["final_sizes"].values()):
        return False

    has_yellow = False
    has_orange = False
    for cell_r in range(meta["nrows"]):
        for cell_c in range(meta["ncols"]):
            r0, c0, r1, c1 = _cell_box(line_rows, line_cols, cell_r, cell_c)
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            unique = set(map(int, np.unique(patch)))
            if meta["final_sizes"][(cell_r, cell_c)] == 1:
                if unique != {YELLOW}:
                    return False
                has_yellow = True
            else:
                if unique != {ORANGE}:
                    return False
                has_orange = True

    if not (has_yellow and has_orange):
        return False

    if grid.ndim != 2 or out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, meta = _generate_case()
        if _validate_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid lattice-repair ARC input")
