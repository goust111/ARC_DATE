from __future__ import annotations

import random
from collections import Counter

import numpy as np


UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

BACKGROUND = 0
GREY_CHOICES = (5, 6)
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
CELL_W = 3


def _most_common(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    counts = Counter(map(int, flat))
    return max(counts, key=counts.get)


def _grid_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best: tuple[int, int, list[int], list[int]] | None = None
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) < 2 or len(cols) < 2:
            continue
        score = len(rows) + len(cols)
        if best is None or score > best[0]:
            best = (score, color, rows, cols)
    assert best is not None, "expected a full-row/full-column partition color"
    return best[1], best[2], best[3]


def _spans(line_idxs: list[int], size: int) -> list[tuple[int, int]]:
    assert line_idxs[0] == 0 and line_idxs[-1] == size - 1, "grid lines must frame the canvas"
    spans: list[tuple[int, int]] = []
    for a, b in zip(line_idxs, line_idxs[1:]):
        assert b - a >= 2, "partitions must have positive interior size"
        spans.append((a + 1, b - 1))
    return spans


def _cell_view(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], rr: int, cc: int) -> np.ndarray:
    r0, r1 = row_spans[rr]
    c0, c1 = col_spans[cc]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _marker_info(cell: np.ndarray, bg: int, grey: int) -> tuple[str, int] | None:
    mask = (cell != bg) & (cell != grey)
    pts = np.argwhere(mask)
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] != 1:
        return None
    r, c = map(int, pts[0])
    h, w = cell.shape
    if r == 0 and 0 < c < w - 1:
        return UP, int(cell[r, c])
    if r == h - 1 and 0 < c < w - 1:
        return DOWN, int(cell[r, c])
    if c == 0 and 0 < r < h - 1:
        return LEFT, int(cell[r, c])
    if c == w - 1 and 0 < r < h - 1:
        return RIGHT, int(cell[r, c])
    return None


def _pattern_mask(cell: np.ndarray, bg: int, grey: int) -> np.ndarray | None:
    mask = (cell != bg) & (cell != grey)
    count = int(np.count_nonzero(mask))
    if count <= 1:
        return None
    return mask


def _legend_candidates(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, n_cols - 4),
        (n_rows - 2, 0),
        (n_rows - 2, n_cols - 4),
    ]


def _find_legend(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    bg: int,
    grey: int,
) -> tuple[set[tuple[int, int]], dict[str, np.ndarray]]:
    n_rows = len(row_spans)
    n_cols = len(col_spans)
    hits: list[tuple[set[tuple[int, int]], dict[str, np.ndarray]]] = []

    for sr, sc in _legend_candidates(n_rows, n_cols):
        if sr < 0 or sc < 0 or sr + 1 >= n_rows or sc + 3 >= n_cols:
            continue

        legend_cells = {(sr + dr, sc + dc) for dr in range(2) for dc in range(4)}
        direction_to_mask: dict[str, np.ndarray] = {}
        swatch_sigs: set[bytes] = set()
        ok = True

        for offset in range(4):
            marker_cell = _cell_view(grid, row_spans, col_spans, sr, sc + offset)
            info = _marker_info(marker_cell, bg, grey)
            if info is None:
                ok = False
                break
            direction, _ = info
            if direction in direction_to_mask:
                ok = False
                break

            swatch_cell = _cell_view(grid, row_spans, col_spans, sr + 1, sc + offset)
            mask = _pattern_mask(swatch_cell, bg, grey)
            if mask is None:
                ok = False
                break
            direction_to_mask[direction] = mask
            swatch_sigs.add(mask.tobytes())

        if ok and set(direction_to_mask) == set(DIRECTIONS) and len(swatch_sigs) == 4:
            hits.append((legend_cells, direction_to_mask))

    assert len(hits) == 1, "expected exactly one corner legend block"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marked partition with the legend pattern chosen by edge-touching marker position and the color borrowed from the adjacent partition in that direction.

    Input structure:
    - Black background cells are separated by full grey grid lines into equal rectangular partitions.
    - Most ordinary partitions are either empty or contain exactly one colored marker pixel touching one partition edge from inside.
    - One corner contains a 2×4 legend block of partitions: the top legend row shows the four possible marker-edge positions, and the partition directly below each sample shows the pattern for that direction.

    Rule:
    - Detect the partition grid and read the corner legend by pairing each sample marker partition with the pattern partition beneath it.
    - For every non-legend partition that contains one marker, use its touched edge to select the legend pattern.
    - Look at the adjacent partition in that same direction; if that neighbor contains a marker, repaint the whole current partition interior with the selected pattern using the neighbor marker's color. If the neighbor is empty, outside the grid, or belongs to the legend, leave the partition unchanged.

    Key insight:
    - The marker position and the borrowed color come from different places: local edge contact selects the pattern, while the neighboring partition contributes the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    grey, grey_rows, grey_cols = _grid_color(g)
    row_spans = _spans(grey_rows, g.shape[0])
    col_spans = _spans(grey_cols, g.shape[1])
    assert len({r1 - r0 + 1 for r0, r1 in row_spans}) == 1, "partition heights must match"
    assert len({c1 - c0 + 1 for c0, c1 in col_spans}) == 1, "partition widths must match"

    non_grey = g[g != grey]
    assert non_grey.size > 0
    bg = _most_common(non_grey)

    legend_cells, direction_to_mask = _find_legend(g, row_spans, col_spans, bg, grey)

    markers: dict[tuple[int, int], tuple[str, int]] = {}
    for rr in range(len(row_spans)):
        for cc in range(len(col_spans)):
            if (rr, cc) in legend_cells:
                continue
            cell = _cell_view(g, row_spans, col_spans, rr, cc)
            mask = (cell != bg) & (cell != grey)
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            info = _marker_info(cell, bg, grey)
            assert info is not None and count == 1, "ordinary partitions must be empty or contain one valid edge marker"
            markers[(rr, cc)] = info

    out = g.copy()
    for (rr, cc), (direction, _) in markers.items():
        dr, dc = DELTAS[direction]
        nr, nc = rr + dr, cc + dc
        if not (0 <= nr < len(row_spans) and 0 <= nc < len(col_spans)):
            continue
        if (nr, nc) in legend_cells:
            continue
        neighbor = markers.get((nr, nc))
        if neighbor is None:
            continue

        fill_color = neighbor[1]
        patch = _cell_view(out, row_spans, col_spans, rr, cc)
        patch[:] = bg
        patch[direction_to_mask[direction]] = fill_color

    return out


def _cell_center_pos(cell_h: int, cell_w: int, direction: str) -> tuple[int, int]:
    mid_r = cell_h // 2
    mid_c = cell_w // 2
    if direction == UP:
        return 0, mid_c
    if direction == DOWN:
        return cell_h - 1, mid_c
    if direction == LEFT:
        return mid_r, 0
    if direction == RIGHT:
        return mid_r, cell_w - 1
    raise ValueError(direction)


def _pattern_library(cell_h: int, cell_w: int) -> list[np.ndarray]:
    rows = np.arange(cell_h)[:, None]
    cols = np.arange(cell_w)[None, :]
    def full(mask: np.ndarray) -> np.ndarray:
        return np.broadcast_to(mask, (cell_h, cell_w)).copy()

    return [
        full(rows % 2 == 0),
        full(rows % 2 == 1),
        full(cols % 2 == 0),
        full(cols % 2 == 1),
        full((rows + cols) % 2 == 0),
        full((rows + cols) % 2 == 1),
        np.ones((cell_h, cell_w), dtype=bool),
        full((rows == 0) | (rows == cell_h - 1) | (cols == 0) | (cols == cell_w - 1)),
    ]


def _distinct_patterns(rng: random.Random, cell_h: int, cell_w: int) -> dict[str, np.ndarray]:
    library = _pattern_library(cell_h, cell_w)
    chosen: list[np.ndarray] = []
    sigs: set[bytes] = set()
    for mask in rng.sample(library, k=len(library)):
        count = int(np.count_nonzero(mask))
        if count <= 1:
            continue
        sig = mask.tobytes()
        if sig in sigs:
            continue
        sigs.add(sig)
        chosen.append(mask.copy())
        if len(chosen) == 4:
            break
    assert len(chosen) == 4
    rng.shuffle(chosen)
    return {direction: chosen[idx] for idx, direction in enumerate(DIRECTIONS)}


def _cell_bounds(rr: int, cc: int, cell_h: int, cell_w: int) -> tuple[int, int, int, int]:
    r0 = 1 + rr * (cell_h + 1)
    c0 = 1 + cc * (cell_w + 1)
    return r0, r0 + cell_h, c0, c0 + cell_w


def _paint_marker(grid: np.ndarray, rr: int, cc: int, cell_h: int, cell_w: int, direction: str, color: int) -> None:
    r0, r1, c0, c1 = _cell_bounds(rr, cc, cell_h, cell_w)
    pr, pc = _cell_center_pos(cell_h, cell_w, direction)
    grid[r0 + pr, c0 + pc] = color


def _paint_pattern(grid: np.ndarray, rr: int, cc: int, cell_h: int, cell_w: int, mask: np.ndarray, color: int) -> None:
    r0, r1, c0, c1 = _cell_bounds(rr, cc, cell_h, cell_w)
    patch = grid[r0:r1, c0:c1]
    patch[mask] = color


def _choose_colors(rng: random.Random, count: int, exclude: set[int]) -> list[int]:
    pool = [color for color in PALETTE if color not in exclude]
    rng.shuffle(pool)
    out: list[int] = []
    while len(out) < count:
        if not pool:
            pool = [color for color in PALETTE if color not in exclude]
            rng.shuffle(pool)
        out.append(pool.pop())
    return out


def _legend_block(rows: int, cols: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, cols - 4
    if corner == "bl":
        return rows - 2, 0
    if corner == "br":
        return rows - 2, cols - 4
    raise ValueError(corner)


def _ordinary_cells(rows: int, cols: int, legend_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(rr, cc) for rr in range(rows) for cc in range(cols) if (rr, cc) not in legend_cells]


def _choose_fail_cells(
    rng: random.Random,
    rows: int,
    cols: int,
    legend_cells: set[tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    fail_cells: dict[str, tuple[int, int]] = {}
    edge_candidates = {
        UP: [(0, cc) for cc in range(cols)],
        DOWN: [(rows - 1, cc) for cc in range(cols)],
        LEFT: [(rr, 0) for rr in range(rows)],
        RIGHT: [(rr, cols - 1) for rr in range(rows)],
    }
    used: set[tuple[int, int]] = set()
    for direction in DIRECTIONS:
        candidates = [cell for cell in edge_candidates[direction] if cell not in legend_cells and cell not in used]
        rng.shuffle(candidates)
        assert candidates, f"missing failure candidate for {direction}"
        fail_cells[direction] = candidates[0]
        used.add(candidates[0])
    return fail_cells


def _success_source_candidates(
    rows: int,
    cols: int,
    legend_cells: set[tuple[int, int]],
    direction: str,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    dr, dc = DELTAS[direction]
    candidates: list[tuple[int, int]] = []
    for rr in range(rows):
        for cc in range(cols):
            if (rr, cc) in legend_cells or (rr, cc) in occupied:
                continue
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in legend_cells:
                continue
            candidates.append((rr, cc))
    return candidates


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.choice((3, 4))
    max_rows = (29 // (cell_h + 1))
    rows = rng.randint(4, max_rows)
    cols = rng.randint(6, 7)
    if rows * (cell_h + 1) + 1 > 30 or cols * (CELL_W + 1) + 1 > 30:
        return None

    corner = rng.choice(("tl", "tr", "bl", "br"))
    grey = rng.choice(GREY_CHOICES)

    h = rows * (cell_h + 1) + 1
    w = cols * (CELL_W + 1) + 1
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[:: cell_h + 1, :] = grey
    grid[:, :: CELL_W + 1] = grey

    legend_r0, legend_c0 = _legend_block(rows, cols, corner)
    legend_cells = {(legend_r0 + dr, legend_c0 + dc) for dr in range(2) for dc in range(4)}

    patterns = _distinct_patterns(rng, cell_h, CELL_W)
    legend_dirs = list(DIRECTIONS)
    rng.shuffle(legend_dirs)
    legend_colors = _choose_colors(rng, 4, {BACKGROUND, grey})
    for offset, direction in enumerate(legend_dirs):
        color = legend_colors[offset]
        _paint_marker(grid, legend_r0, legend_c0 + offset, cell_h, CELL_W, direction, color)
        _paint_pattern(grid, legend_r0 + 1, legend_c0 + offset, cell_h, CELL_W, patterns[direction], color)

    ordinary = _ordinary_cells(rows, cols, legend_cells)
    if len(ordinary) < 8:
        return None

    marker_colors = _choose_colors(rng, len(ordinary) + 8, {BACKGROUND, grey})
    color_idx = 0
    markers: dict[tuple[int, int], tuple[str, int]] = {}

    fail_cells = _choose_fail_cells(rng, rows, cols, legend_cells)
    for direction, cell in fail_cells.items():
        markers[cell] = (direction, marker_colors[color_idx])
        color_idx += 1

    occupied = set(markers)
    for direction in DIRECTIONS:
        candidates = _success_source_candidates(rows, cols, legend_cells, direction, occupied)
        rng.shuffle(candidates)
        if not candidates:
            return None
        source = candidates[0]
        dr, dc = DELTAS[direction]
        neighbor = (source[0] + dr, source[1] + dc)
        markers[source] = (direction, marker_colors[color_idx])
        color_idx += 1
        occupied.add(source)
        if neighbor not in markers:
            neighbor_direction = rng.choice(DIRECTIONS)
            markers[neighbor] = (neighbor_direction, marker_colors[color_idx])
            color_idx += 1
            occupied.add(neighbor)

    extra_prob = rng.uniform(0.18, 0.38)
    for cell in ordinary:
        if cell in markers:
            continue
        if rng.random() < extra_prob:
            markers[cell] = (rng.choice(DIRECTIONS), marker_colors[color_idx])
            color_idx += 1

    if len(markers) < 8:
        return None

    for (rr, cc), (direction, color) in markers.items():
        _paint_marker(grid, rr, cc, cell_h, CELL_W, direction, color)

    transformed = main(grid)
    if np.array_equal(transformed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
