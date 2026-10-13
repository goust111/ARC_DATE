from __future__ import annotations

import random

import numpy as np


BG = 0
LATTICE = 1
RANK_COLORS = (2, 3, 4, 5, 7)
MARKER_COLOR_POOL = RANK_COLORS
GEN_ATTEMPTS = 500

_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_GEN_MARKER_COUNT_BAG: list[int] = []

_HOLE_LAYOUTS = {
    1: [((1, 1), ((0, 0),)), ((2, 2), ((0, 0),)), ((2, 2), ((1, 1),))],
    2: [
        ((1, 2), ((0, 0), (0, 1))),
        ((2, 1), ((0, 0), (1, 0))),
        ((2, 2), ((0, 0), (1, 1))),
        ((2, 2), ((0, 1), (1, 0))),
    ],
    3: [
        ((2, 2), ((0, 0), (0, 1), (1, 0))),
        ((2, 2), ((0, 0), (0, 1), (1, 1))),
        ((2, 2), ((0, 0), (1, 0), (1, 1))),
        ((2, 2), ((0, 1), (1, 0), (1, 1))),
    ],
    4: [((2, 2), ((0, 0), (0, 1), (1, 0), (1, 1)))],
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank embedded markers by enclosed black-hole count, then recolor the lattice bands crossing each marker.

    Input structure:
    - The background is black and blue pixels form a regular orthogonal lattice.
    - Three to five non-black, non-blue monochromatic marker objects are embedded in the lattice.
    - Each marker is 4-connected, has a unique color, and its bounding box contains a unique number of enclosed black holes.

    Rule:
    - Treat every monochromatic non-blue connected component as one marker and count the enclosed 4-connected black regions inside its bounding box.
    - Rank the markers from most holes to fewest holes and map ranks 1..5 to red, green, yellow, gray, and orange.
    - For each marker, collect every row and every column occupied by that marker, then recolor only the blue lattice pixels on those rows or columns to the marker's rank color.
    - If a blue lattice pixel belongs to multiple markers' row/column bands, keep the color of the higher-ranked marker.
    - Preserve every marker pixel and every black background pixel.

    Key insight:
    - A marker's topological complexity controls both its output color and its priority, so hole count must be inferred before any lattice-band recoloring conflicts can be resolved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _extract_markers(g)
    assert 3 <= len(markers) <= 5, "expected three to five markers"

    hole_counts = [_count_marker_holes(g, marker["coords"], marker["color"]) for marker in markers]
    assert len(set(hole_counts)) == len(hole_counts), "marker hole counts must be unique"

    order = sorted(range(len(markers)), key=lambda i: hole_counts[i], reverse=True)
    rank_by_marker = {idx: rank + 1 for rank, idx in enumerate(order)}
    assert max(rank_by_marker.values()) <= len(RANK_COLORS), "too many markers for the rank palette"

    h, w = g.shape
    row_ids = np.arange(h)[:, None]
    col_ids = np.arange(w)[None, :]
    best_rank = np.full((h, w), len(markers) + 1, dtype=int)
    best_color = g.copy()
    lattice_mask = g == LATTICE

    for idx, marker in enumerate(markers):
        rows = np.unique(marker["coords"][:, 0])
        cols = np.unique(marker["coords"][:, 1])
        band = lattice_mask & (np.isin(row_ids, rows) | np.isin(col_ids, cols))
        rank = rank_by_marker[idx]
        recolor = RANK_COLORS[rank - 1]
        better = band & (rank < best_rank)
        best_rank[better] = rank
        best_color[better] = recolor

    out = g.copy()
    out[best_rank <= len(markers)] = best_color[best_rank <= len(markers)]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice hole-ranking input")


def _extract_markers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    markers: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in {BG, LATTICE} or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            markers.append({"color": color, "coords": coords})

    return markers


def _count_marker_holes(grid: np.ndarray, coords: np.ndarray, color: int) -> int:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    black = sub == BG
    h, w = black.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            enclosed = True

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if black[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                        elif not black[nr, nc] and int(sub[nr, nc]) != color:
                            enclosed = False

            if not touches_edge and enclosed:
                holes += 1

    return holes


def _gen_attempt() -> np.ndarray | None:
    marker_count = _draw_marker_count()
    hole_counts = random.sample(range(5), marker_count)
    board_rows, board_cols, occupied_cells = _draw_board(marker_count)
    random.shuffle(occupied_cells)

    masks = [_make_marker_mask(hole_count) for hole_count in hole_counts]
    if _orders_match(hole_counts, [int(mask.sum()) for mask in masks]):
        return None

    row_heights = []
    for row_idx in range(board_rows):
        present = [masks[i].shape[0] for i, (rr, _cc) in enumerate(occupied_cells) if rr == row_idx]
        row_heights.append(max(present) + random.randint(0, 1))

    col_widths = []
    for col_idx in range(board_cols):
        present = [masks[i].shape[1] for i, (_rr, cc) in enumerate(occupied_cells) if cc == col_idx]
        col_widths.append(max(present) + random.randint(0, 1))

    row_gaps = [random.randint(1, 2) for _ in range(board_rows - 1)]
    col_gaps = [random.randint(1, 2) for _ in range(board_cols - 1)]

    need_h = sum(row_heights) + sum(row_gaps) + 2
    need_w = sum(col_widths) + sum(col_gaps) + 2
    if need_h > 20 or need_w > 20:
        return None

    grid_h = random.randint(max(15, need_h), 20)
    grid_w = random.randint(max(15, need_w), 20)
    row_starts = _axis_starts(row_heights, row_gaps, grid_h)
    col_starts = _axis_starts(col_widths, col_gaps, grid_w)

    row_offsets = {}
    for row_idx in range(board_rows):
        max_marker_h = max(masks[i].shape[0] for i, (rr, _cc) in enumerate(occupied_cells) if rr == row_idx)
        row_offsets[row_idx] = random.randint(0, row_heights[row_idx] - max_marker_h)

    col_offsets = {}
    for col_idx in range(board_cols):
        max_marker_w = max(masks[i].shape[1] for i, (_rr, cc) in enumerate(occupied_cells) if cc == col_idx)
        col_offsets[col_idx] = random.randint(0, col_widths[col_idx] - max_marker_w)

    row_gap = random.randint(2, 4)
    col_gap = random.randint(2, 4)
    row_phase = random.randint(0, row_gap)
    col_phase = random.randint(0, col_gap)
    base = _build_lattice(grid_h, grid_w, row_gap, col_gap, row_phase, col_phase)

    rank_order = sorted(range(marker_count), key=lambda i: hole_counts[i], reverse=True)
    rank_color_by_marker = {idx: RANK_COLORS[rank] for rank, idx in enumerate(rank_order)}
    input_colors = _choose_input_colors(marker_count, rank_color_by_marker)
    if input_colors is None:
        return None

    specs: list[dict] = []
    grid = base.copy()
    for idx, ((board_r, board_c), mask) in enumerate(zip(occupied_cells, masks)):
        top = row_starts[board_r] + row_offsets[board_r]
        left = col_starts[board_c] + col_offsets[board_c]
        ph, pw = mask.shape
        grid[top : top + ph, left : left + pw] = BG
        color = input_colors[idx]
        grid[top : top + ph, left : left + pw][mask] = color

        coords = np.argwhere(mask)
        coords[:, 0] += top
        coords[:, 1] += left
        specs.append(
            {
                "color": color,
                "coords": coords,
                "holes": hole_counts[idx],
                "rank": rank_order.index(idx) + 1,
                "rank_color": rank_color_by_marker[idx],
                "area": int(mask.sum()),
            }
        )

    if not _valid_marker_separation(specs):
        return None

    expected, owner, coverage = _oracle_output(grid, specs)
    if not _scene_has_useful_conflicts(owner, coverage, specs):
        return None
    if _orders_match([spec["holes"] for spec in specs], [spec["area"] for spec in specs]):
        return None
    if not any(spec["color"] != spec["rank_color"] for spec in specs):
        return None
    if not _visible_band_change(grid, expected):
        return None

    detected_markers = _extract_markers(grid)
    if len(detected_markers) != marker_count:
        return None
    detected_holes = sorted(_count_marker_holes(grid, m["coords"], m["color"]) for m in detected_markers)
    if detected_holes != sorted(hole_counts):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def _draw_marker_count() -> int:
    if not _GEN_MARKER_COUNT_BAG:
        _GEN_MARKER_COUNT_BAG.extend([3, 4, 5, 4, 3, 5, 4, 5])
        random.shuffle(_GEN_MARKER_COUNT_BAG)
    return _GEN_MARKER_COUNT_BAG.pop()


def _draw_board(marker_count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if marker_count == 3:
        cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
        missing = random.choice(cells)
        return 2, 2, [cell for cell in cells if cell != missing]
    if marker_count == 4:
        return 2, 2, [(0, 0), (0, 1), (1, 0), (1, 1)]

    if random.random() < 0.5:
        cells = [(r, c) for r in range(3) for c in range(2)]
        missing = random.choice(cells)
        return 3, 2, [cell for cell in cells if cell != missing]
    cells = [(r, c) for r in range(2) for c in range(3)]
    missing = random.choice(cells)
    return 2, 3, [cell for cell in cells if cell != missing]


def _make_marker_mask(hole_count: int) -> np.ndarray:
    for _ in range(200):
        if hole_count == 0:
            mask = np.ones((random.randint(3, 5), random.randint(3, 5)), dtype=bool)
        else:
            (rows, cols), positions = random.choice(_HOLE_LAYOUTS[hole_count])
            row_sizes = [random.randint(1, 2) for _ in range(rows)]
            col_sizes = [random.randint(1, 2) for _ in range(cols)]
            if hole_count >= 3:
                row_sizes = [1 if random.random() < 0.7 else size for size in row_sizes]
                col_sizes = [1 if random.random() < 0.7 else size for size in col_sizes]
            h = sum(row_sizes) + rows + 1
            w = sum(col_sizes) + cols + 1
            mask = np.ones((h, w), dtype=bool)
            row_starts = _hole_axis_starts(row_sizes)
            col_starts = _hole_axis_starts(col_sizes)
            for rr, cc in positions:
                r0 = row_starts[rr]
                c0 = col_starts[cc]
                mask[r0 : r0 + row_sizes[rr], c0 : c0 + col_sizes[cc]] = False

        mask = _mutate_mask(mask, hole_count)
        mask = _random_transform(mask)
        if _valid_marker_mask(mask, hole_count):
            return mask
    raise RuntimeError(f"failed to build a {hole_count}-hole marker")


def _hole_axis_starts(sizes: list[int]) -> list[int]:
    starts = []
    pos = 1
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts


def _mutate_mask(mask: np.ndarray, hole_count: int) -> np.ndarray:
    out = _crop_mask(mask)
    for _ in range(random.randint(0, 4)):
        candidates = _boundary_cells(out)
        random.shuffle(candidates)
        changed = False
        for r, c in candidates:
            trial = out.copy()
            trial[r, c] = False
            trial = _crop_mask(trial)
            if _valid_marker_mask(trial, hole_count):
                out = trial
                changed = True
                break
        if not changed:
            break
    return out


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in _NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _valid_marker_mask(mask: np.ndarray, hole_count: int) -> bool:
    if mask.size == 0 or not mask.any():
        return False
    mask = _crop_mask(mask)
    h, w = mask.shape
    area = int(mask.sum())
    if h < 3 or w < 3 or h > 7 or w > 7:
        return False
    if area < 7 or area > 32:
        return False
    if not _is_connected(mask):
        return False
    return _count_mask_holes(mask) == hole_count


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _count_mask_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    background = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not background[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and background[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0].copy()
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _axis_starts(sizes: list[int], gaps: list[int], total: int) -> list[int]:
    used = sum(sizes) + sum(gaps)
    slack = total - used
    left_margin = random.randint(1, max(1, slack - 1))
    starts = []
    pos = left_margin
    for idx, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if idx < len(gaps):
            pos += gaps[idx]
    return starts


def _build_lattice(h: int, w: int, row_gap: int, col_gap: int, row_phase: int, col_phase: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    row_lines = list(range(row_phase, h, row_gap + 1))
    col_lines = list(range(col_phase, w, col_gap + 1))
    assert len(row_lines) >= 3 and len(col_lines) >= 3
    grid[row_lines, :] = LATTICE
    grid[:, col_lines] = LATTICE
    return grid


def _choose_input_colors(marker_count: int, rank_color_by_marker: dict[int, int]) -> list[int] | None:
    for _ in range(100):
        colors = random.sample(MARKER_COLOR_POOL, marker_count)
        if all(colors[idx] != rank_color_by_marker[idx] for idx in range(marker_count)):
            return colors
    return None


def _valid_marker_separation(specs: list[dict]) -> bool:
    intervals = []
    for spec in specs:
        rows = spec["coords"][:, 0]
        cols = spec["coords"][:, 1]
        intervals.append((int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())))

    for i, (r0a, r1a, c0a, c1a) in enumerate(intervals):
        for r0b, r1b, c0b, c1b in intervals[i + 1 :]:
            row_overlap = max(r0a, r0b) <= min(r1a, r1b)
            col_overlap = max(c0a, c0b) <= min(c1a, c1b)
            if row_overlap and col_overlap:
                return False
    return True


def _oracle_output(grid: np.ndarray, specs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    lattice = grid == LATTICE
    coverage = np.zeros((h, w), dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    owner_rank = np.full((h, w), len(specs) + 1, dtype=int)

    for idx, spec in enumerate(specs):
        row_mask = np.zeros(h, dtype=bool)
        col_mask = np.zeros(w, dtype=bool)
        row_mask[np.unique(spec["coords"][:, 0])] = True
        col_mask[np.unique(spec["coords"][:, 1])] = True
        band = lattice & (row_mask[:, None] | col_mask[None, :])
        coverage += band.astype(int)
        better = band & (spec["rank"] < owner_rank)
        owner_rank[better] = spec["rank"]
        owner[better] = idx

    out = grid.copy()
    for idx, spec in enumerate(specs):
        out[owner == idx] = spec["rank_color"]
    return out, owner, coverage


def _scene_has_useful_conflicts(owner: np.ndarray, coverage: np.ndarray, specs: list[dict]) -> bool:
    if not np.any(coverage >= 2):
        return False
    for idx in range(len(specs)):
        if not np.any(owner == idx):
            return False
    return True


def _orders_match(values_a: list[int], values_b: list[int]) -> bool:
    order_a = sorted(range(len(values_a)), key=lambda i: values_a[i], reverse=True)
    order_b = sorted(range(len(values_b)), key=lambda i: values_b[i], reverse=True)
    return order_a == order_b


def _visible_band_change(inp: np.ndarray, out: np.ndarray) -> bool:
    diff = (inp != out) & (inp == LATTICE)
    return int(diff.sum()) >= 6
