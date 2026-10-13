import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
AZURE = 8
MAROON = 9

SEED_COLORS = (BLUE, RED, GREEN, YELLOW)
PRIORITY = ("up", "right", "down", "left")
DIRECTION_MODES = {
    "up": [("up",), ("up", "right"), ("up", "down"), ("up", "left")],
    "right": [("right",), ("right", "down"), ("right", "left")],
    "down": [("down",), ("down", "left")],
    "left": [("left",)],
}
MAX_GENERATE_ATTEMPTS = 300
GRID_CONFIGS = [
    (n_rows, n_cols, cell_h, cell_w)
    for cell_h in range(5, 9)
    for cell_w in range(5, 9)
    for n_rows in range(2, 5)
    for n_cols in range(2, 5)
    if n_rows * cell_h + (n_rows + 1) <= 30 and n_cols * cell_w + (n_cols + 1) <= 30
]


def _contiguous_spans(line_mask):
    spans = []
    start = None
    for idx, is_line in enumerate(line_mask):
        if not is_line and start is None:
            start = idx
        elif is_line and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(line_mask) - 1))
    return spans


def _parse_partition_grid(grid):
    g = np.asarray(grid, dtype=int)
    best = None
    for color in map(int, np.unique(g)):
        row_mask = np.all(g == color, axis=1)
        col_mask = np.all(g == color, axis=0)
        if row_mask.sum() < 1 or col_mask.sum() < 1:
            continue
        if np.any(row_mask[:-1] & row_mask[1:]) or np.any(col_mask[:-1] & col_mask[1:]):
            continue
        row_spans = _contiguous_spans(row_mask)
        col_spans = _contiguous_spans(col_mask)
        if not row_spans or not col_spans:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        score = int(row_mask.sum() + col_mask.sum())
        if best is None or score > best[0]:
            best = (score, color, row_spans, col_spans)
    assert best is not None, "failed to detect the partition lattice"
    return best[1], best[2], best[3]


def _distance_map(r, c, h, w):
    return {
        "up": int(r),
        "right": int(w - 1 - c),
        "down": int(h - 1 - r),
        "left": int(c),
    }


def _resolve_direction(r, c, h, w):
    dists = _distance_map(r, c, h, w)
    best = min(dists.values())
    nearest = tuple(direction for direction in PRIORITY if dists[direction] == best)
    return nearest[0], nearest, best


def _bar_coords(r, c, direction, length):
    dr, dc = {
        "up": (-1, 0),
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
    }[direction]
    return [(int(r + step * dr), int(c + step * dc)) for step in range(length)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored bar toward the seed's nearest partition edge.

    The grid is divided by single-cell lattice lines. Each partition contains
    one seed pixel and one azure connected block; the seed's nearest boundary
    chooses the growth direction and the azure area chooses the growth length.
    """
    g = np.asarray(grid, dtype=int)
    line_color, row_spans, col_spans = _parse_partition_grid(g)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            seed_positions = np.argwhere((cell != BLACK) & (cell != AZURE) & (cell != line_color))
            assert len(seed_positions) == 1, "each partition must contain exactly one seed"
            sr, sc = map(int, seed_positions[0])
            seed_color = int(cell[sr, sc])

            azure_area = int(np.count_nonzero(cell == AZURE))
            assert azure_area >= 1, "each partition must contain one azure block"

            direction, _nearest, nearest_dist = _resolve_direction(sr, sc, cell.shape[0], cell.shape[1])
            bar_length = min(int(azure_area), int(nearest_dist + 1))
            for rr, cc in _bar_coords(sr, sc, direction, bar_length):
                gr = int(r0 + rr)
                gc = int(c0 + cc)
                if out[gr, gc] != AZURE and out[gr, gc] != line_color:
                    out[gr, gc] = seed_color
    return out


def _draw_lattice_grid(n_rows, n_cols, cell_h, cell_w):
    h = n_rows * cell_h + (n_rows + 1)
    w = n_cols * cell_w + (n_cols + 1)
    grid = np.full((h, w), BLACK, dtype=int)

    for r in range(0, h, cell_h + 1):
        grid[r, :] = MAROON
    grid[h - 1, :] = MAROON
    for c in range(0, w, cell_w + 1):
        grid[:, c] = MAROON
    grid[:, w - 1] = MAROON
    return grid


def _partition_origin(row_idx, col_idx, cell_h, cell_w):
    return 1 + row_idx * (cell_h + 1), 1 + col_idx * (cell_w + 1)


def _positions_by_mode(cell_h, cell_w):
    by_mode = {}
    for r in range(cell_h):
        for c in range(cell_w):
            direction, nearest, nearest_dist = _resolve_direction(r, c, cell_h, cell_w)
            if nearest_dist < 1:
                continue
            by_mode.setdefault(nearest, []).append((int(r), int(c), int(nearest_dist + 1), direction))
    return by_mode


def _choose_mode(by_mode, direction, want_tie):
    preferred = [mode for mode in DIRECTION_MODES[direction] if by_mode.get(mode)]
    if not preferred:
        preferred = [mode for mode in by_mode if mode and mode[0] == direction]
    if not preferred:
        return None
    strict = [mode for mode in preferred if len(mode) == 1]
    tied = [mode for mode in preferred if len(mode) > 1]
    if want_tie and tied:
        return random.choice(tied)
    if (not want_tie) and strict:
        return random.choice(strict)
    return random.choice(preferred)


def _choose_area(available_len, want_truncate):
    if want_truncate and available_len < 6:
        return random.randint(available_len + 1, 6)
    max_len = min(available_len, 6)
    min_len = 1 if random.random() < 0.2 else min(2, max_len)
    return random.randint(min_len, max_len)


def _random_polyomino(area):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (int(r + dr), int(c + dc))
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((int(r - min_r), int(c - min_c)) for r, c in cells)
    return normalized


def _place_azure_shape(cell_h, cell_w, area, forbidden):
    for _ in range(120):
        shape = _random_polyomino(area)
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        if shape_h > cell_h or shape_w > cell_w:
            continue
        offsets = [
            (r_off, c_off)
            for r_off in range(cell_h - shape_h + 1)
            for c_off in range(cell_w - shape_w + 1)
        ]
        random.shuffle(offsets)
        for r_off, c_off in offsets:
            coords = {(int(r_off + r), int(c_off + c)) for r, c in shape}
            if coords & forbidden:
                continue
            return coords
    return None


def _gen_attempt():
    n_rows, n_cols, cell_h, cell_w = random.choice(GRID_CONFIGS)
    grid = _draw_lattice_grid(n_rows, n_cols, cell_h, cell_w)

    cell_indices = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    random.shuffle(cell_indices)
    direction_cycle = random.sample(list(PRIORITY), len(PRIORITY))
    color_cycle = random.sample(list(SEED_COLORS), len(SEED_COLORS))
    tie_offset = random.randint(0, 1)
    truncate_offset = random.randint(0, 1)

    changed_cells = 0
    for idx, (pr, pc) in enumerate(cell_indices):
        direction = direction_cycle[idx % len(direction_cycle)]
        want_tie = (idx + tie_offset) % 2 == 0
        want_truncate = (idx + truncate_offset) % 2 == 0

        by_mode = _positions_by_mode(cell_h, cell_w)
        mode = _choose_mode(by_mode, direction, want_tie)
        if mode is None:
            return None

        candidates = by_mode[mode][:]
        random.shuffle(candidates)
        if want_truncate:
            trunc_candidates = [item for item in candidates if item[2] < 6]
            if trunc_candidates:
                candidates = trunc_candidates
            else:
                want_truncate = False
        if not candidates:
            return None

        sr, sc, available_len, _ = candidates[0]
        area = _choose_area(available_len, want_truncate)
        actual_len = min(area, available_len)
        changed_cells += int(actual_len > 1)

        seed_color = color_cycle[(idx + random.randint(0, len(color_cycle) - 1)) % len(color_cycle)]
        bar_coords = set(_bar_coords(sr, sc, mode[0], actual_len))
        azure_coords = _place_azure_shape(cell_h, cell_w, area, bar_coords)
        if azure_coords is None:
            return None

        cell_r0, cell_c0 = _partition_origin(pr, pc, cell_h, cell_w)
        grid[cell_r0 + sr, cell_c0 + sc] = seed_color
        for rr, cc in azure_coords:
            grid[cell_r0 + rr, cell_c0 + cc] = AZURE

    if changed_cells == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATE_ATTEMPTS} attempts")
