import random
from collections import Counter, deque

import numpy as np


BG = 0
GEN_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_line_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert colors, "expected at least one foreground color"
    return max(colors, key=lambda c: int(np.sum(grid == c)))


def _legend_blocks(grid: np.ndarray, bg: int, line_color: int):
    h, w = grid.shape
    blocks = []
    occupied = np.zeros((h, w), dtype=bool)
    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            key = int(patch[0, 0])
            val = int(patch[1, 1])
            if key in (bg, line_color) or val in (bg, line_color) or key == val:
                continue
            border_mask = np.zeros((3, 3), dtype=bool)
            border_mask[0, :] = True
            border_mask[:, 0] = True
            value_mask = np.zeros((3, 3), dtype=bool)
            value_mask[1:, 1:] = True
            if np.all(patch[border_mask] == key) and np.all(patch[value_mask] == val):
                blocks.append((r, c, key, val))
                occupied[r : r + 3, c : c + 3] = True
    assert blocks, "failed to detect legend blocks"
    mapping = {}
    for _, _, key, val in blocks:
        if key in mapping:
            assert mapping[key] == val, "legend key must map unambiguously"
        mapping[key] = val
    return mapping, occupied


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    line_color = _find_line_color(g, bg)
    mapping, legend_mask = _legend_blocks(g, bg, line_color)

    open_mask = (g != line_color) & (~legend_mask)
    out = g.copy()
    h, w = g.shape

    for comp in _components(open_mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        markers = [(r, c) for r, c in comp if int(g[r, c]) != bg]
        assert len(markers) == 1, "each enclosed region must contain exactly one marker"
        marker_r, marker_c = markers[0]
        fill_color = mapping[int(g[marker_r, marker_c])]
        for r, c in comp:
            if int(g[r, c]) == bg:
                out[r, c] = fill_color
    return out


def _legend_patch(key: int, val: int) -> np.ndarray:
    patch = np.full((3, 3), key, dtype=int)
    patch[1:, 1:] = val
    return patch


def _pick_distinct(rng: random.Random, banned: set[int], n: int) -> list[int]:
    pool = [c for c in range(1, 10) if c not in banned]
    rng.shuffle(pool)
    assert len(pool) >= n
    return pool[:n]


def _partition_lengths(rng: random.Random, count: int, lo: int, hi: int) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(count)]


def _expected_output(
    grid: np.ndarray,
    region_boxes: list[tuple[int, int, int, int]],
    region_keys: list[int],
    mapping: dict[int, int],
    marker_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for (r0, c0, r1, c1), key, (mr, mc) in zip(region_boxes, region_keys, marker_positions):
        fill = mapping[key]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if (r, c) != (mr, mc) and out[r, c] == BG:
                    out[r, c] = fill
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    total_h = rng.randint(20, 28)
    total_w = rng.randint(22, 30)

    legend_count = rng.randint(2, 4)
    legend_rows = rng.choice([1, 2]) if legend_count >= 3 else 1
    legend_cols = (legend_count + legend_rows - 1) // legend_rows
    legend_h = legend_rows * 3 + (legend_rows - 1)
    legend_w = legend_cols * 3 + (legend_cols - 1)

    legend_r0 = total_h - legend_h - 1
    legend_c0 = total_w - legend_w - 1
    if legend_r0 < 12 or legend_c0 < 12:
        return None

    n_row_regions = rng.randint(2, 3)
    n_col_regions = rng.randint(2, 3)
    row_sizes = _partition_lengths(rng, n_row_regions, 2, 5)
    col_sizes = _partition_lengths(rng, n_col_regions, 2, 6)
    board_h = sum(row_sizes) + n_row_regions + 1
    board_w = sum(col_sizes) + n_col_regions + 1

    if board_h + 4 > legend_r0 or board_w + 4 > legend_c0:
        return None

    board_r0 = rng.randint(1, max(1, legend_r0 - board_h - 2))
    board_c0 = rng.randint(1, max(1, legend_c0 - board_w - 2))
    board_r1 = board_r0 + board_h - 1
    board_c1 = board_c0 + board_w - 1

    grid = np.zeros((total_h, total_w), dtype=int)
    line_color = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])

    banned = {line_color, BG}
    key_count = min(legend_count, n_row_regions * n_col_regions)
    keys = _pick_distinct(rng, banned, key_count)
    banned.update(keys)
    values = _pick_distinct(rng, banned, key_count)
    mapping = dict(zip(keys, values))

    row_walls = [board_r0]
    cur = board_r0
    for size in row_sizes:
        cur += size + 1
        row_walls.append(cur)
    col_walls = [board_c0]
    cur = board_c0
    for size in col_sizes:
        cur += size + 1
        col_walls.append(cur)

    for r in row_walls:
        grid[r, board_c0 : board_c1 + 1] = line_color
    for c in col_walls:
        grid[board_r0 : board_r1 + 1, c] = line_color

    region_boxes = []
    marker_positions = []
    region_keys = []
    all_regions = [(ri, ci) for ri in range(n_row_regions) for ci in range(n_col_regions)]
    rng.shuffle(all_regions)
    key_assignment = []
    for idx, rc in enumerate(all_regions):
        key_assignment.append((rc, keys[idx % key_count]))
    rng.shuffle(key_assignment)
    key_by_region = {rc: key for rc, key in key_assignment}

    for ri in range(n_row_regions):
        for ci in range(n_col_regions):
            r0 = row_walls[ri]
            r1 = row_walls[ri + 1]
            c0 = col_walls[ci]
            c1 = col_walls[ci + 1]
            cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            if len(cells) < 4:
                return None
            marker = rng.choice(cells)
            key = key_by_region[(ri, ci)]
            grid[marker] = key
            region_boxes.append((r0, c0, r1, c1))
            marker_positions.append(marker)
            region_keys.append(key)

    cells = []
    for i, (key, val) in enumerate(mapping.items()):
        rr = i // legend_cols
        cc = i % legend_cols
        pr = legend_r0 + rr * 4
        pc = legend_c0 + cc * 4
        patch = _legend_patch(key, val)
        grid[pr : pr + 3, pc : pc + 3] = patch
        cells.append((pr, pc))

    expected = _expected_output(grid, region_boxes, region_keys, mapping, marker_positions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
