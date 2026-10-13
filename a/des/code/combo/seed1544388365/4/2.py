import random
from collections import Counter

import numpy as np


_ROT_ORDERS = (4, 2, 1)
_NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _intervals_from_separators(separators: list[int], size: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(int(s) for s in separators)) + [int(size)]
    intervals: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:], strict=False):
        if right - left > 1:
            intervals.append((int(left + 1), int(right)))
    return intervals


def _find_separator_layout(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best = None
    for color in np.unique(grid):
        value = int(color)
        row_seps = [r for r in range(h) if bool(np.all(grid[r, :] == value))]
        col_seps = [c for c in range(w) if bool(np.all(grid[:, c] == value))]
        if len(row_seps) < 2 or len(col_seps) < 2:
            continue
        row_intervals = _intervals_from_separators(row_seps, h)
        col_intervals = _intervals_from_separators(col_seps, w)
        if not row_intervals or not col_intervals:
            continue
        row_sizes = {r1 - r0 for r0, r1 in row_intervals}
        col_sizes = {c1 - c0 for c0, c1 in col_intervals}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if next(iter(row_sizes)) != next(iter(col_sizes)):
            continue
        score = len(row_seps) + len(col_seps)
        candidate = (score, value, row_seps, col_seps)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None, "failed to locate a regular square-cell separator layout"
    return int(best[1]), list(best[2]), list(best[3])


def _infer_background(grid: np.ndarray, separator: int, row_intervals: list[tuple[int, int]], col_intervals: list[tuple[int, int]]) -> int:
    values: list[int] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            values.extend(int(v) for v in grid[r0:r1, c0:c1].ravel())
    values = [v for v in values if v != int(separator)]
    assert values, "cell interiors must be non-empty"
    return int(Counter(values).most_common(1)[0][0])


def _is_filled_rectangle(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    box = np.zeros_like(mask, dtype=bool)
    box[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    return bool(np.array_equal(mask, box))


def _legend_cell_color(cell: np.ndarray, background: int) -> int | None:
    mask = cell != int(background)
    if not bool(mask.any()) or bool(mask.all()):
        return None
    colors = np.unique(cell[mask])
    if len(colors) != 1:
        return None
    if not _is_filled_rectangle(mask):
        return None
    return int(colors[0])


def _find_legend(cells: list[list[np.ndarray]], background: int) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
    n_rows = len(cells)
    n_cols = len(cells[0])
    assert n_rows >= 3 and n_cols >= 1

    candidates: list[tuple[list[tuple[int, int]], list[int]]] = []
    for col in sorted({0, n_cols - 1}):
        for row0 in sorted({0, n_rows - 3}):
            positions = [(int(row0 + k), int(col)) for k in range(3)]
            colors: list[int] = []
            ok = True
            for pos in positions:
                color = _legend_cell_color(cells[pos[0]][pos[1]], background)
                if color is None:
                    ok = False
                    break
                colors.append(int(color))
            if ok and len(set(colors)) == 3:
                candidates.append((positions, colors))

    assert len(candidates) == 1, "legend must be a unique vertical stack in one corner column"
    positions, colors = candidates[0]
    pos_to_order = {
        positions[0]: 4,
        positions[1]: 2,
        positions[2]: 1,
    }
    order_to_color = {
        4: int(colors[0]),
        2: int(colors[1]),
        1: int(colors[2]),
    }
    return pos_to_order, order_to_color


def _rotation_order(mask: np.ndarray) -> int:
    if bool(np.array_equal(mask, np.rot90(mask, 1))):
        return 4
    if bool(np.array_equal(mask, np.rot90(mask, 2))):
        return 2
    return 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each non-legend cell by the legend color associated with its rotation-symmetry order.

    The grid is split into equal square cells by full grey separator rows and columns.
    A corner column contains three legend cells stacked vertically: top/middle/bottom
    correspond to rotational orders 4/2/1. Every other cell contains a black-background
    pattern whose order is determined by comparing the full cell mask against its 90° and
    180° rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator, row_seps, col_seps = _find_separator_layout(g)
    row_intervals = _intervals_from_separators(row_seps, g.shape[0])
    col_intervals = _intervals_from_separators(col_seps, g.shape[1])
    side_lengths = {r1 - r0 for r0, r1 in row_intervals} | {c1 - c0 for c0, c1 in col_intervals}
    assert len(side_lengths) == 1, "all cells must have the same square side length"

    cells = [[g[r0:r1, c0:c1] for c0, c1 in col_intervals] for r0, r1 in row_intervals]
    background = _infer_background(g, separator, row_intervals, col_intervals)
    legend_orders, order_to_color = _find_legend(cells, background)

    out = g.copy()
    for i, (r0, r1) in enumerate(row_intervals):
        for j, (c0, c1) in enumerate(col_intervals):
            if (i, j) in legend_orders:
                fill = int(order_to_color[int(legend_orders[(i, j)])])
            else:
                mask = cells[i][j] != int(background)
                assert bool(mask.any()), "non-legend cells must contain a visible pattern"
                fill = int(order_to_color[_rotation_order(mask)])
            out[r0:r1, c0:c1] = fill
    return out


# --------------- generator ---------------

_GEN_BG = 0
_GEN_SEPARATOR = 5
_GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_CELL_SIZES = [5, 6, 7]
_GEN_ROW_RANGE = (4, 6)
_GEN_COL_RANGE = (4, 6)
_GEN_ATTEMPTS = 300


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            components += 1
            stack = [(int(r), int(c))]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in _NEIGHBORS_8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
    return int(components)


def _orbit90(size: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(size):
        for c in range(size):
            pts = []
            rr, cc = int(r), int(c)
            for _ in range(4):
                pts.append((int(rr), int(cc)))
                rr, cc = int(cc), int(size - 1 - rr)
            orbit = tuple(sorted(set(pts)))
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _orbit180(size: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(size):
        for c in range(size):
            orbit = tuple(sorted({(int(r), int(c)), (int(size - 1 - r), int(size - 1 - c))}))
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _is_valid_pattern_mask(mask: np.ndarray, target_order: int) -> bool:
    size = int(mask.shape[0])
    area = int(mask.sum())
    if area < max(3, size - 1):
        return False
    if area > int(size * size * 0.45):
        return False
    if _rotation_order(mask) != int(target_order):
        return False
    if _is_filled_rectangle(mask):
        return False
    if _component_count(mask) > 4:
        return False
    return True


def _symmetry_mask_from_orbits(size: int, orbits: list[tuple[tuple[int, int], ...]], target_order: int, rng: random.Random) -> np.ndarray | None:
    low = 2
    high = min(len(orbits), max(3, size + 1))
    for _ in range(240):
        mask = np.zeros((size, size), dtype=bool)
        take = rng.randint(low, high)
        for orbit in rng.sample(orbits, take):
            for r, c in orbit:
                mask[int(r), int(c)] = True
        mask = np.rot90(mask, rng.randrange(4))
        if rng.random() < 0.35:
            mask = np.fliplr(mask)
        if _is_valid_pattern_mask(mask, target_order):
            return mask
    return None


def _random_walk_mask(size: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(320):
        mask = np.zeros((size, size), dtype=bool)
        r = rng.randrange(size)
        c = rng.randrange(size)
        mask[r, c] = True
        cells = {(int(r), int(c))}
        steps = rng.randint(max(4, size), max(5, int(size * size * 0.4)))
        for _ in range(steps):
            base_r, base_c = rng.choice(tuple(cells))
            dr, dc = rng.choice(_NEIGHBORS_8)
            nr = max(0, min(size - 1, base_r + dr))
            nc = max(0, min(size - 1, base_c + dc))
            cells.add((int(nr), int(nc)))
            mask[nr, nc] = True
        mask = np.rot90(mask, rng.randrange(4))
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if _is_valid_pattern_mask(mask, 1):
            return mask
    return None


def _generate_pattern_mask(size: int, order: int, rng: random.Random) -> np.ndarray:
    if int(order) == 4:
        result = _symmetry_mask_from_orbits(size, _orbit90(size), 4, rng)
    elif int(order) == 2:
        result = _symmetry_mask_from_orbits(size, _orbit180(size), 2, rng)
    else:
        result = _random_walk_mask(size, rng)
    if result is None:
        raise RuntimeError(f"failed to generate a pattern of order {order}")
    return result


def _make_legend_cell(size: int, color: int, rng: random.Random) -> np.ndarray:
    cell = np.full((size, size), _GEN_BG, dtype=int)
    block_h = rng.randint(2, max(2, size - 2))
    block_w = rng.randint(2, max(2, size - 2))
    r0 = rng.randint(1, size - block_h - 1)
    c0 = rng.randint(1, size - block_w - 1)
    cell[r0 : r0 + block_h, c0 : c0 + block_w] = int(color)
    return cell


def _cell_origin(index: int, cell_size: int) -> int:
    return 1 + int(index) * (int(cell_size) + 1)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    rows = rng.randint(*_GEN_ROW_RANGE)
    cols = rng.randint(*_GEN_COL_RANGE)
    cell_size = rng.choice(_GEN_CELL_SIZES)

    legend_col = rng.choice([0, cols - 1])
    legend_row0 = rng.choice(sorted({0, rows - 3}))
    legend_positions = {(legend_row0 + k, legend_col) for k in range(3)}

    legend_colors = rng.sample(_GEN_COLOR_POOL, 3)
    pattern_color_choices = [c for c in _GEN_COLOR_POOL if c not in legend_colors]
    pattern_color = int(rng.choice(pattern_color_choices))

    grid_h = rows * cell_size + rows + 1
    grid_w = cols * cell_size + cols + 1
    grid = np.full((grid_h, grid_w), _GEN_SEPARATOR, dtype=int)

    positions = [(r, c) for r in range(rows) for c in range(cols)]
    non_legend_positions = [pos for pos in positions if pos not in legend_positions]
    if len(non_legend_positions) < 3:
        return None

    order_list = [4, 2, 1]
    for _ in range(len(non_legend_positions) - 3):
        order_list.append(rng.choice(_ROT_ORDERS))
    rng.shuffle(order_list)

    used_masks: set[bytes] = set()
    order_by_pos: dict[tuple[int, int], int] = {}

    for idx, pos in enumerate(non_legend_positions):
        order = int(order_list[idx])
        mask = None
        for _ in range(200):
            candidate = _generate_pattern_mask(cell_size, order, rng)
            signature = candidate.tobytes()
            if signature in used_masks:
                continue
            used_masks.add(signature)
            mask = candidate
            break
        if mask is None:
            return None
        order_by_pos[pos] = order
        cell = np.full((cell_size, cell_size), _GEN_BG, dtype=int)
        cell[mask] = int(pattern_color)
        r0 = _cell_origin(pos[0], cell_size)
        c0 = _cell_origin(pos[1], cell_size)
        grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cell

    legend_order_colors = {4: legend_colors[0], 2: legend_colors[1], 1: legend_colors[2]}
    for offset, order in enumerate((4, 2, 1)):
        pos = (legend_row0 + offset, legend_col)
        r0 = _cell_origin(pos[0], cell_size)
        c0 = _cell_origin(pos[1], cell_size)
        grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = _make_legend_cell(cell_size, int(legend_order_colors[order]), rng)

    try:
        out = main(grid)
    except AssertionError:
        return None
    except RuntimeError:
        return None

    assert out.shape == grid.shape
    for r, c in non_legend_positions:
        expected = int(legend_order_colors[order_by_pos[(r, c)]])
        r0 = _cell_origin(r, cell_size)
        c0 = _cell_origin(c, cell_size)
        if not bool(np.all(out[r0 : r0 + cell_size, c0 : c0 + cell_size] == expected)):
            return None
    for offset, order in enumerate((4, 2, 1)):
        r, c = legend_row0 + offset, legend_col
        expected = int(legend_order_colors[order])
        r0 = _cell_origin(r, cell_size)
        c0 = _cell_origin(c, cell_size)
        if not bool(np.all(out[r0 : r0 + cell_size, c0 : c0 + cell_size] == expected)):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
