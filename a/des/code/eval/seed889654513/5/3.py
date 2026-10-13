import random

import numpy as np


RED = 2
GREY = 5

GEN_ROWS_RANGE = (2, 4)
GEN_COLS_RANGE = (3, 5)
GEN_CELL_SIZE_RANGE = (3, 6)
GEN_MARGIN_RANGE = (1, 3)
GEN_GRID_LEGEND_GAP_RANGE = (2, 5)
GEN_LEGEND_SIZE_RANGE = (1, 3)
GEN_LEGEND_GAP_RANGE = (1, 2)
GEN_ATTEMPTS = 500


def _bg_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _symmetry_kind(mask: np.ndarray) -> int:
    lr = np.array_equal(mask, np.fliplr(mask))
    ud = np.array_equal(mask, np.flipud(mask))
    mapping = {
        (False, False): 0,
        (True, False): 1,
        (False, True): 2,
        (True, True): 3,
    }
    return mapping[(lr, ud)]


def _extract_cells(grid: np.ndarray) -> tuple[int, list[list[np.ndarray]], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    sep_mask = g == GREY
    assert sep_mask.any(), "grey separators are required"

    bbox = _bbox_from_coords(np.argwhere(sep_mask))
    r0, c0, r1, c1 = bbox

    row_seps = [r for r in range(r0, r1 + 1) if np.all(g[r, c0 : c1 + 1] == GREY)]
    col_seps = [c for c in range(c0, c1 + 1) if np.all(g[r0 : r1 + 1, c] == GREY)]
    assert len(row_seps) >= 2 and len(col_seps) >= 2, "separator lattice not found"

    row_steps = {b - a for a, b in zip(row_seps, row_seps[1:], strict=False)}
    col_steps = {b - a for a, b in zip(col_seps, col_seps[1:], strict=False)}
    assert len(row_steps) == 1 and len(col_steps) == 1, "separator spacing must be regular"
    row_step = next(iter(row_steps))
    col_step = next(iter(col_steps))
    assert row_step == col_step and row_step >= 2, "cells must be square with width >= 1"
    cell_size = row_step - 1

    cells: list[list[np.ndarray]] = []
    for i in range(len(row_seps) - 1):
        row: list[np.ndarray] = []
        rr0 = row_seps[i] + 1
        rr1 = row_seps[i + 1]
        for j in range(len(col_seps) - 1):
            cc0 = col_seps[j] + 1
            cc1 = col_seps[j + 1]
            cell = g[rr0:rr1, cc0:cc1]
            assert cell.shape == (cell_size, cell_size), "all cells must share the same square size"
            row.append(cell.copy())
        cells.append(row)

    return _bg_color(g), cells, bbox


def _legend_colors(grid: np.ndarray, bg: int, grid_bbox: tuple[int, int, int, int]) -> list[int]:
    _, _, _, grid_right = grid_bbox
    region = grid[:, grid_right + 1 :]
    assert region.size > 0, "legend area missing"

    squares: list[tuple[int, int, int]] = []
    for comp in _components(region != bg):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        block = region[r0 : r1 + 1, c0 : c1 + 1]
        if block.shape[0] != block.shape[1]:
            continue
        if comp.shape[0] != block.size:
            continue
        vals = block.ravel()
        if len(set(int(v) for v in vals.tolist())) != 1:
            continue
        color = int(vals[0])
        if color == bg:
            continue
        squares.append((r0, c0, color))

    squares.sort()
    assert len(squares) == 4, "legend must contain four solid squares"
    return [color for _, _, color in squares]


def _output_from_cells(cells: list[list[np.ndarray]], bg: int, legend_colors: list[int]) -> np.ndarray:
    n = len(cells)
    m = len(cells[0])
    out = np.zeros((n, m), dtype=int)
    for r in range(n):
        for c in range(m):
            kind = _symmetry_kind(cells[r][c] != bg)
            out[r, c] = legend_colors[kind]

    for r in range(n):
        for c in range(m):
            if r + 1 < n and np.array_equal(cells[r][c], cells[r + 1][c]):
                out[r, c] = RED
                out[r + 1, c] = RED
            if c + 1 < m and np.array_equal(cells[r][c], cells[r][c + 1]):
                out[r, c] = RED
                out[r, c + 1] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each cell's sprite by left-right/up-down symmetry using the right-side legend,
    then mark horizontally or vertically adjacent duplicate sprites in red.
    """
    g = np.asarray(grid, dtype=int)
    bg, cells, grid_bbox = _extract_cells(g)
    legend_colors = _legend_colors(g, bg, grid_bbox)
    return _output_from_cells(cells, bg, legend_colors)


def _orbit_list(size: int, kind: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(size):
        for c in range(size):
            if kind == 1:
                orbit = {(r, c), (r, size - 1 - c)}
            elif kind == 2:
                orbit = {(r, c), (size - 1 - r, c)}
            elif kind == 3:
                orbit = {
                    (r, c),
                    (r, size - 1 - c),
                    (size - 1 - r, c),
                    (size - 1 - r, size - 1 - c),
                }
            else:
                orbit = {(r, c)}
            key = tuple(sorted(orbit))
            if key not in seen:
                seen.add(key)
                orbits.append(list(key))
    return orbits


def _make_mask(size: int, kind: int) -> np.ndarray:
    min_fg = max(3, size - 1)
    area = size * size

    for _ in range(300):
        if kind == 0:
            density = random.uniform(0.25, 0.55)
            mask = np.array(
                [[random.random() < density for _ in range(size)] for _ in range(size)],
                dtype=bool,
            )
        else:
            density = random.uniform(0.25, 0.65)
            mask = np.zeros((size, size), dtype=bool)
            for orbit in _orbit_list(size, kind):
                if random.random() >= density:
                    continue
                for r, c in orbit:
                    mask[r, c] = True

        fg = int(mask.sum())
        if fg < min_fg or fg > area - 2:
            continue
        if _symmetry_kind(mask) != kind:
            continue

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if (r1 - r0 + 1) + (c1 - c0 + 1) < 5:
            continue
        return mask

    raise RuntimeError(f"failed to generate symmetry class {kind} for size {size}")


def _build_cell(size: int, bg: int, color: int, kind: int) -> np.ndarray:
    cell = np.full((size, size), int(bg), dtype=int)
    mask = _make_mask(size, kind)
    cell[mask] = int(color)
    return cell


def _adjacent_pairs(rows: int, cols: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if r + 1 < rows:
                pairs.append((idx, (r + 1) * cols + c))
            if c + 1 < cols:
                pairs.append((idx, r * cols + (c + 1)))
    return pairs


def _has_adjacent_duplicate(cells: list[list[np.ndarray]]) -> bool:
    rows = len(cells)
    cols = len(cells[0])
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and np.array_equal(cells[r][c], cells[r + 1][c]):
                return True
            if c + 1 < cols and np.array_equal(cells[r][c], cells[r][c + 1]):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GEN_ROWS_RANGE)
    cols = random.randint(*GEN_COLS_RANGE)
    size = random.randint(*GEN_CELL_SIZE_RANGE)

    kinds = [0, 1, 2, 3]
    while len(kinds) < rows * cols:
        kinds.append(random.randrange(4))
    random.shuffle(kinds)

    want_duplicates = random.choice([False, True])
    duplicate_pair: tuple[int, int] | None = None

    if want_duplicates:
        counts = [kinds.count(kind) for kind in range(4)]
        candidates: list[tuple[int, int]] = []
        for a, b in _adjacent_pairs(rows, cols):
            ka = kinds[a]
            kb = kinds[b]
            if ka == kb or counts[kb] > 1:
                candidates.append((a, b))
            if ka == kb or counts[ka] > 1:
                candidates.append((b, a))
        if not candidates:
            return None
        source, target = random.choice(candidates)
        kinds[target] = kinds[source]
        duplicate_pair = (source, target)

    usable_colors = [c for c in range(10) if c not in (RED, GREY)]
    bg = random.choice(usable_colors)
    remaining = [c for c in usable_colors if c != bg]
    legend_colors = random.sample(remaining, 4)
    sprite_pool = [c for c in remaining if c not in legend_colors]
    sprite_color_count = random.randint(2, len(sprite_pool))
    sprite_colors = random.sample(sprite_pool, sprite_color_count)

    flat_cells: list[np.ndarray] = []
    for kind in kinds:
        color = random.choice(sprite_colors)
        flat_cells.append(_build_cell(size, bg, color, kind))

    if duplicate_pair is not None:
        source, target = duplicate_pair
        flat_cells[target] = flat_cells[source].copy()

    cells = [flat_cells[r * cols : (r + 1) * cols] for r in range(rows)]
    if _has_adjacent_duplicate(cells) != want_duplicates:
        return None

    grid_h = rows * size + (rows + 1)
    grid_w = cols * size + (cols + 1)
    legend_size = random.randint(*GEN_LEGEND_SIZE_RANGE)
    legend_gap = random.randint(*GEN_LEGEND_GAP_RANGE)
    legend_h = 4 * legend_size + 3 * legend_gap

    top_margin = random.randint(*GEN_MARGIN_RANGE)
    left_margin = random.randint(*GEN_MARGIN_RANGE)
    right_margin = random.randint(*GEN_MARGIN_RANGE)
    bottom_margin = random.randint(*GEN_MARGIN_RANGE)
    gap = random.randint(*GEN_GRID_LEGEND_GAP_RANGE)
    legend_top = top_margin + random.randint(0, 2)

    height = max(top_margin + grid_h, legend_top + legend_h) + bottom_margin
    width = left_margin + grid_w + gap + legend_size + right_margin
    grid = np.full((height, width), int(bg), dtype=int)

    grid_top = top_margin
    grid_left = left_margin
    lattice = grid[grid_top : grid_top + grid_h, grid_left : grid_left + grid_w]
    for r in range(rows + 1):
        lattice[r * (size + 1), :] = GREY
    for c in range(cols + 1):
        lattice[:, c * (size + 1)] = GREY
    for r in range(rows):
        for c in range(cols):
            rr = r * (size + 1) + 1
            cc = c * (size + 1) + 1
            lattice[rr : rr + size, cc : cc + size] = cells[r][c]

    legend_left = grid_left + grid_w + gap
    for i, color in enumerate(legend_colors):
        rr = legend_top + i * (legend_size + legend_gap)
        grid[rr : rr + legend_size, legend_left : legend_left + legend_size] = int(color)

    if _bg_color(grid) != bg:
        return None

    expected = _output_from_cells(cells, bg, legend_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
