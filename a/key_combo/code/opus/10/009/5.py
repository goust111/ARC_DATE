from __future__ import annotations

import itertools

import numpy as np


BACKGROUND = 0
GRID_COLOR = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid != BACKGROUND)
    assert box is not None, "expected non-empty crop"
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_line_positions(grid: np.ndarray, *, axis: int) -> list[int]:
    counts = np.count_nonzero(grid == GRID_COLOR, axis=1 - axis)
    threshold = int(counts.max()) - 4
    positions = [int(i) for i in np.flatnonzero(counts >= threshold)]
    assert len(positions) >= 3, "expected multiple regular grid lines"
    diffs = np.diff(np.asarray(positions, dtype=int))
    assert diffs.size > 0 and np.all(diffs == diffs[0]) and diffs[0] > 1, "grid lines must be evenly spaced"
    return positions


def _occupied_cell_crops(grid: np.ndarray, row_lines: list[int], col_lines: list[int]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for r_idx in range(len(row_lines) - 1):
        for c_idx in range(len(col_lines) - 1):
            cell = grid[row_lines[r_idx] + 1 : row_lines[r_idx + 1], col_lines[c_idx] + 1 : col_lines[c_idx + 1]]
            colors = sorted(set(map(int, np.unique(cell))) - {BACKGROUND})
            assert GRID_COLOR not in colors, "grid color should not appear inside a cell interior"
            if not colors:
                continue
            assert len(colors) == 1, "each non-empty cell must contain exactly one monochrome sprite"
            assert _is_connected(cell != BACKGROUND), "each sprite must be 4-connected"
            crops.append(_crop_nonzero(cell))
    assert crops, "expected at least one non-empty cell"
    return crops


def _concat_crops(crops: list[np.ndarray]) -> np.ndarray:
    assert crops, "expected at least one crop"
    height = int(crops[0].shape[0])
    assert all(int(crop.shape[0]) == height for crop in crops), "selected crops must share a height"

    width = sum(int(crop.shape[1]) for crop in crops) + len(crops) - 1
    out = np.zeros((height, width), dtype=int)
    c0 = 0
    for i, crop in enumerate(crops):
        w = int(crop.shape[1])
        out[:, c0 : c0 + w] = crop
        c0 += w
        if i + 1 < len(crops):
            c0 += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count missing lattice intersections and use that count as the spacing for selecting cropped cell sprites.

    Input structure:
    - Black cells are the background.
    - Gray single-pixel horizontal and vertical lines form a regular lattice whose outer border is also drawn.
    - Some lattice intersections are missing gray pixels and appear as black background instead.
    - Some cell interiors contain one small monochrome 4-connected sprite; other cells are empty.

    Rule:
    - Detect the regular gray lattice and count how many lattice intersections are missing gray pixels; call that count K.
    - Scan the cell interiors in reading order and collect only the non-empty cells.
    - Select non-empty cells with ranks 1, 1+K, 1+2K, ... .
    - Crop each selected sprite to its tight bounding box and concatenate the crops from left to right with one black separator column.

    Key insight:
    - The selection interval is not fixed in advance; it is encoded by the lattice anomaly itself, namely the number of missing gray intersections.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    row_lines = _find_line_positions(grid, axis=0)
    col_lines = _find_line_positions(grid, axis=1)
    assert row_lines[0] == 0 and row_lines[-1] == grid.shape[0] - 1, "expected outer horizontal border lines"
    assert col_lines[0] == 0 and col_lines[-1] == grid.shape[1] - 1, "expected outer vertical border lines"

    for r in row_lines:
        assert np.all(np.isin(grid[r, :], (BACKGROUND, GRID_COLOR))), "line rows may only contain gray or missing-black pixels"
    for c in col_lines:
        assert np.all(np.isin(grid[:, c], (BACKGROUND, GRID_COLOR))), "line columns may only contain gray or missing-black pixels"

    missing = [(r, c) for r in row_lines for c in col_lines if int(grid[r, c]) == BACKGROUND]
    assert missing, "expected missing intersections"
    assert all(row_lines[0] < r < row_lines[-1] and col_lines[0] < c < col_lines[-1] for r, c in missing), (
        "missing intersections must be interior lattice crossings"
    )
    k = len(missing)
    assert 2 <= k <= 4, "missing-intersection count must define K in [2, 4]"

    occupied = _occupied_cell_crops(grid, row_lines, col_lines)
    selected = occupied[::k]
    assert 2 <= len(selected) <= 5, "expected 2 to 5 selected sprites"
    return _concat_crops(selected)


def _enumerate_masks() -> dict[tuple[int, int], list[np.ndarray]]:
    pools: dict[tuple[int, int], list[np.ndarray]] = {}
    for h, w in ((2, 1), (2, 2), (2, 3), (3, 1), (3, 2)):
        cells = [(r, c) for r in range(h) for c in range(w)]
        masks: list[np.ndarray] = []
        for area in range(2, min(4, h * w) + 1):
            for subset in itertools.combinations(cells, area):
                mask = np.zeros((h, w), dtype=bool)
                for r, c in subset:
                    mask[r, c] = True
                rows = {r for r, _ in subset}
                cols = {c for _, c in subset}
                if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
                    continue
                if _is_connected(mask):
                    masks.append(mask)
        pools[(h, w)] = masks
    return pools


_MASK_POOLS = _enumerate_masks()


def _sample_occupied_indices(
    total_cells: int,
    nonempty_count: int,
    k: int,
    selected_count: int,
    rng: np.random.Generator,
) -> list[int] | None:
    for _ in range(200):
        indices = sorted(int(v) for v in rng.choice(total_cells, size=nonempty_count, replace=False))
        if len(indices[::k]) != selected_count:
            continue
        if not np.any(np.diff(indices) > 1):
            continue
        absolute_selected = indices[::k]
        if len(absolute_selected) > 1 and all(
            absolute_selected[i] - absolute_selected[0] == i * k for i in range(len(absolute_selected))
        ):
            continue
        if total_cells > nonempty_count and rng.random() < 0.7 and indices[0] == 0:
            continue
        return indices
    return None


def _sample_sprite_crop(
    sprite_h: int,
    cell_w: int,
    color: int,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_keys = [key for key, pool in _MASK_POOLS.items() if key[0] == sprite_h and key[1] <= cell_w and pool]
    assert valid_keys, "expected at least one valid mask pool"
    h, w = valid_keys[int(rng.integers(0, len(valid_keys)))]
    mask = _MASK_POOLS[(h, w)][int(rng.integers(0, len(_MASK_POOLS[(h, w)])))]
    crop = np.zeros((h, w), dtype=int)
    crop[mask] = int(color)
    return crop


def _render_input(
    cell_rows: int,
    cell_cols: int,
    cell_h: int,
    cell_w: int,
    occupied_indices: list[int],
    crops: list[np.ndarray],
    holes: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int], list[int]]:
    row_lines = [i * (cell_h + 1) for i in range(cell_rows + 1)]
    col_lines = [i * (cell_w + 1) for i in range(cell_cols + 1)]
    grid = np.zeros((row_lines[-1] + 1, col_lines[-1] + 1), dtype=int)
    grid[row_lines, :] = GRID_COLOR
    grid[:, col_lines] = GRID_COLOR
    for r, c in holes:
        grid[r, c] = BACKGROUND

    for cell_index, crop in zip(occupied_indices, crops):
        cell_r = cell_index // cell_cols
        cell_c = cell_index % cell_cols
        r0 = row_lines[cell_r] + 1
        c0 = col_lines[cell_c] + 1
        h, w = crop.shape
        off_r = int(rng.integers(0, cell_h - h + 1))
        off_c = int(rng.integers(0, cell_w - w + 1))
        grid[r0 + off_r : r0 + off_r + h, c0 + off_c : c0 + off_c + w] = crop
    return grid, row_lines, col_lines


def _expected_output(crops: list[np.ndarray], k: int) -> np.ndarray:
    return _concat_crops(crops[::k])


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    constructed_row_lines: list[int],
    constructed_col_lines: list[int],
    occupied_indices: list[int],
    crops: list[np.ndarray],
    k: int,
) -> bool:
    try:
        detected_row_lines = _find_line_positions(grid, axis=0)
        detected_col_lines = _find_line_positions(grid, axis=1)
        if detected_row_lines != constructed_row_lines or detected_col_lines != constructed_col_lines:
            return False

        missing = [(r, c) for r in constructed_row_lines for c in constructed_col_lines if int(grid[r, c]) == BACKGROUND]
        if len(missing) != k:
            return False
        if not all(
            constructed_row_lines[0] < r < constructed_row_lines[-1]
            and constructed_col_lines[0] < c < constructed_col_lines[-1]
            for r, c in missing
        ):
            return False

        recovered_crops = _occupied_cell_crops(grid, constructed_row_lines, constructed_col_lines)
        if len(recovered_crops) != len(crops):
            return False
        if not all(np.array_equal(a, b) for a, b in zip(recovered_crops, crops)):
            return False

        if not np.array_equal(main(grid), expected):
            return False

        absolute_selected = occupied_indices[::k]
        if len(absolute_selected) < 2:
            return False
        if all(absolute_selected[i] - absolute_selected[0] == i * k for i in range(len(absolute_selected))):
            return False

        for alt_k in (2, 3, 4):
            if alt_k == k or alt_k >= len(crops):
                continue
            alt = _expected_output(crops, alt_k)
            if np.array_equal(alt, expected):
                return False
        return True
    except AssertionError:
        return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.integers(2, 5))
    sprite_h = 2 if rng.random() < 0.55 else 3

    feasible_layouts = []
    for cell_rows in (4, 5):
        for cell_cols in (4, 5):
            total_cells = cell_rows * cell_cols
            max_selected = min(5, 1 + (total_cells - 2) // k)
            if max_selected >= 2:
                feasible_layouts.append((cell_rows, cell_cols))
    if not feasible_layouts:
        return None

    cell_rows, cell_cols = feasible_layouts[int(rng.integers(0, len(feasible_layouts)))]
    cell_h = int(rng.integers(sprite_h, 5))
    cell_w = int(rng.integers(2, 5))
    total_cells = cell_rows * cell_cols

    feasible_selected = [s for s in range(2, min(5, 1 + (total_cells - 2) // k) + 1)]
    if not feasible_selected:
        return None
    weights = np.array([1 if s == 2 else 2 for s in feasible_selected], dtype=float)
    weights /= weights.sum()
    selected_count = int(rng.choice(feasible_selected, p=weights))

    min_nonempty = 1 + (selected_count - 1) * k
    max_nonempty = min(total_cells - 1, min_nonempty + k - 1)
    if min_nonempty > max_nonempty:
        return None
    nonempty_count = int(rng.integers(min_nonempty, max_nonempty + 1))
    occupied_indices = _sample_occupied_indices(total_cells, nonempty_count, k, selected_count, rng)
    if occupied_indices is None:
        return None

    row_lines = [i * (cell_h + 1) for i in range(cell_rows + 1)]
    col_lines = [i * (cell_w + 1) for i in range(cell_cols + 1)]
    interior_intersections = [(r, c) for r in row_lines[1:-1] for c in col_lines[1:-1]]
    if len(interior_intersections) < k:
        return None
    chosen_holes = rng.choice(len(interior_intersections), size=k, replace=False)
    holes = [interior_intersections[int(idx)] for idx in chosen_holes]

    crops: list[np.ndarray] = []
    selected_abs = occupied_indices[::k]
    selected_color_map = dict(zip(selected_abs, rng.choice(SPRITE_COLORS, size=len(selected_abs), replace=False)))
    for cell_index in occupied_indices:
        color = int(selected_color_map.get(cell_index, rng.choice(SPRITE_COLORS)))
        crops.append(_sample_sprite_crop(sprite_h, cell_w, color, rng))

    if len({crop.tobytes() for crop in crops[::k]}) == 1 and len(crops[::k]) > 1:
        return None

    grid, built_row_lines, built_col_lines = _render_input(
        cell_rows=cell_rows,
        cell_cols=cell_cols,
        cell_h=cell_h,
        cell_w=cell_w,
        occupied_indices=occupied_indices,
        crops=crops,
        holes=holes,
        rng=rng,
    )
    expected = _expected_output(crops, k)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(grid, expected, built_row_lines, built_col_lines, occupied_indices, crops, k):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
