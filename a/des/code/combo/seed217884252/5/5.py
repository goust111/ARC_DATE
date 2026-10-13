from collections import Counter

import numpy as np


BLACK = 0
SEP_COLOR = 5
REGION_COLORS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
NONBLACK_REGION_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2))
BAND_SIZE_MIN = 4
BAND_SIZE_MAX = 8
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 3
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _bands(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def _separator_mask(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grey = grid == SEP_COLOR
    sep_rows = np.count_nonzero(grey, axis=1) > (grid.shape[1] // 2)
    sep_cols = np.count_nonzero(grey, axis=0) > (grid.shape[0] // 2)
    sep_mask = np.zeros_like(grey, dtype=bool)
    sep_mask[sep_rows, :] = True
    sep_mask[:, sep_cols] = True
    return sep_rows, sep_cols, sep_mask


def _dominant_color(block: np.ndarray) -> int:
    values, counts = np.unique(block, return_counts=True)
    return int(values[np.argmax(counts)])


def _region_model(
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], dict[int, tuple[int, int, int, int]], np.ndarray]:
    h, w = grid.shape
    sep_rows, sep_cols, sep_mask = _separator_mask(grid)
    row_bands = _bands([r for r in range(h) if not sep_rows[r]])
    col_bands = _bands([c for c in range(w) if not sep_cols[c]])
    assert row_bands and col_bands, "expected at least one non-separator band in each axis"

    cell_ids = np.full((h, w), -1, dtype=int)
    cell_colors: dict[int, int] = {}
    cell_boxes: dict[int, tuple[int, int, int, int]] = {}
    cell_id = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            fill = _dominant_color(block)
            cell_ids[r0 : r1 + 1, c0 : c1 + 1] = cell_id
            cell_colors[cell_id] = fill
            cell_boxes[cell_id] = (r0, r1, c0, c1)
            cell_id += 1

    return cell_ids, cell_colors, cell_boxes, sep_mask


def _object_mask(
    grid: np.ndarray,
    cell_ids: np.ndarray,
    cell_colors: dict[int, int],
    cell_boxes: dict[int, tuple[int, int, int, int]],
    sep_mask: np.ndarray,
) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    mask[sep_mask] = grid[sep_mask] != SEP_COLOR
    for cell_id, (r0, r1, c0, c1) in cell_boxes.items():
        fill = cell_colors[cell_id]
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        mask[r0 : r1 + 1, c0 : c1 + 1] = block != fill
    mask &= (cell_ids >= 0) | sep_mask
    return mask


def _monochrome_components(grid: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _winner_region(
    coords: np.ndarray,
    background_mask: np.ndarray,
    cell_ids: np.ndarray,
) -> int:
    h, w = background_mask.shape
    pixel_touch: Counter[int] = Counter()
    edge_touch: Counter[int] = Counter()
    for r, c in coords:
        touched: set[int] = set()
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not background_mask[nr, nc]:
                continue
            region_id = int(cell_ids[nr, nc])
            if region_id < 0:
                continue
            touched.add(region_id)
            edge_touch[region_id] += 1
        for region_id in touched:
            pixel_touch[region_id] += 1
    assert pixel_touch, "each object must touch at least one visible region"
    return max(pixel_touch, key=lambda rid: (pixel_touch[rid], edge_touch[rid], -rid))


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cell_ids, cell_colors, cell_boxes, sep_mask = _region_model(g)
    obj_mask = _object_mask(g, cell_ids, cell_colors, cell_boxes, sep_mask)
    background_mask = (~sep_mask) & (~obj_mask) & (cell_ids >= 0)

    out = g.copy()
    for coords in _monochrome_components(g, obj_mask):
        region_id = _winner_region(coords, background_mask, cell_ids)
        out[coords[:, 0], coords[:, 1]] = cell_colors[region_id]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome foreground object to the color of the background region
    that touches the largest number of its pixels in the 4-neighborhood.
    """
    return _transform(grid)


def _shape_frontier(shape: set[tuple[int, int]], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in shape:
        for dr, dc in NEIGHBORS4:
            pos = (r + dr, c + dc)
            if pos in allowed and pos not in shape:
                frontier.add(pos)
    return list(frontier)


def _grow_shape(
    allowed: set[tuple[int, int]],
    seeds: set[tuple[int, int]],
    target_size: int,
    rng: np.random.Generator,
    weight_fn=None,
) -> set[tuple[int, int]]:
    shape = set(seeds)
    if not shape or not shape.issubset(allowed):
        return set()
    while len(shape) < target_size:
        frontier = _shape_frontier(shape, allowed)
        if not frontier:
            break
        if weight_fn is None:
            idx = int(rng.integers(len(frontier)))
        else:
            weights = np.array([max(0.01, float(weight_fn(pos))) for pos in frontier], dtype=float)
            weights /= weights.sum()
            idx = int(rng.choice(len(frontier), p=weights))
        shape.add(frontier[idx])
    return shape


def _cell_positions(cell: dict) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(int(cell["r0"]), int(cell["r1"]) + 1)
        for c in range(int(cell["c0"]), int(cell["c1"]) + 1)
    }


def _single_shape(
    cell: dict,
    cell_ids: np.ndarray,
    sep_mask: np.ndarray,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    allowed = _cell_positions(cell)
    area = len(allowed)
    max_size = min(8, (area - 1) // 2)
    if max_size < 3:
        return None
    positions = list(allowed)
    for _ in range(30):
        seed = positions[int(rng.integers(len(positions)))]
        target = int(rng.integers(3, max_size + 1))
        shape = _grow_shape(allowed, {seed}, target, rng)
        if len(shape) < 3:
            continue
        winner = _candidate_winner(shape, cell_ids, sep_mask)
        if winner == int(cell["id"]):
            return shape
    return None


def _candidate_winner(
    shape: set[tuple[int, int]],
    cell_ids: np.ndarray,
    sep_mask: np.ndarray,
) -> int | None:
    h, w = cell_ids.shape
    pixel_touch: Counter[int] = Counter()
    for r, c in shape:
        touched: set[int] = set()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if sep_mask[nr, nc] or (nr, nc) in shape:
                continue
            region_id = int(cell_ids[nr, nc])
            if region_id >= 0:
                touched.add(region_id)
        for region_id in touched:
            pixel_touch[region_id] += 1
    if not pixel_touch:
        return None
    best = max(pixel_touch.values())
    winners = [rid for rid, count in pixel_touch.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _cross_shape(
    pair: dict,
    cells: dict[int, dict],
    cell_ids: np.ndarray,
    sep_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], int] | tuple[None, None]:
    cell_a = cells[int(pair["a"])]
    cell_b = cells[int(pair["b"])]
    allowed = _cell_positions(cell_a) | _cell_positions(cell_b)

    if pair["orient"] == "h":
        sep_col = int(pair["sep"])
        bridge_row = int(rng.integers(int(cell_a["r0"]), int(cell_a["r1"]) + 1))
        bridge = {(bridge_row, sep_col - 1), (bridge_row, sep_col), (bridge_row, sep_col + 1)}
    else:
        sep_row = int(pair["sep"])
        bridge_col = int(rng.integers(int(cell_a["c0"]), int(cell_a["c1"]) + 1))
        bridge = {(sep_row - 1, bridge_col), (sep_row, bridge_col), (sep_row + 1, bridge_col)}

    allowed |= bridge
    for _ in range(40):
        dominant = int(pair["a"] if rng.random() < 0.5 else pair["b"])
        dom_cell = cells[dominant]
        dom_area = (int(dom_cell["r1"]) - int(dom_cell["r0"]) + 1) * (int(dom_cell["c1"]) - int(dom_cell["c0"]) + 1)
        target = int(rng.integers(5, min(10, 4 + dom_area // 3) + 1))

        def weight_fn(pos: tuple[int, int]) -> float:
            if pos in bridge:
                return 0.2
            return 4.0 if int(cell_ids[pos]) == dominant else 1.0

        shape = _grow_shape(allowed, bridge, target, rng, weight_fn=weight_fn)
        if not shape:
            continue
        if not any(int(cell_ids[pos]) == int(pair["a"]) for pos in shape):
            continue
        if not any(int(cell_ids[pos]) == int(pair["b"]) for pos in shape):
            continue

        counts = Counter(int(cell_ids[pos]) for pos in shape if int(cell_ids[pos]) >= 0)
        okay = True
        for cell_id, count in counts.items():
            cell = cells[cell_id]
            area = (int(cell["r1"]) - int(cell["r0"]) + 1) * (int(cell["c1"]) - int(cell["c0"]) + 1)
            if count >= area / 2:
                okay = False
                break
        if not okay:
            continue

        winner = _candidate_winner(shape, cell_ids, sep_mask)
        if winner == dominant:
            return shape, dominant
    return None, None


def _build_layout(rng: np.random.Generator) -> tuple[np.ndarray, dict[int, dict], list[dict], np.ndarray, np.ndarray]:
    n_row_bands, n_col_bands = LAYOUT_OPTIONS[int(rng.integers(len(LAYOUT_OPTIONS)))]
    row_sizes = [int(rng.integers(BAND_SIZE_MIN, BAND_SIZE_MAX + 1)) for _ in range(n_row_bands)]
    col_sizes = [int(rng.integers(BAND_SIZE_MIN, BAND_SIZE_MAX + 1)) for _ in range(n_col_bands)]
    h = sum(row_sizes) + (n_row_bands - 1)
    w = sum(col_sizes) + (n_col_bands - 1)
    grid = np.full((h, w), SEP_COLOR, dtype=int)

    region_count = n_row_bands * n_col_bands
    region_colors = [BLACK] + list(rng.choice(NONBLACK_REGION_COLORS, size=region_count - 1, replace=False))
    rng.shuffle(region_colors)

    cells: dict[int, dict] = {}
    pairs: list[dict] = []
    cell_ids = np.full((h, w), -1, dtype=int)

    row_ranges: list[tuple[int, int]] = []
    start = 0
    for size in row_sizes:
        row_ranges.append((start, start + size - 1))
        start += size + 1

    col_ranges: list[tuple[int, int]] = []
    start = 0
    for size in col_sizes:
        col_ranges.append((start, start + size - 1))
        start += size + 1

    cell_id = 0
    color_idx = 0
    for i, (r0, r1) in enumerate(row_ranges):
        for j, (c0, c1) in enumerate(col_ranges):
            color = int(region_colors[color_idx])
            color_idx += 1
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            cell_ids[r0 : r1 + 1, c0 : c1 + 1] = cell_id
            cells[cell_id] = {"id": cell_id, "row": i, "col": j, "r0": r0, "r1": r1, "c0": c0, "c1": c1, "color": color}
            cell_id += 1

    for i in range(n_row_bands):
        for j in range(n_col_bands):
            a = i * n_col_bands + j
            if j + 1 < n_col_bands:
                sep = int(col_ranges[j][1] + 1)
                pairs.append({"orient": "h", "a": a, "b": a + 1, "sep": sep})
            if i + 1 < n_row_bands:
                sep = int(row_ranges[i][1] + 1)
                pairs.append({"orient": "v", "a": a, "b": a + n_col_bands, "sep": sep})

    _, _, sep_mask = _separator_mask(grid)
    return grid, cells, pairs, cell_ids, sep_mask


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid, cells, pairs, cell_ids, sep_mask = _build_layout(rng)
    region_colors = {int(cell["color"]) for cell in cells.values()}
    object_colors = [color for color in REGION_COLORS if color not in region_colors]
    if len(object_colors) < 2:
        return None

    target_count = int(rng.integers(OBJECT_COUNT_MIN, min(OBJECT_COUNT_MAX, len(cells)) + 1))
    max_cross = min(2, target_count, len(cells) // 2)
    target_cross = int(rng.integers(max_cross + 1))

    used_cells: set[int] = set()
    object_count = 0
    cross_left = target_cross

    for _ in range(80):
        if object_count >= target_count:
            break
        eligible_cross = [pair for pair in pairs if int(pair["a"]) not in used_cells and int(pair["b"]) not in used_cells]
        eligible_single = [cell for cell_id, cell in cells.items() if cell_id not in used_cells]
        if not eligible_cross and not eligible_single:
            break

        remaining = target_count - object_count
        want_cross = False
        if cross_left > 0 and eligible_cross:
            if remaining == cross_left:
                want_cross = True
            elif not eligible_single:
                want_cross = True
            else:
                want_cross = bool(rng.random() < 0.55)
        elif eligible_cross and eligible_single:
            want_cross = bool(rng.random() < 0.45)
        elif eligible_cross:
            want_cross = True

        shape: set[tuple[int, int]] | None = None
        occupied_cells: list[int] = []
        if want_cross:
            pair = eligible_cross[int(rng.integers(len(eligible_cross)))]
            shape, _ = _cross_shape(pair, cells, cell_ids, sep_mask, rng)
            if shape is not None:
                occupied_cells = [int(pair["a"]), int(pair["b"])]
                cross_left = max(0, cross_left - 1)
        else:
            cell = eligible_single[int(rng.integers(len(eligible_single)))]
            shape = _single_shape(cell, cell_ids, sep_mask, rng)
            if shape is not None:
                occupied_cells = [int(cell["id"])]

        if shape is None:
            continue
        if any(pos in used_cells for pos in occupied_cells):
            continue

        color = int(object_colors[int(rng.integers(len(object_colors)))])
        for r, c in shape:
            grid[r, c] = color
        used_cells.update(occupied_cells)
        object_count += 1

    if object_count < target_count or cross_left > 0:
        return None

    detected_ids, detected_colors, _, detected_sep = _region_model(grid)
    if not np.array_equal(sep_mask, detected_sep):
        return None
    for cell_id, cell in cells.items():
        if detected_colors[int(cell_id)] != int(cell["color"]):
            return None
        block_ids = detected_ids[int(cell["r0"]) : int(cell["r1"]) + 1, int(cell["c0"]) : int(cell["c1"]) + 1]
        if not np.all(block_ids == int(cell_id)):
            return None

    obj_mask = _object_mask(grid, detected_ids, detected_colors, {cid: (int(cell["r0"]), int(cell["r1"]), int(cell["c0"]), int(cell["c1"])) for cid, cell in cells.items()}, detected_sep)
    comps = _monochrome_components(grid, obj_mask)
    if len(comps) != object_count:
        return None

    out = _transform(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
