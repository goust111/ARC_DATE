from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400
GROUP_COUNT_RANGE = (2, 4)
COPIES_RANGE = (2, 4)
SHAPE_BOX_RANGE = (1, 4)
SHAPE_AREA_RANGE = (2, 7)
CELL_H = 5
CELL_W = 5
CELL_GAP = 1
OUTER_MARGIN = 1
MAX_COLS = 5


def _components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def _shape_key(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group monochromatic 4-connected objects by translation-normalized shape.
    Keep only the leftmost object in each group and erase the other copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    comps = _components(g, bg=BG)
    grouped: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)
    for comp in comps:
        grouped[_shape_key(comp["coords"])].append(comp)

    out = g.copy()
    for group in grouped.values():
        keeper = min(group, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
        for comp in group:
            if comp is keeper:
                continue
            for r, c in comp["coords"]:
                out[int(r), int(c)] = BG

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        box_h = random.randint(*SHAPE_BOX_RANGE)
        box_w = random.randint(*SHAPE_BOX_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], box_h * box_w)
        if max_area < SHAPE_AREA_RANGE[0]:
            continue

        area = random.randint(SHAPE_AREA_RANGE[0], max_area)
        seed = (random.randrange(box_h), random.randrange(box_w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if 2 <= int(mask.sum()) <= SHAPE_AREA_RANGE[1]:
            return mask

    raise RuntimeError("failed to sample a connected shape")


def _shape_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
    return tuple(sorted(coords))


def _sample_shapes(group_count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    keys: set[tuple[tuple[int, int], ...]] = set()
    while len(masks) < group_count:
        mask = _random_connected_mask()
        key = _shape_from_mask(mask)
        if key in keys:
            continue
        keys.add(key)
        masks.append(mask)
    return masks


def _grid_shape(num_rows: int, num_cols: int) -> tuple[int, int]:
    height = 2 * OUTER_MARGIN + num_rows * CELL_H + (num_rows - 1) * CELL_GAP
    width = 2 * OUTER_MARGIN + num_cols * CELL_W + (num_cols - 1) * CELL_GAP
    return height, width


def _cell_origin(cell_r: int, cell_c: int) -> tuple[int, int]:
    pitch_h = CELL_H + CELL_GAP
    pitch_w = CELL_W + CELL_GAP
    return OUTER_MARGIN + cell_r * pitch_h, OUTER_MARGIN + cell_c * pitch_w


def _choose_layout(copy_counts: list[int]) -> tuple[int, int]:
    total = sum(copy_counts)
    min_cols = max(max(copy_counts), 3)
    max_cols = min(MAX_COLS, total)
    if min_cols > max_cols:
        raise RuntimeError("infeasible copy counts for the cell layout")
    num_cols = random.randint(min_cols, max_cols)
    num_rows = (total + num_cols - 1) // num_cols
    return num_rows, num_cols


def _assign_cells(num_rows: int, num_cols: int, copy_counts: list[int]) -> list[list[tuple[int, int]]]:
    available = {(r, c) for r in range(num_rows) for c in range(num_cols)}
    assignments: list[list[tuple[int, int]] | None] = [None] * len(copy_counts)
    order = sorted(range(len(copy_counts)), key=lambda idx: (-copy_counts[idx], idx))

    for idx in order:
        copies = copy_counts[idx]
        viable_cols = [c for c in range(num_cols) if any(cell_c == c for _, cell_c in available)]
        if len(viable_cols) < copies:
            raise RuntimeError("not enough distinct columns left for a shape group")

        chosen_cols = random.sample(viable_cols, copies)
        cells: list[tuple[int, int]] = []
        for col in chosen_cols:
            candidates = [cell for cell in available if cell[1] == col]
            if not candidates:
                raise RuntimeError("selected an empty column")
            cell = random.choice(candidates)
            available.remove(cell)
            cells.append(cell)
        random.shuffle(cells)
        assignments[idx] = cells

    return [cells for cells in assignments if cells is not None]


def _group_colors(copies: int) -> list[int]:
    first, second = random.sample(PALETTE, 2)
    colors = [first, second]
    while len(colors) < copies:
        colors.append(random.choice(PALETTE))
    random.shuffle(colors)
    return colors


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    rows, cols = np.where(mask)
    for dr, dc in zip(rows.tolist(), cols.tolist()):
        rr = top + int(dr)
        cc = left + int(dc)
        assert int(grid[rr, cc]) == BG
        grid[rr, cc] = int(color)
        coords.append((rr, cc))
    return coords


def _expected_from_groups(
    shape_groups: list[list[dict[str, object]]],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for group in shape_groups:
        keeper = min(group, key=lambda obj: (int(obj["bbox"][1]), int(obj["bbox"][0])))
        for r, c in keeper["coords"]:
            out[int(r), int(c)] = int(keeper["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(*GROUP_COUNT_RANGE)
    copy_counts = [random.randint(*COPIES_RANGE) for _ in range(group_count)]
    if max(copy_counts) > MAX_COLS:
        return None

    shapes = _sample_shapes(group_count)
    num_rows, num_cols = _choose_layout(copy_counts)
    try:
        cell_groups = _assign_cells(num_rows, num_cols, copy_counts)
    except RuntimeError:
        return None

    height, width = _grid_shape(num_rows, num_cols)
    grid = np.zeros((height, width), dtype=int)
    placed_groups: list[list[dict[str, object]]] = []

    for mask, cells, copies in zip(shapes, cell_groups, copy_counts):
        mask_h, mask_w = map(int, mask.shape)
        assert mask_h <= CELL_H and mask_w <= CELL_W
        colors = _group_colors(copies)
        objects: list[dict[str, object]] = []

        for color, (cell_r, cell_c) in zip(colors, cells):
            base_r, base_c = _cell_origin(cell_r, cell_c)
            top = base_r + random.randint(0, CELL_H - mask_h)
            left = base_c + random.randint(0, CELL_W - mask_w)
            coords = _paint_mask(grid, mask, top, left, color)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (top, left, top + mask_h - 1, left + mask_w - 1),
                }
            )

        left_cols = [int(obj["bbox"][1]) for obj in objects]
        if len(set(left_cols)) != len(left_cols):
            return None
        placed_groups.append(objects)

    expected = _expected_from_groups(placed_groups, grid.shape)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(_components(grid, bg=BG)) != sum(copy_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
