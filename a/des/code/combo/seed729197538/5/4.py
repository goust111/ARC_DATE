from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

WINDOW_COLORS = [1, 2, 3, 4, 6, 7, 8]
SYMMETRY_ORDERS = (1, 2, 4)

WINDOW_COUNT_RANGE = (3, 5)
OUTER_SIZE_RANGE = (5, 7)
GAP_RANGE = (1, 3)
SIDE_MARGIN_RANGE = (0, 3)
TOP_MARGIN_RANGE = (0, 3)
ARROW_GAP_RANGE = (2, 4)
BOTTOM_MARGIN_RANGE = (0, 2)
MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400

_ORBIT_CACHE_90: dict[int, list[tuple[tuple[int, int], ...]]] = {}
_ORBIT_CACHE_180: dict[int, list[tuple[tuple[int, int], ...]]] = {}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rotation_order(mask: np.ndarray) -> int:
    if np.array_equal(mask, np.rot90(mask, 1)):
        return 4
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 2
    return 1


def _is_square_frame(g: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    size = r1 - r0 + 1
    if size != c1 - c0 + 1 or size < 3:
        return False
    if coords.shape[0] != 4 * size - 4:
        return False

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((size, size), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(patch[border] == GREY) and np.all(patch[~border] != GREY))


def _extract_windows(g: np.ndarray) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    for coords in _components(g == GREY):
        if not _is_square_frame(g, coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        interior = patch[1:-1, 1:-1]
        colors = {int(v) for v in np.unique(interior) if int(v) not in {BLACK, GREY}}
        assert len(colors) == 1, "each window must contain exactly one pattern color"
        color = colors.pop()
        mask = interior == color
        windows.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "order": _rotation_order(mask),
            }
        )

    windows.sort(key=lambda item: tuple(item["bbox"][1::-1]))  # type: ignore[index]
    assert len(windows) >= 2, "expected multiple grey-framed windows"
    sizes = {int(item["bbox"][2] - item["bbox"][0] + 1) for item in windows}  # type: ignore[index]
    assert len(sizes) == 1, "all windows must have the same size"
    return windows


def _arrow_direction(g: np.ndarray) -> int:
    comps = _components(g == MAROON)
    assert comps, "expected one maroon arrow"
    coords = max(comps, key=lambda item: item.shape[0])
    rows = coords[:, 0]
    cols = coords[:, 1]

    row_values, row_counts = np.unique(rows, return_counts=True)
    shaft_row = int(row_values[int(np.argmax(row_counts))])
    head_cols = cols[rows != shaft_row]
    if head_cols.size:
        return 1 if float(head_cols.mean()) > float(cols.mean()) else -1

    col_values, col_counts = np.unique(cols, return_counts=True)
    peak_col = int(col_values[int(np.argmax(col_counts))])
    return 1 if peak_col > (int(cols.min()) + int(cols.max())) / 2 else -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort grey-framed windows by the rotational symmetry order of their interior patterns.

    Arrow right  -> stable ascending order 1, 2, 4.
    Arrow left   -> stable descending order 4, 2, 1.
    """
    g = np.asarray(grid, dtype=int)
    windows = _extract_windows(g)
    direction = _arrow_direction(g)

    slots = [item["bbox"] for item in windows]
    sorted_windows = sorted(windows, key=lambda item: int(item["order"]) if direction > 0 else -int(item["order"]))

    out = g.copy()
    for bbox, item in zip(slots, sorted_windows):
        r0, c0, r1, c1 = bbox  # type: ignore[misc]
        out[r0 : r1 + 1, c0 : c1 + 1] = item["patch"]  # type: ignore[index]
    return out


def _rotation_orbits(size: int, quarter_turn: bool) -> list[tuple[tuple[int, int], ...]]:
    cache = _ORBIT_CACHE_90 if quarter_turn else _ORBIT_CACHE_180
    if size in cache:
        return cache[size]

    def step(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        if quarter_turn:
            return c, size - 1 - r
        return size - 1 - r, size - 1 - c

    visited: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in visited:
                continue
            orbit: list[tuple[int, int]] = []
            cell = (r, c)
            while cell not in orbit:
                orbit.append(cell)
                cell = step(cell)
            for item in orbit:
                visited.add(item)
            orbits.append(tuple(orbit))
    cache[size] = orbits
    return orbits


def _sample_mask_from_orbits(size: int, orbits: list[tuple[tuple[int, int], ...]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    prob = random.uniform(0.25, 0.6)
    for orbit in orbits:
        if random.random() < prob:
            for r, c in orbit:
                mask[r, c] = True
    return mask


def _mask_density_ok(mask: np.ndarray) -> bool:
    filled = int(mask.sum())
    area = int(mask.size)
    lower = max(2, area // 6)
    upper = area - lower
    if filled < lower or filled > upper:
        return False
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return int(rows.sum()) >= 2 and int(cols.sum()) >= 2


def _make_order4_mask(size: int) -> np.ndarray:
    orbits = _rotation_orbits(size, quarter_turn=True)
    for _ in range(200):
        mask = _sample_mask_from_orbits(size, orbits)
        if _rotation_order(mask) == 4 and _mask_density_ok(mask):
            return mask
    raise RuntimeError("failed to generate order-4 pattern")


def _make_order2_mask(size: int) -> np.ndarray:
    orbits = _rotation_orbits(size, quarter_turn=False)
    for _ in range(300):
        mask = _sample_mask_from_orbits(size, orbits)
        if _rotation_order(mask) == 2 and _mask_density_ok(mask):
            return mask
    raise RuntimeError("failed to generate order-2 pattern")


def _make_order1_mask(size: int) -> np.ndarray:
    for _ in range(300):
        prob = random.uniform(0.2, 0.55)
        mask = np.asarray(
            [[random.random() < prob for _ in range(size)] for _ in range(size)],
            dtype=bool,
        )
        if _rotation_order(mask) == 1 and _mask_density_ok(mask):
            return mask
    raise RuntimeError("failed to generate order-1 pattern")


def _make_mask(size: int, order: int) -> np.ndarray:
    if order == 4:
        return _make_order4_mask(size)
    if order == 2:
        return _make_order2_mask(size)
    assert order == 1
    return _make_order1_mask(size)


def _render_window(size: int, color: int, mask: np.ndarray) -> np.ndarray:
    patch = np.full((size, size), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1:-1, 1:-1][mask] = int(color)
    return patch


def _sample_orders(count: int, ascending: bool) -> list[int]:
    modes = ["free", "duplicate"]
    if count >= 3:
        modes.append("all_three")
    mode = random.choice(modes)
    for _ in range(400):
        orders = [random.choice(SYMMETRY_ORDERS) for _ in range(count)]
        distinct = set(orders)
        if len(distinct) < 2:
            continue
        if mode == "duplicate" and len(distinct) == count:
            continue
        if mode == "all_three" and distinct != set(SYMMETRY_ORDERS) and not set(SYMMETRY_ORDERS).issubset(distinct):
            continue
        target = sorted(orders) if ascending else sorted(orders, reverse=True)
        if orders != target:
            return orders
    raise RuntimeError("failed to sample sortable symmetry orders")


def _draw_arrow(grid: np.ndarray, center_row: int, start_col: int, end_col: int, direction: int) -> None:
    grid[center_row, start_col : end_col + 1] = MAROON
    if direction > 0:
        head_col = end_col - 1
    else:
        head_col = start_col + 1
    grid[center_row - 1, head_col] = MAROON
    grid[center_row + 1, head_col] = MAROON


def _build_candidate() -> np.ndarray | None:
    count = random.randint(*WINDOW_COUNT_RANGE)
    size = random.randint(*OUTER_SIZE_RANGE)
    gap = random.randint(*GAP_RANGE)
    left = random.randint(*SIDE_MARGIN_RANGE)
    right = random.randint(*SIDE_MARGIN_RANGE)
    top = random.randint(*TOP_MARGIN_RANGE)
    arrow_gap = random.randint(*ARROW_GAP_RANGE)
    bottom = random.randint(*BOTTOM_MARGIN_RANGE)

    width = left + count * size + (count - 1) * gap + right
    height = top + size + arrow_gap + 3 + bottom
    if width > MAX_SIDE or height > MAX_SIDE:
        return None

    direction = random.choice((-1, 1))
    ascending = direction > 0
    orders = _sample_orders(count, ascending=ascending)

    colors = random.sample(WINDOW_COLORS, count)
    inner_size = size - 2
    patches = [_render_window(size, color, _make_mask(inner_size, order)) for color, order in zip(colors, orders)]

    grid = np.full((height, width), BLACK, dtype=int)
    slots: list[tuple[int, int]] = []
    row = top
    col = left
    for patch in patches:
        grid[row : row + size, col : col + size] = patch
        slots.append((row, col))
        col += size + gap

    arrow_row = top + size + arrow_gap + 1
    windows_left = slots[0][1]
    windows_right = slots[-1][1] + size - 1
    start_col = max(0, windows_left - random.randint(0, 2))
    end_col = min(width - 1, windows_right + random.randint(0, 2))
    if end_col - start_col < 4:
        return None
    _draw_arrow(grid, arrow_row, start_col, end_col, direction)

    out = main(grid)
    target_orders = sorted(orders) if ascending else sorted(orders, reverse=True)
    if orders == target_orders:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
