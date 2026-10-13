import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORDER_TO_DIRS = {
    4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    2: [(-1, -1), (1, 1), (-1, 1), (1, -1)],
    1: [(0, 1)],
}

RNG = np.random.default_rng()
MAX_RETRIES = 400
GRID_H_RANGE = (24, 35)
GRID_W_RANGE = (24, 35)
OBJECT_COUNT_RANGE = (3, 5)
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

ORDER4_TEMPLATES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
]

ORDER2_TEMPLATES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    ),
]

ORDER1_TEMPLATES = [
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
    np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=int,
    ),
]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _symmetry_order(mask: np.ndarray) -> int:
    if mask.shape[0] == mask.shape[1] and np.array_equal(mask, np.rot90(mask)):
        return 4
    if np.array_equal(mask, np.rot90(mask, 2)):
        return 2
    return 1


def _draw_ray(out: np.ndarray, start_r: int, start_c: int, dr: int, dc: int, color: int) -> None:
    h, w = out.shape
    r, c = start_r, start_c
    while 0 <= r < h and 0 <= c < w:
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw rays around each monochrome object according to its rotational symmetry order.

    - 90-degree rotational symmetry -> 4 axis-aligned rays.
    - 180-degree rotational symmetry -> 2 diagonal lines (implemented as 4 half-rays).
    - no non-trivial rotational symmetry -> 1 rightward ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    comps = _components(g, bg)
    assert comps

    out = g.copy()
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        order = _symmetry_order(comp["mask"])
        for dr, dc in ORDER_TO_DIRS[order]:
            _draw_ray(out, center_r, center_c, dr, dc, int(comp["color"]))
    return out


def _transform_template(mask: np.ndarray) -> np.ndarray:
    result = np.array(mask, copy=True)
    k = int(RNG.integers(0, 4))
    result = np.rot90(result, k)
    if bool(RNG.integers(0, 2)):
        result = np.fliplr(result)
    return result


def _sample_shape(order: int) -> np.ndarray:
    if order == 4:
        base = ORDER4_TEMPLATES[int(RNG.integers(len(ORDER4_TEMPLATES)))]
    elif order == 2:
        base = ORDER2_TEMPLATES[int(RNG.integers(len(ORDER2_TEMPLATES)))]
    else:
        base = ORDER1_TEMPLATES[int(RNG.integers(len(ORDER1_TEMPLATES)))]
    mask = _transform_template(base)
    assert _symmetry_order(mask.astype(bool)) == order
    return mask


def _ray_cells(center_r: int, center_c: int, order: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for dr, dc in ORDER_TO_DIRS[order]:
        r, c = center_r, center_c
        while 0 <= r < h and 0 <= c < w:
            cells.add((r, c))
            r += dr
            c += dc
    return cells


def _dilated(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _placeable(
    object_cells: set[tuple[int, int]],
    output_cells: set[tuple[int, int]],
    input_guard: set[tuple[int, int]],
    output_used: set[tuple[int, int]],
) -> bool:
    return object_cells.isdisjoint(input_guard) and output_cells.isdisjoint(output_used)


def _try_generate() -> np.ndarray | None:
    h = int(RNG.integers(*GRID_H_RANGE))
    w = int(RNG.integers(*GRID_W_RANGE))
    grid = np.zeros((h, w), dtype=int)

    family = int(RNG.integers(0, 3))
    if family == 0:
        orders = [4] + [1] * int(RNG.integers(1, 3))
    elif family == 1:
        orders = [2] + [1] * int(RNG.integers(1, 3))
    else:
        orders = [1] * int(RNG.integers(2, 5))
    count = len(orders)
    colors = RNG.choice(PALETTE, size=count, replace=False)

    input_guard: set[tuple[int, int]] = set()
    output_used: set[tuple[int, int]] = set()

    for idx, order in enumerate(orders):
        mask = _sample_shape(order)
        mh, mw = mask.shape
        assert mh % 2 == 1 and mw % 2 == 1
        half_h = mh // 2
        half_w = mw // 2
        cells_local = {(int(r) - half_h, int(c) - half_w) for r, c in np.argwhere(mask > 0)}
        placed = False

        if family == 0 and order == 4:
            row_lo, row_hi = half_h + 1, max(half_h + 2, h - half_h - 1)
            col_lo, col_hi = half_w + 1, max(half_w + 2, w // 3)
        elif family == 1 and order == 2:
            row_lo, row_hi = half_h + 1, max(half_h + 2, h - half_h - 1)
            col_lo, col_hi = half_w + 1, max(half_w + 2, w // 3)
        elif family in (0, 1) and order == 1:
            row_lo, row_hi = half_h + 1, max(half_h + 2, h - half_h - 1)
            col_lo, col_hi = max(half_w + 1, (2 * w) // 3), w - half_w - 1
        else:
            row_lo, row_hi = half_h + 1, max(half_h + 2, h - half_h - 1)
            col_lo, col_hi = half_w + 1, w - half_w - 1

        if row_lo >= row_hi or col_lo >= col_hi:
            return None

        for _ in range(500):
            center_r = int(RNG.integers(row_lo, row_hi))
            center_c = int(RNG.integers(col_lo, col_hi))
            object_cells = {(center_r + dr, center_c + dc) for dr, dc in cells_local}
            output_cells = _ray_cells(center_r, center_c, order, (h, w))

            if not _placeable(object_cells, output_cells, input_guard, output_used):
                continue

            for r, c in object_cells:
                grid[r, c] = int(colors[idx])
            input_guard |= _dilated(object_cells, h, w)
            output_used |= output_cells
            placed = True
            break

        if not placed:
            return None

    comps = _components(grid, 0)
    if len(comps) != count:
        return None
    found_orders = sorted(_symmetry_order(comp["mask"]) for comp in comps)
    if found_orders != sorted(orders):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
