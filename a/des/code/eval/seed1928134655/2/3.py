import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = tuple(c for c in range(10) if c not in (BLACK, GREY))
SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)
GEN_ATTEMPTS = 300


def _template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "template must contain grey grid lines"
    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _template_structure(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, list[int], list[int]]:
    r0, c0, r1, c1 = _template_bbox(grid)
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((box == BLACK) | (box == GREY)), "template bbox must contain only black and grey"

    line_rows = [int(i) for i in range(box.shape[0]) if np.all(box[i, :] == GREY)]
    line_cols = [int(j) for j in range(box.shape[1]) if np.all(box[:, j] == GREY)]
    assert len(line_rows) >= 2 and len(line_cols) >= 2, "template must have at least one cell"
    assert line_rows[0] == 0 and line_rows[-1] == box.shape[0] - 1
    assert line_cols[0] == 0 and line_cols[-1] == box.shape[1] - 1
    assert all(b - a >= 2 for a, b in zip(line_rows, line_rows[1:])), "cell interiors must be non-empty"
    assert all(b - a >= 2 for a, b in zip(line_cols, line_cols[1:])), "cell interiors must be non-empty"
    return (r0, c0, r1, c1), box, line_rows, line_cols


def _color_centers(grid: np.ndarray) -> list[tuple[int, float, float]]:
    colors = [int(c) for c in np.unique(grid) if c not in (BLACK, GREY)]
    centers = []
    for color in colors:
        coords = np.argwhere(grid == color)
        centers.append((color, float(np.median(coords[:, 0])), float(np.median(coords[:, 1]))))
    return centers


def _ordered_colors(grid: np.ndarray, row_count: int, col_count: int) -> list[int]:
    centers = _color_centers(grid)
    assert len(centers) == row_count * col_count, "template cell count must match the number of colors"

    centers.sort(key=lambda item: (item[1], item[2]))
    ordered: list[int] = []
    for row in range(row_count):
        row_items = centers[row * col_count : (row + 1) * col_count]
        row_items.sort(key=lambda item: item[2])
        ordered.extend(int(item[0]) for item in row_items)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the grey template by ordering each color with the median center of its scattered fragments.
    """
    g = np.asarray(grid, dtype=int)
    (_, _, _, _), box, line_rows, line_cols = _template_structure(g)

    row_count = len(line_rows) - 1
    col_count = len(line_cols) - 1
    ordered = _ordered_colors(g, row_count, col_count)

    out = box.copy()
    for rr in range(row_count):
        for cc in range(col_count):
            color = ordered[rr * col_count + cc]
            r_start = line_rows[rr] + 1
            r_end = line_rows[rr + 1]
            c_start = line_cols[cc] + 1
            c_end = line_cols[cc + 1]
            out[r_start:r_end, c_start:c_end] = color

    return out


def _make_template(row_count: int, col_count: int, cell_h: int, cell_w: int) -> np.ndarray:
    h = row_count * cell_h + row_count + 1
    w = col_count * cell_w + col_count + 1
    template = np.full((h, w), BLACK, dtype=int)
    for rr in range(row_count + 1):
        template[rr * (cell_h + 1), :] = GREY
    for cc in range(col_count + 1):
        template[:, cc * (cell_w + 1)] = GREY
    return template


def _choose_layout(rng: np.random.Generator) -> tuple[int, int]:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return 1, int(rng.integers(2, 6))
    if mode == 1:
        return int(rng.integers(2, 6)), 1
    return ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))[int(rng.integers(0, 5))]


def _can_place(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    coord_set = set(coords)
    h, w = occupied.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc] and (rr, cc) not in coord_set:
                    return False
    return True


def _place_fragments(
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    center: tuple[int, int],
    rng: np.random.Generator,
) -> bool:
    cr, cc = center
    region_r0, region_r1 = cr - 2, cr + 2
    region_c0, region_c1 = cc - 2, cc + 2
    fragment_count = int(rng.integers(1, 5))
    if rng.random() < 0.45:
        fragment_count = max(fragment_count, 2)

    for _ in range(fragment_count):
        placed = False
        for _ in range(60):
            shape = SHAPES[int(rng.integers(0, len(SHAPES)))]
            max_dr = max(dr for dr, _ in shape)
            max_dc = max(dc for _, dc in shape)
            if region_r1 - max_dr < region_r0 or region_c1 - max_dc < region_c0:
                continue
            base_r = int(rng.integers(region_r0, region_r1 - max_dr + 1))
            base_c = int(rng.integers(region_c0, region_c1 - max_dc + 1))
            coords = [(base_r + dr, base_c + dc) for dr, dc in shape]
            if not _can_place(coords, occupied):
                continue
            for r, c in coords:
                grid[r, c] = color
                occupied[r, c] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _ordered_colors_by_layout(grid: np.ndarray, row_count: int, col_count: int) -> tuple[list[int], list[tuple[int, float, float]]]:
    centers = _color_centers(grid)
    assert len(centers) == row_count * col_count
    centers.sort(key=lambda item: (item[1], item[2]))
    ordered: list[int] = []
    for row in range(row_count):
        row_items = centers[row * col_count : (row + 1) * col_count]
        row_items.sort(key=lambda item: item[2])
        ordered.extend(int(item[0]) for item in row_items)
    return ordered, centers


def _centers_are_unambiguous(centers: list[tuple[int, float, float]], row_count: int, col_count: int) -> bool:
    for row in range(row_count):
        row_items = centers[row * col_count : (row + 1) * col_count]
        cols = [item[2] for item in sorted(row_items, key=lambda item: item[2])]
        if any(cols[i] >= cols[i + 1] for i in range(len(cols) - 1)):
            return False
    if row_count > 1:
        row_bounds = []
        for row in range(row_count):
            row_items = centers[row * col_count : (row + 1) * col_count]
            row_vals = [item[1] for item in row_items]
            row_bounds.append((max(row_vals), min(row_vals)))
        for row in range(row_count - 1):
            if row_bounds[row][0] >= row_bounds[row + 1][1]:
                return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_count, col_count = _choose_layout(rng)
    color_count = row_count * col_count
    colors = [int(c) for c in rng.choice(COLOR_POOL, size=color_count, replace=False)]

    cell_h = int(rng.integers(1, 4))
    cell_w = int(rng.integers(1, 4))
    template = _make_template(row_count, col_count, cell_h, cell_w)

    row_spacing = int(rng.integers(6, 9))
    col_spacing = int(rng.integers(6, 9))
    base_r = int(rng.integers(3, 6))
    base_c = int(rng.integers(3, 6))

    row_centers = [base_r + row * row_spacing + int(rng.integers(0, 2)) for row in range(row_count)]
    col_centers = [base_c + col * col_spacing + int(rng.integers(0, 2)) for col in range(col_count)]

    scene_bottom = row_centers[-1] + 3
    scene_right = col_centers[-1] + 3
    template_top = scene_bottom + int(rng.integers(3, 7))
    template_left = scene_right + int(rng.integers(3, 7))
    grid_h = template_top + template.shape[0] + int(rng.integers(0, 4))
    grid_w = template_left + template.shape[1] + int(rng.integers(0, 4))

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    grid[template_top : template_top + template.shape[0], template_left : template_left + template.shape[1]] = template
    occupied[template_top : template_top + template.shape[0], template_left : template_left + template.shape[1]] = template == GREY

    expected_order: list[int] = []
    for rr in range(row_count):
        for cc in range(col_count):
            color = colors[rr * col_count + cc]
            expected_order.append(color)
            if not _place_fragments(grid, occupied, color, (row_centers[rr], col_centers[cc]), rng):
                return None

    actual_order, centers = _ordered_colors_by_layout(grid, row_count, col_count)
    if actual_order != expected_order:
        return None
    if not _centers_are_unambiguous(centers, row_count, col_count):
        return None
    if not np.array_equal(main(grid), main(grid.copy())):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
