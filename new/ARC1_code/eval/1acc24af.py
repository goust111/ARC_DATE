from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, components_mask, find_components, get_bg


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    shifted = coords - np.array([r0, c0])
    h, w = shifted.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    rotations: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        if not any(np.array_equal(rotated, prev) for prev in rotations):
            rotations.append(rotated)
    return rotations


def _can_cover_pocket(
    shape: np.ndarray,
    pocket: np.ndarray,
    blue_cells: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> bool:
    pocket_cells = {tuple(map(int, rc)) for rc in pocket}
    H, W = grid_shape

    for rotated in _unique_rotations(shape):
        shape_cells = np.argwhere(rotated)
        for sr, sc in shape_cells:
            for pr, pc in pocket:
                dr = int(pr) - int(sr)
                dc = int(pc) - int(sc)
                placed = {
                    (dr + int(r), dc + int(c))
                    for r, c in shape_cells
                }
                if not pocket_cells <= placed:
                    continue
                if any(r < 0 or r >= H or c < 0 or c >= W for r, c in placed):
                    continue
                if placed & blue_cells:
                    continue
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray candidate object red when it can rotate to cover a lower black pocket in the blue structure without overlapping blue.

    Input structure
    - A black background contains one upper blue 4-connected structure and gray 4-connected candidate objects below it.
    - The blue structure's bbox is a three-row band whose internal black components form candidate cavities.

    Rule
    - The pockets are the black 4-connected components inside the blue bbox that touch the bbox's bottom edge.
    - For each gray object, try every rotation and translation; if it covers every pixel of any pocket, stays on the grid, and does not overlap a blue pixel, recolor that original gray object red.

    Key insight
    - Matching is by fitting a whole lower cavity, not by the gray object's current position; extra cells may stick out only through surrounding black space.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    assert set(np.unique(g)).issubset({Color.BLACK, Color.BLUE, Color.GREY})

    blue_objects = find_components(
        g,
        bg=bg,
        colors=[Color.BLUE],
        connectivity=4,
        return_format="dict",
    )
    assert len(blue_objects) == 1
    blue_box = bbox(g == Color.BLUE)
    assert blue_box is not None
    r0, c0, r1, c1 = blue_box

    blue_band = g[r0 : r1 + 1, c0 : c1 + 1] == Color.BLUE
    assert blue_band.shape[0] == 3
    pocket_components = components_mask(~blue_band, connectivity=4)
    pockets = [
        coords + np.array([r0, c0])
        for coords in pocket_components
        if np.any(coords[:, 0] == blue_band.shape[0] - 1)
    ]
    assert pockets

    gray_objects = find_components(
        g,
        bg=bg,
        colors=[Color.GREY],
        connectivity=4,
        return_format="dict",
    )
    assert gray_objects

    out = g.copy()
    blue_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == Color.BLUE)}
    for obj in gray_objects:
        coords = obj["coords"]
        shape = _component_mask(coords)
        if any(_can_cover_pocket(shape, pocket, blue_cells, g.shape) for pocket in pockets):
            out[coords[:, 0], coords[:, 1]] = Color.RED
    return out

# --------------- generator (o5x) ---------------

from arc_utils import neighbors4 as _gen_neighbors4, paste

# --- CFG: canvas and blue-band layout ---
GEN_GRID_SHAPE = (12, 12)
GEN_GRID_H, GEN_GRID_W = GEN_GRID_SHAPE
GEN_BAND_HEIGHT = 3
GEN_BAND_WALL_OFFSET, GEN_BAND_BASE_OFFSET = 1, 2
GEN_BAND_BOTTOM_INDEX = GEN_BAND_HEIGHT - 1
GEN_BAND_TOP_ROW_RANGE = (1, 5)  # stop-exclusive for np.random.randint

# --- CFG: arch and pocket structure ---
GEN_ARCH_COUNTS, GEN_ARCH_WIDTHS = (1, 2), (3, 4)
GEN_ARCH_LEFT_MIN, GEN_ARCH_RIGHT_MARGIN, GEN_ARCH_MIN_GAP_COLS = 1, 1, 1
GEN_SINGLE_BOTTOM_GAP_WIDTH = 3
GEN_WIDE_ARCH_SINGLE_GAP_PROB = 0.5
GEN_POCKET_CELLS_MIN, GEN_POCKET_CELLS_MAX = 2, 4

# --- CFG: lower gray-object layout and shapes ---
GEN_OBJECT_COUNT_MIN, GEN_OBJECT_COUNT_MAX = 2, 4
GEN_OBJECT_CELLS_MIN, GEN_OBJECT_CELLS_MAX = 3, 8
GEN_OBJECT_MIN_ROW, GEN_OBJECT_ROW_MAX = 7, 10
GEN_TOP_TO_OBJECT_MIN_ROW_DELTA = 4
GEN_MATCH_SHAPE_MAX_SPAN = 4
GEN_EXTRA_NONMATCH_PROB = 0.45

# --- CFG: generation retry budget ---
GEN_MAX_ATTEMPTS = 300

GEN_NONMATCH_SEED_SHAPES = (
    np.ones((1, 3), dtype=bool),
    np.ones((1, 4), dtype=bool),
    np.ones((1, 5), dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.ones((2, 3), dtype=bool),
    np.ones((2, 4), dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
)


def _gen_choice(items):
    return items[int(np.random.randint(0, len(items)))]


def _gen_in_bounds(r: int, c: int) -> bool:
    return 0 <= r < GEN_GRID_H and 0 <= c < GEN_GRID_W


def _gen_span(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _gen_mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(tuple(cells), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _gen_normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _gen_unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = _gen_normalize_mask(np.rot90(mask, k))
        if not any(np.array_equal(rotated, old) for old in variants):
            variants.append(rotated)
    return variants


def _gen_arch_options(count: int):
    intervals = []
    for width in GEN_ARCH_WIDTHS:
        stop = GEN_GRID_W - width - GEN_ARCH_RIGHT_MARGIN + 1
        for left in range(GEN_ARCH_LEFT_MIN, stop):
            intervals.append((left, left + width - 1, width))
    if count == 1:
        return [(item,) for item in intervals]

    pairs = []
    for a in intervals:
        for b in intervals:
            if b[0] - a[1] - 1 >= GEN_ARCH_MIN_GAP_COLS:
                pairs.append((a, b))
    return pairs


def _gen_extract_pockets(grid: np.ndarray, top: int) -> list[np.ndarray]:
    band = grid[top : top + GEN_BAND_HEIGHT, :] == Color.BLUE
    return [
        coords + np.array([top, 0])
        for coords in components_mask(~band, connectivity=4)
        if np.any(coords[:, 0] == GEN_BAND_BOTTOM_INDEX)
    ]


def _gen_build_band() -> tuple[np.ndarray, int, list[np.ndarray]] | None:
    grid = np.zeros(GEN_GRID_SHAPE, dtype=int)
    top = int(np.random.randint(*GEN_BAND_TOP_ROW_RANGE))
    arch_count = int(_gen_choice(GEN_ARCH_COUNTS))
    arches = _gen_choice(_gen_arch_options(arch_count))

    grid[top + GEN_BAND_BASE_OFFSET, :] = Color.BLUE
    for left, right, width in arches:
        grid[top, left : right + 1] = Color.BLUE
        grid[top + GEN_BAND_WALL_OFFSET, left] = Color.BLUE
        grid[top + GEN_BAND_WALL_OFFSET, right] = Color.BLUE

        interior = list(range(left + 1, right))
        if width == GEN_SINGLE_BOTTOM_GAP_WIDTH:
            bottom_gaps = [interior[0]]
        elif np.random.random() < GEN_WIDE_ARCH_SINGLE_GAP_PROB:
            bottom_gaps = [_gen_choice(interior)]
        else:
            bottom_gaps = interior
        grid[top + GEN_BAND_BASE_OFFSET, bottom_gaps] = Color.BLACK

    pockets = _gen_extract_pockets(grid, top)
    if len(pockets) != arch_count:
        return None
    if any(
        len(pocket) < GEN_POCKET_CELLS_MIN or len(pocket) > GEN_POCKET_CELLS_MAX
        for pocket in pockets
    ):
        return None
    return grid, top, pockets


def _gen_can_cover_any(
    shape: np.ndarray,
    pockets: list[np.ndarray],
    blue_cells: set[tuple[int, int]],
) -> bool:
    return any(_can_cover_pocket(shape, pocket, blue_cells, GEN_GRID_SHAPE) for pocket in pockets)


def _gen_grow_matching_shape(
    grid: np.ndarray,
    pocket: np.ndarray,
    pockets: list[np.ndarray],
    blue_cells: set[tuple[int, int]],
) -> np.ndarray | None:
    cells = {tuple(map(int, rc)) for rc in pocket}
    target = int(
        np.random.randint(max(GEN_OBJECT_CELLS_MIN, len(cells)), GEN_OBJECT_CELLS_MAX + 1)
    )

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _gen_neighbors4(r, c):
                if not _gen_in_bounds(nr, nc):
                    continue
                if (nr, nc) in cells or (nr, nc) in blue_cells:
                    continue
                h, w = _gen_span(cells | {(nr, nc)})
                if h <= GEN_MATCH_SHAPE_MAX_SPAN and w <= GEN_MATCH_SHAPE_MAX_SPAN:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(_gen_choice(sorted(frontier)))

    shape = _gen_mask_from_cells(cells)
    if not _gen_can_cover_any(shape, pockets, blue_cells):
        return None
    return shape


def _gen_base_nonmatch_shapes() -> list[np.ndarray]:
    variants = []
    for mask in GEN_NONMATCH_SEED_SHAPES:
        for flipped in (mask, np.fliplr(mask)):
            for rotated in _gen_unique_rotations(flipped):
                if not any(np.array_equal(rotated, old) for old in variants):
                    variants.append(rotated)
    return variants


def _gen_nonmatching_shape(
    pockets: list[np.ndarray],
    blue_cells: set[tuple[int, int]],
    max_height: int,
) -> np.ndarray | None:
    valid = [
        shape
        for shape in _gen_base_nonmatch_shapes()
        if GEN_OBJECT_CELLS_MIN <= int(shape.sum()) <= GEN_OBJECT_CELLS_MAX
        and shape.shape[0] <= max_height
        and not _gen_can_cover_any(shape, pockets, blue_cells)
    ]
    if not valid:
        return None

    target = int(np.random.randint(GEN_OBJECT_CELLS_MIN, GEN_OBJECT_CELLS_MAX + 1))
    same_size = [shape for shape in valid if int(shape.sum()) == target]
    return _gen_choice(same_size or valid)


def _gen_orient_for_rows(mask: np.ndarray, max_height: int) -> np.ndarray | None:
    fits = [variant for variant in _gen_unique_rotations(mask) if variant.shape[0] <= max_height]
    if not fits:
        return None
    return _gen_choice(fits)


def _gen_valid_locations(grid: np.ndarray, mask: np.ndarray, min_row: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    max_row = GEN_OBJECT_ROW_MAX - h + 1
    if max_row < min_row:
        return []

    local = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    locations = []
    for r0 in range(min_row, max_row + 1):
        for c0 in range(0, GEN_GRID_W - w + 1):
            cells = {(r0 + r, c0 + c) for r, c in local}
            if any(grid[r, c] != Color.BLACK for r, c in cells):
                continue

            ok = True
            for r, c in cells:
                for nr, nc in _gen_neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    if _gen_in_bounds(nr, nc) and grid[nr, nc] != Color.BLACK:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                locations.append((r0, c0))
    return locations


def _gen_paste_gray(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = Color.GREY
    paste(grid, patch, r0, c0, transparent=Color.BLACK)


def _gen_place_objects(
    grid: np.ndarray,
    shapes: list[np.ndarray],
    min_row: int,
    max_height: int,
) -> bool:
    order = list(np.random.permutation(len(shapes)))
    for idx in order:
        oriented = _gen_orient_for_rows(shapes[idx], max_height)
        if oriented is None:
            return False
        locs = _gen_valid_locations(grid, oriented, min_row)
        if not locs:
            return False
        r0, c0 = _gen_choice(locs)
        _gen_paste_gray(grid, oriented, r0, c0)
    return True


def _gen_build_object_shapes(
    grid: np.ndarray,
    pockets: list[np.ndarray],
    blue_cells: set[tuple[int, int]],
    max_height: int,
) -> tuple[list[np.ndarray], int] | None:
    object_count = int(
        np.random.randint(max(GEN_OBJECT_COUNT_MIN, len(pockets) + 1), GEN_OBJECT_COUNT_MAX + 1)
    )
    shapes = []
    for pocket in pockets:
        shape = _gen_grow_matching_shape(grid, pocket, pockets, blue_cells)
        if shape is None:
            return None
        shapes.append(shape)

    unmatched_count = 0
    while len(shapes) < object_count:
        need_unmatched = unmatched_count == 0 and len(shapes) == object_count - 1
        if need_unmatched or np.random.random() < GEN_EXTRA_NONMATCH_PROB:
            shape = _gen_nonmatching_shape(pockets, blue_cells, max_height)
            unmatched_count += 1
        else:
            shape = _gen_grow_matching_shape(grid, _gen_choice(pockets), pockets, blue_cells)
        if shape is None:
            return None
        shapes.append(shape)

    if unmatched_count == 0:
        shape = _gen_nonmatching_shape(pockets, blue_cells, max_height)
        if shape is None:
            return None
        shapes[-1] = shape

    return shapes, object_count


def _gen_validate_gray_objects(grid: np.ndarray, object_count: int) -> bool:
    gray_objects = find_components(
        grid,
        bg=Color.BLACK,
        colors=[Color.GREY],
        connectivity=4,
        return_format="dict",
    )
    if len(gray_objects) != object_count:
        return False
    for obj in gray_objects:
        area = int(obj["area"])
        r0, _, r1, _ = obj["bbox"]
        if (
            area < GEN_OBJECT_CELLS_MIN
            or area > GEN_OBJECT_CELLS_MAX
            or r0 < GEN_OBJECT_MIN_ROW
            or r1 > GEN_OBJECT_ROW_MAX
        ):
            return False
    return True


def _gen_passes_oracle(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        if not np.any(out == Color.RED) or not np.any(out == Color.GREY):
            return False
        if not np.array_equal(main(grid), out):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    band = _gen_build_band()
    if band is None:
        return None
    grid, top, pockets = band

    blue_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == Color.BLUE)}
    min_row = max(GEN_OBJECT_MIN_ROW, top + GEN_TOP_TO_OBJECT_MIN_ROW_DELTA)
    max_height = GEN_OBJECT_ROW_MAX + 1 - min_row
    if max_height < 1:
        return None

    shape_plan = _gen_build_object_shapes(grid, pockets, blue_cells, max_height)
    if shape_plan is None:
        return None
    shapes, object_count = shape_plan

    if not _gen_place_objects(grid, shapes, min_row, max_height):
        return None
    if not _gen_validate_gray_objects(grid, object_count):
        return None
    if not _gen_passes_oracle(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid 1acc24af input")
