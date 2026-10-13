import numpy as np


BACKGROUND = 0
GRID_MAX_SIDE = 30
INTERSECTION_SIZE_RANGE = (4, 8)
EXTENSION_RANGE = (1, 6)
CANVAS_EXTRA_RANGE = (2, 11)
INSIDE_PATCH_COUNT_RANGE = (0, 3)
OUTSIDE_PATCH_COUNT_RANGE = (1, 4)
MAX_PATCH_TRIES = 24
MAX_GENERATE_ATTEMPTS = 400
SIDES = ("top", "bottom", "left", "right")


def _color_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, f"missing color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_intersection(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    r0 = max(first[0], second[0])
    c0 = max(first[1], second[1])
    r1 = min(first[2], second[2])
    c1 = min(first[3], second[3])
    assert r0 <= r1 and c0 <= c1, "foreground bboxes must overlap"
    return r0, c0, r1, c1


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert len(colors) == 2, "expected exactly two foreground colors"

    bbox_a = _color_bbox(g, colors[0])
    bbox_b = _color_bbox(g, colors[1])
    r0, c0, r1, c1 = _bbox_intersection(bbox_a, bbox_b)

    out = np.zeros_like(g)
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only the foreground pixels that lie inside the overlap of the two color-group bounding boxes.

    Input structure:
    - The background is black.
    - Exactly two non-black colors appear as scattered pixels or small monochrome objects.
    - Each color may have multiple disconnected pieces, and the two overall color bounding boxes overlap.

    Rule:
    - Compute one bounding box for each foreground color using all pixels of that color.
    - Take the rectangular intersection of those two bounding boxes.
    - Preserve every original foreground pixel that lies inside that intersection, regardless of which of the two colors it has.
    - Replace every foreground pixel outside the intersection with black; existing background stays black.

    Key insight:
    - The deciding feature is the overlap of the two color extents, not the individual object shapes or the numeric color ids.
    """
    return _transform(grid)


def _focus(intersection: tuple[int, int, int, int], color_index: int) -> tuple[int, int]:
    r0, c0, r1, c1 = intersection
    if color_index == 0:
        return r0 + 1, c0 + 1
    return r1 - 1, c1 - 1


def _inside_region(
    intersection: tuple[int, int, int, int], color_index: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = intersection
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    if color_index == 0:
        return r0, c0, rm, cm
    return rm + 1, cm + 1, r1, c1


def _outside_regions(
    bbox: tuple[int, int, int, int], intersection: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    br0, bc0, br1, bc1 = bbox
    ir0, ic0, ir1, ic1 = intersection
    regions: list[tuple[int, int, int, int]] = []
    if br0 < ir0:
        regions.append((br0, bc0, ir0 - 1, bc1))
    if ir1 < br1:
        regions.append((ir1 + 1, bc0, br1, bc1))
    if bc0 < ic0:
        regions.append((ir0, bc0, ir1, ic0 - 1))
    if ic1 < bc1:
        regions.append((ir0, ic1 + 1, ir1, bc1))
    return regions


def _edge_candidates(
    bbox: tuple[int, int, int, int],
    intersection: tuple[int, int, int, int],
    side: str,
    inside: bool,
) -> list[tuple[int, int]]:
    br0, bc0, br1, bc1 = bbox
    ir0, ic0, ir1, ic1 = intersection
    if side == "top":
        row = ir0 if inside else br0
        col0, col1 = (ic0, ic1) if inside else (bc0, bc1)
        return [(row, col) for col in range(col0, col1 + 1)]
    if side == "bottom":
        row = ir1 if inside else br1
        col0, col1 = (ic0, ic1) if inside else (bc0, bc1)
        return [(row, col) for col in range(col0, col1 + 1)]
    if side == "left":
        col = ic0 if inside else bc0
        row0, row1 = (ir0, ir1) if inside else (br0, br1)
        return [(row, col) for row in range(row0, row1 + 1)]
    if side == "right":
        col = ic1 if inside else bc1
        row0, row1 = (ir0, ir1) if inside else (br0, br1)
        return [(row, col) for row in range(row0, row1 + 1)]
    raise ValueError(f"unknown side: {side}")


def _pick_free_cell(
    candidates: list[tuple[int, int]],
    occupied: np.ndarray,
    focus: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[int, int]:
    assert candidates, "no candidate cells available"
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda cell: abs(cell[0] - focus[0]) + abs(cell[1] - focus[1]))
    for row, col in shuffled:
        if not occupied[row, col]:
            occupied[row, col] = True
            return row, col
    raise RuntimeError("failed to place a non-overlapping anchor")


def _paint_rect(
    occupied: np.ndarray,
    cells: set[tuple[int, int]],
    region: tuple[int, int, int, int],
    rng: np.random.Generator,
    max_h: int,
    max_w: int,
) -> bool:
    r0, c0, r1, c1 = region
    if r0 > r1 or c0 > c1:
        return False

    max_h = min(max_h, r1 - r0 + 1)
    max_w = min(max_w, c1 - c0 + 1)
    if max_h <= 0 or max_w <= 0:
        return False

    for _ in range(MAX_PATCH_TRIES):
        height = int(rng.integers(1, max_h + 1))
        width = int(rng.integers(1, max_w + 1))
        row = int(rng.integers(r0, r1 - height + 2))
        col = int(rng.integers(c0, c1 - width + 2))
        patch = [(rr, cc) for rr in range(row, row + height) for cc in range(col, col + width)]
        if any(occupied[rr, cc] for rr, cc in patch):
            continue
        for rr, cc in patch:
            occupied[rr, cc] = True
            cells.add((rr, cc))
        return True
    return False


def _sample_layout(
    rng: np.random.Generator,
) -> tuple[
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
]:
    while True:
        ih = int(rng.integers(*INTERSECTION_SIZE_RANGE))
        iw = int(rng.integers(*INTERSECTION_SIZE_RANGE))

        owners = rng.integers(0, 2, size=4)
        if np.all(owners == owners[0]):
            continue

        up = [0, 0]
        down = [0, 0]
        left = [0, 0]
        right = [0, 0]

        side_to_ext = {
            "top": up,
            "bottom": down,
            "left": left,
            "right": right,
        }
        for side, owner in zip(SIDES, owners):
            other = 1 - int(owner)
            side_to_ext[side][other] = int(rng.integers(*EXTENSION_RANGE))

        need_h = max(up) + ih + max(down)
        need_w = max(left) + iw + max(right)
        if need_h + 2 > GRID_MAX_SIDE or need_w + 2 > GRID_MAX_SIDE:
            continue

        max_h = min(GRID_MAX_SIDE, need_h + CANVAS_EXTRA_RANGE[1])
        max_w = min(GRID_MAX_SIDE, need_w + CANVAS_EXTRA_RANGE[1])
        min_h = need_h + CANVAS_EXTRA_RANGE[0]
        min_w = need_w + CANVAS_EXTRA_RANGE[0]
        if min_h > max_h or min_w > max_w:
            continue

        height = int(rng.integers(min_h, max_h + 1))
        width = int(rng.integers(min_w, max_w + 1))

        top_pad = int(rng.integers(1, height - need_h))
        left_pad = int(rng.integers(1, width - need_w))
        bottom_pad = height - need_h - top_pad
        right_pad = width - need_w - left_pad
        if bottom_pad < 1 or right_pad < 1:
            continue

        ir0 = top_pad + max(up)
        ic0 = left_pad + max(left)
        intersection = (ir0, ic0, ir0 + ih - 1, ic0 + iw - 1)
        bbox_a = (ir0 - up[0], ic0 - left[0], intersection[2] + down[0], intersection[3] + right[0])
        bbox_b = (ir0 - up[1], ic0 - left[1], intersection[2] + down[1], intersection[3] + right[1])
        return (height, width), intersection, bbox_a, bbox_b


def _build_color_cells(
    occupied: np.ndarray,
    bbox: tuple[int, int, int, int],
    intersection: tuple[int, int, int, int],
    color_index: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    focus = _focus(intersection, color_index)
    assert not occupied[focus]
    occupied[focus] = True
    cells.add(focus)

    br0, bc0, br1, bc1 = bbox
    ir0, ic0, ir1, ic1 = intersection
    ext_by_side = {
        "top": ir0 - br0,
        "bottom": br1 - ir1,
        "left": ic0 - bc0,
        "right": bc1 - ic1,
    }

    for side in SIDES:
        anchor = _pick_free_cell(
            _edge_candidates(bbox, intersection, side, inside=ext_by_side[side] == 0),
            occupied,
            focus,
            rng,
        )
        cells.add(anchor)

    inside_region = _inside_region(intersection, color_index)
    inside_patch_count = int(rng.integers(*INSIDE_PATCH_COUNT_RANGE))
    for _ in range(inside_patch_count):
        _paint_rect(occupied, cells, inside_region, rng, max_h=2, max_w=2)

    outside_regions = _outside_regions(bbox, intersection)
    outside_patch_count = int(rng.integers(*OUTSIDE_PATCH_COUNT_RANGE))
    for _ in range(outside_patch_count):
        region = outside_regions[int(rng.integers(0, len(outside_regions)))]
        _paint_rect(occupied, cells, region, rng, max_h=2, max_w=3)

    return cells


def _validate_generated(
    grid: np.ndarray,
    colors: tuple[int, int],
    intersection: tuple[int, int, int, int],
    bboxes: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > GRID_MAX_SIDE:
        return False

    present = [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]
    if sorted(present) != sorted(colors):
        return False

    mask_inside = np.zeros_like(grid, dtype=bool)
    ir0, ic0, ir1, ic1 = intersection
    mask_inside[ir0 : ir1 + 1, ic0 : ic1 + 1] = True

    for color, bbox in zip(colors, bboxes):
        if _color_bbox(grid, color) != bbox:
            return False
        mask = grid == color
        if not np.any(mask & mask_inside):
            return False
        if not np.any(mask & ~mask_inside):
            return False

    out = _transform(grid)
    expected = np.zeros_like(grid)
    expected[ir0 : ir1 + 1, ic0 : ic1 + 1] = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
    if not np.array_equal(out, expected):
        return False
    if np.count_nonzero(out) == 0:
        return False
    if np.count_nonzero(out) >= np.count_nonzero(grid):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    (height, width), intersection, bbox_a, bbox_b = _sample_layout(rng)
    colors = tuple(int(color) for color in rng.choice(np.arange(1, 10), size=2, replace=False))

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    cells_a = _build_color_cells(occupied, bbox_a, intersection, 0, rng)
    cells_b = _build_color_cells(occupied, bbox_b, intersection, 1, rng)

    for row, col in cells_a:
        grid[row, col] = colors[0]
    for row, col in cells_b:
        grid[row, col] = colors[1]

    if not _validate_generated(grid, colors, intersection, (bbox_a, bbox_b)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
