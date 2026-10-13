import numpy as np


TIE_BREAK_ORDER = (5, 2, 8, 9, 6)
TIE_BREAK_RANK = {color: rank for rank, color in enumerate(TIE_BREAK_ORDER)}

BACKGROUND = 0
FRAME_COLOR = 3
OBJECT_COLORS = (1, 2, 4, 5, 6, 8, 9)

GRID_MIN = 16
GRID_MAX = 24
INTERIOR_MIN = 8
INTERIOR_MAX = 10
OBJECT_COLOR_COUNT_RANGE = (3, 4)
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 180
SAME_SAMPLE_REJECT_RETRIES = 80


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = list(map(int, np.unique(grid)))
    if exclude is None:
        return colors
    excluded = set(map(int, exclude))
    return [color for color in colors if color not in excluded]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _rectangular_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = _bbox(mask)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(crop, expected):
        return None
    return bbox


def _find_unique_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in _colors_present(grid, exclude={bg}):
        bbox = _rectangular_frame_bbox(grid == color)
        if bbox is not None:
            candidates.append((int(color), bbox))
    assert len(candidates) == 1, "expected exactly one hollow rectangular frame"
    return candidates[0]


def _band_count(frame_bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_bbox
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    return (min(interior_h, interior_w) + 1) // 2


def _color_first_occurrence(grid: np.ndarray, colors: list[int]) -> dict[int, int]:
    order: dict[int, int] = {}
    needed = set(colors)
    for idx, value in enumerate(map(int, grid.ravel())):
        if value in needed and value not in order:
            order[value] = idx
            if len(order) == len(needed):
                break
    return order


def _sorted_palette(
    grid: np.ndarray,
    color_counts: dict[int, int],
) -> list[int]:
    colors = list(color_counts)
    first_seen = _color_first_occurrence(grid, colors)
    return sorted(
        colors,
        key=lambda color: (
            -int(color_counts[color]),
            TIE_BREAK_RANK.get(int(color), len(TIE_BREAK_RANK) + first_seen[int(color)]),
        ),
    )


def _paint_nested_bands(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    palette: list[int],
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = frame_bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            band = min(r - (r0 + 1), (r1 - 1) - r, c - (c0 + 1), (c1 - 1) - c)
            out[r, c] = palette[min(band, len(palette) - 1)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the frame interior with color bands ordered by outside component counts.

    Input structure:
    - The grid has a dominant black background, one one-pixel-thick hollow rectangular frame, and several small monochromatic objects outside that frame.
    - The frame encloses an empty interior cavity.
    - Each outside color may appear in multiple 8-connected components, and those component counts are all distinct in valid task instances.

    Rule:
    - Identify the unique hollow rectangular frame and ignore its color when counting.
    - For every other foreground color, count its 8-connected components strictly outside the frame bbox.
    - Sort colors by decreasing component count, breaking ties by the fixed visible palette order grey, red, teal, maroon, pink.
    - Fill the frame interior from the outermost ring inward with that sorted color sequence; if the interior has more rings than colors, keep using the final color for every remaining inner ring and center cell.
    - Preserve the frame and every outside object.

    Key insight:
    The scattered outside objects are not noise: their per-color object multiplicities encode the palette order for the concentric decoration inside the central frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, frame_bbox = _find_unique_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg), "expected an empty frame interior"

    outside = np.ones_like(g, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    color_counts: dict[int, int] = {}
    for color in _colors_present(g, exclude={bg, frame_color}):
        comps = _components((g == color) & outside, connectivity=8)
        if comps:
            color_counts[int(color)] = len(comps)
    assert color_counts, "expected at least one outside foreground color"

    palette = _sorted_palette(g, color_counts)
    return _paint_nested_bands(g, frame_bbox, palette)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    bbox = _bbox(mask)
    assert bbox is not None
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_blob(rng: np.random.Generator, target_size: int) -> np.ndarray:
    canvas_h = int(rng.integers(2, 5))
    canvas_w = int(rng.integers(2, 5))
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    start = (int(rng.integers(0, canvas_h)), int(rng.integers(0, canvas_w)))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < target_size:
        candidates: set[tuple[int, int]] = set()
        for r, c in frontier:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < canvas_h and 0 <= nc < canvas_w and not mask[nr, nc]:
                        candidates.add((nr, nc))
        if not candidates:
            break
        pick = list(candidates)[int(rng.integers(0, len(candidates)))]
        mask[pick] = True
        frontier.add(pick)
    if int(mask.sum()) != target_size:
        return _random_blob(rng, target_size)
    return _crop_true(mask)


def _buffer_is_clear(occupied: np.ndarray, coords: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr0 = max(0, int(r) - 1)
        rr1 = min(h, int(r) + 2)
        cc0 = max(0, int(c) - 1)
        cc1 = min(w, int(c) + 2)
        if occupied[rr0:rr1, cc0:cc1].any():
            return False
    return True


def _side_ranges(
    side: str,
    grid_shape: tuple[int, int],
    blob_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, grid_shape)
    bh, bw = map(int, blob_shape)
    r0, c0, r1, c1 = frame_bbox
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(h - 1, r1 + 1)
    ec1 = min(w - 1, c1 + 1)

    if side == "top":
        if er0 - bh < 0:
            return None
        return (0, er0 - bh), (0, w - bw)
    if side == "bottom":
        if er1 + 1 > h - bh:
            return None
        return (er1 + 1, h - bh), (0, w - bw)
    if side == "left":
        if ec0 - bw < 0:
            return None
        return (0, h - bh), (0, ec0 - bw)
    if side == "right":
        if ec1 + 1 > w - bw:
            return None
        return (0, h - bh), (ec1 + 1, w - bw)
    raise ValueError(f"unknown side {side}")


def _place_blob(
    grid: np.ndarray,
    occupied: np.ndarray,
    allowed: np.ndarray,
    blob: np.ndarray,
    color: int,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> bool:
    ranges = _side_ranges(side, grid.shape, blob.shape, frame_bbox)
    if ranges is None:
        return False
    (rmin, rmax), (cmin, cmax) = ranges
    coords_local = np.argwhere(blob)
    for _ in range(PLACEMENT_RETRIES):
        r_base = int(rng.integers(rmin, rmax + 1))
        c_base = int(rng.integers(cmin, cmax + 1))
        coords = coords_local + np.array([r_base, c_base], dtype=int)
        rr = coords[:, 0]
        cc = coords[:, 1]
        if not allowed[rr, cc].all():
            continue
        if not _buffer_is_clear(occupied, coords):
            continue
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True
        return True
    return False


def _sample_counts_and_frame(
    rng: np.random.Generator,
) -> tuple[
    tuple[int, int, int, int],
    tuple[int, int],
    dict[int, int],
]:
    exact_fit = bool(rng.integers(0, 2))
    if exact_fit:
        interior_h = 8
        interior_w = 8 if bool(rng.integers(0, 2)) else int(rng.integers(8, 11))
        n_colors = 4
    else:
        interior_h = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
        interior_w = int(rng.integers(INTERIOR_MIN, INTERIOR_MAX + 1))
        if min(interior_h, interior_w) == 8 and bool(rng.integers(0, 2)):
            if bool(rng.integers(0, 2)):
                interior_h += 1
            else:
                interior_w += 1
        n_colors = int(rng.integers(OBJECT_COLOR_COUNT_RANGE[0], OBJECT_COLOR_COUNT_RANGE[1] + 1))

    frame_h = interior_h + 2
    frame_w = interior_w + 2
    top = int(rng.integers(3, 6))
    bottom = int(rng.integers(3, 6))
    left = int(rng.integers(3, 6))
    right = int(rng.integers(3, 6))
    height = frame_h + top + bottom
    width = frame_w + left + right
    if height > GRID_MAX or width > GRID_MAX:
        scale = max(height - GRID_MAX, width - GRID_MAX)
        if height > GRID_MAX:
            extra = height - GRID_MAX
            bottom = max(3, bottom - extra)
            height = frame_h + top + bottom
        if width > GRID_MAX:
            extra = width - GRID_MAX
            right = max(3, right - extra)
            width = frame_w + left + right
    assert GRID_MIN <= height <= GRID_MAX and GRID_MIN <= width <= GRID_MAX

    r0 = top
    c0 = left
    frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    band_cap = _band_count(frame_bbox)
    assert n_colors <= band_cap
    colors = list(rng.choice(np.array(OBJECT_COLORS, dtype=int), size=n_colors, replace=False))
    counts = list(rng.choice(np.arange(1, 6, dtype=int), size=n_colors, replace=False))
    rng.shuffle(counts)
    color_counts = {int(color): int(count) for color, count in zip(colors, counts)}
    return frame_bbox, (height, width), color_counts


def _side_schedule(total_objects: int, rng: np.random.Generator) -> list[str]:
    all_sides = ["top", "bottom", "left", "right"]
    side_count = 4 if total_objects >= 4 and bool(rng.integers(0, 2)) else 3
    chosen = list(rng.choice(np.array(all_sides, dtype=object), size=side_count, replace=False))
    schedule = chosen[:]
    while len(schedule) < total_objects:
        schedule.append(str(rng.choice(np.array(all_sides, dtype=object))))
    rng.shuffle(schedule)
    return [str(side) for side in schedule]


def _size_choices_by_rank(rank: int, total_ranks: int) -> tuple[int, int]:
    if rank == 0:
        return (2, 3)
    if rank == total_ranks - 1:
        return (4, 5)
    if rank == total_ranks - 2 and total_ranks >= 4:
        return (3, 5)
    return (3, 4)


def _place_objects(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color_counts: dict[int, int],
    rng: np.random.Generator,
) -> bool:
    occupied = grid != BACKGROUND
    allowed = np.ones_like(grid, dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0] - 1, r1 + 1)
    cc1 = min(grid.shape[1] - 1, c1 + 1)
    allowed[rr0 : rr1 + 1, cc0 : cc1 + 1] = False

    ranked = sorted(color_counts.items(), key=lambda item: -item[1])
    rank_of = {color: rank for rank, (color, _count) in enumerate(ranked)}
    object_specs: list[tuple[int, int]] = []
    for color, count in color_counts.items():
        rank = rank_of[color]
        lo, hi = _size_choices_by_rank(rank, len(ranked))
        for _ in range(count):
            object_specs.append((int(color), int(rng.integers(lo, hi + 1))))

    rng.shuffle(object_specs)
    schedule = _side_schedule(len(object_specs), rng)
    for (color, target_size), side in zip(object_specs, schedule):
        placed = False
        for _ in range(SAME_SAMPLE_REJECT_RETRIES):
            blob = _random_blob(rng, target_size)
            if _place_blob(grid, occupied, allowed, blob, color, side, frame_bbox, rng):
                placed = True
                break
        if not placed:
            return False
    return True


def _outside_region(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.ones(shape, dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = False
    return out


def _component_sizes_and_sides(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[list[int], set[str]]:
    object_mask = (grid != bg) & (grid != frame_color)
    comps = _components(object_mask, connectivity=8)
    sizes: list[int] = []
    sides: set[str] = set()
    r0, c0, r1, c1 = frame_bbox
    for comp in comps:
        sizes.append(int(comp.shape[0]))
        rr = comp[:, 0]
        cc = comp[:, 1]
        if rr.max() < r0 - 1:
            sides.add("top")
        elif rr.min() > r1 + 1:
            sides.add("bottom")
        elif cc.max() < c0 - 1:
            sides.add("left")
        elif cc.min() > c1 + 1:
            sides.add("right")
    return sizes, sides


def _semantic_expected(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color_counts: dict[int, int],
) -> np.ndarray:
    palette = _sorted_palette(grid, color_counts)
    return _paint_nested_bands(grid, frame_bbox, palette)


def _validate_generated(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    intended_counts: dict[int, int],
) -> bool:
    bg = _get_bg(grid)
    try:
        frame_color, found_bbox = _find_unique_frame(grid, bg)
    except AssertionError:
        return False
    if frame_color != FRAME_COLOR or found_bbox != frame_bbox:
        return False

    r0, c0, r1, c1 = frame_bbox
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return False
    if len(intended_counts) != len(set(intended_counts.values())):
        return False
    if len(intended_counts) < 3 or len(intended_counts) > 4:
        return False
    if len(intended_counts) > _band_count(frame_bbox):
        return False

    outside = _outside_region(grid.shape, frame_bbox)
    measured_counts: dict[int, int] = {}
    areas: dict[int, int] = {}
    for color in _colors_present(grid, exclude={bg, frame_color}):
        comps = _components((grid == color) & outside, connectivity=8)
        if not comps:
            continue
        measured_counts[int(color)] = len(comps)
        areas[int(color)] = int(((grid == color) & outside).sum())
    if measured_counts != intended_counts:
        return False

    count_order = _sorted_palette(grid, measured_counts)
    area_order = sorted(measured_counts, key=lambda color: -areas[color])
    if count_order == area_order:
        return False

    object_mask = (grid != bg) & (grid != frame_color)
    for comp in _components(object_mask, connectivity=8):
        colors = set(map(int, grid[comp[:, 0], comp[:, 1]]))
        if len(colors) != 1:
            return False

    sizes, sides = _component_sizes_and_sides(grid, bg, frame_color, frame_bbox)
    if not sizes or min(sizes) < 2 or max(sizes) > 5:
        return False
    if len(sides) < 3:
        return False

    expected = _semantic_expected(grid, frame_bbox, measured_counts)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_bbox, grid_shape, color_counts = _sample_counts_and_frame(rng)
    grid = np.full(grid_shape, BACKGROUND, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    if not _place_objects(grid, frame_bbox, color_counts, rng):
        return None
    if not _validate_generated(grid, frame_bbox, color_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
