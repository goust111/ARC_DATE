import numpy as np


BG_COLOR = 0
GUIDE_COLOR = 5
OBJECT_COLOR = 9
CORNER_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
ODD_SIZES = tuple(range(17, 30, 2))
AREA_VALUES = (3, 4, 5, 6, 7, 8)
MAX_SHAPE_DIM = 4
MAX_SHAPE_ATTEMPTS = 200
MAX_PAIR_ATTEMPTS = 120
MAX_INPUT_ATTEMPTS = 400

_RNG = np.random.default_rng()
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_QUADRANT_ORDER = ("TL", "TR", "BL", "BR")


def _get_bg(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel()).argmax())


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    pts = np.array(list(coords), dtype=int)
    r0, c0 = pts.min(axis=0)
    pts[:, 0] -= r0
    pts[:, 1] -= c0
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_hw(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _transform_shape(
    shape: tuple[tuple[int, int], ...], flip_lr: bool = False, rot_k: int = 0
) -> tuple[tuple[int, int], ...]:
    pts = tuple(shape)
    if flip_lr:
        _h, w = _shape_hw(pts)
        pts = tuple((r, w - 1 - c) for r, c in pts)
        pts = _normalize(pts)
    for _ in range(rot_k % 4):
        h, _w = _shape_hw(pts)
        pts = tuple((c, h - 1 - r) for r, c in pts)
        pts = _normalize(pts)
    return _normalize(pts)


def _all_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    variants = []
    for flip_lr in (False, True):
        for rot_k in range(4):
            variant = _transform_shape(shape, flip_lr=flip_lr, rot_k=rot_k)
            if variant not in seen:
                seen.add(variant)
                variants.append(variant)
    return variants


def _canonical_key(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_all_variants(shape))


def _random_connected_shape() -> tuple[tuple[int, int], ...] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target_area = int(_RNG.choice(AREA_VALUES))
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in _DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            cells.add(frontier[int(_RNG.integers(len(frontier)))])

        shape = _normalize(cells)
        h, w = _shape_hw(shape)
        if max(h, w) > MAX_SHAPE_DIM:
            continue
        if len({r for r, _ in shape}) == 1 and len({c for _, c in shape}) == 1:
            continue
        return _all_variants(shape)[int(_RNG.integers(len(_all_variants(shape))))]
    return None


def _quadrant_boxes(h: int, w: int) -> dict[str, tuple[int, int, int, int]]:
    cr, cc = h // 2, w // 2
    return {
        "TL": (1, cr - 3, 1, cc - 3),
        "TR": (1, cr - 3, cc + 3, w - 2),
        "BL": (cr + 3, h - 2, 1, cc - 3),
        "BR": (cr + 3, h - 2, cc + 3, w - 2),
    }


def _placed_coords(
    shape: tuple[tuple[int, int], ...], top: int, left: int
) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _random_placement(
    shape: tuple[tuple[int, int], ...], box: tuple[int, int, int, int]
) -> set[tuple[int, int]] | None:
    h, w = _shape_hw(shape)
    rmin, rmax, cmin, cmax = box
    if rmax - rmin + 1 < h or cmax - cmin + 1 < w:
        return None
    top = int(_RNG.integers(rmin, rmax - h + 2))
    left = int(_RNG.integers(cmin, cmax - w + 2))
    return _placed_coords(shape, top, left)


def _reflect_coords(
    coords: set[tuple[int, int]], center_r: int, center_c: int
) -> set[tuple[int, int]]:
    return {(2 * center_r - r, 2 * center_c - c) for r, c in coords}


def _overlap_placements(
    shape: tuple[tuple[int, int], ...],
    box: tuple[int, int, int, int],
    target: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    h, w = _shape_hw(shape)
    rmin, rmax, cmin, cmax = box
    candidates = {}
    for sr, sc in shape:
        for tr, tc in target:
            top = tr - sr
            left = tc - sc
            if top < rmin or left < cmin or top + h - 1 > rmax or left + w - 1 > cmax:
                continue
            coords = frozenset(_placed_coords(shape, top, left))
            overlap = len(coords & target)
            if 0 < overlap < min(len(coords), len(target)):
                candidates[coords] = overlap
    ordered = sorted(candidates.items(), key=lambda item: (item[1], sorted(item[0])))
    return [set(coords) for coords, _ in ordered]


def _place_pair(
    shape_a: tuple[tuple[int, int], ...],
    shape_b: tuple[tuple[int, int], ...],
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    center_r: int,
    center_c: int,
    want_overlap: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    variants_a = _all_variants(shape_a)
    variants_b = _all_variants(shape_b)
    for _ in range(MAX_PAIR_ATTEMPTS):
        coords_a = _random_placement(
            variants_a[int(_RNG.integers(len(variants_a)))], box_a
        )
        if coords_a is None:
            continue
        reflected_a = _reflect_coords(coords_a, center_r, center_c)

        if want_overlap:
            candidate_sets = []
            order = _RNG.permutation(len(variants_b))
            for idx in order:
                candidate_sets.extend(_overlap_placements(variants_b[int(idx)], box_b, reflected_a))
            if candidate_sets:
                return coords_a, candidate_sets[int(_RNG.integers(len(candidate_sets)))]
            continue

        for _ in range(MAX_PAIR_ATTEMPTS):
            coords_b = _random_placement(
                variants_b[int(_RNG.integers(len(variants_b)))], box_b
            )
            if coords_b is not None and not (coords_b & reflected_a):
                return coords_a, coords_b
    return None


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in _DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _point_quadrant(r: int, c: int, center_r: int, center_c: int) -> str:
    if r < center_r and c < center_c:
        return "TL"
    if r < center_r and c > center_c:
        return "TR"
    if r > center_r and c < center_c:
        return "BL"
    if r > center_r and c > center_c:
        return "BR"
    raise AssertionError("point lies on a center axis")


def _extract_layout(
    grid: np.ndarray,
) -> tuple[int, int, int, dict[str, int], int, list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1 and h >= 7 and w >= 7

    center_r, center_c = h // 2, w // 2
    patch = g[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2]
    assert patch.shape == (3, 3)

    guide = int(patch[1, 1])
    for rr, cc in ((0, 1), (1, 0), (1, 2), (2, 1)):
        assert int(patch[rr, cc]) == guide

    corner_colors = {
        "TL": int(patch[0, 0]),
        "TR": int(patch[0, 2]),
        "BL": int(patch[2, 0]),
        "BR": int(patch[2, 2]),
    }
    assert len(set(corner_colors.values())) == 4
    assert guide not in set(corner_colors.values())

    bg = _get_bg(g)
    outside_center = np.ones_like(g, dtype=bool)
    outside_center[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = False

    other_colors = set(map(int, np.unique(g[outside_center]))) - {bg}
    assert len(other_colors) == 1
    object_color = next(iter(other_colors))
    assert np.all((g[outside_center] == bg) | (g[outside_center] == object_color))

    object_mask = outside_center & (g == object_color)
    comps = _components4(object_mask)
    assert len(comps) == 4

    seen_quadrants = set()
    seen_shape_keys = set()
    marker_cells = {
        (r, c)
        for r in range(center_r - 1, center_r + 2)
        for c in range(center_c - 1, center_c + 2)
    }
    for comp in comps:
        assert 3 <= len(comp) <= 8
        quad = _point_quadrant(comp[0][0], comp[0][1], center_r, center_c)
        for r, c in comp:
            assert _point_quadrant(r, c, center_r, center_c) == quad
            assert 0 < r < h - 1 and 0 < c < w - 1
            for dr, dc in _DIRS4:
                if (r + dr, c + dc) in marker_cells:
                    raise AssertionError("object touches center marker")
        assert quad not in seen_quadrants
        seen_quadrants.add(quad)
        shape_key = _canonical_key(_normalize(comp))
        assert shape_key not in seen_shape_keys
        seen_shape_keys.add(shape_key)
    assert seen_quadrants == set(_QUADRANT_ORDER)
    return bg, center_r, center_c, corner_colors, object_color, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each monochrome object 180 degrees through the central 3x3 marker
    and repaint the reflected copy with the corner color of its target quadrant.
    """
    g = np.asarray(grid, dtype=int)
    _bg, center_r, center_c, corner_colors, object_color, _comps = _extract_layout(g)

    out = g.copy()
    object_pixels = np.argwhere(g == object_color)
    for r, c in object_pixels:
        rr = 2 * center_r - int(r)
        cc = 2 * center_c - int(c)
        quad = _point_quadrant(rr, cc, center_r, center_c)
        out[rr, cc] = corner_colors[quad]
    return out


def _paint_center_block(grid: np.ndarray, corner_colors: dict[str, int]) -> None:
    h, w = grid.shape
    center_r, center_c = h // 2, w // 2
    grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = GUIDE_COLOR
    grid[center_r - 1, center_c - 1] = corner_colors["TL"]
    grid[center_r - 1, center_c + 1] = corner_colors["TR"]
    grid[center_r + 1, center_c - 1] = corner_colors["BL"]
    grid[center_r + 1, center_c + 1] = corner_colors["BR"]


def _sample_unique_shapes() -> list[tuple[tuple[int, int], ...]] | None:
    shapes = []
    keys = set()
    for _ in range(MAX_SHAPE_ATTEMPTS * 4):
        shape = _random_connected_shape()
        if shape is None:
            continue
        key = _canonical_key(shape)
        if key in keys:
            continue
        shapes.append(shape)
        keys.add(key)
        if len(shapes) == 4:
            return shapes
    return None


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.choice(ODD_SIZES))
    w = int(_RNG.choice(ODD_SIZES))
    center_r, center_c = h // 2, w // 2
    boxes = _quadrant_boxes(h, w)
    if any(box[0] > box[1] or box[2] > box[3] for box in boxes.values()):
        return None

    shapes = _sample_unique_shapes()
    if shapes is None:
        return None
    _RNG.shuffle(shapes)

    want_overlap_a = bool(_RNG.integers(0, 2))
    want_overlap_b = bool(_RNG.integers(0, 2))
    pair_a = _place_pair(
        shapes[0],
        shapes[3],
        boxes["TL"],
        boxes["BR"],
        center_r,
        center_c,
        want_overlap_a,
    )
    if pair_a is None:
        return None
    pair_b = _place_pair(
        shapes[1],
        shapes[2],
        boxes["TR"],
        boxes["BL"],
        center_r,
        center_c,
        want_overlap_b,
    )
    if pair_b is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    corner_values = list(_RNG.choice(CORNER_COLOR_POOL, size=4, replace=False))
    corner_colors = {
        "TL": int(corner_values[0]),
        "TR": int(corner_values[1]),
        "BL": int(corner_values[2]),
        "BR": int(corner_values[3]),
    }
    _paint_center_block(grid, corner_colors)

    all_objects = {
        "TL": pair_a[0],
        "BR": pair_a[1],
        "TR": pair_b[0],
        "BL": pair_b[1],
    }
    for coords in all_objects.values():
        for r, c in coords:
            grid[r, c] = OBJECT_COLOR

    try:
        _extract_layout(grid)
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_INPUT_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
