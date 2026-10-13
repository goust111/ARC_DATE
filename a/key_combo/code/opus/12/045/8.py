import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
REFERENCE_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 18
MATCH_OBJECT_COUNT_OPTIONS = (3, 4)
MAX_GENERATION_ATTEMPTS = 400
MIN_CLEARANCE_CHEBYSHEV = 3


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_mask(coords) -> np.ndarray:
    rel = _normalize(coords)
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in rel:
        mask[r, c] = True
    return mask


def _rotate_shape(coords, k: int) -> tuple[tuple[int, int], ...]:
    mask = _shape_mask(coords)
    rot = np.rot90(mask, k=int(k))
    pts = np.argwhere(rot)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _rotation_signatures(coords) -> tuple[tuple[tuple[int, int], ...], ...]:
    variants = {_rotate_shape(coords, k) for k in range(4)}
    return tuple(sorted(variants))


def _canonical_rotation_class(coords) -> tuple[tuple[int, int], ...]:
    return min(_rotation_signatures(coords))


def _is_connected(coords) -> bool:
    pts = set(coords)
    start = next(iter(pts))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pts and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(pts)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_to_coords(grid: np.ndarray, coords) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _render_shape(shape, color: int) -> np.ndarray:
    mask = _shape_mask(shape)
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _placeable(abs_coords, occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in abs_coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - (MIN_CLEARANCE_CHEBYSHEV - 1), r + MIN_CLEARANCE_CHEBYSHEV):
            for cc in range(c - (MIN_CLEARANCE_CHEBYSHEV - 1), c + MIN_CLEARANCE_CHEBYSHEV):
                if (rr, cc) in occupied:
                    return False
    return True


def _absolute_coords(shape, anchor_r: int, anchor_c: int) -> tuple[tuple[int, int], ...]:
    return tuple((anchor_r + r, anchor_c + c) for r, c in shape)


def _all_connected_shapes() -> list[dict]:
    cells = [(r, c) for r in range(4) for c in range(4)]
    catalog: list[dict] = []
    seen = set()
    for area in range(3, 6):
        for combo in itertools.combinations(cells, area):
            if not _is_connected(combo):
                continue
            norm = _normalize(combo)
            if norm in seen:
                continue
            seen.add(norm)
            rotations_by_k = tuple(_rotate_shape(norm, k) for k in range(4))
            rotations = tuple(sorted(set(rotations_by_k)))
            catalog.append(
                {
                    "shape": norm,
                    "area": area,
                    "rotations_by_k": rotations_by_k,
                    "rotations": rotations,
                    "rotation_count": len(rotations),
                    "class_key": min(rotations),
                }
            )
    return catalog


SHAPE_CATALOG = _all_connected_shapes()
REFERENCE_SHAPES = [item for item in SHAPE_CATALOG if item["rotation_count"] == 4]
SHAPES_BY_AREA = {
    area: [item for item in SHAPE_CATALOG if item["area"] == area] for area in range(3, 6)
}


def main(grid: np.ndarray) -> np.ndarray:
    """Use the gray corner template to erase all rotation-matching distractors and crop the unique non-matching object.

    Input structure:
    - The grid is a black background with one gray reference shape near the upper-left corner.
    - Elsewhere there are several small non-gray 4-connected colored objects, each isolated by background space.
    - Some colored objects are rotated copies of the gray shape, and exactly one colored object is a different shape.

    Rule:
    - Extract the gray reference shape and enumerate its four quarter-turn variants in normalized bbox coordinates.
    - For each non-gray object, compare its normalized shape against those variants and erase every matching object.
    - Tight-crop and return the only remaining non-matching object, preserving its original color pattern.

    Key insight:
    - The reference template defines which objects to discard rather than which object to keep, so rotation matching must be used as an exclusion test before the final crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    reference_components = _connected_components(g == REFERENCE_COLOR)
    assert len(reference_components) == 1, "expected exactly one gray reference component"
    reference_shape = _normalize(reference_components[0])
    reference_variants = set(_rotation_signatures(reference_shape))
    assert len(reference_variants) > 1, "reference shape must be rotation-sensitive"

    object_mask = (g != BACKGROUND) & (g != REFERENCE_COLOR)
    objects = _connected_components(object_mask)
    assert len(objects) >= 2, "expected multiple colored objects"

    anomalies = []
    for comp in objects:
        shape = _normalize(comp)
        if shape not in reference_variants:
            anomalies.append(comp)

    assert len(anomalies) == 1, "expected exactly one non-matching colored object"
    return _crop_to_coords(g, anomalies[0])


def _choose_reference(rng: random.Random) -> dict:
    return rng.choice(REFERENCE_SHAPES)


def _choose_anomaly(reference: dict, rng: random.Random) -> tuple[dict, bool]:
    same_area = bool(rng.randrange(2))
    pool = SHAPES_BY_AREA[reference["area"]] if same_area else SHAPE_CATALOG
    candidates = [item for item in pool if item["class_key"] != reference["class_key"]]
    if not candidates:
        candidates = [item for item in SHAPE_CATALOG if item["class_key"] != reference["class_key"]]
        same_area = False
    return rng.choice(candidates), same_area


def _find_anchor(
    item: dict,
    grid_shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    rng: random.Random,
    *,
    reserve_corner: bool,
) -> tuple[int, int] | None:
    shape = item["shape"]
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    anchors = [(r, c) for r in range(grid_shape[0] - h + 1) for c in range(grid_shape[1] - w + 1)]
    rng.shuffle(anchors)
    for r, c in anchors:
        if reserve_corner and r < 5 and c < 5:
            continue
        abs_coords = _absolute_coords(shape, r, c)
        if _placeable(abs_coords, occupied, grid_shape):
            return r, c
    return None


def _stamp(grid: np.ndarray, abs_coords, color: int):
    for r, c in abs_coords:
        grid[r, c] = int(color)


def _build_expected(anomaly_shape, color: int) -> np.ndarray:
    return _render_shape(anomaly_shape, color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.zeros((h, w), dtype=int)

    reference = _choose_reference(rng)
    anomaly, anomaly_same_area = _choose_anomaly(reference, rng)
    match_count = rng.choice(MATCH_OBJECT_COUNT_OPTIONS)
    object_total = match_count + 1
    colors = rng.sample(PALETTE, object_total)
    anomaly_color = colors[0]
    match_colors = colors[1:]

    ref_anchor_r = rng.randint(0, 1)
    ref_anchor_c = rng.randint(0, 1)
    ref_abs = _absolute_coords(reference["shape"], ref_anchor_r, ref_anchor_c)
    if not _placeable(ref_abs, set(), grid.shape):
        return None
    _stamp(grid, ref_abs, REFERENCE_COLOR)
    occupied = set(ref_abs)

    odd_rotation = rng.choice((1, 3))
    other_rotations = [rng.randrange(4) for _ in range(match_count - 1)]
    rotations = [odd_rotation] + other_rotations
    rng.shuffle(rotations)

    placed_match_rotations = []
    for color, rot in zip(match_colors, rotations):
        rotated_shape = reference["rotations_by_k"][rot]
        item = {"shape": rotated_shape}
        anchor = _find_anchor(item, grid.shape, occupied, rng, reserve_corner=True)
        if anchor is None:
            return None
        abs_coords = _absolute_coords(rotated_shape, anchor[0], anchor[1])
        _stamp(grid, abs_coords, color)
        occupied.update(abs_coords)
        placed_match_rotations.append(rot)

    anomaly_item = {"shape": anomaly["shape"]}
    anomaly_anchor = _find_anchor(anomaly_item, grid.shape, occupied, rng, reserve_corner=True)
    if anomaly_anchor is None:
        return None
    anomaly_abs = _absolute_coords(anomaly["shape"], anomaly_anchor[0], anomaly_anchor[1])
    _stamp(grid, anomaly_abs, anomaly_color)

    expected = _build_expected(anomaly["shape"], anomaly_color)

    colored_components = _connected_components((grid != BACKGROUND) & (grid != REFERENCE_COLOR))
    if len(colored_components) != object_total:
        return None
    if not any(rot % 2 == 1 for rot in placed_match_rotations):
        return None
    if anomaly_same_area and anomaly["area"] != reference["area"]:
        return None
    if (not anomaly_same_area) and anomaly["area"] == reference["area"]:
        return None
    if _normalize(anomaly["shape"]) in set(reference["rotations"]):
        return None

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
