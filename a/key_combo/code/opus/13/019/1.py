import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

TEMPLATE_SHAPES = [
    np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=int,
    ),
]

GREEN_SHAPES = {
    1: [np.array([[1]], dtype=int)],
    2: [
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1], [1], [1], [1]], dtype=int),
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    ],
}


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    components = []
    starts = np.argwhere(mask)
    for r, c in starts:
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        coords = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in dirs:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _color_components(grid: np.ndarray, color: int, connectivity: int = 4) -> list[dict]:
    comps = []
    for coords in _components_from_mask(grid == color, connectivity):
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _stamp(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= dst.shape[0] and left + w <= dst.shape[1]
    view = dst[top : top + h, left : left + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _rot_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(grid.shape), grid.tobytes()


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _normalize_binary(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask != 0)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_binary(mask: np.ndarray, turns: int) -> np.ndarray:
    return _normalize_binary(np.rot90(mask, -turns))


def _extract_template(grid: np.ndarray) -> np.ndarray:
    template = None
    for coords in _components_from_mask(grid != BLACK, 8):
        colors = set(int(grid[r, c]) for r, c in coords)
        if len(colors) >= 2:
            assert template is None, "expected a single multicolor template"
            template = _crop_bbox(grid, _bbox_coords(coords))
    assert template is not None, "template not found"
    assert BLUE in template and RED in template, "template must use blue and red"
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp rotated copies of the top-left multicolor template onto each green marker object according to that marker's pixel count.

    Input structure:
    - The grid has a black background.
    - One separated multicolor template object uses blue and red cells and sits in the upper-left area.
    - Elsewhere there are 3-4 separated green 4-connected objects of area 1-4.

    Rule:
    - Find the blue/red template and each green object.
    - For a green object with area N, rotate the template clockwise by (N - 1) quarter-turns.
    - Align the rotated template's bounding-box midpoint cell with the green object's bounding-box midpoint cell, erase the green object, and stamp the rotated template there.
    - Keep the original source template unchanged.

    Key insight:
    - Each green object encodes two things at once: its location is the stamp anchor, and its pixel count selects which rotation of the template to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template = _extract_template(g)
    green_objects = _color_components(g, GREEN, connectivity=4)
    assert 3 <= len(green_objects) <= 4, "expected 3-4 green objects"

    out = g.copy()
    out[out == GREEN] = BLACK

    for obj in green_objects:
        area = int(obj["area"])
        assert 1 <= area <= 4, "green objects must encode quarter-turns by area 1-4"
        rotated = np.rot90(template, -(area - 1))
        r0, c0, r1, c1 = obj["bbox"]
        anchor_r = (r0 + r1) // 2
        anchor_c = (c0 + c1) // 2
        top = anchor_r - rotated.shape[0] // 2
        left = anchor_c - rotated.shape[1] // 2
        _stamp(out, rotated, top, left)

    return out


def _make_template(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        mask = TEMPLATE_SHAPES[int(rng.integers(len(TEMPLATE_SHAPES)))].copy()
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, -int(rng.integers(4)))
        cell_count = int(mask.sum())
        colors = np.array([BLUE] * cell_count, dtype=int)
        blue_count = int(rng.integers(2, cell_count - 1))
        blue_idx = rng.choice(cell_count, size=blue_count, replace=False)
        colors[:] = RED
        colors[blue_idx] = BLUE
        patch = np.zeros_like(mask, dtype=int)
        coords = np.argwhere(mask != 0)
        for idx, (r, c) in enumerate(coords):
            patch[int(r), int(c)] = int(colors[idx])
        signatures = {_rot_signature(np.rot90(patch, -k)) for k in range(4)}
        if len(signatures) == 4:
            return patch
    raise RuntimeError("failed to build a non-degenerate template")


def _random_green_shape(area: int, rng: np.random.Generator) -> np.ndarray:
    shape = GREEN_SHAPES[area][int(rng.integers(len(GREEN_SHAPES[area])))]
    if rng.random() < 0.5:
        shape = np.fliplr(shape)
    turns = int(rng.integers(4))
    return _rotate_binary(shape, turns)


def _anchor_bounds(size: int, limit: int) -> tuple[int, int]:
    min_anchor = size // 2
    max_anchor = limit - size + size // 2
    return min_anchor, max_anchor


def _place_green_objects(
    grid: np.ndarray,
    template: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    counts: list[int],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    object_boxes = []
    stamp_boxes = []

    for area in rng.permutation(counts):
        area = int(area)
        shape = _random_green_shape(area, rng)
        rotated = np.rot90(template, -(area - 1))
        oh, ow = shape.shape
        th, tw = rotated.shape

        r_lo = max(_anchor_bounds(oh, h)[0], _anchor_bounds(th, h)[0])
        r_hi = min(_anchor_bounds(oh, h)[1], _anchor_bounds(th, h)[1])
        c_lo = max(_anchor_bounds(ow, w)[0], _anchor_bounds(tw, w)[0])
        c_hi = min(_anchor_bounds(ow, w)[1], _anchor_bounds(tw, w)[1])
        if r_lo > r_hi or c_lo > c_hi:
            return False

        placed = False
        for _ in range(300):
            anchor_r = int(rng.integers(r_lo, r_hi + 1))
            anchor_c = int(rng.integers(c_lo, c_hi + 1))
            obj_top = anchor_r - oh // 2
            obj_left = anchor_c - ow // 2
            stamp_top = anchor_r - th // 2
            stamp_left = anchor_c - tw // 2
            obj_box = (obj_top, obj_left, obj_top + oh - 1, obj_left + ow - 1)
            stamp_box = (stamp_top, stamp_left, stamp_top + th - 1, stamp_left + tw - 1)

            if _boxes_overlap(obj_box, template_bbox, pad=2):
                continue
            if _boxes_overlap(stamp_box, template_bbox, pad=1):
                continue
            if any(_boxes_overlap(obj_box, other, pad=1) for other in object_boxes):
                continue
            if any(_boxes_overlap(stamp_box, other, pad=1) for other in stamp_boxes):
                continue

            cells = np.argwhere(shape != 0)
            rr = cells[:, 0] + obj_top
            cc = cells[:, 1] + obj_left
            if np.any(grid[rr, cc] != BLACK):
                continue

            grid[rr, cc] = GREEN
            object_boxes.append(obj_box)
            stamp_boxes.append(stamp_box)
            placed = True
            break

        if not placed:
            return False

    return True


def _validate_input(grid: np.ndarray) -> bool:
    try:
        template = _extract_template(grid)
        template_sigs = {_rot_signature(np.rot90(template, -k)) for k in range(4)}
        if len(template_sigs) != 4:
            return False
        green_objects = _color_components(grid, GREEN, connectivity=4)
        if not (3 <= len(green_objects) <= 4):
            return False
        areas = [int(obj["area"]) for obj in green_objects]
        if any(area < 1 or area > 4 for area in areas):
            return False
        if len(set(areas)) != len(areas):
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        h = int(rng.integers(14, 19))
        w = int(rng.integers(14, 19))
        grid = np.zeros((h, w), dtype=int)

        template = _make_template(rng)
        th, tw = template.shape
        template_top = int(rng.integers(1, 3))
        template_left = int(rng.integers(1, 3))
        if template_top + th > h or template_left + tw > w:
            continue
        _stamp(grid, template, template_top, template_left)
        template_bbox = (template_top, template_left, template_top + th - 1, template_left + tw - 1)

        object_count = 4 if rng.random() < 0.65 else 3
        counts = [1, 2, 3, 4]
        rng.shuffle(counts)
        counts = counts[:object_count]

        if not _place_green_objects(grid, template, template_bbox, counts, rng):
            continue
        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
