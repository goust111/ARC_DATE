import random
from collections import Counter

import numpy as np


# Generated instances use a fixed visible checker pair to avoid hidden
# brightness/id semantics in the task examples.
_DARK_COLOR = 2
_LIGHT_COLOR = 4
_EXTRA_COLOR = 8
_CHECKER_LIGHTNESS = {
    _DARK_COLOR: 0,
    _EXTRA_COLOR: 1,
    _LIGHT_COLOR: 2,
}

_BG_CHOICES = tuple(c for c in range(10) if c not in {_DARK_COLOR, _LIGHT_COLOR, _EXTRA_COLOR})
_OVERLAP_CASES = (
    ("TL", "BR"),
    ("TR", "BL"),
    ("BL", "TR"),
    ("BR", "TL"),
)

_GRID_H_RANGE = (12, 22)
_GRID_W_RANGE = (12, 22)
_OVERLAP_SIZE_RANGE = (1, 4)
_ARM_EXTRA_RANGE = (0, 4)
_THICKNESS_OPTIONS = (1, 2)
_THIRD_OBJECT_PROB = 0.55
_OVERLAP_BRANCH_PROB = 0.5
_PLACEMENT_ATTEMPTS = 80
_GEN_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return objects


def _bbox_intersection(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    r0 = max(int(box_a[0]), int(box_b[0]))
    c0 = max(int(box_a[1]), int(box_b[1]))
    r1 = min(int(box_a[2]), int(box_b[2]))
    c1 = min(int(box_a[3]), int(box_b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


def _light_dark_pair(color_a: int, color_b: int) -> tuple[int, int]:
    rank_a = _CHECKER_LIGHTNESS.get(int(color_a))
    rank_b = _CHECKER_LIGHTNESS.get(int(color_b))
    assert rank_a is not None and rank_b is not None, "unexpected overlap color"
    if rank_a >= rank_b:
        return int(color_a), int(color_b)
    return int(color_b), int(color_a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each pair of foreground objects whose bounding boxes overlap and fill
    the overlap rectangle with a two-color checkerboard, while leaving the
    original objects unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    objects = _find_objects(g, bg)

    out = g.copy()
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            obj_a = objects[i]
            obj_b = objects[j]
            if int(obj_a["color"]) == int(obj_b["color"]):
                continue
            overlap = _bbox_intersection(obj_a["bbox"], obj_b["bbox"])
            if overlap is None:
                continue
            light, dark = _light_dark_pair(int(obj_a["color"]), int(obj_b["color"]))
            r0, c0, r1, c1 = overlap
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if int(g[r, c]) != bg:
                        continue
                    parity = ((r - r0) + (c - c0)) & 1
                    out[r, c] = light if parity == 0 else dark
    return out


def _corner_mask(height: int, width: int, corner: str, thickness: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if corner in ("TL", "TR"):
        mask[:thickness, :] = True
    else:
        mask[height - thickness :, :] = True
    if corner in ("TL", "BL"):
        mask[:, :thickness] = True
    else:
        mask[:, width - thickness :] = True
    return mask


def _solid_rect_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _mask_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rows, cols = np.where(mask)
    return [(int(top + r), int(left + c)) for r, c in zip(rows.tolist(), cols.tolist())]


def _bbox_from_mask(mask: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = map(int, mask.shape)
    return (int(top), int(left), int(top + h - 1), int(left + w - 1))


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _touches_existing(coords: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    if coord_set & occupied:
        return True
    for r, c in coords:
        for nr, nc in _neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) in occupied:
                return True
    return False


def _place_mask(
    rng: random.Random,
    grid_shape: tuple[int, int],
    mask: np.ndarray,
    occupied: set[tuple[int, int]],
    existing_bboxes: list[tuple[int, int, int, int]],
    forbid_bbox_overlap: bool,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]] | None:
    h, w = map(int, grid_shape)
    mh, mw = map(int, mask.shape)
    if mh > h or mw > w:
        return None
    for _ in range(_PLACEMENT_ATTEMPTS):
        top = rng.randint(0, h - mh)
        left = rng.randint(0, w - mw)
        coords = _mask_coords(mask, top, left)
        if _touches_existing(coords, occupied):
            continue
        bbox = _bbox_from_mask(mask, top, left)
        if forbid_bbox_overlap and any(_bbox_intersection(bbox, other) is not None for other in existing_bboxes):
            continue
        return coords, bbox
    return None


def _build_overlap_pair(
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int, int]], tuple[int, int, int, int]]:
    corner_a, corner_b = rng.choice(_OVERLAP_CASES)
    overlap_h = rng.randint(*_OVERLAP_SIZE_RANGE)
    overlap_w = rng.randint(*_OVERLAP_SIZE_RANGE)
    thick_a = rng.choice(_THICKNESS_OPTIONS)
    thick_b = rng.choice(_THICKNESS_OPTIONS)

    ext_a_v = thick_a + rng.randint(*_ARM_EXTRA_RANGE)
    ext_a_h = thick_a + rng.randint(*_ARM_EXTRA_RANGE)
    ext_b_v = thick_b + rng.randint(*_ARM_EXTRA_RANGE)
    ext_b_h = thick_b + rng.randint(*_ARM_EXTRA_RANGE)

    if (corner_a, corner_b) == ("TL", "BR"):
        top, bottom = ext_a_v, ext_b_v
        left, right = ext_a_h, ext_b_h
        pair_h = top + overlap_h + bottom
        pair_w = left + overlap_w + right
        a_top, a_left = 0, 0
        b_top, b_left = top, left
        a_shape = _corner_mask(top + overlap_h, left + overlap_w, "TL", thick_a)
        b_shape = _corner_mask(overlap_h + bottom, overlap_w + right, "BR", thick_b)
    elif (corner_a, corner_b) == ("TR", "BL"):
        top, bottom = ext_a_v, ext_b_v
        left, right = ext_b_h, ext_a_h
        pair_h = top + overlap_h + bottom
        pair_w = left + overlap_w + right
        a_top, a_left = 0, left
        b_top, b_left = top, 0
        a_shape = _corner_mask(top + overlap_h, overlap_w + right, "TR", thick_a)
        b_shape = _corner_mask(overlap_h + bottom, left + overlap_w, "BL", thick_b)
    elif (corner_a, corner_b) == ("BL", "TR"):
        top, bottom = ext_b_v, ext_a_v
        left, right = ext_a_h, ext_b_h
        pair_h = top + overlap_h + bottom
        pair_w = left + overlap_w + right
        a_top, a_left = top, 0
        b_top, b_left = 0, left
        a_shape = _corner_mask(overlap_h + bottom, left + overlap_w, "BL", thick_a)
        b_shape = _corner_mask(top + overlap_h, overlap_w + right, "TR", thick_b)
    else:
        top, bottom = ext_b_v, ext_a_v
        left, right = ext_b_h, ext_a_h
        pair_h = top + overlap_h + bottom
        pair_w = left + overlap_w + right
        a_top, a_left = top, left
        b_top, b_left = 0, 0
        a_shape = _corner_mask(overlap_h + bottom, overlap_w + right, "BR", thick_a)
        b_shape = _corner_mask(top + overlap_h, left + overlap_w, "TL", thick_b)

    canvas = np.zeros((pair_h, pair_w), dtype=int)
    _paint_coords(canvas, _mask_coords(a_shape, a_top, a_left), 1)
    conflict = any(canvas[r, c] for r, c in _mask_coords(b_shape, b_top, b_left))
    if conflict:
        raise AssertionError("overlap pair construction collided")
    _paint_coords(canvas, _mask_coords(b_shape, b_top, b_left), 2)

    overlap_box = (int(top), int(left), int(top + overlap_h - 1), int(left + overlap_w - 1))
    assert np.all(canvas[overlap_box[0] : overlap_box[2] + 1, overlap_box[1] : overlap_box[3] + 1] == 0)

    objects = [
        (a_shape, int(a_top), int(a_left), _DARK_COLOR),
        (b_shape, int(b_top), int(b_left), _LIGHT_COLOR),
    ]
    return canvas, objects, overlap_box


def _random_isolated_mask(rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        height = rng.randint(2, 4)
        width = rng.randint(2, 4)
        return _solid_rect_mask(height, width)
    thickness = rng.choice(_THICKNESS_OPTIONS)
    height = rng.randint(thickness + 2, thickness + 5)
    width = rng.randint(thickness + 2, thickness + 5)
    return _corner_mask(height, width, rng.choice(("TL", "TR", "BL", "BR")), thickness)


def _random_nonoverlap_pair(rng: random.Random) -> list[tuple[np.ndarray, int]]:
    result: list[tuple[np.ndarray, int]] = []
    for color in (_DARK_COLOR, _LIGHT_COLOR):
        thickness = rng.choice(_THICKNESS_OPTIONS)
        height = rng.randint(thickness + 2, thickness + 6)
        width = rng.randint(thickness + 2, thickness + 6)
        mask = _corner_mask(height, width, rng.choice(("TL", "TR", "BL", "BR")), thickness)
        result.append((mask, color))
    return result


def _validate_grid(grid: np.ndarray, expect_overlap: bool) -> bool:
    bg = _bg_color(grid)
    if Counter(grid.flatten()).most_common(1)[0][0] != bg:
        return False
    objects = _find_objects(grid, bg)
    colors = {int(obj["color"]) for obj in objects}
    if len(objects) != len(colors):
        return False
    overlap_count = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _bbox_intersection(objects[i]["bbox"], objects[j]["bbox"]) is not None:
                overlap_count += 1
    if expect_overlap and overlap_count != 1:
        return False
    if not expect_overlap and overlap_count != 0:
        return False
    out = main(grid)
    changed = not np.array_equal(out, grid)
    return changed if expect_overlap else not changed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*_GRID_H_RANGE)
    w = rng.randint(*_GRID_W_RANGE)
    bg = rng.choice(_BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()
    bboxes: list[tuple[int, int, int, int]] = []

    make_overlap = rng.random() < _OVERLAP_BRANCH_PROB
    if make_overlap:
        _, objects, _ = _build_overlap_pair(rng)
        pair_h = max(int(top + mask.shape[0]) for mask, top, _left, _color in objects)
        pair_w = max(int(left + mask.shape[1]) for mask, _top, left, _color in objects)
        if pair_h > h or pair_w > w:
            return None
        base_top = rng.randint(0, h - pair_h)
        base_left = rng.randint(0, w - pair_w)
        for mask, top, left, color in objects:
            coords = _mask_coords(mask, base_top + top, base_left + left)
            bbox = _bbox_from_mask(mask, base_top + top, base_left + left)
            if _touches_existing(coords, occupied):
                return None
            _paint_coords(grid, coords, color)
            occupied.update(coords)
            bboxes.append(bbox)
    else:
        for mask, color in _random_nonoverlap_pair(rng):
            placed = _place_mask(rng, grid.shape, mask, occupied, bboxes, forbid_bbox_overlap=True)
            if placed is None:
                return None
            coords, bbox = placed
            _paint_coords(grid, coords, color)
            occupied.update(coords)
            bboxes.append(bbox)

    if rng.random() < _THIRD_OBJECT_PROB:
        extra_mask = _random_isolated_mask(rng)
        placed = _place_mask(rng, grid.shape, extra_mask, occupied, bboxes, forbid_bbox_overlap=True)
        if placed is None:
            return None
        coords, bbox = placed
        _paint_coords(grid, coords, _EXTRA_COLOR)
        occupied.update(coords)
        bboxes.append(bbox)

    if np.count_nonzero(grid != bg) >= (h * w) // 2:
        return None
    if not _validate_grid(grid, expect_overlap=make_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid handshake-grid input")
