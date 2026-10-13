import random
from collections import deque

import numpy as np


BLACK = 0
FRAME_COLOR = 5
WORKSPACE_SIDE = 30
GEN_ATTEMPTS = 400
LOCK_COUNT_CHOICES = (1, 2, 3, 4)
PLACEMENT_PADDING = 2
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
INNER_SIZE_BY_LOCK_COUNT = {
    1: (5, 8),
    2: (5, 8),
    3: (5, 7),
    4: (5, 6),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _tight_crop_bool(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_crop_bool(mask)
    return tuple(map(int, tight.shape)), tight.tobytes()


def _bbox_gap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b

    def gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return x0 - y1 - 1
        if x1 < y0:
            return y0 - x1 - 1
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _find_locks(grid: np.ndarray, bg: int) -> list[dict]:
    locks: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != int(bg)):
        for comp in _components(grid == int(color)):
            if not _is_rectangular_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inside = grid[r0 + 1 : r1, c0 + 1 : c1]
            inside_colors = {int(x) for x in np.unique(inside)}
            inside_non_bg = inside_colors - {int(bg)}
            if len(inside_non_bg) != 1:
                continue
            fill_color = next(iter(inside_non_bg))
            if inside_colors - {int(bg), int(fill_color)}:
                continue
            hole_mask = inside == int(bg)
            if not hole_mask.any():
                continue
            hole_comps = _components(hole_mask)
            if len(hole_comps) != 1:
                continue
            hole_local = hole_comps[0]
            hole_abs = [(r0 + 1 + r, c0 + 1 + c) for r, c in hole_local]
            locks.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "border_coords": [(int(r), int(c)) for r, c in comp],
                    "fill_color": int(fill_color),
                    "hole_coords": hole_abs,
                    "hole_mask": _tight_mask_from_coords(hole_local),
                    "shape_key": _shape_key(_tight_mask_from_coords(hole_local)),
                }
            )
    locks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return locks


def _find_keys(grid: np.ndarray, bg: int, locks: list[dict]) -> list[dict]:
    lock_area = np.zeros(grid.shape, dtype=bool)
    for lock in locks:
        r0, c0, r1, c1 = lock["bbox"]
        lock_area[r0 : r1 + 1, c0 : c1 + 1] = True
    key_mask = (grid != int(bg)) & (~lock_area)
    keys: list[dict] = []
    for comp in _components(key_mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each key must be monochrome"
        color = next(iter(colors))
        rel_mask = _tight_mask_from_coords(comp)
        keys.append(
            {
                "coords": [(int(r), int(c)) for r, c in comp],
                "color": int(color),
                "shape_key": _shape_key(rel_mask),
            }
        )
    keys.sort(key=lambda item: (len(item["coords"]), item["coords"][0]), reverse=True)
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each key shape into the matching lock hole and recolor that lock's border.

    Input structure:
    - The background is black.
    - Several locks appear as grey 1-pixel rectangular borders around solid colored rectangles with one black connected hole inside.
    - Separate monochrome key shapes lie elsewhere on the background, and each key's shape exactly matches one lock hole.

    Rule:
    - Detect every lock by its rectangular border and extract the black hole shape inside it.
    - Match each outside key to the unique lock whose hole has the same tight binary shape, with translation only.
    - Fill the matched hole with the key color, repaint that lock's grey border to the same key color, and erase the key from its original location.

    Key insight:
    The key-to-lock relation is encoded purely by exact shape equality between each loose key and one internal hole, and the border recolor is part of the required move.
    """
    g = np.asarray(grid, dtype=int)
    bg = BLACK if np.any(g == BLACK) else _most_common_color(g)
    locks = _find_locks(g, bg)
    assert locks, "expected at least one lock"
    keys = _find_keys(g, bg, locks)
    assert len(keys) == len(locks), "number of keys must equal number of locks"

    by_shape = {lock["shape_key"]: lock for lock in locks}
    assert len(by_shape) == len(locks), "lock hole shapes must be unique"

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = int(bg)

    matched = set()
    for key in keys:
        assert key["shape_key"] in by_shape, "each key must match a lock hole"
        lock = by_shape[key["shape_key"]]
        matched.add(lock["shape_key"])
        for r, c in lock["hole_coords"]:
            out[r, c] = int(key["color"])
        for r, c in lock["border_coords"]:
            out[r, c] = int(key["color"])

    assert len(matched) == len(locks), "matching must be one-to-one"
    return out


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], height, width))
    while int(mask.sum()) < int(area):
        rr, cc = random.choice(tuple(frontier))
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        for nr, nc in _neighbors4(rr, cc, height, width):
            if not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _random_hole_mask(max_h: int, max_w: int, used_shapes: set) -> np.ndarray | None:
    for _ in range(200):
        raw_h = random.randint(1, max_h)
        raw_w = random.randint(1, max_w)
        full_area = raw_h * raw_w
        max_area = full_area if raw_h == 1 or raw_w == 1 else full_area - 1
        if max_area <= 0:
            continue
        min_area = 1 if full_area <= 3 else 2
        area = random.randint(min_area, max_area)
        mask = _tight_crop_bool(_grow_connected_mask(raw_h, raw_w, area))
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        if mask.shape[0] > 1 and mask.shape[1] > 1 and bool(mask.all()):
            continue
        key = _shape_key(mask)
        if key in used_shapes:
            continue
        used_shapes.add(key)
        return mask
    return None


def _lock_patch(
    inner_h: int, inner_w: int, frame_color: int, fill_color: int, hole_mask: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    patch = np.full((inner_h + 2, inner_w + 2), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(fill_color)
    hole_h, hole_w = hole_mask.shape
    top = random.randint(1, inner_h - hole_h - 1)
    left = random.randint(1, inner_w - hole_w - 1)
    hole_coords: list[tuple[int, int]] = []
    for r in range(hole_h):
        for c in range(hole_w):
            if hole_mask[r, c]:
                patch[1 + top + r, 1 + left + c] = BLACK
                hole_coords.append((1 + top + r, 1 + left + c))
    border_coords = []
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            if r in (0, ph - 1) or c in (0, pw - 1):
                border_coords.append((r, c))
    return patch, border_coords, hole_coords


def _key_patch(color: int, hole_mask: np.ndarray) -> np.ndarray:
    patch = np.full(hole_mask.shape, BLACK, dtype=int)
    patch[hole_mask] = int(color)
    return patch


def _can_place(
    occupied: np.ndarray, patch_shape: tuple[int, int], top: int, left: int, padding: int
) -> bool:
    h, w = patch_shape
    side_h, side_w = occupied.shape
    if top < 0 or left < 0 or top + h > side_h or left + w > side_w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(side_h, top + h + padding)
    c1 = min(side_w, left + w + padding)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(
    occupied: np.ndarray, patch_shape: tuple[int, int], top: int, left: int
) -> None:
    h, w = patch_shape
    occupied[top : top + h, left : left + w] = True


def _place_patch(
    occupied: np.ndarray, patch_shape: tuple[int, int], padding: int
) -> tuple[int, int] | None:
    h, w = patch_shape
    if h > occupied.shape[0] or w > occupied.shape[1]:
        return None
    positions = [
        (r, c)
        for r in range(occupied.shape[0] - h + 1)
        for c in range(occupied.shape[1] - w + 1)
    ]
    random.shuffle(positions)
    for top, left in positions:
        if _can_place(occupied, patch_shape, top, left, padding):
            _mark_occupied(occupied, patch_shape, top, left)
            return int(top), int(left)
    return None


def _paint(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    target = grid[top : top + h, left : left + w]
    target[mask] = patch[mask]


def _nonzero_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(grid != BLACK)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_bounds_with_random_margin(
    grid: np.ndarray,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _nonzero_bbox(grid)
    top_margin = random.randint(0, min(3, r0))
    left_margin = random.randint(0, min(3, c0))
    bottom_margin = random.randint(0, min(3, grid.shape[0] - 1 - r1))
    right_margin = random.randint(0, min(3, grid.shape[1] - 1 - c1))
    return (
        int(r0 - top_margin),
        int(c0 - left_margin),
        int(r1 + bottom_margin),
        int(c1 + right_margin),
    )


def _gen_attempt() -> np.ndarray | None:
    n_locks = random.choice(LOCK_COUNT_CHOICES)
    inner_lo, inner_hi = INNER_SIZE_BY_LOCK_COUNT[n_locks]
    frame_color = FRAME_COLOR
    colors = random.sample(PALETTE, 2 * n_locks)

    descriptors = []
    used_shapes: set = set()
    for i in range(n_locks):
        inner_h = random.randint(inner_lo, inner_hi)
        inner_w = random.randint(inner_lo, inner_hi)
        hole_mask = _random_hole_mask(inner_h - 2, inner_w - 2, used_shapes)
        if hole_mask is None:
            return None
        fill_color = colors[2 * i]
        key_color = colors[2 * i + 1]
        lock_patch, border_rel, hole_rel = _lock_patch(
            inner_h, inner_w, frame_color, fill_color, hole_mask
        )
        descriptors.append(
            {
                "lock_patch": lock_patch,
                "key_patch": _key_patch(key_color, hole_mask),
                "key_color": int(key_color),
                "border_rel": border_rel,
                "hole_rel": hole_rel,
                "shape_key": _shape_key(hole_mask),
            }
        )

    occupied = np.zeros((WORKSPACE_SIDE, WORKSPACE_SIDE), dtype=bool)
    grid = np.full((WORKSPACE_SIDE, WORKSPACE_SIDE), BLACK, dtype=int)

    lock_order = sorted(
        range(n_locks),
        key=lambda idx: descriptors[idx]["lock_patch"].shape[0]
        * descriptors[idx]["lock_patch"].shape[1],
        reverse=True,
    )
    for idx in lock_order:
        top_left = _place_patch(
            occupied, descriptors[idx]["lock_patch"].shape, PLACEMENT_PADDING
        )
        if top_left is None:
            return None
        top, left = top_left
        descriptors[idx]["lock_pos"] = (int(top), int(left))
        _paint(grid, descriptors[idx]["lock_patch"], top, left)

    key_order = sorted(
        range(n_locks),
        key=lambda idx: descriptors[idx]["key_patch"].shape[0]
        * descriptors[idx]["key_patch"].shape[1],
        reverse=True,
    )
    for idx in key_order:
        top_left = _place_patch(
            occupied, descriptors[idx]["key_patch"].shape, PLACEMENT_PADDING
        )
        if top_left is None:
            return None
        top, left = top_left
        descriptors[idx]["key_pos"] = (int(top), int(left))
        _paint(grid, descriptors[idx]["key_patch"], top, left)

    expected = grid.copy()
    for desc in descriptors:
        key_top, key_left = desc["key_pos"]
        key_patch = desc["key_patch"]
        key_mask = key_patch != BLACK
        key_view = expected[
            key_top : key_top + key_patch.shape[0],
            key_left : key_left + key_patch.shape[1],
        ]
        key_view[key_mask] = BLACK

        lock_top, lock_left = desc["lock_pos"]
        for rr, cc in desc["hole_rel"]:
            expected[lock_top + rr, lock_left + cc] = int(desc["key_color"])
        for rr, cc in desc["border_rel"]:
            expected[lock_top + rr, lock_left + cc] = int(desc["key_color"])

    crop_r0, crop_c0, crop_r1, crop_c1 = _crop_bounds_with_random_margin(grid)
    cropped_input = grid[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    cropped_expected = expected[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
    if max(cropped_input.shape) > 30 or max(cropped_expected.shape) > 30:
        return None
    try:
        solved = main(cropped_input)
    except AssertionError:
        return None
    if not np.array_equal(solved, cropped_expected):
        return None
    return cropped_input


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
