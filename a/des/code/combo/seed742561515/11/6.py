import random
from collections import deque

import numpy as np


MAX_GENERATION_ATTEMPTS = 300
GRID_SIDE_RANGE = (24, 34)
PAIR_COUNT_RANGE = (2, 4)
OUTER_MARGIN = 1
PLACEMENT_PADDING = 1
PLACEMENT_ATTEMPTS = 200
SHAPE_ATTEMPTS = 200
LOCK_ATTEMPTS = 80


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _grid_components(grid: np.ndarray, bg: int) -> list[dict]:
    components = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _mask_components(grid == color):
            bbox = _bbox(coords)
            mask = _coords_to_mask(coords)
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                }
            )
    return components


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    holes = []
    for coords in _mask_components(~mask):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 1] == 0):
            continue
        if np.any(coords[:, 0] == mask.shape[0] - 1) or np.any(coords[:, 1] == mask.shape[1] - 1):
            continue
        holes.append(coords)
    return holes


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _strictly_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = inner
    R0, C0, R1, C1 = outer
    return R0 < r0 and C0 < c0 and r1 < R1 and c1 < C1


def _extract_key(grid: np.ndarray, bg: int, fence: dict) -> dict:
    r0, c0, r1, c1 = fence["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    key_mask = (inner != bg) & (inner != fence["color"])
    key_components = _mask_components(key_mask)
    assert len(key_components) == 1
    rel_coords = key_components[0]
    abs_coords = rel_coords + np.array([r0 + 1, c0 + 1])
    key_color = int(grid[abs_coords[0, 0], abs_coords[0, 1]])
    assert np.all(grid[abs_coords[:, 0], abs_coords[:, 1]] == key_color)
    mask = _coords_to_mask(abs_coords)
    return {
        "color": key_color,
        "coords": abs_coords,
        "mask": mask,
        "signature": _mask_signature(mask),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _grid_components(g, bg)

    fences = [comp for comp in components if _is_rect_frame(comp["mask"])]
    assert fences, "expected at least one fence"
    fence_boxes = [fence["bbox"] for fence in fences]

    keys = [_extract_key(g, bg, fence) for fence in fences]
    key_by_shape = {}
    for key in keys:
        assert key["signature"] not in key_by_shape
        key_by_shape[key["signature"]] = key

    locks = []
    for comp in components:
        if any(comp is fence for fence in fences):
            continue
        if any(_strictly_inside(comp["bbox"], fence_box) for fence_box in fence_boxes):
            continue
        holes = _enclosed_holes(comp["mask"])
        if len(holes) != 1:
            continue
        hole_mask = _coords_to_mask(holes[0])
        locks.append(
            {
                "coords": comp["coords"],
                "signature": _mask_signature(hole_mask),
            }
        )

    assert len(locks) == len(keys) and locks, "keys and locks must pair one-to-one"

    out = g.copy()
    for key in keys:
        out[key["coords"][:, 0], key["coords"][:, 1]] = bg

    matched = 0
    for lock in locks:
        assert lock["signature"] in key_by_shape
        key = key_by_shape[lock["signature"]]
        out[lock["coords"][:, 0], lock["coords"][:, 1]] = key["color"]
        matched += 1

    assert matched == len(keys)
    return out


def _dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _random_key_shape() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        max_area = min(7, h * w - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask = _coords_to_mask(coords)
        mask = _dihedral(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if _enclosed_holes(mask):
            continue
        return mask
    raise RuntimeError("failed to generate key shape")


def _make_fence_patch(shape: np.ndarray, bg: int, fence_color: int, key_color: int) -> np.ndarray:
    sh, sw = shape.shape
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    inner_h = sh + top + bottom
    inner_w = sw + left + right
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = fence_color
    patch[-1, :] = fence_color
    patch[:, 0] = fence_color
    patch[:, -1] = fence_color
    r0 = 1 + top
    c0 = 1 + left
    patch[r0 : r0 + sh, c0 : c0 + sw][shape] = key_color
    return patch


def _make_lock_patch(shape: np.ndarray, bg: int, lock_color: int) -> np.ndarray:
    sh, sw = shape.shape
    for _ in range(LOCK_ATTEMPTS):
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        base_h = sh + top + bottom
        base_w = sw + left + right

        canvas = np.zeros((base_h + 6, base_w + 6), dtype=bool)
        off_r = 3
        off_c = 3
        canvas[off_r : off_r + base_h, off_c : off_c + base_w] = True
        hole_view = canvas[off_r + top : off_r + top + sh, off_c + left : off_c + left + sw]
        hole_view[shape] = False

        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides[: random.randint(0, 2)]:
            if side in {"top", "bottom"}:
                arm_h = random.randint(1, 2)
                arm_w = random.randint(max(2, base_w // 2), base_w)
                arm_c = off_c + random.randint(0, base_w - arm_w)
                if side == "top":
                    canvas[off_r - arm_h : off_r, arm_c : arm_c + arm_w] = True
                else:
                    canvas[off_r + base_h : off_r + base_h + arm_h, arm_c : arm_c + arm_w] = True
            else:
                arm_w = random.randint(1, 2)
                arm_h = random.randint(max(2, base_h // 2), base_h)
                arm_r = off_r + random.randint(0, base_h - arm_h)
                if side == "left":
                    canvas[arm_r : arm_r + arm_h, off_c - arm_w : off_c] = True
                else:
                    canvas[arm_r : arm_r + arm_h, off_c + base_w : off_c + base_w + arm_w] = True

        coords = np.argwhere(canvas)
        mask = _coords_to_mask(coords)
        if _is_rect_frame(mask):
            continue
        holes = _enclosed_holes(mask)
        if len(holes) != 1:
            continue
        if _mask_signature(_coords_to_mask(holes[0])) != _mask_signature(shape):
            continue

        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = lock_color
        return patch
    raise RuntimeError("failed to generate lock patch")


def _reserve_ok(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    r0 = max(0, r - PLACEMENT_PADDING)
    c0 = max(0, c - PLACEMENT_PADDING)
    r1 = min(occupied.shape[0], r + h + PLACEMENT_PADDING)
    c1 = min(occupied.shape[1], c + w + PLACEMENT_PADDING)
    return not occupied[r0:r1, c0:c1].any()


def _place_items(items: list[dict], bg: int) -> np.ndarray | None:
    total_area = sum(item["patch"].shape[0] * item["patch"].shape[1] for item in items)
    base_side = max(
        GRID_SIDE_RANGE[0],
        int(np.ceil(np.sqrt(total_area) * 1.25)),
        max(item["patch"].shape[0] for item in items) + 2 * OUTER_MARGIN,
        max(item["patch"].shape[1] for item in items) + 2 * OUTER_MARGIN,
    )
    low = min(GRID_SIDE_RANGE[1], base_side)
    high = max(low, min(GRID_SIDE_RANGE[1], base_side + 5))
    h = random.randint(low, high)
    w = random.randint(low, high)

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    order = items[:]
    random.shuffle(order)
    order.sort(key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)

    for item in order:
        patch = item["patch"]
        ph, pw = patch.shape
        if ph + 2 * OUTER_MARGIN > h or pw + 2 * OUTER_MARGIN > w:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r = random.randint(OUTER_MARGIN, h - ph - OUTER_MARGIN)
            c = random.randint(OUTER_MARGIN, w - pw - OUTER_MARGIN)
            if not _reserve_ok(occupied, r, c, ph, pw):
                continue
            region = grid[r : r + ph, c : c + pw]
            region[patch != bg] = patch[patch != bg]
            occupied[r : r + ph, c : c + pw] = True
            item["top"] = r
            item["left"] = c
            placed = True
            break
        if not placed:
            return None
    return grid


def _expected_output(grid: np.ndarray, items: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    pair_info: dict[int, dict] = {}
    for item in items:
        info = pair_info.setdefault(item["pair"], {})
        patch = item["patch"]
        r = item["top"]
        c = item["left"]
        if item["kind"] == "fence":
            coords = np.argwhere(patch == item["key_color"])
            info["key_coords"] = coords + np.array([r, c])
            info["key_color"] = item["key_color"]
        else:
            coords = np.argwhere(patch == item["lock_color"])
            info["lock_coords"] = coords + np.array([r, c])
    for info in pair_info.values():
        out[info["key_coords"][:, 0], info["key_coords"][:, 1]] = bg
        out[info["lock_coords"][:, 0], info["lock_coords"][:, 1]] = info["key_color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    palette = list(range(10))
    bg, fence_color, lock_color = random.sample(palette, 3)
    remaining = [color for color in palette if color not in {bg, fence_color, lock_color}]
    key_colors = random.sample(remaining, pair_count)

    shapes = []
    seen_shapes = set()
    for _ in range(pair_count):
        for _ in range(SHAPE_ATTEMPTS):
            shape = _random_key_shape()
            sig = _mask_signature(shape)
            if sig in seen_shapes:
                continue
            seen_shapes.add(sig)
            shapes.append(shape)
            break
        else:
            return None

    items = []
    for idx, (shape, key_color) in enumerate(zip(shapes, key_colors)):
        try:
            fence_patch = _make_fence_patch(shape, bg, fence_color, key_color)
            lock_patch = _make_lock_patch(shape, bg, lock_color)
        except RuntimeError:
            return None
        items.append(
            {
                "pair": idx,
                "kind": "fence",
                "patch": fence_patch,
                "key_color": key_color,
            }
        )
        items.append(
            {
                "pair": idx,
                "kind": "lock",
                "patch": lock_patch,
                "lock_color": lock_color,
            }
        )

    grid = _place_items(items, bg)
    if grid is None:
        return None
    if _get_bg(grid) != bg:
        return None

    expected = _expected_output(grid, items, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
