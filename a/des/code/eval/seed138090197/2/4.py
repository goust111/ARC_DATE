import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

ARROW_COLORS = (RED, BLUE, GREEN)
OBJECT_COLORS = (4, 5, 6, 7, 8, 9)

TRANSFORM_BY_COLOR = {
    RED: "hflip",
    BLUE: "rot180",
    GREEN: "vflip",
}
BITS_BY_COLOR = {
    RED: (1, 0),
    BLUE: (1, 1),
    GREEN: (0, 1),
}
DIR_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_PATTERN_INFO = {
    frozenset(((0, 1), (1, 0), (1, 2))): {"direction": "up", "tip_rel": (0, 1)},
    frozenset(((0, 0), (0, 2), (1, 1))): {"direction": "down", "tip_rel": (1, 1)},
    frozenset(((0, 1), (1, 0), (2, 1))): {"direction": "left", "tip_rel": (1, 0)},
    frozenset(((0, 0), (1, 1), (2, 0))): {"direction": "right", "tip_rel": (1, 1)},
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _find_components(grid: np.ndarray, connectivity: int = 8) -> list[dict]:
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _classify_arrow(comp: dict) -> dict | None:
    if int(comp["color"]) not in TRANSFORM_BY_COLOR or len(comp["coords"]) != 3:
        return None

    r0, c0, _r1, _c1 = comp["bbox"]
    rel = frozenset((int(r) - r0, int(c) - c0) for r, c in comp["coords"])
    info = ARROW_PATTERN_INFO.get(rel)
    if info is None:
        return None

    tip = (r0 + info["tip_rel"][0], c0 + info["tip_rel"][1])
    return {
        "color": int(comp["color"]),
        "direction": info["direction"],
        "tip": tip,
        "coords": comp["coords"],
        "bbox": comp["bbox"],
    }


def _patch_from_comp(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    color = int(comp["color"])
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _apply_transform(patch: np.ndarray, name: str) -> np.ndarray:
    if name == "hflip":
        return np.fliplr(patch)
    if name == "vflip":
        return np.flipud(patch)
    if name == "rot180":
        return np.rot90(patch, 2)
    raise AssertionError(name)


def _target_index(arrow: dict, label_grid: np.ndarray) -> int:
    h, w = map(int, label_grid.shape)
    r, c = arrow["tip"]
    dr, dc = DIR_DELTA[arrow["direction"]]
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        idx = int(label_grid[r, c])
        if idx >= 0:
            return idx
        r += dr
        c += dc
    raise AssertionError("arrow must point to a target object")


def main(grid: np.ndarray) -> np.ndarray:
    """Transform the first object hit by each colored arrow and erase all arrows."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g, connectivity=8)
    arrows = []
    objects = []
    for comp in comps:
        arrow = _classify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
        else:
            objects.append(comp)

    assert arrows, "expected at least one arrow"
    assert objects, "expected at least one object"

    label_grid = np.full(g.shape, -1, dtype=int)
    patches = []
    for idx, comp in enumerate(objects):
        patch = _patch_from_comp(g, comp)
        colors = {int(v) for v in np.unique(patch) if int(v) != BG}
        assert len(colors) == 1
        patches.append(patch)
        for r, c in comp["coords"]:
            label_grid[int(r), int(c)] = idx

    ordered_arrows = sorted(arrows, key=lambda arrow: (arrow["tip"][1], arrow["tip"][0]))
    for arrow in ordered_arrows:
        obj_idx = _target_index(arrow, label_grid)
        patches[obj_idx] = _apply_transform(patches[obj_idx], TRANSFORM_BY_COLOR[arrow["color"]])

    out = np.full_like(g, BG)
    for comp, patch in zip(objects, patches):
        r0, c0, r1, c1 = comp["bbox"]
        mask = patch != BG
        out_slice = out[r0 : r1 + 1, c0 : c1 + 1]
        out_slice[mask] = patch[mask]

    return out


def _grow_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            raise AssertionError("failed to grow connected object")
        mask[random.choice(frontier)] = True
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _shape_is_good(mask: np.ndarray) -> bool:
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    if int(mask.sum()) < 3:
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    return True


def _make_object_patch(color: int) -> np.ndarray:
    for _ in range(200):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        max_area = min(height * width, 8)
        area = random.randint(3, max_area)
        mask = _grow_mask(height, width, area)
        if not _shape_is_good(mask):
            continue
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = color
        return patch
    raise RuntimeError("failed to build asymmetric object patch")


def _rect_clear(rects: list[tuple[int, int, int, int]], cand: tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = cand
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 + pad < rr0 or rr1 + pad < r0 or c1 + pad < cc0 or cc1 + pad < c0):
            return False
    return True


def _place_objects(height: int, width: int, patches: list[np.ndarray]) -> list[dict] | None:
    placed = []
    rects = []
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        positions = [
            (r0, c0)
            for r0 in range(3, height - ph - 2)
            for c0 in range(3, width - pw - 2)
        ]
        random.shuffle(positions)

        found = None
        for r0, c0 in positions:
            rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if _rect_clear(rects, rect, pad=3):
                found = rect
                break
        if found is None:
            return None

        rects.append(found)
        placed.append({"index": idx, "bbox": found, "patch": patch})

    placed.sort(key=lambda item: item["index"])
    return placed


def _arrow_pixels(tip: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r, c = tip
    if direction == "up":
        return [(r, c), (r + 1, c - 1), (r + 1, c + 1)]
    if direction == "down":
        return [(r, c), (r - 1, c - 1), (r - 1, c + 1)]
    if direction == "left":
        return [(r, c), (r - 1, c + 1), (r + 1, c + 1)]
    if direction == "right":
        return [(r, c), (r - 1, c - 1), (r + 1, c - 1)]
    raise AssertionError(direction)


def _object_coords(placed_obj: dict) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = placed_obj["bbox"]
    coords = []
    for rr, cc in np.argwhere(placed_obj["patch"] != BG):
        coords.append((r0 + int(rr), c0 + int(cc)))
    return coords


def _profiles(coords: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    top: dict[int, int] = {}
    bottom: dict[int, int] = {}
    left: dict[int, int] = {}
    right: dict[int, int] = {}
    for r, c in coords:
        top[c] = r if c not in top else min(top[c], r)
        bottom[c] = r if c not in bottom else max(bottom[c], r)
        left[r] = c if r not in left else min(left[r], c)
        right[r] = c if r not in right else max(right[r], c)
    return {
        "down": [(r, c) for c, r in sorted(top.items())],
        "up": [(r, c) for c, r in sorted(bottom.items())],
        "right": [(r, c) for r, c in sorted(left.items())],
        "left": [(r, c) for r, c in sorted(right.items())],
    }


def _can_use_pixels(grid: np.ndarray, pixels: list[tuple[int, int]]) -> bool:
    h, w = map(int, grid.shape)
    pixel_set = set(pixels)
    for r, c in pixels:
        if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != BG:
            return False
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pixel_set and int(grid[nr, nc]) != BG:
                return False
    return True


def _candidate_arrow(grid: np.ndarray, placed_obj: dict, direction: str) -> list[tuple[int, int]] | None:
    coords = _object_coords(placed_obj)
    profiles = _profiles(coords)[direction]
    random.shuffle(profiles)
    for r, c in profiles:
        if direction == "down":
            tip = (r - 2, c)
        elif direction == "up":
            tip = (r + 2, c)
        elif direction == "right":
            tip = (r, c - 2)
        else:
            tip = (r, c + 2)
        pixels = _arrow_pixels(tip, direction)
        if _can_use_pixels(grid, pixels):
            return pixels
    return None


def _xor_bits(values: list[tuple[int, int]]) -> tuple[int, int]:
    a = b = 0
    for x, y in values:
        a ^= x
        b ^= y
    return a, b


def _build_requests(object_count: int) -> list[tuple[int, int]] | None:
    targets = random.sample(range(object_count), 3)
    requests = list(zip(targets, ARROW_COLORS))
    extra_count = random.choices((0, 1, 2, 3), weights=(2, 4, 3, 1), k=1)[0]

    for _ in range(extra_count):
        if random.random() < 0.65:
            target = requests[random.randrange(len(requests))][0]
        else:
            target = random.randrange(object_count)
        color = random.choice(ARROW_COLORS)
        requests.append((target, color))

    by_object: dict[int, list[int]] = defaultdict(list)
    for target, color in requests:
        by_object[target].append(color)

    if any(len(colors) > 4 for colors in by_object.values()):
        return None
    if any(_xor_bits([BITS_BY_COLOR[color] for color in colors]) == (0, 0) for colors in by_object.values()):
        return None
    return requests


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    assert not np.any(region[mask] != BG)
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    object_count = random.randint(3, 5)
    object_colors = random.sample(OBJECT_COLORS, object_count)
    patches = [_make_object_patch(color) for color in object_colors]
    placed = _place_objects(height, width, patches)
    if placed is None:
        return None

    requests = _build_requests(object_count)
    if requests is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    for item in placed:
        r0, c0, _r1, _c1 = item["bbox"]
        _paint_patch(grid, item["patch"], r0, c0)

    directions_by_object: dict[int, list[str]] = defaultdict(list)
    request_order = list(range(len(requests)))
    random.shuffle(request_order)
    request_order.sort(key=lambda idx: requests[idx][0])

    for req_idx in request_order:
        obj_idx, color = requests[req_idx]
        free_dirs = [d for d in DIR_DELTA if d not in directions_by_object[obj_idx]]
        random.shuffle(free_dirs)

        placed_pixels = None
        for direction in free_dirs:
            pixels = _candidate_arrow(grid, placed[obj_idx], direction)
            if pixels is None:
                continue
            for r, c in pixels:
                grid[r, c] = color
            directions_by_object[obj_idx].append(direction)
            placed_pixels = pixels
            break
        if placed_pixels is None:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if any(color not in set(map(int, grid.ravel())) for color in ARROW_COLORS):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid arrow/object puzzle")
