import random
from collections import Counter, deque

import numpy as np


BLACK = 0
YELLOW = 4
OBJECT_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
DIRECTIONS = ("up", "right", "down", "left")
DIR_VECTORS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
# np.rot90 uses counterclockwise turns.
ROT_K_BY_DIRECTION = {
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_from_mask(mask: np.ndarray, conn8: bool) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    dirs = DIR8 if conn8 else DIR4
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        visited[r, c] = True
        q = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((rr, cc))
            for dr, dc in dirs:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.asarray(cells, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _hole_mask(component_mask: np.ndarray) -> np.ndarray:
    walls = np.asarray(component_mask, dtype=bool)
    h, w = walls.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not walls[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not walls[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return (~walls) & (~outside)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    cells = _normalize_cells(set(cells))
    h = 1 + max(r for r, _ in cells)
    w = 1 + max(c for _, c in cells)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mask_to_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _translate_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _bbox_area(cells: set[tuple[int, int]]) -> int:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _is_rotationally_asymmetric(mask: np.ndarray) -> bool:
    base = _crop_true(mask)
    for k in (1, 2, 3):
        rot = _crop_true(np.rot90(base, k))
        if base.shape == rot.shape and np.array_equal(base, rot):
            return False
    return True


def _random_polyomino() -> np.ndarray | None:
    for _ in range(200):
        box_h = random.randint(2, 5)
        box_w = random.randint(2, 6)
        max_area = min(box_h * box_w - 1, 10)
        min_area = max(4, min(max_area, (box_h + box_w + 1) // 2))
        if min_area > max_area:
            continue

        target_area = random.randint(min_area, max_area)
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}

        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        mask = _cells_to_mask(cells)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        mask = _crop_true(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.shape[0] * mask.shape[1]):
            continue
        if _hole_mask(mask).any():
            continue
        if not _is_rotationally_asymmetric(mask):
            continue
        return mask

    return None


def _arrow_cells(direction: str) -> set[tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    right = (dc, -dr)
    cells = {(0, 0), (-dr, -dc), right}
    return _normalize_cells(cells)


def _min_manhattan(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> int:
    best = 10**9
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            best = min(best, abs(ra - rb) + abs(ca - cb))
    return best


def _build_lock_package(cavity_mask: np.ndarray, direction: str) -> dict | None:
    cavity_patch = np.zeros((cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2), dtype=bool)
    cavity_patch[1:-1, 1:-1] = cavity_mask
    ring = _dilate4(cavity_patch) & (~cavity_patch)

    if len(_components_from_mask(ring, conn8=True)) != 1:
        return None
    if not np.array_equal(_hole_mask(ring), cavity_patch):
        return None

    lock_h, lock_w = ring.shape
    canvas_h = lock_h + 6
    canvas_w = lock_w + 6
    lock_r0 = 3
    lock_c0 = 3

    lock_cells = {(lock_r0 + int(r), lock_c0 + int(c)) for r, c in np.argwhere(ring)}
    cavity_cells = {
        (lock_r0 + int(r), lock_c0 + int(c))
        for r, c in np.argwhere(cavity_patch)
        if cavity_patch[int(r), int(c)]
    }

    arrow_mask = _cells_to_mask(_arrow_cells(direction))
    arrow_h, arrow_w = arrow_mask.shape
    arrow_shape = _mask_to_cells(arrow_mask)
    lock_r1 = lock_r0 + lock_h - 1
    lock_c1 = lock_c0 + lock_w - 1

    candidates: list[set[tuple[int, int]]] = []
    for top in range(canvas_h - arrow_h + 1):
        for left in range(canvas_w - arrow_w + 1):
            arrow_cells = {(top + r, left + c) for r, c in arrow_shape}
            if any(lock_r0 <= r <= lock_r1 and lock_c0 <= c <= lock_c1 for r, c in arrow_cells):
                continue
            if _min_manhattan(arrow_cells, lock_cells) != 1:
                continue
            candidates.append(arrow_cells)

    if not candidates:
        return None

    arrow_cells = random.choice(candidates)
    all_cells = lock_cells | arrow_cells | cavity_cells
    r0 = min(r for r, _ in all_cells)
    c0 = min(c for _, c in all_cells)
    return {
        "lock": {(r - r0, c - c0) for r, c in lock_cells},
        "cavity": {(r - r0, c - c0) for r, c in cavity_cells},
        "arrow": {(r - r0, c - c0) for r, c in arrow_cells},
    }


def _can_place(
    occupied: np.ndarray,
    shape_cells: set[tuple[int, int]],
    top: int,
    left: int,
    pad: int,
) -> bool:
    h, w = occupied.shape
    for r, c in shape_cells:
        rr = top + r
        cc = left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0 = max(0, rr - pad)
        r1 = min(h, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(w, cc + pad + 1)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _place_shape(
    occupied: np.ndarray,
    shape_cells: set[tuple[int, int]],
    pad: int,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    shape_h = 1 + max(r for r, _ in shape_cells)
    shape_w = 1 + max(c for _, c in shape_cells)
    if shape_h > h or shape_w > w:
        return None

    positions = [(r, c) for r in range(h - shape_h + 1) for c in range(w - shape_w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if _can_place(occupied, shape_cells, top, left, pad):
            return top, left
    return None


def _arrow_direction(coords: np.ndarray) -> str:
    cells = {(int(r), int(c)) for r, c in coords}
    assert len(cells) == 3, "arrow must contain exactly three cells"
    for direction, (dr, dc) in DIR_VECTORS.items():
        right = (dc, -dr)
        for pr, pc in cells:
            expected = {(pr, pc), (pr - dr, pc - dc), (pr + right[0], pc + right[1])}
            if expected == cells:
                return direction
    raise AssertionError("yellow component is not a supported arrow")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    assert bg == BLACK, "this puzzle uses black background"
    out = g.copy()

    arrow_components = _components_from_mask(g == YELLOW, conn8=True)
    arrows = [{"coords": comp, "direction": _arrow_direction(comp)} for comp in arrow_components]
    used_arrow_ids: set[int] = set()

    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, YELLOW)]
    for color in colors:
        comps = _components_from_mask(g == color, conn8=True)
        assert len(comps) == 2, "each color should have exactly one lock and one key"

        lock_info = None
        key_info = None
        for comp in comps:
            mask, bbox = _tight_mask(comp)
            holes = _hole_mask(mask)
            if holes.any():
                hole_comps = _components_from_mask(holes, conn8=False)
                assert len(hole_comps) == 1, "lock should contain one cavity"
                lock_info = {
                    "coords": comp,
                    "bbox": bbox,
                    "mask": mask,
                    "holes": holes,
                    "hole_tight": _crop_true(holes),
                }
            else:
                key_info = {
                    "coords": comp,
                    "mask": mask,
                }

        assert lock_info is not None and key_info is not None, "failed to split lock and key"

        lock_cells = {(int(r), int(c)) for r, c in lock_info["coords"]}
        matches: list[tuple[int, int]] = []
        for idx, arrow in enumerate(arrows):
            if idx in used_arrow_ids:
                continue
            arrow_cells = {(int(r), int(c)) for r, c in arrow["coords"]}
            dist = _min_manhattan(lock_cells, arrow_cells)
            if dist == 1:
                matches.append((idx, dist))

        assert len(matches) == 1, "each lock should have exactly one adjacent arrow"
        arrow_idx = matches[0][0]
        used_arrow_ids.add(arrow_idx)
        rot_k = ROT_K_BY_DIRECTION[arrows[arrow_idx]["direction"]]

        rotated_key = np.rot90(key_info["mask"], rot_k)
        assert rotated_key.shape == lock_info["hole_tight"].shape
        assert np.array_equal(rotated_key, lock_info["hole_tight"])

        out[key_info["coords"][:, 0], key_info["coords"][:, 1]] = bg
        out[arrows[arrow_idx]["coords"][:, 0], arrows[arrow_idx]["coords"][:, 1]] = bg

        r0, c0, _r1, _c1 = lock_info["bbox"]
        for rr, cc in np.argwhere(lock_info["holes"]):
            out[r0 + int(rr), c0 + int(cc)] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    colors = random.sample(OBJECT_COLORS, n_objects)
    directions = random.sample(DIRECTIONS, n_objects)

    objects: list[dict] = []
    for color, direction in zip(colors, directions):
        cavity_mask = _random_polyomino()
        if cavity_mask is None:
            return None

        package = _build_lock_package(cavity_mask, direction)
        if package is None:
            return None

        key_mask = np.rot90(cavity_mask, (-ROT_K_BY_DIRECTION[direction]) % 4)
        key_cells = _mask_to_cells(key_mask)
        objects.append(
            {
                "color": color,
                "direction": direction,
                "lock": package["lock"],
                "cavity": package["cavity"],
                "arrow": package["arrow"],
                "lock_reserved": package["lock"] | package["cavity"] | package["arrow"],
                "key": key_cells,
            }
        )

    min_side = 15 + 4 * (n_objects - 2)
    max_side = min(30, min_side + 7)
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)
    occupied = np.zeros((h, w), dtype=bool)

    random.shuffle(objects)
    lock_order = sorted(range(n_objects), key=lambda i: _bbox_area(objects[i]["lock_reserved"]), reverse=True)
    key_order = sorted(range(n_objects), key=lambda i: _bbox_area(objects[i]["key"]), reverse=True)

    for idx in lock_order:
        pos = _place_shape(occupied, objects[idx]["lock_reserved"], pad=1)
        if pos is None:
            return None
        top, left = pos
        objects[idx]["lock_global"] = _translate_cells(objects[idx]["lock"], top, left)
        objects[idx]["cavity_global"] = _translate_cells(objects[idx]["cavity"], top, left)
        objects[idx]["arrow_global"] = _translate_cells(objects[idx]["arrow"], top, left)
        objects[idx]["lock_reserved_global"] = _translate_cells(objects[idx]["lock_reserved"], top, left)
        for r, c in objects[idx]["lock_reserved_global"]:
            occupied[r, c] = True

    for idx in key_order:
        pos = _place_shape(occupied, objects[idx]["key"], pad=1)
        if pos is None:
            return None
        top, left = pos
        objects[idx]["key_global"] = _translate_cells(objects[idx]["key"], top, left)
        for r, c in objects[idx]["key_global"]:
            occupied[r, c] = True

    inp = np.full((h, w), BLACK, dtype=int)
    expected = np.full((h, w), BLACK, dtype=int)

    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["lock_global"]:
            inp[r, c] = color
            expected[r, c] = color
        for r, c in obj["arrow_global"]:
            inp[r, c] = YELLOW
        for r, c in obj["key_global"]:
            inp[r, c] = color
        for r, c in obj["cavity_global"]:
            expected[r, c] = color

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
