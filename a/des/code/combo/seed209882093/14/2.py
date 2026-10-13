import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _rotation_order(patch: np.ndarray) -> int:
    count = 0
    for k in range(4):
        if np.array_equal(np.rot90(patch, k), patch):
            count += 1
    return count


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BLACK, GREY) or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            r0 = min(rr for rr, _ in coords)
            r1 = max(rr for rr, _ in coords)
            c0 = min(cc for _, cc in coords)
            c1 = max(cc for _, cc in coords)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
            for rr, cc in coords:
                patch[rr - r0, cc - c0] = color

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "first": min(coords),
                    "patch": patch,
                    "order": _rotation_order(patch),
                }
            )

    return objects


def _stack_objects(objects: list[dict]) -> np.ndarray:
    ordered = sorted(objects, key=lambda obj: (obj["order"], obj["first"][0], obj["first"][1]))
    out_w = max(obj["patch"].shape[1] for obj in ordered)
    out_h = sum(obj["patch"].shape[0] for obj in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)

    row = 0
    for obj in ordered:
        patch = obj["patch"]
        ph, pw = patch.shape
        col = (out_w - pw) // 2
        mask = patch != BLACK
        out[row : row + ph, col : col + pw][mask] = patch[mask]
        row += ph + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each monochrome object, compute its rotational symmetry order,
    sort by that order, and stack the objects vertically.

    Input structure:
    - Background is black.
    - Several non-touching monochrome 4-connected objects of distinct colors.
    - A grey vertical ruler sits at the bottom-right; its height equals the number of objects.

    Rule:
    - Ignore the grey ruler.
    - For each object, crop its bounding box and count how many rotations among
      0, 90, 180, 270 degrees match exactly. The count is 1, 2, or 4.
    - Sort objects by symmetry order ascending; ties use row-major appearance order.
    - Output width is the maximum object bbox width.
    - Output height is the sum of object bbox heights plus one black spacer row
      between consecutive objects.
    - Place objects top-to-bottom, horizontally centered.
    """
    objects = _extract_objects(np.asarray(grid, dtype=int))
    if not objects:
        return np.zeros((0, 0), dtype=int)
    return _stack_objects(objects)


# --------------- generator ---------------

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (4, 7)
MAX_GENERATION_ATTEMPTS = 400
MAX_SHAPE_ATTEMPTS = 300


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(x) for x in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    queue = deque([cells[0]])
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        moves = []
        if r < er:
            moves.append((r + 1, c))
        elif r > er:
            moves.append((r - 1, c))
        if c < ec:
            moves.append((r, c + 1))
        elif c > ec:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        path.append((r, c))
    return path


def _rotate_coord_90_ccw(r: int, c: int, center: int) -> tuple[int, int]:
    dr, dc = r - center, c - center
    return center - dc, center + dr


def _random_variant(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _trim_mask(out)


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask] = int(color)
    return patch


def _generate_order1_mask() -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(2, 7)
        w = random.randint(2, 7)
        max_area = min(h * w - 1, 16)
        min_area = max(4, (h + w) // 2)
        if min_area > max_area:
            continue
        mask = _random_connected_mask(h, w, random.randint(min_area, max_area))
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if not _is_connected(mask):
            continue
        if _rotation_order(mask.astype(int)) != 1:
            continue
        return mask
    raise RuntimeError("failed to build order-1 mask")


def _generate_order2_mask() -> np.ndarray:
    dims = [(h, w) for h in (3, 5, 7) for w in (3, 5, 7) if h != w]
    corners = {
        "tl": lambda h, w: (0, 0),
        "tr": lambda h, w: (0, w - 1),
        "bl": lambda h, w: (h - 1, 0),
        "br": lambda h, w: (h - 1, w - 1),
    }
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h, w = random.choice(dims)
        center = (h // 2, w // 2)
        target = random.choice(list(corners.values()))(h, w)
        base = set(_random_path(center, target))

        for _ in range(random.randint(0, max(1, (h + w) // 3))):
            candidates = set()
            for r, c in base:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in base:
                        candidates.add((nr, nc))
            if not candidates:
                break
            base.add(random.choice(tuple(candidates)))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in base:
            mask[r, c] = True
            mask[h - 1 - r, w - 1 - c] = True

        if not _is_connected(mask):
            continue
        if _rotation_order(mask.astype(int)) != 2:
            continue
        return mask
    raise RuntimeError("failed to build order-2 mask")


def _generate_order4_mask() -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        n = random.choice([3, 5, 7])
        center = n // 2
        targets = [
            (r, c)
            for r in range(center + 1)
            for c in range(center, n)
            if (r, c) != (center, center) and max(abs(r - center), abs(c - center)) == center
        ]
        target = random.choice(targets)
        base = set(_random_path((center, center), target))

        for _ in range(random.randint(0, n)):
            candidates = set()
            for r, c in base:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n and nr <= center and nc >= center and (nr, nc) not in base:
                        candidates.add((nr, nc))
            if not candidates:
                break
            base.add(random.choice(tuple(candidates)))

        mask = np.zeros((n, n), dtype=bool)
        for r, c in base:
            rr, cc = r, c
            for _ in range(4):
                mask[rr, cc] = True
                rr, cc = _rotate_coord_90_ccw(rr, cc, center)

        if not _is_connected(mask):
            continue
        if _rotation_order(mask.astype(int)) != 4:
            continue
        return mask
    raise RuntimeError("failed to build order-4 mask")


def _make_mask(order: int) -> np.ndarray:
    if order == 1:
        return _random_variant(_generate_order1_mask())
    if order == 2:
        return _random_variant(_generate_order2_mask())
    if order == 4:
        return _random_variant(_generate_order4_mask())
    raise ValueError(order)


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_objects(
    grid: np.ndarray, objects: list[dict], usable_width: int
) -> bool:
    h, _ = grid.shape
    occupied = np.zeros_like(grid, dtype=bool)
    order = sorted(range(len(objects)), key=lambda i: int(objects[i]["mask"].sum()), reverse=True)

    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        ph, pw = mask.shape
        origins = [(r0, c0) for r0 in range(h - ph + 1) for c0 in range(usable_width - pw + 1)]
        random.shuffle(origins)
        placed = False
        for r0, c0 in origins:
            if (occupied[r0 : r0 + ph, c0 : c0 + pw] & mask).any():
                continue
            padded = _dilate(occupied)
            if (padded[r0 : r0 + ph, c0 : c0 + pw] & mask).any():
                continue
            patch = obj["patch"]
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            obj["origin"] = (r0, c0)
            obj["coords"] = [(r0 + rr, c0 + cc) for rr, cc in np.argwhere(mask)]
            obj["first"] = min(obj["coords"])
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_from_generated(objects: list[dict]) -> np.ndarray:
    packed = []
    for obj in objects:
        packed.append(
            {
                "patch": obj["patch"],
                "order": _rotation_order(obj["patch"]),
                "first": obj["first"],
            }
        )
    return _stack_objects(packed)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    orders = [1, 2, 4] + [random.choice([1, 2, 4]) for _ in range(count - 3)]
    random.shuffle(orders)
    colors = random.sample(OBJECT_COLORS, count)

    objects = []
    for color, order in zip(colors, orders):
        mask = _make_mask(order)
        patch = _mask_to_patch(mask, color)
        objects.append({"mask": mask, "patch": patch, "order": order})

    max_h = max(obj["patch"].shape[0] for obj in objects)
    max_w = max(obj["patch"].shape[1] for obj in objects)
    min_h = max(12, max_h + 4, count + 4)
    min_w = max(12, max_w + 4)

    H = random.randint(min_h, max(GRID_H_RANGE[0], min_h))
    H = random.randint(H, GRID_H_RANGE[1] - 1)
    W = random.randint(min_w, max(GRID_W_RANGE[0], min_w))
    W = random.randint(W, GRID_W_RANGE[1] - 1)
    usable_width = W - 2
    if usable_width < max_w:
        return None

    grid = np.full((H, W), BLACK, dtype=int)
    if not _place_objects(grid, objects, usable_width):
        return None

    grid[H - count :, W - 1] = GREY

    expected = _expected_from_generated(objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    extracted = _extract_objects(grid)
    if len(extracted) != count:
        return None
    if int(np.sum(grid == GREY)) != count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
