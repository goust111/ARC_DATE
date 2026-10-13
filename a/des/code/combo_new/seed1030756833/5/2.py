import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
FOREGROUND_COLORS = tuple(range(1, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATION_RETRIES = 500
SHAPE_RETRIES = 200
PAIR_ORIENTATION_RETRIES = 80


def _trim(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_key(mask: np.ndarray) -> tuple:
    mask = _trim(mask)
    return (int(mask.shape[0]), int(mask.shape[1]), tuple(map(int, mask.ravel())))


def _orient(mask: np.ndarray, transform_id: int) -> np.ndarray:
    base = np.fliplr(mask) if transform_id >= 4 else mask
    return _trim(np.rot90(base, transform_id % 4))


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    return [_orient(mask, t) for t in range(8)]


def _canonical_shape_key(mask: np.ndarray) -> tuple:
    return min(_mask_key(m) for m in _dihedral_masks(mask))


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(grid[sr, sc]) == BLACK:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not seen[nr, nc] and int(grid[nr, nc]) != BLACK:
                            seen[nr, nc] = True
                            q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            vals = {int(grid[r, c]) for r, c in coords}
            assert len(vals) == 1, "each foreground object must be monochrome"
            color = vals.pop()
            assert color != BLACK

            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True

            objects.append(
                {
                    "coords": arr,
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Swap colors between paired fragments whose shapes match up to rotation or reflection.

    Input structure:
    - The background is Black.
    - The foreground consists of an even number of separated 4-connected objects.
    - Each object is monochrome, all object colors are distinct, and every normalized
      shape class has exactly two objects when rotations and flips are allowed.

    Rule:
    - Extract each foreground object in its tight bounding box and canonicalize its
      binary shape over all four rotations and reflected rotations.
    - For each pair with the same canonical shape, repaint the first object's cells
      with the second object's color and the second object's cells with the first
      object's color.
    - Keep every object in its original location and leave all Black cells unchanged.

    Key insight:
    - Pairing is determined by dihedral-equivalent shape, not by color or position;
      after the hidden shape matches are found, the operation is only a color swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    objects = _find_objects(g)
    assert objects and len(objects) % 2 == 0
    colors = [obj["color"] for obj in objects]
    assert len(colors) == len(set(colors)), "object colors must be distinct"

    groups = defaultdict(list)
    for obj in objects:
        groups[_canonical_shape_key(obj["mask"])].append(obj)

    out = g.copy()
    for pair in groups.values():
        assert len(pair) == 2, "each shape class must define one unambiguous pair"
        a, b = pair
        out[a["coords"][:, 0], a["coords"][:, 1]] = b["color"]
        out[b["coords"][:, 0], b["coords"][:, 1]] = a["color"]

    return out


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        max_area = h * w
        min_area = 4
        max_target = min(13, max_area - 1)
        if max_target < min_area:
            continue
        target = random.randint(min_area, max_target)

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim(mask)

        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if len({_mask_key(m) for m in _dihedral_masks(mask)}) < 3:
            continue
        return mask

    return None


def _choose_pair_masks(mask: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PAIR_ORIENTATION_RETRIES):
        if mode == "rotate":
            a_id = random.randrange(4)
            b_id = random.choice([t for t in range(4) if t != a_id])
        elif mode == "flip":
            a_id = random.randrange(4)
            b_id = random.randrange(4, 8)
        else:
            a_id = random.randrange(8)
            b_id = random.choice([t for t in range(8) if t != a_id])

        a = _orient(mask, a_id)
        b = _orient(mask, b_id)
        if a.shape != b.shape or not np.array_equal(a, b):
            return a, b

    return None


def _place_items(height: int, width: int, items: list[tuple[np.ndarray, int]]) -> np.ndarray | None:
    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    shuffled = items[:]
    random.shuffle(shuffled)

    for mask, color in shuffled:
        mh, mw = map(int, mask.shape)
        if mh + 2 > height or mw + 2 > width:
            return None

        candidates = [
            (r, c)
            for r in range(1, height - mh)
            for c in range(1, width - mw)
        ]
        random.shuffle(candidates)

        placed = False
        for r, c in candidates:
            rr0, rr1 = r - 1, r + mh + 1
            cc0, cc1 = c - 1, c + mw + 1
            if reserved[rr0:rr1, cc0:cc1].any():
                continue

            patch = grid[r : r + mh, c : c + mw]
            if np.any(patch[mask] != BLACK):
                continue
            patch[mask] = color
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    object_count = pair_count * 2
    side_min = min(30, 14 + pair_count * 3)
    height = random.randint(side_min, 30)
    width = random.randint(side_min, 30)

    colors = random.sample(FOREGROUND_COLORS, object_count)
    used_shape_keys = set()
    items = []

    for pair_index in range(pair_count):
        mode = "rotate" if pair_index == 0 else "flip" if pair_index == 1 else random.choice(
            ("rotate", "flip", "any")
        )

        pair_masks = None
        shape_key = None
        for _ in range(SHAPE_RETRIES):
            mask = _random_connected_mask()
            if mask is None:
                continue
            candidate_key = _canonical_shape_key(mask)
            if candidate_key in used_shape_keys:
                continue
            candidate_pair = _choose_pair_masks(mask, mode)
            if candidate_pair is None:
                continue
            shape_key = candidate_key
            pair_masks = candidate_pair
            break

        if pair_masks is None or shape_key is None:
            return None

        used_shape_keys.add(shape_key)
        color_a = colors[2 * pair_index]
        color_b = colors[2 * pair_index + 1]
        items.append((pair_masks[0], color_a))
        items.append((pair_masks[1], color_b))

    grid = _place_items(height, width, items)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-pair color-swap input")
