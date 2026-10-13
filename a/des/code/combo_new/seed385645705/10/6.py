from collections import deque

import numpy as np


BG = 0
GREY = 5
LOCK_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
LOCK_COUNT_CHOICES = (2, 3)
INTERIOR_HEIGHT_RANGE = (2, 6)  # high-exclusive
INTERIOR_WIDTH_RANGE = (2, 7)   # high-exclusive
NOTCH_LENGTH_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 300
SHAPE_RETRIES = 200
CANVAS_RETRIES = 80
PLACEMENT_RETRIES = 250
OUTER_MARGIN = 1
GAP = 1

_RNG = np.random.default_rng()
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    base = _tight_mask(mask)
    return min((rot.shape, rot.tobytes()) for rot in (np.rot90(base, k) for k in range(4)))


def _find_locks(grid: np.ndarray) -> list[dict]:
    locks = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BG, GREY)):
        for coords in _components(grid == color):
            bbox, _border_mask = _coords_to_mask(coords)
            r0, c0, r1, c1 = bbox
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            hole = sub == BG
            assert hole.any(), "lock must contain a black cavity"
            locks.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "hole": hole,
                    "shape": _tight_mask(hole),
                }
            )
    locks.sort(key=lambda item: item["bbox"])
    return locks


def _find_keys(grid: np.ndarray) -> list[dict]:
    keys = []
    for coords in _components(grid == GREY):
        bbox, mask = _coords_to_mask(coords)
        keys.append({"bbox": bbox, "mask": mask})
    keys.sort(key=lambda item: item["bbox"])
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each grey key and insert it into the matching same-colored lock cavity.

    Input structure:
    - The background is black.
    - There are 2-3 colored rectangular lock frames, each one pixel thick, with a 1-2 cell notch removed from one side.
    - Each lock's bbox contains a black cavity shaped like the frame interior plus the side notch.
    - Elsewhere there are 2-3 disconnected grey key sprites, one for each lock.

    Rule:
    - For each grey key, try all four right-angle rotations and find the unique lock whose black cavity has exactly the same shape.
    - Erase the key from its original position.
    - Fill the matched lock cavity with the rotated key, recolored to the lock's frame color.
    - Leave every existing lock-frame pixel unchanged.

    Key insight:
    - The keys do not move by proximity or color; they match only by exact cavity shape after mental rotation.
    """
    g = np.asarray(grid, dtype=int)
    locks = _find_locks(g)
    keys = _find_keys(g)
    assert len(locks) == len(keys) and len(locks) >= 1

    out = g.copy()
    out[out == GREY] = BG

    used_locks: set[int] = set()
    for key in keys:
        candidates: dict[int, tuple[int, np.ndarray]] = {}
        for rot_k in range(4):
            rotated = np.rot90(key["mask"], rot_k)
            for lock_idx, lock in enumerate(locks):
                if lock_idx in candidates:
                    continue
                if rotated.shape == lock["shape"].shape and np.array_equal(rotated, lock["shape"]):
                    candidates[lock_idx] = (rot_k, rotated)

        assert len(candidates) == 1, "each key must match exactly one lock"
        lock_idx, (_rot_k, _rotated_mask) = next(iter(candidates.items()))
        assert lock_idx not in used_locks, "lock matching must be one-to-one"
        used_locks.add(lock_idx)

        lock = locks[lock_idx]
        r0, c0, r1, c1 = lock["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[lock["hole"]] = int(lock["color"])

    return out


def _sample_hole_mask(rng: np.random.Generator) -> np.ndarray:
    ih = int(rng.integers(*INTERIOR_HEIGHT_RANGE))
    iw = int(rng.integers(*INTERIOR_WIDTH_RANGE))
    side = int(rng.integers(0, 4))
    gap_len = int(rng.choice(NOTCH_LENGTH_CHOICES))

    hole = np.zeros((ih + 2, iw + 2), dtype=bool)
    hole[1:-1, 1:-1] = True
    if side == 0:
        start = int(rng.integers(1, iw - gap_len + 2))
        hole[0, start : start + gap_len] = True
    elif side == 1:
        start = int(rng.integers(1, ih - gap_len + 2))
        hole[start : start + gap_len, -1] = True
    elif side == 2:
        start = int(rng.integers(1, iw - gap_len + 2))
        hole[-1, start : start + gap_len] = True
    else:
        start = int(rng.integers(1, ih - gap_len + 2))
        hole[start : start + gap_len, 0] = True
    return hole


def _sample_unique_holes(rng: np.random.Generator, count: int) -> list[np.ndarray] | None:
    holes = []
    seen = set()
    for _ in range(SHAPE_RETRIES):
        if len(holes) == count:
            return holes
        hole = _sample_hole_mask(rng)
        key = _rotation_key(hole)
        if key in seen:
            continue
        seen.add(key)
        holes.append(hole)
    return None


def _sample_canvas_shape(rng: np.random.Generator, dims: list[tuple[int, int]]) -> tuple[int, int] | None:
    total_area = sum(h * w for h, w in dims)
    min_h = max(h for h, _w in dims) + 2 * OUTER_MARGIN
    min_w = max(w for _h, w in dims) + 2 * OUTER_MARGIN
    target_area = total_area * 2 + 40

    for _ in range(CANVAS_RETRIES):
        aspect = float(rng.uniform(0.8, 1.25))
        h = max(min_h, int(np.ceil(np.sqrt(target_area * aspect))))
        w = max(min_w, int(np.ceil(target_area / h)))
        h += int(rng.integers(0, 3))
        w += int(rng.integers(0, 3))
        if rng.random() < 0.5:
            h, w = w, h
        if h <= 30 and w <= 30:
            return int(h), int(w)
    return None


def _place_bboxes(
    rng: np.random.Generator, dims: list[tuple[int, int]]
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    for _ in range(CANVAS_RETRIES):
        canvas_shape = _sample_canvas_shape(rng, dims)
        if canvas_shape is None:
            continue
        h, w = canvas_shape
        occupied = np.zeros((h, w), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(dims)
        order = sorted(range(len(dims)), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)

        ok = True
        for idx in order:
            oh, ow = dims[idx]
            placed = False
            for _ in range(PLACEMENT_RETRIES):
                top = int(rng.integers(OUTER_MARGIN, h - oh - OUTER_MARGIN + 1))
                left = int(rng.integers(OUTER_MARGIN, w - ow - OUTER_MARGIN + 1))
                r0 = max(0, top - GAP)
                c0 = max(0, left - GAP)
                r1 = min(h, top + oh + GAP)
                c1 = min(w, left + ow + GAP)
                if occupied[r0:r1, c0:c1].any():
                    continue
                positions[idx] = (top, left)
                occupied[top : top + oh, left : left + ow] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return (h, w), [pos for pos in positions if pos is not None]
    return None


def _render_lock(grid: np.ndarray, hole: np.ndarray, color: int, top: int, left: int) -> None:
    patch = np.full(hole.shape, int(color), dtype=int)
    patch[hole] = BG
    grid[top : top + hole.shape[0], left : left + hole.shape[1]] = patch


def _render_key(grid: np.ndarray, key_mask: np.ndarray, top: int, left: int) -> None:
    patch = grid[top : top + key_mask.shape[0], left : left + key_mask.shape[1]]
    patch[key_mask] = GREY


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    lock_count = int(rng.choice(LOCK_COUNT_CHOICES))
    holes = _sample_unique_holes(rng, lock_count)
    if holes is None:
        return None

    colors = [int(c) for c in rng.choice(LOCK_COLOR_POOL, size=lock_count, replace=False)]
    rotations = [int(rng.integers(0, 4)) for _ in range(lock_count)]
    key_shapes = [_tight_mask(hole) for hole in holes]
    key_masks = [np.rot90(shape, rot) for shape, rot in zip(key_shapes, rotations)]

    dims = [hole.shape for hole in holes] + [mask.shape for mask in key_masks]
    placed = _place_bboxes(rng, dims)
    if placed is None:
        return None

    (h, w), positions = placed
    assert len(positions) == len(dims)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)

    lock_positions = positions[:lock_count]
    key_positions = positions[lock_count:]

    for hole, color, (top, left) in zip(holes, colors, lock_positions):
        _render_lock(grid, hole, int(color), int(top), int(left))
        expected[top : top + hole.shape[0], left : left + hole.shape[1]] = int(color)

    for key_mask, (top, left) in zip(key_masks, key_positions):
        _render_key(grid, key_mask, int(top), int(left))

    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
