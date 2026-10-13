import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
OUTLINE_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 300
GRID_MIN_SIZE = 18
GRID_MAX_SIZE = 30

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BASE_SHAPES = {
    "square": np.array([[1]], dtype=bool),
    "L": np.array([[1, 0], [1, 1]], dtype=bool),
    "T": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "S": np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
}

SCALE_RANGES = {
    "square": (2, 4),
    "L": (1, 3),
    "T": (1, 3),
    "S": (1, 3),
    "plus": (1, 2),
}


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _compress_axis(mask: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        keep = [0]
        for idx in range(1, mask.shape[0]):
            if not np.array_equal(mask[idx], mask[idx - 1]):
                keep.append(idx)
        return mask[keep, :]

    keep = [0]
    for idx in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, idx], mask[:, idx - 1]):
            keep.append(idx)
    return mask[:, keep]


def _reduce_mask(mask: np.ndarray) -> np.ndarray:
    arr = _crop_mask(mask.astype(bool)).astype(np.uint8)
    while True:
        prev = arr.copy()
        arr = _compress_axis(arr, axis=0)
        arr = _compress_axis(arr, axis=1)
        if arr.shape == prev.shape and np.array_equal(arr, prev):
            return arr.astype(bool)


def _mask_key(mask: np.ndarray) -> str:
    arr = mask.astype(np.uint8)
    bits = "".join("1" if value else "0" for value in arr.ravel())
    return f"{arr.shape[0]}x{arr.shape[1]}:{bits}"


def _shape_signature(mask: np.ndarray) -> str:
    base = _reduce_mask(mask)
    keys = []
    for flip in (False, True):
        transformed = np.fliplr(base) if flip else base
        for rot in range(4):
            keys.append(_mask_key(_reduce_mask(np.rot90(transformed, rot))))
    return min(keys)


def _extract_enclosed_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    walkable = g != OUTLINE_COLOR
    seen = np.zeros((h, w), dtype=bool)
    regions = []

    for r in range(h):
        for c in range(w):
            if not walkable[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_boundary = False

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_boundary = True
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            if touches_boundary:
                continue

            pts = np.array(coords, dtype=int)
            region_mask = np.zeros(
                (int(pts[:, 0].max() - pts[:, 0].min() + 1), int(pts[:, 1].max() - pts[:, 1].min() + 1)),
                dtype=bool,
            )
            region_mask[pts[:, 0] - pts[:, 0].min(), pts[:, 1] - pts[:, 1].min()] = True

            colors = {int(g[rr, cc]) for rr, cc in coords}
            colors.discard(BG_COLOR)
            assert len(colors) <= 1

            regions.append(
                {
                    "coords": pts,
                    "seed_color": next(iter(colors)) if colors else None,
                    "signature": _shape_signature(region_mask),
                }
            )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each enclosed region up to scale/rotation/reflection, inherit its
    seed color from the matching shape class, and fill the whole region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions = _extract_enclosed_regions(g)
    assert regions

    shape_to_color = {}
    for region in regions:
        color = region["seed_color"]
        if color is None:
            continue
        signature = region["signature"]
        if signature in shape_to_color:
            assert shape_to_color[signature] == color
        else:
            shape_to_color[signature] = color

    out = g.copy()
    for region in regions:
        signature = region["signature"]
        assert signature in shape_to_color
        rows = region["coords"][:, 0]
        cols = region["coords"][:, 1]
        out[rows, cols] = shape_to_color[signature]

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask.astype(bool), 1)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _transform_mask(mask: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    transformed = np.fliplr(mask) if flip else mask.copy()
    return np.rot90(transformed, rot)


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)


def _render_patch(mask: np.ndarray, seed_color: int | None, rng: random.Random) -> np.ndarray:
    interior = np.pad(mask.astype(bool), 1)
    outline = _dilate8(interior) & ~interior
    patch = np.zeros(interior.shape, dtype=int)
    patch[outline] = OUTLINE_COLOR
    if seed_color is not None:
        cells = np.argwhere(interior)
        rr, cc = cells[rng.randrange(len(cells))]
        patch[rr, cc] = int(seed_color)
    return patch


def _class_instance_count(rng: random.Random, n_classes: int) -> list[int]:
    counts = [2] * n_classes
    extra_budget = 6 - 2 * n_classes
    while extra_budget > 0 and rng.random() < 0.7:
        idx = rng.randrange(n_classes)
        if counts[idx] < 3:
            counts[idx] += 1
            extra_budget -= 1
    return counts


def _build_instances(rng: random.Random) -> tuple[list[dict], Counter, Counter]:
    n_classes = 2 if rng.random() < 0.55 else 3
    shape_names = rng.sample(list(BASE_SHAPES), n_classes)
    seed_colors = rng.sample(SEED_COLORS, n_classes)
    class_counts = _class_instance_count(rng, n_classes)

    instances = []
    expected_counts = Counter()
    expected_seed_counts = Counter()

    for shape_name, seed_color, count in zip(shape_names, seed_colors, class_counts):
        base = BASE_SHAPES[shape_name]
        signature = _shape_signature(base)
        expected_counts[signature] = count
        expected_seed_counts[signature] = 1

        scale_lo, scale_hi = SCALE_RANGES[shape_name]
        scales = [rng.randint(scale_lo, scale_hi) for _ in range(count)]
        if count > 1 and scale_hi > scale_lo and len(set(scales)) == 1:
            alt = scale_lo if scales[0] != scale_lo else scale_hi
            scales[rng.randrange(count)] = alt

        seeded_index = rng.randrange(count)
        for idx in range(count):
            transformed = _transform_mask(base, rot=rng.randrange(4), flip=bool(rng.randrange(2)))
            scaled = _scale_mask(transformed, scales[idx])
            patch = _render_patch(scaled, seed_color if idx == seeded_index else None, rng)
            instances.append({"patch": patch, "signature": signature})

    rng.shuffle(instances)
    return instances, expected_counts, expected_seed_counts


def _place_instances(instances: list[dict], rng: random.Random) -> np.ndarray | None:
    max_h = max(item["patch"].shape[0] for item in instances)
    max_w = max(item["patch"].shape[1] for item in instances)
    total_area = sum(int(item["patch"].shape[0] * item["patch"].shape[1]) for item in instances)

    for _ in range(80):
        min_h = max(GRID_MIN_SIZE, max_h + 2)
        min_w = max(GRID_MIN_SIZE, max_w + 2)
        if min_h > GRID_MAX_SIZE or min_w > GRID_MAX_SIZE:
            return None

        h = rng.randint(min_h, GRID_MAX_SIZE)
        w = rng.randint(min_w, GRID_MAX_SIZE)
        if h * w < total_area * 2:
            continue

        grid = np.zeros((h, w), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        ok = True

        for item in sorted(instances, key=lambda x: x["patch"].shape[0] * x["patch"].shape[1], reverse=True):
            patch = item["patch"]
            ph, pw = patch.shape
            if h - ph - 1 < 1 or w - pw - 1 < 1:
                ok = False
                break

            placed = False
            for _ in range(250):
                top = rng.randint(1, h - ph - 1)
                left = rng.randint(1, w - pw - 1)
                if reserved[max(0, top - 1) : min(h, top + ph + 1), max(0, left - 1) : min(w, left + pw + 1)].any():
                    continue

                window = grid[top : top + ph, left : left + pw]
                window[patch != BG_COLOR] = patch[patch != BG_COLOR]
                reserved[top : top + ph, left : left + pw] = True
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok:
            return grid

    return None


def _valid_generated_grid(grid: np.ndarray, expected_counts: Counter, expected_seed_counts: Counter) -> bool:
    try:
        regions = _extract_enclosed_regions(grid)
        if Counter(region["signature"] for region in regions) != expected_counts:
            return False
        if Counter(region["signature"] for region in regions if region["seed_color"] is not None) != expected_seed_counts:
            return False
        _ = main(grid)
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    instances, expected_counts, expected_seed_counts = _build_instances(rng)
    grid = _place_instances(instances, rng)
    if grid is None:
        return None
    if not _valid_generated_grid(grid, expected_counts, expected_seed_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
