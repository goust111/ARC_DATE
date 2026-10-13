import random
from collections import deque

import numpy as np


BG_COLOR = 0
GROW_COLOR = 5
SEED_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 300
SEED_RETRIES = 200
LAYOUT_RETRIES = 120


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
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_seed(grid: np.ndarray) -> np.ndarray:
    mask = (grid != BG_COLOR) & (grid != GROW_COLOR)
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _symmetry_flags(seed: np.ndarray) -> dict[str, bool]:
    rot4 = seed.shape[0] == seed.shape[1] and all(
        np.array_equal(seed, np.rot90(seed, k)) for k in (1, 2, 3)
    )
    return {
        "rot4": rot4,
        "rot180": np.array_equal(seed, np.rot90(seed, 2)),
        "lr": np.array_equal(seed, np.fliplr(seed)),
        "ud": np.array_equal(seed, np.flipud(seed)),
    }


def _classify_seed(seed: np.ndarray) -> str:
    flags = _symmetry_flags(seed)
    if flags["rot4"]:
        return "rot4"
    if flags["lr"] and not flags["ud"]:
        return "lr"
    if flags["ud"] and not flags["lr"]:
        return "ud"
    return "none"


def _growth_tile(seed: np.ndarray) -> np.ndarray:
    branch = _classify_seed(seed)
    if branch == "rot4":
        return np.block(
            [
                [seed, np.rot90(seed, -1)],
                [np.rot90(seed, 1), np.rot90(seed, 2)],
            ]
        )
    if branch == "lr":
        return np.vstack([seed, np.flipud(seed)])
    if branch == "ud":
        return np.hstack([seed, np.fliplr(seed)])
    return seed.copy()


def _tile_to_shape(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    reps_r = (h + th - 1) // th
    reps_c = (w + tw - 1) // tw
    tiled = np.tile(tile, (reps_r, reps_c))
    return tiled[:h, :w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a colored seed into each grey rectangle using a tiling rule chosen by the seed's symmetry.

    Input structure:
    - Background is black.
    - One multicolor seed pattern appears away from every grey rectangle.
    - One or more grey rectangular growth regions appear elsewhere.

    Rule:
    - 4-fold rotational seed -> tile a 2x2 unit made from the seed and its 90/180/270 degree rotations.
    - Left-right symmetric seed only -> tile a 2x1 unit made from the seed over its vertical flip.
    - Up-down symmetric seed only -> tile a 1x2 unit made from the seed beside its horizontal flip.
    - Otherwise -> directly repeat the seed.
    - Each region is tiled from its top-left corner; the original seed stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    seed = _extract_seed(g)
    tile = _growth_tile(seed)

    for comp in _components(g == GROW_COLOR):
        coords = np.asarray(comp, dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        patch = _tile_to_shape(tile, (r1 - r0 + 1, c1 - c0 + 1))
        for r, c in comp:
            out[r, c] = int(patch[r - r0, c - c0])

    return out


def _random_palette(rng: random.Random) -> list[int]:
    count = rng.randint(2, 4)
    return rng.sample(SEED_COLOR_POOL, count)


def _random_pattern(shape: tuple[int, int], palette: list[int], rng: random.Random) -> np.ndarray:
    h, w = shape
    arr = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            arr[r, c] = rng.choice(palette)
    if len(set(map(int, arr.flat))) == 1:
        rr = rng.randrange(h)
        cc = rng.randrange(w)
        choices = [color for color in palette if color != int(arr[rr, cc])]
        if choices:
            arr[rr, cc] = rng.choice(choices)
    return arr


def _seed_rot4(rng: random.Random) -> np.ndarray:
    size = rng.randint(3, 5)
    palette = _random_palette(rng)
    arr = np.empty((size, size), dtype=int)
    done: set[tuple[int, int]] = set()
    for r in range(size):
        for c in range(size):
            if (r, c) in done:
                continue
            orbit = {
                (r, c),
                (c, size - 1 - r),
                (size - 1 - r, size - 1 - c),
                (size - 1 - c, r),
            }
            color = rng.choice(palette)
            for rr, cc in orbit:
                arr[rr, cc] = color
                done.add((rr, cc))
    return arr


def _seed_lr(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    palette = _random_palette(rng)
    left = _random_pattern((h, (w + 1) // 2), palette, rng)
    return np.concatenate([left, np.fliplr(left[:, : w // 2])], axis=1)


def _seed_ud(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    palette = _random_palette(rng)
    top = _random_pattern(((h + 1) // 2, w), palette, rng)
    return np.concatenate([top, np.flipud(top[: h // 2, :])], axis=0)


def _seed_none(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    palette = _random_palette(rng)
    return _random_pattern((h, w), palette, rng)


def _generate_seed(branch: str, rng: random.Random) -> np.ndarray:
    makers = {
        "rot4": _seed_rot4,
        "lr": _seed_lr,
        "ud": _seed_ud,
        "none": _seed_none,
    }
    for _ in range(SEED_RETRIES):
        seed = makers[branch](rng)
        flags = _symmetry_flags(seed)
        if len(set(map(int, seed.flat))) < 2:
            continue
        if branch == "rot4" and flags["rot4"]:
            return seed
        if branch == "lr" and _classify_seed(seed) == "lr":
            return seed
        if branch == "ud" and _classify_seed(seed) == "ud":
            return seed
        if branch == "none" and not any(flags.values()):
            return seed
    raise RuntimeError(f"failed to generate {branch} seed")


def _choose_regions(seed_shape: tuple[int, int], branch: str, rng: random.Random) -> list[tuple[int, int]]:
    sh, sw = seed_shape
    max_mult = 3 if max(seed_shape) <= 4 else 2
    pairs = [(a, b) for a in range(1, max_mult + 1) for b in range(1, max_mult + 1)]
    wanted = rng.choice([2, 3]) if len(pairs) >= 3 else 2
    for _ in range(80):
        pool = pairs[:]
        rng.shuffle(pool)
        picked: list[tuple[int, int]] = []
        seen_dims: set[tuple[int, int]] = set()
        for mult_r, mult_c in pool:
            dims = (sh * mult_r, sw * mult_c)
            if dims in seen_dims:
                continue
            picked.append(dims)
            seen_dims.add(dims)
            if len(picked) == wanted:
                break
        if len(picked) != wanted:
            continue
        if branch == "lr" and not any(h > sh for h, _ in picked):
            continue
        if branch == "ud" and not any(w > sw for _, w in picked):
            continue
        if branch in {"rot4", "none"} and not any(h > sh or w > sw for h, w in picked):
            continue
        return picked
    raise RuntimeError("failed to choose growth regions")


def _pack_boxes(sizes: list[tuple[int, int]], rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    order = list(range(len(sizes)))
    rng.shuffle(order)
    max_w = max(w for _, w in sizes)
    limit_lo = max_w + 2
    limit_hi = max(limit_lo, min(32, max_w * 2 + 4))
    row_limit = rng.randint(limit_lo, limit_hi)

    positions = [(-1, -1)] * len(sizes)
    x = 0
    y = 0
    row_h = 0
    content_w = 0

    for idx in order:
        h, w = sizes[idx]
        gap = 0 if x == 0 else rng.randint(2, 4)
        if x > 0 and x + gap + w > row_limit:
            y += row_h + rng.randint(2, 4)
            x = 0
            row_h = 0
            gap = 0
        col = x + gap
        positions[idx] = (y, col)
        x = col + w
        row_h = max(row_h, h)
        content_w = max(content_w, x)

    content_h = y + row_h
    top = rng.randint(1, 3)
    left = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    right = rng.randint(1, 3)
    shifted = [(r + top, c + left) for r, c in positions]
    return shifted, content_h + top + bottom, content_w + left + right


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    branch = rng.choice(["rot4", "lr", "ud", "none"])
    seed = _generate_seed(branch, rng)
    regions = _choose_regions(seed.shape, branch, rng)
    boxes = [seed.shape] + regions

    for _ in range(LAYOUT_RETRIES):
        positions, h, w = _pack_boxes(boxes, rng)
        if h > 36 or w > 36:
            continue

        grid = np.zeros((h, w), dtype=int)
        sr, sc = positions[0]
        sh, sw = seed.shape
        grid[sr : sr + sh, sc : sc + sw] = seed

        for (rr, cc), (rh, rw) in zip(positions[1:], regions, strict=False):
            grid[rr : rr + rh, cc : cc + rw] = GROW_COLOR

        if rng.random() < 0.5:
            grid = np.fliplr(grid)
        if rng.random() < 0.5:
            grid = np.flipud(grid)

        out = main(grid)
        if np.any(out == GROW_COLOR):
            continue
        seed_mask = (grid != BG_COLOR) & (grid != GROW_COLOR)
        if not np.array_equal(grid[seed_mask], out[seed_mask]):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
