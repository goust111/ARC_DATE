import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
REGION_COUNT_CHOICES = (1, 2, 3, 4)
PLACEMENT_RETRIES = 120

HEIGHT_BY_COUNT = {
    1: (6, 11),
    2: (5, 10),
    3: (4, 9),
    4: (4, 8),
}
WIDTH_BY_COUNT = {
    1: (7, 13),
    2: (6, 11),
    3: (5, 10),
    4: (5, 9),
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _extract_enclosed_regions(grid: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[tuple[int, int, int, int, np.ndarray]] = []

    for coords in _components(g != GREY):
        if _touches_border(coords, h, w):
            continue
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        regions.append((r0, c0, r1, c1, mask))

    regions.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return regions


def _prefix_true_length(flags: np.ndarray) -> int:
    k = 0
    while k < len(flags) and bool(flags[k]):
        k += 1
    return k


def _extract_seed(region: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    color_mask = region_mask & (region != BLACK)
    rows_with_color = np.any(color_mask, axis=1)
    cols_with_color = np.any(color_mask, axis=0)

    seed_h = _prefix_true_length(rows_with_color)
    seed_w = _prefix_true_length(cols_with_color)
    assert seed_h > 0 and seed_w > 0, "each enclosed region must have a top-left colored seed"
    assert not np.any(rows_with_color[seed_h:]), "seed colors must be confined to a top prefix of rows"
    assert not np.any(cols_with_color[seed_w:]), "seed colors must be confined to a left prefix of cols"

    seed_mask = region_mask[:seed_h, :seed_w]
    seed = region[:seed_h, :seed_w]
    assert np.all(seed_mask), "seed rectangle must lie completely inside the enclosed region"
    assert np.all(seed != BLACK), "seed rectangle must be fully colored"

    stray = color_mask.copy()
    stray[:seed_h, :seed_w] = False
    assert not np.any(stray), "only the seed rectangle may contain colors in the input"
    return seed


def _tile_seed(seed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    reps_r = (shape[0] + seed.shape[0] - 1) // seed.shape[0]
    reps_c = (shape[1] + seed.shape[1] - 1) // seed.shape[1]
    return np.tile(seed, (reps_r, reps_c))[: shape[0], : shape[1]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region by periodically tiling its top-left seed rectangle.

    The grid uses black as background and grey as the outline color. Every enclosed
    non-grey component that does not touch the outer border is treated as one region.
    The region's seed is the maximal top-left colored rectangle before the first
    fully black row and fully black column. That seed is tiled over the region interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for r0, c0, r1, c1, region_mask in _extract_enclosed_regions(g):
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        seed = _extract_seed(region, region_mask)
        tiled = _tile_seed(seed, region.shape)
        out_region = out[r0 : r1 + 1, c0 : c1 + 1]
        out_region[region_mask] = tiled[region_mask]

    return out


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _make_seed(rng: random.Random, seed_h: int, seed_w: int) -> np.ndarray:
    n_colors = rng.randint(1, min(3, seed_h * seed_w, len(PALETTE)))
    colors = rng.sample(PALETTE, n_colors)
    seed = np.array(
        [[rng.choice(colors) for _ in range(seed_w)] for _ in range(seed_h)],
        dtype=int,
    )
    if n_colors > 1 and np.unique(seed).size == 1:
        rr = rng.randrange(seed_h)
        cc = rng.randrange(seed_w)
        alternatives = [color for color in colors if color != int(seed[rr, cc])]
        seed[rr, cc] = rng.choice(alternatives)
    return seed


def _make_region_mask(rng: random.Random, n_regions: int) -> tuple[np.ndarray, np.ndarray]:
    h_min, h_max = HEIGHT_BY_COUNT[n_regions]
    w_min, w_max = WIDTH_BY_COUNT[n_regions]
    inner_h = rng.randint(h_min, h_max)
    max_w = rng.randint(w_min, w_max)

    seed_h = rng.randint(1, min(4, inner_h - 1))
    seed_w = rng.randint(1, min(4, max_w - 1))

    rectangular = rng.random() < 0.3
    lefts = [0] * inner_h
    rights = [0] * inner_h

    if rectangular:
        width = rng.randint(seed_w + 1, max_w)
        for r in range(inner_h):
            rights[r] = width - 1
    else:
        rights[0] = rng.randint(seed_w, max_w - 1)
        for r in range(1, inner_h):
            min_right = seed_w if r < seed_h else 1
            low = max(min_right, rights[r - 1] - 2)
            high = min(max_w - 1, rights[r - 1] + 2)
            if low > high:
                low = high = max(min_right, min(max_w - 1, rights[r - 1]))
            rights[r] = rng.randint(low, high)

        if inner_h - seed_h >= 2 and rng.random() < 0.65:
            notch_start = rng.randint(seed_h, inner_h - 1)
            max_depth = min(2, min(rights[notch_start:]))
            if max_depth >= 1:
                depth = rng.randint(1, max_depth)
                for r in range(notch_start, inner_h):
                    lefts[r] = min(depth, rights[r] - 1)

    inner_w = max(rights) + 1
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r in range(inner_h):
        mask[r, lefts[r] : rights[r] + 1] = True

    assert np.all(mask[:seed_h, :seed_w]), "seed must fit in the shape's top-left corner"
    assert mask.shape[0] > seed_h and mask.shape[1] > seed_w, "seed should be smaller than the region"
    assert mask.sum() > seed_h * seed_w, "region must extend beyond its seed"

    seed = _make_seed(rng, seed_h, seed_w)
    return mask, seed


def _build_region_patch(mask: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner_h, inner_w = mask.shape
    patch_shape = (inner_h + 2, inner_w + 2)

    shifted_mask = np.zeros(patch_shape, dtype=bool)
    shifted_mask[1 : inner_h + 1, 1 : inner_w + 1] = mask
    wall_mask = _dilate8(shifted_mask) & ~shifted_mask
    assert len(_components(wall_mask)) == 1, "outline must be one connected grey loop"

    inp = np.zeros(patch_shape, dtype=int)
    inp[wall_mask] = GREY
    inp[1 : 1 + seed.shape[0], 1 : 1 + seed.shape[1]] = seed

    out = np.zeros(patch_shape, dtype=int)
    out[wall_mask] = GREY
    tiled = _tile_seed(seed, mask.shape)
    out_region = out[1 : inner_h + 1, 1 : inner_w + 1]
    out_region[mask] = tiled[mask]

    interior_components = [
        coords
        for coords in _components(inp != GREY)
        if not _touches_border(coords, inp.shape[0], inp.shape[1])
    ]
    assert len(interior_components) == 1, "patch must contain exactly one enclosed region"

    return inp, out


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
) -> bool:
    ph, pw = patch.shape
    h, w = grid.shape
    if top < 1 or left < 1 or top + ph >= h or left + pw >= w:
        return False

    r0 = top - 1
    c0 = left - 1
    r1 = top + ph + 1
    c1 = left + pw + 1
    if reserved[r0:r1, c0:c1].any():
        return False

    grid[top : top + ph, left : left + pw] = patch
    reserved[r0:r1, c0:c1] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_regions = rng.choice(REGION_COUNT_CHOICES)

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        for _ in range(n_regions):
            mask, seed = _make_region_mask(rng, n_regions)
            patches.append(_build_region_patch(mask, seed))
    except AssertionError:
        return None

    rng.shuffle(patches)
    for inp_patch, out_patch in patches:
        ph, pw = inp_patch.shape
        if ph + 2 >= h or pw + 2 >= w:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, h - ph - 1)
            left = rng.randint(1, w - pw - 1)
            if not _place_patch(grid, reserved, inp_patch, top, left):
                continue
            expected[top : top + ph, left : left + pw] = out_patch
            placed = True
            break
        if not placed:
            return None

    if len(_extract_enclosed_regions(grid)) != n_regions:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-seed tiling input")
