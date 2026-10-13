from __future__ import annotations

import random

import numpy as np


TAMPER_COLOR = 6
NONBLACK_PALETTE = (1, 2, 3, 4, 7, 8, 9)
TILE_SIZES = ((3, 3), (3, 4), (4, 3), (4, 4))
REPEAT_RANGE = (3, 6)
MAX_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bounding box")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % int(d) == 0]


def _fits_period(grid: np.ndarray, tile_h: int, tile_w: int) -> bool:
    seed = grid[:tile_h, :tile_w]
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) == TAMPER_COLOR:
                continue
            if int(grid[r, c]) != int(seed[r % tile_h, c % tile_w]):
                return False
    return True


def _find_seed_shape(grid: np.ndarray) -> tuple[int, int]:
    H, W = grid.shape
    for tile_h in _divisors(H):
        for tile_w in _divisors(W):
            if _fits_period(grid, tile_h, tile_w):
                return int(tile_h), int(tile_w)
    raise ValueError("no periodic seed matches the visible cells")


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the correct crop of a magenta-damaged tiled region from the intact top-left seed.

    Input structure:
    - The input is a rectangular color grid built by repeating one small seed tile.
    - The intact seed tile is exposed at the upper-left corner, and the grid size is an integer multiple of the seed size.
    - One solid rectangular region away from the upper-left seed has been overwritten with magenta (6).

    Rule:
    - Find the smallest tile height and width whose upper-left crop is consistent with every non-magenta cell under periodic repetition.
    - Use that upper-left tile as the seed and tile it back to the full grid size to reconstruct the intended reference grid.
    - Locate the magenta rectangle and return the reference grid cropped to exactly that rectangle.

    Key insight:
    - The magenta area is not a pattern element; it is a missing rectangle whose correct contents are determined entirely by the grid's surviving periodic structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    tamper = g == TAMPER_COLOR
    assert tamper.any(), "expected a tampered region"
    r0, c0, r1, c1 = _bbox(tamper)
    assert bool(np.all(tamper[r0 : r1 + 1, c0 : c1 + 1])), "tampered cells must form a solid rectangle"

    tile_h, tile_w = _find_seed_shape(g)
    seed = g[:tile_h, :tile_w]
    reference = np.tile(seed, (g.shape[0] // tile_h, g.shape[1] // tile_w))
    return reference[r0 : r1 + 1, c0 : c1 + 1].copy()


def _seed_has_full_period(seed: np.ndarray) -> bool:
    h, w = seed.shape
    for cand_h in _divisors(h):
        for cand_w in _divisors(w):
            if cand_h == h and cand_w == w:
                continue
            if h % cand_h == 0 and w % cand_w == 0:
                test = np.tile(seed[:cand_h, :cand_w], (h // cand_h, w // cand_w))
                if np.array_equal(seed, test):
                    return False
    return True


def _sample_seed(tile_h: int, tile_w: int) -> np.ndarray | None:
    include_black = random.random() < 0.55
    nonblack_count = random.choice((2, 3))
    palette = list(random.sample(NONBLACK_PALETTE, nonblack_count))
    choices = palette + ([0] if include_black else [])

    seed = np.array(
        [[random.choice(choices) for _ in range(tile_w)] for _ in range(tile_h)],
        dtype=int,
    )
    if TAMPER_COLOR in seed:
        return None
    if len(set(int(v) for v in seed.flatten() if int(v) != 0)) < 2:
        return None
    if include_black and not np.any(seed == 0):
        return None
    if np.unique(seed).size < 3:
        return None
    if max(np.count_nonzero(seed == color) for color in np.unique(seed)) > seed.size - 2:
        return None
    if len({tuple(map(int, row)) for row in seed}) < 2:
        return None
    if len({tuple(map(int, col)) for col in seed.T}) < 2:
        return None
    if not _seed_has_full_period(seed):
        return None
    return seed


def _sample_damage_bbox(
    reference: np.ndarray,
    tile_h: int,
    tile_w: int,
) -> tuple[int, int, int, int] | None:
    H, W = reference.shape
    prefer_crossing = random.random() < 0.7
    require_black = random.random() < 0.35

    h_min = 2
    h_max = min(6, H - 1)
    w_min = 2
    w_max = min(6, W - 1)
    if h_min > h_max or w_min > w_max:
        return None

    for _ in range(250):
        damage_h = random.randint(h_min, h_max)
        damage_w = random.randint(w_min, w_max)
        r0 = random.randint(0, H - damage_h)
        c0 = random.randint(0, W - damage_w)
        r1 = r0 + damage_h - 1
        c1 = c0 + damage_w - 1

        if r0 < tile_h and c0 < tile_w:
            continue

        row_phases = {(r0 + dr) % tile_h for dr in range(damage_h)}
        col_phases = {(c0 + dc) % tile_w for dc in range(damage_w)}
        if len(row_phases) < 2 or len(col_phases) < 2:
            continue

        aligned_full_tile = (
            damage_h == tile_h
            and damage_w == tile_w
            and r0 % tile_h == 0
            and c0 % tile_w == 0
        )
        if aligned_full_tile:
            continue

        crosses_boundary = (r0 % tile_h) + damage_h > tile_h or (c0 % tile_w) + damage_w > tile_w
        if prefer_crossing and not crosses_boundary:
            continue

        crop = reference[r0 : r1 + 1, c0 : c1 + 1]
        if np.unique(crop).size < 2:
            continue
        if require_black and not np.any(crop == 0):
            continue
        nonblack_ratio = np.count_nonzero(crop != 0) / crop.size
        if not (0.25 <= nonblack_ratio <= 1.0):
            continue
        return r0, c0, r1, c1
    return None


def _visible_min_period(grid: np.ndarray) -> tuple[int, int]:
    return _find_seed_shape(grid)


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    tile_h, tile_w = random.choice(TILE_SIZES)
    rep_r = random.randint(REPEAT_RANGE[0], min(REPEAT_RANGE[1], 30 // tile_h))
    rep_c = random.randint(REPEAT_RANGE[0], min(REPEAT_RANGE[1], 30 // tile_w))

    seed = _sample_seed(tile_h, tile_w)
    if seed is None:
        return None

    reference = np.tile(seed, (rep_r, rep_c))
    bbox = _sample_damage_bbox(reference, tile_h, tile_w)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox

    grid = reference.copy()
    grid[r0 : r1 + 1, c0 : c1 + 1] = TAMPER_COLOR
    expected = reference[r0 : r1 + 1, c0 : c1 + 1].copy()

    if _visible_min_period(grid) != (tile_h, tile_w):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    repaired = grid.copy()
    repaired[r0 : r1 + 1, c0 : c1 + 1] = expected
    if not np.array_equal(repaired, reference):
        return None
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TAMPER_COLOR):
        return None
    if np.any(grid[:tile_h, :tile_w] == TAMPER_COLOR):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid tampered-tiling input")
