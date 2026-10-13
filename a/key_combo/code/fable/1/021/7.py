from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color


MAX_N = 5
EXACT_SEARCH_N = 4
GEN_RETRY_LIMIT = 400
COLOR_POOL = tuple(c for c in Color.NOT_BLACK if c != Color.BLACK)
_BITCOUNT_CACHE: dict[int, np.ndarray] = {}
_BIT_WEIGHTS_CACHE: dict[int, np.ndarray] = {}


def _bitcounts(bit_count: int) -> np.ndarray:
    table = _BITCOUNT_CACHE.get(bit_count)
    if table is None:
        size = 1 << bit_count
        table = np.fromiter((mask.bit_count() for mask in range(size)), dtype=np.uint8, count=size)
        _BITCOUNT_CACHE[bit_count] = table
    return table


def _bit_weights(bit_count: int) -> np.ndarray:
    weights = _BIT_WEIGHTS_CACHE.get(bit_count)
    if weights is None:
        weights = (1 << np.arange(bit_count, dtype=np.uint32)).astype(np.uint32)
        _BIT_WEIGHTS_CACHE[bit_count] = weights
    return weights


def _infer_n(shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    assert h == w, "expected a square grid"
    n = math.isqrt(h)
    assert 2 <= n <= MAX_N and n * n == h, "grid side must be n^2 with 2 <= n <= 5"
    return n


def _mask_to_array(mask: int, n: int) -> np.ndarray:
    bit_count = n * n
    bits = ((int(mask) >> np.arange(bit_count)) & 1).astype(int)
    return bits.reshape(n, n)


def _array_to_mask(arr: np.ndarray) -> int:
    flat = np.asarray(arr, dtype=np.uint32).reshape(-1)
    return int(flat.dot(_bit_weights(flat.size)))


def _grid_to_block_masks(grid: np.ndarray, fg: int, n: int) -> np.ndarray:
    bit_count = n * n
    weights = _bit_weights(bit_count)
    masks = np.zeros(bit_count, dtype=np.uint32)
    idx = 0
    for br in range(n):
        for bc in range(n):
            block = (grid[br * n : (br + 1) * n, bc * n : (bc + 1) * n] == fg).astype(np.uint32)
            masks[idx] = np.uint32(block.reshape(-1).dot(weights))
            idx += 1
    return masks


def _cost_of_mask(mask: int, block_masks: np.ndarray, popcounts: np.ndarray, base_noise: int) -> int:
    area = int(popcounts[mask])
    cost = int(base_noise) + area * area
    remaining = int(mask)
    while remaining:
        lsb = remaining & -remaining
        idx = lsb.bit_length() - 1
        cost -= 2 * int(popcounts[int(block_masks[idx]) & int(mask)])
        remaining ^= lsb
    return cost


def _best_mask_exhaustive(block_masks: np.ndarray, bit_count: int) -> tuple[int, int, bool]:
    popcounts = _bitcounts(bit_count)
    base_noise = int(popcounts[block_masks].sum())
    full = 1 << bit_count
    best_mask = 0
    best_cost = None
    unique = True

    for mask in range(1, full):
        cost = _cost_of_mask(mask, block_masks, popcounts, base_noise)
        if best_cost is None or cost < best_cost:
            best_mask = mask
            best_cost = cost
            unique = True
        elif cost == best_cost:
            unique = False

    assert best_cost is not None
    return best_mask, int(best_cost), unique


def _seed_support_scores(block_masks: np.ndarray, bit_count: int) -> np.ndarray:
    popcounts = _bitcounts(bit_count)
    row_scores = popcounts[block_masks].astype(float)
    col_scores = np.zeros(bit_count, dtype=float)
    for idx in range(bit_count):
        bit = 1 << idx
        col_scores[idx] = float(np.count_nonzero((block_masks & bit) != 0))
    return row_scores + col_scores


def _local_improve(mask: int, block_masks: np.ndarray, popcounts: np.ndarray, base_noise: int) -> tuple[int, int]:
    current = int(mask)
    current_cost = _cost_of_mask(current, block_masks, popcounts, base_noise)

    improved = True
    while improved:
        improved = False
        best_neighbor = current
        best_neighbor_cost = current_cost
        for idx in range(block_masks.size):
            candidate = current ^ (1 << idx)
            if candidate == 0:
                continue
            cand_cost = _cost_of_mask(candidate, block_masks, popcounts, base_noise)
            if cand_cost < best_neighbor_cost:
                best_neighbor = candidate
                best_neighbor_cost = cand_cost
        if best_neighbor != current:
            current = best_neighbor
            current_cost = best_neighbor_cost
            improved = True

    return current, current_cost


def _best_mask_heuristic(block_masks: np.ndarray, bit_count: int) -> tuple[int, int]:
    popcounts = _bitcounts(bit_count)
    base_noise = int(popcounts[block_masks].sum())
    scores = _seed_support_scores(block_masks, bit_count)
    order = np.argsort(-scores)
    starts = {int(block_mask) for block_mask in block_masks if int(block_mask) != 0}

    for k in range(1, bit_count + 1):
        mask = 0
        for idx in order[:k]:
            mask |= 1 << int(idx)
        starts.add(mask)

    best_mask = 1
    best_cost = _cost_of_mask(best_mask, block_masks, popcounts, base_noise)
    for start in starts:
        improved_mask, improved_cost = _local_improve(int(start), block_masks, popcounts, base_noise)
        if improved_cost < best_cost:
            best_mask = improved_mask
            best_cost = improved_cost
    return best_mask, best_cost


def _recover_seed_mask_from_grid(grid: np.ndarray) -> tuple[int, int, bool]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = Color.BLACK
    assert np.any(g == bg), "expected black background pixels"

    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    n = _infer_n(g.shape)
    block_masks = _grid_to_block_masks(g, fg, n)
    bit_count = n * n

    if n <= EXACT_SEARCH_N:
        mask, cost, unique = _best_mask_exhaustive(block_masks, bit_count)
    else:
        mask, cost = _best_mask_heuristic(block_masks, bit_count)
        unique = True
    return mask, fg, unique


def _render_clean(mask: int, n: int, fg: int) -> np.ndarray:
    seed = _mask_to_array(mask, n).astype(int)
    tile = seed * int(fg)
    return np.kron(seed, tile).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a doubly encoded seed by cross-checking its macro occupancy and micro block copies.

    Input structure:
    - The background is black and exactly one non-black color carries the damaged pattern.
    - The grid side length is `n^2`; viewed as an `n x n` array of `n x n` blocks, each true occupied block is a noisy copy of one unknown `n x n` binary seed.
    - The same seed also determines which macro blocks should be occupied, while empty macro blocks may contain only sparse noise.

    Rule:
    - Infer the binary seed that best explains both scales at once: occupied macro positions must equal the seed, and every occupied block should match that same seed internally.
    - Treat mismatches inside occupied blocks as local corruption and foreground pixels inside empty blocks as noise.
    - Rebuild the clean image by placing the recovered seed copy into every recovered occupied block.

    Key insight:
    - The hidden mask is written twice, once as the block-level occupancy map and once inside every occupied block, so the correct seed is the unique self-consistent pattern across the two scales.
    """
    g = np.asarray(grid, dtype=int)
    n = _infer_n(g.shape)
    mask, fg, unique = _recover_seed_mask_from_grid(g)
    assert unique, "expected a unique self-consistent seed"
    return _render_clean(mask, n, fg)


def _sample_seed_mask(n: int, rng: np.random.Generator) -> int:
    bit_count = n * n
    min_area = 2 if bit_count <= 4 else max(3, n)
    max_area = min(bit_count - 1, max(min_area, int(math.ceil(bit_count * 0.65))))

    for _ in range(300):
        p = float(rng.uniform(0.28, 0.62))
        seed = (rng.random((n, n)) < p).astype(int)
        area = int(seed.sum())
        if not (min_area <= area <= max_area):
            continue
        if int(seed.any(axis=0).sum()) < 2 or int(seed.any(axis=1).sum()) < 2:
            continue
        if np.array_equal(seed, seed[0, 0] * np.ones_like(seed)):
            continue
        return _array_to_mask(seed)

    raise RuntimeError("failed to sample a nondegenerate seed")


def _choose_indices(coords: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0 or coords.size == 0:
        return np.empty((0, 2), dtype=int)
    pick = rng.choice(coords.shape[0], size=count, replace=False)
    return coords[np.asarray(pick, dtype=int)]


def _corrupt_occupied_block(
    seed: np.ndarray,
    rng: np.random.Generator,
    mode: str,
) -> tuple[np.ndarray, bool]:
    block = seed.copy()
    ones = np.argwhere(seed == 1)
    zeros = np.argwhere(seed == 0)
    area = int(ones.shape[0])
    noisy = False

    max_missing = min(max(1, area // 3), area - 1)
    max_extra = min(max(1, max(1, zeros.shape[0] // 4)), int(zeros.shape[0]))
    max_total_mismatch = max(1, (seed.size - 1) // 2)
    miss_count = 0
    extra_count = 0

    if mode == "pristine":
        return block, False
    if mode == "missing":
        miss_count = int(rng.integers(1, max_missing + 1))
    elif mode == "extra":
        extra_count = int(rng.integers(1, max_extra + 1))
    else:
        miss_count = int(rng.integers(1, max_missing + 1))
        extra_budget = min(max_extra, max_total_mismatch - miss_count)
        if extra_budget > 0:
            extra_count = int(rng.integers(1, extra_budget + 1))

    for r, c in _choose_indices(ones, miss_count, rng):
        block[r, c] = 0
        noisy = True
    for r, c in _choose_indices(zeros, extra_count, rng):
        block[r, c] = 1
        noisy = True

    assert int((block == seed).sum()) > seed.size // 2, "occupied block must still mostly match the seed"
    return block, noisy


def _corrupt_empty_block(
    seed: np.ndarray,
    rng: np.random.Generator,
    force_noise: bool,
) -> tuple[np.ndarray, bool]:
    block = np.zeros_like(seed)
    area = int(seed.sum())
    ones = np.argwhere(seed == 1)
    all_coords = np.argwhere(np.ones_like(seed, dtype=bool))

    if not force_noise and rng.random() < 0.45:
        return block, False

    if rng.random() < 0.6:
        max_noise = min(max(1, area - 1), max(1, min(3, area)))
        count = int(rng.integers(1, max_noise + 1))
        coords = _choose_indices(ones, count, rng)
    else:
        max_noise = min(max(1, area - 1), max(1, seed.size // 4))
        count = int(rng.integers(1, max_noise + 1))
        coords = _choose_indices(all_coords, count, rng)

    for r, c in coords:
        block[r, c] = 1
    assert not np.array_equal(block, seed), "empty-block noise must not form the seed"
    return block, True


def _corrupt_clean_output(
    clean: np.ndarray,
    seed_mask: int,
    color: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    n = _infer_n(clean.shape)
    seed = _mask_to_array(seed_mask, n)
    out = np.zeros_like(clean)

    occupied_modes = ["missing", "extra", "mixed"]
    require_pristine = bool(rng.integers(0, 2))
    require_empty_noise = bool(rng.integers(0, 2))
    did_damage = False
    empty_noise_done = False
    pristine_done = False

    for br in range(n):
        for bc in range(n):
            macro_on = bool(seed[br, bc])
            if macro_on:
                if require_pristine and not pristine_done:
                    mode = "pristine"
                    pristine_done = True
                else:
                    mode = occupied_modes[int(rng.integers(0, len(occupied_modes)))]
                block, noisy = _corrupt_occupied_block(seed, rng, mode)
                did_damage |= noisy
            else:
                force_noise = require_empty_noise and not empty_noise_done
                block, noisy = _corrupt_empty_block(seed, rng, force_noise=force_noise)
                if noisy:
                    empty_noise_done = True
                did_damage |= noisy
            out[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = block * int(color)

    if require_empty_noise:
        assert empty_noise_done, "branch coverage expected an empty noisy block"
    return out, did_damage


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_RETRY_LIMIT):
        n = int(rng.choice(np.array([2, 3, 4], dtype=int), p=np.array([0.2, 0.45, 0.35])))
        seed_mask = _sample_seed_mask(n, rng)
        color = int(rng.choice(np.array(COLOR_POOL, dtype=int)))
        clean = _render_clean(seed_mask, n, color)
        noisy, did_damage = _corrupt_clean_output(clean, seed_mask, color, rng)
        if not did_damage or np.array_equal(noisy, clean):
            continue

        recovered_mask, recovered_color, unique = _recover_seed_mask_from_grid(noisy)
        if not unique or recovered_color != color or recovered_mask != seed_mask:
            continue
        if not np.array_equal(main(noisy), clean):
            continue
        return noisy

    raise RuntimeError("failed to generate a valid input for puzzle 7")
