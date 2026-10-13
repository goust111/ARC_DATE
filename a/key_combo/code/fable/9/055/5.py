import random
from functools import lru_cache

import numpy as np


GRID_SIZE = 9
BLOCK_SIZE = 3
FULL_MASK = (1 << (BLOCK_SIZE * BLOCK_SIZE)) - 1
NONZERO_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


def _mask_to_array(mask: int) -> np.ndarray:
    arr = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for bit in range(BLOCK_SIZE * BLOCK_SIZE):
        if (int(mask) >> bit) & 1:
            arr[bit // BLOCK_SIZE, bit % BLOCK_SIZE] = True
    return arr


def _array_to_mask(arr: np.ndarray) -> int:
    mask = 0
    flat = np.asarray(arr, dtype=bool).reshape(-1)
    for bit, value in enumerate(flat):
        if bool(value):
            mask |= 1 << bit
    return int(mask)


def _rotate_mask(mask: int, k: int) -> int:
    return _array_to_mask(np.rot90(_mask_to_array(mask), int(k)))


def _match_rotations(mask_a: int, mask_b: int) -> tuple[int, ...]:
    target = FULL_MASK ^ int(mask_b)
    return tuple(k for k in range(4) if _rotate_mask(mask_a, k) == target)


def _split_blocks(grid: np.ndarray) -> list[np.ndarray]:
    blocks = []
    for br in range(0, GRID_SIZE, BLOCK_SIZE):
        for bc in range(0, GRID_SIZE, BLOCK_SIZE):
            blocks.append(np.asarray(grid[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE], dtype=int).copy())
    return blocks


def _block_mask(block: np.ndarray) -> int:
    return _array_to_mask(np.asarray(block, dtype=int) != 0)


def _relation_graph(masks: list[int]) -> list[list[bool]]:
    n = len(masks)
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _match_rotations(masks[i], masks[j]):
                adj[i][j] = True
                adj[j][i] = True
    return adj


def _count_matchings(adj: list[list[bool]], remaining_mask: int) -> int:
    n = len(adj)

    @lru_cache(maxsize=None)
    def dfs(mask: int) -> int:
        if mask == 0:
            return 1
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ (1 << i)
        total = 0
        m = rest
        while m:
            low_j = m & -m
            j = low_j.bit_length() - 1
            if adj[i][j]:
                total += dfs(rest ^ (1 << j))
                if total > 1:
                    return total
            m ^= low_j
        return total

    assert n <= 30
    return int(dfs(int(remaining_mask)))


def _solve_unmatched_index(masks: list[int]) -> int:
    adj = _relation_graph(masks)
    n = len(masks)
    unique_isolate = None
    for isolate in range(n):
        remaining_mask = ((1 << n) - 1) ^ (1 << isolate)
        matching_count = _count_matchings(adj, remaining_mask)
        if matching_count == 1:
            if unique_isolate is not None:
                raise ValueError("multiple unmatched candidates satisfy the pairing rule")
            unique_isolate = isolate
        elif matching_count != 0:
            raise ValueError("pairing is ambiguous")
    if unique_isolate is None:
        raise ValueError("no unique unmatched block exists")
    return int(unique_isolate)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the unique 3x3 block that cannot be paired by rotated foreground-complement matching.

    Input structure:
    - The input is a 9x9 grid split into nine 3x3 blocks on a black background.
    - Each block contains a monochrome non-black foreground mask, and different blocks may use different colors.
    - Among any two blocks, a pair exists when one block's foreground mask rotated by 0, 90, 180, or 270 degrees equals the other block's black-cell mask.

    Rule:
    - Partition the grid into its nine 3x3 blocks and ignore foreground colors while extracting each block's mask.
    - Build the rotated-complement pairing relation between blocks and find the unique perfect matching on eight of them.
    - Output the only block left unmatched, preserving its original colors and 3x3 shape.

    Key insight:
    - The hidden twin relation is not equality or reflection but complement after rotation, so the answer is the lone block outside that unique rotated-complement matching.
    """
    g = np.asarray(grid, dtype=int)
    if g.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"expected a {GRID_SIZE}x{GRID_SIZE} input grid")
    blocks = _split_blocks(g)
    masks = [_block_mask(block) for block in blocks]
    return blocks[_solve_unmatched_index(masks)].copy()


ALL_GEN_MASKS = tuple(mask for mask in range(1, FULL_MASK) if 2 <= mask.bit_count() <= 7)
GEN_NEIGHBORS = {
    mask: tuple(other for other in ALL_GEN_MASKS if other != mask and _match_rotations(mask, other))
    for mask in ALL_GEN_MASKS
}


def _find_induced_matching(available: set[int], need_pairs: int) -> list[tuple[int, int]] | None:
    if need_pairs == 0:
        return []
    if len(available) < 2 * need_pairs:
        return None

    candidates = [mask for mask in available if any(nei in available for nei in GEN_NEIGHBORS[mask])]
    random.shuffle(candidates)
    for mask_a in candidates:
        partners = [mask_b for mask_b in GEN_NEIGHBORS[mask_a] if mask_b in available]
        random.shuffle(partners)
        for mask_b in partners:
            next_available = set(available)
            next_available.discard(mask_a)
            next_available.discard(mask_b)
            next_available.difference_update(GEN_NEIGHBORS[mask_a])
            next_available.difference_update(GEN_NEIGHBORS[mask_b])
            result = _find_induced_matching(next_available, need_pairs - 1)
            if result is not None:
                return [(int(mask_a), int(mask_b))] + result
    return None


def _build_grid(block_masks: list[int], block_colors: list[int], order: list[int]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for slot, block_idx in enumerate(order):
        br = (slot // BLOCK_SIZE) * BLOCK_SIZE
        bc = (slot % BLOCK_SIZE) * BLOCK_SIZE
        block = np.where(_mask_to_array(block_masks[block_idx]), int(block_colors[block_idx]), 0)
        grid[br : br + BLOCK_SIZE, bc : bc + BLOCK_SIZE] = block
    return grid


def _edge_pairs(adj: list[list[bool]]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(adj)):
        for j in range(i + 1, len(adj)):
            if adj[i][j]:
                pairs.append((i, j))
    return pairs


def _is_good_count_pattern(counts: list[int], isolate_index: int) -> bool:
    isolate_count = counts[isolate_index]
    if counts.count(isolate_count) < 2:
        return False
    if counts.count(BLOCK_SIZE * BLOCK_SIZE - isolate_count) < 1:
        return False
    return any(counts.count(k) >= 2 and counts.count(BLOCK_SIZE * BLOCK_SIZE - k) >= 2 for k in range(1, 9))


def _validate_case(grid: np.ndarray, expected_output: np.ndarray) -> None:
    blocks = _split_blocks(grid)
    masks = [_block_mask(block) for block in blocks]
    adj = _relation_graph(masks)
    edge_pairs = _edge_pairs(adj)
    degrees = [sum(row) for row in adj]
    isolate_index = _solve_unmatched_index(masks)
    counts = [mask.bit_count() for mask in masks]

    assert len(edge_pairs) == 4
    assert sorted(degrees) == [0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert _is_good_count_pattern(counts, isolate_index)
    assert any(set(_match_rotations(masks[i], masks[j])).issubset({1, 3}) for i, j in edge_pairs)
    assert np.array_equal(blocks[isolate_index], expected_output)
    assert np.array_equal(main(grid), expected_output)


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    isolate_mask = random.choice(ALL_GEN_MASKS)
    available = {
        mask
        for mask in ALL_GEN_MASKS
        if mask != isolate_mask and isolate_mask not in GEN_NEIGHBORS[mask]
    }
    pair_masks = _find_induced_matching(available, 4)
    if pair_masks is None:
        return None

    block_masks = [isolate_mask]
    for mask_a, mask_b in pair_masks:
        block_masks.extend([mask_a, mask_b])
    block_masks = [_rotate_mask(mask, random.randrange(4)) for mask in block_masks]

    order = list(range(9))
    random.shuffle(order)
    colors = random.sample(NONZERO_COLORS, 9)
    grid = _build_grid(block_masks, colors, order)
    expected_output = np.where(_mask_to_array(block_masks[0]), colors[0], 0).astype(int)

    if not np.array_equal(main(grid), expected_output):
        return None
    try:
        _validate_case(grid, expected_output)
    except (AssertionError, ValueError):
        return None
    return grid, expected_output


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sampled = _sample_case()
        if sampled is not None:
            return sampled[0]
    raise RuntimeError("failed to generate a valid rotated-complement pairing puzzle")
