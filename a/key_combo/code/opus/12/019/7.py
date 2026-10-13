from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FG = 1
GRID_BLOCKS = 3
COUNT_TO_COLOR = {0: 0, 1: 1, 2: 2, 3: 3}

# The exact rule in the prompt needs offsets +/-2 and +/-3, so a 7x7 block is
# the smallest non-degenerate case; 6x6 blocks cannot host any valid center.
GEN_BLOCK_SIZE = 7
GEN_GRID_SIZE = GRID_BLOCKS * GEN_BLOCK_SIZE
GEN_ATTEMPTS = 400

ARM_BARS = {
    "up": ((0, 3), (1, 3)),
    "down": ((5, 3), (6, 3)),
    "left": ((3, 0), (3, 1)),
    "right": ((3, 5), (3, 6)),
}
ARM_NAMES = tuple(ARM_BARS.keys())
ARM_CELLS = frozenset(cell for pair in ARM_BARS.values() for cell in pair)
RESERVED_CELLS = frozenset({(3, 3), (2, 3), (4, 3), (3, 2), (3, 4)})
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _count_hidden_crosses(block: np.ndarray) -> int:
    """Count implied cross centers inside one block."""
    block = np.asarray(block, dtype=int)
    h, w = block.shape
    total = 0
    for r in range(h):
        for c in range(w):
            if int(block[r, c]) != BG:
                continue
            if not (
                r - 3 >= 0
                and r + 3 < h
                and c - 3 >= 0
                and c + 3 < w
            ):
                continue
            if (
                int(block[r - 3, c]) == FG
                and int(block[r - 2, c]) == FG
                and int(block[r + 2, c]) == FG
                and int(block[r + 3, c]) == FG
                and int(block[r, c - 3]) == FG
                and int(block[r, c - 2]) == FG
                and int(block[r, c + 2]) == FG
                and int(block[r, c + 3]) == FG
            ):
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each implicit block to the color encoding of its hidden-cross count.

    Input structure:
    - The grid is square, black-background, and evenly divisible into a 3x3 array of equal square blocks.
    - Blue two-cell micro-bars appear inside those blocks, with no visible partition lines.
    - A hidden cross center is a black cell whose four cardinal directions contain matching two-cell bars at offsets 2 and 3 within the same block.

    Rule:
    - Split the input into its 3x3 equal blocks.
    - In each block, count every black cell whose up, down, left, and right arm positions are all blue at offsets 2 and 3.
    - Encode the per-block count into the corresponding output cell using 0->black, 1->blue, 2->red, 3->green.

    Key insight:
    - The output does not sample a visible foreground pixel; it samples a blockwise count of an implied pattern whose validity depends on the hidden block boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % GRID_BLOCKS == 0, "expected a square grid split evenly into 3x3 blocks"

    block_size = h // GRID_BLOCKS
    out = np.empty((GRID_BLOCKS, GRID_BLOCKS), dtype=int)
    for br in range(GRID_BLOCKS):
        for bc in range(GRID_BLOCKS):
            block = g[
                br * block_size : (br + 1) * block_size,
                bc * block_size : (bc + 1) * block_size,
            ]
            count = _count_hidden_crosses(block)
            assert count in COUNT_TO_COLOR, "expected block counts to stay within 0..3"
            out[br, bc] = COUNT_TO_COLOR[count]
    return out


def _extra_bar_candidates() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    forbidden = ARM_CELLS | RESERVED_CELLS
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(GEN_BLOCK_SIZE):
        for c in range(GEN_BLOCK_SIZE):
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 >= GEN_BLOCK_SIZE or c2 >= GEN_BLOCK_SIZE:
                    continue
                pair = ((r, c), (r2, c2))
                if pair[0] in forbidden or pair[1] in forbidden:
                    continue
                pairs.append(pair)
    return pairs


EXTRA_BAR_CANDIDATES = _extra_bar_candidates()


def _can_place_bar(
    pair: tuple[tuple[int, int], tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    pair_set = set(pair)
    if pair_set & occupied:
        return False
    for r, c in pair:
        for dr, dc in FOUR_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in pair_set:
                continue
            if (nr, nc) in occupied:
                return False
    return True


def _place_extra_bars(
    occupied: set[tuple[int, int]],
    extra_count: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    for _ in range(120):
        local_occupied = set(occupied)
        chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
        candidates = EXTRA_BAR_CANDIDATES[:]
        random.shuffle(candidates)
        for pair in candidates:
            if len(chosen) >= extra_count:
                break
            if not _can_place_bar(pair, local_occupied):
                continue
            chosen.append(pair)
            local_occupied.update(pair)
        if len(chosen) == extra_count:
            return chosen
    return None


def _render_block(
    target_count: int,
    *,
    forced_arm_count: int | None = None,
    forced_extra_count: int | None = None,
) -> np.ndarray | None:
    for _ in range(200):
        if target_count == 1:
            chosen_arm_names = list(ARM_NAMES)
            arm_count = 4
        else:
            arm_count = forced_arm_count if forced_arm_count is not None else random.choices(
                [0, 1, 2, 3], weights=[1, 2, 3, 3], k=1
            )[0]
            chosen_arm_names = random.sample(ARM_NAMES, arm_count)

        extra_count = forced_extra_count if forced_extra_count is not None else random.randint(0, 3)

        occupied: set[tuple[int, int]] = set()
        for name in chosen_arm_names:
            occupied.update(ARM_BARS[name])

        extra_bars = _place_extra_bars(occupied, extra_count)
        if extra_bars is None:
            continue

        block = np.zeros((GEN_BLOCK_SIZE, GEN_BLOCK_SIZE), dtype=int)
        for r, c in occupied:
            block[r, c] = FG
        for pair in extra_bars:
            for r, c in pair:
                block[r, c] = FG

        if _count_hidden_crosses(block) != target_count:
            continue
        if not _all_blue_components_are_bars(block):
            continue
        return block
    return None


def _all_blue_components_are_bars(grid: np.ndarray) -> bool:
    seen: set[tuple[int, int]] = set()
    blue_cells = np.argwhere(grid == FG)
    for r0, c0 in blue_cells:
        start = (int(r0), int(c0))
        if start in seen:
            continue
        q: deque[tuple[int, int]] = deque([start])
        comp: list[tuple[int, int]] = []
        seen.add(start)
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in FOUR_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == FG:
                    nxt = (nr, nc)
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
        if len(comp) != 2:
            return False
        (r1, c1), (r2, c2) = comp
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    return True


def _arm_presence(block: np.ndarray) -> int:
    total = 0
    for pair in ARM_BARS.values():
        if all(int(block[r, c]) == FG for r, c in pair):
            total += 1
    return total


def _blue_pixel_count(block: np.ndarray) -> int:
    return int(np.count_nonzero(block == FG))


def _generate_counts() -> np.ndarray:
    ones = random.randint(2, 7)
    flat = [1] * ones + [0] * (GRID_BLOCKS * GRID_BLOCKS - ones)
    random.shuffle(flat)
    return np.array(flat, dtype=int).reshape(GRID_BLOCKS, GRID_BLOCKS)


def _compose_grid(blocks: list[list[np.ndarray]]) -> np.ndarray:
    rows = [np.hstack(row) for row in blocks]
    return np.vstack(rows)


def _semantic_check(grid: np.ndarray, target_counts: np.ndarray) -> bool:
    if grid.shape != (GEN_GRID_SIZE, GEN_GRID_SIZE):
        return False

    actual_counts = np.zeros((GRID_BLOCKS, GRID_BLOCKS), dtype=int)
    blue_totals: dict[int, list[int]] = {0: [], 1: []}
    has_near_miss = False
    has_noisy_true = False

    for br in range(GRID_BLOCKS):
        for bc in range(GRID_BLOCKS):
            block = grid[
                br * GEN_BLOCK_SIZE : (br + 1) * GEN_BLOCK_SIZE,
                bc * GEN_BLOCK_SIZE : (bc + 1) * GEN_BLOCK_SIZE,
            ]
            if not _all_blue_components_are_bars(block):
                return False
            actual = _count_hidden_crosses(block)
            actual_counts[br, bc] = actual
            if actual not in (0, 1):
                return False
            blue_totals[actual].append(_blue_pixel_count(block))
            if actual == 0 and _arm_presence(block) == 3:
                has_near_miss = True
            if actual == 1 and _blue_pixel_count(block) > 8:
                has_noisy_true = True

    if not np.array_equal(actual_counts, target_counts):
        return False
    if len(set(map(int, target_counts.flatten()))) < 2:
        return False
    if not has_near_miss or not has_noisy_true:
        return False
    if not (set(blue_totals[0]) & set(blue_totals[1])):
        return False
    return np.array_equal(main(grid), target_counts)


def _gen_attempt() -> np.ndarray | None:
    target_counts = _generate_counts()
    one_positions = [(r, c) for r in range(GRID_BLOCKS) for c in range(GRID_BLOCKS) if int(target_counts[r, c]) == 1]
    zero_positions = [(r, c) for r in range(GRID_BLOCKS) for c in range(GRID_BLOCKS) if int(target_counts[r, c]) == 0]
    if not one_positions or not zero_positions:
        return None

    special_true = random.choice(one_positions)
    special_false = random.choice(zero_positions)

    blocks: list[list[np.ndarray]] = [[None for _ in range(GRID_BLOCKS)] for _ in range(GRID_BLOCKS)]  # type: ignore[list-item]
    for br in range(GRID_BLOCKS):
        for bc in range(GRID_BLOCKS):
            target = int(target_counts[br, bc])
            if (br, bc) == special_true:
                block = _render_block(1, forced_extra_count=1)
            elif (br, bc) == special_false:
                block = _render_block(0, forced_arm_count=3, forced_extra_count=2)
            else:
                block = _render_block(target)
            if block is None:
                return None
            blocks[br][bc] = block

    grid = _compose_grid(blocks)
    if not _semantic_check(grid, target_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
