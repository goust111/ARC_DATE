import random
from collections import Counter
from pathlib import Path
import sys

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import Color, get_bg  # type: ignore
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader robust
    class Color:
        BLACK = 0
        BLUE = 1
        GREY = 5

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


SEP = int(Color.GREY)
BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
INPUT_COLORS = (2, 3, 4, 6, 7, 8, 9)
BLOCK_COUNTS = (4, 5, 6)
BLOCK_SIZES = (4, 5)
MAX_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    start = 0
    for idx in separators + [size]:
        if idx > start:
            bands.append((start, idx))
        start = idx + 1
    return bands


def _extract_blocks(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == SEP))]
    sep_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == SEP))]
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    blocks = [g[r0:r1, c0:c1].copy() for r0, r1 in row_bands for c0, c1 in col_bands]
    assert 4 <= len(blocks) <= 6
    shapes = {block.shape for block in blocks}
    assert len(shapes) == 1
    shape = next(iter(shapes))
    assert shape[0] == shape[1] and shape[0] in BLOCK_SIZES
    assert all(not np.any(block == SEP) for block in blocks)
    return blocks


def _rotation_to_match(block: np.ndarray, ref: np.ndarray) -> int | None:
    matches = [k for k in range(4) if np.array_equal(np.rot90(block, k), ref)]
    return matches[0] if len(matches) == 1 else None


def _compatible_rotation_scores(block: np.ndarray, ref: np.ndarray) -> list[tuple[int, int, int]]:
    candidates = []
    for k in range(4):
        rotated = np.rot90(block, k)
        conflict = (rotated != BLACK) & (ref != BLACK) & (rotated != ref)
        if np.any(conflict):
            continue
        matches = int(np.sum(rotated == ref))
        missing = int(np.sum(rotated == BLACK))
        candidates.append((matches, -missing, k))
    return candidates


def _best_rotation_to_ref(block: np.ndarray, ref: np.ndarray) -> int:
    candidates = _compatible_rotation_scores(block, ref)
    assert candidates
    candidates.sort(reverse=True)
    best = candidates[0]
    tied = [item for item in candidates if item[:2] == best[:2]]
    assert len(tied) == 1
    return int(best[2])


def _majority_pattern(blocks: list[np.ndarray]) -> np.ndarray:
    h, w = blocks[0].shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            votes = [int(block[r, c]) for block in blocks if int(block[r, c]) != BLACK]
            assert votes
            counts = Counter(votes)
            top = counts.most_common()
            assert len(top) == 1 or top[0][1] > top[1][1]
            out[r, c] = int(top[0][0])
    return out


def _solve_blocks(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    exact_rot = [[None for _ in blocks] for _ in blocks]
    degrees = []
    for i, ref in enumerate(blocks):
        degree = 0
        for j, block in enumerate(blocks):
            if i == j:
                continue
            rot = _rotation_to_match(block, ref)
            exact_rot[i][j] = rot
            if rot is not None:
                degree += 1
        degrees.append(degree)

    pivot_idx = max(range(len(blocks)), key=lambda idx: (degrees[idx], -idx))
    pivot = blocks[pivot_idx]
    exact_to_pivot = {pivot_idx: 0}
    for j in range(len(blocks)):
        if j == pivot_idx:
            continue
        rot = exact_rot[pivot_idx][j]
        if rot is not None:
            exact_to_pivot[j] = int(rot)

    mode_counts = Counter(exact_to_pivot.values()).most_common()
    assert mode_counts and (len(mode_counts) == 1 or mode_counts[0][1] > mode_counts[1][1])
    pivot_to_regular = (-int(mode_counts[0][0])) % 4
    regular_pattern = np.rot90(pivot, pivot_to_regular)

    normalized = [np.rot90(block, _best_rotation_to_ref(block, regular_pattern)) for block in blocks]
    consensus = _majority_pattern(normalized)

    damage_masks = []
    for block in normalized:
        mismatch = block != consensus
        damage_mask = (block == BLACK) & (consensus != BLACK)
        assert not np.any(mismatch & ~damage_mask)
        damage_masks.append(damage_mask)

    damaged = [idx for idx, mask in enumerate(damage_masks) if np.any(mask)]
    assert len(damaged) == 1
    return consensus, damage_masks[damaged[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize rotated duplicate tiles, recover their shared motif by majority vote, and mark the damaged cells in blue.

    Input structure:
    - The grid is partitioned by full grey separator rows and columns into 4-6 equal square blocks of size 4x4 or 5x5.
    - Each block contains the same multicolor motif under one of the four quarter-turn rotations.
    - Exactly one block is damaged: some motif cells are replaced by black, while intact blocks contain no black cells.

    Rule:
    - Detect how each block is rotated relative to the unique most-common orientation.
    - Rotate every block into that regular orientation.
    - Reconstruct the consensus motif by per-cell majority vote across the normalized blocks.
    - Find the unique damaged block as the normalized block whose disagreements with the consensus are only black holes.
    - Output the consensus motif in the regular orientation, recoloring those damaged coordinates blue.

    Key insight:
    - Raw voting fails because corresponding motif cells are misaligned across rotations; only after orientation normalization do the intact copies agree enough to expose the damaged black holes.
    """
    blocks = _extract_blocks(np.asarray(grid, dtype=int))
    consensus, damage_mask = _solve_blocks(blocks)
    out = consensus.copy()
    out[damage_mask] = BLUE
    return out


def _sample_pattern(size: int) -> np.ndarray:
    num_colors = random.choice((2, 3))
    colors = random.sample(INPUT_COLORS, num_colors)
    pattern = np.random.choice(colors, size=(size, size))
    for color in colors:
        if not np.any(pattern == color):
            rr = random.randrange(size)
            cc = random.randrange(size)
            pattern[rr, cc] = color
    return pattern.astype(int)


def _layout_for_count(count: int) -> tuple[int, int]:
    if count == 4:
        return 2, 2
    if count == 5:
        return random.choice(((1, 5), (5, 1)))
    return random.choice(((2, 3), (3, 2)))


def _sample_orientations(count: int) -> tuple[list[int], int]:
    regular = random.randrange(4)
    regular_count = random.randint(3, count - 1)
    angles = [regular] * regular_count
    others = [angle for angle in range(4) if angle != regular]
    while len(angles) < count:
        angles.append(random.choice(others))
    random.shuffle(angles)
    assert Counter(angles)[regular] >= 3
    assert len(set(angles)) >= 2
    return angles, regular


def _damage_positions(size: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(cells)
    limit = max(1, size - 1)
    count = random.randint(1, limit)
    return cells[:count]


def _render_blocks(blocks: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    size = blocks[0].shape[0]
    grid = np.full((rows * size + (rows - 1), cols * size + (cols - 1)), SEP, dtype=int)
    idx = 0
    for br in range(rows):
        for bc in range(cols):
            r0 = br * (size + 1)
            c0 = bc * (size + 1)
            grid[r0:r0 + size, c0:c0 + size] = blocks[idx]
            idx += 1
    return grid


def _expected_output(pattern: np.ndarray, regular: int, damaged_angle: int, damaged_cells: list[tuple[int, int]]) -> np.ndarray:
    del damaged_angle
    out = np.rot90(pattern, regular).copy()
    mask = np.zeros_like(pattern, dtype=bool)
    for r, c in damaged_cells:
        mask[r, c] = True
    normalized_mask = np.rot90(mask.astype(int), regular).astype(bool)
    out[normalized_mask] = BLUE
    return out


def _naive_vote(blocks: list[np.ndarray]) -> np.ndarray | None:
    h, w = blocks[0].shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            votes = [int(block[r, c]) for block in blocks if int(block[r, c]) != BLACK]
            if not votes:
                return None
            top = Counter(votes).most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                return None
            out[r, c] = int(top[0][0])
    return out


def _attempt() -> np.ndarray | None:
    count = random.choice(BLOCK_COUNTS)
    size = random.choice(BLOCK_SIZES)
    rows, cols = _layout_for_count(count)

    pattern = _sample_pattern(size)
    if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
        return None

    angles, regular = _sample_orientations(count)
    damaged_idx = random.randrange(count)
    damaged_cells = _damage_positions(size)

    blocks = [np.rot90(pattern, angle) for angle in angles]
    damaged_block = blocks[damaged_idx].copy()
    damage_mask = np.rot90(
        np.array([[1 if (r, c) in damaged_cells else 0 for c in range(size)] for r in range(size)], dtype=int),
        angles[damaged_idx],
    ).astype(bool)
    damaged_block[damage_mask] = BLACK
    blocks[damaged_idx] = damaged_block

    regular_pattern = np.rot90(pattern, regular)
    candidates = _compatible_rotation_scores(blocks[damaged_idx], regular_pattern)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
        return None

    naive = _naive_vote(blocks)
    if naive is not None and np.array_equal(naive, regular_pattern):
        return None

    exact_counts = Counter(angles)
    if exact_counts[regular] < 3 or sum(v == exact_counts[regular] for v in exact_counts.values()) != 1:
        return None

    grid = _render_blocks(blocks, rows, cols)
    expected = _expected_output(pattern, regular, angles[damaged_idx], damaged_cells)
    try:
        got = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(got, expected):
        return None

    if grid.shape[0] > 30 or grid.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
