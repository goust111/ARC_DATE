from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
BLACK = 0
SEPARATOR = 6
PALETTE = (2, 3, 4, 5, 7, 8, 9)

BLOCK_SIZES = (4, 5)
LAYOUTS = ((2, 3), (3, 3))
MAX_SIDE = 30
GEN_RETRIES = 400


def _solid_lines(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    lines: list[tuple[int, int]] = []
    limit = grid.shape[axis]
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if np.all(line == color):
            lines.append((idx, color))
    return lines


def _content_bands(separators: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in [*separators, limit]:
        assert sep > start, "separator lines must leave nonempty content bands"
        bands.append((start, sep))
        start = sep + 1
    return bands


def _extract_blocks(grid: np.ndarray) -> tuple[list[np.ndarray], int, tuple[int, int]]:
    solid_rows = _solid_lines(grid, axis=0)
    solid_cols = _solid_lines(grid, axis=1)
    assert solid_rows and solid_cols, "expected separator rows and columns"

    separator_colors = {color for _, color in solid_rows} | {color for _, color in solid_cols}
    assert len(separator_colors) == 1, "separator lines must share one color"
    separator = next(iter(separator_colors))

    row_bands = _content_bands([idx for idx, _ in solid_rows], grid.shape[0])
    col_bands = _content_bands([idx for idx, _ in solid_cols], grid.shape[1])
    block_rows = len(row_bands)
    block_cols = len(col_bands)
    assert (block_rows, block_cols) in LAYOUTS, "expected a 2x3 or 3x3 tile lattice"

    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all tiles must share one size"
    size_r = heights.pop()
    size_c = widths.pop()
    assert size_r == size_c, "tiles must be square"
    size = size_r

    blocks: list[np.ndarray] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            block = np.asarray(grid[r0:r1, c0:c1], dtype=int)
            assert block.shape == (size, size)
            assert not np.any(block == separator)
            blocks.append(block)
    return blocks, size, (block_rows, block_cols)


def _match_score(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a == b))


def _best_rotation(block: np.ndarray, target: np.ndarray) -> tuple[int, list[int]]:
    scores = [_match_score(np.rot90(block, k), target) for k in range(4)]
    best = max(scores)
    best_ks = [k for k, score in enumerate(scores) if score == best]
    return best_ks[0], scores


def _block_signature(block: np.ndarray) -> bytes:
    return np.ascontiguousarray(block).tobytes()


def _choose_prototype(blocks: list[np.ndarray]) -> np.ndarray:
    sig_counts = Counter(_block_signature(block) for block in blocks)
    area = blocks[0].size
    candidates: list[tuple[tuple[int, int, int, bytes], np.ndarray]] = []
    for idx, block in enumerate(blocks):
        sig = _block_signature(block)
        raw_matches = sig_counts[sig]
        total_score = 0
        full_matches = 0
        for other in blocks:
            _, scores = _best_rotation(other, block)
            best = max(scores)
            total_score += best
            full_matches += int(best == area)
        candidates.append(((raw_matches, full_matches, total_score, sig), np.asarray(block, dtype=int)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _normalize_blocks(blocks: list[np.ndarray], prototype: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    normalized: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    for block in blocks:
        rot, scores = _best_rotation(block, prototype)
        normalized.append(np.rot90(block, rot))
        score_rows.append(np.asarray(scores, dtype=int))
    return normalized, score_rows


def _unique_mode(values: np.ndarray) -> int:
    ranked = Counter(map(int, values.reshape(-1))).most_common()
    assert ranked, "expected nonempty majority vote"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "majority vote must be unique"
    return int(ranked[0][0])


def _majority_tile(blocks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(blocks, axis=0)
    size = stack.shape[1]
    tile = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            tile[r, c] = _unique_mode(stack[:, r, c])
    return tile


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks, size, _layout = _extract_blocks(grid)
    assert size in BLOCK_SIZES
    prototype = _choose_prototype(blocks)
    normalized, _score_rows = _normalize_blocks(blocks, prototype)
    motif = _majority_tile(normalized)

    diffs = np.stack([block != motif for block in normalized], axis=0)
    diff_counts = diffs.reshape(len(normalized), size * size).sum(axis=1)
    damaged = np.flatnonzero(diff_counts > 0)
    assert len(damaged) == 1, "expected exactly one damaged tile"
    damaged_idx = int(damaged[0])
    damage_mask = diffs[damaged_idx]
    assert np.any(damage_mask), "damage mask must be nonempty"
    assert np.all(normalized[damaged_idx][damage_mask] == BLACK), "damage must erase to black"
    assert np.all(motif[damage_mask] != BLACK), "damage must only affect foreground pixels"

    return motif, damage_mask, prototype


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize rotated tile copies, repair the lone damaged copy by consensus, and mark the missing cells blue.

    Input structure:
    - The input is a 2x3 or 3x3 lattice of equal square tiles separated by one-cell solid magenta lines.
    - Every tile is a rotated copy of one multicolor motif on a black background.
    - Exactly one tile is damaged: some of its foreground cells were replaced by black, and blue is absent from the input.

    Rule:
    - Extract the separator-delimited tiles and infer the consensus orientation by finding the repeated raw tile orientation.
    - Rotate every tile into that shared orientation, then recover the motif by unique per-cell majority vote across the normalized tiles.
    - Find the single normalized tile that still disagrees with the consensus motif and recolor its disagreement coordinates blue in the output.

    Key insight:
    - Majority voting only works after the rotated copies are aligned; the repeated observed orientation identifies the normalization target, and the lone damaged copy is then exposed by the consensus mismatch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= MAX_SIDE
    assert not np.any(g == BLUE), "blue is reserved for the output marker"

    motif, damage_mask, _prototype = _solve(g)
    out = motif.copy()
    out[damage_mask] = BLUE
    return out


def _balanced_color_counts(total: int, n_colors: int) -> list[int]:
    counts = [1] * n_colors
    remaining = total - n_colors
    while remaining:
        idx = random.randrange(n_colors)
        counts[idx] += 1
        remaining -= 1
    return counts


def _make_seed_motif(size: int) -> np.ndarray:
    area = size * size
    while True:
        motif = np.zeros((size, size), dtype=int)
        n_colors = random.randint(2, 3)
        colors = random.sample(PALETTE, n_colors)
        fg_count = random.randint(max(6, n_colors + 2), min(area - 3, (area * 11) // 20))
        cells = random.sample(range(area), fg_count)
        counts = _balanced_color_counts(fg_count, n_colors)
        values: list[int] = []
        for color, count in zip(colors, counts):
            values.extend([color] * count)
        random.shuffle(values)
        for flat, color in zip(cells, values):
            r, c = divmod(flat, size)
            motif[r, c] = color

        if len({tuple(coord) for coord in np.argwhere(motif != BLACK)}) < 6:
            continue
        if len({int(r) for r, _ in np.argwhere(motif != BLACK)}) < 2:
            continue
        if len({int(c) for _, c in np.argwhere(motif != BLACK)}) < 2:
            continue
        if any(np.array_equal(motif, np.rot90(motif, k)) for k in (1, 2, 3)):
            continue
        return motif


def _orientation_plan(n_tiles: int) -> tuple[int, list[int]]:
    while True:
        consensus = random.randrange(4)
        dominant = random.randint(3, n_tiles - 1)
        others = [k for k in range(4) if k != consensus]
        used = random.sample(others, random.randint(1, min(3, n_tiles - dominant)))
        counts = [0, 0, 0, 0]
        counts[consensus] = dominant
        remaining = n_tiles - dominant
        for k in used:
            counts[k] = 1
            remaining -= 1
        while remaining:
            choices = [k for k in used if counts[k] < dominant]
            if not choices:
                break
            counts[random.choice(choices)] += 1
            remaining -= 1
        if remaining != 0:
            continue
        if sum(count > 0 for count in counts) < 2:
            continue
        if counts[consensus] <= max(count for i, count in enumerate(counts) if i != consensus):
            continue
        orientations: list[int] = []
        for k, count in enumerate(counts):
            orientations.extend([k] * count)
        random.shuffle(orientations)
        return consensus, orientations


def _rotated_damage_mask(size: int, damage_cells: list[tuple[int, int]], k: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in damage_cells:
        mask[r, c] = True
    return np.rot90(mask, k)


def _assemble_grid(blocks: list[np.ndarray], layout: tuple[int, int]) -> np.ndarray:
    block_rows, block_cols = layout
    size = blocks[0].shape[0]
    height = block_rows * size + (block_rows - 1)
    width = block_cols * size + (block_cols - 1)
    grid = np.full((height, width), SEPARATOR, dtype=int)
    idx = 0
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * (size + 1)
            c0 = bc * (size + 1)
            grid[r0:r0 + size, c0:c0 + size] = blocks[idx]
            idx += 1
    return grid


def _raw_majority(blocks: list[np.ndarray]) -> np.ndarray:
    size = blocks[0].shape[0]
    stack = np.stack(blocks, axis=0)
    out = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            out[r, c] = Counter(map(int, stack[:, r, c])).most_common(1)[0][0]
    return out


def _expected_output(seed: np.ndarray, consensus: int, damage_cells: list[tuple[int, int]]) -> np.ndarray:
    motif = np.rot90(seed, consensus)
    damage_mask = _rotated_damage_mask(seed.shape[0], damage_cells, consensus)
    out = motif.copy()
    out[damage_mask] = BLUE
    return out


def _instance_is_good(
    grid: np.ndarray,
    expected: np.ndarray,
    seed: np.ndarray,
    consensus: int,
    orientations: list[int],
    damaged_idx: int,
    damage_cells: list[tuple[int, int]],
) -> bool:
    if max(grid.shape) > MAX_SIDE or max(expected.shape) > MAX_SIDE:
        return False
    if len(set(orientations)) < 2:
        return False
    if orientations[damaged_idx] == consensus:
        return False

    blocks, _size, _layout = _extract_blocks(grid)
    sig_counts = Counter(_block_signature(block) for block in blocks)
    ranked = sig_counts.most_common()
    if len(ranked) < 2 or ranked[0][1] <= ranked[1][1] or ranked[0][1] < 3:
        return False

    prototype = _choose_prototype(blocks)
    normalized, score_rows = _normalize_blocks(blocks, prototype)
    for idx, scores in enumerate(score_rows):
        best = int(scores.max())
        if int(np.sum(scores == best)) != 1:
            return False
        if idx != damaged_idx and best != seed.size:
            return False
    motif = _majority_tile(normalized)
    if not np.array_equal(motif, np.rot90(seed, consensus)):
        return False

    diffs = np.stack([block != motif for block in normalized], axis=0)
    damaged = np.flatnonzero(diffs.reshape(len(blocks), -1).sum(axis=1) > 0)
    if damaged.tolist() != [damaged_idx]:
        return False
    damage_mask = diffs[damaged_idx]
    if int(damage_mask.sum()) != len(damage_cells):
        return False
    expected_mask = _rotated_damage_mask(seed.shape[0], damage_cells, consensus)
    if not np.array_equal(damage_mask, expected_mask):
        return False

    raw_majority = _raw_majority(blocks)
    if np.array_equal(raw_majority, motif):
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(LAYOUTS)
    n_tiles = layout[0] * layout[1]
    size = random.choice(BLOCK_SIZES)
    seed = _make_seed_motif(size)
    consensus, orientations = _orientation_plan(n_tiles)

    non_consensus = [idx for idx, k in enumerate(orientations) if k != consensus]
    damaged_idx = random.choice(non_consensus)

    fg_cells = [tuple(map(int, coord)) for coord in np.argwhere(seed != BLACK)]
    max_damage = min(len(fg_cells) - 1, seed.size // 3, 4)
    if max_damage < 2:
        return None
    damage_count = random.randint(2, max_damage)
    damage_cells = random.sample(fg_cells, damage_count)
    damaged_seed = seed.copy()
    for r, c in damage_cells:
        damaged_seed[r, c] = BLACK

    blocks: list[np.ndarray] = []
    for idx, orientation in enumerate(orientations):
        base = damaged_seed if idx == damaged_idx else seed
        blocks.append(np.rot90(base, orientation))
    grid = _assemble_grid(blocks, layout)
    expected = _expected_output(seed, consensus, damage_cells)

    if _instance_is_good(grid, expected, seed, consensus, orientations, damaged_idx, damage_cells):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotated-majority-repair ARC input")
