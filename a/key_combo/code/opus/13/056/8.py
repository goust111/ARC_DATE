from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


GRID_SIZE = 12
BLOCK_SIZE = 6
BACKGROUND = 0
OUTPUT_COLOR = 8
DOMINANT_COLORS = (1, 2, 3)
DISTRACTOR_PALETTE = (4, 5, 6, 7, 9)

TEMPLATES = {
    1: ((1, 0), (2, 0), (2, 1)),
    2: ((1, 0), (1, 1), (1, 2)),
    3: ((0, 1), (0, 2), (1, 2)),
}

ALL_BLOCK_CELLS = tuple((r, c) for r in range(BLOCK_SIZE) for c in range(BLOCK_SIZE))
COUNT_PATTERNS = (
    (3, 3, 4, 4),
    (3, 4, 3, 4),
    (4, 3, 4, 3),
    (3, 3, 3, 4),
    (3, 4, 4, 4),
    (4, 4, 3, 4),
)
EDGE_PATTERNS = (
    (True, True, False, False),
    (True, False, True, False),
    (False, True, True, False),
    (True, True, True, False),
    (True, False, True, True),
)
GEN_ATTEMPTS = 400


def _block_origin(block_index: int) -> tuple[int, int]:
    return (block_index // 2) * BLOCK_SIZE, (block_index % 2) * BLOCK_SIZE


def _dominant_non_black(block: np.ndarray) -> int:
    counts = Counter(int(value) for value in block.flat if int(value) != BACKGROUND)
    assert counts, "each block must contain foreground pixels"
    ranked = counts.most_common(2)
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    dominant = int(ranked[0][0])
    assert dominant in TEMPLATES, "dominant color must be blue, red, or green"
    return dominant


def _block_output_mask(
    positions: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    dominant_color: int,
) -> np.ndarray:
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
    for r, c in positions:
        rr = int(r)
        cc = int(c)
        mask[rr, cc] = True
        for tr, tc in TEMPLATES[dominant_color]:
            nr = rr + int(tr) - 1
            nc = cc + int(tc) - 1
            if 0 <= nr < BLOCK_SIZE and 0 <= nc < BLOCK_SIZE:
                mask[nr, nc] = True
    return mask


def _is_far_enough(positions: tuple[tuple[int, int], ...]) -> bool:
    for i, (r1, c1) in enumerate(positions):
        for r2, c2 in positions[i + 1 :]:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) < 3:
                return False
    return True


def _has_edge_cell(positions: tuple[tuple[int, int], ...]) -> bool:
    return any(r in (0, BLOCK_SIZE - 1) or c in (0, BLOCK_SIZE - 1) for r, c in positions)


def _has_interior_cell(positions: tuple[tuple[int, int], ...]) -> bool:
    return any(0 < r < BLOCK_SIZE - 1 and 0 < c < BLOCK_SIZE - 1 for r, c in positions)


def _templates_are_distinguishable(positions: tuple[tuple[int, int], ...]) -> bool:
    outputs = [_block_output_mask(positions, color).tobytes() for color in DOMINANT_COLORS]
    return len(set(outputs)) == len(DOMINANT_COLORS)


def _has_clipping(positions: tuple[tuple[int, int], ...], dominant_color: int) -> bool:
    for r, c in positions:
        for tr, tc in TEMPLATES[dominant_color]:
            nr = int(r) + int(tr) - 1
            nc = int(c) + int(tc) - 1
            if not (0 <= nr < BLOCK_SIZE and 0 <= nc < BLOCK_SIZE):
                return True
    return False


def _build_position_buckets() -> dict[int, dict[str, list[tuple[tuple[int, int], ...]]]]:
    buckets: dict[int, dict[str, list[tuple[tuple[int, int], ...]]]] = {
        3: {"edge": [], "interior": []},
        4: {"edge": [], "interior": []},
    }
    for count in (3, 4):
        for combo in combinations(ALL_BLOCK_CELLS, count):
            if not _is_far_enough(combo):
                continue
            if not _templates_are_distinguishable(combo):
                continue
            if _has_edge_cell(combo) and _has_interior_cell(combo):
                buckets[count]["edge"].append(combo)
            elif not _has_edge_cell(combo):
                buckets[count]["interior"].append(combo)
    for count in (3, 4):
        assert buckets[count]["edge"], f"missing edge-position layouts for {count} pixels"
        assert buckets[count]["interior"], f"missing interior-position layouts for {count} pixels"
    return buckets


POSITION_BUCKETS = _build_position_buckets()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the 12x12 grid into four blocks, decode each block's dominant color into a template, and expand every foreground pixel with that block-local teal stencil.

    Input structure:
    - The input is a 12x12 black grid partitioned into four 6x6 blocks arranged in a 2x2 layout.
    - Each block contains isolated non-black foreground pixels made from one dominant color and one distractor color.
    - The dominant color in each block is uniquely most frequent and is one of blue, red, or green.

    Rule:
    - Process each 6x6 block independently.
    - Find the unique dominant non-black color in the block, then choose its 3x3 stencil from the task codebook.
    - Replace every non-black pixel in that block, regardless of its own color, with teal cells given by the chosen stencil centered on the pixel, clipping anything that would leave the block; always color the source pixel itself teal.

    Key insight:
    - Foreground pixel colors do not determine their own expansion shape; they only vote for a block-level dominant color, and that dominant color selects the stencil shared by every pixel in the same block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected a 12x12 grid"

    out = np.zeros_like(g)
    for block_index in range(4):
        r0, c0 = _block_origin(block_index)
        block = g[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
        dominant_color = _dominant_non_black(block)
        for r, c in np.argwhere(block != BACKGROUND):
            rr = r0 + int(r)
            cc = c0 + int(c)
            out[rr, cc] = OUTPUT_COLOR
            for tr, tc in TEMPLATES[dominant_color]:
                nr = rr + int(tr) - 1
                nc = cc + int(tc) - 1
                if r0 <= nr < r0 + BLOCK_SIZE and c0 <= nc < c0 + BLOCK_SIZE:
                    out[nr, nc] = OUTPUT_COLOR
    return out


def _sample_dominant_layout() -> list[int]:
    branch = random.random()
    if branch < 0.5:
        doubled = random.choice(DOMINANT_COLORS)
        layout = [1, 2, 3, doubled]
    elif branch < 0.8:
        a, b = random.sample(DOMINANT_COLORS, 2)
        layout = [a, a, b, b]
    else:
        a, b = random.sample(DOMINANT_COLORS, 2)
        layout = [a, a, a, b]
    random.shuffle(layout)
    return layout


def _sample_block_positions(
    counts: tuple[int, int, int, int],
    edge_flags: tuple[bool, bool, bool, bool],
) -> list[tuple[tuple[int, int], ...]]:
    chosen: list[tuple[tuple[int, int], ...]] = []
    used: set[tuple[tuple[int, int], ...]] = set()
    for count, need_edge in zip(counts, edge_flags):
        bucket_name = "edge" if need_edge else "interior"
        pool = POSITION_BUCKETS[int(count)][bucket_name]
        options = [combo for combo in pool if combo not in used]
        combo = random.choice(options if options else pool)
        chosen.append(combo)
        used.add(combo)
    return chosen


def _shared_distractor_color(
    dominants: list[int],
    position_sets: list[tuple[tuple[int, int], ...]],
) -> int:
    signature = 0
    for block_index, (dominant, positions) in enumerate(zip(dominants, position_sets)):
        cell_sum = sum((int(r) * BLOCK_SIZE) + int(c) + 1 for r, c in positions)
        signature += (block_index + 1) * (dominant * 41 + len(positions) * 17 + cell_sum)
    return DISTRACTOR_PALETTE[signature % len(DISTRACTOR_PALETTE)]


def _render_candidate(
    dominants: list[int],
    position_sets: list[tuple[tuple[int, int], ...]],
    distractor_color: int,
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for block_index, (dominant, positions) in enumerate(zip(dominants, position_sets)):
        r0, c0 = _block_origin(block_index)
        distractor_pos = max(positions)
        for r, c in positions:
            grid[r0 + int(r), c0 + int(c)] = distractor_color if (r, c) == distractor_pos else dominant
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    dominant_colors: list[int] = []
    distractor_colors: list[int] = []
    edge_blocks = 0
    clipped_blocks = 0
    block_outputs: list[bytes] = []

    for block_index in range(4):
        r0, c0 = _block_origin(block_index)
        block = grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE]
        positions = tuple(tuple(map(int, coord)) for coord in np.argwhere(block != BACKGROUND))
        if len(positions) not in (3, 4):
            return False
        if not _is_far_enough(positions):
            return False
        if not _templates_are_distinguishable(positions):
            return False

        counts = Counter(int(value) for value in block.flat if int(value) != BACKGROUND)
        if len(counts) != 2:
            return False
        ranked = counts.most_common(2)
        dominant_color = int(ranked[0][0])
        distractor_color = int(ranked[1][0])
        if dominant_color not in DOMINANT_COLORS:
            return False
        if ranked[0][1] not in (2, 3) or ranked[1][1] != 1:
            return False
        if ranked[0][1] <= ranked[1][1]:
            return False

        dominant_colors.append(dominant_color)
        distractor_colors.append(distractor_color)
        if _has_edge_cell(positions):
            edge_blocks += 1
        if _has_clipping(positions, dominant_color):
            clipped_blocks += 1
        block_outputs.append(out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE].tobytes())

    if len(set(dominant_colors)) < 2:
        return False
    if len(set(distractor_colors)) != 1:
        return False
    if edge_blocks < 2 or clipped_blocks < 1:
        return False
    if len(set(block_outputs)) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        dominants = _sample_dominant_layout()
        counts = random.choice(COUNT_PATTERNS)
        edge_flags = random.choice(EDGE_PATTERNS)
        position_sets = _sample_block_positions(counts, edge_flags)
        distractor_color = _shared_distractor_color(dominants, position_sets)
        candidate = _render_candidate(dominants, position_sets, distractor_color)
        if _validate_candidate(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid input grid")
