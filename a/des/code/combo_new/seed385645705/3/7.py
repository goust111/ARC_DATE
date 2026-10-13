from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _to_blocks(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w and h % 2 == 0, "expected an even square grid"
    assert h >= 4, "expected at least a 2x2 array of blocks"

    blocks = g[0::2, 0::2].copy()
    assert np.array_equal(g[0::2, 1::2], blocks), "each 2x2 block must be uniform"
    assert np.array_equal(g[1::2, 0::2], blocks), "each 2x2 block must be uniform"
    assert np.array_equal(g[1::2, 1::2], blocks), "each 2x2 block must be uniform"
    return blocks


def _anchor_mask(blocks: np.ndarray) -> np.ndarray:
    n = blocks.shape[0]
    assert n >= 2, "expected four distinct corner blocks"
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    counts = Counter(map(int, blocks.ravel()))
    corner_colors = [int(blocks[r, c]) for r, c in corners]
    assert len(set(corner_colors)) == 4, "corner anchor colors must be distinct"
    for color in corner_colors:
        assert counts[color] == 1, "each anchor color may appear only in its corner"

    mask = np.zeros_like(blocks, dtype=bool)
    for r, c in corners:
        mask[r, c] = True
    return mask


def _vote_color(blocks: np.ndarray, r: int, c: int) -> tuple[int, bool]:
    n = blocks.shape[0]
    counts: dict[int, int] = {}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < n and 0 <= cc < n:
            color = int(blocks[rr, cc])
            counts[color] = counts.get(color, 0) + 1

    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    if len(winners) == 1:
        return int(winners[0]), False
    return int(blocks[r, c]), True


def _vote_round(blocks: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    out = blocks.copy()
    n = blocks.shape[0]
    for r in range(n):
        for c in range(n):
            if anchors[r, c]:
                continue
            out[r, c] = _vote_color(blocks, r, c)[0]
    return out


def _vote_round_stats(blocks: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    out = blocks.copy()
    stats = {"changed": 0, "tied": 0, "majority": 0}
    n = blocks.shape[0]
    for r in range(n):
        for c in range(n):
            if anchors[r, c]:
                continue
            new_color, tied = _vote_color(blocks, r, c)
            if tied:
                stats["tied"] += 1
            else:
                stats["majority"] += 1
            if new_color != int(blocks[r, c]):
                stats["changed"] += 1
            out[r, c] = new_color
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 2x2 block by two rounds of four-neighbor majority voting.

    Input structure:
    - The grid is an even square filled by non-overlapping uniform 2x2 blocks.
    - These blocks form a smaller square array of superpixels with no background.
    - The four corner superpixels have distinct anchor colors, and each anchor
      color appears only in its own corner block.

    Rule:
    - Treat every 2x2 block as one superpixel.
    - In each synchronous round, every non-anchor superpixel looks at its up,
      down, left, and right superpixel neighbors that exist inside the grid.
    - If one neighbor color occurs strictly more often than all others, recolor
      the superpixel to that color; if there is a tie, keep its current color.
    - The four anchor corner superpixels never change, and the vote is applied
      for exactly two rounds.

    Key insight:
    - The operating unit is the 2x2 superpixel, and the visible smoothing is a
      repeated local majority vote; one round is not enough to match the output.
    """
    blocks = _to_blocks(grid)
    anchors = _anchor_mask(blocks)
    for _ in range(2):
        blocks = _vote_round(blocks, anchors)
    return np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)


COLOR_POOL = list(range(10))
SUPER_SIZE_RANGE = (6, 15)
ORDINARY_COLOR_COUNTS = (2, 3, 4)
GENERATION_ATTEMPTS = 500


def _corner_cells(n: int) -> set[tuple[int, int]]:
    return {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}


def _nearest_seed_fill(n: int, colors: list[int]) -> np.ndarray:
    seeds: list[tuple[int, int, int]] = []
    for color in colors:
        for _ in range(random.choice([1, 2])):
            seeds.append((random.randrange(n), random.randrange(n), int(color)))

    blocks = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            best_dist = None
            best_colors: list[int] = []
            for sr, sc, color in seeds:
                dist = abs(r - sr) + abs(c - sc)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_colors = [int(color)]
                elif dist == best_dist:
                    best_colors.append(int(color))
            blocks[r, c] = random.choice(best_colors)
    return blocks


def _add_noise(blocks: np.ndarray, colors: list[int]) -> None:
    n = blocks.shape[0]
    corners = _corner_cells(n)
    noise_rate = random.uniform(0.22, 0.42)
    for r in range(n):
        for c in range(n):
            if (r, c) in corners:
                continue
            if random.random() < noise_rate:
                current = int(blocks[r, c])
                choices = [color for color in colors if int(color) != current]
                blocks[r, c] = random.choice(choices or colors)

    for _ in range(random.randint(n // 2, n + 2)):
        r = random.randrange(n)
        c = random.randrange(n)
        if (r, c) not in corners:
            blocks[r, c] = random.choice(colors)


def _place_anchors(blocks: np.ndarray, anchors: list[int]) -> None:
    n = blocks.shape[0]
    for (r, c), color in zip(
        [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)],
        anchors,
    ):
        blocks[r, c] = int(color)


def _is_useful_candidate(blocks: np.ndarray, ordinary_colors: list[int]) -> bool:
    anchors = _anchor_mask(blocks)
    non_anchor = ~anchors
    present = {int(v) for v in blocks[non_anchor].ravel()}
    if present != set(map(int, ordinary_colors)):
        return False

    one, stats1 = _vote_round_stats(blocks, anchors)
    two, stats2 = _vote_round_stats(one, anchors)

    if stats1["changed"] < 3 or stats2["changed"] < 1:
        return False
    if stats1["tied"] + stats2["tied"] < 2:
        return False
    if not np.any(one != blocks) or not np.any(two != one):
        return False
    if len({int(v) for v in two[non_anchor].ravel()}) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*SUPER_SIZE_RANGE)
    ordinary_count = random.choice(ORDINARY_COLOR_COUNTS)
    colors = random.sample(COLOR_POOL, 4 + ordinary_count)
    anchor_colors = colors[:4]
    ordinary_colors = colors[4:]

    blocks = _nearest_seed_fill(n, ordinary_colors)
    _add_noise(blocks, ordinary_colors)
    _place_anchors(blocks, anchor_colors)

    try:
        if not _is_useful_candidate(blocks, ordinary_colors):
            return None
        grid = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)
        solved = main(grid)
    except Exception:
        return None

    if solved.shape != grid.shape or max(solved.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 7")
