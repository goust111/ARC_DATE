from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

INNER_H_RANGE = (3, 12)
INNER_W_RANGE = (3, 12)
GEN_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top and left edge color stripes as weaving instructions.

    The output keeps the guide stripes in place and fills the interior so that:
    - even (row + col) crossings show the top-edge warp color
    - odd crossings show the left-edge weft color
    - same-color crossings naturally keep that shared color
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "grid must have an edge row and edge column"

    top_colors = g[0, 1:]
    left_colors = g[1:, 0]

    row_colors = left_colors[:, None]
    col_colors = top_colors[None, :]
    row_ids = np.arange(height - 1, dtype=int)[:, None]
    col_ids = np.arange(width - 1, dtype=int)[None, :]

    even_crossings = (row_ids + col_ids) % 2 == 0
    same_color = row_colors == col_colors

    out = g.copy()
    out[1:, 1:] = np.where(same_color | even_crossings, col_colors, row_colors)
    return out


def _sample_disjoint_pools() -> tuple[list[int], list[int], bool]:
    top_size = random.randint(2, 4)
    left_size = random.randint(2, 4)
    chosen = random.sample(PALETTE, top_size + left_size)
    return chosen[:top_size], chosen[top_size:], False


def _sample_partial_overlap_pools() -> tuple[list[int], list[int], bool]:
    colors = list(PALETTE)
    common_size = random.randint(1, 2)
    common = random.sample(colors, common_size)
    remaining = [c for c in colors if c not in common]

    top_only_size = random.randint(1, 2)
    top_only = random.sample(remaining, top_only_size)
    remaining = [c for c in remaining if c not in top_only]

    left_only_size = random.randint(1, 2)
    left_only = random.sample(remaining, left_only_size)
    return common + top_only, common + left_only, True


def _sample_heavy_overlap_pools() -> tuple[list[int], list[int], bool]:
    shared_size = random.randint(2, 4)
    shared = random.sample(PALETTE, shared_size)
    top_pool = list(shared)
    left_pool = list(shared)

    extras = [c for c in PALETTE if c not in shared]
    if extras and random.random() < 0.35:
        extra = random.choice(extras)
        top_pool.append(extra)
        extras.remove(extra)
    if extras and random.random() < 0.35:
        left_pool.append(random.choice(extras))

    return top_pool, left_pool, True


def _sample_sequence(length: int, pool: list[int]) -> np.ndarray:
    min_unique = 2 if length >= 2 and len(pool) >= 2 else 1

    for _ in range(100):
        style = random.choice(("random", "runs", "motif"))

        if style == "random":
            seq = [random.choice(pool) for _ in range(length)]
        elif style == "runs":
            seq = []
            prev = None
            while len(seq) < length:
                color = random.choice(pool)
                if prev is not None and len(pool) > 1 and color == prev:
                    alt = [c for c in pool if c != prev]
                    color = random.choice(alt)
                run = random.randint(1, min(4, length - len(seq)))
                seq.extend([color] * run)
                prev = color
        else:
            motif_len = random.randint(2, min(4, length))
            motif = [random.choice(pool) for _ in range(motif_len)]
            seq = (motif * ((length + motif_len - 1) // motif_len))[:length]
            if length > motif_len and random.random() < 0.5:
                seq[random.randrange(length)] = random.choice(pool)

        if len(set(seq)) >= min_unique:
            return np.array(seq, dtype=int)

    raise RuntimeError("failed to sample a non-degenerate color sequence")


def _valid_branches(top: np.ndarray, left: np.ndarray, expect_shared: bool) -> bool:
    top_colors = top[None, :]
    left_colors = left[:, None]
    same = left_colors == top_colors
    diff = ~same

    row_ids = np.arange(left.shape[0], dtype=int)[:, None]
    col_ids = np.arange(top.shape[0], dtype=int)[None, :]
    even = (row_ids + col_ids) % 2 == 0

    if not np.any(diff & even):
        return False
    if not np.any(diff & ~even):
        return False

    has_shared = bool(np.any(same))
    if expect_shared != has_shared:
        return False
    if expect_shared and not np.any(diff):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)

    mode = random.choices(
        ("disjoint", "partial", "shared"),
        weights=(2, 1, 1),
        k=1,
    )[0]
    if mode == "disjoint":
        top_pool, left_pool, expect_shared = _sample_disjoint_pools()
    elif mode == "partial":
        top_pool, left_pool, expect_shared = _sample_partial_overlap_pools()
    else:
        top_pool, left_pool, expect_shared = _sample_heavy_overlap_pools()

    top = _sample_sequence(inner_w, top_pool)
    left = _sample_sequence(inner_h, left_pool)
    if not _valid_branches(top, left, expect_shared):
        return None

    grid = np.full((inner_h + 1, inner_w + 1), BG, dtype=int)
    grid[0, 1:] = top
    grid[1:, 0] = left

    counts = np.bincount(grid.ravel(), minlength=max(PALETTE) + 1)
    if int(np.argmax(counts)) != BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    woven = out[1:, 1:]
    if woven.shape != (inner_h, inner_w):
        return None
    if np.any(woven == BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving input")
