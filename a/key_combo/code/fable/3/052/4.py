from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = list(range(10))
MAX_ATTEMPTS = 500


def _ring_depths(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    depths = np.minimum(rows, h - 1 - rows)
    depths = np.minimum(depths, cols)
    depths = np.minimum(depths, w - 1 - cols)
    return depths.astype(int)


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.ravel()))
    return int(counts.most_common(1)[0][0])


def _infer_period(seq: list[int]) -> int:
    n = len(seq)
    if n == 0:
        raise ValueError("cannot infer a period from an empty sequence")
    for period in range(1, n):
        if all(seq[i] == seq[i % period] for i in range(n)):
            return period
    return n


def _build_clean_grid(shape: tuple[int, int], ring_colors: list[int]) -> np.ndarray:
    depths = _ring_depths(shape)
    palette = np.asarray(ring_colors, dtype=int)
    return palette[depths]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a noisy, partly occluded concentric-ring grid by voting visible rings and extrapolating the global color cycle.

    Input structure:
    - The whole grid is tiled by one-pixel-thick concentric rectangular rings sharing one center.
    - From the outer ring inward, the true ring colors follow a fixed repeating cycle that contains a repeated color.
    - A single-color rectangular patch whose color is not part of the cycle covers some visible ring arcs and fully hides the innermost few rings; a few other cells are random noise.

    Rule:
    - Use the ring index `min(r, c, H-1-r, W-1-c)` to group cells by concentric ring.
    - For every ring that still has visible arc cells, recover its true color by majority vote over that ring.
    - Treat the innermost suffix whose ring majorities equal the patch color as fully hidden, infer the shortest repeating unit from the visible outer ring colors, and continue that cycle inward to fill the hidden rings.

    Key insight:
    - The noisy pixels and the patch are both corruptions of one global periodic ring sequence, so the solve works by restoring ring colors first and only then fitting the full outer-to-inner cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    depths = _ring_depths(g.shape)
    n_rings = int(depths.max()) + 1

    ring_majorities: list[int] = []
    for ring in range(n_rings):
        ring_majorities.append(_mode_color(g[depths == ring]))

    patch_color = ring_majorities[-1]
    hidden = 1
    while hidden < n_rings and ring_majorities[n_rings - 1 - hidden] == patch_color:
        hidden += 1

    visible_count = n_rings - hidden
    assert visible_count >= 1
    visible_seq = ring_majorities[:visible_count]
    period = _infer_period(visible_seq)
    cycle = visible_seq[:period]

    repaired_ring_colors = np.asarray([cycle[i % period] for i in range(n_rings)], dtype=int)
    return repaired_ring_colors[depths]


def _sample_cycle(period: int) -> list[int] | None:
    for _ in range(200):
        n_unique = random.randint(2, min(4, period))
        base_colors = random.sample(PALETTE, n_unique)
        seq = [random.choice(base_colors) for _ in range(period)]
        if len(set(seq)) >= len(seq):
            continue
        if len(set(seq)) < 2:
            continue
        if _infer_period(seq) != len(seq):
            continue
        return seq
    return None


def _patch_bbox(
    shape: tuple[int, int], hidden_rings: int, partial_rings: int, direction: str
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    n_rings = (min(h, w) + 1) // 2
    start = n_rings - hidden_rings
    inner_r0 = start
    inner_r1 = h - start - 1
    inner_c0 = start
    inner_c1 = w - start - 1

    if direction == "up":
        return inner_r0 - partial_rings, inner_r1, inner_c0, inner_c1
    if direction == "down":
        return inner_r0, inner_r1 + partial_rings, inner_c0, inner_c1
    if direction == "left":
        return inner_r0, inner_r1, inner_c0 - partial_rings, inner_c1
    if direction == "right":
        return inner_r0, inner_r1, inner_c0, inner_c1 + partial_rings
    raise ValueError(f"unknown direction: {direction}")


def _visible_ring_majorities_match(
    grid: np.ndarray, ring_colors: list[int], hidden_rings: int
) -> bool:
    depths = _ring_depths(grid.shape)
    n_rings = len(ring_colors)
    visible_count = n_rings - hidden_rings
    patch_color = _mode_color(grid[depths == (n_rings - 1)])
    for ring in range(visible_count):
        if _mode_color(grid[depths == ring]) != ring_colors[ring]:
            return False
    for ring in range(visible_count, n_rings):
        if _mode_color(grid[depths == ring]) != patch_color:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    min_dim = random.randint(17, 24)
    max_dim = random.randint(min_dim, 30)
    if random.random() < 0.5:
        shape = (min_dim, max_dim)
    else:
        shape = (max_dim, min_dim)

    h, w = shape
    n_rings = (min(h, w) + 1) // 2

    period = random.randint(3, min(5, n_rings - 3))
    if n_rings % period == 0:
        return None

    cycle = _sample_cycle(period)
    if cycle is None:
        return None

    phase = random.randrange(period)
    rotated = cycle[phase:] + cycle[:phase]

    max_hidden = min(3, n_rings - (period + 2))
    if max_hidden < 1:
        return None
    hidden_rings = random.randint(1, max_hidden)
    visible_count = n_rings - hidden_rings
    if visible_count <= period:
        return None

    partial_cap = min(3, visible_count - 2)
    if partial_cap < 1:
        return None
    partial_rings = random.randint(1, partial_cap)
    direction = random.choice(["up", "down", "left", "right"])

    ring_colors = [rotated[i % period] for i in range(n_rings)]
    output_grid = _build_clean_grid(shape, ring_colors)
    input_grid = output_grid.copy()

    r0, r1, c0, c1 = _patch_bbox(shape, hidden_rings, partial_rings, direction)
    patch_mask = np.zeros(shape, dtype=bool)
    patch_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    patch_color_choices = [color for color in PALETTE if color not in set(ring_colors)]
    if not patch_color_choices:
        return None
    patch_color = random.choice(patch_color_choices)

    depths = _ring_depths(shape)
    visible_mask = depths < visible_count
    noise_candidates = np.argwhere(visible_mask & ~patch_mask)
    if len(noise_candidates) == 0:
        return None

    noise_count = random.randint(1, min(8, len(noise_candidates)))
    noise_indices = random.sample(range(len(noise_candidates)), noise_count)
    for idx in noise_indices:
        r, c = map(int, noise_candidates[idx])
        true_color = ring_colors[int(depths[r, c])]
        choices = [color for color in PALETTE if color not in {true_color, patch_color}]
        input_grid[r, c] = random.choice(choices)

    input_grid[patch_mask] = patch_color

    if _infer_period(ring_colors[:visible_count]) != period:
        return None
    if not _visible_ring_majorities_match(input_grid, ring_colors, hidden_rings):
        return None

    try:
        solved = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(solved, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_ATTEMPTS} attempts")
