"""ARC task: cyclically borrowed palettes in vertical rectangular partitions."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _mode(values: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, np.asarray(values).ravel()))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _partition_bounds(grid: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    """Find the full-height wall columns and inclusive non-wall column spans."""
    h, w = grid.shape
    background = _mode(grid)
    candidates: list[tuple[int, int, list[int]]] = []
    for color in map(int, np.unique(grid)):
        if color == background:
            continue
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if cols:
            candidates.append((len(cols), color, cols))
    assert candidates, "expected at least one full-height wall column"
    count, wall, wall_cols = max(candidates, key=lambda item: item[0])
    assert count >= 2, "expected several left-to-right partitions"
    assert all(b > a + 1 for a, b in zip(wall_cols, wall_cols[1:])), \
        "wall lines must be single separated columns"

    cuts = [-1, *wall_cols, w]
    bounds = [(cuts[i] + 1, cuts[i + 1] - 1) for i in range(len(cuts) - 1)]
    assert len(bounds) >= 3 and all(c0 <= c1 for c0, c1 in bounds)
    return wall, bounds


def _read_palettes(
    grid: np.ndarray, bounds: list[tuple[int, int]]
) -> list[list[int]]:
    """Read one depth-ordered marker palette from each rectangular partition."""
    h = grid.shape[0]
    palettes: list[list[int]] = []
    for c0, c1 in bounds:
        region = grid[:, c0 : c1 + 1]
        base = _mode(region)
        by_depth: dict[int, int] = {}
        for r, local_c in np.argwhere(region != base):
            r, local_c = int(r), int(local_c)
            depth = min(r, h - 1 - r, local_c, c1 - c0 - local_c)
            assert depth not in by_depth, "marker depths must be unique within a partition"
            by_depth[depth] = int(region[r, local_c])
        assert by_depth, "each partition needs marker pixels"

        palette = [by_depth[d] for d in sorted(by_depth)]
        if 0 not in by_depth:
            palette.insert(0, base)
        assert len(palette) >= 2
        palettes.append(palette)
    return palettes


def main(grid: np.ndarray) -> np.ndarray:
    """Fill every partition with concentric bands from its cyclic right neighbor's palette.

    Input structure:
    - Full-height wall lines split a common-base grid into three or more rectangular
      partitions arranged from left to right.  Each partition contains isolated,
      differently colored marker pixels at unique Chebyshev depths from its boundary.

    Rule:
    - In each partition, order marker colors by their distance from that partition's
      boundary; prepend its base color only when depth zero has no marker.  Partition
      i is then painted with partition (i+1)'s palette, wrapping the last partition to
      the first, using palette[boundary_distance modulo palette_length].  Walls stay.

    Key insight:
    - Marker palettes do not paint their own regions: the complete list of palettes is
      handed one partition to the left in a cyclic shift before concentric filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, _w = g.shape
    wall, bounds = _partition_bounds(g)
    palettes = _read_palettes(g, bounds)

    out = g.copy()
    for i, (c0, c1) in enumerate(bounds):
        palette = palettes[(i + 1) % len(palettes)]
        for r in range(h):
            for c in range(c0, c1 + 1):
                depth = min(r, h - 1 - r, c - c0, c1 - c)
                out[r, c] = palette[depth % len(palette)]
    out[:, [c for c in range(g.shape[1]) if np.all(g[:, c] == wall)]] = wall
    return out


_COLORS = tuple(range(10))
_MAX_ATTEMPTS = 300


def _ring_cells(h: int, width: int, depth: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(h)
        for c in range(width)
        if min(r, h - 1 - r, c, width - 1 - c) == depth
    ]


def _marker_positions(
    rng: random.Random, h: int, width: int, depths: list[int]
) -> list[tuple[int, int]] | None:
    """Choose visibly isolated positions, with depth-zero markers away from corners."""
    chosen: list[tuple[int, int]] = []
    for depth in depths:
        candidates = _ring_cells(h, width, depth)
        if depth == 0:
            candidates = [
                p for p in candidates
                if min(p[0], h - 1 - p[0]) == 0 and 1 < p[1] < width - 2
            ]
        rng.shuffle(candidates)
        valid = [
            p for p in candidates
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 2 for q in chosen)
        ]
        if not valid:
            return None
        chosen.append(rng.choice(valid))
    return chosen


def _render_known(
    grid: np.ndarray,
    bounds: list[tuple[int, int]],
    palettes: list[list[int]],
    shift: int,
) -> np.ndarray:
    """Independent generator oracle using its known layout and palettes."""
    out = grid.copy()
    h = grid.shape[0]
    for i, (c0, c1) in enumerate(bounds):
        palette = palettes[(i + shift) % len(palettes)]
        for r in range(h):
            for c in range(c0, c1 + 1):
                d = min(r, h - 1 - r, c - c0, c1 - c)
                out[r, c] = palette[d % len(palette)]
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_parts = rng.choice((3, 4))
    if n_parts == 3:
        widths = [rng.randint(7, 9) for _ in range(n_parts)]
    else:
        widths = [rng.randint(6, 7) for _ in range(n_parts)]
    total_w = sum(widths) + n_parts - 1
    if total_w > 30:
        return None
    h = rng.randint(11, 17)

    base, wall = rng.sample(_COLORS, 2)
    marker_colors = [c for c in _COLORS if c not in (base, wall)]
    grid = np.full((h, total_w), base, dtype=int)

    bounds: list[tuple[int, int]] = []
    cursor = 0
    for i, width in enumerate(widths):
        bounds.append((cursor, cursor + width - 1))
        cursor += width
        if i + 1 < n_parts:
            grid[:, cursor] = wall
            cursor += 1

    # Every example contains both placeholder branches; randomize which regions use each.
    flags = [True, False] + [rng.choice((True, False)) for _ in range(n_parts - 2)]
    rng.shuffle(flags)
    palettes: list[list[int]] = []
    for (c0, c1), has_boundary_marker in zip(bounds, flags):
        max_depth = min((h - 1) // 2, (c1 - c0) // 2)
        if has_boundary_marker:
            palette_len = rng.choice((2, 3))
            marker_count = palette_len
            other_depths = rng.sample(range(1, max_depth + 1), marker_count - 1)
            depths = [0, *sorted(other_depths)]
            colors = rng.sample(marker_colors, marker_count)
            palette = colors.copy()
        else:
            palette_len = 3
            marker_count = 2
            depths = sorted(rng.sample(range(1, max_depth + 1), marker_count))
            colors = rng.sample(marker_colors, marker_count)
            palette = [base, *colors]

        positions = _marker_positions(rng, h, c1 - c0 + 1, depths)
        if positions is None:
            return None
        for (r, local_c), color in zip(positions, colors):
            grid[r, c0 + local_c] = color
        palettes.append(palette)

    # Each hand-off must visibly differ from painting locally, including wraparound.
    for i, (c0, c1) in enumerate(bounds):
        max_depth = min((h - 1) // 2, (c1 - c0) // 2)
        local = [palettes[i][d % len(palettes[i])] for d in range(max_depth + 1)]
        borrowed = [
            palettes[(i + 1) % n_parts][d % len(palettes[(i + 1) % n_parts])]
            for d in range(max_depth + 1)
        ]
        if local == borrowed or len(set(borrowed)) < 2:
            return None

    expected = _render_known(grid, bounds, palettes, shift=1)
    local_answer = _render_known(grid, bounds, palettes, shift=0)
    if np.array_equal(expected, local_answer) or np.array_equal(expected, grid):
        return None
    try:
        solved = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate instance satisfying all solver assumptions."""
    rng = random.Random()
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cyclic borrowed-palette instance")
