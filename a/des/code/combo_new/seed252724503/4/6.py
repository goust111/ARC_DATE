from __future__ import annotations

from collections import deque
import numpy as np


_BG = 0
_WALL = 3
_SEED_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
_LAYOUT_SHAPES = ((2, 2), (2, 3), (3, 2))
_MODES = ("balanced", "balanced", "wide", "tall")
_BALANCED_VALUES = np.arange(3, 9, dtype=int)
_SMALL_RANGE = (3, 6)
_LARGE_RANGE = (6, 9)
_MAX_SEEDS_PER_REGION = 5
_MAX_RETRIES = 300
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_regions(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[tuple[int, int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c
            area = 0

            while queue:
                rr, cc = queue.popleft()
                area += 1
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            regions.append((r0, c0, r1, c1, area))

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each region's seeds into full stripes determined by rectangle shape and seed count.

    Input structure:
    - Black background with a one-pixel-thick Green outer border and Green horizontal/vertical wall lines.
    - The Green walls partition the interior into disjoint rectangular regions.
    - Each region contains one or more same-color seed pixels; seed colors are non-black and non-Green.

    Rule:
    - Keep the Green border and all Green walls unchanged.
    - For each rectangular region, count its seed pixels as K and compare region width with region height.
    - If width is at least height, draw a full horizontal stripe on rows 0, K, 2K, ... within that region; otherwise draw full vertical stripes on columns 0, K, 2K, ....
    - Clear every other non-wall cell in the region to black.

    Key insight:
    - Two independent clues must be combined: region aspect ratio decides stripe direction, while the number of seed pixels decides stripe spacing.
    """
    g = np.asarray(grid, dtype=int)
    wall_color = int(g[0, 0])
    out = g.copy()

    for r0, c0, r1, c1, _area in _find_regions(g != wall_color):
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_mask = (patch != _BG) & (patch != wall_color)
        assert np.any(seed_mask), "each region must contain at least one seed"

        seed_colors = np.unique(patch[seed_mask])
        assert seed_colors.size == 1, "all seeds in one region must share a color"

        color = int(seed_colors[0])
        k = int(np.count_nonzero(seed_mask))
        h, w = patch.shape

        out[r0 : r1 + 1, c0 : c1 + 1] = _BG
        if w >= h:
            for dr in range(0, h, k):
                out[r0 + dr, c0 : c1 + 1] = color
        else:
            for dc in range(0, w, k):
                out[r0 : r1 + 1, c0 + dc] = color

    return out


def _sample_balanced_dimensions(
    shape: tuple[int, int], rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    a, b = sorted(int(v) for v in rng.choice(_BALANCED_VALUES, size=2, replace=False))

    if shape == (2, 2):
        rows = [a, b]
        cols = [a, b]
    elif shape == (2, 3):
        rows = [a, b]
        cols = [a, a, b] if rng.random() < 0.5 else [a, b, b]
    else:
        rows = [a, a, b] if rng.random() < 0.5 else [a, b, b]
        cols = [a, b]

    rng.shuffle(rows)
    rng.shuffle(cols)
    return rows, cols


def _sample_monotone_dimensions(
    shape: tuple[int, int], wide: bool, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    nr, nc = shape
    if wide:
        rows = [int(v) for v in rng.integers(_SMALL_RANGE[0], _SMALL_RANGE[1], size=nr)]
        cols = [int(v) for v in rng.integers(_LARGE_RANGE[0], _LARGE_RANGE[1], size=nc)]
    else:
        rows = [int(v) for v in rng.integers(_LARGE_RANGE[0], _LARGE_RANGE[1], size=nr)]
        cols = [int(v) for v in rng.integers(_SMALL_RANGE[0], _SMALL_RANGE[1], size=nc)]
    return rows, cols


def _build_partition(rows: list[int], cols: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    grid_h = sum(rows) + len(rows) + 1
    grid_w = sum(cols) + len(cols) + 1
    grid = np.full((grid_h, grid_w), _BG, dtype=int)

    row_edges = [0]
    pos = 0
    for size in rows:
        pos += size + 1
        row_edges.append(pos)

    col_edges = [0]
    pos = 0
    for size in cols:
        pos += size + 1
        col_edges.append(pos)

    for edge in row_edges:
        grid[edge, :] = _WALL
    for edge in col_edges:
        grid[:, edge] = _WALL

    regions: list[tuple[int, int, int, int]] = []
    for i in range(len(rows)):
        for j in range(len(cols)):
            regions.append(
                (
                    row_edges[i] + 1,
                    col_edges[j] + 1,
                    row_edges[i + 1] - 1,
                    col_edges[j + 1] - 1,
                )
            )

    return grid, regions


def _max_seed_count(h: int, w: int) -> int:
    return min(_MAX_SEEDS_PER_REGION, max(1, (h * w) // 3))


def _sample_seed_counts(
    regions: list[tuple[int, int, int, int]], rng: np.random.Generator
) -> list[int]:
    max_counts = [_max_seed_count(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in regions]

    for _ in range(64):
        counts = [int(rng.integers(1, limit + 1)) for limit in max_counts]
        if max(counts) > 1 and len(set(counts)) >= 2:
            return counts

    counts = [1] * len(regions)
    richest = max(range(len(regions)), key=max_counts.__getitem__)
    counts[richest] = min(2, max_counts[richest])
    if len(regions) > 1:
        counts[(richest + 1) % len(regions)] = min(3, max_counts[(richest + 1) % len(regions)])
    return counts


def _stripe_mask(h: int, w: int, k: int, horizontal: bool) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if horizontal:
        mask[::k, :] = True
    else:
        mask[:, ::k] = True
    return mask


def _sample_seed_positions(
    h: int, w: int, k: int, horizontal: bool, rng: np.random.Generator
) -> list[tuple[int, int]]:
    stripe = _stripe_mask(h, w, k, horizontal)
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    stripe_cells = [(r, c) for r in range(h) for c in range(w) if stripe[r, c]]
    off_cells = [(r, c) for r in range(h) for c in range(w) if not stripe[r, c]]

    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()

    if off_cells:
        cell = off_cells[int(rng.integers(len(off_cells)))]
        chosen.append(cell)
        chosen_set.add(cell)

    if stripe_cells and len(chosen) < k and k >= 2:
        cell = stripe_cells[int(rng.integers(len(stripe_cells)))]
        if cell not in chosen_set:
            chosen.append(cell)
            chosen_set.add(cell)

    remaining = [cell for cell in all_cells if cell not in chosen_set]
    need = k - len(chosen)
    if need > 0:
        picks = rng.choice(len(remaining), size=need, replace=False)
        for idx in np.atleast_1d(picks):
            cell = remaining[int(idx)]
            chosen.append(cell)
            chosen_set.add(cell)

    return chosen


def _gen_attempt(rng: np.random.Generator) -> np.ndarray:
    mode = _MODES[int(rng.integers(len(_MODES)))]
    shape = _LAYOUT_SHAPES[int(rng.integers(len(_LAYOUT_SHAPES)))]

    if mode == "balanced":
        rows, cols = _sample_balanced_dimensions(shape, rng)
    elif mode == "wide":
        rows, cols = _sample_monotone_dimensions(shape, wide=True, rng=rng)
    else:
        rows, cols = _sample_monotone_dimensions(shape, wide=False, rng=rng)

    grid, regions = _build_partition(rows, cols)
    colors = [int(v) for v in rng.choice(_SEED_COLORS, size=len(regions), replace=False)]
    counts = _sample_seed_counts(regions, rng)

    for (r0, c0, r1, c1), color, k in zip(regions, colors, counts):
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        horizontal = w >= h
        positions = _sample_seed_positions(h, w, k, horizontal, rng)
        for dr, dc in positions:
            grid[r0 + dr, c0 + dc] = color

    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    if not (np.all(grid[0, :] == _WALL) and np.all(grid[-1, :] == _WALL)):
        return False
    if not (np.all(grid[:, 0] == _WALL) and np.all(grid[:, -1] == _WALL)):
        return False

    regions = _find_regions(grid != _WALL)
    if len(regions) < 4:
        return False

    seed_counts: list[int] = []
    for r0, c0, r1, c1, area in regions:
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if area != h * w:
            return False

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        seed_mask = (patch != _BG) & (patch != _WALL)
        if not np.any(seed_mask):
            return False

        seed_colors = np.unique(patch[seed_mask])
        if seed_colors.size != 1:
            return False

        seed_counts.append(int(np.count_nonzero(seed_mask)))
    if max(seed_counts) <= 1 or len(set(seed_counts)) < 2:
        return False

    if np.array_equal(grid, main(grid)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(_RNG)
        if _validate_generated_grid(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_MAX_RETRIES} attempts")
