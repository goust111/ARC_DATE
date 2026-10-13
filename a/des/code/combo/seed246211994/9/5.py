from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GRID_SHAPES = [(h, w) for h in range(2, 7) for w in range(2, 7) if 4 <= h * w <= 24]
MAX_REGIONS = 7
MAX_COLORS = 5
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each input cell by a full-grid block showing the mask of its color.

    For an input grid of shape H x W, the output has shape (H*H) x (W*W). The
    block at position (r, c) is the global mask of color grid[r, c]: cells
    equal to that color keep it, and all other cells become black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    colors = g[:, :, None, None]
    same_color = g[None, None, :, :] == colors
    blocks = np.where(same_color, colors, BLACK)
    return blocks.transpose(0, 2, 1, 3).reshape(h * h, w * w)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _choose_region_colors(n_regions: int) -> list[int]:
    assert n_regions >= 3
    n_colors = random.randint(2, min(MAX_COLORS, n_regions - 1, len(PALETTE)))
    palette = random.sample(PALETTE, n_colors)

    region_colors = palette[:]
    region_colors.extend(random.choice(palette) for _ in range(n_regions - n_colors))
    random.shuffle(region_colors)

    counts = [region_colors.count(color) for color in set(region_colors)]
    assert len(counts) >= 2 and max(counts) >= 2
    return region_colors


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    n_regions = int(labels.max()) + 1
    colors = _choose_region_colors(n_regions)
    grid = np.empty(labels.shape, dtype=int)
    for label, color in enumerate(colors):
        grid[labels == label] = color
    return grid


def _grow_regions(shape: tuple[int, int], n_regions: int) -> np.ndarray:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = random.sample(cells, n_regions)

    labels = np.full((h, w), -1, dtype=int)
    frontier: list[tuple[int, int, int]] = []
    frontier_set: set[tuple[int, int, int]] = set()

    def add_frontier(r: int, c: int, label: int) -> None:
        item = (r, c, label)
        if labels[r, c] == -1 and item not in frontier_set:
            frontier.append(item)
            frontier_set.add(item)

    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label

    for label, (r, c) in enumerate(seeds):
        for rr, cc in _neighbors4(r, c, h, w):
            add_frontier(rr, cc, label)

    while frontier:
        idx = random.randrange(len(frontier))
        r, c, label = frontier.pop(idx)
        frontier_set.remove((r, c, label))
        if labels[r, c] != -1:
            continue

        labels[r, c] = label
        for rr, cc in _neighbors4(r, c, h, w):
            add_frontier(rr, cc, label)

    assert np.all(labels >= 0)
    return labels


def _split_rectangles(shape: tuple[int, int], n_regions: int) -> np.ndarray | None:
    h, w = shape
    rects: list[tuple[int, int, int, int]] = [(0, 0, h, w)]

    while len(rects) < n_regions:
        splittable = [i for i, (_r0, _c0, rh, rw) in enumerate(rects) if rh > 1 or rw > 1]
        if not splittable:
            return None

        idx = random.choice(splittable)
        r0, c0, rh, rw = rects.pop(idx)

        axes = []
        if rh > 1:
            axes.append("h")
        if rw > 1:
            axes.append("w")
        axis = random.choice(axes)

        if axis == "h":
            cut = random.randint(1, rh - 1)
            rects.extend([(r0, c0, cut, rw), (r0 + cut, c0, rh - cut, rw)])
        else:
            cut = random.randint(1, rw - 1)
            rects.extend([(r0, c0, rh, cut), (r0, c0 + cut, rh, rw - cut)])

    labels = np.empty((h, w), dtype=int)
    for label, (r0, c0, rh, rw) in enumerate(rects):
        labels[r0 : r0 + rh, c0 : c0 + rw] = label
    return labels


def _build_tiled_grid(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    tile_shapes = [
        (th, tw)
        for th in range(1, min(3, h) + 1)
        for tw in range(1, min(3, w) + 1)
        if th * tw >= 2 and (th < h or tw < w)
    ]
    th, tw = random.choice(tile_shapes)

    tile_area = th * tw
    n_colors = random.randint(2, min(4, tile_area, len(PALETTE)))
    colors = random.sample(PALETTE, n_colors)
    entries = colors[:]
    entries.extend(random.choice(colors) for _ in range(tile_area - n_colors))
    random.shuffle(entries)

    tile = np.array(entries, dtype=int).reshape(th, tw)
    reps_h = (h + th - 1) // th
    reps_w = (w + tw - 1) // tw
    return np.tile(tile, (reps_h, reps_w))[:h, :w].copy()


def _build_growth_grid(shape: tuple[int, int]) -> np.ndarray:
    area = shape[0] * shape[1]
    n_regions = random.randint(3, min(MAX_REGIONS, area - 1))
    labels = _grow_regions(shape, n_regions)
    return _colorize_labels(labels)


def _build_rect_grid(shape: tuple[int, int]) -> np.ndarray | None:
    area = shape[0] * shape[1]
    n_regions = random.randint(3, min(MAX_REGIONS, area - 1))
    labels = _split_rectangles(shape, n_regions)
    if labels is None:
        return None
    return _colorize_labels(labels)


def _is_valid_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or grid.size == 0:
        return False
    if np.any(grid == BLACK):
        return False

    colors, counts = np.unique(grid, return_counts=True)
    if colors.size < 2:
        return False
    if int(counts.max()) < 2:
        return False

    out = main(grid)
    h, w = grid.shape
    if out.shape != (h * h, w * w):
        return False
    if not np.any(out == BLACK):
        return False
    if np.unique(out[out != BLACK]).size < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = random.choice(GRID_SHAPES)
    builder = random.choice((_build_growth_grid, _build_rect_grid, _build_tiled_grid))
    grid = builder(shape)
    if grid is None:
        return None
    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
