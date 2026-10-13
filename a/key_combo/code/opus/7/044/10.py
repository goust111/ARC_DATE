from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8
PARTITION_COLORS = (BLUE, RED, GREEN, YELLOW)
MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_col(grid: np.ndarray) -> int:
    candidates = np.flatnonzero(np.all(grid == GREY, axis=0))
    for col in candidates:
        left = grid[:, :col]
        right = grid[:, col + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        if np.any(left == TEAL) and np.any(np.isin(right, PARTITION_COLORS)):
            return int(col)
    raise AssertionError("expected a full-height grey separator column")


def _extract_template_mask(left_region: np.ndarray) -> np.ndarray:
    box = _bbox(left_region == TEAL)
    assert box is not None, "expected teal tile on the left side"
    r0, c0, r1, c1 = box
    tile = left_region[r0 : r1 + 1, c0 : c1 + 1]
    vals = set(map(int, np.unique(tile)))
    assert vals <= {BLACK, TEAL}, "left tile should only use black and teal"
    return tile == TEAL


def _monochrome_rectangles(region: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    h, w = region.shape
    visited = np.zeros((h, w), dtype=bool)
    rectangles: list[tuple[int, tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if color == BLACK or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(region[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            block = region[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(block == color), "each partition must be a solid rectangle"
            rectangles.append((color, (int(r0), int(c0), int(r1), int(c1))))

    assert rectangles, "expected rectangular partitions on the right side"
    return rectangles


def _downsample_partition(region: np.ndarray) -> np.ndarray:
    rectangles = _monochrome_rectangles(region)
    row_starts = sorted({box[0] for _, box in rectangles})
    col_starts = sorted({box[1] for _, box in rectangles})
    compact = np.zeros((len(row_starts), len(col_starts)), dtype=int)
    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            compact[i, j] = int(region[r, c])
    return compact


def _render_output(template_mask: np.ndarray, compact_colors: np.ndarray) -> np.ndarray:
    tile_h, tile_w = template_mask.shape
    rows, cols = compact_colors.shape
    out = np.zeros((rows * tile_h, cols * tile_w), dtype=int)
    for r in range(rows):
        for c in range(cols):
            block = out[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w]
            block[template_mask] = int(compact_colors[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a repeated tile pattern using the colors sampled from a rectangular partition grid.

    Input structure:
    - A full-height grey separator column splits the grid into a left template area and a right partition area.
    - Left of the separator is one teal-on-black 3x3 or 4x4 tile pattern with possible black margin around it.
    - Right of the separator is a packed grid of differently sized solid-color rectangles using blue, red, green, and yellow.

    Rule:
    - Extract the teal tile from the left side by cropping to its bounding box; teal cells are the template foreground and black cells stay black.
    - Crop the right side to the bounding box of the partition grid, recover each solid rectangle, and collect the sorted unique rectangle top-left rows and columns.
    - Sample the partition color at every recovered top-left coordinate to form a compact color grid, then replace each compact cell by one copy of the template recolored to that sampled color.

    Key insight:
    - The left half provides only shape, while the right half provides only colors; the output appears only after combining the template mask with the partition downsampling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    template_mask = _extract_template_mask(left)

    right_box = _bbox(np.isin(right, PARTITION_COLORS))
    assert right_box is not None, "expected a colored partition grid on the right side"
    r0, c0, r1, c1 = right_box
    partition_region = right[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(np.isin(partition_region, PARTITION_COLORS)), "partition area must be fully tiled by palette colors"

    compact = _downsample_partition(partition_region)
    return _render_output(template_mask, compact)


def _sample_template_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.choice([3, 4]))
        w = int(rng.choice([3, 4]))
        mask = rng.random((h, w)) < rng.uniform(0.35, 0.7)
        if not mask.any() or mask.all():
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if int(mask.sum()) < 3 or int((~mask).sum()) < 2:
            continue
        return mask
    raise RuntimeError("failed to sample a valid tile template")


def _sample_sizes(rng: np.random.Generator, count: int, total_cap: int) -> list[int]:
    for _ in range(200):
        sizes = rng.integers(2, 6, size=count)
        if int(sizes.sum()) <= total_cap:
            return [int(x) for x in sizes]
    raise RuntimeError("failed to sample partition sizes within cap")


def _sample_compact_colors(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    for _ in range(200):
        compact = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                choices = list(PARTITION_COLORS)
                if r > 0 and int(compact[r - 1, c]) in choices:
                    choices.remove(int(compact[r - 1, c]))
                if c > 0 and int(compact[r, c - 1]) in choices:
                    choices.remove(int(compact[r, c - 1]))
                compact[r, c] = int(rng.choice(choices))
        if len(set(map(int, compact.flatten()))) >= min(2, rows * cols):
            return compact
    raise RuntimeError("failed to sample visible partition colors")


def _render_partition(compact: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    part_h = sum(row_heights)
    part_w = sum(col_widths)
    region = np.zeros((part_h, part_w), dtype=int)

    r0 = 0
    for i, height in enumerate(row_heights):
        c0 = 0
        for j, width in enumerate(col_widths):
            region[r0 : r0 + height, c0 : c0 + width] = int(compact[i, j])
            c0 += width
        r0 += height
    return region


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    template_mask = _sample_template_mask(rng)
    tile_h, tile_w = template_mask.shape

    rows = int(rng.integers(2, 6))
    cols = int(rng.integers(2, 6))
    row_heights = _sample_sizes(rng, rows, total_cap=17)
    col_widths = _sample_sizes(rng, cols, total_cap=17)
    compact = _sample_compact_colors(rng, rows, cols)
    partition_region = _render_partition(compact, row_heights, col_widths)

    tile_top = int(rng.integers(1, 4))
    tile_left = int(rng.integers(1, 3))
    tile_right_gap = int(rng.integers(1, 3))
    tile_bottom = int(rng.integers(1, 4))

    part_top = int(rng.integers(1, 4))
    part_left_gap = int(rng.integers(1, 3))
    part_right_margin = int(rng.integers(1, 3))
    part_bottom = int(rng.integers(1, 4))

    sep_col = tile_left + tile_w + tile_right_gap
    part_left = sep_col + 1 + part_left_gap
    h = max(tile_top + tile_h + tile_bottom, part_top + partition_region.shape[0] + part_bottom)
    w = part_left + partition_region.shape[1] + part_right_margin

    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[:, sep_col] = GREY

    tile_block = grid[tile_top : tile_top + tile_h, tile_left : tile_left + tile_w]
    tile_block[template_mask] = TEAL

    grid[
        part_top : part_top + partition_region.shape[0],
        part_left : part_left + partition_region.shape[1],
    ] = partition_region

    expected = _render_output(template_mask, compact)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 10 input")
