from __future__ import annotations

import random

import numpy as np


BG = 0
PANEL_COLOR = 8
SEED_COLOR = 9
LEGEND_W = 5
PALETTE_POOL = (1, 2, 3, 4, 5, 6, 7)
GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    """Minimal arc_utils-style 4-neighborhood helper using (r, c) coordinates."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_by_color(grid: np.ndarray, colors: list[int]) -> list[dict]:
    """Small 4-connected monochrome component finder following arc_utils conventions."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for color in colors:
        points = np.argwhere(grid == color)
        for r0, c0 in points:
            r0 = int(r0)
            c0 = int(c0)
            if seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if (
                        in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return out


def _read_palette(grid: np.ndarray) -> list[int]:
    panel = np.asarray(grid[:, :LEGEND_W], dtype=int)
    assert panel.shape[1] == LEGEND_W, "legend panel must occupy the left 5 columns"
    assert np.all(panel[:, 0] == PANEL_COLOR) and np.all(panel[:, -1] == PANEL_COLOR)

    colors = sorted(
        int(color)
        for color in np.unique(panel)
        if int(color) not in (BG, PANEL_COLOR)
    )
    components = _components_by_color(panel, colors)
    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))

    palette = [int(comp["color"]) for comp in components]
    assert 3 <= len(palette) <= 5
    assert len(palette) == len(set(palette))
    return palette


def _read_seeds(grid: np.ndarray) -> list[tuple[int, int]]:
    canvas = np.asarray(grid[:, LEGEND_W:], dtype=int)
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(canvas == SEED_COLOR)]
    assert 2 <= len(seeds) <= 3
    return seeds


def _chebyshev_partition(
    shape: tuple[int, int], seeds: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_seeds = np.asarray(sorted(seeds), dtype=int)
    rows, cols = np.indices(shape)
    dists = np.maximum(
        np.abs(rows[None, :, :] - sorted_seeds[:, 0, None, None]),
        np.abs(cols[None, :, :] - sorted_seeds[:, 1, None, None]),
    )
    owner = np.argmin(dists, axis=0)
    min_dist = dists.min(axis=0)
    return sorted_seeds, owner, min_dist


def _render_canvas(
    shape: tuple[int, int], palette: list[int], seeds: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_seeds, owner, min_dist = _chebyshev_partition(shape, seeds)
    out = np.zeros(shape, dtype=int)
    palette_arr = np.asarray(palette, dtype=int)
    active = min_dist < len(palette)
    out[active] = palette_arr[min_dist[active]]
    assert np.all(out[sorted_seeds[:, 0], sorted_seeds[:, 1]] == int(palette[0]))
    return out, owner, min_dist


def _render_manhattan(
    shape: tuple[int, int], palette: list[int], seeds: list[tuple[int, int]]
) -> np.ndarray:
    sorted_seeds = np.asarray(sorted(seeds), dtype=int)
    rows, cols = np.indices(shape)
    dists = (
        np.abs(rows[None, :, :] - sorted_seeds[:, 0, None, None])
        + np.abs(cols[None, :, :] - sorted_seeds[:, 1, None, None])
    )
    min_dist = dists.min(axis=0)
    out = np.zeros(shape, dtype=int)
    palette_arr = np.asarray(palette, dtype=int)
    active = min_dist < len(palette)
    out[active] = palette_arr[min_dist[active]]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the canvas by expanding the legend-panel palette around multiple seeds with Chebyshev-distance rings.

    Input structure:
    - The leftmost 5 columns are a teal legend panel with teal borders and teal separator rows.
    - Each legend slot contains one small 4-connected non-teal shape; only the shape's color and top-to-bottom order matter.
    - The remaining canvas is black except for 2 or 3 maroon singleton seed pixels.

    Rule:
    - Read the legend colors from top to bottom to obtain an ordered palette P[0], P[1], ..., P[N-1].
    - For every canvas cell, compute its Chebyshev distance to every seed and assign the cell to the seed with minimum distance, breaking ties by smaller seed row then smaller seed column.
    - If the winning distance is d < N, recolor the cell to P[d]; otherwise leave it black. Seed cells therefore become P[0].
    - Keep the legend panel unchanged.

    Key insight:
    - The legend is not a shape dictionary here: its vertical ordering is the palette index, and the canvas is a clipped multi-seed Chebyshev distance map.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] > LEGEND_W

    palette = _read_palette(g)
    seeds = _read_seeds(g)
    canvas_out, _owner, _min_dist = _render_canvas(g[:, LEGEND_W:].shape, palette, seeds)

    out = g.copy()
    out[:, LEGEND_W:] = canvas_out
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _random_connected_shape(height: int, width: int, size: int) -> list[tuple[int, int]]:
    for _ in range(80):
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        while len(cells) < size:
            frontier = []
            for r, c in sorted(cells):
                for nr, nc in neighbors4(r, c):
                    if in_bounds((height, width), nr, nc) and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == size:
            return sorted(cells)
    raise RuntimeError("failed to sample a connected legend shape")


def _build_panel(height: int, palette: list[int], slot_heights: list[int]) -> np.ndarray:
    panel = np.full((height, LEGEND_W), PANEL_COLOR, dtype=int)
    interior_rows = height - 2
    min_used = sum(slot_heights) + (len(slot_heights) - 1)
    spare = interior_rows - min_used
    assert spare >= 0
    extras = _random_composition(spare, len(slot_heights) + 1)

    row = 1 + extras[0]
    for idx, (color, slot_h) in enumerate(zip(palette, slot_heights)):
        panel[row : row + slot_h, 1:4] = BG
        shape_size = random.randint(2, 4)
        for rr, cc in _random_connected_shape(slot_h, 3, shape_size):
            panel[row + rr, 1 + cc] = int(color)
        row += slot_h
        if idx + 1 < len(slot_heights):
            row += 1 + extras[idx + 1]
    return panel


def _pairwise_chebyshev_ok(
    seeds: list[tuple[int, int]], min_gap: int
) -> bool:
    for i, (r0, c0) in enumerate(seeds):
        for r1, c1 in seeds[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) < min_gap:
                return False
    return True


def _place_seeds(
    height: int, width: int, radius: int, seed_count: int
) -> list[tuple[int, int]] | None:
    min_gap = max(6, 2 * radius)
    mode_gap_pairs = []
    for gap in range(min_gap, min_gap + 3):
        if width >= gap + 3:
            mode_gap_pairs.append(("row", gap))
        if height >= gap + 3:
            mode_gap_pairs.append(("col", gap))
        if height >= gap + 3 and width >= gap + 3:
            mode_gap_pairs.append(("diag", gap))
    random.shuffle(mode_gap_pairs)

    for mode, gap in mode_gap_pairs:
        for _ in range(80):
            if mode == "row":
                r = random.randint(1, height - 2)
                c0 = random.randint(1, width - gap - 2)
                seeds = [(r, c0), (r, c0 + gap)]
            elif mode == "col":
                r0 = random.randint(1, height - gap - 2)
                c = random.randint(1, width - 2)
                seeds = [(r0, c), (r0 + gap, c)]
            else:
                r0 = random.randint(1, height - gap - 2)
                c0 = random.randint(1, width - gap - 2)
                if random.random() < 0.5:
                    seeds = [(r0, c0), (r0 + gap, c0 + gap)]
                else:
                    seeds = [(r0 + gap, c0), (r0, c0 + gap)]

            if seed_count == 2:
                return seeds

            candidates = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if (r, c) not in seeds
            ]
            random.shuffle(candidates)
            for cand in candidates:
                seed_set = seeds + [cand]
                if _pairwise_chebyshev_ok(seed_set, min_gap):
                    return seed_set
    return None


def _has_tie(shape: tuple[int, int], seeds: list[tuple[int, int]], palette_len: int) -> bool:
    sorted_seeds, _owner, min_dist = _chebyshev_partition(shape, seeds)
    rows, cols = np.indices(shape)
    dists = np.maximum(
        np.abs(rows[None, :, :] - sorted_seeds[:, 0, None, None]),
        np.abs(cols[None, :, :] - sorted_seeds[:, 1, None, None]),
    )
    ties = (dists == min_dist[None, :, :]).sum(axis=0) >= 2
    return bool(np.any(ties & (min_dist < palette_len)))


def _validate_instance(grid: np.ndarray, palette: list[int], seeds: list[tuple[int, int]]) -> bool:
    parsed_palette = _read_palette(grid)
    parsed_seeds = sorted(_read_seeds(grid))
    if parsed_palette != palette or parsed_seeds != sorted(seeds):
        return False

    out = main(grid)
    expected_canvas, owner, min_dist = _render_canvas(grid[:, LEGEND_W:].shape, palette, seeds)
    if not np.array_equal(out[:, :LEGEND_W], grid[:, :LEGEND_W]):
        return False
    if not np.array_equal(out[:, LEGEND_W:], expected_canvas):
        return False

    if not all(np.any(expected_canvas == int(color)) for color in palette):
        return False
    if not np.any(expected_canvas == BG):
        return False
    if not _has_tie(expected_canvas.shape, seeds, len(palette)):
        return False
    if not _pairwise_chebyshev_ok(seeds, max(6, 2 * (len(palette) - 1))):
        return False

    seed_order = sorted(seeds)
    for idx in range(len(seed_order)):
        if not np.any((owner == idx) & (min_dist < len(palette))):
            return False

    single_seed_canvas, _owner1, _dist1 = _render_canvas(
        expected_canvas.shape, palette, [seed_order[0]]
    )
    if np.array_equal(single_seed_canvas, expected_canvas):
        return False

    swapped_palette = palette.copy()
    swapped_palette[0], swapped_palette[1] = swapped_palette[1], swapped_palette[0]
    swapped_canvas, _owner2, _dist2 = _render_canvas(
        expected_canvas.shape, swapped_palette, seeds
    )
    if np.array_equal(swapped_canvas, expected_canvas):
        return False

    sorted_palette = sorted(palette)
    if sorted_palette == palette:
        return False
    sorted_canvas, _owner3, _dist3 = _render_canvas(
        expected_canvas.shape, sorted_palette, seeds
    )
    if np.array_equal(sorted_canvas, expected_canvas):
        return False

    if np.array_equal(_render_manhattan(expected_canvas.shape, palette, seeds), expected_canvas):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(3, 5)
    palette = random.sample(PALETTE_POOL, palette_len)
    radius = palette_len - 1

    min_height = 2 + palette_len * 2 + (palette_len - 1)
    height = random.randint(max(18, min_height), 27)
    canvas_w = random.randint(max(12, 2 * radius + 5), 24)
    if LEGEND_W + canvas_w > 30:
        return None

    slot_heights = [random.randint(2, 4) for _ in range(palette_len)]
    if 2 + sum(slot_heights) + (palette_len - 1) > height:
        return None

    seed_count = random.choice((2, 2, 3))
    seeds = _place_seeds(height, canvas_w, radius, seed_count)
    if seeds is None:
        return None

    grid = np.zeros((height, LEGEND_W + canvas_w), dtype=int)
    grid[:, :LEGEND_W] = _build_panel(height, palette, slot_heights)
    for r, c in seeds:
        grid[r, LEGEND_W + c] = SEED_COLOR

    if _validate_instance(grid, palette, seeds):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-ring Voronoi input")
