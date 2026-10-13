from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

PATTERN_SHAPES = (
    ((0, 1), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 1), (0, 1)),
    ((0, 1), (1, 1)),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
)

GENERATE_RETRIES = 400


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
                }
            )
    return comps


def _find_legend_top(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0] - 1) if np.all(g[r] == GRAY)]
    if rows:
        return max(rows)

    grey_counts = np.sum(g == GRAY, axis=1)
    candidates = [r for r in range(g.shape[0] - 1) if grey_counts[r] == int(grey_counts.max()) and grey_counts[r] > 0]
    assert candidates, "expected a legend separator row"
    return max(candidates)


def _parse_legend(g: np.ndarray, legend_top: int) -> dict[int, np.ndarray]:
    band = g[legend_top + 1 :, :]
    assert band.shape[0] >= 2, "legend band must be at least 2 cells high"
    mask = (band != BG) & (band != GRAY)

    textures: dict[int, np.ndarray] = {}
    for comp in _components(mask):
        r0, c0, r1, c1 = comp["bbox"]
        if r0 != 0 or r1 != band.shape[0] - 1:
            continue
        if c0 == 0 or c1 - c0 < 3:
            continue

        marker_count = int(np.sum(band[:, c0 - 1] == GRAY))
        if marker_count not in (1, 2, 3, 4):
            continue

        interior = band[:, c0 + 1 : c1]
        if interior.shape[0] < 2 or interior.shape[1] < 2:
            continue

        pattern = interior[:2, :2].copy()
        tiled = np.empty_like(interior)
        for rr in range(interior.shape[0]):
            for cc in range(interior.shape[1]):
                tiled[rr, cc] = pattern[rr % 2, cc % 2]
        assert np.array_equal(interior, tiled), "legend cell interior must be a tiled 2x2 pattern"
        textures[marker_count] = pattern

    assert set(textures) == {1, 2, 3, 4}, "legend must encode counts 1..4 exactly once"
    return textures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed rectangle with the legend texture matching its grey-marker count.

    The upper scene is partitioned by one-cell-wide colored walls. Grey marker counts inside
    each enclosed region map to a 2x2 texture shown in the bottom legend, where the number of
    grey pixels to the left of a legend cell gives the key. Regions with zero markers stay black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend_top = _find_legend_top(g)
    textures = _parse_legend(g, legend_top)

    scene = g[:legend_top, :]
    mask = (scene == BG) | (scene == GRAY)
    out = g.copy()

    for comp in _components(mask):
        r0, c0, r1, c1 = comp["bbox"]
        if r0 == 0 or c0 == 0 or r1 == scene.shape[0] - 1 or c1 == scene.shape[1] - 1:
            continue

        coords = comp["coords"]
        count = int(np.sum(scene[coords[:, 0], coords[:, 1]] == GRAY))
        if count == 0:
            continue
        assert count in textures, "region marker count must be in legend"
        pattern = textures[count]
        for r, c in coords:
            out[r, c] = int(pattern[(r - r0) % 2, (c - c0) % 2])

    return out


def _make_pattern(shape: tuple[tuple[int, int], tuple[int, int]], colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    arr = np.array(shape, dtype=int)
    out = np.full((2, 2), a, dtype=int)
    out[arr == 1] = b
    return out


def _pick_textures() -> dict[int, np.ndarray]:
    shapes = random.sample(PATTERN_SHAPES, 4)
    textures: dict[int, np.ndarray] = {}
    for count, shape in zip(random.sample((1, 2, 3, 4), 4), shapes):
        colors = tuple(random.sample(COLOR_POOL, 2))
        if random.random() < 0.5:
            colors = (colors[1], colors[0])
        textures[count] = _make_pattern(shape, colors)
    return textures


def _pick_marker_cells(height: int, width: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return random.sample(cells, count)


def _region_boxes(h_lines: list[int], v_lines: list[int]) -> list[tuple[int, int, int, int]]:
    boxes = []
    for r0, r1 in zip(h_lines, h_lines[1:]):
        for c0, c1 in zip(v_lines, v_lines[1:]):
            boxes.append((r0 + 1, c0 + 1, r1 - 1, c1 - 1))
    return boxes


def _fill_region(out: np.ndarray, box: tuple[int, int, int, int], pattern: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = int(pattern[(r - r0) % 2, (c - c0) % 2])


def _build_expected(grid: np.ndarray, assignments: list[tuple[tuple[int, int, int, int], int]], textures: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for box, count in assignments:
        if count == 0:
            continue
        _fill_region(out, box, textures[count])
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_row_cells = random.randint(2, 3)
    n_col_cells = random.randint(4, 5)
    cell_heights = [random.randint(3, 6) for _ in range(n_row_cells)]
    cell_widths = [random.randint(3, 7) for _ in range(n_col_cells)]
    legend_h = random.choice((4, 6))

    scene_h = 2 + sum(cell_heights) + (n_row_cells - 1)
    width = 2 + sum(cell_widths) + (n_col_cells - 1)
    height = scene_h + 1 + legend_h

    if not (14 <= height <= 28 and 23 <= width <= 36):
        return None

    textures = _pick_textures()
    entry_order = random.sample((1, 2, 3, 4), 4)
    entry_inner_widths = [random.randint(2, 4) for _ in range(4)]
    entry_gaps = [random.randint(1, 2) for _ in range(3)]
    entry_span = sum(w + 3 for w in entry_inner_widths) + sum(entry_gaps)
    if entry_span > width:
        return None

    grid = np.full((height, width), BG, dtype=int)
    legend_top = scene_h
    grid[legend_top, :] = GRAY

    h_lines = [0]
    cur = 0
    for size in cell_heights[:-1]:
        cur += size + 1
        h_lines.append(cur)
    h_lines.append(scene_h - 1)

    v_lines = [0]
    cur = 0
    for size in cell_widths[:-1]:
        cur += size + 1
        v_lines.append(cur)
    v_lines.append(width - 1)

    for r in h_lines:
        grid[r, :] = random.choice(COLOR_POOL)
    for c in v_lines:
        grid[:scene_h, c] = random.choice(COLOR_POOL)

    boxes = _region_boxes(h_lines, v_lines)
    counts = [0, 1, 2, 3, 4]
    while len(counts) < len(boxes):
        counts.append(random.randint(0, 4))
    random.shuffle(counts)

    assignments = list(zip(boxes, counts, strict=True))
    for box, count in assignments:
        if count == 0:
            continue
        r0, c0, r1, c1 = box
        cells = _pick_marker_cells(r1 - r0 + 1, c1 - c0 + 1, count)
        for rr, cc in cells:
            grid[r0 + rr, c0 + cc] = GRAY

    start_col = random.randint(0, width - entry_span)
    band = grid[legend_top + 1 :, :]
    col = start_col
    for idx, count in enumerate(entry_order):
        marker_col = col
        left_wall = col + 1
        inner_w = entry_inner_widths[idx]
        right_wall = left_wall + inner_w + 1

        band[:count, marker_col] = GRAY
        band[:, left_wall] = random.choice(COLOR_POOL)
        band[:, right_wall] = random.choice(COLOR_POOL)

        pattern = textures[count]
        for rr in range(legend_h):
            for cc in range(inner_w):
                band[rr, left_wall + 1 + cc] = int(pattern[rr % 2, cc % 2])

        col = right_wall + 1
        if idx < len(entry_gaps):
            col += entry_gaps[idx]

    expected = _build_expected(grid, [(box, count) for box, count in assignments], textures)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        built = _gen_attempt()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
