import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
GREY = 5
SEED_COLORS = [1, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = random.Random()

_SHAPES_BY_AREA = {
    1: [[(0, 0)]],
    2: [[(0, 0), (0, 1)]],
    3: [[(0, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (1, 1)]],
    4: [[(0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 1)]],
    5: [[(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]],
    6: [[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]],
}


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in coords)


def _transform_shape(coords: list[tuple[int, int]], variant: int) -> list[tuple[int, int]]:
    out = []
    reflect = variant >= 4
    rot = variant % 4
    for r, c in coords:
        rr, cc = int(r), int(c)
        if reflect:
            cc = -cc
        for _ in range(rot):
            rr, cc = int(cc), int(-rr)
        out.append((rr, cc))
    return _normalize(out)


def _bbox_dims(coords: list[tuple[int, int]]) -> tuple[int, int]:
    height = max(r for r, _ in coords) + 1
    width = max(c for _, c in coords) + 1
    return int(height), int(width)


def _segments_from_lines(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = sorted(set(int(x) for x in lines))
    if 0 not in bounds:
        bounds = [-1] + bounds
    if limit - 1 not in bounds:
        bounds = bounds + [limit]

    segments = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= 1:
            continue
        segments.append((int(a + 1), int(b - 1)))
    return segments


def _detect_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    best = None

    for color in sorted(set(map(int, g.ravel()))):
        row_lines = [r for r in range(h) if bool((g[r, :] == color).all())]
        col_lines = [c for c in range(w) if bool((g[:, c] == color).all())]
        row_segments = _segments_from_lines(row_lines, h)
        col_segments = _segments_from_lines(col_lines, w)
        if len(row_segments) < 2 or len(col_segments) < 2:
            continue
        row_heights = {r1 - r0 + 1 for r0, r1 in row_segments}
        col_widths = {c1 - c0 + 1 for c0, c1 in col_segments}
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue
        score = (len(row_segments) * len(col_segments), len(row_lines) + len(col_lines))
        if best is None or score > best[0]:
            best = (score, int(color), row_segments, col_segments)

    assert best is not None, "expected a regular separator lattice"
    return best[1], best[2], best[3]


def _background_color(grid: np.ndarray, row_segments: list[tuple[int, int]], col_segments: list[tuple[int, int]]) -> int:
    g = np.asarray(grid, dtype=int)
    values = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            values.extend(map(int, g[r0 : r1 + 1, c0 : c1 + 1].ravel()))
    return Counter(values).most_common(1)[0][0]


def _extract_pattern(patch: np.ndarray, bg: int, line_color: int) -> dict | None:
    mask = (patch != int(bg)) & (patch != int(line_color))
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    colors = {int(patch[r, c]) for r, c in pts}
    assert len(colors) == 1, "each seed cell should contain a single-color pattern"
    coords = sorted((int(r), int(c)) for r, c in pts)
    return {"coords": coords, "area": len(coords), "color": int(next(iter(colors)))}


def _paint_pattern(canvas: np.ndarray, bbox: tuple[int, int, int, int], coords: list[tuple[int, int]], color: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for rr, cc in coords:
        patch[int(rr), int(cc)] = int(color)
    canvas[r0 : r1 + 1, c0 : c1 + 1] = patch


def _neighbor_count_map(occupied: np.ndarray) -> dict[tuple[int, int], int]:
    rows, cols = map(int, occupied.shape)
    out: dict[tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            if bool(occupied[r, c]):
                continue
            k = 0
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < rows and 0 <= cc < cols and bool(occupied[rr, cc]):
                    k += 1
            out[(int(r), int(c))] = int(k)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse seed patterns into neighboring cells of a grey-partitioned table in one non-iterative step.

    Input structure:
    - bg = black. Single-pixel grey separator lines divide the grid into a regular row-by-column array of rectangular cells.
    - A few cells contain one pure-color pattern each; the other cells are empty and contain only bg.
    - Seed patterns may have different shapes, offsets, colors, and occupied areas, but every cell uses the same interior size.

    Rule:
    - Detect the separator lattice, then read each cell's interior pattern as relative occupied coordinates plus its color.
    - Only the originally non-empty cells count as active for this single expansion step.
    - For each originally empty cell, count how many of its four neighboring cells are active:
      k = 0 keeps the cell empty; k = 1 copies that sole neighbor's pattern exactly; k >= 2 copies the largest-area neighboring pattern but recolors it red.
    - Pattern copies preserve the source cell's relative pixel positions inside the destination cell.

    Key insight:
    - The grey lattice defines the true objects at the cell level, and the visible local outcome depends on a two-stage rule: neighbor count first, then either faithful copying or largest-shape selection with a forced red recolor.
    """
    g = np.asarray(grid, dtype=int)
    line_color, row_segments, col_segments = _detect_lattice(g)
    bg = _background_color(g, row_segments, col_segments)

    n_rows = len(row_segments)
    n_cols = len(col_segments)
    occupied = np.zeros((n_rows, n_cols), dtype=bool)
    patterns: list[list[dict | None]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    boxes: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(n_cols)] for _ in range(n_rows)]

    for r, (r0, r1) in enumerate(row_segments):
        for c, (c0, c1) in enumerate(col_segments):
            boxes[r][c] = (int(r0), int(c0), int(r1), int(c1))
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            pattern = _extract_pattern(patch, bg=bg, line_color=line_color)
            patterns[r][c] = pattern
            occupied[r, c] = pattern is not None

    out = g.copy()
    for r in range(n_rows):
        for c in range(n_cols):
            if bool(occupied[r, c]):
                continue
            neighbors = []
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < n_rows and 0 <= cc < n_cols and patterns[rr][cc] is not None:
                    neighbors.append(patterns[rr][cc])

            if len(neighbors) == 0:
                continue
            if len(neighbors) == 1:
                chosen = neighbors[0]
                paint_color = int(chosen["color"])
            else:
                chosen = max(neighbors, key=lambda item: (int(item["area"]), len(item["coords"])))
                paint_color = RED

            _paint_pattern(out, boxes[r][c], chosen["coords"], paint_color, bg)

    return out


def _random_pattern(area: int, cell_h: int, cell_w: int, rng: random.Random) -> list[tuple[int, int]]:
    base = rng.choice(_SHAPES_BY_AREA[int(area)])
    variants = []
    seen = set()
    for variant in range(8):
        coords = tuple(_transform_shape(base, variant))
        if coords in seen:
            continue
        seen.add(coords)
        h, w = _bbox_dims(list(coords))
        if h <= cell_h and w <= cell_w:
            variants.append(list(coords))
    assert variants, "pattern must fit inside the chosen cell"
    coords = rng.choice(variants)
    height, width = _bbox_dims(coords)
    off_r = rng.randint(0, cell_h - height)
    off_c = rng.randint(0, cell_w - width)
    return sorted((int(r + off_r), int(c + off_c)) for r, c in coords)


def _build_grid(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    seeds: dict[tuple[int, int], dict[str, int | list[tuple[int, int]]]],
) -> np.ndarray:
    height = n_rows * cell_h + n_rows + 1
    width = n_cols * cell_w + n_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    grid[:: cell_h + 1, :] = GREY
    grid[:, :: cell_w + 1] = GREY

    for (r, c), info in seeds.items():
        r0 = 1 + r * (cell_h + 1)
        c0 = 1 + c * (cell_w + 1)
        for rr, cc in info["coords"]:
            grid[int(r0 + rr), int(c0 + cc)] = int(info["color"])
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(3, 5)
    n_cols = rng.randint(3, 5)

    max_cell_h = min(6, (29 - n_rows) // n_rows)
    max_cell_w = min(6, (29 - n_cols) // n_cols)
    if max_cell_h < 4 or max_cell_w < 4:
        return None
    cell_h = rng.randint(4, max_cell_h)
    cell_w = rng.randint(4, max_cell_w)

    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    max_seeds = min(5, len(all_cells) - 3, len(SEED_COLORS), len(_SHAPES_BY_AREA))
    if max_seeds < 2:
        return None

    for _ in range(150):
        seed_count = rng.randint(2, max_seeds)
        seed_positions = rng.sample(all_cells, seed_count)
        occupied = np.zeros((n_rows, n_cols), dtype=bool)
        for r, c in seed_positions:
            occupied[r, c] = True

        counts = _neighbor_count_map(occupied)
        values = list(counts.values())
        if not values:
            continue
        if not any(v == 0 for v in values):
            continue
        if not any(v == 1 for v in values):
            continue
        if not any(v >= 2 for v in values):
            continue

        areas = rng.sample(sorted(_SHAPES_BY_AREA), seed_count)
        colors = rng.sample(SEED_COLORS, seed_count)
        seeds = {}
        for (r, c), area, color in zip(seed_positions, areas, colors):
            seeds[(int(r), int(c))] = {
                "coords": _random_pattern(int(area), cell_h, cell_w, rng),
                "area": int(area),
                "color": int(color),
            }

        grid = _build_grid(n_rows, n_cols, cell_h, cell_w, seeds)

        try:
            out = main(grid)
        except Exception:
            continue

        if out.shape != grid.shape:
            continue
        if not np.any(out == RED):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt(_RNG)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid diffusion puzzle input")
