from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
STRIPE_COLORS = (1, 2, 3, 4, 6, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# The 3-pixel L arrow is rendered as a filled 2x2 block with one missing corner.
# Rotating the missing corner clockwise yields up, right, down, left.
DIR_FROM_MISSING = {
    (0, 1): (-1, 0),
    (1, 1): (0, 1),
    (1, 0): (1, 0),
    (0, 0): (0, -1),
}
MISSING_FROM_DIR = {v: k for k, v in DIR_FROM_MISSING.items()}

GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.size > 0, "legend panel border must exist"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert c1 == grid.shape[1] - 1, "legend panel must touch the right edge"
    return r0, c0, r1, c1


def _decode_arrow(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "arrow must occupy a 2x2 box"
    norm = {(int(r - r0), int(c - c0)) for r, c in cells}
    assert len(norm) == 3, "arrow must contain exactly three maroon pixels"
    missing = ({(0, 0), (0, 1), (1, 0), (1, 1)} - norm).pop()
    return DIR_FROM_MISSING[missing]


def _legend_mapping(grid: np.ndarray, panel: tuple[int, int, int, int]) -> dict[int, tuple[int, int]]:
    pr0, pc0, pr1, pc1 = panel
    inner = grid[pr0 + 1 : pr1, pc0 + 1 : pc1]

    samples: list[tuple[float, int]] = []
    for color in sorted(set(map(int, np.unique(inner))) - {BG, GREY, MAROON}):
        pts = np.argwhere(inner == int(color))
        assert len(pts) == 1, "each legend color must appear exactly once as a sample pixel"
        samples.append((float(pts[0, 0]), int(color)))

    arrows: list[tuple[float, tuple[int, int]]] = []
    for comp in _components(inner == MAROON):
        arrows.append((float(np.mean([r for r, _c in comp])), _decode_arrow(comp)))

    samples.sort(key=lambda x: x[0])
    arrows.sort(key=lambda x: x[0])
    assert len(samples) == len(arrows) and samples, "legend must pair every sample color with one arrow"

    return {color: direction for (_row, color), (_arow, direction) in zip(samples, arrows)}


def _stripe_rows(grid: np.ndarray, garden_w: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r, :garden_w]
        color = int(row[0])
        if color != BG and bool(np.all(row == color)):
            rows.append((int(r), int(color)))
    assert rows, "garden must contain at least one stripe row"
    return rows


def _grown_cells(
    seed_cells: list[tuple[int, int]],
    steps: int,
    direction: tuple[int, int],
    top: int,
    bottom: int,
    garden_w: int,
) -> set[tuple[int, int]]:
    dr, dc = direction
    grown = set((int(r), int(c)) for r, c in seed_cells)
    for r, c in seed_cells:
        for step in range(1, int(steps) + 1):
            nr = int(r + dr * step)
            nc = int(c + dc * step)
            if nr < int(top) or nr > int(bottom) or nc < 0 or nc >= int(garden_w):
                break
            grown.add((nr, nc))
    return grown


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each orange seed along the direction encoded by its region stripe color.

    The legend panel on the right maps stripe colors to directions via 3-pixel maroon L arrows.
    A seed grows for as many generations as its area, copying the whole current pattern one step
    each generation. Because growth is one-directional, this is equivalent to extending every seed
    pixel into a ray of length `area`, clipped by region boundaries and grid edges.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    panel = _panel_bbox(g)
    pr0, pc0, pr1, pc1 = panel
    garden_w = pc0

    legend = _legend_mapping(g, panel)
    stripes = _stripe_rows(g, garden_w)

    out = g.copy()
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = BG

    prev_stripe = -1
    for stripe_r, stripe_color in stripes:
        region_top = int(prev_stripe + 1)
        region_bottom = int(stripe_r - 1)
        prev_stripe = int(stripe_r)
        assert region_top <= region_bottom, "every stripe must cap a non-empty region"
        assert stripe_color in legend, "every stripe color must be listed in the legend"

        region = g[region_top : stripe_r, :garden_w]
        seed_rel = np.argwhere(region == ORANGE)
        assert seed_rel.size > 0, "every region must contain an orange seed"
        seed_cells = [(int(region_top + r), int(c)) for r, c in seed_rel]
        steps = len(seed_cells)

        for r, c in _grown_cells(seed_cells, steps, legend[stripe_color], region_top, region_bottom, garden_w):
            out[r, c] = ORANGE

    assert prev_stripe == h - 1, "the final region must be closed by a bottom stripe"
    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return sorted((int(r - r0), int(c - c0)) for r, c in cells)


def _rotate(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _normalize({(int(c), int(-r)) for r, c in cells})


def _flip_h(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    max_c = max(c for _r, c in cells)
    return _normalize({(int(r), int(max_c - c)) for r, c in cells})


def _flip_v(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    max_r = max(r for r, _c in cells)
    return _normalize({(int(max_r - r), int(c)) for r, c in cells})


def _random_seed_shape(area: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                frontier.add((int(r + dr), int(c + dc)))
        frontier -= cells
        cells.add(random.choice(sorted(frontier)))

    shape = _normalize(cells)
    for _ in range(random.randrange(4)):
        shape = _rotate(shape)
    if random.random() < 0.5:
        shape = _flip_h(shape)
    if random.random() < 0.5:
        shape = _flip_v(shape)
    return shape


def _ray_limit(
    cell: tuple[int, int],
    steps: int,
    direction: tuple[int, int],
    top: int,
    bottom: int,
    garden_w: int,
) -> int:
    r, c = cell
    dr, dc = direction
    limit = 0
    for step in range(1, int(steps) + 1):
        nr = int(r + dr * step)
        nc = int(c + dc * step)
        if nr < int(top) or nr > int(bottom) or nc < 0 or nc >= int(garden_w):
            break
        limit += 1
    return limit


def _choose_seed_cells(
    shape: list[tuple[int, int]],
    steps: int,
    direction: tuple[int, int],
    top: int,
    bottom: int,
    garden_w: int,
) -> list[tuple[int, int]] | None:
    shape_h = max(r for r, _c in shape) + 1
    shape_w = max(c for _r, c in shape) + 1
    if shape_h > bottom - top + 1 or shape_w > garden_w:
        return None

    candidates: list[tuple[bool, list[tuple[int, int]]]] = []
    for r0 in range(int(top), int(bottom - shape_h + 2)):
        for c0 in range(0, int(garden_w - shape_w + 1)):
            cells = [(int(r0 + r), int(c0 + c)) for r, c in shape]
            limits = [_ray_limit(cell, steps, direction, top, bottom, garden_w) for cell in cells]
            if max(limits) == 0:
                continue
            clipped = any(limit < int(steps) for limit in limits)
            candidates.append((clipped, cells))

    if not candidates:
        return None

    prefer_clipped = random.random() < 0.5
    pool = [cells for clipped, cells in candidates if clipped == prefer_clipped]
    if not pool:
        pool = [cells for _clipped, cells in candidates]
    return random.choice(pool)


def _paint_arrow(grid: np.ndarray, r0: int, c0: int, direction: tuple[int, int]) -> None:
    missing = MISSING_FROM_DIR[direction]
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                grid[int(r0 + dr), int(c0 + dc)] = MAROON


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.randint(2, 4)
    content_heights = [random.randint(3, 5) for _ in range(n_regions)]
    garden_h = sum(h + 1 for h in content_heights)
    garden_w = random.randint(10, 16)
    panel_w = random.randint(7, 9)
    h = garden_h
    w = garden_w + panel_w
    panel_c0 = garden_w
    panel_c1 = w - 1

    stripe_colors = random.sample(STRIPE_COLORS, n_regions)
    region_directions = random.sample(list(DIRECTIONS), n_regions)
    dir_by_color = {color: direction for color, direction in zip(stripe_colors, region_directions)}

    grid = np.full((h, w), BG, dtype=int)

    stripe_rows: list[tuple[int, int, int, int]] = []
    top = 0
    for idx, content_h in enumerate(content_heights):
        bottom = top + content_h - 1
        stripe_r = bottom + 1
        stripe_color = stripe_colors[idx]
        grid[stripe_r, :garden_w] = int(stripe_color)
        stripe_rows.append((top, bottom, stripe_r, stripe_color))
        top = stripe_r + 1

    assert stripe_rows[-1][2] == h - 1

    grid[:, panel_c0:panel_c1 + 1] = BG
    grid[:, panel_c0] = GREY
    grid[:, panel_c1] = GREY
    grid[0, panel_c0:panel_c1 + 1] = GREY
    grid[h - 1, panel_c0:panel_c1 + 1] = GREY

    sample_rel_options = [x for x in (1, 2) if x < panel_w - 1]
    arrow_rel_options = sorted({panel_w - 4, panel_w - 3})
    arrow_rel_options = [x for x in arrow_rel_options if 1 <= x <= panel_w - 3]
    if not sample_rel_options or not arrow_rel_options:
        return None

    for top, bottom, _stripe_r, stripe_color in stripe_rows:
        entry_r0 = random.randint(int(top), int(bottom - 1))
        sample_r = entry_r0 + random.randint(0, 1)
        sample_c = panel_c0 + random.choice(sample_rel_options)
        arrow_c = panel_c0 + random.choice(arrow_rel_options)
        grid[sample_r, sample_c] = int(stripe_color)
        _paint_arrow(grid, entry_r0, arrow_c, dir_by_color[stripe_color])

    for top, bottom, _stripe_r, stripe_color in stripe_rows:
        area = random.randint(2, 5)
        placed = None
        for _ in range(40):
            shape = _random_seed_shape(area)
            placed = _choose_seed_cells(shape, area, dir_by_color[stripe_color], top, bottom, garden_w)
            if placed is not None:
                break
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = ORANGE

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out == ORANGE) <= np.count_nonzero(grid == ORANGE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
