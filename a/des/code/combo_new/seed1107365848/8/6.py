import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(10))

MIN_GRID = 12
MAX_GRID = 28
MIN_REGION_SPAN = 3
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_palette(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[str, list[int]]] = []

    edge_lines = [
        ("top", list(map(int, grid[0, :]))),
        ("bottom", list(map(int, grid[h - 1, :]))),
        ("left", list(map(int, grid[:, 0]))),
        ("right", list(map(int, grid[:, w - 1]))),
    ]

    for side, line in edge_lines:
        positions = [i for i, value in enumerate(line) if value != bg]
        if len(positions) != 5:
            continue
        if any(positions[i + 1] - positions[i] < 2 for i in range(4)):
            continue
        colors = [line[i] for i in positions]
        if len(set(colors)) != 5:
            continue
        candidates.append((side, colors))

    assert len(candidates) == 1, "expected exactly one edge palette"
    side, colors = candidates[0]

    immutable = np.zeros_like(grid, dtype=bool)
    if side == "top":
        immutable[0, :] = True
    elif side == "bottom":
        immutable[h - 1, :] = True
    elif side == "left":
        immutable[:, 0] = True
    else:
        immutable[:, w - 1] = True
    return colors, immutable


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the palette color indexed by its seed count.

    Input structure:
    - The background is the dominant color.
    - One connected single-color wall network partitions the grid into closed
      regions, and a full edge row or column carries a palette of five single
      pixels separated by background cells.
    - Seed pixels inside regions all share one marker color; a region may contain
      zero or more seeds.

    Rule:
    - Read the five palette colors in edge order; they represent counts 1, 2, 3,
      4, and 5.
    - For every non-wall region away from the palette edge strip, count how many
      seed pixels it contains.
    - If the count is N>0, fill the whole region with the Nth palette color,
      overwriting background and seeds but leaving walls and the palette edge
      unchanged.
    - Regions with no seeds remain unchanged.

    Key insight:
    - The seed color does not encode the fill color directly; only the number of
      seeds matters, and that count must be looked up through the ordered palette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    palette, immutable = _find_palette(g, bg)

    visible = ~immutable
    color_counts = Counter(map(int, g[visible].flatten()))
    color_counts.pop(bg, None)
    for color in palette:
        color_counts.pop(int(color), None)

    assert len(color_counts) == 2, "expected one wall color and one seed color"
    colors = list(color_counts)
    wall = max(colors, key=lambda color: int(color_counts[color]))
    seed = min(colors, key=lambda color: int(color_counts[color]))

    regions = _components_mask(visible & (g != wall))
    out = g.copy()
    for comp in regions:
        seed_count = sum(int(g[r, c] == seed) for r, c in comp)
        if seed_count == 0:
            continue
        assert 1 <= seed_count <= len(palette), "seed count must fit the palette"
        fill = int(palette[seed_count - 1])
        for r, c in comp:
            out[r, c] = fill
    return out


def _layout_bounds(h: int, w: int, side: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if side == "bottom":
        frame = (0, h - 2, 0, w - 1)
    elif side == "top":
        frame = (1, h - 1, 0, w - 1)
    elif side == "left":
        frame = (0, h - 1, 1, w - 1)
    else:
        frame = (0, h - 1, 0, w - 2)
    r0, r1, c0, c1 = frame
    inner = (r0 + 1, r1 - 1, c0 + 1, c1 - 1)
    return frame, inner


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _rand_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _palette_positions(length: int, rng: random.Random) -> list[int]:
    extra = length - 9
    gaps = _rand_composition(extra, 6, rng)
    pos = gaps[0]
    positions = [pos]
    for i in range(1, 5):
        pos += 2 + gaps[i]
        positions.append(pos)
    return positions


def _split_region(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    horizontal: bool,
    line_color: int,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    ra, rb, ca, cb = region
    if horizontal:
        if rb - ra + 1 < 2 * MIN_REGION_SPAN + 1:
            return None
        split_r = rng.randint(ra + MIN_REGION_SPAN, rb - MIN_REGION_SPAN)
        grid[split_r, ca - 1 : cb + 2] = line_color
        return (ra, split_r - 1, ca, cb), (split_r + 1, rb, ca, cb)
    if cb - ca + 1 < 2 * MIN_REGION_SPAN + 1:
        return None
    split_c = rng.randint(ca + MIN_REGION_SPAN, cb - MIN_REGION_SPAN)
    grid[ra - 1 : rb + 2, split_c] = line_color
    return (ra, rb, ca, split_c - 1), (ra, rb, split_c + 1, cb)


def _seed_cells(region: tuple[int, int, int, int], count: int, rng: random.Random) -> list[tuple[int, int]]:
    ra, rb, ca, cb = region
    parity = rng.randint(0, 1)
    cells = [(r, c) for r in range(ra, rb + 1) for c in range(ca, cb + 1) if (r + c) % 2 == parity]
    if len(cells) < count:
        cells = [(r, c) for r in range(ra, rb + 1) for c in range(ca, cb + 1)]
    return rng.sample(cells, count)


def _region_seed_counts(n_regions: int, rng: random.Random) -> list[int]:
    seeded = rng.randint(2, min(5, n_regions - 1))
    if rng.random() < 0.55:
        distinct = rng.randint(2, min(5, seeded))
        base = rng.sample([1, 2, 3, 4, 5], distinct)
        counts = list(base)
        while len(counts) < seeded:
            counts.append(rng.choice(base if rng.random() < 0.7 else [1, 2, 3, 4, 5]))
        rng.shuffle(counts)
    else:
        counts = [rng.randint(1, 5) for _ in range(seeded)]
        if len(set(counts)) == 1:
            counts[0] = 1 + counts[0] % 5
    return counts


def _paint_palette(grid: np.ndarray, side: str, colors: list[int], rng: random.Random, bg: int) -> None:
    if side in {"top", "bottom"}:
        row = 0 if side == "top" else grid.shape[0] - 1
        positions = _palette_positions(grid.shape[1], rng)
        grid[row, :] = bg
        for pos, color in zip(positions, colors):
            grid[row, pos] = color
        return
    col = 0 if side == "left" else grid.shape[1] - 1
    positions = _palette_positions(grid.shape[0], rng)
    grid[:, col] = bg
    for pos, color in zip(positions, colors):
        grid[pos, col] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(["top", "bottom", "left", "right"])
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)
    grid = np.full((h, w), rng.choice(ALL_COLORS), dtype=int)

    colors = rng.sample(ALL_COLORS, 8)
    bg, line_color, seed_color = colors[:3]
    palette = colors[3:]
    grid[:, :] = bg

    frame, inner = _layout_bounds(h, w, side)
    ra, rb, ca, cb = inner
    if rb - ra + 1 < 2 * MIN_REGION_SPAN + 1 or cb - ca + 1 < 2 * MIN_REGION_SPAN + 1:
        return None

    _paint_frame(grid, frame, line_color)
    regions = [inner]
    target_regions = rng.randint(4, 8)
    for _ in range(20):
        if len(regions) >= target_regions:
            break
        candidates = []
        for idx, region in enumerate(regions):
            r0, r1, c0, c1 = region
            opts = []
            if r1 - r0 + 1 >= 2 * MIN_REGION_SPAN + 1:
                opts.append(True)
            if c1 - c0 + 1 >= 2 * MIN_REGION_SPAN + 1:
                opts.append(False)
            if opts:
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                candidates.append((idx, region, opts, area))
        if not candidates:
            break
        weights = [area for _, _, _, area in candidates]
        idx, region, opts, _area = rng.choices(candidates, weights=weights, k=1)[0]
        split = _split_region(grid, region, rng.choice(opts), line_color, rng)
        if split is None:
            continue
        regions.pop(idx)
        regions.extend(split)

    if len(regions) < 4:
        return None

    counts = _region_seed_counts(len(regions), rng)
    if len(set(counts)) < 2:
        return None
    seeded_regions = rng.sample(range(len(regions)), len(counts))
    for region_index, seed_count in zip(seeded_regions, counts):
        for r, c in _seed_cells(regions[region_index], seed_count, rng):
            grid[r, c] = seed_color

    _paint_palette(grid, side, palette, rng, bg)

    if Counter(map(int, grid.flatten())).most_common(1)[0][0] != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    wall_pixels = int(np.sum(grid == line_color))
    seed_pixels = int(np.sum(grid == seed_color))
    bg_pixels = int(np.sum(grid == bg))
    if not (bg_pixels > wall_pixels > seed_pixels >= 2):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
