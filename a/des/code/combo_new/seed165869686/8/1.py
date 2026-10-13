import random
from collections import Counter

import numpy as np


GREY = 5
LEGEND_SIZE = 4
LEGEND_GAP = 1
GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 30


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _parse_legend_patch(patch: np.ndarray) -> tuple[int, np.ndarray] | None:
    if patch.shape != (LEGEND_SIZE, LEGEND_SIZE):
        return None
    border = np.concatenate([patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]])
    border_color = int(border[0])
    if np.any(border != border_color):
        return None
    inner = patch[1:3, 1:3]
    if int(inner[0, 0]) != int(inner[1, 1]):
        return None
    if int(inner[0, 1]) != int(inner[1, 0]):
        return None
    if int(inner[0, 0]) == int(inner[0, 1]):
        return None
    if border_color in set(map(int, inner.flatten())):
        return None
    return border_color, inner.copy()


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for pts in _components(grid != bg):
        if not any(r == 0 or c == 0 for r, c in pts):
            continue
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if (r1 - r0 + 1, c1 - c0 + 1) != (LEGEND_SIZE, LEGEND_SIZE):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        parsed = _parse_legend_patch(patch)
        if parsed is None:
            continue
        border_color, inner = parsed
        mapping[border_color] = inner
    assert mapping, "expected at least one legend block on the top or left edge"
    return mapping


def _grey_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == GREY):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(pts) == _perimeter_cells(r0, c0, r1, c1):
            frames.append((r0, c0, r1, c1))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-framed region by tiling the checkerboard pattern selected by its seed color.

    Input structure:
    - The background color is the most common color.
    - Several 1-pixel-thick grey rectangular frames enclose background interiors, and each interior contains exactly one non-background seed in its upper-left interior cell.
    - Small legend blocks lie on the top edge or left edge; each legend has a uniform border color and a 2x2 checkerboard interior.

    Rule:
    - Read the legend blocks as a mapping from border color to 2x2 checkerboard tile.
    - For each grey rectangular frame, read its seed color and select the legend with the same border color.
    - Tile that 2x2 checkerboard across the full frame interior from the interior's top-left corner, overwriting the seed, while leaving legends and grey borders unchanged.

    Key insight:
    - The seed color is not the final fill color; it is an index into a legend that encodes both the two fill colors and their spatial arrangement.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    legend_map = _legend_mapping(g, bg)
    out = g.copy()

    for r0, c0, r1, c1 in _grey_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((interior != bg) & (interior != GREY))
        assert len(seeds) == 1, "each framed region must contain exactly one seed"
        sr, sc = map(int, seeds[0])
        assert (sr, sc) == (0, 0), "seed must sit in the upper-left interior cell"
        seed_color = int(interior[sr, sc])
        assert seed_color in legend_map, "seed color must match a legend border"
        tile = legend_map[seed_color]
        ih, iw = interior.shape
        for dr in range(ih):
            for dc in range(iw):
                out[r0 + 1 + dr, c0 + 1 + dc] = int(tile[dr % 2, dc % 2])

    return out


def _checkerboard(color_a: int, color_b: int, phase: int) -> np.ndarray:
    if phase == 0:
        return np.array([[color_a, color_b], [color_b, color_a]], dtype=int)
    return np.array([[color_b, color_a], [color_a, color_b]], dtype=int)


def _legend_patch(border_color: int, tile: np.ndarray) -> np.ndarray:
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), border_color, dtype=int)
    patch[1:3, 1:3] = tile
    return patch


def _frame_patch(inner_h: int, inner_w: int, bg: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    return patch


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    h0, w0 = occupied.shape
    occupied[max(0, r0 - 1) : min(h0, r0 + h + 1), max(0, c0 - 1) : min(w0, c0 + w + 1)] = True


def _bg_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    return counts[bg] == max(counts.values())


def _place_regions(
    rng: random.Random,
    occupied: np.ndarray,
    h: int,
    w: int,
    min_r0: int,
    min_c0: int,
    specs: list[dict[str, int]],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["inner_h"] * specs[idx]["inner_w"],
        reverse=True,
    )
    for idx in order:
        total_h = specs[idx]["inner_h"] + 2
        total_w = specs[idx]["inner_w"] + 2
        positions = []
        for r0 in range(min_r0, h - total_h):
            for c0 in range(min_c0, w - total_w):
                if occupied[r0 - 1 : r0 + total_h + 1, c0 - 1 : c0 + total_w + 1].any():
                    continue
                positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        placements[idx] = (r0, c0)
        _reserve(occupied, r0, c0, total_h, total_w)
    return [p for p in placements if p is not None]


def _choose_region_seeds(
    rng: random.Random, legend_colors: list[int], region_count: int, allow_reuse: bool
) -> list[int] | None:
    if not allow_reuse:
        if region_count > len(legend_colors):
            return None
        return rng.sample(legend_colors, region_count)

    first_two = rng.sample(legend_colors, 2)
    seeds = first_two[:]
    while len(seeds) < region_count:
        seeds.append(rng.choice(legend_colors))
    rng.shuffle(seeds)
    return seeds


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([c for c in range(10) if c != GREY])
    color_pool = [c for c in range(10) if c not in (bg, GREY)]

    legend_count = rng.randint(2, 4)
    legend_colors = rng.sample(color_pool, legend_count)
    fill_pool = [c for c in color_pool if c not in legend_colors]
    if len(fill_pool) < 2:
        return None

    legend_tiles: dict[int, np.ndarray] = {}
    for border_color in legend_colors:
        a, b = rng.sample(fill_pool, 2)
        legend_tiles[border_color] = _checkerboard(a, b, rng.randrange(2))
    if len({tuple(tile.flatten()) for tile in legend_tiles.values()}) < 2:
        return None

    allow_reuse = rng.random() < 0.5
    max_regions = 4 if allow_reuse else legend_count
    region_count = rng.randint(2, max_regions)
    region_seeds = _choose_region_seeds(rng, legend_colors, region_count, allow_reuse)
    if region_seeds is None or len(set(region_seeds)) < 2:
        return None

    region_specs = []
    for seed_color in region_seeds:
        region_specs.append(
            {
                "seed": seed_color,
                "inner_h": rng.randint(2, 6),
                "inner_w": rng.randint(2, 6),
            }
        )

    orientation = rng.choice(["top", "left"])
    legend_span = LEGEND_SIZE * legend_count + LEGEND_GAP * (legend_count - 1)
    min_h = max(GRID_MIN, legend_span if orientation == "left" else GRID_MIN)
    min_w = max(GRID_MIN, legend_span if orientation == "top" else GRID_MIN)
    h = rng.randint(min_h, GRID_MAX)
    w = rng.randint(min_w, GRID_MAX)

    grid = np.full((h, w), bg, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((h, w), dtype=bool)

    if orientation == "top":
        start_c = rng.randint(0, w - legend_span)
        c = start_c
        for border_color in legend_colors:
            patch = _legend_patch(border_color, legend_tiles[border_color])
            grid[0:LEGEND_SIZE, c : c + LEGEND_SIZE] = patch
            expected[0:LEGEND_SIZE, c : c + LEGEND_SIZE] = patch
            _reserve(occupied, 0, c, LEGEND_SIZE, LEGEND_SIZE)
            c += LEGEND_SIZE + LEGEND_GAP
        min_r0, min_c0 = LEGEND_SIZE + 1, 1
    else:
        start_r = rng.randint(0, h - legend_span)
        r = start_r
        for border_color in legend_colors:
            patch = _legend_patch(border_color, legend_tiles[border_color])
            grid[r : r + LEGEND_SIZE, 0:LEGEND_SIZE] = patch
            expected[r : r + LEGEND_SIZE, 0:LEGEND_SIZE] = patch
            _reserve(occupied, r, 0, LEGEND_SIZE, LEGEND_SIZE)
            r += LEGEND_SIZE + LEGEND_GAP
        min_r0, min_c0 = 1, LEGEND_SIZE + 1

    placements = _place_regions(rng, occupied, h, w, min_r0, min_c0, region_specs)
    if placements is None:
        return None

    for spec, (r0, c0) in zip(region_specs, placements):
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        seed_color = spec["seed"]
        patch = _frame_patch(inner_h, inner_w, bg)
        total_h, total_w = patch.shape
        grid[r0 : r0 + total_h, c0 : c0 + total_w] = patch
        expected[r0 : r0 + total_h, c0 : c0 + total_w] = patch
        grid[r0 + 1, c0 + 1] = seed_color

        tile = legend_tiles[seed_color]
        for dr in range(inner_h):
            for dc in range(inner_w):
                expected[r0 + 1 + dr, c0 + 1 + dc] = int(tile[dr % 2, dc % 2])

    if not _bg_dominant(grid, bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
