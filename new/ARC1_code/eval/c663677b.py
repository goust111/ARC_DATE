from __future__ import annotations

import numpy as np

from arc_utils import Color, paste, random_free_location
import random
from functools import lru_cache


def _infer_tile(g: np.ndarray, hole: int) -> np.ndarray:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, int, np.ndarray]] = []

    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            tile = np.full((row_period, col_period), -1, dtype=int)
            conflict = False
            for r in range(height):
                if conflict:
                    break
                for c in range(width):
                    color = int(g[r, c])
                    if color == hole:
                        continue
                    tr, tc = r % row_period, c % col_period
                    known = int(tile[tr, tc])
                    if known == -1:
                        tile[tr, tc] = color
                    elif known != color:
                        conflict = True
                        break

            if not conflict and np.all(tile != -1):
                candidates.append((row_period * col_period, row_period, col_period, tile))

    assert candidates, "expected a fully observed periodic tile"
    min_area = min(area for area, _, _, _ in candidates)
    smallest = [candidate for candidate in candidates if candidate[0] == min_area]
    assert len(smallest) == 1, "expected a unique smallest periodic tile"

    _, row_period, col_period, tile = smallest[0]
    assert row_period < height and col_period < width, "expected the tile to repeat across the grid"
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the wallpaper by replacing black holes with the colors from the inferred repeating tile.

    Input structure
    - The input is a two-dimensional grid containing a repeated rectangular color tile.
    - Black pixels are holes in that wallpaper, not part of the tile.
    - Every tile phase is visible at least once in a non-black cell.

    Rule
    - Find the unique smallest row and column period whose non-black cells agree by repeated phase.
    - Reconstruct the full grid by writing the tile color at each cell's phase.

    Key insight
    - The corrupted cells are exactly the black pixels; their colors are determined by the repeated phase shared with non-black copies elsewhere.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.any(g == Color.BLACK), "expected black holes to fill"

    tile = _infer_tile(g, Color.BLACK)
    assert np.all(tile != Color.BLACK), "black is reserved for holes"

    height, width = map(int, g.shape)
    out = np.empty_like(g)
    for r in range(height):
        for c in range(width):
            out[r, c] = tile[r % tile.shape[0], c % tile.shape[1]]

    assert np.all((g == Color.BLACK) | (g == out)), "non-black input cells must match the inferred tile"
    return out

# --------------- generator (o5x) ---------------

# Canvas and tile geometry.
GEN_GRID_SIZE = 27
GEN_TILE_SIZE = 8
GEN_TILE_REPEATS = 4
GEN_TILE_INTERIOR_SIZE = GEN_TILE_SIZE - 1

# Tile palette.
GEN_HOLE_COLOR = Color.BLACK
GEN_TILE_BASE_COLOR = Color.BLUE
GEN_NON_BLUE_COLORS = tuple(range(2, 10))
GEN_COLOR_COUNT_MIN = 2
GEN_COLOR_COUNT_MAX = 8

# Hole counts and total masked area.
GEN_REGION_COUNT_MIN = 4
GEN_REGION_COUNT_MAX = 5
GEN_TOTAL_BLACK_MIN = 60
GEN_TOTAL_BLACK_MAX = 84

# Hole shape catalog.
GEN_RECT_SIDE_MIN = 2
GEN_RECT_SIDE_MAX = 6
GEN_L_HEIGHT_MAX = 9
GEN_L_WIDTH_MAX = 8
GEN_HOLE_AREA_MIN = 4
GEN_HOLE_AREA_MAX = 45
GEN_MAX_L_PATCHES = 2
GEN_L_COUNT_WEIGHTS = (0.02, 0.68, 0.30)

# Placement spacing and retry policy.
GEN_PLACEMENT_BORDER = 0
GEN_PLACEMENT_PADDING = 1
GEN_PLACEMENT_PADDING_CONNECTIVITY = 8
GEN_ATTEMPT_LIMIT = 300

@lru_cache(maxsize=None)
def _gen_rect_catalog() -> dict[int, tuple[tuple[int, int], ...]]:
    catalog: dict[int, list[tuple[int, int]]] = {}
    dims = range(GEN_RECT_SIDE_MIN, GEN_RECT_SIDE_MAX + 1)
    for h in dims:
        for w in dims:
            catalog.setdefault(h * w, []).append((h, w))
    return {area: tuple(dims) for area, dims in catalog.items()}


@lru_cache(maxsize=None)
def _gen_l_catalog() -> dict[int, tuple[np.ndarray, ...]]:
    catalog: dict[int, list[np.ndarray]] = {}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    dims = range(GEN_RECT_SIDE_MIN, GEN_RECT_SIDE_MAX + 1)
    for h1 in dims:
        for w1 in dims:
            for h2 in dims:
                for w2 in dims:
                    patches: list[np.ndarray] = []
                    if w1 != w2 and h1 + h2 <= GEN_L_HEIGHT_MAX and max(w1, w2) <= GEN_L_WIDTH_MAX:
                        height, width = h1 + h2, max(w1, w2)
                        for c2 in (0, width - w2):
                            patch = np.zeros((height, width), dtype=int)
                            patch[:h1, :w1] = 1
                            patch[h1:, c2 : c2 + w2] = 1
                            patches.append(patch)
                    if h1 != h2 and max(h1, h2) <= GEN_L_HEIGHT_MAX and w1 + w2 <= GEN_L_WIDTH_MAX:
                        height, width = max(h1, h2), w1 + w2
                        for r2 in (0, height - h2):
                            patch = np.zeros((height, width), dtype=int)
                            patch[:h1, :w1] = 1
                            patch[r2 : r2 + h2, w1:] = 1
                            patches.append(patch)

                    for patch in patches:
                        area = int(patch.sum())
                        if not (GEN_HOLE_AREA_MIN <= area <= GEN_HOLE_AREA_MAX and area < patch.size):
                            continue
                        for variant in (
                            patch,
                            np.flipud(patch),
                            np.fliplr(patch),
                            np.flipud(np.fliplr(patch)),
                        ):
                            key = tuple(tuple(map(int, row)) for row in variant)
                            if key in seen:
                                continue
                            seen.add(key)
                            catalog.setdefault(area, []).append(variant.copy())
    return {area: tuple(patches) for area, patches in catalog.items()}


@lru_cache(maxsize=None)
def _gen_rect_area_sums(count: int) -> frozenset[int]:
    if count == 0:
        return frozenset({0})
    areas = tuple(_gen_rect_catalog())
    return frozenset(area + rest for area in areas for rest in _gen_rect_area_sums(count - 1))


@lru_cache(maxsize=None)
def _gen_mixed_area_sums(l_count: int, rect_count: int) -> frozenset[int]:
    if l_count == 0:
        return _gen_rect_area_sums(rect_count)
    l_areas = tuple(_gen_l_catalog())
    return frozenset(
        area + rest
        for area in l_areas
        for rest in _gen_mixed_area_sums(l_count - 1, rect_count)
    )


def _gen_rect_patches(count: int, total: int) -> list[np.ndarray] | None:
    rects = _gen_rect_catalog()
    areas = tuple(rects)
    patches: list[np.ndarray] = []
    remaining = int(total)
    for left in range(int(count), 0, -1):
        feasible = [area for area in areas if remaining - area in _gen_rect_area_sums(left - 1)]
        if not feasible:
            return None
        area = random.choice(feasible)
        h, w = random.choice(rects[area])
        patches.append(np.ones((h, w), dtype=int))
        remaining -= area
    return patches if remaining == 0 else None


def _gen_shuffle_largest_first(patches: list[np.ndarray]) -> list[np.ndarray]:
    random.shuffle(patches)
    return sorted(patches, key=lambda patch: int(patch.sum()), reverse=True)


def _gen_hole_patches(count: int, total: int) -> list[np.ndarray] | None:
    l_shapes = _gen_l_catalog()
    max_l = min(GEN_MAX_L_PATCHES, count - 1)
    l_options = list(range(max_l + 1))
    weights = list(GEN_L_COUNT_WEIGHTS[: len(l_options)])
    preferred_l = random.choices(l_options, weights=weights, k=1)[0]
    l_options.remove(preferred_l)
    random.shuffle(l_options)
    l_options.insert(0, preferred_l)

    for l_count in l_options:
        rect_count = count - l_count
        if total not in _gen_mixed_area_sums(l_count, rect_count):
            continue
        patches: list[np.ndarray] = []
        remaining = int(total)
        for left_l in range(l_count, 0, -1):
            feasible_l_areas = [
                area
                for area in l_shapes
                if remaining - area in _gen_mixed_area_sums(left_l - 1, rect_count)
            ]
            if not feasible_l_areas:
                patches = []
                break
            area = random.choice(feasible_l_areas)
            patches.append(random.choice(l_shapes[area]).copy())
            remaining -= area
        if l_count and not patches:
            continue
        rects = _gen_rect_patches(rect_count, remaining)
        if rects is None:
            continue
        patches.extend(rects)
        return _gen_shuffle_largest_first(patches)

    patches = _gen_rect_patches(count, total)
    if patches is None:
        return None
    return _gen_shuffle_largest_first(patches)


def _gen_tile() -> np.ndarray:
    color_count = random.randint(GEN_COLOR_COUNT_MIN, GEN_COLOR_COUNT_MAX)
    colors = [GEN_TILE_BASE_COLOR] + random.sample(list(GEN_NON_BLUE_COLORS), color_count - 1)
    non_blue = colors[1:]

    tile = np.full((GEN_TILE_SIZE, GEN_TILE_SIZE), GEN_TILE_BASE_COLOR, dtype=int)
    tile[:GEN_TILE_INTERIOR_SIZE, :GEN_TILE_INTERIOR_SIZE] = np.array(
        random.choices(colors, k=GEN_TILE_INTERIOR_SIZE * GEN_TILE_INTERIOR_SIZE),
        dtype=int,
    ).reshape(GEN_TILE_INTERIOR_SIZE, GEN_TILE_INTERIOR_SIZE)

    for r in range(GEN_TILE_INTERIOR_SIZE):
        tile[r, random.randrange(GEN_TILE_INTERIOR_SIZE)] = random.choice(non_blue)
    for c in range(GEN_TILE_INTERIOR_SIZE):
        tile[random.randrange(GEN_TILE_INTERIOR_SIZE), c] = random.choice(non_blue)

    interior = [(r, c) for r in range(GEN_TILE_INTERIOR_SIZE) for c in range(GEN_TILE_INTERIOR_SIZE)]
    random.shuffle(interior)
    for color, (r, c) in zip(non_blue, interior):
        tile[r, c] = color

    tile[GEN_TILE_INTERIOR_SIZE, :] = GEN_TILE_BASE_COLOR
    tile[:, GEN_TILE_INTERIOR_SIZE] = GEN_TILE_BASE_COLOR
    return tile


def _gen_phases_visible(hole_mask: np.ndarray) -> bool:
    visible = np.zeros((GEN_TILE_SIZE, GEN_TILE_SIZE), dtype=bool)
    rows, cols = np.where(~hole_mask)
    visible[rows % GEN_TILE_SIZE, cols % GEN_TILE_SIZE] = True
    return bool(visible.all())


def _gen_place_holes(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((GEN_GRID_SIZE, GEN_GRID_SIZE), dtype=int)
    for patch in patches:
        try:
            r, c = random_free_location(
                canvas,
                patch,
                transparent=GEN_HOLE_COLOR,
                border=GEN_PLACEMENT_BORDER,
                padding=GEN_PLACEMENT_PADDING,
                padding_connectivity=GEN_PLACEMENT_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=GEN_HOLE_COLOR)
    return canvas.astype(bool)


def _gen_attempt() -> np.ndarray | None:
    tile = _gen_tile()
    output = np.tile(tile, (GEN_TILE_REPEATS, GEN_TILE_REPEATS))[:GEN_GRID_SIZE, :GEN_GRID_SIZE]

    region_count = random.randint(GEN_REGION_COUNT_MIN, GEN_REGION_COUNT_MAX)
    total_black = random.randint(GEN_TOTAL_BLACK_MIN, GEN_TOTAL_BLACK_MAX)
    patches = _gen_hole_patches(region_count, total_black)
    if patches is None:
        return None

    hole_mask = _gen_place_holes(patches)
    if hole_mask is None:
        return None
    if int(hole_mask.sum()) != total_black:
        return None
    if not _gen_phases_visible(hole_mask):
        return None

    inp = output.copy()
    inp[hole_mask] = GEN_HOLE_COLOR
    try:
        if not np.array_equal(main(inp), output):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c663677b input")
