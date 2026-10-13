from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "expected a unique dominant color"
    return int(winners[0])


def _separator_color(g: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int]] = []
    for color in map(int, np.unique(g)):
        if color == int(bg):
            continue
        row_hits = int(np.count_nonzero(np.all(g == color, axis=1)))
        col_hits = int(np.count_nonzero(np.all(g == color, axis=0)))
        if row_hits and col_hits:
            candidates.append((row_hits + col_hits, int(color)))
    assert candidates, "expected a separator color spanning full rows and columns"
    best = max(score for score, _ in candidates)
    winners = [color for score, color in candidates if score == best]
    assert len(winners) == 1, "expected a unique separator color"
    return int(winners[0])


def _bands_from_lines(lines: np.ndarray, limit: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(map(int, lines))) + [int(limit)]
    bands: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if a + 1 <= b - 1:
            bands.append((int(a + 1), int(b - 1)))
    return bands


def _minimal_period(arr: np.ndarray, axis: int) -> int:
    length = int(arr.shape[axis])
    for period in range(1, length + 1):
        ok = True
        for idx in range(length):
            ref = idx % period
            if axis == 0:
                if not np.array_equal(arr[idx, :], arr[ref, :]):
                    ok = False
                    break
            else:
                if not np.array_equal(arr[:, idx], arr[:, ref]):
                    ok = False
                    break
        if ok:
            return int(period)
    raise AssertionError("period search failed")


def _minimal_tile(room: np.ndarray) -> np.ndarray:
    tile_h = _minimal_period(room, axis=0)
    tile_w = _minimal_period(room, axis=1)
    return room[:tile_h, :tile_w].copy()


def _tile_to_shape(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    tile_h, tile_w = map(int, tile.shape)
    reps_h = (h + tile_h - 1) // tile_h
    reps_w = (w + tile_w - 1) // tile_w
    return np.tile(tile, (reps_h, reps_w))[:h, :w]


def _parse_lattice(
    g: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    bg = _unique_mode(g)
    sep = _separator_color(g, bg)
    row_lines = np.flatnonzero(np.all(g == int(sep), axis=1))
    col_lines = np.flatnonzero(np.all(g == int(sep), axis=0))
    row_bands = _bands_from_lines(row_lines, g.shape[0])
    col_bands = _bands_from_lines(col_lines, g.shape[1])
    assert len(row_bands) >= 2, "expected at least one target row and one library row"
    assert len(col_bands) >= 1, "expected at least one room column"
    return int(bg), int(sep), row_bands, col_bands


def _marker_color(
    g: np.ndarray,
    bg: int,
    sep: int,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> int:
    colors: set[int] = set()
    for r0, r1 in row_bands[:-1]:
        for c0, c1 in col_bands:
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            colors.update(map(int, np.unique(room)))
    colors.discard(int(bg))
    colors.discard(int(sep))
    assert len(colors) == 1, "expected a single marker color in target rooms"
    return int(next(iter(colors)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each target room with the library pattern selected by its marker count.

    Input structure:
    - The dominant color is the background floor.
    - One single-color separator lattice of full grey rows and columns splits the grid into rectangular rooms.
    - The bottom room row is a pattern library: each library room contains one repeated two-color template.
    - Every room above the library contains only background plus several same-color marker pixels.

    Rule:
    - Parse the room grid from the separator rows and columns and treat the bottom row of rooms as the library.
    - In each library room, recover the smallest tile whose repetitions reproduce that room's pattern.
    - For every target room above, count its marker pixels; use that count as a 1-based index into the library from left to right.
    - Tile the chosen library pattern from the target room's top-left corner, covering the whole room and overwriting the markers.
    - Keep the separator lattice and the library row unchanged.

    Key insight:
    - The markers are not payload pixels: their count selects which bottom-row template should be extracted and periodically tiled into that room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, sep, row_bands, col_bands = _parse_lattice(g)
    marker = _marker_color(g, bg, sep, row_bands, col_bands)

    library_r0, library_r1 = row_bands[-1]
    library_tiles: list[np.ndarray] = []
    for c0, c1 in col_bands:
        room = g[library_r0 : library_r1 + 1, c0 : c1 + 1]
        tile = _minimal_tile(room)
        assert np.any(tile != int(bg)), "library tiles must contain a visible pattern"
        library_tiles.append(tile)

    out = g.copy()
    for r0, r1 in row_bands[:-1]:
        for c0, c1 in col_bands:
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            room_colors = set(map(int, np.unique(room)))
            assert room_colors <= {int(bg), int(marker)}, "target rooms may only contain bg and markers"
            count = int(np.count_nonzero(room == int(marker)))
            assert 1 <= count <= len(library_tiles), "marker count must index the library"
            out[r0 : r1 + 1, c0 : c1 + 1] = _tile_to_shape(library_tiles[count - 1], room.shape)

    return out


# --------------- generator ---------------

ARC_LIMIT = 30
GREY = 5
RED = 2
ALL_COLORS = tuple(range(10))
TARGET_ROW_COUNT_RANGE = (2, 3)
PATTERN_COUNT_RANGE = (3, 5)
TARGET_ROOM_HEIGHT_RANGE = (4, 6)
LIBRARY_ROOM_HEIGHT_RANGE = (3, 4)
ROOM_WIDTH_RANGE = (3, 5)
TILE_DENSITY_RANGE = (0.2, 0.7)
GEN_RETRIES = 400


def _base_tiles() -> list[np.ndarray]:
    return [
        np.array([[1, 0]], dtype=int),
        np.array([[1], [0]], dtype=int),
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 0, 1], [0, 1, 0, 0]], dtype=int),
    ]


def _tile_catalog() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in _base_tiles():
        transforms = [
            base,
            np.fliplr(base),
            np.flipud(base),
            np.flipud(np.fliplr(base)),
        ]
        if base.shape[0] != base.shape[1]:
            t = base.T
            transforms.extend([t, np.fliplr(t), np.flipud(t), np.flipud(np.fliplr(t))])
        for variant in transforms:
            for candidate in (variant, 1 - variant):
                if not np.any(candidate) or np.all(candidate):
                    continue
                density = float(candidate.mean())
                if not TILE_DENSITY_RANGE[0] <= density <= TILE_DENSITY_RANGE[1]:
                    continue
                minimal = _minimal_tile(candidate)
                if minimal.shape != candidate.shape or not np.array_equal(minimal, candidate):
                    continue
                sig = (tuple(map(int, candidate.shape)), candidate.astype(np.uint8).tobytes())
                if sig in seen:
                    continue
                seen.add(sig)
                variants.append(candidate.astype(int))
    assert variants, "expected a nonempty tile catalog"
    return variants


TILE_CATALOG = _tile_catalog()


def _sample_room_sizes(count: int, low: int, high: int, max_extent: int) -> list[int] | None:
    for _ in range(200):
        sizes = [random.randint(low, high) for _ in range(int(count))]
        if sum(sizes) + int(count) + 1 <= int(max_extent):
            return sizes
    return None


def _choose_tiles(lib_h: int, widths: list[int], count: int) -> list[np.ndarray] | None:
    chosen: list[np.ndarray] = []
    used: set[tuple[tuple[int, int], bytes]] = set()
    width_order = sorted(range(int(count)), key=lambda i: widths[i])
    slots: list[np.ndarray | None] = [None] * int(count)
    for idx in width_order:
        room_w = int(widths[idx])
        candidates = [
            tile
            for tile in TILE_CATALOG
            if tile.shape[0] <= int(lib_h)
            and tile.shape[1] <= room_w
            and (tuple(map(int, tile.shape)), tile.astype(np.uint8).tobytes()) not in used
        ]
        if not candidates:
            return None
        tile = random.choice(candidates).copy()
        sig = (tuple(map(int, tile.shape)), tile.astype(np.uint8).tobytes())
        used.add(sig)
        slots[idx] = tile
    for tile in slots:
        assert tile is not None
        chosen.append(tile)
    return chosen


def _row_spans(heights: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 1
    for height in heights:
        end = start + int(height) - 1
        spans.append((int(start), int(end)))
        start = end + 2
    return spans


def _col_spans(widths: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 1
    for width in widths:
        end = start + int(width) - 1
        spans.append((int(start), int(end)))
        start = end + 2
    return spans


def _fill_library_room(bg: int, fg: int, shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    mask = _tile_to_shape(tile, shape).astype(bool)
    return np.where(mask, int(fg), int(bg))


def _balanced_counts(room_count: int, pattern_count: int) -> list[int]:
    counts = list(range(1, int(pattern_count) + 1)) * (int(room_count) // int(pattern_count))
    leftovers = int(room_count) % int(pattern_count)
    if leftovers:
        counts.extend(random.sample(range(1, int(pattern_count) + 1), leftovers))
    random.shuffle(counts)
    return counts


def _place_markers(room: np.ndarray, marker_color: int, count: int) -> None:
    h, w = map(int, room.shape)
    cells = random.sample(range(h * w), int(count))
    for idx in cells:
        r = idx // w
        c = idx % w
        room[r, c] = int(marker_color)


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [color for color in ALL_COLORS if color not in (GREY, RED)]
    bg = random.choice(bg_choices)

    pattern_count = random.randint(*PATTERN_COUNT_RANGE)
    target_rows = random.randint(*TARGET_ROW_COUNT_RANGE)
    widths = _sample_room_sizes(pattern_count, *ROOM_WIDTH_RANGE, ARC_LIMIT)
    if widths is None:
        return None

    target_heights = _sample_room_sizes(target_rows, *TARGET_ROOM_HEIGHT_RANGE, ARC_LIMIT)
    if target_heights is None:
        return None
    library_heights = _sample_room_sizes(1, *LIBRARY_ROOM_HEIGHT_RANGE, ARC_LIMIT)
    if library_heights is None:
        return None
    heights = target_heights + library_heights

    grid_h = sum(heights) + len(heights) + 1
    grid_w = sum(widths) + len(widths) + 1
    if grid_h > ARC_LIMIT or grid_w > ARC_LIMIT:
        return None

    palette = [color for color in ALL_COLORS if color not in (bg, GREY, RED)]
    if len(palette) < pattern_count:
        return None
    pattern_colors = random.sample(palette, pattern_count)

    tiles = _choose_tiles(heights[-1], widths, pattern_count)
    if tiles is None:
        return None

    grid = np.full((grid_h, grid_w), GREY, dtype=int)
    row_spans = _row_spans(heights)
    col_spans = _col_spans(widths)

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    library_r0, library_r1 = row_spans[-1]
    for (c0, c1), fg, tile in zip(col_spans, pattern_colors, tiles):
        patch = _fill_library_room(bg, fg, (library_r1 - library_r0 + 1, c1 - c0 + 1), tile)
        grid[library_r0 : library_r1 + 1, c0 : c1 + 1] = patch

    counts = _balanced_counts(target_rows * pattern_count, pattern_count)
    idx = 0
    for r0, r1 in row_spans[:-1]:
        for c0, c1 in col_spans:
            room = grid[r0 : r1 + 1, c0 : c1 + 1]
            _place_markers(room, RED, counts[idx])
            idx += 1

    try:
        if _unique_mode(grid) != int(bg):
            return None
    except AssertionError:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
