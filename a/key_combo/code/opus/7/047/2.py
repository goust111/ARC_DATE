from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
BAR_COLORS = (1, 2, 3, 4)
TILE_COLORS = (6, 7, 8, 9)

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (14, 18)
TILE_COUNT_CHOICES = (3, 4)
BAR_COUNT_BY_TILE_COUNT = {3: (3,), 4: (3, 4)}
TILE_HEIGHT_CHOICES = (2, 3)
TILE_WIDTH_CHOICES = (2, 3)
BAR_LENGTH_RANGE = (2, 5)
GENERATE_RETRIES = 400


@dataclass(frozen=True)
class Component:
    coords: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Bar:
    row: int
    c0: int
    c1: int
    color: int

    @property
    def length(self) -> int:
        return self.c1 - self.c0 + 1


@dataclass(frozen=True)
class Tile:
    r0: int
    c0: int
    pattern: np.ndarray

    @property
    def h(self) -> int:
        return int(self.pattern.shape[0])

    @property
    def w(self) -> int:
        return int(self.pattern.shape[1])

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[Component]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[Component] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            comps.append(Component(tuple(sorted(coords)), bbox))
    return comps


def _detect_objects(grid: np.ndarray) -> tuple[list[Bar], list[Tile]]:
    bars: list[Bar] = []
    tiles: list[Tile] = []
    for comp in _components(grid != BACKGROUND):
        r0, c0, r1, c1 = comp.bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        area = len(comp.coords)
        if h == 1 and area == w:
            color = int(patch[0, 0])
            if color == BACKGROUND or np.any(patch != color):
                raise AssertionError("expected each horizontal segment to be monochrome")
            bars.append(Bar(r0, c0, c1, color))
            continue
        if area == h * w and h >= 2 and w >= 2:
            tiles.append(Tile(r0, c0, patch.copy()))
            continue
        raise AssertionError(f"unrecognized foreground component bbox={comp.bbox}")

    bars.sort(key=lambda bar: (bar.row, bar.c0))
    tiles.sort(key=lambda tile: (tile.r0, tile.c0))
    assert bars, "expected at least one bar"
    assert tiles, "expected at least one tile"
    return bars, tiles


def _settle_bars(grid: np.ndarray, bars: list[Bar]) -> list[Bar]:
    occupied = grid != BACKGROUND
    for bar in bars:
        occupied[bar.row, bar.c0 : bar.c1 + 1] = False

    settled: list[Bar] = []
    for bar in sorted(bars, key=lambda item: (-item.row, item.c0)):
        row = bar.row
        while row + 1 < grid.shape[0] and not occupied[row + 1, bar.c0 : bar.c1 + 1].any():
            row += 1
        landed = Bar(row, bar.c0, bar.c1, bar.color)
        occupied[row, bar.c0 : bar.c1 + 1] = True
        settled.append(landed)
    return sorted(settled, key=lambda bar: (bar.row, bar.c0))


def _pair_bar_to_tile(bar: Bar, tiles: list[Tile]) -> Tile | None:
    matches = [
        tile
        for tile in tiles
        if not (bar.c1 < tile.c0 or tile.c1 < bar.c0)
    ]
    if not matches:
        return None
    assert len(matches) == 1, "each bar must overlap at most one tile interval"
    return matches[0]


def _paint_tile_stack(out: np.ndarray, tile: Tile, repeats: int) -> None:
    for k in range(repeats):
        dest_r0 = tile.r0 - k * tile.h
        dest_r1 = dest_r0 + tile.h - 1
        if dest_r1 < 0:
            break
        src_r0 = max(0, -dest_r0)
        dst_r0 = max(0, dest_r0)
        dst_r1 = dest_r1
        out[dst_r0 : dst_r1 + 1, tile.c0 : tile.c1 + 1] = tile.pattern[src_r0:, :]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each horizontal bar, use its pixel length to repeat the overlapped bottom tile upward, then erase the bar.

    Input structure:
    - The background is black.
    - Several one-cell-thick monochrome horizontal bars sit in upper rows.
    - Several solid rectangular multicolor tiles sit on the bottom rows, separated by black gaps.

    Rule:
    - Remove the bars from the occupancy map, then let them fall straight down from lower bars to upper bars so already-settled lower bars can block higher ones.
    - After falling, pair a bar with the unique bottom tile whose column interval overlaps the bar's column interval.
    - Repeat that tile vertically upward a number of times equal to the bar length, counting the original tile position as the first copy, and crop any part that would leave the top edge.
    - Erase every original bar cell to black; bars with no overlapping tile are simply erased.
    - If repeated tile regions overlap, the stack driven by the longer bar overwrites the shorter one.

    Key insight:
    - The bar color is irrelevant; gravity determines where each bar settles, but the tile repetition count comes only from the bar's geometric length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bars, tiles = _detect_objects(g)
    settled = _settle_bars(g, bars)

    out = g.copy()
    for bar in bars:
        out[bar.row, bar.c0 : bar.c1 + 1] = BACKGROUND

    paint_jobs: list[tuple[int, int, int, Tile]] = []
    for bar in settled:
        tile = _pair_bar_to_tile(bar, tiles)
        if tile is None:
            continue
        paint_jobs.append((bar.length, tile.r0, tile.c0, tile))

    for repeats, _r0, _c0, tile in sorted(paint_jobs):
        _paint_tile_stack(out, tile, repeats)
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _make_tile_pattern(height: int, width: int, used: set[bytes]) -> np.ndarray:
    for _ in range(100):
        n_colors = random.randint(2, min(3, height * width))
        colors = random.sample(TILE_COLORS, n_colors)
        pattern = np.array(
            [random.choice(colors) for _ in range(height * width)],
            dtype=int,
        ).reshape(height, width)
        for color in colors:
            pattern.flat[random.randrange(height * width)] = color
        if len(set(map(int, pattern.flat))) < 2:
            continue
        key = pattern.tobytes() + bytes((height, width))
        if key in used:
            continue
        used.add(key)
        return pattern
    raise RuntimeError("failed to build a unique tile pattern")


def _choose_tiles(h: int, w: int, tile_count: int) -> list[Tile] | None:
    widths = [random.choice(TILE_WIDTH_CHOICES) for _ in range(tile_count)]
    heights = [random.choice(TILE_HEIGHT_CHOICES) for _ in range(tile_count)]
    min_total = sum(widths) + (tile_count - 1)
    if min_total > w:
        return None
    slack = w - min_total
    extras = _random_partition(slack, tile_count + 1)
    gap_extras = extras[1:-1]
    c = extras[0]
    used_patterns: set[bytes] = set()
    tiles: list[Tile] = []
    for idx in range(tile_count):
        tile_h = heights[idx]
        tile_w = widths[idx]
        pattern = _make_tile_pattern(tile_h, tile_w, used_patterns)
        r0 = h - tile_h
        tiles.append(Tile(r0, c, pattern))
        c += tile_w
        if idx + 1 < tile_count:
            c += 1 + gap_extras[idx]
    return tiles


def _bar_start_range(bar_len: int, tile_idx: int, tiles: list[Tile], width: int) -> tuple[int, int] | None:
    tile = tiles[tile_idx]
    prev_right = tiles[tile_idx - 1].c1 if tile_idx > 0 else -1
    next_left = tiles[tile_idx + 1].c0 if tile_idx + 1 < len(tiles) else width
    start_min = max(prev_right + 1, tile.c0 - bar_len + 1)
    start_max = min(tile.c1, next_left - bar_len)
    if start_min > start_max:
        return None
    return start_min, start_max


def _overlap_interval(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    if lo > hi:
        return None
    return lo, hi


def _validate_vertical_clearance(bars: list[Bar], assignment: dict[int, int], tiles: list[Tile]) -> bool:
    for i, bar in enumerate(bars):
        tile = tiles[assignment[i]]
        overlap = _overlap_interval(bar.c0, bar.c1, tile.c0, tile.c1)
        assert overlap is not None
        lo, hi = overlap
        for j, other in enumerate(bars):
            if i == j:
                continue
            if not (bar.row < other.row < tile.r0):
                continue
            if _overlap_interval(lo, hi, other.c0, other.c1) is not None:
                return False
    return True


def _bars_are_separate(bars: list[Bar]) -> bool:
    for i, left in enumerate(bars):
        for right in bars[i + 1 :]:
            if abs(left.row - right.row) != 1:
                continue
            if _overlap_interval(left.c0, left.c1, right.c0, right.c1) is not None:
                return False
    return True


def _build_grid(h: int, w: int, bars: list[Bar], tiles: list[Tile]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for bar in bars:
        grid[bar.row, bar.c0 : bar.c1 + 1] = bar.color
    for tile in tiles:
        grid[tile.r0 : tile.r1 + 1, tile.c0 : tile.c1 + 1] = tile.pattern
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    tile_count = random.choice(TILE_COUNT_CHOICES)
    bar_count = random.choice(BAR_COUNT_BY_TILE_COUNT[tile_count])

    tiles = _choose_tiles(h, w, tile_count)
    if tiles is None:
        return None

    paired_tile_indices = sorted(random.sample(range(tile_count), bar_count))
    random.shuffle(paired_tile_indices)

    lengths: list[int] = []
    for tile_idx in paired_tile_indices:
        tile = tiles[tile_idx]
        height_limit = 1 + tile.r0 // tile.h
        prev_right = tiles[tile_idx - 1].c1 if tile_idx > 0 else -1
        next_left = tiles[tile_idx + 1].c0 if tile_idx + 1 < tile_count else w
        window_limit = next_left - prev_right - 1
        max_len = min(BAR_LENGTH_RANGE[1], height_limit, window_limit)
        if max_len < BAR_LENGTH_RANGE[0]:
            return None
        lengths.append(random.randint(BAR_LENGTH_RANGE[0], max_len))
    if len(set(lengths)) < 2:
        return None

    upper_rows = list(range(1, max(2, h // 2 + 1)))
    if len(upper_rows) < bar_count:
        return None
    rows = sorted(random.sample(upper_rows, bar_count))
    colors = random.sample(BAR_COLORS, bar_count)

    bars: list[Bar] = []
    assignment: dict[int, int] = {}
    for idx, (tile_idx, row, length, color) in enumerate(zip(paired_tile_indices, rows, lengths, colors)):
        start_range = _bar_start_range(length, tile_idx, tiles, w)
        if start_range is None:
            return None
        start = random.randint(*start_range)
        bars.append(Bar(row, start, start + length - 1, color))
        assignment[idx] = tile_idx

    if not _bars_are_separate(bars):
        return None
    if not _validate_vertical_clearance(bars, assignment, tiles):
        return None

    grid = _build_grid(h, w, bars, tiles)

    detected_bars, detected_tiles = _detect_objects(grid)
    if len(detected_bars) != bar_count or len(detected_tiles) != tile_count:
        return None
    settled = _settle_bars(grid, detected_bars)
    tile_hits = []
    for bar in settled:
        tile = _pair_bar_to_tile(bar, detected_tiles)
        if tile is None:
            return None
        tile_hits.append((tile.r0, tile.c0))
    if len(set(tile_hits)) != len(tile_hits):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 2 input")
