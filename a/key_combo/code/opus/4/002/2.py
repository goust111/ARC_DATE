from __future__ import annotations

import itertools
import random

import numpy as np


BLUE = 1
YELLOW = 4
GREY = 5
PINK = 6
PATTERN_COLORS = (BLUE, YELLOW)

MAX_ENUM_UNKNOWN = 12
GENERATION_ATTEMPTS = 500
TILE_ATTEMPTS = 200
MIN_PINK_PCT = 15
MAX_PINK_PCT = 30


def tile_fill(shape: tuple[int, int], tile: np.ndarray, *, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Repeat a small tile over a grid, following the arc_utils (r,c) convention."""
    height, width = map(int, shape)
    tile_h, tile_w = map(int, tile.shape)
    row_offset, col_offset = map(int, offset)
    out = np.empty((height, width), dtype=int)
    for r in range(height):
        rr = (r - row_offset) % tile_h
        for c in range(width):
            cc = (c - col_offset) % tile_w
            out[r, c] = int(tile[rr, cc])
    return out


def _separator_positions(sizes: list[int]) -> list[int]:
    positions: list[int] = []
    cursor = int(sizes[0])
    for size in sizes[1:]:
        positions.append(cursor)
        cursor += 1 + int(size)
    return positions


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in separators) + [int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((left + 1, right))
    return spans


def _partition_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_separators = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_separators = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_spans = _spans(grid.shape[0], row_separators)
    col_spans = _spans(grid.shape[1], col_separators)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected at least a 2x2 partition grid"
    return row_spans, col_spans


def _consistent_partial_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    visible = np.argwhere((grid != GREY) & (grid != PINK))
    if visible.size == 0:
        return None

    for r, c in visible:
        value = int(grid[r, c])
        if value not in PATTERN_COLORS:
            return None
        rr = int(r) % row_period
        cc = int(c) % col_period
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
        elif tile[rr, cc] != value:
            return None
    return tile


def _completed_tiles(partial: np.ndarray) -> list[np.ndarray]:
    unknown = np.argwhere(partial == -1)
    if unknown.size == 0:
        return [partial]
    if len(unknown) > MAX_ENUM_UNKNOWN:
        return []

    tiles: list[np.ndarray] = []
    for values in itertools.product(PATTERN_COLORS, repeat=len(unknown)):
        tile = partial.copy()
        for (rr, cc), value in zip(unknown, values):
            tile[int(rr), int(cc)] = int(value)
        tiles.append(tile)
    return tiles


def _infer_periodic_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    visible_colors = sorted(set(map(int, g[(g != GREY) & (g != PINK)].ravel())))
    assert visible_colors == [BLUE, YELLOW], "expected visible blue/yellow pattern pixels"

    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, np.ndarray]] = []
    best_area: int | None = None
    seen: set[tuple[int, int, bytes]] = set()

    period_pairs = sorted(
        (row_period * col_period, row_period, col_period)
        for row_period in range(1, height + 1)
        for col_period in range(1, width + 1)
    )
    for area, row_period, col_period in period_pairs:
        if best_area is not None and area > best_area:
            break

        partial = _consistent_partial_tile(g, row_period, col_period)
        if partial is None:
            continue

        for tile in _completed_tiles(partial):
            if set(map(int, np.unique(tile))) != set(PATTERN_COLORS):
                continue
            key = (int(row_period), int(col_period), tile.tobytes())
            if key in seen:
                continue
            seen.add(key)
            if best_area is None:
                best_area = int(area)
            candidates.append((int(row_period), int(col_period), tile))

    assert candidates, "visible pixels must define a periodic tile"
    assert len(candidates) == 1, "minimum-area periodic tile must be unique"
    return candidates[0]


def _majority_color(block: np.ndarray) -> int:
    blue_count = int(np.count_nonzero(block == BLUE))
    yellow_count = int(np.count_nonzero(block == YELLOW))
    return BLUE if blue_count >= yellow_count else YELLOW


def _render_output_from_clean(
    clean_pattern: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    row_separators: list[int],
    col_separators: list[int],
) -> np.ndarray:
    out = clean_pattern.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            winner = _majority_color(clean_pattern[r0:r1, c0:c1])
            out[r0:r1, c0:c1] = winner
    if row_separators:
        out[np.array(row_separators), :] = GREY
    if col_separators:
        out[:, np.array(col_separators)] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a pink-corrupted periodic blue/yellow pattern, then recolor each grey-bounded partition by its restored local majority.

    Input structure:
    - The grid contains a global blue (1) / yellow (4) periodic pattern.
    - Some pattern cells are replaced by pink (6) missing-value markers.
    - Full grey (5) rows and columns split the non-grey area into rectangular partitions.

    Rule:
    - Ignore grey separators and treat every pink cell as unknown.
    - Infer the unique minimum-area row/column periodic tile consistent with all visible blue/yellow cells.
    - Restore each pink cell from that periodic tile at the same absolute coordinate.
    - For every rectangle bounded by grey separator rows/columns, count restored blue and yellow cells and repaint the whole partition with its majority color, breaking ties in favor of blue.
    - Keep the grey separator lines unchanged.

    Key insight:
    The pink cells must be completed before counting, because the partition majority is defined on the hidden periodic pattern rather than on the partially occluded observation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans = _partition_spans(g)
    row_period, col_period, tile = _infer_periodic_tile(g)

    restored = g.copy()
    for r, c in np.argwhere(restored == PINK):
        restored[int(r), int(c)] = int(tile[int(r) % row_period, int(c) % col_period])

    out = restored.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            winner = _majority_color(restored[r0:r1, c0:c1])
            out[r0:r1, c0:c1] = winner
    out[g == GREY] = GREY
    return out


def _tile_has_period(tile: np.ndarray, row_period: int, col_period: int) -> bool:
    height, width = map(int, tile.shape)
    for r in range(height):
        for c in range(width):
            if int(tile[r, c]) != int(tile[r % row_period, c % col_period]):
                return False
    return True


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    height, width = map(int, tile.shape)
    candidates = [
        (row_period * col_period, row_period, col_period)
        for row_period in range(1, height + 1)
        for col_period in range(1, width + 1)
        if _tile_has_period(tile, row_period, col_period)
    ]
    _area, row_period, col_period = min(candidates)
    return int(row_period), int(col_period)


def _random_tile() -> np.ndarray | None:
    row_period = random.randint(2, 4)
    col_period = random.randint(2, 4)
    area = row_period * col_period
    low = max(1, area // 3)
    high = min(area - 1, area - area // 3)
    for _ in range(TILE_ATTEMPTS):
        yellow_count = random.randint(low, high)
        flat = [BLUE] * area
        for idx in random.sample(range(area), yellow_count):
            flat[idx] = YELLOW
        tile = np.array(flat, dtype=int).reshape(row_period, col_period)
        if _minimal_period(tile) == (row_period, col_period):
            return tile
    return None


def _partition_infos(
    clean_pattern: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> list[dict]:
    infos: list[dict] = []
    for row_index, (r0, r1) in enumerate(row_spans):
        for col_index, (c0, c1) in enumerate(col_spans):
            block = clean_pattern[r0:r1, c0:c1]
            coords = [(r, c) for r in range(r0, r1) for c in range(c0, c1)]
            blue_count = int(np.count_nonzero(block == BLUE))
            yellow_count = int(np.count_nonzero(block == YELLOW))
            infos.append(
                {
                    "index": (row_index, col_index),
                    "coords": coords,
                    "blue": blue_count,
                    "yellow": yellow_count,
                    "winner": BLUE if blue_count >= yellow_count else YELLOW,
                }
            )
    return infos


def _residue_groups(
    shape: tuple[int, int],
    row_period: int,
    col_period: int,
    row_separators: list[int],
    col_separators: list[int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    row_blocked = set(map(int, row_separators))
    col_blocked = set(map(int, col_separators))
    groups = {(rr, cc): [] for rr in range(row_period) for cc in range(col_period)}
    for r in range(shape[0]):
        if r in row_blocked:
            continue
        for c in range(shape[1]):
            if c in col_blocked:
                continue
            groups[(r % row_period, c % col_period)].append((r, c))
    return groups


def _pink_bounds(total_cells: int) -> tuple[int, int]:
    low = (MIN_PINK_PCT * int(total_cells) + 99) // 100
    high = (MAX_PINK_PCT * int(total_cells)) // 100
    return int(low), int(high)


def _build_instance() -> np.ndarray | None:
    tile = _random_tile()
    if tile is None:
        return None
    row_period, col_period = map(int, tile.shape)

    row_sizes = [random.randint(5, 8), random.randint(5, 8)]
    col_count = random.choice([2, 3])
    col_sizes = [random.randint(5, 8) for _ in range(col_count)]

    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    if max(height, width) > 30:
        return None

    row_separators = _separator_positions(row_sizes)
    col_separators = _separator_positions(col_sizes)
    row_spans = _spans(height, row_separators)
    col_spans = _spans(width, col_separators)
    if any((r1 - r0) * (c1 - c0) < 9 for r0, r1 in row_spans for c0, c1 in col_spans):
        return None

    clean_pattern = tile_fill((height, width), tile)
    partition_infos = _partition_infos(clean_pattern, row_spans, col_spans)
    if len({info["winner"] for info in partition_infos}) < 2:
        return None

    groups = _residue_groups((height, width), row_period, col_period, row_separators, col_separators)
    if any(len(coords) == 0 for coords in groups.values()):
        return None

    preserve: set[tuple[int, int]] = {random.choice(coords) for coords in groups.values()}
    for info in partition_infos:
        preserve.add(random.choice(info["coords"]))

    flip_options: list[tuple[int, dict, list[tuple[int, int]]]] = []
    for info in partition_infos:
        winner = int(info["winner"])
        majority_cells = [
            coord
            for coord in info["coords"]
            if int(clean_pattern[coord]) == winner and coord not in preserve
        ]
        if winner == BLUE:
            needed = int(info["blue"] - info["yellow"] + 1)
        else:
            needed = int(info["yellow"] - info["blue"] + 1)
        if len(majority_cells) >= needed:
            flip_options.append((needed, info, majority_cells))
    if not flip_options:
        return None

    min_needed = min(item[0] for item in flip_options)
    needed, target_info, majority_cells = random.choice([item for item in flip_options if item[0] == min_needed])
    pink_cells = set(random.sample(majority_cells, needed))

    all_non_grey = [(r, c) for r in range(height) for c in range(width) if r not in row_separators and c not in col_separators]
    low_pink, high_pink = _pink_bounds(height * width)
    max_extra = len([coord for coord in all_non_grey if coord not in preserve and coord not in pink_cells])
    if max_extra + len(pink_cells) < low_pink:
        return None

    total_pink = random.randint(max(low_pink, len(pink_cells)), min(high_pink, len(pink_cells) + max_extra))
    extra_candidates = [
        coord
        for coord in all_non_grey
        if coord not in preserve
        and coord not in pink_cells
        and (coord not in target_info["coords"] or int(clean_pattern[coord]) == int(target_info["winner"]))
    ]
    extra_needed = total_pink - len(pink_cells)
    if len(extra_candidates) < extra_needed:
        return None
    pink_cells.update(random.sample(extra_candidates, extra_needed))

    inp = clean_pattern.copy()
    if row_separators:
        inp[np.array(row_separators), :] = GREY
    if col_separators:
        inp[:, np.array(col_separators)] = GREY
    for r, c in pink_cells:
        inp[int(r), int(c)] = PINK

    expected = _render_output_from_clean(clean_pattern, row_spans, col_spans, row_separators, col_separators)
    if not np.array_equal(main(inp), expected):
        return None

    inferred_row_period, inferred_col_period, inferred_tile = _infer_periodic_tile(inp)
    if (inferred_row_period, inferred_col_period) != (row_period, col_period):
        return None
    if not np.array_equal(inferred_tile, tile):
        return None

    visible = (inp != GREY) & (inp != PINK)
    for rr in range(row_period):
        for cc in range(col_period):
            if not np.any(visible & ((np.arange(height)[:, None] % row_period) == rr) & ((np.arange(width)[None, :] % col_period) == cc)):
                return None

    visible_majority_diff = False
    for info in partition_infos:
        r0 = min(r for r, _c in info["coords"])
        r1 = max(r for r, _c in info["coords"]) + 1
        c0 = min(c for _r, c in info["coords"])
        c1 = max(c for _r, c in info["coords"]) + 1
        block = inp[r0:r1, c0:c1]
        blue_visible = int(np.count_nonzero(block == BLUE))
        yellow_visible = int(np.count_nonzero(block == YELLOW))
        visible_winner = BLUE if blue_visible >= yellow_visible else YELLOW
        if visible_winner != int(info["winner"]):
            visible_majority_diff = True
            break
    if not visible_majority_diff:
        return None

    pink_count = int(np.count_nonzero(inp == PINK))
    if not (low_pink <= pink_count <= high_pink):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
