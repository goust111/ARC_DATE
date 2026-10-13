from __future__ import annotations

import random

import numpy as np


BLUE = 1
YELLOW = 4
ORANGE = 7
BACKGROUND = 0

TILE_COUNT = 4
TILE_SIZE = 4
STRIDE = TILE_SIZE + 1
GRID_SIZE = TILE_COUNT * TILE_SIZE + TILE_COUNT + 1
LANES = ((1, 1), (1, 2), (2, 1), (2, 2))


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == ORANGE)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == ORANGE)]


def _tile_starts(separators: list[int]) -> list[int]:
    assert len(separators) == TILE_COUNT + 1, "expected a 4x4 lattice of separator lines"
    starts: list[int] = []
    for left, right in zip(separators, separators[1:]):
        assert right - left == STRIDE, "unexpected tile stride"
        starts.append(left + 1)
    return starts


def _square_info(tile: np.ndarray) -> tuple[int, int, int]:
    blue = np.argwhere(tile == BLUE)
    assert len(blue) > 0, "each tile must contain one blue square"
    r0, c0 = blue.min(axis=0)
    r1, c1 = blue.max(axis=0)
    side_r = int(r1 - r0 + 1)
    side_c = int(c1 - c0 + 1)
    assert side_r == side_c, "expected a square"
    assert int(len(blue)) == side_r * side_c, "expected a solid square"
    assert side_r in (2, 3, 4), "square side must be 2, 3, or 4"
    return side_r, int(r0), int(c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark odd square centers inside a tiled lattice and connect matching center lanes across tile rows and columns.

    Input structure:
    - Orange separator rows and columns partition the grid into a 4x4 lattice of 4x4 interior tiles.
    - Each tile contains exactly one blue solid square of side 2, 3, or 4 on a black background.
    - A 3x3 square can sit in different tile-relative positions, so its center lands on one of four possible subcell lanes.

    Rule:
    - Parse each tile independently and recover the blue square's side length and placement.
    - If the square side is odd (here, only 3), recolor its center pixel yellow and record that center's tile-relative lane.
    - Group marked centers by their tile-relative lane; within each group, connect consecutive centers that share a tile row or a tile column by painting yellow through every intervening tile at that same lane position.
    - Leave all other pixels unchanged.

    Key insight:
    - Oddness decides which tiles contribute endpoints, while the tile-relative center lane decides which endpoints are allowed to connect on the coarser 4x4 tile lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_starts = _tile_starts(_separator_indices(g, axis=0))
    col_starts = _tile_starts(_separator_indices(g, axis=1))

    out = g.copy()
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for tile_r, r0 in enumerate(row_starts):
        for tile_c, c0 in enumerate(col_starts):
            tile = g[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]
            assert tile.shape == (TILE_SIZE, TILE_SIZE), "unexpected tile shape"
            side, top, left = _square_info(tile)
            if side % 2 == 0:
                continue

            lane_r = top + side // 2
            lane_c = left + side // 2
            out[r0 + lane_r, c0 + lane_c] = YELLOW
            groups.setdefault((lane_r, lane_c), []).append((tile_r, tile_c))

    for (lane_r, lane_c), points in groups.items():
        by_row: dict[int, list[int]] = {}
        by_col: dict[int, list[int]] = {}
        for tile_r, tile_c in points:
            by_row.setdefault(tile_r, []).append(tile_c)
            by_col.setdefault(tile_c, []).append(tile_r)

        for tile_r, cols in by_row.items():
            cols = sorted(cols)
            for left_col, right_col in zip(cols, cols[1:]):
                for tile_c in range(left_col, right_col + 1):
                    r = row_starts[tile_r] + lane_r
                    c = col_starts[tile_c] + lane_c
                    out[r, c] = YELLOW

        for tile_c, rows in by_col.items():
            rows = sorted(rows)
            for top_row, bottom_row in zip(rows, rows[1:]):
                for tile_r in range(top_row, bottom_row + 1):
                    r = row_starts[tile_r] + lane_r
                    c = col_starts[tile_c] + lane_c
                    out[r, c] = YELLOW

    return out


def _tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return 1 + tile_r * STRIDE, 1 + tile_c * STRIDE


def _paint_square(grid: np.ndarray, tile_r: int, tile_c: int, side: int, top: int, left: int) -> None:
    r0, c0 = _tile_origin(tile_r, tile_c)
    grid[r0 + top : r0 + top + side, c0 + left : c0 + left + side] = BLUE


def _odd_square_for_lane(lane: tuple[int, int]) -> tuple[int, int, int]:
    lane_r, lane_c = lane
    return 3, lane_r - 1, lane_c - 1


def _even_square_options(
    lane: tuple[int, int] | None = None,
    *,
    must_cover: bool | None = None,
) -> list[tuple[int, int, int]]:
    options: list[tuple[int, int, int]] = []
    for side in (2, 4):
        for top in range(TILE_SIZE - side + 1):
            for left in range(TILE_SIZE - side + 1):
                if lane is None or must_cover is None:
                    options.append((side, top, left))
                    continue
                lane_r, lane_c = lane
                covers = top <= lane_r < top + side and left <= lane_c < left + side
                if covers == must_cover:
                    options.append((side, top, left))
    return options


def _pick_extra_lane(
    tile: tuple[int, int],
    odd_tiles: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int] | None:
    tile_r, tile_c = tile
    choices = list(LANES)
    random.shuffle(choices)
    for lane in choices:
        conflict = False
        for (other_r, other_c), other_lane in odd_tiles.items():
            if lane == other_lane and (tile_r == other_r or tile_c == other_c):
                conflict = True
                break
        if not conflict:
            return lane
    return None


def _build_layout() -> dict[tuple[int, int], tuple[int, int, int]]:
    col_pair, free_col, mid_col = random.choice([((0, 2), 3, 1), ((1, 3), 0, 2)])
    row_pair, free_row, mid_row = random.choice([((0, 2), 3, 1), ((1, 3), 0, 2)])

    row_group_row = free_row
    col_group_col = free_col

    lane_row = random.choice(LANES)
    lane_col = random.choice(LANES)
    distractor_lane = random.choice([lane for lane in LANES if lane != lane_row and lane != lane_col])

    odd_tiles: dict[tuple[int, int], tuple[int, int]] = {
        (row_group_row, col_pair[0]): lane_row,
        (row_group_row, col_pair[1]): lane_row,
        (row_pair[0], col_group_col): lane_col,
        (row_pair[1], col_group_col): lane_col,
        (row_group_row, col_group_col): distractor_lane,
    }

    reserved_even = {
        (row_group_row, mid_col): (lane_row, True),
        (mid_row, col_group_col): (lane_col, False),
    }

    remaining = [
        (tile_r, tile_c)
        for tile_r in range(TILE_COUNT)
        for tile_c in range(TILE_COUNT)
        if (tile_r, tile_c) not in odd_tiles and (tile_r, tile_c) not in reserved_even
    ]
    random.shuffle(remaining)

    extra_target = random.randint(1, 3)
    for tile in remaining:
        if len(odd_tiles) >= 5 + extra_target:
            break
        lane = _pick_extra_lane(tile, odd_tiles)
        if lane is not None:
            odd_tiles[tile] = lane

    assert len(odd_tiles) >= 6, "generator must create at least six odd squares"

    layout: dict[tuple[int, int], tuple[int, int, int]] = {}
    for tile, lane in odd_tiles.items():
        layout[tile] = _odd_square_for_lane(lane)

    generic_even_tiles = [
        (tile_r, tile_c)
        for tile_r in range(TILE_COUNT)
        for tile_c in range(TILE_COUNT)
        if (tile_r, tile_c) not in layout and (tile_r, tile_c) not in reserved_even
    ]
    if generic_even_tiles:
        layout[random.choice(generic_even_tiles)] = (4, 0, 0)

    for tile_r in range(TILE_COUNT):
        for tile_c in range(TILE_COUNT):
            tile = (tile_r, tile_c)
            if tile in layout:
                continue
            if tile in reserved_even:
                lane, must_cover = reserved_even[tile]
                layout[tile] = random.choice(_even_square_options(lane, must_cover=must_cover))
            else:
                layout[tile] = random.choice(_even_square_options())

    return layout


def _make_base_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND, dtype=int)
    for idx in range(0, GRID_SIZE, STRIDE):
        grid[idx, :] = ORANGE
        grid[:, idx] = ORANGE
    return grid


def generate_input() -> np.ndarray:
    grid = _make_base_grid()
    layout = _build_layout()
    for (tile_r, tile_c), (side, top, left) in layout.items():
        _paint_square(grid, tile_r, tile_c, side, top, left)
    return grid
