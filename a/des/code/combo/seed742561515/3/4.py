from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SEED_COLORS = (BLUE, RED, GREEN)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CANONICAL_DIRS = {
    BLUE: ((0, -1), (0, 1)),
    RED: ((-1, 0), (1, 0)),
    GREEN: ((-1, 0), (1, 0), (0, -1), (0, 1)),
}

TOP_ROW_HEIGHT = 3
LEGEND_CELL_SIZE = 3
ROOM_ROW_RANGE = (2, 4)
ROOM_COL_RANGE = (4, 6)
ROOM_SIZE_RANGE = (3, 6)
GEN_RETRIES = 300


def _wall_rows(g: np.ndarray) -> list[int]:
    return [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]


def _wall_cols(g: np.ndarray) -> list[int]:
    return [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]


def _room_spans(size: int, walls: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *walls, size]
    spans: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _assert_wall_grid(g: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> None:
    row_mask = np.zeros(g.shape[0], dtype=bool)
    col_mask = np.zeros(g.shape[1], dtype=bool)
    row_mask[wall_rows] = True
    col_mask[wall_cols] = True
    grey_mask = g == GREY
    expected = row_mask[:, None] | col_mask[None, :]
    assert np.array_equal(grey_mask, expected), "grey pixels must form a complete wall grid"


def _legend_mapping(
    g: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> dict[int, tuple[tuple[int, int], ...]]:
    assert row_spans and len(col_spans) >= 3, "expected a top-left legend strip"
    lr0, lr1 = row_spans[0]
    assert lr1 - lr0 + 1 == LEGEND_CELL_SIZE, "legend row must be 3 cells tall"

    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    for j in range(3):
        lc0, lc1 = col_spans[j]
        assert lc1 - lc0 + 1 == LEGEND_CELL_SIZE, "legend cell must be 3 cells wide"
        cell = g[lr0 : lr1 + 1, lc0 : lc1 + 1]
        colors = [int(v) for v in np.unique(cell) if int(v) != BG]
        assert len(colors) == 1, "each legend example must use exactly one color"
        color = colors[0]

        center = (lr0 + 1, lc0 + 1)
        assert int(g[center]) == color, "legend example must have a colored center"

        dirs: list[tuple[int, int]] = []
        coords = np.argwhere(cell == color)
        for rr, cc in coords:
            ar = lr0 + int(rr)
            ac = lc0 + int(cc)
            if (ar, ac) == center:
                continue
            dr = ar - center[0]
            dc = ac - center[1]
            assert abs(dr) + abs(dc) == 1, "legend growth must use unit cardinal steps"
            dirs.append((dr, dc))

        assert dirs, "legend example must encode at least one direction"
        key = tuple(sorted(dirs))
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = key

    assert len(mapping) == 3, "legend must contain three color-direction pairs"
    return mapping


def _grow(
    base: np.ndarray,
    seeds: list[tuple[tuple[int, int], int]],
    mapping: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    out = base.copy()
    h, w = base.shape
    for (r, c), color in seeds:
        for dr, dc in mapping[color]:
            cr, cc = r, c
            while True:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if int(base[nr, nc]) == GREY:
                    break
                out[nr, nc] = color
                cr, cc = nr, nc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top-left color legend, then grow each room's seed along that color's
    legend directions until a grey wall or the grid edge blocks further progress.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_rows = _wall_rows(g)
    wall_cols = _wall_cols(g)
    assert 0 in wall_rows and 0 in wall_cols, "top-left legend must be enclosed by walls"
    _assert_wall_grid(g, wall_rows, wall_cols)

    row_spans = _room_spans(g.shape[0], wall_rows)
    col_spans = _room_spans(g.shape[1], wall_cols)
    mapping = _legend_mapping(g, row_spans, col_spans)

    seeds: list[tuple[tuple[int, int], int]] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if ri == 0 and ci < 3:
                continue
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            coords = np.argwhere((room != BG) & (room != GREY))
            assert coords.shape[0] == 1, "each non-legend room must contain exactly one seed"
            sr, sc = map(int, coords[0])
            color = int(room[sr, sc])
            assert color in mapping, "room seed color must appear in the legend"
            seeds.append(((r0 + sr, c0 + sc), color))

    return _grow(g, seeds, mapping)


def _build_layout(
    row_heights: list[int],
    col_widths: list[int],
    bottom_border: bool,
    right_border: bool,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = 1 + sum(row_heights) + (len(row_heights) - 1) + (1 if bottom_border else 0)
    width = 1 + sum(col_widths) + (len(col_widths) - 1) + (1 if right_border else 0)
    grid = np.full((height, width), BG, dtype=int)

    grid[0, :] = GREY
    row_spans: list[tuple[int, int]] = []
    r = 1
    for i, room_h in enumerate(row_heights):
        row_spans.append((r, r + room_h - 1))
        r += room_h
        if i < len(row_heights) - 1 or bottom_border:
            grid[r, :] = GREY
            r += 1

    grid[:, 0] = GREY
    col_spans: list[tuple[int, int]] = []
    c = 1
    for j, room_w in enumerate(col_widths):
        col_spans.append((c, c + room_w - 1))
        c += room_w
        if j < len(col_widths) - 1 or right_border:
            grid[:, c] = GREY
            c += 1

    return grid, row_spans, col_spans


def _paint_legend(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> None:
    legend_colors = list(SEED_COLORS)
    random.shuffle(legend_colors)
    r0, r1 = row_spans[0]
    assert r1 - r0 + 1 == LEGEND_CELL_SIZE
    for cell_idx, color in enumerate(legend_colors):
        c0, c1 = col_spans[cell_idx]
        assert c1 - c0 + 1 == LEGEND_CELL_SIZE
        center = (r0 + 1, c0 + 1)
        grid[center] = color
        for dr, dc in CANONICAL_DIRS[color]:
            grid[center[0] + dr, center[1] + dc] = color


def _room_keys(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (ri, ci)
        for ri in range(len(row_spans))
        for ci in range(len(col_spans))
        if not (ri == 0 and ci < 3)
    ]


def _choose_seed_colors(
    rooms: list[tuple[int, int]],
    row_count: int,
    col_count: int,
    bottom_border: bool,
    right_border: bool,
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    assigned = {room: random.choice(SEED_COLORS) for room in rooms}
    locked: set[tuple[int, int]] = set()

    if not bottom_border and not right_border:
        corner = (row_count - 1, col_count - 1)
        assigned[corner] = GREEN
        locked.add(corner)
    elif not bottom_border:
        candidates = [(row_count - 1, ci) for _ri, ci in rooms if _ri == row_count - 1]
        target = random.choice(candidates)
        assigned[target] = random.choice((RED, GREEN))
        locked.add(target)
    elif not right_border:
        candidates = [(ri, col_count - 1) for ri, _ci in rooms if _ci == col_count - 1]
        target = random.choice(candidates)
        assigned[target] = random.choice((BLUE, GREEN))
        locked.add(target)

    missing = [color for color in SEED_COLORS if color not in assigned.values()]
    candidates = [room for room in rooms if room not in locked]
    random.shuffle(candidates)
    for color in missing:
        assert candidates, "expected enough rooms to realize all seed colors"
        assigned[candidates.pop()] = color

    return assigned, locked


def _seed_position_candidates(
    span_r: tuple[int, int],
    span_c: tuple[int, int],
    color: int,
    open_bottom: bool,
    open_right: bool,
) -> list[tuple[int, int]]:
    r0, r1 = span_r
    c0, c1 = span_c
    out: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if open_bottom and color in (RED, GREEN) and r == r1:
                continue
            if open_right and color in (BLUE, GREEN) and c == c1:
                continue
            out.append((r, c))
    return out


def _place_seeds(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    colors_by_room: dict[tuple[int, int], int],
    bottom_border: bool,
    right_border: bool,
) -> list[tuple[tuple[int, int], int]]:
    seeds: list[tuple[tuple[int, int], int]] = []
    last_row = len(row_spans) - 1
    last_col = len(col_spans) - 1

    for (ri, ci), color in colors_by_room.items():
        open_bottom = ri == last_row and not bottom_border
        open_right = ci == last_col and not right_border
        candidates = _seed_position_candidates(row_spans[ri], col_spans[ci], color, open_bottom, open_right)
        assert candidates, "room must contain a legal seed location"
        pos = random.choice(candidates)
        grid[pos] = color
        seeds.append((pos, color))

    return seeds


def _oracle_expected(inp: np.ndarray) -> np.ndarray:
    wall_rows = _wall_rows(inp)
    wall_cols = _wall_cols(inp)
    row_spans = _room_spans(inp.shape[0], wall_rows)
    col_spans = _room_spans(inp.shape[1], wall_cols)
    mapping = _legend_mapping(inp, row_spans, col_spans)
    seeds: list[tuple[tuple[int, int], int]] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if ri == 0 and ci < 3:
                continue
            room = inp[r0 : r1 + 1, c0 : c1 + 1]
            coords = np.argwhere((room != BG) & (room != GREY))
            sr, sc = map(int, coords[0])
            seeds.append(((r0 + sr, c0 + sc), int(room[sr, sc])))
    return _grow(inp, seeds, mapping)


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(*ROOM_ROW_RANGE)
    col_count = random.randint(*ROOM_COL_RANGE)

    row_heights = [TOP_ROW_HEIGHT] + [random.randint(*ROOM_SIZE_RANGE) for _ in range(row_count - 1)]
    col_widths = [LEGEND_CELL_SIZE, LEGEND_CELL_SIZE, LEGEND_CELL_SIZE]
    col_widths.extend(random.randint(*ROOM_SIZE_RANGE) for _ in range(col_count - 3))

    bottom_border = random.choice((False, True))
    right_border = random.choice((False, True))
    grid, row_spans, col_spans = _build_layout(row_heights, col_widths, bottom_border, right_border)
    _paint_legend(grid, row_spans, col_spans)

    rooms = _room_keys(row_spans, col_spans)
    assert len(rooms) >= 3, "generator needs at least three non-legend rooms"
    colors_by_room, _locked = _choose_seed_colors(rooms, row_count, col_count, bottom_border, right_border)
    seeds = _place_seeds(grid, row_spans, col_spans, colors_by_room, bottom_border, right_border)

    counts = {color: sum(seed_color == color for _pos, seed_color in seeds) for color in SEED_COLORS}
    assert all(counts[color] >= 1 for color in SEED_COLORS), "every sample must show all legend behaviors"

    expected = _oracle_expected(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-and-room growth puzzle")
