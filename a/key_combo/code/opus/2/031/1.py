from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, in_bounds


BG_COLOR = Color.BLACK
START_COLOR = Color.GREEN
BLOCKER_COLOR = Color.TEAL
PALETTE_POOL = [Color.BLUE, Color.RED, Color.YELLOW, Color.ORANGE, Color.MAROON]
ENTRY_SIDES = ["top", "bottom", "left", "right"]

# --------------- solver helpers ---------------


def _is_border(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _start_bar(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == START_COLOR)]
    assert len(coords) == 2, "expected exactly one 2-cell green start bar"
    (r0, c0), (r1, c1) = coords
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"

    border_cells = [cell for cell in coords if _is_border(grid.shape, cell)]
    assert len(border_cells) == 1, "expected exactly one border-touching start cell"
    border = border_cells[0]
    inner = coords[1] if coords[0] == border else coords[0]
    direction = (int(inner[0] - border[0]), int(inner[1] - border[1]))
    assert abs(direction[0]) + abs(direction[1]) == 1
    return border, inner, direction


def _extract_palette(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    mask = np.isin(grid, PALETTE_POOL)
    coords = np.argwhere(mask)
    assert 3 <= len(coords) <= 5, "expected a 3-5 cell palette line"
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})

    if len(rows) == 1:
        row = rows[0]
        assert row in {0, grid.shape[0] - 1}, "horizontal palette must lie on an edge row"
        ordered = sorted((int(c), int(grid[row, c])) for c in cols)
        columns = [c for c, _color in ordered]
        assert columns == list(range(columns[0], columns[0] + len(columns))), "palette must be contiguous"
        palette = [color for _c, color in ordered]
        cells = {(row, c) for c in columns}
    else:
        assert len(cols) == 1, "palette must be a single edge row or column"
        col = cols[0]
        assert col in {0, grid.shape[1] - 1}, "vertical palette must lie on an edge column"
        ordered = sorted((int(r), int(grid[r, col])) for r in rows)
        row_ids = [r for r, _color in ordered]
        assert row_ids == list(range(row_ids[0], row_ids[0] + len(row_ids))), "palette must be contiguous"
        palette = [color for _r, color in ordered]
        cells = {(r, col) for r in row_ids}

    assert len(palette) == len(set(palette)), "palette colors must be distinct"
    return palette, cells


def _trace_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    border, inner, (dr, dc) = _start_bar(grid)
    path = [border, inner]
    turn_indices: list[int] = []
    r, c = inner
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "corridor path should not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            break

        if int(grid[nr, nc]) != BLOCKER_COLOR:
            path.append((int(nr), int(nc)))
            r, c = int(nr), int(nc)
            continue

        choices: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) != BLOCKER_COLOR:
                choices.append((int(tdr), int(tdc)))

        assert len(choices) == 1, "each blocker collision must force a unique 90-degree turn"
        turn_indices.append(len(path) - 1)
        dr, dc = choices[0]

    return path, turn_indices


def _split_segments(path: list[tuple[int, int]], turn_indices: list[int]) -> list[list[tuple[int, int]]]:
    segments: list[list[tuple[int, int]]] = []
    start = 0
    for idx in turn_indices:
        segments.append(path[start : idx + 1])
        start = idx + 1
    segments.append(path[start:])
    assert all(segment for segment in segments)
    return segments


def _paint_segments(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    turn_indices: list[int],
    palette: list[int],
    palette_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for seg_index, segment in enumerate(_split_segments(path, turn_indices)):
        color = int(palette[seg_index % len(palette)])
        for r, c in segment:
            if (r, c) not in palette_cells and int(out[r, c]) != BLOCKER_COLOR:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the corridor path from the edge bar and recolor each turn-delimited segment by cycling through the embedded palette line.

    Input structure:
    - The background is black.
    - A 2-cell green start bar has exactly one border cell and points into the grid.
    - Teal single-pixel blockers force a one-cell-wide turning corridor, and a disjoint edge palette row or column contains 3-5 distinct colors from {blue, red, yellow, orange, maroon}.

    Rule:
    - Keep all teal blocker pixels and all palette pixels unchanged.
    - Start from the border pixel of the green bar, move through the other green pixel, and keep stepping through non-blocker cells.
    - Whenever the next cell is teal, turn to the unique perpendicular non-blocker neighbor; when the next step would leave the grid, stop.
    - Split the traced path at turn cells, then recolor segment k with palette color k modulo the palette length, reading the palette left-to-right or top-to-bottom.

    Key insight:
    - The blocker pattern determines the ordered turn events, and those turn boundaries are exactly what drives the cyclic palette coloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    palette, palette_cells = _extract_palette(g)
    path, turn_indices = _trace_path(g)
    return _paint_segments(g, path, turn_indices, palette, palette_cells)


# --------------- generator helpers ---------------

MIN_TURNS = 3
MAX_TURNS = 6
GRID_MIN = 13
GRID_MAX = 19
MAX_SEGMENT_LEN = 6
MIN_FORWARD_FOR_TURN = 2
MIN_FORWARD_AFTER_TURN = 2
MIN_FINAL_SEGMENT = 2
MIN_HIDDEN_PATH = 10
MIN_EXTRA_BLOCKERS = 1
MAX_EXTRA_BLOCKERS = 6
GEN_RETRIES = 400


def _max_forward(r: int, c: int, dr: int, dc: int, h: int, w: int) -> int:
    steps = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return steps
        steps += 1


def _choose_entry_bar(h: int, w: int) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]:
    side = random.choice(ENTRY_SIDES)
    if side == "top":
        col = random.randint(0, w - 1)
        return side, (0, col), (1, col), (1, 0)
    if side == "bottom":
        col = random.randint(0, w - 1)
        return side, (h - 1, col), (h - 2, col), (-1, 0)
    if side == "left":
        row = random.randint(0, h - 1)
        return side, (row, 0), (row, 1), (0, 1)
    row = random.randint(0, h - 1)
    return side, (row, w - 1), (row, w - 2), (0, -1)


def _walk_segment(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    r: int,
    c: int,
    dr: int,
    dc: int,
    steps: int | None,
) -> tuple[int, int] | None:
    walked = 0
    while True:
        if steps is not None and walked == steps:
            return r, c
        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            return (r, c) if steps is None else None
        if int(grid[nr, nc]) != BG_COLOR:
            return None
        grid[nr, nc] = START_COLOR
        path.append((int(nr), int(nc)))
        r, c = int(nr), int(nc)
        walked += 1


def _place_turn_blockers(
    grid: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    turns_left: int,
) -> tuple[int, int] | None:
    ahead_r, ahead_c = r + dr, c + dc
    if not in_bounds(grid.shape, ahead_r, ahead_c) or int(grid[ahead_r, ahead_c]) != BG_COLOR:
        return None

    perps = [(-dc, dr), (dc, -dr)]
    random.shuffle(perps)
    h, w = grid.shape

    for index, (tdr, tdc) in enumerate(perps):
        tr, tc = r + tdr, c + tdc
        if not in_bounds(grid.shape, tr, tc) or int(grid[tr, tc]) != BG_COLOR:
            continue

        required = MIN_FORWARD_AFTER_TURN if turns_left > 0 else MIN_FINAL_SEGMENT
        if _max_forward(r, c, tdr, tdc, h, w) < required:
            continue

        odr, odc = perps[1 - index]
        orow, ocol = r + odr, c + odc
        if in_bounds(grid.shape, orow, ocol):
            other_value = int(grid[orow, ocol])
            if other_value == BG_COLOR:
                grid[orow, ocol] = BLOCKER_COLOR
            elif other_value != BLOCKER_COLOR:
                continue

        grid[ahead_r, ahead_c] = BLOCKER_COLOR
        return int(tdr), int(tdc)

    return None


def _edge_run(side: str, start: int, length: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(start, start + length)]
    if side == "bottom":
        return [(h - 1, c) for c in range(start, start + length)]
    if side == "left":
        return [(r, 0) for r in range(start, start + length)]
    return [(r, w - 1) for r in range(start, start + length)]


def _place_palette(grid: np.ndarray, entry_side: str) -> tuple[list[int], set[tuple[int, int]]] | None:
    h, w = grid.shape
    lengths = [3, 4, 5]
    sides = [side for side in ENTRY_SIDES if side != entry_side]
    random.shuffle(sides)

    for side in sides:
        random.shuffle(lengths)
        extent = w if side in {"top", "bottom"} else h
        for length in lengths:
            if length > extent:
                continue
            starts = list(range(0, extent - length + 1))
            random.shuffle(starts)
            for start in starts:
                cells = _edge_run(side, start, length, h, w)
                if any(int(grid[r, c]) != BG_COLOR for r, c in cells):
                    continue
                palette = random.sample(PALETTE_POOL, k=length)
                for (r, c), color in zip(cells, palette):
                    grid[r, c] = color
                return palette, set(cells)

    return None


def _scatter_extra_blockers(
    grid: np.ndarray,
    protected: set[tuple[int, int]],
) -> int:
    candidates = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG_COLOR or (r, c) in protected:
                continue
            if any((r + dr, c + dc) in protected for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    count = min(random.randint(MIN_EXTRA_BLOCKERS, MAX_EXTRA_BLOCKERS), len(candidates))
    for r, c in candidates[:count]:
        grid[r, c] = BLOCKER_COLOR
    return count


def _expected_output(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    turn_indices: list[int],
    palette: list[int],
    palette_cells: set[tuple[int, int]],
) -> np.ndarray:
    return _paint_segments(grid, path, turn_indices, palette, palette_cells)


def _gen_attempt() -> np.ndarray | None:
    turns = random.choice([3, 3, 4, 4, 5, 6])
    min_dim = max(GRID_MIN, 8 + turns)
    h = random.randint(min_dim, GRID_MAX)
    w = random.randint(min_dim, GRID_MAX)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    entry_side, border, inner, (dr, dc) = _choose_entry_bar(h, w)
    grid[border] = START_COLOR
    grid[inner] = START_COLOR

    path = [border, inner]
    r, c = inner
    turn_indices: list[int] = []

    for turn_index in range(turns):
        max_fwd = _max_forward(r, c, dr, dc, h, w)
        if max_fwd < MIN_FORWARD_FOR_TURN:
            return None

        max_seg_len = min(MAX_SEGMENT_LEN, max_fwd - 1)
        if max_seg_len < 1:
            return None

        seg_len = random.choices(
            list(range(1, max_seg_len + 1)),
            weights=list(range(1, max_seg_len + 1)),
        )[0]
        end = _walk_segment(grid, path, r, c, dr, dc, seg_len)
        if end is None:
            return None
        r, c = end
        turn_indices.append(len(path) - 1)

        chosen = _place_turn_blockers(grid, r, c, dr, dc, turns - turn_index - 1)
        if chosen is None:
            return None
        dr, dc = chosen

    before_final = len(path)
    end = _walk_segment(grid, path, r, c, dr, dc, None)
    if end is None:
        return None
    if len(path) - before_final < MIN_FINAL_SEGMENT:
        return None

    hidden_path = path[2:]
    if len(hidden_path) < MIN_HIDDEN_PATH:
        return None

    palette_info = _place_palette(grid, entry_side)
    if palette_info is None:
        return None
    palette, palette_cells = palette_info

    protected = set(path) | palette_cells
    _scatter_extra_blockers(grid, protected)

    inp = grid.copy()
    for r, c in hidden_path:
        inp[r, c] = BG_COLOR

    expected = _expected_output(inp, path, turn_indices, palette, palette_cells)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
