from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
AZURE = 8
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
VALVE_HORIZONTAL = "horizontal"
VALVE_VERTICAL = "vertical"
PALETTE = [1, 2, 3, 4, 6, 7, 9]

# Generator configuration.
GRID_H_RANGE = (14, 20)
GRID_W_RANGE = (16, 24)
H_LINE_COUNT_RANGE = (2, 3)
V_LINE_COUNT_RANGE = (2, 3)
ROW_GAP = 3
COL_GAP = 3
TOP_BODY_ROW = 3
LEFT_BODY_COL = 1
GEN_RETRIES = 400


def _nonzero_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    c = 0
    while c < mask.size:
        if not mask[c]:
            c += 1
            continue
        start = c
        while c < mask.size and mask[c]:
            c += 1
        spans.append((start, c))
    return spans


def _decode_legend(g: np.ndarray) -> dict[int, str]:
    assert g.ndim == 2 and g.shape[0] >= 2
    top = g[:2, :]
    occupied = np.any(top != BG, axis=0)
    spans = _nonzero_spans(occupied)
    mapping: dict[int, str] = {}
    for c0, c1 in spans:
        patch = top[:, c0:c1]
        colors = {int(v) for v in patch.flat if int(v) not in (BG, AZURE)}
        assert len(colors) == 1, "each legend entry must expose exactly one valve color"
        valve_color = colors.pop()
        azure_counts = np.sum(patch == AZURE, axis=1).tolist()
        assert sorted(azure_counts) == [1, 2], "legend L marker must occupy 3 azure pixels"
        orientation = VALVE_HORIZONTAL if azure_counts[1] > azure_counts[0] else VALVE_VERTICAL
        assert valve_color not in mapping, "legend valve colors must be unique"
        mapping[valve_color] = orientation
    assert mapping, "expected at least one legend entry"
    return mapping


def _passable(g: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and r >= 2 and int(g[r, c]) not in (BG, AZURE)


def _find_valves(g: np.ndarray, valve_map: dict[int, str]) -> dict[tuple[int, int], str]:
    valves: dict[tuple[int, int], str] = {}
    for r in range(2, g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color not in valve_map:
                continue
            if all(_passable(g, r + dr, c + dc) for dr, dc in DIRS):
                valves[(r, c)] = valve_map[color]
    return valves


def _source_cells(g: np.ndarray, valves: dict[tuple[int, int], str]) -> list[tuple[int, int, int]]:
    sources: list[tuple[int, int, int]] = []
    for r in range(2, g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BG, GREY, AZURE):
                continue
            if (r, c) in valves:
                continue
            sources.append((r, c, color))
    return sources


def _allowed_dirs(r: int, c: int, valves: dict[tuple[int, int], str]) -> set[tuple[int, int]]:
    orientation = valves.get((r, c))
    if orientation == VALVE_HORIZONTAL:
        return {(0, -1), (0, 1)}
    if orientation == VALVE_VERTICAL:
        return {(-1, 0), (1, 0)}
    return set(DIRS)


def _graph_neighbors(g: np.ndarray, r: int, c: int, valves: dict[tuple[int, int], str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in _allowed_dirs(r, c, valves):
        nr, nc = r + dr, c + dc
        if not _passable(g, nr, nc):
            continue
        if (-dr, -dc) not in _allowed_dirs(nr, nc, valves):
            continue
        out.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode valve directions from the 2-row legend, then propagate source colors through the grey pipe network.

    Pipes are 4-neighbor orthogonal paths. Colored intersection centers are valves:
      - horizontal valve: flow can continue only left/right through that center
      - vertical valve: flow can continue only up/down through that center

    Each source independently reaches all traversable pipe cells. Grey cells reached by exactly one source
    are recolored to that source color. Grey cells reached by multiple sources remain grey. Valve cells keep
    their original color, and the legend strip is cleared to black in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    valve_map = _decode_legend(g)
    valves = _find_valves(g, valve_map)
    sources = _source_cells(g, valves)
    assert sources, "expected at least one source"

    reach: dict[tuple[int, int], set[int]] = {}
    for sr, sc, source_color in sources:
        seen = {(sr, sc)}
        queue = deque([(sr, sc)])
        while queue:
            r, c = queue.popleft()
            for nr, nc in _graph_neighbors(g, r, c, valves):
                if (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                queue.append((nr, nc))
        for r, c in seen:
            if int(g[r, c]) == GREY:
                reach.setdefault((r, c), set()).add(source_color)

    for (r, c), colors in reach.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        else:
            out[r, c] = GREY

    out[:2, :] = BG
    return out


def _pick_spaced(count: int, lo: int, hi: int, gap: int) -> list[int] | None:
    choices = list(range(lo, hi + 1))
    for _ in range(200):
        vals = sorted(random.sample(choices, count))
        if all(b - a >= gap for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _entry_patch(color: int, orientation: str) -> np.ndarray:
    if orientation == VALVE_HORIZONTAL:
        return np.array([[AZURE, BG, color], [AZURE, AZURE, color]], dtype=int)
    return np.array([[AZURE, AZURE, color], [AZURE, BG, color]], dtype=int)


def _draw_legend(grid: np.ndarray, mapping: dict[int, str]) -> None:
    entries = list(mapping.items())
    random.shuffle(entries)
    total_w = len(entries) * 3 + (len(entries) - 1)
    assert total_w <= grid.shape[1]
    start = random.randint(0, grid.shape[1] - total_w)
    c = start
    for color, orientation in entries:
        patch = _entry_patch(color, orientation)
        grid[:2, c : c + 3] = patch
        c += 4


def _assign_colors(coords: list[tuple[int, int]], pool: list[int]) -> dict[tuple[int, int], int]:
    assert coords and pool
    shuffled = coords[:]
    random.shuffle(shuffled)
    out: dict[tuple[int, int], int] = {}
    for coord, color in zip(shuffled, pool):
        out[coord] = color
    for coord in shuffled[len(pool) :]:
        out[coord] = random.choice(pool)
    return out


def _propagation_signature(inp: np.ndarray, out: np.ndarray, source_colors: list[int]) -> bool:
    if not np.all(out[:2, :] == BG):
        return False
    growth = [int(np.count_nonzero(out == color) - np.count_nonzero(inp == color)) for color in source_colors]
    if max(growth, default=0) < 1:
        return False
    if sum(max(0, delta) for delta in growth) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_h = random.randint(*H_LINE_COUNT_RANGE)
    n_v = random.randint(*V_LINE_COUNT_RANGE)

    rows = _pick_spaced(n_h, TOP_BODY_ROW + 1, h - 3, ROW_GAP)
    cols = _pick_spaced(n_v, 3, w - 4, COL_GAP)
    if rows is None or cols is None:
        return None

    conflict_row = random.choice(rows)
    unique_row_candidates = [r for r in rows if r != conflict_row]
    if not unique_row_candidates:
        return None
    unique_row = random.choice(unique_row_candidates)
    unique_side = random.choice(["left", "right"])
    unique_barrier_col = min(cols) if unique_side == "left" else max(cols)

    use_vertical_source = random.random() < 0.6
    vertical_source_side = random.choice(["top", "bottom"])
    vertical_col = random.choice(cols) if use_vertical_source else None
    vertical_barrier_row = min(rows) if vertical_source_side == "top" else max(rows)
    if use_vertical_source and vertical_col == unique_barrier_col and vertical_barrier_row == unique_row:
        alternatives = [c for c in cols if c != unique_barrier_col]
        if alternatives:
            vertical_col = random.choice(alternatives)
        else:
            use_vertical_source = False
            vertical_col = None

    endpoints: list[tuple[int, int]] = [
        (conflict_row, LEFT_BODY_COL),
        (conflict_row, w - 2),
        (unique_row, LEFT_BODY_COL if unique_side == "left" else w - 2),
    ]
    if use_vertical_source and vertical_col is not None:
        endpoints.append((TOP_BODY_ROW, vertical_col) if vertical_source_side == "top" else (h - 2, vertical_col))
    if len(set(endpoints)) != len(endpoints):
        return None

    if len(PALETTE) < len(endpoints) + 2:
        return None
    source_colors = random.sample(PALETTE, len(endpoints))
    remaining_colors = [c for c in PALETTE if c not in source_colors]

    orientations: dict[tuple[int, int], str] = {}
    for r in rows:
        for c in cols:
            orientations[(r, c)] = random.choice([VALVE_HORIZONTAL, VALVE_VERTICAL])
    for c in cols:
        orientations[(conflict_row, c)] = VALVE_HORIZONTAL
    orientations[(unique_row, unique_barrier_col)] = VALVE_VERTICAL
    if use_vertical_source and vertical_col is not None:
        orientations[(vertical_barrier_row, vertical_col)] = VALVE_HORIZONTAL

    h_coords = [coord for coord, orient in orientations.items() if orient == VALVE_HORIZONTAL]
    v_coords = [coord for coord, orient in orientations.items() if orient == VALVE_VERTICAL]
    if not h_coords or not v_coords:
        return None

    max_h_colors = min(2, len(h_coords), len(remaining_colors) - 1)
    if max_h_colors < 1:
        return None
    h_color_count = random.randint(1, max_h_colors)
    h_colors = random.sample(remaining_colors, h_color_count)
    remaining_colors = [c for c in remaining_colors if c not in h_colors]
    max_v_colors = min(2, len(v_coords), len(remaining_colors))
    if max_v_colors < 1:
        return None
    v_color_count = random.randint(1, max_v_colors)
    v_colors = random.sample(remaining_colors, v_color_count)

    valve_color_map = {color: VALVE_HORIZONTAL for color in h_colors}
    valve_color_map.update({color: VALVE_VERTICAL for color in v_colors})

    if len(valve_color_map) * 3 + (len(valve_color_map) - 1) > w:
        return None

    intersection_colors = _assign_colors(h_coords, h_colors)
    intersection_colors.update(_assign_colors(v_coords, v_colors))

    grid = np.full((h, w), BG, dtype=int)
    left, right = LEFT_BODY_COL, w - 2
    top, bottom = TOP_BODY_ROW, h - 2
    for r in rows:
        grid[r, left : right + 1] = GREY
    for c in cols:
        grid[top : bottom + 1, c] = GREY

    for (r, c), color in intersection_colors.items():
        grid[r, c] = color

    for (r, c), color in zip(endpoints, source_colors, strict=False):
        grid[r, c] = color

    _draw_legend(grid, valve_color_map)

    try:
        out = main(grid)
    except AssertionError:
        return None

    conflict_segment = out[conflict_row, left + 1 : right]
    if int(np.count_nonzero(conflict_segment == GREY)) < max(2, len(cols)):
        return None
    if not _propagation_signature(grid, out, source_colors):
        return None
    if len({int(v) for v in grid.flat}) < 5:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
