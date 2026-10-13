from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRECTIONS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
PALETTE = tuple(range(10))
GENERATION_RETRIES = 400


def _runs_between_stops(length: int, stops: list[int]) -> list[tuple[int, ...]]:
    stop_set = set(stops)
    runs: list[tuple[int, ...]] = []
    start = None
    for idx in range(length):
        if idx in stop_set:
            if start is not None:
                runs.append(tuple(range(start, idx)))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append(tuple(range(start, length)))
    return runs


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    separator_candidates = []
    for color in map(int, np.unique(g)):
        sep_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue
        sep_mask = np.zeros_like(g, dtype=bool)
        sep_mask[sep_rows, :] = True
        sep_mask[:, sep_cols] = True
        if np.array_equal(g == color, sep_mask):
            separator_candidates.append((color, sep_rows, sep_cols, sep_mask))
    assert len(separator_candidates) == 1, "expected a single separator color"

    separator_color, sep_rows, sep_cols, sep_mask = separator_candidates[0]
    row_tiles = _runs_between_stops(height, sep_rows)
    col_tiles = _runs_between_stops(width, sep_cols)
    assert len(row_tiles) >= 2 and len(col_tiles) >= 2, "expected a tile lattice"
    tile_hs = {len(rows) for rows in row_tiles}
    tile_ws = {len(cols) for cols in col_tiles}
    assert len(tile_hs) == 1 and len(tile_ws) == 1, "tiles must be equal-sized"

    tile_h = next(iter(tile_hs))
    tile_w = next(iter(tile_ws))
    assert tile_h >= 3 and tile_w >= 3 and tile_h % 2 == 1 and tile_w % 2 == 1

    content = g[~sep_mask]
    counts = Counter(map(int, content))
    base_color, _ = counts.most_common(1)[0]
    assert base_color != separator_color, "base color must differ from separators"

    row_lookup = {
        r: (tile_r, local_r)
        for tile_r, rows in enumerate(row_tiles)
        for local_r, r in enumerate(rows)
    }
    col_lookup = {
        c: (tile_c, local_c)
        for tile_c, cols in enumerate(col_tiles)
        for local_c, c in enumerate(cols)
    }

    center_r = tile_h // 2
    center_c = tile_w // 2
    pointers: dict[tuple[int, int], dict] = {}
    for r, c in np.argwhere((g != separator_color) & (g != base_color)):
        tile_r, local_r = row_lookup[int(r)]
        tile_c, local_c = col_lookup[int(c)]
        dr = int(local_r - center_r)
        dc = int(local_c - center_c)
        assert (dr, dc) in DIRECTIONS8, "markers must sit on a center-adjacent offset"
        tile = (tile_r, tile_c)
        assert tile not in pointers, "each tile may contain at most one marker"
        pointers[tile] = {"direction": (dr, dc), "color": int(g[r, c]), "pos": (int(r), int(c))}

    assert pointers, "expected at least one marker"
    color_counts = Counter(pointer["color"] for pointer in pointers.values())
    unique_colors = [color for color, n in color_counts.items() if n == 1]
    assert len(unique_colors) == 1, "expected exactly one globally unique marker color"
    start_color = unique_colors[0]
    start_tiles = [tile for tile, pointer in pointers.items() if pointer["color"] == start_color]
    assert len(start_tiles) == 1

    tile_base_masks: dict[tuple[int, int], np.ndarray] = {}
    for tile_r, rows in enumerate(row_tiles):
        for tile_c, cols in enumerate(col_tiles):
            region = g[np.ix_(rows, cols)]
            tile_base_masks[(tile_r, tile_c)] = region == base_color

    return {
        "grid": g,
        "separator_color": separator_color,
        "base_color": base_color,
        "row_tiles": row_tiles,
        "col_tiles": col_tiles,
        "tile_shape": (tile_h, tile_w),
        "pointers": pointers,
        "start_tile": start_tiles[0],
        "tile_base_masks": tile_base_masks,
    }


def _paint_tile(
    out: np.ndarray,
    row_tiles: list[tuple[int, ...]],
    col_tiles: list[tuple[int, ...]],
    tile: tuple[int, int],
    base_mask: np.ndarray,
    color: int,
) -> None:
    rows = row_tiles[tile[0]]
    cols = col_tiles[tile[1]]
    region = out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    region[base_mask] = int(color)


def _follow_chain(parsed: dict) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    g = parsed["grid"]
    row_tiles = parsed["row_tiles"]
    col_tiles = parsed["col_tiles"]
    pointers = parsed["pointers"]
    start_tile = parsed["start_tile"]

    out = g.copy()
    recolored_tiles: set[tuple[int, int]] = set()
    visited_order: list[tuple[int, int]] = []
    incoming_colors: list[int] = []

    current_tile = start_tile
    incoming_color = int(pointers[current_tile]["color"])
    direction = tuple(map(int, pointers[current_tile]["direction"]))
    n_tile_rows = len(row_tiles)
    n_tile_cols = len(col_tiles)

    while True:
        next_tile = (current_tile[0] + direction[0], current_tile[1] + direction[1])
        if not (0 <= next_tile[0] < n_tile_rows and 0 <= next_tile[1] < n_tile_cols):
            break
        if next_tile in recolored_tiles:
            break

        _paint_tile(
            out,
            row_tiles,
            col_tiles,
            next_tile,
            parsed["tile_base_masks"][next_tile],
            incoming_color,
        )
        recolored_tiles.add(next_tile)
        visited_order.append(next_tile)
        incoming_colors.append(incoming_color)

        current_tile = next_tile
        if current_tile not in pointers:
            break
        incoming_color = int(pointers[current_tile]["color"])
        direction = tuple(map(int, pointers[current_tile]["direction"]))

    return out, visited_order, incoming_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the unique start pointer through a tiled lattice and recolor each entered tile with the previous pointer's color.

    Input structure:
    - Full separator rows and columns divide the canvas into equal odd-sized tiles, and every tile starts with the same base color.
    - Some tiles contain exactly one marker pixel in one of the eight center-adjacent positions; that local offset encodes an 8-direction pointer.
    - Exactly one marker color appears only once in the whole grid, and that tile is the start of the chain.

    Rule:
    - Begin at the start tile and move to the adjacent tile indicated by its pointer; after that, whenever the entered tile has a marker, continue along that tile's pointer.
    - Each time a tile is entered, recolor all of that tile's base-color cells with the color of the pointer from the previous tile, while keeping separator cells and marker pixels unchanged.
    - Stop when the current pointer would leave the lattice, when it would enter a tile that has already been recolored, or after entering a tile that has no marker.

    Key insight:
    - The marker pixel's offset from the tile center is the real control signal: the grid is a directed graph on tiles, and the propagated color always comes from the previous stop rather than the current tile.
    """
    parsed = _parse_grid(grid)
    out, _visited_order, _incoming_colors = _follow_chain(parsed)
    return out


def _all_tiles(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


def _in_bounds_tile(n_rows: int, n_cols: int, tile: tuple[int, int]) -> bool:
    return 0 <= tile[0] < n_rows and 0 <= tile[1] < n_cols


def _neighbor_tiles(n_rows: int, n_cols: int, tile: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    out = []
    for dr, dc in DIRECTIONS8:
        nxt = (tile[0] + dr, tile[1] + dc)
        if _in_bounds_tile(n_rows, n_cols, nxt):
            out.append((nxt, (dr, dc)))
    return out


def _outward_dirs(n_rows: int, n_cols: int, tile: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRECTIONS8:
        nxt = (tile[0] + dr, tile[1] + dc)
        if not _in_bounds_tile(n_rows, n_cols, nxt):
            out.append((dr, dc))
    return out


def _random_path(n_rows: int, n_cols: int, path_len: int, stop_mode: str) -> list[tuple[int, int]] | None:
    for _ in range(150):
        path = [random.choice(_all_tiles(n_rows, n_cols))]
        while len(path) < path_len:
            candidates = [tile for tile, _dir in _neighbor_tiles(n_rows, n_cols, path[-1]) if tile not in path]
            random.shuffle(candidates)
            if not candidates:
                break
            path.append(candidates[0])
        if len(path) != path_len:
            continue
        if stop_mode == "out" and not _outward_dirs(n_rows, n_cols, path[-1]):
            continue
        return path
    return None


def _sample_dimensions() -> tuple[int, int, int, int]:
    for _ in range(100):
        n_tile_rows = random.randint(3, 5)
        n_tile_cols = random.randint(3, 5)
        tile_h = random.choice((3, 5))
        tile_w = random.choice((3, 5))
        height = n_tile_rows * tile_h + (n_tile_rows - 1)
        width = n_tile_cols * tile_w + (n_tile_cols - 1)
        if height <= 30 and width <= 30:
            return n_tile_rows, n_tile_cols, tile_h, tile_w
    raise RuntimeError("failed to sample ARC-sized dimensions")


def _render_case(case: dict) -> np.ndarray:
    n_tile_rows = case["n_tile_rows"]
    n_tile_cols = case["n_tile_cols"]
    tile_h = case["tile_h"]
    tile_w = case["tile_w"]
    height = n_tile_rows * tile_h + (n_tile_rows - 1)
    width = n_tile_cols * tile_w + (n_tile_cols - 1)

    grid = np.full((height, width), case["base_color"], dtype=int)
    sep_rows = [tile_h + idx * (tile_h + 1) for idx in range(n_tile_rows - 1)]
    sep_cols = [tile_w + idx * (tile_w + 1) for idx in range(n_tile_cols - 1)]
    if sep_rows:
        grid[sep_rows, :] = case["separator_color"]
    if sep_cols:
        grid[:, sep_cols] = case["separator_color"]

    center_r = tile_h // 2
    center_c = tile_w // 2
    for (tile_r, tile_c), pointer in case["markers"].items():
        raw_r = tile_r * (tile_h + 1) + center_r + pointer["direction"][0]
        raw_c = tile_c * (tile_w + 1) + center_c + pointer["direction"][1]
        grid[raw_r, raw_c] = pointer["color"]
    return grid


def _simulate_case(case: dict) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    parsed = _parse_grid(_render_case(case))
    return _follow_chain(parsed)


def _sample_palette() -> tuple[int, int, int, list[int]]:
    colors = random.sample(PALETTE, k=random.randint(5, 7))
    separator_color = colors[0]
    base_color = colors[1]
    start_color = colors[2]
    repeat_colors = colors[3:]
    random.shuffle(repeat_colors)
    keep = random.randint(2, min(3, len(repeat_colors)))
    repeat_colors = repeat_colors[:keep]
    return separator_color, base_color, start_color, repeat_colors


def _assign_chain_directions(path: list[tuple[int, int]], stop_mode: str, n_rows: int, n_cols: int) -> dict[tuple[int, int], tuple[int, int]]:
    directions = {}
    for idx in range(len(path) - 1):
        curr = path[idx]
        nxt = path[idx + 1]
        directions[curr] = (nxt[0] - curr[0], nxt[1] - curr[1])

    final_tile = path[-1]
    if stop_mode == "out":
        directions[final_tile] = random.choice(_outward_dirs(n_rows, n_cols, final_tile))
    elif stop_mode == "cycle":
        cycle_targets = [
            prev for prev in path[1:-1]
            if max(abs(prev[0] - final_tile[0]), abs(prev[1] - final_tile[1])) == 1
        ]
        if not cycle_targets:
            cycle_targets = [path[-2]]
        target = random.choice(cycle_targets)
        directions[final_tile] = (target[0] - final_tile[0], target[1] - final_tile[1])
    return directions


def _choose_extra_tiles(
    n_rows: int,
    n_cols: int,
    forbidden: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    choices = [tile for tile in _all_tiles(n_rows, n_cols) if tile not in forbidden]
    if len(choices) < count:
        return None
    random.shuffle(choices)
    return choices[:count]


def _build_case() -> dict | None:
    n_tile_rows, n_tile_cols, tile_h, tile_w = _sample_dimensions()
    separator_color, base_color, start_color, repeat_colors = _sample_palette()
    stop_mode = random.choice(("no_pointer", "out", "cycle"))

    max_path_len = min(7, n_tile_rows * n_tile_cols)
    if max_path_len < 3:
        return None
    path_len = random.randint(3, max_path_len)
    path = _random_path(n_tile_rows, n_tile_cols, path_len, stop_mode)
    if path is None:
        return None

    chain_directions = _assign_chain_directions(path, stop_mode, n_tile_rows, n_tile_cols)
    markers: dict[tuple[int, int], dict] = {
        path[0]: {"direction": chain_directions[path[0]], "color": start_color}
    }

    chain_marker_tiles = set(path[:-1] if stop_mode == "no_pointer" else path)
    if len(chain_marker_tiles) < 2:
        return None

    non_start_chain_tiles = [tile for tile in path[1:] if tile in chain_marker_tiles]
    if not non_start_chain_tiles:
        return None
    for tile in non_start_chain_tiles:
        markers[tile] = {
            "direction": chain_directions[tile],
            "color": random.choice(repeat_colors),
        }

    counts = Counter(pointer["color"] for pointer in markers.values())
    extras_by_color: list[int] = []
    for color in repeat_colors:
        deficit = max(0, 2 - counts[color])
        extras_by_color.extend([color] * deficit)
    if not extras_by_color:
        extras_by_color.append(random.choice(repeat_colors))

    extra_total = len(extras_by_color) + random.randint(0, 2)
    chosen_extra_tiles = _choose_extra_tiles(
        n_tile_rows,
        n_tile_cols,
        forbidden=set(path),
        count=extra_total,
    )
    if chosen_extra_tiles is None:
        return None

    extra_colors = extras_by_color[:]
    while len(extra_colors) < extra_total:
        extra_colors.append(random.choice(repeat_colors))
    random.shuffle(extra_colors)

    for tile, color in zip(chosen_extra_tiles, extra_colors):
        markers[tile] = {
            "direction": random.choice(DIRECTIONS8),
            "color": color,
        }

    return {
        "n_tile_rows": n_tile_rows,
        "n_tile_cols": n_tile_cols,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "separator_color": separator_color,
        "base_color": base_color,
        "start_tile": path[0],
        "path": path,
        "stop_mode": stop_mode,
        "markers": markers,
        "chain_marker_tiles": chain_marker_tiles,
        "extra_tiles": set(chosen_extra_tiles),
    }


def _validate_case(case: dict) -> bool:
    grid = _render_case(case)
    try:
        parsed = _parse_grid(grid)
    except AssertionError:
        return False

    expected, visited_order, incoming_colors = _simulate_case(case)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    pointers = parsed["pointers"]
    if set(pointers) != set(case["markers"]):
        return False
    for tile, pointer in case["markers"].items():
        if pointers[tile]["direction"] != tuple(pointer["direction"]):
            return False
        if pointers[tile]["color"] != int(pointer["color"]):
            return False

    if case["start_tile"] != parsed["start_tile"]:
        return False

    color_counts = Counter(pointer["color"] for pointer in case["markers"].values())
    if sum(1 for n in color_counts.values() if n == 1) != 1:
        return False
    if color_counts[case["markers"][case["start_tile"]]["color"]] != 1:
        return False
    if any(n < 2 for color, n in color_counts.items() if color != case["markers"][case["start_tile"]]["color"]):
        return False

    if len(case["extra_tiles"]) < 1:
        return False
    if len(visited_order) < 2:
        return False
    if len(set(incoming_colors)) < 2:
        return False
    if np.array_equal(grid, expected):
        return False

    visited_with_markers = [tile for tile in visited_order if tile in case["markers"]]
    if not visited_with_markers:
        return False
    if not any(
        idx > 0 and tile in case["markers"] and case["markers"][tile]["color"] != incoming_colors[idx]
        for idx, tile in enumerate(visited_order)
    ):
        return False

    direction_sets: dict[int, set[tuple[int, int]]] = {}
    for pointer in case["markers"].values():
        direction_sets.setdefault(pointer["color"], set()).add(tuple(pointer["direction"]))
    if not any(len(dirs) >= 2 for color, dirs in direction_sets.items() if color_counts[color] >= 2):
        return False

    if case["stop_mode"] == "no_pointer" and case["path"][-1] in case["markers"]:
        return False
    if case["stop_mode"] != "no_pointer" and case["path"][-1] not in case["markers"]:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        case = _build_case()
        if case is None:
            continue
        if _validate_case(case):
            return _render_case(case)
    raise RuntimeError("failed to generate a valid pointer-chain lattice input")
