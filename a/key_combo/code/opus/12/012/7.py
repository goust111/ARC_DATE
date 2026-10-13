from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


SEP = 5
BG = 0
CORNER_OFFSETS = ((0, 0), (0, 2), (2, 0), (2, 2))
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

TILE_ROWS_RANGE = (3, 5)
TILE_COL_CHOICES = (3, 5)
MIN_SEG_LEN = 3
MAX_GEN_ATTEMPTS = 500
CONFLICT_PROB = 0.45


def _runs(active: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _parse_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = int(g[0, 0])
    assert np.all(g[0, :] == sep) and np.all(g[-1, :] == sep), "expected separator border rows"
    assert np.all(g[:, 0] == sep) and np.all(g[:, -1] == sep), "expected separator border columns"

    row_runs = _runs([not np.all(g[r, :] == sep) for r in range(g.shape[0])])
    col_runs = _runs([not np.all(g[:, c] == sep) for c in range(g.shape[1])])
    assert row_runs and col_runs, "expected a non-empty tile lattice"
    assert all(end - start == 3 for start, end in row_runs), "expected 3x3 tiles"
    assert all(end - start == 3 for start, end in col_runs), "expected 3x3 tiles"
    return sep, row_runs, col_runs


def _tile_index(tile_r: int, tile_c: int, tile_cols: int) -> int:
    return int(tile_r) * int(tile_cols) + int(tile_c)


def _path_tiles(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(row, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        col = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, col) for r in range(r0, r1 + 1)]
    raise AssertionError("endpoints must share a tile row or tile column")


def _extract_groups(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int, int], list[tuple[int, int]]], Counter]:
    g = np.asarray(grid, dtype=int)
    sep, row_runs, col_runs = _parse_lattice(g)
    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    endpoint_counts: Counter = Counter()

    for tile_r, (r0, _r1) in enumerate(row_runs):
        for tile_c, (c0, _c1) in enumerate(col_runs):
            tile = g[r0 : r0 + 3, c0 : c0 + 3]
            for rr in range(3):
                for cc in range(3):
                    color = int(tile[rr, cc])
                    if color in (BG, sep):
                        continue
                    assert (rr, cc) in CORNER_OFFSETS, "expected markers only at tile-corner pixels"
                    groups[(color, rr, cc)].append((tile_r, tile_c))
                    endpoint_counts[color] += 1

    assert groups, "expected endpoint markers"
    return row_runs, col_runs, groups, endpoint_counts


def _apply_groups(
    base_grid: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    groups: dict[tuple[int, int, int], list[tuple[int, int]]],
    endpoint_counts: Counter,
) -> np.ndarray:
    tile_cols = len(col_runs)
    path_pixels: set[tuple[int, int]] = set()
    proposals: dict[tuple[int, int], list[int]] = defaultdict(list)

    for (color, dr, dc), points in groups.items():
        assert len(points) == 2, "each (color, corner) group must contain exactly two endpoints"
        start, end = sorted(points)
        path = _path_tiles(start, end)
        keep_parity = _tile_index(start[0], start[1], tile_cols) % 2

        for tile_r, tile_c in path:
            pr = row_runs[tile_r][0] + dr
            pc = col_runs[tile_c][0] + dc
            path_pixels.add((pr, pc))
            if _tile_index(tile_r, tile_c, tile_cols) % 2 == keep_parity:
                proposals[(pr, pc)].append(color)

    out = np.asarray(base_grid, dtype=int).copy()
    for r, c in path_pixels:
        out[r, c] = BG

    for rc, colors in proposals.items():
        unique = sorted(set(colors))
        chosen = max(unique, key=lambda col: (endpoint_counts[col], -col))
        out[rc] = chosen

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect matching tile-corner markers, then erase every other tile on each path by global tile parity.

    Input structure:
    - Grey separator rows and columns partition the grid into a regular lattice of 3x3 black tiles.
    - Some tiles contain colored endpoint markers at one of the four corner pixels of the 3x3 tile.
    - Markers are paired by shared color and shared corner position, and each pair lies on one tile row or one tile column.

    Rule:
    - For each pair, fill the same corner pixel through every tile on the straight tile-lattice path between the two endpoints.
    - Compute each tile's row-major tile index; the smaller-index endpoint sets the phase for that path.
    - Keep filled pixels only in tiles whose index parity matches that starting endpoint, erase the others back to black, and resolve kept overlaps by preferring the color with more total endpoints in the input.

    Key insight:
    - The visible dashed pattern is controlled by the global parity of tile positions, not by local path direction or path length.
    """
    g = np.asarray(grid, dtype=int)
    row_runs, col_runs, groups, endpoint_counts = _extract_groups(g)
    return _apply_groups(g, row_runs, col_runs, groups, endpoint_counts)


def _blank_lattice(tile_rows: int, tile_cols: int) -> np.ndarray:
    grid = np.full((tile_rows * 4 + 1, tile_cols * 4 + 1), SEP, dtype=int)
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0 = 1 + tile_r * 4
            c0 = 1 + tile_c * 4
            grid[r0 : r0 + 3, c0 : c0 + 3] = BG
    return grid


def _paint_endpoints(grid: np.ndarray, groups: list[dict]) -> np.ndarray:
    out = grid.copy()
    for group in groups:
        dr, dc = group["pos"]
        color = group["color"]
        for tile_r, tile_c in group["endpoints"]:
            r0 = 1 + tile_r * 4
            c0 = 1 + tile_c * 4
            out[r0 + dr, c0 + dc] = color
    return out


def _group_path(group: dict) -> list[tuple[int, int]]:
    return _path_tiles(group["endpoints"][0], group["endpoints"][1])


def _make_group(color: int, pos: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> dict:
    orientation = "H" if a[0] == b[0] else "V"
    return {"color": color, "pos": pos, "endpoints": (a, b), "orientation": orientation}


def _all_group_candidates(
    tile_rows: int,
    tile_cols: int,
    pos: tuple[int, int],
    color: int,
    orientation: str,
    desired_start_parity: int | None = None,
    cross_tile: tuple[int, int] | None = None,
) -> list[dict]:
    candidates: list[dict] = []
    if orientation == "H":
        rows = [cross_tile[0]] if cross_tile is not None else list(range(tile_rows))
        for tile_r in rows:
            for c0 in range(tile_cols):
                for c1 in range(c0 + MIN_SEG_LEN - 1, tile_cols):
                    if cross_tile is not None:
                        cross_c = cross_tile[1]
                        if not (c0 < cross_c < c1):
                            continue
                        if (cross_c - c0) % 2 != 0:
                            continue
                    start_parity = _tile_index(tile_r, c0, tile_cols) % 2
                    if desired_start_parity is not None and start_parity != desired_start_parity:
                        continue
                    candidates.append(_make_group(color, pos, (tile_r, c0), (tile_r, c1)))
    else:
        cols = [cross_tile[1]] if cross_tile is not None else list(range(tile_cols))
        for tile_c in cols:
            for r0 in range(tile_rows):
                for r1 in range(r0 + MIN_SEG_LEN - 1, tile_rows):
                    if cross_tile is not None:
                        cross_r = cross_tile[0]
                        if not (r0 < cross_r < r1):
                            continue
                        if (cross_r - r0) % 2 != 0:
                            continue
                    start_parity = _tile_index(r0, tile_c, tile_cols) % 2
                    if desired_start_parity is not None and start_parity != desired_start_parity:
                        continue
                    candidates.append(_make_group(color, pos, (r0, tile_c), (r1, tile_c)))
    random.shuffle(candidates)
    return candidates


def _group_ok(
    group: dict,
    groups: list[dict],
    used_endpoint_tiles: set[tuple[int, int]],
    allow_overlap_tiles: set[tuple[int, int]] | None = None,
) -> bool:
    allow_overlap_tiles = allow_overlap_tiles or set()
    pos = group["pos"]
    endpoints = set(group["endpoints"])
    path = set(_group_path(group))

    if endpoints & used_endpoint_tiles:
        return False

    for other in groups:
        if other["pos"] != pos:
            continue
        other_endpoints = set(other["endpoints"])
        other_path = set(_group_path(other))
        if path & other_endpoints:
            return False
        if endpoints & other_path:
            return False
        overlap = path & other_path
        if overlap - allow_overlap_tiles:
            return False
        if overlap & (endpoints | other_endpoints):
            return False

    return True


def _place_group(
    specs: list[dict],
    tile_rows: int,
    tile_cols: int,
    groups: list[dict],
    used_endpoint_tiles: set[tuple[int, int]],
) -> bool:
    spec = specs[0]
    candidates = _all_group_candidates(
        tile_rows,
        tile_cols,
        spec["pos"],
        spec["color"],
        spec["orientation"],
        desired_start_parity=spec.get("desired_start_parity"),
        cross_tile=spec.get("cross_tile"),
    )
    for group in candidates:
        if not _group_ok(group, groups, used_endpoint_tiles, spec.get("allow_overlap_tiles")):
            continue
        groups.append(group)
        used_endpoint_tiles.update(group["endpoints"])
        if len(specs) == 1 or _place_group(specs[1:], tile_rows, tile_cols, groups, used_endpoint_tiles):
            return True
        groups.pop()
        for endpoint in group["endpoints"]:
            used_endpoint_tiles.remove(endpoint)
    return False


def _planned_output(tile_rows: int, tile_cols: int, groups: list[dict]) -> np.ndarray:
    inp = _paint_endpoints(_blank_lattice(tile_rows, tile_cols), groups)
    _, row_runs, col_runs = _parse_lattice(inp)
    extracted: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    endpoint_counts: Counter = Counter()
    for group in groups:
        color = group["color"]
        dr, dc = group["pos"]
        extracted[(color, dr, dc)].extend(group["endpoints"])
        endpoint_counts[color] += 2
    return _apply_groups(inp, row_runs, col_runs, extracted, endpoint_counts)


def _proposal_map(tile_rows: int, tile_cols: int, groups: list[dict]) -> tuple[dict[tuple[int, int], list[int]], Counter]:
    proposals: dict[tuple[int, int], list[int]] = defaultdict(list)
    endpoint_counts = Counter(group["color"] for group in groups for _ in range(2))
    for group in groups:
        color = group["color"]
        start = min(group["endpoints"])
        keep_parity = _tile_index(start[0], start[1], tile_cols) % 2
        for tile in _group_path(group):
            if _tile_index(tile[0], tile[1], tile_cols) % 2 == keep_parity:
                proposals[(tile[0], tile[1], group["pos"][0], group["pos"][1])].append(color)
    return proposals, endpoint_counts


def _validate_generated_case(
    inp: np.ndarray,
    out: np.ndarray,
    tile_rows: int,
    tile_cols: int,
    groups: list[dict],
    expect_conflict: bool,
) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False
    if np.count_nonzero(inp != out) < 4:
        return False

    orientations = {group["orientation"] for group in groups}
    if orientations != {"H", "V"}:
        return False

    start_parities = {_tile_index(*min(group["endpoints"]), tile_cols) % 2 for group in groups}
    if start_parities != {0, 1}:
        return False

    if any(len(_group_path(group)) < MIN_SEG_LEN for group in groups):
        return False

    parsed_rows, parsed_cols, parsed_groups, _counts = _extract_groups(inp)
    if len(parsed_rows) != tile_rows or len(parsed_cols) != tile_cols:
        return False
    if any(len(points) != 2 for points in parsed_groups.values()):
        return False

    for (color, dr, dc), points in parsed_groups.items():
        path = set(_path_tiles(*points))
        for other_key, other_points in parsed_groups.items():
            if other_key == (color, dr, dc):
                continue
            if other_key[1:] != (dr, dc):
                continue
            if set(other_points) & path:
                return False

    proposals, endpoint_counts = _proposal_map(tile_rows, tile_cols, groups)
    has_conflict = False
    for colors in proposals.values():
        unique = set(colors)
        if len(unique) < 2:
            continue
        scores = {endpoint_counts[color] for color in unique}
        if len(scores) != len(unique):
            return False
        has_conflict = True

    if expect_conflict and not has_conflict:
        return False

    return np.array_equal(main(inp), out)


def _nonconflict_specs(tile_rows: int, tile_cols: int, colors: list[int]) -> list[dict]:
    positions = list(CORNER_OFFSETS)
    random.shuffle(positions)
    specs = [
        {
            "color": colors[0],
            "pos": positions[0],
            "orientation": "H",
            "desired_start_parity": 0,
        },
        {
            "color": colors[1],
            "pos": positions[1],
            "orientation": "V",
            "desired_start_parity": 1,
        },
    ]

    extra_count = random.choice((0, 1, 1, 2))
    for _ in range(extra_count):
        choices: list[tuple[int, tuple[int, int]]] = []
        for color in colors:
            used_for_color = {spec["pos"] for spec in specs if spec["color"] == color}
            for pos in CORNER_OFFSETS:
                if pos not in used_for_color:
                    choices.append((color, pos))
        if not choices:
            continue
        color, pos = random.choice(choices)
        specs.append(
            {
                "color": color,
                "pos": pos,
                "orientation": random.choice(("H", "V")),
                "desired_start_parity": random.choice((0, 1)),
            }
        )

    return specs


def _conflict_specs(tile_rows: int, tile_cols: int, colors: list[int]) -> list[dict] | None:
    if tile_rows < 3 or tile_cols < 3:
        return None
    dominant = colors[0]
    minor = colors[1]
    other = colors[2] if len(colors) > 2 else None

    cross_tile = (random.randint(1, tile_rows - 2), random.randint(1, tile_cols - 2))
    shared_pos = random.choice(CORNER_OFFSETS)
    remaining_positions = [pos for pos in CORNER_OFFSETS if pos != shared_pos]
    if not remaining_positions:
        return None

    specs = [
        {
            "color": minor,
            "pos": shared_pos,
            "orientation": "H",
            "desired_start_parity": 0,
            "cross_tile": cross_tile,
        },
        {
            "color": dominant,
            "pos": shared_pos,
            "orientation": "V",
            "desired_start_parity": 1,
            "cross_tile": cross_tile,
            "allow_overlap_tiles": {cross_tile},
        },
        {
            "color": dominant,
            "pos": remaining_positions[0],
            "orientation": random.choice(("H", "V")),
            "desired_start_parity": random.choice((0, 1)),
        },
    ]

    if other is not None and random.random() < 0.6:
        used_positions = {spec["pos"] for spec in specs if spec["color"] == other}
        free_positions = [pos for pos in CORNER_OFFSETS if pos not in used_positions]
        if free_positions:
            specs.append(
                {
                    "color": other,
                    "pos": random.choice(free_positions),
                    "orientation": random.choice(("H", "V")),
                    "desired_start_parity": random.choice((0, 1)),
                }
            )

    return specs


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows = random.randint(*TILE_ROWS_RANGE)
    tile_cols = random.choice(TILE_COL_CHOICES)

    want_conflict = random.random() < CONFLICT_PROB
    if want_conflict:
        color_count = random.choice((2, 3))
        colors = random.sample(list(PALETTE), color_count)
        specs = _conflict_specs(tile_rows, tile_cols, colors)
        if specs is None:
            return None
    else:
        color_count = random.choice((2, 3))
        colors = random.sample(list(PALETTE), color_count)
        specs = _nonconflict_specs(tile_rows, tile_cols, colors)

    groups: list[dict] = []
    used_endpoint_tiles: set[tuple[int, int]] = set()
    if not _place_group(specs, tile_rows, tile_cols, groups, used_endpoint_tiles):
        return None

    inp = _paint_endpoints(_blank_lattice(tile_rows, tile_cols), groups)
    out = _planned_output(tile_rows, tile_cols, groups)
    if not _validate_generated_case(inp, out, tile_rows, tile_cols, groups, want_conflict):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid alternating-visibility lattice input")
