import random
from collections import defaultdict

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
MAROON = 9

RAIL_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
PATH_LEN_RANGE = (6, 9)
FINAL_PATH_LEN_RANGE = (5, 8)
IGNORED_ARM_RANGE = (1, 3)
ISOLATED_LEN_RANGE = (4, 7)
MAX_GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_legend(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    for corner in ("tl", "tr", "bl", "br"):
        if corner in ("tl", "tr"):
            r = 0
            inward_r = 1
        else:
            r = h - 1
            inward_r = h - 2

        if corner in ("tl", "bl"):
            if grid[r, 0] in (BG, MAGENTA):
                continue
            cols = []
            c = 0
            while c < w and grid[r, c] not in (BG, MAGENTA):
                cols.append(c)
                c += 1
            if c < w and grid[r, c] != BG:
                continue
        else:
            if grid[r, w - 1] in (BG, MAGENTA):
                continue
            cols = []
            c = w - 1
            while c >= 0 and grid[r, c] not in (BG, MAGENTA):
                cols.append(c)
                c -= 1
            cols.reverse()
            if c >= 0 and grid[r, c] != BG:
                continue

        if len(cols) < 2:
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != len(colors):
            continue
        if not np.all(grid[inward_r, cols] == BG):
            continue
        candidates.append([(r, c) for c in cols])

    assert candidates, "legend not found"
    candidates.sort(key=len, reverse=True)
    cells = candidates[0]
    colors = [int(grid[r, c]) for r, c in cells]
    return cells, colors


def _reconstruct_segments(
    grid: np.ndarray, line_mask: np.ndarray, legend_colors: list[int]
) -> tuple[dict[int, dict[str, int]], dict[tuple[int, int], list[int]]]:
    visible = np.where(line_mask, grid, BG)
    segments: dict[int, dict[str, int]] = {}
    cell_members: dict[tuple[int, int], list[int]] = defaultdict(list)

    for color in legend_colors:
        pts = np.argwhere(visible == color)
        if len(pts) == 0:
            continue
        rows = np.unique(pts[:, 0])
        cols = np.unique(pts[:, 1])
        if len(rows) == 1 and len(cols) >= 2:
            r = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            assert np.all(line_mask[r, c0 : c1 + 1]), "horizontal segment must be contiguous"
            segments[color] = {"orient": 0, "fixed": r, "lo": c0, "hi": c1}
            for c in range(c0, c1 + 1):
                cell_members[(r, c)].append(color)
        elif len(cols) == 1 and len(rows) >= 2:
            c = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            assert np.all(line_mask[r0 : r1 + 1, c]), "vertical segment must be contiguous"
            segments[color] = {"orient": 1, "fixed": c, "lo": r0, "hi": r1}
            for r in range(r0, r1 + 1):
                cell_members[(r, c)].append(color)
        else:
            raise AssertionError(f"color {color} is not a single straight segment")

    assert segments, "no segments reconstructed"
    return segments, cell_members


def _segment_neighbors(spec: dict[str, int], pos: tuple[int, int]) -> list[tuple[int, int]]:
    orient = spec["orient"]
    fixed = spec["fixed"]
    lo = spec["lo"]
    hi = spec["hi"]
    r, c = pos
    out: list[tuple[int, int]] = []
    if orient == 0:
        assert r == fixed and lo <= c <= hi
        if c > lo:
            out.append((r, c - 1))
        if c < hi:
            out.append((r, c + 1))
    else:
        assert c == fixed and lo <= r <= hi
        if r > lo:
            out.append((r - 1, c))
        if r < hi:
            out.append((r + 1, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_cells, legend_colors = _find_legend(g)
    legend_set = set(legend_cells)

    sources = np.argwhere(g == MAGENTA)
    assert len(sources) == 1, "expected one signal source"
    source = (int(sources[0, 0]), int(sources[0, 1]))
    assert source not in legend_set

    line_mask = g != BG
    line_mask[source] = False
    for cell in legend_cells:
        line_mask[cell] = False

    segments, cell_members = _reconstruct_segments(g, line_mask, legend_colors)
    rank = {color: i for i, color in enumerate(legend_colors)}

    start_neighbors = [
        (nr, nc)
        for nr, nc in neighbors4(*source)
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and line_mask[nr, nc]
    ]
    assert len(start_neighbors) == 1, "source must touch exactly one endpoint"
    current = start_neighbors[0]
    start_colors = cell_members[current]
    assert len(start_colors) == 1, "source endpoint must not be a crossing"
    current_color = start_colors[0]

    out = g.copy()
    visited: set[tuple[int, int]] = set()
    prev = source

    while True:
        out[current] = MAROON
        visited.add(current)

        turn_color = None
        turn_next = None
        for other in sorted(
            [c for c in cell_members[current] if c != current_color and rank[c] < rank[current_color]],
            key=lambda c: rank[c],
        ):
            candidates = [
                nxt
                for nxt in _segment_neighbors(segments[other], current)
                if nxt != prev and nxt not in visited
            ]
            if len(candidates) > 1:
                raise AssertionError("turning direction is ambiguous")
            if len(candidates) == 1:
                turn_color = other
                turn_next = candidates[0]
                break
        if turn_color is not None:
            prev, current, current_color = current, turn_next, turn_color
            continue

        forward = [
            nxt
            for nxt in _segment_neighbors(segments[current_color], current)
            if nxt != prev and nxt not in visited
        ]
        if not forward:
            break
        assert len(forward) == 1, "forward propagation is ambiguous"
        prev, current = current, forward[0]

    return out


def _in_line_area(r: int, c: int, h: int, w: int) -> bool:
    return 2 <= r <= h - 3 and 2 <= c <= w - 3


def _line_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(start[0] + dr * i, start[1] + dc * i) for i in range(length)]


def _can_place_line(
    cells: list[tuple[int, int]],
    occupied: dict[tuple[int, int], set[int]],
    h: int,
    w: int,
    source: tuple[int, int] | None,
    source_neighbor: tuple[int, int] | None,
    shared_cells: set[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    for cell in cells:
        r, c = cell
        if not _in_line_area(r, c, h, w):
            return False
        if cell in occupied:
            if cell not in shared_cells or len(occupied[cell]) != 1:
                return False
        if source is not None and cell != source_neighbor and abs(r - source[0]) + abs(c - source[1]) == 1:
            return False

    for cell in cells:
        if cell in shared_cells:
            continue
        for neigh in neighbors4(*cell):
            if neigh in occupied and neigh not in cell_set:
                return False
            if source is not None and neigh == source and cell != source_neighbor:
                return False
    return True


def _add_line(
    occupied: dict[tuple[int, int], set[int]], color: int, cells: list[tuple[int, int]]
) -> None:
    for cell in cells:
        occupied.setdefault(cell, set()).add(color)


def _remove_line(
    occupied: dict[tuple[int, int], set[int]], color: int, cells: list[tuple[int, int]]
) -> None:
    for cell in cells:
        occupied[cell].remove(color)
        if not occupied[cell]:
            del occupied[cell]


def _perpendicular_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _legend_shadow(h: int, w: int, corner: str, length: int) -> set[tuple[int, int]]:
    if corner == "tl":
        return {(1, c) for c in range(length)}
    if corner == "tr":
        return {(1, c) for c in range(w - length, w)}
    if corner == "bl":
        return {(h - 2, c) for c in range(length)}
    return {(h - 2, c) for c in range(w - length, w)}


def _place_path_recursive(
    rng: random.Random,
    idx: int,
    path_colors: list[int],
    start: tuple[int, int],
    direction: tuple[int, int],
    occupied: dict[tuple[int, int], set[int]],
    specs: list[dict[str, object]],
    h: int,
    w: int,
    source: tuple[int, int],
) -> bool:
    final = idx == len(path_colors) - 1
    lengths = list(range(*(FINAL_PATH_LEN_RANGE if final else PATH_LEN_RANGE)))
    lengths.append((FINAL_PATH_LEN_RANGE if final else PATH_LEN_RANGE)[1])
    rng.shuffle(lengths)

    for length in lengths:
        cells = _line_cells(start, direction, length)
        shared = {start} if idx > 0 else set()
        if not _can_place_line(
            cells,
            occupied,
            h,
            w,
            source,
            start if idx == 0 else None,
            shared,
        ):
            continue

        _add_line(occupied, path_colors[idx], cells)
        if final:
            specs.append(
                {
                    "color": path_colors[idx],
                    "cells": cells,
                    "direction": direction,
                    "turn_idx": None,
                }
            )
            return True

        turn_choices = list(range(2, length - 2))
        rng.shuffle(turn_choices)
        next_dirs = _perpendicular_dirs(direction)
        for turn_idx in turn_choices:
            specs.append(
                {
                    "color": path_colors[idx],
                    "cells": cells,
                    "direction": direction,
                    "turn_idx": turn_idx,
                }
            )
            rng.shuffle(next_dirs)
            next_start = cells[turn_idx]
            for next_dir in next_dirs:
                if _place_path_recursive(
                    rng,
                    idx + 1,
                    path_colors,
                    next_start,
                    next_dir,
                    occupied,
                    specs,
                    h,
                    w,
                    source,
                ):
                    return True
            specs.pop()
        _remove_line(occupied, path_colors[idx], cells)
    return False


def _host_candidates(path_specs: list[dict[str, object]], used_crossings: set[tuple[int, int]]) -> list[tuple[int, int, tuple[int, int]]]:
    out = []
    for spec_index, spec in enumerate(path_specs):
        cells = spec["cells"]
        traversed_end = spec["turn_idx"] if spec["turn_idx"] is not None else len(cells) - 1
        for cell_index in range(1, traversed_end):
            cell = cells[cell_index]
            if cell in used_crossings:
                continue
            out.append((spec_index, cell_index, cell))
    return out


def _place_ignored_crossing(
    rng: random.Random,
    color: int,
    path_specs: list[dict[str, object]],
    occupied: dict[tuple[int, int], set[int]],
    used_crossings: set[tuple[int, int]],
    visibility: dict[tuple[int, int], int],
    h: int,
    w: int,
    source: tuple[int, int],
) -> bool:
    hosts = _host_candidates(path_specs, used_crossings)
    rng.shuffle(hosts)
    for spec_index, _cell_index, cell in hosts:
        host_spec = path_specs[spec_index]
        direction = host_spec["direction"]
        perp_a, perp_b = _perpendicular_dirs(direction)
        up_lengths = list(range(IGNORED_ARM_RANGE[0], IGNORED_ARM_RANGE[1] + 1))
        down_lengths = list(range(IGNORED_ARM_RANGE[0], IGNORED_ARM_RANGE[1] + 1))
        rng.shuffle(up_lengths)
        rng.shuffle(down_lengths)
        for left_len in up_lengths:
            for right_len in down_lengths:
                cells = []
                for step in range(left_len, 0, -1):
                    cells.append((cell[0] + perp_a[0] * step, cell[1] + perp_a[1] * step))
                cells.append(cell)
                for step in range(1, right_len + 1):
                    cells.append((cell[0] + perp_b[0] * step, cell[1] + perp_b[1] * step))
                if not _can_place_line(cells, occupied, h, w, source, None, {cell}):
                    continue
                _add_line(occupied, color, cells)
                used_crossings.add(cell)
                visibility[cell] = rng.choice([host_spec["color"], color])
                return True
    return False


def _place_isolated_segment(
    rng: random.Random,
    color: int,
    occupied: dict[tuple[int, int], set[int]],
    h: int,
    w: int,
    source: tuple[int, int],
) -> bool:
    for _ in range(80):
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        length = rng.randint(ISOLATED_LEN_RANGE[0], ISOLATED_LEN_RANGE[1])
        start = (rng.randint(2, h - 3), rng.randint(2, w - 3))
        cells = _line_cells(start, direction, length)
        if not _can_place_line(cells, occupied, h, w, source, None, set()):
            continue
        _add_line(occupied, color, cells)
        return True
    return False


def _render_grid(
    h: int,
    w: int,
    legend_corner: str,
    legend_colors: list[int],
    source: tuple[int, int],
    occupied: dict[tuple[int, int], set[int]],
    visibility: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    length = len(legend_colors)
    if legend_corner == "tl":
        legend_cells = [(0, c) for c in range(length)]
    elif legend_corner == "tr":
        legend_cells = [(0, w - length + c) for c in range(length)]
    elif legend_corner == "bl":
        legend_cells = [(h - 1, c) for c in range(length)]
    else:
        legend_cells = [(h - 1, w - length + c) for c in range(length)]
    for cell, color in zip(legend_cells, legend_colors):
        grid[cell] = color

    grid[source] = MAGENTA
    for cell, colors in occupied.items():
        if len(colors) == 1:
            grid[cell] = next(iter(colors))
        else:
            grid[cell] = visibility[cell]
    return grid


def _expected_output(grid: np.ndarray, path_specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    visited: set[tuple[int, int]] = set()
    for spec in path_specs:
        cells = spec["cells"]
        end = spec["turn_idx"] if spec["turn_idx"] is not None else len(cells) - 1
        visited.update(cells[: end + 1])
    for cell in visited:
        out[cell] = MAROON
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    n_colors = rng.randint(3, 5)
    legend_colors = rng.sample(list(RAIL_COLORS), n_colors)
    legend_corner = rng.choice(["tl", "tr", "bl", "br"])
    shadow = _legend_shadow(h, w, legend_corner, len(legend_colors))

    start_index = rng.randint(1, min(2, n_colors - 2))
    path_colors = [legend_colors[i] for i in range(start_index, -1, -1)]
    leftover_colors = [color for color in legend_colors if color not in path_colors]

    occupied: dict[tuple[int, int], set[int]] = {}
    path_specs: list[dict[str, object]] = []
    visibility: dict[tuple[int, int], int] = {}

    for _ in range(100):
        path_specs.clear()
        occupied.clear()
        direction = rng.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        start = (rng.randint(2, h - 3), rng.randint(2, w - 3))
        source = (start[0] - direction[0], start[1] - direction[1])
        if not (1 <= source[0] <= h - 2 and 1 <= source[1] <= w - 2):
            continue
        if source in shadow:
            continue
        if _place_path_recursive(rng, 0, path_colors, start, direction, occupied, path_specs, h, w, source):
            break
    else:
        return None

    used_crossings: set[tuple[int, int]] = set()
    for idx in range(len(path_specs) - 1):
        turn_cell = path_specs[idx]["cells"][path_specs[idx]["turn_idx"]]
        used_crossings.add(turn_cell)
        visibility[turn_cell] = path_specs[idx + 1]["color"]

    ignored_to_place = min(len(leftover_colors), rng.randint(1, min(2, len(leftover_colors))))
    for color in leftover_colors[:ignored_to_place]:
        if not _place_ignored_crossing(
            rng,
            color,
            path_specs,
            occupied,
            used_crossings,
            visibility,
            h,
            w,
            source,
        ):
            return None

    for color in leftover_colors[ignored_to_place:]:
        if not _place_isolated_segment(rng, color, occupied, h, w, source):
            return None

    grid = _render_grid(h, w, legend_corner, legend_colors, source, occupied, visibility)
    legend_cells, _ = _find_legend(grid)
    line_mask = grid != BG
    line_mask[source] = False
    for cell in legend_cells:
        line_mask[cell] = False
    visible = np.where(line_mask, grid, BG)

    for color in legend_colors:
        pts = np.argwhere(visible == color)
        if len(pts) < 2:
            return None
        rows = np.unique(pts[:, 0])
        cols = np.unique(pts[:, 1])
        if not ((len(rows) == 1 and len(cols) >= 2) or (len(cols) == 1 and len(rows) >= 2)):
            return None

    if sum(1 for r, c in neighbors4(*source) if 0 <= r < h and 0 <= c < w and grid[r, c] not in (BG, MAGENTA)) != 1:
        return None

    expected = _expected_output(grid, path_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
