import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _trace_path(grid: np.ndarray, source: tuple[int, int]) -> dict:
    r0, c0 = source
    color = int(grid[r0, c0])
    assert color not in (BLACK, GREY), "source must be colored"

    neighbors: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r0 + dr, c0 + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BLACK:
            neighbors.append((dr, dc))
    assert len(neighbors) == 1, "each source must attach to exactly one path direction"

    dr, dc = neighbors[0]
    orientation = "H" if dc != 0 else "V"

    cells: list[tuple[int, int]] = []
    distances: dict[tuple[int, int], int] = {}
    r, c = r0, c0
    dist = 0
    while _in_bounds(grid.shape, r, c) and int(grid[r, c]) != BLACK:
        if dist > 0:
            assert int(grid[r, c]) == GREY, "only the source endpoint may be colored"
        cells.append((r, c))
        distances[(r, c)] = dist

        rr, cc = r + dr, c + dc
        if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) == BLACK:
            break
        r, c = rr, cc
        dist += 1

    assert len(cells) >= 2, "path must have a grey endpoint beyond the source"
    assert int(grid[cells[-1]]) == GREY, "the non-source endpoint must be grey"
    return {
        "color": color,
        "orientation": orientation,
        "cells": cells,
        "distances": distances,
    }


def _extract_paths(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], list[tuple[int, str, int]]]]:
    sources = [tuple(map(int, coord)) for coord in np.argwhere((grid != BLACK) & (grid != GREY))]
    assert sources, "expected at least one colored source endpoint"

    paths = [_trace_path(grid, source) for source in sources]

    cell_entries: dict[tuple[int, int], list[tuple[int, str, int]]] = defaultdict(list)
    for path in paths:
        for cell in path["cells"]:
            cell_entries[cell].append(
                (
                    int(path["distances"][cell]),
                    str(path["orientation"]),
                    int(path["color"]),
                )
            )

    non_black_cells = {tuple(map(int, coord)) for coord in np.argwhere(grid != BLACK)}
    assert set(cell_entries) == non_black_cells, "every non-black pixel must belong to a traced path"

    for cell, entries in cell_entries.items():
        if len(entries) == 1:
            continue
        assert int(grid[cell]) == GREY, "shared cells must be grey intersections"
        assert len(entries) == 2, "each intersection must involve exactly two paths"
        assert {orientation for _, orientation, _ in entries} == {"H", "V"}, "intersections must be horizontal/vertical"

    return paths, cell_entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color along its straight grey path.

    At intersection pixels shared by a horizontal and vertical path, the path
    whose source is fewer steps away owns the cell. If the step counts tie, the
    horizontal path wins the intersection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _, cell_entries = _extract_paths(g)
    out = np.full_like(g, BLACK)

    for cell, entries in cell_entries.items():
        best_dist = min(dist for dist, _, _ in entries)
        tied = [entry for entry in entries if entry[0] == best_dist]
        if len(tied) == 1:
            chosen = tied[0]
        else:
            horizontal = [entry for entry in tied if entry[1] == "H"]
            chosen = horizontal[0] if horizontal else tied[0]
        out[cell] = int(chosen[2])

    return out


def _sample_spaced_positions(
    rng: random.Random,
    low: int,
    high: int,
    count: int,
    min_gap: int,
) -> list[int] | None:
    values = list(range(low, high + 1))
    if count <= 0 or count > len(values):
        return None
    for _ in range(80):
        picked = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _distance_to_intersection(path: dict, row: int, col: int) -> int:
    if path["orientation"] == "H":
        start_col = int(path["start"])
        end_col = int(path["end"])
        return col - start_col if path["source_side"] == "start" else end_col - col
    start_row = int(path["start"])
    end_row = int(path["end"])
    return row - start_row if path["source_side"] == "start" else end_row - row


def _intersection_relations(horizontals: list[dict], verticals: list[dict]) -> list[str]:
    relations: list[str] = []
    for h in horizontals:
        for v in verticals:
            row = int(h["fixed"])
            col = int(v["fixed"])
            dh = _distance_to_intersection(h, row, col)
            dv = _distance_to_intersection(v, row, col)
            if dh == dv:
                relations.append("tie")
            elif dh < dv:
                relations.append("h")
            else:
                relations.append("v")
    return relations


def _render_paths(shape: tuple[int, int], paths: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for path in paths:
        if path["orientation"] == "H":
            r = int(path["fixed"])
            c0 = int(path["start"])
            c1 = int(path["end"])
            grid[r, c0 : c1 + 1] = GREY
            source_c = c0 if path["source_side"] == "start" else c1
            grid[r, source_c] = int(path["color"])
        else:
            c = int(path["fixed"])
            r0 = int(path["start"])
            r1 = int(path["end"])
            grid[r0 : r1 + 1, c] = GREY
            source_r = r0 if path["source_side"] == "start" else r1
            grid[source_r, c] = int(path["color"])
    return grid


def _valid_mode(mode: str, relations: list[str]) -> bool:
    has_tie = "tie" in relations
    has_h = "h" in relations
    has_v = "v" in relations
    if mode == "tie":
        return has_tie
    if mode == "h_closer":
        return has_h and not has_tie
    if mode == "v_closer":
        return has_v and not has_tie
    if mode == "mixed":
        return has_tie and (has_h or has_v)
    raise ValueError(mode)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mode = rng.choice(["tie", "h_closer", "v_closer", "mixed"])
    height = rng.randint(12, 18)
    width = rng.randint(12, 18)

    h_count = rng.randint(1, 3)
    v_count = rng.randint(1, 3)
    if mode == "mixed" and h_count * v_count < 2:
        return None

    rows = _sample_spaced_positions(rng, 3, height - 4, h_count, min_gap=2)
    cols = _sample_spaced_positions(rng, 3, width - 4, v_count, min_gap=2)
    if rows is None or cols is None:
        return None

    colors = rng.sample(PALETTE, h_count + v_count)
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)

    left_room = min_col - 1
    right_room = width - 2 - max_col
    top_room = min_row - 1
    bottom_room = height - 2 - max_row
    if min(left_room, right_room, top_room, bottom_room) < 2:
        return None

    horizontals: list[dict] = []
    for i, row in enumerate(rows):
        left_extra = rng.randint(2, left_room)
        right_extra = rng.randint(2, right_room)
        horizontals.append(
            {
                "orientation": "H",
                "fixed": row,
                "start": min_col - left_extra,
                "end": max_col + right_extra,
                "source_side": rng.choice(["start", "end"]),
                "color": colors[i],
            }
        )

    verticals: list[dict] = []
    for j, col in enumerate(cols):
        top_extra = rng.randint(2, top_room)
        bottom_extra = rng.randint(2, bottom_room)
        verticals.append(
            {
                "orientation": "V",
                "fixed": col,
                "start": min_row - top_extra,
                "end": max_row + bottom_extra,
                "source_side": rng.choice(["start", "end"]),
                "color": colors[h_count + j],
            }
        )

    relations = _intersection_relations(horizontals, verticals)
    if not _valid_mode(mode, relations):
        return None

    grid = _render_paths((height, width), horizontals + verticals)

    try:
        parsed_paths, _ = _extract_paths(grid)
    except AssertionError:
        return None

    if len(parsed_paths) != h_count + v_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
