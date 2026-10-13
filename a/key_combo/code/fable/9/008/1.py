from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
ADJACENT_EXIT_SIDES = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}
ELBOW_SIDE_PAIRS = [
    ("U", "L"),
    ("U", "R"),
    ("D", "L"),
    ("D", "R"),
]
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _boundary_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(w):
        cells.append((0, c))
        cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        cells.append((r, w - 1))
    return cells


def _bbox_from_entry(entry: tuple[int, int], side: str, offset: int, h: int, w: int) -> tuple[int, int]:
    r, c = entry
    if side == "U":
        return r, c - offset
    if side == "D":
        return r - h + 1, c - offset
    if side == "L":
        return r - offset, c
    if side == "R":
        return r - offset, c - w + 1
    raise ValueError(side)


def _opening_coord(r0: int, c0: int, h: int, w: int, side: str, offset: int) -> tuple[int, int]:
    if side == "U":
        return r0, c0 + offset
    if side == "D":
        return r0 + h - 1, c0 + offset
    if side == "L":
        return r0 + offset, c0
    if side == "R":
        return r0 + offset, c0 + w - 1
    raise ValueError(side)


def _bbox_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _expanded_bbox_cells(r0: int, c0: int, h: int, w: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h_grid, w_grid = shape
    cells = set()
    for r in range(max(0, r0 - 1), min(h_grid, r0 + h + 1)):
        for c in range(max(0, c0 - 1), min(w_grid, c0 + w + 1)):
            cells.add((r, c))
    return cells


def _perimeter_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c0 + w):
        cells.add((r0, c))
        cells.add((r0 + h - 1, c))
    for r in range(r0, r0 + h):
        cells.add((r, c0))
        cells.add((r, c0 + w - 1))
    return cells


def _build_piece(r0: int, c0: int, h: int, w: int, openings: dict[str, int]) -> dict:
    cells = _bbox_cells(r0, c0, h, w)
    opening_coords = {
        _opening_coord(r0, c0, h, w, side, offset): side for side, offset in openings.items()
    }
    walls = _perimeter_cells(r0, c0, h, w) - set(opening_coords)
    fill = cells - walls
    return {
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "h": h,
        "w": w,
        "walls": walls,
        "cells": cells,
        "fill": fill,
        "openings": tuple(opening_coords),
        "opening_sides": opening_coords,
        "terminal": len(opening_coords) == 1,
    }


def _piece_from_bbox(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> dict | None:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if any(int(grid[r, c]) != GREY for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
        return None

    openings: list[tuple[tuple[int, int], str]] = []
    for r, c in _boundary_cells(h, w):
        rr = r0 + r
        cc = c0 + c
        value = int(grid[rr, cc])
        if value == GREY:
            continue
        if value != BLACK:
            return None
        if rr == r0:
            side = "U"
        elif rr == r1:
            side = "D"
        elif cc == c0:
            side = "L"
        else:
            side = "R"
        if (rr, cc) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            return None
        openings.append(((rr, cc), side))

    if len(openings) not in (1, 2):
        return None
    if len(openings) == 2:
        sides = {side for _, side in openings}
        if len(sides) != 2:
            return None
        side_a, side_b = tuple(sides)
        if side_b not in ADJACENT_EXIT_SIDES[side_a]:
            return None

    opening_sides = {coord: side for coord, side in openings}
    walls = _perimeter_cells(r0, c0, h, w) - set(opening_sides)
    if any(int(grid[r, c]) != GREY for r, c in walls):
        return None
    cells = _bbox_cells(r0, c0, h, w)
    fill = cells - walls
    return {
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "walls": walls,
        "cells": cells,
        "fill": fill,
        "openings": tuple(opening_sides),
        "opening_sides": opening_sides,
        "terminal": len(opening_sides) == 1,
    }


def _parse_pieces(grid: np.ndarray) -> list[dict]:
    grey = {(int(r), int(c)) for r, c in np.argwhere(grid == GREY)}
    if not grey:
        return []
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r, c in grey:
        row_to_cols.setdefault(r, []).append(c)
        col_to_rows.setdefault(c, []).append(r)
    for cols in row_to_cols.values():
        cols.sort()
    for rows in col_to_rows.values():
        rows.sort()

    pieces: list[dict] = []
    used_walls: set[tuple[int, int]] = set()
    for r0, c0 in sorted(grey):
        if (r0, c0) in used_walls:
            continue
        for c1 in row_to_cols.get(r0, []):
            if c1 < c0 + 3 or (r0, c1) not in grey:
                continue
            for r1 in col_to_rows.get(c0, []):
                if r1 < r0 + 3 or (r1, c1) not in grey:
                    continue
                piece = _piece_from_bbox(grid, r0, c0, r1, c1)
                if piece is None:
                    continue
                if piece["walls"] & used_walls:
                    continue
                pieces.append(piece)
                used_walls.update(piece["walls"])
                break
            else:
                continue
            break
    assert used_walls == grey
    return pieces


def _find_source(grid: np.ndarray) -> tuple[tuple[int, int], int, tuple[int, int]]:
    h, w = grid.shape
    candidates = []
    for r, c in _boundary_cells(h, w):
        value = int(grid[r, c])
        if value not in (BLACK, GREY):
            candidates.append((r, c, value))
    assert len(candidates) == 1
    r, c, color = candidates[0]
    sides = []
    if r == 0:
        sides.append("U")
    if r == h - 1:
        sides.append("D")
    if c == 0:
        sides.append("L")
    if c == w - 1:
        sides.append("R")
    assert len(sides) == 1
    inward = DIRS[OPPOSITE[sides[0]]]
    return (r, c), color, inward


def _trace(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    source, color, direction = _find_source(g)
    pieces = _parse_pieces(g)
    terminals = [piece for piece in pieces if piece["terminal"]]
    assert len(terminals) == 1

    opening_to_piece = {}
    for idx, piece in enumerate(pieces):
        for opening in piece["openings"]:
            assert opening not in opening_to_piece
            opening_to_piece[opening] = idx

    out = g.copy()
    out[source] = color
    pos = source
    path_segments: list[tuple[int, int, int, int]] = []
    visited: list[dict] = []

    while True:
        nr = pos[0] + direction[0]
        nc = pos[1] + direction[1]
        assert _in_bounds(g.shape, nr, nc)
        start = pos
        while True:
            pos = (nr, nc)
            if pos in opening_to_piece:
                path_segments.append((start[0], start[1], pos[0], pos[1]))
                piece = pieces[opening_to_piece[pos]]
                visited.append(piece)
                for rr, cc in piece["fill"]:
                    out[rr, cc] = color
                if piece["terminal"]:
                    return out, path_segments, visited
                exit_opening = piece["openings"][0] if piece["openings"][1] == pos else piece["openings"][1]
                direction = DIRS[piece["opening_sides"][exit_opening]]
                pos = exit_opening
                start = pos
                break
            assert int(g[pos]) == BLACK
            out[pos] = color
            nr = pos[0] + direction[0]
            nc = pos[1] + direction[1]
            assert _in_bounds(g.shape, nr, nc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the boundary light ray through gray elbow pipes until it fills the unique terminal cup.

    Input structure:
    - A black background contains one colored source pixel on the outer border.
    - Inside the grid are several gray hollow rectangular pipe pieces made from 1-cell walls.
    - Most pieces have two single-cell openings on adjacent sides; exactly one piece has one opening.

    Rule:
    - Launch a ray from the source straight inward, perpendicular to the border, and color every traversed black cell with the source color.
    - When the ray reaches a two-opening gray piece through one opening, fill every black cell inside that piece with the source color, then continue from the other opening in that opening's outward normal direction.
    - When the ray reaches the unique one-opening piece, fill every black cell inside it with the source color and stop; untouched pieces remain unchanged.

    Key insight:
    - The single boundary source fixes the initial ray, and each pipe's opening count and opening sides uniquely determine whether it redirects the ray or terminates it.
    """
    out, _segments, _visited = _trace(np.asarray(grid, dtype=int))
    return out


def _side_for_direction(direction: tuple[int, int]) -> str:
    for side, delta in DIRS.items():
        if delta == direction:
            return side
    raise ValueError(direction)


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], gap: int) -> list[tuple[int, int]]:
    return [(start[0] + direction[0] * step, start[1] + direction[1] * step) for step in range(1, gap + 1)]


def _distance_to_edge(pos: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _sample_entry_offset(side: str, h: int, w: int) -> int:
    return random.randint(1, (w - 2) if side in {"U", "D"} else (h - 2))


def _piece_candidate_for_path(
    shape: tuple[int, int],
    entry: tuple[int, int],
    incoming: tuple[int, int],
    outgoing: tuple[int, int] | None,
    path_cells: set[tuple[int, int]],
    piece_cells: set[tuple[int, int]],
    expanded_piece_cells: set[tuple[int, int]],
) -> dict | None:
    entry_side = OPPOSITE[_side_for_direction(incoming)]
    exit_side = None if outgoing is None else _side_for_direction(outgoing)

    for _ in range(120):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        entry_offset = _sample_entry_offset(entry_side, h, w)
        openings = {entry_side: entry_offset}
        if exit_side is not None:
            exit_offset = _sample_entry_offset(exit_side, h, w)
            openings[exit_side] = exit_offset

        r0, c0 = _bbox_from_entry(entry, entry_side, entry_offset, h, w)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if r0 < 1 or c0 < 1 or r1 >= shape[0] - 1 or c1 >= shape[1] - 1:
            continue

        piece = _build_piece(r0, c0, h, w, openings)
        if entry not in piece["openings"]:
            continue
        if piece["cells"] & piece_cells:
            continue
        if any(cell in path_cells and cell != entry for cell in piece["cells"]):
            continue
        if piece["cells"] & expanded_piece_cells:
            continue

        if exit_side is not None:
            exit_candidates = [coord for coord in piece["openings"] if coord != entry]
            assert len(exit_candidates) == 1
            exit_opening = exit_candidates[0]
            if _distance_to_edge(exit_opening, outgoing, shape) < 4:
                continue
            piece["exit_opening"] = exit_opening
            piece["outgoing"] = outgoing
        return piece
    return None


def _random_free_piece(
    shape: tuple[int, int],
    forbidden_cells: set[tuple[int, int]],
    piece_cells: set[tuple[int, int]],
    expanded_piece_cells: set[tuple[int, int]],
) -> dict | None:
    h_grid, w_grid = shape
    for _ in range(200):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        r0 = random.randint(1, h_grid - h - 2)
        c0 = random.randint(1, w_grid - w - 2)
        side_a, side_b = random.choice(ELBOW_SIDE_PAIRS)
        openings = {
            side_a: _sample_entry_offset(side_a, h, w),
            side_b: _sample_entry_offset(side_b, h, w),
        }
        piece = _build_piece(r0, c0, h, w, openings)
        if piece["cells"] & forbidden_cells:
            continue
        if piece["cells"] & piece_cells:
            continue
        if piece["cells"] & expanded_piece_cells:
            continue
        return piece
    return None


def _draw_piece(grid: np.ndarray, piece: dict) -> None:
    for r, c in piece["walls"]:
        grid[r, c] = GREY


def _build_expected(grid: np.ndarray, source_color: int, source: tuple[int, int], path_cells: set[tuple[int, int]], main_pieces: list[dict]) -> np.ndarray:
    out = grid.copy()
    out[source] = source_color
    for r, c in path_cells:
        out[r, c] = source_color
    for piece in main_pieces:
        for r, c in piece["fill"]:
            out[r, c] = source_color
    return out


def _gen_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(20, 29)
    w = random.randint(20, 29)
    grid = np.zeros((h, w), dtype=int)
    source_color = random.choice(SOURCE_COLORS)

    source_side = random.choice(("U", "D", "L", "R"))
    if source_side == "U":
        source = (0, random.randint(2, w - 3))
    elif source_side == "D":
        source = (h - 1, random.randint(2, w - 3))
    elif source_side == "L":
        source = (random.randint(2, h - 3), 0)
    else:
        source = (random.randint(2, h - 3), w - 1)
    grid[source] = source_color
    directions = [DIRS[OPPOSITE[source_side]]]

    elbow_count = random.randint(2, 4)
    for _ in range(elbow_count):
        current_side = _side_for_direction(directions[-1])
        next_side = random.choice(ADJACENT_EXIT_SIDES[current_side])
        directions.append(DIRS[next_side])

    path_cells = {source}
    piece_cells: set[tuple[int, int]] = set()
    expanded_piece_cells: set[tuple[int, int]] = set()
    main_pieces: list[dict] = []
    current_pos = source

    for idx, direction in enumerate(directions):
        outgoing = directions[idx + 1] if idx + 1 < len(directions) else None
        piece = None
        for _ in range(120):
            gap = random.randint(2, 6)
            segment = _segment_cells(current_pos, direction, gap)
            if not all(_in_bounds(grid.shape, r, c) for r, c in segment):
                continue
            if any(cell in path_cells or cell in piece_cells for cell in segment[:-1]):
                continue
            entry = segment[-1]
            if entry in piece_cells:
                continue
            piece = _piece_candidate_for_path(
                grid.shape,
                entry,
                direction,
                outgoing,
                path_cells | set(segment[:-1]),
                piece_cells,
                expanded_piece_cells,
            )
            if piece is None:
                continue
            path_cells.update(segment)
            piece_cells.update(piece["cells"])
            expanded_piece_cells.update(_expanded_bbox_cells(piece["bbox"][0], piece["bbox"][1], piece["h"], piece["w"], grid.shape))
            main_pieces.append(piece)
            current_pos = piece.get("exit_opening", entry)
            break
        if piece is None:
            return None

    decoy_count = random.randint(1, 3)
    decoys: list[dict] = []
    forbidden = set(path_cells)
    for piece in main_pieces:
        forbidden.update(piece["cells"])

    for _ in range(decoy_count):
        piece = _random_free_piece(grid.shape, forbidden, piece_cells, expanded_piece_cells)
        if piece is None:
            return None
        decoys.append(piece)
        piece_cells.update(piece["cells"])
        expanded_piece_cells.update(_expanded_bbox_cells(piece["bbox"][0], piece["bbox"][1], piece["h"], piece["w"], grid.shape))
        forbidden.update(piece["cells"])

    for piece in main_pieces + decoys:
        _draw_piece(grid, piece)

    expected = _build_expected(grid, source_color, source, path_cells, main_pieces)

    try:
        parsed = _parse_pieces(grid)
        if len([piece for piece in parsed if piece["terminal"]]) != 1:
            return None
        actual, segments, visited = _trace(grid)
    except Exception:
        return None

    if len(visited) != elbow_count + 1:
        return None
    if len([piece for piece in visited if not piece["terminal"]]) != elbow_count:
        return None
    if len(path_cells) < 8:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if len(segments) != elbow_count + 1:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _gen_case()
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid elbow-pipe puzzle after {MAX_ATTEMPTS} attempts")
