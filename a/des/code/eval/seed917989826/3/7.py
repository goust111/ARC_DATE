from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

CORNER_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BR": (1, 1),
    "BL": (1, 0),
}
CLOCKWISE_CORNERS = ("TL", "TR", "BR", "BL")
PATTERNS_BY_SOURCE_LOC = {
    "left": {"missing": "BL", "path": ("TL", "TR", "BR"), "target_loc": "down"},
    "up": {"missing": "TL", "path": ("TR", "BR", "BL"), "target_loc": "left"},
    "right": {"missing": "TR", "path": ("BR", "BL", "TL"), "target_loc": "up"},
    "down": {"missing": "BR", "path": ("BL", "TL", "TR"), "target_loc": "right"},
}
PATTERNS_BY_MISSING = {
    spec["missing"]: {
        "source_loc": source_loc,
        "target_loc": spec["target_loc"],
        "path": spec["path"],
    }
    for source_loc, spec in PATTERNS_BY_SOURCE_LOC.items()
}

GRID_MIN = 18
GRID_MAX = 30
MAX_OBJECTS = len(PALETTE)
CHAIN_PLACEMENT_TRIES = 140
STEP_TRIES = 80
ISOLATE_TRIES = 120
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _square_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top_left
    return [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]


def _paint_cells(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _cell_set(cells: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in cells}


def _contact_cell(top_left: tuple[int, int], square_loc: str, idx: int) -> tuple[int, int]:
    r0, c0 = top_left
    if square_loc == "left":
        return r0 + idx, c0 + 2
    if square_loc == "up":
        return r0 + 2, c0 + idx
    if square_loc == "right":
        return r0 + idx, c0
    assert square_loc == "down"
    return r0, c0 + idx


def _step_from_square(square_loc: str) -> tuple[int, int]:
    return {
        "left": (0, 1),
        "up": (1, 0),
        "right": (0, -1),
        "down": (-1, 0),
    }[square_loc]


def _step_to_square(square_loc: str) -> tuple[int, int]:
    return {
        "left": (0, -1),
        "up": (-1, 0),
        "right": (0, 1),
        "down": (1, 0),
    }[square_loc]


def _top_left_from_contact(contact: tuple[int, int], square_loc: str, idx: int) -> tuple[int, int]:
    r, c = contact
    if square_loc == "left":
        return r - idx, c - 2
    if square_loc == "up":
        return r - 2, c - idx
    if square_loc == "right":
        return r - idx, c
    assert square_loc == "down"
    return r, c - idx


def _can_place_square(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = _cell_set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if int(grid[nr, nc]) != BG:
                return False
    return True


def _can_place_arrow(
    grid: np.ndarray,
    arrow_cells: list[tuple[int, int]],
    allowed_contacts: set[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    h, w = grid.shape
    arrow_set = _cell_set(arrow_cells)
    for r, c in arrow_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in arrow_set:
                continue
            value = int(grid[nr, nc])
            if value == BG:
                continue
            if ((r, c), (nr, nc)) in allowed_contacts:
                continue
            return False
    return True


def _mode_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def _decode_arrow(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = _cell_set(comp)
    assert len(coords) == 3, "each arrow must use exactly three grey cells"
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    assert r1 == r0 + 1 and c1 == c0 + 1, "arrow must fit a 2x2 bbox"

    corners = {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BR": (r1, c1),
        "BL": (r1, c0),
    }
    missing = [name for name, cell in corners.items() if cell not in coords]
    assert len(missing) == 1, "grey component must be an L tromino"
    missing_name = missing[0]

    start_idx = (CLOCKWISE_CORNERS.index(missing_name) + 1) % 4
    path_names = [CLOCKWISE_CORNERS[(start_idx + k) % 4] for k in range(3)]
    path = [corners[name] for name in path_names]
    start, bend, end = path
    source_contact = (2 * start[0] - bend[0], 2 * start[1] - bend[1])
    target_contact = (2 * end[0] - bend[0], 2 * end[1] - bend[1])
    return source_contact, target_contact


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each grey L-arrow back to the start of its chain and repaint every
    square on that chain with the source square's color. Grey arrow pixels are
    removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG if np.any(g == BG) else _mode_color(g)

    squares: list[dict[str, object]] = []
    cell_to_square: dict[tuple[int, int], int] = {}
    for color in map(int, np.unique(g)):
        if color in {bg, GREY}:
            continue
        for comp in _components(g == color):
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            r0, c0 = min(rows), min(cols)
            r1, c1 = max(rows), max(cols)
            assert len(comp) == 9, "every colored object must be a 3x3 block"
            assert r1 == r0 + 2 and c1 == c0 + 2, "every colored object must have a 3x3 bbox"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "colored objects must be solid"
            square_id = len(squares)
            cells = _square_cells((r0, c0))
            for cell in cells:
                cell_to_square[cell] = square_id
            squares.append({"color": color, "cells": cells})

    out = g.copy()
    out[g == GREY] = bg

    pred: dict[int, int] = {}
    succ: dict[int, int] = {}
    for comp in _components(g == GREY):
        source_contact, target_contact = _decode_arrow(comp)
        assert source_contact in cell_to_square, "arrow tail must touch a source square"
        assert target_contact in cell_to_square, "arrow tip must touch a target square"
        src = cell_to_square[source_contact]
        dst = cell_to_square[target_contact]
        assert src != dst, "an arrow must connect distinct squares"
        assert src not in succ, "a square may have at most one outgoing arrow"
        assert dst not in pred, "a square may have at most one incoming arrow"
        succ[src] = dst
        pred[dst] = src

    source_cache: dict[int, int] = {}

    def source_of(square_id: int) -> int:
        if square_id in source_cache:
            return source_cache[square_id]
        seen: set[int] = set()
        cur = square_id
        while cur in pred:
            assert cur not in seen, "arrow graph must be acyclic"
            seen.add(cur)
            cur = pred[cur]
        source_cache[square_id] = cur
        return cur

    involved = set(pred) | set(succ)
    for square_id in involved:
        src = source_of(square_id)
        color = int(squares[src]["color"])
        for r, c in squares[square_id]["cells"]:
            out[r, c] = color

    return out


def _random_structure() -> tuple[list[int], int]:
    while True:
        chain_count = random.randint(1, 3)
        chain_lengths = [random.randint(2, 4) for _ in range(chain_count)]
        isolate_count = random.randint(0, 2)
        total = sum(chain_lengths) + isolate_count
        if total <= MAX_OBJECTS and total >= 3:
            return chain_lengths, isolate_count


def _initial_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    margin_r = 4 if h >= 15 else 2
    margin_c = 4 if w >= 15 else 2
    rows = range(margin_r, h - margin_r - 2)
    cols = range(margin_c, w - margin_c - 2)
    positions = [(r, c) for r in rows for c in cols]
    random.shuffle(positions)
    return positions


def _try_add_step(
    grid: np.ndarray,
    source_top_left: tuple[int, int],
    target_color: int,
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    source_locs = list(PATTERNS_BY_SOURCE_LOC)
    random.shuffle(source_locs)
    for source_loc in source_locs:
        spec = PATTERNS_BY_SOURCE_LOC[source_loc]
        combos = [(src_idx, dst_idx) for src_idx in range(3) for dst_idx in range(3)]
        random.shuffle(combos)
        for src_idx, dst_idx in combos[:STEP_TRIES]:
            source_contact = _contact_cell(source_top_left, source_loc, src_idx)
            step_r, step_c = _step_from_square(source_loc)
            start = (source_contact[0] + step_r, source_contact[1] + step_c)

            start_corner = CORNER_OFFSETS[spec["path"][0]]
            box_r0 = start[0] - start_corner[0]
            box_c0 = start[1] - start_corner[1]
            arrow_cells = [
                (box_r0 + CORNER_OFFSETS[name][0], box_c0 + CORNER_OFFSETS[name][1])
                for name in spec["path"]
            ]

            end = arrow_cells[-1]
            tip_r, tip_c = _step_to_square(spec["target_loc"])
            target_contact = (end[0] + tip_r, end[1] + tip_c)
            target_top_left = _top_left_from_contact(target_contact, spec["target_loc"], dst_idx)
            target_cells = _square_cells(target_top_left)
            if not _can_place_square(grid, target_cells):
                continue

            _paint_cells(grid, target_cells, target_color)
            allowed_contacts = {
                (arrow_cells[0], source_contact),
                (arrow_cells[-1], target_contact),
            }
            if _can_place_arrow(grid, arrow_cells, allowed_contacts):
                _paint_cells(grid, arrow_cells, GREY)
                return target_top_left, target_cells, arrow_cells
            _paint_cells(grid, target_cells, BG)
    return None


def _place_chain(
    base_grid: np.ndarray,
    colors: list[int],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    for _ in range(CHAIN_PLACEMENT_TRIES):
        grid = base_grid.copy()
        placed: list[dict[str, object]] = []
        for top_left in _initial_positions(grid.shape):
            cells = _square_cells(top_left)
            if not _can_place_square(grid, cells):
                continue
            _paint_cells(grid, cells, colors[0])
            placed.append({"color": colors[0], "cells": cells, "top_left": top_left})
            break
        if not placed:
            continue

        ok = True
        current_top_left = placed[0]["top_left"]
        for color in colors[1:]:
            step = _try_add_step(grid, current_top_left, color)
            if step is None:
                ok = False
                break
            current_top_left, target_cells, _arrow_cells = step
            placed.append({"color": color, "cells": target_cells, "top_left": current_top_left})
        if ok:
            return grid, placed
    return None


def _place_isolate(grid: np.ndarray, color: int) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    positions = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    random.shuffle(positions)
    for top_left in positions[:ISOLATE_TRIES]:
        cells = _square_cells(top_left)
        if _can_place_square(grid, cells):
            _paint_cells(grid, cells, color)
            return cells
    return None


def _build_expected(
    inp: np.ndarray,
    chains: list[list[dict[str, object]]],
) -> np.ndarray:
    out = inp.copy()
    out[out == GREY] = BG
    for chain in chains:
        source_color = int(chain[0]["color"])
        for square in chain:
            for r, c in square["cells"]:
                out[r, c] = source_color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        chain_lengths, isolate_count = _random_structure()
        total_objects = sum(chain_lengths) + isolate_count
        colors = random.sample(PALETTE, total_objects)

        grid = np.full((h, w), BG, dtype=int)
        chains: list[list[dict[str, object]]] = []
        color_idx = 0
        success = True

        for length in sorted(chain_lengths, reverse=True):
            chain_colors = colors[color_idx : color_idx + length]
            color_idx += length
            placed = _place_chain(grid, chain_colors)
            if placed is None:
                success = False
                break
            grid, chain = placed
            chains.append(chain)

        if not success:
            continue

        for _ in range(isolate_count):
            cells = _place_isolate(grid, colors[color_idx])
            color_idx += 1
            if cells is None:
                success = False
                break

        if not success:
            continue

        expected = _build_expected(grid, chains)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
