from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


Coord = tuple[int, int]


# Generator palette: black background, three gate colors, and disjoint path colors.
BG = 0
GATE_COLORS = (6, 7, 8)
PATH_COLORS = (1, 2, 3, 4, 5, 9)

_RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _is_domino(comp: list[Coord]) -> bool:
    if len(comp) != 2:
        return False
    (r0, c0), (r1, c1) = comp
    return abs(r0 - r1) + abs(c0 - c1) == 1


def _classify_roles(grid: np.ndarray) -> tuple[list[int], dict[int, dict[str, object]]]:
    bg = _get_bg(grid)
    path_colors: list[int] = []
    gate_components: dict[int, dict[str, object]] = {}
    gate_id = 0

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        if comps and all(_is_domino(comp) for comp in comps):
            for comp in comps:
                cells = tuple(sorted(comp))
                orient = "h" if cells[0][0] == cells[1][0] else "v"
                gate_components[gate_id] = {
                    "id": gate_id,
                    "color": int(color),
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "orient": orient,
                }
                gate_id += 1
        else:
            path_colors.append(int(color))

    assert path_colors, "expected at least one path color"
    assert gate_components, "expected at least one gate domino"
    return path_colors, gate_components


def _gap_gate_candidates(
    shape: tuple[int, int],
    r: int,
    c: int,
    axis: str,
) -> list[frozenset[Coord]]:
    candidates: list[frozenset[Coord]] = []
    if axis == "h":
        options = [((r - 2, c), (r - 1, c)), ((r + 1, c), (r + 2, c))]
    else:
        options = [((r, c - 2), (r, c - 1)), ((r, c + 1), (r, c + 2))]
    for a, b in options:
        if _in_bounds(shape, *a) and _in_bounds(shape, *b):
            candidates.append(frozenset((a, b)))
    return candidates


def _reconstruct_path(
    grid: np.ndarray,
    path_color: int,
    gate_components: dict[int, dict[str, object]],
) -> tuple[np.ndarray, dict[Coord, int]]:
    h, w = grid.shape
    full_mask = grid == path_color
    gate_lookup = {
        gate["cell_set"]: gate_id for gate_id, gate in gate_components.items()
    }
    gap_to_gate: dict[Coord, int] = {}

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            horizontal = (
                c - 1 >= 0
                and c + 1 < w
                and int(grid[r, c - 1]) == path_color
                and int(grid[r, c + 1]) == path_color
            )
            vertical = (
                r - 1 >= 0
                and r + 1 < h
                and int(grid[r - 1, c]) == path_color
                and int(grid[r + 1, c]) == path_color
            )
            if horizontal == vertical:
                continue

            axis = "h" if horizontal else "v"
            matches = [
                gate_lookup[cells]
                for cells in _gap_gate_candidates(grid.shape, r, c, axis)
                if cells in gate_lookup
            ]
            if not matches:
                continue
            assert len(matches) == 1, "each gap must match exactly one gate"
            full_mask[r, c] = True
            gap_to_gate[(r, c)] = int(matches[0])

    comps = _components(full_mask)
    assert len(comps) == 1, "each path color must form one complete path after filling gaps"
    return full_mask, gap_to_gate


def _order_path(mask: np.ndarray, start: Coord) -> list[Coord]:
    cell_set = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
    ordered = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxt = [n for n in _neighbors4(*cur) if n in cell_set and n != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must be non-branching"
        prev, cur = cur, nxt[0]
        ordered.append(cur)
    assert len(ordered) == len(cell_set), "path traversal must cover every cell exactly once"
    return ordered


def _endpoint_run_length(ordered: list[Coord], original_cells: set[Coord]) -> int:
    assert ordered
    assert ordered[0] in original_cells
    if len(ordered) == 1:
        return 1

    run_len = 1
    direction = (ordered[1][0] - ordered[0][0], ordered[1][1] - ordered[0][1])
    for i in range(1, len(ordered)):
        if ordered[i] not in original_cells:
            break
        run_len += 1
        if i == len(ordered) - 1:
            break
        next_dir = (
            ordered[i + 1][0] - ordered[i][0],
            ordered[i + 1][1] - ordered[i][1],
        )
        if next_dir != direction:
            break
        direction = next_dir
    return int(run_len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rejoin same-color path fragments across gated 1-cell breaks, then recolor
    each earlier path segment by the next gate encountered from the shorter end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _get_bg(g) == BG, "expected black as the dominant background"

    path_colors, gate_components = _classify_roles(g)
    used_gates: set[int] = set()
    out = g.copy()

    for path_color in path_colors:
        full_mask, gap_to_gate = _reconstruct_path(g, path_color, gate_components)
        cell_set = {tuple(map(int, cell)) for cell in np.argwhere(full_mask)}
        degrees = {
            cell: sum((nbr in cell_set) for nbr in _neighbors4(*cell))
            for cell in cell_set
        }
        endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
        assert len(endpoints) == 2, "completed path must have exactly two endpoints"

        ordered_a = _order_path(full_mask, endpoints[0])
        ordered_b = list(reversed(ordered_a))
        original_cells = {
            tuple(map(int, cell))
            for cell in np.argwhere(g == path_color)
        }
        len_a = _endpoint_run_length(ordered_a, original_cells)
        len_b = _endpoint_run_length(ordered_b, original_cells)
        ordered = ordered_a if (len_a, endpoints[0]) <= (len_b, endpoints[1]) else ordered_b

        gate_events = []
        for idx, cell in enumerate(ordered):
            gate_id = gap_to_gate.get(cell)
            if gate_id is None:
                continue
            gate_events.append((idx, int(gate_components[gate_id]["color"])))
            used_gates.add(gate_id)

        event_ptr = 0
        active_color = path_color if not gate_events else gate_events[0][1]
        for idx, cell in enumerate(ordered):
            out[cell] = active_color
            if event_ptr < len(gate_events) and idx == gate_events[event_ptr][0]:
                event_ptr += 1
                active_color = (
                    path_color if event_ptr == len(gate_events) else gate_events[event_ptr][1]
                )

    assert used_gates == set(gate_components), "every gate must belong to exactly one path gap"
    return out


def _line_gap_offsets(lengths: list[int]) -> list[int]:
    offsets: list[int] = []
    pos = 0
    for i, length in enumerate(lengths[:-1]):
        pos += int(length)
        offsets.append(pos)
        pos += 1
    return offsets


def _sample_fragment_lengths(max_total: int) -> list[int] | None:
    shorter_first = bool(int(_RNG.integers(0, 2)))
    for _ in range(80):
        n_fragments = int(_RNG.integers(2, 5))
        lengths = [int(_RNG.integers(3, 9)) for _ in range(n_fragments)]
        total = sum(lengths) + (n_fragments - 1)
        if total > max_total:
            continue
        if shorter_first and not (lengths[0] < lengths[-1]):
            continue
        if (not shorter_first) and not (lengths[0] > lengths[-1]):
            continue
        return lengths
    return None


def _halo(cells: Iterable[Coord], shape: tuple[int, int]) -> set[Coord]:
    reserved: set[Coord] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_bounds(shape, rr, cc):
                    reserved.add((rr, cc))
    return reserved


def _build_path_spec(
    shape: tuple[int, int],
    blocked: set[Coord],
    path_color: int,
) -> dict[str, object] | None:
    h, w = shape
    orientation = "h" if int(_RNG.integers(0, 2)) == 0 else "v"
    max_total = (w - 2) if orientation == "h" else (h - 2)
    lengths = _sample_fragment_lengths(max_total)
    if lengths is None:
        return None

    total = sum(lengths) + len(lengths) - 1
    gap_offsets = _line_gap_offsets(lengths)
    gate_colors = [int(v) for v in _RNG.choice(GATE_COLORS, size=len(gap_offsets), replace=True)]
    gate_sides = [int(v) for v in _RNG.choice(np.array([-1, 1]), size=len(gap_offsets), replace=True)]
    gap_color_map: dict[Coord, int] = {}

    if orientation == "h":
        if h < 5 or w < total + 2:
            return None
        row = int(_RNG.integers(2, h - 2))
        col0 = int(_RNG.integers(1, w - total))
        full_cells = [(row, col0 + dx) for dx in range(total)]
        gap_cells = {(row, col0 + off) for off in gap_offsets}
        gate_cells: list[tuple[int, tuple[Coord, Coord]]] = []
        for off, gate_color, side in zip(gap_offsets, gate_colors, gate_sides):
            gap = (row, col0 + off)
            gc = (row + side, col0 + off), (row + 2 * side, col0 + off)
            gap_color_map[gap] = int(gate_color)
            gate_cells.append((int(gate_color), gc))
    else:
        if w < 5 or h < total + 2:
            return None
        col = int(_RNG.integers(2, w - 2))
        row0 = int(_RNG.integers(1, h - total))
        full_cells = [(row0 + dy, col) for dy in range(total)]
        gap_cells = {(row0 + off, col) for off in gap_offsets}
        gate_cells = []
        for off, gate_color, side in zip(gap_offsets, gate_colors, gate_sides):
            gap = (row0 + off, col)
            gc = (row0 + off, col + side), (row0 + off, col + 2 * side)
            gap_color_map[gap] = int(gate_color)
            gate_cells.append((int(gate_color), gc))

    actual_cells = set(full_cells) - gap_cells
    for _gate_color, cells in gate_cells:
        actual_cells.update(cells)
    if any(cell in blocked for cell in actual_cells):
        return None

    return {
        "path_color": int(path_color),
        "orientation": orientation,
        "lengths": lengths,
        "full_cells": full_cells,
        "gap_cells": gap_cells,
        "gap_colors": gap_color_map,
        "gate_components": [(int(color), tuple(cells)) for color, cells in gate_cells],
        "reserved": _halo(actual_cells, shape),
    }


def _render_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        color = int(spec["path_color"])
        for cell in spec["full_cells"]:
            if cell not in spec["gap_cells"]:
                grid[cell] = color
        for gate_color, cells in spec["gate_components"]:
            for cell in cells:
                grid[cell] = int(gate_color)
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        path_color = int(spec["path_color"])
        full_cells = list(spec["full_cells"])
        gap_colors: dict[Coord, int] = spec["gap_colors"]  # type: ignore[assignment]
        if int(spec["lengths"][0]) > int(spec["lengths"][-1]):
            full_cells = list(reversed(full_cells))

        gate_events = [(idx, gap_colors[cell]) for idx, cell in enumerate(full_cells) if cell in gap_colors]
        event_ptr = 0
        active_color = path_color if not gate_events else int(gate_events[0][1])
        for idx, cell in enumerate(full_cells):
            out[cell] = active_color
            if event_ptr < len(gate_events) and idx == gate_events[event_ptr][0]:
                event_ptr += 1
                active_color = path_color if event_ptr == len(gate_events) else int(gate_events[event_ptr][1])

        for gate_color, cells in spec["gate_components"]:
            for cell in cells:
                out[cell] = int(gate_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(18, 31))
    w = int(_RNG.integers(18, 31))
    shape = (h, w)
    path_count = int(_RNG.integers(3, 5))
    path_colors = [int(v) for v in _RNG.choice(PATH_COLORS, size=path_count, replace=False)]

    specs: list[dict[str, object]] = []
    blocked: set[Coord] = set()
    for path_color in path_colors:
        placed = None
        for _ in range(120):
            candidate = _build_path_spec(shape, blocked, path_color)
            if candidate is not None:
                placed = candidate
                break
        if placed is None:
            return None
        specs.append(placed)
        blocked.update(placed["reserved"])  # type: ignore[arg-type]

    inp = _render_input(shape, specs)
    out = _render_expected(shape, specs)

    counts = np.bincount(inp.ravel(), minlength=10)
    if int(np.argmax(counts)) != BG:
        return None
    if len(np.unique(inp)) < 5:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
