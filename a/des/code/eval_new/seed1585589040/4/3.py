from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

BASE_WIRE_COLORS = (RED, BLUE, GREEN, YELLOW)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mix_colors(a: int, b: int) -> int:
    pair = frozenset((int(a), int(b)))
    if pair == frozenset((RED, BLUE)):
        return MAGENTA
    if pair == frozenset((RED, GREEN)):
        return YELLOW
    if pair == frozenset((BLUE, GREEN)):
        return AZURE
    if pair == frozenset((RED, YELLOW)):
        return ORANGE
    return MAROON


def _trace_wires(g: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], list[int]]]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    assert markers, "expected at least one grey marker"

    wires: list[dict[str, object]] = []
    source_cells: set[tuple[int, int]] = set()
    for mr, mc in markers:
        adjacent: list[tuple[int, int, int, int]] = []
        for dr, dc in DIR4:
            nr, nc = mr + dr, mc + dc
            if not _in_bounds(g.shape, nr, nc):
                continue
            if int(g[nr, nc]) in (BLACK, GREY):
                continue
            adjacent.append((nr, nc, dr, dc))
        assert len(adjacent) == 1, "each marker must touch exactly one wire endpoint"

        sr, sc, dr, dc = adjacent[0]
        assert (sr, sc) not in source_cells, "markers must identify distinct wires"
        source_cells.add((sr, sc))

        cells: list[tuple[int, int]] = []
        r, c = sr, sc
        while _in_bounds(g.shape, r, c) and int(g[r, c]) not in (BLACK, GREY):
            cells.append((r, c))
            r += dr
            c += dc

        assert cells, "wire trace must contain at least one cell"
        wires.append(
            {
                "marker": (mr, mc),
                "source": (sr, sc),
                "dir": (dr, dc),
                "kind": "h" if dr == 0 else "v",
                "cells": cells,
            }
        )

    cell_to_wires: dict[tuple[int, int], list[int]] = {}
    for wire_id, wire in enumerate(wires):
        for cell in wire["cells"]:
            cell_to_wires.setdefault(cell, []).append(wire_id)

    for cell, wire_ids in cell_to_wires.items():
        assert 1 <= len(wire_ids) <= 2, f"invalid overlap multiplicity at {cell}"
        if len(wire_ids) == 2:
            kind_a = wires[wire_ids[0]]["kind"]
            kind_b = wires[wire_ids[1]]["kind"]
            assert kind_a != kind_b, f"overlap at {cell} must be an orthogonal crossing"

    for wire in wires:
        colors = {
            int(g[r, c])
            for r, c in wire["cells"]
            if len(cell_to_wires[(r, c)]) == 1
        }
        if colors:
            assert len(colors) == 1, "non-crossing wire pixels must keep a single base color"
            color = next(iter(colors))
        else:
            sr, sc = wire["source"]
            color = int(g[sr, sc])
        wire["color"] = color
        wire["crossings"] = [
            cell for cell in wire["cells"] if len(cell_to_wires[cell]) == 2
        ]

    for cell, wire_ids in cell_to_wires.items():
        if len(wire_ids) != 2:
            continue
        visible = int(g[cell])
        colors = {int(wires[wire_ids[0]]["color"]), int(wires[wire_ids[1]]["color"])}
        assert visible in colors, "crossing must show one of the two original wire colors"

    return wires, cell_to_wires


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each orthogonal wire crossing with the mixed signal color implied by the grey-marked source directions.

    Input structure:
    - The background is black, and every non-grey foreground object is a one-pixel-thick horizontal or vertical wire.
    - Each wire has exactly one grey pixel adjacent to its source endpoint; the signal travels away from that grey marker.
    - Wires may cross orthogonally at shared cells, and an input crossing cell shows only one of the two base wire colors.

    Rule:
    - Reconstruct every straight wire by tracing from its grey-marked endpoint to its unmarked endpoint.
    - Along each wire, the carried signal starts as that wire's base color and updates only when it passes through a crossing.
    - At a crossing, mix the arriving horizontal and vertical signal colors with the visible table:
      red+blue->magenta, red+green->yellow, blue+green->azure, red+yellow->orange, all other pairs->maroon.
    - Keep all non-crossing wire pixels and all grey marker pixels unchanged, and replace each crossing cell with its mixed signal color.

    Key insight:
    - The grey marker fixes the source-to-sink order along each wire, so every crossing color is determined by upstream crossings on the same wire rather than by the wire's locally visible color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wires, cell_to_wires = _trace_wires(g)
    crossings = [cell for cell, ids in cell_to_wires.items() if len(ids) == 2]
    if not crossings:
        return g.copy()

    prev_on_wire: dict[tuple[int, tuple[int, int]], tuple[int, int] | None] = {}
    indegree = {cell: 0 for cell in crossings}
    forward: dict[tuple[int, int], set[tuple[int, int]]] = {cell: set() for cell in crossings}

    for wire_id, wire in enumerate(wires):
        ordered = list(wire["crossings"])
        for idx, cell in enumerate(ordered):
            prev = ordered[idx - 1] if idx else None
            prev_on_wire[(wire_id, cell)] = prev
            if prev is not None and cell not in forward[prev]:
                forward[prev].add(cell)
                indegree[cell] += 1

    ready = sorted(cell for cell, deg in indegree.items() if deg == 0)
    topo: list[tuple[int, int]] = []
    while ready:
        cell = ready.pop(0)
        topo.append(cell)
        for nxt in sorted(forward[cell]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()

    assert len(topo) == len(crossings), "crossing dependencies must be acyclic"

    crossing_color: dict[tuple[int, int], int] = {}
    for cell in topo:
        wire_ids = cell_to_wires[cell]
        h_id, v_id = sorted(
            wire_ids,
            key=lambda wire_id: 0 if wires[wire_id]["kind"] == "h" else 1,
        )
        assert wires[h_id]["kind"] == "h"
        assert wires[v_id]["kind"] == "v"

        prev_h = prev_on_wire[(h_id, cell)]
        prev_v = prev_on_wire[(v_id, cell)]
        color_h = int(wires[h_id]["color"]) if prev_h is None else crossing_color[prev_h]
        color_v = int(wires[v_id]["color"]) if prev_v is None else crossing_color[prev_v]
        crossing_color[cell] = _mix_colors(color_h, color_v)

    out = g.copy()
    for (r, c), color in crossing_color.items():
        out[r, c] = color
    return out


GEN_HEIGHT_RANGE = (12, 24)
GEN_WIDTH_RANGE = (12, 24)
GEN_WIRE_COUNT_RANGE = (2, 3)
GEN_MIN_GAP = 3
GEN_ATTEMPTS = 400
GEN_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
GEN_TEMPLATES = (
    ((RED, BLUE), (GREEN, RED)),
    ((RED, YELLOW), (BLUE, GREEN)),
    ((BLUE, YELLOW), (GREEN, RED)),
    ((RED, BLUE), (YELLOW, GREEN)),
)


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    for _ in range(120):
        picked = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _wire_cells(start: int, end: int) -> list[int]:
    step = 1 if end >= start else -1
    return list(range(start, end + step, step))


def _ordered_lists(
    rows: list[int],
    cols: list[int],
    corner: str,
) -> tuple[list[int], list[int], tuple[int, int], tuple[int, int]]:
    h_dir = (0, 1) if corner in ("top_left", "bottom_left") else (0, -1)
    v_dir = (1, 0) if corner in ("top_left", "top_right") else (-1, 0)
    ordered_rows = rows if v_dir[0] == 1 else list(reversed(rows))
    ordered_cols = cols if h_dir[1] == 1 else list(reversed(cols))
    return ordered_rows, ordered_cols, h_dir, v_dir


def _assign_wire_colors(
    rows: list[int],
    cols: list[int],
    corner: str,
) -> tuple[dict[int, int], dict[int, int]]:
    ordered_rows, ordered_cols, _h_dir, _v_dir = _ordered_lists(rows, cols, corner)
    template_h, template_v = random.choice(GEN_TEMPLATES)

    h_colors = list(template_h)
    v_colors = list(template_v)
    while len(h_colors) < len(rows):
        h_colors.append(random.choice(BASE_WIRE_COLORS))
    while len(v_colors) < len(cols):
        v_colors.append(random.choice(BASE_WIRE_COLORS))

    row_to_color = {row: int(h_colors[i]) for i, row in enumerate(ordered_rows)}
    col_to_color = {col: int(v_colors[i]) for i, col in enumerate(ordered_cols)}
    return row_to_color, col_to_color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    num_h = random.randint(*GEN_WIRE_COUNT_RANGE)
    num_v = random.randint(*GEN_WIRE_COUNT_RANGE)

    rows = _sample_spaced_values(2, h - 3, num_h, GEN_MIN_GAP)
    cols = _sample_spaced_values(2, w - 3, num_v, GEN_MIN_GAP)
    if rows is None or cols is None:
        return None

    corner = random.choice(GEN_CORNERS)
    row_to_color, col_to_color = _assign_wire_colors(rows, cols, corner)
    ordered_rows, ordered_cols, h_dir, v_dir = _ordered_lists(rows, cols, corner)

    assert ordered_rows[:2]
    assert ordered_cols[:2]

    grid = np.full((h, w), BLACK, dtype=int)
    vertical_cols = set(cols)
    horizontal_rows = set(rows)

    if h_dir[1] == 1:
        h_start = 1
        min_h_end = max(cols) + 1
        h_end_choices = list(range(min_h_end, w - 1))
        h_markers = {row: (row, 0) for row in rows}
    else:
        h_start = w - 2
        max_h_end = min(cols) - 1
        h_end_choices = list(range(1, max_h_end + 1))
        h_markers = {row: (row, w - 1) for row in rows}
    if not h_end_choices:
        return None

    if v_dir[0] == 1:
        v_start = 1
        min_v_end = max(rows) + 1
        v_end_choices = list(range(min_v_end, h - 1))
        v_markers = {col: (0, col) for col in cols}
    else:
        v_start = h - 2
        max_v_end = min(rows) - 1
        v_end_choices = list(range(1, max_v_end + 1))
        v_markers = {col: (h - 1, col) for col in cols}
    if not v_end_choices:
        return None

    h_ends = {row: random.choice(h_end_choices) for row in rows}
    v_ends = {col: random.choice(v_end_choices) for col in cols}

    for row in rows:
        mr, mc = h_markers[row]
        grid[mr, mc] = GREY
        for c in _wire_cells(h_start, h_ends[row]):
            if c in vertical_cols:
                continue
            grid[row, c] = row_to_color[row]

    for col in cols:
        mr, mc = v_markers[col]
        assert grid[mr, mc] == BLACK or grid[mr, mc] == GREY
        grid[mr, mc] = GREY
        for r in _wire_cells(v_start, v_ends[col]):
            if r in horizontal_rows:
                continue
            grid[r, col] = col_to_color[col]

    crossing_positions = [(row, col) for row in rows for col in cols]
    random.shuffle(crossing_positions)
    for row, col in crossing_positions:
        if random.random() < 0.5:
            grid[row, col] = row_to_color[row]
        else:
            grid[row, col] = col_to_color[col]

    if len(crossing_positions) < 4:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    mixed_colors = {int(out[r, c]) for r, c in crossing_positions}
    if len(mixed_colors) < 2:
        return None
    if not mixed_colors & {MAGENTA, YELLOW, AZURE, ORANGE}:
        return None
    if not any(int(out[r, c]) != int(grid[r, c]) for r, c in crossing_positions):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wire-mixing input")
