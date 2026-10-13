from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WIRE = 7
COLUMN_COLORS = (1, 2, 3, 4)

GEN_RETRIES = 400
GRID_HEIGHT_RANGE = (15, 20)
GRID_WIDTH_RANGES = {
    4: (16, 22),
    5: (19, 25),
    6: (23, 28),
}
MIN_COLUMN_GAP = 3


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_active_columns(g: np.ndarray) -> list[int]:
    active = [int(c) for c in range(g.shape[1]) if int(g[-1, c]) not in (BG, WIRE)]
    assert 4 <= len(active) <= 6, "expected 4-6 bottom-row markers"
    return active


def _filled_column(g: np.ndarray, col: int) -> np.ndarray:
    h = g.shape[0]
    out = np.zeros(h, dtype=int)
    current = int(g[h - 1, col])
    assert current not in (BG, WIRE), "active columns must be bottom-marked"
    for r in range(h - 1, -1, -1):
        value = int(g[r, col])
        assert value != WIRE, "wire pixels cannot occupy active columns"
        if value != BG:
            current = value
        out[r] = current
    return out


def _wire_infos(g: np.ndarray, active_cols: list[int]) -> list[dict[str, object]]:
    h, w = g.shape
    active_set = set(active_cols)
    infos: list[dict[str, object]] = []
    for comp in _components(g == WIRE):
        cols = set()
        rows = {int(r) for r, _c in comp}
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if nc in active_set and int(g[nr, nc]) not in (BG, WIRE):
                    cols.add(int(nc))
        assert len(cols) == 2, "each wire must attach to exactly two active columns"
        infos.append(
            {
                "cells": tuple(comp),
                "rows": rows,
                "cols": tuple(sorted(cols)),
            }
        )
    return infos


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h > 0 and w > 0

    active_cols = _extract_active_columns(g)
    filled = {col: _filled_column(g, col) for col in active_cols}
    complexity = {col: len(set(map(int, filled[col]))) for col in active_cols}

    out = np.zeros_like(g)
    out[g == WIRE] = WIRE
    for col, values in filled.items():
        out[:, int(col)] = values

    for info in _wire_infos(g, active_cols):
        col_a, col_b = info["cols"]  # type: ignore[misc]
        da = int(complexity[int(col_a)])
        db = int(complexity[int(col_b)])
        assert da != db, "wire direction must be uniquely determined"
        src, dst = (int(col_a), int(col_b)) if da < db else (int(col_b), int(col_a))
        for row in info["rows"]:  # type: ignore[union-attr]
            out[int(row), dst] = int(filled[src][int(row)])

    out[g == WIRE] = WIRE
    out[-1, active_cols] = g[-1, active_cols]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill marked columns into color bands, then use wire-linked color diversity to direct row-wise transmission.

    Input structure:
    - A black background contains 4-6 bottom-row marker pixels whose columns are active.
    - Each active column has 2-4 additional colored pixels above the bottom marker, using only the same small palette.
    - Orange wire paths are 4-connected, one cell thick, unbranched, and each wire touches exactly two active columns at its ends.

    Rule:
    - Read the non-black bottom-row markers to identify the active columns.
    - In each active column, scan upward from the bottom marker; every black cell becomes the current color, and every colored cell updates that current color.
    - Count the number of distinct non-black colors in each filled active column.
    - For every orange wire connecting columns A and B, the column with fewer distinct colors sends to the one with more.
    - For each row occupied by that wire, copy the sender column's color at that row into the receiver column at the same row.
    - Keep the wires unchanged, preserve the bottom markers, and leave all non-active columns black.

    Key insight:
    The bottom markers define which sparse columns must first become full vertical color bands; only after that fill can the wire network use each column's color diversity to choose a transmission direction, and the transmitted payload itself is row-dependent band color.
    """
    return _solve(np.asarray(grid, dtype=int))


def _pattern_edges(count: int) -> list[tuple[int, int]]:
    if count == 4:
        patterns = [
            [(0, 1), (2, 3)],
            [(0, 1), (1, 2), (2, 3)],
        ]
    elif count == 5:
        patterns = [
            [(0, 1), (1, 2), (3, 4)],
            [(0, 1), (2, 3), (3, 4)],
        ]
    else:
        assert count == 6
        patterns = [[(0, 1), (2, 3), (4, 5)]]
    return random.choice(patterns)


def _sample_columns(count: int) -> tuple[int, list[int]]:
    w_lo, w_hi = GRID_WIDTH_RANGES[count]
    for _ in range(200):
        width = random.randint(w_lo, w_hi)
        cols = sorted(random.sample(range(1, width - 1), count))
        if all(cols[i + 1] - cols[i] >= MIN_COLUMN_GAP for i in range(count - 1)):
            return width, cols
    raise RuntimeError("failed to place active columns")


def _sample_complexities(count: int, edges: list[tuple[int, int]]) -> list[int]:
    neighbors = {i: [] for i in range(count)}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    for _ in range(500):
        values = [random.randint(1, 4) for _ in range(count)]
        if any(values[a] == values[b] for a, b in edges):
            continue
        if len(edges) >= 2 and len({values[a] < values[b] for a, b in edges}) < 2:
            continue
        if max(min(values[a], values[b]) for a, b in edges) < 2:
            continue
        if any(len(neighbors[i]) == 2 and all(values[i] > values[j] for j in neighbors[i]) for i in range(count)):
            continue
        return values
    raise RuntimeError("failed to sample non-degenerate complexities")


def _choose_corridors(height: int, wire_count: int) -> list[int]:
    candidates = list(range(1, (height // 2) - 1))
    assert len(candidates) >= wire_count
    return random.sample(candidates, wire_count)


def _choose_anchor_rows(height: int, count: int, edges: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    lower_rows = list(range((height // 2) + 1, height - 1))
    incident: dict[int, list[int]] = {i: [] for i in range(count)}
    for edge_idx, (a, b) in enumerate(edges):
        incident[a].append(edge_idx)
        incident[b].append(edge_idx)

    anchors: dict[tuple[int, int], int] = {}
    for col_idx, edge_ids in incident.items():
        need = len(edge_ids)
        assert 1 <= need <= 2
        for _ in range(200):
            picked = sorted(random.sample(lower_rows, need))
            if need == 2 and picked[1] - picked[0] < 2:
                continue
            for edge_id, row in zip(sorted(edge_ids), picked):
                anchors[(col_idx, edge_id)] = int(row)
            break
        else:
            raise RuntimeError("failed to choose anchor rows")
    return anchors


def _pick_transition_row(anchor: int, corridor: int, used_rows: set[int]) -> int:
    lo = min(anchor, corridor) + 1
    hi = max(anchor, corridor) - 1
    choices = [row for row in range(lo, hi + 1) if row not in used_rows]
    assert choices, "expected room for an internal color transition"
    return int(random.choice(choices))


def _assign_column_rows(
    height: int,
    count: int,
    edges: list[tuple[int, int]],
    complexities: list[int],
    corridors: list[int],
    anchors: dict[tuple[int, int], int],
) -> dict[int, list[int]]:
    incident: dict[int, list[int]] = {i: [] for i in range(count)}
    for edge_idx, (a, b) in enumerate(edges):
        incident[a].append(edge_idx)
        incident[b].append(edge_idx)

    rows_by_col: dict[int, list[int]] = {}
    for col_idx in range(count):
        required = {int(anchors[(col_idx, edge_id)]) for edge_id in incident[col_idx]}
        for edge_id in incident[col_idx]:
            a, b = edges[edge_id]
            other = b if a == col_idx else a
            if complexities[col_idx] < complexities[other] and complexities[col_idx] > 1:
                required.add(_pick_transition_row(anchors[(col_idx, edge_id)], corridors[edge_id], required))

        min_count = max(2, complexities[col_idx] - 1, len(required))
        seed_count = random.randint(min_count, 4)
        extras = [
            row
            for row in range(1, height - 1)
            if row not in required and row not in corridors
        ]
        random.shuffle(extras)
        while len(required) < seed_count:
            assert extras, "expected spare rows for extra seeds"
            required.add(int(extras.pop()))
        rows_by_col[col_idx] = sorted(required, reverse=True)

    return rows_by_col


def _assign_column_colors(seed_rows: list[int], complexity: int) -> tuple[int, dict[int, int]]:
    used_colors = random.sample(list(COLUMN_COLORS), complexity)
    marker = int(random.choice(used_colors))
    if complexity == 1:
        colors_bottom_to_top = [marker] * len(seed_rows)
    else:
        remaining = [color for color in used_colors if color != marker]
        random.shuffle(remaining)
        colors_bottom_to_top = remaining[:]
        prev = colors_bottom_to_top[-1] if colors_bottom_to_top else marker
        while len(colors_bottom_to_top) < len(seed_rows):
            choices = [color for color in used_colors if color != prev]
            chosen = int(random.choice(choices))
            colors_bottom_to_top.append(chosen)
            prev = chosen
    return marker, {row: int(color) for row, color in zip(seed_rows, colors_bottom_to_top)}


def _wire_cells(col_left: int, col_right: int, row_left: int, row_right: int, corridor: int) -> list[tuple[int, int]]:
    x1 = col_left + 1
    x2 = col_right - 1
    assert x1 < x2, "adjacent active columns need a visible gap"

    cells: list[tuple[int, int]] = [(int(row_left), int(x1))]
    step = -1 if corridor < row_left else 1
    for row in range(row_left + step, corridor + step, step):
        cells.append((int(row), int(x1)))

    for col in range(x1 + 1, x2 + 1):
        cells.append((int(corridor), int(col)))

    step = 1 if row_right > corridor else -1
    for row in range(corridor + step, row_right + step, step):
        cells.append((int(row), int(x2)))

    return cells


def _build_wire_metadata(
    height: int,
    width: int,
    cols: list[int],
    edges: list[tuple[int, int]],
    corridors: list[int],
    anchors: dict[tuple[int, int], int],
) -> list[dict[str, object]]:
    used: set[tuple[int, int]] = set()
    wires: list[dict[str, object]] = []
    for edge_idx, (left_idx, right_idx) in enumerate(edges):
        cells = _wire_cells(
            cols[left_idx],
            cols[right_idx],
            anchors[(left_idx, edge_idx)],
            anchors[(right_idx, edge_idx)],
            corridors[edge_idx],
        )
        cell_set = set(cells)
        assert len(cell_set) == len(cells)
        for r, c in cells:
            assert 0 <= r < height and 0 <= c < width
            assert (r, c) not in used
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in used:
                    raise RuntimeError("wires must stay separated")
        used.update(cell_set)
        wires.append(
            {
                "edge": (left_idx, right_idx),
                "rows": {int(r) for r, _ in cells},
                "cells": cells,
            }
        )
    return wires


def _render_input(
    height: int,
    width: int,
    cols: list[int],
    column_meta: dict[int, dict[str, object]],
    wires: list[dict[str, object]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for col_idx, col in enumerate(cols):
        meta = column_meta[col_idx]
        grid[height - 1, col] = int(meta["marker"])
        for row, color in meta["seeds"].items():  # type: ignore[union-attr]
            grid[int(row), col] = int(color)
    for wire in wires:
        for r, c in wire["cells"]:  # type: ignore[index]
            assert grid[r, c] == BG
            grid[r, c] = WIRE
    return grid


def _expected_from_metadata(
    height: int,
    width: int,
    cols: list[int],
    column_meta: dict[int, dict[str, object]],
    wires: list[dict[str, object]],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    filled: dict[int, np.ndarray] = {}
    for col_idx, col in enumerate(cols):
        meta = column_meta[col_idx]
        marker = int(meta["marker"])
        seeds = meta["seeds"]  # type: ignore[assignment]
        values = np.zeros(height, dtype=int)
        current = marker
        for row in range(height - 1, -1, -1):
            if row in seeds:
                current = int(seeds[row])
            values[row] = current
        filled[col_idx] = values
        out[:, col] = values

    for wire in wires:
        left_idx, right_idx = wire["edge"]  # type: ignore[misc]
        left_d = int(column_meta[left_idx]["complexity"])
        right_d = int(column_meta[right_idx]["complexity"])
        src_idx, dst_idx = (left_idx, right_idx) if left_d < right_d else (right_idx, left_idx)
        src_col = cols[src_idx]
        dst_col = cols[dst_idx]
        for row in wire["rows"]:  # type: ignore[union-attr]
            out[int(row), dst_col] = int(filled[src_idx][int(row)])
        for r, c in wire["cells"]:  # type: ignore[index]
            out[r, c] = WIRE

    for col_idx, col in enumerate(cols):
        out[height - 1, col] = int(column_meta[col_idx]["marker"])
    return out


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and 0 < h <= 30 and 0 < w <= 30
    assert set(map(int, np.unique(g))).issubset({BG, WIRE, *COLUMN_COLORS})
    assert not bool(np.any(g[-1] == WIRE))

    active_cols = _extract_active_columns(g)
    active_set = set(active_cols)

    non_wire_fg = np.argwhere((g != BG) & (g != WIRE))
    for r, c in non_wire_fg:
        assert int(c) in active_set, "only active columns may contain non-wire colors"

    seeds_by_col: dict[int, list[int]] = {}
    for col in active_cols:
        seed_rows = [int(r) for r in np.where((g[:-1, col] != BG) & (g[:-1, col] != WIRE))[0]]
        assert 2 <= len(seed_rows) <= 4
        seeds_by_col[col] = seed_rows

    filled = {col: _filled_column(g, col) for col in active_cols}
    complexity = {col: len(set(map(int, filled[col]))) for col in active_cols}
    wires = _wire_infos(g, active_cols)
    assert 2 <= len(wires) <= 3

    degree_by_col = {col: 0 for col in active_cols}
    targeted_rows = {col: set() for col in active_cols}
    varied_source_wire = False
    directions: list[bool] = []

    for wire in wires:
        cells = list(wire["cells"])  # type: ignore[arg-type]
        cell_set = set(cells)
        degs = []
        for r, c in cells:
            degs.append(sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c)))
        assert max(degs) <= 2 and degs.count(1) == 2

        col_a, col_b = wire["cols"]  # type: ignore[misc]
        degree_by_col[int(col_a)] += 1
        degree_by_col[int(col_b)] += 1
        assert complexity[int(col_a)] != complexity[int(col_b)]

        src, dst = (int(col_a), int(col_b)) if complexity[int(col_a)] < complexity[int(col_b)] else (int(col_b), int(col_a))
        rows = set(map(int, wire["rows"]))  # type: ignore[arg-type]
        assert any(int(g[row, src]) == BG for row in rows), "wire should rely on column fill, not only pre-existing pixels"
        if len({int(filled[src][row]) for row in rows}) > 1:
            varied_source_wire = True
        assert not (targeted_rows[dst] & rows), "independent wires may not race to the same target rows"
        targeted_rows[dst].update(rows)
        directions.append(src < dst)

    assert all(count >= 1 for count in degree_by_col.values())
    if len(wires) >= 2:
        assert len(set(directions)) == 2
    assert varied_source_wire


def _generate_candidate() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(*GRID_HEIGHT_RANGE)
    column_count = random.choice((4, 5, 6))
    width, cols = _sample_columns(column_count)
    edges = _pattern_edges(column_count)
    complexities = _sample_complexities(column_count, edges)
    corridors = _choose_corridors(height, len(edges))
    anchors = _choose_anchor_rows(height, column_count, edges)
    row_map = _assign_column_rows(height, column_count, edges, complexities, corridors, anchors)

    column_meta: dict[int, dict[str, object]] = {}
    for col_idx in range(column_count):
        marker, seeds = _assign_column_colors(row_map[col_idx], complexities[col_idx])
        column_meta[col_idx] = {
            "marker": marker,
            "seeds": seeds,
            "complexity": int(complexities[col_idx]),
        }

    wires = _build_wire_metadata(height, width, cols, edges, corridors, anchors)
    grid = _render_input(height, width, cols, column_meta, wires)
    expected = _expected_from_metadata(height, width, cols, column_meta, wires)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _generate_candidate()
            _validate_generated_grid(grid)
        except Exception:
            continue

        try:
            actual = main(grid)
        except Exception:
            continue

        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
