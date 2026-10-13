from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _true_runs(flags: list[bool] | np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags):
        if bool(flag):
            if start is None:
                start = int(idx)
        elif start is not None:
            runs.append((start, int(idx - 1)))
            start = None
    if start is not None:
        runs.append((start, int(len(flags) - 1)))
    return runs


def _graph_components(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj = [set() for _ in range(n_nodes)]
    for a, b in edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in range(n_nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(int(node))
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(int(nxt))
                    stack.append(int(nxt))
        comps.append(comp)
    return comps


def _parse_rooms(grid: np.ndarray) -> tuple[int, list[dict[str, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    fg = np.argwhere(g != bg)
    assert fg.size > 0, "expected at least one non-background wall pixel"

    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, sub.shape)

    row_is_wall = (np.count_nonzero(sub != bg, axis=1) * 2) > w
    col_is_wall = (np.count_nonzero(sub != bg, axis=0) * 2) > h
    row_bands = _true_runs(row_is_wall.tolist())
    col_bands = _true_runs(col_is_wall.tolist())

    assert row_bands and col_bands, "room lattice walls were not detected"
    assert row_bands[0][0] == 0 and row_bands[-1][1] == h - 1
    assert col_bands[0][0] == 0 and col_bands[-1][1] == w - 1

    n_room_rows = len(row_bands) - 1
    n_room_cols = len(col_bands) - 1
    assert n_room_rows >= 1 and n_room_cols >= 1

    rooms: list[dict[str, int]] = []
    room_id_by_cell: dict[tuple[int, int], int] = {}
    room_id = 0
    for rr in range(n_room_rows):
        top_border = int(row_bands[rr][1])
        bottom_border = int(row_bands[rr + 1][0])
        assert bottom_border - top_border >= 2, "room interior must be non-empty"
        ir0 = top_border + 1
        ir1 = bottom_border - 1

        for cc in range(n_room_cols):
            left_border = int(col_bands[cc][1])
            right_border = int(col_bands[cc + 1][0])
            assert right_border - left_border >= 2, "room interior must be non-empty"
            ic0 = left_border + 1
            ic1 = right_border - 1

            border_vals: list[int] = []
            border_vals.extend(int(v) for v in sub[top_border, left_border : right_border + 1].tolist())
            border_vals.extend(int(v) for v in sub[bottom_border, left_border : right_border + 1].tolist())
            border_vals.extend(int(v) for v in sub[top_border : bottom_border + 1, left_border].tolist())
            border_vals.extend(int(v) for v in sub[top_border : bottom_border + 1, right_border].tolist())
            border_vals = [v for v in border_vals if v != bg]
            assert border_vals, "room border color must remain visible despite doors"
            color = int(Counter(border_vals).most_common(1)[0][0])

            rooms.append(
                {
                    "room_row": int(rr),
                    "room_col": int(cc),
                    "top_border": int(r0 + top_border),
                    "bottom_border": int(r0 + bottom_border),
                    "left_border": int(c0 + left_border),
                    "right_border": int(c0 + right_border),
                    "ir0": int(r0 + ir0),
                    "ir1": int(r0 + ir1),
                    "ic0": int(c0 + ic0),
                    "ic1": int(c0 + ic1),
                    "area": int((ir1 - ir0 + 1) * (ic1 - ic0 + 1)),
                    "color": color,
                }
            )
            room_id_by_cell[(rr, cc)] = int(room_id)
            room_id += 1

    door_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for rr in range(n_room_rows):
        rel_ir0 = int(row_bands[rr][1] + 1)
        rel_ir1 = int(row_bands[rr + 1][0] - 1)
        for cc in range(n_room_cols - 1):
            band_c0, band_c1 = col_bands[cc + 1]
            cells: list[tuple[int, int]] = []
            for rel_r in range(rel_ir0, rel_ir1 + 1):
                if np.all(sub[rel_r, band_c0 : band_c1 + 1] == bg):
                    for rel_c in range(int(band_c0), int(band_c1) + 1):
                        cells.append((int(r0 + rel_r), int(c0 + rel_c)))
            if cells:
                a = room_id_by_cell[(rr, cc)]
                b = room_id_by_cell[(rr, cc + 1)]
                door_cells[tuple(sorted((a, b)))] = cells

    for rr in range(n_room_rows - 1):
        band_r0, band_r1 = row_bands[rr + 1]
        for cc in range(n_room_cols):
            rel_ic0 = int(col_bands[cc][1] + 1)
            rel_ic1 = int(col_bands[cc + 1][0] - 1)
            cells = []
            for rel_c in range(rel_ic0, rel_ic1 + 1):
                if np.all(sub[band_r0 : band_r1 + 1, rel_c] == bg):
                    for rel_r in range(int(band_r0), int(band_r1) + 1):
                        cells.append((int(r0 + rel_r), int(c0 + rel_c)))
            if cells:
                a = room_id_by_cell[(rr, cc)]
                b = room_id_by_cell[(rr + 1, cc)]
                door_cells[tuple(sorted((a, b)))] = cells

    return bg, rooms, door_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect rectangular rooms in a dense colored wall lattice, connect rooms through
    bg door gaps on shared walls, and fill each connected group with the border
    color of its largest room. Isolated rooms remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, rooms, door_cells = _parse_rooms(g)
    out = g.copy()

    edges = [tuple(edge) for edge in door_cells]
    comps = _graph_components(len(rooms), edges)

    for comp in comps:
        if len(comp) <= 1:
            continue
        leader = max(comp, key=lambda idx: rooms[idx]["area"])
        fill_color = int(rooms[leader]["color"])
        comp_set = set(comp)

        for idx in comp:
            room = rooms[idx]
            out[room["ir0"] : room["ir1"] + 1, room["ic0"] : room["ic1"] + 1] = fill_color

        for edge, cells in door_cells.items():
            if edge[0] in comp_set and edge[1] in comp_set:
                for r, c in cells:
                    out[int(r), int(c)] = fill_color

    return out


def _build_axis(lengths: list[int], start: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    bands = [(int(start), int(start))]
    spans: list[tuple[int, int]] = []
    cursor = int(start + 1)
    for idx, length in enumerate(lengths):
        spans.append((int(cursor), int(cursor + length - 1)))
        cursor += int(length)
        if idx < len(lengths) - 1:
            bands.append((int(cursor), int(cursor + 1)))
            cursor += 2
    bands.append((int(cursor), int(cursor)))
    return bands, spans, int(cursor - start + 1)


def _room_index(room_row: int, room_col: int, n_cols: int) -> int:
    return int(room_row * n_cols + room_col)


def _door_graph_edges(
    n_room_rows: int,
    n_room_cols: int,
    require_chain: bool,
) -> list[tuple[str, int, int]]:
    possible: list[tuple[str, int, int]] = []
    for rr in range(n_room_rows):
        for cc in range(n_room_cols - 1):
            possible.append(("h", int(rr), int(cc)))
    for rr in range(n_room_rows - 1):
        for cc in range(n_room_cols):
            possible.append(("v", int(rr), int(cc)))

    for _ in range(200):
        chosen = [edge for edge in possible if random.random() < 0.45]
        room_edges: list[tuple[int, int]] = []
        for orient, rr, cc in chosen:
            if orient == "h":
                a = _room_index(rr, cc, n_room_cols)
                b = _room_index(rr, cc + 1, n_room_cols)
            else:
                a = _room_index(rr, cc, n_room_cols)
                b = _room_index(rr + 1, cc, n_room_cols)
            room_edges.append((int(a), int(b)))

        comps = _graph_components(n_room_rows * n_room_cols, room_edges)
        sizes = sorted((len(comp) for comp in comps), reverse=True)
        if not any(size > 1 for size in sizes):
            continue
        if not any(size == 1 for size in sizes):
            continue
        if require_chain and max(sizes) < 3:
            continue
        return chosen

    raise RuntimeError("failed to sample a door graph with both connected and isolated rooms")


def _areas_have_unique_group_max(
    areas: list[int],
    edges: list[tuple[int, int]],
) -> bool:
    for comp in _graph_components(len(areas), edges):
        if len(comp) <= 1:
            continue
        vals = sorted((areas[idx] for idx in comp), reverse=True)
        if len(vals) >= 2 and vals[0] == vals[1]:
            return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = int(random.randrange(10))
    n_room_rows = int(random.randint(2, 4))
    n_room_cols = int(random.randint(2, 4))
    if n_room_rows * n_room_cols < 4:
        return None

    room_heights = [int(random.randint(2, 5)) for _ in range(n_room_rows)]
    room_widths = [int(random.randint(2, 5)) for _ in range(n_room_cols)]

    top_margin = int(random.randint(1, 4))
    bottom_margin = int(random.randint(1, 4))
    left_margin = int(random.randint(1, 4))
    right_margin = int(random.randint(1, 4))

    row_bands, row_spans, complex_h = _build_axis(room_heights, top_margin)
    col_bands, col_spans, complex_w = _build_axis(room_widths, left_margin)
    height = int(top_margin + complex_h + bottom_margin)
    width = int(left_margin + complex_w + right_margin)

    grid = np.full((height, width), bg, dtype=int)
    n_rooms = n_room_rows * n_room_cols

    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    active = palette[: random.randint(3, min(6, len(palette)))]
    room_colors = [int(random.choice(active)) for _ in range(n_rooms)]
    if len(set(room_colors)) < 2:
        return None

    rooms: list[dict[str, int]] = []
    for rr in range(n_room_rows):
        top_border = int(row_bands[rr][1])
        bottom_border = int(row_bands[rr + 1][0])
        ir0, ir1 = row_spans[rr]
        for cc in range(n_room_cols):
            left_border = int(col_bands[cc][1])
            right_border = int(col_bands[cc + 1][0])
            ic0, ic1 = col_spans[cc]
            rid = _room_index(rr, cc, n_room_cols)
            color = int(room_colors[rid])
            grid[top_border, left_border : right_border + 1] = color
            grid[bottom_border, left_border : right_border + 1] = color
            grid[top_border : bottom_border + 1, left_border] = color
            grid[top_border : bottom_border + 1, right_border] = color
            rooms.append(
                {
                    "ir0": int(ir0),
                    "ir1": int(ir1),
                    "ic0": int(ic0),
                    "ic1": int(ic1),
                    "area": int((ir1 - ir0 + 1) * (ic1 - ic0 + 1)),
                    "color": color,
                }
            )

    require_chain = bool(random.random() < 0.5)
    chosen_doors = _door_graph_edges(n_room_rows, n_room_cols, require_chain=require_chain)
    room_edges: list[tuple[int, int]] = []
    door_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for orient, rr, cc in chosen_doors:
        if orient == "h":
            band_c0, band_c1 = col_bands[cc + 1]
            door_r = int(random.randint(row_spans[rr][0], row_spans[rr][1]))
            cells = [(int(door_r), int(col)) for col in range(int(band_c0), int(band_c1) + 1)]
            a = _room_index(rr, cc, n_room_cols)
            b = _room_index(rr, cc + 1, n_room_cols)
        else:
            band_r0, band_r1 = row_bands[rr + 1]
            door_c = int(random.randint(col_spans[cc][0], col_spans[cc][1]))
            cells = [(int(row), int(door_c)) for row in range(int(band_r0), int(band_r1) + 1)]
            a = _room_index(rr, cc, n_room_cols)
            b = _room_index(rr + 1, cc, n_room_cols)

        for r, c in cells:
            grid[r, c] = bg
        room_edges.append((int(a), int(b)))
        door_cells[tuple(sorted((a, b)))] = cells

    areas = [int(room["area"]) for room in rooms]
    if not _areas_have_unique_group_max(areas, room_edges):
        return None

    comps = _graph_components(n_rooms, room_edges)
    if not any(len(comp) > 1 and len({room_colors[idx] for idx in comp}) >= 2 for comp in comps):
        return None

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return None
    ranked = sorted(int(v) for v in counts.tolist())[::-1]
    if len(ranked) >= 2 and ranked[0] == ranked[1]:
        return None

    expected = grid.copy()
    for comp in comps:
        if len(comp) <= 1:
            continue
        leader = max(comp, key=lambda idx: rooms[idx]["area"])
        fill_color = int(rooms[leader]["color"])
        comp_set = set(comp)
        for idx in comp:
            room = rooms[idx]
            expected[room["ir0"] : room["ir1"] + 1, room["ic0"] : room["ic1"] + 1] = fill_color
        for edge, cells in door_cells.items():
            if edge[0] in comp_set and edge[1] in comp_set:
                for r, c in cells:
                    expected[int(r), int(c)] = fill_color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_case()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("generate_input failed to build a valid room-door puzzle")
