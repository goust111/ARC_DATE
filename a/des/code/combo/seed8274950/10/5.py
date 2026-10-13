from __future__ import annotations

import random

import numpy as np


GREY = 5
MIN_GRID = 13
MAX_GRID = 18
MAX_GENERATE_ATTEMPTS = 600
PALETTE = tuple(color for color in range(10) if color != GREY)

Coord = tuple[int, int]


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _line_cells(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise AssertionError("markers must align horizontally, vertically, or diagonally")
    step_r = _sign(dr)
    step_c = _sign(dc)
    length = max(abs(dr), abs(dc))
    return [(r0 + i * step_r, c0 + i * step_c) for i in range(length + 1)]


def _pair_key(a: Coord, b: Coord) -> tuple[int, int, int, int]:
    p, q = sorted((a, b))
    return (p[0], p[1], q[0], q[1])


def _marker_pairs(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    pairs: list[dict[str, object]] = []
    seen_colors: set[int] = set()
    for r, c in np.argwhere((g != bg) & (g != GREY)):
        color = int(g[r, c])
        if color in seen_colors:
            continue
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        assert len(coords) == 2, "each marker color must appear exactly twice"
        for mr, mc in coords:
            same_color_neighbors = [
                (nr, nc)
                for nr, nc in _neighbors4(mr, mc)
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == color
            ]
            assert not same_color_neighbors, "markers must be isolated single pixels"
        coords.sort()
        pairs.append({"color": color, "a": coords[0], "b": coords[1]})
        seen_colors.add(color)
    pairs.sort(key=lambda rec: _pair_key(rec["a"], rec["b"]))  # type: ignore[index]
    return pairs


def _grey_components(grid: np.ndarray) -> tuple[list[list[Coord]], dict[Coord, int]]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    components: list[list[Coord]] = []
    cell_to_component: dict[Coord, int] = {}

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or int(g[r, c]) != GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == GREY
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            idx = len(components)
            components.append(coords)
            for cell in coords:
                cell_to_component[cell] = idx
    return components, cell_to_component


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each pair of same-color markers with a horizontal, vertical, or 45-degree line.
    Every grey 4-connected object touched by that line is recolored to the marker color.
    The lines themselves stay invisible, and markers are erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    components, cell_to_component = _grey_components(g)
    pairs = _marker_pairs(g, bg)

    for rec in pairs:
        a = rec["a"]  # type: ignore[index]
        b = rec["b"]  # type: ignore[index]
        color = int(rec["color"])
        hit_components: set[int] = set()
        for cell in _line_cells(a, b):
            idx = cell_to_component.get(cell)
            if idx is not None:
                hit_components.add(idx)
        for idx in hit_components:
            for r, c in components[idx]:
                out[r, c] = color

    for rec in pairs:
        r0, c0 = rec["a"]  # type: ignore[index]
        r1, c1 = rec["b"]  # type: ignore[index]
        out[r0, c0] = bg
        out[r1, c1] = bg
    return out


def _king_expand(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    expanded.add((nr, nc))
    return expanded


def _random_shape(size: int) -> set[Coord] | None:
    cells: set[Coord] = {(0, 0)}
    while len(cells) < size:
        base_r, base_c = random.choice(tuple(cells))
        step_r, step_c = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((base_r + step_r, base_c + step_c))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if max(rows) - min(rows) > 3 or max(cols) - min(cols) > 3:
            return None
    return cells


def _shape_hits(cells: set[Coord], line_sets: list[set[Coord]]) -> set[int]:
    hits: set[int] = set()
    for idx, line_set in enumerate(line_sets):
        if cells & line_set:
            hits.add(idx)
    return hits


def _place_object(
    shape: tuple[int, int],
    anchor_candidates: list[Coord],
    allowed_hits: set[int],
    blocked: set[Coord],
    line_sets: list[set[Coord]],
) -> list[Coord] | None:
    candidates = anchor_candidates[:]
    random.shuffle(candidates)
    for anchor_r, anchor_c in candidates:
        for _ in range(60):
            size = random.randint(2, 5)
            rel_cells = _random_shape(size)
            if rel_cells is None:
                continue
            abs_cells = {(anchor_r + dr, anchor_c + dc) for dr, dc in rel_cells}
            if len(abs_cells) != len(rel_cells):
                continue
            if not all(_in_bounds(shape, r, c) for r, c in abs_cells):
                continue
            if abs_cells & blocked:
                continue
            if _shape_hits(abs_cells, line_sets) != allowed_hits:
                continue
            return sorted(abs_cells)
    return None


def _line_through_point(
    height: int,
    width: int,
    kind: str,
    center: Coord,
) -> dict[str, object] | None:
    cross_r, cross_c = center
    if kind == "H":
        if cross_c < 3 or cross_c > width - 4:
            return None
        left = random.randint(0, cross_c - 3)
        right = random.randint(cross_c + 3, width - 1)
        a, b = (cross_r, left), (cross_r, right)
    elif kind == "V":
        if cross_r < 3 or cross_r > height - 4:
            return None
        top = random.randint(0, cross_r - 3)
        bottom = random.randint(cross_r + 3, height - 1)
        a, b = (top, cross_c), (bottom, cross_c)
    else:
        diag_options: list[tuple[int, int, int]] = []
        up_left = min(cross_r, cross_c)
        down_right = min(height - 1 - cross_r, width - 1 - cross_c)
        if up_left >= 3 and down_right >= 3:
            diag_options.append((1, up_left, down_right))
        up_right = min(cross_r, width - 1 - cross_c)
        down_left = min(height - 1 - cross_r, cross_c)
        if up_right >= 3 and down_left >= 3:
            diag_options.append((-1, up_right, down_left))
        if not diag_options:
            return None
        slope, before_cap, after_cap = random.choice(diag_options)
        before = random.randint(3, before_cap)
        after = random.randint(3, after_cap)
        if slope == 1:
            a = (cross_r - before, cross_c - before)
            b = (cross_r + after, cross_c + after)
        else:
            a = (cross_r - before, cross_c + before)
            b = (cross_r + after, cross_c - after)

    return {"kind": kind, "a": a, "b": b, "cells": _line_cells(a, b)}


def _propose_cross_lines(
    height: int,
    width: int,
    shared_kinds: list[str],
) -> tuple[list[dict[str, object]], Coord] | None:
    for _ in range(80):
        cross_r = random.randint(3, height - 4)
        cross_c = random.randint(3, width - 4)
        center = (cross_r, cross_c)
        lines: list[dict[str, object]] = []
        for kind in shared_kinds:
            rec = _line_through_point(height, width, kind, center)
            if rec is None:
                lines = []
                break
            lines.append(rec)
        if len(lines) == len(shared_kinds):
            return lines, center
    return None


def _propose_line(
    height: int,
    width: int,
    kind: str,
    used_markers: set[Coord],
    used_line_cells: set[Coord],
    marker_blocked: set[Coord],
) -> dict[str, object] | None:
    for _ in range(240):
        if kind == "H":
            row = random.randint(0, height - 1)
            left, right = sorted(random.sample(range(width), 2))
            if right - left < 4:
                continue
            a, b = (row, left), (row, right)
        elif kind == "V":
            col = random.randint(0, width - 1)
            top, bottom = sorted(random.sample(range(height), 2))
            if bottom - top < 4:
                continue
            a, b = (top, col), (bottom, col)
        else:
            diag_len = random.randint(4, min(height, width) - 1)
            slope = random.choice((1, -1))
            top = random.randint(0, height - diag_len - 1)
            if slope == 1:
                left = random.randint(0, width - diag_len - 1)
                a, b = (top, left), (top + diag_len, left + diag_len)
            else:
                left = random.randint(diag_len, width - 1)
                a, b = (top, left), (top + diag_len, left - diag_len)

        cells = _line_cells(a, b)
        cell_set = set(cells)
        if a in used_markers or b in used_markers:
            continue
        if a in marker_blocked or b in marker_blocked:
            continue
        if cell_set & used_line_cells:
            continue
        if any(cell in used_markers for cell in cells):
            continue
        return {"kind": kind, "a": a, "b": b, "cells": cells}
    return None


def _line_kinds(pair_count: int, overlap_mode: bool) -> list[str]:
    if overlap_mode:
        kinds = random.sample(["H", "V", "D"], 2)
        if pair_count >= 3:
            kinds.append(next(kind for kind in ["H", "V", "D"] if kind not in kinds))
        while len(kinds) < pair_count:
            kinds.append(random.choice(("H", "V", "D")))
        return kinds

    if pair_count == 2:
        kinds = random.sample(["H", "V", "D"], 2)
    elif pair_count == 3:
        kinds = ["H", "V", "D"]
    else:
        kinds = ["H", "V", "D", random.choice(("H", "V", "D"))]
    random.shuffle(kinds)
    return kinds


def _order_positions(lines: list[dict[str, object]]) -> dict[int, int]:
    ordered = sorted(
        range(len(lines)),
        key=lambda idx: _pair_key(lines[idx]["a"], lines[idx]["b"]),  # type: ignore[index]
    )
    return {line_idx: pos for pos, line_idx in enumerate(ordered)}


def _render_case(
    shape: tuple[int, int],
    bg: int,
    lines: list[dict[str, object]],
    objects: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    expected = np.full(shape, bg, dtype=int)

    for obj in objects:
        for r, c in obj["cells"]:  # type: ignore[index]
            inp[r, c] = GREY

    for line in lines:
        color = int(line["color"])
        r0, c0 = line["a"]  # type: ignore[index]
        r1, c1 = line["b"]  # type: ignore[index]
        inp[r0, c0] = color
        inp[r1, c1] = color

    order_pos = _order_positions(lines)
    for obj in objects:
        obj_hits = obj["hits"]  # type: ignore[index]
        final_color = GREY
        if obj_hits:
            last_hit = max(obj_hits, key=lambda idx: order_pos[idx])
            final_color = int(lines[last_hit]["color"])
        for r, c in obj["cells"]:  # type: ignore[index]
            expected[r, c] = final_color

    return inp, expected


def _build_attempt() -> np.ndarray | None:
    height = random.randint(MIN_GRID, MAX_GRID)
    width = random.randint(MIN_GRID, MAX_GRID)
    bg = random.choice(PALETTE)
    if bg == GREY:
        return None

    pair_count = random.randint(2, 4)
    overlap_mode = random.random() < 0.5
    kinds = _line_kinds(pair_count, overlap_mode)

    line_records: list[dict[str, object]] = []
    used_markers: set[Coord] = set()
    used_line_cells: set[Coord] = set()
    marker_blocked: set[Coord] = set()
    shared_cell: Coord | None = None

    if overlap_mode:
        cross = _propose_cross_lines(height, width, kinds[:2])
        if cross is None:
            return None
        cross_lines, shared_cell = cross
        line_records.extend(cross_lines)
        for rec in cross_lines:
            used_markers.add(rec["a"])  # type: ignore[arg-type]
            used_markers.add(rec["b"])  # type: ignore[arg-type]
            used_line_cells.update(rec["cells"])  # type: ignore[arg-type]
        marker_blocked = _king_expand(used_markers, (height, width))

    extra_kinds = kinds[len(line_records) :]
    for kind in extra_kinds:
        rec = _propose_line(height, width, kind, used_markers, used_line_cells, marker_blocked)
        if rec is None:
            return None
        line_records.append(rec)
        used_markers.add(rec["a"])  # type: ignore[arg-type]
        used_markers.add(rec["b"])  # type: ignore[arg-type]
        used_line_cells.update(rec["cells"])  # type: ignore[arg-type]
        marker_blocked = _king_expand(used_markers, (height, width))

    color_pool = [color for color in PALETTE if color != bg]
    if len(color_pool) < len(line_records):
        return None
    for rec, color in zip(line_records, random.sample(color_pool, len(line_records))):
        rec["color"] = color

    line_sets = [set(rec["cells"]) for rec in line_records]  # type: ignore[arg-type]
    blocked = _king_expand(used_markers, (height, width))
    objects: list[dict[str, object]] = []

    if shared_cell is not None:
        shared_obj = _place_object(
            (height, width),
            [shared_cell],
            {0, 1},
            blocked,
            line_sets,
        )
        if shared_obj is None:
            return None
        shared_cells = set(shared_obj)
        objects.append({"cells": shared_obj, "hits": {0, 1}})
        blocked |= _king_expand(shared_cells, (height, width))

    for idx, rec in enumerate(line_records):
        candidates = [cell for cell in rec["cells"][1:-1] if cell != shared_cell]  # type: ignore[index]
        obj = _place_object((height, width), candidates, {idx}, blocked, line_sets)
        if obj is None:
            return None
        obj_cells = set(obj)
        objects.append({"cells": obj, "hits": {idx}})
        blocked |= _king_expand(obj_cells, (height, width))

    untouched_count = random.randint(1, 3)
    off_line_cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in used_line_cells
    ]
    for _ in range(untouched_count):
        obj = _place_object((height, width), off_line_cells, set(), blocked, line_sets)
        if obj is None:
            return None
        obj_cells = set(obj)
        objects.append({"cells": obj, "hits": set()})
        blocked |= _king_expand(obj_cells, (height, width))

    if not any(obj["hits"] for obj in objects):
        return None
    if not any(not obj["hits"] for obj in objects):
        return None
    if overlap_mode and not any(len(obj["hits"]) > 1 for obj in objects):
        return None
    if (not overlap_mode) and any(len(obj["hits"]) > 1 for obj in objects):
        return None

    inp, expected = _render_case((height, width), bg, line_records, objects)
    counts = np.bincount(inp.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return None
    if len(_marker_pairs(inp, bg)) != len(line_records):
        return None
    grey_components, _ = _grey_components(inp)
    if len(grey_components) != len(objects):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
