from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]


_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_in_scan_order(grid: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    seen = {int(bg)}
    for raw in grid.ravel():
        color = int(raw)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    assert connectivity in (4, 8)
    h, w = mask.shape
    dirs = _DIR4 if connectivity == 4 else _DIR8
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(visited[r, c]):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not bool(mask[nr, nc]) or bool(visited[nr, nc]):
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4_unbounded(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _DIR4]


def _path_endpoints(coords: np.ndarray) -> list[Coord]:
    cells = {tuple(map(int, rc)) for rc in coords}
    degrees = {
        cell: sum((nbr in cells) for nbr in _neighbors4_unbounded(cell))
        for cell in cells
    }
    assert all(deg in (1, 2) for deg in degrees.values()), "path fragment must be a simple 4-connected path"
    endpoints = sorted(cell for cell, deg in degrees.items() if deg == 1)
    assert len(endpoints) == 2, "path fragment must have exactly two endpoints"
    return endpoints


def _closest_endpoints(comp_a: np.ndarray, comp_b: np.ndarray) -> tuple[Coord, Coord]:
    ends_a = _path_endpoints(comp_a)
    ends_b = _path_endpoints(comp_b)
    pairs = []
    for a in ends_a:
        for b in ends_b:
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            pairs.append((dist, a, b))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    assert pairs and (len(pairs) == 1 or pairs[0][0] < pairs[1][0]), "nearest endpoint pair must be unique"
    return pairs[0][1], pairs[0][2]


def _decode_template(coords: np.ndarray) -> list[int]:
    shifted = np.asarray(coords, dtype=int) - np.min(coords, axis=0)
    col_to_rows: dict[int, set[int]] = {}
    for r, c in shifted:
        col_to_rows.setdefault(int(c), set()).add(int(r))

    cols = sorted(col_to_rows)
    assert len(cols) >= 2, "template must span at least two columns"
    assert cols == list(range(cols[0], cols[-1] + 1)), "template columns must be contiguous"

    rows = []
    for c in cols:
        assert len(col_to_rows[c]) == 1, "template must have exactly one pixel in each occupied column"
        rows.append(next(iter(col_to_rows[c])))

    seq = [rows[i] - rows[i - 1] for i in range(1, len(rows))]
    assert all(delta in (-1, 0, 1) for delta in seq), "template row deltas must be -1, 0, or 1"
    return seq


def _orient_bridge(a: Coord, b: Coord) -> tuple[str, Coord, Coord]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert dr or dc, "bridge endpoints must be distinct"
    if abs(dc) > abs(dr):
        return ("h", a, b) if a[1] <= b[1] else ("h", b, a)
    if abs(dr) > abs(dc):
        return ("v", a, b) if a[0] <= b[0] else ("v", b, a)
    raise AssertionError("main axis must be unambiguous")


def _bridge_cells(a: Coord, b: Coord, seq: list[int], axis: str) -> list[Coord]:
    assert seq, "template sequence must be non-empty"
    r, c = map(int, a)
    tr, tc = map(int, b)
    cells: list[Coord] = []
    step_idx = 0

    if axis == "h":
        assert c < tc, "horizontal bridge must run left to right"
        while c < tc and r != tr:
            delta = int(seq[step_idx % len(seq)])
            r += delta
            c += 1
            cells.append((r, c))
            step_idx += 1
        assert r == tr or c < tc, "horizontal bridge must align before exhausting columns"
        while c < tc:
            c += 1
            cells.append((r, c))
    else:
        assert axis == "v"
        assert r < tr, "vertical bridge must run top to bottom"
        while r < tr and c != tc:
            delta = int(seq[step_idx % len(seq)])
            r += 1
            c += delta
            cells.append((r, c))
            step_idx += 1
        assert c == tc or r < tr, "vertical bridge must align before exhausting rows"
        while r < tr:
            r += 1
            cells.append((r, c))

    assert (r, c) == (tr, tc), "bridge must end exactly on the target endpoint"
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group broken path fragments by color, decode the same-color bridge template at the bottom,
    and fill the gap by repeating the template's column-to-column offset pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 8 and w >= 8

    bg = _background_color(g)
    out = g.copy()

    for color in _colors_in_scan_order(g, bg):
        comps = _components(g == int(color), connectivity=8)
        assert len(comps) == 3, "each color must form exactly two path fragments and one template"

        comps.sort(key=lambda comp: (_bbox(comp)[2], _bbox(comp)[0], _bbox(comp)[1]))
        path_a, path_b, template = comps
        t_r0, _t_c0, _t_r1, _t_c1 = _bbox(template)
        assert all(_bbox(comp)[2] < t_r0 for comp in (path_a, path_b)), "template must be below both path fragments"

        seq = _decode_template(template)
        out[template[:, 0], template[:, 1]] = bg

        end_a, end_b = _closest_endpoints(path_a, path_b)
        axis, start, stop = _orient_bridge(end_a, end_b)
        bridge = _bridge_cells(start, stop, seq, axis)

        for r, c in bridge:
            assert 0 <= r < h and 0 <= c < w
            assert int(out[r, c]) in (bg, int(color)), "bridge may only pass through background or its own endpoint"
            out[r, c] = int(color)

    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (22, 30)
TEMPLATE_BAND_RANGE = (6, 7)
N_COLORS_RANGE = (2, 4)

TEMPLATE_LEN_RANGE = (2, 5)
MAX_TEMPLATE_HEIGHT = 5
ALIGNMENT_SCAN_STEPS = 6
MAX_SIDEWAYS_OFFSET = 3

TAIL_LEN_RANGE = (2, 4)
STRAIGHT_SUFFIX_RANGE = (1, 4)

PAIR_ATTEMPTS = 120
TEMPLATE_ATTEMPTS = 120
GEN_ATTEMPTS = 400


def _prefix_offsets(seq: list[int], steps: int) -> list[int]:
    cur = 0
    vals: list[int] = []
    for i in range(steps):
        cur += int(seq[i % len(seq)])
        vals.append(cur)
    return vals


def _template_rows(seq: list[int]) -> list[int]:
    rows = [0]
    cur = 0
    for delta in seq:
        cur += int(delta)
        rows.append(cur)
    base = min(rows)
    return [r - base for r in rows]


def _template_coords(seq: list[int], top: int, left: int) -> set[Coord]:
    rows = _template_rows(seq)
    return {(top + rows[c], left + c) for c in range(len(rows))}


def _sample_template_sequence() -> list[int] | None:
    for _ in range(80):
        length = random.randint(*TEMPLATE_LEN_RANGE)
        seq = [random.choice([-1, 0, 1]) for _ in range(length)]
        if not any(delta != 0 for delta in seq):
            continue
        rows = _template_rows(seq)
        if max(rows) + 1 > MAX_TEMPLATE_HEIGHT:
            continue
        return seq
    return None


def _alignment_options(seq: list[int]) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    seen = {0}
    cur = 0
    for step in range(1, ALIGNMENT_SCAN_STEPS + 1):
        cur += int(seq[(step - 1) % len(seq)])
        if cur not in seen and cur != 0 and abs(cur) <= MAX_SIDEWAYS_OFFSET:
            options.append((step, cur))
        seen.add(cur)
    return options


def _tail_from(endpoint: Coord, axis: str, direction: int, length: int) -> list[Coord]:
    r, c = map(int, endpoint)
    out = [(r, c)]
    for _ in range(length):
        if axis == "h":
            c += direction
        else:
            r += direction
        out.append((r, c))
    return out


def _intended_bridge(start: Coord, axis: str, seq: list[int], bend_steps: int, straight_steps: int) -> tuple[list[Coord], Coord]:
    r, c = map(int, start)
    cells: list[Coord] = []
    for i in range(bend_steps):
        delta = int(seq[i % len(seq)])
        if axis == "h":
            r += delta
            c += 1
        else:
            r += 1
            c += delta
        cells.append((r, c))
    for _ in range(straight_steps):
        if axis == "h":
            c += 1
        else:
            r += 1
        cells.append((r, c))
    return cells, (r, c)


def _cells_clear(existing: np.ndarray, cells: set[Coord]) -> bool:
    h, w = existing.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(existing[r0:r1, c0:c1].any()):
            return False
    return True


def _paint(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _sample_pair(
    existing: np.ndarray,
    play_h: int,
    full_w: int,
    color: int,
    axis: str,
    seq: list[int],
    bend_steps: int,
    target_offset: int,
    straight_steps: int,
) -> dict | None:
    top_tail = random.randint(*TAIL_LEN_RANGE)
    bottom_tail = random.randint(*TAIL_LEN_RANGE)
    bridge_prefix = _prefix_offsets(seq, bend_steps)
    bridge_total = bend_steps + straight_steps

    if axis == "h":
        row_offsets = [0] + bridge_prefix
        row_lo = 1 - min(row_offsets)
        row_hi = (play_h - 2) - max(row_offsets)
        col_lo = 1 + top_tail
        col_hi = (full_w - 2) - (bridge_total + bottom_tail)
        if row_lo > row_hi or col_lo > col_hi:
            return None
    else:
        col_offsets = [0] + bridge_prefix
        row_lo = 1 + top_tail
        row_hi = (play_h - 2) - (bridge_total + bottom_tail)
        col_lo = 1 - min(col_offsets)
        col_hi = (full_w - 2) - max(col_offsets)
        if row_lo > row_hi or col_lo > col_hi:
            return None

    for _ in range(PAIR_ATTEMPTS):
        if axis == "h":
            start = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
        else:
            start = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))

        bridge, stop = _intended_bridge(start, axis, seq, bend_steps, straight_steps)
        if axis == "h":
            assert stop[0] - start[0] == target_offset
        else:
            assert stop[1] - start[1] == target_offset

        comp_a = _tail_from(start, axis, -1, top_tail)
        comp_b = _tail_from(stop, axis, 1, bottom_tail)
        input_cells = set(comp_a) | set(comp_b)
        output_cells = input_cells | set(bridge)

        if not _cells_clear(existing, output_cells):
            continue
        if len(input_cells) != len(comp_a) + len(comp_b):
            continue

        return {
            "color": int(color),
            "axis": axis,
            "seq": list(seq),
            "input_cells": input_cells,
            "output_cells": output_cells,
        }
    return None


def _place_templates(existing: np.ndarray, plans: list[dict], template_top: int, band_h: int) -> bool:
    full_h, full_w = existing.shape
    for plan in plans:
        seq = plan["seq"]
        rows = _template_rows(seq)
        height = max(rows) + 1
        width = len(rows)
        row_lo = template_top
        row_hi = template_top + band_h - height
        col_lo = 1
        col_hi = full_w - width - 1
        if row_lo > row_hi or col_lo > col_hi:
            return False
        placed = False
        for _ in range(TEMPLATE_ATTEMPTS):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            cells = _template_coords(seq, r0, c0)
            if not _cells_clear(existing, cells):
                continue
            plan["template_cells"] = cells
            for r, c in cells:
                existing[r, c] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _render_input(h: int, w: int, plans: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for plan in plans:
        _paint(grid, plan["input_cells"], plan["color"])
        _paint(grid, plan["template_cells"], plan["color"])
    return grid


def _render_output(h: int, w: int, plans: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for plan in plans:
        _paint(grid, plan["output_cells"], plan["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    band_h = random.randint(*TEMPLATE_BAND_RANGE)
    play_h = h - band_h - 1
    template_top = play_h + 1
    if play_h < 12:
        return None

    n_colors = random.randint(*N_COLORS_RANGE)
    colors = random.sample(COLOR_POOL, n_colors)
    existing = np.zeros((h, w), dtype=bool)
    plans: list[dict] = []

    for color in colors:
        seq = _sample_template_sequence()
        if seq is None:
            return None
        options = _alignment_options(seq)
        if not options:
            return None
        bend_steps, target_offset = random.choice(options)
        straight_steps = random.randint(*STRAIGHT_SUFFIX_RANGE)
        axis = random.choice(["h", "v"])
        plan = _sample_pair(
            existing=existing,
            play_h=play_h,
            full_w=w,
            color=color,
            axis=axis,
            seq=seq,
            bend_steps=bend_steps,
            target_offset=target_offset,
            straight_steps=straight_steps,
        )
        if plan is None:
            return None
        for r, c in plan["output_cells"]:
            existing[r, c] = True
        plans.append(plan)

    if not _place_templates(existing, plans, template_top, band_h):
        return None

    inp = _render_input(h, w, plans)
    expected = _render_output(h, w, plans)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-template ARC input")
