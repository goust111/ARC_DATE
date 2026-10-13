import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PALETTE_POOL = [1, 2, 3, 6, 7, 8, 9]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _edge_distance(p: tuple[int, int], h: int, w: int) -> int:
    r, c = p
    return min(r, c, h - 1 - r, w - 1 - c)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_palette(row: np.ndarray) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    cur: list[int] = []
    for v in map(int, row):
        if v in (BG, GREY):
            if len(cur) >= 2:
                mapping[cur[0]] = cur[:]
            cur = []
        else:
            cur.append(v)
    if len(cur) >= 2:
        mapping[cur[0]] = cur[:]
    return mapping


def _order_path(cells: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    pts = set(cells)
    deg = {
        p: sum((nr, nc) in pts for nr, nc in _neighbors4(*p))
        for p in pts
    }
    endpoints = [p for p, d in deg.items() if d == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    start = min(endpoints, key=lambda p: (_edge_distance(p, h, w), p[0], p[1]))
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = []
        for nr, nc in _neighbors4(*cur):
            if (nr, nc) in pts and (nr, nc) != prev:
                nxts.append((nr, nc))
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(cells), "path traversal must cover all cells"
    return ordered


def _turn_cells(path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    turns = set()
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns.add(path[i])
    return turns


def _switch_turns(grid: np.ndarray, path: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = grid.shape
    turns = _turn_cells(path)
    switched = set()
    for r, c in turns:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h - 1 and 0 <= nc < w and int(grid[nr, nc]) == YELLOW:
                switched.add((r, c))
                break
    return switched


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    palette = _parse_palette(g[-1, :])
    out = g.copy()
    out[:-1, :][g[:-1, :] == YELLOW] = BG

    upper = g[:-1, :]
    path_colors = sorted(set(int(v) for v in np.unique(upper)) - {BG, YELLOW, GREY})
    for color in path_colors:
        assert color in palette, "every path color must appear at the start of a palette bar"
        for comp in _components(upper == color):
            ordered = _order_path(comp, h, w)
            seq = palette[color]
            switched = _switch_turns(g, ordered)
            state = 0
            for cell in ordered:
                if cell in switched:
                    state = (state + 1) % len(seq)
                out[cell] = seq[state]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orthogonal path by advancing through its palette whenever it passes a marked turning point.

    Input structure:
    - bg = black background.
    - The grid contains 2-3 separate one-cell-wide orthogonal paths, each drawn in a single color.
    - Some turning points have an adjacent yellow marker pixel outside the bend; these markers are switches.
    - The bottom row is a palette made of short color bars separated by grey pixels, and each bar starts with one path's original color.

    Rule:
    - Match every path to the palette bar whose first color equals the path color.
    - Order each path from the endpoint closer to the grid edge toward the other endpoint.
    - Traverse the ordered path with the palette state initially at the first bar color; whenever a switched turning point is reached, advance cyclically to the next palette color and recolor that turning point and all later cells until the next switch with the new state color.
    - Remove the yellow switch markers and keep the palette row unchanged.

    Key insight:
    The puzzle combines three independent inferences: identify which palette belongs to which path, infer the traversal direction from edge proximity, and then simulate a cyclic state machine that changes color only at marked bends.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


def _polyline_cells(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(vertices) - 1):
        r1, c1 = vertices[i]
        r2, c2 = vertices[i + 1]
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            seg = [(r1, c) for c in range(c1, c2 + step, step)]
        else:
            step = 1 if r2 > r1 else -1
            seg = [(r, c1) for r in range(r1, r2 + step, step)]
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _random_parts(total: int, parts: int, min_part: int) -> list[int]:
    vals = [min_part] * parts
    extra = total - parts * min_part
    for _ in range(extra):
        vals[random.randrange(parts)] += 1
    random.shuffle(vals)
    return vals


def _choose_switch_cells(
    h: int,
    w: int,
    path: list[tuple[int, int]],
    switch_turns: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    path_set = set(path)
    path_index = {p: i for i, p in enumerate(path)}
    reserved = set(occupied) | path_set
    chosen: list[tuple[int, int]] = []
    for turn in switch_turns:
        i = path_index[turn]
        assert 0 < i < len(path) - 1
        candidates = []
        for nr, nc in _neighbors4(*turn):
            if not (0 <= nr < h - 1 and 0 <= nc < w):
                continue
            if (nr, nc) in reserved:
                continue
            nbrs = {(rr, cc) for rr, cc in _neighbors4(nr, nc) if (rr, cc) in path_set}
            if nbrs != {turn}:
                continue
            if any((rr, cc) in occupied for rr, cc in _neighbors4(nr, nc)):
                continue
            if any((rr, cc) in chosen for rr, cc in _neighbors4(nr, nc)):
                continue
            candidates.append((nr, nc))
        if not candidates:
            return None
        pick = random.choice(candidates)
        reserved.add(pick)
        chosen.append(pick)
    return chosen


def _build_horizontal_spec(
    h: int,
    w: int,
    band_top: int,
    band_h: int,
    occupied: set[tuple[int, int]],
):
    row_levels = random.choice([2, 3])
    rows = random.sample(list(range(band_top + 1, band_top + band_h - 1)), k=row_levels)
    side = random.choice(["left", "right"])
    start_c = random.choice([0, 1]) if side == "left" else w - 1 - random.choice([0, 1])
    far_margin = random.randint(3, 5)
    max_total = (w - 1 - far_margin - start_c) if side == "left" else (start_c - far_margin)
    min_total = row_levels * 3
    if max_total < min_total:
        return None
    total = random.randint(min_total, max_total)
    lens = _random_parts(total, row_levels, 3)

    cols = [start_c]
    direction = 1 if side == "left" else -1
    for length in lens:
        cols.append(cols[-1] + direction * length)

    if row_levels == 2:
        vertices = [(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[1]), (rows[1], cols[2])]
    else:
        vertices = [
            (rows[0], cols[0]),
            (rows[0], cols[1]),
            (rows[1], cols[1]),
            (rows[1], cols[2]),
            (rows[2], cols[2]),
            (rows[2], cols[3]),
        ]

    path = _polyline_cells(vertices)
    if len(path) != len(set(path)):
        return None
    if any(not (0 <= r < h - 1 and 0 <= c < w) for r, c in path):
        return None
    if any((r, c) in occupied for r, c in path):
        return None
    if _edge_distance(path[0], h, w) >= _edge_distance(path[-1], h, w):
        return None

    turns = vertices[1:-1]
    seq_len = random.choice([2, 3])
    min_switches = 2 if seq_len == 3 else 1
    if len(turns) < min_switches:
        return None
    switch_count = random.randint(min_switches, len(turns))
    switch_turns = random.sample(turns, k=switch_count)
    switch_cells = _choose_switch_cells(h, w, path, switch_turns, occupied)
    if switch_cells is None:
        return None

    return {
        "path": path,
        "turns": turns,
        "switch_turns": switch_turns,
        "switch_cells": switch_cells,
        "seq_len": seq_len,
    }


def _build_vertical_spec(
    h: int,
    w: int,
    band_left: int,
    band_w: int,
    occupied: set[tuple[int, int]],
):
    col_levels = random.choice([2, 3])
    cols = random.sample(list(range(band_left + 1, band_left + band_w - 1)), k=col_levels)
    side = random.choice(["top", "bottom"])
    if side == "top":
        start_r = random.choice([0, 1])
        direction = 1
        far_margin = random.randint(4, 6)
        max_total = (h - 3 - far_margin - start_r)
    else:
        start_r = h - 3 - random.choice([0, 1])
        direction = -1
        far_margin = random.randint(2, 4)
        max_total = start_r - far_margin
    min_total = col_levels * 3
    if max_total < min_total:
        return None
    total = random.randint(min_total, max_total)
    lens = _random_parts(total, col_levels, 3)

    rows = [start_r]
    for length in lens:
        rows.append(rows[-1] + direction * length)

    if col_levels == 2:
        vertices = [(rows[0], cols[0]), (rows[1], cols[0]), (rows[1], cols[1]), (rows[2], cols[1])]
    else:
        vertices = [
            (rows[0], cols[0]),
            (rows[1], cols[0]),
            (rows[1], cols[1]),
            (rows[2], cols[1]),
            (rows[2], cols[2]),
            (rows[3], cols[2]),
        ]

    path = _polyline_cells(vertices)
    if len(path) != len(set(path)):
        return None
    if any(not (0 <= r < h - 1 and 0 <= c < w) for r, c in path):
        return None
    if any((r, c) in occupied for r, c in path):
        return None
    if _edge_distance(path[0], h, w) >= _edge_distance(path[-1], h, w):
        return None

    turns = vertices[1:-1]
    seq_len = random.choice([2, 3])
    min_switches = 2 if seq_len == 3 else 1
    if len(turns) < min_switches:
        return None
    switch_count = random.randint(min_switches, len(turns))
    switch_turns = random.sample(turns, k=switch_count)
    switch_cells = _choose_switch_cells(h, w, path, switch_turns, occupied)
    if switch_cells is None:
        return None

    return {
        "path": path,
        "turns": turns,
        "switch_turns": switch_turns,
        "switch_cells": switch_cells,
        "seq_len": seq_len,
    }


def _paint_palette(grid: np.ndarray, sequences: list[list[int]]) -> bool:
    h, w = grid.shape
    order = sequences[:]
    random.shuffle(order)
    total = sum(len(seq) for seq in order) + max(0, len(order) - 1)
    if total > w:
        return False
    start = random.randint(0, w - total)
    c = start
    for i, seq in enumerate(order):
        grid[h - 1, c : c + len(seq)] = np.array(seq, dtype=int)
        c += len(seq)
        if i != len(order) - 1:
            grid[h - 1, c] = GREY
            c += 1
    return True


def _validate_generated(grid: np.ndarray, expected_paths: int) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False
    palette = _parse_palette(grid[-1, :])
    upper = grid[:-1, :]
    path_colors = sorted(set(int(v) for v in np.unique(upper)) - {BG, YELLOW, GREY})
    if len(path_colors) != expected_paths or len(palette) != expected_paths:
        return False
    if not np.any(upper == YELLOW):
        return False
    multi_switch = False
    for color in path_colors:
        comps = _components(upper == color)
        if len(comps) != 1:
            return False
        if color not in palette:
            return False
        ordered = _order_path(comps[0], h, w)
        if ordered[0] == ordered[-1]:
            return False
        if _edge_distance(ordered[0], h, w) >= _edge_distance(ordered[-1], h, w):
            return False
        switched = _switch_turns(grid, ordered)
        if not switched or not switched.issubset(_turn_cells(ordered)):
            return False
        if len(switched) >= 2:
            multi_switch = True
    if not multi_switch:
        return False
    try:
        out = _apply_rule(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    path_count = random.choice([2, 3])
    layout = random.choice(["horizontal", "vertical"])

    if layout == "horizontal":
        band_h = random.choice([5, 6, 7])
        h = path_count * band_h + (path_count - 1) + 2
        w = random.randint(16, 27)
        grid = np.full((h, w), BG, dtype=int)
        band_starts = [i * (band_h + 1) for i in range(path_count)]
        builder = lambda start, occ: _build_horizontal_spec(h, w, start, band_h, occ)
    else:
        band_w = random.choice([5, 6, 7])
        extra_left = random.randint(0, 2)
        extra_right = random.randint(0, 2)
        w = extra_left + path_count * band_w + (path_count - 1) + extra_right
        h = random.randint(16, 27)
        if w > 30:
            return None
        grid = np.full((h, w), BG, dtype=int)
        band_starts = [extra_left + i * (band_w + 1) for i in range(path_count)]
        builder = lambda start, occ: _build_vertical_spec(h, w, start, band_w, occ)

    occupied: set[tuple[int, int]] = set()
    specs = []
    for start in band_starts:
        spec = builder(start, occupied)
        if spec is None:
            return None
        specs.append(spec)
        occupied.update(spec["path"])
        occupied.update(spec["switch_cells"])

    if max(len(spec["switch_turns"]) for spec in specs) < 2:
        return None

    path_colors = random.sample(PALETTE_POOL, k=path_count)
    sequences: list[list[int]] = []
    for spec, color in zip(specs, path_colors):
        extras = random.sample([v for v in PALETTE_POOL if v != color], k=spec["seq_len"] - 1)
        seq = [color] + extras
        spec["path_color"] = color
        spec["sequence"] = seq
        sequences.append(seq)
        for r, c in spec["path"]:
            grid[r, c] = color
        for r, c in spec["switch_cells"]:
            grid[r, c] = YELLOW

    if not _paint_palette(grid, sequences):
        return None
    if np.any(grid[-2, :] != BG):
        return None
    if not _validate_generated(grid, path_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
