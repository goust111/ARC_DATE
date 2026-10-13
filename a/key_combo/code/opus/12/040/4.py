from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, bg: int, ignore: set[tuple[int, int]] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    ignore = ignore or set()
    seen = np.zeros((H, W), dtype=bool)
    out: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or (r, c) in ignore or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or (nr, nc) in ignore or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            out.append({"color": color, "coords": coords})
    return out


def _trace_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    nbrs = {
        pt: sorted(npt for npt in _neighbors4(*pt) if npt in pts)
        for pt in pts
    }
    degrees = {pt: len(nlist) for pt, nlist in nbrs.items()}
    if any(d == 0 or d > 2 for d in degrees.values()):
        raise AssertionError("line object must be a 1-pixel-wide path")
    endpoints = sorted(pt for pt, d in degrees.items() if d == 1)
    if len(endpoints) != 2:
        raise AssertionError("line object must have exactly two endpoints")

    start = min(endpoints, key=lambda rc: (rc[0], rc[1]))
    path = [start]
    prev = None
    cur = start
    while True:
        choices = [npt for npt in nbrs[cur] if npt != prev]
        if not choices:
            break
        if len(choices) != 1:
            raise AssertionError("line object path must not branch")
        nxt = choices[0]
        path.append(nxt)
        prev, cur = cur, nxt
    if len(path) != len(coords):
        raise AssertionError("line object must be one simple path")
    return path


def _is_valid_line_object(coords: list[tuple[int, int]]) -> bool:
    if len(coords) < 2:
        return False
    pts = set(coords)
    for r, c in pts:
        if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= pts:
            return False

    try:
        path = _trace_path(coords)
    except AssertionError:
        return False

    drs = [path[i + 1][0] - path[i][0] for i in range(len(path) - 1)]
    dcs = [path[i + 1][1] - path[i][1] for i in range(len(path) - 1)]
    if not all(abs(dr) + abs(dc) == 1 for dr, dc in zip(drs, dcs)):
        return False

    if all(dr == 0 for dr in drs) and all(abs(dc) == 1 for dc in dcs):
        cols = [c for _, c in path]
        return max(cols) - min(cols) + 1 == len(cols)
    if all(dc == 0 for dc in dcs) and all(abs(dr) == 1 for dr in drs):
        rows = [r for r, _ in path]
        return max(rows) - min(rows) + 1 == len(rows)
    if all(dr in (0, 1) for dr in drs) and all(dc in (0, 1) for dc in dcs):
        return any(dr == 1 for dr in drs) and any(dc == 1 for dc in dcs)
    if all(dr in (0, 1) for dr in drs) and all(dc in (0, -1) for dc in dcs):
        return any(dr == 1 for dr in drs) and any(dc == -1 for dc in dcs)
    return False


def _corner_templates(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int], np.ndarray, set[tuple[int, int]]]]:
    H, W = grid.shape
    windows = [
        ((0, 0), grid[:2, :2]),
        ((0, W - 2), grid[:2, W - 2 : W]),
        ((H - 2, 0), grid[H - 2 : H, :2]),
        ((H - 2, W - 2), grid[H - 2 : H, W - 2 : W]),
    ]
    candidates = []
    for (r0, c0), sub in windows:
        flat = list(map(int, sub.flat))
        if bg in flat:
            continue
        if len(set(flat)) != 4:
            continue
        coords = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
        candidates.append(((r0, c0), np.asarray(sub, dtype=int).copy(), coords))
    return candidates


def _oracle_from_template(template: np.ndarray, n_segments: int) -> np.ndarray:
    blocks = []
    for k in range(n_segments):
        rotated = np.rot90(template, k=-k)
        blocks.append(np.kron(rotated, np.ones((2, 2), dtype=int)))
    return np.vstack(blocks)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the line objects by height order and use that order to choose template rotations.

    Input structure:
    - A black-background grid contains one 2x2 multicolor template flush in a grid corner.
    - Elsewhere there are three or four non-touching monochrome 4-connected 1-pixel-wide line objects.
    - Each line object has a unique topmost row, and template colors are disjoint from line colors.

    Rule:
    - Find the unique corner 2x2 template and remove it from consideration.
    - Extract the remaining line objects and sort them by their topmost occupied row from top to bottom.
    - For the object at position k, rotate the template clockwise by k*90 degrees, scale it by 2, and stack the resulting 4x4 stamp blocks vertically in that order.

    Key insight:
    - The lines contribute only their vertical ordering; that ordering encodes which rotation of the shared template must appear at each output block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    template_candidates = _corner_templates(g, bg)
    assert len(template_candidates) == 1, "expected exactly one corner 2x2 multicolor template"
    _template_pos, template, template_cells = template_candidates[0]

    comps = _components(g, bg=bg, ignore=template_cells)
    assert 3 <= len(comps) <= 4, "expected three or four line objects"
    colors = [comp["color"] for comp in comps]
    assert len(set(colors)) == len(colors), "line objects must use distinct colors"

    ordered: list[tuple[int, int]] = []
    for comp in comps:
        assert _is_valid_line_object(comp["coords"]), "foreground object is not a valid line segment"
        top = min(r for r, _ in comp["coords"])
        ordered.append((top, comp["color"]))

    top_rows = [top for top, _ in ordered]
    assert len(set(top_rows)) == len(top_rows), "line-object top rows must be distinct"
    ordered.sort()

    return _oracle_from_template(template, len(ordered))


_BG = 0
_GRID_H_RANGE = (14, 18)
_GRID_W_RANGE = (14, 18)
_RETRY_LIMIT = 400
_SEGMENT_COUNT_CHOICES = [3, 4]
_ALL_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_ORIENTATIONS = ["h", "v", "dr", "dl"]


def _template_at_corner(shape: tuple[int, int], corner: str, template: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    H, W = shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, W - 2
    elif corner == "bl":
        r0, c0 = H - 2, 0
    else:
        r0, c0 = H - 2, W - 2
    cells = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
    return (r0, c0), cells


def _random_template(colors: list[int]) -> np.ndarray:
    shuffled = colors[:]
    random.shuffle(shuffled)
    return np.array(shuffled, dtype=int).reshape(2, 2)


def _line_cells_h(length: int, row: int, col0: int) -> list[tuple[int, int]]:
    return [(row, col0 + i) for i in range(length)]


def _line_cells_v(length: int, row0: int, col: int) -> list[tuple[int, int]]:
    return [(row0 + i, col) for i in range(length)]


def _stair_cells(top: int, start_col: int, steps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [(top, start_col)]
    r, c = top, start_col
    for dr, dc in steps:
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _random_stair_steps(total_moves: int, horizontal_dc: int) -> list[tuple[int, int]]:
    down_moves = random.randint(1, total_moves - 1)
    side_moves = total_moves - down_moves
    seq = [(1, 0)] * down_moves + [(0, horizontal_dc)] * side_moves
    random.shuffle(seq)
    return seq


def _candidate_segment(shape: tuple[int, int], orientation: str) -> list[tuple[int, int]]:
    H, W = shape
    if orientation == "h":
        length = random.randint(2, 5)
        row = random.randint(0, H - 1)
        col0 = random.randint(0, W - length)
        return _line_cells_h(length, row, col0)
    if orientation == "v":
        length = random.randint(2, 5)
        row0 = random.randint(0, H - length)
        col = random.randint(0, W - 1)
        return _line_cells_v(length, row0, col)

    length = random.randint(3, 5)
    steps = _random_stair_steps(length - 1, 1 if orientation == "dr" else -1)
    down_moves = sum(dr for dr, _ in steps)
    side_moves = sum(abs(dc) for _, dc in steps)
    top = random.randint(0, H - 1 - down_moves)
    if orientation == "dr":
        start_col = random.randint(0, W - 1 - side_moves)
    else:
        start_col = random.randint(side_moves, W - 1)
    return _stair_cells(top, start_col, steps)


def _is_clear(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    for r, c in cell_set:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied and (rr, cc) not in cell_set:
                    return False
    return True


def _render_grid(
    shape: tuple[int, int],
    template: np.ndarray,
    corner: str,
    segments: list[dict],
) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    (tr, tc), _template_cells = _template_at_corner(shape, corner, template)
    grid[tr : tr + 2, tc : tc + 2] = template
    for seg in segments:
        for r, c in seg["cells"]:
            grid[r, c] = seg["color"]
    return grid


def _attempt() -> np.ndarray | None:
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)
    n_segments = random.choice(_SEGMENT_COUNT_CHOICES)
    picked = random.sample(_ALL_COLORS, 4 + n_segments)
    template_colors = picked[:4]
    segment_colors = picked[4:]
    template = _random_template(template_colors)
    corner = random.choice(["tl", "tr", "bl", "br"])

    _template_pos, template_cells = _template_at_corner((H, W), corner, template)
    occupied = set(template_cells)

    if n_segments == 4:
        orientations = _ORIENTATIONS[:]
        random.shuffle(orientations)
    else:
        orientations = random.sample(_ORIENTATIONS, 3)
        random.shuffle(orientations)

    segments: list[dict] = []
    used_top_rows: set[int] = set()
    for color, orientation in zip(segment_colors, orientations):
        placed = False
        for _ in range(120):
            cells = _candidate_segment((H, W), orientation)
            top = min(r for r, _ in cells)
            if top in used_top_rows:
                continue
            if not _is_valid_line_object(cells):
                continue
            if not _is_clear(cells, occupied):
                continue
            segments.append({"color": color, "cells": cells, "top": top, "orientation": orientation})
            occupied.update(cells)
            used_top_rows.add(top)
            placed = True
            break
        if not placed:
            return None

    random.shuffle(segments)
    grid = _render_grid((H, W), template, corner, segments)
    expected = _oracle_from_template(template, n_segments)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Reject easy degeneracies where a fixed rotation hypothesis would also work.
    blocks = [expected[i * 4 : (i + 1) * 4] for i in range(n_segments)]
    if n_segments >= 3 and len({blk.tobytes() for blk in blocks}) != n_segments:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
