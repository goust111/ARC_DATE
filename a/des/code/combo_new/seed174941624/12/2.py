from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 200
RAY_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in RAY_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _segment_info(coords: np.ndarray) -> dict:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if bool(np.all(rows == rows[0])):
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert int(coords.shape[0]) == c1 - c0 + 1
        return {"orientation": "H", "row": int(rows[0]), "c0": c0, "c1": c1}
    assert bool(np.all(cols == cols[0]))
    r0 = int(rows.min())
    r1 = int(rows.max())
    assert int(coords.shape[0]) == r1 - r0 + 1
    return {"orientation": "V", "col": int(cols[0]), "r0": r0, "r1": r1}


def _cast_to_bend(
    grid: np.ndarray,
    color: int,
    bg: int,
    bend_set: set[tuple[int, int]],
    start: tuple[int, int],
    step: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = start[0] + step[0], start[1] + step[1]
    trail: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == bg:
        trail.append((r, c))
        r += step[0]
        c += step[1]
    if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == color and (r, c) in bend_set:
        return trail + [(r, c)]
    return []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect each color's scattered orthogonal line fragments through its isolated bend markers.

    Input structure:
    - The background is the dominant color.
    - Each non-background color forms several disconnected straight line fragments
      plus one or more isolated single-pixel markers.
    - Every fragment is horizontal or vertical, and every isolated single pixel of
      that color is a bend location on the full path.

    Rule:
    - For each color, split its pixels into connected components.
    - Treat single-pixel components as bend markers and longer 1-cell-thick
      components as line fragments.
    - From each fragment endpoint, extend a ray in the fragment's direction. If the
      first non-background pixel reached is a same-color bend marker, fill the
      background cells on that ray with the fragment color.
    - Keep all original fragments and bend markers; the added fills complete the
      orthogonal polyline for every color.

    Key insight:
    - The isolated single pixels are not noise: they are the turning points, so each
      missing connection is recovered by extending a fragment endpoint until it hits
      a same-color bend marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        comps = _components(g == color)
        bends: set[tuple[int, int]] = set()
        segments: list[dict] = []

        for comp in comps:
            if int(comp.shape[0]) == 1:
                bends.add((int(comp[0, 0]), int(comp[0, 1])))
            else:
                segments.append(_segment_info(comp))

        if not bends:
            continue

        bend_hits = {bend: {"H": 0, "V": 0} for bend in bends}
        fills: list[tuple[int, int]] = []

        for seg in segments:
            if seg["orientation"] == "H":
                row = int(seg["row"])
                left = _cast_to_bend(g, color, bg, bends, (row, int(seg["c0"])), (0, -1))
                right = _cast_to_bend(g, color, bg, bends, (row, int(seg["c1"])), (0, 1))
                for path in (left, right):
                    if path:
                        bend_hits[path[-1]]["H"] += 1
                        fills.extend(path)
            else:
                col = int(seg["col"])
                up = _cast_to_bend(g, color, bg, bends, (int(seg["r0"]), col), (-1, 0))
                down = _cast_to_bend(g, color, bg, bends, (int(seg["r1"]), col), (1, 0))
                for path in (up, down):
                    if path:
                        bend_hits[path[-1]]["V"] += 1
                        fills.extend(path)

        for bend in bends:
            assert bend_hits[bend]["H"] == 1
            assert bend_hits[bend]["V"] == 1

        if fills:
            rr = [r for r, _ in fills]
            cc = [c for _, c in fills]
            out[rr, cc] = color

    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        step = 1 if b[1] >= a[1] else -1
        return [(r, c) for c in range(a[1], b[1] + step, step)]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _neighbors4(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in RAY_DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _can_use_start(start: tuple[int, int], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    if start in occupied:
        return False
    return all(nb not in occupied for nb in _neighbors4(start, h, w))


def _can_add_segment(
    start: tuple[int, int],
    end: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    cells = _line_cells(start, end)
    for idx, cell in enumerate(cells):
        if idx == 0:
            continue
        if cell in occupied:
            return False
        for nb in _neighbors4(cell, h, w):
            if nb != start and nb in occupied:
                return False
    return True


def _random_vertex_path(
    rows: list[int],
    cols: list[int],
    bends: int,
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    nseg = bends + 1
    starts = [(r, c) for r in rows for c in cols]
    random.shuffle(starts)

    col_index = {c: i for i, c in enumerate(cols)}
    row_index = {r: i for i, r in enumerate(rows)}

    for start in starts:
        if not _can_use_start(start, occupied, h, w):
            continue
        first_axis = random.choice(["H", "V"])
        local_occ = set(occupied)
        local_occ.add(start)
        vertices = [start]

        def dfs(step_idx: int, axis: str, current: tuple[int, int]) -> bool:
            if step_idx == nseg:
                return True

            candidates: list[tuple[int, int]] = []
            if axis == "H":
                base = col_index[current[1]]
                for delta in (1, 2, 3):
                    if base - delta >= 0:
                        candidates.append((current[0], cols[base - delta]))
                    if base + delta < len(cols):
                        candidates.append((current[0], cols[base + delta]))
            else:
                base = row_index[current[0]]
                for delta in (1, 2, 3):
                    if base - delta >= 0:
                        candidates.append((rows[base - delta], current[1]))
                    if base + delta < len(rows):
                        candidates.append((rows[base + delta], current[1]))
            random.shuffle(candidates)

            for nxt in candidates:
                if not _can_add_segment(current, nxt, local_occ, h, w):
                    continue
                cells = _line_cells(current, nxt)
                new_cells = cells[1:]
                for cell in new_cells:
                    local_occ.add(cell)
                vertices.append(nxt)
                if dfs(step_idx + 1, "V" if axis == "H" else "H", nxt):
                    return True
                vertices.pop()
                for cell in new_cells:
                    local_occ.remove(cell)
            return False

        if dfs(0, first_axis, start):
            return vertices

    return None


def _fragment_spans(distance: int, start_is_bend: bool, end_is_bend: bool) -> list[tuple[int, int]]:
    min_pos = 2 if start_is_bend else 0
    max_pos = distance - 2 if end_is_bend else distance
    spans: list[tuple[int, int]] = []
    for length in range(2, 5):
        for start in range(min_pos, max_pos - length + 2):
            end = start + length - 1
            if end <= max_pos:
                spans.append((start, end))
    return spans


def _point_on_segment(a: tuple[int, int], b: tuple[int, int], offset: int) -> tuple[int, int]:
    dr = 0 if a[0] == b[0] else (1 if b[0] > a[0] else -1)
    dc = 0 if a[1] == b[1] else (1 if b[1] > a[1] else -1)
    return (a[0] + dr * offset, a[1] + dc * offset)


def _build_grids(
    shape: tuple[int, int],
    paths: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for path in paths:
        color = int(path["color"])
        vertices = path["vertices"]
        for i in range(len(vertices) - 1):
            for r, c in _line_cells(vertices[i], vertices[i + 1]):
                out[r, c] = color

        for r, c in vertices[1:-1]:
            inp[r, c] = color

        nseg = len(vertices) - 1
        for i in range(nseg):
            a = vertices[i]
            b = vertices[i + 1]
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            spans = _fragment_spans(dist, start_is_bend=i > 0, end_is_bend=i < nseg - 1)
            assert spans
            s0, s1 = random.choice(spans)
            for offset in range(s0, s1 + 1):
                r, c = _point_on_segment(a, b, offset)
                inp[r, c] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    rows = list(range(1, h - 1, 5))
    cols = list(range(1, w - 1, 5))
    if len(rows) < 3 or len(cols) < 3:
        return None

    num_colors = random.randint(1, 3)
    colors = random.sample(PALETTE, num_colors)
    occupied: set[tuple[int, int]] = set()
    paths: list[dict] = []

    for idx, color in enumerate(colors):
        max_bends = 4 if num_colors == 1 else 3
        bends = random.randint(1, max_bends)
        vertices = None
        for _ in range(PATH_ATTEMPTS):
            vertices = _random_vertex_path(rows, cols, bends, occupied, h, w)
            if vertices is not None:
                break
        if vertices is None:
            return None

        cells: set[tuple[int, int]] = set()
        for i in range(len(vertices) - 1):
            cells.update(_line_cells(vertices[i], vertices[i + 1]))
        occupied.update(cells)
        paths.append({"color": color, "vertices": vertices, "index": idx})

    inp, expected = _build_grids((h, w), paths)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
