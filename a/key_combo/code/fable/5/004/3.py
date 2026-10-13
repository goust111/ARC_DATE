from __future__ import annotations

from collections import Counter

import numpy as np


CYAN = 8
RED = 2
MAX_ATTEMPTS = 2000


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _group_by_row(points: set[tuple[int, int]]) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = {}
    for r, c in points:
        rows.setdefault(int(r), []).append(int(c))
    for cols in rows.values():
        cols.sort()
    return rows


def _group_by_col(points: set[tuple[int, int]]) -> dict[int, list[int]]:
    cols: dict[int, list[int]] = {}
    for r, c in points:
        cols.setdefault(int(c), []).append(int(r))
    for rows in cols.values():
        rows.sort()
    return cols


def _paint_cell(out: np.ndarray, r: int, c: int, bg: int) -> None:
    if int(out[r, c]) in (bg, CYAN):
        out[r, c] = CYAN


def _draw_paired_edges(out: np.ndarray, corners: set[tuple[int, int]], bg: int) -> None:
    for r, cols in _group_by_row(corners).items():
        assert len(cols) % 2 == 0, "every completed corner row must contain an even number of corners"
        for c0, c1 in zip(cols[0::2], cols[1::2]):
            assert c0 < c1, "horizontal edge endpoints must be distinct"
            for c in range(c0, c1 + 1):
                _paint_cell(out, r, c, bg)

    for c, rows in _group_by_col(corners).items():
        assert len(rows) % 2 == 0, "every completed corner column must contain an even number of corners"
        for r0, r1 in zip(rows[0::2], rows[1::2]):
            assert r0 < r1, "vertical edge endpoints must be distinct"
            for r in range(r0, r1 + 1):
                _paint_cell(out, r, c, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a rectilinear loop from cyan turn points and mark its one missing corner.

    Input structure:
    - The grid has a dominant background color.
    - Cyan pixels are all turn points of one axis-aligned rectilinear closed loop,
      except that exactly one turn point has been erased.
    - A few isolated pixels in other colors are distractors and are not part of the loop.

    Rule:
    - Count cyan turn points in each row and in each column; the only odd row and
      the only odd column intersect at the missing turn point.
    - Add that point to the turn-point set for pairing, draw cyan horizontal edges
      between adjacent paired turn points in each row, and draw cyan vertical edges
      between adjacent paired turn points in each column.
    - Paint the missing turn point red after drawing the loop, and leave distractors unchanged.

    Key insight:
    - In a complete rectilinear loop, every row and column contains an even number
      of turn points, so the erased corner is revealed by row-column parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must not exceed 30x30"

    bg = _background_color(g)
    cyan_points = {(int(r), int(c)) for r, c in np.argwhere(g == CYAN)}
    assert len(cyan_points) >= 3, "expected visible cyan turn points"

    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in cyan_points:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    odd_rows = [r for r, n in row_counts.items() if n % 2 == 1]
    odd_cols = [c for c, n in col_counts.items() if n % 2 == 1]
    assert len(odd_rows) == 1 and len(odd_cols) == 1, "expected exactly one odd row and one odd column"

    missing = (int(odd_rows[0]), int(odd_cols[0]))
    mr, mc = missing
    assert 0 <= mr < h and 0 <= mc < w, "missing turn point must be in bounds"
    assert int(g[mr, mc]) == bg, "the missing turn point cell should be blank"
    assert missing not in cyan_points, "the missing point should not already be cyan"

    corners = set(cyan_points)
    corners.add(missing)

    out = g.copy()
    _draw_paired_edges(out, corners, bg)
    out[mr, mc] = RED
    return out


def _simplify_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = [tuple(map(int, p)) for p in vertices]
    while len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()

    changed = True
    while changed and len(pts) >= 4:
        changed = False
        new: list[tuple[int, int]] = []
        n = len(pts)
        for i, p in enumerate(pts):
            prev = pts[(i - 1) % n]
            nxt = pts[(i + 1) % n]
            if p == prev or p == nxt:
                changed = True
                continue
            if prev[0] == p[0] == nxt[0] or prev[1] == p[1] == nxt[1]:
                changed = True
                continue
            new.append(p)
        pts = new
    return pts


def _edge_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    n = len(vertices)
    for i in range(n):
        r0, c0 = vertices[i]
        r1, c1 = vertices[(i + 1) % n]
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            for c in range(lo, hi + 1):
                cells.add((r0, c))
        elif c0 == c1:
            lo, hi = sorted((r0, r1))
            for r in range(lo, hi + 1):
                cells.add((r, c0))
        else:
            raise ValueError("loop edges must be axis aligned")
    return cells


def _draw_sequence_edges(out: np.ndarray, vertices: list[tuple[int, int]], bg: int) -> None:
    for r, c in _edge_cells(vertices):
        _paint_cell(out, r, c, bg)


def _sample_breakpoints(rng: np.random.Generator, width: int, intervals: int) -> list[int] | None:
    min_gap = int(rng.integers(2, 4))
    capacity = (width - 2) - 1
    base_span = intervals * min_gap
    if base_span > capacity:
        return None

    extra = int(rng.integers(0, capacity - base_span + 1))
    gaps = [min_gap] * intervals
    for _ in range(extra):
        gaps[int(rng.integers(0, intervals))] += 1

    start_slack = capacity - base_span - extra
    start = int(rng.integers(1, 1 + start_slack + 1))
    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + gap)
    return cols


def _sample_spaced_levels(
    rng: np.random.Generator,
    candidates: list[int],
    max_count: int = 3,
) -> list[int] | None:
    if len(candidates) < 2:
        return None
    upper = min(max_count, len(candidates))
    for _ in range(100):
        k = int(rng.integers(2, upper + 1))
        levels = sorted(map(int, rng.choice(candidates, size=k, replace=False).tolist()))
        if all(b - a >= 2 for a, b in zip(levels, levels[1:])):
            return levels
    return None


def _sample_chain(rng: np.random.Generator, levels: list[int], intervals: int) -> list[int]:
    seq = [int(rng.choice(levels))]
    for _ in range(1, intervals):
        choices = [x for x in levels if x != seq[-1]]
        repeated = [x for x in choices if x in seq[:-1]]
        if repeated and rng.random() < 0.7:
            seq.append(int(rng.choice(repeated)))
        else:
            seq.append(int(rng.choice(choices)))
    return seq


def _build_loop(cols: list[int], top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    intervals = len(top)
    vertices: list[tuple[int, int]] = [(top[0], cols[0])]

    for i in range(intervals):
        vertices.append((top[i], cols[i + 1]))
        if i < intervals - 1:
            vertices.append((top[i + 1], cols[i + 1]))

    vertices.append((bottom[-1], cols[-1]))
    for i in range(intervals - 1, -1, -1):
        vertices.append((bottom[i], cols[i]))
        if i > 0:
            vertices.append((bottom[i - 1], cols[i]))

    return _simplify_vertices(vertices)


def _row_col_counts(vertices: list[tuple[int, int]]) -> tuple[dict[int, int], dict[int, int]]:
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for r, c in vertices:
        rows[r] = rows.get(r, 0) + 1
        cols[c] = cols.get(c, 0) + 1
    return rows, cols


def _choose_missing(
    rng: np.random.Generator,
    vertices: list[tuple[int, int]],
) -> tuple[int, int] | None:
    row_counts, col_counts = _row_col_counts(vertices)
    rich = [p for p in vertices if row_counts[p[0]] >= 4 or col_counts[p[1]] >= 4]
    if not rich:
        return None

    rs = [r for r, _ in vertices]
    cs = [c for _, c in vertices]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    non_bbox_corners = [p for p in rich if not (p[0] in (r0, r1) and p[1] in (c0, c1))]
    candidates = non_bbox_corners or rich
    return tuple(map(int, candidates[int(rng.integers(0, len(candidates)))]))


def _bbox_outline(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    rs = [r for r, _ in vertices]
    cs = [c for _, c in vertices]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _has_unique_missing_parity(
    visible: set[tuple[int, int]],
    missing: tuple[int, int],
) -> bool:
    rows, cols = _row_col_counts(list(visible))
    odd_rows = [r for r, n in rows.items() if n % 2 == 1]
    odd_cols = [c for c, n in cols.items() if n % 2 == 1]
    return odd_rows == [missing[0]] and odd_cols == [missing[1]]


def _isolated_from_non_bg(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != bg:
                return False
    return True


def _add_distractors(
    rng: np.random.Generator,
    grid: np.ndarray,
    bg: int,
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    colors = [c for c in range(10) if c not in (bg, CYAN, RED)]
    target = int(rng.integers(2, 7))

    for _ in range(target):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if int(grid[r, c]) == bg
            and (r, c) not in blocked
            and _isolated_from_non_bg(grid, r, c, bg)
        ]
        if not candidates:
            return False
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        grid[r, c] = int(rng.choice(colors))
    return True


def _expected_output(
    inp: np.ndarray,
    vertices: list[tuple[int, int]],
    missing: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    _draw_sequence_edges(out, vertices, bg)
    out[missing] = RED
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(14, 31))
    w = int(rng.integers(14, 31))
    intervals = int(rng.integers(4, min(8, max(5, (w - 3) // 2 + 1))))

    cols = _sample_breakpoints(rng, w, intervals)
    if cols is None:
        return None

    top_candidates = list(range(1, max(3, h // 2 - 1)))
    bottom_candidates = list(range(min(h - 2, h // 2 + 2), h - 1))
    top_levels = _sample_spaced_levels(rng, top_candidates)
    bottom_levels = _sample_spaced_levels(rng, bottom_candidates)
    if top_levels is None or bottom_levels is None:
        return None
    if max(top_levels) + 3 >= min(bottom_levels):
        return None

    top = _sample_chain(rng, top_levels, intervals)
    bottom = _sample_chain(rng, bottom_levels, intervals)
    vertices = _build_loop(cols, top, bottom)
    if len(vertices) < 8 or len(set(vertices)) != len(vertices):
        return None

    full_rows, full_cols = _row_col_counts(vertices)
    if any(n % 2 for n in full_rows.values()) or any(n % 2 for n in full_cols.values()):
        return None
    if max(full_rows.values()) < 4 and max(full_cols.values()) < 4:
        return None

    cells = _edge_cells(vertices)
    if cells == _bbox_outline(vertices):
        return None

    missing = _choose_missing(rng, vertices)
    if missing is None:
        return None

    visible = set(vertices)
    visible.remove(missing)
    if not _has_unique_missing_parity(visible, missing):
        return None

    bg_choices = [0, 1, 3, 4, 5, 6, 7, 9]
    bg = int(rng.choice(bg_choices))
    grid = np.full((h, w), bg, dtype=int)
    for r, c in visible:
        grid[r, c] = CYAN

    if not _add_distractors(rng, grid, bg, cells):
        return None

    expected = _expected_output(grid, vertices, missing, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if int(np.sum(expected == RED)) != 1:
        return None
    if int(np.sum(expected == CYAN)) <= int(np.sum(grid == CYAN)):
        return None
    if _background_color(grid) != bg:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid missing-corner loop input")
