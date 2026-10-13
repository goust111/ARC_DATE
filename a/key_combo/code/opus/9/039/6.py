from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
MARKER_COLORS = (1, 4, 6, 7, 8)

MIN_SIDE = 12
MAX_SIDE = 18
MIN_MARKERS = 3
MAX_MARKERS = 5
MIN_GAP = 2
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BG)
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert MIN_MARKERS <= len(markers) <= MAX_MARKERS, "expected 3-5 marker pixels"

    rows = [r for r, _c, _color in markers]
    cols = [c for _r, c, _color in markers]
    colors = [color for _r, _c, color in markers]

    assert len(set(rows)) == len(rows), "marker rows must be unique"
    assert len(set(cols)) == len(cols), "marker columns must be unique"
    assert len(set(colors)) == len(colors), "marker colors must be unique"
    assert set(colors).issubset(MARKER_COLORS), "only the allowed marker colors may appear"

    h, w = grid.shape
    assert all(0 < r < h - 1 for r in rows), "markers must stay off the outer border"
    assert all(0 < c < w - 1 for c in cols), "markers must stay off the outer border"
    return markers


def _paint_guides(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    marker_map = {(r, c): color for r, c, color in markers}
    for r, c, color in markers:
        out[r, :] = color
        out[:, c] = color
        out[r, c] = color

    for r, _c, _color in markers:
        for _r, c, _other_color in markers:
            if (r, c) not in marker_map:
                out[r, c] = RED

    for (r, c), color in marker_map.items():
        out[r, c] = color
    return out


def _fill_regions(guide_grid: np.ndarray) -> np.ndarray:
    out = guide_grid.copy()
    edge_regions = 0
    inner_regions = 0
    for coords in _components_mask(out == BG):
        fill = GREEN if _touches_edge(coords, out.shape) else GREY
        out[coords[:, 0], coords[:, 1]] = fill
        edge_regions += int(fill == GREEN)
        inner_regions += int(fill == GREY)
    assert edge_regions > 0, "expected at least one outer background region"
    assert inner_regions > 0, "expected at least one enclosed background region"
    return out


def _segments(size: int, guides: list[int]) -> list[tuple[int, int]]:
    anchors = [-1] + sorted(guides) + [size]
    spans = []
    for left, right in zip(anchors, anchors[1:]):
        start = left + 1
        end = right - 1
        if start <= end:
            spans.append((start, end))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker into full row/column guides, then recolor the resulting background partitions by edge reachability.

    Input structure:
    - The grid has black background and 3-5 isolated single-pixel markers in distinct non-red colors.
    - Marker rows are all different, marker columns are all different, and every marker stays off the outer border.

    Rule:
    - Paint each marker's entire row and entire column with that marker's color.
    - Recolor every crossing between one marker row and another marker column red, while keeping the original marker pixels in their own colors.
    - Treat the remaining black cells as 4-connected background regions: regions touching the grid border become green, and enclosed regions become grey.

    Key insight:
    - The sparse markers first define a full rectilinear partition grid; only after that guide network is built can the black gaps be classified as outside-reachable or enclosed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    markers = _extract_markers(g)
    guides = _paint_guides(g.shape, markers)
    return _fill_regions(guides)


def _sample_spaced_positions(size: int, count: int, *, prefer_edge: bool) -> tuple[int, ...]:
    interior = range(1, size - 1)
    candidates = [
        combo
        for combo in itertools.combinations(interior, count)
        if all(b - a >= MIN_GAP for a, b in zip(combo, combo[1:]))
    ]
    assert candidates, "grid side too small for requested marker count"

    if prefer_edge:
        edge_candidates = [combo for combo in candidates if combo[0] == 1 or combo[-1] == size - 2]
        if edge_candidates:
            candidates = edge_candidates
    else:
        central_candidates = [combo for combo in candidates if combo[0] >= 2 and combo[-1] <= size - 3]
        if central_candidates:
            candidates = central_candidates

    if count >= 3:
        irregular = [
            combo
            for combo in candidates
            if len({b - a for a, b in zip(combo, combo[1:])}) > 1
        ]
        if irregular:
            candidates = irregular

    return random.choice(candidates)


def _expected_output_from_markers(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    return _fill_regions(_paint_guides(shape, markers))


def _validate_case(grid: np.ndarray) -> bool:
    try:
        markers = _extract_markers(grid)
        h, w = grid.shape
        rows = sorted(r for r, _c, _color in markers)
        cols = sorted(c for _r, c, _color in markers)
        if any(b - a < MIN_GAP for a, b in zip(rows, rows[1:])):
            return False
        if any(b - a < MIN_GAP for a, b in zip(cols, cols[1:])):
            return False

        row_spans = _segments(h, rows)
        col_spans = _segments(w, cols)
        if len(row_spans) != len(markers) + 1 or len(col_spans) != len(markers) + 1:
            return False
        if len({r1 - r0 + 1 for r0, r1 in row_spans}) == 1 and len({c1 - c0 + 1 for c0, c1 in col_spans}) == 1:
            return False

        guide_grid = _paint_guides(grid.shape, markers)
        out = _fill_regions(guide_grid)
        if not np.array_equal(main(grid), out):
            return False

        marker_map = {(r, c): color for r, c, color in markers}
        row_to_color = {r: color for r, _c, color in markers}
        col_to_color = {c: color for _r, c, color in markers}
        crossings = {(r, c) for r in row_to_color for c in col_to_color if (r, c) not in marker_map}

        if int(np.count_nonzero(guide_grid == RED)) != len(crossings):
            return False

        for (r, c), color in marker_map.items():
            if int(guide_grid[r, c]) != color:
                return False

        for r in range(h):
            for c in range(w):
                expected = None
                if (r, c) in marker_map:
                    expected = marker_map[(r, c)]
                elif (r, c) in crossings:
                    expected = RED
                elif r in row_to_color:
                    expected = row_to_color[r]
                elif c in col_to_color:
                    expected = col_to_color[c]
                if expected is not None and int(guide_grid[r, c]) != expected:
                    return False

        saw_green = False
        saw_grey = False
        for ri, (r0, r1) in enumerate(row_spans):
            for ci, (c0, c1) in enumerate(col_spans):
                expected = GREEN if ri in (0, len(row_spans) - 1) or ci in (0, len(col_spans) - 1) else GREY
                block = out[r0 : r1 + 1, c0 : c1 + 1]
                if block.size == 0 or not np.all(block == expected):
                    return False
                saw_green = saw_green or expected == GREEN
                saw_grey = saw_grey or expected == GREY

        if not saw_green or not saw_grey:
            return False
        if GREEN not in out or GREY not in out:
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    count = random.randint(MIN_MARKERS, MAX_MARKERS)

    mode = random.choice(("edge_both", "edge_rows", "edge_cols", "central"))
    rows = _sample_spaced_positions(h, count, prefer_edge=mode in {"edge_both", "edge_rows"})
    cols = list(_sample_spaced_positions(w, count, prefer_edge=mode in {"edge_both", "edge_cols"}))
    colors = random.sample(MARKER_COLORS, count)

    for _ in range(8):
        random.shuffle(cols)
        if tuple(cols) != tuple(sorted(cols)) or count == 3:
            break

    grid = np.zeros((h, w), dtype=int)
    markers = [(rows[i], cols[i], colors[i]) for i in range(count)]
    for r, c, color in markers:
        grid[r, c] = color

    if not _validate_case(grid):
        return None

    expected = _expected_output_from_markers(grid.shape, markers)
    if np.array_equal(expected, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
