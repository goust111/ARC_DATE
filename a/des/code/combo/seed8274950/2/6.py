from __future__ import annotations

import numpy as np


GREY = 5
GEN_ATTEMPTS = 400
PAIR_TRIES = 200
DISTRACTOR_TRIES = 80
SIZE_MIN = 12
SIZE_MAX = 21
GAP_MIN = 3
GAP_MAX = 9


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": color, "cells": cells})
    return components


def _marker_cells(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for comp in _monochrome_components(grid, bg):
        cells = comp["cells"]
        if len(cells) == 1:
            markers.append(cells[0])
    return markers


def _horizontal_pairs(markers: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    by_row: dict[int, list[int]] = {}
    for r, c in markers:
        by_row.setdefault(r, []).append(c)
    pairs: list[tuple[int, int, int]] = []
    for r, cols in by_row.items():
        cols = sorted(cols)
        for c0, c1 in zip(cols, cols[1:]):
            pairs.append((r, c0, c1))
    return pairs


def _vertical_pairs(markers: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    by_col: dict[int, list[int]] = {}
    for r, c in markers:
        by_col.setdefault(c, []).append(r)
    pairs: list[tuple[int, int, int]] = []
    for c, rows in by_col.items():
        rows = sorted(rows)
        for r0, r1 in zip(rows, rows[1:]):
            pairs.append((c, r0, r1))
    return pairs


def _path_color(
    grid: np.ndarray, bg: int, marker_set: set[tuple[int, int]], path_cells: list[tuple[int, int]]
) -> int:
    if not path_cells:
        return GREY
    mid = (len(path_cells) - 1) / 2.0
    hits: list[tuple[float, int, int]] = []
    for idx, (r, c) in enumerate(path_cells):
        if int(grid[r, c]) != bg and (r, c) not in marker_set:
            hits.append((abs(idx - mid), idx, int(grid[r, c])))
    if not hits:
        return GREY
    hits.sort(key=lambda item: (item[0], item[1]))
    return hits[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect consecutive aligned markers.

    - Adjacent markers on the same row get a horizontal solid line.
    - Adjacent markers on the same column get a vertical dashed line.
    - The line color comes from the nearest non-marker color already present on
      the path; if the path is plain background, use grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    markers = _marker_cells(g, bg)
    marker_set = set(markers)
    out = g.copy()

    for r, c0, c1 in _horizontal_pairs(markers):
        path = [(r, c) for c in range(c0 + 1, c1)]
        color = _path_color(g, bg, marker_set, path)
        if path:
            out[r, c0 + 1 : c1] = color

    for c, r0, r1 in _vertical_pairs(markers):
        path = [(r, c) for r in range(r0 + 1, r1)]
        color = _path_color(g, bg, marker_set, path)
        for idx, (r, _c) in enumerate(path):
            if idx % 2 == 0:
                out[r, c] = color

    return out


def _sample_colors(rng: np.random.Generator, k: int, exclude: set[int]) -> list[int]:
    pool = [c for c in range(10) if c not in exclude]
    if not pool:
        raise ValueError("empty color pool")
    if len(pool) >= k:
        choice = rng.choice(pool, size=k, replace=False)
        return [int(x) for x in choice]
    choice = rng.choice(pool, size=k, replace=True)
    return [int(x) for x in choice]


def _box(cells: set[tuple[int, int]], h: int, w: int, margin: int = 1) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return (
        max(0, min(rs) - margin),
        max(0, min(cs) - margin),
        min(h - 1, max(rs) + margin),
        min(w - 1, max(cs) + margin),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _start_bounds(lo_marker: int, hi_marker: int, length: int) -> tuple[int, int] | None:
    low = lo_marker + 1
    high = hi_marker - length
    if low > high:
        return None
    if high - low >= 2:
        low += 1
        high -= 1
    return low, high


def _horizontal_region_patch(
    rng: np.random.Generator, h: int, r: int, c0: int, c1: int
) -> list[tuple[int, int]] | None:
    options: list[tuple[str, int, int, int, int | None]] = []
    for length in (2, 3):
        bounds = _start_bounds(c0, c1, length)
        if bounds is not None:
            low, high = bounds
            options.append(("line", length, low, high, None))
    bounds = _start_bounds(c0, c1, 2)
    if bounds is not None:
        low, high = bounds
        if r - 1 >= 0:
            options.append(("block", 2, low, high, r - 1))
        if r + 1 < h:
            options.append(("block", 2, low, high, r))
    if not options:
        return None
    kind, length, low, high, row0 = options[int(rng.integers(len(options)))]
    if kind == "line":
        start = int(rng.integers(low, high + 1))
        return [(r, c) for c in range(start, start + length)]
    start = int(rng.integers(low, high + 1))
    assert row0 is not None
    return [(row0 + dr, start + dc) for dr in range(2) for dc in range(2)]


def _vertical_region_patch(
    rng: np.random.Generator, w: int, c: int, r0: int, r1: int
) -> list[tuple[int, int]] | None:
    options: list[tuple[str, int, int, int, int | None]] = []
    for length in (2, 3):
        bounds = _start_bounds(r0, r1, length)
        if bounds is not None:
            low, high = bounds
            options.append(("line", length, low, high, None))
    bounds = _start_bounds(r0, r1, 2)
    if bounds is not None:
        low, high = bounds
        if c - 1 >= 0:
            options.append(("block", 2, low, high, c - 1))
        if c + 1 < w:
            options.append(("block", 2, low, high, c))
    if not options:
        return None
    kind, length, low, high, col0 = options[int(rng.integers(len(options)))]
    if kind == "line":
        start = int(rng.integers(low, high + 1))
        return [(r, c) for r in range(start, start + length)]
    start = int(rng.integers(low, high + 1))
    assert col0 is not None
    return [(start + dr, col0 + dc) for dr in range(2) for dc in range(2)]


def _sample_horizontal_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    with_region: bool,
    used_rows: set[int],
    used_cols: set[int],
    boxes: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    rows = [r for r in range(1, h - 1) if r not in used_rows]
    cols = [c for c in range(1, w - 1) if c not in used_cols]
    if not rows or len(cols) < 2:
        return None

    gap_hi = min(GAP_MAX, w - 3)
    for _ in range(PAIR_TRIES):
        r = int(rng.choice(rows))
        c0 = int(rng.choice(cols))
        partners = [c for c in cols if c != c0 and GAP_MIN <= abs(c - c0) - 1 <= gap_hi]
        if not partners:
            continue
        c1 = int(rng.choice(partners))
        if c0 > c1:
            c0, c1 = c1, c0

        path = [(r, c) for c in range(c0 + 1, c1)]
        region_cells: list[tuple[int, int]] = []
        line_color = GREY
        if with_region:
            patch = _horizontal_region_patch(rng, h, r, c0, c1)
            if patch is None:
                continue
            region_cells = patch
            line_color = _sample_colors(rng, 1, {bg, GREY})[0]

        marker_colors = _sample_colors(rng, 2, {bg, GREY, line_color} if with_region else {bg, GREY})
        markers = [(r, c0), (r, c1)]
        cells = set(markers) | set(path) | set(region_cells)
        pair_box = _box(cells, h, w, margin=1)
        if any(_boxes_overlap(pair_box, old) for old in boxes):
            continue

        return {
            "orientation": "h",
            "markers": markers,
            "marker_colors": marker_colors,
            "path": path,
            "region_cells": region_cells,
            "line_color": line_color,
            "box": pair_box,
        }
    return None


def _sample_vertical_pair(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    with_region: bool,
    used_rows: set[int],
    used_cols: set[int],
    boxes: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    rows = [r for r in range(1, h - 1) if r not in used_rows]
    cols = [c for c in range(1, w - 1) if c not in used_cols]
    if len(rows) < 2 or not cols:
        return None

    gap_hi = min(GAP_MAX, h - 3)
    for _ in range(PAIR_TRIES):
        c = int(rng.choice(cols))
        r0 = int(rng.choice(rows))
        partners = [r for r in rows if r != r0 and GAP_MIN <= abs(r - r0) - 1 <= gap_hi]
        if not partners:
            continue
        r1 = int(rng.choice(partners))
        if r0 > r1:
            r0, r1 = r1, r0

        path = [(r, c) for r in range(r0 + 1, r1)]
        region_cells = []
        line_color = GREY
        if with_region:
            patch = _vertical_region_patch(rng, w, c, r0, r1)
            if patch is None:
                continue
            region_cells = patch
            line_color = _sample_colors(rng, 1, {bg, GREY})[0]

        marker_colors = _sample_colors(rng, 2, {bg, GREY, line_color} if with_region else {bg, GREY})
        markers = [(r0, c), (r1, c)]
        cells = set(markers) | set(path) | set(region_cells)
        pair_box = _box(cells, h, w, margin=1)
        if any(_boxes_overlap(pair_box, old) for old in boxes):
            continue

        return {
            "orientation": "v",
            "markers": markers,
            "marker_colors": marker_colors,
            "path": path,
            "region_cells": region_cells,
            "line_color": line_color,
            "box": pair_box,
        }
    return None


def _render_input(
    h: int,
    w: int,
    bg: int,
    pairs: list[dict[str, object]],
    distractors: list[tuple[list[tuple[int, int]], int]],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for pair in pairs:
        line_color = int(pair["line_color"])
        for r, c in pair["region_cells"]:
            grid[r, c] = line_color
    for cells, color in distractors:
        for r, c in cells:
            grid[r, c] = color
    for pair in pairs:
        markers = pair["markers"]
        marker_colors = pair["marker_colors"]
        for (r, c), color in zip(markers, marker_colors):
            grid[r, c] = int(color)
    return grid


def _expected_output(inp: np.ndarray, pairs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for pair in pairs:
        line_color = int(pair["line_color"])
        if pair["orientation"] == "h":
            r = int(pair["markers"][0][0])
            c0 = int(pair["markers"][0][1])
            c1 = int(pair["markers"][1][1])
            out[r, c0 + 1 : c1] = line_color
        else:
            c = int(pair["markers"][0][1])
            r0 = int(pair["markers"][0][0])
            r1 = int(pair["markers"][1][0])
            for idx, r in enumerate(range(r0 + 1, r1)):
                if idx % 2 == 0:
                    out[r, c] = line_color
    return out


def _sample_distractor_shape(
    rng: np.random.Generator, h: int, w: int
) -> list[tuple[int, int]] | None:
    kind = str(rng.choice(["h2", "v2", "b22"]))
    if kind == "h2":
        if w < 2:
            return None
        r = int(rng.integers(h))
        c = int(rng.integers(w - 1))
        return [(r, c), (r, c + 1)]
    if kind == "v2":
        if h < 2:
            return None
        r = int(rng.integers(h - 1))
        c = int(rng.integers(w))
        return [(r, c), (r + 1, c)]
    if h < 2 or w < 2:
        return None
    r = int(rng.integers(h - 1))
    c = int(rng.integers(w - 1))
    return [(r + dr, c + dc) for dr in range(2) for dc in range(2)]


def _place_distractors(
    rng: np.random.Generator,
    h: int,
    w: int,
    bg: int,
    pairs: list[dict[str, object]],
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[list[tuple[int, int]], int]]:
    distractors: list[tuple[list[tuple[int, int]], int]] = []
    target = int(rng.integers(0, 4))
    path_cells = {cell for pair in pairs for cell in pair["path"]}
    occupied = {cell for pair in pairs for cell in pair["markers"] + pair["region_cells"]}

    for _ in range(target):
        placed = False
        for _ in range(DISTRACTOR_TRIES):
            cells = _sample_distractor_shape(rng, h, w)
            if cells is None:
                continue
            cell_set = set(cells)
            if cell_set & path_cells or cell_set & occupied:
                continue
            d_box = _box(cell_set, h, w, margin=1)
            if any(_boxes_overlap(d_box, b) for b in boxes):
                continue
            color = _sample_colors(rng, 1, {bg, GREY})[0]
            distractors.append((cells, color))
            occupied |= cell_set
            boxes.append(d_box)
            placed = True
            break
        if not placed:
            break
    return distractors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(SIZE_MIN, SIZE_MAX))
    w = int(rng.integers(SIZE_MIN, SIZE_MAX))
    bg = _sample_colors(rng, 1, {GREY})[0]

    max_pairs = 3 if min(h, w) >= 16 else 2
    h_pairs = int(rng.integers(1, max_pairs + 1))
    v_pairs = int(rng.integers(1, max_pairs + 1))
    if h_pairs + 2 * v_pairs > h - 2 or 2 * h_pairs + v_pairs > w - 2:
        return None

    total_pairs = h_pairs + v_pairs
    n_region = total_pairs // 2 + (int(rng.integers(0, 2)) if total_pairs % 2 else 0)
    region_flags = [True] * n_region + [False] * (total_pairs - n_region)
    rng.shuffle(region_flags)

    orientations = ["h"] * h_pairs + ["v"] * v_pairs
    rng.shuffle(orientations)

    pairs: list[dict[str, object]] = []
    boxes: list[tuple[int, int, int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for orientation, with_region in zip(orientations, region_flags):
        if orientation == "h":
            pair = _sample_horizontal_pair(rng, h, w, bg, with_region, used_rows, used_cols, boxes)
            if pair is None:
                return None
            used_rows.add(int(pair["markers"][0][0]))
            used_cols.update(int(c) for _r, c in pair["markers"])
        else:
            pair = _sample_vertical_pair(rng, h, w, bg, with_region, used_rows, used_cols, boxes)
            if pair is None:
                return None
            used_cols.add(int(pair["markers"][0][1]))
            used_rows.update(int(r) for r, _c in pair["markers"])
        boxes.append(pair["box"])
        pairs.append(pair)

    if not any(int(pair["line_color"]) == GREY for pair in pairs):
        return None
    if not any(int(pair["line_color"]) != GREY for pair in pairs):
        return None

    distractor_boxes = list(boxes)
    distractors = _place_distractors(rng, h, w, bg, pairs, distractor_boxes)
    inp = _render_input(h, w, bg, pairs, distractors)
    expected = _expected_output(inp, pairs)

    actual = main(inp)
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
