from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400
MIN_COUNT = 3
MAX_COUNT = 4
BAND_HEIGHT = 7

SYMMETRIC_SHAPES = (
    np.array([[1], [1], [1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
)

ASYMMETRIC_SHAPES = (
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=int),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        queue = deque([(rr, cc)])
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            comp.append((cr, cc))
            for dr, dc in DIRS4:
                nr, nc = cr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _line_nodes(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 2 if bc >= ac else -2
        assert (bc - ac) % 2 == 0
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 2 if br >= ar else -2
        assert (br - ar) % 2 == 0
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("line must be axis-aligned")


def _path_step_neighbors(grid: np.ndarray, node: tuple[int, int], color: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        mr, mc = r + dr, c + dc
        nr, nc = r + 2 * dr, c + 2 * dc
        if not (_in_bounds(grid.shape, mr, mc) and _in_bounds(grid.shape, nr, nc)):
            continue
        if int(grid[mr, mc]) == BLACK and int(grid[nr, nc]) == color:
            out.append((nr, nc))
    return out


def _trace_path_nodes(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in _path_step_neighbors(grid, node, color):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _extract_shape(
    grid: np.ndarray, path_nodes: set[tuple[int, int]], color: int
) -> list[tuple[int, int]]:
    seeds: set[tuple[int, int]] = set()
    for r, c in path_nodes:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color and (nr, nc) not in path_nodes:
                seeds.add((nr, nc))
    assert seeds, "path endpoint must touch a same-color far shape"

    seen = set(seeds)
    queue = deque(seeds)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) != color or (nr, nc) in path_nodes or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return sorted(seen)


def _crop_component(coords: list[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        crop[r - r0, c - c0] = color
    return crop


def _find_outlines(grid: np.ndarray) -> list[dict]:
    outlines: list[dict] = []
    for coords in _component_coords(grid == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        expected = 2 * h + 2 * w - 4
        if len(coords) != expected:
            continue
        good = True
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border and int(grid[rr, cc]) != GRAY:
                    good = False
                if not on_border and int(grid[rr, cc]) == GRAY:
                    good = False
        if good:
            outlines.append({"bbox": (r0, c0, r1, c1), "coords": coords})
    outlines.sort(key=lambda item: item["bbox"])
    return outlines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray enclosed region only when its path-linked far shape is vertically symmetric.

    Input structure:
    - The grid has a black background, 3-4 gray rectangular outline rings, sparse same-color guide
      pixels on a two-cell lattice, and one small same-color far shape per path.
    - Each gray outline encloses a black interior and has exactly one adjacent non-black, non-gray
      guide pixel that identifies its path color.

    Rule:
    - Find every gray rectangular outline and its enclosed interior.
    - For each outline, locate the unique adjacent guide pixel and follow same-color guide nodes by
      horizontal or vertical jumps of two cells, requiring the skipped midpoint to stay black.
    - At the reached path endpoint, collect the adjacent same-color far shape, crop it to its bbox,
      and test whether that crop equals its left-right mirror image.
    - If the far shape is vertically symmetric, fill the outline's interior black cells with that
      path color; otherwise leave the interior black. Keep outlines, guide paths, and far shapes unchanged.

    Key insight:
    - The path establishes which far shape controls each enclosure, and only the far shape's vertical
      mirror symmetry decides whether the associated interior gets filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    outlines = _find_outlines(g)
    assert MIN_COUNT <= len(outlines) <= MAX_COUNT

    for outline in outlines:
        r0, c0, r1, c1 = outline["bbox"]
        adjacent: set[tuple[int, int]] = set()
        for rr, cc in outline["coords"]:
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) not in (BLACK, GRAY):
                    adjacent.add((nr, nc))
        assert len(adjacent) == 1
        start = next(iter(adjacent))
        color = int(g[start])

        path_nodes = _trace_path_nodes(g, start, color)
        shape_coords = _extract_shape(g, path_nodes, color)
        crop = _crop_component(shape_coords, color)
        symmetric = np.array_equal(crop, np.fliplr(crop))

        if symmetric:
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == BLACK] = color

    return out


def _shape_options(symmetric: bool) -> tuple[np.ndarray, ...]:
    return SYMMETRIC_SHAPES if symmetric else ASYMMETRIC_SHAPES


def _attachment_rows(shape: np.ndarray) -> list[int]:
    return [r for r in range(shape.shape[0]) if int(shape[r, 0]) == 1]


def _paint_rectangle_outline(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0, c0 : c0 + w] = GRAY
    grid[r0 + h - 1, c0 : c0 + w] = GRAY
    grid[r0 : r0 + h, c0] = GRAY
    grid[r0 : r0 + h, c0 + w - 1] = GRAY


def _paint_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(shape == 1):
        grid[top + int(r), left + int(c)] = color


def _sample_path(
    start: tuple[int, int], end: tuple[int, int], band_top: int, band_bottom: int
) -> list[tuple[int, int]] | None:
    sr, sc = start
    er, ec = end
    if sr == er:
        return _line_nodes(start, end)

    bend_candidates = [c for c in range(sc + 2, ec - 1, 2)]
    random.shuffle(bend_candidates)
    for bend_c in bend_candidates:
        if bend_c >= ec:
            continue
        path = _line_nodes(start, (sr, bend_c))
        path += _line_nodes((sr, bend_c), (er, bend_c))[1:]
        path += _line_nodes((er, bend_c), end)[1:]
        if all(band_top <= r <= band_bottom for r, _ in path):
            return path
    return None


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    count = random.randint(MIN_COUNT, MAX_COUNT)
    height = BAND_HEIGHT * count
    width = random.randint(22, 29)
    grid = np.zeros((height, width), dtype=int)

    colors = random.sample(PATH_COLORS, count)
    sym_count = random.randint(1, count - 1)
    sym_flags = [True] * sym_count + [False] * (count - sym_count)
    random.shuffle(sym_flags)

    specs: list[dict] = []
    for idx in range(count):
        band_top = idx * BAND_HEIGHT
        band_bottom = band_top + BAND_HEIGHT - 1
        rect_h = 4
        rect_w = random.randint(4, 6)
        rect_r0 = band_top + 1
        rect_c0 = random.randint(1, 3)
        rect_r1 = rect_r0 + rect_h - 1
        rect_c1 = rect_c0 + rect_w - 1
        color = colors[idx]
        symmetric = sym_flags[idx]

        _paint_rectangle_outline(grid, rect_r0, rect_c0, rect_h, rect_w)

        start_row = rect_r0 + random.choice([1, 2])
        start = (start_row, rect_c1 + 1)
        if start[1] >= width - 8:
            raise ValueError("canvas too narrow")
        grid[start] = color

        shape = random.choice(_shape_options(symmetric))
        attach_rows = _attachment_rows(shape)
        valid_shape_positions: list[tuple[int, int, int]] = []
        for attach_r in attach_rows:
            for top in range(band_top, band_bottom - shape.shape[0] + 2):
                end_row = top + attach_r
                if (end_row - start_row) % 2 != 0:
                    continue
                for left in range(max(rect_c1 + 7, width - shape.shape[1] - 5), width - shape.shape[1]):
                    end_col = left - 1
                    if (end_col - start[1]) % 2 != 0:
                        continue
                    if end_col - start[1] < 4:
                        continue
                    valid_shape_positions.append((top, left, attach_r))
        if not valid_shape_positions:
            raise ValueError("no valid shape placement")

        shape_top, shape_left, attach_r = random.choice(valid_shape_positions)
        end = (shape_top + attach_r, shape_left - 1)
        path = _sample_path(start, end, band_top, band_bottom)
        if path is None:
            raise ValueError("no valid path")

        for cell in path[1:]:
            if int(grid[cell]) != BLACK:
                raise ValueError("path overlap")
            grid[cell] = color

        shape_cells = [(shape_top + int(r), shape_left + int(c)) for r, c in np.argwhere(shape == 1)]
        for cell in shape_cells:
            if int(grid[cell]) != BLACK:
                raise ValueError("shape overlap")
        _paint_shape(grid, shape, shape_top, shape_left, color)

        specs.append(
            {
                "bbox": (rect_r0, rect_c0, rect_r1, rect_c1),
                "color": color,
                "symmetric": symmetric,
                "path": path,
                "shape_cells": shape_cells,
            }
        )

    out = grid.copy()
    for spec in specs:
        if spec["symmetric"]:
            r0, c0, r1, c1 = spec["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = spec["color"]

    _validate_case(grid, out, specs)
    return grid, out


def _validate_case(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> None:
    assert max(grid.shape) <= 30
    assert len(specs) == len({spec["color"] for spec in specs})
    assert any(spec["symmetric"] for spec in specs)
    assert any(not spec["symmetric"] for spec in specs)

    outlines = _find_outlines(grid)
    assert len(outlines) == len(specs)

    changed = 0
    unchanged = 0
    for outline in outlines:
        r0, c0, r1, c1 = outline["bbox"]
        adjacent: set[tuple[int, int]] = set()
        for rr, cc in outline["coords"]:
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) not in (BLACK, GRAY):
                    adjacent.add((nr, nc))
        assert len(adjacent) == 1
        start = next(iter(adjacent))
        color = int(grid[start])
        path_nodes = _trace_path_nodes(grid, start, color)
        shape_coords = _extract_shape(grid, path_nodes, color)
        assert 3 <= len(shape_coords) <= 6
        crop = _crop_component(shape_coords, color)
        symmetric = np.array_equal(crop, np.fliplr(crop))

        interior_in = grid[r0 + 1 : r1, c0 + 1 : c1]
        interior_out = expected[r0 + 1 : r1, c0 + 1 : c1]
        if symmetric:
            assert np.all(interior_out == color)
            changed += int(interior_out.size)
        else:
            assert np.array_equal(interior_in, interior_out)
            unchanged += int(interior_out.size)

    assert changed > 0
    assert unchanged > 0
    actual = main(grid)
    assert np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, _expected = _build_case()
        except (AssertionError, ValueError):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
