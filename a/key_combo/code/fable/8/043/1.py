from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
LINE_COUNT_RANGE = (3, 4)
LINE_LENGTH_RANGE = (5, 8)
MAX_GENERATION_ATTEMPTS = 400

SINGLE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, -1)),
    ((0, 0), (1, 0)),
    ((0, 0), (-1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, -1)),
    ((0, 0), (-1, 0), (-1, 1)),
    ((0, 0), (-1, 0), (-1, -1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (-1, 1)),
    ((0, 0), (0, -1), (1, -1)),
    ((0, 0), (0, -1), (-1, -1)),
)
FREE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, -1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _diag_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "endpoint pair must define a strict diagonal"
    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + k * sr, c0 + k * sc) for k in range(abs(dr) + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor or erase grey regions according to which hidden same-colored diagonals pass through them.

    Input structure:
    - The background is black.
    - Several non-grey foreground colors each appear exactly twice, and each pair forms the endpoints of a strict 45-degree diagonal.
    - Separate grey 4-connected regions lie on the canvas, and endpoint pixels are never edge-adjacent to grey.

    Rule:
    - For each colored endpoint pair, infer the full diagonal segment between the two endpoints without drawing it.
    - If a grey region is intersected by exactly one inferred diagonal, recolor the whole region to that diagonal's color.
    - If a grey region is intersected by two or more inferred diagonals, erase the whole region to the black background.
    - Grey regions hit by no inferred diagonal stay grey, and all endpoint pixels are erased to black.

    Key insight:
    - The diagonals are invisible generators: only their crossings with grey components matter, so the output is determined by reconstructing hidden diagonal paths from the endpoint pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = BG
    grey = GREY
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in (bg, grey))
    assert colors, "expected at least one colored endpoint pair"

    line_cells_by_color: dict[int, set[tuple[int, int]]] = {}
    endpoints = set()
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each visible color must appear exactly twice"
        a = tuple(int(x) for x in coords[0])
        b = tuple(int(x) for x in coords[1])
        cells = _diag_cells(a, b)
        for r, c in cells:
            assert int(g[r, c]) in (bg, grey, color), "diagonal path may only pass through background, grey, or its own endpoints"
        line_cells_by_color[color] = set(cells)
        endpoints.add(a)
        endpoints.add(b)

    out = g.copy()
    for r, c in endpoints:
        out[r, c] = bg

    grey_components = _connected_components(g == grey)
    for comp in grey_components:
        comp_set = {tuple(int(x) for x in rc) for rc in comp}
        hits = [color for color, cells in line_cells_by_color.items() if comp_set & cells]
        if len(hits) == 1:
            out[comp[:, 0], comp[:, 1]] = hits[0]
        elif len(hits) >= 2:
            out[comp[:, 0], comp[:, 1]] = bg

    return out


def _segment_cells(r0: int, c0: int, step_c: int, length: int) -> list[tuple[int, int]]:
    return [(r0 + k, c0 + k * step_c) for k in range(length + 1)]


def _sample_lines(height: int, width: int) -> list[dict] | None:
    n_lines = random.randint(*LINE_COUNT_RANGE)
    colors = random.sample(PALETTE, n_lines)
    occupied = set()
    lines = []
    specs = []
    for color in colors:
        specs.append((color, random.randint(*LINE_LENGTH_RANGE), random.choice((-1, 1))))
    random.shuffle(specs)

    for color, length, step_c in specs:
        choices = []
        for r0 in range(1, height - length - 1):
            for c0 in range(1, width - 1):
                c1 = c0 + step_c * length
                if not (1 <= c1 < width - 1):
                    continue
                cells = _segment_cells(r0, c0, step_c, length)
                if any(cell in occupied for cell in cells):
                    continue
                choices.append(cells)
        if not choices:
            return None
        cells = random.choice(choices)
        occupied.update(cells)
        lines.append(
            {
                "color": color,
                "cells": cells,
                "set": set(cells),
                "endpoints": {cells[0], cells[-1]},
                "interior": cells[1:-1],
            }
        )
    return lines


def _component_is_valid(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    used_grey: set[tuple[int, int]],
    endpoint_forbidden: set[tuple[int, int]],
) -> bool:
    height, width = shape
    if not cells:
        return False
    if any(not (0 <= r < height and 0 <= c < width) for r, c in cells):
        return False
    if cells & used_grey:
        return False
    if cells & endpoint_forbidden:
        return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in used_grey:
                return False
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return len(_connected_components(mask)) == 1


def _line_hit_ids(
    cells: set[tuple[int, int]],
    line_lookup: dict[tuple[int, int], set[int]],
) -> set[int]:
    hits = set()
    for cell in cells:
        hits |= line_lookup.get(cell, set())
    return hits


def _place_single_component(
    line_idx: int,
    lines: list[dict],
    shape: tuple[int, int],
    line_lookup: dict[tuple[int, int], set[int]],
    used_grey: set[tuple[int, int]],
    endpoint_forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    anchors = lines[line_idx]["interior"][:]
    random.shuffle(anchors)
    shapes = list(SINGLE_SHAPES)
    random.shuffle(shapes)
    for ar, ac in anchors:
        for offsets in shapes:
            cells = {(ar + dr, ac + dc) for dr, dc in offsets}
            if not _component_is_valid(cells, shape, used_grey, endpoint_forbidden):
                continue
            if _line_hit_ids(cells, line_lookup) == {line_idx}:
                return cells
    return None


def _path_cells(a: tuple[int, int], b: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    cells = [(r1, c1)]
    if row_first:
        step_c = _sign(c2 - c1)
        for c in range(c1 + step_c, c2 + step_c, step_c or 1):
            cells.append((r1, c))
        step_r = _sign(r2 - r1)
        for r in range(r1 + step_r, r2 + step_r, step_r or 1):
            cells.append((r, c2))
    else:
        step_r = _sign(r2 - r1)
        for r in range(r1 + step_r, r2 + step_r, step_r or 1):
            cells.append((r, c1))
        step_c = _sign(c2 - c1)
        for c in range(c1 + step_c, c2 + step_c, step_c or 1):
            cells.append((r2, c))
    out = []
    seen = set()
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _place_multi_component(
    lines: list[dict],
    shape: tuple[int, int],
    line_lookup: dict[tuple[int, int], set[int]],
    used_grey: set[tuple[int, int]],
    endpoint_forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    line_pairs = [(i, j) for i in range(len(lines)) for j in range(i + 1, len(lines))]
    random.shuffle(line_pairs)
    for i, j in line_pairs:
        anchors_i = lines[i]["interior"][:]
        anchors_j = lines[j]["interior"][:]
        random.shuffle(anchors_i)
        random.shuffle(anchors_j)
        for a in anchors_i:
            for b in anchors_j:
                dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
                if not (3 <= dist <= 10):
                    continue
                for row_first in random.sample([True, False], 2):
                    cells = set(_path_cells(a, b, row_first))
                    if not _component_is_valid(cells, shape, used_grey, endpoint_forbidden):
                        continue
                    if _line_hit_ids(cells, line_lookup) == {i, j}:
                        return cells
    return None


def _place_untouched_component(
    shape: tuple[int, int],
    line_lookup: dict[tuple[int, int], set[int]],
    used_grey: set[tuple[int, int]],
    endpoint_forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    height, width = shape
    anchors = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(anchors)
    shapes = list(FREE_SHAPES)
    random.shuffle(shapes)
    for ar, ac in anchors:
        for offsets in shapes:
            cells = {(ar + dr, ac + dc) for dr, dc in offsets}
            if not _component_is_valid(cells, shape, used_grey, endpoint_forbidden):
                continue
            if not _line_hit_ids(cells, line_lookup):
                return cells
    return None


def _render_input(height: int, width: int, lines: list[dict], grey_components: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for comp in grey_components:
        for r, c in comp:
            grid[r, c] = GREY
    for line in lines:
        a, b = line["cells"][0], line["cells"][-1]
        grid[a] = line["color"]
        grid[b] = line["color"]
    return grid


def _manual_output(
    height: int,
    width: int,
    lines: list[dict],
    grey_components: list[set[tuple[int, int]]],
    line_lookup: dict[tuple[int, int], set[int]],
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for comp in grey_components:
        hits = _line_hit_ids(comp, line_lookup)
        color = GREY
        if len(hits) == 1:
            color = lines[next(iter(hits))]["color"]
        elif len(hits) >= 2:
            color = BG
        for r, c in comp:
            out[r, c] = color
    return out


def _semantic_check(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in (BG, GREY))
    if not (3 <= len(colors) <= 4):
        return False

    line_lookup: dict[tuple[int, int], set[int]] = {}
    endpoint_forbidden = set()
    for idx, color in enumerate(colors):
        coords = np.argwhere(grid == color)
        if len(coords) != 2:
            return False
        a = tuple(int(x) for x in coords[0])
        b = tuple(int(x) for x in coords[1])
        cells = _diag_cells(a, b)
        for cell in cells:
            line_lookup.setdefault(cell, set()).add(idx)
        endpoint_forbidden.add(a)
        endpoint_forbidden.add(b)
        for endpoint in (a, b):
            endpoint_forbidden.update(_neighbors4(*endpoint))

    grey_components = _connected_components(grid == GREY)
    if len(grey_components) < len(colors) + 2:
        return False

    untouched = 0
    multi = 0
    single_hits = set()
    for comp in grey_components:
        comp_set = {tuple(int(x) for x in rc) for rc in comp}
        if comp_set & endpoint_forbidden:
            return False
        hits = _line_hit_ids(comp_set, line_lookup)
        if len(hits) == 0:
            untouched += 1
        elif len(hits) == 1:
            single_hits |= hits
        else:
            multi += 1
    if untouched < 1 or multi < 1:
        return False
    if single_hits != set(range(len(colors))):
        return False
    if np.array_equal(grid, expected):
        return False
    if len(np.unique(expected)) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    lines = _sample_lines(height, width)
    if lines is None:
        return None

    line_lookup: dict[tuple[int, int], set[int]] = {}
    endpoint_forbidden = set()
    for idx, line in enumerate(lines):
        for cell in line["cells"]:
            line_lookup.setdefault(cell, set()).add(idx)
        for endpoint in line["endpoints"]:
            endpoint_forbidden.add(endpoint)
            endpoint_forbidden.update(_neighbors4(*endpoint))

    used_grey: set[tuple[int, int]] = set()
    grey_components: list[set[tuple[int, int]]] = []

    for idx in range(len(lines)):
        comp = _place_single_component(
            idx,
            lines,
            (height, width),
            line_lookup,
            used_grey,
            endpoint_forbidden,
        )
        if comp is None:
            return None
        grey_components.append(comp)
        used_grey |= comp

    multi = _place_multi_component(
        lines,
        (height, width),
        line_lookup,
        used_grey,
        endpoint_forbidden,
    )
    if multi is None:
        return None
    grey_components.append(multi)
    used_grey |= multi

    untouched = _place_untouched_component(
        (height, width),
        line_lookup,
        used_grey,
        endpoint_forbidden,
    )
    if untouched is None:
        return None
    grey_components.append(untouched)
    used_grey |= untouched

    if random.random() < 0.6:
        extra = _place_untouched_component(
            (height, width),
            line_lookup,
            used_grey,
            endpoint_forbidden,
        )
        if extra is not None:
            grey_components.append(extra)
            used_grey |= extra

    grid = _render_input(height, width, lines, grey_components)
    expected = _manual_output(height, width, lines, grey_components, line_lookup)
    if not _semantic_check(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
