from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
MAGENTA = 6
ORANGE = 7

DIR_BY_COLOR = {
    RED: (0, -1),
    BLUE: (0, 1),
    ORANGE: (-1, 0),
    GREEN: (1, 0),
}

TEMPLATE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
)

CELL_COUNT_CHOICES = (4, 5)
CELL_SIZE_CHOICES = (3, 4)
MARKER_COUNT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 500


def _groups(indices: list[int]) -> list[tuple[int, int]]:
    assert indices, "expected separator lines"
    spans: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        spans.append((start, prev))
        start = prev = idx
    spans.append((start, prev))
    return spans


def _spans_between(line_groups: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line_start, line_end in line_groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))
    assert spans, "expected cell spans between separators"
    return spans


def _cell_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_spans = _spans_between(_groups(row_lines), int(grid.shape[0]))
    col_spans = _spans_between(_groups(col_lines), int(grid.shape[1]))
    row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "all cells must have a common size"
    cell_size = row_sizes.pop()
    assert cell_size == col_sizes.pop(), "cells must be square"
    assert len(row_spans) == len(col_spans) and len(row_spans) in CELL_COUNT_CHOICES
    return row_spans, col_spans, int(cell_size)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected4(coords: list[tuple[int, int]]) -> bool:
    todo = [coords[0]]
    seen = {coords[0]}
    coord_set = set(coords)
    while todo:
        r, c = todo.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == len(coord_set)


def _classify_cell(block: np.ndarray) -> tuple[str, dict]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(block != BLACK)]
    if not coords:
        return "empty", {}

    colors = {int(block[r, c]) for r, c in coords}
    assert len(colors) == 1, "each cell must contain at most one non-black color"
    color = colors.pop()

    if len(coords) == 1:
        assert color in DIR_BY_COLOR, "single-pixel cells must be direction markers"
        return "marker", {"color": color, "pixel": coords[0]}

    assert color == MAGENTA, "multi-pixel cell must be the magenta template"
    assert 2 <= len(coords) <= 4, "template size must be 2..4 pixels"
    assert _connected4(coords), "template must be 4-connected"
    return "template", {"coords": tuple(sorted(coords))}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the unique magenta template into marker cells using color-coded direction and cell-distance-coded magnitude.

    Input structure:
    - Full grey separator rows and columns partition the black canvas into a 4x4 or
      5x5 array of equal square cells.
    - Exactly one cell contains a 4-connected multi-pixel magenta template sprite.
    - Several other cells contain exactly one colored marker pixel: red means left,
      blue means right, orange means up, and green means down.

    Rule:
    - Compress the divider grid into cell coordinates and locate the template cell.
    - For each marker cell, decode its direction from the marker color and compute
      the Manhattan distance from that marker cell to the template cell in cell units.
    - Multiply the unit direction by that distance to obtain an internal pixel
      offset, then stamp the template inside the marker cell after shifting every
      template pixel by that offset, clipping any shifted pixels that leave the cell.
    - Clear the original marker pixel unless the magenta stamp occupies that position;
      keep the original template cell, empty cells, and grey separators unchanged.

    Key insight:
    - The displacement is factorized across two visible cues: marker color supplies
      only direction, while the marker cell's grid distance from the template cell
      supplies only magnitude.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans, cell_size = _cell_spans(g)
    template_cell: tuple[int, int] | None = None
    template_coords: tuple[tuple[int, int], ...] | None = None
    markers: list[tuple[int, int, int]] = []

    for gr, (r0, r1) in enumerate(row_spans):
        for gc, (c0, c1) in enumerate(col_spans):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            kind, payload = _classify_cell(block)
            if kind == "template":
                assert template_cell is None, "expected exactly one template cell"
                template_cell = (gr, gc)
                template_coords = payload["coords"]
            elif kind == "marker":
                markers.append((gr, gc, int(payload["color"])))

    assert template_cell is not None and template_coords is not None, "template cell missing"
    assert len(markers) >= 3, "expected at least three markers"
    assert len({color for _r, _c, color in markers}) >= 2, "expected at least two directions"

    tr, tc = template_cell
    out = g.copy()
    for mr, mc, color in markers:
        distance = abs(mr - tr) + abs(mc - tc)
        assert 1 <= distance <= 3, "marker distance must be 1..3"
        dr, dc = DIR_BY_COLOR[color]
        shift_r = dr * distance
        shift_c = dc * distance

        r0, r1 = row_spans[mr]
        c0, c1 = col_spans[mc]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        for pr, pc in template_coords:
            rr = pr + shift_r
            cc = pc + shift_c
            if 0 <= rr < cell_size and 0 <= cc < cell_size:
                out[r0 + rr, c0 + cc] = MAGENTA

    return out


def _shape_fits(shape: tuple[tuple[int, int], ...], cell_size: int) -> bool:
    r0, c0, r1, c1 = _bbox(list(shape))
    return r1 < cell_size and c1 < cell_size


def _shift_coords(
    coords: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in coords))


def _full_fit(
    coords: tuple[tuple[int, int], ...],
    cell_size: int,
    dr: int,
    dc: int,
) -> bool:
    return all(0 <= r + dr < cell_size and 0 <= c + dc < cell_size for r, c in coords)


def _cell_origin(cell_size: int, gr: int, gc: int) -> tuple[int, int]:
    stride = cell_size + 1
    return gr * stride, gc * stride


def _render_input(
    n_cells: int,
    cell_size: int,
    template_cell: tuple[int, int],
    template_coords: tuple[tuple[int, int], ...],
    markers: list[dict],
) -> np.ndarray:
    side = n_cells * cell_size + (n_cells - 1)
    grid = np.zeros((side, side), dtype=int)
    stride = cell_size + 1
    for cut in range(1, n_cells):
        idx = cut * stride - 1
        grid[idx, :] = GREY
        grid[:, idx] = GREY

    tr0, tc0 = _cell_origin(cell_size, *template_cell)
    for r, c in template_coords:
        grid[tr0 + r, tc0 + c] = MAGENTA

    for marker in markers:
        mr0, mc0 = _cell_origin(cell_size, marker["gr"], marker["gc"])
        pr, pc = marker["pixel"]
        grid[mr0 + pr, mc0 + pc] = marker["color"]

    return grid


def _expected_output(
    input_grid: np.ndarray,
    cell_size: int,
    template_coords: tuple[tuple[int, int], ...],
    markers: list[dict],
) -> np.ndarray:
    out = input_grid.copy()
    for marker in markers:
        mr0, mc0 = _cell_origin(cell_size, marker["gr"], marker["gc"])
        out[mr0 : mr0 + cell_size, mc0 : mc0 + cell_size] = BLACK
        dr, dc = DIR_BY_COLOR[marker["color"]]
        shift_r = dr * marker["distance"]
        shift_c = dc * marker["distance"]
        for r, c in template_coords:
            out[mr0 + r + shift_r, mc0 + c + shift_c] = MAGENTA
    return out


def _candidate_markers(
    n_cells: int,
    cell_size: int,
    template_cell: tuple[int, int],
    template_coords: tuple[tuple[int, int], ...],
) -> list[dict]:
    tr, tc = template_cell
    candidates: list[dict] = []
    for gr in range(n_cells):
        for gc in range(n_cells):
            if (gr, gc) == template_cell:
                continue
            distance = abs(gr - tr) + abs(gc - tc)
            if not (1 <= distance <= 3):
                continue
            natural_dirs = set()
            if gr < tr:
                natural_dirs.add((-1, 0))
            elif gr > tr:
                natural_dirs.add((1, 0))
            if gc < tc:
                natural_dirs.add((0, -1))
            elif gc > tc:
                natural_dirs.add((0, 1))

            for color, (dr, dc) in DIR_BY_COLOR.items():
                shift_r = dr * distance
                shift_c = dc * distance
                if not _full_fit(template_coords, cell_size, shift_r, shift_c):
                    continue
                stamp = _shift_coords(template_coords, shift_r, shift_c)
                candidates.append(
                    {
                        "gr": gr,
                        "gc": gc,
                        "color": color,
                        "distance": distance,
                        "shift": (shift_r, shift_c),
                        "stamp": stamp,
                        "diagonal": gr != tr and gc != tc,
                        "mismatch": (dr, dc) not in natural_dirs,
                    }
                )
    return candidates


def _pair_flags(items: list[dict]) -> tuple[bool, bool]:
    same_dir_diff_dist = any(
        a["color"] == b["color"] and a["distance"] != b["distance"]
        for a, b in combinations(items, 2)
    )
    same_dist_diff_dir = any(
        a["distance"] == b["distance"] and a["color"] != b["color"]
        for a, b in combinations(items, 2)
    )
    return same_dir_diff_dist, same_dist_diff_dir


def _valid_marker_subset(selected: list[dict], pool: list[dict]) -> bool:
    if len(selected) < 3:
        return False
    if len({(item["gr"], item["gc"]) for item in selected}) != len(selected):
        return False
    if len({item["color"] for item in selected}) < 2:
        return False
    if len({item["distance"] for item in selected}) < 2:
        return False
    if len({item["shift"] for item in selected}) < 2:
        return False
    if len({item["stamp"] for item in selected}) < 2:
        return False
    if not any(item["diagonal"] for item in selected):
        return False
    if not any(item["mismatch"] for item in selected):
        return False

    selected_pair_flags = _pair_flags(selected)
    pool_pair_flags = _pair_flags(pool)
    if pool_pair_flags[0] and not selected_pair_flags[0]:
        return False
    if pool_pair_flags[1] and not selected_pair_flags[1]:
        return False
    return True


def _choose_markers(pool: list[dict]) -> list[dict] | None:
    if not pool:
        return None
    by_cell: dict[tuple[int, int], list[dict]] = {}
    for item in pool:
        by_cell.setdefault((item["gr"], item["gc"]), []).append(item)
    cells = list(by_cell)
    if len(cells) < MARKER_COUNT_RANGE[0]:
        return None

    for _ in range(400):
        count = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(cells)))
        chosen_cells = random.sample(cells, count)
        selected = [random.choice(by_cell[cell]).copy() for cell in chosen_cells]
        if not _valid_marker_subset(selected, pool):
            continue
        return selected
    return None


def _assign_marker_pixels(markers: list[dict], cell_size: int) -> None:
    all_pixels = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    for item in markers:
        stamp = list(item["stamp"])
        stamp_set = set(stamp)
        uncovered = [pixel for pixel in all_pixels if pixel not in stamp_set]
        prefer_cover = (item["gr"] + item["gc"] + item["distance"]) % 2 == 0
        if prefer_cover or not uncovered:
            item["pixel"] = stamp[0]
        else:
            item["pixel"] = uncovered[0]


def _sample_template(cell_size: int) -> tuple[tuple[int, int], ...]:
    shapes = [shape for shape in TEMPLATE_SHAPES if _shape_fits(shape, cell_size)]
    shape = random.choice(shapes)
    _r0, _c0, r1, c1 = _bbox(list(shape))
    top = random.randint(0, cell_size - 1 - r1)
    left = random.randint(0, cell_size - 1 - c1)
    return _shift_coords(shape, top, left)


def _attempt_generate() -> np.ndarray | None:
    n_cells = random.choice(CELL_COUNT_CHOICES)
    cell_size = random.choice(CELL_SIZE_CHOICES)
    for _ in range(80):
        template_coords = _sample_template(cell_size)
        template_cell = (random.randrange(n_cells), random.randrange(n_cells))

        pool = _candidate_markers(n_cells, cell_size, template_cell, template_coords)
        markers = _choose_markers(pool)
        if markers is None:
            continue
        _assign_marker_pixels(markers, cell_size)

        grid = _render_input(n_cells, cell_size, template_cell, template_coords, markers)
        expected = _expected_output(grid, cell_size, template_coords, markers)

        if not np.array_equal(main(grid), expected):
            continue

        # Independent semantic checks on the sampled abstract instance.
        if len({marker["color"] for marker in markers}) < 2:
            continue
        if len({marker["distance"] for marker in markers}) < 2:
            continue
        if not any(marker["diagonal"] for marker in markers):
            continue
        if not any(marker["mismatch"] for marker in markers):
            continue
        if len({marker["stamp"] for marker in markers}) < 2:
            continue

        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
