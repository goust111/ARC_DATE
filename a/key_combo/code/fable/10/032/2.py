from __future__ import annotations

from collections import Counter

import numpy as np


Coord = tuple[int, int]
Cell = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _mode_count(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _parse_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    legend_rows = np.flatnonzero(g[:, 0] != bg)
    assert legend_rows.size >= 2, "expected a left-edge legend"
    legend_colors = [int(g[r, 0]) for r in legend_rows]
    assert len(legend_colors) == len(set(legend_colors)), "legend colors must be distinct"
    assert np.all(np.diff(legend_rows) == 1), "legend must be a contiguous color bar"

    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    lattice_candidates = [c for c in non_bg if c not in set(legend_colors)]
    assert len(lattice_candidates) == 1, "expected exactly one lattice color besides the legend"
    lattice = lattice_candidates[0]

    lattice_mask = g == lattice
    r0, c0, r1, c1 = _bbox(lattice_mask)
    sub = lattice_mask[r0 : r1 + 1, c0 : c1 + 1]
    row_counts = np.sum(sub, axis=1)
    col_counts = np.sum(sub, axis=0)

    row_base = _mode_count(row_counts)
    col_base = _mode_count(col_counts)
    line_rows = [r0 + i for i, n in enumerate(row_counts) if int(n) > row_base]
    line_cols = [c0 + i for i, n in enumerate(col_counts) if int(n) > col_base]

    assert len(line_rows) >= 3 and len(line_cols) >= 3, "expected a framed lattice with internal lines"
    assert line_rows[0] == r0 and line_rows[-1] == r1
    assert line_cols[0] == c0 and line_cols[-1] == c1

    cells: list[tuple[int, int, int, int]] = []
    for ra, rb in zip(line_rows[:-1], line_rows[1:]):
        assert rb - ra >= 2
        for ca, cb in zip(line_cols[:-1], line_cols[1:]):
            assert cb - ca >= 2
            cells.append((ra + 1, rb - 1, ca + 1, cb - 1))

    return {
        "bg": bg,
        "legend_colors": legend_colors,
        "lattice": lattice,
        "line_rows": line_rows,
        "line_cols": line_cols,
        "cells": cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed hall of a broken lattice with the legend color indexed by how many lattice cells that hall spans.

    Input structure:
    - A single-color rectangular lattice frame occupies the main canvas area, with unequal row heights and column widths defining non-uniform cells.
    - Some interior wall segments are missing, so several adjacent cells merge into larger 4-connected background halls.
    - A detached one-column color legend sits on the left edge; its cells are distinct colors ordered from top to bottom.

    Rule:
    - Keep the lattice pixels and the legend bar unchanged.
    - For each 4-connected background hall enclosed by the lattice, count how many underlying lattice cell positions it reaches.
    - If a hall spans k cells, recolor that entire hall with the color from the k-th legend cell counted from the top.

    Key insight:
    - The measurement unit is the lattice-defined cell slot, not pixel area: irregular cell sizes and selective wall breaks force you to recover the hidden cell partition before coloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    parsed = _parse_structure(g)
    bg = parsed["bg"]
    legend_colors = parsed["legend_colors"]
    cells = parsed["cells"]

    out = g.copy()
    for comp in _components(g == bg):
        if _touches_boundary(comp, g.shape):
            continue
        comp_mask = np.zeros(g.shape, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True

        cell_count = 0
        for r0, r1, c0, c1 in cells:
            if np.any(comp_mask[r0 : r1 + 1, c0 : c1 + 1]):
                cell_count += 1

        assert 1 <= cell_count <= len(legend_colors), "hall size exceeds legend"
        out[comp[:, 0], comp[:, 1]] = legend_colors[cell_count - 1]

    return out


LAYOUTS = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2)]
BACKGROUND = 0
SIZE_RANGE = (2, 5)
LATTICE_LEFT_RANGE = (3, 6)
EXTRA_BOTTOM_RANGE = (0, 3)
MAX_ATTEMPTS = 500
OPENING_TRIES = 80


def _sample_sizes(rng: np.random.Generator, count: int) -> list[int]:
    while True:
        sizes = rng.integers(SIZE_RANGE[0], SIZE_RANGE[1] + 1, size=count).tolist()
        if len(set(sizes)) >= 2:
            return [int(v) for v in sizes]


def _line_positions(spans: list[int]) -> list[int]:
    positions = [0]
    cur = 0
    for span in spans:
        cur += int(span) + 1
        positions.append(cur)
    return positions


def _enumerate_walls(
    n_rows: int,
    n_cols: int,
    row_heights: list[int],
    col_widths: list[int],
    line_rows: list[int],
    line_cols: list[int],
    lattice_left: int,
) -> list[dict]:
    walls: list[dict] = []
    for r in range(n_rows - 1):
        for c in range(n_cols):
            walls.append(
                {
                    "kind": "h",
                    "a": (r, c),
                    "b": (r + 1, c),
                    "line": line_rows[r + 1],
                    "start": lattice_left + line_cols[c] + 1,
                    "span": int(col_widths[c]),
                }
            )
    for r in range(n_rows):
        for c in range(n_cols - 1):
            walls.append(
                {
                    "kind": "v",
                    "a": (r, c),
                    "b": (r, c + 1),
                    "line": lattice_left + line_cols[c + 1],
                    "start": line_rows[r] + 1,
                    "span": int(row_heights[r]),
                }
            )
    return walls


def _cell_components(n_rows: int, n_cols: int, opened: list[dict]) -> list[list[Cell]]:
    neighbors = { (r, c): [] for r in range(n_rows) for c in range(n_cols) }
    for wall in opened:
        a = tuple(wall["a"])
        b = tuple(wall["b"])
        neighbors[a].append(b)
        neighbors[b].append(a)

    comps: list[list[Cell]] = []
    seen: set[Cell] = set()
    for cell in neighbors:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        comp: list[Cell] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in neighbors[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _apply_opening(grid: np.ndarray, wall: dict, rng: np.random.Generator) -> list[Coord]:
    span = int(wall["span"])
    gap_len = int(rng.integers(1, min(2, span) + 1))
    offset = int(rng.integers(0, span - gap_len + 1))
    coords: list[Coord] = []
    if wall["kind"] == "h":
        r = int(wall["line"])
        c0 = int(wall["start"]) + offset
        for c in range(c0, c0 + gap_len):
            grid[r, c] = BACKGROUND
            coords.append((r, c))
    else:
        c = int(wall["line"])
        r0 = int(wall["start"]) + offset
        for r in range(r0, r0 + gap_len):
            grid[r, c] = BACKGROUND
            coords.append((r, c))
    return coords


def _room_cells(
    row_heights: list[int],
    col_widths: list[int],
    line_rows: list[int],
    line_cols: list[int],
    lattice_left: int,
) -> dict[Cell, tuple[int, int, int, int]]:
    cells: dict[Cell, tuple[int, int, int, int]] = {}
    for r in range(len(row_heights)):
        for c in range(len(col_widths)):
            cells[(r, c)] = (
                line_rows[r] + 1,
                line_rows[r + 1] - 1,
                lattice_left + line_cols[c] + 1,
                lattice_left + line_cols[c + 1] - 1,
            )
    return cells


def _render_expected(
    grid: np.ndarray,
    legend_colors: list[int],
    room_components: list[list[Cell]],
    cell_boxes: dict[Cell, tuple[int, int, int, int]],
    opening_pixels: dict[tuple[Cell, Cell], list[Coord]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = grid.copy()
    room_stats: list[tuple[int, int]] = []
    room_of: dict[Cell, int] = {}
    for idx, comp in enumerate(room_components):
        for cell in comp:
            room_of[cell] = idx

    for idx, comp in enumerate(room_components):
        size = len(comp)
        color = int(legend_colors[size - 1])
        area = 0
        for cell in comp:
            r0, r1, c0, c1 = cell_boxes[cell]
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            area += (r1 - r0 + 1) * (c1 - c0 + 1)
        for key, pixels in opening_pixels.items():
            a, b = key
            if room_of[a] == idx and room_of[b] == idx:
                for r, c in pixels:
                    out[r, c] = color
                area += len(pixels)
        room_stats.append((size, area))
    return out, room_stats


def _has_area_counterexample(room_stats: list[tuple[int, int]]) -> bool:
    for size_a, area_a in room_stats:
        for size_b, area_b in room_stats:
            if size_a > size_b and area_a < area_b:
                return True
    return False


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_rows, n_cols = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    total_cells = n_rows * n_cols

    row_heights = _sample_sizes(rng, n_rows)
    col_widths = _sample_sizes(rng, n_cols)
    line_rows = _line_positions(row_heights)
    line_cols = _line_positions(col_widths)

    lattice_h = line_rows[-1] + 1
    lattice_w = line_cols[-1] + 1
    lattice_left = int(rng.integers(LATTICE_LEFT_RANGE[0], LATTICE_LEFT_RANGE[1]))
    canvas_h = lattice_h + int(rng.integers(EXTRA_BOTTOM_RANGE[0], EXTRA_BOTTOM_RANGE[1]))
    canvas_h = max(canvas_h, total_cells + int(rng.integers(0, 3)))
    canvas_w = lattice_left + lattice_w
    if canvas_h > 30 or canvas_w > 30:
        return None

    palette = rng.choice(np.arange(1, 10), size=total_cells + 1, replace=False)
    lattice_color = int(palette[0])
    legend_colors = [int(v) for v in palette[1:]]

    grid = np.full((canvas_h, canvas_w), BACKGROUND, dtype=int)
    legend_top = int(rng.integers(0, canvas_h - total_cells + 1))
    for idx, color in enumerate(legend_colors):
        grid[legend_top + idx, 0] = color

    grid[0:lattice_h, lattice_left : lattice_left + lattice_w] = BACKGROUND
    for r in line_rows:
        grid[int(r), lattice_left : lattice_left + lattice_w] = lattice_color
    for c in line_cols:
        grid[0:lattice_h, lattice_left + int(c)] = lattice_color

    walls = _enumerate_walls(n_rows, n_cols, row_heights, col_widths, line_rows, line_cols, lattice_left)
    opened: list[dict] | None = None
    room_components: list[list[Cell]] | None = None
    for _ in range(OPENING_TRIES):
        open_count = int(rng.integers(1, len(walls)))
        chosen_idx = rng.choice(len(walls), size=open_count, replace=False)
        attempt_opened = [walls[int(i)] for i in chosen_idx]
        comps = _cell_components(n_rows, n_cols, attempt_opened)
        sizes = [len(comp) for comp in comps]
        if len(comps) < 2:
            continue
        if max(sizes) <= 1:
            continue
        if len(set(sizes)) < 2:
            continue
        opened = attempt_opened
        room_components = comps
        break
    if opened is None or room_components is None:
        return None

    opening_pixels: dict[tuple[Cell, Cell], list[Coord]] = {}
    for wall in opened:
        key = tuple(sorted((tuple(wall["a"]), tuple(wall["b"]))))  # type: ignore[arg-type]
        opening_pixels[key] = _apply_opening(grid, wall, rng)

    cell_boxes = _room_cells(row_heights, col_widths, line_rows, line_cols, lattice_left)
    expected, room_stats = _render_expected(grid, legend_colors, room_components, cell_boxes, opening_pixels)

    if not _has_area_counterexample(room_stats):
        return None
    if len(set(size for size, _ in room_stats)) < 2:
        return None

    try:
        parsed = _parse_structure(grid)
        if len(parsed["legend_colors"]) != total_cells:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        result = _build_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid lattice-hall coloring input")
