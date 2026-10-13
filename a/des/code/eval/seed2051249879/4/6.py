from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in (
                    (cr - 1, cc),
                    (cr + 1, cc),
                    (cr, cc - 1),
                    (cr, cc + 1),
                ):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    assert indices
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _intervals_between(bands: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(bands[i][1] + 1, bands[i + 1][0] - 1) for i in range(len(bands) - 1)]


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int | None:
    for idx, (start, end) in enumerate(bands):
        if start <= pos <= end:
            return idx
    return None


def _cell_index(pos: int, bands: list[tuple[int, int]]) -> int | None:
    for idx in range(len(bands) - 1):
        start = bands[idx][1] + 1
        end = bands[idx + 1][0] - 1
        if start <= pos <= end:
            return idx
    return None


def _cell_centers(cells: list[tuple[int, int]]) -> list[float]:
    return [(start + end) / 2.0 for start, end in cells]


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    comps = _components4(g != bg)
    assert comps, "expected a non-background grid board"
    board = max(comps, key=len)
    rs = [r for r, _c in board]
    cs = [c for _r, c in board]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    board_mask = np.zeros_like(g, dtype=bool)
    for r, c in board:
        board_mask[r, c] = True
    submask = board_mask[r0 : r1 + 1, c0 : c1 + 1]

    full_rows = [r0 + rr for rr in range(submask.shape[0]) if np.all(submask[rr, :])]
    full_cols = [c0 + cc for cc in range(submask.shape[1]) if np.all(submask[:, cc])]
    assert len(full_rows) >= 2 and len(full_cols) >= 2

    row_bands = _group_runs(full_rows)
    col_bands = _group_runs(full_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2

    row_thicknesses = {end - start + 1 for start, end in row_bands}
    col_thicknesses = {end - start + 1 for start, end in col_bands}
    assert len(row_thicknesses) == 1 and len(col_thicknesses) == 1
    line_thickness = row_thicknesses.pop()
    assert col_thicknesses.pop() == line_thickness

    row_cells = _intervals_between(row_bands)
    col_cells = _intervals_between(col_bands)
    row_sizes = {end - start + 1 for start, end in row_cells}
    col_sizes = {end - start + 1 for start, end in col_cells}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    cell_size = row_sizes.pop()
    assert col_sizes.pop() == cell_size
    assert cell_size >= 1

    board_values = [int(g[r, c]) for r, c in board]
    line_color = Counter(board_values).most_common(1)[0][0]
    arrow_pixels = sorted((r, c) for r, c in board if int(g[r, c]) != line_color)

    return {
        "bg": int(bg),
        "line_color": int(line_color),
        "line_thickness": int(line_thickness),
        "cell_size": int(cell_size),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "row_cells": row_cells,
        "col_cells": col_cells,
        "row_centers": _cell_centers(row_cells),
        "col_centers": _cell_centers(col_cells),
        "arrow_pixels": arrow_pixels,
    }


def _decode_arrows(grid: np.ndarray, structure: dict) -> list[tuple[int, int, int, int]]:
    row_bands = structure["row_bands"]
    col_bands = structure["col_bands"]
    row_centers = structure["row_centers"]
    col_centers = structure["col_centers"]
    arrows: list[tuple[int, int, int, int]] = []

    for r, c in structure["arrow_pixels"]:
        rb = _band_index(int(r), row_bands)
        cb = _band_index(int(c), col_bands)
        if rb is not None and cb is None:
            cell_col = _cell_index(int(c), col_bands)
            assert cell_col is not None
            assert 0 < rb < len(row_bands) - 1
            above = rb - 1
            below = rb
            d_above = abs(float(r) - row_centers[above])
            d_below = abs(float(r) - row_centers[below])
            assert d_above != d_below, "horizontal arrow is ambiguous"
            if d_above < d_below:
                arrows.append((above, cell_col, below, cell_col))
            else:
                arrows.append((below, cell_col, above, cell_col))
            continue

        if cb is not None and rb is None:
            cell_row = _cell_index(int(r), row_bands)
            assert cell_row is not None
            assert 0 < cb < len(col_bands) - 1
            left = cb - 1
            right = cb
            d_left = abs(float(c) - col_centers[left])
            d_right = abs(float(c) - col_centers[right])
            assert d_left != d_right, "vertical arrow is ambiguous"
            if d_left < d_right:
                arrows.append((cell_row, left, cell_row, right))
            else:
                arrows.append((cell_row, right, cell_row, left))
            continue

        raise AssertionError("arrow pixel must lie on exactly one grid-edge band")

    return arrows


def _project_pattern(
    out: np.ndarray,
    src_patch: np.ndarray,
    dst_rows: tuple[int, int],
    dst_cols: tuple[int, int],
    bg: int,
) -> None:
    r0, r1 = dst_rows
    c0, c1 = dst_cols
    dst_patch = out[r0 : r1 + 1, c0 : c1 + 1]
    paint = (src_patch != bg) & (dst_patch == bg)
    dst_patch[paint] = src_patch[paint]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each cell's interior stamp to the adjacent cell indicated by its edge arrow.

    The board is partitioned into equal square cells by a connected grid-line lattice.
    Arrow pixels sit on the grid edges. Each arrow causes its source cell's interior
    non-background pixels to be stamped onto the adjacent target cell at matching
    local coordinates, without overwriting existing non-background pixels.
    Arrow pixels are erased back to the grid-line color in the output.
    """
    g = np.asarray(grid, dtype=int)
    structure = _extract_structure(g)
    bg = structure["bg"]
    line_color = structure["line_color"]
    row_cells = structure["row_cells"]
    col_cells = structure["col_cells"]

    out = g.copy()
    for r, c in structure["arrow_pixels"]:
        out[r, c] = line_color

    arrows = _decode_arrows(g, structure)
    for src_r, src_c, dst_r, dst_c in arrows:
        src_rows = row_cells[src_r]
        src_cols = col_cells[src_c]
        dst_rows = row_cells[dst_r]
        dst_cols = col_cells[dst_c]
        src_patch = g[src_rows[0] : src_rows[1] + 1, src_cols[0] : src_cols[1] + 1]
        _project_pattern(out, src_patch, dst_rows, dst_cols, bg)

    return out


# --------------- generator ---------------

_MAX_ATTEMPTS = 400
_CELL_SIZE_OPTIONS = (5, 6)
_LINE_THICKNESS_OPTIONS = (2, 3)
_ROW_COUNT_RANGE = (2, 4)
_COL_COUNT_RANGE = (2, 4)
_MARGIN_RANGE = (1, 3)
_FAMILY_NAMES = ("up", "down", "left", "right", "center")


def _valid_dirs(n_rows: int, n_cols: int, cell: tuple[int, int]) -> list[str]:
    r, c = cell
    dirs: list[str] = []
    if r > 0:
        dirs.append("up")
    if r + 1 < n_rows:
        dirs.append("down")
    if c > 0:
        dirs.append("left")
    if c + 1 < n_cols:
        dirs.append("right")
    return dirs


def _move(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = cell
    if direction == "up":
        return (r - 1, c)
    if direction == "down":
        return (r + 1, c)
    if direction == "left":
        return (r, c - 1)
    assert direction == "right"
    return (r, c + 1)


def _dir_toward(src: tuple[int, int], dst: tuple[int, int]) -> str:
    sr, sc = src
    dr, dc = dst
    if dr == sr - 1 and dc == sc:
        return "up"
    if dr == sr + 1 and dc == sc:
        return "down"
    if dr == sr and dc == sc - 1:
        return "left"
    assert dr == sr and dc == sc + 1
    return "right"


def _mid_choices(start: int, end: int) -> list[int]:
    size = end - start + 1
    if size % 2 == 1:
        return [start + size // 2]
    return [start + size // 2 - 1, start + size // 2]


def _pattern_mask(cell_size: int, family: str) -> np.ndarray:
    mask = np.zeros((cell_size, cell_size), dtype=bool)
    inner = cell_size - 2
    assert inner in (3, 4)
    if inner == 3:
        lookup = {
            "up": [(0, 1)],
            "down": [(2, 1)],
            "left": [(1, 0)],
            "right": [(1, 2)],
            "center": [(1, 1)],
        }
    else:
        lookup = {
            "up": [(0, 1), (0, 2)],
            "down": [(3, 1), (3, 2)],
            "left": [(1, 0), (2, 0)],
            "right": [(1, 3), (2, 3)],
            "center": [(1, 1), (1, 2), (2, 1), (2, 2)],
        }
    for rr, cc in lookup[family]:
        mask[rr + 1, cc + 1] = True
    return mask


def _board_geometry(
    top: int,
    left: int,
    n_rows: int,
    n_cols: int,
    cell_size: int,
    line_thickness: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    row_bands = []
    col_bands = []
    for idx in range(n_rows + 1):
        start = top + idx * (cell_size + line_thickness)
        row_bands.append((start, start + line_thickness - 1))
    for idx in range(n_cols + 1):
        start = left + idx * (cell_size + line_thickness)
        col_bands.append((start, start + line_thickness - 1))
    row_cells = _intervals_between(row_bands)
    col_cells = _intervals_between(col_bands)
    return row_bands, col_bands, row_cells, col_cells


def _place_grid(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    line_color: int,
) -> None:
    r0 = row_bands[0][0]
    r1 = row_bands[-1][1]
    c0 = col_bands[0][0]
    c1 = col_bands[-1][1]
    for rs, re in row_bands:
        grid[rs : re + 1, c0 : c1 + 1] = line_color
    for cs, ce in col_bands:
        grid[r0 : r1 + 1, cs : ce + 1] = line_color


def _arrow_pixel(
    cell: tuple[int, int],
    direction: str,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
) -> tuple[int, int]:
    r, c = cell
    if direction == "up":
        band = row_bands[r]
        rr = band[1]
        cc = random.choice(_mid_choices(*col_cells[c]))
        return (rr, cc)
    if direction == "down":
        band = row_bands[r + 1]
        rr = band[0]
        cc = random.choice(_mid_choices(*col_cells[c]))
        return (rr, cc)
    if direction == "left":
        band = col_bands[c]
        rr = random.choice(_mid_choices(*row_cells[r]))
        cc = band[1]
        return (rr, cc)
    assert direction == "right"
    band = col_bands[c + 1]
    rr = random.choice(_mid_choices(*row_cells[r]))
    cc = band[0]
    return (rr, cc)


def _families_for_cells(
    n_rows: int,
    n_cols: int,
    arrows: dict[tuple[int, int], str],
    indegree: dict[tuple[int, int], int],
    focus_target: tuple[int, int],
) -> dict[tuple[int, int], str]:
    families: dict[tuple[int, int], str] = {}
    for r in range(n_rows):
        for c in range(n_cols):
            cell = (r, c)
            if cell == focus_target:
                families[cell] = "center"
            elif cell in arrows:
                families[cell] = arrows[cell]
            elif indegree.get(cell, 0) > 0:
                families[cell] = random.choice(("center", "up", "down", "left", "right"))
            else:
                families[cell] = random.choice(_FAMILY_NAMES)
    return families


def _build_candidate() -> np.ndarray | None:
    cell_size = random.choice(_CELL_SIZE_OPTIONS)
    line_thickness = random.choice(_LINE_THICKNESS_OPTIONS)
    n_rows = random.randint(*_ROW_COUNT_RANGE)
    n_cols = random.randint(*_COL_COUNT_RANGE)

    top_margin = random.randint(*_MARGIN_RANGE)
    left_margin = random.randint(*_MARGIN_RANGE)
    bottom_margin = random.randint(*_MARGIN_RANGE)
    right_margin = random.randint(*_MARGIN_RANGE)

    board_h = (n_rows + 1) * line_thickness + n_rows * cell_size
    board_w = (n_cols + 1) * line_thickness + n_cols * cell_size
    height = top_margin + board_h + bottom_margin
    width = left_margin + board_w + right_margin

    palette = list(range(10))
    bg, line_color, arrow_color = random.sample(palette, 3)
    stamp_colors = [color for color in palette if color not in {bg, line_color, arrow_color}]

    grid = np.full((height, width), bg, dtype=int)
    row_bands, col_bands, row_cells, col_cells = _board_geometry(
        top_margin,
        left_margin,
        n_rows,
        n_cols,
        cell_size,
        line_thickness,
    )
    _place_grid(grid, row_bands, col_bands, line_color)

    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    focus_target = random.choice([cell for cell in cells if len(_valid_dirs(n_rows, n_cols, cell)) >= 2])
    neighbor_sources = [_move(focus_target, d) for d in _valid_dirs(n_rows, n_cols, focus_target)]
    incoming_count = 2 if len(neighbor_sources) == 2 else random.randint(2, min(4, len(neighbor_sources)))
    random.shuffle(neighbor_sources)
    chosen_sources = neighbor_sources[:incoming_count]

    arrows: dict[tuple[int, int], str] = {}
    indegree: dict[tuple[int, int], int] = {}
    for src in chosen_sources:
        direction = _dir_toward(src, focus_target)
        arrows[src] = direction
        indegree[focus_target] = indegree.get(focus_target, 0) + 1

    extra_cells = [cell for cell in cells if cell not in arrows and cell != focus_target]
    random.shuffle(extra_cells)
    extra_limit = random.randint(0, max(1, len(extra_cells) // 2))
    for cell in extra_cells:
        if len(arrows) >= incoming_count + extra_limit:
            break
        if random.random() < 0.45:
            direction = random.choice(_valid_dirs(n_rows, n_cols, cell))
            arrows[cell] = direction
            dst = _move(cell, direction)
            indegree[dst] = indegree.get(dst, 0) + 1

    families = _families_for_cells(n_rows, n_cols, arrows, indegree, focus_target)

    for r in range(n_rows):
        for c in range(n_cols):
            family = families[(r, c)]
            color = random.choice(stamp_colors)
            mask = _pattern_mask(cell_size, family)
            rs, re = row_cells[r]
            cs, ce = col_cells[c]
            patch = grid[rs : re + 1, cs : ce + 1]
            patch[mask] = color

    arrow_pixels: set[tuple[int, int]] = set()
    for cell, direction in arrows.items():
        pos = _arrow_pixel(cell, direction, row_bands, col_bands, row_cells, col_cells)
        if pos in arrow_pixels:
            return None
        arrow_pixels.add(pos)
        grid[pos] = arrow_color

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    interior_changed = False
    for rs, re in row_cells:
        for cs, ce in col_cells:
            if not np.array_equal(out[rs : re + 1, cs : ce + 1], grid[rs : re + 1, cs : ce + 1]):
                interior_changed = True
                break
        if interior_changed:
            break
    if not interior_changed:
        return None

    if indegree.get(focus_target, 0) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid grid after {_MAX_ATTEMPTS} attempts")
