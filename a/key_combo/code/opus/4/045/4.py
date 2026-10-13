import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREEN = 3
SEP = 5

CELL_SIZE = 4
MAX_GEN_ATTEMPTS = 400


def _partition_spans(full_sep: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in np.flatnonzero(full_sep):
        idx = int(idx)
        if start <= idx - 1:
            spans.append((start, idx - 1))
        start = idx + 1
    if start <= len(full_sep) - 1:
        spans.append((start, len(full_sep) - 1))
    return spans


def _grid_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _partition_spans(np.all(grid == SEP, axis=1))
    col_spans = _partition_spans(np.all(grid == SEP, axis=0))
    assert len(row_spans) in (3, 4)
    assert len(col_spans) in (3, 4)
    return row_spans, col_spans


def _cell_view(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    cr: int,
    cc: int,
) -> np.ndarray:
    r0, r1 = row_spans[cr]
    c0, c1 = col_spans[cc]
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _marker_cell(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for cr in range(len(row_spans)):
        for cc in range(len(col_spans)):
            patch = _cell_view(grid, row_spans, col_spans, cr, cc)
            coords = np.argwhere(patch == GREEN)
            if coords.shape[0] != 4:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if r1 - r0 == 1 and c1 - c0 == 1 and np.all(patch[r0 : r1 + 1, c0 : c1 + 1] == GREEN):
                candidates.append((cr, cc))
    assert len(candidates) == 1
    return candidates[0]


def _flip_mask(mask: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = mask
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return out


def _place_red_mask(dst: np.ndarray, mask: np.ndarray) -> None:
    writable = mask & (dst == BG)
    dst[writable] = RED


def _facing_row(patch: np.ndarray, side: str) -> int | None:
    coords = np.argwhere(patch == RED)
    if coords.size == 0:
        return None
    rows = np.unique(coords[:, 0])
    if len(rows) != 1:
        return None
    cols = np.sort(coords[:, 1])
    if int(cols[-1]) - int(cols[0]) + 1 != len(cols):
        return None
    if side == "left" and int(cols[0]) == 0:
        return int(rows[0])
    if side == "right" and int(cols[-1]) == patch.shape[1] - 1:
        return int(rows[0])
    return None


def _facing_col(patch: np.ndarray, side: str) -> int | None:
    coords = np.argwhere(patch == RED)
    if coords.size == 0:
        return None
    cols = np.unique(coords[:, 1])
    if len(cols) != 1:
        return None
    rows = np.sort(coords[:, 0])
    if int(rows[-1]) - int(rows[0]) + 1 != len(rows):
        return None
    if side == "top" and int(rows[0]) == 0:
        return int(cols[0])
    if side == "bottom" and int(rows[-1]) == patch.shape[0] - 1:
        return int(cols[0])
    return None


def _cell_is_empty(patch: np.ndarray) -> bool:
    return bool(np.all(patch == BG))


def _apply_symmetry(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    marker_rc: tuple[int, int],
) -> np.ndarray:
    mr, mc = marker_rc
    out = grid.copy()
    sources: list[tuple[int, int, np.ndarray]] = []
    for cr in range(len(row_spans)):
        for cc in range(len(col_spans)):
            patch = _cell_view(grid, row_spans, col_spans, cr, cc)
            red_mask = patch == RED
            if red_mask.any():
                sources.append((cr, cc, red_mask.copy()))

    for cr, cc, red_mask in sources:
        targets = [
            (cr, 2 * mc - cc, False, True),
            (2 * mr - cr, cc, True, False),
            (2 * mr - cr, 2 * mc - cc, True, True),
        ]
        for tr, tc, flip_ud, flip_lr in targets:
            if not (0 <= tr < len(row_spans) and 0 <= tc < len(col_spans)):
                continue
            dst = _cell_view(out, row_spans, col_spans, tr, tc)
            transformed = _flip_mask(red_mask, flip_ud, flip_lr)
            assert transformed.shape == dst.shape
            _place_red_mask(dst, transformed)
    return out


def _bridge_segments(
    sym_grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    bridges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    n_rows = len(row_spans)
    n_cols = len(col_spans)

    for cr in range(n_rows):
        for left in range(n_cols):
            left_patch = _cell_view(sym_grid, row_spans, col_spans, cr, left)
            local_r = _facing_row(left_patch, "right")
            if local_r is None:
                continue
            for right in range(left + 1, n_cols):
                right_patch = _cell_view(sym_grid, row_spans, col_spans, cr, right)
                other_r = _facing_row(right_patch, "left")
                if other_r != local_r:
                    continue
                if all(
                    _cell_is_empty(_cell_view(sym_grid, row_spans, col_spans, cr, cc))
                    for cc in range(left + 1, right)
                ):
                    rr = row_spans[cr][0] + local_r
                    bridges.append(((rr, col_spans[left][1]), (rr, col_spans[right][0])))

    for cc in range(n_cols):
        for top in range(n_rows):
            top_patch = _cell_view(sym_grid, row_spans, col_spans, top, cc)
            local_c = _facing_col(top_patch, "bottom")
            if local_c is None:
                continue
            for bottom in range(top + 1, n_rows):
                bottom_patch = _cell_view(sym_grid, row_spans, col_spans, bottom, cc)
                other_c = _facing_col(bottom_patch, "top")
                if other_c != local_c:
                    continue
                if all(
                    _cell_is_empty(_cell_view(sym_grid, row_spans, col_spans, rr, cc))
                    for rr in range(top + 1, bottom)
                ):
                    global_c = col_spans[cc][0] + local_c
                    bridges.append(((row_spans[top][1], global_c), (row_spans[bottom][0], global_c)))
    return bridges


def _paint_bridges(grid: np.ndarray, bridges: Iterable[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    out = grid.copy()
    for (r0, c0), (r1, c1) in bridges:
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            out[r0, lo : hi + 1] = RED
        else:
            lo, hi = sorted((r0, r1))
            out[lo : hi + 1, c0] = RED
    return out


def _solve_with_debug(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _grid_structure(g)
    marker_rc = _marker_cell(g, row_spans, col_spans)
    sym_grid = _apply_symmetry(g, row_spans, col_spans, marker_rc)
    bridges = _bridge_segments(sym_grid, row_spans, col_spans)
    out = _paint_bridges(sym_grid, bridges)
    debug = {
        "row_spans": row_spans,
        "col_spans": col_spans,
        "marker_rc": marker_rc,
        "sym_grid": sym_grid,
        "bridges": bridges,
    }
    return out, debug


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete red cell-local mirror copies around the marked grid cell, then connect newly facing endpoints across empty partitions.

    Input structure:
    - The background is black, and full grey rows and columns partition the canvas into a 3x3 or 4x4 grid of rectangular cells.
    - Exactly one cell contains a solid 2x2 green marker block, while other non-empty cells contain thin red horizontal or vertical segments.
    - Red content lies inside cells and grey separators lie only on the partition lines.

    Rule:
    - Use the marker cell as the symmetry center in cell coordinates.
    - For every cell containing red pixels, copy its red mask into the horizontal mirror, vertical mirror, and 180-degree mirror cells, applying the matching left-right, up-down, or combined flip inside the target cell, and only writing onto background pixels.
    - After symmetry completion, whenever two cells in the same partition row or column have red endpoints facing each other at the same local coordinate and every intermediate cell is empty, draw a straight red connection through the gap, overwriting separator pixels if crossed.
    - Keep the green marker unchanged.

    Key insight:
    - The grey grid defines both the object coordinate system and the empty-gap test, while the green marker cell defines the mirror center that creates the endpoint pairs needed for the later bridge fill.
    """
    out, _ = _solve_with_debug(grid)
    return out


def _orbit(cell: tuple[int, int], marker: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = cell
    mr, mc = marker
    return [
        (cr, cc),
        (cr, 2 * mc - cc),
        (2 * mr - cr, cc),
        (2 * mr - cr, 2 * mc - cc),
    ]


def _to_global(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], cr: int, cc: int) -> tuple[int, int]:
    return row_spans[cr][0], col_spans[cc][0]


def _segment_patch(size: int, orientation: str, face: str, offset: int, length: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    if orientation == "h":
        cols = np.arange(0, length) if face == "left" else np.arange(size - length, size)
        patch[offset, cols] = RED
    else:
        rows = np.arange(0, length) if face == "top" else np.arange(size - length, size)
        patch[rows, offset] = RED
    return patch


def _pick_marker(n_cells: int) -> tuple[int, int]:
    if n_cells == 3:
        return 1, 1
    return random.choice([(1, 1), (1, 2), (2, 1), (2, 2)])


def _candidate_bridge_cells(n_cells: int, marker: tuple[int, int], mode: str) -> list[tuple[int, int]]:
    mr, mc = marker
    cells: list[tuple[int, int]] = []
    for cr in range(n_cells):
        for cc in range(n_cells):
            if (cr, cc) == marker:
                continue
            orbit = _orbit((cr, cc), marker)
            if any(not (0 <= rr < n_cells and 0 <= rc < n_cells) for rr, rc in orbit):
                continue
            if len(set(orbit)) != 4:
                continue
            if mode == "h":
                if cr == mr or cc == mc:
                    continue
                if not (0 <= 2 * mc - cc < n_cells):
                    continue
            else:
                if cr == mr or cc == mc:
                    continue
                if not (0 <= 2 * mr - cr < n_cells):
                    continue
            cells.append((cr, cc))
    random.shuffle(cells)
    return cells


def _place_patch(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    cr: int,
    cc: int,
    patch: np.ndarray,
) -> None:
    r0, c0 = _to_global(row_spans, col_spans, cr, cc)
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch != BG, patch, grid[r0 : r0 + h, c0 : c0 + w])


def _make_input_grid(n_cells: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    side = n_cells * CELL_SIZE + (n_cells - 1)
    grid = np.zeros((side, side), dtype=int)
    for k in range(1, n_cells):
        cut = k * CELL_SIZE + (k - 1)
        grid[cut, :] = SEP
        grid[:, cut] = SEP
    row_spans, col_spans = _grid_structure(grid)
    return grid, row_spans, col_spans


def _inject_marker(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    marker: tuple[int, int],
) -> None:
    patch = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    patch[1:3, 1:3] = GREEN
    _place_patch(grid, row_spans, col_spans, marker[0], marker[1], patch)


def _build_case() -> np.ndarray | None:
    n_cells = random.choice([3, 4])
    grid, row_spans, col_spans = _make_input_grid(n_cells)
    marker = _pick_marker(n_cells)
    _inject_marker(grid, row_spans, col_spans, marker)

    used_cells: set[tuple[int, int]] = {marker}
    forced_modes = random.sample(["h", "v"], k=random.choice([1, 2]))
    chosen_specs: list[tuple[tuple[int, int], str, np.ndarray]] = []

    for mode in forced_modes:
        placed = False
        for cell in _candidate_bridge_cells(n_cells, marker, mode):
            orbit_cells = set(_orbit(cell, marker))
            if orbit_cells & used_cells:
                continue
            if mode == "h":
                face = "right" if cell[1] < marker[1] else "left"
                offset = random.randint(0, CELL_SIZE - 1)
                length = random.randint(1, 3)
            else:
                face = "bottom" if cell[0] < marker[0] else "top"
                offset = random.randint(0, CELL_SIZE - 1)
                length = random.randint(1, 3)
            base_patch = _segment_patch(CELL_SIZE, mode, face, offset, length)
            chosen_specs.append((cell, mode, base_patch))
            used_cells |= orbit_cells
            placed = True
            break
        if not placed:
            return None

    for cell, mode, base_patch in chosen_specs:
        _place_patch(grid, row_spans, col_spans, cell[0], cell[1], base_patch)
        orbit = _orbit(cell, marker)
        extras = []
        if mode == "h":
            extras = [(orbit[2], _flip_mask(base_patch == RED, True, False)), (orbit[3], _flip_mask(base_patch == RED, True, True))]
        else:
            extras = [(orbit[1], _flip_mask(base_patch == RED, False, True)), (orbit[3], _flip_mask(base_patch == RED, True, True))]
        random.shuffle(extras)
        for extra_cell, extra_mask in extras:
            if random.random() < 0.4:
                extra_patch = np.where(extra_mask, RED, BG)
                _place_patch(grid, row_spans, col_spans, extra_cell[0], extra_cell[1], extra_patch)
                break

    extra_count = random.randint(0, 2)
    extra_candidates = [
        cell
        for mode in ("h", "v")
        for cell in _candidate_bridge_cells(n_cells, marker, mode)
        if cell not in used_cells
    ]
    random.shuffle(extra_candidates)
    for cell in extra_candidates[:extra_count]:
        mode = random.choice(["h", "v"])
        face = random.choice(["left", "right"]) if mode == "h" else random.choice(["top", "bottom"])
        offset = random.randint(0, CELL_SIZE - 1)
        length = random.randint(1, 3)
        _place_patch(grid, row_spans, col_spans, cell[0], cell[1], _segment_patch(CELL_SIZE, mode, face, offset, length))
        used_cells.add(cell)

    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray, debug: dict) -> bool:
    sym_grid = debug["sym_grid"]
    bridges = debug["bridges"]
    if not np.array_equal(main(inp), out):
        return False
    if not np.any((sym_grid == RED) & (inp != RED)):
        return False
    if len(bridges) == 0:
        return False
    if not np.any((out == RED) & (sym_grid != RED)):
        return False
    row_spans = debug["row_spans"]
    col_spans = debug["col_spans"]
    marker_r, marker_c = debug["marker_rc"]
    marker_patch_out = _cell_view(out, row_spans, col_spans, marker_r, marker_c)
    if np.count_nonzero(marker_patch_out == GREEN) != 4:
        return False
    if not np.all(marker_patch_out[1:3, 1:3] == GREEN):
        return False
    for (r0, c0), (r1, c1) in bridges:
        if r0 == r1:
            if not np.all(out[r0, min(c0, c1) : max(c0, c1) + 1] == RED):
                return False
        else:
            if not np.all(out[min(r0, r1) : max(r0, r1) + 1, c0] == RED):
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        inp = _build_case()
        if inp is None:
            continue
        try:
            out, debug = _solve_with_debug(inp)
        except Exception:
            continue
        if _semantic_checks(inp, out, debug):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
