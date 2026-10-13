from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
            next_id += 1
    return labels, components


def _bbox_area(coords: np.ndarray) -> int:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _trace_paths(grid: np.ndarray, starts: list[int] | None = None) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    if starts is None:
        starts = sorted(int(c) for c in np.where(out[-1] == RED)[0])
    traces: dict[int, list[tuple[int, int]]] = {}
    for start_c in starts:
        r = h - 1
        c = int(start_c)
        trace = [(r, c)]
        while r > 0:
            above = int(out[r - 1, c])
            if above == BLACK:
                r -= 1
                out[r, c] = RED
                trace.append((r, c))
            elif above == GREY:
                if c + 1 >= w or int(out[r, c + 1]) != BLACK:
                    break
                c += 1
                out[r, c] = RED
                trace.append((r, c))
            elif above == RED:
                break
            else:
                raise AssertionError(f"unexpected color above path: {above}")
        traces[int(start_c)] = trace
    return out, traces


def _fill_regions(path_grid: np.ndarray, *, return_meta: bool = False):
    out = np.array(path_grid, dtype=int, copy=True)
    chamber_labels, chambers = _label_components(out != GREY)
    chamber_info: dict[int, dict[str, bool | int]] = {}
    for idx, coords in enumerate(chambers, start=1):
        colors = out[coords[:, 0], coords[:, 1]]
        chamber_info[idx] = {
            "has_red": bool(np.any(colors == RED)),
            "rectangular": int(coords.shape[0]) == _bbox_area(coords),
        }

    black_labels, black_components = _label_components(out == BLACK)
    region_meta = []
    for coords in black_components:
        region = {
            "touches_border": _touches_border(coords, out.shape),
            "component_id": int(black_labels[coords[0, 0], coords[0, 1]]),
            "chamber_id": int(chamber_labels[coords[0, 0], coords[0, 1]]),
        }
        info = chamber_info[region["chamber_id"]]
        region["has_red"] = bool(info["has_red"])
        region["rectangular"] = bool(info["rectangular"])
        if not region["touches_border"]:
            fill = GREEN if region["rectangular"] and not region["has_red"] else YELLOW
            out[coords[:, 0], coords[:, 1]] = fill
            region["fill"] = int(fill)
        else:
            region["fill"] = BLACK
        region_meta.append(region)
    if return_meta:
        return out, {"regions": region_meta, "chambers": chamber_info}
    return out


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    path_grid, traces = _trace_paths(grid)
    if return_meta:
        out, fill_meta = _fill_regions(path_grid, return_meta=True)
        return out, {"path_grid": path_grid, "traces": traces, **fill_meta}
    return _fill_regions(path_grid)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow bottom red paths through a broken grey lattice, then color each enclosed black remnant by whether its chamber stayed intact or was path-breached.
    
    Input structure:
    - Black background surrounds a grey rectangular lattice with outer frame and internal orthogonal walls.
    - A few one-pixel wall gaps break selected horizontal wall segments inside the lattice.
    - Two or three red control pixels lie on the bottom row; at least one sits below the lattice footprint.
    
    Rule:
    - From each bottom red control, repeatedly move upward through black cells, painting visited cells red.
    - When the cell above is grey, sidestep one cell right if that right neighbor is black; otherwise stop. If the cell above is already red, stop.
    - After all paths are drawn, treat every grey-free 4-connected chamber as one structural region, but only recolor black subregions that no longer touch the grid border.
    - A non-border black subregion becomes green when its containing chamber is a full rectangle and contains no red path pixels; otherwise it becomes yellow. Keep red paths and grey walls unchanged.
    
    Key insight:
    - The same grey lattice both bends the upward paths and defines which black remnants belong to the same chamber, so the final fill color depends on combining chamber geometry with actual path intrusion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    return _solve(g)


def _build_lattice(
    h: int,
    w: int,
    top: int,
    left: int,
    rows: int,
    cols: int,
    cell_size: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    grid = np.zeros((h, w), dtype=int)
    row_lines = [top + k * (cell_size + 1) for k in range(rows + 1)]
    col_lines = [left + k * (cell_size + 1) for k in range(cols + 1)]
    for rr in row_lines:
        grid[rr, left : col_lines[-1] + 1] = GREY
    for cc in col_lines:
        grid[top : row_lines[-1] + 1, cc] = GREY
    return grid, row_lines, col_lines


def _segment_cols(col_lines: list[int], cell_c: int) -> list[int]:
    return list(range(col_lines[cell_c] + 1, col_lines[cell_c + 1]))


def _place_gap(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    wall_idx: int,
    cell_c: int,
    offset: int,
) -> tuple[int, int]:
    rr = row_lines[wall_idx]
    cc = col_lines[cell_c] + 1 + offset
    grid[rr, cc] = BLACK
    return rr, cc


def _validate_semantics(grid: np.ndarray, meta: dict[str, object]) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False
    starts = sorted(int(c) for c in np.where(grid[-1] == RED)[0])
    if len(starts) not in (2, 3):
        return False
    if any(b - a < 2 for a, b in zip(starts, starts[1:])):
        return False
    if any(int(grid[-2, c]) != BLACK for c in starts):
        return False

    out_a, solve_meta_a = _solve(grid, return_meta=True)
    path_lr, traces_lr = _trace_paths(grid, starts=starts)
    path_rl, _ = _trace_paths(grid, starts=list(reversed(starts)))
    final_lr = _fill_regions(path_lr)
    final_rl = _fill_regions(path_rl)
    if not np.array_equal(final_lr, final_rl):
        return False

    main_start = int(meta["main_start"])
    main_trace = traces_lr[main_start]
    if len(main_trace) - 1 < 3:
        return False
    if not any(r0 == r1 and c1 == c0 + 1 for (r0, c0), (r1, c1) in zip(main_trace, main_trace[1:])):
        return False

    if not np.any(out_a == GREEN) or not np.any(out_a == YELLOW):
        return False

    regions = solve_meta_a["regions"]
    has_green = any(region["fill"] == GREEN for region in regions)
    has_yellow_from_red = any(region["fill"] == YELLOW and region["has_red"] for region in regions)
    has_yellow_broken_unvisited = any(
        region["fill"] == YELLOW and (not region["has_red"]) and (not region["rectangular"])
        for region in regions
    )
    if not (has_green and has_yellow_from_red and has_yellow_broken_unvisited):
        return False

    row_lines = meta["row_lines"]
    col_lines = meta["col_lines"]
    main_gap_cols = meta["main_gap_cols"]
    cp = int(meta["path_col"])
    top_cell = 0
    bottom_cell = int(meta["rows"]) - 1
    if not (col_lines[cp] < main_gap_cols[0] < col_lines[cp + 1]):
        return False
    if not (row_lines[bottom_cell] < main_trace[-1][0] < row_lines[top_cell + 1] or main_trace[-1][0] < row_lines[1]):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]] | None:
    rows = int(rng.integers(2, 4))
    cols = int(rng.integers(3, 5))
    cell_size = int(rng.integers(3, 5))
    top = int(rng.integers(1, 3))
    left = int(rng.integers(1, 4))
    right_margin = int(rng.integers(2, 5))
    bottom_margin = int(rng.integers(3, 6))

    lattice_h = rows * cell_size + rows + 1
    lattice_w = cols * cell_size + cols + 1
    h = top + lattice_h + bottom_margin
    w = left + lattice_w + right_margin
    if h > 30 or w > 30:
        return None

    grid, row_lines, col_lines = _build_lattice(h, w, top, left, rows, cols, cell_size)

    path_col = int(rng.integers(cols))
    main_path_cells = rows
    offsets = [int(rng.integers(cell_size))]
    for _ in range(main_path_cells - 1):
        offsets.append(int(rng.integers(offsets[-1], cell_size)))

    main_gap_positions = []
    main_gap_cols = []
    for idx, off in enumerate(offsets):
        wall_idx = rows - idx
        rr, cc = _place_gap(grid, row_lines, col_lines, wall_idx, path_col, off)
        main_gap_positions.append((rr, cc))
        main_gap_cols.append(cc)

    gap_keys = {(rr, cc) for rr, cc in main_gap_positions}

    break_col_choices = [c for c in range(cols) if c != path_col]
    if not break_col_choices:
        return None
    rng.shuffle(break_col_choices)
    extra_gap_count = 1 + int(rows == 2 and cols >= 4 and rng.random() < 0.5)
    extra_gap_positions = []
    for i in range(extra_gap_count):
        cell_c = break_col_choices[i % len(break_col_choices)]
        wall_idx = int(rng.integers(1, rows))
        off = int(rng.integers(cell_size))
        rr, cc = row_lines[wall_idx], col_lines[cell_c] + 1 + off
        if (rr, cc) in gap_keys:
            return None
        grid[rr, cc] = BLACK
        gap_keys.add((rr, cc))
        extra_gap_positions.append((rr, cc))

    start_offset = int(rng.integers(offsets[0] + 1))
    main_start = col_lines[path_col] + 1 + start_offset

    control_cols = [main_start]
    side_candidates = []
    if left >= 1:
        side_candidates.append(0)
    if w - col_lines[-1] - 1 >= 1:
        side_candidates.append(w - 1)
    rng.shuffle(side_candidates)
    extra_controls = 1 + int(bool(side_candidates) and rng.random() < 0.5)
    for col in side_candidates[:extra_controls]:
        control_cols.append(int(col))
    control_cols = sorted(set(control_cols))
    if len(control_cols) < 2 or len(control_cols) > 3:
        return None
    if any(b - a < 2 for a, b in zip(control_cols, control_cols[1:])):
        return None

    for c in control_cols:
        grid[h - 1, c] = RED

    meta = {
        "rows": rows,
        "cols": cols,
        "cell_size": cell_size,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "path_col": path_col,
        "main_start": main_start,
        "main_gap_cols": main_gap_cols,
        "main_gap_positions": main_gap_positions,
        "extra_gap_positions": extra_gap_positions,
    }
    if not _validate_semantics(grid, meta):
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        attempt = _gen_attempt(rng)
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input")
