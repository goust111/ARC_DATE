from __future__ import annotations

import numpy as np

Coord = tuple[int, int]


BLACK = 0
YELLOW = 4
PAIR_COLORS = (1, 2, 3, 6, 7, 8, 9)
GRID_ROWS_RANGE = (11, 16)
GRID_COLS_RANGE = (11, 16)
PAIR_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 300


def _find_marker_top_left(grid: np.ndarray) -> Coord:
    h, w = grid.shape
    candidates: list[Coord] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == BLACK):
                candidates.append((r, c))
    assert len(candidates) == 1, "expected exactly one 2x2 black marker"
    return candidates[0]


def _cross_mask(shape: tuple[int, int], marker: Coord) -> np.ndarray:
    h, w = shape
    r0, c0 = marker
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + 2, :] = True
    mask[:, c0 : c0 + 2] = True
    return mask


def _is_cross_cell(cell: Coord, marker: Coord) -> bool:
    r, c = cell
    r0, c0 = marker
    return r in (r0, r0 + 1) or c in (c0, c0 + 1)


def _ray_path(survivor: Coord, erased: Coord, marker: Coord) -> list[Coord]:
    sr, sc = survivor
    er, ec = erased
    r0, c0 = marker
    r1, c1 = r0 + 1, c0 + 1

    assert not _is_cross_cell(survivor, marker), "survivor must be outside the cross"
    assert _is_cross_cell(erased, marker), "erased endpoint must be inside the cross"

    if sr == er:
        dc = 1 if ec > sc else -1
        boundary_col = c0 if dc > 0 else c1
        return [(sr, c) for c in range(sc + dc, boundary_col, dc)]

    if sc == ec:
        dr = 1 if er > sr else -1
        boundary_row = r0 if dr > 0 else r1
        return [(r, sc) for r in range(sr + dr, boundary_row, dr)]

    dc = 1 if ec > sc else -1
    boundary_col = c0 if dc > 0 else c1
    return [(sr, c) for c in range(sc + dc, boundary_col, dc)]


def _pair_colors(grid: np.ndarray) -> list[int]:
    colors = sorted(set(map(int, np.unique(grid))) - {BLACK, YELLOW})
    assert 3 <= len(colors) <= 5, "expected 3-5 pair colors"
    return colors


def _classify_pair(coords: list[Coord], marker: Coord) -> str:
    inside = [cell for cell in coords if _is_cross_cell(cell, marker)]
    outside = [cell for cell in coords if not _is_cross_cell(cell, marker)]
    assert len(inside) in (0, 1), "no pair may place both endpoints in the cross"
    if not inside:
        return "complete"

    assert len(outside) == 1
    erased = inside[0]
    survivor = outside[0]
    if survivor[0] == erased[0]:
        return "same_row"
    if survivor[1] == erased[1]:
        return "same_col"
    return "offset"


def _describe_pairs(grid: np.ndarray) -> tuple[set[str], int]:
    marker = _find_marker_top_left(grid)
    damaged_modes: set[str] = set()
    complete_count = 0
    for color in _pair_colors(grid):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) == 2, "each pair color must appear exactly twice"
        kind = _classify_pair(coords, marker)
        if kind == "complete":
            complete_count += 1
        else:
            damaged_modes.add(kind)
    return damaged_modes, complete_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase colored cross cells and restore each half-damaged same-color pair with a ray from the survivor.

    Input structure:
    - A mostly yellow grid contains exactly one interior 2x2 black marker block.
    - Several distinct non-yellow, non-black pair colors each appear exactly twice.
    - For some colors, exactly one endpoint lies in the marker's two rows or two columns; other colors may have both endpoints outside that cross.

    Rule:
    - Use the 2x2 black marker to define a cross: its two rows and two columns.
    - Recolor every cross cell to black unless it is already black or yellow.
    - For each pair color whose two endpoints are split by this erasure, keep the surviving endpoint and cast a same-colored ray toward the erased endpoint: same row -> horizontal, same column -> vertical, otherwise horizontal toward the erased endpoint's column.
    - Draw from the cell after the survivor up to but not including the nearest cross boundary, painting only black or yellow cells.

    Key insight:
    - The erased endpoint is inferred from the original same-color pairing, and the survivor uses that lost position only to choose a direction for partial restoration toward the cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    marker = _find_marker_top_left(g)
    cross = _cross_mask(g.shape, marker)

    out = g.copy()
    erase = cross & (g != BLACK) & (g != YELLOW)
    out[erase] = BLACK

    for color in _pair_colors(g):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each pair color must appear exactly twice"

        inside = [cell for cell in coords if _is_cross_cell(cell, marker)]
        outside = [cell for cell in coords if not _is_cross_cell(cell, marker)]
        assert len(inside) in (0, 1), "pairs may not place both endpoints inside the cross"
        if not inside:
            continue

        assert len(outside) == 1
        survivor = outside[0]
        erased = inside[0]
        for r, c in _ray_path(survivor, erased, marker):
            if out[r, c] in (BLACK, YELLOW):
                out[r, c] = color

    return out


def _random_int(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _random_choice(items):
    return items[int(np.random.randint(len(items)))]


def _marker_cells(marker: Coord) -> set[Coord]:
    r0, c0 = marker
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}


def _candidate_available(cells: list[Coord], used: set[Coord]) -> bool:
    return all(cell not in used for cell in cells)


def _same_row_candidates(shape: tuple[int, int], marker: Coord, used: set[Coord]):
    h, w = shape
    r0, c0 = marker
    c1 = c0 + 1
    candidates = []
    non_cross_rows = [r for r in range(h) if r not in (r0, r0 + 1)]
    for er in non_cross_rows:
        for ec in (c0, c1):
            for sc in range(0, c0 - 1):
                survivor = (er, sc)
                erased = (er, ec)
                path = _ray_path(survivor, erased, marker)
                cells = [survivor, erased, *path]
                if path and _candidate_available(cells, used):
                    candidates.append((survivor, erased, path))
            for sc in range(c1 + 2, w):
                survivor = (er, sc)
                erased = (er, ec)
                path = _ray_path(survivor, erased, marker)
                cells = [survivor, erased, *path]
                if path and _candidate_available(cells, used):
                    candidates.append((survivor, erased, path))
    return candidates


def _same_col_candidates(shape: tuple[int, int], marker: Coord, used: set[Coord]):
    h, w = shape
    r0, c0 = marker
    r1 = r0 + 1
    candidates = []
    non_cross_cols = [c for c in range(w) if c not in (c0, c0 + 1)]
    for ec in non_cross_cols:
        for er in (r0, r1):
            for sr in range(0, r0 - 1):
                survivor = (sr, ec)
                erased = (er, ec)
                path = _ray_path(survivor, erased, marker)
                cells = [survivor, erased, *path]
                if path and _candidate_available(cells, used):
                    candidates.append((survivor, erased, path))
            for sr in range(r1 + 2, h):
                survivor = (sr, ec)
                erased = (er, ec)
                path = _ray_path(survivor, erased, marker)
                cells = [survivor, erased, *path]
                if path and _candidate_available(cells, used):
                    candidates.append((survivor, erased, path))
    return candidates


def _offset_candidates(shape: tuple[int, int], marker: Coord, used: set[Coord]):
    h, w = shape
    r0, c0 = marker
    c1 = c0 + 1
    candidates = []
    non_cross_rows = [r for r in range(h) if r not in (r0, r0 + 1)]
    for er in non_cross_rows:
        for sr in non_cross_rows:
            if sr == er:
                continue
            for ec in (c0, c1):
                for sc in range(0, c0 - 1):
                    survivor = (sr, sc)
                    erased = (er, ec)
                    path = _ray_path(survivor, erased, marker)
                    cells = [survivor, erased, *path]
                    if path and _candidate_available(cells, used):
                        candidates.append((survivor, erased, path))
                for sc in range(c1 + 2, w):
                    survivor = (sr, sc)
                    erased = (er, ec)
                    path = _ray_path(survivor, erased, marker)
                    cells = [survivor, erased, *path]
                    if path and _candidate_available(cells, used):
                        candidates.append((survivor, erased, path))
    return candidates


def _complete_candidates(shape: tuple[int, int], marker: Coord, used: set[Coord]):
    h, w = shape
    safe_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not _is_cross_cell((r, c), marker) and (r, c) not in used
    ]
    candidates = []
    for i, a in enumerate(safe_cells):
        for b in safe_cells[i + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 3:
                candidates.append((a, b))
    return candidates


def _pair_type_plan(n_pairs: int) -> list[str]:
    kinds = ["same_row", "same_col", "offset"]
    if n_pairs >= 4:
        kinds.append("complete")
    extras = n_pairs - len(kinds)
    if extras > 0:
        extra_pool = ["same_row", "same_col", "offset", "complete"]
        for _ in range(extras):
            kinds.append(_random_choice(extra_pool))
    order = np.random.permutation(len(kinds))
    return [kinds[int(i)] for i in order]


def _gen_attempt() -> np.ndarray | None:
    h = _random_int(*GRID_ROWS_RANGE)
    w = _random_int(*GRID_COLS_RANGE)
    r0 = _random_int(1, h - 3)
    c0 = _random_int(1, w - 3)
    marker = (r0, c0)

    n_pairs = _random_int(*PAIR_COUNT_RANGE)
    colors = list(np.random.permutation(PAIR_COLORS)[:n_pairs].astype(int))
    kinds = _pair_type_plan(n_pairs)

    grid = np.full((h, w), YELLOW, dtype=int)
    for r, c in _marker_cells(marker):
        grid[r, c] = BLACK

    used: set[Coord] = set(_marker_cells(marker))

    placements: list[tuple[int, tuple[Coord, Coord, list[Coord]] | tuple[Coord, Coord]]] = []
    pending_complete: list[int] = []

    for color, kind in zip(colors, kinds, strict=False):
        if kind == "same_row":
            candidates = _same_row_candidates((h, w), marker, used)
            if not candidates:
                return None
            survivor, erased, path = _random_choice(candidates)
            used.update([survivor, erased, *path])
            placements.append((color, (survivor, erased, path)))
        elif kind == "same_col":
            candidates = _same_col_candidates((h, w), marker, used)
            if not candidates:
                return None
            survivor, erased, path = _random_choice(candidates)
            used.update([survivor, erased, *path])
            placements.append((color, (survivor, erased, path)))
        elif kind == "offset":
            candidates = _offset_candidates((h, w), marker, used)
            if not candidates:
                return None
            survivor, erased, path = _random_choice(candidates)
            used.update([survivor, erased, *path])
            placements.append((color, (survivor, erased, path)))
        else:
            pending_complete.append(color)

    for color in pending_complete:
        candidates = _complete_candidates((h, w), marker, used)
        if not candidates:
            return None
        a, b = _random_choice(candidates)
        used.update([a, b])
        placements.append((color, (a, b)))

    for color, placement in placements:
        if len(placement) == 2:
            a, b = placement
            grid[a] = color
            grid[b] = color
        else:
            survivor, erased, _path = placement
            grid[survivor] = color
            grid[erased] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    damaged_modes, complete_count = _describe_pairs(grid)
    if damaged_modes != {"same_row", "same_col", "offset"}:
        return None
    if n_pairs >= 4 and complete_count < 1:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
