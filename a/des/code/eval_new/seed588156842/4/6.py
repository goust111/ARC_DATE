from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Tuple

import numpy as np


AZURE = 8
GREY = 5
BLACK = 0

TYPE_H = "horizontal"
TYPE_V = "vertical"
TYPE_R = "rot90"
TYPE_C = "central"
TYPES = (TYPE_H, TYPE_V, TYPE_R, TYPE_C)
TYPE_PRIORITY = {
    TYPE_R: 0,
    TYPE_H: 1,
    TYPE_V: 2,
    TYPE_C: 3,
}

CONTENT_SIZE_CHOICES = (5,)
PUZZLE_ROW_CHOICES = (1, 2)
COL_COUNT = 4
MARKER_POOL = [1, 2, 3, 4, 6, 7, 9]
PATTERN_POOL = [1, 2, 3, 4, 6, 7, 9]
ICON_COLOR = BLACK
GEN_ATTEMPTS = 400
BANK_TARGET = 12


def _transform_coord(pos: Tuple[int, int], size: int, kind: str) -> Tuple[int, int]:
    r, c = pos
    if kind == TYPE_H:
        return size - 1 - r, c
    if kind == TYPE_V:
        return r, size - 1 - c
    if kind == TYPE_R:
        return c, size - 1 - r
    if kind == TYPE_C:
        return size - 1 - r, size - 1 - c
    raise ValueError(f"unknown symmetry type: {kind}")


def _apply_transform(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == TYPE_H:
        return np.flipud(mask)
    if kind == TYPE_V:
        return np.fliplr(mask)
    if kind == TYPE_R:
        return np.rot90(mask, -1)
    if kind == TYPE_C:
        return np.rot90(mask, 2)
    raise ValueError(f"unknown symmetry type: {kind}")


def _mismatch_score(mask: np.ndarray, kind: str) -> int:
    return int(np.count_nonzero(mask != _apply_transform(mask, kind)))


def _best_types(mask: np.ndarray) -> Tuple[List[str], Dict[str, int]]:
    scores = {kind: _mismatch_score(mask, kind) for kind in TYPES}
    best_score = min(scores.values())
    best = [kind for kind, score in scores.items() if score == best_score]
    return best, scores


def _infer_type(mask: np.ndarray) -> str:
    best, scores = _best_types(mask)
    return min(best, key=lambda kind: TYPE_PRIORITY[kind])


def _repair_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    repaired = mask.copy()
    if kind == TYPE_R:
        for k in range(1, 4):
            repaired |= np.rot90(mask, -k)
        return repaired
    repaired |= _apply_transform(mask, kind)
    return repaired


def _full_separator_indices(line_mask: np.ndarray) -> List[int]:
    return [int(i) for i, ok in enumerate(line_mask.tolist()) if ok]


def _intervals_from_separators(indices: List[int], length: int) -> List[Tuple[int, int]]:
    assert indices and indices[0] == 0 and indices[-1] == length - 1
    intervals: List[Tuple[int, int]] = []
    for left, right in zip(indices, indices[1:], strict=False):
        if right - left > 1:
            intervals.append((left + 1, right))
    return intervals


def _cell_intervals(grid: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    row_seps = _full_separator_indices(np.all(grid == GREY, axis=1))
    col_seps = _full_separator_indices(np.all(grid == GREY, axis=0))
    rows = _intervals_from_separators(row_seps, grid.shape[0])
    cols = _intervals_from_separators(col_seps, grid.shape[1])
    assert rows and cols
    row_sizes = {r1 - r0 for r0, r1 in rows}
    col_sizes = {c1 - c0 for c0, c1 in cols}
    assert len(row_sizes) == 1 and len(col_sizes) == 1 and row_sizes == col_sizes
    return rows, cols


def _content_mask(cell: np.ndarray) -> np.ndarray:
    return cell[:-1, :-1] != AZURE


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start = tuple(map(int, points[0]))
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(points.shape[0])


def _orbits(size: int, kind: str) -> List[List[Tuple[int, int]]]:
    seen = set()
    out: List[List[Tuple[int, int]]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit: List[Tuple[int, int]] = []
            cur = (r, c)
            while cur not in orbit:
                orbit.append(cur)
                cur = _transform_coord(cur, size, kind)
            for pos in orbit:
                seen.add(pos)
            out.append(orbit)
    return out


def _build_mask_from_orbits(size: int, orbit_list: Iterable[List[Tuple[int, int]]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for orbit in orbit_list:
        for r, c in orbit:
            mask[r, c] = True
    return mask


def _random_full_mask(size: int, kind: str, rng: np.random.Generator) -> np.ndarray | None:
    all_orbits = _orbits(size, kind)
    for _ in range(120):
        density = float(rng.uniform(0.35, 0.65))
        chosen = [orbit for orbit in all_orbits if rng.random() < density]
        if not chosen:
            continue
        area = sum(len(orbit) for orbit in chosen)
        if area < max(5, size + 1) or area > size * size - 2:
            continue
        mask = _build_mask_from_orbits(size, chosen)
        if not _is_connected(mask):
            continue
        if _mismatch_score(mask, kind) != 0:
            continue
        forbidden_zero = [other for other in TYPES if other != kind]
        if kind == TYPE_R:
            forbidden_zero = [other for other in forbidden_zero if other != TYPE_C]
        if any(_mismatch_score(mask, other) == 0 for other in forbidden_zero):
            continue
        competing = [other for other in TYPES if TYPE_PRIORITY[other] <= TYPE_PRIORITY[kind] and other != kind]
        if competing and min(_mismatch_score(mask, other) for other in competing) < 4:
            continue
        if not any(len(orbit) > 1 for orbit in chosen):
            continue
        return mask
    return None


def _erase_mask(full_mask: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray | None:
    size = full_mask.shape[0]
    visible = np.zeros_like(full_mask)
    erased_any = False
    for orbit in _orbits(size, kind):
        if not full_mask[orbit[0]]:
            continue
        if len(orbit) == 1:
            keep = orbit
        else:
            max_drop = len(orbit) - 1
            drop = int(rng.integers(0, max_drop + 1))
            if drop > 0:
                erased_any = True
            keep_count = len(orbit) - drop
            picks = rng.choice(len(orbit), size=keep_count, replace=False)
            keep = [orbit[int(i)] for i in np.sort(picks)]
        for r, c in keep:
            visible[r, c] = True
    if not erased_any:
        expandable = [orbit for orbit in _orbits(size, kind) if full_mask[orbit[0]] and len(orbit) > 1]
        if not expandable:
            return None
        orbit = expandable[int(rng.integers(len(expandable)))]
        victim = orbit[int(rng.integers(len(orbit)))]
        visible[victim] = False
    if visible.sum() < 3:
        return None
    if _infer_type(visible) != kind:
        return None
    if not np.array_equal(_repair_mask(visible, kind), full_mask):
        return None
    return visible


def _random_observed_mask(size: int, kind: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray] | None:
    for _ in range(160):
        full_mask = _random_full_mask(size, kind, rng)
        if full_mask is None:
            continue
        visible = _erase_mask(full_mask, kind, rng)
        if visible is None:
            continue
        return full_mask, visible
    return None


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _build_template_bank() -> Dict[Tuple[int, str], List[np.ndarray]]:
    rng = np.random.default_rng(0)
    bank: Dict[Tuple[int, str], List[np.ndarray]] = {}
    for size in CONTENT_SIZE_CHOICES:
        for kind in TYPES:
            key = (size, kind)
            bank[key] = []
            seen = set()
            for _ in range(2400):
                mask = _random_full_mask(size, kind, rng)
                if mask is None:
                    continue
                sig = _mask_key(mask)
                if sig in seen:
                    continue
                seen.add(sig)
                bank[key].append(mask)
                if len(bank[key]) >= BANK_TARGET:
                    break
            assert bank[key], f"failed to build template bank for {key}"
    return bank


TEMPLATE_BANK = _build_template_bank()


def _make_cell(content_mask: np.ndarray, color: int, cell_size: int, marker: int | None) -> np.ndarray:
    cell = np.full((cell_size, cell_size), AZURE, dtype=int)
    cell[:-1, :-1][content_mask] = int(color)
    if marker is not None:
        cell[-1, -1] = int(marker)
    return cell


def _paint_cell(grid: np.ndarray, row: int, col: int, cell: np.ndarray) -> None:
    cell_size = cell.shape[0]
    r0 = 1 + row * (cell_size + 1)
    c0 = 1 + col * (cell_size + 1)
    grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair erased monochrome motifs by first identifying which symmetry each cell follows, then writing a legend-derived type marker.

    Input structure:
    - The background is azure and full grey separator rows/columns partition the board into equal square cells.
    - All rows except the last contain puzzle cells; each puzzle cell has a monochrome motif inside its inner square content area, with some motif pixels erased back to azure.
    - The bottom row is a legend row: each legend cell contains a small intact motif whose symmetry type matches one of the four allowed types, and the legend cell's bottom-right pixel is that type's marker color.

    Rule:
    - Split the grid by the grey separators into equal cells and treat the last row of cells as the legend.
    - For each legend motif, determine which of the four candidate symmetries best fits it: horizontal reflection, vertical reflection, 90-degree rotation, or 180-degree central symmetry. This yields the mapping from symmetry type to marker color.
    - For each puzzle cell, compare the observed motif with all four transforms, choose the best-fitting symmetry by minimum mismatch count (preferring the stricter 90-degree rotation when it ties with 180-degree symmetry), repair the motif by copying occupied pixels across that symmetry orbit, and write the corresponding legend marker color into the cell's bottom-right pixel.

    Key insight:
    - The task is two-stage: the missing pixels can only be restored correctly after the motif has been classified into the right symmetry family, and the legend icons must be interpreted rather than assumed by position.
    """
    g = np.asarray(grid, dtype=int)
    rows, cols = _cell_intervals(g)
    assert len(cols) == COL_COUNT and len(rows) >= 2

    out = g.copy()
    legend_row = len(rows) - 1

    type_to_marker: Dict[str, int] = {}
    for col_idx, (c0, c1) in enumerate(cols):
        r0, r1 = rows[legend_row]
        cell = g[r0:r1, c0:c1]
        marker = int(cell[-1, -1])
        icon_mask = _content_mask(cell)
        icon_type = _infer_type(icon_mask)
        type_to_marker[icon_type] = marker
    assert set(type_to_marker) == set(TYPES)

    for row_idx, (r0, r1) in enumerate(rows[:-1]):
        for col_idx, (c0, c1) in enumerate(cols):
            cell = g[r0:r1, c0:c1]
            content = cell[:-1, :-1]
            colors = sorted({int(v) for v in np.unique(content) if int(v) != AZURE})
            assert len(colors) == 1, "each puzzle motif must use one foreground color"
            color = colors[0]
            mask = content != AZURE
            kind = _infer_type(mask)
            repaired = _repair_mask(mask, kind)

            new_cell = np.full_like(cell, AZURE)
            new_cell[:-1, :-1][repaired] = color
            new_cell[-1, -1] = type_to_marker[kind]
            out[r0:r1, c0:c1] = new_cell

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    content_size = int(rng.choice(CONTENT_SIZE_CHOICES))
    cell_size = content_size + 1
    puzzle_rows = int(rng.choice(PUZZLE_ROW_CHOICES))
    total_rows = puzzle_rows + 1

    height = total_rows * cell_size + total_rows + 1
    width = COL_COUNT * cell_size + COL_COUNT + 1
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY
    for row in range(1, total_rows):
        sep = row * cell_size + row
        grid[sep, :] = GREY
    for col in range(1, COL_COUNT):
        sep = col * cell_size + col
        grid[:, sep] = GREY

    marker_colors = rng.choice(MARKER_POOL, size=4, replace=False).tolist()
    marker_perm = rng.permutation(4).tolist()
    type_to_marker = {TYPES[i]: int(marker_colors[marker_perm[i]]) for i in range(4)}

    puzzle_types = list(TYPES) * puzzle_rows
    rng.shuffle(puzzle_types)

    available_pattern_colors = [c for c in PATTERN_POOL if c not in type_to_marker.values()]
    if not available_pattern_colors:
        available_pattern_colors = PATTERN_POOL[:]

    expected = grid.copy()

    for idx, kind in enumerate(puzzle_types):
        full_mask = TEMPLATE_BANK[(content_size, kind)][int(rng.integers(len(TEMPLATE_BANK[(content_size, kind)])))]
        observed_mask = None
        for _ in range(40):
            observed_mask = _erase_mask(full_mask, kind, rng)
            if observed_mask is not None:
                break
        if observed_mask is None:
            return None
        color = int(rng.choice(available_pattern_colors))
        row = idx // COL_COUNT
        col = idx % COL_COUNT
        inp_cell = _make_cell(observed_mask, color, cell_size, None)
        out_cell = _make_cell(full_mask, color, cell_size, type_to_marker[kind])
        _paint_cell(grid, row, col, inp_cell)
        _paint_cell(expected, row, col, out_cell)

    legend_types = list(TYPES)
    rng.shuffle(legend_types)
    for col, kind in enumerate(legend_types):
        choices = TEMPLATE_BANK[(content_size, kind)]
        icon_mask = choices[int(rng.integers(len(choices)))]
        legend_cell = _make_cell(icon_mask, ICON_COLOR, cell_size, type_to_marker[kind])
        _paint_cell(grid, puzzle_rows, col, legend_cell)
        _paint_cell(expected, puzzle_rows, col, legend_cell)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-repair puzzle")
