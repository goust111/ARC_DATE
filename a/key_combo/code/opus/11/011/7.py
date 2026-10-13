from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7

_MAX_PATTERN_AREA = 4
_GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _pattern_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.astype(int).ravel())


def _build_pattern_catalog() -> dict[int, list[np.ndarray]]:
    by_area: dict[int, list[np.ndarray]] = {area: [] for area in range(1, _MAX_PATTERN_AREA + 1)}
    for bits in range(1, 1 << 9):
        area = bits.bit_count()
        if area > _MAX_PATTERN_AREA:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        if _is_connected(mask):
            by_area[area].append(mask)
    for area in by_area:
        by_area[area].sort(key=_pattern_signature)
    return by_area


_PATTERNS_BY_AREA = _build_pattern_catalog()


def _lattice_lines(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _parse_cells(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    rows = _lattice_lines(grid, ORANGE, axis=0)
    cols = _lattice_lines(grid, ORANGE, axis=1)
    assert len(rows) >= 2 and len(cols) >= 2
    assert rows[0] == 0 and rows[-1] == grid.shape[0] - 1
    assert cols[0] == 0 and cols[-1] == grid.shape[1] - 1

    row_steps = np.diff(rows)
    col_steps = np.diff(cols)
    assert np.all(row_steps == row_steps[0]) and np.all(col_steps == col_steps[0])
    cell_h = int(row_steps[0] - 1)
    cell_w = int(col_steps[0] - 1)
    assert cell_h > 0 and cell_w > 0

    n_rows = len(rows) - 1
    n_cols = len(cols) - 1
    cells: list[dict[str, object]] = []
    for i in range(n_rows):
        for j in range(n_cols):
            r0 = rows[i] + 1
            r1 = rows[i + 1]
            c0 = cols[j] + 1
            c1 = cols[j + 1]
            window = grid[r0:r1, c0:c1]
            assert window.shape == (cell_h, cell_w)
            mask = window != BLACK
            count = int(mask.sum())
            assert count > 0
            cells.append(
                {
                    "row": i,
                    "col": j,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "mask": mask,
                    "count": count,
                    "edge": i in (0, n_rows - 1) or j in (0, n_cols - 1),
                }
            )
    return n_rows, n_cols, cells


def _representative_mask(cells: list[dict[str, object]], edge: bool) -> np.ndarray:
    group = [cell for cell in cells if bool(cell["edge"]) == edge]
    assert group
    rep = min(group, key=lambda cell: (int(cell["count"]), int(cell["row"]), int(cell["col"])))
    return np.array(rep["mask"], dtype=bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap lattice-edge and lattice-interior cell patterns by copying each group's smallest representative template.

    Input structure:
    - Orange grid lines form a regular lattice whose cells all have the same interior size.
    - Every lattice cell contains a small foreground pattern on a black background.
    - Cells on the outer border of the lattice belong to the edge group; the remaining cells belong to the interior group.

    Rule:
    - Parse the lattice and split cells into edge cells and interior cells by row/column position.
    - In each group, choose the representative pattern with the fewest foreground pixels, breaking ties by topmost then leftmost cell position.
    - Clear every edge cell and redraw the interior representative in blue; clear every interior cell and redraw the edge representative in red.
    - Keep the orange lattice unchanged.

    Key insight:
    - The swap happens simultaneously at the group level: all edge cells copy one interior template and all interior cells copy one edge template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _n_rows, _n_cols, cells = _parse_cells(g)
    edge_template = _representative_mask(cells, edge=True)
    interior_template = _representative_mask(cells, edge=False)

    out = g.copy()
    for cell in cells:
        r0 = int(cell["r0"])
        r1 = int(cell["r1"])
        c0 = int(cell["c0"])
        c1 = int(cell["c1"])
        out[r0:r1, c0:c1] = BLACK
        template = interior_template if bool(cell["edge"]) else edge_template
        color = BLUE if bool(cell["edge"]) else RED
        out[r0:r1, c0:c1][template] = color
    return out


def _pick_pattern(
    rng: np.random.Generator,
    allowed_areas: list[int],
    used: set[tuple[int, ...]],
) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for area in allowed_areas:
        candidates.extend(mask for mask in _PATTERNS_BY_AREA[area] if _pattern_signature(mask) not in used)
    if not candidates:
        return None
    chosen = np.array(candidates[int(rng.integers(0, len(candidates)))], dtype=bool)
    used.add(_pattern_signature(chosen))
    return chosen


def _cell_positions(n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    edge, interior = [], []
    for i in range(n):
        for j in range(n):
            if i in (0, n - 1) or j in (0, n - 1):
                edge.append((i, j))
            else:
                interior.append((i, j))
    return edge, interior


def _paint_lattice_base(n: int) -> np.ndarray:
    size = 4 * n + 1
    grid = np.zeros((size, size), dtype=int)
    grid[::4, :] = ORANGE
    grid[:, ::4] = ORANGE
    return grid


def _choose_group_patterns(
    rng: np.random.Generator,
    positions: list[tuple[int, int]],
    rep_pos: tuple[int, int],
    min_area: int,
    used: set[tuple[int, ...]],
) -> dict[tuple[int, int], np.ndarray] | None:
    mapping: dict[tuple[int, int], np.ndarray] = {}
    rep_pattern = _pick_pattern(rng, [min_area], used)
    if rep_pattern is None:
        return None
    mapping[rep_pos] = rep_pattern
    larger_areas = list(range(min_area + 1, _MAX_PATTERN_AREA + 1))
    if not larger_areas:
        return None
    for pos in positions:
        if pos == rep_pos:
            continue
        pattern = _pick_pattern(rng, larger_areas, used)
        if pattern is None:
            return None
        mapping[pos] = pattern
    return mapping


def _paint_patterns(
    grid: np.ndarray,
    cell_patterns: dict[tuple[int, int], np.ndarray],
    edge_positions: set[tuple[int, int]],
) -> None:
    for (i, j), pattern in cell_patterns.items():
        r0 = 1 + 4 * i
        c0 = 1 + 4 * j
        color = RED if (i, j) in edge_positions else BLUE
        block = grid[r0 : r0 + 3, c0 : c0 + 3]
        block[pattern] = color


def _group_counts(cell_patterns: dict[tuple[int, int], np.ndarray], positions: list[tuple[int, int]]) -> list[int]:
    return sorted(int(cell_patterns[pos].sum()) for pos in positions)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice([4, 5]))
    grid = _paint_lattice_base(n)
    edge_positions, interior_positions = _cell_positions(n)
    edge_rep = edge_positions[int(rng.integers(0, len(edge_positions)))]
    interior_rep = interior_positions[int(rng.integers(0, len(interior_positions)))]

    edge_min_area = int(rng.integers(1, _MAX_PATTERN_AREA))
    interior_min_area = int(rng.integers(1, _MAX_PATTERN_AREA))

    used: set[tuple[int, ...]] = set()
    edge_patterns = _choose_group_patterns(rng, edge_positions, edge_rep, edge_min_area, used)
    if edge_patterns is None:
        return None
    interior_patterns = _choose_group_patterns(rng, interior_positions, interior_rep, interior_min_area, used)
    if interior_patterns is None:
        return None

    cell_patterns = {}
    cell_patterns.update(edge_patterns)
    cell_patterns.update(interior_patterns)
    _paint_patterns(grid, cell_patterns, set(edge_positions))

    edge_counts = _group_counts(cell_patterns, edge_positions)
    interior_counts = _group_counts(cell_patterns, interior_positions)
    if not (len(edge_counts) >= 2 and edge_counts[0] < edge_counts[1]):
        return None
    if not (len(interior_counts) >= 2 and interior_counts[0] < interior_counts[1]):
        return None
    if _pattern_signature(cell_patterns[edge_rep]) == _pattern_signature(cell_patterns[interior_rep]):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
