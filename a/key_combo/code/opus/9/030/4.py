from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _cell_key(cell: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in cell.tolist())


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(sorted(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple((int(r), int(c)) for r, c in arr.tolist())


def _bbox_of_coords(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _offset_candidates(cell_h: int, cell_w: int, shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    shape_h, shape_w = _bbox_of_coords(shape)
    candidates: list[tuple[int, int]] = []
    for r0 in range(cell_h - shape_h + 1):
        for c0 in range(cell_w - shape_w + 1):
            candidates.append((r0, c0))
    return candidates


def _make_cell(
    cell_h: int,
    cell_w: int,
    shape: tuple[tuple[int, int], ...],
    offset: tuple[int, int],
    color: int,
) -> np.ndarray:
    cell = np.full((cell_h, cell_w), BLACK, dtype=int)
    r0, c0 = offset
    for dr, dc in shape:
        cell[r0 + dr, c0 + dc] = int(color)
    return cell


def _separator_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    if len(separators) != 2:
        raise AssertionError("expected exactly two separator lines")
    s0, s1 = sorted(int(v) for v in separators)
    spans = [(0, s0 - 1), (s0 + 1, s1 - 1), (s1 + 1, length - 1)]
    if any(a > b for a, b in spans):
        raise AssertionError("invalid separator layout")
    return spans


def _split_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    row_separators = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    col_separators = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    row_spans = _separator_spans(g.shape[0], row_separators)
    col_spans = _separator_spans(g.shape[1], col_separators)

    row_heights = [r1 - r0 + 1 for r0, r1 in row_spans]
    col_widths = [c1 - c0 + 1 for c0, c1 in col_spans]
    if len(set(row_heights)) != 1 or len(set(col_widths)) != 1:
        raise AssertionError("expected a regular 3x3 cell lattice")

    cells: list[np.ndarray] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cells.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())
    return row_spans, col_spans, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the uniquely occurring cell pattern as a global stamp template for the other cells.

    Input structure:
    - A 3x3 array of equal-sized black cells is separated by full grey horizontal and vertical lines.
    - Each cell contains exactly one small 4-connected monochrome pattern on black background.
    - Four cells share pattern A, four cells share pattern B, and one cell contains unique pattern C.

    Rule:
    - Split the grid into its nine cells by removing the grey separator lines.
    - Compare the full cell arrays to find the single cell whose pattern occurs once.
    - Keep that exceptional cell unchanged, but use its non-black pixel offsets as a template.
    - For every other cell, read its own non-black color, repaint the exceptional template with that color,
      and replace the whole cell by that recolored template on black background.
    - Preserve the grey separator lines.

    Key insight:
    - The odd cell is not the answer itself; it provides the shape that gets recolored and stamped into every
      non-exception cell using each target cell's original color.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    row_spans, col_spans, cells = _split_cells(g)
    keys = [_cell_key(cell) for cell in cells]
    counts = Counter(keys)
    if sorted(counts.values()) != [1, 4, 4]:
        raise AssertionError("expected cell pattern frequencies 4, 4, 1")

    unique_index = next(i for i, key in enumerate(keys) if counts[key] == 1)
    template_coords = np.argwhere(cells[unique_index] != BLACK)
    if not (3 <= len(template_coords) <= 6):
        raise AssertionError("template must have 3..6 colored pixels")

    out = g.copy()
    for idx, cell in enumerate(cells):
        rr = idx // 3
        cc = idx % 3
        r0, r1 = row_spans[rr]
        c0, c1 = col_spans[cc]
        if idx == unique_index:
            out[r0 : r1 + 1, c0 : c1 + 1] = cell
            continue

        colors = set(int(v) for v in np.unique(cell) if int(v) != BLACK)
        if len(colors) != 1:
            raise AssertionError("non-exception cells must contain exactly one non-black color")
        color = next(iter(colors))
        new_cell = np.full_like(cell, BLACK)
        new_cell[template_coords[:, 0], template_coords[:, 1]] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = new_cell

    return out


def _random_shape(size: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...] | None:
    for _ in range(200):
        coords: set[tuple[int, int]] = {(0, 0)}
        while len(coords) < size:
            base = tuple(int(v) for v in rng.choice(np.array(list(coords), dtype=int)))
            dr, dc = DIR4[int(rng.integers(0, len(DIR4)))]
            coords.add((base[0] + dr, base[1] + dc))
        shape = _normalize_coords(coords)
        shape_h, shape_w = _bbox_of_coords(shape)
        if shape_h < 2 or shape_w < 2:
            continue
        if shape_h * shape_w == size:
            continue
        return shape
    return None


def _pick_offsets(
    cell_h: int,
    cell_w: int,
    shapes: dict[str, tuple[tuple[int, int], ...]],
    rng: np.random.Generator,
) -> dict[str, tuple[int, int]] | None:
    candidates = {name: _offset_candidates(cell_h, cell_w, shape) for name, shape in shapes.items()}
    special_c = [
        pos
        for pos in candidates["C"]
        if (pos[0] > 0 or pos[1] > 0)
        and (pos[0] + _bbox_of_coords(shapes["C"])[0] < cell_h or pos[1] + _bbox_of_coords(shapes["C"])[1] < cell_w)
    ]
    if not special_c:
        return None

    c_off = special_c[int(rng.integers(0, len(special_c)))]
    a_choices = [pos for pos in candidates["A"] if pos != c_off]
    b_choices = [pos for pos in candidates["B"] if pos != c_off]
    if not a_choices or not b_choices:
        return None

    for _ in range(100):
        a_off = a_choices[int(rng.integers(0, len(a_choices)))]
        b_off = b_choices[int(rng.integers(0, len(b_choices)))]
        if a_off != b_off:
            return {"A": a_off, "B": b_off, "C": c_off}
    return None


def _assemble_grid(cell_h: int, cell_w: int, cells: list[np.ndarray]) -> np.ndarray:
    grid_h = 3 * cell_h + 2
    grid_w = 3 * cell_w + 2
    grid = np.full((grid_h, grid_w), GREY, dtype=int)

    for idx, cell in enumerate(cells):
        rr = idx // 3
        cc = idx % 3
        r0 = rr * (cell_h + 1)
        c0 = cc * (cell_w + 1)
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
    return grid


def _expected_output(
    cell_h: int,
    cell_w: int,
    labels: list[str],
    cell_colors: dict[str, int],
    shapes: dict[str, tuple[tuple[int, int], ...]],
    offsets: dict[str, tuple[int, int]],
) -> np.ndarray:
    template = _make_cell(cell_h, cell_w, shapes["C"], offsets["C"], cell_colors["C"])
    cells: list[np.ndarray] = []
    for label in labels:
        if label == "C":
            cells.append(template.copy())
        else:
            cells.append(_make_cell(cell_h, cell_w, shapes["C"], offsets["C"], cell_colors[label]))
    return _assemble_grid(cell_h, cell_w, cells)


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        in_row_spans, in_col_spans, in_cells = _split_cells(inp)
        out_row_spans, out_col_spans, out_cells = _split_cells(out)
    except AssertionError:
        return False

    if in_row_spans != out_row_spans or in_col_spans != out_col_spans:
        return False

    in_keys = [_cell_key(cell) for cell in in_cells]
    in_counts = Counter(in_keys)
    if sorted(in_counts.values()) != [1, 4, 4]:
        return False

    rep_indices = {}
    for idx, key in enumerate(in_keys):
        rep_indices.setdefault(key, idx)
    if len(rep_indices) != 3:
        return False

    distinct_areas = []
    distinct_colors = []
    distinct_offsets = []
    unique_index = next(i for i, key in enumerate(in_keys) if in_counts[key] == 1)
    for idx in rep_indices.values():
        cell = in_cells[idx]
        colors = [int(v) for v in np.unique(cell) if int(v) != BLACK]
        if len(colors) != 1:
            return False
        distinct_colors.append(colors[0])
        coords = np.argwhere(cell != BLACK)
        distinct_areas.append(int(coords.shape[0]))
        distinct_offsets.append(tuple(int(v) for v in coords.min(axis=0)))

    if len(set(distinct_areas)) != 3 or any(not (3 <= area <= 6) for area in distinct_areas):
        return False
    if len(set(distinct_colors)) != 3:
        return False
    if len(set(distinct_offsets)) != 3:
        return False

    template_coords = np.argwhere(in_cells[unique_index] != BLACK)
    out_keys = [_cell_key(cell) for cell in out_cells]
    if len(set(out_keys)) != 3:
        return False

    changed = 0
    for idx, (inp_cell, out_cell) in enumerate(zip(in_cells, out_cells)):
        if idx == unique_index:
            if not np.array_equal(inp_cell, out_cell):
                return False
            continue

        colors = [int(v) for v in np.unique(inp_cell) if int(v) != BLACK]
        if len(colors) != 1:
            return False
        expected_cell = np.full_like(inp_cell, BLACK)
        expected_cell[template_coords[:, 0], template_coords[:, 1]] = colors[0]
        if not np.array_equal(out_cell, expected_cell):
            return False
        if np.array_equal(inp_cell, out_cell):
            return False
        changed += 1

    if changed != 8:
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sizes = [int(v) for v in rng.choice(np.array([3, 4, 5, 6], dtype=int), size=3, replace=False)]
    shapes: dict[str, tuple[tuple[int, int], ...]] = {}
    for name, size in zip(("A", "B", "C"), sizes):
        shape = _random_shape(size, rng)
        if shape is None:
            return None
        shapes[name] = shape

    max_h = max(_bbox_of_coords(shape)[0] for shape in shapes.values())
    max_w = max(_bbox_of_coords(shape)[1] for shape in shapes.values())
    min_cell_h = max(5, max_h + 1)
    min_cell_w = max(5, max_w + 1)
    if min_cell_h > 9 or min_cell_w > 9:
        return None

    cell_h = int(rng.integers(min_cell_h, 10))
    cell_w = int(rng.integers(min_cell_w, 10))
    offsets = _pick_offsets(cell_h, cell_w, shapes, rng)
    if offsets is None:
        return None

    colors = [int(v) for v in rng.choice(COLOR_POOL, size=3, replace=False)]
    cell_colors = {"A": colors[0], "B": colors[1], "C": colors[2]}
    labels = ["A"] * 4 + ["B"] * 4 + ["C"]
    rng.shuffle(labels)

    cells = [
        _make_cell(cell_h, cell_w, shapes[label], offsets[label], cell_colors[label])
        for label in labels
    ]
    inp = _assemble_grid(cell_h, cell_w, cells)
    expected = _expected_output(cell_h, cell_w, labels, cell_colors, shapes, offsets)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if not _validate_instance(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
