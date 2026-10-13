from __future__ import annotations

import random
from collections import deque
from itertools import permutations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

LEFT_OBJECT_COLORS = (RED, GREEN, YELLOW)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _compress_positions(indices: list[int]) -> list[int]:
    if not indices:
        return []
    runs: list[list[int]] = [[indices[0]]]
    for idx in indices[1:]:
        if idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return [run[len(run) // 2] for run in runs]


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, f"expected one full-height grey separator, got {cols}"
    return cols[0]


def _line_positions(counts: np.ndarray, threshold: int) -> list[int]:
    return _compress_positions([i for i, count in enumerate(counts.tolist()) if int(count) > int(threshold)])


def _extract_left_summary(grid: np.ndarray, sep_col: int) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    left = grid[:, :sep_col]
    orange_rows = _line_positions(np.sum(left == ORANGE, axis=1), threshold=4)
    orange_cols = _line_positions(np.sum(left == ORANGE, axis=0), threshold=4)
    assert len(orange_rows) == 4, f"expected 4 horizontal orange lines, got {orange_rows}"
    assert len(orange_cols) == 4, f"expected 4 vertical orange lines, got {orange_cols}"

    summary = np.zeros((3, 3), dtype=int)
    slots: list[list[tuple[int, int, int, int]]] = []
    for i in range(3):
        row_slots: list[tuple[int, int, int, int]] = []
        r0 = orange_rows[i] + 1
        r1 = orange_rows[i + 1] - 1
        assert r0 <= r1, "slot height must be positive"
        for j in range(3):
            c0 = orange_cols[j] + 1
            c1 = orange_cols[j + 1] - 1
            assert c0 <= c1, "slot width must be positive"
            view = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = sorted(set(map(int, np.unique(view))) - {BLACK, ORANGE})
            assert len(colors) <= 1, f"slot {(i, j)} has multiple colors {colors}"
            if colors:
                summary[i, j] = colors[0]
            row_slots.append((r0, c0, r1, c1))
        slots.append(row_slots)
    return summary, slots


def _extract_right_cells(grid: np.ndarray, sep_col: int) -> list[list[tuple[int, int, int, int]]]:
    right = grid[:, sep_col + 1 :]
    yellow_rows = _line_positions(np.sum(right == YELLOW, axis=1), threshold=3)
    yellow_cols = _line_positions(np.sum(right == YELLOW, axis=0), threshold=3)
    assert len(yellow_rows) == 3, f"expected 3 horizontal yellow lines, got {yellow_rows}"
    assert len(yellow_cols) == 3, f"expected 3 vertical yellow lines, got {yellow_cols}"

    cell_h = yellow_rows[1] - yellow_rows[0] - 1
    cell_w = yellow_cols[1] - yellow_cols[0] - 1
    assert cell_h == yellow_rows[2] - yellow_rows[1] - 1 and cell_h > 0
    assert cell_w == yellow_cols[2] - yellow_cols[1] - 1 and cell_w > 0

    row_spans = [
        (yellow_rows[0] + 1, yellow_rows[1] - 1),
        (yellow_rows[1] + 1, yellow_rows[2] - 1),
        (yellow_rows[2] + 1, yellow_rows[2] + cell_h),
    ]
    col_spans_local = [
        (yellow_cols[0] + 1, yellow_cols[1] - 1),
        (yellow_cols[1] + 1, yellow_cols[2] - 1),
        (yellow_cols[2] + 1, yellow_cols[2] + cell_w),
    ]
    cells: list[list[tuple[int, int, int, int]]] = []
    for r0, r1 in row_spans:
        assert 0 <= r0 <= r1 < grid.shape[0]
        row_cells: list[tuple[int, int, int, int]] = []
        for c0_local, c1_local in col_spans_local:
            c0 = sep_col + 1 + c0_local
            c1 = sep_col + 1 + c1_local
            assert 0 <= c0 <= c1 < grid.shape[1]
            row_cells.append((r0, c0, r1, c1))
        cells.append(row_cells)
    return cells


def _apply_summary_to_cells(grid: np.ndarray, summary: np.ndarray, cells: list[list[tuple[int, int, int, int]]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for i in range(3):
        for j in range(3):
            color = int(summary[i, j])
            if color == BLACK:
                continue
            r0, c0, r1, c1 = cells[i][j]
            window = out[r0 : r1 + 1, c0 : c1 + 1]
            mask = window == BLUE
            window[mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the left orange table as a 3x3 color summary that recolors blue pixels in the aligned right-side lattice cells.

    Input structure:
    - A full-height grey separator column splits the grid into a left orange table and a right yellow lattice.
    - The left side contains a 3x3 array of orange-bounded slots on black backgrounds; each slot is empty or contains one small monochrome object.
    - The right side contains a 3x3 array of lattice cells; each cell has blue pixels on black background, with yellow grid lines preserved around and between cells.

    Rule:
    - Read the left slots in row-major order and summarize each slot by its unique non-black, non-orange color, or by black if the slot is empty.
    - Match that 3x3 summary matrix to the right 3x3 lattice cells by the same row and column positions.
    - In every right cell whose summary color is non-black, recolor all blue pixels in that cell to the summary color.
    - Leave empty-summary cells unchanged, and preserve the separator, left table, yellow lattice lines, and black background pixels everywhere.

    Key insight:
    - The left side is not output directly; it becomes a color-and-mask controller whose slot summaries are transferred across the separator onto the corresponding right-side cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_col = _find_separator_col(g)
    summary, _slots = _extract_left_summary(g, sep_col)
    cells = _extract_right_cells(g, sep_col)
    return _apply_summary_to_cells(g, summary, cells)


def _dihedral_transforms(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(arr, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    unique: list[np.ndarray] = []
    for variant in variants:
        if not any(np.array_equal(variant, seen) for seen in unique):
            unique.append(variant)
    return unique


def _random_connected_shape(height: int, width: int, max_area: int = 4) -> set[tuple[int, int]]:
    area_cap = min(max_area, height * width - 1)
    assert area_cap >= 1
    target = random.randint(1, area_cap)
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))
    return cells


def _random_summary() -> np.ndarray:
    for _ in range(200):
        zero_count = random.randint(2, 4)
        nonzero_count = 9 - zero_count
        colors = [BLACK] * zero_count
        nonzeros = [random.choice(LEFT_OBJECT_COLORS) for _ in range(nonzero_count)]
        if len(set(nonzeros)) < 2:
            continue
        colors.extend(nonzeros)
        random.shuffle(colors)
        summary = np.array(colors, dtype=int).reshape(3, 3)
        if any(np.array_equal(summary, variant) for variant in _dihedral_transforms(summary)[1:]):
            continue
        return summary
    raise RuntimeError("failed to sample an asymmetric summary")


def _mask_to_coords(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _random_blue_patterns() -> list[np.ndarray]:
    seen: set[tuple[tuple[int, ...], ...]] = set()
    patterns: list[np.ndarray] = []
    while len(patterns) < 9:
        mask = np.zeros((3, 3), dtype=bool)
        count = random.randint(2, 6)
        chosen = random.sample(range(9), count)
        for idx in chosen:
            mask[idx // 3, idx % 3] = True
        key = _mask_to_coords(mask)
        if key in seen:
            continue
        seen.add(key)
        patterns.append(mask)
    random.shuffle(patterns)
    return patterns


def _place_left_table(grid: np.ndarray, slot_heights: list[int], top: int, left_col: int, summary: np.ndarray) -> None:
    row_lines = [top]
    for height in slot_heights:
        row_lines.append(row_lines[-1] + height + 1)
    col_lines = [left_col + 3 * i for i in range(4)]

    for r in row_lines:
        grid[r, col_lines[0] : col_lines[-1] + 1] = ORANGE
    for c in col_lines:
        grid[row_lines[0] : row_lines[-1] + 1, c] = ORANGE

    for i in range(3):
        r0 = row_lines[i] + 1
        r1 = row_lines[i + 1] - 1
        height = r1 - r0 + 1
        for j in range(3):
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1] - 1
            width = c1 - c0 + 1
            color = int(summary[i, j])
            if color == BLACK:
                continue
            shape = _random_connected_shape(height, width, max_area=4)
            for dr, dc in shape:
                grid[r0 + dr, c0 + dc] = color


def _place_right_lattice(grid: np.ndarray, top: int, patterns: list[np.ndarray], sep_col: int) -> list[list[tuple[int, int, int, int]]]:
    line_rows = [top + 4 * i for i in range(3)]
    line_cols = [sep_col + 1 + 4 * i for i in range(3)]
    right_end = line_cols[-1] + 3

    for r in line_rows:
        grid[r, line_cols[0] : right_end + 1] = YELLOW
    for c in line_cols:
        grid[line_rows[0] : line_rows[-1] + 4, c] = YELLOW

    cells: list[list[tuple[int, int, int, int]]] = []
    idx = 0
    for i in range(3):
        row_cells: list[tuple[int, int, int, int]] = []
        r0 = line_rows[i] + 1
        r1 = r0 + 2
        for j in range(3):
            c0 = line_cols[j] + 1
            c1 = c0 + 2
            pattern = patterns[idx]
            idx += 1
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            cell[pattern] = BLUE
            row_cells.append((r0, c0, r1, c1))
        cells.append(row_cells)
    return cells


def _alternate_fill_whole_cells(grid: np.ndarray, summary: np.ndarray, cells: list[list[tuple[int, int, int, int]]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for i in range(3):
        for j in range(3):
            color = int(summary[i, j])
            if color == BLACK:
                continue
            r0, c0, r1, c1 = cells[i][j]
            out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        sep_col = _find_separator_col(grid)
        summary, slots = _extract_left_summary(grid, sep_col)
        cells = _extract_right_cells(grid, sep_col)
    except AssertionError:
        return False

    if summary.shape != (3, 3) or len(slots) != 3 or len(cells) != 3:
        return False

    zero_count = int(np.count_nonzero(summary == BLACK))
    nonzeros = sorted(set(map(int, summary.flatten())) - {BLACK})
    if zero_count < 2 or zero_count > 7:
        return False
    if len(nonzeros) < 2:
        return False

    pattern_keys = set()
    for i in range(3):
        for j in range(3):
            r0, c0, r1, c1 = cells[i][j]
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            if cell.shape != (3, 3):
                return False
            colors = set(map(int, np.unique(cell)))
            if not colors <= {BLACK, BLUE}:
                return False
            blue_count = int(np.count_nonzero(cell == BLUE))
            if not (2 <= blue_count <= 6):
                return False
            if blue_count == 9:
                return False
            pattern_keys.add(_mask_to_coords(cell == BLUE))
    if len(pattern_keys) != 9:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(_apply_summary_to_cells(grid, summary, cells), out):
        return False

    fixed_color_summary = np.where(summary != BLACK, nonzeros[0], BLACK)
    if np.array_equal(_apply_summary_to_cells(grid, fixed_color_summary, cells), out):
        return False

    if np.array_equal(_alternate_fill_whole_cells(grid, summary, cells), out):
        return False

    dihedral = _dihedral_transforms(summary)
    for variant in dihedral[1:]:
        if np.array_equal(_apply_summary_to_cells(grid, variant, cells), out):
            return False

    for row_perm in permutations(range(3)):
        for col_perm in permutations(range(3)):
            if row_perm == (0, 1, 2) and col_perm == (0, 1, 2):
                continue
            variant = summary[list(row_perm)][:, list(col_perm)]
            if np.array_equal(_apply_summary_to_cells(grid, variant, cells), out):
                return False

    visible_noop = any(int(summary[i, j]) == BLACK for i in range(3) for j in range(3))
    visible_recolor = any(int(summary[i, j]) != BLACK for i in range(3) for j in range(3))
    if not (visible_noop and visible_recolor):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 20)
    width = random.randint(25, 29)
    sep_col = 12

    slot_heights = [random.randint(2, 3) for _ in range(3)]
    left_active_h = sum(slot_heights) + 4
    left_top = random.randint(0, height - left_active_h)
    left_col = random.randint(0, 2)

    right_active_h = 12
    right_top = random.randint(0, height - right_active_h)

    summary = _random_summary()
    patterns = _random_blue_patterns()

    grid = np.zeros((height, width), dtype=int)
    grid[:, sep_col] = GREY
    _place_left_table(grid, slot_heights, left_top, left_col, summary)
    _place_right_lattice(grid, right_top, patterns, sep_col)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
