from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BLACK = 0
SPRITE_COLORS = (1, 2, 3, 4)
SEPARATOR = 6
ROW_COUNT = 4
COL_COUNT = 5
GEN_ATTEMPTS = 500
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _detect_separator(grid: np.ndarray) -> int:
    colors: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r, :]
        if bool(np.all(row == row[0])) and int(row[0]) != BLACK:
            colors.append(int(row[0]))
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if bool(np.all(col == col[0])) and int(col[0]) != BLACK:
            colors.append(int(col[0]))
    assert colors, "expected at least one non-black full separator line"
    return int(Counter(colors).most_common(1)[0][0])


def _full_separator_indices(grid: np.ndarray, sep: int, axis: int) -> list[int]:
    if axis == 0:
        return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == sep))]
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == sep))]


def _spans(size: int, lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line in lines + [int(size)]:
        assert line > start, "separator lines must leave non-empty cells"
        spans.append((int(start), int(line)))
        start = int(line + 1)
    assert start == size + 1
    return spans


def _extract_cells(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    sep = _detect_separator(grid)
    row_lines = _full_separator_indices(grid, sep, axis=0)
    col_lines = _full_separator_indices(grid, sep, axis=1)
    row_spans = _spans(grid.shape[0], row_lines)
    col_spans = _spans(grid.shape[1], col_lines)

    assert len(row_spans) == ROW_COUNT, f"expected {ROW_COUNT} cell rows"
    assert len(col_spans) == COL_COUNT, f"expected {COL_COUNT} cell columns"

    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "all cells must share the same size"

    cells = [
        [grid[r0:r1, c0:c1].copy() for c0, c1 in col_spans]
        for r0, r1 in row_spans
    ]
    return sep, row_spans, col_spans, cells


def _row_patterns(row_cells: list[np.ndarray]) -> tuple[int, np.ndarray, np.ndarray]:
    signatures = [cell.tobytes() for cell in row_cells]
    counts = Counter(signatures)
    assert sorted(counts.values()) == [1, 4], "each row must contain four matching cells and one anomaly"
    odd_sig = next(sig for sig, count in counts.items() if count == 1)
    common_sig = next(sig for sig, count in counts.items() if count == 4)
    odd_index = signatures.index(odd_sig)
    common_index = signatures.index(common_sig)
    return int(odd_index), row_cells[odd_index].copy(), row_cells[common_index].copy()


def _clip_paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + h)
    cc1 = min(dst.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    dst[rr0:rr1, cc0:cc1] = patch[pr0:pr1, pc0:pc1]


def _build_output(anomaly_cells: list[np.ndarray], offsets: list[int], sep: int) -> np.ndarray:
    cell_h, cell_w = anomaly_cells[0].shape
    out_h = ROW_COUNT * cell_h + (ROW_COUNT - 1)
    out_w = cell_w + max(int(v) for v in offsets)
    out = np.full((out_h, out_w), BLACK, dtype=int)
    for row in range(ROW_COUNT - 1):
        out[(row + 1) * cell_h + row, :] = int(sep)
    for row, (cell, offset) in enumerate(zip(anomaly_cells, offsets)):
        r0 = row * (cell_h + 1)
        _clip_paste(out, cell, r0, int(offset))
    return out


def _metadata(grid: np.ndarray) -> tuple[int, list[int], list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    sep, row_spans, col_spans, cells = _extract_cells(np.asarray(grid, dtype=int))
    offsets: list[int] = []
    anomaly_cells: list[np.ndarray] = []
    common_cells: list[np.ndarray] = []
    for row_cells in cells:
        odd_col, odd_cell, common_cell = _row_patterns(row_cells)
        offsets.append(int(odd_col))
        anomaly_cells.append(odd_cell)
        common_cells.append(common_cell)
    cell_h = row_spans[0][1] - row_spans[0][0]
    cell_w = col_spans[0][1] - col_spans[0][0]
    return sep, offsets, anomaly_cells, common_cells, (cell_h, cell_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each row's unique odd cell and shear the four extracted cells by their source column indices.

    Input structure:
    - The grid is split by full magenta separator lines into a 4x5 array of equal rectangular cells.
    - Each cell has a black background and contains one small 4-connected colored sprite.
    - Within each row, four cells are pixel-identical and one cell is a unique anomaly.

    Rule:
    - Detect the separator color, split the input into the 4x5 cell array, and compare the five cells in each row pixel-for-pixel.
    - For each row, extract the unique cell that appears exactly once and record its source column index.
    - Build a single-column output with the same cell height, horizontal separator lines between rows, and enough extra width for the largest recorded column index.
    - Paste row r's extracted cell into output row r, shifted right by its recorded source column index on the black canvas.

    Key insight:
    - The anomaly cell contributes both the sprite content and the horizontal offset, so extraction and placement are driven by the same row-wise evidence.
    """
    sep, offsets, anomaly_cells, _common_cells, _cell_shape = _metadata(grid)
    return _build_output(anomaly_cells, offsets, sep)


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    coords = {start}
    while len(coords) < int(area):
        frontier = []
        for r, c in coords:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        coords.add(random.choice(frontier))
    for r, c in coords:
        mask[r, c] = True
    return mask


def _random_sprite_spec(cell_h: int, cell_w: int, *, require_alt_position: bool = False) -> tuple[np.ndarray, int, int, int]:
    for _ in range(100):
        box_h = random.randint(2, min(4, cell_h))
        box_w = random.randint(2, min(4, cell_w))
        if require_alt_position and box_h == cell_h and box_w == cell_w:
            continue
        area = random.randint(2, min(5, box_h * box_w))
        mask = _random_connected_mask(box_h, box_w, area)
        top = random.randint(0, cell_h - box_h)
        left = random.randint(0, cell_w - box_w)
        return mask, top, left, random.choice(SPRITE_COLORS)
    raise AssertionError("failed to sample sprite spec")


def _draw_cell(cell_h: int, cell_w: int, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    cell = np.full((cell_h, cell_w), BLACK, dtype=int)
    h, w = mask.shape
    view = cell[top : top + h, left : left + w]
    view[mask] = int(color)
    return cell


def _different_position(cell_h: int, cell_w: int, mask: np.ndarray, top: int, left: int) -> tuple[int, int]:
    h, w = mask.shape
    choices = [
        (r, c)
        for r in range(cell_h - h + 1)
        for c in range(cell_w - w + 1)
        if (r, c) != (top, left)
    ]
    assert choices, "position variant requires at least two placements"
    return random.choice(choices)


def _different_shape(shape: tuple[int, int], original: np.ndarray) -> np.ndarray:
    h, w = shape
    area = int(original.sum())
    attempts = []
    if area > 2:
        attempts.append(area - 1)
    attempts.append(area)
    if area < min(5, h * w):
        attempts.append(area + 1)
    for target_area in attempts:
        for _ in range(40):
            candidate = _random_connected_mask(h, w, target_area)
            if not np.array_equal(candidate, original):
                return candidate
    raise AssertionError("failed to sample a distinct shape")


def _make_row_cells(cell_h: int, cell_w: int, relation: str, odd_col: int) -> list[np.ndarray]:
    base_mask, base_top, base_left, base_color = _random_sprite_spec(
        cell_h, cell_w, require_alt_position=(relation == "position")
    )
    base_cell = _draw_cell(cell_h, cell_w, base_mask, base_top, base_left, base_color)

    if relation == "color":
        anomaly_color = random.choice([c for c in SPRITE_COLORS if c != base_color])
        anomaly_cell = _draw_cell(cell_h, cell_w, base_mask, base_top, base_left, anomaly_color)
    elif relation == "shape":
        anomaly_mask = _different_shape(base_mask.shape, base_mask)
        anomaly_cell = _draw_cell(cell_h, cell_w, anomaly_mask, base_top, base_left, base_color)
    elif relation == "position":
        anomaly_top, anomaly_left = _different_position(cell_h, cell_w, base_mask, base_top, base_left)
        anomaly_cell = _draw_cell(cell_h, cell_w, base_mask, anomaly_top, anomaly_left, base_color)
    else:
        anomaly_mask, anomaly_top, anomaly_left, anomaly_color = _random_sprite_spec(cell_h, cell_w)
        anomaly_cell = _draw_cell(cell_h, cell_w, anomaly_mask, anomaly_top, anomaly_left, anomaly_color)

    assert not np.array_equal(base_cell, anomaly_cell), "anomaly cell must differ from the repeated cell"

    row_cells = [base_cell.copy() for _ in range(COL_COUNT)]
    row_cells[int(odd_col)] = anomaly_cell
    return row_cells


def _nonmonotonic_offset_sequence() -> list[int]:
    candidates = []
    pool = list(range(COL_COUNT))
    for omitted in pool:
        seq = [v for v in pool if v != omitted]
        for perm in set(itertools.permutations(seq)):
            values = list(map(int, perm))
            if values != sorted(values) and values != sorted(values, reverse=True):
                candidates.append(values)
    return random.choice(candidates)


def _compose_grid(rows: list[list[np.ndarray]], cell_h: int, cell_w: int) -> np.ndarray:
    grid_h = ROW_COUNT * cell_h + (ROW_COUNT - 1)
    grid_w = COL_COUNT * cell_w + (COL_COUNT - 1)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for row in range(ROW_COUNT - 1):
        grid[(row + 1) * cell_h + row, :] = SEPARATOR
    for col in range(COL_COUNT - 1):
        grid[:, (col + 1) * cell_w + col] = SEPARATOR
    for r, row_cells in enumerate(rows):
        for c, cell in enumerate(row_cells):
            r0 = r * (cell_h + 1)
            c0 = c * (cell_w + 1)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    sep, offsets, anomaly_cells, common_cells, (_cell_h, _cell_w) = _metadata(inp)
    if len(set(offsets)) != ROW_COUNT:
        return False
    if offsets == sorted(offsets) or offsets == sorted(offsets, reverse=True):
        return False

    correct = _build_output(anomaly_cells, offsets, sep)
    if not np.array_equal(main(inp), correct):
        return False

    zero_shift = np.full_like(correct, BLACK)
    cell_h = anomaly_cells[0].shape[0]
    for row, cell in enumerate(anomaly_cells):
        _clip_paste(zero_shift, cell, row * (cell_h + 1), 0)
    for row in range(ROW_COUNT - 1):
        zero_shift[(row + 1) * cell_h + row, :] = sep
    if np.array_equal(correct, zero_shift):
        return False

    common_output = _build_output(common_cells, offsets, sep)
    if np.array_equal(correct, common_output):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    cell_h = random.randint(4, 6)
    cell_w = random.randint(4, 5)
    if ROW_COUNT * cell_h + (ROW_COUNT - 1) > 30:
        return None
    if COL_COUNT * cell_w + (COL_COUNT - 1) > 30:
        return None

    forced = ["color", random.choice(["shape", "position"])]
    extras = [random.choice(["color", "shape", "position", "mixed"]) for _ in range(ROW_COUNT - len(forced))]
    relations = forced + extras
    random.shuffle(relations)

    odd_cols = _nonmonotonic_offset_sequence()
    rows: list[list[np.ndarray]] = []
    for relation, forced_odd_col in zip(relations, odd_cols):
        row_cells = _make_row_cells(cell_h, cell_w, relation, forced_odd_col)
        rows.append(row_cells)

    inp = _compose_grid(rows, cell_h, cell_w)
    if max(inp.shape) > 30:
        return None
    if not _semantic_checks(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anomaly-shear ARC input")
