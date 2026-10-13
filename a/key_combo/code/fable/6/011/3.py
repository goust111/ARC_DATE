from __future__ import annotations

import itertools
import random

import numpy as np


GREY = 5
CELL_SIZE = 5
PATTERN_SLICE = slice(1, 4)
OPS = ("I", "LR", "UD", "R180")
CORNERS = (
    ("tl", (0, 0)),
    ("tr", (0, CELL_SIZE - 1)),
    ("bl", (CELL_SIZE - 1, 0)),
    ("br", (CELL_SIZE - 1, CELL_SIZE - 1)),
)
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _apply_op(pattern: np.ndarray, op: str) -> np.ndarray:
    if op == "I":
        return pattern.copy()
    if op == "LR":
        return np.fliplr(pattern)
    if op == "UD":
        return np.flipud(pattern)
    if op == "R180":
        return np.rot90(pattern, 2)
    raise ValueError(f"unknown op {op}")


def _find_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected one separator color"
    sep = next(iter(sep_colors))

    sep_rows = [r for r in range(h) if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == sep)]
    assert sep_rows and sep_cols, "expected separator rows and columns"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands, "expected at least one cell band"
    assert {r1 - r0 + 1 for r0, r1 in row_bands} == {CELL_SIZE}, "unexpected cell height"
    assert {c1 - c0 + 1 for c0, c1 in col_bands} == {CELL_SIZE}, "unexpected cell width"
    return sep, row_bands, col_bands


def _cell_records(
    grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            records.append({"ri": ri, "ci": ci, "r0": r0, "c0": c0, "cell": cell})
    return records


def _find_marker_color(records: list[dict[str, object]]) -> int:
    corner_counts: dict[int, int] = {}
    for rec in records:
        cell = np.asarray(rec["cell"], dtype=int)
        for _, (r, c) in CORNERS:
            color = int(cell[r, c])
            corner_counts[color] = corner_counts.get(color, 0) + 1

    candidates = []
    for color, count in corner_counts.items():
        if count != len(records):
            continue
        if all(sum(int(np.asarray(rec["cell"])[r, c] == color) for _, (r, c) in CORNERS) == 1 for rec in records):
            candidates.append(color)
    assert len(candidates) == 1, "expected one marker color"
    return int(candidates[0])


def _corner_name(cell: np.ndarray, marker_color: int) -> str:
    hits = [name for name, (r, c) in CORNERS if int(cell[r, c]) == marker_color]
    assert len(hits) == 1, "each cell must have exactly one marked corner"
    return hits[0]


def _decode_missing_pattern(records: list[dict[str, object]], marker_color: int) -> tuple[np.ndarray, dict[str, object]]:
    visible: list[tuple[str, np.ndarray]] = []
    missing_rec: dict[str, object] | None = None
    for rec in records:
        cell = np.asarray(rec["cell"], dtype=int)
        corner = _corner_name(cell, marker_color)
        pattern = cell[PATTERN_SLICE, PATTERN_SLICE]
        if np.all(pattern == GREY):
            assert missing_rec is None, "expected exactly one masked cell"
            missing_rec = {"corner": corner, **rec}
        else:
            visible.append((corner, pattern.copy()))
    assert missing_rec is not None, "missing cell not found"
    assert visible, "expected visible cells"

    predicted: dict[bytes, np.ndarray] = {}
    for _, pattern0 in visible:
        for base_op in OPS:
            base = _apply_op(pattern0, base_op)
            for perm in itertools.permutations(OPS):
                mapping = {name: op for (name, _), op in zip(CORNERS, perm, strict=True)}
                if all(np.array_equal(pattern, _apply_op(base, mapping[corner])) for corner, pattern in visible):
                    fill = _apply_op(base, mapping[str(missing_rec["corner"])])
                    predicted[fill.tobytes()] = fill.copy()

    assert len(predicted) == 1, "decoded fill must be unique"
    fill = next(iter(predicted.values()))
    return fill, missing_rec


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Restore the grey-masked cell in a separator lattice of transformed 3x3 motifs.

    Each cell contains the same asymmetric 3x3 motif under one of four transforms:
    identity, left-right mirror, up-down mirror, or 180-degree rotation. A single
    marker pixel in a cell corner encodes which transform was used, but the mapping
    from corner to transform is not given directly and must be inferred from the
    visible cells. Exactly one cell has its 3x3 motif replaced by grey while its
    marker remains visible.
    """
    grid = np.asarray(input_grid, dtype=int)
    sep, row_bands, col_bands = _find_lattice(grid)
    _ = sep
    records = _cell_records(grid, row_bands, col_bands)
    marker_color = _find_marker_color(records)
    fill, missing_rec = _decode_missing_pattern(records, marker_color)

    out = grid.copy()
    r0 = int(missing_rec["r0"])
    c0 = int(missing_rec["c0"])
    out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = fill
    return out


def _label_layout(rows: int, cols: int) -> np.ndarray:
    names = [name for name, _ in CORNERS]
    total = rows * cols
    while True:
        flat = [random.choice(names) for _ in range(total)]
        counts = {name: flat.count(name) for name in names}
        if min(counts.values()) < 2:
            continue
        arr = np.array(flat, dtype=object).reshape(rows, cols)
        if np.array_equal(arr, np.fliplr(arr)):
            continue
        if np.array_equal(arr, np.flipud(arr)):
            continue
        if np.array_equal(arr, np.rot90(arr, 2)):
            continue
        return arr


def _base_pattern(colors: list[int]) -> np.ndarray:
    while True:
        palette = random.sample(colors, k=random.randint(3, min(5, len(colors))))
        pattern = np.array([random.choice(palette) for _ in range(9)], dtype=int).reshape(3, 3)
        if len(set(map(int, pattern.flat))) < 3:
            continue
        variants = {_apply_op(pattern, op).tobytes() for op in OPS}
        if len(variants) != 4:
            continue
        return pattern


def _render_puzzle(
    rows: int,
    cols: int,
    sep: int,
    bg: int,
    marker: int,
    base: np.ndarray,
    mapping: dict[str, str],
    labels: np.ndarray,
    missing_rc: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h = rows * CELL_SIZE + (rows - 1)
    w = cols * CELL_SIZE + (cols - 1)
    grid = np.full((h, w), sep, dtype=int)
    solved = np.full((h, w), sep, dtype=int)
    for ri in range(rows):
        for ci in range(cols):
            label = str(labels[ri, ci])
            cell = np.full((CELL_SIZE, CELL_SIZE), bg, dtype=int)
            pattern = _apply_op(base, mapping[label])
            cell[PATTERN_SLICE, PATTERN_SLICE] = pattern
            for name, (r, c) in CORNERS:
                if name == label:
                    cell[r, c] = marker
                    break
            r0 = ri * (CELL_SIZE + 1)
            c0 = ci * (CELL_SIZE + 1)
            solved[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = cell
            if (ri, ci) == missing_rc:
                cell = cell.copy()
                cell[PATTERN_SLICE, PATTERN_SLICE] = GREY
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = cell
    return grid, solved


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    if rows * CELL_SIZE + (rows - 1) > 30 or cols * CELL_SIZE + (cols - 1) > 30:
        return None

    sep, bg, marker = random.sample(PALETTE, k=3)
    motif_colors = [c for c in PALETTE if c not in {sep, bg, marker}]
    base = _base_pattern(motif_colors)
    labels = _label_layout(rows, cols)
    missing_rc = (random.randrange(rows), random.randrange(cols))
    mapping = {name: op for (name, _), op in zip(CORNERS, random.sample(list(OPS), k=4), strict=True)}

    grid, solved = _render_puzzle(rows, cols, sep, bg, marker, base, mapping, labels, missing_rc)
    try:
        if not np.array_equal(main(grid), solved):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
