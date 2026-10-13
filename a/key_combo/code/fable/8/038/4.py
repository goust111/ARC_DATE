from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BG = int(Color.BLACK)
ROW_HEADER_COLOR = int(Color.GREY)
CELL = 4
BODY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GAP_CHOICES = (1, 2)
MARGIN_CHOICES = (1, 2)
ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
MAX_ATTEMPTS = 400

SHAPE_LIBRARY = [
    np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _foreground_color(patch: np.ndarray) -> int:
    values = sorted(set(map(int, patch.flatten())) - {BG})
    assert len(values) == 1
    return values[0]


def _extract_layout(grid: np.ndarray) -> tuple[list[dict], list[int], list[int], dict[tuple[int, int], np.ndarray], tuple[int, int]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected visible table pieces"

    sizes = {(comp["bbox"][2] - comp["bbox"][0] + 1, comp["bbox"][3] - comp["bbox"][1] + 1) for comp in comps}
    assert len(sizes) == 1, "all pieces must share one cell size"
    cell_shape = sizes.pop()

    row_starts = sorted({int(comp["bbox"][0]) for comp in comps})
    col_starts = sorted({int(comp["bbox"][1]) for comp in comps})
    assert len(row_starts) >= 3 and len(col_starts) >= 3, "expected header row/col plus body rows/cols"

    row_index = {r: i for i, r in enumerate(row_starts)}
    col_index = {c: j for j, c in enumerate(col_starts)}
    cell_map: dict[tuple[int, int], np.ndarray] = {}
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        key = (row_index[r0], col_index[c0])
        assert key not in cell_map, "multiple components in one table slot"
        cell_map[key] = crop_bbox(grid, comp["bbox"])

    assert (0, 0) not in cell_map, "top-left corner cell must be blank"
    return comps, row_starts, col_starts, cell_map, cell_shape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a bidirectional sprite multiplication table by recovering missing row-shape and column-color headers, then filling blank body cells.

    Input structure:
    - A black background contains a rectangular table laid out on a regular lattice of equal-sized cells.
    - The left header column holds grey sprite masks, one per body row, and the top header row holds solid color blocks, one per body column.
    - Interior body cells, when present, contain the row sprite mask recolored with the column color; some body cells and some headers may be missing.

    Rule:
    - Recover every missing row header from any existing body cell in that row by taking its sprite mask.
    - Recover every missing column header from any existing body cell in that column by taking its color.
    - Leave all existing content unchanged, draw recovered headers in the header bands, and fill each blank body slot with the Cartesian-product combination of its row mask and column color.

    Key insight:
    - Shape information propagates horizontally by row while color information propagates vertically by column, so solving requires combining the two orthogonal constraints rather than copying a single nearby cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _comps, row_starts, col_starts, cell_map, cell_shape = _extract_layout(g)
    cell_h, cell_w = cell_shape
    n_rows = len(row_starts) - 1
    n_cols = len(col_starts) - 1

    row_masks: list[np.ndarray] = []
    for i in range(1, n_rows + 1):
        candidates: list[np.ndarray] = []
        header_patch = cell_map.get((i, 0))
        if header_patch is not None:
            assert _foreground_color(header_patch) == ROW_HEADER_COLOR
            candidates.append(header_patch != BG)
        for j in range(1, n_cols + 1):
            body_patch = cell_map.get((i, j))
            if body_patch is not None:
                candidates.append(body_patch != BG)
        assert candidates, "each row must be recoverable from its header or body"
        ref = candidates[0]
        assert all(np.array_equal(mask, ref) for mask in candidates[1:]), "inconsistent row shape"
        row_masks.append(ref)

    col_colors: list[int] = []
    for j in range(1, n_cols + 1):
        candidates: list[int] = []
        header_patch = cell_map.get((0, j))
        if header_patch is not None:
            header_color = _foreground_color(header_patch)
            assert np.all(header_patch == header_color), "column header must be a solid block"
            candidates.append(header_color)
        for i in range(1, n_rows + 1):
            body_patch = cell_map.get((i, j))
            if body_patch is not None:
                candidates.append(_foreground_color(body_patch))
        assert candidates, "each column must be recoverable from its header or body"
        ref = candidates[0]
        assert ref not in (BG, ROW_HEADER_COLOR)
        assert all(color == ref for color in candidates[1:]), "inconsistent column color"
        col_colors.append(ref)

    out = g.copy()
    for i in range(1, n_rows + 1):
        header_patch = _mask_to_patch(row_masks[i - 1], ROW_HEADER_COLOR)
        if (i, 0) in cell_map:
            assert np.array_equal(cell_map[(i, 0)], header_patch), "row header disagrees with recovered row mask"
        else:
            paste(out, header_patch, row_starts[i], col_starts[0], transparent=BG)

    for j in range(1, n_cols + 1):
        header_patch = np.full((cell_h, cell_w), col_colors[j - 1], dtype=int)
        if (0, j) in cell_map:
            assert np.array_equal(cell_map[(0, j)], header_patch), "column header disagrees with recovered column color"
        else:
            paste(out, header_patch, row_starts[0], col_starts[j])

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            body_patch = _mask_to_patch(row_masks[i - 1], col_colors[j - 1])
            if (i, j) in cell_map:
                assert np.array_equal(cell_map[(i, j)], body_patch), "body cell disagrees with row/column constraints"
            else:
                paste(out, body_patch, row_starts[i], col_starts[j], transparent=BG)

    return out


def _axis_positions(data_count: int) -> tuple[int, list[int]]:
    total_cells = data_count + 1
    options: list[tuple[int, int, list[int]]] = []
    for gap in GAP_CHOICES:
        for margin0 in MARGIN_CHOICES:
            for margin1 in MARGIN_CHOICES:
                size = margin0 + margin1 + total_cells * CELL + (total_cells - 1) * gap
                if size <= 30:
                    starts = [margin0 + i * (CELL + gap) for i in range(total_cells)]
                    options.append((size, gap, starts))
    assert options
    size, _gap, starts = random.choice(options)
    return size, starts


def _choose_header_case(n_rows: int, n_cols: int) -> tuple[set[int], set[int]]:
    case = random.choices(["none", "row", "col", "both"], weights=[1, 2, 2, 4])[0]
    row_missing: set[int] = set()
    col_missing: set[int] = set()

    if case in {"row", "both"}:
        row_missing = set(random.sample(range(n_rows), random.randint(1, min(2, n_rows - 1))))
    if case in {"col", "both"}:
        col_missing = set(random.sample(range(n_cols), random.randint(1, min(2, n_cols - 1))))
    return row_missing, col_missing


def _remove_if_valid(present: np.ndarray, r: int, c: int) -> bool:
    if not present[r, c]:
        return False
    if present[r].sum() <= 1 or present[:, c].sum() <= 1:
        return False
    present[r, c] = False
    return True


def _body_mask(n_rows: int, n_cols: int, row_missing: set[int], col_missing: set[int]) -> np.ndarray | None:
    present = np.ones((n_rows, n_cols), dtype=bool)

    if row_missing and col_missing:
        forced = random.choice([(r, c) for r in row_missing for c in col_missing])
    elif row_missing:
        forced = (random.choice(tuple(row_missing)), random.randrange(n_cols))
    elif col_missing:
        forced = (random.randrange(n_rows), random.choice(tuple(col_missing)))
    else:
        forced = (random.randrange(n_rows), random.randrange(n_cols))

    if not _remove_if_valid(present, *forced):
        return None

    removable = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if present[r, c] and present[r].sum() > 1 and present[:, c].sum() > 1
    ]
    random.shuffle(removable)
    extra_limit = min(len(removable), max(1, (n_rows * n_cols) // 2))
    extra_count = random.randint(0, extra_limit)
    removed = 1
    for r, c in removable:
        if removed >= 1 + extra_count:
            break
        if _remove_if_valid(present, r, c):
            removed += 1

    if present.all():
        return None
    if np.any(present.sum(axis=1) == 0) or np.any(present.sum(axis=0) == 0):
        return None
    if row_missing and col_missing and not any(not present[r, c] for r in row_missing for c in col_missing):
        return None
    if row_missing and not any((~present[r]).any() for r in row_missing):
        return None
    if col_missing and not any((~present[:, c]).any() for c in col_missing):
        return None
    return present


def _render_full_output(
    row_masks: list[np.ndarray],
    col_colors: list[int],
    row_starts: list[int],
    col_starts: list[int],
    height: int,
    width: int,
) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for i, mask in enumerate(row_masks, start=1):
        paste(out, _mask_to_patch(mask, ROW_HEADER_COLOR), row_starts[i], col_starts[0], transparent=BG)
    for j, color in enumerate(col_colors, start=1):
        paste(out, np.full((CELL, CELL), color, dtype=int), row_starts[0], col_starts[j])
    for i, mask in enumerate(row_masks, start=1):
        for j, color in enumerate(col_colors, start=1):
            paste(out, _mask_to_patch(mask, color), row_starts[i], col_starts[j], transparent=BG)
    return out


def _erase_cell(grid: np.ndarray, r0: int, c0: int) -> None:
    grid[r0 : r0 + CELL, c0 : c0 + CELL] = BG


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    row_masks: list[np.ndarray],
    col_colors: list[int],
    row_missing: set[int],
    col_missing: set[int],
    body_present: np.ndarray,
) -> bool:
    n_rows = len(row_masks)
    n_cols = len(col_colors)

    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False
    if len({mask.tobytes() for mask in row_masks}) != n_rows:
        return False
    if len(set(col_colors)) != n_cols:
        return False
    if any(body_present[r].sum() == 0 for r in range(n_rows)):
        return False
    if any(body_present[:, c].sum() == 0 for c in range(n_cols)):
        return False
    if body_present.all():
        return False

    missing_cells = [(r, c) for r in range(n_rows) for c in range(n_cols) if not body_present[r, c]]
    for r, c in missing_cells:
        if body_present[r].sum() < 1 or body_present[:, c].sum() < 1:
            return False

    if row_missing and col_missing and not any((r in row_missing and c in col_missing) for r, c in missing_cells):
        return False
    if row_missing and not any(r in row_missing for r, _c in missing_cells):
        return False
    if col_missing and not any(c in col_missing for _r, c in missing_cells):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*ROW_COUNT_RANGE)
    n_cols = random.randint(*COL_COUNT_RANGE)
    height, row_starts = _axis_positions(n_rows)
    width, col_starts = _axis_positions(n_cols)

    row_masks = random.sample(SHAPE_LIBRARY, n_rows)
    col_colors = random.sample(BODY_COLORS, n_cols)
    row_missing, col_missing = _choose_header_case(n_rows, n_cols)
    body_present = _body_mask(n_rows, n_cols, row_missing, col_missing)
    if body_present is None:
        return None

    out = _render_full_output(row_masks, col_colors, row_starts, col_starts, height, width)
    inp = out.copy()

    for r in row_missing:
        _erase_cell(inp, row_starts[r + 1], col_starts[0])
    for c in col_missing:
        _erase_cell(inp, row_starts[0], col_starts[c + 1])
    for r in range(n_rows):
        for c in range(n_cols):
            if not body_present[r, c]:
                _erase_cell(inp, row_starts[r + 1], col_starts[c + 1])

    if not _validate_case(inp, out, row_masks, col_colors, row_missing, col_missing, body_present):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
