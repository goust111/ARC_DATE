from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, crop_bbox, get_bg


BG = 0
BLUE = 1
RED = 2
GREY = 5
GEN_RETRIES = 400


def _is_regular_spacing(positions: list[int]) -> bool:
    if len(positions) < 2:
        return False
    gaps = np.diff(np.asarray(positions, dtype=int))
    return bool(np.all(gaps == gaps[0]))


def _find_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []

    for color in colors_present(grid, exclude={bg}):
        rows = [r for r in range(h) if np.count_nonzero(grid[r, :] == color) >= w - 2]
        cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == color) >= h - 2]
        if len(rows) >= 4 and len(cols) >= 4 and _is_regular_spacing(rows) and _is_regular_spacing(cols):
            candidates.append((int(color), rows, cols))

    assert len(candidates) == 1, "expected a unique regular lattice color"
    return candidates[0]


def _hole_bbox(grid: np.ndarray, rows: list[int], cols: list[int], bg: int) -> tuple[int, int, int, int]:
    holes = [(r, c) for r in rows for c in cols if int(grid[r, c]) == bg]
    assert len(holes) == 4, "expected exactly four background holes on lattice intersections"

    hole_rows = sorted({r for r, _ in holes})
    hole_cols = sorted({c for _, c in holes})
    assert len(hole_rows) == 2 and len(hole_cols) == 2, "holes must be rectangle corners"

    r0, r1 = hole_rows
    c0, c1 = hole_cols
    assert set(holes) == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return r0, c0, r1, c1


def _invert_triggered_rows(crop: np.ndarray, line_color: int, bg: int) -> np.ndarray:
    out = crop.copy()
    for r in range(out.shape[0]):
        row = out[r]
        if not np.any(row == RED):
            continue

        src = row.copy()
        line_mask = src == int(line_color)
        blue_mask = src == BLUE
        bg_mask = src == int(bg)
        red_mask = src == RED

        row[blue_mask] = int(bg)
        row[bg_mask & (~line_mask)] = BLUE
        row[red_mask] = int(bg)
        out[r] = row

    assert not np.any(out == RED), "output must consume every red trigger pixel"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the rectangle marked by four lattice holes, then invert only the cropped rows that contain a red trigger pixel.

    Input structure:
    - A black background contains a regular one-pixel-thick grey lattice with evenly spaced horizontal and vertical lines.
    - Exactly four lattice intersections are black holes instead of grey, and those four holes form the corners of one axis-aligned rectangle.
    - Cell interiors contain sparse blue pixels, and some cropped-region rows contain exactly one red trigger pixel inside a cell.

    Rule:
    - Detect the unique regular lattice color and find the four black intersection holes.
    - Crop the inclusive rectangle whose corners are those holes.
    - Process each cropped row independently: if the row contains a red trigger pixel, keep grey lattice pixels fixed, turn blue pixels to black, turn black non-lattice pixels to blue, and erase the red trigger pixel to black; rows without red stay unchanged.

    Key insight:
    - The holes determine where to crop, but the red pixels determine which cropped rows invert; the inversion is row-local and the trigger pixel is consumed by the operation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    line_color, rows, cols = _find_lattice(g, bg)
    bbox = _hole_bbox(g, rows, cols, bg)
    crop = crop_bbox(g, bbox)
    return _invert_triggered_rows(crop, line_color, bg)


def _cell_bands(lines: list[int], total: int) -> list[list[int]]:
    bounds = [-1, *lines, total]
    return [list(range(bounds[i] + 1, bounds[i + 1])) for i in range(len(bounds) - 1)]


def _choose_rectangle_indices(rng: np.random.Generator, n_lines: int) -> tuple[int, int]:
    pairs = [(i, j) for i in range(n_lines) for j in range(i + 3, n_lines)]
    choice = pairs[int(rng.integers(len(pairs)))]
    return int(choice[0]), int(choice[1])


def _scatter_blue_patterns(
    grid: np.ndarray,
    row_bands: list[list[int]],
    col_bands: list[list[int]],
    rng: np.random.Generator,
) -> None:
    for rows in row_bands:
        for cols in col_bands:
            if rng.random() < 0.58:
                cells = [(r, c) for r in rows for c in cols]
                count = int(rng.integers(1, min(3, len(cells)) + 1))
                picks = rng.choice(len(cells), size=count, replace=False)
                for idx in np.atleast_1d(picks):
                    r, c = cells[int(idx)]
                    grid[r, c] = BLUE


def _ensure_blue_on_row(
    grid: np.ndarray,
    row: int,
    eligible_cols: list[int],
    rng: np.random.Generator,
    forbidden_col: int | None = None,
) -> int:
    existing = [c for c in eligible_cols if int(grid[row, c]) == BLUE and c != forbidden_col]
    if existing:
        return int(existing[0])

    choices = [c for c in eligible_cols if c != forbidden_col]
    col = int(choices[int(rng.integers(len(choices)))])
    grid[row, col] = BLUE
    return col


def _place_single_red(
    grid: np.ndarray,
    row: int,
    eligible_cols: list[int],
    rng: np.random.Generator,
    forbidden_col: int | None = None,
) -> int:
    assert not np.any(grid[row, eligible_cols] == RED), "row must not already contain a red trigger"

    candidates = [c for c in eligible_cols if int(grid[row, c]) == BG and c != forbidden_col]
    if not candidates:
        fallback = [c for c in eligible_cols if c != forbidden_col]
        col = int(fallback[int(rng.integers(len(fallback)))])
        grid[row, col] = BG
        candidates = [col]

    col = int(candidates[int(rng.integers(len(candidates)))])
    grid[row, col] = RED
    return col


def _make_case(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    cell_size = int(rng.choice([3, 4]))
    n_h_lines = int(rng.choice([4, 5]))
    n_v_lines = int(rng.choice([4, 5]))

    h = (n_h_lines + 1) * cell_size + n_h_lines
    w = (n_v_lines + 1) * cell_size + n_v_lines
    assert max(h, w) <= 30

    line_rows = [cell_size + i * (cell_size + 1) for i in range(n_h_lines)]
    line_cols = [cell_size + i * (cell_size + 1) for i in range(n_v_lines)]

    grid = np.full((h, w), BG, dtype=int)
    grid[line_rows, :] = GREY
    grid[:, line_cols] = GREY

    hole_r0_idx, hole_r1_idx = _choose_rectangle_indices(rng, n_h_lines)
    hole_c0_idx, hole_c1_idx = _choose_rectangle_indices(rng, n_v_lines)
    hole_rows = [line_rows[hole_r0_idx], line_rows[hole_r1_idx]]
    hole_cols = [line_cols[hole_c0_idx], line_cols[hole_c1_idx]]
    for r in hole_rows:
        for c in hole_cols:
            grid[r, c] = BG

    row_bands = _cell_bands(line_rows, h)
    col_bands = _cell_bands(line_cols, w)
    _scatter_blue_patterns(grid, row_bands, col_bands, rng)

    bbox = (hole_rows[0], hole_cols[0], hole_rows[1], hole_cols[1])
    r0, c0, r1, c1 = bbox
    crop_h = r1 - r0 + 1

    line_row_set = set(line_rows)
    line_col_set = set(line_cols)
    crop_nonline_rows = [r for r in range(r0 + 1, r1) if r not in line_row_set]
    crop_nonline_cols = [c for c in range(c0 + 1, c1) if c not in line_col_set]
    assert len(crop_nonline_rows) >= 3 * cell_size
    assert crop_nonline_cols

    trigger_cap = min(len(crop_nonline_rows) // 2, crop_h // 2)
    trigger_count = int(rng.integers(1, trigger_cap + 1))
    trigger_rows = sorted(int(x) for x in rng.choice(crop_nonline_rows, size=trigger_count, replace=False).tolist())
    idle_rows = [r for r in crop_nonline_rows if r not in trigger_rows]
    assert idle_rows, "need non-trigger rows inside the crop"

    active_demo_row = int(trigger_rows[int(rng.integers(len(trigger_rows)))])
    passive_demo_row = int(idle_rows[int(rng.integers(len(idle_rows)))])

    active_blue_col = _ensure_blue_on_row(grid, active_demo_row, crop_nonline_cols, rng)
    _place_single_red(grid, active_demo_row, crop_nonline_cols, rng, forbidden_col=active_blue_col)

    for row in trigger_rows:
        if row == active_demo_row:
            continue
        _place_single_red(grid, row, crop_nonline_cols, rng)

    _ensure_blue_on_row(grid, passive_demo_row, crop_nonline_cols, rng)

    outside_crop = np.ones(grid.shape, dtype=bool)
    outside_crop[r0 : r1 + 1, c0 : c1 + 1] = False
    if not np.any(grid[outside_crop] == BLUE):
        outside_rows = [r for r in range(h) if r not in line_row_set and not (r0 <= r <= r1)]
        outside_cols = [c for c in range(w) if c not in line_col_set]
        row = int(outside_rows[int(rng.integers(len(outside_rows)))])
        col = int(outside_cols[int(rng.integers(len(outside_cols)))])
        grid[row, col] = BLUE

    meta = {
        "bbox": bbox,
        "trigger_rows": trigger_rows,
        "active_demo_row": active_demo_row,
        "passive_demo_row": passive_demo_row,
        "cell_size": cell_size,
        "line_rows": line_rows,
        "line_cols": line_cols,
    }
    return grid, meta


def _expected_output(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    crop = crop_bbox(grid, bbox)
    return _invert_triggered_rows(crop, GREY, BG)


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    bbox = tuple(int(v) for v in meta["bbox"])
    r0, c0, r1, c1 = bbox
    trigger_rows = [int(r) for r in meta["trigger_rows"]]
    active_demo_row = int(meta["active_demo_row"])
    passive_demo_row = int(meta["passive_demo_row"])
    cell_size = int(meta["cell_size"])

    assert max(grid.shape) <= 30
    assert int(get_bg(grid)) == BG
    assert np.count_nonzero(grid == RED) == len(trigger_rows)
    assert not np.any(grid[:r0, :] == RED) and not np.any(grid[r1 + 1 :, :] == RED)
    assert not np.any(grid[:, :c0] == RED) and not np.any(grid[:, c1 + 1 :] == RED)

    line_color, rows, cols = _find_lattice(grid, BG)
    assert line_color == GREY
    assert rows == meta["line_rows"] and cols == meta["line_cols"]
    assert _hole_bbox(grid, rows, cols, BG) == bbox

    row_span_steps = abs(rows.index(r1) - rows.index(r0))
    col_span_steps = abs(cols.index(c1) - cols.index(c0))
    assert row_span_steps >= 3 and col_span_steps >= 3
    assert (r1 - r0 + 1) >= 3 * cell_size + 1
    assert (c1 - c0 + 1) >= 3 * cell_size + 1

    crop = crop_bbox(grid, bbox)
    out = main(grid)
    expected = _expected_output(grid, bbox)
    assert np.array_equal(out, expected)
    assert out.shape == crop.shape
    assert not np.any(out == RED)

    rel_trigger_rows = {r - r0 for r in trigger_rows}
    rel_active = active_demo_row - r0
    rel_passive = passive_demo_row - r0

    for rr in range(crop.shape[0]):
        has_red = bool(np.any(crop[rr] == RED))
        assert has_red == (rr in rel_trigger_rows)
        if has_red:
            assert np.any(out[rr] != crop[rr]), "trigger rows must visibly change"
            assert np.array_equal(out[rr][crop[rr] == GREY], crop[rr][crop[rr] == GREY])
        else:
            assert np.array_equal(out[rr], crop[rr]), "non-trigger rows must remain unchanged"

    active_in = crop[rel_active]
    active_out = out[rel_active]
    assert np.count_nonzero(active_in == RED) == 1
    assert np.any(active_in == BLUE) and np.any(active_in == BG)
    assert np.any((active_in == BLUE) & (active_out == BG))
    assert np.any((active_in == BG) & (active_out == BLUE))

    passive_in = crop[rel_passive]
    passive_out = out[rel_passive]
    assert not np.any(passive_in == RED)
    assert np.any(passive_in == BLUE)
    assert np.array_equal(passive_in, passive_out)

    outside_crop = np.ones(grid.shape, dtype=bool)
    outside_crop[r0 : r1 + 1, c0 : c1 + 1] = False
    assert np.any(grid[outside_crop] == BLUE), "crop must not equal the bbox of all blue evidence"
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid, meta = _make_case(rng)
        try:
            if _validate_case(grid, meta):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 4")
