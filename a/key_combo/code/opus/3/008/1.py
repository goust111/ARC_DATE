from __future__ import annotations

import numpy as np


BG = 0
GRID = 5
PATTERN_COLORS = (1, 2, 3, 7)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _cell_slice(lines: list[int], idx: int) -> slice:
    return slice(lines[idx] + 1, lines[idx + 1])


def _cell_patch(grid: np.ndarray, line_rows: list[int], line_cols: list[int], rr: int, cc: int) -> np.ndarray:
    return grid[_cell_slice(line_rows, rr), _cell_slice(line_cols, cc)]


def _foreground_count(patch: np.ndarray) -> int:
    return int(np.count_nonzero((patch != BG) & (patch != GRID)))


def _mirror_patch(patch: np.ndarray, axis_kind: str) -> np.ndarray:
    if axis_kind == "horizontal":
        return np.flipud(patch)
    if axis_kind == "vertical":
        return np.fliplr(patch)
    raise ValueError(f"unknown axis kind: {axis_kind}")


def _detect_grid_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = map(int, grid.shape)
    row_counts = np.count_nonzero(grid == GRID, axis=1)
    col_counts = np.count_nonzero(grid == GRID, axis=0)

    line_rows = [int(r) for r, count in enumerate(row_counts) if int(count) > w // 2]
    line_cols = [int(c) for c, count in enumerate(col_counts) if int(count) > h // 2]

    assert len(line_rows) >= 3 and len(line_cols) >= 3, "expected a regular gray grid"
    row_steps = {line_rows[i + 1] - line_rows[i] for i in range(len(line_rows) - 1)}
    col_steps = {line_cols[i + 1] - line_cols[i] for i in range(len(line_cols) - 1)}
    assert len(row_steps) == 1 and len(col_steps) == 1, "grid spacing must be regular"
    assert next(iter(row_steps)) >= 3 and next(iter(col_steps)) >= 3, "cells must have interior area"
    return line_rows, line_cols


def _find_missing_crossings(grid: np.ndarray, line_rows: list[int], line_cols: list[int]) -> list[tuple[int, int]]:
    missing = [(r, c) for r in line_rows for c in line_cols if int(grid[r, c]) == BG]
    assert len(missing) == 2, "expected exactly two missing gray crossings"
    return missing


def _axis_from_missing(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
) -> tuple[str, int]:
    missing = _find_missing_crossings(grid, line_rows, line_cols)
    if missing[0][0] == missing[1][0]:
        axis_kind = "horizontal"
        axis_value = int(missing[0][0])
        assert axis_value in line_rows, "missing crossings must lie on a grid row"
        axis_idx = line_rows.index(axis_value)
        n_rows = len(line_rows) - 1
        assert n_rows % 2 == 0 and axis_idx == n_rows // 2, "horizontal axis must split cell rows equally"
        return axis_kind, axis_idx
    if missing[0][1] == missing[1][1]:
        axis_kind = "vertical"
        axis_value = int(missing[0][1])
        assert axis_value in line_cols, "missing crossings must lie on a grid column"
        axis_idx = line_cols.index(axis_value)
        n_cols = len(line_cols) - 1
        assert n_cols % 2 == 0 and axis_idx == n_cols // 2, "vertical axis must split cell columns equally"
        return axis_kind, axis_idx
    raise AssertionError("missing crossings must align on one row or one column")


def _repair_with_axis(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    axis_kind: str,
    *,
    require_unique_sparse: bool = True,
) -> tuple[np.ndarray | None, str | None, tuple[int, int] | None]:
    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    axis_idx = n_rows // 2 if axis_kind == "horizontal" else n_cols // 2

    out = grid.copy()
    changed_pairs = 0

    if axis_kind == "horizontal":
        assert n_rows % 2 == 0, "horizontal repair requires an even number of cell rows"
        top_total = sum(
            _foreground_count(_cell_patch(grid, line_rows, line_cols, rr, cc))
            for rr in range(axis_idx)
            for cc in range(n_cols)
        )
        bottom_total = sum(
            _foreground_count(_cell_patch(grid, line_rows, line_cols, rr, cc))
            for rr in range(axis_idx, n_rows)
            for cc in range(n_cols)
        )
        if top_total == bottom_total:
            assert not require_unique_sparse, "horizontal sparse side would be ambiguous"
            return None, None, (top_total, bottom_total)
        sparse_side = "top" if top_total < bottom_total else "bottom"

        dense_rows = range(axis_idx, n_rows) if sparse_side == "top" else range(axis_idx)
        assert all(
            _foreground_count(_cell_patch(grid, line_rows, line_cols, rr, cc)) > 0
            for rr in dense_rows
            for cc in range(n_cols)
        ), "dense side cells must all contain patterns"

        for top_r in range(axis_idx):
            bottom_r = n_rows - 1 - top_r
            for cc in range(n_cols):
                top_sl = (_cell_slice(line_rows, top_r), _cell_slice(line_cols, cc))
                bottom_sl = (_cell_slice(line_rows, bottom_r), _cell_slice(line_cols, cc))
                if sparse_side == "top":
                    source = grid[bottom_sl]
                    target = grid[top_sl]
                    out[top_sl] = _mirror_patch(source, axis_kind)
                else:
                    source = grid[top_sl]
                    target = grid[bottom_sl]
                    out[bottom_sl] = _mirror_patch(source, axis_kind)
                if not np.array_equal(target, _mirror_patch(source, axis_kind)):
                    changed_pairs += 1
        assert changed_pairs > 0, "repair must change at least one sparse-side cell"
        return out, sparse_side, (top_total, bottom_total)

    assert axis_kind == "vertical", "axis kind must be horizontal or vertical"
    assert n_cols % 2 == 0, "vertical repair requires an even number of cell columns"
    left_total = sum(
        _foreground_count(_cell_patch(grid, line_rows, line_cols, rr, cc))
        for rr in range(n_rows)
        for cc in range(axis_idx)
    )
    right_total = sum(
        _foreground_count(_cell_patch(grid, line_rows, line_cols, rr, cc))
        for rr in range(n_rows)
        for cc in range(axis_idx, n_cols)
    )
    if left_total == right_total:
        assert not require_unique_sparse, "vertical sparse side would be ambiguous"
        return None, None, (left_total, right_total)
    sparse_side = "left" if left_total < right_total else "right"

    dense_cols = range(axis_idx, n_cols) if sparse_side == "left" else range(axis_idx)
    assert all(
        _foreground_count(_cell_patch(grid, line_rows, line_cols, rr, cc)) > 0
        for rr in range(n_rows)
        for cc in dense_cols
    ), "dense side cells must all contain patterns"

    for rr in range(n_rows):
        for left_c in range(axis_idx):
            right_c = n_cols - 1 - left_c
            left_sl = (_cell_slice(line_rows, rr), _cell_slice(line_cols, left_c))
            right_sl = (_cell_slice(line_rows, rr), _cell_slice(line_cols, right_c))
            if sparse_side == "left":
                source = grid[right_sl]
                target = grid[left_sl]
                out[left_sl] = _mirror_patch(source, axis_kind)
            else:
                source = grid[left_sl]
                target = grid[right_sl]
                out[right_sl] = _mirror_patch(source, axis_kind)
            if not np.array_equal(target, _mirror_patch(source, axis_kind)):
                changed_pairs += 1
    assert changed_pairs > 0, "repair must change at least one sparse-side cell"
    return out, sparse_side, (left_total, right_total)


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the sparser half of a gridded pattern by mirroring the denser half across the axis marked by two missing crossings.

    Input structure:
    - A black background contains a regular single-pixel gray grid that partitions the canvas into equal rectangular cells.
    - Each populated cell holds a small 4-connected pattern drawn with non-gray, non-black colors.
    - Exactly two gray grid intersections are missing and appear as black pixels on one common grid line.

    Rule:
    - Detect the regular gray grid and find the two missing intersections.
    - Their shared grid line is the mirror axis: horizontal if they share a row, vertical if they share a column.
    - Compare total foreground pixel counts on the two sides of that axis; the side with fewer foreground pixels is the sparse side.
    - Replace every sparse-side cell with the mirrored contents of its paired cell across the axis, while leaving the denser side and all grid lines unchanged.

    Key insight:
    - The missing crossings do not mark damaged cells directly; they mark the symmetry axis, and density decides which side must be overwritten by mirrored cell contents.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"

    line_rows, line_cols = _detect_grid_lines(g)
    axis_kind, axis_idx = _axis_from_missing(g, line_rows, line_cols)
    expected_idx = (len(line_rows) - 1) // 2 if axis_kind == "horizontal" else (len(line_cols) - 1) // 2
    assert axis_idx == expected_idx, "axis must be the central splitter for the corresponding side"

    out, _sparse_side, _totals = _repair_with_axis(g, line_rows, line_cols, axis_kind, require_unique_sparse=True)
    assert out is not None
    return out


def _shape_order(cell_size: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    start = (int(rng.integers(cell_size)), int(rng.integers(cell_size)))
    order = [start]
    used = {start}
    while len(order) < count:
        frontier: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in order:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in used and (nr, nc) not in seen:
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
        choice = frontier[int(rng.integers(len(frontier)))]
        order.append(choice)
        used.add(choice)
    return order


def _mirror_coords(order: list[tuple[int, int]], cell_size: int, axis_kind: str) -> list[tuple[int, int]]:
    if axis_kind == "horizontal":
        return [(cell_size - 1 - r, c) for r, c in order]
    return [(r, cell_size - 1 - c) for r, c in order]


def _sample_colors(count: int, rng: np.random.Generator) -> list[int]:
    palette_size = int(rng.integers(1, min(3, count) + 1))
    palette = [int(x) for x in rng.choice(np.array(PATTERN_COLORS, dtype=int), size=palette_size, replace=False)]
    return [int(palette[int(rng.integers(len(palette)))]) for _ in range(count)]


def _render_patch(cell_size: int, order: list[tuple[int, int]], colors: list[int], keep: int | None = None) -> np.ndarray:
    patch = np.zeros((cell_size, cell_size), dtype=int)
    limit = len(order) if keep is None else keep
    for (r, c), color in zip(order[:limit], colors[:limit]):
        patch[r, c] = int(color)
    return patch


def _make_base_grid(n_rows: int, n_cols: int, cell_size: int) -> tuple[np.ndarray, list[int], list[int]]:
    height = n_rows * cell_size + n_rows + 1
    width = n_cols * cell_size + n_cols + 1
    grid = np.zeros((height, width), dtype=int)
    line_rows = [idx * (cell_size + 1) for idx in range(n_rows + 1)]
    line_cols = [idx * (cell_size + 1) for idx in range(n_cols + 1)]
    grid[line_rows, :] = GRID
    grid[:, line_cols] = GRID
    return grid, line_rows, line_cols


def _place_patch(
    grid: np.ndarray,
    line_rows: list[int],
    line_cols: list[int],
    rr: int,
    cc: int,
    patch: np.ndarray,
) -> None:
    grid[_cell_slice(line_rows, rr), _cell_slice(line_cols, cc)] = patch


def _opposite_axis(axis_kind: str) -> str:
    return "vertical" if axis_kind == "horizontal" else "horizontal"


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    meta: dict,
) -> bool:
    try:
        line_rows, line_cols = _detect_grid_lines(inp)
        axis_kind, axis_idx = _axis_from_missing(inp, line_rows, line_cols)
        if axis_kind != meta["axis_kind"]:
            return False
        if axis_idx != meta["axis_idx"]:
            return False

        actual, sparse_side, totals = _repair_with_axis(inp, line_rows, line_cols, axis_kind, require_unique_sparse=True)
        if actual is None or sparse_side != meta["sparse_side"] or totals != meta["totals"]:
            return False
        if not np.array_equal(actual, expected):
            return False

        sparse_counts: list[int] = []
        full_sparse_counts: list[int] = []
        needs_flip = False
        for dense_rr, dense_cc, sparse_rr, sparse_cc in meta["pairs"]:
            dense_patch = _cell_patch(inp, line_rows, line_cols, dense_rr, dense_cc)
            sparse_patch = _cell_patch(inp, line_rows, line_cols, sparse_rr, sparse_cc)
            full_sparse_patch = _cell_patch(expected, line_rows, line_cols, sparse_rr, sparse_cc)
            sparse_counts.append(_foreground_count(sparse_patch))
            full_sparse_counts.append(_foreground_count(full_sparse_patch))
            if not np.array_equal(full_sparse_patch, dense_patch):
                needs_flip = True
        if meta["has_empty_sparse"] != any(count == 0 for count in sparse_counts):
            return False
        if not any(0 < count < full for count, full in zip(sparse_counts, full_sparse_counts)):
            return False
        if not needs_flip:
            return False
        if np.array_equal(inp, expected):
            return False

        other_axis = _opposite_axis(axis_kind)
        alt, _alt_side, _alt_totals = _repair_with_axis(inp, line_rows, line_cols, other_axis, require_unique_sparse=False)
        if alt is not None and np.array_equal(alt, expected):
            return False
        if not np.array_equal(main(inp), expected):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict] | None:
    axis_kind = "horizontal" if bool(rng.integers(2)) else "vertical"
    n_rows = int(rng.choice(np.array([2, 4], dtype=int)))
    n_cols = int(rng.choice(np.array([2, 4], dtype=int)))
    cell_size = int(rng.integers(3, 5))

    inp, line_rows, line_cols = _make_base_grid(n_rows, n_cols, cell_size)
    expected = inp.copy()

    axis_idx = n_rows // 2 if axis_kind == "horizontal" else n_cols // 2
    dense_side = (
        ("top" if bool(rng.integers(2)) else "bottom")
        if axis_kind == "horizontal"
        else ("left" if bool(rng.integers(2)) else "right")
    )
    sparse_side = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[dense_side]

    pair_slots: list[tuple[int, int, int, int]] = []
    if axis_kind == "horizontal":
        for top_r in range(axis_idx):
            bottom_r = n_rows - 1 - top_r
            for cc in range(n_cols):
                if dense_side == "top":
                    pair_slots.append((top_r, cc, bottom_r, cc))
                else:
                    pair_slots.append((bottom_r, cc, top_r, cc))
    else:
        for rr in range(n_rows):
            for left_c in range(axis_idx):
                right_c = n_cols - 1 - left_c
                if dense_side == "left":
                    pair_slots.append((rr, left_c, rr, right_c))
                else:
                    pair_slots.append((rr, right_c, rr, left_c))

    if len(pair_slots) < 2:
        return None

    has_empty_sparse = bool(rng.integers(2))
    empty_pair_idx = int(rng.integers(len(pair_slots))) if has_empty_sparse else -1
    partial_candidates = [idx for idx in range(len(pair_slots)) if idx != empty_pair_idx]
    partial_pair_idx = int(partial_candidates[int(rng.integers(len(partial_candidates)))])

    pairs_meta: list[tuple[int, int, int, int]] = []
    source_asymmetric = False
    changed_sparse = 0

    for idx, (dense_rr, dense_cc, sparse_rr, sparse_cc) in enumerate(pair_slots):
        pixel_count = int(rng.integers(2, 5))
        order = _shape_order(cell_size, pixel_count, rng)
        colors = _sample_colors(pixel_count, rng)
        dense_patch = _render_patch(cell_size, order, colors)
        mirrored_order = _mirror_coords(order, cell_size, axis_kind)
        full_sparse_patch = _render_patch(cell_size, mirrored_order, colors)

        if idx == empty_pair_idx:
            keep = 0
        elif idx == partial_pair_idx:
            keep = int(rng.integers(1, pixel_count))
        elif has_empty_sparse and rng.random() < 0.25:
            keep = 0
        else:
            keep = int(rng.integers(1, pixel_count))

        sparse_patch = _render_patch(cell_size, mirrored_order, colors, keep=keep)
        _place_patch(inp, line_rows, line_cols, dense_rr, dense_cc, dense_patch)
        _place_patch(inp, line_rows, line_cols, sparse_rr, sparse_cc, sparse_patch)
        _place_patch(expected, line_rows, line_cols, dense_rr, dense_cc, dense_patch)
        _place_patch(expected, line_rows, line_cols, sparse_rr, sparse_cc, full_sparse_patch)

        if not np.array_equal(dense_patch, _mirror_patch(dense_patch, axis_kind)):
            source_asymmetric = True
        if not np.array_equal(sparse_patch, full_sparse_patch):
            changed_sparse += 1
        pairs_meta.append((dense_rr, dense_cc, sparse_rr, sparse_cc))

    if not source_asymmetric or changed_sparse == 0:
        return None

    if axis_kind == "horizontal":
        axis_line = line_rows[axis_idx]
        hole_positions = [int(x) for x in rng.choice(np.array(line_cols, dtype=int), size=2, replace=False)]
        for cc in hole_positions:
            inp[axis_line, cc] = BG
            expected[axis_line, cc] = BG
    else:
        axis_line = line_cols[axis_idx]
        hole_positions = [int(x) for x in rng.choice(np.array(line_rows, dtype=int), size=2, replace=False)]
        for rr in hole_positions:
            inp[rr, axis_line] = BG
            expected[rr, axis_line] = BG

    actual, parsed_sparse_side, totals = _repair_with_axis(inp, line_rows, line_cols, axis_kind, require_unique_sparse=True)
    if actual is None or not np.array_equal(actual, expected):
        return None

    meta = {
        "axis_kind": axis_kind,
        "axis_idx": axis_idx,
        "sparse_side": parsed_sparse_side,
        "totals": totals,
        "pairs": pairs_meta,
        "has_empty_sparse": has_empty_sparse,
    }
    if not _semantic_ok(inp, expected, meta):
        return None
    return inp, expected, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        built = _gen_attempt(rng)
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate a valid missing-crossing mirror-repair input")
