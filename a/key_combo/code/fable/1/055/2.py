from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox, crop_bbox, get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import bbox, crop_bbox, get_bg


BLACK = 0
ROTATIONS = {0, 1, 2, 3}
REFLECTIONS = {4, 5, 6, 7}
INVERSE_TRANSFORM = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
GRID_LAYOUTS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
MAX_GENERATION_ATTEMPTS = 400


def _apply_transform(tile: np.ndarray, op: int) -> np.ndarray:
    op = int(op)
    if op == 0:
        return np.array(tile, copy=True)
    if op == 1:
        return np.rot90(tile, 1)
    if op == 2:
        return np.rot90(tile, 2)
    if op == 3:
        return np.rot90(tile, 3)
    if op == 4:
        return np.fliplr(tile)
    if op == 5:
        return np.flipud(tile)
    if op == 6:
        return tile.T.copy()
    if op == 7:
        return np.fliplr(tile).T.copy()
    raise ValueError(f"unknown transform {op}")


def _full_line_positions(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _intervals(size: int, cuts: list[int]) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in [*cuts, size]:
        if cut <= start:
            return None
        spans.append((start, cut - 1))
        start = cut + 1
    return spans


def _parse_divider_grid(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    colors = list(map(int, np.unique(g)))
    candidates = []

    for color in colors:
        if color == BLACK:
            continue
        row_cuts = _full_line_positions(g, color, axis=0)
        col_cuts = _full_line_positions(g, color, axis=1)
        if not row_cuts or not col_cuts:
            continue

        row_spans = _intervals(g.shape[0], row_cuts)
        col_spans = _intervals(g.shape[1], col_cuts)
        if row_spans is None or col_spans is None:
            continue
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue

        row_sizes = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        cell_size = next(iter(row_sizes))
        if cell_size != next(iter(col_sizes)):
            continue

        cells: list[np.ndarray] = []
        ok = True
        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                cell = crop_bbox(g, (r0, c0, r1, c1))
                if np.any(cell == color):
                    ok = False
                    break
                cells.append(cell)
            if not ok:
                break
        if ok:
            candidates.append((color, cell_size, row_spans, col_spans, cells))

    assert len(candidates) == 1, "expected exactly one divider-grid interpretation"
    return candidates[0]


def _visible_consistent(a: np.ndarray, b: np.ndarray) -> bool:
    overlap = (a != BLACK) & (b != BLACK)
    return not np.any(a[overlap] != b[overlap])


def _merge_tiles(tiles: list[np.ndarray]) -> np.ndarray | None:
    merged = np.zeros_like(tiles[0], dtype=int)
    for tile in tiles:
        visible = tile != BLACK
        if np.any((merged != BLACK) & visible & (merged != tile)):
            return None
        merged[visible] = tile[visible]
    return merged


def _reconstruct_solutions(cells: list[np.ndarray], limit: int = 2) -> list[tuple[tuple[int, ...], np.ndarray]]:
    n_cells = len(cells)
    assert 2 <= n_cells <= 8, "cell count must fit within the eight D4 transforms"
    transformed = [[_apply_transform(cell, op) for op in range(8)] for cell in cells]
    pair_ok = [
        [
            [
                [_visible_consistent(transformed[i][op_i], transformed[j][op_j]) for op_j in range(8)]
                for op_i in range(8)
            ]
            for j in range(n_cells)
        ]
        for i in range(n_cells)
    ]

    assignment = [-1] * n_cells
    assignment[0] = 0
    used = {0}
    solutions: list[tuple[tuple[int, ...], np.ndarray]] = []

    def feasible(cell_idx: int, op: int) -> bool:
        if op in used:
            return False
        for other_idx, other_op in enumerate(assignment):
            if other_op == -1:
                continue
            if not pair_ok[cell_idx][other_idx][op][other_op]:
                return False
        return True

    def choose_next() -> tuple[int | None, list[int]]:
        best_idx = None
        best_ops: list[int] | None = None
        for cell_idx in range(1, n_cells):
            if assignment[cell_idx] != -1:
                continue
            options = [op for op in range(1, 8) if feasible(cell_idx, op)]
            if best_ops is None or len(options) < len(best_ops):
                best_idx = cell_idx
                best_ops = options
        return best_idx, [] if best_ops is None else best_ops

    def dfs() -> None:
        if len(solutions) >= limit:
            return
        if all(op != -1 for op in assignment):
            aligned = [transformed[i][assignment[i]] for i in range(n_cells)]
            merged = _merge_tiles(aligned)
            if merged is not None and not np.any(merged == BLACK):
                solutions.append((tuple(assignment), merged))
            return

        cell_idx, options = choose_next()
        if cell_idx is None or not options:
            return
        for op in options:
            assignment[cell_idx] = op
            used.add(op)
            dfs()
            used.remove(op)
            assignment[cell_idx] = -1

    dfs()
    return solutions


def _solution_output(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    assignment: tuple[int, ...],
    merged: np.ndarray,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    idx = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            restored = _apply_transform(merged, INVERSE_TRANSFORM[assignment[idx]])
            out[r0 : r1 + 1, c0 : c1 + 1] = restored
            idx += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the damaged transformed copies, merge their visible evidence, and restore every cell to a full version of the shared pattern.

    Input structure:
    - One non-black divider color forms complete horizontal and vertical bars that split the canvas into equal square cells.
    - Every cell contains the same fully colored square pattern under a different rotation or mirror transform.
    - Black cells inside a pattern cell are damage marks, not part of the true pattern.

    Rule:
    - Infer which D4 transform each damaged cell uses by testing how its visible colored pixels can align with the others.
    - In one common orientation, merge all non-black evidence cell-by-cell; overlapping visible pixels must agree and together cover the whole pattern.
    - Transform that reconstructed full pattern back into each cell's original orientation and keep the divider bars unchanged.

    Key insight:
    - No single cell is complete, so the task is a joint reasoning problem: the hidden orientation of each fragment must be solved before the fragments can complement one another.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _ = get_bg(g)
    divider, _cell_size, row_spans, col_spans, cells = _parse_divider_grid(g)
    assert divider != BLACK

    solutions = _reconstruct_solutions(cells, limit=2)
    assert solutions, "no consistent reconstruction found"

    first_assignment, first_merged = solutions[0]
    out = _solution_output(g, row_spans, col_spans, first_assignment, first_merged)

    for assignment, merged in solutions[1:]:
        candidate = _solution_output(g, row_spans, col_spans, assignment, merged)
        assert np.array_equal(candidate, out), "ambiguous clues lead to different outputs"
    return out


def _pattern_is_asymmetric(pattern: np.ndarray) -> bool:
    for op in range(1, 8):
        if np.array_equal(pattern, _apply_transform(pattern, op)):
            return False
    return True


def _paint_bar(pattern: np.ndarray, color: int) -> None:
    n = pattern.shape[0]
    if random.random() < 0.5:
        r = random.randrange(n)
        c0 = random.randrange(n - 1)
        c1 = random.randrange(c0 + 2, n + 1)
        pattern[r, c0:c1] = color
    else:
        c = random.randrange(n)
        r0 = random.randrange(n - 1)
        r1 = random.randrange(r0 + 2, n + 1)
        pattern[r0:r1, c] = color


def _paint_block(pattern: np.ndarray, color: int) -> None:
    n = pattern.shape[0]
    h = random.randint(2, min(3, n - 1))
    w = random.randint(2, min(3, n - 1))
    r0 = random.randint(0, n - h)
    c0 = random.randint(0, n - w)
    pattern[r0 : r0 + h, c0 : c0 + w] = color


def _paint_diagonal(pattern: np.ndarray, color: int) -> None:
    n = pattern.shape[0]
    if random.random() < 0.5:
        r0 = random.randrange(n - 1)
        c0 = random.randrange(n - 1)
        length = random.randint(2, min(n - r0, n - c0))
        for k in range(length):
            pattern[r0 + k, c0 + k] = color
    else:
        r0 = random.randrange(n - 1)
        c0 = random.randrange(1, n)
        length = random.randint(2, min(n - r0, c0 + 1))
        for k in range(length):
            pattern[r0 + k, c0 - k] = color


def _paint_l_shape(pattern: np.ndarray, color: int) -> None:
    n = pattern.shape[0]
    corner = random.choice(((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)))
    vr = 1 if corner[0] == 0 else -1
    vc = 1 if corner[1] == 0 else -1
    vlen = random.randint(2, n)
    hlen = random.randint(2, n)
    for k in range(vlen):
        pattern[corner[0] + vr * k, corner[1]] = color
    for k in range(hlen):
        pattern[corner[0], corner[1] + vc * k] = color


def _generate_pattern(cell_size: int, divider_color: int) -> np.ndarray | None:
    palette = [c for c in range(1, 10) if c != divider_color]
    random.shuffle(palette)
    quad_colors = palette[:4]
    extras = palette[4:]
    row_cut = random.randint(1, cell_size - 2)
    col_cut = random.randint(1, cell_size - 2)

    pattern = np.empty((cell_size, cell_size), dtype=int)
    pattern[:row_cut, :col_cut] = quad_colors[0]
    pattern[:row_cut, col_cut:] = quad_colors[1]
    pattern[row_cut:, :col_cut] = quad_colors[2]
    pattern[row_cut:, col_cut:] = quad_colors[3]

    painters = [_paint_bar, _paint_block, _paint_diagonal, _paint_l_shape]
    random.shuffle(painters)
    for color, painter in zip(extras[:3], painters[:3]):
        painter(pattern, color)

    if len(np.unique(pattern)) < 4:
        return None
    if not _pattern_is_asymmetric(pattern):
        return None
    return pattern


def _erase_patch(mask: np.ndarray) -> None:
    n = mask.shape[0]
    kind = random.choice(("rect", "row_band", "col_band", "corner", "sprinkle"))
    if kind == "rect":
        h = random.randint(1, max(1, n - 2))
        w = random.randint(1, max(1, n - 2))
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        mask[r0 : r0 + h, c0 : c0 + w] = False
        return
    if kind == "row_band":
        band_h = random.randint(1, max(1, n // 2))
        r0 = random.randint(0, n - band_h)
        c0 = random.randint(0, n - 2)
        c1 = random.randint(c0 + 2, n)
        mask[r0 : r0 + band_h, c0:c1] = False
        return
    if kind == "col_band":
        band_w = random.randint(1, max(1, n // 2))
        c0 = random.randint(0, n - band_w)
        r0 = random.randint(0, n - 2)
        r1 = random.randint(r0 + 2, n)
        mask[r0:r1, c0 : c0 + band_w] = False
        return
    if kind == "corner":
        h = random.randint(1, max(1, n - 2))
        w = random.randint(1, max(1, n - 2))
        top = 0 if random.random() < 0.5 else n - h
        left = 0 if random.random() < 0.5 else n - w
        mask[top : top + h, left : left + w] = False
        return

    coords = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(coords)
    for r, c in coords[: random.randint(1, max(1, n // 2))]:
        mask[r, c] = False


def _generate_visibility_masks(cell_size: int, n_cells: int) -> np.ndarray | None:
    min_visible = max(6, (cell_size * cell_size * 9) // 25)
    max_visible = (cell_size * cell_size * 4) // 5
    masks = np.ones((n_cells, cell_size, cell_size), dtype=bool)

    for i in range(n_cells):
        for _ in range(random.randint(2, 4)):
            _erase_patch(masks[i])
        visible = int(masks[i].sum())
        if visible < min_visible or visible > max_visible:
            return None
        if visible == cell_size * cell_size:
            return None

    coverage = np.any(masks, axis=0)
    if not np.any(~coverage):
        pass
    else:
        for r, c in np.argwhere(~coverage):
            i = random.randrange(n_cells)
            masks[i, int(r), int(c)] = True

    coords = [(r, c) for r in range(cell_size) for c in range(cell_size)]
    random.shuffle(coords)
    witness_coords = coords[:n_cells]
    if len(witness_coords) != n_cells:
        return None
    for i, (r, c) in enumerate(witness_coords):
        masks[:, r, c] = False
        masks[i, r, c] = True

    coverage = np.any(masks, axis=0)
    if np.any(~coverage):
        return None

    for i in range(n_cells):
        visible = int(masks[i].sum())
        if visible < min_visible or visible >= cell_size * cell_size:
            return None
        if visible > max_visible + 2:
            return None

    flat_masks = {masks[i].tobytes() for i in range(n_cells)}
    if len(flat_masks) != n_cells:
        return None

    contribution_counts = masks.sum(axis=0)
    if np.any(contribution_counts == 0):
        return None
    if any(not np.any(masks[i] & (contribution_counts == 1)) for i in range(n_cells)):
        return None
    return masks


def _build_sample(
    pattern: np.ndarray,
    divider_color: int,
    layout: tuple[int, int],
    orientations: list[int],
    canonical_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = layout
    cell_size = pattern.shape[0]
    height = rows * cell_size + (rows - 1)
    width = cols * cell_size + (cols - 1)

    inp = np.full((height, width), divider_color, dtype=int)
    out = np.full((height, width), divider_color, dtype=int)

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (cell_size + 1)
            c0 = cc * (cell_size + 1)
            full = _apply_transform(pattern, orientations[idx])
            vis = _apply_transform(canonical_masks[idx].astype(int), orientations[idx]).astype(bool)
            damaged = full.copy()
            damaged[~vis] = BLACK
            inp[r0 : r0 + cell_size, c0 : c0 + cell_size] = damaged
            out[r0 : r0 + cell_size, c0 : c0 + cell_size] = full
            idx += 1
    return inp, out


def _raw_merge_shortcut_fails(cells: list[np.ndarray]) -> bool:
    merged = _merge_tiles(cells)
    if merged is None:
        return True
    return np.any(merged == BLACK)


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(GRID_LAYOUTS)
    rows, cols = layout
    n_cells = rows * cols
    max_cell = min((30 - (rows - 1)) // rows, (30 - (cols - 1)) // cols, 7)
    if max_cell < 4:
        return None
    cell_size = random.randint(4, max_cell)

    divider_color = random.randint(1, 9)
    pattern = _generate_pattern(cell_size, divider_color)
    if pattern is None:
        return None

    masks = _generate_visibility_masks(cell_size, n_cells)
    if masks is None:
        return None

    if n_cells == 8:
        orientations = list(range(8))
        random.shuffle(orientations)
    else:
        orientations = random.sample(range(8), n_cells)
    if not any(op in ROTATIONS for op in orientations):
        return None
    if not any(op in REFLECTIONS for op in orientations):
        return None
    if len(set(orientations)) != n_cells:
        return None

    inp, expected = _build_sample(pattern, divider_color, layout, orientations, masks)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    divider, parsed_size, row_spans, col_spans, cells = _parse_divider_grid(inp)
    if divider != divider_color or parsed_size != cell_size:
        return None
    if len(cells) != n_cells or len(row_spans) != rows or len(col_spans) != cols:
        return None

    if not all(np.any(cell == BLACK) for cell in cells):
        return None
    if not _raw_merge_shortcut_fails(cells):
        return None

    solutions = _reconstruct_solutions(cells, limit=3)
    if len(solutions) != 1:
        return None

    assignment, merged = solutions[0]
    actual = _solution_output(inp, row_spans, col_spans, assignment, merged)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None

    canonical_views = []
    for op, cell in zip(assignment, cells):
        canonical_views.append(_apply_transform(cell, op))
    merged_check = _merge_tiles(canonical_views)
    if merged_check is None or np.any(merged_check == BLACK):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-consensus restoration puzzle")
