# Combo variant of 38007db0 (index 21)
# combo_spec: D5/2+D11/2
# Name: 无格线裁剪输出

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox, paste


# Design analysis for the combo variant
# =====================================
# fusion_summary:
#   - Keep the original semantic core: in each logical row of tiles, one tile is
#     the odd one out and must be extracted.
#   - Apply D11/2 to the INPUT side: remove the explicit lattice color entirely.
#     Tiles are separated only by 1px background gaps, so the solver must infer
#     tile boundaries from repeated non-background content bands.
#   - Apply D5/2 to the OUTPUT side: after selecting the odd tile in each row,
#     discard the cell border/padding and crop to the sprite's non-background
#     bbox, then stack those crops vertically with no separator rows.
#
# a) Concrete behavior trace:
#   - Original task:
#       input  = colored lattice lines + square cells with 1px bg padding.
#       output = one-column lattice of the selected full cells.
#   - Fused task:
#       input  = same square cells, but the lattice color is gone; only bg gaps
#                remain between cells and at the outside border.
#       output = no lattice, no cell border, only the bbox-cropped odd sprites.
#   - Small row example (bg='.', sp='#'):
#       base cell:
#         .....
#         ..#..
#         .###.
#         ..#..
#         .....
#       odd cell:
#         .....
#         .....
#         .###.
#         ..#..
#         .....
#       original output would copy the full 5x5 odd cell into a bordered slot.
#       fused output first crops the odd cell to:
#         ###
#         .#.
#       and then stacks that crop directly above/below the crops from other rows.
#   - Difference:
#       D11 changes how rows/cols are segmented.
#       D5 changes what portion of the selected cell is emitted.
#       They interact: the solver must first recover logical cells without using
#       lattice color, then crop the recovered odd cells.
#
# b) Complexity calibration:
#   - If the no-gridline input were paired with arbitrary random sprites, missing
#     sprite rows/cols could create fake separators and make cell inference
#     ambiguous. So the generator uses base sprites that occupy every inner row
#     and every inner column.
#   - If the crop variant used arbitrary odd cells, many outputs would still have
#     the full inner bbox and the crop rule would be visually weak. So the
#     generator uses a small library of "arm-removal" odd variants that always
#     shrink at least one bbox dimension.
#   - The odd variants come from four reusable templates per cell size, keeping
#     the rule learnable from a few examples instead of exploding the hypothesis
#     space.
#
# c) Implicit assumption review against the original main():
#   - "uniform rows/cols reveal lattice lines" breaks completely, because many
#     rows/cols are now uniform bg and there is no dedicated lattice color.
#   - "most common uniform color is lattice color" also breaks for the same
#     reason.
#   - "output keeps full cell slots with border lines" breaks because D5 removes
#     both the lattice and the padding from the output.
#   - "most common non-lattice color is bg" becomes "most common color overall is
#     bg"; this remains valid because bg fills the canvas, gaps, and cell border.
#   - "row-wise odd cell is detected by exact cell equality" still holds and is
#     preserved.
#   - New combined assumption:
#       every logical cell column must contain at least one base cell, otherwise a
#       unique variant that removes an inner column could erase the global content
#       band for that physical x-position and break D11 segmentation. The
#       generator enforces this when choosing odd-column positions.
#
# d) Placement legality:
#   - Cells remain square and are placed on a bg canvas with a fixed 1px outer
#     gap and 1px inter-cell gap.
#   - Paste offsets use pitch = cell_size + gap, so cells cannot overlap and are
#     always fully in-bounds.
#   - Output crops are stacked in disjoint vertical bands whose heights are the
#     crop heights, so output placement is also overlap-free.
#
# e) Visual observability:
#   - Inputs visibly differ from the source task because the third lattice color
#     disappears; the grid must be inferred from spacing.
#   - Outputs visibly differ because they are much tighter than a column of full
#     cells and row bands can have different heights/widths.
#   - The crop is not a hidden numeric convention: the odd variants delete outer
#     arms, so the bbox shrink is directly visible in the bitmap.
#
# f) Visual cleanliness:
#   - The scene uses only bg and sprite colors; there are no extra overlays.
#   - The 1px gutters keep neighboring cells visually separated even without
#     explicit gridlines.
#   - Output stacking is flush-left on a bg canvas of max crop width, which keeps
#     alignment deterministic without clutter.
#
# g) Change inferability:
#   - There is no color remapping. The only human-inferable changes are geometric:
#       infer the hidden grid from regular spacing, find the odd tile per row, and
#       crop each extracted sprite to its tight bbox.
#   - Because the odd variants repeat the same kind of "remove one outer arm"
#     operation across examples, a human can infer the crop rule from 2-4 pairs.
#
# Main solver assumptions after the change, and generator guarantees:
#   - bg is globally most frequent: guaranteed by bg canvas + gaps + padding.
#   - Base sprite touches every inner row/col: guaranteed by the pattern library.
#   - Each logical cell column has at least one base cell: guaranteed when odd
#     columns are sampled.
#   - Every chosen odd cell contains at least one sprite pixel: guaranteed by the
#     pattern library, so bbox() always exists.


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    return runs


def _infer_spaced_cells(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    """
    Infer logical cell slices from repeated non-bg bands.

    This replaces the original lattice-color parsing. It assumes:
    - bg is the most common color,
    - each logical cell contributes one consecutive non-bg run of inner rows/cols,
    - the full cell is exactly 1px larger on every side than that inner run.
    """
    h, w = map(int, g.shape)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    content_rows = [r for r in range(h) if np.any(g[r, :] != int(bg))]
    content_cols = [c for c in range(w) if np.any(g[:, c] != int(bg))]
    assert content_rows and content_cols

    row_runs = _group_runs(content_rows)
    col_runs = _group_runs(content_cols)
    assert row_runs and col_runs

    inner_heights = {int(b - a + 1) for a, b in row_runs}
    inner_widths = {int(b - a + 1) for a, b in col_runs}
    assert len(inner_heights) == 1 and len(inner_widths) == 1

    r_slices: list[tuple[int, int]] = []
    for a, b in row_runs:
        assert a > 0 and b < h - 1
        assert np.all(g[int(a - 1), :] == int(bg))
        assert np.all(g[int(b + 1), :] == int(bg))
        r_slices.append((int(a - 1), int(b + 2)))

    c_slices: list[tuple[int, int]] = []
    for a, b in col_runs:
        assert a > 0 and b < w - 1
        assert np.all(g[:, int(a - 1)] == int(bg))
        assert np.all(g[:, int(b + 1)] == int(bg))
        c_slices.append((int(a - 1), int(b + 2)))

    cell_hs = {int(rb - ra) for ra, rb in r_slices}
    cell_ws = {int(cb - ca) for ca, cb in c_slices}
    assert len(cell_hs) == 1 and len(cell_ws) == 1
    cell_size = int(next(iter(cell_hs)))
    assert cell_size == int(next(iter(cell_ws)))

    return bg, r_slices, c_slices, cell_size


def _crop_sprite(cell: np.ndarray, bg: int) -> np.ndarray:
    mask = cell != int(bg)
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = map(int, box)
    return cell[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _stack_crops(crops: list[np.ndarray], bg: int) -> np.ndarray:
    assert crops
    out_h = int(sum(int(c.shape[0]) for c in crops))
    out_w = int(max(int(c.shape[1]) for c in crops))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    top = 0
    for crop in crops:
        h, w = map(int, crop.shape)
        out[int(top) : int(top) + int(h), 0:int(w)] = crop
        top += int(h)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Infer equal square cells from bg-separated spacing, without explicit
       gridlines.
    2. In each logical row, find the cell whose full pattern is unique.
    3. Crop each selected cell to the bbox of its non-bg sprite.
    4. Stack those crops vertically, flush-left, on a bg canvas.
    """
    g = np.array(grid, dtype=int)

    bg, r_slices, c_slices, _cell_size = _infer_spaced_cells(g)

    crops: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs)
        uniq_idx = next((j for j, sig in enumerate(sigs) if int(counts[sig]) == 1), None)
        assert uniq_idx is not None
        crops.append(_crop_sprite(cells[int(uniq_idx)], bg))

    return _stack_crops(crops, bg)


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = int(inner.shape[0] + 2)
    cell = np.full((size, size), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_spaced(cells: list[list[np.ndarray]], bg: int, gap: int = 1) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = int(cells[0][0].shape[0])
    h = int(rows * cell_size + (rows + 1) * gap)
    w = int(cols * cell_size + (cols + 1) * gap)
    grid = np.full((h, w), int(bg), dtype=int)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, gap + int(r) * (cell_size + gap), gap + int(c) * (cell_size + gap))
    return grid


def _mask_to_inner(mask: np.ndarray, sp: int, bg: int) -> np.ndarray:
    inner = np.full(mask.shape, int(bg), dtype=int)
    inner[mask.astype(bool)] = int(sp)
    return inner


def _pattern_library(inner_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    One base mask plus four odd masks.

    Design goals:
    - base uses every inner row and every inner column, supporting D11 slicing;
    - each odd mask removes an outer arm, making D5 cropping visually explicit;
    - every odd mask remains non-empty.
    """
    if int(inner_size) == 3:
        base = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=int,
        )
        variants = [
            np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),  # 2x3 crop
            np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),  # 3x2 crop
            np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=int),  # 2x2 crop
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),  # 2x3 crop
        ]
        return base.astype(bool), [v.astype(bool) for v in variants]

    assert int(inner_size) == 4
    base = np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
        ],
        dtype=int,
    )
    variants = [
        np.array([[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=int),  # 3x4
        np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=int),  # 4x3
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=int),  # 3x3
        np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 0]], dtype=int),  # 3x4
    ]
    return base.astype(bool), [v.astype(bool) for v in variants]


def _transform(mask: np.ndarray, rot: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, int(rot))
    if bool(flip_lr):
        out = np.fliplr(out)
    return out.copy()


def _choose_odd_columns(rng: np.random.Generator, n_rows: int, n_cols: int) -> list[int]:
    """
    Choose odd-column positions while keeping at least one base cell in every
    logical column. This protects the D11 segmentation assumption that each
    physical inner-column band is present somewhere in the grid.
    """
    for _ in range(200):
        odd_cols = list(map(int, rng.integers(0, n_cols, size=n_rows)))
        counts = Counter(odd_cols)
        if all(int(counts[c]) < n_rows for c in range(n_cols)):
            return odd_cols
    raise AssertionError("failed to sample non-degenerate odd columns")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg, sp = map(int, rng.choice(10, size=2, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = int(cell_size - 2)
    base_mask, variant_masks = _pattern_library(inner_size)

    rot = int(rng.integers(0, 4))
    flip_lr = bool(rng.integers(0, 2))
    base_mask = _transform(base_mask, rot, flip_lr)
    variant_masks = [_transform(mask, rot, flip_lr) for mask in variant_masks]

    chosen_variant_ids = list(map(int, rng.choice(len(variant_masks), size=n_rows, replace=False)))
    base_inner = _mask_to_inner(base_mask, sp, bg)
    unique_inners = [_mask_to_inner(variant_masks[idx], sp, bg) for idx in chosen_variant_ids]

    base_cell = _make_cell(base_inner, bg)
    unique_cells = [_make_cell(inner, bg) for inner in unique_inners]

    out = _stack_crops([_crop_sprite(cell, bg) for cell in unique_cells], bg)

    odd_cols = _choose_odd_columns(rng, n_rows, n_cols)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[int(odd_cols[row])] = unique_cells[int(row)].copy()
        in_cells.append(row_cells)

    inp = _build_spaced(in_cells, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
