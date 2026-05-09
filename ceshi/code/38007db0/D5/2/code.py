# Variant of 38007db0
# Dimension: D5 — 网格输出形状 (grid_shape)
# Change: tight crop — 输出为紧凑的单列格子（宽度=cell_size+2）... → 自适应裁剪到精灵外接矩形

from __future__ import annotations

# Design analysis for the D5 variant
# ==================================
# reference_task_review:
#   requested: a251c730
#   solver_read: /home/lixinyu/snap/new_code/solvers/arc2/a251c730.py
#   relevant_mechanism:
#     - the reference task keeps its object-selection / placement logic unchanged
#       and changes only the final presented region by cropping to the meaningful
#       rectangle
#     - that is the baseline here too: keep the original per-row odd-one-out
#       detection, but replace the final "single-column lattice" presentation
#       with a tighter crop around each extracted sprite
#
# variant_spec:
#   input_layout:
#     - unchanged from the base task
#     - a lattice of equal square cells separated by 1px gridlines of color L
#     - each cell contains bg padding and a sprite in color sp
#     - within each row, exactly one cell differs from the row's repeated base cell
#   output_layout:
#     - no lattice lines
#     - no 1px cell padding
#     - for each row's unique cell, crop the non-bg sprite to its minimal bbox
#     - stack these cropped sprites vertically in row order
#     - align all crops flush-left in a bg canvas of width = max cropped width
#     - total height = sum of cropped heights; no blank separator rows
#
# behavior_trace:
#   concrete_example:
#     - suppose a chosen unique cell is:
#         bg bg bg bg bg bg
#         bg bg sp bg bg bg
#         bg sp sp sp bg bg
#         bg bg sp bg bg bg
#         bg bg bg bg bg bg
#         bg bg bg bg bg bg
#       the base solver would copy the full 6x6 cell into the output lattice
#       including the outer bg ring
#     - this variant first crops that cell to the sprite bbox:
#         bg sp bg
#         sp sp sp
#         bg sp bg
#       then places this 3x3 crop into the stacked output
#   difference:
#     - original: output height is still the full lattice height and width is
#       fixed to cell_size + 2 because each row keeps a bordered cell slot
#     - variant: each row contributes only its sprite bbox, so both width and
#       per-row height are governed by sprite geometry rather than by cell size
#
# complexity_calibration:
#   - the core reasoning stays "find the odd cell in each row", so the task does
#     not gain extra symbolic complexity unrelated to D5
#   - to make the crop observable, the generator uses base sprites with removable
#     edge arms; unique variants delete different arms, producing visibly
#     different cropped widths/heights across rows
#   - the number of crop shapes is controlled (four templates per inner size),
#     which keeps the rule learnable from 2-4 examples and avoids combinatorial
#     ambiguity
#
# implicit_assumption_review:
#   - uniform-row / uniform-col detection assumes only lattice lines are fully
#     uniform across the whole grid; this still holds because input generation is
#     unchanged and interior rows/cols are broken by lattice columns/rows
#   - lattice color selection assumes the most common value among uniform rows and
#     cols is the gridline color; unchanged input preserves that
#   - bg inference assumes the most common non-lattice color is the cell
#     background; still valid because bg fills most cell area
#   - per-row uniqueness detection assumes exact cell copies for the majority and
#     exactly one differing cell in each row; generator still guarantees this
#   - the base solver's final assumption "output slots have fixed cell size" no
#     longer holds; this is the single changed assumption, so only the assembly
#     stage is replaced by crop-and-stack logic
#
# placement_legality:
#   - each input cell is still fully contained inside the lattice and non-
#     overlapping by construction
#   - output stacking is legal because crops are placed into disjoint vertical
#     bands whose heights equal the crop heights
#   - left alignment inside a width=max(crop widths) canvas avoids overlap while
#     remaining deterministic
#
# visual_observability:
#   - the variant is easy to see: outputs lose lattice lines entirely and become
#     much shorter/narrower than the original single-column lattice
#   - crop sizes differ between rows because the odd variants remove different
#     outer arms; the output therefore shows unequal row-band heights/widths,
#     making "adaptive crop" visible rather than implicit
#
# cleanliness_and_color_inference:
#   - no new color mapping is introduced; only L, bg, sp remain
#   - the output is visually cleaner than the base task because distracting
#     borders and empty padding are removed
#
# main_assumptions_after_change:
#   - every chosen unique cell must contain at least one sp pixel so a bbox exists
#   - stacking assumes row order is preserved top-to-bottom; generator does not
#     permute row semantics
#   - flush-left alignment is the deterministic convention for a variable-width
#     one-column stack; generator builds expected outputs with the same rule

from collections import Counter

import numpy as np

from arc_utils import bbox, paste


def _crop_sprite(cell: np.ndarray, bg: int) -> np.ndarray:
    """Crop a cell down to the bbox of its non-bg sprite pixels."""
    box = bbox(cell != int(bg))
    assert box is not None
    r0, c0, r1, c1 = map(int, box)
    return cell[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _stack_crops(crops: list[np.ndarray], bg: int) -> np.ndarray:
    """Stack cropped sprites vertically, flush-left, padding only to max width."""
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
    Extract the unique cell from each lattice row, crop each selected sprite to
    its non-background bounding box, and stack those crops vertically.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # Gridlines are the only fully uniform rows/cols spanning the whole grid.
    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted([r for r in row_lines if int(g[r, 0]) == int(L)])
    col_lines = sorted([c for c in col_lines if int(g[0, c]) == int(L)])

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    crops: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs)
        uniq_idx = next((j for j, sig in enumerate(sigs) if int(counts[sig]) == 1), None)
        assert uniq_idx is not None
        crops.append(_crop_sprite(cells[int(uniq_idx)], bg))

    return _stack_crops(crops, bg)


# --------------- generator ---------------

def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = int(inner.shape[0] + 2)
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = int(cells[0][0].shape[0])
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[int(r * (cell_size + 1)), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, int(c * (cell_size + 1))] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + int(r) * (cell_size + 1), 1 + int(c) * (cell_size + 1))
    return grid


def _mask_to_inner(mask: np.ndarray, sp: int, bg: int) -> np.ndarray:
    inner = np.full(mask.shape, int(bg), dtype=int)
    inner[mask.astype(bool)] = int(sp)
    return inner


def _pattern_library(inner_size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Return one base mask plus four odd-variant masks.

    The base mask has removable outer arms so variants produce visibly different
    cropped bboxes after deleting selected arms.
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
            np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),  # remove top -> 2x3 crop
            np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),  # remove left -> 3x2 crop
            np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=int),  # remove top+left -> 2x2 crop
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),  # remove bottom -> 2x3 crop
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
        np.array([[0, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=int),
        np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=int),
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=int),
        np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 0]], dtype=int),
    ]
    return base.astype(bool), [v.astype(bool) for v in variants]


def _transform(mask: np.ndarray, rot: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, int(rot))
    if bool(flip_lr):
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
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
    unique_cells = [_make_cell(inner, bg) for inner in unique_inners]

    out = _stack_crops([_crop_sprite(cell, bg) for cell in unique_cells], bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[int(odd_col)] = unique_cells[int(row)].copy()
        in_cells.append(row_cells)

    inp = _build_lattice(in_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
