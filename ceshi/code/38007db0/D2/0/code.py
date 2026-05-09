# Variant of 38007db0
# Dimension: D2 — 源对象：行内唯一精灵单元格 (src)
# Change: 每行中模式只出现一次的方形单元格（内含独特精灵变体）... → 异常检测：颜色异常项
#
# Design analysis
# ===============
# Reference-task note:
# - The prompt cites `b99e7126` as the anomaly-detection inspiration, but its solver
#   source is not present in this workspace; only the JSON exists. Per instruction,
#   the missing solver is skipped and the variant is designed from the original
#   38007db0 solver/spec only.
#
# a) Concrete behavior and trace
# - The lattice structure stays the same: 1-pixel gridlines, square cells, 3-4 rows,
#   3-4 columns, and the output is still a single-column lattice containing one cell
#   per input row.
# - The changed part is what makes a source cell "special":
#   original task: one cell in the row had a unique full sprite pattern.
#   variant task: cell shapes in the row are no longer a reliable key; instead,
#   exactly one cell uses an anomalous sprite color B while the other cells use
#   sprite color A.
# - Small trace:
#   original row bytes: [shape0(A), shape0(A), shape1(A)] -> unique pattern is shape1.
#   variant row bytes:  [shape0(A), shape1(B), shape2(A)] -> all three full cells are
#   byte-unique, so pattern counting is useless; the solver must read each cell's
#   sprite color and select the only B-colored one.
#
# b) Complexity calibration
# - If every majority cell were an exact copy and only the odd cell changed color,
#   the legacy pattern-uniqueness solver would still succeed accidentally because the
#   odd-colored cell would also be byte-unique. That would be a degenerate variant.
# - To avoid that, each row contains several different sprite shapes. The anomaly is
#   carried only by color, not by shape multiplicity.
# - Colors are kept to exactly four global roles:
#   lattice L, background bg, majority sprite color A, anomaly sprite color B.
#   This keeps the color rule legible to a human from a few examples.
#
# c) Old-solver assumptions that break here
# - `sigs = [cell.tobytes() ...]` and `Counter(sigs)` assumed "one exact cell copy
#   family + one odd cell". In this variant, many or all cells are byte-unique.
# - `uniq_idx = next(... cnt[s] == 1 ...)` becomes wrong because the first unique cell
#   is usually just the first column, not the anomaly.
# - Grid parsing, lattice color detection, slice extraction, and output assembly do
#   not depend on the old uniqueness property, so they can remain unchanged.
#
# d) Placement legality
# - Cells are still placed inside fixed-size lattice slots with 1-pixel separators.
# - Each cell has a 1-pixel bg padding border, and its inner sprite uses exactly one
#   non-bg color, so the solver can safely recover the cell's sprite color.
# - The generator ensures all row cells are valid cell-sized arrays, no overlaps, and
#   the anomaly cell is one of those already-placed cells.
#
# e) Visual observability
# - A human sees each row as "different shapes, but almost all in one color, with one
#   differently colored cell". The output keeps exactly those odd-colored cells.
# - The anomaly color is global across the whole puzzle, so 2-4 examples are enough to
#   infer that color is the selection criterion rather than shape.
#
# f) Visual cleanliness
# - The lattice framing and bg border are unchanged, keeping rows and cells easy to
#   segment.
# - Sprite masks are generated from small perturbations of a row base mask so the row
#   stays visually coherent instead of becoming random noise.
#
# g) Color-change inferability
# - Only two sprite colors participate in the rule: majority color A and anomaly
#   color B. There is no hidden numeric ordering of colors.
# - The generator intentionally keeps the anomaly away from column 0 while making all
#   row cells byte-unique, so the new color-based solver and the legacy pattern-based
#   solver disagree on the same inputs. This makes the D2 change observable rather
#   than a superficial recoloring of the original task.

from __future__ import annotations

from collections import Counter
import numpy as np
import math

from arc_utils import paste


def _sprite_color(cell: np.ndarray, bg: int) -> int:
    colors = {int(v) for v in cell.flatten() if int(v) != int(bg)}
    assert len(colors) == 1
    return next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the anomalous-color cell from each lattice row into a single-column lattice.

    The lattice parsing and output layout are the same as 38007db0. The only changed
    rule is row selection: cells may all have different shapes, but exactly one cell
    in each row uses the minority sprite color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == int(L))
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == int(L))

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra):int(rb), int(ca):int(cb)] for ca, cb in c_slices]
        sprite_colors = [_sprite_color(cell, bg) for cell in cells]
        color_counts = Counter(sprite_colors)
        odd_color = next((color for color, cnt in color_counts.items() if int(cnt) == 1), None)
        assert odd_color is not None
        odd_idx = next(i for i, color in enumerate(sprite_colors) if int(color) == int(odd_color))
        chosen_cells.append(cells[int(odd_idx)])

    out_w = int(cell_w + 2)
    out = np.full((H, out_w), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra):int(rb), 1:1 + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(mask: np.ndarray, sprite: int, bg: int) -> np.ndarray:
    inner = np.full(mask.shape, int(bg), dtype=int)
    inner[mask] = int(sprite)
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _gen_row_masks(rng: np.random.Generator, inner_size: int, n_cols: int) -> list[np.ndarray] | None:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.8 * inner_area)

    base = np.zeros((inner_size, inner_size), dtype=bool)
    on_idx = rng.choice(inner_area, size=int(rng.integers(min_fill, max_fill + 1)), replace=False)
    base.flat[on_idx] = True

    masks = [base.copy()]
    seen = {base.tobytes()}
    attempts = 0
    while len(masks) < n_cols and attempts < 200:
        attempts += 1
        flip_count = int(rng.integers(1, 4))
        flip_idx = rng.choice(inner_area, size=flip_count, replace=False)
        mask = base.copy()
        flat = mask.reshape(-1)
        flat[flip_idx] = ~flat[flip_idx]
        fill = int(mask.sum())
        if fill == 0 or fill == inner_area:
            continue
        sig = mask.tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        masks.append(mask)

    return masks if len(masks) == n_cols else None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, majority_sp, anomaly_sp = map(int, rng.choice(10, size=4, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))
    inner_size = cell_size - 2

    chosen_cells: list[np.ndarray] = []
    in_cells: list[list[np.ndarray]] = []

    for _ in range(n_rows):
        masks = _gen_row_masks(rng, inner_size, n_cols)
        if masks is None:
            return None

        odd_col = int(rng.integers(1, n_cols))
        row_cells: list[np.ndarray] = []
        for col, mask in enumerate(masks):
            sprite = anomaly_sp if int(col) == int(odd_col) else majority_sp
            row_cells.append(_make_cell(mask, sprite, bg))

        chosen_cells.append(row_cells[int(odd_col)].copy())
        in_cells.append(row_cells)

    out = _build_lattice([[cell] for cell in chosen_cells], lattice, bg)
    inp = _build_lattice(in_cells, lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
