# Combo variant of 38007db0 (index 10)
# combo_spec: D4/1+D9/0
# Name: 多色精灵取反输出

from __future__ import annotations

from collections import Counter
import itertools
import math
import sys

import numpy as np

ROOT = "/home/lixinyu/snap/new"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    The lattice geometry is unchanged from 38007db0: 1px gridlines, square cells, 1px bg ring,
#    and exactly one odd cell per lattice row. The D9 part changes the cell contents: the inner
#    sprite uses two foreground colors rather than one, and the odd cell differs only by recoloring
#    some support pixels from one sprite color to the other. The D4 part is then fused at the output
#    stage, but plain "bg <-> sprite" is under-specified once there are two sprite colors. This
#    combo resolves that by using a role-based three-color cycle on the extracted odd cell:
#      bg -> major sprite color
#      major sprite color -> minor sprite color
#      minor sprite color -> bg
#    The major/minor roles are inferred from the input by global frequency over all non-bg,
#    non-lattice pixels. Example with roles bg=6, major=2, minor=8:
#      chosen odd cell
#        6 6 6 6 6
#        6 2 8 2 6
#        6 8 2 8 6
#        6 6 6 6 6
#        6 6 6 6 6
#      original task output would copy that cell as-is.
#      fused variant output becomes
#        2 2 2 2 2
#        2 8 6 8 2
#        2 6 8 6 2
#        2 2 2 2 2
#        2 2 2 2 2
#    So the row-wise odd-one-out detection is unchanged, but the extracted cell is remapped through
#    a visible three-role cycle instead of direct copy.
# b) Complexity calibration:
#    Keep exactly four colors overall: lattice L, background bg, major sprite sp_major, minor sprite
#    sp_minor. This is enough to make the fusion visible without creating an unbounded color mapping
#    problem. The repeated cells in each row still provide a clear majority reference, and every odd
#    cell uses both sprite colors so the output cannot collapse to a single-color D4 inversion.
# c) Assumption audit against the original solver:
#    - Uniform full rows/cols identify lattice lines: still valid because layout is unchanged.
#    - "Most common non-lattice color is bg": no longer reliable with two sprite colors, so bg is
#      inferred from cell corners, which are guaranteed to be bg by construction.
#    - "There is exactly one sprite color": broken by D9, replaced with exactly two inferred sprite
#      colors and a strict major/minor role ordering by global frequency.
#    - Bytewise per-cell comparison for oddball detection: still valid because majority cells are
#      exact copies and each odd cell is an exact, distinct recoloring.
#    - A binary inversion like np.where(cell == bg, sp, bg) is invalid now; the combo instead uses a
#      three-way mapping whose roles are inferred from the input rather than numeric color ids.
# d) Placement legality:
#    Cells are built first and then pasted into fixed lattice slots. Cell size, row count, and column
#    count remain within the original spec ranges, so there is no overlap or out-of-bounds risk.
#    The output is still a single-column lattice with the same row slices as the input.
# e) Visual observability:
#    A human can compare each odd input cell with the output cell and see three repeated effects:
#    the bg ring becomes the globally dominant sprite color, that dominant sprite color turns into
#    the other sprite color, and the minor sprite color turns into bg. Because this happens on every
#    extracted row, the mapping is recoverable from 2-3 examples without hidden numeric ordering.
# f) Visual cleanliness:
#    The generator keeps the same lattice spacing as the seed task and constrains the color system to
#    four colors total. Each inner sprite is dense enough to be legible, while the 1px bg border keeps
#    cell boundaries readable even after inversion.
# g) Color-rule inferability:
#    The only nontrivial mapping is the role-based cycle {bg, major, minor}. The generator enforces
#    both sprite colors to appear in the base and in every odd cell, and it also enforces major/minor
#    counts to be unequal, so the role split is observable from repetition in the input itself.
# h) Quantitative anti-degeneration target:
#    Generated tasks should differ from (1) the original direct-copy output and (2) every pairwise
#    two-color swap on the same extracted cell. This will be checked after implementation.
# i) New solver assumptions introduced here:
#    - Every cell corner is bg.
#    - Exactly two sprite colors appear outside lattice/bg.
#    - Their global counts are unequal, so major/minor is well-defined without using numeric ids.
#    - Every odd cell contains bg plus both sprite colors, making the three-way transform observable.
#    The generator below explicitly constructs inputs satisfying these assumptions.


def _grid_structure(
    g: np.ndarray,
) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
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

    return L, row_lines, col_lines, r_slices, c_slices


def _infer_bg(g: np.ndarray, r_slices: list[tuple[int, int]], c_slices: list[tuple[int, int]]) -> int:
    corner_samples: list[int] = []
    for ra, rb in r_slices:
        for ca, cb in c_slices:
            corner_samples.extend(
                [
                    int(g[ra, ca]),
                    int(g[ra, cb - 1]),
                    int(g[rb - 1, ca]),
                    int(g[rb - 1, cb - 1]),
                ]
            )
    assert corner_samples
    return int(Counter(corner_samples).most_common(1)[0][0])


def _infer_sprite_roles(g: np.ndarray, lattice: int, bg: int) -> tuple[int, int]:
    sprite_counts = Counter(int(x) for x in g.flatten() if int(x) not in (int(lattice), int(bg)))
    assert len(sprite_counts) == 2
    ranked = sprite_counts.most_common()
    assert int(ranked[0][1]) != int(ranked[1][1])
    return int(ranked[0][0]), int(ranked[1][0])


def _extract_unique_cells(
    g: np.ndarray, r_slices: list[tuple[int, int]], c_slices: list[tuple[int, int]]
) -> list[np.ndarray]:
    chosen: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        counts = Counter(sigs)
        uniq_idx = next((j for j, sig in enumerate(sigs) if int(counts[sig]) == 1), None)
        assert uniq_idx is not None
        chosen.append(cells[int(uniq_idx)].copy())
    return chosen


def _cycle_invert_cell(cell: np.ndarray, bg: int, sp_major: int, sp_minor: int) -> np.ndarray:
    vals = {int(v) for v in cell.flatten()}
    assert vals <= {int(bg), int(sp_major), int(sp_minor)}
    out = np.empty_like(cell)
    out[cell == int(bg)] = int(sp_major)
    out[cell == int(sp_major)] = int(sp_minor)
    out[cell == int(sp_minor)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row, then apply a role-based three-color inversion:
    bg -> major sprite, major sprite -> minor sprite, minor sprite -> bg.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    L, row_lines, _col_lines, r_slices, c_slices = _grid_structure(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])
    bg = _infer_bg(g, r_slices, c_slices)
    sp_major, sp_minor = _infer_sprite_roles(g, L, bg)

    chosen_cells = _extract_unique_cells(g, r_slices, c_slices)
    transformed = [_cycle_invert_cell(cell, bg, sp_major, sp_minor) for cell in chosen_cells]

    out = np.full((H, int(cell_w + 2)), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = transformed[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
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


def _support_positions(inner: np.ndarray, bg: int) -> list[int]:
    return [int(i) for i, v in enumerate(inner.flat) if int(v) != int(bg)]


def _is_connected(indices: tuple[int, ...], width: int) -> bool:
    coords = {(int(idx) // width, int(idx) % width) for idx in indices}
    stack = [next(iter(coords))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _connected_recolor_candidates(inner: np.ndarray, bg: int) -> list[tuple[int, ...]]:
    support = _support_positions(inner, bg)
    width = inner.shape[1]
    out: list[tuple[int, ...]] = []
    max_k = min(4, len(support))
    for k in range(2, max_k + 1):
        for combo in itertools.combinations(support, k):
            if _is_connected(combo, width):
                out.append(tuple(sorted(map(int, combo))))
    return out


def _swap_colors(inner: np.ndarray, indices: tuple[int, ...], sp_major: int, sp_minor: int) -> np.ndarray:
    out = inner.copy()
    for idx in indices:
        cur = int(out.flat[int(idx)])
        if cur == int(sp_major):
            out.flat[int(idx)] = int(sp_minor)
        elif cur == int(sp_minor):
            out.flat[int(idx)] = int(sp_major)
        else:
            raise AssertionError("recolor set must stay on sprite support")
    return out


def _random_base_inner(
    rng: np.random.Generator, inner_size: int, bg: int, sp_major: int, sp_minor: int
) -> np.ndarray | None:
    inner_area = inner_size * inner_size
    min_fill = max(5, math.ceil(0.5 * inner_area))
    max_fill = min(inner_area - 1, math.floor(0.85 * inner_area))
    if min_fill > max_fill:
        return None
    fill = int(rng.integers(min_fill, max_fill + 1))

    inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    support = list(map(int, rng.choice(inner_area, size=fill, replace=False)))

    min_major = max(3, math.ceil(0.58 * fill))
    max_major = min(fill - 2, math.floor(0.78 * fill))
    if min_major > max_major:
        return None
    major_count = int(rng.integers(min_major, max_major + 1))
    minor_count = fill - major_count
    if major_count <= minor_count:
        return None

    major_idx = set(map(int, rng.choice(support, size=major_count, replace=False)))
    for idx in support:
        inner.flat[int(idx)] = int(sp_major if idx in major_idx else sp_minor)

    colors = {int(v) for v in inner.flat if int(v) != int(bg)}
    if colors != {int(sp_major), int(sp_minor)}:
        return None
    if len(_connected_recolor_candidates(inner, bg)) < 6:
        return None
    return inner


def _has_both_sprite_colors(inner: np.ndarray, bg: int, sp_major: int, sp_minor: int) -> bool:
    colors = {int(v) for v in inner.flat if int(v) != int(bg)}
    return colors == {int(sp_major), int(sp_minor)}


def _pairwise_swap(cell: np.ndarray, a: int, b: int) -> np.ndarray:
    out = cell.copy()
    mask_a = cell == int(a)
    mask_b = cell == int(b)
    out[mask_a] = int(b)
    out[mask_b] = int(a)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp_major, sp_minor = map(int, rng.choice(10, size=4, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    base_inner = _random_base_inner(rng, inner_size, bg, sp_major, sp_minor)
    if base_inner is None:
        return None

    candidates = _connected_recolor_candidates(base_inner, bg)
    rng.shuffle(candidates)

    unique_inners: list[np.ndarray] = []
    used_sigs = {base_inner.tobytes()}
    for subset in candidates:
        variant = _swap_colors(base_inner, subset, sp_major, sp_minor)
        sig = variant.tobytes()
        if sig in used_sigs:
            continue
        if not _has_both_sprite_colors(variant, bg, sp_major, sp_minor):
            continue
        unique_inners.append(variant)
        used_sigs.add(sig)
        if len(unique_inners) == n_rows:
            break
    if len(unique_inners) != n_rows:
        return None

    base_cell = _make_cell(base_inner, bg)
    unique_cells = [_make_cell(inner, bg) for inner in unique_inners]
    transformed_cells = [_cycle_invert_cell(cell, bg, sp_major, sp_minor) for cell in unique_cells]

    for cell, transformed in zip(unique_cells, transformed_cells):
        if np.array_equal(cell, transformed):
            return None
        if np.array_equal(transformed, _pairwise_swap(cell, bg, sp_major)):
            return None
        if np.array_equal(transformed, _pairwise_swap(cell, bg, sp_minor)):
            return None
        if np.array_equal(transformed, _pairwise_swap(cell, sp_major, sp_minor)):
            return None

    out = _build_lattice([[cell] for cell in transformed_cells], lattice, bg)

    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    inferred_bg = _infer_bg(inp, _grid_structure(inp)[3], _grid_structure(inp)[4])
    inferred_major, inferred_minor = _infer_sprite_roles(inp, lattice, inferred_bg)
    if (inferred_major, inferred_minor) != (int(sp_major), int(sp_minor)):
        return None

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 400 attempts")
