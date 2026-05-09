# Variant of 38007db0
# Dimension: D9 — 颜色系统 (meta)
# Change: 恰好3种颜色：格子线色L、背景色bg、精灵色sp... → 多精灵颜色

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

# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior:
#    The lattice layout is unchanged: 1-pixel gridlines, square cells, and one odd cell per lattice row.
#    The change happens inside each cell. Instead of a binary sprite (bg vs sp), the inner sprite uses
#    two foreground colors sp1/sp2 on a shared support mask. The odd cell in a row keeps the same support
#    pixels as the repeated base cell, but recolors a small connected subset from sp1<->sp2.
#    Example trace:
#      original base inner:
#        2 2 0
#        3 2 3
#        0 3 3
#      original oddball logic changed support by flipping 0/2 pixels.
#      new oddball logic keeps the non-bg positions fixed and instead swaps colors on a subset, e.g.
#        2 3 0
#        3 2 3
#        0 3 2
#      The solver still extracts the unique whole cell per row, but it must not assume "most common
#      non-gridline color == bg", because there are now two sprite colors.
# b) Complexity calibration:
#    The variant uses exactly 4 visible colors: lattice L, background bg, sprite colors sp1 and sp2.
#    Unique cells are recolor-only variants, but the recolored subset is 2-4 connected pixels so the
#    difference is easy to see and repeated base cells in the same row provide direct visual comparison.
# c) Assumption audit:
#    - Gridline detection from uniform rows/cols still holds.
#    - Exact bytewise cell comparison still holds because majority cells are exact copies and the odd cell
#      is an exact but different recoloring.
#    - Old bg inference from "most common non-L color" is no longer robust, so bg is inferred from cell
#      corners; those corners are guaranteed bg by construction.
# d) Placement legality:
#    Cells are still assembled by a lattice builder, so there is no overlap and every cell lies strictly
#    inside the surrounding gridlines. The 1-pixel bg padding around the inner sprite is preserved.
# e/f/g) Visual observability and cleanliness:
#    Every sample shows repeated two-color sprites plus one recolored counterpart in each row. Because the
#    support mask is unchanged, humans can spot that only colors moved, not the shape footprint itself.
#    The color count stays small and the recolor rule is local, repeated, and visible.
# h/i) New solver assumptions and generator guarantees:
#    - Each cell corner is bg. `_make_cell` enforces this.
#    - Every row contains exactly one unique recolored cell and at least two repeated base cells.
#    - Unique cells differ from the base only by swapping sp1/sp2 on support pixels; support is unchanged.
#      The generator constructs and validates this explicitly.


def _grid_structure(g: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each lattice row into a single-column lattice.

    D9 variant:
    - Cell sprites may use two foreground colors instead of one.
    - The odd cell may differ only by recoloring sprite pixels, not by turning pixels on/off.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    L, row_lines, _col_lines, r_slices, c_slices = _grid_structure(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])
    bg = _infer_bg(g, r_slices, c_slices)

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(cells[int(uniq_idx)].copy())

    out = np.full((H, int(cell_w + 2)), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_cells[int(i)]

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
    coords = {(idx // width, idx % width) for idx in indices}
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


def _random_base_inner(rng: np.random.Generator, inner_size: int, bg: int, sp1: int, sp2: int) -> np.ndarray | None:
    inner_area = inner_size * inner_size
    min_fill = max(5, math.ceil(0.45 * inner_area))
    max_fill = min(inner_area - 1, math.floor(0.85 * inner_area))
    if min_fill > max_fill:
        return None
    fill = int(rng.integers(min_fill, max_fill + 1))

    inner = np.full((inner_size, inner_size), int(bg), dtype=int)
    support = rng.choice(inner_area, size=fill, replace=False)

    min_sp1 = max(2, math.ceil(0.3 * fill))
    max_sp1 = min(fill - 2, math.floor(0.7 * fill))
    if min_sp1 > max_sp1:
        return None
    sp1_count = int(rng.integers(min_sp1, max_sp1 + 1))
    sp1_idx = set(map(int, rng.choice(support, size=sp1_count, replace=False)))
    for idx in map(int, support):
        inner.flat[idx] = int(sp1 if idx in sp1_idx else sp2)

    colors = {int(v) for v in inner.flat if int(v) != int(bg)}
    if colors != {int(sp1), int(sp2)}:
        return None
    if len(_connected_recolor_candidates(inner, bg)) < 4:
        return None
    return inner


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


def _swap_colors(inner: np.ndarray, indices: tuple[int, ...], sp1: int, sp2: int) -> np.ndarray:
    out = inner.copy()
    for idx in indices:
        cur = int(out.flat[int(idx)])
        if cur == int(sp1):
            out.flat[int(idx)] = int(sp2)
        elif cur == int(sp2):
            out.flat[int(idx)] = int(sp1)
        else:
            raise AssertionError("recolor set must stay on sprite support")
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp1, sp2 = map(int, rng.choice(10, size=4, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    base_inner = _random_base_inner(rng, inner_size, bg, sp1, sp2)
    if base_inner is None:
        return None

    candidates = _connected_recolor_candidates(base_inner, bg)
    rng.shuffle(candidates)

    unique_inners: list[np.ndarray] = []
    used_sigs = {base_inner.tobytes()}
    for subset in candidates:
        variant = _swap_colors(base_inner, subset, sp1, sp2)
        sig = variant.tobytes()
        if sig in used_sigs:
            continue
        unique_inners.append(variant)
        used_sigs.add(sig)
        if len(unique_inners) == n_rows:
            break
    if len(unique_inners) != n_rows:
        return None

    base_cell = _make_cell(base_inner, bg)
    unique_cells = [_make_cell(inner, bg) for inner in unique_inners]

    out = _build_lattice([[cell] for cell in unique_cells], lattice, bg)

    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
