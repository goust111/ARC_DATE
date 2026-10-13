from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_for_fable_8_035_1", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()
Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BG = int(Color.BLACK)
MASK = int(Color.GREY)
PALETTE_POOL = tuple(c for c in Color.NOT_BLACK if int(c) != MASK)
RNG = np.random.default_rng()
GEN_ATTEMPTS = 500


def _depth_map(side: int) -> np.ndarray:
    rr, cc = np.indices((side, side))
    return np.minimum.reduce([rr, cc, side - 1 - rr, side - 1 - cc])


def _extract_objects(grid: np.ndarray) -> tuple[list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    assert components
    components = sorted(components, key=lambda comp: tuple(map(int, comp["bbox"][:2])))

    objects: list[dict[str, object]] = []
    layer_count: int | None = None
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1
        assert side % 2 == 1 and side >= 7
        depth_count = side // 2
        if layer_count is None:
            layer_count = depth_count
        assert depth_count == layer_count

        crop = g[r0 : r1 + 1, c0 : c1 + 1]
        dist = _depth_map(side)
        center = depth_count
        assert int(crop[center, center]) == bg
        assert np.all(crop[dist < depth_count] != bg)
        assert int(np.count_nonzero(crop == bg)) == 1

        row: list[int] = []
        for depth in range(depth_count):
            vals = np.unique(crop[dist == depth])
            assert vals.size == 1
            color = int(vals[0])
            assert color != bg
            row.append(color)

        objects.append({"bbox": (r0, c0, r1, c1), "side": side, "row": row})

    assert layer_count is not None
    return objects, int(layer_count)


def _solve_unique(partial: np.ndarray, palette: list[int]) -> tuple[int, np.ndarray | None]:
    board = np.asarray(partial, dtype=int).copy()
    n = int(board.shape[0])
    palette_set = set(map(int, palette))
    row_missing = []
    col_missing = []
    for r in range(n):
        visible = [int(v) for v in board[r] if int(v) != MASK]
        assert len(set(visible)) == len(visible)
        assert set(visible).issubset(palette_set)
        row_missing.append(set(palette_set) - set(visible))
    for c in range(n):
        visible = [int(v) for v in board[:, c] if int(v) != MASK]
        assert len(set(visible)) == len(visible)
        assert set(visible).issubset(palette_set)
        col_missing.append(set(palette_set) - set(visible))

    solutions: list[np.ndarray] = []

    def search() -> None:
        if len(solutions) >= 2:
            return
        best_cell = None
        best_cands = None
        for r in range(n):
            for c in range(n):
                if int(board[r, c]) != MASK:
                    continue
                cands = row_missing[r] & col_missing[c]
                if not cands:
                    return
                if best_cands is None or len(cands) < len(best_cands):
                    best_cell = (r, c)
                    best_cands = set(cands)
        if best_cell is None:
            solutions.append(board.copy())
            return

        r, c = best_cell
        for color in sorted(best_cands):
            board[r, c] = color
            row_missing[r].remove(color)
            col_missing[c].remove(color)
            search()
            col_missing[c].add(color)
            row_missing[r].add(color)
            board[r, c] = MASK
            if len(solutions) >= 2:
                return

    search()
    return len(solutions), (solutions[0] if len(solutions) == 1 else None)


def _paint_from_table(shape: tuple[int, int], positions: list[tuple[int, int]], table: np.ndarray) -> np.ndarray:
    n = int(table.shape[0])
    side = 2 * n + 1
    dist = _depth_map(side)
    out = np.full(shape, BG, dtype=int)
    for row_idx, (r0, c0) in enumerate(positions):
        view = out[r0 : r0 + side, c0 : c0 + side]
        for depth in range(n):
            view[dist == depth] = int(table[row_idx, depth])
    return out


def _expected_output_from_partial(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    objects, n = _extract_objects(grid)
    partial = np.array([obj["row"] for obj in objects], dtype=int)
    palette = colors_present(partial, exclude={MASK})
    assert len(palette) == n
    assert len(objects) == n
    count, solved = _solve_unique(partial, palette)
    assert count == 1 and solved is not None

    out = np.asarray(grid, dtype=int).copy()
    side = 2 * n + 1
    dist = _depth_map(side)
    for row_idx, obj in enumerate(objects):
        r0, c0, _r1, _c1 = map(int, obj["bbox"])
        view = out[r0 : r0 + side, c0 : c0 + side]
        for depth in range(n):
            if int(partial[row_idx, depth]) == MASK:
                view[dist == depth] = int(solved[row_idx, depth])
    return out, solved


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey concentric ring by uniquely completing the object-by-depth Latin square.

    Input structure:
    - A black background contains several separated multicolor square objects.
    - Each object is a stack of one-cell-thick concentric square rings with the same
      odd side length, leaving a one-cell black center hole.
    - Grey rings are masked unknowns; visible non-grey rings obey row and column
      uniqueness when rows are objects and columns are outer-to-inner depth levels.

    Rule:
    - Detect each multicolor object and read its ring colors from outermost depth to
      innermost depth, treating grey as blank.
    - View the resulting table as `object x depth`; every row and every depth column
      must contain each visible palette color exactly once.
    - Compute the unique Latin-square completion and recolor only the grey rings with
      their solved colors, leaving all other cells unchanged.

    Key insight:
    - The grid's spatial rows and columns are irrelevant; the real constraints live on
      the matrix whose rows are objects and whose columns are concentric layer depths.
    """
    out, _solved = _expected_output_from_partial(np.asarray(grid, dtype=int))
    return out


def _latin_square(n: int, palette: list[int]) -> np.ndarray:
    row_offsets = RNG.permutation(n)
    col_offsets = RNG.permutation(n)
    symbols = np.array(RNG.permutation(palette), dtype=int)
    table = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            table[r, c] = int(symbols[(int(row_offsets[r]) + int(col_offsets[c])) % n])
    return table


def _mask_pattern(full: np.ndarray) -> np.ndarray | None:
    n = int(full.shape[0])
    cells = [(r, c) for r in range(n) for c in range(n)]
    max_masked = min(n * n - 2, 2 * n + 2)
    for _ in range(300):
        masked_count = int(RNG.integers(n + 1, max_masked + 1))
        picks = RNG.choice(len(cells), size=masked_count, replace=False)
        mask = np.zeros((n, n), dtype=bool)
        for idx in np.atleast_1d(picks):
            r, c = cells[int(idx)]
            mask[r, c] = True

        row_sums = mask.sum(axis=1)
        col_sums = mask.sum(axis=0)
        if row_sums.max() < 2 or col_sums.max() < 2:
            continue
        if np.any(row_sums == n) or np.any(col_sums == n):
            continue

        partial = full.copy()
        partial[mask] = MASK
        palette = sorted(int(v) for v in np.unique(full))
        visible_counts = {color: int(np.count_nonzero(partial == color)) for color in palette}
        if min(visible_counts.values()) <= 0:
            continue

        count, solved = _solve_unique(partial, palette)
        if count != 1 or solved is None or not np.array_equal(solved, full):
            continue

        row_missing = [set(palette) - {int(v) for v in partial[r] if int(v) != MASK} for r in range(n)]
        col_missing = [set(palette) - {int(v) for v in partial[:, c] if int(v) != MASK} for c in range(n)]
        cross_needed = any(
            mask[r, c] and len(row_missing[r]) > 1 and len(col_missing[c]) > 1
            for r in range(n)
            for c in range(n)
        )
        if not cross_needed:
            continue
        return mask
    return None


def _sample_layout(n: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    side = 2 * n + 1
    layouts = []
    if n == 3:
        layouts.extend(
            [
                ((1, 3), [(0, 0), (0, 1), (0, 2)]),
                ((3, 1), [(0, 0), (1, 0), (2, 0)]),
                ((2, 2), [(0, 0), (0, 1), (1, 0)]),
                ((2, 2), [(0, 0), (0, 1), (1, 1)]),
                ((2, 2), [(0, 0), (1, 0), (1, 1)]),
                ((2, 2), [(0, 1), (1, 0), (1, 1)]),
            ]
        )
    else:
        layouts.append(((2, 2), [(0, 0), (0, 1), (1, 0), (1, 1)]))

    for _ in range(200):
        (rows, cols), slots = layouts[int(RNG.integers(len(layouts)))]
        gap_r = int(RNG.integers(2, 5)) if rows > 1 else 0
        gap_c = int(RNG.integers(2, 5)) if cols > 1 else 0
        top = int(RNG.integers(1, 5))
        bottom = int(RNG.integers(1, 5))
        left = int(RNG.integers(1, 5))
        right = int(RNG.integers(1, 5))
        height = rows * side + (rows - 1) * gap_r + top + bottom
        width = cols * side + (cols - 1) * gap_c + left + right
        object_area = side * side - 1
        if max(height, width) > 30:
            continue
        if height * width - n * object_area <= object_area:
            continue
        positions = []
        for rr, cc in slots:
            r0 = top + rr * (side + gap_r)
            c0 = left + cc * (side + gap_c)
            positions.append((int(r0), int(c0)))
        return (height, width), sorted(positions)
    raise RuntimeError("failed to sample a valid layout")


def _validate_instance(inp: np.ndarray, full: np.ndarray, positions: list[tuple[int, int]]) -> None:
    objects, n = _extract_objects(inp)
    assert len(objects) == n
    partial = np.array([obj["row"] for obj in objects], dtype=int)
    assert np.array_equal(sorted(positions), [tuple(map(int, obj["bbox"][:2])) for obj in objects])
    count, solved = _solve_unique(partial, sorted(int(v) for v in np.unique(full)))
    assert count == 1 and solved is not None
    assert np.array_equal(solved, full)
    row_masks = np.count_nonzero(partial == MASK, axis=1)
    col_masks = np.count_nonzero(partial == MASK, axis=0)
    assert int(row_masks.max()) >= 2
    assert int(col_masks.max()) >= 2
    row_missing = [{int(v) for v in np.unique(full)} - {int(x) for x in partial[r] if int(x) != MASK} for r in range(n)]
    col_missing = [{int(v) for v in np.unique(full)} - {int(x) for x in partial[:, c] if int(x) != MASK} for c in range(n)]
    assert any(
        int(partial[r, c]) == MASK and len(row_missing[r]) > 1 and len(col_missing[c]) > 1
        for r in range(n)
        for c in range(n)
    )
    expected = _paint_from_table(inp.shape, positions, full)
    assert np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    n = 4 if float(RNG.random()) < 0.6 else 3
    palette = [int(v) for v in RNG.choice(PALETTE_POOL, size=n, replace=False)]
    full = _latin_square(n, palette)
    mask = _mask_pattern(full)
    if mask is None:
        return None

    partial = full.copy()
    partial[mask] = MASK
    shape, positions = _sample_layout(n)
    inp = _paint_from_table(shape, positions, partial)
    _validate_instance(inp, full, positions)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring Latin-square input")
