from __future__ import annotations

import random
from collections import Counter

import numpy as np


_ARC_COLORS = tuple(range(10))
_GEN_RETRIES = 4000


def _fold_adjacent_rows(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    assert grid.ndim == 2 and grid.shape[0] > 0
    rows = [grid[0].copy()]
    widths = [1]
    for r in range(1, grid.shape[0]):
        if np.array_equal(grid[r], rows[-1]):
            widths[-1] += 1
        else:
            rows.append(grid[r].copy())
            widths.append(1)
    return np.stack(rows, axis=0), widths


def _fold_adjacent_cols(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    assert grid.ndim == 2 and grid.shape[1] > 0
    cols = [grid[:, 0].copy()]
    widths = [1]
    for c in range(1, grid.shape[1]):
        if np.array_equal(grid[:, c], cols[-1]):
            widths[-1] += 1
        else:
            cols.append(grid[:, c].copy())
            widths.append(1)
    return np.stack(cols, axis=1), widths


def _compress(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    row_folded, row_heights = _fold_adjacent_rows(grid)
    core, col_widths = _fold_adjacent_cols(row_folded)
    return core, row_heights, col_widths


def _expand(core: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    out = np.repeat(core, np.asarray(row_heights, dtype=int), axis=0)
    out = np.repeat(out, np.asarray(col_widths, dtype=int), axis=1)
    return out.astype(int, copy=False)


def _damage_color_candidates(core: np.ndarray) -> list[int]:
    h, w = core.shape
    assert w % 2 == 1, "compressed core must have a central column"
    axis = w // 2
    values = set(map(int, core.ravel()))
    candidates: list[int] = []

    for color in values:
        # A damaged fixed-point cell cannot be repaired from a distinct mirror.
        if np.any(core[:, axis] == color):
            continue

        mismatches = 0
        valid = True
        for r in range(h):
            for c in range(axis):
                a = int(core[r, c])
                b = int(core[r, w - 1 - c])
                if a == b:
                    if a == color:
                        valid = False
                        break
                    continue
                if (a == color) ^ (b == color):
                    mismatches += 1
                else:
                    valid = False
                    break
            if not valid:
                break

        if valid and mismatches > 0 and int(np.sum(core == color)) == mismatches:
            candidates.append(color)
    return candidates


def _repair_core(core: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = core.shape
    assert w % 2 == 1, "compressed core width must be odd"
    candidates = _damage_color_candidates(core)
    assert len(candidates) == 1, "damage color must be uniquely determined"
    damage = candidates[0]

    out = core.copy()
    for r, c in np.argwhere(core == damage):
        r = int(r)
        c = int(c)
        mirror_c = w - 1 - c
        assert mirror_c != c, "damage must not lie on the mirror axis"
        assert int(core[r, mirror_c]) != damage, "mirror source must not also be damaged"
        out[r, c] = int(core[r, mirror_c])

    assert np.array_equal(out, out[:, ::-1]), "repaired core must be left-right symmetric"
    return out, damage


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold stretched duplicate bands, repair damaged core cells by horizontal mirror symmetry, then re-expand.

    Input structure:
    - The grid is a stretched copy of a smaller core pattern, made by duplicating
      adjacent row bands and adjacent column bands.
    - The undamaged core is symmetric about its central column, while opposite
      column bands in the stretched grid may have different widths.
    - One damage color, absent from the true core palette, covers whole
      row-band by column-band rectangles and never covers both cells of a
      mirror pair.

    Rule:
    - Collapse each run of adjacent identical rows and columns to obtain the
      compressed core and remember every row and column band width.
    - In the compressed core, replace every damage-color cell by the color at
      the cell reflected across the central column.
    - Expand the repaired core with the original band-width sequences and
      return a grid of the same shape as the input.

    Key insight:
    - The mirror correspondence exists only after compression; reflecting raw
      stretched-grid columns can point into the wrong band when opposite band
      widths differ.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    core, row_heights, col_widths = _compress(g)
    repaired_core, _damage = _repair_core(core)
    out = _expand(repaired_core, row_heights, col_widths)
    assert out.shape == g.shape
    return out


def _has_adjacent_equal_rows(grid: np.ndarray) -> bool:
    return any(np.array_equal(grid[r - 1], grid[r]) for r in range(1, grid.shape[0]))


def _has_adjacent_equal_cols(grid: np.ndarray) -> bool:
    return any(np.array_equal(grid[:, c - 1], grid[:, c]) for c in range(1, grid.shape[1]))


def _sample_widths(n: int, *, max_total: int = 30, require_repeat: bool = False) -> list[int] | None:
    for _ in range(200):
        widths = [random.randint(1, 4) for _ in range(n)]
        if sum(widths) <= max_total and (not require_repeat or any(w > 1 for w in widths)):
            return widths
    return None


def _make_symmetric_core(h: int, w: int, palette: list[int]) -> np.ndarray:
    axis = w // 2
    core = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(axis + 1):
            color = random.choice(palette)
            core[r, c] = color
            core[r, w - 1 - c] = color
    return core


def _palette_has_visible_support(core: np.ndarray, palette: list[int]) -> bool:
    axis = core.shape[1] // 2
    supported: set[int] = set(map(int, core[:, axis]))
    for c in range(axis):
        mirror_c = core.shape[1] - 1 - c
        for color in palette:
            if np.any((core[:, c] == color) & (core[:, mirror_c] == color)):
                supported.add(int(color))
    return set(map(int, palette)) <= supported


def _choose_damage_cells(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    axis = w // 2
    mirror_pairs = [(r, c, w - 1 - c) for r in range(h) for c in range(axis)]
    random.shuffle(mirror_pairs)
    cells: list[tuple[int, int]] = []
    used_pairs: set[tuple[int, int]] = set()

    for r, left_c, right_c in mirror_pairs:
        if len(cells) >= count:
            break
        pair_key = (r, left_c)
        if pair_key in used_pairs:
            continue
        c = left_c if random.random() < 0.5 else right_c
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in cells):
            continue
        cells.append((r, c))
        used_pairs.add(pair_key)

    if len(cells) < count:
        return None

    if count > 1:
        sides = [c < axis for _r, c in cells]
        if all(sides) or not any(sides):
            r, c = cells[-1]
            cells[-1] = (r, w - 1 - c)
            if any(
                i != len(cells) - 1 and abs(cells[-1][0] - rr) <= 1 and abs(cells[-1][1] - cc) <= 1
                for i, (rr, cc) in enumerate(cells)
            ):
                return None
    return cells


def _naive_raw_repair(grid: np.ndarray, damage_color: int) -> np.ndarray:
    out = grid.copy()
    width = grid.shape[1]
    for r, c in np.argwhere(grid == damage_color):
        out[int(r), int(c)] = int(grid[int(r), width - 1 - int(c)])
    return out


def _nondegenerate_damage(core: np.ndarray, damaged_core: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    axis = core.shape[1] // 2
    source_colors = [int(core[r, core.shape[1] - 1 - c]) for r, c in cells]
    damaged_rows = {r for r, _c in cells}
    damaged_cols = {c for _r, c in cells}
    return (
        len(set(source_colors)) >= 2
        and len(damaged_rows) >= 2
        and len(damaged_cols) >= 2
        and any(c < axis for _r, c in cells)
        and any(c > axis for _r, c in cells)
        and not _has_adjacent_equal_rows(damaged_core)
        and not _has_adjacent_equal_cols(damaged_core)
    )


def _gen_attempt() -> np.ndarray | None:
    core_h = random.randint(4, 9)
    core_w = random.choice([5, 7, 9])
    palette_size = random.randint(3, 5)
    chosen = random.sample(_ARC_COLORS, palette_size + 1)
    damage_color = chosen[0]
    palette = chosen[1:]

    row_heights = _sample_widths(core_h, require_repeat=True)
    col_widths = _sample_widths(core_w, require_repeat=True)
    if row_heights is None or col_widths is None:
        return None

    axis = core_w // 2
    if not any(col_widths[c] != col_widths[core_w - 1 - c] for c in range(axis)):
        return None

    core = _make_symmetric_core(core_h, core_w, palette)
    if (
        _has_adjacent_equal_rows(core)
        or _has_adjacent_equal_cols(core)
        or not _palette_has_visible_support(core, palette)
    ):
        return None

    max_damage = min(5, core_h * axis)
    damage_count = random.randint(2, max_damage)
    cells = _choose_damage_cells(core_h, core_w, damage_count)
    if cells is None:
        return None

    damaged_core = core.copy()
    for r, c in cells:
        damaged_core[r, c] = damage_color

    if not _nondegenerate_damage(core, damaged_core, cells):
        return None
    if not _palette_has_visible_support(damaged_core, palette):
        return None

    try:
        repaired_core, inferred_damage = _repair_core(damaged_core)
    except AssertionError:
        return None
    if inferred_damage != damage_color or not np.array_equal(repaired_core, core):
        return None

    inp = _expand(damaged_core, row_heights, col_widths)
    expected = _expand(core, row_heights, col_widths)
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.array_equal(inp, expected):
        return None
    if np.array_equal(_naive_raw_repair(inp, damage_color), expected):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Ensure dimensions, colors, and repaired output vary visibly across samples.
    if len(Counter(map(int, inp.ravel()))) < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compressed mirror-repair input")
