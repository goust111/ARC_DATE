from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    """Return the uniquely most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], "background must be unique"
    return int(ordered[0][0])


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Convert sorted row/column indices to inclusive contiguous bands."""
    values = [int(x) for x in indices]
    assert values, "expected foreground rows/columns"
    result: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def _compress(grid: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Compress monochrome noisy block intersections; -1 denotes an empty intersection."""
    bg = _background(grid)
    row_bands = _runs(np.flatnonzero(np.any(grid != bg, axis=1)))
    col_bands = _runs(np.flatnonzero(np.any(grid != bg, axis=0)))
    coarse = np.full((len(row_bands), len(col_bands)), -1, dtype=int)

    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = set(map(int, np.unique(patch))) - {bg}
            assert len(colors) <= 1, "each band intersection must use at most one foreground color"
            if colors:
                coarse[i, j] = colors.pop()
    return coarse, bg, row_bands, col_bands


def _learn_pair_map(coarse: np.ndarray, missing: tuple[int, int]) -> dict[int, int]:
    """Read the involutive color exchange from all intact 180-degree cell pairs."""
    h, w = coarse.shape
    mapping: dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            rr, cc = h - 1 - r, w - 1 - c
            if (r, c) == missing or (rr, cc) == missing:
                continue
            source, target = int(coarse[r, c]), int(coarse[rr, cc])
            assert source >= 0 and target >= 0
            if source in mapping:
                assert mapping[source] == target, "inconsistent color pairing"
            mapping[source] = target

    for source, target in mapping.items():
        assert mapping.get(target) == source, "color pairing must exchange in both directions"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a noisy block lattice and restore its one missing cell using learned color-exchanged point symmetry.

    Input structure:
    - Background gaps separate consecutive row bands and column bands of monochrome blocks.
    - Every present block contains background holes, while exactly one whole band intersection is background.
    - At block scale, 180-degree partner cells use a consistent involutive color pairing.

    Rule:
    - Find contiguous foreground row and column bands and replace each intersection by its sole foreground color.
    - Infer the color-pair map from every intact pair of cells opposite about the coarse grid center.
    - Map the missing cell's opposite partner color through that learned table and return the completed coarse grid.

    Key insight:
    - Pixel noise destroys symmetry, but compression reveals point symmetry whose colors must be exchanged by a mapping visible elsewhere in the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    coarse, _bg, _rows, _cols = _compress(g)
    holes = np.argwhere(coarse < 0)
    assert holes.shape[0] == 1, "expected exactly one completely missing block"
    missing = tuple(map(int, holes[0]))

    h, w = coarse.shape
    opposite = (h - 1 - missing[0], w - 1 - missing[1])
    assert opposite != missing, "a central missing block has no visible partner"
    opposite_color = int(coarse[opposite])
    assert opposite_color >= 0, "the opposite block must be present"

    mapping = _learn_pair_map(coarse, missing)
    assert opposite_color in mapping, "the needed color pairing must be demonstrated by intact cells"
    out = coarse.copy()
    out[missing] = mapping[opposite_color]
    assert np.all(out >= 0)
    return out


_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()
_MAX_ATTEMPTS = 500


def _sample_layout(count: int) -> tuple[list[int], list[int], int]:
    """Choose unequal-capable band sizes, separator widths, and an outer margin."""
    sizes = [random.randint(2, 4) for _ in range(count)]
    gaps = [random.randint(1, 2) for _ in range(count - 1)]
    margin = random.randint(1, 3)
    return sizes, gaps, margin


def _origins(sizes: list[int], gaps: list[int], margin: int) -> tuple[list[int], int]:
    origins: list[int] = []
    cursor = margin
    for i, size in enumerate(sizes):
        origins.append(cursor)
        cursor += size
        if i < len(gaps):
            cursor += gaps[i]
    return origins, cursor + margin


def _noisy_mask(height: int, width: int) -> np.ndarray:
    """Make a visibly perforated mask without emptying any local row or column."""
    area = height * width
    max_holes = min(area - max(height, width), max(1, area // 3))
    for _ in range(100):
        mask = np.ones((height, width), dtype=bool)
        hole_count = random.randint(1, max_holes)
        for index in random.sample(range(area), hole_count):
            mask[index // width, index % width] = False
        if np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            return mask
    raise RuntimeError("could not perforate block")


def _coarse_and_mapping(rows: int, cols: int, palette: list[int]):
    """Build a paired coarse grid and choose a non-self orbit that can safely be removed."""
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        if cell in seen:
            continue
        partner = (rows - 1 - cell[0], cols - 1 - cell[1])
        seen.update((cell, partner))
        orbits.append((cell, partner))

    # Pair zero is non-self and occurs at least twice, so its rule survives removal.
    pair_count = random.choice((2, 3)) if len(orbits) >= 5 else 2
    # Odd-by-odd grids need a self-paired color at their central cell; grids
    # without a central cell may instead exercise the all-exchanging branch.
    max_nonself = pair_count - 1 if rows % 2 == 1 and cols % 2 == 1 else pair_count
    n_nonself = random.randint(1, max_nonself)
    needed_colors = 2 * n_nonself + (pair_count - n_nonself)
    chosen = random.sample(palette, needed_colors)
    pairs: list[tuple[int, int]] = []
    cursor = 0
    for _ in range(n_nonself):
        pairs.append((chosen[cursor], chosen[cursor + 1]))
        cursor += 2
    for _ in range(pair_count - n_nonself):
        pairs.append((chosen[cursor], chosen[cursor]))
        cursor += 1

    # A non-self color pair cannot occupy the one-cell central orbit of an
    # odd-by-odd grid.  Assign required demonstrations explicitly before
    # filling the remaining orbits, and reserve two demonstrations of pair 0.
    labels: list[int | None] = [None] * len(orbits)
    noncentral = [i for i, (a_pos, b_pos) in enumerate(orbits) if a_pos != b_pos]
    for index in random.sample(noncentral, 2):
        labels[index] = 0
    for label in range(1, pair_count):
        candidates = [
            i for i, (a_pos, b_pos) in enumerate(orbits)
            if labels[i] is None and (pairs[label][0] == pairs[label][1] or a_pos != b_pos)
        ]
        labels[random.choice(candidates)] = label
    for i, (a_pos, b_pos) in enumerate(orbits):
        if labels[i] is None:
            valid_labels = [
                label for label, (a, b) in enumerate(pairs)
                if a == b or a_pos != b_pos
            ]
            labels[i] = random.choice(valid_labels)

    coarse = np.empty((rows, cols), dtype=int)
    pair_zero_orbits: list[int] = []
    for index, ((a_pos, b_pos), raw_label) in enumerate(zip(orbits, labels)):
        assert raw_label is not None
        label = int(raw_label)
        a, b = pairs[label]
        if a != b and random.random() < 0.5:
            a, b = b, a
        coarse[a_pos], coarse[b_pos] = a, b
        if label == 0 and a_pos != b_pos:
            pair_zero_orbits.append(index)

    assert len(pair_zero_orbits) >= 2
    missing_orbit = random.choice(pair_zero_orbits)
    missing = random.choice(orbits[missing_orbit])
    mapping = {a: b for a, b in pairs}
    mapping.update({b: a for a, b in pairs})
    return coarse, mapping, missing


def _render(coarse: np.ndarray, missing: tuple[int, int], bg: int,
            row_sizes: list[int], col_sizes: list[int],
            row_origins: list[int], col_origins: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for r in range(coarse.shape[0]):
        for c in range(coarse.shape[1]):
            if (r, c) == missing:
                continue
            mask = _noisy_mask(row_sizes[r], col_sizes[c])
            patch = grid[row_origins[r] : row_origins[r] + row_sizes[r],
                         col_origins[c] : col_origins[c] + col_sizes[c]]
            patch[mask] = int(coarse[r, c])
    return grid


def _semantic_check(inp: np.ndarray, expected: np.ndarray, intended_map: dict[int, int],
                    missing: tuple[int, int], intended_rows: int, intended_cols: int) -> bool:
    """Check the stated semantics and reject common shortcut-compatible examples."""
    try:
        partial, _bg, row_bands, col_bands = _compress(inp)
        if partial.shape != (intended_rows, intended_cols):
            return False
        if len(row_bands) != intended_rows or len(col_bands) != intended_cols:
            return False
        holes = np.argwhere(partial < 0)
        if holes.shape != (1, 2) or tuple(map(int, holes[0])) != missing:
            return False
        learned = _learn_pair_map(partial, missing)
        opposite = (intended_rows - 1 - missing[0], intended_cols - 1 - missing[1])
        source = int(partial[opposite])
        if source not in learned or learned[source] != intended_map[source]:
            return False
        # The direct same-color mirror shortcut must give the wrong missing color.
        if int(expected[missing]) == source:
            return False
        # Compression and the hidden exchange must both materially affect the result.
        if inp.shape == expected.shape or np.array_equal(inp, np.rot90(inp, 2)):
            return False
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False
        return bool(np.array_equal(actual[partial >= 0], partial[partial >= 0]))
    except (AssertionError, RuntimeError, ValueError):
        return False


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 6)
    if rows * cols < 12:
        return None

    row_sizes, row_gaps, row_margin = _sample_layout(rows)
    col_sizes, col_gaps, col_margin = _sample_layout(cols)
    row_origins, height = _origins(row_sizes, row_gaps, row_margin)
    col_origins, width = _origins(col_sizes, col_gaps, col_margin)
    if height > 30 or width > 30:
        return None

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    coarse, mapping, missing = _coarse_and_mapping(rows, cols, palette)
    expected = coarse.copy()
    opposite = (rows - 1 - missing[0], cols - 1 - missing[1])
    expected[missing] = mapping[int(coarse[opposite])]

    inp = _render(coarse, missing, bg, row_sizes, col_sizes, row_origins, col_origins, (height, width))
    try:
        if _background(inp) != bg:
            return None
    except AssertionError:
        return None
    if not _semantic_check(inp, expected, mapping, missing, rows, cols):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked lattice with a uniquely recoverable missing block."""
    for _ in range(_MAX_ATTEMPTS):
        result = _generate_attempt()
        if result is None:
            continue
        inp, expected = result
        signature = (expected.shape, expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return inp
    raise RuntimeError(f"failed to generate a valid input after {_MAX_ATTEMPTS} attempts")
