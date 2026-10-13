from __future__ import annotations

import random

import numpy as np


SEPARATOR = 0
ERASED = 5
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _is_separator(line: np.ndarray) -> bool:
    return bool(np.all(line == SEPARATOR))


def _block_intervals(separator_indices: list[int], length: int) -> list[tuple[int, int]]:
    """Return half-open payload intervals between single-cell separator lines."""
    assert separator_indices, "expected at least one black separator line"
    bounds = [-1, *separator_indices, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [b - a for a, b in intervals]
    assert sizes and min(sizes) > 0, "separators must enclose nonempty blocks"
    assert len(set(sizes)) == 1, "blocks must be equally sized on an axis"
    for a, b in zip(separator_indices, separator_indices[1:]):
        assert b > a + 1, "separator lines must be single and separated by blocks"
    return intervals


def _layout(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rows = [r for r in range(g.shape[0]) if _is_separator(g[r, :])]
    cols = [c for c in range(g.shape[1]) if _is_separator(g[:, c])]
    row_intervals = _block_intervals(rows, g.shape[0])
    col_intervals = _block_intervals(cols, g.shape[1])
    assert {b - a for a, b in row_intervals} == {b - a for a, b in col_intervals}, (
        "encoded blocks must be square"
    )
    return row_intervals, col_intervals


def _infer_sprite(
    g: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> tuple[np.ndarray, int, int]:
    mixed: list[np.ndarray] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = g[r0:r1, c0:c1]
            colors = set(map(int, np.unique(block)))
            if len(colors) == 2 and ERASED not in colors and SEPARATOR not in colors:
                mixed.append(block.copy())

    assert mixed, "expected a fully visible mixed-color sprite block"
    sprite = mixed[0]
    assert all(np.array_equal(sprite, other) for other in mixed), (
        "all mixed blocks must be identical sprite copies"
    )
    colors, counts = np.unique(sprite, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1], (
        "sprite must have two colors with a strict majority/minority split"
    )
    majority = int(colors[int(np.argmax(counts))])
    minority = int(colors[int(np.argmin(counts))])
    return sprite, majority, minority


def _smallest_period(known: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Find the unique least-area rectangular residue table supported by known cells."""
    h, w = values.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            tile = np.full((pr, pc), -1, dtype=int)
            ok = True
            for r, c in np.argwhere(known):
                rr, cc = int(r) % pr, int(c) % pc
                value = int(values[r, c])
                if tile[rr, cc] < 0:
                    tile[rr, cc] = value
                elif tile[rr, cc] != value:
                    ok = False
                    break
            if ok and np.all(tile >= 0):
                candidates.append((pr, pc, tile))

    assert candidates, "no completely observed periodic tile fits the decoded cells"
    least_area = min(pr * pc for pr, pc, _ in candidates)
    least = [(pr, pc, tile) for pr, pc, tile in candidates if pr * pc == least_area]
    assert len(least) == 1, "the visible evidence does not determine a unique smallest period"
    pr, pc, tile = least[0]
    return tile, pr, pc


def main(grid: np.ndarray) -> np.ndarray:
    """Decode a sprite/blank block grid and repair erased macro-cells from its smallest period.

    Input structure:
    - Single black separator rows and columns divide the grid into equal square blocks.
    - Each block is an identical two-color sprite, a uniform block of the sprite's
      minority color, or a solid gray erased block whose macro-state is unknown.

    Rule:
    - Decode sprite blocks as the sprite majority color and uniform minority blocks
      as the minority color.  Infer the unique least-area rectangular period that
      agrees with every known macro-cell, then fill gray unknowns from their phases.
    - Return only the repaired macro-grid, one output pixel per input block.

    Key insight:
    - Periodicity exists at the compressed two-state macro level, so the repeated
      sprite must be recognized and decoded before erased cells can be repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D grid"
    row_intervals, col_intervals = _layout(g)
    sprite, majority, minority = _infer_sprite(g, row_intervals, col_intervals)

    macro = np.full((len(row_intervals), len(col_intervals)), -1, dtype=int)
    known = np.zeros_like(macro, dtype=bool)
    for br, (r0, r1) in enumerate(row_intervals):
        for bc, (c0, c1) in enumerate(col_intervals):
            block = g[r0:r1, c0:c1]
            if np.all(block == ERASED):
                continue
            matches_sprite = np.array_equal(block, sprite)
            matches_blank = bool(np.all(block == minority))
            assert matches_sprite != matches_blank, (
                "each visible block must be exactly the sprite or a uniform minority block"
            )
            macro[br, bc] = majority if matches_sprite else minority
            known[br, bc] = True

    assert np.any(~known), "expected at least one gray erased macro-cell"
    tile, pr, pc = _smallest_period(known, macro)
    out = tile[np.arange(macro.shape[0])[:, None] % pr,
               np.arange(macro.shape[1])[None, :] % pc]
    assert np.all(out[known] == macro[known])
    return out.astype(int, copy=False)


def _intrinsic_period(full: np.ndarray) -> tuple[int, int]:
    """Independent full-grid oracle: componentwise smallest exact modulo period."""
    h, w = full.shape
    row_periods = [
        p for p in range(1, h + 1)
        if all(int(full[r, c]) == int(full[r % p, c]) for r in range(h) for c in range(w))
    ]
    col_periods = [
        p for p in range(1, w + 1)
        if all(int(full[r, c]) == int(full[r, c % p]) for r in range(h) for c in range(w))
    ]
    return min(row_periods), min(col_periods)


def _make_sprite(size: int, majority: int, minority: int) -> np.ndarray:
    area = size * size
    minority_count = random.randint(1, max(1, area // 3))
    while True:
        mask = np.zeros((size, size), dtype=bool)
        for idx in random.sample(range(area), minority_count):
            mask[idx // size, idx % size] = True
        # Avoid visually symmetric glyphs and uniform rows/columns that look like dividers.
        symmetric = any(np.array_equal(mask, transform) for transform in (
            np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2)
        ))
        if not symmetric or size == 2:
            return np.where(mask, minority, majority).astype(int)


def _make_tile(pr: int, pc: int, majority: int, minority: int) -> np.ndarray:
    """Sample a nondegenerate binary tile whose two axis periods are both intrinsic."""
    for _ in range(200):
        tile = np.where(
            np.random.random((pr, pc)) < random.uniform(0.35, 0.65),
            majority,
            minority,
        ).astype(int)
        if len(np.unique(tile)) == 2 and _intrinsic_period(tile) == (pr, pc):
            return tile
    raise RuntimeError("could not sample an aperiodic base tile")


def _render_input(
    full: np.ndarray,
    erased: set[tuple[int, int]],
    sprite: np.ndarray,
    majority: int,
    minority: int,
) -> np.ndarray:
    size = sprite.shape[0]
    h, w = full.shape
    canvas = np.full((h * size + h - 1, w * size + w - 1), SEPARATOR, dtype=int)
    for r in range(h):
        for c in range(w):
            r0, c0 = r * (size + 1), c * (size + 1)
            if (r, c) in erased:
                block = np.full_like(sprite, ERASED)
            elif int(full[r, c]) == majority:
                block = sprite
            else:
                assert int(full[r, c]) == minority
                block = np.full_like(sprite, minority)
            canvas[r0:r0 + size, c0:c0 + size] = block
    return canvas


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    tile: np.ndarray,
    erased: set[tuple[int, int]],
    period: tuple[int, int],
) -> bool:
    """Check clue coverage, true minimality, nondegeneracy, and solver agreement."""
    pr, pc = period
    if expected.shape[0] < 2 * pr or expected.shape[1] < 2 * pc:
        return False
    if _intrinsic_period(expected) != period or len(np.unique(tile)) != 2:
        return False
    if not erased or {int(expected[p]) for p in erased} != set(map(int, np.unique(tile))):
        return False  # rules out replacing every gray block by one fixed state
    for rr in range(pr):
        for cc in range(pc):
            occurrences = [
                (r, c) for r in range(expected.shape[0]) for c in range(expected.shape[1])
                if r % pr == rr and c % pc == cc
            ]
            if not any(p not in erased for p in occurrences):
                return False
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    """Generate diverse, independently checked block grids up to 30x30."""
    for _ in range(400):
        size = random.choice((2, 3, 4))
        pr, pc = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
        macro_h, macro_w = 2 * pr, 2 * pc
        physical_h = macro_h * size + macro_h - 1
        physical_w = macro_w * size + macro_w - 1
        if max(physical_h, physical_w) > 30:
            continue

        majority, minority = random.sample(PAYLOAD_COLORS, 2)
        sprite = _make_sprite(size, majority, minority)
        tile = _make_tile(pr, pc, majority, minority)
        full = np.tile(tile, (2, 2))

        positions_by_color = {
            color: [tuple(map(int, p)) for p in np.argwhere(full == color)]
            for color in (majority, minority)
        }
        erased = {
            random.choice(positions_by_color[majority]),
            random.choice(positions_by_color[minority]),
        }
        extra_count = random.randint(0, max(1, full.size // 4))
        available = [
            (r, c) for r in range(macro_h) for c in range(macro_w)
            if (r, c) not in erased
        ]
        erased.update(random.sample(available, min(extra_count, len(available))))

        # Preserve one visible example of every phase, and both visible block states.
        phase_counts = np.zeros((pr, pc), dtype=int)
        for r in range(macro_h):
            for c in range(macro_w):
                if (r, c) not in erased:
                    phase_counts[r % pr, c % pc] += 1
        if np.any(phase_counts == 0):
            continue
        visible_values = {int(full[r, c]) for r in range(macro_h) for c in range(macro_w)
                          if (r, c) not in erased}
        if visible_values != {majority, minority}:
            continue

        inp = _render_input(full, erased, sprite, majority, minority)
        if _semantic_check(inp, full, tile, erased, (pr, pc)):
            return inp
    raise RuntimeError("failed to generate a valid macro-period repair puzzle")
