import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
TILE_SIDE_MIN = 2
TILE_SIDE_MAX = 6
MAX_DEVIANTS = 6


def _repeat_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols].copy()


def _has_row_period(tile: np.ndarray, period: int) -> bool:
    rows = np.arange(tile.shape[0])[:, None] % period
    cols = np.arange(tile.shape[1])[None, :]
    return np.array_equal(tile, tile[rows, cols])


def _has_col_period(tile: np.ndarray, period: int) -> bool:
    rows = np.arange(tile.shape[0])[:, None]
    cols = np.arange(tile.shape[1])[None, :] % period
    return np.array_equal(tile, tile[rows, cols])


def _is_minimal_tile(tile: np.ndarray) -> bool:
    height, width = tile.shape
    return (
        all(not _has_row_period(tile, period) for period in range(1, height))
        and all(not _has_col_period(tile, period) for period in range(1, width))
    )


def _majority_period_info(
    grid: np.ndarray, period_r: int, period_c: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    height, width = grid.shape
    tile = np.zeros((period_r, period_c), dtype=int)
    deviants: list[tuple[int, int, int]] = []

    for rr in range(period_r):
        row_idx = np.arange(rr, height, period_r)
        for cc in range(period_c):
            col_idx = np.arange(cc, width, period_c)
            block = grid[np.ix_(row_idx, col_idx)]
            values = block.ravel()
            counts = np.bincount(values, minlength=10)
            best = counts.max()
            winners = np.flatnonzero(counts == best)
            if len(winners) != 1:
                return None

            color = int(winners[0])
            tile[rr, cc] = color
            mismatch = np.argwhere(block != color)
            for br, bc in mismatch:
                deviants.append(
                    (
                        int(rr + br * period_r),
                        int(cc + bc * period_c),
                        int(block[br, bc]),
                    )
                )

    return tile, deviants


def _infer_periods(
    grid: np.ndarray,
) -> tuple[int, int, np.ndarray, list[tuple[int, int, int]]]:
    height, width = grid.shape
    assert height >= 4 and width >= 4, "grid must repeat in both dimensions"

    candidates: list[
        tuple[tuple[int, int], int, int, np.ndarray, list[tuple[int, int, int]]]
    ] = []
    for period_r in range(1, height // 2 + 1):
        for period_c in range(1, width // 2 + 1):
            info = _majority_period_info(grid, period_r, period_c)
            if info is None:
                continue
            tile, deviants = info
            candidates.append(((len(deviants), period_r * period_c), period_r, period_c, tile, deviants))

    assert candidates, "no repeated period with unique majority classes was found"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_rank = candidates[0][0]
    best = [item for item in candidates if item[0] == best_rank]
    assert len(best) == 1, f"periods are ambiguous under majority vote: {best_rank}"

    _rank, period_r, period_c, tile, deviants = best[0]
    return period_r, period_c, tile, deviants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the periodic-rule-breaking pixels back into the inferred tile coordinates.

    Input structure:
    - The grid is a dense multicolor pattern formed by repeating one unknown minimal
      tile with row period p and column period q.
    - A small number of cells break that periodic rule: within each (row mod p,
      col mod q) class, one minority color disagrees with the majority base color.
    - The deviant colors are ordinary palette colors already used elsewhere in the pattern.

    Rule:
    - Search for the unique smallest-area repeating period whose residue classes all
      have a unique majority color, using only periods that repeat at least twice on
      each axis of the observed grid.
    - For that inferred p x q tile, treat each residue-class majority as the base color
      and mark every minority cell as a deviant.
    - Output a black p x q grid and place each deviant's wrong color at its folded
      coordinate (row mod p, col mod q); leave all other output cells black.

    Key insight:
    - The sparse message is not localized in image space; it is hidden inside a periodic
      quotient space, so the right operation is majority-based period inference followed
      by coordinate folding, not cropping a visible bbox.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2

    period_r, period_c, tile, deviants = _infer_periods(grid)
    assert deviants, "expected at least one periodic-rule-breaking cell"

    out = np.zeros((period_r, period_c), dtype=int)
    for r, c, color in deviants:
        rr = int(r) % period_r
        cc = int(c) % period_c
        assert color != BLACK, "deviant colors must remain visible in the folded output"
        assert color != int(tile[rr, cc]), "deviant color must disagree with its base tile color"
        assert out[rr, cc] == BLACK, "different deviants may not fold onto the same tile coordinate"
        out[rr, cc] = color
    return out


def _sample_tile(period_r: int, period_c: int) -> tuple[np.ndarray, list[int]]:
    tile_area = period_r * period_c
    palette_size = random.randint(3, min(6, tile_area))
    palette = random.sample(ARC_COLORS, palette_size)

    for _ in range(200):
        cells = palette.copy()
        if tile_area >= 2 * palette_size:
            cells.extend(palette)
        while len(cells) < tile_area:
            cells.append(random.choice(palette))
        random.shuffle(cells)
        tile = np.array(cells[:tile_area], dtype=int).reshape(period_r, period_c)
        if _is_minimal_tile(tile):
            return tile, palette

    raise RuntimeError("failed to sample a minimal periodic tile")


def _sample_extent(period: int) -> int:
    repeats = random.randint(2, min(4, 30 // period))
    full = repeats * period
    extra = random.randint(0, min(period - 1, 30 - full))
    return full + extra


def _bbox_of_coords(coords: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c, _color in coords]
    cols = [c for _r, c, _color in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_shortcut_matches(
    expected: np.ndarray, deviants: list[tuple[int, int, int]]
) -> bool:
    r0, c0, r1, c1 = _bbox_of_coords(deviants)
    bbox_grid = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c, color in deviants:
        bbox_grid[r - r0, c - c0] = color
    return bbox_grid.shape == expected.shape and np.array_equal(bbox_grid, expected)


def _support_count(length: int, period: int, residue: int) -> int:
    return (length - 1 - residue) // period + 1


def _choose_deviant_residues(period_r: int, period_c: int) -> list[tuple[int, int]]:
    residues = [(rr, cc) for rr in range(period_r) for cc in range(period_c)]
    limit = min(MAX_DEVIANTS, len(residues))
    count = random.randint(2, limit)

    for _ in range(200):
        chosen = random.sample(residues, count)
        if len({rr for rr, _cc in chosen}) >= 2 and len({cc for _rr, cc in chosen}) >= 2:
            return chosen
    raise RuntimeError("failed to choose informative deviant residues")


def _inject_deviants(
    clean: np.ndarray,
    tile: np.ndarray,
    palette: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    period_r, period_c = tile.shape
    height, width = clean.shape
    residues = _choose_deviant_residues(period_r, period_c)

    grid = clean.copy()
    out = np.zeros((period_r, period_c), dtype=int)
    injected: list[tuple[int, int, int]] = []
    folded: list[tuple[int, int, int]] = []

    for rr, cc in residues:
        assert _support_count(height, period_r, rr) >= 2
        assert _support_count(width, period_c, cc) >= 2
        positions = [(r, c) for r in range(rr, height, period_r) for c in range(cc, width, period_c)]
        r, c = random.choice(positions)
        base_color = int(tile[rr, cc])
        wrong_choices = [color for color in palette if color != base_color]
        wrong_color = random.choice(wrong_choices)
        assert int(np.count_nonzero(clean == wrong_color)) >= 2
        grid[r, c] = wrong_color
        out[rr, cc] = wrong_color
        injected.append((r, c, wrong_color))
        folded.append((rr, cc, wrong_color))

    return grid, out, injected, folded


def _semantic_checks(
    clean: np.ndarray,
    grid: np.ndarray,
    expected: np.ndarray,
    tile: np.ndarray,
    injected: list[tuple[int, int, int]],
    folded: list[tuple[int, int, int]],
) -> bool:
    period_r, period_c = tile.shape

    if not _is_minimal_tile(tile):
        return False
    if _bbox_shortcut_matches(expected, injected):
        return False
    if len({(rr, cc) for rr, cc, _color in folded}) != len(folded):
        return False
    if len({rr for rr, _cc, _color in folded}) < 2 or len({cc for _rr, cc, _color in folded}) < 2:
        return False

    try:
        got_period_r, got_period_c, inferred_tile, inferred_deviants = _infer_periods(grid)
    except Exception:
        return False

    if (got_period_r, got_period_c) != (period_r, period_c):
        return False
    if not np.array_equal(inferred_tile, tile):
        return False

    injected_set = {(int(r), int(c), int(color)) for r, c, color in injected}
    inferred_set = {(int(r), int(c), int(color)) for r, c, color in inferred_deviants}
    if injected_set != inferred_set:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(clean, grid):
        return False
    return True


def _generate_case() -> np.ndarray | None:
    period_r = random.randint(TILE_SIDE_MIN, TILE_SIDE_MAX)
    period_c = random.randint(TILE_SIDE_MIN, TILE_SIDE_MAX)
    tile, palette = _sample_tile(period_r, period_c)

    height = _sample_extent(period_r)
    width = _sample_extent(period_c)
    clean = _repeat_tile(tile, height, width)
    grid, expected, injected, folded = _inject_deviants(clean, tile, palette)

    if not _semantic_checks(clean, grid, expected, tile, injected, folded):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-folding ARC input")
