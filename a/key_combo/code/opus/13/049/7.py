from __future__ import annotations

import numpy as np


BLACK = 0
ORANGE = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
GEN_ATTEMPTS = 400


def _widest_orange_span(grid: np.ndarray) -> tuple[int, int, int]:
    orange = grid == ORANGE
    rows = np.where(np.any(orange, axis=1))[0]
    assert len(rows) > 0, "expected an orange chevron"

    best: tuple[int, int, int, int] | None = None
    for r in rows:
        cols = np.where(orange[r])[0]
        c0 = int(cols.min())
        c1 = int(cols.max())
        span = c1 - c0
        candidate = (span, int(r), c0, c1)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    _span, r, c0, c1 = best
    return r, c0, c1


def _tile_for_period(
    grid: np.ndarray, known: np.ndarray, period: tuple[int, int]
) -> np.ndarray | None:
    pr, pc = period
    tile = np.full((pr, pc), -1, dtype=int)

    for r_raw, c_raw in known:
        r = int(r_raw)
        c = int(c_raw)
        rr = r % pr
        cc = c % pc
        color = int(grid[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = color
        elif tile[rr, cc] != color:
            return None

    if np.any(tile == -1):
        return None
    return tile


def _infer_unique_period(
    grid: np.ndarray,
    known: np.ndarray,
    max_pr: int,
    max_pc: int,
) -> tuple[int, int, np.ndarray]:
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for pr in range(1, max_pr + 1):
        for pc in range(1, max_pc + 1):
            tile = _tile_for_period(grid, known, (pr, pc))
            if tile is not None:
                candidates.append((pr * pc, pr, pc, tile))

    assert candidates, "visible strip must define at least one periodic tile"
    min_area = min(area for area, _pr, _pc, _tile in candidates)
    best = [(pr, pc, tile) for area, pr, pc, tile in candidates if area == min_area]
    assert len(best) == 1, "visible strip must determine one smallest tile"
    return best[0]


def _complete_pattern(
    shape: tuple[int, int],
    row_range: tuple[int, int],
    period: tuple[int, int],
    tile: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    r0, r1 = row_range
    pr, pc = period
    for r in range(r0, r1 + 1):
        for c in range(shape[1]):
            out[r, c] = int(tile[r % pr, c % pc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden periodic band from the columns exposed by an orange chevron and erase the chevron.

    Input structure:
    - The grid has black background, one orange chevron/V object in the upper part, and a lower periodic color band.
    - The lower band uses only non-black, non-orange colors, but in the input it is visible only in the columns spanned by the chevron's widest row.
    - Outside those covered columns, the lower band is black and acts as the blind area that must be inferred.

    Rule:
    - Find the orange chevron and take the widest row span between its leftmost and rightmost orange pixels as the covered column interval.
    - From the visible lower-band pixels inside that interval, infer the unique smallest rectangular period tile.
    - Fill every black blind-area cell in the same lower-band rows by repeating that tile with absolute residues `(r mod pH, c mod pW)`.
    - Remove the orange chevron by repainting all orange pixels black.

    Key insight:
    - The chevron is not transformed itself; it defines exactly which columns reveal enough samples to infer the hidden two-dimensional periodic pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    widest_row, c0, c1 = _widest_orange_span(g)
    data_mask = (g != BLACK) & (g != ORANGE)
    data_rows = np.where(np.any(data_mask, axis=1))[0]
    assert len(data_rows) > 0, "expected a visible lower periodic band"

    band_r0 = int(data_rows.min())
    band_r1 = int(data_rows.max())
    known = np.argwhere(data_mask)
    pr, pc, tile = _infer_unique_period(g, known, band_r1 - band_r0 + 1, c1 - c0 + 1)

    out = g.copy()
    covered = np.zeros(g.shape[1], dtype=bool)
    covered[c0 : c1 + 1] = True
    for r in range(band_r0, band_r1 + 1):
        for c in range(g.shape[1]):
            if covered[c]:
                continue
            if out[r, c] == BLACK:
                out[r, c] = int(tile[r % pr, c % pc])

    out[out == ORANGE] = BLACK
    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _full_known_period(tile: np.ndarray) -> tuple[int, int]:
    rows, cols = tile.shape
    rr, cc = np.indices(tile.shape)
    known = np.column_stack([rr.ravel(), cc.ravel()])
    pr, pc, _ = _infer_unique_period(tile, known, rows, cols)
    return pr, pc


def _random_tile(rng: np.random.Generator, p_h: int) -> np.ndarray:
    for _ in range(300):
        n_colors = _randint(rng, 2, 3)
        colors = rng.choice(PALETTE, size=n_colors, replace=False)
        tile = rng.choice(colors, size=(p_h, 2), replace=True).astype(int)
        if len(np.unique(tile)) != n_colors:
            continue
        if _full_known_period(tile) != (p_h, 2):
            continue
        return tile
    raise RuntimeError("failed to construct a non-degenerate periodic tile")


def _build_case(
    height: int,
    width: int,
    vertex_row: int,
    center_col: int,
    depth: int,
    band_r0: int,
    band_h: int,
    tile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = center_col - depth
    right = center_col + depth
    band_r1 = band_r0 + band_h - 1
    expected = _complete_pattern((height, width), (band_r0, band_r1), tile.shape, tile)

    inp = expected.copy()
    inp[:, :left] = inp[:, :left]
    inp[:, right + 1 :] = inp[:, right + 1 :]
    inp[band_r0 : band_r1 + 1, :left] = BLACK
    inp[band_r0 : band_r1 + 1, right + 1 :] = BLACK

    for i in range(depth + 1):
        inp[vertex_row + i, center_col - i] = ORANGE
        inp[vertex_row + i, center_col + i] = ORANGE

    return inp, expected


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        widest_row, c0, c1 = _widest_orange_span(inp)
        if widest_row >= inp.shape[0] - 2 or c0 == 0 or c1 == inp.shape[1] - 1:
            return False

        visible_mask = (inp != BLACK) & (inp != ORANGE)
        if not np.any(visible_mask):
            return False

        data_rows = np.where(np.any(visible_mask, axis=1))[0]
        band_r0 = int(data_rows.min())
        band_r1 = int(data_rows.max())
        if band_r0 <= widest_row:
            return False
        if not np.array_equal(data_rows, np.arange(band_r0, band_r1 + 1)):
            return False

        blind_left = (band_r1 - band_r0 + 1) * c0
        blind_right = (band_r1 - band_r0 + 1) * (inp.shape[1] - c1 - 1)
        if blind_left == 0 or blind_right == 0:
            return False

        known = np.argwhere(visible_mask)
        pr, pc, tile = _infer_unique_period(inp, known, band_r1 - band_r0 + 1, c1 - c0 + 1)
        if (pr, pc) not in {(2, 2), (3, 2)}:
            return False

        for r in range(band_r0, band_r1 + 1):
            for c in range(inp.shape[1]):
                if c0 <= c <= c1:
                    if int(inp[r, c]) != int(tile[r % pr, c % pc]):
                        return False
                else:
                    if int(inp[r, c]) != BLACK:
                        return False
                    if int(expected[r, c]) != int(tile[r % pr, c % pc]):
                        return False

        solved = main(inp)
        if not np.array_equal(solved, expected):
            return False
        if np.any(solved == ORANGE):
            return False
        return True
    except Exception:
        return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = _randint(rng, 15, 22)
    width = _randint(rng, 14, 22)

    valid_depths = [depth for depth in (2, 3, 4) if 2 * depth + 3 <= width]
    if not valid_depths:
        return None
    depth = int(rng.choice(valid_depths))
    p_h = int(rng.choice([2, 3]))
    tile = _random_tile(rng, p_h)

    vertex_row = _randint(rng, 1, 3)
    widest_row = vertex_row + depth
    gap = _randint(rng, 1, 2)
    band_r0 = widest_row + gap + 1
    min_band_h = max(2 * p_h + 1, 6)
    max_band_h = min(10, height - band_r0)
    if max_band_h < min_band_h:
        return None
    band_h = _randint(rng, min_band_h, max_band_h)

    min_center = depth + 1
    max_center = width - depth - 2
    if min_center > max_center:
        return None
    center_col = _randint(rng, min_center, max_center)

    inp, expected = _build_case(
        height=height,
        width=width,
        vertex_row=vertex_row,
        center_col=center_col,
        depth=depth,
        band_r0=band_r0,
        band_h=band_h,
        tile=tile,
    )
    if not _validate_case(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chevron-period input")
