import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MIN_SIDE = 3
MAX_SIDE = 5
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    stack = [tuple(map(int, coords[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(coords)


def _split_panels(grid: np.ndarray):
    h, w = grid.shape
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert len(sep_cols) == 3, "expected exactly three full-height grey separator columns"
    cuts = [-1, *sep_cols, w]
    spans = [(cuts[i] + 1, cuts[i + 1]) for i in range(4)]
    widths = {c1 - c0 for c0, c1 in spans}
    assert len(widths) == 1, "all four panels must have the same width"
    return [grid[:, c0:c1].copy() for c0, c1 in spans]


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    panels = _split_panels(g)
    inner_panel, outer_panel, panel_a, panel_b = panels

    color_a = int(panel_a[0, 0])
    color_b = int(panel_b[0, 0])
    assert np.all(panel_a == color_a) and np.all(panel_b == color_b), "palette panels must be solid"
    assert color_a != color_b, "palette colors must differ"

    inner_mask = inner_panel == GREEN
    outer_mask = outer_panel == GREEN

    assert np.all(np.isin(inner_panel, [BG, GREEN]))
    assert np.all(np.isin(outer_panel, [BG, GREEN]))
    assert inner_mask.any() and outer_mask.any()

    return inner_mask, outer_mask, color_a, color_b


def _parity_kron(inner_mask: np.ndarray, outer_mask: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    h, w = inner_mask.shape
    even_tile = np.where(inner_mask, color_a, color_b)
    odd_tile = np.where(inner_mask, color_b, color_a)
    out = np.full((outer_mask.shape[0] * h, outer_mask.shape[1] * w), color_b, dtype=int)
    for r, c in np.argwhere(outer_mask):
        r0 = int(r) * h
        c0 = int(c) * w
        tile = even_tile if (int(r) + int(c)) % 2 == 0 else odd_tile
        out[r0 : r0 + h, c0 : c0 + w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the inner mask across the outer mask, swapping the two palette colors by outer-cell parity.

    Input structure:
    - Three full-height grey separator columns split the grid into four equal-width panels.
    - The first two panels are same-size binary masks drawn with green foreground on black background.
    - The last two panels are solid color swatches that provide the two output colors.

    Rule:
    - Treat the first panel as the inner stamp mask and the second panel as the outer layout mask.
    - Build an output whose shape is the Kronecker expansion of the two mask shapes.
    - For each green cell of the outer mask, place one copy of the inner mask: even (row+col) uses swatch3 on inner foreground and swatch4 on inner background, odd (row+col) swaps those two colors.
    - Outer-mask background cells contribute solid swatch4 blocks, and the grey separators do not appear in the output.

    Key insight:
    - This is not a plain mask-stamping product: the outer coordinate parity flips the inner tile palette, so outer-scale position information changes every stamped copy.
    """
    inner_mask, outer_mask, color_a, color_b = _parse_input(grid)
    return _parity_kron(inner_mask, outer_mask, color_a, color_b)


def _random_connected_mask(rng: np.random.Generator, side: int, min_cells: int, max_cells: int) -> np.ndarray:
    upper = min(max_cells, side * side - 1)
    lower = min(min_cells, upper)
    assert lower >= 1
    for _ in range(200):
        target = int(rng.integers(lower, upper + 1))
        start = (int(rng.integers(side)), int(rng.integers(side)))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for rr, cc in _neighbors4(r, c, side, side):
                    if (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            rr, cc = frontier[int(rng.integers(len(frontier)))]
            cells.add((rr, cc))
        if len(cells) != target:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rot = int(rng.integers(4))
        if rot:
            mask = np.rot90(mask, rot)
        if int(rng.integers(2)):
            mask = np.flipud(mask)
        if int(rng.integers(2)):
            mask = np.fliplr(mask)

        if _is_connected(mask):
            return mask
    raise RuntimeError("failed to sample a connected mask")


def _outer_mask_ok(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) < 3:
        return False
    parities = {(int(r) + int(c)) % 2 for r, c in coords}
    return len(parities) == 2


def _validate_generated_case(grid: np.ndarray) -> None:
    inner_mask, outer_mask, color_a, color_b = _parse_input(grid)
    side_h, side_w = inner_mask.shape

    assert side_h == side_w
    assert MIN_SIDE <= side_h <= MAX_SIDE
    assert 3 <= int(inner_mask.sum()) <= 7
    assert 3 <= int(outer_mask.sum()) <= 6
    assert _is_connected(inner_mask)
    assert _is_connected(outer_mask)
    assert _outer_mask_ok(outer_mask)

    out = _parity_kron(inner_mask, outer_mask, color_a, color_b)
    naive = np.full_like(out, color_b)
    plain_tile = np.where(inner_mask, color_a, color_b)
    h, w = inner_mask.shape
    for r, c in np.argwhere(outer_mask):
        r0 = int(r) * h
        c0 = int(c) * w
        naive[r0 : r0 + h, c0 : c0 + w] = plain_tile

    assert not np.array_equal(out, naive), "parity flip must materially affect the answer"
    assert set(map(int, np.unique(out))) == {color_a, color_b}
    assert max(grid.shape) <= 30 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        side = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
        inner_mask = _random_connected_mask(rng, side, 3, 7)
        if inner_mask.all() or (~inner_mask).all():
            continue

        outer_mask = _random_connected_mask(rng, side, 3, 6)
        if not _outer_mask_ok(outer_mask):
            continue

        width = side * 4 + 3
        grid = np.zeros((side, width), dtype=int)
        separators = [side, 2 * side + 1, 3 * side + 2]
        grid[:, separators] = GREY

        spans = [(0, side), (side + 1, 2 * side + 1), (2 * side + 2, 3 * side + 2), (3 * side + 3, 4 * side + 3)]

        inner_panel = np.zeros((side, side), dtype=int)
        inner_panel[inner_mask] = GREEN
        outer_panel = np.zeros((side, side), dtype=int)
        outer_panel[outer_mask] = GREEN

        grid[:, spans[0][0] : spans[0][1]] = inner_panel
        grid[:, spans[1][0] : spans[1][1]] = outer_panel
        grid[:, spans[2][0] : spans[2][1]] = BLUE
        grid[:, spans[3][0] : spans[3][1]] = RED

        try:
            _validate_generated_case(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
