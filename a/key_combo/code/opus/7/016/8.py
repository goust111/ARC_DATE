import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
BLACK = 0


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _color_components4(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    assert h == w and h % 2 == 0, "expected an even square grid"
    half = h // 2
    return (
        _crop_bbox(grid, (0, 0, half - 1, half - 1)),
        _crop_bbox(grid, (0, half, half - 1, h - 1)),
        _crop_bbox(grid, (half, 0, h - 1, half - 1)),
        _crop_bbox(grid, (half, half, h - 1, h - 1)),
    )


def _mark_isolated_template_pixels(template: np.ndarray) -> np.ndarray:
    out = np.asarray(template, dtype=int).copy()
    for comp in _color_components4(out, ORANGE):
        if len(comp) == 1:
            r, c = comp[0]
            out[r, c] = PINK
    return out


def _render(height: int, width: int, source: np.ndarray, marked_template: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=int)
    marked_template = np.asarray(marked_template, dtype=int)
    source_colors = sorted(int(v) for v in np.unique(source))
    assert len(source_colors) == 2 and BLACK not in source_colors, "source quadrant must use exactly two non-black colors"
    swap = {
        source_colors[0]: source_colors[1],
        source_colors[1]: source_colors[0],
    }
    qh, qw = source.shape
    out = np.zeros((int(height), int(width)), dtype=int)
    for r in range(int(height)):
        for c in range(int(width)):
            tile_color = int(marked_template[r % qh, c % qw])
            src = int(source[r % qh, c % qw])
            if tile_color == BLACK:
                out[r, c] = src
            elif tile_color == ORANGE:
                out[r, c] = ORANGE
            else:
                assert tile_color == PINK
                out[r, c] = swap[src]
    return out


def _solve_core(grid: np.ndarray) -> np.ndarray:
    ul, ur, ll, rr = _quadrants(grid)
    out_h = int(np.count_nonzero(ul != BLACK))
    out_w = int(np.count_nonzero(ur != BLACK))
    marked = _mark_isolated_template_pixels(rr)
    return _render(out_h, out_w, ll, marked)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a classified sparse template over a two-color source, using isolated template pixels as color-inversion markers.

    Input structure:
    - The input is an even square grid split into four equal quadrants on a black background.
    - The upper-left quadrant contains scattered blue pixels, whose count determines the output height.
    - The upper-right quadrant contains scattered red pixels, whose count determines the output width.
    - The lower-left quadrant is a full two-color source tile with no black cells, and the lower-right quadrant is a sparse black/orange template.

    Rule:
    - Count non-black pixels in the upper-left quadrant to get H and in the upper-right quadrant to get W.
    - In the lower-right quadrant, find 4-connected orange components; recolor size-1 components to pink and keep larger components orange.
    - Periodically tile this modified template across an H by W canvas, aligned to the top-left of the template.
    - At each tiled position, black template cells copy the lower-left source color, orange template cells stay orange, and pink template cells take the other of the source tile's two colors at the same local coordinate.

    Key insight:
    - The template is not copied literally: isolated orange pixels become inversion switches, so connectivity classification inside the template changes how the two-color source tile is sampled during tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve_core(g)


CLUSTER_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
]


def _choose_counts(side: int) -> tuple[int, int]:
    max_count = min(18, side * side - 2)
    lo = side
    hi = max(lo + 1, max_count)
    h = random.randint(lo, hi)
    w = random.randint(lo, hi)
    return h, w


def _scatter_count_quadrant(side: int, color: int, count: int) -> np.ndarray:
    quad = np.zeros((side, side), dtype=int)
    coords = random.sample([(r, c) for r in range(side) for c in range(side)], count)
    for r, c in coords:
        quad[r, c] = int(color)
    return quad


def _make_source_tile(side: int) -> np.ndarray:
    mode = random.choice(["vstripe", "hstripe", "checker", "diagonal", "blocks"])
    rr, cc = np.indices((side, side))
    if mode == "vstripe":
        period = random.choice([2, 3])
        phase = random.randint(0, period - 1)
        mask = ((cc + phase) % period) == 0
    elif mode == "hstripe":
        period = random.choice([2, 3])
        phase = random.randint(0, period - 1)
        mask = ((rr + phase) % period) == 0
    elif mode == "checker":
        mask = ((rr + cc + random.randint(0, 1)) % 2) == 0
    elif mode == "diagonal":
        period = random.choice([2, 3])
        shift = random.randint(0, period - 1)
        mask = ((rr + 2 * cc + shift) % period) == 0
    else:
        split_r = random.randint(1, side - 2)
        split_c = random.randint(1, side - 2)
        mask = np.zeros((side, side), dtype=bool)
        mask[:split_r, :split_c] = True
        mask[split_r:, split_c:] = True
        if random.choice([True, False]):
            mask = ~mask

    tile = np.where(mask, GREEN, YELLOW).astype(int)

    if np.all(tile == tile[0, 0]) or np.all(tile != tile[0, 0]):
        tile[0, 0] = GREEN
        tile[-1, -1] = YELLOW

    for _ in range(random.randint(1, 3)):
        r0 = random.randint(0, side - 2)
        c0 = random.randint(0, side - 2)
        rh = random.randint(1, min(3, side - r0))
        cw = random.randint(1, min(3, side - c0))
        tile[r0 : r0 + rh, c0 : c0 + cw] = GREEN if tile[r0, c0] == YELLOW else YELLOW

    if len(np.unique(tile)) != 2:
        tile[0, 0] = GREEN
        tile[-1, -1] = YELLOW
    return tile


def _touches_orange_neighbor(template: np.ndarray, r: int, c: int) -> bool:
    side = template.shape[0]
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < side and 0 <= nc < side and template[nr, nc] == ORANGE:
            return True
    return False


def _place_shape(template: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    cells = [(top + dr, left + dc) for dr, dc in shape]
    side = template.shape[0]
    for r, c in cells:
        if not (0 <= r < side and 0 <= c < side):
            return False
        if template[r, c] != BLACK:
            return False
    for r, c in cells:
        template[r, c] = ORANGE
    return True


def _make_template_tile(side: int, source_tile: np.ndarray) -> np.ndarray | None:
    template = np.zeros((side, side), dtype=int)
    cluster_count = random.randint(1, 3)
    singleton_count = random.randint(2, 4)

    placed_clusters = 0
    for _ in range(80):
        if placed_clusters >= cluster_count:
            break
        shape = random.choice(CLUSTER_SHAPES)
        max_r = side - max(dr for dr, _ in shape)
        max_c = side - max(dc for _, dc in shape)
        top = random.randint(0, max_r - 1)
        left = random.randint(0, max_c - 1)
        if _place_shape(template, shape, top, left):
            placed_clusters += 1
    if placed_clusters < 1:
        return None

    placed_singletons = 0
    candidates = [(r, c) for r in range(side) for c in range(side)]
    random.shuffle(candidates)
    for r, c in candidates:
        if placed_singletons >= singleton_count:
            break
        if template[r, c] != BLACK or _touches_orange_neighbor(template, r, c):
            continue
        template[r, c] = ORANGE
        placed_singletons += 1
    if placed_singletons < 1:
        return None

    comps = _color_components4(template, ORANGE)
    sizes = sorted(len(comp) for comp in comps)
    if sizes[0] != 1 or sizes[-1] < 2:
        return None

    singletons = [comp[0] for comp in comps if len(comp) == 1]
    black_cells = np.argwhere(template == BLACK)
    if len(singletons) < 2 or len(black_cells) < 2:
        return None

    singleton_source_colors = {int(source_tile[r, c]) for r, c in singletons}
    black_source_colors = {int(source_tile[r, c]) for r, c in black_cells}
    if singleton_source_colors != {GREEN, YELLOW}:
        return None
    if black_source_colors != {GREEN, YELLOW}:
        return None
    return template


def _validate_semantics(grid: np.ndarray) -> None:
    side2 = grid.shape[0]
    assert grid.shape == (side2, side2)
    ul, ur, ll, rr = _quadrants(grid)
    side = ll.shape[0]

    assert side2 <= 30
    assert np.all((ul == BLACK) | (ul == BLUE))
    assert np.all((ur == BLACK) | (ur == RED))
    assert BLACK not in np.unique(ll)
    assert set(np.unique(ll)) == {GREEN, YELLOW}
    assert set(np.unique(rr)).issubset({BLACK, ORANGE})

    out_h = int(np.count_nonzero(ul))
    out_w = int(np.count_nonzero(ur))
    assert side <= out_h <= 30
    assert side <= out_w <= 30

    comps = _color_components4(rr, ORANGE)
    sizes = sorted(len(comp) for comp in comps)
    assert sizes and sizes[0] == 1 and sizes[-1] >= 2

    marked = _mark_isolated_template_pixels(rr)
    expected = _render(out_h, out_w, ll, marked)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert set(np.unique(actual)).issubset({GREEN, YELLOW, ORANGE})
    assert ORANGE in np.unique(actual)
    assert GREEN in np.unique(actual) and YELLOW in np.unique(actual)

    wrong_no_classification = _render(out_h, out_w, ll, rr)
    wrong_source_only = _render(out_h, out_w, ll, np.zeros_like(rr))
    assert not np.array_equal(actual, wrong_no_classification)
    assert not np.array_equal(actual, wrong_source_only)


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(5, 7)
    out_h, out_w = _choose_counts(side)
    ul = _scatter_count_quadrant(side, BLUE, out_h)
    ur = _scatter_count_quadrant(side, RED, out_w)
    ll = _make_source_tile(side)
    rr = _make_template_tile(side, ll)
    if rr is None:
        return None

    grid = np.zeros((2 * side, 2 * side), dtype=int)
    grid[:side, :side] = ul
    grid[:side, side:] = ur
    grid[side:, :side] = ll
    grid[side:, side:] = rr

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
