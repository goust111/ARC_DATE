import numpy as np


BG = 0
BLUE = 1
PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _reshape_tiles(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    h, w = grid.shape
    rows = h // tile_h
    cols = w // tile_w
    return grid.reshape(rows, tile_h, cols, tile_w).transpose(0, 2, 1, 3)


def _candidate_solution(
    grid: np.ndarray, tile_h: int, tile_w: int
) -> tuple[np.ndarray, tuple[int, int], np.ndarray] | None:
    h, w = grid.shape
    if h % tile_h != 0 or w % tile_w != 0:
        return None

    rows = h // tile_h
    cols = w // tile_w
    tile_count = rows * cols
    if tile_count < 3:
        return None

    tiles = _reshape_tiles(grid, tile_h, tile_w)
    consensus = np.empty((tile_h, tile_w), dtype=int)

    for r in range(tile_h):
        for c in range(tile_w):
            values, counts = np.unique(tiles[:, :, r, c], return_counts=True)
            order = np.argsort(-counts)
            values = values[order]
            counts = counts[order]
            if len(counts) > 1 and counts[0] == counts[1]:
                return None
            consensus[r, c] = int(values[0])

    mismatches = tiles != consensus[None, None, :, :]
    bad_tiles = np.argwhere(mismatches.any(axis=(2, 3)))
    if bad_tiles.shape != (1, 2):
        return None

    bad_r, bad_c = map(int, bad_tiles[0])
    bad_mask = mismatches[bad_r, bad_c]
    if not bad_mask.any():
        return None

    bad_tile = tiles[bad_r, bad_c]
    if not np.all(bad_tile[bad_mask] == BG):
        return None
    if not np.all(consensus[bad_mask] != BG):
        return None
    if np.any(tiles[np.arange(rows)[:, None], np.arange(cols)] if False else False):
        raise AssertionError("unreachable")

    good_mask = np.ones((rows, cols), dtype=bool)
    good_mask[bad_r, bad_c] = False
    if np.any(mismatches[good_mask]):
        return None

    return consensus, (bad_r, bad_c), bad_mask


def _infer_solution(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
    for tile_h in _divisors(h):
        for tile_w in _divisors(w):
            result = _candidate_solution(g, tile_h, tile_w)
            if result is None:
                continue
            consensus, (bad_r, bad_c), bad_mask = result
            rows = h // tile_h
            cols = w // tile_w
            score = (tile_h * tile_w, tile_h, tile_w, -(rows * cols))
            candidates.append((score, consensus, bad_mask))

    assert candidates, "could not infer a valid tile size"
    candidates.sort(key=lambda item: item[0])
    _score, consensus, bad_mask = candidates[0]
    return consensus, bad_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify a repeated tile pattern, repair the unique damaged copy, and mark the
    damaged pixels in blue.
    """
    consensus, bad_mask = _infer_solution(grid)
    out = consensus.copy()
    out[bad_mask] = BLUE
    return out


def _has_row_period(tile: np.ndarray, period: int) -> bool:
    return np.array_equal(tile, tile[np.arange(tile.shape[0]) % period])


def _has_col_period(tile: np.ndarray, period: int) -> bool:
    return np.array_equal(tile, tile[:, np.arange(tile.shape[1]) % period])


def _is_primitive(tile: np.ndarray) -> bool:
    h, w = tile.shape
    if len(set(map(int, tile.ravel()))) < 2:
        return False
    for period in _divisors(h)[:-1]:
        if _has_row_period(tile, period):
            return False
    for period in _divisors(w)[:-1]:
        if _has_col_period(tile, period):
            return False
    return True


def _paint_rect(tile: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = tile.shape
    r0 = int(rng.integers(0, h))
    r1 = int(rng.integers(r0, h))
    c0 = int(rng.integers(0, w))
    c1 = int(rng.integers(c0, w))
    tile[r0 : r1 + 1, c0 : c1 + 1] = color


def _paint_line(tile: np.ndarray, color: int, rng: np.random.Generator, vertical: bool) -> None:
    h, w = tile.shape
    if vertical:
        c = int(rng.integers(0, w))
        r0 = int(rng.integers(0, h))
        r1 = int(rng.integers(r0, h))
        tile[r0 : r1 + 1, c] = color
    else:
        r = int(rng.integers(0, h))
        c0 = int(rng.integers(0, w))
        c1 = int(rng.integers(c0, w))
        tile[r, c0 : c1 + 1] = color


def _paint_l(tile: np.ndarray, color: int, rng: np.random.Generator) -> None:
    h, w = tile.shape
    r = int(rng.integers(0, h))
    c = int(rng.integers(0, w))
    r1 = int(rng.integers(r, h))
    c1 = int(rng.integers(c, w))
    tile[r : r1 + 1, c] = color
    tile[r, c : c1 + 1] = color


def _make_tile(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        tile_h = int(rng.integers(3, 8))
        tile_w = int(rng.integers(3, 8))
        color_count = int(rng.integers(2, 5))
        colors = rng.choice(PALETTE, size=color_count, replace=False)
        tile = np.full((tile_h, tile_w), int(colors[0]), dtype=int)

        op_count = int(rng.integers(4, 9))
        for _ in range(op_count):
            color = int(colors[int(rng.integers(0, len(colors)))])
            shape = int(rng.integers(0, 5))
            if shape == 0:
                _paint_rect(tile, color, rng)
            elif shape == 1:
                _paint_line(tile, color, rng, vertical=False)
            elif shape == 2:
                _paint_line(tile, color, rng, vertical=True)
            elif shape == 3:
                _paint_l(tile, color, rng)
            else:
                r = int(rng.integers(0, tile_h))
                c = int(rng.integers(0, tile_w))
                tile[r, c] = color

        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if not _is_primitive(tile):
            continue
        return tile

    raise RuntimeError("failed to generate a primitive tile")


def _sample_layout(rng: np.random.Generator) -> tuple[int, int]:
    for _ in range(100):
        rows = int(rng.integers(1, 5))
        cols = int(rng.integers(1, 5))
        if rows * cols >= 3:
            return rows, cols
    raise RuntimeError("failed to sample a valid tile layout")


def _damage_mask(tile: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = tile.shape
    total = h * w
    max_cells = max(1, min(6, total // 3))
    mask = np.zeros((h, w), dtype=bool)
    mode = int(rng.integers(0, 3))

    if mode == 0:
        count = int(rng.integers(1, max_cells + 1))
        indices = rng.choice(total, size=count, replace=False)
        mask.flat[indices] = True
        return mask

    if mode == 1:
        r0 = int(rng.integers(0, h))
        r1 = int(rng.integers(r0, h))
        c0 = int(rng.integers(0, w))
        c1 = int(rng.integers(c0, w))
        rect_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if rect_area <= max_cells:
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
            return mask

    count = int(rng.integers(1, max_cells + 1))
    start = int(rng.integers(0, total))
    frontier = [start]
    seen = {start}
    while frontier and np.count_nonzero(mask) < count:
        current = frontier.pop(0)
        mask.flat[current] = True
        r, c = divmod(current, w)
        neighbors = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                nxt = nr * w + nc
                if nxt not in seen:
                    seen.add(nxt)
                    neighbors.append(nxt)
        rng.shuffle(neighbors)
        frontier.extend(neighbors)

    if not mask.any():
        mask.flat[start] = True
    return mask


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        tile = _make_tile(rng)
        rows, cols = _sample_layout(rng)

        grid = np.tile(tile, (rows, cols))
        damage_mask = _damage_mask(tile, rng)
        bad_r = int(rng.integers(0, rows))
        bad_c = int(rng.integers(0, cols))

        r0 = bad_r * tile.shape[0]
        c0 = bad_c * tile.shape[1]
        grid[r0 : r0 + tile.shape[0], c0 : c0 + tile.shape[1]][damage_mask] = BG

        expected = tile.copy()
        expected[damage_mask] = BLUE
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid input")
