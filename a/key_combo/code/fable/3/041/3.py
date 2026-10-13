import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 800


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return [np.array(comp, dtype=int) for comp in comps]


def _mirrored_index(x: int, period: int) -> int:
    tile, local = divmod(int(x), int(period))
    if tile % 2 == 1:
        return int(period - 1 - local)
    return int(local)


def _tile_color(tile: np.ndarray, r: int, c: int) -> int:
    return int(tile[_mirrored_index(r, tile.shape[0]), _mirrored_index(c, tile.shape[1])])


def _fit_mirrored_tile(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray | None:
    h, w = grid.shape
    if h % tile_h != 0 or w % tile_w != 0:
        return None
    if h // tile_h < 2 or w // tile_w < 2:
        return None

    visible = np.argwhere(grid != BLACK)
    if visible.size == 0:
        return None

    parities: set[tuple[int, int]] = set()
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    for r, c in visible:
        r = int(r)
        c = int(c)
        parities.add(((r // tile_h) % 2, (c // tile_w) % 2))
        br = _mirrored_index(r, tile_h)
        bc = _mirrored_index(c, tile_w)
        color = int(grid[r, c])
        if tile[br, bc] == -1:
            tile[br, bc] = color
        elif int(tile[br, bc]) != color:
            return None

    if len(parities) < 4:
        return None
    if np.any(tile == -1):
        return None
    return tile


def _valid_tiles(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    h, w = grid.shape
    out: list[tuple[int, int, np.ndarray]] = []
    for tile_h in range(2, h + 1):
        if h % tile_h != 0:
            continue
        for tile_w in range(2, w + 1):
            if w % tile_w != 0:
                continue
            tile = _fit_mirrored_tile(grid, tile_h, tile_w)
            if tile is not None:
                out.append((tile_h, tile_w, tile))
    out.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return out


def _infer_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    candidates = _valid_tiles(grid)
    assert candidates, "no mirrored tiling candidate fits the visible mosaic"

    best_h, best_w, best_tile = candidates[0]
    best_area = best_h * best_w
    hole_coords = np.argwhere(grid == BLACK)

    signatures = set()
    for tile_h, tile_w, tile in candidates:
        if tile_h * tile_w != best_area:
            break
        sig = tuple(int(tile[_mirrored_index(r, tile_h), _mirrored_index(c, tile_w)]) for r, c in hole_coords)
        signatures.add(sig)
    assert len(signatures) == 1, "equally small mirrored tiles predict different hole colors"
    return int(best_h), int(best_w), best_tile


def main(grid: np.ndarray) -> np.ndarray:
    """Extract the missing negative shape from a two-way mirrored tiling mosaic.

    Input structure:
    - The grid is a non-black colored mosaic made by repeating one base tile.
    - Neighboring tile copies alternate by vertical and horizontal mirror flips.
    - One 4-connected black region is a missing hole; all other cells show the mosaic.

    Rule:
    - Infer the smallest base tile whose mirrored copies agree with every
      visible non-black pixel.
    - Take the tight bounding box of all black hole cells.
    - In that crop, fill only the original black cells with their inferred
      mosaic colors and set every original mosaic cell to black.

    Key insight:
    - The neighboring tiles are mirror copies, not plain translations, and the
      black hole is the foreground shape to extract rather than a blemish to hide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    hole = g == BLACK
    assert np.any(hole), "expected one black hole"
    comps = _components(hole)
    assert len(comps) == 1, "expected one connected black hole"

    _tile_h, _tile_w, tile = _infer_tile(g)
    r0, c0, r1, c1 = _bbox(hole)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in comps[0]:
        r = int(r)
        c = int(c)
        out[r - r0, c - c0] = _tile_color(tile, r, c)
    assert max(out.shape) <= 30
    return out


def _render_mosaic(tile: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    th, tw = tile.shape
    out = np.empty((th * tile_rows, tw * tile_cols), dtype=int)
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            patch = tile
            if tr % 2 == 1:
                patch = np.flipud(patch)
            if tc % 2 == 1:
                patch = np.fliplr(patch)
            r0 = tr * th
            c0 = tc * tw
            out[r0 : r0 + th, c0 : c0 + tw] = patch
    return out


def _random_tile(tile_h: int, tile_w: int) -> np.ndarray:
    colors = random.sample(PALETTE, random.randint(4, min(7, len(PALETTE))))
    for _ in range(200):
        rows = np.arange(tile_h)[:, None]
        cols = np.arange(tile_w)[None, :]
        base = (rows * random.randint(1, 3) + cols * random.randint(2, 5) + rows * cols) % len(colors)
        tile = np.array([[colors[int(base[r, c])] for c in range(tile_w)] for r in range(tile_h)], dtype=int)

        noise_count = random.randint(max(1, tile_h * tile_w // 8), max(2, tile_h * tile_w // 3))
        for _noise in range(noise_count):
            rr = random.randrange(tile_h)
            cc = random.randrange(tile_w)
            tile[rr, cc] = random.choice(colors)

        if len(set(map(int, tile.ravel()))) < 3:
            continue
        if np.array_equal(tile, np.flipud(tile)):
            continue
        if np.array_equal(tile, np.fliplr(tile)):
            continue
        return tile
    raise RuntimeError("failed to build a non-degenerate tile")


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                out.add((rr, cc))
    return list(out)


def _random_hole_shape(box_h: int, box_w: int) -> np.ndarray | None:
    area = box_h * box_w
    min_cells = max(5, box_h + box_w - 2)
    max_cells = max(min_cells, min(area - 2, (area * 2) // 3))
    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        cells = {(random.randrange(box_h), random.randrange(box_w))}
        while len(cells) < target or not _touches_all_sides(cells, box_h, box_w):
            choices = _frontier(cells, box_h, box_w)
            if not choices:
                break
            cells.add(random.choice(choices))
            if len(cells) >= area - 1:
                break

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _touches_all_sides(cells, box_h, box_w):
            continue
        if int(mask.sum()) >= area - 1:
            continue
        if _bbox(mask) != (0, 0, box_h - 1, box_w - 1):
            continue
        return mask
    return None


def _touches_all_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return 0 in rows and h - 1 in rows and 0 in cols and w - 1 in cols


def _crossing_origins(length: int, box_len: int, period: int) -> list[int]:
    boundaries = [b for b in range(period, length, period)]
    out = []
    for start in range(0, length - box_len + 1):
        end = start + box_len - 1
        if any(start < boundary < end for boundary in boundaries):
            out.append(start)
    return out


def _expected_negative(full: np.ndarray, hole: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(hole)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    crop_hole = hole[r0 : r1 + 1, c0 : c1 + 1]
    out[crop_hole] = full[r0 : r1 + 1, c0 : c1 + 1][crop_hole]
    return out


def _visible_translation_conflict(grid: np.ndarray, dr: int, dc: int) -> bool:
    h, w = grid.shape
    r0a = max(0, -dr)
    r1a = min(h, h - dr)
    c0a = max(0, -dc)
    c1a = min(w, w - dc)
    if r0a >= r1a or c0a >= c1a:
        return False
    a = grid[r0a:r1a, c0a:c1a]
    b = grid[r0a + dr : r1a + dr, c0a + dc : c1a + dc]
    visible = (a != BLACK) & (b != BLACK)
    return bool(np.any(visible & (a != b)))


def _hole_orientations(hole: np.ndarray, tile_h: int, tile_w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in np.argwhere(hole):
        out.add(((int(r) // tile_h) % 2, (int(c) // tile_w) % 2))
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_h = random.randint(3, 6)
    tile_w = random.randint(3, 6)
    tile_rows = random.randint(3, min(5, 30 // tile_h))
    tile_cols = random.randint(3, min(5, 30 // tile_w))

    tile = _random_tile(tile_h, tile_w)
    full = _render_mosaic(tile, tile_rows, tile_cols)
    h, w = full.shape

    box_h = random.randint(3, min(8, h - 1))
    box_w = random.randint(3, min(8, w - 1))
    shape = _random_hole_shape(box_h, box_w)
    if shape is None:
        return None

    row_choices = _crossing_origins(h, box_h, tile_h)
    col_choices = _crossing_origins(w, box_w, tile_w)
    if not row_choices or not col_choices:
        return None
    r0 = random.choice(row_choices)
    c0 = random.choice(col_choices)

    hole = np.zeros(full.shape, dtype=bool)
    hole[r0 : r0 + box_h, c0 : c0 + box_w] = shape
    if len(_components(hole)) != 1:
        return None
    if len(_hole_orientations(hole, tile_h, tile_w)) < 3:
        return None

    inp = full.copy()
    inp[hole] = BLACK
    expected = _expected_negative(full, hole)
    if not np.any(expected == BLACK):
        return None
    if len(set(map(int, expected[expected != BLACK].ravel()))) < 2:
        return None
    if not _visible_translation_conflict(inp, tile_h, 0):
        return None
    if not _visible_translation_conflict(inp, 0, tile_w):
        return None

    try:
        inferred_h, inferred_w, _inferred_tile = _infer_tile(inp)
        actual = main(inp)
    except AssertionError:
        return None
    if (inferred_h, inferred_w) != (tile_h, tile_w):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored tiling puzzle input")
