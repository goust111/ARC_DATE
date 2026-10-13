import random

import numpy as np


BG = 0
MASK = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
GRID_MAX = 30
GEN_RETRIES = 400


def _valid_horizontal_period(grid: np.ndarray, visible: np.ndarray, period: int) -> bool:
    width = grid.shape[1]
    for c in range(period, width):
        same = visible[:, c] & visible[:, c - period]
        if np.any(grid[:, c][same] != grid[:, c - period][same]):
            return False
    return True


def _valid_vertical_period(grid: np.ndarray, visible: np.ndarray, period: int) -> bool:
    height = grid.shape[0]
    for r in range(period, height):
        same = visible[r, :] & visible[r - period, :]
        if np.any(grid[r, :][same] != grid[r - period, :][same]):
            return False
    return True


def _minimal_period_1d(seq: list[int]) -> int:
    n = len(seq)
    for period in range(1, n + 1):
        if n % period != 0:
            continue
        ok = True
        for i in range(period, n):
            if seq[i] != seq[i - period]:
                ok = False
                break
        if ok:
            return period
    return n


def _find_periods(grid: np.ndarray) -> tuple[int, int]:
    visible = grid != MASK
    assert np.any(visible), "expected at least one visible pixel"
    height, width = grid.shape

    tile_h = None
    for period in range(1, height + 1):
        if _valid_vertical_period(grid, visible, period):
            tile_h = period
            break
    tile_w = None
    for period in range(1, width + 1):
        if _valid_horizontal_period(grid, visible, period):
            tile_w = period
            break

    assert tile_h is not None and tile_w is not None
    return tile_h, tile_w


def _extract_tile(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    visible = grid != MASK

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not visible[r, c]:
                continue
            rr = r % tile_h
            cc = c % tile_w
            value = int(grid[r, c])
            if tile[rr, cc] == -1:
                tile[rr, cc] = value
            else:
                assert tile[rr, cc] == value, "visible residue class must be consistent"

    assert np.all(tile != -1), "every tile residue must be visible at least once"
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a periodic tiling pattern that has been partially covered by magenta masks.

    Input structure:
    - The black-background grid is fully filled by a repeating non-black tile pattern.
    - One or two solid magenta rectangles overwrite parts of that tiling.
    - At least one full tile copy remains visible outside the masks.

    Rule:
    - Find the smallest vertical and horizontal periods consistent with all visible
      (non-magenta) cells.
    - Recover the tile content from visible residue classes.
    - Replace every magenta cell with the color from the matching tile residue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == MASK), "expected at least one magenta mask"

    tile_h, tile_w = _find_periods(g)
    tile = _extract_tile(g, tile_h, tile_w)

    out = g.copy()
    mask_coords = np.argwhere(out == MASK)
    for r, c in mask_coords:
        out[r, c] = tile[int(r) % tile_h, int(c) % tile_w]
    return out


def _build_sequence(length: int, color_count: int) -> list[int] | None:
    for _ in range(200):
        seq = [random.randrange(color_count) for _ in range(length)]
        if len(set(seq)) < 2:
            continue
        if _minimal_period_1d(seq) == length:
            return seq
    return None


def _make_tile(tile_h: int, tile_w: int, colors: tuple[int, ...]) -> np.ndarray | None:
    color_count = len(colors)
    for _ in range(300):
        row_seq = _build_sequence(tile_h, color_count)
        col_seq = _build_sequence(tile_w, color_count)
        if row_seq is None or col_seq is None:
            continue

        mode = random.randrange(3)
        tile = np.zeros((tile_h, tile_w), dtype=int)
        for r in range(tile_h):
            for c in range(tile_w):
                if mode == 0:
                    idx = (row_seq[r] + col_seq[c]) % color_count
                elif mode == 1:
                    idx = (row_seq[r] + 2 * col_seq[c]) % color_count
                else:
                    idx = (2 * row_seq[r] + col_seq[c]) % color_count
                tile[r, c] = colors[idx]

        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _minimal_height(tile) != tile_h or _minimal_width(tile) != tile_w:
            continue
        return tile
    return None


def _minimal_height(tile: np.ndarray) -> int:
    height = tile.shape[0]
    for period in range(1, height + 1):
        if height % period != 0:
            continue
        ok = True
        for r in range(period, height):
            if not np.array_equal(tile[r], tile[r - period]):
                ok = False
                break
        if ok:
            return period
    return height


def _minimal_width(tile: np.ndarray) -> int:
    width = tile.shape[1]
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        ok = True
        for c in range(period, width):
            if not np.array_equal(tile[:, c], tile[:, c - period]):
                ok = False
                break
        if ok:
            return period
    return width


def _full_grid_from_tile(
    tile: np.ndarray,
    repeats_h: int,
    repeats_w: int,
    row_phase: int,
    col_phase: int,
) -> np.ndarray:
    base = np.tile(tile, (repeats_h, repeats_w))
    return np.roll(np.roll(base, shift=-row_phase, axis=0), shift=-col_phase, axis=1)


def _tile_starts(size: int, period: int, phase: int) -> list[int]:
    start0 = (-phase) % period
    starts = []
    pos = start0
    while pos + period <= size:
        starts.append(pos)
        pos += period
    return starts


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _sample_masks(
    shape: tuple[int, int],
    tile_h: int,
    tile_w: int,
    reserved_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]] | None:
    height, width = shape
    n_masks = random.randint(1, 2)
    masks: list[tuple[int, int, int, int]] = []

    for _ in range(200):
        masks.clear()
        aligned_count = 0
        for idx in range(n_masks):
            placed = False
            for _ in range(200):
                max_h = min(height - 1, max(tile_h * 2, height // 2))
                max_w = min(width - 1, max(tile_w * 2, width // 2))
                rect_h = random.randint(1, max_h)
                rect_w = random.randint(1, max_w)
                r0 = random.randint(0, height - rect_h)
                c0 = random.randint(0, width - rect_w)
                box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

                if _boxes_overlap(box, reserved_box):
                    continue
                if any(_boxes_overlap(box, other) for other in masks):
                    continue

                if rect_h % tile_h == 0 and rect_w % tile_w == 0:
                    aligned_count += 1
                masks.append(box)
                placed = True
                break
            if not placed:
                break
        if len(masks) != n_masks:
            continue
        if n_masks == 1 or aligned_count < n_masks:
            return list(masks)
    return None


def _masked_input(full: np.ndarray, masks: list[tuple[int, int, int, int]]) -> np.ndarray:
    inp = full.copy()
    for r0, c0, r1, c1 in masks:
        inp[r0 : r1 + 1, c0 : c1 + 1] = MASK
    return inp


def _gen_attempt() -> np.ndarray | None:
    tile_h = random.randint(2, 5)
    tile_w = random.randint(2, 6)
    repeats_h = random.randint(3, GRID_MAX // tile_h)
    repeats_w = random.randint(3, GRID_MAX // tile_w)
    height = tile_h * repeats_h
    width = tile_w * repeats_w

    color_count = random.randint(2, 3)
    colors = tuple(random.sample(PALETTE, color_count))
    tile = _make_tile(tile_h, tile_w, colors)
    if tile is None:
        return None

    row_phase = random.randrange(tile_h)
    col_phase = random.randrange(tile_w)
    full = _full_grid_from_tile(tile, repeats_h, repeats_w, row_phase, col_phase)

    starts_r = _tile_starts(height, tile_h, row_phase)
    starts_c = _tile_starts(width, tile_w, col_phase)
    if not starts_r or not starts_c:
        return None
    rr = random.choice(starts_r)
    cc = random.choice(starts_c)
    reserved_box = (rr, cc, rr + tile_h - 1, cc + tile_w - 1)

    masks = _sample_masks(full.shape, tile_h, tile_w, reserved_box)
    if masks is None:
        return None

    inp = _masked_input(full, masks)
    if not np.any(inp == MASK):
        return None
    if not np.array_equal(main(inp), full):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
