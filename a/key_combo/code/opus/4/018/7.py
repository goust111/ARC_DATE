from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste


BG = Color.BLACK
DEFECT = Color.MAROON
RUN_MARK = Color.GREY
VISIBLE_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
]

TILE_H_RANGE = (5, 8)
TILE_W_RANGE = (5, 8)
RUN_LEN_OPTIONS = (1, 2, 3)
PALETTE_LEN_OPTIONS = (2, 3)
MAX_GENERATION_ATTEMPTS = 500


def _split_tiles(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    assert h % 2 == 0, "the tiled region must have an even height"
    assert w >= 3 and (w - 1) % 2 == 0, "expected a 2x2 tiled region plus one legend column"

    tile_h = h // 2
    tile_w = (w - 1) // 2
    tiles = [
        g[:tile_h, :tile_w].copy(),
        g[:tile_h, tile_w : 2 * tile_w].copy(),
        g[tile_h:, :tile_w].copy(),
        g[tile_h:, tile_w : 2 * tile_w].copy(),
    ]
    return tiles, g[:, -1].copy()


def _majority_tile(tiles: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(tiles, axis=0)
    _, tile_h, tile_w = stack.shape
    out = np.empty((tile_h, tile_w), dtype=int)

    for r in range(tile_h):
        for c in range(tile_w):
            counts = Counter(map(int, stack[:, r, c]))
            color, freq = counts.most_common(1)[0]
            assert freq >= 3, "each tile pixel must have a unique 3-of-4 majority"
            assert list(counts.values()).count(freq) == 1, "majority color must be unique"
            out[r, c] = int(color)
    return out


def _find_defect(tiles: list[np.ndarray], base_tile: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    clean_tiles = 0
    total_defects = 0

    for tile_idx, tile in enumerate(tiles):
        mismatches = np.argwhere(tile != base_tile)
        total_defects += int(np.count_nonzero(tile == DEFECT))
        if mismatches.shape[0] == 0:
            clean_tiles += 1
            continue
        if mismatches.shape[0] != 1:
            continue
        r, c = map(int, mismatches[0])
        if int(tile[r, c]) == int(DEFECT):
            candidates.append((tile_idx, r, c))

    assert clean_tiles == 3, "expected three intact tile copies"
    assert total_defects == 1, "expected one maroon defect pixel in the tiled region"
    assert len(candidates) == 1, "expected exactly one one-pixel maroon defect"
    return candidates[0]


def _read_legend(edge: np.ndarray, bg: int) -> tuple[int, list[int]]:
    edge = np.asarray(edge, dtype=int)
    nz = np.flatnonzero(edge != int(bg))
    assert nz.size > 0, "legend column must contain non-background markers"

    start = int(nz[0])
    pos = start
    run_len = 0
    while pos < edge.shape[0] and int(edge[pos]) == int(RUN_MARK):
        run_len += 1
        pos += 1
    assert 1 <= run_len <= 3, "grey legend run must have height 1..3"

    palette: list[int] = []
    while pos < edge.shape[0] and int(edge[pos]) != int(bg):
        color = int(edge[pos])
        assert color not in (int(bg), int(RUN_MARK), int(DEFECT)), "palette colors must be visible and distinct from markers"
        palette.append(color)
        pos += 1

    assert 2 <= len(palette) <= 3, "expected a palette of length 2 or 3"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert np.all(edge[:start] == int(bg)), "legend markers must begin at the first non-background cell"
    assert np.all(edge[pos:] == int(bg)), "legend column must stop after the palette"
    return run_len, palette


def _cyclic_recolor(shape: tuple[int, int], palette: list[int], run_len: int, offset: int) -> np.ndarray:
    tile_h, tile_w = map(int, shape)
    cycle = len(palette) * int(run_len)
    assert cycle > 0

    out = np.empty((tile_h, tile_w), dtype=int)
    for idx in range(tile_h * tile_w):
        rr, cc = divmod(idx, tile_w)
        j = (idx + int(offset)) % cycle
        out[rr, cc] = int(palette[j // int(run_len)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the repeated tile, decode the unique maroon defect position as a phase offset, and recolor the tile by the right-edge palette cycle.

    Input structure:
    - The rightmost column is a legend: a grey vertical run marks the per-color run length, and the cells directly below it form a 2- or 3-color palette.
    - All remaining columns form a 2x2 tiling of one base tile, so the grid height is twice the tile height and the non-legend width is twice the tile width.
    - Exactly one of the four tile copies contains a single maroon defect pixel; the other three copies match the base tile exactly.

    Rule:
    - Split off the legend column and partition the remaining area into four equal tile copies.
    - Determine the base tile by taking the pixelwise majority color across the four copies.
    - Locate the unique tile whose only mismatch from that majority tile is the maroon defect, and convert the defect's local `(row, col)` into the row-major offset `D = row * tile_width + col`.
    - Repaint the single recovered tile in row-major order using the palette colors cyclically, with each palette color repeated for `R` consecutive cells and with the cycle phase shifted by `D`.

    Key insight:
    - The maroon pixel is not damage to repair: its tile-local row-major index is the signal that sets the starting phase of the palette cycle.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    tiles, legend = _split_tiles(g)
    base_tile = _majority_tile(tiles)
    _tile_idx, defect_r, defect_c = _find_defect(tiles, base_tile)
    run_len, palette = _read_legend(legend, bg)

    tile_w = int(base_tile.shape[1])
    offset = defect_r * tile_w + defect_c
    return _cyclic_recolor(base_tile.shape, palette, run_len, offset)


def _neighbors4(h: int, w: int, r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def _grow_connected_mask(
    h: int,
    w: int,
    area: int,
    rng: random.Random,
    blocked: np.ndarray | None = None,
) -> np.ndarray | None:
    blocked_mask = np.zeros((h, w), dtype=bool) if blocked is None else blocked.astype(bool)
    free = np.argwhere(~blocked_mask)
    if free.shape[0] < area or free.shape[0] == 0:
        return None

    mask = np.zeros((h, w), dtype=bool)
    start_r, start_c = map(int, free[rng.randrange(free.shape[0])])
    mask[start_r, start_c] = True

    while int(mask.sum()) < area:
        frontier: list[tuple[int, int]] = []
        cells = np.argwhere(mask)
        for rr, cc in cells:
            for nr, nc in _neighbors4(h, w, int(rr), int(cc)):
                if blocked_mask[nr, nc] or mask[nr, nc]:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            return None
        pick_r, pick_c = frontier[rng.randrange(len(frontier))]
        mask[pick_r, pick_c] = True

    rows, cols = np.where(mask)
    if len(set(map(int, rows))) < 2 or len(set(map(int, cols))) < 2:
        return None
    return mask


def _choose_component_areas(tile_h: int, tile_w: int, rng: random.Random) -> tuple[int, ...]:
    total = tile_h * tile_w
    component_count = 1 if rng.random() < 0.5 else 2
    if component_count == 1:
        lo = max(7, total // 4)
        hi = max(lo, int(total * 0.55))
        return (rng.randint(lo, hi),)

    first_lo = max(6, total // 5)
    first_hi = max(first_lo, int(total * 0.35))
    first = rng.randint(first_lo, first_hi)
    remaining = total - first
    second_lo = 4
    second_hi = min(max(second_lo, total // 5), remaining)
    if second_hi < second_lo:
        return (rng.randint(max(7, total // 4), max(7, total // 2)),)
    second = rng.randint(second_lo, second_hi)
    return (first, second)


def _build_tile_mask(tile_h: int, tile_w: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((tile_h, tile_w), dtype=bool)
    areas = _choose_component_areas(tile_h, tile_w, rng)

    first = _grow_connected_mask(tile_h, tile_w, areas[0], rng)
    if first is None:
        return None
    mask |= first

    if len(areas) == 2:
        second = _grow_connected_mask(tile_h, tile_w, areas[1], rng, blocked=_dilate4(mask))
        if second is None:
            return None
        mask |= second

    comps = find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=False, return_format="dict")
    if len(comps) != len(areas):
        return None
    return mask


def _colorize_tile(mask: np.ndarray, colors: list[int], rng: random.Random) -> np.ndarray:
    tile = np.zeros(mask.shape, dtype=int)
    coords = np.argwhere(mask)
    phases = rng.randrange(len(colors))
    style = rng.randrange(3)

    for idx, (r, c) in enumerate(coords):
        rr = int(r)
        cc = int(c)
        if style == 0:
            color_idx = (idx + phases) % len(colors)
        elif style == 1:
            color_idx = (2 * rr + cc + phases) % len(colors)
        else:
            color_idx = (rr + 2 * cc + phases) % len(colors)
        tile[rr, cc] = int(colors[color_idx])

    return tile


def _make_base_tile(
    tile_h: int,
    tile_w: int,
    palette_len: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[int]] | None:
    color_count = rng.randint(3, 4)
    tile_colors = rng.sample(VISIBLE_COLORS, color_count)
    palette_candidates = [color for color in VISIBLE_COLORS if color not in tile_colors]
    if len(palette_candidates) < palette_len:
        return None

    mask = _build_tile_mask(tile_h, tile_w, rng)
    if mask is None:
        return None

    tile = _colorize_tile(mask, tile_colors, rng)
    used = sorted(set(map(int, tile.ravel())) - {int(BG)})
    if len(used) < 3:
        return None

    comps = find_components(tile, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    if not (1 <= len(comps) <= 2):
        return None

    palette = rng.sample(palette_candidates, palette_len)
    return tile, palette


def _choose_defect_index(
    tile: np.ndarray,
    run_len: int,
    palette_len: int,
    rng: random.Random,
) -> int | None:
    total = int(tile.size)
    max_offset = total - run_len * palette_len
    if max_offset < 0:
        return None

    allowed = list(range(max_offset + 1))
    fg_indices = [idx for idx in allowed if int(tile.flat[idx]) != int(BG)]
    bg_indices = [idx for idx in allowed if int(tile.flat[idx]) == int(BG)]
    if not fg_indices or not bg_indices:
        return None

    prefer_fg = rng.random() < 0.5
    pool = fg_indices if prefer_fg else bg_indices
    return int(rng.choice(pool))


def _build_input_from_meta(
    tile: np.ndarray,
    defect_index: int,
    defect_tile_idx: int,
    palette: list[int],
    run_len: int,
) -> np.ndarray:
    tile_h, tile_w = tile.shape
    grid = np.full((2 * tile_h, 2 * tile_w + 1), BG, dtype=int)

    positions = [
        (0, 0),
        (0, tile_w),
        (tile_h, 0),
        (tile_h, tile_w),
    ]
    defect_r, defect_c = divmod(int(defect_index), tile_w)

    for idx, (r0, c0) in enumerate(positions):
        patch = tile.copy()
        if idx == defect_tile_idx:
            patch[defect_r, defect_c] = int(DEFECT)
        paste(grid, patch, r0, c0)

    grid[:run_len, -1] = int(RUN_MARK)
    for i, color in enumerate(palette):
        grid[run_len + i, -1] = int(color)
    return grid


def _oracle_output(tile_shape: tuple[int, int], palette: list[int], run_len: int, defect_index: int) -> np.ndarray:
    return _cyclic_recolor(tile_shape, palette, run_len, defect_index)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile_h = rng.randint(*TILE_H_RANGE)
    tile_w = rng.randint(*TILE_W_RANGE)
    run_len = rng.choice(RUN_LEN_OPTIONS)
    palette_len = rng.choice(PALETTE_LEN_OPTIONS)

    base = _make_base_tile(tile_h, tile_w, palette_len, rng)
    if base is None:
        return None
    tile, palette = base

    defect_index = _choose_defect_index(tile, run_len, palette_len, rng)
    if defect_index is None:
        return None

    defect_tile_idx = rng.randrange(4)
    grid = _build_input_from_meta(tile, defect_index, defect_tile_idx, palette, run_len)
    expected = _oracle_output(tile.shape, palette, run_len, defect_index)

    if int(get_bg(grid)) != int(BG):
        return None
    if np.count_nonzero(grid[:, :-1] == int(DEFECT)) != 1:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
