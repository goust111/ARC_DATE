from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
PALETTE = (1, 4, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ROTATE_TO_UP = {"up": 0, "right": 1, "down": 2, "left": 3}

MIN_TILES = 2
MAX_TILES = 4
MIN_TILE_SIZE = 5
GEN_RETRIES = 400


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    vals = [int(x) for x in indices]
    assert vals, "expected at least one non-separator band"
    out: list[tuple[int, int]] = []
    start = prev = vals[0]
    for x in vals[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return out


def _arrow_cells_up(tip_r: int, tip_c: int) -> set[tuple[int, int]]:
    return {
        (int(tip_r), int(tip_c)),
        (int(tip_r) + 1, int(tip_c) - 1),
        (int(tip_r) + 1, int(tip_c) + 1),
    }


def _arrow_clearance(size: int, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    zone = set(cells)
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < size and 0 <= nc < size:
                zone.add((nr, nc))
    return zone


def _tip_candidates(size: int) -> list[tuple[int, int]]:
    rows = sorted({1, size // 2, size - 2})
    cols = sorted({1, size // 2, size - 2})
    out: list[tuple[int, int]] = []
    for r in rows:
        for c in cols:
            if 0 <= r < size - 1 and 1 <= c < size - 1:
                out.append((int(r), int(c)))
    assert out
    return out


def _arrow_direction(tile: np.ndarray) -> str:
    coords = [tuple(map(int, xy)) for xy in np.argwhere(tile == GREEN)]
    assert len(coords) == 3, "each tile must contain exactly one 3-pixel green arrow"
    cells = set(coords)

    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 2 and len(cols) == 3:
        row_counts = {r: sum(rr == r for rr, _ in coords) for r in rows}
        tip_r = next(r for r, count in row_counts.items() if count == 1)
        tip_c = next(c for r, c in coords if r == tip_r)
        if tip_r == rows[0]:
            assert cells == _arrow_cells_up(tip_r, tip_c)
            return "up"
        expected = {
            (tip_r, tip_c),
            (tip_r - 1, tip_c - 1),
            (tip_r - 1, tip_c + 1),
        }
        assert cells == expected
        return "down"

    if len(rows) == 3 and len(cols) == 2:
        col_counts = {c: sum(cc == c for _, cc in coords) for c in cols}
        tip_c = next(c for c, count in col_counts.items() if count == 1)
        tip_r = next(r for r, c in coords if c == tip_c)
        if tip_c == cols[0]:
            expected = {
                (tip_r, tip_c),
                (tip_r - 1, tip_c + 1),
                (tip_r + 1, tip_c + 1),
            }
            assert cells == expected
            return "left"
        expected = {
            (tip_r, tip_c),
            (tip_r - 1, tip_c - 1),
            (tip_r + 1, tip_c - 1),
        }
        assert cells == expected
        return "right"

    raise AssertionError("green pixels do not form a valid arrow")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each arrow-marked tile back to a shared upright orientation, erase the arrows, and stitch the tiles together.

    Input structure:
    - The background is black.
    - Single-pixel red full rows and full columns partition the grid into equal-sized square tiles.
    - Each tile contains a multicolor sprite fragment plus exactly one green 3-pixel V-shaped arrow.

    Rule:
    - Read each tile independently and determine the green arrow direction from its tip-and-tail geometry.
    - Rotate the whole tile until that arrow would point upward, using the unique already-up arrow tile as the reference orientation.
    - Remove all green arrow pixels, discard the red separator lines, and concatenate the corrected tile interiors in their original row/column order.

    Key insight:
    - Red lines only define tile boundaries, while the green V marker encodes the inverse rotation needed to restore each tile before assembly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    row_bands = _bands(np.where(~np.all(g == RED, axis=1))[0])
    col_bands = _bands(np.where(~np.all(g == RED, axis=0))[0])
    assert len(row_bands) >= 2 and len(col_bands) >= 2, "expected a multi-tile grid"

    heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all tiles must have the same size"
    tile_h = heights.pop()
    tile_w = widths.pop()
    assert tile_h == tile_w, "90-degree tile rotation requires square tiles"

    tiles: list[np.ndarray] = []
    up_count = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            direction = _arrow_direction(tile)
            up_count += int(direction == "up")
            corrected = np.rot90(tile, ROTATE_TO_UP[direction]).copy()
            corrected[corrected == GREEN] = BG
            tiles.append(corrected)

    assert up_count == 1, "exactly one tile should already point upward"
    out = np.zeros((len(row_bands) * tile_h, len(col_bands) * tile_w), dtype=int)
    idx = 0
    for tr in range(len(row_bands)):
        for tc in range(len(col_bands)):
            r0 = tr * tile_h
            c0 = tc * tile_w
            out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tiles[idx]
            idx += 1
    return out


def _axis_layout(count: int, tile_size: int, with_border: bool) -> tuple[int, list[int], list[int]]:
    starts: list[int] = []
    lines: list[int] = []
    cursor = 1 if with_border else 0
    total = count * tile_size + (count - 1) + 2 * int(with_border)
    if with_border:
        lines.append(0)
    for i in range(count):
        starts.append(int(cursor))
        cursor += tile_size
        if i < count - 1 or with_border:
            lines.append(int(cursor))
            cursor += 1
    assert cursor == total
    return total, starts, lines


def _choose_layout() -> tuple[int, int, int, bool, bool] | None:
    n_rows = random.randint(MIN_TILES, MAX_TILES)
    n_cols = random.randint(MIN_TILES, MAX_TILES)
    border_rows = bool(random.getrandbits(1))
    border_cols = bool(random.getrandbits(1))
    line_rows = (n_rows - 1) + 2 * int(border_rows)
    line_cols = (n_cols - 1) + 2 * int(border_cols)
    max_tile = min((30 - line_rows) // n_rows, (30 - line_cols) // n_cols)
    if max_tile < MIN_TILE_SIZE:
        return None
    tile_size = random.randint(MIN_TILE_SIZE, max_tile)
    return n_rows, n_cols, tile_size, border_rows, border_cols


def _sample_line_colors(n_rows: int, n_cols: int) -> tuple[list[int], list[int]]:
    need = n_rows + n_cols
    if need <= len(PALETTE):
        picked = random.sample(PALETTE, need)
    else:
        picked = list(random.sample(PALETTE, len(PALETTE)))
        picked.append(random.choice(PALETTE))
        random.shuffle(picked)
    return picked[:n_rows], picked[n_rows : n_rows + n_cols]


def _paint_rows(canvas: np.ndarray, reserved: np.ndarray, tile_size: int, colors: list[int]) -> None:
    max_thickness = 2 if tile_size >= 6 else 1
    for tr, color in enumerate(colors):
        thickness = random.randint(1, max_thickness)
        local_r = random.randint(0, tile_size - thickness)
        r0 = tr * tile_size + local_r
        region = canvas[r0 : r0 + thickness]
        region[~reserved[r0 : r0 + thickness]] = int(color)


def _paint_cols(canvas: np.ndarray, reserved: np.ndarray, tile_size: int, colors: list[int]) -> None:
    max_thickness = 2 if tile_size >= 6 else 1
    for tc, color in enumerate(colors):
        thickness = random.randint(1, max_thickness)
        local_c = random.randint(0, tile_size - thickness)
        c0 = tc * tile_size + local_c
        region = canvas[:, c0 : c0 + thickness]
        region[~reserved[:, c0 : c0 + thickness]] = int(color)


def _paint_rectangle(
    canvas: np.ndarray,
    reserved: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    color: int,
    filled: bool,
) -> None:
    region = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = np.zeros((h, w), dtype=bool)
    if filled or h < 3 or w < 3:
        mask[:] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    region[mask & ~reserved[r0 : r0 + h, c0 : c0 + w]] = int(color)


def _build_reserved(
    n_rows: int, n_cols: int, tile_size: int
) -> tuple[np.ndarray, list[list[set[tuple[int, int]]]]]:
    height = n_rows * tile_size
    width = n_cols * tile_size
    reserved = np.zeros((height, width), dtype=bool)
    arrows: list[list[set[tuple[int, int]]]] = []
    for tr in range(n_rows):
        row_info: list[set[tuple[int, int]]] = []
        for tc in range(n_cols):
            tip_r, tip_c = random.choice(_tip_candidates(tile_size))
            local_arrow = _arrow_cells_up(tip_r, tip_c)
            for rr, cc in _arrow_clearance(tile_size, local_arrow):
                reserved[tr * tile_size + rr, tc * tile_size + cc] = True
            row_info.append(local_arrow)
        arrows.append(row_info)
    return reserved, arrows


def _build_output(
    n_rows: int, n_cols: int, tile_size: int, reserved: np.ndarray
) -> np.ndarray:
    height = n_rows * tile_size
    width = n_cols * tile_size
    canvas = np.zeros((height, width), dtype=int)

    row_colors, col_colors = _sample_line_colors(n_rows, n_cols)
    _paint_rows(canvas, reserved, tile_size, row_colors)
    _paint_cols(canvas, reserved, tile_size, col_colors)

    extra_shapes = random.randint(1, 3)
    max_h = max(2, min(height, tile_size * 2))
    max_w = max(2, min(width, tile_size * 2))
    for _ in range(extra_shapes):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        r0 = random.randint(0, height - h)
        c0 = random.randint(0, width - w)
        color = random.choice(PALETTE)
        filled = random.random() < 0.65
        _paint_rectangle(canvas, reserved, r0, c0, h, w, color, filled)

    return canvas


def _tile_is_useful(tile: np.ndarray) -> bool:
    nonzero = int(np.count_nonzero(tile))
    if nonzero < tile.shape[0]:
        return False
    colors = {int(x) for x in np.unique(tile) if int(x) != BG}
    return bool(colors)


def _assemble_input(
    output: np.ndarray,
    arrows: list[list[set[tuple[int, int]]]],
    tile_size: int,
    border_rows: bool,
    border_cols: bool,
) -> np.ndarray:
    n_rows = len(arrows)
    n_cols = len(arrows[0])
    in_h, row_starts, row_lines = _axis_layout(n_rows, tile_size, border_rows)
    in_w, col_starts, col_lines = _axis_layout(n_cols, tile_size, border_cols)
    grid = np.zeros((in_h, in_w), dtype=int)
    grid[row_lines, :] = RED
    grid[:, col_lines] = RED

    base_index = random.randrange(n_rows * n_cols)
    idx = 0
    for tr in range(n_rows):
        for tc in range(n_cols):
            tile = output[
                tr * tile_size : (tr + 1) * tile_size,
                tc * tile_size : (tc + 1) * tile_size,
            ].copy()
            arrow_cells = arrows[tr][tc]
            for rr, cc in arrow_cells:
                tile[int(rr), int(cc)] = GREEN

            k_cw = 0 if idx == base_index else random.randint(1, 3)
            scrambled = np.rot90(tile, -k_cw).copy()
            r0 = row_starts[tr]
            c0 = col_starts[tc]
            grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = scrambled
            idx += 1

    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None
    n_rows, n_cols, tile_size, border_rows, border_cols = layout

    reserved, arrows = _build_reserved(n_rows, n_cols, tile_size)
    output = _build_output(n_rows, n_cols, tile_size, reserved)

    if len({int(x) for x in np.unique(output) if int(x) != BG}) < 2:
        return None

    for tr in range(n_rows):
        for tc in range(n_cols):
            tile = output[
                tr * tile_size : (tr + 1) * tile_size,
                tc * tile_size : (tc + 1) * tile_size,
            ]
            if not _tile_is_useful(tile):
                return None
            if any(int(tile[rr, cc]) != BG for rr, cc in arrows[tr][tc]):
                return None

    inp = _assemble_input(output, arrows, tile_size, border_rows, border_cols)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
