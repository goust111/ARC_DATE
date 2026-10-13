import random
from collections import Counter

import numpy as np


BG = 0
MARKER_COLOR = 7
PALETTE_COLORS = tuple(range(1, 10))
TILE_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_vertical_bar(grid: np.ndarray, bg: int, cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    if c0 != c1:
        return False
    if len(cells) != r1 - r0 + 1:
        return False
    colors = {int(grid[r, c]) for r, c in cells}
    return len(colors) == 1 and next(iter(colors)) != bg


def _find_repeated_tile(block: np.ndarray) -> np.ndarray:
    h, w = block.shape
    candidates: list[np.ndarray] = []
    if w % 2 == 0:
        left = block[:, : w // 2]
        right = block[:, w // 2 :]
        if np.array_equal(left, right):
            candidates.append(left.copy())
    if h % 2 == 0:
        top = block[: h // 2, :]
        bottom = block[h // 2 :, :]
        if np.array_equal(top, bottom):
            candidates.append(top.copy())
    assert len(candidates) == 1, "bottom pattern must have exactly one repeated axis"
    return candidates[0]


def _tile_canvas(tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    th, tw = tile.shape
    rr = np.arange(out_h)[:, None] % th
    cc = np.arange(out_w)[None, :] % tw
    return tile[rr, cc]


def _parse_input(grid: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    h, w = g.shape
    assert h >= 5 and w >= 6

    top = g[0]
    gap_cols = np.flatnonzero(top == bg)
    assert gap_cols.size == 1, "expected exactly one palette gap"
    gap_col = int(gap_cols[0])
    assert np.count_nonzero(top != bg) == w - 1

    lower_mask = g[1:] != bg
    components = [
        [(r + 1, c) for r, c in comp]
        for comp in _components4(lower_mask)
    ]
    assert len(components) >= 2, "expected marker bar and repeated block"

    bar_candidates = [comp for comp in components if _is_vertical_bar(g, bg, comp)]
    assert len(bar_candidates) == 1, "expected one vertical marker bar"
    bar = bar_candidates[0]
    marker_color = int(g[bar[0][0], bar[0][1]])
    marker_height = len(bar)
    marker_col = bar[0][1]
    assert abs(marker_col - gap_col) <= 1

    pattern_components = [comp for comp in components if comp is not bar]
    assert len(pattern_components) == 1, "expected one repeated pattern component"
    pattern = pattern_components[0]
    r0, c0, r1, c1 = _bbox(pattern)
    block = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(block != bg), "repeated block must be a solid rectangle"
    tile = _find_repeated_tile(block)
    return gap_col, marker_color, marker_height, tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the repeated bottom tile, spread it across a canvas, and reveal it only through a wedge centered on the palette gap.

    Input structure:
    - The top row is a palette row with exactly one background gap.
    - Below it sits one vertical marker bar and one separated rectangular repeated-tile block made of two identical halves.
    - The marker bar is in the gap column or an orthogonally adjacent column, and the repeated block is duplicated either horizontally or vertically.

    Rule:
    - Find the unique repeated axis in the bottom block and keep the first half as the source tile.
    - Read the palette gap column as the wedge center and the marker-bar height as the output height.
    - Tile the extracted tile from the output's top-left corner across a canvas of marker_height x input_width, then erase everything outside the centered wedge whose radius grows by one cell per row.
    - Overwrite the entire gap column inside the output with the marker-bar color.

    Key insight:
    - The bottom rectangle provides the content pattern, while the palette gap plus marker bar define an independent visibility mask; the answer is their overlay, not either source alone.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    gap_col, marker_color, out_h, tile = _parse_input(g)
    out_w = g.shape[1]

    tiled = _tile_canvas(tile, out_h, out_w)
    out = np.full((out_h, out_w), bg, dtype=int)
    for r in range(out_h):
        c0 = max(0, gap_col - r)
        c1 = min(out_w - 1, gap_col + r)
        out[r, c0 : c1 + 1] = tiled[r, c0 : c1 + 1]
    out[:, gap_col] = marker_color
    return out


def _has_equal_halves(tile: np.ndarray) -> bool:
    h, w = tile.shape
    if w % 2 == 0 and np.array_equal(tile[:, : w // 2], tile[:, w // 2 :]):
        return True
    if h % 2 == 0 and np.array_equal(tile[: h // 2, :], tile[h // 2 :, :]):
        return True
    return False


def _make_tile(rng: random.Random, tile_h: int, tile_w: int) -> np.ndarray:
    area = tile_h * tile_w
    max_colors = min(5, area, len(TILE_COLORS))
    min_colors = min(3, max_colors)
    for _ in range(80):
        color_count = rng.randint(min_colors, max_colors)
        colors = rng.sample(TILE_COLORS, color_count)
        flat = colors[:]
        flat.extend(rng.choice(colors) for _ in range(area - color_count))
        rng.shuffle(flat)
        tile = np.array(flat, dtype=int).reshape(tile_h, tile_w)
        if not _has_equal_halves(tile):
            return tile
    raise RuntimeError("failed to create a non-degenerate tile")


def _pick_gap_col(rng: random.Random, width: int) -> int:
    center_cols = [c for c in range(width) if 2 <= c <= width - 3]
    edge_cols = sorted(set([0, 1, width - 2, width - 1]))
    if center_cols and rng.random() < 0.55:
        return rng.choice(center_cols)
    return rng.choice(edge_cols)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.randint(8, 10)
    max_height = min(5, width // 2)
    marker_height = rng.randint(3, max_height)
    gap_col = _pick_gap_col(rng, width)

    valid_offsets = [d for d in (-1, 0, 1) if 0 <= gap_col + d < width]
    marker_col = gap_col + rng.choice(valid_offsets)

    axis = rng.choice(("horizontal", "vertical"))
    tile_h = rng.randint(3, 4)
    if axis == "horizontal":
        tile_w = rng.randint(3, min(4, width // 2))
    else:
        tile_w = rng.randint(3, min(4, width))
    tile = _make_tile(rng, tile_h, tile_w)
    repeated = np.concatenate([tile, tile], axis=1 if axis == "horizontal" else 0)

    left = rng.randint(0, width - repeated.shape[1])
    spacer = rng.randint(1, 2)
    repeated_top = marker_height + spacer + 1
    grid_h = repeated_top + repeated.shape[0]
    if grid_h > 18:
        return None

    grid = np.full((grid_h, width), BG, dtype=int)

    palette = rng.sample(PALETTE_COLORS, width - 1)
    palette.insert(gap_col, BG)
    grid[0] = np.array(palette, dtype=int)
    grid[1 : marker_height + 1, marker_col] = MARKER_COLOR
    grid[
        repeated_top : repeated_top + repeated.shape[0],
        left : left + repeated.shape[1],
    ] = repeated

    try:
        out = main(grid)
    except AssertionError:
        return None

    visible_tile_cells = np.count_nonzero((out != BG) & (out != MARKER_COLOR))
    if visible_tile_cells < max(3, marker_height):
        return None
    if len(set(np.unique(out)) - {BG}) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wedge-tiling input")
