from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 7, 8)
WIDTH_CHOICES = (5, 6, 7)
MAX_GENERATION_ATTEMPTS = 500

TILE_MASKS: dict[int, tuple[np.ndarray, ...]] = {
    2: (
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ),
    3: (
        np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
    ),
}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + h)
    cc1 = min(dst.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    patch_view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = patch_view != int(transparent)
    dst_view = dst[rr0:rr1, cc0:cc1]
    dst_view[mask] = patch_view[mask]


def _tile_centers(width: int, tile_w: int, gap_col: int) -> list[int]:
    return list(range(int(gap_col) % int(tile_w), int(width), int(tile_w)))


def _extract_parts(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and 5 <= w <= 7, "expected a small palette row over a taller body"

    palette = g[0].copy()
    gap_cols = np.flatnonzero(palette == BLACK)
    assert gap_cols.size == 1, "palette row must contain exactly one black gap"
    gap_col = int(gap_cols[0])

    palette_non_black = [int(v) for v in palette if int(v) != BLACK]
    assert len(palette_non_black) == w - 1
    assert len(set(palette_non_black)) == len(palette_non_black), "palette colors must be unique"
    assert set(palette_non_black).issubset(PALETTE_COLORS), "palette colors must come from the allowed set"

    body = g[1:, :]
    assert set(map(int, np.unique(body))).issubset({BLACK, GREY}), "only the grey tile may appear below the palette row"
    comps = _components4(body == GREY)
    assert len(comps) == 1, "expected exactly one grey tile component"

    r0, c0, r1, c1 = _bbox(comps[0])
    tile = body[r0 : r1 + 1, c0 : c1 + 1].copy()
    tile_h, tile_w = map(int, tile.shape)
    assert tile_h == 2 and tile_w in (2, 3), "tile bbox must be 2x2 or 2x3"
    assert np.any(tile == GREY) and np.any(tile == BLACK), "tile must contain both grey and black cells"
    assert len(_components4(tile == GREY)) == 1, "tile shape must stay 4-connected"
    return palette, gap_col, tile


def _render_output(palette: np.ndarray, gap_col: int, tile: np.ndarray, height: int) -> np.ndarray:
    width = int(palette.shape[0])
    tile_h, tile_w = map(int, tile.shape)
    center_offset = (tile_w - 1) // 2

    out = np.zeros((int(height), width), dtype=int)
    out[0, :] = palette
    centers = _tile_centers(width, tile_w, gap_col)
    assert gap_col in centers

    for center_col in centers:
        paint_color = GREY if center_col == gap_col else int(palette[center_col])
        patch = tile.copy()
        if paint_color != GREY:
            patch[patch == GREY] = paint_color
        left = int(center_col) - center_offset
        for top in range(1, int(height), tile_h):
            _paste_clip(out, patch, top, left, transparent=BLACK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a tiled field below the palette row, centered on the row's single gap, and recolor each tile column by the palette cell at that column's anchor.

    Input structure:
    - Row 0 is a palette row with exactly one black gap; every other cell is a distinct non-black palette color.
    - Below the palette row, the only foreground is one grey tile pattern whose tight bounding box is 2x2 or 2x3 and contains both grey and black cells.
    - The lower tile is the prototype to repeat; its own position is irrelevant once its shape is extracted.

    Rule:
    - Extract the palette row, locate its unique gap column, and crop the grey tile from its tight bounding box.
    - Starting at row 1, repeat that tile downward and place tile columns so that their anchor columns are every tile-width apart, aligned to the gap column.
    - For each tile column, read the palette color at that column's anchor: recolor all grey tile cells to that color, except the column anchored at the gap itself, which stays grey.
    - Keep the palette row unchanged and leave uncovered cells black; edge tiles may be clipped by the grid boundary.

    Key insight:
    - The top-row gap has two linked meanings at once: it fixes the horizontal phase of the whole tile array and also marks the one anchor column whose tiles remain un-recolored grey.
    """
    palette, gap_col, tile = _extract_parts(np.asarray(grid, dtype=int))
    return _render_output(palette, gap_col, tile, int(grid.shape[0]))


def _gap_options(width: int, tile_w: int) -> list[dict]:
    center_offset = (tile_w - 1) // 2
    options: list[dict] = []
    for gap_col in range(width):
        centers = _tile_centers(width, tile_w, gap_col)
        if len(centers) < 2 or gap_col not in centers:
            continue
        partial_edge = any(
            center - center_offset < 0 or center - center_offset + tile_w > width
            for center in centers
        )
        has_left = any(center < gap_col for center in centers)
        has_right = any(center > gap_col for center in centers)
        if has_left and has_right:
            bias = "middle"
        elif has_right:
            bias = "leftish"
        else:
            bias = "rightish"
        options.append(
            {
                "gap_col": int(gap_col),
                "centers": centers,
                "partial_edge": bool(partial_edge),
                "bias": bias,
            }
        )
    return options


def _choose_geometry() -> tuple[int, int, int, bool]:
    want_partial_edge = random.choice((False, True))
    want_bottom_clip = random.choice((False, True))
    want_bias = random.choice(("leftish", "middle", "rightish"))

    candidates: list[tuple[int, int, int]] = []
    fallback: list[tuple[int, int, int]] = []
    for width in WIDTH_CHOICES:
        for tile_w in (2, 3):
            for option in _gap_options(width, tile_w):
                item = (width, tile_w, option["gap_col"])
                if option["partial_edge"] == want_partial_edge:
                    fallback.append(item)
                    if option["bias"] == want_bias:
                        candidates.append(item)
    pool = candidates or fallback
    if not pool:
        raise RuntimeError("failed to find a valid geometry")

    width, tile_w, gap_col = random.choice(pool)
    full_rows = random.randint(5, 9)
    height = 1 + 2 * full_rows + (1 if want_bottom_clip else 0)
    return int(height), int(width), int(tile_w), int(gap_col)


def _build_palette(width: int, gap_col: int) -> np.ndarray:
    colors = random.sample(PALETTE_COLORS, width - 1)
    palette = np.zeros(width, dtype=int)
    color_iter = iter(colors)
    for c in range(width):
        if c == gap_col:
            continue
        palette[c] = int(next(color_iter))
    return palette


def _source_position(height: int, width: int, tile_w: int, gap_col: int, pattern_idx: int) -> tuple[int, int]:
    tile_h = 2
    base_row = height // 2 - 1
    row_shift = -1 if (gap_col + pattern_idx) % 2 == 0 else 1
    src_r = max(1, min(height - tile_h, base_row + row_shift))

    left_col = max(0, min(width - tile_w, width // 2 - tile_w))
    right_col = max(0, min(width - tile_w, width // 2))
    if gap_col < width // 2:
        src_c = right_col
    elif gap_col > width // 2:
        src_c = left_col
    else:
        src_c = right_col if pattern_idx % 2 else left_col
    return int(src_r), int(src_c)


def _swap_palette_cells(grid: np.ndarray, c0: int, c1: int) -> np.ndarray:
    alt = grid.copy()
    alt[0, c0], alt[0, c1] = alt[0, c1], alt[0, c0]
    return alt


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        palette, gap_col, tile = _extract_parts(grid)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[0], palette):
        return False
    if np.count_nonzero(out[1:, :] != BLACK) <= np.count_nonzero(grid[1:, :] != BLACK):
        return False

    tile_w = int(tile.shape[1])
    centers = _tile_centers(int(grid.shape[1]), tile_w, gap_col)
    non_gap_centers = [c for c in centers if c != gap_col]
    if not non_gap_centers:
        return False
    used_center = int(non_gap_centers[0])
    if not np.any(out[1:, gap_col] == GREY):
        return False
    if not np.any(out[1:, used_center] == int(palette[used_center])):
        return False

    swap_gap = next((c for c in range(grid.shape[1]) if c != gap_col), None)
    swap_color = next((c for c in range(grid.shape[1]) if c not in (gap_col, used_center)), None)
    if swap_gap is None or swap_color is None:
        return False

    try:
        moved_gap_out = main(_swap_palette_cells(grid, gap_col, int(swap_gap)))
        moved_color_out = main(_swap_palette_cells(grid, used_center, int(swap_color)))
    except Exception:
        return False

    if np.array_equal(moved_gap_out[1:, :], out[1:, :]):
        return False
    if np.array_equal(moved_color_out[1:, :], out[1:, :]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height, width, tile_w, gap_col = _choose_geometry()
    palette = _build_palette(width, gap_col)

    pattern_idx = random.randrange(len(TILE_MASKS[tile_w]))
    tile_mask = TILE_MASKS[tile_w][pattern_idx]
    tile = np.where(tile_mask == 1, GREY, BLACK).astype(int)

    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = palette

    # Tie the source tile position to output-relevant parameters so distinct inputs
    # do not collapse onto the same output just by moving the prototype tile around.
    src_r, src_c = _source_position(height, width, tile_w, gap_col, pattern_idx)
    _paste_clip(grid, tile, src_r, src_c, transparent=BLACK)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/8/053/6")
