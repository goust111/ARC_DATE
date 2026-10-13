from __future__ import annotations

import numpy as np


MASK_COLOR = 9
MAX_GENERATION_ATTEMPTS = 400

TILE_HEIGHTS = (3, 4, 5, 6)
TILE_WIDTHS = (3, 4, 5, 6)
TILE_ROW_COUNTS = (2, 3, 4)
TILE_COL_COUNTS = (2, 3, 4)

PALETTE_SIZE_OPTIONS = (2, 3, 4, 5)
COLOR_POOL = tuple(range(9))
FEATURE_TYPES = ("rect", "hline", "vline", "diag", "anti", "block")


def _candidate_tile_sizes(height: int, width: int) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for tile_h in range(2, height):
        if height % tile_h != 0 or height // tile_h < 2:
            continue
        for tile_w in range(2, width):
            if width % tile_w != 0 or width // tile_w < 2:
                continue
            sizes.append((tile_h, tile_w))
    sizes.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return sizes


def _extract_template(
    grid: np.ndarray,
    tile_h: int,
    tile_w: int,
    mask_color: int | None,
    require_masked_tiles: bool,
) -> np.ndarray | None:
    height, width = grid.shape
    tile_rows = height // tile_h
    tile_cols = width // tile_w

    template = np.full((tile_h, tile_w), -1, dtype=int)
    masked_counts = np.zeros((tile_rows, tile_cols), dtype=int)
    visible_counts = np.zeros((tile_rows, tile_cols), dtype=int)

    for r in range(tile_h):
        for c in range(tile_w):
            seen: int | None = None
            for tr in range(tile_rows):
                for tc in range(tile_cols):
                    value = int(grid[tr * tile_h + r, tc * tile_w + c])
                    if mask_color is not None and value == mask_color:
                        masked_counts[tr, tc] += 1
                        continue
                    visible_counts[tr, tc] += 1
                    if seen is None:
                        seen = value
                    elif value != seen:
                        return None
            if seen is None:
                return None
            template[r, c] = seen

    if require_masked_tiles:
        if np.any(masked_counts == 0):
            return None
        if np.any(visible_counts == 0):
            return None

    return template


def _find_partitions(
    grid: np.ndarray,
    mask_color: int | None,
    require_masked_tiles: bool,
) -> list[tuple[tuple[int, int], np.ndarray]]:
    candidates: list[tuple[tuple[int, int], np.ndarray]] = []
    height, width = grid.shape
    for tile_h, tile_w in _candidate_tile_sizes(height, width):
        template = _extract_template(grid, tile_h, tile_w, mask_color, require_masked_tiles)
        if template is not None:
            candidates.append(((tile_h, tile_w), template))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a tiled pattern by cross-referencing the same pixel position across tiles.

    Input structure:
    - The full grid is a tight arrangement of equally sized tiles.
    - Every tile should contain the same internal pattern.
    - Some cells inside each tile are replaced by the maroon mask color.
    - No tile is complete, but every pattern position remains visible in at least one tile.

    Rule:
    - Infer a valid tile partition of the grid.
    - For each relative position inside a tile, gather all non-maroon values seen at that
      position across the tiles.
    - Those visible values must agree; use them to reconstruct the shared tile template.
    - Replace every maroon cell with the recovered template value.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert bool(np.any(g == MASK_COLOR)), "expected maroon masked cells"

    candidates = _find_partitions(g, MASK_COLOR, require_masked_tiles=True)
    assert candidates, "no valid tile partition found"

    (tile_h, tile_w), template = candidates[0]
    tile_rows = g.shape[0] // tile_h
    tile_cols = g.shape[1] // tile_w
    return np.tile(template, (tile_rows, tile_cols))


def _repeat_tile(tile: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    return np.tile(tile, (tile_rows, tile_cols))


def _draw_feature(tile: np.ndarray, rng: np.random.Generator, colors: list[int]) -> None:
    height, width = tile.shape
    color = int(rng.choice(colors))
    kind = str(rng.choice(FEATURE_TYPES))

    if kind == "rect":
        r0 = int(rng.integers(0, height))
        r1 = int(rng.integers(r0 + 1, height + 1))
        c0 = int(rng.integers(0, width))
        c1 = int(rng.integers(c0 + 1, width + 1))
        tile[r0:r1, c0:c1] = color
        return

    if kind == "hline":
        r = int(rng.integers(0, height))
        c0 = int(rng.integers(0, width))
        c1 = int(rng.integers(c0 + 1, width + 1))
        tile[r, c0:c1] = color
        return

    if kind == "vline":
        c = int(rng.integers(0, width))
        r0 = int(rng.integers(0, height))
        r1 = int(rng.integers(r0 + 1, height + 1))
        tile[r0:r1, c] = color
        return

    if kind == "diag":
        length = int(rng.integers(1, min(height, width) + 1))
        r0 = int(rng.integers(0, height - length + 1))
        c0 = int(rng.integers(0, width - length + 1))
        for k in range(length):
            tile[r0 + k, c0 + k] = color
        return

    if kind == "anti":
        length = int(rng.integers(1, min(height, width) + 1))
        r0 = int(rng.integers(0, height - length + 1))
        c0 = int(rng.integers(length - 1, width))
        for k in range(length):
            tile[r0 + k, c0 - k] = color
        return

    block_h = int(rng.integers(1, min(3, height) + 1))
    block_w = int(rng.integers(1, min(3, width) + 1))
    r0 = int(rng.integers(0, height - block_h + 1))
    c0 = int(rng.integers(0, width - block_w + 1))
    tile[r0 : r0 + block_h, c0 : c0 + block_w] = color


def _make_tile(rng: np.random.Generator, tile_h: int, tile_w: int) -> np.ndarray | None:
    palette_size = int(rng.choice(PALETTE_SIZE_OPTIONS))
    palette = [int(c) for c in rng.choice(COLOR_POOL, size=palette_size, replace=False)]
    tile = np.full((tile_h, tile_w), palette[0], dtype=int)

    feature_count = int(rng.integers(3, 7))
    for _ in range(feature_count):
        _draw_feature(tile, rng, palette[1:] if len(palette) > 1 else palette)

    # Add a few single-cell tweaks to break accidental sub-periods.
    tweak_count = int(rng.integers(1, 4))
    for _ in range(tweak_count):
        r = int(rng.integers(0, tile_h))
        c = int(rng.integers(0, tile_w))
        tile[r, c] = int(rng.choice(palette))

    if len(np.unique(tile)) < 2:
        return None
    return tile


def _mask_plan(
    rng: np.random.Generator,
    tile_count: int,
    tile_area: int,
) -> np.ndarray | None:
    hidden = np.zeros((tile_count, tile_area), dtype=bool)

    for pos in range(tile_area):
        hidden_count = int(rng.integers(1, tile_count))
        chosen = rng.choice(tile_count, size=hidden_count, replace=False)
        hidden[chosen, pos] = True

    for tile_idx in range(tile_count):
        if not hidden[tile_idx].any():
            for pos in rng.permutation(tile_area):
                if int(hidden[:, pos].sum()) < tile_count - 1:
                    hidden[tile_idx, int(pos)] = True
                    break
        if hidden[tile_idx].all():
            visible_pos = int(rng.choice(np.flatnonzero(hidden[tile_idx])))
            hidden[tile_idx, visible_pos] = False

    if np.any(hidden.sum(axis=0) == tile_count):
        return None
    if np.any(hidden.sum(axis=1) == 0):
        return None
    if np.any(hidden.sum(axis=1) == tile_area):
        return None
    return hidden


def _apply_masks(
    full_grid: np.ndarray,
    tile_h: int,
    tile_w: int,
    tile_rows: int,
    tile_cols: int,
    hidden: np.ndarray,
) -> np.ndarray:
    out = full_grid.copy()
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            tile_idx = tr * tile_cols + tc
            patch = out[tr * tile_h : (tr + 1) * tile_h, tc * tile_w : (tc + 1) * tile_w]
            flat_indices = np.flatnonzero(hidden[tile_idx])
            rows = flat_indices // tile_w
            cols = flat_indices % tile_w
            patch[rows, cols] = MASK_COLOR
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile_h = int(rng.choice(TILE_HEIGHTS))
    tile_w = int(rng.choice(TILE_WIDTHS))
    tile_rows = int(rng.choice(TILE_ROW_COUNTS))
    tile_cols = int(rng.choice(TILE_COL_COUNTS))

    tile = _make_tile(rng, tile_h, tile_w)
    if tile is None:
        return None

    full_grid = _repeat_tile(tile, tile_rows, tile_cols)
    complete_partitions = _find_partitions(full_grid, None, require_masked_tiles=False)
    if not complete_partitions or complete_partitions[0][0] != (tile_h, tile_w):
        return None

    hidden = _mask_plan(rng, tile_rows * tile_cols, tile_h * tile_w)
    if hidden is None:
        return None

    inp = _apply_masks(full_grid, tile_h, tile_w, tile_rows, tile_cols, hidden)
    if not np.array_equal(main(inp), full_grid):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled-mask reconstruction input")
