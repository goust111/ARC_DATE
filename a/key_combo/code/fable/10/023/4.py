from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ARC_COLORS = tuple(range(10))
GEN_RETRY_LIMIT = 500


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _rot180(tile: np.ndarray) -> np.ndarray:
    return np.rot90(tile, 2)


def _split_blocks(grid: np.ndarray, block_h: int, block_w: int) -> list[list[np.ndarray]]:
    rows = []
    for r in range(0, grid.shape[0], int(block_h)):
        row = []
        for c in range(0, grid.shape[1], int(block_w)):
            row.append(grid[r : r + block_h, c : c + block_w])
        rows.append(row)
    return rows


def _constant_color(block: np.ndarray) -> int | None:
    color = int(block[0, 0])
    return color if np.all(block == color) else None


def _candidate_layout(grid: np.ndarray, block_h: int, block_w: int) -> dict | None:
    if grid.shape[0] % block_h != 0 or grid.shape[1] % block_w != 0:
        return None

    blocks = _split_blocks(grid, block_h, block_w)
    anchor = np.array(blocks[0][0], dtype=int, copy=True)
    if len(np.unique(anchor)) < 2:
        return None

    rotated = _rot180(anchor)
    if np.array_equal(anchor, rotated):
        return None

    bg_colors = {
        color
        for row in blocks
        for block in row
        for color in [_constant_color(block)]
        if color is not None
    }
    if len(bg_colors) != 1:
        return None
    bg = int(next(iter(bg_colors)))

    out = np.full((len(blocks), len(blocks[0])), BLACK, dtype=int)
    for br, row in enumerate(blocks):
        for bc, block in enumerate(row):
            if np.all(block == bg):
                out[br, bc] = BLACK
            elif np.array_equal(block, anchor):
                out[br, bc] = GREEN
            elif np.array_equal(block, rotated):
                out[br, bc] = RED
            else:
                return None

    if out[0, 0] != GREEN:
        return None
    if not np.any(out == RED):
        return None
    if not np.any(out == BLACK):
        return None

    return {
        "block_h": int(block_h),
        "block_w": int(block_w),
        "bg": bg,
        "anchor": anchor,
        "rotated": rotated,
        "output": out,
    }


def _find_layouts(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        return []

    layouts = []
    for block_h in _divisors(g.shape[0]):
        for block_w in _divisors(g.shape[1]):
            cand = _candidate_layout(g, block_h, block_w)
            if cand is not None:
                layouts.append(cand)
    return layouts


def main(grid: np.ndarray) -> np.ndarray:
    """Compress a tile-expanded block mosaic back into an orientation mask.

    Input structure:
    - The full grid is partitionable into equal-sized rectangular blocks with no drawn separators.
    - The top-left block is the anchor tile in its original pose.
    - Every other block is either a uniform background block, the same tile unchanged, or the tile rotated 180 degrees.

    Rule:
    - Infer the unique block size from the anchor block and the global blockwise consistency.
    - Compare every block against the top-left anchor tile and its 180-degree rotation.
    - Return one output cell per block: green for an unchanged copy, red for a 180-degree copy, and black for an empty block.

    Key insight:
    - The large image is a pixel-to-block expansion of a hidden small mask, and each occupied mask cell carries one extra bit encoded by the tile's orientation relative to the anchor.
    """
    layouts = _find_layouts(np.asarray(grid, dtype=int))
    assert len(layouts) == 1, "expected exactly one valid block decomposition"
    return layouts[0]["output"]


def _render_input(tile: np.ndarray, states: np.ndarray, bg: int) -> np.ndarray:
    tile = np.asarray(tile, dtype=int)
    states = np.asarray(states, dtype=int)
    tile_h, tile_w = tile.shape
    out = np.full((states.shape[0] * tile_h, states.shape[1] * tile_w), int(bg), dtype=int)
    rotated = _rot180(tile)

    for br in range(states.shape[0]):
        for bc in range(states.shape[1]):
            r0 = br * tile_h
            c0 = bc * tile_w
            state = int(states[br, bc])
            if state == BLACK:
                continue
            if state == GREEN:
                out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
            else:
                assert state == RED
                out[r0 : r0 + tile_h, c0 : c0 + tile_w] = rotated
    return out


def _sample_tile(tile_h: int, tile_w: int, bg: int, rng: np.random.Generator) -> np.ndarray:
    non_bg = [c for c in ARC_COLORS if c != bg]
    for _ in range(200):
        use_bg_inside = bool(rng.random() < 0.35)
        extra = 1 if use_bg_inside else 0
        n_non_bg = int(rng.integers(2, min(5 - extra, len(non_bg)) + 1))
        palette = list(rng.choice(non_bg, size=n_non_bg, replace=False).astype(int))
        if use_bg_inside:
            palette.append(int(bg))

        tile = np.array(rng.choice(palette, size=(tile_h, tile_w)), dtype=int)
        rotated = _rot180(tile)
        diff = int(np.count_nonzero(tile != rotated))

        if diff < max(3, (tile_h * tile_w) // 3):
            continue
        if len(np.unique(tile)) < 2:
            continue
        if np.all(tile == tile[0, 0]):
            continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate tile")


def _sample_states(block_rows: int, block_cols: int, rng: np.random.Generator) -> np.ndarray:
    total = block_rows * block_cols
    for _ in range(200):
        flat = np.full(total, BLACK, dtype=int)
        flat[0] = GREEN

        others = np.arange(1, total)
        rng.shuffle(others)
        red_count = int(rng.integers(1, max(2, total // 2) + 1))
        black_count = int(rng.integers(1, max(2, total // 2) + 1))
        green_extra_max = max(0, total - 1 - red_count - black_count)
        green_extra = int(rng.integers(0, green_extra_max + 1))

        need = red_count + black_count + green_extra
        if need > total - 1:
            continue

        red_idx = others[:red_count]
        black_idx = others[red_count : red_count + black_count]
        green_idx = others[red_count + black_count : red_count + black_count + green_extra]
        flat[red_idx] = RED
        flat[black_idx] = BLACK
        flat[green_idx] = GREEN

        remaining = others[need:]
        if remaining.size:
            flat[remaining] = rng.choice([BLACK, GREEN, RED], size=remaining.size, p=[0.34, 0.33, 0.33]).astype(int)

        states = flat.reshape(block_rows, block_cols)
        states[0, 0] = GREEN
        if not np.any(states == RED):
            continue
        if not np.any(states == BLACK):
            continue
        if len(np.unique(states)) < 3:
            continue
        return states
    raise RuntimeError("failed to sample block states")


def _sample_dimensions(rng: np.random.Generator) -> tuple[int, int, int, int]:
    for _ in range(500):
        block_rows = int(rng.integers(2, 6))
        block_cols = int(rng.integers(2, 6))
        tile_h = int(rng.integers(2, min(7, 30 // block_rows + 1)))
        tile_w = int(rng.integers(2, min(7, 30 // block_cols + 1)))
        if block_rows * tile_h <= 30 and block_cols * tile_w <= 30:
            return tile_h, tile_w, block_rows, block_cols
    raise RuntimeError("failed to sample valid dimensions")


def _validate_instance(
    grid: np.ndarray,
    expected_output: np.ndarray,
    tile_h: int,
    tile_w: int,
    states: np.ndarray,
) -> bool:
    layouts = _find_layouts(grid)
    if len(layouts) != 1:
        return False
    layout = layouts[0]
    if layout["block_h"] != tile_h or layout["block_w"] != tile_w:
        return False
    if not np.array_equal(layout["output"], expected_output):
        return False
    if not np.array_equal(expected_output, states):
        return False

    anchor = grid[:tile_h, :tile_w]
    rotated = _rot180(anchor)
    if np.array_equal(anchor, rotated):
        return False

    for br in range(states.shape[0]):
        for bc in range(states.shape[1]):
            block = grid[br * tile_h : (br + 1) * tile_h, bc * tile_w : (bc + 1) * tile_w]
            state = int(states[br, bc])
            if state == BLACK and _constant_color(block) is None:
                return False
            if state == GREEN and not np.array_equal(block, anchor):
                return False
            if state == RED and not np.array_equal(block, rotated):
                return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_RETRY_LIMIT):
        tile_h, tile_w, block_rows, block_cols = _sample_dimensions(rng)
        bg = int(rng.integers(0, 10))
        tile = _sample_tile(tile_h, tile_w, bg, rng)
        states = _sample_states(block_rows, block_cols, rng)
        grid = _render_input(tile, states, bg)

        if max(grid.shape) > 30:
            continue
        if not _validate_instance(grid, states, tile_h, tile_w, states):
            continue
        if not np.array_equal(main(grid), states):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
