import random

import numpy as np


ARC_GRID_MAX = 30
MAX_NOISE_PER_COPY = 3
MAX_GENERATION_ATTEMPTS = 500
PALETTE = tuple(range(10))
MOTIF_KINDS = ("point", "rect", "hline", "vline", "el")


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _extract_blocks(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    n_rows = grid.shape[0] // tile_h
    n_cols = grid.shape[1] // tile_w
    return (
        grid.reshape(n_rows, tile_h, n_cols, tile_w)
        .transpose(0, 2, 1, 3)
        .reshape(n_rows * n_cols, tile_h, tile_w)
    )


def _is_primitive_tile(tile: np.ndarray) -> bool:
    tile_h, tile_w = tile.shape
    for sub_h in _divisors(tile_h):
        for sub_w in _divisors(tile_w):
            if sub_h == tile_h and sub_w == tile_w:
                continue
            base = tile[:sub_h, :sub_w]
            if np.array_equal(tile, np.tile(base, (tile_h // sub_h, tile_w // sub_w))):
                return False
    return True


def _reconstruct_from_candidate(blocks: np.ndarray, noise_color: int):
    n_blocks, tile_h, tile_w = blocks.shape
    tile = np.empty((tile_h, tile_w), dtype=int)

    for r in range(tile_h):
        for c in range(tile_w):
            vals = blocks[:, r, c]
            visible = vals[vals != noise_color]
            if visible.size == 0:
                return None
            colors, counts = np.unique(visible, return_counts=True)
            best = counts.max()
            if np.count_nonzero(counts == best) != 1:
                return None
            tile[r, c] = int(colors[counts.argmax()])

    mismatches = blocks != tile[None, :, :]
    if np.any(blocks[mismatches] != noise_color):
        return None

    per_block = mismatches.reshape(n_blocks, -1).sum(axis=1)
    if np.any(per_block > MAX_NOISE_PER_COPY):
        return None

    total_noise = int(mismatches.sum())
    if total_noise == 0:
        return None

    return tile, total_noise


def _candidate_explanations(grid: np.ndarray) -> list[dict]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    colors = [int(v) for v in np.unique(grid)]
    candidates: list[dict] = []

    for tile_h in _divisors(h):
        for tile_w in _divisors(w):
            n_blocks = (h // tile_h) * (w // tile_w)
            if n_blocks < 2:
                continue
            blocks = _extract_blocks(grid, tile_h, tile_w)
            for noise_color in colors:
                rebuilt = _reconstruct_from_candidate(blocks, noise_color)
                if rebuilt is None:
                    continue
                tile, total_noise = rebuilt
                output = np.tile(tile, (h // tile_h, w // tile_w))
                candidates.append(
                    {
                        "tile_shape": (tile_h, tile_w),
                        "noise_color": noise_color,
                        "tile": tile,
                        "output": output,
                        "score": (tile_h * tile_w, total_noise, tile_h + tile_w),
                    }
                )
    return candidates


def _select_best_explanation(grid: np.ndarray) -> dict:
    candidates = _candidate_explanations(grid)
    if not candidates:
        raise AssertionError("no valid tiling/noise explanation found")

    best_score = min(candidate["score"] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate["score"] == best_score]

    unique_outputs: list[np.ndarray] = []
    for candidate in best:
        if not any(np.array_equal(candidate["output"], seen) for seen in unique_outputs):
            unique_outputs.append(candidate["output"])
    if len(unique_outputs) != 1:
        raise AssertionError("ambiguous minimal explanation")

    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clean repeated tile by cross-validating multiple noise-corrupted copies.

    Input structure:
    - The grid is fully tiled by identical copies of one small rectangular tile.
    - Each copy contains the same background-and-foreground pattern, but 0-3 cells
      may be replaced by one special noise color.
    - The noise color is shared by all copies and never belongs to the clean tile.

    Rule:
    - Search for a tile height and width that exactly partition the grid into copies.
    - For each candidate partition, treat one present color as the possible noise
      color and reconstruct the clean tile by taking, at every tile position, the
      unique most frequent non-noise color across all copies.
    - Keep the smallest valid partition whose reconstructed tile explains every copy
      using only noise-color substitutions, then replace all copies with that clean tile.

    Key insight:
    The true tile is the smallest primitive period whose copies agree everywhere
    except for a sparse color that appears inconsistently from copy to copy.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2
    return _select_best_explanation(grid)["output"]


def _paint_motif(tile: np.ndarray, color: int, kind: str) -> None:
    tile_h, tile_w = tile.shape

    if kind == "point":
        tile[random.randrange(tile_h), random.randrange(tile_w)] = color
        return

    if kind == "rect":
        span_h = random.randint(1, tile_h)
        span_w = random.randint(1, tile_w)
        if tile_h >= 3 and span_h == tile_h:
            span_h -= 1
        if tile_w >= 3 and span_w == tile_w:
            span_w -= 1
        r0 = random.randint(0, tile_h - span_h)
        c0 = random.randint(0, tile_w - span_w)
        tile[r0 : r0 + span_h, c0 : c0 + span_w] = color
        return

    if kind == "hline":
        row = random.randrange(tile_h)
        c0 = random.randrange(tile_w)
        c1 = random.randint(c0, tile_w - 1)
        tile[row, c0 : c1 + 1] = color
        return

    if kind == "vline":
        col = random.randrange(tile_w)
        r0 = random.randrange(tile_h)
        r1 = random.randint(r0, tile_h - 1)
        tile[r0 : r1 + 1, col] = color
        return

    assert kind == "el"
    arm_h = random.randint(2, tile_h)
    arm_w = random.randint(2, tile_w)
    r0 = random.randint(0, tile_h - arm_h)
    c0 = random.randint(0, tile_w - arm_w)
    variant = random.randrange(4)
    if variant == 0:
        tile[r0 : r0 + arm_h, c0] = color
        tile[r0 + arm_h - 1, c0 : c0 + arm_w] = color
    elif variant == 1:
        tile[r0 : r0 + arm_h, c0 + arm_w - 1] = color
        tile[r0 + arm_h - 1, c0 : c0 + arm_w] = color
    elif variant == 2:
        tile[r0 : r0 + arm_h, c0] = color
        tile[r0, c0 : c0 + arm_w] = color
    else:
        tile[r0 : r0 + arm_h, c0 + arm_w - 1] = color
        tile[r0, c0 : c0 + arm_w] = color


def _build_tile(tile_h: int, tile_w: int, bg: int, fg_colors: list[int]):
    area = tile_h * tile_w
    for _ in range(150):
        tile = np.full((tile_h, tile_w), bg, dtype=int)
        motif_count = random.randint(3, 6)
        for _ in range(motif_count):
            kind = random.choice(MOTIF_KINDS)
            color = random.choice(fg_colors)
            _paint_motif(tile, color, kind)

        for color in fg_colors:
            if not np.any(tile == color):
                tile[random.randrange(tile_h), random.randrange(tile_w)] = color

        for _ in range(random.randint(1, 3)):
            tile[random.randrange(tile_h), random.randrange(tile_w)] = random.choice(fg_colors)

        non_bg = int(np.count_nonzero(tile != bg))
        if non_bg < max(2, area // 5):
            continue
        if non_bg > min(area - 2, int(round(area * 0.7))):
            continue
        if len(np.unique(tile)) < 2:
            continue
        if not _is_primitive_tile(tile):
            continue
        return tile
    return None


def _sample_layout():
    dims: list[tuple[int, int, int, int]] = []
    for tile_h in range(2, 9):
        for tile_w in range(2, 9):
            if tile_h * tile_w < 6:
                continue
            max_rows = ARC_GRID_MAX // tile_h
            max_cols = ARC_GRID_MAX // tile_w
            for n_rows in range(2, max_rows + 1):
                for n_cols in range(2, max_cols + 1):
                    if n_rows * n_cols >= 6:
                        dims.append((tile_h, tile_w, n_rows, n_cols))
    return random.choice(dims)


def _sample_noise_masks(tile: np.ndarray, n_blocks: int, bg: int):
    tile_h, tile_w = tile.shape
    cells = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    coverage_limit = (n_blocks - 1) // 2

    for _ in range(200):
        counts = [random.randint(0, MAX_NOISE_PER_COPY) for _ in range(n_blocks)]
        if sum(counts) == 0:
            counts[random.randrange(n_blocks)] = random.randint(1, MAX_NOISE_PER_COPY)
        if all(count > 0 for count in counts):
            counts[random.randrange(n_blocks)] = 0

        masks: list[list[tuple[int, int]]] = []
        coverage = {cell: 0 for cell in cells}
        touched: set[tuple[int, int]] = set()
        for count in counts:
            chosen = random.sample(cells, count)
            masks.append(chosen)
            for cell in chosen:
                coverage[cell] += 1
                touched.add(cell)

        if max(coverage.values(), default=0) > coverage_limit:
            continue
        if len(touched) < min(4, len(cells)):
            continue

        touches_bg = any(tile[r, c] == bg for r, c in touched)
        touches_fg = any(tile[r, c] != bg for r, c in touched)
        if not (touches_bg and touches_fg):
            continue

        return masks
    return None


def _apply_noise(clean: np.ndarray, tile_shape: tuple[int, int], n_cols: int, masks, noise_color: int) -> np.ndarray:
    tile_h, tile_w = tile_shape
    noisy = clean.copy()
    for block_index, positions in enumerate(masks):
        block_r = block_index // n_cols
        block_c = block_index % n_cols
        row0 = block_r * tile_h
        col0 = block_c * tile_w
        for r, c in positions:
            noisy[row0 + r, col0 + c] = noise_color
    return noisy


def _gen_attempt():
    tile_h, tile_w, n_rows, n_cols = _sample_layout()
    fg_count = random.randint(1, 3)
    colors = random.sample(PALETTE, fg_count + 2)
    bg = colors[0]
    noise_color = colors[1]
    fg_colors = colors[2:]

    tile = _build_tile(tile_h, tile_w, bg, fg_colors)
    if tile is None:
        return None

    masks = _sample_noise_masks(tile, n_rows * n_cols, bg)
    if masks is None:
        return None

    clean = np.tile(tile, (n_rows, n_cols))
    noisy = _apply_noise(clean, (tile_h, tile_w), n_cols, masks, noise_color)

    try:
        selected = _select_best_explanation(noisy)
    except AssertionError:
        return None

    if not np.array_equal(selected["output"], clean):
        return None

    return noisy


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
