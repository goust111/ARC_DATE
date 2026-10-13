from __future__ import annotations

import random

import numpy as np


GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 800


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _seed_col(global_c: int, tile_w: int) -> int:
    tile_col = int(global_c) // int(tile_w)
    local_c = int(global_c) % int(tile_w)
    return local_c if tile_col % 2 == 0 else int(tile_w) - 1 - local_c


def _merge_mirrored(
    g: np.ndarray,
    tile_h: int,
    tile_w: int,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    counts = np.zeros((tile_h, tile_w), dtype=int)
    parity_seen = np.zeros((tile_h, tile_w, 2), dtype=bool)

    for r, c in np.argwhere(g != GREY):
        r = int(r)
        c = int(c)
        rr = r % tile_h
        cc = _seed_col(c, tile_w)
        color = int(g[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = color
        elif int(tile[rr, cc]) != color:
            return None, counts, parity_seen
        counts[rr, cc] += 1
        parity_seen[rr, cc, (c // tile_w) % 2] = True

    return tile, counts, parity_seen


def _translation_conflict_keys(g: np.ndarray, tile_h: int, tile_w: int) -> set[tuple[int, int]]:
    seen: dict[tuple[int, int], int] = {}
    conflicts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g != GREY):
        r = int(r)
        c = int(c)
        key = (r % tile_h, c % tile_w)
        color = int(g[r, c])
        if key in seen and seen[key] != color:
            conflicts.add(key)
        else:
            seen.setdefault(key, color)
    return conflicts


def _candidate_tiles(g: np.ndarray) -> list[tuple[int, int, int, np.ndarray, int, int]]:
    rows, cols = g.shape
    candidates: list[tuple[int, int, int, np.ndarray, int, int]] = []

    for tile_h in _divisors(rows):
        if tile_h < 2 or rows // tile_h < 2:
            continue
        for tile_w in _divisors(cols):
            if tile_w < 2 or cols // tile_w < 3:
                continue

            conflict_count = len(_translation_conflict_keys(g, tile_h, tile_w))
            if conflict_count == 0:
                continue

            tile, counts, parity_seen = _merge_mirrored(g, tile_h, tile_w)
            if tile is None or bool((tile < 0).any()):
                continue
            if bool((counts == 0).any()):
                continue

            both_parities = int((parity_seen[:, :, 0] & parity_seen[:, :, 1]).sum())
            candidates.append((tile_h * tile_w, tile_h, tile_w, tile, conflict_count, both_parities))

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the canonical seed tile from a grey-occluded alternating mirror tiling.

    Input structure:
    - The whole grid is tiled by one hidden seed tile with no gaps or separators.
    - Tile columns alternate orientation: the leftmost tile column is original, the next
      is horizontally mirrored, and so on.
    - Solid grey rectangles cover parts of the tiling; grey is not a seed color, and no
      tile instance remains completely visible.

    Rule:
    - Search for the fundamental tile height and width whose mirrored congruence classes
      make all non-grey evidence agree.
    - For each seed coordinate, merge all visible cells with the same tile row and the
      same mirror-normalized tile column.
    - Return the complete seed in the same orientation as the upper-left tile instance.

    Key insight:
    - The contradiction under plain translation is the clue: odd tile columns must be
      unmirrored before their fragments can be combined with even tile columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= 30, "ARC grids are at most 30x30"
    assert np.any(g == GREY), "expected grey occluders"

    candidates = _candidate_tiles(g)
    assert candidates, "no unique mirrored tiling reconstruction found"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    min_area = candidates[0][0]
    best = [item for item in candidates if item[0] == min_area]
    assert len(best) == 1, "ambiguous fundamental tile"
    return best[0][3].copy()


def _has_small_period(seed: np.ndarray) -> bool:
    h, w = seed.shape
    for rh in _divisors(h):
        if rh < h and np.array_equal(seed, np.tile(seed[:rh, :], (h // rh, 1))):
            return True
    for cw in _divisors(w):
        if cw < w and np.array_equal(seed, np.tile(seed[:, :cw], (1, w // cw))):
            return True
    return False


def _seed_is_usable(seed: np.ndarray) -> bool:
    if int(np.unique(seed).size) < 3:
        return False
    if np.array_equal(seed, np.fliplr(seed)):
        return False
    if _has_small_period(seed):
        return False

    h, w = seed.shape
    asymmetric_pairs = 0
    for r in range(h):
        for c in range(w):
            if int(seed[r, c]) != int(seed[r, w - 1 - c]):
                asymmetric_pairs += 1
    return asymmetric_pairs >= max(4, seed.size // 2)


def _make_seed(tile_h: int, tile_w: int) -> np.ndarray:
    for _ in range(200):
        color_count = random.randint(3, min(5, len(PALETTE)))
        palette = random.sample(list(PALETTE), color_count)
        seed = np.array(
            [[random.choice(palette) for _ in range(tile_w)] for _ in range(tile_h)],
            dtype=int,
        )
        if _seed_is_usable(seed):
            return seed
    raise RuntimeError("failed to build a non-degenerate seed")


def _render_tiling(seed: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    tile_h, tile_w = seed.shape
    grid = np.zeros((tile_rows * tile_h, tile_cols * tile_w), dtype=int)
    mirrored = np.fliplr(seed)
    for tr in range(tile_rows):
        r0 = tr * tile_h
        for tc in range(tile_cols):
            c0 = tc * tile_w
            block = seed if tc % 2 == 0 else mirrored
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = block
    return grid


def _interior_start(limit: int, length: int) -> int:
    if limit - length >= 2:
        return random.randint(1, limit - length - 1)
    return random.randint(0, limit - length)


def _occlude_every_tile(grid: np.ndarray, tile_h: int, tile_w: int, tile_rows: int, tile_cols: int) -> np.ndarray:
    out = grid.copy()
    size_options = [(1, 1), (1, 2), (2, 1)]
    if tile_h >= 5 and tile_w >= 5:
        size_options.append((2, 2))
    if tile_h >= 6:
        size_options.append((3, 1))
    if tile_w >= 6:
        size_options.append((1, 3))

    for tr in range(tile_rows):
        for tc in range(tile_cols):
            local_options = [(rh, rw) for rh, rw in size_options if rh < tile_h and rw < tile_w]
            rh, rw = random.choice(local_options)
            lr = _interior_start(tile_h, rh)
            lc = _interior_start(tile_w, rw)
            r0 = tr * tile_h + lr
            c0 = tc * tile_w + lc
            out[r0 : r0 + rh, c0 : c0 + rw] = GREY
    return out


def _no_complete_tiles(inp: np.ndarray, tile_h: int, tile_w: int) -> bool:
    tile_rows = inp.shape[0] // tile_h
    tile_cols = inp.shape[1] // tile_w
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            block = inp[tr * tile_h : (tr + 1) * tile_h, tc * tile_w : (tc + 1) * tile_w]
            grey_count = int(np.count_nonzero(block == GREY))
            if grey_count == 0 or grey_count >= block.size:
                return False
    return True


def _valid_generation(inp: np.ndarray, seed: np.ndarray, tile_h: int, tile_w: int) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if bool(np.any(seed == GREY)):
        return False
    if not _no_complete_tiles(inp, tile_h, tile_w):
        return False

    merged, counts, parity_seen = _merge_mirrored(inp, tile_h, tile_w)
    if merged is None or not np.array_equal(merged, seed):
        return False
    if bool((counts == 0).any()):
        return False

    conflict_count = len(_translation_conflict_keys(inp, tile_h, tile_w))
    if conflict_count < max(2, tile_w // 2):
        return False

    both_parities = int((parity_seen[:, :, 0] & parity_seen[:, :, 1]).sum())
    if both_parities < max(3, seed.size // 3):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, seed):
        return False

    candidates = _candidate_tiles(inp)
    if not candidates:
        return False
    min_area = min(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == min_area]
    return len(best) == 1 and best[0][1] == tile_h and best[0][2] == tile_w


def _choose_geometry() -> tuple[int, int, int, int]:
    for _ in range(200):
        tile_h = random.randint(4, 7)
        tile_w = random.randint(4, 7)
        max_tile_rows = 30 // tile_h
        max_tile_cols = 30 // tile_w
        if max_tile_rows < 3 or max_tile_cols < 4:
            continue
        tile_rows = random.randint(3, max_tile_rows)
        tile_cols = random.randint(4, max_tile_cols)
        return tile_h, tile_w, tile_rows, tile_cols
    raise RuntimeError("failed to choose legal geometry")


def _gen_attempt() -> np.ndarray | None:
    tile_h, tile_w, tile_rows, tile_cols = _choose_geometry()
    seed = _make_seed(tile_h, tile_w)
    full = _render_tiling(seed, tile_rows, tile_cols)
    inp = _occlude_every_tile(full, tile_h, tile_w, tile_rows, tile_cols)
    if _valid_generation(inp, seed, tile_h, tile_w):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating mirror tiling input")
