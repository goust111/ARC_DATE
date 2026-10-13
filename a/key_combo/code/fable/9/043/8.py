import random
from typing import Any

import numpy as np


ARC_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 600


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _is_asymmetric(tile: np.ndarray) -> bool:
    return not np.array_equal(tile, np.fliplr(tile)) and not np.array_equal(tile, np.flipud(tile))


def _build_wallpaper(tile: np.ndarray, q: int, p: int) -> np.ndarray:
    rows = []
    for block_r in range(q):
        row_blocks = []
        for block_c in range(p):
            block = tile
            if block_r % 2 == 1:
                block = block[::-1, :]
            if block_c % 2 == 1:
                block = block[:, ::-1]
            row_blocks.append(block)
        rows.append(np.concatenate(row_blocks, axis=1))
    return np.concatenate(rows, axis=0)


def _recover_candidate(grid: np.ndarray, h: int, w: int, *, mirrored: bool) -> dict[str, Any]:
    H, W = map(int, grid.shape)
    q, p = H // h, W // w
    counts = np.zeros((h, w, 10), dtype=int)

    for block_r in range(q):
        for block_c in range(p):
            block = grid[block_r * h : (block_r + 1) * h, block_c * w : (block_c + 1) * w]
            if mirrored and block_r % 2 == 1:
                block = block[::-1, :]
            if mirrored and block_c % 2 == 1:
                block = block[:, ::-1]
            for color in ARC_COLORS:
                counts[:, :, color] += block == color

    tile = counts.argmax(axis=2).astype(int)
    max_counts = counts.max(axis=2)
    copies = q * p
    return {
        "h": h,
        "w": w,
        "copies": copies,
        "tile": tile,
        "min_fraction": float(max_counts.min()) / float(copies),
        "total_majority": int(max_counts.sum()),
    }


def _candidate_key(stats: dict[str, Any]) -> tuple[float, int, int, int, int]:
    return (
        stats["min_fraction"],
        stats["total_majority"],
        -stats["h"] * stats["w"],
        -stats["h"],
        -stats["w"],
    )


def _enumerate_candidates(grid: np.ndarray) -> list[dict[str, Any]]:
    H, W = map(int, grid.shape)
    h_options = [d for d in _divisors(H) if H // d >= 2] or [H]
    w_options = [d for d in _divisors(W) if W // d >= 2] or [W]
    candidates = []
    for h in h_options:
        for w in w_options:
            candidates.append(_recover_candidate(grid, h, w, mirrored=True))
    return candidates


def _choose_best_candidate(grid: np.ndarray) -> dict[str, Any]:
    candidates = _enumerate_candidates(grid)
    asymmetric = [cand for cand in candidates if _is_asymmetric(cand["tile"])]
    pool = asymmetric if asymmetric else candidates
    best = max(pool, key=_candidate_key)
    assert best["min_fraction"] > 0.5, "no valid mirrored-tile folding found"
    return best


def _recover_plain_translation(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    return _recover_candidate(grid, h, w, mirrored=False)["tile"]


def main(grid: np.ndarray) -> np.ndarray:
    """Fold a noisy alternating mirror wallpaper back into its clean base tile.

    Input structure:
    - The full grid is a q-by-p wallpaper built from one unknown base tile with no separators.
    - Tile block (i, j) is left-right flipped when j is odd and up-down flipped when i is odd, using zero-based block indices.
    - A minority of instances of each aligned tile cell may be replaced by noise colors.

    Rule:
    - Infer the tile height and width from the repeating mirror-wallpaper structure.
    - Undo the row-parity and column-parity flips on every block to align all copies into the tile's standard orientation.
    - For each tile cell, output the majority color across all aligned copies.

    Key insight:
    - The wallpaper repeats by alternating reflections rather than plain translation, so the blocks must be unflipped before majority voting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0
    return _choose_best_candidate(g)["tile"].astype(int)


def _ensure_palette_coverage(tile: np.ndarray, palette: list[int]) -> None:
    cells = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]
    random.shuffle(cells)
    present = set(int(v) for v in tile.ravel())
    for color in palette:
        if color not in present:
            r, c = cells.pop()
            tile[r, c] = color
            present.add(color)


def _sample_tile(h: int, w: int) -> np.ndarray | None:
    cell_count = h * w
    color_count = random.randint(3, min(6, cell_count, 10))
    palette = random.sample(list(ARC_COLORS), color_count)
    tile = np.random.choice(palette, size=(h, w)).astype(int)
    _ensure_palette_coverage(tile, palette)

    left_c = random.randrange(max(1, w // 2))
    right_c = w - 1 - left_c
    row = random.randrange(h)
    tile[row, left_c] = palette[0]
    tile[row, right_c] = palette[1 % len(palette)]

    top_r = random.randrange(max(1, h // 2))
    bottom_r = h - 1 - top_r
    col = random.randrange(w)
    tile[top_r, col] = palette[-1]
    tile[bottom_r, col] = palette[0]

    if len(set(int(v) for v in tile.ravel())) < 3:
        return None
    if not _is_asymmetric(tile):
        return None
    return tile


def _add_noise(tile: np.ndarray, q: int, p: int) -> tuple[np.ndarray, int]:
    noisy = _build_wallpaper(tile, q, p).copy()
    copies = q * p
    slots = [(block_r, block_c) for block_r in range(q) for block_c in range(p)]
    cell_noise_prob = random.uniform(0.14, 0.30)
    changed = 0

    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            max_k = (copies - 1) // 2
            if max_k <= 0 or random.random() >= cell_noise_prob:
                continue
            k = 1
            if max_k >= 2 and random.random() < 0.35:
                k = 2
            k = min(k, max_k)
            original = int(tile[r, c])
            for block_r, block_c in random.sample(slots, k):
                rr = block_r * tile.shape[0] + (tile.shape[0] - 1 - r if block_r % 2 == 1 else r)
                cc = block_c * tile.shape[1] + (tile.shape[1] - 1 - c if block_c % 2 == 1 else c)
                replacement = random.choice([color for color in ARC_COLORS if color != original])
                noisy[rr, cc] = replacement
                changed += 1

    if changed == 0:
        r = random.randrange(tile.shape[0])
        c = random.randrange(tile.shape[1])
        block_r, block_c = random.choice(slots)
        rr = block_r * tile.shape[0] + (tile.shape[0] - 1 - r if block_r % 2 == 1 else r)
        cc = block_c * tile.shape[1] + (tile.shape[1] - 1 - c if block_c % 2 == 1 else c)
        original = int(tile[r, c])
        noisy[rr, cc] = random.choice([color for color in ARC_COLORS if color != original])
        changed = 1

    return noisy, changed


def _is_unique_best(candidates: list[dict[str, Any]], truth: tuple[int, int]) -> bool:
    best = max(candidates, key=_candidate_key)
    best_key = _candidate_key(best)
    tied = [
        cand
        for cand in candidates
        if cand["h"] != truth[0] or cand["w"] != truth[1]
        if _candidate_key(cand) == best_key
    ]
    return best["h"] == truth[0] and best["w"] == truth[1] and not tied


def _valid_attempt(tile: np.ndarray, q: int, p: int, noisy: np.ndarray) -> bool:
    h, w = map(int, tile.shape)
    clean = _build_wallpaper(tile, q, p)
    best_clean = _choose_best_candidate(clean)
    if (best_clean["h"], best_clean["w"]) != (h, w):
        return False
    if not np.array_equal(best_clean["tile"], tile):
        return False

    candidates_clean = _enumerate_candidates(clean)
    if not _is_unique_best(candidates_clean, (h, w)):
        return False

    candidates_noisy = _enumerate_candidates(noisy)
    best_noisy = max([cand for cand in candidates_noisy if _is_asymmetric(cand["tile"])] or candidates_noisy, key=_candidate_key)
    if (best_noisy["h"], best_noisy["w"]) != (h, w):
        return False
    if not np.array_equal(best_noisy["tile"], tile):
        return False
    if not _is_unique_best(candidates_noisy, (h, w)):
        return False

    naive = _recover_plain_translation(noisy, h, w)
    if np.array_equal(naive, tile):
        return False

    if best_noisy["min_fraction"] <= 0.5 + (1.0 / float(best_noisy["copies"])):
        return False

    return np.array_equal(main(noisy), tile)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(2, 7)
    w = random.randint(2, 7)
    max_q = min(5, 30 // h)
    max_p = min(5, 30 // w)
    if max_q < 2 or max_p < 2:
        return None

    q = random.randint(2, max_q)
    p = random.randint(2, max_p)
    tile = _sample_tile(h, w)
    if tile is None:
        return None

    noisy, changed = _add_noise(tile, q, p)
    if changed <= 0:
        return None
    if max(noisy.shape) > 30:
        return None
    if not _valid_attempt(tile, q, p, noisy):
        return None
    return noisy


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-wallpaper input")
