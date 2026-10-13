import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GEN_ATTEMPTS = 600
NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
TILE_SHAPES = [
    (h, w)
    for h in range(2, 8)
    for w in range(2, 8)
    if h * w >= 5 and h + w <= 9
]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _has_proper_repeat(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for sh in _divisors(h):
        for sw in _divisors(w):
            if sh == h and sw == w:
                continue
            base = mask[:sh, :sw]
            if np.array_equal(mask, np.tile(base, (h // sh, w // sw))):
                return True
    return False


def _reshape_blocks(mask: np.ndarray, bh: int, bw: int) -> np.ndarray:
    h, w = mask.shape
    return mask.reshape(h // bh, bh, w // bw, bw).transpose(0, 2, 1, 3)


def _candidate_for_size(mask: np.ndarray, bh: int, bw: int) -> dict | None:
    H, W = mask.shape
    if H % bh != 0 or W % bw != 0:
        return None

    R, C = H // bh, W // bw
    blocks = _reshape_blocks(mask, bh, bw)
    empty = ~blocks.any(axis=(2, 3))
    if not empty.any():
        return None

    by_key: dict[bytes, np.ndarray] = {}
    counts: dict[bytes, int] = {}
    positions: dict[bytes, list[tuple[int, int]]] = {}
    for r in range(R):
        for c in range(C):
            block = blocks[r, c]
            if not block.any():
                continue
            key = block.tobytes()
            if key not in by_key:
                by_key[key] = block.copy()
                positions[key] = []
                counts[key] = 0
            counts[key] += 1
            positions[key].append((r, c))

    if len(by_key) != 2:
        return None

    keys = sorted(by_key, key=lambda key: counts[key])
    odd_key, main_key = keys
    odd_block = by_key[odd_key]
    main_block = by_key[main_key]
    odd_count = counts[odd_key]
    main_count = counts[main_key]
    if odd_count != 1 or main_count < 3:
        return None
    if np.array_equal(main_block, odd_block):
        return None
    if not np.array_equal(odd_block, np.rot90(main_block, 2)):
        return None

    summary = np.zeros((R, C), dtype=int)
    summary[~empty] = BLUE
    odd_pos = positions[odd_key][0]
    summary[odd_pos] = RED

    return {
        "block_shape": (bh, bw),
        "layout_shape": (R, C),
        "common": main_block,
        "odd": odd_block,
        "summary": summary,
    }


def _candidate_rank(candidate: dict) -> tuple[int, int, int, int, int]:
    tile = candidate["common"]
    layout_mask = candidate["summary"] != BLACK
    bh, bw = candidate["block_shape"]
    R, C = candidate["layout_shape"]
    return (
        0 if _touches_all_edges(tile) else 1,
        0 if not _has_proper_repeat(tile) else 1,
        0 if not _has_proper_repeat(layout_mask) else 1,
        bh * bw,
        -(R * C),
    )


def _find_candidates(mask: np.ndarray) -> list[dict]:
    H, W = mask.shape
    candidates = []
    for bh in _divisors(H):
        for bw in _divisors(W):
            candidate = _candidate_for_size(mask, bh, bw)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=_candidate_rank)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden block grid, then summarize which macro-cells are empty,
    normal, or occupied by the single 180-degree-rotated odd tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BLACK if np.any(g == BLACK) else _dominant_color(g)
    colors = {int(color) for color in np.unique(g)}
    fg_colors = colors - {bg}
    assert len(fg_colors) == 1

    mask = g != bg
    candidates = _find_candidates(mask)
    assert candidates, "failed to infer a valid tiling"
    return candidates[0]["summary"]


def _neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in NEIGHBORS_4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _grow_tile(h: int, w: int, count: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    bar_row = random.randrange(h)
    bar_col = random.randrange(w)
    mask[:, bar_col] = True
    mask[bar_row, :] = True

    if int(mask.sum()) > count:
        return None

    while int(mask.sum()) < count:
        frontier = set()
        for cell in map(tuple, np.argwhere(mask)):
            for nb in _neighbors(cell, h, w):
                if not mask[nb]:
                    frontier.add(nb)
        if not frontier:
            return None
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True

    return mask


def _random_tile() -> np.ndarray | None:
    h, w = random.choice(TILE_SHAPES)
    min_cells = max(5, h + w - 1)
    max_cells = min(8, h * w - 1)
    if min_cells > max_cells:
        return None

    for _ in range(80):
        count = random.randint(min_cells, max_cells)
        mask = _grow_tile(h, w, count)
        if mask is None:
            continue
        if not _touches_all_edges(mask):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if _has_proper_repeat(mask):
            continue
        return mask
    return None


def _random_layout(R: int, C: int) -> np.ndarray | None:
    total = R * C
    if total < 5:
        return None

    for _ in range(120):
        active_count = random.randint(4, total - 1)
        layout = np.zeros((R, C), dtype=bool)
        picked = random.sample(range(total), active_count)
        layout.reshape(-1)[picked] = True
        if len(np.where(layout.any(axis=1))[0]) < 2:
            continue
        if len(np.where(layout.any(axis=0))[0]) < 2:
            continue
        return layout
    return None


def _render_grid(layout: np.ndarray, tile: np.ndarray, odd_pos: tuple[int, int]) -> np.ndarray:
    R, C = layout.shape
    h, w = tile.shape
    out = np.zeros((R * h, C * w), dtype=int)
    odd_tile = np.rot90(tile, 2)
    for r in range(R):
        for c in range(C):
            if not layout[r, c]:
                continue
            block = odd_tile if (r, c) == odd_pos else tile
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = block.astype(int) * BLUE
    return out


def _expected_summary(layout: np.ndarray, odd_pos: tuple[int, int]) -> np.ndarray:
    summary = np.zeros(layout.shape, dtype=int)
    summary[layout] = BLUE
    summary[odd_pos] = RED
    return summary


def _gen_attempt() -> np.ndarray | None:
    tile = _random_tile()
    if tile is None:
        return None

    h, w = tile.shape
    max_R = min(10, 30 // h)
    max_C = min(10, 30 // w)
    if max_R < 2 or max_C < 2:
        return None

    for _ in range(80):
        R = random.randint(2, max_R)
        C = random.randint(2, max_C)
        layout = _random_layout(R, C)
        if layout is None:
            continue

        active_cells = [tuple(pos) for pos in np.argwhere(layout)]
        odd_pos = random.choice(active_cells)
        grid = _render_grid(layout, tile, odd_pos)
        expected = _expected_summary(layout, odd_pos)

        candidates = _find_candidates(grid != BLACK)
        if len(candidates) != 1:
            continue
        if not np.array_equal(candidates[0]["summary"], expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
