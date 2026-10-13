from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import colors_present, get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.asarray(grid).ravel()))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)


MAX_ATTEMPTS = 500
COLOR_POOL = tuple(range(10))
BLOCK_SIZE_RANGE = (2, 5)
BLOCK_GRID_RANGE = (3, 6)


def _divisors(n: int, *, min_value: int = 1, max_value: int | None = None) -> list[int]:
    if max_value is None:
        max_value = n
    return [d for d in range(min_value, max_value + 1) if n % d == 0]


def _is_repetition(tile: np.ndarray) -> bool:
    h, w = map(int, tile.shape)
    for sh in _divisors(h):
        for sw in _divisors(w):
            if (sh, sw) == (h, w):
                continue
            patch = tile[:sh, :sw]
            if np.array_equal(np.tile(patch, (h // sh, w // sw)), tile):
                return True
    return False


def _mask_has_period(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    for dh in _divisors(h):
        for dw in _divisors(w):
            if (dh, dw) == (h, w):
                continue
            patch = mask[:dh, :dw]
            if np.array_equal(np.tile(patch, (h // dh, w // dw)), mask):
                return True
    return False


def _random_prototype(block_h: int, block_w: int, bg: int) -> np.ndarray | None:
    palette = random.sample([c for c in COLOR_POOL if c != bg], k=random.choice((2, 3)))
    proto = np.array(
        [[random.choice(palette) for _ in range(block_w)] for _ in range(block_h)],
        dtype=int,
    )
    for color in palette:
        if color not in proto:
            rr = random.randrange(block_h)
            cc = random.randrange(block_w)
            proto[rr, cc] = color
    if max(Counter(map(int, proto.ravel())).values()) > proto.size - 2:
        return None
    if len(colors_present(proto)) < 2:
        return None
    if len({tuple(row) for row in proto.tolist()}) == 1:
        return None
    if len({tuple(col) for col in proto.T.tolist()}) == 1:
        return None
    if _is_repetition(proto):
        return None
    return proto


def _random_active_mask(block_rows: int, block_cols: int) -> np.ndarray | None:
    total = block_rows * block_cols
    for _ in range(80):
        density = random.uniform(0.45, 0.68)
        mask = np.array(
            [[random.random() < density for _ in range(block_cols)] for _ in range(block_rows)],
            dtype=bool,
        )
        count = int(mask.sum())
        if count < 5 or count > total - 3:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if _mask_has_period(mask):
            continue
        return mask
    return None


def _pick_defects(
    proto: np.ndarray,
    active_mask: np.ndarray,
    bg: int,
) -> dict[tuple[int, int], tuple[int, int, int]]:
    block_rows, block_cols = map(int, active_mask.shape)
    active_positions = [(r, c) for r in range(block_rows) for c in range(block_cols) if active_mask[r, c]]
    max_defects = min(max(1, len(active_positions) // 3), max(1, proto.size), 4)
    defect_count = random.randint(1, max_defects)
    chosen_positions = random.sample(active_positions, k=defect_count)
    cell_choices = [(r, c) for r in range(proto.shape[0]) for c in range(proto.shape[1])]
    random.shuffle(cell_choices)
    chosen_cells = cell_choices[:defect_count]
    defects: dict[tuple[int, int], tuple[int, int, int]] = {}
    for (br, bc), (rr, cc) in zip(chosen_positions, chosen_cells, strict=False):
        wrong_options = [c for c in COLOR_POOL if c not in {bg, int(proto[rr, cc])}]
        defects[(br, bc)] = (rr, cc, random.choice(wrong_options))
    return defects


def _render_instance(
    proto: np.ndarray,
    active_mask: np.ndarray,
    defects: dict[tuple[int, int], tuple[int, int, int]],
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    block_h, block_w = map(int, proto.shape)
    block_rows, block_cols = map(int, active_mask.shape)
    grid = np.full((block_rows * block_h, block_cols * block_w), bg, dtype=int)
    output = np.full((block_rows, block_cols), bg, dtype=int)

    for br in range(block_rows):
        for bc in range(block_cols):
            if not active_mask[br, bc]:
                continue
            block = proto.copy()
            defect = defects.get((br, bc))
            if defect is not None:
                rr, cc, wrong = defect
                block[rr, cc] = wrong
                output[br, bc] = wrong
            r0 = br * block_h
            c0 = bc * block_w
            grid[r0 : r0 + block_h, c0 : c0 + block_w] = block
    return grid, output


def _analyze_candidate(grid: np.ndarray, block_h: int, block_w: int, bg: int) -> dict | None:
    H, W = map(int, grid.shape)
    if H % block_h or W % block_w:
        return None
    block_rows = H // block_h
    block_cols = W // block_w
    if block_rows < 2 or block_cols < 2:
        return None

    blocks: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * block_h
            c0 = bc * block_w
            block = grid[r0 : r0 + block_h, c0 : c0 + block_w]
            if np.all(block == bg):
                continue
            blocks.append(block)
            positions.append((br, bc))
    if len(blocks) < 4:
        return None

    stack = np.stack(blocks, axis=0)
    proto = np.empty((block_h, block_w), dtype=int)
    for rr in range(block_h):
        for cc in range(block_w):
            counts = Counter(map(int, stack[:, rr, cc]))
            top_two = counts.most_common(2)
            color, count = top_two[0]
            if count <= len(blocks) // 2:
                return None
            if len(top_two) > 1 and top_two[1][1] == count:
                return None
            proto[rr, cc] = color

    proto_colors = colors_present(proto)
    if len(proto_colors) < 2 or np.all(proto == bg):
        return None

    defect_map = np.full((block_rows, block_cols), bg, dtype=int)
    defect_count = 0
    for block, (br, bc) in zip(blocks, positions, strict=False):
        mismatch = np.argwhere(block != proto)
        if mismatch.shape[0] == 0:
            continue
        if mismatch.shape[0] != 1:
            return None
        rr, cc = map(int, mismatch[0])
        defect_map[br, bc] = int(block[rr, cc])
        defect_count += 1

    if defect_count == 0 or defect_count > len(blocks) // 2:
        return None

    return {
        "block_shape": (block_h, block_w),
        "prototype": proto,
        "output": defect_map,
        "tile_count": len(blocks),
        "defect_count": defect_count,
    }


def _find_valid_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    H, W = map(int, grid.shape)
    candidates = []
    for block_h in _divisors(H, min_value=2, max_value=H):
        for block_w in _divisors(W, min_value=2, max_value=W):
            candidate = _analyze_candidate(grid, block_h, block_w, bg)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Downsample the hidden tile-stamping grid into a defect-color map.

    Input structure:
    - The full grid is partitionable into equal rectangular blocks with no separator lines.
    - Each block is either solid background or a copy of one shared multicolor prototype tile.
    - A minority of the tile copies contain exactly one wrong-colored pixel, while the others are exact.

    Rule:
    - Search for the unique block size whose non-background blocks admit a single prototype recovered by per-cell majority vote.
    - Every non-background block must match that prototype except for at most one pixel.
    - Return the smaller block-layout grid: defective copies contribute the wrong pixel color at their block position, while clean copies and background blocks become background.

    Key insight:
    - The large grid is a hidden pixel-scaling of a smaller control mask, but the output is not the mask itself; it records only which block copies contain a one-pixel defect and what wrong color appeared there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    candidates = _find_valid_candidates(g, bg)
    assert len(candidates) == 1, "expected a unique valid block size"
    return candidates[0]["output"]


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    block_rows = random.randint(*BLOCK_GRID_RANGE)
    block_cols = random.randint(*BLOCK_GRID_RANGE)
    max_block_h = min(BLOCK_SIZE_RANGE[1], 30 // block_rows)
    max_block_w = min(BLOCK_SIZE_RANGE[1], 30 // block_cols)
    if max_block_h < BLOCK_SIZE_RANGE[0] or max_block_w < BLOCK_SIZE_RANGE[0]:
        return None
    block_h = random.randint(BLOCK_SIZE_RANGE[0], max_block_h)
    block_w = random.randint(BLOCK_SIZE_RANGE[0], max_block_w)
    bg = random.choice(COLOR_POOL)

    proto = _random_prototype(block_h, block_w, bg)
    if proto is None:
        return None
    active_mask = _random_active_mask(block_rows, block_cols)
    if active_mask is None:
        return None
    defects = _pick_defects(proto, active_mask, bg)
    if len(defects) >= int(active_mask.sum()):
        return None

    grid, expected = _render_instance(proto, active_mask, defects, bg)
    if get_bg(grid) != bg:
        return None
    candidates = _find_valid_candidates(grid, bg)
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    if candidate["block_shape"] != (block_h, block_w):
        return None
    if not np.array_equal(candidate["prototype"], proto):
        return None
    if not np.array_equal(candidate["output"], expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pair = _gen_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid hidden-tiling defect grid")
