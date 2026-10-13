import random

import numpy as np


BLACK = 0
DAMAGE = 9
PATTERN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
SYMMETRIES = ("horizontal", "vertical", "rot180")

BLOCK_COUNT_RANGE = (3, 6)
BLOCK_SIZE_RANGE = (5, 8)
PALETTE_SIZE_RANGE = (2, 4)
GRID_SIDE_LIMITS = (22, 34)
PATTERN_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _partner(shape: tuple[int, int], symmetry: str, r: int, c: int) -> tuple[int, int]:
    h, w = shape
    if symmetry == "horizontal":
        return h - 1 - r, c
    if symmetry == "vertical":
        return r, w - 1 - c
    if symmetry == "rot180":
        return h - 1 - r, w - 1 - c
    raise ValueError(f"unknown symmetry: {symmetry}")


def _supports_symmetry(block: np.ndarray, symmetry: str) -> bool:
    h, w = block.shape
    for r in range(h):
        for c in range(w):
            rr, cc = _partner((h, w), symmetry, r, c)
            a = int(block[r, c])
            b = int(block[rr, cc])
            if a != DAMAGE and b != DAMAGE and a != b:
                return False
    return True


def _repair_with_symmetry(block: np.ndarray, symmetry: str) -> np.ndarray | None:
    repaired = block.copy()
    h, w = repaired.shape
    changed = True
    while changed:
        changed = False
        for r in range(h):
            for c in range(w):
                if int(repaired[r, c]) != DAMAGE:
                    continue
                rr, cc = _partner((h, w), symmetry, r, c)
                if int(repaired[rr, cc]) != DAMAGE:
                    repaired[r, c] = repaired[rr, cc]
                    changed = True
    if np.any(repaired == DAMAGE):
        return None
    if not _supports_symmetry(repaired, symmetry):
        return None
    return repaired


def _dedupe_arrays(arrays: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for arr in arrays:
        if not any(np.array_equal(arr, other) for other in unique):
            unique.append(arr)
    return unique


def _candidate_repairs(block: np.ndarray) -> list[np.ndarray]:
    repairs = []
    for symmetry in SYMMETRIES:
        if not _supports_symmetry(block, symmetry):
            continue
        repaired = _repair_with_symmetry(block, symmetry)
        if repaired is not None:
            repairs.append(repaired)
    return _dedupe_arrays(repairs)


def _block_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for coords in _components(grid != BLACK):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        boxes.append((int(r0), int(r1) + 1, int(c0), int(c1) + 1))
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair maroon-marked pixels inside colored rectangular pattern blocks by
    inferring each block's original symmetry.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, r1, c0, c1 in _block_boxes(g):
        block = g[r0:r1, c0:c1]
        if not np.any(block == DAMAGE):
            continue
        repairs = _candidate_repairs(block)
        assert len(repairs) == 1, "damaged block must have a unique symmetric repair"
        out[r0:r1, c0:c1] = repairs[0]

    return out


def _exact_symmetries(block: np.ndarray) -> set[str]:
    return {symmetry for symmetry in SYMMETRIES if _supports_symmetry(block, symmetry)}


def _random_symmetric_pattern(h: int, w: int, symmetry: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        palette_size = rng.randint(*PALETTE_SIZE_RANGE)
        palette = rng.sample(PATTERN_COLORS, palette_size)
        pattern = np.empty((h, w), dtype=int)
        filled = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if filled[r, c]:
                    continue
                rr, cc = _partner((h, w), symmetry, r, c)
                color = rng.choice(palette)
                pattern[r, c] = color
                pattern[rr, cc] = color
                filled[r, c] = True
                filled[rr, cc] = True
        if len(np.unique(pattern)) < 2:
            continue
        if _exact_symmetries(pattern) == {symmetry}:
            return pattern
    return None


def _damage_candidates(shape: tuple[int, int], symmetry: str) -> list[tuple[int, int]]:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            rr, cc = _partner(shape, symmetry, r, c)
            if (r, c) == (rr, cc):
                continue
            if (r, c) < (rr, cc):
                candidates.append((r, c))
    return candidates


def _make_block(symmetry: str, damaged: bool, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATTERN_ATTEMPTS):
        h = rng.randint(*BLOCK_SIZE_RANGE)
        w = rng.randint(*BLOCK_SIZE_RANGE)
        full = _random_symmetric_pattern(h, w, symmetry, rng)
        if full is None:
            continue
        if not damaged:
            return full, full.copy()

        choices = _damage_candidates(full.shape, symmetry)
        if not choices:
            continue
        count = rng.choice([n for n in (1, 2, 3) if n <= len(choices)])
        damaged_block = full.copy()
        for r, c in rng.sample(choices, count):
            damaged_block[r, c] = DAMAGE
        repairs = _candidate_repairs(damaged_block)
        if len(repairs) == 1 and np.array_equal(repairs[0], full):
            return damaged_block, full
    return None


def _choose_grid_shape(blocks: list[dict], rng: random.Random) -> tuple[int, int]:
    total_area = sum(block["input"].shape[0] * block["input"].shape[1] for block in blocks)
    max_h = max(block["input"].shape[0] for block in blocks)
    max_w = max(block["input"].shape[1] for block in blocks)
    base = int(np.ceil(np.sqrt(total_area * 2.4)))
    min_side = max(GRID_SIDE_LIMITS[0], base, max_h + 2, max_w + 2)
    max_side = max(min_side, GRID_SIDE_LIMITS[1])
    h_low = min_side
    h_high = min(max_side, min_side + 6)
    w_low = min_side
    w_high = min(max_side, min_side + 8)
    return rng.randint(h_low, h_high), rng.randint(w_low, w_high)


def _place_blocks(blocks: list[dict], grid_h: int, grid_w: int, rng: random.Random):
    order = sorted(
        range(len(blocks)),
        key=lambda idx: blocks[idx]["input"].shape[0] * blocks[idx]["input"].shape[1],
        reverse=True,
    )
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = [None] * len(blocks)

    for idx in order:
        h, w = blocks[idx]["input"].shape
        if grid_h - h < 2 or grid_w - w < 2:
            return None
        row_min, row_max = 1, grid_h - h - 1
        col_min, col_max = 1, grid_w - w - 1
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(row_min, row_max)
            c0 = rng.randint(col_min, col_max)
            rr0 = max(0, r0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid_w, c0 + w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            placements[idx] = (r0, c0)
            break
        else:
            return None
    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_blocks = rng.randint(*BLOCK_COUNT_RANGE)
    symmetries = list(SYMMETRIES)
    while len(symmetries) < n_blocks:
        symmetries.append(rng.choice(SYMMETRIES))
    rng.shuffle(symmetries)

    broken_count = rng.randint(1, n_blocks - 1)
    broken_indices = set(rng.sample(range(n_blocks), broken_count))

    blocks = []
    for idx, symmetry in enumerate(symmetries):
        pair = _make_block(symmetry, idx in broken_indices, rng)
        if pair is None:
            return None
        inp, out = pair
        blocks.append({"input": inp, "output": out, "symmetry": symmetry})

    grid_h, grid_w = _choose_grid_shape(blocks, rng)
    placements = _place_blocks(blocks, grid_h, grid_w, rng)
    if placements is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)
    for block, placement in zip(blocks, placements):
        r0, c0 = placement
        h, w = block["input"].shape
        inp[r0 : r0 + h, c0 : c0 + w] = block["input"]
        expected[r0 : r0 + h, c0 : c0 + w] = block["output"]

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetry-repair input")
