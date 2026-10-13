from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

SPRITE_SIZE = 4
SPRITE_ROWS = 2
SPRITE_COLS = 3
SPRITE_GAP = 1
BLOCK_HEIGHT = SPRITE_ROWS * SPRITE_SIZE + (SPRITE_ROWS - 1) * SPRITE_GAP
BLOCK_WIDTH = SPRITE_COLS * SPRITE_SIZE + (SPRITE_COLS - 1) * SPRITE_GAP

GRID_WIDTH_RANGE = (16, 24)
TOP_MARGIN_RANGE = (0, 2)
BOTTOM_MARGIN_RANGE = (0, 2)
VERTICAL_GAP_RANGE = (2, 4)
TARGET_HEIGHT_RANGE = (6, 8)
TARGET_WIDTH_RANGE = (8, 12)
GEN_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0 = min(rows)
            r1 = max(rows)
            c0 = min(cols)
            c1 = max(cols)
            boxes.append((r0, c0, r1, c1, len(coords)))
    return boxes


def _find_target_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for r0, c0, r1, c1, area in _component_bboxes(grid == RED):
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height > SPRITE_SIZE and width > SPRITE_SIZE and area == height * width:
            candidates.append((r0, c0, r1, c1, area))

    assert len(candidates) == 1, "expected one solid red target rectangle"
    r0, c0, r1, c1, _ = candidates[0]
    return r0, c0, r1, c1


def _extract_sprite_masks(grid: np.ndarray, target_bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    bg = _get_bg(grid)
    upper = grid.copy()
    r0, c0, r1, c1 = target_bbox
    upper[r0 : r1 + 1, c0 : c1 + 1] = bg

    fg_mask = upper == RED
    row_runs = _active_runs(np.any(fg_mask, axis=1))
    col_runs = _active_runs(np.any(fg_mask, axis=0))
    assert len(row_runs) == SPRITE_ROWS and len(col_runs) == SPRITE_COLS, "expected a 2x3 sprite block"
    assert all(end - start + 1 == SPRITE_SIZE for start, end in row_runs), "sprite rows must be 4 high"
    assert all(end - start + 1 == SPRITE_SIZE for start, end in col_runs), "sprite cols must be 4 wide"

    masks: list[np.ndarray] = []
    for rr0, rr1 in row_runs:
        for cc0, cc1 in col_runs:
            patch = upper[rr0 : rr1 + 1, cc0 : cc1 + 1]
            assert patch.shape == (SPRITE_SIZE, SPRITE_SIZE)
            assert np.all((patch == bg) | (patch == RED)), "sprite cells must use only bg and red"
            mask = patch == RED
            assert np.any(mask), "sprite cells must be non-empty"
            masks.append(mask)
    return masks


def _union_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for dst in range(len(masks)):
        others = [idx for idx in range(len(masks)) if idx != dst]
        for src_a, src_b in itertools.combinations(others, 2):
            mask_a = masks[src_a]
            mask_b = masks[src_b]
            if not np.array_equal(mask_a | mask_b, masks[dst]):
                continue
            if not np.any(mask_a & ~mask_b):
                continue
            if not np.any(mask_b & ~mask_a):
                continue
            candidates.append((dst, src_a, src_b))
    return sorted(candidates)


def _choose_union_triple(masks: list[np.ndarray]) -> tuple[int, int, int]:
    candidates = _union_candidates(masks)
    assert candidates, "expected at least one mask-union relation"
    return candidates[0]


def _build_template(mask_a: np.ndarray, mask_b: np.ndarray, mask_c: np.ndarray) -> np.ndarray:
    assert np.array_equal(mask_a | mask_b, mask_c), "destination mask must be the union of the operands"
    template = np.zeros(mask_c.shape, dtype=int)
    template[mask_a & ~mask_b] = BLUE
    template[mask_b & ~mask_a] = GREEN
    template[mask_a & mask_b] = YELLOW
    return template


def _axis_indices(source_len: int, target_len: int) -> list[int]:
    assert target_len >= source_len and source_len >= 3
    fixed = 2
    interior_source = source_len - fixed
    interior_target = target_len - fixed
    assert interior_target > 0

    base = interior_target // interior_source
    rem = interior_target % interior_source
    counts = [base + (1 if idx < rem else 0) for idx in range(interior_source)]

    indices = [0]
    for offset, count in enumerate(counts, start=1):
        indices.extend([offset] * count)
    indices.append(source_len - 1)
    assert len(indices) == target_len
    return indices


def _stretch_template(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_indices = _axis_indices(template.shape[0], int(shape[0]))
    col_indices = _axis_indices(template.shape[1], int(shape[1]))
    return template[np.ix_(row_indices, col_indices)]


def main(grid: np.ndarray) -> np.ndarray:
    """Find the unique union-result sprite, recolor its operand zones, and elastically stretch that template into the large red rectangle.

    Input structure:
    - The grid has a black background with six red 4x4 sprite cells arranged as a 2-by-3 block in the upper area.
    - Each sprite cell contains a non-empty red mask, and exactly one larger solid red rectangle appears lower in the grid.
    - Among the six sprite masks, exactly one sprite equals the pixelwise union of two others, with both operands contributing at least one exclusive pixel.

    Rule:
    - Enumerate sprite triples in row-major order and find the earliest valid relation mask(A) OR mask(B) = mask(C).
    - Convert C into a 4x4 multicolor template: A-only pixels become blue, B-only pixels become green, shared pixels become yellow, and empty cells stay black.
    - Stretch that 4x4 template to the target rectangle size while keeping the outermost rows and columns one cell thick and distributing the interior rows and columns evenly.
    - Replace the solid red target rectangle with the stretched template and leave the six source sprites unchanged.

    Key insight:
    - The large rectangle cannot be filled until the solver first discovers which small sprite is a union result, because that relation also determines the blue/green/yellow partition that must survive the later stretch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"
    colors = {int(v) for v in np.unique(g)}
    assert colors <= {BLACK, RED}, "expected only black and red in the input"

    target_bbox = _find_target_bbox(g, bg)
    masks = _extract_sprite_masks(g, target_bbox)
    dst, src_a, src_b = _choose_union_triple(masks)

    template = _build_template(masks[src_a], masks[src_b], masks[dst])
    r0, c0, r1, c1 = target_bbox
    stretched = _stretch_template(template, (r1 - r0 + 1, c1 - c0 + 1))

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = stretched
    return out


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _random_cover_mask(min_pixels: int, max_pixels: int) -> np.ndarray:
    count = random.randint(min_pixels, max_pixels)
    cols = random.sample(range(SPRITE_SIZE), SPRITE_SIZE)
    cells = {(row, col) for row, col in enumerate(cols)}
    while len(cells) < count:
        cells.add((random.randrange(SPRITE_SIZE), random.randrange(SPRITE_SIZE)))

    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    for row, col in cells:
        mask[row, col] = True
    return mask


def _generate_operands() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_overlap = random.random() < 0.65
    for _ in range(300):
        if require_overlap:
            mask_a = _random_cover_mask(4, 6)
            mask_b = _random_cover_mask(4, 6)
            overlap = int(np.count_nonzero(mask_a & mask_b))
            union_size = int(np.count_nonzero(mask_a | mask_b))
            if overlap == 0 or not (5 <= union_size <= 8):
                continue
        else:
            mask_a = _random_cover_mask(4, 4)
            mask_b = _random_cover_mask(4, 4)
            if np.any(mask_a & mask_b):
                continue
            if int(np.count_nonzero(mask_a | mask_b)) != 8:
                continue

        if not np.any(mask_a & ~mask_b):
            continue
        if not np.any(mask_b & ~mask_a):
            continue

        mask_c = mask_a | mask_b
        return mask_a, mask_b, mask_c
    raise RuntimeError("failed to generate operand masks")


def _is_intended_candidate(
    masks: list[np.ndarray],
    candidate: tuple[int, int, int],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dst_index: int,
) -> bool:
    dst, src_a, src_b = candidate
    if dst != dst_index:
        return False

    first = masks[src_a]
    second = masks[src_b]
    return (
        (np.array_equal(first, mask_a) and np.array_equal(second, mask_b))
        or (np.array_equal(first, mask_b) and np.array_equal(second, mask_a))
    )


def _valid_mask_set(masks: list[np.ndarray], mask_a: np.ndarray, mask_b: np.ndarray, mask_c: np.ndarray) -> bool:
    c_positions = [idx for idx, mask in enumerate(masks) if np.array_equal(mask, mask_c)]
    if len(c_positions) != 1:
        return False

    dst_index = c_positions[0]
    candidates = _union_candidates(masks)
    if not candidates:
        return False
    if {dst for dst, _, _ in candidates} != {dst_index}:
        return False
    if not all(_is_intended_candidate(masks, cand, mask_a, mask_b, dst_index) for cand in candidates):
        return False
    return True


def _sample_mask_set() -> list[np.ndarray]:
    mask_a, mask_b, mask_c = _generate_operands()
    use_duplicate_operand = random.random() < 0.45

    distractors: list[np.ndarray] = []
    used_keys = {_mask_key(mask_a), _mask_key(mask_b), _mask_key(mask_c)}
    if use_duplicate_operand:
        duplicate = mask_a.copy() if random.random() < 0.5 else mask_b.copy()
        distractors.append(duplicate)

    while len(distractors) < 3:
        distractor = _random_cover_mask(4, 8)
        key = _mask_key(distractor)
        if key in used_keys:
            continue
        if any(np.array_equal(distractor, prev) for prev in distractors):
            continue
        distractors.append(distractor)
        used_keys.add(key)

    masks = [mask.copy() for mask in [mask_a, mask_b, mask_c] + distractors]
    random.shuffle(masks)
    if not _valid_mask_set(masks, mask_a, mask_b, mask_c):
        raise RuntimeError("invalid shuffled mask set")
    return masks


def _place_sprite_block(grid: np.ndarray, masks: list[np.ndarray], top: int, left: int) -> None:
    idx = 0
    for block_r in range(SPRITE_ROWS):
        for block_c in range(SPRITE_COLS):
            r0 = top + block_r * (SPRITE_SIZE + SPRITE_GAP)
            c0 = left + block_c * (SPRITE_SIZE + SPRITE_GAP)
            patch = masks[idx]
            grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE][patch] = RED
            idx += 1


def _gen_attempt() -> np.ndarray | None:
    try:
        masks = _sample_mask_set()
    except RuntimeError:
        return None

    target_h = random.randint(*TARGET_HEIGHT_RANGE)
    target_w = random.randint(*TARGET_WIDTH_RANGE)
    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    vertical_gap = random.randint(*VERTICAL_GAP_RANGE)

    min_width = max(BLOCK_WIDTH, target_w)
    width = random.randint(min_width, GRID_WIDTH_RANGE[1])
    height = top_margin + BLOCK_HEIGHT + vertical_gap + target_h + bottom_margin
    if height > 30:
        return None

    block_left = random.randint(0, width - BLOCK_WIDTH)
    target_left = random.randint(0, width - target_w)
    target_top = top_margin + BLOCK_HEIGHT + vertical_gap

    grid = np.zeros((height, width), dtype=int)
    _place_sprite_block(grid, masks, top_margin, block_left)
    grid[target_top : target_top + target_h, target_left : target_left + target_w] = RED

    try:
        out = main(grid)
    except Exception:
        return None

    target_patch = out[target_top : target_top + target_h, target_left : target_left + target_w]
    output_colors = {int(v) for v in np.unique(target_patch) if int(v) != BLACK}
    if len(output_colors) < 2:
        return None
    if np.all(target_patch == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid elastic-union sample")
