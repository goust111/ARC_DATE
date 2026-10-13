import random
from itertools import combinations

import numpy as np


BG = 0
IN_COLOR = 2
OUT_COLOR = 3
CELL = 4
ROLE_COUNT = 6
GEN_RETRY_LIMIT = 600


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _mask_from_cells(cells: set[int]) -> np.ndarray:
    mask = np.zeros((CELL, CELL), dtype=bool)
    if cells:
        coords = np.array(sorted(cells), dtype=int)
        mask.flat[coords] = True
    return mask


def _active_runs(bits: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(bits.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(bits) - 1))
    return runs


def _axis_partitions(indices: np.ndarray, axis_len: int, band_count: int) -> list[tuple[int, ...]]:
    if indices.size == 0:
        return []
    used = sorted(set(int(x) for x in indices.tolist()))
    out: list[tuple[int, ...]] = []

    def rec(starts: list[int], next_start: int) -> None:
        if len(starts) == band_count:
            if all(any(s <= idx < s + CELL for s in starts) for idx in used):
                out.append(tuple(starts))
            return

        remaining = band_count - len(starts)
        max_start = axis_len - CELL - (remaining - 1) * (CELL + 1)
        for s in range(next_start, max_start + 1):
            rec(starts + [s], s + CELL + 1)

    rec([], 0)
    return out


def _extract_sprite_masks(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, IN_COLOR}, "expected only black and red input pixels"
    coords = np.argwhere(g != BG)
    assert len(coords) > 0, "expected foreground sprites"

    row_candidates = _axis_partitions(coords[:, 0], g.shape[0], 2)
    col_candidates = _axis_partitions(coords[:, 1], g.shape[1], 3)
    layouts: list[list[np.ndarray]] = []

    for row_starts in row_candidates:
        for col_starts in col_candidates:
            masks: list[np.ndarray] = []
            ok = True
            for r0 in row_starts:
                for c0 in col_starts:
                    patch = g[r0 : r0 + CELL, c0 : c0 + CELL]
                    if patch.shape != (CELL, CELL):
                        ok = False
                        break
                    mask = patch != BG
                    if not np.any(mask):
                        ok = False
                        break
                    masks.append(mask)
                if not ok:
                    break
            if ok:
                layouts.append(masks)

    assert len(layouts) == 1, "expected a unique 2x3 array of non-empty 4x4 sprites"
    return layouts[0]


def _xor_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for dst in range(len(masks)):
        for i, j in combinations([x for x in range(len(masks)) if x != dst], 2):
            a = masks[i]
            b = masks[j]
            if not np.array_equal(np.logical_xor(a, b), masks[dst]):
                continue
            if not np.any(a & b):
                continue
            if not np.any(a & ~b):
                continue
            if not np.any(b & ~a):
                continue
            candidates.append((i, j, dst))
    return candidates


def _union_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for dst in range(len(masks)):
        for i, j in combinations([x for x in range(len(masks)) if x != dst], 2):
            a = masks[i]
            b = masks[j]
            if not np.array_equal(a | b, masks[dst]):
                continue
            if not np.any(a & ~b):
                continue
            if not np.any(b & ~a):
                continue
            candidates.append((i, j, dst))
    return candidates


def _select_xor_operands(masks: list[np.ndarray]) -> tuple[int, int, int]:
    candidates = _xor_candidates(masks)
    assert candidates, "expected at least one XOR relation"
    triple_sets = {frozenset(candidate) for candidate in candidates}
    assert len(triple_sets) == 1, "expected exactly one XOR triple"

    ranked = []
    for i, j, dst in candidates:
        overlap_size = int(np.sum(masks[i] & masks[j]))
        ranked.append((overlap_size, i, j, dst))
    ranked.sort(reverse=True)
    assert len(ranked) >= 2, "expected multiple XOR rearrangements inside the triple"
    assert ranked[0][0] > ranked[1][0], "XOR operands must have a unique largest overlap"
    _score, i, j, dst = ranked[0]
    return i, j, dst


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the cancelled overlap hidden by the unique sprite XOR relation.

    Input structure:
    - A black background contains six non-empty red 4x4 sprite masks arranged in
      a 2-by-3 array, with blank rows and columns separating the sprite cells.
    - Exactly one triple of sprites satisfies a pixelwise symmetric-difference
      relation: one sprite equals the XOR of two others.
    - A different triple is a decoy that satisfies only a simple union relation.

    Rule:
    - Find the unique triple (A, B, C) such that C is exactly the pixelwise XOR
      of A and B, with A∩B non-empty and each of A, B having at least one unique
      pixel outside the overlap.
    - Within that XOR triple, treat the two sprites with the strictly largest
      shared overlap as the original overprinted operands.
    - Output a new 4x4 grid on black background where only A∩B is colored green.

    Key insight:
    - The missing answer is the overlap erased by XOR cancellation: the residual
      sprite is the symmetric difference, while the hidden operand pair is the
      one that shares the most pixels before cancellation.
    """
    masks = _extract_sprite_masks(grid)
    a_idx, b_idx, _dst = _select_xor_operands(masks)
    union_decoys = _union_candidates(masks)
    assert len(union_decoys) == 1, "expected exactly one union decoy triple"
    assert {a_idx, b_idx, _dst} != set(union_decoys[0]), "XOR triple must differ from union decoy"
    overlap = masks[a_idx] & masks[b_idx]
    out = np.zeros((CELL, CELL), dtype=int)
    out[overlap] = OUT_COLOR
    return out


def _sample_xor_triplet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(200):
        overlap = random.randint(2, 5)
        only_a = random.randint(1, 5)
        only_b = random.randint(1, 5)
        sizes = (overlap + only_a, overlap + only_b, only_a + only_b)
        if min(sizes) < 4 or max(sizes) > 11:
            continue
        total = overlap + only_a + only_b
        if total > 14:
            continue

        universe = random.sample(range(CELL * CELL), total)
        shared = set(random.sample(universe, overlap))
        rest = [x for x in universe if x not in shared]
        a_only = set(random.sample(rest, only_a))
        b_only = set(x for x in rest if x not in a_only)
        a = _mask_from_cells(shared | a_only)
        b = _mask_from_cells(shared | b_only)
        c = _mask_from_cells(a_only | b_only)
        return a, b, c
    raise AssertionError("failed to sample XOR triplet")


def _sample_union_triplet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(200):
        overlap = random.randint(1, 4)
        only_d = random.randint(2, 5)
        only_e = random.randint(2, 5)
        sizes = (overlap + only_d, overlap + only_e, overlap + only_d + only_e)
        if min(sizes) < 4 or max(sizes) > 12:
            continue
        total = overlap + only_d + only_e
        if total > 14:
            continue

        universe = random.sample(range(CELL * CELL), total)
        shared = set(random.sample(universe, overlap))
        rest = [x for x in universe if x not in shared]
        d_only = set(random.sample(rest, only_d))
        e_only = set(x for x in rest if x not in d_only)
        d = _mask_from_cells(shared | d_only)
        e = _mask_from_cells(shared | e_only)
        f = _mask_from_cells(shared | d_only | e_only)
        return d, e, f
    raise AssertionError("failed to sample union triplet")


def _role_masks_valid(masks: list[np.ndarray]) -> bool:
    if len({_mask_key(mask) for mask in masks}) != ROLE_COUNT:
        return False

    xor_answer = masks[0] & masks[1]
    if not (2 <= int(xor_answer.sum()) <= 6):
        return False
    if int(xor_answer.sum()) <= int(np.sum(masks[0] & masks[2])):
        return False
    if int(xor_answer.sum()) <= int(np.sum(masks[1] & masks[2])):
        return False
    if any(np.array_equal(xor_answer, mask) for mask in masks):
        return False

    xor_candidates = _xor_candidates(masks)
    if {frozenset(candidate) for candidate in xor_candidates} != {frozenset({0, 1, 2})}:
        return False
    try:
        if _select_xor_operands(masks) != (0, 1, 2):
            return False
    except AssertionError:
        return False
    if _union_candidates(masks) != [(3, 4, 5)]:
        return False

    same_overlap_pairs = 0
    for i, j in combinations(range(ROLE_COUNT), 2):
        if np.array_equal(masks[i] & masks[j], xor_answer):
            same_overlap_pairs += 1
    if same_overlap_pairs != 1:
        return False

    decoy_overlap = masks[3] & masks[4]
    if np.array_equal(decoy_overlap, xor_answer):
        return False
    if np.array_equal((masks[3] | masks[4]) & ~masks[5], xor_answer):
        return False

    return True


def _sample_layout() -> tuple[list[int], list[int], int, int]:
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    row_gap = random.randint(1, 3)
    col_gap1 = random.randint(1, 3)
    col_gap2 = random.randint(1, 3)

    row_starts = [top, top + CELL + row_gap]
    col_starts = [left, left + CELL + col_gap1, left + 2 * CELL + col_gap1 + col_gap2]
    height = row_starts[-1] + CELL + bottom
    width = col_starts[-1] + CELL + right
    assert height <= 30 and width <= 30
    return row_starts, col_starts, height, width


def _band_edges_visible(placed_masks: list[np.ndarray]) -> bool:
    for band in range(2):
        band_union = np.any(np.stack(placed_masks[band * 3 : band * 3 + 3], axis=0), axis=0)
        rows = np.any(band_union, axis=1)
        if not rows[0] or not rows[-1]:
            return False
    for band in range(3):
        band_union = np.any(np.stack([placed_masks[band], placed_masks[band + 3]], axis=0), axis=0)
        cols = np.any(band_union, axis=0)
        if not cols[0] or not cols[-1]:
            return False
    return True


def _render_input(
    masks: list[np.ndarray], order: list[int], row_starts: list[int], col_starts: list[int], height: int, width: int
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for pos, role in enumerate(order):
        r0 = row_starts[pos // 3]
        c0 = col_starts[pos % 3]
        patch = grid[r0 : r0 + CELL, c0 : c0 + CELL]
        patch[masks[role]] = IN_COLOR
    return grid


def _attempt_grid() -> np.ndarray | None:
    xor_masks = _sample_xor_triplet()
    union_masks = _sample_union_triplet()
    role_masks = [*xor_masks, *union_masks]
    if not _role_masks_valid(role_masks):
        return None

    row_starts, col_starts, height, width = _sample_layout()
    order = random.sample(range(ROLE_COUNT), ROLE_COUNT)
    placed_masks = [role_masks[idx] for idx in order]
    if not _band_edges_visible(placed_masks):
        return None

    grid = _render_input(role_masks, order, row_starts, col_starts, height, width)
    expected = np.zeros((CELL, CELL), dtype=int)
    expected[role_masks[0] & role_masks[1]] = OUT_COLOR

    try:
        parsed_masks = _extract_sprite_masks(grid)
        if len(parsed_masks) != ROLE_COUNT:
            return None
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
