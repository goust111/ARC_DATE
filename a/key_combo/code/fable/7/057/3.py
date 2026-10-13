import itertools
import random
from collections import Counter

import numpy as np


BG = 0
FG_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_SIDE = 9
TILE_SIDE = 3
MASK_SIDE = 3
MAX_ATTEMPTS = 500


def _tile_key(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in tile.astype(int).ravel())


def _key_tile(key: tuple[int, ...]) -> np.ndarray:
    return np.array(key, dtype=bool).reshape(MASK_SIDE, MASK_SIDE)


def _split_tiles(mask_grid: np.ndarray) -> list[np.ndarray]:
    return [
        mask_grid[br * TILE_SIDE : (br + 1) * TILE_SIDE, bc * TILE_SIDE : (bc + 1) * TILE_SIDE].copy()
        for br in range(MASK_SIDE)
        for bc in range(MASK_SIDE)
    ]


def _render_nested(outer: np.ndarray, inner: np.ndarray, fg: int) -> np.ndarray:
    out = np.zeros((GRID_SIDE, GRID_SIDE), dtype=int)
    patch = np.where(inner, int(fg), BG).astype(int)
    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            if outer[br, bc]:
                r0 = br * TILE_SIDE
                c0 = bc * TILE_SIDE
                out[r0 : r0 + TILE_SIDE, c0 : c0 + TILE_SIDE] = patch
    return out


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < MASK_SIDE and 0 <= cc < MASK_SIDE and mask[rr, cc] and (rr, cc) not in seen:
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _bbox_ok(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.max() - rs.min() + 1) >= 2 and int(cs.max() - cs.min() + 1) >= 2


def _symbol_pool() -> list[np.ndarray]:
    masks = []
    for bits in range(1 << 9):
        arr = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        cells = int(arr.sum())
        if 3 <= cells <= 6 and _connected4(arr) and _bbox_ok(arr):
            masks.append(arr)
    return masks


def _damage_pool() -> list[np.ndarray]:
    masks = []
    for bits in range(1 << 9):
        arr = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        cells = int(arr.sum())
        if 1 <= cells <= 6 and _connected4(arr) and _bbox_ok(arr):
            masks.append(arr)
    return masks


SYMBOLS = _symbol_pool()
DAMAGE_MASKS = _damage_pool()
EMPTY_KEY = _tile_key(np.zeros((3, 3), dtype=bool))


def _find_candidates(mask_grid: np.ndarray) -> list[dict]:
    tiles = _split_tiles(mask_grid)
    tile_keys = [_tile_key(tile) for tile in tiles]
    nonempty_keys = sorted(set(tile_keys) - {EMPTY_KEY})

    seen = set()
    candidates = []
    for b_key in nonempty_keys:
        b_mask = _key_tile(b_key)
        for damaged_idx in range(9):
            a_bits = [0] * 9
            ok = True
            for idx, key in enumerate(tile_keys):
                if idx == damaged_idx:
                    continue
                if key == EMPTY_KEY:
                    a_bits[idx] = 0
                elif key == b_key:
                    a_bits[idx] = 1
                else:
                    ok = False
                    break
            if not ok:
                continue

            intact_copies = sum(a_bits[idx] for idx in range(9) if idx != damaged_idx)
            if intact_copies == 0:
                continue

            for damaged_bit in (0, 1):
                expected_key = b_key if damaged_bit else EMPTY_KEY
                if tile_keys[damaged_idx] == expected_key:
                    continue

                a_bits[damaged_idx] = damaged_bit
                a_mask = np.array(a_bits, dtype=bool).reshape(3, 3)
                if np.array_equal(a_mask, b_mask):
                    continue

                sig = (_tile_key(a_mask), b_key, damaged_idx)
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append(
                    {
                        "A": a_mask.copy(),
                        "B": b_mask.copy(),
                        "damage_idx": damaged_idx,
                        "damage_expected": damaged_bit,
                    }
                )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the unique damaged nested 3x3 symbol product and swap which symbol is outer versus inner.

    Input structure:
    - The input is a 9x9 grid with one background color and one foreground color.
    - It is partitioned into nine 3x3 blocks that should encode a nested product of two
      different 3x3 binary symbols A and B.
    - In the clean structure, each A=1 block contains a full copy of B and each A=0 block is empty,
      except that exactly one block has been corrupted.

    Rule:
    - Enumerate the candidate inner symbol B and the candidate damaged block so that the other
      eight blocks are all exactly either empty or a full B copy.
    - Keep only the unique explanation whose outer 3x3 occupancy A is different from B and whose
      damaged block truly disagrees with its expected clean state.
    - Output the repaired swapped nesting: place a full A copy in every B=1 block and leave every
      B=0 block empty.

    Key insight:
    - The repeated intact 3x3 copies reveal B, while the 3x3 arrangement of empty-versus-filled
      blocks reveals A; once the unique damaged block is identified, the task is to reverse the
      nesting order from A[B] to B[A].
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIDE, GRID_SIDE), "expected a 9x9 grid"

    bg = _get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    mask_grid = g == fg
    candidates = _find_candidates(mask_grid)
    assert len(candidates) == 1, "expected a unique nested-symbol explanation"
    chosen = candidates[0]
    return _render_nested(chosen["B"], chosen["A"], fg)


def _choose_variant(base: np.ndarray, branch: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    ones = [tuple(map(int, rc)) for rc in np.argwhere(base)]
    diff_rc = random.choice(ones)
    A = base.copy()
    B = base.copy()
    if branch == "empty":
        A[diff_rc] = False
    else:
        B[diff_rc] = False
    return A, B, diff_rc


def _anomalous_tile(A: np.ndarray, B: np.ndarray, branch: str) -> np.ndarray | None:
    bad = []
    for candidate in DAMAGE_MASKS:
        if np.array_equal(candidate, A) or np.array_equal(candidate, B):
            continue
        if branch == "filled" and int(candidate.sum()) == 0:
            continue
        bad.append(candidate)
    if not bad:
        return None

    if branch == "filled":
        scored = []
        for candidate in bad:
            overlap = int((candidate & B).sum())
            flips = int(np.count_nonzero(candidate ^ B))
            scored.append((flips, -overlap, candidate))
        scored.sort(key=lambda item: (item[0], item[1]))
        best = [candidate for flips, overlap, candidate in scored if flips == scored[0][0]]
        return random.choice(best).copy()

    return random.choice(bad).copy()


def _grid_from_masks(A: np.ndarray, B: np.ndarray, damage_rc: tuple[int, int], damaged_tile: np.ndarray, fg: int) -> np.ndarray:
    grid = _render_nested(A, B, fg)
    dr, dc = damage_rc
    r0 = dr * TILE_SIDE
    c0 = dc * TILE_SIDE
    grid[r0 : r0 + TILE_SIDE, c0 : c0 + TILE_SIDE] = np.where(damaged_tile, fg, BG)
    return grid


def _expected_output(A: np.ndarray, B: np.ndarray, fg: int) -> np.ndarray:
    return _render_nested(B, A, fg)


def _semantic_ok(grid: np.ndarray, A: np.ndarray, B: np.ndarray, damage_rc: tuple[int, int], branch: str, fg: int) -> bool:
    mask_grid = grid == fg
    candidates = _find_candidates(mask_grid)
    if len(candidates) != 1:
        return False
    cand = candidates[0]
    if not np.array_equal(cand["A"], A) or not np.array_equal(cand["B"], B):
        return False
    if cand["damage_idx"] != damage_rc[0] * 3 + damage_rc[1]:
        return False

    expected = _expected_output(A, B, fg)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, _render_nested(A, A, fg)):
        return False
    if np.array_equal(expected, _render_nested(B, B, fg)):
        return False

    observed_outer = np.array(
        [[mask_grid[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3].any() for bc in range(3)] for br in range(3)],
        dtype=bool,
    )
    if branch == "empty":
        if np.array_equal(observed_outer, A):
            return False
        if np.array_equal(expected, _render_nested(B, observed_outer, fg)):
            return False
    else:
        damaged_tile = mask_grid[damage_rc[0] * 3 : (damage_rc[0] + 1) * 3, damage_rc[1] * 3 : (damage_rc[1] + 1) * 3]
        if np.array_equal(damaged_tile, B) or np.array_equal(damaged_tile, np.zeros((3, 3), dtype=bool)):
            return False
        if np.array_equal(expected, _render_nested(A, damaged_tile, fg)):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    base = random.choice(SYMBOLS).copy()
    if int(base.sum()) < 4:
        return None

    branch = random.choice(["empty", "filled"])
    A, B, damage_rc = _choose_variant(base, branch)

    if int(A.sum()) < 3 or int(B.sum()) < 3:
        return None
    if int(A.sum()) > 6 or int(B.sum()) > 6:
        return None
    if np.array_equal(A, B):
        return None

    damaged_tile = _anomalous_tile(A, B, branch)
    if damaged_tile is None:
        return None

    fg = random.choice(FG_CHOICES)
    grid = _grid_from_masks(A, B, damage_rc, damaged_tile, fg)
    if np.count_nonzero(grid == fg) >= np.count_nonzero(grid == BG):
        return None
    if not _semantic_ok(grid, A, B, damage_rc, branch, fg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-symbol swap input")
