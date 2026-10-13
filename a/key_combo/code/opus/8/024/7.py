from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[counts.argmax()])


BG = 0
FG = 1
SLOT_SIZE = 5
GAP = 1
SLOTS_PER_AXIS = 4
ARRAY_SPAN = SLOTS_PER_AXIS * SLOT_SIZE + (SLOTS_PER_AXIS - 1) * GAP
STEP = SLOT_SIZE + GAP
MAX_GENERATE_ATTEMPTS = 400
MIN_CORRUPT_SLOTS = 2
MAX_CORRUPT_SLOTS = 4
MIN_NOISE = 3
MAX_NOISE = 7

BASE_TWO_SEED = np.array(
    [
        [1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0],
    ],
    dtype=int,
)

BASE_ONE_SEED = np.array(
    [
        [0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 0, 0],
    ],
    dtype=int,
)

SPRITE_LIBRARY = tuple(
    np.rot90(base, k).astype(int)
    for base in (BASE_TWO_SEED, BASE_ONE_SEED)
    for k in range(4)
)


def _band_cells(start: int) -> list[int]:
    rows: list[int] = []
    for i in range(SLOTS_PER_AXIS):
        band_start = int(start + i * STEP)
        rows.extend(range(band_start, band_start + SLOT_SIZE))
    return rows


def _gap_cells(start: int) -> list[int]:
    rows: list[int] = []
    for i in range(SLOTS_PER_AXIS - 1):
        rows.append(int(start + SLOT_SIZE + i * STEP))
    return rows


def _find_array_start(counts: np.ndarray) -> int:
    counts = np.asarray(counts, dtype=int)
    best_score = None
    best_starts: list[int] = []
    for start in range(0, len(counts) - ARRAY_SPAN + 1):
        band_idx = _band_cells(start)
        gap_idx = _gap_cells(start)
        score = int(counts[band_idx].sum()) * 3 - int(counts[gap_idx].sum())
        if best_score is None or score > best_score:
            best_score = score
            best_starts = [start]
        elif score == best_score:
            best_starts.append(start)
    assert best_score is not None and len(best_starts) == 1
    return int(best_starts[0])


def _slot_starts(start: int) -> list[int]:
    return [int(start + i * STEP) for i in range(SLOTS_PER_AXIS)]


def _slot_mask(shape: tuple[int, int], row_start: int, col_start: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0 in _slot_starts(row_start):
        for c0 in _slot_starts(col_start):
            mask[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = True
    return mask


def _extract_blocks(grid: np.ndarray, row_start: int, col_start: int) -> list[np.ndarray]:
    blocks = []
    for r0 in _slot_starts(row_start):
        for c0 in _slot_starts(col_start):
            blocks.append(np.asarray(grid[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE], dtype=int))
    return blocks


def _majority_sprite(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    stack = np.stack(blocks, axis=0)
    fg_counts = np.count_nonzero(stack != int(bg), axis=0)
    sprite = np.full((SLOT_SIZE, SLOT_SIZE), int(bg), dtype=int)
    sprite[fg_counts > len(blocks) // 2] = FG
    return sprite


def _find_anchor(sprite: np.ndarray) -> tuple[int, int]:
    hits = []
    for r in range(SLOT_SIZE - 1):
        for c in range(SLOT_SIZE - 1):
            if np.all(sprite[r : r + 2, c : c + 2] == FG):
                hits.append((r, c))
    assert len(hits) == 1, "expected exactly one 2x2 anchor square"
    return hits[0]


def _seed_directions(sprite: np.ndarray, anchor_r: int, anchor_c: int) -> list[tuple[int, int, int, int]]:
    seeds: list[tuple[int, int, int, int]] = []
    candidates = [
        (anchor_r - 1, anchor_c - 1, -1, -1),
        (anchor_r - 1, anchor_c + 2, -1, 1),
        (anchor_r + 2, anchor_c - 1, 1, -1),
        (anchor_r + 2, anchor_c + 2, 1, 1),
    ]
    for seed_r, seed_c, dr, dc in candidates:
        if 0 <= seed_r < SLOT_SIZE and 0 <= seed_c < SLOT_SIZE and int(sprite[seed_r, seed_c]) == FG:
            seeds.append((seed_r, seed_c, dr, dc))
    assert 1 <= len(seeds) <= 2, "expected one or two diagonal seeds"
    return seeds


def _extend_slot_rays(sprite: np.ndarray) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    anchor_r, anchor_c = _find_anchor(out)
    for seed_r, seed_c, dr, dc in _seed_directions(out, anchor_r, anchor_c):
        r = seed_r + dr
        c = seed_c + dc
        while 0 <= r < SLOT_SIZE and 0 <= c < SLOT_SIZE:
            out[r, c] = FG
            r += dr
            c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the repeated anchor sprite by majority vote, then draw its slot-confined diagonal rays and erase outside noise.

    Input structure:
    - The background is black and the foreground is blue.
    - Blue cells form a 4x4 array of 5x5 sprite slots separated by 1-cell black gaps, plus sparse blue noise outside the slot array.
    - Each slot contains a corrupted copy of one shared sprite whose true form includes a unique solid 2x2 square and one or two diagonal seed pixels.

    Rule:
    - Detect the dense row bands and dense column bands of the slot array to locate the 16 slots.
    - For each relative cell inside the 5x5 slot, take the majority color across the 16 slot copies to recover the standard sprite.
    - Find the unique 2x2 anchor square in that recovered sprite, read any diagonal seed pixels attached to its corners, replace every slot with the recovered sprite, and extend each seed along the same diagonal until the slot boundary.
    - Set every cell outside the slot array to black.

    Key insight:
    - Individual slots may have broken anchors or seeds, so the anchor location and ray directions are reliable only after majority-vote recovery across the full slot array.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    fg_mask = g != bg

    row_start = _find_array_start(fg_mask.sum(axis=1))
    col_start = _find_array_start(fg_mask.sum(axis=0))
    blocks = _extract_blocks(g, row_start, col_start)
    sprite = _majority_sprite(blocks, bg)
    solved_sprite = _extend_slot_rays(sprite)

    out = np.full_like(g, bg)
    for r0 in _slot_starts(row_start):
        for c0 in _slot_starts(col_start):
            out[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = solved_sprite
    return out


def _latent_seed_cells(sprite: np.ndarray) -> list[tuple[int, int]]:
    anchor_r, anchor_c = _find_anchor(sprite)
    return [(r, c) for r, c, _dr, _dc in _seed_directions(sprite, anchor_r, anchor_c)]


def _oracle_output(shape: tuple[int, int], row_start: int, col_start: int, sprite: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    solved_sprite = _extend_slot_rays(sprite)
    for r0 in _slot_starts(row_start):
        for c0 in _slot_starts(col_start):
            out[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = solved_sprite
    return out


def _outside_positions(shape: tuple[int, int], row_start: int, col_start: int) -> list[tuple[int, int]]:
    r0 = int(row_start)
    c0 = int(col_start)
    r1 = r0 + ARRAY_SPAN
    c1 = c0 + ARRAY_SPAN
    cells = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if not (r0 <= r < r1 and c0 <= c < c1):
                cells.append((r, c))
    return cells


def _apply_corruption(
    slot: np.ndarray,
    rng: np.random.Generator,
    sprite: np.ndarray,
    require_critical: bool,
) -> tuple[np.ndarray, bool]:
    out = np.asarray(slot, dtype=int).copy()
    anchor_r, anchor_c = _find_anchor(sprite)
    critical = {
        (anchor_r, anchor_c),
        (anchor_r, anchor_c + 1),
        (anchor_r + 1, anchor_c),
        (anchor_r + 1, anchor_c + 1),
        *_latent_seed_cells(sprite),
    }
    ones = {tuple(map(int, rc)) for rc in np.argwhere(sprite == FG)}
    zeros = {tuple(map(int, rc)) for rc in np.argwhere(sprite == BG)}
    touched_critical = False
    mods_left = int(rng.integers(1, 3))

    if require_critical:
        removable = [pos for pos in sorted(critical) if pos in ones and out[pos] == FG]
        assert removable
        pos = removable[int(rng.integers(len(removable)))]
        out[pos] = BG
        touched_critical = True
        mods_left -= 1

    for _ in range(max(0, mods_left)):
        if rng.random() < 0.65 and any(out[pos] == FG for pos in ones):
            removable = [pos for pos in sorted(ones) if out[pos] == FG]
            pos = removable[int(rng.integers(len(removable)))]
            out[pos] = BG
            if pos in critical:
                touched_critical = True
        else:
            addable = [pos for pos in sorted(zeros) if out[pos] == BG]
            pos = addable[int(rng.integers(len(addable)))]
            out[pos] = FG

    return out, touched_critical


def _build_input(
    shape: tuple[int, int],
    row_start: int,
    col_start: int,
    sprite: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    slot_origins = [(r0, c0) for r0 in _slot_starts(row_start) for c0 in _slot_starts(col_start)]

    for r0, c0 in slot_origins:
        inp[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = sprite

    corrupt_count = int(rng.integers(MIN_CORRUPT_SLOTS, MAX_CORRUPT_SLOTS + 1))
    corrupt_ids = rng.choice(len(slot_origins), size=corrupt_count, replace=False).tolist()
    touched_critical = False
    for i, idx in enumerate(corrupt_ids):
        r0, c0 = slot_origins[int(idx)]
        corrupted, hit = _apply_corruption(
            inp[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE],
            rng,
            sprite,
            require_critical=(i == 0),
        )
        inp[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = corrupted
        touched_critical |= hit

    assert touched_critical
    assert any(
        not np.array_equal(
            inp[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE],
            sprite,
        )
        for r0, c0 in slot_origins
    )

    outside = _outside_positions(shape, row_start, col_start)
    noise_count = int(rng.integers(MIN_NOISE, MAX_NOISE + 1))
    chosen = rng.choice(len(outside), size=noise_count, replace=False).tolist()
    for idx in chosen:
        r, c = outside[int(idx)]
        inp[r, c] = FG

    expected = _oracle_output(shape, row_start, col_start, sprite)
    return inp, expected


def _validate_sprite(sprite: np.ndarray) -> None:
    assert sprite.shape == (SLOT_SIZE, SLOT_SIZE)
    assert np.all((sprite == BG) | (sprite == FG))
    assert np.all(sprite.sum(axis=0) >= 1)
    assert np.all(sprite.sum(axis=1) >= 1)
    _find_anchor(sprite)
    seeds = _latent_seed_cells(sprite)
    assert 1 <= len(seeds) <= 2
    solved = _extend_slot_rays(sprite)
    assert np.count_nonzero(solved) > np.count_nonzero(sprite)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sprite = np.asarray(SPRITE_LIBRARY[int(rng.integers(len(SPRITE_LIBRARY)))], dtype=int)
    _validate_sprite(sprite)

    while True:
        top = int(rng.integers(0, 4))
        bottom = int(rng.integers(0, 4))
        left = int(rng.integers(0, 4))
        right = int(rng.integers(0, 4))
        if top + bottom + left + right > 0:
            break

    shape = (ARRAY_SPAN + top + bottom, ARRAY_SPAN + left + right)
    row_start = top
    col_start = left
    inp, expected = _build_input(shape, row_start, col_start, sprite, rng)

    slot_mask = _slot_mask(shape, row_start, col_start)
    if np.any(expected[~slot_mask] != BG):
        return None

    blocks = _extract_blocks(inp, row_start, col_start)
    recovered = _majority_sprite(blocks, BG)
    if not np.array_equal(recovered, sprite):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
