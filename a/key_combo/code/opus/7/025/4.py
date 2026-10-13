from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
TILE_SIZE = 3
STACKED_SHAPE = (9, 3)
OUTPUT_SHAPE = (9, 9)
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _is_main_diagonal_symmetric(tile: np.ndarray) -> bool:
    return bool(np.array_equal(tile, tile.T))


def _split_tiles(grid: np.ndarray) -> list[np.ndarray]:
    assert tuple(grid.shape) == STACKED_SHAPE, f"expected shape {STACKED_SHAPE}, got {grid.shape}"
    return [grid[r : r + TILE_SIZE, :] for r in range(0, STACKED_SHAPE[0], TILE_SIZE)]


def _binary_mask(tile: np.ndarray) -> np.ndarray:
    colors = set(map(int, np.unique(tile)))
    assert colors - {BLACK}, "selected tile must contain a foreground color"
    assert colors <= {BLACK, *NON_BLACK_COLORS}, f"unexpected colors: {colors}"
    assert len(colors - {BLACK}) == 1, "selected tile must use exactly one non-black foreground color"
    return tile != BLACK


def _output_from_mask(mask: np.ndarray) -> np.ndarray:
    assert mask.shape == (TILE_SIZE, TILE_SIZE)
    assert mask.dtype == bool
    template = np.where(mask, GREEN, BLACK).astype(int)
    # Background cells of the selected tile receive the green template;
    # foreground cells become all-black blocks.
    return np.kron((~mask).astype(int), template).astype(int)


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(int).ravel())


def _enumerate_masks(require_symmetric: bool) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (TILE_SIZE * TILE_SIZE)):
        mask = np.array([(bits >> i) & 1 for i in range(TILE_SIZE * TILE_SIZE)], dtype=bool).reshape(TILE_SIZE, TILE_SIZE)
        fg_count = int(mask.sum())
        bg_count = mask.size - fg_count
        if fg_count < 2 or bg_count < 2:
            continue
        if _is_main_diagonal_symmetric(mask) != require_symmetric:
            continue
        masks.append(mask)
    return masks


SYMMETRIC_MASKS = _enumerate_masks(require_symmetric=True)
ASYMMETRIC_MASKS = _enumerate_masks(require_symmetric=False)
SYMMETRIC_MASKS_BY_COUNT: dict[int, list[np.ndarray]] = {}
ASYMMETRIC_MASKS_BY_COUNT: dict[int, list[np.ndarray]] = {}
for _mask in SYMMETRIC_MASKS:
    SYMMETRIC_MASKS_BY_COUNT.setdefault(int(_mask.sum()), []).append(_mask)
for _mask in ASYMMETRIC_MASKS:
    ASYMMETRIC_MASKS_BY_COUNT.setdefault(int(_mask.sum()), []).append(_mask)

_SEEN_OUTPUT_SIGNATURES: set[tuple[int, ...]] = set()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the unique diagonally asymmetric tile into a self-product of its own foreground/background mask.

    Input structure:
    - The input is a 9x3 grid formed by three vertically stacked 3x3 tiles.
    - Every tile uses black and exactly one non-black foreground color.
    - Exactly two tiles are symmetric across the main diagonal, and exactly one is not.

    Rule:
    - Split the input into the three 3x3 tiles and select the only tile that is not equal to its transpose.
    - Convert that tile into a binary mask where foreground cells are 1 and black cells are 0.
    - Build a 9x9 output by replacing each foreground mask cell with an all-black 3x3 block, and each background mask cell with a green rendering of the same mask.

    Key insight:
    - The chosen tile serves two coupled roles at once: its asymmetry identifies it, and its own foreground/background pattern is the kernel that gets expanded.
    """
    g = np.asarray(grid, dtype=int)
    tiles = _split_tiles(g)
    asymmetric = [tile for tile in tiles if not _is_main_diagonal_symmetric(tile)]
    assert len(asymmetric) == 1, "expected exactly one asymmetric tile"
    mask = _binary_mask(asymmetric[0])
    return _output_from_mask(mask)


def _tile_from_mask(mask: np.ndarray, fg: int) -> np.ndarray:
    return np.where(mask, int(fg), BLACK).astype(int)


def _choose_asymmetric_mask() -> np.ndarray:
    unseen = [mask for mask in ASYMMETRIC_MASKS if _mask_signature(mask) not in _SEEN_OUTPUT_SIGNATURES]
    pool = unseen if unseen else ASYMMETRIC_MASKS
    return random.choice(pool)


def _choose_symmetric_masks(target_fg_count: int, force_count_match: bool) -> list[np.ndarray]:
    chosen: list[np.ndarray] = []
    used: set[tuple[int, ...]] = set()

    if force_count_match and target_fg_count in SYMMETRIC_MASKS_BY_COUNT:
        candidate = random.choice(SYMMETRIC_MASKS_BY_COUNT[target_fg_count])
        chosen.append(candidate)
        used.add(_mask_signature(candidate))

    while len(chosen) < 2:
        candidate = random.choice(SYMMETRIC_MASKS)
        sig = _mask_signature(candidate)
        if sig in used:
            continue
        chosen.append(candidate)
        used.add(sig)

    return chosen


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    tiles = _split_tiles(grid)
    asymmetric_tiles = [tile for tile in tiles if not _is_main_diagonal_symmetric(tile)]
    assert len(asymmetric_tiles) == 1
    return _output_from_mask(_binary_mask(asymmetric_tiles[0]))


def _validate_grid(grid: np.ndarray) -> tuple[bool, np.ndarray | None]:
    try:
        tiles = _split_tiles(grid)
        for tile in tiles:
            colors = set(map(int, np.unique(tile)))
            if len(colors) != 2 or BLACK not in colors or len(colors - {BLACK}) != 1:
                return False, None
        asym_indices = [idx for idx, tile in enumerate(tiles) if not _is_main_diagonal_symmetric(tile)]
        if len(asym_indices) != 1:
            return False, None
        asym_tile = tiles[asym_indices[0]]
        mask = _binary_mask(asym_tile)
        if int(mask.sum()) < 2 or int((~mask).sum()) < 2:
            return False, None

        expected = _oracle_output(grid)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False, None
        if actual.shape != OUTPUT_SHAPE:
            return False, None
        if not np.any(actual == GREEN):
            return False, None
        if not np.any(actual == BLACK):
            return False, None

        # Every 3x3 block must be either all black or an exact green template.
        template = np.where(mask, GREEN, BLACK).astype(int)
        for i in range(TILE_SIZE):
            for j in range(TILE_SIZE):
                block = actual[i * TILE_SIZE : (i + 1) * TILE_SIZE, j * TILE_SIZE : (j + 1) * TILE_SIZE]
                is_black_block = bool(np.all(block == BLACK))
                is_template_block = bool(np.array_equal(block, template))
                if not (is_black_block or is_template_block):
                    return False, None

        return True, expected
    except Exception:
        return False, None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        asym_mask = _choose_asymmetric_mask()
        force_count_match = bool(random.getrandbits(1))
        symmetric_masks = _choose_symmetric_masks(int(asym_mask.sum()), force_count_match)

        all_masks: list[np.ndarray] = symmetric_masks[:]
        asym_slot = random.randrange(3)
        all_masks.insert(asym_slot, asym_mask)

        colors = random.sample(NON_BLACK_COLORS, 3)
        tiles = [_tile_from_mask(mask, fg) for mask, fg in zip(all_masks, colors)]
        grid = np.vstack(tiles).astype(int)

        ok, out = _validate_grid(grid)
        if not ok or out is None:
            continue

        out_sig = _mask_signature(_binary_mask(tiles[asym_slot]))
        _SEEN_OUTPUT_SIGNATURES.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
