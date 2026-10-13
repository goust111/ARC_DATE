from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

GEN_RETRY_LIMIT = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    top = counts.most_common()
    assert len(top) >= 1
    if len(top) >= 2:
        assert top[0][1] > top[1][1], f"majority must be unique, got {top}"
    return int(top[0][0])


def _separator_layout(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, int]:
    sep_box = _bbox(grid == GREY)
    assert sep_box is not None, "expected one grey separator cross"
    sep_crop = _crop(grid, sep_box)
    full_rows = [r for r in range(sep_crop.shape[0]) if np.all(sep_crop[r, :] == GREY)]
    full_cols = [c for c in range(sep_crop.shape[1]) if np.all(sep_crop[:, c] == GREY)]
    assert len(full_rows) == 1 and len(full_cols) == 1, "expected one separator row and one separator column"
    sep_r = full_rows[0]
    sep_c = full_cols[0]
    expected = np.zeros_like(sep_crop, dtype=bool)
    expected[sep_r, :] = True
    expected[:, sep_c] = True
    assert np.array_equal(sep_crop == GREY, expected), "grey pixels must form a clean cross"
    assert sep_crop.shape[0] == 2 * sep_r + 1, "separator row must split the block lattice into equal heights"
    assert sep_crop.shape[1] == 2 * sep_c + 1, "separator column must split the block lattice into equal widths"
    assert sep_r >= 2 and sep_c >= 2, "blocks must be at least 2x2"
    return sep_box, sep_r, sep_c


def _extract_tiles(
    grid: np.ndarray,
) -> tuple[list[np.ndarray], tuple[int, int, int, int], tuple[int, int]]:
    sep_box, tile_h, tile_w = _separator_layout(grid)
    r0, c0, _r1, _c1 = sep_box
    tiles = [
        grid[r0 : r0 + tile_h, c0 : c0 + tile_w],
        grid[r0 : r0 + tile_h, c0 + tile_w + 1 : c0 + 2 * tile_w + 1],
        grid[r0 + tile_h + 1 : r0 + 2 * tile_h + 1, c0 : c0 + tile_w],
        grid[r0 + tile_h + 1 : r0 + 2 * tile_h + 1, c0 + tile_w + 1 : c0 + 2 * tile_w + 1],
    ]
    assert len({tile.shape for tile in tiles}) == 1, "all lattice cells must be equal-sized"
    return tiles, sep_box, (tile_h, tile_w)


def _find_slot(grid: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    slot_box = _bbox(grid == BLUE)
    assert slot_box is not None, "expected one blue target slot"
    slot = _crop(grid, slot_box)
    assert slot.shape == shape, "target slot must match the tile size"
    assert np.all(slot == BLUE), "target slot must be a solid blue rectangle"
    return slot_box


def _recover_consensus(tiles: list[np.ndarray]) -> tuple[np.ndarray, int, np.ndarray]:
    tile_stack = np.stack(tiles, axis=0)
    tile_h, tile_w = tile_stack.shape[1:]
    motif = np.empty((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            motif[r, c] = _unique_mode(tile_stack[:, r, c])

    diff_masks = tile_stack != motif
    diff_counts = diff_masks.reshape(len(tiles), tile_h * tile_w).sum(axis=1)
    damaged = np.flatnonzero(diff_counts > 0)
    assert damaged.size == 1, "expected exactly one damaged tile"
    damaged_idx = int(damaged[0])
    damage_mask = diff_masks[damaged_idx]
    assert damage_mask.any(), "damaged tile must differ from the consensus"
    assert np.all(tile_stack[damaged_idx][damage_mask] == BLACK), "damage must overwrite with black"
    assert np.all(motif[damage_mask] == RED), "only red motif cells may be damaged"
    intact = [i for i in range(len(tiles)) if i != damaged_idx]
    for idx in intact:
        assert np.array_equal(tile_stack[idx], motif), "all non-damaged tiles must equal the consensus motif"
    return motif, damaged_idx, damage_mask


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    tiles, sep_box, tile_shape = _extract_tiles(g)
    motif, damaged_idx, damage_mask = _recover_consensus(tiles)
    slot_box = _find_slot(g, tile_shape)

    return {
        "grid": g,
        "tiles": tiles,
        "sep_box": sep_box,
        "tile_shape": tile_shape,
        "motif": motif,
        "damaged_idx": damaged_idx,
        "damage_mask": damage_mask,
        "slot_box": slot_box,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the consensus tile, turn the damaged coordinates into a mirrored stamp, and reveal it inside the blue slot.

    Input structure:
    - The grid has a black background with a grey cross that partitions the left side into a 2x2 lattice of equal tiles.
    - The four tiles are copies of one red-and-black motif, except exactly one copy has some red cells overwritten by black.
    - A solid blue rectangle on the right has the same shape as one tile and acts as the destination slot.

    Rule:
    - Take the unique per-coordinate majority across the four tiles to recover the consensus motif.
    - Find the lone tile that differs from the consensus and record its differing local coordinates.
    - Build a stamp that keeps the consensus color only at those damaged coordinates, mirror that stamp left-to-right, and paste its non-black cells over the blue slot.
    - Leave the entire left lattice, separator cross, and all other background cells unchanged.

    Key insight:
    - The repair step produces more than the recovered motif: the damaged coordinate set itself is the hidden shape that defines the mirrored stamp.
    """
    info = _analyze(grid)
    g = np.asarray(info["grid"], dtype=int)
    motif = np.asarray(info["motif"], dtype=int)
    damage_mask = np.asarray(info["damage_mask"], dtype=bool)
    slot_box = info["slot_box"]
    assert isinstance(slot_box, tuple)

    stamp = np.where(damage_mask, motif, BLACK)
    mirrored = np.fliplr(stamp)

    out = g.copy()
    r0, c0, r1, c1 = slot_box
    slot = out[r0 : r1 + 1, c0 : c1 + 1]
    slot[mirrored != BLACK] = mirrored[mirrored != BLACK]
    return out


def _damage_bbox_is_strict(mask: np.ndarray, tile_shape: tuple[int, int]) -> bool:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) < tile_shape[0] and (c1 - c0 + 1) < tile_shape[1]


def _is_horizontally_asymmetric(mask: np.ndarray) -> bool:
    return not np.array_equal(mask, np.fliplr(mask))


def _gen_motif(tile_h: int, tile_w: int) -> np.ndarray | None:
    area = tile_h * tile_w
    red_count = int(np.random.randint(max(6, area // 3), area - 1))
    motif = np.zeros((tile_h, tile_w), dtype=int)
    red_cells = np.random.choice(area, size=red_count, replace=False)
    motif.flat[red_cells] = RED

    if np.all(motif == RED) or np.all(motif == BLACK):
        return None
    if len(np.argwhere(motif == RED)) < 5:
        return None
    return motif


def _choose_damage(motif: np.ndarray) -> np.ndarray | None:
    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(motif == RED)]
    np.random.shuffle(red_coords)
    for k in np.random.permutation([2, 3]):
        if len(red_coords) < k:
            continue
        for _ in range(80):
            picks = np.random.choice(len(red_coords), size=k, replace=False)
            mask = np.zeros_like(motif, dtype=bool)
            for idx in picks:
                r, c = red_coords[int(idx)]
                mask[r, c] = True
            if not _damage_bbox_is_strict(mask, motif.shape):
                continue
            if not _is_horizontally_asymmetric(mask):
                continue
            return mask
    return None


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _build_grid(
    motif: np.ndarray,
    damage_mask: np.ndarray,
    damaged_idx: int,
    left_top: tuple[int, int],
    gap: int,
    slot_top: int,
    right_margin: int,
) -> np.ndarray:
    tile_h, tile_w = motif.shape
    left_h = 2 * tile_h + 1
    left_w = 2 * tile_w + 1
    left_r, left_c = left_top
    slot_c = left_c + left_w + gap
    height = max(left_r + left_h, slot_top + tile_h) + int(np.random.randint(0, 4))
    width = slot_c + tile_w + right_margin
    grid = np.zeros((height, width), dtype=int)

    sep_r = left_r + tile_h
    sep_c = left_c + tile_w
    grid[sep_r, left_c : left_c + left_w] = GREY
    grid[left_r : left_r + left_h, sep_c] = GREY

    tiles = [motif.copy() for _ in range(4)]
    tiles[damaged_idx][damage_mask] = BLACK
    placements = [
        (left_r, left_c),
        (left_r, left_c + tile_w + 1),
        (left_r + tile_h + 1, left_c),
        (left_r + tile_h + 1, left_c + tile_w + 1),
    ]
    for tile, (r0, c0) in zip(tiles, placements):
        grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile

    grid[slot_top : slot_top + tile_h, slot_c : slot_c + tile_w] = BLUE
    return grid


def _validate_instance(
    grid: np.ndarray,
    motif: np.ndarray,
    damage_mask: np.ndarray,
    damaged_idx: int,
) -> bool:
    info = _analyze(grid)
    recovered_motif = np.asarray(info["motif"], dtype=int)
    recovered_mask = np.asarray(info["damage_mask"], dtype=bool)
    recovered_damaged = int(info["damaged_idx"])
    slot_box = info["slot_box"]
    assert isinstance(slot_box, tuple)

    if not np.array_equal(recovered_motif, motif):
        return False
    if not np.array_equal(recovered_mask, damage_mask):
        return False
    if recovered_damaged != damaged_idx:
        return False
    if not _damage_bbox_is_strict(damage_mask, motif.shape):
        return False
    if not _is_horizontally_asymmetric(damage_mask):
        return False

    out = _expected_output(grid)
    r0, c0, r1, c1 = slot_box
    slot_before = grid[r0 : r1 + 1, c0 : c1 + 1]
    slot_after = out[r0 : r1 + 1, c0 : c1 + 1]
    mirrored_stamp = np.fliplr(np.where(damage_mask, motif, BLACK))
    expected_slot = slot_before.copy()
    expected_slot[mirrored_stamp != BLACK] = mirrored_stamp[mirrored_stamp != BLACK]
    if not np.array_equal(slot_after, expected_slot):
        return False

    sep_box = info["sep_box"]
    assert isinstance(sep_box, tuple)
    sr0, sc0, sr1, sc1 = sep_box
    if not np.array_equal(out[sr0 : sr1 + 1, sc0 : sc1 + 1], grid[sr0 : sr1 + 1, sc0 : sc1 + 1]):
        return False
    if np.array_equal(slot_after, slot_before):
        return False
    if np.array_equal(slot_after, np.where(damage_mask, motif, BLUE)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile_h = int(np.random.randint(4, 7))
    tile_w = int(np.random.randint(4, 7))
    motif = _gen_motif(tile_h, tile_w)
    if motif is None:
        return None

    damage_mask = _choose_damage(motif)
    if damage_mask is None:
        return None

    damaged_idx = int(np.random.randint(4))
    left_top = (int(np.random.randint(0, 4)), int(np.random.randint(0, 3)))
    gap = int(np.random.randint(2, 5))
    slot_top = int(np.random.randint(0, left_top[0] + 2 * tile_h + 2))
    right_margin = int(np.random.randint(0, 4))
    grid = _build_grid(motif, damage_mask, damaged_idx, left_top, gap, slot_top, right_margin)

    if max(grid.shape) > 30:
        return None
    if not _validate_instance(grid, motif, damage_mask, damaged_idx):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRY_LIMIT} attempts")
