from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox as _arc_bbox
    from arc_utils import crop_bbox as _arc_crop_bbox
    from arc_utils import paste as _arc_paste
except Exception:
    _arc_bbox = None
    _arc_crop_bbox = None
    _arc_paste = None


BG = 0
KEEP_MARKER = 2
SWAP_MARKER = 8
SOURCE_PALETTE = (1, 3, 4, 5, 6, 7, 9)
GRID_SIDE_RANGE = (9, 15)
MAX_ATTEMPTS = 400

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

VERTICAL_POSITIONS = ((0, 1), (2, 1))
HORIZONTAL_POSITIONS = ((1, 0), (1, 2))
CORNER_POSITIONS = ((0, 0), (0, 2), (2, 0), (2, 2))
ALL_POSITIONS = tuple((r, c) for r in range(3) for c in range(3))


def bbox(mask: np.ndarray) -> BBox | None:
    if _arc_bbox is not None:
        return _arc_bbox(mask)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    if _arc_crop_bbox is not None:
        return _arc_crop_bbox(grid, box)
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    if _arc_paste is not None:
        return _arc_paste(dst, patch, r0, c0)
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def _boxes_separated(a: Coord, b: Coord, pad: int = 1) -> bool:
    ar0, ac0 = map(int, a)
    br0, bc0 = map(int, b)
    ar1, ac1 = ar0 + 2, ac0 + 2
    br1, bc1 = br0 + 2, bc0 + 2
    return bool(
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _swap_colors(tile: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = tile.copy()
    mask_a = out == int(color_a)
    mask_b = out == int(color_b)
    out[mask_a] = int(color_b)
    out[mask_b] = int(color_a)
    return out


def _tile_signature(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in tile.flat)


def _reflect_tile(tile: np.ndarray, slot_r: int, slot_c: int) -> np.ndarray:
    out = tile
    if int(slot_r) != 1:
        out = out[::-1, :]
    if int(slot_c) != 1:
        out = out[:, ::-1]
    return out


def _build_output(
    source: np.ndarray,
    key: np.ndarray,
    *,
    enable_reflection: bool = True,
    enable_swap: bool = True,
) -> np.ndarray:
    colors = sorted(set(map(int, source.flat)))
    assert len(colors) == 2
    color_a, color_b = colors
    out = np.zeros((9, 9), dtype=int)

    for slot_r in range(3):
        for slot_c in range(3):
            marker = int(key[slot_r, slot_c])
            if marker == BG:
                continue
            assert marker in {KEEP_MARKER, SWAP_MARKER}
            tile = source.copy()
            if enable_reflection:
                tile = _reflect_tile(tile, slot_r, slot_c)
            if enable_swap and marker == SWAP_MARKER:
                tile = _swap_colors(tile, color_a, color_b)
            paste(out, tile, slot_r * 3, slot_c * 3)

    return out


def _find_source_candidates(grid: np.ndarray) -> list[BBox]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[BBox] = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(patch == BG):
                continue
            if len(set(map(int, patch.flat))) != 2:
                continue

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + 4)
            cc1 = min(w, c0 + 4)
            halo = g[rr0:rr1, cc0:cc1].copy()
            halo[r0 - rr0 : r0 - rr0 + 3, c0 - cc0 : c0 - cc0 + 3] = BG
            if np.any(halo != BG):
                continue

            candidates.append((r0, c0, r0 + 2, c0 + 2))

    return candidates


def _extract_parts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source_candidates = _find_source_candidates(g)
    assert len(source_candidates) == 1, "expected exactly one source-tile candidate"
    source_box = source_candidates[0]
    source = crop_bbox(g, source_box)
    assert source.shape == (3, 3)
    source_colors = set(map(int, source.flat))
    assert BG not in source_colors and len(source_colors) == 2

    remaining = g != BG
    sr0, sc0, sr1, sc1 = source_box
    remaining[sr0 : sr1 + 1, sc0 : sc1 + 1] = False
    key_box = bbox(remaining)
    assert key_box is not None, "expected a separate key grid"
    kr0, kc0, kr1, kc1 = key_box
    assert (kr1 - kr0 + 1, kc1 - kc0 + 1) == (3, 3), "key bbox must span exactly 3x3"

    outside = remaining.copy()
    outside[kr0 : kr1 + 1, kc0 : kc1 + 1] = False
    assert not outside.any(), "input should contain only the tile and the key grid"
    assert _boxes_separated((sr0, sc0), (kr0, kc0), pad=1)

    key = crop_bbox(g, key_box)
    assert key.shape == (3, 3)
    key_colors = set(map(int, key.flat)) - {BG}
    assert key_colors == {KEEP_MARKER, SWAP_MARKER}, "key must use the two visible marker colors"
    assert np.any(key == BG), "key must contain black slots"
    return source, key


def _tile_has_all_variants(tile: np.ndarray) -> bool:
    colors = sorted(set(map(int, tile.flat)))
    if len(colors) != 2:
        return False
    counts = [int(np.count_nonzero(tile == color)) for color in colors]
    if min(counts) < 3:
        return False

    sigs: set[tuple[int, ...]] = set()
    for slot_r, slot_c in ((1, 1), (0, 1), (1, 0), (0, 0)):
        reflected = _reflect_tile(tile, slot_r, slot_c)
        sigs.add(_tile_signature(reflected))
        sigs.add(_tile_signature(_swap_colors(reflected, colors[0], colors[1])))
    return len(sigs) == 8


def _key_is_semantic(key: np.ndarray) -> bool:
    if key.shape != (3, 3):
        return False
    if np.count_nonzero(key == BG) < 1:
        return False
    if int(key[0, 1]) + int(key[2, 1]) != KEEP_MARKER + SWAP_MARKER:
        return False
    if int(key[1, 0]) + int(key[1, 2]) != KEEP_MARKER + SWAP_MARKER:
        return False
    if not any(int(key[r, c]) in {KEEP_MARKER, SWAP_MARKER} for r, c in CORNER_POSITIONS):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the 3x3 output slots selected by a key grid with mirrored copies of a separate source tile.

    Input structure:
    - The input contains two separated 3x3 regions on a black background.
    - One region is a full source tile with exactly two non-black colors and no black cells.
    - The other region is a key grid whose black cells mean empty slots and whose two marker colors are the fixed visible keep/swap markers.

    Rule:
    - Produce a 9x9 output split into nine 3x3 slots corresponding to the key-grid cells.
    - Black key cells leave their slots black.
    - Marked slots receive the source tile reflected only by the slot position: center keeps the tile, top/bottom reverse rows, left/right reverse columns, and corners reverse both.
    - If the marker is the swap marker color, swap the tile's two colors after reflection.

    Key insight:
    - The key grid carries two independent instructions at once: position chooses the mirror orientation, while marker color chooses whether the mirrored tile keeps or swaps its two colors.
    """
    source, key = _extract_parts(np.asarray(grid, dtype=int))
    return _build_output(source, key)


def _random_source_tile() -> np.ndarray:
    while True:
        color_a, color_b = random.sample(SOURCE_PALETTE, 2)
        count_a = random.randint(3, 6)
        picks = set(random.sample(range(9), count_a))
        mask = np.array([i in picks for i in range(9)], dtype=bool).reshape(3, 3)
        tile = np.where(mask, int(color_a), int(color_b)).astype(int)
        if _tile_has_all_variants(tile):
            return tile


def _random_key_grid() -> np.ndarray:
    key = np.zeros((3, 3), dtype=int)

    vertical = [KEEP_MARKER, SWAP_MARKER]
    horizontal = [KEEP_MARKER, SWAP_MARKER]
    random.shuffle(vertical)
    random.shuffle(horizontal)
    key[0, 1], key[2, 1] = vertical
    key[1, 0], key[1, 2] = horizontal

    corner = random.choice(CORNER_POSITIONS)
    key[corner] = random.choice((KEEP_MARKER, SWAP_MARKER))

    if random.random() < 0.7:
        key[1, 1] = random.choice((KEEP_MARKER, SWAP_MARKER))

    remaining = [pos for pos in ALL_POSITIONS if int(key[pos]) == BG]
    extra_cap = min(2, max(0, len(remaining) - 1))
    extra_count = random.randint(0, extra_cap)
    for pos in random.sample(remaining, extra_count):
        key[pos] = random.choice((KEEP_MARKER, SWAP_MARKER))

    assert _key_is_semantic(key)
    return key


def _placement_pair(height: int, width: int) -> tuple[Coord, Coord] | None:
    anchors = [
        (r, c)
        for r in range(1, height - 3)
        for c in range(1, width - 3)
    ]
    pairs = [
        (a, b)
        for i, a in enumerate(anchors)
        for b in anchors[i + 1 :]
        if _boxes_separated(a, b, pad=1)
    ]
    if not pairs:
        return None
    return random.choice(pairs)


def _compose_input(source: np.ndarray, key: np.ndarray) -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    pair = _placement_pair(height, width)
    if pair is None:
        return None

    first, second = pair
    if random.random() < 0.5:
        source_pos, key_pos = first, second
    else:
        source_pos, key_pos = second, first

    grid = np.zeros((height, width), dtype=int)
    paste(grid, source, source_pos[0], source_pos[1])
    paste(grid, key, key_pos[0], key_pos[1])
    return grid


def _generated_input_is_valid(grid: np.ndarray) -> bool:
    try:
        source, key = _extract_parts(grid)
        out = _build_output(source, key)
    except Exception:
        return False

    if not _tile_has_all_variants(source):
        return False
    if not _key_is_semantic(key):
        return False
    if not np.array_equal(main(grid), out):
        return False

    if np.array_equal(out, _build_output(source, key, enable_swap=False)):
        return False
    if np.array_equal(out, _build_output(source, key, enable_reflection=False)):
        return False

    filled_key = np.where(key == BG, KEEP_MARKER, key)
    if np.array_equal(out, _build_output(source, filled_key)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        source = _random_source_tile()
        key = _random_key_grid()
        grid = _compose_input(source, key)
        if grid is not None and _generated_input_is_valid(grid):
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
