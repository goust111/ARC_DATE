from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
COPIES = 5
GENERATION_ATTEMPTS = 200

_GEN_TARGETS = (2, 4, 5, 6)
_GEN_CALL_COUNT = 0


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected True components using ARC's (r, c) convention."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        stack = [(rr, cc)]
        seen[rr, cc] = True
        while stack:
            cr, cc = stack.pop()
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
    return count


def _alternating_strip(tile: np.ndarray, copies: int = COPIES) -> np.ndarray:
    mirrored = np.fliplr(tile)
    blocks = [tile if i % 2 == 0 else mirrored for i in range(int(copies))]
    return np.concatenate(blocks, axis=1)


def _has_seam_contact(strip: np.ndarray, tile_w: int) -> bool:
    for seam in range(int(tile_w), strip.shape[1], int(tile_w)):
        if np.any((strip[:, seam - 1] == BLUE) & (strip[:, seam] == BLUE)):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the input five times with alternating left-right flips, count the blue components in the strip, and encode that count as a blue diagonal.

    Input structure:
    - The input is a small black-and-blue tile.
    - Blue pixels may form one or more 4-connected components.
    - Blue touches both the left and right tile edges, so mirrored neighbors can connect at seams.

    Rule:
    - Build a horizontal strip of five copies of the tile.
    - Copies 1, 3, and 5 stay unchanged; copies 2 and 4 are left-right mirrors of the input tile.
    - In that strip, count the number M of 4-connected blue components, allowing components to merge across tile seams when aligned edge pixels meet.
    - Return an MxM black grid whose main diagonal is blue.

    Key insight:
    - The needed count is an emergent property of the alternating tiled strip, not of the single tile viewed in isolation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D tile"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE} and BLUE in colors, "expected a black-and-blue tile"
    assert np.any(g[:, 0] == BLUE) and np.any(g[:, -1] == BLUE), "blue must touch both side edges"

    strip = _alternating_strip(g, copies=COPIES)
    m = _component_count(strip == BLUE)
    out = np.zeros((m, m), dtype=int)
    np.fill_diagonal(out, BLUE)
    return out


def _maybe_flip_vertical(tile: np.ndarray) -> np.ndarray:
    return np.flipud(tile).copy() if random.random() < 0.5 else tile


def _make_target_2() -> np.ndarray:
    h = random.choice((4, 5))
    w = random.randint(4, 6)
    tile = np.zeros((h, w), dtype=int)

    bend = random.randint(1, w - 2)
    if random.random() < 0.5:
        tile[0, : bend + 1] = BLUE
        tile[1, bend:] = BLUE
    else:
        tile[1, : bend + 1] = BLUE
        tile[0, bend:] = BLUE
    tile[h - 1, :] = BLUE
    return _maybe_flip_vertical(tile)


def _make_target_4() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(4, 6)
    tile = np.zeros((h, w), dtype=int)

    tile[0, :] = BLUE
    length = random.randint(2, w - 2)
    if random.random() < 0.5:
        tile[h - 1, :length] = BLUE
    else:
        tile[h - 1, w - length :] = BLUE
    return _maybe_flip_vertical(tile)


def _make_target_5() -> np.ndarray:
    w = random.randint(4, 6)
    tile = np.zeros((5, w), dtype=int)

    tile[0, :] = BLUE
    tile[2, :] = BLUE
    length = random.randint(2, w - 2)
    if random.random() < 0.5:
        tile[4, :length] = BLUE
    else:
        tile[4, w - length :] = BLUE
    return _maybe_flip_vertical(tile)


def _make_target_6() -> np.ndarray:
    h = random.choice((3, 5))
    w = random.randint(4, 6)
    tile = np.zeros((h, w), dtype=int)

    left_len = random.randint(2, w - 2)
    right_len = random.randint(2, w - 2)
    if random.random() < 0.5:
        tile[0, :left_len] = BLUE
        tile[h - 1, w - right_len :] = BLUE
    else:
        tile[0, w - right_len :] = BLUE
        tile[h - 1, :left_len] = BLUE
    return _maybe_flip_vertical(tile)


def _expected_diagonal(m: int) -> np.ndarray:
    out = np.zeros((int(m), int(m)), dtype=int)
    np.fill_diagonal(out, BLUE)
    return out


def _validate_tile(tile: np.ndarray, expected_m: int) -> bool:
    g = np.asarray(tile, dtype=int)
    h, w = g.shape
    if not (3 <= h <= 5 and 4 <= w <= 6):
        return False
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if set(map(int, np.unique(g))) - {BLACK, BLUE}:
        return False
    if not np.any(g == BLUE):
        return False
    if np.array_equal(g, np.fliplr(g)):
        return False
    if not np.any(g[:, 0] == BLUE) or not np.any(g[:, -1] == BLUE):
        return False

    tile_components = _component_count(g == BLUE)
    if tile_components <= 0:
        return False

    strip5 = _alternating_strip(g, copies=5)
    count5 = _component_count(strip5 == BLUE)
    if count5 != int(expected_m) or not (2 <= count5 <= 6):
        return False
    if count5 >= 5 * tile_components:
        return False
    if not _has_seam_contact(strip5, w):
        return False

    count3 = _component_count(_alternating_strip(g, copies=3) == BLUE)
    if expected_m != 2 and count3 == count5:
        return False

    return np.array_equal(main(g), _expected_diagonal(expected_m))


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    maker_by_target = {
        2: _make_target_2,
        4: _make_target_4,
        5: _make_target_5,
        6: _make_target_6,
    }
    start = _GEN_CALL_COUNT % len(_GEN_TARGETS)
    _GEN_CALL_COUNT += 1
    target_order = list(_GEN_TARGETS[start:]) + list(_GEN_TARGETS[:start])

    for target in target_order:
        maker = maker_by_target[int(target)]
        for _ in range(GENERATION_ATTEMPTS):
            tile = maker()
            if _validate_tile(tile, int(target)):
                return tile
    raise RuntimeError("failed to generate a valid alternating-strip tile")
