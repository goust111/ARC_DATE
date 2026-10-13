from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


def _extract_layout(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    H, W = map(int, g.shape)
    divider_cols = [c for c in range(W) if bool(np.all(g[:, c] == GREY))]
    assert len(divider_cols) == 1, "expected exactly one full grey divider column"
    divider = int(divider_cols[0])
    assert 0 < divider < W - 1, "divider must split non-empty left and right regions"
    assert H % 3 == 0, "canvas height should be three times the mask height"

    left = g[:, :divider]
    right = g[:, divider + 1 :]
    mask_h = H // 3
    assert right.shape == (H, 2 * divider), "right canvas should be 3x2 tiles of the left mask size"

    sep_rows = [
        r
        for r in range(H)
        if bool(np.all(left[r] == YELLOW)) and r - mask_h >= 0 and r + mask_h < H
    ]
    assert len(sep_rows) == 1, "expected exactly one valid yellow separator row"
    sep = int(sep_rows[0])

    top = g[sep - mask_h : sep, :divider]
    bottom = g[sep + 1 : sep + 1 + mask_h, :divider]
    assert top.shape == (mask_h, divider) == bottom.shape
    assert bool(np.isin(top, (BLACK, RED)).all()), "top mask must be binary red/black"
    assert bool(np.isin(bottom, (BLACK, RED)).all()), "bottom mask must be binary red/black"

    return {
        "grid": g,
        "H": H,
        "W": W,
        "divider": divider,
        "sep": sep,
        "mask_h": mask_h,
        "mask_w": divider,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def _xor_seed(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return np.where((top == RED) != (bottom == RED), GREEN, BLACK)


def _alternate_mirror_fill(seed: np.ndarray) -> np.ndarray:
    rows = []
    for tile_r in range(3):
        row_tiles = []
        for tile_c in range(2):
            tile = seed
            if tile_c % 2 == 1:
                tile = np.fliplr(tile)
            if tile_r % 2 == 1:
                tile = np.flipud(tile)
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def _random_binary_mask(h: int, w: int, min_active: int, max_active: int) -> np.ndarray:
    area = h * w
    low = max(1, int(min_active))
    high = min(area - 1, int(max_active))
    assert low <= high
    active = random.randint(low, high)
    flat = np.array([True] * active + [False] * (area - active), dtype=bool)
    np.random.shuffle(flat)
    return flat.reshape(h, w)


def _axis_asymmetric(mask: np.ndarray) -> tuple[bool, bool]:
    return (not np.array_equal(mask, np.flipud(mask)), not np.array_equal(mask, np.fliplr(mask)))


def _seed_ok(mask: np.ndarray) -> bool:
    area = int(mask.size)
    ones = int(mask.sum())
    if ones < 2 or ones > area - 2:
        return False
    if ones == 0 or ones == area:
        return False
    asym_ud, asym_lr = _axis_asymmetric(mask)
    return asym_ud and asym_lr


def _generated_case_ok(grid: np.ndarray) -> None:
    layout = _extract_layout(grid)
    assert bool(np.all(layout["right"] == BLACK)), "input canvas must start black"

    top = np.asarray(layout["top"], dtype=int)
    bottom = np.asarray(layout["bottom"], dtype=int)
    seed = _xor_seed(top, bottom)
    seed_mask = seed == GREEN

    assert _seed_ok(seed_mask), "seed must visibly change under both mirror directions"
    assert int((top == RED).sum()) not in (0, top.size), "top mask should contain both states"
    assert int((bottom == RED).sum()) not in (0, bottom.size), "bottom mask should contain both states"
    assert not np.array_equal(top, bottom), "xor seed must be non-zero"

    expected = _alternate_mirror_fill(seed)
    assert expected.shape == np.asarray(layout["right"]).shape

    h = int(layout["mask_h"])
    w = int(layout["mask_w"])
    tiles = [
        [expected[r * h : (r + 1) * h, c * w : (c + 1) * w] for c in range(2)]
        for r in range(3)
    ]
    for r in range(3):
        assert np.array_equal(tiles[r][1], np.fliplr(tiles[r][0]))
    for r in range(2):
        for c in range(2):
            assert np.array_equal(tiles[r + 1][c], np.flipud(tiles[r][c]))

    out = main(grid)
    divider = int(layout["divider"])
    assert np.array_equal(out[:, : divider + 1], grid[:, : divider + 1]), "left structure must stay fixed"
    assert np.array_equal(out[:, divider + 1 :], expected), "output canvas must be the mirrored tiling"


def main(grid: np.ndarray) -> np.ndarray:
    """
    XOR the two left masks to recover a hidden seed, then fill the right canvas with alternating mirrored tiles.

    Input structure:
    - A full grey column splits the grid into a left mask area and a right black canvas.
    - Inside the left area, one full yellow row separates two equal-size red/black masks.
    - The right area is an empty canvas whose height is three mask-heights and whose width is two mask-widths.

    Rule:
    - Locate the grey divider column and the yellow separator row.
    - Extract the two masks, then compute a green/black seed by pixelwise XOR: green where exactly one mask cell is red, black otherwise.
    - Fill the right canvas as a 3-by-2 tiling of that seed, flipping left-right across horizontal tile neighbors and up-down across vertical tile neighbors.
    - Leave the left masks, separator row, and divider column unchanged.

    Key insight:
    - The visible tile pattern is not present directly in the input; it must first be inferred as the XOR seed, and only then mirrored across both tile axes.
    """
    layout = _extract_layout(grid)
    out = np.array(layout["grid"], copy=True)
    seed = _xor_seed(np.asarray(layout["top"]), np.asarray(layout["bottom"]))
    out[:, int(layout["divider"]) + 1 :] = _alternate_mirror_fill(seed)
    return out


def generate_input() -> np.ndarray:
    for _ in range(500):
        mask_h = random.randint(3, 8)
        mask_w = random.randint(3, 7)
        H = 3 * mask_h
        W = 3 * mask_w + 1
        top_pad = random.randint(0, mask_h - 1)
        sep = top_pad + mask_h

        seed_mask = None
        for _seed_try in range(200):
            candidate = _random_binary_mask(mask_h, mask_w, min_active=2, max_active=mask_h * mask_w - 2)
            if _seed_ok(candidate):
                seed_mask = candidate
                break
        if seed_mask is None:
            continue

        top_mask = None
        bottom_mask = None
        for _mask_try in range(200):
            candidate_top = _random_binary_mask(mask_h, mask_w, min_active=1, max_active=mask_h * mask_w - 1)
            candidate_bottom = np.logical_xor(candidate_top, seed_mask)
            top_count = int(candidate_top.sum())
            bottom_count = int(candidate_bottom.sum())
            if top_count in (0, mask_h * mask_w) or bottom_count in (0, mask_h * mask_w):
                continue
            top_mask = candidate_top
            bottom_mask = candidate_bottom
            break
        if top_mask is None or bottom_mask is None:
            continue

        grid = np.zeros((H, W), dtype=int)
        grid[:, mask_w] = GREY
        grid[sep, :mask_w] = YELLOW
        grid[sep - mask_h : sep, :mask_w] = np.where(top_mask, RED, BLACK)
        grid[sep + 1 : sep + 1 + mask_h, :mask_w] = np.where(bottom_mask, RED, BLACK)

        try:
            _generated_case_ok(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
