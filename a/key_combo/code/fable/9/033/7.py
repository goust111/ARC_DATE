from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
TEAL = 8
MAROON = 9

_GEN_ATTEMPTS = 400


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == BLUE)]
    assert len(cols) == 1, "expected exactly one full-height blue separator"
    return int(cols[0])


def _bbox_of_color(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tile_rect(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = tile.shape
    H, W = shape
    assert H % h == 0 and W % w == 0
    return np.tile(tile, (H // h, W // w))


def _divisors(n: int) -> Iterable[int]:
    for d in range(1, int(n) + 1):
        if n % d == 0:
            yield d


def _is_primitive_tile(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for hh in _divisors(h):
        for ww in _divisors(w):
            if hh == h and ww == w:
                continue
            if np.array_equal(tile, _tile_rect(tile[:hh, :ww], (h, w))):
                return False
    return True


def _extract_left_tile_and_right(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bar = _find_separator(grid)
    left_zone = np.asarray(grid[:, :bar], dtype=int)
    right = np.asarray(grid[:, bar + 1 :], dtype=int)
    assert left_zone.size > 0 and right.size > 0
    assert set(np.unique(left_zone)).issubset({BLACK, MAROON})
    assert set(np.unique(right)).issubset({BLACK, MAROON})

    bbox = _bbox_of_color(left_zone == MAROON)
    assert bbox is not None, "left generator must contain maroon"
    tile = _crop(left_zone, bbox)
    r0, c0, r1, c1 = bbox
    padded = left_zone.copy()
    padded[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    assert np.all(padded == BLACK), "left zone must contain a single compact generator"

    assert tile.shape[0] < right.shape[0] and tile.shape[1] < right.shape[1]
    assert right.shape[0] % tile.shape[0] == 0 and right.shape[1] % tile.shape[1] == 0
    return tile, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the small left generator and take its logical NOR with the right mask.

    Input structure:
    - A full-height blue vertical separator splits the grid into a left zone and a right panel.
    - The right panel is a large maroon/black binary mask.
    - The left zone contains one smaller maroon/black generator motif on black background; the motif's bounding-box height and width each divide the right panel's corresponding size.

    Rule:
    - Crop the smaller left generator by its maroon bounding box.
    - Repeat that generator in both row and column directions until it matches the right panel's size.
    - Output only the right-panel-sized result: a cell is teal exactly when both the tiled generator and the right panel are not maroon there; otherwise it is black.

    Key insight:
    - The left operand is not already drawn at the comparison scale; it must first be reconstructed by rectangular tiling before the pixelwise NOR can be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    tile, right = _extract_left_tile_and_right(g)
    tiled = _tile_rect(tile, right.shape)

    out = np.full(right.shape, BLACK, dtype=int)
    out[(tiled != MAROON) & (right != MAROON)] = TEAL
    return out


def _sample_dims() -> tuple[int, int, int, int]:
    choices: list[tuple[int, int, int, int]] = []
    for tile_h in range(2, 7):
        for row_mult in range(2, min(5, 30 // tile_h) + 1):
            right_h = tile_h * row_mult
            if right_h > 30:
                continue
            for tile_w in range(2, 8):
                for col_mult in range(2, 6):
                    right_w = tile_w * col_mult
                    total_w = tile_w + 1 + right_w
                    if total_w <= 30:
                        choices.append((tile_h, tile_w, row_mult, col_mult))
    assert choices
    return random.choice(choices)


def _sample_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    for _ in range(200):
        p = random.uniform(0.35, 0.65)
        tile = np.where(np.random.random((tile_h, tile_w)) < p, MAROON, BLACK).astype(int)
        if not np.any(tile == MAROON):
            continue
        if not np.any(tile[0] == MAROON):
            continue
        if not np.any(tile[-1] == MAROON):
            continue
        if not np.any(tile[:, 0] == MAROON):
            continue
        if not np.any(tile[:, -1] == MAROON):
            continue
        if not _is_primitive_tile(tile):
            continue
        return tile
    return None


def _sample_right(tiled: np.ndarray) -> np.ndarray | None:
    H, W = tiled.shape
    for _ in range(200):
        p = random.uniform(0.30, 0.70)
        right = np.where(np.random.random((H, W)) < p, MAROON, BLACK).astype(int)
        combos = {
            (int(tiled[r, c] == MAROON), int(right[r, c] == MAROON))
            for r in range(H)
            for c in range(W)
        }
        if combos != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue

        out = np.full((H, W), BLACK, dtype=int)
        out[(tiled != MAROON) & (right != MAROON)] = TEAL
        if not np.any(out == TEAL) or not np.any(out == BLACK):
            continue

        right_black = right != MAROON
        tile_black = tiled != MAROON
        if not (np.any(right_black & tile_black) and np.any(right_black & ~tile_black)):
            continue
        if not (np.any(tile_black & right_black) and np.any(tile_black & ~right_black)):
            continue
        return right
    return None


def _build_input(tile: np.ndarray, right: np.ndarray) -> np.ndarray:
    right_h, _right_w = right.shape
    tile_h, tile_w = tile.shape
    left_zone = np.full((right_h, tile_w), BLACK, dtype=int)
    tile_r0 = random.randint(0, right_h - tile_h)
    left_zone[tile_r0 : tile_r0 + tile_h, :] = tile

    bar = np.full((right_h, 1), BLUE, dtype=int)
    return np.concatenate([left_zone, bar, right], axis=1)


def _semantic_checks(inp: np.ndarray) -> None:
    assert inp.ndim == 2
    assert max(inp.shape) <= 30
    tile, right = _extract_left_tile_and_right(inp)
    assert _is_primitive_tile(tile)

    tiled = _tile_rect(tile, right.shape)
    combos = {
        (int(tiled[r, c] == MAROON), int(right[r, c] == MAROON))
        for r in range(right.shape[0])
        for c in range(right.shape[1])
    }
    assert combos == {(0, 0), (0, 1), (1, 0), (1, 1)}

    out = main(inp)
    assert np.any(out == TEAL) and np.any(out == BLACK)

    right_only = np.full(right.shape, BLACK, dtype=int)
    right_only[right != MAROON] = TEAL
    tile_only = np.full(right.shape, BLACK, dtype=int)
    tile_only[tiled != MAROON] = TEAL
    assert not np.array_equal(out, right_only), "result must depend on tiled generator"
    assert not np.array_equal(out, tile_only), "result must depend on right mask"

    # Same right value must sometimes yield different outputs depending on the tiled generator.
    for value in (BLACK, MAROON):
        mask = right == value
        if np.any(mask):
            vals = set(map(int, np.unique(out[mask])))
            if value == BLACK:
                assert vals == {BLACK, TEAL}
            else:
                assert vals == {BLACK}

    # Same tiled value must sometimes yield different outputs depending on the right panel.
    mask = tiled == BLACK
    assert set(map(int, np.unique(out[mask]))) == {BLACK, TEAL}


def _gen_attempt() -> np.ndarray | None:
    tile_h, tile_w, row_mult, col_mult = _sample_dims()
    tile = _sample_tile(tile_h, tile_w)
    if tile is None:
        return None

    right = _sample_right(_tile_rect(tile, (tile_h * row_mult, tile_w * col_mult)))
    if right is None:
        return None

    inp = _build_input(tile, right)
    try:
        _semantic_checks(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to construct a valid input")
