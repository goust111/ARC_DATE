from __future__ import annotations

import numpy as np


BG = 0
TOP_FG = 1
BOTTOM_FG = 2
XOR_FG = 4

SUPPORTED_ORDERS = (3, 4)
GEN_MAX_DIM = 30
GEN_MAX_ATTEMPTS = 500

_rng = np.random.default_rng()


def _array_side(order: int) -> int:
    return order * order + order - 1


def _tile_start(origin: tuple[int, int], tile_r: int, tile_c: int, order: int) -> tuple[int, int]:
    r0 = origin[0] + tile_r * (order + 1)
    c0 = origin[1] + tile_c * (order + 1)
    return r0, c0


def _tile_pattern(order: int, tile_r: int, tile_c: int) -> np.ndarray:
    tile = np.full((order, order), XOR_FG, dtype=int)
    tile[tile_r, tile_c] = BG
    return tile


def _xor_halves(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0
    half = g.shape[0] // 2
    top = g[:half]
    bottom = g[half:]
    return np.where((top != BG) ^ (bottom != BG), XOR_FG, BG)


def _all_zero_outside(mask: np.ndarray, origin: tuple[int, int], side: int) -> bool:
    r0, c0 = origin
    region = np.zeros_like(mask, dtype=bool)
    region[r0 : r0 + side, c0 : c0 + side] = True
    return not np.any(mask & ~region)


def _find_array_candidates(grid: np.ndarray) -> list[dict]:
    mask = np.asarray(grid, dtype=int) != BG
    h, w = mask.shape
    total_fg = int(mask.sum())
    candidates: list[dict] = []

    for order in SUPPORTED_ORDERS:
        side = _array_side(order)
        if side > h or side > w:
            continue
        separators = [i * (order + 1) - 1 for i in range(1, order)]
        for r0 in range(h - side + 1):
            for c0 in range(w - side + 1):
                if total_fg == 0:
                    continue
                if not _all_zero_outside(mask, (r0, c0), side):
                    continue
                region = mask[r0 : r0 + side, c0 : c0 + side]
                if not region.any():
                    continue
                if any(region[sr, :].any() for sr in separators):
                    continue
                if any(region[:, sc].any() for sc in separators):
                    continue

                present_tiles: set[tuple[int, int]] = set()
                row_seen = [False] * order
                col_seen = [False] * order
                valid = True
                for tile_r in range(order):
                    for tile_c in range(order):
                        rr = tile_r * (order + 1)
                        cc = tile_c * (order + 1)
                        sub = region[rr : rr + order, cc : cc + order]
                        if not sub.any():
                            continue
                        expected = np.ones((order, order), dtype=bool)
                        expected[tile_r, tile_c] = False
                        if not np.array_equal(sub, expected):
                            valid = False
                            break
                        present_tiles.add((tile_r, tile_c))
                        row_seen[tile_r] = True
                        col_seen[tile_c] = True
                    if not valid:
                        break
                if not valid:
                    continue
                if len(present_tiles) < order or len(present_tiles) == order * order:
                    continue
                if not (all(row_seen) or all(col_seen)):
                    continue
                candidates.append(
                    {
                        "order": order,
                        "origin": (r0, c0),
                        "side": side,
                        "present_tiles": frozenset(present_tiles),
                    }
                )
    return candidates


def _detect_unique_array(grid: np.ndarray) -> dict:
    candidates = _find_array_candidates(grid)
    assert len(candidates) == 1, f"expected unique tile-array candidate, got {len(candidates)}"
    return candidates[0]


def _render_array(
    shape: tuple[int, int],
    order: int,
    origin: tuple[int, int],
    present_tiles: set[tuple[int, int]] | frozenset[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for tile_r, tile_c in present_tiles:
        r0, c0 = _tile_start(origin, tile_r, tile_c, order)
        out[r0 : r0 + order, c0 : c0 + order] = XOR_FG
        out[r0 + tile_r, c0 + tile_c] = BG
    return out


def _render_completed_array(shape: tuple[int, int], order: int, origin: tuple[int, int]) -> np.ndarray:
    all_tiles = {(tile_r, tile_c) for tile_r in range(order) for tile_c in range(order)}
    return _render_array(shape, order, origin, all_tiles)


def _sample_present_tiles(order: int) -> set[tuple[int, int]]:
    all_tiles = [(r, c) for r in range(order) for c in range(order)]
    row_reps = [(r, int(_rng.integers(order))) for r in range(order)]
    col_reps = [(int(_rng.integers(order)), c) for c in range(order)]
    present = set(row_reps) | set(col_reps)

    extra_pool = [tile for tile in all_tiles if tile not in present]
    if extra_pool:
        extra_count = int(_rng.integers(0, len(extra_pool) + 1))
        if extra_count:
            picks = _rng.choice(len(extra_pool), size=extra_count, replace=False)
            present.update(extra_pool[int(idx)] for idx in np.atleast_1d(picks))

    if len(present) == order * order:
        drop_candidates = [tile for tile in all_tiles if tile not in row_reps or tile not in col_reps]
        if not drop_candidates:
            drop_candidates = all_tiles
        present.remove(drop_candidates[int(_rng.integers(len(drop_candidates)))])

    assert len(present) >= order
    assert len(present) < order * order
    assert all(any(tile[0] == r for tile in present) for r in range(order))
    assert all(any(tile[1] == c for tile in present) for c in range(order))
    return present


def _encode_xor_target(target: np.ndarray) -> np.ndarray:
    h, w = target.shape
    density = float(_rng.uniform(0.35, 0.65))
    top_mask = _rng.random((h, w)) < density
    bottom_mask = top_mask ^ (target != BG)

    top = np.where(top_mask, TOP_FG, BG)
    bottom = np.where(bottom_mask, BOTTOM_FG, BG)
    return np.vstack([top, bottom])


def _choose_generation_layout(order: int) -> tuple[tuple[int, int], tuple[int, int]]:
    side = _array_side(order)
    max_half_h = GEN_MAX_DIM // 2
    half_h = int(_rng.integers(side, max_half_h + 1))
    half_w = int(_rng.integers(side, min(GEN_MAX_DIM, side + 8) + 1))
    origin = (
        int(_rng.integers(0, half_h - side + 1)),
        int(_rng.integers(0, half_w - side + 1)),
    )
    return (half_h, half_w), origin


def _validate_instance(
    inp: np.ndarray,
    partial: np.ndarray,
    expected: np.ndarray,
    order: int,
    origin: tuple[int, int],
    present_tiles: set[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > GEN_MAX_DIM:
        return False

    xor_grid = _xor_halves(inp)
    if not np.array_equal(xor_grid, partial):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    try:
        meta = _detect_unique_array(xor_grid)
    except Exception:
        return False

    if meta["order"] != order or meta["origin"] != origin or set(meta["present_tiles"]) != set(present_tiles):
        return False
    if _find_array_candidates(inp[: inp.shape[0] // 2]):
        return False
    if _find_array_candidates(inp[inp.shape[0] // 2 :]):
        return False

    top = inp[: inp.shape[0] // 2] != BG
    bottom = inp[inp.shape[0] // 2 :] != BG
    overlap = int(np.sum(top & bottom))
    top_only = int(np.sum(top & ~bottom))
    bottom_only = int(np.sum(~top & bottom))
    if overlap == 0 or top_only == 0 or bottom_only == 0:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the XOR-emergent coordinate-coded tile array.

    Input structure:
    The input has even height and is split into equal top and bottom halves on a black background.
    The top half contains blue foreground scatter and the bottom half contains red foreground scatter.
    Viewed separately, neither half directly shows the target structure; after aligned XOR they reveal
    a partial n x n array of n x n yellow tiles separated by one-cell black rows and columns.

    Rule:
    XOR the two halves by position, treating any non-black cell as foreground and emitting yellow where
    exactly one half is foreground. In that XOR result, locate the separated tile array, verify that each
    visible tile has exactly one black hole at local coordinate (tile_row, tile_col), and fill every
    missing tile with the same coordinate-coded one-hole pattern.

    Key insight:
    The tile array does not exist in either half alone; it emerges only after XOR, and the hole inside
    each visible tile encodes the tile's array coordinates, which determines how to complete the absent tiles.
    """
    xor_grid = _xor_halves(grid)
    meta = _detect_unique_array(xor_grid)
    out = np.zeros_like(xor_grid)
    out[:] = _render_completed_array(xor_grid.shape, meta["order"], meta["origin"])
    return out


def generate_input() -> np.ndarray:
    valid_orders = [order for order in SUPPORTED_ORDERS if 2 * _array_side(order) <= GEN_MAX_DIM]
    for _ in range(GEN_MAX_ATTEMPTS):
        order = int(_rng.choice(valid_orders))
        shape, origin = _choose_generation_layout(order)
        present_tiles = _sample_present_tiles(order)

        partial = _render_array(shape, order, origin, present_tiles)
        expected = _render_completed_array(shape, order, origin)
        inp = _encode_xor_target(partial)

        if _validate_instance(inp, partial, expected, order, origin, present_tiles):
            return inp
    raise RuntimeError("failed to generate a valid XOR tile-array instance")
