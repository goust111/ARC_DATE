from __future__ import annotations

import random

import numpy as np


UNKNOWN_COLORS = {0, 1}
PATTERN_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
TILE_SHAPES = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)]
GEN_RETRY_LIMIT = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_occluder_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(grid == 1)
    assert box is not None, "expected a blue occluder"
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w, "expected a square occluder"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == 1), "occluder must be solid"
    assert int(np.count_nonzero(grid == 1)) == h * w, "blue must appear only in the occluder"
    return r0, c0, r1, c1


def _infer_minimal_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    known = ~np.isin(g, tuple(UNKNOWN_COLORS))
    assert np.any(known), "expected known wallpaper pixels"

    candidates: list[tuple[int, np.ndarray]] = []
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            tile = np.full((tile_h, tile_w), -1, dtype=int)
            valid = True
            for r, c in np.argwhere(known):
                tr = int(r) % tile_h
                tc = int(c) % tile_w
                value = int(g[r, c])
                if tile[tr, tc] == -1:
                    tile[tr, tc] = value
                elif tile[tr, tc] != value:
                    valid = False
                    break
            if valid and np.all(tile != -1):
                candidates.append((tile_h * tile_w, tile))

    assert candidates, "expected at least one fully observed periodic tile"
    best_area = min(area for area, _ in candidates)
    best_tiles = [tile for area, tile in candidates if area == best_area]
    assert len(best_tiles) == 1, "expected a unique minimal tile"
    return best_tiles[0]


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = tile.shape
    rows = np.arange(shape[0])[:, None] % tile_h
    cols = np.arange(shape[1])[None, :] % tile_w
    return tile[rows, cols]


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    for th in range(1, h + 1):
        for tw in range(1, w + 1):
            if np.array_equal(tile, _repeat_tile(tile[:th, :tw], (h, w))):
                return th, tw
    return h, w


def _phase_counts(shape: tuple[int, int], tile_shape: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = tile_shape
    rows = np.arange(shape[0]) % tile_h
    cols = np.arange(shape[1]) % tile_w
    counts = np.zeros((tile_h, tile_w), dtype=int)
    np.add.at(counts, (rows[:, None], cols[None, :]), 1)
    return counts


def _tile_candidates(tile_h: int, tile_w: int) -> np.ndarray | None:
    n_colors = random.randint(3, min(5, len(PATTERN_COLORS)))
    colors = random.sample(PATTERN_COLORS, n_colors)
    tile = np.random.choice(colors, size=(tile_h, tile_w)).astype(int)

    for color in colors:
        if color not in tile:
            rr = random.randrange(tile_h)
            cc = random.randrange(tile_w)
            tile[rr, cc] = color

    if _minimal_period(tile) != (tile_h, tile_w):
        return None
    if any(len(set(map(int, tile[r, :]))) == 1 for r in range(tile_h)):
        return None
    if any(len(set(map(int, tile[:, c]))) == 1 for c in range(tile_w)):
        return None
    if len({tuple(map(int, row)) for row in tile}) == 1:
        return None
    if len({tuple(map(int, tile[:, c])) for c in range(tile_w)}) == 1:
        return None
    return tile


def _sample_tile() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        tile_h, tile_w = random.choice(TILE_SHAPES)
        tile = _tile_candidates(tile_h, tile_w)
        if tile is not None:
            return tile
    raise AssertionError("failed to sample a full-period tile")


def _select_black_positions(
    shape: tuple[int, int],
    tile_shape: tuple[int, int],
    occluder_cells: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    h, w = shape
    tile_h, tile_w = tile_shape
    occluded_phases = {(r % tile_h, c % tile_w) for r, c in occluder_cells}
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occluder_cells]
    preferred = [rc for rc in candidates if (rc[0] % tile_h, rc[1] % tile_w) in occluded_phases]
    others = [rc for rc in candidates if rc not in set(preferred)]
    random.shuffle(preferred)
    random.shuffle(others)

    selected: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    for pool in (preferred, others):
        for r, c in pool:
            if len(selected) == count:
                break
            if (r, c) in blocked:
                continue
            selected.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    blocked.add((r + dr, c + dc))
        if len(selected) == count:
            break

    if len(selected) < count:
        pool = preferred + others
        for rc in pool:
            if len(selected) == count:
                break
            if rc not in selected:
                selected.append(rc)
    return selected


def _make_example() -> tuple[np.ndarray, np.ndarray]:
    tile = _sample_tile()
    tile_h, tile_w = tile.shape

    rep_r = random.randint(4, min(7, 30 // tile_h))
    rep_c = random.randint(4, min(7, 30 // tile_w))
    height = tile_h * rep_r
    width = tile_w * rep_c
    if min(height, width) < 10:
        raise AssertionError("grid too small")

    full = _repeat_tile(tile, (height, width))
    size = random.randint(4, min(6, height - 2, width - 2))
    r0 = random.randint(1, height - size - 1)
    c0 = random.randint(1, width - size - 1)
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    occluder_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    black_count = random.randint(4, 8)
    black_positions = _select_black_positions(full.shape, tile.shape, occluder_cells, black_count)
    if len(black_positions) != black_count:
        raise AssertionError("failed to place black holes")

    phase_counts = _phase_counts(full.shape, tile.shape)
    for r, c in occluder_cells:
        phase_counts[r % tile_h, c % tile_w] -= 1
    for r, c in black_positions:
        phase_counts[r % tile_h, c % tile_w] -= 1
    if not np.all(phase_counts > 0):
        raise AssertionError("some tile phase became fully hidden")

    inp = full.copy()
    for r, c in black_positions:
        inp[r, c] = 0
    inp[r0 : r1 + 1, c0 : c1 + 1] = 1

    expected = full[r0 : r1 + 1, c0 : c1 + 1].copy()

    inferred = _infer_minimal_tile(inp)
    if not np.array_equal(inferred, tile):
        raise AssertionError("observations no longer determine the intended minimal tile")
    if not np.array_equal(main(inp), expected):
        raise AssertionError("solver does not recover the hidden patch")
    return inp, expected


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the latent wallpaper period while ignoring black holes and a blue occluder, then return the occluded patch.

    Input structure:
    - The input is a 2D wallpaper made by repeating a small multicolor tile.
    - Some isolated pixels are black and represent missing data rather than real pattern color.
    - One solid blue square occluder covers a rectangular region of the wallpaper, and blue appears nowhere else.

    Rule:
    - Treat both black `0` cells and blue `1` cells as unknown values.
    - Infer the unique minimal tile `(H, W)` such that all known cells agree by phase `(r mod H, c mod W)`.
    - Fill every unknown cell from the inferred tile, then crop the repaired grid to the blue square's bounding box.

    Key insight:
    - The hidden patch is recoverable from periodicity, not symmetry: both sparse black erasures and the dense blue block are just missing observations of the same repeating wallpaper.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_occluder_bbox(g)
    tile = _infer_minimal_tile(g)
    repaired = _repeat_tile(tile, g.shape)
    known = ~np.isin(g, tuple(UNKNOWN_COLORS))
    assert np.array_equal(repaired[known], g[known]), "inferred tile must preserve all visible wallpaper cells"
    return repaired[r0 : r1 + 1, c0 : c1 + 1]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            return _make_example()[0]
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid puzzle instance")
