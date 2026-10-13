from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color  # type: ignore  # noqa: E402


MAGENTA = int(Color.PINK)
BG = int(Color.BLACK)
SAFE_COLORS = tuple(
    color for color in Color.NOT_BLACK if int(color) != MAGENTA
)
GEN_RETRIES = 400
LAYOUTS = (
    (3, 3, 4, 4),
    (4, 3, 3, 4),
)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _symmetry_kind(tile: np.ndarray) -> str | None:
    lr = np.array_equal(tile, np.fliplr(tile))
    ud = np.array_equal(tile, np.flipud(tile))
    if lr == ud:
        return None
    return "lr" if lr else "ud"


def _partner(coord: tuple[int, int], shape: tuple[int, int], symmetry: str) -> tuple[int, int]:
    r, c = map(int, coord)
    h, w = map(int, shape)
    if symmetry == "lr":
        return r, w - 1 - c
    if symmetry == "ud":
        return h - 1 - r, c
    raise ValueError(f"unknown symmetry {symmetry}")


def _iter_tiles(grid: np.ndarray, tile_h: int, tile_w: int):
    for r0 in range(0, grid.shape[0], tile_h):
        for c0 in range(0, grid.shape[1], tile_w):
            yield r0, c0, grid[r0 : r0 + tile_h, c0 : c0 + tile_w]


def _candidate_for_size(grid: np.ndarray, tile_h: int, tile_w: int) -> dict[str, Any] | None:
    h, w = map(int, grid.shape)
    if h % tile_h != 0 or w % tile_w != 0:
        return None
    if (h // tile_h) * (w // tile_w) < 2:
        return None

    tiles = list(_iter_tiles(grid, tile_h, tile_w))
    damaged = [(r0, c0, tile.copy()) for r0, c0, tile in tiles if np.any(tile == MAGENTA)]
    normal = [tile.copy() for _r0, _c0, tile in tiles if not np.any(tile == MAGENTA)]
    if len(damaged) != 1 or len(normal) < 1:
        return None

    template = normal[0]
    if np.any(template == MAGENTA):
        return None
    if any(not np.array_equal(tile, template) for tile in normal[1:]):
        return None

    symmetry = _symmetry_kind(template)
    if symmetry is None:
        return None

    r0, c0, damaged_tile = damaged[0]
    magenta_mask = damaged_tile == MAGENTA
    magenta_coords = [tuple(map(int, coord)) for coord in np.argwhere(magenta_mask)]
    if len(magenta_coords) not in (1, 2):
        return None
    if not np.array_equal(damaged_tile != template, magenta_mask):
        return None

    for coord in magenta_coords:
        partner = _partner(coord, damaged_tile.shape, symmetry)
        if partner == coord:
            return None
        if int(damaged_tile[partner]) == MAGENTA:
            return None
        if int(template[coord]) != int(template[partner]):
            return None

    return {
        "tile_shape": (int(tile_h), int(tile_w)),
        "area": int(tile_h * tile_w),
        "origin": (int(r0), int(c0)),
        "template": template,
        "damaged_tile": damaged_tile,
        "symmetry": symmetry,
        "magenta_coords": magenta_coords,
    }


def _all_candidates(grid: np.ndarray) -> list[dict[str, Any]]:
    h, w = map(int, grid.shape)
    candidates: list[dict[str, Any]] = []
    for tile_h in _divisors(h):
        for tile_w in _divisors(w):
            candidate = _candidate_for_size(grid, int(tile_h), int(tile_w))
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda item: (int(item["area"]), item["tile_shape"][0], item["tile_shape"][1]))
    return candidates


def _infer_unique_candidate(grid: np.ndarray) -> dict[str, Any]:
    candidates = _all_candidates(grid)
    assert candidates, "expected at least one valid tile decomposition"
    min_area = int(candidates[0]["area"])
    minimal = [candidate for candidate in candidates if int(candidate["area"]) == min_area]
    assert len(minimal) == 1, "expected a unique minimal tile decomposition"
    return minimal[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the unique damaged periodic tile and repair its magenta cells from the tile's mirror symmetry.

    Input structure:
    - The grid has black background and can be partitioned exactly into equal rectangular tiles.
    - All tiles are intended to be identical copies of one multicolor motif.
    - Exactly one tile contains 1-2 magenta cells, and magenta does not occur in normal tiles.

    Rule:
    - Enumerate tile sizes and keep the unique smallest size whose partition has exactly one magenta-bearing tile and identical normal tiles.
    - Infer whether the normal tile is left-right symmetric or up-down symmetric, but not both.
    - For each magenta cell in the damaged tile, replace it with the color at its mirrored partner within that same tile.

    Key insight:
    - Period detection identifies the comparison unit, while the tile's own symmetry determines the missing color once the damaged positions are known.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidate = _infer_unique_candidate(g)
    symmetry = str(candidate["symmetry"])
    r0, c0 = map(int, candidate["origin"])

    out = g.copy()
    for coord in candidate["magenta_coords"]:
        pr, pc = _partner(coord, candidate["tile_shape"], symmetry)
        out[r0 + coord[0], c0 + coord[1]] = int(g[r0 + pr, c0 + pc])

    assert not np.any(out == MAGENTA), "repair must remove all magenta cells"
    return out


def _repeat_from_half_lr(left_half: np.ndarray, full_w: int) -> np.ndarray:
    if full_w % 2 == 0:
        return np.concatenate([left_half, np.fliplr(left_half)], axis=1)
    return np.concatenate([left_half, np.fliplr(left_half[:, :-1])], axis=1)


def _repeat_from_half_ud(top_half: np.ndarray, full_h: int) -> np.ndarray:
    if full_h % 2 == 0:
        return np.concatenate([top_half, np.flipud(top_half)], axis=0)
    return np.concatenate([top_half, np.flipud(top_half[:-1])], axis=0)


def _is_repetition_of_smaller(tile: np.ndarray) -> bool:
    h, w = map(int, tile.shape)
    for sub_h in _divisors(h):
        for sub_w in _divisors(w):
            if sub_h == h and sub_w == w:
                continue
            if h % sub_h != 0 or w % sub_w != 0:
                continue
            seed = tile[:sub_h, :sub_w]
            tiled = np.tile(seed, (h // sub_h, w // sub_w))
            if np.array_equal(tiled, tile):
                return True
    return False


def _damage_sites(tile: np.ndarray, symmetry: str) -> list[tuple[int, int]]:
    h, w = map(int, tile.shape)
    sites: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(tile[r, c]) == BG:
                continue
            partner = _partner((r, c), tile.shape, symmetry)
            if partner == (r, c):
                continue
            if (r, c) < partner:
                sites.append((r, c))
    return sites


def _build_symmetric_tile(rng: np.random.Generator, tile_h: int, tile_w: int, symmetry: str) -> np.ndarray | None:
    num_colors = int(rng.integers(2, 5))
    fg_colors = rng.choice(np.array(SAFE_COLORS, dtype=int), size=num_colors, replace=False)
    palette = np.array([BG, *map(int, fg_colors)], dtype=int)

    weights = rng.random(len(palette)) + 0.2
    weights[0] += rng.uniform(0.4, 1.1)
    weights = weights / weights.sum()

    if symmetry == "lr":
        left_half = rng.choice(palette, size=(tile_h, (tile_w + 1) // 2), p=weights)
        tile = _repeat_from_half_lr(left_half, tile_w)
    else:
        top_half = rng.choice(palette, size=((tile_h + 1) // 2, tile_w), p=weights)
        tile = _repeat_from_half_ud(top_half, tile_h)

    non_black = sorted(set(map(int, tile.flatten())) - {BG})
    if len(non_black) < 2:
        return None
    fg_count = int(np.count_nonzero(tile != BG))
    if fg_count < max(4, (tile_h * tile_w) // 3):
        return None
    if fg_count > tile_h * tile_w - 1:
        return None
    if _symmetry_kind(tile) != symmetry:
        return None
    if _is_repetition_of_smaller(tile):
        return None
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return None
    if len({tuple(map(int, col)) for col in tile.T}) < 2:
        return None
    if len(_damage_sites(tile, symmetry)) < 1:
        return None
    return tile.astype(int, copy=False)


def _validate_instance(
    grid: np.ndarray,
    expected_output: np.ndarray,
    tile_h: int,
    tile_w: int,
    symmetry: str,
    damage_origin: tuple[int, int],
    damage_sites: list[tuple[int, int]],
) -> bool:
    if int(np.count_nonzero(grid == MAGENTA)) not in (1, 2):
        return False
    if np.array_equal(grid, expected_output):
        return False
    if np.any(expected_output == MAGENTA):
        return False

    candidates = _all_candidates(grid)
    if not candidates:
        return False
    min_area = int(candidates[0]["area"])
    minimal = [candidate for candidate in candidates if int(candidate["area"]) == min_area]
    if len(minimal) != 1:
        return False

    candidate = minimal[0]
    if tuple(candidate["tile_shape"]) != (tile_h, tile_w):
        return False
    if str(candidate["symmetry"]) != symmetry:
        return False
    if tuple(candidate["origin"]) != tuple(map(int, damage_origin)):
        return False

    repaired = main(grid)
    if not np.array_equal(repaired, expected_output):
        return False

    for local_site in damage_sites:
        partner = _partner(local_site, (tile_h, tile_w), symmetry)
        rr = int(damage_origin[0] + local_site[0])
        cc = int(damage_origin[1] + local_site[1])
        pr = int(damage_origin[0] + partner[0])
        pc = int(damage_origin[1] + partner[1])
        if int(grid[rr, cc]) != MAGENTA:
            return False
        if int(grid[pr, pc]) == MAGENTA:
            return False
        if int(repaired[rr, cc]) != int(repaired[pr, pc]):
            return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile_rows, tile_cols, tile_h, tile_w = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    symmetry = "lr" if bool(rng.integers(2)) else "ud"

    tile = _build_symmetric_tile(rng, tile_h, tile_w, symmetry)
    if tile is None:
        return None

    sites = _damage_sites(tile, symmetry)
    if not sites:
        return None

    damage_count = 1
    if len(sites) >= 2 and bool(rng.integers(2)):
        damage_count = 2
    chosen_indices = rng.choice(len(sites), size=damage_count, replace=False)
    chosen_sites = [sites[int(idx)] for idx in np.atleast_1d(chosen_indices)]

    clean = np.tile(tile, (tile_rows, tile_cols))
    grid = clean.copy()

    damage_tile_r = int(rng.integers(tile_rows))
    damage_tile_c = int(rng.integers(tile_cols))
    r0 = damage_tile_r * tile_h
    c0 = damage_tile_c * tile_w
    for rr, cc in chosen_sites:
        grid[r0 + rr, c0 + cc] = MAGENTA

    if not _validate_instance(
        grid,
        clean,
        tile_h,
        tile_w,
        symmetry,
        (r0, c0),
        chosen_sites,
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic damaged-tile input")
