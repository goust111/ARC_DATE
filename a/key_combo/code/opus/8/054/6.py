from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

INPUT_COLORS = {BLACK, RED, GREEN, GREY}
TERRAIN_COLORS = {RED, GREEN}
MAX_GENERATION_ATTEMPTS = 400


def _validate_tile(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 5 <= h <= 15 and 4 <= w <= 15, "tile dimensions must support the reflected output"
    assert set(map(int, np.unique(g))).issubset(INPUT_COLORS), "unexpected input color"
    assert np.all(g[0, :] == GREY), "row 0 must be the grey separator"

    surfaces = np.zeros(w, dtype=int)
    marker_cols: list[int] = []
    for c in range(w):
        terrain_rows = np.flatnonzero(np.isin(g[:, c], tuple(TERRAIN_COLORS)))
        assert terrain_rows.size > 0, "every column must contain terrain"
        surface = int(terrain_rows[0])
        assert surface >= 1, "terrain must start below the separator row"
        assert np.array_equal(terrain_rows, np.arange(surface, h)), "terrain must be a contiguous suffix"
        assert np.all(g[1:surface, c] == BLACK), "cells above terrain must be black"
        red_rows = np.flatnonzero(g[:, c] == RED)
        if red_rows.size:
            assert red_rows.size == 1, "a column may contain at most one marker"
            assert int(red_rows[0]) == surface, "the marker must replace the terrain top"
            assert np.all(g[surface + 1 :, c] == GREEN), "terrain below the marker must stay green"
            marker_cols.append(c)
        else:
            assert np.all(g[surface:, c] == GREEN), "unmarked terrain columns must be solid green"
        surfaces[c] = surface

    assert len(marker_cols) == 2 and len(set(marker_cols)) == 2, "expected exactly two marker columns"
    return surfaces, marker_cols


def _reflect_quadrants(tile: np.ndarray) -> np.ndarray:
    top = np.concatenate((tile, np.fliplr(tile)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _composite_surfaces(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    surfaces = np.zeros(w, dtype=int)
    for c in range(w):
        terrain_rows = np.flatnonzero(np.isin(grid[:, c], tuple(TERRAIN_COLORS)))
        assert terrain_rows.size > 0, "each composite column must contain terrain"
        surfaces[c] = int(terrain_rows[0])
    return surfaces


def _paint_column_gaps(grid: np.ndarray, col: int, color: int) -> None:
    terrain_rows = np.flatnonzero(np.isin(grid[:, col], tuple(TERRAIN_COLORS)))
    top_surface = int(terrain_rows[0])
    bottom_surface = int(terrain_rows[-1])

    if top_surface > 1:
        top_block = grid[1:top_surface, col]
        top_block[top_block == BLACK] = color
        grid[1:top_surface, col] = top_block

    if bottom_surface < grid.shape[0] - 2:
        bottom_block = grid[bottom_surface + 1 : -1, col]
        bottom_block[bottom_block == BLACK] = color
        grid[bottom_surface + 1 : -1, col] = bottom_block


def _draw_rays(grid: np.ndarray, marker_positions: list[tuple[int, int]], tile_h: int) -> None:
    total_h = int(grid.shape[0])
    for r, c in marker_positions:
        if r < tile_h:
            for rr in range(r - 1, 0, -1):
                if grid[rr, c] == BLACK:
                    grid[rr, c] = RED
        else:
            for rr in range(r + 1, total_h):
                if grid[rr, c] == BLACK:
                    grid[rr, c] = RED


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    tile = np.asarray(grid, dtype=int)
    _tile_surfaces, _marker_cols = _validate_tile(tile)
    tile_h, tile_w = map(int, tile.shape)

    composite = _reflect_quadrants(tile)
    surfaces = _composite_surfaces(composite)
    deepest_col = int(np.flatnonzero(surfaces == int(surfaces.max()))[0])
    shallowest_col = int(np.flatnonzero(surfaces == int(surfaces.min()))[-1])
    assert deepest_col != shallowest_col, "extreme columns must be distinct"

    out = composite.copy()
    _paint_column_gaps(out, deepest_col, ORANGE)
    _paint_column_gaps(out, shallowest_col, YELLOW)

    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere(composite == RED)]
    _draw_rays(out, marker_positions, tile_h)

    return out, {
        "tile": tile,
        "composite": composite,
        "surfaces": surfaces,
        "deepest_col": deepest_col,
        "shallowest_col": shallowest_col,
        "marker_positions": marker_positions,
        "tile_h": tile_h,
        "tile_w": tile_w,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect a terrain tile into four quadrants, mark the composite deepest and shallowest columns, and shoot marker rays toward the nearest horizontal edge.

    Input structure:
    - The grid has a black background with a solid grey separator row at row 0.
    - Below the separator, each column contains a contiguous terrain suffix made of green cells; exactly two columns replace the top terrain cell with a red marker.
    - The terrain profile is irregular and left-right asymmetric, so reflecting the tile changes the composite column-extreme layout.

    Rule:
    - Build a 2x2 reflection tiling: upper-left is the input, upper-right is its left-right mirror, and the lower half is the vertical reflection of that full top strip.
    - On the composite grid, recompute each column surface as the highest green-or-red terrain cell; color black gap cells in the leftmost deepest column orange and in the rightmost shallowest column yellow.
    - For every red marker copy, extend a red ray through black cells toward the nearest horizontal edge: upward for markers in the upper half and downward for markers in the lower half.
    - Keep the reflected terrain cells, separator rows, and the red marker pixels themselves unchanged.

    Key insight:
    - The mirror tiling is not a cosmetic preprocessing step: it duplicates terrain columns into a larger symmetric landscape, so the deepest and shallowest selected columns must be computed after reflection rather than on the original tile.
    """
    return _solve_with_metadata(grid)[0]


def _wrong_original_extremes_then_reflect(tile: np.ndarray) -> np.ndarray:
    surfaces, _marker_cols = _validate_tile(tile)
    deepest_col = int(np.flatnonzero(surfaces == int(surfaces.max()))[0])
    shallowest_col = int(np.flatnonzero(surfaces == int(surfaces.min()))[-1])

    painted = np.asarray(tile, dtype=int).copy()
    top_surface = int(surfaces[deepest_col])
    painted[1:top_surface, deepest_col] = ORANGE
    top_surface = int(surfaces[shallowest_col])
    painted[1:top_surface, shallowest_col] = YELLOW

    tile_h = painted.shape[0]
    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere(painted == RED)]
    _draw_rays(painted, marker_positions, tile_h)
    return _reflect_quadrants(painted)


def _expected_gap_rows(composite: np.ndarray, col: int) -> tuple[np.ndarray, np.ndarray]:
    terrain_rows = np.flatnonzero(np.isin(composite[:, col], tuple(TERRAIN_COLORS)))
    top_surface = int(terrain_rows[0])
    bottom_surface = int(terrain_rows[-1])
    top = np.arange(1, top_surface, dtype=int)
    bottom = np.arange(bottom_surface + 1, composite.shape[0] - 1, dtype=int)
    return top, bottom


def _semantic_checks(tile: np.ndarray, out: np.ndarray, meta: dict[str, object]) -> bool:
    composite = np.asarray(meta["composite"], dtype=int)
    surfaces = np.asarray(meta["surfaces"], dtype=int)
    deepest_col = int(meta["deepest_col"])
    shallowest_col = int(meta["shallowest_col"])
    tile_h = int(meta["tile_h"])
    tile_w = int(meta["tile_w"])
    marker_positions = [tuple(map(int, rc)) for rc in meta["marker_positions"]]

    assert out.shape == (2 * tile_h, 2 * tile_w)
    assert np.array_equal(composite[:tile_h, :tile_w], tile)
    assert np.array_equal(composite[:tile_h, tile_w:], np.fliplr(tile))
    assert np.array_equal(composite[tile_h:, :], np.flipud(composite[:tile_h, :]))

    assert deepest_col < tile_w, "leftmost deepest tie-break must land in the left half"
    assert shallowest_col >= tile_w, "rightmost shallowest tie-break must land in the right half"
    assert int(surfaces[deepest_col]) == int(surfaces.max())
    assert int(surfaces[shallowest_col]) == int(surfaces.min())

    orange_cells = np.argwhere(out == ORANGE)
    yellow_cells = np.argwhere(out == YELLOW)
    orange_set = {tuple(map(int, rc)) for rc in orange_cells}
    yellow_set = {tuple(map(int, rc)) for rc in yellow_cells}
    assert orange_cells.size > 0 and yellow_cells.size > 0, "extreme columns must be visible"
    assert np.all(orange_cells[:, 1] == deepest_col), "orange may only appear in the deepest column"
    assert np.all(yellow_cells[:, 1] == shallowest_col), "yellow may only appear in the shallowest column"

    top_rows, bottom_rows = _expected_gap_rows(composite, deepest_col)
    for r in np.concatenate((top_rows, bottom_rows)):
        if composite[int(r), deepest_col] == BLACK:
            assert out[int(r), deepest_col] == ORANGE
    top_rows, bottom_rows = _expected_gap_rows(composite, shallowest_col)
    for r in np.concatenate((top_rows, bottom_rows)):
        if composite[int(r), shallowest_col] == BLACK:
            assert out[int(r), shallowest_col] == YELLOW

    for r, c in marker_positions:
        if r < tile_h:
            ray_rows = range(r - 1, 0, -1)
        else:
            ray_rows = range(r + 1, out.shape[0])
        saw_visible_ray = False
        for rr in ray_rows:
            if composite[rr, c] == BLACK and (rr, c) not in orange_set and (rr, c) not in yellow_set:
                saw_visible_ray = True
                assert out[rr, c] == RED
            elif composite[rr, c] != BLACK:
                assert out[rr, c] == composite[rr, c] or out[rr, c] in (ORANGE, YELLOW)
        assert saw_visible_ray, "each marker copy must produce at least one visible ray cell"

    assert not np.array_equal(out, _wrong_original_extremes_then_reflect(tile)), "mirror-time extreme analysis must matter"
    return True


def _sample_surface_profile(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    lo = 2
    hi = h - 2
    if hi - lo < 2:
        return None

    for _ in range(200):
        profile = np.empty(w, dtype=int)
        profile[0] = int(rng.integers(lo, hi + 1))
        for c in range(1, w):
            prev = int(profile[c - 1])
            candidates = [value for value in range(lo, hi + 1) if abs(value - prev) <= 2]
            weights = np.array([4 if value == prev else 3 if abs(value - prev) == 1 else 1 for value in candidates], dtype=float)
            weights /= weights.sum()
            profile[c] = int(rng.choice(candidates, p=weights))

        diffs = np.diff(profile)
        if int(profile.min()) == int(profile.max()):
            continue
        if np.count_nonzero(profile == int(profile.min())) != 1:
            continue
        if np.count_nonzero(profile == int(profile.max())) != 1:
            continue
        if int(profile.max()) - int(profile.min()) < 2:
            continue
        if not np.any(diffs > 0) or not np.any(diffs < 0):
            continue
        if np.array_equal(profile, profile[::-1]):
            continue
        return profile
    return None


def _choose_marker_columns(rng: np.random.Generator, w: int, deepest: int, shallowest: int) -> list[int] | None:
    candidates = [c for c in range(w) if c not in {deepest, shallowest}]
    if len(candidates) < 2:
        return None

    left = [c for c in candidates if c < w // 2]
    right = [c for c in candidates if c >= w // 2]
    if left and right:
        return [int(rng.choice(left)), int(rng.choice(right))]
    picks = rng.choice(candidates, size=2, replace=False)
    return [int(picks[0]), int(picks[1])]


def _build_tile(surface_profile: np.ndarray, marker_cols: list[int]) -> np.ndarray:
    w = int(surface_profile.size)
    h = int(surface_profile.max()) + 2
    tile = np.full((h, w), BLACK, dtype=int)
    tile[0, :] = GREY
    for c, surface in enumerate(surface_profile):
        tile[int(surface) :, c] = GREEN
    for c in marker_cols:
        tile[int(surface_profile[c]), c] = RED
    return tile


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    w = int(rng.integers(6, 11))

    for _ in range(120):
        provisional_h = int(rng.integers(7, 11))
        surface_profile = _sample_surface_profile(rng, provisional_h, w)
        if surface_profile is None:
            continue

        deepest = int(np.flatnonzero(surface_profile == int(surface_profile.max()))[0])
        shallowest = int(np.flatnonzero(surface_profile == int(surface_profile.min()))[-1])
        marker_cols = _choose_marker_columns(rng, w, deepest, shallowest)
        if marker_cols is None:
            continue

        tile = _build_tile(surface_profile, marker_cols)
        if max(tile.shape) > 15:
            continue

        try:
            out, meta = _solve_with_metadata(tile)
            _semantic_checks(tile, out, meta)
        except AssertionError:
            continue
        return tile
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile = _gen_attempt()
        if tile is not None:
            return tile
    raise RuntimeError("failed to generate a valid mirrored-terrain input")
