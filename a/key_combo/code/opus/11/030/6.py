from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
PALETTE = np.array([1, 2, 4, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 500


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    sentinel = list(map(bool, values)) + [False]
    for idx, value in enumerate(sentinel):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    return runs


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _reflection_orbit(row: int, col: int, height: int, width: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (row, col),
                (row, width - 1 - col),
                (height - 1 - row, col),
                (height - 1 - row, width - 1 - col),
            }
        )
    )


def _repair_tile(tile: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = map(int, tile.shape)
    out = tile.copy()
    seen: set[tuple[tuple[int, int], ...]] = set()
    repaired = 0
    for r in range(h):
        for c in range(w):
            orbit = _reflection_orbit(r, c, h, w)
            if orbit in seen:
                continue
            seen.add(orbit)
            values = [int(tile[rr, cc]) for rr, cc in orbit]
            visible = {value for value in values if value != GREEN}
            assert len(visible) == 1, "each damaged orbit must expose exactly one non-green color"
            fill = visible.pop()
            for rr, cc in orbit:
                if int(tile[rr, cc]) == GREEN:
                    out[rr, cc] = fill
                    repaired += 1
    assert repaired == int(np.count_nonzero(tile == GREEN))
    return out, repaired


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "tile must have a unique representative color"
    return int(ranked[0][0])


def _band_tiles(grid: np.ndarray) -> list[dict]:
    full_separator_rows = np.all(grid == GREY, axis=1)
    band_runs = _runs(~full_separator_rows)
    assert band_runs, "expected at least one band"

    tiles: list[dict] = []
    for band_index, (r0, r1) in enumerate(band_runs):
        band = grid[r0 : r1 + 1, :]
        mask = (band != BLACK) & (band != GREY)
        comps = _components(mask)
        assert len(comps) == 1, "each band must contain exactly one tile component"
        bbox = _bbox(comps[0])
        tr0, tc0, tr1, tc1 = bbox
        tile = band[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
        assert tile.shape[0] == tile.shape[1] and tile.shape[0] in (3, 4), "tile must be 3x3 or 4x4"
        assert not np.any((tile == BLACK) | (tile == GREY)), "tile bbox must be fully occupied"
        repaired_tile, repaired_count = _repair_tile(tile)
        tiles.append(
            {
                "band_index": band_index,
                "tile": tile,
                "repaired_tile": repaired_tile,
                "repaired_count": repaired_count,
                "representative": _unique_mode(repaired_tile),
            }
        )
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair symmetric damaged tiles in separator-defined bands, then compress and sort them by repair cost.

    Input structure:
    - Full-width grey rows split the canvas into horizontal bands on a black background.
    - Each band contains exactly one square non-black tile, sized 3x3 or 4x4.
    - Inside each tile, green pixels are placeholders inside a pattern that is symmetric under both left-right and top-bottom reflection.

    Rule:
    - In each band, locate the unique tile and repair every green cell by copying the unique visible color from its horizontal/vertical reflection orbit.
    - Count how many pixels were repaired in that tile, then take the unique majority color of the repaired tile as its representative color.
    - Sort tiles by repaired-pixel count from fewest to most, breaking ties by their original top-to-bottom order, and output a 1xN row of those representative colors.

    Key insight:
    - The green damage encodes two things at once: it must be resolved by local two-axis symmetry to recover each tile's true majority color, and its amount determines the output order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    tiles = _band_tiles(g)
    ordered = sorted(tiles, key=lambda item: (item["repaired_count"], item["band_index"]))
    return np.array([[item["representative"] for item in ordered]], dtype=int)


def _orbit_cells_4x4() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0), (0, 3), (3, 0), (3, 3)],
        [(0, 1), (0, 2), (3, 1), (3, 2)],
        [(1, 0), (1, 3), (2, 0), (2, 3)],
        [(1, 1), (1, 2), (2, 1), (2, 2)],
    ]


def _sample_color(exclude: set[int]) -> int:
    choices = [int(color) for color in PALETTE if int(color) not in exclude]
    return int(RNG.choice(np.array(choices, dtype=int)))


def _make_3x3_tile(base: int, damage_count: int) -> np.ndarray:
    accent = _sample_color({base})
    tile = np.full((3, 3), base, dtype=int)
    tile[0, 1] = accent
    tile[1, 0] = accent
    tile[1, 2] = accent
    tile[2, 1] = accent
    corner_cells = np.array([(0, 0), (0, 2), (2, 0), (2, 2)], dtype=int)
    chosen = RNG.choice(len(corner_cells), size=damage_count, replace=False)
    for idx in chosen:
        r, c = corner_cells[int(idx)]
        tile[r, c] = GREEN
    return tile


def _make_4x4_tile(base: int, damage_count: int) -> np.ndarray:
    orbits = _orbit_cells_4x4()
    tile = np.empty((4, 4), dtype=int)
    mode = int(RNG.integers(2))
    orbit_colors = [base] * 4

    if mode == 0:
        accent_orbit = int(RNG.integers(4))
        orbit_colors[accent_orbit] = _sample_color({base})
        repair_orbits = [idx for idx in range(4) if idx != accent_orbit]
    else:
        accent_orbits = list(map(int, RNG.choice(4, size=2, replace=False)))
        accent_a = _sample_color({base})
        accent_b = _sample_color({base, accent_a})
        orbit_colors[accent_orbits[0]] = accent_a
        orbit_colors[accent_orbits[1]] = accent_b
        repair_orbits = [idx for idx in range(4) if idx not in accent_orbits]

    for orbit_idx, cells in enumerate(orbits):
        for r, c in cells:
            tile[r, c] = orbit_colors[orbit_idx]

    repair_orbit = int(RNG.choice(np.array(repair_orbits, dtype=int)))
    cells = np.array(orbits[repair_orbit], dtype=int)
    chosen = RNG.choice(len(cells), size=damage_count, replace=False)
    for idx in chosen:
        r, c = cells[int(idx)]
        tile[r, c] = GREEN
    return tile


def _visible_majority_matches(tile: np.ndarray, representative: int) -> bool:
    visible = tile[tile != GREEN]
    counts = Counter(map(int, visible.reshape(-1)))
    ranked = counts.most_common()
    if not ranked:
        return False
    if len(ranked) == 1:
        return int(ranked[0][0]) == int(representative)
    if ranked[0][1] == ranked[1][1]:
        return False
    return int(ranked[0][0]) == int(representative)


def _assemble_grid(tiles: list[np.ndarray], band_height: int, width: int) -> np.ndarray:
    band_count = len(tiles)
    height = band_count * band_height + (band_count - 1)
    grid = np.full((height, width), BLACK, dtype=int)
    for idx in range(band_count - 1):
        grid[(idx + 1) * band_height + idx, :] = GREY

    for band_index, tile in enumerate(tiles):
        band_top = band_index * (band_height + 1)
        tile_h, tile_w = tile.shape
        row_offset = 0 if tile_h == band_height else int(RNG.integers(0, band_height - tile_h + 1))
        col_offset = (width - tile_w) // 2
        grid[band_top + row_offset : band_top + row_offset + tile_h, col_offset : col_offset + tile_w] = tile
    return grid


def _make_band_spec() -> tuple[int, list[int], list[int], list[int]]:
    mode = int(RNG.integers(2))
    band_count = int(RNG.choice(np.array([4, 5], dtype=int)))
    if mode == 0:
        band_height = 3
        tile_sizes = [3] * band_count
    else:
        band_height = 4
        tile_sizes = list(map(int, RNG.choice(np.array([3, 4], dtype=int), size=band_count, replace=True)))
        if 3 not in tile_sizes:
            tile_sizes[int(RNG.integers(band_count))] = 3
        if 4 not in tile_sizes:
            tile_sizes[int(RNG.integers(band_count))] = 4

    counts = list(map(int, RNG.choice(np.array([1, 2, 3], dtype=int), size=band_count, replace=True)))
    if len(set(counts)) < 2 or counts == sorted(counts):
        return _make_band_spec()
    return band_height, tile_sizes, counts, list(map(int, RNG.choice(PALETTE, size=band_count, replace=False)))


def _oracle_output(representatives: list[int], counts: list[int]) -> np.ndarray:
    order = sorted(range(len(representatives)), key=lambda idx: (counts[idx], idx))
    return np.array([[int(representatives[idx]) for idx in order]], dtype=int)


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert expected.ndim == 2 and max(expected.shape) <= 30

    tiles = _band_tiles(grid)
    assert len(tiles) in (4, 5), "generator must produce four or five bands"
    assert np.array_equal(main(grid), expected), "solver must match the generator oracle"

    representatives = [int(item["representative"]) for item in tiles]
    repaired_counts = [int(item["repaired_count"]) for item in tiles]
    assert len(set(representatives)) == len(representatives), "representative colors must stay distinct"
    assert repaired_counts != sorted(repaired_counts), "sorting must change band order"
    assert len(set(repaired_counts)) >= 2, "repair counts must not collapse to one value"

    needs_completion = False
    has_small_tile = False
    has_large_tile = False
    for item in tiles:
        tile = item["tile"]
        repaired = item["repaired_tile"]
        rep = int(item["representative"])
        if tile.shape == (3, 3):
            has_small_tile = True
        if tile.shape == (4, 4):
            has_large_tile = True
        assert np.array_equal(repaired, _repair_tile(tile)[0])
        assert int(np.count_nonzero(tile == GREEN)) == int(item["repaired_count"])
        if not _visible_majority_matches(tile, rep):
            needs_completion = True

    assert has_small_tile, "every example must include at least one 3x3 tile"
    assert needs_completion, "at least one tile must require symmetry repair to recover its majority"
    if np.any(np.all(grid == GREY, axis=1)):
        assert len(_runs(~np.all(grid == GREY, axis=1))) == len(tiles)
    if has_large_tile:
        assert any(tile["tile"].shape == (4, 4) for tile in tiles)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        band_height, tile_sizes, counts, representatives = _make_band_spec()
        width = int(RNG.integers(9 if band_height == 3 else 10, 15 if band_height == 3 else 17))
        width = max(width, max(tile_sizes) + 4)

        tiles: list[np.ndarray] = []
        for size, count, rep in zip(tile_sizes, counts, representatives):
            tile = _make_3x3_tile(rep, count) if size == 3 else _make_4x4_tile(rep, count)
            tiles.append(tile)

        grid = _assemble_grid(tiles, band_height, width)
        expected = _oracle_output(representatives, counts)
        try:
            _validate_generated(grid, expected)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
