from __future__ import annotations

import numpy as np


BLACK = 0
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 400


def _mirror_orbit(size: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    orbit = {
        (int(r), int(c)),
        (size - 1 - int(r), int(c)),
        (int(r), size - 1 - int(c)),
        (size - 1 - int(r), size - 1 - int(c)),
    }
    return tuple(sorted(orbit))


def _all_orbits(size: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = _mirror_orbit(size, r, c)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _mirror_restore(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    size = int(g.shape[0])
    out = g.copy()
    for orbit in _all_orbits(size):
        colors = {int(g[r, c]) for r, c in orbit if int(g[r, c]) != BLACK}
        if not colors:
            continue
        assert len(colors) == 1, "mirror orbit colors must agree"
        color = colors.pop()
        for r, c in orbit:
            out[r, c] = color
    return out


def _tile_for_period(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray | None:
    g = np.asarray(grid, dtype=int)
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    for r, c in np.argwhere(g != BLACK):
        phase_r = int(r) % tile_h
        phase_c = int(c) % tile_w
        color = int(g[r, c])
        current = int(tile[phase_r, phase_c])
        if current == -1:
            tile[phase_r, phase_c] = color
        elif current != color:
            return None
    if np.any(tile < 0):
        return None
    return tile


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    tile_h, tile_w = map(int, tile.shape)
    rows = np.arange(h)[:, None] % tile_h
    cols = np.arange(w)[None, :] % tile_w
    return tile[rows, cols]


def _infer_minimal_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    best_area: int | None = None
    best_tile: np.ndarray | None = None
    best_output: np.ndarray | None = None

    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            area = tile_h * tile_w
            if best_area is not None and area > best_area:
                continue
            tile = _tile_for_period(g, tile_h, tile_w)
            if tile is None:
                continue
            tiled = _repeat_tile(tile, g.shape)
            if best_area is None or area < best_area:
                best_area = area
                best_tile = tile
                best_output = tiled
            else:
                assert best_output is not None
                assert np.array_equal(best_output, tiled), "minimal tile is ambiguous"

    assert best_tile is not None, "no fully observed periodic tile fits the known cells"
    return best_tile


def _minimal_period(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)

    row_period = h
    for period in range(1, h + 1):
        if all(np.array_equal(g[r], g[r % period]) for r in range(h)):
            row_period = period
            break

    col_period = w
    for period in range(1, w + 1):
        if all(np.array_equal(g[:, c], g[:, c % period]) for c in range(w)):
            col_period = period
            break

    return int(row_period), int(col_period)


def _phase_groups(period: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    groups: list[tuple[tuple[int, int], ...]] = []
    for r in range(period):
        for c in range(period):
            if (r, c) in seen:
                continue
            group = {
                (r, c),
                (period - 1 - r, c),
                (r, period - 1 - c),
                (period - 1 - r, period - 1 - c),
            }
            frozen = tuple(sorted(group))
            seen.update(frozen)
            groups.append(frozen)
    return groups


def _missing_phases(grid: np.ndarray, period: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    missing: list[tuple[int, int]] = []
    for r in range(period):
        for c in range(period):
            phase_mask = (
                (np.arange(g.shape[0])[:, None] % period == r)
                & (np.arange(g.shape[1])[None, :] % period == c)
            )
            if not np.any(g[phase_mask] != BLACK):
                missing.append((r, c))
    return missing


def _build_symmetric_tile(period: int) -> np.ndarray:
    for _ in range(200):
        color_count = int(np.random.choice([3, 4]))
        colors = np.random.choice(PALETTE, size=color_count, replace=False)
        if period == 3 and color_count == 3:
            a, b, c = map(int, colors)
            d = a
        elif period == 4 and color_count == 3:
            a, b, c = map(int, colors)
            d = b
        else:
            a, b, c, d = map(int, colors[:4])
        if period == 3:
            tile = np.array(
                [
                    [a, b, a],
                    [c, d, c],
                    [a, b, a],
                ],
                dtype=int,
            )
        else:
            tile = np.array(
                [
                    [a, b, b, a],
                    [c, d, d, c],
                    [c, d, d, c],
                    [a, b, b, a],
                ],
                dtype=int,
            )
        tile_color_count = len(np.unique(tile))
        if tile_color_count not in (3, 4):
            continue
        probe = _repeat_tile(tile, (period * 4, period * 4))
        if _minimal_period(probe) != (period, period):
            continue
        return tile
    raise RuntimeError("failed to build a valid symmetric tile")


def _generate_case() -> np.ndarray | None:
    period = int(np.random.choice([3, 4]))
    size = 12 if period == 3 else int(np.random.choice([12, 16]))
    tile = _build_symmetric_tile(period)
    full = _repeat_tile(tile, (size, size))

    inp = full.copy()
    nontrivial_groups = [group for group in _phase_groups(period) if len(group) > 1]
    group_count = 1 if np.random.random() < 0.55 else min(2, len(nontrivial_groups))
    chosen_groups = np.random.choice(len(nontrivial_groups), size=group_count, replace=False)
    hidden_phases = [nontrivial_groups[int(idx)][np.random.randint(len(nontrivial_groups[int(idx)]))] for idx in chosen_groups]

    rr = np.arange(size)[:, None]
    cc = np.arange(size)[None, :]
    for phase_r, phase_c in hidden_phases:
        inp[(rr % period == phase_r) & (cc % period == phase_c)] = BLACK

    orbits = _all_orbits(size)
    np.random.shuffle(orbits)
    full_orbit_count = int(np.random.randint(1, 5 if size == 12 else 7))
    for orbit in orbits[:full_orbit_count]:
        for r, c in orbit:
            inp[r, c] = BLACK

    mirrored = _mirror_restore(inp)

    if not np.any(inp == BLACK):
        return None
    if not np.any((inp == BLACK) & (mirrored != BLACK)):
        return None
    if not np.any(mirrored == BLACK):
        return None
    if np.array_equal(mirrored, full):
        return None
    if len(np.unique(inp[inp != BLACK])) < 3:
        return None
    if not _missing_phases(inp, period):
        return None
    if _missing_phases(mirrored, period):
        return None
    if _minimal_period(_repeat_tile(tile, (period * 4, period * 4))) != (period, period):
        return None

    solved = main(inp)
    if not np.array_equal(solved, full):
        return None

    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a doubly symmetric periodic wallpaper by mirroring visible orbit-mates first, then inferring the repeating tile.

    Input structure:
    - The input is a square grid whose hidden full pattern uses no black and is symmetric across both the horizontal and vertical center axes.
    - The same hidden pattern is also a periodic wallpaper obtained by repeating one small square tile.
    - Some cells are black holes: some mirror orbits still expose a non-black mate, while other orbits are fully black and stay unresolved after mirroring.

    Rule:
    - For each black cell, inspect its horizontal, vertical, and 180-degree mirror orbit inside the square grid.
    - If any cell in that orbit is non-black, copy that orbit color into every black member of the orbit.
    - Ignore any black cells still left, infer the smallest fully determined repeating tile consistent with the mirrored grid, and retile the whole canvas from it.

    Key insight:
    - Mirroring and periodicity are complementary: mirroring restores enough hidden phases to make the wallpaper tile uniquely inferable, and periodicity finishes the orbits that were completely black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"

    mirrored = _mirror_restore(g)
    tile = _infer_minimal_tile(mirrored)
    out = _repeat_tile(tile, g.shape)
    assert np.all(out != BLACK), "recovered wallpaper must be fully colored"
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-period recovery input")
