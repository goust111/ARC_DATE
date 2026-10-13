from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
PATTERN_COLORS = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _axis_support(grid: np.ndarray, missing: np.ndarray, shift: int, axis: int) -> tuple[int, bool]:
    if axis == 0:
        left = grid[:-shift, :]
        right = grid[shift:, :]
        valid = (~missing[:-shift, :]) & (~missing[shift:, :])
    else:
        left = grid[:, :-shift]
        right = grid[:, shift:]
        valid = (~missing[:, :-shift]) & (~missing[:, shift:])
    support = int(valid.sum())
    if support == 0:
        return 0, True
    return support, bool(np.any(left[valid] != right[valid]))


def _detect_period(grid: np.ndarray, missing: np.ndarray, axis: int) -> int:
    length = int(grid.shape[axis])
    candidates: list[tuple[int, int]] = []
    for shift in range(1, length):
        support, conflict = _axis_support(grid, missing, shift, axis)
        if support > 0 and not conflict:
            candidates.append((support, shift))
    assert candidates, "no translational period found"
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return int(candidates[0][1])


def _marker_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    reds = np.argwhere(grid == RED)
    assert len(reds) == 4, "expected exactly four red markers"
    r0, c0 = reds.min(axis=0)
    r1, c1 = reds.max(axis=0)
    corners = {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}
    assert {tuple(map(int, rc)) for rc in reds} == corners, "red markers must be rectangle corners"
    return int(r0), int(c0), int(r1), int(c1)


def _fill_missing(grid: np.ndarray, missing: np.ndarray, period_r: int, period_c: int) -> np.ndarray:
    filled = grid.copy()
    height, width = map(int, filled.shape)
    for r, c in np.argwhere(missing):
        colors = {
            int(filled[rr, cc])
            for rr in range(int(r) % period_r, height, period_r)
            for cc in range(int(c) % period_c, width, period_c)
            if not missing[rr, cc]
        }
        assert colors, "every missing cell needs a visible orbit representative"
        assert len(colors) == 1, "orbit representatives must agree"
        filled[int(r), int(c)] = colors.pop()
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the translational pattern under the black mask and red corner markers, then crop the marker-defined rectangle.

    Input structure:
    - The whole grid is covered by a 2-3 color translational pattern with no true background.
    - One black rectangle occludes part of that pattern.
    - Exactly four red pixels lie strictly inside the grid and form the four corners of an axis-aligned rectangle.

    Rule:
    - Treat every black cell and every red marker as missing pattern cells.
    - Detect the row and column translation periods from the remaining visible cells.
    - Reconstruct each missing cell from another visible cell in the same translation orbit.
    - Crop the inclusive rectangle whose four corners are the red markers, using the restored colors at those marker positions.

    Key insight:
    - The red pixels have a dual role: they define the crop window, but they are also occlusions that must be repaired before that crop is read out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    marker_r0, marker_c0, marker_r1, marker_c1 = _marker_box(g)

    black_box = _bbox(g == BLACK)
    assert black_box is not None, "expected a black occlusion rectangle"
    br0, bc0, br1, bc1 = black_box
    assert np.all(g[br0 : br1 + 1, bc0 : bc1 + 1] == BLACK), "black cells must form one solid rectangle"

    missing = (g == BLACK) | (g == RED)
    period_r = _detect_period(g, missing, axis=0)
    period_c = _detect_period(g, missing, axis=1)
    restored = _fill_missing(g, missing, period_r, period_c)
    return restored[marker_r0 : marker_r1 + 1, marker_c0 : marker_c1 + 1].astype(int)


def _fundamental_axis_period(tile: np.ndarray, axis: int) -> int:
    length = int(tile.shape[axis])
    for shift in range(1, length):
        if axis == 0:
            if np.array_equal(tile[:-shift, :], tile[shift:, :]):
                return shift
        else:
            if np.array_equal(tile[:, :-shift], tile[:, shift:]):
                return shift
    return length


def _make_tile(rng: np.random.Generator, period_r: int, period_c: int) -> np.ndarray | None:
    color_count = int(rng.integers(2, 4))
    palette = rng.choice(PATTERN_COLORS, size=color_count, replace=False)
    for _ in range(200):
        tile = rng.choice(palette, size=(period_r, period_c), replace=True).astype(int)
        if len(np.unique(tile)) != color_count:
            continue
        if _fundamental_axis_period(tile, axis=0) != period_r:
            continue
        if _fundamental_axis_period(tile, axis=1) != period_c:
            continue
        if any(len(set(map(int, row))) == 1 for row in tile) and any(len(set(map(int, col))) == 1 for col in tile.T):
            continue
        return tile
    return None


def _tile_pattern(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % tile.shape[0]
    cc = np.arange(width)[None, :] % tile.shape[1]
    return tile[rr, cc].astype(int)


def _all_missing_have_visible_orbit(missing: np.ndarray, period_r: int, period_c: int) -> bool:
    height, width = map(int, missing.shape)
    for r, c in np.argwhere(missing):
        found = False
        for rr in range(int(r) % period_r, height, period_r):
            for cc in range(int(c) % period_c, width, period_c):
                if not missing[rr, cc]:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def _best_detected_periods(grid: np.ndarray) -> tuple[int, int]:
    missing = (grid == BLACK) | (grid == RED)
    return _detect_period(grid, missing, axis=0), _detect_period(grid, missing, axis=1)


def _sample_rect(
    rng: np.random.Generator,
    height: int,
    width: int,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    border: int,
) -> tuple[int, int, int, int] | None:
    if height - 2 * border < min_h or width - 2 * border < min_w:
        return None
    rect_h = int(rng.integers(min_h, min(max_h, height - 2 * border) + 1))
    rect_w = int(rng.integers(min_w, min(max_w, width - 2 * border) + 1))
    r0 = int(rng.integers(border, height - border - rect_h + 1))
    c0 = int(rng.integers(border, width - border - rect_w + 1))
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _semantic_ok(
    input_grid: np.ndarray,
    expected: np.ndarray,
    clean: np.ndarray,
    period_r: int,
    period_c: int,
    marker_box: tuple[int, int, int, int],
    black_box: tuple[int, int, int, int],
) -> bool:
    mr0, mc0, mr1, mc1 = marker_box
    br0, bc0, br1, bc1 = black_box
    missing = (input_grid == BLACK) | (input_grid == RED)
    if not _all_missing_have_visible_orbit(missing, period_r, period_c):
        return False
    if _best_detected_periods(input_grid) != (period_r, period_c):
        return False
    if np.array_equal(expected, input_grid[mr0 : mr1 + 1, mc0 : mc1 + 1]):
        return False
    if (mr1 - mr0) % period_r == 0 and (mc1 - mc0) % period_c == 0:
        return False
    corner_colors = {
        int(clean[mr0, mc0]),
        int(clean[mr0, mc1]),
        int(clean[mr1, mc0]),
        int(clean[mr1, mc1]),
    }
    if len(corner_colors) < 2:
        return False
    if len(np.unique(clean[br0 : br1 + 1, bc0 : bc1 + 1])) < 2:
        return False
    if np.any(expected == BLACK) or np.any(expected == RED):
        return False
    if expected.shape[0] <= (br1 - br0 + 1) or expected.shape[1] <= (bc1 - bc0 + 1):
        return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    period_r = int(rng.choice([3, 4]))
    period_c = int(rng.choice([3, 4]))
    height = int(rng.integers(max(11, period_r * 3), min(19, period_r * 5 + 1)))
    width = int(rng.integers(max(11, period_c * 3), min(19, period_c * 5 + 1)))

    tile = _make_tile(rng, period_r, period_c)
    if tile is None:
        return None
    clean = _tile_pattern(tile, height, width)

    marker_box = _sample_rect(rng, height, width, min_h=5, max_h=min(10, height - 2), min_w=6, max_w=min(10, width - 2), border=1)
    if marker_box is None:
        return None
    mr0, mc0, mr1, mc1 = marker_box
    inner_h = mr1 - mr0
    inner_w = mc1 - mc0
    black_box = _sample_rect(rng, inner_h + 1, inner_w + 1, min_h=2, max_h=min(4, inner_h), min_w=3, max_w=min(5, inner_w), border=1)
    if black_box is None:
        return None
    br0, bc0, br1, bc1 = black_box
    br0 += mr0
    br1 += mr0
    bc0 += mc0
    bc1 += mc0

    grid = clean.copy()
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = BLACK
    corners = [(mr0, mc0), (mr0, mc1), (mr1, mc0), (mr1, mc1)]
    for r, c in corners:
        grid[r, c] = RED

    expected = clean[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()
    if not _semantic_ok(grid, expected, clean, period_r, period_c, marker_box, (br0, bc0, br1, bc1)):
        return None
    try:
        if not np.array_equal(main(grid.copy()), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid.astype(int)
    raise RuntimeError("failed to generate a valid dual-marker symmetry-repair input")
