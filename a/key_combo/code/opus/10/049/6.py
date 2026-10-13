from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
NOISE = 7
PATTERN_COLORS = (1, 2, 3, 4, 6, 8, 9)
GENERATION_ATTEMPTS = 500
TILE_ATTEMPTS = 200
NOISE_ATTEMPTS = 120


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _find_noise_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid[grid != BLACK], return_counts=True)
    assert colors.size > 0, "expected at least one non-black color"
    best = int(np.argmin(counts))
    assert int(np.sum(counts == counts[best])) == 1, "expected a unique least frequent noise color"
    return int(colors[best])


def _axis_shift_stats(
    grid: np.ndarray,
    shift: int,
    axis: str,
    ignore_colors: tuple[int, ...] = (),
) -> tuple[bool, int, int]:
    g = np.asarray(grid, dtype=int)
    ignored = np.isin(g, np.array(ignore_colors, dtype=int)) if ignore_colors else np.zeros_like(g, dtype=bool)
    valid = ~ignored
    if axis == "row":
        if shift <= 0 or shift >= g.shape[0]:
            return False, 0, 0
        overlap = valid[:-shift, :] & valid[shift:, :]
        if not overlap.any():
            return False, 0, 0
        left = g[:-shift, :][overlap]
        right = g[shift:, :][overlap]
    else:
        if shift <= 0 or shift >= g.shape[1]:
            return False, 0, 0
        overlap = valid[:, :-shift] & valid[:, shift:]
        if not overlap.any():
            return False, 0, 0
        left = g[:, :-shift][overlap]
        right = g[:, shift:][overlap]
    mismatches = int(np.count_nonzero(left != right))
    return mismatches == 0, int(overlap.sum()), mismatches


def _detect_axis_period(grid: np.ndarray, axis: str, ignore_colors: tuple[int, ...]) -> int:
    limit = grid.shape[0] if axis == "row" else grid.shape[1]
    for shift in range(1, limit):
        ok, support, _mismatches = _axis_shift_stats(grid, shift, axis, ignore_colors)
        if support > 0 and ok:
            return int(shift)
    raise AssertionError(f"failed to detect a {axis} period")


def _minimal_full_axis_period(grid: np.ndarray, axis: str) -> int:
    limit = grid.shape[0] if axis == "row" else grid.shape[1]
    for shift in range(1, limit):
        ok, support, _mismatches = _axis_shift_stats(grid, shift, axis)
        if support > 0 and ok:
            return int(shift)
    return int(limit)


def _orbit_source_value(
    grid: np.ndarray,
    r: int,
    c: int,
    dy: int,
    dx: int,
    forbidden: tuple[int, ...],
) -> int:
    h, w = grid.shape
    blocked = set(map(int, forbidden))
    for dist in range(1, max(h, w) + 1):
        for sign in (-1, 1):
            rr = r + sign * dist * dy
            cc = c + sign * dist * dx
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) not in blocked:
                return int(grid[rr, cc])
    raise AssertionError(f"no orbit source found for {(r, c)}")


def _has_orbit_source(
    shape: tuple[int, int],
    r: int,
    c: int,
    dy: int,
    dx: int,
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = shape
    for dist in range(1, max(h, w) + 1):
        for sign in (-1, 1):
            rr = r + sign * dist * dy
            cc = c + sign * dist * dx
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in blocked:
                return True
    return False


def _restore_full_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    noise = _find_noise_color(g)
    black_mask = g == BLACK
    box = _bbox(black_mask)
    r0, c0, r1, c1 = box
    assert np.all(black_mask[r0 : r1 + 1, c0 : c1 + 1]), "black cells must form one solid rectangle"
    assert int(np.sum(black_mask)) == int((r1 - r0 + 1) * (c1 - c0 + 1)), "expected only the occluding rectangle to be black"

    dy = _detect_axis_period(g, "row", (BLACK, noise))
    dx = _detect_axis_period(g, "col", (BLACK, noise))

    restored = g.copy()
    noise_positions = np.argwhere(g == noise)
    for rr, cc in noise_positions:
        restored[int(rr), int(cc)] = _orbit_source_value(g, int(rr), int(cc), dy, dx, (BLACK, noise))

    for rr, cc in np.argwhere(black_mask):
        restored[int(rr), int(cc)] = _orbit_source_value(restored, int(rr), int(cc), dy, dx, (BLACK,))

    return restored, box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the contents hidden by a black rectangle from a noisy translationally symmetric pattern.

    Input structure:
    - The whole grid is a repeated colored tile with no true background color.
    - One interior black rectangle occludes part of the pattern.
    - Outside the rectangle, a few pixels are replaced by one minority noise color.

    Rule:
    - Count non-black colors and treat the unique least frequent one as noise.
    - Detect the vertical and horizontal translation periods using only cells that are neither black nor noise.
    - Restore the noise pixels first, then restore each black pixel from the nearest valid cell on the same translation orbit.
    - Return only the restored rectangle contents, cropped to the occluder box.

    Key insight:
    - Noise and occlusion are diagnosed differently, but both are repaired by the same translation orbit once the minority noise color has been excluded from period detection.
    """
    restored, box = _restore_full_grid(np.asarray(grid, dtype=int))
    return _crop(restored, box)


def _make_tile(tile_h: int, tile_w: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(TILE_ATTEMPTS):
        tile = np.random.choice(np.array(palette, dtype=int), size=(tile_h, tile_w), replace=True).astype(int)
        if set(map(int, np.unique(tile))) != set(map(int, palette)):
            continue
        if any(len(set(map(int, row))) < 2 for row in tile):
            continue
        if any(len(set(map(int, col))) < 2 for col in tile.T):
            continue
        if _minimal_full_axis_period(tile, "row") != tile_h:
            continue
        if _minimal_full_axis_period(tile, "col") != tile_w:
            continue
        return tile
    return None


def _pick_non_adjacent_noise_positions(
    clean: np.ndarray,
    rect_box: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = clean.shape
    r0, c0, r1, c1 = rect_box
    candidates = [(r, c) for r in range(h) for c in range(w) if not (r0 <= r <= r1 and c0 <= c <= c1)]
    for _ in range(NOISE_ATTEMPTS):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for r, c in candidates:
            if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in chosen):
                continue
            chosen.append((r, c))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            continue
        if len({int(clean[r, c]) for r, c in chosen}) < 2:
            continue
        return chosen
    return None


def _semantic_instance_ok(
    clean: np.ndarray,
    corrupted: np.ndarray,
    rect_box: tuple[int, int, int, int],
    noise_positions: list[tuple[int, int]],
    dy: int,
    dx: int,
) -> bool:
    h, w = corrupted.shape
    black_mask = corrupted == BLACK
    if _bbox(black_mask) != rect_box:
        return False
    r0, c0, r1, c1 = rect_box
    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return False
    if len(_components(black_mask)) != 1:
        return False
    if not np.all(black_mask[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    if int(np.sum(black_mask)) != int((r1 - r0 + 1) * (c1 - c0 + 1)):
        return False

    for i, (r, c) in enumerate(noise_positions):
        if black_mask[r, c]:
            return False
        for rr, cc in noise_positions[i + 1 :]:
            if abs(r - rr) + abs(c - cc) == 1:
                return False

    non_black = corrupted[corrupted != BLACK]
    colors, counts = np.unique(non_black, return_counts=True)
    if set(map(int, colors)) - set(np.unique(clean)) != {NOISE}:
        return False
    least = int(np.argmin(counts))
    if int(colors[least]) != NOISE or int(np.sum(counts == counts[least])) != 1:
        return False

    if _minimal_full_axis_period(clean, "row") != dy:
        return False
    if _minimal_full_axis_period(clean, "col") != dx:
        return False
    if _detect_axis_period(corrupted, "row", (BLACK, NOISE)) != dy:
        return False
    if _detect_axis_period(corrupted, "col", (BLACK, NOISE)) != dx:
        return False

    for shift in range(1, dy):
        ok, support, _mismatches = _axis_shift_stats(corrupted, shift, "row", (BLACK, NOISE))
        if support > 0 and ok:
            return False
    for shift in range(1, dx):
        ok, support, _mismatches = _axis_shift_stats(corrupted, shift, "col", (BLACK, NOISE))
        if support > 0 and ok:
            return False

    noise_breaks_row, row_support, _ = _axis_shift_stats(corrupted, dy, "row", (BLACK,))
    noise_breaks_col, col_support, _ = _axis_shift_stats(corrupted, dx, "col", (BLACK,))
    if row_support == 0 or col_support == 0:
        return False
    if noise_breaks_row or noise_breaks_col:
        return False

    black_cells = {(int(r), int(c)) for r, c in np.argwhere(black_mask)}
    noise_cells = {(int(r), int(c)) for r, c in noise_positions}
    if not all(_has_orbit_source(corrupted.shape, r, c, dy, dx, black_cells | noise_cells) for r, c in noise_positions):
        return False
    if not all(_has_orbit_source(corrupted.shape, r, c, dy, dx, black_cells) for r, c in black_cells):
        return False

    target = _crop(clean, rect_box)
    if len(set(map(int, np.unique(target)))) < 2:
        return False
    if not np.array_equal(main(corrupted.copy()), target):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        tile_h = random.choice((3, 4))
        tile_w = random.choice((3, 4))
        rep_r = random.randint(4, 6)
        rep_c = random.randint(4, 6)
        height = tile_h * rep_r
        width = tile_w * rep_c

        palette_size = random.randint(3, 4)
        palette = random.sample(PATTERN_COLORS, palette_size)
        tile = _make_tile(tile_h, tile_w, palette)
        if tile is None:
            continue

        clean = np.tile(tile, (rep_r, rep_c)).astype(int)
        noise_count = random.randint(5, 8)
        color_counts = Counter(map(int, clean.flat))
        if min(color_counts.values()) <= noise_count:
            continue

        max_rect_h = min(6, height - 2)
        max_rect_w = min(6, width - 2)
        if max_rect_h < 4 or max_rect_w < 4:
            continue
        rect_h = random.randint(4, max_rect_h)
        rect_w = random.randint(4, max_rect_w)
        r0 = random.randint(1, height - rect_h - 1)
        c0 = random.randint(1, width - rect_w - 1)
        rect_box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if len(set(map(int, np.unique(_crop(clean, rect_box))))) < 2:
            continue

        noise_positions = _pick_non_adjacent_noise_positions(clean, rect_box, noise_count)
        if noise_positions is None:
            continue

        corrupted = clean.copy()
        corrupted[r0 : r0 + rect_h, c0 : c0 + rect_w] = BLACK
        for rr, cc in noise_positions:
            corrupted[rr, cc] = NOISE

        if _semantic_instance_ok(clean, corrupted, rect_box, noise_positions, tile_h, tile_w):
            return corrupted

    raise RuntimeError("failed to generate a valid noisy translational-occlusion ARC input")
