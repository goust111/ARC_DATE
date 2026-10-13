from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 6
GEN_RETRIES = 400
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _canonicalize_band(band: np.ndarray) -> tuple[np.ndarray, bool]:
    coords = np.argwhere(band != BG)
    assert len(coords) > 0, "expected a non-empty band"
    diag_offsets = coords[:, 1] - coords[:, 0]
    anti_offsets = coords[:, 1] + coords[:, 0]
    diag_span = int(diag_offsets.max() - diag_offsets.min())
    anti_span = int(anti_offsets.max() - anti_offsets.min())
    if diag_span < anti_span:
        return band.copy(), False
    return np.fliplr(band), True


def _learn_band_pattern(band: np.ndarray) -> tuple[int, tuple[int, ...], dict[tuple[int, int], int]]:
    coords = np.argwhere(band != BG)
    assert len(coords) > 0, "expected visible band cells"
    offsets = coords[:, 1] - coords[:, 0]
    min_offset = int(offsets.min())
    max_offset = int(offsets.max())
    offset_seq = tuple(range(min_offset, max_offset + 1))

    h, w = band.shape
    observations: list[tuple[int, int, int]] = []
    for r in range(h):
        for offset in offset_seq:
            c = r + offset
            if 0 <= c < w:
                observations.append((r, offset, int(band[r, c])))

    assert observations, "expected diagonal observations"
    for period in range(1, h + 1):
        pattern: dict[tuple[int, int], int] = {}
        ok = True
        for r, offset, color in observations:
            key = (offset, r % period)
            if key in pattern and pattern[key] != color:
                ok = False
                break
            pattern[key] = color
        if not ok:
            continue
        if any((offset, phase) not in pattern for offset in offset_seq for phase in range(period)):
            continue
        return period, offset_seq, pattern

    raise AssertionError("failed to infer a periodic diagonal band")


def _paint_band(
    shape: tuple[int, int],
    offsets: tuple[int, ...],
    period: int,
    pattern: dict[tuple[int, int], int],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    for r in range(h):
        phase = r % period
        for offset in offsets:
            c = r + offset
            if 0 <= c < w:
                color = pattern[(offset, phase)]
                if color != BG:
                    out[r, c] = color
    return out


def _mirror_walls(shape: tuple[int, int], walls: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    out = np.zeros((2 * h, 2 * w), dtype=bool)
    for r, c in walls:
        for rr, cc in (
            (r, c),
            (r, 2 * w - 1 - c),
            (2 * h - 1 - r, c),
            (2 * h - 1 - r, 2 * w - 1 - c),
        ):
            out[rr, cc] = True
    return out


def _reachable_background(band: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    h, w = band.shape
    passable = (band == BG) & (~wall_mask)
    reachable = np.zeros((h, w), dtype=bool)
    if not passable[0, 0]:
        return reachable

    queue = deque([(0, 0)])
    reachable[0, 0] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if _in_bounds((h, w), rr, cc) and passable[rr, cc] and not reachable[rr, cc]:
                reachable[rr, cc] = True
                queue.append((rr, cc))
    return reachable


def _apply_visibility(band: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    reachable = _reachable_background(band, wall_mask)
    out = np.zeros_like(band)
    h, w = band.shape
    for r in range(h):
        for c in range(w):
            if band[r, c] == BG or wall_mask[r, c]:
                continue
            if any(
                _in_bounds((h, w), r + dr, c + dc) and reachable[r + dr, c + dc]
                for dr, dc in DIRS4
            ):
                out[r, c] = band[r, c]
    out[wall_mask] = WALL
    return out


def _solve_layers(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert max(h, w) <= 15, "input should be a small ARC grid"

    band_only = np.where(g == WALL, BG, g)
    canonical_band, flipped = _canonicalize_band(band_only)
    period, offsets, pattern = _learn_band_pattern(canonical_band)

    expanded_band = _paint_band((2 * h, 2 * w), offsets, period, pattern)
    if flipped:
        expanded_band = np.fliplr(expanded_band)

    walls = [tuple(map(int, rc)) for rc in np.argwhere(g == WALL)]
    wall_mask = _mirror_walls((h, w), walls)
    final = _apply_visibility(expanded_band, wall_mask)
    return expanded_band, wall_mask, final


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a periodic diagonal band to a doubled canvas, mirror the walls, and erase wall-hidden band segments.

    Input structure:
    - The grid is mostly black background with one narrow diagonal color band and a few magenta wall pixels.
    - The band occupies one or two diagonal offsets and uses a row-periodic two-color pattern that may include background gaps.
    - Walls never belong to the band and act only as later visibility blockers.

    Rule:
    - Infer the band direction, canonical diagonal offsets, and smallest row period; if needed, horizontally flip to learn the down-right form.
    - Paint that same periodic band onto a 2H x 2W canvas, undo the flip, and mirror every input wall pixel into all four quadrants.
    - Flood-fill reachable background from (0, 0); keep only band pixels adjacent to reachable background, erase the rest to black, and leave walls magenta.

    Key insight:
    - The band is generated first and only then filtered by reachability, so the mirrored walls matter by hiding already-drawn band cells rather than by stopping the drawing process.
    """
    _expanded_band, _wall_mask, final = _solve_layers(grid)
    return final


def _tile_has_exact_period(tile: np.ndarray) -> bool:
    period = tile.shape[0]
    for smaller in range(1, period):
        if np.array_equal(tile, np.roll(tile, -smaller, axis=0)):
            return False
    return True


def _sample_tile(period: int, width: int, colors: tuple[int, int]) -> np.ndarray | None:
    color_a, color_b = colors
    palette = [BG, color_a, color_b]
    for _ in range(80):
        tile = np.array(
            [[random.choice(palette) for _ in range(width)] for _ in range(period)],
            dtype=int,
        )
        if set(int(x) for x in tile.ravel()) & {color_a, color_b} != {color_a, color_b}:
            continue
        if any(np.all(tile[:, col] == BG) for col in range(width)):
            continue
        if not _tile_has_exact_period(tile):
            continue
        return tile
    return None


def _paint_canonical_full(
    h: int,
    w: int,
    min_offset: int,
    tile: np.ndarray,
) -> np.ndarray:
    offsets = tuple(min_offset + i for i in range(tile.shape[1]))
    pattern = {
        (offsets[col], row): int(tile[row, col])
        for row in range(tile.shape[0])
        for col in range(tile.shape[1])
    }
    return _paint_band((2 * h, 2 * w), offsets, tile.shape[0], pattern)


def _wall_l_shape(h: int, w: int, top_len: int, left_len: int) -> list[tuple[int, int]]:
    row0 = h - left_len
    col0 = w - top_len
    cells = {(row0, c) for c in range(col0, w)}
    cells |= {(r, col0) for r in range(row0, h)}
    return sorted(cells)


def _band_variant(
    canonical_full: np.ndarray,
    h: int,
    w: int,
    slash: bool,
) -> np.ndarray:
    cropped = canonical_full[:h, :w].copy()
    return np.fliplr(cropped) if slash else cropped


def _without_mirrored_walls(full_band: np.ndarray, h: int, w: int, walls: list[tuple[int, int]]) -> np.ndarray:
    wall_mask = np.zeros_like(full_band, dtype=bool)
    for r, c in walls:
        wall_mask[r, c] = True
    return _apply_visibility(full_band, wall_mask)


def _semantic_checks(inp: np.ndarray) -> bool:
    h, w = inp.shape
    full_band, wall_mask, final = _solve_layers(inp)

    if not np.any(full_band[h:, :] != BG) and not np.any(full_band[:, w:] != BG):
        return False

    visible_band = (final != BG) & (final != WALL)
    if np.count_nonzero(visible_band) == 0:
        return False

    erased_to_bg = (full_band != BG) & (~wall_mask) & (final == BG)
    if np.count_nonzero(erased_to_bg) == 0:
        return False

    overlaid_only = full_band.copy()
    overlaid_only[wall_mask] = WALL
    if np.array_equal(overlaid_only, final):
        return False

    walls = [tuple(map(int, rc)) for rc in np.argwhere(inp == WALL)]
    no_mirror_result = _without_mirrored_walls(full_band, h, w, walls)
    if np.array_equal(no_mirror_result, final):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(7, 10)
    w = random.randint(7, 10)
    period = random.choice([2, 3])
    width = random.choice([1, 2])
    min_offset = random.choice([0, 1])
    if min_offset == 0 and width == 2 and w < 3:
        return None

    colors = tuple(random.sample([1, 2, 3, 4, 5, 7, 8, 9], 2))
    tile = _sample_tile(period, width, colors)
    if tile is None:
        return None

    if min_offset == 0:
        tile[0, 0] = BG
        if any(np.all(tile[:, col] == BG) for col in range(width)):
            return None
        if set(int(x) for x in tile.ravel()) & set(colors) != set(colors):
            return None
        if not _tile_has_exact_period(tile):
            return None

    canonical_full = _paint_canonical_full(h, w, min_offset, tile)
    slash = random.choice([False, True])
    band_input = _band_variant(canonical_full, h, w, slash)

    valid_lengths = [
        (top_len, left_len)
        for top_len in (2, 3)
        for left_len in (2, 3, 4)
        if top_len + left_len - 1 <= 6
    ]
    random.shuffle(valid_lengths)
    for top_len, left_len in valid_lengths:
        walls = _wall_l_shape(h, w, top_len, left_len)
        if any(band_input[r, c] != BG for r, c in walls):
            continue
        if any((r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)} for r, c in walls):
            continue

        inp = band_input.copy()
        for r, c in walls:
            inp[r, c] = WALL
        try:
            _ = main(inp)
        except Exception:
            continue
        if _semantic_checks(inp):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
