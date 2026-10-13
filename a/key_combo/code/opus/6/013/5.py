from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEP = 5
PATTERN_COLORS = (BLUE, GREEN)
OCCLUDER_COLORS = (RED, YELLOW)
TILE_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
HALF_HEIGHT_RANGE = (6, 8)
WIDTH_RANGE = (9, 13)
MAX_GENERATION_ATTEMPTS = 500
RECTANGLE_ATTEMPTS = 120
SET_ATTEMPTS = 120
TILE_ATTEMPTS = 80
MASK_COVER_MIN = 0.15
MASK_COVER_MAX = 0.38


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_full_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _repeat_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    return tile[rr, cc]


def _phase_repeat(tile: np.ndarray, shape: tuple[int, int], row_phase: int, col_phase: int) -> np.ndarray:
    h, w = shape
    rr = (np.arange(h)[:, None] + row_phase) % tile.shape[0]
    cc = (np.arange(w)[None, :] + col_phase) % tile.shape[1]
    return tile[rr, cc]


def _row_period(grid: np.ndarray) -> int:
    h = grid.shape[0]
    for period in range(1, h + 1):
        ok = True
        for r in range(h):
            if not np.array_equal(grid[r], grid[r % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("row period should exist")


def _col_period(grid: np.ndarray) -> int:
    w = grid.shape[1]
    for period in range(1, w + 1):
        ok = True
        for c in range(w):
            if not np.array_equal(grid[:, c], grid[:, c % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("column period should exist")


def _minimal_tile(grid: np.ndarray) -> np.ndarray:
    rp = _row_period(grid)
    cp = _col_period(grid)
    return grid[:rp, :cp].copy()


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEP)]
    assert len(rows) == 1, "expected one full separator row"
    sep = rows[0]
    assert grid.shape[0] == 2 * sep + 1, "separator must split the grid into equal halves"
    return int(sep)


def _validate_half(half: np.ndarray) -> np.ndarray:
    allowed = set(PATTERN_COLORS) | set(OCCLUDER_COLORS)
    assert set(map(int, np.unique(half))).issubset(allowed), "unexpected color in half-grid"
    occluder_mask = ~np.isin(half, PATTERN_COLORS)
    parts = _components(occluder_mask)
    assert 1 <= len(parts) <= 2, "each half should contain one or two occluders"
    for coords in parts:
        assert _is_full_rectangle(coords), "occluders must be solid rectangles"
        color = int(half[coords[0, 0], coords[0, 1]])
        assert color in OCCLUDER_COLORS
        assert np.all(half[coords[:, 0], coords[:, 1]] == color)
    return occluder_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the lower half, merge its complementary visible pattern, and repaint the upper half with the recovered minimal tile.

    Input structure:
    - A black-based ARC grid is split by one full grey separator row into equal-height upper and lower halves.
    - Both halves are filled by the same blue/green periodic background, with the lower half being the vertical mirror of the upper half.
    - Each half also contains one or two solid rectangular occluders in red or yellow, and the mirrored lower occluders do not overlap the upper occluders.

    Rule:
    - Flip the lower half vertically across the separator so it aligns with the upper half.
    - Wherever the upper half has an occluder, copy the aligned visible pattern cell from the mirrored lower half; keep all other upper cells unchanged.
    - Extract the smallest repeating rectangular tile from the merged upper canvas and return a fresh upper-half-sized tiling of that tile.

    Key insight:
    - The two halves hide different parts of the same periodic image, so only after mirroring and overlaying them do you recover enough evidence to infer the exact repeating unit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    top = g[:sep].copy()
    bottom = g[sep + 1 :].copy()
    assert top.shape == bottom.shape

    top_mask = _validate_half(top)
    mirrored_bottom = np.flipud(bottom)
    mirrored_mask = _validate_half(mirrored_bottom)
    assert not np.any(top_mask & mirrored_mask), "upper and mirrored lower occluders must be complementary"

    merged = top.copy()
    merged[top_mask] = mirrored_bottom[top_mask]
    assert set(map(int, np.unique(merged))).issubset(set(PATTERN_COLORS)), "merged canvas must be pure pattern"

    tile = _minimal_tile(merged)
    out = _repeat_tile(tile, merged.shape)
    assert np.array_equal(out, merged), "minimal tile must exactly regenerate the merged pattern"
    return out


def _random_tile(shape: tuple[int, int]) -> np.ndarray | None:
    rp, cp = shape
    for _ in range(TILE_ATTEMPTS):
        tile = np.array(
            [[random.choice(PATTERN_COLORS) for _ in range(cp)] for _ in range(rp)],
            dtype=int,
        )
        if set(map(int, tile.ravel())) != set(PATTERN_COLORS):
            continue
        if _minimal_tile(tile).shape != tile.shape:
            continue
        return tile
    return None


def _random_rectangles(
    h: int,
    w: int,
    count: int,
    forbidden: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]] | None:
    occ = np.zeros((h, w), dtype=bool) if forbidden is None else forbidden.copy()
    same_half_occ = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    for idx in range(count):
        placed = False
        remaining = count - idx
        for _ in range(RECTANGLE_ATTEMPTS):
            rect_h = random.randint(1, max(1, h - 1))
            rect_w = random.randint(1, max(1, min(w - 1, max(3, w // 2 + 1))))
            if rect_h > h or rect_w > w:
                continue
            r0 = random.randint(0, h - rect_h)
            c0 = random.randint(0, w - rect_w)
            patch = occ[r0 : r0 + rect_h, c0 : c0 + rect_w]
            if patch.any():
                continue
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h, r0 + rect_h + 1)
            pc1 = min(w, c0 + rect_w + 1)
            if same_half_occ[pr0:pr1, pc0:pc1].any():
                continue
            new_area = sum(rh * rw for _, _, rh, rw in rects) + rect_h * rect_w
            min_possible = new_area + (remaining - 1)
            max_possible = new_area + (h * w) * (remaining - 1)
            low = int(np.ceil(MASK_COVER_MIN * h * w))
            high = int(np.floor(MASK_COVER_MAX * h * w))
            if min_possible > high or max_possible < low:
                continue
            rects.append((r0, c0, rect_h, rect_w))
            occ[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
            same_half_occ[pr0:pr1, pc0:pc1] = True
            placed = True
            break
        if not placed:
            return None
    area = sum(rh * rw for _, _, rh, rw in rects)
    if not (int(np.ceil(MASK_COVER_MIN * h * w)) <= area <= int(np.floor(MASK_COVER_MAX * h * w))):
        return None
    return rects


def _mask_from_rectangles(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, rh, rw in rects:
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _mirror_rectangles(rects: list[tuple[int, int, int, int]], h: int) -> list[tuple[int, int, int, int]]:
    return [(h - (r0 + rh), c0, rh, rw) for r0, c0, rh, rw in rects]


def _residues_touched(mask: np.ndarray, rp: int, cp: int) -> int:
    seen = np.zeros((rp, cp), dtype=bool)
    for r, c in np.argwhere(mask):
        seen[int(r) % rp, int(c) % cp] = True
    return int(seen.sum())


def _make_input(
    full_top: np.ndarray,
    top_rects: list[tuple[int, int, int, int]],
    mirrored_bottom_rects: list[tuple[int, int, int, int]],
    top_color: int,
    bottom_color: int,
) -> np.ndarray:
    h, w = full_top.shape
    grid = np.zeros((2 * h + 1, w), dtype=int)
    top = full_top.copy()
    bottom = np.flipud(full_top).copy()
    top_mask = _mask_from_rectangles((h, w), top_rects)
    bottom_rects = _mirror_rectangles(mirrored_bottom_rects, h)
    bottom_mask = _mask_from_rectangles((h, w), bottom_rects)
    top[top_mask] = top_color
    bottom[bottom_mask] = bottom_color
    grid[:h] = top
    grid[h] = SEP
    grid[h + 1 :] = bottom
    return grid


def _semantic_checks(
    grid: np.ndarray,
    full_top: np.ndarray,
    top_rects: list[tuple[int, int, int, int]],
    mirrored_bottom_rects: list[tuple[int, int, int, int]],
    period_shape: tuple[int, int],
) -> bool:
    rp, cp = period_shape
    h, w = full_top.shape
    top_mask = _mask_from_rectangles((h, w), top_rects)
    mirrored_mask = _mask_from_rectangles((h, w), mirrored_bottom_rects)
    if np.any(top_mask & mirrored_mask):
        return False

    if _residues_touched(top_mask, rp, cp) < max(2, (rp * cp) // 3):
        return False
    if _residues_touched(mirrored_mask, rp, cp) < max(2, (rp * cp) // 3):
        return False
    if not np.any(top_mask) or not np.any(mirrored_mask):
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, full_top)


def _attempt() -> np.ndarray | None:
    h = random.randint(*HALF_HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    rp, cp = random.choice(TILE_SHAPES)
    if rp > h or cp > w:
        return None

    tile = _random_tile((rp, cp))
    if tile is None:
        return None
    row_phase = random.randrange(rp)
    col_phase = random.randrange(cp)
    full_top = _phase_repeat(tile, (h, w), row_phase, col_phase)

    top_count = random.choice((1, 2, 2))
    bottom_count = random.choice((1, 2, 2))
    top_color = random.choice(OCCLUDER_COLORS)
    bottom_color = RED if top_color == YELLOW else YELLOW

    for _ in range(SET_ATTEMPTS):
        top_rects = _random_rectangles(h, w, top_count)
        if top_rects is None:
            continue
        top_mask = _mask_from_rectangles((h, w), top_rects)
        mirrored_bottom_rects = _random_rectangles(h, w, bottom_count, forbidden=top_mask)
        if mirrored_bottom_rects is None:
            continue
        grid = _make_input(full_top, top_rects, mirrored_bottom_rects, top_color, bottom_color)
        if _semantic_checks(grid, full_top, top_rects, mirrored_bottom_rects, (rp, cp)):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid mirrored-complement tiling puzzle")
