from __future__ import annotations

import random

import numpy as np


MAX_SEARCH_PERIOD = 8

GEN_MAX_ATTEMPTS = 500
GEN_TILE_SIDE_RANGE = (2, 5)
GEN_REPEAT_RANGE = (3, 6)
GEN_COLOR_COUNT_RANGE = (2, 5)
GEN_TILE_ATTEMPTS = 120
GEN_PATCH_DENSITY_MIN = 0.35
GEN_PATCH_FILL_RANGE = (0.58, 0.92)
GEN_TILE_SHIFT_DIFF_MIN = 0.40
GEN_VISIBLE_MISMATCH_MIN = 6


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 0
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _count_components(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = map(int, mask.shape)
    components = 0
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            components += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return components


def _dilate4(mask: np.ndarray) -> np.ndarray:
    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


def _axis_error(grid: np.ndarray, axis: int, period: int) -> float:
    if axis == 0:
        if period >= grid.shape[0]:
            return 1.0
        return float(np.mean(grid[period:, :] != grid[:-period, :]))
    if period >= grid.shape[1]:
        return 1.0
    return float(np.mean(grid[:, period:] != grid[:, :-period]))


def _reconstruct_wallpaper(grid: np.ndarray, tile_h: int, tile_w: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, grid.shape)
    palette_size = int(grid.max()) + 1
    tile = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            values = grid[r:height:tile_h, c:width:tile_w].ravel()
            counts = np.bincount(values, minlength=palette_size)
            tile[r, c] = int(np.argmax(counts))
    expected = tile[np.arange(height) % tile_h][:, np.arange(width) % tile_w]
    return expected, grid != expected


def _infer_clean_wallpaper(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, grid.shape)
    best_score = None
    best_expected = None
    best_diff = None

    max_h = min(MAX_SEARCH_PERIOD, height - 1)
    max_w = min(MAX_SEARCH_PERIOD, width - 1)
    for tile_h in range(1, max_h + 1):
        row_error = _axis_error(grid, axis=0, period=tile_h)
        for tile_w in range(1, max_w + 1):
            col_error = _axis_error(grid, axis=1, period=tile_w)
            expected, diff = _reconstruct_wallpaper(grid, tile_h, tile_w)
            mismatches = int(diff.sum())
            if mismatches == 0:
                continue
            box = _bbox(diff)
            bbox_area = _bbox_area(box)
            components = _count_components(diff)
            grown_components = _count_components(_dilate4(diff))
            score = (
                (grown_components - 1) * 10000
                + (components - 1) * 80
                + abs(bbox_area - mismatches) * 6
                + mismatches * 3
                + tile_h * tile_w * 8
                + int(round((row_error + col_error) * 20))
            )
            if best_score is None or score < best_score:
                best_score = score
                best_expected = expected
                best_diff = diff

    assert best_expected is not None and best_diff is not None, "failed to infer a non-trivial wallpaper period"
    return best_expected, best_diff


def _crop_actual_pixels(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected at least one visible mismatch"
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    local_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    out[local_mask] = window[local_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the clean periodic wallpaper and reveal the phase-shifted patch by cropping its visible disagreements.

    Input structure:
    - The whole grid is formed by repeating one small rectangular tile in both directions.
    - One connected region has been overwritten by the same wallpaper under a different translation phase.
    - Some overwritten cells may coincidentally still match the original wallpaper.

    Rule:
    - Try small row/column periods, and for each tile position take the majority color seen across the whole grid.
    - Use the best-scoring period pair to reconstruct the clean wallpaper that most cells agree on.
    - Compare the observed grid with that reconstruction, keep the observed colors only on mismatching cells, and crop that mask tightly with black elsewhere.

    Key insight:
    - The patch is not made of foreign colors at all; the only signal is that one connected area follows the same wallpaper with the wrong global phase, so the true pattern must be recovered first by majority consensus.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 2, "grid must be at least 2x2"

    expected, diff = _infer_clean_wallpaper(g)
    visible_patch = g != expected
    return _crop_actual_pixels(g, visible_patch)


def _tile_has_smaller_period(tile: np.ndarray) -> bool:
    tile_h, tile_w = map(int, tile.shape)
    for sub_h in range(1, tile_h):
        if np.array_equal(tile, tile[np.arange(tile_h) % sub_h, :]):
            return True
    for sub_w in range(1, tile_w):
        if np.array_equal(tile, tile[:, np.arange(tile_w) % sub_w]):
            return True
    return False


def _phase_shift_candidates(tile: np.ndarray) -> list[tuple[int, int]]:
    tile_h, tile_w = map(int, tile.shape)
    rr = np.arange(tile_h)[:, None]
    cc = np.arange(tile_w)[None, :]
    candidates = []
    for dr in range(tile_h):
        for dc in range(tile_w):
            if dr == 0 and dc == 0:
                continue
            shifted = tile[(rr + dr) % tile_h, (cc + dc) % tile_w]
            if float(np.mean(shifted != tile)) >= GEN_TILE_SHIFT_DIFF_MIN:
                candidates.append((dr, dc))
    return candidates


def _make_wallpaper(tile: np.ndarray, repeats_h: int, repeats_w: int, shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    tile_h, tile_w = map(int, tile.shape)
    height = tile_h * repeats_h
    width = tile_w * repeats_w
    dr, dc = shift
    rows = (np.arange(height) + dr) % tile_h
    cols = (np.arange(width) + dc) % tile_w
    return tile[rows[:, None], cols[None, :]]


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    area = max(1, min(area, height * width))
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while int(mask.sum()) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        r, c = cell
        if mask[r, c]:
            continue
        mask[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _sample_tile() -> tuple[np.ndarray, list[tuple[int, int]]]:
    for _ in range(GEN_TILE_ATTEMPTS):
        tile_h = random.randint(*GEN_TILE_SIDE_RANGE)
        tile_w = random.randint(*GEN_TILE_SIDE_RANGE)
        color_count = random.randint(*GEN_COLOR_COUNT_RANGE)
        palette = random.sample(range(1, 10), color_count)
        tile = np.array(random.choices(palette, k=tile_h * tile_w), dtype=int).reshape(tile_h, tile_w)
        if len(np.unique(tile)) < 2:
            continue
        if _tile_has_smaller_period(tile):
            continue
        shifts = _phase_shift_candidates(tile)
        if shifts:
            return tile, shifts
    raise RuntimeError("failed to sample a sufficiently distinctive tile")


def _expected_output(grid: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return _crop_actual_pixels(grid, grid != clean)


def _generate_attempt() -> np.ndarray | None:
    tile, shifts = _sample_tile()
    tile_h, tile_w = map(int, tile.shape)

    min_repeats_h = max(GEN_REPEAT_RANGE[0], (10 + tile_h - 1) // tile_h)
    min_repeats_w = max(GEN_REPEAT_RANGE[0], (10 + tile_w - 1) // tile_w)
    max_repeats_h = min(GEN_REPEAT_RANGE[1], 30 // tile_h)
    max_repeats_w = min(GEN_REPEAT_RANGE[1], 30 // tile_w)
    if min_repeats_h > max_repeats_h or min_repeats_w > max_repeats_w:
        return None

    repeats_h = random.randint(min_repeats_h, max_repeats_h)
    repeats_w = random.randint(min_repeats_w, max_repeats_w)
    clean = _make_wallpaper(tile, repeats_h, repeats_w)
    height, width = map(int, clean.shape)

    patch_h_min = max(tile_h + 1, 4)
    patch_w_min = max(tile_w + 1, 4)
    patch_h_max = min(height - 2, max(patch_h_min, (2 * height) // 3))
    patch_w_max = min(width - 2, max(patch_w_min, (2 * width) // 3))
    if patch_h_min > patch_h_max or patch_w_min > patch_w_max:
        return None

    patch_h = random.randint(patch_h_min, patch_h_max)
    patch_w = random.randint(patch_w_min, patch_w_max)
    top = random.randint(1, height - patch_h - 1)
    left = random.randint(1, width - patch_w - 1)

    if random.random() < 0.45:
        local_mask = np.ones((patch_h, patch_w), dtype=bool)
    else:
        patch_area = patch_h * patch_w
        lo = max(5, int(round(patch_area * GEN_PATCH_FILL_RANGE[0])))
        hi = max(lo, int(round(patch_area * GEN_PATCH_FILL_RANGE[1])))
        local_mask = _random_connected_mask(patch_h, patch_w, random.randint(lo, hi))

    full_mask = np.zeros_like(clean, dtype=bool)
    full_mask[top : top + patch_h, left : left + patch_w] = local_mask

    shifted = _make_wallpaper(tile, repeats_h, repeats_w, shift=random.choice(shifts))
    observed = clean.copy()
    observed[full_mask] = shifted[full_mask]

    diff = observed != clean
    mismatch_count = int(diff.sum())
    if mismatch_count < GEN_VISIBLE_MISMATCH_MIN:
        return None

    box = _bbox(diff)
    bbox_area = _bbox_area(box)
    if bbox_area == 0 or mismatch_count / float(bbox_area) < GEN_PATCH_DENSITY_MIN:
        return None

    expected = _expected_output(observed, clean)
    try:
        predicted = main(observed)
    except AssertionError:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return observed


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wallpaper-patch input")
