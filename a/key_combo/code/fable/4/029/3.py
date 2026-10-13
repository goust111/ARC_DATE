from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER = 9
WALL_COLORS = tuple(range(1, 8))
MAX_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_marker_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        if len(coords) != 2:
            continue
        (r1, c1), (r2, c2) = coords
        same_row = int(r1) == int(r2)
        same_col = int(c1) == int(c2)
        if same_row == same_col:
            continue
        if same_row and ((c1 < w // 2 and c2 < w // 2) or (c1 >= w // 2 and c2 >= w // 2)):
            candidates.append(color)
        if same_col and ((r1 < h // 2 and r2 < h // 2) or (r1 >= h // 2 and r2 >= h // 2)):
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one aligned two-pixel marker color"
    return candidates[0]


def _crop_bounds(grid: np.ndarray, marker: int) -> tuple[slice, slice]:
    h, w = grid.shape
    coords = np.argwhere(grid == marker)
    assert len(coords) == 2
    (r1, c1), (r2, c2) = coords
    if int(r1) == int(r2):
        keep_left = int(c1) < w // 2
        return slice(None), slice(0, w // 2) if keep_left else slice(w // 2, w)
    keep_top = int(r1) < h // 2
    return slice(0, h // 2) if keep_top else slice(h // 2, h), slice(None)


def _infer_visible_period(
    grid: np.ndarray, bg: int, ignored: set[int]
) -> tuple[int, int, dict[tuple[int, int], int]]:
    h, w = grid.shape
    visible = [
        (r, c, int(grid[r, c]))
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) != bg and int(grid[r, c]) not in ignored
    ]
    assert visible, "need some visible non-background wallpaper evidence"
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            classes: dict[tuple[int, int], int] = {}
            ok = True
            for r, c, color in visible:
                key = (r % pr, c % pc)
                prev = classes.get(key)
                if prev is not None and prev != color:
                    ok = False
                    break
                classes[key] = color
            if ok:
                return pr, pc, classes
    raise AssertionError("some period should always fit")


def _fundamental_period_full(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            classes: dict[tuple[int, int], int] = {}
            ok = True
            for r in range(h):
                for c in range(w):
                    key = (r % pr, c % pc)
                    color = int(tile[r, c])
                    prev = classes.get(key)
                    if prev is not None and prev != color:
                        ok = False
                        break
                    classes[key] = color
                if not ok:
                    break
            if ok:
                return pr, pc
    raise AssertionError("full tile must have a period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the marker-selected half and restore its erased wallpaper cells from the true global period.

    Input structure:
    - The full grid shows a periodic wallpaper whose background color is also a legal wallpaper color.
    - Some non-background wallpaper cells have been erased to the background color.
    - A unique marker color appears exactly twice; the two marker pixels share one row or one column and lie in the same half.

    Rule:
    - Use the whole input, not just the kept half, and ignore both the background cells and the two marker pixels.
    - Find the smallest row/column period whose residue classes consistently explain every visible non-background wallpaper pixel.
    - Crop the half selected by the markers, then repaint every residue class in that half that has non-background evidence anywhere in the full grid; residue classes with no evidence stay background.

    Key insight:
    - The kept half alone supports a smaller fake period, so the discarded half is needed both to reject that fake period and to tell which background cells in the kept half are actually erased wallpaper.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0 or w % 2 == 0

    bg = _get_bg(g)
    marker = _find_marker_color(g, bg)
    pr, pc, classes = _infer_visible_period(g, bg, {marker})
    row_slice, col_slice = _crop_bounds(g, marker)

    out = g[row_slice, col_slice].copy()
    r0 = 0 if row_slice.start is None else int(row_slice.start)
    c0 = 0 if col_slice.start is None else int(col_slice.start)
    for rr in range(out.shape[0]):
        for cc in range(out.shape[1]):
            color = classes.get(((r0 + rr) % pr, (c0 + cc) % pc))
            out[rr, cc] = bg if color is None else color
    return out


def _sample_base_tile(pr: int, fake_pc: int) -> np.ndarray | None:
    colors = [BG, *WALL_COLORS]
    probs = [0.25, 0.16, 0.15, 0.14, 0.11, 0.10, 0.09, 0.00]
    for _ in range(300):
        tile = np.random.choice(colors, size=(pr, fake_pc), p=probs)
        if np.count_nonzero(tile != BG) < max(2, pr):
            continue
        if _fundamental_period_full(tile) == (pr, fake_pc):
            return tile
    return None


def _generate_vertical_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    pr = random.randint(1, 3)
    fake_pc = random.randint(1, 3)
    base = _sample_base_tile(pr, fake_pc)
    if base is None:
        return None

    true_tile = np.concatenate([base, base.copy()], axis=1)
    choices = [(r, c) for r in range(pr) for c in range(fake_pc)]
    random.shuffle(choices)
    change_count = random.randint(1, max(1, min(len(choices), pr + 1)))
    changed: list[tuple[int, int]] = []
    for r, c in choices[:change_count]:
        old = int(true_tile[r, c + fake_pc])
        new_color = random.choice([color for color in WALL_COLORS if color != old])
        true_tile[r, c + fake_pc] = new_color
        changed.append((r, c + fake_pc))
    true_pc = 2 * fake_pc
    if _fundamental_period_full(true_tile) != (pr, true_pc):
        return None

    row_reps = random.randint(max(3, (8 + pr - 1) // pr), min(8, 30 // pr))
    max_half_reps = 30 // (2 * true_pc)
    if max_half_reps < 2:
        return None
    half_reps = random.randint(2, max_half_reps)

    h = pr * row_reps
    w = 2 * half_reps * true_pc
    wallpaper = np.tile(true_tile, (row_reps, 2 * half_reps))
    keep_left = random.choice([True, False])
    grid = wallpaper.copy()

    kept_cols = range(0, w // 2) if keep_left else range(w // 2, w)
    erased: list[tuple[int, int]] = []
    for rr, cc in changed:
        for r in range(h):
            if r % pr != rr:
                continue
            for c in kept_cols:
                if c % true_pc == cc and int(wallpaper[r, c]) != BG:
                    grid[r, c] = BG
                    erased.append((r, c))
    if len(erased) < 2:
        return None

    row_to_cols: dict[int, list[int]] = {}
    for r, c in erased:
        row_to_cols.setdefault(r, []).append(c)
    marker_rows = [r for r, cols in row_to_cols.items() if len(cols) >= 2]
    if not marker_rows:
        return None
    marker_row = random.choice(marker_rows)
    marker_cols = random.sample(row_to_cols[marker_row], 2)
    for c in marker_cols:
        grid[marker_row, c] = MARKER

    if _get_bg(grid) != BG:
        return None
    try:
        found_marker = _find_marker_color(grid, BG)
    except AssertionError:
        return None
    if found_marker != MARKER:
        return None

    full_pr, full_pc, _ = _infer_visible_period(grid, BG, {MARKER})
    crop = grid[:, : w // 2] if keep_left else grid[:, w // 2 :]
    _crop_pr, crop_pc, _ = _infer_visible_period(crop, BG, {MARKER})
    if (full_pr, full_pc) != (pr, true_pc):
        return None
    if crop_pc >= full_pc:
        return None

    expected = wallpaper[:, : w // 2] if keep_left else wallpaper[:, w // 2 :]
    if np.array_equal(crop, expected):
        return None
    if np.count_nonzero(crop != expected) < 3:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        built = _generate_vertical_attempt()
        if built is None:
            continue
        grid, expected = built
        if random.choice([True, False]):
            return grid
        t_grid = grid.T
        t_expected = expected.T
        if np.array_equal(main(t_grid), t_expected):
            return t_grid
    raise RuntimeError("failed to generate a valid fake-period wallpaper instance")
