from __future__ import annotations

import random

import numpy as np


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _filled_rectangle(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return r0, c0, r1, c1


def _pattern_from_mode(grid: np.ndarray, ph: int, pw: int) -> np.ndarray | None:
    pattern = np.empty((ph, pw), dtype=int)
    for rr in range(ph):
        for cc in range(pw):
            values = grid[rr::ph, cc::pw].ravel()
            counts: dict[int, int] = {}
            for value in values:
                color = int(value)
                counts[color] = counts.get(color, 0) + 1
            best = max(counts.values())
            winners = [color for color, count in counts.items() if count == best]
            if len(winners) != 1:
                return None
            pattern[rr, cc] = winners[0]
    return pattern


def _predict(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    row_idx = np.arange(h) % pattern.shape[0]
    col_idx = np.arange(w) % pattern.shape[1]
    return pattern[row_idx[:, None], col_idx[None, :]]


def _valid_reconstruction(grid: np.ndarray, pattern: np.ndarray) -> np.ndarray | None:
    pred = _predict(pattern, grid.shape)
    mismatch = pred != grid
    if not mismatch.any():
        return pred

    mismatch_colors: set[int] = set()
    for coords in _components(mismatch):
        bbox = _filled_rectangle(coords)
        if bbox is None:
            return None
        r0, c0, r1, c1 = bbox
        rect = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(set(map(int, rect.ravel()))) != 1:
            return None
        if not mismatch[r0 : r1 + 1, c0 : c1 + 1].all():
            return None
        mismatch_colors.add(int(rect[0, 0]))

    for color in mismatch_colors:
        if np.any(pattern == color):
            return None
        if np.any((grid == color) & (~mismatch)):
            return None
    return pred


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a periodic background pattern hidden under one or more solid rectangles.

    Input structure:
    - The full grid is tiled by a small repeating pattern.
    - One or more monochrome filled rectangles overwrite parts of that tiling.
    - Obstruction colors do not belong to the underlying tile palette.

    Rule:
    - Infer the repeating tile period and tile contents from the visible cells.
    - Replace every obstructing rectangle cell with the pattern color that belongs there.
    - Leave all visible non-obstructed cells unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 2

    candidates: list[tuple[int, int]] = []
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if h // ph < 2 or w // pw < 2:
                continue
            candidates.append((ph, pw))

    candidates.sort(key=lambda item: (item[0] * item[1], item[0] + item[1], item[0], item[1]))
    for ph, pw in candidates:
        pattern = _pattern_from_mode(g, ph, pw)
        if pattern is None:
            continue
        restored = _valid_reconstruction(g, pattern)
        if restored is not None:
            return restored

    raise AssertionError("could not infer a valid periodic pattern with rectangular occluders")


def _axis_period(tile: np.ndarray, axis: int) -> int:
    size = tile.shape[axis]
    for step in range(1, size + 1):
        if np.array_equal(tile, np.roll(tile, -step, axis=axis)):
            return step
    raise AssertionError("axis period search failed")


def _make_tile(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    colors = rng.sample(list(range(10)), rng.randint(3, 6))
    tile_color_count = rng.randint(2, min(4, len(colors)))
    tile_colors = colors[:tile_color_count]

    for _ in range(200):
        tile = np.array([[rng.choice(tile_colors) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _axis_period(tile, 0) != h or _axis_period(tile, 1) != w:
            continue
        return tile
    return None


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _visible_majority_ok(mask: np.ndarray, ph: int, pw: int) -> bool:
    for rr in range(ph):
        for cc in range(pw):
            total = int(mask[rr::ph, cc::pw].size)
            hidden = int(mask[rr::ph, cc::pw].sum())
            if total - hidden <= hidden:
                return False
            if total - hidden < 2:
                return False
    return True


def _sample_rectangles(
    shape: tuple[int, int],
    pattern_shape: tuple[int, int],
    occluder_colors: list[int],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int, int]] | None:
    h, w = shape
    ph, pw = pattern_shape
    max_h = max(2, h // 2)
    max_w = max(2, w // 2)

    for _ in range(300):
        rects: list[tuple[int, int, int, int, int]] = []
        hidden = np.zeros((h, w), dtype=bool)
        for _rect_idx in range(count):
            placed = False
            for _ in range(120):
                rh = rng.randint(1, max_h)
                rw = rng.randint(1, max_w)
                if rh * rw < 2:
                    continue
                r0 = rng.randint(0, h - rh)
                c0 = rng.randint(0, w - rw)
                bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if any(_bbox_intersects(bbox, other[:4], gap=1) for other in rects):
                    continue
                color = rng.choice(occluder_colors)
                rects.append((bbox[0], bbox[1], bbox[2], bbox[3], color))
                hidden[r0 : r0 + rh, c0 : c0 + rw] = True
                placed = True
                break
            if not placed:
                break
        if len(rects) != count:
            continue

        cover = int(hidden.sum())
        area = h * w
        if cover < max(4, area // 12):
            continue
        if cover > area // 3:
            continue
        if not _visible_majority_ok(hidden, ph, pw):
            continue
        return rects
    return None


def _generate_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    tile = _make_tile(rng)
    if tile is None:
        return None

    ph, pw = tile.shape
    rep_h = rng.randint(3, 6)
    rep_w = rng.randint(3, 6)
    base = np.tile(tile, (rep_h, rep_w))
    h, w = base.shape

    tile_colors = sorted(set(map(int, tile.ravel())))
    free_colors = [color for color in range(10) if color not in tile_colors]
    occ_color_count = rng.randint(1, min(2, len(free_colors)))
    occ_colors = rng.sample(free_colors, occ_color_count)
    rect_count = rng.randint(1, 3)
    rects = _sample_rectangles((h, w), (ph, pw), occ_colors, rect_count, rng)
    if rects is None:
        return None

    inp = base.copy()
    for r0, c0, r1, c1, color in rects:
        inp[r0 : r1 + 1, c0 : c1 + 1] = color

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, base):
        return None
    return inp, base


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        pair = _generate_pair(rng)
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid periodic-occlusion puzzle")
