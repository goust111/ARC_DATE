from __future__ import annotations

import numpy as np


AZURE = 8
GREY = 5
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    rel = coords - np.array([[r0, c0]])
    patch[rel[:, 0], rel[:, 1]] = grid[coords[:, 0], coords[:, 1]]
    return patch, r0, c0


def _half_counts(mask: np.ndarray) -> tuple[int, int]:
    split = mask.shape[0] // 2
    top = int(mask[:split].sum())
    bottom = int(mask[split:].sum())
    return top, bottom


def _drop_row(obstacle_mask: np.ndarray, occ: np.ndarray, start_r0: int, c0: int) -> int:
    rel_rows, rel_cols = np.where(occ)
    drop = 0
    while True:
        next_rows = start_r0 + rel_rows + drop + 1
        next_cols = c0 + rel_cols
        if np.any(next_rows >= obstacle_mask.shape[0]):
            break
        if np.any(obstacle_mask[next_rows, next_cols]):
            break
        drop += 1
    return start_r0 + drop


def _transformed_objects(grid: np.ndarray, bg: int = AZURE, obstacle: int = GREY) -> list[tuple[np.ndarray, int, int]]:
    fg_mask = (grid != int(bg)) & (grid != int(obstacle))
    objects = []
    for coords in _components(fg_mask):
        patch, r0, c0 = _extract_patch(grid, coords, bg)
        occ = patch != int(bg)
        top, bottom = _half_counts(occ)
        if top > bottom:
            patch = np.flipud(patch)
            occ = np.flipud(occ)
        final_r0 = _drop_row(grid == int(obstacle), occ, r0, c0)
        objects.append((patch, final_r0, c0))
    objects.sort(key=lambda item: (item[1], item[2]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each foreground object so its heavier vertical half points downward.

    The background is azure, grey cells are fixed ground or obstacles, and each
    4-connected foreground object is vertically asymmetric inside its bbox. If
    the upper half contains more pixels than the lower half, vertically flip the
    object first, then move it straight down until the next step would hit grey
    or leave the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = np.full_like(g, AZURE)
    out[g == GREY] = GREY

    for patch, r0, c0 in _transformed_objects(g):
        occ = patch != AZURE
        h, w = patch.shape
        target = out[r0 : r0 + h, c0 : c0 + w]
        assert not np.any(occ & (target != AZURE)), "objects must not overlap after falling"
        target[occ] = patch[occ]

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

GRID_H_RANGE = (12, 19)
OBJECT_COUNT_RANGE = (2, 5)
HEIGHT_CHOICES = (4, 6)
WIDTH_RANGE = (2, 5)
MIN_GAP = 1
EXTRA_WIDTH_MAX = 6
MIN_DROP = 2
MAX_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _grow(mask: np.ndarray, rows: range, amount: int) -> bool:
    h, w = mask.shape
    allowed = set(rows)
    for _ in range(amount):
        frontier = []
        for r in allowed:
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return False
        rr, cc = frontier[int(_RNG.integers(len(frontier)))]
        mask[rr, cc] = True
    return True


def _make_shape(heavy_top: bool) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        h = int(_RNG.choice(HEIGHT_CHOICES))
        w = int(_RNG.integers(WIDTH_RANGE[0], WIDTH_RANGE[1]))
        mask = np.zeros((h, w), dtype=bool)

        col = int(_RNG.integers(w))
        for r in range(h):
            if r > 0:
                col = int(np.clip(col + int(_RNG.integers(-1, 2)), 0, w - 1))
            mask[r, col] = True

        split = h // 2
        top_rows = range(0, split)
        bottom_rows = range(split, h)
        heavy_rows = top_rows if heavy_top else bottom_rows
        light_rows = bottom_rows if heavy_top else top_rows

        heavy_capacity = w * len(heavy_rows) - len(heavy_rows)
        light_capacity = w * len(light_rows) - len(light_rows)
        heavy_add = int(_RNG.integers(1, min(heavy_capacity, 4) + 1))
        light_add_max = min(light_capacity, heavy_add - 1)
        light_add = int(_RNG.integers(light_add_max + 1)) if light_add_max > 0 else 0

        if not _grow(mask, light_rows, light_add):
            continue
        if not _grow(mask, heavy_rows, heavy_add):
            continue

        if _RNG.random() < 0.5:
            mask = np.fliplr(mask)
        mask = _trim_mask(mask)
        if mask.shape[1] < 2:
            continue

        top, bottom = _half_counts(mask)
        if heavy_top and top > bottom:
            return mask
        if not heavy_top and bottom > top:
            return mask
    raise RuntimeError("failed to build an asymmetric object")


def _balanced_flags(count: int) -> list[bool]:
    num_true = count // 2
    if count % 2 == 1 and _RNG.random() < 0.5:
        num_true += 1
    flags = [True] * num_true + [False] * (count - num_true)
    _RNG.shuffle(flags)
    return flags


def _spread_total(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[int(_RNG.integers(parts))] += 1
    return out


def _gen_attempt() -> np.ndarray | None:
    count = int(_RNG.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1]))
    heavy_top_flags = _balanced_flags(count)
    support_flags = _balanced_flags(count)
    if all(support_flags) or not any(support_flags):
        support_flags[0] = True
        support_flags[-1] = False

    masks = [_make_shape(flag) for flag in heavy_top_flags]
    heights = [int(mask.shape[0]) for mask in masks]
    widths = [int(mask.shape[1]) for mask in masks]

    min_h = max(max(heights) + 6, GRID_H_RANGE[0])
    if min_h >= GRID_H_RANGE[1]:
        return None
    grid_h = int(_RNG.integers(min_h, GRID_H_RANGE[1]))

    min_w = sum(widths) + MIN_GAP * (count + 1)
    if min_w > 24:
        return None
    grid_w_lo = max(12, min_w)
    grid_w_hi = min(24, grid_w_lo + EXTRA_WIDTH_MAX)
    grid_w = int(_RNG.integers(grid_w_lo, grid_w_hi + 1))

    extras = _spread_total(grid_w - min_w, count + 1)
    gaps = [MIN_GAP + extra for extra in extras]
    cols = []
    cursor = gaps[0]
    for width, gap in zip(widths, gaps[1:]):
        cols.append(cursor)
        cursor += width + gap

    colors = _RNG.choice(FOREGROUND_COLORS, size=count, replace=False).tolist()
    grid = np.full((grid_h, grid_w), AZURE, dtype=int)
    grid[-1, :] = GREY

    landed_on_platform = 0
    for idx, (mask, color, c0, use_platform) in enumerate(zip(masks, colors, cols, support_flags)):
        h, w = mask.shape
        if use_platform:
            min_top = h + MIN_DROP + 2
            if min_top > grid_h - 3:
                return None
            support_top = int(_RNG.integers(min_top, grid_h - 2))
            grid[support_top : grid_h - 1, c0 : c0 + w] = GREY
            landed_on_platform += 1
        else:
            support_top = grid_h - 1

        final_r0 = support_top - h
        start_max = final_r0 - MIN_DROP
        if start_max < 1:
            return None
        start_r0 = int(_RNG.integers(1, start_max + 1))

        patch = np.full(mask.shape, AZURE, dtype=int)
        patch[mask] = int(color)
        region = grid[start_r0 : start_r0 + h, c0 : c0 + w]
        if np.any((patch != AZURE) & (region != AZURE)):
            return None
        region[patch != AZURE] = patch[patch != AZURE]

    if landed_on_platform == 0 or landed_on_platform == count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    object_mask = (grid != AZURE) & (grid != GREY)
    comps = _components(object_mask)
    if len(comps) != count:
        return None
    flips = 0
    for coords in comps:
        patch, _, _ = _extract_patch(grid, coords, AZURE)
        top, bottom = _half_counts(patch != AZURE)
        if top == bottom:
            return None
        flips += int(top > bottom)
    if flips == 0 or flips == len(comps):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-flip sample")
