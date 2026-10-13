from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MIRROR = 5
WALL = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
GEN_ATTEMPTS = 600


PRIMARY_SPRITES = (
    ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (3, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (3, 1)),
)

ALL_SPRITES = PRIMARY_SPRITES + (
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _normalize_cells(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return sorted({(int(r - r0), int(c - c0)) for r, c in cells})


def _transform_cells(base: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    cells = [(int(r), int(c)) for r, c in base]
    if random.random() < 0.5:
        cells = [(r, -c) for r, c in cells]
    for _ in range(random.randint(0, 3)):
        cells = [(c, -r) for r, c in cells]
    return _normalize_cells(cells)


def _sprite_shape(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _sample_primary_sprite() -> list[tuple[int, int]]:
    for _ in range(40):
        cells = _transform_cells(random.choice(PRIMARY_SPRITES))
        height, width = _sprite_shape(cells)
        if height >= 4 and width <= 3:
            return cells
    raise RuntimeError("failed to sample a primary sprite")


def _sample_sprite() -> list[tuple[int, int]]:
    for _ in range(40):
        cells = _transform_cells(random.choice(ALL_SPRITES))
        height, width = _sprite_shape(cells)
        if height <= 5 and width <= 3:
            return cells
    raise RuntimeError("failed to sample a sprite")


def _random_split(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1)) if parts > 1 else []
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _row_interval(cols: list[int], mirror_col: int) -> tuple[int, int] | None:
    distances = [mirror_col - c for c in cols]
    lo = max(distances) // 2 + 1
    hi = min(distances) - 1
    if lo > hi:
        return None
    return lo, hi


def _find_partial_band(rows: list[int], row_to_cols: dict[int, list[int]], mirror_col: int) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int, int]] = []
    for i in range(len(rows)):
        for j in range(i, len(rows)):
            if i == 0 and j == len(rows) - 1:
                continue
            lo = 1
            hi = mirror_col - 1
            ok = True
            for row in rows[i : j + 1]:
                interval = _row_interval(row_to_cols[row], mirror_col)
                if interval is None:
                    ok = False
                    break
                lo = max(lo, interval[0])
                hi = min(hi, interval[1])
            if ok and lo <= hi:
                length = j - i + 1
                candidates.append((length, rows[i], rows[j], random.randint(lo, hi)))
    if not candidates:
        return None
    max_len = max(length for length, _r0, _r1, _k in candidates)
    long_candidates = [item for item in candidates if item[0] == max_len]
    _length, r0, r1, k = random.choice(long_candidates)
    return r0, r1, k


def _find_any_band(rows: list[int], row_to_cols: dict[int, list[int]], mirror_col: int) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int]] = []
    for i in range(len(rows)):
        for j in range(i, len(rows)):
            lo = 1
            hi = mirror_col - 1
            ok = True
            for row in rows[i : j + 1]:
                interval = _row_interval(row_to_cols[row], mirror_col)
                if interval is None:
                    ok = False
                    break
                lo = max(lo, interval[0])
                hi = min(hi, interval[1])
            if ok and lo <= hi:
                candidates.append((rows[i], rows[j], random.randint(lo, hi)))
    if not candidates:
        return None
    return random.choice(candidates)


def _destination_col(src_col: int, mirror_col: int, wall_col: int | None) -> int:
    target = 2 * mirror_col - src_col
    if wall_col is None or wall_col > target:
        return target
    return 2 * wall_col - target


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect left-side pixels across the grey mirror, rebounding rows that hit a maroon wall.

    Input structure:
    - A full-height grey vertical mirror column splits the grid into left and right halves.
    - The left half contains colored pixel patterns on a plain background.
    - The right half contains only maroon vertical wall segments and background.

    Rule:
    - Keep the original left patterns, the mirror column, and every wall cell unchanged.
    - For each non-background pixel left of the mirror, first place its ordinary mirror image on the right.
    - If that row contains a maroon wall cell between the mirror and the mirrored target, use the nearest such wall cell as a second mirror so the pixel bounces back toward the mirror.

    Key insight:
    - Each row is solved independently: unblocked rows are simple mirror reflections, while blocked rows undergo two reflections, which turns the would-be overshoot past the wall into a rebound distance before the wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    mirror_cols = np.flatnonzero(np.all(g == MIRROR, axis=0))
    assert mirror_cols.size == 1, "expected exactly one full-height grey mirror column"
    mirror_col = int(mirror_cols[0])

    bg = _background_color(g)
    mirror_cells = np.argwhere(g == MIRROR)
    assert np.all(mirror_cells[:, 1] == mirror_col), "grey pixels must belong only to the mirror column"

    right_side = g[:, mirror_col + 1 :]
    assert not np.any((right_side != bg) & (right_side != WALL)), "right side may contain only walls and background"

    out = g.copy()
    has_source = False
    for r in range(h):
        wall_cols = np.flatnonzero(g[r, mirror_col + 1 :] == WALL) + mirror_col + 1
        source_cols = np.flatnonzero((g[r, :mirror_col] != bg) & (g[r, :mirror_col] != MIRROR) & (g[r, :mirror_col] != WALL))
        for c in source_cols:
            has_source = True
            color = int(g[r, c])
            target = 2 * mirror_col - int(c)
            assert mirror_col < target < w, "mirrored target must stay inside the grid"

            blockers = wall_cols[wall_cols <= target]
            if blockers.size:
                wall_col = int(blockers.min())
                dest = 2 * wall_col - target
                assert mirror_col < dest < wall_col, "rebounded pixel must land between the mirror and the wall"
            else:
                dest = target

            assert out[r, dest] in {bg, color}, "reflected pixels must not conflict"
            out[r, dest] = color

    assert has_source, "expected at least one colored source pixel left of the mirror"
    return out


def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.randint(2, 4)
    sprites = [_sample_primary_sprite()]
    sprites.extend(_sample_sprite() for _ in range(n_sprites - 1))
    random.shuffle(sprites)

    heights = [_sprite_shape(cells)[0] for cells in sprites]
    min_height = sum(heights) + (n_sprites - 1) + 2
    if min_height > 24:
        return None

    h = random.randint(max(10, min_height), min(24, min_height + 6))
    extras = _random_split(h - min_height, n_sprites + 1)
    gaps = [1 + extra for extra in extras]

    tops = []
    current = gaps[0]
    for idx, cells in enumerate(sprites):
        tops.append(current)
        current += _sprite_shape(cells)[0]
        if idx < n_sprites - 1:
            current += gaps[idx + 1]

    lefts = [random.randint(1, 4) for _ in range(n_sprites)]
    rightmost = max(left + _sprite_shape(cells)[1] - 1 for left, cells in zip(lefts, sprites))
    if rightmost + 8 > 14:
        return None

    mirror_col = rightmost + random.randint(5, 8)
    if mirror_col > 14:
        return None

    w_min = 2 * mirror_col + 1
    if w_min > 30:
        return None
    w = random.randint(w_min, min(30, w_min + 5))

    grid = np.full((h, w), BG, dtype=int)
    grid[:, mirror_col] = MIRROR

    colors = random.sample(PALETTE, n_sprites)
    row_to_cols: dict[int, list[int]] = {}
    sprite_rows: list[list[int]] = []

    for cells, top, left, color in zip(sprites, tops, lefts, colors):
        rows_here = set()
        for dr, dc in cells:
            r = top + dr
            c = left + dc
            if grid[r, c] != BG:
                return None
            grid[r, c] = color
            row_to_cols.setdefault(r, []).append(c)
            rows_here.add(r)
        sprite_rows.append(sorted(rows_here))

    primary_candidates = [idx for idx, rows in enumerate(sprite_rows) if len(rows) >= 3]
    if not primary_candidates:
        return None
    primary_idx = max(primary_candidates, key=lambda idx: len(sprite_rows[idx]))

    wall_segments: list[tuple[int, int, int]] = []
    primary_band = _find_partial_band(sprite_rows[primary_idx], row_to_cols, mirror_col)
    if primary_band is None:
        return None
    wall_segments.append(primary_band)

    for idx, rows in enumerate(sprite_rows):
        if idx == primary_idx or random.random() < 0.5:
            continue
        band = _find_any_band(rows, row_to_cols, mirror_col)
        if band is not None:
            wall_segments.append(band)

    wall_by_row: dict[int, int] = {}
    for r0, r1, k in wall_segments:
        wall_col = mirror_col + k
        for r in range(r0, r1 + 1):
            if r in wall_by_row and wall_by_row[r] != wall_col:
                return None
            wall_by_row[r] = wall_col
            if grid[r, wall_col] != BG:
                return None
            grid[r, wall_col] = WALL

    source_rows = sorted(row_to_cols)
    bounced_rows = [r for r in source_rows if r in wall_by_row]
    plain_rows = [r for r in source_rows if r not in wall_by_row]
    if not bounced_rows or not plain_rows:
        return None

    changed_rows = 0
    for r in bounced_rows:
        changed = False
        for src_col in row_to_cols[r]:
            plain_dest = 2 * mirror_col - src_col
            bounced_dest = _destination_col(src_col, mirror_col, wall_by_row[r])
            if bounced_dest != plain_dest:
                changed = True
                break
        changed_rows += int(changed)
    if changed_rows == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    right_new = (out != grid) & (out != BG)
    if not np.any(right_new[:, mirror_col + 1 :]):
        return None
    if np.count_nonzero(out[:, :mirror_col] != grid[:, :mirror_col]) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
