from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]

DOWN_RIGHT = "down_right"
DOWN_LEFT = "down_left"
PATTERN_PERIODS = (3, 2)

# Two-color, width-1 diagonal stripes collapse to an ambiguous checkerboard, so the
# generator uses a three-color dark stripe cycle while keeping the same "read the
# background direction, then grow by local seed counts" mechanism.
GROUP_TEMPLATES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (0, 2)),
    ((0, 0), (2, 0)),
    ((0, 0), (1, 2)),
    ((0, 0), (2, 1)),
    ((0, 0), (0, 2), (2, 1)),
    ((0, 0), (2, 0), (1, 2)),
    ((0, 0), (2, -1), (2, 1)),
    ((0, 0), (0, 2), (2, 1), (2, -1)),
)

GRID_MIN = 11
GRID_MAX = 18
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160


def _diag_key(r: int, c: int, direction: str) -> int:
    return int(r - c) if direction == DOWN_RIGHT else int(r + c)


def _key_bounds(shape: tuple[int, int], direction: str) -> tuple[int, int]:
    h, w = shape
    if direction == DOWN_RIGHT:
        return -(w - 1), h - 1
    return 0, h + w - 2


def _fit_background_pattern(g: np.ndarray) -> dict[str, object]:
    h, w = g.shape
    best: dict[str, object] | None = None

    for direction in (DOWN_RIGHT, DOWN_LEFT):
        min_key, _ = _key_bounds(g.shape, direction)
        for period in PATTERN_PERIODS:
            residue_counts = [Counter() for _ in range(period)]
            for r in range(h):
                for c in range(w):
                    residue = (_diag_key(r, c, direction) - min_key) % period
                    residue_counts[residue][int(g[r, c])] += 1

            colors = [counts.most_common(1)[0][0] for counts in residue_counts]
            distinct = len(set(colors))
            if distinct != period:
                continue

            mismatch = 0
            for r in range(h):
                for c in range(w):
                    residue = (_diag_key(r, c, direction) - min_key) % period
                    mismatch += int(int(g[r, c]) != int(colors[residue]))

            candidate = {
                "direction": direction,
                "period": period,
                "colors": tuple(int(color) for color in colors),
                "min_key": int(min_key),
                "mismatch": int(mismatch),
                "distinct": int(distinct),
            }
            score = (int(mismatch), 0 if period == 3 else 1, -int(distinct))
            if best is None or score < best["score"]:
                candidate["score"] = score
                best = candidate

    if best is None:
        raise AssertionError("failed to fit a diagonal stripe background")
    return best


def _background_color_at(pattern: dict[str, object], r: int, c: int) -> int:
    direction = str(pattern["direction"])
    period = int(pattern["period"])
    min_key = int(pattern["min_key"])
    colors = tuple(int(color) for color in pattern["colors"])
    residue = (_diag_key(r, c, direction) - min_key) % period
    return int(colors[residue])


def _extract_seeds(g: np.ndarray, pattern: dict[str, object]) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            expected = _background_color_at(pattern, r, c)
            color = int(g[r, c])
            if color != expected:
                seeds.append((int(r), int(c), color))
    return seeds


def _neighbor_counts(seeds: list[tuple[int, int, int]]) -> list[int]:
    counts: list[int] = []
    for i, (r0, c0, _color) in enumerate(seeds):
        count = 0
        for j, (r1, c1, _other_color) in enumerate(seeds):
            if i == j:
                continue
            if abs(int(r0 - r1)) <= 2 and abs(int(c0 - c1)) <= 2:
                count += 1
        counts.append(int(count))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed along the diagonal direction encoded by the background stripes.

    The seed's extra growth length equals the count of other seeds in its 5x5
    neighborhood. The generator uses a three-color diagonal texture because a
    strict two-color width-1 diagonal texture is direction-ambiguous.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    pattern = _fit_background_pattern(g)
    seeds = _extract_seeds(g, pattern)
    counts = _neighbor_counts(seeds)
    direction = str(pattern["direction"])
    dr, dc = (1, 1) if direction == DOWN_RIGHT else (1, -1)

    out = g.copy()
    items = sorted(zip(seeds, counts), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    for (r, c, color), length in items:
        for step in range(1, int(length) + 1):
            rr = int(r + dr * step)
            cc = int(c + dc * step)
            if not (0 <= rr < h and 0 <= cc < w):
                break
            out[rr, cc] = int(color)

    return out


def _make_background(h: int, w: int, direction: str, bg_colors: tuple[int, int, int]) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if direction == DOWN_RIGHT:
                index = (r - c) % len(bg_colors)
            else:
                index = (r + c) % len(bg_colors)
            grid[r, c] = int(bg_colors[index])
    return grid


def _transform_offsets(offsets: tuple[Coord, ...]) -> tuple[Coord, ...]:
    pts = list(offsets)
    if random.choice((False, True)):
        pts = [(dc, dr) for dr, dc in pts]
    if random.choice((False, True)):
        pts = [(-dr, dc) for dr, dc in pts]
    if random.choice((False, True)):
        pts = [(dr, -dc) for dr, dc in pts]
    return tuple((int(dr), int(dc)) for dr, dc in pts)


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(int(a[0] - b[0])), abs(int(a[1] - b[1])))


def _path_key(pos: Coord, direction: str) -> int:
    return _diag_key(int(pos[0]), int(pos[1]), direction)


def _valid_local_group(coords: list[Coord], direction: str) -> bool:
    if len(set(coords)) != len(coords):
        return False
    keys = [_path_key(coord, direction) for coord in coords]
    if len(set(keys)) != len(keys):
        return False
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            if i >= j:
                continue
            if _chebyshev(a, b) <= 1:
                return False
    return True


def _place_group(
    h: int,
    w: int,
    direction: str,
    existing: list[Coord],
    size: int,
    boundary_bias: bool,
) -> list[Coord] | None:
    templates = [template for template in GROUP_TEMPLATES if len(template) == size]
    if not templates:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        offsets = _transform_offsets(random.choice(templates))
        if offsets[0] != (0, 0):
            offsets = tuple((int(dr - offsets[0][0]), int(dc - offsets[0][1])) for dr, dc in offsets)
        drs = [dr for dr, _dc in offsets]
        dcs = [dc for _dr, dc in offsets]
        r_lo, r_hi = -min(drs), h - 1 - max(drs)
        c_lo, c_hi = -min(dcs), w - 1 - max(dcs)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        if boundary_bias:
            row_start = max(r_lo, r_hi - 3)
            anchor_r = random.randint(row_start, r_hi)
            if direction == DOWN_RIGHT:
                if random.choice((False, True)):
                    col_start = max(c_lo, c_hi - 3)
                    anchor_c = random.randint(col_start, c_hi)
                else:
                    anchor_c = random.randint(c_lo, c_hi)
            else:
                if random.choice((False, True)):
                    col_stop = min(c_hi, c_lo + 3)
                    anchor_c = random.randint(c_lo, col_stop)
                else:
                    anchor_c = random.randint(c_lo, c_hi)
        else:
            anchor_r = random.randint(r_lo, r_hi)
            anchor_c = random.randint(c_lo, c_hi)

        coords = [(int(anchor_r + dr), int(anchor_c + dc)) for dr, dc in offsets]
        if not _valid_local_group(coords, direction):
            continue

        existing_keys = {_path_key(coord, direction) for coord in existing}
        if any(_path_key(coord, direction) in existing_keys for coord in coords):
            continue
        if any(_chebyshev(coord, other) <= 4 for coord in coords for other in existing):
            continue
        return coords
    return None


def _place_isolated(h: int, w: int, direction: str, existing: list[Coord]) -> Coord | None:
    existing_keys = {_path_key(coord, direction) for coord in existing}
    for _ in range(PLACEMENT_ATTEMPTS):
        coord = (random.randint(0, h - 1), random.randint(0, w - 1))
        if _path_key(coord, direction) in existing_keys:
            continue
        if any(_chebyshev(coord, other) <= 4 for other in existing):
            continue
        return coord
    return None


def _lengths_for_positions(coords: list[Coord]) -> list[int]:
    seeds = [(r, c, 0) for r, c in coords]
    return _neighbor_counts(seeds)


def _paths_overlap(
    coords: list[Coord],
    lengths: list[int],
    direction: str,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    dr, dc = (1, 1) if direction == DOWN_RIGHT else (1, -1)
    seed_set = set(coords)
    painted: set[Coord] = set()
    for (r, c), length in zip(coords, lengths):
        for step in range(1, int(length) + 1):
            rr = int(r + dr * step)
            cc = int(c + dc * step)
            if not (0 <= rr < h and 0 <= cc < w):
                break
            cell = (rr, cc)
            if cell in seed_set:
                return True
            if cell in painted:
                return True
            painted.add(cell)
    return False


def _gen_positions(h: int, w: int, direction: str) -> list[Coord] | None:
    if min(h, w) >= 15 and random.choice((False, True)):
        group_sizes = [random.choice((2, 3)), random.choice((2, 3))]
        isolated_count = 1
    else:
        group_sizes = [random.choice((2, 3, 4))]
        isolated_count = random.choice((1, 2, 3))

    coords: list[Coord] = []
    for i, _group_size in enumerate(group_sizes):
        placed = _place_group(
            h=h,
            w=w,
            direction=direction,
            existing=coords,
            size=int(_group_size),
            boundary_bias=(i == 0 and random.random() < 0.4),
        )
        if placed is None:
            return None
        coords.extend(placed)

    for _ in range(isolated_count):
        isolated = _place_isolated(h, w, direction, coords)
        if isolated is None:
            return None
        coords.append(isolated)

    lengths = _lengths_for_positions(coords)
    if not any(length == 0 for length in lengths):
        return None
    if not any(length > 0 for length in lengths):
        return None
    if max(lengths) > 4:
        return None
    if _paths_overlap(coords, lengths, direction, (h, w)):
        return None

    return coords


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    direction = random.choice((DOWN_RIGHT, DOWN_LEFT))

    coords = _gen_positions(h, w, direction)
    if coords is None:
        return None

    total_colors = 3 + len(coords)
    if total_colors > 10:
        return None
    palette = random.sample(range(10), total_colors)
    bg_colors = tuple(int(color) for color in palette[:3])
    seed_colors = [int(color) for color in palette[3:]]

    grid = _make_background(h, w, direction, bg_colors)
    for (r, c), color in zip(coords, seed_colors):
        grid[r, c] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
