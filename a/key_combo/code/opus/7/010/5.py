from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 5
RECT = 0
SOURCE_COLORS = (1, 2, 3, 4, 7)
GEN_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_slices(rect: tuple[int, int, int, int]) -> tuple[slice, slice]:
    r0, c0, r1, c1 = rect
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _segment(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = src
    tr, tc = dst
    if (sr, sc) == (tr, tc):
        return [src]
    if sr == tr:
        step = _sign(tc - sc)
        return [(sr, c) for c in range(sc, tc + step, step)]
    if sc == tc:
        step = _sign(tr - sr)
        return [(r, sc) for r in range(sr, tr + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _row_target(rect: tuple[int, int, int, int], r: int, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    if side == "left":
        return (r, c0 - 1) if r0 <= r <= r1 else (r, c0)
    return (r, c1 + 1) if r0 <= r <= r1 else (r, c1)


def _col_target(rect: tuple[int, int, int, int], c: int, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return (r0 - 1, c) if c0 <= c <= c1 else (r0, c)
    return (r1 + 1, c) if c0 <= c <= c1 else (r1, c)


def _projection_options(
    coord: tuple[int, int],
    rect: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int, int, int, tuple[int, int], list[tuple[int, int]]]]:
    r, c = coord
    r0, c0, r1, c1 = rect
    h, w = shape
    options = []

    if r0 - 1 <= r <= r1 + 1 and c < c0:
        target = _row_target(rect, r, "left")
        if 0 <= target[0] < h and 0 <= target[1] < w:
            dist = abs(c - target[1])
            options.append((dist, r, c, 0, target, _segment(coord, target)))
    if r0 - 1 <= r <= r1 + 1 and c > c1:
        target = _row_target(rect, r, "right")
        if 0 <= target[0] < h and 0 <= target[1] < w:
            dist = abs(c - target[1])
            options.append((dist, r, c, 0, target, _segment(coord, target)))
    if c0 - 1 <= c <= c1 + 1 and r < r0:
        target = _col_target(rect, c, "top")
        if 0 <= target[0] < h and 0 <= target[1] < w:
            dist = abs(r - target[0])
            options.append((dist, r, c, 1, target, _segment(coord, target)))
    if c0 - 1 <= c <= c1 + 1 and r > r1:
        target = _col_target(rect, c, "bottom")
        if 0 <= target[0] < h and 0 <= target[1] < w:
            dist = abs(r - target[0])
            options.append((dist, r, c, 1, target, _segment(coord, target)))
    return options


def _choose_projection(
    coord: tuple[int, int],
    rect: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    options = _projection_options(coord, rect, shape)
    if not options:
        raise AssertionError(f"source {coord} is not aligned with the rectangle")
    dist, src_r, src_c, axis_rank, target, path = min(options)
    _ = dist, src_r, src_c, axis_rank
    return target, path


def _simulate_draw(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int, int, int]]]]:
    out = grid.copy()
    occupancy: dict[tuple[int, int], tuple[tuple[int, int, int], int]] = {}
    contenders: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)

    for r, c in np.argwhere((grid != BG) & (grid != RECT)):
        sr, sc = int(r), int(c)
        color = int(grid[sr, sc])
        _target, path = _choose_projection((sr, sc), rect, grid.shape)
        for pr, pc in path:
            dist = abs(pr - sr) + abs(pc - sc)
            contenders[(pr, pc)].append((sr, sc, color, dist))
            key = (dist, sr, sc)
            prev = occupancy.get((pr, pc))
            if prev is None or key < prev[0]:
                occupancy[(pr, pc)] = (key, color)

    for (r, c), (_key, color) in occupancy.items():
        out[r, c] = color
    return out, contenders


def _boundary_cells(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    h, w = shape
    cells = []
    if r0 - 1 >= 0:
        cells.extend((r0 - 1, c) for c in range(c0, c1 + 1))
    if r1 + 1 < h:
        cells.extend((r1 + 1, c) for c in range(c0, c1 + 1))
    if c0 - 1 >= 0:
        cells.extend((r, c0 - 1) for r in range(r0, r1 + 1))
    if c1 + 1 < w:
        cells.extend((r, c1 + 1) for r in range(r0, r1 + 1))
    return cells


def _vote_color(out: np.ndarray, rect: tuple[int, int, int, int]) -> int | None:
    touches = []
    for cell in _boundary_cells(rect, out.shape):
        color = int(out[cell])
        if color not in (BG, RECT):
            touches.append((cell, color))
    if not touches:
        return None
    counts = Counter(color for _cell, color in touches)
    best_count = max(counts.values())
    tied = {color for color, count in counts.items() if count == best_count}
    best_cell, best_color = min((cell, color) for cell, color in touches if color in tied)
    _ = best_cell
    return best_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw competing straight projections into a central rectangle and recolor the rectangle by the winning boundary color.

    Input structure:
    - Grey background surrounds one filled black rectangle near the center and several isolated colored source pixels.
    - Each source lies on a row or column that can project straight toward a rectangle side, including one-cell corner extensions just outside the rectangle.

    Rule:
    - For each source, project a straight horizontal or vertical segment toward the rectangle and stop on the last outside cell that touches the rectangle side it reaches.
    - If both a row-based and a column-based projection are possible, use the shorter one; equal distances prefer the row-based projection.
    - When multiple segments claim the same cell, keep the color from the source with smaller Manhattan distance to that cell; equal distances break ties by source row, then source column.
    - After drawing all resolved segments, inspect the non-background cells orthogonally adjacent to the rectangle and recolor the whole rectangle with the most frequent touching color; equal vote counts use row-major boundary position.

    Key insight:
    - The rectangle color is decided by the colors that actually survive all projection collisions on the boundary, not by the raw source counts.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    rect_mask = g == RECT
    if not rect_mask.any():
        raise AssertionError("expected a black rectangle")
    rect = _bbox(rect_mask)
    rs, cs = _rect_slices(rect)
    if not np.all(g[rs, cs] == RECT):
        raise AssertionError("black object must be a filled rectangle")

    out, _contenders = _simulate_draw(g, rect)
    fill = _vote_color(out, rect)
    if fill is not None:
        out[rs, cs] = fill
    return out


def _candidate_positions(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect
    candidates = set()
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(0, c0):
            candidates.add((r, c))
        for c in range(c1 + 1, w):
            candidates.add((r, c))
    for c in range(max(0, c0 - 1), min(w, c1 + 2)):
        for r in range(0, r0):
            candidates.add((r, c))
        for r in range(r1 + 1, h):
            candidates.add((r, c))
    return [coord for coord in candidates if _projection_options(coord, rect, shape)]


def _path_is_clear(
    coord: tuple[int, int],
    occupied: set[tuple[int, int]],
    rect: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> bool:
    _target, path = _choose_projection(coord, rect, shape)
    return all(cell == coord or cell not in occupied for cell in path)


def _choose_conflict_sources(
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[tuple[int, int], int]]:
    r0, c0, r1, c1 = rect
    corner = rng.choice(("tl", "tr", "bl", "br"))
    colors = rng.sample(SOURCE_COLORS, 2)

    if corner == "tl":
        hsrc = (r0 - 1, rng.randrange(0, c0))
        vsrc = (rng.randrange(0, r0), c0)
    elif corner == "tr":
        hsrc = (r0 - 1, rng.randrange(c1 + 1, shape[1]))
        vsrc = (rng.randrange(0, r0), c1)
    elif corner == "bl":
        hsrc = (r1 + 1, rng.randrange(0, c0))
        vsrc = (rng.randrange(r1 + 1, shape[0]), c0)
    else:
        hsrc = (r1 + 1, rng.randrange(c1 + 1, shape[1]))
        vsrc = (rng.randrange(r1 + 1, shape[0]), c1)

    result = [(hsrc, colors[0]), (vsrc, colors[1])]
    if rng.random() < 0.35:
        bonus = {
            "tl": (r0 - 1, c0 - 1),
            "tr": (r0 - 1, c1 + 1),
            "bl": (r1 + 1, c0 - 1),
            "br": (r1 + 1, c1 + 1),
        }[corner]
        result.append((bonus, rng.choice(SOURCE_COLORS)))
    return result


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(11, 15)
    w = rng.randint(11, 15)
    rect_h = rng.randint(3, 4)
    rect_w = rng.randint(3, 4)

    r_choices = list(range(2, h - rect_h - 1))
    c_choices = list(range(2, w - rect_w - 1))
    if not r_choices or not c_choices:
        return None
    center_r = (h - rect_h) // 2
    center_c = (w - rect_w) // 2
    r0 = min(r_choices, key=lambda x: (abs(x - center_r), rng.random()))
    c0 = min(c_choices, key=lambda x: (abs(x - center_c), rng.random()))
    rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

    grid = np.full((h, w), BG, dtype=int)
    rs, cs = _rect_slices(rect)
    grid[rs, cs] = RECT

    occupied = {tuple(map(int, rc)) for rc in np.argwhere(grid != BG)}
    placements: dict[tuple[int, int], int] = {}

    for coord, color in _choose_conflict_sources(grid.shape, rect, rng):
        if coord in occupied or not _path_is_clear(coord, occupied, rect, grid.shape):
            return None
        placements[coord] = color
        occupied.add(coord)

    candidates = _candidate_positions(grid.shape, rect)
    rng.shuffle(candidates)
    target_count = rng.randint(5, 8)
    for coord in candidates:
        if len(placements) >= target_count:
            break
        if coord in occupied:
            continue
        if not _path_is_clear(coord, occupied, rect, grid.shape):
            continue
        placements[coord] = rng.choice(SOURCE_COLORS)
        occupied.add(coord)

    if len(placements) < 5:
        return None
    if len(set(placements.values())) < 2:
        return None

    for (r, c), color in placements.items():
        grid[r, c] = color

    drawn, contenders = _simulate_draw(grid, rect)
    conflict_cells = [cell for cell, claims in contenders.items() if len(claims) >= 2]
    if not conflict_cells:
        return None

    boundary = _boundary_cells(rect, grid.shape)
    if not any(cell in conflict_cells for cell in boundary):
        return None

    boundary_colors = [int(drawn[cell]) for cell in boundary if int(drawn[cell]) not in (BG, RECT)]
    if len(boundary_colors) < 3:
        return None
    if len(set(boundary_colors)) < 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    raw_majority = Counter(placements.values()).most_common(1)[0][0]
    final_color = int(out[rs, cs][0, 0])
    if rng.random() < 0.5 and final_color == raw_majority:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
