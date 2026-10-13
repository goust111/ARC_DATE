from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


class Color:
    """Canonical ARC palette ids, mirroring arc_utils.Color."""

    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Small self-contained 4-connected components helper for generator validation."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _trace_ray_cells(walls: np.ndarray, source: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Return horizontal and vertical cells reached by a source, excluding the source cell."""
    h, w = walls.shape
    r, c = source
    horizontal: set[tuple[int, int]] = set()
    vertical: set[tuple[int, int]] = set()

    for dc in (-1, 1):
        cc = c + dc
        while 0 <= cc < w and not walls[r, cc]:
            horizontal.add((r, cc))
            cc += dc

    for dr in (-1, 1):
        rr = r + dr
        while 0 <= rr < h and not walls[rr, c]:
            vertical.add((rr, c))
            rr += dr

    return horizontal, vertical


def _solve_fixpoint(grid: np.ndarray) -> np.ndarray:
    """Generic fixpoint solver: supports any number of yellow/red sources."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    walls = out == Color.GREY
    sources = {tuple(map(int, coord)) for coord in np.argwhere((out == Color.YELLOW) | (out == Color.RED))}

    if not sources:
        return out

    while True:
        horizontal_hits: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        vertical_hits: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        green_targets: set[tuple[int, int]] = set()

        for source in sources:
            horizontal, vertical = _trace_ray_cells(walls, source)
            for cell in horizontal:
                horizontal_hits[cell].add(source)
                green_targets.add(cell)
            for cell in vertical:
                vertical_hits[cell].add(source)
                green_targets.add(cell)

        new_sources: set[tuple[int, int]] = set()
        for cell in set(horizontal_hits) & set(vertical_hits):
            if int(out[cell]) != Color.BLACK:
                continue
            if len(horizontal_hits[cell] | vertical_hits[cell]) >= 2:
                new_sources.add(cell)

        for r, c in green_targets:
            if (r, c) in new_sources:
                continue
            if int(out[r, c]) == Color.BLACK:
                out[r, c] = Color.GREEN

        for r, c in new_sources:
            out[r, c] = Color.RED

        if not new_sources:
            return out
        sources |= new_sources


def main(grid: np.ndarray) -> np.ndarray:
    """Let wall-blocked cardinal rays ignite red crosspoints until the source set reaches a fixpoint.

    Input structure:
    - A black background contains grey wall segments and exactly two yellow seed pixels.
    - The two seeds are on different rows and different columns.
    - Walls are the only cells that block motion; other colored cells are pass-through.

    Rule:
    - Every source pixel emits green rays left, right, up, and down until the next grey wall or the grid edge.
    - When a horizontal ray from one source and a vertical ray from a different source meet on a black cell, that cell turns red and becomes a new source.
    - Repeat the same emission-and-ignition process until no new red source appears; never overwrite existing non-black cells, so the original yellow seeds stay yellow.

    Key insight:
    - The output is the stable endpoint of a causal process: red intersections are both results and new causes, while grey walls determine where each ray chain can continue or die out.
    """
    return _solve_fixpoint(grid)


def _segment_clear_row(walls: np.ndarray, r: int, c0: int, c1: int) -> bool:
    step = 1 if c1 > c0 else -1
    for c in range(c0 + step, c1 + step, step):
        if walls[r, c]:
            return False
    return True


def _segment_clear_col(walls: np.ndarray, c: int, r0: int, r1: int) -> bool:
    step = 1 if r1 > r0 else -1
    for r in range(r0 + step, r1 + step, step):
        if walls[r, c]:
            return False
    return True


def _solve_two_seed_closed_form(grid: np.ndarray) -> np.ndarray:
    """
    Independent validator for generator cases.

    With exactly two initial seeds, every red source can only lie at one of the two
    row/column crosspoints, so the fixpoint reduces to testing those two corners.
    """
    g = np.asarray(grid, dtype=int)
    seeds = [tuple(map(int, coord)) for coord in np.argwhere(g == Color.YELLOW)]
    assert len(seeds) == 2
    (r1, c1), (r2, c2) = seeds
    walls = g == Color.GREY

    out = g.copy()
    green_targets: set[tuple[int, int]] = set()
    for source in seeds:
        horizontal, vertical = _trace_ray_cells(walls, source)
        green_targets |= horizontal | vertical

    red_targets: set[tuple[int, int]] = set()
    cross_a = (r1, c2)
    cross_b = (r2, c1)
    if int(g[cross_a]) == Color.BLACK and _segment_clear_row(walls, r1, c1, c2) and _segment_clear_col(walls, c2, r2, r1):
        red_targets.add(cross_a)
    if int(g[cross_b]) == Color.BLACK and _segment_clear_row(walls, r2, c2, c1) and _segment_clear_col(walls, c1, r1, r2):
        red_targets.add(cross_b)

    for r, c in green_targets:
        if (r, c) not in red_targets and int(out[r, c]) == Color.BLACK:
            out[r, c] = Color.GREEN
    for r, c in red_targets:
        out[r, c] = Color.RED
    return out


def _apply_segment(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = Color.GREY


def _segment_cells(start_r: int, start_c: int, length: int, horizontal: bool) -> list[tuple[int, int]]:
    if horizontal:
        return [(start_r, start_c + i) for i in range(length)]
    return [(start_r + i, start_c) for i in range(length)]


def _pick_segment_within(interval_lo: int, interval_hi: int, fixed: int, horizontal: bool) -> list[tuple[int, int]]:
    length_max = min(4, interval_hi - interval_lo + 1)
    length = random.randint(2, length_max)
    start = random.randint(interval_lo, interval_hi - length + 1)
    if horizontal:
        return _segment_cells(fixed, start, length, True)
    return _segment_cells(start, fixed, length, False)


def _place_outer_caps(
    grid: np.ndarray,
    seed1: tuple[int, int],
    seed2: tuple[int, int],
) -> None:
    h, w = grid.shape
    (r1, c1), (r2, c2) = seed1, seed2

    if c1 < c2:
        _apply_segment(grid, _pick_segment_within(1, c1 - 2, r1, True))
        _apply_segment(grid, _pick_segment_within(c2 + 2, w - 2, r2, True))
    else:
        _apply_segment(grid, _pick_segment_within(c1 + 2, w - 2, r1, True))
        _apply_segment(grid, _pick_segment_within(1, c2 - 2, r2, True))

    if r1 < r2:
        _apply_segment(grid, _pick_segment_within(1, r1 - 2, c1, False))
        _apply_segment(grid, _pick_segment_within(r2 + 2, h - 2, c2, False))
    else:
        _apply_segment(grid, _pick_segment_within(r1 + 2, h - 2, c1, False))
        _apply_segment(grid, _pick_segment_within(1, r2 - 2, c2, False))


def _place_blocker_for_cross(
    grid: np.ndarray,
    seed1: tuple[int, int],
    seed2: tuple[int, int],
    cross: tuple[int, int],
) -> None:
    (r1, c1), (r2, c2) = seed1, seed2
    rr, cc = cross
    use_row = random.random() < 0.5

    if (rr, cc) == (r1, c2):
        if use_row:
            lo, hi = sorted((c1, c2))
            _apply_segment(grid, _pick_segment_within(lo + 1, hi - 1, r1, True))
        else:
            lo, hi = sorted((r1, r2))
            _apply_segment(grid, _pick_segment_within(lo + 1, hi - 1, c2, False))
        return

    lo, hi = sorted((c1, c2))
    if use_row:
        _apply_segment(grid, _pick_segment_within(lo + 1, hi - 1, r2, True))
    else:
        lo, hi = sorted((r1, r2))
        _apply_segment(grid, _pick_segment_within(lo + 1, hi - 1, c1, False))


def _random_free_segment(grid: np.ndarray, forbidden_rows: set[int], forbidden_cols: set[int]) -> bool:
    h, w = grid.shape
    for _ in range(40):
        horizontal = random.random() < 0.5
        length = random.randint(2, 4)
        if horizontal:
            r = random.randint(1, h - 2)
            c = random.randint(1, w - length - 1)
            if r in forbidden_rows:
                continue
            cells = _segment_cells(r, c, length, True)
        else:
            r = random.randint(1, h - length - 1)
            c = random.randint(1, w - 2)
            if c in forbidden_cols:
                continue
            cells = _segment_cells(r, c, length, False)

        ok = True
        cell_set = set(cells)
        for rr, cc in cells:
            if int(grid[rr, cc]) != Color.BLACK:
                ok = False
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in cell_set:
                        continue
                    if int(grid[nr, nc]) != Color.BLACK:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            _apply_segment(grid, cells)
            return True
    return False


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


_GEN_BRANCH_COUNTER = 0
_GEN_MAX_ATTEMPTS = 400


def _next_target_red_count() -> int:
    global _GEN_BRANCH_COUNTER
    target = _GEN_BRANCH_COUNTER % 3
    _GEN_BRANCH_COUNTER += 1
    return target


def _build_case(target_red_count: int) -> np.ndarray:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    grid = np.full((h, w), Color.BLACK, dtype=int)

    top = random.randint(4, h - 9)
    bottom = random.randint(top + 4, h - 5)
    left = random.randint(4, w - 9)
    right = random.randint(left + 4, w - 5)

    if random.random() < 0.5:
        seed1 = (top, left)
        seed2 = (bottom, right)
    else:
        seed1 = (top, right)
        seed2 = (bottom, left)

    grid[seed1] = Color.YELLOW
    grid[seed2] = Color.YELLOW

    _place_outer_caps(grid, seed1, seed2)

    cross_a = (seed1[0], seed2[1])
    cross_b = (seed2[0], seed1[1])
    if target_red_count == 0:
        blocked = {cross_a, cross_b}
    elif target_red_count == 1:
        blocked = {random.choice([cross_a, cross_b])}
    else:
        blocked = set()

    for cross in blocked:
        _place_blocker_for_cross(grid, seed1, seed2, cross)

    forbidden_rows = {seed1[0], seed2[0]}
    forbidden_cols = {seed1[1], seed2[1]}
    for _ in range(random.randint(0, 2)):
        _random_free_segment(grid, forbidden_rows, forbidden_cols)

    return _random_transform(grid)


def _valid_case(grid: np.ndarray, target_red_count: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all(np.isin(grid, [Color.BLACK, Color.YELLOW, Color.GREY])):
        return False

    seeds = np.argwhere(grid == Color.YELLOW)
    if seeds.shape != (2, 2):
        return False
    if int(seeds[0, 0]) == int(seeds[1, 0]) or int(seeds[0, 1]) == int(seeds[1, 1]):
        return False

    wall_components = _components_mask(grid == Color.GREY)
    if len(wall_components) < 4:
        return False

    out = main(grid)
    expected = _solve_two_seed_closed_form(grid)
    if not np.array_equal(out, expected):
        return False
    if not np.array_equal(main(out), out):
        return False

    if int(np.count_nonzero(out == Color.RED)) != target_red_count:
        return False

    green_count = int(np.count_nonzero(out == Color.GREEN))
    if green_count < 6:
        return False
    if green_count > grid.size // 3:
        return False

    return True


def generate_input() -> np.ndarray:
    target_red_count = _next_target_red_count()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _build_case(target_red_count)
        if _valid_case(grid, target_red_count):
            return grid
    raise RuntimeError(f"failed to generate a valid case for target_red_count={target_red_count}")
