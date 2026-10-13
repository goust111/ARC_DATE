from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BACKGROUND = 0
ORANGE = 7
MASK = 9
PATTERN_PALETTE = (1, 2, 3, 4, 5, 6, 8)
SIZE_CHOICES = (11, 13, 15)
GEN_RETRIES = 500


def _h_partner(width: int, r: int, c: int) -> tuple[int, int]:
    return r, width - 1 - c


def _v_partner(height: int, r: int, c: int) -> tuple[int, int]:
    return height - 1 - r, c


def _d_partner(height: int, width: int, r: int, c: int) -> tuple[int, int]:
    return height - 1 - r, width - 1 - c


def _orbit_key(height: int, width: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = {
        (r, c),
        _h_partner(width, r, c),
        _v_partner(height, r, c),
        _d_partner(height, width, r, c),
    }
    return tuple(sorted(cells))


def _forbidden_from_obstacles(shape: tuple[int, int], obstacles: list[tuple[int, int]]) -> np.ndarray:
    height, width = map(int, shape)
    forbidden = np.zeros((height, width), dtype=bool)
    for r, c in obstacles:
        forbidden[max(0, r - 1) : min(height, r + 2), max(0, c - 1) : min(width, c + 2)] = True
    return forbidden


def _first_safe_source(
    grid: np.ndarray, forbidden: np.ndarray, r: int, c: int
) -> tuple[tuple[int, int] | None, str | None]:
    height, width = map(int, grid.shape)
    candidates = (
        ("h", _h_partner(width, r, c)),
        ("v", _v_partner(height, r, c)),
        ("d", _d_partner(height, width, r, c)),
    )
    for branch, (rr, cc) in candidates:
        if int(grid[rr, cc]) != MASK and not forbidden[rr, cc]:
            return (rr, cc), branch
    return None, None


def _solve_with_branches(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], str | None]]:
    g = np.asarray(grid, dtype=int)
    forbidden = _forbidden_from_obstacles(g.shape, [tuple(map(int, p)) for p in np.argwhere(g == ORANGE)])
    out = g.copy()
    branches: dict[tuple[int, int], str | None] = {}
    for r, c in np.argwhere(g == MASK):
        rr, cc = map(int, (r, c))
        src, branch = _first_safe_source(g, forbidden, rr, cc)
        branches[(rr, cc)] = branch
        if src is not None:
            out[rr, cc] = int(g[src])
    return out, branches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair masked pixels in a doubly symmetric pattern while refusing sources near orange obstacles.

    Input structure:
    - The grid is square with a black background and a hidden pattern that is symmetric
      under both left-right and top-bottom reflection.
    - Damaged pixels are marked maroon (9).
    - One or two orange (7) obstacle pixels are overlaid after the pattern is drawn.

    Rule:
    - Treat each obstacle and its full 3x3 neighborhood as a forbidden trust zone.
    - For every maroon pixel at (r, c), try to copy its horizontal mirror partner first,
      then its vertical partner, then its 180-degree partner.
    - A candidate source is usable only if it is not maroon and not inside any forbidden zone.
    - Replace the maroon pixel with the first usable source color; keep all obstacle pixels
      and all other visible pixels unchanged.

    Key insight:
    - The hidden pattern supplies several symmetric repair candidates, but obstacle zones
      mark nearby visible colors as untrustworthy, so the solution depends on the fallback
      order horizontal -> vertical -> diagonal.
    """
    out, _branches = _solve_with_branches(np.asarray(grid, dtype=int))
    return out


def _apply_symmetry(seed: np.ndarray) -> np.ndarray:
    size = seed.shape[0] * 2 - 1
    mid = size // 2
    base = np.zeros((size, size), dtype=int)
    for r, c in np.argwhere(seed != BACKGROUND):
        color = int(seed[r, c])
        for rr, cc in {
            (int(r), int(c)),
            (int(r), size - 1 - int(c)),
            (size - 1 - int(r), int(c)),
            (size - 1 - int(r), size - 1 - int(c)),
        }:
            base[rr, cc] = color
    assert np.array_equal(base, np.fliplr(base))
    assert np.array_equal(base, np.flipud(base))
    assert base[mid, mid] == base[mid, mid]
    return base


def _shape_variants() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ]


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _randomize_shape(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = list(coords)
    max_r = max(r for r, _c in pts)
    max_c = max(c for _r, c in pts)
    if random.random() < 0.5:
        pts = [(r, max_c - c) for r, c in pts]
    if random.random() < 0.5:
        pts = [(max_r - r, c) for r, c in pts]
    if random.random() < 0.5:
        pts = [(c, r) for r, c in pts]
    return _normalize(pts)


def _build_base_pattern(size: int) -> np.ndarray | None:
    mid = size // 2
    seed = np.zeros((mid + 1, mid + 1), dtype=int)
    colors = random.sample(PATTERN_PALETTE, random.randint(3, 4))
    motif_colors = colors + [random.choice(colors) for _ in range(random.randint(3, 5))]
    random.shuffle(motif_colors)
    shapes = _shape_variants()

    for color in colors:
        placed = False
        for _ in range(120):
            coords = _randomize_shape(random.choice(shapes))
            max_r = max(r for r, _c in coords)
            max_c = max(c for _r, c in coords)
            if max_r > mid or max_c > mid:
                continue
            r0 = random.randint(0, mid - max_r)
            c0 = random.randint(0, mid - max_c)
            cells = [(r0 + dr, c0 + dc) for dr, dc in coords]
            if any(seed[r, c] not in (BACKGROUND, color) for r, c in cells):
                continue
            for r, c in cells:
                seed[r, c] = color
            placed = True
            break
        if not placed:
            return None

    for color in motif_colors:
        for _ in range(60):
            coords = _randomize_shape(random.choice(shapes))
            max_r = max(r for r, _c in coords)
            max_c = max(c for _r, c in coords)
            if max_r > mid or max_c > mid:
                continue
            r0 = random.randint(0, mid - max_r)
            c0 = random.randint(0, mid - max_c)
            cells = [(r0 + dr, c0 + dc) for dr, dc in coords]
            if any(seed[r, c] not in (BACKGROUND, color) for r, c in cells):
                continue
            for r, c in cells:
                seed[r, c] = color
            break

    strict_ul = seed[:mid, :mid]
    if np.count_nonzero(strict_ul) < 7:
        return None

    base = _apply_symmetry(seed)
    non_bg = int(np.count_nonzero(base))
    if non_bg < size + 12 or non_bg > (size * size * 3) // 5:
        return None
    if len({int(c) for c in np.unique(base) if int(c) != BACKGROUND}) < 3:
        return None
    return base


def _shared_obstacle_centers(
    size: int, s1: tuple[int, int], s2: tuple[int, int]
) -> list[tuple[int, int]]:
    rows = range(max(1, max(s1[0], s2[0]) - 1), min(size - 1, min(s1[0], s2[0]) + 2))
    cols = range(max(1, max(s1[1], s2[1]) - 1), min(size - 1, min(s1[1], s2[1]) + 2))
    centers = []
    for r in rows:
        for c in cols:
            if max(abs(r - s1[0]), abs(c - s1[1])) <= 1 and max(abs(r - s2[0]), abs(c - s2[1])) <= 1:
                if c > size // 2 and r < size // 2 and (r, c) not in {s1, s2}:
                    centers.append((r, c))
    return centers


def _choose_core_cases(
    base: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
] | None:
    size = base.shape[0]
    mid = size // 2
    ul_cells = [(r, c) for r in range(mid) for c in range(mid) if int(base[r, c]) != BACKGROUND]
    random.shuffle(ul_cells)

    for p_v, p_d in combinations(ul_cells, 2):
        s_v = _h_partner(size, *p_v)
        s_d = _h_partner(size, *p_d)
        centers = _shared_obstacle_centers(size, s_v, s_d)
        if not centers:
            continue
        obstacle = random.choice(centers)
        forbidden = _forbidden_from_obstacles(base.shape, [obstacle])
        if forbidden[_v_partner(size, *p_v)] or forbidden[_d_partner(size, size, *p_d)]:
            continue
        used_orbits = {
            _orbit_key(size, size, *p_v),
            _orbit_key(size, size, *p_d),
        }
        h_candidates = [
            p_h
            for p_h in ul_cells
            if _orbit_key(size, size, *p_h) not in used_orbits
            and not forbidden[_h_partner(size, *p_h)]
        ]
        if not h_candidates:
            continue
        p_h = random.choice(h_candidates)
        return [obstacle], p_h, p_v, p_d, _v_partner(size, *p_d)
    return None


def _pick_second_obstacle(
    base: np.ndarray,
    first_obstacle: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    size = base.shape[0]
    cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    random.shuffle(cells)
    for cell in cells:
        if cell == first_obstacle:
            continue
        if max(abs(cell[0] - first_obstacle[0]), abs(cell[1] - first_obstacle[1])) <= 2:
            continue
        forbidden = _forbidden_from_obstacles(base.shape, [cell])
        if any(forbidden[r, c] for r, c in reserved):
            continue
        return cell
    return None


def _choose_recolor(base_color: int) -> int:
    choices = [color for color in PATTERN_PALETTE if color != base_color]
    return random.choice(choices)


def _add_extra_masks(
    base: np.ndarray,
    grid: np.ndarray,
    forbidden: np.ndarray,
    used_orbits: set[tuple[tuple[int, int], ...]],
    masks: dict[tuple[int, int], str],
    polluted_sources: set[tuple[int, int]],
) -> None:
    size = base.shape[0]
    mid = size // 2
    candidates = [(r, c) for r in range(mid) for c in range(mid) if int(base[r, c]) != BACKGROUND]
    random.shuffle(candidates)
    extra_budget = random.randint(0, 3)
    for r, c in candidates:
        if extra_budget <= 0:
            break
        orbit = _orbit_key(size, size, r, c)
        if orbit in used_orbits:
            continue
        h = _h_partner(size, r, c)
        v = _v_partner(size, r, c)
        if not forbidden[h]:
            grid[r, c] = MASK
            masks[(r, c)] = "h"
            used_orbits.add(orbit)
            extra_budget -= 1
            continue
        if forbidden[h] and not forbidden[v] and h not in polluted_sources:
            grid[r, c] = MASK
            grid[h] = _choose_recolor(int(base[h]))
            masks[(r, c)] = "v"
            polluted_sources.add(h)
            used_orbits.add(orbit)
            extra_budget -= 1


def _validate_instance(
    base: np.ndarray,
    grid: np.ndarray,
    expected_branches: dict[tuple[int, int], str],
    polluted_sources: set[tuple[int, int]],
) -> None:
    assert base.shape == grid.shape
    size = base.shape[0]
    assert size == base.shape[1] and size <= 30
    assert np.array_equal(base, np.flipud(base))
    assert np.array_equal(base, np.fliplr(base))

    obstacles = [tuple(map(int, p)) for p in np.argwhere(grid == ORANGE)]
    assert 1 <= len(obstacles) <= 2
    forbidden = _forbidden_from_obstacles(grid.shape, obstacles)

    out, actual_branches = _solve_with_branches(grid)
    mask_cells = [tuple(map(int, p)) for p in np.argwhere(grid == MASK)]
    assert len(mask_cells) >= 4
    assert np.array_equal(out[grid != MASK], grid[grid != MASK])

    branch_set = set()
    for cell, branch in expected_branches.items():
        assert actual_branches[cell] == branch
        branch_set.add(branch)
        assert int(out[cell]) == int(base[cell])
        src, _branch = _first_safe_source(grid, forbidden, *cell)
        assert src is not None
        assert int(grid[src]) == int(base[cell])
    assert branch_set >= {"h", "v", "d"}

    for cell in polluted_sources:
        assert forbidden[cell]
        assert int(grid[cell]) not in (ORANGE, MASK)
        assert int(grid[cell]) != int(base[cell])

    untouched = (grid != MASK) & ~forbidden & (grid != ORANGE)
    assert np.array_equal(grid[untouched], base[untouched])

    for cell, branch in expected_branches.items():
        r, c = cell
        h = _h_partner(size, r, c)
        v = _v_partner(size, r, c)
        if branch == "v":
            assert forbidden[h] and int(grid[h]) != int(base[cell])
            assert not forbidden[v]
        elif branch == "d":
            assert forbidden[h] and int(grid[h]) != int(base[cell])
            assert int(grid[v]) == MASK
            d = _d_partner(size, size, r, c)
            assert not forbidden[d]


def _generate_attempt() -> np.ndarray | None:
    size = random.choice(SIZE_CHOICES)
    base = _build_base_pattern(size)
    if base is None:
        return None

    core = _choose_core_cases(base)
    if core is None:
        return None
    obstacles, p_h, p_v, p_d, p_d_v = core

    reserved_cells = {
        p_h,
        p_v,
        p_d,
        p_d_v,
        _h_partner(size, *p_h),
        _v_partner(size, *p_v),
        _d_partner(size, size, *p_d),
        _h_partner(size, *p_v),
        _h_partner(size, *p_d),
    }
    if random.random() < 0.55:
        second = _pick_second_obstacle(base, obstacles[0], reserved_cells)
        if second is not None:
            obstacles.append(second)

    forbidden = _forbidden_from_obstacles(base.shape, obstacles)
    if forbidden[_h_partner(size, *p_h)]:
        return None
    if forbidden[_v_partner(size, *p_v)]:
        return None
    if forbidden[_d_partner(size, size, *p_d)]:
        return None

    grid = base.copy()
    for r, c in obstacles:
        grid[r, c] = ORANGE

    polluted_sources: set[tuple[int, int]] = set()
    s_v = _h_partner(size, *p_v)
    s_d = _h_partner(size, *p_d)
    for cell in (s_v, s_d):
        if cell in obstacles:
            return None
        grid[cell] = _choose_recolor(int(base[cell]))
        polluted_sources.add(cell)

    expected_branches = {
        p_h: "h",
        p_v: "v",
        p_d: "d",
        p_d_v: "h",
    }
    for cell in expected_branches:
        grid[cell] = MASK

    used_orbits = {
        _orbit_key(size, size, *p_h),
        _orbit_key(size, size, *p_v),
        _orbit_key(size, size, *p_d),
    }
    _add_extra_masks(base, grid, forbidden, used_orbits, expected_branches, polluted_sources)

    extra_forbidden_cells = [
        cell
        for cell in np.argwhere(forbidden)
        if tuple(map(int, cell)) not in polluted_sources
        and tuple(map(int, cell)) not in obstacles
        and int(base[tuple(map(int, cell))]) != BACKGROUND
        and int(grid[tuple(map(int, cell))]) != MASK
    ]
    random.shuffle(extra_forbidden_cells)
    for cell_arr in extra_forbidden_cells[: random.randint(0, 2)]:
        cell = tuple(map(int, cell_arr))
        grid[cell] = _choose_recolor(int(base[cell]))
        polluted_sources.add(cell)

    try:
        _validate_instance(base, grid, expected_branches, polluted_sources)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-constrained symmetry-repair input")
