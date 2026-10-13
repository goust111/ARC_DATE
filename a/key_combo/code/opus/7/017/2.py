from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SEED_COLORS = (BLUE, RED, GREEN)
RECOLOR_BY_BLOCKED = {1: BLUE, 2: RED, 3: GREEN, 4: YELLOW}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _seed_col_for_cell(r: int, c: int, h: int) -> int:
    return int(r + c - (h - 1))


def _canonical_shape(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sweep the colored bottom-row pixels diagonally upward-right, suppress hits on grey obstacles, then recolor each obstacle by how many sweep pixels it blocked.

    Input structure:
    - A black background grid contains several non-black seed pixels on the bottom row.
    - Grey pixels form 4-connected obstacle sprites in the interior.
    - Seed colors are copied along upward-right diagonals while obstacles stay fixed.

    Rule:
    - For every non-black bottom-row seed, repeatedly translate it by (-1, +1), keeping its color, until it leaves the grid.
    - When a sweep pixel would land on a grey obstacle cell, do not draw that sweep pixel and instead count one blocked hit for that obstacle component.
    - After all sweeps finish, recolor each obstacle uniformly by its blocked-hit count: 1->blue, 2->red, 3->green, 4->yellow.

    Key insight:
    - An obstacle's final color is determined by the exact set of diagonal trajectories that intersect it, so solving requires simulating the sweep and counting blocked intersections per connected obstacle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    obstacle_mask = g == GREY
    obstacle_comps = _components_mask(obstacle_mask)
    obstacle_id = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(obstacle_comps):
        obstacle_id[comp[:, 0], comp[:, 1]] = idx

    out = g.copy()
    blocked = [0] * len(obstacle_comps)
    bottom = g[h - 1]
    seeds = [(c, int(bottom[c])) for c in range(w) if int(bottom[c]) not in (BLACK, GREY)]

    for seed_c, color in seeds:
        for step in range(h):
            r = h - 1 - step
            c = seed_c + step
            if c >= w:
                break
            if obstacle_mask[r, c]:
                blocked[int(obstacle_id[r, c])] += 1
            else:
                out[r, c] = color

    for idx, comp in enumerate(obstacle_comps):
        count = int(blocked[idx])
        assert count in RECOLOR_BY_BLOCKED, f"invalid blocked count {count}"
        out[comp[:, 0], comp[:, 1]] = RECOLOR_BY_BLOCKED[count]

    return out


_LEFT_SHAPES = {
    1: [(0, 0), (0, -1), (-1, 0)],
    2: [(0, 0), (0, 1), (-1, 0)],
    3: [(0, 0), (0, 1), (0, 2), (-1, 0)],
    4: [(0, 0), (0, 1), (0, 2), (0, 3)],
}
_RIGHT_SHAPES = {
    1: [(0, 0), (0, 1), (1, 0)],
    2: [(0, 0), (0, 1), (1, 1)],
    3: [(0, 0), (0, 1), (0, 2), (1, 2)],
    4: [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)],
}

_COUNT_CHOICES = {
    2: [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    3: [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
    4: [(1, 2, 3, 4)],
}


def _choose_counts(rng: np.random.Generator) -> list[int]:
    obstacle_count = int(rng.choice([2, 3, 4], p=[0.18, 0.57, 0.25]))
    counts = list(_COUNT_CHOICES[obstacle_count][int(rng.integers(len(_COUNT_CHOICES[obstacle_count])))] )
    if rng.random() < 0.5:
        counts.reverse()
    return counts


def _build_seed_groups(rng: np.random.Generator, width: int, counts: list[int]) -> list[int] | None:
    n = len(counts)
    gap_count = n - 1
    base_required = sum(counts) + gap_count + 2
    if base_required > width:
        return None

    extra = width - base_required
    left_pad = 1 + int(rng.integers(extra + 1))
    extra -= left_pad - 1
    right_pad = 1 + int(rng.integers(extra + 1))
    extra -= right_pad - 1
    gaps = [1] * gap_count
    for _ in range(extra):
        gaps[int(rng.integers(gap_count))] += 1

    starts = []
    cur = left_pad
    for i, count in enumerate(counts):
        starts.append(cur)
        cur += count
        if i < gap_count:
            cur += gaps[i]

    if cur + right_pad != width:
        return None
    return starts


def _shape_coords(count: int, side: str) -> list[tuple[int, int]]:
    if side == "left":
        return list(_LEFT_SHAPES[count])
    return list(_RIGHT_SHAPES[count])


def _row_options(
    height: int,
    width: int,
    starts: list[int],
    counts: list[int],
    sides: list[str],
) -> list[tuple[int, ...]]:
    mins = []
    maxs = []
    for start, count, side in zip(starts, counts, sides):
        shape = _shape_coords(count, side)
        min_dr = min(dr for dr, _ in shape)
        max_dr = max(dr for dr, _ in shape)
        max_dc = max(dc for _, dc in shape)
        lower = max(-min_dr, height - width + start + max_dc)
        upper = height - 1 - max_dr
        if lower > upper:
            return []
        mins.append(lower)
        maxs.append(upper)

    out = []
    for rows in itertools.combinations(range(1, height - 1), len(starts)):
        if any(rows[i + 1] - rows[i] < 2 for i in range(len(rows) - 1)):
            continue
        if all(mins[i] <= rows[i] <= maxs[i] for i in range(len(rows))):
            out.append(rows)
    return out


def _place_obstacles(
    height: int,
    starts: list[int],
    counts: list[int],
    sides: list[str],
    rows: tuple[int, ...],
) -> list[list[tuple[int, int]]] | None:
    occupied: set[tuple[int, int]] = set()
    shapes = []
    for start, count, side, row in zip(starts, counts, sides, rows):
        shift = height - 1 - row
        col0 = start + shift
        coords = [(row + dr, col0 + dc) for dr, dc in _shape_coords(count, side)]
        if len(coords) != len(set(coords)):
            return None
        if any(r < 0 or c < 0 for r, c in coords):
            return None
        shapes.append(coords)

    for i, coords in enumerate(shapes):
        coord_set = set(coords)
        if occupied & coord_set:
            return None
        for r, c in coord_set:
            for rr, cc in occupied:
                if max(abs(r - rr), abs(c - cc)) <= 1:
                    return None
        occupied |= coord_set
    return shapes


def _available_distractors(
    width: int,
    forbidden_seed_cols: set[int],
) -> list[int]:
    return [c for c in range(width) if c not in forbidden_seed_cols]


def _colorize_bottom(
    rng: np.random.Generator,
    width: int,
    active_seed_cols: list[int],
    distractor_cols: list[int],
) -> np.ndarray:
    bottom = np.zeros(width, dtype=int)
    all_cols = active_seed_cols + distractor_cols
    colors = rng.choice(SEED_COLORS, size=len(all_cols), replace=True).astype(int)
    if len(all_cols) >= 2 and len(set(map(int, colors))) == 1:
        colors[0] = SEED_COLORS[(SEED_COLORS.index(int(colors[0])) + 1) % len(SEED_COLORS)]
    for col, color in zip(all_cols, colors):
        bottom[int(col)] = int(color)
    return bottom


def _expected_counts_from_grid(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    comps = _components_mask(grid == GREY)
    bottom = grid[h - 1]
    counts = []
    for comp in comps:
        count = 0
        for r, c in comp:
            seed_c = _seed_col_for_cell(int(r), int(c), h)
            if 0 <= seed_c < w and int(bottom[seed_c]) != BLACK:
                count += 1
        counts.append(int(count))
    return counts


def _oracle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    obstacle_mask = g == GREY
    counts = _expected_counts_from_grid(g)
    comps = _components_mask(obstacle_mask)

    out = g.copy()
    bottom = g[h - 1]
    for seed_c in range(w):
        color = int(bottom[seed_c])
        if color in (BLACK, GREY):
            continue
        for step in range(h):
            r = h - 1 - step
            c = seed_c + step
            if c >= w:
                break
            if not obstacle_mask[r, c]:
                out[r, c] = color

    for comp, count in zip(comps, counts):
        out[comp[:, 0], comp[:, 1]] = RECOLOR_BY_BLOCKED[count]
    return out


def _valid_instance(grid: np.ndarray, counts: list[int]) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False

    comps = _components_mask(grid == GREY)
    if len(comps) != len(counts):
        return False
    if any(not (3 <= len(comp) <= 6) for comp in comps):
        return False

    shapes = [_canonical_shape([(int(r), int(c)) for r, c in comp]) for comp in comps]
    if len(set(shapes)) != len(shapes):
        return False

    blocked = sorted(_expected_counts_from_grid(grid))
    if blocked != sorted(counts):
        return False
    if len(set(blocked)) != len(blocked):
        return False
    if any(not (1 <= count <= 4) for count in blocked):
        return False

    per_comp_seed_sets = []
    for comp in comps:
        seeds = {
            _seed_col_for_cell(int(r), int(c), h)
            for r, c in comp
            if 0 <= _seed_col_for_cell(int(r), int(c), h) < w and int(grid[h - 1, _seed_col_for_cell(int(r), int(c), h)]) != BLACK
        }
        per_comp_seed_sets.append(seeds)

    for i in range(len(per_comp_seed_sets)):
        for j in range(i + 1, len(per_comp_seed_sets)):
            if per_comp_seed_sets[i] & per_comp_seed_sets[j]:
                return False

    if sorted(len(s) for s in per_comp_seed_sets) != sorted(counts):
        return False
    if not np.array_equal(main(grid), _oracle(grid)):
        return False
    if np.array_equal(main(grid), grid):
        return False
    if len(set(int(v) for v in grid[h - 1] if int(v) != BLACK)) < 2:
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    counts = _choose_counts(rng)
    min_width = max(12, sum(counts) + len(counts) - 1 + 2)
    width = int(rng.integers(min_width, min(17, min_width + 3) + 1))
    height = int(rng.integers(max(10, len(counts) * 2 + 3), 15))

    starts = _build_seed_groups(rng, width, counts)
    if starts is None:
        return None

    sides = []
    for idx in range(len(counts)):
        if idx == 0:
            sides.append("left")
        elif idx == len(counts) - 1:
            sides.append("right")
        else:
            sides.append("left" if rng.random() < 0.5 else "right")

    row_choices = _row_options(height, width, starts, counts, sides)
    if not row_choices:
        return None
    row_choices.sort(key=sum, reverse=True)
    keep = max(1, len(row_choices) // 3)
    rows = row_choices[int(rng.integers(keep))]

    obstacle_shapes = _place_obstacles(height, starts, counts, sides, rows)
    if obstacle_shapes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    active_seed_cols: list[int] = []
    forbidden_seed_cols: set[int] = set()

    for start, count, coords in zip(starts, counts, obstacle_shapes):
        group = set(range(start, start + count))
        active_seed_cols.extend(sorted(group))
        for r, c in coords:
            grid[r, c] = GREY
            forbidden_seed_cols.add(_seed_col_for_cell(r, c, height))

    distractor_pool = _available_distractors(width, forbidden_seed_cols | set(active_seed_cols))
    rng.shuffle(distractor_pool)
    distractor_count = 0
    if distractor_pool:
        distractor_count = int(rng.choice([0, 0, 1, 1, 2]))
    distractor_cols = sorted(distractor_pool[:distractor_count])
    grid[height - 1] = _colorize_bottom(rng, width, sorted(active_seed_cols), distractor_cols)

    if not _valid_instance(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        rng = np.random.default_rng()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-sweep obstacle-counting puzzle")
