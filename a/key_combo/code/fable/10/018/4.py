from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

Coord = tuple[int, int]


def _chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _is_edge(cell: Coord, h: int, w: int) -> bool:
    r, c = map(int, cell)
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _raw_stamp_cells(center: Coord, h: int, w: int) -> tuple[Coord, ...]:
    r, c = map(int, center)
    cells: list[Coord] = []
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            cells.append((rr, cc))
    return tuple(cells)


def _stamp_cells(center: Coord, h: int, w: int, yellow_mask: np.ndarray) -> tuple[Coord, ...]:
    cells: list[Coord] = []
    for rr, cc in _raw_stamp_cells(center, h, w):
        if not bool(yellow_mask[rr, cc]):
            cells.append((rr, cc))
    return tuple(cells)


def _render_grid(h: int, w: int, centers: Iterable[Coord], yellows: Iterable[Coord]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    yellow_set = {tuple(map(int, cell)) for cell in yellows}
    yellow_mask = np.zeros((h, w), dtype=bool)
    for r, c in yellow_set:
        yellow_mask[r, c] = True
        grid[r, c] = YELLOW

    blue_mask = np.zeros((h, w), dtype=bool)
    for center in centers:
        for rr, cc in _stamp_cells(center, h, w, yellow_mask):
            blue_mask[rr, cc] = True

    grid[blue_mask] = BLUE
    grid[yellow_mask] = YELLOW
    return grid


def _build_candidate_data(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    yellow_mask = g == YELLOW
    blue_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    blue_index = {coord: idx for idx, coord in enumerate(blue_coords)}

    candidates: list[tuple[Coord, tuple[Coord, ...], int]] = []
    cell_to_candidates: list[list[int]] = [[] for _ in blue_coords]
    cell_to_candidate_mask = [0 for _ in blue_coords]

    for r in range(h):
        for c in range(w):
            cells = _stamp_cells((r, c), h, w, yellow_mask)
            if not cells:
                continue
            mask = 0
            valid = True
            for cell in cells:
                if cell not in blue_index:
                    valid = False
                    break
                mask |= 1 << blue_index[cell]
            if not valid:
                continue
            cand_id = len(candidates)
            candidates.append(((r, c), cells, mask))
            bitset = mask
            while bitset:
                lsb = bitset & -bitset
                bit = lsb.bit_length() - 1
                cell_to_candidates[bit].append(cand_id)
                cell_to_candidate_mask[bit] |= 1 << cand_id
                bitset ^= lsb

    return blue_coords, candidates, cell_to_candidates, cell_to_candidate_mask


def _pick_pivot(remaining: int, cell_to_candidates: list[list[int]]) -> int:
    best_bit = -1
    best_count = 10**9
    bitset = remaining
    while bitset:
        lsb = bitset & -bitset
        bit = lsb.bit_length() - 1
        count = len(cell_to_candidates[bit])
        if count < best_count:
            best_count = count
            best_bit = bit
            if count == 1:
                break
        bitset ^= lsb
    return best_bit


def _solve_one(candidates: list[tuple[Coord, tuple[Coord, ...], int]], cell_to_candidates: list[list[int]], n_blue: int):
    full_mask = (1 << n_blue) - 1
    memo: dict[int, tuple[int, ...] | None] = {}

    def search(remaining: int) -> tuple[int, ...] | None:
        if remaining == 0:
            return ()
        if remaining in memo:
            return memo[remaining]

        pivot = _pick_pivot(remaining, cell_to_candidates)
        if pivot < 0 or not cell_to_candidates[pivot]:
            memo[remaining] = None
            return None

        options = sorted(
            cell_to_candidates[pivot],
            key=lambda idx: (candidates[idx][2] & remaining).bit_count(),
            reverse=True,
        )
        for idx in options:
            next_remaining = remaining & ~candidates[idx][2]
            suffix = search(next_remaining)
            if suffix is not None:
                memo[remaining] = (idx,) + suffix
                return memo[remaining]

        memo[remaining] = None
        return None

    return search(full_mask)


def _count_solutions_exact(
    candidate_masks: list[int],
    cell_to_candidate_mask: list[int],
    n_blue: int,
    limit: int = 2,
) -> int:
    undecided_all = (1 << len(candidate_masks)) - 1
    memo: dict[tuple[int, int], int] = {}

    def rec(remaining: int, undecided: int) -> int:
        if remaining == 0:
            return 1 if undecided == 0 else limit
        key = (remaining, undecided)
        if key in memo:
            return memo[key]

        pivot_options = 0
        best_count = 10**9
        bitset = remaining
        while bitset:
            lsb = bitset & -bitset
            bit = lsb.bit_length() - 1
            options = undecided & cell_to_candidate_mask[bit]
            count = options.bit_count()
            if count == 0:
                memo[key] = 0
                return 0
            if count < best_count:
                best_count = count
                pivot_options = options
                if count == 1:
                    break
            bitset ^= lsb

        chosen_lsb = pivot_options & -pivot_options
        cand = chosen_lsb.bit_length() - 1

        total = rec(remaining & ~candidate_masks[cand], undecided & ~chosen_lsb)
        if total < limit:
            total += rec(remaining, undecided & ~chosen_lsb)
        total = min(total, limit)
        memo[key] = total
        return total

    return rec((1 << n_blue) - 1, undecided_all)


def _blue_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse a blue stamp trace into the unique set of 3x3 stamp centers and mark those centers red.

    Input structure:
    - A black background contains scattered yellow pixels and one or more blue regions.
    - The blue cells are exactly the union of several solid 3x3 stamps after clipping at the grid border.
    - Yellow pixels are never painted by a stamp, so they can puncture holes inside the blue trace and may even sit on a stamp center.

    Rule:
    - Find the unique set of stamp centers whose clipped 3x3 neighborhoods, after skipping every yellow cell, cover all and only the blue cells.
    - Output the original grid with each recovered center recolored red.

    Key insight:
    - The blue shape is not a region to extend or fill; it is the visible residue of overlapping clipped stamps, so the centers must be recovered by exact causal inversion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, YELLOW}

    blue_coords, candidates, cell_to_candidates, _cell_to_candidate_mask = _build_candidate_data(g)
    assert blue_coords, "expected at least one blue cell"
    assert all(cell_to_candidates), "every blue cell must be coverable by some candidate stamp center"

    solution = _solve_one(candidates, cell_to_candidates, len(blue_coords))
    assert solution is not None, "expected a valid stamp decomposition"

    out = g.copy()
    for idx in solution:
        r, c = candidates[idx][0]
        out[r, c] = RED
    return out


# --------------- generator ---------------

HEIGHT_RANGE = (8, 16)
WIDTH_RANGE = (8, 16)
CENTER_COUNT_RANGE = (2, 5)
MAX_CANDIDATES_FOR_EXACT_CHECK = 17
MAX_GENERATION_ATTEMPTS = 600
LOCAL_PLACEMENT_ATTEMPTS = 120
EXTRA_CANDIDATE_PROB = 0.6


def _sample_centers(
    h: int,
    w: int,
    count: int,
    require_overlap: bool,
    require_edge: bool,
    require_multi_component: bool,
) -> tuple[Coord, ...] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    edge_cells = [cell for cell in cells if _is_edge(cell, h, w)]

    for _ in range(LOCAL_PLACEMENT_ATTEMPTS):
        centers: list[Coord] = []
        used: set[Coord] = set()
        overlap_done = False
        edge_done = False
        multi_done = not require_multi_component

        first_pool = edge_cells if require_edge and edge_cells and random.random() < 0.75 else cells
        first = random.choice(first_pool)
        centers.append(first)
        used.add(first)
        edge_done = _is_edge(first, h, w)

        if require_overlap:
            near = [cell for cell in cells if cell not in used and 1 <= _chebyshev(cell, first) <= 2]
            if not near:
                continue
            second = random.choice(near)
            centers.append(second)
            used.add(second)
            overlap_done = True
            edge_done = edge_done or _is_edge(second, h, w)

        while len(centers) < count:
            remaining = [cell for cell in cells if cell not in used]
            if not remaining:
                break

            pool = remaining
            if require_edge and not edge_done:
                forced_edge = [cell for cell in remaining if _is_edge(cell, h, w)]
                if forced_edge and random.random() < 0.85:
                    pool = forced_edge
            elif require_multi_component and not multi_done:
                far = [cell for cell in remaining if min(_chebyshev(cell, other) for other in centers) >= 4]
                if far:
                    pool = far
            elif require_overlap and not overlap_done:
                near = [cell for cell in remaining if any(1 <= _chebyshev(cell, other) <= 2 for other in centers)]
                if near:
                    pool = near
            elif not require_multi_component and random.random() < 0.7:
                clustered = [cell for cell in remaining if any(_chebyshev(cell, other) <= 4 for other in centers)]
                if clustered:
                    pool = clustered

            chosen = random.choice(pool)
            centers.append(chosen)
            used.add(chosen)
            edge_done = edge_done or _is_edge(chosen, h, w)
            overlap_done = overlap_done or any(1 <= _chebyshev(chosen, other) <= 2 for other in centers[:-1])
            if require_multi_component and not multi_done:
                multi_done = min(_chebyshev(chosen, other) for other in centers[:-1]) >= 4

        if len(centers) != count:
            continue
        if require_edge and not edge_done:
            continue
        if require_overlap and not overlap_done:
            continue
        if require_multi_component and not multi_done:
            continue
        return tuple(sorted(centers))

    return None


def _sample_yellows(
    h: int,
    w: int,
    centers: tuple[Coord, ...],
    require_hole: bool,
    require_center_hole: bool,
) -> set[Coord] | None:
    raw_union: set[Coord] = set()
    raw_cover_count: dict[Coord, int] = {}
    for center in centers:
        for cell in _raw_stamp_cells(center, h, w):
            raw_union.add(cell)
            raw_cover_count[cell] = raw_cover_count.get(cell, 0) + 1

    if not raw_union:
        return None

    all_cells = [(r, c) for r in range(h) for c in range(w)]
    nearby_outside = [
        cell
        for cell in all_cells
        if cell not in raw_union and any(_chebyshev(cell, cover) <= 2 for cover in raw_union)
    ]

    for _ in range(LOCAL_PLACEMENT_ATTEMPTS):
        yellows: set[Coord] = set()

        if require_center_hole:
            center_choices = [center for center in centers if center in raw_union]
            if not center_choices:
                continue
            yellows.add(random.choice(center_choices))

        hole_min = 1 if require_hole else 0
        hole_max = min(4, len(raw_union))
        target_holes = random.randint(hole_min, hole_max)
        preferred_holes = [
            cell for cell in raw_union if raw_cover_count.get(cell, 0) >= 2 and cell not in yellows
        ]
        ordinary_holes = [cell for cell in raw_union if cell not in yellows]
        random.shuffle(preferred_holes)
        random.shuffle(ordinary_holes)
        for pool in (preferred_holes, ordinary_holes):
            for cell in pool:
                if len([y for y in yellows if y in raw_union]) >= target_holes:
                    break
                yellows.add(cell)
            if len([y for y in yellows if y in raw_union]) >= target_holes:
                break

        if require_hole and not any(cell in raw_union for cell in yellows):
            continue

        extra_count = random.randint(0, 3)
        random.shuffle(nearby_outside)
        for cell in nearby_outside[:extra_count]:
            yellows.add(cell)

        if yellows:
            return yellows

    return None


def _validate_instance(
    grid: np.ndarray,
    actual_centers: tuple[Coord, ...],
    *,
    require_overlap: bool,
    require_edge: bool,
    require_hole: bool,
    require_multi_component: bool,
    require_extra_candidates: bool,
) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False

    blue_mask = grid == BLUE
    yellow_mask = grid == YELLOW
    blue_count = int(np.count_nonzero(blue_mask))
    if blue_count < 6 or blue_count > 70:
        return False
    if int(np.count_nonzero(yellow_mask)) == 0:
        return False

    blue_coords, candidates, cell_to_candidates, cell_to_candidate_mask = _build_candidate_data(grid)
    if not blue_coords or len(candidates) > MAX_CANDIDATES_FOR_EXACT_CHECK:
        return False
    if any(len(lst) == 0 for lst in cell_to_candidates):
        return False
    if require_extra_candidates and len(candidates) <= len(actual_centers):
        return False

    actual_set = set(actual_centers)
    candidate_centers = {center for center, _cells, _mask in candidates}
    if not actual_set <= candidate_centers:
        return False

    solution = _solve_one(candidates, cell_to_candidates, len(blue_coords))
    if solution is None:
        return False
    solved_centers = {candidates[idx][0] for idx in solution}
    if solved_centers != actual_set:
        return False

    solution_count = _count_solutions_exact(
        [mask for _center, _cells, mask in candidates],
        cell_to_candidate_mask,
        len(blue_coords),
        limit=2,
    )
    if solution_count != 1:
        return False

    per_center_cells = {center: _stamp_cells(center, h, w, yellow_mask) for center in actual_centers}
    if any(len(cells) < 2 for cells in per_center_cells.values()):
        return False

    coverage_count: dict[Coord, int] = {}
    for cells in per_center_cells.values():
        for cell in cells:
            coverage_count[cell] = coverage_count.get(cell, 0) + 1
    if any(not any(coverage_count[cell] == 1 for cell in cells) for cells in per_center_cells.values()):
        return False

    if require_overlap:
        if not any(coverage_count[cell] >= 2 for cell in coverage_count):
            return False
    if require_edge and not any(_is_edge(center, h, w) for center in actual_centers):
        return False
    if require_hole:
        if not any(any(yellow_mask[cell] for cell in _raw_stamp_cells(center, h, w)) for center in actual_centers):
            return False
    if require_multi_component and _blue_components(blue_mask) < 2:
        return False

    for center in actual_centers:
        visible_cells = per_center_cells[center]
        if any(int(grid[r, c]) != BLUE for r, c in visible_cells):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    center_count = random.randint(*CENTER_COUNT_RANGE)

    require_overlap = random.random() < 0.75
    require_edge = random.random() < 0.6
    require_hole = random.random() < 0.8
    require_center_hole = require_hole and random.random() < 0.5
    require_multi_component = center_count >= 3 and random.random() < 0.35
    require_extra_candidates = random.random() < EXTRA_CANDIDATE_PROB

    centers = _sample_centers(
        h,
        w,
        center_count,
        require_overlap=require_overlap,
        require_edge=require_edge,
        require_multi_component=require_multi_component,
    )
    if centers is None:
        return None

    yellows = _sample_yellows(
        h,
        w,
        centers,
        require_hole=require_hole,
        require_center_hole=require_center_hole,
    )
    if yellows is None:
        return None

    grid = _render_grid(h, w, centers, yellows)
    if not _validate_instance(
        grid,
        centers,
        require_overlap=require_overlap,
        require_edge=require_edge,
        require_hole=require_hole,
        require_multi_component=require_multi_component,
        require_extra_candidates=require_extra_candidates,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-trace inversion input")
