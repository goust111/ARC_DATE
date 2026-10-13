from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5

_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for dr, dc in _NEIGHBORS4:
                nr = cr + dr
                nc = cc2 + dc
                if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(sorted(comp))
    return comps


def _detect_diagonal(cells: list[tuple[int, int]]) -> tuple[str, int]:
    diffs = {r - c for r, c in cells}
    sums = {r + c for r, c in cells}
    if len(diffs) == 1:
        return "main", next(iter(diffs))
    if len(sums) == 1:
        return "anti", next(iter(sums))
    raise ValueError("diagonal cells are not aligned on a single diagonal")


def _distance_to_cells(cell: tuple[int, int], targets: list[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(r - tr) + abs(c - tc) for tr, tc in targets)


def _tip_for_component(
    comp: list[tuple[int, int]], red_cells: list[tuple[int, int]]
) -> tuple[tuple[int, int], int]:
    distances = [(cell, _distance_to_cells(cell, red_cells)) for cell in comp]
    max_dist = max(dist for _cell, dist in distances)
    tip = min(cell for cell, dist in distances if dist == max_dist)
    return tip, max_dist


def _draw_full_diagonal(grid: np.ndarray, start: tuple[int, int], orientation: str, color: int) -> None:
    if orientation == "main":
        directions = [(-1, -1), (1, 1)]
    else:
        directions = [(-1, 1), (1, -1)]
    sr, sc = start
    grid[sr, sc] = color
    for dr, dc in directions:
        r = sr + dr
        c = sc + dc
        while _in_bounds(grid.shape, r, c):
            grid[r, c] = color
            r += dr
            c += dc


def _diag_invariant(cell: tuple[int, int], orientation: str) -> int:
    r, c = cell
    return r - c if orientation == "main" else r + c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend parallel red diagonals from the far tips of only the large gray guide objects, then erase every gray pixel.

    Input structure:
    - The grid has a black background, one red diagonal segment, and several gray 4-connected objects.
    - Gray objects have mixed areas: some are small noise components of area 1-2, while large guide components have area 4-8.
    - Each large guide has a unique pixel farthest from the original red diagonal under Manhattan distance.

    Rule:
    - Detect all gray 4-connected components and classify them by area.
    - Ignore components of area 1-2; for each component of area at least 4, find the unique farthest pixel from the original red diagonal, breaking ties by smallest row then smallest column.
    - Determine whether the original red diagonal follows the main diagonal direction or the anti-diagonal direction, then draw a full red diagonal through each large-component tip in that same direction until the grid edge.
    - Keep the original red diagonal unchanged and erase all gray pixels to black.

    Key insight:
    - Area is the role selector: only large gray components act as guides, and their farthest tips seed the new parallel diagonals while small gray noise must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    diagonal_colors = sorted(set(map(int, g.ravel())) - {BLACK, GREY})
    if len(diagonal_colors) != 1:
        raise ValueError("expected exactly one non-gray foreground color for the source diagonal")
    diagonal_color = diagonal_colors[0]

    red_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == diagonal_color)]
    orientation, _ = _detect_diagonal(red_cells)

    grey_components = _find_components(g, GREY)
    for comp in grey_components:
        if len(comp) < 4:
            continue
        tip, _ = _tip_for_component(comp, red_cells)
        _draw_full_diagonal(out, tip, orientation, diagonal_color)

    out[g == GREY] = BLACK
    return out


def _line_coords(shape: tuple[int, int], orientation: str, invariant: int) -> list[tuple[int, int]]:
    h, w = shape
    coords: list[tuple[int, int]] = []
    if orientation == "main":
        for r in range(h):
            c = r - invariant
            if 0 <= c < w:
                coords.append((r, c))
    else:
        for r in range(h):
            c = invariant - r
            if 0 <= c < w:
                coords.append((r, c))
    return coords


def _line_length(shape: tuple[int, int], orientation: str, invariant: int) -> int:
    return len(_line_coords(shape, orientation, invariant))


def _distance_to_line(cell: tuple[int, int], orientation: str, invariant: int) -> int:
    return abs(_diag_invariant(cell, orientation) - invariant)


def _toward_line_moves(orientation: str, side: int) -> list[tuple[int, int]]:
    if orientation == "main":
        return [(-1, 0), (0, 1)] if side > 0 else [(1, 0), (0, -1)]
    return [(-1, 0), (0, -1)] if side > 0 else [(1, 0), (0, 1)]


def _bbox_ok(cells: set[tuple[int, int]], limit: int = 5) -> bool:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return max(rows) - min(rows) + 1 <= limit and max(cols) - min(cols) + 1 <= limit


def _separated_from_existing(cells: set[tuple[int, int]], grey_cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in grey_cells:
            return False
        for dr, dc in _NEIGHBORS8:
            if (r + dr, c + dc) in grey_cells:
                return False
    return True


def _build_guide(
    shape: tuple[int, int],
    orientation: str,
    line_invariant: int,
    offset: int,
    tip: tuple[int, int],
    area: int,
) -> set[tuple[int, int]] | None:
    side = 1 if offset > 0 else -1
    tip_dist = abs(offset)
    moves = _toward_line_moves(orientation, side)

    cells: set[tuple[int, int]] = {tip}
    path = [tip]
    current = tip

    for _ in range(tip_dist - 1):
        options = moves[:]
        random.shuffle(options)
        next_cell = None
        for dr, dc in options:
            cand = (current[0] + dr, current[1] + dc)
            if not _in_bounds(shape, cand[0], cand[1]):
                continue
            if cand in cells:
                continue
            if _distance_to_line(cand, orientation, line_invariant) != _distance_to_line(
                current, orientation, line_invariant
            ) - 1:
                continue
            next_cell = cand
            break
        if next_cell is None:
            return None
        cells.add(next_cell)
        path.append(next_cell)
        current = next_cell

    while len(cells) < area:
        candidates: list[tuple[int, int]] = []
        for cell in sorted(cells, key=lambda rc: (-_distance_to_line(rc, orientation, line_invariant), rc)):
            cell_dist = _distance_to_line(cell, orientation, line_invariant)
            if cell_dist <= 1:
                continue
            for dr, dc in moves:
                cand = (cell[0] + dr, cell[1] + dc)
                if not _in_bounds(shape, cand[0], cand[1]):
                    continue
                if cand in cells:
                    continue
                if _distance_to_line(cand, orientation, line_invariant) != cell_dist - 1:
                    continue
                trial = set(cells)
                trial.add(cand)
                if _bbox_ok(trial):
                    candidates.append(cand)
        if not candidates:
            return None
        candidates = sorted(set(candidates), key=lambda rc: (-_distance_to_line(rc, orientation, line_invariant), rc))
        chosen = random.choice(candidates[: min(4, len(candidates))])
        cells.add(chosen)

    if len({r for r, _c in cells}) < 2 or len({c for _r, c in cells}) < 2:
        return None
    if not any(_distance_to_line(cell, orientation, line_invariant) == 1 for cell in cells):
        return None
    return cells


def _build_noise(
    shape: tuple[int, int],
    orientation: str,
    line_invariant: int,
    offset: int,
    tip: tuple[int, int],
    size: int,
) -> set[tuple[int, int]] | None:
    cells = {tip}
    if size == 1:
        return cells
    if abs(offset) < 2:
        return None
    side = 1 if offset > 0 else -1
    moves = _toward_line_moves(orientation, side)
    options = moves[:]
    random.shuffle(options)
    for dr, dc in options:
        cand = (tip[0] + dr, tip[1] + dc)
        if not _in_bounds(shape, cand[0], cand[1]):
            continue
        if _distance_to_line(cand, orientation, line_invariant) != abs(offset) - 1:
            continue
        cells.add(cand)
        return cells
    return None


def _place_component(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _erase_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c in comp:
        out[r, c] = BLACK
    return out


def _object_line_key(comp: list[tuple[int, int]], red_cells: list[tuple[int, int]], orientation: str) -> tuple[int, int]:
    tip, dist = _tip_for_component(comp, red_cells)
    return _diag_invariant(tip, orientation), dist


def _validate_generated_instance(grid: np.ndarray) -> None:
    if grid.ndim != 2 or max(grid.shape) > 30:
        raise AssertionError(f"invalid shape {grid.shape}")

    diagonal_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    if len(diagonal_cells) < 6:
        raise AssertionError("source diagonal is too short")
    orientation, source_key = _detect_diagonal(diagonal_cells)

    comps = _find_components(grid, GREY)
    if not (4 <= len(comps) <= 6):
        raise AssertionError(f"unexpected component count {len(comps)}")
    areas = [len(comp) for comp in comps]
    if any(area == 3 or area > 8 for area in areas):
        raise AssertionError(f"invalid grey areas {areas}")

    guides = [comp for comp in comps if len(comp) >= 4]
    noises = [comp for comp in comps if len(comp) <= 2]
    if len(guides) < 2 or len(noises) < 2:
        raise AssertionError("must have at least two guides and two noise components")

    guide_keys: set[int] = set()
    guide_distances: list[int] = []
    for comp in guides:
        tip, max_dist = _tip_for_component(comp, diagonal_cells)
        if sum(_distance_to_cells(cell, diagonal_cells) == max_dist for cell in comp) != 1:
            raise AssertionError("guide tip must be unique")
        line_key = _diag_invariant(tip, orientation)
        if line_key == source_key or line_key in guide_keys:
            raise AssertionError("guide line keys must be distinct from the source and from each other")
        guide_keys.add(line_key)
        guide_distances.append(max_dist)

    noise_info = []
    for comp in noises:
        tip, dist = _tip_for_component(comp, diagonal_cells)
        noise_info.append((_diag_invariant(tip, orientation), dist))

    has_near_counterexample = any(key not in guide_keys and key != source_key and dist <= 2 for key, dist in noise_info)
    has_far_counterexample = any(
        key not in guide_keys and key != source_key and dist > min(guide_distances) for key, dist in noise_info
    )
    if not has_near_counterexample or not has_far_counterexample:
        raise AssertionError("noise must include both near-line and farther-than-guide counterexamples")

    out = main(grid)
    if np.any(out == GREY):
        raise AssertionError("output must erase all grey pixels")

    for comp in guides:
        pruned = _erase_component(grid, comp)
        if np.array_equal(main(pruned), out):
            raise AssertionError("every guide must be semantically necessary")


def _candidate_source_invariants(shape: tuple[int, int], orientation: str) -> list[int]:
    h, w = shape
    if orientation == "main":
        values = range(-(w - 1), h)
    else:
        values = range(h + w - 1)

    candidates = []
    for invariant in values:
        if _line_length(shape, orientation, invariant) < 6:
            continue
        guide_offsets = [
            off
            for off in (-5, -4, -3, 3, 4, 5)
            if _line_length(shape, orientation, invariant + off) >= 4
        ]
        near_offsets = [off for off in (-2, -1, 1, 2) if _line_length(shape, orientation, invariant + off) >= 2]
        far_noise_offsets = [
            off
            for off in (-5, -4, 4, 5)
            if _line_length(shape, orientation, invariant + off) >= 2
        ]
        if len(guide_offsets) >= 2 and near_offsets and far_noise_offsets:
            candidates.append(invariant)
    return candidates


def generate_input() -> np.ndarray:
    for _attempt in range(500):
        h = random.randint(13, 18)
        w = random.randint(13, 18)
        orientation = random.choice(["main", "anti"])
        shape = (h, w)

        source_candidates = _candidate_source_invariants(shape, orientation)
        if not source_candidates:
            continue
        source_invariant = random.choice(source_candidates)

        grid = np.zeros(shape, dtype=int)
        for r, c in _line_coords(shape, orientation, source_invariant):
            grid[r, c] = RED

        grey_cells: set[tuple[int, int]] = set()
        used_offsets: set[int] = set()

        guide_offsets = [
            off
            for off in (-5, -4, -3, 3, 4, 5)
            if _line_length(shape, orientation, source_invariant + off) >= 4
        ]
        near_noise_offsets = [
            off for off in (-2, -1, 1, 2) if _line_length(shape, orientation, source_invariant + off) >= 2
        ]
        far_noise_offsets = [
            off
            for off in (-5, -4, 4, 5)
            if _line_length(shape, orientation, source_invariant + off) >= 2
        ]

        guide_count = random.randint(2, min(3, len(guide_offsets)))
        chosen_guide_offsets = random.sample(guide_offsets, guide_count)
        random.shuffle(chosen_guide_offsets)

        placed = True
        for off in chosen_guide_offsets:
            invariant = source_invariant + off
            tip_candidates = _line_coords(shape, orientation, invariant)
            random.shuffle(tip_candidates)
            guide_cells = None
            for tip in tip_candidates:
                area = random.randint(max(4, abs(off) + 1), min(8, abs(off) + 3))
                trial = _build_guide(shape, orientation, source_invariant, off, tip, area)
                if trial is None:
                    continue
                if not _separated_from_existing(trial, grey_cells):
                    continue
                guide_cells = trial
                break
            if guide_cells is None:
                placed = False
                break
            _place_component(grid, guide_cells, GREY)
            grey_cells |= guide_cells
            used_offsets.add(off)
        if not placed:
            continue

        near_offset_choices = [off for off in near_noise_offsets if off not in used_offsets]
        far_offset_choices = [off for off in far_noise_offsets if off not in used_offsets]
        if not near_offset_choices or not far_offset_choices:
            continue

        noise_specs: list[tuple[int, int]] = []
        near_off = random.choice(near_offset_choices)
        near_size = 1 if abs(near_off) == 1 else random.choice([1, 2])
        noise_specs.append((near_off, near_size))
        used_offsets.add(near_off)

        far_offset_choices = [off for off in far_offset_choices if off not in used_offsets]
        if not far_offset_choices:
            continue
        far_off = random.choice(far_offset_choices)
        noise_specs.append((far_off, random.choice([1, 2])))
        used_offsets.add(far_off)

        remaining_offsets = [
            off
            for off in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
            if off not in used_offsets and _line_length(shape, orientation, source_invariant + off) >= 2
        ]
        extra_noise_count = random.randint(0, min(1, len(remaining_offsets)))
        for _ in range(extra_noise_count):
            extra_off = random.choice(remaining_offsets)
            remaining_offsets.remove(extra_off)
            extra_size = 1 if abs(extra_off) == 1 else random.choice([1, 2])
            noise_specs.append((extra_off, extra_size))

        for off, size in noise_specs:
            invariant = source_invariant + off
            tip_candidates = _line_coords(shape, orientation, invariant)
            random.shuffle(tip_candidates)
            noise_cells = None
            for tip in tip_candidates:
                trial = _build_noise(shape, orientation, source_invariant, off, tip, size)
                if trial is None:
                    continue
                if not _separated_from_existing(trial, grey_cells):
                    continue
                noise_cells = trial
                break
            if noise_cells is None:
                placed = False
                break
            _place_component(grid, noise_cells, GREY)
            grey_cells |= noise_cells
        if not placed:
            continue

        try:
            _validate_generated_instance(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
