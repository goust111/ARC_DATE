from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _separator_row(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    rows = [r for r in range(height) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one grey separator row"
    row = rows[0]
    assert 2 * row + 1 == height, "separator must split the grid into equal halves"
    assert width >= 1
    return row


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    height, width = map(int, mask.shape)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _grow_shared_mask(height: int, width: int, size: int) -> np.ndarray | None:
    if height < 5 or width < 7:
        return None

    rows = range(1, height - 1)
    cols = range(2, width - 2)
    if not rows or not cols:
        return None

    start = (random.choice(tuple(rows)), random.choice(tuple(cols)))
    cells = {start}

    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if nr in rows and nc in cols and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    row_span = len({r for r, _ in cells})
    col_span = len({c for _, c in cells})
    if row_span < 3 or col_span < 4:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = map(int, mask.shape)
    frontier = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 1 <= nr < height - 1 and 1 <= nc < width - 1 and not mask[nr, nc]:
                frontier.add((nr, nc))
    return list(frontier)


def _choose_targets(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    def backtrack(start: int) -> bool:
        if len(chosen) == count:
            return True
        if len(shuffled) - start < count - len(chosen):
            return False

        for idx in range(start, len(shuffled)):
            r, c = shuffled[idx]
            if r in used_rows or c in used_cols:
                continue
            chosen.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
            if backtrack(idx + 1):
                return True
            chosen.pop()
            used_rows.remove(r)
            used_cols.remove(c)
        return False

    return chosen[:] if backtrack(0) else None


def _is_isolated_from_red(mask: np.ndarray, r: int, c: int) -> bool:
    height, width = map(int, mask.shape)
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
            return False
    return True


def _marker_candidates(
    lower_mask: np.ndarray, targets: list[tuple[int, int]], target: tuple[int, int]
) -> list[tuple[int, int, str, frozenset[tuple[int, int]], tuple[int, int]]]:
    height, width = map(int, lower_mask.shape)
    tr, tc = target
    other_rows = {r for r, _ in targets if r != tr}
    other_cols = {c for _, c in targets if c != tc}
    candidates: list[tuple[int, int, str, frozenset[tuple[int, int]], tuple[int, int]]] = []

    for mc in range(tc - 1, -1, -1):
        if lower_mask[tr, mc]:
            break
        if tc - mc >= 2 and mc not in other_cols and _is_isolated_from_red(lower_mask, tr, mc):
            path = frozenset((tr, cc) for cc in range(mc + 1, tc))
            candidates.append((tr, mc, "h", path, (tr, tc - 1)))

    for mc in range(tc + 1, width):
        if lower_mask[tr, mc]:
            break
        if mc - tc >= 2 and mc not in other_cols and _is_isolated_from_red(lower_mask, tr, mc):
            path = frozenset((tr, cc) for cc in range(tc + 1, mc))
            candidates.append((tr, mc, "h", path, (tr, tc + 1)))

    for mr in range(tr - 1, -1, -1):
        if lower_mask[mr, tc]:
            break
        if tr - mr >= 2 and mr not in other_rows and _is_isolated_from_red(lower_mask, mr, tc):
            path = frozenset((rr, tc) for rr in range(mr + 1, tr))
            candidates.append((mr, tc, "v", path, (tr - 1, tc)))

    for mr in range(tr + 1, height):
        if lower_mask[mr, tc]:
            break
        if mr - tr >= 2 and mr not in other_rows and _is_isolated_from_red(lower_mask, mr, tc):
            path = frozenset((rr, tc) for rr in range(tr + 1, mr))
            candidates.append((mr, tc, "v", path, (tr + 1, tc)))

    random.shuffle(candidates)
    return candidates


def _place_markers(
    lower_mask: np.ndarray, targets: list[tuple[int, int]]
) -> dict[tuple[int, int], tuple[int, int]] | None:
    candidate_lists = [_marker_candidates(lower_mask, targets, target) for target in targets]
    if any(not options for options in candidate_lists):
        return None

    order = sorted(range(len(targets)), key=lambda idx: len(candidate_lists[idx]))
    all_axes = {axis for options in candidate_lists for _, _, axis, _, _ in options}
    target_set = set(targets)
    assigned: dict[int, tuple[int, int, str, frozenset[tuple[int, int]], tuple[int, int]]] = {}
    used_markers: set[tuple[int, int]] = set()
    used_landings: set[tuple[int, int]] = set()

    def adjacent_to_marker(marker: tuple[int, int]) -> bool:
        mr, mc = marker
        return any(abs(mr - rr) + abs(mc - cc) == 1 for rr, cc in used_markers)

    def marker_on_existing_path(marker: tuple[int, int]) -> bool:
        return any(marker in path for _, _, _, path, _ in assigned.values())

    def path_hits_existing_marker(path: frozenset[tuple[int, int]]) -> bool:
        return not path.isdisjoint(used_markers)

    def backtrack(depth: int) -> bool:
        if depth == len(order):
            used_axes = {axis for _, _, axis, _, _ in assigned.values()}
            if len(targets) > 1 and len(all_axes) == 2 and len(used_axes) < 2:
                return False
            return True

        idx = order[depth]
        options = candidate_lists[idx][:]
        random.shuffle(options)
        for mr, mc, axis, path, landing in options:
            marker = (mr, mc)
            if marker in used_markers or landing in used_landings or landing in target_set:
                continue
            if adjacent_to_marker(marker) or marker_on_existing_path(marker) or path_hits_existing_marker(path):
                continue
            assigned[idx] = (mr, mc, axis, path, landing)
            used_markers.add(marker)
            used_landings.add(landing)
            if backtrack(depth + 1):
                return True
            used_markers.remove(marker)
            used_landings.remove(landing)
            del assigned[idx]
        return False

    if not backtrack(0):
        return None
    return {targets[idx]: (assigned[idx][0], assigned[idx][1]) for idx in range(len(targets))}


def _assemble_input(
    upper_mask: np.ndarray,
    lower_mask: np.ndarray,
    markers: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    height, width = map(int, upper_mask.shape)
    grid = np.zeros((2 * height + 1, width), dtype=int)
    upper = grid[:height, :]
    upper[upper_mask] = BLUE
    grid[height, :] = GREY
    lower = grid[height + 1 :, :]
    lower[lower_mask] = RED
    for mr, mc in markers.values():
        assert not lower_mask[mr, mc], "marker must sit outside the red mask"
        grid[height + 1 + mr, mc] = YELLOW
    return grid


def _assemble_output(
    shape: tuple[int, int],
    targets: list[tuple[int, int]],
    markers: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for tr, tc in targets:
        out[tr, tc] = GREEN
    for tr, tc in targets:
        mr, mc = markers[(tr, tc)]
        dr = 0 if mr == tr else (1 if mr > tr else -1)
        dc = 0 if mc == tc else (1 if mc > tc else -1)
        landing = (tr + dr, tc + dc)
        assert out[landing] == BG, "yellow markers cannot overwrite green or each other"
        out[landing] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compute the xor disagreement field of two separator-stacked masks, then slide each yellow marker next to its unique green target.

    Input structure:
    - The grid has a black background and one full-width grey separator row splitting two equal-height halves.
    - The upper half contains one blue 4-connected mask; the lower half contains one red 4-connected mask plus several isolated yellow markers.
    - Each yellow marker shares exactly one row or one column with a unique xor-disagreement cell of the aligned blue/red masks.

    Rule:
    - Remove the grey separator row and align the upper and lower halves cell by cell.
    - Paint green wherever exactly one of the aligned cells is foreground, using the blue mask above and the red mask below.
    - For each yellow marker, find its unique aligned green cell and move the marker along that row or column to the cell immediately adjacent to the green target.

    Key insight:
    - The yellow markers do not point to fixed anchors; their destinations are created dynamically by first computing the xor of the two separated masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g)
    upper = g[:sep, :]
    lower = g[sep + 1 :, :]
    assert upper.shape == lower.shape, "upper and lower halves must match"

    xor_mask = (upper == BLUE) ^ (lower == RED)
    targets = [tuple(map(int, coord)) for coord in np.argwhere(xor_mask)]
    markers = [tuple(map(int, coord)) for coord in np.argwhere(lower == YELLOW)]
    assert targets, "expected at least one xor target"
    assert len(markers) == len(targets), "expected one yellow marker per xor target"

    out = np.zeros_like(upper, dtype=int)
    out[xor_mask] = GREEN
    used_landings: set[tuple[int, int]] = set()

    for mr, mc in markers:
        aligned = [(tr, tc) for tr, tc in targets if (tr == mr) ^ (tc == mc)]
        assert len(aligned) == 1, "each marker must align with exactly one xor target"
        tr, tc = aligned[0]

        if tr == mr:
            c0, c1 = sorted((mc, tc))
            assert not np.any(lower[mr, c0 + 1 : c1] != BG), "horizontal marker path must be empty"
        else:
            r0, r1 = sorted((mr, tr))
            assert not np.any(lower[r0 + 1 : r1, mc] != BG), "vertical marker path must be empty"

        dr = 0 if mr == tr else (1 if mr > tr else -1)
        dc = 0 if mc == tc else (1 if mc > tc else -1)
        landing = (tr + dr, tc + dc)
        lr, lc = landing
        assert 0 <= lr < out.shape[0] and 0 <= lc < out.shape[1]
        assert landing not in used_landings, "two markers cannot land on the same cell"
        assert out[landing] == BG, "landing cell must be empty"
        used_landings.add(landing)
        out[landing] = YELLOW

    return out


def _generate_attempt() -> np.ndarray | None:
    target_count = random.randint(3, 6)
    height = random.randint(max(7, target_count + 2), 12)
    width = random.randint(max(14, target_count + 8), 24)

    upper_only = random.randint(1, target_count - 1)
    lower_only = target_count - upper_only
    base_min = max(8 - upper_only, 8 - lower_only, 6)
    base_max = min(15 - upper_only, 15 - lower_only, 10)
    if base_min > base_max:
        return None

    base_size = random.randint(base_min, base_max)
    shared = _grow_shared_mask(height, width, base_size)
    if shared is None:
        return None

    frontier = _frontier_cells(shared)
    if len(frontier) < target_count:
        return None

    targets = _choose_targets(frontier, target_count)
    if targets is None:
        return None
    random.shuffle(targets)

    upper_mask = shared.copy()
    lower_mask = shared.copy()
    for r, c in targets[:upper_only]:
        upper_mask[r, c] = True
    for r, c in targets[upper_only:]:
        lower_mask[r, c] = True

    if not _is_connected(upper_mask) or not _is_connected(lower_mask):
        return None
    if not (8 <= int(upper_mask.sum()) <= 15 and 8 <= int(lower_mask.sum()) <= 15):
        return None

    markers = _place_markers(lower_mask, targets)
    if markers is None:
        return None

    grid = _assemble_input(upper_mask, lower_mask, markers)
    expected = _assemble_output((height, width), targets, markers)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
