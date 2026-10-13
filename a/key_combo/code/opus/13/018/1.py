import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
MAGENTA = 6


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(connectivity) == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _classify_walls(grid: np.ndarray) -> tuple[set[tuple[int, int]], list[np.ndarray]]:
    h, w = grid.shape
    structural: set[tuple[int, int]] = set()
    decorative: list[np.ndarray] = []
    for comp in components_mask(grid == BLUE, connectivity=4):
        touches_border = bool(
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == h - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == w - 1)
        )
        if touches_border:
            structural |= {tuple(map(int, rc)) for rc in comp}
        else:
            decorative.append(comp)
    return structural, decorative


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MAGENTA)
    assert coords.shape[0] == 1, "expected exactly one marker"
    r, c = map(int, coords[0])
    assert (r, c) in {
        (0, 0),
        (0, grid.shape[1] - 1),
        (grid.shape[0] - 1, 0),
        (grid.shape[0] - 1, grid.shape[1] - 1),
    }, "marker must be on a corner"
    return r, c


def _reachable_mask(grid: np.ndarray, structural: set[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    start = _find_marker(grid)
    seen = np.zeros((h, w), dtype=bool)
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if seen[nr, nc] or (nr, nc) in structural:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return seen


def _visible_height(grid: np.ndarray) -> int:
    red_rows = np.flatnonzero(np.any(grid == RED, axis=1))
    assert red_rows.size > 0, "expected visible pattern rows"
    return int(red_rows.max()) + 1


def _shift_row_mask(mask: np.ndarray, shift: int) -> np.ndarray:
    width = mask.size
    out = np.zeros(width, dtype=int)
    cols = np.flatnonzero(mask)
    dst = cols + int(shift)
    keep = (0 <= dst) & (dst < width)
    out[dst[keep]] = 1
    return out


def _observed_pattern_rows(grid: np.ndarray, count: int) -> np.ndarray:
    rows = np.zeros((count, grid.shape[1]), dtype=int)
    rows.fill(0)
    rows[grid[:count] == BLUE] = -1
    rows[grid[:count] == RED] = 1
    return rows


def _pair_compatible(row_a: np.ndarray, row_b: np.ndarray, shift: int) -> bool:
    width = row_a.size
    for c, val in enumerate(row_b):
        if val == -1:
            continue
        src = c - int(shift)
        if not (0 <= src < width):
            if val == 1:
                return False
            continue
        other = int(row_a[src])
        if other != -1 and other != int(val):
            return False
    return True


def _reconstruct_base_rows(observed: np.ndarray, period: int, shift: int) -> list[np.ndarray]:
    count, width = observed.shape
    base_rows: list[np.ndarray] = []
    for residue in range(period):
        base = np.full(width, -1, dtype=int)
        rows = list(range(residue, count, period))
        for k, r in enumerate(rows):
            delta = k * int(shift)
            for c, val in enumerate(observed[r]):
                val = int(val)
                if val == -1:
                    continue
                src = c - delta
                if 0 <= src < width:
                    if base[src] == -1:
                        base[src] = val
                    elif base[src] != val:
                        raise ValueError("conflicting evidence for base row")
                elif val == 1:
                    raise ValueError("clipped source cannot produce visible red")
        if np.any(base == -1):
            raise ValueError("base row remains ambiguous")
        base_rows.append(base)

    for r in range(count):
        residue = r % period
        k = r // period
        expected = _shift_row_mask(base_rows[residue], k * int(shift))
        known = observed[r] != -1
        if not np.array_equal(expected[known], observed[r][known]):
            raise ValueError("reconstructed rows do not match observations")
    return base_rows


def _find_period_and_shift(observed: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    count, width = observed.shape
    candidates: list[tuple[int, int, list[np.ndarray]]] = []
    for period in range(1, count):
        shifts = []
        for shift in range(-(width - 1), width):
            if all(_pair_compatible(observed[r - period], observed[r], shift) for r in range(period, count)):
                try:
                    base_rows = _reconstruct_base_rows(observed, period, shift)
                except ValueError:
                    continue
                shifts.append((shift, base_rows))
        if len(shifts) == 1:
            shift, base_rows = shifts[0]
            return period, shift, base_rows
        for shift, base_rows in shifts:
            candidates.append((period, shift, base_rows))
    assert len(candidates) == 1, "pattern period/shift must be uniquely inferable"
    return candidates[0]


def _pattern_row(base_rows: list[np.ndarray], period: int, shift: int, row: int) -> np.ndarray:
    residue = row % period
    k = row // period
    return _shift_row_mask(base_rows[residue], k * int(shift))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the reachable part of a top row pattern by inferring a horizontally drifting period beneath wall occlusions.

    Input structure:
    - The background is black, blue pixels form wall components, and one magenta marker sits on a corner.
    - Border-touching blue components are structural walls; interior blue components are decorative fragments.
    - The top few rows contain a red row pattern, but some red cells may be hidden where structural walls overlap them.

    Rule:
    - Classify blue components into structural vs decorative using border contact.
    - Flood-fill from the magenta corner, blocked only by structural walls, to identify the marker-side reachable region.
    - Infer the minimal vertical period and horizontal shift of the visible red rows while treating blue cells as unknown masked positions.
    - Extend the row pattern downward; for each missing row, shift the row one period above by the inferred amount and draw red only on reachable background cells.
    - Erase every decorative blue fragment to black and keep structural walls unchanged.

    Key insight:
    - The same blue pixels both hide evidence of the row pattern and define where the pattern is allowed to continue, so the hidden periodic template must be reconstructed before filling only the reachable side.
    """
    g = np.asarray(grid, dtype=int)
    structural, decorative = _classify_walls(g)
    reachable = _reachable_mask(g, structural)
    visible = _visible_height(g)
    observed = _observed_pattern_rows(g, visible)
    period, shift, base_rows = _find_period_and_shift(observed)

    out = g.copy()
    for comp in decorative:
        out[comp[:, 0], comp[:, 1]] = BLACK

    for r in range(visible, g.shape[0]):
        mask = _pattern_row(base_rows, period, shift, r)
        for c in np.flatnonzero(mask):
            if reachable[r, c] and out[r, c] == BLACK:
                out[r, c] = RED
    return out


def _marker_corner(h: int, w: int) -> tuple[int, int]:
    return random.choice([(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)])


def _reachable_side(corner: tuple[int, int], w: int, split_col: int) -> tuple[int, int]:
    _r, c = corner
    if c == 0:
        return 0, split_col - 1
    return split_col + 1, w - 1


def _unreachable_side(corner: tuple[int, int], w: int, split_col: int) -> tuple[int, int]:
    _r, c = corner
    if c == 0:
        return split_col + 1, w - 1
    return 0, split_col - 1


def _random_component_cells(count: int) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    seen = {(0, 0)}
    while len(cells) < count:
        r, c = random.choice(cells)
        nr, nc = random.choice(list(neighbors4(r, c)))
        if (nr, nc) not in seen:
            seen.add((nr, nc))
            cells.append((nr, nc))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return [(r - r0, c - c0) for r, c in cells]


def _place_decorations(grid: np.ndarray, ucols: tuple[int, int], structural: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    uc0, uc1 = ucols
    count = random.randint(0, 3)
    occupied = set(structural)
    for _ in range(count):
        size = random.randint(2, 4)
        shape = _random_component_cells(size)
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if uc1 - uc0 + 1 < sw + 2 or h < sh + 2:
            return False
        placed = False
        for _ in range(30):
            r0 = random.randint(1, h - sh - 1)
            c0 = random.randint(uc0 + 1, uc1 - sw)
            coords = {(r0 + dr, c0 + dc) for dr, dc in shape}
            if any(r in {0, h - 1} or c in {0, w - 1} for r, c in coords):
                continue
            if any((r, c) in occupied for r, c in coords):
                continue
            if any(grid[r, c] != BLACK for r, c in coords):
                continue
            if any(
                (nr, nc) in occupied
                for r, c in coords
                for nr, nc in neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w
            ):
                continue
            for r, c in coords:
                grid[r, c] = BLUE
            occupied |= coords
            placed = True
            break
        if not placed:
            return False
    return True


def _build_pattern_rows(
    h: int,
    w: int,
    visible: int,
    period: int,
    shift: int,
    corner: tuple[int, int],
    split_col: int,
) -> list[np.ndarray] | None:
    rc0, rc1 = _reachable_side(corner, w, split_col)
    usable = list(range(rc0, rc1 + 1))
    if corner[0] == 0 and corner[1] in usable:
        usable = [c for c in usable if c != corner[1]]
    if len(usable) < 5:
        return None

    near_wall = [c for c in usable if abs(c - split_col) <= 3]
    if not near_wall:
        near_wall = usable[-3:] if split_col > usable[-1] else usable[:3]

    base_rows: list[np.ndarray] = []
    for residue in range(period):
        row = np.zeros(w, dtype=int)
        anchor = random.choice(near_wall)
        row[anchor] = 1
        extra_count = random.randint(1, 3)
        others = [c for c in usable if c != anchor]
        random.shuffle(others)
        for c in others[:extra_count]:
            row[c] = 1
        if row.sum() < 2:
            return None
        base_rows.append(row)

    full = []
    for r in range(h):
        mask = _pattern_row(base_rows, period, shift, r)
        full.append(mask)
    full_arr = np.array(full, dtype=int)

    if any(full_arr[r].sum() == 0 for r in range(visible)):
        return None
    if np.sum(full_arr[visible:]) == 0:
        return None
    if period == 2 and np.array_equal(base_rows[0], base_rows[1]):
        return None
    if period == 1 and shift == 0 and base_rows[0].sum() < 3:
        return None
    return [row.copy() for row in full_arr]


def _make_candidate() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    grid = np.full((h, w), BLACK, dtype=int)

    corner = _marker_corner(h, w)
    period = random.choice([1, 2])
    visible = random.randint(max(3, 2 * period + 1), 5)
    shift = random.choice([-2, -1, 0, 1, 2])

    split_col = random.randint(5, w - 6)
    reachable_width = _reachable_side(corner, w, split_col)[1] - _reachable_side(corner, w, split_col)[0] + 1
    unreachable_width = _unreachable_side(corner, w, split_col)[1] - _unreachable_side(corner, w, split_col)[0] + 1
    if min(reachable_width, unreachable_width) < 5:
        return None

    grid[:, split_col] = BLUE
    spur_rows = random.sample(list(range(visible)), k=random.randint(1, min(3, visible)))
    for r in spur_rows:
        max_len = max(2, min(4, reachable_width - 1))
        length = random.randint(1, max_len)
        if corner[1] == 0:
            c0 = split_col - length
            grid[r, c0 : split_col + 1] = BLUE
        else:
            c1 = split_col + length
            grid[r, split_col : c1 + 1] = BLUE

    grid[corner] = MAGENTA
    structural, _decorative = _classify_walls(grid)
    pattern_rows = _build_pattern_rows(h, w, visible, period, shift, corner, split_col)
    if pattern_rows is None:
        return None

    masked_reds = 0
    for r in range(visible):
        for c in np.flatnonzero(pattern_rows[r]):
            if grid[r, c] == BLACK:
                grid[r, c] = RED
            elif grid[r, c] == BLUE:
                masked_reds += 1
    if masked_reds == 0:
        return None

    if not _place_decorations(grid, _unreachable_side(corner, w, split_col), structural):
        return None

    if any(np.count_nonzero(grid[r] == RED) == 0 for r in range(visible)):
        return None

    observed = _observed_pattern_rows(grid, visible)
    try:
        found_period, found_shift, _base_rows = _find_period_and_shift(observed)
    except AssertionError:
        return None
    if (found_period, found_shift) != (period, shift):
        return None

    structural, decorative = _classify_walls(grid)
    if not decorative and random.random() < 0.35:
        return None

    reachable = _reachable_mask(grid, structural)
    reach_cols = np.flatnonzero(reachable[:visible].any(axis=0))
    if reach_cols.size < reachable_width - 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == BLUE) and not structural:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
