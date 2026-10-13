from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR8 = [
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
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _flood_mask(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    sr, sc = start
    seen = np.zeros((h, w), dtype=bool)
    if not mask[sr, sc]:
        return seen
    stack = [(sr, sc)]
    seen[sr, sc] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    unseen = mask.copy()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not unseen[r, c]:
                continue
            comp_mask = _flood_mask(unseen, (r, c))
            coords = [tuple(map(int, rc)) for rc in np.argwhere(comp_mask)]
            unseen[comp_mask] = False
            comps.append(coords)
    return comps


def _wall_extents(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    top = np.full(w, -1, dtype=int)
    bottom = np.full(w, -1, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == RED)
        assert rows.size > 0, "wall must occupy every column"
        top[c] = int(rows.min())
        bottom[c] = int(rows.max())
    assert top.min() > 0 and bottom.max() < h - 1, "wall must avoid the top and bottom borders"
    return top, bottom


def _partition_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    passable = grid != RED
    lower = _flood_mask(passable, (grid.shape[0] - 1, 0))
    upper = passable & ~lower
    assert lower[grid.shape[0] - 1, 0], "lower-left corner must belong to the lower region"
    assert upper.any(), "wall must create a non-empty upper region"
    assert len(_components(passable)) == 2, "wall must split the passable cells into exactly two regions"
    return lower, upper


def _slide_region(
    blues: list[tuple[int, int]],
    *,
    region_name: str,
    toward_wall: bool,
    wall_top: np.ndarray,
    wall_bottom: np.ndarray,
    h: int,
) -> list[tuple[int, int]]:
    if not blues:
        return []

    if region_name == "upper":
        if toward_wall:
            target = lambda c: int(wall_top[c] - 1)
            priority = lambda rc: (target(rc[1]) - rc[0], -rc[0], rc[1])
            retreat = -1
            valid = lambda src_r, dst_r: dst_r >= src_r
        else:
            target = lambda c: 0
            priority = lambda rc: (rc[0], rc[0], rc[1])
            retreat = 1
            valid = lambda src_r, dst_r: dst_r <= src_r
    else:
        if toward_wall:
            target = lambda c: int(wall_bottom[c] + 1)
            priority = lambda rc: (rc[0] - target(rc[1]), rc[0], rc[1])
            retreat = 1
            valid = lambda src_r, dst_r: dst_r <= src_r
        else:
            target = lambda c: h - 1
            priority = lambda rc: ((h - 1) - rc[0], -rc[0], rc[1])
            retreat = -1
            valid = lambda src_r, dst_r: dst_r >= src_r

    occupied: set[tuple[int, int]] = set()
    landed: list[tuple[int, int]] = []
    for r, c in sorted(blues, key=priority):
        dst_r = target(c)
        while (dst_r, c) in occupied:
            dst_r += retreat
        assert 0 <= dst_r < h, "collision stack must stay inside the grid"
        assert valid(r, dst_r), "generated input overfills a destination column"
        occupied.add((dst_r, c))
        landed.append((dst_r, c))
    return landed


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_top, wall_bottom = _wall_extents(g)
    lower, upper = _partition_regions(g)

    lower_blues = [tuple(map(int, rc)) for rc in np.argwhere((g == BLUE) & lower)]
    upper_blues = [tuple(map(int, rc)) for rc in np.argwhere((g == BLUE) & upper)]
    assert lower_blues and upper_blues, "both regions must contain blue pixels"

    lower_odd = len(lower_blues) % 2 == 1
    upper_odd = len(upper_blues) % 2 == 1
    assert lower_odd != upper_odd, "the two regions must have opposite parity"

    out = np.full((h, w), BLACK, dtype=int)
    out[g == RED] = RED

    for r, c in _slide_region(
        upper_blues,
        region_name="upper",
        toward_wall=upper_odd,
        wall_top=wall_top,
        wall_bottom=wall_bottom,
        h=h,
    ):
        out[r, c] = BLUE

    for r, c in _slide_region(
        lower_blues,
        region_name="lower",
        toward_wall=lower_odd,
        wall_top=wall_top,
        wall_bottom=wall_bottom,
        h=h,
    ):
        out[r, c] = BLUE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each blue pixel either toward the red wall or toward the outer edge, depending on the parity of its wall-defined region.

    Input structure:
    - The grid uses black background, blue isolated pixels, and a red wall that spans from the left border to the right border.
    - The red wall does not touch the top or bottom border and splits the non-red cells into exactly two 4-connected regions.
    - The lower region is the one reachable from the lower-left corner through non-red cells; the other region is the upper region.

    Rule:
    - Count the blue pixels in the upper region and in the lower region separately.
    - In the odd-count region, slide each blue pixel vertically toward the nearest red wall cell in its column and stop one cell before the wall.
    - In the even-count region, slide each blue pixel vertically toward that region's outer border row.
    - When multiple pixels in one column target the same destination line, move the ones already nearer to that line first and stack later pixels immediately behind them.

    Key insight:
    - The wall first defines the counting regions, then each region's blue-count parity selects the motion target, so the same local blue arrangement can move in different directions across examples.
    """
    return _transform_grid(grid)


def _touches_wall(wall: np.ndarray, r: int, c: int) -> bool:
    h, w = wall.shape
    for dr, dc in _DIR8:
        nr, nc = r + dr, c + dc
        if _in_bounds((h, w), nr, nc) and wall[nr, nc]:
            return True
    return False


def _isolated_from_existing(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    for rr, cc in occupied:
        if max(abs(rr - r), abs(cc - c)) <= 1:
            return False
    return True


def _draw_wall(h: int, w: int) -> np.ndarray:
    center_lo = max(2, h // 2 - 2)
    center_hi = min(h - 3, h // 2 + 2)
    profile = [random.randint(center_lo, center_hi)]
    for _c in range(1, w):
        prev = profile[-1]
        step = random.choices([-1, 0, 1], weights=[2, 5, 2])[0]
        nxt = max(2, min(h - 3, prev + step))
        profile.append(nxt)

    wall = np.zeros((h, w), dtype=bool)
    wall[profile[0], 0] = True
    for c in range(1, w):
        a = profile[c - 1]
        b = profile[c]
        wall[min(a, b) : max(a, b) + 1, c] = True
    return wall


def _candidate_rows_by_col(
    region_mask: np.ndarray,
    wall: np.ndarray,
    wall_top: np.ndarray,
    wall_bottom: np.ndarray,
    *,
    region_name: str,
    toward_wall: bool,
) -> dict[int, list[int]]:
    h, w = region_mask.shape
    out: dict[int, list[int]] = {}
    for c in range(w):
        rows: list[int] = []
        for r in range(h):
            if not region_mask[r, c] or _touches_wall(wall, r, c):
                continue
            if region_name == "upper":
                if toward_wall and r > wall_top[c] - 3:
                    continue
                if not toward_wall and r < 2:
                    continue
            else:
                if toward_wall and r < wall_bottom[c] + 3:
                    continue
                if not toward_wall and r > h - 3:
                    continue
            rows.append(r)
        if rows:
            out[c] = rows
    return out


def _pick_spaced_rows(rows: list[int], count: int) -> list[int] | None:
    rows = rows[:]
    random.shuffle(rows)
    chosen: list[int] = []
    for r in rows:
        if all(abs(r - prev) >= 2 for prev in chosen):
            chosen.append(r)
            if len(chosen) == count:
                return chosen
    return None


def _place_region_pixels(
    region_mask: np.ndarray,
    wall: np.ndarray,
    wall_top: np.ndarray,
    wall_bottom: np.ndarray,
    *,
    region_name: str,
    toward_wall: bool,
    count: int,
    require_collision: bool,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = _candidate_rows_by_col(
        region_mask,
        wall,
        wall_top,
        wall_bottom,
        region_name=region_name,
        toward_wall=toward_wall,
    )
    if not candidates:
        return None

    chosen: list[tuple[int, int]] = []
    blocked_columns: dict[int, int] = {}
    collision_column: int | None = None

    if require_collision:
        cols = list(candidates.keys())
        random.shuffle(cols)
        for c in cols:
            rows = _pick_spaced_rows(candidates[c], 2)
            if rows is None:
                continue
            pair = [(r, c) for r in rows]
            if all(_isolated_from_existing(r, c, occupied | set(chosen)) for r, c in pair):
                chosen.extend(pair)
                blocked_columns[c] = 2
                collision_column = c
                break
        if collision_column is None:
            return None

    cells: list[tuple[int, int]] = []
    for c, rows in candidates.items():
        for r in rows:
            cells.append((r, c))
    random.shuffle(cells)

    for r, c in cells:
        if len(chosen) >= count:
            break
        if (r, c) in chosen:
            continue
        limit = blocked_columns.get(c, 1 if c != collision_column else 2)
        if sum(1 for _r, cc in chosen if cc == c) >= limit:
            continue
        if not _isolated_from_existing(r, c, occupied | set(chosen)):
            continue
        chosen.append((r, c))

    if len(chosen) != count:
        return None
    return chosen


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h > 30 or w > 30:
        return False

    wall = g == RED
    if wall[0].any() or wall[-1].any():
        return False

    try:
        wall_top, wall_bottom = _wall_extents(g)
        lower, upper = _partition_regions(g)
        out = _transform_grid(g)
    except AssertionError:
        return False

    all_blues = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    if not (4 <= len(all_blues) <= 10):
        return False

    for r, c in all_blues:
        if _touches_wall(wall, r, c):
            return False
        for rr, cc in all_blues:
            if (rr, cc) != (r, c) and max(abs(rr - r), abs(cc - c)) <= 1:
                return False

    lower_blues = [tuple(map(int, rc)) for rc in np.argwhere((g == BLUE) & lower)]
    upper_blues = [tuple(map(int, rc)) for rc in np.argwhere((g == BLUE) & upper)]
    if not (2 <= len(lower_blues) <= 5 and 2 <= len(upper_blues) <= 5):
        return False
    if (len(lower_blues) % 2) == (len(upper_blues) % 2):
        return False

    has_collision_column = False
    for group in (lower_blues, upper_blues):
        counts: dict[int, int] = {}
        for _r, c in group:
            counts[c] = counts.get(c, 0) + 1
        if max(counts.values(), default=0) >= 2:
            has_collision_column = True
    if not has_collision_column:
        return False

    if np.array_equal(g, out):
        return False
    if int(np.count_nonzero(out == BLUE)) != len(all_blues):
        return False

    upper_odd = len(upper_blues) % 2 == 1
    lower_odd = len(lower_blues) % 2 == 1
    upper_moves_to_wall = upper_odd
    lower_moves_to_wall = lower_odd

    for r, c in upper_blues:
        if upper_moves_to_wall and r >= wall_top[c] - 1:
            return False
        if not upper_moves_to_wall and r <= 1:
            return False
    for r, c in lower_blues:
        if lower_moves_to_wall and r <= wall_bottom[c] + 1:
            return False
        if not lower_moves_to_wall and r >= h - 2:
            return False

    return True


def _build_input_attempt() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    wall = _draw_wall(h, w)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[wall] = RED

    try:
        wall_top, wall_bottom = _wall_extents(grid)
        lower_mask, upper_mask = _partition_regions(grid)
    except AssertionError:
        return None

    odd_region = random.choice(["upper", "lower"])
    upper_count = random.choice([3, 5]) if odd_region == "upper" else random.choice([2, 4])
    lower_count = random.choice([3, 5]) if odd_region == "lower" else random.choice([2, 4])

    collision_region = random.choice(["upper", "lower"])
    occupied: set[tuple[int, int]] = set()

    upper_cells = _place_region_pixels(
        upper_mask,
        wall,
        wall_top,
        wall_bottom,
        region_name="upper",
        toward_wall=(odd_region == "upper"),
        count=upper_count,
        require_collision=(collision_region == "upper"),
        occupied=occupied,
    )
    if upper_cells is None:
        return None
    occupied.update(upper_cells)

    lower_cells = _place_region_pixels(
        lower_mask,
        wall,
        wall_top,
        wall_bottom,
        region_name="lower",
        toward_wall=(odd_region == "lower"),
        count=lower_count,
        require_collision=(collision_region == "lower"),
        occupied=occupied,
    )
    if lower_cells is None:
        return None

    for r, c in upper_cells + lower_cells:
        grid[r, c] = BLUE

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
