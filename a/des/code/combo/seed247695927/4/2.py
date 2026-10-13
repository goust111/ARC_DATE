from __future__ import annotations

import random
import numpy as np


Axis = tuple[int, str, int]
AXIS_KINDS = ("row", "col", "main_diag", "anti_diag")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    blocked = set() if exclude is None else set(exclude)
    return [int(color) for color in np.unique(grid) if int(color) not in blocked]


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_axis(grid: np.ndarray, bg: int) -> Axis:
    h, w = grid.shape
    candidates: list[Axis] = []
    for color in _colors_present(grid, exclude={bg}):
        mask = grid == color
        count = int(mask.sum())
        if count == w:
            for r in np.flatnonzero(mask.all(axis=1)):
                candidates.append((color, "row", int(r)))
        if count == h:
            for c in np.flatnonzero(mask.all(axis=0)):
                candidates.append((color, "col", int(c)))
        if h == w and count == h:
            main = np.eye(h, dtype=bool)
            anti = np.fliplr(main)
            if np.all(mask[main]):
                candidates.append((color, "main_diag", 0))
            if np.all(mask[anti]):
                candidates.append((color, "anti_diag", 0))
    assert len(candidates) == 1, "expected exactly one full straight axis"
    return candidates[0]


def _reflect(axis: Axis, shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    _axis_color, kind, index = axis
    h, w = shape
    if kind == "row":
        return 2 * index - r, c
    if kind == "col":
        return r, 2 * index - c
    if kind == "main_diag":
        assert h == w
        return c, r
    if kind == "anti_diag":
        assert h == w
        return h - 1 - c, w - 1 - r
    raise AssertionError(f"unknown axis kind: {kind}")


def _smallest_non_axis_color(grid: np.ndarray, bg: int, axis_color: int) -> int:
    areas = {
        color: int(np.count_nonzero(grid == color))
        for color in _colors_present(grid, exclude={bg, axis_color})
    }
    assert len(areas) >= 2, "expected at least two non-axis foreground colors"
    min_area = min(areas.values())
    smallest = [color for color, area in areas.items() if area == min_area]
    assert len(smallest) == 1, "expected unique smallest non-axis color"
    return smallest[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    axis = _find_axis(g, bg)
    axis_color = axis[0]
    mirror_color = _smallest_non_axis_color(g, bg, axis_color)

    out = g.copy()
    h, w = g.shape
    for r_raw, c_raw in np.argwhere(g == mirror_color):
        r = int(r_raw)
        c = int(c_raw)
        rr, cc = _reflect(axis, g.shape, r, c)
        assert 0 <= rr < h and 0 <= cc < w
        assert int(out[rr, cc]) in {bg, mirror_color}
        out[rr, cc] = mirror_color
    return out


def _side_mask(shape: tuple[int, int], axis_kind: str, side: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    if axis_kind == "row":
        axis_row = h // 2
        return rr < axis_row if side == 0 else rr > axis_row
    if axis_kind == "col":
        axis_col = w // 2
        return cc < axis_col if side == 0 else cc > axis_col
    if axis_kind == "main_diag":
        return cc > rr if side == 0 else cc < rr
    return rr + cc < h - 1 if side == 0 else rr + cc > h - 1


def _draw_axis(grid: np.ndarray, axis_kind: str, axis_color: int) -> None:
    h, w = grid.shape
    if axis_kind == "row":
        grid[h // 2, :] = axis_color
    elif axis_kind == "col":
        grid[:, w // 2] = axis_color
    elif axis_kind == "main_diag":
        np.fill_diagonal(grid, axis_color)
    else:
        np.fill_diagonal(np.fliplr(grid), axis_color)


def _component_cells(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _partition_total(total: int, count: int, lo: int, hi: int) -> list[int]:
    sizes: list[int] = []
    remaining = total
    for idx in range(count):
        left = count - idx - 1
        cur_lo = max(lo, remaining - left * hi)
        cur_hi = min(hi, remaining - left * lo)
        sizes.append(random.randint(cur_lo, cur_hi))
        remaining -= sizes[-1]
    random.shuffle(sizes)
    return sizes


def _grow_component(
    allowed: np.ndarray,
    occupied: np.ndarray,
    blocked_same: np.ndarray,
    target_size: int,
) -> np.ndarray | None:
    available = allowed & ~occupied & ~blocked_same
    regions = [comp for comp in _component_cells(available) if len(comp) >= target_size]
    if not regions:
        return None

    region = random.choice(regions)
    seed = tuple(int(v) for v in region[random.randrange(len(region))])
    chosen = {seed}
    frontier: set[tuple[int, int]] = set()
    h, w = available.shape

    def extend(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and available[nr, nc] and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    extend(*seed)
    while len(chosen) < target_size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        extend(*cell)
    return np.array(sorted(chosen), dtype=int)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        blocked[r, c] = True
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _sample_shape(axis_kind: str) -> tuple[int, int]:
    if axis_kind in {"main_diag", "anti_diag"}:
        n = random.randint(9, 16)
        return n, n
    if axis_kind == "row":
        h = random.choice([v for v in range(9, 18) if v % 2 == 1])
        w = random.randint(9, 18)
        return h, w
    h = random.randint(9, 18)
    w = random.choice([v for v in range(9, 18) if v % 2 == 1])
    return h, w


def _place_color(
    grid: np.ndarray,
    axis_kind: str,
    color: int,
    total: int,
    side: int,
    occupied: np.ndarray,
) -> bool:
    count_choices = [k for k in range(1, 4) if k * 2 <= total <= k * 8]
    if not count_choices:
        return False
    allowed = _side_mask(grid.shape, axis_kind, side)
    blocked_same = np.zeros_like(grid, dtype=bool)
    for size in _partition_total(total, random.choice(count_choices), 2, 8):
        coords = _grow_component(allowed, occupied, blocked_same, size)
        if coords is None:
            return False
        rows = coords[:, 0]
        cols = coords[:, 1]
        reflected = np.array([_reflect((0, axis_kind, grid.shape[0] // 2 if axis_kind == "row" else grid.shape[1] // 2), grid.shape, int(r), int(c)) for r, c in coords], dtype=int) if axis_kind in {"row", "col"} else np.array([_reflect((0, axis_kind, 0), grid.shape, int(r), int(c)) for r, c in coords], dtype=int)
        if np.any(occupied[reflected[:, 0], reflected[:, 1]]):
            return False
        if np.any(grid[reflected[:, 0], reflected[:, 1]] != _get_bg(grid)):
            return False
        grid[rows, cols] = color
        occupied[rows, cols] = True
        _mark_blocked(blocked_same, coords)
    return True


def _axis_index(shape: tuple[int, int], axis_kind: str) -> int:
    h, w = shape
    if axis_kind == "row":
        return h // 2
    if axis_kind == "col":
        return w // 2
    return 0


def _make_attempt() -> np.ndarray | None:
    axis_kind = random.choice(AXIS_KINDS)
    shape = _sample_shape(axis_kind)
    bg = random.randint(0, 9)
    axis_color = random.choice([c for c in range(10) if c != bg])
    remaining = [c for c in range(10) if c not in {bg, axis_color}]
    random.shuffle(remaining)
    extra_count = random.randint(1, min(3, len(remaining) - 1))
    mirror_color = remaining[0]
    other_colors = remaining[1 : 1 + extra_count]

    grid = np.full(shape, bg, dtype=int)
    _draw_axis(grid, axis_kind, axis_color)
    occupied = grid != bg

    mirror_total = random.randint(2, 5)
    other_totals = []
    lower_bound = mirror_total + 1
    for _ in other_colors:
        other_totals.append(random.randint(lower_bound, lower_bound + 5))

    side = random.randint(0, 1)
    placement_order = [(mirror_color, mirror_total)] + list(zip(other_colors, other_totals))
    random.shuffle(placement_order[1:])
    for color, total in placement_order:
        if not _place_color(grid, axis_kind, color, total, side, occupied):
            return None

    try:
        axis = _find_axis(grid, bg)
        if axis[0] != axis_color or axis[1] != axis_kind or axis[2] != _axis_index(shape, axis_kind):
            return None
        if _smallest_non_axis_color(grid, bg, axis_color) != mirror_color:
            return None
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
