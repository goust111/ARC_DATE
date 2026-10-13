import random
from functools import lru_cache

import numpy as np


BG = 0
PATH = 5
LEGEND_COLORS = (1, 2, 3, 4)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _shortest_period(seq: list[int]) -> list[int]:
    n = len(seq)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        base = seq[:p]
        if base * (n // p) == seq:
            return base
    raise AssertionError("legend row has no repeating period")


def _find_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    mask = (grid != BG) & (grid != PATH)
    rows = [r for r in range(grid.shape[0]) if np.any(mask[r])]
    assert len(rows) == 1, "expected exactly one legend row"
    row = rows[0]
    cols = np.flatnonzero(mask[row])
    assert cols.size > 0 and np.all(np.diff(cols) == 1), "legend row must be one contiguous run"
    seq = [int(v) for v in grid[row, cols[0] : cols[-1] + 1]]
    return int(row), _shortest_period(seq)


def _column_groups(grid: np.ndarray) -> list[np.ndarray]:
    nonempty = np.any(grid == PATH, axis=0)
    groups: list[np.ndarray] = []
    c = 0
    while c < grid.shape[1]:
        if not nonempty[c]:
            c += 1
            continue
        c0 = c
        while c < grid.shape[1] and nonempty[c]:
            c += 1
        groups.append((grid[:, c0:c] == PATH))
    assert groups, "expected at least one path fragment"
    return groups


def _degree(mask: np.ndarray, r: int, c: int) -> int:
    deg = 0
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
            deg += 1
    return deg


def _endpoint(mask: np.ndarray, side: str) -> tuple[int, int]:
    assert side in {"left", "right"}
    c = 0 if side == "left" else mask.shape[1] - 1
    endpoints = []
    for r in range(mask.shape[0]):
        if mask[r, c] and _degree(mask, r, c) == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 1, f"expected one {side} endpoint"
    return endpoints[0]


def _shift_mask(mask: np.ndarray, dr: int) -> np.ndarray:
    shifted = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    rows = coords[:, 0] + int(dr)
    assert np.all((0 <= rows) & (rows < mask.shape[0])), "vertical shift moved a fragment out of bounds"
    shifted[rows, coords[:, 1]] = True
    return shifted


def _assemble_groups(groups: list[np.ndarray]) -> np.ndarray:
    height = groups[0].shape[0]
    width = sum(group.shape[1] for group in groups)
    out = np.zeros((height, width), dtype=bool)

    c = 0
    first = groups[0]
    out[:, : first.shape[1]] = first
    prev_right_row = _endpoint(first, "right")[0]
    c += first.shape[1]

    for group in groups[1:]:
        dr = prev_right_row - _endpoint(group, "left")[0]
        shifted = _shift_mask(group, dr)
        out[:, c : c + shifted.shape[1]] = shifted
        prev_right_row = _endpoint(shifted, "right")[0]
        c += shifted.shape[1]

    return out


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert pts, "expected a non-empty path"
    endpoints = [pt for pt in pts if sum((nr, nc) in pts for nr, nc in _neighbors4(*pt)) == 1]
    assert len(endpoints) == 2, "assembled result must be one simple path"
    start = min(endpoints, key=lambda rc: (rc[1], rc[0]))

    path = [start]
    prev = None
    cur = start
    while True:
        nxt = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must be unbranched"
        prev, cur = cur, nxt[0]
        path.append(cur)
        assert len(path) <= len(pts)

    assert len(path) == len(pts), "path tracing did not visit every pixel"
    return path


def _recolor_path(mask: np.ndarray, cycle: list[int]) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    for i, (r, c) in enumerate(_trace_path(mask)):
        out[r, c] = int(cycle[i % len(cycle)])
    return out


def _is_simple_path(mask: np.ndarray) -> bool:
    try:
        _trace_path(mask)
    except AssertionError:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Concatenate gray path fragments by matching endpoints, then recolor the walked chain by a legend cycle.

    Input structure:
    - Black background contains several gray path fragments, each occupying a contiguous column run and separated from the next fragment by one or more all-black columns.
    - Every fragment is a 4-connected unbranched path with one degree-1 endpoint on its leftmost column and one degree-1 endpoint on its rightmost column.
    - Near the bottom there is one horizontal legend row made of non-black, non-gray colors repeated in a fixed periodic order.

    Rule:
    - Read the legend row and infer its shortest repeating color cycle.
    - Delete the separator columns between gray fragments, keep the first fragment at its original vertical position, and shift each later fragment vertically so its left endpoint row matches the previous fragment's right endpoint row before placing it immediately to the right.
    - Walk the assembled 4-connected chain from its left endpoint to its right endpoint and recolor the i-th visited pixel with cycle[i mod period].
    - Remove the legend from the output; keep the original height and use only the compacted path width.

    Key insight:
    - The color phase advances along path-traversal order, not by geometric row or column, so bends matter because two cells in the same column can receive different cycle phases.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _legend_row, cycle = _find_legend(g)
    groups = _column_groups(g)
    assembled = _assemble_groups(groups)
    return _recolor_path(assembled, cycle)


def _render_path(height: int, width: int, path: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    mask[rows, cols] = True
    return mask


def _path_is_valid(height: int, width: int, path: tuple[tuple[int, int], ...], left_row: int, right_row: int) -> bool:
    pts = set(path)
    if len(pts) != len(path):
        return False
    if path[0] != (left_row, 0) or path[-1] != (right_row, width - 1):
        return False
    if set(c for _, c in path) != set(range(width)):
        return False
    if not all(0 <= r < height and 0 <= c < width for r, c in path):
        return False

    deg1 = 0
    left_endpoints = 0
    right_endpoints = 0
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if deg not in (1, 2):
            return False
        if deg == 1:
            deg1 += 1
            if c == 0:
                left_endpoints += 1
            if c == width - 1:
                right_endpoints += 1
    return deg1 == 2 and left_endpoints == 1 and right_endpoints == 1


@lru_cache(maxsize=None)
def _enumerate_paths(height: int, width: int, cells: int, left_row: int, right_row: int):
    start = (left_row, 0)
    goal = (right_row, width - 1)
    out = []

    def dfs(path: tuple[tuple[int, int], ...]):
        cur = path[-1]
        remaining = cells - len(path)
        if remaining == 0:
            if cur == goal and _path_is_valid(height, width, path, left_row, right_row):
                out.append(path)
            return

        dist = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
        if dist > remaining:
            return

        for nr, nc in _neighbors4(*cur):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in path:
                continue
            next_remaining = cells - (len(path) + 1)
            if abs(nr - goal[0]) + abs(nc - goal[1]) > next_remaining:
                continue
            dfs(path + ((nr, nc),))

    dfs((start,))
    return tuple(out)


def _has_vertical_step(mask: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        if r + 1 < mask.shape[0] and mask[r + 1, c]:
            return True
    return False


def _group_choices(height: int, left_row: int):
    choices = []
    for width in range(2, 6):
        for cells in range(max(3, width), 9):
            for right_row in range(height):
                for path in _enumerate_paths(height, width, cells, left_row, right_row):
                    mask = _render_path(height, width, path)
                    choices.append((mask, right_row))
    return choices


def _choose_output_groups(path_height: int, n_groups: int) -> list[np.ndarray] | None:
    for _ in range(120):
        groups: list[np.ndarray] = []
        left_row = random.randrange(path_height)
        seen_vertical = False
        for i in range(n_groups):
            choices = _group_choices(path_height, left_row)
            if i == n_groups - 1 and not seen_vertical:
                choices = [item for item in choices if _has_vertical_step(item[0])]
            if not choices:
                break
            picked = None
            for mask, right_row in random.sample(choices, min(len(choices), 80)):
                trial_groups = groups + [mask]
                if not _is_simple_path(_assemble_groups(trial_groups)):
                    continue
                picked = (mask, right_row)
                break
            if picked is None:
                break
            mask, right_row = picked
            groups.append(mask)
            seen_vertical |= _has_vertical_step(mask)
            left_row = right_row
        if len(groups) != n_groups or not seen_vertical:
            continue
        assembled = _assemble_groups(groups)
        if not _has_vertical_step(assembled) or not _is_simple_path(assembled):
            continue
        return groups
    return None


def _valid_shifts(group: np.ndarray) -> list[int]:
    rows = np.flatnonzero(np.any(group, axis=1))
    assert rows.size > 0
    lo = -int(rows.min())
    hi = int(group.shape[0] - 1 - rows.max())
    return list(range(lo, hi + 1))


def _choose_input_groups(output_groups: list[np.ndarray]) -> list[np.ndarray] | None:
    shifted = [output_groups[0]]
    prev_shift = 0
    for group in output_groups[1:]:
        choices = [dr for dr in _valid_shifts(group) if dr != prev_shift]
        if not choices:
            return None
        dr = random.choice(choices)
        shifted.append(_shift_mask(group, dr))
        prev_shift = dr
    return shifted


def _paint_groups(height: int, groups: list[np.ndarray], separators: list[int]) -> np.ndarray:
    width = sum(group.shape[1] for group in groups) + sum(separators)
    grid = np.zeros((height, width), dtype=int)
    c = 0
    for i, group in enumerate(groups):
        patch = np.where(group, PATH, BG).astype(int)
        grid[:, c : c + group.shape[1]] = np.maximum(grid[:, c : c + group.shape[1]], patch)
        c += group.shape[1]
        if i < len(separators):
            c += separators[i]
    return grid


def _legend_sequence(width: int) -> list[int]:
    feasible_periods = [p for p in (2, 3, 4) if 2 * p <= width]
    assert feasible_periods
    period = random.choice(feasible_periods)
    reps = random.randint(2, min(4, width // period))
    cycle = random.sample(list(LEGEND_COLORS), period)
    return cycle * reps


def _direct_output(input_grid: np.ndarray) -> np.ndarray:
    _legend_row, cycle = _find_legend(input_grid)
    groups = _column_groups(input_grid)
    assembled = _assemble_groups(groups)
    return _recolor_path(assembled, cycle)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(8, 11)
    gap_rows = random.randint(1, 2)
    legend_row = height - 1
    path_height = legend_row - gap_rows
    if path_height < 4:
        return None

    n_groups = random.randint(3, 5)
    output_groups = _choose_output_groups(path_height, n_groups)
    if output_groups is None:
        return None

    input_groups = _choose_input_groups(output_groups)
    if input_groups is None:
        return None

    separators = [random.randint(1, 3) for _ in range(n_groups - 1)]
    path_grid = _paint_groups(path_height, input_groups, separators)
    width = path_grid.shape[1]
    if width > 30:
        return None

    legend = _legend_sequence(width)
    if len(legend) > width:
        return None

    full = np.zeros((height, width), dtype=int)
    full[:path_height] = path_grid
    start = random.randint(0, width - len(legend))
    full[legend_row, start : start + len(legend)] = legend

    # Every adjacent input pair must need a vertical realignment.
    input_only_groups = _column_groups(full[:path_height])
    for a, b in zip(input_only_groups, input_only_groups[1:]):
        if _endpoint(a, "right")[0] == _endpoint(b, "left")[0]:
            return None

    solved = main(full)
    direct = _direct_output(full)
    if not np.array_equal(solved, direct):
        return None

    # Reject degenerate examples where path phase could be mistaken for a pure column rule.
    cols = np.where(np.sum(solved != BG, axis=0) >= 2)[0]
    if cols.size == 0:
        return None
    informative = False
    for c in cols:
        colors = [int(v) for v in solved[:, c] if v != BG]
        if len(set(colors)) >= 2:
            informative = True
            break
    if not informative:
        return None

    return full


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
