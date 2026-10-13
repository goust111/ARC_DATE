from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_rectangle(cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _infer_semantic_colors(
    grid: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, int, int]], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _most_common_color(g)
    colors = sorted(int(x) for x in np.unique(g) if int(x) != bg)

    rect_candidates: list[tuple[int, int, list[list[tuple[int, int]]]]] = []
    for color in colors:
        comps = _components_of_color(g, color)
        if not comps:
            continue
        if any(any(r == 0 for r, _c in comp) for comp in comps):
            continue
        if not all(len(comp) >= 4 and _is_solid_rectangle(comp) for comp in comps):
            continue
        if min(_bbox(comp)[0] for comp in comps) < h // 2:
            continue
        rect_candidates.append((sum(len(comp) for comp in comps), color, comps))

    assert len(rect_candidates) == 1, "expected one rectangle color"
    rect_color = rect_candidates[0][1]
    rect_components = rect_candidates[0][2]

    start_cells = [
        (0, c, int(g[0, c]))
        for c in range(g.shape[1])
        if int(g[0, c]) != bg and int(g[0, c]) != rect_color
    ]
    assert start_cells, "expected top-row start pixels"
    start_colors = {color for _r, _c, color in start_cells}
    assert len(start_colors) == len(start_cells), "start colors should be distinct"

    blocker_candidates = [color for color in colors if color != rect_color and color not in start_colors]
    assert len(blocker_candidates) == 1, "expected one blocker color"
    blocker_color = blocker_candidates[0]
    return bg, blocker_color, rect_color, start_cells, rect_components


def _follow_path(
    grid: np.ndarray,
    start_col: int,
    blocker_color: int,
    rect_lookup: dict[tuple[int, int], int],
) -> int:
    h, w = grid.shape
    r, c = 0, int(start_col)
    dr, dc = 1, 0
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "path should not cycle"
        seen.add(state)

        nr, nc = r + dr, c + dc
        assert 0 <= nr < h and 0 <= nc < w, "path left the grid before hitting a rectangle"

        if (nr, nc) in rect_lookup:
            return rect_lookup[(nr, nc)]

        if int(grid[nr, nc]) != blocker_color:
            r, c = nr, nc
            continue

        choices: list[tuple[int, int]] = []
        for tdr, tdc in ((-dc, dr), (dc, -dr)):
            tr, tc = r + tdr, c + tdc
            if 0 <= tr < h and 0 <= tc < w and int(grid[tr, tc]) != blocker_color:
                choices.append((tdr, tdc))
        assert len(choices) == 1, "each collision must force exactly one turn"
        dr, dc = choices[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each bottom rectangle by following its top-row start pixel through blocker-forced turns.

    Input structure:
    - The background is the dominant color.
    - The top row contains several distinct single-pixel starts in different colors.
    - A single blocker color forms one-cell-wide turning constraints in the middle.
    - The lower part contains several solid rectangles of one shared color.

    Rule:
    - From each top-row start, trace an invisible ray straight downward.
    - If the next cell ahead is a blocker, turn to the unique perpendicular non-blocker neighbor and continue.
    - When the ray first enters a bottom rectangle, recolor that entire rectangle with the start color.
    - Leave the starts, blockers, and the invisible path cells unchanged.

    Key insight:
    - The rectangles are labeled by the exit columns of hidden corridor walks, not by the columns directly under the starts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _bg, blocker_color, _rect_color, start_cells, rect_components = _infer_semantic_colors(g)

    rect_lookup: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(rect_components):
        for cell in comp:
            rect_lookup[cell] = idx

    out = g.copy()
    used_targets: set[int] = set()
    for _r, c, start_color in start_cells:
        target_idx = _follow_path(g, c, blocker_color, rect_lookup)
        assert target_idx not in used_targets, "each rectangle should be reached by at most one start"
        used_targets.add(target_idx)
        for rr, cc in rect_components[target_idx]:
            out[rr, cc] = int(start_color)

    return out


_ZONE_W = 6
_MIN_GAP = 2
_MIN_H = 16
_MAX_H = 26
_MAX_ATTEMPTS = 400
_PALETTE = list(range(10))


def _sample_layout(n_rects: int) -> tuple[int, list[tuple[int, int]]]:
    base_width = 2 + n_rects * _ZONE_W + (n_rects - 1) * _MIN_GAP
    assert base_width <= 30
    width = random.randint(base_width, 30)
    slack = width - base_width
    gaps = [1] + [_MIN_GAP] * (n_rects - 1) + [1]
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    zones: list[tuple[int, int]] = []
    col = gaps[0]
    for i in range(n_rects):
        zones.append((col, col + _ZONE_W - 1))
        col += _ZONE_W
        if i < n_rects - 1:
            col += gaps[i + 1]
    return width, zones


def _choose_turn_rows(rect_top: int, turns: int) -> list[int] | None:
    rows: list[int] = []
    prev = 0
    for i in range(turns):
        turns_left = turns - i - 1
        low = prev + 2
        high = rect_top - 2 * turns_left - 2
        if high < low:
            return None
        row = random.randint(low, high)
        rows.append(row)
        prev = row
    return rows


def _choose_waypoint_cols(
    start_c: int,
    target_c: int,
    zone: tuple[int, int],
    turns: int,
) -> list[int] | None:
    left, right = zone
    inner_cols = list(range(left + 1, right))
    for _ in range(40):
        cols: list[int] = []
        current = start_c
        ok = True
        for _i in range(turns - 1):
            choices = [c for c in inner_cols if c != current]
            if not choices:
                ok = False
                break
            nxt = random.choice(choices)
            cols.append(nxt)
            current = nxt
        if not ok or current == target_c:
            continue
        cols.append(target_c)
        return cols
    return None


def _route_path(
    height: int,
    width: int,
    start_c: int,
    target_c: int,
    rect_top: int,
    turn_rows: list[int],
    waypoint_cols: list[int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]] | None:
    path_cells: set[tuple[int, int]] = set()
    blocker_cells: set[tuple[int, int]] = set()
    r, c = 0, start_c

    def add_path(cell: tuple[int, int]) -> bool:
        rr, cc = cell
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if cell in blocker_cells:
            return False
        path_cells.add(cell)
        return True

    def add_blocker(cell: tuple[int, int]) -> bool:
        rr, cc = cell
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if cell in path_cells or rr == 0:
            return False
        blocker_cells.add(cell)
        return True

    for turn_row, next_c in zip(turn_rows, waypoint_cols):
        for rr in range(r + 1, turn_row + 1):
            if not add_path((rr, c)):
                return None

        if not add_blocker((turn_row + 1, c)):
            return None

        step = 1 if next_c > c else -1
        wrong_side = (turn_row, c - step)
        if 0 <= wrong_side[1] < width and not add_blocker(wrong_side):
            return None

        for cc in range(c + step, next_c + step, step):
            if not add_path((turn_row, cc)):
                return None

        if not add_blocker((turn_row, next_c + step)):
            return None
        if not add_blocker((turn_row - 1, next_c)):
            return None
        r, c = turn_row, next_c

    for rr in range(r + 1, rect_top):
        if not add_path((rr, target_c)):
            return None

    return path_cells, blocker_cells, (rect_top, target_c)


def _safe_extra_blockers(
    grid: np.ndarray,
    blocker_color: int,
    forbidden: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(w):
            if (r, c) in forbidden or int(grid[r, c]) != _most_common_color(grid):
                continue
            good = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) in forbidden:
                        good = False
                        break
                if not good:
                    break
            if good:
                candidates.append((r, c))
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(0, 3)]:
        grid[r, c] = blocker_color


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(2, 3)
    height = random.randint(_MIN_H, _MAX_H)
    width, zones = _sample_layout(n_rects)

    rect_color = random.choice(_PALETTE)
    blocker_color = random.choice([c for c in _PALETTE if c != rect_color])
    bg = random.choice([c for c in _PALETTE if c not in {rect_color, blocker_color}])
    start_colors = random.sample([c for c in _PALETTE if c not in {bg, rect_color, blocker_color}], n_rects)

    inp = np.full((height, width), bg, dtype=int)
    out = inp.copy()

    used_cells: set[tuple[int, int]] = set()
    path_related: set[tuple[int, int]] = set()

    for zone, start_color in zip(zones, start_colors):
        left, right = zone
        rect_h = random.randint(2, 4)
        rect_w = random.randint(2, 3)
        rect_top = height - rect_h
        rect_left = left + random.randint(1, _ZONE_W - rect_w - 1)
        rect_cells = [(r, c) for r in range(rect_top, height) for c in range(rect_left, rect_left + rect_w)]
        rect_cols = list(range(rect_left, rect_left + rect_w))

        start_choices = [c for c in range(left, right + 1) if c not in rect_cols]
        if not start_choices:
            return None
        start_c = random.choice(start_choices)
        target_c = random.choice(rect_cols)

        turns = random.randint(1, 3)
        turn_rows = _choose_turn_rows(rect_top, turns)
        if turn_rows is None:
            return None
        waypoint_cols = _choose_waypoint_cols(start_c, target_c, zone, turns)
        if waypoint_cols is None:
            return None

        routed = _route_path(height, width, start_c, target_c, rect_top, turn_rows, waypoint_cols)
        if routed is None:
            return None
        path_cells, blocker_cells, hit_cell = routed
        if hit_cell != (rect_top, target_c):
            return None

        occupied = set(rect_cells) | blocker_cells | path_cells | {(0, start_c)}
        if occupied & (used_cells | path_related):
            return None

        for r, c in rect_cells:
            inp[r, c] = rect_color
            out[r, c] = start_color
        for r, c in blocker_cells:
            inp[r, c] = blocker_color
            out[r, c] = blocker_color
        inp[0, start_c] = start_color
        out[0, start_c] = start_color

        used_cells |= set(rect_cells) | blocker_cells | {(0, start_c)}
        path_related |= path_cells | blocker_cells | {(0, start_c)}

    _safe_extra_blockers(inp, blocker_color, path_related | used_cells)
    extra_mask = inp == blocker_color
    out[extra_mask] = blocker_color

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze-exit-label input")
