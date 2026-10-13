from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Door = tuple[str, int]

BG = 0
CYCLE = (2, 4, 3, 1)  # red -> yellow -> green -> blue -> red
NON_CYCLE = (5, 6, 7, 8, 9)
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _u_frame_cells(center: Coord, open_dir: Coord) -> set[Coord]:
    r, c = center
    dr, dc = open_dir
    cells: set[Coord] = set()
    if (dr, dc) in ((0, -1), (0, 1)):
        side_c = c - dc
        for cc in range(c - 1, c + 2):
            cells.add((r - 1, cc))
            cells.add((r + 1, cc))
        cells.add((r, side_c))
    else:
        side_r = r - dr
        for rr in range(r - 1, r + 2):
            cells.add((rr, c - 1))
            cells.add((rr, c + 1))
        cells.add((side_r, c))
    return cells


def _find_u_frame_center(g: np.ndarray, marker: int) -> tuple[Coord, Coord] | None:
    h, w = g.shape
    hits: list[tuple[Coord, Coord]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) == int(marker):
                continue
            for open_dir in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                cells = _u_frame_cells((r, c), open_dir)
                if all(int(g[rr, cc]) == int(marker) for rr, cc in cells):
                    orow, ocol = r + open_dir[0], c + open_dir[1]
                    if int(g[orow, ocol]) != int(marker):
                        hits.append(((r, c), open_dir))
    if len(hits) != 1:
        return None
    return hits[0]


def _infer_roles(g: np.ndarray) -> tuple[int, int, int, Coord, int, Coord]:
    bg = _get_bg(g)
    cycle_cells = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere(np.isin(g, CYCLE))
    ]
    assert len(cycle_cells) == 1, "expected exactly one colored start pixel from the fixed cycle"
    sr, sc, start_color = cycle_cells[0]
    start = (sr, sc)

    colors = set(map(int, np.unique(g))) - {int(bg)} - set(CYCLE)
    marker_hits: list[tuple[int, Coord, Coord]] = []
    for color in colors:
        if int(np.count_nonzero(g == int(color))) != 7:
            continue
        found = _find_u_frame_center(g, int(color))
        if found is not None:
            center, open_dir = found
            marker_hits.append((int(color), center, open_dir))
    assert len(marker_hits) == 1, "expected one seven-pixel uncolored endpoint frame"
    marker, target, _open_dir = marker_hits[0]

    wall_candidates = list(colors - {int(marker)})
    assert len(wall_candidates) == 1, "expected one wall color besides the endpoint frame"
    wall = int(wall_candidates[0])
    assert int(np.count_nonzero(g == wall)) > int(np.count_nonzero(g == marker))
    return int(bg), wall, int(marker), start, int(start_color), target


def _shortest_path_unique(g: np.ndarray, wall: int, marker: int, start: Coord, target: Coord) -> list[Coord]:
    h, w = g.shape
    passable = (g != int(wall)) & (g != int(marker))
    assert bool(passable[start]) and bool(passable[target])

    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=int)
    q: deque[Coord] = deque([start])
    dist[start] = 0
    ways[start] = 1
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc) or not bool(passable[nr, nc]):
                continue
            nd = int(dist[r, c]) + 1
            if int(dist[nr, nc]) == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = int(ways[r, c])
                q.append((nr, nc))
            elif int(dist[nr, nc]) == nd:
                ways[nr, nc] = min(2, int(ways[nr, nc]) + int(ways[r, c]))

    assert int(dist[target]) >= 0, "endpoint interior must be reachable"
    assert int(ways[target]) == 1, "shortest route must be unique"

    path = [target]
    cur = target
    while cur != start:
        r, c = cur
        preds = [
            (r + dr, c + dc)
            for dr, dc in DIR4
            if _in_bounds((h, w), r + dr, c + dc)
            and bool(passable[r + dr, c + dc])
            and int(dist[r + dr, c + dc]) == int(dist[r, c]) - 1
        ]
        assert len(preds) == 1, "unique shortest route must have one predecessor at every step"
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


def _door_info(g: np.ndarray, wall: int, cell: Coord) -> Door | None:
    r, c = cell
    if int(g[r, c]) == int(wall):
        return None

    up = 0
    rr = r - 1
    while rr >= 0 and int(g[rr, c]) == int(wall):
        up += 1
        rr -= 1
    down = 0
    rr = r + 1
    while rr < g.shape[0] and int(g[rr, c]) == int(wall):
        down += 1
        rr += 1

    left = 0
    cc = c - 1
    while cc >= 0 and int(g[r, cc]) == int(wall):
        left += 1
        cc -= 1
    right = 0
    cc = c + 1
    while cc < g.shape[1] and int(g[r, cc]) == int(wall):
        right += 1
        cc += 1

    candidates: list[Door] = []
    if up >= 1 and down >= 1 and up + down + 1 >= 5 and up != down:
        candidates.append(("V", 1 if up < down else -1))
    if left >= 1 and right >= 1 and left + right + 1 >= 5 and left != right:
        candidates.append(("H", 1 if left < right else -1))
    assert len(candidates) <= 1, "a path cell cannot be two polarized door gaps"
    return candidates[0] if candidates else None


def _advance_color(color: int, delta: int) -> int:
    idx = CYCLE.index(int(color))
    return int(CYCLE[(idx + int(delta)) % len(CYCLE)])


def _solve(g: np.ndarray) -> tuple[np.ndarray, list[Coord], list[Door]]:
    bg, wall, marker, start, start_color, target = _infer_roles(g)
    path = _shortest_path_unique(g, wall, marker, start, target)

    out = g.copy()
    current = int(start_color)
    doors: list[Door] = []
    for cell in path:
        door = _door_info(g, wall, cell)
        if door is not None:
            doors.append(door)
            current = _advance_color(current, door[1])
            continue
        r, c = cell
        assert int(g[r, c]) != int(bg) or (r, c) == target or (r, c) in path
        out[r, c] = int(current)
    return out, path, doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the unique endpoint path, changing color only after polarized door gaps.

    Input structure:
    - The background is dominant. One non-cycle color forms thin walls around a one-cell
      route split by several wall-door gaps, and another non-cycle color forms one U-shaped
      endpoint frame whose interior cell is uncolored.
    - Exactly one pixel is colored from the fixed visible cycle red -> yellow -> green -> blue;
      this is the start. Wall gaps are background cells in a thin wall line and each gap is
      offset toward one end of that wall line.

    Rule:
    - Treat wall and endpoint-frame pixels as fixed blockers, and find the unique shortest
      4-connected path from the colored start pixel to the interior of the endpoint frame.
    - Paint non-door cells of that path with the current cycle color, beginning with the
      start color. Door-gap cells stay unchanged.
    - When the path crosses a door gap, inspect the gap's position along that wall's own
      row or column span: a gap in the smaller-coordinate half advances the cycle one step,
      while a gap in the larger-coordinate half retreats one step. The new color applies
      from the next path cell onward.

    Key insight:
    - The color changes count door-crossing events, not walking distance. Each door is a
      polarized switch whose sign comes from the gap's offset within its local wall span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out, _path, _doors = _solve(g)
    return out


def _choose_door_extents(step: int) -> tuple[int, int]:
    small = random.randint(1, 2)
    large = random.randint(3, 5)
    if int(step) == 1:
        return small, large
    return large, small


def _paint_canonical(
    wall: int,
    marker: int,
    start_color: int,
    door_steps: list[int],
    spaces: list[int],
    height: int,
) -> np.ndarray | None:
    max_ext = 5
    if height < 2 * max_ext + 3:
        return None
    r = random.randint(max_ext + 1, height - max_ext - 2)

    door_positions: list[int] = []
    pos = int(spaces[0])
    for i in range(len(door_steps)):
        door_positions.append(pos)
        pos += 1 + int(spaces[i + 1])
    target_pos = pos
    start_col = 2
    target_col = start_col + target_pos
    width = target_col + 3
    if width > 30:
        return None

    g = np.full((height, width), BG, dtype=int)
    cap_c = start_col - 1

    g[r - 1, cap_c : target_col + 2] = int(wall)
    g[r + 1, cap_c : target_col + 2] = int(wall)
    g[r - 1 : r + 2, cap_c] = int(wall)

    for rel_pos, step in zip(door_positions, door_steps):
        c = start_col + rel_pos
        before, after = _choose_door_extents(step)
        g[r - before : r, c] = int(wall)
        g[r + 1 : r + after + 1, c] = int(wall)
        g[r, c] = BG

    g[r, start_col] = int(start_color)

    for rr, cc in _u_frame_cells((r, target_col), (0, -1)):
        g[rr, cc] = int(marker)
    g[r, target_col] = BG
    g[r, target_col - 1] = BG
    return g


def _transform_orientation(g: np.ndarray, vertical: bool, reverse: bool) -> np.ndarray:
    out = g.T.copy() if vertical else g.copy()
    if reverse:
        out = np.flipud(out).copy() if vertical else np.fliplr(out).copy()
    return out


def _semantic_check(g: np.ndarray) -> bool:
    try:
        out, path, doors = _solve(g)
    except AssertionError:
        return False

    if max(g.shape) > 30 or max(out.shape) > 30:
        return False
    if len(doors) < 3:
        return False
    steps = [step for _axis, step in doors]
    if not ({1, -1} <= set(steps)):
        return False
    door_cells = {cell for cell in path if _door_info(g, _infer_roles(g)[1], cell) is not None}
    if len(door_cells) != len(doors):
        return False

    painted = [cell for cell in path if cell not in door_cells]
    if len({int(out[cell]) for cell in painted}) < 2:
        return False

    run_lengths: list[int] = []
    run = 0
    for cell in path:
        if cell in door_cells:
            if run:
                run_lengths.append(run)
            run = 0
        else:
            run += 1
    if run:
        run_lengths.append(run)
    if len(set(run_lengths)) < 2:
        return False

    wall = _infer_roles(g)[1]
    marker = _infer_roles(g)[2]
    fixed = (g == wall) | (g == marker)
    if not np.array_equal(g[fixed], out[fixed]):
        return False
    for r, c in door_cells:
        if int(out[r, c]) != int(g[r, c]):
            return False
    return bool(np.any(out != g))


def _gen_attempt() -> np.ndarray | None:
    wall, marker = random.sample(NON_CYCLE, 2)
    start_color = random.choice(CYCLE)

    door_count = random.randint(3, 5)
    forward_count = random.randint(1, door_count - 1)
    door_steps = [1] * forward_count + [-1] * (door_count - forward_count)
    random.shuffle(door_steps)

    spaces = [random.randint(2, 5)]
    spaces.extend(random.randint(2, 6) for _ in range(door_count))
    if len(set(spaces)) < 2:
        return None

    height = random.randint(13, 19)
    base = _paint_canonical(wall, marker, start_color, door_steps, spaces, height)
    if base is None:
        return None

    vertical = random.choice([False, True])
    reverse = random.choice([False, True])
    grid = _transform_orientation(base, vertical=vertical, reverse=reverse)
    if _semantic_check(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid door-polarity path input")
