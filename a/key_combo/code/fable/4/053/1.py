from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]

SEP_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
BOUNCE_COUNT_CHOICES = [0, 1, 2, 3, 4]
BOUNCE_COUNT_WEIGHTS = [1, 3, 3, 2, 1]
EVENT_TYPE_WEIGHTS = {"corner": 3, "row": 2, "col": 2}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_arrow(cells: list[Coord]) -> bool:
    if len(cells) != 3:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    return (r1 - r0 == 1) and (c1 - c0 == 1)


def _arrow_start_and_dir(cells: list[Coord]) -> tuple[Coord, Coord]:
    pts = set(cells)
    r0, c0, r1, c1 = _bbox(cells)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = list(corners - pts)
    assert len(missing) == 1
    mr, mc = missing[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    degrees: dict[Coord, int] = {}
    for r, c in cells:
        deg = 0
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts:
                deg += 1
        degrees[(r, c)] = deg
    starts = [cell for cell, deg in degrees.items() if deg == 2]
    assert len(starts) == 1
    return starts[0], (int(dr), int(dc))


def _mirror_col(left_col: int, sep: int) -> int:
    return 2 * sep - int(left_col)


def _left_in_bounds(h: int, sep: int, pos: Coord) -> bool:
    r, c = pos
    return 0 <= int(r) < h and 0 <= int(c) < sep


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    assert w % 2 == 1
    sep = w // 2
    sep_color = int(grid[0, sep])
    assert np.all(grid[:, sep] == sep_color)
    return sep, sep_color


def _find_arrow(grid: np.ndarray, sep: int, bg: int, sep_color: int) -> tuple[int, list[Coord]]:
    left = grid[:, :sep]
    colors = sorted(int(c) for c in np.unique(left) if int(c) not in {bg, sep_color})
    arrows: list[tuple[int, list[Coord]]] = []
    for color in colors:
        for comp in _components4(left == color):
            global_comp = [(int(r), int(c)) for r, c in comp]
            if _is_arrow(global_comp):
                arrows.append((int(color), global_comp))
    assert len(arrows) == 1
    return arrows[0]


def _hidden_walls(grid: np.ndarray, sep: int, bg: int, sep_color: int) -> dict[Coord, int]:
    h, w = grid.shape
    walls: dict[Coord, int] = {}
    for r in range(h):
        for c in range(sep + 1, w):
            color = int(grid[r, c])
            if color in {bg, sep_color}:
                continue
            walls[(int(r), int(_mirror_col(c, sep)))] = color
    return walls


def _trace_ray(
    grid: np.ndarray,
    start: Coord,
    direction: Coord,
    start_color: int,
    hidden_walls: dict[Coord, int],
    sep: int,
    bg: int,
) -> tuple[np.ndarray, list[tuple[str, Coord, int]], set[Coord]]:
    out = np.array(grid, dtype=int, copy=True)
    h, _w = out.shape
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(direction[0]), int(direction[1])
    ray_color = int(start_color)
    seen: set[tuple[Coord, int, int, int]] = set()
    events: list[tuple[str, Coord, int]] = []
    path_cells: set[Coord] = set()

    while True:
        state = (cur, int(dr), int(dc), int(ray_color))
        if state in seen:
            break
        seen.add(state)

        nxt = (int(cur[0] + dr), int(cur[1] + dc))
        if not _left_in_bounds(h, sep, nxt):
            break

        if nxt in hidden_walls:
            wall_color = int(hidden_walls[nxt])
            out[nxt] = wall_color
            ray_color = wall_color
            dr, dc = -dr, -dc
            events.append(("corner", nxt, wall_color))
            continue

        cur = nxt
        path_cells.add(cur)
        if int(out[cur]) == bg:
            out[cur] = ray_color

        row_wall = (int(cur[0] + dr), int(cur[1]))
        if _left_in_bounds(h, sep, row_wall) and row_wall in hidden_walls:
            wall_color = int(hidden_walls[row_wall])
            out[row_wall] = wall_color
            ray_color = wall_color
            dr = -dr
            events.append(("row", row_wall, wall_color))

        col_wall = (int(cur[0]), int(cur[1] + dc))
        if _left_in_bounds(h, sep, col_wall) and col_wall in hidden_walls:
            wall_color = int(hidden_walls[col_wall])
            out[col_wall] = wall_color
            ray_color = wall_color
            dc = -dc
            events.append(("col", col_wall, wall_color))

    return out, events, path_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a diagonal ray through the empty half, reflecting from separator-mirrored phantom walls and revealing every wall it hits.

    Input structure:
    - A full-height separator column splits the grid into equal left and right halves.
    - The left half contains exactly one monochrome 3-pixel L arrow and is otherwise background.
    - The right half contains scattered single-pixel colored walls whose mirrored positions in the left half are hidden phantom walls.

    Rule:
    - Reflect every visible right-half wall across the separator to obtain the hidden left-half wall set.
    - Fire a diagonal ray from the arrow's inner corner in the direction away from the arrow's missing corner, painting each visited background cell with the current ray color.
    - A diagonal head-on hit flips both axes; a side scrape flips only the touched axis; each hit reveals the corresponding hidden wall cell in its wall color and the ray continues in that new color.
    - Stop when the ray would leave the left half or step onto the separator; keep the separator and the whole right half unchanged.

    Key insight:
    - The left half's true obstacles are never drawn in the input; they must be reconstructed by mirroring the visible right-half wall pixels across the separator before tracing the ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    sep, sep_color = _find_separator(g)
    arrow_color, arrow_cells = _find_arrow(g, sep, bg, sep_color)
    start, direction = _arrow_start_and_dir(arrow_cells)
    hidden = _hidden_walls(g, sep, bg, sep_color)
    out, _events, _path = _trace_ray(g, start, direction, arrow_color, hidden, sep, bg)
    return out


def _make_arrow_cells(start: Coord, direction: Coord) -> list[Coord]:
    sr, sc = start
    dr, dc = direction
    mr, mc = sr - dr, sc - dc
    return [
        (int(r), int(c))
        for r in (mr, sr)
        for c in (mc, sc)
        if (int(r), int(c)) != (int(mr), int(mc))
    ]


def _forward_room(h: int, sep: int, pos: Coord, direction: Coord) -> int:
    r, c = pos
    dr, dc = direction
    row_room = (h - 1 - r) if dr == 1 else r
    col_room = (sep - 1 - c) if dc == 1 else c
    return int(min(row_room, col_room))


def _existing_side_hit(pos: Coord, direction: Coord, walls: dict[Coord, int], h: int, sep: int) -> bool:
    r, c = pos
    dr, dc = direction
    return (
        (_left_in_bounds(h, sep, (r + dr, c)) and (r + dr, c) in walls)
        or (_left_in_bounds(h, sep, (r, c + dc)) and (r, c + dc) in walls)
    )


def _choose_weighted_event(candidates: list[tuple[str, int, Coord]]) -> tuple[str, int, Coord]:
    weights = [EVENT_TYPE_WEIGHTS[event_type] for event_type, _steps, _wall in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def _event_candidates(
    h: int,
    sep: int,
    cur: Coord,
    direction: Coord,
    arrow_cells: set[Coord],
    path_cells: set[Coord],
    walls: dict[Coord, int],
) -> list[tuple[str, int, Coord]]:
    dr, dc = direction
    blocked = set(arrow_cells) | set(path_cells) | set(walls)
    max_steps = _forward_room(h, sep, cur, direction)
    candidates: list[tuple[str, int, Coord]] = []
    trail: list[Coord] = []

    for steps in range(1, max_steps + 1):
        pos = (int(cur[0] + dr * steps), int(cur[1] + dc * steps))
        if pos in blocked:
            break
        if _existing_side_hit(pos, direction, walls, h, sep):
            break
        trail.append(pos)
        last = trail[-1]

        diag_wall = (int(last[0] + dr), int(last[1] + dc))
        if steps < max_steps and _left_in_bounds(h, sep, diag_wall) and diag_wall not in blocked:
            candidates.append(("corner", steps, diag_wall))

        row_wall = (int(last[0] + dr), int(last[1]))
        col_wall = (int(last[0]), int(last[1] + dc))
        if _left_in_bounds(h, sep, row_wall) and row_wall not in blocked and col_wall not in walls:
            candidates.append(("row", steps, row_wall))
        if _left_in_bounds(h, sep, col_wall) and col_wall not in blocked and row_wall not in walls:
            candidates.append(("col", steps, col_wall))

    return candidates


def _paint_segment(path_cells: set[Coord], cur: Coord, direction: Coord, steps: int) -> Coord:
    dr, dc = direction
    last = cur
    for step in range(1, steps + 1):
        last = (int(cur[0] + dr * step), int(cur[1] + dc * step))
        path_cells.add(last)
    return last


def _safe_extra_candidates(h: int, sep: int, reserved: set[Coord]) -> list[Coord]:
    candidates: list[Coord] = []
    for r in range(h):
        for c in range(sep):
            pos = (int(r), int(c))
            if pos in reserved:
                continue
            keep = True
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) in reserved:
                        keep = False
                        break
                if not keep:
                    break
            if keep:
                candidates.append(pos)
    return candidates


def _build_input_grid(
    h: int,
    sep: int,
    bg: int,
    arrow_color: int,
    arrow_cells: list[Coord],
    walls: dict[Coord, int],
) -> np.ndarray:
    w = 2 * sep + 1
    grid = np.full((h, w), int(bg), dtype=int)
    grid[:, sep] = SEP_COLOR
    for r, c in arrow_cells:
        grid[r, c] = int(arrow_color)
    for (r, c_left), color in walls.items():
        grid[r, _mirror_col(c_left, sep)] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 20)
    sep = random.randint(6, 14)
    bg = random.choice([color for color in range(10) if color != SEP_COLOR])
    palette = [color for color in range(10) if color not in {bg, SEP_COLOR}]
    arrow_color = random.choice(palette)
    direction = random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    start_row_min = 1 if direction[0] == 1 else 0
    start_row_max = h - 1 if direction[0] == 1 else h - 2
    start_col_min = 1 if direction[1] == 1 else 0
    start_col_max = sep - 1 if direction[1] == 1 else sep - 2
    if start_row_min > start_row_max or start_col_min > start_col_max:
        return None

    starts = [
        (r, c)
        for r in range(start_row_min, start_row_max + 1)
        for c in range(start_col_min, start_col_max + 1)
        if _forward_room(h, sep, (r, c), direction) >= 2
    ]
    if not starts:
        return None
    start = random.choice(starts)
    arrow_cells = _make_arrow_cells(start, direction)

    n_bounces = random.choices(BOUNCE_COUNT_CHOICES, weights=BOUNCE_COUNT_WEIGHTS, k=1)[0]
    path_cells: set[Coord] = set()
    hidden_walls: dict[Coord, int] = {}
    cur = start
    ray_dir = direction
    planned_events = 0

    for _ in range(n_bounces):
        candidates = _event_candidates(h, sep, cur, ray_dir, set(arrow_cells), path_cells, hidden_walls)
        if not candidates:
            break
        event_type, steps, wall_pos = _choose_weighted_event(candidates)
        cur = _paint_segment(path_cells, cur, ray_dir, steps)
        hidden_walls[wall_pos] = random.choice(palette)
        planned_events += 1
        if event_type == "corner":
            ray_dir = (-ray_dir[0], -ray_dir[1])
        elif event_type == "row":
            ray_dir = (-ray_dir[0], ray_dir[1])
        else:
            ray_dir = (ray_dir[0], -ray_dir[1])

    if planned_events == 0 and random.random() < 0.75:
        return None

    base_grid = _build_input_grid(h, sep, bg, arrow_color, arrow_cells, hidden_walls)
    expected, events, path = _trace_ray(base_grid, start, direction, arrow_color, hidden_walls, sep, bg)

    if len(events) != planned_events:
        return None
    if len(path) < 3:
        return None

    reserved = set(arrow_cells) | set(path) | set(hidden_walls)
    extra_limit = 3 if planned_events > 0 else 4
    extra_min = 1 if planned_events == 0 else 0
    extra_count = random.randint(extra_min, extra_limit)
    extras = _safe_extra_candidates(h, sep, reserved)
    random.shuffle(extras)
    for pos in extras[:extra_count]:
        hidden_walls[pos] = random.choice(palette)

    grid = _build_input_grid(h, sep, bg, arrow_color, arrow_cells, hidden_walls)
    expected_after_extras, _extra_events, _extra_path = _trace_ray(
        grid, start, direction, arrow_color, hidden_walls, sep, bg
    )
    actual = main(grid)
    if not np.array_equal(actual[:, sep + 1 :], grid[:, sep + 1 :]):
        return None
    if not np.array_equal(actual[:, sep], grid[:, sep]):
        return None
    if not np.array_equal(actual, expected_after_extras):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
