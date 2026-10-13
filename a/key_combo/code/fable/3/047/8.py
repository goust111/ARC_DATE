from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _right(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _left(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


TURN_RULES = {RED: _right, BLUE: _left}


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _find_start(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    h, w = g.shape
    start = [tuple(map(int, rc)) for rc in np.argwhere(g == GREEN)]
    assert len(start) == 2, "expected exactly two green start-bar pixels"
    (r0, c0), (r1, c1) = start
    assert abs(r0 - r1) + abs(c0 - c1) == 1, "start bar must be 4-connected"

    border = [p for p in start if _is_border((h, w), *p)]
    assert len(border) == 1, "exactly one start-bar pixel must touch the border"
    border_pixel = border[0]
    inner_pixel = start[1] if start[0] == border_pixel else start[0]

    dr = inner_pixel[0] - border_pixel[0]
    dc = inner_pixel[1] - border_pixel[1]
    assert (dr, dc) in DIRS
    return border_pixel, inner_pixel, int(dr), int(dc)


def _trace(
    grid: np.ndarray,
    turn_rules: dict[int, Callable[[int, int], tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    turn_rules = TURN_RULES if turn_rules is None else turn_rules
    border_pixel, _inner_pixel, dr, dc = _find_start(g)

    out = g.copy()
    r, c = border_pixel
    current_color = GREEN
    hits: list[tuple[int, int, int]] = []
    visits: list[tuple[int, int]] = [(int(r), int(c))]
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (int(r), int(c), int(dr), int(dc))
        assert state not in seen_states, "beam path should not cycle"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        front = int(g[nr, nc])
        if front in turn_rules:
            hits.append((int(nr), int(nc), int(front)))
            current_color = front
            dr, dc = turn_rules[front](dr, dc)
            continue

        out[nr, nc] = current_color
        r, c = int(nr), int(nc)
        visits.append((r, c))

    return out, hits, visits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the boundary-entering beam, turning at red and blue deflectors and recoloring each segment from the last deflector.

    Input structure:
    - The grid has a black background, red and blue single-pixel deflectors, and exactly one 2-pixel green start bar.
    - One green pixel lies on the border and the other lies immediately inside it, giving the incoming direction.
    - Red and blue cells are blockers: the beam can see them in the next cell but never steps onto them.

    Rule:
    - Start at the border green pixel and move through the inner green pixel, painting the first segment green.
    - If the next cell is red, turn clockwise in place; if it is blue, turn counterclockwise in place.
    - After each turn, paint all later traveled cells with that deflector's color until the next turn; later crossings overwrite earlier path colors.
    - Stop when the next step would leave the grid, while leaving all red and blue deflectors unchanged.

    Key insight:
    - Each deflector color is both a geometric instruction and the color inherited by the following beam segment.
    """
    out, _hits, _visits = _trace(np.asarray(grid, dtype=int))
    return out


# --------------- generator ---------------

GEN_H_RANGE = (10, 19)
GEN_W_RANGE = (10, 19)
GEN_TURNS_RANGE = (4, 8)
GEN_SEGMENT_MAX = 7
GEN_FINAL_MIN = 2
GEN_UNIQUE_PATH_MIN = 11
GEN_UNIQUE_PATH_MAX = 70
GEN_EXTRA_DEFLECTORS_RANGE = (1, 5)
GEN_RETRIES = 700
GEN_FORCE_CROSS_PROB = 0.35


def _choose_entry(h: int, w: int) -> tuple[tuple[int, int], int, int]:
    side = random.choice(("top", "right", "bottom", "left"))
    if side == "top":
        return (0, random.randint(1, w - 2)), 1, 0
    if side == "right":
        return (random.randint(1, h - 2), w - 1), 0, -1
    if side == "bottom":
        return (h - 1, random.randint(1, w - 2)), -1, 0
    return (random.randint(1, h - 2), 0), 0, 1


def _ray_cells_until_exit(
    h: int,
    w: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    cells = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells
        cells.append((int(r), int(c)))


def _turn_options(
    h: int,
    w: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    used_colors: set[int],
    turn_index: int,
    num_turns: int,
    path_cells: list[tuple[int, int]],
    deflectors: dict[tuple[int, int], int],
    seen_states: set[tuple[int, int, int, int]],
) -> list[tuple[int, int, list[tuple[int, int]], tuple[int, int], tuple[int, int]]]:
    path_set = set(path_cells)
    deflector_cells = set(deflectors)
    missing = {RED, BLUE} - used_colors
    remaining_after = num_turns - turn_index - 1
    options = []

    for color in (RED, BLUE):
        if len(missing) > remaining_after and color not in missing:
            continue
        ndr, ndc = TURN_RULES[color](dr, dc)

        for seg_len in range(1, GEN_SEGMENT_MAX + 1):
            cells = []
            ok = True
            cr, cc = r, c
            for _ in range(seg_len):
                cr += dr
                cc += dc
                if not (0 <= cr < h and 0 <= cc < w):
                    ok = False
                    break
                if (cr, cc) in deflector_cells:
                    ok = False
                    break
                if (int(cr), int(cc), int(dr), int(dc)) in seen_states:
                    ok = False
                    break
                cells.append((int(cr), int(cc)))
            if not ok:
                continue

            br, bc = cr + dr, cc + dc
            if not (1 <= br < h - 1 and 1 <= bc < w - 1):
                continue
            if (br, bc) in path_set or (br, bc) in cells or (br, bc) in deflector_cells:
                continue
            if any(abs(br - rr) + abs(bc - cc2) == 1 for rr, cc2 in deflector_cells):
                continue

            first_after_turn = (cr + ndr, cc + ndc)
            if not (0 <= first_after_turn[0] < h and 0 <= first_after_turn[1] < w):
                continue
            if first_after_turn in deflector_cells or first_after_turn == (br, bc):
                continue

            new_state = (int(cr), int(cc), int(ndr), int(ndc))
            if new_state in seen_states:
                continue
            options.append((int(color), int(seg_len), cells, (int(br), int(bc)), (int(ndr), int(ndc))))

    return options


def _build_route(
    h: int,
    w: int,
    num_turns: int,
) -> tuple[tuple[int, int], int, int, list[tuple[int, int]], dict[tuple[int, int], int], list[int]] | None:
    border, dr, dc = _choose_entry(h, w)
    r, c = border
    path_cells = [(int(r), int(c))]
    deflectors: dict[tuple[int, int], int] = {}
    seen_states = {(int(r), int(c), int(dr), int(dc))}
    hit_colors: list[int] = []
    used_colors: set[int] = set()

    for turn_index in range(num_turns):
        options = _turn_options(
            h, w, r, c, dr, dc, used_colors, turn_index, num_turns,
            path_cells, deflectors, seen_states
        )
        if not options:
            return None

        random.shuffle(options)
        # Prefer real self-crossings once a path is established, but do not force them.
        crossing = [opt for opt in options if any(cell in set(path_cells) for cell in opt[2])]
        if crossing and random.random() < 0.45:
            color, _seg_len, cells, blocker, ndir = random.choice(crossing)
        else:
            color, _seg_len, cells, blocker, ndir = random.choice(options)

        for cell in cells:
            path_cells.append(cell)
            seen_states.add((cell[0], cell[1], int(dr), int(dc)))

        deflectors[blocker] = int(color)
        hit_colors.append(int(color))
        used_colors.add(int(color))
        r, c = cells[-1]
        dr, dc = ndir
        seen_states.add((int(r), int(c), int(dr), int(dc)))

    final_cells = _ray_cells_until_exit(h, w, r, c, dr, dc)
    if len(final_cells) < GEN_FINAL_MIN:
        return None
    if any(cell in deflectors for cell in final_cells):
        return None
    for cell in final_cells:
        state = (cell[0], cell[1], int(dr), int(dc))
        if state in seen_states:
            return None
        path_cells.append(cell)
        seen_states.add(state)

    unique_path = len(set(path_cells))
    if not (GEN_UNIQUE_PATH_MIN <= unique_path <= GEN_UNIQUE_PATH_MAX):
        return None
    if set(hit_colors) != {RED, BLUE}:
        return None
    return border, int(dr), int(dc), path_cells, deflectors, hit_colors


def _input_from_route(
    h: int,
    w: int,
    border: tuple[int, int],
    initial_dr: int,
    initial_dc: int,
    deflectors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[border] = GREEN
    grid[border[0] + initial_dr, border[1] + initial_dc] = GREEN
    for (r, c), color in deflectors.items():
        grid[r, c] = int(color)
    return grid


def _place_extra_deflectors(
    grid: np.ndarray,
    route_cells: set[tuple[int, int]],
    route_deflectors: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    target = random.randint(*GEN_EXTRA_DEFLECTORS_RANGE)
    placed = 0
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if placed >= target:
            return
        if (r, c) in route_cells or (r, c) in route_deflectors:
            continue
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in route_cells):
            continue
        if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in route_deflectors):
            continue
        if any(
            _in_bounds(grid.shape, r + dr, c + dc) and int(grid[r + dr, c + dc]) in (RED, BLUE)
            for dr, dc in DIRS
        ):
            continue
        if grid[r, c] != BLACK:
            continue
        grid[r, c] = random.choice((RED, BLUE))
        placed += 1


def _path_color_counts(inp: np.ndarray, out: np.ndarray) -> dict[int, int]:
    counts = {GREEN: 0, RED: 0, BLUE: 0}
    deflectors = (inp == RED) | (inp == BLUE)
    changed_or_path = (out != BLACK) & ~deflectors
    for color in counts:
        counts[color] = int(np.count_nonzero(changed_or_path & (out == color)))
    return counts


def _semantic_accepts(inp: np.ndarray, expected: np.ndarray, want_cross: bool) -> bool:
    try:
        solved, hits, visits = _trace(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if len(hits) < GEN_TURNS_RANGE[0] or {hit[2] for hit in hits} != {RED, BLUE}:
        return False
    if want_cross and len(visits) == len(set(visits)):
        return False

    counts = _path_color_counts(inp, expected)
    if counts[GREEN] < 3 or counts[RED] < 2 or counts[BLUE] < 2:
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.all(expected[inp == RED] == RED) or not np.all(expected[inp == BLUE] == BLUE):
        return False

    swapped = {RED: _left, BLUE: _right}
    all_right = {RED: _right, BLUE: _right}
    all_left = {RED: _left, BLUE: _left}
    for rules in (swapped, all_right, all_left):
        try:
            alt, _alt_hits, _alt_visits = _trace(inp, rules)
        except Exception:
            continue
        if np.array_equal(alt, expected):
            return False
    return True


def _gen_attempt(want_cross: bool) -> np.ndarray | None:
    h = random.randint(*GEN_H_RANGE)
    w = random.randint(*GEN_W_RANGE)
    num_turns = random.randint(*GEN_TURNS_RANGE)
    built = _build_route(h, w, num_turns)
    if built is None:
        return None

    border, final_dr, final_dc, path_cells, deflectors, _hit_colors = built
    # Recover the entry vector from the border and its unique neighboring path cell.
    del final_dr, final_dc
    first_step = path_cells[1]
    initial_dr = first_step[0] - border[0]
    initial_dc = first_step[1] - border[1]
    inp = _input_from_route(h, w, border, initial_dr, initial_dc, deflectors)
    _place_extra_deflectors(inp, set(path_cells), set(deflectors))

    try:
        expected, _hits, _visits = _trace(inp)
    except Exception:
        return None
    if not _semantic_accepts(inp, expected, want_cross):
        return None
    return inp


def generate_input() -> np.ndarray:
    want_cross = random.random() < GEN_FORCE_CROSS_PROB
    for attempt in range(GEN_RETRIES):
        result = _gen_attempt(want_cross)
        if result is not None:
            return result
        if attempt == GEN_RETRIES // 2:
            want_cross = False
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
