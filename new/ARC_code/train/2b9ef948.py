from __future__ import annotations

from collections import deque

import numpy as np

from arc_utils import Color, draw_line, get_bg, neighbors4
import random

Coord = tuple[int, int]


def _gray_path_connects(g: np.ndarray, start: Coord, end: Coord) -> bool:
    seen = {start}
    q = deque([start])

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                continue
            p = (int(rr), int(cc))
            if p in seen:
                continue
            if p == end or int(g[rr, cc]) == Color.GREY:
                seen.add(p)
                q.append(p)

    gray_seen = sum(1 for r, c in seen if int(g[r, c]) == Color.GREY)
    return end in seen and gray_seen == int((g == Color.GREY).sum())


def _find_frame_center(g: np.ndarray, control: int) -> Coord:
    centers: list[Coord] = []
    border = np.ones((3, 3), dtype=bool)
    border[1, 1] = False

    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            if int(g[r, c]) != int(control):
                continue
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            if bool((patch[border] == Color.YELLOW).all()):
                centers.append((int(r), int(c)))

    assert len(centers) == 1
    return centers[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the yellow 3x3 frame by the displacement encoded by the gray path, then draw its diagonal yellow corner rays.

    Input structure:
    - The background is black.
    - One 3x3 yellow frame has a non-yellow control-color center.
    - A second control-color pixel is connected through gray path pixels to a lone yellow endpoint.

    Rule:
    - Use the vector from the second control-color pixel to the lone yellow endpoint as the frame's translation.
    - Fill the whole output with the control color.
    - Draw the translated yellow frame and extend yellow diagonal rays outward from its four corners to the grid edge.

    Key insight: The gray path's turns are irrelevant; its two colored endpoints encode the net displacement to apply to the frame.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    bg = get_bg(g)
    assert bg == Color.BLACK

    colors = set(map(int, np.unique(g)))
    role_colors = {int(bg), Color.YELLOW, Color.GREY}
    control_colors = colors - role_colors
    assert len(control_colors) == 1
    control = control_colors.pop()
    assert int((g == control).sum()) == 2
    assert int((g == Color.GREY).sum()) > 0

    frame_center = _find_frame_center(g, control)
    fr, fc = frame_center
    frame_yellow = {
        (r, c)
        for r in range(fr - 1, fr + 2)
        for c in range(fc - 1, fc + 2)
        if (r, c) != frame_center
    }

    yellow_cells = set(map(tuple, np.argwhere(g == Color.YELLOW)))
    endpoints = yellow_cells - frame_yellow
    assert len(endpoints) == 1
    endpoint = endpoints.pop()

    control_cells = set(map(tuple, np.argwhere(g == control)))
    path_controls = control_cells - {frame_center}
    assert len(path_controls) == 1
    path_control = path_controls.pop()
    assert _gray_path_connects(g, path_control, endpoint)

    dr = int(endpoint[0] - path_control[0])
    dc = int(endpoint[1] - path_control[1])
    out_center = (int(fr + dr), int(fc + dc))
    r, c = out_center
    assert 1 <= r < H - 1 and 1 <= c < W - 1

    out = np.full_like(g, int(control))
    for rr, cc, direction in (
        (r - 1, c - 1, (-1, -1)),
        (r - 1, c + 1, (-1, 1)),
        (r + 1, c - 1, (1, -1)),
        (r + 1, c + 1, (1, 1)),
    ):
        draw_line(out, rr, cc, direction=direction, color=Color.YELLOW)

    out[r - 1 : r + 2, c - 1 : c + 2] = Color.YELLOW
    out[r, c] = int(control)
    return out

# --------------- generator (o5x) ---------------

# CFG: colors.
FRAME_COLOR = Color.YELLOW
PATH_COLOR = Color.GREY
CONTROL_COLORS = [Color.BLUE, Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL]

# CFG: canvas and frame placement.
ROW_RANGE, COL_RANGE = (12, 30), (13, 30)
OUTPUT_CENTER_MARGIN, FRAME_EDGE_MARGIN, FRAME_CLEARANCE = 2, 1, 4
FRAME_CORNER_RAYS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# CFG: displacement and connector-path shape.
DISPLACEMENT_MAG_RANGE = (4, 9)
DIRECT_PATH_MIN_MAG, DETOUR_PATH_MAX_MAG = 5, 7
PATH_LEN_RANGE = (6, 10)
MAX_PATH_TURNS, DETOUR_OFFSETS = 2, (-1, 1)

# CFG: rejection sampling.
MAX_GENERATION_ATTEMPTS = 300


def _sgn(x: int) -> int:
    return 1 if x > 0 else -1


def _path_from_waypoints(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = 0, 0
    path = [(r, c)]
    for nr, nc in waypoints:
        if r != nr and c != nc:
            raise ValueError("waypoints must form cardinal segments")
        while r != nr:
            r += _sgn(nr - r)
            path.append((r, c))
        while c != nc:
            c += _sgn(nc - c)
            path.append((r, c))
    return path


def _path_turns(path: list[tuple[int, int]]) -> int:
    dirs: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((_sgn(r1 - r0) if r1 != r0 else 0, _sgn(c1 - c0) if c1 != c0 else 0))
    return sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)


def _path_offsets(dr: int, dc: int) -> list[tuple[int, int]]:
    m = abs(dr) + abs(dc)
    candidates: list[list[tuple[int, int]]] = []

    if m >= DIRECT_PATH_MIN_MAG:
        if dr == 0 or dc == 0:
            candidates.append(_path_from_waypoints([(dr, dc)]))
        else:
            candidates.append(_path_from_waypoints([(dr, 0), (dr, dc)]))
            candidates.append(_path_from_waypoints([(0, dc), (dr, dc)]))
            if abs(dr) >= 2:
                split = random.randint(1, abs(dr) - 1) * _sgn(dr)
                candidates.append(_path_from_waypoints([(split, 0), (split, dc), (dr, dc)]))
            if abs(dc) >= 2:
                split = random.randint(1, abs(dc) - 1) * _sgn(dc)
                candidates.append(_path_from_waypoints([(0, split), (dr, split), (dr, dc)]))

    if m <= DETOUR_PATH_MAX_MAG:
        if dr == 0:
            off = random.choice(DETOUR_OFFSETS)
            candidates.append(_path_from_waypoints([(off, 0), (off, dc), (0, dc)]))
            candidates.append(_path_from_waypoints([(-off, 0), (-off, dc), (0, dc)]))
        elif dc == 0:
            off = random.choice(DETOUR_OFFSETS)
            candidates.append(_path_from_waypoints([(0, off), (dr, off), (dr, 0)]))
            candidates.append(_path_from_waypoints([(0, -off), (dr, -off), (dr, 0)]))
        else:
            sc = _sgn(dc)
            sr = _sgn(dr)
            candidates.append(_path_from_waypoints([(0, -sc), (dr, -sc), (dr, dc)]))
            candidates.append(_path_from_waypoints([(0, dc + sc), (dr, dc + sc), (dr, dc)]))
            candidates.append(_path_from_waypoints([(-sr, 0), (-sr, dc), (dr, dc)]))
            candidates.append(_path_from_waypoints([(dr + sr, 0), (dr + sr, dc), (dr, dc)]))

    valid = [
        p
        for p in candidates
        if PATH_LEN_RANGE[0] <= len(p) <= PATH_LEN_RANGE[1]
        and len(set(p)) == len(p)
        and _path_turns(p) <= MAX_PATH_TURNS
    ]
    if not valid:
        raise ValueError("no valid path shape")
    return random.choice(valid)


def _draw_frame(grid: np.ndarray, r: int, c: int, control: int) -> None:
    grid[r - 1 : r + 2, c - 1 : c + 2] = FRAME_COLOR
    grid[r, c] = int(control)


def _make_output(H: int, W: int, control: int, r: int, c: int) -> np.ndarray:
    out = np.full((H, W), int(control), dtype=int)
    for dr, dc in FRAME_CORNER_RAYS:
        draw_line(out, r + dr, c + dc, direction=(dr, dc), color=FRAME_COLOR)
    _draw_frame(out, r, c, control)
    return out


def _sample_output_scene() -> tuple[int, int, int, int, int, np.ndarray]:
    H, W = random.randint(*ROW_RANGE), random.randint(*COL_RANGE)
    control = random.choice(CONTROL_COLORS)
    out_r = random.randint(OUTPUT_CENTER_MARGIN, H - OUTPUT_CENTER_MARGIN - 1)
    out_c = random.randint(OUTPUT_CENTER_MARGIN, W - OUTPUT_CENTER_MARGIN - 1)
    expected = _make_output(H, W, control, out_r, out_c)
    return H, W, control, out_r, out_c, expected


def _valid_displacements(H: int, W: int, r: int, c: int, mag: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for dr in range(-mag, mag + 1):
        dc_abs = mag - abs(dr)
        dc_values = [0] if dc_abs == 0 else [-dc_abs, dc_abs]
        for dc in dc_values:
            fr, fc = r - dr, c - dc
            if FRAME_EDGE_MARGIN <= fr < H - FRAME_EDGE_MARGIN and FRAME_EDGE_MARGIN <= fc < W - FRAME_EDGE_MARGIN:
                choices.append((dr, dc))
    return choices


def _sample_frame_displacement(H: int, W: int, out_r: int, out_c: int) -> tuple[int, int] | None:
    mag = random.randint(*DISPLACEMENT_MAG_RANGE)
    displacements = _valid_displacements(H, W, out_r, out_c, mag)
    if not displacements:
        return None
    return random.choice(displacements)


def _valid_starts(
    H: int,
    W: int,
    offsets: list[tuple[int, int]],
    frame_r: int,
    frame_c: int,
) -> list[tuple[int, int]]:
    rs = [r for r, _ in offsets]
    cs = [c for _, c in offsets]
    r_lo, r_hi = -min(rs), H - 1 - max(rs)
    c_lo, c_hi = -min(cs), W - 1 - max(cs)
    if r_lo > r_hi or c_lo > c_hi:
        return []

    starts: list[tuple[int, int]] = []
    for r0 in range(r_lo, r_hi + 1):
        for c0 in range(c_lo, c_hi + 1):
            ok = True
            for rr, cc in offsets:
                ar, ac = r0 + rr, c0 + cc
                if max(abs(ar - frame_r), abs(ac - frame_c)) < FRAME_CLEARANCE:
                    ok = False
                    break
            if ok:
                starts.append((r0, c0))
    random.shuffle(starts)
    return starts


def _place_path(
    grid: np.ndarray, path_r: int, path_c: int, offsets: list[tuple[int, int]], control: int
) -> None:
    path = [(path_r + rr, path_c + cc) for rr, cc in offsets]
    grid[path[0]] = int(control)
    for cell in path[1:-1]:
        grid[cell] = PATH_COLOR
    grid[path[-1]] = FRAME_COLOR


def _make_input(
    H: int, W: int, control: int, frame_r: int, frame_c: int, offsets: list[tuple[int, int]]
) -> np.ndarray | None:
    starts = _valid_starts(H, W, offsets, frame_r, frame_c)
    if not starts:
        return None

    inp = np.zeros((H, W), dtype=int)
    _draw_frame(inp, frame_r, frame_c, control)
    path_r, path_c = starts[0]
    _place_path(inp, path_r, path_c, offsets, control)
    return inp


def _gen_attempt() -> np.ndarray | None:
    H, W, control, out_r, out_c, expected = _sample_output_scene()

    displacement = _sample_frame_displacement(H, W, out_r, out_c)
    if displacement is None:
        return None
    dr, dc = displacement
    frame_r, frame_c = out_r - dr, out_c - dc
    try:
        offsets = _path_offsets(dr, dc)
    except ValueError:
        return None

    inp = _make_input(H, W, control, frame_r, frame_c, offsets)
    if inp is None:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2b9ef948 input")
