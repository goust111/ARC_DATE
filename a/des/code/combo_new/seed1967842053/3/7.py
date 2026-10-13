import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _side_dirs(d):
    dr, dc = d
    return [(dc, -dr), (-dc, dr)]


def _perp_dirs(d):
    dr, dc = d
    return [(dc, -dr), (-dc, dr)]


def _expand_section(mask, seeds, direction):
    seeds = [p for p in seeds if _in_bounds(mask.shape, p[0], p[1]) and mask[p]]
    if not seeds:
        return []
    p1, p2 = _perp_dirs(direction)
    cells = set()
    for seed in seeds:
        comp = {seed}
        for step in (p1, p2):
            cur = seed
            while True:
                cur = _add(cur, step)
                if not _in_bounds(mask.shape, cur[0], cur[1]) or not mask[cur]:
                    break
                comp.add(cur)
        if len(comp) > len(cells):
            cells = comp
    return sorted(cells)


def _find_signal(grid):
    colors = [int(x) for x in np.unique(grid) if x not in (BLACK, BLUE)]
    for color in colors:
        pts = np.argwhere(grid == color)
        if len(pts) < 4:
            continue
        seen = set()
        point_set = {tuple(p) for p in pts}
        for start in list(point_set):
            if start in seen:
                continue
            q = deque([start])
            seen.add(start)
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r + d[0], c + d[1])
                    if nb in point_set and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if (
                len(comp) == 4
                and max(rs) - min(rs) == 1
                and max(cs) - min(cs) == 1
            ):
                return color, sorted(comp)
    if colors:
        color = colors[0]
        comp = [tuple(p) for p in np.argwhere(grid == color)]
        return color, comp
    return None, []


def _initial_state(grid, block):
    block_set = set(block)
    mask = grid == BLUE
    best = None
    for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        touched = []
        for r, c in block:
            nb = (r + d[0], c + d[1])
            if nb not in block_set and _in_bounds(grid.shape, nb[0], nb[1]) and mask[nb]:
                touched.append(nb)
        if touched and (best is None or len(touched) > len(best[1])):
            best = (d, touched)
    if best is None:
        return None
    d, touched = best
    section = _expand_section(mask, touched, d)
    return d, section


def _fill_slot(out, section, direction, color):
    for r, c in section:
        for depth in (1, 2):
            nr = r + direction[0] * depth
            nc = c + direction[1] * depth
            if _in_bounds(out.shape, nr, nc) and out[nr, nc] == BLACK:
                out[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Route a colored signal through blue pipe branches and fill selected slots.

    Input structure:
    A black background contains one connected blue pipe network made from
    horizontal and vertical channels of width one to three. One pipe endpoint
    touches a single 2x2 non-blue signal block. Other pipe endpoints face black
    collection slots two pixels deep, visually framed by blue walls.

    Rule:
    The signal starts at the pipe endpoint touching the colored block and moves
    away from that block along the pipe. At each T junction, all outgoing side
    branches are measured by their cross-section width, and the signal enters
    only the strictly widest branch; if several branches tie for greatest
    width, the signal copies into all tied branches. When a signal reaches a
    dead end, the two-pixel-deep black slot immediately ahead of that pipe end
    is filled with the signal color. Blue pipe pixels are preserved, and the
    original signal block is erased to black.

    Key insight:
    The path is not chosen by color or position labels; the visible thickness
    of each branch is the routing cue, so the pipe topology and branch widths
    must be read together.
    """
    out = np.array(grid, copy=True)
    color, block = _find_signal(grid)
    if color is None or not block:
        return out
    for p in block:
        out[p] = BLACK

    state = _initial_state(grid, block)
    if state is None:
        return out
    mask = grid == BLUE
    stack = [state]
    visited = set()

    while stack:
        direction, section = stack.pop()
        if not section:
            continue
        for _ in range(grid.size + 5):
            key = (direction, tuple(section))
            if key in visited:
                break
            visited.add(key)

            sec_set = set(section)
            side_candidates = []
            for sd in _side_dirs(direction):
                seeds = []
                for p in section:
                    nb = _add(p, sd)
                    if (
                        nb not in sec_set
                        and _in_bounds(grid.shape, nb[0], nb[1])
                        and mask[nb]
                    ):
                        seeds.append(nb)
                cand = _expand_section(mask, seeds, sd)
                if cand:
                    side_candidates.append((sd, cand))

            if side_candidates:
                max_width = max(len(cand) for _, cand in side_candidates)
                for sd, cand in side_candidates:
                    if len(cand) == max_width:
                        stack.append((sd, cand))
                break

            seeds = []
            for p in section:
                nb = _add(p, direction)
                if (
                    nb not in sec_set
                    and _in_bounds(grid.shape, nb[0], nb[1])
                    and mask[nb]
                ):
                    seeds.append(nb)
            forward = _expand_section(mask, seeds, direction)
            if forward:
                section = forward
                continue

            _fill_slot(out, section, direction, color)
            break

    return out


def _draw_horizontal_slot(g, r0, r1, end_col, direction):
    if direction < 0:
        slot_cols = [end_col - 1, end_col - 2]
        cap_col = end_col - 3
    else:
        slot_cols = [end_col + 1, end_col + 2]
        cap_col = end_col + 3
    wall_rows = [r0 - 1, r1 + 1]
    for wr in wall_rows:
        if 0 <= wr < g.shape[0]:
            for c in slot_cols:
                if 0 <= c < g.shape[1]:
                    g[wr, c] = BLUE
    if 0 <= cap_col < g.shape[1]:
        for r in range(r0 - 1, r1 + 2):
            if 0 <= r < g.shape[0]:
                g[r, cap_col] = BLUE


def generate_input() -> np.ndarray:
    left_len = random.randint(4, 8)
    right_len = random.randint(4, 8)
    stem_len = random.randint(5, 11)
    stem_w = random.randint(1, 3)

    mode = random.randrange(3)
    if mode == 0:
        left_w = random.randint(1, 2)
        right_w = random.randint(left_w + 1, 3)
    elif mode == 1:
        right_w = random.randint(1, 2)
        left_w = random.randint(right_w + 1, 3)
    else:
        left_w = right_w = random.randint(1, 3)

    max_branch_w = max(left_w, right_w)
    junction_row = 2 + max_branch_w
    stem_bottom = junction_row + stem_len - 1
    left_end = 4
    stem_c0 = left_end + left_len
    right_start = stem_c0 + stem_w
    right_end = right_start + right_len - 1

    min_h = stem_bottom + 4
    min_w = right_end + 5
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    off_r = random.randint(0, h - min_h)
    off_c = random.randint(0, w - min_w)

    g = np.zeros((h, w), dtype=int)
    jr = off_r + junction_row
    sb = off_r + stem_bottom
    le = off_c + left_end
    sc0 = off_c + stem_c0
    rs = off_c + right_start
    re = off_c + right_end

    g[jr : sb + 1, sc0 : sc0 + stem_w] = BLUE

    left_r0 = jr - left_w + 1
    right_r0 = jr - right_w + 1
    g[left_r0 : jr + 1, le:sc0] = BLUE
    g[right_r0 : jr + 1, rs : re + 1] = BLUE

    _draw_horizontal_slot(g, left_r0, jr, le, -1)
    _draw_horizontal_slot(g, right_r0, jr, re, 1)

    if stem_w == 1:
        block_c0 = sc0 + random.choice([-1, 0])
    elif stem_w == 2:
        block_c0 = sc0
    else:
        block_c0 = sc0 + random.choice([0, 1])
    color = random.choice([2, 3, 4, 5, 6, 7, 8, 9])
    g[sb + 1 : sb + 3, block_c0 : block_c0 + 2] = color

    k = random.randrange(4)
    if k:
        g = np.rot90(g, k)
    return np.array(g, dtype=int)
