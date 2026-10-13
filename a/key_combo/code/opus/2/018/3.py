import numpy as np
from collections import deque


_BG = 0
_GREEN = 3
_RED = 2
_CYAN = 8
_YELLOW = 4
_MAGENTA = 6
_GRAY = 5

_generate_calls = 0


def _rotate(mat, k):
    return np.rot90(mat, -k)


def _find_component(grid, color):
    coords = np.argwhere(grid == color)
    if len(coords) == 0:
        return []
    h, w = grid.shape
    vis = np.zeros((h, w), dtype=bool)
    r0, c0 = coords[0]
    q = deque([(r0, c0)])
    vis[r0, c0] = True
    comp = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not vis[nr, nc] and grid[nr, nc] == color:
                vis[nr, nc] = True
                q.append((nr, nc))
    return comp


def _neighbors(r, c, h, w):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _simulate_path(grid):
    h, w = grid.shape
    green_comp = _find_component(grid, _GREEN)
    if not green_comp:
        return []
    Gset = set(green_comp)
    # choose one end as start, try both
    min_r = min(r for r, c in green_comp)
    max_r = max(r for r, c in green_comp)
    min_c = min(c for r, c in green_comp)
    max_c = max(c for r, c in green_comp)
    if max_r == min_r:
        endpoints = [(min_r, min_c), (min_r, max_c)]
    else:
        endpoints = [(min_r, min_c), (max_r, min_c)]

    red_cells = {(r, c) for r, c in _find_component(grid, _RED)}
    cyan = _GREEN in ()

    def try_start(s):
        sr, sc = s
        if (sr, sc) not in Gset:
            return []
        nbrs = [ (nr, nc) for nr, nc in _neighbors(sr, sc, h, w) if (nr, nc) in Gset and grid[nr, nc] == _GREEN]
        if not nbrs:
            return []
        dr0, dc0 = nbrs[0][0] - sr, nbrs[0][1] - sc
        start = (sr, sc)
        direction = (dr0, dc0)
        path = [start]
        prev_dir = direction
        seen = {start}

        while len(path) < h * w:
            pr, pc = path[-1]
            fr, fc = pr + prev_dir[0], pc + prev_dir[1]
            # if next is red, stop at current cell
            if (fr, fc) in red_cells:
                return path, prev_dir

            def in_front_is_free(pos, direction):
                r, c = pos
                nr, nc = r + direction[0], c + direction[1]
                if not (0 <= nr < h and 0 <= nc < w):
                    return False
                return grid[nr, nc] not in (_CYAN,)

            if in_front_is_free((pr, pc), prev_dir):
                nxt = (fr, fc)
            else:
                left = ((-prev_dir[1], prev_dir[0]))
                right = ((prev_dir[1], -prev_dir[0]))
                found = None
                candidates = [left, right]
                for d in candidates:
                    nr, nc = pr + d[0], pc + d[1]
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if grid[nr, nc] == _CYAN:
                        continue
                    if (nr, nc) in seen:
                        continue
                    found = d
                    break
                if found is None:
                    return []
                nxt = (pr + found[0], pc + found[1])
                prev_dir = found
            if grid[nxt[0], nxt[1]] == _RED:
                return path, prev_dir
            if nxt in seen:
                return []
            path.append(nxt)
            seen.add(nxt)
        return []

    best = []
    for s in endpoints:
        result = try_start(s)
        if result:
            path, direction = result
            if any(cell in red_cells for cell in path):
                continue
            best = (path, direction)
            break
    if best:
        return best
    return []


def _stamp(out, stamp_arr, center, color):
    h, w = out.shape
    sh, sw = stamp_arr.shape
    cr, cc = center
    top = cr - sh // 2
    left = cc - sw // 2
    for r in range(sh):
        for c in range(sw):
            if stamp_arr[r, c] == _BG:
                continue
            tr = top + r
            tc = left + c
            if 0 <= tr < h and 0 <= tc < w:
                out[tr, tc] = color


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(3000 + _generate_calls)
    _generate_calls += 1

    h = 18
    w = 18
    grid = np.zeros((h, w), dtype=int)

    # template in top-left 4x4 region
    template_shape = np.zeros((4, 4), dtype=int)
    yy, xx = np.where(np.ones((4, 4), dtype=bool))
    n = 4 + (rng.randint(0, 3))
    idx = rng.choice(np.arange(16), size=n, replace=False)
    pts = list(zip(yy[idx], xx[idx]))
    for y, x in pts:
        template_shape[y, x] = _YELLOW
    # enforce non-symmetry
    if np.array_equal(template_shape, np.fliplr(template_shape)):
        template_shape[0, 3] = _YELLOW if template_shape[1, 2] == 0 else 0
    for y in range(4):
        for x in range(4):
            if template_shape[y, x] != 0:
                grid[y, x] = template_shape[y, x]

    # build a deterministic obstacle-turned path layout
    init_orient = rng.randint(0, 4)
    starts = [(3, 2), (3, 15), (15, 3), (15, 15)]
    sr, sc = starts[init_orient]
    if init_orient in (0, 2):
        dir0 = (0, 1) if init_orient == 0 else (0, -1)
    else:
        dir0 = (1, 0) if init_orient == 3 else (-1, 0)

    path = [
        (sr, sc),
        (sr + dir0[0], sc + dir0[1]),
        (sr + 2 * dir0[0], sc + 2 * dir0[1]),
    ]
    for r, c in path:
        grid[r, c] = _GREEN

    # carve a winding black path under rule constraints by deterministic turns
    cur_r, cur_c = path[-1]
    cur_d = dir0
    steps = 20
    for _ in range(steps):
        # place marker candidates occasionally
        nxt_r, nxt_c = cur_r + cur_d[0], cur_c + cur_d[1]
        if not (0 <= nxt_r < h and 0 <= nxt_c < w):
            break
        if grid[nxt_r, nxt_c] == 0:
            path.append((nxt_r, nxt_c))
            cur_r, cur_c = nxt_r, nxt_c
            continue

        # obstacle-trigger turn
        left = (-cur_d[1], cur_d[0])
        right = (cur_d[1], -cur_d[0])
        candidates = []
        for d in (left, right):
            rr, cc = cur_r + d[0], cur_c + d[1]
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 0:
                candidates.append(d)
        if not candidates:
            break
        cur_d = candidates[0]

    # mark final few path cells and red target
    if len(path) >= 10:
        end_r, end_c = path[-1]
        grid[end_r, end_c] = _RED
        grid[end_r - 1 if end_r > 0 else end_r, end_c] = _RED

    # markers on path (excluding start and red tail)
    if len(path) > 6:
        mark_count = 2 + rng.randint(0, 3)
        chosen = rng.choice(np.arange(1, max(3, len(path) - 3)), size=min(mark_count, max(1, len(path) // 6)), replace=False)
        for idx in chosen:
            mr, mc = path[idx]
            if grid[mr, mc] == 0:
                grid[mr, mc] = _MAGENTA

    # obstacles in middle block
    for _ in range(12):
        rr = 5 + rng.randint(0, 8)
        cc = 5 + rng.randint(0, 8)
        if grid[rr, cc] == 0:
            grid[rr, cc] = _CYAN

    # keep path as black except start marker
    for r, c in path:
        if grid[r, c] not in (_GREEN, _RED, _MAGENTA):
            grid[r, c] = 0

    # ensure endpoint is reachable and unique
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """1) Parse green start segment to obtain initial heading.
2) Deterministically follow forward-vs-obstacle rule until touching red marker line.
3) Extract yellow template at the corner and rotate per step direction.
4) Stamp template center-aligned at each magenta marker; later markers override earlier; keep base features."""

    g = np.array(grid)
    h, w = g.shape

    # template from corner component
    yy, xx = np.where(g == _YELLOW)
    if len(yy) == 0:
        return g
    tminr, tmaxr = yy.min(), yy.max()
    tminc, tmaxc = xx.min(), xx.max()
    template = g[tminr : tmaxr + 1, tminc : tmaxc + 1]
    mask = template > 0
    template = template.copy()

    # locate start segment and simulate path
    green_cells = list(zip(*np.where(g == _GREEN)))
    if not green_cells:
        return g
    green_set = set(green_cells)
    endpoint = None
    for p in green_cells:
        r, c = p
        neigh = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (r + dr, c + dc) in green_set]
        if len(neigh) == 1:
            endpoint = p
            break
    if endpoint is None:
        endpoint = green_cells[0]

    red_set = set(tuple(p) for p in np.argwhere(g == _RED))
    g_for_move = g.copy()
    g_for_move[g == _GREEN] = 0

    # two endpoint candidates, choose one reaches red
    start = endpoint
    for candidate in [endpoint]:
        sr, sc = candidate
        neigh = [(sr + dr, sc + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (sr + dr, sc + dc) in green_set]
        if not neigh:
            continue
        path = [candidate]
        prev = (neigh[0][0] - sr, neigh[0][1] - sc)
        pos = candidate
        directions = []
        for _ in range(h * w):
            r, c = pos
            fr, fc = r + prev[0], c + prev[1]
            if (fr, fc) in red_set:
                break
            if (0 <= fr < h and 0 <= fc < w and g[fr, fc] != _CYAN):
                nxt = (fr, fc)
                move_dir = prev
            else:
                left = (-prev[1], prev[0])
                right = (prev[1], -prev[0])
                nxt = None
                move_dir = None
                for d in (left, right):
                    rr, cc = r + d[0], c + d[1]
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if g[rr, cc] == _CYAN:
                        continue
                    if (rr, cc) not in path:
                        nxt = (rr, cc)
                        move_dir = d
                        break
                if nxt is None:
                    break
            path.append(nxt)
            directions.append(move_dir)
            pos = nxt
        if path and (path[-1][0], path[-1][1]) in red_set:
            break
        if not path:
            continue

    out = g.copy()

    # restore immutable frame features
    out[g != 0] = g[g != 0]

    # markers and directions derived from first pass over black path
    sim = _simulate_path(g)
    if not sim:
        return out
    path_cells, _ = sim
    dirs = []
    for i in range(1, len(path_cells)):
        a = path_cells[i - 1]
        b = path_cells[i]
        dr, dc = b[0] - a[0], b[1] - a[1]
        dirs.append((dr, dc))

    marker_cells = list(zip(*np.where(g == _MAGENTA)))
    marker_dirs = {}
    for idx, p in enumerate(path_cells[:-1]):
        if p in marker_cells and idx < len(dirs):
            marker_dirs[p] = dirs[idx]

    for p in path_cells:
        if p not in marker_dirs:
            continue
        dr, dc = marker_dirs[p]
        if (dr, dc) == (0, 1):
            k = 0
        elif (dr, dc) == (1, 0):
            k = 1
        elif (dr, dc) == (0, -1):
            k = 2
        else:
            k = 3
        stamp = _rotate(template, k)
        r, c = p
        _stamp(out, stamp, (r, c), _YELLOW)

    return out


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample).shape)
