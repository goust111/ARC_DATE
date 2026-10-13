import numpy as np
from collections import deque


def _normalize(points):
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    min_r, min_c = min(rs), min(cs)
    return tuple(sorted(((r - min_r, c - min_c) for r, c in points)))


def _rotate_shape(points, k):
    if k % 4 == 0:
        rot = points
    elif k % 4 == 1:
        rot = [(c, -r) for r, c in points]
    elif k % 4 == 2:
        rot = [(-r, -c) for r, c in points]
    else:
        rot = [(-c, r) for r, c in points]
    return _normalize(rot)


def _shape_rotations(shape):
    rots = []
    seen = set()
    for k in range(4):
        rot = _rotate_shape(shape, k)
        key = tuple(rot)
        if key not in seen:
            seen.add(key)
            rots.append(rot)
    return rots


def _random_shape(rng, area):
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < area:
        r, c = frontier[int(rng.integers(0, len(frontier)))]
        dr, dc = rng.choice(np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int16))
        nxt = (r + int(dr), c + int(dc))
        if nxt in cells:
            continue
        cells.add(nxt)
        frontier.append(nxt)
    return _normalize(tuple(cells))


def _flood_fill_exterior(wall):
    h, w = wall.shape
    outside = np.zeros_like(wall, dtype=bool)
    q = deque()
    for x in range(w):
        if not wall[0, x]:
            outside[0, x] = True
            q.append((0, x))
        if not wall[h - 1, x] and not outside[h - 1, x]:
            outside[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if not wall[y, 0] and not outside[y, 0]:
            outside[y, 0] = True
            q.append((y, 0))
        if not wall[y, w - 1] and not outside[y, w - 1]:
            outside[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        r, c = q.popleft()
        nr = r - 1
        if nr >= 0 and (not wall[nr, c]) and (not outside[nr, c]):
            outside[nr, c] = True
            q.append((nr, c))
        nr = r + 1
        if nr < h and (not wall[nr, c]) and (not outside[nr, c]):
            outside[nr, c] = True
            q.append((nr, c))
        nc = c - 1
        if nc >= 0 and (not wall[r, nc]) and (not outside[r, nc]):
            outside[r, nc] = True
            q.append((r, nc))
        nc = c + 1
        if nc < w and (not wall[r, nc]) and (not outside[r, nc]):
            outside[r, nc] = True
            q.append((r, nc))
    return outside


def _touches_full_wall(shape_abs, wall):
    rs = [r for r, _ in shape_abs]
    cs = [c for _, c in shape_abs]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    top = [(r, c) for r, c in shape_abs if r == min_r]
    if len(top) >= 1:
        if all(0 <= r - 1 and wall[r - 1, c] for r, c in top):
            return True

    bot = [(r, c) for r, c in shape_abs if r == max_r]
    if len(bot) >= 1:
        h = wall.shape[0]
        if all(r + 1 < h and wall[r + 1, c] for r, c in bot):
            return True

    left = [(r, c) for r, c in shape_abs if c == min_c]
    if len(left) >= 1:
        if all(0 <= c - 1 and wall[r, c - 1] for r, c in left):
            return True

    right = [(r, c) for r, c in shape_abs if c == max_c]
    if len(right) >= 1:
        w = wall.shape[1]
        if all(c + 1 < w and wall[r, c + 1] for r, c in right):
            return True

    return False


def _find_unique_placement(shape, inside, wall, occupied):
    h, w = inside.shape
    shape_cells = shape
    rs = [r for r, c in shape_cells]
    cs = [c for _, c in shape_cells]
    min_r, min_c = min(rs), min(cs)
    max_r, max_c = max(rs), max(cs)

    candidates_by_rot = []
    for rot in range(4):
        rot_cells = _rotate_shape(shape_cells, rot)
        rrs = [r for r, c in rot_cells]
        ccs = [c for _, c in rot_cells]
        r0_min, c0_min = min(rrs), min(ccs)
        r0_max, c0_max = max(rrs), max(ccs)
        local = []
        for tr in range(-r0_min, h - r0_max):
            for tc in range(-c0_min, w - c0_max):
                abs_cells = [(tr + r, tc + c) for r, c in rot_cells]
                ok = True
                for rr, cc in abs_cells:
                    if not inside[rr, cc] or occupied[rr, cc]:
                        ok = False
                        break
                if not ok:
                    continue
                if not _touches_full_wall(abs_cells, wall):
                    continue
                local.append((tr, tc))
        candidates_by_rot.append(local)

    valid_rots = [i for i, lst in enumerate(candidates_by_rot) if len(lst) == 1]
    if len(valid_rots) != 1:
        return None
    rot = valid_rots[0]
    if len(candidates_by_rot[rot]) != 1:
        return None
    pos = candidates_by_rot[rot][0]
    return rot, pos


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                nr = rr - 1
                if nr >= 0 and mask[nr, cc] and not seen[nr, cc]:
                    seen[nr, cc] = True
                    q.append((nr, cc))
                    comp.append((nr, cc))
                nr = rr + 1
                if nr < h and mask[nr, cc] and not seen[nr, cc]:
                    seen[nr, cc] = True
                    q.append((nr, cc))
                    comp.append((nr, cc))
                nc = cc - 1
                if nc >= 0 and mask[rr, nc] and not seen[rr, nc]:
                    seen[rr, nc] = True
                    q.append((rr, nc))
                    comp.append((rr, nc))
                nc = cc + 1
                if nc < w and mask[rr, nc] and not seen[rr, nc]:
                    seen[rr, nc] = True
                    q.append((rr, nc))
                    comp.append((rr, nc))
            comps.append(np.array(comp, dtype=np.int16))
    return comps


def _sample_notches(rng, left, right, count):
    if right - left < 8:
        return []
    max_width = max(2, (right - left) // 4)
    intervals = []
    attempts = 0
    while len(intervals) < count and attempts < 400:
        attempts += 1
        span = int(rng.integers(2, max_width + 1))
        if right - left - span - 2 < 1:
            continue
        s = int(rng.integers(left + 1, right - span))
        e = s + span
        if any(not (e <= a0 - 1 or s >= e0 + 1) for a0, e0 in intervals):
            continue
        intervals.append((s, e))
    if len(intervals) < count:
        return []
    intervals = sorted(intervals[:count], key=lambda x: x[0])
    out = []
    for s, e in intervals:
        out.append((s, e))
    return out


def _build_top_path(row, left, right, notches, depths):
    path = []
    cur = left
    for i, (s, e) in enumerate(notches):
        d = depths[i]
        for x in range(cur, s + 1):
            path.append((row, x))
        for rr in range(row + 1, row + d + 1):
            path.append((rr, s))
        for x in range(s + 1, e + 1):
            path.append((row + d, x))
        for rr in range(row + d - 1, row - 1, -1):
            path.append((rr, e))
        cur = e
    for x in range(cur + 1, right + 1):
        path.append((row, x))
    return path


def _build_bottom_path(row, left, right, notches, depths):
    path = []
    cur = right
    for idx in range(len(notches) - 1, -1, -1):
        s, e = notches[idx]
        d = depths[idx]
        for x in range(cur, s - 1, -1):
            path.append((row, x))
        for rr in range(row - 1, row - d - 1, -1):
            path.append((rr, s))
        for x in range(s - 1, e - 1, -1):
            path.append((row - d, x))
        for rr in range(row - d + 1, row + 1):
            path.append((rr, e))
        cur = e
    for x in range(cur - 1, left - 1, -1):
        path.append((row, x))
    return path


def _build_frame_wall(h, w, rng):
    for _ in range(120):
        margin = int(rng.integers(2, 5))
        top = margin
        left = margin
        max_h = h - margin - 2
        max_w = w - margin - 2
        if max_h - margin < 10 or max_w - margin < 10:
            continue
        frame_h = int(rng.integers(10, max_h - margin + 1))
        frame_w = int(rng.integers(10, max_w - margin + 1))
        bottom = top + frame_h - 1
        right = left + frame_w - 1
        if frame_h < 9 or frame_w < 9:
            continue

        max_depth_top = max(2, min(5, frame_h - 3))
        notch_t = int(rng.integers(0, 3))
        notch_b = int(rng.integers(0, 2))
        if notch_t == 0 and notch_b == 0 and rng.random() < 0.75:
            notch_t = 1
        top_n = _sample_notches(rng, left, right, notch_t)
        bot_n = _sample_notches(rng, left, right, notch_b)
        top_depths = [int(rng.integers(2, max_depth_top + 1)) for _ in top_n]
        bot_depths = [int(rng.integers(2, max_depth_top + 1)) for _ in bot_n]

        path = []
        path.extend(_build_top_path(top, left, right, top_n, top_depths))
        for r in range(top + 1, bottom + 1):
            path.append((r, right))
        path.extend(_build_bottom_path(bottom, left, right, bot_n, bot_depths))
        for r in range(bottom - 1, top - 1, -1):
            path.append((r, left))

        path_set = set(path)
        wall = np.zeros((h, w), dtype=bool)
        for r, c in path_set:
            wall[r, c] = True

        # validate path simplicity
        if len(path) != len(path_set):
            continue
        valid = True
        for r, c in path_set:
            deg = 0
            if (r - 1, c) in path_set:
                deg += 1
            if (r + 1, c) in path_set:
                deg += 1
            if (r, c - 1) in path_set:
                deg += 1
            if (r, c + 1) in path_set:
                deg += 1
            if deg != 2:
                valid = False
                break
        if not valid:
            continue

        outside = _flood_fill_exterior(wall)
        inside = (~outside) & (~wall)
        if inside.sum() < 25:
            continue

        # inside must be one component for cleaner puzzle instances
        comps = _components(inside)
        if len(comps) != 1:
            continue

        return wall, inside
    raise RuntimeError("failed to build contour")


def _candidate_fragments(shape, inside, wall, occupied):
    cands = []
    for rot in range(4):
        rot_cells = _rotate_shape(shape, rot)
        rs = [r for r, c in rot_cells]
        cs = [c for _, c in rot_cells]
        rmin, cmin = min(rs), min(cs)
        rmax, cmax = max(rs), max(cs)
        h, w = inside.shape
        local = []
        for tr in range(-rmin, h - rmax):
            for tc in range(-cmin, w - cmax):
                abs_cells = [(tr + rr, tc + cc) for rr, cc in rot_cells]
                ok = True
                for rr, cc in abs_cells:
                    if not inside[rr, cc] or occupied[rr, cc]:
                        ok = False
                        break
                if not ok:
                    continue
                if not _touches_full_wall(abs_cells, wall):
                    continue
                local.append((tr, tc))
        if local:
            cands.append((rot, local))
    if len(cands) != 1:
        return None
    rot, pos = cands[0]
    if len(pos) != 1:
        return None
    return rot, pos[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
Assemble scattered color pieces into the grey contour cavity in descending area order using unique valid orientation/position constraints.
Input structure: black background, one-cell grey orthogonal closed contour, and several external monochrome fragments (4-connected), each with one color.
Rule: detect fragment components excluding background/grey, sort by area descending, and place each one in turn by trying four rotations; choose the only rotation that yields exactly one legal placement where all cells land in the contour interior, do not overlap already placed pieces, and at least one piece edge fully touches the grey wall.
Key insight: the hidden priority is area sorting first, and each piece has a uniquely valid orientation/position conditioned on the progressively filled interior.
    """
    if not isinstance(grid, np.ndarray):
        return grid
    grid = np.asarray(grid)
    h, w = grid.shape

    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    # Choose the dominant non-background color as the frame color.
    frame_color = None
    non_bg_colors = colors[colors != bg]
    if len(non_bg_colors) == 0:
        return grid.copy()
    for c in non_bg_colors:
        cnt = int(np.sum(grid == c))
        if frame_color is None or cnt > np.sum(grid == frame_color):
            frame_color = int(c)

    wall = (grid == frame_color)
    outside = _flood_fill_exterior(wall)
    inside = (~outside) & (~wall)

    piece_mask = (grid != bg) & (grid != frame_color)
    comps = []
    h_idxs, w_idxs = np.where(piece_mask)
    visited = np.zeros_like(grid, dtype=bool)
    for i in range(len(h_idxs)):
        r, c = int(h_idxs[i]), int(w_idxs[i])
        if visited[r, c]:
            continue
        col = int(grid[r, c])
        q = deque([(r, c)])
        visited[r, c] = True
        coords = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == col:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        comps.append((len(coords), col, coords))

    comps.sort(key=lambda it: (-it[0], it[1]))
    out = np.full_like(grid, bg)
    out[wall] = frame_color

    filled = np.zeros_like(wall, dtype=bool)
    for _, _, cells in comps:
        norm = _normalize(cells)
        result = _candidate_fragments(norm, inside, wall, filled)
        if result is None:
            continue
        rot, (tr, tc) = result
        placed_shape = _rotate_shape(norm, rot)
        for rr, cc in placed_shape:
            pr, pc = tr + rr, tc + cc
            out[pr, pc] = int(grid[cells[0][0], cells[0][1]])
            filled[pr, pc] = True

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    inp = np.zeros((h, w), dtype=np.int16)

    r0 = int(rng.integers(3, max(4, h // 3)))
    c0 = int(rng.integers(3, max(4, w // 3)))
    r1 = int(rng.integers(max(r0 + 6, h // 2), h - 2))
    c1 = int(rng.integers(max(c0 + 6, w // 2), w - 2))
    inp[r0 : r1 + 1, c0] = 1
    inp[r0 : r1 + 1, c1] = 1
    inp[r0, c0 : c1 + 1] = 1
    inp[r1, c0 : c1 + 1] = 1

    outside = np.ones((h, w), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    outside[inp == 1] = False
    occupied = inp != 0

    k = int(rng.integers(2, 5))
    colors = list(range(2, 2 + k))
    for color in colors:
        placed = False
        for _ in range(200):
            area = int(rng.integers(4, 8))
            shape = _random_shape(rng, area)
            shape = _rotate_shape(shape, int(rng.integers(0, 4)))
            rr = int(rng.integers(0, h))
            cc = int(rng.integers(0, w))
            cells = [(rr + a, cc + b) for a, b in shape]
            if all(
                0 <= r < h and 0 <= c < w and outside[r, c] and not occupied[r, c]
                for r, c in cells
            ):
                for r, c in cells:
                    inp[r, c] = color
                    occupied[r, c] = True
                placed = True
                break
        if not placed:
            return generate_input()

    return inp
