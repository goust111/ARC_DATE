from __future__ import annotations

import numpy as np

BLACK = 0
RED = 2
YELLOW = 4
GREY = 5


def _neighbors4(r: int, c: int, H: int, W: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < H:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < W:
        yield r, c + 1


def _components(mask: np.ndarray):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray):
    if len(points) == 0:
        return None
    r0 = int(points[:, 0].min())
    c0 = int(points[:, 1].min())
    r1 = int(points[:, 0].max())
    c1 = int(points[:, 1].max())
    return r0, c0, r1, c1


def _find_triangle_and_marker(grid: np.ndarray):
    H, W = grid.shape
    marker = np.argwhere(grid == YELLOW)
    if len(marker) == 0:
        return None, None
    mr, mc = map(int, marker[0])
    tri_mask = (grid != BLACK) & ((grid == RED) | (grid == YELLOW))
    tri_comps = _components(tri_mask)
    tri_coords = None
    for comp in tri_comps:
        if (mr, mc) in {tuple(p) for p in comp}:
            tri_coords = comp
            break
    if tri_coords is None:
        tri_coords = np.argwhere(tri_mask)
    return tri_coords, (mr, mc)


def _find_frames(grid: np.ndarray):
    H, W = grid.shape
    gray = grid == GREY
    comps = _components(gray)
    frames = []
    for comp in comps:
        if len(comp) < 8:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        area = (h * w)
        perim_expected = 2 * (h + w) - 4
        if len(comp) != perim_expected:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            if grid[r, c0] != GREY or grid[r, c1] != GREY:
                ok = False
                break
        if not ok:
            continue
        for c in range(c0, c1 + 1):
            if grid[r0, c] != GREY or grid[r1, c] != GREY:
                ok = False
                break
        if not ok:
            continue
        # strict frame: no grey pixels on interior
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size > 0 and np.any(interior == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def _inside_frame(frame, r: int, c: int):
    r0, c0, r1, c1 = frame
    return r0 < r < r1 and c0 < c < c1


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    H, W = grid.shape
    out = np.array(grid, copy=True)

    tri_coords, marker_pos = _find_triangle_and_marker(grid)
    if marker_pos is None:
        return out

    tri_set = {tuple(p) for p in tri_coords} if tri_coords is not None else set()

    mr, mc = marker_pos
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    exposed = (0, 1)
    for dr, dc in dirs:
        rr, cc = mr + dr, mc + dc
        if 0 <= rr < H and 0 <= cc < W:
            if (rr, cc) not in tri_set:
                exposed = (-dr, -dc)
                break
        else:
            exposed = (dr * -1, dc * -1)
            break

    # tip: farthest along ray direction inside triangle
    tri_list = list(tri_set)
    dr, dc = exposed
    tr, tc = max(tri_list, key=lambda rc: rc[0] * dr + rc[1] * dc)

    frames = _find_frames(grid)
    active = set()

    r, c = tr, tc
    while 0 <= r < H and 0 <= c < W:
        out[r, c] = YELLOW
        for idx, f in enumerate(frames):
            if _inside_frame(f, r, c):
                active.add(idx)
        r += dr
        c += dc

    # Fill activated frames by their marker color
    for idx in active:
        r0, c0, r1, c1 = frames[idx]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (inner != BLACK) & (inner != GREY)
        colors = np.unique(inner[marker_mask])
        if len(colors) != 1:
            if len(colors) == 0:
                continue
            fill_color = int(colors[0])
        else:
            fill_color = int(colors[0])
        bg_mask = inner == BLACK
        inner[bg_mask] = fill_color

    return out


def _rot90_point(r: int, c: int, h: int, t: int):
    if t == 0:
        return r, c
    if t == 1:
        return c, h - 1 - r
    if t == 2:
        return h - 1 - r, h - 1 - c
    # t == 3
    return h - 1 - c, r


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    H = W = 18
    out = np.zeros((H, W), dtype=int)

    # Right triangle-like shape with triangular marker on base center
    h = 5
    tri = [(i, j) for i in range(h) for j in range(i + 1)]
    mr_local = (h - 1, h // 2)
    orientation = int(rng.integers(4))
    tset = {_rot90_point(r, c, h, orientation) for r, c in tri}
    tr, tc = _rot90_point(*mr_local, h, orientation)
    trs = np.array(list(tset), dtype=int)
    trs_max = trs.max(axis=0)
    trs_min = trs.min(axis=0)
    h_t = trs_max[0] - trs_min[0] + 1
    w_t = trs_max[1] - trs_min[1] + 1
    off_r = int(rng.integers(1, H - h_t))
    off_c = int(rng.integers(1, W - w_t))
    tri_pts = []
    for rr, cc in tri:
        rr, cc = _rot90_point(rr, cc, h, orientation)
        rr += off_r - trs_min[0]
        cc += off_c - trs_min[1]
        out[rr, cc] = RED
        tri_pts.append((rr, cc))
    tr += off_r - trs_min[0]
    tc += off_c - trs_min[1]
    out[tr, tc] = YELLOW

    # Random hollow grey rectangular frames
    frame_num = int(rng.integers(2, 5))
    marker_colors = [1, 3, 6, 7]
    rng.shuffle(marker_colors)
    used = []
    frames = []
    for k in range(frame_num):
        fh = int(rng.integers(4, 7))
        fw = int(rng.integers(4, 7))
        attempts = 0
        ok = False
        while attempts < 200 and not ok:
            attempts += 1
            fr = int(rng.integers(0, H - fh))
            fc = int(rng.integers(0, W - fw))
            cells = []
            for rr in range(fr, fr + fh):
                cells.append((rr, fc))
                cells.append((rr, fc + fw - 1))
            for cc in range(fc + 1, fc + fw - 1):
                cells.append((fr, cc))
                cells.append((fr + fh - 1, cc))
            # no overlap or touching existing frames
            if any(out[r, c] != BLACK for r, c in cells):
                continue
            if any(
                abs(r - tr) <= 1 and abs(c - tc) <= 1 for r, c in cells
            ):
                continue
            if any(
                any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in used)
                for r, c in cells
            ):
                continue
            ok = True
            for rr in range(fr, fr + fh):
                out[rr, fc] = GREY
                out[rr, fc + fw - 1] = GREY
            for cc in range(fc + 1, fc + fw - 1):
                out[fr, cc] = GREY
                out[fr + fh - 1, cc] = GREY
            used.extend(cells)
            frames.append((fr, fc, fh, fw))

    # Put one non-bg non-grey marker in each frame interior.
    for idx, (fr, fc, fh, fw) in enumerate(frames):
        if idx >= len(marker_colors):
            break
        ir0, ir1 = fr + 1, fr + fh - 2
        ic0, ic1 = fc + 1, fc + fw - 2
        if ir0 > ir1 or ic0 > ic1:
            continue
        r = int(rng.integers(ir0, ir1 + 1))
        c = int(rng.integers(ic0, ic1 + 1))
        if out[r, c] != BLACK:
            out[r, c] = marker_colors[idx % len(marker_colors)]

    return out
