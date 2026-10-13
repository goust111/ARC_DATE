import numpy as np
from collections import deque


def _connected(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _border_path(r0, c0, r1, c1):
    path = []
    for c in range(c0, c1 + 1):
        path.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        path.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        path.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        path.append((r, c0))
    return path


def _segment_start_and_len(flags):
    p = len(flags)
    if p == 0:
        return None, 0
    if all(flags):
        return 0, p
    d = flags + flags
    best = (None, 0)
    i = 0
    while i < 2 * p:
        if not d[i]:
            i += 1
            continue
        j = i
        while j < 2 * p and d[j]:
            j += 1
        length = j - i
        if i < p and length <= p and length > best[1]:
            best = (i, length)
        i = j
    if best[0] is None:
        return None, 0
    return best[0], best[1]


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    nz = np.argwhere(out != 0)
    if len(nz) == 0:
        return out

    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    n = int(np.sum(interior == 5))

    path = _border_path(int(r0), int(c0), int(r1), int(c1))
    p = len(path)
    if p == 0:
        return out

    border_coords = set(path)
    flags = []
    colors = []
    for rr, cc in path:
        v = int(out[rr, cc])
        colors.append(v)
        flags.append(v != 0 and v != 5)

    if not any(flags):
        out[out == 5] = 0
        return out

    seg_color = colors[flags.index(True)]
    # Keep only one contiguous color segment
    start, k = _segment_start_and_len(flags)
    if start is None or k == 0 or k > p:
        out[out == 5] = 0
        return out
    k = min(k, 4)

    for rr, cc in path:
        out[rr, cc] = 0

    step = k + n
    idx = start
    while idx < start + p:
        for t in range(k):
            rr, cc = path[(idx + t) % p]
            out[rr, cc] = seg_color
        idx += step

    out[out == 5] = 0
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    h = 10
    w = 10
    grid = np.zeros((h, w), dtype=np.int64)

    top = int(rng.integers(0, 3))
    left = int(rng.integers(0, 3))
    rect_h = int(rng.integers(5, 9))
    rect_w = int(rng.integers(5, 9))
    r1 = top + rect_h - 1
    c1 = left + rect_w - 1
    if r1 >= h:
        r1 = h - 1
        rect_h = r1 - top + 1
    if c1 >= w:
        c1 = w - 1
        rect_w = c1 - left + 1
    rect_h = max(4, min(rect_h, h))
    rect_w = max(4, min(rect_w, w))
    r1 = min(h - 1, top + rect_h - 1)
    c1 = min(w - 1, left + rect_w - 1)
    top = min(top, r1 - 3)
    left = min(left, c1 - 3)

    frame_color = int(rng.choice([1, 2, 3, 4, 7, 8, 9]))
    for c in range(left, c1 + 1):
        grid[top, c] = frame_color
        grid[r1, c] = frame_color
    for r in range(top + 1, r1):
        grid[r, left] = frame_color
        grid[r, c1] = frame_color

    side = str(rng.choice(["top", "bottom", "left", "right"]))
    seg_len = int(rng.integers(2, 5))
    seg_len = min(seg_len, max(2, (rect_w if side in ("top", "bottom") else rect_h) - 1))
    if side in ("top", "bottom"):
        if side == "top":
            r = top
            c_start = int(rng.integers(left, c1 - seg_len + 1))
            for c in range(c_start, c_start + seg_len):
                grid[r, c] = frame_color
        else:
            r = r1
            c_start = int(rng.integers(left, c1 - seg_len + 1))
            for c in range(c_start, c_start + seg_len):
                grid[r, c] = frame_color
    else:
        if side == "left":
            c = left
            r_start = int(rng.integers(top, r1 - seg_len + 1))
            for r in range(r_start, r_start + seg_len):
                grid[r, c] = frame_color
        else:
            c = c1
            r_start = int(rng.integers(top, r1 - seg_len + 1))
            for r in range(r_start, r_start + seg_len):
                grid[r, c] = frame_color

    cnt = int(rng.integers(1, 6))
    for _ in range(cnt):
        rr = int(rng.integers(top + 1, r1))
        cc = int(rng.integers(left + 1, c1))
        if grid[rr, cc] != 5:
            grid[rr, cc] = 5
    return grid
