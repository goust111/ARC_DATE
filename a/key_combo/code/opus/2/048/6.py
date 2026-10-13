from __future__ import annotations

from collections import Counter, deque
import numpy as np

BLACK = 0
GREY = 5
ORANGE = 7


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
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_gray_frame(grid: np.ndarray):
    H, W = grid.shape
    comps = _components(grid == GREY)
    best = None
    for comp in comps:
        if len(comp) < 12:
            continue
        r0, c0 = comp[:, 0].min(), comp[:, 1].min()
        r1, c1 = comp[:, 0].max(), comp[:, 1].max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        perim = 2 * (h + w) - 4
        if len(comp) != perim:
            continue
        ok = True
        for rr in range(r0, r1 + 1):
            if grid[rr, c0] != GREY or grid[rr, c1] != GREY:
                ok = False
                break
        if not ok:
            continue
        for cc in range(c0, c1 + 1):
            if grid[r0, cc] != GREY or grid[r1, cc] != GREY:
                ok = False
                break
        if not ok:
            continue
        best = (r0, c0, r1, c1)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    H, W = out.shape
    frame = _find_gray_frame(out)
    if frame is None:
        return out
    r0, c0, r1, c1 = frame
    ir0, ir1 = r0 + 1, r1 - 1
    ic0, ic1 = c0 + 1, c1 - 1

    # detect bars on the right of frame
    bar_mask = (out != BLACK) & (out != GREY)
    comps = _components(bar_mask)
    bars = []
    for comp in comps:
        if len(comp) < 2:
            continue
        cols = comp[:, 1]
        rows = comp[:, 0]
        width = cols.max() - cols.min() + 1
        if width > 2:
            continue
        cols_min, cols_max = cols.min(), cols.max()
        if cols_max <= c1:
            continue
        colors, counts = np.unique(out[comp[:, 0], comp[:, 1]], return_counts=True)
        cnt = Counter(int(c) for c in colors if int(c) != ORANGE)
        bar_color = max(cnt.items(), key=lambda x: x[1])[0] if cnt else int(colors[0])
        bars.append((int(rows.size), bar_color, comp, int(rows.min()), int(rows.max())))
    if not bars:
        return out

    bars_sorted = sorted(bars, key=lambda x: x[0])
    palette = [b[1] for b in bars_sorted]
    n = len(palette)

    # Determine marker bar and position
    marker_info = None
    for i, (h, color, comp, mn, mx) in enumerate(bars):
        if np.any(out[comp[:, 0], comp[:, 1]] == ORANGE):
            marker_info = (i, comp)
            break
    if marker_info is None:
        return out
    mid_idx, marker_comp = marker_info
    mr = marker_comp[:, 0]
    mc = marker_comp[:, 1]
    marker_rows = mr[out[mr, mc] == ORANGE]
    row_mid = (mr.min() + mr.max()) / 2
    marker_upper = marker_rows.max() <= row_mid

    # ring center from marker half
    inner_h = ir1 - ir0 + 1
    half = inner_h // 2
    if marker_upper:
        a = ir0
        b = ir0 + half - 1
    else:
        a = ir0 + half
        b = ir1
    center_r = (a + b) // 2
    center_c = (ic0 + ic1) // 2

    # concentric fill from outer to inner
    for rr in range(ir0, ir1 + 1):
        for cc in range(ic0, ic1 + 1):
            d = max(abs(rr - center_r), abs(cc - center_c))
            if d < n:
                out[rr, cc] = palette[n - d - 1]

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    out = np.zeros((18, 20), dtype=int)

    # left frame
    r0, c0, r1, c1 = 1, 1, 16, 8
    for r in range(r0, r1 + 1):
        out[r, c0] = GREY
        out[r, c1] = GREY
    for c in range(c0, c1 + 1):
        out[r0, c] = GREY
        out[r1, c] = GREY

    colors = list(map(int, rng.choice([1, 2, 3, 4, 6, 8, 9], size=5, replace=False)))
    heights = [3, 4, 5, 6, 7]
    bar_cols = [9, 11, 13, 15, 17]
    mark_idx = int(rng.integers(0, 5))
    for i, col in enumerate(bar_cols):
        h = heights[i]
        start = 16 - h
        color = colors[i]
        out[start:16, col] = color
        if i == mark_idx:
            # replace half by orange to mark this bar
            if h % 2 == 1:
                h2 = h // 2 + 1
            else:
                h2 = h // 2
            if rng.integers(0, 2) == 0:
                # upper half
                out[start:start + h2, col] = ORANGE
            else:
                # lower half
                out[start + h - h2 : 16, col] = ORANGE

    return out
