from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


def _arr(grid) -> np.ndarray:
    a = np.asarray(grid)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim == 1:
        a = a.reshape(1, a.size)
    elif a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
    if a.size == 0:
        return np.zeros((1, 1), dtype=int)
    return np.array(a[:30, :30], dtype=int, copy=True)


def _finish(grid) -> np.ndarray:
    a = _arr(grid)
    return np.array(a[:30, :30], dtype=int, copy=True)


def _bbox(mask: np.ndarray):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _comp_from(mask: np.ndarray, start):
    h, w = mask.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w and mask[sr, sc]):
        return []
    seen = np.zeros_like(mask, dtype=bool)
    q = deque([(int(sr), int(sc))])
    seen[sr, sc] = True
    pts = []
    while q:
        r, c = q.popleft()
        pts.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return pts


def _crop(arr: np.ndarray, fill=0) -> np.ndarray:
    box = _bbox(arr != fill)
    if box is None:
        return np.zeros((1, 1), dtype=int)
    r0, r1, c0, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(out: np.ndarray, stamp: np.ndarray, r: int, c: int, transparent=0):
    h, w = out.shape
    for rr in range(stamp.shape[0]):
        for cc in range(stamp.shape[1]):
            if stamp[rr, cc] != transparent and 0 <= r + rr < h and 0 <= c + cc < w:
                out[r + rr, c + cc] = int(stamp[rr, cc])


def _majority(vals, default=0):
    vals = [int(v) for v in vals if int(v) != 0]
    return Counter(vals).most_common(1)[0][0] if vals else default


def _split_by_full_lines(arr: np.ndarray, line_color=5):
    h, w = arr.shape
    rows = [r for r in range(h) if np.all(arr[r] == line_color)]
    cols = [c for c in range(w) if np.all(arr[:, c] == line_color)]
    r_intervals = [(a + 1, b - 1) for a, b in zip([-1] + rows, rows + [h]) if b - a > 1]
    c_intervals = [(a + 1, b - 1) for a, b in zip([-1] + cols, cols + [w]) if b - a > 1]
    return rows, cols, r_intervals, c_intervals


def _fold_inlay(arr: np.ndarray) -> np.ndarray:
    """Fold small hinged fragments into grey holes by matching the folded bbox."""
    out = arr.copy()
    grey = arr == 5
    box = _bbox(grey)
    if box is None:
        return _crop(out)
    r0, r1, c0, c1 = box
    frame = out[r0 : r1 + 1, c0 : c1 + 1].copy()
    holes = _components(frame == 0)
    sprites = []
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5, 7)]:
        for comp in _components(arr == color):
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp):
                continue
            b = _bbox(arr == color)
            if b is None:
                continue
            a, btm, x, y = b
            sprites.append((color, arr[a : btm + 1, x : y + 1] == color))
    result = np.where(frame == 5, 5, 0)
    for hole in holes:
        if any(r in (0, frame.shape[0] - 1) or c in (0, frame.shape[1] - 1) for r, c in hole):
            continue
        hb = _bbox(np.isin(np.indices(frame.shape)[0] * 100 + np.indices(frame.shape)[1], [r * 100 + c for r, c in hole]))
        if hb is None:
            continue
        hr0, hr1, hc0, hc1 = hb
        hh, hw = hr1 - hr0 + 1, hc1 - hc0 + 1
        chosen = None
        for color, mask in sprites:
            variants = [mask, np.fliplr(mask), np.flipud(mask)]
            variants += [np.rot90(mask, k) for k in range(1, 4)]
            for v in variants:
                if v.shape == (hh, hw):
                    chosen = color
                    break
            if chosen is not None:
                break
        for r, c in hole:
            result[r, c] = chosen if chosen is not None else 1
    return result


def _nor_region_fill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    _, _, rints, cints = _split_by_full_lines(arr, 5)
    if not rints or not cints:
        comps = _components(arr != 5)
        rints = []
        cints = []
        for comp in comps:
            box = _bbox(np.isin(np.indices(arr.shape)[0] * 100 + np.indices(arr.shape)[1], [r * 100 + c for r, c in comp]))
            if box:
                r0, r1, c0, c1 = box
                rints.append((r0, r1))
                cints.append((c0, c1))
    regions = []
    for r0, r1 in rints:
        for c0, c1 in cints:
            sub = arr[r0 : r1 + 1, c0 : c1 + 1]
            colors = [int(c) for c in np.unique(sub) if c not in (0, 5)]
            if len(colors) < 2:
                continue
            a, b = colors[:2]
            empty = int(np.sum((sub != a) & (sub != b) & (sub != 5)))
            fill = min(colors, key=lambda c: np.sum(sub == c))
            regions.append((empty, fill, r0, r1, c0, c1))
    if not regions:
        return out
    _, fill, r0, r1, c0, c1 = max(regions, key=lambda x: x[0])
    mask = (out[r0 : r1 + 1, c0 : c1 + 1] == 0)
    out[r0 : r1 + 1, c0 : c1 + 1][mask] = fill
    return out


def _complete_frame_mirror_ray(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    pts = np.argwhere(arr != 0)
    if len(pts) == 0:
        return out
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    color = _majority(arr[arr != 0], 2)
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size:
        if (c1 - c0) >= (r1 - r0):
            out[r0 + 1 : r1, c0 + 1 : c1] = np.maximum(inner, np.fliplr(inner))
            rr = (r0 + r1) // 2
            for cc in range(c1 + 1, out.shape[1]):
                if out[rr, cc] == 0:
                    out[rr, cc] = color
        else:
            out[r0 + 1 : r1, c0 + 1 : c1] = np.maximum(inner, np.flipud(inner))
            cc = (c0 + c1) // 2
            for rr in range(r1 + 1, out.shape[0]):
                if out[rr, cc] == 0:
                    out[rr, cc] = color
    return out


def _stamp_repeat_recolor(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    comps = []
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5, 9)]:
        for comp in _components(arr == color):
            if len(comp) >= 3:
                b = _bbox(arr == color)
                if b:
                    r0, r1, c0, c1 = b
                    comps.append((color, arr[r0 : r1 + 1, c0 : c1 + 1].copy()))
    controls = np.argwhere(arr == 9)
    n = max(1, min(5, len(controls)))
    targets = np.argwhere(arr == 5)
    if comps:
        color, stamp = max(comps, key=lambda x: np.sum(x[1] == x[0]))
    else:
        color, stamp = 1, np.array([[1, 1], [1, 0]])
    control_color = int(arr[tuple(controls[0])]) if len(controls) else 2
    for i, (r, c) in enumerate(targets[:8]):
        for k in range(n):
            sr, sc = int(r), int(c) + k * stamp.shape[1]
            recolored = np.where(stamp != 0, color if k % 2 == 0 else control_color, 0)
            _paste(out, recolored, sr, sc)
    return out


def _three_point_expand(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        pts = np.argwhere(arr == color)
        if len(pts) < 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        corners = {(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))}
        dist = max(1, sum((int(r), int(c)) in corners for r, c in pts))
        for d in range(1, dist + 1):
            a, b = max(0, r0 - d), min(out.shape[0] - 1, r1 + d)
            x, y = max(0, c0 - d), min(out.shape[1] - 1, c1 + d)
            out[a, x : y + 1] = 5
            out[b, x : y + 1] = 5
            out[a : b + 1, x] = 5
            out[a : b + 1, y] = 5
    return out


def _stripe_marker_mirror(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    markers = [c for c in range(arr.shape[1]) if np.sum(arr[:, c] == 5) >= 2]
    if not markers:
        return out
    m = markers[len(markers) // 2]
    colors = [int(c) for c in np.unique(arr) if c not in (0, 5)]
    if len(colors) < 2:
        colors = [1, 2]
    left = np.zeros_like(arr[:, :m])
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            left[r, c] = colors[(r + c) % 2]
    out[:, :m] = np.where(out[:, :m] == 5, 5, left)
    right = np.fliplr(left)
    swap = {colors[0]: colors[1], colors[1]: colors[0]}
    for r in range(out.shape[0]):
        for c in range(m + 1, out.shape[1]):
            src = right[r, (c - m - 1) % max(1, right.shape[1])]
            out[r, c] = swap.get(int(src), int(src))
    return out


def _nearest_seed_flip(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    horizontal = np.argwhere(arr == 8)
    vertical = np.argwhere(arr == 7)
    comps = _components((arr != 0) & (arr != 7) & (arr != 8) & (arr != 5))
    for comp in comps:
        if len(comp) <= 1:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        sub = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        center = np.array([(r0 + r1) / 2, (c0 + c1) / 2])
        dh = min((abs(center[0] - r) for r, _ in horizontal), default=999)
        dv = min((abs(center[1] - c) for _, c in vertical), default=999)
        out[r0 : r1 + 1, c0 : c1 + 1] = np.flipud(sub) if dh <= dv else np.fliplr(sub)
    return out


def _band_majority_compress(arr: np.ndarray) -> np.ndarray:
    seed_rows = sorted(set(int(r) for r, _ in np.argwhere(arr == 8)))
    if not seed_rows:
        seed_rows = [arr.shape[0] // 2]
    cuts = [-1] + seed_rows + [arr.shape[0]]
    rows = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a <= 1:
            continue
        sub = arr[a + 1 : b]
        vals = [int(v) for v in sub.flatten() if v not in (0, 5, 7, 8)]
        rows.append([_majority(vals, 0)] * arr.shape[1])
    return np.array(rows or [[0]], dtype=int)


def _flood_repair_period(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    start = tuple(np.argwhere(arr == 9)[0]) if np.any(arr == 9) else (0, 0)
    reachable = set(_comp_from(arr != 5, start))
    vals = [int(arr[r, c]) for r, c in reachable if arr[r, c] not in (0, 9)]
    period_colors = sorted(set(vals)) or [1, 2, 3]
    for r, c in reachable:
        if out[r, c] in (0, 9):
            out[r, c] = period_colors[(r + c) % len(period_colors)]
    out[arr == 5] = 0
    return out


def _hollow_object_rays(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr)
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        for comp in _components(arr == color):
            rows, cols = [r for r, _ in comp], [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            hollow = (r1 - r0 >= 2 and c1 - c0 >= 2 and np.any(arr[r0 + 1 : r1, c0 + 1 : c1] == 0))
            if not hollow:
                continue
            for r, c in comp:
                out[r, c] = color
            for dr, dc, sr, sc in ((-1, -1, r0, c0), (-1, 1, r0, c1), (1, -1, r1, c0), (1, 1, r1, c1)):
                rr, cc = sr + dr, sc + dc
                while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    if arr[rr, cc] == 5:
                        out[rr, cc] = color
                    elif out[rr, cc] == 0:
                        out[rr, cc] = 2
                    rr += dr
                    cc += dc
    return out


def _panel_chain_overlay(arr: np.ndarray) -> np.ndarray:
    _, _, rints, cints = _split_by_full_lines(arr, 5)
    cells = []
    for r0, r1 in rints:
        for c0, c1 in cints:
            sub = _crop(arr[r0 : r1 + 1, c0 : c1 + 1])
            if sub.size:
                cells.append(sub)
    if not cells:
        return arr.copy()
    h = max(c.shape[0] for c in cells)
    w = max(c.shape[1] for c in cells)
    out = np.zeros((h, w), dtype=int)
    for sub in sorted(cells, key=lambda x: (x.shape[0] + x.shape[1], np.count_nonzero(x))):
        _paste(out, sub, 0, 0)
    return out


def _glyph_table_fill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    _, _, rints, cints = _split_by_full_lines(arr, 5)
    for ri, (r0, r1) in enumerate(rints):
        for ci, (c0, c1) in enumerate(cints):
            sub = arr[r0 : r1 + 1, c0 : c1 + 1]
            vals = [int(v) for v in sub.flatten() if v not in (0, 5)]
            if not vals:
                continue
            decoded = (sum(vals) + len(vals)) % 8 + 1
            corner = ri in (0, len(rints) - 1) and ci in (0, len(cints) - 1)
            edge = ri in (0, len(rints) - 1) or ci in (0, len(cints) - 1)
            if corner:
                out[r0 : r1 + 1, c0 : c1 + 1] = decoded
            elif edge:
                out[r0 : r1 + 1, c0 : c1 + 1] = 0
            else:
                out[r0 : r1 + 1, c0 : c1 + 1] = decoded
                mask = sub != 0
                out[r0 : r1 + 1, c0 : c1 + 1][mask] = 9 - decoded if decoded < 9 else 1
    return out


def _gravity_stamp(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    grey_boxes = []
    for comp in _components(arr == 5):
        b = _bbox(np.isin(np.indices(arr.shape)[0] * 100 + np.indices(arr.shape)[1], [r * 100 + c for r, c in comp]))
        if b:
            grey_boxes.append(b)
    pixels = [(int(r), int(c), int(arr[r, c])) for r, c in np.argwhere((arr != 0) & (arr != 5))]
    for r, c, color in pixels:
        candidates = [b for b in grey_boxes if b[2] <= c <= b[3] and b[0] > r]
        if candidates:
            b = min(candidates, key=lambda x: x[0])
            out[b[0] : b[1] + 1, b[2] : b[3] + 1] = np.where(out[b[0] : b[1] + 1, b[2] : b[3] + 1] == 5, color, out[b[0] : b[1] + 1, b[2] : b[3] + 1])
    return out


def _count_upscale_symmetry(arr: np.ndarray) -> np.ndarray:
    small = sum(1 for comp in _components((arr != 0) & (arr != 5) & (arr != 4)) if len(comp) <= 3)
    k = max(1, min(3, small or 2))
    mask = arr == 4
    box = _bbox((arr != 0) & ~mask)
    if box is None:
        return arr.copy()
    r0, r1, c0, c1 = box
    pat = arr[r0 : r1 + 1, c0 : c1 + 1]
    up = np.repeat(np.repeat(pat, k, axis=0), k, axis=1)
    if up.shape[1] <= 30:
        up = np.maximum(up, np.fliplr(up))
    if up.shape[0] <= 30:
        up = np.maximum(up, np.flipud(up))
    return _finish(up)


def _nested_triangles(arr: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(arr) if c not in (0, 5)]
    size = min(30, max(4, 2 * len(colors) + 3))
    out = np.zeros((size, size), dtype=int)
    corner = tuple(np.argwhere(arr == 9)[0]) if np.any(arr == 9) else (0, 0)
    flip_r = corner[0] > arr.shape[0] // 2
    flip_c = corner[1] > arr.shape[1] // 2
    for depth, color in enumerate(colors or [1, 2, 3]):
        lim = size - depth
        for r in range(depth, lim):
            for c in range(depth, lim):
                if r + c - 2 * depth < max(1, len(colors) + 1 - depth):
                    out[r, c] = color
    if flip_r:
        out = np.flipud(out)
    if flip_c:
        out = np.fliplr(out)
    return out


def _region_count_corner_copy(arr: np.ndarray) -> np.ndarray:
    regions = _components(arr == 0)
    k = max(1, min(4, len(regions)))
    box = _bbox((arr != 0) & (arr != 5))
    if box is None:
        return arr.copy()
    r0, r1, c0, c1 = box
    pat = arr[r0 : r1 + 1, c0 : c1 + 1]
    up = np.repeat(np.repeat(pat, k, axis=0), k, axis=1)
    out = np.zeros_like(arr)
    stamps = [up, np.rot90(up), np.rot90(up, 2), np.rot90(up, 3)]
    anchors = [(0, 0), (0, out.shape[1] - up.shape[1]), (out.shape[0] - up.shape[0], 0), (out.shape[0] - up.shape[0], out.shape[1] - up.shape[1])]
    for stamp, (r, c) in zip(stamps, anchors):
        _paste(out, _finish(stamp), max(0, r), max(0, c))
    return out


def _bridge_stretch(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    grey = arr == 5
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        pts = np.argwhere(arr == color)
        if len(pts) == 0:
            continue
        for r, c in pts:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r), int(c)
                while 0 <= rr + dr < out.shape[0] and 0 <= cc + dc < out.shape[1]:
                    rr += dr
                    cc += dc
                    if grey[rr, cc]:
                        out[rr, cc] = 0
                        continue
                    if out[rr, cc] not in (0, color):
                        break
                    out[rr, cc] = color
    return out


def _domino_rays_scan_fill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    endpoints_by_row = defaultdict(list)
    endpoints_by_col = defaultdict(list)
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        pts = np.argwhere(arr == color)
        if len(pts) < 2:
            continue
        horizontal = len(set(int(r) for r, _ in pts)) == 1
        for r, c in pts:
            dirs = [(0, -1), (0, 1)] if horizontal else [(-1, 0), (1, 0)]
            for dr, dc in dirs:
                rr, cc = int(r), int(c)
                while 0 <= rr + dr < out.shape[0] and 0 <= cc + dc < out.shape[1] and arr[rr + dr, cc + dc] != 5:
                    rr += dr
                    cc += dc
                    if out[rr, cc] == 0:
                        out[rr, cc] = color
                if horizontal:
                    endpoints_by_row[rr].append(cc)
                else:
                    endpoints_by_col[cc].append(rr)
    for r, cols in endpoints_by_row.items():
        cols.sort()
        for a, b in zip(cols[::2], cols[1::2]):
            out[r, min(a, b) : max(a, b) + 1] = np.where(out[r, min(a, b) : max(a, b) + 1] == 0, 2, out[r, min(a, b) : max(a, b) + 1])
    for c, rows in endpoints_by_col.items():
        rows.sort()
        for a, b in zip(rows[::2], rows[1::2]):
            out[min(a, b) : max(a, b) + 1, c] = np.where(out[min(a, b) : max(a, b) + 1, c] == 0, 2, out[min(a, b) : max(a, b) + 1, c])
    return out


def _legend_quadrant_upscale(arr: np.ndarray) -> np.ndarray:
    colors = [int(v) for v in arr[:2, :2].flatten() if int(v) != 0]
    while len(colors) < 4:
        colors.append(len(colors) + 1)
    mask = arr != 0
    box = _bbox(mask)
    if box is None:
        return arr.copy()
    r0, r1, c0, c1 = box
    sub = arr[r0 : r1 + 1, c0 : c1 + 1].copy()
    cr, cc = sub.shape[0] // 2, sub.shape[1] // 2
    recol = np.zeros_like(sub)
    for r, c in np.argwhere(sub != 0):
        idx = (r >= cr) * 2 + (c >= cc)
        recol[r, c] = colors[idx]
    return _finish(np.repeat(np.repeat(recol, 2, axis=0), 2, axis=1))


def _anti_diag_path_fill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    valid_colors = [int(c) for c in np.unique(arr) if c not in (0, 5)]
    for color in valid_colors:
        for s in range(h + w - 1):
            pts = [(r, s - r) for r in range(h) if 0 <= s - r < w]
            if any(arr[r, c] == color for r, c in pts):
                for r, c in pts:
                    if out[r, c] == 0:
                        out[r, c] = color
    return out


def _scanline_slide_objects(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for r in range(out.shape[0]):
        xs = np.argwhere(out[r] != 0).flatten()
        if len(xs) >= 2:
            a, b = int(xs.min()), int(xs.max())
            fill = 1 if r % 2 == 0 else 8
            out[r, a : b + 1] = np.where(out[r, a : b + 1] == 0, fill, out[r, a : b + 1])
    for color, direction in ((1, 1), (8, -1)):
        carrier = out == color
        for comp in _components((out != 0) & ~carrier & (out != 5)):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            sub = out[r0 : r1 + 1, c0 : c1 + 1].copy()
            out[r0 : r1 + 1, c0 : c1 + 1][sub != 5] = 0
            nc = min(max(0, c0 + direction), out.shape[1] - sub.shape[1])
            _paste(out, sub, r0, nc)
    return out


def _direction_field_absorb(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    dirs = {1: (-1, -1), 8: (0, 1), 2: (1, 0)}
    fields = [(int(r), int(c), int(arr[r, c])) for r, c in np.argwhere(np.isin(arr, list(dirs)))]
    for r, c in np.argwhere((arr != 0) & ~np.isin(arr, list(dirs)) & (arr != 5)):
        if not fields:
            continue
        fr, fc, fcol = min(fields, key=lambda p: abs(p[0] - r) + abs(p[1] - c))
        dr, dc = dirs[fcol]
        nr, nc = int(r), int(c)
        out[nr, nc] = 0
        while 0 <= nr + dr < out.shape[0] and 0 <= nc + dc < out.shape[1] and out[nr + dr, nc + dc] == 0:
            nr += dr
            nc += dc
        out[nr, nc] = int(arr[r, c])
    return out


def _decode_upscale_fill_windows(arr: np.ndarray) -> np.ndarray:
    box = _bbox(arr != 0)
    if box is None:
        return arr.copy()
    r0, r1, c0, c1 = box
    grid = arr[r0 : r1 + 1, c0 : c1 + 1]
    decoded = np.where(grid == 5, 0, grid)
    up = np.repeat(np.repeat(decoded, 2, axis=0), 2, axis=1)
    for r in range(0, up.shape[0] - 1, 2):
        for c in range(0, up.shape[1] - 1, 2):
            if np.all(up[r : r + 2, c : c + 2] == 0):
                up[r : r + 2, c : c + 2] = 1
    return _finish(up)


def _slide_hollow_filter(arr: np.ndarray) -> np.ndarray:
    moved = np.zeros_like(arr)
    dir_by_color = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    comps = []
    for color, (dr, dc) in dir_by_color.items():
        for comp in _components(arr == color):
            rows, cols = [r for r, _ in comp], [c for _, c in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            shift = 0
            while True:
                nr0, nr1 = r0 + dr * (shift + 1), r1 + dr * (shift + 1)
                nc0, nc1 = c0 + dc * (shift + 1), c1 + dc * (shift + 1)
                if nr0 < 0 or nc0 < 0 or nr1 >= arr.shape[0] or nc1 >= arr.shape[1]:
                    break
                shift += 1
            pts = [(r + dr * shift, c + dc * shift) for r, c in comp]
            comps.append((color, pts))
            for r, c in pts:
                moved[r, c] = color
    masks = [set(pts) for _, pts in comps]
    touch = [False] * len(comps)
    for i, s in enumerate(masks):
        for j, t in enumerate(masks):
            if i != j and any(abs(r - a) + abs(c - b) == 1 for r, c in s for a, b in t):
                touch[i] = True
    out = np.zeros_like(arr)
    for keep, (color, pts) in zip(touch, comps):
        if keep or len(comps) == 1:
            for r, c in pts:
                out[r, c] = color
    return out


def _bars_to_concentric(arr: np.ndarray) -> np.ndarray:
    bars = []
    for color in [int(c) for c in np.unique(arr) if c != 0]:
        for comp in _components(arr == color):
            rows, cols = [r for r, _ in comp], [c for _, c in comp]
            if len(set(cols)) == 1:
                bars.append((min(cols), len(comp), color))
    bars.sort(key=lambda x: x[1], reverse=True)
    size = min(30, max(3, 2 * len(bars) + 3))
    out = np.zeros((size, size), dtype=int)
    for i, (_, thick, color) in enumerate(sorted(bars, key=lambda x: x[0])):
        for d in range(i, min(i + max(1, thick // 3), size // 2 + 1)):
            out[d, d : size - d] = color
            out[size - d - 1, d : size - d] = color
            out[d : size - d, d] = color
            out[d : size - d, size - d - 1] = color
    return out


def _bouncing_palette_ray(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    palette = [int(v) for v in arr[0] if v not in (0, 5)]
    if not palette:
        palette = [1, 2, 3]
    starts = np.argwhere(arr == 9)
    r, c = (int(starts[0][0]), int(starts[0][1])) if len(starts) else (1, 1)
    dr, dc = 1, 1
    idx = 0
    for _ in range(arr.shape[0] * arr.shape[1]):
        if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
            break
        out[r, c] = palette[idx % len(palette)]
        nr, nc = r + dr, c + dc
        bounced = False
        if nr < 0 or nr >= out.shape[0] or arr[nr, c] == 5:
            dr *= -1
            bounced = True
        if nc < 0 or nc >= out.shape[1] or arr[r, nc] == 5:
            dc *= -1
            bounced = True
        if bounced:
            idx += 1
        r += dr
        c += dc
    return out


def _track_symmetry(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    tracks = {int(c): np.argwhere(arr == c) for c in np.unique(arr) if c not in (0, 5)}
    for color, pts in tracks.items():
        if len(pts) < 2:
            continue
        rs, cs = pts[:, 0], pts[:, 1]
        if len(set(rs)) == 1:
            axis = int(rs[0])
            for r, c in np.argwhere(out == color):
                rr = 2 * axis - int(r)
                if 0 <= rr < out.shape[0]:
                    out[rr, c] = color
        elif len(set(cs)) == 1:
            axis = int(cs[0])
            for r, c in np.argwhere(out == color):
                cc = 2 * axis - int(c)
                if 0 <= cc < out.shape[1]:
                    out[r, cc] = color
    out[arr == 5] = 0
    return out


def _legend_filter_upscale(arr: np.ndarray) -> np.ndarray:
    legend = set(int(v) for v in arr[0] if v != 0)
    comps = _components(arr != 0)
    best = None
    for comp in comps:
        vals = set(int(arr[r, c]) for r, c in comp)
        if legend and legend.issubset(vals):
            best = comp
            break
    if best is None and comps:
        best = max(comps, key=len)
    if best is None:
        return arr.copy()
    mask = np.zeros_like(arr, dtype=bool)
    for r, c in best:
        mask[r, c] = True
    stamp = _crop(np.where(mask, arr, 0))
    return _finish(np.repeat(np.repeat(stamp, 2, axis=0), 2, axis=1))


def _height_depth_overlay(arr: np.ndarray) -> np.ndarray:
    comps = []
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        for comp in _components(arr == color):
            box = _bbox(np.isin(np.indices(arr.shape)[0] * 100 + np.indices(arr.shape)[1], [r * 100 + c for r, c in comp]))
            if box:
                comps.append((min(r for r, _ in comp), color, _crop(np.where(arr == color, color, 0))))
    if not comps:
        return arr.copy()
    h = max(s.shape[0] for _, _, s in comps)
    w = max(s.shape[1] for _, _, s in comps)
    out = np.zeros((h, w), dtype=int)
    for _, _, stamp in sorted(comps):
        _paste(out, stamp, 0, 0)
    return out


def _cross_gap_glyph(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    glyph_color = _majority(arr[arr != 0], 1)
    for r in range(1, out.shape[0] - 1):
        for c in range(1, out.shape[1] - 1):
            nbrs = [arr[r + 1, c], arr[r - 1, c], arr[r, c + 1], arr[r, c - 1]]
            if arr[r, c] == 0 and sum(v != 0 for v in nbrs) >= 3:
                out[r - 1 : r + 2, c] = glyph_color
                out[r, c - 1 : c + 2] = glyph_color
    return out


def _cell_layer_overlay(arr: np.ndarray) -> np.ndarray:
    return _panel_chain_overlay(arr)


def _nearest_marker_move(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    markers = [(int(r), int(c), int(arr[r, c])) for r, c in np.argwhere(np.isin(arr, [1, 2, 3, 4, 6, 7, 8]))]
    marker_cells = {(r, c) for r, c, _ in markers}
    for comp in _components(arr == 5):
        rows, cols = [r for r, _ in comp], [c for _, c in comp]
        cr = sum(rows) / len(rows)
        cc = sum(cols) / len(cols)
        if not markers:
            continue
        mr, mc, color = min(markers, key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
        dr = 0 if mr == cr else (1 if mr > cr else -1)
        dc = 0 if mc == cc else (1 if mc > cc else -1)
        dist = int(abs(mr - cr) + abs(mc - cc))
        step = max(1, min(dist, 6))
        nr0 = min(max(0, min(rows) + dr * step), arr.shape[0] - (max(rows) - min(rows) + 1))
        nc0 = min(max(0, min(cols) + dc * step), arr.shape[1] - (max(cols) - min(cols) + 1))
        stamp = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=int)
        for r, c in comp:
            out[r, c] = 0
            stamp[r - min(rows), c - min(cols)] = 5
        _paste(out, stamp, nr0, nc0)
        if (mr, mc) in marker_cells:
            out[mr, mc] = color
    return out


def _seed_tile_transform_layers(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    seed = arr[:3, :3]
    for color in [int(c) for c in np.unique(arr) if c != 0]:
        locs = np.argwhere(seed == color)
        if len(locs) == 0:
            continue
        r, c = locs[0]
        mask = out == color
        transformed = np.zeros_like(out)
        transformed[mask] = color
        if r == 0:
            transformed = np.flipud(transformed)
        elif c == 0:
            transformed = np.fliplr(transformed)
        elif r == c:
            transformed = np.roll(transformed, 1, axis=1)
        out[mask] = 0
        out = np.where(transformed != 0, transformed, out)
    return out


def _nested_reverse_concat(arr: np.ndarray) -> np.ndarray:
    comps = []
    for comp in _components(arr != 0):
        mask = np.zeros_like(arr, dtype=bool)
        for r, c in comp:
            mask[r, c] = True
        crop = _crop(np.where(mask, arr, 0))
        colors = [int(c) for c in np.unique(crop) if c != 0]
        rev = {a: b for a, b in zip(colors, reversed(colors))}
        for a, b in rev.items():
            crop[crop == a] = -b
        crop = np.abs(crop)
        comps.append((len(colors), crop))
    if not comps:
        return arr.copy()
    comps.sort(key=lambda x: x[0])
    h = max(c.shape[0] for _, c in comps)
    w = sum(c.shape[1] for _, c in comps)
    out = np.zeros((min(30, h), min(30, w)), dtype=int)
    cc = 0
    for _, crop in comps:
        _paste(out, crop, 0, cc)
        cc += crop.shape[1]
    return out


def _edge_stretch_symmetry(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, 1, mode="constant")
    return _finish(np.maximum(np.maximum(padded, np.fliplr(padded)), np.flipud(padded)))


def _damaged_quadrant_repair(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    quads = [(0, h // 2, 0, w // 2), (0, h // 2, w // 2, w), (h // 2, h, 0, w // 2), (h // 2, h, w // 2, w)]
    chosen = max(quads, key=lambda q: np.sum(arr[q[0] : q[1], q[2] : q[3]] == 0))
    r0, r1, c0, c1 = chosen
    out = arr[r0:r1, c0:c1].copy()
    for r, c in np.argwhere(out == 0):
        vals = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] != 0:
                vals.append(out[rr, cc])
        if vals:
            out[r, c] = 1
    return out


def _periodic_strip_mirror(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    rows = [r for r in range(arr.shape[0]) if np.count_nonzero(arr[r]) >= 2]
    if not rows:
        return out
    r = rows[len(rows) // 2]
    vals = [int(v) for v in arr[r] if v != 0]
    period = vals[: max(1, min(4, len(vals)))]
    for c in range(arr.shape[1]):
        out[r, c] = period[c % len(period)]
    indicator = _majority([v for v in arr.flatten() if v not in (0, *period)], period[0])
    target_r = r - 1 if r > arr.shape[0] // 2 else r + 1
    if 0 <= target_r < arr.shape[0]:
        out[target_r] = np.where(out[r] != 0, indicator, 0)
    return out


def _sort_segments_wedge(arr: np.ndarray) -> np.ndarray:
    segs = []
    for color in [int(c) for c in np.unique(arr) if c != 0]:
        for comp in _components(arr == color):
            rows = {r for r, _ in comp}
            if len(rows) == 1:
                segs.append((len(comp), color))
    if not segs:
        return arr.copy()
    segs.sort()
    width = min(30, max(l for l, _ in segs) + len(segs))
    out = np.zeros((min(30, len(segs) * 2), width), dtype=int)
    base = out.shape[0] - 1
    for i, (length, color) in enumerate(segs):
        r = base - 2 * i
        if r < 0:
            break
        out[r, :length] = color
        if i > 0:
            prev = segs[i - 1][0]
            for rr in range(r + 1, min(base - 2 * (i - 1), out.shape[0])):
                out[rr, min(prev, length) : max(prev, length)] = color
    return out


def _frame_mapping_upscale(arr: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(arr) if c != 0]
    mapping = {a: b for a, b in zip(colors, reversed(colors))}
    box = _bbox(arr == 0)
    pattern = arr.copy()
    for a, b in mapping.items():
        pattern[arr == a] = b
    return _finish(np.repeat(np.repeat(_crop(pattern), 2, axis=0), 2, axis=1))


def _two_panel_density_overlay(arr: np.ndarray) -> np.ndarray:
    _, cols, _, _ = _split_by_full_lines(arr, 5)
    if cols:
        c = cols[0]
        left, right = arr[:, :c], arr[:, c + 1 :]
    else:
        mid = arr.shape[1] // 2
        left, right = arr[:, :mid], arr[:, mid:]
    h, w = max(left.shape[0], right.shape[0]), max(left.shape[1], right.shape[1])
    out = np.zeros((h, w), dtype=int)
    density_left = np.count_nonzero(left == 9)
    density_right = np.count_nonzero(right == 9)
    for panel, prefer in ((left, density_left >= density_right), (right, density_right > density_left)):
        for r in range(min(h, panel.shape[0])):
            for c in range(min(w, panel.shape[1])):
                if panel[r, c] != 0 and (out[r, c] == 0 or prefer):
                    out[r, c] = panel[r, c]
    return out


def _outlier_quadrant_stamp(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    quads = [(0, h // 2, 0, w // 2), (0, h // 2, w // 2, w), (h // 2, h, 0, w // 2), (h // 2, h, w // 2, w)]
    counts = [np.count_nonzero(arr[r0:r1, c0:c1]) for r0, r1, c0, c1 in quads]
    idx = int(np.argmax(np.abs(np.array(counts) - np.median(counts))))
    sr0, sr1, sc0, sc1 = quads[idx]
    stamp = arr[sr0:sr1, sc0:sc1] != 0
    out = arr.copy()
    for j, (r0, r1, c0, c1) in enumerate(quads):
        if j == idx:
            continue
        color = _majority(arr[r0:r1, c0:c1].flatten(), j + 1)
        sub = out[r0:r1, c0:c1]
        hh, ww = min(sub.shape[0], stamp.shape[0]), min(sub.shape[1], stamp.shape[1])
        sub[:hh, :ww][stamp[:hh, :ww]] = color
    return out


def _diagonal_scan_crop(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    trace = np.zeros_like(arr, dtype=bool)
    for offset in range(-h, w):
        for r in range(h):
            c = r + offset
            if 0 <= c < w:
                trace[r, c] = True
    best = None
    for comp in _components(arr == 5):
        rows, cols = [r for r, _ in comp], [c for _, c in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        score = int(np.sum(trace[r0 : r1 + 1, c0 : c1 + 1]))
        if best is None or score > best[0]:
            best = (score, r0, r1, c0, c1)
    if best is None:
        return _crop(arr)
    _, r0, r1, c0, c1 = best
    return arr[r0 + 1 : max(r0 + 2, r1), c0 + 1 : max(c0 + 2, c1)].copy()


def _chamber_majority(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    regions = _components(arr != 5)
    seeds = []
    for comp in regions:
        vals = [int(arr[r, c]) for r, c in comp if arr[r, c] not in (0, 5)]
        if vals:
            seeds.append((_majority(vals), set(comp)))
    for comp in regions:
        if any(arr[r, c] not in (0, 5) for r, c in comp):
            continue
        neighbors = []
        s = set(comp)
        for color, seed in seeds:
            if any(abs(r - a) + abs(c - b) == 2 for r, c in s for a, b in seed):
                neighbors.append(color)
        fill = _majority(neighbors, 1)
        for r, c in comp:
            if out[r, c] == 0:
                out[r, c] = fill
    return out


def _gravity_into_frames(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    frames = []
    for comp in _components(arr == 5):
        rows, cols = [r for r, _ in comp], [c for _, c in comp]
        if len(set(rows)) > 2 and len(set(cols)) > 2:
            frames.append((min(rows), max(rows), min(cols), max(cols)))
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        for comp in _components(arr == color):
            rows, cols = [r for r, _ in comp], [c for _, c in comp]
            h, w = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
            target = next((f for f in frames if f[1] - f[0] - 1 >= h and f[3] - f[2] - 1 >= w), None)
            for r, c in comp:
                out[r, c] = 0
            if target:
                r0, _, c0, _ = target
                for r, c in comp:
                    out[r0 + 1 + r - min(rows), c0 + 1 + c - min(cols)] = color
            else:
                out[-1, :] = color
    return out


def _compress_rotate_neighbors(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr)
    markers = np.argwhere(arr == 5)
    if len(markers) == 0:
        return arr.copy()
    mr, mc = map(int, markers[0])
    buckets = {(mr - 1, mc): [], (mr, mc + 1): [], (mr + 1, mc): [], (mr, mc - 1): []}
    for r, c in np.argwhere((arr != 0) & (arr != 5)):
        if abs(r - mr) >= abs(c - mc):
            key = (mr - 1, mc) if r < mr else (mr + 1, mc)
        else:
            key = (mr, mc - 1) if c < mc else (mr, mc + 1)
        buckets[key].append(int(arr[r, c]))
    order = [(mr - 1, mc), (mr, mc + 1), (mr + 1, mc), (mr, mc - 1)]
    vals = [_majority(buckets[k], 0) for k in order]
    vals = [vals[-1]] + vals[:-1]
    out[mr, mc] = 5
    for (r, c), v in zip(order, vals):
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            out[r, c] = v
    return out


def _detour_crop_tile(arr: np.ndarray) -> np.ndarray:
    path = np.zeros_like(arr)
    c = arr.shape[1] // 2
    for r in range(arr.shape[0]):
        while c < arr.shape[1] and arr[r, c] == 5:
            c += 1
        if c >= arr.shape[1]:
            break
        path[r, c] = 2
        if r % 2 == 0:
            c = max(0, c - 1)
        else:
            c = min(arr.shape[1] - 1, c + 1)
    tile = _crop(path)
    out = np.zeros_like(arr)
    for r in range(0, out.shape[0], max(1, tile.shape[0])):
        for c in range(0, out.shape[1], max(1, tile.shape[1])):
            stamp = tile if (r // max(1, tile.shape[0]) + c // max(1, tile.shape[1])) % 2 == 0 else np.fliplr(tile)
            _paste(out, stamp, r, c)
    return out


def _closed_regions_concat(arr: np.ndarray) -> np.ndarray:
    crops = []
    for comp in _components(arr == 0):
        if any(r in (0, arr.shape[0] - 1) or c in (0, arr.shape[1] - 1) for r, c in comp):
            continue
        mask = np.zeros_like(arr, dtype=int)
        for r, c in comp:
            mask[r, c] = 3
        crops.append(_crop(mask))
    if not crops:
        return arr.copy()
    crops.sort(key=lambda x: np.count_nonzero(x))
    h = max(c.shape[0] for c in crops)
    w = min(30, sum(c.shape[1] for c in crops))
    out = np.zeros((h, w), dtype=int)
    cc = 0
    for crop in crops:
        _paste(out, crop, 0, cc)
        cc += crop.shape[1]
    return out


def _ring_depth_upscale(arr: np.ndarray) -> np.ndarray:
    comps = []
    center = np.array(arr.shape) / 2
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        for comp in _components(arr == color):
            rows, cols = [r for r, _ in comp], [c for _, c in comp]
            depth = int(max(1, min(3, min(abs(np.mean(rows) - center[0]), abs(np.mean(cols) - center[1])) // 2 + 1)))
            stamp = _crop(np.where(arr == color, color, 0))
            comps.append((min(cols), np.repeat(np.repeat(stamp, depth, axis=0), depth, axis=1)))
    if not comps:
        return arr.copy()
    comps.sort()
    h = min(30, max(s.shape[0] for _, s in comps))
    w = min(30, sum(s.shape[1] for _, s in comps))
    out = np.zeros((h, w), dtype=int)
    cc = 0
    for _, s in comps:
        _paste(out, _finish(s), 0, cc)
        cc += s.shape[1]
    return out


def _fork_ray_fill_container(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    starts = np.argwhere((arr != 0) & (arr != 5))
    if len(starts) == 0:
        return out
    sr, sc = min((int(r), int(c)) for r, c in starts)
    color = int(arr[sr, sc])
    q = deque([(sr, sc, 0, 1)])
    seen = set()
    while q:
        r, c, dr, dc = q.popleft()
        while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            if (r, c, dr, dc) in seen:
                break
            seen.add((r, c, dr, dc))
            if arr[r, c] == 5:
                q.append((r - 1, c, -1, 0))
                q.append((r + 1, c, 1, 0))
                break
            if out[r, c] == 0:
                out[r, c] = color
            r += dr
            c += dc
    for comp in _components(arr == 5):
        rows, cols = [r for r, _ in comp], [c for _, c in comp]
        if len(set(rows)) >= 2 and len(set(cols)) >= 2:
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if np.any(out[r0 : r1 + 1, c0 : c1 + 1] == color):
                out[r0 + 1 : r1, c0 + 1 : c1] = color
    return out


def _frame_ratio_sort(arr: np.ndarray) -> np.ndarray:
    frames = []
    for comp in _components(arr == 5):
        rows, cols = [r for r, _ in comp], [c for _, c in comp]
        if len(set(rows)) >= 2 and len(set(cols)) >= 2:
            frames.append((max(cols) - min(cols) + 1, max(rows) - min(rows) + 1))
    horizontal = not frames or frames[0][0] >= frames[0][1]
    sprites = []
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        stamp = _crop(np.where(arr == color, color, 0))
        sprites.append((np.count_nonzero(stamp), stamp))
    sprites.sort()
    if not sprites:
        return arr.copy()
    if horizontal:
        h = max(s.shape[0] for _, s in sprites)
        w = sum(s.shape[1] for _, s in sprites)
        out = np.zeros((min(30, h), min(30, w)), dtype=int)
        cc = 0
        for _, s in sprites:
            _paste(out, s, 0, cc)
            cc += s.shape[1]
    else:
        h = sum(s.shape[0] for _, s in sprites)
        w = max(s.shape[1] for _, s in sprites)
        out = np.zeros((min(30, h), min(30, w)), dtype=int)
        rr = 0
        for _, s in sprites:
            _paste(out, s, rr, 0)
            rr += s.shape[0]
    return out


def _overlap_frames_to_rings(arr: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(arr) if c != 0]
    size = min(30, max(3, 2 * len(colors) + 5))
    out = np.zeros((size, size), dtype=int)
    for i, color in enumerate(reversed(colors)):
        d = i
        out[d, d : size - d] = color
        out[size - d - 1, d : size - d] = color
        out[d : size - d, d] = color
        out[d : size - d, size - d - 1] = color
    return out


def _detour_histogram(arr: np.ndarray) -> np.ndarray:
    counts = {}
    for color in [int(c) for c in np.unique(arr[-1]) if c not in (0, 5)]:
        starts = np.argwhere(arr[-1] == color).flatten()
        counts[color] = max(1, len(starts))
    if not counts:
        return arr.copy()
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    h = max(v for _, v in items)
    out = np.zeros((h, len(items)), dtype=int)
    for c, (color, val) in enumerate(items):
        out[h - val :, c] = color
    return out


def _downsample_connect(arr: np.ndarray) -> np.ndarray:
    small = arr[::2, ::2].copy()
    out = small.copy()
    for color in [int(c) for c in np.unique(small) if c not in (0, 5)]:
        pts = np.argwhere(small == color)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                r0, c0 = map(int, pts[i])
                r1, c1 = map(int, pts[j])
                if abs(r1 - r0) == abs(c1 - c0):
                    blocked = False
                    dr = 1 if r1 > r0 else -1
                    dc = 1 if c1 > c0 else -1
                    for k in range(abs(r1 - r0) + 1):
                        rr, cc = 2 * (r0 + dr * k), 2 * (c0 + dc * k)
                        if 0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1] and arr[rr, cc] == 5:
                            blocked = True
                    if not blocked:
                        for k in range(abs(r1 - r0) + 1):
                            out[r0 + dr * k, c0 + dc * k] = 2
    return out


def _arith_rect_fill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    pts = np.argwhere(arr != 0)
    if len(pts) >= 2:
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        out[r0, c0 : c1 + 1] = 5
        out[r1, c0 : c1 + 1] = 5
        out[r0 : r1 + 1, c0] = 5
        out[r0 : r1 + 1, c1] = 5
    outside = set()
    for start in [(0, 0), (0, out.shape[1] - 1), (out.shape[0] - 1, 0), (out.shape[0] - 1, out.shape[1] - 1)]:
        outside.update(_comp_from(out != 5, start))
    for r, c in np.argwhere(out == 0):
        if (int(r), int(c)) not in outside:
            out[r, c] = 2
    return out


def _gravity_symbols_rays(arr: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(arr) if c not in (0, 5)]
    out = np.zeros_like(arr)
    cols = defaultdict(list)
    for color in colors:
        for r, c in np.argwhere(arr == color):
            cols[int(c)].append(color)
    for c, vals in cols.items():
        row = out.shape[0] - 1
        for color in vals:
            while row >= 0 and out[row, c] != 0:
                row -= 1
            if row < 0:
                break
            out[row, c] = color
            out[row, : c + 1] = np.where(out[row, : c + 1] == 0, color, out[row, : c + 1])
            out[: row + 1, c] = np.where(out[: row + 1, c] == 0, color, out[: row + 1, c])
    return out


def _projection_l_paths(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    for comp in _components(arr != 0):
        rows, cols = [r for r, _ in comp], [c for _, c in comp]
        cr = sum(rows) / len(rows)
        cc = int(round(sum(cols) / len(cols)))
        out[h - 1, cc] = 2
        if cc < w // 2:
            out[int(cr) : h, cc] = 1
            out[int(cr), : cc + 1] = 1
        else:
            out[int(cr) : h, cc] = 1
            out[int(cr), cc:] = 1
    return out


def _legend_stamp_rank_fill(arr: np.ndarray) -> np.ndarray:
    _, _, rints, cints = _split_by_full_lines(arr, 5)
    out = arr.copy()
    slots = []
    for r0, r1 in rints:
        for c0, c1 in cints:
            area = max(1, (r1 - r0 - 1) * (c1 - c0 - 1))
            slots.append((area, r0, r1, c0, c1))
    rank_colors = [1, 2, 3, 4, 6, 7, 8]
    for rank, (_, r0, r1, c0, c1) in enumerate(sorted(slots)):
        out[r0 : r1 + 1, c0] = 5
        out[r0 : r1 + 1, c1] = 5
        out[r0, c0 : c1 + 1] = 5
        out[r1, c0 : c1 + 1] = 5
        if r1 - r0 > 1 and c1 - c0 > 1:
            out[r0 + 1 : r1, c0 + 1 : c1] = rank_colors[rank % len(rank_colors)]
    return out


def _cartesian_gap_fill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    boxes = []
    for comp in _components(arr != 0):
        rows, cols = [r for r, _ in comp], [c for _, c in comp]
        boxes.append((min(rows), max(rows), min(cols), max(cols), _majority([arr[r, c] for r, c in comp], 1)))
    if len(boxes) < 2:
        return out
    row_starts = sorted(set(b[0] for b in boxes))
    col_starts = sorted(set(b[2] for b in boxes))
    h = max(b[1] - b[0] + 1 for b in boxes)
    w = max(b[3] - b[2] + 1 for b in boxes)
    color = _majority([b[4] for b in boxes], 1)
    for r in row_starts:
        for c in col_starts:
            if not any(abs(r - b[0]) <= 1 and abs(c - b[2]) <= 1 for b in boxes):
                out[r : min(out.shape[0], r + h), c : min(out.shape[1], c + w)] = color
    return out


_SOLVERS = {
    (51, 1): _fold_inlay,
    (51, 2): _nor_region_fill,
    (51, 3): _complete_frame_mirror_ray,
    (51, 4): _stamp_repeat_recolor,
    (51, 5): _three_point_expand,
    (51, 6): _stripe_marker_mirror,
    (51, 7): _nearest_seed_flip,
    (51, 8): _band_majority_compress,
    (52, 1): _flood_repair_period,
    (52, 2): _hollow_object_rays,
    (52, 3): _panel_chain_overlay,
    (52, 4): _glyph_table_fill,
    (52, 5): _gravity_stamp,
    (52, 6): _count_upscale_symmetry,
    (52, 7): _nested_triangles,
    (52, 8): _region_count_corner_copy,
    (53, 1): _bridge_stretch,
    (53, 2): _domino_rays_scan_fill,
    (53, 3): _legend_quadrant_upscale,
    (53, 4): _panel_chain_overlay,
    (53, 5): _flood_repair_period,
    (53, 6): _bridge_stretch,
    (53, 7): _legend_quadrant_upscale,
    (53, 8): _anti_diag_path_fill,
    (53, 9): _scanline_slide_objects,
    (53, 10): _direction_field_absorb,
    (53, 11): _decode_upscale_fill_windows,
    (53, 12): _slide_hollow_filter,
    (54, 1): _bars_to_concentric,
    (54, 2): _bouncing_palette_ray,
    (54, 3): _track_symmetry,
    (54, 4): _legend_filter_upscale,
    (54, 5): _bouncing_palette_ray,
    (54, 6): _height_depth_overlay,
    (54, 7): _cross_gap_glyph,
    (54, 8): _cell_layer_overlay,
    (54, 9): _nearest_marker_move,
    (54, 10): _seed_tile_transform_layers,
    (55, 1): _nested_reverse_concat,
    (55, 2): _edge_stretch_symmetry,
    (55, 3): _damaged_quadrant_repair,
    (55, 4): _periodic_strip_mirror,
    (55, 5): _sort_segments_wedge,
    (55, 6): _frame_mapping_upscale,
    (55, 7): _two_panel_density_overlay,
    (55, 8): _outlier_quadrant_stamp,
    (55, 9): _diagonal_scan_crop,
    (55, 10): _chamber_majority,
    (56, 1): _gravity_into_frames,
    (56, 2): _compress_rotate_neighbors,
    (56, 3): _detour_crop_tile,
    (56, 4): _closed_regions_concat,
    (56, 5): _ring_depth_upscale,
    (56, 6): _fork_ray_fill_container,
    (56, 7): _frame_ratio_sort,
    (56, 8): _overlap_frames_to_rings,
    (57, 1): _detour_histogram,
    (57, 2): _downsample_connect,
    (57, 3): _arith_rect_fill,
    (57, 4): _gravity_symbols_rays,
    (57, 5): _projection_l_paths,
    (57, 6): _legend_stamp_rank_fill,
    (57, 7): _cartesian_gap_fill,
}


def main_for(round_id: int, puzzle_id: int, grid) -> np.ndarray:
    arr = _arr(grid)
    solver = _SOLVERS.get((round_id, puzzle_id), _bridge_stretch)
    try:
        return _finish(solver(arr))
    except Exception:
        return _finish(arr)


def _rng():
    return random.Random(random.randrange(1 << 62))


def _gen_grid_lines(rng: random.Random) -> np.ndarray:
    cells = rng.randint(2, 4)
    cell = rng.randint(3, 5)
    h = cells * cell + cells - 1
    w = cells * cell + cells - 1
    g = np.zeros((h, w), dtype=int)
    for r in range(cell, h, cell + 1):
        g[r, :] = 5
    for c in range(cell, w, cell + 1):
        g[:, c] = 5
    for r in range(0, h, cell + 1):
        for c in range(0, w, cell + 1):
            color = rng.choice([1, 2, 3, 4, 6, 7, 8])
            rr = r + rng.randrange(cell)
            cc = c + rng.randrange(cell)
            g[rr, cc] = color
            if rr + 1 < min(h, r + cell):
                g[rr + 1, cc] = color
    return g


def _gen_frames(rng: random.Random) -> np.ndarray:
    h = rng.randint(14, 22)
    w = rng.randint(14, 22)
    g = np.zeros((h, w), dtype=int)
    for _ in range(rng.randint(2, 4)):
        r0 = rng.randint(1, h - 7)
        c0 = rng.randint(1, w - 7)
        hh = rng.randint(4, min(8, h - r0))
        ww = rng.randint(4, min(8, w - c0))
        color = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
        g[r0, c0 : c0 + ww] = color
        g[r0 + hh - 1, c0 : c0 + ww] = color
        g[r0 : r0 + hh, c0] = color
        g[r0 : r0 + hh, c0 + ww - 1] = color
    for _ in range(rng.randint(4, 10)):
        g[rng.randrange(h), rng.randrange(w)] = rng.choice([1, 2, 3, 4, 6, 7, 8, 9])
    return g


def _gen_sparse(rng: random.Random) -> np.ndarray:
    h = rng.randint(8, 20)
    w = rng.randint(8, 20)
    g = np.zeros((h, w), dtype=int)
    for _ in range(rng.randint(12, 35)):
        r, c = rng.randrange(h), rng.randrange(w)
        color = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        g[r, c] = color
        if rng.random() < 0.35 and r + 1 < h:
            g[r + 1, c] = color
        if rng.random() < 0.35 and c + 1 < w:
            g[r, c + 1] = color
    return g


def _gen_bars(rng: random.Random) -> np.ndarray:
    h = rng.randint(10, 20)
    w = rng.randint(10, 20)
    g = np.zeros((h, w), dtype=int)
    for i, color in enumerate(rng.sample([1, 2, 3, 4, 6, 7, 8], rng.randint(3, 5))):
        col = 1 + i * 2
        if col >= w:
            break
        height = rng.randint(3, h - 1)
        g[h - height : h, col] = color
    return g


def _gen_regions(rng: random.Random) -> np.ndarray:
    h = rng.randint(12, 20)
    w = rng.randint(12, 20)
    g = np.zeros((h, w), dtype=int)
    for r in sorted(rng.sample(range(3, h - 3), rng.randint(1, 3))):
        g[r, :] = 5
    for c in sorted(rng.sample(range(3, w - 3), rng.randint(1, 3))):
        g[:, c] = 5
    for _ in range(rng.randint(6, 16)):
        r, c = rng.randrange(h), rng.randrange(w)
        if g[r, c] == 0:
            g[r, c] = rng.choice([1, 2, 3, 4, 6, 7, 8, 9])
    return g


def _gen_hollow_green(rng: random.Random) -> np.ndarray:
    h = rng.randint(12, 20)
    w = rng.randint(12, 20)
    g = np.zeros((h, w), dtype=int)
    for _ in range(rng.randint(1, 3)):
        r0 = rng.randint(1, h - 6)
        c0 = rng.randint(1, w - 6)
        hh = rng.randint(4, min(7, h - r0))
        ww = rng.randint(4, min(7, w - c0))
        g[r0, c0 : c0 + ww] = 3
        g[r0 + hh - 1, c0 : c0 + ww] = 3
        g[r0 : r0 + hh, c0] = 3
        g[r0 : r0 + hh, c0 + ww - 1] = 3
    return g


def _gen_for(round_id: int, puzzle_id: int, rng: random.Random) -> np.ndarray:
    key = (round_id, puzzle_id)
    if key in {(51, 2), (52, 3), (52, 4), (54, 8), (57, 6)}:
        return _gen_grid_lines(rng)
    if key in {(51, 1), (51, 3), (51, 5), (52, 2), (55, 1), (55, 6), (56, 1), (56, 7), (56, 8)}:
        return _gen_frames(rng)
    if key in {(54, 1), (55, 5), (57, 1)}:
        return _gen_bars(rng)
    if key in {(55, 10), (56, 4), (57, 3)}:
        return _gen_hollow_green(rng)
    if key in {(51, 6), (51, 8), (52, 1), (53, 11)}:
        return _gen_regions(rng)
    return _gen_sparse(rng)


def generate_for(round_id: int, puzzle_id: int) -> np.ndarray:
    rng = _rng()
    return _finish(_gen_for(round_id, puzzle_id, rng))
