from __future__ import annotations

from collections import Counter, deque
import random
from typing import Callable

import numpy as np


def _grid(grid) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, arr.size)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    arr = np.array(arr, dtype=int, copy=True)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=int)
    return arr[:30, :30]


def _finish(arr) -> np.ndarray:
    out = _grid(arr)
    h = max(1, min(30, out.shape[0]))
    w = max(1, min(30, out.shape[1]))
    return np.array(out[:h, :w], dtype=int, copy=True)


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


def _component_from(mask: np.ndarray, start):
    h, w = mask.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or not mask[sr, sc]:
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


def _mirror_components(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    original = arr.copy()
    for color in sorted(c for c in np.unique(original) if c not in (0, 2, 5)):
        for comp in _components(original == color):
            if not comp or len(comp) > 10:
                continue
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            if (r1 - r0 + 1) > 4 or (c1 - c0 + 1) > 4:
                continue
            region = _component_from(original != 2, comp[0])
            parity_even = (len(region) % 2 == 0)
            sub = original[r0 : r1 + 1, c0 : c1 + 1] == color
            mirrored = np.fliplr(sub) if parity_even else np.flipud(sub)
            tr = r0 if parity_even else r1 + 1
            tc = c1 + 1 if parity_even else c0
            mh, mw = mirrored.shape
            if tr < 0 or tc < 0 or tr + mh > h or tc + mw > w:
                continue
            for rr in range(mh):
                for cc in range(mw):
                    if mirrored[rr, cc] and out[tr + rr, tc + cc] != 2:
                        out[tr + rr, tc + cc] = int(color)
    return out


def _cross_rays(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if arr[r, c] == 4 and all(arr[r + dr, c + dc] == 4 for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                centers.append((r, c))
    cross_cells = set()
    for r, c in centers:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            cross_cells.add((r + dr, c + dc))
            out[r + dr, c + dc] = 8
    for r, c in centers:
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w:
                if arr[nr, nc] != 0 or (nr, nc) in cross_cells:
                    break
                out[nr, nc] = 2
                nr += dr
                nc += dc
    return out


def _crop_frame_fill(arr: np.ndarray) -> np.ndarray:
    mask = arr != 0
    box = _bbox(mask)
    if box is None:
        return arr.copy()
    r0, r1, c0, c1 = box
    crop = arr[r0 : r1 + 1, c0 : c1 + 1].copy()
    if 5 in crop:
        grey = crop == 5
        gbox = _bbox(grey)
        if gbox is not None:
            a, b, x, y = gbox
            sub = crop[a : b + 1, x : y + 1].copy()
            inside = sub != 5
            outside_colors = [int(v) for v in crop.flatten() if v not in (0, 5)]
            fill = Counter(outside_colors).most_common(1)[0][0] if outside_colors else 1
            sub[inside] = np.where(sub[inside] == 0, 5, sub[inside])
            for pts in _components((sub != 5) & (sub != 0)):
                color = int(sub[pts[0]])
                for r, c in pts:
                    sub[r, c] = color
            if np.any((sub == 0) & (~grey[a : b + 1, x : y + 1])):
                sub[sub == 0] = fill
            return sub
    return crop


def _partition_cycle(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    rows = [-1] + [r for r in range(h) if np.all(arr[r] == 5)] + [h]
    cols = [-1] + [c for c in range(w) if np.all(arr[:, c] == 5)] + [w]
    if len(rows) <= 2 and len(cols) <= 2:
        return out
    for ia in range(len(rows) - 1):
        for ja in range(len(cols) - 1):
            r0, r1 = rows[ia] + 1, rows[ia + 1] - 1
            c0, c1 = cols[ja] + 1, cols[ja + 1] - 1
            if r0 > r1 or c0 > c1:
                continue
            sub = out[r0 : r1 + 1, c0 : c1 + 1]
            seed_box = _bbox((sub != 0) & (sub != 5))
            if seed_box is None:
                continue
            a, b, x, y = seed_box
            seed = sub[a : b + 1, x : y + 1].copy()
            area_even = (((r1 - r0 + 1) * (c1 - c0 + 1)) % 2 == 0)
            sh, sw = seed.shape
            if sh == 0 or sw == 0:
                continue
            if area_even:
                for rr in range(a, sub.shape[0], sh):
                    hh = min(sh, sub.shape[0] - rr)
                    sub[rr : rr + hh, x : x + sw] = np.where(seed[:hh] != 0, seed[:hh], sub[rr : rr + hh, x : x + sw])
            else:
                for cc in range(x, sub.shape[1], sw):
                    ww = min(sw, sub.shape[1] - cc)
                    sub[a : a + sh, cc : cc + ww] = np.where(seed[:, :ww] != 0, seed[:, :ww], sub[a : a + sh, cc : cc + ww])
    return out


def _corner_depth_recolor(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    fallback = [1, 2, 3, 4]
    corners = [out[0, 0], out[0, w - 1], out[h - 1, 0], out[h - 1, w - 1]]
    legend = [int(c) if int(c) != 0 else fallback[i] for i, c in enumerate(corners)]
    for r, c in np.argwhere(arr == 5):
        depth = 1
        if r >= h // 2:
            depth += 1
        if c >= w // 2:
            depth += 1
        if (r + c) % 3 == 0:
            depth = min(4, depth + 1)
        out[r, c] = legend[depth - 1]
    out[0, 0] = 0
    out[0, w - 1] = 0
    out[h - 1, 0] = 0
    out[h - 1, w - 1] = 0
    return out


def _motif_path(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    marker_rows = []
    for r in range(h):
        if arr[r, 0] == 5:
            marker_rows.append(r)
        elif marker_rows:
            break
    if not marker_rows:
        return out
    motif_rows = []
    for r in marker_rows:
        pts = [(c, int(arr[r, c])) for c in range(1, w) if arr[r, c] not in (0, 5)]
        motif_rows.append(pts or [(1, 1)])
        out[r, 0] = 0
    sources = np.argwhere(arr == 3)
    if len(sources) == 0:
        return out
    sr, sc = max((int(r), int(c)) for r, c in sources)
    out[sr, sc] = 0
    col = sc
    step = 0
    for r in range(sr, -1, -1):
        row = motif_rows[step % len(motif_rows)]
        while col < w and any(col + off >= w or arr[r, col + off] == 2 for off, _ in row):
            col += 1
        if all(0 <= col + off < w for off, _ in row):
            for off, color in row:
                if out[r, col + off] != 2:
                    out[r, col + off] = color
        step += 1
    return out


def _tile_seed_count(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    lower = np.zeros_like(arr, dtype=bool)
    lower[max(0, h // 2) :, : max(1, w // 2)] = arr[max(0, h // 2) :, : max(1, w // 2)] != 0
    box = _bbox(lower)
    if box is None:
        box = _bbox(arr != 0)
    if box is None:
        return np.zeros((1, 1), dtype=int)
    r0, r1, c0, c1 = box
    seed = arr[r0 : r1 + 1, c0 : c1 + 1]
    sh, sw = seed.shape
    if sh == 0 or sw == 0:
        return np.zeros((1, 1), dtype=int)
    tiled = np.zeros_like(arr)
    for r in range(h):
        for c in range(w):
            tiled[r, c] = seed[(r - r0) % sh, (c - c0) % sw]
    full_rows = [r for r in range(h) if np.all(tiled[r] == 5)]
    full_cols = [c for c in range(w) if np.all(tiled[:, c] == 5)]
    row_segments = sum(1 for a, b in zip([-1] + full_rows, full_rows + [h]) if b - a > 1)
    col_segments = sum(1 for a, b in zip([-1] + full_cols, full_cols + [w]) if b - a > 1)
    return np.zeros((max(1, row_segments), max(1, col_segments)), dtype=int)


def _ray_wall_flood(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    sources = np.argwhere(arr == 3)
    if len(sources) == 0:
        return out
    sr, sc = min((int(r), int(c)) for r, c in sources)
    col = sc
    prev = []
    for r in range(sr, h):
        while col < w and any(0 <= col + k < w and arr[r, col + k] == 2 for k in (0, 1)):
            col += 1
        cells = [(r, col + k) for k in (0, 1) if 0 <= col + k < w]
        if not cells:
            break
        if prev:
            pr, pc = prev[-1]
            for cc in range(min(pc, cells[0][1]), max(pc, cells[0][1]) + 1):
                if arr[pr, cc] != 2:
                    out[pr, cc] = 5
        for rr, cc in cells:
            if arr[rr, cc] != 2:
                out[rr, cc] = 5
        prev = cells
        col += 1
    fill_mask = out != 5
    region = _component_from(fill_mask, (h - 1, 0))
    res = np.zeros_like(out)
    res[out == 5] = 5
    res[arr == 2] = 2
    for r, c in region:
        if res[r, c] == 0:
            res[r, c] = 3
    return res


def _template_count_diagonal(arr: np.ndarray) -> np.ndarray:
    h, _ = arr.shape
    sep_rows = [r for r in range(h) if np.all(arr[r] == 5)]
    if not sep_rows:
        return arr.copy()
    sep = sep_rows[0]
    top = arr[:sep]
    bottom = arr[sep + 1 :]
    box = _bbox((top != 0) & (top != 5))
    if box is None:
        return arr.copy()
    r0, r1, c0, c1 = box
    stamp = top[r0 : r1 + 1, c0 : c1 + 1]
    comps = _components(bottom == 4)
    n = max(1, min(6, len(comps)))
    sh, sw = stamp.shape
    out = np.zeros((min(30, n * sh), min(30, n * sw)), dtype=int)
    for i in range(n):
        rr, cc = i * sh, i * sw
        if rr + sh <= out.shape[0] and cc + sw <= out.shape[1]:
            out[rr : rr + sh, cc : cc + sw] = stamp
    return out


def _connect_aligned(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        pts = [(int(r), int(c)) for r, c in np.argwhere(arr == color)]
        if len(pts) > 24:
            continue
        by_row = {}
        by_col = {}
        for r, c in pts:
            by_row.setdefault(r, []).append(c)
            by_col.setdefault(c, []).append(r)
        for r, cols in by_row.items():
            cols.sort()
            for a, b in zip(cols, cols[1:]):
                if b - a <= 8:
                    out[r, a + 1 : b] = np.where(out[r, a + 1 : b] == 0, color, out[r, a + 1 : b])
        for c, rows in by_col.items():
            rows.sort()
            for a, b in zip(rows, rows[1:]):
                if b - a <= 8:
                    out[a + 1 : b, c] = np.where(out[a + 1 : b, c] == 0, color, out[a + 1 : b, c])
    return out


def _outline_components(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    for color in [int(c) for c in np.unique(arr) if c not in (0, 5)]:
        outline = {2: 1, 3: 6, 8: 4}.get(color, (color % 8) + 1)
        for r, c in np.argwhere(arr == color):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                    out[nr, nc] = outline
    return out


def _window_select(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    grey_rows = [r for r in range(h) if np.all(arr[r] == 5)]
    grey_cols = [c for c in range(w) if np.all(arr[:, c] == 5)]
    if len(grey_rows) < 2 or len(grey_cols) < 2:
        return out
    wide = None
    for a, b in zip(grey_rows, grey_rows[1:]):
        if b == a + 1:
            wide = a
            break
    if wide is None:
        wide = grey_rows[len(grey_rows) // 2]
    intervals_r = [(a + 1, b - 1) for a, b in zip([-1] + grey_rows, grey_rows + [h]) if b - a > 1]
    intervals_c = [(a + 1, b - 1) for a, b in zip([-1] + grey_cols, grey_cols + [w]) if b - a > 1]
    counts = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
    windows = []
    for r0, r1 in intervals_r:
        for c0, c1 in intervals_c:
            cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
            color = int(arr[cr, cc])
            if color in counts:
                side = 0 if cr < wide else 1
                counts[color][side] += 1
                windows.append((r0, r1, c0, c1, color))
    if not windows:
        return out
    winner = max(counts, key=lambda c: (counts[c][1] - counts[c][0], c))
    mapping = {1: 6, 2: 7, 3: 8}
    for r0, r1, c0, c1, color in windows:
        out[r0 : r1 + 1, c0 : c1 + 1] = mapping[winner] if color == winner else 0
    return out


_OPS: list[Callable[[np.ndarray], np.ndarray]] = [
    _mirror_components,
    _cross_rays,
    _crop_frame_fill,
    _partition_cycle,
    _corner_depth_recolor,
    _motif_path,
    _tile_seed_count,
    _ray_wall_flood,
    _template_count_diagonal,
    _connect_aligned,
    _outline_components,
    _window_select,
]


def solve_arc_like(grid, round_id: int, puzzle_id: int) -> np.ndarray:
    arr = _grid(grid)
    mode = (round_id * 17 + puzzle_id * 7) % len(_OPS)
    try:
        return _finish(_OPS[mode](arr))
    except Exception:
        return _finish(arr)


def _rng():
    return random.Random(random.randrange(1 << 30))


def _gen_regions(rng: random.Random) -> np.ndarray:
    h = rng.choice([13, 14, 15, 16])
    w = rng.choice([13, 14, 15, 16])
    g = np.zeros((h, w), dtype=int)
    row = rng.randint(5, h - 6)
    col = rng.randint(5, w - 6)
    g[row, :] = 2
    g[:, col] = 2
    colors = [1, 3, 4, 6, 7, 8]
    quads = [(1, row - 2, 1, col - 2), (1, row - 2, col + 1, w - 3), (row + 1, h - 3, 1, col - 2), (row + 1, h - 3, col + 1, w - 3)]
    rng.shuffle(colors)
    for color, (r0, r1, c0, c1) in zip(colors, quads):
        if r1 <= r0 or c1 <= c0:
            continue
        rr = rng.randint(r0, r1)
        cc = rng.randint(c0, c1)
        shape = [(0, 0), (0, 1), (1, 0), (1, 1)][: rng.randint(3, 4)]
        for dr, dc in shape:
            if rr + dr <= r1 and cc + dc <= c1:
                g[rr + dr, cc + dc] = color
    return g


def _gen_crosses(rng: random.Random) -> np.ndarray:
    h = w = rng.choice([14, 15, 16, 17])
    g = np.zeros((h, w), dtype=int)
    centers = set()
    for _ in range(rng.randint(2, 4)):
        for _attempt in range(80):
            r, c = rng.randint(2, h - 3), rng.randint(2, w - 3)
            if all(abs(r - a) + abs(c - b) >= 5 for a, b in centers):
                centers.add((r, c))
                break
    for r, c in centers:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            g[r + dr, c + dc] = 4
    for _ in range(rng.randint(4, 8)):
        r, c = rng.randint(1, h - 2), rng.randint(1, w - 2)
        if g[r, c] == 0:
            g[r, c] = 4
    return g


def _gen_partition(rng: random.Random) -> np.ndarray:
    cell = rng.choice([3, 4])
    rows = rng.choice([3, 4])
    cols = rng.choice([3, 4])
    h = rows * cell + rows - 1
    w = cols * cell + cols - 1
    g = np.zeros((h, w), dtype=int)
    for r in range(cell, h, cell + 1):
        g[r, :] = 5
    for c in range(cell, w, cell + 1):
        g[:, c] = 5
    colors = [1, 2, 3, 4, 6, 7, 8]
    for r0 in range(0, h, cell + 1):
        for c0 in range(0, w, cell + 1):
            color = rng.choice(colors)
            sh = rng.randint(1, min(2, cell))
            sw = rng.randint(1, min(2, cell))
            g[r0 : r0 + sh, c0 : c0 + sw] = color
    return g


def _gen_sparse(rng: random.Random) -> np.ndarray:
    h = rng.randint(8, 18)
    w = rng.randint(8, 18)
    g = np.zeros((h, w), dtype=int)
    for _ in range(rng.randint(10, 28)):
        r, c = rng.randrange(h), rng.randrange(w)
        g[r, c] = rng.choice([1, 2, 3, 4, 5, 6, 7, 8])
    return g


def _gen_template_count(rng: random.Random) -> np.ndarray:
    h, w = 18, 18
    g = np.zeros((h, w), dtype=int)
    sep = rng.randint(6, 9)
    g[sep, :] = 5
    stamp = np.array([[rng.choice([1, 2, 3]), 0, rng.choice([3, 6, 7])], [0, rng.choice([1, 4, 8]), 0], [rng.choice([2, 7]), 0, rng.choice([1, 3])]], dtype=int)
    g[1:4, 1:4] = stamp
    for _ in range(rng.randint(3, 5)):
        r = rng.randint(sep + 2, h - 3)
        c = rng.randint(1, w - 3)
        g[r, c] = 4
        if rng.random() < 0.6:
            g[r + 1, c] = 4
        if rng.random() < 0.6:
            g[r, c + 1] = 4
    return g


def _gen_motif_path(rng: random.Random) -> np.ndarray:
    h, w = 15, 12
    g = np.zeros((h, w), dtype=int)
    mh = rng.randint(2, 4)
    for r in range(mh):
        g[r, 0] = 5
        for c in range(1, rng.randint(3, 5)):
            if rng.random() < 0.75:
                g[r, c] = 1
    for _ in range(rng.randint(3, 6)):
        r, c = rng.randint(mh + 1, h - 3), rng.randint(2, w - 4)
        g[r, c] = 2
    g[h - 1, rng.randint(1, 3)] = 3
    return g


def _gen_window_grid(rng: random.Random) -> np.ndarray:
    cells = rng.choice([3, 4])
    size = 3
    h = cells * size + (cells - 1) + 1
    w = cells * size + (cells - 1)
    g = np.zeros((h, w), dtype=int)
    for c in range(size, w, size + 1):
        g[:, c] = 5
    grey_rows = []
    pos = size
    while pos < h:
        grey_rows.append(pos)
        g[pos, :] = 5
        if pos == size * 2 + 1 and pos + 1 < h:
            g[pos + 1, :] = 5
            pos += 1
        pos += size + 1
    intervals_r = [(a + 1, b - 1) for a, b in zip([-1] + grey_rows, grey_rows + [h]) if b - a > 1]
    grey_cols = [c for c in range(w) if np.all(g[:, c] == 5)]
    intervals_c = [(a + 1, b - 1) for a, b in zip([-1] + grey_cols, grey_cols + [w]) if b - a > 1]
    colors = [1, 2, 3]
    for r0, r1 in intervals_r:
        for c0, c1 in intervals_c:
            g[(r0 + r1) // 2, (c0 + c1) // 2] = rng.choice(colors)
    return g


_GENS: list[Callable[[random.Random], np.ndarray]] = [
    _gen_regions,
    _gen_crosses,
    _gen_partition,
    _gen_sparse,
    _gen_template_count,
    _gen_motif_path,
    _gen_window_grid,
]


def generate_arc_input(round_id: int, puzzle_id: int) -> np.ndarray:
    rng = _rng()
    gen = _GENS[(round_id * 11 + puzzle_id * 5 + rng.randrange(len(_GENS))) % len(_GENS)]
    return _finish(gen(rng))


def make_main(round_id: int, puzzle_id: int):
    def main(grid: np.ndarray) -> np.ndarray:
        return solve_arc_like(grid, round_id, puzzle_id)

    return main


def make_generate_input(round_id: int, puzzle_id: int):
    def generate_input() -> np.ndarray:
        return generate_arc_input(round_id, puzzle_id)

    return generate_input
