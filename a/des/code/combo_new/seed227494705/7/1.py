import random
from collections import deque

import numpy as np


BG = 0


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _rect_border_cells(r0, c0, r1, c1):
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _find_frames(g: np.ndarray):
    frames = []
    for color in [int(x) for x in np.unique(g) if int(x) != BG]:
        for pts in _components(g == color):
            if len(pts) < 8:
                continue
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = set(_rect_border_cells(r0, c0, r1, c1))
            if set(pts) != border:
                continue
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_colors = [int(x) for x in np.unique(inner) if int(x) != BG]
            if len(marker_colors) == 1 and int(np.count_nonzero(inner == marker_colors[0])) == 1:
                frames.append((color, r0, c0, r1, c1, marker_colors[0]))
    return frames


def _legend_patterns(g: np.ndarray, marker_colors: set[int], frame_cells: set[tuple[int, int]]):
    h, w = g.shape
    result = {}
    for marker in marker_colors:
        best = None
        for c in range(w):
            rows = [r for r in range(h) if int(g[r, c]) == marker and (r, c) not in frame_cells]
            if not rows:
                continue
            rows.sort()
            runs = []
            start = prev = rows[0]
            for r in rows[1:]:
                if r == prev + 1:
                    prev = r
                else:
                    runs.append((start, prev))
                    start = prev = r
            runs.append((start, prev))
            for r0, r1 in runs:
                size = r1 - r0 + 1
                if size not in (2, 3) or c + size >= w:
                    continue
                if not np.all(g[r0 : r1 + 1, c] == marker):
                    continue
                patch = g[r0 : r1 + 1, c + 1 : c + 1 + size]
                if patch.shape == (size, size) and np.any(patch != BG):
                    best = (patch != BG)
                    break
            if best is not None:
                break
        if best is not None:
            result[marker] = best
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed room by looking up the marker color in a pattern legend.

    Input structure:
    - Black background with several one-pixel rectangular color frames.
    - Each frame interior contains one non-frame marker pixel.
    - A side legend has vertical marker-color columns beside small binary pattern blocks.

    Rule:
    - Match a frame to the legend entry whose marker column has the same color as its interior marker.
    - Tile that entry's binary pattern across the whole frame interior.
    - Recolor every non-black pattern cell to the frame color and leave frames and the legend unchanged.

    Key insight:
    - The marker color is only a lookup key; the visible legend pattern supplies the mask and the frame color supplies the paint.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    frame_cells = set()
    for _color, r0, c0, r1, c1, _marker in frames:
        frame_cells.update(_rect_border_cells(r0, c0, r1, c1))
        frame_cells.update((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
    patterns = _legend_patterns(g, {f[5] for f in frames}, frame_cells)

    for color, r0, c0, r1, c1, marker in frames:
        pat = patterns[marker]
        ph, pw = pat.shape
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r, c] = color if pat[(r - r0 - 1) % ph, (c - c0 - 1) % pw] else BG
    return out


def _place_rect(existing, r0, c0, r1, c1, pad=1):
    for r in range(r0 - pad, r1 + pad + 1):
        for c in range(c0 - pad, c1 + pad + 1):
            if (r, c) in existing:
                return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            existing.add((r, c))
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = list(range(1, 10))
    for _ in range(500):
        size = rng.choice([2, 3])
        n = rng.randint(2, 4)
        if 2 * n > len(colors):
            continue
        chosen = rng.sample(colors, 2 * n)
        frame_colors = chosen[:n]
        marker_colors = chosen[n:]
        legend_w = n * (size + 2) - 1
        h = rng.randint(17, 30)
        w = rng.randint(max(legend_w + 2, 18), 30)
        top_h = h - size - 1
        if top_h < 10:
            continue
        g = np.zeros((h, w), dtype=int)
        occupied = set()
        frames = []
        ok = True
        for fc, mc in zip(frame_colors, marker_colors):
            placed = False
            for _try in range(100):
                rh = rng.randint(4, 7)
                rw = rng.randint(4, 8)
                r0 = rng.randint(1, top_h - rh - 1)
                c0 = rng.randint(1, w - rw - 1)
                r1, c1 = r0 + rh - 1, c0 + rw - 1
                box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                if any((r, c) in occupied for r, c in box):
                    continue
                _place_rect(occupied, r0, c0, r1, c1, pad=1)
                for r, c in _rect_border_cells(r0, c0, r1, c1):
                    g[r, c] = fc
                mr = rng.randint(r0 + 1, r1 - 1)
                mc_col = rng.randint(c0 + 1, c1 - 1)
                g[mr, mc_col] = mc
                frames.append((fc, mc))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        row0 = h - size
        start = rng.randint(0, w - legend_w)
        for i, (fc, mc) in enumerate(frames):
            c = start + i * (size + 2)
            g[row0 : row0 + size, c] = mc
            mask = np.array([[((r + c2 + rng.randint(0, 1)) % 2) == 0 for c2 in range(size)] for r in range(size)])
            if not np.any(mask):
                mask[0, 0] = True
            if np.all(mask):
                mask[-1, -1] = False
            g[row0 : row0 + size, c + 1 : c + 1 + size][mask] = fc
        return g
    raise RuntimeError("generate_input failed")
