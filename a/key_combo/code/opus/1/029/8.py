import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _template_from(grid, cells):
    r0, c0, r1, c1 = _bbox(cells)
    pix = {}
    for r, c in cells:
        pix[(r - r0, c - c0)] = int(grid[r, c])
    return {"pixels": pix, "area": len(pix), "shape": (r1 - r0 + 1, c1 - c0 + 1)}


def _work_components(work):
    return _components(work != 0)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    gray_cols = np.where(np.any(g == 5, axis=0))[0]
    split = int(gray_cols.max()) + 1 if len(gray_cols) else 6
    left = g[:, :split]
    work = g[:, split:]

    templates = []
    for comp in _components((left != 0) & (left != 5)):
        templates.append(_template_from(left, comp))

    placements = []
    for comp in _work_components(work):
        partial = {(r, c): int(work[r, c]) for r, c in comp}
        matches = []
        for ti, tmpl in enumerate(templates):
            by_color = {}
            for pos, color in tmpl["pixels"].items():
                by_color.setdefault(color, []).append(pos)
            first_pos, first_color = next(iter(partial.items()))
            for tr, tc in by_color.get(first_color, []):
                off = (first_pos[0] - tr, first_pos[1] - tc)
                ok = True
                for (pr, pc), color in partial.items():
                    if tmpl["pixels"].get((pr - off[0], pc - off[1])) != color:
                        ok = False
                        break
                if ok:
                    matches.append((ti, off))
        if matches:
            ti, off = matches[0]
            placements.append((templates[ti], off))

    out = np.zeros_like(work)
    placements.sort(key=lambda x: -x[0]["area"])
    for tmpl, (orow, ocol) in placements:
        for (r, c), color in tmpl["pixels"].items():
            rr, cc = orow + r, ocol + c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
    return out


_TEMPLATES = [
    np.array(
        [
            [1, 0, 2, 0, 1],
            [1, 2, 2, 0, 0],
            [0, 2, 0, 3, 3],
            [0, 0, 3, 3, 0],
            [1, 0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [4, 4, 0, 2],
            [0, 4, 2, 0],
            [3, 0, 2, 2],
            [3, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 4, 0],
            [1, 4, 4, 0],
            [0, 0, 4, 3],
        ],
        dtype=int,
    ),
]


def _draw(arr, top, left, grid, transparent_zero=True):
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c] != 0 or not transparent_zero:
                grid[top + r, left + c] = int(arr[r, c])


def _partial(arr):
    cells = [(r, c) for r in range(arr.shape[0]) for c in range(arr.shape[1]) if arr[r, c] != 0]
    random.shuffle(cells)
    keep = set(cells[: max(2, len(cells) // 2)])
    # Close under one neighbor when possible so the visible clue is connected.
    changed = True
    while changed:
        changed = False
        for r, c in list(keep):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if p in cells and random.random() < 0.25:
                    keep.add(p)
                    changed = True
    out = np.zeros_like(arr)
    for r, c in keep:
        out[r, c] = arr[r, c]
    return out


def generate_input() -> np.ndarray:
    h, w, split = 20, 20, 6
    g = np.zeros((h, w), dtype=int)
    g[:, :split] = 5
    rows = [1, 8, 14]
    for arr, top in zip(_TEMPLATES, rows):
        _draw(arr, top, 1, g)
    placements = [(4, 2), (7, 5), (6, 7)]
    order = list(range(3))
    random.shuffle(order)
    used = np.zeros((h, w - split), bool)
    for i in order:
        arr = _partial(_TEMPLATES[i])
        top, left = placements[i]
        # Keep the clue sparse and non-overlapping; the completed stamps still overlap.
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if arr[r, c] != 0 and not used[top + r, left + c]:
                    g[top + r, split + left + c] = arr[r, c]
                    used[top + r, left + c] = True
    return g
