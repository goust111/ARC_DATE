import numpy as np
from collections import deque

BG = 0
GRAY = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not vis[r, c]:
                q = deque([(r, c)])
                vis[r, c] = True
                pts = []
                while q:
                    rr, cc = q.popleft()
                    pts.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                comps.append(np.asarray(pts, dtype=np.int32))
    return comps


def _bbox(pts: np.ndarray):
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _row_symmetric(row: np.ndarray) -> bool:
    return np.array_equal(row, row[::-1])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)

    # Control mask is the gray object.
    gray_components = _components(g == GRAY)
    if not gray_components:
        return g.copy()

    # Choose the largest gray component as control sprite.
    gray = max(gray_components, key=lambda p: len(p))
    r0, c0, r1, c1 = _bbox(gray)
    control = g[r0 : r1 + 1, c0 : c1 + 1] == GRAY
    H, W = control.shape

    # Tile: non-gray object, full square of size 2 or 3.
    tile_comp = None
    tile_bbox = None
    for comp in _components((g != BG) & (g != GRAY)):
        tr0, tc0, tr1, tc1 = _bbox(comp)
        h = tr1 - tr0 + 1
        w = tc1 - tc0 + 1
        if h == w and h in (2, 3) and comp.shape[0] == h * w:
            tile_comp = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
            tile_bbox = (tr0, tc0, h)
            break

    if tile_comp is None or tile_bbox is None:
        return g.copy()

    tr0, tc0, tile_h = tile_bbox
    tile = tile_comp.copy()
    tile_mirror = np.fliplr(tile)

    row_sym = [_row_symmetric(control[r]) for r in range(H)]
    out = np.zeros((H * tile_h, W * tile_h), dtype=np.int64)

    for r in range(H):
        for c in range(W):
            if not control[r, c]:
                continue
            rr0 = r * tile_h
            cc0 = c * tile_h
            if row_sym[r]:
                out[rr0 : rr0 + tile_h, cc0 : cc0 + tile_h] = tile
            else:
                out[rr0 : rr0 + tile_h, cc0 : cc0 + tile_h] = tile_mirror

    return out


def _control_mask(rng: np.random.Generator, h: int, w: int):
    if (h, w) == (3, 4):
        base = [
            np.array([1, 1, 1, 1], dtype=np.int8),
            np.array([1, 1, 0, 0], dtype=np.int8),
            np.array([1, 1, 1, 1], dtype=np.int8),
        ]
    else:
        base = [
            np.array([1, 1, 1, 1, 1], dtype=np.int8),
            np.array([1, 1, 0, 1, 1], dtype=np.int8),
            np.array([1, 1, 0, 0, 1], dtype=np.int8),
            np.array([1, 1, 1, 1, 1], dtype=np.int8),
        ]
    # Add tiny random variation while preserving at least one symmetric and one asymmetric row.
    mask = np.array(base, dtype=np.int8)
    # ensure connectivity
    if mask.ndim != 2:
        mask = np.ones((h, w), dtype=np.int8)
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = np.ones((h, w), dtype=np.int8)

    # enforce at least one symmetric and one asymmetric row
    sym_rows = [_row_symmetric(mask[r]) for r in range(h)]
    if not any(sym_rows):
        mid = h // 2
        mask[mid, : mid + 1] = 1
        mask[mid, mid:] = mask[mid, : h - mid][::-1] if w == h else mask[mid, : w - mid][::-1].tolist() if w - mid > 0 else 1
    if all(sym_rows):
        r = int(rng.integers(0, h))
        if w >= 3:
            mask[r, 1 : w - 1] = 0
            mask[r, 0] = 1
            mask[r, w - 1] = 1
    return mask


def _no_touch_bbox(r0, c0, h, w, m0, n0, p0, q0):
    # object1: rows r0..r0+h-1 cols c0..c0+w-1
    # object2: rows m0..m0+p0-1 cols n0..n0+q0-1
    # require at least one-cell separation.
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    m1 = m0 + p0 - 1
    q1 = n0 + q0 - 1
    # expand first box by one for 4-neighbor separation
    if r1 + 1 < m0 or m1 + 1 < r0 or c1 + 1 < n0 or q1 + 1 < c0:
        return True
    return False


def _random_tile(rng: np.random.Generator, k: int) -> np.ndarray:
    while True:
        tile = np.zeros((k, k), dtype=np.int64)
        vals = rng.integers(1, 10, size=(k, k))
        tile[:] = vals
        if not np.array_equal(tile, np.fliplr(tile)) and np.unique(tile).size >= 2:
            return tile


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 10
    g = np.zeros((h, w), dtype=np.int64)

    # place control mask
    ch = int(rng.choice([3, 4]))
    cw = 4 if ch == 3 else 5
    control = _control_mask(rng, ch, cw)

    # ensure control mask has both symmetric and asymmetric rows
    while not any(_row_symmetric(control[r]) for r in range(ch)) or all(
        _row_symmetric(control[r]) for r in range(ch)
    ):
        control = _control_mask(rng, ch, cw)

    for attempt in range(200):
        mr = int(rng.integers(0, h - ch))
        mc = int(rng.integers(0, w - cw))
        if 0 < mr < h - ch and mc < w - cw:
            break
    for r in range(ch):
        for c in range(cw):
            if control[r, c]:
                g[mr + r, mc + c] = GRAY

    # place tile non-touching control
    tile_k = int(rng.choice([2, 3]))
    tile = _random_tile(rng, tile_k)

    for _ in range(500):
        tr = int(rng.integers(0, h - tile_k + 1))
        tc = int(rng.integers(0, w - tile_k + 1))
        if _no_touch_bbox(tr, tc, tile_k, tile_k, mr, mc, ch, cw):
            for rr in range(tile_k):
                for cc in range(tile_k):
                    g[tr + rr, tc + cc] = int(tile[rr, cc])
            break

    return g
