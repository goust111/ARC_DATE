import numpy as np
from collections import deque


BG = 0
GRAY = 5


def _components(grid: np.ndarray, target_mask: np.ndarray):
    h, w = target_mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if target_mask[r, c] and not vis[r, c]:
                color = int(grid[r, c])
                q = deque([(r, c)])
                vis[r, c] = True
                pts = []
                while q:
                    rr, cc = q.popleft()
                    pts.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and target_mask[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                comps.append((color, np.asarray(pts, dtype=np.int32)))
    return comps


def _is_left_right_symmetric(obj_coords: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in obj_coords:
        sub[rr - r0, cc - c0] = True
    for row in sub:
        if not np.array_equal(row, row[::-1]):
            return False
    return True


def _draw_ray(
    out: np.ndarray,
    original: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    color: int,
) -> None:
    h, w = out.shape
    r, c = start_r + dr, start_c + dc
    d = 0
    while 0 <= r < h and 0 <= c < w:
        if original[r, c] == BG:
            out[r, c] = color if (d % 2 == 0) else GRAY
        d += 1
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape
    out = g.copy()
    mask = g != BG

    objects = _components(g, mask)
    if not objects:
        return out

    asymmetric = []
    symmetric = []

    for color, pts in objects:
        r0, c0 = pts[:, 0].min(), pts[:, 1].min()
        r1, c1 = pts[:, 0].max(), pts[:, 1].max()
        is_sym = _is_left_right_symmetric(pts, r0, c0, r1, c1)
        info = (color, pts, r0, c0, r1, c1)
        if is_sym:
            symmetric.append(info)
        else:
            asymmetric.append(info)

    # Draw rays from vertically symmetric objects first.
    for color, pts, r0, c0, r1, c1 in symmetric:
        # Start from the leftmost pixel in the bottom row.
        row_mask = pts[pts[:, 0] == r1]
        start_c = int(row_mask[:, 1].min())
        _draw_ray(out, g, r1, start_c, 1, 0, color)

    # Unique asymmetric object has priority if rays overlap.
    if asymmetric:
        color, pts, r0, c0, r1, c1 = asymmetric[0]
        col_mask = pts[pts[:, 1] == c1]
        start_r = int(col_mask[:, 0].min())
        _draw_ray(out, g, start_r, c1, 0, 1, color)

    return out


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    # Masks with cell count in [3, 6], 4-connected, bbox <= 3x3.
    shapes = [
        np.array([[1, 0, 1], [1, 1, 1]], dtype=np.int8),
        np.array([[1, 1], [1, 1]], dtype=np.int8),
        np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.int8),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int8),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int8),
    ]
    idx = int(rng.integers(0, len(shapes)))
    return shapes[idx]


def _random_asym_shape(rng: np.random.Generator) -> np.ndarray:
    shapes = [
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
        np.array([[1, 0], [1, 1]], dtype=np.int8),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=np.int8),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int8),
        np.array([[1, 1], [0, 1], [1, 0]], dtype=np.int8),
    ]
    idx = int(rng.integers(0, len(shapes)))
    return shapes[idx]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    h = rng.integers(12, 16)
    w = rng.integers(12, 16)
    grid = np.zeros((h, w), dtype=np.int64)

    colors = [1, 2, 3, 4, 7, 8, 9]
    rng.shuffle(colors)
    n_objects = int(rng.integers(4, 6))

    rects = []

    # Place asymmetric object.
    for idx in range(2):
        color = colors[idx]
        is_asym = idx == 0
        placed = False
        for _ in range(2000):
            shape = _random_asym_shape(rng) if is_asym else _random_shape(rng)
            sh, sw = shape.shape
            r0 = int(rng.integers(0, max(1, h - sh - 1)))
            c0 = int(rng.integers(0, max(1, w - sw - 1)))
            r1, c1 = r0 + sh - 1, c0 + sw - 1
            if r1 >= h - 1 or c1 >= w - 1:
                continue
            if grid[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            # keep bbox spacing >=2 with existing objects
            bad = False
            for er0, ec0, er1, ec1 in rects:
                if not (
                    r1 + 2 < er0
                    or r0 - 2 > er1
                    or c1 + 2 < ec0
                    or c0 - 2 > ec1
                ):
                    bad = True
                    break
            if bad:
                continue

            rr, cc = np.where(shape == 1)
            grid[r0 + rr, c0 + cc] = color
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            # guaranteed fallback: overwrite a tiny region near center
            color = colors.pop()
            shape = np.array([[1, 1], [1, 0]], dtype=np.int8)
            rr, cc = np.where(shape == 1)
            r0, c0 = 1 + idx, 1 + idx
            grid[r0 + rr, c0 + cc] = color
            rects.append((r0, c0, r0 + 1, c0 + 1))

    # Place remaining symmetric objects.
    for i in range(2, n_objects):
        color = colors[i]
        for _ in range(2000):
            shape = _random_shape(rng)
            sh, sw = shape.shape
            r0 = int(rng.integers(0, max(1, h - sh - 1)))
            c0 = int(rng.integers(0, max(1, w - sw - 1)))
            r1, c1 = r0 + sh - 1, c0 + sw - 1
            if r1 >= h - 1 or c1 >= w - 1:
                continue
            if grid[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            bad = False
            for er0, ec0, er1, ec1 in rects:
                if not (
                    r1 + 2 < er0
                    or r0 - 2 > er1
                    or c1 + 2 < ec0
                    or c0 - 2 > ec1
                ):
                    bad = True
                    break
            if bad:
                continue

            rr, cc = np.where(shape == 1)
            grid[r0 + rr, c0 + cc] = color
            rects.append((r0, c0, r1, c1))
            break

    return grid
