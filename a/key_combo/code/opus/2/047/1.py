import numpy as np
from collections import deque


def _trace_path(h: int, w: int, start_r: int, start_c: int):
    path = []
    r, c = start_r, start_c
    dc = 1
    while True:
        path.append((r, c))
        nr = r - 1
        if nr < 0:
            break
        nc = c + dc
        if nc < 0 or nc >= w:
            dc *= -1
            nc = c + dc
        r, c = nr, nc
    return path


def _bbox_from_coords(coords):
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_hollow_rectangle(coords: np.ndarray):
    if len(coords) == 0:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0 + 1, r1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))
    coord_set = {(int(r), int(c)) for r, c in coords}
    if not coord_set.issubset(perimeter):
        return None
    if len(coord_set) != len(perimeter):
        return None
    return (r0, c0, r1, c1, coords)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy().astype(np.int64)
    h, w = out.shape

    blue_coords = np.argwhere(out == 1)
    if len(blue_coords) != 1:
        return out

    start_r, start_c = map(int, blue_coords[0])
    path = _trace_path(h, w, start_r, start_c)
    path_set = set(path)

    gray_mask = out == 5
    for comp in _components(gray_mask):
        rect = _is_hollow_rectangle(comp)
        if rect is None:
            continue
        r0, c0, r1, c1, comp_coords = rect
        selected = any((int(r), int(c)) in path_set for r, c in comp_coords)
        if selected:
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr, cc] = 4

    for rr, cc in path:
        if out[rr, cc] != 5:
            out[rr, cc] = 1
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 12

    for _ in range(400):
        grid = np.zeros((h, w), dtype=np.int64)
        forbid = np.zeros((h, w), dtype=bool)
        rects = []

        for _ in range(rng.integers(3, 6)):
            for _ in range(200):
                rw = int(rng.integers(3, 7))
                rw = min(rw, h - 2, w - 2)
                rh = int(rng.integers(3, 7))
                rh = min(rh, h - 2, w - 2)
                if rw < 3 or rh < 3:
                    continue
                r0 = int(rng.integers(1, h - rh))
                c0 = int(rng.integers(1, w - rw))
                coords = []
                for c in range(c0, c0 + rw):
                    coords.append((r0, c))
                    coords.append((r0 + rh - 1, c))
                for r in range(r0 + 1, r0 + rh - 1):
                    coords.append((r, c0))
                    coords.append((r, c0 + rw - 1))
                if any(forbid[r, c] for r, c in coords):
                    continue
                for r, c in coords:
                    grid[r, c] = 5
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w:
                                forbid[rr, cc] = True
                rects.append((r0, c0, rw, rh))
                break

        if len(rects) < 3:
            continue

        black_cells = list(zip(*np.where(grid == 0)))
        rng.shuffle(black_cells)
        for sr, sc in black_cells:
            path = _trace_path(h, w, sr, sc)
            path_set = set(path)
            touched = 0
            for r0, c0, rw, rh in rects:
                found = False
                for r in range(r0, r0 + rh):
                    if (r, c0) in path_set or (r, c0 + rw - 1) in path_set:
                        found = True
                        break
                if not found:
                    for c in range(c0, c0 + rw):
                        if (r0, c) in path_set or (r0 + rh - 1, c) in path_set:
                            found = True
                            break
                if found:
                    touched += 1
            if 0 < touched < len(rects):
                grid[sr, sc] = 1
                return grid

    # Fallback: keep minimal valid frame and center blue
    grid[:, :] = 0
    grid[1:12, 1] = 5
    grid[1, 1:11] = 5
    grid[10, 1:11] = 5
    grid[1:11, 10] = 5
    grid[3, 6] = 1
    return grid

