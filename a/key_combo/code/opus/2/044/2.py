import numpy as np
from collections import deque

BG = 0
CORNER = 4


def _components(mask: np.ndarray, full: np.ndarray):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not vis[r, c]:
                color = int(full[r, c])
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
                comps.append((color, np.asarray(pts, dtype=np.int32)))
    return comps


def _rectangles_from_yellow(grid: np.ndarray):
    h, w = grid.shape
    ys = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == CORNER]
    ys_set = set(ys)
    rows = sorted({r for r, _ in ys_set})
    cols = sorted({c for _, c in ys_set})

    candidates = []
    used = set()
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                    if not corners.issubset(ys_set):
                        continue
                    # Ensure each corner is unique and rectangle area is clean.
                    count = 0
                    for rr, cc in corners:
                        if (rr, cc) in used:
                            count = 1
                            break
                    if count:
                        continue

                    valid = True
                    for rr in range(r0, r1 + 1):
                        for cc in range(c0, c1 + 1):
                            if grid[rr, cc] == CORNER and (rr, cc) not in corners:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        candidates.append((r0, c0, r1, c1))
                        used.update(corners)
    candidates.sort()
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape
    out = g.copy()

    rects = _rectangles_from_yellow(g)

    # Precompute interior mask per rectangle for sprite filtering.
    rect_masks = []
    for r0, c0, r1, c1 in rects:
        inside = np.zeros_like(g, dtype=bool)
        inside[r0 : r1 + 1, c0 : c1 + 1] = True
        rect_masks.append(inside)

    all_rect_mask = np.zeros_like(g, dtype=bool)
    for m in rect_masks:
        all_rect_mask |= m

    # Count exterior sprites per label color (by connected components).
    for (r0, c0, r1, c1), inside_rect in zip(rects, rect_masks):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        interior_coords = []
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                v = int(g[rr, cc])
                if v != BG and v != CORNER:
                    interior_coords.append((rr, cc, v))
        if not interior_coords:
            continue
        _, label_color = interior_coords[0][:2], interior_coords[0][2]

        mask_color = g == label_color
        comps = _components(mask_color & (~all_rect_mask), g)
        k = len(comps)

        if k <= 0:
            k = 1

        inner_h = r1 - r0 - 1
        for rr in range(r0 + 1, r1):
            local_r = rr - (r0 + 1)
            if local_r % (k + 1) == 0:
                for cc in range(c0 + 1, c1):
                    if g[rr, cc] == CORNER:
                        continue
                    out[rr, cc] = label_color

    # Erase all non-rectangle sprites of each label color.
    for r in range(h):
        for c in range(w):
            if out[r, c] != BG and (not all_rect_mask[r, c]):
                out[r, c] = BG

    return out


def _inside_rect(point: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> bool:
    r, c = point
    for r0, c0, r1, c1 in rects:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _new_sprite(mask, rng: np.random.Generator, size: int, max_r: int, max_c: int):
    for _ in range(5000):
        cells = set()
        r0 = int(rng.integers(1, max_r - 1))
        c0 = int(rng.integers(1, max_c - 1))
        cells.add((r0, c0))
        frontier = [(r0, c0)]
        while len(cells) < size and frontier:
            rr, cc = frontier[int(rng.integers(0, len(frontier)))]
            dr, dc = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < max_r and 0 <= nc < max_c and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
        if len(cells) == size:
            return cells
    # fallback
    return {(0, 0), (0, 1)}


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = 15
    w = 15
    grid = np.zeros((h, w), dtype=np.int64)

    colors = [1, 2, 3, 7]
    rng.shuffle(colors)

    used = np.zeros_like(grid, dtype=bool)
    rects = []

    n_rects = int(rng.integers(2, 4))
    for i in range(n_rects):
        color = colors[i % len(colors)]
        for _ in range(4000):
            r0 = int(rng.integers(1, h - 9))
            r1 = int(rng.integers(r0 + 7, h - 2))
            c0 = int(rng.integers(0, w - 8))
            c1 = int(rng.integers(c0 + 4, min(w - 1, c0 + 8)))
            if c1 >= w - 1:
                continue
            if r1 >= h - 1:
                continue

            # spacing check
            ok = True
            for ar0, ac0, ar1, ac1 in rects:
                if not (
                    r1 < ar0 - 1
                    or r0 > ar1 + 1
                    or c1 < ac0 - 1
                    or c0 > ac1 + 1
                ):
                    ok = False
                    break
            if not ok:
                continue

            if used[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            rects.append((r0, c0, r1, c1))
            used[r0 : r1 + 1, c0 : c1 + 1] = True
            grid[r0, c0] = CORNER
            grid[r0, c1] = CORNER
            grid[r1, c0] = CORNER
            grid[r1, c1] = CORNER

            rr = int(rng.integers(r0 + 1, r1))
            cc = int(rng.integers(c0 + 1, c1))
            while grid[rr, cc] != BG:
                rr = int(rng.integers(r0 + 1, r1))
                cc = int(rng.integers(c0 + 1, c1))
            grid[rr, cc] = color
            break

    for i, color in enumerate(colors[:n_rects]):
        k = int(rng.integers(1, 5))
        for _ in range(k):
            size = int(rng.integers(2, 5))
            comp = _new_sprite(grid, rng, size=size, max_r=h, max_c=w)
            if not comp:
                continue
            good = True
            for rr, cc in comp:
                if used[rr, cc]:
                    good = False
                    break
                for r0, c0, r1, c1 in rects:
                    if r0 <= rr <= r1 and c0 <= cc <= c1:
                        good = False
                        break
                if not good:
                    break

            if not good or len(comp) != size:
                continue

            for rr, cc in comp:
                grid[rr, cc] = color

            # Avoid degenerate single-pixel or full-rectangle sprites.
            if size >= 2:
                vals = np.asarray(sorted(comp), dtype=np.int32)
                if len(vals) >= 2:
                    used[vals[:, 0], vals[:, 1]] = True

    # Ensure each label color has >=1 external sprite by placing one explicit filler if missing.
    for i, color in enumerate(colors[:n_rects]):
        present = 0
        for rr in range(h):
            for cc in range(w):
                if grid[rr, cc] == color and not _inside_rect((rr, cc), rects):
                    if (rr, cc) not in rects:
                        present += 1
        if present == 0:
            rr = int(rng.integers(1, h - 1))
            cc = int(rng.integers(0, w))
            while _inside_rect((rr, cc), rects):
                rr = int(rng.integers(1, h - 1))
                cc = int(rng.integers(0, w))
            grid[rr, cc] = color

    return grid
