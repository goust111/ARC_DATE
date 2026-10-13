import numpy as np
from collections import deque
from itertools import permutations


def _share_coord(a, b):
    return a[0] == b[0] or a[1] == b[1]


def _draw_straight_line(arr, color, p0, p1):
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        lo, hi = (c0, c1) if c0 <= c1 else (c1, c0)
        arr[r0, lo : hi + 1] = color
        return
    if c0 == c1:
        lo, hi = (r0, r1) if r0 <= r1 else (r1, r0)
        arr[lo : hi + 1, c0] = color


def _extract_segments(grid, color):
    mask = grid == color
    visited = np.zeros_like(mask, dtype=bool)
    segments = []
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                if cr > 0 and mask[cr - 1, cc] and not visited[cr - 1, cc]:
                    visited[cr - 1, cc] = True
                    queue.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not visited[cr + 1, cc]:
                    visited[cr + 1, cc] = True
                    queue.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not visited[cr, cc - 1]:
                    visited[cr, cc - 1] = True
                    queue.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not visited[cr, cc + 1]:
                    visited[cr, cc + 1] = True
                    queue.append((cr, cc + 1))

            if len(comp) < 2:
                continue

            rows = [x for x, _ in comp]
            cols = [y for _, y in comp]
            if len(set(rows)) == 1:
                r = rows[0]
                segments.append({"ends": ((r, min(cols)), (r, max(cols)))})
            elif len(set(cols)) == 1:
                c = cols[0]
                segments.append({"ends": ((min(rows), c), (max(rows), c))})

    return segments


def _find_cycle(segments):
    if len(segments) < 3:
        return None

    n = len(segments)
    ends = np.array([seg["ends"] for seg in segments], dtype=int)

    def backtrack(pos, perm, first_seg, first_in_point, prev_out, assignment):
        if pos == n:
            if _share_coord(prev_out, first_in_point):
                return assignment
            return None

        seg = perm[pos]
        s0 = tuple(ends[seg, 0].tolist())
        s1 = tuple(ends[seg, 1].tolist())

        for in_idx, in_pt in ((0, s0), (1, s1)):
            if not _share_coord(prev_out, in_pt):
                continue
            out_idx = 1 - in_idx
            out_pt = s0 if out_idx == 0 else s1
            assignment[seg] = (in_idx, out_idx)
            out = backtrack(pos + 1, perm, first_seg, first_in_point, out_pt, assignment)
            if out is not None:
                return out
            del assignment[seg]

        return None

    for perm in permutations(range(n)):
        first_seg = perm[0]
        f0 = tuple(ends[first_seg, 0].tolist())
        f1 = tuple(ends[first_seg, 1].tolist())
        for first_in in (0, 1):
            first_in_pt = f0 if first_in == 0 else f1
            first_out_pt = f1 if first_in == 0 else f0
            assignment = {first_seg: (first_in, 1 - first_in)}
            result = backtrack(1, perm, first_seg, first_in_pt, first_out_pt, assignment)
            if result is not None:
                return [perm[i] for i in range(n)], result

    return None


def _fill_inside_color(arr, color):
    boundary = arr == color
    if not np.any(boundary):
        return

    rows = np.where(boundary)[0]
    cols = np.where(boundary)[1]
    h, w = arr.shape

    r0 = max(0, int(rows.min()) - 1)
    r1 = min(h - 1, int(rows.max()) + 1)
    c0 = max(0, int(cols.min()) - 1)
    c1 = min(w - 1, int(cols.max()) + 1)

    wall = arr != 0
    vis = np.zeros_like(arr, dtype=bool)
    q = deque()

    for r in range(r0, r1 + 1):
        for c in (c0, c1):
            if not wall[r, c] and arr[r, c] == 0 and not vis[r, c]:
                vis[r, c] = True
                q.append((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            if not wall[r, c] and arr[r, c] == 0 and not vis[r, c]:
                vis[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        if r > r0 and not wall[r - 1, c] and not vis[r - 1, c]:
            vis[r - 1, c] = True
            q.append((r - 1, c))
        if r < r1 and not wall[r + 1, c] and not vis[r + 1, c]:
            vis[r + 1, c] = True
            q.append((r + 1, c))
        if c > c0 and not wall[r, c - 1] and not vis[r, c - 1]:
            vis[r, c - 1] = True
            q.append((r, c - 1))
        if c < c1 and not wall[r, c + 1] and not vis[r, c + 1]:
            vis[r, c + 1] = True
            q.append((r, c + 1))

    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    inside_box = (rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1)
    fill_mask = (arr == 0) & (~vis) & inside_box
    arr[fill_mask] = color


def _segment_cells(seg):
    (r0, c0), (r1, c1) = seg
    if r0 == r1:
        lo, hi = (c0, c1) if c0 <= c1 else (c1, c0)
        return [(r0, c) for c in range(lo, hi + 1)]

    lo, hi = (r0, r1) if r0 <= r1 else (r1, r0)
    return [(r, c0) for r in range(lo, hi + 1)]


def _can_place(grid, cells):
    h, w = grid.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != 0:
                    return False
    return True


def _shift_segments(segments, dr, dc):
    return [((r0 + dr, c0 + dc), (r1 + dr, c1 + dc)) for (r0, c0), (r1, c1) in segments]


def _build_shape_segments(count, rng):
    if count == 3:
        top = 0
        left = 1
        right = int(rng.integers(4, 8))
        bottom = int(rng.integers(6, 9))
        return [
            ((top, left), (top, right)),
            ((top + 2, right), (top + 3, right)),
            ((bottom, left), (bottom, right)),
        ]

    if count == 4:
        top = 0
        x0 = int(rng.integers(1, 4))
        x2 = x0 + int(rng.integers(3, 7))
        return [
            ((top, x0), (top, x2)),
            ((top + 2, x2), (top + 3, x2)),
            ((top + 6, x0), (top + 6, x2)),
            ((top + 2, x0), (top + 3, x0)),
        ]

    # count == 5
    top = 0
    right = int(rng.integers(5, 10))
    depth = int(rng.integers(5, 9))
    return [
        ((top, 0), (top, right)),
        ((top + 2, right), (top + depth - 1, right)),
        ((top + depth + 1, 2), (top + depth + 1, right)),
        ((top + depth + 1, 0), (top + depth + 1, 1)),
        ((top + 2, 0), (top + depth - 1, 0)),
    ]


def generate_input() -> np.ndarray:
    """Generate a random input grid for the puzzle using integer ARC colors."""

    rng = np.random.default_rng()
    h = int(rng.integers(20, 31))
    w = int(rng.integers(20, 31))
    grid = np.zeros((h, w), dtype=np.int32)

    palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng.shuffle(palette)
    num_colors = int(rng.integers(1, 4))

    for color in palette[:num_colors]:
        placed = False
        for _ in range(200):
            k = int(rng.integers(3, 5))
            segments = _build_shape_segments(k, rng)
            rng.shuffle(segments)

            cells = []
            for seg in segments:
                cells.extend(_segment_cells(seg))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            h_shape = max(rs) - min(rs) + 1
            w_shape = max(cs) - min(cs) + 1

            max_dr = h - h_shape - 2
            max_dc = w - w_shape - 2
            if max_dr < 1 or max_dc < 1:
                continue

            dr = int(rng.integers(1, max_dr + 1))
            dc = int(rng.integers(1, max_dc + 1))
            moved = _shift_segments(segments, dr - min(rs), dc - min(cs))

            moved_cells = []
            for seg in moved:
                moved_cells.extend(_segment_cells(seg))

            if not _can_place(grid, moved_cells):
                continue

            for seg in moved:
                _draw_straight_line(grid, color, *seg)
            placed = True
            break

        if not placed:
            continue

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Solve grouped broken segments into closed colored polygons.

    Input/assumptions:
        The background is 0. Non-zero colors are disjoint straight axis-aligned segments.
        For each color (3-5 segments), there is a unique cyclic endpoint order.

    Reconstruction:
        Extract connected straight segments, try segment permutations + endpoint orientations,
        and pick the cycle where consecutive chosen endpoints share row/column.

    Fill strategy:
        Draw connector segments, then flood-fill background pixels unreachable from the boundary
        of each color's bounding box while treating all non-zero pixels as walls.

    Returns:
        The solved grid as np.ndarray with each enclosed interior filled by its boundary color.
    """

    out = np.array(grid, copy=True)
    colors = [int(c) for c in np.unique(out) if c != 0]

    for color in colors:
        segments = _extract_segments(out, color)
        if len(segments) < 3:
            continue

        found = _find_cycle(segments)
        if found is None:
            continue

        order, endpoints = found
        n = len(order)

        for i in range(n):
            cur = segments[order[i]]
            nxt = segments[order[(i + 1) % n]]
            p_out = cur["ends"][endpoints[order[i]][1]]
            p_in = nxt["ends"][endpoints[order[(i + 1) % n]][0]]
            _draw_straight_line(out, color, p_out, p_in)

        _fill_inside_color(out, color)

    return out
