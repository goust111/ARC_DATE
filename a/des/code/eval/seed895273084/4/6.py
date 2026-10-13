import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 4
TRACK_COLOR = 5
TRAIN_COLORS = [1, 2, 3, 6, 7, 8, 9]

MIN_TRACKS = 1
MAX_TRACKS = 4
MIN_GRID = 18
MAX_GRID = 32
MAX_MARKERS_PER_TRACK = 8
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_TRIES = 200


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _clockwise_cycle(component):
    pts = set(component)
    nbrs = {pt: [nb for nb in _neighbors4(*pt) if nb in pts] for pt in pts}
    assert all(len(v) == 2 for v in nbrs.values()), "track must be a simple 4-connected cycle"

    start = min(pts)
    order = [start]
    prev = None
    cur = start
    while True:
        a, b = nbrs[cur]
        nxt = a if a != prev else b
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        assert len(order) <= len(pts), "cycle traversal exceeded component size"
    assert len(order) == len(pts), "cycle traversal missed cells"

    area2 = sum(c1 * r2 - c2 * r1 for (r1, c1), (r2, c2) in zip(order, order[1:] + order[:1]))
    if area2 < 0:
        order = [order[0]] + list(reversed(order[1:]))
    return order


def _inside_coords(component):
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    blocked = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
    for r, c in component:
        blocked[r - r0 + 1, c - c0 + 1] = True

    outside = np.zeros_like(blocked, dtype=bool)
    q = deque()
    h, w = blocked.shape
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    inside = []
    for rr, cc in np.argwhere((~blocked) & (~outside)):
        inside.append((r0 + int(rr) - 1, c0 + int(cc) - 1))
    return inside


def _infer_track_color(grid: np.ndarray, bg: int) -> int:
    if TRACK_COLOR in grid:
        return TRACK_COLOR
    colors = [int(v) for v in grid.flat if int(v) not in (bg, MARKER_COLOR)]
    if not colors:
        return TRACK_COLOR
    return Counter(colors).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll each colored train clockwise along its grey loop by the number of yellow markers inside that loop.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    track_color = _infer_track_color(g, bg)

    track_mask = (g != bg) & (g != MARKER_COLOR)
    out = g.copy()
    for component in _components(track_mask):
        order = _clockwise_cycle(component)
        train_colors = {int(g[r, c]) for r, c in component if int(g[r, c]) != track_color}
        assert len(train_colors) == 1, "each track must contain exactly one train color"
        train_color = next(iter(train_colors))

        marker_count = sum(int(g[r, c]) == MARKER_COLOR for r, c in _inside_coords(component))
        occupied = [i for i, (r, c) in enumerate(order) if int(g[r, c]) == train_color]
        assert occupied, "train must occupy at least one cell"

        for r, c in component:
            out[r, c] = track_color

        shift = marker_count % len(order)
        for idx in occupied:
            r, c = order[(idx + shift) % len(order)]
            out[r, c] = train_color

    return out


def _walk_vertices(vertices):
    order = []
    closed = list(vertices) + [vertices[0]]
    for (r1, c1), (r2, c2) in zip(closed, closed[1:]):
        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        dc = 0 if c1 == c2 else (1 if c2 > c1 else -1)
        steps = max(abs(r2 - r1), abs(c2 - c1))
        for k in range(steps):
            order.append((r1 + dr * k, c1 + dc * k))
    return order


def _rectangle_cycle():
    h = random.randint(4, 10)
    w = random.randint(4, 10)
    return _walk_vertices([(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)])


def _bump_cycle():
    side = random.choice(["right", "left", "top", "bottom"])
    depth = random.randint(2, 4)
    if side in ("right", "left"):
        h = random.randint(6, 10)
        w = random.randint(4, 8)
        a = random.randint(1, h - 4)
        b = random.randint(a + 2, h - 2)
        if side == "right":
            vertices = [(0, 0), (0, w - 1), (a, w - 1), (a, w + depth - 1), (b, w + depth - 1), (b, w - 1), (h - 1, w - 1), (h - 1, 0)]
        else:
            vertices = [(0, depth), (0, w + depth - 1), (h - 1, w + depth - 1), (h - 1, depth), (b, depth), (b, 0), (a, 0), (a, depth)]
    else:
        h = random.randint(4, 8)
        w = random.randint(6, 10)
        a = random.randint(1, w - 4)
        b = random.randint(a + 2, w - 2)
        if side == "top":
            vertices = [(depth, 0), (depth, a), (0, a), (0, b), (depth, b), (depth, w - 1), (h + depth - 1, w - 1), (h + depth - 1, 0)]
        else:
            vertices = [(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, b), (h + depth - 1, b), (h + depth - 1, a), (h - 1, a), (h - 1, 0)]
    return _walk_vertices(vertices)


def _build_patch(train_color: int):
    raw_cycle = _rectangle_cycle() if random.random() < 0.5 else _bump_cycle()
    cycle = _clockwise_cycle(raw_cycle)
    h = max(r for r, _ in cycle) + 1
    w = max(c for _, c in cycle) + 1

    patch = np.full((h, w), BG_COLOR, dtype=int)
    for r, c in cycle:
        patch[r, c] = TRACK_COLOR

    inside = _inside_coords(cycle)
    assert inside, "generated track must enclose at least one interior cell"

    max_markers = min(len(inside), MAX_MARKERS_PER_TRACK)
    if max_markers == 0:
        marker_count = 0
    elif random.random() < 0.5:
        marker_count = 0
    else:
        marker_count = random.randint(1, max_markers)
    if marker_count:
        for r, c in random.sample(inside, marker_count):
            patch[r, c] = MARKER_COLOR

    train_len = random.randint(1, max(1, len(cycle) - 2))
    start = random.randrange(len(cycle))
    for k in range(train_len):
        r, c = cycle[(start + k) % len(cycle)]
        patch[r, c] = train_color

    return {"patch": patch, "markers": marker_count}


def _boxes_overlap(a, b) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_patches(patches, rows: int, cols: int):
    grid = np.full((rows, cols), BG_COLOR, dtype=int)
    occupied_boxes = []

    for spec in sorted(patches, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        patch = spec["patch"]
        ph, pw = patch.shape
        if ph + 2 > rows or pw + 2 > cols:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(1, rows - ph - 1)
            c0 = random.randint(1, cols - pw - 1)
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            expanded = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
            if any(_boxes_overlap(expanded, other) for other in occupied_boxes):
                continue

            mask = patch != BG_COLOR
            region = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(region[mask] != BG_COLOR):
                continue
            region[mask] = patch[mask]
            occupied_boxes.append(expanded)
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt():
    n_tracks = random.randint(MIN_TRACKS, MAX_TRACKS)
    colors = random.sample(TRAIN_COLORS, n_tracks)
    patches = [_build_patch(color) for color in colors]
    if sum(spec["markers"] for spec in patches) == 0:
        return None

    max_h = max(spec["patch"].shape[0] for spec in patches)
    max_w = max(spec["patch"].shape[1] for spec in patches)
    rows_lb = max(MIN_GRID, max_h + 2, 18 + 3 * (n_tracks - 1))
    cols_lb = max(MIN_GRID, max_w + 2, 18 + 3 * (n_tracks - 1))
    if rows_lb > MAX_GRID or cols_lb > MAX_GRID:
        return None

    rows = random.randint(rows_lb, MAX_GRID)
    cols = random.randint(cols_lb, MAX_GRID)
    grid = _place_patches(patches, rows, cols)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid loop-train input")
