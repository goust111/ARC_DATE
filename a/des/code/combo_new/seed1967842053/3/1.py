import numpy as np


def _component(mask, seen, start_r, start_c):
    h, w = mask.shape
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    cells = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return cells


def _frame_from_component(grid, color, cells):
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    r1, r2 = min(rows), max(rows)
    c1, c2 = min(cols), max(cols)
    height = r2 - r1 + 1
    width = c2 - c1 + 1
    if height < 3 or width < 3:
        return None
    expected = 2 * height + 2 * width - 4
    if len(cells) != expected:
        return None
    for r, c in cells:
        if not (r in (r1, r2) or c in (c1, c2)):
            return None
    if not (
        np.all(grid[r1, c1 : c2 + 1] == color)
        and np.all(grid[r2, c1 : c2 + 1] == color)
        and np.all(grid[r1 : r2 + 1, c1] == color)
        and np.all(grid[r1 : r2 + 1, c2] == color)
    ):
        return None
    return (r1, c1, r2, c2, color)


def _find_frames(grid):
    frames = []
    h, w = grid.shape
    for value in np.unique(grid):
        color = int(value)
        if color == 0:
            continue
        mask = grid == color
        seen = np.zeros((h, w), dtype=bool)
        starts = np.argwhere(mask)
        for r, c in starts:
            r, c = int(r), int(c)
            if seen[r, c]:
                continue
            cells = _component(mask, seen, r, c)
            frame = _frame_from_component(grid, color, cells)
            if frame is not None:
                frames.append(frame)
    return frames


def _find_seed(grid, frame):
    r1, c1, r2, c2, frame_color = frame
    inner = grid[r1 + 1 : r2, c1 + 1 : c2]
    locs = np.argwhere((inner != 0) & (inner != frame_color))
    if len(locs) == 0:
        return None
    rr, cc = locs[0]
    sr, sc = r1 + 1 + int(rr), c1 + 1 + int(cc)
    return sr, sc, int(grid[sr, sc])


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each colored seed into clipped concentric rectangular layers inside its frame.

    Input structure: The grid has black background, several one-pixel hollow rectangular
    frames in distinct colors, exactly one differently colored seed inside each frame,
    and isolated outside marker pixels whose colors match those seeds.
    Rule: For every frame, pair it with its interior seed color C, count the pixels of C
    outside all frame boxes, erase those outside markers, then color the seed-centered
    rectangle of Chebyshev radius N inside the frame interior, clipping at the frame.
    Key insight: The outside pixels are counters for the matching seed color; their count
    becomes the number of rectangular growth layers rather than decoration.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    frames = _find_frames(grid)
    if not frames:
        return out

    box_mask = np.zeros(grid.shape, dtype=bool)
    for r1, c1, r2, c2, _ in frames:
        box_mask[r1 : r2 + 1, c1 : c2 + 1] = True
    outside = ~box_mask

    tasks = []
    for frame in frames:
        seed = _find_seed(grid, frame)
        if seed is not None:
            tasks.append((frame, seed))

    for _, (_, _, color) in tasks:
        out[(grid == color) & outside] = 0

    for frame, seed in tasks:
        r1, c1, r2, c2, _ = frame
        sr, sc, color = seed
        layers = int(np.count_nonzero((grid == color) & outside))
        rlo = max(r1 + 1, sr - layers)
        rhi = min(r2 - 1, sr + layers)
        clo = max(c1 + 1, sc - layers)
        chi = min(c2 - 1, sc + layers)
        out[rlo : rhi + 1, clo : chi + 1] = color

    return out


def _overlaps(a, b, pad=0):
    ar1, ac1, ar2, ac2 = a
    br1, bc1, br2, bc2 = b
    return not (
        ar2 + pad < br1
        or br2 + pad < ar1
        or ac2 + pad < bc1
        or bc2 + pad < ac1
    )


def _draw_frame(grid, rect, color):
    r1, c1, r2, c2 = rect
    grid[r1, c1 : c2 + 1] = color
    grid[r2, c1 : c2 + 1] = color
    grid[r1 : r2 + 1, c1] = color
    grid[r1 : r2 + 1, c2] = color


def _near_rect(r, c, rect, pad):
    r1, c1, r2, c2 = rect
    return r1 - pad <= r <= r2 + pad and c1 - pad <= c <= c2 + pad


def _marker_candidates(grid, rects, used):
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 or (r, c) in used:
                continue
            if any(_near_rect(r, c, rect, 1) for rect in rects):
                continue
            blocked = False
            for ur, uc in used:
                if abs(r - ur) <= 1 and abs(c - uc) <= 1:
                    blocked = True
                    break
            if not blocked:
                candidates.append((r, c))
    return candidates


def _choose_seed(rng, rect, want_clip):
    r1, c1, r2, c2 = rect
    if not want_clip and r2 - r1 >= 4 and c2 - c1 >= 4:
        sr = int(rng.integers(r1 + 2, r2 - 1))
        sc = int(rng.integers(c1 + 2, c2 - 1))
    else:
        sr = int(rng.integers(r1 + 1, r2))
        sc = int(rng.integers(c1 + 1, c2))
    return sr, sc


def _layer_count(rng, rect, seed, want_clip):
    r1, c1, r2, c2 = rect
    sr, sc = seed
    dists = [sr - (r1 + 1), (r2 - 1) - sr, sc - (c1 + 1), (c2 - 1) - sc]
    clear = min(dists)
    reach = max(dists)
    if want_clip:
        lo = clear + 1
        hi = min(reach + 3, 8)
        if hi < lo:
            hi = lo
        return int(rng.integers(lo, hi + 1))
    hi = max(1, min(clear, 4))
    return int(rng.integers(1, hi + 1))


def _try_generate(rng):
    height = int(rng.integers(16, 31))
    width = int(rng.integers(16, 31))
    object_count = int(rng.integers(1, 4))
    colors = rng.choice(np.arange(1, 10), size=object_count * 2, replace=False)
    frame_colors = [int(x) for x in colors[:object_count]]
    seed_colors = [int(x) for x in colors[object_count:]]

    rects = []
    for _ in range(object_count):
        placed = False
        for _attempt in range(160):
            max_fh = min(10, height - 3)
            max_fw = min(10, width - 3)
            if max_fh < 5 or max_fw < 5:
                return None
            fh = int(rng.integers(5, max_fh + 1))
            fw = int(rng.integers(5, max_fw + 1))
            if height - fh <= 1 or width - fw <= 1:
                continue
            r1 = int(rng.integers(1, height - fh))
            c1 = int(rng.integers(1, width - fw))
            rect = (r1, c1, r1 + fh - 1, c1 + fw - 1)
            if all(not _overlaps(rect, other, pad=2) for other in rects):
                rects.append(rect)
                placed = True
                break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    for rect, color in zip(rects, frame_colors):
        _draw_frame(grid, rect, color)

    marker_plan = []
    for i, rect in enumerate(rects):
        want_clip = bool(rng.integers(0, 2))
        seed = _choose_seed(rng, rect, want_clip)
        layers = _layer_count(rng, rect, seed, want_clip)
        sr, sc = seed
        grid[sr, sc] = seed_colors[i]
        marker_plan.append((seed_colors[i], layers))

    used_markers = set()
    for color, count in marker_plan:
        for _ in range(count):
            candidates = _marker_candidates(grid, rects, used_markers)
            if not candidates:
                return None
            idx = int(rng.integers(0, len(candidates)))
            r, c = candidates[idx]
            grid[r, c] = color
            used_markers.add((r, c))

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _try_generate(rng)
        if grid is not None:
            return grid

    grid = np.zeros((18, 18), dtype=int)
    rect = (3, 3, 10, 10)
    _draw_frame(grid, rect, 2)
    grid[6, 6] = 5
    grid[2, 14] = 5
    grid[14, 2] = 5
    grid[15, 15] = 5
    return grid
