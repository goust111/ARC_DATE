import random
from collections import deque

import numpy as np


BLACK = 0
AZURE = 8
COLORS = [1, 2, 3, 4, 5, 6, 7, 9]


def _line_splits(grid):
    h, w = grid.shape
    hlines = [r for r in range(h) if np.all(grid[r, :] == AZURE)]
    vlines = [c for c in range(w) if np.all(grid[:, c] == AZURE)]

    row_bounds = []
    start = 0
    for r in hlines + [h]:
        if start < r:
            row_bounds.append((start, r))
        start = r + 1

    col_bounds = []
    start = 0
    for c in vlines + [w]:
        if start < c:
            col_bounds.append((start, c))
        start = c + 1

    return row_bounds, col_bounds


def _same_color_components(sub):
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(sub[r, c])
            if color in (BLACK, AZURE) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pixels = []
            while q:
                cr, cc = q.popleft()
                pixels.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(sub[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "pixels": pixels})
    return comps


def _stamp_offsets(pixels):
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    ar = (min(rows) + max(rows)) // 2
    ac = (min(cols) + max(cols)) // 2
    center = ((min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0)
    return [(r - ar, c - ac) for r, c in pixels], center


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp each local template onto its colored ink dots, from nearest dot to farthest.

    Input structure:
    A black grid is divided by full azure horizontal and vertical separator lines into
    rectangular subgrids.  Each subgrid contains one same-color connected stamp pattern
    and several one-cell ink dots whose colors differ from the stamp and from each other.
    Rule:
    For every subgrid independently, identify the multi-cell stamp component and the
    single-cell ink dots.  Clear the original stamp to black.  Sort the ink dots by their
    Euclidean distance from the stamp bounding-box center, nearest first.  At each dot,
    draw the stamp shape with the dot's color, aligning the stamp's bounding-box center
    to the dot.  Later, farther stamps overwrite earlier, nearer stamps.  Azure separator
    lines remain unchanged.
    Key insight:
    The stamp is a reusable template, the ink-dot color supplies the paint, and overlap
    colors reveal that distance controls the drawing order from near to far.
    """
    out = np.array(grid, copy=True)
    row_bounds, col_bounds = _line_splits(grid)

    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            sub = grid[r0:r1, c0:c1]
            comps = _same_color_components(sub)
            large = [comp for comp in comps if len(comp["pixels"]) > 1]
            if not large:
                continue

            stamp = max(large, key=lambda comp: len(comp["pixels"]))
            offsets, center = _stamp_offsets(stamp["pixels"])
            dots = [
                comp
                for comp in comps
                if len(comp["pixels"]) == 1 and comp["color"] != stamp["color"]
            ]

            for rr, cc in stamp["pixels"]:
                out[r0 + rr, c0 + cc] = BLACK

            ordered = sorted(
                dots,
                key=lambda comp: (
                    (comp["pixels"][0][0] - center[0]) ** 2
                    + (comp["pixels"][0][1] - center[1]) ** 2,
                    comp["pixels"][0][0],
                    comp["pixels"][0][1],
                ),
            )

            for dot in ordered:
                dr, dc = dot["pixels"][0]
                color = dot["color"]
                for off_r, off_c in offsets:
                    rr, cc = dr + off_r, dc + off_c
                    if 0 <= rr < r1 - r0 and 0 <= cc < c1 - c0:
                        out[r0 + rr, c0 + cc] = color

    return out


def _normalize_offsets(offsets):
    rows = [p[0] for p in offsets]
    cols = [p[1] for p in offsets]
    ar = (min(rows) + max(rows)) // 2
    ac = (min(cols) + max(cols)) // 2
    norm = {(r - ar, c - ac) for r, c in offsets}
    norm.add((0, 0))
    return sorted(norm)


def _transform_shape(offsets):
    k = random.randrange(4)
    flip = random.choice([False, True])
    pts = []
    for r, c in offsets:
        for _ in range(k):
            r, c = -c, r
        if flip:
            c = -c
        pts.append((r, c))
    return _normalize_offsets(pts)


def _shape_library():
    return [
        [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
        [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 0)],
        [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, -1)],
        [(-1, 0), (-1, 1), (0, 0), (1, -1), (1, 0)],
        [(-1, 0), (0, -1), (0, 0), (0, 1), (1, -1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1), (1, 2)],
    ]


def _draw(grid, r0, c0, anchor_r, anchor_c, offsets, color):
    for dr, dc in offsets:
        grid[r0 + anchor_r + dr, c0 + anchor_c + dc] = color


def _valid_dot_candidates(h, w, offsets, stamp_pixels, center, used):
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)
    candidates = []
    distances = set()
    for r in range(-min_r, h - max_r):
        for c in range(-min_c, w - max_c):
            if (r, c) in stamp_pixels or (r, c) in used:
                continue
            if r < center[0] + 2 and c < center[1] + 2:
                continue
            dist = (r - center[0]) ** 2 + (c - center[1]) ** 2
            if dist in distances:
                continue
            distances.add(dist)
            candidates.append((r, c, dist))
    return candidates


def _choose_dots(h, w, offsets, stamp_anchor, k):
    stamp_pixels = {
        (stamp_anchor[0] + dr, stamp_anchor[1] + dc) for dr, dc in offsets
    }
    rows = [r for r, _ in stamp_pixels]
    cols = [c for _, c in stamp_pixels]
    center = ((min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0)

    candidates = _valid_dot_candidates(h, w, offsets, stamp_pixels, center, set())
    random.shuffle(candidates)

    for base_r, base_c, _ in candidates:
        pool = []
        used = {(base_r, base_c)}
        base_dist = (base_r - center[0]) ** 2 + (base_c - center[1]) ** 2
        dists = {base_dist}
        near_shifts = [
            (-2, -1),
            (-2, 0),
            (-1, -2),
            (-1, 1),
            (0, -2),
            (0, 2),
            (1, -1),
            (1, 2),
            (2, 0),
            (2, 1),
        ]
        random.shuffle(near_shifts)
        for sr, sc in near_shifts:
            rr, cc = base_r + sr, base_c + sc
            local = _valid_dot_candidates(h, w, offsets, stamp_pixels, center, used)
            valid = {(r, c): d for r, c, d in local}
            dist = valid.get((rr, cc))
            if dist is not None and dist not in dists:
                pool.append((rr, cc))
                used.add((rr, cc))
                dists.add(dist)
                if len(pool) + 1 == k:
                    dots = [(base_r, base_c)] + pool
                    return sorted(
                        dots,
                        key=lambda p: (p[0] - center[0]) ** 2
                        + (p[1] - center[1]) ** 2,
                    )

    chosen = []
    used = set()
    dists = set()
    candidates = _valid_dot_candidates(h, w, offsets, stamp_pixels, center, used)
    random.shuffle(candidates)
    for r, c, dist in sorted(candidates, key=lambda x: x[2] + random.random()):
        if dist in dists:
            continue
        chosen.append((r, c))
        used.add((r, c))
        dists.add(dist)
        if len(chosen) == k:
            break
    return chosen


def generate_input() -> np.ndarray:
    row_count = random.choice([2, 2, 3])
    col_count = random.choice([2, 2, 3])
    heights = [random.randint(7, 9) for _ in range(row_count)]
    widths = [random.randint(7, 9) for _ in range(col_count)]
    total_h = sum(heights) + row_count - 1
    total_w = sum(widths) + col_count - 1
    grid = np.zeros((total_h, total_w), dtype=int)

    r = 0
    row_starts = []
    for i, h in enumerate(heights):
        row_starts.append(r)
        r += h
        if i + 1 < row_count:
            grid[r, :] = AZURE
            r += 1

    c = 0
    col_starts = []
    for i, w in enumerate(widths):
        col_starts.append(c)
        c += w
        if i + 1 < col_count:
            grid[:, c] = AZURE
            c += 1

    for r0, h in zip(row_starts, heights):
        for c0, w in zip(col_starts, widths):
            offsets = _transform_shape(random.choice(_shape_library()))
            min_r = min(r for r, _ in offsets)
            max_r = max(r for r, _ in offsets)
            min_c = min(c for _, c in offsets)
            max_c = max(c for _, c in offsets)

            ar_min = max(0, -min_r)
            ac_min = max(0, -min_c)
            ar_max = min(h - 1 - max_r, ar_min + random.randint(1, 3))
            ac_max = min(w - 1 - max_c, ac_min + random.randint(1, 3))
            stamp_r = random.randint(ar_min, ar_max)
            stamp_c = random.randint(ac_min, ac_max)

            stamp_color = random.choice(COLORS)
            dot_colors = [color for color in COLORS if color != stamp_color]
            random.shuffle(dot_colors)
            dot_count = random.choice([2, 3, 3, 4])

            _draw(grid, r0, c0, stamp_r, stamp_c, offsets, stamp_color)
            dots = _choose_dots(h, w, offsets, (stamp_r, stamp_c), dot_count)
            for (dr, dc), color in zip(dots, dot_colors):
                grid[r0 + dr, c0 + dc] = color

    return grid
