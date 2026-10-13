from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in _neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        nxt = [(nr, nc) for nr, nc in _neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(nxt) == 1, (cur, nxt)
        prev, cur = cur, nxt[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _concave_count(path: list[tuple[int, int]]) -> int:
    counts: Counter[int] = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross != 0:
            counts[1 if cross > 0 else -1] += 1
    assert counts
    assert abs(counts[1] - counts[-1]) == 4, counts
    return min(counts[1], counts[-1])


def _interior_mask_from_cycle(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())

    local_h = r1 - r0 + 3
    local_w = c1 - c0 + 3
    wall = np.zeros((local_h, local_w), dtype=bool)
    for r, c in coords:
        wall[int(r) - r0 + 1, int(c) - c0 + 1] = True

    outside = np.zeros_like(wall, dtype=bool)
    q = deque([(0, 0)])
    outside[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < local_h and 0 <= nc < local_w and not wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    interior_local = (~wall) & (~outside)
    interior = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    interior[:] = interior_local[1:-1, 1:-1]
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue loop interior with parity-colored square rings whose maximum depth equals that loop's concave-bend count.

    Input structure:
    - The grid uses black background, blue one-pixel-wide closed orthogonal loops, and exactly one green control pixel inside each loop.
    - Different blue loops are 4-connected cycles; every loop pixel has degree 2 and loops do not intersect or nest.
    - Each loop encloses a black interior region containing its own green control pixel.

    Rule:
    - Trace each blue loop, inspect the turn sign at every non-straight path pixel, and count the minority-sign bends; this count is the loop's concave-bend total K.
    - Preserve every blue loop pixel and every green control pixel.
    - For black pixels strictly inside that loop, compute Chebyshev distance D from the loop's green control pixel.
    - If 1 <= D <= K, recolor the pixel yellow when D is odd and red when D is even; interior pixels with D > K stay black.

    Key insight:
    - The loop's geometry is first reduced to a single parameter K via minority turn counting, and that parameter then limits how many concentric Chebyshev rings may be drawn around the loop's control point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for coords in _components(g == BLUE):
        path = _trace_cycle(coords)
        k = _concave_count(path)

        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        interior_local = _interior_mask_from_cycle(coords)

        greens = np.argwhere((g[r0 : r0 + interior_local.shape[0], c0 : c0 + interior_local.shape[1]] == GREEN) & interior_local)
        assert greens.shape[0] == 1, greens
        control_r = r0 + int(greens[0, 0])
        control_c = c0 + int(greens[0, 1])

        for lr, lc in np.argwhere(interior_local):
            r = r0 + int(lr)
            c = c0 + int(lc)
            if g[r, c] != BLACK:
                continue
            dist = max(abs(r - control_r), abs(c - control_c))
            if 1 <= dist <= k:
                out[r, c] = YELLOW if dist % 2 == 1 else RED

    return out


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[0] == cells[0] and len(path) > 1:
        cells = cells[1:]
    elif path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    for c in range(width - 1):
        _append_segment(path, top[c], c, top[c], c + 1)
        _append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _append_segment(path, bottom[c], c, bottom[c], c - 1)
        _append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _sample_profiles() -> tuple[list[int], list[int]] | None:
    width = random.randint(8, 16)
    height = random.randint(8, 16)
    top = [0] * width
    bottom = [height - 1] * width

    used_top = set()
    used_bottom = set()
    complexity = random.choices([0, 1, 2, 3, 4], weights=[1, 2, 3, 3, 2])[0]
    for _ in range(complexity):
        kind = random.choice(["tl", "tr", "bl", "br", "top", "bottom"])
        depth_cap = max(1, height // 2 - 1)

        if kind == "tl":
            max_len = min(4, width - 2)
            if max_len < 1:
                continue
            length = random.randint(1, max_len)
            cols = range(0, length)
            if any(c in used_top for c in cols):
                continue
            depth = random.randint(1, depth_cap)
            if any(depth + 2 > bottom[c] for c in cols):
                continue
            for c in cols:
                top[c] = depth
                used_top.add(c)
        elif kind == "tr":
            max_len = min(4, width - 2)
            if max_len < 1:
                continue
            length = random.randint(1, max_len)
            cols = range(width - length, width)
            if any(c in used_top for c in cols):
                continue
            depth = random.randint(1, depth_cap)
            if any(depth + 2 > bottom[c] for c in cols):
                continue
            for c in cols:
                top[c] = depth
                used_top.add(c)
        elif kind == "bl":
            max_len = min(4, width - 2)
            if max_len < 1:
                continue
            length = random.randint(1, max_len)
            cols = range(0, length)
            if any(c in used_bottom for c in cols):
                continue
            depth = random.randint(1, depth_cap)
            level = height - 1 - depth
            if any(top[c] + 2 > level for c in cols):
                continue
            for c in cols:
                bottom[c] = level
                used_bottom.add(c)
        elif kind == "br":
            max_len = min(4, width - 2)
            if max_len < 1:
                continue
            length = random.randint(1, max_len)
            cols = range(width - length, width)
            if any(c in used_bottom for c in cols):
                continue
            depth = random.randint(1, depth_cap)
            level = height - 1 - depth
            if any(top[c] + 2 > level for c in cols):
                continue
            for c in cols:
                bottom[c] = level
                used_bottom.add(c)
        elif kind == "top":
            if width < 5:
                continue
            start = random.randint(1, width - 3)
            end = random.randint(start, min(width - 2, start + 2))
            cols = range(start, end + 1)
            if any(c in used_top for c in cols):
                continue
            depth = random.randint(1, depth_cap)
            if any(depth + 2 > bottom[c] for c in cols):
                continue
            for c in cols:
                top[c] = depth
                used_top.add(c)
        else:
            if width < 5:
                continue
            start = random.randint(1, width - 3)
            end = random.randint(start, min(width - 2, start + 2))
            cols = range(start, end + 1)
            if any(c in used_bottom for c in cols):
                continue
            depth = random.randint(1, depth_cap)
            level = height - 1 - depth
            if any(top[c] + 2 > level for c in cols):
                continue
            for c in cols:
                bottom[c] = level
                used_bottom.add(c)

    if any(top[c] >= bottom[c] for c in range(width)):
        return None
    return top, bottom


def _build_patch() -> np.ndarray | None:
    sampled = _sample_profiles()
    if sampled is None:
        return None
    top, bottom = sampled
    path = _path_from_profiles(top, bottom)
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    norm_path = [(r - min_r, c - min_c) for r, c in path]

    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in norm_path:
        patch[r, c] = BLUE

    coords = np.argwhere(patch == BLUE)
    try:
        traced = _trace_cycle(coords)
        k = _concave_count(traced)
    except AssertionError:
        return None
    if k > 6:
        return None

    interior = _interior_mask_from_cycle(coords)
    interior_cells = np.argwhere(interior)
    if interior_cells.size == 0:
        return None

    loop_pts = np.argwhere(patch == BLUE)
    candidates: list[tuple[int, int, int]] = []
    for r, c in interior_cells:
        dist = min(max(abs(int(r) - int(br)), abs(int(c) - int(bc))) for br, bc in loop_pts)
        if dist >= k:
            candidates.append((int(r), int(c), int(dist)))
    if not candidates:
        return None

    best_clearance = max(dist for _, _, dist in candidates)
    eligible = [(r, c) for r, c, dist in candidates if dist == best_clearance or dist >= max(k, best_clearance - 1)]
    control_r, control_c = random.choice(eligible)
    patch[control_r, control_c] = GREEN

    if random.random() < 0.75:
        patch = np.rot90(patch, random.randint(0, 3))
    return patch


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 2) -> bool:
    ph, pw = patch.shape
    h, w = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc1 = min(w, c0 + pw + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _stamp(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    view = dst[r0 : r0 + ph, c0 : c0 + pw]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_loops = random.choices([1, 2, 3], weights=[2, 3, 2])[0]
    ks: list[int] = []
    for _ in range(n_loops):
        patch = None
        for _ in range(80):
            patch = _build_patch()
            if patch is not None:
                break
        if patch is None:
            return None

        coords = np.argwhere(patch == BLUE)
        k = _concave_count(_trace_cycle(coords))
        ks.append(k)

        ph, pw = patch.shape
        placed = False
        for _ in range(120):
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            if not _can_place(occupied, patch, r0, c0):
                continue
            _stamp(grid, patch, r0, c0)
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch != BLACK
            placed = True
            break
        if not placed:
            return None

    if max(ks) == 0 and random.random() < 0.7:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or not np.any(out == RED) and not np.any(out == YELLOW):
        if max(ks) > 0:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
