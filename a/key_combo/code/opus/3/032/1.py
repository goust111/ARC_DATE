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
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    """4-neighborhood in arc_utils (r,c) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Boolean-mask connected components, matching arc_utils coordinate conventions."""
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        nxt = [(nr, nc) for nr, nc in neighbors4(*cur) if (nr, nc) in pts and (nr, nc) != prev]
        assert len(nxt) == 1, (cur, nxt)
        prev, cur = cur, nxt[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _classify_bends(path: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    turn_signs: dict[tuple[int, int], int] = {}
    counts: Counter[int] = Counter()
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            continue
        sign = 1 if cross > 0 else -1
        turn_signs[(r, c)] = sign
        counts[sign] += 1

    assert turn_signs
    assert abs(counts[1] - counts[-1]) == 4, counts
    convex_sign = 1 if counts[1] > counts[-1] else -1
    convex = [cell for cell, sign in turn_signs.items() if sign == convex_sign]
    concave = [cell for cell, sign in turn_signs.items() if sign != convex_sign]
    return convex, concave


def _mark_halo(mask: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    h, w = mask.shape
    r0 = max(0, r - 1)
    r1 = min(h, r + 2)
    c0 = max(0, c - 1)
    c1 = min(w, c + 2)
    mask[r0:r1, c0:c1] = True


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp yellow halos around convex bends and red halos around concave bends, recoloring mixed overlap to blue.

    Input structure:
    - The background is black, and the only foreground object color forms one or more one-pixel-wide 4-connected orthogonal cycles.
    - Every foreground pixel has exactly two foreground 4-neighbors, so each component is a simple closed loop.
    - Loops stay away from the border and may contain both outward and inward rectilinear bends.

    Rule:
    - Trace each loop in cycle order and classify every non-straight pixel by turn sign.
    - Within one loop, the majority turn direction marks convex bends and the minority turn direction marks concave bends.
    - For every convex bend, paint black cells in its 3x3 neighborhood yellow; for every concave bend, paint black cells in its 3x3 neighborhood red.
    - Any original background cell covered by both a convex halo and a concave halo is recolored blue, while loop pixels stay unchanged.

    Key insight:
    - Convex versus concave cannot be read from a local corner alone; the loop must be traced so the minority turn sign reveals the inward corners, and only then can the overlap-of-two-halos rule produce blue conflict pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    colors = sorted(set(map(int, g.reshape(-1))) - {bg})
    assert len(colors) == 1
    fg = int(colors[0])

    convex_cover = np.zeros_like(g, dtype=bool)
    concave_cover = np.zeros_like(g, dtype=bool)

    for coords in components_mask(g == fg, connectivity=4):
        path = _trace_cycle(coords)
        convex, concave = _classify_bends(path)
        for cell in convex:
            _mark_halo(convex_cover, cell)
        for cell in concave:
            _mark_halo(concave_cover, cell)

    bg_mask = g == bg
    yellow_mask = convex_cover & bg_mask
    red_mask = concave_cover & bg_mask
    blue_mask = yellow_mask & red_mask

    out = g.copy()
    out[yellow_mask] = YELLOW
    out[red_mask] = RED
    out[blue_mask] = BLUE
    return out


# --------------- generator ---------------

GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (15, 24)
LOOP_COUNT_CHOICES = [1, 2, 2, 3]
PATCH_ATTEMPTS = 80
GEN_RETRIES = 400

MIN_PATCH_H = 5
MAX_PATCH_H_SINGLE = 11
MAX_PATCH_H_MULTI = 9
MIN_PATCH_W = 6
MAX_PATCH_W_SINGLE = 13
MAX_PATCH_W_MULTI = 10

NOTCH_COUNT_CHOICES = [1, 1, 2, 2, 3]
MIN_NOTCH_WIDTH = 2
MAX_NOTCH_WIDTH = 4
EDGE_MARGIN = 2
SAME_SIDE_GAP = 2
MAX_NOTCH_DEPTH = 2
PLACEMENT_MARGIN = 1
MAX_OUTPUT_DENSITY = 0.42


def _append_segment(path: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _path_from_profiles(top: list[int], bottom: list[int]) -> list[tuple[int, int]]:
    width = len(top)
    path: list[tuple[int, int]] = []
    _append_segment(path, top[0], 0, top[0], 0)
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


def _sample_intervals(width: int, count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    max_width = min(MAX_NOTCH_WIDTH, width - 2 * EDGE_MARGIN)
    if max_width < MIN_NOTCH_WIDTH:
        return None

    candidates: list[tuple[int, int]] = []
    for span in range(MIN_NOTCH_WIDTH, max_width + 1):
        start_hi = width - EDGE_MARGIN - span
        for start in range(EDGE_MARGIN, start_hi + 1):
            end = start + span - 1
            candidates.append((start, end))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for start, end in candidates:
        if any(not (end + SAME_SIDE_GAP < s or start > e + SAME_SIDE_GAP) for s, e in chosen):
            continue
        chosen.append((start, end))
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _sample_depth(max_depth: int) -> int:
    if max_depth <= 1:
        return 1
    return random.choice([1, 1, 2])


def _apply_notch(top: list[int], bottom: list[int], side: str, start: int, end: int) -> bool:
    cols = range(start, end + 1)
    if side == "top":
        max_depth = min(MAX_NOTCH_DEPTH, min(bottom[c] - 2 for c in cols))
        if max_depth < 1:
            return False
        depth = _sample_depth(max_depth)
        for c in cols:
            top[c] = depth
        return True

    max_raise = min(MAX_NOTCH_DEPTH, min(bottom[c] - top[c] - 2 for c in cols))
    if max_raise < 1:
        return False
    raise_amt = _sample_depth(max_raise)
    for c in cols:
        bottom[c] -= raise_amt
    return True


def _bend_spacing_ok(bends: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(bends):
        for r2, c2 in bends[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _build_patch(top: list[int], bottom: list[int]) -> np.ndarray | None:
    path = _path_from_profiles(top, bottom)
    if len(set(path)) != len(path):
        return None

    coords = np.asarray(path, dtype=int)
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[coords[:, 0], coords[:, 1]] = GREEN

    try:
        traced = _trace_cycle(np.argwhere(patch == GREEN))
        convex, concave = _classify_bends(traced)
    except AssertionError:
        return None

    if not concave or not convex:
        return None
    if not _bend_spacing_ok(convex + concave):
        return None
    return patch


def _random_orient(patch: np.ndarray) -> np.ndarray:
    out = np.rot90(patch, random.randrange(4)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    return out


def _generate_patch(loop_count: int) -> np.ndarray | None:
    max_h = MAX_PATCH_H_SINGLE if loop_count == 1 else MAX_PATCH_H_MULTI
    max_w = MAX_PATCH_W_SINGLE if loop_count == 1 else MAX_PATCH_W_MULTI
    for _ in range(PATCH_ATTEMPTS):
        height = random.randint(MIN_PATCH_H, max_h)
        width = random.randint(MIN_PATCH_W, max_w)
        notch_count = random.choice(NOTCH_COUNT_CHOICES)

        top_count = random.randint(0, notch_count)
        bottom_count = notch_count - top_count
        top_intervals = _sample_intervals(width, top_count)
        bottom_intervals = _sample_intervals(width, bottom_count)
        if top_intervals is None or bottom_intervals is None:
            continue

        top = [0] * width
        bottom = [height - 1] * width
        plan = [("top", s, e) for s, e in top_intervals] + [("bottom", s, e) for s, e in bottom_intervals]
        random.shuffle(plan)

        ok = True
        for side, start, end in plan:
            if not _apply_notch(top, bottom, side, start, end):
                ok = False
                break
        if not ok or (top == [0] * width and bottom == [height - 1] * width):
            continue

        patch = _build_patch(top, bottom)
        if patch is None:
            continue
        return _random_orient(patch)
    return None


def _can_place(coords: np.ndarray, r0: int, c0: int, occupied: set[tuple[int, int]]) -> bool:
    for pr, pc in coords:
        r = r0 + int(pr)
        c = c0 + int(pc)
        for dr, dc in DIRS8:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _place_patches(canvas: np.ndarray, patches: list[np.ndarray]) -> bool:
    occupied: set[tuple[int, int]] = set()
    h, w = canvas.shape
    for patch in sorted(patches, key=lambda x: int(np.count_nonzero(x)), reverse=True):
        ph, pw = patch.shape
        coords = np.argwhere(patch == GREEN)
        candidates: list[tuple[int, int]] = []
        for r0 in range(PLACEMENT_MARGIN, h - ph - PLACEMENT_MARGIN + 1):
            for c0 in range(PLACEMENT_MARGIN, w - pw - PLACEMENT_MARGIN + 1):
                if _can_place(coords, r0, c0, occupied):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        canvas[r0 : r0 + ph, c0 : c0 + pw][patch == GREEN] = GREEN
        occupied.update((r0 + int(pr), c0 + int(pc)) for pr, pc in coords)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    loop_count = random.choice(LOOP_COUNT_CHOICES)

    patches: list[np.ndarray] = []
    for _ in range(loop_count):
        patch = _generate_patch(loop_count)
        if patch is None:
            return None
        patches.append(patch)

    grid = np.zeros((height, width), dtype=int)
    if not _place_patches(grid, patches):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    required_colors = {BLUE, RED, YELLOW, GREEN}
    if not required_colors.issubset(set(map(int, out.reshape(-1)))):
        return None
    if float(np.count_nonzero(out)) / float(out.size) > MAX_OUTPUT_DENSITY:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
