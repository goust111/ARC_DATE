from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# --------------- solver ---------------


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _loop_interior(loop_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _c in loop_cells]
    cols = [c for _r, c in loop_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    walls = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in loop_cells:
        walls[r - r0 + 1, c - c0 + 1] = True

    outside = np.zeros_like(walls, dtype=bool)
    q = deque()
    for r in range(h + 2):
        for c in (0, w + 1):
            if not bool(walls[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))
    for c in range(w + 2):
        for r in (0, h + 1):
            if not bool(walls[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))

    while q:
        cr, cc = q.popleft()
        for dr, dc in DIR4:
            nr = int(cr + dr)
            nc = int(cc + dc)
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not bool(walls[nr, nc]) and not bool(outside[nr, nc]):
                outside[nr, nc] = True
                q.append((nr, nc))

    inside: list[tuple[int, int]] = []
    for lr in range(1, h + 1):
        for lc in range(1, w + 1):
            if not bool(walls[lr, lc]) and not bool(outside[lr, lc]):
                inside.append((int(r0 + lr - 1), int(c0 + lc - 1)))
    return inside


def _clockwise_loop_order(loop_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    start = min(loop_cells)
    first_step = (int(start[0]), int(start[1] + 1))
    assert first_step in loop_cells, "top-leftmost loop cell must continue to the right"

    order = [start]
    prev = start
    curr = first_step
    while curr != start:
        order.append(curr)
        nxts = []
        for dr, dc in DIR4:
            nb = (int(curr[0] + dr), int(curr[1] + dc))
            if nb in loop_cells and nb != prev:
                nxts.append(nb)
        assert len(nxts) == 1, "loop must be a single simple cycle"
        prev, curr = curr, nxts[0]

    assert len(order) == len(loop_cells), "loop traversal must visit every track cell exactly once"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each bead ring clockwise by the pixel-count encoded by its inner indicator color.

    Input structure:
    - The background is black.
    - Each non-black ring track is a separate one-pixel-wide 4-connected simple loop with no branches.
    - Every ring encloses exactly one non-black single-pixel indicator whose color also appears on that ring.

    Rule:
    - For each ring, read the color C of its enclosed indicator pixel.
    - Count how many ring pixels have color C; call this count S.
    - Traverse the ring clockwise, starting from its topmost-leftmost pixel and first moving right, and cyclically shift all ring colors clockwise by S steps.
    - Keep all indicator pixels unchanged.

    Key insight:
    The indicator color encodes a shift amount through its total frequency on the loop, and the actual movement is a cyclic rotation along ring topology rather than a planar translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if bool((g == BG).any()) else int(Counter(map(int, g.flat)).most_common(1)[0][0])
    out = g.copy()

    loops: list[set[tuple[int, int]]] = []
    for comp in _mask_components(g != bg):
        if len(comp) < 4:
            continue
        pts = {(int(r), int(c)) for r, c in comp}
        if all(sum((int(r + dr), int(c + dc)) in pts for dr, dc in DIR4) == 2 for r, c in pts):
            loops.append(pts)

    assert loops, "expected at least one loop track"

    for loop_cells in loops:
        inside = _loop_interior(loop_cells)
        indicators = [(r, c) for r, c in inside if int(g[r, c]) != bg]
        assert len(indicators) == 1, "each loop must contain exactly one indicator"

        indicator_color = int(g[indicators[0]])
        order = _clockwise_loop_order(loop_cells)
        shift = sum(int(g[r, c]) == indicator_color for r, c in order) % len(order)
        if shift == 0:
            continue

        colors = [int(g[r, c]) for r, c in order]
        rotated = colors[-shift:] + colors[:-shift]
        for (r, c), color in zip(order, rotated):
            out[r, c] = int(color)

    return out


# --------------- generator ---------------

GEN_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_LOOP_COUNT = [1, 2, 3]
GEN_MAX_ATTEMPTS = 400
GEN_PADDING = 2
GEN_SCALE = 2


def _rasterize_loop(vertices: list[tuple[int, int]], scale: int = GEN_SCALE) -> tuple[list[tuple[int, int]], int, int]:
    order: list[tuple[int, int]] = []
    n = len(vertices)
    for i in range(n):
        r0, c0 = vertices[i]
        r1, c1 = vertices[(i + 1) % n]
        r0 *= scale
        c0 *= scale
        r1 *= scale
        c1 *= scale
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            segment = [(int(r0), int(c)) for c in range(c0, c1 + step, step)]
        else:
            step = 1 if r1 > r0 else -1
            segment = [(int(r), int(c0)) for r in range(r0, r1 + step, step)]
        if order:
            segment = segment[1:]
        order.extend(segment)
    if order[-1] == order[0]:
        order.pop()

    h = max(r for r, _c in order) + 1
    w = max(c for _r, c in order) + 1
    return order, int(h), int(w)


def _rotate_order(order: list[tuple[int, int]], height: int, width: int, turns: int) -> tuple[list[tuple[int, int]], int, int]:
    pts = list(order)
    h, w = int(height), int(width)
    for _ in range(turns % 4):
        pts = [(int(c), int(h - 1 - r)) for r, c in pts]
        h, w = w, h

    min_r = min(r for r, _c in pts)
    min_c = min(c for _r, c in pts)
    pts = [(int(r - min_r), int(c - min_c)) for r, c in pts]
    h = max(r for r, _c in pts) + 1
    w = max(c for _r, c in pts) + 1
    return pts, int(h), int(w)


def _random_partition(rng: random.Random, total: int, parts: int, min_each: int) -> list[int]:
    extra = total - parts * min_each
    assert extra >= 0
    buckets = [min_each] * parts
    for _ in range(extra):
        buckets[rng.randrange(parts)] += 1
    rng.shuffle(buckets)
    return buckets


def _rectangle_order(rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    coarse_h = rng.randint(2, 5)
    coarse_w = rng.randint(2, 6)
    vertices = [(0, 0), (0, coarse_w), (coarse_h, coarse_w), (coarse_h, 0)]
    order, height, width = _rasterize_loop(vertices)
    return _rotate_order(order, height, width, rng.randrange(4))


def _bump_order(rng: random.Random) -> tuple[list[tuple[int, int]], int, int]:
    coarse_h = rng.randint(3, 5)
    coarse_w = rng.randint(4, 6)
    bump_left = rng.randint(1, coarse_w - 2)
    bump_right = rng.randint(bump_left + 1, coarse_w - 1)
    vertices = [
        (1, 0),
        (1, bump_left),
        (0, bump_left),
        (0, bump_right),
        (1, bump_right),
        (1, coarse_w),
        (coarse_h, coarse_w),
        (coarse_h, 0),
    ]
    order, height, width = _rasterize_loop(vertices)
    return _rotate_order(order, height, width, rng.randrange(4))


def _make_loop_object(rng: random.Random) -> np.ndarray:
    family = rng.choice(["rect", "bump"])
    if family == "rect":
        order, height, width = _rectangle_order(rng)
    else:
        order, height, width = _bump_order(rng)

    shift0 = rng.randrange(len(order))
    order = order[shift0:] + order[:shift0]
    loop_cells = set(order)
    inside = _loop_interior(loop_cells)
    assert inside, "generated loop must enclose at least one interior cell"
    safe_inside = [
        (r, c)
        for r, c in inside
        if all((int(r + dr), int(c + dc)) not in loop_cells for dr, dc in DIR4)
    ]
    assert safe_inside, "indicator must remain visually separate from the ring"

    max_colors = min(4, len(order) // 2)
    color_count = rng.randint(2, max_colors)
    segment_lengths = _random_partition(rng, len(order), color_count, 2)
    colors = rng.sample(GEN_COLOR_POOL, color_count)
    indicator_color = rng.choice(colors)

    grid = np.full((height, width), BG, dtype=int)
    idx = 0
    for color, seg_len in zip(colors, segment_lengths):
        for r, c in order[idx : idx + seg_len]:
            grid[r, c] = int(color)
        idx += seg_len

    indicator = rng.choice(safe_inside)
    grid[indicator] = int(indicator_color)
    return grid


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = blocked.shape
    r0 = max(0, top - GEN_PADDING)
    c0 = max(0, left - GEN_PADDING)
    r1 = min(h, top + height + GEN_PADDING)
    c1 = min(w, left + width + GEN_PADDING)
    return not bool(blocked[r0:r1, c0:c1].any())


def _place_object(grid: np.ndarray, blocked: np.ndarray, obj: np.ndarray, top: int, left: int) -> None:
    height, width = obj.shape
    mask = obj != BG
    grid[top : top + height, left : left + width][mask] = obj[mask]

    h, w = blocked.shape
    r0 = max(0, top - GEN_PADDING)
    c0 = max(0, left - GEN_PADDING)
    r1 = min(h, top + height + GEN_PADDING)
    c1 = min(w, left + width + GEN_PADDING)
    blocked[r0:r1, c0:c1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    loop_count = rng.choice(GEN_LOOP_COUNT)
    if loop_count == 1:
        height = rng.randint(11, 21)
        width = rng.randint(11, 21)
    elif loop_count == 2:
        height = rng.randint(16, 27)
        width = rng.randint(16, 27)
    else:
        height = rng.randint(20, 30)
        width = rng.randint(20, 30)

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    objects: list[np.ndarray] = []
    for _ in range(loop_count):
        obj = _make_loop_object(rng)
        oh, ow = obj.shape
        if oh > height - 2 or ow > width - 2:
            return None
        objects.append(obj)

    rng.shuffle(objects)
    for obj in objects:
        oh, ow = obj.shape
        placed = False
        for _ in range(120):
            top = rng.randint(1, height - oh - 1)
            left = rng.randint(1, width - ow - 1)
            if not _can_place(blocked, top, left, oh, ow):
                continue
            _place_object(grid, blocked, obj, top, left)
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-shift input")
