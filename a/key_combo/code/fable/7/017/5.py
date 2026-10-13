"""ARC task: uncoil a length-counted rope and route it around obstacles."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1                 # counterclockwise
RED = 2                  # clockwise
OBSTACLE = 5
ROPE_COLORS = (3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_generation_index = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components, using the shared ARC (row, col) convention."""
    seen = np.zeros(mask.shape, dtype=bool)
    answer: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = [(r, c)]
        comp = []
        while q:
            x, y = q.pop()
            comp.append((x, y))
            for dx, dy in DIR4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        answer.append(comp)
    return answer


def _rope_color(g: np.ndarray) -> int:
    colors = set(map(int, np.unique(g))) - {BG, BLUE, RED, OBSTACLE}
    assert len(colors) == 1, "expected one rope color besides marker and obstacle colors"
    return colors.pop()


def _anchor_and_rotation(g: np.ndarray, rope: int) -> tuple[tuple[int, int], int]:
    """Find the unique rope-colored border anchor and rotation taking it to the top."""
    h, w = g.shape
    border = []
    for r, c in np.argwhere(g == rope):
        r, c = int(r), int(c)
        if r in (0, h - 1) or c in (0, w - 1):
            border.append((r, c))
    assert len(border) == 1, "the rope must have exactly one border anchor"
    r, c = border[0]
    assert not (r in (0, h - 1) and c in (0, w - 1)), "corner anchors are ambiguous"
    if r == 0:
        k = 0
    elif c == 0:
        k = -1
    elif r == h - 1:
        k = 2
    else:
        k = 1
    return (r, c), k


def _rectangles(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects = []
    for comp in _components(g == OBSTACLE):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        box = (min(rs), min(cs), max(rs), max(cs))
        r0, c0, r1, c1 = box
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), \
            "obstacle sprites must be solid rectangles"
        rects.append(box)
    return rects


def _clockwise_ring(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Clockwise cells on the one-cell-wide outer contour of an obstacle bbox."""
    r0, c0, r1, c1 = box
    top, left, bottom, right = r0 - 1, c0 - 1, r1 + 1, c1 + 1
    ring = [(top, c) for c in range(left, right + 1)]
    ring += [(r, right) for r in range(top + 1, bottom + 1)]
    ring += [(bottom, c) for c in range(right - 1, left - 1, -1)]
    ring += [(r, left) for r in range(bottom - 1, top, -1)]
    return ring


def _route_top(g: np.ndarray, anchor_col: int, clockwise: bool) -> list[tuple[int, int]]:
    """Construct the complete in-grid route in the normalized top-edge reference frame."""
    h, w = g.shape
    rects = _rectangles(g)
    route = [(0, anchor_col)]
    r, c = route[0]
    while r + 1 < h:
        nxt = (r + 1, c)
        hit = next((b for b in rects if b[0] <= nxt[0] <= b[2] and b[1] <= c <= b[3]), None)
        if hit is None:
            route.append(nxt)
            r += 1
            continue

        ring = _clockwise_ring(hit)
        entry = (hit[0] - 1, c)
        exit_cell = (hit[2] + 1, c)
        assert (r, c) == entry, "obstacles on the route must be approached from their near face"
        assert all(0 <= x < h and 0 <= y < w for x, y in ring), \
            "each obstacle needs one cell of visible clearance"
        i, j = ring.index(entry), ring.index(exit_cell)
        step = 1 if clockwise else -1
        p = i
        detour = []
        while p != j:
            p = (p + step) % len(ring)
            detour.append(ring[p])
        assert all(int(g[x, y]) != OBSTACLE for x, y in detour)
        assert not (set(detour) & set(route)), "the rope route must not self-overlap"
        route.extend(detour)
        r, c = exit_cell
    return route


def main(grid: np.ndarray) -> np.ndarray:
    """Uncoil a rope from its border anchor, rerouting it around rectangular obstacles.

    Input structure:
    - A black background contains one rope-colored border anchor and a separate,
      irregular 4-connected rope coil of the same color.
    - One red or blue marker touches the coil, and one or more separated solid
      gray rectangular obstacle sprites cross the anchor's inward centerline.

    Rule:
    - Erase the coil and marker. Starting with the anchor as the first counted
      cell, lay as many rope pixels as the erased coil contained.
    - Travel perpendicular to the anchor's border. On meeting an obstacle, follow
      its outer one-cell contour clockwise for red or counterclockwise for blue
      until the centerline is reached beyond it, then resume straight travel.
      Stop when the rope length is exhausted; pixels that would be out of bounds
      are omitted.

    Key insight:
    - The coil contributes only its pixel count; anchor orientation, obstacle
      contours, and the marker's chirality uniquely determine the repainted path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    rope = _rope_color(g)
    _, k = _anchor_and_rotation(g, rope)
    marker_cells = np.argwhere((g == RED) | (g == BLUE))
    assert len(marker_cells) == 1, "expected exactly one chirality marker"
    marker = int(g[tuple(marker_cells[0])])

    normalized = np.rot90(g, k).copy()
    top_anchors = np.argwhere((normalized[0] == rope))
    assert len(top_anchors) == 1
    anchor_col = int(top_anchors[0, 0])

    rope_comps = _components(normalized == rope)
    anchor_comp = next(comp for comp in rope_comps if (0, anchor_col) in comp)
    assert len(anchor_comp) == 1, "anchor must be isolated before laying"
    coils = [comp for comp in rope_comps if comp is not anchor_comp]
    assert len(coils) == 1 and len(coils[0]) >= 2, "expected one nontrivial rope coil"
    length = len(coils[0])

    out = normalized.copy()
    out[out == rope] = BG
    out[(out == RED) | (out == BLUE)] = BG
    route = _route_top(normalized, anchor_col, marker == RED)
    for r, c in route[:length]:
        assert int(out[r, c]) == BG, "route must have clear space around obstacles"
        out[r, c] = rope
    return np.rot90(out, -k).copy()


def _grow_coil(eligible: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    """Grow a varied exact-size 4-connected polyomino inside eligible cells."""
    if len(eligible) < size:
        return None
    for _ in range(50):
        coil = {random.choice(tuple(eligible))}
        frontier = set()
        while len(coil) < size:
            for r, c in tuple(coil):
                for dr, dc in DIR4:
                    p = (r + dr, c + dc)
                    if p in eligible and p not in coil:
                        frontier.add(p)
            if not frontier:
                break
            # Random frontier growth makes the coil's geometry deliberately uninformative.
            p = random.choice(tuple(frontier))
            frontier.discard(p)
            coil.add(p)
        if len(coil) == size:
            return coil
    return None


def _generate_attempt(index: int) -> np.ndarray | None:
    h = random.randint(26, 30)
    w = random.randint(23, 30)
    line = random.randint(9, w - 10)
    n_obstacles = 1 + index % 3
    marker = RED if (index // 3) % 2 == 0 else BLUE
    rope = random.choice(ROPE_COLORS)
    g = np.full((h, w), BG, dtype=int)
    g[0, line] = rope

    boxes = []
    cursor = random.randint(3, 5)
    for _ in range(n_obstacles):
        oh = random.randint(2, 3)
        ow = random.randint(2, 5)
        left_part = random.randint(0, ow - 1)
        r0, c0 = cursor, line - left_part
        r1, c1 = r0 + oh - 1, c0 + ow - 1
        if c0 < 2 or c1 > w - 3 or r1 > h - 4:
            return None
        g[r0:r1 + 1, c0:c1 + 1] = OBSTACLE
        boxes.append((r0, c0, r1, c1))
        cursor = r1 + random.randint(4, 5)

    route = _route_top(g, line, marker == RED)
    last_exit = boxes[-1][2] + 1
    after_last = next((i for i, p in enumerate(route) if p == (last_exit, line)), None)
    if after_last is None:
        return None
    # Every sample uses every obstacle. Every third sample also exercises clipping.
    clipped = index % 3 == 2
    if clipped:
        length = len(route) + random.randint(2, 7)
    else:
        lo = after_last + 3
        if lo > len(route):
            return None
        length = random.randint(lo, len(route))

    forbidden = set(route)
    for r0, c0, r1, c1 in boxes:
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                forbidden.add((r, c))
    # Keep the coil on one visually separate side of the route and off the border.
    use_left = random.choice((True, False))
    cols = range(1, max(2, line - 4)) if use_left else range(min(w - 1, line + 5), w - 1)
    eligible = {(r, c) for r in range(2, h - 1) for c in cols if (r, c) not in forbidden}
    coil = _grow_coil(eligible, length)
    if coil is None:
        return None
    marker_choices = []
    for r, c in coil:
        for dr, dc in DIR4:
            p = (r + dr, c + dc)
            if p not in coil and p in eligible:
                marker_choices.append(p)
    if not marker_choices:
        return None
    marker_pos = random.choice(marker_choices)
    for r, c in coil:
        g[r, c] = rope
    g[marker_pos] = marker

    # Present all four anchor directions via rotations, preserving chirality.
    presented = np.rot90(g, random.choice((0, 1, 2, 3))).copy()
    try:
        out = main(presented)
    except (AssertionError, ValueError):
        return None

    # Semantic checks: erasure, preservation, count/clipping, both chiralities,
    # and necessity of every obstacle (compare rope masks, not obstacle pixels).
    assert np.count_nonzero(out == rope) == min(length, len(route))
    assert np.count_nonzero(out == marker) == 0 or marker == rope
    assert np.count_nonzero(out == OBSTACLE) == sum((b[2]-b[0]+1)*(b[3]-b[1]+1) for b in boxes)
    other = presented.copy()
    other[(other == RED) | (other == BLUE)] = BLUE if marker == RED else RED
    assert not np.array_equal(main(other) == rope, out == rope), "chirality must affect the route"

    # Independently ensure each obstacle is reached and materially changes the
    # rope route: removing any one rectangle must change rope-colored cells.
    base_mask = main(g) == rope
    for box in boxes:
        ring = set(_clockwise_ring(box))
        assert len(ring & set(route[:min(length, len(route))])) >= 2
        without = g.copy()
        r0, c0, r1, c1 = box
        without[r0:r1 + 1, c0:c1 + 1] = BG
        assert not np.array_equal(main(without) == rope, base_mask), \
            "every obstacle must be necessary to the laid route"
    return presented


def generate_input() -> np.ndarray:
    """Generate diverse, nondegenerate examples while cycling hard branches."""
    global _generation_index
    index = _generation_index
    _generation_index += 1
    for _ in range(300):
        grid = _generate_attempt(index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rope-routing puzzle")
