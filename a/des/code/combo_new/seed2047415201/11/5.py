import random
from collections import deque

import numpy as np


AZURE = 8
GREY = 5
LEGEND_COLORS = (2, 1, 3, 4)  # red, blue, green, yellow
SIDES = ("top", "bottom", "left", "right")
MAX_RETRIES = 500


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touch_edges(coords: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> set[str]:
    edges: set[str] = set()
    for r, c in coords:
        if r == 0:
            edges.add("top")
        if r == h - 1:
            edges.add("bottom")
        if c == 0:
            edges.add("left")
        if c == w - 1:
            edges.add("right")
    return edges


def _find_legend_colors(g: np.ndarray) -> list[int]:
    swatches: list[tuple[float, int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (AZURE, GREY):
            continue
        for comp in _components_mask(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height != width or len(comp) != height * width:
                continue
            swatches.append(((r0 + r1) / 2.0, c0, r0, color))

    assert len(swatches) == 4, "expected exactly four colored legend swatches"
    swatches.sort()
    return [color for _center, _c0, _r0, color in swatches]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by how many grid edges clip it.

    Input structure:
    - The background is Azure.
    - Several Grey 4-connected foreground objects lie on the background; some
      are fully inside the grid and others touch one or more of the top,
      bottom, left, and right grid edges.
    - A small corner legend contains four colored square swatches, ordered from
      top to bottom as Red, Blue, Green, and Yellow.

    Rule:
    - Read the legend colors in top-to-bottom order.
    - For each Grey object, count how many of the four grid edges it touches.
    - Objects touching 0 edges take the first legend color, 1 edge the second,
      2 edges the third, and 3 or 4 edges the fourth.
    - The legend and background remain unchanged, and the output has the same
      shape as the input.

    Key insight:
    - The object's size and shape are irrelevant; only the set of grid edges
      touched by that connected component selects its output color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = _find_legend_colors(g)

    out = g.copy()
    for comp in _components_mask(g == GREY):
        edge_count = len(_touch_edges(comp, h, w))
        color = colors[min(edge_count, 3)]
        for r, c in comp:
            out[r, c] = color
    return out


def _dilate_cells(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    dilated: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    dilated.add((nr, nc))
    return dilated


def _can_place(coords: set[tuple[int, int]], blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w) or blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, coords: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in _dilate_cells(coords, h, w):
        blocked[r, c] = True


def _place_legend(grid: np.ndarray, blocked: np.ndarray) -> None:
    h, w = grid.shape
    size = random.choice([1, 2])
    gap = random.choice([1, 1, 2])
    stack_h = 4 * size + 3 * gap
    corner = random.choice(["TL", "TR", "BL", "BR"])
    r0 = 0 if "T" in corner else h - stack_h
    c0 = 0 if "L" in corner else w - size

    cells: set[tuple[int, int]] = set()
    for i, color in enumerate(LEGEND_COLORS):
        rr = r0 + i * (size + gap)
        grid[rr : rr + size, c0 : c0 + size] = color
        for r in range(rr, rr + size):
            for c in range(c0, c0 + size):
                cells.add((r, c))
    _mark_blocked(blocked, cells)


def _line_between(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    coords: set[tuple[int, int]] = set()
    if random.choice([False, True]):
        step = 1 if br >= ar else -1
        for r in range(ar, br + step, step):
            coords.add((r, ac))
        step = 1 if bc >= ac else -1
        for c in range(ac, bc + step, step):
            coords.add((br, c))
    else:
        step = 1 if bc >= ac else -1
        for c in range(ac, bc + step, step):
            coords.add((ar, c))
        step = 1 if br >= ar else -1
        for r in range(ar, br + step, step):
            coords.add((r, bc))
    return coords


def _anchor_for_side(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _single_side_endpoint(side: str, anchor: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    ar, ac = anchor
    length = random.randint(2, 5)
    drift = random.randint(-3, 3)
    if side == "top":
        return min(h - 3, length), min(w - 2, max(1, ac + drift))
    if side == "bottom":
        return max(2, h - 1 - length), min(w - 2, max(1, ac + drift))
    if side == "left":
        return min(h - 2, max(1, ar + drift)), min(w - 3, length)
    return min(h - 2, max(1, ar + drift)), max(2, w - 1 - length)


def _add_bumps(
    coords: set[tuple[int, int]],
    allowed_edges: set[str],
    blocked: np.ndarray,
) -> set[tuple[int, int]]:
    h, w = blocked.shape
    coords = set(coords)
    for _ in range(random.randint(0, 6)):
        candidates: list[tuple[int, int]] = []
        for r, c in coords:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in coords or blocked[nr, nc]:
                    continue
                if not _touch_edges(coords | {(nr, nc)}, h, w).issubset(allowed_edges):
                    continue
                candidates.append((nr, nc))
        if not candidates:
            break
        coords.add(random.choice(candidates))
    return coords


def _edge_object(
    edge_count: int,
    blocked: np.ndarray,
    exact_edges: set[str] | None = None,
) -> set[tuple[int, int]] | None:
    h, w = blocked.shape
    for _ in range(250):
        if exact_edges is None:
            edges = set(SIDES) if edge_count == 4 else set(random.sample(SIDES, edge_count))
        else:
            edges = set(exact_edges)

        anchors = {side: _anchor_for_side(side, h, w) for side in edges}
        if edge_count == 1:
            side = next(iter(edges))
            coords = _line_between(anchors[side], _single_side_endpoint(side, anchors[side], h, w))
        else:
            junction = (random.randint(2, h - 3), random.randint(2, w - 3))
            coords: set[tuple[int, int]] = set()
            for anchor in anchors.values():
                coords |= _line_between(anchor, junction)

        coords = _add_bumps(coords, edges, blocked)
        if _touch_edges(coords, h, w) != edges:
            continue
        if _can_place(coords, blocked):
            return coords
    return None


def _random_blob_mask() -> set[tuple[int, int]]:
    height = random.randint(2, 5)
    width = random.randint(2, 5)
    target = random.randint(3, min(9, height * width))
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < target:
        border: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    border.append((nr, nc))
        cells.add(random.choice(border))

    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _interior_object(blocked: np.ndarray) -> set[tuple[int, int]] | None:
    h, w = blocked.shape
    for _ in range(250):
        local = _random_blob_mask()
        mh = max(r for r, _ in local) + 1
        mw = max(c for _, c in local) + 1
        if mh > h - 4 or mw > w - 4:
            continue
        r0 = random.randint(2, h - mh - 2)
        c0 = random.randint(2, w - mw - 2)
        coords = {(r0 + r, c0 + c) for r, c in local}
        if _touch_edges(coords, h, w):
            continue
        if _can_place(coords, blocked):
            return coords
    return None


def _object_for_count(edge_count: int, blocked: np.ndarray) -> set[tuple[int, int]] | None:
    if edge_count == 0:
        return _interior_object(blocked)
    return _edge_object(edge_count, blocked)


def _render_expected(inp: np.ndarray) -> np.ndarray | None:
    h, w = inp.shape
    out = inp.copy()
    comps = _components_mask(inp == GREY)
    seen_counts = set()
    for comp in comps:
        edge_count = len(_touch_edges(comp, h, w))
        seen_counts.add(min(edge_count, 3))
        color = LEGEND_COLORS[min(edge_count, 3)]
        for r, c in comp:
            out[r, c] = color
    if seen_counts != {0, 1, 2, 3}:
        return None
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    inp = np.full((h, w), AZURE, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    _place_legend(inp, blocked)

    high_count = random.choice([3, 4])
    counts = [0, 1, 2, high_count]
    counts += random.choices([0, 1, 2, 3, 4], k=random.randint(0, 2))
    counts.sort(reverse=True)

    placed = 0
    for edge_count in counts:
        coords = _object_for_count(edge_count, blocked)
        if coords is None:
            return None
        if len(_touch_edges(coords, h, w)) != edge_count:
            return None
        for r, c in coords:
            inp[r, c] = GREY
        _mark_blocked(blocked, coords)
        placed += 1

    if len(_components_mask(inp == GREY)) != placed:
        return None
    expected = _render_expected(inp)
    if expected is None:
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid edge-count legend input")
