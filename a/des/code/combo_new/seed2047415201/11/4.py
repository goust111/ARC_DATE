from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
COLOR_POOL = (1, 3, 4, 5, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 500
ATTACH_RETRIES = 350
MIN_MARGIN = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        r0, c0 = int(sr), int(sc)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        q: deque[tuple[int, int]] = deque([(r0, c0)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for rr, cc in _neighbors4(r, c, h, w):
                if mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def _has_no_holes(mask: np.ndarray) -> bool:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, h, w):
            if not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    return not bool((~mask & ~outside).any())


def _articulation_points(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    index = {cell: i for i, cell in enumerate(coords)}
    n = len(coords)
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    is_cut = [False] * n
    time = 0

    def dfs(u: int) -> None:
        nonlocal time
        disc[u] = time
        low[u] = time
        time += 1
        child_count = 0
        r, c = coords[u]

        for rr, cc in _neighbors4(r, c, mask.shape[0], mask.shape[1]):
            v = index.get((rr, cc))
            if v is None:
                continue
            if disc[v] == -1:
                parent[v] = u
                child_count += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1:
                    if child_count > 1:
                        is_cut[u] = True
                elif low[v] >= disc[u]:
                    is_cut[u] = True
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if disc[i] == -1:
            dfs(i)

    return [coords[i] for i, cut in enumerate(is_cut) if cut]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every bridge pixel of the single foreground object in red.

    Input structure:
    - The grid has a black background and exactly one non-black foreground color that is not red.
    - All foreground pixels form one large 4-connected monochrome object.
    - The object has no enclosed black holes, but it may contain one-pixel-wide bottlenecks.

    Rule:
    - For each foreground pixel, consider temporarily removing only that pixel.
    - If the remaining foreground would split into more 4-connected components, recolor that pixel red.
    - Non-bridge foreground pixels keep their original color, and black background pixels stay black.

    Key insight:
    - Bridge pixels are the cut vertices of the foreground object's 4-neighbor graph, usually visible as narrow necks between wider parts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    foreground_colors = [int(v) for v in np.unique(g) if int(v) != BLACK]
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    fg = foreground_colors[0]
    assert fg != RED, "red is reserved for output bridge markers"

    mask = g == fg
    comps = _components(mask)
    assert len(comps) == 1, "foreground pixels must form one 4-connected object"
    assert len(comps[0]) > 1, "foreground object must have at least two pixels"
    assert _has_no_holes(mask), "foreground object must not enclose black holes"

    bridges = _articulation_points(mask)
    assert bridges, "expected at least one bridge pixel"

    out = g.copy()
    for r, c in bridges:
        out[r, c] = RED
    return out


def _rect_cells(r0: int, c0: int, height: int, width: int) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r in range(r0, r0 + height)
        for c in range(c0, c0 + width)
    ]


def _in_inner_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(
        MIN_MARGIN <= r < h - MIN_MARGIN and MIN_MARGIN <= c < w - MIN_MARGIN
        for r, c in cells
    )


def _old_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    return [(rr, cc) for rr, cc in _neighbors4(r, c, mask.shape[0], mask.shape[1]) if mask[rr, cc]]


def _proposal_is_valid(
    mask: np.ndarray,
    proposed: list[tuple[int, int]],
    first: tuple[int, int],
    door: tuple[int, int],
) -> bool:
    h, w = mask.shape
    proposed_set = set(proposed)
    if len(proposed_set) != len(proposed):
        return False
    if not _in_inner_bounds(proposed, h, w):
        return False
    if any(mask[r, c] for r, c in proposed):
        return False

    for r, c in proposed:
        old = _old_neighbors(mask, r, c)
        if (r, c) == first:
            if old != [door]:
                return False
        elif old:
            return False
    return True


def _attachment_cells(
    door: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    room_h: int,
    room_w: int,
    offset: int,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    r, c = door
    dr, dc = direction
    corridor = [(r + dr * i, c + dc * i) for i in range(1, length + 1)]

    if direction == (0, 1):
        room_r0 = r - offset
        room_c0 = c + length + 1
    elif direction == (0, -1):
        room_r0 = r - offset
        room_c0 = c - length - room_w
    elif direction == (1, 0):
        room_r0 = r + length + 1
        room_c0 = c - offset
    else:
        room_r0 = r - length - room_h
        room_c0 = c - offset

    room = _rect_cells(room_r0, room_c0, room_h, room_w)
    return corridor + room, corridor[0]


def _attach_room(mask: np.ndarray) -> bool:
    occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not occupied:
        return False

    for _ in range(ATTACH_RETRIES):
        door = random.choice(occupied)
        direction = random.choice(DIR4)
        room_h = random.randint(3, 6)
        room_w = random.randint(3, 6)
        length = random.randint(2, 7)
        if direction in ((0, 1), (0, -1)):
            offset = random.randint(1, room_h - 2)
        else:
            offset = random.randint(1, room_w - 2)

        proposed, first = _attachment_cells(door, direction, length, room_h, room_w, offset)
        if not _proposal_is_valid(mask, proposed, first, door):
            continue
        for r, c in proposed:
            mask[r, c] = True
        return True
    return False


def _place_initial_room(mask: np.ndarray) -> bool:
    h, w = mask.shape
    room_h = random.randint(4, min(7, h - 4))
    room_w = random.randint(4, min(7, w - 4))
    r0 = random.randint(2, h - room_h - 2)
    c0 = random.randint(2, w - room_w - 2)
    mask[r0 : r0 + room_h, c0 : c0 + room_w] = True
    return True


def _render(mask: np.ndarray, color: int) -> np.ndarray:
    grid = np.full(mask.shape, BLACK, dtype=int)
    grid[mask] = int(color)
    return grid


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    mask = grid != BLACK
    bridge_count = int(np.count_nonzero(out == RED))
    fg_count = int(mask.sum())
    return (
        max(grid.shape) <= 30
        and len(_components(mask)) == 1
        and _has_no_holes(mask)
        and 2 <= bridge_count <= fg_count - 8
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 30)
    width = random.randint(13, 30)
    mask = np.zeros((height, width), dtype=bool)
    _place_initial_room(mask)

    target_rooms = random.randint(2, 5)
    for _ in range(target_rooms - 1):
        if not _attach_room(mask):
            return None

    if random.random() < 0.45:
        _attach_room(mask)

    if not _has_no_holes(mask):
        return None
    if len(_components(mask)) != 1:
        return None

    color = random.choice(COLOR_POOL)
    grid = _render(mask, color)
    if _valid_generated(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
