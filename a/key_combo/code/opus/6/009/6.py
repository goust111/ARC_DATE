from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 1
RED = 2
MARKER = 6

SIDES = ("top", "bottom", "left", "right")
CORNERS_BY_SIDE = {
    "top": [(0, 0), (0, -1)],
    "bottom": [(-1, 0), (-1, -1)],
    "left": [(0, 0), (-1, 0)],
    "right": [(0, -1), (-1, -1)],
}
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OBJECT_SHAPES = [
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
]

DECOR_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_masks(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _flood(start: tuple[int, int], passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    sr, sc = start
    if not passable[sr, sc]:
        return seen
    q = deque([start])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _marker_cell(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == MARKER)
    assert pts.shape == (1, 2), "expected exactly one corner marker"
    r, c = map(int, pts[0])
    h, w = grid.shape
    assert (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return r, c


def _structural_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    wall = grid == WALL
    structural = np.zeros_like(wall, dtype=bool)
    for comp in _component_masks(wall):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            for r, c in comp:
                structural[r, c] = True
    return structural


def _corner_side_from_open_arm(structural: np.ndarray, marker: tuple[int, int]) -> str:
    h, w = structural.shape
    r, c = marker
    if (r, c) == (0, 0):
        candidates = {"top": (0, 1), "left": (1, 0)}
    elif (r, c) == (0, w - 1):
        candidates = {"top": (0, w - 2), "right": (1, w - 1)}
    elif (r, c) == (h - 1, 0):
        candidates = {"bottom": (h - 1, 1), "left": (h - 2, 0)}
    else:
        candidates = {"bottom": (h - 1, w - 2), "right": (h - 2, w - 1)}
    open_sides = [side for side, (rr, cc) in candidates.items() if not structural[rr, cc]]
    assert len(open_sides) == 1, "corner marker must have exactly one open border arm"
    return open_sides[0]


def _red_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _component_masks(grid == RED)


def _classify_objects(grid: np.ndarray, reach: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    h, w = grid.shape
    reachable: list[list[tuple[int, int]]] = []
    unreachable: list[list[tuple[int, int]]] = []
    for comp in _red_components(grid):
        is_reachable = False
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and reach[nr, nc]:
                    is_reachable = True
                    break
            if is_reachable:
                break
        if is_reachable:
            reachable.append(comp)
        else:
            unreachable.append(comp)
    return reachable, unreachable


def _erase_left_two(obj: list[tuple[int, int]]) -> list[tuple[int, int]]:
    _r0, c0, _r1, _c1 = _bbox(obj)
    return [(r, c) for r, c in obj if c >= c0 + 2]


def _sort_key(obj: list[tuple[int, int]], side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(obj)
    if side == "left":
        return (c0, r0, c1, r1)
    if side == "right":
        return (-c1, r0, c0, r1)
    if side == "top":
        return (r0, c0, r1, c1)
    return (-r1, c0, r0, c1)


def _can_shift(coords: list[tuple[int, int]], dr: int, dc: int, blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if blocked[nr, nc]:
            return False
    return True


def _compress_objects(
    objects: list[list[tuple[int, int]]],
    side: str,
    structural: np.ndarray,
    static_blocked: np.ndarray,
) -> list[list[tuple[int, int]]]:
    dr, dc = DIR_BY_SIDE[side]
    blocked = structural | static_blocked
    placed: list[list[tuple[int, int]]] = []
    for obj in sorted(objects, key=lambda o: _sort_key(o, side)):
        cur = list(obj)
        while _can_shift(cur, dr, dc, blocked):
            cur = [(r + dr, c + dc) for r, c in cur]
        for r, c in cur:
            blocked[r, c] = True
        placed.append(cur)
    return placed


def _oracle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    structural = _structural_mask(g)
    out = g.copy()
    out[(g == WALL) & (~structural)] = BG
    marker = _marker_cell(out)
    side = _corner_side_from_open_arm(structural, marker)
    passable = (out == BG) | (out == MARKER)
    reach = _flood(marker, passable)
    reachable, unreachable = _classify_objects(out, reach)

    for obj in reachable:
        for r, c in obj:
            out[r, c] = BG
        for r, c in _erase_left_two(obj):
            out[r, c] = RED

    static_blocked = (out == MARKER)
    for obj in unreachable:
        for r, c in obj:
            out[r, c] = BG
    for moved in _compress_objects(unreachable, side, structural, static_blocked):
        for r, c in moved:
            out[r, c] = RED

    out[structural] = WALL
    out[marker] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split red objects by corner-reachability, then erase reachable ones locally and compress unreachable ones toward the marked edge.

    Input structure:
    - Black background contains blue wall components, a single magenta corner marker, and several 4-connected red objects.
    - Border-touching blue components are structural walls; interior blue components are decorative fragments.
    - The corner marker has exactly one open border arm among its two adjacent edge directions, making one edge side uniquely active.

    Rule:
    - Keep only blue components that are 4-connected to the grid border; erase the other blue fragments to background.
    - Flood-fill from the magenta corner through background cells, blocked by structural blue walls and red objects.
    - For every red object adjacent to the reachable background region, erase its red pixels inside the leftmost two columns of that object's own bounding box.
    - For every other red object, translate the whole object toward the unique edge side indicated by the marker's open border arm until another step would hit a structural wall, the grid edge, or a previously compressed object.
    - Preserve the structural walls and the magenta marker.

    Key insight:
    - The same red-object type branches into two different local transformations, and the only reliable way to choose the branch is to reason globally about which background region is reachable from the corner marker after discarding decorative blue fragments.
    """
    return _oracle(grid)


def _draw_segment(mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int]) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        a, b = sorted((c0, c1))
        mask[r0, a : b + 1] = True
    else:
        a, b = sorted((r0, r1))
        mask[a : b + 1, c0] = True


def _make_structural(h: int, w: int, side: str, marker: tuple[int, int]) -> np.ndarray:
    structural = np.zeros((h, w), dtype=bool)
    if side in ("top", "bottom"):
        bend_c = random.randint(4, w - 5)
        left_r = random.randint(5, h - 7)
        delta = random.choice([-3, -2, 2, 3])
        right_r = min(h - 6, max(5, left_r + delta))
        _draw_segment(structural, (left_r, 0), (left_r, bend_c))
        _draw_segment(structural, (left_r, bend_c), (right_r, bend_c))
        _draw_segment(structural, (right_r, bend_c), (right_r, w - 1))
    else:
        bend_r = random.randint(4, h - 5)
        top_c = random.randint(5, w - 7)
        delta = random.choice([-3, -2, 2, 3])
        bottom_c = min(w - 6, max(5, top_c + delta))
        _draw_segment(structural, (0, top_c), (bend_r, top_c))
        _draw_segment(structural, (bend_r, top_c), (bend_r, bottom_c))
        _draw_segment(structural, (bend_r, bottom_c), (h - 1, bottom_c))

    mr, mc = marker
    if side == "top":
        if (mr, mc) == (0, 0):
            structural[1:3, 0] = True
        else:
            structural[1:3, w - 1] = True
    elif side == "bottom":
        if (mr, mc) == (h - 1, 0):
            structural[h - 3 : h - 1, 0] = True
        else:
            structural[h - 3 : h - 1, w - 1] = True
    elif side == "left":
        if (mr, mc) == (0, 0):
            structural[0, 1:3] = True
        else:
            structural[h - 1, 1:3] = True
    else:
        if (mr, mc) == (0, w - 1):
            structural[0, w - 3 : w - 1] = True
        else:
            structural[h - 1, w - 3 : w - 1] = True
    structural[mr, mc] = False
    return structural


def _free_mask(grid: np.ndarray, structural: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    free = np.ones((h, w), dtype=bool)
    free[0, :] = False
    free[-1, :] = False
    free[:, 0] = False
    free[:, -1] = False
    free &= ~structural
    free[marker] = False
    near_wall = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(structural):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                near_wall[nr, nc] = True
    free &= ~near_wall
    mr, mc = marker
    for rr in range(max(0, mr - 2), min(h, mr + 3)):
        for cc in range(max(0, mc - 2), min(w, mc + 3)):
            free[rr, cc] = False
    return free


def _place_shape(
    grid: np.ndarray,
    shape: list[tuple[int, int]],
    color: int,
    allowed: np.ndarray,
    occupied: np.ndarray,
    *,
    require_row_overlap: tuple[int, int] | None = None,
    require_col_overlap: tuple[int, int] | None = None,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    anchors = [(r0, c0) for r0 in range(h - max_r) for c0 in range(w - max_c)]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(not allowed[r, c] or occupied[r, c] or grid[r, c] != BG for r, c in cells):
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if require_row_overlap is not None:
            lo, hi = require_row_overlap
            if max(rows) < lo or min(rows) > hi:
                continue
        if require_col_overlap is not None:
            lo, hi = require_col_overlap
            if max(cols) < lo or min(cols) > hi:
                continue
        blocked = False
        for r, c in cells:
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if blocked:
            continue
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        return cells
    return None


def _render_from_scene(
    h: int,
    w: int,
    structural: np.ndarray,
    marker: tuple[int, int],
    red_objects: list[list[tuple[int, int]]],
    decors: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[structural] = WALL
    for comp in decors:
        for r, c in comp:
            grid[r, c] = WALL
    for comp in red_objects:
        for r, c in comp:
            grid[r, c] = RED
    grid[marker] = MARKER
    return grid


def _expected_from_scene(
    h: int,
    w: int,
    structural: np.ndarray,
    marker: tuple[int, int],
    reachable_red: list[list[tuple[int, int]]],
    unreachable_red: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    for obj in reachable_red:
        for r, c in _erase_left_two(obj):
            out[r, c] = RED
    moved = _compress_objects(unreachable_red, _corner_side_from_open_arm(structural, marker), structural, out == MARKER)
    for obj in moved:
        for r, c in obj:
            out[r, c] = RED
    out[marker] = MARKER
    return out


def _decor_allowed(grid: np.ndarray, structural: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    allowed = np.ones((h, w), dtype=bool)
    allowed[0, :] = False
    allowed[-1, :] = False
    allowed[:, 0] = False
    allowed[:, -1] = False
    allowed &= grid == BG
    for r, c in np.argwhere(structural):
        rr = int(r)
        cc = int(c)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    allowed[nr, nc] = False
    return allowed


def _validate_scene(
    inp: np.ndarray,
    expected: np.ndarray,
    structural: np.ndarray,
    marker: tuple[int, int],
    reachable_red: list[list[tuple[int, int]]],
    unreachable_red: list[list[tuple[int, int]]],
    decors: list[list[tuple[int, int]]],
) -> bool:
    h, w = inp.shape
    if marker not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
        return False
    if len(decors) < 2 or len(decors) > 4:
        return False
    if not reachable_red or not unreachable_red:
        return False
    for comp in _component_masks(inp == WALL):
        touches = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        rendered_as_struct = all(structural[r, c] for r, c in comp)
        rendered_as_decor = all(not structural[r, c] for r, c in comp)
        if touches != rendered_as_struct or (not touches and not rendered_as_decor):
            return False
    passable = (inp != WALL) & (inp != RED)
    passable[(inp == MARKER)] = True
    reach = _flood(marker, passable)
    if not np.any(reach & (inp == BG)):
        return False
    if not np.any((~reach) & (inp == BG) & (~structural)):
        return False
    for obj in reachable_red + unreachable_red:
        area = len(obj)
        r0, c0, r1, c1 = _bbox(obj)
        if not (4 <= area <= 8) or (c1 - c0 + 1) < 3:
            return False
    if not any(len(_erase_left_two(obj)) < len(obj) for obj in reachable_red):
        return False
    side = _corner_side_from_open_arm(structural, marker)
    moved = _compress_objects(unreachable_red, side, structural, np.zeros((h, w), dtype=bool))
    if not any(sorted(obj) != sorted(new_obj) for obj, new_obj in zip(sorted(unreachable_red, key=lambda o: _sort_key(o, side)), moved)):
        return False
    return np.array_equal(_oracle(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 24)
    w = random.randint(17, 24)
    side = random.choice(SIDES)
    marker_choices = []
    for rr, cc in CORNERS_BY_SIDE[side]:
        marker_choices.append((rr if rr >= 0 else h - 1, cc if cc >= 0 else w - 1))
    marker = random.choice(marker_choices)

    structural = _make_structural(h, w, side, marker)
    base = np.full((h, w), BG, dtype=int)
    base[structural] = WALL
    base[marker] = MARKER

    passable = (base == BG) | (base == MARKER)
    reach = _flood(marker, passable)
    free = _free_mask(base, structural, marker)
    reachable_allowed = free & reach
    unreachable_allowed = free & (~reach)
    if reachable_allowed.sum() < 20 or unreachable_allowed.sum() < 20:
        return None

    grid = base.copy()
    occupied = structural.copy()
    occupied[marker] = True
    reachable_red: list[list[tuple[int, int]]] = []
    unreachable_red: list[list[tuple[int, int]]] = []

    for _ in range(random.randint(1, 3)):
        shape = random.choice(OBJECT_SHAPES)
        placed = _place_shape(grid, shape, RED, reachable_allowed, occupied)
        if placed is None:
            return None
        reachable_red.append(placed)

    need_stack = random.random() < 0.6
    first_unreachable: list[tuple[int, int]] | None = None
    n_unreachable = random.randint(2 if need_stack else 1, 3)
    for i in range(n_unreachable):
        shape = random.choice(OBJECT_SHAPES)
        kwargs = {}
        if need_stack and i == 1 and first_unreachable is not None:
            r0, c0, r1, c1 = _bbox(first_unreachable)
            if side in ("left", "right"):
                kwargs["require_row_overlap"] = (r0, r1)
            else:
                kwargs["require_col_overlap"] = (c0, c1)
        placed = _place_shape(grid, shape, RED, unreachable_allowed, occupied, **kwargs)
        if placed is None:
            return None
        if first_unreachable is None:
            first_unreachable = placed
        unreachable_red.append(placed)

    decor_allowed = _decor_allowed(grid, structural)
    decor_allowed &= grid == BG
    decors: list[list[tuple[int, int]]] = []
    decor_occupied = occupied.copy()
    for _ in range(random.randint(2, 4)):
        shape = random.choice(DECOR_SHAPES)
        placed = _place_shape(grid, shape, WALL, decor_allowed, decor_occupied)
        if placed is None:
            return None
        decors.append(placed)

    inp = _render_from_scene(h, w, structural, marker, reachable_red + unreachable_red, decors)
    expected = _expected_from_scene(h, w, structural, marker, reachable_red, unreachable_red)
    if not _validate_scene(inp, expected, structural, marker, reachable_red, unreachable_red, decors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
