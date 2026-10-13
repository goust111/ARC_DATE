from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
COLOR_POOL = (1, 3, 4, 5, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_rings(g: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(g) if int(c) not in (BLACK, RED)]
    assert len(colors) >= 2, "expected at least two colored rings"

    rings: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        assert box[2] - box[0] >= 2 and box[3] - box[1] >= 2, "rings must enclose a region"

        border = _border_cells(box)
        actual = set((int(r), int(c)) for r, c in coords)
        missing = border - actual
        assert not (actual - border), "ring color must lie only on its rectangle border"
        assert len(missing) == 1, "each ring must have exactly one one-cell door"
        gap = next(iter(missing))
        assert int(g[gap]) == BLACK, "ring door must be a background cell"
        rings.append({"color": color, "bbox": box, "gap": gap, "area": len(border)})

    rings.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0] + 1)
                                * (item["bbox"][3] - item["bbox"][1] + 1),
               reverse=True)

    for outer, inner in zip(rings, rings[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1, "rings must be nested"
        assert or0 + or1 == ir0 + ir1 and oc0 + oc1 == ic0 + ic1, "rings must be concentric"
    return rings


def _cell_depth(r: int, c: int, rings: list[dict]) -> int:
    depth = 0
    for ring in rings:
        r0, c0, r1, c1 = ring["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            depth += 1
    return depth


def _unique_shortest_path(
    wall_mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = wall_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=int)
    dist[start] = 0
    ways[start] = 1
    q: deque[tuple[int, int]] = deque([start])

    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc) or bool(wall_mask[nr, nc]):
                continue
            if int(dist[nr, nc]) == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = int(ways[r, c])
                q.append((nr, nc))
            elif int(dist[nr, nc]) == nd:
                ways[nr, nc] = min(2, int(ways[nr, nc]) + int(ways[r, c]))

    assert int(dist[goal]) >= 0, "expected connected red endpoints"
    assert int(ways[goal]) == 1, "expected a unique shortest path"

    path = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        predecessors = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if _in_bounds((h, w), nr, nc)
            and not bool(wall_mask[nr, nc])
            and int(dist[nr, nc]) == int(dist[r, c]) - 1
        ]
        assert len(predecessors) == 1, "shortest path reconstruction is ambiguous"
        cur = predecessors[0]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique red-to-red path through nested ring doors and recolor it by mirrored ring depth.

    Input structure:
    - A black grid contains several one-cell-thick concentric rectangular rings with distinct colors.
    - Every ring has exactly one black door cell on its border, and the doors are offset from layer to layer.
    - Exactly two red endpoint pixels are present: one in the outer annulus and one inside the innermost ring.

    Rule:
    - Keep every colored ring cell and both red endpoint pixels unchanged.
    - Treat ring cells as walls and black door/background cells as passable, then find the unique shortest
      4-neighbor path between the two red endpoints.
    - A path cell has depth d when it lies within d nested ring bounding boxes, so the outer annulus has
      depth 1 and the innermost region has depth n.
    - Recolor every non-red cell on the path with the color of ring n+1-d, mirroring the depth order.

    Key insight:
    - The maze route is determined by the single doors, but its colors are read from the opposite depth:
      outer path segments take inner ring colors and inner path segments take outer ring colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rings = _find_rings(g)
    ring_colors = [int(ring["color"]) for ring in rings]
    red_coords = np.argwhere(g == RED)
    assert red_coords.shape == (2, 2), "expected exactly two red endpoints"
    endpoints = [tuple(map(int, rc)) for rc in red_coords]

    depths = sorted(_cell_depth(r, c, rings) for r, c in endpoints)
    assert depths == [1, len(rings)], "red endpoints must mark the outer annulus and innermost region"

    wall_mask = np.isin(g, ring_colors)
    path = _unique_shortest_path(wall_mask, endpoints[0], endpoints[1])

    out = g.copy()
    for r, c in path[1:-1]:
        assert int(g[r, c]) == BLACK, "path interior must pass only through background and doors"
        depth = _cell_depth(r, c, rings)
        assert 1 <= depth <= len(rings), "path must stay inside the nested maze"
        out[r, c] = ring_colors[len(rings) - depth]
    return out


# --------------- generator ---------------

GEN_MAX_ATTEMPTS = 600
LAYER_CHOICES = (4, 5, 6)
MARGIN_RANGE = (1, 4)
SIDE_NAMES = ("top", "right", "bottom", "left")
MIN_PATH_TURNS = 5
MIN_PAINTED_DEPTHS = 3


def _side_gap(box: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    return random.randint(r0 + 1, r1 - 1), c1


def _draw_ring(g: np.ndarray, box: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = box
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color
    g[gap] = BLACK


def _ring_boxes(top: int, left: int, n: int) -> list[tuple[int, int, int, int]]:
    side = 4 * n - 1
    return [
        (top + 2 * i, left + 2 * i, top + side - 1 - 2 * i, left + side - 1 - 2 * i)
        for i in range(n)
    ]


def _cycle_cells_just_inside(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        cells.append((r0 + 1, c))
    for r in range(r0 + 2, r1):
        cells.append((r, c1 - 1))
    for c in range(c1 - 2, c0, -1):
        cells.append((r1 - 1, c))
    for r in range(r1 - 2, r0 + 1, -1):
        cells.append((r, c0 + 1))
    return cells


def _turn_count(path: list[tuple[int, int]]) -> int:
    return sum(
        (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])
        for a, b, c in zip(path, path[1:], path[2:])
    )


def _painted_depths_for_path(path: list[tuple[int, int]], rings: list[dict]) -> set[int]:
    return {_cell_depth(r, c, rings) for r, c in path[1:-1]}


def _has_offset_doors(rings: list[dict]) -> bool:
    gaps = [ring["gap"] for ring in rings]
    for a, b in zip(gaps, gaps[1:]):
        if a[0] == b[0] or a[1] == b[1]:
            return False
    return True


def _candidate_ok(g: np.ndarray) -> bool:
    try:
        rings = _find_rings(g)
        red_coords = np.argwhere(g == RED)
        if red_coords.shape != (2, 2):
            return False
        endpoints = [tuple(map(int, rc)) for rc in red_coords]
        depths = sorted(_cell_depth(r, c, rings) for r, c in endpoints)
        if depths != [1, len(rings)]:
            return False
        if not _has_offset_doors(rings):
            return False

        wall_mask = np.isin(g, [ring["color"] for ring in rings])
        path = _unique_shortest_path(wall_mask, endpoints[0], endpoints[1])
        painted_depths = _painted_depths_for_path(path, rings)
        if len(painted_depths) < MIN_PAINTED_DEPTHS or painted_depths != set(range(1, len(rings) + 1)):
            return False
        if _turn_count(path) < MIN_PATH_TURNS:
            return False
        if len(path) <= 2 * len(rings) + 4:
            return False

        out = main(g)
        changed = np.argwhere(out != g)
        if changed.shape[0] != len(path) - 2:
            return False
        for r, c in changed:
            depth = _cell_depth(int(r), int(c), rings)
            expected = int(rings[len(rings) - depth]["color"])
            if int(out[r, c]) != expected:
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(LAYER_CHOICES)
    side = 4 * n - 1
    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    h = top + side + bottom_margin
    w = left + side + right_margin
    if max(h, w) > 30:
        return None

    colors = random.sample(COLOR_POOL, n)
    boxes = _ring_boxes(top, left, n)
    sides: list[str] = []
    last_side: str | None = None
    for _ in range(n):
        choices = [s for s in SIDE_NAMES if s != last_side]
        side_name = random.choice(choices)
        sides.append(side_name)
        last_side = side_name

    g = np.zeros((h, w), dtype=int)
    gaps: list[tuple[int, int]] = []
    for box, color, side_name in zip(boxes, colors, sides):
        gap = _side_gap(box, side_name)
        gaps.append(gap)
        _draw_ring(g, box, color, gap)

    outer_cycle = [cell for cell in _cycle_cells_just_inside(boxes[0]) if cell not in gaps]
    random.shuffle(outer_cycle)
    goal_box = boxes[-1]
    goal = (goal_box[0] + 1, goal_box[1] + 1)

    for start in outer_cycle[:20]:
        trial = g.copy()
        if trial[start] != BLACK or trial[goal] != BLACK:
            continue
        trial[start] = RED
        trial[goal] = RED
        if _candidate_ok(trial):
            return trial
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring mirrored-depth maze")
