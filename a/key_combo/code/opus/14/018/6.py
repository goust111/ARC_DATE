from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

GEN_RETRY_LIMIT = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TEMPLATE_INPUTS = [
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 7],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 7, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 7, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
            [7, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 7, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_blue_objects(grid: np.ndarray) -> list[dict]:
    seen: set[tuple[int, int]] = set()
    objects: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLUE or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) == BLUE and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            objects.append({"coords": coords, "bbox": _bbox(coords)})
    return objects


def _is_complete_square(obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    size_r = r1 - r0 + 1
    size_c = c1 - c0 + 1
    if size_r != size_c or size_r < 2:
        return False
    coord_set = set(obj["coords"])
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return False
    return True


def _corner_marker_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _build_topology(grid: np.ndarray) -> dict:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    square_blocked = np.zeros((h, w), dtype=bool)
    marker_cells: list[tuple[int, int]] = []
    non_square_cells: set[tuple[int, int]] = set()
    complete_count = 0

    for obj in _find_blue_objects(grid):
        if _is_complete_square(obj):
            complete_count += 1
            for r, c in obj["coords"]:
                blocked[r, c] = True
                square_blocked[r, c] = True
            for r, c in _corner_marker_coords(obj["bbox"]):
                if _in_bounds((h, w), r, c) and int(grid[r, c]) == BLACK:
                    blocked[r, c] = True
                    marker_cells.append((r, c))
        else:
            non_square_cells.update(obj["coords"])

    return {
        "blocked": blocked,
        "square_blocked": square_blocked,
        "marker_cells": marker_cells,
        "non_square_cells": non_square_cells,
        "objects": _find_blue_objects(grid),
        "complete_count": complete_count,
    }


def _unique_shortest_path(
    blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    if bool(blocked[start]) or bool(blocked[goal]):
        return None

    h, w = blocked.shape
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=np.uint8)
    q = deque([start])
    dist[start] = 0
    ways[start] = 1

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc) or bool(blocked[nr, nc]):
                continue
            nd = dist[r, c] + 1
            if dist[nr, nc] == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                q.append((nr, nc))
            elif dist[nr, nc] == nd:
                ways[nr, nc] = min(2, int(ways[nr, nc]) + int(ways[r, c]))

    if dist[goal] == -1 or ways[goal] != 1:
        return None

    path = [goal]
    cur = goal
    while cur != start:
        preds = []
        cr, cc = cur
        for dr, dc in NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if _in_bounds((h, w), nr, nc) and dist[nr, nc] == dist[cr, cc] - 1:
                preds.append((nr, nc))
        if len(preds) != 1:
            return None
        cur = preds[0]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify blue objects by square completeness, then route the unique shortest path through the resulting maze.

    Input structure:
    - The grid contains black background, 4-connected blue foreground objects, and exactly two orange endpoint pixels.
    - Some blue objects are complete squares: their bounding boxes are NxN and every bbox-edge cell is blue.
    - Other blue objects are non-squares or broken square-like shapes and remain passable.

    Rule:
    - Detect each blue object and mark every complete square with red corner-exterior cells on in-bounds black background.
    - Treat only complete-square blue cells plus those red corner markers as blocked.
    - Find the unique 4-neighbor shortest path between the two orange endpoints through everything else, including non-square blue cells.
    - Recolor the interior of that path yellow, preserving the endpoints and all other cells.

    Key insight:
    - Geometric completeness, not color alone, decides which blue shapes become maze walls; the path must be solved on that derived topology.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, ORANGE}, "unexpected input colors"

    endpoints = np.argwhere(g == ORANGE)
    assert endpoints.shape == (2, 2), "expected exactly two orange endpoints"
    start = tuple(map(int, endpoints[0]))
    goal = tuple(map(int, endpoints[1]))

    topo = _build_topology(g)
    path = _unique_shortest_path(topo["blocked"], start, goal)
    assert path is not None, "expected exactly one shortest path under square-derived blockers"

    out = g.copy()
    for r, c in topo["marker_cells"]:
        out[r, c] = RED
    for r, c in path[1:-1]:
        out[r, c] = YELLOW
    return out


def _render_complete_square(rng: random.Random, size: int | None = None) -> np.ndarray:
    if size is None:
        size = rng.choices([2, 3, 4, 5], weights=[2, 4, 3, 2])[0]
    patch = np.zeros((size, size), dtype=int)
    if size == 2 or rng.random() < 0.45:
        patch[:, :] = BLUE
    else:
        patch[0, :] = BLUE
        patch[-1, :] = BLUE
        patch[:, 0] = BLUE
        patch[:, -1] = BLUE
        if rng.random() < 0.35 and size >= 4:
            patch[1:-1, 1:-1] = (rng.random() < 0.5) * BLUE
    return patch


def _random_non_square_patch(rng: random.Random, prefer_small: bool = False) -> np.ndarray:
    kind = rng.choice(["rect", "lshape", "broken_frame", "tshape"])
    if prefer_small and rng.random() < 0.55:
        kind = rng.choice(["rect", "lshape", "broken_frame"])

    if kind == "rect":
        h = rng.choice([2, 2, 3, 3, 4] if prefer_small else [2, 3, 4, 5])
        w_choices = [v for v in [2, 3, 4, 5] if v != h]
        w = rng.choice(w_choices)
        patch = np.full((h, w), BLUE, dtype=int)
    elif kind == "lshape":
        h = rng.choice([2, 3, 4] if prefer_small else [2, 3, 4, 5])
        w = rng.choice([2, 3, 4] if prefer_small else [2, 3, 4, 5])
        patch = np.zeros((h, w), dtype=int)
        patch[:, 0] = BLUE
        patch[0, :] = BLUE
        if h >= 3 and w >= 3 and rng.random() < 0.3:
            patch[-1, :] = BLUE
    elif kind == "broken_frame":
        size = rng.choice([3, 4] if prefer_small else [3, 4, 5])
        patch = np.zeros((size, size), dtype=int)
        patch[0, :] = BLUE
        patch[-1, :] = BLUE
        patch[:, 0] = BLUE
        patch[:, -1] = BLUE
        side = rng.choice(["top", "bottom", "left", "right"])
        idx = rng.randrange(1, size - 1)
        if side == "top":
            patch[0, idx] = BLACK
        elif side == "bottom":
            patch[-1, idx] = BLACK
        elif side == "left":
            patch[idx, 0] = BLACK
        else:
            patch[idx, -1] = BLACK
    else:
        patch = np.array(
            [
                [0, BLUE, 0],
                [BLUE, BLUE, BLUE],
                [0, BLUE, 0],
            ],
            dtype=int,
        )
        if rng.random() < 0.5:
            patch[0, 1] = BLUE
            patch[2, 1] = BLACK

    rot = rng.randrange(4)
    patch = np.rot90(patch, rot).copy()
    if rng.random() < 0.5:
        patch = np.fliplr(patch).copy()

    coords = np.argwhere(patch == BLUE)
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    patch = patch[r0 : r1 + 1, c0 : c1 + 1]
    obj = {"coords": [tuple(map(int, rc)) for rc in np.argwhere(patch == BLUE)], "bbox": (0, 0, patch.shape[0] - 1, patch.shape[1] - 1)}
    assert not _is_complete_square(obj)
    return patch


def _can_place_patch(
    grid: np.ndarray,
    forbidden: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    endpoints: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    if np.any((patch == BLUE) & (grid[r0 : r0 + h, c0 : c0 + w] != BLACK)):
        return False
    if np.any(forbidden[r0 : r0 + h, c0 : c0 + w] & (patch == BLUE)):
        return False
    if endpoints is not None:
        coords = np.argwhere(patch == BLUE)
        for er, ec in endpoints:
            for pr, pc in coords:
                if max(abs(int(pr) + r0 - er), abs(int(pc) + c0 - ec)) <= 1:
                    return False
    return True


def _stamp_patch(grid: np.ndarray, forbidden: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    h, w = patch.shape
    coords = []
    for pr, pc in np.argwhere(patch == BLUE):
        rr = int(pr) + r0
        cc = int(pc) + c0
        grid[rr, cc] = BLUE
        coords.append((rr, cc))
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    forbidden[rr0:rr1, cc0:cc1] = True
    return coords


def _eligible_endpoints(grid: np.ndarray, blocked: np.ndarray) -> list[tuple[int, int]]:
    blue_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)}
    h, w = grid.shape
    border = (
        [(0, c) for c in range(w)]
        + [(h - 1, c) for c in range(w)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    inside = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    candidates = border + inside
    usable = []
    for r, c in candidates:
        if bool(blocked[r, c]) or int(grid[r, c]) != BLACK:
            continue
        touches_blue = False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in blue_cells:
                    touches_blue = True
                    break
            if touches_blue:
                break
        if not touches_blue:
            usable.append((r, c))
    return usable


def _choose_endpoints(
    rng: random.Random, grid: np.ndarray, require_marker_impact: bool
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    topo = _build_topology(grid)
    blocked = topo["blocked"]
    no_marker_blocked = topo["square_blocked"]
    candidates = _eligible_endpoints(grid, blocked)
    if len(candidates) < 2:
        return None

    border_first = [pt for pt in candidates if pt[0] in (0, grid.shape[0] - 1) or pt[1] in (0, grid.shape[1] - 1)]
    sampled = border_first + [pt for pt in candidates if pt not in set(border_first)]
    rng.shuffle(sampled)
    sampled = sampled[: min(len(sampled), 32)]

    best: tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None = None
    for i, start in enumerate(sampled):
        for goal in sampled[i + 1 :]:
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            if manhattan < max(grid.shape) - 4:
                continue
            path = _unique_shortest_path(blocked, start, goal)
            if path is None or len(path) < max(grid.shape) - 2:
                continue
            if require_marker_impact:
                no_marker_path = _unique_shortest_path(no_marker_blocked, start, goal)
                if no_marker_path == path:
                    continue
            score = len(path) + 3 * int(start in border_first) + 3 * int(goal in border_first)
            if best is None or score > best[0]:
                best = (score, start, goal, path)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _place_complete_squares(
    rng: random.Random, h: int, w: int, count: int
) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.zeros((h, w), dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)
    for _ in range(count):
        placed = False
        for _ in range(120):
            patch = _render_complete_square(rng)
            ph, pw = patch.shape
            r0 = rng.randrange(0, h - ph + 1)
            c0 = rng.randrange(0, w - pw + 1)
            if not _can_place_patch(grid, forbidden, patch, r0, c0):
                continue
            _stamp_patch(grid, forbidden, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None
    return grid, forbidden


def _marker_target_bboxes(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    r, c = cell
    candidates = [
        (r + 1, c),
        (r, c + 1),
        (r + 1, c - size + 1),
        (r, c - size),
        (r - size, c),
        (r - size + 1, c + 1),
        (r - size, c - size + 1),
        (r - size + 1, c - size),
    ]
    dedup = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup


def _place_marker_sensitive_square(
    rng: random.Random, grid: np.ndarray, forbidden: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    provisional = _choose_endpoints(rng, grid, require_marker_impact=False)
    if provisional is None:
        return None
    _, _, base_path = provisional
    path_cells = list(base_path[3:-3])
    rng.shuffle(path_cells)

    for anchor in path_cells[:16]:
        size_order = [2, 3, 4]
        rng.shuffle(size_order)
        for size in size_order:
            patch = _render_complete_square(rng, size=size)
            for r0, c0 in _marker_target_bboxes(anchor, size):
                if not _can_place_patch(grid, forbidden, patch, r0, c0):
                    continue
                temp_grid = grid.copy()
                temp_forbidden = forbidden.copy()
                _stamp_patch(temp_grid, temp_forbidden, patch, r0, c0)
                final_choice = _choose_endpoints(rng, temp_grid, require_marker_impact=True)
                if final_choice is None:
                    continue
                _stamp_patch(grid, forbidden, patch, r0, c0)
                return final_choice
    return None


def _place_path_non_square(
    rng: random.Random,
    grid: np.ndarray,
    forbidden: np.ndarray,
    path: list[tuple[int, int]],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    if len(path) < 8:
        return False
    path_cells = set(path[3:-3])
    path_list = list(path[3:-3])
    rng.shuffle(path_list)
    candidate_patches = [
        np.array([[BLUE, BLUE], [BLUE, BLACK]], dtype=int),
        np.array([[BLUE, BLUE, BLUE], [BLUE, BLACK, BLACK]], dtype=int),
        np.array([[BLUE, BLUE], [BLUE, BLUE], [BLUE, BLACK]], dtype=int),
        np.array([[BLUE, BLUE, BLUE], [BLACK, BLUE, BLACK]], dtype=int),
    ]
    for patch in candidate_patches:
        for variant in range(8):
            transformed = np.rot90(patch, variant % 4).copy()
            if variant >= 4:
                transformed = np.fliplr(transformed).copy()
            shape_cells = [tuple(map(int, rc)) for rc in np.argwhere(transformed == BLUE)]
            for anchor in path_list[:36]:
                for pr, pc in shape_cells:
                    r0 = anchor[0] - pr
                    c0 = anchor[1] - pc
                    if not _can_place_patch(grid, forbidden, transformed, r0, c0, endpoints):
                        continue
                    placed = [(r0 + rr, c0 + cc) for rr, cc in shape_cells]
                    if not any(cell in path_cells for cell in placed):
                        continue
                    _stamp_patch(grid, forbidden, transformed, r0, c0)
                    return True
    return False


def _place_extra_non_squares(
    rng: random.Random,
    grid: np.ndarray,
    forbidden: np.ndarray,
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    count: int,
) -> bool:
    for _ in range(count):
        placed = False
        for _ in range(180):
            patch = _random_non_square_patch(rng, prefer_small=False)
            ph, pw = patch.shape
            r0 = rng.randrange(0, grid.shape[0] - ph + 1)
            c0 = rng.randrange(0, grid.shape[1] - pw + 1)
            if not _can_place_patch(grid, forbidden, patch, r0, c0, endpoints):
                continue
            _stamp_patch(grid, forbidden, patch, r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _min_inter_object_gap(objects: list[dict]) -> int:
    best = 99
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            for ar, ac in obj_a["coords"]:
                for br, bc in obj_b["coords"]:
                    best = min(best, max(abs(ar - br), abs(ac - bc)))
    return best if len(objects) >= 2 else 99


def _validate_generated_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))) <= {BLACK, BLUE, ORANGE}

    endpoints_arr = np.argwhere(grid == ORANGE)
    assert endpoints_arr.shape == (2, 2)
    start, goal = tuple(map(int, endpoints_arr[0])), tuple(map(int, endpoints_arr[1]))

    topo = _build_topology(grid)
    objects = topo["objects"]
    assert 4 <= len(objects) <= 6
    assert 1 <= topo["complete_count"] < len(objects)
    assert _min_inter_object_gap(objects) >= 2

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        assert 2 <= r1 - r0 + 1 <= 5
        assert 2 <= c1 - c0 + 1 <= 5

    path = _unique_shortest_path(topo["blocked"], start, goal)
    assert path is not None and len(path) >= max(grid.shape) - 1
    interior = path[1:-1]
    assert any((r, c) in topo["non_square_cells"] for r, c in interior)

    no_marker_path = _unique_shortest_path(topo["square_blocked"], start, goal)
    assert no_marker_path != path

    all_blue_blocked = topo["blocked"].copy()
    all_blue_blocked[grid == BLUE] = True
    all_blue_path = _unique_shortest_path(all_blue_blocked, start, goal)
    assert all_blue_path != path

    all_blue_passable = np.zeros_like(topo["blocked"], dtype=bool)
    marker_only_path = _unique_shortest_path(all_blue_passable, start, goal)
    assert marker_only_path != path

    out = main(grid)
    assert out.shape == grid.shape
    assert RED in set(map(int, np.unique(out)))
    assert YELLOW in set(map(int, np.unique(out)))
    for r, c in topo["marker_cells"]:
        assert int(out[r, c]) == RED
    for r, c in interior:
        assert int(out[r, c]) == YELLOW


def _symmetry_variants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    return [
        g.copy(),
        np.rot90(g, 1).copy(),
        np.rot90(g, 2).copy(),
        np.rot90(g, 3).copy(),
        np.fliplr(g).copy(),
        np.flipud(g).copy(),
        g.T.copy(),
        np.fliplr(np.rot90(g, 1)).copy(),
    ]


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        template = rng.choice(TEMPLATE_INPUTS)
        grid = rng.choice(_symmetry_variants(template))
        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid.copy()
    raise RuntimeError("failed to sample a valid transformed template")
