from __future__ import annotations

from collections import deque
import random

import numpy as np


FLOOR = 7
WALL = 5
MAROON = 9
EMPTY = 0
INF = 10**9
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components using the shared ARC (row, column) convention."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = [(sr, sc)]
        component = []
        for r, c in q:
            component.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        result.append(component)
    return result


def _warehouses(g: np.ndarray) -> list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]]:
    """Recognize solid black rectangles surrounded immediately by a complete grey frame."""
    h, w = g.shape
    warehouses = []
    for comp in _components(g == EMPTY):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if r0 < 1 or c0 < 1 or r1 + 1 >= h or c1 + 1 >= w:
            continue
        ring = (
            [(r0 - 1, c) for c in range(c0 - 1, c1 + 2)]
            + [(r1 + 1, c) for c in range(c0 - 1, c1 + 2)]
            + [(r, c0 - 1) for r in range(r0, r1 + 1)]
            + [(r, c1 + 1) for r in range(r0, r1 + 1)]
        )
        if all(int(g[r, c]) == WALL for r, c in ring):
            cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            warehouses.append((cells, (r0 - 1, c0 - 1, r1 + 1, c1 + 1)))
    warehouses.sort(key=lambda item: item[1])
    return warehouses


def _outside_ring(g: np.ndarray, frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Floor-side cells adjacent to a warehouse's outer wall boundary."""
    h, w = g.shape
    r0, c0, r1, c1 = frame
    candidates = (
        [(r0 - 1, c) for c in range(c0, c1 + 1)]
        + [(r1 + 1, c) for c in range(c0, c1 + 1)]
        + [(r, c0 - 1) for r in range(r0, r1 + 1)]
        + [(r, c1 + 1) for r in range(r0, r1 + 1)]
    )
    return [
        (r, c)
        for r, c in candidates
        if 0 <= r < h and 0 <= c < w and int(g[r, c]) in (FLOOR, MAROON)
    ]


def _distance_map(g: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    """Multi-source shortest paths restricted to the visible orange/maroon floor."""
    h, w = g.shape
    dist = np.full((h, w), INF, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in starts:
        if dist[r, c] != 0:
            dist[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if (
                0 <= rr < h
                and 0 <= cc < w
                and int(g[rr, cc]) in (FLOOR, MAROON)
                and dist[rr, cc] == INF
            ):
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _analyze(g: np.ndarray):
    warehouses = _warehouses(g)
    assert len(warehouses) >= 2, "expected at least two grey-framed rectangular warehouses"
    maps = [_distance_map(g, _outside_ring(g, frame)) for _, frame in warehouses]
    points = [tuple(map(int, p)) for p in np.argwhere(g == MAROON)]
    return warehouses, maps, points


def _assign(
    warehouses,
    maps: list[np.ndarray],
    points: list[tuple[int, int]],
    order: str = "distance",
):
    records = []
    for r, c in points:
        ds = tuple(int(d[r, c]) for d in maps)
        finite = [d for d in ds if d < INF]
        nearest = min(finite) if finite else INF
        records.append((nearest, r, c, ds))
    if order == "distance":
        records.sort(key=lambda x: (x[0], x[1], x[2]))
    elif order == "row-major":
        records.sort(key=lambda x: (x[1], x[2]))
    else:
        records.reverse()

    used = [0] * len(warehouses)
    assignments: dict[tuple[int, int], int] = {}
    for nearest, r, c, ds in records:
        if nearest == INF:
            continue
        choices = sorted(range(len(warehouses)), key=lambda j: ds[j])
        for j in choices:
            if ds[j] < INF and used[j] < len(warehouses[j][0]):
                used[j] += 1
                assignments[(r, c)] = j
                break
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather floor points into the nearest non-full grey-walled rectangular warehouses.

    Input structure:
    - Orange cells are traversable floor, grey cells are walls, and maroon cells are
      points standing on the floor. Each warehouse is a solid black rectangular cavity
      immediately enclosed by a complete one-cell grey frame; its black-cell count is
      its capacity. The floor-side cells touching that frame are its approach boundary.

    Rule:
    - Measure four-connected shortest-path distance through orange or maroon floor from
      every point to each warehouse's approach boundary. Process reachable points by
      increasing distance to their own nearest warehouse. Put each into its nearest
      reachable warehouse that still has capacity, filling that cavity in row-major
      order, and restore its source to orange. A point reaching no warehouse stays put.

    Key insight:
    - Capacity makes this a global distance-ordered greedy allocation: early arrivals
      fill a nearby warehouse, so a later point may overflow into its next-nearest one;
      grey-wall reachability, rather than geometric proximity, determines all distances.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))) <= {EMPTY, WALL, FLOOR, MAROON}, "unexpected color"
    warehouses, maps, points = _analyze(g)
    assignments = _assign(warehouses, maps, points)

    out = g.copy()
    counts = [0] * len(warehouses)
    for point, warehouse_index in assignments.items():
        out[point] = FLOOR
        counts[warehouse_index] += 1
    for j, count in enumerate(counts):
        for r, c in warehouses[j][0][:count]:
            out[r, c] = MAROON
    return out


def _draw_warehouse(g: np.ndarray, r: int, c: int, ih: int, iw: int) -> None:
    g[r : r + ih + 2, c : c + iw + 2] = WALL
    g[r + 1 : r + ih + 1, c + 1 : c + iw + 1] = EMPTY


def _render_expected(g: np.ndarray, assignments, warehouses) -> np.ndarray:
    """Independent renderer used by generator-side semantic validation."""
    out = g.copy()
    per_wh = [[] for _ in warehouses]
    for point, j in assignments.items():
        out[point] = FLOOR
        per_wh[j].append(point)
    for j, pts in enumerate(per_wh):
        for cell in warehouses[j][0][: len(pts)]:
            out[cell] = MAROON
    return out


def _manhattan_to_frame(point: tuple[int, int], frame: tuple[int, int, int, int]) -> int:
    r, c = point
    r0, c0, r1, c1 = frame
    dr = max(r0 - r, 0, r - r1)
    dc = max(c0 - c, 0, c - c1)
    return dr + dc


def _attempt() -> np.ndarray | None:
    h = random.randint(19, 27)
    w = random.randint(19, 29)
    g = np.full((h, w), FLOOR, dtype=int)

    # A full divider creates a genuine unreachable floor component. Its orientation,
    # isolated-side location, and opening-free extent all vary.
    vertical = random.choice((True, False))
    if vertical:
        cut = random.randint(4, 6) if random.random() < 0.5 else w - random.randint(5, 7)
        g[:, cut] = WALL
        accessible = (0, 1, h - 1, cut - 2) if cut > w // 2 else (0, cut + 2, h - 1, w - 1)
        isolated_cells = [(r, c) for r in range(1, h - 1) for c in range(w) if (c < cut) != (accessible[1] < cut)]
    else:
        cut = random.randint(4, 6) if random.random() < 0.5 else h - random.randint(5, 7)
        g[cut, :] = WALL
        accessible = (0, 1, cut - 2, w - 2) if cut > h // 2 else (cut + 2, 1, h - 1, w - 2)
        isolated_cells = [(r, c) for r in range(h) for c in range(1, w - 1) if (r < cut) != (accessible[0] < cut)]

    ar0, ac0, ar1, ac1 = accessible
    warehouse_specs = []
    target_count = random.choice((2, 2, 3))
    for _ in range(200):
        if len(warehouse_specs) >= target_count:
            break
        ih, iw = random.choice(((1, 2), (1, 3), (2, 2), (2, 3)))
        if ar1 - ar0 < ih + 5 or ac1 - ac0 < iw + 5:
            continue
        r = random.randint(ar0 + 2, ar1 - ih - 2)
        c = random.randint(ac0 + 2, ac1 - iw - 2)
        box = (r - 2, c - 2, r + ih + 3, c + iw + 3)
        if any(not (box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1]) for b in warehouse_specs):
            continue
        _draw_warehouse(g, r, c, ih, iw)
        warehouse_specs.append(box)
    if len(warehouse_specs) != target_count:
        return None

    # Interior wall segments create detours while leaving broad, legible corridors.
    for _ in range(random.randint(1, 3)):
        trial = g.copy()
        if random.random() < 0.5:
            r = random.randint(ar0 + 1, ar1 - 1)
            c0 = random.randint(ac0 + 1, max(ac0 + 1, ac1 - 7))
            trial[r, c0 : min(ac1, c0 + random.randint(4, 8)) + 1] = WALL
        else:
            c = random.randint(ac0 + 1, ac1 - 1)
            r0 = random.randint(ar0 + 1, max(ar0 + 1, ar1 - 7))
            trial[r0 : min(ar1, r0 + random.randint(4, 8)) + 1, c] = WALL
        # Never damage a warehouse and never introduce a new black interpretation.
        if len(_warehouses(trial)) == target_count:
            g = trial

    warehouses = _warehouses(g)
    maps = [_distance_map(g, _outside_ring(g, frame)) for _, frame in warehouses]
    capacities = [len(cells) for cells, _ in warehouses]

    candidates = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != FLOOR:
                continue
            ds = tuple(int(d[r, c]) for d in maps)
            if all(d < INF for d in ds) and len(set(ds)) == len(ds):
                candidates.append(((r, c), ds, int(np.argmin(ds))))
    random.shuffle(candidates)

    # Force one warehouse to receive more independent-nearest demand than capacity.
    chosen = None
    for popular in sorted(range(len(warehouses)), key=lambda j: capacities[j]):
        near = [x for x in candidates if x[2] == popular]
        need = capacities[popular] + 1
        if len(near) < need:
            continue
        for _ in range(100):
            base = random.sample(near, need)
            other = [x for x in candidates if x[2] != popular and x not in base]
            extra_n = random.randint(1, min(3, len(other))) if other else 0
            sample = base + random.sample(other, extra_n)
            if len(sample) <= sum(capacities):
                chosen = sample
                break
        if chosen is not None:
            break
    if chosen is None:
        return None

    # Every visible point-to-warehouse distance is globally distinct, as stipulated.
    values = [d for _, ds, _ in chosen for d in ds]
    if len(values) != len(set(values)):
        return None

    points = [x[0] for x in chosen]
    if len(set(points)) != len(points):
        return None
    unreachable_pool = [(r, c) for r, c in isolated_cells if 0 <= r < h and 0 <= c < w and g[r, c] == FLOOR]
    if len(unreachable_pool) < 2:
        return None
    points += random.sample(unreachable_pool, random.randint(1, min(3, len(unreachable_pool))))
    for r, c in points:
        g[r, c] = MAROON

    warehouses2, maps2, points2 = _analyze(g)
    actual = _assign(warehouses2, maps2, points2, "distance")
    if len(actual) != len(chosen):
        return None

    # Semantic anti-shortcut checks: overflow really redirects at least one point;
    # processing order matters; at least one route is longer than direct geometry.
    independent = {p: int(np.argmin(ds)) for p, ds, _ in chosen}
    if not any(actual.get(p) != independent[p] for p in independent):
        return None
    if actual == _assign(warehouses2, maps2, points2, "row-major"):
        return None
    detour = False
    for p, ds, _ in chosen:
        for j, d in enumerate(ds):
            if d > _manhattan_to_frame(p, warehouses2[j][1]):
                detour = True
    if not detour:
        return None

    expected = _render_expected(g, actual, warehouses2)
    solved = main(g)
    if not np.array_equal(solved, expected) or np.array_equal(solved, g):
        return None
    return g


def generate_input() -> np.ndarray:
    """Generate varied, semantically validated instances with overflow and distractors."""
    for _ in range(1200):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: could not construct a valid non-degenerate instance")
