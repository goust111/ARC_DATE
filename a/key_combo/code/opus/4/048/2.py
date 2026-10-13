import heapq
import random

import numpy as np


BLACK = 0
GREEN = 3
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

GEN_RETRIES = 500
TWO_STRUCTURE_RETRIES = 400
THIRD_STRUCTURE_PROB = 0.4


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(comp, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _shell_depth(grid: np.ndarray, coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    depth = 0
    while r0 + depth <= r1 - depth and c0 + depth <= c1 - depth:
        border = _border_coords(r0 + depth, c0 + depth, r1 - depth, c1 - depth)
        if any(int(grid[r, c]) in (BLACK, GREEN) for r, c in border):
            break
        depth += 1
    return depth


def _extract_structures(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREEN)
    comps = _components(mask)
    structures = []
    for coords in comps:
        depth = _shell_depth(grid, coords)
        structures.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "depth": depth,
            }
        )
    structures.sort(key=lambda item: item["bbox"])
    return structures


def _build_forbidden(shape: tuple[int, int], structures: list[dict], radii: list[int] | None = None) -> np.ndarray:
    h, w = shape
    forbidden = np.zeros((h, w), dtype=bool)
    if radii is None:
        radii = [int(structure["depth"]) for structure in structures]
    for structure, radius in zip(structures, radii):
        radius = int(radius)
        for r, c in structure["coords"]:
            rr0 = max(0, int(r) - radius)
            rr1 = min(h - 1, int(r) + radius)
            cc0 = max(0, int(c) - radius)
            cc1 = min(w - 1, int(c) + radius)
            forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return forbidden


def _start_state(grid: np.ndarray) -> tuple[int, int, int, str, int]:
    pts = np.argwhere(grid == GREEN)
    assert pts.shape[0] == 1, "expected exactly one green start pixel"
    sr, sc = map(int, pts[0])
    h, w = grid.shape
    if sr == 0:
        return sr, sc, 2, "row", h - 1
    if sr == h - 1:
        return sr, sc, 0, "row", 0
    if sc == 0:
        return sr, sc, 1, "col", w - 1
    if sc == w - 1:
        return sr, sc, 3, "col", 0
    raise AssertionError("green start must lie on the border")


def _goal_match(kind: str, value: int, r: int, c: int) -> bool:
    return r == value if kind == "row" else c == value


def _path_search(grid: np.ndarray, radii: list[int] | None = None) -> dict | None:
    structures = _extract_structures(grid)
    if not structures:
        return None
    if radii is not None and len(radii) != len(structures):
        return None

    h, w = grid.shape
    sr, sc, init_dir, goal_kind, goal_value = _start_state(grid)
    blocked = ((grid != BLACK) & (grid != GREEN)) | _build_forbidden(grid.shape, structures, radii)
    blocked[sr, sc] = False

    start = (sr, sc, init_dir)
    best = {start: (0, 0)}
    count = {start: 1}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int], int, int, int]] = [((0, 0), sr, sc, init_dir)]

    while heap:
        cost, r, c, dprev = heapq.heappop(heap)
        if best.get((r, c, dprev)) != cost:
            continue
        for nd, (dr, dc) in enumerate(DIRS):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if blocked[rr, cc]:
                continue
            ncost = (cost[0] + int(nd != dprev), cost[1] + 1)
            key = (rr, cc, nd)
            old = best.get(key)
            if old is None or ncost < old:
                best[key] = ncost
                count[key] = count[(r, c, dprev)]
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))
            elif ncost == old:
                count[key] = min(2, count[key] + count[(r, c, dprev)])

    goal_states = []
    for state, cost in best.items():
        r, c, _ = state
        if (r, c) == (sr, sc):
            continue
        if _goal_match(goal_kind, goal_value, r, c):
            goal_states.append((state, cost, count[state]))
    if not goal_states:
        return None

    best_cost = min(cost for _state, cost, _cnt in goal_states)
    total_count = min(2, sum(cnt for _state, cost, cnt in goal_states if cost == best_cost))
    end_state = min(state for state, cost, _cnt in goal_states if cost == best_cost)

    path: list[tuple[int, int]] = []
    cur = end_state
    while True:
        path.append((cur[0], cur[1]))
        if cur == start:
            break
        cur = prev[cur]
    path.reverse()

    return {
        "structures": structures,
        "radii": [int(structure["depth"]) for structure in structures] if radii is None else [int(x) for x in radii],
        "cost": best_cost,
        "count": total_count,
        "path": path,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the unique lowest-turn green path that stays outside each concentric ring's depth-derived exclusion zone.

    Input structure:
    - Black background with 2 or 3 separated concentric rectangular ring structures.
    - Each structure is a single 4-connected multicolor object made of 1 to 3 nested one-pixel-thick rectangular shells.
    - A single green pixel lies on one border edge and marks the path start toward the opposite edge.

    Rule:
    - For each ring structure, count how many nested shells it contains; that shell count N is the structure's exclusion radius.
    - A legal path uses only horizontal and vertical steps on black cells, never overwrites ring pixels, and every path cell must have Chebyshev distance strictly greater than N from every pixel of that structure.
    - Among all legal border-to-opposite-border paths, choose the one with the fewest turns; if several remain, choose the shortest by path length.
    - Keep all ring structures unchanged and paint the chosen path green.

    Key insight:
    - The concentric depth is not decorative: each structure's shell count directly becomes its keep-out radius, so the solver must infer nested depth before pathfinding.
    """
    g = np.asarray(grid, dtype=int)
    result = _path_search(g)
    assert result is not None, "expected a solvable grid with at least one ring structure"

    out = g.copy()
    for r, c in result["path"]:
        if out[r, c] == BLACK:
            out[r, c] = GREEN
    return out


def _draw_ring_structure(grid: np.ndarray, r0: int, c0: int, h: int, w: int, depth: int) -> None:
    colors = random.sample(PALETTE, depth)
    for offset, color in enumerate(colors):
        rt = r0 + offset
        ct = c0 + offset
        rb = r0 + h - 1 - offset
        cb = c0 + w - 1 - offset
        grid[rt, ct : cb + 1] = color
        grid[rb, ct : cb + 1] = color
        grid[rt : rb + 1, ct] = color
        grid[rt : rb + 1, cb] = color


def _semantic_ok(grid: np.ndarray, expected_count: int) -> dict | None:
    structures = _extract_structures(grid)
    if len(structures) != expected_count:
        return None
    depths = [int(structure["depth"]) for structure in structures]
    if any(depth < 1 or depth > 3 for depth in depths):
        return None
    if len(set(depths)) < 2:
        return None

    result = _path_search(grid)
    if result is None or result["count"] != 1:
        return None
    turns, steps = result["cost"]
    if turns < 2 or steps <= turns:
        return None

    true_path = tuple(result["path"])
    for idx in range(expected_count):
        altered = list(depths)
        altered[idx] = max(0, altered[idx] - 1)
        other = _path_search(grid, altered)
        if other is None:
            return None
        if other["cost"] == result["cost"] and tuple(other["path"]) == true_path:
            return None
    return result


def _generate_two_structure_candidate() -> np.ndarray | None:
    for _ in range(TWO_STRUCTURE_RETRIES):
        h = random.randint(22, 30)
        w = random.randint(22, 30)
        grid = np.zeros((h, w), dtype=int)

        depths = random.sample([1, 2, 3], 2)
        side0 = random.randint(0, 1)
        next_c = 1
        placed: list[tuple[int, int, int, int, int]] = []
        ok = True

        for i, depth in enumerate(depths):
            box_h = 2 * depth + 2 + random.randint(0, 2)
            box_w = 2 * depth + 4 + random.randint(0, 4)
            side = (side0 + i) % 2
            if side == 0:
                r0 = depth + random.randint(0, 1)
            else:
                r0 = h - box_h - depth - random.randint(0, 1)
            c0 = next_c + random.randint(0, 1)
            if placed and c0 <= placed[-1][1] + placed[-1][3] - 1:
                c0 = placed[-1][1] + placed[-1][3] + 1
            if c0 + box_w > w - 2:
                ok = False
                break
            placed.append((r0, c0, box_h, box_w, depth))
            next_c = c0 + box_w + random.randint(1, 3)

        if not ok:
            continue

        for r0, c0, box_h, box_w, depth in placed:
            _draw_ring_structure(grid, r0, c0, box_h, box_w, depth)

        rows = list(range(h))
        random.shuffle(rows)
        for start_r in rows:
            if grid[start_r, 0] != BLACK:
                continue
            candidate = grid.copy()
            candidate[start_r, 0] = GREEN
            if _semantic_ok(candidate, 2) is not None:
                return candidate
    return None


def _margin_is_black(grid: np.ndarray, r0: int, c0: int, box_h: int, box_w: int) -> bool:
    h, w = grid.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + box_h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + box_w + 1)
    patch = grid[rr0:rr1, cc0:cc1]
    return np.all((patch == BLACK) | (patch == GREEN))


def _try_add_third_structure(grid: np.ndarray) -> np.ndarray | None:
    base = _semantic_ok(grid, 2)
    if base is None:
        return None

    path = base["path"]
    h, w = grid.shape
    row_pool = sorted({r for r, _c in path if 1 <= r <= h - 7})
    col_pool = sorted({c for _r, c in path if 1 <= c <= w - 9})
    if not row_pool or not col_pool:
        return None

    positions = [(r, c) for r in row_pool for c in col_pool]
    random.shuffle(positions)

    for depth in random.sample([1, 2], 2):
        for r0, c0 in positions[:160]:
            box_h = 2 * depth + 2 + random.randint(0, 1)
            box_w = 2 * depth + 4 + random.randint(0, 2)
            if r0 + box_h > h - 1 or c0 + box_w > w - 1:
                continue
            if not _margin_is_black(grid, r0, c0, box_h, box_w):
                continue
            candidate = grid.copy()
            _draw_ring_structure(candidate, r0, c0, box_h, box_w, depth)
            if _semantic_ok(candidate, 3) is not None:
                return candidate
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        base = _generate_two_structure_candidate()
        if base is None:
            continue

        candidate = base
        if random.random() < THIRD_STRUCTURE_PROB:
            third = _try_add_third_structure(base)
            if third is not None:
                candidate = third

        expected_count = len(_extract_structures(candidate))
        rot = np.rot90(candidate, k=random.randint(0, 3))
        if _semantic_ok(rot, expected_count) is not None:
            return rot
        if _semantic_ok(candidate, expected_count) is not None:
            return candidate
    raise RuntimeError("failed to generate a valid nested-ring pathfinding input")
