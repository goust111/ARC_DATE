from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR_VECS = {
    BLUE: (-1, 0),   # up
    RED: (1, 0),     # down
    GREEN: (0, -1),  # left
    YELLOW: (0, 1),  # right
}
DIR_ORDER = [BLUE, RED, GREEN, YELLOW]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _sorted_component_coords(coords: np.ndarray) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords.tolist())


def _sort_key(coords: np.ndarray, direction: int) -> tuple[int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == BLUE:
        return int(rows.min()), int(cols.min())
    if direction == RED:
        return -int(rows.max()), int(cols.min())
    if direction == GREEN:
        return int(cols.min()), int(rows.min())
    if direction == YELLOW:
        return -int(cols.max()), int(rows.min())
    raise ValueError(f"unsupported direction color: {direction}")


def _parse_input(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _components_of_color(g, GREY)
    parsed = []
    markers = {(int(r), int(c)): int(g[r, c]) for r, c in np.argwhere((g != BLACK) & (g != GREY))}

    for idx, coords in enumerate(comps):
        touching = {}
        coord_set = {(int(r), int(c)) for r, c in coords.tolist()}
        for r, c in coord_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in markers:
                    touching[(nr, nc)] = markers[(nr, nc)]
        if len(touching) != 1:
            raise ValueError("each grey component must touch exactly one marker pixel")
        (mr, mc), color = next(iter(touching.items()))
        if color not in DIR_VECS:
            raise ValueError("marker color must encode one of the four directions")
        parsed.append(
            {
                "id": idx,
                "marker": (mr, mc),
                "color": int(color),
                "coords": np.array(_sorted_component_coords(coords), dtype=int),
            }
        )

    for (mr, mc), color in markers.items():
        if color not in DIR_VECS:
            raise ValueError("unexpected foreground color")
        adjacent_components = 0
        for nr, nc in neighbors4(mr, mc):
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == GREY:
                adjacent_components += 1
        if adjacent_components == 0:
            raise ValueError("marker must label a component")
    return parsed


def _recolor_and_erase(grid: np.ndarray, comps: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    moved = []
    for comp in comps:
        color = int(comp["color"])
        marker_r, marker_c = comp["marker"]
        out[marker_r, marker_c] = BLACK
        cells = np.array(comp["coords"], dtype=int)
        out[cells[:, 0], cells[:, 1]] = color
        moved.append({"id": int(comp["id"]), "color": color, "coords": cells.copy()})
    return out, moved


def _simulate_colored(board: np.ndarray, comps: list[dict]) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(board, dtype=int).copy()
    h, w = g.shape
    owner = -np.ones((h, w), dtype=int)
    states = [{"id": int(comp["id"]), "color": int(comp["color"]), "coords": np.array(comp["coords"], dtype=int)} for comp in comps]
    for comp in states:
        for r, c in comp["coords"]:
            owner[int(r), int(c)] = int(comp["id"])

    ordered = sorted(states, key=lambda comp: (DIR_ORDER.index(int(comp["color"])), _sort_key(comp["coords"], int(comp["color"]))))
    traces = []

    for comp in ordered:
        color = int(comp["color"])
        dr, dc = DIR_VECS[color]
        cells = {(int(r), int(c)) for r, c in comp["coords"].tolist()}
        steps = 0
        block_reason = "edge"
        blockers: set[int] = set()

        while True:
            nxt = [(r + dr, c + dc) for r, c in cells]
            if any(not (0 <= nr < h and 0 <= nc < w) for nr, nc in nxt):
                block_reason = "edge"
                break
            occupied_by = {int(owner[nr, nc]) for nr, nc in nxt if owner[nr, nc] != -1 and (nr, nc) not in cells}
            if occupied_by:
                block_reason = "component"
                blockers = occupied_by
                break

            for r, c in cells:
                g[r, c] = BLACK
                owner[r, c] = -1
            cells = {(int(nr), int(nc)) for nr, nc in nxt}
            for r, c in cells:
                g[r, c] = color
                owner[r, c] = int(comp["id"])
            steps += 1

        new_coords = np.array(sorted(cells), dtype=int)
        comp["coords"] = new_coords
        traces.append(
            {
                "id": int(comp["id"]),
                "color": color,
                "steps": int(steps),
                "block_reason": block_reason,
                "blockers": set(blockers),
                "coords": new_coords.copy(),
            }
        )

    return g, traces


def _boundary_only(board_shape: tuple[int, int], comps: list[dict]) -> np.ndarray | None:
    h, w = board_shape
    out = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    for comp in comps:
        color = int(comp["color"])
        dr, dc = DIR_VECS[color]
        cells = {(int(r), int(c)) for r, c in comp["coords"].tolist()}
        while True:
            nxt = [(r + dr, c + dc) for r, c in cells]
            if any(not (0 <= nr < h and 0 <= nc < w) for nr, nc in nxt):
                break
            cells = {(int(nr), int(nc)) for nr, nc in nxt}
        if occupied & cells:
            return None
        occupied |= cells
        for r, c in cells:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer each marker color to its adjacent grey component, then slide each recolored component in the direction encoded by that color until blocked.

    Input structure:
    - Black background with 4-connected grey components and isolated non-black marker pixels.
    - Each grey component touches exactly one marker pixel by 4-neighbor adjacency.
    - Marker colors are blue, red, green, or yellow, and each marker is erased after labeling its component.

    Rule:
    - Find every grey component and its unique adjacent marker pixel.
    - Recolor the entire grey component with the marker color, then erase the marker to black.
    - Interpret blue as up, red as down, green as left, and yellow as right.
    - Slide components one cell at a time in the fixed processing order: all up components, then down, then left, then right; within each direction, process the component already closest to that target edge first.
    - A component stops as soon as any next-step cell would leave the grid or hit a non-black cell belonging to another component.

    Key insight:
    - The marker color is both a label and a movement instruction, and the final arrangement depends on sequential multi-component blocking rather than independent sliding.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    parsed = _parse_input(g)
    recolored, comps = _recolor_and_erase(g, parsed)
    final_grid, _trace = _simulate_colored(recolored, comps)
    return final_grid


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return np.array(norm, dtype=int)


def _random_shape(size: int | None = None) -> np.ndarray:
    target = int(size if size is not None else random.randint(3, 7))
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        shape = _normalize_shape(cells)
        h = int(shape[:, 0].max()) + 1
        w = int(shape[:, 1].max()) + 1
        if h > 4 or w > 4:
            continue
        return shape
    raise RuntimeError("failed to build shape")


def _shape_rows(shape: np.ndarray) -> list[int]:
    return sorted({int(r) for r, _ in shape.tolist()})


def _row_min_col(shape: np.ndarray, row: int) -> int:
    return min(int(c) for r, c in shape.tolist() if int(r) == int(row))


def _row_max_col(shape: np.ndarray, row: int) -> int:
    return max(int(c) for r, c in shape.tolist() if int(r) == int(row))


def _place(shape: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.array([(int(r0) + int(r), int(c0) + int(c)) for r, c in shape.tolist()], dtype=int)


def _fits(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < h) and np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < w))


def _min_component_gap(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for r1, c1 in a.tolist():
        for r2, c2 in b.tolist():
            best = min(best, abs(int(r1) - int(r2)) + abs(int(c1) - int(c2)))
    return int(best)


def _marker_candidates(coords: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    cells = {(int(r), int(c)) for r, c in coords.tolist()}
    candidates = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                candidates.add((nr, nc))
    return sorted(candidates)


def _chebyshev_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _build_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    n = random.randint(4, 6)

    base_shapes = {
        BLUE: _random_shape(),
        RED: _random_shape(),
        GREEN: _random_shape(),
        YELLOW: _random_shape(),
    }
    extras = [random.choice(DIR_ORDER) for _ in range(n - 4)]

    shapes: list[tuple[int, np.ndarray]] = [(d, base_shapes[d]) for d in DIR_ORDER] + [(d, base_shapes[d]) for d in extras]
    comps: list[dict] = []

    shape_u = base_shapes[BLUE]
    shape_d = base_shapes[RED]
    shape_l = base_shapes[GREEN]
    shape_r = base_shapes[YELLOW]
    h_u = int(shape_u[:, 0].max()) + 1
    w_u = int(shape_u[:, 1].max()) + 1
    h_d = int(shape_d[:, 0].max()) + 1
    w_d = int(shape_d[:, 1].max()) + 1
    h_l = int(shape_l[:, 0].max()) + 1
    w_l = int(shape_l[:, 1].max()) + 1
    h_r = int(shape_r[:, 0].max()) + 1
    w_r = int(shape_r[:, 1].max()) + 1

    if min(h, w) < 14:
        return None

    c_u_max = max(1, w // 2 - w_u - 4)
    c_u = random.randint(1, c_u_max)
    r_u = random.randint(2, max(2, h // 2 - h_u - 2))
    coords_u = _place(shape_u, r_u, c_u)

    c_d_min = max(c_u + w_u + 4, w // 2)
    c_d_max = w - w_d - 2
    if c_d_min > c_d_max:
        return None
    c_d = random.randint(c_d_min, c_d_max)
    r_d_min = max(h // 2, h_u + 4)
    r_d_max = h - h_d - 3
    if r_d_min > r_d_max:
        return None
    r_d = random.randint(r_d_min, r_d_max)
    coords_d = _place(shape_d, r_d, c_d)

    row_u_target = random.choice(_shape_rows(shape_u))
    row_l_local = random.choice([row for row in _shape_rows(shape_l) if row <= row_u_target])
    r_l = row_u_target - row_l_local
    if r_l + h_l > h:
        return None
    u_abs_row = row_u_target
    coords_u_final = _place(shape_u, 0, c_u)
    u_row_right = max(int(c) for r, c in coords_u_final.tolist() if int(r) == int(u_abs_row))
    l_row_min = _row_min_col(shape_l, row_l_local)
    c_l_final = u_row_right + 1 - l_row_min
    c_l_min = c_l_final + 2
    c_l_max = w - w_l - 1
    if c_l_min > c_l_max:
        return None
    c_l = random.randint(c_l_min, c_l_max)
    coords_l = _place(shape_l, r_l, c_l)

    row_d_target_local = random.choice(_shape_rows(shape_d))
    d_abs_row = (h - h_d) + row_d_target_local
    row_r_local = random.choice([row for row in _shape_rows(shape_r) if row <= d_abs_row and d_abs_row - row + h_r <= h])
    r_r = d_abs_row - row_r_local
    if not (0 <= r_r and r_r + h_r <= h):
        return None
    d_row_left = min(int(c) for r, c in (_place(shape_d, h - h_d, c_d)).tolist() if int(r) == int(d_abs_row))
    r_row_max = _row_max_col(shape_r, row_r_local)
    c_r_final = d_row_left - 1 - r_row_max
    c_r_min = 1
    c_r_max = c_r_final - 2
    if c_r_min > c_r_max:
        return None
    c_r = random.randint(c_r_min, c_r_max)
    coords_r = _place(shape_r, r_r, c_r)

    base = [
        {"id": 0, "color": BLUE, "coords": coords_u},
        {"id": 1, "color": RED, "coords": coords_d},
        {"id": 2, "color": GREEN, "coords": coords_l},
        {"id": 3, "color": YELLOW, "coords": coords_r},
    ]

    if not all(_fits(comp["coords"], h, w) for comp in base):
        return None
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            if _min_component_gap(base[i]["coords"], base[j]["coords"]) < 3:
                return None
    comps.extend(base)

    next_id = 4
    for extra_dir in extras:
        anchor = next(comp for comp in comps if int(comp["color"]) == int(extra_dir))
        shape = base_shapes[extra_dir]
        rows = anchor["coords"][:, 0]
        cols = anchor["coords"][:, 1]
        h_s = int(shape[:, 0].max()) + 1
        w_s = int(shape[:, 1].max()) + 1
        placed = None

        if extra_dir == BLUE:
            c0 = int(cols.min())
            start_r = int(rows.max()) + 4
            if start_r + h_s <= h - 1:
                placed = _place(shape, start_r, c0)
        elif extra_dir == RED:
            c0 = int(cols.min())
            start_r = int(rows.min()) - h_s - 4
            if start_r >= 1:
                placed = _place(shape, start_r, c0)
        elif extra_dir == GREEN:
            r0 = int(rows.min())
            start_c = int(cols.max()) + 4
            if start_c + w_s <= w - 1:
                placed = _place(shape, r0, start_c)
        else:
            r0 = int(rows.min())
            start_c = int(cols.min()) - w_s - 4
            if start_c >= 1:
                placed = _place(shape, r0, start_c)

        if placed is None or not _fits(placed, h, w):
            return None
        if any(_min_component_gap(placed, comp["coords"]) < 3 for comp in comps):
            return None
        comps.append({"id": next_id, "color": int(extra_dir), "coords": placed})
        next_id += 1

    irregular_count = 0
    for comp in comps:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        area = (int(rows.max()) - int(rows.min()) + 1) * (int(cols.max()) - int(cols.min()) + 1)
        if area > len(comp["coords"]):
            irregular_count += 1
    if irregular_count < 2:
        return None

    markers: list[tuple[int, int]] = []
    for comp in comps:
        candidates = _marker_candidates(comp["coords"], h, w)
        random.shuffle(candidates)
        chosen = None
        for cand in candidates:
            if any(cand == (int(r), int(c)) for other in comps for r, c in other["coords"].tolist()):
                continue
            adjacent_to_others = 0
            for other in comps:
                if other["id"] == comp["id"]:
                    continue
                other_cells = {(int(r), int(c)) for r, c in other["coords"].tolist()}
                if any(nei in other_cells for nei in neighbors4(*cand)):
                    adjacent_to_others += 1
            if adjacent_to_others:
                continue
            if any(_chebyshev_adjacent(cand, mark) for mark in markers):
                continue
            chosen = cand
            break
        if chosen is None:
            return None
        comp["marker"] = chosen
        markers.append(chosen)

    grid = np.zeros((h, w), dtype=int)
    for comp in comps:
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1]
        grid[rr, cc] = GREY
        mr, mc = comp["marker"]
        grid[mr, mc] = int(comp["color"])

    parsed = _parse_input(grid)
    if len(parsed) != n:
        return None
    recolored, recolored_comps = _recolor_and_erase(grid, parsed)
    actual_out, trace = _simulate_colored(recolored, recolored_comps)
    if not np.array_equal(actual_out, main(grid)):
        return None

    if any(item["steps"] < 1 for item in trace):
        return None
    if max(item["steps"] for item in trace) < 3:
        return None
    if not any(item["block_reason"] == "component" for item in trace):
        return None

    boundary = _boundary_only(recolored.shape, recolored_comps)
    if boundary is None or np.array_equal(boundary, actual_out):
        return None

    direction_sensitive = 0
    for comp in recolored_comps:
        original = int(comp["color"])
        changed = False
        for alt in DIR_ORDER:
            if alt == original:
                continue
            counterfactual = [{"id": int(c["id"]), "color": (alt if c["id"] == comp["id"] else int(c["color"])), "coords": np.array(c["coords"], dtype=int)} for c in recolored_comps]
            altered_grid = np.zeros_like(recolored)
            for c in counterfactual:
                altered_grid[c["coords"][:, 0], c["coords"][:, 1]] = int(c["color"])
            altered_out, _ = _simulate_colored(altered_grid, counterfactual)
            if not np.array_equal(altered_out, actual_out):
                changed = True
                break
        direction_sensitive += int(changed)
    if direction_sensitive < len(recolored_comps):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(1200):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
