from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PATH = 3
MARKER = 5
GATE_END = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

Coord = tuple[int, int]


def _add(a: Coord, b: Coord, k: int = 1) -> Coord:
    return a[0] + k * b[0], a[1] + k * b[1]


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[Coord] = set()
            while q:
                cell = q.popleft()
                comp.add(cell)
                for nr, nc in _neighbors4(cell):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_colored_cells(cells: dict[Coord, int]) -> tuple[np.ndarray, dict[str, object]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for (r, c), color in cells.items():
        patch[r - min_r, c - min_c] = color
    meta = {
        "bbox": (min_r, min_c, max_r, max_c),
        "height": patch.shape[0],
        "width": patch.shape[1],
    }
    return patch, meta


def _build_active_component(approach: Coord, shape_kind: str) -> tuple[np.ndarray, dict[str, object]]:
    perp_dirs = [d for d in DIR4 if d != approach and d != (-approach[0], -approach[1])]
    gate_left, gate_right = perp_dirs
    cells: dict[Coord, int] = {(0, 0): PATH, gate_left: GATE_END, gate_right: GATE_END}

    parking = _add((0, 0), approach)
    path_cells = [parking]
    cursor = parking
    last_dir = approach

    if shape_kind == "straight":
        tail_len = random.randint(2, 5)
        for _ in range(tail_len - 1):
            cursor = _add(cursor, approach)
            path_cells.append(cursor)
    else:
        seg1 = random.randint(1, 3)
        turn_dir = random.choice(perp_dirs)
        seg2 = random.randint(1, 3)
        for _ in range(seg1 - 1):
            cursor = _add(cursor, approach)
            path_cells.append(cursor)
        for _ in range(seg2):
            cursor = _add(cursor, turn_dir)
            path_cells.append(cursor)
        last_dir = turn_dir

    for cell in path_cells:
        cells[cell] = PATH
    marker = _add(path_cells[-1], last_dir)
    cells[marker] = MARKER

    patch, meta = _normalize_colored_cells(cells)
    meta.update(
        {
            "kind": "active",
            "shape_kind": shape_kind,
            "approach": approach,
            "path_len": len(path_cells),
        }
    )
    return patch, meta


def _build_inactive_component(start_dir: Coord, shape_kind: str) -> tuple[np.ndarray, dict[str, object]]:
    cells: dict[Coord, int] = {(0, 0): PATH}
    path_cells = [(0, 0)]
    cursor = (0, 0)
    last_dir = start_dir

    if shape_kind == "straight":
        tail_len = random.randint(3, 6)
        for _ in range(tail_len - 1):
            cursor = _add(cursor, start_dir)
            path_cells.append(cursor)
    else:
        seg1 = random.randint(1, 3)
        turn_choices = [d for d in DIR4 if d != start_dir and d != (-start_dir[0], -start_dir[1])]
        turn_dir = random.choice(turn_choices)
        seg2 = random.randint(1, 3)
        for _ in range(seg1 - 1):
            cursor = _add(cursor, start_dir)
            path_cells.append(cursor)
        for _ in range(seg2):
            cursor = _add(cursor, turn_dir)
            path_cells.append(cursor)
        last_dir = turn_dir

    for cell in path_cells:
        cells[cell] = PATH
    marker = _add((0, 0), (-start_dir[0], -start_dir[1]))
    cells[marker] = MARKER

    patch, meta = _normalize_colored_cells(cells)
    meta.update(
        {
            "kind": "inactive",
            "shape_kind": shape_kind,
            "start_dir": start_dir,
            "path_len": len(path_cells),
        }
    )
    return patch, meta


def _component_info(grid: np.ndarray, comp: set[Coord]) -> dict[str, object]:
    colors = {int(grid[r, c]) for r, c in comp}
    marker_cells = {cell for cell in comp if int(grid[cell]) == MARKER}
    path_cells = {cell for cell in comp if int(grid[cell]) == PATH}
    gate_cells = {cell for cell in comp if int(grid[cell]) == GATE_END}
    info: dict[str, object] = {
        "coords": set(comp),
        "active": False,
        "has_gate": False,
        "valid_colors": colors <= {PATH, MARKER, GATE_END},
        "shape_kind": None,
    }
    if not info["valid_colors"] or len(marker_cells) != 1 or not path_cells:
        return info

    marker = next(iter(marker_cells))
    gate_candidates: list[dict[str, object]] = []
    for center in path_cells:
        for axis in [((0, -1), (0, 1)), ((-1, 0), (1, 0))]:
            end_a = _add(center, axis[0])
            end_b = _add(center, axis[1])
            if end_a not in gate_cells or end_b not in gate_cells:
                continue
            path_neighbors = [n for n in _neighbors4(center) if n in path_cells and n != center]
            if len(path_neighbors) != 1:
                continue
            parking = path_neighbors[0]
            if axis[0][0] == 0:
                if parking[0] == center[0]:
                    continue
            else:
                if parking[1] == center[1]:
                    continue
            gate_candidates.append(
                {
                    "center": center,
                    "parking": parking,
                    "gate_ends": {end_a, end_b},
                }
            )

    if gate_candidates:
        info["has_gate"] = True
    if len(gate_candidates) != 1 or len(gate_cells) != 2:
        return info

    gate = gate_candidates[0]
    path_sub = set(path_cells)
    path_sub.remove(gate["center"])
    if gate["parking"] not in path_sub:
        return info

    marker_touches = [cell for cell in path_sub if abs(cell[0] - marker[0]) + abs(cell[1] - marker[1]) == 1]
    if len(marker_touches) != 1:
        return info
    far_end = marker_touches[0]
    if any(abs(end[0] - marker[0]) + abs(end[1] - marker[1]) == 1 for end in gate["gate_ends"]):
        return info

    degrees = {cell: sum(n in path_sub for n in _neighbors4(cell)) for cell in path_sub}
    if any(deg > 2 for deg in degrees.values()):
        return info
    if len(path_sub) == 1:
        if gate["parking"] != far_end:
            return info
    else:
        endpoints = {cell for cell, deg in degrees.items() if deg == 1}
        middles_ok = all(degrees[cell] == 2 for cell in path_sub - endpoints)
        if endpoints != {gate["parking"], far_end} or not middles_ok:
            return info
        q = deque([gate["parking"]])
        seen = {gate["parking"]}
        while q:
            cell = q.popleft()
            for nxt in _neighbors4(cell):
                if nxt in path_sub and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if seen != path_sub:
            return info

    turn_count = 0
    if len(path_sub) >= 3:
        for cell in path_sub:
            neigh = [n for n in _neighbors4(cell) if n in path_sub]
            if len(neigh) == 2:
                d0 = (neigh[0][0] - cell[0], neigh[0][1] - cell[1])
                d1 = (neigh[1][0] - cell[0], neigh[1][1] - cell[1])
                if d0[0] != -d1[0] or d0[1] != -d1[1]:
                    turn_count += 1
    info.update(
        {
            "active": True,
            "marker": marker,
            "gate_center": gate["center"],
            "gate_ends": gate["gate_ends"],
            "parking": gate["parking"],
            "path_sub": path_sub,
            "far_end": far_end,
            "shape_kind": "elbow" if turn_count else "straight",
        }
    )
    return info


def _settled_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    infos: list[dict[str, object]] = []
    for comp in _find_components(g != BG):
        info = _component_info(g, comp)
        infos.append(info)
        if not info["active"]:
            continue
        marker = info["marker"]
        parking = info["parking"]
        gate_center = info["gate_center"]
        out[marker] = BG
        for cell in info["path_sub"]:
            out[cell] = BG
        out[gate_center] = PATH
        for end in info["gate_ends"]:
            out[end] = GATE_END
        out[parking] = MARKER
    return out, infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the markers that can slide to a valid gate and return a diagonal grid of that size.

    Input structure:
    - The background is black.
    - Each non-background 4-connected component contains a one-pixel-wide green path, one gray marker pixel,
      and optionally a gate made of maroon-green-maroon in a straight line.
    - A valid gate is at a path endpoint: its green center has exactly one neighboring green path cell, and the
      two maroon gate ends lie on the perpendicular axis.

    Rule:
    - For every component with a valid gate, slide the gray marker along the connected green path to the unique
      green cell adjacent to the gate center, erasing the traversed green path and replacing that parking cell
      with the gray marker.
    - Components without a valid gate stay unchanged.
    - Let N be the number of markers that successfully stop in front of a gate, and output an N x N black grid
      whose main diagonal is green.

    Key insight:
    - The gate is both a structural filter and a stopping condition: only components whose marker sits on a
      simple path terminating at a perpendicular maroon-green-maroon gate contribute to the output count.
    """
    _settled, infos = _settled_scene(np.asarray(grid, dtype=int))
    n = sum(1 for info in infos if info["active"])
    out = np.full((n, n), BG, dtype=int)
    np.fill_diagonal(out, PATH)
    return out


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    cells = [(top + r, left + c) for r in range(h) for c in range(w) if int(patch[r, c]) != BG]
    if not cells:
        return False
    grid_h, grid_w = grid.shape
    for r, c in cells:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return False
        if grid[r, c] != BG:
            return False
        for nr, nc in _neighbors4((r, c)):
            if 0 <= nr < grid_h and 0 <= nc < grid_w and grid[nr, nc] != BG and (nr, nc) not in cells:
                return False
    return True


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _active_shape_plan(n_active: int) -> list[tuple[Coord, str]]:
    approaches = DIR4.copy()
    random.shuffle(approaches)
    shapes = ["straight", "elbow"]
    while len(shapes) < n_active:
        shapes.append(random.choice(["straight", "elbow"]))
    random.shuffle(shapes)
    plan = []
    for i in range(n_active):
        plan.append((approaches[i % len(approaches)], shapes[i]))
    return plan


def _inactive_shape_plan(n_inactive: int) -> list[tuple[Coord, str]]:
    dirs = DIR4.copy()
    random.shuffle(dirs)
    shapes = ["straight"] if n_inactive == 1 else ["straight", "elbow"]
    while len(shapes) < n_inactive:
        shapes.append(random.choice(["straight", "elbow"]))
    random.shuffle(shapes)
    return [(dirs[i % len(dirs)], shapes[i]) for i in range(n_inactive)]


def _component_shape_counts(infos: list[dict[str, object]]) -> tuple[int, int, int]:
    active = [info for info in infos if info["active"]]
    inactive = [info for info in infos if not info["active"]]
    elbow_active = sum(1 for info in active if info["shape_kind"] == "elbow")
    return len(active), len(inactive), elbow_active


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))) <= {BG, PATH, MARKER, GATE_END}:
        return False

    settled, infos = _settled_scene(grid)
    active_count, inactive_count, elbow_active = _component_shape_counts(infos)
    if active_count < 3 or inactive_count < 1:
        return False
    if elbow_active < 1:
        return False

    active_infos = [info for info in infos if info["active"]]
    inactive_infos = [info for info in infos if not info["active"]]
    if not all(info["valid_colors"] for info in infos):
        return False
    if not all(info["has_gate"] for info in active_infos):
        return False
    if any(info["has_gate"] for info in inactive_infos):
        return False

    for info in active_infos:
        if settled[info["marker"]] != BG:
            return False
        if settled[info["parking"]] != MARKER:
            return False
        if settled[info["gate_center"]] != PATH:
            return False
        for end in info["gate_ends"]:
            if settled[end] != GATE_END:
                return False
        for cell in info["path_sub"]:
            if cell != info["parking"] and settled[cell] != BG:
                return False

    for info in inactive_infos:
        coords = list(info["coords"])
        before = np.array([grid[cell] for cell in coords], dtype=int)
        after = np.array([settled[cell] for cell in coords], dtype=int)
        if not np.array_equal(before, after):
            return False

    out = main(grid)
    if out.shape != (active_count, active_count):
        return False
    expected = np.full((active_count, active_count), BG, dtype=int)
    np.fill_diagonal(expected, PATH)
    if not np.array_equal(out, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(24, 30)
    w = random.randint(24, 30)
    grid = np.full((h, w), BG, dtype=int)

    n_active = random.randint(3, 12)
    n_inactive = random.randint(1, 4)
    components: list[np.ndarray] = []

    for approach, shape_kind in _active_shape_plan(n_active):
        patch, _ = _build_active_component(approach, shape_kind)
        components.append(patch)
    for start_dir, shape_kind in _inactive_shape_plan(n_inactive):
        patch, _ = _build_inactive_component(start_dir, shape_kind)
        components.append(patch)

    random.shuffle(components)
    for patch in sorted(components, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
        ph, pw = patch.shape
        if ph > h - 2 or pw > w - 2:
            return None
        positions = [(r, c) for r in range(1, h - ph) for c in range(1, w - pw)]
        random.shuffle(positions)
        placed = False
        for top, left in positions[:250]:
            if _can_place(grid, patch, top, left):
                _paste(grid, patch, top, left)
                placed = True
                break
        if not placed:
            return None

    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ARC input")
