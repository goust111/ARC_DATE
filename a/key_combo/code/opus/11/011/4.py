from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1),
}
LEFT_TURN = {
    (-1, 0): (0, -1),
    (1, 0): (0, 1),
    (0, -1): (1, 0),
    (0, 1): (-1, 0),
}
RIGHT_TURN = {v: k for k, v in LEFT_TURN.items()}
MAX_GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_gate(grid: np.ndarray, comp_set: set[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    candidates: list[tuple[tuple[int, int], int]] = []
    for r, c in comp_set:
        if int(grid[r, c]) != GREEN:
            continue
        for (dr1, dc1), (dr2, dc2) in [((0, -1), (0, 1)), ((-1, 0), (1, 0))]:
            p1 = (r + dr1, c + dc1)
            p2 = (r + dr2, c + dc2)
            if p1 not in comp_set or p2 not in comp_set:
                continue
            end_color = int(grid[p1])
            if end_color not in (BLUE, RED) or int(grid[p2]) != end_color:
                continue
            has_frame_neighbor = any(
                _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == ORANGE
                for nr, nc in _neighbors4(r, c)
            )
            has_path_neighbor = any(
                _in_bounds(grid.shape, nr, nc)
                and (nr, nc) not in {p1, p2}
                and int(grid[nr, nc]) == GREEN
                for nr, nc in _neighbors4(r, c)
            )
            if has_frame_neighbor and has_path_neighbor:
                candidates.append(((r, c), end_color))
                break
    assert len(candidates) == 1, "expected exactly one gate center per component"
    return candidates[0]


def _frame_bbox(grid: np.ndarray, comp_set: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    orange = [(r, c) for r, c in comp_set if int(grid[r, c]) == ORANGE]
    assert orange, "component must contain one orange frame"
    r0, c0, r1, c1 = _bbox(orange)
    for r, c in orange:
        assert r in (r0, r1) or c in (c0, c1), "orange pixels must lie on a rectangular border"
    for c in range(c0, c1 + 1):
        assert int(grid[r0, c]) == ORANGE and int(grid[r1, c]) == ORANGE
    for r in range(r0, r1 + 1):
        assert int(grid[r, c0]) == ORANGE and int(grid[r, c1]) == ORANGE
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have an interior"
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray path marker to its gate and use that landing point to drive a gate-colored concentric fill.

    Input structure
    - The background is black.
    - Each foreground component contains a gray marker, a one-cell-thick green path, a three-cell gate with a green center and red or blue ends, and an orange hollow rectangle attached behind the gate.
    - The marker touches one end of the path, while the gate center touches the other end and also touches one side of the orange frame.

    Rule
    - In each component, locate the gray marker and the gate center defined by a green cell between two same-colored red or blue gate ends.
    - Erase the original gray marker and every green cell, including the path and the gate center, then place a gray marker on the former gate-center cell.
    - Keep the red or blue gate ends and the orange frame unchanged, and fill the frame interior by Chebyshev distance from the new marker.
    - Red gates use odd-distance red and even-distance yellow; blue gates use odd-distance blue and even-distance green.

    Key insight
    - The path only determines where the gray marker stops: that dynamic stopping point becomes the center of the concentric fill, and the gate-end color selects which alternating color pair to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _components(g != BG):
        comp_set = set(comp)
        gate_center, gate_color = _find_gate(g, comp_set)
        markers = [(r, c) for r, c in comp if int(g[r, c]) == GREY]
        assert len(markers) == 1, "expected one marker per component"
        marker = markers[0]
        frame = _frame_bbox(g, comp_set)

        green_cells = [(r, c) for r, c in comp if int(g[r, c]) == GREEN]
        for r, c in green_cells:
            out[r, c] = BG
        out[marker] = BG
        out[gate_center] = GREY

        even_color = YELLOW if gate_color == RED else GREEN
        r0, c0, r1, c1 = frame
        gr, gc = gate_center
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                dist = max(abs(r - gr), abs(c - gc))
                out[r, c] = gate_color if dist % 2 == 1 else even_color

    return out


def _frame_cells(
    frame_dir: tuple[int, int], outer_h: int, outer_w: int
) -> set[tuple[int, int]]:
    if frame_dir == (1, 0):
        top, bottom = 1, outer_h
        left, right = -(outer_w // 2), outer_w // 2
    elif frame_dir == (-1, 0):
        top, bottom = -outer_h, -1
        left, right = -(outer_w // 2), outer_w // 2
    elif frame_dir == (0, 1):
        top, bottom = -(outer_h // 2), outer_h // 2
        left, right = 1, outer_w
    else:
        assert frame_dir == (0, -1)
        top, bottom = -(outer_h // 2), outer_h // 2
        left, right = -outer_w, -1

    cells = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _candidate_marker_positions(
    endpoint: tuple[int, int],
    tail_dir: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    er, ec = endpoint
    ordered_dirs = [tail_dir, LEFT_TURN[tail_dir], RIGHT_TURN[tail_dir], OPPOSITE[tail_dir]]
    candidates: list[tuple[int, int]] = []
    for dr, dc in ordered_dirs:
        pos = (er + dr, ec + dc)
        if pos in forbidden:
            continue
        candidates.append(pos)
    return candidates


def _build_component(gate_color: int) -> dict[tuple[int, int], int] | None:
    gate_axis_horizontal = random.choice([True, False])
    frame_dir = random.choice([(1, 0), (-1, 0)]) if gate_axis_horizontal else random.choice([(0, 1), (0, -1)])
    path_dir = OPPOSITE[frame_dir]

    outer_h = random.choice([5, 7])
    outer_w = random.choice([5, 7])
    if gate_axis_horizontal:
        outer_w = random.choice([5, 7])
    else:
        outer_h = random.choice([5, 7])

    cells: dict[tuple[int, int], int] = {(0, 0): GREEN}
    if gate_axis_horizontal:
        cells[(0, -1)] = gate_color
        cells[(0, 1)] = gate_color
    else:
        cells[(-1, 0)] = gate_color
        cells[(1, 0)] = gate_color

    frame = _frame_cells(frame_dir, outer_h, outer_w)
    for pos in frame:
        if pos in cells:
            return None
        cells[pos] = ORANGE

    body_len = random.randint(4, 9)
    turn_count = random.choices([0, 1, 2], weights=[2, 4, 3])[0]
    dirs = [path_dir]
    if turn_count >= 1:
        dirs.append(random.choice([LEFT_TURN[path_dir], RIGHT_TURN[path_dir]]))
    if turn_count == 2:
        dirs.append(path_dir)

    if turn_count == 0:
        seg_lengths = [body_len]
    elif turn_count == 1:
        first = random.randint(1, body_len - 1)
        seg_lengths = [first, body_len - first]
    else:
        first = random.randint(1, body_len - 2)
        second = random.randint(1, body_len - first - 1)
        seg_lengths = [first, second, body_len - first - second]

    reserved = set(cells)
    path_cells: list[tuple[int, int]] = []
    used = {(0, 0)}
    cur = (0, 0)
    for direction, length in zip(dirs, seg_lengths):
        dr, dc = direction
        for _ in range(length):
            cur = (cur[0] + dr, cur[1] + dc)
            if cur in reserved or cur in used:
                return None
            path_cells.append(cur)
            used.add(cur)

    path_set = set(path_cells)
    for idx, (r, c) in enumerate(path_cells):
        allowed = set()
        if idx > 0:
            allowed.add(path_cells[idx - 1])
        if idx + 1 < len(path_cells):
            allowed.add(path_cells[idx + 1])
        if idx == 0:
            allowed.add((0, 0))
        for nb in _neighbors4(r, c):
            if nb in reserved and nb not in allowed:
                return None

    endpoint = path_cells[-1]
    tail_dir = dirs[-1]
    marker = None
    forbidden = reserved | path_set | {(0, 0)}
    for cand in _candidate_marker_positions(endpoint, tail_dir, forbidden):
        touching = {nb for nb in _neighbors4(cand[0], cand[1]) if nb in forbidden or nb in path_set}
        if touching == {endpoint}:
            marker = cand
            break
    if marker is None:
        return None

    for pos in path_cells:
        cells[pos] = GREEN
    cells[marker] = GREY

    return cells


def _translated_component(
    comp: dict[tuple[int, int], int], top: int, left: int
) -> dict[tuple[int, int], int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    min_r, min_c = min(rs), min(cs)
    dr = top - min_r
    dc = left - min_c
    return {(r + dr, c + dc): color for (r, c), color in comp.items()}


def _component_size(comp: dict[tuple[int, int], int]) -> tuple[int, int]:
    coords = list(comp)
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _paint_component(grid: np.ndarray, comp: dict[tuple[int, int], int]) -> bool:
    h, w = grid.shape
    occupied = set(comp)
    for r, c in occupied:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
            return False
    for r, c in occupied:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG and (nr, nc) not in occupied:
                return False
    for (r, c), color in comp.items():
        grid[r, c] = color
    return True


def _validate_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    comps = _components(grid != BG)
    if len(comps) != 2:
        return False
    gate_colors = []
    for comp in comps:
        comp_set = set(comp)
        markers = [(r, c) for r, c in comp if int(grid[r, c]) == GREY]
        if len(markers) != 1:
            return False
        try:
            _, gate_color = _find_gate(grid, comp_set)
            _frame_bbox(grid, comp_set)
        except AssertionError:
            return False
        gate_colors.append(gate_color)
    return sorted(gate_colors) == [BLUE, RED]


def _gen_attempt() -> np.ndarray | None:
    component_specs = [RED, BLUE]
    random.shuffle(component_specs)

    components: list[dict[tuple[int, int], int]] = []
    for gate_color in component_specs:
        comp = None
        for _ in range(80):
            comp = _build_component(gate_color)
            if comp is not None:
                break
        if comp is None:
            return None
        components.append(comp)

    sizes = [_component_size(comp) for comp in components]
    gap = random.randint(2, 5)

    horizontal_dims = (max(s[0] for s in sizes) + 2, sizes[0][1] + sizes[1][1] + gap + 3)
    vertical_dims = (sizes[0][0] + sizes[1][0] + gap + 3, max(s[1] for s in sizes) + 2)
    feasible = []
    if max(horizontal_dims) <= 30:
        feasible.append("horizontal")
    if max(vertical_dims) <= 30:
        feasible.append("vertical")
    if not feasible:
        return None

    arrangement = random.choice(feasible)
    if arrangement == "horizontal":
        min_h, min_w = horizontal_dims
        h_lo = max(16, min_h)
        h_hi = min(30, h_lo + 6)
        w_lo = max(18, min_w)
        w_hi = min(30, w_lo + 8)
        h = random.randint(h_lo, h_hi)
        w = random.randint(w_lo, w_hi)
        left1 = random.randint(1, w - sizes[0][1] - sizes[1][1] - gap - 2)
        top1 = random.randint(1, h - sizes[0][0] - 1)
        left2_min = left1 + sizes[0][1] + gap
        left2 = random.randint(left2_min, w - sizes[1][1] - 1)
        top2 = random.randint(1, h - sizes[1][0] - 1)
        placements = [(top1, left1), (top2, left2)]
    else:
        min_h, min_w = vertical_dims
        h_lo = max(18, min_h)
        h_hi = min(30, h_lo + 8)
        w_lo = max(16, min_w)
        w_hi = min(30, w_lo + 6)
        h = random.randint(h_lo, h_hi)
        w = random.randint(w_lo, w_hi)
        top1 = random.randint(1, h - sizes[0][0] - sizes[1][0] - gap - 2)
        left1 = random.randint(1, w - sizes[0][1] - 1)
        top2_min = top1 + sizes[0][0] + gap
        top2 = random.randint(top2_min, h - sizes[1][0] - 1)
        left2 = random.randint(1, w - sizes[1][1] - 1)
        placements = [(top1, left1), (top2, left2)]

    grid = np.zeros((h, w), dtype=int)
    for comp, (top, left) in zip(components, placements):
        placed = _translated_component(comp, top, left)
        if not _paint_component(grid, placed):
            return None

    if not _validate_input(grid):
        return None
    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
