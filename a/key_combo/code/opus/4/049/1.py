from __future__ import annotations

from collections import deque
from importlib import util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.append(parent / "new" / "solvers" / "arc_utils.py")
    candidates.append(Path("/home/lixinyu/snap/new/solvers/arc_utils.py"))
    for path in candidates:
        if path.is_file():
            spec = util.spec_from_file_location("_arc_utils_shared", path)
            if spec is None or spec.loader is None:
                continue
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError("could not locate arc_utils.py")


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox = _ARC.bbox
in_bounds = _ARC.in_bounds
neighbors4 = _ARC.neighbors4


BACKGROUND = int(Color.BLACK)
PATH_COLOR = int(Color.BLUE)
WALL_COLOR = int(Color.RED)
CONTROL_COLOR = int(Color.GREEN)
CHANNEL_COLOR = int(Color.YELLOW)
FRAME_COLOR = int(Color.GREY)

GEN_RETRIES = 500
CANVAS_H_RANGE = (12, 19)
CANVAS_W_RANGE = (12, 19)
FRAME_H_RANGE = (9, 17)
FRAME_W_RANGE = (9, 17)
FRAME_MARGIN = 1
PATTERN_WEIGHTS = (
    ("odd_even", 3),
    ("even_odd", 3),
    ("even_even", 2),
)


def _extract_structure(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, WALL_COLOR, CONTROL_COLOR, FRAME_COLOR}, "unexpected input colors"

    frame_mask = g == FRAME_COLOR
    frame_bbox = bbox(frame_mask)
    assert frame_bbox is not None, "expected one grey frame"
    r0, c0, r1, c1 = map(int, frame_bbox)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame interior must be at least 3x3"

    expected_frame = np.zeros_like(frame_mask, dtype=bool)
    expected_frame[r0, c0 : c1 + 1] = True
    expected_frame[r1, c0 : c1 + 1] = True
    expected_frame[r0 : r1 + 1, c0] = True
    expected_frame[r0 : r1 + 1, c1] = True
    assert np.array_equal(frame_mask, expected_frame), "grey pixels must form one rectangular border"

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert np.all((interior == BACKGROUND) | (interior == WALL_COLOR) | (interior == CONTROL_COLOR)), "interior may contain only bg, endpoints, and control"

    controls = np.argwhere(g == CONTROL_COLOR)
    assert controls.shape == (1, 2), "expected exactly one green control pixel"
    control = tuple(map(int, controls[0]))
    assert r0 < control[0] < r1 and c0 < control[1] < c1, "control must lie inside the frame"

    endpoints_arr = np.argwhere(g == WALL_COLOR)
    assert endpoints_arr.shape == (2, 2), "expected exactly two red endpoints"
    endpoints = tuple(sorted((tuple(map(int, endpoints_arr[0])), tuple(map(int, endpoints_arr[1])))))
    for r, c in endpoints:
        assert r0 < r < r1 and c0 < c < c1, "endpoints must lie inside the frame"
        assert (r, c) != control, "control and endpoints must be distinct"

    return (r0, c0, r1, c1), control, endpoints


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _paint_parity_maze(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = frame_bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(out[r, c]) != BACKGROUND:
                continue
            distance = _chebyshev((r, c), control)
            out[r, c] = CHANNEL_COLOR if distance % 2 == 1 else WALL_COLOR
    return out


def _passable_mask(
    filled: np.ndarray,
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    mask = filled == CHANNEL_COLOR
    for r, c in endpoints:
        mask[r, c] = True
    return mask


def _unique_shortest_path(
    passable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, passable.shape)
    dist = np.full((h, w), -1, dtype=int)
    ways = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    ways[start] = 1

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc) or not bool(passable[nr, nc]):
                continue
            nd = int(dist[r, c]) + 1
            if int(dist[nr, nc]) == -1:
                dist[nr, nc] = nd
                ways[nr, nc] = ways[r, c]
                q.append((nr, nc))
            elif int(dist[nr, nc]) == nd:
                ways[nr, nc] = min(2, int(ways[nr, nc]) + int(ways[r, c]))

    assert int(dist[goal]) >= 0, "expected the endpoints to be connected through yellow channels"
    assert int(ways[goal]) == 1, "expected a unique shortest path"

    path = [goal]
    cur = goal
    while cur != start:
        r, c = cur
        predecessors = [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if in_bounds((h, w), nr, nc)
            and bool(passable[nr, nc])
            and int(dist[nr, nc]) == int(dist[r, c]) - 1
        ]
        assert len(predecessors) == 1, "shortest path predecessor chain must be unique"
        cur = predecessors[0]
        path.append(cur)
    path.reverse()
    return path


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], tuple[int, int, int, int], tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    frame_bbox, control, endpoints = _extract_structure(grid)
    filled = _paint_parity_maze(grid, frame_bbox, control)
    path = _unique_shortest_path(_passable_mask(filled, endpoints), endpoints[0], endpoints[1])
    out = filled.copy()
    for r, c in path[1:-1]:
        assert int(out[r, c]) == CHANNEL_COLOR, "only yellow channel pixels may be recolored to blue"
        out[r, c] = PATH_COLOR
    return out, filled, path, frame_bbox, control, endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill an eccentric framed interior with parity-based square rings and mark the unique channel path between the red endpoints.

    Input structure:
    - A black background contains exactly one grey single-pixel rectangular frame.
    - Inside the frame there is one green control pixel and exactly two red endpoint pixels.
    - Every other interior cell is black.

    Rule:
    - Preserve the grey frame, the green control pixel, and the two red endpoints.
    - For each black cell inside the frame, compute its Chebyshev distance to the green control pixel.
    - Recolor odd-distance cells yellow and even-distance cells red.
    - Treat yellow cells plus the two original red endpoints as passable and the red parity cells as walls.
    - Find the unique shortest 4-neighbor path between the two endpoints and recolor only its yellow interior cells blue.

    Key insight:
    - The green control pixel generates concentric square-distance rings, but their odd/even parity matters as maze topology rather than decoration: only odd rings are traversable, so the eccentric ring clipping by the frame determines the unique route.
    """
    out, _filled, _path, _frame_bbox, _control, _endpoints = _solve(grid)
    return out


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _direction_signature(cell: tuple[int, int], control: tuple[int, int]) -> tuple[int, int]:
    return _sign(cell[0] - control[0]), _sign(cell[1] - control[1])


def _path_turns(path: list[tuple[int, int]]) -> int:
    return sum(
        (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])
        for a, b, c in zip(path, path[1:], path[2:])
    )


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q: deque[tuple[int, int]] = deque([(rr, cc)])
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            comp.append((cr, cc2))
            for nr, nc in neighbors4(cr, cc2):
                if in_bounds((h, w), nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _is_simple_corridor(cells: list[tuple[int, int]]) -> bool:
    if len(cells) < 6:
        return False
    cell_set = set(cells)
    edges = 0
    degrees = []
    for cell in cells:
        deg = sum((nbr in cell_set) for nbr in neighbors4(*cell))
        degrees.append(int(deg))
        edges += int(deg)
    edges //= 2
    return edges == len(cells) - 1 and max(degrees, default=0) <= 2 and sum(d == 1 for d in degrees) == 2


def _free_space_path_length(
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    shape: tuple[int, int],
) -> int:
    r0, c0, r1, c1 = frame_bbox
    passable = np.zeros(shape, dtype=bool)
    passable[r0 + 1 : r1, c0 + 1 : c1] = True
    passable[control] = False
    for cell in endpoints:
        passable[cell] = True
    h, w = map(int, shape)
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque([endpoints[0]])
    dist[endpoints[0]] = 0
    while q:
        r, c = q.popleft()
        if (r, c) == endpoints[1]:
            return int(dist[r, c]) + 1
        for nr, nc in neighbors4(r, c):
            if not in_bounds((h, w), nr, nc) or not bool(passable[nr, nc]) or int(dist[nr, nc]) != -1:
                continue
            dist[nr, nc] = int(dist[r, c]) + 1
            q.append((nr, nc))
    raise AssertionError("free-space endpoints should be connected")


def _ring_component_for_path(
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
    filled: np.ndarray,
    path: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    interior_path = path[1:-1]
    if not interior_path:
        return None
    sample = interior_path[0]
    ring_distance = _chebyshev(sample, control)
    assert ring_distance % 2 == 1
    r0, c0, r1, c1 = frame_bbox
    mask = np.zeros_like(filled, dtype=bool)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if _chebyshev((r, c), control) == ring_distance and int(filled[r, c]) == CHANNEL_COLOR:
                mask[r, c] = True
    for comp in _component_cells(mask):
        if sample in set(comp):
            return comp
    return None


def _semantic_sample_ok(inp: np.ndarray) -> bool:
    try:
        out, filled, path, frame_bbox, control, endpoints = _solve(inp)
    except AssertionError:
        return False

    d0 = _chebyshev(endpoints[0], control)
    d1 = _chebyshev(endpoints[1], control)
    if d0 == d1:
        return False

    if _direction_signature(endpoints[0], control) == _direction_signature(endpoints[1], control):
        return False

    if len(path) < 7 or _path_turns(path) < 2:
        return False

    free_len = _free_space_path_length(frame_bbox, control, endpoints, inp.shape)
    if len(path) < free_len + 2:
        return False

    up = control[0] - (frame_bbox[0] + 1)
    down = (frame_bbox[2] - 1) - control[0]
    left = control[1] - (frame_bbox[1] + 1)
    right = (frame_bbox[3] - 1) - control[1]
    if max(up, down, left, right) - min(up, down, left, right) < 2:
        return False

    ring_comp = _ring_component_for_path(frame_bbox, control, filled, path)
    if ring_comp is None or not _is_simple_corridor(ring_comp):
        return False

    expected = filled.copy()
    for r, c in path[1:-1]:
        expected[r, c] = PATH_COLOR
    if not np.array_equal(out, expected):
        return False

    return True


def _sample_frame() -> tuple[int, int, int, int, int, int]:
    h = random.randint(*CANVAS_H_RANGE)
    w = random.randint(*CANVAS_W_RANGE)
    frame_h = random.randint(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], h - 2 * FRAME_MARGIN))
    frame_w = random.randint(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], w - 2 * FRAME_MARGIN))
    r0 = random.randint(FRAME_MARGIN, h - frame_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, w - frame_w - FRAME_MARGIN)
    return h, w, r0, c0, r0 + frame_h - 1, c0 + frame_w - 1


def _control_candidates(
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    center_r = (r0 + r1) / 2
    center_c = (c0 + c1) / 2
    candidates = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if abs(r - center_r) < 1e-9 and abs(c - center_c) < 1e-9:
                continue
            up = r - (r0 + 1)
            down = (r1 - 1) - r
            left = c - (c0 + 1)
            right = (c1 - 1) - c
            if max(up, down, left, right) < 3:
                continue
            if max(up, down, left, right) - min(up, down, left, right) < 2:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _odd_corridors(
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    corridors: list[dict] = []
    for dist in range(1, max(r1 - r0, c1 - c0) + 1, 2):
        mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
        found = False
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if _chebyshev((r, c), control) == dist:
                    mask[r, c] = True
                    found = True
        if not found:
            continue
        for comp in _component_cells(mask):
            if _is_simple_corridor(comp):
                corridors.append({"distance": dist, "cells": comp, "cell_set": set(comp)})
    random.shuffle(corridors)
    return corridors


def _endpoint_infos(
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
    corridor: dict,
) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    cell_set = corridor["cell_set"]
    infos = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            cell = (r, c)
            if cell == control:
                continue
            adj = [nbr for nbr in neighbors4(r, c) if nbr in cell_set]
            on_corridor = cell in cell_set
            if not on_corridor and not adj:
                continue
            infos.append(
                {
                    "cell": cell,
                    "dist": _chebyshev(cell, control),
                    "sig": _direction_signature(cell, control),
                    "on_corridor": on_corridor,
                    "adj_count": len(adj),
                }
            )
    random.shuffle(infos)
    return infos


def _pattern_choices(pattern: str, infos: list[dict], corridor_distance: int) -> list[tuple[dict, dict]]:
    odd_infos = [info for info in infos if info["on_corridor"] and info["dist"] == corridor_distance]
    even_infos = [info for info in infos if not info["on_corridor"] and info["dist"] % 2 == 0]
    pairs: list[tuple[dict, dict]] = []

    if pattern == "odd_even":
        for a in odd_infos:
            for b in even_infos:
                pairs.append((a, b))
    elif pattern == "even_odd":
        for a in even_infos:
            for b in odd_infos:
                pairs.append((a, b))
    else:
        for i, a in enumerate(even_infos):
            for b in even_infos[i + 1 :]:
                if a["dist"] != b["dist"]:
                    pairs.append((a, b))
    random.shuffle(pairs)
    return pairs


def _build_input(
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    control: tuple[int, int],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[control] = CONTROL_COLOR
    for cell in endpoints:
        grid[cell] = WALL_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    h, w, r0, c0, r1, c1 = _sample_frame()
    frame_bbox = (r0, c0, r1, c1)
    controls = _control_candidates(frame_bbox)
    if not controls:
        return None

    patterns, weights = zip(*PATTERN_WEIGHTS)
    for control in controls[:20]:
        corridors = _odd_corridors(frame_bbox, control)
        if not corridors:
            continue
        for corridor in corridors[:8]:
            infos = _endpoint_infos(frame_bbox, control, corridor)
            if not infos:
                continue
            for pattern in random.choices(patterns, weights=weights, k=3):
                for a, b in _pattern_choices(pattern, infos, int(corridor["distance"]))[:60]:
                    if a["cell"] == b["cell"]:
                        continue
                    if a["dist"] == b["dist"]:
                        continue
                    if a["sig"] == b["sig"]:
                        continue
                    endpoints = tuple(sorted((a["cell"], b["cell"])))
                    inp = _build_input(h, w, frame_bbox, control, endpoints)
                    if _semantic_sample_ok(inp):
                        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-maze sample")
