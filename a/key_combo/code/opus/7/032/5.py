import itertools
import random
from collections import deque

import numpy as np


BG = 0
CUP = 5
MARKER_COLORS = (1, 2, 3, 7)
QUADRANTS = ("ul", "ur", "ll", "lr")
DIRECTIONS = ("up", "down", "left", "right")
DIR_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CENTER_FACING = {
    "ul": {"right", "down"},
    "ur": {"left", "down"},
    "ll": {"right", "up"},
    "lr": {"left", "up"},
}
GRID_SIZE_RANGE = (13, 16)
GEN_RETRY_LIMIT = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _locate_marker_block(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r in range(h - 1):
        for c in range(w - 1):
            block = grid[r : r + 2, c : c + 2]
            vals = [int(v) for v in block.flatten()]
            if any(v in (BG, CUP) for v in vals):
                continue
            if len(set(vals)) != 4:
                continue
            candidates.append((r, c))
    assert len(candidates) == 1, "expected exactly one 2x2 marker block"
    return candidates[0]


def _quadrant_for_bbox(box: tuple[int, int, int, int], marker_top_left: tuple[int, int]) -> str:
    r0, c0, r1, c1 = box
    mr, mc = marker_top_left
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    marker_center_r = mr + 0.5
    marker_center_c = mc + 0.5
    if center_r < marker_center_r and center_c < marker_center_c:
        return "ul"
    if center_r < marker_center_r and center_c > marker_center_c:
        return "ur"
    if center_r > marker_center_r and center_c < marker_center_c:
        return "ll"
    return "lr"


def _opening_direction(coords: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    counts = {
        "up": sum(r == r0 for r, _ in coords),
        "down": sum(r == r1 for r, _ in coords),
        "left": sum(c == c0 for _, c in coords),
        "right": sum(c == c1 for _, c in coords),
    }
    direction = min(counts, key=counts.get)
    assert list(counts.values()).count(counts[direction]) == 1, "opening side must be unique"
    assert counts[direction] == 2, "cup opening must have two endpoints"
    return direction


def _mouth_and_endpoints(box: tuple[int, int, int, int], direction: str):
    r0, c0, r1, c1 = box
    if direction == "up":
        return (
            [(r0, c) for c in range(c0 + 1, c1)],
            [(r0, c0), (r0, c1)],
            (-1, 0),
            [(-1, -1), (-1, 1)],
        )
    if direction == "down":
        return (
            [(r1, c) for c in range(c0 + 1, c1)],
            [(r1, c0), (r1, c1)],
            (1, 0),
            [(1, -1), (1, 1)],
        )
    if direction == "left":
        return (
            [(r, c0) for r in range(r0 + 1, r1)],
            [(r0, c0), (r1, c0)],
            (0, -1),
            [(-1, -1), (1, -1)],
        )
    return (
        [(r, c1) for r in range(r0 + 1, r1)],
        [(r0, c1), (r1, c1)],
        (0, 1),
        [(-1, 1), (1, 1)],
    )


def _cavity_cells(grid: np.ndarray, coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    walls = set(coords)
    cavity = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) not in walls and int(grid[r, c]) == BG:
                cavity.append((r, c))
    return cavity


def _paint_ray(out: np.ndarray, source: np.ndarray, start: tuple[int, int], step: tuple[int, int], color: int) -> None:
    r, c = start
    dr, dc = step
    while _in_bounds(out.shape, r, c) and int(source[r, c]) == BG:
        out[r, c] = color
        r += dr
        c += dc


def _apply_ray_action(out: np.ndarray, source: np.ndarray, box: tuple[int, int, int, int], direction: str, color: int) -> None:
    mouth, endpoints, forward_step, diagonal_steps = _mouth_and_endpoints(box, direction)
    for cell in mouth:
        start = (cell[0] + forward_step[0], cell[1] + forward_step[1])
        _paint_ray(out, source, start, forward_step, color)
    for endpoint, diag_step in zip(endpoints, diagonal_steps):
        start = (endpoint[0] + diag_step[0], endpoint[1] + diag_step[1])
        _paint_ray(out, source, start, diag_step, color)


def _cup_geometry(direction: str, inner_width: int, depth: int) -> dict:
    if direction in ("up", "down"):
        h = depth + 1
        w = inner_width + 2
    else:
        h = inner_width + 2
        w = depth + 1

    if direction == "up":
        walls = {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif direction == "down":
        walls = {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)} | {(0, c) for c in range(w)}
    elif direction == "left":
        walls = {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)} | {(r, w - 1) for r in range(h)}
    else:
        walls = {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)}

    cavity = {(r, c) for r in range(h) for c in range(w) if (r, c) not in walls}
    box = (0, 0, h - 1, w - 1)
    mouth, endpoints, forward_step, diagonal_steps = _mouth_and_endpoints(box, direction)
    return {
        "shape": (h, w),
        "walls": tuple(sorted(walls)),
        "cavity": tuple(sorted(cavity)),
        "mouth": mouth,
        "endpoints": endpoints,
        "forward_step": forward_step,
        "diagonal_steps": diagonal_steps,
        "direction": direction,
    }


def _shift_cells(cells, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r + r0, c + c0) for r, c in cells]


def _cell_in_quadrant(cell: tuple[int, int], quadrant: str, marker_top_left: tuple[int, int]) -> bool:
    r, c = cell
    mr, mc = marker_top_left
    if quadrant == "ul":
        return r < mr and c < mc
    if quadrant == "ur":
        return r < mr and c > mc + 1
    if quadrant == "ll":
        return r > mr + 1 and c < mc
    return r > mr + 1 and c > mc + 1


def _ray_path(shape: tuple[int, int], blocked: set[tuple[int, int]], start: tuple[int, int], step: tuple[int, int]):
    r, c = start
    dr, dc = step
    path = []
    while _in_bounds(shape, r, c) and (r, c) not in blocked:
        path.append((r, c))
        r += dr
        c += dc
    if not path:
        return None
    last_r, last_c = path[-1]
    if last_r not in (0, shape[0] - 1) and last_c not in (0, shape[1] - 1):
        return None
    return path


def _candidate_cups(
    shape: tuple[int, int],
    marker_top_left: tuple[int, int],
    quadrant: str,
    direction: str,
    blocked: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    outward = direction not in CENTER_FACING[quadrant]
    inner_widths = [1] if outward else [1, 2]
    candidates = []
    for inner_width in inner_widths:
        for depth in (2, 3, 4):
            geom = _cup_geometry(direction, inner_width, depth)
            gh, gw = geom["shape"]
            for r0 in range(h - gh + 1):
                for c0 in range(w - gw + 1):
                    walls = _shift_cells(geom["walls"], r0, c0)
                    cavity = _shift_cells(geom["cavity"], r0, c0)
                    if any((cell in blocked) or (not _cell_in_quadrant(cell, quadrant, marker_top_left)) for cell in walls):
                        continue
                    if any(not _cell_in_quadrant(cell, quadrant, marker_top_left) for cell in cavity):
                        continue
                    info = {
                        "quadrant": quadrant,
                        "direction": direction,
                        "inner_width": inner_width,
                        "depth": depth,
                        "walls": walls,
                        "cavity": cavity,
                        "box": (r0, c0, r0 + gh - 1, c0 + gw - 1),
                        "ray_cells": [],
                    }
                    if outward:
                        local_blocked = set(blocked) | set(walls)
                        ray_cells: list[tuple[int, int]] = []
                        ok = True
                        for cell in _shift_cells(geom["mouth"], r0, c0):
                            start = (cell[0] + geom["forward_step"][0], cell[1] + geom["forward_step"][1])
                            path = _ray_path(shape, local_blocked, start, geom["forward_step"])
                            if path is None:
                                ok = False
                                break
                            ray_cells.extend(path)
                            local_blocked.update(path)
                        if not ok:
                            continue
                        for endpoint, diag_step in zip(_shift_cells(geom["endpoints"], r0, c0), geom["diagonal_steps"]):
                            start = (endpoint[0] + diag_step[0], endpoint[1] + diag_step[1])
                            path = _ray_path(shape, local_blocked, start, diag_step)
                            if path is None:
                                ok = False
                                break
                            ray_cells.extend(path)
                            local_blocked.update(path)
                        if not ok:
                            continue
                        info["ray_cells"] = ray_cells
                    candidates.append(info)
    return candidates


_DIRECTION_ASSIGNMENTS = []
for perm in itertools.permutations(DIRECTIONS):
    assignment = dict(zip(QUADRANTS, perm))
    if sum(assignment[q] in CENTER_FACING[q] for q in QUADRANTS) == 2:
        _DIRECTION_ASSIGNMENTS.append(assignment)
_DIRECTION_ASSIGNMENTS = tuple(_DIRECTION_ASSIGNMENTS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each quadrant cup from the marker block's diagonal mapping, filling center-facing cups and ray-casting from outward-facing cups.

    Input structure:
    - A black background contains one central 2x2 marker block whose four pixels are distinct non-gray colors.
    - One gray cup-shaped container lies in each of the four quadrants around the marker block.
    - Each cup is a 1-cell-thick three-sided container with a black cavity and a single open side.

    Rule:
    - Read the marker block with diagonal mapping: top-left -> lower-right quadrant, top-right -> lower-left, bottom-left -> upper-right, bottom-right -> upper-left.
    - For each cup, use the mapped quadrant color. If the cup opening faces toward the marker block, fill its cavity with that color.
    - If the cup opening faces away from the marker block, keep the cavity black and emit one straight ray from the opening plus two diagonal rays from the opening endpoints, extending to the grid edge.

    Key insight:
    - The marker block encodes the color independently of the cup action, while each cup's opening direction independently encodes whether to fill or to cast rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mr, mc = _locate_marker_block(g)
    block = g[mr : mr + 2, mc : mc + 2]
    quadrant_color = {
        "ul": int(block[1, 1]),
        "ur": int(block[1, 0]),
        "ll": int(block[0, 1]),
        "lr": int(block[0, 0]),
    }

    marker_center = (mr + 0.5, mc + 0.5)
    cups = _components_of_color(g, CUP)
    assert len(cups) == 4, "expected exactly four cups"

    out = g.copy()
    for cup in cups:
        box = _bbox(cup)
        quadrant = _quadrant_for_bbox(box, (mr, mc))
        direction = _opening_direction(cup)
        cup_center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        to_marker = (marker_center[0] - cup_center[0], marker_center[1] - cup_center[1])
        dr, dc = DIR_VEC[direction]
        faces_marker = dr * to_marker[0] + dc * to_marker[1] > 0
        color = quadrant_color[quadrant]

        if faces_marker:
            for r, c in _cavity_cells(g, cup):
                out[r, c] = color
        else:
            _apply_ray_action(out, g, box, direction, color)

    return out


def _build_expected_output(inp: np.ndarray, cups: list[dict], marker_top_left: tuple[int, int]) -> np.ndarray:
    mr, mc = marker_top_left
    block = inp[mr : mr + 2, mc : mc + 2]
    quadrant_color = {
        "ul": int(block[1, 1]),
        "ur": int(block[1, 0]),
        "ll": int(block[0, 1]),
        "lr": int(block[0, 0]),
    }
    out = inp.copy()
    for cup in cups:
        color = quadrant_color[cup["quadrant"]]
        if cup["direction"] in CENTER_FACING[cup["quadrant"]]:
            for r, c in cup["cavity"]:
                out[r, c] = color
        else:
            for r, c in cup["ray_cells"]:
                out[r, c] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    marker_top_left = (h // 2 - 1, w // 2 - 1)
    mr, mc = marker_top_left
    marker_perm = random.sample(list(MARKER_COLORS), k=4)
    grid[mr : mr + 2, mc : mc + 2] = np.array(marker_perm, dtype=int).reshape(2, 2)

    blocked = {(mr + dr, mc + dc) for dr in range(2) for dc in range(2)}
    assignment = random.choice(_DIRECTION_ASSIGNMENTS)
    order = sorted(QUADRANTS, key=lambda q: assignment[q] in CENTER_FACING[q])

    cups: list[dict] = []
    for quadrant in order:
        direction = assignment[quadrant]
        candidates = _candidate_cups((h, w), marker_top_left, quadrant, direction, blocked)
        if not candidates:
            return None
        choice = random.choice(candidates)
        for r, c in choice["walls"]:
            grid[r, c] = CUP
        blocked.update(choice["walls"])
        blocked.update(choice["ray_cells"])
        cups.append(choice)

    if len(_components_of_color(grid, CUP)) != 4:
        return None

    expected = _build_expected_output(grid, cups, marker_top_left)
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
