from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
OBSTACLE = 5
ARROW = 8
LAUNCH_COLORS = (1, 2, 3, 4, 6, 7, 9)

GEN_RETRIES = 400
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
OBSTACLE_COUNT_RANGE = (1, 3)
OBSTACLE_SIDE_RANGE = (3, 7)
ARROW_COUNT_RANGE = (2, 5)
MIN_PATH_LEN = 2
MIN_BAR_LEN = 3

ORIENTATIONS = {
    "up": {
        "dir": (-1, 0),
        "white": frozenset({(0, 0), (1, 0), (1, 1)}),
        "tip": (0, 0),
        "color": (2, 0),
    },
    "right": {
        "dir": (0, 1),
        "white": frozenset({(0, 0), (0, 1), (1, 0)}),
        "tip": (0, 1),
        "color": (0, -1),
    },
    "down": {
        "dir": (1, 0),
        "white": frozenset({(0, 0), (0, 1), (1, 1)}),
        "tip": (1, 1),
        "color": (-1, 1),
    },
    "left": {
        "dir": (0, -1),
        "white": frozenset({(0, 1), (1, 0), (1, 1)}),
        "tip": (1, 0),
        "color": (1, 2),
    },
}


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _is_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    return int(r) in (0, int(shape[0]) - 1) or int(c) in (0, int(shape[1]) - 1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(int(r), int(c))])
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if not _inside((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    queue.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _match_arrow(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    assert len(coords) == 3, "each arrow must use exactly three marker pixels"
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    assert (r1 - r0, c1 - c0) == (1, 1), "arrow marker must fit a 2x2 box"

    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    for spec in ORIENTATIONS.values():
        if norm != spec["white"]:
            continue
        tip_r = int(r0 + spec["tip"][0])
        tip_c = int(c0 + spec["tip"][1])
        color_r = int(r0 + spec["color"][0])
        color_c = int(c0 + spec["color"][1])
        assert _inside(grid.shape, color_r, color_c), "launch color must stay inside the grid"
        color = int(grid[color_r, color_c])
        assert color not in (BG, OBSTACLE, ARROW), "launch color must be distinct from bg, obstacle, and marker"
        return {
            "tip": (tip_r, tip_c),
            "dir": tuple(spec["dir"]),
            "color": color,
            "color_pos": (color_r, color_c),
            "white": [(int(r), int(c)) for r, c in coords],
        }
    raise AssertionError("unrecognized three-pixel arrow marker")


def _surface_bar(grid: np.ndarray, r: int, c: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    step = (0, 1) if dc == 0 else (1, 0)
    bar = [(int(r), int(c))]
    for sign in (-1, 1):
        for dist in range(1, 3):
            rr = int(r + sign * step[0] * dist)
            cc = int(c + sign * step[1] * dist)
            if not _inside(grid.shape, rr, cc) or int(grid[rr, cc]) != OBSTACLE:
                break
            bar.append((rr, cc))
    return bar


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project colored rays from arrow markers and flare them across obstacle faces on impact.

    Input structure:
    - The grid has a black background with one or more solid gray rectangular obstacles.
    - Each launcher is a three-pixel white arrow marker arranged as an L-shaped 2x2 triomino.
    - A single colored pixel sits just outside the arrow tail on the side opposite the tip and supplies the ray color.

    Rule:
    - Detect every white 3-pixel arrow, recover its pointing direction, and read the adjacent colored pixel behind its tail.
    - Starting just beyond the tip, extend a straight ray in that direction through black cells only.
    - Stop before painting the outer boundary cell; if the next cell is gray, replace the hit surface cell and up to two neighboring surface cells on each side along the perpendicular axis with the ray color.
    - Keep every original arrow marker and color key unchanged.

    Key insight:
    - The white triomino encodes direction, the colored tail-adjacent key encodes paint color, and obstacle collisions do not continue through the block but spread sideways across the struck face.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    arrows = [_match_arrow(g, comp) for comp in _components_of_color(g, ARROW)]
    assert arrows, "expected at least one arrow marker"

    out = g.copy()
    for arrow in sorted(arrows, key=lambda item: (item["tip"][0], item["tip"][1], item["dir"])):
        r, c = map(int, arrow["tip"])
        dr, dc = map(int, arrow["dir"])
        color = int(arrow["color"])

        while True:
            nr = int(r + dr)
            nc = int(c + dc)
            if not _inside((h, w), nr, nc):
                break
            if _is_boundary((h, w), nr, nc):
                break
            cell = int(g[nr, nc])
            if cell == OBSTACLE:
                for br, bc in _surface_bar(g, nr, nc, (dr, dc)):
                    out[br, bc] = color
                break
            assert cell == BG, "ray path must pass only through black background"
            out[nr, nc] = color
            r, c = nr, nc

    return out


def _rect_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, gap: int = 1) -> bool:
    rr0 = max(0, int(r0 - gap))
    cc0 = max(0, int(c0 - gap))
    rr1 = min(mask.shape[0] - 1, int(r1 + gap))
    cc1 = min(mask.shape[1] - 1, int(c1 + gap))
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_obstacles(grid: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    obstacles: list[tuple[int, int, int, int]] = []
    target = random.randint(*OBSTACLE_COUNT_RANGE)
    for _ in range(target):
        placed = False
        for _attempt in range(120):
            max_h = min(OBSTACLE_SIDE_RANGE[1], h - 4)
            max_w = min(OBSTACLE_SIDE_RANGE[1], w - 4)
            if max_h < OBSTACLE_SIDE_RANGE[0] or max_w < OBSTACLE_SIDE_RANGE[0]:
                return None
            oh = random.randint(OBSTACLE_SIDE_RANGE[0], max_h)
            ow = random.randint(OBSTACLE_SIDE_RANGE[0], max_w)
            r0 = random.randint(1, h - oh - 2)
            c0 = random.randint(1, w - ow - 2)
            r1 = int(r0 + oh - 1)
            c1 = int(c0 + ow - 1)
            if not _rect_clear(occupied, r0, c0, r1, c1, gap=1):
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            grid[r0 : r1 + 1, c0 : c1 + 1] = OBSTACLE
            obstacles.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return obstacles


def _candidate_cells(name: str, r0: int, c0: int) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    spec = ORIENTATIONS[name]
    whites = [(int(r0 + dr), int(c0 + dc)) for dr, dc in spec["white"]]
    tip = (int(r0 + spec["tip"][0]), int(c0 + spec["tip"][1]))
    color_pos = (int(r0 + spec["color"][0]), int(c0 + spec["color"][1]))
    return whites, tip, color_pos


def _cells_are_isolated(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    own = set((int(r), int(c)) for r, c in cells)
    for r, c in own:
        for nr in range(int(r) - 1, int(r) + 2):
            for nc in range(int(c) - 1, int(c) + 2):
                if not _inside(occupied.shape, nr, nc):
                    continue
                if (nr, nc) in own:
                    continue
                if occupied[nr, nc]:
                    return False
    return True


def _trace_ray(grid: np.ndarray, tip: tuple[int, int], direction: tuple[int, int]) -> dict | None:
    h, w = grid.shape
    dr, dc = map(int, direction)
    r, c = map(int, tip)
    path: list[tuple[int, int]] = []
    while True:
        nr = int(r + dr)
        nc = int(c + dc)
        if not _inside((h, w), nr, nc):
            return None
        if _is_boundary((h, w), nr, nc):
            if len(path) < MIN_PATH_LEN:
                return None
            return {"branch": "edge", "path": path, "bar": []}
        cell = int(grid[nr, nc])
        if cell == BG:
            path.append((nr, nc))
            r, c = nr, nc
            continue
        if cell == OBSTACLE:
            bar = _surface_bar(grid, nr, nc, (dr, dc))
            if len(path) < MIN_PATH_LEN or len(bar) < MIN_BAR_LEN:
                return None
            return {"branch": "hit", "path": path, "bar": bar}
        return None


def _enumerate_arrow_candidates(
    grid: np.ndarray,
    occupied_input: np.ndarray,
    reserved_output: set[tuple[int, int]],
    desired_branch: str,
) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for name, spec in ORIENTATIONS.items():
        for r0 in range(h - 1):
            for c0 in range(w - 1):
                whites, tip, color_pos = _candidate_cells(name, r0, c0)
                all_cells = whites + [color_pos]
                if any(not _inside((h, w), r, c) for r, c in all_cells):
                    continue
                if any(occupied_input[r, c] for r, c in all_cells):
                    continue
                if any((r, c) in reserved_output for r, c in all_cells):
                    continue
                if not _cells_are_isolated(occupied_input, all_cells):
                    continue

                trace = _trace_ray(grid, tip, spec["dir"])
                if trace is None or trace["branch"] != desired_branch:
                    continue

                draw_cells = set(trace["path"]) | set(trace["bar"])
                if draw_cells & reserved_output:
                    continue

                candidates.append(
                    {
                        "name": name,
                        "whites": whites,
                        "tip": tip,
                        "color_pos": color_pos,
                        "dir": tuple(spec["dir"]),
                        "path": list(trace["path"]),
                        "bar": list(trace["bar"]),
                        "branch": desired_branch,
                    }
                )
    random.shuffle(candidates)
    return candidates


def _apply_arrow(grid: np.ndarray, occupied_input: np.ndarray, candidate: dict, color: int) -> None:
    for r, c in candidate["whites"]:
        grid[r, c] = ARROW
        occupied_input[r, c] = True
    cr, cc = candidate["color_pos"]
    grid[cr, cc] = int(color)
    occupied_input[cr, cc] = True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    if _place_obstacles(grid) is None:
        return None

    occupied_input = grid != BG
    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    hit_count = random.randint(1, arrow_count - 1)
    branches = ["hit"] * hit_count + ["edge"] * (arrow_count - hit_count)
    random.shuffle(branches)

    palette = list(LAUNCH_COLORS)
    random.shuffle(palette)
    reserved_output: set[tuple[int, int]] = set()
    placed: list[dict] = []

    for idx, branch in enumerate(branches):
        candidates = _enumerate_arrow_candidates(grid, occupied_input, reserved_output, branch)
        if not candidates:
            return None

        candidate = None
        remaining = len(branches) - idx - 1
        for cand in candidates:
            test_axes = {abs(int(p["dir"][0])) for p in placed}
            test_axes.add(abs(int(cand["dir"][0])))
            if remaining == 0 and len(test_axes) < 2:
                continue
            candidate = cand
            break
        if candidate is None:
            candidate = candidates[0]

        color = palette[idx % len(palette)]
        _apply_arrow(grid, occupied_input, candidate, color)
        reserved_output.update(candidate["path"])
        reserved_output.update(candidate["bar"])
        placed.append({**candidate, "color": color})

    if {item["branch"] for item in placed} != {"hit", "edge"}:
        return None
    if len({abs(int(item["dir"][0])) for item in placed}) < 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray puzzle")
