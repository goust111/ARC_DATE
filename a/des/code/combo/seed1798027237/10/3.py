from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _on_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _is_corner(shape: tuple[int, int], r: int, c: int) -> bool:
    return (r in (0, shape[0] - 1)) and (c in (0, shape[1] - 1))


def _is_interior(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 < r < shape[0] - 1 and 0 < c < shape[1] - 1


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comps.append({"color": color, "coords": coords})

    return comps


def _initial_direction(shape: tuple[int, int], start: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    assert _on_boundary(shape, r, c) and not _is_corner(shape, r, c), "endpoint must lie on exactly one edge"
    if r == 0:
        return (1, 0)
    if r == shape[0] - 1:
        return (-1, 0)
    if c == 0:
        return (0, 1)
    return (0, -1)


def _steps_to_boundary(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    if dr == 1:
        return shape[0] - 1 - r
    if dr == -1:
        return r
    if dc == 1:
        return shape[1] - 1 - c
    return c


def _orthogonals(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    assert (dr == 0) != (dc == 0)
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _parse_arrow(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    pts = set(coords)
    if len(pts) != 4:
        return None

    degrees = {
        p: sum((p[0] + dr, p[1] + dc) in pts for dr, dc in DIRS4)
        for p in pts
    }
    if sorted(degrees.values()) != [1, 1, 2, 2]:
        return None

    center = None
    for p, degree in degrees.items():
        if degree != 2:
            continue
        neighbors = [(p[0] + dr, p[1] + dc) for dr, dc in DIRS4 if (p[0] + dr, p[1] + dc) in pts]
        assert len(neighbors) == 2
        v0 = (neighbors[0][0] - p[0], neighbors[0][1] - p[1])
        v1 = (neighbors[1][0] - p[0], neighbors[1][1] - p[1])
        if v0 == (-v1[0], -v1[1]):
            continue
        if center is not None:
            return None
        center = p

    if center is None:
        return None

    center_neighbors = [
        (center[0] + dr, center[1] + dc)
        for dr, dc in DIRS4
        if (center[0] + dr, center[1] + dc) in pts
    ]
    tail_candidates = [p for p in center_neighbors if degrees[p] == 1]
    head_mid_candidates = [p for p in center_neighbors if degrees[p] == 2]
    if len(tail_candidates) != 1 or len(head_mid_candidates) != 1:
        return None

    tail = tail_candidates[0]
    head_mid = head_mid_candidates[0]
    in_dir = (center[0] - tail[0], center[1] - tail[1])
    out_dir = (head_mid[0] - center[0], head_mid[1] - center[1])
    head_tip = (head_mid[0] + out_dir[0], head_mid[1] + out_dir[1])
    if head_tip not in pts or degrees[head_tip] != 1:
        return None

    return {
        "tail": tail,
        "center": center,
        "head_mid": head_mid,
        "head_tip": head_tip,
        "in_dir": in_dir,
        "out_dir": out_dir,
        "cells": pts,
    }


def _trace(
    grid: np.ndarray,
    bg: int,
    arrows_by_tail: dict[tuple[int, int], dict[str, object]],
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]] | None:
    direction = _initial_direction(grid.shape, start)
    current = start
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (current[0], current[1], direction[0], direction[1])
        if state in seen:
            return None
        seen.add(state)

        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if not _in_bounds(grid.shape, nr, nc):
            return None

        nxt = (nr, nc)
        if nxt == target:
            return path

        value = int(grid[nr, nc])
        if value == bg:
            path.append(nxt)
            current = nxt
            continue

        arrow = arrows_by_tail.get(nxt)
        if arrow is None or tuple(arrow["in_dir"]) != direction:
            return None

        current = tuple(arrow["head_tip"])
        direction = tuple(arrow["out_dir"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the directed grey turn markers from the boundary source and paint the ray path.

    Input structure:
    - Black background.
    - Grey 4-pixel arrow markers shaped as an orthogonal path:
      one tail cell, one center cell, and a 2-cell head arm.
    - Two distinct singleton boundary pixels mark the source and terminal.

    Rule:
    - Launch a ray inward from the source edge cell.
    - Paint every background cell the ray traverses with the source color.
    - When the ray hits an arrow's tail from the encoded entry side, bend it through
      the arrow and continue in the encoded exit direction.
    - Stop when the ray reaches the other boundary singleton.
    - Leave the markers and the two boundary pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    components = _components(g, bg)

    endpoints: list[tuple[tuple[int, int], int]] = []
    arrows: list[dict[str, object]] = []
    arrow_colors: set[int] = set()

    for comp in components:
        color = int(comp["color"])
        coords = list(comp["coords"])
        if len(coords) == 1:
            cell = coords[0]
            assert _on_boundary(g.shape, *cell), "terminal markers must lie on the boundary"
            endpoints.append((cell, color))
            continue

        arrow = _parse_arrow(coords)
        assert arrow is not None, "every non-endpoint object must be a valid turn marker"
        arrows.append(arrow)
        arrow_colors.add(color)

    assert len(endpoints) == 2, "expected exactly two singleton boundary markers"
    assert endpoints[0][1] != endpoints[1][1], "source and target must use different colors"
    assert len(arrow_colors) == 1, "all turn markers must share one color"

    arrows_by_tail: dict[tuple[int, int], dict[str, object]] = {}
    for arrow in arrows:
        tail = tuple(arrow["tail"])
        assert tail not in arrows_by_tail, "turn markers must have distinct entry cells"
        arrows_by_tail[tail] = arrow

    valid_traces: list[tuple[int, list[tuple[int, int]]]] = []
    for i in range(2):
        path = _trace(g, bg, arrows_by_tail, endpoints[i][0], endpoints[1 - i][0])
        if path is not None:
            valid_traces.append((i, path))

    assert len(valid_traces) == 1, "exactly one boundary singleton must act as the source"
    source_index, path = valid_traces[0]
    source_color = int(endpoints[source_index][1])

    out = g.copy()
    for r, c in path:
        assert int(out[r, c]) == bg
        out[r, c] = source_color
    return out


# --------------- generator ---------------

BG = 0
ARROW = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (9, 17)
GRID_W_RANGE = (9, 17)
TURN_COUNT_RANGE = (1, 5)
MIN_STRAIGHT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
MIN_BACKGROUND_FRACTION = 0.55


def _random_source(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    edge = random.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        return (0, random.randint(1, w - 2))
    if edge == "bottom":
        return (h - 1, random.randint(1, w - 2))
    if edge == "left":
        return (random.randint(1, h - 2), 0)
    return (random.randint(1, h - 2), w - 1)


def _gray_clear(new_cells: set[tuple[int, int]], gray_cells: set[tuple[int, int]]) -> bool:
    for r, c in new_cells:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in gray_cells:
                return False
    return True


def _route_search(
    shape: tuple[int, int],
    pos: tuple[int, int],
    direction: tuple[int, int],
    turns_left: int,
    occupied: set[tuple[int, int]],
    gray_cells: set[tuple[int, int]],
    arrows: list[dict[str, object]],
) -> dict[str, object] | None:
    dist = _steps_to_boundary(shape, pos, direction)

    if turns_left == 0:
        if dist < MIN_STRAIGHT_GAP + 1:
            return None
        segment = [
            (pos[0] + direction[0] * step, pos[1] + direction[1] * step)
            for step in range(1, dist + 1)
        ]
        target = segment[-1]
        if _is_corner(shape, *target):
            return None
        if any(cell in occupied for cell in segment):
            return None
        return {
            "target": target,
            "occupied": occupied | set(segment),
            "gray_cells": gray_cells,
            "arrows": arrows,
        }

    max_gap = dist - 3
    if max_gap < MIN_STRAIGHT_GAP:
        return None

    gap_choices = list(range(MIN_STRAIGHT_GAP, max_gap + 1))
    random.shuffle(gap_choices)
    turn_dirs = _orthogonals(direction)

    for gap in gap_choices:
        bg_cells = [
            (pos[0] + direction[0] * step, pos[1] + direction[1] * step)
            for step in range(1, gap + 1)
        ]
        tail = (pos[0] + direction[0] * (gap + 1), pos[1] + direction[1] * (gap + 1))
        center = (pos[0] + direction[0] * (gap + 2), pos[1] + direction[1] * (gap + 2))

        if any(cell in occupied for cell in bg_cells):
            continue
        if tail in occupied or center in occupied:
            continue
        if not _is_interior(shape, *tail) or not _is_interior(shape, *center):
            continue

        dir_choices = list(turn_dirs)
        random.shuffle(dir_choices)
        for new_dir in dir_choices:
            head_mid = (center[0] + new_dir[0], center[1] + new_dir[1])
            head_tip = (center[0] + 2 * new_dir[0], center[1] + 2 * new_dir[1])
            new_gray = {tail, center, head_mid, head_tip}

            if not all(_is_interior(shape, *cell) for cell in new_gray):
                continue
            if any(cell in occupied for cell in new_gray):
                continue
            if not _gray_clear(new_gray, gray_cells):
                continue

            new_arrow = {
                "tail": tail,
                "center": center,
                "head_mid": head_mid,
                "head_tip": head_tip,
                "in_dir": direction,
                "out_dir": new_dir,
                "cells": new_gray,
            }
            result = _route_search(
                shape,
                head_tip,
                new_dir,
                turns_left - 1,
                occupied | set(bg_cells) | new_gray,
                gray_cells | new_gray,
                arrows + [new_arrow],
            )
            if result is not None:
                return result

    return None


def _sample_route(shape: tuple[int, int]) -> dict[str, object] | None:
    source = _random_source(shape)
    start_dir = _initial_direction(shape, source)
    turn_cap = min(TURN_COUNT_RANGE[1], max(TURN_COUNT_RANGE[0], (shape[0] + shape[1]) // 6))
    turn_count = random.randint(TURN_COUNT_RANGE[0], turn_cap)

    route = _route_search(
        shape,
        source,
        start_dir,
        turn_count,
        occupied={source},
        gray_cells=set(),
        arrows=[],
    )
    if route is None:
        return None
    route["source"] = source
    return route


def _build_grid(shape: tuple[int, int], route: dict[str, object], source_color: int, target_color: int) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for arrow in route["arrows"]:
        for r, c in arrow["cells"]:
            grid[r, c] = ARROW
    sr, sc = route["source"]
    tr, tc = route["target"]
    grid[sr, sc] = source_color
    grid[tr, tc] = target_color
    return grid


def _expected_output(grid: np.ndarray, route: dict[str, object], source_color: int) -> np.ndarray:
    out = grid.copy()
    special = set(route["gray_cells"]) | {route["source"], route["target"]}
    for cell in route["occupied"]:
        if cell in special:
            continue
        out[cell] = source_color
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    route = _sample_route(shape)
    if route is None:
        return None

    source_color, target_color = random.sample(PALETTE, 2)
    grid = _build_grid(shape, route, source_color, target_color)
    if float(np.mean(grid == BG)) < MIN_BACKGROUND_FRACTION:
        return None

    expected = _expected_output(grid, route, source_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
