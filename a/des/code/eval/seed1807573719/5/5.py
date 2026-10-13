from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
AZURE = 8
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ARROW_STEPS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}

ARROW_PATTERNS = {
    frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}): ("right", (1, 2)),
    frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}): ("left", (1, 0)),
    frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}): ("down", (2, 1)),
    frozenset({(0, 1), (1, 1), (2, 1), (0, 0), (0, 2)}): ("up", (0, 1)),
}

COUNT_TO_COLOR = {
    0: AZURE,
    1: YELLOW,
    2: RED,
}

GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _classify_arrow(coords: np.ndarray) -> tuple[str, tuple[int, int], tuple[int, int]]:
    assert coords.shape == (5, 2), "arrow must contain exactly five pixels"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    normalized = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    assert normalized in ARROW_PATTERNS, "unexpected green component shape"
    direction, tip_offset = ARROW_PATTERNS[normalized]
    tip = (int(r0 + tip_offset[0]), int(c0 + tip_offset[1]))
    return direction, tip, ARROW_STEPS[direction]


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = step
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _between_cells(start: tuple[int, int], end: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = step
    cells: list[tuple[int, int]] = []
    r += dr
    c += dc
    while (r, c) != end:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _marker_positions(target: tuple[int, int]) -> dict[int, list[tuple[int, int]]]:
    r, c = target
    return {
        2: [(int(r), int(c))],
        1: [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))],
        0: [(int(r - 2), int(c)), (int(r + 2), int(c)), (int(r), int(c - 2)), (int(r), int(c + 2))],
    }


def _arrow_cells(direction: str, tip: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = tip
    if direction == "right":
        return [(r, c - 2), (r, c - 1), (r, c), (r - 1, c), (r + 1, c)]
    if direction == "left":
        return [(r, c), (r, c + 1), (r, c + 2), (r - 1, c), (r + 1, c)]
    if direction == "down":
        return [(r - 2, c), (r - 1, c), (r, c), (r, c - 1), (r, c + 1)]
    if direction == "up":
        return [(r, c), (r + 1, c), (r + 2, c), (r, c - 1), (r, c + 1)]
    raise AssertionError(f"unknown direction: {direction}")


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrow_components = _components(g == GREEN)
    assert arrow_components, "expected at least one arrow"

    arrows: list[dict[str, object]] = []
    for coords in arrow_components:
        direction, tip, step = _classify_arrow(coords)
        arrows.append(
            {
                "coords": coords,
                "direction": direction,
                "tip": tip,
                "step": step,
                "ray": set(_ray_cells(g.shape, tip, step)),
            }
        )

    common = set.intersection(*(arrow["ray"] for arrow in arrows))
    assert len(common) == 1, "arrows must converge at one target cell"
    target = next(iter(common))

    out = g.copy()
    groups_present: set[int] = set()
    for arrow in arrows:
        tip = arrow["tip"]
        step = arrow["step"]
        between = _between_cells(tip, target, step)
        count = sum(int(g[r, c] == MAROON) for r, c in between)
        assert 0 <= count <= 2, "obstacle count must be 0, 1, or 2"
        groups_present.add(int(count))
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = COUNT_TO_COLOR[int(count)]

    for group, positions in _marker_positions(target).items():
        color = COUNT_TO_COLOR[group] if group in groups_present else BLACK
        for cell in positions:
            if _in_bounds(g.shape, cell):
                out[cell] = int(color)

    return out


def _choose_directions(rng: np.random.Generator) -> list[str]:
    horizontal_options = [["right"], ["left"], ["right", "left"]]
    vertical_options = [["down"], ["up"], ["down", "up"]]
    directions = list(horizontal_options[int(rng.integers(len(horizontal_options)))])
    directions += list(vertical_options[int(rng.integers(len(vertical_options)))])
    rng.shuffle(directions)
    return directions


def _tip_for_direction(direction: str, target: tuple[int, int], distance: int) -> tuple[int, int]:
    r, c = target
    if direction == "right":
        return (int(r), int(c - distance))
    if direction == "left":
        return (int(r), int(c + distance))
    if direction == "down":
        return (int(r - distance), int(c))
    if direction == "up":
        return (int(r + distance), int(c))
    raise AssertionError(f"unknown direction: {direction}")


def _max_distance(direction: str, target: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = target
    if direction == "right":
        return int(c - 2)
    if direction == "left":
        return int(w - 3 - c)
    if direction == "down":
        return int(r - 2)
    if direction == "up":
        return int(h - 3 - r)
    raise AssertionError(f"unknown direction: {direction}")


def _pick_obstacles(
    rng: np.random.Generator,
    direction: str,
    tip: tuple[int, int],
    target: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    step = ARROW_STEPS[direction]
    between = _between_cells(tip, target, step)
    candidates = between[:-2]
    assert len(candidates) >= count, "insufficient room for requested obstacle count"
    if count == 0:
        return []
    if count == 1:
        idx = int(rng.integers(len(candidates)))
        return [candidates[idx]]

    spaced_pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates)) if j - i > 1]
    if spaced_pairs:
        i, j = spaced_pairs[int(rng.integers(len(spaced_pairs)))]
        return [candidates[i], candidates[j]]

    picks = rng.choice(len(candidates), size=2, replace=False)
    return [candidates[int(picks[0])], candidates[int(picks[1])]]


def _chebyshev_far_enough(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in occupied:
        if max(abs(int(r - rr)), abs(int(c - cc))) <= 1:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 27))
    w = int(rng.integers(18, 27))
    grid = np.full((h, w), BLACK, dtype=int)

    target = (int(rng.integers(8, h - 8)), int(rng.integers(8, w - 8)))
    directions = _choose_directions(rng)

    max_groups = min(3, len(directions))
    present_count = int(rng.integers(1, max_groups + 1))
    present_groups = list(rng.choice(np.array([0, 1, 2], dtype=int), size=present_count, replace=False))
    counts = present_groups[:]
    while len(counts) < len(directions):
        counts.append(int(present_groups[int(rng.integers(len(present_groups)))]))
    rng.shuffle(counts)

    occupied: set[tuple[int, int]] = set()
    obstacle_cells: set[tuple[int, int]] = set()
    arrows: list[dict[str, object]] = []

    for direction, count in zip(directions, counts):
        min_distance = 6 if int(count) == 2 else 5
        max_distance = _max_distance(direction, target, grid.shape)
        if max_distance < min_distance:
            return None
        distance = int(rng.integers(min_distance, max_distance + 1))
        tip = _tip_for_direction(direction, target, distance)
        arrow_cells = _arrow_cells(direction, tip)
        if not all(_in_bounds(grid.shape, cell) for cell in arrow_cells):
            return None
        if any(cell in occupied for cell in arrow_cells):
            return None

        obstacles = _pick_obstacles(rng, direction, tip, target, int(count))
        if any(cell in occupied for cell in obstacles):
            return None

        for cell in arrow_cells:
            occupied.add(cell)
            grid[cell] = GREEN
        for cell in obstacles:
            obstacle_cells.add(cell)
            occupied.add(cell)
            grid[cell] = MAROON

        arrows.append(
            {
                "direction": direction,
                "tip": tip,
                "count": int(count),
                "distance": int(distance),
            }
        )

    marker_forbidden = set()
    for positions in _marker_positions(target).values():
        marker_forbidden.update(positions)

    extra_count = int(rng.integers(0, 6))
    extra_candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (int(r), int(c))
            if cell in occupied or cell in marker_forbidden:
                continue
            if r == target[0] or c == target[1]:
                continue
            if not _chebyshev_far_enough(cell, occupied | marker_forbidden):
                continue
            extra_candidates.append(cell)

    if extra_candidates:
        rng.shuffle(extra_candidates)
        for cell in extra_candidates[:extra_count]:
            grid[cell] = MAROON
            occupied.add(cell)

    for arrow in arrows:
        direction = str(arrow["direction"])
        tip = tuple(int(v) for v in arrow["tip"])
        expected = int(arrow["count"])
        actual = sum(int(grid[r, c] == MAROON) for r, c in _between_cells(tip, target, ARROW_STEPS[direction]))
        if actual != expected:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
