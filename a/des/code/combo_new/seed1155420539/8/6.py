from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 0
SHAPE_COLOR = 1
MARKER_COLOR = 2
OVERLAP_COLOR = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_BY_SIDE = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 14
GRID_MAX = 26
TIME_RANGE = (1, 5)
TOTAL_SHAPES_RANGE = (2, 4)
GEN_ATTEMPTS = 400

BASE_SHAPES = [
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
]


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _rotate90(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(c, -r) for r, c in shape])


def _flip_lr(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(r, -c) for r, c in shape])


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _absolute_cells(shape: tuple[tuple[int, int], ...], origin: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = origin
    return [(int(r0 + r), int(c0 + c)) for r, c in shape]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords.sort()
            r0 = min(rr for rr, _ in coords)
            c0 = min(cc for _, cc in coords)
            r1 = max(rr for rr, _ in coords)
            c1 = max(cc for _, cc in coords)
            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "coord_set": set(coords),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return comps


def _marker_for_component(
    grid: np.ndarray,
    marker_color: int,
    comp: dict,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = comp["bbox"]
    coords = comp["coord_set"]
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for mr, mc in np.argwhere(grid == marker_color):
        r = int(mr)
        c = int(mc)
        if r == r0 - 1 and c0 <= c <= c1 and (r + 1, c) in coords:
            candidates.append(((r, c), DIR_BY_SIDE["up"]))
        if r == r1 + 1 and c0 <= c <= c1 and (r - 1, c) in coords:
            candidates.append(((r, c), DIR_BY_SIDE["down"]))
        if c == c0 - 1 and r0 <= r <= r1 and (r, c + 1) in coords:
            candidates.append(((r, c), DIR_BY_SIDE["left"]))
        if c == c1 + 1 and r0 <= r <= r1 and (r, c - 1) in coords:
            candidates.append(((r, c), DIR_BY_SIDE["right"]))
    unique = list(dict.fromkeys(candidates))
    assert len(unique) == 1, "each shape must have exactly one adjacent marker"
    return unique[0]


def _render_overlap(
    shape: tuple[int, int],
    positions: list[list[tuple[int, int]]],
    bg: int,
    shape_color: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    counts: Counter[tuple[int, int]] = Counter()
    for cells in positions:
        counts.update(cells)
    for (r, c), count in counts.items():
        out[r, c] = OVERLAP_COLOR if count >= 2 else int(shape_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide several same-colored shapes synchronously in the directions indicated by their adjacent markers and stop at the first overlap.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms several 4-connected shapes, and the shapes do not initially overlap.
    - A second foreground color forms one single-cell marker for each shape, placed just outside one side of that shape's bounding box.

    Rule:
    - Match each shape with its unique adjacent marker and read the movement direction from the marker side: above means up, below means down, left means left, and right means right.
    - Move all shapes one cell per step at the same time along their own directions.
    - Stop at the first step whose moved positions create any pixel overlap, erase all markers, and paint every overlap cell green while every other moved shape cell keeps the shape color.

    Key insight:
    - The stopping condition is global and synchronous: every shape freezes at the earliest step where any pair first overlaps, so even non-colliding shapes stop at that same instant.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    colors = [int(v) for v in np.unique(g) if int(v) != int(bg)]
    assert len(colors) == 2

    counts = Counter(map(int, g.flatten()))
    shape_color, marker_color = sorted(colors, key=lambda color: counts[color], reverse=True)

    comps = _find_components(g, shape_color)
    assert len(comps) >= 2

    positions: list[list[tuple[int, int]]] = []
    directions: list[tuple[int, int]] = []
    seen_markers: set[tuple[int, int]] = set()
    for comp in comps:
        marker, direction = _marker_for_component(g, marker_color, comp)
        assert marker not in seen_markers
        seen_markers.add(marker)
        positions.append([(int(r), int(c)) for r, c in comp["coords"]])
        directions.append(direction)

    max_steps = g.shape[0] + g.shape[1] + 5
    current = positions
    for _step in range(max_steps):
        moved: list[list[tuple[int, int]]] = []
        for cells, (dr, dc) in zip(current, directions):
            next_cells = [(int(r + dr), int(c + dc)) for r, c in cells]
            assert all(_in_bounds(g.shape, r, c) for r, c in next_cells), "shape left the grid before any overlap"
            moved.append(next_cells)
        counts_after_step: Counter[tuple[int, int]] = Counter()
        for cells in moved:
            counts_after_step.update(cells)
        if any(count >= 2 for count in counts_after_step.values()):
            return _render_overlap(g.shape, moved, bg, shape_color)
        current = moved

    raise AssertionError("no overlap found")


def _random_shape() -> tuple[tuple[int, int], ...]:
    shape = random.choice(BASE_SHAPES)
    if random.random() < 0.5:
        shape = _flip_lr(shape)
    for _ in range(random.randrange(4)):
        shape = _rotate90(shape)
    return _normalize(shape)


def _sample_distinct_shapes(count: int) -> list[tuple[tuple[int, int], ...]] | None:
    chosen: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(200):
        if len(chosen) == count:
            return chosen
        shape = _random_shape()
        if shape in seen:
            continue
        seen.add(shape)
        chosen.append(shape)
    return None


def _marker_cell(
    shape: tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0 = origin
    h, w = _shape_dims(shape)
    if side == "up":
        cols = [c for r, c in shape if r == 0]
        return int(r0 - 1), int(c0 + random.choice(cols))
    if side == "down":
        cols = [c for r, c in shape if r == h - 1]
        return int(r0 + h), int(c0 + random.choice(cols))
    if side == "left":
        rows = [r for r, c in shape if c == 0]
        return int(r0 + random.choice(rows)), int(c0 - 1)
    rows = [r for r, c in shape if c == w - 1]
    return int(r0 + random.choice(rows)), int(c0 + w)


def _render_input(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for placement in placements:
        for r, c in _absolute_cells(placement["shape"], placement["input_origin"]):
            grid[r, c] = SHAPE_COLOR
    for placement in placements:
        mr, mc = placement["marker"]
        grid[mr, mc] = MARKER_COLOR
    return grid


def _render_expected(shape: tuple[int, int], placements: list[dict], t: int) -> np.ndarray:
    final_positions: list[list[tuple[int, int]]] = []
    for placement in placements:
        dr, dc = placement["dir"]
        r0, c0 = placement["input_origin"]
        final_origin = (int(r0 + t * dr), int(c0 + t * dc))
        final_positions.append(_absolute_cells(placement["shape"], final_origin))
    return _render_overlap(shape, final_positions, BG_COLOR, SHAPE_COLOR)


def _validate_placements(shape: tuple[int, int], placements: list[dict], t: int) -> bool:
    markers: set[tuple[int, int]] = set()
    step0_cells: list[set[tuple[int, int]]] = []
    occupied0: set[tuple[int, int]] = set()

    for placement in placements:
        cells = set(_absolute_cells(placement["shape"], placement["input_origin"]))
        if not all(_in_bounds(shape, r, c) for r, c in cells):
            return False
        if occupied0 & cells:
            return False
        occupied0 |= cells
        step0_cells.append(cells)

    for idx, placement in enumerate(placements):
        marker = placement["marker"]
        if not _in_bounds(shape, marker[0], marker[1]):
            return False
        if marker in occupied0 or marker in markers:
            return False
        touches_own = any((marker[0] + dr, marker[1] + dc) in step0_cells[idx] for dr, dc in DIRS4)
        if not touches_own:
            return False
        for jdx, cells in enumerate(step0_cells):
            if jdx != idx and any((marker[0] + dr, marker[1] + dc) in cells for dr, dc in DIRS4):
                return False
        if any(max(abs(marker[0] - mr), abs(marker[1] - mc)) <= 1 for mr, mc in markers):
            return False
        markers.add(marker)

    overlap_cells = 0
    for step in range(t + 1):
        counts: Counter[tuple[int, int]] = Counter()
        for placement in placements:
            dr, dc = placement["dir"]
            r0, c0 = placement["input_origin"]
            origin = (int(r0 + step * dr), int(c0 + step * dc))
            cells = _absolute_cells(placement["shape"], origin)
            if not all(_in_bounds(shape, r, c) for r, c in cells):
                return False
            counts.update(cells)
        overlaps = [cell for cell, count in counts.items() if count >= 2]
        if step < t and overlaps:
            return False
        if step == t:
            overlap_cells = len(overlaps)

    return 1 <= overlap_cells <= 3


def _collider_dirs(kind: str) -> list[str]:
    if kind == "hpair":
        return ["right", "left"]
    if kind == "vpair":
        return ["down", "up"]
    if kind == "orth":
        return list(random.choice([("right", "down"), ("right", "up"), ("left", "down"), ("left", "up")]))
    return list(random.choice([("up", "left", "right"), ("down", "left", "right"), ("left", "up", "down"), ("right", "up", "down")]))


def _build_collision_group(
    grid_shape: tuple[int, int],
    t: int,
    shapes: list[tuple[tuple[int, int], ...]],
    kind: str,
) -> list[dict] | None:
    directions = _collider_dirs(kind)
    for _ in range(150):
        target = (
            random.randint(2, grid_shape[0] - 3),
            random.randint(2, grid_shape[1] - 3),
        )
        placements: list[dict] = []
        for shape, side in zip(shapes, directions):
            hit_cell = random.choice(shape)
            dr, dc = DIR_BY_SIDE[side]
            final_origin = (int(target[0] - hit_cell[0]), int(target[1] - hit_cell[1]))
            input_origin = (int(final_origin[0] - t * dr), int(final_origin[1] - t * dc))
            marker = _marker_cell(shape, input_origin, side)
            placements.append(
                {
                    "shape": shape,
                    "dir": (dr, dc),
                    "side": side,
                    "input_origin": input_origin,
                    "marker": marker,
                }
            )
        if _validate_placements(grid_shape, placements, t):
            return placements
    return None


def _input_origin_bounds(
    grid_shape: tuple[int, int],
    shape: tuple[tuple[int, int], ...],
    side: str,
    t: int,
) -> tuple[int, int, int, int] | None:
    h, w = _shape_dims(shape)
    r_lo, r_hi = 0, grid_shape[0] - h
    c_lo, c_hi = 0, grid_shape[1] - w
    dr, dc = DIR_BY_SIDE[side]

    if dr == -1:
        r_lo = max(r_lo, t, 1)
    elif dr == 1:
        r_hi = min(r_hi, grid_shape[0] - h - t, grid_shape[0] - h - 1)
    else:
        if side == "up":
            r_lo = max(r_lo, 1)
        elif side == "down":
            r_hi = min(r_hi, grid_shape[0] - h - 1)

    if dc == -1:
        c_lo = max(c_lo, t, 1)
    elif dc == 1:
        c_hi = min(c_hi, grid_shape[1] - w - t, grid_shape[1] - w - 1)
    else:
        if side == "left":
            c_lo = max(c_lo, 1)
        elif side == "right":
            c_hi = min(c_hi, grid_shape[1] - w - 1)

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return int(r_lo), int(r_hi), int(c_lo), int(c_hi)


def _add_extras(
    grid_shape: tuple[int, int],
    t: int,
    placements: list[dict],
    extra_shapes: list[tuple[tuple[int, int], ...]],
) -> list[dict] | None:
    out = list(placements)
    for shape in extra_shapes:
        placed = False
        sides = ["up", "down", "left", "right"]
        random.shuffle(sides)
        for _ in range(120):
            side = random.choice(sides)
            bounds = _input_origin_bounds(grid_shape, shape, side, t)
            if bounds is None:
                continue
            r_lo, r_hi, c_lo, c_hi = bounds
            input_origin = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
            marker = _marker_cell(shape, input_origin, side)
            trial = out + [
                {
                    "shape": shape,
                    "dir": DIR_BY_SIDE[side],
                    "side": side,
                    "input_origin": input_origin,
                    "marker": marker,
                }
            ]
            if _validate_placements(grid_shape, trial, t):
                out = trial
                placed = True
                break
        if not placed:
            return None
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid_shape = (h, w)
    t = random.randint(*TIME_RANGE)

    kind = random.choice(["hpair", "vpair", "orth", "triple"])
    collider_count = 3 if kind == "triple" else 2
    total_shapes = random.randint(max(TOTAL_SHAPES_RANGE[0], collider_count), TOTAL_SHAPES_RANGE[1])

    shapes = _sample_distinct_shapes(total_shapes)
    if shapes is None:
        return None

    collision_group = _build_collision_group(grid_shape, t, shapes[:collider_count], kind)
    if collision_group is None:
        return None

    placements = _add_extras(grid_shape, t, collision_group, shapes[collider_count:])
    if placements is None:
        return None

    inp = _render_input(grid_shape, placements)
    expected = _render_expected(grid_shape, placements, t)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid synchronized-overlap input")
