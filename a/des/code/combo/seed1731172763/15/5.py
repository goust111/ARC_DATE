from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
MARKER = 5
OBSTACLE = 9
EMITTER_COLORS = (1, 2, 3, 4, 6, 7, 8)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (16, 22)
EMITTER_COUNT_RANGE = (1, 3)
BOUNCE_OPTIONS = (0, 1, 2, 3)
STRAIGHT_LEN_RANGE = (3, 7)
SEGMENT_LEN_RANGE = (2, 4)
OBSTACLE_LEN_RANGE = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
MAX_SYSTEM_ATTEMPTS = 160


Cell = tuple[int, int]
Vector = tuple[int, int]


@dataclass(frozen=True)
class Emitter:
    color: int
    start: Cell
    direction: Vector
    target: Cell


@dataclass(frozen=True)
class SystemPlan:
    color: int
    start: Cell
    direction: Vector
    block: tuple[Cell, ...]
    marker: Cell
    target: Cell
    path: tuple[Cell, ...]
    obstacles: tuple[Cell, ...]

    @property
    def visible_cells(self) -> set[Cell]:
        return set(self.block) | {self.marker, self.target} | set(self.obstacles)

    @property
    def reserved_cells(self) -> set[Cell]:
        return set(self.block) | set(self.path) | set(self.obstacles)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> tuple[Cell, Cell, Cell, Cell]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Cell]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Cell]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Cell] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _is_filled_2x2(coords: list[Cell]) -> bool:
    if len(coords) != 4:
        return False
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return (r1 == r0 + 1) and (c1 == c0 + 1) and set(coords) == {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    }


def _marker_for_block(grid: np.ndarray, coords: list[Cell]) -> tuple[Cell, Cell, Vector]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    candidates: list[tuple[Cell, Cell, Vector]] = []
    if r0 > 0:
        for c in range(c0, c1 + 1):
            if int(grid[r0 - 1, c]) == MARKER:
                candidates.append(((r0 - 1, c), (r0, c), UP))
    if r1 + 1 < grid.shape[0]:
        for c in range(c0, c1 + 1):
            if int(grid[r1 + 1, c]) == MARKER:
                candidates.append(((r1 + 1, c), (r1, c), DOWN))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if int(grid[r, c0 - 1]) == MARKER:
                candidates.append(((r, c0 - 1), (r, c0), LEFT))
    if c1 + 1 < grid.shape[1]:
        for r in range(r0, r1 + 1):
            if int(grid[r, c1 + 1]) == MARKER:
                candidates.append(((r, c1 + 1), (r, c1), RIGHT))
    if len(candidates) != 1:
        raise AssertionError("each emitter must have exactly one aligned grey marker")
    return candidates[0]


def _extract_emitters(grid: np.ndarray) -> list[Emitter]:
    emitters: list[Emitter] = []
    for color in EMITTER_COLORS:
        comps = _components_of_color(grid, color)
        emitter_blocks = [comp for comp in comps if _is_filled_2x2(comp)]
        targets = [comp for comp in comps if len(comp) == 1]
        if not comps:
            continue
        if len(emitter_blocks) != 1 or len(targets) != 1 or len(comps) != 2:
            raise AssertionError("each active color must form one 2x2 emitter and one single-pixel target")
        _marker, start, direction = _marker_for_block(grid, emitter_blocks[0])
        target = targets[0][0]
        emitters.append(Emitter(color=color, start=start, direction=direction, target=target))
    if not emitters:
        raise AssertionError("expected at least one emitter-target pair")
    return sorted(emitters, key=lambda item: (item.color, item.start, item.target))


def _turn_perpendicular(direction: Vector) -> Vector:
    dr, dc = direction
    return (dc, dr)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each 2x2 emitter with the same-color single-pixel target and trace a ray.

    Grey pixels mark the emitting side of a solid 2x2 block. The aligned side cell
    inside the block is the launch point, and the erased grey marker becomes the
    first traversable cell immediately outside the block. Rays turn perpendicularly
    when the next cell is maroon and stop when they reach their same-color target.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    emitters = _extract_emitters(g)
    base = g.copy()
    base[base == MARKER] = BG
    out = g.copy()
    out[out == MARKER] = BG

    for emitter in emitters:
        r, c = emitter.start
        dr, dc = emitter.direction
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, dr, dc)
            if state in seen:
                raise AssertionError("ray path must not loop")
            seen.add(state)

            nr, nc = r + dr, c + dc
            if not _in_bounds(base.shape, nr, nc):
                break

            if int(base[nr, nc]) == OBSTACLE:
                dr, dc = _turn_perpendicular((dr, dc))
                continue

            r, c = nr, nc
            if int(out[r, c]) == BG:
                out[r, c] = emitter.color
            if (r, c) == emitter.target:
                out[r, c] = emitter.color
                break

    return out


def _block_offsets(direction: Vector, anchor: int) -> tuple[Cell, ...]:
    if direction == UP:
        return ((0, 0), (0, 1), (1, 0), (1, 1)) if anchor == 0 else ((0, 0), (0, -1), (1, 0), (1, -1))
    if direction == DOWN:
        return ((0, 0), (0, 1), (-1, 0), (-1, 1)) if anchor == 0 else ((0, 0), (0, -1), (-1, 0), (-1, -1))
    if direction == LEFT:
        return ((0, 0), (1, 0), (0, 1), (1, 1)) if anchor == 0 else ((0, 0), (-1, 0), (0, 1), (-1, 1))
    if direction == RIGHT:
        return ((0, 0), (1, 0), (0, -1), (1, -1)) if anchor == 0 else ((0, 0), (-1, 0), (0, -1), (-1, -1))
    raise AssertionError("unexpected direction")


def _absolute_cells(start: Cell, rel_cells: tuple[Cell, ...] | list[Cell] | set[Cell]) -> tuple[Cell, ...]:
    sr, sc = start
    return tuple((sr + dr, sc + dc) for dr, dc in rel_cells)


def _build_relative_plan(direction: Vector, anchor: int, segment_lengths: list[int], obstacle_lengths: list[int]) -> dict[str, object]:
    block = _block_offsets(direction, anchor)
    current = (0, 0)
    path: list[Cell] = []
    obstacles: list[Cell] = []
    step_dir = direction

    for idx, seg_len in enumerate(segment_lengths):
        for _ in range(seg_len):
            current = (current[0] + step_dir[0], current[1] + step_dir[1])
            path.append(current)
        if idx < len(obstacle_lengths):
            obs_len = obstacle_lengths[idx]
            for step in range(1, obs_len + 1):
                obstacles.append((current[0] + step_dir[0] * step, current[1] + step_dir[1] * step))
            step_dir = _turn_perpendicular(step_dir)

    if len(set(path)) != len(path):
        raise AssertionError("constructed path must be simple")
    if set(path) & set(obstacles):
        raise AssertionError("path and obstacles must be disjoint")

    marker = path[0]
    target = path[-1]
    rel_cells = list(block) + path + obstacles
    rel_rows = [r for r, _ in rel_cells]
    rel_cols = [c for _, c in rel_cells]
    return {
        "block": block,
        "marker": marker,
        "target": target,
        "path": tuple(path),
        "obstacles": tuple(obstacles),
        "min_r": min(rel_rows),
        "max_r": max(rel_rows),
        "min_c": min(rel_cols),
        "max_c": max(rel_cols),
    }


def _halo(cells: set[Cell], shape: tuple[int, int]) -> set[Cell]:
    h, w = shape
    out: set[Cell] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _sample_segment_lengths(bounces: int) -> tuple[list[int], list[int]]:
    if bounces == 0:
        return [random.randint(*STRAIGHT_LEN_RANGE)], []
    segments = [random.randint(*SEGMENT_LEN_RANGE) for _ in range(bounces + 1)]
    obstacles = [random.randint(*OBSTACLE_LEN_RANGE) for _ in range(bounces)]
    return segments, obstacles


def _place_system(
    shape: tuple[int, int],
    color: int,
    blocked_cells: set[Cell],
    visible_halo: set[Cell],
) -> SystemPlan | None:
    h, w = shape
    for _ in range(MAX_SYSTEM_ATTEMPTS):
        direction = random.choice(DIRECTIONS)
        anchor = random.randint(0, 1)
        bounces = random.choice(BOUNCE_OPTIONS)
        segment_lengths, obstacle_lengths = _sample_segment_lengths(bounces)
        rel = _build_relative_plan(direction, anchor, segment_lengths, obstacle_lengths)

        r_lo = -int(rel["min_r"])
        r_hi = h - 1 - int(rel["max_r"])
        c_lo = -int(rel["min_c"])
        c_hi = w - 1 - int(rel["max_c"])
        if r_lo > r_hi or c_lo > c_hi:
            continue

        starts = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        random.shuffle(starts)
        for start in starts:
            block = _absolute_cells(start, rel["block"])
            marker = _absolute_cells(start, [rel["marker"]])[0]
            target = _absolute_cells(start, [rel["target"]])[0]
            path = _absolute_cells(start, rel["path"])
            obstacles = _absolute_cells(start, rel["obstacles"])

            plan = SystemPlan(
                color=color,
                start=start,
                direction=direction,
                block=block,
                marker=marker,
                target=target,
                path=path,
                obstacles=obstacles,
            )
            reserved = plan.reserved_cells
            if reserved & blocked_cells:
                continue
            if reserved & visible_halo:
                continue
            return plan
    return None


def _render_input(shape: tuple[int, int], plans: list[SystemPlan]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for plan in plans:
        for r, c in plan.block:
            grid[r, c] = plan.color
        grid[plan.marker] = MARKER
        grid[plan.target] = plan.color
        for r, c in plan.obstacles:
            grid[r, c] = OBSTACLE
    return grid


def _expected_output(inp: np.ndarray, plans: list[SystemPlan]) -> np.ndarray:
    out = inp.copy()
    out[out == MARKER] = BG
    for plan in plans:
        for r, c in plan.path:
            out[r, c] = plan.color
    return out


def _generate_case() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    max_emitters = EMITTER_COUNT_RANGE[1] if min(h, w) >= 18 else 2
    emitter_count = random.randint(EMITTER_COUNT_RANGE[0], max_emitters)
    colors = random.sample(EMITTER_COLORS, emitter_count)

    plans: list[SystemPlan] = []
    blocked_cells: set[Cell] = set()
    visible_halo: set[Cell] = set()

    for color in colors:
        plan = _place_system((h, w), color, blocked_cells, visible_halo)
        if plan is None:
            return None
        plans.append(plan)
        blocked_cells |= plan.reserved_cells
        visible_halo |= _halo(plan.visible_cells, (h, w))

    inp = _render_input((h, w), plans)
    expected = _expected_output(inp, plans)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid emitter-ray puzzle")
