from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIR_ORDER = (UP, RIGHT, DOWN, LEFT)

GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Wall:
    wall_id: int
    orientation: str  # "H" or "V"
    grey_cells: tuple[tuple[int, int], ...]
    markers: tuple[tuple[int, int], tuple[int, int]]
    out_dirs: tuple[tuple[int, int], tuple[int, int]]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _buffer(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _grey_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR_ORDER:
                    nr = cr + dr
                    nc = cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_walls(grid: np.ndarray) -> tuple[list[Wall], dict[tuple[int, int], int], set[tuple[int, int]]]:
    h, w = grid.shape
    walls: list[Wall] = []
    grey_to_wall: dict[tuple[int, int], int] = {}
    marker_cells: set[tuple[int, int]] = set()

    for comp in _grey_components(grid):
        assert len(comp) >= 2, "relay wall must contain at least two grey cells"
        rows = {r for r, _c in comp}
        cols = {c for _r, c in comp}

        if len(rows) == 1:
            r = next(iter(rows))
            c0 = min(cols)
            c1 = max(cols)
            assert len(comp) == c1 - c0 + 1, "horizontal wall grey cells must be contiguous"
            assert 0 < c0 and c1 < w - 1, "horizontal wall must leave room for endpoint markers"
            left = (r, c0 - 1)
            right = (r, c1 + 1)
            assert int(grid[left]) not in (BLACK, GREY)
            assert int(grid[right]) not in (BLACK, GREY)
            wall = Wall(
                wall_id=len(walls),
                orientation="H",
                grey_cells=tuple(sorted(comp)),
                markers=(left, right),
                out_dirs=(LEFT, RIGHT),
            )
        else:
            assert len(cols) == 1, "relay wall grey component must be a straight line"
            c = next(iter(cols))
            r0 = min(rows)
            r1 = max(rows)
            assert len(comp) == r1 - r0 + 1, "vertical wall grey cells must be contiguous"
            assert 0 < r0 and r1 < h - 1, "vertical wall must leave room for endpoint markers"
            top = (r0 - 1, c)
            bottom = (r1 + 1, c)
            assert int(grid[top]) not in (BLACK, GREY)
            assert int(grid[bottom]) not in (BLACK, GREY)
            wall = Wall(
                wall_id=len(walls),
                orientation="V",
                grey_cells=tuple(sorted(comp)),
                markers=(top, bottom),
                out_dirs=(UP, DOWN),
            )

        for cell in wall.grey_cells:
            assert cell not in grey_to_wall
            grey_to_wall[cell] = wall.wall_id
        for marker in wall.markers:
            assert marker not in marker_cells, "wall markers must be unique"
            marker_cells.add(marker)
        walls.append(wall)

    return walls, grey_to_wall, marker_cells


def _detect_towers(grid: np.ndarray, marker_cells: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    towers: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (BLACK, GREY) or (r, c) in marker_cells:
                continue
            towers.append((r, c, value))
    towers.sort(key=lambda item: (item[0], item[1]))
    return towers


def _nearest_marker(wall: Wall, hit: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if wall.orientation == "H":
        dist0 = abs(hit[1] - wall.markers[0][1])
        dist1 = abs(hit[1] - wall.markers[1][1])
    else:
        dist0 = abs(hit[0] - wall.markers[0][0])
        dist1 = abs(hit[0] - wall.markers[1][0])
    assert dist0 != dist1, "generator must avoid ambiguous nearest-endpoint hits"
    idx = 0 if dist0 < dist1 else 1
    return wall.markers[idx], wall.out_dirs[idx]


def _simulate(grid: np.ndarray, need_stats: bool = False) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    walls, grey_to_wall, marker_cells = _detect_walls(g)
    towers = _detect_towers(g, marker_cells)
    assert towers, "expected at least one signal tower"

    out = np.zeros_like(g)
    occupied = np.zeros_like(g, dtype=bool)
    stats = {
        "wall_hits": 0,
        "collision_stops": 0,
        "boundary_stops": 0,
        "reuse_stops": 0,
        "used_walls": set(),
        "painted_cells": 0,
    }

    def claim(cell: tuple[int, int], color: int) -> None:
        r, c = cell
        if not occupied[r, c]:
            occupied[r, c] = True
            out[r, c] = int(color)
            stats["painted_cells"] += 1

    def emit(start: tuple[int, int], direction: tuple[int, int], color: int, used_walls: frozenset[int]) -> None:
        claim(start, color)
        r, c = start
        dr, dc = direction

        while True:
            nr = r + dr
            nc = c + dc
            if not _in_bounds(g.shape, nr, nc):
                stats["boundary_stops"] += 1
                return
            if occupied[nr, nc]:
                stats["collision_stops"] += 1
                return

            wall_id = grey_to_wall.get((nr, nc))
            if wall_id is not None:
                wall = walls[wall_id]
                expected = "H" if dr != 0 else "V"
                assert wall.orientation == expected, "relay hits are expected to meet walls orthogonally"
                if wall_id in used_walls:
                    stats["reuse_stops"] += 1
                    return
                marker, out_dir = _nearest_marker(wall, (nr, nc))
                stats["wall_hits"] += 1
                stats["used_walls"].add(wall_id)
                emit(marker, out_dir, int(g[marker]), used_walls | frozenset({wall_id}))
                return

            claim((nr, nc), color)
            r, c = nr, nc

    for r, c, color in towers:
        for direction in DIR_ORDER:
            emit((r, c), direction, color, frozenset())

    return out, stats if need_stats else {}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace tower rays through capped grey relay walls.

    A relay wall is a straight grey segment whose two line endpoints are marked by
    colored pixels on the same axis. A ray that reaches a grey wall cell jumps to
    the nearer endpoint marker, changes to that marker's color, and continues away
    from the grey segment along the wall axis.
    """
    return _simulate(grid, need_stats=False)[0]


@dataclass
class Builder:
    h: int
    w: int
    grid: np.ndarray
    occupied: np.ndarray
    forbidden: np.ndarray
    ray_reserved: np.ndarray

    @classmethod
    def create(cls, h: int, w: int) -> "Builder":
        return cls(
            h=h,
            w=w,
            grid=np.zeros((h, w), dtype=int),
            occupied=np.zeros((h, w), dtype=bool),
            forbidden=np.zeros((h, w), dtype=bool),
            ray_reserved=np.zeros((h, w), dtype=bool),
        )

    def clone(self) -> "Builder":
        return Builder(
            h=self.h,
            w=self.w,
            grid=self.grid.copy(),
            occupied=self.occupied.copy(),
            forbidden=self.forbidden.copy(),
            ray_reserved=self.ray_reserved.copy(),
        )

    def can_place_object(self, cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if not _in_bounds((self.h, self.w), r, c):
                return False
            if self.occupied[r, c] or self.forbidden[r, c] or self.ray_reserved[r, c]:
                return False
        return True

    def add_object(self, colored_cells: dict[tuple[int, int], int]) -> bool:
        cells = list(colored_cells.keys())
        if not self.can_place_object(cells):
            return False
        for (r, c), color in colored_cells.items():
            self.grid[r, c] = int(color)
            self.occupied[r, c] = True
        for r, c in _buffer(cells, (self.h, self.w)):
            self.forbidden[r, c] = True
        return True

    def add_tower(self, pos: tuple[int, int], color: int) -> bool:
        return self.add_object({pos: int(color)})

    def add_horizontal_wall(self, row: int, c0: int, c1: int, left_color: int, right_color: int) -> bool:
        cells = {(row, c): GREY for c in range(c0, c1 + 1)}
        cells[(row, c0 - 1)] = int(left_color)
        cells[(row, c1 + 1)] = int(right_color)
        return self.add_object(cells)

    def add_vertical_wall(self, col: int, r0: int, r1: int, top_color: int, bottom_color: int) -> bool:
        cells = {(r, col): GREY for r in range(r0, r1 + 1)}
        cells[(r0 - 1, col)] = int(top_color)
        cells[(r1 + 1, col)] = int(bottom_color)
        return self.add_object(cells)

    def reserve_path(self, cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if not _in_bounds((self.h, self.w), r, c):
                return False
            if self.occupied[r, c]:
                return False
        for r, c in cells:
            self.ray_reserved[r, c] = True
        return True


def _line_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r + dr * i, c + dc * i) for i in range(1, steps + 1)]


def _to_boundary(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    out: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return out
        out.append((r, c))


def _sample_color(exclude: set[int] | None = None) -> int:
    banned = set() if exclude is None else set(exclude)
    choices = [color for color in PALETTE if color not in banned]
    return int(random.choice(choices))


def _sample_wall_for_ray(
    builder: Builder,
    source: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[str, int, int, int, int, int], tuple[int, int], tuple[int, int]] | None:
    dr, dc = direction
    for _ in range(80):
        gap = random.randint(2, 6)
        hit_r = source[0] + dr * gap
        hit_c = source[1] + dc * gap
        if not _in_bounds((builder.h, builder.w), hit_r, hit_c):
            continue

        path_cells = _line_cells(source, direction, gap - 1)
        if any(builder.occupied[r, c] for r, c in path_cells):
            continue

        if dr != 0:
            out_dir = random.choice((LEFT, RIGHT))
            if out_dir == LEFT:
                left_dist = random.randint(1, 3)
                right_dist = random.randint(left_dist + 1, left_dist + 4)
            else:
                right_dist = random.randint(1, 3)
                left_dist = random.randint(right_dist + 1, right_dist + 4)

            c0 = hit_c - (left_dist - 1)
            c1 = hit_c + (right_dist - 1)
            if c0 <= 0 or c1 >= builder.w - 1:
                continue
            left_color = _sample_color()
            right_color = _sample_color()
            cells = [(hit_r, c) for c in range(c0 - 1, c1 + 2)]
            if not builder.can_place_object(cells):
                continue
            if any(builder.occupied[hit_r, c] for c in range(c0, c1 + 1)):
                continue

            marker = (hit_r, c0 - 1) if out_dir == LEFT else (hit_r, c1 + 1)
            spec = ("H", hit_r, c0, c1, left_color, right_color)
            return path_cells, spec, marker, out_dir

        out_dir = random.choice((UP, DOWN))
        if out_dir == UP:
            top_dist = random.randint(1, 3)
            bottom_dist = random.randint(top_dist + 1, top_dist + 4)
        else:
            bottom_dist = random.randint(1, 3)
            top_dist = random.randint(bottom_dist + 1, bottom_dist + 4)

        r0 = hit_r - (top_dist - 1)
        r1 = hit_r + (bottom_dist - 1)
        if r0 <= 0 or r1 >= builder.h - 1:
            continue
        top_color = _sample_color()
        bottom_color = _sample_color()
        cells = [(r, hit_c) for r in range(r0 - 1, r1 + 2)]
        if not builder.can_place_object(cells):
            continue
        if any(builder.occupied[r, hit_c] for r in range(r0, r1 + 1)):
            continue

        marker = (r0 - 1, hit_c) if out_dir == UP else (r1 + 1, hit_c)
        spec = ("V", hit_c, r0, r1, top_color, bottom_color)
        return path_cells, spec, marker, out_dir

    return None


def _build_tower_chain(
    builder: Builder,
    pos: tuple[int, int],
    color: int,
    start_dir: tuple[int, int],
    chain_len: int,
) -> tuple[Builder, dict] | None:
    work = builder.clone()
    if not work.add_tower(pos, color):
        return None

    path_cells = [pos]
    source = pos
    direction = start_dir

    for _ in range(chain_len):
        sampled = _sample_wall_for_ray(work, source, direction)
        if sampled is None:
            return None
        seg_cells, spec, marker, next_dir = sampled
        if not work.reserve_path(seg_cells):
            return None

        if spec[0] == "H":
            _, row, c0, c1, left_color, right_color = spec
            if not work.add_horizontal_wall(row, c0, c1, left_color, right_color):
                return None
        else:
            _, col, r0, r1, top_color, bottom_color = spec
            if not work.add_vertical_wall(col, r0, r1, top_color, bottom_color):
                return None

        path_cells.extend(seg_cells)
        path_cells.append(marker)
        source = marker
        direction = next_dir

    tail = _to_boundary(source, direction, (work.h, work.w))
    if not work.reserve_path(tail):
        return None
    path_cells.extend(tail)
    return work, {"tower": pos, "path_cells": path_cells, "tail_dir": direction}


def _try_random_chain_tower(
    builder: Builder,
    color: int,
    chain_len: int,
    allowed_dirs: tuple[tuple[int, int], ...],
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
) -> tuple[Builder, dict] | None:
    r0, r1 = row_range if row_range is not None else (1, builder.h - 2)
    c0, c1 = col_range if col_range is not None else (1, builder.w - 2)
    if r0 > r1 or c0 > c1:
        return None

    for _ in range(120):
        pos = (random.randint(r0, r1), random.randint(c0, c1))
        start_dir = random.choice(allowed_dirs)
        result = _build_tower_chain(builder, pos, color, start_dir, chain_len)
        if result is not None:
            return result
    return None


def _add_collision_tower(
    builder: Builder,
    primary_info: dict,
    color: int,
) -> Builder | None:
    primary_tower = primary_info["tower"]
    candidates = [
        cell
        for cell in primary_info["path_cells"]
        if cell != primary_tower and not builder.occupied[cell]
        and (cell[0] > primary_tower[0] or (cell[0] == primary_tower[0] and cell[1] > primary_tower[1]))
    ]
    random.shuffle(candidates)

    for hit in candidates:
        for gap in range(2, 7):
            pos = (hit[0], hit[1] + gap)
            if not _in_bounds((builder.h, builder.w), pos[0], pos[1]):
                continue
            corridor = [(hit[0], c) for c in range(hit[1] + 1, pos[1])]
            if any(builder.occupied[r, c] for r, c in corridor):
                continue
            work = builder.clone()
            if not work.add_tower(pos, color):
                continue
            if not work.reserve_path(corridor):
                continue
            return work
    return None


def _add_random_tower_only(
    builder: Builder,
    color: int,
    row_range: tuple[int, int] | None = None,
) -> Builder | None:
    r0, r1 = row_range if row_range is not None else (1, builder.h - 2)
    for _ in range(100):
        pos = (random.randint(r0, r1), random.randint(1, builder.w - 2))
        work = builder.clone()
        if work.add_tower(pos, color):
            return work
    return None


def _add_distractor_wall(builder: Builder) -> Builder | None:
    for _ in range(120):
        work = builder.clone()
        if random.choice((True, False)):
            length = random.randint(2, 5)
            row = random.randint(1, work.h - 2)
            c0 = random.randint(1, work.w - length - 2)
            if work.add_horizontal_wall(row, c0, c0 + length - 1, _sample_color(), _sample_color()):
                return work
        else:
            length = random.randint(2, 5)
            col = random.randint(1, work.w - 2)
            r0 = random.randint(1, work.h - length - 2)
            if work.add_vertical_wall(col, r0, r0 + length - 1, _sample_color(), _sample_color()):
                return work
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    builder = Builder.create(h, w)

    num_towers = random.randint(2, 3)
    tower_colors = random.sample(PALETTE, num_towers)
    collision_mode = random.choice((True, False))

    primary_chain = random.choices([1, 2, 3], weights=[3, 4, 3], k=1)[0]
    primary_dirs = (DOWN, RIGHT) if collision_mode else DIR_ORDER
    primary = _try_random_chain_tower(
        builder,
        tower_colors[0],
        primary_chain,
        primary_dirs,
        row_range=(1, h // 2),
    )
    if primary is None:
        return None
    builder, primary_info = primary

    if collision_mode:
        placed = _add_collision_tower(builder, primary_info, tower_colors[1])
        if placed is None:
            return None
        builder = placed
    else:
        secondary_chain = random.choices([0, 1, 2], weights=[2, 3, 2], k=1)[0]
        secondary = _try_random_chain_tower(
            builder,
            tower_colors[1],
            secondary_chain,
            DIR_ORDER,
            row_range=(max(1, primary_info["tower"][0]), h - 2),
        )
        if secondary is None:
            return None
        builder, _secondary_info = secondary

    if num_towers == 3:
        if random.random() < 0.5:
            tertiary = _add_random_tower_only(builder, tower_colors[2], row_range=(h // 2, h - 2))
        else:
            tertiary = _try_random_chain_tower(
                builder,
                tower_colors[2],
                random.choice((0, 1)),
                DIR_ORDER,
                row_range=(h // 2, h - 2),
            )
        if tertiary is None:
            return None
        if isinstance(tertiary, tuple):
            builder = tertiary[0]
        else:
            builder = tertiary

    for _ in range(random.randint(0, 2)):
        if random.random() < 0.65:
            extra = _add_distractor_wall(builder)
            if extra is not None:
                builder = extra

    try:
        out, stats = _simulate(builder.grid, need_stats=True)
    except AssertionError:
        return None

    non_black_out = int(np.count_nonzero(out))
    if stats["wall_hits"] < 1:
        return None
    if collision_mode and stats["collision_stops"] < 1:
        return None
    if non_black_out < 18:
        return None
    if len(np.unique(out[out != BLACK])) < 2:
        return None

    walls, _grey_to_wall, marker_cells = _detect_walls(builder.grid)
    towers = _detect_towers(builder.grid, marker_cells)
    if len(walls) < 1 or len(towers) != num_towers:
        return None

    actual = main(builder.grid)
    if not np.array_equal(actual, out):
        return None
    return builder.grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-wall puzzle")
