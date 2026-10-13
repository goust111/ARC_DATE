from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 8)

DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}

WALL_BASE = {
    "N": ((0, -1), (1, 0), (0, 1)),
    "S": ((0, -1), (-1, 0), (0, 1)),
    "W": ((-1, 0), (0, 1), (1, 0)),
    "E": ((-1, 0), (0, -1), (1, 0)),
}

WALL_EXTRAS = {
    "N": ((1, -1), (1, 1)),
    "S": ((-1, -1), (-1, 1)),
    "W": ((-1, 1), (1, 1)),
    "E": ((-1, -1), (1, -1)),
}

GRID_MIN = 12
GRID_MAX = 20
MAX_GENERATION_ATTEMPTS = 400
PAIR_ATTEMPTS = 160
FREE_BRANCH_ATTEMPTS = 300
SCENARIOS = ("none", "equal", "unequal")


@dataclass
class BranchRuntime:
    seed: tuple[int, int]
    direction_name: str
    color: int
    path: list[tuple[int, int]] = field(default_factory=list)
    dist_by_cell: dict[tuple[int, int], int] = field(default_factory=dict)
    active: bool = True
    paint_color: int = -1

    def __post_init__(self) -> None:
        if self.paint_color < 0:
            self.paint_color = int(self.color)

    @property
    def direction(self) -> tuple[int, int]:
        return DIRS[self.direction_name]


@dataclass(frozen=True)
class BranchSpec:
    seed: tuple[int, int]
    direction_name: str
    color: int
    extra_ids: tuple[int, ...]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _wall_cells(spec: BranchSpec) -> set[tuple[int, int]]:
    r, c = spec.seed
    cells = {(r + dr, c + dc) for dr, dc in WALL_BASE[spec.direction_name]}
    extra_offsets = WALL_EXTRAS[spec.direction_name]
    for idx in spec.extra_ids:
        dr, dc = extra_offsets[idx]
        cells.add((r + dr, c + dc))
    return cells


def _footprint(spec: BranchSpec) -> set[tuple[int, int]]:
    return {spec.seed} | _wall_cells(spec)


def _ray_cells(seed: tuple[int, int], direction_name: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction_name]
    r, c = seed
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _detect_branches(g: np.ndarray) -> list[BranchRuntime]:
    colored = np.argwhere((g != BG) & (g != GREY))
    branches: list[BranchRuntime] = []

    for r0, c0 in colored:
        r = int(r0)
        c = int(c0)
        if r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1:
            continue

        grey_dirs = []
        open_dirs = []
        for name, (dr, dc) in DIRS.items():
            nr = r + dr
            nc = c + dc
            if int(g[nr, nc]) == GREY:
                grey_dirs.append(name)
            else:
                open_dirs.append(name)

        if len(grey_dirs) == 3 and len(open_dirs) == 1:
            branches.append(
                BranchRuntime(seed=(r, c), direction_name=open_dirs[0], color=int(g[r, c]))
            )

    assert branches, "expected at least one seed"
    assert len(branches) == len(colored), "every colored pixel must be a valid seed"
    branches.sort(key=lambda branch: (branch.seed[0], branch.seed[1]))
    return branches


def _apply_collision(
    branches: list[BranchRuntime], a: int, a_len: int, b: int, b_len: int
) -> str:
    color_a = branches[a].paint_color
    color_b = branches[b].paint_color

    if a_len < b_len:
        branches[a].paint_color = color_b
        kind = "unequal"
    elif b_len < a_len:
        branches[b].paint_color = color_a
        kind = "unequal"
    else:
        branches[a].paint_color = ORANGE
        branches[b].paint_color = ORANGE
        kind = "equal"

    branches[a].active = False
    branches[b].active = False
    return kind


def _simulate(
    grid: np.ndarray, return_meta: bool = False
) -> np.ndarray | tuple[np.ndarray, list[BranchRuntime], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    branches = _detect_branches(g)

    owner = np.full(g.shape, -1, dtype=int)
    maroon_cells: set[tuple[int, int]] = set()
    collisions: list[dict[str, object]] = []

    while any(branch.active for branch in branches):
        blocked: set[int] = set()
        empty_targets: dict[tuple[int, int], list[int]] = {}
        occupied_targets: dict[tuple[int, int], list[int]] = {}

        for idx, branch in enumerate(branches):
            if not branch.active:
                continue

            step = len(branch.path) + 1
            dr, dc = branch.direction
            nr = branch.seed[0] + dr * step
            nc = branch.seed[1] + dc * step

            if not _in_bounds(g.shape, nr, nc):
                blocked.add(idx)
                continue
            if int(g[nr, nc]) != BG:
                blocked.add(idx)
                continue

            target = (nr, nc)
            occupier = int(owner[target])
            if occupier == -1:
                empty_targets.setdefault(target, []).append(idx)
            else:
                occupied_targets.setdefault(target, []).append(idx)

        involved: set[int] = set()

        for cell in sorted(empty_targets):
            ids = sorted(empty_targets[cell])
            if len(ids) < 2:
                continue

            if len(ids) == 2:
                a, b = ids
                kind = _apply_collision(
                    branches,
                    a,
                    len(branches[a].path) + 1,
                    b,
                    len(branches[b].path) + 1,
                )
            else:
                for idx in ids:
                    branches[idx].paint_color = ORANGE
                    branches[idx].active = False
                kind = "equal"

            owner[cell] = -2
            maroon_cells.add(cell)
            involved.update(ids)
            collisions.append({"kind": kind, "cell": cell, "branches": tuple(ids)})

        for cell in sorted(occupied_targets):
            arrivals = [idx for idx in occupied_targets[cell] if idx not in involved]
            if not arrivals:
                continue

            occupier = int(owner[cell])
            if occupier < 0 or occupier in involved:
                for idx in arrivals:
                    branches[idx].active = False
                    involved.add(idx)
                continue

            if len(arrivals) == 1:
                idx = arrivals[0]
                occupier_len = branches[occupier].dist_by_cell[cell]
                arrival_len = len(branches[idx].path) + 1
                kind = _apply_collision(branches, occupier, occupier_len, idx, arrival_len)
                ids = (occupier, idx)
            else:
                ids = tuple(sorted(arrivals + [occupier]))
                for idx in ids:
                    branches[idx].paint_color = ORANGE
                    branches[idx].active = False
                kind = "equal"

            owner[cell] = -2
            maroon_cells.add(cell)
            involved.update(ids)
            collisions.append({"kind": kind, "cell": cell, "branches": ids})

        for idx in blocked:
            if idx not in involved:
                branches[idx].active = False

        for cell in sorted(empty_targets):
            ids = empty_targets[cell]
            if len(ids) != 1:
                continue
            idx = ids[0]
            if idx in blocked or idx in involved or not branches[idx].active:
                continue
            owner[cell] = idx
            branches[idx].path.append(cell)
            branches[idx].dist_by_cell[cell] = len(branches[idx].path)

    out = g.copy()
    for branch in branches:
        for r, c in branch.path:
            out[r, c] = branch.paint_color
    for r, c in maroon_cells:
        out[r, c] = MAROON

    if return_meta:
        return out, branches, collisions
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow straight branches from colored seeds using the open side of each grey wall,
    then merge colliding branches by the rule in the prompt.
    """
    return _simulate(grid)


def _random_extra_ids() -> tuple[int, ...]:
    count = random.randint(0, 2)
    if count == 0:
        return ()
    return tuple(sorted(random.sample((0, 1), k=count)))


def _spec_valid(spec: BranchSpec, shape: tuple[int, int]) -> bool:
    return all(_in_bounds(shape, r, c) for r, c in _footprint(spec))


def _place_collision_pair(
    shape: tuple[int, int], colors: list[int], mode: str
) -> list[BranchSpec] | None:
    h, w = shape

    for _ in range(PAIR_ATTEMPTS):
        hdir = random.choice(("E", "W"))
        vdir = random.choice(("N", "S"))

        if mode == "equal":
            dist_h = dist_v = random.randint(2, 5)
        else:
            dist_h = random.randint(2, 5)
            dist_v = random.randint(2, 5)
            if dist_h == dist_v:
                continue

        pr = random.randint(2, h - 3)
        pc = random.randint(2, w - 3)
        p = (pr, pc)

        hdr, hdc = DIRS[hdir]
        vdr, vdc = DIRS[vdir]
        seed_h = (pr - hdr * dist_h, pc - hdc * dist_h)
        seed_v = (pr - vdr * dist_v, pc - vdc * dist_v)

        spec_h = BranchSpec(seed=seed_h, direction_name=hdir, color=colors[0], extra_ids=_random_extra_ids())
        spec_v = BranchSpec(seed=seed_v, direction_name=vdir, color=colors[1], extra_ids=_random_extra_ids())

        if not (_spec_valid(spec_h, shape) and _spec_valid(spec_v, shape)):
            continue

        footprint_h = _footprint(spec_h)
        footprint_v = _footprint(spec_v)
        if footprint_h & footprint_v:
            continue

        path_h = set(_ray_cells(spec_h.seed, spec_h.direction_name, shape))
        path_v = set(_ray_cells(spec_v.seed, spec_v.direction_name, shape))
        if p not in path_h or p not in path_v:
            continue
        if path_h & path_v != {p}:
            continue
        if footprint_h & path_v or footprint_v & path_h:
            continue

        return [spec_h, spec_v]

    return None


def _place_free_branch(
    shape: tuple[int, int],
    used_fixed: set[tuple[int, int]],
    used_paths: set[tuple[int, int]],
    color: int,
) -> BranchSpec | None:
    h, w = shape

    for _ in range(FREE_BRANCH_ATTEMPTS):
        direction_name = random.choice(tuple(DIRS))
        seed = (random.randint(1, h - 2), random.randint(1, w - 2))
        spec = BranchSpec(seed=seed, direction_name=direction_name, color=color, extra_ids=_random_extra_ids())

        if not _spec_valid(spec, shape):
            continue

        footprint = _footprint(spec)
        path = set(_ray_cells(spec.seed, spec.direction_name, shape))
        if len(path) < 2:
            continue
        if footprint & used_fixed or footprint & used_paths:
            continue
        if path & used_fixed or path & used_paths:
            continue

        return spec

    return None


def _render_input(shape: tuple[int, int], specs: list[BranchSpec]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        for r, c in _wall_cells(spec):
            grid[r, c] = GREY
        sr, sc = spec.seed
        grid[sr, sc] = spec.color
    return grid


def _scenario_matches(grid: np.ndarray, mode: str, branch_count: int) -> bool:
    try:
        _out, branches, collisions = _simulate(grid, return_meta=True)
    except AssertionError:
        return False

    if len(branches) != branch_count:
        return False

    kinds = [str(record["kind"]) for record in collisions]
    if mode == "none":
        return not kinds
    if mode == "equal":
        return bool(kinds) and all(kind == "equal" for kind in kinds)
    return bool(kinds) and all(kind == "unequal" for kind in kinds)


def _generate_attempt(mode: str) -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    branch_count = random.randint(3, 5) if mode == "none" else random.randint(2, 5)
    colors = random.sample(SEED_COLORS, branch_count)

    specs: list[BranchSpec] = []
    used_fixed: set[tuple[int, int]] = set()
    used_paths: set[tuple[int, int]] = set()

    color_idx = 0
    if mode != "none":
        pair = _place_collision_pair(shape, colors[:2], mode)
        if pair is None:
            return None
        for spec in pair:
            specs.append(spec)
            used_fixed.update(_footprint(spec))
            used_paths.update(_ray_cells(spec.seed, spec.direction_name, shape))
        color_idx = 2

    while len(specs) < branch_count:
        spec = _place_free_branch(shape, used_fixed, used_paths, colors[color_idx])
        if spec is None:
            return None
        specs.append(spec)
        used_fixed.update(_footprint(spec))
        used_paths.update(_ray_cells(spec.seed, spec.direction_name, shape))
        color_idx += 1

    grid = _render_input(shape, specs)
    if _scenario_matches(grid, mode, len(specs)):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = random.choice(SCENARIOS)
        grid = _generate_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
