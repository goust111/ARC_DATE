from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_task_6", arc_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {arc_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arc_utils = _load_arc_utils()
Color = _arc_utils.Color
find_components = _arc_utils.find_components
in_bounds = _arc_utils.in_bounds

BG = Color.BLACK
WATER = Color.BLUE
WALL = Color.GREY

GEN_RETRIES = 400


@dataclass(frozen=True)
class Container:
    r0: int
    c0: int
    r1: int
    c1: int
    gap_side: str | None
    gap_row: int | None

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    def interior_contains(self, r: int, c: int) -> bool:
        return self.r0 <= r <= self.r1 - 1 and self.c0 < c < self.c1

    def gap_cell(self) -> tuple[int, int] | None:
        if self.gap_side is None or self.gap_row is None:
            return None
        return (self.gap_row, self.c0 if self.gap_side == "L" else self.c1)

    def outflow_col(self) -> int | None:
        if self.gap_side is None:
            return None
        return self.c0 - 1 if self.gap_side == "L" else self.c1 + 1


def _paint_vertical(out: np.ndarray, col: int, start_row: int, end_row: int) -> None:
    if end_row < start_row:
        return
    for r in range(start_row, end_row + 1):
        assert in_bounds(out.shape, r, col), "vertical water left the grid"
        assert int(out[r, col]) != WALL, "water column collides with a wall"
        out[r, col] = WATER


def _find_first_container(containers: list[Container], col: int, start_row: int) -> tuple[Container | None, int | None]:
    matches = []
    for container in containers:
        if container.c0 < col < container.c1 and start_row <= container.r1 - 1:
            entry_row = max(start_row, container.r0)
            if entry_row <= container.r1 - 1:
                matches.append((entry_row, container))
    if not matches:
        return None, None
    matches.sort(key=lambda item: (item[0], item[1].r0, item[1].c0))
    entry_row = matches[0][0]
    tied = [container for row, container in matches if row == entry_row]
    assert len(tied) == 1, "drop column ambiguously enters multiple containers"
    return tied[0], entry_row


def _simulate_from_containers(grid: np.ndarray, containers: list[Container]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, _width = g.shape

    faucet_cols = np.flatnonzero(g[0] == WATER)
    assert len(faucet_cols) == 1, "expected exactly one blue faucet pixel on the top row"
    drop_col = int(faucet_cols[0])

    row = 1
    while row < height:
        container, entry_row = _find_first_container(containers, drop_col, row)
        if container is None:
            _paint_vertical(out, drop_col, row, height - 1)
            break

        assert entry_row is not None
        _paint_vertical(out, drop_col, row, entry_row)
        assert container.gap_side is not None and container.gap_row is not None, "reached container must have one side gap"

        surface_row = int(container.gap_row)
        assert container.r0 + 1 <= surface_row <= container.r1 - 1, "gap must be on a side wall cell"
        assert entry_row <= surface_row, "water entered below the leak level"

        if entry_row < surface_row:
            _paint_vertical(out, drop_col, entry_row, surface_row - 1)

        out[surface_row : container.r1, container.c0 + 1 : container.c1] = WATER
        gap_r, gap_c = container.gap_cell()
        out[gap_r, gap_c] = WATER

        drop_col = int(container.outflow_col())
        row = surface_row

    assert max(out.shape) <= 30
    return out


def _parse_container(grid: np.ndarray, comp: dict) -> Container | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return None
    if not np.all(patch[-1, :] == WALL):
        return None
    if int(patch[0, 0]) != WALL or int(patch[0, -1]) != WALL:
        return None
    if np.any(patch[:-1, 1:-1] == WALL):
        return None

    left_missing = [r0 + i for i in range(patch.shape[0]) if int(patch[i, 0]) != WALL]
    right_missing = [r0 + i for i in range(patch.shape[0]) if int(patch[i, -1]) != WALL]
    if any(r in (r0, r1) for r in left_missing + right_missing):
        return None

    gaps = [(r, "L") for r in left_missing] + [(r, "R") for r in right_missing]
    if len(gaps) > 1:
        return None

    gap_row = None
    gap_side = None
    if gaps:
        gap_row, gap_side = gaps[0]

    return Container(r0=r0, c0=c0, r1=r1, c1=c1, gap_side=gap_side, gap_row=gap_row)


def _detect_containers(grid: np.ndarray) -> list[Container]:
    g = np.asarray(grid, dtype=int)
    grey_comps = find_components(g, bg=BG, colors=[WALL], connectivity=4, monochromatic=True)
    containers: list[Container] = []
    for comp in grey_comps:
        container = _parse_container(g, comp)
        if container is not None:
            containers.append(container)
    assert containers, "expected at least one grey container"
    return sorted(containers, key=lambda box: (box.r0, box.c0, box.r1, box.c1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the steady cascading water route by filling each reached basin only up to its lowest side gap.

    Input structure:
    - The background is black, with one blue faucet pixel on the top row.
    - Grey objects are one-cell-thick upward-open containers; some side walls have exactly one missing gap cell.
    - Containers are vertically staggered, and only the containers that the water column can actually enter participate in the cascade.

    Rule:
    - Start a blue vertical stream directly below the faucet and continue until it enters the interior of a container, or to the bottom if none is reached.
    - Inside a reached container, fill all interior cells from the bottom up to the row of its unique side gap, also filling that gap cell.
    - Resume the next vertical stream from the cell immediately outside that gap, and repeat until the final stream reaches the bottom edge; untouched containers stay empty and walls stay grey.

    Key insight:
    - Water follows only one causally reachable chain, and each visited container stops filling at its leak row rather than at its rim or full capacity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    containers = _detect_containers(g)
    return _simulate_from_containers(g, containers)


def _box_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return max(ar0 - pad, br0 - pad) <= min(ar1 + pad, br1 + pad) and max(ac0 - pad, bc0 - pad) <= min(ac1 + pad, bc1 + pad)


def _sample_gap_row(rng: random.Random, r0: int, r1: int) -> int:
    choices = list(range(r0 + 1, r1))
    weighted = [choices[0], choices[-1], rng.choice(choices)]
    return int(rng.choice(weighted))


def _choose_path_count(rng: random.Random) -> int:
    return int(rng.choice([1, 2, 2, 3, 3, 4]))


def _draw_container(grid: np.ndarray, container: Container) -> None:
    grid[container.r1, container.c0 : container.c1 + 1] = WALL
    grid[container.r0 : container.r1 + 1, container.c0] = WALL
    grid[container.r0 : container.r1 + 1, container.c1] = WALL
    gap = container.gap_cell()
    if gap is not None:
        grid[gap] = BG


def _render_input(shape: tuple[int, int], faucet_col: int, containers: list[Container]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[0, faucet_col] = WATER
    for container in containers:
        _draw_container(grid, container)
    return grid


def _render_expected(shape: tuple[int, int], faucet_col: int, containers: list[Container]) -> np.ndarray:
    inp = _render_input(shape, faucet_col, containers)
    return _simulate_from_containers(inp, containers)


def _path_drop_columns(path_containers: list[Container], faucet_col: int) -> list[int]:
    cols = [int(faucet_col)]
    current = int(faucet_col)
    for container in path_containers:
        assert container.c0 < current < container.c1
        current = int(container.outflow_col())
        cols.append(current)
    return cols


def _build_path_containers(rng: random.Random, height: int, width: int, faucet_col: int) -> list[Container] | None:
    count = _choose_path_count(rng)
    containers: list[Container] = []
    drop_col = int(faucet_col)
    min_top = 2

    for i in range(count):
        remaining = count - i - 1
        min_h = 4
        h = rng.randint(4, 7)
        w = rng.randint(5, 9)
        min_needed = remaining * (min_h + 2) + 1
        max_r0 = height - min_needed - h
        if max_r0 < min_top:
            return None

        c0_min = max(0, drop_col - (w - 2))
        c0_max = min(drop_col - 1, width - w)
        if c0_min > c0_max:
            return None

        c0_options = list(range(c0_min, c0_max + 1))
        rng.shuffle(c0_options)

        placed = None
        for c0 in c0_options:
            c1 = c0 + w - 1
            sides = []
            if remaining > 0:
                if c0 - 1 >= 1:
                    sides.append("L")
                if c1 + 1 <= width - 2:
                    sides.append("R")
            else:
                if c0 - 1 >= 0:
                    sides.append("L")
                if c1 + 1 < width:
                    sides.append("R")
            rng.shuffle(sides)
            if not sides:
                continue

            r0 = rng.randint(min_top, max_r0)
            r1 = r0 + h - 1
            side = sides[0]
            gap_row = _sample_gap_row(rng, r0, r1)
            candidate = Container(r0=r0, c0=c0, r1=r1, c1=c1, gap_side=side, gap_row=gap_row)
            if any(_box_overlaps((candidate.r0, candidate.c0, candidate.r1, candidate.c1), (old.r0, old.c0, old.r1, old.c1), pad=1) for old in containers):
                continue
            placed = candidate
            break

        if placed is None:
            return None
        containers.append(placed)
        drop_col = int(placed.outflow_col())
        min_top = placed.r1 + 2

    return containers


def _try_place_distractor(
    rng: random.Random,
    height: int,
    width: int,
    forbidden_cols: list[int],
    existing: list[Container],
    allow_gap: bool,
) -> Container | None:
    for _ in range(120):
        h = rng.randint(4, 7)
        w = rng.randint(5, 9)
        r0 = rng.randint(2, height - h)
        c0 = rng.randint(0, width - w)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if any(c0 <= col <= c1 for col in forbidden_cols):
            continue

        gap_side = None
        gap_row = None
        if allow_gap and rng.random() < 0.55:
            gap_side = rng.choice(["L", "R"])
            gap_row = _sample_gap_row(rng, r0, r1)

        candidate = Container(r0=r0, c0=c0, r1=r1, c1=c1, gap_side=gap_side, gap_row=gap_row)
        box = (r0, c0, r1, c1)
        if any(_box_overlaps(box, (other.r0, other.c0, other.r1, other.c1), pad=1) for other in existing):
            continue
        return candidate
    return None


def _generate_layout(rng: random.Random) -> tuple[tuple[int, int], int, list[Container]] | None:
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)
    faucet_col = rng.randint(3, width - 4)

    path = _build_path_containers(rng, height, width, faucet_col)
    if path is None:
        return None

    forbidden_cols = _path_drop_columns(path, faucet_col)
    containers = list(path)

    distractor_target = rng.randint(1, 3)
    want_gapped = rng.random() < 0.7
    gapped_added = False
    for i in range(distractor_target):
        distractor = _try_place_distractor(
            rng,
            height,
            width,
            forbidden_cols,
            containers,
            allow_gap=(want_gapped and (not gapped_added or i == distractor_target - 1)),
        )
        if distractor is None:
            continue
        gapped_added = gapped_added or distractor.gap_side is not None
        containers.append(distractor)

    if want_gapped and not gapped_added:
        distractor = _try_place_distractor(rng, height, width, forbidden_cols, containers, allow_gap=True)
        if distractor is None or distractor.gap_side is None:
            return None
        containers.append(distractor)

    return (height, width), faucet_col, containers


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert int(np.count_nonzero(inp[0] == WATER)) == 1
    assert np.count_nonzero(inp == WALL) > 0
    assert np.count_nonzero(out == WATER) > np.count_nonzero(inp == WATER)
    assert np.array_equal(inp[inp == WALL], out[inp == WALL])
    assert np.array_equal(inp == WALL, out == WALL)
    reachable = _detect_reachable_containers(inp)
    unreachable = [c for c in _detect_containers(inp) if c not in reachable]
    for container in unreachable:
        interior = out[container.r0 : container.r1, container.c0 + 1 : container.c1]
        assert not np.any(interior == WATER), "unreached container should stay empty"
        gap = container.gap_cell()
        if gap is not None:
            assert int(out[gap]) != WATER, "unreached gap should stay empty"


def _detect_reachable_containers(grid: np.ndarray) -> list[Container]:
    containers = _detect_containers(grid)
    g = np.asarray(grid, dtype=int)
    faucet_col = int(np.flatnonzero(g[0] == WATER)[0])
    row = 1
    col = faucet_col
    reached: list[Container] = []
    while row < g.shape[0]:
        container, entry_row = _find_first_container(containers, col, row)
        if container is None:
            break
        assert entry_row is not None
        reached.append(container)
        row = int(container.gap_row)
        col = int(container.outflow_col())
    return reached


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _generate_layout(rng)
    if layout is None:
        return None
    shape, faucet_col, containers = layout
    inp = _render_input(shape, faucet_col, containers)
    expected = _render_expected(shape, faucet_col, containers)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    try:
        _semantic_checks(inp, actual)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascading-water ARC input")
