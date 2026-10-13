from __future__ import annotations

from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS = SourceFileLoader("_arc_utils_shared_017_3", str(_ARC_UTILS_PATH)).load_module()
Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords


Coord = tuple[int, int]


@dataclass(frozen=True)
class Arm:
    point: Coord
    anchor: Coord
    direction: Coord
    between: tuple[Coord, ...]
    ray_to_edge: tuple[Coord, ...]
    clear_to_edge: bool


@dataclass(frozen=True)
class Scene:
    orientation: str
    trunk_endpoints: tuple[Coord, Coord]
    trunk_cells: tuple[Coord, ...]
    arms: tuple[Arm, ...]


def _sorted_pair(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


def _find_scene(grid: np.ndarray) -> Scene:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.RED)]
    assert 5 <= len(red_coords) <= 8, "expected 5-8 red control pixels"

    shared_pairs: list[tuple[Coord, Coord]] = []
    for i in range(len(red_coords)):
        r1, c1 = red_coords[i]
        for j in range(i + 1, len(red_coords)):
            r2, c2 = red_coords[j]
            if r1 == r2 or c1 == c2:
                shared_pairs.append(_sorted_pair((r1, c1), (r2, c2)))

    assert len(shared_pairs) == 1, "expected exactly one raw red pair sharing a row or column"
    p0, p1 = shared_pairs[0]

    if p0[0] == p1[0]:
        orientation = "horizontal"
        trunk_row = p0[0]
        c0, c1 = sorted((p0[1], p1[1]))
        assert c1 - c0 >= 2, "horizontal trunk needs interior columns for arms"
        trunk_cells = tuple((trunk_row, c) for c in range(c0, c1 + 1))
        other_points = [p for p in red_coords if p not in {p0, p1}]
        arms: list[Arm] = []
        for r, c in sorted(other_points):
            assert c0 < c < c1, "arm control must align to an interior trunk column"
            assert r != trunk_row, "arm control cannot lie on the trunk row"
            step = 1 if r > trunk_row else -1
            between = tuple((rr, c) for rr in range(trunk_row + step, r, step))
            ray_to_edge = tuple((rr, c) for rr in range(r + step, g.shape[0], step))
            assert all(g[rr, cc] == Color.BLACK for rr, cc in between), "arm corridor to trunk must be empty"
            clear_to_edge = all(g[rr, cc] == Color.BLACK for rr, cc in ray_to_edge)
            arms.append(
                Arm(
                    point=(r, c),
                    anchor=(trunk_row, c),
                    direction=(step, 0),
                    between=between,
                    ray_to_edge=ray_to_edge,
                    clear_to_edge=clear_to_edge,
                )
            )
    else:
        orientation = "vertical"
        trunk_col = p0[1]
        r0, r1 = sorted((p0[0], p1[0]))
        assert r1 - r0 >= 2, "vertical trunk needs interior rows for arms"
        trunk_cells = tuple((r, trunk_col) for r in range(r0, r1 + 1))
        other_points = [p for p in red_coords if p not in {p0, p1}]
        arms = []
        for r, c in sorted(other_points):
            assert r0 < r < r1, "arm control must align to an interior trunk row"
            assert c != trunk_col, "arm control cannot lie on the trunk column"
            step = 1 if c > trunk_col else -1
            between = tuple((r, cc) for cc in range(trunk_col + step, c, step))
            ray_to_edge = tuple((r, cc) for cc in range(c + step, g.shape[1], step))
            assert all(g[rr, cc] == Color.BLACK for rr, cc in between), "arm corridor to trunk must be empty"
            clear_to_edge = all(g[rr, cc] == Color.BLACK for rr, cc in ray_to_edge)
            arms.append(
                Arm(
                    point=(r, c),
                    anchor=(r, trunk_col),
                    direction=(0, step),
                    between=between,
                    ray_to_edge=ray_to_edge,
                    clear_to_edge=clear_to_edge,
                )
            )

    return Scene(
        orientation=orientation,
        trunk_endpoints=_sorted_pair(p0, p1),
        trunk_cells=trunk_cells,
        arms=tuple(arms),
    )


def _render_scene(grid: np.ndarray, scene: Scene) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for r, c in scene.trunk_cells[1:-1]:
        out[r, c] = Color.ORANGE

    for arm in scene.arms:
        for r, c in arm.between:
            out[r, c] = Color.ORANGE
        if arm.clear_to_edge:
            for r, c in arm.between:
                out[r, c] = Color.GREEN
            out[arm.point] = Color.GREEN
            for r, c in arm.ray_to_edge:
                out[r, c] = Color.GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Connect the unique red trunk, then recolor each perpendicular arm by whether its outward corridor is clear.

    Input structure:
    - The grid has a black background with only red control pixels and a few gray obstacle pixels.
    - Exactly one pair of red pixels shares a row or column; those two pixels are the trunk endpoints.
    - Every other red pixel aligns perpendicularly to an interior point of that trunk, with an empty corridor back to it.

    Rule:
    - Draw the trunk segment between the unique aligned red pair, keeping the endpoints red and painting the interior orange.
    - For each non-trunk red pixel, draw the perpendicular connection back to the trunk in orange.
    - Test the ray that continues past that red pixel away from the trunk: if the whole ray to the border is black, recolor the arm pixel and its connection green and extend green to the edge; if any foreground blocks the ray, leave that arm red-plus-orange and do not extend it.

    Key insight:
    - The same red arm anchor can produce two visibly different outcomes, and the deciding evidence is not the gap to the trunk but the separate corridor extending beyond the arm away from the trunk.
    """
    scene = _find_scene(np.asarray(grid, dtype=int))
    return _render_scene(grid, scene)


GRID_MIN = 13
GRID_MAX = 19
ARM_COUNT_CHOICES = (3, 4, 5, 6)
TRUNK_LENGTH_MIN = 5
TRUNK_LENGTH_MAX = 10
GEN_ATTEMPTS = 400


def _rng_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _pick_arm_lines(rng: np.random.Generator, trunk_line: int, limit: int, count: int) -> list[int] | None:
    low_side = list(range(1, trunk_line))
    high_side = list(range(trunk_line + 1, limit - 1))
    if len(low_side) + len(high_side) < count:
        return None

    min_low = 1 if low_side and high_side and count >= 2 else 0
    max_low = min(len(low_side), count)
    low_lo = max(min_low, count - len(high_side))
    low_hi = max_low
    if low_lo > low_hi:
        return None
    low_count = _rng_int(rng, low_lo, low_hi)
    high_count = count - low_count
    picked = list(rng.choice(low_side, size=low_count, replace=False))
    picked += list(rng.choice(high_side, size=high_count, replace=False))
    rng.shuffle(picked)
    return [int(v) for v in picked]


def _choose_trunk_params(rng: np.random.Generator, size: int, arm_count: int) -> tuple[int, int]:
    max_len = min(TRUNK_LENGTH_MAX, size - 2)
    min_len = max(TRUNK_LENGTH_MIN, arm_count + 2)
    if min_len > max_len:
        raise ValueError("grid too small for requested trunk and arm count")
    trunk_len = _rng_int(rng, min_len, max_len)
    start = _rng_int(rng, 1, size - trunk_len - 1)
    return start, trunk_len


def _allocate_greys(
    rng: np.random.Generator,
    blocked_indices: list[int],
    ray_lengths: list[int],
) -> dict[int, list[int]] | None:
    max_total = min(4, sum(ray_lengths))
    min_total = max(2, len(blocked_indices))
    if min_total > max_total:
        return None

    total_greys = _rng_int(rng, min_total, max_total)
    per_arm = {idx: 1 for idx in blocked_indices}
    extra = total_greys - len(blocked_indices)

    expandable = [idx for idx, length in zip(blocked_indices, ray_lengths) if length >= 2]
    while extra > 0:
        if not expandable:
            return None
        idx = int(rng.choice(expandable))
        if per_arm[idx] < ray_lengths[blocked_indices.index(idx)]:
            per_arm[idx] += 1
            extra -= 1
        if per_arm[idx] >= ray_lengths[blocked_indices.index(idx)]:
            expandable = [j for j in expandable if j != idx]

    return per_arm


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = _rng_int(rng, GRID_MIN, GRID_MAX)
    width = _rng_int(rng, GRID_MIN, GRID_MAX)
    orientation = str(rng.choice(["horizontal", "vertical"]))
    arm_count = int(rng.choice(ARM_COUNT_CHOICES))

    try:
        if orientation == "horizontal":
            c0, trunk_len = _choose_trunk_params(rng, width, arm_count)
            c1 = c0 + trunk_len - 1
            trunk_row = _rng_int(rng, 2, height - 3)
            arm_rows = _pick_arm_lines(rng, trunk_row, height, arm_count)
            if arm_rows is None:
                return None
            arm_cols = [int(v) for v in rng.choice(np.arange(c0 + 1, c1), size=arm_count, replace=False)]
        else:
            r0, trunk_len = _choose_trunk_params(rng, height, arm_count)
            r1 = r0 + trunk_len - 1
            trunk_col = _rng_int(rng, 2, width - 3)
            arm_cols = _pick_arm_lines(rng, trunk_col, width, arm_count)
            if arm_cols is None:
                return None
            arm_rows = [int(v) for v in rng.choice(np.arange(r0 + 1, r1), size=arm_count, replace=False)]
    except ValueError:
        return None

    grid = np.zeros((height, width), dtype=int)

    if orientation == "horizontal":
        grid[trunk_row, c0] = Color.RED
        grid[trunk_row, c1] = Color.RED
        arm_points = sorted(zip(arm_rows, arm_cols))
        ray_cells: list[list[Coord]] = []
        for r, c in arm_points:
            grid[r, c] = Color.RED
            step = 1 if r > trunk_row else -1
            ray = [(rr, c) for rr in range(r + step, height if step > 0 else -1, step)]
            if not ray:
                return None
            ray_cells.append(ray)
    else:
        grid[r0, trunk_col] = Color.RED
        grid[r1, trunk_col] = Color.RED
        arm_points = sorted(zip(arm_rows, arm_cols))
        ray_cells = []
        for r, c in arm_points:
            grid[r, c] = Color.RED
            step = 1 if c > trunk_col else -1
            ray = [(r, cc) for cc in range(c + step, width if step > 0 else -1, step)]
            if not ray:
                return None
            ray_cells.append(ray)

    max_blockable = sum(1 for ray in ray_cells if ray)
    if max_blockable < 1 or arm_count < 2:
        return None
    blocked_count = _rng_int(rng, 1, min(arm_count - 1, 4, max_blockable))
    blocked_indices = sorted(int(v) for v in rng.choice(np.arange(arm_count), size=blocked_count, replace=False))
    allocation = _allocate_greys(rng, blocked_indices, [len(ray_cells[idx]) for idx in blocked_indices])
    if allocation is None:
        return None

    for idx in blocked_indices:
        ray = ray_cells[idx]
        grey_count = allocation[idx]
        choices = sorted(int(v) for v in rng.choice(np.arange(len(ray)), size=grey_count, replace=False))
        for pos in choices:
            r, c = ray[pos]
            grid[r, c] = Color.GREY

    if not _validate_generated_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")


def _validate_generated_scene(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30:
            return False
        if not set(map(int, np.unique(g))).issubset({Color.BLACK, Color.RED, Color.GREY}):
            return False
        if not (2 <= int(np.count_nonzero(g == Color.GREY)) <= 4):
            return False

        scene = _find_scene(g)
        if not (3 <= len(scene.arms) <= 6):
            return False

        clear_count = sum(arm.clear_to_edge for arm in scene.arms)
        blocked_count = len(scene.arms) - clear_count
        if clear_count < 1 or blocked_count < 1:
            return False

        red_bbox = bbox_coords(np.argwhere(g == Color.RED))
        if scene.orientation == "horizontal":
            trunk_row = scene.trunk_endpoints[0][0]
            c0, c1 = scene.trunk_endpoints[0][1], scene.trunk_endpoints[1][1]
            if any(g[trunk_row, c] != Color.BLACK for c in range(c0 + 1, c1)):
                return False
            arm_rows = [arm.point[0] for arm in scene.arms]
            arm_cols = [arm.point[1] for arm in scene.arms]
            if len(set(arm_rows)) != len(arm_rows) or len(set(arm_cols)) != len(arm_cols):
                return False
        else:
            trunk_col = scene.trunk_endpoints[0][1]
            r0, r1 = scene.trunk_endpoints[0][0], scene.trunk_endpoints[1][0]
            if any(g[r, trunk_col] != Color.BLACK for r in range(r0 + 1, r1)):
                return False
            arm_rows = [arm.point[0] for arm in scene.arms]
            arm_cols = [arm.point[1] for arm in scene.arms]
            if len(set(arm_rows)) != len(arm_rows) or len(set(arm_cols)) != len(arm_cols):
                return False

        out = _render_scene(g, scene)
        if np.array_equal(out, g):
            return False

        for r, c in scene.trunk_endpoints:
            if out[r, c] != Color.RED:
                return False
        for r, c in scene.trunk_cells[1:-1]:
            if out[r, c] != Color.ORANGE:
                return False

        for arm in scene.arms:
            pr, pc = arm.point
            if arm.clear_to_edge:
                if out[pr, pc] != Color.GREEN:
                    return False
                if any(out[r, c] != Color.GREEN for r, c in arm.between):
                    return False
                if any(out[r, c] != Color.GREEN for r, c in arm.ray_to_edge):
                    return False
            else:
                if out[pr, pc] != Color.RED:
                    return False
                if any(out[r, c] != Color.ORANGE for r, c in arm.between):
                    return False

        if np.count_nonzero(out == Color.GREEN) == 0 or np.count_nonzero(out == Color.ORANGE) == 0:
            return False

        _ = red_bbox
        return True
    except AssertionError:
        return False

