from __future__ import annotations

import importlib.util
import random

import numpy as np

_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"

try:
    from arc_utils import Color, in_bounds
except Exception:
    _spec = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
    assert _spec is not None and _spec.loader is not None
    _arc_utils = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_arc_utils)
    Color = _arc_utils.Color
    in_bounds = _arc_utils.in_bounds


BACKGROUND = Color.BLACK
GREY = Color.GREY
LEFT_MIRROR = Color.RED
RIGHT_MIRROR = Color.TEAL
MARKER_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
SIDES = ("top", "bottom", "left", "right")
INWARD = {"top": DOWN, "bottom": UP, "left": RIGHT, "right": LEFT}
TURN_SEGMENT_MIN = 2
TURN_SEGMENT_MAX = 6
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
RAY_COUNT_CHOICES = (2, 3, 4)
MAX_SAMPLE_ATTEMPTS = 500
MAX_RAY_ATTEMPTS = 80


def _is_boundary(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    return {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}[direction]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    return {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}[direction]


def _marker_direction(shape: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = cell
    touches = [r == 0, r == h - 1, c == 0, c == w - 1]
    assert sum(touches) == 1, "marker must lie on exactly one edge"
    if r == 0:
        return DOWN
    if r == h - 1:
        return UP
    if c == 0:
        return RIGHT
    return LEFT


def _edge_cells(shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    return [(r, w - 1) for r in range(1, h - 1)]


def _cells_to_edge(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not in_bounds(shape, r, c):
            break
        cells.append((r, c))
    return cells


def _collect_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    markers = set()
    for c in range(w):
        if grid[0, c] != BACKGROUND:
            markers.add((0, c))
        if grid[h - 1, c] != BACKGROUND:
            markers.add((h - 1, c))
    for r in range(1, h - 1):
        if grid[r, 0] != BACKGROUND:
            markers.add((r, 0))
        if grid[r, w - 1] != BACKGROUND:
            markers.add((r, w - 1))
    ordered = sorted(markers)
    for marker in ordered:
        assert int(grid[marker]) != GREY, "edge markers cannot be grey"
        _marker_direction(grid.shape, marker)
    return ordered


def _trace_ray(grid: np.ndarray, start: tuple[int, int]) -> dict:
    h, w = grid.shape
    color = int(grid[start])
    direction = _marker_direction(grid.shape, start)
    r, c = start
    path = [start]
    mirrors: dict[tuple[int, int], int] = {}
    segment_lengths: list[int] = []
    visited = {start}
    segment_steps = 0

    for _ in range(h * w + 5):
        nr = r + direction[0]
        nc = c + direction[1]
        if not in_bounds(grid.shape, nr, nc):
            segment_lengths.append(segment_steps)
            return {
                "start": start,
                "color": color,
                "path": path,
                "mirrors": mirrors,
                "segment_lengths": segment_lengths,
            }

        r, c = nr, nc
        cell = (r, c)
        assert cell not in visited, "ray cannot loop back onto itself"
        visited.add(cell)
        path.append(cell)
        segment_steps += 1

        value = int(grid[cell])
        if value == BACKGROUND:
            continue
        if value in (LEFT_MIRROR, RIGHT_MIRROR):
            assert not _is_boundary(grid.shape, cell), "mirror must be internal"
            mirrors[cell] = value
            segment_lengths.append(segment_steps)
            segment_steps = 0
            direction = _turn_left(direction) if value == LEFT_MIRROR else _turn_right(direction)
            continue
        raise AssertionError("ray hit another marker or an unexpected foreground cell")

    raise AssertionError("ray tracing exceeded grid budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit each edge marker as an inward ray that alternates marker color and grey across mirror-guided travel.

    Input structure:
    - A black background grid has one or more non-grey single-cell markers on the outer border.
    - Interior single-cell mirrors are teal for a right turn and red for a left turn.
    - Mirror cells are the only non-black interior cells, and markers are the only non-black edge cells.

    Rule:
    - Each marker launches a ray perpendicular to its edge and continues until the next step would leave the grid.
    - A teal mirror turns the current direction right by 90 degrees; a red mirror turns it left by 90 degrees.
    - Count path steps cumulatively with the marker cell as step 0: paint non-mirror path cells with the marker color on even steps and grey on odd steps, while mirror cells keep their mirror colors.

    Key insight:
    - The alternating phase follows total path length through reflections; it does not restart after a turn and cannot be inferred from a local row or column parity rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g[(g != BACKGROUND) & (g != GREY)] >= 0)

    interior = g[1:-1, 1:-1]
    if interior.size:
        assert np.all(
            (interior == BACKGROUND) | (interior == LEFT_MIRROR) | (interior == RIGHT_MIRROR)
        ), "interior foreground must be mirrors"

    markers = _collect_markers(g)
    assert markers, "expected at least one edge marker"

    rays = [_trace_ray(g, marker) for marker in markers]
    seen: set[tuple[int, int]] = set()
    for ray in rays:
        for cell in ray["path"]:
            assert cell not in seen, "rays must not intersect"
            seen.add(cell)

    out = g.copy()
    for ray in rays:
        color = int(ray["color"])
        mirrors = ray["mirrors"]
        for step, cell in enumerate(ray["path"]):
            if cell in mirrors:
                out[cell] = mirrors[cell]
            else:
                out[cell] = color if step % 2 == 0 else GREY
    return out


def _search_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    reflections: int,
    occupied: set[tuple[int, int]],
) -> dict | None:
    path = [start]
    mirrors: dict[tuple[int, int], int] = {}
    segment_lengths: list[int] = []

    def dfs(
        current: tuple[int, int],
        current_dir: tuple[int, int],
        reflections_left: int,
        used: set[tuple[int, int]],
    ) -> bool:
        ahead_all = _cells_to_edge(shape, current, current_dir)
        free_prefix: list[tuple[int, int]] = []
        for cell in ahead_all:
            if cell in used:
                break
            free_prefix.append(cell)

        if reflections_left == 0:
            if len(free_prefix) != len(ahead_all) or len(ahead_all) < TURN_SEGMENT_MIN:
                return False
            path.extend(ahead_all)
            segment_lengths.append(len(ahead_all))
            return True

        candidates = [
            idx
            for idx, cell in enumerate(free_prefix, start=1)
            if TURN_SEGMENT_MIN <= idx <= TURN_SEGMENT_MAX and not _is_boundary(shape, cell)
        ]
        random.shuffle(candidates)

        for seg_len in candidates:
            segment_cells = free_prefix[:seg_len]
            mirror_cell = segment_cells[-1]
            used_after = used | set(segment_cells)

            turns = [
                (LEFT_MIRROR, _turn_left(current_dir)),
                (RIGHT_MIRROR, _turn_right(current_dir)),
            ]
            random.shuffle(turns)
            for mirror_color, new_dir in turns:
                next_leg = _cells_to_edge(shape, mirror_cell, new_dir)
                min_leg = TURN_SEGMENT_MIN if reflections_left == 1 else TURN_SEGMENT_MIN + 1
                if len(next_leg) < min_leg:
                    continue
                if next_leg and next_leg[0] in used_after:
                    continue

                path_len = len(path)
                seg_count = len(segment_lengths)
                path.extend(segment_cells)
                segment_lengths.append(seg_len)
                mirrors[mirror_cell] = mirror_color
                if dfs(mirror_cell, new_dir, reflections_left - 1, used_after):
                    return True
                del path[path_len:]
                del segment_lengths[seg_count:]
                mirrors.pop(mirror_cell, None)
        return False

    if not dfs(start, direction, reflections, set(occupied)):
        return None
    return {"start": start, "path": path, "mirrors": mirrors, "segment_lengths": segment_lengths}


def _build_ray(
    shape: tuple[int, int],
    side: str,
    color: int,
    occupied: set[tuple[int, int]],
) -> dict | None:
    starts = _edge_cells(shape, side)
    random.shuffle(starts)
    for start in starts:
        if start in occupied:
            continue
        reflections = 3 if random.random() < 0.45 else 2
        ray = _search_ray(shape, start, INWARD[side], reflections, occupied | {start})
        if ray is None:
            continue
        ray["color"] = int(color)
        ray["side"] = side
        return ray
    return None


def _sample_sides(ray_count: int) -> list[str]:
    sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    while len(sides) < ray_count:
        sides.append(random.choice(SIDES))
    random.shuffle(sides)
    return sides


def _render_input_output(shape: tuple[int, int], rays: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BACKGROUND, dtype=int)
    for ray in rays:
        inp[ray["start"]] = int(ray["color"])
        for cell, mirror_color in ray["mirrors"].items():
            inp[cell] = int(mirror_color)

    out = inp.copy()
    for ray in rays:
        color = int(ray["color"])
        mirrors = ray["mirrors"]
        for step, cell in enumerate(ray["path"]):
            if cell in mirrors:
                out[cell] = mirrors[cell]
            else:
                out[cell] = color if step % 2 == 0 else GREY
    return inp, out


def _render_reset_phase(shape: tuple[int, int], rays: list[dict]) -> np.ndarray:
    inp, out = _render_input_output(shape, rays)
    out[:, :] = inp
    for ray in rays:
        color = int(ray["color"])
        mirrors = ray["mirrors"]
        path = ray["path"]
        idx = 0
        for seg_len in ray["segment_lengths"]:
            segment = path[idx : idx + seg_len + 1]
            for offset, cell in enumerate(segment):
                if offset == 0 or cell in mirrors:
                    continue
                out[cell] = color if offset % 2 == 0 else GREY
            idx += seg_len
    return out


def _validate_generated(inp: np.ndarray, out: np.ndarray, rays: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(rays) < 2:
        return False

    sides = {ray["side"] for ray in rays}
    if not any(side in {"top", "bottom"} for side in sides):
        return False
    if not any(side in {"left", "right"} for side in sides):
        return False

    mirror_colors = {int(color) for ray in rays for color in ray["mirrors"].values()}
    if mirror_colors != {LEFT_MIRROR, RIGHT_MIRROR}:
        return False

    total_path_cells = 0
    seen: set[tuple[int, int]] = set()
    for ray in rays:
        if len(ray["mirrors"]) < 2:
            return False
        if len(ray["segment_lengths"]) != len(ray["mirrors"]) + 1:
            return False
        if any(seg_len < TURN_SEGMENT_MIN for seg_len in ray["segment_lengths"]):
            return False
        total_path_cells += len(ray["path"])
        for cell in ray["path"]:
            if cell in seen:
                return False
            seen.add(cell)
    if total_path_cells > (inp.shape[0] * inp.shape[1]) // 3:
        return False

    if np.array_equal(_render_reset_phase(inp.shape, rays), out):
        return False

    traced = [_trace_ray(inp, marker) for marker in _collect_markers(inp)]
    if len(traced) != len(rays):
        return False

    traced_by_start = {ray["start"]: ray for ray in traced}
    for ray in rays:
        traced_ray = traced_by_start.get(ray["start"])
        if traced_ray is None:
            return False
        if traced_ray["path"] != ray["path"]:
            return False
        if traced_ray["mirrors"] != ray["mirrors"]:
            return False
        if traced_ray["segment_lengths"] != ray["segment_lengths"]:
            return False

    if not np.array_equal(main(inp), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    ray_count = random.choice(RAY_COUNT_CHOICES)
    sides = _sample_sides(ray_count)
    colors = random.sample(MARKER_COLORS, ray_count)

    occupied: set[tuple[int, int]] = set()
    rays: list[dict] = []
    for side, color in zip(sides, colors):
        ray = None
        for _ in range(MAX_RAY_ATTEMPTS):
            ray = _build_ray(shape, side, color, occupied)
            if ray is not None:
                break
        if ray is None:
            return None
        occupied.update(ray["path"])
        rays.append(ray)

    inp, out = _render_input_output(shape, rays)
    if _validate_generated(inp, out, rays):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray ARC input")
