from __future__ import annotations

from collections import defaultdict, deque
import importlib.util
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_arc_utils_task_036_2", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    in_bounds = _ARC_UTILS.in_bounds
    find_components = _ARC_UTILS.find_components
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | tuple[int, ...] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        assert connectivity == 4
        assert monochromatic
        assert return_format == "dict"
        g = np.asarray(grid, dtype=int)
        if bg is None:
            values, counts = np.unique(g, return_counts=True)
            bg = int(values[np.argmax(counts)])
        allowed = set(int(v) for v in (colors if colors is not None else np.unique(g)))
        allowed.discard(int(bg))
        h, w = g.shape
        seen = np.zeros((h, w), dtype=bool)
        out: list[dict] = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(g[r, c]) not in allowed:
                    continue
                color = int(g[r, c])
                q = deque([(r, c)])
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if not in_bounds(g.shape, nr, nc) or seen[nr, nc] or int(g[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        q.append((nr, nc))
                if len(coords) < int(min_area):
                    continue
                rows = [coord[0] for coord in coords]
                cols = [coord[1] for coord in coords]
                out.append(
                    {
                        "color": color,
                        "coords": np.array(coords, dtype=int),
                        "bbox": (min(rows), min(cols), max(rows), max(cols)),
                        "area": len(coords),
                    }
                )
        return out


SQUARE_COLOR = Color.GREY
OBSTACLE_COLOR = Color.MAROON
BACKGROUND_COLOR = Color.BLACK

RAY_ORDER = (
    ("nw", (-1, -1), (-1, -1), Color.BLUE),
    ("ne", (-1, 2), (-1, 1), Color.RED),
    ("sw", (2, -1), (1, -1), Color.GREEN),
    ("se", (2, 2), (1, 1), Color.YELLOW),
)

GEN_RETRY_LIMIT = 300
GRID_HEIGHT_RANGE = (13, 20)
GRID_WIDTH_RANGE = (15, 26)
TWO_SQUARE_GAPS = (5, 7, 9)
THREE_SQUARE_GAPS = (5, 7)
TARGET_OBSTACLE_RANGE = (5, 8)


def _square_cells(square: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = square
    return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]


def _marker_specs(square: tuple[int, int]) -> list[dict]:
    r0, c0 = square
    specs = []
    for name, (mr, mc), direction, color in RAY_ORDER:
        specs.append(
            {
                "name": name,
                "marker": (r0 + mr, c0 + mc),
                "direction": direction,
                "color": int(color),
            }
        )
    return specs


def _build_blockers(squares: list[tuple[int, int]]) -> set[tuple[int, int]]:
    blockers: set[tuple[int, int]] = set()
    for square in squares:
        blockers.update(_square_cells(square))
        for spec in _marker_specs(square):
            blockers.add(spec["marker"])
    return blockers


def _trace_ray(
    shape: tuple[int, int],
    marker: tuple[int, int],
    direction: tuple[int, int],
    blockers: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str, tuple[int, int] | None]:
    dr, dc = direction
    r, c = marker
    path: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(shape, nr, nc):
            return path, "boundary", None
        if (nr, nc) in blockers:
            return path, "blocker", (nr, nc)
        if (nr, nc) in obstacles:
            return path, "obstacle", (nr, nc)
        path.append((nr, nc))
        r, c = nr, nc


def _ray_descriptors(squares: list[tuple[int, int]], shape: tuple[int, int]) -> list[dict]:
    blockers = _build_blockers(squares)
    rays: list[dict] = []
    for square_index, square in enumerate(squares):
        for spec in _marker_specs(square):
            path, _, _ = _trace_ray(shape, spec["marker"], spec["direction"], blockers, set())
            rays.append(
                {
                    "square_index": square_index,
                    "name": spec["name"],
                    "marker": spec["marker"],
                    "direction": spec["direction"],
                    "color": spec["color"],
                    "path": path,
                }
            )
    return rays


def _find_squares(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    squares = find_components(
        g,
        bg=BACKGROUND_COLOR,
        connectivity=4,
        monochromatic=True,
        colors=[SQUARE_COLOR],
    )
    out: list[tuple[int, int]] = []
    for component in squares:
        if int(component["area"]) != 4:
            continue
        r0, c0, r1, c1 = component["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == SQUARE_COLOR):
            continue
        out.append((int(r0), int(c0)))
    out.sort()
    return out


def _square_distance_ok(squares: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(squares):
        for r1, c1 in squares[i + 1 :]:
            cells0 = _square_cells((r0, c0))
            cells1 = _square_cells((r1, c1))
            best = min(abs(a_r - b_r) + abs(a_c - b_c) for a_r, a_c in cells0 for b_r, b_c in cells1)
            if best < 4:
                return False
    return True


def _obstacle_far_from_squares(cell: tuple[int, int], squares: list[tuple[int, int]]) -> bool:
    r, c = cell
    for square in squares:
        for sr, sc in _square_cells(square):
            if abs(r - sr) + abs(c - sc) < 3:
                return False
    return True


def _isolated(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in others:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _build_grid(
    shape: tuple[int, int],
    squares: list[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0 in squares:
        grid[r0 : r0 + 2, c0 : c0 + 2] = SQUARE_COLOR
    for r, c in obstacles:
        grid[r, c] = OBSTACLE_COLOR
    return grid


def _simulate_hits(
    shape: tuple[int, int],
    squares: list[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], list[dict]], dict[tuple[int, int], set[int]]]:
    blockers = _build_blockers(squares)
    hits: dict[tuple[int, int], list[dict]] = defaultdict(list)
    painted_by: dict[tuple[int, int], set[int]] = defaultdict(set)
    for ray_index, ray in enumerate(_ray_descriptors(squares, shape)):
        path, stop_type, stop_coord = _trace_ray(shape, ray["marker"], ray["direction"], blockers, obstacles)
        for cell in path:
            painted_by[cell].add(ray_index)
        if stop_type == "obstacle" and stop_coord is not None:
            hits[stop_coord].append(ray)
    return hits, painted_by


def main(grid: np.ndarray) -> np.ndarray:
    """Add colored corner markers, fire diagonal rays, and recolor maroon obstacles by how many rays hit them.

    Input structure:
    - The grid has a black background, 2 or 3 solid gray 2x2 squares, and several isolated maroon single-pixel obstacles.
    - Each gray square has empty diagonal corner cells around it, and squares are spaced so their markers and rays are distinguishable.

    Rule:
    - Put blue, red, green, and yellow corner markers at the northwest, northeast, southwest, and southeast diagonal outside corners of every gray 2x2 square.
    - From each marker, extend a same-colored diagonal ray away from the square until the next step would leave the grid, hit a maroon obstacle, or enter another square cell or corner marker.
    - Leave squares, markers, and ray trails visible, then recolor each maroon obstacle to its single hitting ray color when hit once, to gray when hit two or more times, or keep it maroon when never hit.

    Key insight:
    - The colored corner decorations are also directional ray sources, and each obstacle's final color depends on the global pattern of ray convergence rather than its local neighborhood alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    squares = _find_squares(g)
    assert 2 <= len(squares) <= 3

    obstacles = {tuple(map(int, coord)) for coord in np.argwhere(g == OBSTACLE_COLOR)}
    blockers = _build_blockers(squares)
    rays = _ray_descriptors(squares, g.shape)

    out = g.copy()
    hits: dict[tuple[int, int], list[int]] = defaultdict(list)

    for ray in rays:
        mr, mc = ray["marker"]
        out[mr, mc] = ray["color"]

    for ray in rays:
        path, stop_type, stop_coord = _trace_ray(g.shape, ray["marker"], ray["direction"], blockers, obstacles)
        for r, c in path:
            out[r, c] = ray["color"]
        if stop_type == "obstacle" and stop_coord is not None:
            hits[stop_coord].append(ray["color"])

    for obstacle in obstacles:
        colors = hits.get(obstacle, [])
        if len(colors) == 1:
            out[obstacle] = colors[0]
        elif len(colors) >= 2:
            out[obstacle] = SQUARE_COLOR

    return out


def _crossing_obstacles(squares: list[tuple[int, int]]) -> list[tuple[int, int]]:
    row = squares[0][0]
    crossings: list[tuple[int, int]] = []
    for i in range(len(squares) - 1):
        left_c = squares[i][1]
        right_c = squares[i + 1][1]
        gap = right_c - left_c
        t = (gap - 3) // 2
        crossings.append((row - 1 - t, left_c + 2 + t))
        crossings.append((row + 2 + t, left_c + 2 + t))
    return crossings


def _outer_rays(rays: list[dict], square_count: int) -> list[dict]:
    wanted = {
        (0, "nw"),
        (0, "sw"),
        (square_count - 1, "ne"),
        (square_count - 1, "se"),
    }
    return [ray for ray in rays if (ray["square_index"], ray["name"]) in wanted]


def _choose_single_obstacles(
    rays: list[dict],
    square_count: int,
    existing: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    singles: set[tuple[int, int]] = set()
    candidates = _outer_rays(rays, square_count)
    random.shuffle(candidates)
    count = 1 if square_count == 3 else random.choice([1, 2])
    for ray in candidates:
        choices = [cell for cell in ray["path"][:-1] or ray["path"] if cell not in existing and _isolated(cell, existing | singles)]
        if not choices:
            continue
        singles.add(random.choice(choices))
        if len(singles) == count:
            return singles
    return None


def _fillers(
    shape: tuple[int, int],
    squares: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    need: int,
) -> set[tuple[int, int]] | None:
    candidates: list[tuple[int, int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            cell = (r, c)
            if cell in forbidden or not _obstacle_far_from_squares(cell, squares):
                continue
            candidates.append(cell)
    random.shuffle(candidates)
    placed: set[tuple[int, int]] = set()
    for cell in candidates:
        if _isolated(cell, forbidden | placed):
            placed.add(cell)
            if len(placed) == need:
                return placed
    return None


def _generate_attempt() -> np.ndarray | None:
    square_count = random.choice([2, 3])
    gap_choices = TWO_SQUARE_GAPS if square_count == 2 else THREE_SQUARE_GAPS
    gaps = [random.choice(gap_choices) for _ in range(square_count - 1)]
    gap_sum = sum(gaps)
    max_t = max((gap - 3) // 2 for gap in gaps)

    min_h = max(GRID_HEIGHT_RANGE[0], 7 + max_t)
    min_w = max(GRID_WIDTH_RANGE[0], gap_sum + 8)
    max_h = GRID_HEIGHT_RANGE[1]
    max_w = GRID_WIDTH_RANGE[1]
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    row_max = min(height - 5, height - 3 - max_t)
    col0_max = width - 5 - gap_sum
    if row_max < 3 or col0_max < 3:
        return None

    row = random.randint(3, row_max)
    col0 = random.randint(3, col0_max)
    squares = [(row, col0)]
    for gap in gaps:
        squares.append((row, squares[-1][1] + gap))
    if not _square_distance_ok(squares):
        return None

    shape = (height, width)
    rays = _ray_descriptors(squares, shape)
    crossings = set(_crossing_obstacles(squares))
    if not all(_obstacle_far_from_squares(cell, squares) for cell in crossings):
        return None
    if any(not _isolated(cell, crossings - {cell}) for cell in crossings):
        return None

    singles = _choose_single_obstacles(rays, square_count, crossings)
    if singles is None:
        return None

    hit_obstacles = crossings | singles
    min_total = max(TARGET_OBSTACLE_RANGE[0], len(hit_obstacles) + 1)
    if min_total > TARGET_OBSTACLE_RANGE[1]:
        return None
    target_total = random.randint(min_total, TARGET_OBSTACLE_RANGE[1])

    ray_cells = {cell for ray in rays for cell in ray["path"]}
    forbidden = _build_blockers(squares) | ray_cells | hit_obstacles
    fillers = _fillers(shape, squares, forbidden, target_total - len(hit_obstacles))
    if fillers is None:
        return None

    obstacles = hit_obstacles | fillers
    hits, painted_by = _simulate_hits(shape, squares, obstacles)
    if any(len(ray_ids) > 1 for ray_ids in painted_by.values()):
        return None

    counts = [len(hits.get(obstacle, [])) for obstacle in obstacles]
    if not any(count == 0 for count in counts):
        return None
    if not any(count == 1 for count in counts):
        return None
    if not any(count >= 2 for count in counts):
        return None
    if sum(count > 0 for count in counts) < 2:
        return None

    for obstacle, ray_list in hits.items():
        direction_names = [ray["name"] for ray in ray_list]
        if len(direction_names) != len(set(direction_names)):
            return None
        if len(ray_list) > 4:
            return None
        if obstacle in fillers:
            return None

    for obstacle in obstacles:
        if not _obstacle_far_from_squares(obstacle, squares):
            return None
        if not _isolated(obstacle, obstacles - {obstacle}):
            return None

    grid = _build_grid(shape, squares, obstacles)
    if max(grid.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-ray obstacle input")
