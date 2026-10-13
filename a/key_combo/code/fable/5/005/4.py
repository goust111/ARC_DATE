from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]

BACKGROUND = 0
MIRROR = 5
GLASS = 9
OPTICAL_COLORS = {MIRROR, GLASS}
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIAGONALS = ("\\", "/")


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], pos: Coord) -> bool:
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _optical_orientations(grid: np.ndarray) -> dict[Coord, str]:
    """Map every optical pixel to the visible orientation of its 8-connected segment."""
    h, w = grid.shape
    result: dict[Coord, str] = {}
    for color in (MIRROR, GLASS):
        remaining = set(map(tuple, np.argwhere(grid == color)))
        while remaining:
            seed = remaining.pop()
            component = {seed}
            queue = [seed]
            while queue:
                r, c = queue.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        nxt = (r + dr, c + dc)
                        if nxt in remaining:
                            remaining.remove(nxt)
                            component.add(nxt)
                            queue.append(nxt)

            assert len(component) >= 2, "each optical object must be a diagonal segment"
            rows = [p[0] for p in component]
            cols = [p[1] for p in component]
            assert all(0 < r < h - 1 and 0 < c < w - 1 for r, c in component), (
                "optical segments must be internal"
            )
            assert len(set(rows)) == len(component) == len(set(cols)), (
                "segment pixels must advance one row and one column at a time"
            )
            if len({c - r for r, c in component}) == 1:
                orientation = "\\"
            else:
                assert len({c + r for r, c in component}) == 1, (
                    "optical objects must be straight / or \\ diagonal segments"
                )
                orientation = "/"
            assert max(rows) - min(rows) == len(component) - 1
            assert max(cols) - min(cols) == len(component) - 1
            for pos in component:
                result[pos] = orientation
    return result


def _boundary_markers(grid: np.ndarray, bg: int) -> list[tuple[Coord, int, Direction]]:
    h, w = grid.shape
    records: list[tuple[Coord, int, Direction]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            assert color not in OPTICAL_COLORS, "optical segments may not touch the boundary"
            inward: list[Direction] = []
            if r == 0:
                inward.append((1, 0))
            if r == h - 1:
                inward.append((-1, 0))
            if c == 0:
                inward.append((0, 1))
            if c == w - 1:
                inward.append((0, -1))
            assert len(inward) == 1, "a corner marker has two possible inward directions"
            records.append(((r, c), color, inward[0]))
    assert len(records) >= 2, "expected multiple boundary light markers"
    return records


def _reflect(direction: Direction, orientation: str) -> Direction:
    dr, dc = direction
    if orientation == "\\":
        return dc, dr
    assert orientation == "/"
    return -dc, -dr


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored boundary rays through transmitting glass and reflecting diagonal mirrors.

    Input structure:
    - A dominant background contains several single-cell colored markers on non-corner boundary cells.
    - Internal gray (5) and maroon/brown (9) objects are separated straight diagonal segments,
      each visibly oriented as either \\ or /; different marker paths do not overlap.

    Rule:
    - Emit each marker color inward, painting every background cell on its cardinal ray.
    - Gray diagonal cells are mirrors: preserve the cell and turn the ray by 90 degrees,
      with \\ and / applying their corresponding opposite reflection mappings.
    - Maroon/brown diagonal cells are glass: preserve the cell and continue straight.
      Stop only after the ray crosses the outer boundary, preserving every optical segment.

    Key insight:
    The examples reveal the two colors' different optical roles; segment orientation matters
    only for the reflecting color, while the equally visible glass segment is traversed unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a two-dimensional ARC grid"
    bg = _background(g)
    assert bg not in OPTICAL_COLORS, "background must differ from both optical colors"

    orientations = _optical_orientations(g)
    markers = _boundary_markers(g, bg)
    out = g.copy()
    used_route: set[Coord] = set()

    for start, ray_color, direction in markers:
        r, c = start
        seen_states: set[tuple[int, int, int, int]] = set()
        while True:
            dr, dc = direction
            r, c = r + dr, c + dc
            if not _in_bounds(g.shape, (r, c)):
                break
            state = (r, c, dr, dc)
            assert state not in seen_states, "mirror arrangement traps a ray in a loop"
            seen_states.add(state)
            assert (r, c) not in used_route, "light paths must not overlap"
            used_route.add((r, c))

            cell = int(g[r, c])
            if cell == bg:
                out[r, c] = ray_color
            elif cell == GLASS:
                pass
            elif cell == MIRROR:
                direction = _reflect(direction, orientations[(r, c)])
            else:
                raise AssertionError("a light path may cross only background or optical segments")
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _segment_options(hit: Coord, orientation: str, shape: tuple[int, int]) -> list[set[Coord]]:
    """All length/anchor variants of a diagonal segment containing hit."""
    h, w = shape
    slope = 1 if orientation == "\\" else -1
    options: list[set[Coord]] = []
    for length in (2, 3, 4):
        for anchor in range(length):
            cells = {
                (hit[0] + i - anchor, hit[1] + slope * (i - anchor))
                for i in range(length)
            }
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells):
                options.append(cells)
    random.shuffle(options)
    return options


def _separated(a: set[Coord], b: set[Coord]) -> bool:
    return all(max(abs(ar - br), abs(ac - bc)) > 1 for ar, ac in a for br, bc in b)


def _entry_geometry(
    edge: str, edge_coord: int, depth: int, shape: tuple[int, int]
) -> tuple[Coord, Direction, Coord]:
    h, w = shape
    if edge == "top":
        return (0, edge_coord), (1, 0), (depth, edge_coord)
    if edge == "bottom":
        return (h - 1, edge_coord), (-1, 0), (h - 1 - depth, edge_coord)
    if edge == "left":
        return (edge_coord, 0), (0, 1), (edge_coord, depth)
    assert edge == "right"
    return (edge_coord, w - 1), (0, -1), (edge_coord, w - 1 - depth)


def _cardinal_run(start: Coord, direction: Direction, shape: tuple[int, int]) -> list[Coord]:
    cells: list[Coord] = []
    r, c = start
    dr, dc = direction
    while True:
        r, c = r + dr, c + dc
        if not _in_bounds(shape, (r, c)):
            return cells
        cells.append((r, c))


def _try_add_ray(
    grid: np.ndarray,
    edge: str,
    marker_color: int,
    mirror_orientation: str,
    glass_orientation: str,
    glass_order: str,
    used_route: set[Coord],
    optical_components: list[set[Coord]],
) -> dict | None:
    h, w = grid.shape
    axis_limit = w if edge in ("top", "bottom") else h
    depth_limit = h if edge in ("top", "bottom") else w

    for _ in range(220):
        edge_coord = random.randint(4, axis_limit - 5)
        depth = random.randint(5, depth_limit - 6)
        marker, incoming, mirror_hit = _entry_geometry(edge, edge_coord, depth, grid.shape)
        outgoing = _reflect(incoming, mirror_orientation)

        initial = []
        r, c = marker
        for _step in range(depth):
            r, c = r + incoming[0], c + incoming[1]
            initial.append((r, c))
        assert initial[-1] == mirror_hit
        after = _cardinal_run(mirror_hit, outgoing, grid.shape)
        route = [marker, *initial, *after]
        route_set = set(route)
        if len(route_set) != len(route) or route_set & used_route:
            continue
        existing_optics = set().union(*optical_components) if optical_components else set()
        if route_set & existing_optics:
            continue

        before_candidates = [
            p for p in initial[:-1]
            if min(p[0], p[1], h - 1 - p[0], w - 1 - p[1]) >= 2
            and abs(p[0] - mirror_hit[0]) + abs(p[1] - mirror_hit[1]) >= 4
        ]
        after_candidates = [
            p for p in after
            if min(p[0], p[1], h - 1 - p[0], w - 1 - p[1]) >= 2
            and abs(p[0] - mirror_hit[0]) + abs(p[1] - mirror_hit[1]) >= 4
        ]
        candidates = before_candidates if glass_order == "before" else after_candidates
        if not candidates:
            continue
        random.shuffle(candidates)

        for glass_hit in candidates:
            mirror_options = _segment_options(mirror_hit, mirror_orientation, grid.shape)
            glass_options = _segment_options(glass_hit, glass_orientation, grid.shape)
            for mirror_cells in mirror_options:
                if mirror_cells & route_set != {mirror_hit}:
                    continue
                if any(not _separated(mirror_cells, old) for old in optical_components):
                    continue
                if (mirror_cells - {mirror_hit}) & used_route:
                    continue
                for glass_cells in glass_options:
                    if glass_cells & route_set != {glass_hit}:
                        continue
                    if not _separated(mirror_cells, glass_cells):
                        continue
                    if any(not _separated(glass_cells, old) for old in optical_components):
                        continue
                    if (glass_cells - {glass_hit}) & used_route:
                        continue

                    for pos in mirror_cells:
                        grid[pos] = MIRROR
                    for pos in glass_cells:
                        grid[pos] = GLASS
                    grid[marker] = marker_color
                    used_route.update(route_set)
                    optical_components.extend((mirror_cells, glass_cells))
                    return {
                        "marker": marker,
                        "color": marker_color,
                        "route": route,
                        "mirror_hit": mirror_hit,
                        "glass_hit": glass_hit,
                        "mirror_orientation": mirror_orientation,
                        "glass_orientation": glass_orientation,
                        "glass_order": glass_order,
                    }
    return None


def _add_distractors(
    grid: np.ndarray, count: int, used_route: set[Coord], optical_components: list[set[Coord]]
) -> None:
    h, w = grid.shape
    for _ in range(count):
        for _attempt in range(150):
            hit = (random.randint(2, h - 3), random.randint(2, w - 3))
            orientation = random.choice(DIAGONALS)
            options = _segment_options(hit, orientation, grid.shape)
            if not options:
                continue
            cells = random.choice(options)
            if cells & used_route:
                continue
            if any(not _separated(cells, old) for old in optical_components):
                continue
            color = random.choice((MIRROR, GLASS))
            for pos in cells:
                grid[pos] = color
            optical_components.append(cells)
            break


def _expected_from_records(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for record in records:
        for pos in record["route"][1:]:
            if int(grid[pos]) == BACKGROUND:
                expected[pos] = int(record["color"])
    return expected


def _simulate_alternative(
    grid: np.ndarray,
    reflecting: set[int],
    *,
    stop_at_optics: bool = False,
    forced_mirror_orientation: str | None = None,
) -> np.ndarray:
    """Small independent counterfactual tracer used only to reject shortcut-friendly scenes."""
    bg = _background(grid)
    orientations = _optical_orientations(grid)
    out = grid.copy()
    for start, color, direction in _boundary_markers(grid, bg):
        r, c = start
        seen: set[tuple[int, int, int, int]] = set()
        for _ in range(grid.size * 5):
            dr, dc = direction
            r, c = r + dr, c + dc
            if not _in_bounds(grid.shape, (r, c)):
                break
            state = (r, c, dr, dc)
            if state in seen:
                break
            seen.add(state)
            cell = int(grid[r, c])
            if cell == bg:
                out[r, c] = color
            elif cell in OPTICAL_COLORS:
                if stop_at_optics:
                    break
                if cell in reflecting:
                    orientation = orientations[(r, c)]
                    if cell == MIRROR and forced_mirror_orientation is not None:
                        orientation = forced_mirror_orientation
                    direction = _reflect(direction, orientation)
            else:
                break
    return out


def _validate_scene(grid: np.ndarray, records: list[dict], components: list[set[Coord]]) -> bool:
    if not (2 <= len(records) <= 4):
        return False
    if {r["mirror_orientation"] for r in records} != set(DIAGONALS):
        return False
    if {r["glass_orientation"] for r in records} != set(DIAGONALS):
        return False
    if {r["glass_order"] for r in records} != {"before", "after"}:
        return False
    if any(not _separated(a, b) for i, a in enumerate(components) for b in components[i + 1 :]):
        return False

    expected = _expected_from_records(grid, records)
    try:
        actual = main(grid)
    except (AssertionError, KeyError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False

    # Each tempting simplification must visibly disagree with the semantic oracle.
    counterfactuals = (
        _simulate_alternative(grid, set()),                 # treat mirrors as glass
        _simulate_alternative(grid, {MIRROR, GLASS}),       # treat glass as mirror
        _simulate_alternative(grid, set(), stop_at_optics=True),
        _simulate_alternative(grid, {MIRROR}, forced_mirror_orientation="\\"),
        _simulate_alternative(grid, {MIRROR}, forced_mirror_orientation="/"),
    )
    return all(not np.array_equal(expected, wrong) for wrong in counterfactuals)


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(18, 27)
    w = random.randint(18, 27)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    ray_count = random.randint(2, 4)
    marker_colors = random.sample(MARKER_COLORS, ray_count)

    edges = random.sample(("top", "bottom", "left", "right"), ray_count)
    random.shuffle(edges)
    mirror_orientations = ["\\", "/", *[random.choice(DIAGONALS) for _ in range(ray_count - 2)]]
    glass_orientations = ["/", "\\", *[random.choice(DIAGONALS) for _ in range(ray_count - 2)]]
    glass_orders = ["before", "after", *[random.choice(("before", "after")) for _ in range(ray_count - 2)]]
    tail = list(zip(mirror_orientations, glass_orientations, glass_orders))
    random.shuffle(tail)

    used_route: set[Coord] = set()
    components: list[set[Coord]] = []
    records: list[dict] = []
    for edge, color, (mirror_orientation, glass_orientation, glass_order) in zip(
        edges, marker_colors, tail
    ):
        record = _try_add_ray(
            grid,
            edge,
            color,
            mirror_orientation,
            glass_orientation,
            glass_order,
            used_route,
            components,
        )
        if record is None:
            return None
        records.append(record)

    _add_distractors(grid, random.randint(0, 2), used_route, components)
    if _background(grid) != BACKGROUND:
        return None
    return grid if _validate_scene(grid, records, components) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate mirror/glass ray-tracing input."""
    for _ in range(350):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-and-glass light-path scene")
