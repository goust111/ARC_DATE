import random
from collections import Counter, defaultdict, deque

import numpy as np


Coord = tuple[int, int]
Component = dict[str, object]


TRACK_COLOR = 7  # orange


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def find_components(grid: np.ndarray, bg: int) -> list[Component]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = np.min(arr, axis=0)
            r1, c1 = np.max(arr, axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return components


def _segment_orientation(coords: np.ndarray) -> str | None:
    rows = sorted(int(r) for r in coords[:, 0])
    cols = sorted(int(c) for c in coords[:, 1])
    if len(set(rows)) == 1 and cols == list(range(cols[0], cols[0] + len(cols))):
        return "h"
    if len(set(cols)) == 1 and rows == list(range(rows[0], rows[0] + len(rows))):
        return "v"
    return None


def _pair_portals(grid: np.ndarray, bg: int, wall_color: int) -> dict[Coord, Coord]:
    comps = find_components(grid, bg=bg)
    singleton_by_color: dict[int, list[Coord]] = defaultdict(list)
    for comp in comps:
        color = int(comp["color"])
        if color == wall_color:
            continue
        coords = comp["coords"]
        if int(len(coords)) == 1:
            singleton_by_color[color].append(tuple(map(int, coords[0])))

    portal_map: dict[Coord, Coord] = {}
    for points in singleton_by_color.values():
        if len(points) != 2:
            continue
        a, b = points
        portal_map[a] = b
        portal_map[b] = a
    return portal_map


def _object_components(grid: np.ndarray, bg: int, wall_color: int) -> list[tuple[Component, str]]:
    objects: list[tuple[Component, str]] = []
    for comp in find_components(grid, bg=bg):
        if int(comp["color"]) == wall_color:
            continue
        coords = comp["coords"]
        if len(coords) not in (2, 3):
            continue
        orientation = _segment_orientation(coords)
        if orientation is not None:
            objects.append((comp, orientation))
    return objects


def _front_cell(coords: list[Coord], direction: Coord) -> Coord:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return max(coords, key=lambda p: p[1])
    if (dr, dc) == (1, 0):
        return max(coords, key=lambda p: p[0])
    raise AssertionError("unsupported direction")


def _emerge_from_portal(portal: Coord, direction: Coord, length: int) -> list[Coord]:
    r, c = portal
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return [(r, c + 1 + i) for i in range(length)]
    if (dr, dc) == (1, 0):
        return [(r + 1 + i, c) for i in range(length)]
    raise AssertionError("unsupported direction")


def _simulate_object(
    grid: np.ndarray,
    coords: list[Coord],
    direction: Coord,
    obstacle_mask: np.ndarray,
    portal_map: dict[Coord, Coord],
    bg: int,
) -> tuple[list[Coord], set[Coord]]:
    current = sorted(coords)
    visited_black: set[Coord] = set()
    teleported = False

    while True:
        for cell in current:
            if int(grid[cell]) == bg:
                visited_black.add(cell)

        front_r, front_c = _front_cell(current, direction)
        next_r = front_r + direction[0]
        next_c = front_c + direction[1]
        if not in_bounds(grid.shape, next_r, next_c):
            return current, visited_black

        if bool(obstacle_mask[next_r, next_c]):
            if not teleported and (next_r, next_c) in portal_map:
                current = _emerge_from_portal(portal_map[(next_r, next_c)], direction, len(current))
                for r, c in current:
                    assert in_bounds(grid.shape, r, c), "portal exit must fit"
                    assert not bool(obstacle_mask[r, c]), "portal exit side must be clear"
                teleported = True
                continue
            return current, visited_black

        current = [(r + direction[0], c + direction[1]) for r, c in current]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each short line object toward the next wall, letting it pass through a paired wall portal and leaving an orange trail on the traversed corridor.

    Input structure:
    - The background is black.
    - Grey pixels form one-pixel-thick horizontal and vertical wall segments that
      partition the board into regions.
    - Some wall cells are replaced by paired single-pixel portal colors; each
      portal color appears exactly twice.
    - Moving objects are monochrome length-2 or length-3 horizontal or vertical
      line segments whose colors differ from the wall and portal colors.

    Rule:
    - A horizontal object moves right; a vertical object moves down.
    - The object advances until its front would hit a wall. If that contacted wall
      cell is a portal, the whole object reappears on the exit side of the paired
      portal and continues in the same direction.
    - After at most one teleport, the object keeps moving until the next wall or
      grid edge and then stops.
    - Clear each object's start position. Any originally black cell that the
      moving object covers anywhere along its route becomes orange, except for
      the object's final resting cells, which are repainted with the object color.
      Walls and portal pixels stay unchanged.

    Key insight:
    - The segment orientation encodes motion direction, while the singleton wall
      colors are not decorations but paired portals that relocate the moving
      segment before its final straight run.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    non_bg_colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg_colors, "expected walls or objects"
    counts = Counter(int(v) for v in g.ravel() if int(v) != bg)
    wall_color = max(non_bg_colors, key=lambda color: counts[color])

    portal_map = _pair_portals(g, bg=bg, wall_color=wall_color)
    portal_cells = set(portal_map)
    obstacle_mask = (g == wall_color) | np.isin(g, list({int(g[p]) for p in portal_cells}))

    objects = _object_components(g, bg=bg, wall_color=wall_color)
    assert objects, "expected at least one moving object"

    out = g.copy()
    for comp, _orientation in objects:
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg

    traces: set[Coord] = set()
    finals: list[tuple[list[Coord], int]] = []
    for comp, orientation in objects:
        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        direction = (0, 1) if orientation == "h" else (1, 0)
        final_coords, visited_black = _simulate_object(
            g,
            coords,
            direction,
            obstacle_mask=obstacle_mask,
            portal_map=portal_map,
            bg=bg,
        )
        traces.update(visited_black - set(final_coords))
        finals.append((final_coords, int(comp["color"])))

    for r, c in traces:
        if int(out[r, c]) == bg:
            out[r, c] = TRACK_COLOR
    for coords, color in finals:
        for r, c in coords:
            out[r, c] = color
    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_WALL = 5
GEN_TRACK = 7
GEN_OBJECT_COLORS = [1, 2, 3]
GEN_PORTAL_COLORS = [6, 8, 9, 4]
GEN_BAND_MIN = 6
GEN_BAND_MAX = 9
GEN_LENS = [2, 3]
GEN_RETRIES = 300


def _sample_bands() -> tuple[list[int], list[int]]:
    row_bands = [random.randint(GEN_BAND_MIN, GEN_BAND_MAX) for _ in range(3)]
    col_bands = [random.randint(GEN_BAND_MIN, GEN_BAND_MAX) for _ in range(3)]
    return row_bands, col_bands


def _build_canvas(row_bands: list[int], col_bands: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[int], list[int]]:
    h = sum(row_bands) + 2
    w = sum(col_bands) + 2
    grid = np.full((h, w), GEN_BG, dtype=int)

    h1 = row_bands[0]
    h2 = row_bands[0] + 1 + row_bands[1]
    v1 = col_bands[0]
    v2 = col_bands[0] + 1 + col_bands[1]

    grid[h1, :] = GEN_WALL
    grid[h2, :] = GEN_WALL
    grid[:, v1] = GEN_WALL
    grid[:, v2] = GEN_WALL

    row_ranges = [
        (0, h1 - 1),
        (h1 + 1, h2 - 1),
        (h2 + 1, h - 1),
    ]
    col_ranges = [
        (0, v1 - 1),
        (v1 + 1, v2 - 1),
        (v2 + 1, w - 1),
    ]
    return grid, row_ranges, col_ranges, [h1, h2], [v1, v2]


def _region_width(col_ranges: list[tuple[int, int]], idx: int) -> int:
    left, right = col_ranges[idx]
    return right - left + 1


def _region_height(row_ranges: list[tuple[int, int]], idx: int) -> int:
    top, bottom = row_ranges[idx]
    return bottom - top + 1


def _place_horizontal(
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    v_walls: list[int],
    reserved: set[tuple[int, int]],
    blocked_portals: set[Coord],
    teleport: bool,
    color: int,
    h: int,
    w: int,
) -> dict[str, object] | None:
    for _ in range(200):
        source_candidates = [(rb, cb) for rb in range(3) for cb in (0, 1) if (rb, cb) not in reserved]
        if not source_candidates:
            return None
        rb, cb = random.choice(source_candidates)
        width = _region_width(col_ranges, cb)
        lengths = [length for length in GEN_LENS if width - length >= (2 if teleport else length + 1)]
        if not lengths:
            continue
        length = random.choice(lengths)
        row = random.randint(*row_ranges[rb])
        wall_col = v_walls[cb]
        left, _right = col_ranges[cb]
        min_steps = 2 if teleport else length + 1
        max_start = wall_col - length - min_steps
        if max_start < left:
            continue
        start_col = random.randint(left, max_start)
        cells = [(row, start_col + i) for i in range(length)]
        stop_cell = (row, wall_col)

        if not teleport:
            return {
                "orientation": "h",
                "color": color,
                "cells": cells,
                "source_region": (rb, cb),
                "dest_region": (rb, cb),
                "portals": [],
                "blocked": [stop_cell],
            }

        exit_col = v_walls[1 - cb]
        dest_cb = 1 if exit_col == v_walls[0] else 2
        if _region_width(col_ranges, dest_cb) < length:
            continue
        exit_rb = random.randrange(3)
        dest_region = (exit_rb, dest_cb)
        if dest_region in reserved and dest_region != (rb, cb):
            continue
        exit_row = random.randint(*row_ranges[exit_rb])
        entry = stop_cell
        exit_portal = (exit_row, exit_col)
        if entry in blocked_portals or exit_portal in blocked_portals or entry == exit_portal:
            continue

        emerge_cells = [(exit_row, exit_col + 1 + i) for i in range(length)]
        if any(not in_bounds((h, w), r, c) for r, c in emerge_cells):
            continue
        next_steps = (
            col_ranges[dest_cb][1] - emerge_cells[-1][1]
            if dest_cb == 2
            else v_walls[1] - 1 - emerge_cells[-1][1]
        )
        if next_steps < 1:
            continue

        blocked = [entry]
        if dest_cb == 1:
            blocked.append((exit_row, v_walls[1]))
        return {
            "orientation": "h",
            "color": color,
            "cells": cells,
            "source_region": (rb, cb),
            "dest_region": dest_region,
            "portals": [entry, exit_portal],
            "blocked": blocked,
        }
    return None


def _place_vertical(
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    h_walls: list[int],
    reserved: set[tuple[int, int]],
    blocked_portals: set[Coord],
    teleport: bool,
    color: int,
    h: int,
    w: int,
) -> dict[str, object] | None:
    for _ in range(200):
        source_candidates = [(rb, cb) for rb in (0, 1) for cb in range(3) if (rb, cb) not in reserved]
        if not source_candidates:
            return None
        rb, cb = random.choice(source_candidates)
        height = _region_height(row_ranges, rb)
        lengths = [length for length in GEN_LENS if height - length >= (2 if teleport else length + 1)]
        if not lengths:
            continue
        length = random.choice(lengths)
        col = random.randint(*col_ranges[cb])
        wall_row = h_walls[rb]
        top, _bottom = row_ranges[rb]
        min_steps = 2 if teleport else length + 1
        max_start = wall_row - length - min_steps
        if max_start < top:
            continue
        start_row = random.randint(top, max_start)
        cells = [(start_row + i, col) for i in range(length)]
        stop_cell = (wall_row, col)

        if not teleport:
            return {
                "orientation": "v",
                "color": color,
                "cells": cells,
                "source_region": (rb, cb),
                "dest_region": (rb, cb),
                "portals": [],
                "blocked": [stop_cell],
            }

        exit_row = h_walls[1 - rb]
        dest_rb = 1 if exit_row == h_walls[0] else 2
        if _region_height(row_ranges, dest_rb) < length:
            continue
        exit_cb = random.randrange(3)
        dest_region = (dest_rb, exit_cb)
        if dest_region in reserved and dest_region != (rb, cb):
            continue
        exit_col = random.randint(*col_ranges[exit_cb])
        entry = stop_cell
        exit_portal = (exit_row, exit_col)
        if entry in blocked_portals or exit_portal in blocked_portals or entry == exit_portal:
            continue

        emerge_cells = [(exit_row + 1 + i, exit_col) for i in range(length)]
        if any(not in_bounds((h, w), r, c) for r, c in emerge_cells):
            continue
        next_steps = (
            row_ranges[dest_rb][1] - emerge_cells[-1][0]
            if dest_rb == 2
            else h_walls[1] - 1 - emerge_cells[-1][0]
        )
        if next_steps < 1:
            continue

        blocked = [entry]
        if dest_rb == 1:
            blocked.append((h_walls[1], exit_col))
        return {
            "orientation": "v",
            "color": color,
            "cells": cells,
            "source_region": (rb, cb),
            "dest_region": dest_region,
            "portals": [entry, exit_portal],
            "blocked": blocked,
        }
    return None


def _paint_object(grid: np.ndarray, spec: dict[str, object]) -> None:
    for r, c in spec["cells"]:
        grid[r, c] = int(spec["color"])


def _add_portal_pair(grid: np.ndarray, color: int, p1: Coord, p2: Coord) -> None:
    grid[p1] = color
    grid[p2] = color


def _place_unused_portal_pair(
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    h_walls: list[int],
    v_walls: list[int],
    blocked_portals: set[Coord],
) -> tuple[Coord, Coord] | None:
    choices: list[tuple[Coord, Coord]] = []
    for exit_row in range(3):
        for r1 in range(row_ranges[exit_row][0], row_ranges[exit_row][1] + 1):
            p = (r1, v_walls[0])
            q = (r1, v_walls[1])
            if p not in blocked_portals and q not in blocked_portals:
                choices.append((p, q))
    for exit_col in range(3):
        for c1 in range(col_ranges[exit_col][0], col_ranges[exit_col][1] + 1):
            p = (h_walls[0], c1)
            q = (h_walls[1], c1)
            if p not in blocked_portals and q not in blocked_portals:
                choices.append((p, q))
    if not choices:
        return None
    return random.choice(choices)


def _gen_attempt() -> np.ndarray | None:
    row_bands, col_bands = _sample_bands()
    grid, row_ranges, col_ranges, h_walls, v_walls = _build_canvas(row_bands, col_bands)
    h, w = grid.shape

    reserved: set[tuple[int, int]] = set()
    blocked_portals: set[Coord] = set()
    objects: list[dict[str, object]] = []
    portal_pairs: list[tuple[int, Coord, Coord]] = []

    object_colors = random.sample(GEN_OBJECT_COLORS, k=3)
    portal_colors = random.sample(GEN_PORTAL_COLORS, k=4)

    horizontal_teleport = bool(random.getrandbits(1))
    vertical_teleport = bool(random.getrandbits(1))

    horizontal = _place_horizontal(
        row_ranges,
        col_ranges,
        v_walls,
        reserved,
        blocked_portals,
        teleport=horizontal_teleport,
        color=object_colors[0],
        h=h,
        w=w,
    )
    if horizontal is None:
        return None
    objects.append(horizontal)
    reserved.add(horizontal["source_region"])
    reserved.add(horizontal["dest_region"])
    blocked_portals.update(horizontal["blocked"])
    if horizontal["portals"]:
        portal_pairs.append((portal_colors[len(portal_pairs)], horizontal["portals"][0], horizontal["portals"][1]))
        blocked_portals.update(horizontal["portals"])

    vertical = _place_vertical(
        row_ranges,
        col_ranges,
        h_walls,
        reserved,
        blocked_portals,
        teleport=vertical_teleport,
        color=object_colors[1],
        h=h,
        w=w,
    )
    if vertical is None:
        return None
    objects.append(vertical)
    reserved.add(vertical["source_region"])
    reserved.add(vertical["dest_region"])
    blocked_portals.update(vertical["blocked"])
    if vertical["portals"]:
        portal_pairs.append((portal_colors[len(portal_pairs)], vertical["portals"][0], vertical["portals"][1]))
        blocked_portals.update(vertical["portals"])

    if random.random() < 0.55:
        extra_orientation = random.choice(["h", "v"])
        extra_spec = (
            _place_horizontal(
                row_ranges,
                col_ranges,
                v_walls,
                reserved,
                blocked_portals,
                teleport=False,
                color=object_colors[2],
                h=h,
                w=w,
            )
            if extra_orientation == "h"
            else _place_vertical(
                row_ranges,
                col_ranges,
                h_walls,
                reserved,
                blocked_portals,
                teleport=False,
                color=object_colors[2],
                h=h,
                w=w,
            )
        )
        if extra_spec is not None:
            objects.append(extra_spec)
            reserved.add(extra_spec["source_region"])
            blocked_portals.update(extra_spec["blocked"])

    if not portal_pairs:
        spare = _place_unused_portal_pair(row_ranges, col_ranges, h_walls, v_walls, blocked_portals)
        if spare is None:
            return None
        portal_pairs.append((portal_colors[0], spare[0], spare[1]))
        blocked_portals.update(spare)
    elif len(portal_pairs) < 3 and random.random() < 0.35:
        spare = _place_unused_portal_pair(row_ranges, col_ranges, h_walls, v_walls, blocked_portals)
        if spare is not None:
            portal_pairs.append((portal_colors[len(portal_pairs)], spare[0], spare[1]))
            blocked_portals.update(spare)

    for color, p1, p2 in portal_pairs:
        _add_portal_pair(grid, color, p1, p2)
    for spec in objects:
        _paint_object(grid, spec)

    if GEN_TRACK in np.unique(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-wall input")
