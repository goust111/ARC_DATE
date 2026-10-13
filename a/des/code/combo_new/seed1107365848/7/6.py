from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    MAGENTA = 6
    ORANGE = 7
    MAROON = 9


SOURCE_COLORS = (Color.RED, Color.BLUE, Color.YELLOW)
INPUT_COLORS = {Color.BLACK, Color.GREY, *SOURCE_COLORS}
MIX_COLOR = {
    frozenset({Color.RED}): Color.RED,
    frozenset({Color.BLUE}): Color.BLUE,
    frozenset({Color.YELLOW}): Color.YELLOW,
    frozenset({Color.RED, Color.BLUE}): Color.MAGENTA,
    frozenset({Color.RED, Color.YELLOW}): Color.ORANGE,
    frozenset({Color.BLUE, Color.YELLOW}): Color.GREEN,
    frozenset({Color.RED, Color.BLUE, Color.YELLOW}): Color.MAROON,
}
ALL_SOURCE_SETS = (
    (Color.RED,),
    (Color.BLUE,),
    (Color.YELLOW,),
    (Color.RED, Color.BLUE),
    (Color.RED, Color.YELLOW),
    (Color.BLUE, Color.YELLOW),
    (Color.RED, Color.BLUE, Color.YELLOW),
)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 26
WALL_COUNT_MIN = 1
WALL_COUNT_MAX = 3
ROOM_SPAN_MIN = 3
WALL_MARGIN = 3
WALL_SEPARATION_MIN = ROOM_SPAN_MIN + 1
TARGET_COMPONENT_MIN = 2
TARGET_COMPONENT_MAX = 4
EXTRA_GAP_PROB = 0.2
DUPLICATE_SOURCE_PROB = 0.35
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in NEIGHBORS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _open_components(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == Color.GREY:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            source_set: set[int] = set()

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                value = int(grid[cr, cc])
                if value in SOURCE_COLORS:
                    source_set.add(value)

                for nr, nc in _neighbors4(cr, cc, height, width):
                    if seen[nr, nc] or int(grid[nr, nc]) == Color.GREY:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            components.append({"coords": coords, "sources": frozenset(source_set)})

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-connected background region with the mix of all source colors that can reach it through gaps.

    Input structure:
    - The background is black.
    - Grey pixels form one-cell-thick horizontal and vertical wall segments, with some one-cell gaps.
    - Single source pixels in red, blue, and yellow sit on non-wall cells.

    Rule:
    - Treat every non-grey cell as open space, so wall gaps connect the background areas on both sides.
    - For each 4-connected open region, collect the set of source colors present anywhere in that region.
    - Recolor every black cell in the region with the corresponding single color or mixed color:
      red+blue -> magenta, red+yellow -> orange, blue+yellow -> green, all three -> maroon.
    - Keep all source pixels in their original source colors and keep all grey wall pixels unchanged.

    Key insight:
    - The hidden structure is the topology of the non-wall space: a wall gap merges two apparent rooms into one
      reachability component, and the final color depends only on the full set of sources inside that component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    unexpected = set(map(int, np.unique(g))) - INPUT_COLORS
    assert not unexpected, f"unexpected input colors: {sorted(unexpected)}"
    assert np.any(g == Color.GREY), "expected at least one grey wall"
    assert np.any(np.isin(g, SOURCE_COLORS)), "expected at least one source pixel"

    out = g.copy()
    for component in _open_components(g):
        source_set = component["sources"]
        if not source_set:
            continue

        fill_color = MIX_COLOR[frozenset(source_set)]
        for r, c in component["coords"]:
            if int(g[r, c]) == Color.BLACK:
                out[r, c] = fill_color

    return out


class _DSU:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.count = size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.count -= 1
        return True


def _choose_wall_positions(length: int, count: int, rng: random.Random) -> list[int] | None:
    choices = list(range(WALL_MARGIN, length - WALL_MARGIN))
    if len(choices) < count:
        return None

    for _ in range(200):
        positions = sorted(rng.sample(choices, count))
        if all(b - a >= WALL_SEPARATION_MIN for a, b in zip(positions, positions[1:])):
            return positions
    return None


def _build_rooms(
    height: int,
    width: int,
    horizontal_walls: list[int],
    vertical_walls: list[int],
):
    row_bounds = [-1] + horizontal_walls + [height]
    col_bounds = [-1] + vertical_walls + [width]
    rooms: list[dict[str, object]] = []
    room_lookup: dict[tuple[int, int], int] = {}

    room_id = 0
    for band_r in range(len(row_bounds) - 1):
        for band_c in range(len(col_bounds) - 1):
            r0 = row_bounds[band_r] + 1
            r1 = row_bounds[band_r + 1] - 1
            c0 = col_bounds[band_c] + 1
            c1 = col_bounds[band_c + 1] - 1
            rooms.append({"bbox": (r0, c0, r1, c1), "band_r": band_r, "band_c": band_c})
            room_lookup[(band_r, band_c)] = room_id
            room_id += 1

    return rooms, row_bounds, col_bounds, room_lookup


def _build_room_edges(
    horizontal_walls: list[int],
    vertical_walls: list[int],
    row_bounds: list[int],
    col_bounds: list[int],
    room_lookup: dict[tuple[int, int], int],
) -> list[dict[str, object]]:
    room_rows = len(horizontal_walls) + 1
    room_cols = len(vertical_walls) + 1
    edges: list[dict[str, object]] = []

    for band_r in range(room_rows):
        gap_r0 = row_bounds[band_r] + 1
        gap_r1 = row_bounds[band_r + 1] - 1
        for wall_index, wall_col in enumerate(vertical_walls):
            a = room_lookup[(band_r, wall_index)]
            b = room_lookup[(band_r, wall_index + 1)]
            cells = [(r, wall_col) for r in range(gap_r0, gap_r1 + 1)]
            edges.append({"a": a, "b": b, "cells": cells})

    for wall_index, wall_row in enumerate(horizontal_walls):
        for band_c in range(room_cols):
            a = room_lookup[(wall_index, band_c)]
            b = room_lookup[(wall_index + 1, band_c)]
            gap_c0 = col_bounds[band_c] + 1
            gap_c1 = col_bounds[band_c + 1] - 1
            cells = [(wall_row, c) for c in range(gap_c0, gap_c1 + 1)]
            edges.append({"a": a, "b": b, "cells": cells})

    return edges


def _choose_open_edges(
    room_count: int,
    edges: list[dict[str, object]],
    rng: random.Random,
) -> tuple[set[int], list[list[int]]] | None:
    max_components = min(TARGET_COMPONENT_MAX, room_count - 2)
    if max_components < TARGET_COMPONENT_MIN:
        return None

    target_components = rng.randint(TARGET_COMPONENT_MIN, max_components)
    order = list(range(len(edges)))
    rng.shuffle(order)

    dsu = _DSU(room_count)
    open_edges: set[int] = set()
    for edge_index in order:
        if dsu.count == target_components:
            break
        edge = edges[edge_index]
        if dsu.union(int(edge["a"]), int(edge["b"])):
            open_edges.add(edge_index)

    if dsu.count != target_components or not open_edges:
        return None

    for edge_index in order:
        if edge_index in open_edges:
            continue
        edge = edges[edge_index]
        if dsu.find(int(edge["a"])) == dsu.find(int(edge["b"])) and rng.random() < EXTRA_GAP_PROB:
            open_edges.add(edge_index)

    groups: dict[int, list[int]] = defaultdict(list)
    for room_id in range(room_count):
        groups[dsu.find(room_id)].append(room_id)

    components = sorted(groups.values(), key=len, reverse=True)
    if not any(len(component) > 1 for component in components):
        return None

    return open_edges, components


def _choose_component_source_sets(component_count: int, rng: random.Random) -> list[tuple[int, ...]]:
    focus = rng.choice(ALL_SOURCE_SETS)
    selected: list[tuple[int, ...]] = [focus]

    if component_count >= 2:
        contrast_pool = [
            source_set
            for source_set in ALL_SOURCE_SETS
            if source_set != focus and (len(source_set) == 1) != (len(focus) == 1)
        ]
        selected.append(rng.choice(contrast_pool))

    remaining = [source_set for source_set in ALL_SOURCE_SETS if source_set not in selected]
    rng.shuffle(remaining)
    while len(selected) < component_count:
        if remaining:
            selected.append(remaining.pop())
        else:
            selected.append(rng.choice(ALL_SOURCE_SETS))

    if len(selected) > 1:
        tail = selected[1:]
        rng.shuffle(tail)
        selected = [selected[0]] + tail

    return selected


def _room_free_cells(room_bbox: tuple[int, int, int, int], occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in occupied]


def _place_sources_in_component(
    grid: np.ndarray,
    rooms: list[dict[str, object]],
    room_ids: list[int],
    source_set: tuple[int, ...],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    shuffled_rooms = room_ids[:]
    rng.shuffle(shuffled_rooms)
    colors = list(source_set)
    rng.shuffle(colors)
    used_rooms: set[int] = set()

    for color in colors:
        candidate_rooms = [room_id for room_id in shuffled_rooms if room_id not in used_rooms]
        if not candidate_rooms:
            candidate_rooms = shuffled_rooms[:]
        rng.shuffle(candidate_rooms)

        placed = False
        for room_id in candidate_rooms:
            cells = _room_free_cells(rooms[room_id]["bbox"], occupied)
            if not cells:
                continue
            cell = rng.choice(cells)
            grid[cell] = color
            occupied.add(cell)
            used_rooms.add(room_id)
            placed = True
            break

        if not placed:
            return False

    if rng.random() < DUPLICATE_SOURCE_PROB:
        duplicate_color = rng.choice(colors)
        candidate_rooms = shuffled_rooms[:]
        rng.shuffle(candidate_rooms)
        for room_id in candidate_rooms:
            cells = _room_free_cells(rooms[room_id]["bbox"], occupied)
            if not cells:
                continue
            cell = rng.choice(cells)
            grid[cell] = duplicate_color
            occupied.add(cell)
            break

    return True


def _validate_generated_grid(grid: np.ndarray) -> bool:
    components = _open_components(grid)
    if len(components) < TARGET_COMPONENT_MIN:
        return False
    if any(not component["sources"] for component in components):
        return False
    if not any(len(component["sources"]) == 1 for component in components):
        return False
    if not any(len(component["sources"]) > 1 for component in components):
        return False

    out = main(grid)
    mixed_colors = {Color.MAGENTA, Color.ORANGE, Color.GREEN, Color.MAROON}
    if not np.any(np.isin(out, list(mixed_colors))):
        return False

    for component in components:
        fill_color = MIX_COLOR[frozenset(component["sources"])]
        visible_fill = 0
        for r, c in component["coords"]:
            if int(grid[r, c]) == Color.BLACK:
                if int(out[r, c]) != fill_color:
                    return False
                visible_fill += 1
            elif int(grid[r, c]) in SOURCE_COLORS and int(out[r, c]) != int(grid[r, c]):
                return False
        if visible_fill == 0:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    horizontal_count = rng.randint(WALL_COUNT_MIN, WALL_COUNT_MAX)
    vertical_count = rng.randint(WALL_COUNT_MIN, WALL_COUNT_MAX)

    horizontal_walls = _choose_wall_positions(height, horizontal_count, rng)
    vertical_walls = _choose_wall_positions(width, vertical_count, rng)
    if horizontal_walls is None or vertical_walls is None:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    for wall_row in horizontal_walls:
        grid[wall_row, :] = Color.GREY
    for wall_col in vertical_walls:
        grid[:, wall_col] = Color.GREY

    rooms, row_bounds, col_bounds, room_lookup = _build_rooms(height, width, horizontal_walls, vertical_walls)
    edges = _build_room_edges(horizontal_walls, vertical_walls, row_bounds, col_bounds, room_lookup)
    chosen = _choose_open_edges(len(rooms), edges, rng)
    if chosen is None:
        return None
    open_edges, components = chosen

    for edge_index in open_edges:
        gap_cell = rng.choice(edges[edge_index]["cells"])
        grid[gap_cell] = Color.BLACK

    source_sets = _choose_component_source_sets(len(components), rng)
    occupied: set[tuple[int, int]] = set()
    for room_ids, source_set in zip(components, source_sets):
        if not _place_sources_in_component(grid, rooms, room_ids, source_set, occupied, rng):
            return None

    if not _validate_generated_grid(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
