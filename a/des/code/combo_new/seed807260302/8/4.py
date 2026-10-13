import random
from collections import deque

import numpy as np


BG_COLOR = 0
WALL_COLOR = 5
ARROW_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
DIRS = (UP, RIGHT, DOWN, LEFT)
CW_DIR = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}

ARROW_WINGS = {
    UP: ((1, -1), (1, 1)),
    RIGHT: ((-1, -1), (1, -1)),
    DOWN: ((-1, -1), (-1, 1)),
    LEFT: ((-1, 1), (1, 1)),
}

GEN_H_RANGE = (18, 24)
GEN_W_RANGE = (18, 24)
GEN_RETRIES = 400
SCENARIOS = ("edge_plain", "wall_plain", "portal_edge", "portal_wall")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _add(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return cell[0] + steps * direction[0], cell[1] + steps * direction[1]


def _arrow_cells(head: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = head
    cells = {(r, c)}
    for dr, dc in ARROW_WINGS[direction]:
        cells.add((r + dr, c + dc))
    return cells


def _identify_arrow(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    matches = []
    for head in coords:
        for direction in DIRS:
            if _arrow_cells(head, direction) == coords:
                matches.append((head, direction))
    if len(matches) != 1:
        return None
    return matches[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            components.append(comp)
    return components


def _parse_scene(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall_mask = np.zeros_like(g, dtype=bool)
    portal_cells: set[tuple[int, int]] = set()
    arrows: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        arrow_parts: list[tuple[set[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]] = []
        singletons: list[tuple[int, int]] = []
        non_arrow_parts: list[list[tuple[int, int]]] = []

        for comp in comps:
            coord_set = set(comp)
            arrow_info = _identify_arrow(coord_set)
            if arrow_info is not None:
                arrow_parts.append((coord_set, arrow_info))
            elif len(comp) == 1:
                singletons.append(comp[0])
            else:
                non_arrow_parts.append(comp)

        if arrow_parts:
            assert len(arrow_parts) == 1, "each arrow color must contain exactly one arrow component"
            assert not non_arrow_parts, "arrow colors may only include the arrow and optional portals"
            assert len(singletons) in (0, 2), "portals must appear in pairs"
            arrow_cells, (head, direction) = arrow_parts[0]
            arrows.append(
                {
                    "color": color,
                    "head": head,
                    "direction": direction,
                    "cells": arrow_cells,
                    "portals": tuple(singletons),
                }
            )
            portal_cells.update(singletons)
        else:
            for comp in comps:
                for r, c in comp:
                    wall_mask[r, c] = True

    assert arrows, "at least one arrow is required"
    return bg, wall_mask, arrows, portal_cells


def _simulate_arrow(
    shape: tuple[int, int],
    head: tuple[int, int],
    direction: tuple[int, int],
    portals: tuple[tuple[int, int], ...],
    wall_mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    portal_jump = {}
    if len(portals) == 2:
        portal_jump[portals[0]] = portals[1]
        portal_jump[portals[1]] = portals[0]

    cur_head = head
    cur_dir = direction
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (cur_head[0], cur_head[1], cur_dir[0], cur_dir[1])
        assert state not in seen, "arrow trajectory must not cycle"
        seen.add(state)

        nxt = _add(cur_head, cur_dir)
        if not _in_bounds(shape, nxt):
            break
        if wall_mask[nxt]:
            break
        cur_head = nxt
        if cur_head in portal_jump:
            cur_head = portal_jump[cur_head]
            cur_dir = CW_DIR[cur_dir]

    return cur_head, cur_dir


def main(grid: np.ndarray) -> np.ndarray:
    """Launch each colored arrow forward, teleporting through same-color portal pairs before redrawing it at rest.

    Input structure:
    - The grid has a dominant black background, several 3-pixel monochrome arrows, optional same-color singleton portal pairs, and optional grey wall pixels on the boundary.
    - Each arrow is one 8-connected component whose head and two diagonal wing pixels encode one of the four cardinal directions.
    - Portal pixels share a color with exactly one arrow and are disconnected singleton components.

    Rule:
    - Erase every arrow from its starting position while leaving portals and boundary walls in place.
    - Move each arrow by advancing its head one cell at a time in the encoded direction.
    - Whenever the head lands on one of its own portal pixels, jump immediately to the paired portal and rotate the travel direction 90 degrees clockwise.
    - Stop when the next head step would leave the grid or enter a wall pixel, then draw the arrow in its final orientation and color while preserving portal pixels.

    Key insight:
    - The task combines local shape decoding for arrow orientation with full trajectory tracing, because same-color singleton pixels act as paired teleporters that also deterministically rotate the arrow.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_mask, arrows, portal_cells = _parse_scene(g)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = bg

    static_mask = wall_mask.copy()
    for r, c in portal_cells:
        static_mask[r, c] = True

    for arrow in arrows:
        final_head, final_dir = _simulate_arrow(
            g.shape,
            arrow["head"],
            arrow["direction"],
            arrow["portals"],
            wall_mask,
        )
        for r, c in _arrow_cells(final_head, final_dir):
            assert _in_bounds(g.shape, (r, c)), "final arrow must fit in the grid"
            if static_mask[r, c]:
                continue
            out[r, c] = arrow["color"]

    return out


def _head_path(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    return [_add(start, direction, k) for k in range(steps + 1)]


def _final_head_for_stop(
    h: int,
    w: int,
    direction: tuple[int, int],
    stop_kind: str,
    rng: random.Random,
) -> tuple[int, int]:
    if direction == RIGHT:
        return rng.randrange(1, h - 1), w - 1 if stop_kind == "edge" else w - 2
    if direction == LEFT:
        return rng.randrange(1, h - 1), 0 if stop_kind == "edge" else 1
    if direction == DOWN:
        return h - 1 if stop_kind == "edge" else h - 2, rng.randrange(1, w - 1)
    return 0 if stop_kind == "edge" else 1, rng.randrange(1, w - 1)


def _wall_segment(
    block_cell: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    rng: random.Random,
) -> set[tuple[int, int]]:
    h, w = shape
    r, c = block_cell
    cells = {block_cell}
    left_span = rng.randint(0, 2)
    right_span = rng.randint(0, 2)
    if direction in (LEFT, RIGHT):
        for rr in range(max(0, r - left_span), min(h - 1, r + right_span) + 1):
            cells.add((rr, c))
    else:
        for cc in range(max(0, c - left_span), min(w - 1, c + right_span) + 1):
            cells.add((r, cc))
    return cells


def _cells_disjoint(cells: set[tuple[int, int]], *others: set[tuple[int, int]]) -> bool:
    merged: set[tuple[int, int]] = set()
    for other in others:
        merged |= other
    return cells.isdisjoint(merged)


def _portal1_candidates(h: int, w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if direction == RIGHT and c >= 3:
                result.append((r, c))
            elif direction == LEFT and c <= w - 4:
                result.append((r, c))
            elif direction == DOWN and r >= 3:
                result.append((r, c))
            elif direction == UP and r <= h - 4:
                result.append((r, c))
    return result


def _starts_behind(target: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    starts: list[tuple[int, int]] = []
    step = 2
    while True:
        start = _add(target, direction, -step)
        if not _in_bounds(shape, start):
            break
        if all(_in_bounds(shape, cell) for cell in _arrow_cells(start, direction)):
            starts.append(start)
            step += 1
            continue
        break
    if direction == RIGHT:
        starts = [cell for cell in starts if cell[1] >= 1]
    elif direction == LEFT:
        starts = [cell for cell in starts if cell[1] <= w - 2]
    elif direction == DOWN:
        starts = [cell for cell in starts if cell[0] >= 1]
    else:
        starts = [cell for cell in starts if cell[0] <= h - 2]
    return starts


def _place_plain(
    shape: tuple[int, int],
    stop_kind: str,
    occupied_input: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    occupied_final: set[tuple[int, int]],
    static_output: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    h, w = shape
    directions = list(DIRS)
    rng.shuffle(directions)
    for direction in directions:
        for _ in range(50):
            final_head = _final_head_for_stop(h, w, direction, stop_kind, rng)
            final_cells = _arrow_cells(final_head, direction)
            if not _cells_disjoint(final_cells, occupied_paths, occupied_final, static_output):
                continue

            walls: set[tuple[int, int]] = set()
            if stop_kind == "wall":
                block_cell = _add(final_head, direction)
                walls = _wall_segment(block_cell, direction, shape, rng)
                if not _cells_disjoint(walls, occupied_input, occupied_paths, occupied_final):
                    continue
                if final_cells & walls:
                    continue

            start_candidates = _starts_behind(final_head, direction, shape)
            rng.shuffle(start_candidates)
            for start_head in start_candidates:
                path_cells = set(_head_path(start_head, direction, abs(final_head[0] - start_head[0]) + abs(final_head[1] - start_head[1])))
                start_cells = _arrow_cells(start_head, direction)
                if not _cells_disjoint(start_cells, occupied_input, occupied_paths, occupied_final):
                    continue
                if not _cells_disjoint(path_cells, occupied_input, occupied_paths, occupied_final):
                    continue
                return {
                    "direction": direction,
                    "start_head": start_head,
                    "start_cells": start_cells,
                    "portals": (),
                    "walls": walls,
                    "path_cells": path_cells,
                    "final_cells": final_cells,
                }
    return None


def _place_portal(
    shape: tuple[int, int],
    stop_kind: str,
    occupied_input: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    occupied_final: set[tuple[int, int]],
    static_output: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    h, w = shape
    directions = list(DIRS)
    rng.shuffle(directions)
    for direction in directions:
        post_direction = CW_DIR[direction]
        portal1_candidates = _portal1_candidates(h, w, direction)
        rng.shuffle(portal1_candidates)
        for _ in range(40):
            final_head = _final_head_for_stop(h, w, post_direction, stop_kind, rng)
            final_cells = _arrow_cells(final_head, post_direction)
            if not _cells_disjoint(final_cells, occupied_paths, occupied_final, static_output):
                continue

            walls: set[tuple[int, int]] = set()
            if stop_kind == "wall":
                block_cell = _add(final_head, post_direction)
                walls = _wall_segment(block_cell, post_direction, shape, rng)
                if not _cells_disjoint(walls, occupied_input, occupied_paths, occupied_final):
                    continue
                if final_cells & walls:
                    continue

            portal2_candidates = _starts_behind(final_head, post_direction, shape)
            portal2_candidates = [cell for cell in portal2_candidates if 0 < cell[0] < h - 1 and 0 < cell[1] < w - 1]
            rng.shuffle(portal2_candidates)
            for portal2 in portal2_candidates:
                post_steps = abs(final_head[0] - portal2[0]) + abs(final_head[1] - portal2[1])
                post_path = set(_head_path(portal2, post_direction, post_steps))
                if not _cells_disjoint(post_path, occupied_input, occupied_paths, occupied_final):
                    continue
                for portal1 in portal1_candidates:
                    if portal1 == portal2:
                        continue
                    if max(abs(portal1[0] - portal2[0]), abs(portal1[1] - portal2[1])) <= 1:
                        continue
                    if portal1 in post_path:
                        continue
                    start_candidates = _starts_behind(portal1, direction, shape)
                    rng.shuffle(start_candidates)
                    for start_head in start_candidates:
                        start_cells = _arrow_cells(start_head, direction)
                        if any(max(abs(cell[0] - portal1[0]), abs(cell[1] - portal1[1])) <= 1 for cell in start_cells):
                            continue
                        if any(max(abs(cell[0] - portal2[0]), abs(cell[1] - portal2[1])) <= 1 for cell in start_cells):
                            continue
                        pre_steps = abs(portal1[0] - start_head[0]) + abs(portal1[1] - start_head[1])
                        pre_path = set(_head_path(start_head, direction, pre_steps))
                        if portal2 in pre_path:
                            continue
                        if pre_path & post_path:
                            continue
                        portals = {portal1, portal2}
                        candidate_input = start_cells | portals | walls
                        full_path = pre_path | post_path
                        if not _cells_disjoint(candidate_input, occupied_input, occupied_paths, occupied_final):
                            continue
                        if not _cells_disjoint(full_path, occupied_input, occupied_paths, occupied_final):
                            continue
                        return {
                            "direction": direction,
                            "start_head": start_head,
                            "start_cells": start_cells,
                            "portals": (portal1, portal2),
                            "walls": walls,
                            "path_cells": full_path,
                            "final_cells": final_cells,
                        }
    return None


def _scenario_list(rng: random.Random) -> list[str]:
    if rng.random() < 0.5:
        omitted = rng.choice(SCENARIOS)
        scenarios = [name for name in SCENARIOS if name != omitted]
    else:
        scenarios = list(SCENARIOS)
    complexity = {"portal_wall": 0, "portal_edge": 1, "wall_plain": 2, "edge_plain": 3}
    scenarios.sort(key=lambda name: complexity[name])
    return scenarios


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_H_RANGE)
    w = rng.randint(*GEN_W_RANGE)
    shape = (h, w)

    scenarios = _scenario_list(rng)
    colors = rng.sample(ARROW_COLOR_POOL, k=len(scenarios))

    occupied_input: set[tuple[int, int]] = set()
    occupied_paths: set[tuple[int, int]] = set()
    occupied_final: set[tuple[int, int]] = set()
    static_output: set[tuple[int, int]] = set()
    specs: list[dict] = []

    for color, scenario in zip(colors, scenarios):
        if scenario.startswith("portal"):
            spec = _place_portal(
                shape,
                "edge" if scenario.endswith("edge") else "wall",
                occupied_input,
                occupied_paths,
                occupied_final,
                static_output,
                rng,
            )
        else:
            spec = _place_plain(
                shape,
                "edge" if scenario.startswith("edge") else "wall",
                occupied_input,
                occupied_paths,
                occupied_final,
                static_output,
                rng,
            )
        if spec is None:
            return None

        spec["color"] = color
        specs.append(spec)
        occupied_input |= spec["start_cells"] | set(spec["portals"]) | spec["walls"]
        occupied_paths |= spec["path_cells"]
        occupied_final |= spec["final_cells"]
        static_output |= set(spec["portals"]) | spec["walls"]

    grid = np.full(shape, BG_COLOR, dtype=int)
    for spec in specs:
        for r, c in spec["walls"]:
            grid[r, c] = WALL_COLOR
    for spec in specs:
        for r, c in spec["portals"]:
            grid[r, c] = spec["color"]
        for r, c in spec["start_cells"]:
            grid[r, c] = spec["color"]

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-portal ARC input")
