import random
from collections import deque

import numpy as np


BG_COLOR = 0
ARROW_COLOR = 2
WALL_COLOR = 5
BLOCK_PALETTE = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATE_ATTEMPTS = 400

ARROW_SPECS = {
    "right": {
        "cells": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
        "front": (1, 2),
        "delta": (0, 1),
    },
    "left": {
        "cells": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
        "front": (1, 0),
        "delta": (0, -1),
    },
    "up": {
        "cells": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
        "front": (0, 1),
        "delta": (-1, 0),
    },
    "down": {
        "cells": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
        "front": (2, 1),
        "delta": (1, 0),
    },
}

SCENARIO_TYPES = ("push_wall", "push_edge", "noop_wall", "noop_edge")
QUADRANT_DIRECTIONS = {
    "tl": ("left", "up"),
    "tr": ("right", "up"),
    "bl": ("left", "down"),
    "br": ("right", "down"),
}


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        comp: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            comp.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(comp)
    return comps


def _match_arrow_component(comp: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(comp) != 5:
        return None
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0 = min(rows)
    c0 = min(cols)
    if max(rows) - r0 != 2 or max(cols) - c0 != 2:
        return None
    rel = frozenset((r - r0, c - c0) for r, c in comp)
    for name, spec in ARROW_SPECS.items():
        if rel != spec["cells"]:
            continue
        fr, fc = spec["front"]
        dr, dc = spec["delta"]
        return {
            "name": name,
            "front": (r0 + fr, c0 + fc),
            "delta": (dr, dc),
            "cells": tuple(sorted(comp)),
        }
    return None


def _find_arrow_color_and_arrows(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    best_color = None
    best_arrows = None
    best_score = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components_of_color(grid, color)
        arrows = []
        unmatched = 0
        for comp in comps:
            matched = _match_arrow_component(comp)
            if matched is None:
                unmatched += 1
            else:
                arrows.append(matched)
        if not arrows:
            continue
        score = (len(arrows), -unmatched, int(np.sum(grid == color)))
        if best_score is None or score > best_score:
            best_color = color
            best_arrows = arrows
            best_score = score
    assert best_color is not None and best_arrows is not None, "expected at least one arrow"
    best_arrows.sort(key=lambda arrow: arrow["front"])
    return int(best_color), best_arrows


def _infer_wall_color(grid: np.ndarray, bg: int, arrow_color: int) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    candidates = [
        (int(count), int(color))
        for color, count in zip(colors, counts)
        if int(color) not in (bg, arrow_color)
    ]
    assert candidates, "expected wall and block colors"
    candidates.sort()
    return candidates[-1][1]


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)
    arrow_color, arrows = _find_arrow_color_and_arrows(g, bg)
    wall_color = _infer_wall_color(g, bg, arrow_color)

    out = g.copy()
    arrow_cells = set()
    for arrow in arrows:
        for cell in arrow["cells"]:
            arrow_cells.add(cell)
            out[cell] = bg

    moves: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    seen_sources: set[tuple[int, int]] = set()
    seen_dests: set[tuple[int, int]] = set()
    for arrow in arrows:
        fr, fc = arrow["front"]
        dr, dc = arrow["delta"]
        rr = fr + dr
        cc = fc + dc
        source = None
        while 0 <= rr < h and 0 <= cc < w:
            value = int(g[rr, cc])
            if value == wall_color:
                break
            if value == bg or value == arrow_color:
                rr += dr
                cc += dc
                continue
            source = (rr, cc)
            break
        if source is None:
            continue

        assert source not in seen_sources, "a block cannot be selected by multiple arrows"
        seen_sources.add(source)
        color = int(g[source])
        dst_r, dst_c = source
        rr = dst_r + dr
        cc = dst_c + dc
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != wall_color:
            value = int(g[rr, cc])
            assert value == bg or value == arrow_color, "movement path must stay otherwise empty"
            dst_r, dst_c = rr, cc
            rr += dr
            cc += dc
        dest = (dst_r, dst_c)
        assert dest not in seen_dests, "moved blocks must land on distinct cells"
        seen_dests.add(dest)
        moves.append((source, dest, color))

    for source, _, _ in moves:
        out[source] = bg
    for _, dest, color in moves:
        out[dest] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the first colored square seen by each arrow until it reaches the nearest wall or the grid boundary.

    Input structure:
    - The background is black.
    - A single wall color draws thin horizontal or vertical wall segments on the edges and inside the grid.
    - Several colored 1x1 squares sit in the open floor cells.
    - Several red 5-pixel arrow sprites point up, down, left, or right.

    Rule:
    - For each arrow, cast a ray from the arrow tip along its pointing direction.
    - If the ray hits a wall or leaves the grid before reaching a colored square, that arrow has no effect.
    - Otherwise move the first colored square on that ray in the same direction until the next step would hit a wall or leave the grid.
    - Remove every arrow from the output while keeping all walls and all untouched squares unchanged.

    Key insight:
    - Solving the task requires two separate inferences: decode direction from each 5-pixel arrow shape, then use ray casting plus wall-aware stopping logic to relocate exactly one square per arrow.
    """
    return _transform(grid)


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - pad), min(h, r + pad + 1)):
            for nc in range(max(0, c - pad), min(w, c + pad + 1)):
                expanded.add((nr, nc))
    return expanded


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _arrow_from_bbox(direction: str, r0: int, c0: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    spec = ARROW_SPECS[direction]
    cells = [(r0 + dr, c0 + dc) for dr, dc in spec["cells"]]
    fr, fc = spec["front"]
    return cells, (r0 + fr, c0 + fc)


def _ray_cells(front: tuple[int, int], delta: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = delta
    r = front[0] + dr
    c = front[1] + dc
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _next_block_color(colors: list[int], index_box: list[int]) -> int:
    color = colors[index_box[0] % len(colors)]
    index_box[0] += 1
    return color


def _free_cells(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return all(cell not in occupied for cell in cells)


def _wall_segment_candidates(
    zone: tuple[int, int, int, int],
    hit_cell: tuple[int, int],
    direction: str,
) -> list[list[tuple[int, int]]]:
    r0, r1, c0, c1 = zone
    r, c = hit_cell
    candidates = []
    if direction in ("left", "right"):
        for top in range(r0 + 1, r + 1):
            for bottom in range(r, r1):
                if bottom - top + 1 < 3:
                    continue
                if top == r0 or bottom == r1:
                    continue
                candidates.append([(rr, c) for rr in range(top, bottom + 1)])
    else:
        for left in range(c0 + 1, c + 1):
            for right in range(c, c1):
                if right - left + 1 < 3:
                    continue
                if left == c0 or right == c1:
                    continue
                candidates.append([(r, cc) for cc in range(left, right + 1)])
    return candidates


def _choose_direction_map(rng: random.Random) -> dict[str, str]:
    while True:
        direction_map = {
            quadrant: rng.choice(options)
            for quadrant, options in QUADRANT_DIRECTIONS.items()
        }
        chosen = set(direction_map.values())
        if any(direction in chosen for direction in ("left", "right")) and any(
            direction in chosen for direction in ("up", "down")
        ):
            return direction_map


def _scenario_geometry(
    zone: tuple[int, int, int, int],
    direction: str,
    shape: tuple[int, int],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]] | None:
    r0, r1, c0, c1 = zone
    for _ in range(60):
        if direction == "right":
            row = rng.randint(r0 + 1, r1 - 1)
            bbox_c0 = rng.randint(c0, c1 - 5)
            arrow_cells, front = _arrow_from_bbox(direction, row - 1, bbox_c0)
        elif direction == "left":
            row = rng.randint(r0 + 1, r1 - 1)
            bbox_c0 = rng.randint(c0 + 3, c1 - 2)
            arrow_cells, front = _arrow_from_bbox(direction, row - 1, bbox_c0)
        elif direction == "down":
            col = rng.randint(c0 + 1, c1 - 1)
            bbox_r0 = rng.randint(r0, r1 - 5)
            arrow_cells, front = _arrow_from_bbox(direction, bbox_r0, col - 1)
        else:
            col = rng.randint(c0 + 1, c1 - 1)
            bbox_r0 = rng.randint(r0 + 3, r1 - 2)
            arrow_cells, front = _arrow_from_bbox(direction, bbox_r0, col - 1)
        line = _ray_cells(front, ARROW_SPECS[direction]["delta"], shape)
        if len(line) >= 5:
            return arrow_cells, front, line
    return None


def _place_scenario(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    direction: str,
    scenario: str,
    rng: random.Random,
    block_colors: list[int],
    index_box: list[int],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> bool:
    shape = grid.shape
    for _ in range(80):
        geometry = _scenario_geometry(zone, direction, shape, rng)
        if geometry is None:
            return False
        arrow_cells, _front, line = geometry
        if not _free_cells(arrow_cells, occupied):
            continue

        wall_cells: list[tuple[int, int]] = []
        block_cell = None
        if scenario == "push_wall":
            wall_idx = rng.randint(2, len(line) - 2)
            candidates = _wall_segment_candidates(zone, line[wall_idx], direction)
            candidates = [cells for cells in candidates if _free_cells(cells, occupied)]
            if not candidates:
                continue
            wall_cells = rng.choice(candidates)
            block_idx = rng.randint(0, wall_idx - 2)
            block_cell = line[block_idx]
        elif scenario == "push_edge":
            block_idx = rng.randint(0, len(line) - 2)
            block_cell = line[block_idx]
        elif scenario == "noop_wall":
            wall_idx = rng.randint(0, len(line) - 2)
            candidates = _wall_segment_candidates(zone, line[wall_idx], direction)
            candidates = [cells for cells in candidates if _free_cells(cells, occupied)]
            if not candidates:
                continue
            wall_cells = rng.choice(candidates)
            if wall_idx + 1 < len(line) and rng.random() < 0.75:
                block_cell = rng.choice(line[wall_idx + 1 :])
        else:
            block_cell = None

        paint_cells = list(arrow_cells) + wall_cells + ([] if block_cell is None else [block_cell])
        if not _free_cells(paint_cells, occupied):
            continue

        _paint(grid, arrow_cells, ARROW_COLOR)
        for cell in arrow_cells:
            occupied.add(cell)
        if wall_cells:
            _paint(grid, wall_cells, WALL_COLOR)
            for cell in wall_cells:
                occupied.add(cell)
        if block_cell is not None:
            block_color = _next_block_color(block_colors, index_box)
            grid[block_cell] = block_color
            occupied.add(block_cell)

        reserve = set(line) | set(arrow_cells) | set(wall_cells)
        if block_cell is not None:
            reserve.add(block_cell)
        protected.update(_expand(reserve, shape, pad=1))
        return True
    return False


def _add_internal_wall(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    rng: random.Random,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> None:
    r0, r1, c0, c1 = zone
    if r1 - r0 < 4 or c1 - c0 < 4:
        return
    for _ in range(40):
        if rng.random() < 0.5:
            row = rng.randint(r0 + 1, r1 - 1)
            length = rng.randint(3, max(3, min(6, c1 - c0 - 1)))
            start = rng.randint(c0 + 1, c1 - length)
            cells = [(row, c) for c in range(start, start + length)]
        else:
            col = rng.randint(c0 + 1, c1 - 1)
            length = rng.randint(3, max(3, min(6, r1 - r0 - 1)))
            start = rng.randint(r0 + 1, r1 - length)
            cells = [(r, col) for r in range(start, start + length)]
        if any(cell in protected or cell in occupied for cell in cells):
            continue
        _paint(grid, cells, WALL_COLOR)
        for cell in cells:
            occupied.add(cell)
        protected.update(_expand(set(cells), grid.shape, pad=1))
        return


def _add_decoy_blocks(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    rng: random.Random,
    block_colors: list[int],
    index_box: list[int],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> None:
    r0, r1, c0, c1 = zone
    count = rng.randint(0, 2)
    for _ in range(count):
        for _ in range(30):
            r = rng.randint(r0 + 1, r1 - 1)
            c = rng.randint(c0 + 1, c1 - 1)
            cell = (r, c)
            if cell in occupied or cell in protected:
                continue
            grid[cell] = _next_block_color(block_colors, index_box)
            occupied.add(cell)
            protected.update(_expand({cell}, grid.shape, pad=1))
            break


def _add_edge_wall(grid: np.ndarray, rng: random.Random, occupied: set[tuple[int, int]], protected: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for _ in range(80):
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            row = 0 if side == "top" else h - 1
            length = rng.randint(2, min(6, w))
            start = rng.randint(0, w - length)
            cells = [(row, c) for c in range(start, start + length)]
        else:
            col = 0 if side == "left" else w - 1
            length = rng.randint(2, min(6, h))
            start = rng.randint(0, h - length)
            cells = [(r, col) for r in range(start, start + length)]
        if any(cell in protected or cell in occupied for cell in cells):
            continue
        _paint(grid, cells, WALL_COLOR)
        for cell in cells:
            occupied.add(cell)
        protected.update(_expand(set(cells), grid.shape, pad=0))
        return True
    return False


def _zones_for_grid(h: int, w: int, rng: random.Random) -> dict[str, tuple[int, int, int, int]]:
    gap_r = rng.randint(2, 4)
    gap_c = rng.randint(2, 4)
    top_h = rng.randint(8, h - gap_r - 8)
    left_w = rng.randint(8, w - gap_c - 8)
    bottom_r0 = top_h + gap_r
    right_c0 = left_w + gap_c
    return {
        "tl": (0, top_h - 1, 0, left_w - 1),
        "tr": (0, top_h - 1, right_c0, w - 1),
        "bl": (bottom_r0, h - 1, 0, left_w - 1),
        "br": (bottom_r0, h - 1, right_c0, w - 1),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    zones = _zones_for_grid(h, w, rng)
    scenario_types = list(SCENARIO_TYPES)
    rng.shuffle(scenario_types)
    directions = _choose_direction_map(rng)

    occupied: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()
    block_colors = list(BLOCK_PALETTE)
    rng.shuffle(block_colors)
    index_box = [0]

    for quadrant, scenario in zip(("tl", "tr", "bl", "br"), scenario_types):
        ok = _place_scenario(
            grid,
            zones[quadrant],
            directions[quadrant],
            scenario,
            rng,
            block_colors,
            index_box,
            occupied,
            protected,
        )
        if not ok:
            return None

    for quadrant in ("tl", "tr", "bl", "br"):
        if rng.random() < 0.65:
            _add_internal_wall(grid, zones[quadrant], rng, occupied, protected)
        _add_decoy_blocks(grid, zones[quadrant], rng, block_colors, index_box, occupied, protected)

    if not _add_edge_wall(grid, rng, occupied, protected):
        return None
    if rng.random() < 0.7:
        _add_edge_wall(grid, rng, occupied, protected)

    bg = _dominant_color(grid)
    arrow_color, arrows = _find_arrow_color_and_arrows(grid, bg)
    wall_color = _infer_wall_color(grid, bg, arrow_color)
    if arrow_color != ARROW_COLOR or wall_color != WALL_COLOR:
        return None
    if len(arrows) != 4:
        return None
    _ = _transform(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
