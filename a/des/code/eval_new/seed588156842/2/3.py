import random
from collections import defaultdict

import numpy as np


BG = 0
WIRE = 5
SIGNAL_COLORS = (1, 2, 3, 7)
PORTAL_COLORS = (4, 6, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
PERPENDICULARS = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}
EXIT_DIR_FROM_SIDE = {
    "top": RIGHT,
    "bottom": LEFT,
    "left": UP,
    "right": DOWN,
}
PORTAL_SIDE_FROM_DIR = {
    RIGHT: "top",
    LEFT: "bottom",
    UP: "left",
    DOWN: "right",
}

MIN_SIDE = 14
MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 300
MAX_PLACEMENT_ATTEMPTS = 300
MIN_SEG_LEN = 3
MAX_SEG_LEN = 7
MAX_SIGNALS = 3
MAX_PORTAL_HOPS = 3


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _portal_cells(touch: tuple[int, int], direction: tuple[int, int], mode: str) -> list[tuple[int, int]]:
    r, c = touch
    if direction == RIGHT:
        cols = (c - 1, c) if mode == "start" else (c, c + 1)
        rows = (r - 2, r - 1)
    elif direction == LEFT:
        cols = (c, c + 1) if mode == "start" else (c - 1, c)
        rows = (r + 1, r + 2)
    elif direction == UP:
        rows = (r, r + 1) if mode == "start" else (r - 1, r)
        cols = (c - 2, c - 1)
    elif direction == DOWN:
        rows = (r - 1, r) if mode == "start" else (r, r + 1)
        cols = (c + 1, c + 2)
    else:
        raise AssertionError(f"unexpected direction: {direction}")
    return [(rr, cc) for rr in rows for cc in cols]


def _touch_side_and_exit(
    portal_cells: list[tuple[int, int]], grid: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox(portal_cells)
    touches: set[tuple[int, int]] = set()
    for r, c in portal_cells:
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == WIRE:
                touches.add((nr, nc))
    assert len(touches) == 1, f"portal must touch exactly one wire pixel, found {touches}"
    touch = next(iter(touches))
    tr, tc = touch
    if tr == r0 - 1 and c0 <= tc <= c1:
        side = "top"
    elif tr == r1 + 1 and c0 <= tc <= c1:
        side = "bottom"
    elif tc == c0 - 1 and r0 <= tr <= r1:
        side = "left"
    elif tc == c1 + 1 and r0 <= tr <= r1:
        side = "right"
    else:
        raise AssertionError(f"touch {touch} is not side-adjacent to portal bbox {(r0, c0, r1, c1)}")
    return touch, EXIT_DIR_FROM_SIDE[side]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each colored endpoint signal through paired 2x2 portals and recolor every wire cell it traverses.

    Input structure:
    - Background cells are black, wires are one-cell-wide grey straight segments, and each signal source is a unique colored endpoint pixel attached to exactly one wire.
    - Each portal is a solid 2x2 block in a non-signal color, and each portal color appears exactly twice.
    - Every portal touches exactly one grey wire pixel; the touching side determines the portal's outgoing direction.

    Rule:
    - Start from each source and walk along its unique wire direction, repainting every grey cell with the source color.
    - When the walk reaches a wire pixel touching a portal, jump to the same-colored partner portal.
    - Resume from that partner's touching wire pixel, turning onto the orthogonal axis encoded by the exit portal, and continue until the wire ends or another portal is reached.
    - Portal blocks and source pixels keep their original colors.

    Key insight: The paired 2x2 blocks are teleporters, and the geometry of the exit portal tells the signal which perpendicular wire direction to take after the jump.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, WIRE)]
    portals_by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    sources: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for color in colors:
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == 4 and r1 == r0 + 1 and c1 == c0 + 1:
                portals_by_color[color].append({"color": color, "cells": comp})
                continue
            assert len(comp) == 1, f"unexpected non-portal component for color {color}: {comp}"
            r, c = comp[0]
            wire_neighbors = []
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == WIRE:
                    wire_neighbors.append((dr, dc))
            assert len(wire_neighbors) == 1, f"source at {(r, c)} must attach to exactly one wire"
            sources.append(((r, c), wire_neighbors[0], color))

    portals: list[dict[str, object]] = []
    portal_pairs: dict[int, int] = {}
    touch_to_portal: dict[tuple[int, int], int] = {}
    for color, same_color_portals in portals_by_color.items():
        assert len(same_color_portals) == 2, f"portal color {color} must appear exactly twice"
        ids = []
        for item in same_color_portals:
            cells = item["cells"]
            assert isinstance(cells, list)
            touch, exit_dir = _touch_side_and_exit(cells, g)
            idx = len(portals)
            portals.append(
                {
                    "color": color,
                    "cells": cells,
                    "touch": touch,
                    "exit_dir": exit_dir,
                }
            )
            assert touch not in touch_to_portal, f"wire pixel {touch} touches multiple portals"
            touch_to_portal[touch] = idx
            ids.append(idx)
        portal_pairs[ids[0]] = ids[1]
        portal_pairs[ids[1]] = ids[0]

    out = g.copy()
    for source_pos, direction, color in sources:
        pos = source_pos
        dr, dc = direction
        ignore_portal: int | None = None
        seen: set[tuple[int, int, int, int, int]] = set()
        while True:
            state = (pos[0], pos[1], dr, dc, -1 if ignore_portal is None else ignore_portal)
            assert state not in seen, "signal propagation entered a loop"
            seen.add(state)

            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if g[nr, nc] != WIRE:
                break

            pos = (nr, nc)
            out[nr, nc] = color
            portal_id = touch_to_portal.get(pos)
            if portal_id is None or portal_id == ignore_portal:
                ignore_portal = None
                continue

            pair_id = portal_pairs[portal_id]
            exit_dir = portals[pair_id]["exit_dir"]
            assert isinstance(exit_dir, tuple)
            assert (dr == 0) != (exit_dir[0] == 0), "portal exit must turn onto the orthogonal axis"
            exit_touch = portals[pair_id]["touch"]
            assert isinstance(exit_touch, tuple)
            pos = exit_touch
            out[pos[0], pos[1]] = color
            dr, dc = exit_dir
            ignore_portal = pair_id

    return out


def _all_compositions(total: int, parts: int, max_part: int) -> list[tuple[int, ...]]:
    if parts == 0:
        return [()] if total == 0 else []
    out: list[tuple[int, ...]] = []
    for first in range(min(max_part, total) + 1):
        for tail in _all_compositions(total - first, parts - 1, max_part):
            out.append((first,) + tail)
    return out


def _segment_object(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    signal_color: int,
    start_portal_color: int | None,
    end_portal_color: int | None,
    has_source: bool,
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    dr, dc = direction
    grey_cells = [(start[0] + i * dr, start[1] + i * dc) for i in range(length)]
    placements: list[tuple[tuple[int, int], int]] = [(cell, WIRE) for cell in grey_cells]

    if has_source:
        source = (start[0] - dr, start[1] - dc)
        placements.append((source, signal_color))
    if start_portal_color is not None:
        for cell in _portal_cells(grey_cells[0], direction, "start"):
            placements.append((cell, start_portal_color))
    if end_portal_color is not None:
        for cell in _portal_cells(grey_cells[-1], direction, "end"):
            placements.append((cell, end_portal_color))

    assert len({cell for cell, _ in placements}) == len(placements), "segment object must not self-overlap"
    cell_colors: dict[tuple[int, int], int] = {cell: color for cell, color in placements}
    footprint = set(cell_colors)
    return cell_colors, footprint


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))
    return halo


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _build_specs() -> list[dict[str, object]]:
    n_signals = random.randint(1, MAX_SIGNALS)
    total_pairs = random.randint(0, min(len(PORTAL_COLORS), n_signals * MAX_PORTAL_HOPS))
    compositions = _all_compositions(total_pairs, n_signals, MAX_PORTAL_HOPS)
    assert compositions
    counts = list(random.choice(compositions))
    random.shuffle(counts)

    signal_colors = random.sample(SIGNAL_COLORS, n_signals)
    portal_colors = list(random.sample(PORTAL_COLORS, total_pairs))
    portal_idx = 0
    specs: list[dict[str, object]] = []
    for signal_color, hops in zip(signal_colors, counts, strict=False):
        direction = random.choice(DIRS)
        directions = [direction]
        for _ in range(hops):
            direction = random.choice(PERPENDICULARS[direction])
            directions.append(direction)
        pair_colors = portal_colors[portal_idx : portal_idx + hops]
        portal_idx += hops
        for idx, seg_dir in enumerate(directions):
            specs.append(
                {
                    "signal_color": signal_color,
                    "direction": seg_dir,
                    "length": random.randint(MIN_SEG_LEN, MAX_SEG_LEN),
                    "has_source": idx == 0,
                    "start_portal_color": None if idx == 0 else pair_colors[idx - 1],
                    "end_portal_color": pair_colors[idx] if idx < hops else None,
                }
            )
    random.shuffle(specs)
    return specs


def _place_specs(specs: list[dict[str, object]], h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()
    order = sorted(
        specs,
        key=lambda spec: (
            int(spec["start_portal_color"] is not None) + int(spec["end_portal_color"] is not None),
            int(spec["length"]),
        ),
        reverse=True,
    )

    for spec in order:
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            start = (random.randrange(h), random.randrange(w))
            cells, footprint = _segment_object(
                start=start,
                direction=spec["direction"],
                length=int(spec["length"]),
                signal_color=int(spec["signal_color"]),
                start_portal_color=spec["start_portal_color"],
                end_portal_color=spec["end_portal_color"],
                has_source=bool(spec["has_source"]),
            )
            if not _in_bounds(footprint, h, w):
                continue
            if footprint & reserved:
                continue

            portal_cells = [cell for cell, color in cells.items() if color in PORTAL_COLORS]
            if portal_cells:
                test_grid = grid.copy()
                for (r, c), color in cells.items():
                    test_grid[r, c] = color
                for comp in _components(np.isin(test_grid, PORTAL_COLORS)):
                    comp_colors = {int(test_grid[r, c]) for r, c in comp}
                    if len(comp_colors) != 1:
                        return None

            for (r, c), color in cells.items():
                grid[r, c] = color
            reserved |= _halo(footprint, h, w)
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    specs = _build_specs()
    grid = _place_specs(specs, h, w)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    signal_cells = sum(int(np.count_nonzero(grid == color)) for color in SIGNAL_COLORS)
    if signal_cells < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-wire ARC input")
