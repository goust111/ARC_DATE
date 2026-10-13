import random
from typing import Optional

import numpy as np


BG = 0
RED = 2
GREY = 5
RELAY_COLORS = (1, 3, 4, 6, 7, 8, 9)

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PERPENDICULAR_DIRS = {
    "up": ["left", "right"],
    "down": ["left", "right"],
    "left": ["up", "down"],
    "right": ["up", "down"],
}

# Offsets are expressed relative to the head pixel.
ARROW_OFFSETS = {
    "up": ((0, 0), (1, -1), (1, 1)),
    "down": ((0, 0), (-1, -1), (-1, 1)),
    "left": ((0, 0), (-1, 1), (1, 1)),
    "right": ((0, 0), (-1, -1), (1, -1)),
}

PATTERN_TO_ARROW = {
    frozenset({(0, 0), (2, 0), (1, 1)}): ("right", (1, 1)),
    frozenset({(0, 0), (0, 2), (1, 1)}): ("down", (1, 1)),
    frozenset({(0, 1), (2, 1), (1, 0)}): ("left", (1, 0)),
    frozenset({(1, 0), (1, 2), (0, 1)}): ("up", (0, 1)),
}

GEN_RETRIES = 300
GRID_MIN = 15
GRID_MAX = 30
RELAY_COUNT_MIN = 1
RELAY_COUNT_MAX = 3
MIN_SEGMENT_TO_ARROW = 4
MIN_SEGMENT_TO_BOUNDARY = 4
DISTRACTOR_COUNT_MAX = 3
DISTRACTOR_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _arrow_cells(head: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in ARROW_OFFSETS[direction]}


def _expand_chebyshev(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _line_cells(head: tuple[int, int], direction: str, start_step: int, end_step: int) -> list[tuple[int, int]]:
    dr, dc = DIR_VECS[direction]
    hr, hc = head
    return [(hr + dr * step, hc + dc * step) for step in range(start_step, end_step + 1)]


def _max_head_distance(head: tuple[int, int], direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = head
    if direction == "up":
        return r - 1
    if direction == "down":
        return (h - 2) - r
    if direction == "left":
        return c - 1
    return (w - 2) - c


def _edge_distance(head: tuple[int, int], direction: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = head
    if direction == "up":
        return r
    if direction == "down":
        return h - 1 - r
    if direction == "left":
        return c
    return w - 1 - c


def _monochrome_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if not _in_bounds(g.shape, nr, nc):
                            continue
                        if seen[nr, nc] or int(g[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _decode_arrow(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    pattern = frozenset((r - r0, c - c0) for r, c in coords)
    if pattern not in PATTERN_TO_ARROW:
        raise AssertionError("every foreground component must be a valid 3-pixel arrow")
    direction, (hr, hc) = PATTERN_TO_ARROW[pattern]
    return direction, (r0 + hr, c0 + hc)


def _extract_arrows(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    arrows: list[dict[str, object]] = []
    pixel_to_arrow: dict[tuple[int, int], int] = {}

    for comp in _monochrome_components(grid):
        coords = [tuple(map(int, p)) for p in comp["coords"]]
        if len(coords) != 3:
            raise AssertionError("every arrow must contain exactly 3 pixels")
        direction, head = _decode_arrow(coords)
        color = int(comp["color"])
        role = "relay"
        if color == RED:
            role = "emitter"
        elif color == GREY:
            role = "terminal"

        arrow = {
            "color": color,
            "role": role,
            "direction": direction,
            "head": head,
            "cells": set(coords),
            "vec": DIR_VECS[direction],
        }
        idx = len(arrows)
        arrows.append(arrow)
        for cell in coords:
            pixel_to_arrow[cell] = idx

    return arrows, pixel_to_arrow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a colored signal through arrow relays and draw each straight ray segment on the grid.

    Input structure:
    - bg = black.
    - Every foreground object is a monochrome 3-pixel arrow whose shape encodes one of the four cardinal directions.
    - There is exactly one red emitter arrow, zero or more non-red/non-grey relay arrows, and optionally one or more grey terminal arrows.

    Rule:
    - Start from the red arrow's head and follow its facing direction, painting traversed background cells red.
    - A ray continues straight until it reaches any pixel of another arrow or exits the grid.
    - If the hit arrow is a non-grey relay, restart the signal from that relay's head, change the ray color to the relay color, and follow the relay's own direction.
    - If the hit arrow is grey, stop immediately; all arrows remain unchanged throughout.

    Key insight:
    - The arrow glyph itself carries the control information: its head location gives the launch direction, while the arrow color determines the color of the next ray segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows, pixel_to_arrow = _extract_arrows(g)
    emitters = [idx for idx, arrow in enumerate(arrows) if arrow["role"] == "emitter"]
    assert len(emitters) == 1, "expected exactly one red emitter arrow"

    out = g.copy()
    current = emitters[0]
    launched: set[int] = set()
    h, w = g.shape

    while True:
        if current in launched:
            raise AssertionError("signal chain must not cycle through relays")
        launched.add(current)

        arrow = arrows[current]
        color = int(arrow["color"])
        dr, dc = arrow["vec"]
        r, c = arrow["head"]

        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                return out

            hit = pixel_to_arrow.get((r, c))
            if hit is not None:
                role = str(arrows[hit]["role"])
                if role == "terminal":
                    return out
                if role != "relay":
                    raise AssertionError("only relays may continue the signal")
                current = hit
                break

            assert int(g[r, c]) == BG, "rays may only traverse background cells"
            out[r, c] = color


def _try_place_arrow(
    head: tuple[int, int],
    direction: str,
    shape: tuple[int, int],
    occupied_buffer: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
) -> Optional[set[tuple[int, int]]]:
    cells = _arrow_cells(head, direction)
    if not all(_in_bounds(shape, r, c) for r, c in cells):
        return None
    if cells & occupied_buffer:
        return None
    if cells & reserved_path:
        return None
    return cells


def _extend_chain(
    shape: tuple[int, int],
    remaining_relays: int,
    terminal_mode: bool,
    current_head: tuple[int, int],
    current_dir: str,
    arrow_specs: list[dict[str, object]],
    occupied_buffer: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
    rng: random.Random,
) -> Optional[tuple[list[dict[str, object]], set[tuple[int, int]]]]:
    if remaining_relays == 0:
        if terminal_mode:
            max_dist = _max_head_distance(current_head, current_dir, shape)
            if max_dist < MIN_SEGMENT_TO_ARROW:
                return None

            distances = list(range(MIN_SEGMENT_TO_ARROW, max_dist + 1))
            rng.shuffle(distances)
            term_dirs = list(DIR_VECS)
            rng.shuffle(term_dirs)

            for dist in distances:
                target_head = _line_cells(current_head, current_dir, dist, dist)[0]
                pre_path = set(_line_cells(current_head, current_dir, 1, dist - 1))
                if pre_path & reserved_path:
                    continue

                for term_dir in term_dirs:
                    cells = _try_place_arrow(target_head, term_dir, shape, occupied_buffer, reserved_path | pre_path)
                    if cells is None:
                        continue
                    terminal = {"head": target_head, "direction": term_dir, "role": "terminal", "cells": cells}
                    return arrow_specs + [terminal], reserved_path | pre_path
            return None

        final_len = _edge_distance(current_head, current_dir, shape)
        if final_len < MIN_SEGMENT_TO_BOUNDARY:
            return None
        final_cells = set(_line_cells(current_head, current_dir, 1, final_len))
        if final_cells & reserved_path:
            return None
        return arrow_specs, reserved_path | final_cells

    max_dist = _max_head_distance(current_head, current_dir, shape)
    if max_dist < MIN_SEGMENT_TO_ARROW:
        return None

    distances = list(range(MIN_SEGMENT_TO_ARROW, max_dist + 1))
    rng.shuffle(distances)
    next_dirs = PERPENDICULAR_DIRS[current_dir][:]
    rng.shuffle(next_dirs)

    for dist in distances:
        target_head = _line_cells(current_head, current_dir, dist, dist)[0]
        pre_path = set(_line_cells(current_head, current_dir, 1, dist - 1))
        if pre_path & reserved_path:
            continue

        for next_dir in next_dirs:
            cells = _try_place_arrow(target_head, next_dir, shape, occupied_buffer, reserved_path | pre_path)
            if cells is None:
                continue

            next_spec = {"head": target_head, "direction": next_dir, "role": "relay", "cells": cells}
            result = _extend_chain(
                shape=shape,
                remaining_relays=remaining_relays - 1,
                terminal_mode=terminal_mode,
                current_head=target_head,
                current_dir=next_dir,
                arrow_specs=arrow_specs + [next_spec],
                occupied_buffer=occupied_buffer | _expand_chebyshev(cells),
                reserved_path=reserved_path | pre_path,
                rng=rng,
            )
            if result is not None:
                return result

    return None


def _place_active_chain(
    shape: tuple[int, int],
    relay_count: int,
    terminal_mode: bool,
    rng: random.Random,
) -> Optional[tuple[list[dict[str, object]], set[tuple[int, int]]]]:
    h, w = shape
    directions = list(DIR_VECS)
    rng.shuffle(directions)

    start_heads = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(start_heads)

    for start_dir in directions:
        for start_head in start_heads:
            cells = _arrow_cells(start_head, start_dir)
            if not all(_in_bounds(shape, r, c) for r, c in cells):
                continue

            start_spec = {"head": start_head, "direction": start_dir, "role": "emitter", "cells": cells}
            result = _extend_chain(
                shape=shape,
                remaining_relays=relay_count,
                terminal_mode=terminal_mode,
                current_head=start_head,
                current_dir=start_dir,
                arrow_specs=[start_spec],
                occupied_buffer=_expand_chebyshev(cells),
                reserved_path=set(),
                rng=rng,
            )
            if result is not None:
                return result

    return None


def _render(shape: tuple[int, int], arrow_specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in arrow_specs:
        color = int(spec["color"])
        for r, c in spec["cells"]:
            grid[r, c] = color
    return grid


def _place_distractor(
    shape: tuple[int, int],
    arrow_specs: list[dict[str, object]],
    reserved_path: set[tuple[int, int]],
    color: int,
    rng: random.Random,
) -> Optional[dict[str, object]]:
    occupied = set().union(*(spec["cells"] for spec in arrow_specs))
    occupied_buffer = _expand_chebyshev(occupied)

    for _ in range(DISTRACTOR_ATTEMPTS):
        head = (rng.randint(1, shape[0] - 2), rng.randint(1, shape[1] - 2))
        direction = rng.choice(list(DIR_VECS))
        cells = _arrow_cells(head, direction)
        if not all(_in_bounds(shape, r, c) for r, c in cells):
            continue
        if cells & occupied_buffer:
            continue
        if cells & reserved_path:
            continue
        return {"head": head, "direction": direction, "role": "terminal" if color == GREY else "relay", "cells": cells, "color": color}

    return None


def _assign_colors(active_specs: list[dict[str, object]], relay_count: int, rng: random.Random) -> list[dict[str, object]]:
    colored: list[dict[str, object]] = []
    relay_colors = rng.sample(RELAY_COLORS, relay_count)
    relay_idx = 0

    for spec in active_specs:
        spec2 = dict(spec)
        if spec["role"] == "emitter":
            spec2["color"] = RED
        elif spec["role"] == "terminal":
            spec2["color"] = GREY
        else:
            spec2["color"] = relay_colors[relay_idx]
            relay_idx += 1
        colored.append(spec2)

    return colored


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    relay_count = rng.randint(RELAY_COUNT_MIN, RELAY_COUNT_MAX)
    terminal_mode = rng.random() < 0.5
    shape = (rng.randint(GRID_MIN, GRID_MAX), rng.randint(GRID_MIN, GRID_MAX))

    chain = _place_active_chain(shape, relay_count, terminal_mode, rng)
    if chain is None:
        return None
    active_specs, reserved_path = chain

    arrow_specs = _assign_colors(active_specs, relay_count, rng)

    # In boundary-ending cases, keep at least one grey distractor so both arrow roles appear in input.
    mandatory_grey = 0 if terminal_mode else 1
    for _ in range(mandatory_grey):
        distractor = _place_distractor(shape, arrow_specs, reserved_path, GREY, rng)
        if distractor is None:
            return None
        arrow_specs.append(distractor)

    extra_count = rng.randint(0, DISTRACTOR_COUNT_MAX)
    for _ in range(extra_count):
        color = GREY if rng.random() < 0.35 else rng.choice(RELAY_COLORS)
        distractor = _place_distractor(shape, arrow_specs, reserved_path, color, rng)
        if distractor is not None:
            arrow_specs.append(distractor)

    grid = _render(shape, arrow_specs)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid signal-relay puzzle input")
