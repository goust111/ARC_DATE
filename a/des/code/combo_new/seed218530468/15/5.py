from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


GREY = 5
BLACK = 0


DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _is_connected(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    if not cells:
        return False
    todo = [next(iter(cells))]
    seen = {todo[0]}
    while todo:
        r, c = todo.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == len(cells)


def _translate(cells: frozenset[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in cells)


def _box_cells(r0: int, c0: int, h: int, w: int) -> frozenset[tuple[int, int]]:
    return frozenset((r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w))


def _expand_chebyshev(
    cells: frozenset[tuple[int, int]] | set[tuple[int, int]],
    h: int,
    w: int,
) -> frozenset[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return frozenset(out)


def _in_bounds(cells: frozenset[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _container_pattern(direction: str) -> frozenset[tuple[int, int]]:
    cells = {(0, c) for c in range(5)}
    cells |= {(4, c) for c in range(5)}
    cells |= {(r, 0) for r in range(5)}
    cells |= {(r, 4) for r in range(5)}
    if direction == "up":
        cells -= {(0, 1), (0, 2), (0, 3)}
    elif direction == "down":
        cells -= {(4, 1), (4, 2), (4, 3)}
    elif direction == "left":
        cells -= {(1, 0), (2, 0), (3, 0)}
    elif direction == "right":
        cells -= {(1, 4), (2, 4), (3, 4)}
    else:
        raise ValueError(direction)
    return frozenset(cells)


CONTAINER_PATTERNS = {name: _container_pattern(name) for name in DIRS}


def _front_cells(cells: frozenset[tuple[int, int]], direction: str) -> frozenset[tuple[int, int]]:
    if direction == "up":
        edge = min(r for r, _ in cells)
        return frozenset((r, c) for r, c in cells if r == edge)
    if direction == "down":
        edge = max(r for r, _ in cells)
        return frozenset((r, c) for r, c in cells if r == edge)
    if direction == "left":
        edge = min(c for _, c in cells)
        return frozenset((r, c) for r, c in cells if c == edge)
    edge = max(c for _, c in cells)
    return frozenset((r, c) for r, c in cells if c == edge)


def _extract_object(window: np.ndarray, bg: int, gray: int) -> list[tuple[int, int, int]] | None:
    pts = [
        (r, c, int(window[r, c]))
        for r in range(1, 4)
        for c in range(1, 4)
        if int(window[r, c]) != bg and int(window[r, c]) != gray
    ]
    if not (2 <= len(pts) <= 4):
        return None
    cells = {(r, c) for r, c, _ in pts}
    if not _is_connected(cells):
        return None
    return pts


def _scan_containers(grid: np.ndarray, gray: int, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    gray_mask = grid == gray
    records: list[dict[str, object]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            window_mask = frozenset(
                (r, c)
                for r in range(5)
                for c in range(5)
                if bool(gray_mask[r0 + r, c0 + c])
            )
            matched_dir = None
            for direction, pattern in CONTAINER_PATTERNS.items():
                if window_mask == pattern:
                    matched_dir = direction
                    break
            if matched_dir is None:
                continue
            obj = _extract_object(grid[r0 : r0 + 5, c0 : c0 + 5], bg, gray)
            if obj is None:
                continue
            records.append(
                {
                    "top": int(r0),
                    "left": int(c0),
                    "direction": matched_dir,
                    "pixels": [(int(r0 + r), int(c0 + c), color) for r, c, color in obj],
                    "gray_cells": frozenset((int(r0 + r), int(c0 + c)) for r, c in CONTAINER_PATTERNS[matched_dir]),
                }
            )
    return records


def _detect_gray_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    scored: list[tuple[int, int]] = []
    for color in colors:
        count = len(_scan_containers(grid, color, bg))
        if count > 0:
            scored.append((count, color))
    if not scored:
        raise AssertionError("expected at least one U-container color")
    scored.sort()
    return int(scored[-1][1])


def _simulate_motion(
    initial: frozenset[tuple[int, int]],
    dr: int,
    dc: int,
    blockers: frozenset[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[frozenset[tuple[int, int]], str, frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    current = frozenset(initial)
    trail = set(current)
    while True:
        nxt = _translate(current, dr, dc)
        in_bounds_next = frozenset((r, c) for r, c in nxt if 0 <= r < h and 0 <= c < w)
        if len(in_bounds_next) != len(nxt):
            return current, "edge", frozenset(trail), in_bounds_next
        if nxt & blockers:
            return current, "bumper", frozenset(trail), in_bounds_next
        current = nxt
        trail.update(current)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each grey U-container and eject its interior object through the opening until it meets a blocker or the grid edge.

    Input structure:
    - The background is black.
    - Several grey 5x5 U-shaped containers appear on the grid; each is a square frame with the middle three pixels missing from exactly one side.
    - Inside each container's 3x3 cavity there is exactly one 4-connected colored object made of 2 to 4 pixels.
    - Separate grey 2x2 solid squares act as blockers, and they are not attached to the containers.

    Rule:
    - Detect the opening side of every grey U-container.
    - Move the colored object from that container straight through the opening direction.
    - Keep translating it until the next step would place any pixel on a grey blocker or outside the grid.
    - Remove all grey container pixels in the output; keep the moved colored objects and the original grey blockers.

    Key insight:
    The missing side of the U-frame is the direction cue, and the grey frame itself vanishes before movement, so only blockers and the border can stop the flying object.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    gray = _detect_gray_color(g, bg)
    containers = _scan_containers(g, gray, bg)

    container_cells: set[tuple[int, int]] = set()
    for record in containers:
        container_cells.update(record["gray_cells"])  # type: ignore[arg-type]

    blockers = frozenset(
        (int(r), int(c))
        for r in range(h)
        for c in range(w)
        if int(g[r, c]) == gray and (int(r), int(c)) not in container_cells
    )

    out = np.full_like(g, int(bg))
    for r, c in blockers:
        out[r, c] = gray

    for record in containers:
        direction = str(record["direction"])
        dr, dc = DIRS[direction]
        initial = frozenset((int(r), int(c)) for r, c, _color in record["pixels"])  # type: ignore[index]
        final_cells, _reason, _trail, _next = _simulate_motion(initial, dr, dc, blockers, h, w)
        shift_r = int(min(r for r, _ in final_cells) - min(r for r, _ in initial))
        shift_c = int(min(c for _, c in final_cells) - min(c for _, c in initial))
        for r0, c0, color in record["pixels"]:  # type: ignore[index]
            out[int(r0 + shift_r), int(c0 + shift_c)] = int(color)

    return out


def _all_cavity_shapes() -> dict[int, tuple[frozenset[tuple[int, int]], ...]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    by_size: dict[int, list[frozenset[tuple[int, int]]]] = {2: [], 3: [], 4: []}
    for size in (2, 3, 4):
        for combo in combinations(cells, size):
            shape = frozenset((int(r), int(c)) for r, c in combo)
            if _is_connected(shape):
                by_size[size].append(shape)
    return {size: tuple(shapes) for size, shapes in by_size.items()}


SHAPES_BY_SIZE = _all_cavity_shapes()
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 300
CASE_ATTEMPTS = 240


def _random_edge_origin(
    shape: frozenset[tuple[int, int]],
    direction: str,
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    r_lo = -min(rows)
    r_hi = h - 1 - max(rows)
    c_lo = -min(cols)
    c_hi = w - 1 - max(cols)
    if direction == "up":
        return -min(rows), rng.randint(c_lo, c_hi)
    if direction == "down":
        return h - 1 - max(rows), rng.randint(c_lo, c_hi)
    if direction == "left":
        return rng.randint(r_lo, r_hi), -min(cols)
    return rng.randint(r_lo, r_hi), w - 1 - max(cols)


def _random_inbounds_origin(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    r_lo = -min(rows)
    r_hi = h - 1 - max(rows)
    c_lo = -min(cols)
    c_hi = w - 1 - max(cols)
    return rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi)


def _feasible_steps(
    final_origin: tuple[int, int],
    dr: int,
    dc: int,
    h: int,
    w: int,
) -> list[int]:
    values: list[int] = []
    for steps in range(2, max(h, w) + 1):
        top = final_origin[0] - steps * dr - 1
        left = final_origin[1] - steps * dc - 1
        if 0 <= top <= h - 5 and 0 <= left <= w - 5:
            values.append(int(steps))
    return values


def _possible_bumpers(
    final_cells: frozenset[tuple[int, int]],
    direction: str,
    h: int,
    w: int,
) -> tuple[frozenset[tuple[int, int]], ...]:
    out: set[frozenset[tuple[int, int]]] = set()
    for r, c in _front_cells(final_cells, direction):
        if direction == "right":
            tops = ((r, c + 1), (r - 1, c + 1))
        elif direction == "left":
            tops = ((r, c - 2), (r - 1, c - 2))
        elif direction == "down":
            tops = ((r + 1, c), (r + 1, c - 1))
        else:
            tops = ((r - 2, c), (r - 2, c - 1))
        for br, bc in tops:
            bumper = _box_cells(int(br), int(bc), 2, 2)
            if _in_bounds(bumper, h, w) and not (bumper & final_cells):
                out.add(bumper)
    return tuple(out)


def _sample_case_specs(rng: random.Random) -> list[dict[str, object]]:
    n = rng.randint(2, 4)
    directions = list(DIRS.keys())
    rng.shuffle(directions)
    directions = directions[:n]

    if n % 2 == 0:
        stops = ["edge"] * (n // 2) + ["bumper"] * (n // 2)
    elif rng.random() < 0.5:
        stops = ["edge"] * ((n + 1) // 2) + ["bumper"] * (n // 2)
    else:
        stops = ["edge"] * (n // 2) + ["bumper"] * ((n + 1) // 2)
    rng.shuffle(stops)

    colors = rng.sample(list(OBJECT_COLORS), n)
    specs: list[dict[str, object]] = []
    for i in range(n):
        size = rng.choice((2, 3, 4))
        specs.append(
            {
                "direction": directions[i],
                "stop": stops[i],
                "color": colors[i],
                "shape": rng.choice(SHAPES_BY_SIZE[size]),
            }
        )
    return specs


def _place_case(
    spec: dict[str, object],
    h: int,
    w: int,
    structure_forbidden: frozenset[tuple[int, int]],
    output_occ: frozenset[tuple[int, int]],
    output_padding: frozenset[tuple[int, int]],
    bumpers: frozenset[tuple[int, int]],
    trajectory_forbidden: frozenset[tuple[int, int]],
    next_step_forbidden: frozenset[tuple[int, int]],
    rng: random.Random,
) -> dict[str, object] | None:
    direction = str(spec["direction"])
    stop = str(spec["stop"])
    shape = spec["shape"]
    if not isinstance(shape, frozenset):
        return None
    dr, dc = DIRS[direction]

    for _ in range(CASE_ATTEMPTS):
        if stop == "edge":
            final_origin = _random_edge_origin(shape, direction, h, w, rng)
        else:
            final_origin = _random_inbounds_origin(shape, h, w, rng)
        final_cells = _translate(shape, int(final_origin[0]), int(final_origin[1]))
        if final_cells & output_padding:
            continue

        steps = _feasible_steps(final_origin, dr, dc, h, w)
        rng.shuffle(steps)
        for dist in steps:
            interior_r = int(final_origin[0] - dist * dr)
            interior_c = int(final_origin[1] - dist * dc)
            container_top = (interior_r - 1, interior_c - 1)
            container_region = _box_cells(container_top[0], container_top[1], 5, 5)
            if container_region & structure_forbidden:
                continue

            initial_cells = _translate(shape, interior_r, interior_c)
            if stop == "edge":
                stopped, reason, trail, next_cells = _simulate_motion(initial_cells, dr, dc, bumpers, h, w)
                if reason != "edge" or stopped != final_cells:
                    continue
                return {
                    "direction": direction,
                    "color": int(spec["color"]),
                    "shape": shape,
                    "container_top": container_top,
                    "initial_cells": initial_cells,
                    "final_cells": final_cells,
                    "bumper": frozenset(),
                    "trail": trail,
                    "next_cells": next_cells,
                    "container_region": container_region,
                }

            bumper_options = list(_possible_bumpers(final_cells, direction, h, w))
            rng.shuffle(bumper_options)
            for bumper in bumper_options:
                if bumper & output_padding:
                    continue
                if bumper & structure_forbidden:
                    continue
                if bumper & trajectory_forbidden:
                    continue
                if bumper & next_step_forbidden:
                    continue
                if bumper & _expand_chebyshev(container_region, h, w):
                    continue

                blocked = frozenset(set(bumpers) | set(bumper))
                stopped, reason, trail, next_cells = _simulate_motion(initial_cells, dr, dc, blocked, h, w)
                if reason != "bumper" or stopped != final_cells:
                    continue
                return {
                    "direction": direction,
                    "color": int(spec["color"]),
                    "shape": shape,
                    "container_top": container_top,
                    "initial_cells": initial_cells,
                    "final_cells": final_cells,
                    "bumper": bumper,
                    "trail": trail,
                    "next_cells": next_cells,
                    "container_region": container_region,
                }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    specs = _sample_case_specs(rng)
    n = len(specs)
    min_dim = 14 + 3 * n
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    order = sorted(range(n), key=lambda idx: str(specs[idx]["stop"]) != "bumper")
    placed: list[dict[str, object]] = []

    structure_forbidden = frozenset()
    output_occ = frozenset()
    output_padding = frozenset()
    bumpers = frozenset()
    trajectory_forbidden = frozenset()
    next_step_forbidden = frozenset()

    for idx in order:
        record = _place_case(
            specs[idx],
            h,
            w,
            structure_forbidden,
            output_occ,
            output_padding,
            bumpers,
            trajectory_forbidden,
            next_step_forbidden,
            rng,
        )
        if record is None:
            return None

        placed.append(record)
        container_region = record["container_region"]
        bumper = record["bumper"]
        final_cells = record["final_cells"]
        trail = record["trail"]
        next_cells = record["next_cells"]

        if not isinstance(container_region, frozenset):
            return None
        if not isinstance(bumper, frozenset):
            return None
        if not isinstance(final_cells, frozenset):
            return None
        if not isinstance(trail, frozenset):
            return None
        if not isinstance(next_cells, frozenset):
            return None

        structure_forbidden = frozenset(
            set(structure_forbidden)
            | set(_expand_chebyshev(container_region, h, w))
            | set(_expand_chebyshev(bumper, h, w))
        )
        output_occ = frozenset(set(output_occ) | set(final_cells) | set(bumper))
        output_padding = _expand_chebyshev(output_occ, h, w)
        bumpers = frozenset(set(bumpers) | set(bumper))
        trajectory_forbidden = frozenset(set(trajectory_forbidden) | set(trail))
        next_step_forbidden = frozenset(set(next_step_forbidden) | set(next_cells))

    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)

    for r, c in bumpers:
        inp[r, c] = GREY
        out[r, c] = GREY

    for record in placed:
        top, left = record["container_top"]  # type: ignore[misc]
        direction = str(record["direction"])
        color = int(record["color"])
        initial_cells = record["initial_cells"]
        final_cells = record["final_cells"]
        if not isinstance(initial_cells, frozenset) or not isinstance(final_cells, frozenset):
            return None

        for dr, dc in CONTAINER_PATTERNS[direction]:
            inp[int(top + dr), int(left + dc)] = GREY
        for r, c in initial_cells:
            inp[r, c] = color
        for r, c in final_cells:
            out[r, c] = color

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
