from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERPENDICULAR = {
    "up": ((0, -1), (0, 1)),
    "down": ((0, -1), (0, 1)),
    "left": ((-1, 0), (1, 0)),
    "right": ((-1, 0), (1, 0)),
}
GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 160


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for dr, dc in DIR_TO_VEC.values():
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(comp)
    return components


def _direction_axis(direction: tuple[int, int]) -> str:
    dr, dc = direction
    assert (dr == 0) != (dc == 0)
    return "h" if dr == 0 else "v"


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _step(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return (cell[0] + direction[0] * steps, cell[1] + direction[1] * steps)


def _line_from(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    cur = start
    while _in_bounds(shape, cur):
        cells.append(cur)
        cur = _step(cur, direction)
    return cells


def _classify_component(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict[str, object]:
    colors = [int(grid[r, c]) for r, c in cells]
    unique_colors = sorted(set(colors))

    if len(cells) == 1:
        assert unique_colors == [GREY], "singleton foreground cells must be grey targets"
        return {
            "type": "target",
            "cell": cells[0],
            "cells": tuple(cells),
            "color": GREY,
        }

    if len(cells) == 2:
        assert len(unique_colors) == 2, "relay stations must be bichromatic"
        (r0, c0), (r1, c1) = cells
        assert abs(r0 - r1) + abs(c0 - c1) == 1, "relay cells must form a domino"
        axis = "h" if r0 == r1 else "v"
        ordered = sorted(cells)
        return {
            "type": "relay",
            "cells": tuple(ordered),
            "axis": axis,
            "colors": tuple(int(grid[r, c]) for r, c in ordered),
        }

    if len(cells) == 3:
        assert len(unique_colors) == 1, "arrows must be monochrome"
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        assert max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1, "arrow must be a 2x2 L triomino"
        pts = set(cells)
        degrees = {}
        for r, c in cells:
            deg = 0
            for dr, dc in DIR_TO_VEC.values():
                if (r + dr, c + dc) in pts:
                    deg += 1
            degrees[(r, c)] = deg
        elbows = [cell for cell, deg in degrees.items() if deg == 2]
        arms = [cell for cell, deg in degrees.items() if deg == 1]
        assert len(elbows) == 1 and len(arms) == 2, "arrow must have one elbow and two arm tips"
        elbow = elbows[0]
        candidate_dirs = []
        for arm in arms:
            direction = (elbow[0] - arm[0], elbow[1] - arm[1])
            candidate_dirs.append(direction)
        return {
            "type": "arrow",
            "cells": tuple(sorted(cells)),
            "color": unique_colors[0],
            "elbow": elbow,
            "candidate_dirs": tuple(candidate_dirs),
        }

    raise AssertionError("unexpected non-black component shape")


def _parse_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, object]]]:
    components = _components4(grid != BLACK)
    objects = [_classify_component(grid, comp) for comp in components]
    arrows = [obj for obj in objects if obj["type"] == "arrow"]
    cell_to_object: dict[tuple[int, int], dict[str, object]] = {}
    for obj in objects:
        for cell in obj["cells"]:
            cell_to_object[cell] = obj
    arrows.sort(key=lambda obj: (obj["elbow"], obj["color"], obj["cells"]))
    return arrows, cell_to_object


def _scan_first_hit(
    grid: np.ndarray,
    cell_to_object: dict[tuple[int, int], dict[str, object]],
    arrow: dict[str, object],
    direction: tuple[int, int],
) -> dict[str, object]:
    start = _step(arrow["elbow"], direction)
    cur = start
    distance = 1
    while _in_bounds(grid.shape, cur):
        if int(grid[cur]) != BLACK:
            obj = cell_to_object[cur]
            axis_ok = obj["type"] != "relay" or obj["axis"] == _direction_axis(direction)
            return {
                "kind": str(obj["type"]),
                "distance": distance,
                "axis_ok": axis_ok,
                "object": obj,
            }
        cur = _step(cur, direction)
        distance += 1
    return {"kind": "edge", "distance": distance, "axis_ok": False, "object": None}


def _candidate_score(event: dict[str, object]) -> tuple[int, int]:
    kind = str(event["kind"])
    if kind == "target":
        score = 4
    elif kind == "relay" and bool(event["axis_ok"]):
        score = 4
    elif kind == "arrow":
        score = 3
    else:
        score = 0
    return (score, -int(event["distance"]))


def _choose_direction(
    grid: np.ndarray,
    cell_to_object: dict[tuple[int, int], dict[str, object]],
    arrow: dict[str, object],
) -> tuple[int, int]:
    evaluations = []
    for direction in arrow["candidate_dirs"]:
        event = _scan_first_hit(grid, cell_to_object, arrow, direction)
        evaluations.append((direction, event, _candidate_score(event)))

    evaluations.sort(key=lambda item: (item[2][0], item[2][1], item[0]), reverse=True)
    best_direction, _best_event, best_score = evaluations[0]
    assert best_score[0] > 0, "every arrow must have one context-supported firing direction"
    if len(evaluations) > 1 and evaluations[1][2] == best_score:
        raise AssertionError("arrow direction must be uniquely implied by the surrounding objects")
    return best_direction


def _relay_output_color(relay: dict[str, object], current_color: int) -> int | None:
    a, b = relay["colors"]
    if current_color == a:
        return int(b)
    if current_color == b:
        return int(a)
    return None


def _relay_exit_cell(relay: dict[str, object], direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return max(relay["cells"], key=lambda cell: cell[0] * dr + cell[1] * dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each arrow's color signal through aligned two-color relay stations until a grey target is recolored.

    Input structure:
    - The background is black.
    - Foreground objects are separated by black gaps and come in exactly three forms:
      monochrome 3-pixel L-shaped arrows, bichromatic 1x2 or 2x1 relay stations,
      and grey singleton target pixels.
    - Each relay station lies on the same row or column as the ray that can use it.

    Rule:
    - Remove all arrow pixels from the output, but keep relay stations and untouched grey targets.
    - For each arrow, decode its elbow and the two candidate cardinal firing directions implied by the L shape.
      The valid direction is the unique side whose straight sightline first reaches a context-compatible object.
    - Launch a straight signal from the arrow elbow in that direction, carrying the arrow color.
      When the signal reaches an aligned relay whose two colors include the current signal color, replace the
      signal color with the relay's other color and continue past the relay in the same direction.
    - If the first reached relay does not contain the current signal color, or if the signal hits another arrow
      or the boundary, stop with no further effect. If the signal reaches a grey target, recolor that target
      with the current signal color and stop.

    Key insight:
    - The L shape only tells you an elbow and two possible launch axes; the surrounding relay/target layout
      disambiguates which side is the real arrow direction, after which the puzzle is a straight-line color
      transformation chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    arrows, cell_to_object = _parse_scene(g)
    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BLACK

    for arrow in arrows:
        direction = _choose_direction(g, cell_to_object, arrow)
        current_color = int(arrow["color"])
        current_pos = tuple(arrow["elbow"])
        while True:
            cur = _step(current_pos, direction)
            while _in_bounds(g.shape, cur) and int(g[cur]) == BLACK:
                cur = _step(cur, direction)
            if not _in_bounds(g.shape, cur):
                break

            obj = cell_to_object[cur]
            kind = str(obj["type"])
            if kind == "target":
                out[cur] = current_color
                break
            if kind == "arrow":
                break
            assert kind == "relay"
            if obj["axis"] != _direction_axis(direction):
                break
            new_color = _relay_output_color(obj, current_color)
            if new_color is None:
                break
            current_color = new_color
            current_pos = _relay_exit_cell(obj, direction)

    return out


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _can_place_cells(
    shape: tuple[int, int],
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
) -> bool:
    for cell in cells:
        if not _in_bounds(shape, cell):
            return False
        if cell in occupied or cell in reserved_black:
            return False
    for cell in cells:
        for nbr in _neighbors4(cell):
            if nbr in occupied and nbr not in cells:
                return False
    return True


def _scatter_slack(slack: int, slots: int) -> list[int]:
    values = [0] * slots
    for _ in range(slack):
        values[random.randrange(slots)] += 1
    return values


def _choose_positions(line_len: int, sizes: list[int]) -> list[int] | None:
    need = sum(sizes) + len(sizes)
    if line_len < need:
        return None
    extras = _scatter_slack(line_len - need, len(sizes) + 1)
    starts: list[int] = []
    cursor = 1 + extras[0]
    for i, size in enumerate(sizes):
        starts.append(cursor)
        cursor += size
        if i + 1 < len(sizes):
            cursor += 1 + extras[i + 1]
    return starts


def _choose_positions_before_blocker(pre_len: int, sizes: list[int]) -> list[int] | None:
    need = sum(sizes) + len(sizes) + 1
    if pre_len < need:
        return None
    extras = _scatter_slack(pre_len - need, len(sizes) + 1)
    starts: list[int] = []
    cursor = 1 + extras[0]
    for i, size in enumerate(sizes):
        starts.append(cursor)
        cursor += size
        cursor += 1 + extras[i + 1]
    return starts


def _choose_new_color(current_color: int) -> int:
    choices = [color for color in PALETTE if color != current_color]
    return int(random.choice(choices))


def _build_relay_colors(start_color: int, matched_count: int, final_kind: str) -> tuple[list[tuple[int, int]], int]:
    current = int(start_color)
    relays: list[tuple[int, int]] = []
    for _ in range(matched_count):
        nxt = _choose_new_color(current)
        relays.append((current, nxt))
        current = nxt
    if final_kind == "mismatch":
        choices = [color for color in PALETTE if color != current]
        a, b = random.sample(choices, 2)
        relays.append((int(a), int(b)))
    return relays, current


def _make_arrow_cells(
    elbow: tuple[int, int],
    direction: tuple[int, int],
    side: tuple[int, int],
) -> set[tuple[int, int]]:
    protrusion = (-direction[0], -direction[1])
    return {elbow, _step(elbow, protrusion), _step(elbow, side)}


def _line_object_cells(line: list[tuple[int, int]], start: int, size: int) -> list[tuple[int, int]]:
    return line[start : start + size]


def _reserve_corridor(
    reserved_black: set[tuple[int, int]],
    line_cells: list[tuple[int, int]],
    occupied_segment: set[tuple[int, int]],
) -> None:
    for cell in line_cells:
        if cell not in occupied_segment:
            reserved_black.add(cell)


def _commit_relay(grid: np.ndarray, cells: list[tuple[int, int]], colors: tuple[int, int]) -> None:
    order = list(colors)
    if random.random() < 0.5:
        order.reverse()
    for cell, color in zip(cells, order, strict=True):
        grid[cell] = int(color)


def _place_independent_arrow(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    color: int,
    final_kind: str,
) -> dict[str, object] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        direction_name = random.choice(list(DIR_TO_VEC))
        direction = DIR_TO_VEC[direction_name]
        side = random.choice(PERPENDICULAR[direction_name])
        matched_count = random.randint(0, 2)
        sizes = [2] * matched_count + [1 if final_kind == "target" else 2]
        needed = sum(sizes) + len(sizes)

        for _ in range(PLACEMENT_ATTEMPTS):
            elbow = (random.randrange(h), random.randrange(w))
            arrow_cells = _make_arrow_cells(elbow, direction, side)
            if not _can_place_cells(grid.shape, arrow_cells, occupied, reserved_black):
                continue

            line = _line_from(grid.shape, _step(elbow, direction), direction)
            if len(line) < needed:
                continue
            starts = _choose_positions(len(line), sizes)
            if starts is None:
                continue

            relay_colors, final_color = _build_relay_colors(color, matched_count, final_kind)
            planned_cells: set[tuple[int, int]] = set()
            object_cells: list[list[tuple[int, int]]] = []
            for start, size in zip(starts, sizes, strict=True):
                cells = _line_object_cells(line, start, size)
                object_cells.append(cells)
                planned_cells.update(cells)

            alt_direction = (-side[0], -side[1])
            alt_line = _line_from(grid.shape, _step(elbow, alt_direction), alt_direction)
            if any(cell in occupied for cell in alt_line):
                continue
            if not _can_place_cells(grid.shape, planned_cells, occupied, reserved_black):
                continue

            grid_cells = set(arrow_cells) | planned_cells
            for cell in arrow_cells:
                grid[cell] = color
            for cells, relay in zip(object_cells[:matched_count], relay_colors[:matched_count], strict=True):
                _commit_relay(grid, cells, relay)
            final_cells = object_cells[-1]
            if final_kind == "target":
                grid[final_cells[0]] = GREY
            else:
                _commit_relay(grid, final_cells, relay_colors[-1])

            occupied.update(grid_cells)
            reserved_black.update(alt_line)
            corridor_end = starts[-1] + sizes[-1]
            _reserve_corridor(reserved_black, line[:corridor_end], planned_cells)
            return {
                "cells": tuple(sorted(arrow_cells)),
                "elbow": elbow,
                "direction": direction,
                "side": side,
                "color": color,
                "final_kind": final_kind,
                "final_color": final_color,
            }
    return None


def _place_blocking_arrow(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    color: int,
    blocker_arrow: dict[str, object],
) -> dict[str, object] | None:
    blocker_cells = set(blocker_arrow["cells"])
    rows = list({r for r, _ in blocker_cells})
    random.shuffle(rows)

    for row in rows:
        row_blockers = sorted(c for r, c in blocker_cells if r == row)
        modes = []
        if row_blockers:
            modes.append(("right", min(row_blockers)))
            modes.append(("left", max(row_blockers)))
        random.shuffle(modes)

        for direction_name, block_col in modes:
            direction = DIR_TO_VEC[direction_name]
            matched_count = random.randint(0, 2)
            relay_sizes = [2] * matched_count

            for side in random.sample(list(PERPENDICULAR[direction_name]), k=2):
                for _ in range(PLACEMENT_ATTEMPTS):
                    if direction_name == "right":
                        if block_col <= 0:
                            continue
                        elbow_col = random.randrange(0, max(1, block_col))
                        pre_len = block_col - elbow_col - 1
                    else:
                        if block_col >= grid.shape[1] - 1:
                            continue
                        elbow_col = random.randrange(block_col + 1, grid.shape[1])
                        pre_len = elbow_col - block_col - 1
                    elbow = (row, elbow_col)
                    arrow_cells = _make_arrow_cells(elbow, direction, side)
                    if not _can_place_cells(grid.shape, arrow_cells, occupied, reserved_black):
                        continue

                    starts = _choose_positions_before_blocker(pre_len, relay_sizes)
                    if starts is None:
                        continue

                    line = _line_from(grid.shape, _step(elbow, direction), direction)
                    if not line:
                        continue
                    if direction_name == "right":
                        if block_col <= elbow_col:
                            continue
                    else:
                        if block_col >= elbow_col:
                            continue

                    planned_relay_cells: list[list[tuple[int, int]]] = []
                    planned_cell_set: set[tuple[int, int]] = set()
                    for start, size in zip(starts, relay_sizes, strict=True):
                        cells = _line_object_cells(line, start, size)
                        planned_relay_cells.append(cells)
                        planned_cell_set.update(cells)
                    if not _can_place_cells(grid.shape, planned_cell_set, occupied, reserved_black):
                        continue

                    first_hit = None
                    for cell in line:
                        if cell in blocker_cells:
                            first_hit = cell
                            break
                    if first_hit is None:
                        continue
                    if first_hit[0] != row or first_hit[1] != block_col:
                        continue

                    alt_direction = (-side[0], -side[1])
                    alt_line = _line_from(grid.shape, _step(elbow, alt_direction), alt_direction)
                    if any(cell in occupied for cell in alt_line):
                        continue

                    relay_colors, _ = _build_relay_colors(color, matched_count, "block")
                    for cell in arrow_cells:
                        grid[cell] = color
                    for cells, relay in zip(planned_relay_cells, relay_colors, strict=True):
                        _commit_relay(grid, cells, relay)

                    occupied.update(arrow_cells)
                    occupied.update(planned_cell_set)
                    reserved_black.update(alt_line)
                    blocker_index = line.index(first_hit)
                    _reserve_corridor(reserved_black, line[:blocker_index], planned_cell_set)
                    return {
                        "cells": tuple(sorted(arrow_cells)),
                        "elbow": elbow,
                        "direction": direction,
                        "side": side,
                        "color": color,
                        "final_kind": "block",
                    }
    return None


def _place_unused_targets(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    count = random.randint(0, 2)
    for _ in range(count):
        for _ in range(PLACEMENT_ATTEMPTS):
            cell = (random.randrange(h), random.randrange(w))
            if not _can_place_cells(grid.shape, {cell}, occupied, reserved_black):
                continue
            grid[cell] = GREY
            occupied.add(cell)
            break


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 26)
    w = random.randint(16, 26)
    grid = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved_black: set[tuple[int, int]] = set()
    colors = random.sample(list(PALETTE), k=4)

    blocker = _place_independent_arrow(grid, occupied, reserved_black, colors[0], random.choice(("target", "mismatch")))
    if blocker is None:
        return None

    blocked = _place_blocking_arrow(grid, occupied, reserved_black, colors[1], blocker)
    if blocked is None:
        return None

    opposite_kind = "mismatch" if blocker["final_kind"] == "target" else "target"
    independent = _place_independent_arrow(grid, occupied, reserved_black, colors[2], opposite_kind)
    if independent is None:
        return None

    if random.random() < 0.55:
        extra_kind = random.choice(("target", "mismatch"))
        extra = _place_independent_arrow(grid, occupied, reserved_black, colors[3], extra_kind)
        if extra is None:
            return None

    _place_unused_targets(grid, occupied, reserved_black)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-signal puzzle")
