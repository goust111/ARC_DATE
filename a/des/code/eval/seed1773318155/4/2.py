from __future__ import annotations

import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PATH_COLOR = 5
SOURCE_COLORS = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _step(cell: tuple[int, int], direction: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return cell[0] + dr, cell[1] + dc


def _turn_left(direction: int) -> int:
    return (direction - 1) % 4


def _turn_right(direction: int) -> int:
    return (direction + 1) % 4


def _direction_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for direction, (drr, dcc) in enumerate(DIRS):
        if (dr, dc) == (drr, dcc):
            return direction
    raise AssertionError("cells must be 4-neighbors")


def _support_neighbors(mask: np.ndarray, r: int, c: int) -> list[int]:
    neighbors = []
    for direction, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
            neighbors.append(direction)
    return neighbors


def _choose_next_direction(mask: np.ndarray, prev: tuple[int, int], curr: tuple[int, int]) -> int | None:
    travel = _direction_between(prev, curr)
    back = (travel + 2) % 4
    forward = travel
    left = _turn_left(travel)
    right = _turn_right(travel)

    available = set(_support_neighbors(mask, curr[0], curr[1]))
    candidates = available - {back}
    if not candidates:
        return None
    if forward in candidates and len(available) >= 3:
        return forward
    if forward not in candidates and left in candidates and right in candidates:
        return left
    if len(candidates) == 1:
        return next(iter(candidates))
    raise AssertionError("ambiguous path topology")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored signal through a grey path network and paint the terminal endpoint.

    Input structure:
    - The background is the most common color.
    - One non-background color forms the thin path network.
    - Colored source pixels sit on path dead ends.

    Rule:
    - Each source advances along the 4-connected path.
    - At junctions with a forward continuation, the signal keeps going straight.
    - At T-junctions where forward is blocked but both side exits exist, the signal turns left.
    - At ordinary bends, the signal follows the only exit.
    - When the signal reaches its terminal dead end, that endpoint is recolored to the source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    non_bg = [int(color) for color in np.unique(g) if int(color) != bg]
    assert non_bg, "expected a path network"

    counts = {color: int(np.sum(g == color)) for color in non_bg}
    path_color = max(non_bg, key=lambda color: counts[color])
    support = g != bg

    sources: list[tuple[int, int]] = []
    for r, c in np.argwhere((g != bg) & (g != path_color)):
        rr = int(r)
        cc = int(c)
        if len(_support_neighbors(support, rr, cc)) == 1:
            sources.append((rr, cc))
    assert sources, "expected at least one source"

    out = g.copy()
    step_budget = int(np.sum(support)) * 4 + 1
    for source in sources:
        source_color = int(g[source])
        first_dirs = _support_neighbors(support, source[0], source[1])
        assert len(first_dirs) == 1, "source must be attached to exactly one path cell"
        prev = source
        curr = _step(source, first_dirs[0])
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        for _ in range(step_budget):
            state = (prev, curr)
            if state in seen:
                raise AssertionError("signal entered a cycle")
            seen.add(state)

            next_direction = _choose_next_direction(support, prev, curr)
            if next_direction is None:
                out[curr] = source_color
                break
            prev, curr = curr, _step(curr, next_direction)
        else:
            raise AssertionError("signal propagation did not terminate")

    return out


class _BuildError(RuntimeError):
    pass


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    return [_step(cell, direction) for direction in range(4)]


def _can_attach(
    occupied: set[tuple[int, int]],
    cell: tuple[int, int],
    allowed_neighbors: set[tuple[int, int]],
) -> bool:
    if cell in occupied:
        return False
    touching = {neighbor for neighbor in _neighbors4(cell) if neighbor in occupied}
    return touching == allowed_neighbors


def _add_chain(
    occupied: set[tuple[int, int]],
    path: set[tuple[int, int]],
    start: tuple[int, int],
    direction: int,
    length: int,
) -> list[tuple[int, int]]:
    assert length >= 1
    chain = []
    prev = start
    for _ in range(length):
        cell = _step(prev, direction)
        if not _can_attach(occupied, cell, {prev}):
            raise _BuildError("chain collision")
        occupied.add(cell)
        path.add(cell)
        chain.append(cell)
        prev = cell
    return chain


def _rotate_cell(cell: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = cell
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _normalize_cells(cells: set[tuple[int, int]], source: tuple[int, int]) -> tuple[set[tuple[int, int]], tuple[int, int], int, int]:
    all_cells = set(cells)
    all_cells.add(source)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shifted_cells = {(r - min_r, c - min_c) for r, c in cells}
    shifted_source = (source[0] - min_r, source[1] - min_c)
    max_r = max(r for r, _ in shifted_cells | {shifted_source})
    max_c = max(c for _, c in shifted_cells | {shifted_source})
    return shifted_cells, shifted_source, max_r + 1, max_c + 1


def _optional_tail(
    occupied: set[tuple[int, int]],
    path: set[tuple[int, int]],
    curr: tuple[int, int],
    direction: int,
) -> tuple[tuple[int, int], int]:
    if random.random() >= 0.5:
        return curr, direction
    new_direction = random.choice([_turn_left(direction), _turn_right(direction)])
    chain = _add_chain(occupied, path, curr, new_direction, random.randint(1, 2))
    curr = chain[-1]
    direction = new_direction
    if random.random() < 0.7:
        curr = _add_chain(occupied, path, curr, direction, random.randint(1, 2))[-1]
    return curr, direction


def _build_canonical_component(kind: str) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    source = (0, 0)
    occupied = {source}
    path: set[tuple[int, int]] = set()

    first = (0, 1)
    if not _can_attach(occupied, first, {source}):
        raise _BuildError("invalid start")
    occupied.add(first)
    path.add(first)
    curr = first
    direction = 1

    stem = random.randint(1, 3)
    curr = _add_chain(occupied, path, curr, direction, stem)[-1]

    if kind == "L":
        direction = random.choice([_turn_left(direction), _turn_right(direction)])
        curr = _add_chain(occupied, path, curr, direction, random.randint(1, 3))[-1]
        curr = _add_chain(occupied, path, curr, direction, random.randint(1, 3))[-1]
    elif kind == "T":
        _add_chain(occupied, path, curr, _turn_right(direction), random.randint(1, 2))
        left_direction = _turn_left(direction)
        curr = _add_chain(occupied, path, curr, left_direction, random.randint(1, 3))[-1]
        direction = left_direction
        curr = _add_chain(occupied, path, curr, direction, random.randint(1, 3))[-1]
        curr, direction = _optional_tail(occupied, path, curr, direction)
    elif kind == "X":
        side_mode = random.choice(["left", "right", "both"])
        if side_mode in {"left", "both"}:
            _add_chain(occupied, path, curr, _turn_left(direction), random.randint(1, 2))
        if side_mode in {"right", "both"}:
            _add_chain(occupied, path, curr, _turn_right(direction), random.randint(1, 2))
        curr = _add_chain(occupied, path, curr, direction, random.randint(1, 4))[-1]
        curr, direction = _optional_tail(occupied, path, curr, direction)
    else:
        raise AssertionError("unknown component kind")

    return path, source


def _make_component(kind: str, source_color: int) -> dict:
    for _ in range(100):
        try:
            path, source = _build_canonical_component(kind)
        except _BuildError:
            continue
        turns = random.randrange(4)
        rotated_path = {_rotate_cell(cell, turns) for cell in path}
        rotated_source = _rotate_cell(source, turns)
        norm_path, norm_source, height, width = _normalize_cells(rotated_path, rotated_source)
        return {
            "path": norm_path,
            "source": norm_source,
            "height": height,
            "width": width,
            "source_color": int(source_color),
        }
    raise RuntimeError(f"failed to build {kind} component")


def _paint_component(grid: np.ndarray, component: dict, top: int, left: int, bg: int) -> None:
    h = int(component["height"])
    w = int(component["width"])
    r0 = max(0, top - 1)
    r1 = min(grid.shape[0], top + h + 1)
    c0 = max(0, left - 1)
    c1 = min(grid.shape[1], left + w + 1)
    if np.any(grid[r0:r1, c0:c1] != bg):
        raise _BuildError("placement collision")

    for r, c in component["path"]:
        grid[top + r, left + c] = PATH_COLOR
    sr, sc = component["source"]
    grid[top + sr, left + sc] = int(component["source_color"])


def _place_components(grid: np.ndarray, components: list[dict], bg: int) -> bool:
    ordered = sorted(components, key=lambda comp: int(comp["height"]) * int(comp["width"]), reverse=True)
    for component in ordered:
        h = int(component["height"])
        w = int(component["width"])
        placed = False
        for _ in range(200):
            top = random.randint(0, grid.shape[0] - h)
            left = random.randint(0, grid.shape[1] - w)
            try:
                _paint_component(grid, component, top, left, bg)
            except _BuildError:
                continue
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    if int(np.sum(grid == bg)) <= grid.size // 2:
        return False
    try:
        out = main(grid)
    except Exception:
        return False

    path_mask = grid == PATH_COLOR
    source_mask = (grid != bg) & (~path_mask)
    source_count = int(np.sum(source_mask))
    changed = np.argwhere(grid != out)
    if len(changed) != source_count:
        return False
    for r, c in changed:
        if int(grid[r, c]) != PATH_COLOR:
            return False
        if int(out[r, c]) in {bg, PATH_COLOR}:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        bg = random.choice([color for color in SOURCE_COLORS if color != PATH_COLOR])
        kinds = ["L", "T", "X"]
        if random.random() < 0.5:
            kinds.append(random.choice(kinds))

        needed_colors = random.sample([color for color in SOURCE_COLORS if color != bg], len(kinds))
        components = [_make_component(kind, color) for kind, color in zip(kinds, needed_colors)]

        min_height = max(component["height"] for component in components) + 4
        min_width = max(component["width"] for component in components) + 4
        height = random.randint(max(20, min_height), max(28, min_height + 8))
        width = random.randint(max(20, min_width), max(28, min_width + 8))
        grid = np.full((height, width), bg, dtype=int)

        if not _place_components(grid, components, bg):
            continue
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
