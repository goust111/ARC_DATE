import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)

REFLECT_SLASH = {
    DIR_RIGHT: DIR_UP,
    DIR_UP: DIR_RIGHT,
    DIR_LEFT: DIR_DOWN,
    DIR_DOWN: DIR_LEFT,
}

REFLECT_BACKSLASH = {
    DIR_RIGHT: DIR_DOWN,
    DIR_DOWN: DIR_RIGHT,
    DIR_LEFT: DIR_UP,
    DIR_UP: DIR_LEFT,
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _on_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _is_interior(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 < r < h - 1 and 0 < c < w - 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BLACK:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if not _in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def _mirror_orientation(cells: list[tuple[int, int]]) -> str:
    assert len(cells) == 2
    rows = sorted(r for r, _ in cells)
    cols = sorted(c for _, c in cells)
    assert rows[1] == rows[0] + 1 and cols[1] == cols[0] + 1, "mirror must fit in a 2x2 box"
    box_r, box_c = rows[0], cols[0]
    cell_set = set(cells)
    if cell_set == {(box_r, box_c), (box_r + 1, box_c + 1)}:
        return "\\"
    if cell_set == {(box_r, box_c + 1), (box_r + 1, box_c)}:
        return "/"
    raise AssertionError("mirror cells must form a diagonal pair")


def _start_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    sides = []
    if r == 0:
        sides.append(DIR_DOWN)
    if r == h - 1:
        sides.append(DIR_UP)
    if c == 0:
        sides.append(DIR_RIGHT)
    if c == w - 1:
        sides.append(DIR_LEFT)
    assert len(sides) == 1, "start marker must lie on exactly one edge"
    return sides[0]


def _parse_scene(grid: np.ndarray):
    start = None
    mirrors: dict[tuple[int, int], tuple[str, int]] = {}
    for color, cells in _find_components(grid):
        if len(cells) == 1:
            r, c = cells[0]
            assert _on_boundary(grid.shape, r, c), "singleton must be the edge start marker"
            assert start is None, "expected a unique start marker"
            start = (r, c, color)
            continue
        assert len(cells) == 2, "every mirror must be a 2-pixel component"
        orient = _mirror_orientation(cells)
        for r, c in cells:
            assert _is_interior(grid.shape, r, c), "mirror cells must be interior"
            mirrors[(r, c)] = (orient, color)
    assert start is not None, "expected one edge start marker"
    return start, mirrors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the beam from the edge marker through colored diagonal mirrors and paint the resulting route.

    Input structure:
    - A black background contains one single-pixel start marker on exactly one grid edge.
    - The interior contains isolated 2-pixel diagonal mirror objects, each monochrome and shaped as "/" or "\\".
    - No other foreground objects are present.

    Rule:
    - Launch a beam from the start marker straight inward, with initial beam color equal to the start color.
    - Every non-mirror cell visited by the beam is repainted to the current beam color.
    - When the beam enters a mirror cell, a same-colored mirror is transparent; otherwise the beam reflects 90 degrees by the mirror orientation and the beam color becomes that mirror's color. Mirror pixels themselves stay unchanged.

    Key insight:
    - The state is not only the beam position and direction: the current carried color determines whether each mirror reflects or lets the beam pass straight through.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    start, mirrors = _parse_scene(g)

    out = g.copy()
    r, c, beam_color = start
    dr, dc = _start_direction(g.shape, r, c)
    first_cell = True
    seen_states = set()

    while True:
        state = (r, c, dr, dc, beam_color)
        assert state not in seen_states, "beam entered a cycle"
        seen_states.add(state)

        mirror = mirrors.get((r, c))
        if mirror is None:
            out[r, c] = beam_color
        else:
            orient, mirror_color = mirror
            if beam_color != mirror_color:
                if orient == "/":
                    dr, dc = REFLECT_SLASH[(dr, dc)]
                else:
                    dr, dc = REFLECT_BACKSLASH[(dr, dc)]
                beam_color = mirror_color

        if not first_cell and _on_boundary(g.shape, r, c):
            break
        first_cell = False

        nr, nc = r + dr, c + dc
        assert _in_bounds(g.shape, nr, nc), "beam must stop when it reaches the boundary"
        r, c = nr, nc

    return out


def _ray_from(cell: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    dr, dc = direction
    out = []
    r += dr
    c += dc
    while _in_bounds(shape, r, c):
        out.append((r, c))
        r += dr
        c += dc
    return out


def _companion_candidates(cell: tuple[int, int], orient: str) -> list[tuple[int, int]]:
    r, c = cell
    if orient == "/":
        return [(r - 1, c + 1), (r + 1, c - 1)]
    return [(r - 1, c - 1), (r + 1, c + 1)]


def _can_place_nonblack(
    cells: list[tuple[int, int]],
    shape: tuple[int, int],
    occupied_nonblack: set[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied_nonblack:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in occupied_nonblack and (nr, nc) not in cell_set:
                return False
    return True


def _choose_start(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _build_event_plan() -> list[str]:
    n_reflect = random.randint(1, 3)
    n_transparent = random.choices([0, 1, 2], weights=[3, 5, 2])[0]
    if n_transparent == 0 and random.random() < 0.6:
        n_transparent = 1
    events = ["reflect"] * n_reflect + ["transparent"] * n_transparent
    random.shuffle(events)
    return events


def _pick_reflect_color(current_color: int) -> int:
    choices = [color for color in COLORS if color != current_color]
    return random.choice(choices)


def _place_unused_mirrors(
    grid: np.ndarray,
    occupied_nonblack: set[tuple[int, int]],
    reserved_path: set[tuple[int, int]],
) -> None:
    h, w = grid.shape
    extra_count = random.choices([0, 1, 2], weights=[3, 4, 2])[0]
    for _ in range(extra_count):
        placed = False
        for _attempt in range(40):
            orient = random.choice(["/", "\\"])
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            cell = (r, c)
            companions = _companion_candidates(cell, orient)
            random.shuffle(companions)
            for companion in companions:
                if not _is_interior(grid.shape, companion[0], companion[1]):
                    continue
                pair = [cell, companion]
                if set(pair) & reserved_path:
                    continue
                if not _can_place_nonblack(pair, grid.shape, occupied_nonblack):
                    continue
                color = random.choice(COLORS)
                for rr, cc in pair:
                    grid[rr, cc] = color
                    occupied_nonblack.add((rr, cc))
                placed = True
                break
            if placed:
                break


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    shape = (h, w)

    start = _choose_start(shape)
    start_color = random.choice(COLORS)
    current = start
    direction = _start_direction(shape, *start)
    beam_color = start_color
    events = _build_event_plan()

    occupied_nonblack = {start}
    reserved_path = {start}
    mirror_specs: list[tuple[tuple[int, int], tuple[int, int], str, int]] = []
    visible_path_cells = {start}

    for index, event_type in enumerate(events):
        remaining_events = len(events) - index - 1
        ray = _ray_from(current, direction, shape)
        if len(ray) < 2:
            return None

        segment_prefix: list[tuple[int, int]] = []
        options = []
        for beam_cell in ray[:-1]:
            if beam_cell in reserved_path:
                break

            if not _is_interior(shape, beam_cell[0], beam_cell[1]):
                segment_prefix.append(beam_cell)
                continue

            if event_type == "reflect":
                orientations = ["/", "\\"]
            else:
                orientations = ["/", "\\"]

            for orient in orientations:
                next_direction = direction if event_type == "transparent" else (
                    REFLECT_SLASH[direction] if orient == "/" else REFLECT_BACKSLASH[direction]
                )
                next_ray = _ray_from(beam_cell, next_direction, shape)
                if not next_ray:
                    continue
                if remaining_events > 0 and len(next_ray) < 2:
                    continue

                companions = _companion_candidates(beam_cell, orient)
                random.shuffle(companions)
                for companion in companions:
                    if companion in reserved_path:
                        continue
                    if not _is_interior(shape, companion[0], companion[1]):
                        continue
                    pair = [beam_cell, companion]
                    if not _can_place_nonblack(pair, shape, occupied_nonblack):
                        continue
                    options.append((list(segment_prefix), beam_cell, companion, orient, next_direction))

            segment_prefix.append(beam_cell)

        if not options:
            return None

        segment, beam_cell, companion, orient, next_direction = random.choice(options)
        if any(cell in reserved_path for cell in segment):
            return None
        reserved_path.update(segment)
        visible_path_cells.update(segment)

        reserved_path.add(beam_cell)
        reserved_path.add(companion)
        occupied_nonblack.add(beam_cell)
        occupied_nonblack.add(companion)

        mirror_color = beam_color if event_type == "transparent" else _pick_reflect_color(beam_color)
        mirror_specs.append((beam_cell, companion, orient, mirror_color))
        current = beam_cell
        direction = next_direction
        if event_type == "reflect":
            beam_color = mirror_color

    tail = _ray_from(current, direction, shape)
    if not tail:
        return None
    if any(cell in reserved_path for cell in tail):
        return None
    reserved_path.update(tail)
    visible_path_cells.update(tail)

    grid = np.zeros(shape, dtype=int)
    sr, sc = start
    grid[sr, sc] = start_color
    for beam_cell, companion, _orient, mirror_color in mirror_specs:
        for r, c in (beam_cell, companion):
            grid[r, c] = mirror_color

    _place_unused_mirrors(grid, occupied_nonblack, reserved_path)

    if len(visible_path_cells) < 8:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
