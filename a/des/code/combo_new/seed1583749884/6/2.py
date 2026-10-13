from __future__ import annotations

import random

import numpy as np


BG_COLOR = 8
MARKER_COLOR = 4
BODY_COLORS = (1, 2, 3, 5, 6, 7, 9)
DIRECTIONS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN_SIDE = 12
GRID_MAX_SIDE = 24
OBJECT_COUNT_RANGE = (2, 4)
SHAPE_HEIGHT_RANGE = {
    "up": (4, 7),
    "down": (4, 7),
    "left": (3, 6),
    "right": (3, 6),
}
SHAPE_WIDTH_RANGE = {
    "up": (3, 6),
    "down": (3, 6),
    "left": (4, 7),
    "right": (4, 7),
}
TARGET_DENSITY_RANGE = (0.45, 0.78)
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _direction_from_marker(
    marker: tuple[int, int], bbox: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = marker[0] - center_r
    dc = marker[1] - center_c
    adr = abs(dr)
    adc = abs(dc)
    assert adr != adc and (adr > 0 or adc > 0), "marker must lean toward one axis"
    if adr > adc:
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each marked monochrome object toward the edge indicated by its embedded yellow pixel.

    Input structure:
    - The background is Azure.
    - Each foreground object is a 4-connected single-color body with exactly one embedded Yellow marker pixel.
    - The marker lies inside the object's bounding box and is visually biased toward one of the four cardinal directions.

    Rule:
    - For each foreground component, compare the Yellow marker position with the component bbox center.
    - If the marker is more above, below, left, or right of center than along the other axis, move the non-Yellow body pixels in that direction.
    - Translate those body pixels rigidly until the component bbox reaches the corresponding grid edge, clear the old body location, and keep the Yellow marker fixed in place.

    Key insight:
    - The Yellow pixel is not moved payload; it is a directional stamp whose offset from the object center determines which way the rest of the object gets launched.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])

    comps = _components4(g != bg)
    assert comps, "expected at least one marked object"

    out = np.full_like(g, bg)
    marker_cells: set[tuple[int, int]] = set()
    moved: list[tuple[list[tuple[int, int]], int, int, int]] = []

    for comp in comps:
        marker_coords = [(r, c) for r, c in comp if int(g[r, c]) == MARKER_COLOR]
        assert len(marker_coords) == 1, "each object must contain exactly one yellow marker"
        marker = marker_coords[0]
        body = [(r, c) for r, c in comp if int(g[r, c]) != MARKER_COLOR]
        assert body, "each object must have non-marker payload pixels"
        body_colors = {int(g[r, c]) for r, c in body}
        assert len(body_colors) == 1, "payload must be monochromatic"
        color = next(iter(body_colors))

        direction = _direction_from_marker(marker, _bbox(comp))
        r0, c0, r1, c1 = _bbox(comp)
        if direction == "up":
            shift = -r0
            dr, dc = shift, 0
        elif direction == "down":
            shift = h - 1 - r1
            dr, dc = shift, 0
        elif direction == "left":
            shift = -c0
            dr, dc = 0, shift
        else:
            assert direction == "right"
            shift = w - 1 - c1
            dr, dc = 0, shift

        marker_cells.add(marker)
        moved.append((body, color, dr, dc))

    for r, c in marker_cells:
        out[r, c] = MARKER_COLOR

    for body, color, dr, dc in moved:
        for r, c in body:
            rr = r + dr
            cc = c + dc
            assert 0 <= rr < h and 0 <= cc < w
            if int(out[rr, cc]) == bg:
                out[rr, cc] = int(color)
            else:
                assert int(out[rr, cc]) == MARKER_COLOR, "only stationary markers may block moved bodies"

    return out


def _rand_range(bounds: tuple[int, int]) -> int:
    return random.randint(int(bounds[0]), int(bounds[1]))


def _choose_directions(n_obj: int) -> list[str]:
    dirs = list(DIRECTIONS)
    random.shuffle(dirs)
    chosen = dirs[: min(n_obj, len(dirs))]
    while len(chosen) < n_obj:
        extra = list(DIRECTIONS)
        random.shuffle(extra)
        chosen.append(extra[0])
    random.shuffle(chosen)
    return chosen


def _marker_candidates(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0
    candidates: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            dr = r - center_r
            dc = c - center_c
            adr = abs(dr)
            adc = abs(dc)
            if adr == adc:
                continue
            if direction == "up" and dr < 0 and adr > adc:
                candidates.append((r, c))
            elif direction == "down" and dr > 0 and adr > adc:
                candidates.append((r, c))
            elif direction == "left" and dc < 0 and adc > adr:
                candidates.append((r, c))
            elif direction == "right" and dc > 0 and adc > adr:
                candidates.append((r, c))
    return candidates


def _frontier(cells: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for r, c in cells:
        for nbr in _neighbors4(r, c, height, width):
            if nbr not in cells:
                result.add(nbr)
    return list(result)


def _add_path(cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    cells.add((r, c))
    axes = [0, 1]
    random.shuffle(axes)
    for axis in axes:
        while (axis == 0 and r != er) or (axis == 1 and c != ec):
            if axis == 0:
                r += 1 if er > r else -1
            else:
                c += 1 if ec > c else -1
            cells.add((r, c))


def _generate_shape(direction: str) -> tuple[set[tuple[int, int]], tuple[int, int], int, int] | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = _rand_range(SHAPE_HEIGHT_RANGE[direction])
        width = _rand_range(SHAPE_WIDTH_RANGE[direction])
        marker_options = _marker_candidates(height, width, direction)
        if not marker_options:
            continue
        marker = random.choice(marker_options)

        cells = {
            marker,
            (marker[0] - 1, marker[1]),
            (marker[0] + 1, marker[1]),
            (marker[0], marker[1] - 1),
            (marker[0], marker[1] + 1),
        }

        edge_points = [
            (0, random.randint(0, width - 1)),
            (height - 1, random.randint(0, width - 1)),
            (random.randint(0, height - 1), 0),
            (random.randint(0, height - 1), width - 1),
        ]
        random.shuffle(edge_points)
        for point in edge_points:
            _add_path(cells, point, marker)

        total = height * width
        min_area = max(len(cells), int(np.ceil(total * TARGET_DENSITY_RANGE[0])))
        max_area = min(total - 1, int(np.floor(total * TARGET_DENSITY_RANGE[1])))
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)

        while len(cells) < target_area:
            options = _frontier(cells, height, width)
            if not options:
                break
            cells.add(random.choice(options))

        if len(cells) != target_area:
            continue
        if _bbox(cells) != (0, 0, height - 1, width - 1):
            continue
        if any(nbr not in cells for nbr in _neighbors4(marker[0], marker[1], height, width)):
            continue
        return cells, marker, height, width
    return None


def _expand_with_padding(
    cells: set[tuple[int, int]], height: int, width: int, padding: int
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - padding), min(height, r + padding + 1)):
            for cc in range(max(0, c - padding), min(width, c + padding + 1)):
                expanded.add((rr, cc))
    return expanded


def _candidate_positions(
    direction: str, shape_h: int, shape_w: int, grid_h: int, grid_w: int
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    if direction == "up":
        for r0 in range(1, grid_h - shape_h + 1):
            for c0 in range(0, grid_w - shape_w + 1):
                positions.append((r0, c0))
    elif direction == "down":
        for r0 in range(0, grid_h - shape_h):
            for c0 in range(0, grid_w - shape_w + 1):
                positions.append((r0, c0))
    elif direction == "left":
        for r0 in range(0, grid_h - shape_h + 1):
            for c0 in range(1, grid_w - shape_w + 1):
                positions.append((r0, c0))
    else:
        assert direction == "right"
        for r0 in range(0, grid_h - shape_h + 1):
            for c0 in range(0, grid_w - shape_w):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _shift_to_edge(
    direction: str, top: int, left: int, shape_h: int, shape_w: int, grid_h: int, grid_w: int
) -> tuple[int, int]:
    if direction == "up":
        return -top, 0
    if direction == "down":
        return grid_h - shape_h - top, 0
    if direction == "left":
        return 0, -left
    assert direction == "right"
    return 0, grid_w - shape_w - left


def _place_patch(grid: np.ndarray, cells: set[tuple[int, int]], marker: tuple[int, int], top: int, left: int, color: int) -> None:
    for r, c in cells:
        rr = top + r
        cc = left + c
        grid[rr, cc] = MARKER_COLOR if (r, c) == marker else int(color)


def _attempt_generate() -> np.ndarray | None:
    n_obj = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_MIN_SIDE, 8 + 3 * n_obj)
    grid_h = random.randint(min_side, GRID_MAX_SIDE)
    grid_w = random.randint(min_side, GRID_MAX_SIDE)

    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    expected = np.full_like(inp, BG_COLOR)

    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()
    output_used: set[tuple[int, int]] = set()

    colors = list(BODY_COLORS)
    if n_obj <= len(colors):
        body_colors = random.sample(colors, n_obj)
    else:
        body_colors = [random.choice(colors) for _ in range(n_obj)]

    directions = _choose_directions(n_obj)

    for direction, color in zip(directions, body_colors):
        shape_info = _generate_shape(direction)
        if shape_info is None:
            return None
        cells, marker_local, shape_h, shape_w = shape_info

        placed = False
        for top, left in _candidate_positions(direction, shape_h, shape_w, grid_h, grid_w):
            shift_r, shift_c = _shift_to_edge(direction, top, left, shape_h, shape_w, grid_h, grid_w)
            marker_abs = (top + marker_local[0], left + marker_local[1])

            input_cells = {(top + r, left + c) for r, c in cells}
            if input_cells & input_blocked:
                continue

            final_body = {
                (top + r + shift_r, left + c + shift_c)
                for r, c in cells
                if (r, c) != marker_local
            }
            if marker_abs in final_body:
                continue
            if final_body & output_used:
                continue
            final_fg = set(final_body)
            final_fg.add(marker_abs)
            if final_fg & output_blocked:
                continue

            _place_patch(inp, cells, marker_local, top, left, color)
            for r, c in final_body:
                expected[r, c] = int(color)
            expected[marker_abs] = MARKER_COLOR

            input_blocked |= _expand_with_padding(input_cells, grid_h, grid_w, PLACEMENT_PADDING)
            output_used |= final_fg
            output_blocked |= _expand_with_padding(final_fg, grid_h, grid_w, PLACEMENT_PADDING)
            placed = True
            break

        if not placed:
            return None

    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid launch-marker ARC input")
