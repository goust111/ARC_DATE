import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = values[counts == max_count]
    assert len(winners) == 1, "background color must be uniquely most common"
    return int(winners[0])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(cells, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_solid_rectangle(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _direction_from_singleton(
    singleton: tuple[int, int], bbox: tuple[int, int, int, int]
) -> str:
    sr, sc = singleton
    r0, c0, r1, c1 = bbox

    if c0 <= sc <= c1:
        if sr < r0:
            return "up"
        if sr > r1:
            return "down"
    if r0 <= sr <= r1:
        if sc < c0:
            return "left"
        if sc > c1:
            return "right"

    raise AssertionError("singleton must be cardinally aligned with its rectangle")


def _direction_delta(direction: str, distance: int) -> tuple[int, int]:
    if direction == "up":
        return -distance, 0
    if direction == "down":
        return distance, 0
    if direction == "left":
        return 0, -distance
    if direction == "right":
        return 0, distance
    raise AssertionError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each solid rectangle toward its same-color singleton marker by its own short side length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _most_common_color(g)
    out = np.full(g.shape, bg, dtype=int)
    occupied = np.zeros(g.shape, dtype=bool)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        components = _connected_components(g == color)
        assert len(components) == 2, "each color must have one rectangle and one singleton"

        singletons = [comp for comp in components if comp.shape[0] == 1]
        rectangles = [comp for comp in components if comp.shape[0] > 1]
        assert len(singletons) == 1 and len(rectangles) == 1

        singleton = tuple(map(int, singletons[0][0]))
        rect = rectangles[0]
        r0, c0, r1, c1 = _bbox(rect)
        assert _is_solid_rectangle(rect, (r0, c0, r1, c1))

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        step = min(height, width)
        direction = _direction_from_singleton(singleton, (r0, c0, r1, c1))
        dr, dc = _direction_delta(direction, step)

        nr0, nc0 = r0 + dr, c0 + dc
        nr1, nc1 = r1 + dr, c1 + dc
        assert 0 <= nr0 <= nr1 < g.shape[0]
        assert 0 <= nc0 <= nc1 < g.shape[1]
        assert not occupied[nr0 : nr1 + 1, nc0 : nc1 + 1].any(), "translated rectangles must stay disjoint"

        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = color
        occupied[nr0 : nr1 + 1, nc0 : nc1 + 1] = True

    return out


BG_COLOR = 5
PALETTE = tuple(color for color in range(10) if color != BG_COLOR)
DIRECTIONS = ("up", "down", "left", "right")
RELATIONS = ("less", "equal", "greater")

GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 24
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 5
SHORT_MIN = 1
SHORT_MAX = 5
LONG_EXTRA_MAX = 4
MARKER_EXTRA_MAX = 5

OUTER_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _rectangle_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                out.add((rr, cc))
    return out


def _sample_directions(n_objects: int) -> list[str]:
    if n_objects >= 4:
        dirs = list(DIRECTIONS)
        while len(dirs) < n_objects:
            dirs.append(random.choice(DIRECTIONS))
        random.shuffle(dirs)
        return dirs
    return random.sample(list(DIRECTIONS), n_objects)


def _sample_relations(n_objects: int) -> list[str]:
    rels = list(RELATIONS)
    while len(rels) < n_objects:
        rels.append(random.choice(RELATIONS))
    random.shuffle(rels)
    return rels[:n_objects]


def _sample_dims(relation: str) -> tuple[int, int]:
    if relation == "less":
        short = random.randint(3, SHORT_MAX)
    elif relation == "equal":
        short = random.randint(2, SHORT_MAX)
    else:
        short = random.randint(SHORT_MIN, SHORT_MAX)

    mode = random.choice(("square", "horizontal", "vertical"))
    if short == 1 and mode == "square":
        mode = random.choice(("horizontal", "vertical"))

    if mode == "square":
        return short, short

    long_side = random.randint(max(short + 1, 2), short + LONG_EXTRA_MAX)
    if mode == "horizontal":
        return short, long_side
    return long_side, short


def _sample_marker_distance(short_side: int, relation: str) -> int:
    if relation == "less":
        return random.randint(2, short_side - 1)
    if relation == "equal":
        return short_side
    return random.randint(max(2, short_side + 1), short_side + MARKER_EXTRA_MAX)


def _placement_limits(
    h: int, w: int, rect_h: int, rect_w: int, direction: str, step: int, marker_dist: int
) -> tuple[int, int, int, int] | None:
    if direction == "up":
        r0_min = max(step, marker_dist)
        r0_max = h - rect_h
        c0_min = 0
        c0_max = w - rect_w
    elif direction == "down":
        r0_min = 0
        r0_max = h - rect_h - max(step, marker_dist)
        c0_min = 0
        c0_max = w - rect_w
    elif direction == "left":
        r0_min = 0
        r0_max = h - rect_h
        c0_min = max(step, marker_dist)
        c0_max = w - rect_w
    else:
        r0_min = 0
        r0_max = h - rect_h
        c0_min = 0
        c0_max = w - rect_w - max(step, marker_dist)

    if r0_min > r0_max or c0_min > c0_max:
        return None
    return r0_min, r0_max, c0_min, c0_max


def _build_marker(
    r0: int, c0: int, rect_h: int, rect_w: int, direction: str, marker_dist: int
) -> tuple[int, int]:
    if direction == "up":
        return r0 - marker_dist, random.randint(c0, c0 + rect_w - 1)
    if direction == "down":
        return r0 + rect_h - 1 + marker_dist, random.randint(c0, c0 + rect_w - 1)
    if direction == "left":
        return random.randint(r0, r0 + rect_h - 1), c0 - marker_dist
    return random.randint(r0, r0 + rect_h - 1), c0 + rect_w - 1 + marker_dist


def _expected_from_specs(
    shape: tuple[int, int], specs: list[dict[str, int | str]], bg: int
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        rect_h = int(spec["h"])
        rect_w = int(spec["w"])
        step = int(spec["step"])
        direction = str(spec["direction"])
        dr, dc = _direction_delta(direction, step)
        out[r0 + dr : r0 + dr + rect_h, c0 + dc : c0 + dc + rect_w] = int(spec["color"])
    return out


def _build_input(
    shape: tuple[int, int], specs: list[dict[str, int | str]], bg: int
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        rect_h = int(spec["h"])
        rect_w = int(spec["w"])
        color = int(spec["color"])
        marker_r = int(spec["marker_r"])
        marker_c = int(spec["marker_c"])
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        grid[marker_r, marker_c] = color
    return grid


def _try_place_specs(
    shape: tuple[int, int], proto_specs: list[dict[str, int | str]]
) -> list[dict[str, int | str]] | None:
    h, w = shape
    specs = sorted(proto_specs, key=lambda spec: int(spec["h"]) * int(spec["w"]), reverse=True)

    input_cells_all: set[tuple[int, int]] = set()
    output_cells_all: set[tuple[int, int]] = set()
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()
    placed: list[dict[str, int | str]] = []

    for proto in specs:
        rect_h = int(proto["h"])
        rect_w = int(proto["w"])
        step = int(proto["step"])
        marker_dist = int(proto["marker_dist"])
        direction = str(proto["direction"])

        limits = _placement_limits(h, w, rect_h, rect_w, direction, step, marker_dist)
        if limits is None:
            return None
        r0_min, r0_max, c0_min, c0_max = limits

        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(r0_min, r0_max)
            c0 = random.randint(c0_min, c0_max)
            marker_r, marker_c = _build_marker(r0, c0, rect_h, rect_w, direction, marker_dist)
            src_cells = _rectangle_cells(r0, c0, rect_h, rect_w)
            input_cells = set(src_cells)
            input_cells.add((marker_r, marker_c))

            dr, dc = _direction_delta(direction, step)
            dst_cells = _rectangle_cells(r0 + dr, c0 + dc, rect_h, rect_w)

            if input_cells & input_forbidden:
                continue
            if dst_cells & output_forbidden:
                continue
            if input_cells & output_cells_all:
                continue
            if dst_cells & input_cells_all:
                continue

            placed_spec = dict(proto)
            placed_spec["r0"] = r0
            placed_spec["c0"] = c0
            placed_spec["marker_r"] = marker_r
            placed_spec["marker_c"] = marker_c
            placed.append(placed_spec)

            input_cells_all |= input_cells
            output_cells_all |= dst_cells
            input_forbidden |= _halo(input_cells, h, w)
            output_forbidden |= _halo(dst_cells, h, w)
            success = True
            break

        if not success:
            return None

    return placed


def _background_is_unique(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_by_color = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = count_by_color.get(bg, 0)
    return all(bg_count > count for color, count in count_by_color.items() if color != bg)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    colors = random.sample(PALETTE, n_objects)
    directions = _sample_directions(n_objects)
    relations = _sample_relations(n_objects)

    proto_specs: list[dict[str, int | str]] = []
    for color, direction, relation in zip(colors, directions, relations):
        rect_h, rect_w = _sample_dims(relation)
        step = min(rect_h, rect_w)
        marker_dist = _sample_marker_distance(step, relation)
        proto_specs.append(
            {
                "color": color,
                "direction": direction,
                "h": rect_h,
                "w": rect_w,
                "step": step,
                "marker_dist": marker_dist,
            }
        )

    placed = _try_place_specs((h, w), proto_specs)
    if placed is None:
        return None

    grid = _build_input((h, w), placed, BG_COLOR)
    expected = _expected_from_specs((h, w), placed, BG_COLOR)

    if not _background_is_unique(grid, BG_COLOR):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(OUTER_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {OUTER_ATTEMPTS} attempts")
