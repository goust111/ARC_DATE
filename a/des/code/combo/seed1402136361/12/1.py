import random

import numpy as np


MAROON = 9
ALL_COLORS = tuple(range(10))

GRID_MIN_SIZE = 14
GRID_MAX_SIZE = 26
OUTER_HEIGHT_RANGE = (5, 10)
OUTER_WIDTH_RANGE = (5, 10)
MAX_CONTAINERS_SMALL = 3
MAX_CONTAINERS_LARGE = 4
PLACEMENT_MARGIN = 2
MAX_MARKERS_PER_AXIS = 3
MAX_CONTAINER_TRIES = 200
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                if cr > 0 and mask[cr - 1, cc] and not seen[cr - 1, cc]:
                    seen[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not seen[cr + 1, cc]:
                    seen[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not seen[cr, cc - 1]:
                    seen[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not seen[cr, cc + 1]:
                    seen[cr, cc + 1] = True
                    stack.append((cr, cc + 1))
            comps.append(np.array(cells, dtype=int))
    return comps


def _boundary_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_containers(grid: np.ndarray, bg: int) -> list[dict]:
    containers = []
    for coords in _components(grid != bg):
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "container must enclose an interior"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _boundary_mask(height, width)
        assert np.all(patch[border] != bg), "container border must be fully occupied"
        assert np.all(patch[~border] == bg), "container interior must start empty"

        corner_colors = {
            int(patch[0, 0]),
            int(patch[0, -1]),
            int(patch[-1, 0]),
            int(patch[-1, -1]),
        }
        assert len(corner_colors) == 1, "corners must reveal a unique wall color"
        wall_color = next(iter(corner_colors))

        vertical: dict[int, int] = {}
        horizontal: dict[int, int] = {}

        for c in range(c0 + 1, c1):
            top_color = int(grid[r0, c])
            if top_color != wall_color:
                previous = vertical.setdefault(c, top_color)
                assert previous == top_color, "conflicting vertical markers in one column"
            bottom_color = int(grid[r1, c])
            if bottom_color != wall_color:
                previous = vertical.setdefault(c, bottom_color)
                assert previous == bottom_color, "conflicting vertical markers in one column"

        for r in range(r0 + 1, r1):
            left_color = int(grid[r, c0])
            if left_color != wall_color:
                previous = horizontal.setdefault(r, left_color)
                assert previous == left_color, "conflicting horizontal markers in one row"
            right_color = int(grid[r, c1])
            if right_color != wall_color:
                previous = horizontal.setdefault(r, right_color)
                assert previous == right_color, "conflicting horizontal markers in one row"

        assert vertical or horizontal, "each container must have at least one marker"
        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "vertical": vertical,
                "horizontal": horizontal,
            }
        )

    assert containers, "expected at least one container"
    containers.sort(key=lambda item: item["bbox"])
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw colored cross-lines inside hollow rectangular containers based on border markers.

    Each non-background connected component is a one-pixel-thick rectangular wall with
    optional colored marker pixels on its inner-facing border. Top/bottom markers launch
    vertical lines; left/right markers launch horizontal lines. Lines fill only interior
    background cells, and vertical/horizontal crossings become maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    out = g.copy()
    for container in _extract_containers(g, bg):
        r0, c0, r1, c1 = container["bbox"]

        for c, color in container["vertical"].items():
            for r in range(r0 + 1, r1):
                assert int(g[r, c]) == bg, "vertical line must travel through interior background"
                out[r, c] = color

        for r, color in container["horizontal"].items():
            for c in range(c0 + 1, c1):
                assert int(g[r, c]) == bg, "horizontal line must travel through interior background"
                out[r, c] = MAROON if out[r, c] != bg else color

    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _sample_box(
    rng: random.Random,
    height: int,
    width: int,
    placed: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    max_outer_h = min(OUTER_HEIGHT_RANGE[1], height - 2)
    max_outer_w = min(OUTER_WIDTH_RANGE[1], width - 2)
    min_outer_h = OUTER_HEIGHT_RANGE[0]
    min_outer_w = OUTER_WIDTH_RANGE[0]
    if max_outer_h < min_outer_h or max_outer_w < min_outer_w:
        return None

    for _ in range(MAX_CONTAINER_TRIES):
        outer_h = rng.randint(min_outer_h, max_outer_h)
        outer_w = rng.randint(min_outer_w, max_outer_w)
        r0 = rng.randint(0, height - outer_h)
        c0 = rng.randint(0, width - outer_w)
        box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if all(not _boxes_too_close(box, other, PLACEMENT_MARGIN) for other in placed):
            return box
    return None


def _max_axis_markers(axis_len: int) -> int:
    return min(MAX_MARKERS_PER_AXIS, max(1, axis_len - 1))


def _paint_container(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    wall_color: int,
) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = wall_color
    grid[r1, c0 : c1 + 1] = wall_color
    grid[r0 : r1 + 1, c0] = wall_color
    grid[r0 : r1 + 1, c1] = wall_color


def _place_markers(
    rng: random.Random,
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    wall_color: int,
    marker_palette: list[int],
    mode: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    vertical_count = 0
    horizontal_count = 0

    if mode in {"vertical", "both"}:
        columns = list(range(c0 + 1, c1))
        rng.shuffle(columns)
        columns = columns[: rng.randint(1, _max_axis_markers(inner_w))]
        for c in columns:
            color = rng.choice(marker_palette)
            side_r = r0 if rng.random() < 0.5 else r1
            grid[side_r, c] = color
        vertical_count = len(columns)

    if mode in {"horizontal", "both"}:
        rows = list(range(r0 + 1, r1))
        rng.shuffle(rows)
        rows = rows[: rng.randint(1, _max_axis_markers(inner_h))]
        for r in rows:
            color = rng.choice(marker_palette)
            side_c = c0 if rng.random() < 0.5 else c1
            grid[r, side_c] = color
        horizontal_count = len(rows)

    assert vertical_count or horizontal_count
    return vertical_count, horizontal_count


def _dominant_background(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if int(values[order[0]]) != bg:
        return False
    if len(order) == 1:
        return True
    return int(counts[order[0]]) > int(counts[order[1]])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = rng.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    bg = rng.choice([color for color in ALL_COLORS if color != MAROON])
    wall = rng.choice([color for color in ALL_COLORS if color not in {bg, MAROON}])
    marker_palette = [color for color in ALL_COLORS if color not in {bg, wall, MAROON}]

    max_containers = MAX_CONTAINERS_LARGE if height * width >= 380 else MAX_CONTAINERS_SMALL
    container_count = rng.randint(1, max_containers)
    want_crossings = rng.random() < 0.5

    if want_crossings:
        crossed_indices = set(rng.sample(range(container_count), rng.randint(1, container_count)))
    else:
        crossed_indices = set()

    grid = np.full((height, width), bg, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    modes: list[str] = []

    for idx in range(container_count):
        box = _sample_box(rng, height, width, boxes)
        if box is None:
            return None
        boxes.append(box)
        if idx in crossed_indices:
            modes.append("both")
        else:
            modes.append(rng.choice(["vertical", "horizontal"]))

    rng.shuffle(boxes)
    rng.shuffle(modes)

    total_vertical = 0
    total_horizontal = 0
    for box, mode in zip(boxes, modes):
        _paint_container(grid, box, wall)
        v_count, h_count = _place_markers(rng, grid, box, wall, marker_palette, mode)
        total_vertical += v_count
        total_horizontal += h_count

    if want_crossings and (total_vertical == 0 or total_horizontal == 0):
        return None

    if not _dominant_background(grid, bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None

    has_maroon = bool(np.any(out == MAROON))
    if has_maroon != want_crossings:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
