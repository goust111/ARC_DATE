import numpy as np
from collections import deque


WALL_COLOR = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- generator config ---------------

RNG = np.random.default_rng()
MAX_RETRIES = 300
GROUP_ATTEMPTS = 250
GROUP_COUNT_OPTIONS = (1, 2, 3)
GRID_MIN = 12
GRID_MAX = 30
SHAPE_BOX_MIN = 1
SHAPE_BOX_MAX = 4
SHAPE_AREA_MAX = 6
WALL_PAD_MAX = 4


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    comp_id = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            queue = deque([(r, c)])
            labels[r, c] = comp_id
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(mask.shape, nr, nc):
                        continue
                    if not mask[nr, nc] or labels[nr, nc] != 0:
                        continue
                    labels[nr, nc] = comp_id
                    queue.append((nr, nc))
            components.append(np.array(cells, dtype=int))

    return labels, components


def _find_walls(grid: np.ndarray, bg: int) -> tuple[list[dict], set[tuple[int, int]]]:
    wall_components = _label_components(grid == WALL_COLOR)[1]
    walls = []
    markers: set[tuple[int, int]] = set()

    for comp in wall_components:
        if comp.shape[0] < 2:
            continue

        rows = comp[:, 0]
        cols = comp[:, 1]
        if np.all(rows == rows[0]):
            orientation = "h"
            axis = int(rows[0])
            lo = int(cols.min())
            hi = int(cols.max())
            if comp.shape[0] != hi - lo + 1:
                continue
            marker_candidates = []
            for pos in ((axis, lo - 1), (axis, hi + 1)):
                if _in_bounds(grid.shape, pos[0], pos[1]) and int(grid[pos]) not in {bg, WALL_COLOR}:
                    marker_candidates.append(pos)
            if len(marker_candidates) != 1:
                continue
            marker = marker_candidates[0]
            other_end = (axis, hi + 1) if marker == (axis, lo - 1) else (axis, lo - 1)
            if _in_bounds(grid.shape, other_end[0], other_end[1]) and int(grid[other_end]) != bg:
                continue
            full_lo = min(lo, marker[1])
            full_hi = max(hi, marker[1])
            if full_hi - full_lo + 1 < 3:
                continue
            coords = np.array([(axis, c) for c in range(full_lo, full_hi + 1)], dtype=int)
        elif np.all(cols == cols[0]):
            orientation = "v"
            axis = int(cols[0])
            lo = int(rows.min())
            hi = int(rows.max())
            if comp.shape[0] != hi - lo + 1:
                continue
            marker_candidates = []
            for pos in ((lo - 1, axis), (hi + 1, axis)):
                if _in_bounds(grid.shape, pos[0], pos[1]) and int(grid[pos]) not in {bg, WALL_COLOR}:
                    marker_candidates.append(pos)
            if len(marker_candidates) != 1:
                continue
            marker = marker_candidates[0]
            other_end = (hi + 1, axis) if marker == (lo - 1, axis) else (lo - 1, axis)
            if _in_bounds(grid.shape, other_end[0], other_end[1]) and int(grid[other_end]) != bg:
                continue
            full_lo = min(lo, marker[0])
            full_hi = max(hi, marker[0])
            if full_hi - full_lo + 1 < 3:
                continue
            coords = np.array([(r, axis) for r in range(full_lo, full_hi + 1)], dtype=int)
        else:
            continue

        walls.append(
            {
                "orientation": orientation,
                "axis": axis,
                "coords": coords,
                "marker": marker,
            }
        )
        markers.add(marker)

    walls.sort(key=lambda wall: (int(wall["coords"][:, 0].min()), int(wall["coords"][:, 1].min())))
    return walls, markers


def _adjacent_component_ids(
    labels: np.ndarray, coords: np.ndarray, dr: int, dc: int
) -> set[int]:
    ids = set()
    for r_raw, c_raw in coords:
        r = int(r_raw) + dr
        c = int(c_raw) + dc
        if _in_bounds(labels.shape, r, c) and labels[r, c] != 0:
            ids.add(int(labels[r, c]))
    return ids


def _source_component_id(wall: dict, labels: np.ndarray) -> int:
    coords = wall["coords"]
    if wall["orientation"] == "v":
        left_ids = _adjacent_component_ids(labels, coords, 0, -1)
        right_ids = _adjacent_component_ids(labels, coords, 0, 1)
        if len(left_ids) == 1 and not right_ids:
            return next(iter(left_ids))
        if len(right_ids) == 1 and not left_ids:
            return next(iter(right_ids))
    else:
        up_ids = _adjacent_component_ids(labels, coords, -1, 0)
        down_ids = _adjacent_component_ids(labels, coords, 1, 0)
        if len(up_ids) == 1 and not down_ids:
            return next(iter(up_ids))
        if len(down_ids) == 1 and not up_ids:
            return next(iter(down_ids))
    raise AssertionError("each wall must have exactly one adjacent source component on one side")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each source object across its maroon mirror wall and recolor the reflection with the endpoint marker.

    Input structure:
    - bg = most common color.
    - Maroon straight segments form one-pixel horizontal or vertical mirror walls, and one wall endpoint is replaced by a non-bg marker color.
    - Exactly one monochrome source object lies orthogonally adjacent to one side of each wall, while the mirrored side starts as background.

    Rule:
    - Detect every wall together with its non-maroon endpoint marker.
    - Find the source component adjacent to the occupied side of that wall.
    - Reflect the source pixels across the wall axis onto the opposite side and paint the reflection with the wall's marker color.
    - Keep the original source and wall unchanged, and drop reflected pixels that fall outside the grid.

    Key insight:
    - The wall orientation defines the mirror axis, adjacency reveals which component is the source, and the endpoint marker determines the reflection color instead of the source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _most_common_color(g)
    walls, marker_positions = _find_walls(g, bg)
    assert walls, "expected at least one mirror wall"

    marker_mask = np.zeros_like(g, dtype=bool)
    for r, c in marker_positions:
        marker_mask[r, c] = True

    source_mask = (g != bg) & (g != WALL_COLOR) & (~marker_mask)
    labels, components = _label_components(source_mask)

    out = g.copy()
    for wall in walls:
        comp_id = _source_component_id(wall, labels)
        source = components[comp_id - 1]
        marker_color = int(g[wall["marker"]])

        if wall["orientation"] == "v":
            axis = int(wall["axis"])
            for r_raw, c_raw in source:
                r = int(r_raw)
                c = int(c_raw)
                cc = 2 * axis - c
                if 0 <= cc < g.shape[1]:
                    out[r, cc] = marker_color
        else:
            axis = int(wall["axis"])
            for r_raw, c_raw in source:
                r = int(r_raw)
                c = int(c_raw)
                rr = 2 * axis - r
                if 0 <= rr < g.shape[0]:
                    out[rr, c] = marker_color

    return out


def _random_connected_shape(height: int, width: int, touch_mode: str) -> np.ndarray | None:
    max_area = min(height * width, SHAPE_AREA_MAX)
    area = int(RNG.integers(1, max_area + 1))
    if touch_mode == "col0":
        start = (int(RNG.integers(height)), 0)
    else:
        start = (0, int(RNG.integers(width)))

    cells = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(cells) < area:
        if not frontier:
            return None
        choices = tuple(frontier)
        pick = choices[int(RNG.integers(len(choices)))]
        frontier.remove(pick)
        cells.add(pick)
        add_frontier(pick)

    coords = np.array(sorted(cells), dtype=int)
    coords[:, 0] -= int(coords[:, 0].min())
    coords[:, 1] -= int(coords[:, 1].min())
    return coords


def _occupied_with_padding_clear(occupied: np.ndarray, coords: np.ndarray, pad: int = 1) -> bool:
    h, w = occupied.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if not _in_bounds((h, w), r, c):
            continue
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _mark_occupied(occupied: np.ndarray, coords: np.ndarray) -> None:
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if _in_bounds(occupied.shape, r, c):
            occupied[r, c] = True


def _build_group_spec(
    shape: tuple[int, int], source_color: int, marker_color: int
) -> dict | None:
    h, w = shape
    orientation = "v" if int(RNG.integers(2)) == 0 else "h"

    if orientation == "v":
        src_rel = _random_connected_shape(
            int(RNG.integers(SHAPE_BOX_MIN, SHAPE_BOX_MAX + 1)),
            int(RNG.integers(SHAPE_BOX_MIN, SHAPE_BOX_MAX + 1)),
            "col0",
        )
        if src_rel is None:
            return None
        src_h = int(src_rel[:, 0].max() + 1)
        src_w = int(src_rel[:, 1].max() + 1)
        wall_len_lo = max(3, src_h)
        wall_len_hi = min(h, src_h + WALL_PAD_MAX)
        if wall_len_lo > wall_len_hi:
            return None
        wall_len = int(RNG.integers(wall_len_lo, wall_len_hi + 1))
        side = "left" if int(RNG.integers(2)) == 0 else "right"
        use_clip = bool(int(RNG.integers(2))) and src_w >= 2

        if use_clip:
            if side == "left":
                axis_lo = max(src_w, w - src_w)
                axis_hi = w - 2
            else:
                axis_lo = 1
                axis_hi = min(src_w - 1, w - src_w - 1)
        else:
            axis_lo, axis_hi = src_w, w - src_w - 1
        if axis_lo > axis_hi:
            return None

        row0 = int(RNG.integers(0, h - wall_len + 1))
        axis = int(RNG.integers(axis_lo, axis_hi + 1))
        offset = int(RNG.integers(0, wall_len - src_h + 1))
        marker_at_top = int(RNG.integers(2)) == 0

        wall_coords = np.array([(row0 + i, axis) for i in range(wall_len)], dtype=int)
        marker = (row0, axis) if marker_at_top else (row0 + wall_len - 1, axis)

        source_coords = []
        reflect_coords = []
        for rr_raw, cc_raw in src_rel:
            rr = row0 + offset + int(rr_raw)
            cc = int(cc_raw)
            if side == "left":
                source_coords.append((rr, axis - 1 - cc))
                reflect_coords.append((rr, axis + 1 + cc))
            else:
                source_coords.append((rr, axis + 1 + cc))
                reflect_coords.append((rr, axis - 1 - cc))
    else:
        src_rel = _random_connected_shape(
            int(RNG.integers(SHAPE_BOX_MIN, SHAPE_BOX_MAX + 1)),
            int(RNG.integers(SHAPE_BOX_MIN, SHAPE_BOX_MAX + 1)),
            "row0",
        )
        if src_rel is None:
            return None
        src_h = int(src_rel[:, 0].max() + 1)
        src_w = int(src_rel[:, 1].max() + 1)
        wall_len_lo = max(3, src_w)
        wall_len_hi = min(w, src_w + WALL_PAD_MAX)
        if wall_len_lo > wall_len_hi:
            return None
        wall_len = int(RNG.integers(wall_len_lo, wall_len_hi + 1))
        side = "up" if int(RNG.integers(2)) == 0 else "down"
        use_clip = bool(int(RNG.integers(2))) and src_h >= 2

        if use_clip:
            if side == "up":
                axis_lo = max(src_h, h - src_h)
                axis_hi = h - 2
            else:
                axis_lo = 1
                axis_hi = min(src_h - 1, h - src_h - 1)
        else:
            axis_lo, axis_hi = src_h, h - src_h - 1
        if axis_lo > axis_hi:
            return None

        col0 = int(RNG.integers(0, w - wall_len + 1))
        axis = int(RNG.integers(axis_lo, axis_hi + 1))
        offset = int(RNG.integers(0, wall_len - src_w + 1))
        marker_at_left = int(RNG.integers(2)) == 0

        wall_coords = np.array([(axis, col0 + i) for i in range(wall_len)], dtype=int)
        marker = (axis, col0) if marker_at_left else (axis, col0 + wall_len - 1)

        source_coords = []
        reflect_coords = []
        for rr_raw, cc_raw in src_rel:
            rr = int(rr_raw)
            cc = col0 + offset + int(cc_raw)
            if side == "up":
                source_coords.append((axis - 1 - rr, cc))
                reflect_coords.append((axis + 1 + rr, cc))
            else:
                source_coords.append((axis + 1 + rr, cc))
                reflect_coords.append((axis - 1 - rr, cc))

    source_coords = np.array(source_coords, dtype=int)
    reflect_coords = np.array(reflect_coords, dtype=int)
    visible_reflection = np.array(
        [coord for coord in reflect_coords if _in_bounds(shape, int(coord[0]), int(coord[1]))],
        dtype=int,
    )
    if visible_reflection.size == 0:
        return None

    all_coords = np.vstack((wall_coords, source_coords, visible_reflection))
    return {
        "wall_coords": wall_coords,
        "marker": marker,
        "source_coords": source_coords,
        "reflect_coords": reflect_coords,
        "occupied_coords": all_coords,
        "source_color": source_color,
        "marker_color": marker_color,
    }


def _render_input(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        for r_raw, c_raw in spec["wall_coords"]:
            grid[int(r_raw), int(c_raw)] = WALL_COLOR
        mr, mc = spec["marker"]
        grid[mr, mc] = spec["marker_color"]
        for r_raw, c_raw in spec["source_coords"]:
            grid[int(r_raw), int(c_raw)] = spec["source_color"]
    return grid


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        color = spec["marker_color"]
        for r_raw, c_raw in spec["reflect_coords"]:
            r = int(r_raw)
            c = int(c_raw)
            if _in_bounds(grid.shape, r, c):
                out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = int(RNG.choice(GROUP_COUNT_OPTIONS))
    min_side = 10 + 4 * group_count
    height = int(RNG.integers(min_side, GRID_MAX + 1))
    width = int(RNG.integers(min_side, GRID_MAX + 1))
    bg_choices = [color for color in range(9)]
    bg = int(RNG.choice(bg_choices))

    palette = [color for color in range(10) if color not in {bg, WALL_COLOR}]
    RNG.shuffle(palette)
    if len(palette) < 2 * group_count:
        return None

    specs = []
    occupied = np.zeros((height, width), dtype=bool)
    for index in range(group_count):
        source_color = int(palette[2 * index])
        marker_color = int(palette[2 * index + 1])

        placed = False
        for _ in range(GROUP_ATTEMPTS):
            spec = _build_group_spec((height, width), source_color, marker_color)
            if spec is None:
                continue
            if not _occupied_with_padding_clear(occupied, spec["occupied_coords"], pad=1):
                continue
            _mark_occupied(occupied, spec["occupied_coords"])
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    grid = _render_input((height, width), bg, specs)
    expected = _render_expected(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if len(_find_walls(grid, bg)[0]) != group_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-wall input")
