from __future__ import annotations

import random
import numpy as np


OBSTACLE_COLOR = 5
EMPTY = -1
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# The 3-cell 2x2 marker uses the missing corner as the direction cue.
# We read the four corners in clockwise order as up, right, down, left.
MISSING_TO_DIR = {
    (0, 0): (-1, 0),
    (0, 1): (0, 1),
    (1, 1): (1, 0),
    (1, 0): (0, -1),
}
DIR_TO_MISSING = {
    (-1, 0): (0, 0),
    (0, 1): (0, 1),
    (1, 0): (1, 1),
    (0, -1): (1, 0),
}


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _marker_direction(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    colors = sorted({int(grid[r, c]) for r, c in coords.tolist()})
    directions: list[tuple[int, int]] = []
    for color in colors:
        pts = [(int(r), int(c)) for r, c in coords.tolist() if int(grid[r, c]) == color]
        if len(pts) != 3:
            continue
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        marker = set(pts)
        if marker - corners:
            continue
        missing = corners - marker
        if len(missing) != 1:
            continue
        mr, mc = next(iter(missing))
        directions.append(MISSING_TO_DIR[(mr - r0, mc - c0)])

    assert len(directions) == 1, "each object must contain exactly one valid 3-cell L marker"
    return directions[0]


def _leading_edge(coords: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    lanes: dict[int, tuple[int, int]] = {}
    for r, c in coords.tolist():
        rr = int(r)
        cc = int(c)
        if dr != 0:
            key = cc
            if key not in lanes or (dr < 0 and rr < lanes[key][0]) or (dr > 0 and rr > lanes[key][0]):
                lanes[key] = (rr, cc)
        else:
            key = rr
            if key not in lanes or (dc < 0 and cc < lanes[key][1]) or (dc > 0 and cc > lanes[key][1]):
                lanes[key] = (rr, cc)
    return list(lanes.values())


def _steps_to_blocker(
    grid: np.ndarray,
    coords: np.ndarray,
    direction: tuple[int, int],
    bg: int,
) -> int:
    h, w = grid.shape
    dr, dc = direction
    front = _leading_edge(coords, direction)
    gaps: list[int] = []
    for r, c in front:
        gap = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != OBSTACLE_COLOR:
            gap += 1
            rr += dr
            cc += dc
        gaps.append(gap)

    assert gaps
    steps = min(gaps)
    for r, c in front:
        for delta in range(1, steps + 1):
            rr = r + dr * delta
            cc = c + dc * delta
            assert 0 <= rr < h and 0 <= cc < w
            assert int(grid[rr, cc]) == bg, "movement lanes must be background for the actual travel distance"
    return steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each foreground object along the direction encoded by its internal L marker.

    The marker is a 3-cell 2x2 L of a second color. The missing corner indicates the
    direction, and the object moves until its nearest front edge reaches a grey obstacle
    or the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    movable_mask = (g != bg) & (g != OBSTACLE_COLOR)
    components = _components(movable_mask)
    if not components:
        return g.copy()

    out = g.copy()
    placements: list[tuple[np.ndarray, np.ndarray]] = []
    for coords in components:
        values = g[coords[:, 0], coords[:, 1]].copy()
        direction = _marker_direction(g, coords)
        steps = _steps_to_blocker(g, coords, direction, bg)
        shifted = coords + np.array(direction, dtype=int) * steps
        placements.append((shifted, values))
        out[coords[:, 0], coords[:, 1]] = bg

    occupied = {tuple(cell) for cell in np.argwhere(g == OBSTACLE_COLOR).tolist()}
    for shifted, values in placements:
        for (r, c), value in zip(shifted.tolist(), values.tolist(), strict=False):
            rr, cc = int(r), int(c)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            assert (rr, cc) not in occupied, "moved objects must not overlap obstacles or each other"
            occupied.add((rr, cc))
            out[rr, cc] = int(value)

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _rotate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _normalize({(c, -r) for r, c in cells})


def _flip(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _normalize({(r, -c) for r, c in cells})


def _full_blocks(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    for r in range(max_r):
        for c in range(max_c):
            block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if block <= cells:
                blocks.append((r, c))
    return blocks


def _random_shape(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(100):
        area = rng.randint(7, 12)
        cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.append(nxt)
            rng.shuffle(frontier)
            placed = False
            for nr, nc in frontier:
                trial = _normalize(cells | {(nr, nc)})
                height = max(r for r, _c in trial) + 1
                width = max(c for _r, c in trial) + 1
                if height <= 5 and width <= 5:
                    cells.add((nr, nc))
                    placed = True
                    break
            if not placed:
                break

        if len(cells) != area:
            continue

        cells = _normalize(cells)
        for _ in range(rng.randrange(4)):
            cells = _rotate(cells)
        if rng.random() < 0.5:
            cells = _flip(cells)
        cells = _normalize(cells)
        if _full_blocks(cells):
            return cells
    raise RuntimeError("failed to build a valid object shape")


def _make_patch(
    rng: random.Random,
    direction: tuple[int, int],
    body_color: int,
    marker_color: int,
) -> np.ndarray:
    cells = _random_shape(rng)
    block_r, block_c = rng.choice(_full_blocks(cells))
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    patch = np.full((h, w), EMPTY, dtype=int)
    for r, c in cells:
        patch[r, c] = body_color
    miss_r, miss_c = DIR_TO_MISSING[direction]
    for dr in (0, 1):
        for dc in (0, 1):
            if (dr, dc) != (miss_r, miss_c):
                patch[block_r + dr, block_c + dc] = marker_color
    return patch


def _patch_positions(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(patch != EMPTY):
        out.append((top + int(r), left + int(c), int(patch[r, c])))
    return out


def _coords_only(cells: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    return {(r, c) for r, c, _value in cells}


def _leading_local_cells(patch: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    coords = np.argwhere(patch != EMPTY).astype(int)
    return _leading_edge(coords, direction)


def _too_close(coords: set[tuple[int, int]], occupied: set[tuple[int, int]], margin: int = 1) -> bool:
    for r, c in coords:
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if (rr, cc) in occupied:
                    return True
    return False


def _balanced_blocker_types(rng: random.Random, count: int) -> list[str]:
    types = ["boundary"] * (count // 2) + ["obstacle"] * (count - count // 2)
    if "boundary" not in types:
        types[0] = "boundary"
    if "obstacle" not in types:
        types[-1] = "obstacle"
    rng.shuffle(types)
    return types


def _sample_final_placement(
    rng: random.Random,
    patch: np.ndarray,
    direction: tuple[int, int],
    blocker_type: str,
    height: int,
    width: int,
    obstacle_cells: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
    input_occ: set[tuple[int, int]],
    clear_reserved: set[tuple[int, int]],
) -> tuple[int, int, tuple[int, int] | None] | None:
    ph, pw = patch.shape
    dr, dc = direction

    if blocker_type == "boundary":
        if dr < 0:
            row_options = [0]
            col_options = range(0, width - pw + 1)
        elif dr > 0:
            row_options = [height - ph]
            col_options = range(0, width - pw + 1)
        elif dc < 0:
            row_options = range(0, height - ph + 1)
            col_options = [0]
        else:
            row_options = range(0, height - ph + 1)
            col_options = [width - pw]
    else:
        if dr < 0:
            if height - ph < 1:
                return None
            row_options = range(1, height - ph + 1)
            col_options = range(0, width - pw + 1)
        elif dr > 0:
            if height - ph - 1 < 0:
                return None
            row_options = range(0, height - ph)
            col_options = range(0, width - pw + 1)
        elif dc < 0:
            if width - pw < 1:
                return None
            row_options = range(0, height - ph + 1)
            col_options = range(1, width - pw + 1)
        else:
            if width - pw - 1 < 0:
                return None
            row_options = range(0, height - ph + 1)
            col_options = range(0, width - pw)

    row_options = list(row_options)
    col_options = list(col_options)
    if not row_options or not col_options:
        return None

    candidates = [(r, c) for r in row_options for c in col_options]
    rng.shuffle(candidates)
    for top, left in candidates:
        final_cells = _patch_positions(patch, top, left)
        final_coords = _coords_only(final_cells)
        if _too_close(final_coords, output_occ, margin=1):
            continue
        if _too_close(final_coords, obstacle_cells, margin=1):
            continue

        obstacle: tuple[int, int] | None = None
        if blocker_type == "obstacle":
            front = rng.sample(_leading_local_cells(patch, direction), k=len(_leading_local_cells(patch, direction)))
            for rr, cc in front:
                candidate = (top + rr + dr, left + cc + dc)
                if candidate in final_coords or candidate in obstacle_cells or candidate in output_occ:
                    continue
                if _too_close({candidate}, output_occ, margin=1):
                    continue
                if _too_close({candidate}, input_occ, margin=1):
                    continue
                if candidate in clear_reserved:
                    continue
                obstacle = candidate
                break
            if obstacle is None:
                continue
        return top, left, obstacle

    return None


def _valid_shifts(
    patch: np.ndarray,
    final_top: int,
    final_left: int,
    direction: tuple[int, int],
    height: int,
    width: int,
    obstacle_cells: set[tuple[int, int]],
    input_occ: set[tuple[int, int]],
    clear_reserved: set[tuple[int, int]],
) -> list[int]:
    dr, dc = direction
    valid: list[int] = []
    for shift in range(1, 7):
        input_top = final_top - dr * shift
        input_left = final_left - dc * shift
        input_cells = _patch_positions(patch, input_top, input_left)
        input_coords = _coords_only(input_cells)
        if any(r < 0 or r >= height or c < 0 or c >= width for r, c in input_coords):
            continue
        if _too_close(input_coords, input_occ, margin=1):
            continue
        if _too_close(input_coords, obstacle_cells, margin=1):
            continue
        if input_coords & clear_reserved:
            continue

        clear = True
        for step in range(1, shift + 1):
            swept_coords = _coords_only(
                _patch_positions(
                    patch,
                    input_top + dr * step,
                    input_left + dc * step,
                )
            )
            if swept_coords & input_occ:
                clear = False
                break
            if swept_coords & obstacle_cells:
                clear = False
                break
        if clear:
            valid.append(shift)
    return valid


def _paint(base: np.ndarray, cells: list[tuple[int, int, int]]) -> None:
    for r, c, value in cells:
        base[r, c] = value


def _try_add_extra_obstacles(
    rng: random.Random,
    inp: np.ndarray,
    out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result_in = inp.copy()
    result_out = out.copy()
    free_cells = [(int(r), int(c)) for r, c in np.argwhere(result_in == _bg_color(result_in))]
    rng.shuffle(free_cells)
    extra_goal = rng.randint(0, 2)
    added = 0
    for r, c in free_cells:
        if added >= extra_goal:
            break
        trial_in = result_in.copy()
        trial_out = result_out.copy()
        trial_in[r, c] = OBSTACLE_COLOR
        trial_out[r, c] = OBSTACLE_COLOR
        try:
            solved = main(trial_in)
        except AssertionError:
            continue
        if np.array_equal(solved, trial_out):
            result_in = trial_in
            result_out = trial_out
            added += 1
    return result_in, result_out


def _generate_case(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(10, 18)
    width = rng.randint(10, 18)
    bg = rng.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    object_count = rng.randint(2, 4)

    palette = [color for color in range(10) if color not in {bg, OBSTACLE_COLOR}]
    rng.shuffle(palette)
    if len(palette) < 2 * object_count:
        return None
    marker_colors = palette[:object_count]
    body_colors = palette[object_count : 2 * object_count]
    directions = rng.sample(DIRS4, k=object_count)
    blocker_types = _balanced_blocker_types(rng, object_count)

    obstacle_cells: set[tuple[int, int]] = set()
    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()
    clear_reserved: set[tuple[int, int]] = set()
    input_objects: list[list[tuple[int, int, int]]] = []
    output_objects: list[list[tuple[int, int, int]]] = []

    order = list(range(object_count))
    rng.shuffle(order)
    for idx in order:
        patch = _make_patch(rng, directions[idx], body_colors[idx], marker_colors[idx])
        placed = False
        for _ in range(150):
            placement = _sample_final_placement(
                rng,
                patch,
                directions[idx],
                blocker_types[idx],
                height,
                width,
                obstacle_cells,
                output_occ,
                input_occ,
                clear_reserved,
            )
            if placement is None:
                continue
            final_top, final_left, dedicated_obstacle = placement
            future_obstacles = set(obstacle_cells)
            if dedicated_obstacle is not None:
                future_obstacles.add(dedicated_obstacle)

            shifts = _valid_shifts(
                patch,
                final_top,
                final_left,
                directions[idx],
                height,
                width,
                future_obstacles,
                input_occ,
                clear_reserved,
            )
            if not shifts:
                continue

            shift = rng.choice(shifts)
            input_top = final_top - directions[idx][0] * shift
            input_left = final_left - directions[idx][1] * shift
            input_cells = _patch_positions(patch, input_top, input_left)
            output_cells = _patch_positions(patch, final_top, final_left)
            input_coords = _coords_only(input_cells)
            output_coords = _coords_only(output_cells)
            reserved_now: set[tuple[int, int]] = set()
            for step in range(1, shift + 1):
                reserved_now |= _coords_only(
                    _patch_positions(
                        patch,
                        input_top + directions[idx][0] * step,
                        input_left + directions[idx][1] * step,
                    )
                )

            if _too_close(output_coords, output_occ, margin=1):
                continue
            if _too_close(input_coords, input_occ, margin=1):
                continue
            if _too_close(input_coords, future_obstacles, margin=1):
                continue
            if input_coords & clear_reserved:
                continue

            input_objects.append(input_cells)
            output_objects.append(output_cells)
            input_occ |= input_coords
            output_occ |= output_coords
            obstacle_cells = future_obstacles
            clear_reserved |= reserved_now
            placed = True
            break
        if not placed:
            return None

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    for r, c in obstacle_cells:
        inp[r, c] = OBSTACLE_COLOR
        out[r, c] = OBSTACLE_COLOR
    for obj in input_objects:
        _paint(inp, obj)
    for obj in output_objects:
        _paint(out, obj)

    if _bg_color(inp) != bg:
        return None

    inp, out = _try_add_extra_obstacles(rng, inp, out)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None

    movable = _components((inp != bg) & (inp != OBSTACLE_COLOR))
    if len(movable) != object_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
