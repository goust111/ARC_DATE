from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GRAY = 5

AREA_THRESHOLD = 4
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400

Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_cells(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _translate_with_clip(coords: np.ndarray, dr: int, dc: int, shape: tuple[int, int]) -> np.ndarray:
    shifted = coords + np.array([dr, dc], dtype=int)
    keep = (
        (0 <= shifted[:, 0])
        & (shifted[:, 0] < shape[0])
        & (0 <= shifted[:, 1])
        & (shifted[:, 1] < shape[1])
    )
    return shifted[keep]


def _drop_to_bottom(coords: np.ndarray, height: int) -> np.ndarray:
    dr = height - 1 - int(coords[:, 0].max())
    return coords + np.array([dr, 0], dtype=int)


def _extract_path_roles(grid: np.ndarray) -> tuple[Coord, Coord, Coord, Coord, set[Coord]]:
    gray_cells = set(map(tuple, np.argwhere(grid == GRAY)))
    assert gray_cells, "expected a gray path"

    gray_comps = _component_cells(grid == GRAY)
    assert len(gray_comps) == 1, "expected exactly one gray component"

    endpoints = []
    for r, c in gray_cells:
        deg = sum((nr, nc) in gray_cells for nr, nc in _neighbors4(r, c))
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "expected exactly two gray endpoints"

    red_cells = np.argwhere(grid == RED)
    yellow_cells = np.argwhere(grid == YELLOW)
    assert len(red_cells) == 1 and len(yellow_cells) == 1, "expected one red and one yellow marker"
    red = tuple(map(int, red_cells[0]))
    yellow = tuple(map(int, yellow_cells[0]))

    red_adj = [ep for ep in endpoints if abs(ep[0] - red[0]) + abs(ep[1] - red[1]) == 1]
    yellow_adj = [ep for ep in endpoints if abs(ep[0] - yellow[0]) + abs(ep[1] - yellow[1]) == 1]
    assert len(red_adj) == 1 and len(yellow_adj) == 1, "markers must attach to distinct gray endpoints"
    start = red_adj[0]
    end = yellow_adj[0]
    assert start != end

    return red, yellow, start, end, gray_cells


def _render_blue_output(
    blue_components: list[np.ndarray],
    shape: tuple[int, int],
    vector: tuple[int, int],
    *,
    area_mode: str = "count",
    endpoint_vector: tuple[int, int] | None = None,
    force_all: str | None = None,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)

    if endpoint_vector is not None:
        vec = endpoint_vector
    else:
        vec = vector

    big_cells: list[np.ndarray] = []
    small_cells: list[np.ndarray] = []
    for coords in blue_components:
        if force_all == "big":
            is_big = True
        elif force_all == "small":
            is_big = False
        elif area_mode == "bbox":
            r0, c0, r1, c1 = _bbox(coords)
            is_big = (r1 - r0 + 1) * (c1 - c0 + 1) > AREA_THRESHOLD
        else:
            is_big = len(coords) > AREA_THRESHOLD
        if is_big:
            big_cells.append(_translate_with_clip(coords, vec[0], vec[1], shape))
        else:
            small_cells.append(_drop_to_bottom(coords, shape[0]))

    for coords in small_cells:
        if len(coords):
            out[coords[:, 0], coords[:, 1]] = BLUE
    for coords in big_cells:
        if len(coords):
            out[coords[:, 0], coords[:, 1]] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the red-to-yellow displacement and apply it only to large blue objects while small blue objects fall to the bottom.

    Input structure:
    - The background is black.
    - One 4-connected gray path of width one has exactly two gray endpoints.
    - A single red pixel is 4-adjacent to one gray endpoint, a single yellow pixel is 4-adjacent to the other, and the remaining foreground objects are blue 4-connected components.

    Rule:
    - Compute the displacement vector from the red pixel to the yellow pixel.
    - Split blue components by pixel area: components with area greater than four are large, and components with area at most four are small.
    - Erase the original blue objects, path, and endpoint markers; translate each large object by the decoded vector with out-of-bounds pixels discarded, and drop each small object straight downward until its lowest pixel reaches the bottom row.
    - If a translated large object and a dropped small object target the same cell, keep the large object.

    Key insight:
    - The gray path identifies which colored singletons encode the vector, but only the large blue components obey that vector; the small blue components ignore it and instead use gravity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red, yellow, _start, _end, _gray_cells = _extract_path_roles(g)
    dy = int(yellow[0] - red[0])
    dx = int(yellow[1] - red[1])

    blue_components = _component_cells(g == BLUE)
    assert 4 <= len(blue_components) <= 6, "expected 4-6 blue objects"
    assert any(len(comp) <= AREA_THRESHOLD for comp in blue_components)
    assert any(len(comp) > AREA_THRESHOLD for comp in blue_components)

    return _render_blue_output(blue_components, g.shape, (dy, dx))


def _normalize(coords: list[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = sorted((r - min_r, c - min_c) for r, c in coords)
    return tuple(norm)


def _transform_shape(shape: tuple[Coord, ...]) -> tuple[Coord, ...]:
    cells = np.array(shape, dtype=int)
    if random.random() < 0.5:
        cells[:, 0] *= -1
    if random.random() < 0.5:
        cells[:, 1] *= -1
    k = random.randrange(4)
    for _ in range(k):
        cells = np.stack((-cells[:, 1], cells[:, 0]), axis=1)
    return _normalize([tuple(map(int, cell)) for cell in cells])


SHAPE_BANK: dict[int, list[tuple[Coord, ...]]] = {
    1: [((0, 0),)],
    2: [((0, 0), (0, 1))],
    3: [((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (1, 1))],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ],
    5: [
        ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ],
    6: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)),
    ],
    7: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (1, 2)),
    ],
    8: [
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
    ],
}


def _shape_coords(area: int, *, require_irregular: bool = False) -> np.ndarray:
    options = SHAPE_BANK[area]
    if require_irregular:
        options = [
            shape
            for shape in options
            if len({r for r, _ in shape}) * len({c for _, c in shape}) > area
        ]
    shape = _transform_shape(random.choice(options))
    return np.array(shape, dtype=int)


def _path_from_waypoints(waypoints: list[Coord]) -> list[Coord]:
    path = [(0, 0)]
    r, c = 0, 0
    for nr, nc in waypoints:
        if r != nr and c != nc:
            raise ValueError("waypoints must form axis-aligned segments")
        while r != nr:
            r += 1 if nr > r else -1
            path.append((r, c))
        while c != nc:
            c += 1 if nc > c else -1
            path.append((r, c))
    return path


def _path_candidates(dr: int, dc: int) -> list[list[Coord]]:
    candidates: list[list[Coord]] = []
    if dr == 0 or dc == 0:
        candidates.append(_path_from_waypoints([(dr, dc)]))
        if dr == 0:
            for off in (-1, 1):
                candidates.append(_path_from_waypoints([(off, 0), (off, dc), (dr, dc)]))
        else:
            for off in (-1, 1):
                candidates.append(_path_from_waypoints([(0, off), (dr, off), (dr, dc)]))
    else:
        candidates.append(_path_from_waypoints([(dr, 0), (dr, dc)]))
        candidates.append(_path_from_waypoints([(0, dc), (dr, dc)]))
        sr = 1 if dr > 0 else -1
        sc = 1 if dc > 0 else -1
        candidates.append(_path_from_waypoints([(-sr, 0), (-sr, dc), (dr, dc)]))
        candidates.append(_path_from_waypoints([(0, -sc), (dr, -sc), (dr, dc)]))

    valid = []
    for path in candidates:
        if len(set(path)) != len(path):
            continue
        if 4 <= len(path) <= 10:
            valid.append(path)
    return valid


def _reserve_with_neighbors(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    reserved = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc):
                reserved.add((nr, nc))
    return reserved


def _can_place(coords: np.ndarray, occupied: set[Coord], shape: tuple[int, int]) -> bool:
    for r, c in map(tuple, coords):
        r = int(r)
        c = int(c)
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(shape, nr, nc) and (nr, nc) in occupied:
                return False
    return True


def _coords_set(coords: np.ndarray) -> set[Coord]:
    return {tuple(map(int, cell)) for cell in coords}


def _valid_color_dirs(endpoint: Coord, path_set: set[Coord], shape: tuple[int, int]) -> list[Coord]:
    options = []
    for dr, dc in DIRS4:
        rr = endpoint[0] + dr
        cc = endpoint[1] + dc
        if not _in_bounds(shape, rr, cc) or (rr, cc) in path_set:
            continue
        bad = False
        for nr, nc in _neighbors4(rr, cc):
            if (nr, nc) != endpoint and (nr, nc) in path_set:
                bad = True
                break
        if not bad:
            options.append((dr, dc))
    return options


def _sample_path_scene(H: int, W: int) -> tuple[list[Coord], Coord, Coord, tuple[int, int]]:
    for _ in range(300):
        path_dr = random.choice([d for d in range(-5, 6) if d != 0] + [0] * 2)
        path_dc = random.choice([d for d in range(-5, 6) if d != 0] + [0] * 2)
        if path_dr == 0 and path_dc == 0:
            continue
        if abs(path_dr) + abs(path_dc) < 3 or abs(path_dr) + abs(path_dc) > 7:
            continue

        candidates = _path_candidates(path_dr, path_dc)
        if not candidates:
            continue
        offsets = random.choice(candidates)
        rel = np.array(offsets, dtype=int)
        min_r, min_c = rel.min(axis=0)
        max_r, max_c = rel.max(axis=0)
        starts = [
            (r0, c0)
            for r0 in range(-int(min_r), H - int(max_r))
            for c0 in range(-int(min_c), W - int(max_c))
        ]
        random.shuffle(starts)
        for r0, c0 in starts:
            path = [(r0 + dr, c0 + dc) for dr, dc in offsets]
            path_set = set(path)
            start = path[0]
            end = path[-1]
            red_dirs = _valid_color_dirs(start, path_set, (H, W))
            yellow_dirs = _valid_color_dirs(end, path_set, (H, W))
            pairs = []
            for rdir in red_dirs:
                for ydir in yellow_dirs:
                    if rdir == ydir:
                        continue
                    red = (start[0] + rdir[0], start[1] + rdir[1])
                    yellow = (end[0] + ydir[0], end[1] + ydir[1])
                    vector = (yellow[0] - red[0], yellow[1] - red[1])
                    if vector == (path_dr, path_dc):
                        continue
                    if abs(vector[0]) + abs(vector[1]) < 3 or abs(vector[0]) + abs(vector[1]) > 7:
                        continue
                    if max(abs(vector[0]), abs(vector[1])) > 5:
                        continue
                    pairs.append((red, yellow, vector))
            if not pairs:
                continue
            red, yellow, vector = random.choice(pairs)
            return path, red, yellow, vector
    raise RuntimeError("failed to sample a valid path scene")


def _random_positions(H: int, W: int, h: int, w: int) -> list[Coord]:
    pos = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(pos)
    return pos


def _place_small_objects(
    specs: list[np.ndarray],
    H: int,
    W: int,
    input_occ: set[Coord],
    output_occ: set[Coord],
) -> tuple[list[np.ndarray], set[Coord], set[Coord]] | None:
    placed: list[np.ndarray] = []
    occ_in = set(input_occ)
    occ_out = set(output_occ)
    for rel in specs:
        r1, c1 = rel.max(axis=0)
        success = False
        for r0, c0 in _random_positions(H, W, int(r1) + 1, int(c1) + 1):
            coords = rel + np.array([r0, c0], dtype=int)
            if int(coords[:, 0].max()) >= H - 1:
                continue
            final = _drop_to_bottom(coords, H)
            if not _can_place(coords, occ_in, (H, W)):
                continue
            if not _can_place(final, occ_out, (H, W)):
                continue
            placed.append(coords)
            occ_in |= _coords_set(coords)
            occ_out |= _coords_set(final)
            success = True
            break
        if not success:
            return None
    return placed, occ_in, occ_out


def _place_big_objects(
    specs: list[np.ndarray],
    H: int,
    W: int,
    vector: tuple[int, int],
    input_occ: set[Coord],
    output_occ: set[Coord],
    *,
    force_clip: bool,
) -> tuple[list[np.ndarray], bool] | None:
    placed: list[np.ndarray] = []
    occ_in = set(input_occ)
    occ_out = set(output_occ)
    clip_done = False
    for idx, rel in enumerate(specs):
        r1, c1 = rel.max(axis=0)
        positions = _random_positions(H, W, int(r1) + 1, int(c1) + 1)
        success = False
        need_clip = force_clip and not clip_done and (idx == len(specs) - 1 or random.random() < 0.6)
        for r0, c0 in positions:
            coords = rel + np.array([r0, c0], dtype=int)
            if not _can_place(coords, occ_in, (H, W)):
                continue
            final = _translate_with_clip(coords, vector[0], vector[1], (H, W))
            if len(final) == 0:
                continue
            clipped = len(final) < len(coords)
            if need_clip and not clipped:
                continue
            if not need_clip and clipped:
                continue
            if not _can_place(final, occ_out, (H, W)):
                continue
            placed.append(coords)
            occ_in |= _coords_set(coords)
            occ_out |= _coords_set(final)
            clip_done = clip_done or clipped
            success = True
            break
        if not success:
            return None
    return placed, clip_done


def _build_grid(H: int, W: int, path: list[Coord], red: Coord, yellow: Coord, objects: list[np.ndarray]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for r, c in path:
        grid[r, c] = GRAY
    grid[red] = RED
    grid[yellow] = YELLOW
    for coords in objects:
        grid[coords[:, 0], coords[:, 1]] = BLUE
    return grid


def _validate_scene(
    grid: np.ndarray,
    expected_vector: tuple[int, int],
    path_endpoint_vector: tuple[int, int],
    *,
    require_clip: bool,
) -> bool:
    try:
        red, yellow, start, end, gray_cells = _extract_path_roles(grid)
    except Exception:
        return False

    if (yellow[0] - red[0], yellow[1] - red[1]) != expected_vector:
        return False
    if (end[0] - start[0], end[1] - start[1]) != path_endpoint_vector:
        return False

    blue_components = _component_cells(grid == BLUE)
    if not (4 <= len(blue_components) <= 6):
        return False
    areas = sorted(len(comp) for comp in blue_components)
    if 4 not in areas or 5 not in areas:
        return False
    if sum(area <= AREA_THRESHOLD for area in areas) < 2:
        return False
    if sum(area > AREA_THRESHOLD for area in areas) < 2:
        return False

    correct = _render_blue_output(blue_components, grid.shape, expected_vector)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, correct):
        return False

    if np.array_equal(correct, _render_blue_output(blue_components, grid.shape, expected_vector, force_all="big")):
        return False
    if np.array_equal(correct, _render_blue_output(blue_components, grid.shape, expected_vector, force_all="small")):
        return False
    if np.array_equal(correct, _render_blue_output(blue_components, grid.shape, expected_vector, area_mode="bbox")):
        return False
    if np.array_equal(
        correct,
        _render_blue_output(
            blue_components,
            grid.shape,
            expected_vector,
            endpoint_vector=path_endpoint_vector,
        ),
    ):
        return False

    if require_clip:
        any_clipped = False
        for comp in blue_components:
            if len(comp) > AREA_THRESHOLD:
                moved = _translate_with_clip(comp, expected_vector[0], expected_vector[1], grid.shape)
                if len(moved) < len(comp):
                    any_clipped = True
                    break
        if not any_clipped:
            return False

    # Keep path markers visually isolated.
    reserved = gray_cells | {red, yellow}
    for cell in np.argwhere(grid == BLUE):
        r, c = map(int, cell)
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in reserved:
                return False

    return True


def _generate_once() -> np.ndarray | None:
    H = random.randint(13, 16)
    W = random.randint(13, 16)
    path, red, yellow, vector = _sample_path_scene(H, W)
    start, end = path[0], path[-1]
    endpoint_vector = (end[0] - start[0], end[1] - start[1])

    reserved = _reserve_with_neighbors(set(path) | {red, yellow}, (H, W))

    small_count = random.randint(2, 3)
    large_count = random.randint(2, 3)
    total = small_count + large_count
    if total > 6:
        if random.random() < 0.5:
            small_count -= 1
        else:
            large_count -= 1
    total = small_count + large_count
    if total < 4:
        return None

    small_specs = [_shape_coords(4, require_irregular=True)]
    for _ in range(small_count - 1):
        area = random.choice([1, 2, 3, 4])
        small_specs.append(_shape_coords(area, require_irregular=(area == 4 and random.random() < 0.7)))
    random.shuffle(small_specs)

    large_specs = [_shape_coords(5, require_irregular=True)]
    for _ in range(large_count - 1):
        area = random.choice([5, 6, 7, 8])
        large_specs.append(_shape_coords(area, require_irregular=(area == 5 and random.random() < 0.7)))
    random.shuffle(large_specs)

    small_result = _place_small_objects(small_specs, H, W, reserved, set())
    if small_result is None:
        return None
    small_objects, occ_in, occ_out = small_result

    require_clip = random.random() < 0.35
    big_result = _place_big_objects(
        large_specs,
        H,
        W,
        vector,
        occ_in,
        occ_out,
        force_clip=require_clip,
    )
    if big_result is None:
        return None
    big_objects, clipped = big_result
    if require_clip and not clipped:
        return None

    objects = small_objects + big_objects
    grid = _build_grid(H, W, path, red, yellow, objects)
    if not _validate_scene(grid, vector, endpoint_vector, require_clip=require_clip):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
