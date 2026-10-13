from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
SEPARATOR = 6
BASE = 8
ENDPOINT = 9

GRID_MIN = 13
GRID_MAX = 17
LATTICE_SIZE = 11
TILE_COUNT = 3
TILE_SIZE = 3
TILE_STEP = 4
SEPARATOR_INDEXES = {3, 7}
MARKER_COLORS = (1, 2, 3, 4, 7)
VECTOR_OPTIONS = (
    (0, 1),
    (0, 2),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, 0),
)
MAX_ATTEMPTS = 400

REL_TO_TILE = {
    0: (0, 0),
    1: (0, 1),
    2: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    8: (2, 0),
    9: (2, 1),
    10: (2, 2),
}
LOCAL_OFFSETS = tuple((lr, lc) for lr in range(TILE_SIZE) for lc in range(TILE_SIZE))
TILE_DIRS = {
    (-1, 0): "U",
    (1, 0): "D",
    (0, -1): "L",
    (0, 1): "R",
}
ANCHORS = {"U": (0, 1), "D": (2, 1), "L": (1, 0), "R": (1, 2)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _tile_local_from_raw(top: int, left: int, r: int, c: int) -> tuple[int, int, int, int]:
    rel_r = int(r - top)
    rel_c = int(c - left)
    assert 0 <= rel_r < LATTICE_SIZE and 0 <= rel_c < LATTICE_SIZE
    assert rel_r not in SEPARATOR_INDEXES and rel_c not in SEPARATOR_INDEXES
    tile_r, local_r = REL_TO_TILE[rel_r]
    tile_c, local_c = REL_TO_TILE[rel_c]
    return tile_r, tile_c, local_r, local_c


def _raw_from_tile(top: int, left: int, tile_r: int, tile_c: int, local_r: int, local_c: int) -> tuple[int, int]:
    return int(top + tile_r * TILE_STEP + local_r), int(left + tile_c * TILE_STEP + local_c)


def _endpoint_order_key(item: tuple[int, int, int, int]) -> tuple[int, int]:
    tile_r, tile_c, _r, _c = item
    return tile_r, tile_c


def _validate_path_component(region: np.ndarray, endpoints: list[tuple[int, int]]) -> None:
    mask = (region == PATH) | (region == ENDPOINT)
    assert int(mask.sum()) >= 3
    local_endpoints = {(r, c) for r, c in endpoints}
    seen = set()
    q = deque([endpoints[0]])
    seen.add(endpoints[0])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < region.shape[0] and 0 <= nc < region.shape[1]):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))

    assert seen == set(map(tuple, np.argwhere(mask)))
    for r, c in seen:
        degree = sum(
            1
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < region.shape[0] and 0 <= nc < region.shape[1] and mask[nr, nc]
        )
        if (r, c) in local_endpoints:
            assert degree == 1
        else:
            assert degree == 2


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset(set(range(10)))

    bbox = _bbox(g != BG)
    top, left, bottom, right = bbox
    assert bottom - top + 1 == LATTICE_SIZE
    assert right - left + 1 == LATTICE_SIZE
    region = g[top : bottom + 1, left : right + 1]
    assert np.all(region != BG)

    colors, counts = np.unique(region, return_counts=True)
    freq = sorted(
        (
            (int(count), int(color))
            for color, count in zip(colors.tolist(), counts.tolist(), strict=False)
            if int(color) not in (PATH, SEPARATOR, ENDPOINT)
        ),
        reverse=True,
    )
    assert freq
    base = freq[0][1]
    assert base == BASE

    endpoint_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == ENDPOINT)]
    assert len(endpoint_cells) == 2
    endpoint_info = []
    for r, c in endpoint_cells:
        tile_r, tile_c, local_r, local_c = _tile_local_from_raw(top, left, r, c)
        assert (local_r, local_c) == (1, 1)
        endpoint_info.append((tile_r, tile_c, r, c))
    endpoint_info.sort(key=_endpoint_order_key)
    src_tile_r, src_tile_c, src_r, src_c = endpoint_info[0]
    dst_tile_r, dst_tile_c, dst_r, dst_c = endpoint_info[1]
    vector = (dst_tile_r - src_tile_r, dst_tile_c - src_tile_c)
    assert vector != (0, 0)

    local_endpoints = [(src_r - top, src_c - left), (dst_r - top, dst_c - left)]
    _validate_path_component(region, local_endpoints)

    markers = []
    for r, c in np.argwhere(g != BG):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if color in (BASE, PATH, SEPARATOR, ENDPOINT):
            continue
        tile_r, tile_c, local_r, local_c = _tile_local_from_raw(top, left, r, c)
        markers.append(
            {
                "color": color,
                "tile": (tile_r, tile_c),
                "local": (local_r, local_c),
                "raw": (r, c),
            }
        )
    assert 2 <= len(markers) <= 4
    assert len({m["tile"] for m in markers}) == len(markers)

    return {
        "grid": g,
        "top": top,
        "left": left,
        "region": region,
        "base": base,
        "vector": vector,
        "markers": markers,
        "endpoints": ((src_r, src_c), (dst_r, dst_c)),
    }


def _render_output_from_scene(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    top = scene["top"]
    left = scene["left"]
    base = scene["base"]
    out[top : top + LATTICE_SIZE, left : left + LATTICE_SIZE][
        out[top : top + LATTICE_SIZE, left : left + LATTICE_SIZE] == PATH
    ] = base
    out[top : top + LATTICE_SIZE, left : left + LATTICE_SIZE][
        out[top : top + LATTICE_SIZE, left : left + LATTICE_SIZE] == ENDPOINT
    ] = base

    dr, dc = scene["vector"]
    for marker in scene["markers"]:
        tile_r, tile_c = marker["tile"]
        local_r, local_c = marker["local"]
        target_tile_r = tile_r + dr
        target_tile_c = tile_c + dc
        if 0 <= target_tile_r < TILE_COUNT and 0 <= target_tile_c < TILE_COUNT:
            rr, cc = _raw_from_tile(top, left, target_tile_r, target_tile_c, local_r, local_c)
            out[rr, cc] = marker["color"]
    return out


def _render_wrong_center_output(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    top = scene["top"]
    left = scene["left"]
    out[(out == PATH) | (out == ENDPOINT)] = scene["base"]
    dr, dc = scene["vector"]
    for marker in scene["markers"]:
        tile_r, tile_c = marker["tile"]
        target_tile_r = tile_r + dr
        target_tile_c = tile_c + dc
        if 0 <= target_tile_r < TILE_COUNT and 0 <= target_tile_c < TILE_COUNT:
            rr, cc = _raw_from_tile(top, left, target_tile_r, target_tile_c, 1, 1)
            out[rr, cc] = marker["color"]
    return out


def _render_wrong_pixel_step_output(scene: dict) -> np.ndarray:
    out = scene["grid"].copy()
    out[(out == PATH) | (out == ENDPOINT)] = scene["base"]
    dr, dc = scene["vector"]
    for marker in scene["markers"]:
        r, c = marker["raw"]
        rr = r + dr
        cc = c + dc
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = marker["color"]
    return out


def _render_with_vector(scene: dict, vector: tuple[int, int]) -> np.ndarray:
    out = scene["grid"].copy()
    out[(out == PATH) | (out == ENDPOINT)] = scene["base"]
    dr, dc = vector
    top = scene["top"]
    left = scene["left"]
    for marker in scene["markers"]:
        tile_r, tile_c = marker["tile"]
        local_r, local_c = marker["local"]
        rr_tile = tile_r + dr
        cc_tile = tile_c + dc
        if 0 <= rr_tile < TILE_COUNT and 0 <= cc_tile < TILE_COUNT:
            rr, cc = _raw_from_tile(top, left, rr_tile, cc_tile, local_r, local_c)
            out[rr, cc] = marker["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each tile marker by the tile displacement encoded by the maroon path endpoints while keeping its local tile offset.

    Input structure:
    - A black background surrounds one solid 11x11 non-black lattice region that contains a 3x3 array of 3x3 tiles.
    - The tiles share one base color, with two separator rows and two separator columns originally in magenta; some separator cells may be overwritten by a gray path.
    - Exactly two maroon pixels sit at tile centers, and a single 4-connected gray path links them.
    - Two to four other foreground marker pixels appear inside non-endpoint tiles at arbitrary local offsets within their 3x3 tiles.

    Rule:
    - Read the two maroon center pixels in tile coordinates, ordered top-to-bottom and then left-to-right, and use their tile-coordinate difference as a displacement vector.
    - For every marker pixel, identify its tile and its local offset inside that tile, then copy that color to the tile reached by adding the displacement vector while preserving the same local offset.
    - Keep the original marker pixels unchanged.
    - Restore every gray-path pixel and maroon endpoint pixel to the tile base color.

    Key insight:
    - The gray connector shape is a distractor: only the ordered pair of maroon endpoints matters, and markers move by tile displacement while keeping their within-tile position fixed.
    """
    scene = _parse_grid(np.asarray(grid, dtype=int))
    return _render_output_from_scene(scene)


def _centered_origin(limit: int) -> int:
    slack = limit - LATTICE_SIZE
    lo = slack // 2
    hi = slack - lo
    return random.choice((lo, hi))


def _valid_endpoint_sources(vector: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = vector
    valid = []
    for tile_r in range(TILE_COUNT):
        for tile_c in range(TILE_COUNT):
            rr = tile_r + dr
            cc = tile_c + dc
            if 0 <= rr < TILE_COUNT and 0 <= cc < TILE_COUNT:
                remaining = [
                    (r, c)
                    for r in range(TILE_COUNT)
                    for c in range(TILE_COUNT)
                    if (r, c) not in {(tile_r, tile_c), (rr, cc)}
                    and 0 <= r + dr < TILE_COUNT
                    and 0 <= c + dc < TILE_COUNT
                ]
                if len(remaining) >= 2:
                    valid.append((tile_r, tile_c))
    return valid


def _candidate_marker_tiles(
    vector: tuple[int, int], endpoint_tiles: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    dr, dc = vector
    return [
        (tile_r, tile_c)
        for tile_r in range(TILE_COUNT)
        for tile_c in range(TILE_COUNT)
        if (tile_r, tile_c) not in endpoint_tiles
        and 0 <= tile_r + dr < TILE_COUNT
        and 0 <= tile_c + dc < TILE_COUNT
    ]


def _make_base_grid(height: int, width: int, top: int, left: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[top : top + LATTICE_SIZE, left : left + LATTICE_SIZE] = BASE
    for rel in SEPARATOR_INDEXES:
        grid[top + rel, left : left + LATTICE_SIZE] = SEPARATOR
        grid[top : top + LATTICE_SIZE, left + rel] = SEPARATOR
    return grid


def _build_segments(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [points[0]]
    for nr, nc in points[1:]:
        r, c = path[-1]
        if r != nr and c != nc:
            return []
        step_r = 0 if r == nr else (1 if nr > r else -1)
        step_c = 0 if c == nc else (1 if nc > c else -1)
        while (r, c) != (nr, nc):
            r += step_r
            c += step_c
            path.append((r, c))
    return path


def _tile_routes(start: tuple[int, int], end: tuple[int, int]) -> list[list[tuple[int, int]]]:
    routes: list[list[tuple[int, int]]] = []

    def dfs(path: list[tuple[int, int]]) -> None:
        tile_r, tile_c = path[-1]
        if (tile_r, tile_c) == end:
            routes.append(path[:])
            return
        if len(path) >= 5:
            return
        for dr, dc in TILE_DIRS:
            nr = tile_r + dr
            nc = tile_c + dc
            nxt = (nr, nc)
            if not (0 <= nr < TILE_COUNT and 0 <= nc < TILE_COUNT):
                continue
            if nxt in path:
                continue
            dfs(path + [nxt])

    dfs([start])
    shortest = min(len(route) for route in routes)
    return [route for route in routes if len(route) <= shortest + 2]


def _local_paths(
    start_local: tuple[int, int],
    end_local: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    paths: list[list[tuple[int, int]]] = []

    def dfs(path: list[tuple[int, int]]) -> None:
        r, c = path[-1]
        if (r, c) == end_local:
            paths.append(path[:])
            return
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if not (0 <= nr < TILE_SIZE and 0 <= nc < TILE_SIZE):
                continue
            if nxt in path:
                continue
            if nxt in blocked and nxt != end_local:
                continue
            dfs(path + [nxt])

    if start_local in blocked or end_local in blocked:
        return []
    dfs([start_local])
    dedup = {}
    for path in paths:
        dedup[tuple(path)] = path
    return sorted(dedup.values(), key=lambda p: (len(p), p))


def _separator_cell(
    top: int,
    left: int,
    src_tile: tuple[int, int],
    dst_tile: tuple[int, int],
) -> tuple[int, int]:
    dr = dst_tile[0] - src_tile[0]
    dc = dst_tile[1] - src_tile[1]
    direction = TILE_DIRS[(dr, dc)]
    exit_local = ANCHORS[direction]
    rr, cc = _raw_from_tile(top, left, src_tile[0], src_tile[1], exit_local[0], exit_local[1])
    if direction == "U":
        return rr - 1, cc
    if direction == "D":
        return rr + 1, cc
    if direction == "L":
        return rr, cc - 1
    return rr, cc + 1


def _path_options_for_route(
    top: int,
    left: int,
    route: list[tuple[int, int]],
    blocked_raw: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    by_tile_options: list[list[list[tuple[int, int]]]] = []
    used_tiles = set(route)
    assert len(used_tiles) == len(route)

    for idx, tile in enumerate(route):
        if idx == 0:
            start_local = (1, 1)
            next_tile = route[idx + 1]
            move = (next_tile[0] - tile[0], next_tile[1] - tile[1])
            end_local = ANCHORS[TILE_DIRS[move]]
        elif idx == len(route) - 1:
            prev_tile = route[idx - 1]
            move = (tile[0] - prev_tile[0], tile[1] - prev_tile[1])
            start_local = ANCHORS[OPPOSITE[TILE_DIRS[move]]]
            end_local = (1, 1)
        else:
            prev_tile = route[idx - 1]
            next_tile = route[idx + 1]
            prev_move = (tile[0] - prev_tile[0], tile[1] - prev_tile[1])
            next_move = (next_tile[0] - tile[0], next_tile[1] - tile[1])
            start_local = ANCHORS[OPPOSITE[TILE_DIRS[prev_move]]]
            end_local = ANCHORS[TILE_DIRS[next_move]]

        blocked_local = set()
        for raw in blocked_raw:
            try:
                tile_r, tile_c, local_r, local_c = _tile_local_from_raw(top, left, raw[0], raw[1])
            except AssertionError:
                continue
            if (tile_r, tile_c) == tile:
                blocked_local.add((local_r, local_c))
        blocked_local.discard(start_local)
        blocked_local.discard(end_local)
        options = _local_paths(start_local, end_local, blocked_local)
        if not options:
            return []
        by_tile_options.append(options)

    def assemble(choice_indexes: list[int]) -> list[tuple[int, int]]:
        combined: list[tuple[int, int]] = []
        for idx, tile in enumerate(route):
            local_path = by_tile_options[idx][choice_indexes[idx]]
            raw_path = [
                _raw_from_tile(top, left, tile[0], tile[1], local_r, local_c)
                for local_r, local_c in local_path
            ]
            if idx == 0:
                combined.extend(raw_path)
            else:
                combined.append(_separator_cell(top, left, route[idx - 1], tile))
                combined.extend(raw_path)
        return combined

    seen: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    base_indexes = [0] * len(route)
    base_path = assemble(base_indexes)
    seen[tuple(base_path)] = base_path
    for idx, options in enumerate(by_tile_options):
        for alt_idx in range(1, min(len(options), 4)):
            choice_indexes = base_indexes[:]
            choice_indexes[idx] = alt_idx
            alt_path = assemble(choice_indexes)
            seen[tuple(alt_path)] = alt_path
    return sorted(seen.values(), key=lambda p: (len(p), p))


def _valid_path(
    path: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    reserved: set[tuple[int, int]],
) -> bool:
    if not path:
        return False
    top, left, bottom, right = bounds
    if len(set(path)) != len(path):
        return False
    for i, (r, c) in enumerate(path):
        if not (top <= r <= bottom and left <= c <= right):
            return False
        rel_r = r - top
        rel_c = c - left
        if rel_r in SEPARATOR_INDEXES or rel_c in SEPARATOR_INDEXES:
            return False
        if i not in (0, len(path) - 1) and (r, c) in reserved:
            return False
    path_set = set(path)
    for i, (r, c) in enumerate(path):
        degree = sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c))
        if i in (0, len(path) - 1):
            if degree != 1:
                return False
        elif degree != 2:
            return False
    return True


def _enumerate_paths(
    start: tuple[int, int],
    end: tuple[int, int],
    bounds: tuple[int, int, int, int],
    reserved: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    top, left, bottom, right = bounds
    sr, sc = start
    er, ec = end
    candidates: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}

    waypoint_sets = [
        [start, (sr, ec), end],
        [start, (er, sc), end],
    ]
    for row in range(top, bottom + 1):
        waypoint_sets.append([start, (row, sc), (row, ec), end])
    for col in range(left, right + 1):
        waypoint_sets.append([start, (sr, col), (er, col), end])
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            waypoint_sets.append([start, (sr, col), (row, col), (row, ec), end])
            waypoint_sets.append([start, (row, sc), (row, col), (er, col), end])

    for points in waypoint_sets:
        path = _build_segments(points)
        if _valid_path(path, bounds, reserved):
            candidates[tuple(path)] = path

    ordered = sorted(candidates.values(), key=lambda p: (len(p), p))
    return ordered


def _pick_marker_offsets(
    top: int,
    left: int,
    marker_tiles: list[tuple[int, int]],
    endpoint_raw: set[tuple[int, int]],
    vector: tuple[int, int],
    blocked_raw: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    dr, dc = vector
    chosen: list[tuple[int, int]] = []
    source_raw: list[tuple[int, int]] = []
    target_raw: set[tuple[int, int]] = set()

    tiles = marker_tiles[:]
    random.shuffle(tiles)
    for tile_r, tile_c in tiles:
        random_offsets = list(LOCAL_OFFSETS)
        random.shuffle(random_offsets)
        placed = False
        for local_r, local_c in random_offsets:
            src = _raw_from_tile(top, left, tile_r, tile_c, local_r, local_c)
            dst = _raw_from_tile(top, left, tile_r + dr, tile_c + dc, local_r, local_c)
            if dst in endpoint_raw or dst in target_raw:
                continue
            if dst in source_raw:
                continue
            if src in blocked_raw or dst in blocked_raw:
                continue
            chosen.append((local_r, local_c))
            source_raw.append(src)
            target_raw.add(dst)
            placed = True
            break
        if not placed:
            raise ValueError("could not assign marker offsets")

    if not any(offset != (1, 1) for offset in chosen):
        raise ValueError("all markers landed at tile centers")
    return chosen


def _render_input_scene(
    height: int,
    width: int,
    top: int,
    left: int,
    vector: tuple[int, int],
    endpoint_src: tuple[int, int],
    marker_tiles: list[tuple[int, int]],
    marker_offsets: list[tuple[int, int]],
    marker_colors: list[int],
    path: list[tuple[int, int]],
) -> np.ndarray:
    grid = _make_base_grid(height, width, top, left)
    src_tile_r, src_tile_c = endpoint_src
    dr, dc = vector
    end_tile = (src_tile_r + dr, src_tile_c + dc)

    for r, c in path[1:-1]:
        grid[r, c] = PATH

    sr, sc = _raw_from_tile(top, left, src_tile_r, src_tile_c, 1, 1)
    er, ec = _raw_from_tile(top, left, end_tile[0], end_tile[1], 1, 1)
    grid[sr, sc] = ENDPOINT
    grid[er, ec] = ENDPOINT

    for (tile_r, tile_c), (local_r, local_c), color in zip(
        marker_tiles, marker_offsets, marker_colors, strict=False
    ):
        rr, cc = _raw_from_tile(top, left, tile_r, tile_c, local_r, local_c)
        grid[rr, cc] = color
    return grid


def _scene_metadata_from_choices(
    grid: np.ndarray,
    vector: tuple[int, int],
    marker_tiles: list[tuple[int, int]],
    marker_offsets: list[tuple[int, int]],
) -> dict:
    scene = _parse_grid(grid)
    assert scene["vector"] == vector
    parsed_pairs = sorted((m["tile"], m["local"]) for m in scene["markers"])
    expected_pairs = sorted(zip(marker_tiles, marker_offsets, strict=False))
    assert parsed_pairs == expected_pairs
    return scene


def _check_semantics(input_grid: np.ndarray, alt_grid: np.ndarray | None = None) -> None:
    scene = _parse_grid(input_grid)
    expected = _render_output_from_scene(scene)
    actual = main(input_grid)
    assert np.array_equal(actual, expected)
    if alt_grid is not None:
        assert not np.array_equal(alt_grid, input_grid)
        assert np.array_equal(main(alt_grid), expected)
    assert not np.array_equal(expected, _render_wrong_center_output(scene))
    assert not np.array_equal(expected, _render_wrong_pixel_step_output(scene))
    assert not np.array_equal(
        expected,
        _render_with_vector(scene, (-scene["vector"][0], -scene["vector"][1])),
    )


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        top = _centered_origin(height)
        left = _centered_origin(width)

        vector = random.choice(VECTOR_OPTIONS)
        endpoint_sources = _valid_endpoint_sources(vector)
        if not endpoint_sources:
            continue
        endpoint_src = random.choice(endpoint_sources)
        endpoint_dst = (endpoint_src[0] + vector[0], endpoint_src[1] + vector[1])
        endpoint_tiles = {endpoint_src, endpoint_dst}

        route_options = _tile_routes(endpoint_src, endpoint_dst)
        random.shuffle(route_options)
        path = None
        alt_path = None
        blocked_for_markers: set[tuple[int, int]] | None = None
        for route in route_options:
            route_paths = _path_options_for_route(top, left, route, set())
            if len(route_paths) < 2:
                continue
            path, alt_path = random.sample(route_paths, 2)
            blocked_for_markers = set(path) | set(alt_path)
            break
        if path is None or alt_path is None or blocked_for_markers is None:
            continue

        candidates = _candidate_marker_tiles(vector, endpoint_tiles)
        if len(candidates) < 2:
            continue
        n_markers = random.randint(2, min(4, len(candidates)))

        endpoint_raw = {
            _raw_from_tile(top, left, endpoint_src[0], endpoint_src[1], 1, 1),
            _raw_from_tile(top, left, endpoint_dst[0], endpoint_dst[1], 1, 1),
        }
        marker_tiles = None
        marker_offsets = None
        for _marker_attempt in range(40):
            sampled_tiles = random.sample(candidates, n_markers)
            try:
                sampled_offsets = _pick_marker_offsets(
                    top,
                    left,
                    sampled_tiles,
                    endpoint_raw,
                    vector,
                    blocked_for_markers,
                )
            except ValueError:
                continue
            marker_tiles = sampled_tiles
            marker_offsets = sampled_offsets
            break
        if marker_tiles is None or marker_offsets is None:
            continue

        marker_colors = random.sample(MARKER_COLORS, n_markers)

        input_grid = _render_input_scene(
            height,
            width,
            top,
            left,
            vector,
            endpoint_src,
            marker_tiles,
            marker_offsets,
            marker_colors,
            path,
        )
        alt_grid = _render_input_scene(
            height,
            width,
            top,
            left,
            vector,
            endpoint_src,
            marker_tiles,
            marker_offsets,
            marker_colors,
            alt_path,
        )

        try:
            _check_semantics(input_grid, alt_grid)
        except AssertionError:
            continue
        return input_grid

    raise RuntimeError("failed to generate a valid puzzle instance")
