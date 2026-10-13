from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
from itertools import permutations
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_for_opus_8_024_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color
components_mask = _ARC_UTILS.components_mask
in_bounds = _ARC_UTILS.in_bounds


BACKGROUND = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREEN = Color.GREEN
GRAY = Color.GRAY
ORANGE = Color.ORANGE

TOP_HEIGHT_RANGE = (6, 8)
GRID_HEIGHT_RANGE = (19, 24)
GRID_WIDTH_RANGE = (20, 24)
PATH_COUNT_OPTIONS = (2, 3)
VECTOR_DRS = (-1, 0, 1)
VECTOR_DCS = (-3, -2, 2, 3)
MARKER_MIN_GAP = 4
SEED_ROW_MIN_GAP = 3
MAX_GENERATION_ATTEMPTS = 300


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev = None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        step = (0 if r1 == r0 else (1 if r1 > r0 else -1), 0 if c1 == c0 else (1 if c1 > c0 else -1))
        if prev is not None and step != prev:
            turns += 1
        prev = step
    return turns


def _walk_path(start: tuple[int, int], waypoints: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    r, c = start
    path = [(int(r), int(c))]
    for nr, nc in waypoints:
        nr = int(nr)
        nc = int(nc)
        if r != nr and c != nc:
            return None
        dr = 0 if nr == r else (1 if nr > r else -1)
        dc = 0 if nc == c else (1 if nc > c else -1)
        while (r, c) != (nr, nc):
            r += dr
            c += dc
            path.append((int(r), int(c)))
    return path


def _extract_paths(grid: np.ndarray) -> list[dict[str, tuple[int, int] | tuple[int, int, int, int] | int]]:
    path_mask = np.isin(grid, [BLUE, RED, GRAY])
    comps = components_mask(path_mask, connectivity=4)
    paths: list[dict[str, tuple[int, int] | tuple[int, int, int, int] | int]] = []
    for coords in comps:
        values = grid[coords[:, 0], coords[:, 1]]
        blue_coords = coords[values == BLUE]
        red_coords = coords[values == RED]
        gray_count = int(np.sum(values == GRAY))
        assert blue_coords.shape[0] == 1, "each path component must have one blue endpoint"
        assert red_coords.shape[0] == 1, "each path component must have one red endpoint"
        assert gray_count >= 1, "each path must contain gray connector pixels"
        blue = tuple(map(int, blue_coords[0]))
        red = tuple(map(int, red_coords[0]))
        paths.append(
            {
                "blue": blue,
                "red": red,
                "vector": (int(red[0] - blue[0]), int(red[1] - blue[1])),
            }
        )

    assert len(paths) in PATH_COUNT_OPTIONS, "expected 2 or 3 encoded paths"
    blue_cols = [int(item["blue"][1]) for item in paths]
    assert len(set(blue_cols)) == len(blue_cols), "blue endpoint columns must be unique"
    return sorted(paths, key=lambda item: (int(item["blue"][1]), int(item["blue"][0])))


def _seed_output(shape: tuple[int, int], obstacles: set[tuple[int, int]], seeds: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in obstacles:
        out[r, c] = GREEN
    for r, c in seeds:
        out[r, c] = ORANGE
    return out


def _simulate_rays(
    base: np.ndarray,
    seeds: list[tuple[int, int]],
    *,
    with_trace: bool = False,
) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    out = np.array(base, dtype=int, copy=True)
    h, w = out.shape
    queue = deque((int(r), int(c)) for r, c in seeds)
    seen: set[tuple[int, int]] = set()
    collisions: list[tuple[int, int]] = []
    branches: list[tuple[int, int]] = []

    def is_obstacle(r: int, c: int) -> bool:
        return in_bounds((h, w), r, c) and int(out[r, c]) == GREEN

    def paint_if_black(r: int, c: int) -> None:
        if in_bounds((h, w), r, c) and int(out[r, c]) == BACKGROUND:
            out[r, c] = ORANGE

    def add_side_wake(r: int, c: int, sign: int) -> None:
        wake = [(r + sign, c), (r + 2 * sign, c), (r + 2 * sign, c + 1)]
        for rr, cc in wake:
            paint_if_black(int(rr), int(cc))

    while queue:
        r, start_c = queue.popleft()
        if not (0 <= r < h):
            continue
        beam_key = (int(r), int(start_c))
        if beam_key in seen:
            continue
        seen.add(beam_key)

        c = int(start_c) + 1
        while c < w:
            if is_obstacle(r, c):
                collisions.append((int(r), int(c)))
                add_side_wake(r, c, -1)
                add_side_wake(r, c, 1)
                if is_obstacle(r - 1, c + 1) and is_obstacle(r + 1, c + 1):
                    branches.append((int(r), int(c)))
                    queue.append((int(r - 1), int(c)))
                    queue.append((int(r + 1), int(c)))
                    break
                c += 1
                continue

            paint_if_black(r, c)
            c += 1

    trace = {"collisions": collisions, "branches": branches}
    if not with_trace:
        trace = {"collisions": [], "branches": []}
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each gray path's endpoint displacement, relocate orange seeds, then fire branching rightward rays through the green obstacle field.

    Input structure:
    - The background is black.
    - The upper region contains 2 or 3 disjoint 4-connected path components made of gray pixels with exactly one blue endpoint and one red endpoint each.
    - The lower region contains the same number of orange marker pixels plus scattered single green obstacle pixels.

    Rule:
    - For each path component, ignore the bends and compute the displacement vector from its blue endpoint to its red endpoint.
    - Sort paths by blue-endpoint column and orange markers by column, then pair them in that order.
    - Translate each paired orange marker by the matched displacement to place a new orange seed, and erase all original path pixels, endpoints, and original orange markers.
    - From every new seed, extend an orange horizontal ray to the right; a hit green obstacle stays fixed, creates upper and lower 3-pixel L-shaped side wakes on black cells only, and if both forward diagonals are green obstacles the parent ray stops while new rays continue on the two adjacent rows.

    Key insight:
    - The gray path geometry is noise: only the blue-to-red endpoint displacement matters, and that decoded vector determines the seed positions that control the entire ray-and-branch output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and max(g.shape) <= 30

    paths = _extract_paths(g)
    vectors = [tuple(map(int, item["vector"])) for item in paths]
    assert len(set(vectors)) == len(vectors), "path displacement vectors must be unique"

    markers = [tuple(map(int, cell)) for cell in np.argwhere(g == ORANGE)]
    assert len(markers) == len(paths), "orange marker count must match path count"
    marker_cols = [c for _r, c in markers]
    assert len(set(marker_cols)) == len(marker_cols), "orange marker columns must be unique"
    markers = sorted(markers, key=lambda p: (p[1], p[0]))

    out = g.copy()
    out[np.isin(out, [BLUE, RED, GRAY, ORANGE])] = BACKGROUND

    seeds: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for marker, vector in zip(markers, vectors):
        sr = int(marker[0] + vector[0])
        sc = int(marker[1] + vector[1])
        assert in_bounds((h, w), sr, sc), "translated seed must stay inside the grid"
        assert int(out[sr, sc]) == BACKGROUND, "translated seed must land on black background"
        assert (sr, sc) not in used, "translated seeds must be distinct"
        out[sr, sc] = ORANGE
        seeds.append((sr, sc))
        used.add((sr, sc))

    simulated, _trace = _simulate_rays(out, seeds, with_trace=False)
    return simulated


def _sample_spaced(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(int(lo), int(hi) + 1))
    random.shuffle(values)
    chosen: list[int] = []
    for value in values:
        if all(abs(value - other) >= min_gap for other in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _make_zones(width: int, count: int) -> list[tuple[int, int]]:
    usable = width - 2 - (count - 1)
    base = usable // count
    extra = usable % count
    zones: list[tuple[int, int]] = []
    c = 1
    for i in range(count):
        zone_w = base + (1 if i < extra else 0)
        zones.append((int(c), int(c + zone_w - 1)))
        c += zone_w + 1
    return zones


def _path_candidates(
    blue: tuple[int, int],
    red: tuple[int, int],
    top_h: int,
    zone: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    br, bc = blue
    rr, rc = red
    c0, c1 = zone
    candidates: list[list[tuple[int, int]]] = []

    def add(waypoints: list[tuple[int, int]]) -> None:
        path = _walk_path(blue, waypoints)
        if path is None or len(path) < 3:
            return
        if len(set(path)) != len(path):
            return
        for r, c in path:
            if not (0 <= r < top_h and c0 <= c <= c1):
                return
            if (r, c) in occupied:
                return
        candidates.append(path)

    add([red])
    add([(br, rc), red])
    add([(rr, bc), red])

    detour_rows = [r for r in range(1, top_h - 1) if r not in (br, rr)]
    random.shuffle(detour_rows)
    for detour_r in detour_rows[:3]:
        add([(detour_r, bc), (detour_r, rc), red])

    detour_cols = [c for c in range(c0 + 1, c1) if c not in (bc, rc)]
    random.shuffle(detour_cols)
    for detour_c in detour_cols[:3]:
        add([(br, detour_c), (rr, detour_c), red])

    for detour_r in detour_rows[:2]:
        for detour_c in detour_cols[:2]:
            add([(detour_r, bc), (detour_r, detour_c), (rr, detour_c), red])
            add([(br, detour_c), (detour_r, detour_c), (detour_r, rc), red])

    uniq: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for path in candidates:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            uniq.append(path)
    return uniq


def _sample_paths(
    top_h: int,
    width: int,
    vectors: list[tuple[int, int]],
) -> list[dict[str, tuple[int, int] | list[tuple[int, int]]]] | None:
    zones = _make_zones(width, len(vectors))
    occupied: set[tuple[int, int]] = set()
    path_specs: list[dict[str, tuple[int, int] | list[tuple[int, int]]]] = []
    need_bent = True

    for vector, zone in zip(vectors, zones):
        dr, dc = vector
        c0, c1 = zone
        endpoint_candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for br in range(1, top_h - 1):
            rr = br + dr
            if not (1 <= rr < top_h - 1):
                continue
            for bc in range(c0 + 1, c1):
                rc = bc + dc
                if not (c0 + 1 <= rc < c1):
                    continue
                endpoint_candidates.append(((int(br), int(bc)), (int(rr), int(rc))))
        random.shuffle(endpoint_candidates)
        if not endpoint_candidates:
            return None

        chosen = None
        fallback = None
        for blue, red in endpoint_candidates:
            candidates = _path_candidates(blue, red, top_h, zone, occupied)
            if not candidates:
                continue
            bent = [path for path in candidates if _turn_count(path) >= 1]
            if need_bent and bent:
                chosen = (blue, red, random.choice(bent))
                break
            if fallback is None:
                fallback = (blue, red, random.choice(candidates))
        if chosen is None:
            chosen = fallback
        if chosen is None:
            return None

        blue, red, path = chosen
        need_bent = need_bent and _turn_count(path) == 0
        occupied.update(path)
        path_specs.append({"blue": blue, "red": red, "path": path})

    if any(_turn_count(spec["path"]) >= 1 for spec in path_specs):
        return path_specs
    return None


def _sample_vectors(count: int) -> list[tuple[int, int]]:
    dcs = list(VECTOR_DCS)
    random.shuffle(dcs)
    vectors = []
    for dc in dcs[:count]:
        dr = random.choice(VECTOR_DRS)
        vectors.append((int(dr), int(dc)))
    return vectors


def _render_input(
    shape: tuple[int, int],
    path_specs: list[dict[str, tuple[int, int] | list[tuple[int, int]]]],
    markers: list[tuple[int, int]],
    obstacles: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in obstacles:
        grid[r, c] = GREEN
    for r, c in markers:
        grid[r, c] = ORANGE
    for spec in path_specs:
        blue = spec["blue"]
        red = spec["red"]
        path = spec["path"]
        for r, c in path[1:-1]:
            grid[r, c] = GRAY
        grid[blue[0], blue[1]] = BLUE
        grid[red[0], red[1]] = RED
    return grid


def _output_from_assignment(
    shape: tuple[int, int],
    obstacles: set[tuple[int, int]],
    markers: list[tuple[int, int]],
    vectors: list[tuple[int, int]],
) -> np.ndarray | None:
    seeds: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for marker, vector in zip(markers, vectors):
        sr = int(marker[0] + vector[0])
        sc = int(marker[1] + vector[1])
        if not in_bounds(shape, sr, sc):
            return None
        if (sr, sc) in obstacles or (sr, sc) in used:
            return None
        seeds.append((sr, sc))
        used.add((sr, sc))
    base = _seed_output(shape, obstacles, seeds)
    out, _trace = _simulate_rays(base, seeds, with_trace=False)
    return out


def _ambiguous_under_other_pairing(
    shape: tuple[int, int],
    obstacles: set[tuple[int, int]],
    markers: list[tuple[int, int]],
    vectors: list[tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    truth = tuple(vectors)
    for perm in permutations(vectors):
        if perm == truth:
            continue
        alt = _output_from_assignment(shape, obstacles, markers, list(perm))
        if alt is not None and np.array_equal(alt, expected):
            return True
    return False


def _build_lower_scene(
    height: int,
    width: int,
    top_h: int,
    vectors: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]], bool] | None:
    count = len(vectors)
    marker_cols = _sample_spaced(4, width - 9, count, MARKER_MIN_GAP)
    seed_rows = _sample_spaced(top_h + 3, height - 4, count, SEED_ROW_MIN_GAP)
    if marker_cols is None or seed_rows is None:
        return None

    random.shuffle(seed_rows)
    markers: list[tuple[int, int]] = []
    seeds: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for seed_row, marker_col, vector in zip(seed_rows, marker_cols, vectors):
        dr, dc = vector
        marker = (int(seed_row - dr), int(marker_col))
        seed = (int(seed_row), int(marker_col + dc))
        if not in_bounds((height, width), marker[0], marker[1]):
            return None
        if not in_bounds((height, width), seed[0], seed[1]):
            return None
        if marker[0] <= top_h or seed[0] <= top_h:
            return None
        if marker in used or seed in used:
            return None
        markers.append(marker)
        seeds.append(seed)
        used.add(marker)
        used.add(seed)

    branch_case = random.random() < 0.6
    branch_index = random.randrange(count) if branch_case else None
    obstacles: set[tuple[int, int]] = set()

    for idx, (r, c) in enumerate(seeds):
        if branch_index == idx:
            possible = [c + gap for gap in (2, 3, 4) if c + gap <= width - 4 and 2 <= r <= height - 3]
            if not possible:
                return None
            parent_c = random.choice(possible)
            cluster = {(r, parent_c), (r - 1, parent_c + 1), (r + 1, parent_c + 1)}
            if cluster & used:
                return None
            obstacles |= cluster
            used |= cluster
        else:
            possible = [c + gap for gap in (2, 3, 4, 5) if c + gap <= width - 3]
            if not possible:
                return None
            obstacle = (int(r), int(random.choice(possible)))
            if obstacle in used:
                return None
            obstacles.add(obstacle)
            used.add(obstacle)

    return markers, seeds, obstacles, branch_case


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    branch_case: bool,
    obstacles: set[tuple[int, int]],
    markers: list[tuple[int, int]],
    vectors: list[tuple[int, int]],
) -> bool:
    parsed = _extract_paths(grid)
    parsed_vectors = [tuple(map(int, item["vector"])) for item in parsed]
    if parsed_vectors != vectors:
        return False

    base = np.array(expected, dtype=int, copy=True)
    trace_base = np.zeros_like(base)
    for r, c in obstacles:
        trace_base[r, c] = GREEN
    seeds = [(int(marker[0] + vector[0]), int(marker[1] + vector[1])) for marker, vector in zip(markers, vectors)]
    for r, c in seeds:
        trace_base[r, c] = ORANGE
    traced, trace = _simulate_rays(trace_base, seeds, with_trace=True)
    if not np.array_equal(traced, expected):
        return False
    if len(trace["collisions"]) < len(seeds):
        return False
    if branch_case and len(trace["branches"]) < 1:
        return False
    if not branch_case and len(trace["branches"]) != 0:
        return False
    if _ambiguous_under_other_pairing(grid.shape, obstacles, markers, vectors, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    top_h = random.randint(*TOP_HEIGHT_RANGE)
    count = random.choice(PATH_COUNT_OPTIONS)

    vectors = _sample_vectors(count)
    path_specs = _sample_paths(top_h, width, vectors)
    if path_specs is None:
        return None

    lower_scene = _build_lower_scene(height, width, top_h, vectors)
    if lower_scene is None:
        return None
    markers, _seeds, obstacles, branch_case = lower_scene

    grid = _render_input((height, width), path_specs, markers, obstacles)

    try:
        expected = main(grid)
    except AssertionError:
        return None

    if not _semantic_checks(grid, expected, branch_case, obstacles, sorted(markers, key=lambda p: (p[1], p[0])), vectors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-6 input")
