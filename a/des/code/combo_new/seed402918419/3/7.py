from __future__ import annotations

import random

import numpy as np


BG = 0
MIRROR = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS: dict[str, tuple[int, int]] = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
CW: dict[str, str] = {"U": "R", "R": "D", "D": "L", "L": "U"}
HORIZONTAL_DIRS = {"L", "R"}
VERTICAL_DIRS = {"U", "D"}
NEIGHBORS_4 = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
GRID_SIDE_RANGE = (14, 24)
TOWER_COUNT_CHOICES = (2, 3)
OUTER_ATTEMPTS = 400
TOWER_PLACEMENT_ATTEMPTS = 80
RAY_PLANNING_ATTEMPTS = 120


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if (
                        _in_bounds(mask.shape, nr, nc)
                        and bool(mask[nr, nc])
                        and not bool(seen[nr, nc])
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _tower_cells(br: int, bc: int, direction: str, offset: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    body = {
        (br, bc),
        (br, bc + 1),
        (br + 1, bc),
        (br + 1, bc + 1),
    }
    if direction == "U":
        tip = (br - 1, bc + offset)
    elif direction == "D":
        tip = (br + 2, bc + offset)
    elif direction == "L":
        tip = (br + offset, bc - 1)
    else:
        assert direction == "R"
        tip = (br + offset, bc + 2)
    return body | {tip}, tip


def _parse_tower_component(coords: list[tuple[int, int]]) -> dict[str, object]:
    pts = set(coords)
    assert len(pts) == 5, "tower must occupy exactly 5 cells"
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    candidates: list[dict[str, object]] = []
    for br in range(min(rows) - 1, max(rows) + 1):
        for bc in range(min(cols) - 1, max(cols) + 1):
            body = {
                (br, bc),
                (br, bc + 1),
                (br + 1, bc),
                (br + 1, bc + 1),
            }
            if not body.issubset(pts):
                continue
            extras = pts - body
            if len(extras) != 1:
                continue
            tip = next(iter(extras))
            tr, tc = tip
            direction = None
            if tr == br - 1 and tc in (bc, bc + 1):
                direction = "U"
            elif tr == br + 2 and tc in (bc, bc + 1):
                direction = "D"
            elif tc == bc - 1 and tr in (br, br + 1):
                direction = "L"
            elif tc == bc + 2 and tr in (br, br + 1):
                direction = "R"
            if direction is None:
                continue
            candidates.append(
                {
                    "body_top_left": (br, bc),
                    "body": body,
                    "tip": tip,
                    "dir": direction,
                }
            )
    assert len(candidates) == 1, "tower shape must encode a unique direction"
    return candidates[0]


def _find_towers(g: np.ndarray) -> list[dict[str, object]]:
    towers: list[dict[str, object]] = []
    colors = sorted(set(map(int, g.flatten())) - {BG, MIRROR})
    for color in colors:
        for comp in _connected_components(g == int(color)):
            tower = _parse_tower_component(comp)
            tower["color"] = int(color)
            towers.append(tower)
    towers.sort(key=lambda tower: (tower["tip"][0], tower["tip"][1], tower["color"]))  # type: ignore[index]
    assert 2 <= len(towers) <= 3, "expected 2-3 towers"
    return towers


def _find_mirrors(g: np.ndarray) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    for comp in _connected_components(g == MIRROR):
        pts = sorted(comp)
        assert len(pts) == 3, "mirror must occupy exactly 3 cells"
        rows = sorted({r for r, _c in pts})
        cols = sorted({c for _r, c in pts})
        if len(rows) == 1:
            assert cols == list(range(cols[0], cols[0] + 3)), "horizontal mirror must be contiguous"
            orientation = "H"
        else:
            assert len(cols) == 1, "mirror must be a straight segment"
            assert rows == list(range(rows[0], rows[0] + 3)), "vertical mirror must be contiguous"
            orientation = "V"
        for cell in pts:
            orientations[cell] = orientation
    return orientations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays from tower antennas, turning clockwise at orthogonal grey mirrors.

    Input structure:
    - The background is black.
    - There are 2-3 signal towers, each a monochromatic 2x2 body with one same-color antenna pixel attached on one side; the antenna side gives the emission direction.
    - Grey foreground objects are isolated 1x3 or 3x1 mirror segments.

    Rule:
    - For each tower, start at the antenna tip and advance one cell at a time in the antenna's direction.
    - If the next cell is outside the grid, the ray stops. If the next cell is a perpendicular mirror, the ray turns 90 degrees clockwise without recoloring the mirror.
    - Every visited black cell becomes the tower's color, while towers and mirrors remain unchanged.

    Key insight:
    - The puzzle combines local shape parsing and path simulation: first recover each tower's direction from its asymmetric 5-cell shape, then follow the ray with clockwise turns only when it meets the opposite mirror orientation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    towers = _find_towers(g)
    mirrors = _find_mirrors(g)

    for tower in towers:
        r, c = tower["tip"]  # type: ignore[index]
        direction = tower["dir"]  # type: ignore[index]
        color = int(tower["color"])  # type: ignore[index]
        seen: set[tuple[int, int, str]] = set()

        while True:
            state = (int(r), int(c), str(direction))
            if state in seen:
                break
            seen.add(state)

            dr, dc = DIRS[str(direction)]
            nr, nc = int(r) + dr, int(c) + dc
            if not _in_bounds(g.shape, nr, nc):
                break

            orientation = mirrors.get((nr, nc))
            if orientation == "V" and str(direction) in HORIZONTAL_DIRS:
                direction = CW[str(direction)]
                continue
            if orientation == "H" and str(direction) in VERTICAL_DIRS:
                direction = CW[str(direction)]
                continue

            if int(g[nr, nc]) == BG:
                out[nr, nc] = color
            r, c = nr, nc

    return out


def _expand_object_forbidden(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc):
                forbidden.add((nr, nc))
    return forbidden


def _mirror_cells(hit: tuple[int, int], incoming_dir: str, shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    r, c = hit
    if incoming_dir in HORIZONTAL_DIRS:
        cells = {(r - 1, c), (r, c), (r + 1, c)}
    else:
        cells = {(r, c - 1), (r, c), (r, c + 1)}
    if all(_in_bounds(shape, rr, cc) for rr, cc in cells):
        return cells
    return None


def _walk_to_boundary(
    start: tuple[int, int],
    direction: str,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = start
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            break
        cells.append((r, c))
    return cells


def _plan_ray(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: str,
    reflections: int,
    blocked_objects: set[tuple[int, int]],
    blocked_paths: set[tuple[int, int]],
    object_forbidden: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[set[tuple[int, int]]]] | None:
    local_path: list[tuple[int, int]] = []
    local_path_set: set[tuple[int, int]] = set()
    local_objects: set[tuple[int, int]] = set()
    local_object_forbidden: set[tuple[int, int]] = set()
    current = start
    cur_dir = direction

    for _ in range(reflections):
        candidates: list[tuple[list[tuple[int, int]], set[tuple[int, int]], tuple[int, int], str]] = []
        dr, dc = DIRS[cur_dir]
        distance = 0
        segment_cells: list[tuple[int, int]] = []
        while True:
            distance += 1
            pr = current[0] + dr * distance
            pc = current[1] + dc * distance
            if not _in_bounds(shape, pr, pc):
                break
            if (
                (pr, pc) in blocked_objects
                or (pr, pc) in local_objects
                or (pr, pc) in blocked_paths
                or (pr, pc) in local_path_set
            ):
                break
            segment_cells.append((pr, pc))

            hit = (current[0] + dr * (distance + 1), current[1] + dc * (distance + 1))
            if not _in_bounds(shape, hit[0], hit[1]):
                break

            mirror = _mirror_cells(hit, cur_dir, shape)
            if mirror is None:
                break
            if (
                mirror & blocked_objects
                or mirror & local_objects
                or mirror & blocked_paths
                or mirror & local_path_set
                or mirror & object_forbidden
                or mirror & local_object_forbidden
            ):
                continue

            endpoint = segment_cells[-1]
            next_dir = CW[cur_dir]
            first_after_turn = (endpoint[0] + DIRS[next_dir][0], endpoint[1] + DIRS[next_dir][1])
            if not _in_bounds(shape, first_after_turn[0], first_after_turn[1]):
                continue
            if (
                first_after_turn in blocked_objects
                or first_after_turn in local_objects
                or first_after_turn in blocked_paths
                or first_after_turn in local_path_set
            ):
                continue

            candidates.append((segment_cells.copy(), mirror, endpoint, next_dir))

        if not candidates:
            return None

        segment, mirror_cells, endpoint, cur_dir = random.choice(candidates)
        local_path.extend(segment)
        local_path_set.update(segment)
        local_objects |= mirror_cells
        local_object_forbidden |= _expand_object_forbidden(mirror_cells, shape)
        current = endpoint

    final_cells = _walk_to_boundary(current, cur_dir, shape)
    if not final_cells:
        return None
    if any(
        cell in blocked_objects
        or cell in local_objects
        or cell in blocked_paths
        or cell in local_path_set
        for cell in final_cells
    ):
        return None
    local_path.extend(final_cells)

    mirror_components: list[set[tuple[int, int]]] = []
    for comp in _connected_components(_mask_from_cells(local_objects, shape)):
        mirror_components.append(set(comp))
    return local_path, mirror_components


def _mask_from_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _tower_candidates(
    shape: tuple[int, int],
    direction: str,
    object_forbidden: set[tuple[int, int]],
    blocked_paths: set[tuple[int, int]],
) -> list[tuple[int, int, int, set[tuple[int, int]], tuple[int, int]]]:
    h, w = shape
    candidates: list[tuple[int, int, int, set[tuple[int, int]], tuple[int, int]]] = []
    for br in range(h - 1):
        for bc in range(w - 1):
            for offset in (0, 1):
                cells, tip = _tower_cells(br, bc, direction, offset)
                if not all(_in_bounds(shape, r, c) for r, c in cells):
                    continue
                dr, dc = DIRS[direction]
                first_ray_cell = (tip[0] + dr, tip[1] + dc)
                if not _in_bounds(shape, first_ray_cell[0], first_ray_cell[1]):
                    continue
                if cells & object_forbidden:
                    continue
                if cells & blocked_paths:
                    continue
                candidates.append((br, bc, offset, cells, tip))
    random.shuffle(candidates)
    return candidates


def _build_expected(
    inp: np.ndarray,
    towers: list[dict[str, object]],
    path_by_color: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = inp.copy()
    for tower in towers:
        color = int(tower["color"])  # type: ignore[index]
        for r, c in path_by_color[color]:
            assert int(inp[r, c]) == BG
            out[r, c] = color
    return out


def _count_reflections(path: list[tuple[int, int]], tip: tuple[int, int]) -> int:
    prev = tip
    prev_delta = None
    turns = 0
    for cell in path:
        delta = (cell[0] - prev[0], cell[1] - prev[1])
        if prev_delta is not None and delta != prev_delta:
            turns += 1
        prev_delta = delta
        prev = cell
    return turns


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    tower_count = random.choice(TOWER_COUNT_CHOICES)
    colors = random.sample(TOWER_COLORS, tower_count)

    directions = [
        random.choice(["L", "R"]),
        random.choice(["U", "D"]),
    ]
    if tower_count == 3:
        directions.append(random.choice(list(DIRS)))
    random.shuffle(directions)

    reflection_counts = [0, random.choice([1, 2])]
    if tower_count == 3:
        reflection_counts.append(random.choice([0, 1, 2]))
    random.shuffle(reflection_counts)

    object_cells: set[tuple[int, int]] = set()
    object_forbidden: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    towers: list[dict[str, object]] = []
    mirror_components: list[set[tuple[int, int]]] = []
    path_by_color: dict[int, list[tuple[int, int]]] = {}

    for color, direction, reflections in zip(colors, directions, reflection_counts, strict=True):
        tower_built = False
        candidates = _tower_candidates(shape, direction, object_forbidden, path_cells)
        for br, bc, offset, cells, tip in candidates[:TOWER_PLACEMENT_ATTEMPTS]:
            tower_forbidden = _expand_object_forbidden(cells, shape)
            blocked_objects = object_cells | cells
            blocked_paths = path_cells
            combined_forbidden = object_forbidden | tower_forbidden
            for _ in range(RAY_PLANNING_ATTEMPTS):
                planned = _plan_ray(
                    shape,
                    tip,
                    direction,
                    reflections,
                    blocked_objects,
                    blocked_paths,
                    combined_forbidden,
                )
                if planned is None:
                    continue
                ray_path, ray_mirrors = planned
                object_cells |= cells
                object_forbidden |= tower_forbidden
                towers.append(
                    {
                        "color": color,
                        "dir": direction,
                        "tip": tip,
                        "body_top_left": (br, bc),
                        "offset": offset,
                    }
                )
                path_cells |= set(ray_path)
                path_by_color[color] = ray_path
                for mirror in ray_mirrors:
                    object_cells |= mirror
                    object_forbidden |= _expand_object_forbidden(mirror, shape)
                    mirror_components.append(mirror)
                tower_built = True
                break
            if tower_built:
                break
        if not tower_built:
            return None

    if not any(len(path_by_color[tower["color"]]) > 0 for tower in towers):  # type: ignore[index]
        return None
    if not any(reflection_counts):
        return None
    if not any(count == 0 for count in reflection_counts):
        return None

    inp = np.full(shape, BG, dtype=int)
    for mirror in mirror_components:
        for r, c in mirror:
            inp[r, c] = MIRROR
    for tower in towers:
        br, bc = tower["body_top_left"]  # type: ignore[index]
        offset = int(tower["offset"])  # type: ignore[index]
        cells, _tip = _tower_cells(int(br), int(bc), str(tower["dir"]), offset)
        for r, c in cells:
            inp[r, c] = int(tower["color"])  # type: ignore[index]

    try:
        parsed_towers = _find_towers(inp)
        parsed_mirrors = _find_mirrors(inp)
    except AssertionError:
        return None
    if len(parsed_towers) != tower_count:
        return None
    if len({frozenset(comp) for comp in mirror_components}) != len(_connected_components(inp == MIRROR)):
        return None
    if not parsed_mirrors and mirror_components:
        return None

    expected = _build_expected(inp, towers, path_by_color)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    actual_reflections = [_count_reflections(path_by_color[int(tower["color"])], tower["tip"]) for tower in towers]  # type: ignore[index]
    if not any(count == 0 for count in actual_reflections):
        return None
    if not any(count > 0 for count in actual_reflections):
        return None
    if max(actual.shape) > 30:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower mirror puzzle")
