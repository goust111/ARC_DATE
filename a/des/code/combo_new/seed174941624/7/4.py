from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


Coord = tuple[int, int]
Dir = tuple[int, int]

DIRS4: tuple[Dir, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _add(a: Coord, b: Dir) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _neg(d: Dir) -> Dir:
    return -d[0], -d[1]


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _turn(d: Dir, sign: int) -> Dir:
    dr, dc = d
    if sign > 0:
        return -dc, dr
    return dc, -dr


def _rotate(cell: Coord, quarter_turns: int) -> Coord:
    r, c = cell
    for _ in range(quarter_turns % 4):
        r, c = c, -r
    return r, c


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                cell = q.popleft()
                coords.append(cell)
                for nr, nc in _neighbors4(cell):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _path_neighbors(cell: Coord, cells: set[Coord]) -> list[Coord]:
    return [nbr for nbr in _neighbors4(cell) if nbr in cells]


def _shortest_path_length(cells: set[Coord], start: Coord, goal: Coord) -> int:
    q: deque[tuple[Coord, int]] = deque([(start, 0)])
    seen = {start}
    while q:
        cell, dist = q.popleft()
        if cell == goal:
            return dist
        for nbr in _path_neighbors(cell, cells):
            if nbr not in seen:
                seen.add(nbr)
                q.append((nbr, dist + 1))
    raise AssertionError("component path is disconnected")


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        cells = []
        c = c0
        while c != c1:
            c += step
            cells.append((r0, c))
        return cells
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        cells = []
        r = r0
        while r != r1:
            r += step
            cells.append((r, c0))
        return cells
    raise AssertionError("segment endpoints must be aligned")


def _ray_steps(start: Coord, end: Coord, direction: Dir) -> int:
    dr, dc = direction
    if dr != 0:
        if start[1] != end[1]:
            raise AssertionError("ray changed columns")
        delta = end[0] - start[0]
        if delta == 0 or delta % dr != 0:
            raise AssertionError("invalid ray direction")
        steps = delta // dr
    else:
        if start[0] != end[0]:
            raise AssertionError("ray changed rows")
        delta = end[1] - start[1]
        if delta == 0 or delta % dc != 0:
            raise AssertionError("invalid ray direction")
        steps = delta // dc
    if steps <= 0:
        raise AssertionError("ray must move forward")
    return steps


def _gap_cells(broken_a: Coord, dir_a: Dir, broken_b: Coord, dir_b: Dir) -> list[Coord]:
    if dir_a == _neg(dir_b):
        _ray_steps(broken_a, broken_b, dir_a)
        cells = _segment_cells(broken_a, broken_b)[:-1]
        assert cells, "gap must contain at least one erased cell"
        first = cells[0]
        assert (first[0] - broken_a[0], first[1] - broken_a[1]) == dir_a
        return cells

    if dir_a[0] == 0 and dir_b[1] == 0:
        corner = (broken_a[0], broken_b[1])
    elif dir_a[1] == 0 and dir_b[0] == 0:
        corner = (broken_b[0], broken_a[1])
    else:
        raise AssertionError("gap endpoints do not define a valid rectilinear repair")

    _ray_steps(broken_a, corner, dir_a)
    _ray_steps(broken_b, corner, dir_b)
    first_leg = _segment_cells(broken_a, corner)
    second_leg = _segment_cells(corner, broken_b)[:-1]
    assert first_leg, "elbow gap needs a first leg"
    cells = first_leg + second_leg
    assert cells, "gap must contain erased cells"
    return cells


def _component_info(comp_coords: list[Coord], marker_map: dict[Coord, int]) -> dict[str, object]:
    cells = set(comp_coords)
    degrees = {cell: len(_path_neighbors(cell, cells)) for cell in cells}
    assert all(deg <= 2 for deg in degrees.values()), "path pieces must be non-branching"
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "each broken path piece must have exactly two ends"

    tagged: list[tuple[Coord, Coord]] = []
    untagged: list[Coord] = []
    for cell in endpoints:
        touching_markers = [nbr for nbr in _neighbors4(cell) if nbr in marker_map]
        if touching_markers:
            assert len(touching_markers) == 1, "endpoint marker must be unique"
            tagged.append((cell, touching_markers[0]))
        else:
            untagged.append(cell)

    assert len(tagged) == 1 and len(untagged) == 1, "one end should carry the marker and one should face the gap"
    endpoint, marker_pos = tagged[0]
    broken_end = untagged[0]
    inward_neighbors = _path_neighbors(broken_end, cells)
    assert len(inward_neighbors) == 1
    inward = inward_neighbors[0]
    extend_dir = (broken_end[0] - inward[0], broken_end[1] - inward[1])

    return {
        "endpoint": endpoint,
        "broken_end": broken_end,
        "marker_color": marker_map[marker_pos],
        "distance": _shortest_path_length(cells, endpoint, broken_end),
        "extend_dir": extend_dir,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each erased path gap with the marker color belonging to the endpoint that lies closer to the break.

    Input structure:
    - The background is black.
    - Each path color forms two disconnected 4-connected orthogonal path pieces because one contiguous run of that path has been erased to background.
    - Every full path has two true endpoints, and each true endpoint is 4-neighbor-adjacent to a single-pixel marker of a different color.

    Rule:
    - Group the visible path pieces by color; each path color contributes exactly two non-branching pieces.
    - On each piece, the endpoint touching a marker is the true path end, while the other degree-1 pixel is the dangling break end.
    - Reconstruct the missing straight or single-elbow rectilinear segment between the two dangling ends by extending their current directions through the background.
    - Paint only the reconstructed gap pixels, using the marker color from the side whose true endpoint is closer to the break along the surviving path piece; keep all original path pixels and markers unchanged.

    Key insight:
    - The missing geometry is determined by the dangling path directions, but the fill color comes from a different object class: whichever endpoint marker sits on the shorter surviving side of the broken polyline.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    marker_map: dict[Coord, int] = {}
    path_groups: dict[int, list[list[Coord]]] = defaultdict(list)
    for comp in comps:
        color = int(comp["color"])
        coords = list(comp["coords"])
        if len(coords) == 1:
            marker_map[coords[0]] = color
        else:
            path_groups[color].append(coords)

    assert marker_map, "expected endpoint markers"
    assert path_groups, "expected path pieces"

    out = g.copy()
    for color, pieces in path_groups.items():
        assert len(pieces) == 2, f"path color {color} should split into exactly two pieces"
        infos = [_component_info(piece, marker_map) for piece in pieces]
        distances = [int(info["distance"]) for info in infos]
        assert distances[0] != distances[1], "the closer endpoint marker must be unambiguous"
        fill_info = infos[0] if distances[0] < distances[1] else infos[1]

        gap = _gap_cells(
            broken_a=infos[0]["broken_end"],
            dir_a=infos[0]["extend_dir"],
            broken_b=infos[1]["broken_end"],
            dir_b=infos[1]["extend_dir"],
        )
        fill_color = int(fill_info["marker_color"])
        for r, c in gap:
            assert int(g[r, c]) == bg, "repaired cells must start as erased background"
            out[r, c] = fill_color
    return out


BG_COLOR = 0
GRID_MIN = 16
GRID_MAX = 26
PATH_COUNT_CHOICES = (1, 2)
GAP_KIND_CHOICES = ("straight", "elbow")
STRAIGHT_GAP_RANGE = (1, 5)
ELBOW_LEG_RANGE = (1, 4)
SHORT_SIDE_RANGE = (1, 4)
LONG_EXTRA_RANGE = (1, 5)
MAX_TAIL_SEGMENTS = 3
LOCAL_BUILD_RETRIES = 150
TAIL_BUILD_RETRIES = 120
PLACEMENT_RETRIES = 160
GENERATE_RETRIES = 400
COLOR_POOL = tuple(range(1, 10))


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _build_gap_spec(kind: str) -> dict[str, object]:
    broken_a = (0, 0)
    if kind == "straight":
        gap_len = random.randint(*STRAIGHT_GAP_RANGE)
        broken_b = (0, gap_len + 1)
        gap_cells = [(0, c) for c in range(1, gap_len + 1)]
        dir_a = (0, 1)
        dir_b = (0, -1)
    else:
        leg_h = random.randint(*ELBOW_LEG_RANGE)
        leg_v = random.randint(*ELBOW_LEG_RANGE)
        broken_b = (leg_v, leg_h)
        gap_cells = [(0, c) for c in range(1, leg_h + 1)] + [(r, leg_h) for r in range(1, leg_v)]
        dir_a = (0, 1)
        dir_b = (-1, 0)
    return {
        "broken_a": broken_a,
        "broken_b": broken_b,
        "dir_a": dir_a,
        "dir_b": dir_b,
        "gap_cells": gap_cells,
    }


def _generate_tail(start: Coord, initial_dir: Dir, total_len: int, existing_path: set[Coord]) -> dict[str, object] | None:
    for _ in range(TAIL_BUILD_RETRIES):
        seg_count = random.randint(1, min(MAX_TAIL_SEGMENTS, total_len))
        lengths = _random_composition(total_len, seg_count)
        dirs = [initial_dir]
        cur_dir = initial_dir
        for _ in range(seg_count - 1):
            cur_dir = _turn(cur_dir, random.choice((-1, 1)))
            dirs.append(cur_dir)

        path_cells = [start]
        path_set = set(existing_path)
        ok = True
        for direction, seg_len in zip(dirs, lengths):
            for _ in range(seg_len):
                prev = path_cells[-1]
                nxt = _add(prev, direction)
                touching = {nbr for nbr in _neighbors4(nxt) if nbr in path_set}
                if nxt in path_set or touching != {prev}:
                    ok = False
                    break
                path_cells.append(nxt)
                path_set.add(nxt)
            if not ok:
                break
        if not ok:
            continue

        endpoint = path_cells[-1]
        marker = _add(endpoint, dirs[-1])
        if marker in path_set:
            continue
        if {nbr for nbr in _neighbors4(marker) if nbr in path_set} != {endpoint}:
            continue
        return {"cells": path_cells, "endpoint": endpoint, "marker": marker}
    return None


def _component_count(cells: set[Coord]) -> int:
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        start = remaining.pop()
        q = deque([start])
        while q:
            cell = q.popleft()
            for nbr in _path_neighbors(cell, cells):
                if nbr in remaining:
                    remaining.remove(nbr)
                    q.append(nbr)
    return count


def _is_simple_path(cells: set[Coord]) -> bool:
    if not cells:
        return False
    degrees = {cell: len(_path_neighbors(cell, cells)) for cell in cells}
    if any(deg > 2 for deg in degrees.values()):
        return False
    if sum(deg == 1 for deg in degrees.values()) != 2:
        return False
    return _component_count(cells) == 1


def _valid_marker(marker: Coord, endpoint: Coord, full_path: set[Coord], other_marker: Coord) -> bool:
    if marker == other_marker:
        return False
    touching_path = {nbr for nbr in _neighbors4(marker) if nbr in full_path}
    if touching_path != {endpoint}:
        return False
    if other_marker in _neighbors4(marker):
        return False
    return True


def _build_local_object() -> dict[str, object] | None:
    for _ in range(LOCAL_BUILD_RETRIES):
        gap_spec = _build_gap_spec(random.choice(GAP_KIND_CHOICES))
        short_side_first = random.choice((True, False))
        short_len = random.randint(*SHORT_SIDE_RANGE)
        long_len = short_len + random.randint(*LONG_EXTRA_RANGE)
        len_a = short_len if short_side_first else long_len
        len_b = long_len if short_side_first else short_len

        broken_a = gap_spec["broken_a"]
        broken_b = gap_spec["broken_b"]
        dir_a = gap_spec["dir_a"]
        dir_b = gap_spec["dir_b"]
        gap_cells = list(gap_spec["gap_cells"])
        base_path = {broken_a, broken_b, *gap_cells}

        tail_a = _generate_tail(broken_a, _neg(dir_a), len_a, base_path)
        if tail_a is None:
            continue
        path_after_a = base_path | set(tail_a["cells"])
        tail_b = _generate_tail(broken_b, _neg(dir_b), len_b, path_after_a)
        if tail_b is None:
            continue

        visible_path = set(tail_a["cells"]) | set(tail_b["cells"])
        full_path = visible_path | set(gap_cells)
        if not _is_simple_path(full_path):
            continue
        if _component_count(visible_path) != 2:
            continue

        marker_a = tail_a["marker"]
        marker_b = tail_b["marker"]
        if not _valid_marker(marker_a, tail_a["endpoint"], full_path, marker_b):
            continue
        if not _valid_marker(marker_b, tail_b["endpoint"], full_path, marker_a):
            continue

        return {
            "visible_path": visible_path,
            "full_path": full_path,
            "gap_cells": set(gap_cells),
            "marker_a": marker_a,
            "marker_b": marker_b,
            "fill_side": "a" if len_a < len_b else "b",
        }
    return None


def _transform_object(spec: dict[str, object], rotation: int) -> dict[str, object]:
    return {
        "visible_path": {_rotate(cell, rotation) for cell in spec["visible_path"]},
        "full_path": {_rotate(cell, rotation) for cell in spec["full_path"]},
        "gap_cells": {_rotate(cell, rotation) for cell in spec["gap_cells"]},
        "marker_a": _rotate(spec["marker_a"], rotation),
        "marker_b": _rotate(spec["marker_b"], rotation),
        "fill_side": spec["fill_side"],
    }


def _inflate(cells: set[Coord], h: int, w: int) -> set[Coord]:
    expanded: set[Coord] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _shift_cells(cells: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def _shift_cell(cell: Coord, dr: int, dc: int) -> Coord:
    return cell[0] + dr, cell[1] + dc


def _place_object(
    grid: np.ndarray,
    forbidden: set[Coord],
    local_spec: dict[str, object],
    path_color: int,
    marker_color_a: int,
    marker_color_b: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    rotations = [0, 1, 2, 3]
    random.shuffle(rotations)
    for rotation in rotations:
        spec = _transform_object(local_spec, rotation)
        reserved = set(spec["full_path"]) | {spec["marker_a"], spec["marker_b"]}
        rows = [r for r, _ in reserved]
        cols = [c for _, c in reserved]
        min_dr = -min(rows)
        max_dr = h - 1 - max(rows)
        min_dc = -min(cols)
        max_dc = w - 1 - max(cols)
        if min_dr > max_dr or min_dc > max_dc:
            continue

        for _ in range(PLACEMENT_RETRIES):
            shift_r = random.randint(min_dr, max_dr)
            shift_c = random.randint(min_dc, max_dc)
            placed_reserved = _shift_cells(reserved, shift_r, shift_c)
            if placed_reserved & forbidden:
                continue

            placed_visible = _shift_cells(set(spec["visible_path"]), shift_r, shift_c)
            placed_gap = _shift_cells(set(spec["gap_cells"]), shift_r, shift_c)
            marker_a = _shift_cell(spec["marker_a"], shift_r, shift_c)
            marker_b = _shift_cell(spec["marker_b"], shift_r, shift_c)
            fill_color = marker_color_a if spec["fill_side"] == "a" else marker_color_b

            for r, c in placed_visible:
                grid[r, c] = path_color
            grid[marker_a] = marker_color_a
            grid[marker_b] = marker_color_b

            return {
                "forbidden": forbidden | _inflate(placed_reserved, h, w),
                "gap_cells": placed_gap,
                "fill_color": fill_color,
            }
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    path_count = random.choice(PATH_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, 3 * path_count)

    forbidden: set[Coord] = set()
    placed_specs: list[dict[str, object]] = []
    for i in range(path_count):
        local_spec = _build_local_object()
        if local_spec is None:
            return None
        placement = _place_object(
            grid=grid,
            forbidden=forbidden,
            local_spec=local_spec,
            path_color=colors[3 * i],
            marker_color_a=colors[3 * i + 1],
            marker_color_b=colors[3 * i + 2],
        )
        if placement is None:
            return None
        forbidden = placement["forbidden"]
        placed_specs.append(placement)

    expected = grid.copy()
    for spec in placed_specs:
        for r, c in spec["gap_cells"]:
            expected[r, c] = int(spec["fill_color"])

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid broken-path color-transfer puzzle")
