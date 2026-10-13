from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
START_COLOR = 3
END_COLOR = 4
PATH_COLOR = 5

DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_GEN_ATTEMPTS = 500
_RECENT_INPUT_SIGS: set[tuple[bytes, tuple[int, int]]] = set()
_RECENT_OUTPUT_SIGS: set[tuple[bytes, tuple[int, int]]] = set()
_GEN_CALL_COUNT = 0


def _sgn(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_of_coords(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_paths(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g == PATH_COLOR)
    assert 2 <= len(comps) <= 4, "expected 2-4 gray path components"

    paths: list[dict] = []
    for comp in comps:
        comp_set = set(comp)
        starts: set[tuple[int, int]] = set()
        ends: set[tuple[int, int]] = set()
        other_colors: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(g.shape, nr, nc):
                    continue
                val = int(g[nr, nc])
                if val == START_COLOR:
                    starts.add((nr, nc))
                elif val == END_COLOR:
                    ends.add((nr, nc))
                elif val not in (BG, PATH_COLOR):
                    other_colors.add(val)
        assert not other_colors, "unexpected colored cells adjacent to gray path"
        assert len(starts) == 1 and len(ends) == 1, "each gray path needs one green and one yellow endpoint"
        start = next(iter(starts))
        end = next(iter(ends))
        assert sum((nr, nc) in comp_set for nr, nc in _neighbors4(*start)) == 1
        assert sum((nr, nc) in comp_set for nr, nc in _neighbors4(*end)) == 1
        dr = int(end[0] - start[0])
        dc = int(end[1] - start[1])
        direction = (_sgn(dr), _sgn(dc))
        assert direction != (0, 0)

        degs = []
        for r, c in comp:
            deg = sum((nr, nc) in comp_set for nr, nc in _neighbors4(r, c))
            degs.append(deg)
            assert deg <= 2, "gray path must be a width-1 chain"
        if len(comp) > 1:
            assert degs.count(1) == 2 and degs.count(2) == len(comp) - 2

        paths.append(
            {
                "gray": sorted(comp),
                "gray_set": comp_set,
                "start": start,
                "end": end,
                "direction": direction,
            }
        )

    starts = [item["start"] for item in paths]
    assert len(starts) == len(set(starts)), "start endpoints must be unique"
    dirs = [item["direction"] for item in paths]
    assert len(dirs) == len(set(dirs)), "path directions must be unique"
    return sorted(paths, key=lambda item: (item["start"][0], item["start"][1]))


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = direction
    out: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _render_output(shape: tuple[int, int], paths: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    colored: set[tuple[int, int]] = set()
    for item in paths:
        for cell in _ray_cells(shape, item["start"], item["direction"]):
            out[cell] = START_COLOR
            colored.add(cell)
    assert colored
    return _crop_bbox(out, _bbox_of_coords(colored))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each gray path's endpoint displacement into a ray direction, draw the start-colored rays, then crop their joint extent.

    Input structure:
    - The background is black.
    - There are 2-4 separate gray 4-connected width-1 paths.
    - Each gray path has exactly one green endpoint pixel and one yellow endpoint pixel attached to its two ends.

    Rule:
    - For each gray path, read only the green/yellow endpoint coordinates; ignore the path's bends.
    - Convert the displacement from the green endpoint to the yellow endpoint into a unit compass direction by taking the sign of the row and column differences.
    - Starting from the green endpoint, draw a green ray in that direction until the grid edge, then erase every gray path and yellow endpoint implicitly by keeping only those rays.
    - Crop the union of all kept non-background pixels to its minimal bounding box.

    Key insight:
    - The gray path shape is deliberate distraction: only the two colored endpoint positions matter, because their net displacement encodes the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    paths = _extract_paths(g)
    return _render_output(g.shape, paths)


def _path_from_waypoints(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if not waypoints:
        return None
    path = [waypoints[0]]
    r, c = waypoints[0]
    for nr, nc in waypoints[1:]:
        if r != nr and c != nc:
            return None
        while r != nr:
            r += _sgn(nr - r)
            path.append((r, c))
        while c != nc:
            c += _sgn(nc - c)
            path.append((r, c))
    return path


def _count_turns(path: list[tuple[int, int]]) -> int:
    dirs: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((_sgn(r1 - r0), _sgn(c1 - c0)))
    return sum(a != b for a, b in zip(dirs, dirs[1:]))


def _chain_is_simple(path: list[tuple[int, int]]) -> bool:
    if path is None or len(path) < 2 or len(set(path)) != len(path):
        return False
    pos = {cell: i for i, cell in enumerate(path)}
    for i, (r, c) in enumerate(path):
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pos:
                deg += 1
                if abs(pos[(nr, nc)] - i) != 1:
                    return False
        if i in (0, len(path) - 1):
            if deg != 1:
                return False
        elif deg != 2:
            return False
        for nr, nc in _neighbors8(r, c):
            j = pos.get((nr, nc))
            if j is None or j == i:
                continue
            # Allow the diagonal touch induced by one ordinary 90-degree bend
            # (indices differ by 2), but reject more distant self-contacts.
            if max(abs(nr - r), abs(nc - c)) == 1 and abs(j - i) > 2:
                return False
    return True


def _candidate_gray_paths(
    shape: tuple[int, int],
    first_gray: tuple[int, int],
    last_gray: tuple[int, int],
    *,
    require_bent: bool,
) -> list[list[tuple[int, int]]]:
    h, w = shape
    r0, c0 = first_gray
    r1, c1 = last_gray

    candidates: list[list[tuple[int, int]]] = []

    def add(waypoints: list[tuple[int, int]]) -> None:
        path = _path_from_waypoints(waypoints)
        if path is None:
            return
        if path[0] != first_gray or path[-1] != last_gray:
            return
        if not all(_in_bounds(shape, r, c) for r, c in path):
            return
        if not _chain_is_simple(path):
            return
        turns = _count_turns(path)
        if require_bent and turns == 0:
            return
        candidates.append(path)

    add([first_gray, last_gray])
    add([first_gray, (r0, c1), last_gray])
    add([first_gray, (r1, c0), last_gray])

    for off in (1, 2, 3):
        for rr in (min(r0, r1) - off, max(r0, r1) + off):
            if 0 <= rr < h:
                add([first_gray, (rr, c0), (rr, c1), last_gray])
        for cc in (min(c0, c1) - off, max(c0, c1) + off):
            if 0 <= cc < w:
                add([first_gray, (r0, cc), (r1, cc), last_gray])

    for off_r in (1, 2):
        for off_c in (1, 2):
            rr_choices = [min(r0, r1) - off_r, max(r0, r1) + off_r]
            cc_choices = [min(c0, c1) - off_c, max(c0, c1) + off_c]
            for rr in rr_choices:
                for cc in cc_choices:
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    add([first_gray, (rr, c0), (rr, cc), (r1, cc), last_gray])
                    add([first_gray, (r0, cc), (rr, cc), (rr, c1), last_gray])

    uniq: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for path in candidates:
        uniq.setdefault(tuple(path), path)
    paths = list(uniq.values())
    paths.sort(key=lambda p: (_count_turns(p), len(p)))
    return paths


def _mislead_step_options(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    if dc == 0:
        return [(0, -1), (0, 1)]
    return [(dr, 0), (0, dc), (-dr, 0), (0, -dc)]


def _choose_detour_path(
    shape: tuple[int, int],
    first_gray: tuple[int, int],
    last_gray: tuple[int, int],
    *,
    require_bent: bool,
) -> list[tuple[int, int]] | None:
    h, w = shape
    r0, c0 = first_gray
    r1, c1 = last_gray

    waypoint_lists: list[list[tuple[int, int]]] = [
        [first_gray, (r0, c1), last_gray],
        [first_gray, (r1, c0), last_gray],
    ]

    for off in (1, 2):
        for rr in (min(r0, r1) - off, max(r0, r1) + off):
            if 0 <= rr < h:
                waypoint_lists.append([first_gray, (rr, c0), (rr, c1), last_gray])
        for cc in (min(c0, c1) - off, max(c0, c1) + off):
            if 0 <= cc < w:
                waypoint_lists.append([first_gray, (r0, cc), (r1, cc), last_gray])

    random.shuffle(waypoint_lists)
    for waypoints in waypoint_lists:
        path = _path_from_waypoints(waypoints)
        if path is None or not all(_in_bounds(shape, r, c) for r, c in path):
            continue
        if not _chain_is_simple(path):
            continue
        if require_bent and _count_turns(path) == 0:
            continue
        return path
    return None


def _steps_to_edge(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = start
    dr, dc = direction
    steps = 0
    while 0 <= r + dr < h and 0 <= c + dc < w:
        r += dr
        c += dc
        steps += 1
    return steps


def _sample_delta(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    if dr == 0:
        return 0, dc * random.randint(3, 8)
    if dc == 0:
        return dr * random.randint(3, 8), 0
    for _ in range(20):
        a = random.randint(2, 7)
        b = random.randint(2, 7)
        if a != b or random.random() < 0.25:
            return dr * a, dc * b
    return dr * 2, dc * 3


def _touches_existing(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in occupied:
            return True
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _endpoint_neighbor_options(
    shape: tuple[int, int],
    point: tuple[int, int],
    *,
    forbid: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    opts = []
    for dr, dc in CARDINALS:
        if forbid is not None and (dr, dc) == forbid:
            continue
        nr, nc = point[0] + dr, point[1] + dc
        if _in_bounds(shape, nr, nc):
            opts.append((nr, nc))
    random.shuffle(opts)
    return opts


def _scene_family() -> str:
    global _GEN_CALL_COUNT
    idx = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    if idx % 3 == 0:
        return "mixed"
    if idx % 3 == 1:
        return "axis"
    return "diag"


def _sample_directions(n_paths: int, family: str) -> list[tuple[int, int]]:
    axis = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if family == "axis":
        return random.sample(axis, n_paths)
    if family == "diag":
        return random.sample(diag, n_paths)
    need_axis = min(2, n_paths - 1)
    need_diag = 1
    chosen = random.sample(axis, need_axis) + random.sample(diag, need_diag)
    remain_pool = [d for d in DIRECTIONS if d not in chosen]
    if len(chosen) < n_paths:
        chosen += random.sample(remain_pool, n_paths - len(chosen))
    random.shuffle(chosen)
    return chosen


def _preferred_start_positions(
    shape: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction

    if dr < 0:
        r0, r1 = max(2, h // 2), h - 3
    elif dr > 0:
        r0, r1 = 2, min(h // 2, h - 3)
    else:
        r0, r1 = 2, h - 3

    if dc < 0:
        c0, c1 = max(2, w // 2), w - 3
    elif dc > 0:
        c0, c1 = 2, min(w // 2, w - 3)
    else:
        c0, c1 = 2, w - 3

    if r0 > r1 or c0 > c1:
        return []
    points = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(points)
    return points


def _build_one_path(
    shape: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_rays: set[tuple[int, int]],
    *,
    force_mislead: bool,
) -> dict | None:
    h, w = shape
    starts = _preferred_start_positions(shape, direction)
    if not starts:
        return None
    for _ in range(10):
        delta = _sample_delta(direction)
        for start in starts[:50]:
            sr, sc = start
            end = (sr + delta[0], sc + delta[1])
            if not (1 <= end[0] < h - 1 and 1 <= end[1] < w - 1):
                continue
            if _steps_to_edge(shape, start, direction) < 2:
                continue
            ray = set(_ray_cells(shape, start, direction))
            if ray & occupied or ray & reserved_rays:
                continue

            start_steps = _mislead_step_options(direction) if force_mislead else _mislead_step_options(direction) + [d for d in CARDINALS if d not in _mislead_step_options(direction)]
            end_steps = _mislead_step_options(direction)
            random.shuffle(start_steps)
            random.shuffle(end_steps)
            for sdr, sdc in start_steps:
                first_gray = (sr + sdr, sc + sdc)
                if not _in_bounds(shape, *first_gray):
                    continue
                if force_mislead and (sdr, sdc) == direction:
                    continue
                for edr, edc in end_steps:
                    last_gray = (end[0] + edr, end[1] + edc)
                    if not _in_bounds(shape, *last_gray):
                        continue
                    require_bent = force_mislead or random.random() < 0.8
                    path = _choose_detour_path(shape, first_gray, last_gray, require_bent=require_bent)
                    if path is None:
                        continue
                    gray = set(path)
                    cells = gray | {start, end}
                    if _touches_existing(cells, occupied):
                        continue
                    if gray & ray:
                        continue
                    if end in ray:
                        continue
                    if any(cell in reserved_rays for cell in cells):
                        continue

                    start_touch = sum((nr, nc) in gray for nr, nc in _neighbors4(*start))
                    end_touch = sum((nr, nc) in gray for nr, nc in _neighbors4(*end))
                    if start_touch != 1 or end_touch != 1:
                        continue

                    return {
                        "start": start,
                        "end": end,
                        "direction": direction,
                        "gray": sorted(gray),
                        "gray_set": gray,
                        "ray": ray,
                        "turns": _count_turns(path),
                        "first_step": (first_gray[0] - sr, first_gray[1] - sc),
                        "delta": delta,
                    }
    return None


def _place_scene() -> tuple[np.ndarray, list[dict]] | None:
    family = _scene_family()
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    if family == "mixed":
        n_paths = random.choice([3, 4])
    elif family == "axis":
        n_paths = random.choice([2, 3, 4])
    else:
        n_paths = random.choice([2, 3, 4])
    directions = _sample_directions(n_paths, family)

    occupied: set[tuple[int, int]] = set()
    reserved_rays: set[tuple[int, int]] = set()
    paths: list[dict] = []

    forced_index = random.randrange(n_paths)
    for i, direction in enumerate(directions):
        item = _build_one_path(
            (h, w),
            direction,
            occupied,
            reserved_rays,
            force_mislead=(i == forced_index or direction not in CARDINALS),
        )
        if item is None:
            return None
        paths.append(item)
        occupied |= item["gray_set"] | {item["start"], item["end"]}
        reserved_rays |= item["ray"]

    grid = np.zeros((h, w), dtype=int)
    for item in paths:
        for r, c in item["gray"]:
            grid[r, c] = PATH_COLOR
        sr, sc = item["start"]
        er, ec = item["end"]
        grid[sr, sc] = START_COLOR
        grid[er, ec] = END_COLOR
    return grid, paths


def _validate_scene(grid: np.ndarray, meta: list[dict]) -> bool:
    try:
        parsed = _extract_paths(grid)
    except AssertionError:
        return False

    if len(parsed) != len(meta):
        return False

    meta_by_start = {item["start"]: item for item in meta}
    seen_rays: set[tuple[int, int]] = set()
    any_bent = False
    any_axis = False
    any_diag_rect = False
    for item in parsed:
        m = meta_by_start.get(item["start"])
        if m is None:
            return False
        if item["end"] != m["end"] or item["direction"] != m["direction"]:
            return False

        ray_list = _ray_cells(grid.shape, item["start"], item["direction"])
        ray = set(ray_list)
        if ray & seen_rays:
            return False
        seen_rays |= ray

        for cell in ray_list[1:]:
            if int(grid[cell]) != BG:
                return False
        if item["direction"][0] == 0 or item["direction"][1] == 0:
            any_axis = True
        else:
            if abs(m["delta"][0]) != abs(m["delta"][1]):
                any_diag_rect = True
        if m["turns"] > 0 and m["first_step"] != m["direction"]:
            any_bent = True

    if not any_bent:
        return False
    if any(not _steps_to_edge(grid.shape, item["start"], item["direction"]) >= 2 for item in parsed):
        return False
    if any_axis and not any_bent:
        return False
    if any(item["direction"][0] != 0 and item["direction"][1] != 0 for item in parsed) and not any_diag_rect:
        return False

    expected = _render_output(grid.shape, meta)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _place_scene()
        if built is None:
            continue
        grid, meta = built
        if not _validate_scene(grid, meta):
            continue

        out = main(grid)
        in_sig = (grid.tobytes(), tuple(map(int, grid.shape)))
        out_sig = (out.tobytes(), tuple(map(int, out.shape)))
        if in_sig in _RECENT_INPUT_SIGS or out_sig in _RECENT_OUTPUT_SIGS:
            continue
        _RECENT_INPUT_SIGS.add(in_sig)
        _RECENT_OUTPUT_SIGS.add(out_sig)
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
