from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MIN_SEG_LEN = 2
MIN_FINAL_LEN = 2
MAX_REFLECTIONS = 4
MAX_SOURCES = 2
SCENE_ATTEMPTS = 400
SYSTEM_ATTEMPTS = 50
MIRROR_PATTERNS = tuple(
    (length, start)
    for length in (2, 3, 4)
    for start in range(-length + 1, 1)
)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 8:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel())
    return int(np.argmax(counts))


def _diag_orientation(coords: np.ndarray) -> str | None:
    pts = {(int(r), int(c)) for r, c in coords}
    size = len(pts)
    if not (2 <= size <= 4):
        return None

    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]

    if len({r - c for r, c in pts}) == 1:
        r0 = min(rows)
        c0 = min(cols)
        if pts == {(r0 + k, c0 + k) for k in range(size)}:
            return "\\"

    if len({r + c for r, c in pts}) == 1:
        r0 = min(rows)
        c0 = max(cols)
        if pts == {(r0 + k, c0 - k) for k in range(size)}:
            return "/"

    return None


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int]:
    bg = _dominant_color(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert len(colors) == 2, "expected exactly two non-background colors"

    source_candidates: list[int] = []
    mirror_candidates: list[int] = []

    for color in colors:
        comps = _components(grid == color, connectivity=8)
        if comps and all(len(comp) == 1 for comp in comps):
            source_candidates.append(color)
        if comps and all(_diag_orientation(comp) is not None for comp in comps):
            mirror_candidates.append(color)

    assert len(source_candidates) == 1, "expected one singleton source color"
    assert len(mirror_candidates) == 1, "expected one diagonal mirror color"
    source_color = int(source_candidates[0])
    mirror_color = int(mirror_candidates[0])
    assert source_color != mirror_color
    return bg, source_color, mirror_color


def _reflect(direction: tuple[int, int], orient: str) -> tuple[int, int]:
    dr, dc = direction
    if orient == "\\":
        return int(dc), int(dr)
    return int(-dc), int(-dr)


def _initial_hit(
    grid: np.ndarray,
    source_pos: tuple[int, int],
    bg: int,
    mirror_color: int,
) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    h, w = grid.shape
    sr, sc = source_pos
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for dr, dc in DIRS:
        r = sr + dr
        c = sc + dc
        dist = 1
        while _in_bounds(h, w, r, c):
            value = int(grid[r, c])
            if value == bg:
                r += dr
                c += dc
                dist += 1
                continue
            if value == mirror_color:
                candidates.append((dist, (dr, dc), (r, c)))
            break
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    if len(candidates) >= 2:
        if candidates[0][0] == candidates[1][0]:
            return None
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each light source ray through diagonal mirror segments and draw the reflected path.

    Input structure:
    - The background is a dominant blank color.
    - One non-background color forms isolated single-pixel light sources.
    - The other non-background color forms short 8-connected diagonal mirror segments
      of length 2-4, each lying on either the main diagonal or the anti-diagonal.

    Rule:
    - For each source, find the nearest mirror that is first visible in one of the four
      cardinal directions and cast the incoming ray toward it.
    - When a ray reaches a mirror pixel, keep the mirror unchanged and rotate the ray by
      90 degrees according to that mirror segment's diagonal orientation.
    - Continue the reflected ray segment by segment until it leaves the grid or would loop.
    - Paint every traversed background cell with the source color while preserving all
      original mirror pixels.

    Key insight:
    - The task is a ray-tracing problem on a grid: identify which color is made of singleton
      sources, classify each diagonal mirror component by its slash direction, and then apply
      repeated orthogonal reflections at the first mirror hit on each segment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, source_color, mirror_color = _infer_colors(g)

    mirror_orientation: dict[tuple[int, int], str] = {}
    for comp in _components(g == mirror_color, connectivity=8):
        orient = _diag_orientation(comp)
        assert orient is not None
        for r, c in comp:
            mirror_orientation[(int(r), int(c))] = orient

    out = g.copy()
    sources = np.argwhere(g == source_color)

    for sr, sc in sources:
        sr = int(sr)
        sc = int(sc)
        start = _initial_hit(g, (sr, sc), bg, mirror_color)
        if start is None:
            continue

        direction = start[1]
        anchor = (sr, sc)
        seen_states: set[tuple[int, int, int, int]] = set()

        while True:
            state = (anchor[0], anchor[1], direction[0], direction[1])
            if state in seen_states:
                break
            seen_states.add(state)

            dr, dc = direction
            r, c = anchor
            hit_mirror = False

            while True:
                r += dr
                c += dc
                if not _in_bounds(h, w, r, c):
                    break

                value = int(g[r, c])
                if value == mirror_color:
                    direction = _reflect(direction, mirror_orientation[(r, c)])
                    anchor = (r, c)
                    hit_mirror = True
                    break

                if value == bg or value == source_color:
                    out[r, c] = source_color
                    continue

                break

            if not hit_mirror:
                break

    return out


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _mirror_for_turn(direction: tuple[int, int], next_direction: tuple[int, int]) -> str:
    if next_direction == _reflect(direction, "\\"):
        return "\\"
    assert next_direction == _reflect(direction, "/")
    return "/"


def _mirror_cells(
    center: tuple[int, int],
    orient: str,
    length: int,
    start: int,
) -> list[tuple[int, int]]:
    r, c = center
    cells: list[tuple[int, int]] = []
    for offset in range(start, start + length):
        if orient == "\\":
            cells.append((r + offset, c + offset))
        else:
            cells.append((r + offset, c - offset))
    return cells


def _object_clear(
    cells: list[tuple[int, int]],
    object_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    cand = set(cells)
    for r, c in cells:
        if not _in_bounds(h, w, r, c):
            return False
        if (r, c) in object_cells or (r, c) in path_cells:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in object_cells and (nr, nc) not in cand:
                    return False
    return True


def _extend_system(
    h: int,
    w: int,
    cursor: tuple[int, int],
    direction: tuple[int, int],
    reflections_left: int,
    global_objects: set[tuple[int, int]],
    global_paths: set[tuple[int, int]],
    local_objects: set[tuple[int, int]],
    local_paths: set[tuple[int, int]],
    mirrors: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]],
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]],
] | None:
    blocked = global_objects | global_paths | local_objects | local_paths

    if reflections_left == 0:
        segment: list[tuple[int, int]] = []
        r, c = cursor
        dr, dc = direction
        while True:
            r += dr
            c += dc
            if not _in_bounds(h, w, r, c):
                break
            if (r, c) in blocked:
                return None
            segment.append((r, c))
        if len(segment) < MIN_FINAL_LEN:
            return None
        return local_objects, local_paths | set(segment), mirrors

    distances: list[int] = []
    r, c = cursor
    dr, dc = direction
    step = 0
    while True:
        step += 1
        nr = r + dr * step
        nc = c + dc * step
        if not _in_bounds(h, w, nr, nc):
            break
        if (nr, nc) in blocked:
            break
        if step >= MIN_SEG_LEN and 0 < nr < h - 1 and 0 < nc < w - 1:
            distances.append(step)

    random.shuffle(distances)
    distances = distances[:10]

    for dist in distances:
        center = (r + dr * dist, c + dc * dist)
        segment = [(r + dr * k, c + dc * k) for k in range(1, dist)]
        if any(cell in blocked for cell in segment):
            continue

        next_dirs = [_turn_left(direction), _turn_right(direction)]
        random.shuffle(next_dirs)

        for next_direction in next_dirs:
            orient = _mirror_for_turn(direction, next_direction)
            patterns = list(MIRROR_PATTERNS)
            random.shuffle(patterns)
            patterns = patterns[:5]

            for length, start in patterns:
                mirror_cells = _mirror_cells(center, orient, length, start)
                if not _object_clear(
                    mirror_cells,
                    global_objects | local_objects,
                    global_paths | local_paths,
                    h,
                    w,
                ):
                    continue

                rr = center[0] + next_direction[0]
                cc = center[1] + next_direction[1]
                if not _in_bounds(h, w, rr, cc):
                    continue
                if (rr, cc) in blocked or (rr, cc) in mirror_cells:
                    continue

                result = _extend_system(
                    h=h,
                    w=w,
                    cursor=center,
                    direction=next_direction,
                    reflections_left=reflections_left - 1,
                    global_objects=global_objects,
                    global_paths=global_paths,
                    local_objects=local_objects | set(mirror_cells),
                    local_paths=local_paths | set(segment),
                    mirrors=mirrors + [(center, tuple(mirror_cells))],
                )
                if result is not None:
                    return result

    return None


def _build_system(
    h: int,
    w: int,
    global_objects: set[tuple[int, int]],
    global_paths: set[tuple[int, int]],
) -> dict[str, object] | None:
    for _ in range(SYSTEM_ATTEMPTS):
        start = (random.randint(2, h - 3), random.randint(2, w - 3))
        if not _object_clear([start], global_objects, global_paths, h, w):
            continue

        direction = random.choice(DIRS)
        reflections = random.randint(1, MAX_REFLECTIONS)
        result = _extend_system(
            h=h,
            w=w,
            cursor=start,
            direction=direction,
            reflections_left=reflections,
            global_objects=global_objects,
            global_paths=global_paths,
            local_objects={start},
            local_paths=set(),
            mirrors=[],
        )
        if result is None:
            continue

        object_cells, path_cells, mirrors = result
        if not mirrors:
            continue
        first_center = tuple(int(v) for v in mirrors[0][0])
        return {
            "source": start,
            "direction": direction,
            "reflections": reflections,
            "objects": object_cells,
            "paths": path_cells,
            "mirrors": mirrors,
            "first_center": first_center,
        }

    return None


def _validate_scene(
    grid: np.ndarray,
    source_color: int,
    mirror_color: int,
    systems: list[dict[str, object]],
) -> bool:
    bg = 0
    if len(np.unique(grid)) != 3:
        return False

    try:
        inferred_bg, inferred_source, inferred_mirror = _infer_colors(grid)
    except AssertionError:
        return False

    if (inferred_bg, inferred_source, inferred_mirror) != (bg, source_color, mirror_color):
        return False

    for system in systems:
        source = tuple(system["source"])
        first_center = tuple(system["first_center"])
        start = _initial_hit(grid, source, bg, mirror_color)
        if start is None:
            return False
        if tuple(start[2]) != first_center or tuple(start[1]) != tuple(system["direction"]):
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if int(np.sum(out == source_color)) <= len(systems):
        return False
    if np.any(grid[(grid != bg) & (grid != source_color) & (grid != mirror_color)]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        h = random.randint(16, 26)
        w = random.randint(16, 26)
        bg = 0
        source_color, mirror_color = random.sample([c for c in range(1, 10)], 2)

        grid = np.full((h, w), bg, dtype=int)
        global_objects: set[tuple[int, int]] = set()
        global_paths: set[tuple[int, int]] = set()
        systems: list[dict[str, object]] = []

        source_count = random.randint(1, MAX_SOURCES)
        ok = True
        for _idx in range(source_count):
            system = _build_system(h, w, global_objects, global_paths)
            if system is None:
                ok = False
                break
            systems.append(system)
            global_objects |= set(system["objects"])
            global_paths |= set(system["paths"])

        if not ok or not systems:
            continue

        for system in systems:
            sr, sc = system["source"]
            grid[sr, sc] = source_color
            for _center, cells in system["mirrors"]:
                for r, c in cells:
                    grid[r, c] = mirror_color

        if _validate_scene(grid, source_color, mirror_color, systems):
            return grid

    raise RuntimeError("failed to generate a valid mirror-ray scene")
