from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400

Coord = tuple[int, int]
Center = tuple[float, float]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[Coord] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != BLACK)]
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_point(r: int, c: int, center: Center, turns: int = 1) -> Coord:
    cr, cc = center
    dr = r - cr
    dc = c - cc
    turns %= 4
    for _ in range(turns):
        dr, dc = dc, -dr
    rr = int(round(cr + dr))
    cc2 = int(round(cc + dc))
    return rr, cc2


def _orbit(r: int, c: int, center: Center) -> tuple[Coord, ...]:
    pts = []
    cur = (r, c)
    for _ in range(4):
        if cur not in pts:
            pts.append(cur)
        cur = _rotate_point(cur[0], cur[1], center, 1)
    return tuple(pts)


def _singleton_markers(grid: np.ndarray) -> list[Coord]:
    comps = _components(grid == GREY)
    markers: list[Coord] = []
    for comp in comps:
        if len(comp) == 1:
            markers.append(comp[0])
        else:
            raise AssertionError("grey cells must be isolated singleton markers")
    markers.sort()
    return markers


def _object_cells(grid: np.ndarray, markers: list[Coord]) -> list[Coord]:
    marker_set = set(markers)
    coords: list[Coord] = []
    for r, c in np.argwhere(grid != BLACK):
        rc = (int(r), int(c))
        if rc not in marker_set:
            coords.append(rc)
    return coords


def _candidate_centers(shape: tuple[int, int]) -> list[Center]:
    h, w = shape
    return [(r + z0, c + z1) for r in range(h) for c in range(w) for z0 in (0.0, 0.5) for z1 in (0.0, 0.5)]


def _analyze_center(grid: np.ndarray, center: Center, markers: list[Coord]) -> dict | None:
    marker_set = set(markers)
    obj = _object_cells(grid, markers)
    if not obj:
        return None

    completion: dict[Coord, int] = {}
    support_hits = 0
    for r, c in obj:
        color = int(grid[r, c])
        for turns in (1, 2, 3):
            rr, cc = _rotate_point(r, c, center, turns)
            if not _in_bounds(grid.shape, rr, cc):
                return None
            if (rr, cc) in marker_set:
                return None
            target = int(grid[rr, cc])
            if target == BLACK:
                prev = completion.get((rr, cc))
                if prev is not None and prev != color:
                    return None
                completion[(rr, cc)] = color
            elif target != color:
                return None
            else:
                support_hits += 1

    if not completion:
        return None

    completed = grid.copy()
    for (r, c), color in completion.items():
        completed[r, c] = color
    obj_mask = completed != BLACK
    for r, c in markers:
        obj_mask[r, c] = False
    if len(_components(obj_mask)) != 1:
        return None

    coords = list(completion)
    r0, c0, r1, c1 = _bbox(coords)
    stamp = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in completion.items():
        stamp[r - r0, c - c0] = color
    return {
        "center": center,
        "completion": completion,
        "completed": completed,
        "stamp": stamp,
        "score": (len(completion), -support_hits, stamp.shape[0] * stamp.shape[1]),
    }


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Coord], Center]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    markers = _singleton_markers(g)
    assert markers, "expected at least one grey marker"

    analyses = []
    for center in _candidate_centers(g.shape):
        info = _analyze_center(g, center, markers)
        if info is not None:
            analyses.append(info)

    assert analyses, "no valid rotational center found"
    analyses.sort(key=lambda item: item["score"])
    best = analyses[0]
    best_score = best["score"]
    ties = [item for item in analyses if item["score"] == best_score]
    assert len(ties) == 1, "rotational center is ambiguous"

    out = best["completed"].copy()
    stamp = best["stamp"]
    for r0, c0 in markers:
        for dr, dc in np.argwhere(stamp != BLACK):
            rr = int(r0 + dr)
            cc = int(c0 + dc)
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = int(stamp[dr, dc])
    return out, stamp, markers, best["center"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the rotationally completed missing pixels as a hidden stamp and print it at every grey marker.

    Input structure:
    - Black background with one larger colored object that would be 90-degree rotationally symmetric when complete.
    - Some cells of that object are missing and appear as black, while all visible object cells already agree with the same rotation center.
    - Additional grey singleton pixels act as isolated stamp markers far from the object.

    Rule:
    - Find the object's 90-degree rotation center and rotate every visible object pixel through the other three quarter-turn positions.
    - Whenever a rotated counterpart lands on black, fill that cell with the source color to complete the object.
    - Take exactly the newly added colored cells, crop them to their own bounding box as a stamp, and place that stamp with each grey marker as the stamp's top-left corner, clipping past the grid edge.

    Key insight:
    - The stamp is not drawn anywhere in the input; it is the color-preserving difference between the incomplete object and its rotationally completed form.
    """
    out, _stamp, _markers, _center = _solve_core(grid)
    return out


def _center_for_size(size: int) -> Center:
    mid = (size - 1) / 2.0
    return mid, mid


def _add_orbit(sprite: np.ndarray, center: Center, cell: Coord, color: int) -> bool:
    orb = _orbit(cell[0], cell[1], center)
    for r, c in orb:
        if not _in_bounds(sprite.shape, r, c):
            return False
        if sprite[r, c] != BLACK:
            return False
    for r, c in orb:
        sprite[r, c] = color
    return True


def _grow_symmetric_object(size: int) -> np.ndarray | None:
    center = _center_for_size(size)
    sprite = np.zeros((size, size), dtype=int)
    palette = random.sample(OBJECT_COLORS, k=random.randint(2, min(4, len(OBJECT_COLORS))))
    if size % 2 == 1:
        sprite[size // 2, size // 2] = random.choice(palette)
    else:
        if not _add_orbit(sprite, center, (size // 2 - 1, size // 2 - 1), random.choice(palette)):
            return None

    min_orbits = 6 if size >= 9 else 5
    max_orbits = min((size * size) // 6, 12)
    target_orbits = random.randint(min_orbits, max_orbits)
    placed_orbits = 1
    stalled = 0
    while placed_orbits < target_orbits and stalled < 50:
        frontier: list[Coord] = []
        mask = sprite != BLACK
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if _in_bounds(sprite.shape, nr, nc) and sprite[nr, nc] == BLACK:
                    frontier.append((nr, nc))
        random.shuffle(frontier)
        added = False
        for cell in frontier:
            color = random.choice(palette)
            if _add_orbit(sprite, center, cell, color):
                placed_orbits += 1
                added = True
                break
        if added:
            stalled = 0
        else:
            stalled += 1

    if len(_components(sprite != BLACK)) != 1:
        return None
    coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite != BLACK)]
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
        return None
    return sprite


def _quadrant(cell: Coord, center: Center) -> int | None:
    dr = cell[0] - center[0]
    dc = cell[1] - center[1]
    if abs(dr) < 1e-9 or abs(dc) < 1e-9:
        return None
    if dr < 0 and dc < 0:
        return 0
    if dr < 0 and dc > 0:
        return 1
    if dr > 0 and dc > 0:
        return 2
    return 3


def _distance_from_center(cell: Coord, center: Center) -> float:
    return abs(cell[0] - center[0]) + abs(cell[1] - center[1])


def _sample_missing_cluster(sprite: np.ndarray, center: Center) -> list[Coord] | None:
    obj_mask = sprite != BLACK
    quadrant_cells: dict[int, set[Coord]] = {0: set(), 1: set(), 2: set(), 3: set()}
    for r, c in np.argwhere(obj_mask):
        q = _quadrant((int(r), int(c)), center)
        if q is not None:
            quadrant_cells[q].add((int(r), int(c)))

    quadrants = [q for q, cells in quadrant_cells.items() if len(cells) >= 4]
    random.shuffle(quadrants)
    for q in quadrants:
        qcells = quadrant_cells[q]
        ordered = sorted(qcells, key=lambda cell: _distance_from_center(cell, center), reverse=True)
        starts = ordered[: max(2, len(ordered) // 2)]
        for _ in range(60):
            start = random.choice(starts)
            target = random.randint(3, min(6, len(qcells)))
            cluster = [start]
            cluster_set = {start}
            while len(cluster) < target:
                frontier = []
                for cell in list(cluster_set):
                    for nxt in _neighbors4(*cell):
                        if nxt in qcells and nxt not in cluster_set:
                            frontier.append(nxt)
                if not frontier:
                    break
                random.shuffle(frontier)
                expanded = False
                for nxt in frontier:
                    trial = cluster + [nxt]
                    r0, c0, r1, c1 = _bbox(trial)
                    if (r1 - r0 + 1) <= 5 and (c1 - c0 + 1) <= 5:
                        cluster.append(nxt)
                        cluster_set.add(nxt)
                        expanded = True
                        break
                if not expanded:
                    break
            if len(cluster) < 3:
                continue
            r0, c0, r1, c1 = _bbox(cluster)
            if (r1 - r0 + 1) == 1 or (c1 - c0 + 1) == 1:
                continue
            partial = sprite.copy()
            for r, c in cluster:
                partial[r, c] = BLACK
            if len(_components(partial != BLACK)) != 1:
                continue
            return cluster
    return None


def _place_stamp(grid: np.ndarray, stamp: np.ndarray, anchor: Coord) -> None:
    r0, c0 = anchor
    for dr, dc in np.argwhere(stamp != BLACK):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if _in_bounds(grid.shape, rr, cc):
            grid[rr, cc] = int(stamp[dr, dc])


def _fits_reserved(reserved: np.ndarray, anchor: Coord, stamp_shape: tuple[int, int], pad: int = 1) -> bool:
    h, w = reserved.shape
    sh, sw = stamp_shape
    r0, c0 = anchor
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + sh + pad)
    cc1 = min(w, c0 + sw + pad)
    return not reserved[rr0:rr1, cc0:cc1].any()


def _reserve(reserved: np.ndarray, anchor: Coord, stamp_shape: tuple[int, int], pad: int = 1) -> None:
    h, w = reserved.shape
    sh, sw = stamp_shape
    r0, c0 = anchor
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + sh + pad)
    cc1 = min(w, c0 + sw + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _place_markers(
    grid_shape: tuple[int, int],
    object_bbox: tuple[int, int, int, int],
    stamp_shape: tuple[int, int],
) -> list[Coord] | None:
    h, w = grid_shape
    sh, sw = stamp_shape
    reserved = np.zeros(grid_shape, dtype=bool)
    r0, c0, r1, c1 = object_bbox
    rr0 = max(0, r0 - 2)
    cc0 = max(0, c0 - 2)
    rr1 = min(h, r1 + 3)
    cc1 = min(w, c1 + 3)
    reserved[rr0:rr1, cc0:cc1] = True

    regular = [(r, c) for r in range(h) for c in range(w) if _fits_reserved(reserved, (r, c), stamp_shape)]
    clipped = [
        (r, c)
        for r, c in regular
        if r + sh > h or c + sw > w
    ]
    if not regular:
        return None

    markers: list[Coord] = []
    need_clipped = bool(clipped) and random.random() < 0.55
    count = random.randint(1, 4)

    if need_clipped:
        pos = random.choice(clipped)
        markers.append(pos)
        _reserve(reserved, pos, stamp_shape)

    while len(markers) < count:
        choices = [(r, c) for r in range(h) for c in range(w) if _fits_reserved(reserved, (r, c), stamp_shape)]
        if not choices:
            break
        pos = random.choice(choices)
        markers.append(pos)
        _reserve(reserved, pos, stamp_shape)

    return markers if markers else None


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    size = random.randint(7, 12)
    sprite = _grow_symmetric_object(size)
    if sprite is None:
        return None
    center = _center_for_size(size)
    cluster = _sample_missing_cluster(sprite, center)
    if cluster is None:
        return None

    partial = sprite.copy()
    for r, c in cluster:
        partial[r, c] = BLACK

    stamp = _crop_nonblack(np.where(partial != sprite, sprite, BLACK))
    sh, sw = stamp.shape
    if sh > 6 or sw > 6:
        return None
    if np.count_nonzero(stamp) < 3:
        return None

    top_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    bottom_margin = random.randint(max(sh + 1, 4), 8)
    right_margin = random.randint(max(sw + 1, 4), 8)
    h = size + top_margin + bottom_margin
    w = size + left_margin + right_margin
    if h > 30 or w > 30:
        return None

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    r0 = top_margin
    c0 = left_margin
    inp[r0 : r0 + size, c0 : c0 + size] = partial
    out[r0 : r0 + size, c0 : c0 + size] = sprite
    object_bbox = (r0, c0, r0 + size - 1, c0 + size - 1)

    markers = _place_markers(inp.shape, object_bbox, stamp.shape)
    if markers is None or len(markers) == 0:
        return None
    for r, c in markers:
        inp[r, c] = GREY
        _place_stamp(out, stamp, (r, c))

    if len(_components((inp != BLACK) & (inp != GREY))) != 1:
        return None

    solved, solved_stamp, found_markers, found_center = _solve_core(inp)
    if not np.array_equal(solved, out):
        return None
    if not np.array_equal(solved_stamp, stamp):
        return None
    if sorted(found_markers) != sorted(markers):
        return None
    if found_center != (center[0] + r0, center[1] + c0):
        return None

    analyses = []
    for cand in _candidate_centers(inp.shape):
        info = _analyze_center(inp, cand, found_markers)
        if info is not None:
            analyses.append(info)
    if len([a for a in analyses if a["score"] == analyses[0]["score"]]) != 1:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid rotational-stamp ARC input")
