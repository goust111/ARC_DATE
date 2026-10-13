from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
DIRS: tuple[Coord, ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))
COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _pad_bbox(box: BBox, shape: tuple[int, int], pad: int = 1) -> BBox:
    h, w = shape
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_checker(patch: np.ndarray, bg: int) -> bool:
    vals = set(map(int, patch.ravel()))
    if bg in vals or len(vals) != 2:
        return False
    a = int(patch[0, 0])
    b = int(patch[0, 1])
    if a == b:
        return False
    for r in range(3):
        for c in range(3):
            expected = a if (r + c) % 2 == 0 else b
            if int(patch[r, c]) != expected:
                return False
    return True


def _is_ring(patch: np.ndarray, bg: int) -> bool:
    if int(patch[1, 1]) != int(bg):
        return False
    border = np.concatenate((patch[0, :], patch[2, :], patch[1:2, 0], patch[1:2, 2]))
    vals = set(map(int, border.ravel()))
    return len(vals) == 1 and int(next(iter(vals))) != int(bg)


def _find_source_and_rings(grid: np.ndarray, bg: int) -> tuple[dict, list[dict]]:
    h, w = grid.shape
    sources: list[dict] = []
    rings: list[dict] = []
    for r in range(h - 2):
        for c in range(w - 2):
            patch = grid[r : r + 3, c : c + 3]
            box = (r, c, r + 2, c + 2)
            if _is_checker(patch, bg):
                sources.append(
                    {
                        "bbox": box,
                        "center": (r + 1, c + 1),
                        "top_left": int(patch[0, 0]),
                        "other": int(patch[0, 1]),
                    }
                )
            elif _is_ring(patch, bg):
                rings.append(
                    {
                        "bbox": box,
                        "center": (r + 1, c + 1),
                        "color": int(patch[0, 0]),
                    }
                )

    assert len(sources) == 1, "expected one 3x3 checkerboard source"
    source = sources[0]
    source_box = source["bbox"]
    rings = [ring for ring in rings if not _boxes_overlap(source_box, ring["bbox"])]
    assert len(rings) >= 2, "expected multiple 3x3 ring candidates"
    return source, rings


def _object_cells(source: dict, rings: list[dict]) -> set[Coord]:
    cells = set(_bbox_cells(source["bbox"]))
    for ring in rings:
        cells.update(_bbox_cells(ring["bbox"]))
    return cells


def _guide_nodes(grid: np.ndarray, source: dict, rings: list[dict]) -> set[Coord]:
    key_colors = {int(source["top_left"]), int(source["other"])}
    blocked = _object_cells(source, rings)
    out: set[Coord] = set()
    for r, c in np.argwhere(np.isin(grid, list(key_colors))):
        pt = (int(r), int(c))
        if pt not in blocked:
            out.add(pt)
    return out


def _midpoint(a: Coord, b: Coord) -> Coord:
    return (int((a[0] + b[0]) // 2), int((a[1] + b[1]) // 2))


def _lattice_neighbors(grid: np.ndarray, node: Coord, guide_nodes: set[Coord]) -> list[Coord]:
    out = []
    for dr, dc in DIRS:
        nb = (node[0] + 2 * dr, node[1] + 2 * dc)
        mid = (node[0] + dr, node[1] + dc)
        if nb in guide_nodes and int(grid[mid]) == _get_bg(grid):
            out.append(nb)
    return out


def _source_adjacent_guides(grid: np.ndarray, source: dict, guide_nodes: set[Coord]) -> list[Coord]:
    sr, sc = source["center"]
    bg = _get_bg(grid)
    starts = []
    for dr, dc in DIRS:
        gap = (sr + 2 * dr, sc + 2 * dc)
        node = (sr + 3 * dr, sc + 3 * dc)
        if _in_bounds(grid.shape, *gap) and node in guide_nodes and int(grid[gap]) == bg:
            starts.append(node)
    return starts


def _adjacent_ring_ids(grid: np.ndarray, node: Coord, rings: list[dict], bg: int) -> list[int]:
    hits = []
    nr, nc = node
    for i, ring in enumerate(rings):
        rr, rc = ring["center"]
        for dr, dc in DIRS:
            if (nr, nc) != (rr + 3 * dr, rc + 3 * dc):
                continue
            gap = (rr + 2 * dr, rc + 2 * dc)
            if _in_bounds(grid.shape, *gap) and int(grid[gap]) == int(bg):
                hits.append(i)
    return hits


def _legal_nexts(grid: np.ndarray, node: Coord, prev: Coord | None, guide_nodes: set[Coord]) -> list[Coord]:
    color = int(grid[node])
    out = []
    for nb in _lattice_neighbors(grid, node, guide_nodes):
        if prev is not None and nb == prev:
            continue
        if int(grid[nb]) != color:
            out.append(nb)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the checkerboard-key guide chain and crop the ring object it reaches.

    Input structure:
    - The background is the dominant color.
    - One 3x3 two-color checkerboard source gives the two guide colors and its upper-left cell gives the required first guide color.
    - Several 3x3 ring objects are candidates; guide pixels use only the checkerboard colors and lie on a two-cell lattice with visible branches.

    Rule:
    - Start from the guide pixel beside the source whose color equals the source checkerboard's upper-left color.
    - Move only between guide pixels two cells apart with a background gap, and only when each next color strictly alternates with the previous guide color.
    - Same-color branches and wrong-phase starts are ignored; the unique legal chain ends beside one ring, whose 3x3 crop is returned.

    Key insight:
    - Connectivity alone is a decoy: the source checkerboard supplies both the two-color key and the starting phase, so only the alternating keyed path identifies the target ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    source, rings = _find_source_and_rings(g, bg)
    guides = _guide_nodes(g, source, rings)

    starts = [
        pt
        for pt in _source_adjacent_guides(g, source, guides)
        if int(g[pt]) == int(source["top_left"])
    ]
    assert len(starts) == 1, "the checkerboard phase must select one first guide pixel"

    prev: Coord | None = None
    cur = starts[0]
    seen: set[Coord] = set()
    while True:
        assert cur not in seen, "legal alternating chain must not cycle"
        seen.add(cur)
        ring_hits = _adjacent_ring_ids(g, cur, rings, bg)
        nexts = [pt for pt in _legal_nexts(g, cur, prev, guides) if pt not in seen]
        if not nexts:
            assert len(ring_hits) == 1, "the legal chain must end beside exactly one ring"
            return _crop_bbox(g, rings[ring_hits[0]]["bbox"])
        assert len(ring_hits) == 0, "only the terminal guide may point to a ring"
        assert len(nexts) == 1, "alternation must leave a unique next guide"
        prev, cur = cur, nexts[0]


# --------------- generator ---------------

GRID_SIZE_RANGE = (21, 29)
TRUE_PATH_LEN_RANGE = (5, 8)
BRANCH_LEN_RANGE = (3, 5)
STRAY_GUIDE_RANGE = (3, 7)
EXTRA_RING_RANGE = (0, 2)
MAX_ATTEMPTS = 600


def _unit_dir(a: Coord, b: Coord) -> Coord:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert abs(dr) + abs(dc) == 2
    return (dr // 2, dc // 2)


def _step(node: Coord, direction: Coord, dist: int = 2) -> Coord:
    return (node[0] + direction[0] * dist, node[1] + direction[1] * dist)


def _fits_center(center: Coord, shape: tuple[int, int]) -> bool:
    return 1 <= center[0] < shape[0] - 1 and 1 <= center[1] < shape[1] - 1


def _core_node(node: Coord, shape: tuple[int, int]) -> bool:
    return 3 <= node[0] < shape[0] - 3 and 3 <= node[1] < shape[1] - 3


def _lattice_adjacent(a: Coord, b: Coord) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 2 and (a[0] == b[0] or a[1] == b[1])


def _can_add_node(node: Coord, existing: set[Coord], allowed: set[Coord], shape: tuple[int, int]) -> bool:
    if not _core_node(node, shape) or node in existing:
        return False
    for other in existing:
        if other not in allowed and _lattice_adjacent(node, other):
            return False
    return True


def _make_true_path(shape: tuple[int, int]) -> list[Coord] | None:
    h, w = shape
    parity_r = random.randrange(2)
    parity_c = random.randrange(2)
    candidates = [
        (r, c)
        for r in range(6, h - 6)
        for c in range(6, w - 6)
        if r % 2 == parity_r and c % 2 == parity_c
    ]
    random.shuffle(candidates)
    desired = random.randint(*TRUE_PATH_LEN_RANGE)

    for start in candidates[:80]:
        path = [start]
        for _ in range(desired - 1):
            dirs = list(DIRS)
            random.shuffle(dirs)
            placed = False
            for d in dirs:
                nxt = _step(path[-1], d)
                if _can_add_node(nxt, set(path), {path[-1]}, shape):
                    path.append(nxt)
                    placed = True
                    break
            if not placed:
                break
        if len(path) != desired:
            continue
        d0 = _unit_dir(path[0], path[1])
        d1 = _unit_dir(path[-2], path[-1])
        source_center = _step(path[0], (-d0[0], -d0[1]), 3)
        wrong_start = _step(source_center, (-d0[0], -d0[1]), 3)
        target_center = _step(path[-1], d1, 3)
        if all(_fits_center(pt, shape) for pt in (source_center, target_center)) and _core_node(wrong_start, shape):
            return path
    return None


def _grow_branch(
    seed: Coord,
    preferred_dir: Coord,
    existing: set[Coord],
    shape: tuple[int, int],
    length: int,
) -> list[Coord] | None:
    branch = [seed]
    used = set(existing)
    used.add(seed)
    last_dir = preferred_dir
    while len(branch) < length:
        dirs = [last_dir] + [d for d in DIRS if d != last_dir]
        random.shuffle(dirs[1:])
        placed = False
        for d in dirs:
            nxt = _step(branch[-1], d)
            if _can_add_node(nxt, used, {branch[-1]}, shape):
                branch.append(nxt)
                used.add(nxt)
                last_dir = d
                placed = True
                break
        if not placed:
            return None
    end_dir = last_dir if len(branch) == 1 else _unit_dir(branch[-2], branch[-1])
    if not _fits_center(_step(branch[-1], end_dir, 3), shape):
        return None
    return branch


def _make_same_color_branch(path: list[Coord], shape: tuple[int, int]) -> list[Coord] | None:
    anchors = list(range(1, len(path) - 1))
    random.shuffle(anchors)
    existing = set(path)
    for idx in anchors:
        anchor = path[idx]
        blocked_dirs = {_unit_dir(anchor, path[idx - 1]), _unit_dir(anchor, path[idx + 1])}
        dirs = [d for d in DIRS if d not in blocked_dirs]
        random.shuffle(dirs)
        for d in dirs:
            first = _step(anchor, d)
            if not _can_add_node(first, existing, {anchor}, shape):
                continue
            length = random.randint(*BRANCH_LEN_RANGE)
            branch = _grow_branch(first, d, existing | {anchor}, shape, length)
            if branch is not None:
                branch.insert(0, anchor)
                return branch
    return None


def _make_wrong_phase_branch(path: list[Coord], shape: tuple[int, int]) -> list[Coord] | None:
    d0 = _unit_dir(path[0], path[1])
    source_center = _step(path[0], (-d0[0], -d0[1]), 3)
    wrong_dir = (-d0[0], -d0[1])
    seed = _step(source_center, wrong_dir, 3)
    existing = set(path)
    if not _can_add_node(seed, existing, set(), shape):
        return None
    length = random.randint(*BRANCH_LEN_RANGE)
    return _grow_branch(seed, wrong_dir, existing, shape, length)


def _ring_bbox(center: Coord) -> BBox:
    r, c = center
    return (r - 1, c - 1, r + 1, c + 1)


def _box_clear(box: BBox, boxes: list[BBox], shape: tuple[int, int]) -> bool:
    padded = _pad_bbox(box, shape, 1)
    return all(not _boxes_overlap(padded, _pad_bbox(other, shape, 1)) for other in boxes)


def _draw_source(grid: np.ndarray, center: Coord, a: int, b: int) -> None:
    r0, c0 = center[0] - 1, center[1] - 1
    for rr in range(3):
        for cc in range(3):
            grid[r0 + rr, c0 + cc] = a if (rr + cc) % 2 == 0 else b


def _draw_ring(grid: np.ndarray, center: Coord, color: int, bg: int) -> None:
    r0, c0 = center[0] - 1, center[1] - 1
    grid[r0 : r0 + 3, c0 : c0 + 3] = color
    grid[r0 + 1, c0 + 1] = bg


def _place_extra_ring(
    shape: tuple[int, int],
    boxes: list[BBox],
    guide_nodes: set[Coord],
) -> Coord | None:
    candidates = [(r, c) for r in range(2, shape[0] - 2) for c in range(2, shape[1] - 2)]
    random.shuffle(candidates)
    for center in candidates:
        box = _ring_bbox(center)
        if not _box_clear(box, boxes, shape):
            continue
        padded = _bbox_cells(_pad_bbox(box, shape, 1))
        if guide_nodes & padded:
            continue
        return center
    return None


def _add_stray_guides(
    guide_colors: dict[Coord, int],
    shape: tuple[int, int],
    parity: tuple[int, int],
    boxes: list[BBox],
    colors: tuple[int, int],
) -> None:
    count = random.randint(*STRAY_GUIDE_RANGE)
    blocked_cells: set[Coord] = set()
    for box in boxes:
        blocked_cells.update(_bbox_cells(_pad_bbox(box, shape, 1)))
    candidates = [
        (r, c)
        for r in range(3, shape[0] - 3)
        for c in range(3, shape[1] - 3)
        if r % 2 == parity[0] and c % 2 == parity[1] and (r, c) not in blocked_cells
    ]
    random.shuffle(candidates)
    added = 0
    for pt in candidates:
        if added >= count:
            break
        if pt in guide_colors:
            continue
        if any(_lattice_adjacent(pt, other) for other in guide_colors):
            continue
        guide_colors[pt] = random.choice(colors)
        added += 1


def _alternating_hits_from(
    grid: np.ndarray,
    starts: list[Coord],
    rings: list[dict],
    guides: set[Coord],
    bg: int,
) -> set[int]:
    hits: set[int] = set()
    stack: list[tuple[Coord, Coord | None, frozenset[Coord]]] = [
        (start, None, frozenset({start})) for start in starts
    ]
    while stack:
        node, prev, seen = stack.pop()
        hits.update(_adjacent_ring_ids(grid, node, rings, bg))
        for nb in _legal_nexts(grid, node, prev, guides):
            if nb not in seen:
                stack.append((nb, node, seen | {nb}))
    return hits


def _semantic_probe(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        source, rings = _find_source_and_rings(grid, bg)
        guides = _guide_nodes(grid, source, rings)
        adjacent = _source_adjacent_guides(grid, source, guides)
        good = [pt for pt in adjacent if int(grid[pt]) == int(source["top_left"])]
        wrong = [pt for pt in adjacent if int(grid[pt]) != int(source["top_left"])]
        if len(good) != 1 or not wrong:
            return False

        q = deque(good)
        reachable = set(good)
        while q:
            cur = q.popleft()
            for nb in _lattice_neighbors(grid, cur, guides):
                if nb not in reachable:
                    reachable.add(nb)
                    q.append(nb)
        reachable_rings = {
            rid
            for node in reachable
            for rid in _adjacent_ring_ids(grid, node, rings, bg)
        }
        if len(reachable_rings) < 2:
            return False
        has_same_color_edge = any(
            int(grid[node]) == int(grid[nb])
            for node in reachable
            for nb in _lattice_neighbors(grid, node, guides)
        )
        if not has_same_color_edge:
            return False
        if not _alternating_hits_from(grid, wrong, rings, guides, bg):
            return False
        return True
    except Exception:
        return False


def _render_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    shape = (h, w)
    path = _make_true_path(shape)
    if path is None:
        return None

    same_branch = _make_same_color_branch(path, shape)
    wrong_branch = _make_wrong_phase_branch(path, shape)
    if same_branch is None or wrong_branch is None:
        return None

    d0 = _unit_dir(path[0], path[1])
    dend = _unit_dir(path[-2], path[-1])
    source_center = _step(path[0], (-d0[0], -d0[1]), 3)
    target_center = _step(path[-1], dend, 3)
    same_end_dir = _unit_dir(same_branch[-2], same_branch[-1])
    same_ring_center = _step(same_branch[-1], same_end_dir, 3)
    wrong_end_dir = _unit_dir(wrong_branch[-2], wrong_branch[-1])
    wrong_ring_center = _step(wrong_branch[-1], wrong_end_dir, 3)

    object_centers = [source_center, target_center, same_ring_center, wrong_ring_center]
    if not all(_fits_center(center, shape) for center in object_centers):
        return None
    boxes = [_ring_bbox(center) for center in object_centers]
    for i, box in enumerate(boxes):
        if not _box_clear(box, boxes[:i], shape):
            return None

    guide_nodes = set(path) | set(same_branch[1:]) | set(wrong_branch)
    padded_objects = set()
    for box in boxes:
        padded_objects.update(_bbox_cells(_pad_bbox(box, shape, 1)))
    if guide_nodes & padded_objects:
        return None

    palette = list(COLORS)
    random.shuffle(palette)
    bg, a, b = palette[:3]
    ring_colors = palette[3:]
    if len(ring_colors) < 5:
        return None

    guide_colors: dict[Coord, int] = {}
    for i, pt in enumerate(path):
        guide_colors[pt] = a if i % 2 == 0 else b
    anchor = same_branch[0]
    anchor_color = guide_colors[anchor]
    for i, pt in enumerate(same_branch[1:]):
        guide_colors[pt] = anchor_color if i % 2 == 0 else (b if anchor_color == a else a)
    for i, pt in enumerate(wrong_branch):
        guide_colors[pt] = b if i % 2 == 0 else a

    extra_centers = []
    for _ in range(random.randint(*EXTRA_RING_RANGE)):
        center = _place_extra_ring(shape, boxes, set(guide_colors))
        if center is not None:
            boxes.append(_ring_bbox(center))
            extra_centers.append(center)

    _add_stray_guides(guide_colors, shape, (path[0][0] % 2, path[0][1] % 2), boxes, (a, b))

    grid = np.full(shape, bg, dtype=int)
    _draw_source(grid, source_center, a, b)
    ring_specs = [
        (target_center, ring_colors[0]),
        (same_ring_center, ring_colors[1]),
        (wrong_ring_center, ring_colors[2]),
    ] + [(center, ring_colors[3 + i]) for i, center in enumerate(extra_centers)]
    for center, color in ring_specs:
        _draw_ring(grid, center, color, bg)
    for (r, c), color in guide_colors.items():
        if int(grid[r, c]) != int(bg):
            return None
        grid[r, c] = color

    expected = np.full((3, 3), int(ring_colors[0]), dtype=int)
    expected[1, 1] = bg
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_probe(grid):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _render_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid checkerboard-key path input")
