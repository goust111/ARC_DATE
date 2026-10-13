import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPES = [
    np.array([(0, 0), (0, 1)], dtype=int),
    np.array([(0, 0), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1)], dtype=int),
]
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_dims(shape: np.ndarray) -> tuple[int, int]:
    _r0, _c0, r1, c1 = _bbox(shape)
    return r1 + 1, c1 + 1


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(pts), dtype=int))
    return comps


def _edge_distance(coords: np.ndarray, direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    if dr == -1:
        return int(coords[:, 0].min())
    if dr == 1:
        return int(h - 1 - coords[:, 0].max())
    if dc == -1:
        return int(coords[:, 1].min())
    return int(w - 1 - coords[:, 1].max())


def _marker_candidates(shape: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    shape_set = _coord_set(shape)
    candidates = []
    for r, c in shape:
        mr, mc = int(r + dr), int(c + dc)
        if (mr, mc) in shape_set:
            continue
        touch = sum(1 for nr, nc in _neighbors4(mr, mc) if (nr, nc) in shape_set)
        if touch == 1:
            candidates.append((mr, mc))
    return sorted(set(candidates))


def _sorted_objects(grid: np.ndarray):
    source_coords = np.argwhere(grid == RED)
    assert source_coords.shape == (1, 2), "expected exactly one red source"
    source = tuple(map(int, source_coords[0]))

    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)]
    objects = []
    used_markers: set[tuple[int, int]] = set()
    for comp in _components(grid == BLUE):
        comp_set = _coord_set(comp)
        adjacent = []
        for marker in markers:
            dirs = {
                (int(marker[0] - r), int(marker[1] - c))
                for r, c in comp_set
                if abs(marker[0] - r) + abs(marker[1] - c) == 1
            }
            if dirs:
                assert len(dirs) == 1, "direction marker must indicate one side"
                adjacent.append((marker, next(iter(dirs))))
        assert len(adjacent) == 1, "each blue object must have exactly one adjacent yellow marker"
        marker, direction = adjacent[0]
        assert marker not in used_markers, "direction marker cannot serve multiple objects"
        used_markers.add(marker)
        r0, c0, r1, c1 = _bbox(comp)
        objects.append(
            {
                "coords": comp.copy(),
                "direction": direction,
                "marker": marker,
                "bbox": (r0, c0, r1, c1),
                "priority": _edge_distance(comp, direction, grid.shape),
            }
        )
    assert len(used_markers) == len(markers), "every yellow marker must belong to one object"
    ordered = sorted(
        objects,
        key=lambda obj: (obj["priority"], obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]),
    )
    return source, ordered


def _simulate(grid: np.ndarray):
    h, w = grid.shape
    source, ordered = _sorted_objects(grid)
    fixed: set[tuple[int, int]] = set()
    final_objects = []
    for obj in ordered:
        coords = obj["coords"].copy()
        dr, dc = obj["direction"]
        trail = [coords.copy()]
        while True:
            moved = _translate(coords, dr, dc)
            if not ((0 <= moved[:, 0]).all() and (moved[:, 0] < h).all() and (0 <= moved[:, 1]).all() and (moved[:, 1] < w).all()):
                stop_reason = "edge"
                break
            moved_set = _coord_set(moved)
            if moved_set & fixed:
                stop_reason = "obstacle"
                break
            coords = moved
            trail.append(coords.copy())
        fixed |= _coord_set(coords)
        final_objects.append(
            {
                "initial": obj["coords"],
                "final": coords,
                "direction": obj["direction"],
                "priority": obj["priority"],
                "stop_reason": stop_reason,
                "trail": trail,
            }
        )
    return source, final_objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each blue object using its yellow direction marker, then cast green rays from the red source until the slid blue obstacles stop them.

    Input structure:
    - A black background contains one red source pixel, plus 3-4 disconnected blue 4-connected objects of size 2-3.
    - Each blue object has exactly one adjacent yellow pixel on one outer side, which indicates its slide direction.
    - Yellow is only an instruction marker; blue objects are the only obstacles after sliding.

    Rule:
    - Read each blue object's direction from its adjacent yellow marker: yellow above/below/left/right means slide up/down/left/right.
    - Process blue objects from nearest to farthest relative to their target edge in that direction; each object slides rigidly until the next step would leave the grid or hit an already stopped blue object.
    - Erase all yellow markers, keep the red source and final blue objects, then cast rays from the red source in the four cardinal directions, painting traversed black cells green until a blue pixel or the grid edge stops the ray.

    Key insight:
    - The visible blue positions are only temporary: you must first execute the instruction markers to build the final blue obstacle layout, and only then can you determine the ray lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    source, final_objects = _simulate(g)
    out = np.zeros((h, w), dtype=int)
    out[source] = RED
    for obj in final_objects:
        out[obj["final"][:, 0], obj["final"][:, 1]] = BLUE

    sr, sc = source
    for dr, dc in DIRS4:
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w:
            if out[r, c] == BLUE:
                break
            if out[r, c] == BG:
                out[r, c] = GREEN
            r += dr
            c += dc
    return out


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _final_coords(shape: np.ndarray, direction: tuple[int, int], top_left: tuple[int, int]) -> np.ndarray:
    r0, c0 = top_left
    return shape + np.array([int(r0), int(c0)], dtype=int)


def _build_chain(rng: np.random.Generator, h: int, w: int, source: tuple[int, int], occupied: set[tuple[int, int]]):
    for _ in range(80):
        direction = _choice(rng, DIRS4)
        shape = _choice(rng, SHAPES)
        sh, sw = _shape_dims(shape)
        dr, dc = direction
        if dc == 1:
            if w < 2 * sw:
                continue
            r0 = int(rng.integers(0, h - sh + 1))
            a_tl = (r0, w - sw)
            b_tl = (r0, w - 2 * sw)
        elif dc == -1:
            if w < 2 * sw:
                continue
            r0 = int(rng.integers(0, h - sh + 1))
            a_tl = (r0, 0)
            b_tl = (r0, sw)
        elif dr == 1:
            if h < 2 * sh:
                continue
            c0 = int(rng.integers(0, w - sw + 1))
            a_tl = (h - sh, c0)
            b_tl = (h - 2 * sh, c0)
        else:
            if h < 2 * sh:
                continue
            c0 = int(rng.integers(0, w - sw + 1))
            a_tl = (0, c0)
            b_tl = (sh, c0)
        a_coords = _final_coords(shape, direction, a_tl)
        b_coords = _final_coords(shape, direction, b_tl)
        a_set = _coord_set(a_coords)
        b_set = _coord_set(b_coords)
        if source in a_set or source in b_set:
            continue
        if a_set & occupied or b_set & occupied or a_set & b_set:
            continue
        return [
            {"shape": shape, "direction": direction, "final": a_coords, "role": "edge_chain"},
            {"shape": shape, "direction": direction, "final": b_coords, "role": "obstacle_chain"},
        ]
    return None


def _build_source_blocker(
    rng: np.random.Generator,
    h: int,
    w: int,
    source: tuple[int, int],
    occupied: set[tuple[int, int]],
):
    sr, sc = source
    for _ in range(120):
        direction = _choice(rng, DIRS4)
        shape = _choice(rng, SHAPES)
        sh, sw = _shape_dims(shape)
        if direction[0] != 0:
            aligned = [int(c) for _r, c in shape]
            col = _choice(rng, aligned)
            c0 = sc - col
            r0 = 0 if direction[0] == -1 else h - sh
        else:
            aligned = [int(r) for r, _c in shape]
            row = _choice(rng, aligned)
            r0 = sr - row
            c0 = 0 if direction[1] == -1 else w - sw
        if not (0 <= r0 <= h - sh and 0 <= c0 <= w - sw):
            continue
        coords = _final_coords(shape, direction, (r0, c0))
        coord_set = _coord_set(coords)
        if source in coord_set or coord_set & occupied:
            continue
        if not any(r == sr or c == sc for r, c in coord_set):
            continue
        return {"shape": shape, "direction": direction, "final": coords, "role": "source_blocker"}
    return None


def _build_random_edge(rng: np.random.Generator, h: int, w: int, source: tuple[int, int], occupied: set[tuple[int, int]]):
    for _ in range(120):
        direction = _choice(rng, DIRS4)
        shape = _choice(rng, SHAPES)
        sh, sw = _shape_dims(shape)
        dr, dc = direction
        if dc == 1:
            r0 = int(rng.integers(0, h - sh + 1))
            c0 = w - sw
        elif dc == -1:
            r0 = int(rng.integers(0, h - sh + 1))
            c0 = 0
        elif dr == 1:
            r0 = h - sh
            c0 = int(rng.integers(0, w - sw + 1))
        else:
            r0 = 0
            c0 = int(rng.integers(0, w - sw + 1))
        coords = _final_coords(shape, direction, (r0, c0))
        coord_set = _coord_set(coords)
        if source in coord_set or coord_set & occupied:
            continue
        return {"shape": shape, "direction": direction, "final": coords, "role": "edge_extra"}
    return None


def _initial_options(spec: dict, h: int, w: int):
    shape = spec["shape"]
    direction = spec["direction"]
    final_coords = spec["final"]
    r0, c0, _r1, _c1 = _bbox(final_coords)
    marker_rel = _marker_candidates(shape, direction)
    if not marker_rel:
        return []
    options = []
    for steps in range(1, 6):
        init_r = int(r0 - direction[0] * steps)
        init_c = int(c0 - direction[1] * steps)
        coords = _final_coords(shape, direction, (init_r, init_c))
        if not ((0 <= coords[:, 0]).all() and (coords[:, 0] < h).all() and (0 <= coords[:, 1]).all() and (coords[:, 1] < w).all()):
            continue
        for mr, mc in marker_rel:
            marker = (int(init_r + mr), int(init_c + mc))
            if 0 <= marker[0] < h and 0 <= marker[1] < w:
                options.append({"coords": coords, "marker": marker})
    return options


def _can_place_bundle(
    coords: np.ndarray,
    marker: tuple[int, int],
    blue_used: set[tuple[int, int]],
    mark_used: set[tuple[int, int]],
    source: tuple[int, int],
) -> bool:
    cells = _coord_set(coords)
    if source in cells or marker == source:
        return False
    if cells & blue_used or cells & mark_used or marker in blue_used or marker in mark_used:
        return False
    occupied = blue_used | mark_used | {source}
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    for nr, nc in _neighbors4(marker[0], marker[1]):
        if (nr, nc) in occupied:
            return False
    return True


def _assign_initial_layout(specs: list[dict], h: int, w: int, source: tuple[int, int], rng: np.random.Generator):
    option_lists = [_initial_options(spec, h, w) for spec in specs]
    if any(not opts for opts in option_lists):
        return None

    order = sorted(range(len(specs)), key=lambda idx: len(option_lists[idx]))
    chosen = [None] * len(specs)
    blue_used: set[tuple[int, int]] = set()
    mark_used: set[tuple[int, int]] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            priorities = [
                _edge_distance(chosen[idx]["coords"], specs[idx]["direction"], (h, w))
                for idx in range(len(specs))
            ]
            return len(set(priorities)) == len(priorities)

        idx = order[pos]
        perm = rng.permutation(len(option_lists[idx]))
        for opt_idx in perm:
            opt = option_lists[idx][int(opt_idx)]
            if not _can_place_bundle(opt["coords"], opt["marker"], blue_used, mark_used, source):
                continue
            chosen[idx] = opt
            cells = _coord_set(opt["coords"])
            blue_used.update(cells)
            mark_used.add(opt["marker"])
            if backtrack(pos + 1):
                return True
            blue_used.difference_update(cells)
            mark_used.remove(opt["marker"])
            chosen[idx] = None
        return False

    if not backtrack(0):
        return None
    return chosen


def _ray_blocked(out: np.ndarray, source: tuple[int, int], direction: tuple[int, int]) -> bool:
    h, w = out.shape
    r, c = source[0] + direction[0], source[1] + direction[1]
    while 0 <= r < h and 0 <= c < w:
        if out[r, c] == BLUE:
            return True
        r += direction[0]
        c += direction[1]
    return False


def _valid_input(grid: np.ndarray) -> bool:
    try:
        source, final_objects = _simulate(grid)
    except AssertionError:
        return False

    if not (3 <= len(final_objects) <= 4):
        return False

    priorities = [obj["priority"] for obj in final_objects]
    if len(set(priorities)) != len(priorities):
        return False
    if {obj["stop_reason"] for obj in final_objects} != {"edge", "obstacle"}:
        return False
    if len({obj["direction"] for obj in final_objects}) < 2:
        return False

    for obj in final_objects:
        if np.array_equal(obj["initial"], obj["final"]):
            return False
        for step_coords in obj["trail"]:
            if source in _coord_set(step_coords):
                return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.sum(out == GREEN)) == 0:
        return False
    blocked_dirs = sum(_ray_blocked(out, source, direction) for direction in DIRS4)
    if blocked_dirs < 1:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(13, 17))
    w = int(rng.integers(13, 17))
    source = (h // 2, w // 2)
    num_objects = 3 if rng.random() < 0.55 else 4

    specs: list[dict] = []
    occupied_final = {source}

    chain = _build_chain(rng, h, w, source, occupied_final)
    if chain is None:
        return None
    specs.extend(chain)
    for spec in chain:
        occupied_final |= _coord_set(spec["final"])

    blocker = _build_source_blocker(rng, h, w, source, occupied_final)
    if blocker is None:
        return None
    specs.append(blocker)
    occupied_final |= _coord_set(blocker["final"])

    if num_objects == 4:
        extra = _build_random_edge(rng, h, w, source, occupied_final)
        if extra is None:
            return None
        specs.append(extra)
        occupied_final |= _coord_set(extra["final"])

    chosen = _assign_initial_layout(specs, h, w, source, rng)
    if chosen is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[source] = RED
    for opt in chosen:
        coords = opt["coords"]
        grid[coords[:, 0], coords[:, 1]] = BLUE
        grid[opt["marker"]] = YELLOW

    return grid if _valid_input(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-ray ARC input")
