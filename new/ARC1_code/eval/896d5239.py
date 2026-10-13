import numpy as np

from arc_utils import Color, in_bounds


_WEDGE_SIDES = (
    ("down", ((1, -1), (1, 1))),
    ("up", ((-1, -1), (-1, 1))),
    ("left", ((-1, 1), (1, 1))),
    ("right", ((-1, -1), (1, -1))),
)


def _prefix_len(distances: list[int]) -> int:
    present = set(distances)
    d = 1
    while d in present:
        d += 1
    return d - 1


def _ray_marker_distances(g: np.ndarray, r: int, c: int, dr: int, dc: int) -> list[int]:
    distances: list[int] = []
    d = 1
    while in_bounds(g.shape, r + dr * d, c + dc * d):
        if g[r + dr * d, c + dc * d] == Color.GREEN:
            distances.append(d)
        d += 1
    return distances


def _candidate_wedges(g: np.ndarray) -> list[tuple[str, int, int, int]]:
    wedges: list[tuple[str, int, int, int]] = []
    for r, c in np.argwhere(g == Color.GREEN):
        rr = int(r)
        cc = int(c)
        for kind, (side_a, side_b) in _WEDGE_SIDES:
            da = _ray_marker_distances(g, rr, cc, *side_a)
            db = _ray_marker_distances(g, rr, cc, *side_b)
            common = set(da) & set(db)
            if not common:
                continue
            height = max(max(common), _prefix_len(da), _prefix_len(db))
            if height >= 2:
                wedges.append((kind, rr, cc, height))
    return wedges


def _wedge_cells(
    kind: str, r: int, c: int, height: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if kind == "down":
        for d in range(1, height + 1):
            rr = r + d
            for cc in range(c - d, c + d + 1):
                if in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    elif kind == "up":
        for d in range(1, height + 1):
            rr = r - d
            for cc in range(c - d, c + d + 1):
                if in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    elif kind == "left":
        for d in range(1, height + 1):
            cc = c + d
            for rr in range(r - d, r + d + 1):
                if in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    elif kind == "right":
        for d in range(1, height + 1):
            cc = c - d
            for rr in range(r - d, r + d + 1):
                if in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    else:
        raise AssertionError(f"unexpected wedge kind: {kind}")
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the triangular wedges indicated by green diagonal marker pixels with teal.

    Input structure
    A same-size grid with black/blue background texture and green marker pixels.
    The green pixels lie on pairs of 45-degree sides of one or more triangular
    wedges; some side or base pixels may be omitted or clipped by the grid edge.

    Rule
    For each green marker that acts as a wedge tip, find the two diagonal side
    rays in one of the four orientations.  The side rays must contain green
    clues at a shared distance from the tip.  The wedge height is the farthest
    shared clue distance, extended by any contiguous green clue run from the tip.
    Paint every non-green pixel inside the resulting triangular wedge teal.

    Key insight
    The green pixels are not objects to move; they are sparse boundary clues for
    45-degree triangular fill regions, and unrelated far-away diagonal markers
    are ignored unless they are part of a shared-distance clue or near-tip run.
    """
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, g.ravel()))
    assert Color.GREEN in colors
    assert Color.TEAL not in colors
    assert colors <= {Color.BLACK, Color.BLUE, Color.GREEN}

    wedges = _candidate_wedges(g)
    assert wedges

    out = g.copy()
    painted = False
    for kind, r, c, height in wedges:
        for rr, cc in _wedge_cells(kind, r, c, height, g.shape):
            if out[rr, cc] != Color.GREEN:
                out[rr, cc] = Color.TEAL
                painted = True

    assert painted
    return out

# --------------- generator (o5x) ---------------

# Canvas and background sampling bounds.
GEN_ROW_MIN, GEN_ROW_MAX = 13, 15
GEN_COL_MIN, GEN_COL_MAX = 12, 18
GEN_BLUE_RATIO_RANGE = (0.45, 0.55)

# Wedge structure and placement constraints.
GEN_WEDGE_COUNT_MIN, GEN_WEDGE_COUNT_MAX = 2, 4
GEN_LOW_HEIGHT = 2
GEN_HIGH_HEIGHT = 3
GEN_ORIENTATIONS = ["up", "down", "left", "right"]
GEN_SIDE_DIRS = dict(_WEDGE_SIDES)
GEN_MIN_REGION_GAP = 4
GEN_MAX_EDGE_TOUCHING = 2
GEN_EDGE_TOUCH_BIAS = 0.35

# Marker sampling and retry limits.
GEN_MARKER_MIN_TOTAL = 2
GEN_MARKER_MAX_FACTOR = 2
GEN_RETRY_LIMIT = 300


def _gen_side_dirs(kind: str) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        return GEN_SIDE_DIRS[kind]
    except KeyError as exc:
        raise AssertionError(f"unexpected wedge kind: {kind}") from exc


def _gen_side_positions(kind: str, r: int, c: int, height: int) -> list[tuple[int, int, int]]:
    side_a, side_b = _gen_side_dirs(kind)
    pos: list[tuple[int, int, int]] = []
    for d in range(1, height + 1):
        for dr, dc in (side_a, side_b):
            pos.append((d, r + dr * d, c + dc * d))
    return pos


def _gen_region(kind: str, r: int, c: int, height: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    region = {(r, c)}
    region.update(_wedge_cells(kind, r, c, height, shape))
    region.update((rr, cc) for _, rr, cc in _gen_side_positions(kind, r, c, height))
    return region


def _gen_all_fit(kind: str, r: int, c: int, height: int, shape: tuple[int, int]) -> bool:
    expected_cells = height * (height + 2)
    if not in_bounds(shape, r, c):
        return False
    if len(_wedge_cells(kind, r, c, height, shape)) != expected_cells:
        return False
    return all(in_bounds(shape, rr, cc) for _, rr, cc in _gen_side_positions(kind, r, c, height))


def _gen_edge_touch(kind: str, r: int, c: int, height: int, shape: tuple[int, int]) -> bool:
    H, W = shape
    return any(
        rr == 0 or rr == H - 1 or cc == 0 or cc == W - 1
        for rr, cc in _wedge_cells(kind, r, c, height, shape)
    )


def _gen_valid_placements(
    kind: str, height: int, shape: tuple[int, int]
) -> list[tuple[int, int, set[tuple[int, int]], bool]]:
    H, W = shape
    placements = []
    for r in range(H):
        for c in range(W):
            if _gen_all_fit(kind, r, c, height, shape):
                placements.append(
                    (r, c, _gen_region(kind, r, c, height, shape), _gen_edge_touch(kind, r, c, height, shape))
                )
    return placements


def _gen_far_enough(
    region: set[tuple[int, int]],
    used: set[tuple[int, int]],
    gap: int = GEN_MIN_REGION_GAP,
) -> bool:
    for r1, c1 in region:
        for r2, c2 in used:
            if abs(r1 - r2) + abs(c1 - c2) < gap:
                return False
    return True


def _gen_pick(seq):
    return seq[int(np.random.randint(0, len(seq)))]


def _gen_background(shape: tuple[int, int]) -> np.ndarray:
    total = shape[0] * shape[1]
    lo, hi = GEN_BLUE_RATIO_RANGE
    blue = int(round(total * float(np.random.uniform(lo, hi))))
    flat = np.full(total, Color.BLACK, dtype=int)
    flat[:blue] = Color.BLUE
    np.random.shuffle(flat)
    return flat.reshape(shape)


def _gen_marker_subset(kind: str, r: int, c: int, height: int) -> set[tuple[int, int]]:
    side_pos = _gen_side_positions(kind, r, c, height)
    required = {(rr, cc) for d, rr, cc in side_pos if d == height}
    remaining = [(rr, cc) for d, rr, cc in side_pos if d != height]
    total = int(np.random.randint(max(height, GEN_MARKER_MIN_TOTAL), GEN_MARKER_MAX_FACTOR * height + 1))
    extra = max(0, total - len(required))
    if extra:
        order = np.random.permutation(len(remaining))[:extra]
        required.update(remaining[int(i)] for i in order)
    return required


def _gen_wedge_specs(count: int) -> list[tuple[str, int]]:
    heights = [GEN_HIGH_HEIGHT] + [GEN_LOW_HEIGHT] * (count - 1)
    np.random.shuffle(heights)

    chosen_kinds = [str(_gen_pick(GEN_ORIENTATIONS)) for _ in range(count)]
    if count > 1 and len(set(chosen_kinds)) == 1:
        alternatives = [kind for kind in GEN_ORIENTATIONS if kind != chosen_kinds[0]]
        replacement = str(_gen_pick(alternatives))
        chosen_kinds[int(np.random.randint(0, count))] = replacement

    specs = list(zip(chosen_kinds, heights))
    np.random.shuffle(specs)
    return specs


def _gen_place_wedges(
    specs: list[tuple[str, int]], shape: tuple[int, int]
) -> list[tuple[str, int, int, int]] | None:
    used: set[tuple[int, int]] = set()
    placed: list[tuple[str, int, int, int]] = []
    edge_count = 0
    for kind, height in specs:
        candidates = [
            p
            for p in _gen_valid_placements(kind, height, shape)
            if _gen_far_enough(p[2], used)
        ]
        if edge_count >= GEN_MAX_EDGE_TOUCHING:
            candidates = [p for p in candidates if not p[3]]
        elif np.random.random() < GEN_EDGE_TOUCH_BIAS:
            edge = [p for p in candidates if p[3]]
            if edge:
                candidates = edge
        if not candidates:
            return None
        r, c, region, touches_edge = _gen_pick(candidates)
        used.update(region)
        edge_count += int(touches_edge)
        placed.append((kind, int(r), int(c), int(height)))
    return placed


def _gen_attempt() -> np.ndarray | None:
    H = int(np.random.randint(GEN_ROW_MIN, GEN_ROW_MAX + 1))
    W = int(np.random.randint(GEN_COL_MIN, GEN_COL_MAX + 1))
    shape = (H, W)
    count = int(np.random.randint(GEN_WEDGE_COUNT_MIN, GEN_WEDGE_COUNT_MAX + 1))

    specs = _gen_wedge_specs(count)
    placed = _gen_place_wedges(specs, shape)
    if placed is None:
        return None

    background = _gen_background(shape)
    out = background.copy()
    for kind, r, c, height in placed:
        out[r, c] = Color.GREEN
        for rr, cc in _gen_marker_subset(kind, r, c, height):
            out[rr, cc] = Color.GREEN

    for kind, r, c, height in placed:
        for rr, cc in _wedge_cells(kind, r, c, height, shape):
            if out[rr, cc] != Color.GREEN:
                out[rr, cc] = Color.TEAL

    inp = out.copy()
    inp[out == Color.TEAL] = background[out == Color.TEAL]

    candidates = _candidate_wedges(inp)
    if sorted(candidates) != sorted(placed):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 896d5239 input")
