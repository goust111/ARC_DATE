import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 14
GRID_MAX = 30
SEED_COUNT_MIN = 3
SEED_COUNT_MAX = 6
SEGMENT_LEN_MIN = 2
SEGMENT_LEN_MAX = 7
DECOY_COUNT_MAX = 2
PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 300

_RNG = np.random.default_rng()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _grey_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
            neighbors.append((dr, dc))
    return neighbors


def _seed_targets(grid: np.ndarray, r: int, c: int) -> tuple[str, list[tuple[int, int]]] | None:
    neighbors = _grey_neighbors(grid, r, c)
    if len(neighbors) == 1:
        dr, dc = neighbors[0]
        away_r = -dr
        away_c = -dc
        return "single", [(r + away_r, c + away_c), (r + 2 * away_r, c + 2 * away_c)]

    if len(neighbors) != 2:
        return None

    (dr1, dc1), (dr2, dc2) = neighbors
    if dr1 == -dr2 and dc1 == -dc2:
        if dr1 == 0:
            return "cross", [(r - 1, c), (r + 1, c)]
        return "cross", [(r, c - 1), (r, c + 1)]

    away_r = -dr1 - dr2
    away_c = -dc1 - dc2
    if away_r == 0 or away_c == 0:
        return None
    return "corner", [(r + away_r, c), (r, c + away_c), (r + away_r, c + away_c)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored seed away from adjacent grey line segments with a shape chosen by local grey-neighbor geometry.

    Input structure:
    - The background is black.
    - Grey pixels form horizontal or vertical line segments.
    - Every other non-black pixel is a singleton seed color touching one or two grey pixels in the 4-neighborhood.

    Rule:
    - For each seed, count its 4-neighbor grey pixels.
    - If it touches exactly one grey neighbor, extend the seed into a 3-cell line pointing directly away from that grey pixel.
    - If it touches two opposite grey neighbors on the same line, add one seed-colored pixel on each side along the perpendicular axis, forming a 3-cell cross arm.
    - If it touches two orthogonal grey neighbors at a corner, fill the 2x2 block in the diagonal direction away from both grey line segments.
    - Grey pixels and existing seed pixels stay unchanged and are never overwritten.

    Key insight:
    - The transformation depends on two coupled cues: how many grey neighbors a seed has, and whether those neighbors are opposite or orthogonal, which together determine both the growth pattern and its direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    paint: dict[tuple[int, int], int] = {}
    conflicts: set[tuple[int, int]] = set()

    for r, c in np.argwhere((g != BG) & (g != GREY)):
        rr = int(r)
        cc = int(c)
        parsed = _seed_targets(g, rr, cc)
        if parsed is None:
            continue
        _kind, targets = parsed
        color = int(g[rr, cc])
        for tr, tc in targets:
            if not _in_bounds(g.shape, tr, tc):
                continue
            if int(g[tr, tc]) != BG:
                continue
            pos = (tr, tc)
            if pos in paint and paint[pos] != color:
                conflicts.add(pos)
            else:
                paint[pos] = color

    for (r, c), color in paint.items():
        if (r, c) not in conflicts:
            out[r, c] = color
    return out


def _make_single(color: int) -> dict[str, object]:
    direction = DIRS4[int(_RNG.integers(0, len(DIRS4)))]
    dr, dc = direction
    seg_len = int(_RNG.integers(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX + 1))
    grey = [(dr * step, dc * step) for step in range(1, seg_len + 1)]
    seed = (0, 0, color)
    targets = [(-dr, -dc), (-2 * dr, -2 * dc)]
    return {"kind": "single", "grey": grey, "seed": seed, "targets": targets}


def _make_cross(color: int) -> dict[str, object]:
    horizontal = bool(int(_RNG.integers(0, 2)))
    left_len = int(_RNG.integers(1, SEGMENT_LEN_MAX))
    right_len = int(_RNG.integers(1, SEGMENT_LEN_MAX))
    if horizontal:
        grey = [(0, -step) for step in range(1, left_len + 1)]
        grey += [(0, step) for step in range(1, right_len + 1)]
        targets = [(-1, 0), (1, 0)]
    else:
        grey = [(-step, 0) for step in range(1, left_len + 1)]
        grey += [(step, 0) for step in range(1, right_len + 1)]
        targets = [(0, -1), (0, 1)]
    seed = (0, 0, color)
    return {"kind": "cross", "grey": grey, "seed": seed, "targets": targets}


def _make_corner(color: int) -> dict[str, object]:
    vertical = [(-1, 0), (1, 0)][int(_RNG.integers(0, 2))]
    horizontal = [(0, -1), (0, 1)][int(_RNG.integers(0, 2))]
    vdr, vdc = vertical
    hdr, hdc = horizontal
    v_len = int(_RNG.integers(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX + 1))
    h_len = int(_RNG.integers(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX + 1))
    grey = [(vdr * step, vdc * step) for step in range(1, v_len + 1)]
    grey += [(hdr * step, hdc * step) for step in range(1, h_len + 1)]
    away_r = -vdr
    away_c = -hdc
    seed = (0, 0, color)
    targets = [(away_r, 0), (0, away_c), (away_r, away_c)]
    return {"kind": "corner", "grey": grey, "seed": seed, "targets": targets}


def _motif_points(motif: dict[str, object]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int, int]]:
    grey = [(int(r), int(c)) for r, c in motif["grey"]]
    sr, sc, color = motif["seed"]
    seed = (int(sr), int(sc), int(color))
    targets = [(int(r), int(c)) for r, c in motif["targets"]]
    return grey, targets, seed


def _reserve_cells(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((r + dr, c + dc))
    return reserved


def _place_motif(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    motif: dict[str, object],
) -> bool:
    grey, targets, seed = _motif_points(motif)
    sr, sc, color = seed
    shape_points = set(grey + targets + [(sr, sc)])

    rows = [r for r, _ in shape_points]
    cols = [c for _, c in shape_points]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    row_lo = -min_r
    row_hi = grid.shape[0] - 1 - max_r
    col_lo = -min_c
    col_hi = grid.shape[1] - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        off_r = int(_RNG.integers(row_lo, row_hi + 1))
        off_c = int(_RNG.integers(col_lo, col_hi + 1))
        placed = {(r + off_r, c + off_c) for r, c in shape_points}
        reserved = _reserve_cells(placed)
        if any(cell in blocked for cell in reserved):
            continue

        for r, c in grey:
            grid[r + off_r, c + off_c] = GREY
        grid[sr + off_r, sc + off_c] = color
        blocked.update(reserved)
        return True
    return False


def _place_decoy_segment(grid: np.ndarray, blocked: set[tuple[int, int]]) -> bool:
    horizontal = bool(int(_RNG.integers(0, 2)))
    length = int(_RNG.integers(3, SEGMENT_LEN_MAX + 2))
    if horizontal:
        local = [(0, step) for step in range(length)]
    else:
        local = [(step, 0) for step in range(length)]

    rows = [r for r, _ in local]
    cols = [c for _, c in local]
    row_hi = grid.shape[0] - 1 - max(rows)
    col_hi = grid.shape[1] - 1 - max(cols)
    if row_hi < 0 or col_hi < 0:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        off_r = int(_RNG.integers(0, row_hi + 1))
        off_c = int(_RNG.integers(0, col_hi + 1))
        placed = {(r + off_r, c + off_c) for r, c in local}
        reserved = _reserve_cells(placed)
        if any(cell in blocked for cell in reserved):
            continue
        for r, c in placed:
            grid[r, c] = GREY
        blocked.update(reserved)
        return True
    return False


def _balanced_kinds(count: int) -> list[str]:
    kinds = ["single", "cross", "corner"]
    base = count // len(kinds)
    result = [kind for kind in kinds for _ in range(base)]
    extra = count - len(result)
    if extra:
        extras = _RNG.choice(np.array(kinds, dtype=object), size=extra, replace=False).tolist()
        result.extend(str(kind) for kind in extras)
    _RNG.shuffle(result)
    return result


def _count_added_pixels(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.count_nonzero((inp == BG) & (out != BG)))


def _validate_grid(grid: np.ndarray, expected_seed_count: int) -> bool:
    seeds = np.argwhere((grid != BG) & (grid != GREY))
    if len(seeds) != expected_seed_count:
        return False

    colors = [int(grid[int(r), int(c)]) for r, c in seeds]
    if len(set(colors)) != len(colors):
        return False

    expected_added = 0
    kinds_seen: set[str] = set()
    for r, c in seeds:
        rr = int(r)
        cc = int(c)
        parsed = _seed_targets(grid, rr, cc)
        if parsed is None:
            return False
        kind, targets = parsed
        kinds_seen.add(kind)
        if len(_grey_neighbors(grid, rr, cc)) not in (1, 2):
            return False
        for tr, tc in targets:
            if not _in_bounds(grid.shape, tr, tc):
                return False
            if int(grid[tr, tc]) != BG:
                return False
        expected_added += len(targets)

    if kinds_seen != {"single", "cross", "corner"}:
        return False

    out = main(grid)
    if _count_added_pixels(grid, out) != expected_added:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    width = int(_RNG.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((height, width), BG, dtype=int)

    seed_count = int(_RNG.integers(SEED_COUNT_MIN, SEED_COUNT_MAX + 1))
    if seed_count > len(PALETTE):
        return None

    colors = _RNG.choice(np.array(PALETTE, dtype=int), size=seed_count, replace=False).tolist()
    kinds = _balanced_kinds(seed_count)
    blocked: set[tuple[int, int]] = set()

    builders = {"single": _make_single, "cross": _make_cross, "corner": _make_corner}
    for kind, color in zip(kinds, colors):
        motif = builders[kind](int(color))
        if not _place_motif(grid, blocked, motif):
            return None

    decoy_count = int(_RNG.integers(0, DECOY_COUNT_MAX + 1))
    for _ in range(decoy_count):
        _place_decoy_segment(grid, blocked)

    if not _validate_grid(grid, seed_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
