from __future__ import annotations

from collections import Counter

import numpy as np


RNG = np.random.default_rng()

BG = 0
WALL = 5
PAIR_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_SIZE = 12
MAX_SIZE = 24
MIN_PAIRS = 2
MAX_PAIRS = 5
MIN_ARM = 2
MAX_ARM = 7
SEG_EXT_MIN = 1
SEG_EXT_MAX = 3
PAIR_TRIES = 250
EXTRA_SEGMENT_TRIES = 120
GEN_TRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _diag_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ra, ca = a
    rb, cb = b
    dr = rb - ra
    dc = cb - ca
    assert abs(dr) == abs(dc), f"not a diagonal segment: {a} -> {b}"
    if dr == 0:
        return [a]
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(int(ra + i * sr), int(ca + i * sc)) for i in range(abs(dr) + 1)]


def _wall_orientations(grid: np.ndarray, wall: int) -> dict[tuple[int, int], str]:
    mask = grid == int(wall)
    h, w = grid.shape
    out: dict[tuple[int, int], str] = {}
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        horiz = (c > 0 and bool(mask[r, c - 1])) or (c + 1 < w and bool(mask[r, c + 1]))
        vert = (r > 0 and bool(mask[r - 1, c])) or (r + 1 < h and bool(mask[r + 1, c]))
        assert horiz ^ vert, "every wall cell must belong to a single straight segment"
        out[(r, c)] = "H" if horiz else "V"
    return out


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _validate_candidate(
    grid: np.ndarray,
    color: int,
    bg: int,
    a: tuple[int, int],
    b: tuple[int, int],
    pre: tuple[int, int],
    post: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if not (_in_bounds(grid.shape, pre) and _in_bounds(grid.shape, post)):
        return None
    if int(grid[pre]) != int(bg) or int(grid[post]) != int(bg):
        return None

    seg1 = _diag_cells(a, pre)
    seg2 = _diag_cells(post, b)

    if any(int(grid[r, c]) != int(bg) for r, c in seg1[1:]):
        return None
    if any(int(grid[r, c]) != int(bg) for r, c in seg2[:-1]):
        return None

    return seg1 + seg2


def _find_reflection_path(
    grid: np.ndarray,
    color: int,
    a: tuple[int, int],
    b: tuple[int, int],
    wall: int,
    orientations: dict[tuple[int, int], str],
) -> list[tuple[int, int]]:
    bg = _get_bg(grid)
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

    for (r, c), axis in orientations.items():
        if axis == "H":
            if a[0] == r or b[0] == r:
                continue
            if (a[0] < r) != (b[0] < r):
                continue
            side = -1 if a[0] < r else 1
            da = abs(a[0] - r)
            db = abs(b[0] - r)
            for s in (-1, 1):
                if a[1] != c - s * da:
                    continue
                if b[1] != c + s * db:
                    continue
                pre = (int(r + side), int(c - s))
                post = (int(r + side), int(c + s))
                path = _validate_candidate(grid, color, bg, a, b, pre, post)
                if path is not None:
                    candidates.append(((int(r), int(c)), path))
        else:
            if a[1] == c or b[1] == c:
                continue
            if (a[1] < c) != (b[1] < c):
                continue
            side = -1 if a[1] < c else 1
            da = abs(a[1] - c)
            db = abs(b[1] - c)
            for s in (-1, 1):
                if a[0] != r - s * da:
                    continue
                if b[0] != r + s * db:
                    continue
                pre = (int(r - s), int(c + side))
                post = (int(r + s), int(c + side))
                path = _validate_candidate(grid, color, bg, a, b, pre, post)
                if path is not None:
                    candidates.append(((int(r), int(c)), path))

    dedup: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], list[tuple[int, int]]] = {}
    for mirror, path in candidates:
        dedup[(mirror, tuple(path))] = path

    assert len(dedup) == 1, f"expected exactly one reflection path for color {color}, got {len(dedup)}"
    return next(iter(dedup.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color pixel pair with a single 45-degree mirror bounce off the gray wall segments.

    Input structure:
    - The background is the dominant color.
    - Gray pixels form one-pixel-thick straight wall segments, each entirely horizontal or vertical.
    - Every other foreground color appears exactly twice as two isolated same-color pixels away from the walls.

    Rule:
    - For each same-color pair, find the unique path made of two diagonal 45-degree segments with exactly one reflection on a wall cell.
    - A horizontal wall flips the vertical travel direction, while a vertical wall flips the horizontal travel direction.
    - Paint every visited non-wall cell along that reflected path with the pair color, including the endpoints; keep all wall pixels unchanged.

    Key insight:
    The connectors are not straight or Manhattan paths: each pair is solved by identifying the one mirror cell where a diagonal ray can bounce and then drawing the two reflected diagonal arms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    counts = Counter(map(int, g.ravel()))
    non_bg = [int(c) for c in counts if int(c) != int(bg)]
    wall_candidates = [int(c) for c in non_bg if int(counts[c]) > 2]
    assert len(wall_candidates) == 1, "expected a single wall color"
    wall = int(wall_candidates[0])
    assert int(wall) == WALL, "wall color should be the visible gray color"

    pair_colors = [int(c) for c in non_bg if int(c) != int(wall)]
    assert pair_colors, "expected at least one colored pair"
    assert all(int(counts[c]) == 2 for c in pair_colors), "every non-wall foreground color must appear exactly twice"

    orientations = _wall_orientations(g, wall)
    out = g.copy()

    for color in sorted(pair_colors):
        pts = [tuple(map(int, p)) for p in np.argwhere(g == int(color))]
        assert len(pts) == 2
        a, b = pts
        path = _find_reflection_path(g, int(color), a, b, wall, orientations)
        for r, c in path:
            assert int(out[r, c]) in (int(bg), int(color))
            out[r, c] = int(color)

    return out


def _segment_ok(
    segment: list[tuple[int, int]],
    walls: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    for cell in segment:
        if cell in walls or cell in blocked:
            return False
        r, c = cell
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in walls:
                    return False
    return True


def _sample_horizontal_pair(
    h: int,
    w: int,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    list[tuple[int, int]],
    list[tuple[int, int]],
] | None:
    left_ext = int(RNG.integers(SEG_EXT_MIN, SEG_EXT_MAX + 1))
    right_ext = int(RNG.integers(SEG_EXT_MIN, SEG_EXT_MAX + 1))
    if h < 2 * MIN_ARM + 1 or w < left_ext + right_ext + 5:
        return None

    r = int(RNG.integers(MIN_ARM, h - MIN_ARM))
    c_lo = left_ext
    c_hi = w - 1 - right_ext
    if c_hi < c_lo:
        return None
    c = int(RNG.integers(c_lo, c_hi + 1))
    side = int(RNG.choice((-1, 1)))
    s = int(RNG.choice((-1, 1)))

    row_room = r if side == -1 else (h - 1 - r)
    a_col_room = c if s == 1 else (w - 1 - c)
    b_col_room = (w - 1 - c) if s == 1 else c
    max_da = min(MAX_ARM, row_room, a_col_room)
    max_db = min(MAX_ARM, row_room, b_col_room)
    if max_da < MIN_ARM or max_db < MIN_ARM:
        return None

    da = int(RNG.integers(MIN_ARM, max_da + 1))
    db = int(RNG.integers(MIN_ARM, max_db + 1))
    a = (int(r + side * da), int(c - s * da))
    b = (int(r + side * db), int(c + s * db))
    pre = (int(r + side), int(c - s))
    post = (int(r + side), int(c + s))

    seg1 = _diag_cells(a, pre)
    seg2 = _diag_cells(post, b)
    wall_segment = [(int(r), int(cc)) for cc in range(int(c - left_ext), int(c + right_ext) + 1)]
    return a, b, wall_segment, seg1 + seg2


def _sample_vertical_pair(
    h: int,
    w: int,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    list[tuple[int, int]],
    list[tuple[int, int]],
] | None:
    up_ext = int(RNG.integers(SEG_EXT_MIN, SEG_EXT_MAX + 1))
    down_ext = int(RNG.integers(SEG_EXT_MIN, SEG_EXT_MAX + 1))
    if w < 2 * MIN_ARM + 1 or h < up_ext + down_ext + 5:
        return None

    c = int(RNG.integers(MIN_ARM, w - MIN_ARM))
    r_lo = up_ext
    r_hi = h - 1 - down_ext
    if r_hi < r_lo:
        return None
    r = int(RNG.integers(r_lo, r_hi + 1))
    side = int(RNG.choice((-1, 1)))
    s = int(RNG.choice((-1, 1)))

    col_room = c if side == -1 else (w - 1 - c)
    a_row_room = r if s == 1 else (h - 1 - r)
    b_row_room = (h - 1 - r) if s == 1 else r
    max_da = min(MAX_ARM, col_room, a_row_room)
    max_db = min(MAX_ARM, col_room, b_row_room)
    if max_da < MIN_ARM or max_db < MIN_ARM:
        return None

    da = int(RNG.integers(MIN_ARM, max_da + 1))
    db = int(RNG.integers(MIN_ARM, max_db + 1))
    a = (int(r - s * da), int(c + side * da))
    b = (int(r + s * db), int(c + side * db))
    pre = (int(r - s), int(c + side))
    post = (int(r + s), int(c + side))

    seg1 = _diag_cells(a, pre)
    seg2 = _diag_cells(post, b)
    wall_segment = [(int(rr), int(c)) for rr in range(int(r - up_ext), int(r + down_ext) + 1)]
    return a, b, wall_segment, seg1 + seg2


def _sample_extra_segment(
    h: int,
    w: int,
    walls: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    axis = str(RNG.choice(("H", "V")))
    if axis == "H":
        length = int(RNG.integers(3, 7))
        if w < length or h < 1:
            return None
        r = int(RNG.integers(0, h))
        c0 = int(RNG.integers(0, w - length + 1))
        segment = [(int(r), int(c0 + dc)) for dc in range(length)]
    else:
        length = int(RNG.integers(3, 7))
        if h < length or w < 1:
            return None
        c = int(RNG.integers(0, w))
        r0 = int(RNG.integers(0, h - length + 1))
        segment = [(int(r0 + dr), int(c)) for dr in range(length)]

    if not _segment_ok(segment, walls, blocked):
        return None
    return segment


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(MIN_SIZE, MAX_SIZE + 1))
    w = int(RNG.integers(MIN_SIZE, MAX_SIZE + 1))
    grid = np.full((h, w), BG, dtype=int)

    max_pairs = min(MAX_PAIRS, len(PAIR_PALETTE))
    pair_count = int(RNG.integers(MIN_PAIRS, max_pairs + 1))
    pair_colors = list(map(int, RNG.choice(PAIR_PALETTE, size=pair_count, replace=False)))

    axes = [str(RNG.choice(("H", "V"))) for _ in range(pair_count)]
    if pair_count >= 2 and RNG.random() < 0.5:
        axes[0] = "H"
        axes[1] = "V"
        RNG.shuffle(axes)

    walls: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    endpoints: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for color, axis in zip(pair_colors, axes):
        placed = False
        for _ in range(PAIR_TRIES):
            if axis == "H":
                sample = _sample_horizontal_pair(h, w)
            else:
                sample = _sample_vertical_pair(h, w)
            if sample is None:
                continue

            a, b, wall_segment, path_cells = sample
            wall_set = set(wall_segment)
            path_set = set(path_cells)

            if a == b:
                continue
            if wall_set & blocked:
                continue
            if path_set & walls:
                continue
            if path_set & blocked:
                continue
            if not _segment_ok(wall_segment, walls, blocked):
                continue

            walls |= wall_set
            blocked |= path_set
            endpoints.append((int(color), a, b))
            placed = True
            break
        if not placed:
            return None

    extra_segments = int(RNG.integers(0, 3))
    for _ in range(extra_segments):
        for _ in range(EXTRA_SEGMENT_TRIES):
            segment = _sample_extra_segment(h, w, walls, blocked)
            if segment is None:
                continue
            walls |= set(segment)
            break

    for r, c in walls:
        grid[r, c] = WALL
    for color, a, b in endpoints:
        if a in walls or b in walls:
            return None
        grid[a] = int(color)
        grid[b] = int(color)

    if int((grid == WALL).sum()) < 3:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mirror-reflection ARC input")
