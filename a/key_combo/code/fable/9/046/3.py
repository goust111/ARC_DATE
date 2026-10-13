from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
BROWN = 9

GEN_RETRIES = 400
SIZE_OPTIONS = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
DIRECTIONS = ("U", "D", "L", "R")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    comps.sort(key=lambda arr: (int(arr[:, 0].min()), int(arr[:, 1].min())))
    return comps


def _min_chebyshev(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    diff = np.abs(coords_a[:, None, :] - coords_b[None, :, :])
    return int(diff.max(axis=2).min())


def _point_to_coords_chebyshev(point: tuple[int, int], coords: np.ndarray) -> int:
    pr, pc = point
    diff = np.abs(coords - np.array([pr, pc], dtype=int))
    return int(diff.max(axis=1).min())


def _generation_color(depth: int) -> int:
    if depth == 1:
        return YELLOW
    if depth == 2:
        return ORANGE
    return BROWN


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "grid must stay within ARC size limits"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREY}, "expected only black, red, and gray in the input"
    assert _get_bg(g) == BLACK, "expected black as the dominant background"

    sparks = np.argwhere(g == RED)
    assert len(sparks) == 1, "expected exactly one red spark pixel"
    spark = tuple(map(int, sparks[0]))

    components = _find_components(g == GREY)
    assert components, "expected at least one gray object"

    n = len(components)
    spark_dists = [_point_to_coords_chebyshev(spark, comp) for comp in components]
    obj_dists = [[999 for _ in range(n)] for _ in range(n)]
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        obj_dists[i][i] = 0
        for j in range(i + 1, n):
            d = _min_chebyshev(components[i], components[j])
            obj_dists[i][j] = d
            obj_dists[j][i] = d
            if d <= 2:
                adj[i].append(j)
                adj[j].append(i)

    generations: list[int | None] = [None] * n
    q = deque()
    for i, d in enumerate(spark_dists):
        if d <= 2:
            generations[i] = 1
            q.append(i)

    while q:
        i = q.popleft()
        next_depth = int(generations[i]) + 1
        for j in adj[i]:
            if generations[j] is None:
                generations[j] = next_depth
                q.append(j)

    return {
        "spark": spark,
        "components": components,
        "spark_dists": spark_dists,
        "obj_dists": obj_dists,
        "adj": adj,
        "generations": generations,
    }


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    info = _analyze_scene(g)
    out = g.copy()
    for comp, depth in zip(info["components"], info["generations"]):
        if depth is None:
            continue
        out[comp[:, 0], comp[:, 1]] = _generation_color(int(depth))
    spark_r, spark_c = info["spark"]
    out[spark_r, spark_c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite gray objects by halo-hop generations from the unique red spark and recolor each object by the hop count that reaches it.

    Input structure:
    - A black background contains several disconnected gray foreground objects and exactly one red spark pixel.
    - Gray objects may have different shapes and sizes, and the gaps between them are not uniform.
    - The spark is placed on black near one gray object, and all gray objects are interpreted as separate 8-connected components.

    Rule:
    - Treat two gray objects as linked when their minimum Chebyshev distance is at most 2, meaning a one-cell-wide black gap between them can be crossed.
    - Treat the spark as linked to any gray object whose minimum Chebyshev distance from the spark pixel is at most 2.
    - Run object-level breadth-first propagation from the spark: generation 1 objects become yellow, generation 2 objects become orange, and generation 3 or later objects become brown.
    - Gray objects never reached by this propagation remain gray, and the spark itself stays red.

    Key insight:
    - The output colors encode hidden propagation depth through an object graph defined by a Chebyshev-distance threshold: one-cell gaps transmit, but two-cell gaps block the whole downstream chain.
    """
    return _oracle_output(np.asarray(grid, dtype=int))


def _rect_bounds(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, h, w = rect
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _rect_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = _rect_bounds(a)
    br0, bc0, br1, bc1 = _rect_bounds(b)
    if ar1 < br0:
        row_gap = br0 - ar1
    elif br1 < ar0:
        row_gap = ar0 - br1
    else:
        row_gap = 0
    if ac1 < bc0:
        col_gap = bc0 - ac1
    elif bc1 < ac0:
        col_gap = ac0 - bc1
    else:
        col_gap = 0
    return int(max(row_gap, col_gap))


def _point_rect_distance(point: tuple[int, int], rect: tuple[int, int, int, int]) -> int:
    pr, pc = point
    r0, c0, r1, c1 = _rect_bounds(rect)
    if pr < r0:
        row_gap = r0 - pr
    elif pr > r1:
        row_gap = pr - r1
    else:
        row_gap = 0
    if pc < c0:
        col_gap = c0 - pc
    elif pc > c1:
        col_gap = pc - c1
    else:
        col_gap = 0
    return int(max(row_gap, col_gap))


def _offset_rect(
    anchor: tuple[int, int, int, int],
    size: tuple[int, int],
    direction: str,
    gap: int,
) -> tuple[int, int, int, int]:
    ar0, ac0, ar1, ac1 = _rect_bounds(anchor)
    h, w = size
    if direction == "R":
        c0 = ac1 + gap
        r0 = random.randint(ar0 - h + 1, ar1)
    elif direction == "L":
        c1 = ac0 - gap
        c0 = c1 - w + 1
        r0 = random.randint(ar0 - h + 1, ar1)
    elif direction == "D":
        r0 = ar1 + gap
        c0 = random.randint(ac0 - w + 1, ac1)
    else:
        r1 = ar0 - gap
        r0 = r1 - h + 1
        c0 = random.randint(ac0 - w + 1, ac1)
    return int(r0), int(c0), int(h), int(w)


def _sample_rect_size() -> tuple[int, int]:
    return random.choice(SIZE_OPTIONS)


def _candidate_ok(
    candidate: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    anchor_idx: int,
    exact_anchor_gap: int,
) -> bool:
    for idx, rect in enumerate(placed):
        dist = _rect_distance(candidate, rect)
        if idx == anchor_idx:
            if dist != exact_anchor_gap:
                return False
        elif dist <= 2:
            return False
    return True


def _place_relative(
    placed: list[tuple[int, int, int, int]],
    anchor_idx: int,
    gap: int,
    forbidden: set[str] | None = None,
    tries: int = 80,
) -> tuple[int, int, int, int] | None:
    size = _sample_rect_size()
    directions = list(DIRECTIONS)
    random.shuffle(directions)
    if forbidden:
        directions = [d for d in directions if d not in forbidden] or directions
    for _ in range(tries):
        direction = random.choice(directions)
        candidate = _offset_rect(placed[anchor_idx], size, direction, gap)
        if _candidate_ok(candidate, placed, anchor_idx, gap):
            return candidate
    return None


def _layout_span(rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    rows = []
    cols = []
    for rect in rects:
        r0, c0, r1, c1 = _rect_bounds(rect)
        rows.extend([r0, r1])
        cols.extend([c0, c1])
    return min(rows), min(cols), max(rows), max(cols)


def _place_spark(rects: list[tuple[int, int, int, int]], root_idx: int) -> tuple[int, int] | None:
    root = rects[root_idx]
    r0, c0, r1, c1 = _rect_bounds(root)
    all_points = set()
    for rect in rects:
        rr0, cc0, rr1, cc1 = _rect_bounds(rect)
        for rr in range(rr0, rr1 + 1):
            for cc in range(cc0, cc1 + 1):
                all_points.add((rr, cc))

    desired_gaps = [random.choice((1, 2)), 1, 2]
    for gap in desired_gaps:
        candidates = []
        for rr in range(r0 - 2, r1 + 3):
            for cc in range(c0 - 2, c1 + 3):
                point = (rr, cc)
                if point in all_points:
                    continue
                if _point_rect_distance(point, root) != gap:
                    continue
                if any(_point_rect_distance(point, rect) <= 2 for idx, rect in enumerate(rects) if idx != root_idx):
                    continue
                candidates.append(point)
        if candidates:
            return random.choice(candidates)
    return None


def _render_scene(
    rects: list[tuple[int, int, int, int]],
    spark: tuple[int, int],
) -> np.ndarray | None:
    min_r, min_c, max_r, max_c = _layout_span(rects)
    min_r = min(min_r, spark[0])
    min_c = min(min_c, spark[1])
    max_r = max(max_r, spark[0])
    max_c = max(max_c, spark[1])

    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    if span_h > 24 or span_w > 24:
        return None

    top = random.randint(1, 4)
    left = random.randint(1, 4)
    max_extra_bottom = min(30 - (span_h + top), 5)
    max_extra_right = min(30 - (span_w + left), 5)
    if max_extra_bottom < 1 or max_extra_right < 1:
        return None
    bottom = random.randint(1, max_extra_bottom)
    right = random.randint(1, max_extra_right)

    h = span_h + top + bottom
    w = span_w + left + right
    grid = np.full((h, w), BLACK, dtype=int)
    row_shift = top - min_r
    col_shift = left - min_c

    for rect in rects:
        r0, c0, hh, ww = rect
        rr = r0 + row_shift
        cc = c0 + col_shift
        grid[rr : rr + hh, cc : cc + ww] = GREY

    sr, sc = spark
    grid[sr + row_shift, sc + col_shift] = RED
    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        info = _analyze_scene(grid)
        expected = _oracle_output(grid)
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    generations = info["generations"]
    reachable = [i for i, depth in enumerate(generations) if depth is not None]
    unreachable = [i for i, depth in enumerate(generations) if depth is None]
    if len(info["components"]) < 5:
        return False
    if not reachable or not unreachable:
        return False
    if not any(depth == 1 for depth in generations if depth is not None):
        return False
    if not any(depth == 2 for depth in generations if depth is not None):
        return False
    if not any(int(depth) >= 3 for depth in generations if depth is not None):
        return False
    if not any(info["spark_dists"][i] > 2 for i in reachable if generations[i] is not None and generations[i] >= 2):
        return False
    if not any(
        info["obj_dists"][i][j] <= 2
        for i in range(len(generations))
        for j in range(i + 1, len(generations))
    ):
        return False
    if not any(
        info["obj_dists"][i][j] == 3
        for i in unreachable
        for j in reachable
    ):
        return False
    if not any(
        info["obj_dists"][i][j] <= 2
        for i in unreachable
        for j in unreachable
        if i < j
    ):
        return False
    out_colors = set(map(int, np.unique(expected)))
    if not {RED, YELLOW, ORANGE, BROWN, GREY}.issubset(out_colors):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    backbone_len = random.randint(3, 5)
    extra_count = random.randint(0, 2)

    root = (0, 0, *_sample_rect_size())
    rects: list[tuple[int, int, int, int]] = [root]
    backbone = [0]

    for _ in range(backbone_len - 1):
        candidate = _place_relative(rects, backbone[-1], gap=2)
        if candidate is None:
            return None
        rects.append(candidate)
        backbone.append(len(rects) - 1)

    for _ in range(extra_count):
        anchor_idx = random.choice(backbone)
        candidate = _place_relative(rects, anchor_idx, gap=2)
        if candidate is None:
            return None
        rects.append(candidate)

    blocked_anchor = random.choice(backbone[max(1, len(backbone) // 2) - 1 :])
    blocked_root = _place_relative(rects, blocked_anchor, gap=3)
    if blocked_root is None:
        return None
    rects.append(blocked_root)
    blocked_root_idx = len(rects) - 1

    blocked_child = _place_relative(rects, blocked_root_idx, gap=2)
    if blocked_child is None:
        return None
    rects.append(blocked_child)

    span = _layout_span(rects)
    if (span[2] - span[0] + 1) > 22 or (span[3] - span[1] + 1) > 22:
        return None

    spark = _place_spark(rects, root_idx=0)
    if spark is None:
        return None

    grid = _render_scene(rects, spark)
    if grid is None:
        return None
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid halo-wavefront ARC input")
