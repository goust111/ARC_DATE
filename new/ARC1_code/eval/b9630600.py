from __future__ import annotations
import numpy as np

from arc_utils import colors_present, find_components, get_bg
import random


def _rectangles(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 1
    color = colors[0]

    rects = []
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[color])
    assert len(comps) > 1
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3

        frame = np.zeros((h, w), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(patch == color, frame)
        assert int(comp["area"]) == int(frame.sum())
        rects.append((r0, c0, r1, c1))

    return color, rects


def _candidate_edges(g: np.ndarray, bg: int, rects: list[tuple[int, int, int, int]]) -> list[dict]:
    edges = []
    for i, a in enumerate(rects):
        for j in range(i + 1, len(rects)):
            b = rects[j]

            if a[3] < b[1] or b[3] < a[1]:
                if a[3] < b[1]:
                    left_i, right_i, left, right = i, j, a, b
                else:
                    left_i, right_i, left, right = j, i, b, a
                r0 = max(left[0] + 1, right[0] + 1)
                r1 = min(left[2] - 1, right[2] - 1)
                c0 = left[3] + 1
                c1 = right[1] - 1
                if r0 <= r1 and c0 <= c1 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg):
                    edges.append(
                        {
                            "gap": c1 - c0 + 1,
                            "kind": "h",
                            "a": left_i,
                            "b": right_i,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                        }
                    )

            if a[2] < b[0] or b[2] < a[0]:
                if a[2] < b[0]:
                    upper_i, lower_i, upper, lower = i, j, a, b
                else:
                    upper_i, lower_i, upper, lower = j, i, b, a
                c0 = max(upper[1] + 1, lower[1] + 1)
                c1 = min(upper[3] - 1, lower[3] - 1)
                r0 = upper[2] + 1
                r1 = lower[0] - 1
                if c0 <= c1 and r0 <= r1 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == bg):
                    edges.append(
                        {
                            "gap": r1 - r0 + 1,
                            "kind": "v",
                            "a": upper_i,
                            "b": lower_i,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                        }
                    )
    return edges


def _spanning_edges(edges: list[dict], n_rects: int) -> list[dict]:
    parent = list(range(n_rects))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        assert ra != rb
        parent[rb] = ra

    chosen = []
    for gap in sorted({edge["gap"] for edge in edges}):
        group = [edge for edge in edges if edge["gap"] == gap]
        eligible = [edge for edge in group if find(edge["a"]) != find(edge["b"])]

        temp_parent: dict[int, int] = {}

        def temp_find(x: int) -> int:
            temp_parent.setdefault(x, x)
            while temp_parent[x] != x:
                temp_parent[x] = temp_parent[temp_parent[x]]
                x = temp_parent[x]
            return x

        for edge in eligible:
            ra = find(edge["a"])
            rb = find(edge["b"])
            ta = temp_find(ra)
            tb = temp_find(rb)
            assert ta != tb
            temp_parent[tb] = ta

        for edge in eligible:
            union(edge["a"], edge["b"])
            chosen.append(edge)

        if len(chosen) == n_rects - 1:
            break

    assert len(chosen) == n_rects - 1
    root = find(0)
    assert all(find(i) == root for i in range(n_rects))
    return chosen


def _apply_edge(
    out: np.ndarray,
    bg: int,
    color: int,
    rects: list[tuple[int, int, int, int]],
    edge: dict,
) -> None:
    r0 = edge["r0"]
    r1 = edge["r1"]
    c0 = edge["c0"]
    c1 = edge["c1"]
    if edge["kind"] == "h":
        left = rects[edge["a"]]
        right = rects[edge["b"]]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        if r0 + 1 <= r1 - 1:
            out[r0 + 1 : r1, left[3]] = bg
            out[r0 + 1 : r1, right[1]] = bg
    else:
        upper = rects[edge["a"]]
        lower = rects[edge["b"]]
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        if c0 + 1 <= c1 - 1:
            out[upper[2], c0 + 1 : c1] = bg
            out[lower[0], c0 + 1 : c1] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect hollow rectangular frames into one shortest visible corridor network.

    Input structure
    - A single foreground color forms disconnected hollow rectangular frames on the background.
    - A visible candidate gap is horizontal or vertical, has overlapping frame interiors, and is clear background between the two frames.

    Rule
    - Select the minimum-length spanning set of visible frame-to-frame gaps.
    - For a horizontal selected gap, paint the two boundary rows of the overlapping interiors across the gap, then erase the facing wall cells strictly between those rows.
    - For a vertical selected gap, do the transposed operation with boundary columns and the facing top/bottom wall cells.

    Key insight
    - The connector is determined by the overlap of the rooms' interiors, and longer redundant visible gaps are ignored once the shortest spanning network connects all frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    color, rects = _rectangles(g, bg)
    edges = _candidate_edges(g, bg, rects)
    chosen = _spanning_edges(edges, len(rects))

    out = g.copy()
    for edge in chosen:
        _apply_edge(out, bg, color, rects, edge)

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, colors=[color])
    assert len(comps) == 1
    return out

# --------------- generator (o5x) ---------------

GEN_BG, GEN_COLOR = 0, 3

# Canvas bounds.
GEN_GRID_SIZE = 30

# Room count and size distribution.
GEN_ROOM_COUNT_MIN, GEN_ROOM_COUNT_MAX = 6, 8
GEN_SMALL_ROOM_MIN, GEN_SMALL_ROOM_MAX = 3, 4
GEN_LARGE_ROOM_MIN = 5
GEN_LARGE_ROOM_FALLBACK_MAX = 11
GEN_SIZE_PROFILE_BY_ROOM_COUNT = {
    8: (8, 0.34),
    7: (10, 0.25),
    6: (11, 0.18),
}

# Corridor geometry constraints.
GEN_GAP_MIN, GEN_GAP_MAX = 1, 7
GEN_OVERLAP_MIN, GEN_OVERLAP_MAX = 1, 7

# Placement scoring and retry limits.
GEN_OVERLAP_SCORE_WEIGHT = 30.0
GEN_SCORE_JITTER = 24.0
GEN_TOP_VALID_CHOICES = 8
GEN_TARGET_RETRIES, GEN_FALLBACK_RETRIES = 220, 80

GenRect = tuple[int, int, int, int]


def _gen_size(max_dim: int, small_rate: float) -> tuple[int, int]:
    if random.random() < small_rate:
        return (
            random.randint(GEN_SMALL_ROOM_MIN, GEN_SMALL_ROOM_MAX),
            random.randint(GEN_SMALL_ROOM_MIN, GEN_SMALL_ROOM_MAX),
        )
    return (
        random.randint(GEN_LARGE_ROOM_MIN, max_dim),
        random.randint(GEN_LARGE_ROOM_MIN, max_dim),
    )


def _gen_room_sizes(n_rooms: int) -> list[tuple[int, int]]:
    max_dim, small_rate = GEN_SIZE_PROFILE_BY_ROOM_COUNT[n_rooms]

    sizes = [_gen_size(max_dim, small_rate) for _ in range(n_rooms)]
    if not any(h <= GEN_SMALL_ROOM_MAX and w <= GEN_SMALL_ROOM_MAX for h, w in sizes):
        sizes[random.randrange(n_rooms)] = (
            random.randint(GEN_SMALL_ROOM_MIN, GEN_SMALL_ROOM_MAX),
            random.randint(GEN_SMALL_ROOM_MIN, GEN_SMALL_ROOM_MAX),
        )
    if not any(h >= GEN_LARGE_ROOM_MIN or w >= GEN_LARGE_ROOM_MIN for h, w in sizes):
        sizes[random.randrange(n_rooms)] = (
            random.randint(GEN_LARGE_ROOM_MIN, GEN_LARGE_ROOM_FALLBACK_MAX),
            random.randint(GEN_LARGE_ROOM_MIN, GEN_LARGE_ROOM_FALLBACK_MAX),
        )
    random.shuffle(sizes)
    return sizes


def _gen_overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _gen_child_positions(
    parent: GenRect,
    size: tuple[int, int],
    kind: str,
    gap: int,
) -> list[GenRect]:
    pr0, pc0, pr1, pc1 = parent
    h, w = size
    by_overlap: dict[int, list[GenRect]] = {}

    if kind == "h":
        for r0 in range(pr0 - h + 3, pr1 - 1):
            r1 = r0 + h - 1
            overlap = _gen_overlap_len(pr0 + 1, pr1 - 1, r0 + 1, r1 - 1)
            if GEN_OVERLAP_MIN <= overlap <= GEN_OVERLAP_MAX:
                left = (r0, pc0 - gap - w, r1, pc0 - gap - 1)
                right = (r0, pc1 + gap + 1, r1, pc1 + gap + w)
                by_overlap.setdefault(overlap, []).extend([left, right])
    else:
        for c0 in range(pc0 - w + 3, pc1 - 1):
            c1 = c0 + w - 1
            overlap = _gen_overlap_len(pc0 + 1, pc1 - 1, c0 + 1, c1 - 1)
            if GEN_OVERLAP_MIN <= overlap <= GEN_OVERLAP_MAX:
                upper = (pr0 - gap - h, c0, pr0 - gap - 1, c1)
                lower = (pr1 + gap + 1, c0, pr1 + gap + h, c1)
                by_overlap.setdefault(overlap, []).extend([upper, lower])

    overlaps = list(by_overlap)
    random.shuffle(overlaps)
    options: list[GenRect] = []
    for overlap in overlaps:
        group = by_overlap[overlap]
        random.shuffle(group)
        options.extend(group)
    return options


def _gen_rect_ok(rect: GenRect, rects: list[GenRect]) -> bool:
    r0, c0, r1, c1 = rect
    for q0, q1, q2, q3 in rects:
        rows_overlap = not (r1 < q0 or q2 < r0)
        cols_overlap = not (c1 < q1 or q3 < c0)
        if rows_overlap and cols_overlap:
            return False
        if rows_overlap and (c1 + 1 == q1 or q3 + 1 == c0):
            return False
        if cols_overlap and (r1 + 1 == q0 or q2 + 1 == r0):
            return False
    return True


def _gen_rect_bounds(rects: list[GenRect]) -> GenRect:
    min_r = min(r0 for r0, _, _, _ in rects)
    min_c = min(c0 for _, c0, _, _ in rects)
    max_r = max(r1 for _, _, r1, _ in rects)
    max_c = max(c1 for _, _, _, c1 in rects)
    return min_r, min_c, max_r, max_c


def _gen_span_ok(rects: list[GenRect]) -> bool:
    min_r, min_c, max_r, max_c = _gen_rect_bounds(rects)
    return max_r - min_r + 1 <= GEN_GRID_SIZE and max_c - min_c + 1 <= GEN_GRID_SIZE


def _gen_bbox_area(rects: list[GenRect]) -> int:
    min_r, min_c, max_r, max_c = _gen_rect_bounds(rects)
    return (max_r - min_r + 1) * (max_c - min_c + 1)


def _gen_parent_overlap(parent: GenRect, rect: GenRect, kind: str) -> int:
    pr0, pc0, pr1, pc1 = parent
    r0, c0, r1, c1 = rect
    if kind == "h":
        return _gen_overlap_len(pr0 + 1, pr1 - 1, r0 + 1, r1 - 1)
    return _gen_overlap_len(pc0 + 1, pc1 - 1, c0 + 1, c1 - 1)


def _gen_place_rooms(n_rooms: int) -> list[GenRect] | None:
    sizes = _gen_room_sizes(n_rooms)
    parents = [-1] + [random.randrange(i) for i in range(1, n_rooms)]
    kinds = [random.choice(["h", "v"]) for _ in range(n_rooms - 1)]
    gaps = random.sample(range(GEN_GAP_MIN, GEN_GAP_MAX + 1), n_rooms - 1)
    random.shuffle(gaps)

    h0, w0 = sizes[0]
    rects = [(0, 0, h0 - 1, w0 - 1)]
    for i in range(1, n_rooms):
        options = _gen_child_positions(rects[parents[i]], sizes[i], kinds[i - 1], gaps[i - 1])
        valid = []
        for rect in options:
            trial = rects + [rect]
            if _gen_rect_ok(rect, rects) and _gen_span_ok(trial):
                overlap = _gen_parent_overlap(rects[parents[i]], rect, kinds[i - 1])
                score = (
                    _gen_bbox_area(trial)
                    - overlap * GEN_OVERLAP_SCORE_WEIGHT
                    + random.random() * GEN_SCORE_JITTER
                )
                valid.append((score, rect))
        if not valid:
            return None
        valid.sort(key=lambda item: item[0])
        rects.append(random.choice(valid[: min(GEN_TOP_VALID_CHOICES, len(valid))])[1])
    return rects


def _gen_shift_rooms(rects: list[GenRect]) -> list[GenRect]:
    min_r, min_c, max_r, max_c = _gen_rect_bounds(rects)
    free_r = GEN_GRID_SIZE - (max_r - min_r + 1)
    free_c = GEN_GRID_SIZE - (max_c - min_c + 1)
    dr = -min_r + random.randint(0, free_r)
    dc = -min_c + random.randint(0, free_c)
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _gen_draw_frames(rects: list[GenRect]) -> np.ndarray:
    grid = np.zeros((GEN_GRID_SIZE, GEN_GRID_SIZE), dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0, c0 : c1 + 1] = GEN_COLOR
        grid[r1, c0 : c1 + 1] = GEN_COLOR
        grid[r0 : r1 + 1, c0] = GEN_COLOR
        grid[r0 : r1 + 1, c1] = GEN_COLOR
    return grid


def _gen_edge_overlap(edge: dict) -> int:
    if edge["kind"] == "h":
        return edge["r1"] - edge["r0"] + 1
    return edge["c1"] - edge["c0"] + 1


def _gen_expected_output(inp: np.ndarray, n_rooms: int) -> np.ndarray | None:
    try:
        color, rects = _rectangles(inp, GEN_BG)
        if color != GEN_COLOR or len(rects) != n_rooms:
            return None
        edges = _candidate_edges(inp, GEN_BG, rects)
        if not (n_rooms - 1 <= len(edges) <= n_rooms):
            return None
        if any(edge["gap"] < GEN_GAP_MIN or edge["gap"] > GEN_GAP_MAX for edge in edges):
            return None
        if any(
            _gen_edge_overlap(edge) < GEN_OVERLAP_MIN
            or _gen_edge_overlap(edge) > GEN_OVERLAP_MAX
            for edge in edges
        ):
            return None
        chosen = _spanning_edges(edges, len(rects))
        out = inp.copy()
        for edge in chosen:
            _apply_edge(out, GEN_BG, color, rects, edge)
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return out


def _gen_attempt(n_rooms: int | None = None) -> np.ndarray | None:
    if n_rooms is None:
        n_rooms = random.randint(GEN_ROOM_COUNT_MIN, GEN_ROOM_COUNT_MAX)
    rects = _gen_place_rooms(n_rooms)
    if rects is None:
        return None

    rects = _gen_shift_rooms(rects)
    inp = _gen_draw_frames(rects)
    out = _gen_expected_output(inp, n_rooms)
    if out is None:
        return None
    return inp


def generate_input() -> np.ndarray:
    target_n = random.randint(GEN_ROOM_COUNT_MIN, GEN_ROOM_COUNT_MAX)
    for _ in range(GEN_TARGET_RETRIES):
        grid = _gen_attempt(target_n)
        if grid is not None:
            return grid
    for _ in range(GEN_FALLBACK_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b9630600 input")
