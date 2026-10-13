import numpy as np

from collections import Counter, deque


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _ordered_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    point_set = set(points)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in points:
        ns = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (r + dr, c + dc)
            if q in point_set:
                ns.append(q)
        assert len(ns) == 2, "path component must be a simple 4-connected cycle"
        neighbors[(r, c)] = sorted(ns)

    start = min(points)
    ordered = [start]
    prev = start
    curr = neighbors[start][0]
    while curr != start:
        ordered.append(curr)
        n0, n1 = neighbors[curr]
        nxt = n0 if n0 != prev else n1
        prev, curr = curr, nxt
        assert len(ordered) <= len(points), "cycle traversal exceeded component size"
    assert len(ordered) == len(points), "cycle traversal missed component pixels"
    return ordered


def _clockwise_turn_counts(points: list[tuple[int, int]]) -> tuple[int, int]:
    cycle = _ordered_cycle(points)
    m = len(cycle)
    dirs = []
    for i in range(m):
        r0, c0 = cycle[i]
        r1, c1 = cycle[(i + 1) % m]
        dirs.append((r1 - r0, c1 - c0))

    left = 0
    right = 0
    for i in range(m):
        dr0, dc0 = dirs[i - 1]
        dr1, dc1 = dirs[i]
        cross = dr0 * dc1 - dc0 * dr1
        if cross > 0:
            left += 1
        elif cross < 0:
            right += 1

    area2 = 0
    for i in range(m):
        r0, c0 = cycle[i]
        r1, c1 = cycle[(i + 1) % m]
        area2 += r0 * c1 - c0 * r1
    assert area2 != 0, "cycle area must be non-zero"
    if area2 > 0:
        left, right = right, left
    return left, right


def _interior_background_mask(grid: np.ndarray, points: list[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    local_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    h, w = local_bg.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(local_bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(local_bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(local_bg[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))

    inside = local_bg & ~seen
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = inside
    return mask


def _find_lookup_table(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], set[tuple[int, int]], np.ndarray]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4 or h != w:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch[0, :] == color):
                continue
            if not np.all(patch[-1, :] == color):
                continue
            if not np.all(patch[:, 0] == color):
                continue
            if not np.all(patch[:, -1] == color):
                continue
            inner = patch[1:-1, 1:-1]
            if inner.size == 0:
                continue
            if np.any(inner == color) or np.any(inner == bg):
                continue
            candidates.append((r0, c1, h, color, set(comp), inner.copy()))

    assert candidates, "lookup table not found"
    r0, _c1, h, color, points, inner = min(candidates, key=lambda item: (item[0], -item[1], -item[2]))
    c0 = min(c for _, c in points)
    r1 = r0 + h - 1
    c1 = c0 + h - 1
    return color, (r0, c0, r1, c1), points, inner


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey orthogonal loop using a lookup-table color selected by its turn counts."""
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    border_color, table_bbox, table_points, lookup = _find_lookup_table(g, bg)

    out = g.copy()
    tr0, tc0, tr1, tc1 = table_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = bg

    path_mask = g == border_color
    for r, c in table_points:
        path_mask[r, c] = False

    for comp in _components(path_mask):
        left, right = _clockwise_turn_counts(comp)
        assert 0 <= left < lookup.shape[0] and 0 <= right < lookup.shape[1], "turn index outside lookup table"
        fill_color = int(lookup[left, right])
        inside = _interior_background_mask(g, comp, bg)
        out[inside] = fill_color

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

_BG = 0
_GREY = 5
_LOOKUP_SIZE = 8
_FILL_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_ATTEMPTS = 400


def _normalize_points(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    point_set = sorted(set(points))
    min_r = min(r for r, _ in point_set)
    min_c = min(c for _, c in point_set)
    normalized = sorted((r - min_r, c - min_c) for r, c in point_set)
    h = max(r for r, _ in normalized) + 1
    w = max(c for _, c in normalized) + 1
    return normalized, h, w


def _rasterize_cycle(vertices: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    points: list[tuple[int, int]] = []
    m = len(vertices)
    for i in range(m):
        r0, c0 = vertices[i]
        r1, c1 = vertices[(i + 1) % m]
        dr = int(np.sign(r1 - r0))
        dc = int(np.sign(c1 - c0))
        assert dr == 0 or dc == 0
        assert not (dr == 0 and dc == 0)
        r, c = r0, c0
        points.append((r, c))
        while (r, c) != (r1, c1):
            r += dr
            c += dc
            points.append((r, c))
    if points[-1] == points[0]:
        points.pop()
    return _normalize_points(points)


def _rotate_points(points: list[tuple[int, int]], turns: int) -> tuple[list[tuple[int, int]], int, int]:
    normalized, h, w = _normalize_points(points)
    pts = normalized
    for _ in range(turns % 4):
        pts = [(c, h - 1 - r) for r, c in pts]
        h, w = w, h
    return _normalize_points(pts)


def _flip_points(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    normalized, _h, w = _normalize_points(points)
    return _normalize_points([(r, w - 1 - c) for r, c in normalized])


def _shape_kind_0() -> tuple[list[tuple[int, int]], int, int]:
    h = int(_RNG.integers(4, 9))
    w = int(_RNG.integers(4, 10))
    return _rasterize_cycle([(0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)])


def _shape_kind_1() -> tuple[list[tuple[int, int]], int, int]:
    h = int(_RNG.integers(6, 11))
    w = int(_RNG.integers(6, 11))
    cut_h = int(_RNG.integers(2, h - 2))
    cut_w = int(_RNG.integers(2, w - 2))
    return _rasterize_cycle(
        [(0, cut_w), (0, w - 1), (h - 1, w - 1), (h - 1, 0), (cut_h, 0), (cut_h, cut_w)]
    )


def _shape_kind_2() -> tuple[list[tuple[int, int]], int, int]:
    h = int(_RNG.integers(6, 11))
    w = int(_RNG.integers(8, 13))
    notch_left = int(_RNG.integers(2, w - 5))
    notch_right = int(_RNG.integers(notch_left + 2, w - 2))
    depth = int(_RNG.integers(2, h - 2))
    return _rasterize_cycle(
        [
            (0, 0),
            (0, notch_left),
            (depth, notch_left),
            (depth, notch_right),
            (0, notch_right),
            (0, w - 1),
            (h - 1, w - 1),
            (h - 1, 0),
        ]
    )


def _shape_kind_3() -> tuple[list[tuple[int, int]], int, int]:
    h = int(_RNG.integers(8, 13))
    w = int(_RNG.integers(11, 16))
    cut_h = int(_RNG.integers(2, h - 3))
    cut_w = int(_RNG.integers(1, 4))
    notch_left = int(_RNG.integers(cut_w + 2, w - 5))
    notch_right = int(_RNG.integers(notch_left + 2, w - 2))
    depth = int(_RNG.integers(2, h - 2))
    return _rasterize_cycle(
        [
            (0, cut_w),
            (0, notch_left),
            (depth, notch_left),
            (depth, notch_right),
            (0, notch_right),
            (0, w - 1),
            (h - 1, w - 1),
            (h - 1, 0),
            (cut_h, 0),
            (cut_h, cut_w),
        ]
    )


def _make_shape(kind: int) -> dict[str, object]:
    builders = {
        0: _shape_kind_0,
        1: _shape_kind_1,
        2: _shape_kind_2,
        3: _shape_kind_3,
    }
    for _ in range(100):
        points, _h, _w = builders[kind]()
        turns = int(_RNG.integers(0, 4))
        points, _h, _w = _rotate_points(points, turns)
        if bool(_RNG.integers(0, 2)):
            points, _h, _w = _flip_points(points)
        left, right = _clockwise_turn_counts(points)
        local = np.full((_h, _w), _BG, dtype=int)
        for r, c in points:
            local[r, c] = _GREY
        inside = _interior_background_mask(local, points, _BG)
        if left == kind and right == kind + 4 and int(inside.sum()) >= 2:
            return {"points": points, "h": _h, "w": _w, "left": left, "right": right}
    raise RuntimeError(f"failed to build shape kind {kind}")


def _place_shapes(
    height: int, width: int, shapes: list[dict[str, object]], reserved: np.ndarray
) -> list[tuple[int, int]]:
    occupied = reserved.copy()
    placements: list[tuple[int, int]] = [(-1, -1)] * len(shapes)
    order = list(_RNG.permutation(len(shapes)))
    for idx in order:
        shape = shapes[idx]
        sh = int(shape["h"])
        sw = int(shape["w"])
        placed = False
        for _ in range(200):
            r = int(_RNG.integers(1, height - sh))
            c = int(_RNG.integers(1, width - sw))
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(height, r + sh + 1)
            c1 = min(width, c + sw + 1)
            if bool(occupied[r0:r1, c0:c1].any()):
                continue
            occupied[r0:r1, c0:c1] = True
            placements[idx] = (r, c)
            placed = True
            break
        if not placed:
            return []
    return placements


def _gen_attempt() -> np.ndarray | None:
    object_count = int(_RNG.integers(2, 5))
    height = int(_RNG.integers(18 + object_count * 2, 24 + object_count * 3))
    width = int(_RNG.integers(24 + object_count * 3, 32 + object_count * 4))

    table_span = _LOOKUP_SIZE + 2
    if height <= table_span + 2 or width <= table_span + 6:
        return None

    kinds = list(_RNG.permutation(4))[:object_count]
    shapes = [_make_shape(kind) for kind in kinds]

    grid = np.full((height, width), _BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    table_left = width - table_span
    reserved[: min(height, table_span + 1), max(0, table_left - 1) : width] = True

    placements = _place_shapes(height, width, shapes, reserved)
    if not placements:
        return None

    used_pairs = [(int(shape["left"]), int(shape["right"])) for shape in shapes]
    unique_pairs = []
    for pair in used_pairs:
        if pair not in unique_pairs:
            unique_pairs.append(pair)

    lookup = _RNG.choice(_FILL_COLORS, size=(_LOOKUP_SIZE, _LOOKUP_SIZE), replace=True)
    distinct_colors = list(_RNG.permutation(_FILL_COLORS))
    for idx, pair in enumerate(unique_pairs):
        lookup[pair] = int(distinct_colors[idx])

    grid[0:table_span, table_left:width] = _GREY
    grid[1 : table_span - 1, table_left + 1 : width - 1] = lookup

    for shape, (top, left) in zip(shapes, placements):
        for r, c in shape["points"]:
            grid[top + r, left + c] = _GREY

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_components(grid == _GREY)) != object_count + 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
