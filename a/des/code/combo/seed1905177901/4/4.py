import numpy as np
from collections import deque


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRANSFORM_CODE = {
    "hflip": (1, 1),
    "vflip": (1, 0),
    "rot90": (0, 1),
}
CODE_TO_TRANSFORM = {value: key for key, value in TRANSFORM_CODE.items()}

GEN_BG = 5
GEN_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300
GEN_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _path_adjacency(points):
    pset = set(points)
    adj = {}
    for r, c in pset:
        nbrs = []
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in pset:
                nbrs.append((nr, nc))
        adj[(r, c)] = nbrs
    return adj


def _follow_path(points, start):
    adj = _path_adjacency(points)
    order = [start]
    prev = None
    cur = start
    while len(order) < len(points):
        nxts = [p for p in adj[cur] if p != prev]
        assert len(nxts) == 1, "path must be a single non-branching chain"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _normalize(points):
    min_r, min_c, _, _ = _bbox(points)
    return [(r - min_r, c - min_c) for r, c in points]


def _transform_points(points, kind: str):
    r0, c0, r1, c1 = _bbox(points)
    if kind == "hflip":
        out = [(r, c0 + c1 - c) for r, c in points]
    elif kind == "vflip":
        out = [(r0 + r1 - r, c) for r, c in points]
    elif kind == "rot90":
        cr2 = r0 + r1
        cc2 = c0 + c1
        out = []
        for r, c in points:
            nr2 = cr2 + cc2 - 2 * c
            nc2 = cc2 + 2 * r - cr2
            assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation must land on grid cells"
            out.append((nr2 // 2, nc2 // 2))
    else:
        raise ValueError(f"unknown transform: {kind}")

    assert len(set(out)) == len(points), "transform must stay one-to-one"
    for (r0p, c0p), (r1p, c1p) in zip(out, out[1:]):
        assert abs(r0p - r1p) + abs(c0p - c1p) == 1, "transform must preserve path adjacency"
    return out


def _parse_legend(grid: np.ndarray, bg: int):
    assert grid.shape[0] >= 3 and grid.shape[1] >= 3
    cycle = [int(grid[r, 0]) for r in range(3)]
    assert len(set(cycle)) == 3, "legend start colors must be distinct"
    assert all(color != bg for color in cycle), "legend colors must differ from background"

    mapping = {}
    for r, color in enumerate(cycle):
        code = tuple(int(grid[r, c] != bg) for c in range(1, 3))
        assert code in CODE_TO_TRANSFORM, f"unsupported legend code: {code}"
        mapping[color] = CODE_TO_TRANSFORM[code]

    assert set(mapping.values()) == set(TRANSFORM_CODE), "legend must expose all transform types"
    return cycle, mapping


def _sequence_matches_cycle(colors, cycle, cycle_index):
    if not colors or any(color not in cycle_index for color in colors):
        return False
    start = cycle_index[colors[0]]
    for i, color in enumerate(colors):
        if color != cycle[(start + i) % len(cycle)]:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract cyclic-color paths, look up a transform from the legend using the
    path's starting color, transform the path geometry, and repaint the same
    color cycle along the transformed direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    cycle, transform_by_color = _parse_legend(g, bg)
    cycle_index = {color: i for i, color in enumerate(cycle)}

    mask = g != bg
    mask[:3, :3] = False
    comps = _components(mask)

    out = np.full((h, w), bg, dtype=int)
    painted = set()
    for pts in comps:
        adj = _path_adjacency(pts)
        assert all(1 <= len(nbrs) <= 2 for nbrs in adj.values()), "component must be a simple path"
        endpoints = [p for p, nbrs in adj.items() if len(nbrs) == 1]
        assert len(endpoints) == 2, "path must have exactly two endpoints"

        order = _follow_path(pts, endpoints[0])
        colors_fwd = [int(g[r, c]) for r, c in order]
        colors_rev = list(reversed(colors_fwd))
        ok_fwd = _sequence_matches_cycle(colors_fwd, cycle, cycle_index)
        ok_rev = _sequence_matches_cycle(colors_rev, cycle, cycle_index)
        assert ok_fwd != ok_rev, "path direction must be uniquely determined by the color cycle"
        if not ok_fwd:
            order = list(reversed(order))
            colors_fwd = [int(g[r, c]) for r, c in order]

        start_color = colors_fwd[0]
        transform = transform_by_color[start_color]
        new_order = _transform_points(order, transform)
        start_idx = cycle_index[start_color]

        for i, (r, c) in enumerate(new_order):
            assert 0 <= r < h and 0 <= c < w, "transformed path must remain on the grid"
            assert (r, c) not in painted, "transformed paths must not overlap"
            painted.add((r, c))
            out[r, c] = cycle[(start_idx + i) % len(cycle)]

    return out


def _draw_legend(grid: np.ndarray, cycle, transform_by_color, marker_color: int):
    for r, color in enumerate(cycle):
        grid[r, 0] = color
        code = TRANSFORM_CODE[transform_by_color[color]]
        if code[0]:
            grid[r, 1] = marker_color
        if code[1]:
            grid[r, 2] = marker_color


def _valid_step(nr: int, nc: int, prev, used: set, h: int, w: int) -> bool:
    if not (0 <= nr < h and 0 <= nc < w):
        return False
    if (nr, nc) in used:
        return False
    for ar, ac in DIRS4:
        rr, cc = nr + ar, nc + ac
        if (rr, cc) in used and (rr, cc) != prev:
            return False
    return True


def _random_simple_path(local_h: int, local_w: int, target_len: int):
    for _ in range(30):
        start = (int(GEN_RNG.integers(local_h)), int(GEN_RNG.integers(local_w)))
        order = [start]
        used = {start}

        def dfs():
            if len(order) == target_len:
                return True
            cur = order[-1]
            candidates = []
            for nr, nc in _neighbors4(*cur):
                if _valid_step(nr, nc, cur, used, local_h, local_w):
                    degree_room = 0
                    for xr, xc in _neighbors4(nr, nc):
                        if _valid_step(xr, xc, (nr, nc), used | {(nr, nc)}, local_h, local_w):
                            degree_room += 1
                    candidates.append(((nr, nc), degree_room))
            if not candidates:
                return False
            order_idx = GEN_RNG.permutation(len(candidates))
            ranked = [candidates[i] for i in order_idx]
            ranked.sort(key=lambda item: item[1])
            for (nxt, _) in ranked:
                used.add(nxt)
                order.append(nxt)
                if dfs():
                    return True
                order.pop()
                used.remove(nxt)
            return False

        if dfs():
            return order
    return None


def _make_path_object(start_color: int, transform: str, cycle):
    for _ in range(80):
        local_h = int(GEN_RNG.integers(4, 8))
        local_w = int(GEN_RNG.integers(4, 8))
        max_len = min(12, local_h * local_w)
        if max_len < 5:
            continue
        target_len = int(GEN_RNG.integers(5, max_len + 1))
        order = _random_simple_path(local_h, local_w, target_len)
        if order is None:
            continue
        order = _normalize(order)
        try:
            transformed = _transform_points(order, transform)
        except AssertionError:
            continue
        if transformed == order:
            continue

        order = list(order)
        transformed = list(transformed)
        all_pts = order + transformed
        min_r, min_c, _, _ = _bbox(all_pts)
        shift_r, shift_c = -min_r, -min_c
        order = [(r + shift_r, c + shift_c) for r, c in order]
        transformed = [(r + shift_r, c + shift_c) for r, c in transformed]
        union = set(order) | set(transformed)
        _, _, max_r, max_c = _bbox(list(union))

        cycle_idx = cycle.index(start_color)
        input_colors = [cycle[(cycle_idx + i) % len(cycle)] for i in range(len(order))]
        output_colors = [cycle[(cycle_idx + i) % len(cycle)] for i in range(len(transformed))]

        in_render = {pt: color for pt, color in zip(order, input_colors)}
        out_render = {pt: color for pt, color in zip(transformed, output_colors)}
        if in_render == out_render:
            continue

        return {
            "start_color": start_color,
            "order": order,
            "transformed": transformed,
            "union": union,
            "foot_h": max_r + 1,
            "foot_w": max_c + 1,
        }
    return None


def _mark_reserved(reserved: np.ndarray, points):
    h, w = reserved.shape
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved[nr, nc] = True


def _try_place_objects(grid_h: int, grid_w: int, objects):
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_reserved(reserved, [(r, c) for r in range(0, 4) for c in range(0, 4) if r < grid_h and c < grid_w])

    placed = []
    for obj in sorted(objects, key=lambda item: len(item["union"]), reverse=True):
        success = False
        max_top = grid_h - obj["foot_h"]
        max_left = grid_w - obj["foot_w"]
        if max_top < 0 or max_left < 0:
            return None
        for _ in range(120):
            top = int(GEN_RNG.integers(0, max_top + 1))
            left = int(GEN_RNG.integers(0, max_left + 1))
            abs_union = [(top + r, left + c) for r, c in obj["union"]]
            if any(reserved[r, c] for r, c in abs_union):
                continue
            _mark_reserved(reserved, abs_union)
            placed.append(
                {
                    "start_color": obj["start_color"],
                    "order": [(top + r, left + c) for r, c in obj["order"]],
                }
            )
            success = True
            break
        if not success:
            return None
    return placed


def _build_candidate():
    cycle = list(GEN_RNG.choice(GEN_COLOR_POOL, size=3, replace=False))
    remaining = [color for color in GEN_COLOR_POOL if color not in cycle]
    marker_color = int(GEN_RNG.choice(remaining))

    transforms = list(TRANSFORM_CODE)
    GEN_RNG.shuffle(transforms)
    transform_by_color = {cycle[i]: transforms[i] for i in range(3)}

    num_paths = int(GEN_RNG.integers(2, 6))
    start_colors = list(GEN_RNG.permutation(cycle))
    while len(start_colors) < num_paths:
        start_colors.append(int(GEN_RNG.choice(cycle)))
    GEN_RNG.shuffle(start_colors)

    objects = []
    for start_color in start_colors:
        obj = _make_path_object(start_color, transform_by_color[start_color], cycle)
        if obj is None:
            return None
        objects.append(obj)

    total_area = sum(obj["foot_h"] * obj["foot_w"] for obj in objects)
    base = max(14, int(np.ceil(np.sqrt(total_area * 2.4))))
    grid_h = int(GEN_RNG.integers(base, min(base + 5, 24) + 1))
    grid_w = int(GEN_RNG.integers(base, min(base + 6, 25) + 1))

    placed = _try_place_objects(grid_h, grid_w, objects)
    if placed is None:
        return None

    grid = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    _draw_legend(grid, cycle, transform_by_color, marker_color)
    cycle_index = {color: i for i, color in enumerate(cycle)}
    for obj in placed:
        start_idx = cycle_index[obj["start_color"]]
        for i, (r, c) in enumerate(obj["order"]):
            grid[r, c] = cycle[(start_idx + i) % len(cycle)]

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
