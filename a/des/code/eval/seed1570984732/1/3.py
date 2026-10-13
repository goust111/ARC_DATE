from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
DIVIDER = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

LAYOUTS = [(2, 2), (2, 3), (3, 2)]
MIN_REGION_SIZE = 6
MAX_REGION_SIZE = 9
MAX_BRIDGE = 3
MAX_GENERATION_ATTEMPTS = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_divider_color(grid: np.ndarray, bg: int) -> int:
    height, width = grid.shape
    best_color = None
    best_score = -1
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        full_rows = sum(int(np.all(grid[r, :] == color)) for r in range(height))
        full_cols = sum(int(np.all(grid[:, c] == color)) for c in range(width))
        score = full_rows + full_cols
        if score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None and best_score > 0, "divider color must form full rows/cols"
    return int(best_color)


def _split_ranges(size: int, cuts: list[int]) -> list[tuple[int, int]]:
    boundaries = [-1] + sorted(cuts) + [size]
    parts = []
    for lo, hi in zip(boundaries, boundaries[1:], strict=False):
        start = lo + 1
        end = hi - 1
        if start <= end:
            parts.append((start, end))
    assert parts, "partition must leave at least one region"
    return parts


def _extract_regions(grid: np.ndarray, divider: int) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    height, width = grid.shape
    divider_rows = [r for r in range(height) if np.all(grid[r, :] == divider)]
    divider_cols = [c for c in range(width) if np.all(grid[:, c] == divider)]
    row_parts = _split_ranges(height, divider_rows)
    col_parts = _split_ranges(width, divider_cols)

    region_of = np.full((height, width), -1, dtype=int)
    regions: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_parts:
        for c0, c1 in col_parts:
            rid = len(regions)
            regions.append((int(r0), int(c0), int(r1), int(c1)))
            region_of[r0 : r1 + 1, c0 : c1 + 1] = rid
    return regions, region_of


def _is_boundary_cell(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = coord
    r0, c0, r1, c1 = bbox
    return r == r0 or r == r1 or c == c0 or c == c1


def _inward_direction(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = coord
    r0, c0, r1, c1 = bbox
    dirs = []
    if r == r0:
        dirs.append((1, 0))
    if r == r1:
        dirs.append((-1, 0))
    if c == c0:
        dirs.append((0, 1))
    if c == c1:
        dirs.append((0, -1))
    assert len(dirs) == 1, "portal must sit on exactly one side, not a corner"
    return dirs[0]


def _path_endpoints(mask: np.ndarray) -> set[tuple[int, int]]:
    coords = {tuple(map(int, p)) for p in np.argwhere(mask)}
    assert coords, "region must contain a path"

    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == coords, "path pixels must form one connected component"

    endpoints = set()
    for r, c in coords:
        degree = sum((r + dr, c + dc) in coords for dr, dc in DIRS4)
        assert degree in (1, 2), "region path must be a simple non-branching polyline"
        if degree == 1:
            endpoints.add((r, c))
    assert len(endpoints) == 2, "each regional polyline must have exactly two endpoints"
    return endpoints


def _find_portal_pairs(
    grid: np.ndarray,
    bg: int,
    divider: int,
    regions: list[tuple[int, int, int, int]],
    region_of: np.ndarray,
) -> dict[int, list[tuple[int, int]]]:
    portal_pairs: dict[int, list[tuple[int, int]]] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {bg, divider}):
        pts = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        if len(pts) != 2:
            continue
        region_ids = set()
        ok = True
        for r, c in pts:
            if sum(
                0 <= r + dr < grid.shape[0]
                and 0 <= c + dc < grid.shape[1]
                and int(grid[r + dr, c + dc]) == color
                for dr, dc in DIRS4
            ):
                ok = False
                break
            rid = int(region_of[r, c])
            if rid < 0:
                ok = False
                break
            if not _is_boundary_cell((r, c), regions[rid]):
                ok = False
                break
            region_ids.add(rid)
        if ok and len(region_ids) == 2:
            portal_pairs[color] = pts
    assert portal_pairs, "expected at least one portal pair"
    return portal_pairs


def _trace_bridge(
    grid: np.ndarray,
    line_mask: np.ndarray,
    endpoint_set: set[tuple[int, int]],
    portal: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
    dr, dc = _inward_direction(portal, bbox)
    axis = 0 if dr != 0 else 1
    r0, c0, r1, c1 = bbox
    r, c = portal
    painted = [portal]
    while True:
        r += dr
        c += dc
        assert r0 <= r <= r1 and c0 <= c <= c1, "portal must raycast into its region path"
        if bool(line_mask[r, c]):
            endpoint = (r, c)
            break
        assert int(grid[r, c]) == BG, "bridge corridor must be empty in the input"
        painted.append((r, c))
    assert endpoint in endpoint_set, "portal must connect to a path endpoint"
    return painted, endpoint, axis


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect region-local polylines through paired portals and redraw them with one shared color.

    Input structure:
    - Background is black.
    - Grey full rows/cols partition the grid into rectangular regions.
    - Each region contains one single-pixel orthogonal polyline.
    - Portal pixels are singleton boundary cells; each portal color appears exactly twice.

    Rule:
    - Pair portals by color.
    - At each portal, extend straight inward until the region's path endpoint is reached.
    - Paired portals must use perpendicular bridge axes.
    - Recolor every path pixel and every portal/bridge pixel to the most common path color.
    - Keep the grey divider lines unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    divider = _find_divider_color(g, bg)
    regions, region_of = _extract_regions(g, divider)
    portal_pairs = _find_portal_pairs(g, bg, divider, regions, region_of)
    portal_coords = {coord for coords in portal_pairs.values() for coord in coords}

    line_mask = (g != bg) & (g != divider)
    for r, c in portal_coords:
        line_mask[r, c] = False
    assert np.any(line_mask), "expected visible path pixels"

    line_colors = [int(g[r, c]) for r, c in np.argwhere(line_mask)]
    common_line_color = Counter(line_colors).most_common(1)[0][0]

    region_endpoints: list[set[tuple[int, int]]] = []
    for bbox in regions:
        r0, c0, r1, c1 = bbox
        local_mask = np.zeros_like(line_mask)
        local_mask[r0 : r1 + 1, c0 : c1 + 1] = line_mask[r0 : r1 + 1, c0 : c1 + 1]
        region_endpoints.append(_path_endpoints(local_mask))

    region_graph: dict[int, set[int]] = defaultdict(set)
    used_endpoints: dict[int, set[tuple[int, int]]] = defaultdict(set)
    out = g.copy()
    out[line_mask] = int(common_line_color)

    for color, pts in portal_pairs.items():
        axes = []
        pair_regions = []
        for portal in pts:
            rid = int(region_of[portal])
            bbox = regions[rid]
            painted, endpoint, axis = _trace_bridge(g, line_mask, region_endpoints[rid], portal, bbox)
            axes.append(axis)
            pair_regions.append(rid)
            used_endpoints[rid].add(endpoint)
            for r, c in painted:
                out[r, c] = int(common_line_color)
        assert len(set(axes)) == 2, f"paired portal color {color} must connect perpendicular axes"
        r1, r2 = pair_regions
        assert r1 != r2, "a portal pair must connect two different regions"
        region_graph[r1].add(r2)
        region_graph[r2].add(r1)

    assert len(region_graph) == len(regions), "all regions must participate in the global path"
    start_regions = [rid for rid, nbrs in region_graph.items() if len(nbrs) == 1]
    middle_regions = [rid for rid, nbrs in region_graph.items() if len(nbrs) == 2]
    assert len(start_regions) == 2 and len(start_regions) + len(middle_regions) == len(regions)
    seen = set()
    stack = [next(iter(region_graph))]
    while stack:
        rid = stack.pop()
        if rid in seen:
            continue
        seen.add(rid)
        stack.extend(region_graph[rid] - seen)
    assert len(seen) == len(regions), "portal graph must form one connected chain"

    for rid, endpoints in enumerate(region_endpoints):
        expected = 1 if rid in start_regions else 2
        assert len(used_endpoints[rid]) == expected, "portal bridges must land on distinct endpoints"
        assert used_endpoints[rid].issubset(endpoints)
    return out


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(row, c) for c in range(c0, c1 + 1)]
    assert a[1] == b[1]
    col = a[1]
    r0, r1 = sorted((a[0], b[0]))
    return [(r, col) for r in range(r0, r1 + 1)]


def _sample_portal(h: int, w: int, axis_type: str) -> dict[str, object]:
    if axis_type == "v":
        side = random.choice(["top", "bottom"])
        col = random.randint(1, w - 2)
        depth = random.randint(1, min(MAX_BRIDGE, h - 2))
        if side == "top":
            portal = (0, col)
            anchor = (depth, col)
            bridge = [(r, col) for r in range(0, depth)]
        else:
            portal = (h - 1, col)
            anchor = (h - 1 - depth, col)
            bridge = [(r, col) for r in range(h - depth, h)]
    else:
        side = random.choice(["left", "right"])
        row = random.randint(1, h - 2)
        depth = random.randint(1, min(MAX_BRIDGE, w - 2))
        if side == "left":
            portal = (row, 0)
            anchor = (row, depth)
            bridge = [(row, c) for c in range(0, depth)]
        else:
            portal = (row, w - 1)
            anchor = (row, w - 1 - depth)
            bridge = [(row, c) for c in range(w - depth, w)]
    return {
        "axis_type": axis_type,
        "side": side,
        "portal": portal,
        "anchor": anchor,
        "bridge": bridge,
    }


def _single_region_plan(h: int, w: int, axis_type: str) -> dict[str, object]:
    for _ in range(200):
        portal = _sample_portal(h, w, axis_type)
        ar, ac = portal["anchor"]
        if axis_type == "v":
            end = (random.randint(1, h - 2), random.randint(1, w - 2))
            if end[1] == ac:
                continue
            if abs(end[0] - ar) + abs(end[1] - ac) < 2:
                continue
            bend = (ar, end[1])
        else:
            end = (random.randint(1, h - 2), random.randint(1, w - 2))
            if end[0] == ar:
                continue
            if abs(end[0] - ar) + abs(end[1] - ac) < 2:
                continue
            bend = (end[0], ac)
        path = _segment((ar, ac), bend)
        if bend != end:
            path += _segment(bend, end)[1:]
        if len(set(path)) != len(path):
            continue
        return {"portals": [portal], "path": path}
    raise RuntimeError("failed to plan single-portal region")


def _double_region_plan(h: int, w: int) -> dict[str, object]:
    for _ in range(200):
        vertical = _sample_portal(h, w, "v")
        horizontal = _sample_portal(h, w, "h")
        vr, vc = vertical["anchor"]
        hr, hc = horizontal["anchor"]
        if vr == hr or vc == hc:
            continue
        bend = (vr, hc)
        path = _segment((vr, vc), bend) + _segment(bend, (hr, hc))[1:]
        if len(set(path)) != len(path):
            continue
        return {"portals": [vertical, horizontal], "path": path}
    raise RuntimeError("failed to plan two-portal region")


def _layout_boxes(row_sizes: list[int], col_sizes: list[int]) -> tuple[list[tuple[int, int, int, int]], tuple[int, int], list[int], list[int]]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1

    row_starts = []
    cur = 0
    divider_rows = []
    for i, size in enumerate(row_sizes):
        row_starts.append(cur)
        cur += size
        if i + 1 < len(row_sizes):
            divider_rows.append(cur)
            cur += 1

    col_starts = []
    cur = 0
    divider_cols = []
    for i, size in enumerate(col_sizes):
        col_starts.append(cur)
        cur += size
        if i + 1 < len(col_sizes):
            divider_cols.append(cur)
            cur += 1

    boxes = []
    for r0, hs in zip(row_starts, row_sizes, strict=False):
        for c0, ws in zip(col_starts, col_sizes, strict=False):
            boxes.append((r0, c0, r0 + hs - 1, c0 + ws - 1))
    return boxes, (height, width), divider_rows, divider_cols


def _choose_line_colors(lengths: list[int], available: list[int]) -> list[int]:
    dominant = available[0]
    colors = [dominant] * len(lengths)
    if len(available) >= 2 and random.random() < 0.7:
        total = sum(lengths)
        candidates = [i for i, length in enumerate(lengths) if total - length > length]
        if candidates:
            colors[random.choice(candidates)] = random.choice(available[1:])
    return colors


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = random.choice(LAYOUTS)
    row_sizes = [random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE) for _ in range(n_rows)]
    col_sizes = [random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE) for _ in range(n_cols)]
    boxes, shape, divider_rows, divider_cols = _layout_boxes(row_sizes, col_sizes)
    n_regions = len(boxes)
    n_pairs = n_regions - 1

    all_colors = random.sample(COLOR_POOL, len(COLOR_POOL))
    portal_colors = all_colors[:n_pairs]
    line_palette = all_colors[n_pairs:]
    assert len(line_palette) >= 1
    output_color = line_palette[0]

    chain = list(range(n_regions))
    random.shuffle(chain)
    first_axis_vertical = bool(random.getrandbits(1))

    pair_specs = []
    region_requests: dict[int, list[dict[str, object]]] = defaultdict(list)
    for i in range(n_pairs):
        earlier = chain[i]
        later = chain[i + 1]
        earlier_axis = "v" if first_axis_vertical else "h"
        later_axis = "h" if first_axis_vertical else "v"
        spec = {"color": portal_colors[i], "regions": (earlier, later)}
        pair_specs.append(spec)
        region_requests[earlier].append({"color": portal_colors[i], "axis_type": earlier_axis})
        region_requests[later].append({"color": portal_colors[i], "axis_type": later_axis})

    region_plans: list[dict[str, object]] = []
    for rid, bbox in enumerate(boxes):
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        requests = region_requests[rid]
        assert len(requests) in (1, 2)
        axis_types = sorted(req["axis_type"] for req in requests)
        if len(axis_types) == 1:
            plan = _single_region_plan(h, w, axis_types[0])
        else:
            assert axis_types == ["h", "v"]
            plan = _double_region_plan(h, w)
            plan["portals"].sort(key=lambda item: str(item["axis_type"]))
        portal_by_axis = {item["axis_type"]: item for item in plan["portals"]}
        ordered_portals = []
        for req in requests:
            portal = dict(portal_by_axis[req["axis_type"]])
            portal["color"] = int(req["color"])
            ordered_portals.append(portal)
        region_plans.append({"path": list(plan["path"]), "portals": ordered_portals})

    path_lengths = [len(plan["path"]) for plan in region_plans]
    region_colors = _choose_line_colors(path_lengths, line_palette)
    assert Counter(region_colors).most_common(1)[0][0] == output_color

    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    for r in divider_rows:
        inp[r, :] = DIVIDER
        out[r, :] = DIVIDER
    for c in divider_cols:
        inp[:, c] = DIVIDER
        out[:, c] = DIVIDER

    portal_lookup: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for rid, bbox in enumerate(boxes):
        r0, c0, _r1, _c1 = bbox
        line_color = region_colors[rid]
        plan = region_plans[rid]
        path = [tuple(map(int, p)) for p in plan["path"]]
        for lr, lc in path:
            inp[r0 + lr, c0 + lc] = int(line_color)
            out[r0 + lr, c0 + lc] = int(output_color)
        for portal in plan["portals"]:
            pr, pc = portal["portal"]
            color = int(portal["color"])
            inp[r0 + pr, c0 + pc] = color
            out[r0 + pr, c0 + pc] = int(output_color)
            for br, bc in portal["bridge"][1:]:
                out[r0 + br, c0 + bc] = int(output_color)
            portal_lookup[color].append((rid, r0 + pr, c0 + pc))

    assert all(len(v) == 2 for v in portal_lookup.values())

    for color, entries in portal_lookup.items():
        (rid1, r1, c1), (rid2, r2, c2) = entries
        assert rid1 != rid2
        box1 = boxes[rid1]
        box2 = boxes[rid2]
        axis1 = 0 if _inward_direction((r1, c1), box1)[0] != 0 else 1
        axis2 = 0 if _inward_direction((r2, c2), box2)[0] != 0 else 1
        assert axis1 != axis2, f"portal pair {color} must rotate direction by 90 degrees"

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp, expected = _build_case()
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid portal-chain puzzle")
