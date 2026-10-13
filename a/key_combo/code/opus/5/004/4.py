from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
MARKER_COLORS = (4, 6, 8, 9)

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
]

MAX_GENERATE_ATTEMPTS = 400
ANGLE_EPS = 1e-6


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    moves = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in moves:
                nr, nc = cr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _clockwise_order(frame_infos: list[dict]) -> list[int]:
    centers = [info["center"] for info in frame_infos]
    mean_r = sum(r for r, _ in centers) / len(centers)
    mean_c = sum(c for _, c in centers) / len(centers)
    indexed = []
    for idx, (r, c) in enumerate(centers):
        angle = math.atan2(r - mean_r, c - mean_c)
        indexed.append((angle, idx))
    indexed.sort()
    for i in range(len(indexed) - 1):
        assert abs(indexed[i + 1][0] - indexed[i][0]) > ANGLE_EPS, "frame angles must be distinct"
    return [idx for _angle, idx in indexed]


def _border_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _validate_blue_cycle(grid: np.ndarray, frame_infos: list[dict]) -> None:
    blue_comps = _components(grid == BLUE, connectivity=8)
    assert len(blue_comps) == len(frame_infos), "connector count must match frame count"
    frame_borders = [info["border_coords"] for info in frame_infos]
    degrees = [0] * len(frame_infos)
    touched_edges: set[tuple[int, int]] = set()
    for comp in blue_comps:
        touched: list[int] = []
        for idx, border in enumerate(frame_borders):
            ok = False
            for r, c in comp:
                for dr, dc in NEIGHBORS8:
                    if (r + dr, c + dc) in border:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                touched.append(idx)
        assert len(touched) == 2 and touched[0] != touched[1], "each connector must join two frames"
        a, b = sorted(touched)
        assert (a, b) not in touched_edges, "duplicate connector edge"
        touched_edges.add((a, b))
        degrees[a] += 1
        degrees[b] += 1
    assert all(deg == 2 for deg in degrees), "cycle degree must be two at every frame"

    graph = {i: [] for i in range(len(frame_infos))}
    for a, b in touched_edges:
        graph[a].append(b)
        graph[b].append(a)
    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph[node])
    assert len(seen) == len(frame_infos), "blue connector graph must be one cycle"


def _analyze_fold_only(local: np.ndarray) -> dict:
    n = local.shape[0]
    assert local.shape == (n, n)
    assert n in (4, 5), "interior size must be 4 or 5"

    rod_coords = [(int(r), int(c)) for r, c in np.argwhere(np.isin(local, (GREEN, GREY, ORANGE)))]
    assert len(rod_coords) == n, "rod must span the full interior width or height"
    assert len(_components(np.isin(local, (GREEN, GREY, ORANGE)), connectivity=4)) == 1, "rod must be contiguous"

    grey_coords = [(int(r), int(c)) for r, c in np.argwhere(local == GREY)]
    hinge_coords = [(int(r), int(c)) for r, c in np.argwhere(local == ORANGE)]
    assert len(grey_coords) == 1 and len(hinge_coords) == 1, "rod markers must be unique"
    endpoint = grey_coords[0]
    hinge = hinge_coords[0]

    rows = {r for r, _ in rod_coords}
    cols = {c for _, c in rod_coords}
    rod_set = set(rod_coords)
    tail: list[tuple[int, int]]
    folded: list[tuple[int, int]]
    if len(rows) == 1:
        row = next(iter(rows))
        assert set(cols) == set(range(n)), "horizontal rod must cover the full interior row"
        assert endpoint[0] == row and hinge[0] == row
        fixed_sign = _sign(endpoint[1] - hinge[1])
        assert fixed_sign != 0
        tail_step = (0, -fixed_sign)
        up_space = hinge[0]
        down_space = n - 1 - hinge[0]
        assert up_space != down_space, "fold direction must be unique"
        fold_step = (-1, 0) if up_space > down_space else (1, 0)
    else:
        assert len(cols) == 1
        col = next(iter(cols))
        assert set(rows) == set(range(n)), "vertical rod must cover the full interior column"
        assert endpoint[1] == col and hinge[1] == col
        fixed_sign = _sign(endpoint[0] - hinge[0])
        assert fixed_sign != 0
        tail_step = (-fixed_sign, 0)
        left_space = hinge[1]
        right_space = n - 1 - hinge[1]
        assert left_space != right_space, "fold direction must be unique"
        fold_step = (0, -1) if left_space > right_space else (0, 1)

    tail = []
    step = 1
    while True:
        rr = hinge[0] + tail_step[0] * step
        cc = hinge[1] + tail_step[1] * step
        if (rr, cc) not in rod_set:
            break
        assert local[rr, cc] == GREEN, "tail may contain only green pixels"
        tail.append((rr, cc))
        step += 1
    assert tail, "tail length must be positive"

    folded = []
    for step in range(1, len(tail) + 1):
        rr = hinge[0] + fold_step[0] * step
        cc = hinge[1] + fold_step[1] * step
        assert 0 <= rr < n and 0 <= cc < n, "folded tail must stay in bounds"
        folded.append((rr, cc))
    assert folded[-1][0] in (0, n - 1) or folded[-1][1] in (0, n - 1), "folded tail must reach a wall"
    assert len(set(folded)) == len(folded), "folded tail must not self-overlap"

    barrier = {endpoint, hinge}
    for cell in rod_coords:
        if cell not in tail:
            barrier.add(cell)
    barrier.update(folded)
    walkable = np.ones((n, n), dtype=bool)
    for r, c in barrier:
        walkable[r, c] = False
    comps = _components(walkable, connectivity=4)
    assert len(comps) == 2, "folded rod must split the interior into exactly two regions"

    return {
        "tail_local": tail,
        "folded_local": folded,
        "barrier_local": barrier,
        "walkable_components": comps,
        "rod_set": rod_set,
        "endpoint": endpoint,
        "hinge": hinge,
    }


def _analyze_frame(local: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    n = local.shape[0]
    fold = _analyze_fold_only(local)

    marker_coords = [(int(r), int(c)) for r, c in np.argwhere(np.isin(local, MARKER_COLORS))]
    assert len(marker_coords) == 1, "each frame must contain exactly one marker pixel"
    marker_r, marker_c = marker_coords[0]
    marker_color = int(local[marker_r, marker_c])
    assert (marker_r, marker_c) not in fold["rod_set"], "marker must start in background"
    assert (marker_r, marker_c) not in set(fold["folded_local"]), "marker cannot lie on the folded tail"

    selected_region = None
    for comp in fold["walkable_components"]:
        if (marker_r, marker_c) in comp:
            selected_region = comp
            break
    assert selected_region is not None, "marker must lie in one folded region"
    fill_candidates = [cell for cell in selected_region if cell != (marker_r, marker_c)]
    assert fill_candidates, "selected region must contain fillable cells"

    return {
        "bbox": bbox,
        "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        "border_coords": _border_coords(bbox),
        "local_marker": (marker_r, marker_c),
        "marker_color": marker_color,
        "tail_local": fold["tail_local"],
        "folded_local": fold["folded_local"],
        "selected_region_local": selected_region,
    }


def _extract_frame_infos(grid: np.ndarray, validate_blue: bool = True) -> list[dict]:
    red_comps = _components(grid == RED, connectivity=4)
    frame_infos: list[dict] = []
    for comp in red_comps:
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        size = r1 - r0 + 1
        if size != c1 - c0 + 1 or size not in (6, 7):
            continue
        expected = 4 * size - 4
        if len(comp) != expected:
            continue
        ok = True
        comp_set = set(comp)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border != ((rr, cc) in comp_set):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        local = grid[r0 + 1 : r1, c0 + 1 : c1]
        frame_infos.append(_analyze_frame(local, bbox))

    assert 3 <= len(frame_infos) <= 4, "expected 3 or 4 valid frames"
    marker_colors = [info["marker_color"] for info in frame_infos]
    assert len(set(marker_colors)) == len(marker_colors), "frame marker colors must be distinct"
    if validate_blue:
        _validate_blue_cycle(grid, frame_infos)
    return frame_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the marked rod inside each red frame, then fill the marker side with the next clockwise frame's marker color.

    Input structure:
    - The grid has black background, 3-4 hollow red square frames, and blue connector segments linking the frames into one cycle.
    - Inside each frame, a straight rod spans the full inner width or height using green pixels plus one gray endpoint marker and one orange hinge marker.
    - Each frame also contains exactly one single marker pixel of color 4, 6, 8, or 9 in a background cell.

    Rule:
    - In every frame, keep the gray-to-orange segment fixed and rotate the green tail beyond the orange hinge by 90 degrees toward the side with more interior space.
    - Treat the folded L-shaped rod as a barrier together with the frame wall; this splits the frame interior into exactly two 4-connected regions.
    - Sort frames clockwise by their centers around the shared centroid, and fill the marker-containing region of each frame with the single-marker color from the next clockwise frame, wrapping around.
    - Keep the red walls, blue connectors, gray endpoint, orange hinge, green rod cells, and the original single marker pixels unchanged.

    Key insight:
    - The result depends on both local folding geometry and global clockwise color transfer: the fold decides which cells belong to the chosen compartment, while the ring order decides its fill color.
    """
    g = np.asarray(grid, dtype=int)
    frame_infos = _extract_frame_infos(g, validate_blue=True)
    order = _clockwise_order(frame_infos)

    out = g.copy()
    for info in frame_infos:
        r0, c0, _r1, _c1 = info["bbox"]
        for rr, cc in info["tail_local"]:
            out[r0 + 1 + rr, c0 + 1 + cc] = BG
        for rr, cc in info["folded_local"]:
            out[r0 + 1 + rr, c0 + 1 + cc] = GREEN

    for k, idx in enumerate(order):
        next_idx = order[(k + 1) % len(order)]
        fill_color = frame_infos[next_idx]["marker_color"]
        info = frame_infos[idx]
        r0, c0, _r1, _c1 = info["bbox"]
        marker = info["local_marker"]
        for rr, cc in info["selected_region_local"]:
            if (rr, cc) != marker:
                out[r0 + 1 + rr, c0 + 1 + cc] = fill_color
    return out


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    dr = _sign(er - sr)
    dc = _sign(ec - sc)
    steps = max(abs(er - sr), abs(ec - sc))
    assert sr == er or sc == ec or abs(er - sr) == abs(ec - sc), "only horizontal, vertical, or 45-degree lines are supported"
    return [(sr + dr * k, sc + dc * k) for k in range(steps + 1)]


def _fold_patterns(n: int) -> list[dict]:
    mid = (n - 1) / 2.0
    patterns: list[dict] = []
    for h in range(1, n - 1):
        if h < mid:
            patterns.append({"axis": "h", "endpoint_side": "left", "hinge": (n - 1 - h, h)})
            patterns.append({"axis": "h", "endpoint_side": "right", "hinge": (h, h)})
            patterns.append({"axis": "v", "endpoint_side": "top", "hinge": (h, n - 1 - h)})
            patterns.append({"axis": "v", "endpoint_side": "bottom", "hinge": (h, h)})
        if h > mid:
            patterns.append({"axis": "h", "endpoint_side": "left", "hinge": (h, h)})
            patterns.append({"axis": "h", "endpoint_side": "right", "hinge": (n - 1 - h, h)})
            patterns.append({"axis": "v", "endpoint_side": "top", "hinge": (h, h)})
            patterns.append({"axis": "v", "endpoint_side": "bottom", "hinge": (h, n - 1 - h)})
    dedup: dict[tuple[str, str, tuple[int, int]], dict] = {}
    for pat in patterns:
        key = (pat["axis"], pat["endpoint_side"], pat["hinge"])
        if key in dedup:
            continue
        try:
            local = np.zeros((n, n), dtype=int)
            rod, endpoint, hinge = _build_input_rod(n, pat)
            for r, c in rod:
                local[r, c] = GREEN
            local[endpoint] = GREY
            local[hinge] = ORANGE
            _analyze_fold_only(local)
            dedup[key] = pat
        except AssertionError:
            continue
    return list(dedup.values())


def _build_input_rod(n: int, pattern: dict) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    hinge = pattern["hinge"]
    axis = pattern["axis"]
    endpoint_side = pattern["endpoint_side"]
    rod: list[tuple[int, int]] = []
    if axis == "h":
        row = hinge[0]
        rod = [(row, c) for c in range(n)]
        endpoint = (row, 0) if endpoint_side == "left" else (row, n - 1)
    else:
        col = hinge[1]
        rod = [(r, col) for r in range(n)]
        endpoint = (0, col) if endpoint_side == "top" else (n - 1, col)
    return rod, endpoint, hinge


def _compute_regions_for_pattern(n: int, pattern: dict) -> dict:
    local = np.zeros((n, n), dtype=int)
    rod, endpoint, hinge = _build_input_rod(n, pattern)
    for r, c in rod:
        local[r, c] = GREEN
    local[endpoint] = GREY
    local[hinge] = ORANGE
    fold = _analyze_fold_only(local)
    rod_set = set(rod)
    valid_regions = []
    for comp in fold["walkable_components"]:
        bg_candidates = [cell for cell in comp if cell not in rod_set]
        if len(bg_candidates) >= 2:
            valid_regions.append(bg_candidates)
    assert len(valid_regions) == 2, "generator requires two nondegenerate marker regions"
    return {
        "local": local,
        "rod": rod,
        "endpoint": endpoint,
        "hinge": hinge,
        "fold_info": fold,
        "valid_regions": valid_regions,
    }


def _draw_frame_border(grid: np.ndarray, top: int, left: int, size: int) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = RED
    grid[bottom, left : right + 1] = RED
    grid[top : bottom + 1, left] = RED
    grid[top : bottom + 1, right] = RED


def _orth_frame_connectors(size: int, gap_h: int, gap_v: int) -> list[list[tuple[int, int]]]:
    mid = size // 2
    return [
        _line_points((mid, size), (mid, size + gap_h - 1)),
        _line_points((size, size + gap_h + mid), (size + gap_v - 1, size + gap_h + mid)),
        _line_points((size + gap_v + mid, size + gap_h - 1), (size + gap_v + mid, size)),
        _line_points((size + gap_v - 1, mid), (size, mid)),
    ]


def _triangle_right_connectors(size: int, gap: int) -> list[list[tuple[int, int]]]:
    mid = size // 2
    return [
        _line_points((mid, size), (mid, size + gap - 1)),
        _line_points((size, size + gap + mid), (size + gap - 1, size + gap + mid)),
        _line_points((size, size), (size + gap - 1, size + gap - 1)),
    ]


def _triangle_left_connectors(size: int, gap: int) -> list[list[tuple[int, int]]]:
    mid = size // 2
    return [
        _line_points((mid, size), (mid, size + gap - 1)),
        _line_points((size, mid), (size + gap - 1, mid)),
        _line_points((size, size + gap - 1), (size + gap - 1, size)),
    ]


def _layout_records(size: int) -> list[dict]:
    gap_h = random.randint(4, 8)
    gap_v = random.randint(4, 8)
    gap_t = random.randint(4, 8)
    return [
        {
            "frame_count": 4,
            "shape": (2 * size + gap_v, 2 * size + gap_h),
            "frames": [(0, 0), (0, size + gap_h), (size + gap_v, size + gap_h), (size + gap_v, 0)],
            "connectors": _orth_frame_connectors(size, gap_h, gap_v),
        },
        {
            "frame_count": 3,
            "shape": (2 * size + gap_t, 2 * size + gap_t),
            "frames": [(0, 0), (0, size + gap_t), (size + gap_t, size + gap_t)],
            "connectors": _triangle_right_connectors(size, gap_t),
        },
        {
            "frame_count": 3,
            "shape": (2 * size + gap_t, 2 * size + gap_t),
            "frames": [(0, 0), (0, size + gap_t), (size + gap_t, 0)],
            "connectors": _triangle_left_connectors(size, gap_t),
        },
    ]


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    infos = _extract_frame_infos(inp, validate_blue=True)
    order = _clockwise_order(infos)
    expected = inp.copy()
    for info in infos:
        r0, c0, _r1, _c1 = info["bbox"]
        for rr, cc in info["tail_local"]:
            expected[r0 + 1 + rr, c0 + 1 + cc] = BG
        for rr, cc in info["folded_local"]:
            expected[r0 + 1 + rr, c0 + 1 + cc] = GREEN
    for k, idx in enumerate(order):
        next_idx = order[(k + 1) % len(order)]
        fill_color = infos[next_idx]["marker_color"]
        info = infos[idx]
        marker = info["local_marker"]
        r0, c0, _r1, _c1 = info["bbox"]
        for rr, cc in info["selected_region_local"]:
            if (rr, cc) != marker:
                expected[r0 + 1 + rr, c0 + 1 + cc] = fill_color
    assert np.array_equal(expected, out), "output must match the semantic transformation"

    changed = np.argwhere(inp != out)
    assert len(changed) > 0, "output must differ from input"
    for idx in order:
        info = infos[idx]
        r0, c0, _r1, _c1 = info["bbox"]
        marker = info["local_marker"]
        assert out[r0 + 1 + marker[0], c0 + 1 + marker[1]] == inp[r0 + 1 + marker[0], c0 + 1 + marker[1]]


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([6, 7])
    n = size - 2
    layouts = _layout_records(size)
    layout = random.choice(layouts)
    base_h, base_w = layout["shape"]

    max_h = min(28, base_h + random.randint(2, 8))
    max_w = min(28, base_w + random.randint(2, 8))
    if max_h < base_h or max_w < base_w:
        return None
    top_margin = random.randint(0, max_h - base_h)
    left_margin = random.randint(0, max_w - base_w)
    grid = np.zeros((max_h, max_w), dtype=int)

    frame_count = layout["frame_count"]
    marker_palette = random.sample(list(MARKER_COLORS), frame_count)
    pattern_pool = _fold_patterns(n)
    random.shuffle(pattern_pool)

    frame_records = []
    for idx, (top0, left0) in enumerate(layout["frames"]):
        top = top_margin + top0
        left = left_margin + left0
        _draw_frame_border(grid, top, left, size)
        pattern = random.choice(pattern_pool)
        prepared = _compute_regions_for_pattern(n, pattern)
        marker_region = random.choice(prepared["valid_regions"])
        marker_cell = random.choice(marker_region)

        local = np.zeros((n, n), dtype=int)
        for r, c in prepared["rod"]:
            local[r, c] = GREEN
        local[prepared["endpoint"]] = GREY
        local[prepared["hinge"]] = ORANGE
        local[marker_cell] = marker_palette[idx]
        grid[top + 1 : top + size - 1, left + 1 : left + size - 1] = local

        frame_records.append(
            {
                "top": top,
                "left": left,
                "marker_color": marker_palette[idx],
                "marker_cell": marker_cell,
            }
        )

    all_connector_cells: set[tuple[int, int]] = set()
    for segment in layout["connectors"]:
        for rr0, cc0 in segment:
            rr = top_margin + rr0
            cc = left_margin + cc0
            if not _in_bounds(grid.shape, rr, cc):
                return None
            if grid[rr, cc] != BG:
                return None
            all_connector_cells.add((rr, cc))
    for rr, cc in all_connector_cells:
        grid[rr, cc] = BLUE

    try:
        analyzed = _extract_frame_infos(grid, validate_blue=True)
        if len(analyzed) != frame_count:
            return None
        out = main(grid)
        _semantic_check(grid, out)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
