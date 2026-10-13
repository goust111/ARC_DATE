from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BLUE = 1
BACKGROUND = 0
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400
GRID_SIZE_RANGE = (18, 23)
HOST_SIZE_CHOICES = ((8, 8), (8, 9), (9, 8), (9, 9))
EDGE_NAMES = ("top", "bottom", "left", "right")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


SHAPE_LIBRARY = {
    "h3": np.array([(0, 0), (0, 1), (0, 2)], dtype=int),
    "v3": np.array([(0, 0), (1, 0), (2, 0)], dtype=int),
    "plus5": np.array([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], dtype=int),
    "t_up5": np.array([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)], dtype=int),
    "t_down5": np.array([(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)], dtype=int),
    "t_left5": np.array([(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)], dtype=int),
    "t_right5": np.array([(0, 1), (1, 0), (1, 1), (2, 1), (2, 0)], dtype=int),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    origin = coords.min(axis=0)
    norm = coords - origin
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    cells = tuple(sorted((int(r), int(c)) for r, c in norm))
    return h, w, cells


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _enclosed_holes(local_host_mask: np.ndarray) -> list[np.ndarray]:
    h, w = local_host_mask.shape
    passable = ~local_host_mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return _components(passable & ~outside)


def _hole_center(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _nearest_edge_or_none(center: tuple[int, int], host_shape: tuple[int, int]) -> str | None:
    r, c = map(int, center)
    h, w = map(int, host_shape)
    distances = {
        "top": r,
        "bottom": h - 1 - r,
        "left": c,
        "right": w - 1 - c,
    }
    best = min(distances.values())
    winners = [name for name, value in distances.items() if value == best]
    return winners[0] if len(winners) == 1 else None


def _nearest_edge(center: tuple[int, int], host_shape: tuple[int, int]) -> str:
    edge = _nearest_edge_or_none(center, host_shape)
    assert edge is not None, f"hole center has tied nearest edges for shape {host_shape}: {center}"
    return edge


def _draw_ray(grid: np.ndarray, start: tuple[int, int], edge: str, color: int) -> None:
    r, c = map(int, start)
    if edge == "top":
        for rr in range(r, -1, -1):
            grid[rr, c] = int(color)
    elif edge == "bottom":
        for rr in range(r, grid.shape[0]):
            grid[rr, c] = int(color)
    elif edge == "left":
        for cc in range(c, -1, -1):
            grid[r, cc] = int(color)
    else:
        for cc in range(c, grid.shape[1]):
            grid[r, cc] = int(color)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask)
    assert comps, "expected at least one component"
    areas = [len(comp) for comp in comps]
    best = int(np.argmax(areas))
    assert areas.count(areas[best]) == 1, "largest component must be unique"
    return comps[best]


def _marker_infos(grid: np.ndarray) -> list[dict]:
    infos = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BACKGROUND, BLUE)):
        for coords in _components(grid == color):
            infos.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "shape_key": _shape_key(coords),
                    "bbox": _bbox_coords(coords),
                }
            )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hole by shape-matching an external marker, then extend a same-colored ray from the hole toward its nearest host-bbox edge and crop the host.

    Input structure:
    - Black background surrounds one large blue 4-connected host object and several smaller external marker objects.
    - Inside the blue host's bounding box are 2-3 enclosed black hole components.
    - Each hole has a unique normalized shape, and each matching external marker has the same normalized shape but its own solid color.

    Rule:
    - Find the blue host object and the enclosed black holes inside its bounding box.
    - Match every hole to the external marker with the same normalized shape and fill the hole with that marker's color.
    - For each filled hole, use the hole bbox center to find the uniquely nearest side of the host bbox, then draw a 1-cell-wide ray of the same fill color from that center to that side, overwriting blue and black cells.
    - Return only the cropped host bounding box after the fills and rays.

    Key insight:
    - The hole's shape identifies its color through external marker matching, while the hole's position inside the host identifies the ray direction; both signals meet in the final ray drawing step.
    """
    g = np.asarray(grid, dtype=int)
    host = _largest_component(g == BLUE)
    host_box = _bbox_coords(host)
    crop = _crop_bbox(g, host_box)
    local_host = host - np.array([host_box[0], host_box[1]])
    local_host_mask = _mask_from_coords(local_host, crop.shape)
    holes = _enclosed_holes(local_host_mask)

    markers_by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[dict]] = defaultdict(list)
    for info in _marker_infos(g):
        markers_by_shape[info["shape_key"]].append(info)

    out = crop.copy()
    matched_colors: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    for hole in holes:
        key = _shape_key(hole)
        matches = markers_by_shape.get(key, [])
        assert len(matches) == 1, f"expected one external marker for hole shape {key}, got {len(matches)}"
        color = int(matches[0]["color"])
        matched_colors[key] = color
        out[hole[:, 0], hole[:, 1]] = color

    used_edges = set()
    for hole in holes:
        center = _hole_center(hole)
        edge = _nearest_edge(center, out.shape)
        assert edge not in used_edges, "hole directions must be distinct"
        used_edges.add(edge)
        _draw_ray(out, center, edge, matched_colors[_shape_key(hole)])

    return out


def _shape_dims(shape: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_coords(shape)
    return r1 - r0 + 1, c1 - c0 + 1


def _shifted(coords: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return coords + np.array([int(r0), int(c0)])


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _render_shape(shape: np.ndarray, color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.zeros((h, w), dtype=int)
    patch[shape[:, 0], shape[:, 1]] = int(color)
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    block = patch != 0
    dst[r0 : r0 + h, c0 : c0 + w][block] = patch[block]


def _sample_edges(count: int) -> tuple[str, ...]:
    edge_ids = RNG.choice(len(EDGE_NAMES), size=count, replace=False)
    return tuple(EDGE_NAMES[int(i)] for i in edge_ids)


def _place_holes(host_shape: tuple[int, int], edges: tuple[str, ...], shape_names: tuple[str, ...]) -> list[dict] | None:
    host_h, host_w = map(int, host_shape)
    chosen = []
    for _ in range(120):
        chosen.clear()
        boxes = []
        centers_by_edge = {}
        ok = True
        for edge, shape_name in zip(edges, shape_names):
            shape = SHAPE_LIBRARY[shape_name]
            sh, sw = _shape_dims(shape)
            if edge in ("top", "bottom"):
                c_low = 2
                c_high = host_w - sw - 2
                if c_low > c_high:
                    ok = False
                    break
                c0 = int(RNG.integers(c_low, c_high + 1))
                r0 = 1 if edge == "top" else host_h - sh - 1
            else:
                r_low = 2
                r_high = host_h - sh - 2
                if r_low > r_high:
                    ok = False
                    break
                r0 = int(RNG.integers(r_low, r_high + 1))
                c0 = 1 if edge == "left" else host_w - sw - 1

            placed = _shifted(shape, r0, c0)
            box = _bbox_coords(placed)
            center = _hole_center(placed)
            if _nearest_edge_or_none(center, host_shape) != edge:
                ok = False
                break
            if any(not _bbox_separated(box, other, pad=0) for other in boxes):
                ok = False
                break
            centers_by_edge[edge] = center
            boxes.append(box)
            chosen.append(
                {
                    "edge": edge,
                    "shape_name": shape_name,
                    "coords": placed,
                    "bbox": box,
                    "center": center,
                }
            )

        if not ok:
            continue
        if "top" in centers_by_edge and "bottom" in centers_by_edge:
            if centers_by_edge["top"][1] == centers_by_edge["bottom"][1]:
                continue
        if "left" in centers_by_edge and "right" in centers_by_edge:
            if centers_by_edge["left"][0] == centers_by_edge["right"][0]:
                continue
        hole_mask = np.zeros(host_shape, dtype=bool)
        for item in chosen:
            hole_mask[item["coords"][:, 0], item["coords"][:, 1]] = True
            center_r, center_c = item["center"]
            if not hole_mask[center_r, center_c]:
                ok = False
                break
        if not ok or len(_components(hole_mask)) != len(chosen):
            continue
        return [dict(item) for item in chosen]
    return None


def _place_patch_random(
    grid: np.ndarray,
    patch: np.ndarray,
    existing_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = patch.shape
    candidates = []
    for r0 in range(grid.shape[0] - h + 1):
        for c0 in range(grid.shape[1] - w + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(not _bbox_separated(box, other) for other in existing_boxes):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = candidates[int(RNG.integers(len(candidates)))]
    _paste(grid, patch, r0, c0)
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _expected_crop(host_shape: tuple[int, int], hole_specs: list[dict]) -> np.ndarray:
    out = np.full(host_shape, BLUE, dtype=int)
    for spec in hole_specs:
        coords = spec["coords"]
        local = coords.copy()
        out[local[:, 0], local[:, 1]] = int(spec["color"])
    for spec in hole_specs:
        _draw_ray(out, spec["center"], spec["edge"], int(spec["color"]))
    return out


def _semantic_analysis(grid: np.ndarray) -> dict:
    host = _largest_component(grid == BLUE)
    host_box = _bbox_coords(host)
    crop = _crop_bbox(grid, host_box)
    local_host = host - np.array([host_box[0], host_box[1]])
    local_mask = _mask_from_coords(local_host, crop.shape)
    holes = _enclosed_holes(local_mask)
    assert 2 <= len(holes) <= 3, "expected 2-3 holes"
    shape_keys = [_shape_key(hole) for hole in holes]
    assert len(set(shape_keys)) == len(shape_keys), "hole shapes must be unique"

    markers = _marker_infos(grid)
    markers_by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[dict]] = defaultdict(list)
    for marker in markers:
        markers_by_shape[marker["shape_key"]].append(marker)

    matched = []
    matched_keys = set()
    used_edges = set()
    for hole in holes:
        key = _shape_key(hole)
        assert len(markers_by_shape[key]) == 1, "hole-marker pairing must be unique"
        marker = markers_by_shape[key][0]
        matched_keys.add(id(marker))
        center = _hole_center(hole)
        edge = _nearest_edge(center, crop.shape)
        assert edge not in used_edges, "nearest edges must be distinct"
        used_edges.add(edge)
        matched.append((key, int(marker["color"]), edge))

    distractors = [marker for marker in markers if id(marker) not in matched_keys]
    assert len(distractors) <= 1, "expected at most one distractor marker"
    for distractor in distractors:
        assert distractor["shape_key"] not in shape_keys, "distractor must not match any hole"

    return {"host_box": host_box, "hole_count": len(holes), "matched": matched}


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    host_h, host_w = HOST_SIZE_CHOICES[int(RNG.integers(len(HOST_SIZE_CHOICES)))]
    hole_count = 2 if RNG.random() < 0.4 else 3
    edges = _sample_edges(hole_count)
    shape_names = tuple(RNG.choice(list(SHAPE_LIBRARY.keys()), size=hole_count, replace=False))
    hole_specs = _place_holes((host_h, host_w), edges, shape_names)
    if hole_specs is None:
        return None

    color_pool = list(RNG.choice(MARKER_COLORS, size=hole_count + 1, replace=False))
    for spec, color in zip(hole_specs, color_pool):
        spec["color"] = int(color)

    host_input = np.full((host_h, host_w), BLUE, dtype=int)
    for spec in hole_specs:
        host_input[spec["coords"][:, 0], spec["coords"][:, 1]] = BACKGROUND

    expected = _expected_crop((host_h, host_w), hole_specs)

    grid_h = int(RNG.integers(*GRID_SIZE_RANGE))
    grid_w = int(RNG.integers(*GRID_SIZE_RANGE))
    if grid_h < host_h + 6 or grid_w < host_w + 6:
        return None
    grid = np.zeros((grid_h, grid_w), dtype=int)

    host_r0 = int(RNG.integers(2, grid_h - host_h - 1))
    host_c0 = int(RNG.integers(2, grid_w - host_w - 1))
    _paste(grid, host_input, host_r0, host_c0)
    placed_boxes = [(host_r0, host_c0, host_r0 + host_h - 1, host_c0 + host_w - 1)]

    for spec in hole_specs:
        patch = _render_shape(SHAPE_LIBRARY[spec["shape_name"]], int(spec["color"]))
        placed = _place_patch_random(grid, patch, placed_boxes)
        if placed is None:
            return None
        placed_boxes.append(placed)

    if RNG.random() < 0.55:
        remaining = [name for name in SHAPE_LIBRARY if name not in shape_names]
        if not remaining:
            return None
        distractor_shape = SHAPE_LIBRARY[str(RNG.choice(remaining))]
        distractor_color = int(color_pool[-1])
        patch = _render_shape(distractor_shape, distractor_color)
        placed = _place_patch_random(grid, patch, placed_boxes)
        if placed is None:
            return None
        placed_boxes.append(placed)

    try:
        analysis = _semantic_analysis(grid)
        solved = main(grid)
    except AssertionError:
        return None
    assert analysis["hole_count"] == hole_count
    assert np.array_equal(solved, expected), "constructed instance must solve to the intended crop"
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
