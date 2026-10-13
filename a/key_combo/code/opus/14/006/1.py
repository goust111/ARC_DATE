import numpy as np


BG = 0
MARKER = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 1500

SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 1), (2, 1), (2, 0)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _ring_pixels(
    center: tuple[int, int], layers: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    pts = {(cr, cc)} if _in_bounds(shape, cr, cc) else set()
    for k in range(1, layers + 1):
        r0, c0 = cr - k, cc - k
        r1, c1 = cr + k, cc + k
        for c in range(c0, c1 + 1):
            if _in_bounds(shape, r0, c):
                pts.add((r0, c))
            if _in_bounds(shape, r1, c):
                pts.add((r1, c))
        for r in range(r0, r1 + 1):
            if _in_bounds(shape, r, c0):
                pts.add((r, c0))
            if _in_bounds(shape, r, c1):
                pts.add((r, c1))
    return pts


def _slide_distance(coords: list[tuple[int, int]], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    if dr == 1:
        return h - 1 - max(r for r, _ in coords)
    if dr == -1:
        return min(r for r, _ in coords)
    if dc == 1:
        return w - 1 - max(c for _, c in coords)
    return min(c for _, c in coords)


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = sorted(int(v) for v in np.unique(g) if v not in (BG, MARKER))
    objects: list[dict] = []
    by_color: dict[int, dict] = {}

    for color in colors:
        comps = _component_coords(g == color)
        assert len(comps) == 1, "each object color must form exactly one component"
        coords = comps[0]
        box = _bbox(coords)
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        obj = {
            "color": color,
            "coords": coords,
            "bbox": box,
            "center": center,
            "marker": None,
            "direction": None,
        }
        objects.append(obj)
        by_color[color] = obj

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert len(markers) == len(objects), "expected one grey marker per object"

    for mr, mc in markers:
        nbrs = []
        for dr, dc in DIRS4:
            nr, nc = mr + dr, mc + dc
            if _in_bounds(g.shape, nr, nc) and g[nr, nc] not in (BG, MARKER):
                nbrs.append((nr, nc, int(g[nr, nc])))
        colors_seen = {color for _, _, color in nbrs}
        assert len(colors_seen) == 1, "marker must touch exactly one object color"
        nr, nc, color = nbrs[0]
        obj = by_color[color]
        assert obj["marker"] is None, "object must have exactly one marker"
        obj["marker"] = (mr, mc)
        obj["direction"] = (nr - mr, nc - mc)

    for obj in objects:
        assert obj["marker"] is not None and obj["direction"] is not None
        dr, dc = obj["direction"]
        dist = _slide_distance(obj["coords"], (dr, dc), (h, w))
        obj["distance"] = dist
        obj["layers"] = dist // 2
        obj["moved_coords"] = [(r + dr * dist, c + dc * dist) for r, c in obj["coords"]]

    objects.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each grey-marked object to the boundary, then grow concentric square borders from its starting center using half the slide distance.

    Input structure:
    - A black background contains 2-3 small 4-connected monochrome objects in distinct non-black, non-grey colors.
    - Each object has exactly one grey marker pixel on one cardinal side, adjacent to the object and identifying the source side.
    - Different objects, markers, and slide lanes are separated so every object can move straight to the boundary without initial foreground blockers.

    Rule:
    - For each object, read the grey marker as the source side and slide the whole object one step away from that marker until some object pixel first reaches the grid boundary.
    - Measure that translation distance D using the object's bounding-box-center movement; erase the grey marker.
    - At the object's original bounding-box center, draw the center pixel and then floor(D/2) concentric square borders of the same color, clipping any out-of-bounds border cells.
    - If border pixels from multiple objects overlap, process objects by bounding-box top-left from left to right, then top to bottom, so later objects overwrite earlier borders; finally paint the slid objects in their new positions.

    Key insight:
    - The grey marker does not survive in the output: it only encodes slide direction, and the resulting slide distance becomes the parameter that controls how many concentric square layers grow back at the starting center.
    """
    g = np.asarray(grid, dtype=int)
    objects = _analyze_objects(g)
    out = np.full_like(g, BG)

    for obj in objects:
        for r, c in _ring_pixels(obj["center"], obj["layers"], g.shape):
            out[r, c] = obj["color"]

    for obj in objects:
        for r, c in obj["moved_coords"]:
            out[r, c] = obj["color"]

    return out


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    out = sorted((r - r0, c - c0) for r, c in coords)
    assert _bbox(out) == (0, 0, r1 - r0, c1 - c0)
    return out


def _rotate(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _normalize([(c, -r) for r, c in coords])


def _random_shape() -> list[tuple[int, int]]:
    coords = list(SHAPES[int(np.random.randint(len(SHAPES)))])
    if np.random.rand() < 0.5:
        coords = [(r, -c) for r, c in coords]
        coords = _normalize(coords)
    for _ in range(int(np.random.randint(4))):
        coords = _rotate(coords)
    return _normalize(coords)


def _boundary_cells(coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 1:
        edge = min(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == edge]
    if dr == -1:
        edge = max(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == edge]
    if dc == 1:
        edge = min(c for _, c in coords)
        return [(r, c) for r, c in coords if c == edge]
    edge = max(c for _, c in coords)
    return [(r, c) for r, c in coords if c == edge]


def _path_cells(coords: list[tuple[int, int]], direction: tuple[int, int], distance: int) -> set[tuple[int, int]]:
    dr, dc = direction
    cells: set[tuple[int, int]] = set()
    for step in range(1, distance + 1):
        for r, c in coords:
            cells.add((r + dr * step, c + dc * step))
    return cells


def _candidate_positions(
    low_r: int,
    high_r: int,
    low_c: int,
    high_c: int,
    center_offset: tuple[int, int],
    preferred_center: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    picks: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(r: int, c: int) -> None:
        if (r, c) not in seen:
            picks.append((r, c))
            seen.add((r, c))

    if preferred_center is not None:
        base_r = preferred_center[0] - center_offset[0]
        base_c = preferred_center[1] - center_offset[1]
        for _ in range(18):
            rr = int(np.clip(base_r + np.random.randint(-2, 3), low_r, high_r))
            cc = int(np.clip(base_c + np.random.randint(-2, 3), low_c, high_c))
            add(rr, cc)

    for _ in range(48):
        rr = int(np.random.randint(low_r, high_r + 1))
        cc = int(np.random.randint(low_c, high_c + 1))
        add(rr, cc)

    return picks


def _touches_existing(group: set[tuple[int, int]], clearance: set[tuple[int, int]]) -> bool:
    return any(cell in clearance for cell in group)


def _add_clearance(clearance: set[tuple[int, int]], group: set[tuple[int, int]], shape: tuple[int, int]) -> None:
    h, w = shape
    for r, c in group:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    clearance.add((rr, cc))


def _place_object(
    h: int,
    w: int,
    coords: list[tuple[int, int]],
    color: int,
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    clearance: set[tuple[int, int]],
    reserved_paths: set[tuple[int, int]],
    preferred_center: tuple[int, int] | None,
) -> dict | None:
    anchors = _boundary_cells(coords, direction)
    anchor = anchors[int(np.random.randint(len(anchors)))]
    dr, dc = direction
    marker_rel = (anchor[0] - dr, anchor[1] - dc)
    rel_group = coords + [marker_rel]
    min_r = min(r for r, _ in rel_group)
    max_r = max(r for r, _ in rel_group)
    min_c = min(c for _, c in rel_group)
    max_c = max(c for _, c in rel_group)
    low_r = -min_r
    high_r = h - 1 - max_r
    low_c = -min_c
    high_c = w - 1 - max_c
    if low_r > high_r or low_c > high_c:
        return None

    obj_r1 = max(r for r, _ in coords)
    obj_c1 = max(c for _, c in coords)
    center_offset = (obj_r1 // 2, obj_c1 // 2)

    for top_r, top_c in _candidate_positions(low_r, high_r, low_c, high_c, center_offset, preferred_center):
        placed = [(top_r + r, top_c + c) for r, c in coords]
        marker = (top_r + marker_rel[0], top_c + marker_rel[1])
        group = set(placed) | {marker}
        if any(not _in_bounds((h, w), r, c) for r, c in group):
            continue
        if group & occupied or group & reserved_paths or _touches_existing(group, clearance):
            continue

        distance = _slide_distance(placed, direction, (h, w))
        if distance < 4:
            continue
        path = _path_cells(placed, direction, distance)
        if path & occupied or path & reserved_paths or marker in path:
            continue

        box = _bbox(placed)
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        return {
            "color": color,
            "coords": placed,
            "marker": marker,
            "direction": direction,
            "distance": distance,
            "layers": distance // 2,
            "bbox": box,
            "center": center,
            "path": path,
        }

    return None


def _assemble_grid(h: int, w: int, placed: list[dict]) -> np.ndarray:
    g = np.full((h, w), BG, dtype=int)
    for obj in placed:
        for r, c in obj["coords"]:
            g[r, c] = obj["color"]
        mr, mc = obj["marker"]
        g[mr, mc] = MARKER
    return g


def _ring_overlap_count(placed: list[dict], shape: tuple[int, int]) -> int:
    seen: set[tuple[int, int]] = set()
    overlap: set[tuple[int, int]] = set()
    for obj in placed:
        ring = _ring_pixels(obj["center"], obj["layers"], shape)
        overlap |= seen & ring
        seen |= ring
    return len(overlap)


def _output_is_clear(inp: np.ndarray, placed: list[dict]) -> bool:
    out = main(inp)
    ring_union: set[tuple[int, int]] = set()
    moved_union: set[tuple[int, int]] = set()
    for obj in placed:
        ring_union |= _ring_pixels(obj["center"], obj["layers"], inp.shape)
        moved_union |= set(obj["moved_coords"])
    if ring_union & moved_union:
        return False
    fill_ratio = float(np.count_nonzero(out)) / float(out.size)
    return 0.08 <= fill_ratio <= 0.50


def _generate_attempt(
    target_n_objects: int | None = None,
    target_overlap: bool | None = None,
) -> np.ndarray | None:
    h = int(np.random.randint(14, 19))
    w = int(np.random.randint(14, 19))
    n_objects = target_n_objects if target_n_objects is not None else (2 if np.random.rand() < 0.5 else 3)
    colors = list(np.random.choice(PALETTE, size=n_objects, replace=False))
    directions = [DIRS4[i] for i in np.random.permutation(len(DIRS4))[:n_objects]]
    target_overlap = target_overlap if target_overlap is not None else (np.random.rand() < 0.4)

    occupied: set[tuple[int, int]] = set()
    clearance: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    placed: list[dict] = []
    reference_center: tuple[int, int] | None = None

    for idx in range(n_objects):
        obj = None
        for _ in range(80):
            preferred = None
            if reference_center is not None and target_overlap and idx > 0 and np.random.rand() < 0.75:
                preferred = (
                    int(np.clip(reference_center[0] + np.random.randint(-4, 5), 2, h - 3)),
                    int(np.clip(reference_center[1] + np.random.randint(-4, 5), 2, w - 3)),
                )
            candidate = _place_object(
                h=h,
                w=w,
                coords=_random_shape(),
                color=colors[idx],
                direction=directions[idx],
                occupied=occupied,
                clearance=clearance,
                reserved_paths=reserved_paths,
                preferred_center=preferred,
            )
            if candidate is not None:
                obj = candidate
                break
        if obj is None:
            return None

        placed.append(obj)
        group = set(obj["coords"]) | {obj["marker"]}
        occupied |= group
        reserved_paths |= obj["path"]
        _add_clearance(clearance, group, (h, w))
        if reference_center is None:
            reference_center = obj["center"]

    inp = _assemble_grid(h, w, placed)

    try:
        analyzed = _analyze_objects(inp)
    except AssertionError:
        return None
    if len(analyzed) != n_objects:
        return None
    if any(obj["distance"] < 4 or obj["layers"] < 2 for obj in analyzed):
        return None

    overlap_count = _ring_overlap_count(analyzed, inp.shape)
    if target_overlap:
        if not (1 <= overlap_count <= 20):
            return None
    elif overlap_count != 0:
        return None

    if not _output_is_clear(inp, analyzed):
        return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        target_n_objects = 3 if np.random.rand() < 0.6 else 2
        target_overlap = np.random.rand() < 0.4
        inp = _generate_attempt(target_n_objects=target_n_objects, target_overlap=target_overlap)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
