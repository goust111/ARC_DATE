import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4)
ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0:
        step = (0, _sign(dc))
        n = abs(dc)
    elif dc == 0:
        step = (_sign(dr), 0)
        n = abs(dr)
    elif abs(dr) == abs(dc):
        step = (_sign(dr), _sign(dc))
        n = abs(dr)
    else:
        return None
    return [(r0 + i * step[0], c0 + i * step[1]) for i in range(n + 1)]


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _angle_sort(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    decorated = []
    for r, c in points:
        angle = float(np.arctan2(-(r - center[0]), c - center[1]) % (2 * np.pi))
        dist2 = (r - center[0]) ** 2 + (c - center[1]) ** 2
        decorated.append((angle, dist2, (r, c)))
    decorated.sort(key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
    return [pt for _angle, _dist2, pt in decorated]


def _vertex_groups(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    vertices = [tuple(map(int, p)) for p in np.argwhere(grid == BLUE)]
    if not vertices:
        return []

    allowed = {BG, ORANGE, *MARKER_COLORS}
    adj = {pt: set() for pt in vertices}
    for i, a in enumerate(vertices):
        for b in vertices[i + 1 :]:
            pts = _line_points(a, b)
            if pts is None:
                continue
            interior = pts[1:-1]
            if all(int(grid[r, c]) in allowed for r, c in interior):
                adj[a].add(b)
                adj[b].add(a)

    groups: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in vertices:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        group: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(sorted(group))
    return groups


def _edge_draw_status(
    grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]
) -> tuple[str, list[tuple[int, int]] | None]:
    pts = _line_points(a, b)
    if pts is None:
        return "unaligned", None
    interior = pts[1:-1]
    has_orange = any(int(grid[r, c]) == ORANGE for r, c in interior)
    if has_orange:
        return "blocked", pts
    allowed = {BG, *MARKER_COLORS}
    if all(int(grid[r, c]) in allowed for r, c in interior):
        return "draw", pts
    return "occupied", pts


def _draw_outline(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    groups = _vertex_groups(out)
    edge_info: list[dict] = []
    for group_idx, group in enumerate(groups):
        cycle = _angle_sort(group)
        if len(cycle) < 2:
            continue
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i + 1) % len(cycle)]
            status, pts = _edge_draw_status(out, a, b)
            edge_info.append(
                {
                    "group_idx": group_idx,
                    "a": a,
                    "b": b,
                    "status": status,
                    "points": pts,
                }
            )
            if status == "draw" and pts is not None:
                for r, c in pts:
                    out[r, c] = BLUE
    return out, groups, edge_info


def _external_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not ext[r, c]:
                ext[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(grid.shape, nr, nc) and not ext[nr, nc] and grid[nr, nc] == BG:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def _enclosed_background_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    external = _external_background(grid)
    enclosed = (grid == BG) & (~external)
    return _components_from_mask(enclosed)


def _adjacent_markers(
    grid: np.ndarray, comp: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    markers: dict[tuple[int, int], int] = {}
    for r, c in comp:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) in MARKER_COLORS:
                markers[(nr, nc)] = int(grid[nr, nc])
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect angle-sorted blue vertices into drawable polygon outlines and fill each closed interior by its contained marker color.

    Input structure:
    - Black background contains 6-10 blue vertex pixels that belong to one polygon or to two spatially separate polygons.
    - Orange pixels lie on some candidate horizontal, vertical, or 45-degree edges and can block those edges from being drawn.
    - Each truly closed region contains exactly one single-pixel marker whose color is red, green, or yellow.

    Rule:
    - For each blue-vertex group, sort the vertices by polar angle around that group's centroid and consider consecutive pairs cyclically.
    - Draw only edges whose endpoints are horizontal, vertical, or 45-degree diagonal aligned and whose interior contains no orange pixel; marker pixels may be crossed.
    - Flood-fill background from the outer border without crossing blue outline pixels, then treat the remaining background components as closed interiors.
    - If a closed interior is adjacent to exactly one marker pixel, fill that background component and the marker pixel itself with the marker's color, while preserving blue outlines and orange blockers.

    Key insight:
    - The blue vertices define which boundaries can exist, but the final fill color comes from spatial containment of an interior marker rather than from the outline color.
    """
    g = np.asarray(grid, dtype=int)
    outlined, _groups, _edge_info = _draw_outline(g)
    out = outlined.copy()
    for comp in _enclosed_background_components(outlined):
        markers = _adjacent_markers(outlined, comp)
        if len(markers) != 1:
            continue
        (mr, mc), color = next(iter(markers.items()))
        for r, c in comp:
            out[r, c] = color
        out[mr, mc] = color
    return out


def _make_triangle(size: int) -> dict:
    vertices = [(0, 0), (0, size), (size, size)]
    return {"vertices": vertices, "shape": (size + 1, size + 1), "kind": "triangle"}


def _make_rectangle(height: int, width: int) -> dict:
    vertices = [(0, 0), (0, width), (height, width), (height, 0)]
    return {"vertices": vertices, "shape": (height + 1, width + 1), "kind": "rectangle"}


def _make_diamond(arm: int) -> dict:
    vertices = [(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]
    return {"vertices": vertices, "shape": (2 * arm + 1, 2 * arm + 1), "kind": "diamond"}


def _transform_point(
    pt: tuple[int, int], shape: tuple[int, int], rot: int, mirror: bool
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = pt
    h, w = shape
    if mirror:
        c = w - 1 - c
    for _ in range(rot % 4):
        r, c = c, h - 1 - r
        h, w = w, h
    return (r, c), (h, w)


def _transform_group(spec: dict) -> dict:
    rot = random.randrange(4)
    mirror = random.random() < 0.5
    new_vertices = []
    new_shape = spec["shape"]
    for pt in spec["vertices"]:
        new_pt, new_shape = _transform_point(pt, spec["shape"], rot, mirror)
        new_vertices.append(new_pt)
    result = {
        "vertices": sorted(new_vertices),
        "shape": new_shape,
        "kind": spec["kind"],
    }
    if "marker" in spec:
        marker, _ = _transform_point(spec["marker"], spec["shape"], rot, mirror)
        result["marker"] = marker
    if "oranges" in spec:
        result["oranges"] = [_transform_point(pt, spec["shape"], rot, mirror)[0] for pt in spec["oranges"]]
    return result


def _closed_template(kind: str) -> dict | None:
    if kind == "triangle":
        spec = _make_triangle(random.randint(4, 7))
    elif kind == "rectangle":
        spec = _make_rectangle(random.randint(4, 6), random.randint(4, 7))
    else:
        spec = _make_diamond(random.randint(3, 4))

    temp = np.zeros(spec["shape"], dtype=int)
    for r, c in spec["vertices"]:
        temp[r, c] = BLUE
    outlined, groups, _edge_info = _draw_outline(temp)
    comps = _enclosed_background_components(outlined)
    if len(groups) != 1 or len(comps) != 1 or len(comps[0]) < 3:
        return None
    spec["marker"] = random.choice(comps[0])
    return _transform_group(spec)


def _open_template(orange_count: int) -> dict | None:
    spec = _make_triangle(random.randint(5, 7))
    cycle = _angle_sort(spec["vertices"])
    edge_options = []
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        pts = _line_points(a, b)
        if pts is None or len(pts) <= 3:
            continue
        edge_options.append(pts[1:-1])
    if not edge_options:
        return None
    interior = max(edge_options, key=len)
    if len(interior) < orange_count:
        return None
    spec["oranges"] = random.sample(interior, orange_count)
    transformed = _transform_group(spec)

    temp = np.zeros(transformed["shape"], dtype=int)
    for r, c in transformed["vertices"]:
        temp[r, c] = BLUE
    for r, c in transformed["oranges"]:
        temp[r, c] = ORANGE
    outlined, groups, edge_info = _draw_outline(temp)
    if len(groups) != 1:
        return None
    if not any(info["status"] == "blocked" for info in edge_info):
        return None
    if _enclosed_background_components(outlined):
        return None
    return transformed


def _place_groups(groups: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    placed: list[dict] = []
    taken_boxes: list[tuple[int, int, int, int]] = []

    def boxes_far_enough(box: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = box
        for rr0, cc0, rr1, cc1 in taken_boxes:
            if not (r1 + 2 < rr0 or rr1 + 2 < r0 or c1 + 2 < cc0 or cc1 + 2 < c0):
                return False
        return True

    order = list(groups)
    random.shuffle(order)
    for group in order:
        gh, gw = group["shape"]
        for _ in range(120):
            top = random.randint(1, h - gh - 2)
            left = random.randint(1, w - gw - 2)
            box = (top, left, top + gh - 1, left + gw - 1)
            if not boxes_far_enough(box):
                continue
            shifted = {
                "vertices": [(r + top, c + left) for r, c in group["vertices"]],
                "shape": group["shape"],
                "kind": group["kind"],
                "closed": group.get("closed", False),
            }
            if "marker" in group:
                mr, mc = group["marker"]
                shifted["marker"] = (mr + top, mc + left)
            if "oranges" in group:
                shifted["oranges"] = [(r + top, c + left) for r, c in group["oranges"]]
            placed.append(shifted)
            taken_boxes.append(box)
            break
        else:
            return None
    return placed


def _build_grid(shape: tuple[int, int], groups: list[dict], marker_colors: list[int]) -> tuple[np.ndarray, dict]:
    grid = np.zeros(shape, dtype=int)
    meta_groups = []
    color_iter = iter(marker_colors)
    for group in groups:
        vertex_set = set(group["vertices"])
        if len(vertex_set) != len(group["vertices"]):
            raise ValueError("duplicate vertices in a group")
        for r, c in group["vertices"]:
            if grid[r, c] != BG:
                raise ValueError("vertex overlap")
            grid[r, c] = BLUE
        oranges = list(group.get("oranges", []))
        for r, c in oranges:
            if grid[r, c] != BG:
                raise ValueError("orange overlap")
            grid[r, c] = ORANGE
        marker_info = None
        if group.get("closed", False):
            color = next(color_iter)
            mr, mc = group["marker"]
            if grid[mr, mc] != BG:
                raise ValueError("marker overlap")
            grid[mr, mc] = color
            marker_info = {"coord": (mr, mc), "color": color}
        meta_groups.append(
            {
                "vertices": sorted(group["vertices"]),
                "closed": group.get("closed", False),
                "oranges": oranges,
                "marker": marker_info,
            }
        )
    return grid, {"groups": meta_groups}


def _match_groups(
    intended: list[dict], detected: list[list[tuple[int, int]]]
) -> dict[int, list[tuple[int, int]]] | None:
    detected_sets = [set(group) for group in detected]
    used: set[int] = set()
    mapping: dict[int, list[tuple[int, int]]] = {}
    for idx, group in enumerate(intended):
        target = set(group["vertices"])
        match = None
        for j, detected_set in enumerate(detected_sets):
            if j not in used and detected_set == target:
                match = j
                break
        if match is None:
            return None
        used.add(match)
        mapping[idx] = detected[match]
    return mapping


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    detected_groups = _vertex_groups(grid)
    if len(detected_groups) != len(meta["groups"]):
        return False
    group_map = _match_groups(meta["groups"], detected_groups)
    if group_map is None:
        return False

    outlined, _groups, edge_info = _draw_outline(grid)
    blocked_infos = [info for info in edge_info if info["status"] == "blocked"]
    if not blocked_infos:
        return False

    enclosed = _enclosed_background_components(outlined)
    closed_groups = [group for group in meta["groups"] if group["closed"]]
    if len(enclosed) != len(closed_groups):
        return False
    if not enclosed or min(len(comp) for comp in enclosed) < 3:
        return False

    comp_of_marker: dict[tuple[int, int], int] = {}
    for comp_idx, comp in enumerate(enclosed):
        markers = _adjacent_markers(grid, comp)
        if len(markers) != 1:
            return False
        marker = next(iter(markers))
        if marker in comp_of_marker:
            return False
        comp_of_marker[marker] = comp_idx

    for group in closed_groups:
        marker = group["marker"]
        assert marker is not None
        if marker["coord"] not in comp_of_marker:
            return False

    if not any(group["oranges"] for group in meta["groups"] if not group["closed"]):
        return False

    out = main(grid)
    for group in closed_groups:
        marker = group["marker"]
        assert marker is not None
        mr, mc = marker["coord"]
        color = marker["color"]
        if out[mr, mc] != color:
            return False
        comp_idx = comp_of_marker[(mr, mc)]
        for r, c in enclosed[comp_idx]:
            if out[r, c] != color:
                return False

    if not any(int(v) in MARKER_COLORS for v in np.unique(out)):
        return False

    orange_targets = [coord for group in meta["groups"] for coord in group["oranges"]]
    probe = orange_targets[0]
    no_block = grid.copy()
    no_block[probe] = BG
    if np.array_equal(main(no_block), out):
        return False

    first_marker = closed_groups[0]["marker"]
    assert first_marker is not None
    changed_marker = grid.copy()
    mr, mc = first_marker["coord"]
    alt_color = next(color for color in MARKER_COLORS if color != first_marker["color"])
    changed_marker[mr, mc] = alt_color
    if np.array_equal(main(changed_marker), out):
        return False
    return True


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    closed_count = 1 if random.random() < 0.58 else 2
    orange_count = random.randint(1, 3)

    if closed_count == 1:
        closed_kinds = [random.choice(["triangle", "rectangle", "diamond"])]
    else:
        closed_kinds = [random.choice(["triangle", "rectangle", "diamond"]), "triangle"]
        random.shuffle(closed_kinds)

    groups: list[dict] = []
    for kind in closed_kinds:
        group = _closed_template(kind)
        if group is None:
            return None
        group["closed"] = True
        groups.append(group)

    open_group = _open_template(orange_count)
    if open_group is None:
        return None
    open_group["closed"] = False
    groups.append(open_group)

    total_vertices = sum(len(group["vertices"]) for group in groups)
    if not (6 <= total_vertices <= 10):
        return None

    placed = _place_groups(groups, (h, w))
    if placed is None:
        return None

    marker_colors = random.sample(list(MARKER_COLORS), k=closed_count)
    try:
        grid, meta = _build_grid((h, w), placed, marker_colors)
    except ValueError:
        return None

    if not _validate_case(grid, meta):
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
