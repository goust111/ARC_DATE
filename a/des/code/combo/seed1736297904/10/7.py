from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _point_on_segment(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    px, py = p
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-9:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -1e-9:
        return False
    sq_len = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= sq_len + 1e-9


def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if _point_on_segment(point, (x1, y1), (x2, y2)):
            return False
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_cross > x:
                inside = not inside
    return inside


def _supercover_line(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    x0, y0 = a[1], a[0]
    x1, y1 = b[1], b[0]
    dx = x1 - x0
    dy = y1 - y0
    nx = abs(dx)
    ny = abs(dy)
    sx = 0 if dx == 0 else (1 if dx > 0 else -1)
    sy = 0 if dy == 0 else (1 if dy > 0 else -1)

    x, y = x0, y0
    cells = {(y, x)}
    ix = 0
    iy = 0
    while ix < nx or iy < ny:
        lhs = (1 + 2 * ix) * ny
        rhs = (1 + 2 * iy) * nx
        if lhs == rhs:
            x += sx
            y += sy
            ix += 1
            iy += 1
        elif lhs < rhs:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        cells.add((y, x))
    return cells


def _ordered_path(
    comp_set: set[tuple[int, int]],
    start: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    if len(comp_set) == 1:
        return [start]

    h, w = shape
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [p for p in _neighbors4(cur[0], cur[1], h, w) if p in comp_set and p != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "grey path must be a simple non-branching polyline"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(comp_set), "failed to traverse the full grey path"
    return ordered


def _pair_interior(
    shape: tuple[int, int],
    marker_a: tuple[int, int],
    marker_b: tuple[int, int],
    ordered_path: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    polygon = [(float(marker_a[1]), float(marker_a[0]))]
    polygon.extend((float(c), float(r)) for r, c in ordered_path)
    polygon.append((float(marker_b[1]), float(marker_b[0])))

    boundary = set(ordered_path)
    boundary.add(marker_a)
    boundary.add(marker_b)
    boundary |= _supercover_line(marker_a, marker_b)

    h, w = shape
    r0, c0, r1, c1 = _bbox(boundary)
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h - 1, r1)
    c1 = min(w - 1, c1)

    inside: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in boundary:
                continue
            if _point_in_polygon((float(c), float(r)), polygon):
                inside.add((r, c))
    return inside


def _detect_palette(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    candidates: list[tuple[int, int, dict[int, int], np.ndarray]] = []
    for comp in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if ww != 5 or hh < 5 or hh % 2 == 0:
            continue

        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r == r0 or r == r1 or c == c0 or c == c1
                value = int(grid[r, c])
                if on_border:
                    if value != MAROON:
                        ok = False
                        break
                elif value == MAROON:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        mapping: dict[int, int] = {}
        for rel_r in range(1, hh - 1):
            r = r0 + rel_r
            row = grid[r, c0 + 1 : c1]
            if rel_r % 2 == 1:
                source = int(row[0])
                shade = int(row[2])
                if int(row[1]) != BLACK or source in (BLACK, MAROON) or shade in (BLACK, MAROON):
                    ok = False
                    break
                if source in mapping:
                    ok = False
                    break
                mapping[source] = shade
            else:
                if not np.all(row == BLACK):
                    ok = False
                    break
        if ok and mapping:
            mask = np.zeros_like(grid, dtype=bool)
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
            candidates.append((r1, c1, mapping, mask))

    assert candidates, "failed to locate the maroon palette"
    _, _, mapping, mask = max(candidates, key=lambda item: (item[0], item[1]))
    return mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, palette_mask = _detect_palette(g)
    active_mask = ~palette_mask

    marker_positions: dict[int, list[tuple[int, int]]] = {}
    for source in mapping:
        coords = [tuple(map(int, rc)) for rc in np.argwhere((g == source) & active_mask)]
        if not coords:
            continue
        assert len(coords) == 2, "each used source color must have exactly two markers"
        marker_positions[int(source)] = coords
    assert marker_positions, "no marker pairs found outside the palette"

    out = g.copy()
    grey_components = _components((g == GREY) & active_mask)
    paths_by_color: dict[int, list[tuple[int, int]]] = {}

    for comp in grey_components:
        comp_set = set(comp)
        adjacent_markers: set[tuple[int, int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                color = int(g[nr, nc])
                if color in marker_positions and (nr, nc) in marker_positions[color]:
                    adjacent_markers.add((nr, nc, color))

        assert len(adjacent_markers) == 2, "each grey path must touch exactly one marker pair"
        colors = {color for _, _, color in adjacent_markers}
        assert len(colors) == 1, "both ends of a grey path must match the same marker color"
        color = next(iter(colors))
        assert color not in paths_by_color, "each marker color must have exactly one grey path"

        if len(comp_set) == 1:
            ordered = comp[:]
        else:
            endpoints = [
                cell
                for cell in comp
                if sum(nei in comp_set for nei in _neighbors4(cell[0], cell[1], h, w)) == 1
            ]
            assert len(endpoints) == 2, "grey path must have exactly two endpoints"

            marker_a, marker_b = marker_positions[color]
            if _adjacent(endpoints[0], marker_a):
                start = endpoints[0]
            elif _adjacent(endpoints[1], marker_a):
                start = endpoints[1]
            elif _adjacent(endpoints[0], marker_b):
                marker_a, marker_b = marker_b, marker_a
                start = endpoints[0]
            else:
                marker_a, marker_b = marker_b, marker_a
                start = endpoints[1]

            ordered = _ordered_path(comp_set, start, g.shape)
            assert _adjacent(ordered[0], marker_a) and _adjacent(ordered[-1], marker_b)
            marker_positions[color] = [marker_a, marker_b]

        paths_by_color[color] = ordered

    assert set(paths_by_color) == set(marker_positions), "every marker pair must have one matching grey path"

    for color, ordered in paths_by_color.items():
        marker_a, marker_b = marker_positions[color]
        inside = _pair_interior(g.shape, marker_a, marker_b, ordered)
        shade = mapping[color]
        for r, c in inside:
            if not bool(palette_mask[r, c]) and int(out[r, c]) == BLACK:
                out[r, c] = shade

    return out


def _trace_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [vertices[0]]
    cur = vertices[0]
    for target in vertices[1:]:
        dr = target[0] - cur[0]
        dc = target[1] - cur[1]
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
        assert step_r == 0 or step_c == 0
        while cur != target:
            cur = (cur[0] + step_r, cur[1] + step_c)
            path.append(cur)
    return path


def _build_template(rng: np.random.Generator) -> dict[str, object]:
    for _ in range(200):
        span = int(rng.integers(6, 12))
        max_segments = min(3, span - 2)
        segments = int(rng.integers(1, max_segments + 1))
        max_depth = int(rng.integers(3, 6))

        if segments == 1:
            break_cols: list[int] = []
        else:
            choices = np.arange(1, span, dtype=int)
            break_cols = sorted(int(x) for x in rng.choice(choices, size=segments - 1, replace=False))

        depths: list[int] = []
        prev_depth = -1
        for _seg in range(segments):
            candidates = [d for d in range(2, max_depth + 1) if d != prev_depth]
            depth = int(rng.choice(candidates))
            depths.append(depth)
            prev_depth = depth

        vertices = [(1, 0), (depths[0], 0)]
        for idx, depth in enumerate(depths):
            end_col = span if idx == segments - 1 else break_cols[idx]
            vertices.append((depth, end_col))
            if idx < segments - 1:
                vertices.append((depths[idx + 1], end_col))
        vertices.append((1, span))

        path = _trace_vertices(vertices)
        if len(path) != len(set(path)):
            continue

        markers = [(0, 0), (0, span)]
        if rng.random() < 0.5:
            max_r = max(r for r, _ in path)
            path = [(max_r - r, c) for r, c in path]
            markers = [(max_r - r, c) for r, c in markers]

        if rng.random() < 0.5:
            path = [(c, r) for r, c in path]
            markers = [(c, r) for r, c in markers]

        all_cells = path + markers
        height = max(r for r, _ in all_cells) + 1
        width = max(c for _, c in all_cells) + 1
        inside = _pair_interior((height, width), markers[0], markers[1], path)
        if len(inside) < 3:
            continue
        if inside & set(path):
            continue

        return {
            "shape": (height, width),
            "markers": tuple(markers),
            "path": list(path),
        }

    raise RuntimeError("failed to build a valid object template")


def _stamp_palette(
    grid: np.ndarray,
    mapping_rows: list[tuple[int, int]],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    palette_h = 2 * len(mapping_rows) + 1
    palette_w = 5
    r0 = h - palette_h
    c0 = w - palette_w
    r1 = h - 1
    c1 = w - 1

    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON

    for idx, (source, shade) in enumerate(mapping_rows):
        rr = r0 + 1 + 2 * idx
        grid[rr, c0 + 1] = source
        grid[rr, c0 + 3] = shade

    return grid, (r0, c0, r1, c1)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 5))
    active_keys = [int(x) for x in rng.choice(np.array(USABLE_COLORS), size=pair_count, replace=False)]
    shade_pool = [color for color in USABLE_COLORS if color not in active_keys]
    shades = [int(rng.choice(shade_pool)) for _ in range(pair_count)]
    mapping_rows = list(zip(active_keys, shades))
    rng.shuffle(mapping_rows)

    height = int(rng.integers(18, 27))
    width = int(rng.integers(20, 31))
    palette_h = 2 * pair_count + 1
    if height <= palette_h + 4 or width <= 12:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid, palette_bbox = _stamp_palette(grid, mapping_rows)
    pr0, pc0, pr1, pc1 = palette_bbox

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[max(0, pr0 - 1) : min(height, pr1 + 2), max(0, pc0 - 1) : min(width, pc1 + 2)] = True

    templates = [_build_template(rng) for _ in range(pair_count)]
    order = sorted(range(pair_count), key=lambda idx: np.prod(templates[idx]["shape"]), reverse=True)
    placements: list[tuple[int, int] | None] = [None] * pair_count

    for idx in order:
        obj_h, obj_w = templates[idx]["shape"]
        if obj_h >= height - 2 or obj_w >= width - 2:
            return None

        placed = False
        for _ in range(250):
            top = int(rng.integers(1, height - obj_h))
            left = int(rng.integers(1, width - obj_w))
            rr0 = max(0, top - 1)
            cc0 = max(0, left - 1)
            rr1 = min(height, top + obj_h + 1)
            cc1 = min(width, left + obj_w + 1)
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None

    for (source, _shade), template, placement in zip(mapping_rows, templates, placements):
        assert placement is not None
        top, left = placement
        marker_a, marker_b = template["markers"]
        for r, c in template["path"]:
            assert grid[top + r, left + c] == BLACK
            grid[top + r, left + c] = GREY
        for r, c in (marker_a, marker_b):
            assert grid[top + r, left + c] == BLACK
            grid[top + r, left + c] = source

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
