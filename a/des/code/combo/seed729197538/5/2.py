import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]
    raise ValueError("line_points expects an orthogonal segment")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _fill_closed_mask(boundary_mask: np.ndarray) -> np.ndarray:
    padded = np.pad(boundary_mask.astype(bool), 1, constant_values=False)
    outside = np.zeros_like(padded, dtype=bool)
    h, w = padded.shape
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not padded[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not padded[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not padded[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    filled = padded | (~padded & ~outside)
    return filled[1:-1, 1:-1]


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    all_colors = {int(v) for v in np.unique(grid) if int(v) != bg}
    candidates = []
    for c in (0, w - 1):
        rows = [r for r in range(h) if int(grid[r, c]) != bg]
        if not rows:
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        if set(colors) != all_colors:
            continue
        ok = True
        for r in rows:
            color = int(grid[r, c])
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((len(rows), c, rows, colors))
    if not candidates:
        for c in (0, w - 1):
            rows = [r for r in range(h) if int(grid[r, c]) != bg]
            if not rows:
                continue
            colors = [int(grid[r, c]) for r in rows]
            if len(set(colors)) == len(colors):
                candidates.append((len(rows), c, rows, colors))
    if not candidates:
        return [], set()
    _, col, rows, colors = max(candidates, key=lambda x: x[0])
    return colors, {(r, col) for r in rows}


def _extract_segments(
    grid: np.ndarray, color: int, legend_cells: set[tuple[int, int]]
) -> tuple[list[dict], np.ndarray]:
    mask = grid == color
    for r, c in legend_cells:
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            mask[r, c] = False
    segments = []
    for comp in _components(mask):
        if len(comp) < 2:
            continue
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        if len(rows) == 1:
            r = rows[0]
            cs = sorted(c for _, c in comp)
            if len(cs) != cs[-1] - cs[0] + 1:
                continue
            endpoints = ((r, cs[0]), (r, cs[-1]))
        elif len(cols) == 1:
            c = cols[0]
            rs = sorted(r for r, _ in comp)
            if len(rs) != rs[-1] - rs[0] + 1:
                continue
            endpoints = ((rs[0], c), (rs[-1], c))
        else:
            continue
        segments.append({"cells": comp, "endpoints": endpoints})
    segments.sort(key=lambda s: min(s["cells"]))
    return segments, mask


def _clear_between(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    r1, c1 = a
    r2, c2 = b
    if a == b:
        return False
    if r1 == r2:
        step = 1 if c2 > c1 else -1
        for c in range(c1 + step, c2, step):
            if mask[r1, c]:
                return False
        return True
    if c1 == c2:
        step = 1 if r2 > r1 else -1
        for r in range(r1 + step, r2, step):
            if mask[r, c1]:
                return False
        return True
    return False


def _endpoint_pairs(segments: list[dict], color_mask: np.ndarray):
    endpoints = []
    for seg_idx, seg in enumerate(segments):
        for point in seg["endpoints"]:
            endpoints.append({"seg": seg_idx, "point": point})

    choices: dict[int, int] = {}
    for i, src in enumerate(endpoints):
        best_dist = None
        best: list[int] = []
        for j, dst in enumerate(endpoints):
            if i == j or src["seg"] == dst["seg"]:
                continue
            if not _clear_between(color_mask, src["point"], dst["point"]):
                continue
            dist = abs(src["point"][0] - dst["point"][0]) + abs(src["point"][1] - dst["point"][1])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = [j]
            elif dist == best_dist:
                best.append(j)
        if len(best) == 1:
            choices[i] = best[0]

    pairs = []
    for i, j in choices.items():
        if i < j and choices.get(j) == i:
            pairs.append((i, j))
    return endpoints, pairs


def _draw_line(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    for r, c in _line_points(a, b):
        mask[r, c] = True


def _render_color_layer(shape: tuple[int, int], segments: list[dict], color_mask: np.ndarray) -> np.ndarray:
    h, w = shape
    layer = np.zeros((h, w), dtype=bool)
    for seg in segments:
        for r, c in seg["cells"]:
            layer[r, c] = True

    endpoints, pairs = _endpoint_pairs(segments, color_mask)
    deg = [0] * len(segments)
    adj = [set() for _ in segments]
    for i, j in pairs:
        a = endpoints[i]["point"]
        b = endpoints[j]["point"]
        _draw_line(layer, a, b)
        sa = endpoints[i]["seg"]
        sb = endpoints[j]["seg"]
        if sa != sb:
            deg[sa] += 1
            deg[sb] += 1
            adj[sa].add(sb)
            adj[sb].add(sa)

    closed = False
    if segments:
        seen = set()
        stack = [0]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(n for n in adj[cur] if n not in seen)
        closed = len(seen) == len(segments) and len(pairs) == len(segments) and all(d == 2 for d in deg)

    if closed:
        layer = _fill_closed_mask(layer)
    return layer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color short segments into ordered rectilinear paths.
    Closed paths are filled; open paths remain as polylines.
    Colors are rendered in the edge-legend order, then the legend is removed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    order, legend_cells = _find_legend(g, bg)

    if not order:
        return g.copy()

    out = np.full((h, w), bg, dtype=int)
    for color in order:
        segments, color_mask = _extract_segments(g, color, legend_cells)
        if not segments:
            continue
        layer = _render_color_layer((h, w), segments, color_mask)
        out[layer] = int(color)
    return out


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_TRIES = 120
OVERLAP_PROB = 0.55


def _transform_vertices(vertices: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    rot = rng.randrange(4)
    flip_h = rng.random() < 0.5
    pts = []
    for r, c in vertices:
        x, y = r, c
        for _ in range(rot):
            x, y = y, -x
        if flip_h:
            y = -y
        pts.append((x, y))
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _sample_closed_vertices(rng: random.Random) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        height = rng.randint(5, 10)
        width = rng.randint(5, 10)
        vertices = [(0, 0), (0, width), (height, width), (height, 0)]
    else:
        width = rng.randint(8, 12)
        height = rng.randint(8, 12)
        inner_r = rng.randint(3, height - 3)
        inner_c = rng.randint(3, width - 3)
        vertices = [(0, 0), (0, width), (inner_r, width), (inner_r, inner_c), (height, inner_c), (height, 0)]
    return _transform_vertices(vertices, rng)


def _sample_open_vertices(rng: random.Random) -> list[tuple[int, int]]:
    kind = rng.randrange(3)
    if kind == 0:
        width = rng.randint(5, 10)
        height = rng.randint(5, 10)
        vertices = [(0, 0), (0, width), (height, width)]
    elif kind == 1:
        c1 = rng.randint(5, 10)
        r1 = rng.randint(4, 7)
        delta = rng.randint(4, 7)
        if rng.random() < 0.5 and c1 - delta >= 0:
            c2 = c1 - delta
        else:
            c2 = c1 + delta
        r2 = r1 + rng.randint(4, 7)
        vertices = [(0, 0), (0, c1), (r1, c1), (r1, c2), (r2, c2)]
    else:
        c1 = rng.randint(5, 9)
        r1 = rng.randint(4, 6)
        delta1 = rng.randint(4, 6)
        if rng.random() < 0.5 and c1 - delta1 >= 0:
            c2 = c1 - delta1
        else:
            c2 = c1 + delta1
        r2 = r1 + rng.randint(4, 6)
        delta2 = rng.randint(4, 6)
        if c2 <= c1 and c2 + delta2 >= 0:
            c3 = c2 + delta2
        elif c2 - delta2 >= 0:
            c3 = c2 - delta2
        else:
            c3 = c2 + delta2
        r3 = r2 + rng.randint(4, 6)
        vertices = [(0, 0), (0, c1), (r1, c1), (r1, c2), (r2, c2), (r2, c3), (r3, c3)]
    return _transform_vertices(vertices, rng)


def _mask_from_points(points: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in points:
        mask[r, c] = True
    return mask


def _build_shape(rng: random.Random, closed: bool):
    vertices = _sample_closed_vertices(rng) if closed else _sample_open_vertices(rng)
    edge_count = len(vertices) if closed else len(vertices) - 1
    attach_to_start = rng.random() < 0.5

    input_points: set[tuple[int, int]] = set()
    boundary_points: set[tuple[int, int]] = set()
    for i in range(edge_count):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)] if closed else vertices[i + 1]
        full = _line_points(a, b)
        steps = len(full) - 1
        max_visible = min(4, steps - 1)
        if max_visible < 2:
            return None
        visible_len = rng.randint(2, max_visible)
        segment = full[:visible_len] if attach_to_start else full[-visible_len:]
        if input_points.intersection(segment):
            return None
        input_points.update(segment)
        boundary_points.update(full)

    height = max(r for r, _ in boundary_points) + 1
    width = max(c for _, c in boundary_points) + 1
    input_mask = _mask_from_points(input_points, (height, width))
    if len(_components(input_mask)) != edge_count:
        return None

    boundary_mask = _mask_from_points(boundary_points, (height, width))
    output_mask = _fill_closed_mask(boundary_mask) if closed else boundary_mask
    if closed and int(output_mask.sum()) <= int(boundary_mask.sum()):
        return None

    output_points = {tuple(p) for p in np.argwhere(output_mask)}
    return {
        "input_points": input_points,
        "output_points": output_points,
        "height": height,
        "width": width,
        "closed": closed,
    }


def _translate(points: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in points}


def _place_shape(
    rng: random.Random,
    grid_shape: tuple[int, int],
    legend_side: str,
    shape: dict,
    blocked: np.ndarray,
    occupied_output: np.ndarray,
    require_overlap: bool,
):
    h, w = grid_shape
    max_r = h - shape["height"] - 1
    min_r = 1
    if legend_side == "left":
        min_c = 2
        max_c = w - shape["width"] - 1
    else:
        min_c = 1
        max_c = w - shape["width"] - 2
    if max_r < min_r or max_c < min_c:
        return None

    for _ in range(PLACEMENT_TRIES):
        dr = rng.randint(min_r, max_r)
        dc = rng.randint(min_c, max_c)
        input_points = _translate(shape["input_points"], dr, dc)
        if any(blocked[r, c] for r, c in input_points):
            continue
        output_points = _translate(shape["output_points"], dr, dc)
        overlaps = any(occupied_output[r, c] for r, c in output_points)
        if require_overlap and not overlaps:
            continue
        return input_points, output_points
    return None


def _mark_blocked(blocked: np.ndarray, points: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in points:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    color_count = rng.randint(1, 3)
    colors = rng.sample(COLOR_POOL, color_count)
    legend_side = rng.choice(["left", "right"])
    legend_col = 0 if legend_side == "left" else w - 1
    legend_top = rng.randint(1, h - color_count - 1)

    need_overlap = color_count >= 2 and rng.random() < OVERLAP_PROB
    closed_flags = [rng.random() < 0.5 for _ in range(color_count)]
    if need_overlap and not any(closed_flags):
        closed_flags[rng.randrange(color_count)] = True

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    occupied_output = np.zeros((h, w), dtype=bool)
    made_overlap = False

    for idx, color in enumerate(colors):
        shape = _build_shape(rng, closed_flags[idx])
        if shape is None:
            return None
        require_overlap = need_overlap and idx > 0 and not made_overlap
        placed = _place_shape(rng, (h, w), legend_side, shape, blocked, occupied_output, require_overlap)
        if placed is None and require_overlap:
            placed = _place_shape(rng, (h, w), legend_side, shape, blocked, occupied_output, False)
        if placed is None:
            return None

        input_points, output_points = placed
        if require_overlap and any(occupied_output[r, c] for r, c in output_points):
            made_overlap = True

        for r, c in input_points:
            inp[r, c] = color
        for r, c in output_points:
            expected[r, c] = color

        _mark_blocked(blocked, input_points)
        for r, c in output_points:
            occupied_output[r, c] = True

    if need_overlap and not made_overlap:
        return None

    for i, color in enumerate(colors):
        inp[legend_top + i, legend_col] = color

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
