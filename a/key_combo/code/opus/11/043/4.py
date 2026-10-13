import itertools
import math
import random
from collections import deque

import numpy as np


BG = 0
CORRIDOR = 7
WALL = 9
PLACEHOLDER = 5
PARTICLE_COLORS = (1, 2, 3, 4, 6, 8)
MAX_ATTEMPTS = 800
EPS = 1e-9


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_point(point, box):
    r, c = point
    r0, c0, r1, c1 = box
    rr = 0.0 if r1 == r0 else (float(r) - float(r0)) / float(r1 - r0)
    cc = 0.0 if c1 == c0 else (float(c) - float(c0)) / float(c1 - c0)
    return rr, cc


def _centroid(coords):
    arr = np.asarray(coords, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _components_of_color(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(cells))
    return comps


def _assignment_details(src_points, dst_points):
    n = len(src_points)
    best_perm = None
    best_cost = None
    second_cost = None
    second_perm = None
    for perm in itertools.permutations(range(n)):
        cost = 0.0
        for i, j in enumerate(perm):
            sr, sc = src_points[i]
            dr, dc = dst_points[j]
            cost += math.hypot(sr - dr, sc - dc)
        if best_cost is None or cost < best_cost - EPS:
            second_cost = best_cost
            second_perm = best_perm
            best_cost = cost
            best_perm = perm
        elif abs(cost - best_cost) <= EPS:
            second_cost = cost
            second_perm = perm
        elif second_cost is None or cost < second_cost - EPS:
            second_cost = cost
            second_perm = perm
    unique = second_cost is None or second_cost > best_cost + 1e-4
    gap = None if second_cost is None else second_cost - best_cost
    return {
        "perm": best_perm,
        "best_cost": best_cost,
        "second_cost": second_cost,
        "second_perm": second_perm,
        "unique": unique,
        "gap": gap,
    }


def _axis_sort_perm(src_points, dst_points, axis):
    other = 1 - axis
    src_order = sorted(range(len(src_points)), key=lambda i: (round(src_points[i][axis], 6), round(src_points[i][other], 6), i))
    dst_order = sorted(range(len(dst_points)), key=lambda i: (round(dst_points[i][axis], 6), round(dst_points[i][other], 6), i))
    perm = [None] * len(src_points)
    for si, dj in zip(src_order, dst_order):
        perm[si] = dj
    return tuple(perm)


def _analyze(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    corridor_cells = [(int(r), int(c)) for r, c in np.argwhere(g == CORRIDOR)]
    assert corridor_cells, "expected orange corridor cells"
    corridor_box = _bbox(corridor_cells)

    rows = sorted({r for r, _ in corridor_cells} | {int(r) for r, _ in np.argwhere(g == WALL)})
    particles = []
    for r in rows:
        orange_cols = [int(c) for c in np.where(g[r] == CORRIDOR)[0]]
        if not orange_cols:
            continue
        wall_cols = [int(c) for c in np.where(g[r] == WALL)[0]]
        assert len(wall_cols) == 1, "each corridor row must end in exactly one wall"
        wall_col = wall_cols[0]
        assert orange_cols == list(range(min(orange_cols), max(orange_cols) + 1)), "corridor must be contiguous"
        assert max(orange_cols) == wall_col - 1, "corridor must stop immediately before wall"
        start_col = min(orange_cols) - 1
        assert 0 <= start_col < w, "missing particle before corridor"
        color = int(g[r, start_col])
        assert color not in (BG, CORRIDOR, WALL, PLACEHOLDER), "invalid particle color"
        particles.append(
            {
                "row": int(r),
                "start": (int(r), int(start_col)),
                "end": (int(r), int(wall_col - 1)),
                "color": color,
            }
        )
    assert 3 <= len(particles) <= 4, "expected 3-4 particle rows"

    end_cols = [p["end"][1] for p in particles]
    assert len(set(end_cols)) == len(end_cols), "particle end columns must differ"

    placeholders = _components_of_color(g, PLACEHOLDER)
    assert len(placeholders) == len(particles), "placeholder count must match particle count"
    for comp in placeholders:
        assert 3 <= len(comp) <= 5, "placeholder size must be 3-5"

    placeholder_box = _bbox([cell for comp in placeholders for cell in comp])
    end_points = [_normalize_point(p["end"], corridor_box) for p in particles]
    start_points = [_normalize_point(p["start"], corridor_box) for p in particles]
    placeholder_points = [_normalize_point(_centroid(comp), placeholder_box) for comp in placeholders]
    matching = _assignment_details(end_points, placeholder_points)
    assert matching["unique"], "particle-stop to placeholder matching must be unique"
    return {
        "particles": particles,
        "placeholders": placeholders,
        "corridor_box": corridor_box,
        "placeholder_box": placeholder_box,
        "end_points": end_points,
        "start_points": start_points,
        "placeholder_points": placeholder_points,
        "matching": matching,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each lower placeholder block by matching it to a particle's collision-stop endpoint.

    Input structure:
    - The background is black, with an upper particle zone and a lower placeholder zone.
    - In the upper zone, 3-4 colored single-pixel particles each sit immediately left of a horizontal orange corridor whose right end is blocked by one maroon wall pixel.
    - In the lower zone, the same number of grey 4-connected placeholder blocks appear as separate 3-5 cell irregular components.

    Rule:
    - Move every particle right until it would enter the maroon wall, so its endpoint is the cell just before the wall.
    - Compute the normalized coordinates of all particle endpoints inside the bounding box of all orange corridor cells.
    - Compute the normalized coordinates of all placeholder centroids inside the bounding box of all placeholder cells.
    - Find the unique one-to-one assignment with minimum total Euclidean distance between endpoint coordinates and centroid coordinates.
    - Recolor each entire placeholder block to the original color of its matched particle, then clear the whole upper zone to black.

    Key insight:
    - The particles' final stopped layout, not their start layout, is the only reliable code that links the top transport scene to the lower grey blocks.
    """
    info = _analyze(grid)
    out = np.zeros_like(np.asarray(grid, dtype=int))
    perm = info["matching"]["perm"]
    for particle_index, placeholder_index in enumerate(perm):
        color = int(info["particles"][particle_index]["color"])
        for r, c in info["placeholders"][placeholder_index]:
            out[r, c] = color
    return out


def _normalize_shape(coords):
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _rotate(coords):
    return [(-c, r) for r, c in coords]


def _reflect(coords):
    return [(r, -c) for r, c in coords]


def _build_shape_variants():
    bases = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    ]
    seen = set()
    shapes = []
    for base in bases:
        forms = [base]
        forms.append(_reflect(base))
        for form in forms:
            cur = list(form)
            for _ in range(4):
                key = _normalize_shape(cur)
                if key not in seen:
                    seen.add(key)
                    shapes.append(key)
                cur = _rotate(cur)
    return shapes


SHAPES = _build_shape_variants()


def _shift_shape(shape, anchor_r, anchor_c):
    return [(anchor_r + r, anchor_c + c) for r, c in shape]


def _shape_centroid(shape):
    return _centroid(shape)


def _cells_in_bounds(cells, h, w, min_row):
    return all(min_row <= r < h and 0 <= c < w for r, c in cells)


def _buffer_cells(cells):
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _paint_top(grid, particles):
    for p in particles:
        r = p["row"]
        grid[r, p["start_col"]] = p["color"]
        grid[r, p["start_col"] + 1 : p["wall_col"]] = CORRIDOR
        grid[r, p["wall_col"]] = WALL


def _particle_end_points_from_spec(particles):
    corridor_cells = []
    for p in particles:
        for c in range(p["start_col"] + 1, p["wall_col"]):
            corridor_cells.append((p["row"], c))
    corridor_box = _bbox(corridor_cells)
    return [_normalize_point((p["row"], p["wall_col"] - 1), corridor_box) for p in particles]


def _build_particle_rows(h, w, rng, n):
    top_limit = rng.randint(5, 7)
    if top_limit + 7 >= h:
        return None
    rows = sorted(rng.sample(range(1, top_limit + 1), n))
    if rows[-1] - rows[0] < 2:
        return None
    start_cols = [rng.randint(1, 3) for _ in range(n)]
    min_endpoint = max(start_cols) + 3
    endpoint_candidates = list(range(min_endpoint, w - 2))
    if len(endpoint_candidates) < n:
        return None
    end_cols = rng.sample(endpoint_candidates, n)
    if max(end_cols) - min(end_cols) < 4:
        return None
    rng.shuffle(end_cols)
    colors = rng.sample(list(PARTICLE_COLORS), n)
    particles = []
    for row, start_col, end_col, color in zip(rows, start_cols, end_cols, colors):
        if end_col - start_col < 3:
            return None
        particles.append(
            {
                "row": row,
                "start_col": start_col,
                "wall_col": end_col + 1,
                "color": color,
            }
        )
    return particles


def _rank_offsets(values, scale):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    base = {
        3: (-0.8, 0.8, 0.0),
        4: (-1.0, 0.7, -0.3, 0.9),
    }[len(values)]
    offsets = [0.0] * len(values)
    for pos, idx in enumerate(order):
        offsets[idx] = base[pos] * scale
    return offsets


def _desired_centers(end_points, h, w, min_row, rng):
    row_span_hi = min(6.0, h - min_row - 1.8)
    col_span_hi = min(8.5, w - 3.0)
    if row_span_hi < 2.0 or col_span_hi < 4.5:
        return None
    row_span = rng.uniform(2.0, row_span_hi)
    col_span = rng.uniform(4.5, col_span_hi)
    base_r_max = h - 2.2 - row_span
    base_c_max = w - 2.2 - col_span
    if base_r_max < min_row + 0.4 or base_c_max < 0.8:
        return None
    base_r = rng.uniform(min_row + 0.4, base_r_max)
    base_c = rng.uniform(0.8, base_c_max)
    row_cross = _rank_offsets([cc for _, cc in end_points], rng.uniform(0.45, 0.7))
    col_cross = _rank_offsets([rr for rr, _ in end_points], rng.uniform(0.55, 0.85))
    centers = []
    for i, (rr, cc) in enumerate(end_points):
        target_r = base_r + rr * row_span + row_cross[i] + rng.uniform(-0.12, 0.12)
        target_c = base_c + cc * col_span + col_cross[i] + rng.uniform(-0.18, 0.18)
        centers.append((target_r, target_c))
    return centers


def _place_placeholders(grid, end_points, min_row, rng):
    h, w = grid.shape
    for _ in range(150):
        centers = _desired_centers(end_points, h, w, min_row, rng)
        if centers is None:
            continue
        blocked = set()
        comps = []
        order = list(range(len(end_points)))
        rng.shuffle(order)
        success = True
        for idx in order:
            shape = rng.choice(SHAPES)
            centroid_r, centroid_c = _shape_centroid(shape)
            anchor_r = int(round(centers[idx][0] - centroid_r))
            anchor_c = int(round(centers[idx][1] - centroid_c))
            cells = _shift_shape(shape, anchor_r, anchor_c)
            if not _cells_in_bounds(cells, h, w, min_row):
                success = False
                break
            if any(cell in blocked for cell in cells):
                success = False
                break
            comps.append(cells)
            blocked.update(_buffer_cells(cells))
        if not success:
            continue
        if len(comps) != len(end_points):
            continue
        for cells in comps:
            for r, c in cells:
                grid[r, c] = PLACEHOLDER
        return comps
    return None


def _validate_non_degenerate(inp):
    info = _analyze(inp)
    perm = info["matching"]["perm"]
    row_perm = _axis_sort_perm(info["end_points"], info["placeholder_points"], axis=0)
    col_perm = _axis_sort_perm(info["end_points"], info["placeholder_points"], axis=1)
    start_matching = _assignment_details(info["start_points"], info["placeholder_points"])
    if row_perm == perm and col_perm == perm:
        return False
    if start_matching["perm"] == perm:
        return False
    if not start_matching["unique"]:
        return False
    if info["matching"]["gap"] is None or info["matching"]["gap"] < 0.05:
        return False
    out = main(inp)
    assert out.shape == inp.shape
    assert np.all(out[np.asarray(inp) == CORRIDOR] == BG)
    assert np.all(out[np.asarray(inp) == WALL] == BG)
    assert np.all(out[: min(r for r, _ in np.argwhere(inp == PLACEHOLDER))] == BG)
    colors = sorted(int(v) for v in np.unique(out) if int(v) != BG)
    if len(colors) != len(info["particles"]):
        return False
    return True


def _gen_attempt(rng):
    h = rng.randint(15, 19)
    w = rng.randint(15, 19)
    n = rng.choice((3, 4))
    grid = np.zeros((h, w), dtype=int)

    particles = _build_particle_rows(h, w, rng, n)
    if particles is None:
        return None
    _paint_top(grid, particles)

    end_points = _particle_end_points_from_spec(particles)
    bottom_min_row = max(p["row"] for p in particles) + rng.randint(3, 4)
    if bottom_min_row > h - 4:
        return None
    placed = _place_placeholders(grid, end_points, bottom_min_row, rng)
    if placed is None:
        return None
    if not _validate_non_degenerate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid particle-endpoint matching sample")
