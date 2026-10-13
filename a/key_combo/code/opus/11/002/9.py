import itertools
import math
import random

import numpy as np


BLACK = 0
YELLOW = 4
SPRITE_COLORS = (1, 2, 3, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
TEMPLATES = ("diamond", "hhex", "vhex")
SCALES = (2, 3, 4)


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def bbox_of_coords(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def sort_vertices(vertices):
    pts = [(int(r), int(c)) for r, c in vertices]
    cr = sum(r for r, _c in pts) / len(pts)
    cc = sum(c for _r, c in pts) / len(pts)
    pts.sort(key=lambda rc: math.atan2(rc[0] - cr, rc[1] - cc))
    return pts


def line_points(a: tuple[int, int], b: tuple[int, int]):
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0:
        sr, sc, steps = 0, 1 if dc > 0 else -1, abs(dc)
    elif dc == 0:
        sr, sc, steps = 1 if dr > 0 else -1, 0, abs(dr)
    else:
        assert abs(dr) == abs(dc)
        sr = 1 if dr > 0 else -1
        sc = 1 if dc > 0 else -1
        steps = abs(dr)
    return [(r0 + i * sr, c0 + i * sc) for i in range(steps + 1)]


def convex_inside(point: tuple[int, int], vertices) -> bool:
    r, c = point
    crosses = []
    for i in range(len(vertices)):
        r0, c0 = vertices[i]
        r1, c1 = vertices[(i + 1) % len(vertices)]
        cross = (c - c0) * (r1 - r0) - (r - r0) * (c1 - c0)
        if cross == 0:
            return False
        crosses.append(cross)
    return all(v > 0 for v in crosses) or all(v < 0 for v in crosses)


def polygon_masks(shape: tuple[int, int], vertices):
    h, w = shape
    boundary = np.zeros(shape, dtype=bool)
    for i in range(len(vertices)):
        for r, c in line_points(vertices[i], vertices[(i + 1) % len(vertices)]):
            boundary[r, c] = True
    interior = np.zeros(shape, dtype=bool)
    for r in range(h):
        for c in range(w):
            if boundary[r, c]:
                continue
            if convex_inside((r, c), vertices):
                interior[r, c] = True
    return boundary, interior


def enumerate_candidates(sprite: np.ndarray, region_mask: np.ndarray, evidence, k_values=None):
    h, w = region_mask.shape
    if k_values is None:
        max_k = max(2, min(h, w))
        k_values = range(2, max_k + 1)
    sprite_mask = sprite != BLACK
    candidates = []
    for k in k_values:
        scaled = scale_sprite(sprite, k)
        sh, sw = scaled.shape
        if sh > h or sw > w:
            continue
        scaled_mask = np.kron(sprite_mask, np.ones((k, k), dtype=bool))
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                if not np.all(region_mask[r0 : r0 + sh, c0 : c0 + sw][scaled_mask]):
                    continue
                ok = True
                for er, ec, color in evidence:
                    rr = er - r0
                    cc = ec - c0
                    if rr < 0 or rr >= sh or cc < 0 or cc >= sw or int(scaled[rr, cc]) != int(color):
                        ok = False
                        break
                if ok:
                    candidates.append((int(k), int(r0), int(c0)))
    return candidates


def placement_lookup(sprite: np.ndarray, placement: tuple[int, int, int]):
    k, r0, c0 = placement
    scaled = scale_sprite(sprite, k)
    sh, sw = scaled.shape
    return scaled, r0, c0, sh, sw


def cells_for_placement(sprite: np.ndarray, placement: tuple[int, int, int]):
    scaled, r0, c0, sh, sw = placement_lookup(sprite, placement)
    cells = {}
    for rr in range(sh):
        for cc in range(sw):
            color = int(scaled[rr, cc])
            if color != BLACK:
                cells[(r0 + rr, c0 + cc)] = color
    return cells


def agreeing_cells(sprite: np.ndarray, a: tuple[int, int, int], b: tuple[int, int, int], interior_mask: np.ndarray):
    cells_a = cells_for_placement(sprite, a)
    cells_b = cells_for_placement(sprite, b)
    shared = []
    for cell in set(cells_a) & set(cells_b):
        if cells_a[cell] == cells_b[cell] and interior_mask[cell]:
            shared.append((cell[0], cell[1], cells_a[cell]))
    return shared


def spaced_subset(evidence):
    pts = [(r, c) for r, c, _color in evidence]
    for i, (r0, c0) in enumerate(pts):
        for r1, c1 in pts[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def choose_sprite(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 4)
    w = rng.randint(3, 4)
    colors = rng.sample(SPRITE_COLORS, rng.randint(3, 4))
    tile = np.array(
        [
            [colors[0], colors[1]],
            [colors[2], colors[0]],
        ],
        dtype=int,
    )
    sprite = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            sprite[r, c] = int(tile[r % 2, c % 2])
    for _ in range(rng.randint(1, 2)):
        rr = rng.randrange(h)
        cc = rng.randrange(w)
        sprite[rr, cc] = rng.choice(colors)
    if len(np.unique(sprite)) < 3:
        sprite[rng.randrange(h), rng.randrange(w)] = colors[2]
    if all(np.all(sprite[r] == sprite[0, 0]) for r in range(h)):
        sprite[0, 1] = colors[1]
    return sprite


def local_vertices(template: str, scaled_shape: tuple[int, int], rng: random.Random):
    sh, sw = scaled_shape
    if template == "diamond":
        a_min = max((sh + 2) // 2 + 1, (sw + 2) // 2 + 1, 4)
        a_max = min(10, 14)
        if a_min > a_max:
            return None
        a = rng.randint(a_min, a_max)
        return [(0, a), (a, 2 * a), (2 * a, a), (a, 0)]
    if template == "hhex":
        a_min = max((sh + 2) // 2 + 1, 4)
        a_max = min(10, 11)
        if a_min > a_max:
            return None
        a = rng.randint(a_min, a_max)
        w_min = max(sw + 4, 2 * a + 5)
        w_max = min(24, sw + 10)
        if w_min > w_max:
            return None
        width = rng.randint(w_min, w_max)
        return [(0, a), (0, width - 1 - a), (a, width - 1), (2 * a, width - 1 - a), (2 * a, a), (a, 0)]
    a_min = max((sw + 2) // 2 + 1, 4)
    a_max = min(10, 11)
    if a_min > a_max:
        return None
    a = rng.randint(a_min, a_max)
    h_min = max(sh + 4, 2 * a + 5)
    h_max = min(24, sh + 10)
    if h_min > h_max:
        return None
    height = rng.randint(h_min, h_max)
    return [(0, a), (a, 2 * a), (height - 1 - a, 2 * a), (height - 1, a), (height - 1 - a, 0), (a, 0)]


def translated_vertices(local, top: int, left: int):
    return [(top + r, left + c) for r, c in local]


def bbox_inner_mask(shape: tuple[int, int], vertices):
    r0, c0, r1, c1 = bbox_of_coords(vertices)
    mask = np.zeros(shape, dtype=bool)
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def build_output(grid: np.ndarray, vertices, sprite: np.ndarray, placement: tuple[int, int, int]):
    out = np.array(grid, copy=True)
    for i in range(len(vertices)):
        for r, c in line_points(vertices[i], vertices[(i + 1) % len(vertices)]):
            out[r, c] = YELLOW
    k, r0, c0 = placement
    scaled = scale_sprite(sprite, k)
    sh, sw = scaled.shape
    out[r0 : r0 + sh, c0 : c0 + sw] = scaled
    rmin, cmin, rmax, cmax = bbox_of_coords(vertices)
    return out[rmin : rmax + 1, cmin : cmax + 1]


def find_solution_data(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    vertices = sort_vertices(np.argwhere(g == YELLOW))
    boundary, interior = polygon_masks(g.shape, vertices)
    colored = (g != BLACK) & (g != YELLOW)
    evidence = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(colored & interior)]
    sprite_pixels = np.argwhere(colored & ~interior)
    assert len(sprite_pixels) > 0
    r0, c0, r1, c1 = bbox_of_coords(sprite_pixels)
    sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    candidates = enumerate_candidates(sprite, interior, evidence)
    assert len(candidates) == 1
    return vertices, interior, evidence, sprite, candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Scale the outside sprite into the yellow-vertex polygon so it fits strictly inside and matches the interior evidence.

    Input structure:
    - A mostly black grid contains 4-6 yellow singleton vertices of a convex polygon; only the vertices are drawn in the input.
    - Inside that future polygon lie 2-4 colored evidence pixels whose colors match some cells of a scaled copy of the outside sprite.
    - One small multicolor sprite sits outside the polygon area and uses no yellow cells.

    Rule:
    - Sort the yellow vertices by polar angle and connect consecutive vertices with yellow horizontal, vertical, or 45-degree outline segments.
    - Treat the polygon interior as the only legal placement region, then test integer scale factors and offsets for the outside sprite.
    - Keep the unique scaled placement whose non-black cells stay strictly inside the polygon and whose covered evidence pixels match in both position and color; draw that placement and keep the yellow outline.
    - Return the bounding-box crop of the polygon vertices.

    Key insight:
    - The yellow points do not mark a rectangle; they define a non-rectangular convex boundary, so the correct scale and offset are determined jointly by strict inside-polygon geometry and the sparse color evidence.
    """
    vertices, _interior, _evidence, sprite, placement = find_solution_data(grid)
    return build_output(np.asarray(grid, dtype=int), vertices, sprite, placement)


def _gen_attempt(rng: random.Random):
    sprite = choose_sprite(rng)
    scale = rng.choice(SCALES)
    scaled = scale_sprite(sprite, scale)
    sh, sw = scaled.shape

    template_order = list(TEMPLATES)
    rng.shuffle(template_order)
    for template in template_order:
        local = local_vertices(template, (sh, sw), rng)
        if local is None:
            continue
        lr0, lc0, lr1, lc1 = bbox_of_coords(local)
        poly_h = lr1 - lr0 + 1
        poly_w = lc1 - lc0 + 1
        min_h = max(poly_h + 4, sh + 8, sprite.shape[0] + 6, 20)
        min_w = max(poly_w + 4, sw + 8, sprite.shape[1] + 6, 20)
        if min_h > 29 or min_w > 29:
            continue
        grid_h = rng.randint(min_h, 29)
        grid_w = rng.randint(min_w, 29)
        if grid_h - poly_h - 6 < 0 or grid_w - poly_w - 6 < 0:
            continue
        top = rng.randint(3, grid_h - poly_h - 3)
        left = rng.randint(3, grid_w - poly_w - 3)
        vertices = translated_vertices(local, top, left)
        boundary, interior = polygon_masks((grid_h, grid_w), vertices)
        bbox_mask = bbox_inner_mask((grid_h, grid_w), vertices)

        poly_candidates = enumerate_candidates(sprite, interior, [], k_values=SCALES)
        bbox_candidates = enumerate_candidates(sprite, bbox_mask, [], k_values=SCALES)
        extra_bbox = [cand for cand in bbox_candidates if cand not in poly_candidates]
        if len(poly_candidates) < 2 or not extra_bbox:
            continue

        target = None
        evidence = None
        bbox_evidence = None
        for candidate in poly_candidates:
            witness_list = extra_bbox[:]
            rng.shuffle(witness_list)
            for witness in witness_list:
                pool = agreeing_cells(sprite, candidate, witness, interior)
                rng.shuffle(pool)
                if len(pool) < 2:
                    continue
                trial_points = pool[: min(12, len(pool))]
                for size in (2, 3, 4):
                    if len(trial_points) < size:
                        continue
                    for combo in itertools.combinations(trial_points, size):
                        ev = list(combo)
                        if not spaced_subset(ev):
                            continue
                        full = enumerate_candidates(sprite, interior, ev, k_values=SCALES)
                        if full != [candidate]:
                            continue
                        evidence_only = enumerate_candidates(
                            sprite,
                            np.ones((grid_h, grid_w), dtype=bool),
                            ev,
                            k_values=SCALES,
                        )
                        bbox_with = enumerate_candidates(sprite, bbox_mask, ev, k_values=SCALES)
                        if len(evidence_only) <= 1 or len(bbox_with) <= 1:
                            continue
                        target = candidate
                        evidence = ev
                        bbox_evidence = bbox_with
                        break
                    if target is not None:
                        break
                if target is not None:
                    break
            if target is not None:
                break
        if target is None:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        for r, c in vertices:
            grid[r, c] = YELLOW
        for r, c, color in evidence:
            grid[r, c] = int(color)

        vr0, vc0, vr1, vc1 = bbox_of_coords(vertices)
        corners = [
            (1, 1),
            (1, grid_w - sprite.shape[1] - 1),
            (grid_h - sprite.shape[0] - 1, 1),
            (grid_h - sprite.shape[0] - 1, grid_w - sprite.shape[1] - 1),
        ]
        rng.shuffle(corners)
        sprite_pos = None
        for sr0, sc0 in corners:
            sr1 = sr0 + sprite.shape[0] - 1
            sc1 = sc0 + sprite.shape[1] - 1
            if sr0 < 0 or sc0 < 0:
                continue
            if not (sr1 < grid_h and sc1 < grid_w):
                continue
            disjoint = sr1 < vr0 - 1 or sr0 > vr1 + 1 or sc1 < vc0 - 1 or sc0 > vc1 + 1
            if disjoint:
                sprite_pos = (sr0, sc0)
                break
        if sprite_pos is None:
            continue
        sr0, sc0 = sprite_pos
        grid[sr0 : sr0 + sprite.shape[0], sc0 : sc0 + sprite.shape[1]] = sprite

        try:
            found_vertices, found_interior, found_evidence, found_sprite, found_target = find_solution_data(grid)
        except Exception:
            continue
        if found_vertices != vertices:
            continue
        if found_target != target:
            continue
        if len(found_evidence) != len(evidence):
            continue
        if not np.array_equal(found_sprite, sprite):
            continue

        expected = build_output(grid, vertices, sprite, target)
        actual = main(grid)
        if not np.array_equal(expected, actual):
            continue
        if len(bbox_evidence) <= 1:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
