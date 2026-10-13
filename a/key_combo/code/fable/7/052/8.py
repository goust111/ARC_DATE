import math
import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _isolated_same_color(grid: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    for r, c in np.argwhere(grid == color):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = int(r) + dr
                cc = int(c) + dc
                if in_bounds((h, w), rr, cc) and int(grid[rr, cc]) == color:
                    return False
    return True


def _aligned_8(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = _step_dir(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + sr * k, a[1] + sc * k) for k in range(steps + 1)]


def _angle_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points) / len(points)
    cc = sum(c for _, c in points) / len(points)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return ang, rad, r, c

    return sorted(points, key=key)


def _interior_from_boundary(boundary: np.ndarray) -> np.ndarray:
    h, w = boundary.shape
    outside = np.zeros((h, w), dtype=bool)
    stack = [(r, c) for r in range(h) for c in range(w) if (r in (0, h - 1) or c in (0, w - 1)) and not boundary[r, c]]
    while stack:
        r, c = stack.pop()
        if not in_bounds((h, w), r, c) or outside[r, c] or boundary[r, c]:
            continue
        outside[r, c] = True
        for dr, dc in DIR4:
            stack.append((r + dr, c + dc))
    return (~boundary) & (~outside)


def _analyze_polygon(grid: np.ndarray, color: int) -> dict | None:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    if len(pts) < 4:
        return None
    if not _isolated_same_color(grid, color):
        return None

    cycle = _angle_cycle(pts)
    boundary_hits: dict[tuple[int, int], int] = {}
    segments: list[list[tuple[int, int]]] = []
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        if not _aligned_8(a, b):
            return None
        seg = _line_cells(a, b)
        for rr, cc in seg[1:-1]:
            if int(grid[rr, cc]) != BG:
                return None
        segments.append(seg)
        for cell in seg:
            boundary_hits[cell] = boundary_hits.get(cell, 0) + 1

    vertices = set(cycle)
    for cell, hits in boundary_hits.items():
        if cell in vertices:
            if hits != 2:
                return None
        elif hits != 1:
            return None

    boundary_cells = sorted(boundary_hits)
    r0, c0, r1, c1 = bbox_coords(boundary_cells)
    local_boundary = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in boundary_cells:
        local_boundary[rr - r0, cc - c0] = True
    interior = _interior_from_boundary(local_boundary)
    if not interior.any():
        return None

    seed_pixels = []
    for rr, cc in np.argwhere(interior):
        vr = r0 + int(rr)
        vc = c0 + int(cc)
        val = int(grid[vr, vc])
        if val != BG:
            if val == color:
                return None
            seed_pixels.append((vr, vc, val))
    if len(seed_pixels) != 1:
        return None

    return {
        "color": int(color),
        "cycle": cycle,
        "boundary_cells": boundary_cells,
        "bbox": (r0, c0, r1, c1),
        "interior": interior,
        "seed": seed_pixels[0],
    }


def _find_polygons(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == BG, "expected black background"
    infos = []
    used_boundary = set()
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        info = _analyze_polygon(g, color)
        if info is None:
            continue
        boundary = set(info["boundary_cells"])
        assert used_boundary.isdisjoint(boundary), "polygon outlines must not overlap"
        used_boundary |= boundary
        infos.append(info)
    assert infos, "expected at least one valid polygon vertex group"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored vertex set into a polygon outline and fill its interior with the color of the unique seed pixel inside it.

    Input structure:
    - The grid has a black background with several colors appearing as isolated pixels.
    - Some colors form groups of at least four isolated vertices; each such group's polar-angle cycle is a simple polygon whose edges are horizontal, vertical, or 45-degree diagonals.
    - Inside each implied polygon there is exactly one isolated non-black seed pixel of another color, and there may also be isolated distractor dots outside every polygon.

    Rule:
    - For every color that forms a valid vertex group, sort its pixels by polar angle around their centroid and connect consecutive vertices cyclically with that same color.
    - Treat the connected cycle as the polygon outline.
    - Fill every interior cell of that outline with the color of the unique non-black seed pixel inside it, leaving exterior distractor dots unchanged.

    Key insight:
    - The important objects are the un-drawn polygons implied by same-colored vertices; a seed color matters only when it lies inside one of those reconstructed outlines, while the same color outside is just noise.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for info in _find_polygons(g):
        outline_color = info["color"]
        for r, c in info["boundary_cells"]:
            out[r, c] = outline_color
        r0, c0, _, _ = info["bbox"]
        seed_color = info["seed"][2]
        for rr, cc in np.argwhere(info["interior"]):
            out[r0 + int(rr), c0 + int(cc)] = seed_color
    return out


def _shape_rectangle(rng: random.Random) -> list[tuple[int, int]]:
    h = rng.randint(4, 8)
    w = rng.randint(4, 8)
    return [(0, 0), (0, w), (h, w), (h, 0)]


def _shape_diamond(rng: random.Random) -> list[tuple[int, int]]:
    arm = rng.randint(2, 4)
    return [(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]


def _shape_house(rng: random.Random) -> list[tuple[int, int]]:
    roof = rng.randint(2, 4)
    wall = rng.randint(2, 5)
    return [(0, roof), (roof, 0), (roof + wall, 0), (roof + wall, 2 * roof), (roof, 2 * roof)]


def _shape_hexagon(rng: random.Random) -> list[tuple[int, int]]:
    slant = rng.randint(2, 4)
    wall = rng.randint(2, 4)
    return [
        (0, slant),
        (slant, 0),
        (slant + wall, 0),
        (2 * slant + wall, slant),
        (slant + wall, 2 * slant),
        (slant, 2 * slant),
    ]


def _shape_trapezoid(rng: random.Random) -> list[tuple[int, int]]:
    slant = rng.randint(2, 4)
    top = rng.randint(2, 5)
    return [(0, slant), (0, slant + top), (slant, top + 2 * slant), (slant, 0)]


ALL_BUILDERS = [_shape_rectangle, _shape_diamond, _shape_house, _shape_hexagon, _shape_trapezoid]
DIAGONAL_BUILDERS = [_shape_diamond, _shape_house, _shape_hexagon, _shape_trapezoid]
COMPLEX_BUILDERS = [_shape_house, _shape_hexagon]


def _transform_points(points: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]]:
    arr = np.asarray(points, dtype=int)
    rot = rng.randrange(4)
    for _ in range(rot):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    if rng.random() < 0.5:
        arr[:, 1] *= -1
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return [tuple(map(int, rc)) for rc in arr]


def _render_shape(vertices: list[tuple[int, int]], outline_color: int, seed_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h = max(r for r, _ in vertices) + 1
    w = max(c for _, c in vertices) + 1
    boundary = np.zeros((h, w), dtype=bool)
    for i in range(len(vertices)):
        for r, c in _line_cells(vertices[i], vertices[(i + 1) % len(vertices)]):
            boundary[r, c] = True
    interior = _interior_from_boundary(boundary)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
    assert len(cells) >= 2, "generator expects non-trivial polygon interior"
    seed_r, seed_c = rng.choice(cells)

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for r, c in vertices:
        inp[r, c] = outline_color
    inp[seed_r, seed_c] = seed_color
    out[boundary] = outline_color
    out[interior] = seed_color
    return inp, out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return None
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    rng.shuffle(positions)
    for r0, c0 in positions[:200]:
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if all(_boxes_separated(box, prev, gap=1) for prev in boxes):
            view = canvas[r0 : r0 + ph, c0 : c0 + pw]
            if not np.any((view != BG) & (patch != BG)):
                canvas[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]
                boxes.append(box)
                return r0, c0
    return None


def _pick_far_cell(grid: np.ndarray, forbidden: set[tuple[int, int]], rng: random.Random) -> tuple[int, int] | None:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    for r, c in cells:
        if int(grid[r, c]) != BG or (r, c) in forbidden:
            continue
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if in_bounds((h, w), rr, cc) and int(grid[rr, cc]) != BG:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return r, c
    return None


def _generate_case(rng: random.Random) -> np.ndarray | None:
    shape_count = rng.randint(2, 4)
    chosen_builders = [rng.choice(DIAGONAL_BUILDERS), rng.choice(COMPLEX_BUILDERS)]
    while len(chosen_builders) < shape_count:
        chosen_builders.append(rng.choice(ALL_BUILDERS))
    rng.shuffle(chosen_builders)

    colors = rng.sample(PALETTE, 2 * shape_count)
    outline_colors = colors[:shape_count]
    seed_colors = colors[shape_count:]

    patches = []
    diagonal_present = False
    for builder, outline_color, seed_color in zip(chosen_builders, outline_colors, seed_colors):
        vertices = _transform_points(builder(rng), rng)
        inp_patch, out_patch = _render_shape(vertices, outline_color, seed_color, rng)
        patch_has_diag = any(
            abs(vertices[(i + 1) % len(vertices)][0] - vertices[i][0])
            == abs(vertices[(i + 1) % len(vertices)][1] - vertices[i][1])
            and vertices[(i + 1) % len(vertices)][0] != vertices[i][0]
            for i in range(len(vertices))
        )
        diagonal_present = diagonal_present or patch_has_diag
        patches.append(
            {
                "input": inp_patch,
                "output": out_patch,
                "outline": outline_color,
                "seed": seed_color,
                "vertices": vertices,
            }
        )
    if not diagonal_present:
        return None

    max_h = max(p["input"].shape[0] for p in patches)
    max_w = max(p["input"].shape[1] for p in patches)
    grid_h = rng.randint(max(18, max_h + 6), 30)
    grid_w = rng.randint(max(18, max_w + 6), 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    for patch in sorted(patches, key=lambda item: item["input"].size, reverse=True):
        placed = _place_patch(grid, patch["input"], placed_boxes, rng)
        if placed is None:
            return None
        r0, c0 = placed
        ph, pw = patch["input"].shape
        patch["origin"] = (r0, c0)
        patch["box"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        out_patch = patch["output"]
        for rr in range(ph):
            for cc in range(pw):
                if out_patch[rr, cc] != BG:
                    forbidden.add((r0 + rr, c0 + cc))

    # Add outside distractors matching some seed colors, but keep each seed color below 4 total pixels.
    used_per_seed = {color: 1 for color in seed_colors}
    distractor_total = rng.randint(1, min(4, shape_count + 1))
    for _ in range(distractor_total):
        allowed = [color for color in seed_colors if used_per_seed[color] < 3]
        if not allowed:
            break
        cell = _pick_far_cell(grid, forbidden, rng)
        if cell is None:
            return None
        color = rng.choice(allowed)
        grid[cell] = color
        used_per_seed[color] += 1
        forbidden.add(cell)

    return grid


def _validate_generated(inp: np.ndarray) -> None:
    infos = _find_polygons(inp)
    assert 2 <= len(infos) <= 4
    assert any(
        abs(info["cycle"][(i + 1) % len(info["cycle"])][0] - info["cycle"][i][0])
        == abs(info["cycle"][(i + 1) % len(info["cycle"])][1] - info["cycle"][i][1])
        and info["cycle"][(i + 1) % len(info["cycle"])][0] != info["cycle"][i][0]
        for info in infos
        for i in range(len(info["cycle"]))
    ), "at least one polygon should require diagonal edges"
    assert any(len(info["cycle"]) > 4 for info in infos), "include a non-rectangular polygon"

    out = main(inp)
    assert out.shape == inp.shape and max(out.shape) <= 30
    assert not np.array_equal(inp, out)

    polygon_cells = set()
    for info in infos:
        r0, c0, _, _ = info["bbox"]
        polygon_cells.update(info["boundary_cells"])
        for rr, cc in np.argwhere(info["interior"]):
            polygon_cells.add((r0 + int(rr), c0 + int(cc)))

        seed_r, seed_c, seed_color = info["seed"]
        assert int(inp[seed_r, seed_c]) == seed_color
        for rr, cc in info["boundary_cells"]:
            assert int(out[rr, cc]) == info["color"]
        for rr, cc in np.argwhere(info["interior"]):
            assert int(out[r0 + int(rr), c0 + int(cc)]) == seed_color

    distractors = []
    for r, c in np.argwhere(inp != BG):
        rc = (int(r), int(c))
        if rc not in polygon_cells:
            distractors.append(rc)
    assert distractors, "need outside distractor dots"
    for r, c in distractors:
        assert int(inp[r, c]) == int(out[r, c])


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _generate_case(rng)
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid ARC puzzle input")
