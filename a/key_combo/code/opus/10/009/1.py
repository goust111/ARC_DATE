import random
from collections import Counter

import numpy as np


BACKGROUND = 0
DEFAULT_OUTLINE_COLOR = 3
MARKER_POOL = (1, 3, 4, 8, 9)
CONTROL_POOL = (2, 5, 6, 7)
MAX_GENERATION_ATTEMPTS = 600


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    length = max(abs(dr), abs(dc))
    assert length > 0
    assert abs(dr) in (0, length)
    assert abs(dc) in (0, length)
    sr = _sign(dr)
    sc = _sign(dc)
    return [(r0 + i * sr, c0 + i * sc) for i in range(length + 1)]


def _cross(a: tuple[int, int], b: tuple[int, int], p: tuple[int, int]) -> int:
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    px, py = p[1], p[0]
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _strictly_inside_polygon(point: tuple[int, int], poly: list[tuple[int, int]]) -> bool:
    signs = []
    for i in range(len(poly)):
        z = _cross(poly[i], poly[(i + 1) % len(poly)], point)
        if z == 0:
            return False
        signs.append(1 if z > 0 else -1)
    return all(s == signs[0] for s in signs)


def _decompose_vector(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    assert abs_dr > 0 and abs_dc > 0 and abs_dr != abs_dc

    diag_len = min(abs_dr, abs_dc)
    sr = _sign(dr)
    sc = _sign(dc)
    diag_vec = (sr * diag_len, sc * diag_len)
    if abs_dr > abs_dc:
        straight_vec = (sr * (abs_dr - abs_dc), 0)
    else:
        straight_vec = (0, sc * (abs_dc - abs_dr))
    return straight_vec, diag_vec


def _shape_from_markers(p: tuple[int, int], q: tuple[int, int]) -> dict:
    straight_vec, diag_vec = _decompose_vector(q[0] - p[0], q[1] - p[1])
    straight_corner = (p[0] + straight_vec[0], p[1] + straight_vec[1])
    diag_corner = (p[0] + diag_vec[0], p[1] + diag_vec[1])

    edges = (
        (p, straight_corner),
        (p, diag_corner),
        (straight_corner, q),
        (diag_corner, q),
    )
    outline = set()
    for a, b in edges:
        outline.update(_segment_cells(a, b))

    polygon = [p, straight_corner, q, diag_corner]
    rows = [r for r, _c in polygon]
    cols = [c for _r, c in polygon]
    interior = set()
    for r in range(min(rows), max(rows) + 1):
        for c in range(min(cols), max(cols) + 1):
            cell = (r, c)
            if cell in outline:
                continue
            if _strictly_inside_polygon(cell, polygon):
                interior.add(cell)

    return {
        "markers": (p, q),
        "outline": outline,
        "interior": interior,
        "polygon": polygon,
        "area": len(interior),
        "abs_dr": abs(q[0] - p[0]),
        "abs_dc": abs(q[1] - p[1]),
    }


def _extract_problem(grid: np.ndarray) -> tuple[list[dict], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, _W = g.shape
    assert H >= 2

    control_cols = np.flatnonzero(g[-1] != BACKGROUND)
    assert control_cols.size > 0
    control_colors = [int(g[-1, c]) for c in control_cols]
    assert len(control_colors) == len(set(control_colors))
    control_set = set(control_colors)

    body = g[:-1]
    counts = Counter(int(v) for v in body.flat if int(v) != BACKGROUND)
    marker_colors = [color for color, count in counts.items() if count == 2 and color not in control_set]
    assert len(marker_colors) == len(control_colors)

    shapes = []
    for color in sorted(marker_colors):
        pts = np.argwhere(body == color)
        assert pts.shape == (2, 2)
        p, q = sorted((tuple(map(int, pts[0])), tuple(map(int, pts[1]))))
        shape = _shape_from_markers(p, q)
        shape["marker_color"] = int(color)
        shapes.append(shape)

    areas = [shape["area"] for shape in shapes]
    assert all(area > 0 for area in areas)
    assert len(set(areas)) == len(areas)
    return shapes, control_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each implied parallelogram outline, rank the shapes by enclosed area, and recolor those outlines from the bottom control row.

    Input structure:
    - A black background contains several same-color marker pairs above a bottom control row.
    - Each marker color appears exactly twice and the bottom row contains one pixel of each distinct control color in left-to-right rank order.
    - For every marker pair, the row/column offset has both components and their magnitudes are unequal, so the diagonal-plus-axis decomposition is unique.

    Rule:
    - For each marker pair, decompose the marker-to-marker vector into a 45-degree diagonal vector plus the remaining axis-aligned vector, then use those as adjacent sides of the parallelogram whose opposite corners are the markers.
    - Draw the four outline edges, keep marker vertices excluded from recoloring, and count the strictly interior cells enclosed by that outline.
    - Rank the parallelograms by enclosed area from largest to smallest, recolor the non-marker outline cells of rank i with the i-th control-row color, then erase the control row and the marker pixels.

    Key insight:
    - The final outline color is determined only indirectly: you must reconstruct every parallelogram, measure its enclosed interior, and only then use the relative area ranking to read the correct replacement color from the control row.
    """
    g = np.asarray(grid, dtype=int)
    shapes, control_colors = _extract_problem(g)

    out = np.zeros_like(g)
    ranked = sorted(shapes, key=lambda shape: shape["area"], reverse=True)
    for rank, shape in enumerate(ranked):
        color = int(control_colors[rank])
        marker_set = set(shape["markers"])
        for r, c in shape["outline"]:
            if (r, c) not in marker_set:
                out[r, c] = color
    return out


def _rank_inversion_exists(primary: list[int], secondary: list[int]) -> bool:
    for i in range(len(primary)):
        for j in range(i + 1, len(primary)):
            if primary[i] > primary[j] and secondary[i] < secondary[j]:
                return True
            if primary[i] < primary[j] and secondary[i] > secondary[j]:
                return True
    return False


def _expand8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _translated_shape(abs_dr: int, abs_dc: int, sr: int, sc: int) -> dict:
    p = (0, 0)
    q = (sr * abs_dr, sc * abs_dc)
    shape = _shape_from_markers(p, q)
    shape["dr"] = q[0] - p[0]
    shape["dc"] = q[1] - p[1]
    return shape


def _all_blueprints() -> list[dict]:
    blueprints = []
    for abs_dr in range(2, 12):
        for abs_dc in range(2, 12):
            if abs_dr == abs_dc:
                continue
            shape = _translated_shape(abs_dr, abs_dc, 1, 1)
            area = int(shape["area"])
            if area <= 0:
                continue
            blueprints.append(
                {
                    "abs_dr": abs_dr,
                    "abs_dc": abs_dc,
                    "area": area,
                    "manhattan": abs_dr + abs_dc,
                    "outline_len": len(shape["outline"]),
                    "dominant": "row" if abs_dr > abs_dc else "col",
                }
            )
    return blueprints


CANDIDATE_BLUEPRINTS = _all_blueprints()


def _choose_blueprints(k: int) -> list[dict] | None:
    for _ in range(300):
        picks = random.sample(CANDIDATE_BLUEPRINTS, k)
        areas = [pick["area"] for pick in picks]
        if len(set(areas)) != k:
            continue
        if {pick["dominant"] for pick in picks} != {"row", "col"}:
            continue
        if not _rank_inversion_exists(areas, [pick["manhattan"] for pick in picks]):
            continue
        if not _rank_inversion_exists(areas, [pick["outline_len"] for pick in picks]):
            continue
        return picks
    return None


def _place_shapes(h: int, w: int, blueprints: list[dict]) -> list[dict] | None:
    body_h = h - 1
    used = set()
    padded = set()
    placed = []

    order = sorted(blueprints, key=lambda item: (-item["area"], -(item["abs_dr"] + item["abs_dc"])))
    sign_choices = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    random.shuffle(sign_choices)

    for idx, blueprint in enumerate(order):
        shape = None
        choices = sign_choices[:] if idx < len(sign_choices) else sign_choices[:] + sign_choices[:]
        random.shuffle(choices)
        for sr, sc in choices:
            candidate = _translated_shape(blueprint["abs_dr"], blueprint["abs_dc"], sr, sc)
            footprint = candidate["outline"] | candidate["interior"]
            rows = [r for r, _c in footprint]
            cols = [c for _r, c in footprint]
            r_min, r_max = min(rows), max(rows)
            c_min, c_max = min(cols), max(cols)
            r_lo = -r_min
            r_hi = body_h - 1 - r_max
            c_lo = -c_min
            c_hi = w - 1 - c_max
            if r_lo > r_hi or c_lo > c_hi:
                continue

            translations = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
            random.shuffle(translations)
            for tr, tc in translations[:160]:
                translated_outline = {(r + tr, c + tc) for r, c in candidate["outline"]}
                translated_interior = {(r + tr, c + tc) for r, c in candidate["interior"]}
                future_cells = translated_outline | translated_interior
                if future_cells & padded:
                    continue
                candidate = {
                    **candidate,
                    "outline": translated_outline,
                    "interior": translated_interior,
                    "markers": tuple((r + tr, c + tc) for r, c in candidate["markers"]),
                    "polygon": [(r + tr, c + tc) for r, c in candidate["polygon"]],
                }
                shape = candidate
                break
            if shape is not None:
                break
        if shape is None:
            return None

        footprint = shape["outline"] | shape["interior"]
        used |= footprint
        padded |= _expand8(footprint, body_h, w)
        placed.append(shape)

    return placed


def _sample_control_columns(w: int, k: int) -> list[int] | None:
    candidates = list(range(1, w - 1))
    for _ in range(100):
        cols = sorted(random.sample(candidates, k))
        if all(cols[i + 1] - cols[i] >= 2 for i in range(k - 1)):
            return cols
    return None


def _build_expected_output(shape_specs: list[dict], control_colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ranked = sorted(shape_specs, key=lambda spec: spec["area"], reverse=True)
    for rank, spec in enumerate(ranked):
        marker_set = set(spec["markers"])
        for r, c in spec["outline"]:
            if (r, c) not in marker_set:
                out[r, c] = int(control_colors[rank])
    return out


def _semantic_checks(grid: np.ndarray, shape_specs: list[dict], control_colors: list[int]) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False
    if np.any(grid[:-1] == np.array(control_colors)[:, None, None]):
        return False

    areas = [spec["area"] for spec in shape_specs]
    if len(set(areas)) != len(areas):
        return False
    if not _rank_inversion_exists(
        areas, [spec["abs_dr"] + spec["abs_dc"] for spec in shape_specs]
    ):
        return False
    centers = [(spec["markers"][0][0] + spec["markers"][1][0], spec["markers"][0][1] + spec["markers"][1][1]) for spec in shape_specs]
    if sorted(range(len(shape_specs)), key=lambda i: areas[i], reverse=True) == sorted(range(len(shape_specs)), key=lambda i: centers[i]):
        return False

    parsed_shapes, parsed_controls = _extract_problem(grid)
    if parsed_controls != control_colors:
        return False
    if sorted(spec["area"] for spec in parsed_shapes) != sorted(areas):
        return False

    expected = _build_expected_output(shape_specs, control_colors, grid.shape)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual[-1] != BACKGROUND):
        return False
    if set(np.unique(actual)) - {BACKGROUND} != set(control_colors):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 26)
    k = random.randint(3, 4)

    blueprints = _choose_blueprints(k)
    if blueprints is None:
        return None
    shape_specs = _place_shapes(h, w, blueprints)
    if shape_specs is None:
        return None

    marker_colors = random.sample(MARKER_POOL, k)
    control_colors = random.sample(CONTROL_POOL, k)
    random.shuffle(control_colors)
    control_cols = _sample_control_columns(w, k)
    if control_cols is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for color, spec in zip(marker_colors, shape_specs):
        spec["marker_color"] = int(color)
        for r, c in spec["markers"]:
            grid[r, c] = int(color)
    for c, color in zip(control_cols, control_colors):
        grid[-1, c] = int(color)

    if not _semantic_checks(grid, shape_specs, control_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
