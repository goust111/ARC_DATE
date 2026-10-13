from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
FILL = 4
FRAME = 5
LEAF_COLORS = (BLUE, RED, GREEN)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _rect_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    return out


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    return pts - pts.min(axis=0)


def _coords_to_patch(coords: np.ndarray, color: int) -> np.ndarray:
    pts = _normalize_coords(coords)
    h = int(pts[:, 0].max()) + 1
    w = int(pts[:, 1].max()) + 1
    patch = np.full((h, w), BG, dtype=int)
    patch[pts[:, 0], pts[:, 1]] = int(color)
    return patch


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _bbox(g == FRAME)
    assert frame_bbox is not None, "expected a grey frame"
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must have a meaningful interior"
    assert np.array_equal(g == FRAME, _rect_border_mask(g.shape, frame_bbox)), "grey pixels must form one rectangle border"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == FILL), "frame interior must start as solid yellow"

    leaves = []
    templates: dict[int, dict] = {}
    template_colors = set()

    for color in LEAF_COLORS:
        for comp in _components_of_color(g, color):
            area = int(comp.shape[0])
            if area == 1:
                r, c = map(int, comp[0])
                frame_neighbors = []
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == FRAME:
                        frame_neighbors.append((nr, nc))
                assert len(frame_neighbors) == 1, "leaf must touch exactly one frame cell"
                nr, nc = frame_neighbors[0]
                if nr == r:
                    axis = "row"
                    value = r
                else:
                    axis = "col"
                    value = c
                leaves.append({"color": int(color), "axis": axis, "value": int(value), "coord": (r, c)})
            else:
                assert 3 <= area <= 5, "template sprite area must be 3..5"
                tr0, tc0, tr1, tc1 = _bbox_from_coords(comp)
                assert tr1 < r0 - 1 or tr0 > r1 + 1 or tc1 < c0 - 1 or tc0 > c1 + 1, (
                    "template sprite must be outside and away from the frame"
                )
                assert int(color) not in templates, "expected at most one template per color"
                templates[int(color)] = {
                    "color": int(color),
                    "coords_abs": comp.copy(),
                    "coords_rel": _normalize_coords(comp),
                    "patch": _coords_to_patch(comp, int(color)),
                }
                template_colors.add(int(color))

    row_leaves = [leaf for leaf in leaves if leaf["axis"] == "row"]
    col_leaves = [leaf for leaf in leaves if leaf["axis"] == "col"]
    assert 2 <= len(leaves) <= 6 and row_leaves and col_leaves, "expected row and column leaves"
    assert set(leaf["color"] for leaf in leaves).issubset(template_colors), "each leaf color needs a template"

    return {
        "grid": g,
        "frame_bbox": frame_bbox,
        "leaves": leaves,
        "row_leaves": row_leaves,
        "col_leaves": col_leaves,
        "templates": templates,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each color-matched row/column leaf intersection with its external template, then erase the templates.

    Input structure:
    - A black background contains one grey one-pixel-thick rectangle whose interior is solid yellow.
    - Single-cell red/blue/green leaves sit just outside the frame, each touching one frame cell on a side.
    - Small irregular monochrome template sprites of the same colors appear elsewhere outside the frame.

    Rule:
    - A leaf touching the frame from the left or right selects its row; a leaf touching from above or below selects its column.
    - Pair every row-selecting leaf with every column-selecting leaf, and keep only the pairs whose colors match.
    - For each surviving pair, in row-major order, align that color's template bbox top-left to the row/column intersection and paint only the yellow cells there.
    - Erase the external template sprites to black, while keeping the frame, yellow fill, and leaf markers.

    Key insight:
    - The leaves provide coordinates, but color equality gates which row/column intersections become stamp anchors.
    """
    scene = _parse_scene(grid)
    g = scene["grid"]
    out = g.copy()

    for template in scene["templates"].values():
        coords = template["coords_abs"]
        out[coords[:, 0], coords[:, 1]] = BG

    valid_points = []
    for row_leaf in scene["row_leaves"]:
        for col_leaf in scene["col_leaves"]:
            if row_leaf["color"] == col_leaf["color"]:
                valid_points.append((row_leaf["value"], col_leaf["value"], row_leaf["color"]))
    valid_points.sort()

    for row, col, color in valid_points:
        for dr, dc in scene["templates"][color]["coords_rel"]:
            rr = int(row + dr)
            cc = int(col + dc)
            if int(out[rr, cc]) == FILL:
                out[rr, cc] = int(color)

    return out


_BASE_SHAPES = [
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1), (1, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 1)], dtype=int),
    np.array([(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)], dtype=int),
]


def _transform_shape(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    transforms = [
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    ]
    fn = random.choice(transforms)
    moved = np.array([fn(int(r), int(c)) for r, c in pts], dtype=int)
    return _normalize_coords(moved)


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    pts = _normalize_coords(coords)
    return int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1


def _choose_shapes(used_colors: list[int]) -> dict[int, np.ndarray]:
    attempts = 0
    while attempts < 200:
        attempts += 1
        base_choices = random.sample(_BASE_SHAPES, len(used_colors))
        shapes = {color: _transform_shape(base) for color, base in zip(used_colors, base_choices)}
        keys = {color: tuple(sorted(map(tuple, shapes[color].tolist()))) for color in used_colors}
        if len(set(keys.values())) == len(used_colors):
            return shapes
    raise RuntimeError("failed to choose distinct template shapes")


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_leaf_spec(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    shapes: dict[int, np.ndarray],
    used_colors: list[int],
) -> tuple[list[dict], list[dict], list[tuple[int, int, int]], set[tuple[int, int]]]:
    for _ in range(400):
        n_rows = random.randint(2, 3)
        n_cols = random.randint(2, 3)
        row_colors = [random.choice(used_colors) for _ in range(n_rows)]
        col_colors = [random.choice(used_colors) for _ in range(n_cols)]

        valid_count = sum(int(rc == cc) for rc in row_colors for cc in col_colors)
        if valid_count == 0 or valid_count == n_rows * n_cols:
            continue

        row_leaves = []
        used_rows: list[int] = []
        ok = True
        for color in row_colors:
            sh, _sw = _shape_dims(shapes[color])
            candidates = [
                r
                for r in range(r0 + 1, r1 - sh + 1)
                if all(abs(r - other) > 1 for other in used_rows)
            ]
            if not candidates:
                ok = False
                break
            row = random.choice(candidates)
            used_rows.append(row)
            row_leaves.append(
                {"axis": "row", "color": int(color), "value": int(row), "side": random.choice(("left", "right"))}
            )
        if not ok:
            continue

        col_leaves = []
        used_cols: list[int] = []
        for color in col_colors:
            _sh, sw = _shape_dims(shapes[color])
            candidates = [
                c
                for c in range(c0 + 1, c1 - sw + 1)
                if all(abs(c - other) > 1 for other in used_cols)
            ]
            if not candidates:
                ok = False
                break
            col = random.choice(candidates)
            used_cols.append(col)
            col_leaves.append(
                {"axis": "col", "color": int(color), "value": int(col), "side": random.choice(("top", "bottom"))}
            )
        if not ok:
            continue

        valid = []
        rects = []
        covered_cells: set[tuple[int, int]] = set()
        for row_leaf in row_leaves:
            for col_leaf in col_leaves:
                if row_leaf["color"] != col_leaf["color"]:
                    continue
                color = int(row_leaf["color"])
                row = int(row_leaf["value"])
                col = int(col_leaf["value"])
                sh, sw = _shape_dims(shapes[color])
                rects.append((row, col, row + sh - 1, col + sw - 1))
                valid.append((row, col, color))
                for dr, dc in shapes[color]:
                    covered_cells.add((row + int(dr), col + int(dc)))

        if any(_rectangles_overlap(rects[i], rects[j]) for i in range(len(rects)) for j in range(i + 1, len(rects))):
            continue

        mismatch_visible = False
        for row_leaf in row_leaves:
            for col_leaf in col_leaves:
                if row_leaf["color"] == col_leaf["color"]:
                    continue
                anchor = (int(row_leaf["value"]), int(col_leaf["value"]))
                if anchor not in covered_cells:
                    mismatch_visible = True
                    break
            if mismatch_visible:
                break
        if not mismatch_visible:
            continue

        valid.sort()
        return row_leaves, col_leaves, valid, covered_cells

    raise RuntimeError("failed to sample a non-degenerate leaf layout")


def _build_input_from_spec(
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    shapes: dict[int, np.ndarray],
    row_leaves: list[dict],
    col_leaves: list[dict],
) -> np.ndarray | None:
    r0, c0, r1, c1 = frame_bbox
    grid = np.full((h, w), BG, dtype=int)
    border = _rect_border_mask((h, w), frame_bbox)
    grid[border] = FRAME
    grid[r0 + 1 : r1, c0 + 1 : c1] = FILL

    for leaf in row_leaves:
        r = int(leaf["value"])
        c = c0 - 1 if leaf["side"] == "left" else c1 + 1
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(leaf["color"])
        leaf["coord"] = (r, c)

    for leaf in col_leaves:
        r = r0 - 1 if leaf["side"] == "top" else r1 + 1
        c = int(leaf["value"])
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(leaf["color"])
        leaf["coord"] = (r, c)

    zones = [
        (0, r0 - 2, 0, c0 - 2),
        (0, r0 - 2, c1 + 2, w - 1),
        (r1 + 2, h - 1, 0, c0 - 2),
        (r1 + 2, h - 1, c1 + 2, w - 1),
    ]
    zone_order = random.sample(zones, len(shapes))

    occupied = grid != BG
    for color, zone in zip(random.sample(list(shapes.keys()), len(shapes)), zone_order):
        zr0, zr1, zc0, zc1 = zone
        patch = _coords_to_patch(shapes[color], int(color))
        ph, pw = patch.shape
        candidates = []
        for rr in range(zr0, zr1 - ph + 2):
            for cc in range(zc0, zc1 - pw + 2):
                ok = True
                for dr, dc in np.argwhere(patch == int(color)):
                    ar = rr + int(dr)
                    ac = cc + int(dc)
                    nr0 = max(0, ar - 1)
                    nr1 = min(h, ar + 2)
                    nc0 = max(0, ac - 1)
                    nc1 = min(w, ac + 2)
                    if np.any(occupied[nr0:nr1, nc0:nc1]):
                        ok = False
                        break
                if ok:
                    candidates.append((rr, cc))
        if not candidates:
            return None
        rr, cc = random.choice(candidates)
        for dr, dc in np.argwhere(patch == int(color)):
            grid[rr + int(dr), cc + int(dc)] = int(color)
        occupied = grid != BG

    return grid


def _expected_output(
    inp: np.ndarray,
    valid_points: list[tuple[int, int, int]],
    shapes: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    scene = _parse_scene(inp)
    for template in scene["templates"].values():
        coords = template["coords_abs"]
        out[coords[:, 0], coords[:, 1]] = BG
    for row, col, color in valid_points:
        for dr, dc in shapes[color]:
            rr = int(row + dr)
            cc = int(col + dc)
            if out[rr, cc] == FILL:
                out[rr, cc] = int(color)
    return out


def _generate_one() -> np.ndarray | None:
    h = random.randint(20, 24)
    w = random.randint(20, 24)
    frame_h = random.randint(10, 13)
    frame_w = random.randint(10, 13)
    if h - frame_h < 10 or w - frame_w < 10:
        return None

    r0 = random.randint(5, h - frame_h - 5)
    c0 = random.randint(5, w - frame_w - 5)
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    used_colors = sorted(random.sample(LEAF_COLORS, random.randint(2, 3)))
    shapes = _choose_shapes(used_colors)
    row_leaves, col_leaves, valid_points, _covered = _sample_leaf_spec(r0, c0, r1, c1, shapes, used_colors)
    inp = _build_input_from_spec(h, w, (r0, c0, r1, c1), shapes, row_leaves, col_leaves)
    if inp is None:
        return None

    expected = _expected_output(inp, valid_points, shapes)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    scene = _parse_scene(inp)
    if not (4 <= len(scene["leaves"]) <= 6):
        return None
    total_pairs = len(scene["row_leaves"]) * len(scene["col_leaves"])
    if not (1 <= len(valid_points) < total_pairs):
        return None

    stamped_cells = set(map(tuple, np.argwhere(actual != inp)))
    if not stamped_cells:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
