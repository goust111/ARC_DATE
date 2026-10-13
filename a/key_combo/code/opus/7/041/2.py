from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 1
RED = 2
GREEN = 3
YELLOW = 4
OVERLAP_COLOR = 5
ORIGIN_COLOR = 7
MARKER_PALETTE = (GREEN, YELLOW, RED, ORIGIN_COLOR)

ROTATION_BY_CORNER = {
    "tl": 0,
    "tr": 90,
    "br": 180,
    "bl": 270,
}
ROT90_K = {
    0: 0,
    90: 3,
    180: 2,
    270: 1,
}

TEMPLATE_DIMS = ((3, 4), (3, 5), (4, 5))
TEMPLATE_AREA_RANGE = (5, 8)
GRID_SIDE_RANGE = (18, 28)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotated_mask(mask: np.ndarray, angle: int) -> np.ndarray:
    return np.rot90(mask, k=ROT90_K[angle])


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _layout_from_input(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int], dict[int, int], list[tuple[int, int, int, bool]]]:
    blue_components = _components(grid == TEMPLATE_COLOR)
    assert blue_components, "expected a blue template"
    template_coords = max(blue_components, key=len)
    assert len(template_coords) >= 5, "template must be the largest blue component"

    r0, c0, r1, c1 = _bbox(template_coords)
    template_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == TEMPLATE_COLOR
    h, w = template_mask.shape

    corner_positions = {
        "tl": (r0 - 1, c0 - 1),
        "tr": (r0 - 1, c1 + 1),
        "br": (r1 + 1, c1 + 1),
        "bl": (r1 + 1, c0 - 1),
    }
    rotation_lookup: dict[int, int] = {}
    for name, (rr, cc) in corner_positions.items():
        assert 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1], "markers must be in bounds"
        color = int(grid[rr, cc])
        assert color in MARKER_PALETTE, "invalid marker color"
        rotation_lookup[color] = ROTATION_BY_CORNER[name]
    assert len(rotation_lookup) == 4, "marker colors must be distinct"

    excluded = set(template_coords) | set(corner_positions.values())
    remaining = np.zeros_like(grid, dtype=bool)
    for rr, cc in np.argwhere(grid != BG):
        cell = (int(rr), int(cc))
        if cell not in excluded:
            remaining[cell] = True

    singles = _components(remaining)
    assert singles, "expected singleton layout markers"
    points: list[tuple[int, int, int]] = []
    for comp in singles:
        assert len(comp) == 1, "layout points must be isolated singletons"
        rr, cc = comp[0]
        points.append((rr, cc, int(grid[rr, cc])))

    origin_points = [(rr, cc) for rr, cc, color in points if color == TEMPLATE_COLOR]
    assert len(origin_points) == 1, "expected exactly one blue origin singleton"
    origin_r, origin_c = origin_points[0]

    layout: list[tuple[int, int, int, bool]] = [(0, 0, 0, True)]
    for rr, cc, color in points:
        if color == TEMPLATE_COLOR:
            continue
        dr = rr - origin_r
        dc = cc - origin_c
        assert dr % h == 0 and dc % w == 0, "layout must follow the template lattice"
        grid_dr = dr // h
        grid_dc = dc // w
        layout.append((int(grid_dr), int(grid_dc), rotation_lookup[color], False))

    return template_mask, (r0, c0), (r0, c0, r1, c1), rotation_lookup, layout


def _render_output(
    shape: tuple[int, int],
    template_mask: np.ndarray,
    template_top: tuple[int, int],
    layout: list[tuple[int, int, int, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = template_mask.shape
    r0, c0 = template_top
    coverage = np.zeros(shape, dtype=int)
    origin_coverage = np.zeros(shape, dtype=bool)

    for dr, dc, angle, is_origin in layout:
        stamp = _rotated_mask(template_mask, angle)
        sr = r0 + dr * h
        sc = c0 + dc * w
        for rr, cc in _mask_coords(stamp):
            gr = sr + rr
            gc = sc + cc
            assert 0 <= gr < shape[0] and 0 <= gc < shape[1], "stamp placement out of bounds"
            coverage[gr, gc] += 1
            if is_origin:
                origin_coverage[gr, gc] = True

    out = np.zeros(shape, dtype=int)
    out[coverage > 0] = TEMPLATE_COLOR
    out[(coverage == 1) & origin_coverage] = ORIGIN_COLOR
    out[coverage >= 2] = OVERLAP_COLOR
    return out, coverage, origin_coverage


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp rotated copies of a blue template according to diagonal marker colors and a singleton lattice.

    Input structure:
    - The background is black, with one largest blue 4-connected template component.
    - Four colored singleton markers sit diagonally just outside the template bbox corners.
    - Elsewhere, isolated singleton pixels form a lattice: one blue origin pixel and several non-blue guide pixels.

    Rule:
    - Read the four diagonal marker colors as a rotation legend: top-left -> 0 degrees, top-right -> 90 degrees clockwise,
      bottom-right -> 180 degrees, bottom-left -> 270 degrees clockwise.
    - Convert each non-blue singleton's offset from the blue origin into lattice units using the template bbox height and width.
    - Place a rotated copy of the template at each lattice offset; place the origin copy unrotated and recolor it orange.
    - Remove the original template, markers, and singleton pixels; any output cell covered by two or more copies becomes gray.

    Key insight:
    - The diagonal corner markers are not paint instructions for nearby cells; they are a local legend that decodes which
      rotation each distant singleton requests before the template is stamped across the lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    template_mask, template_top, _template_bbox, _lookup, layout = _layout_from_input(g)
    out, _coverage, _origin = _render_output(g.shape, template_mask, template_top, layout)
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    for nb in _neighbors4(*start):
        if 0 <= nb[0] < h and 0 <= nb[1] < w:
            frontier.add(nb)

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for nb in _neighbors4(*cell):
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                frontier.add(nb)

    if len(cells) != area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _valid_template(mask: np.ndarray) -> bool:
    if mask.sum() < TEMPLATE_AREA_RANGE[0] or mask.sum() > TEMPLATE_AREA_RANGE[1]:
        return False
    if mask.all():
        return False
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    rots = [_rotated_mask(mask, angle) for angle in (0, 90, 180, 270)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rots[i], rots[j]):
                return False
    return True


def _generate_template() -> np.ndarray:
    for _ in range(200):
        h, w = random.choice(TEMPLATE_DIMS)
        max_area = min(TEMPLATE_AREA_RANGE[1], h * w - 1)
        area = random.randint(TEMPLATE_AREA_RANGE[0], max_area)
        mask = _random_connected_mask(h, w, area)
        if mask is not None and _valid_template(mask):
            return mask
    raise RuntimeError("failed to generate template")


def _base_patterns() -> list[dict[str, tuple[int, int]]]:
    return [
        {
            "origin": (0, 0),
            "driver": (-1, 0),
            "zero": (0, 2),
            "half": (1, 2),
            "right_angle": (0, -2),
        },
        {
            "origin": (0, 0),
            "driver": (1, 0),
            "zero": (-1, 2),
            "half": (2, 0),
            "right_angle": (0, -2),
        },
        {
            "origin": (0, 0),
            "driver": (-1, 0),
            "zero": (1, 3),
            "half": (0, 2),
            "right_angle": (2, -1),
        },
    ]


def _choose_layout_points(rotation_colors: dict[int, int]) -> list[tuple[int, int, int]]:
    pattern = random.choice(_base_patterns()).copy()
    if random.random() < 0.5:
        for key, (dr, dc) in list(pattern.items()):
            pattern[key] = (dr, -dc)

    driver_angle = random.choice((90, 270))
    other_right_angle = 270 if driver_angle == 90 else 90
    angle_to_color = {angle: color for color, angle in rotation_colors.items()}

    points = [
        (0, 0, TEMPLATE_COLOR),
        (*pattern["driver"], angle_to_color[driver_angle]),
        (*pattern["zero"], angle_to_color[0]),
        (*pattern["half"], angle_to_color[180]),
        (*pattern["right_angle"], angle_to_color[other_right_angle]),
    ]

    if random.random() < 0.45:
        extras = [(2, 2), (-2, -2), (1, -3), (-2, 3), (3, -1)]
        if random.random() < 0.5:
            extras = [(dr, -dc) for dr, dc in extras]
        used = {(dr, dc) for dr, dc, _ in points}
        options = [pos for pos in extras if pos not in used]
        if options:
            extra_pos = random.choice(options)
            extra_angle = random.choice((0, 90, 180, 270))
            points.append((*extra_pos, angle_to_color[extra_angle]))

    return points


def _interval_disjoint(a0: int, a1: int, b0: int, b1: int, gap: int) -> bool:
    return a1 + gap < b0 or b1 + gap < a0


def _placement_bounds(
    grid_h: int,
    grid_w: int,
    template_mask: np.ndarray,
    layout: list[tuple[int, int, int, bool]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = template_mask.shape
    row_mins: list[int] = []
    row_maxs: list[int] = []
    col_mins: list[int] = []
    col_maxs: list[int] = []
    for dr, dc, angle, _ in layout:
        stamp = _rotated_mask(template_mask, angle)
        sh, sw = stamp.shape
        row_mins.append(dr * h)
        row_maxs.append(dr * h + sh - 1)
        col_mins.append(dc * w)
        col_maxs.append(dc * w + sw - 1)

    min_row = min(row_mins)
    max_row = max(row_maxs)
    min_col = min(col_mins)
    max_col = max(col_maxs)

    r_lo = max(1, -min_row)
    c_lo = max(1, -min_col)
    r_hi = min(grid_h - h - 1, grid_h - 1 - max_row)
    c_hi = min(grid_w - w - 1, grid_w - 1 - max_col)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (r_lo, r_hi), (c_lo, c_hi)


def _singleton_origin_bounds(
    grid_h: int,
    grid_w: int,
    template_mask: np.ndarray,
    layout_points: list[tuple[int, int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    h, w = template_mask.shape
    rows = [dr * h for dr, _dc, _color in layout_points]
    cols = [dc * w for _dr, dc, _color in layout_points]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)
    return (0 - min_r, grid_h - 1 - max_r), (0 - min_c, grid_w - 1 - max_c), (min_r, min_c, max_r, max_c)


def _attempt_grid() -> np.ndarray | None:
    template_mask = _generate_template()
    marker_colors = random.sample(MARKER_PALETTE, 4)
    rotation_colors = {
        marker_colors[0]: 0,
        marker_colors[1]: 90,
        marker_colors[2]: 180,
        marker_colors[3]: 270,
    }
    angle_to_marker = {angle: color for color, angle in rotation_colors.items()}

    layout_points = _choose_layout_points(rotation_colors)
    layout = [(0, 0, 0, True)]
    for dr, dc, color in layout_points:
        if color == TEMPLATE_COLOR:
            continue
        layout.append((dr, dc, rotation_colors[color], False))

    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    placement = _placement_bounds(grid_h, grid_w, template_mask, layout)
    if placement is None:
        return None
    (r_lo, r_hi), (c_lo, c_hi) = placement
    template_top = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))

    out, coverage, _origin = _render_output((grid_h, grid_w), template_mask, template_top, layout)
    if int(coverage.max()) != 2 or np.count_nonzero(coverage == 2) == 0:
        return None

    (orr_lo, orr_hi), (orc_lo, orc_hi), (min_sr, min_sc, max_sr, max_sc) = _singleton_origin_bounds(
        grid_h, grid_w, template_mask, layout_points
    )
    if orr_lo > orr_hi or orc_lo > orc_hi:
        return None

    r0, c0 = template_top
    h, w = template_mask.shape
    template_zone = (r0 - 1, c0 - 1, r0 + h, c0 + w)
    singleton_positions = None
    for _ in range(100):
        orr = random.randint(orr_lo, orr_hi)
        orc = random.randint(orc_lo, orc_hi)
        cluster_box = (orr + min_sr, orc + min_sc, orr + max_sr, orc + max_sc)
        if not (
            _interval_disjoint(cluster_box[0], cluster_box[2], template_zone[0], template_zone[2], 1)
            or _interval_disjoint(cluster_box[1], cluster_box[3], template_zone[1], template_zone[3], 1)
        ):
            continue
        positions = []
        occupied = set()
        ok = True
        for dr, dc, color in layout_points:
            rr = orr + dr * h
            cc = orc + dc * w
            if (rr, cc) in occupied or not (0 <= rr < grid_h and 0 <= cc < grid_w):
                ok = False
                break
            occupied.add((rr, cc))
            positions.append((rr, cc, color))
        if ok:
            singleton_positions = positions
            break
    if singleton_positions is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    for rr, cc in _mask_coords(template_mask):
        inp[r0 + rr, c0 + cc] = TEMPLATE_COLOR

    corners = {
        (r0 - 1, c0 - 1): angle_to_marker[0],
        (r0 - 1, c0 + w): angle_to_marker[90],
        (r0 + h, c0 + w): angle_to_marker[180],
        (r0 + h, c0 - 1): angle_to_marker[270],
    }
    for (rr, cc), color in corners.items():
        if inp[rr, cc] != BG:
            return None
        inp[rr, cc] = color

    for rr, cc, color in singleton_positions:
        if inp[rr, cc] != BG:
            return None
        inp[rr, cc] = color

    # Keep the input visually clean: all non-template blue cells must be isolated.
    blue_components = _components(inp == TEMPLATE_COLOR)
    if len(blue_components) != 2:
        return None
    if sorted(len(comp) for comp in blue_components) != [1, int(template_mask.sum())]:
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
