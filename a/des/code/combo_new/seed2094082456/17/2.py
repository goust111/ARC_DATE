from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 12
GRID_MAX = 26
RECT_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
RECT_HEIGHT_RANGE = (3, 9)
RECT_WIDTH_RANGE = (3, 9)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
OVERLAP_PROB = 0.5
NON_OVERLAP_GAP = 1


class RectSpec(NamedTuple):
    color: int
    bbox: tuple[int, int, int, int]
    corner_cells: tuple[tuple[int, int], tuple[int, int]]
    directions: tuple[tuple[int, int], tuple[int, int]]
    marker_cells: tuple[tuple[int, int], ...]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _build_rect_spec(color: int, bbox: tuple[int, int, int, int], family: int) -> RectSpec:
    r0, c0, r1, c1 = bbox
    if family == 0:
        corner_a = (r0, c0)
        corner_b = (r1, c1)
        dir_a = (1, 1)
        dir_b = (-1, -1)
        marker_cells = (
            (r0, c0),
            (r0, c0 + 1),
            (r0 + 1, c0),
            (r1, c1),
            (r1, c1 - 1),
            (r1 - 1, c1),
        )
    else:
        corner_a = (r0, c1)
        corner_b = (r1, c0)
        dir_a = (1, -1)
        dir_b = (-1, 1)
        marker_cells = (
            (r0, c1),
            (r0, c1 - 1),
            (r0 + 1, c1),
            (r1, c0),
            (r1, c0 + 1),
            (r1 - 1, c0),
        )
    return RectSpec(
        color=color,
        bbox=bbox,
        corner_cells=(corner_a, corner_b),
        directions=(dir_a, dir_b),
        marker_cells=marker_cells,
    )


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = map(int, grid.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _parse_l_component(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(cells) != 3:
        raise AssertionError("each L marker must contain exactly three cells")
    pts = {tuple(map(int, cell)) for cell in cells}
    degrees: dict[tuple[int, int], int] = {}
    for cell in pts:
        degrees[cell] = sum((nbr in pts) for nbr in _neighbors4(*cell))

    corners = [cell for cell, deg in degrees.items() if deg == 2]
    arms = [cell for cell, deg in degrees.items() if deg == 1]
    if len(corners) != 1 or len(arms) != 2:
        raise AssertionError("component must be an L shape with one corner and two arm cells")

    corner = corners[0]
    offsets = [(arm[0] - corner[0], arm[1] - corner[1]) for arm in arms]
    vertical = [off for off in offsets if off[1] == 0 and abs(off[0]) == 1]
    horizontal = [off for off in offsets if off[0] == 0 and abs(off[1]) == 1]
    if len(vertical) != 1 or len(horizontal) != 1:
        raise AssertionError("L marker arms must extend one step horizontally and vertically")

    direction = (vertical[0][0], horizontal[0][1])
    return corner, direction


def _discover_rectangles(grid: np.ndarray) -> list[RectSpec]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) not in (BLACK, GREY)]
    specs: list[tuple[tuple[int, int], RectSpec]] = []

    for color in colors:
        comps = _components_of_color(g, color)
        if len(comps) != 2:
            raise AssertionError("each non-background color must form exactly two L markers")

        parsed = [_parse_l_component(comp) for comp in comps]
        (corner_a, dir_a), (corner_b, dir_b) = parsed
        if dir_a != (-dir_b[0], -dir_b[1]):
            raise AssertionError("paired L markers must point in opposite directions")

        dr = _sign(corner_b[0] - corner_a[0])
        dc = _sign(corner_b[1] - corner_a[1])
        if dr == 0 or dc == 0:
            raise AssertionError("paired corners must lie on opposite rectangle corners")

        valid_pair = (dir_a == (dr, dc) and dir_b == (-dr, -dc)) or (
            dir_b == (dr, dc) and dir_a == (-dr, -dc)
        )
        if not valid_pair:
            raise AssertionError("L orientations must point inward across the rectangle diagonal")

        r0 = min(corner_a[0], corner_b[0])
        c0 = min(corner_a[1], corner_b[1])
        r1 = max(corner_a[0], corner_b[0])
        c1 = max(corner_a[1], corner_b[1])
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise AssertionError("rectangle must be large enough for two non-overlapping L markers")

        marker_cells = tuple(sorted(tuple(map(int, cell)) for comp in comps for cell in comp))
        spec = RectSpec(
            color=color,
            bbox=(r0, c0, r1, c1),
            corner_cells=(corner_a, corner_b),
            directions=(dir_a, dir_b),
            marker_cells=marker_cells,
        )
        specs.append((min(marker_cells), spec))

    specs.sort(key=lambda item: item[0])
    return [spec for _first_pos, spec in specs]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair opposite L markers to fill their implied rectangles while keeping grey hole templates empty.

    Input structure:
    - The grid background is black.
    - Each non-black, non-grey color appears as exactly two separate 3-cell L markers.
    - The two same-color L markers point inward from opposite rectangle corners.
    - Grey pixels inside those implied rectangles act as visible hole templates.

    Rule:
    - For each color, identify its two L-marker corner cells and take the bounding rectangle whose diagonal endpoints are those corners.
    - Process colors in the order of their first appearance in row-major scan order.
    - Repaint every non-marker cell of that rectangle with the marker color, but keep any input grey template cell black instead of filling it.
    - Preserve all original L-marker pixels, and let later rectangles overwrite earlier fills where rectangles overlap.

    Key insight:
    - The paired L markers are not shapes to copy; they are diagonal corner indicators for a rectangle, while grey cells are carve-out masks that must stay empty.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D ARC grid")

    specs = _discover_rectangles(g)
    marker_mask = (g != BLACK) & (g != GREY)
    hole_mask = g == GREY

    out = g.copy()
    out[hole_mask] = BLACK

    for spec in specs:
        r0, c0, r1, c1 = spec.bbox
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region_markers = marker_mask[r0 : r1 + 1, c0 : c1 + 1]
        region_holes = hole_mask[r0 : r1 + 1, c0 : c1 + 1]

        fill_mask = ~region_markers & ~region_holes
        region[fill_mask] = spec.color
        region[region_holes] = BLACK

    return out


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bboxes_intersect_with_gap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _marker_cells_outside_bbox(rect: RectSpec, other: RectSpec) -> bool:
    return all(not _point_in_bbox(cell, other.bbox) for cell in rect.marker_cells)


def _rect_candidate_ok(
    rect: RectSpec, existing: list[RectSpec], allow_overlap_with: int | None = None
) -> bool:
    marker_set = set(rect.marker_cells)
    if len(marker_set) != len(rect.marker_cells):
        return False

    for idx, other in enumerate(existing):
        if marker_set & set(other.marker_cells):
            return False
        if not _marker_cells_outside_bbox(rect, other):
            return False
        if not _marker_cells_outside_bbox(other, rect):
            return False
        if idx == allow_overlap_with:
            continue
        if _bboxes_intersect_with_gap(rect.bbox, other.bbox, NON_OVERLAP_GAP):
            return False
    return True


def _sample_generic_rect(
    rng: random.Random, h: int, w: int, color: int, size_bias: str | None = None
) -> RectSpec | None:
    max_h = min(RECT_HEIGHT_RANGE[1], h)
    max_w = min(RECT_WIDTH_RANGE[1], w)
    if max_h < RECT_HEIGHT_RANGE[0] or max_w < RECT_WIDTH_RANGE[0]:
        return None

    if size_bias == "tall":
        rect_h = rng.randint(max(6, RECT_HEIGHT_RANGE[0]), max_h)
        rect_w = rng.randint(RECT_WIDTH_RANGE[0], min(5, max_w))
    elif size_bias == "wide":
        rect_h = rng.randint(RECT_HEIGHT_RANGE[0], min(5, max_h))
        rect_w = rng.randint(max(6, RECT_WIDTH_RANGE[0]), max_w)
    else:
        rect_h = rng.randint(RECT_HEIGHT_RANGE[0], max_h)
        rect_w = rng.randint(RECT_WIDTH_RANGE[0], max_w)

    if rect_h > h or rect_w > w:
        return None

    r0 = rng.randint(0, h - rect_h)
    c0 = rng.randint(0, w - rect_w)
    bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    family = rng.randint(0, 1)
    return _build_rect_spec(color, bbox, family)


def _sample_overlap_pair(
    rng: random.Random, h: int, w: int, color_a: int, color_b: int
) -> tuple[RectSpec, RectSpec] | None:
    if h < 7 or w < 7:
        return None

    for _ in range(PLACEMENT_RETRIES):
        max_a_h = min(10, h)
        max_b_w = min(10, w)
        if max_a_h < 7 or max_b_w < 7:
            return None

        a_h = rng.randint(7, max_a_h)
        b_h_max = min(5, a_h - 4, h)
        if b_h_max < 3:
            continue
        b_h = rng.randint(3, b_h_max)

        b_w = rng.randint(7, max_b_w)
        a_w_max = min(5, b_w - 4, w)
        if a_w_max < 3:
            continue
        a_w = rng.randint(3, a_w_max)

        top_margin_max = a_h - b_h - 2
        left_margin_max = b_w - a_w - 2
        if top_margin_max < 2 or left_margin_max < 2:
            continue

        r0_a = rng.randint(0, h - a_h)
        c0_b = rng.randint(0, w - b_w)
        top_margin = rng.randint(2, top_margin_max)
        left_margin = rng.randint(2, left_margin_max)

        r0_b = r0_a + top_margin
        c0_a = c0_b + left_margin

        rect_a = _build_rect_spec(
            color_a,
            (r0_a, c0_a, r0_a + a_h - 1, c0_a + a_w - 1),
            rng.randint(0, 1),
        )
        rect_b = _build_rect_spec(
            color_b,
            (r0_b, c0_b, r0_b + b_h - 1, c0_b + b_w - 1),
            rng.randint(0, 1),
        )

        if not _bboxes_intersect(rect_a.bbox, rect_b.bbox):
            continue
        if not _rect_candidate_ok(rect_b, [rect_a], allow_overlap_with=0):
            continue
        return rect_a, rect_b
    return None


def _strict_interior_cells(rect: RectSpec) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect.bbox
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            cells.append((r, c))
    return cells


def _sample_holes(
    rng: random.Random, rects: list[RectSpec], hole_mode: str
) -> set[tuple[int, int]]:
    hole_cells: set[tuple[int, int]] = set()
    candidate_lists = [_strict_interior_cells(rect) for rect in rects]

    enabled: list[bool] = [False] * len(rects)
    if hole_mode == "none":
        enabled = [False] * len(rects)
    elif hole_mode == "all":
        enabled = [bool(cells) for cells in candidate_lists]
    else:
        if len(rects) == 1:
            enabled = [bool(candidate_lists[0])]
        else:
            order = list(range(len(rects)))
            rng.shuffle(order)
            enabled = [False] * len(rects)
            enabled[order[0]] = bool(candidate_lists[order[0]])
            for idx in order[1:]:
                enabled[idx] = bool(candidate_lists[idx]) and rng.random() < 0.5
            if all(enabled):
                enabled[order[-1]] = False

    for rect, candidates, use_holes in zip(rects, candidate_lists, enabled):
        if not use_holes or not candidates:
            continue
        fillable = (rect.bbox[2] - rect.bbox[0] + 1) * (rect.bbox[3] - rect.bbox[1] + 1) - len(rect.marker_cells)
        max_holes = min(len(candidates), max(1, fillable - 1))
        hole_count = rng.randint(1, max_holes)
        chosen = rng.sample(candidates, hole_count)
        hole_cells.update(chosen)
    return hole_cells


def _render_input(h: int, w: int, rects: list[RectSpec], holes: set[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in holes:
        grid[r, c] = GREY
    for rect in rects:
        for r, c in rect.marker_cells:
            grid[r, c] = rect.color
    return grid


def _choose_hole_mode(rng: random.Random, rect_count: int) -> str:
    if rect_count == 1:
        return rng.choice(("none", "all"))
    return rng.choice(("none", "all", "mixed"))


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    rect_count = rng.choice(RECT_COUNT_CHOICES)
    if min(h, w) < 14:
        rect_count = min(rect_count, 3)

    want_overlap = rect_count >= 2 and rng.random() < OVERLAP_PROB
    if want_overlap and rect_count < 2:
        rect_count = 2

    colors = list(rng.sample(PALETTE, rect_count))
    rects: list[RectSpec] = []

    if want_overlap:
        pair = _sample_overlap_pair(rng, h, w, colors[0], colors[1])
        if pair is None:
            return None
        rects.extend(pair)

    start_idx = len(rects)
    for idx in range(start_idx, rect_count):
        color = colors[idx]
        for _ in range(PLACEMENT_RETRIES):
            bias = None
            if want_overlap and idx == 2:
                bias = rng.choice((None, "tall", "wide"))
            rect = _sample_generic_rect(rng, h, w, color, size_bias=bias)
            if rect is None:
                return None
            if _rect_candidate_ok(rect, rects):
                rects.append(rect)
                break
        else:
            return None

    hole_mode = _choose_hole_mode(rng, len(rects))
    holes = _sample_holes(rng, rects, hole_mode)
    grid = _render_input(h, w, rects, holes)

    discovered = _discover_rectangles(grid)
    if len(discovered) != len(rects):
        return None

    overlap_pairs = 0
    for i in range(len(discovered)):
        for j in range(i + 1, len(discovered)):
            if _bboxes_intersect(discovered[i].bbox, discovered[j].bbox):
                overlap_pairs += 1
    if want_overlap and overlap_pairs == 0:
        return None
    if not want_overlap and overlap_pairs != 0:
        return None

    if hole_mode == "none" and np.any(grid == GREY):
        return None
    if hole_mode == "all":
        for rect in rects:
            r0, c0, r1, c1 = rect.bbox
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.any(patch == GREY):
                return None
    if hole_mode == "mixed" and len(rects) > 1:
        grey_counts = []
        for rect in rects:
            r0, c0, r1, c1 = rect.bbox
            grey_counts.append(int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY)))
        if not any(count > 0 for count in grey_counts) or not any(count == 0 for count in grey_counts):
            return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
