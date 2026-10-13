from __future__ import annotations

import random

import numpy as np


BG = 7
TASK_COLORS = (8, 3, 5)

GRID_SIDE_RANGE = (20, 24)
DIAMOND_RADIUS_RANGE = (2, 6)
RECT_SIDE_RANGE = (2, 7)

OUTER_MARGIN = 1
BOX_GAP = 1
PLACEMENT_TRIES = 300
MAX_GENERATION_ATTEMPTS = 600
SHAPE_TRIES_PER_SPEC = 12

RELATION_PATTERNS = (
    ("fill", "border", "equal"),
    ("fill", "border", "border"),
    ("fill", "fill", "border"),
    ("fill", "equal", "border"),
)
SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for nr, nc in _neighbors8(cr, cc2, mask.shape):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _diamond_outline_points(cr: int, cc: int, radius: int) -> list[tuple[int, int]]:
    if radius == 0:
        return [(cr, cc)]
    pts: list[tuple[int, int]] = []
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        pts.append((cr + dr, cc + dc))
        if dc != 0:
            pts.append((cr + dr, cc - dc))
    return pts


def _diamond_radius(points: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox_from_points(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height == width and height % 2 == 1, "diamond outline must have an odd square bbox"
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    radius = (height - 1) // 2
    expected = set(_diamond_outline_points(cr, cc, radius))
    assert expected == set(points), "foreground component must be a full Manhattan-diamond outline"
    return int(radius)


def _contracted_pixel_count(radius: int) -> int:
    shrunk_radius = max(int(radius) - 1, 0)
    return 1 if shrunk_radius == 0 else 4 * shrunk_radius


def _paint_rect_border(out: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each color's markers with a filled or outlined rectangle according to the same-color diamond's one-step contraction size.

    Input structure:
    - The grid has an orange background and several non-overlapping foreground colors.
    - For each foreground color, there is one one-cell-thick Manhattan-diamond outline and two isolated marker pixels of that same color.
    - The two markers lie on opposite corners of an axis-aligned rectangle, and the marker-defined rectangles do not overlap each other or any diamond.

    Rule:
    - For each color, shrink its diamond outline by one layer while keeping the center fixed, and count the pixels that the shrunken outline would contain.
    - Use the two same-color markers as opposite corners of an inclusive rectangle and compute that rectangle's perimeter as `2 * (height + width)` in grid cells.
    - If the shrunken-diamond count is larger than the rectangle perimeter, fill the whole rectangle with that color; otherwise draw only the rectangle's one-pixel border. Remove the diamonds and markers themselves from the output.

    Key insight:
    - The diamonds are not output objects: each one only encodes a number, and that number is compared against the perimeter of the same-color marker rectangle to choose between solid fill and outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _majority_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    out = np.full(g.shape, bg, dtype=int)
    occupied = np.zeros(g.shape, dtype=bool)

    for color in colors:
        comps = _components8(g == color)
        outline_comps = [comp for comp in comps if len(comp) > 1]
        marker_comps = [comp for comp in comps if len(comp) == 1]
        assert len(outline_comps) == 1 and len(marker_comps) == 2, "each color must have one diamond and two marker pixels"

        radius = _diamond_radius(outline_comps[0])
        shrunken_count = _contracted_pixel_count(radius)

        markers = [comp[0] for comp in marker_comps]
        r0, c0, r1, c1 = _bbox_from_points(markers)
        assert r0 != r1 and c0 != c1, "marker pixels must be diagonally separated"
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        perimeter = 2 * (height + width)

        rect_mask = np.zeros(g.shape, dtype=bool)
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        assert not np.any(occupied & rect_mask), "marker rectangles must not overlap"

        if shrunken_count > perimeter:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            _paint_rect_border(out, (r0, c0, r1, c1), color)
        occupied |= rect_mask

    return out


def _relation_candidates() -> dict[str, list[tuple[int, int, int]]]:
    out: dict[str, list[tuple[int, int, int]]] = {"fill": [], "border": [], "equal": []}
    for radius in range(DIAMOND_RADIUS_RANGE[0], DIAMOND_RADIUS_RANGE[1] + 1):
        count = _contracted_pixel_count(radius)
        for height in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
            for width in range(RECT_SIDE_RANGE[0], RECT_SIDE_RANGE[1] + 1):
                if height == 2 and width == 2:
                    continue
                perimeter = 2 * (height + width)
                if count > perimeter:
                    out["fill"].append((radius, height, width))
                elif count == perimeter:
                    out["equal"].append((radius, height, width))
                else:
                    out["border"].append((radius, height, width))
    return out


CANDIDATES = _relation_candidates()


def _sample_specs() -> list[dict[str, int]] | None:
    for _ in range(200):
        relations = list(random.choice(RELATION_PATTERNS))
        random.shuffle(relations)
        colors = list(TASK_COLORS)
        random.shuffle(colors)

        specs: list[dict[str, int]] = []
        for color, relation in zip(colors, relations):
            radius, height, width = random.choice(CANDIDATES[relation])
            specs.append(
                {
                    "color": int(color),
                    "relation": relation,
                    "radius": int(radius),
                    "height": int(height),
                    "width": int(width),
                    "perimeter": 2 * (int(height) + int(width)),
                    "count": _contracted_pixel_count(int(radius)),
                }
            )

        fills = [spec for spec in specs if spec["relation"] == "fill"]
        nonfills = [spec for spec in specs if spec["relation"] != "fill"]
        assert fills and nonfills

        if not any(fill["radius"] < other["radius"] for fill in fills for other in nonfills):
            continue
        if not any(fill["perimeter"] >= other["perimeter"] for fill in fills for other in nonfills):
            continue
        if len({(spec["radius"], spec["height"], spec["width"]) for spec in specs}) < 3:
            continue
        return specs
    return None


def _boxes_far_enough(
    candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int
) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        row_separated = r1 + gap < pr0 or pr1 + gap < r0
        col_separated = c1 + gap < pc0 or pc1 + gap < c0
        if not (row_separated or col_separated):
            return False
    return True


def _candidate_boxes(
    height: int, width: int, shape: tuple[int, int], placed: list[tuple[int, int, int, int]]
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    r0_min = OUTER_MARGIN
    c0_min = OUTER_MARGIN
    r0_max = h - OUTER_MARGIN - height
    c0_max = w - OUTER_MARGIN - width
    if r0_max < r0_min or c0_max < c0_min:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for r0 in range(r0_min, r0_max + 1):
        for c0 in range(c0_min, c0_max + 1):
            box = (r0, c0, r0 + height - 1, c0 + width - 1)
            if _boxes_far_enough(box, placed, BOX_GAP):
                boxes.append(box)
    random.shuffle(boxes)
    return boxes[:PLACEMENT_TRIES]


def _place_specs(shape: tuple[int, int], specs: list[dict[str, int]]) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    jobs: list[tuple[int, str, int, int]] = []
    for spec in specs:
        diamond_side = 2 * spec["radius"] + 1
        jobs.append((spec["color"], "diamond_box", diamond_side, diamond_side))
        jobs.append((spec["color"], "rect_box", spec["height"], spec["width"]))

    jobs.sort(key=lambda item: (max(item[2], item[3]), item[2] * item[3]), reverse=True)

    by_color = {spec["color"]: spec for spec in specs}

    def backtrack(job_idx: int) -> bool:
        if job_idx == len(jobs):
            return True

        color, key, height, width = jobs[job_idx]
        for box in _candidate_boxes(height, width, shape, placed):
            by_color[color][key] = box
            placed.append(box)
            if backtrack(job_idx + 1):
                return True
            placed.pop()
            by_color[color].pop(key, None)
        return False

    return backtrack(0)


def _render_input(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)

    for spec in specs:
        r0, c0, r1, c1 = spec["diamond_box"]
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        for r, c in _diamond_outline_points(cr, cc, spec["radius"]):
            grid[r, c] = spec["color"]

    for spec in specs:
        r0, c0, r1, c1 = spec["rect_box"]
        if random.random() < 0.5:
            markers = [(r0, c0), (r1, c1)]
        else:
            markers = [(r0, c1), (r1, c0)]
        for r, c in markers:
            assert grid[r, c] == BG
            grid[r, c] = spec["color"]

    return grid


def _render_output(shape: tuple[int, int], specs: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["rect_box"]
        if spec["count"] > spec["perimeter"]:
            out[r0 : r1 + 1, c0 : c1 + 1] = spec["color"]
        else:
            _paint_rect_border(out, (r0, c0, r1, c1), spec["color"])
    return out


def _validate_instance(grid: np.ndarray, expected_output: np.ndarray) -> None:
    h, w = grid.shape
    assert GRID_SIDE_RANGE[0] <= h <= GRID_SIDE_RANGE[1]
    assert GRID_SIDE_RANGE[0] <= w <= GRID_SIDE_RANGE[1]

    for color in TASK_COLORS:
        comps = _components8(grid == color)
        outline_comps = [comp for comp in comps if len(comp) > 1]
        marker_comps = [comp for comp in comps if len(comp) == 1]
        assert len(outline_comps) == 1 and len(marker_comps) == 2

        markers = [comp[0] for comp in marker_comps]
        (r0, c0), (r1, c1) = markers
        assert r0 != r1 and c0 != c1
        assert abs(r0 - r1) > 1 or abs(c0 - c1) > 1

        for mr, mc in markers:
            for nr, nc in _neighbors8(mr, mc, grid.shape):
                assert grid[nr, nc] == BG

    actual_output = main(grid)
    assert np.array_equal(actual_output, expected_output)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        specs = _sample_specs()
        if specs is None:
            continue

        for _ in range(SHAPE_TRIES_PER_SPEC):
            height = random.randint(*GRID_SIDE_RANGE)
            width = random.randint(*GRID_SIDE_RANGE)
            placed_specs = [dict(spec) for spec in specs]
            if not _place_specs((height, width), placed_specs):
                continue

            output = _render_output((height, width), placed_specs)
            grid = _render_input((height, width), placed_specs)
            try:
                _validate_instance(grid, output)
            except AssertionError:
                continue

            signature = (tuple(map(int, output.shape)), output.tobytes())
            if signature in SEEN_OUTPUT_SIGNATURES:
                continue
            SEEN_OUTPUT_SIGNATURES.add(signature)
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
