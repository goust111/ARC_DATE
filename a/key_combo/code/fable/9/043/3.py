from __future__ import annotations

from collections import Counter
import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
OBJECT_COUNT_CHOICES = (2, 3)
RADIUS_RANGE = (1, 5)
MAX_ATTEMPTS = 400

SIDE_TOP = "top"
SIDE_BOTTOM = "bottom"
SIDE_LEFT = "left"
SIDE_RIGHT = "right"


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != int(bg))


def _diamond_outline(center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = map(int, center)
    radius = int(radius)
    if radius == 0:
        return [(cr, cc)]
    coords: list[tuple[int, int]] = []
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        if dc == 0:
            coords.append((cr + dr, cc))
        else:
            coords.append((cr + dr, cc - dc))
            coords.append((cr + dr, cc + dc))
    return coords


def _bbox_for_spec(spec: dict) -> tuple[int, int, int, int]:
    cr, cc = spec["center"]
    radius = int(spec["radius"])
    return cr - radius, cc - radius, cr + radius, cc + radius


def _expanded_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], margin: int = 1) -> bool:
    r0, c0, r1, c1 = map(int, box)
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + margin < pr0 or pr1 + margin < r0 or c1 + margin < pc0 or pc1 + margin < c0):
            return False
    return True


def _marker_positions(shape: tuple[int, int], spec: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = map(int, shape)
    cr, cc = spec["center"]
    if spec["h_side"] == SIDE_TOP:
        horizontal = (0, cc)
    else:
        horizontal = (h - 1, cc)
    if spec["v_side"] == SIDE_LEFT:
        vertical = (cr, 0)
    else:
        vertical = (cr, w - 1)
    return horizontal, vertical


def _build_output(shape: tuple[int, int], specs: list[dict], bg: int = BG) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        horizontal, vertical = _marker_positions(shape, spec)
        out[horizontal] = color
        out[vertical] = color
        for r, c in _diamond_outline(spec["center"], spec["radius"]):
            out[int(r), int(c)] = color
    return out


def _all_outline_cells(specs: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for spec in specs:
        cells.update(_diamond_outline(spec["center"], spec["radius"]))
    return cells


def _priority_anchor(shape: tuple[int, int], spec: dict) -> tuple[int, int]:
    h, w = map(int, shape)
    anchor_r = h - 2 if spec["h_side"] == SIDE_TOP else 1
    anchor_c = w - 2 if spec["v_side"] == SIDE_LEFT else 1
    return anchor_r, anchor_c


def _dot_candidates(
    shape: tuple[int, int],
    spec: dict,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cr, cc = spec["center"]
    row_sign = 1 if spec["h_side"] == SIDE_TOP else -1
    col_sign = 1 if spec["v_side"] == SIDE_LEFT else -1
    anchor_r, anchor_c = _priority_anchor(shape, spec)

    cells: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            pos = (r, c)
            if pos in forbidden:
                continue
            if r == cr or c == cc:
                continue
            cells.append(pos)

    def score(pos: tuple[int, int]) -> tuple[int, int, int, int, int]:
        r, c = pos
        quadrant_penalty = 0
        if (r - cr) * row_sign <= 0:
            quadrant_penalty += 1
        if (c - cc) * col_sign <= 0:
            quadrant_penalty += 1
        return (
            quadrant_penalty,
            abs(r - anchor_r) + abs(c - anchor_c),
            -abs(r - cr) - abs(c - cc),
            r,
            c,
        )

    return sorted(cells, key=score)


def _select_dot_cells(
    shape: tuple[int, int],
    spec: dict,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    radius = int(spec["radius"])
    chosen: list[tuple[int, int]] = []
    candidates = _dot_candidates(shape, spec, forbidden)
    same_color_fixed = set(_marker_positions(shape, spec))

    for pos in candidates:
        if pos in forbidden:
            continue
        if any(max(abs(pos[0] - r), abs(pos[1] - c)) <= 1 for r, c in same_color_fixed):
            continue
        if any(max(abs(pos[0] - r), abs(pos[1] - c)) <= 1 for r, c in chosen):
            continue
        chosen.append(pos)
        if len(chosen) == radius:
            return chosen
    return None


def _input_from_specs(shape: tuple[int, int], specs: list[dict], bg: int = BG) -> np.ndarray | None:
    grid = np.full(shape, int(bg), dtype=int)
    forbidden = _all_outline_cells(specs)
    for spec in specs:
        horizontal, vertical = _marker_positions(shape, spec)
        grid[horizontal] = spec["color"]
        grid[vertical] = spec["color"]
        forbidden.add(horizontal)
        forbidden.add(vertical)

    occupied = set(forbidden)
    for spec in specs:
        dots = _select_dot_cells(shape, spec, occupied)
        if dots is None:
            return None
        for pos in dots:
            grid[pos] = spec["color"]
            occupied.add(pos)
    return grid


def _parse_specs_from_input(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _get_bg(g)
    specs: list[dict] = []
    for color in _colors_present(g, bg):
        coords = [tuple(map(int, pos)) for pos in np.argwhere(g == color)]
        edge_coords = [pos for pos in coords if pos[0] in (0, h - 1) or pos[1] in (0, w - 1)]
        interior = [pos for pos in coords if pos not in edge_coords]
        horizontal = [pos for pos in edge_coords if pos[0] in (0, h - 1)]
        vertical = [pos for pos in edge_coords if pos[1] in (0, w - 1)]
        assert len(edge_coords) == 2, "each color must have exactly two edge markers"
        assert len(horizontal) == 1 and len(vertical) == 1, "expected one horizontal-edge marker and one vertical-edge marker"
        assert horizontal[0] != vertical[0], "markers must not share a corner"
        h_side = SIDE_TOP if horizontal[0][0] == 0 else SIDE_BOTTOM
        v_side = SIDE_LEFT if vertical[0][1] == 0 else SIDE_RIGHT
        specs.append(
            {
                "color": color,
                "center": (vertical[0][0], horizontal[0][1]),
                "radius": len(interior),
                "h_side": h_side,
                "v_side": v_side,
                "dots": interior,
            }
        )
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use same-color edge crosshairs plus interior dot counts to draw hollow diamonds.

    Input structure:
    - The background is black, and the foreground contains two or three colors.
    - For each foreground color, exactly one marker lies on the top or bottom edge and exactly one marker lies on the left or right edge.
    - The remaining pixels of that color are isolated interior dots whose positions are irrelevant except for their count.

    Rule:
    - For each color, take the row of its left/right marker and the column of its top/bottom marker; their intersection is the diamond center.
    - Count that color's interior dots to obtain radius r, erase all of those dots, and keep both edge markers unchanged.
    - Paint every cell at Manhattan distance exactly r from the center in that color, forming a hollow diamond.

    Key insight:
    - The edge markers define only the center, while the interior dots define only the radius; both clues are necessary and must be combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    out = np.full(g.shape, int(bg), dtype=int)

    for spec in _parse_specs_from_input(g):
        color = int(spec["color"])
        horizontal, vertical = _marker_positions(g.shape, spec)
        out[horizontal] = color
        out[vertical] = color
        for r, c in _diamond_outline(spec["center"], spec["radius"]):
            out[int(r), int(c)] = color

    return out


def _sample_specs(shape: tuple[int, int], rng: np.random.Generator) -> list[dict] | None:
    h, w = map(int, shape)
    n_objects = int(rng.choice(OBJECT_COUNT_CHOICES))
    colors = [int(color) for color in rng.choice(COLOR_POOL, size=n_objects, replace=False)]
    raw_specs: list[dict] = []
    for color in colors:
        raw_specs.append(
            {
                "color": color,
                "radius": int(rng.integers(RADIUS_RANGE[0], RADIUS_RANGE[1] + 1)),
                "h_side": SIDE_TOP if rng.random() < 0.5 else SIDE_BOTTOM,
                "v_side": SIDE_LEFT if rng.random() < 0.5 else SIDE_RIGHT,
            }
        )

    specs: list[dict] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for draft in sorted(raw_specs, key=lambda spec: int(spec["radius"]), reverse=True):
        radius = int(draft["radius"])
        row_lo = radius + 1
        row_hi = h - radius - 2
        col_lo = radius + 1
        col_hi = w - radius - 2
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [
            (r, c)
            for r in range(row_lo, row_hi + 1)
            for c in range(col_lo, col_hi + 1)
        ]
        rng.shuffle(candidates)

        placed = False
        for center in candidates:
            spec = dict(draft)
            spec["center"] = center
            box = _bbox_for_spec(spec)
            if not _expanded_clear(box, placed_boxes, margin=1):
                continue
            specs.append(spec)
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return sorted(specs, key=lambda spec: int(spec["color"]))


def _validate_specs(shape: tuple[int, int], specs: list[dict]) -> None:
    assert 2 <= len(specs) <= 3, "expected two or three colored objects"
    h, w = map(int, shape)
    boxes = [_bbox_for_spec(spec) for spec in specs]
    for spec, box in zip(specs, boxes):
        radius = int(spec["radius"])
        cr, cc = spec["center"]
        assert radius >= 1, "radius must stay positive"
        assert radius < min(cr, h - 1 - cr, cc, w - 1 - cc), "diamond must stay strictly inside the grid"
        horizontal, vertical = _marker_positions(shape, spec)
        assert horizontal[0] in (0, h - 1)
        assert vertical[1] in (0, w - 1)
        assert horizontal != vertical

    for i, box in enumerate(boxes):
        assert _expanded_clear(box, boxes[:i] + boxes[i + 1 :], margin=1), "diamond boxes must stay separated"

    slack_values = {
        min(spec["center"][0], h - 1 - spec["center"][0], spec["center"][1], w - 1 - spec["center"][1]) - int(spec["radius"])
        for spec in specs
    }
    assert len(slack_values) > 1, "radius must not collapse to one fixed edge-distance formula"


def _validate_input(grid: np.ndarray, expected: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "grid must stay within ARC bounds"
    assert _get_bg(g) == BG, "generator uses black background"

    specs = _parse_specs_from_input(g)
    _validate_specs(g.shape, specs)
    outline_cells = _all_outline_cells(specs)

    for spec in specs:
        dots = [tuple(map(int, pos)) for pos in spec["dots"]]
        assert len(dots) == int(spec["radius"])
        for dot in dots:
            assert dot not in outline_cells, "dot must not lie on any diamond outline"
        for i, a in enumerate(dots):
            for b in dots[i + 1 :]:
                assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1, "same-color dots must remain singleton pixels"

    actual = main(g)
    assert np.array_equal(actual, expected), "solver output must match constructed output"
    assert not np.array_equal(g, expected), "input must differ from output"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    w = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))
    specs = _sample_specs((h, w), rng)
    if specs is None:
        return None

    output = _build_output((h, w), specs, bg=BG)
    grid = _input_from_specs((h, w), specs, bg=BG)
    if grid is None:
        return None

    try:
        _validate_input(grid, output)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the crosshair diamond task")
