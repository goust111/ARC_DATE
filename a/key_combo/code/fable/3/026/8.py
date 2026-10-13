from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
PINK = 6
GENERATOR_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)

MIN_SIZE = 20
MAX_SIZE = 30
MAX_GENERATION_ATTEMPTS = 600
MAX_PLACE_ATTEMPTS = 250


Coord = tuple[int, int]


@dataclass(frozen=True)
class ShapeSpec:
    kind: str
    color: int
    points: tuple[Coord, ...]
    draw: frozenset[Coord]


def _line_cells(a: Coord, b: Coord) -> set[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        return {(ar, c) for c in range(c0, c1 + 1)}
    if ac == bc:
        r0, r1 = sorted((ar, br))
        return {(r, ac) for r in range(r0, r1 + 1)}
    raise AssertionError("line endpoints must share a row or a column")


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[Coord]:
    assert r0 < r1 and c0 < c1, "rectangle must have positive height and width"
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _draw_from_generator_points(points: list[Coord]) -> set[Coord] | None:
    n = len(points)
    assert 1 <= n <= 4, "each color must appear one to four times"
    if n == 1:
        return None

    rows = sorted({r for r, _c in points})
    cols = sorted({c for _r, c in points})

    if n == 2:
        a, b = points
        if a[0] == b[0] or a[1] == b[1]:
            return _line_cells(a, b)
        return _rect_border(min(rows), min(cols), max(rows), max(cols))

    assert len(rows) == 2 and len(cols) == 2, "three or four points must be rectangle corners"
    corner_set = {(r, c) for r in rows for c in cols}
    point_set = set(points)
    assert point_set <= corner_set, "points must sit on the same two rows and two columns"
    if n == 3:
        assert len(point_set) == 3, "three generator points must be distinct"
    else:
        assert point_set == corner_set, "four generator points must be the four rectangle corners"
    return _rect_border(rows[0], cols[0], rows[1], cols[1])


def main(grid: np.ndarray) -> np.ndarray:
    """Decode scattered color generators and reconstruct lines or hollow rectangles with pink intersections.

    Input structure:
    - The canvas is black and contains only isolated colored generator pixels.
    - Each non-black color appears one to four times.
    - Two points of one color may be collinear or opposite rectangle corners; three or four points are aligned rectangle corners.

    Rule:
    - A color appearing once is noise and is erased.
    - A color appearing twice draws either the inclusive straight segment through collinear endpoints or the hollow rectangle whose opposite corners are the two points.
    - A color appearing three times first completes the missing fourth corner, then draws the hollow rectangle; four points draw their corner rectangle directly.
    - Cells covered by two or more different reconstructed figures become pink.

    Key insight:
    The visible pixels are not the objects themselves; their count together with their row/column geometry jointly determines which missing figure to rebuild.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    layers: dict[Coord, set[int]] = {}
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BLACK):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        drawn = _draw_from_generator_points(coords)
        if drawn is None:
            continue
        for cell in drawn:
            layers.setdefault(cell, set()).add(color)

    out = np.zeros_like(g)
    for (r, c), colors in layers.items():
        out[r, c] = next(iter(colors)) if len(colors) == 1 else PINK
    return out


def _spec_line(kind: str, color: int, a: Coord, b: Coord) -> ShapeSpec:
    return ShapeSpec(kind, color, (a, b), frozenset(_line_cells(a, b)))


def _spec_rect_from_corners(kind: str, color: int, corners: list[Coord]) -> ShapeSpec:
    rows = sorted({r for r, _c in corners})
    cols = sorted({c for _r, c in corners})
    assert len(rows) == 2 and len(cols) == 2
    return ShapeSpec(kind, color, tuple(corners), frozenset(_rect_border(rows[0], cols[0], rows[1], cols[1])))


def _input_points(specs: list[ShapeSpec]) -> list[Coord]:
    return [p for spec in specs for p in spec.points]


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _points_are_isolated(points: list[Coord]) -> bool:
    seen: set[Coord] = set()
    for i, a in enumerate(points):
        if a in seen:
            return False
        seen.add(a)
        for b in points[:i]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                return False
    return True


def _can_add_spec(shape: tuple[int, int], specs: list[ShapeSpec], candidate: ShapeSpec, *, allow_overlap: int = 0) -> bool:
    if any(not _in_bounds(shape, p) for p in candidate.points):
        return False
    if any(not _in_bounds(shape, p) for p in candidate.draw):
        return False

    if not _points_are_isolated(_input_points(specs) + list(candidate.points)):
        return False

    existing_draw = set().union(*(set(spec.draw) for spec in specs if spec.kind != "noise")) if specs else set()
    overlap = len(set(candidate.draw) & existing_draw) if candidate.kind != "noise" else 0
    if overlap > allow_overlap:
        return False

    if candidate.kind == "noise" and any(p in existing_draw for p in candidate.points):
        return False
    return True


def _render_specs(shape: tuple[int, int], specs: list[ShapeSpec]) -> np.ndarray:
    layers: dict[Coord, set[int]] = {}
    for spec in specs:
        if spec.kind == "noise":
            continue
        for cell in spec.draw:
            layers.setdefault(cell, set()).add(spec.color)

    out = np.zeros(shape, dtype=int)
    for (r, c), colors in layers.items():
        out[r, c] = next(iter(colors)) if len(colors) == 1 else PINK
    return out


def _build_input(shape: tuple[int, int], specs: list[ShapeSpec]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in spec.points:
            assert grid[r, c] == BLACK, "generator points must not overlap"
            grid[r, c] = spec.color
    return grid


def _random_rect_candidate(kind: str, color: int, shape: tuple[int, int], rng: random.Random) -> ShapeSpec:
    H, W = shape
    height = rng.randint(5, min(9, H - 2))
    width = rng.randint(5, min(10, W - 2))
    r0 = rng.randint(1, H - height - 1)
    c0 = rng.randint(1, W - width - 1)
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]

    if kind == "diagonal_rect":
        if rng.random() < 0.5:
            points = [corners[0], corners[3]]
        else:
            points = [corners[1], corners[2]]
        return _spec_rect_from_corners(kind, color, points)

    if kind == "three_rect":
        missing = rng.randrange(4)
        points = [corner for i, corner in enumerate(corners) if i != missing]
        rng.shuffle(points)
        return _spec_rect_from_corners(kind, color, points)

    assert kind == "four_rect"
    rng.shuffle(corners)
    return _spec_rect_from_corners(kind, color, corners)


def _add_random_rect(shape: tuple[int, int], specs: list[ShapeSpec], kind: str, color: int, rng: random.Random) -> bool:
    for _ in range(MAX_PLACE_ATTEMPTS):
        candidate = _random_rect_candidate(kind, color, shape, rng)
        if _can_add_spec(shape, specs, candidate, allow_overlap=0):
            specs.append(candidate)
            return True
    return False


def _add_noise(shape: tuple[int, int], specs: list[ShapeSpec], color: int, rng: random.Random) -> bool:
    H, W = shape
    for _ in range(MAX_PLACE_ATTEMPTS):
        point = (rng.randrange(H), rng.randrange(W))
        candidate = ShapeSpec("noise", color, (point,), frozenset())
        if _can_add_spec(shape, specs, candidate, allow_overlap=0):
            specs.append(candidate)
            return True
    return False


def _forced_cross_specs(shape: tuple[int, int], colors: list[int], rng: random.Random) -> list[ShapeSpec] | None:
    H, W = shape
    cross_r = rng.randint(5, H - 6)
    cross_c = rng.randint(5, W - 6)

    c0 = rng.randint(max(1, cross_c - 9), cross_c - 3)
    c1 = rng.randint(cross_c + 3, min(W - 2, cross_c + 9))
    r0 = rng.randint(max(1, cross_r - 9), cross_r - 3)
    r1 = rng.randint(cross_r + 3, min(H - 2, cross_r + 9))

    h_line = _spec_line("horizontal_line", colors[0], (cross_r, c0), (cross_r, c1))
    v_line = _spec_line("vertical_line", colors[1], (r0, cross_c), (r1, cross_c))
    specs = [h_line]
    if _can_add_spec(shape, specs, v_line, allow_overlap=1):
        specs.append(v_line)
        return specs
    return None


def _classify_input(grid: np.ndarray) -> dict[str, int]:
    counts = {
        "noise": 0,
        "horizontal_line": 0,
        "vertical_line": 0,
        "diagonal_rect": 0,
        "three_rect": 0,
        "four_rect": 0,
    }
    for color in [int(c) for c in np.unique(grid) if int(c) != BLACK]:
        points = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if len(points) == 1:
            counts["noise"] += 1
        elif len(points) == 2:
            if points[0][0] == points[1][0]:
                counts["horizontal_line"] += 1
            elif points[0][1] == points[1][1]:
                counts["vertical_line"] += 1
            else:
                counts["diagonal_rect"] += 1
        elif len(points) == 3:
            rows = {r for r, _c in points}
            cols = {c for _r, c in points}
            if len(rows) == 2 and len(cols) == 2:
                counts["three_rect"] += 1
        elif len(points) == 4:
            rows = {r for r, _c in points}
            cols = {c for _r, c in points}
            if len(rows) == 2 and len(cols) == 2 and set(points) == {(r, c) for r in rows for c in cols}:
                counts["four_rect"] += 1
    return counts


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[ShapeSpec]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if expected.shape != inp.shape:
        return False
    if not _points_are_isolated(_input_points(specs)):
        return False

    color_counts = {int(c): int(np.sum(inp == c)) for c in np.unique(inp) if int(c) != BLACK}
    if not color_counts or any(count < 1 or count > 4 for count in color_counts.values()):
        return False
    if PINK in color_counts:
        return False

    branch_counts = _classify_input(inp)
    required = ("noise", "horizontal_line", "vertical_line", "diagonal_rect", "three_rect", "four_rect")
    if any(branch_counts[name] < 1 for name in required):
        return False

    pink_count = int(np.sum(expected == PINK))
    if not (1 <= pink_count <= 6):
        return False

    for spec in specs:
        if spec.kind == "noise":
            if expected[spec.points[0]] != BLACK:
                return False
            continue

        own_cells = set(spec.draw)
        other_cells = set().union(*(set(other.draw) for other in specs if other is not spec and other.kind != "noise"))
        unique_cells = own_cells - other_cells
        if len(unique_cells) < 3:
            return False
        if spec.kind == "three_rect":
            rows = sorted({r for r, _c in spec.points})
            cols = sorted({c for _r, c in spec.points})
            missing = ({(r, c) for r in rows for c in cols} - set(spec.points)).pop()
            if missing not in unique_cells or int(expected[missing]) != spec.color:
                return False

        if spec.kind in {"horizontal_line", "vertical_line"}:
            interior = own_cells - set(spec.points)
            if not interior or all(int(expected[p]) == PINK for p in interior):
                return False
        elif spec.kind == "diagonal_rect":
            if len(own_cells - set(spec.points)) < 6:
                return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(MIN_SIZE, MAX_SIZE)
    W = rng.randint(MIN_SIZE, MAX_SIZE)
    shape = (H, W)

    color_count = rng.choice((6, 7, 8))
    colors = rng.sample(list(GENERATOR_COLORS), color_count)
    specs = _forced_cross_specs(shape, colors, rng)
    if specs is None:
        return None

    for kind, color in zip(("diagonal_rect", "three_rect", "four_rect"), colors[2:5]):
        if not _add_random_rect(shape, specs, kind, color, rng):
            return None

    for color in colors[5:]:
        if not _add_noise(shape, specs, color, rng):
            return None

    inp = _build_input(shape, specs)
    expected = _render_specs(shape, specs)
    if not _semantic_checks(inp, expected, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid generator-classification reconstruction input")
