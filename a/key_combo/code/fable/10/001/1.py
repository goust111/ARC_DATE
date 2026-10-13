from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
CORNER_KINDS = ("tl", "tr", "bl", "br")

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 28
SQUARE_COUNT_CHOICES = (2, 3)
SQUARE_COUNT_WEIGHTS = (3, 2)
SIDE_MIN = 4
GEN_RETRY_LIMIT = 500
PAIR_RETRY_LIMIT = 250
THIRD_RETRY_LIMIT = 250

MIN_OVERLAP_SPAN = 2
MIN_UNIQUE_AREA = 12
MAX_OVERLAP_FRACTION = 0.28
MIN_SEPARATION_FROM_THIRD = 1


def _square_corners(r0: int, c0: int, side: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c0 + side),
        "bl": (r0 + side, c0),
        "br": (r0 + side, c0 + side),
    }


def _square_from_markers(coords: list[tuple[int, int]]) -> dict[str, object]:
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    assert len(coords) == 3, "each color must have exactly three markers"
    assert len(rows) == 2 and len(cols) == 2, "three markers must occupy square corner rows and cols"

    side_r = rows[1] - rows[0]
    side_c = cols[1] - cols[0]
    assert side_r == side_c and side_r > 0, "marker spans must define a non-degenerate square"

    corners = _square_corners(rows[0], cols[0], side_r)
    present = set(coords)
    all_corners = set(corners.values())
    assert present <= all_corners, "markers must be literal square corners"

    missing = all_corners - present
    assert len(missing) == 1, "exactly one square corner must be missing"
    missing_pos = next(iter(missing))
    missing_kind = next(kind for kind, pos in corners.items() if pos == missing_pos)

    return {
        "r0": rows[0],
        "c0": cols[0],
        "r1": rows[1],
        "c1": cols[1],
        "side": side_r,
        "corners": corners,
        "missing": missing_pos,
        "missing_kind": missing_kind,
    }


def _render_squares(shape: tuple[int, int], specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(shape, dtype=int)
    owner = np.full(shape, -1, dtype=int)

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        side = int(spec["side"])
        color = int(spec["color"])
        block_counts = counts[r0 : r0 + side + 1, c0 : c0 + side + 1]
        block_owner = owner[r0 : r0 + side + 1, c0 : c0 + side + 1]
        block_owner[block_counts == 0] = color
        block_counts += 1

    out = np.full(shape, BLACK, dtype=int)
    out[counts == 1] = owner[counts == 1]
    out[counts == 2] = RED
    return out, counts


def _bbox_overlap(a: dict[str, object], b: dict[str, object]) -> tuple[int, int, int]:
    r0 = max(int(a["r0"]), int(b["r0"]))
    c0 = max(int(a["c0"]), int(b["c0"]))
    r1 = min(int(a["r1"]), int(b["r1"]))
    c1 = min(int(a["c1"]), int(b["c1"]))
    if r0 > r1 or c0 > c1:
        return 0, 0, 0
    return r1 - r0 + 1, c1 - c0 + 1, (r1 - r0 + 1) * (c1 - c0 + 1)


def _point_in_square(spec: dict[str, object], point: tuple[int, int]) -> bool:
    r, c = point
    return int(spec["r0"]) <= r <= int(spec["r1"]) and int(spec["c0"]) <= c <= int(spec["c1"])


def _expanded_boxes_overlap(a: dict[str, object], b: dict[str, object], pad: int) -> bool:
    return not (
        int(a["r1"]) + pad < int(b["r0"])
        or int(b["r1"]) + pad < int(a["r0"])
        or int(a["c1"]) + pad < int(b["c0"])
        or int(b["c1"]) + pad < int(a["c0"])
    )


def _all_visible_markers(specs: list[dict[str, object]]) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for spec in specs:
        corners = spec["corners"]
        missing_kind = spec["missing_kind"]
        assert isinstance(corners, dict)
        for kind, pos in corners.items():
            if kind != missing_kind:
                pts.append(pos)
    return pts


def _covered_corner_kinds(spec: dict[str, object], others: list[dict[str, object]]) -> list[str]:
    covered = []
    for kind, point in spec["corners"].items():
        if any(_point_in_square(other, point) for other in others):
            covered.append(kind)
    return covered


def _markers_are_isolated(specs: list[dict[str, object]]) -> bool:
    pts = _all_visible_markers(specs)
    if len(pts) != len(set(pts)):
        return False
    for (r1, c1), (r2, c2) in combinations(pts, 2):
        if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
            return False
    return True


def _visible_markers_are_clear(specs: list[dict[str, object]]) -> bool:
    for i, spec in enumerate(specs):
        missing_kind = spec["missing_kind"]
        for kind, point in spec["corners"].items():
            if kind == missing_kind:
                continue
            for j, other in enumerate(specs):
                if i == j:
                    continue
                if _point_in_square(other, point):
                    return False
    return True


def _pair_is_valid(a: dict[str, object], b: dict[str, object]) -> bool:
    overlap_h, overlap_w, overlap_area = _bbox_overlap(a, b)
    if overlap_area == 0:
        return False
    if overlap_h < MIN_OVERLAP_SPAN or overlap_w < MIN_OVERLAP_SPAN:
        return False

    area_a = (int(a["side"]) + 1) ** 2
    area_b = (int(b["side"]) + 1) ** 2
    if overlap_area > int(min(area_a, area_b) * MAX_OVERLAP_FRACTION):
        return False
    if area_a - overlap_area < MIN_UNIQUE_AREA or area_b - overlap_area < MIN_UNIQUE_AREA:
        return False
    return True


def _sample_square(shape: tuple[int, int], side: int, rng: random.Random) -> dict[str, object]:
    h, w = shape
    r0 = rng.randint(0, h - side - 1)
    c0 = rng.randint(0, w - side - 1)
    corners = _square_corners(r0, c0, side)
    return {
        "r0": r0,
        "c0": c0,
        "r1": r0 + side,
        "c1": c0 + side,
        "side": side,
        "corners": corners,
    }


def _choose_missing_kinds(
    placed: list[dict[str, object]], rng: random.Random
) -> list[str] | None:
    forced: list[str | None] = []
    for i, spec in enumerate(placed):
        covered = _covered_corner_kinds(spec, placed[:i] + placed[i + 1 :])
        if len(covered) > 1:
            return None
        forced.append(covered[0] if covered else None)

    n = len(placed)
    for _ in range(64):
        kinds = [
            forced[i] if forced[i] is not None else rng.choice(CORNER_KINDS)
            for i in range(n)
        ]
        if len(set(kinds)) >= min(2, n):
            return kinds
    return None


def _build_input(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        corners = spec["corners"]
        missing_kind = spec["missing_kind"]
        assert isinstance(corners, dict)
        for kind, (r, c) in corners.items():
            if kind != missing_kind:
                grid[r, c] = color
    return grid


def _place_overlap_pair(shape: tuple[int, int], sides: list[int], rng: random.Random) -> list[dict[str, object]] | None:
    side_a, side_b = sides[:2]
    for _ in range(PAIR_RETRY_LIMIT):
        a = _sample_square(shape, side_a, rng)
        b = _sample_square(shape, side_b, rng)
        if _pair_is_valid(a, b):
            return [a, b]
    return None


def _place_third_square(
    shape: tuple[int, int],
    side: int,
    placed: list[dict[str, object]],
    rng: random.Random,
) -> dict[str, object] | None:
    for _ in range(THIRD_RETRY_LIMIT):
        cand = _sample_square(shape, side, rng)
        if any(_bbox_overlap(cand, other)[2] != 0 for other in placed):
            continue
        if any(_expanded_boxes_overlap(cand, other, MIN_SEPARATION_FROM_THIRD) for other in placed):
            continue
        if any(_point_in_square(other, point) for other in placed for point in cand["corners"].values()):
            continue
        if any(_point_in_square(cand, point) for other in placed for point in other["corners"].values()):
            continue
        return cand
    return None


def _sample_layout(rng: random.Random) -> tuple[tuple[int, int], list[dict[str, object]]] | None:
    h = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    shape = (h, w)
    square_count = rng.choices(SQUARE_COUNT_CHOICES, weights=SQUARE_COUNT_WEIGHTS, k=1)[0]

    max_side = min(10, min(shape) - 7)
    if max_side < SIDE_MIN + square_count - 1:
        return None

    sides = rng.sample(list(range(SIDE_MIN, max_side + 1)), k=square_count)
    rng.shuffle(sides)

    placed = _place_overlap_pair(shape, sides, rng)
    if placed is None:
        return None
    if square_count == 3:
        third = _place_third_square(shape, sides[2], placed, rng)
        if third is None:
            return None
        placed.append(third)

    kinds = _choose_missing_kinds(placed, rng)
    if kinds is None:
        return None
    colors = rng.sample(PALETTE, k=square_count)
    specs: list[dict[str, object]] = []
    for base, color, missing_kind in zip(placed, colors, kinds):
        spec = dict(base)
        spec["color"] = color
        spec["missing_kind"] = missing_kind
        specs.append(spec)
    return shape, specs


def _validate_generated_sample(inp: np.ndarray, specs: list[dict[str, object]]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    colors = [int(color) for color in np.unique(inp) if int(color) != BLACK]
    if RED in colors or len(colors) != len(specs):
        return False
    if any(int(np.sum(inp == color)) != 3 for color in colors):
        return False

    if not _markers_are_isolated(specs):
        return False
    if not _visible_markers_are_clear(specs):
        return False

    pair_overlaps = 0
    for a, b in combinations(specs, 2):
        if _bbox_overlap(a, b)[2] > 0:
            pair_overlaps += 1
    if pair_overlaps != 1:
        return False

    reconstructed = []
    for color in colors:
        coords = [tuple(map(int, pos)) for pos in np.argwhere(inp == color)]
        try:
            sq = _square_from_markers(coords)
        except AssertionError:
            return False
        sq["color"] = color
        reconstructed.append(sq)

    sides = [int(spec["side"]) for spec in reconstructed]
    if len(sides) != len(set(sides)):
        return False

    expected, counts = _render_squares(inp.shape, reconstructed)
    if int(np.max(counts)) > 2:
        return False
    if int(np.sum(counts == 2)) == 0:
        return False

    for spec in reconstructed:
        unique_area = int(np.sum((counts == 1) & (expected == int(spec["color"]))))
        if unique_area < MIN_UNIQUE_AREA:
            return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each invisible square from its three corner markers and color double-covered cells red.

    Input structure:
    - The background is black.
    - Each non-black, non-red color appears exactly three times as isolated singleton pixels.
    - The three pixels of one color are three corners of one axis-aligned square, with the fourth corner missing.

    Rule:
    - Process each color independently and infer the missing fourth corner of its square.
    - Treat that completed corner set as the boundary of a filled square region, including the border.
    - Count how many reconstructed squares cover each cell.
    - Cells covered by exactly one square take that square's color, cells covered by exactly two squares become red, and uncovered cells stay black.

    Key insight:
    - The input never draws the squares themselves; only the three-corner generators are visible, so the answer comes from reconstructing hidden filled regions and then resolving their overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert colors, "expected at least one marker color"
    assert RED not in colors, "red is reserved for overlap output and must be absent from the input"

    specs: list[dict[str, object]] = []
    for color in colors:
        coords = [tuple(map(int, pos)) for pos in np.argwhere(g == color)]
        spec = _square_from_markers(coords)
        spec["color"] = color
        specs.append(spec)

    assert _markers_are_isolated(specs), "marker pixels must stay isolated singletons"
    assert _visible_markers_are_clear(specs), "visible markers may not lie inside another square"
    sides = [int(spec["side"]) for spec in specs]
    assert len(sides) == len(set(sides)), "different marker groups must correspond to different square sizes"

    out, counts = _render_squares(g.shape, specs)
    assert int(np.max(counts)) <= 2, "valid inputs may not create triple overlaps"
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        sampled = _sample_layout(rng)
        if sampled is None:
            continue
        shape, specs = sampled
        inp = _build_input(shape, specs)
        if _validate_generated_sample(inp, specs):
            return inp
    raise RuntimeError("failed to generate a valid invisible-square-overlap input")
