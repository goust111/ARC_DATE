"""ARC task: reconstruct and z-order solid rectangles from corner residues."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 600


@dataclass(frozen=True)
class _Rectangle:
    color: int
    bbox: tuple[int, int, int, int]
    missing: tuple[int, int] | None


def _corners(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _strictly_contains(box: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    r, c = point
    return r0 < r < r1 and c0 < c < c1


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    return max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)


def _infer_rectangles(grid: np.ndarray) -> list[_Rectangle]:
    rectangles: list[_Rectangle] = []
    for color in (int(x) for x in np.unique(grid) if int(x) != BACKGROUND):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] in (3, 4), "each color must leave three or four corners"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        assert r0 < r1 and c0 < c1, "corners must determine a non-degenerate rectangle"
        visible = set(map(tuple, coords.tolist()))
        corners = _corners(box)
        assert visible <= corners, "every residue must be a bbox corner"
        missing_set = corners - visible
        assert len(missing_set) == 4 - len(visible)
        missing = next(iter(missing_set)) if missing_set else None
        rectangles.append(_Rectangle(color, box, missing))
    assert len(rectangles) >= 2, "expected several colored rectangles"
    return rectangles


def _occlusion_edges(rectangles: list[_Rectangle]) -> set[tuple[int, int]]:
    """Return (lower, upper) edges and validate every visible overlap."""
    edges: set[tuple[int, int]] = set()
    for lower, rect in enumerate(rectangles):
        if rect.missing is None:
            continue
        covers = [
            upper
            for upper, other in enumerate(rectangles)
            if upper != lower and _strictly_contains(other.bbox, rect.missing)
        ]
        assert covers, "a missing corner must be explained by an upper rectangle"
        edges.update((lower, upper) for upper in covers)

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            if _intersection_area(rectangles[i].bbox, rectangles[j].bbox) == 0:
                continue
            directions = int((i, j) in edges) + int((j, i) in edges)
            assert directions == 1, "each overlapping pair needs one unambiguous upper rectangle"
    return edges


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct solid rectangles from their corner residues and render them in inferred occlusion order.

    Input structure:
    - A black background contains several colors, each occurring only at three or four
      corners of one axis-aligned rectangle; no rectangle is already drawn solid.
    - A three-corner color is missing exactly the corner covered by another rectangle.

    Rule:
    - Recover each rectangle from the row and column extremes of its visible corners.
    - If rectangle A's missing corner lies strictly inside rectangle B, place B above A.
    - Fill all rectangles from lower to upper, allowing upper colors to overwrite overlaps.

    Key insight:
    - The absent fourth corner is not empty noise: it is the evidence that uniquely
      orients every overlapping pair in the z-order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors"

    rectangles = _infer_rectangles(g)
    edges = _occlusion_edges(rectangles)
    remaining = set(range(len(rectangles)))
    out = np.zeros_like(g)

    while remaining:
        level = {
            i for i in remaining
            if not any(upper == i and lower in remaining for lower, upper in edges)
        }
        assert level, "occlusion relations must be acyclic"
        # Rectangles in the same level must be disjoint, so their iteration order is immaterial.
        for i in level:
            for j in level:
                if i < j:
                    assert _intersection_area(rectangles[i].bbox, rectangles[j].bbox) == 0
        for i in level:
            r0, c0, r1, c1 = rectangles[i].bbox
            out[r0 : r1 + 1, c0 : c1 + 1] = rectangles[i].color
        remaining -= level
    return out


def _random_box(height: int, width: int) -> tuple[int, int, int, int]:
    h = random.randint(4, 8)
    w = random.randint(4, 8)
    r0 = random.randint(1, height - h - 1)
    c0 = random.randint(1, width - w - 1)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _next_upper(
    lower: tuple[int, int, int, int],
    earlier: list[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    """Find a modest overlap that contains exactly one corner of `lower`."""
    choices = list(_corners(lower))
    random.shuffle(choices)
    for missing in choices:
        for _ in range(350):
            h = random.randint(4, 8)
            w = random.randint(4, 8)
            r_lo = max(0, missing[0] - h + 2)
            r_hi = min(missing[0] - 1, height - h)
            c_lo = max(0, missing[1] - w + 2)
            c_hi = min(missing[1] - 1, width - w)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            upper = (r0, c0, r0 + h - 1, c0 + w - 1)
            lower_in_upper = sum(_strictly_contains(upper, p) for p in _corners(lower))
            upper_in_lower = sum(_strictly_contains(lower, p) for p in _corners(upper))
            overlap = _intersection_area(lower, upper)
            union = (lower[2] - lower[0] + 1) * (lower[3] - lower[1] + 1)
            union += h * w - overlap
            if lower_in_upper != 1 or upper_in_lower != 0:
                continue
            if overlap < 2 or overlap * 4 > union:
                continue
            if any(_intersection_area(upper, box) for box in earlier):
                continue
            return upper, missing
    return None


def _isolated(points: list[tuple[int, int]]) -> bool:
    return all(
        max(abs(r1 - r2), abs(c1 - c2)) > 1
        for i, (r1, c1) in enumerate(points)
        for r2, c2 in points[i + 1 :]
    )


def _render_expected(
    shape: tuple[int, int], rectangles: list[_Rectangle], order: list[int]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for i in order:
        r0, c0, r1, c1 = rectangles[i].bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = rectangles[i].color
    return out


def _semantic_validate(
    inp: np.ndarray,
    rectangles: list[_Rectangle],
    chain_length: int,
) -> bool:
    """Check intended geometry, uniqueness, and necessity independently of main()."""
    try:
        parsed = _infer_rectangles(inp)
    except AssertionError:
        return False
    by_color = {rect.color: rect for rect in parsed}
    if by_color != {rect.color: rect for rect in rectangles}:
        return False

    intended_edges = {(i, i + 1) for i in range(chain_length - 1)}
    actual_edges: set[tuple[int, int]] = set()
    for i, a in enumerate(rectangles):
        for j, b in enumerate(rectangles):
            if i != j and a.missing is not None and _strictly_contains(b.bbox, a.missing):
                actual_edges.add((i, j))
    if actual_edges != intended_edges:
        return False

    expected = _render_expected(inp.shape, rectangles, list(range(len(rectangles))))
    # Every inferred edge is necessary: reversing it changes at least one output cell.
    for lower, upper in intended_edges:
        altered_order = list(range(len(rectangles)))
        altered_order[lower], altered_order[upper] = altered_order[upper], altered_order[lower]
        if np.array_equal(_render_expected(inp.shape, rectangles, altered_order), expected):
            return False
    try:
        return np.array_equal(main(inp), expected)
    except AssertionError:
        return False


def _generate_attempt(target_count: int) -> np.ndarray | None:
    min_side = 23 if target_count == 4 else 17
    height = random.randint(min_side, 28)
    width = random.randint(min_side, 28)
    assert target_count in (2, 3, 4)
    # Three-object scenes sometimes provide a full three-layer stack. Four-object
    # scenes pair one forced overlap with two independent four-corner rectangles;
    # the latter explicitly exercises the no-occlusion branch without a fragile
    # geometric walk dominating generation time.
    if target_count == 2:
        chain_length, standalone_count = 2, 0
    elif target_count == 3:
        standalone_count = 1 if random.random() < 0.4 else 0
        chain_length = target_count - standalone_count
    else:
        chain_length, standalone_count = 2, 2

    boxes = [_random_box(height, width)]
    missing: list[tuple[int, int] | None] = []
    for _ in range(chain_length - 1):
        found = _next_upper(boxes[-1], boxes[:-1], height, width)
        if found is None:
            return None
        upper, covered_corner = found
        boxes.append(upper)
        missing.append(covered_corner)
    missing.append(None)

    for _standalone_index in range(standalone_count):
        occupied_corners = [
            point
            for box, miss in zip(boxes, missing)
            for point in _corners(box) - ({miss} if miss else set())
        ]
        for _ in range(1000):
            box = _random_box(height, width)
            new_corners = list(_corners(box))
            if (
                all(_intersection_area(box, other) == 0 for other in boxes)
                and _isolated(occupied_corners + new_corners)
            ):
                boxes.append(box)
                missing.append(None)
                break
        else:
            return None

    colors = random.sample(PALETTE, len(boxes))
    rectangles = [_Rectangle(color, box, miss) for color, box, miss in zip(colors, boxes, missing)]
    residues: list[tuple[int, int, int]] = []
    for rect in rectangles:
        for r, c in _corners(rect.bbox) - ({rect.missing} if rect.missing else set()):
            residues.append((r, c, rect.color))
    if not _isolated([(r, c) for r, c, _ in residues]):
        return None

    # For longer chains, reject area-monotone layers so area cannot substitute for residues.
    if chain_length >= 3:
        areas = [
            (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
            for box in boxes[:chain_length]
        ]
        if areas == sorted(areas) or areas == sorted(areas, reverse=True):
            return None

    inp = np.zeros((height, width), dtype=int)
    for r, c, color in residues:
        inp[r, c] = color
    return inp if _semantic_validate(inp, rectangles, chain_length) else None


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate corner-residue occlusion puzzles."""
    # Choose complexity once, before retries, to avoid rejection sampling bias
    # toward the easiest (two-object) branch.
    target_count = random.choice((2, 3, 4))
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(target_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-occlusion puzzle")
