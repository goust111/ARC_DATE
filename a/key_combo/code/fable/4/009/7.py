from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Rect:
    """A monochromatic solid rectangle with an inclusive (row, col) bbox."""

    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r1, self.c1


@dataclass(frozen=True)
class _Pair:
    """Two ordered facing rectangles: left/right or top/bottom."""

    first: _Rect
    second: _Rect
    orientation: str


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _solid_rectangles(grid: np.ndarray, bg: int) -> list[_Rect]:
    """Find 4-connected monochromatic components and require solid bboxes."""
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    rectangles: list[_Rect] = []

    for start_r in range(height):
        for start_c in range(width):
            if seen[start_r, start_c] or int(grid[start_r, start_c]) == bg:
                continue
            color = int(grid[start_r, start_c])
            queue = deque([(start_r, start_c)])
            seen[start_r, start_c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= rr < height
                        and 0 <= cc < width
                        and not seen[rr, cc]
                        and int(grid[rr, cc]) == color
                    ):
                        seen[rr, cc] = True
                        queue.append((rr, cc))

            rows, cols = zip(*coords)
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == expected_area, "every foreground component must be a solid rectangle"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            rectangles.append(_Rect(color, r0, c0, r1, c1))

    assert rectangles, "expected at least one facing rectangle pair"
    return rectangles


def _candidate_pair(a: _Rect, b: _Rect, grid: np.ndarray, bg: int) -> _Pair | None:
    if a.color == b.color:
        return None

    # Horizontal facing uses the shared row band in the original grid.
    if a.r0 == b.r0 and a.r1 == b.r1:
        left, right = (a, b) if a.c0 < b.c0 else (b, a)
        if left.c1 + 1 < right.c0:
            corridor = grid[left.r0 : left.r1 + 1, left.c1 + 1 : right.c0]
            if corridor.size and np.all(corridor == bg):
                return _Pair(left, right, "horizontal")

    # Vertical facing uses the shared column band in the original grid.
    if a.c0 == b.c0 and a.c1 == b.c1:
        top, bottom = (a, b) if a.r0 < b.r0 else (b, a)
        if top.r1 + 1 < bottom.r0:
            corridor = grid[top.r1 + 1 : bottom.r0, top.c0 : top.c1 + 1]
            if corridor.size and np.all(corridor == bg):
                return _Pair(top, bottom, "vertical")
    return None


def _facing_pairs(grid: np.ndarray, bg: int) -> list[_Pair]:
    rectangles = _solid_rectangles(grid, bg)
    assert len(rectangles) % 2 == 0, "rectangles must form disjoint pairs"

    candidates: list[tuple[int, int, _Pair]] = []
    degrees = [0] * len(rectangles)
    for i, first in enumerate(rectangles):
        for j in range(i + 1, len(rectangles)):
            pair = _candidate_pair(first, rectangles[j], grid, bg)
            if pair is not None:
                candidates.append((i, j, pair))
                degrees[i] += 1
                degrees[j] += 1

    assert all(degree == 1 for degree in degrees), "every rectangle must have one unique facing partner"
    assert len(candidates) * 2 == len(rectangles)
    return [pair for _i, _j, pair in candidates]


def _grow_pair(out: np.ndarray, pair: _Pair) -> None:
    first, second = pair.first, pair.second
    total_area = first.area + second.area

    if pair.orientation == "horizontal":
        gap = second.c0 - first.c1 - 1
        numerator = gap * first.area
        assert numerator % total_area == 0, "gap must divide exactly in the area ratio"
        first_growth = numerator // total_area
        second_growth = gap - first_growth
        assert first_growth > 0 and second_growth > 0
        out[first.r0 : first.r1 + 1, first.c1 + 1 : first.c1 + 1 + first_growth] = first.color
        out[second.r0 : second.r1 + 1, second.c0 - second_growth : second.c0] = second.color
    else:
        gap = second.r0 - first.r1 - 1
        numerator = gap * first.area
        assert numerator % total_area == 0, "gap must divide exactly in the area ratio"
        first_growth = numerator // total_area
        second_growth = gap - first_growth
        assert first_growth > 0 and second_growth > 0
        out[first.r1 + 1 : first.r1 + 1 + first_growth, first.c0 : first.c1 + 1] = first.color
        out[second.r0 - second_growth : second.r0, second.c0 : second.c1 + 1] = second.color


def main(grid: np.ndarray) -> np.ndarray:
    """Fill every facing-rectangle corridor by growing both colors in their area ratio.

    Input structure:
    - A dominant background contains several disjoint pairs of differently colored
      solid rectangles.  A pair has exactly the same facing row band or column
      band, with a nonempty all-background corridor between its two members.
    - Every rectangle has one unique partner; different pair corridors do not overlap.

    Rule:
    - For a corridor of length g between rectangles of areas a and b, grow the
      first rectangle g*a/(a+b) cells toward its partner and the second rectangle
      g*b/(a+b) cells toward the first.  The divisions are guaranteed integral,
      so the two original colors fill the entire corridor and meet once.
    - Preserve the grid shape, all original rectangles, and all other background.

    Key insight:
    - The meeting point is fixed by the two full rectangle areas, not by the
      corridor midpoint or by either rectangle consuming the whole gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    bg = _background(g)
    pairs = _facing_pairs(g, bg)

    out = g.copy()
    for pair in pairs:
        _grow_pair(out, pair)
    return out


# ------------------------------ generator ------------------------------

_GENERATION_ATTEMPTS = 500


def _slot_boxes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Four separated placement regions, returned as inclusive bboxes."""
    row_cut = height // 2
    col_cut = width // 2
    return [
        (1, 1, row_cut - 2, col_cut - 2),
        (1, col_cut + 1, row_cut - 2, width - 2),
        (row_cut + 1, 1, height - 2, col_cut - 2),
        (row_cut + 1, col_cut + 1, height - 2, width - 2),
    ]


def _paint_rect(grid: np.ndarray, rect: _Rect) -> None:
    grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color


def _make_pair_in_slot(
    rng: np.random.Generator,
    slot: tuple[int, int, int, int],
    orientation: str,
    unequal: bool,
    colors: tuple[int, int],
) -> tuple[_Pair, int] | None:
    sr0, sc0, sr1, sc1 = slot
    slot_h, slot_w = sr1 - sr0 + 1, sc1 - sc0 + 1

    if unequal:
        # 1:3 makes the meeting point visibly different from the midpoint.
        depths = [1, 3]
        if bool(rng.integers(0, 2)):
            depths.reverse()
    else:
        depth = int(rng.integers(1, 3))
        depths = [depth, depth]

    longitudinal = slot_w if orientation == "horizontal" else slot_h
    transverse = slot_h if orientation == "horizontal" else slot_w
    valid_multipliers = [
        k for k in (1, 2) if (k + 1) * sum(depths) <= longitudinal
    ]
    if not valid_multipliers or transverse < 2:
        return None
    multiplier = int(rng.choice(valid_multipliers))
    gap = multiplier * sum(depths)
    total_length = sum(depths) + gap
    thickness = int(rng.integers(2, min(4, transverse) + 1))

    if orientation == "horizontal":
        r0 = int(rng.integers(sr0, sr1 - thickness + 2))
        c0 = int(rng.integers(sc0, sc1 - total_length + 2))
        first = _Rect(colors[0], r0, c0, r0 + thickness - 1, c0 + depths[0] - 1)
        second_c0 = first.c1 + gap + 1
        second = _Rect(
            colors[1], r0, second_c0, r0 + thickness - 1, second_c0 + depths[1] - 1
        )
    else:
        r0 = int(rng.integers(sr0, sr1 - total_length + 2))
        c0 = int(rng.integers(sc0, sc1 - thickness + 2))
        first = _Rect(colors[0], r0, c0, r0 + depths[0] - 1, c0 + thickness - 1)
        second_r0 = first.r1 + gap + 1
        second = _Rect(
            colors[1], second_r0, c0, second_r0 + depths[1] - 1, c0 + thickness - 1
        )
    return _Pair(first, second, orientation), gap


def _expected_output(grid: np.ndarray, specs: list[tuple[_Pair, int]]) -> np.ndarray:
    """Independent construction from generator-owned pair specifications."""
    out = grid.copy()
    for pair, gap in specs:
        first, second = pair.first, pair.second
        numerator = gap * first.area
        denominator = first.area + second.area
        if numerator % denominator:
            raise AssertionError("generator produced a nonintegral ratio")
        first_growth = numerator // denominator
        if pair.orientation == "horizontal":
            out[first.r0 : first.r1 + 1, first.c1 + 1 : first.c1 + 1 + first_growth] = first.color
            out[first.r0 : first.r1 + 1, first.c1 + 1 + first_growth : second.c0] = second.color
        else:
            out[first.r1 + 1 : first.r1 + 1 + first_growth, first.c0 : first.c1 + 1] = first.color
            out[first.r1 + 1 + first_growth : second.r0, first.c0 : first.c1 + 1] = second.color
    return out


def _semantic_validation(
    grid: np.ndarray, expected: np.ndarray, specs: list[tuple[_Pair, int]], bg: int
) -> bool:
    """Check rule dependencies, uniqueness, nondegeneracy, and counterexamples."""
    try:
        parsed = _facing_pairs(grid, bg)
        actual = main(grid)
    except (AssertionError, ValueError):
        return False

    if len(parsed) != len(specs) or not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if {pair.orientation for pair, _gap in specs} != {"horizontal", "vertical"}:
        return False

    equal_seen = False
    unequal_seen = False
    changed = expected != grid
    claimed = np.zeros(grid.shape, dtype=bool)
    generated_boxes = {
        frozenset((pair.first.box, pair.second.box)) for pair, _gap in specs
    }
    parsed_boxes = {frozenset((pair.first.box, pair.second.box)) for pair in parsed}
    if generated_boxes != parsed_boxes:
        return False

    for pair, gap in specs:
        first, second = pair.first, pair.second
        first_growth = gap * first.area // (first.area + second.area)
        if first.area == second.area:
            equal_seen = True
        else:
            unequal_seen = True
            # This pair is an explicit counterexample to midpoint filling.
            if first_growth == gap // 2 or 2 * first_growth == gap:
                return False
        if first_growth <= 0 or first_growth >= gap:
            return False

        corridor = np.zeros(grid.shape, dtype=bool)
        if pair.orientation == "horizontal":
            corridor[first.r0 : first.r1 + 1, first.c1 + 1 : second.c0] = True
        else:
            corridor[first.r1 + 1 : second.r0, first.c0 : first.c1 + 1] = True
        if np.any(claimed & corridor) or not np.all(grid[corridor] == bg):
            return False
        if not np.all(changed[corridor]) or np.any(expected[corridor] == bg):
            return False
        claimed |= corridor

    # No original cell or unrelated background cell may change.
    if np.any(changed & ~claimed):
        return False
    return equal_seen and unequal_seen


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(22, 31))
    width = int(rng.integers(22, 31))
    pair_count = int(rng.integers(2, 5))
    bg = int(rng.integers(0, 10))
    palette = [color for color in range(10) if color != bg]
    colors = [int(c) for c in rng.choice(palette, size=2 * pair_count, replace=False)]

    slots = _slot_boxes(height, width)
    chosen_slots = [slots[int(i)] for i in rng.permutation(4)[:pair_count]]

    orientations = ["horizontal", "vertical"]
    orientations.extend(
        "horizontal" if bool(rng.integers(0, 2)) else "vertical"
        for _ in range(pair_count - 2)
    )
    rng.shuffle(orientations)

    unequal_flags = [False, True]
    unequal_flags.extend(bool(rng.integers(0, 2)) for _ in range(pair_count - 2))
    rng.shuffle(unequal_flags)

    grid = np.full((height, width), bg, dtype=int)
    specs: list[tuple[_Pair, int]] = []
    for index, (slot, orientation, unequal) in enumerate(
        zip(chosen_slots, orientations, unequal_flags)
    ):
        result = _make_pair_in_slot(
            rng,
            slot,
            orientation,
            unequal,
            (colors[2 * index], colors[2 * index + 1]),
        )
        if result is None:
            return None
        pair, _gap = result
        _paint_rect(grid, pair.first)
        _paint_rect(grid, pair.second)
        specs.append(result)

    # The background must be structurally visible rather than inferred from an id.
    if _background(grid) != bg:
        return None
    expected = _expected_output(grid, specs)
    if not _semantic_validation(grid, expected, specs, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, semantically checked ARC input grid."""
    rng = np.random.default_rng()
    for _ in range(_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-ratio corridor puzzle")
