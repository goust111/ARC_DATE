from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
MARKER = 8
BODY_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 500

BBox = tuple[int, int, int, int]
Line = tuple[str, int]  # ("h", row) or ("v", column)


def _rectangles(grid: np.ndarray) -> list[tuple[BBox, int]]:
    """Return solid, monochromatic, non-marker foreground rectangles."""
    mask = (grid != BACKGROUND) & (grid != MARKER)
    seen = np.zeros(grid.shape, dtype=bool)
    rectangles: list[tuple[BBox, int]] = []
    height, width = grid.shape

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        color = int(grid[sr, sc])
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and mask[nr, nc]
                    and not seen[nr, nc]
                ):
                    assert int(grid[nr, nc]) == color, "each rectangle must be monochromatic"
                    seen[nr, nc] = True
                    q.append((nr, nc))

        rows, cols = zip(*cells)
        box = (min(rows), min(cols), max(rows), max(cols))
        r0, c0, r1, c1 = box
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert len(cells) == region.size, "foreground objects must be solid rectangles"
        assert np.all(region == color), "each rectangle must have one visible color"
        rectangles.append((box, color))

    assert rectangles, "expected at least one solid rectangle"
    return rectangles


def _marker_lines(grid: np.ndarray) -> list[Line]:
    height, width = grid.shape
    coords = np.argwhere(grid == MARKER)
    assert 1 <= len(coords) <= 2, "expected one or two teal edge markers"
    lines: list[Line] = []
    for r, c in coords:
        r, c = int(r), int(c)
        on_horizontal_edge = r in (0, height - 1)
        on_vertical_edge = c in (0, width - 1)
        assert on_horizontal_edge ^ on_vertical_edge, "markers must lie on one non-corner edge"
        lines.append(("v", c) if on_horizontal_edge else ("h", r))
    assert len(set(lines)) == len(lines), "each marker must define a distinct cut line"
    return lines


def _relation(box: BBox, line: Line) -> str:
    r0, c0, r1, c1 = box
    axis, position = line
    lo, hi = (r0, r1) if axis == "h" else (c0, c1)
    if lo < position < hi:
        return "cross"
    if position in (lo, hi):
        return "tangent"
    return "miss"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split solid rectangles along invisible lines emitted by teal edge markers.

    Input structure:
    - A black grid contains separated solid monochromatic rectangles and one or two single teal pixels on non-corner edges.
    - A left/right marker specifies its row; a top/bottom marker specifies its column, defining a full-grid invisible line.

    Rule:
    - Remove all edge markers. If exactly one specified line crosses a rectangle's interior, erase the rectangle row or column on that line and shift its two remaining halves one cell away, perpendicular to the line.
    - Rectangles merely touched along an outer edge, and rectangles missed by every line, stay unchanged; the invisible lines are never painted.

    Key insight:
    - The edge pixels are remote cut controls: only an interior crossing fractures a rectangle, while tangency is a deliberate no-op counterexample.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "ARC grids are at most 30x30"
    lines = _marker_lines(g)
    rectangles = _rectangles(g)

    out = np.full_like(g, BACKGROUND)
    occupied = np.zeros(g.shape, dtype=bool)
    height, width = g.shape

    def paint(r: int, c: int, color: int) -> None:
        assert 0 <= r < height and 0 <= c < width, "shifted halves must remain in bounds"
        assert not occupied[r, c], "shifted rectangles must not collide"
        occupied[r, c] = True
        out[r, c] = color

    for box, color in rectangles:
        r0, c0, r1, c1 = box
        crossing = [line for line in lines if _relation(box, line) == "cross"]
        assert len(crossing) <= 1, "a rectangle may be cut by at most one marker line"

        if not crossing:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    paint(r, c, color)
            continue

        axis, position = crossing[0]
        if axis == "h":
            for r in range(r0, position):
                for c in range(c0, c1 + 1):
                    paint(r - 1, c, color)
            for r in range(position + 1, r1 + 1):
                for c in range(c0, c1 + 1):
                    paint(r + 1, c, color)
        else:
            for r in range(r0, r1 + 1):
                for c in range(c0, position):
                    paint(r, c - 1, color)
                for c in range(position + 1, c1 + 1):
                    paint(r, c + 1, color)

    return out


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded(box: BBox, amount: int = 1) -> BBox:
    r0, c0, r1, c1 = box
    return r0 - amount, c0 - amount, r1 + amount, c1 + amount


def _sample_line_positions(size: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = list(range(4, size - 4))
    random.shuffle(candidates)
    chosen: list[int] = []
    for value in candidates:
        if all(abs(value - old) >= 5 for old in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return chosen
    return None


def _choose_lines(height: int, width: int, count: int) -> list[Line] | None:
    if count == 1:
        orientations = [random.choice(("h", "v"))]
    else:
        orientations = list(random.choice((("h", "v"), ("h", "h"), ("v", "v"))))

    h_positions = _sample_line_positions(height, orientations.count("h"))
    v_positions = _sample_line_positions(width, orientations.count("v"))
    if h_positions is None or v_positions is None:
        return None
    lines: list[Line] = []
    hi = vi = 0
    for axis in orientations:
        if axis == "h":
            lines.append((axis, h_positions[hi]))
            hi += 1
        else:
            lines.append((axis, v_positions[vi]))
            vi += 1
    return lines


def _candidate_boxes(
    height: int,
    width: int,
    lines: list[Line],
    desired: tuple[str, int | None],
) -> list[BBox]:
    relation, target = desired
    candidates: list[BBox] = []
    for h in range(2, 6):
        for w in range(2, 6):
            for r0 in range(2, height - h - 1):
                for c0 in range(2, width - w - 1):
                    box = (r0, c0, r0 + h - 1, c0 + w - 1)
                    relations = [_relation(box, line) for line in lines]
                    if relation == "miss":
                        valid = all(value == "miss" for value in relations)
                    else:
                        valid = (
                            target is not None
                            and relations[target] == relation
                            and all(value == "miss" for i, value in enumerate(relations) if i != target)
                        )
                    if valid:
                        candidates.append(box)
    random.shuffle(candidates)
    return candidates


def _place_specs(
    height: int,
    width: int,
    lines: list[Line],
    requirements: list[tuple[str, int | None]],
) -> list[tuple[BBox, int]] | None:
    specs: list[tuple[BBox, int]] = []
    reserved: list[BBox] = []
    for desired in requirements:
        options = _candidate_boxes(height, width, lines, desired)
        box = next(
            (candidate for candidate in options if all(not _boxes_overlap(_expanded(candidate), old) for old in reserved)),
            None,
        )
        if box is None:
            return None
        specs.append((box, random.choice(BODY_COLORS)))
        reserved.append(_expanded(box))
    return specs


def _draw_input(height: int, width: int, lines: list[Line], specs: list[tuple[BBox, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for (r0, c0, r1, c1), color in specs:
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    for axis, position in lines:
        if axis == "h":
            grid[position, random.choice((0, width - 1))] = MARKER
        else:
            grid[random.choice((0, height - 1)), position] = MARKER
    return grid


def _expected_from_specs(
    shape: tuple[int, int], lines: list[Line], specs: list[tuple[BBox, int]]
) -> np.ndarray:
    """Metadata oracle, intentionally independent of component/marker discovery."""
    expected = np.zeros(shape, dtype=int)
    for box, color in specs:
        r0, c0, r1, c1 = box
        crossing = [line for line in lines if _relation(box, line) == "cross"]
        if not crossing:
            expected[r0 : r1 + 1, c0 : c1 + 1] = color
        elif crossing[0][0] == "h":
            cut = crossing[0][1]
            expected[r0 - 1 : cut - 1, c0 : c1 + 1] = color
            expected[cut + 2 : r1 + 2, c0 : c1 + 1] = color
        else:
            cut = crossing[0][1]
            expected[r0 : r1 + 1, c0 - 1 : cut - 1] = color
            expected[r0 : r1 + 1, cut + 2 : c1 + 2] = color
    return expected


def _semantic_checks(
    grid: np.ndarray, lines: list[Line], specs: list[tuple[BBox, int]], expected: np.ndarray
) -> bool:
    discovered = sorted(_rectangles(grid))
    if discovered != sorted(specs) or set(_marker_lines(grid)) != set(lines):
        return False

    all_relations = [[_relation(box, line) for line in lines] for box, _ in specs]
    if not all(any(value == "cross" for value in relations) for relations in zip(*all_relations)):
        return False
    if not any("tangent" in relations for relations in all_relations):
        return False
    if not any(all(value == "miss" for value in relations) for relations in all_relations):
        return False
    if any(sum(value == "cross" for value in relations) > 1 for relations in all_relations):
        return False

    marker_only = grid.copy()
    marker_only[marker_only == MARKER] = BACKGROUND
    erase_only = marker_only.copy()
    keep_cut_row = expected.copy()
    for (r0, c0, r1, c1), color in specs:
        crossing = [line for line in lines if _relation((r0, c0, r1, c1), line) == "cross"]
        if not crossing:
            continue
        axis, cut = crossing[0]
        if axis == "h":
            erase_only[cut, c0 : c1 + 1] = BACKGROUND
            keep_cut_row[cut, c0 : c1 + 1] = color
        else:
            erase_only[r0 : r1 + 1, cut] = BACKGROUND
            keep_cut_row[r0 : r1 + 1, cut] = color
    if np.array_equal(expected, marker_only) or np.array_equal(expected, erase_only):
        return False
    if np.array_equal(expected, keep_cut_row) or np.any(expected == MARKER):
        return False

    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    return bool(np.array_equal(actual, expected))


def _generation_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    line_count = random.choice((1, 2))
    lines = _choose_lines(height, width, line_count)
    if lines is None:
        return None

    requirements: list[tuple[str, int | None]] = [("cross", i) for i in range(line_count)]
    requirements.extend((("tangent", random.randrange(line_count)), ("miss", None)))
    extra_count = random.randint(0 if line_count == 2 else 1, 2)
    for _ in range(extra_count):
        if random.random() < 0.65:
            requirements.append(("cross", random.randrange(line_count)))
        else:
            requirements.append(("miss", None))
    random.shuffle(requirements)

    specs = _place_specs(height, width, lines, requirements)
    if specs is None:
        return None
    grid = _draw_input(height, width, lines, specs)
    expected = _expected_from_specs(grid.shape, lines, specs)
    if not _semantic_checks(grid, lines, specs, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked invisible-fracture puzzle instance."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-fracture input")
