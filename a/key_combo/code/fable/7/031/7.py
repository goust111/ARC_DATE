"""ARC task 7: connect paired markers with shorter-segment priority."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Segment:
    color: int
    a: tuple[int, int]
    b: tuple[int, int]

    @property
    def length(self) -> int:
        """Endpoint distance (and therefore the priority measure)."""
        return abs(self.a[0] - self.b[0]) + abs(self.a[1] - self.b[1])

    @property
    def horizontal(self) -> bool:
        return self.a[0] == self.b[0]

    def cells(self, strict: bool = False) -> list[tuple[int, int]]:
        r0, c0 = self.a
        r1, c1 = self.b
        dr = (r1 > r0) - (r1 < r0)
        dc = (c1 > c0) - (c1 < c0)
        cells = [(r0 + k * dr, c0 + k * dc) for k in range(self.length + 1)]
        return cells[1:-1] if strict else cells


def _read_segments(grid: np.ndarray) -> list[_Segment]:
    """Recover the unambiguous marker pair belonging to every visible color."""
    segments: list[_Segment] = []
    for color in map(int, np.unique(grid)):
        if color == 0:
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) == 2, "each non-black color must occur at exactly two markers"
        a, b = coords
        assert a[0] == b[0] or a[1] == b[1], "paired markers must align in one row or column"
        assert a != b
        segments.append(_Segment(color, a, b))
    assert segments, "expected at least one marker pair"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect every same-colored marker pair, giving a crossing cell to the shorter segment.

    Input structure:
    - A black background contains exactly two marker pixels of every foreground color.
    - Each same-colored pair is aligned in one row or one column and defines a line segment.
    - Intersecting segments have unequal endpoint distances.

    Rule:
    - Fill the cells strictly between each marker pair with that pair's color.
    - At a cell traversed by two segments, use the color of the segment with the
      shorter endpoint distance; all original marker pixels remain unchanged.

    Key insight:
    - Crossing priority is determined by geometric segment length, not by color
      number or by whether a segment is horizontal or vertical.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"

    segments = _read_segments(g)
    cell_sets = {s: set(s.cells()) for s in segments}
    for i, first in enumerate(segments):
        for second in segments[i + 1 :]:
            if cell_sets[first] & cell_sets[second]:
                assert first.length != second.length, "crossing segments must have different lengths"

    # Original markers are already present.  Short segments claim blank cells first;
    # longer ones subsequently fill only cells which remain black.
    out = g.copy()
    for segment in sorted(segments, key=lambda s: s.length):
        for r, c in segment.cells(strict=True):
            if out[r, c] == 0:
                out[r, c] = segment.color
    return out


def _intersection(a: _Segment, b: _Segment) -> tuple[int, int] | None:
    if a.horizontal == b.horizontal:
        return None
    h, v = (a, b) if a.horizontal else (b, a)
    r, c = h.a[0], v.a[1]
    h_lo, h_hi = sorted((h.a[1], h.b[1]))
    v_lo, v_hi = sorted((v.a[0], v.b[0]))
    # Generator intersections are interior-to-interior, so markers never acquire
    # a second role as somebody else's crossing cell.
    if h_lo < c < h_hi and v_lo < r < v_hi:
        return r, c
    return None


def _semantic_oracle(shape: tuple[int, int], segments: list[_Segment]) -> np.ndarray:
    """Cell-wise specification, deliberately independent of main's paint loop."""
    expected = np.zeros(shape, dtype=int)
    endpoints: dict[tuple[int, int], int] = {}
    for segment in segments:
        endpoints[segment.a] = segment.color
        endpoints[segment.b] = segment.color

    for r in range(shape[0]):
        for c in range(shape[1]):
            if (r, c) in endpoints:
                expected[r, c] = endpoints[(r, c)]
                continue
            covering = [s for s in segments if (r, c) in set(s.cells(strict=True))]
            if covering:
                expected[r, c] = min(covering, key=lambda s: s.length).color
    return expected


def _try_geometry(height: int, width: int, count: int) -> list[_Segment] | None:
    orientations = [True, True, False, False]
    orientations += [bool(random.getrandbits(1)) for _ in range(count - 4)]
    random.shuffle(orientations)

    raw: list[_Segment] = []
    used_lengths: set[int] = set()
    for idx, horizontal in enumerate(orientations):
        placed = False
        for _ in range(100):
            if horizontal:
                r = random.randint(2, height - 3)
                c0, c1 = sorted(random.sample(range(1, width - 1), 2))
                candidate = _Segment(idx + 1, (r, c0), (r, c1))
            else:
                c = random.randint(2, width - 3)
                r0, r1 = sorted(random.sample(range(1, height - 1), 2))
                candidate = _Segment(idx + 1, (r0, c), (r1, c))
            if not 4 <= candidate.length <= 15 or candidate.length in used_lengths:
                continue
            candidate_cells = set(candidate.cells())
            candidate_ends = {candidate.a, candidate.b}
            valid = True
            for old in raw:
                old_cells = set(old.cells())
                overlap = candidate_cells & old_cells
                if candidate.horizontal == old.horizontal:
                    valid &= not overlap
                else:
                    # Any overlap must be one clean interior crossing.
                    valid &= (not overlap) or (
                        len(overlap) == 1
                        and overlap.isdisjoint(candidate_ends)
                        and overlap.isdisjoint({old.a, old.b})
                    )
                if min(
                    abs(x[0] - y[0]) + abs(x[1] - y[1])
                    for x in candidate_ends for y in (old.a, old.b)
                ) <= 1:
                    valid = False
            if valid:
                raw.append(candidate)
                used_lengths.add(candidate.length)
                placed = True
                break
        if not placed:
            return None

    crossings = [(a, b, _intersection(a, b)) for i, a in enumerate(raw) for b in raw[i + 1 :]]
    crossings = [(a, b, p) for a, b, p in crossings if p is not None]
    if not 2 <= len(crossings) <= 6:
        return None
    winner_directions = {
        min((a, b), key=lambda s: s.length).horizontal for a, b, _ in crossings
    }
    if winner_directions != {False, True}:
        return None
    return raw


def _assign_colors(segments: list[_Segment]) -> list[_Segment] | None:
    """Randomize colors while visibly refuting low-id and high-id priority rules."""
    for _ in range(100):
        colors = random.sample(range(1, 10), len(segments))
        colored = [_Segment(color, s.a, s.b) for color, s in zip(colors, segments)]
        comparisons: set[bool] = set()
        for i, a in enumerate(colored):
            for b in colored[i + 1 :]:
                if _intersection(a, b) is not None:
                    short, long = sorted((a, b), key=lambda s: s.length)
                    comparisons.add(short.color < long.color)
        if comparisons == {False, True}:
            return colored
    return None


def _validate_generated(inp: np.ndarray, segments: list[_Segment]) -> bool:
    reconstructed = _read_segments(inp)
    if {(s.color, s.a, s.b) for s in reconstructed} != {(s.color, s.a, s.b) for s in segments}:
        return False
    expected = _semantic_oracle(inp.shape, segments)
    actual = main(inp)
    if not np.array_equal(actual, expected) or np.array_equal(inp, actual):
        return False

    crossings = []
    for i, a in enumerate(segments):
        for b in segments[i + 1 :]:
            point = _intersection(a, b)
            if point is not None:
                crossings.append((a, b, point))
    if len(crossings) < 2:
        return False
    # Check the actual conflict cells against the stated length comparison.
    if any(actual[p] != min((a, b), key=lambda s: s.length).color for a, b, p in crossings):
        return False
    # Every pair contributes visible interior cells, so connection is never decorative.
    return all(any(actual[cell] == s.color for cell in s.cells(strict=True)) for s in segments)


def generate_input() -> np.ndarray:
    """Generate varied, anti-shortcut marker layouts with informative crossings."""
    for _ in range(600):
        height = random.randint(14, 24)
        width = random.randint(14, 24)
        count = random.randint(4, 6)
        geometry = _try_geometry(height, width, count)
        if geometry is None:
            continue
        segments = _assign_colors(geometry)
        if segments is None:
            continue
        inp = np.zeros((height, width), dtype=int)
        for segment in segments:
            inp[segment.a] = segment.color
            inp[segment.b] = segment.color
        if _validate_generated(inp, segments):
            return inp
    raise RuntimeError("failed to generate a valid short-line-priority input")
