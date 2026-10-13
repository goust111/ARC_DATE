from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

INPUT_COLORS = (RED, GREEN, ORANGE, MAROON)


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    h: int
    w: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def interior_area(self) -> int:
        return (self.h - 2) * (self.w - 2)

    @property
    def outer_area(self) -> int:
        return self.h * self.w

    def shifted(self, dr: int) -> "Rect":
        return Rect(self.color, self.r0 + dr, self.c0, self.h, self.w)


def _grid_hash(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode()).hexdigest()


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_rectangles(grid: np.ndarray) -> list[Rect]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[Rect] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rh, rw = r1 - r0 + 1, c1 - c0 + 1
            assert rh >= 3 and rw >= 3, "every foreground component must be a rectangle with size >= 3x3"
            assert len(coords) == rh * rw, "every foreground component must fill its bounding box"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "foreground components must be solid rectangles"
            rects.append(Rect(color, r0, c0, rh, rw))

    assert 3 <= len(rects) <= 5, "expected 3-5 rectangles"
    return rects


def _drop_rectangles(rects: list[Rect], shape: tuple[int, int]) -> list[Rect]:
    h, _w = shape
    settled_mask = np.zeros(shape, dtype=bool)
    settled: list[Rect] = []

    order = sorted(rects, key=lambda rect: (-rect.r1, -rect.r0, rect.c0))
    for rect in order:
        shift = 0
        while rect.r1 + shift + 1 < h:
            next_row = rect.r1 + shift + 1
            if bool(settled_mask[next_row, rect.c0 : rect.c1 + 1].any()):
                break
            shift += 1
        placed = rect.shifted(shift)
        assert not bool(settled_mask[placed.r0 : placed.r1 + 1, placed.c0 : placed.c1 + 1].any())
        settled_mask[placed.r0 : placed.r1 + 1, placed.c0 : placed.c1 + 1] = True
        settled.append(placed)

    return settled


def _rank_rectangles(rects: list[Rect]) -> list[Rect]:
    areas = [rect.interior_area for rect in rects]
    assert len(set(areas)) == len(areas), "internal areas must be unique"
    return sorted(rects, key=lambda rect: (-rect.interior_area, rect.c0, rect.r0))


def _paint_rect(out: np.ndarray, rect: Rect, rank: int) -> None:
    if rank == 1:
        border_color = YELLOW
        interior_color = RED
    elif rank == 2:
        border_color = YELLOW
        interior_color = rect.color
    else:
        border_color = rect.color
        interior_color = rect.color

    out[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = interior_color
    out[rect.r0, rect.c0 : rect.c1 + 1] = border_color
    out[rect.r1, rect.c0 : rect.c1 + 1] = border_color
    out[rect.r0 : rect.r1 + 1, rect.c0] = border_color
    out[rect.r0 : rect.r1 + 1, rect.c1] = border_color
    out[(rect.r0, rect.r0, rect.r1, rect.r1), (rect.c0, rect.c1, rect.c0, rect.c1)] = BLUE


def _render_output(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ranked = _rank_rectangles(rects)
    for rank, rect in enumerate(ranked, start=1):
        _paint_rect(out, rect, rank)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop all floating solid rectangles, rank them by settled internal area, then recolor each rectangle by rank-dependent structural depth.

    Input structure:
    - The grid is black background with 3-5 solid single-color rectangles.
    - Each foreground object is a filled axis-aligned rectangle of size at least 3x3.
    - Rectangles may start at different heights and must be interpreted with the grid's row/column coordinates.

    Rule:
    - Identify every solid rectangle and apply downward gravity to the rectangles from lower to higher, keeping each shape rigid until its bottom touches the grid floor or a previously settled rectangle.
    - After all rectangles settle, compute each rectangle's internal area as `(height - 2) * (width - 2)` and rank rectangles from largest to smallest, breaking ties by smaller left column.
    - Recolor by rank: rank 1 gets blue corners, yellow non-corner border, and red strict interior; rank 2 gets blue corners, yellow non-corner border, and its original interior color; rank 3 and below get only blue corners while all other cells keep the original color.
    - Leave all background cells black.

    Key insight:
    - The final colors are determined by a strict three-stage dependency: gravity fixes the rectangles' settled objects, settled geometry determines the internal-area ranking, and the ranking selects how deep the corner/border/interior recoloring goes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    rects = _find_rectangles(g)
    settled = _drop_rectangles(rects, g.shape)
    return _render_output(g.shape, settled)


_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()
_GEN_ATTEMPTS = 2000
_GEN_MIN_SIZE = 14
_GEN_MAX_SIZE = 24


def _sample_dimensions(rng: np.random.Generator, count: int) -> list[tuple[int, int]] | None:
    candidates = [(h, w) for h in range(3, 9) for w in range(3, 10)]
    for _ in range(400):
        pick_idx = rng.choice(len(candidates), size=count, replace=False)
        dims = [candidates[int(i)] for i in np.atleast_1d(pick_idx)]
        inner_areas = [(h - 2) * (w - 2) for h, w in dims]
        if len(set(inner_areas)) != count:
            continue
        outer_rank = sorted(range(count), key=lambda i: (-dims[i][0] * dims[i][1], i))
        inner_rank = sorted(range(count), key=lambda i: (-inner_areas[i], i))
        if outer_rank == inner_rank:
            continue
        total_width = sum(w for _, w in dims) + (count - 1)
        if total_width > _GEN_MAX_SIZE - 2:
            continue
        return dims
    return None


def _compose_width(rng: np.random.Generator, widths: list[int]) -> tuple[int, list[int]]:
    count = len(widths)
    gaps = [int(rng.integers(1, 4)) for _ in range(count - 1)]
    left_margin = int(rng.integers(0, 3))
    right_margin = int(rng.integers(0, 3))
    base_width = left_margin + sum(widths) + sum(gaps) + right_margin
    extra = int(rng.integers(0, max(1, _GEN_MAX_SIZE - base_width + 1)))
    if count > 1:
        bonus = rng.multinomial(extra, np.full(count + 1, 1.0 / (count + 1)))
    else:
        bonus = np.array([extra, 0], dtype=int)
    left_margin += int(bonus[0])
    right_margin += int(bonus[-1])
    for i in range(count - 1):
        gaps[i] += int(bonus[i + 1])
    positions = []
    cursor = left_margin
    for idx, width in enumerate(widths):
        positions.append(cursor)
        cursor += width
        if idx < count - 1:
            cursor += gaps[idx]
    return cursor + right_margin, positions


def _assign_colors_by_rank(rng: np.random.Generator, count: int) -> list[int]:
    remaining = list(INPUT_COLORS)
    non_red = [color for color in remaining if color != RED]
    rank1 = int(non_red[int(rng.integers(len(non_red)))])
    remaining.remove(rank1)
    non_red = [color for color in remaining if color != RED]
    rank2 = int(non_red[int(rng.integers(len(non_red)))])
    remaining.remove(rank2)
    rng.shuffle(remaining)
    ranked_colors = [rank1, rank2, *remaining[: count - 2]]
    return ranked_colors


def _build_specs(rng: np.random.Generator) -> tuple[tuple[int, int], list[Rect]] | None:
    count = int(rng.integers(3, 5))
    dims = _sample_dimensions(rng, count)
    if dims is None:
        return None

    order = list(range(count))
    rng.shuffle(order)
    dims = [dims[i] for i in order]

    widths = [w for _h, w in dims]
    grid_w, cols = _compose_width(rng, widths)
    if not (_GEN_MIN_SIZE <= grid_w <= _GEN_MAX_SIZE):
        return None

    max_h = max(h for h, _w in dims)
    min_grid_h = max(_GEN_MIN_SIZE, max_h + count + 1)
    max_grid_h = min(_GEN_MAX_SIZE, max_h + count + 6)
    if min_grid_h > max_grid_h:
        return None
    grid_h = int(rng.integers(min_grid_h, max_grid_h + 1))

    clearances = list(map(int, rng.choice(np.arange(1, grid_h - max_h + 1), size=count, replace=False)))
    rng.shuffle(clearances)
    if max(clearances) == 1:
        return None

    provisional = [Rect(-1, grid_h - h - clearances[idx], cols[idx], h, w) for idx, (h, w) in enumerate(dims)]
    ranked = _rank_rectangles(provisional)
    ranked_colors = _assign_colors_by_rank(rng, count)
    color_by_rank = {rect: ranked_colors[rank - 1] for rank, rect in enumerate(ranked, start=1)}

    specs = [
        Rect(color_by_rank[rect], rect.r0, rect.c0, rect.h, rect.w)
        for rect in provisional
    ]
    return (grid_h, grid_w), specs


def _manual_expected(shape: tuple[int, int], specs: list[Rect]) -> np.ndarray:
    settled = [Rect(rect.color, shape[0] - rect.h, rect.c0, rect.h, rect.w) for rect in specs]
    return _render_output(shape, settled)


def _column_ranges_disjoint(rects: list[Rect]) -> bool:
    ordered = sorted(rects, key=lambda rect: rect.c0)
    for left, right in zip(ordered, ordered[1:]):
        if left.c1 >= right.c0:
            return False
    return True


def _semantic_checks(inp: np.ndarray, specs: list[Rect], expected: np.ndarray) -> bool:
    try:
        rects = _find_rectangles(inp)
    except AssertionError:
        return False

    if len(rects) != len(specs):
        return False
    if not _column_ranges_disjoint(rects):
        return False
    if set(rect.color for rect in rects) != set(rect.color for rect in specs):
        return False
    if len(set(rect.color for rect in rects)) != len(rects):
        return False

    bottoms = [rect.r1 for rect in rects]
    if len(set(bottoms)) != len(bottoms):
        return False

    clearances = [inp.shape[0] - rect.h - rect.r0 for rect in rects]
    if min(clearances) < 1 or max(clearances) <= 1:
        return False

    settled = [Rect(rect.color, inp.shape[0] - rect.h, rect.c0, rect.h, rect.w) for rect in rects]
    inner_order = [(rect.h, rect.w, rect.c0) for rect in _rank_rectangles(settled)]
    outer_order = [(rect.h, rect.w, rect.c0) for rect in sorted(settled, key=lambda rect: (-rect.outer_area, rect.c0))]
    if inner_order == outer_order:
        return False

    ranked = _rank_rectangles(settled)
    if ranked[0].color == RED or ranked[1].color == RED:
        return False

    rank1, rank2 = ranked[0], ranked[1]
    if not np.all(expected[rank1.r0 + 1 : rank1.r1, rank1.c0 + 1 : rank1.c1] == RED):
        return False
    if not np.all(expected[rank2.r0 + 1 : rank2.r1, rank2.c0 + 1 : rank2.c1] == rank2.color):
        return False
    for rect in ranked[2:]:
        block = expected[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1]
        if np.any(block == YELLOW):
            return False
        if np.count_nonzero(block == BLUE) != 4:
            return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_specs(rng)
    if built is None:
        return None
    shape, specs = built

    inp = np.zeros(shape, dtype=int)
    for rect in specs:
        inp[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = rect.color

    expected = _manual_expected(shape, specs)
    if not _semantic_checks(inp, specs, expected):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    in_sig = _grid_hash(inp)
    out_sig = _grid_hash(expected)
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(in_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
