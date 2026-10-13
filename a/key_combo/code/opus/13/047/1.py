from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BLACK = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_ROWS = (11, 17)
GRID_COLS = (11, 17)
PAIR_COUNT_OPTIONS = (2, 3)
PAIR_COUNT_WEIGHTS = (0.55, 0.45)
TIE_MODE_PROB = 0.4
THIRD_OVERLAP_PROB = 0.6
MAX_GENERATION_ATTEMPTS = 800

RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 7
MAX_OVERLAP_FRACTION = 0.35


@dataclass(frozen=True)
class RectSpec:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int
    markers: tuple[tuple[int, int], tuple[int, int]]

    @property
    def area(self) -> int:
        return (self.r1 - self.r0 + 1) * (self.c1 - self.c0 + 1)

    @property
    def key(self) -> tuple[int, int, int]:
        return self.area, self.r0, self.c0

    def covers(self, r: int, c: int) -> bool:
        return self.r0 <= r <= self.r1 and self.c0 <= c <= self.c1

    def cell_value(self, r: int, c: int, *, local_phase: bool = True) -> int:
        if (r, c) in self.markers:
            return self.color
        if local_phase:
            even = ((r - self.r0) + (c - self.c0)) % 2 == 0
        else:
            even = (r + c) % 2 == 0
        return self.color if even else BLACK


def _extract_rectangles(grid: np.ndarray) -> list[RectSpec]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BLACK]
    assert 2 <= len(colors) <= 3, "expected 2 or 3 distinct marker colors"

    rects: list[RectSpec] = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each color must appear exactly twice"
        (r_a, c_a), (r_b, c_b) = coords.tolist()
        assert r_a != r_b and c_a != c_b, "marker pair cannot share a row or column"
        r0, r1 = sorted((int(r_a), int(r_b)))
        c0, c1 = sorted((int(c_a), int(c_b)))
        rects.append(
            RectSpec(
                color=color,
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                markers=((int(r_a), int(c_a)), (int(r_b), int(c_b))),
            )
        )
    assert len({rect.key for rect in rects}) == len(rects), "priority keys must be unique"
    return rects


def _render_by_overlay(rects: list[RectSpec], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in sorted(rects, key=lambda spec: spec.key, reverse=True):
        for r in range(rect.r0, rect.r1 + 1):
            for c in range(rect.c0, rect.c1 + 1):
                out[r, c] = rect.cell_value(r, c, local_phase=True)
    return out


def _render_by_priority(
    rects: list[RectSpec],
    shape: tuple[int, int],
    *,
    local_phase: bool = True,
    reverse_area: bool = False,
    reverse_equal_tie: bool = False,
) -> np.ndarray:
    if reverse_area:
        priority = sorted(rects, key=lambda spec: (-spec.area, spec.r0, spec.c0))
    elif reverse_equal_tie:
        priority = sorted(rects, key=lambda spec: (spec.area, -spec.r0, -spec.c0))
    else:
        priority = sorted(rects, key=lambda spec: spec.key)

    out = np.zeros(shape, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            chosen = None
            for rect in priority:
                if rect.covers(r, c):
                    chosen = rect
                    break
            if chosen is not None:
                out[r, c] = chosen.cell_value(r, c, local_phase=local_phase)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each marker-defined rectangle with a locally anchored checker weave and let smaller rectangles win overlaps.

    Input structure:
    - A black background contains 2 or 3 color-distinct marker pairs, with each color appearing exactly twice.
    - Each same-color pair is isolated and its two cells lie on different rows and different columns.
    - Every pair defines the opposite corners of one hidden axis-aligned rectangle.

    Rule:
    - For each same-color pair, take the inclusive rectangle spanning the two markers.
    - Inside that rectangle, use the rectangle's own top-left corner as the parity anchor: cells with even
      `(r-r_min) + (c-c_min)` become the pair color and odd cells become black.
    - The two marker cells themselves always stay in their own color.
    - When rectangles overlap, choose the rectangle with smaller area; equal areas break ties by smaller top-left row,
      then smaller top-left column. Cells outside every rectangle remain black.

    Key insight:
    - The visible color of an overlap cell depends on two coupled facts at once: which rectangle wins the area-based
      priority contest, and that winner's own local checker phase rather than any global board parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects = _extract_rectangles(g)
    return _render_by_overlay(rects, g.shape)


def _intersection_area(a: RectSpec, b: RectSpec) -> int:
    r0 = max(a.r0, b.r0)
    c0 = max(a.c0, b.c0)
    r1 = min(a.r1, b.r1)
    c1 = min(a.c1, b.c1)
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _all_markers(rects: list[RectSpec]) -> list[tuple[int, int]]:
    return [marker for rect in rects for marker in rect.markers]


def _marker_separation_ok(rects: list[RectSpec]) -> bool:
    markers = _all_markers(rects)
    for i, (r0, c0) in enumerate(markers):
        for r1, c1 in markers[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) < 2:
                return False
    return True


def _markers_unambiguous(rects: list[RectSpec]) -> bool:
    for i, rect in enumerate(rects):
        for marker in rect.markers:
            for j, other in enumerate(rects):
                if i != j and other.covers(*marker):
                    return False
    return True


def _unique_overlap_cells(rects: list[RectSpec], shape: tuple[int, int]) -> tuple[int, int]:
    overlap = 0
    union = 0
    for r in range(shape[0]):
        for c in range(shape[1]):
            count = sum(rect.covers(r, c) for rect in rects)
            if count:
                union += 1
            if count > 1:
                overlap += 1
    return overlap, union


def _priority_effects_exist(rects: list[RectSpec], shape: tuple[int, int]) -> tuple[bool, bool]:
    black_override = False
    color_override = False
    priority = sorted(rects, key=lambda spec: spec.key)
    for r in range(shape[0]):
        for c in range(shape[1]):
            covering = [rect for rect in priority if rect.covers(r, c)]
            if len(covering) < 2:
                continue
            top = covering[0]
            top_value = top.cell_value(r, c)
            lower_values = [rect.cell_value(r, c) for rect in covering[1:]]
            if top_value == BLACK and any(value != BLACK for value in lower_values):
                black_override = True
            if top_value != BLACK and any(value != top_value for value in lower_values):
                color_override = True
    return black_override, color_override


def _every_rectangle_matters(rects: list[RectSpec], shape: tuple[int, int], expected: np.ndarray) -> bool:
    for idx in range(len(rects)):
        reduced = rects[:idx] + rects[idx + 1 :]
        if np.array_equal(_render_by_priority(reduced, shape), expected):
            return False
    return True


def _scene_valid(rects: list[RectSpec], shape: tuple[int, int], *, require_tie: bool) -> bool:
    if len(rects) not in (2, 3):
        return False
    if len({rect.color for rect in rects}) != len(rects):
        return False
    if len({rect.key for rect in rects}) != len(rects):
        return False
    if not _marker_separation_ok(rects):
        return False
    if not _markers_unambiguous(rects):
        return False

    overlaps = [
        (i, j, _intersection_area(rects[i], rects[j]))
        for i in range(len(rects))
        for j in range(i + 1, len(rects))
    ]
    if not any(area > 0 for _, _, area in overlaps):
        return False

    overlap_cells, union_cells = _unique_overlap_cells(rects, shape)
    if overlap_cells < 2:
        return False
    if overlap_cells > max(6, int(np.floor(union_cells * MAX_OVERLAP_FRACTION))):
        return False

    expected = _render_by_priority(rects, shape)
    if not np.array_equal(expected, _render_by_overlay(rects, shape)):
        return False
    if np.count_nonzero(expected) < 10:
        return False
    if not _every_rectangle_matters(rects, shape, expected):
        return False

    wrong_global_phase = _render_by_priority(rects, shape, local_phase=False)
    if np.array_equal(expected, wrong_global_phase):
        return False

    black_override, color_override = _priority_effects_exist(rects, shape)
    if not (black_override and color_override):
        return False

    if require_tie:
        tie_overlap = any(
            area > 0 and rects[i].area == rects[j].area
            for i, j, area in overlaps
        )
        if not tie_overlap:
            return False
        wrong_tie = _render_by_priority(rects, shape, reverse_equal_tie=True)
        if np.array_equal(expected, wrong_tie):
            return False
    else:
        strict_overlap = any(
            area > 0 and rects[i].area != rects[j].area
            for i, j, area in overlaps
        )
        if not strict_overlap:
            return False
        wrong_area = _render_by_priority(rects, shape, reverse_area=True)
        if np.array_equal(expected, wrong_area):
            return False

    return True


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1):
        if area % h == 0:
            w = area // h
            if RECT_SIDE_MIN <= w <= RECT_SIDE_MAX:
                pairs.append((h, w))
    return pairs


def _build_rect(color: int, r0: int, c0: int, h: int, w: int, diag: int) -> RectSpec:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if diag == 0:
        markers = ((r0, c0), (r1, c1))
    else:
        markers = ((r0, c1), (r1, c0))
    return RectSpec(color=color, r0=r0, c0=c0, r1=r1, c1=c1, markers=markers)


def _sample_bounds(rng: np.random.Generator, rows: int, cols: int, area: int | None = None) -> tuple[int, int, int, int]:
    if area is None:
        h = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
        w = int(rng.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
    else:
        candidates = _factor_pairs(area)
        assert candidates
        h, w = candidates[int(rng.integers(len(candidates)))]
    r0 = int(rng.integers(0, rows - h + 1))
    c0 = int(rng.integers(0, cols - w + 1))
    return r0, c0, h, w


def _sample_primary_pair(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    colors: tuple[int, ...],
    *,
    require_tie: bool,
) -> list[RectSpec] | None:
    areas = [h * w for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1) for w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1)]
    for _ in range(250):
        if require_tie:
            area = int(rng.choice(areas))
            if not _factor_pairs(area):
                continue
            r0_a, c0_a, h_a, w_a = _sample_bounds(rng, rows, cols, area=area)
            r0_b, c0_b, h_b, w_b = _sample_bounds(rng, rows, cols, area=area)
        else:
            r0_a, c0_a, h_a, w_a = _sample_bounds(rng, rows, cols)
            r0_b, c0_b, h_b, w_b = _sample_bounds(rng, rows, cols)
            if h_a * w_a == h_b * w_b:
                continue

        if (r0_a, c0_a) == (r0_b, c0_b):
            continue
        if (r0_a + c0_a) % 2 == (r0_b + c0_b) % 2:
            continue

        base_a = _build_rect(colors[0], r0_a, c0_a, h_a, w_a, diag=0)
        base_b = _build_rect(colors[1], r0_b, c0_b, h_b, w_b, diag=0)
        overlap = _intersection_area(base_a, base_b)
        if overlap < 2:
            continue
        if overlap >= min(base_a.area, base_b.area):
            continue

        for diag_a in rng.permutation(2):
            for diag_b in rng.permutation(2):
                rects = [
                    _build_rect(colors[0], r0_a, c0_a, h_a, w_a, int(diag_a)),
                    _build_rect(colors[1], r0_b, c0_b, h_b, w_b, int(diag_b)),
                ]
                if _scene_valid(rects, (rows, cols), require_tie=require_tie):
                    return rects
    return None


def _sample_third_rect(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    color: int,
    existing: list[RectSpec],
    *,
    require_overlap: bool,
) -> RectSpec | None:
    for _ in range(200):
        r0, c0, h, w = _sample_bounds(rng, rows, cols)
        if any((h * w, r0, c0) == rect.key for rect in existing):
            continue
        overlaps = [
            _intersection_area(_build_rect(color, r0, c0, h, w, 0), rect)
            for rect in existing
        ]
        if require_overlap and not any(area > 0 for area in overlaps):
            continue
        if any(area >= min(h * w, rect.area) for area, rect in zip(overlaps, existing)):
            continue

        for diag in rng.permutation(2):
            rect = _build_rect(color, r0, c0, h, w, int(diag))
            if not _marker_separation_ok(existing + [rect]):
                continue
            if not _markers_unambiguous(existing + [rect]):
                continue
            return rect
    return None


def _build_input(rects: list[RectSpec], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        for r, c in rect.markers:
            grid[r, c] = rect.color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        rows = int(rng.integers(*GRID_ROWS))
        cols = int(rng.integers(*GRID_COLS))
        pair_count = int(rng.choice(PAIR_COUNT_OPTIONS, p=PAIR_COUNT_WEIGHTS))
        require_tie = bool(rng.random() < TIE_MODE_PROB)
        colors = tuple(int(color) for color in rng.choice(COLOR_POOL, size=pair_count, replace=False))

        rects = _sample_primary_pair(rng, rows, cols, colors, require_tie=require_tie)
        if rects is None:
            continue

        if pair_count == 3:
            third = _sample_third_rect(
                rng,
                rows,
                cols,
                colors[2],
                rects,
                require_overlap=bool(rng.random() < THIRD_OVERLAP_PROB),
            )
            if third is None:
                continue
            rects = rects + [third]

        if not _scene_valid(rects, (rows, cols), require_tie=require_tie):
            continue

        grid = _build_input(rects, (rows, cols))
        if not np.array_equal(main(grid), _render_by_priority(rects, (rows, cols))):
            continue
        return grid

    raise RuntimeError("failed to generate a valid marker-pair weave input")
