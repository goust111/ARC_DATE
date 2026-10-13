from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

TOP_BAND_LAST_ROW = 9
MAX_GENERATION_ATTEMPTS = 600
AREA_DIMS: dict[int, tuple[tuple[int, int], ...]] = {
    4: ((2, 2),),
    6: ((2, 3), (3, 2)),
    8: ((2, 4), (4, 2)),
    9: ((3, 3),),
    10: ((2, 5), (5, 2)),
    12: ((3, 4), (4, 3)),
}
PAIR_AREA_CHOICES = (
    (4, 9),
    (6, 9),
    (8, 12),
    (6, 12),
)


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    h: int
    w: int
    area: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def final_top(self) -> int:
        return self.r0 + self.area

    @property
    def final_bottom(self) -> int:
        return self.final_top + self.h - 1

    def visible_height(self, height: int) -> int:
        if self.final_top >= height:
            return 0
        return min(self.final_bottom, height - 1) - self.final_top + 1

    def is_fully_clipped(self, height: int) -> bool:
        return self.visible_height(height) == 0

    def is_partially_clipped(self, height: int) -> bool:
        return 0 < self.visible_height(height) < self.h

    def clipped_rows(self, height: int) -> int:
        return max(0, self.final_bottom - (height - 1))

    def local_color_at(self, out_r: int, out_c: int) -> int:
        local_r = out_r - self.final_top
        local_c = out_c - self.c0
        assert 0 <= local_r < self.h and 0 <= local_c < self.w
        on_top_bottom = local_r in (0, self.h - 1)
        on_left_right = local_c in (0, self.w - 1)
        if on_top_bottom and on_left_right:
            return BLUE
        if on_top_bottom or on_left_right:
            return YELLOW
        return RED


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_gray_rectangles(grid: np.ndarray) -> list[Rect]:
    g = np.asarray(grid, dtype=int)
    mask = g == GREY
    seen = np.zeros_like(mask, dtype=bool)
    rectangles: list[Rect] = []
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            area = len(coords)
            assert area == rect_h * rect_w, "gray component must fill its bounding box"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY), "gray rectangle must be solid"
            rectangles.append(Rect(r0=r0, c0=c0, h=rect_h, w=rect_w, area=area))

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray rectangle downward by its area, then recolor the moved shape by local position with small rectangles on top.

    Input structure:
    - The grid uses black background and contains 2-4 gray 4-connected solid rectangles.
    - Every rectangle is axis-aligned, at least 2x2, lies in the upper part of the grid, and all rectangle areas are distinct.

    Rule:
    - Detect every gray rectangle and compute its area in pixels.
    - Remove the original rectangles, move each rectangle straight downward by exactly its own area, and clip any pixels that fall below the bottom edge.
    - On the moved rectangle, recolor original-corner pixels blue, original non-corner border pixels yellow, and original interior pixels red.
    - If moved rectangles overlap, the rectangle with smaller area is on top, so painting proceeds from larger area to smaller area.

    Key insight:
    - The rectangle area has two roles at once: it controls the downward displacement and also resolves overlap priority after movement, while the recoloring must still be computed from each rectangle's original full geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "expected only black and gray"

    rectangles = _find_gray_rectangles(g)
    assert 2 <= len(rectangles) <= 4, "expected 2-4 gray rectangles"
    assert all(rect.h >= 2 and rect.w >= 2 for rect in rectangles), "rectangles must be at least 2x2"
    assert len({rect.area for rect in rectangles}) == len(rectangles), "rectangle areas must be distinct"

    out = np.full_like(g, BLACK)
    height, width = g.shape

    for rect in sorted(rectangles, key=lambda item: item.area, reverse=True):
        for local_r in range(rect.h):
            out_r = rect.final_top + local_r
            if out_r >= height:
                break
            for local_c in range(rect.w):
                out_c = rect.c0 + local_c
                assert 0 <= out_c < width
                out[out_r, out_c] = rect.local_color_at(out_r, out_c)

    return out


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _rectangles_separated(a: Rect, b: Rect) -> bool:
    return (
        a.r1 + 1 < b.r0
        or b.r1 + 1 < a.r0
        or a.c1 + 1 < b.c0
        or b.c1 + 1 < a.c0
    )


def _render_input(height: int, width: int, rects: list[Rect]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for rect in rects:
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = GREY
    return grid


def _output_cover_count(height: int, width: int, rects: list[Rect]) -> np.ndarray:
    cover = np.zeros((height, width), dtype=int)
    for rect in rects:
        visible_h = rect.visible_height(height)
        if visible_h == 0:
            continue
        cover[rect.final_top : rect.final_top + visible_h, rect.c0 : rect.c1 + 1] += 1
    return cover


def _distinct_overlap_colors(top_rect: Rect, bottom_rect: Rect, height: int) -> bool:
    row0 = max(top_rect.final_top, bottom_rect.final_top)
    row1 = min(min(top_rect.final_bottom, height - 1), min(bottom_rect.final_bottom, height - 1))
    col0 = max(top_rect.c0, bottom_rect.c0)
    col1 = min(top_rect.c1, bottom_rect.c1)
    if row0 > row1 or col0 > col1:
        return False
    for out_r in range(row0, row1 + 1):
        for out_c in range(col0, col1 + 1):
            if top_rect.local_color_at(out_r, out_c) != bottom_rect.local_color_at(out_r, out_c):
                return True
    return False


def _choose_overlap_pair(height: int, width: int, rng: np.random.Generator) -> list[Rect] | None:
    pair_choices = list(PAIR_AREA_CHOICES)
    rng.shuffle(pair_choices)
    for small_area, large_area in pair_choices:
        small_dims = list(AREA_DIMS[small_area])
        large_dims = list(AREA_DIMS[large_area])
        rng.shuffle(small_dims)
        rng.shuffle(large_dims)
        for small_h, small_w in small_dims:
            for large_h, large_w in large_dims:
                candidates: list[tuple[Rect, Rect]] = []
                for small_r0 in range(TOP_BAND_LAST_ROW - small_h + 2):
                    for large_r0 in range(TOP_BAND_LAST_ROW - large_h + 2):
                        # Pair columns will overlap, so input rows need a blank row gap.
                        if not (
                            small_r0 + small_h < large_r0 or large_r0 + large_h < small_r0
                        ):
                            continue
                        for small_c0 in range(width - small_w + 1):
                            for large_c0 in range(width - large_w + 1):
                                col_overlap = _overlap_len(
                                    small_c0,
                                    small_c0 + small_w - 1,
                                    large_c0,
                                    large_c0 + large_w - 1,
                                )
                                if not (1 <= col_overlap <= 2):
                                    continue
                                small = Rect(
                                    r0=small_r0,
                                    c0=small_c0,
                                    h=small_h,
                                    w=small_w,
                                    area=small_area,
                                )
                                large = Rect(
                                    r0=large_r0,
                                    c0=large_c0,
                                    h=large_h,
                                    w=large_w,
                                    area=large_area,
                                )
                                if small.is_partially_clipped(height) or large.is_partially_clipped(height):
                                    continue
                                row_overlap = _overlap_len(
                                    small.final_top,
                                    min(small.final_bottom, height - 1),
                                    large.final_top,
                                    min(large.final_bottom, height - 1),
                                )
                                if not (1 <= row_overlap <= 3):
                                    continue
                                if not _distinct_overlap_colors(small, large, height):
                                    continue
                                candidates.append((small, large))
                if candidates:
                    picked = candidates[int(rng.integers(len(candidates)))]
                    return [picked[0], picked[1]]
    return None


def _extra_candidates(
    height: int,
    width: int,
    existing: list[Rect],
    area: int,
    dims: tuple[int, int],
    require_clip: bool,
) -> list[Rect]:
    h, w = dims
    base_overlap = int(np.count_nonzero(_output_cover_count(height, width, existing) > 1))
    candidates: list[Rect] = []

    for r0 in range(TOP_BAND_LAST_ROW - h + 2):
        for c0 in range(width - w + 1):
            rect = Rect(r0=r0, c0=c0, h=h, w=w, area=area)
            if not all(_rectangles_separated(rect, other) for other in existing):
                continue
            if rect.is_fully_clipped(height):
                continue
            if require_clip:
                if not rect.is_partially_clipped(height):
                    continue
                if rect.visible_height(height) < 2 or rect.clipped_rows(height) < 1:
                    continue
            elif rect.is_partially_clipped(height):
                continue

            cover = _output_cover_count(height, width, existing + [rect])
            overlap_count = int(np.count_nonzero(cover > 1))
            added_overlap = overlap_count - base_overlap
            if require_clip:
                if added_overlap != 0:
                    continue
            elif added_overlap > 1:
                continue
            if np.max(cover) > 2:
                continue
            candidates.append(rect)

    return candidates


def _validate_layout(height: int, width: int, rects: list[Rect], want_clip: bool) -> bool:
    areas = [rect.area for rect in rects]
    if len(rects) < 2 or len(rects) > 4 or len(set(areas)) != len(areas):
        return False
    if any(rect.r1 > TOP_BAND_LAST_ROW for rect in rects):
        return False
    if any(rect.c0 < 0 or rect.c1 >= width for rect in rects):
        return False
    if any(not all(_rectangles_separated(rect, other) for other in rects[i + 1 :]) for i, rect in enumerate(rects)):
        return False
    if not any(rect.h >= 3 and rect.w >= 3 for rect in rects):
        return False

    clipped = [rect for rect in rects if rect.is_partially_clipped(height)]
    if want_clip:
        if len(clipped) != 1:
            return False
    elif clipped:
        return False

    cover = _output_cover_count(height, width, rects)
    overlap_count = int(np.count_nonzero(cover > 1))
    if not (1 <= overlap_count <= 4):
        return False
    if np.max(cover) > 2:
        return False

    sorted_rects = sorted(rects, key=lambda item: item.area)
    has_revealing_overlap = False
    for i, small in enumerate(sorted_rects):
        for large in sorted_rects[i + 1 :]:
            if _distinct_overlap_colors(small, large, height):
                row0 = max(small.final_top, large.final_top)
                row1 = min(min(small.final_bottom, height - 1), min(large.final_bottom, height - 1))
                col0 = max(small.c0, large.c0)
                col1 = min(small.c1, large.c1)
                if row0 <= row1 and col0 <= col1:
                    has_revealing_overlap = True
                    break
        if has_revealing_overlap:
            break
    if not has_revealing_overlap:
        return False

    grid = _render_input(height, width, rects)
    found = _find_gray_rectangles(grid)
    if sorted((rect.area, rect.h, rect.w) for rect in found) != sorted(
        (rect.area, rect.h, rect.w) for rect in rects
    ):
        return False

    out = main(grid)
    if out.shape != (height, width):
        return False
    if not (np.any(out == BLUE) and np.any(out == YELLOW) and np.any(out == RED)):
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    want_clip = bool(rng.random() < 0.45)
    height = int(rng.integers(18, 22 if want_clip else 25))
    width = int(rng.integers(19, 27))
    rects = _choose_overlap_pair(height, width, rng)
    if rects is None:
        return None

    if want_clip:
        count = int(rng.choice(np.array([3, 4], dtype=int)))
        extra_areas = [10]
        remaining = [area for area in AREA_DIMS if area not in {rect.area for rect in rects} | {10}]
        if count == 4:
            extra_areas.append(int(rng.choice(np.array(remaining, dtype=int))))
        extra_specs = []
        for idx, area in enumerate(extra_areas):
            if idx == 0:
                extra_specs.append((area, (5, 2), True))
            else:
                dims = AREA_DIMS[area][int(rng.integers(len(AREA_DIMS[area])))]
                extra_specs.append((area, dims, False))
    else:
        count = int(rng.choice(np.array([2, 3, 3, 4], dtype=int)))
        remaining = [area for area in AREA_DIMS if area not in {rect.area for rect in rects}]
        rng.shuffle(remaining)
        extra_specs = []
        for area in remaining[: count - 2]:
            dims = AREA_DIMS[area][int(rng.integers(len(AREA_DIMS[area])))]
            extra_specs.append((area, dims, False))

    for area, dims, require_clip in extra_specs:
        candidates = _extra_candidates(height, width, rects, area, dims, require_clip)
        if not candidates:
            return None
        rects.append(candidates[int(rng.integers(len(candidates)))])

    if not _validate_layout(height, width, rects, want_clip):
        return None
    return _render_input(height, width, rects)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
