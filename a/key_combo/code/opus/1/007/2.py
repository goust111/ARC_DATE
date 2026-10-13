from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


@dataclass
class Rectangle:
    top: int
    left: int
    height: int
    width: int
    base: int
    patch: np.ndarray
    markers: tuple[tuple[int, int, int], ...]

    @property
    def dest_left(self) -> int:
        return self.left + self.width

    @property
    def dest_right(self) -> int:
        return self.dest_left + self.width - 1

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(coords)
    return result


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[Rectangle]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    rects: list[Rectangle] = []

    for comp in _components(g != bg):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        top, left = min(rows), min(cols)
        bottom, right = max(rows), max(cols)
        patch = g[top : bottom + 1, left : right + 1].copy()
        assert np.all(patch != bg), "foreground objects must be filled rectangles"

        counts = Counter(int(v) for v in patch.ravel())
        ranked = counts.most_common()
        assert ranked, "empty foreground object"
        if len(ranked) > 1:
            assert ranked[0][1] > ranked[1][1], "base color must be uniquely dominant"
        base = int(ranked[0][0])

        markers: list[tuple[int, int, int]] = []
        for rr, cc in np.argwhere(patch != base):
            markers.append((int(rr), int(cc), int(patch[int(rr), int(cc)])))

        rects.append(
            Rectangle(
                top=top,
                left=left,
                height=patch.shape[0],
                width=patch.shape[1],
                base=base,
                patch=patch,
                markers=tuple(markers),
            )
        )

    rects.sort(key=lambda rect: (rect.top, rect.left))
    return bg, rects


def _translated_bboxes_overlap(a: Rectangle, b: Rectangle) -> bool:
    rows_overlap = not (a.bottom < b.top or b.bottom < a.top)
    cols_overlap = not (a.dest_right < b.dest_left or b.dest_right < a.dest_left)
    return rows_overlap and cols_overlap


def _render_shifted(shape: tuple[int, int], bg: int, rects: list[Rectangle]) -> np.ndarray:
    if not rects:
        return np.full(shape, bg, dtype=int)

    bases = {rect.base for rect in rects}
    assert len(bases) == 1, "all rectangles must share one base color"
    base = next(iter(bases))
    out = np.full(shape, bg, dtype=int)

    for rect in rects:
        assert rect.dest_right < shape[1], "shifted rectangle must stay inside the grid"
        out[rect.top : rect.top + rect.height, rect.dest_left : rect.dest_left + rect.width] = base

    marker_winners: dict[tuple[int, int], tuple[int, int, int]] = {}
    for rect in rects:
        for local_r, local_c, color in rect.markers:
            rr = rect.top + local_r
            cc = rect.dest_left + local_c
            key = (rr, cc)
            priority = (rect.left, rect.top, color)
            best = marker_winners.get(key)
            if best is None or priority[:2] < best[:2]:
                marker_winners[key] = priority

    for (rr, cc), (_left, _top, color) in marker_winners.items():
        out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shift filled rectangles right by their own widths and overlay the translated marker collisions.

    Input structure:
    - The grid has a black background and 3 to 5 separated filled rectangles.
    - Every rectangle is a solid base-color block containing 1 to 3 non-base marker pixels.
    - The rectangles share one base color, but their widths can differ, so the right-shift amount is object-specific.

    Rule:
    - Find each 4-connected non-background rectangle object and measure its bounding-box width.
    - Clear the canvas to the background and redraw every rectangle shifted right by its own width.
    - Where shifted rectangles overlap, keep the shared base color in ordinary cells and place every translated marker pixel at its landed output cell.
    - If two translated marker pixels land on the same output cell, keep the marker from the rectangle whose original left edge was farther left.

    Key insight:
    - The movement amount comes from each rectangle's own geometry, so collisions are determined by comparing widths with the original horizontal gaps.
    """
    g = np.asarray(grid, dtype=int)
    bg, rects = _extract_rectangles(g)
    assert rects, "expected at least one filled rectangle"
    return _render_shifted(g.shape, bg, rects)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _pair_overlap_kind(left_width: int, gap: int, right_width: int) -> str:
    if left_width >= gap + 2 * right_width:
        return "contained"
    if left_width > gap + right_width:
        return "partial"
    return "none"


def _make_overlap_pair(kind: str) -> tuple[int, int, int]:
    for _ in range(200):
        right_width = random.randint(3, 5)
        gap = random.randint(1, 3)
        if kind == "contained":
            low = gap + 2 * right_width
            high = 10
        else:
            low = gap + right_width + 1
            high = min(10, gap + 2 * right_width - 1)
        if low > high:
            continue
        left_width = random.randint(low, high)
        if _pair_overlap_kind(left_width, gap, right_width) == kind:
            return left_width, right_width, gap
    raise RuntimeError(f"failed to make {kind} overlap pair")


def _sample_markers(
    height: int,
    width: int,
    colors: list[int],
    forbidden: set[tuple[int, int]] | None = None,
    require_outside: set[int] | None = None,
) -> tuple[tuple[int, int, int], ...]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    forbidden = forbidden or set()
    require_outside = require_outside or set()
    count = random.randint(1, min(3, len(cells)))
    chosen: list[tuple[int, int]] = []

    for r, c in cells:
        if (r, c) in forbidden:
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            break

    if len(chosen) < count:
        raise RuntimeError("not enough cells for markers")

    if require_outside:
        if not any(c not in require_outside for _r, c in chosen):
            outside = [cell for cell in cells if cell[1] not in require_outside and cell not in forbidden]
            if not outside:
                raise RuntimeError("could not place an outside marker")
            chosen[-1] = outside[0]

    color_choices = random.sample(colors, len(chosen))
    markers = [(r, c, color) for (r, c), color in zip(chosen, color_choices)]
    return tuple(markers)


def _paint_rectangle(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    base: int,
    markers: tuple[tuple[int, int, int], ...],
) -> None:
    grid[top : top + height, left : left + width] = base
    for r, c, color in markers:
        grid[top + r, left + c] = color


def _build_candidate() -> np.ndarray | None:
    height = random.randint(3, 5)
    rows = random.randint(16, 24)
    cols = random.randint(22, 30)
    count = random.randint(3, 5)
    base = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    marker_palette = [c for c in range(10) if c not in {0, base}]

    overlap_kind = random.choice(["contained", "partial"])
    left_width, right_width, gap = _make_overlap_pair(overlap_kind)

    row1_specs: list[tuple[int, int]] = [(left_width, 0), (right_width, gap)]
    remaining = count - 2

    if remaining >= 1:
        extra_width = random.randint(3, 7)
        extra_gap = random.randint(3, 6)
        row1_specs.append((extra_width, extra_gap))
        remaining -= 1

    row2_specs: list[tuple[int, int]] = []
    if remaining == 1:
        row2_specs.append((random.randint(3, 7), 0))
    elif remaining == 2:
        if random.random() < 0.5:
            row2_specs.append((random.randint(3, 7), 0))
            row2_specs.append((random.randint(3, 6), random.randint(3, 6)))
        else:
            second_kind = random.choice(["contained", "partial"])
            w1, w2, g2 = _make_overlap_pair(second_kind)
            row2_specs.append((w1, 0))
            row2_specs.append((w2, g2))

    row_tops = [random.randint(1, rows - height - 1)]
    if row2_specs:
        min_top = row_tops[0] + height + 2
        if min_top > rows - height - 1:
            return None
        row_tops.append(random.randint(min_top, rows - height - 1))

    def materialize(specs: list[tuple[int, int]], left_margin: int) -> list[tuple[int, int]]:
        placed: list[tuple[int, int]] = []
        cursor = left_margin
        for idx, (width, gap_before) in enumerate(specs):
            if idx == 0:
                left = cursor
            else:
                cursor += gap_before
                left = cursor
            placed.append((left, width))
            cursor = left + width
        return placed

    left_margin = random.randint(1, 3)
    row1 = materialize(row1_specs, left_margin)
    row2 = materialize(row2_specs, left_margin) if row2_specs else []

    rect_specs: list[tuple[int, int, int]] = []
    for top, row_specs in zip(row_tops, [row1, row2] if row2 else [row1]):
        for left, width in row_specs:
            rect_specs.append((top, left, width))

    max_dest = max(left + 2 * width for top, left, width in rect_specs)
    if max_dest > cols - 1:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    rectangles: list[Rectangle] = []

    for idx, (top, left, width) in enumerate(rect_specs):
        patch = np.full((height, width), base, dtype=int)
        rectangles.append(
            Rectangle(
                top=top,
                left=left,
                height=height,
                width=width,
                base=base,
                patch=patch,
                markers=(),
            )
        )

    rect_by_pos = {(rect.top, rect.left): rect for rect in rectangles}
    occupied_targets: set[tuple[int, int]] = set()
    assigned: dict[tuple[int, int], tuple[tuple[int, int, int], ...]] = {}

    row_groups: dict[int, list[Rectangle]] = {}
    for rect in rectangles:
        row_groups.setdefault(rect.top, []).append(rect)
    for rects in row_groups.values():
        rects.sort(key=lambda rect: rect.left)

    for rects in row_groups.values():
        for i, rect in enumerate(rects):
            overlap_cols: set[int] = set()
            for other in rects:
                if other.left == rect.left:
                    continue
                if _translated_bboxes_overlap(rect, other):
                    start = max(rect.dest_left, other.dest_left) - rect.dest_left
                    stop = min(rect.dest_right, other.dest_right) - rect.dest_left
                    overlap_cols.update(range(start, stop + 1))

            require_outside = overlap_cols if overlap_cols and rect.width > len(overlap_cols) else None
            forbidden: set[tuple[int, int]] = set()
            tries = 0
            while True:
                tries += 1
                if tries > 200:
                    return None
                markers = _sample_markers(
                    rect.height,
                    rect.width,
                    marker_palette,
                    forbidden=forbidden,
                    require_outside=require_outside,
                )
                translated = {(rect.top + r, rect.dest_left + c) for r, c, _color in markers}
                if translated & occupied_targets:
                    conflict_cells = translated & occupied_targets
                    forbidden.update((r, c - rect.dest_left) for r, c in conflict_cells if rect.dest_left <= c < rect.dest_left + rect.width)
                    continue
                occupied_targets.update(translated)
                assigned[(rect.top, rect.left)] = markers
                break

    for rect in rectangles:
        markers = assigned[(rect.top, rect.left)]
        _paint_rectangle(grid, rect.top, rect.left, rect.height, rect.width, base, markers)

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        bg, rects = _extract_rectangles(grid)
    except AssertionError:
        return False

    if bg != 0 or not (3 <= len(rects) <= 5):
        return False
    if len({rect.base for rect in rects}) != 1:
        return False
    if len({rect.width for rect in rects}) < 2:
        return False
    if any(not (1 <= len(rect.markers) <= 3) for rect in rects):
        return False

    overlap_pairs = 0
    isolated = 0
    for i, rect in enumerate(rects):
        degree = 0
        for j, other in enumerate(rects):
            if i == j:
                continue
            if _translated_bboxes_overlap(rect, other):
                degree += 1
        overlap_pairs += degree
        if degree == 0:
            isolated += 1

    if overlap_pairs == 0 or isolated == 0:
        return False

    target_cells: set[tuple[int, int]] = set()
    for rect in rects:
        for r, c, _color in rect.markers:
            cell = (rect.top + r, rect.dest_left + c)
            if cell in target_cells:
                return False
            target_cells.add(cell)

    out = _render_shifted(grid.shape, bg, rects)
    if np.array_equal(out, grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        if not _semantic_checks(grid):
            continue
        if not np.array_equal(main(grid), _render_shifted(grid.shape, 0, _extract_rectangles(grid)[1])):
            continue
        return grid
    raise RuntimeError("failed to generate a valid translated-overlay puzzle")
