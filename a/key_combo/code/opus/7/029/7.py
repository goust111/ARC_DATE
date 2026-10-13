from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
FRAME_COLORS = (1, 2, 3, 4, 6)


@dataclass(frozen=True)
class Frame:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def h(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def w(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def interior_area(self) -> int:
        return (self.h - 2) * (self.w - 2)

    @property
    def fill_color(self) -> int:
        return GREY if self.interior_area == 1 else ORANGE

    @property
    def is_active(self) -> bool:
        return self.interior_area > 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _frame_from_component(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> Frame:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    color = int(grid[cells[0]])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3, "frame must enclose a nonempty interior"
    assert len(cells) == 2 * (h + w) - 4, "component must be a one-cell-thick rectangle border"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.all(patch[border] == color), "frame border must be monochrome"
    assert np.all(patch[~border] == bg), "frame interior must be background-colored"

    return Frame(color=color, r0=r0, c0=c0, r1=r1, c1=c1)


def _paint_frame(out: np.ndarray, frame: Frame) -> None:
    out[frame.r0 : frame.r1 + 1, frame.c0 : frame.c1 + 1] = frame.fill_color
    out[frame.r0, frame.c0 : frame.c1 + 1] = frame.color
    out[frame.r1, frame.c0 : frame.c1 + 1] = frame.color
    out[frame.r0 : frame.r1 + 1, frame.c0] = frame.color
    out[frame.r0 : frame.r1 + 1, frame.c1] = frame.color


def _grow_frame(out: np.ndarray, frame: Frame) -> None:
    h, w = out.shape
    top = frame.r0 - 1
    bottom = frame.r1 + 1
    left = frame.c0 - 1
    right = frame.c1 + 1

    if top >= 0:
        out[top, max(0, left) : min(w, right + 1)] = frame.color
    if bottom < h:
        out[bottom, max(0, left) : min(w, right + 1)] = frame.color
    if left >= 0:
        out[max(0, top) : min(h, bottom + 1), left] = frame.color
    if right < w:
        out[max(0, top) : min(h, bottom + 1), right] = frame.color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow rectangle by interior area, then edge-grow only the larger ones.

    Input structure:
    - The background is the dominant color and is typically black.
    - Foreground objects are separated monochrome one-cell-thick rectangular frames.
    - Each frame encloses only background cells, with some frames having a 1-cell interior and others a larger interior.

    Rule:
    - Detect every hollow rectangular frame and sort them by original top-left corner in row-major order.
    - Repaint a frame's interior grey when its enclosed area is 1, otherwise repaint the whole interior orange.
    - For orange-filled frames only, copy the frame border outward by one cell on all four sides, clipped to the grid, with later frames overwriting earlier overlaps.

    Key insight:
    - The same geometric test, interior area equals 1 versus greater than 1, controls both the fill color and whether the frame performs the outward border growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    components = _find_components(g, bg)
    frames = sorted((_frame_from_component(g, comp, bg) for comp in components), key=lambda f: (f.r0, f.c0))
    assert frames, "expected at least one hollow rectangular frame"

    out = g.copy()
    for frame in frames:
        _paint_frame(out, frame)
        if frame.is_active:
            _grow_frame(out, frame)
    return out


GRID_ROWS_RANGE = (18, 24)
GRID_COLS_RANGE = (18, 24)
FRAME_COUNT_RANGE = (4, 5)
SMALL_SIZE = (3, 3)
LARGE_SIZES = (
    (3, 4),
    (4, 3),
    (4, 4),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
    (5, 5),
)
MARGIN = 1
NORMAL_GAP = 2
CONFLICT_GAP = 1
MAX_ATTEMPTS = 500
CONFLICT_PROBABILITY = 0.55


@dataclass(frozen=True)
class FrameSpec:
    color: int
    h: int
    w: int


def _boxes_clear(
    box: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    gap: int,
) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _sample_specs() -> list[FrameSpec]:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    small_count = random.randint(1, min(2, frame_count - 1))
    large_count = frame_count - small_count
    colors = random.sample(FRAME_COLORS, frame_count)

    specs = [FrameSpec(color=colors.pop(), h=SMALL_SIZE[0], w=SMALL_SIZE[1]) for _ in range(small_count)]
    for _ in range(large_count):
        h, w = random.choice(LARGE_SIZES)
        specs.append(FrameSpec(color=colors.pop(), h=h, w=w))
    random.shuffle(specs)
    return specs


def _place_conflict_pair(
    rows: int,
    cols: int,
    first: FrameSpec,
    second: FrameSpec,
    placed: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, FrameSpec]], list[tuple[int, int]], int] | None:
    for _ in range(80):
        horizontal = random.choice((True, False))
        if horizontal:
            max_h = max(first.h, second.h)
            if cols < first.w + second.w + CONFLICT_GAP + 2 * MARGIN or rows < max_h + 2 * MARGIN:
                continue
            r0 = random.randint(MARGIN, rows - MARGIN - max_h)
            c0 = random.randint(MARGIN, cols - MARGIN - (first.w + second.w + CONFLICT_GAP))
            left_box = (r0, c0, r0 + first.h - 1, c0 + first.w - 1)
            right_c0 = c0 + first.w + CONFLICT_GAP
            right_box = (r0, right_c0, r0 + second.h - 1, right_c0 + second.w - 1)
            if not _boxes_clear(left_box, placed, NORMAL_GAP) or not _boxes_clear(right_box, placed, NORMAL_GAP):
                continue
            overlap_col = c0 + first.w
            overlap_rows = range(r0 - 1, min(r0 + first.h, r0 + second.h) + 1)
            overlap_cells = [(rr, overlap_col) for rr in overlap_rows if 0 <= rr < rows]
            return [(r0, c0, first), (r0, right_c0, second)], overlap_cells, second.color

        max_w = max(first.w, second.w)
        if rows < first.h + second.h + CONFLICT_GAP + 2 * MARGIN or cols < max_w + 2 * MARGIN:
            continue
        c0 = random.randint(MARGIN, cols - MARGIN - max_w)
        r0 = random.randint(MARGIN, rows - MARGIN - (first.h + second.h + CONFLICT_GAP))
        top_box = (r0, c0, r0 + first.h - 1, c0 + first.w - 1)
        bottom_r0 = r0 + first.h + CONFLICT_GAP
        bottom_box = (bottom_r0, c0, bottom_r0 + second.h - 1, c0 + second.w - 1)
        if not _boxes_clear(top_box, placed, NORMAL_GAP) or not _boxes_clear(bottom_box, placed, NORMAL_GAP):
            continue
        overlap_row = r0 + first.h
        overlap_cols = range(c0 - 1, min(c0 + first.w, c0 + second.w) + 1)
        overlap_cells = [(overlap_row, cc) for cc in overlap_cols if 0 <= cc < cols]
        return [(r0, c0, first), (bottom_r0, c0, second)], overlap_cells, second.color
    return None


def _place_remaining(
    rows: int,
    cols: int,
    specs: list[FrameSpec],
    placed_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, FrameSpec]] | None:
    placements: list[tuple[int, int, FrameSpec]] = []
    for spec in specs:
        candidates = [
            (r0, c0)
            for r0 in range(MARGIN, rows - MARGIN - spec.h + 1)
            for c0 in range(MARGIN, cols - MARGIN - spec.w + 1)
        ]
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + spec.h - 1, c0 + spec.w - 1)
            if _boxes_clear(box, placed_boxes, NORMAL_GAP):
                chosen = (r0, c0)
                placed_boxes.append(box)
                placements.append((r0, c0, spec))
                break
        if chosen is None:
            return None
    return placements


def _draw_input(rows: int, cols: int, placements: list[tuple[int, int, FrameSpec]]) -> np.ndarray:
    grid = np.full((rows, cols), BLACK, dtype=int)
    for r0, c0, spec in placements:
        r1 = r0 + spec.h - 1
        c1 = c0 + spec.w - 1
        grid[r0, c0 : c1 + 1] = spec.color
        grid[r1, c0 : c1 + 1] = spec.color
        grid[r0 : r1 + 1, c0] = spec.color
        grid[r0 : r1 + 1, c1] = spec.color
    return grid


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    specs = _sample_specs()
    large_specs = [spec for spec in specs if (spec.h, spec.w) != SMALL_SIZE]

    placements: list[tuple[int, int, FrameSpec]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    conflict_cells: list[tuple[int, int]] = []
    conflict_winner = None

    use_conflict = len(large_specs) >= 2 and random.random() < CONFLICT_PROBABILITY
    if use_conflict:
        first, second = large_specs[:2]
        result = _place_conflict_pair(rows, cols, first, second, placed_boxes)
        if result is None:
            return None
        pair_placements, conflict_cells, conflict_winner = result
        placements.extend(pair_placements)
        for r0, c0, spec in pair_placements:
            placed_boxes.append((r0, c0, r0 + spec.h - 1, c0 + spec.w - 1))
        used_specs = {id(first), id(second)}
        remaining_specs = [spec for spec in specs if id(spec) not in used_specs]
    else:
        remaining_specs = specs[:]

    extra = _place_remaining(rows, cols, remaining_specs, placed_boxes)
    if extra is None:
        return None
    placements.extend(extra)

    grid = _draw_input(rows, cols, placements)
    out = main(grid)

    if not np.any(out == GREY) or not np.any(out == ORANGE):
        return None
    if np.any(grid != BLACK) and int(np.count_nonzero(grid == BLACK)) <= int(np.count_nonzero(grid != BLACK)):
        return None
    if use_conflict:
        in_bounds_cells = [(r, c) for r, c in conflict_cells if 0 <= r < rows and 0 <= c < cols]
        if not in_bounds_cells:
            return None
        if not any(int(grid[r, c]) == BLACK and int(out[r, c]) == conflict_winner for r, c in in_bounds_cells):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
