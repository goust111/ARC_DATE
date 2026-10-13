from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
DEPTH_COLORS = {1: 1, 2: 2, 3: 3, 4: 4}
FRAME_COLOR_POOL = (5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int
    side: str
    gap_cells: tuple[tuple[int, int], ...]
    pixels: tuple[tuple[int, int], ...]

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            cells.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            cells.append((r, c0))
    return cells


def _make_frame(r0: int, c0: int, r1: int, c1: int, side: str, start: int, gap_len: int) -> Frame:
    assert side in SIDE_TO_DIR
    if side == "top":
        gap_cells = tuple((r0, c) for c in range(start, start + gap_len))
    elif side == "bottom":
        gap_cells = tuple((r1, c) for c in range(start, start + gap_len))
    elif side == "left":
        gap_cells = tuple((r, c0) for r in range(start, start + gap_len))
    else:
        gap_cells = tuple((r, c1) for r in range(start, start + gap_len))
    gap_set = set(gap_cells)
    pixels = tuple(cell for cell in _boundary_cells(r0, c0, r1, c1) if cell not in gap_set)
    return Frame(r0=r0, c0=c0, r1=r1, c1=c1, side=side, gap_cells=gap_cells, pixels=pixels)


def _is_consecutive(values: list[int]) -> bool:
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


def _parse_frame_component(points: list[tuple[int, int]]) -> Frame:
    cell_set = set(points)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 5 and width >= 5, "frame must leave room for a 2-cell gap away from corners"

    for r, c in cell_set:
        assert r in (r0, r1) or c in (c0, c1), "frame pixels must stay on the rectangle border"

    missing = {
        "top": [c for c in range(c0 + 1, c1) if (r0, c) not in cell_set],
        "bottom": [c for c in range(c0 + 1, c1) if (r1, c) not in cell_set],
        "left": [r for r in range(r0 + 1, r1) if (r, c0) not in cell_set],
        "right": [r for r in range(r0 + 1, r1) if (r, c1) not in cell_set],
    }
    gap_sides = [side for side, values in missing.items() if values]
    assert len(gap_sides) == 1, "expected exactly one gapped side"
    side = gap_sides[0]
    values = missing[side]
    assert len(values) >= 2 and _is_consecutive(values), "gap must be a continuous run of width >= 2"

    if side in ("top", "bottom"):
        frame = _make_frame(r0, c0, r1, c1, side, values[0], len(values))
    else:
        frame = _make_frame(r0, c0, r1, c1, side, values[0], len(values))

    assert set(frame.pixels) == cell_set, "component must match a single rectangular frame with one gap"
    return frame


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    nonblack = sorted(int(v) for v in np.unique(g) if int(v) != BLACK)
    assert nonblack, "expected at least one frame"
    assert len(nonblack) == 1, "all input frames must share one foreground color"

    frames = [_parse_frame_component(comp) for comp in _components4(g != BLACK)]
    _validate_frame_relations(frames)
    return frames


def _bbox_contains(outer: Frame, inner: Frame) -> bool:
    return outer.r0 < inner.r0 and outer.c0 < inner.c0 and outer.r1 > inner.r1 and outer.c1 > inner.c1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _point_in_bbox(cell: tuple[int, int], frame: Frame) -> bool:
    r, c = cell
    return frame.r0 <= r <= frame.r1 and frame.c0 <= c <= frame.c1


def _validate_frame_relations(frames: list[Frame]) -> None:
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            if _bbox_contains(a, b) or _bbox_contains(b, a):
                continue
            assert not _boxes_overlap((a.r0, a.c0, a.r1, a.c1), (b.r0, b.c0, b.r1, b.c1)), (
                "frames must be disjoint or fully nested"
            )


def _frame_depths(frames: list[Frame]) -> dict[Frame, int]:
    depths: dict[Frame, int] = {}
    for frame in frames:
        depth = 1 + sum(1 for outer in frames if outer is not frame and _bbox_contains(outer, frame))
        assert depth in DEPTH_COLORS, "maximum supported nesting depth is 4"
        depths[frame] = depth
    return depths


def _frame_occupancy(shape: tuple[int, int], frames: list[Frame]) -> np.ndarray:
    occ = np.zeros(shape, dtype=bool)
    for frame in frames:
        for r, c in frame.pixels:
            occ[r, c] = True
    return occ


def _trace_band(frame: Frame, occupied: np.ndarray) -> list[tuple[int, int]]:
    h, w = occupied.shape
    dr, dc = SIDE_TO_DIR[frame.side]
    cells: list[tuple[int, int]] = []
    for r, c in frame.gap_cells:
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w and not occupied[rr, cc]:
            cells.append((rr, cc))
            rr += dr
            cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each frame by nesting depth, then extend a color band outward from its opening.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    depths = _frame_depths(frames)
    occupied = _frame_occupancy(g.shape, frames)

    out = np.zeros_like(g)
    for frame in frames:
        color = DEPTH_COLORS[depths[frame]]
        for r, c in frame.pixels:
            out[r, c] = color

    for frame in frames:
        color = DEPTH_COLORS[depths[frame]]
        for r, c in _trace_band(frame, occupied):
            out[r, c] = color

    return out


GRID_MIN = 18
GRID_MAX = 34
GEN_RETRIES = 400
MIN_FRAME_SIDE = 5
ROOT_SEPARATION = 2
ROOT_EXTRA_BY_COUNT = {1: 8, 2: 6, 3: 4}
ROOT_PLACE_TRIES = 120
GAP_ASSIGN_TRIES = 80
TOTAL_FRAME_CHOICES = (1, 2, 3, 4, 5, 6)
TOTAL_FRAME_WEIGHTS = (1, 2, 4, 4, 3, 2)


def _root_min_side(depth: int) -> int:
    return MIN_FRAME_SIDE + 4 * (depth - 1)


def _sample_depths(rng: random.Random) -> list[int]:
    total_frames = rng.choices(TOTAL_FRAME_CHOICES, weights=TOTAL_FRAME_WEIGHTS, k=1)[0]
    feasible_roots = [r for r in (1, 2, 3) if r <= total_frames <= 4 * r]
    root_weights = [4 - r for r in feasible_roots]
    root_count = rng.choices(feasible_roots, weights=root_weights, k=1)[0]

    depths = [1] * root_count
    remaining = total_frames - root_count
    while remaining > 0:
        candidates = [i for i, depth in enumerate(depths) if depth < 4]
        idx = rng.choice(candidates)
        depths[idx] += 1
        remaining -= 1
    rng.shuffle(depths)
    return depths


def _choose_grid_shape(rng: random.Random, depths: list[int]) -> tuple[int, int] | None:
    root_count = len(depths)
    max_root = max(_root_min_side(depth) for depth in depths)
    span = sum(_root_min_side(depth) for depth in depths) + ROOT_SEPARATION * max(0, root_count - 1) + 2
    long_axis = max(GRID_MIN, span)
    short_axis = max(GRID_MIN, max_root + 4)
    if long_axis > GRID_MAX or short_axis > GRID_MAX:
        return None
    if rng.random() < 0.5:
        h_min, w_min = short_axis, long_axis
    else:
        h_min, w_min = long_axis, short_axis
    return rng.randint(h_min, GRID_MAX), rng.randint(w_min, GRID_MAX)


def _place_root_boxes(
    rng: random.Random, height: int, width: int, depths: list[int]
) -> list[tuple[int, int, int, int]] | None:
    extra = ROOT_EXTRA_BY_COUNT[len(depths)]
    placed: list[tuple[int, tuple[int, int, int, int]]] = []
    order = sorted(range(len(depths)), key=lambda idx: depths[idx], reverse=True)
    for idx in order:
        depth = depths[idx]
        min_side = _root_min_side(depth)
        max_h = min(height - 2, min_side + extra)
        max_w = min(width - 2, min_side + extra)
        if max_h < min_side or max_w < min_side:
            return None

        box = None
        for _ in range(ROOT_PLACE_TRIES):
            box_h = rng.randint(min_side, max_h)
            box_w = rng.randint(min_side, max_w)
            r0 = rng.randint(1, height - box_h - 1)
            c0 = rng.randint(1, width - box_w - 1)
            candidate = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(
                not _boxes_overlap(candidate, other_box, pad=ROOT_SEPARATION)
                for _, other_box in placed
            ):
                box = candidate
                break
        if box is None:
            return None
        placed.append((idx, box))

    by_index = {idx: box for idx, box in placed}
    return [by_index[i] for i in range(len(depths))]


def _build_chain_boxes(
    rng: random.Random, outer_box: tuple[int, int, int, int], depth: int
) -> list[tuple[int, int, int, int]] | None:
    boxes = [outer_box]
    current = outer_box
    for level in range(2, depth + 1):
        remaining_after = depth - level
        min_h = _root_min_side(remaining_after + 1)
        min_w = _root_min_side(remaining_after + 1)

        r0, c0, r1, c1 = current
        cur_h = r1 - r0 + 1
        cur_w = c1 - c0 + 1

        max_top = cur_h - min_h - 2
        max_left = cur_w - min_w - 2
        if max_top < 2 or max_left < 2:
            return None

        top = rng.randint(2, max_top)
        bottom = rng.randint(2, cur_h - min_h - top)
        left = rng.randint(2, max_left)
        right = rng.randint(2, cur_w - min_w - left)

        inner = (r0 + top, c0 + left, r1 - bottom, c1 - right)
        boxes.append(inner)
        current = inner
    return boxes


def _frame_from_box(rng: random.Random, box: tuple[int, int, int, int]) -> Frame:
    r0, c0, r1, c1 = box
    side = rng.choice(SIDES)
    if side in ("top", "bottom"):
        span = c1 - c0 - 1
        gap_len = rng.randint(2, span)
        start = rng.randint(c0 + 1, c1 - gap_len)
    else:
        span = r1 - r0 - 1
        gap_len = rng.randint(2, span)
        start = rng.randint(r0 + 1, r1 - gap_len)
    return _make_frame(r0, c0, r1, c1, side, start, gap_len)


def _bands_are_clear(shape: tuple[int, int], frames: list[Frame]) -> bool:
    occupied = _frame_occupancy(shape, frames)
    seen: set[tuple[int, int]] = set()
    total_extension = 0

    for frame in frames:
        band_cells = _trace_band(frame, occupied)
        total_extension += len(band_cells) - len(frame.gap_cells)
        for cell in band_cells:
            if cell in seen:
                return False
            seen.add(cell)
            for other in frames:
                if other is frame:
                    continue
                if _bbox_contains(other, frame) or _bbox_contains(frame, other):
                    continue
                if _point_in_bbox(cell, other):
                    return False

    return total_extension >= 2 and len(seen) <= (shape[0] * shape[1]) // 3


def _draw_input(shape: tuple[int, int], frames: list[Frame], frame_color: int) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        for r, c in frame.pixels:
            grid[r, c] = frame_color
    return grid


def _is_valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        frames = _extract_frames(grid)
        depths = _frame_depths(frames)
        if max(depths.values()) > 4:
            return False
        if not _bands_are_clear(grid.shape, frames):
            return False
        out = main(grid)
    except AssertionError:
        return False

    return not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    depths = _sample_depths(rng)
    shape = _choose_grid_shape(rng, depths)
    if shape is None:
        return None
    height, width = shape

    root_boxes = _place_root_boxes(rng, height, width, depths)
    if root_boxes is None:
        return None

    chain_boxes: list[list[tuple[int, int, int, int]]] = []
    for root_box, depth in zip(root_boxes, depths):
        boxes = _build_chain_boxes(rng, root_box, depth)
        if boxes is None:
            return None
        chain_boxes.append(boxes)

    for _ in range(GAP_ASSIGN_TRIES):
        frames = [
            _frame_from_box(rng, box)
            for boxes in chain_boxes
            for box in boxes
        ]
        if not _bands_are_clear((height, width), frames):
            continue
        grid = _draw_input((height, width), frames, rng.choice(FRAME_COLOR_POOL))
        if _is_valid_generated_grid(grid):
            return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
