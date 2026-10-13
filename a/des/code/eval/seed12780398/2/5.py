from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
FRAME_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_LAYERS = 4
GRID_MAX = 30
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _grow_bbox(bbox: tuple[int, int, int, int], layers: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 - layers, c0 - layers, r1 + layers, c1 + layers)


def _boxes_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _ring_cells(
    bbox: tuple[int, int, int, int],
    layer: int,
    shape: tuple[int, int] | None = None,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rr0, cc0, rr1, cc1 = r0 - layer, c0 - layer, r1 + layer, c1 + layer
    cells: set[tuple[int, int]] = set()
    for c in range(cc0, cc1 + 1):
        cells.add((rr0, c))
        cells.add((rr1, c))
    for r in range(rr0 + 1, rr1):
        cells.add((r, cc0))
        cells.add((r, cc1))
    if shape is None:
        return cells
    h, w = shape
    return {(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w}


def _paint_ring(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    layer: int,
    color: int,
) -> None:
    for r, c in _ring_cells(bbox, layer, out.shape):
        if int(out[r, c]) == BG:
            out[r, c] = color


def _find_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    colors = [int(v) for v in np.unique(g) if int(v) not in (BG, YELLOW)]
    for color in colors:
        for pts in _components(g == color):
            rows = [r for r, _ in pts]
            cols = [c for _, c in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            if len(pts) != 2 * height + 2 * width - 4:
                continue

            point_set = set(pts)
            ok = True
            for c in range(c0, c1 + 1):
                if (r0, c) not in point_set or (r1, c) not in point_set:
                    ok = False
                    break
            if not ok:
                continue
            for r in range(r0, r1 + 1):
                if (r, c0) not in point_set or (r, c1) not in point_set:
                    ok = False
                    break
            if not ok:
                continue
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
                continue

            seeds = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == YELLOW))
            frames.append({"bbox": (r0, c0, r1, c1), "color": color, "layers": seeds})

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    out[out == YELLOW] = BG

    for frame in _find_frames(g):
        bbox = frame["bbox"]
        base_color = int(frame["color"])
        layers = int(frame["layers"])
        for layer in range(1, layers + 1):
            color = GREY if layer % 2 == 1 else base_color
            _paint_ring(out, bbox, layer, color)

    return out


def _frame_expansion_cells(
    bbox: tuple[int, int, int, int],
    layers: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for layer in range(1, layers + 1):
        cells |= _ring_cells(bbox, layer, shape)
    return cells


def _sample_sizes(rng: random.Random, num_frames: int) -> tuple[int, int]:
    if num_frames == 2:
        return rng.randint(4, 7), rng.randint(4, 8)
    if num_frames == 3:
        return rng.randint(4, 6), rng.randint(4, 7)
    return rng.randint(4, 5), rng.randint(4, 6)


def _place_non_overlapping(
    rng: random.Random,
    frame_h: int,
    frame_w: int,
    counts: list[int],
) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    num_frames = len(counts)
    if num_frames == 2:
        rows, cols = ((1, 2) if rng.random() < 0.5 else (2, 1))
        cells = [(r, c) for r in range(rows) for c in range(cols)]
    else:
        rows, cols = 2, 2
        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        cells = rng.sample(all_cells, num_frames)

    assignment = list(zip(range(num_frames), cells))
    rng.shuffle(assignment)

    row_heights = [0] * rows
    col_widths = [0] * cols
    for idx, (row, col) in assignment:
        span_h = frame_h + 2 * counts[idx]
        span_w = frame_w + 2 * counts[idx]
        row_heights[row] = max(row_heights[row], span_h)
        col_widths[col] = max(col_widths[col], span_w)

    row_gap = rng.randint(1, 3) if rows > 1 else 0
    col_gap = rng.randint(1, 3) if cols > 1 else 0
    top_margin = rng.randint(0, 3)
    left_margin = rng.randint(0, 3)
    bottom_margin = rng.randint(0, 3)
    right_margin = rng.randint(0, 3)

    height = top_margin + sum(row_heights) + row_gap * (rows - 1) + bottom_margin
    width = left_margin + sum(col_widths) + col_gap * (cols - 1) + right_margin
    if height > GRID_MAX or width > GRID_MAX:
        return None

    row_tops = [top_margin]
    for i in range(1, rows):
        row_tops.append(row_tops[-1] + row_heights[i - 1] + row_gap)
    col_lefts = [left_margin]
    for i in range(1, cols):
        col_lefts.append(col_lefts[-1] + col_widths[i - 1] + col_gap)

    bboxes: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * num_frames
    for idx, (row, col) in assignment:
        count = counts[idx]
        span_h = frame_h + 2 * count
        span_w = frame_w + 2 * count
        span_r = row_tops[row] + rng.randint(0, row_heights[row] - span_h)
        span_c = col_lefts[col] + rng.randint(0, col_widths[col] - span_w)
        frame_r = span_r + count
        frame_c = span_c + count
        bboxes[idx] = (frame_r, frame_c, frame_r + frame_h - 1, frame_c + frame_w - 1)

    return height, width, bboxes


def _place_with_overlap(
    rng: random.Random,
    frame_h: int,
    frame_w: int,
    counts: list[int],
) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    num_frames = len(counts)
    orientation = rng.choice(("horizontal", "vertical"))
    first_layers, second_layers = counts[0], counts[1]
    gap = rng.randint(1, first_layers + second_layers - 1)

    if orientation == "horizontal":
        pair_h = frame_h + 2 * max(first_layers, second_layers)
        pair_w = 2 * frame_w + first_layers + second_layers + gap
    else:
        pair_h = 2 * frame_h + first_layers + second_layers + gap
        pair_w = frame_w + 2 * max(first_layers, second_layers)

    min_h = max(pair_h, 18)
    min_w = max(pair_w, 18)
    if num_frames > 2:
        min_h = max(min_h, 24)
        min_w = max(min_w, 24)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    height = rng.randint(min_h, GRID_MAX)
    width = rng.randint(min_w, GRID_MAX)
    bboxes: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * num_frames

    if orientation == "horizontal":
        row = rng.randint(max(first_layers, second_layers), height - frame_h - max(first_layers, second_layers))
        left_min = first_layers
        left_max = width - (2 * frame_w + gap + second_layers)
        if left_min > left_max:
            return None
        left = rng.randint(left_min, left_max)
        bboxes[0] = (row, left, row + frame_h - 1, left + frame_w - 1)
        second_left = left + frame_w + gap
        bboxes[1] = (row, second_left, row + frame_h - 1, second_left + frame_w - 1)
    else:
        col = rng.randint(max(first_layers, second_layers), width - frame_w - max(first_layers, second_layers))
        top_min = first_layers
        top_max = height - (2 * frame_h + gap + second_layers)
        if top_min > top_max:
            return None
        top = rng.randint(top_min, top_max)
        bboxes[0] = (top, col, top + frame_h - 1, col + frame_w - 1)
        second_top = top + frame_h + gap
        bboxes[1] = (second_top, col, second_top + frame_h - 1, col + frame_w - 1)

    expanded = [_grow_bbox(bboxes[0], counts[0]), _grow_bbox(bboxes[1], counts[1])]

    for idx in range(2, num_frames):
        layers = counts[idx]
        placed = False
        for _ in range(200):
            r = rng.randint(layers, height - frame_h - layers)
            c = rng.randint(layers, width - frame_w - layers)
            bbox = (r, c, r + frame_h - 1, c + frame_w - 1)
            if any(_boxes_intersect(bbox, other, pad=1) for other in bboxes[:idx]):
                continue
            grown = _grow_bbox(bbox, layers)
            if any(_boxes_intersect(grown, other, pad=1) for other in expanded):
                continue
            bboxes[idx] = bbox
            expanded.append(grown)
            placed = True
            break
        if not placed:
            return None

    return height, width, bboxes


def _draw_input(
    height: int,
    width: int,
    frame_h: int,
    frame_w: int,
    colors: list[int],
    counts: list[int],
    bboxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for color, count, bbox in zip(colors, counts, bboxes):
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        for r, c in rng.sample(interior, count):
            grid[r, c] = YELLOW
    return grid


def _validate_generated(
    grid: np.ndarray,
    counts: list[int],
    overlap_mode: bool,
) -> bool:
    frames = _find_frames(grid)
    if len(frames) != len(counts):
        return False
    if Counter(int(frame["layers"]) for frame in frames) != Counter(counts):
        return False

    expansion_sets = [
        _frame_expansion_cells(frame["bbox"], int(frame["layers"]), grid.shape) for frame in frames
    ]
    has_overlap = any(expansion_sets[i] & expansion_sets[j] for i in range(len(frames)) for j in range(i + 1, len(frames)))
    if overlap_mode != has_overlap:
        return False

    out = main(grid)
    if np.any(out == YELLOW):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_frames = rng.choices((2, 3, 4), weights=(3, 3, 2), k=1)[0]
    frame_h, frame_w = _sample_sizes(rng, num_frames)
    counts = [rng.randint(1, MAX_LAYERS) for _ in range(num_frames)]
    colors = rng.sample(FRAME_COLORS, num_frames)
    overlap_mode = bool(rng.getrandbits(1))

    if overlap_mode:
        placed = _place_with_overlap(rng, frame_h, frame_w, counts)
    else:
        placed = _place_non_overlapping(rng, frame_h, frame_w, counts)
    if placed is None:
        return None

    height, width, bboxes = placed
    grid = _draw_input(height, width, frame_h, frame_w, colors, counts, bboxes, rng)
    if not _validate_generated(grid, counts, overlap_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
