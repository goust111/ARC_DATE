from __future__ import annotations

import random

import numpy as np


BG = 0
NOISE = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
FRAME_COUNT_OPTIONS = (2, 3, 4)
INNER_SIZE_BY_FRAME_COUNT = {
    2: (4, 8),
    3: (4, 7),
    4: (4, 6),
}
GAP_RANGE_BY_FRAME_COUNT = {
    2: (1, 3),
    3: (1, 3),
    4: (1, 2),
}
OUTER_MARGIN_RANGE = (0, 3)
GRID_MAX_SIZE = 26
PATTERN_COLOR_RANGE = (2, 3)
PATTERN_CELL_RANGE = (4, 7)
GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _frame_bbox(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return None
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return None
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return None
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    actual = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        actual[r - r0, c - c0] = True
    if not np.array_equal(actual, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _extract_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components(grid == color):
            bbox = _frame_bbox(grid, coords, color)
            if bbox is not None:
                frames.append(bbox)
    frames.sort(key=lambda b: (-(b[2] - b[0] + 1) * (b[3] - b[1] + 1), b[0], b[1]))
    assert 2 <= len(frames) <= 4, "expected 2 to 4 nested rectangular frames"
    for i in range(len(frames) - 1):
        assert _contains(frames[i], frames[i + 1]), "frames must form a single nested chain"
    return frames


def _interior_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 - 1), int(c1 - c0 - 1)


def _apply_layer_transform(canvas: np.ndarray, depth: int) -> np.ndarray:
    return np.fliplr(canvas) if depth % 2 == 1 else np.flipud(canvas)


def _reconstruct_from_frames(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int]],
    bg: int,
) -> np.ndarray:
    ir0, ic0, ir1, ic1 = frames[-1]
    content = np.asarray(grid[ir0 + 1 : ir1, ic0 + 1 : ic1], dtype=int).copy()
    for idx in range(len(frames) - 1, -1, -1):
        if idx < len(frames) - 1:
            pr0, pc0, pr1, pc1 = frames[idx]
            cr0, cc0, _cr1, _cc1 = frames[idx + 1]
            canvas = np.full(_interior_shape(frames[idx]), int(bg), dtype=int)
            off_r = int(cr0 - pr0)
            off_c = int(cc0 - pc0)
            canvas[off_r : off_r + content.shape[0], off_c : off_c + content.shape[1]] = content
        else:
            canvas = content
        content = _apply_layer_transform(canvas, idx + 1)
    return content


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the nested rectangular frame chain, treat the innermost interior as the
    base raster, and rebuild the content from inner to outer. Each layer applies a
    parity-controlled flip on the whole current interior canvas: odd depths flip
    left-right, even depths flip top-bottom. Output the outermost interior only.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _extract_frames(g, bg)
    return _reconstruct_from_frames(g, frames, bg)


def _grow_connected_cells(h: int, w: int, target: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()
    while len(cells) < target:
        frontier.clear()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _make_pattern(h: int, w: int, colors: list[int]) -> np.ndarray | None:
    max_cells = min(PATTERN_CELL_RANGE[1], h * w - 1)
    min_cells = max(PATTERN_CELL_RANGE[0], len(colors))
    if min_cells > max_cells:
        return None
    for _ in range(80):
        count = random.randint(min_cells, max_cells)
        cells = _grow_connected_cells(h, w, count)
        if cells is None:
            continue
        canvas = np.full((h, w), BG, dtype=int)
        order = list(cells)
        random.shuffle(order)
        for idx, (r, c) in enumerate(order):
            color = colors[idx] if idx < len(colors) else random.choice(colors)
            canvas[r, c] = int(color)
        if np.count_nonzero(canvas == BG) == 0:
            continue
        if np.array_equal(canvas, np.fliplr(canvas)):
            continue
        if np.array_equal(canvas, np.flipud(canvas)):
            continue
        if np.array_equal(canvas, np.flipud(np.fliplr(canvas))):
            continue
        return canvas
    return None


def _sample_geometry() -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]], tuple[int, int, int, int]] | None:
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    inner_range = INNER_SIZE_BY_FRAME_COUNT[frame_count]
    gap_range = GAP_RANGE_BY_FRAME_COUNT[frame_count]
    inner_h = random.randint(*inner_range)
    inner_w = random.randint(*inner_range)

    shapes_inner_to_outer = [(inner_h + 2, inner_w + 2)]
    gaps_inner_to_outer: list[tuple[int, int, int, int]] = []
    cur_h, cur_w = shapes_inner_to_outer[0]
    for _ in range(frame_count - 1):
        top = random.randint(*gap_range)
        bottom = random.randint(*gap_range)
        left = random.randint(*gap_range)
        right = random.randint(*gap_range)
        gaps_inner_to_outer.append((top, bottom, left, right))
        cur_h = top + cur_h + bottom + 2
        cur_w = left + cur_w + right + 2
        shapes_inner_to_outer.append((cur_h, cur_w))

    outer_h, outer_w = shapes_inner_to_outer[-1]
    max_top = min(OUTER_MARGIN_RANGE[1], GRID_MAX_SIZE - outer_h)
    max_left = min(OUTER_MARGIN_RANGE[1], GRID_MAX_SIZE - outer_w)
    if max_top < OUTER_MARGIN_RANGE[0] or max_left < OUTER_MARGIN_RANGE[0]:
        return None
    top_margin = random.randint(OUTER_MARGIN_RANGE[0], max_top)
    left_margin = random.randint(OUTER_MARGIN_RANGE[0], max_left)
    max_bottom = min(OUTER_MARGIN_RANGE[1], GRID_MAX_SIZE - outer_h - top_margin)
    max_right = min(OUTER_MARGIN_RANGE[1], GRID_MAX_SIZE - outer_w - left_margin)
    if max_bottom < OUTER_MARGIN_RANGE[0] or max_right < OUTER_MARGIN_RANGE[0]:
        return None
    bottom_margin = random.randint(OUTER_MARGIN_RANGE[0], max_bottom)
    right_margin = random.randint(OUTER_MARGIN_RANGE[0], max_right)

    grid_h = top_margin + outer_h + bottom_margin
    grid_w = left_margin + outer_w + right_margin
    shapes_outer_to_inner = list(reversed(shapes_inner_to_outer))
    gaps_outer_to_inner = list(reversed(gaps_inner_to_outer))

    bboxes: list[tuple[int, int, int, int]] = []
    r0, c0 = top_margin, left_margin
    for idx, (fh, fw) in enumerate(shapes_outer_to_inner):
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        bboxes.append(bbox)
        if idx < len(gaps_outer_to_inner):
            top, _bottom, left, _right = gaps_outer_to_inner[idx]
            r0 = r0 + 1 + top
            c0 = c0 + 1 + left
    return shapes_outer_to_inner, bboxes, (grid_h, grid_w, frame_count, inner_h * 100 + inner_w)


def _ring_cells(
    parent: tuple[int, int, int, int],
    child: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    pr0, pc0, pr1, pc1 = parent
    cr0, cc0, cr1, cc1 = child
    cells: list[tuple[int, int]] = []
    for r in range(pr0 + 1, pr1):
        for c in range(pc0 + 1, pc1):
            if cr0 <= r <= cr1 and cc0 <= c <= cc1:
                continue
            cells.append((int(r), int(c)))
    return cells


def _oracle(pattern: np.ndarray, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    content = np.asarray(pattern, dtype=int).copy()
    for idx in range(len(frames) - 1, -1, -1):
        if idx < len(frames) - 1:
            parent = frames[idx]
            child = frames[idx + 1]
            canvas = np.full(_interior_shape(parent), BG, dtype=int)
            off_r = child[0] - parent[0]
            off_c = child[1] - parent[1]
            canvas[off_r : off_r + content.shape[0], off_c : off_c + content.shape[1]] = content
        else:
            canvas = content
        content = _apply_layer_transform(canvas, idx + 1)
    return content


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_geometry()
    if sampled is None:
        return None
    _shapes, bboxes, (grid_h, grid_w, frame_count, inner_key) = sampled
    inner_h, inner_w = divmod(inner_key, 100)

    pattern_color_count = random.randint(*PATTERN_COLOR_RANGE)
    needed_colors = frame_count + pattern_color_count
    if needed_colors > len(PALETTE):
        return None
    chosen = random.sample(PALETTE, needed_colors)
    frame_colors = chosen[:frame_count]
    pattern_colors = chosen[frame_count:]

    pattern = _make_pattern(inner_h, inner_w, pattern_colors)
    if pattern is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(frame_colors, bboxes):
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)

    ir0, ic0, ir1, ic1 = bboxes[-1]
    grid[ir0 + 1 : ir1, ic0 + 1 : ic1] = pattern

    for parent, child in zip(bboxes[:-1], bboxes[1:]):
        candidates = _ring_cells(parent, child)
        if not candidates:
            return None
        max_noise = min(4, len(candidates))
        noise_count = random.randint(1, max_noise)
        for r, c in random.sample(candidates, noise_count):
            grid[r, c] = NOISE

    expected = _oracle(pattern, bboxes)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame flip task")
