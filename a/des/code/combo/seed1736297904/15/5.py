from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_rect_border(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected_count = 2 * (h + w) - 4
    if len(points) != expected_count:
        return False
    point_set = set(points)
    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return False
    return True


def _find_palette_row(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int]]] = []
    for row in (0, h - 1):
        cols = [c for c in range(w) if int(grid[row, c]) != bg]
        if len(cols) < 2:
            continue
        colors = [int(grid[row, c]) for c in cols]
        if len(colors) != len(set(colors)):
            continue
        if h >= 2:
            adj = 1 if row == 0 else h - 2
            if any(int(grid[adj, c]) != bg for c in cols):
                continue
        candidates.append((row, colors))
    assert candidates, "expected a palette row on the top or bottom border"
    return max(candidates, key=lambda item: len(item[1]))


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    by_color: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in colors:
        rects: list[tuple[int, int, int, int]] = []
        for pts in _components(grid == color):
            if _is_rect_border(pts):
                rects.append(_bbox(pts))
        if rects:
            by_color[color] = rects
    assert by_color, "expected at least one rectangular frame"
    frame_color = max(
        by_color,
        key=lambda color: (
            len(by_color[color]),
            sum(_bbox_area(bbox) for bbox in by_color[color]),
        ),
    )
    return sorted(by_color[frame_color])


def _paint_layers(out: np.ndarray, bbox: tuple[int, int, int, int], palette: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            depth = min(r - (r0 + 1), (r1 - 1) - r, c - (c0 + 1), (c1 - 1) - c)
            out[r, c] = int(palette[min(depth, len(palette) - 1)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior with nested palette layers read from a palette row.

    The palette row sits on the top or bottom border. Reading its non-background
    colors from left to right yields the outer-to-inner color order.
    Each detected one-pixel-thick rectangular frame keeps its border unchanged,
    and its interior is completed by rectangular rings whose depth from the frame
    border determines the palette color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    _palette_row, palette = _find_palette_row(g, bg)
    frames = _find_frames(g, bg)

    out = g.copy()
    for bbox in frames:
        _paint_layers(out, bbox, palette)
    return out


FRAME_COLOR = 5
GRID_HEIGHT_RANGE = (14, 26)
GRID_WIDTH_RANGE = (16, 32)
PALETTE_LEN_RANGE = (2, 4)
FRAME_COUNT_BY_PALETTE = {2: (1, 4), 3: (1, 3), 4: (1, 3)}
MAX_GENERATION_ATTEMPTS = 400


def _capacity_from_dims(height: int, width: int) -> int:
    return (min(height, width) + 1) // 2


def _full_interior(height: int, width: int, palette: list[int]) -> tuple[np.ndarray, np.ndarray]:
    full = np.zeros((height, width), dtype=int)
    depth_grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            depth = min(r, height - 1 - r, c, width - 1 - c)
            depth_grid[r, c] = depth
            full[r, c] = int(palette[min(depth, len(palette) - 1)])
    return full, depth_grid


def _sample_interior_size(palette_len: int, mode: str) -> tuple[int, int]:
    if mode == "small":
        target_capacity = random.randint(1, max(1, palette_len - 1))
    else:
        target_capacity = random.randint(palette_len, palette_len + 2)
    min_dim = random.choice([2 * target_capacity - 1, 2 * target_capacity])
    other_dim = random.randint(min_dim, min_dim + 4)
    if random.random() < 0.5:
        return min_dim, other_dim
    return other_dim, min_dim


def _sample_clues(full: np.ndarray, depth_grid: np.ndarray, bg: int) -> np.ndarray:
    clue = np.full(full.shape, int(bg), dtype=int)
    max_depth = int(depth_grid.max())
    density = random.uniform(0.2, 0.45)
    for depth in range(max_depth + 1):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(depth_grid == depth)]
        if not coords:
            continue
        if len(coords) == 1:
            chosen = coords
        else:
            max_keep = min(len(coords) - 1, max(1, int(np.ceil(len(coords) * density))))
            keep = random.randint(1, max_keep)
            chosen = random.sample(coords, keep)
        for r, c in chosen:
            clue[r, c] = int(full[r, c])
    return clue


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_frames(
    rows: int,
    cols: int,
    palette_side: str,
    frame_shapes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    if palette_side == "top":
        min_r, max_r = 2, rows - 2
    else:
        min_r, max_r = 1, rows - 3
    min_c, max_c = 1, cols - 2

    order = sorted(range(len(frame_shapes)), key=lambda i: frame_shapes[i][0] * frame_shapes[i][1], reverse=True)
    placed: list[tuple[int, tuple[int, int, int, int]]] = []

    for idx in order:
        box_h, box_w = frame_shapes[idx]
        positions: list[tuple[int, int, int, int]] = []
        for r0 in range(min_r, max_r - box_h + 2):
            for c0 in range(min_c, max_c - box_w + 2):
                bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
                if all(not _boxes_too_close(bbox, prev_bbox) for _prev_idx, prev_bbox in placed):
                    positions.append(bbox)
        if not positions:
            return None
        placed.append((idx, random.choice(positions)))

    out: list[tuple[int, int, int, int] | None] = [None] * len(frame_shapes)
    for idx, bbox in placed:
        out[idx] = bbox
    return [bbox for bbox in out if bbox is not None]


def _palette_columns(width: int, palette_len: int) -> list[int]:
    cols = random.sample(list(range(1, width - 1)), palette_len)
    cols.sort()
    return cols


def _draw_frame_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(*GRID_HEIGHT_RANGE)
    cols = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice([c for c in range(10) if c != FRAME_COLOR])
    available_colors = [c for c in range(10) if c not in {bg, FRAME_COLOR}]

    palette_len = random.randint(*PALETTE_LEN_RANGE)
    palette = random.sample(available_colors, palette_len)
    palette_side = random.choice(["top", "bottom"])
    frame_count = random.randint(*FRAME_COUNT_BY_PALETTE[palette_len])

    if frame_count == 1:
        modes = [random.choice(["small", "large"])]
    else:
        first = random.choice(["small", "large"])
        modes = [first, "large" if first == "small" else "small"]
        while len(modes) < frame_count:
            modes.append(random.choice(["small", "large"]))
        random.shuffle(modes)

    interior_shapes = [_sample_interior_size(palette_len, mode) for mode in modes]
    frame_shapes = [(h + 2, w + 2) for h, w in interior_shapes]
    frame_bboxes = _place_frames(rows, cols, palette_side, frame_shapes)
    if frame_bboxes is None:
        return None

    inp = np.full((rows, cols), int(bg), dtype=int)
    out = inp.copy()

    palette_row = 0 if palette_side == "top" else rows - 1
    for col, color in zip(_palette_columns(cols, palette_len), palette):
        inp[palette_row, col] = int(color)
        out[palette_row, col] = int(color)

    has_small = False
    has_large = False
    for bbox, (inner_h, inner_w) in zip(frame_bboxes, interior_shapes):
        r0, c0, r1, c1 = bbox
        _draw_frame_border(inp, bbox, FRAME_COLOR)
        _draw_frame_border(out, bbox, FRAME_COLOR)

        full, depth_grid = _full_interior(inner_h, inner_w, palette)
        clue = _sample_clues(full, depth_grid, bg)
        inp[r0 + 1 : r1, c0 + 1 : c1] = clue
        out[r0 + 1 : r1, c0 + 1 : c1] = full

        capacity = _capacity_from_dims(inner_h, inner_w)
        has_small |= capacity < palette_len
        has_large |= capacity >= palette_len

    if frame_count >= 2 and not (has_small and has_large):
        return None

    vals, cnts = np.unique(inp, return_counts=True)
    if int(vals[int(np.argmax(cnts))]) != bg:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_case()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid input")
