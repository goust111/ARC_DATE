from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# Generator configuration.
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (14, 30)
PAIR_COUNT_RANGE = (2, 4)
FRAME_COUNT_RANGE = (1, 5)
FRAME_H_RANGE = (3, 9)
FRAME_W_RANGE = (3, 11)
FRAME_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
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


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(points) != 2 * (h + w) - 4:
        return None

    sub = np.zeros((h, w), dtype=bool)
    for r, c in points:
        sub[r - r0, c - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(sub, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _parse_palette(g: np.ndarray) -> dict[int, int]:
    row = g[0]
    mapping: dict[int, int] = {}
    c = 0
    while c < row.size:
        if int(row[c]) == BG:
            c += 1
            continue
        start = c
        while c < row.size and int(row[c]) != BG:
            c += 1
        assert c - start == 2, "palette entries must be two adjacent non-black pixels"
        key = int(row[start])
        value = int(row[start + 1])
        assert key != BG and value != BG
        assert key not in mapping, "fragment color must appear exactly once in the palette"
        mapping[key] = value
    assert mapping, "expected a non-empty top-row palette"
    return mapping


def _find_frames(g: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in np.unique(g[1:, :]):
        color = int(color)
        if color == BG:
            continue
        frames: list[tuple[int, int, int, int]] = []
        for pts in _components(g[1:, :] == color):
            global_pts = [(r + 1, c) for r, c in pts]
            bbox = _frame_bbox(global_pts)
            if bbox is not None:
                frames.append(bbox)
        if frames:
            candidates.append((color, frames))

    assert len(candidates) == 1, "expected one border color that forms rectangular frames"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    palette = _parse_palette(g)
    _border_color, frames = _find_frames(g)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        fragment_colors = [int(v) for v in np.unique(interior) if int(v) != BG]
        assert len(fragment_colors) == 1, "each frame must contain exactly one fragment color"
        fragment = fragment_colors[0]
        assert fragment in palette, "frame fragment color must exist in the palette"
        out[r0 + 1 : r1, c0 + 1 : c1] = palette[fragment]
    return out


def _palette_span(pair_count: int) -> int:
    return 3 * pair_count - 1


def _choose_palette() -> tuple[int, dict[int, int], list[int]]:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    border_color = random.choice(COLORS)
    remaining = [c for c in COLORS if c != border_color]
    chosen = random.sample(remaining, 2 * pair_count)
    fragment_colors = chosen[:pair_count]
    fill_colors = chosen[pair_count:]
    mapping = {frag: fill for frag, fill in zip(fragment_colors, fill_colors, strict=False)}
    return border_color, mapping, fragment_colors


def _choose_frame_keys(fragment_colors: list[int], frame_count: int) -> list[int]:
    used_count = random.randint(1, min(len(fragment_colors), frame_count))
    used = random.sample(fragment_colors, used_count)
    keys = used[:] + [random.choice(used) for _ in range(frame_count - used_count)]
    random.shuffle(keys)
    return keys


def _size_options(rows: int, cols: int, frame_count: int) -> list[tuple[int, int]]:
    max_h = min(FRAME_H_RANGE[1], rows - 3)
    max_w = min(FRAME_W_RANGE[1], cols - 2)
    area_cap = 54 if frame_count <= 2 else 36 if frame_count <= 4 else 24
    options = [
        (h, w)
        for h in range(FRAME_H_RANGE[0], max_h + 1)
        for w in range(FRAME_W_RANGE[0], max_w + 1)
        if h * w <= area_cap
    ]
    return options


def _place_frames(rows: int, cols: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    occupancy = np.zeros((rows, cols), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for h, w in sorted(sizes, key=lambda item: item[0] * item[1], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(2, rows - h):
            for c0 in range(1, cols - w):
                r1 = r0 + h - 1
                c1 = c0 + w - 1
                pad_r0 = r0 - FRAME_PADDING
                pad_c0 = c0 - FRAME_PADDING
                pad_r1 = r1 + FRAME_PADDING
                pad_c1 = c1 + FRAME_PADDING
                if occupancy[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        occupancy[r0 - FRAME_PADDING : r1 + FRAME_PADDING + 1, c0 - FRAME_PADDING : c1 + FRAME_PADDING + 1] = True
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    return boxes


def _draw_frame(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _add_fragments(g: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    interior_rows = list(range(r0 + 1, r1))
    interior_cols = list(range(c0 + 1, c1))
    cells = [(r, c) for r in interior_rows for c in interior_cols]
    area = len(cells)
    assert area >= 1
    max_fragments = min(area, 1 + area // 4, 5)
    count = random.randint(1, max_fragments)
    for r, c in random.sample(cells, count):
        g[r, c] = color


def _build_palette_row(width: int, mapping: dict[int, int]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    order = list(mapping)
    random.shuffle(order)
    span = _palette_span(len(order))
    start = random.randint(0, width - span)
    c = start
    for i, key in enumerate(order):
        row[c] = key
        row[c + 1] = mapping[key]
        c += 2
        if i + 1 < len(order):
            c += 1
    return row


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    border_color, mapping, fragment_colors = _choose_palette()
    palette_width = _palette_span(len(mapping))
    frame_count = random.randint(*FRAME_COUNT_RANGE)

    min_rows = max(GRID_H_RANGE[0], 8 + frame_count)
    min_cols = max(GRID_W_RANGE[0], palette_width + 1)
    if min_rows > GRID_H_RANGE[1] or min_cols > GRID_W_RANGE[1]:
        return None

    rows = random.randint(min_rows, GRID_H_RANGE[1])
    cols = random.randint(min_cols, GRID_W_RANGE[1])
    size_options = _size_options(rows, cols, frame_count)
    if not size_options:
        return None

    sizes = [random.choice(size_options) for _ in range(frame_count)]
    boxes = _place_frames(rows, cols, sizes)
    if boxes is None:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    grid[0, :] = _build_palette_row(cols, mapping)

    out = grid.copy()
    frame_keys = _choose_frame_keys(fragment_colors, frame_count)
    for bbox, fragment in zip(boxes, frame_keys, strict=False):
        r0, c0, r1, c1 = bbox
        _draw_frame(grid, r0, c0, r1, c1, border_color)
        _draw_frame(out, r0, c0, r1, c1, border_color)
        _add_fragments(grid, bbox, fragment)
        out[r0 + 1 : r1, c0 + 1 : c1] = mapping[fragment]

    if not np.array_equal(main(grid), out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
