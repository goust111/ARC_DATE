from __future__ import annotations

import random

import numpy as np


FRAME_COLOR = 5
COLOR_COUNT = 10
GENERATION_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (10, 26)
GRID_WIDTH_RANGE = (12, 30)
PALETTE_LEN_RANGE = (3, 6)
FRAME_COUNT_RANGE = (2, 5)
FRAME_HEIGHT_RANGE = (4, 9)
FRAME_WIDTH_RANGE = (4, 11)
PLACEMENT_PADDING = 1
PLACEMENT_TRIES = 80


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _frame_bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = {
        (r0, c) for c in range(c0, c1 + 1)
    } | {
        (r1, c) for c in range(c0, c1 + 1)
    } | {
        (r, c0) for r in range(r0, r1 + 1)
    } | {
        (r, c1) for r in range(r0, r1 + 1)
    }
    if set(component) != border:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _parse_palette(row: np.ndarray, bg: int) -> list[int]:
    positions = [int(c) for c in np.flatnonzero(row != bg)]
    assert positions, "expected a non-empty palette row"

    colors: list[int] = []
    prev = None
    for c in positions:
        if prev is not None:
            assert c - prev >= 2, "palette chips must be separated by background"
        if c > 0:
            assert int(row[c - 1]) == bg, "palette chips must be isolated"
        if c + 1 < row.size:
            assert int(row[c + 1]) == bg, "palette chips must be isolated"
        colors.append(int(row[c]))
        prev = c

    assert len(colors) == len(set(colors)), "palette colors must be distinct"
    return colors


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    search = grid[:-1, :]
    colors = [int(v) for v in np.unique(search) if int(v) != bg]
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        for comp in _components(search == color):
            bbox = _frame_bbox(comp)
            if bbox is not None:
                frames.append((int(color), bbox))

    assert frames, "expected at least one hollow frame"
    frame_colors = {color for color, _bbox in frames}
    assert len(frame_colors) == 1, "expected all frames to share one border color"
    frames.sort(key=lambda item: item[1])
    return frames


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow frame by counting the interior marker pixels and indexing into the bottom palette row.

    Input structure:
    - One dominant background color fills most of the grid.
    - The bottom row contains a left-to-right palette of isolated single-pixel color chips, with background gaps between adjacent chips.
    - Above the palette are one or more one-pixel-thick hollow rectangular frames sharing the same border color.
    - Each frame interior contains zero or more scattered non-background single pixels of arbitrary colors.

    Rule:
    - Read the palette colors from the bottom row in left-to-right order.
    - For each hollow frame, count the number n of non-background pixels strictly inside the frame.
    - Erase those interior pixels to background, then recolor the whole frame border with the nth palette color when n >= 1.
    - If n == 0, leave that frame border in its original color, and keep the bottom palette row unchanged.

    Key insight:
    The interior pixels are only a counter; their quantity selects an entry from the ordered palette, and the counted pixels themselves disappear in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert g.shape[0] >= 4 and g.shape[1] >= 5, "expected room for frames and a palette row"

    bg = _mode_color(g)
    palette = _parse_palette(g[-1], bg)
    frames = _find_frames(g, bg)

    out = g.copy()
    for frame_color, bbox in frames:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(interior != bg))
        assert count <= len(palette), "frame count must fit inside the palette"
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        target = int(frame_color) if count == 0 else int(palette[count - 1])
        _paint_frame(out, bbox, target)

    out[-1, :] = g[-1, :]
    return out


def _isolated_capacity(inner_h: int, inner_w: int) -> int:
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _sample_frame_size(count: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for h in range(FRAME_HEIGHT_RANGE[0], FRAME_HEIGHT_RANGE[1] + 1):
        for w in range(FRAME_WIDTH_RANGE[0], FRAME_WIDTH_RANGE[1] + 1):
            inner_h = h - 2
            inner_w = w - 2
            if inner_h * inner_w < max(2, count):
                continue
            if _isolated_capacity(inner_h, inner_w) < count:
                continue
            candidates.append((int(h), int(w)))
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_counts(frame_count: int, palette_len: int) -> list[int]:
    mode = random.choice(("mixed", "mixed", "all_zero", "all_positive"))
    if mode == "all_zero":
        return [0] * frame_count
    if mode == "all_positive":
        return [random.randint(1, palette_len) for _ in range(frame_count)]

    counts = [0 if random.random() < 0.5 else random.randint(1, palette_len) for _ in range(frame_count)]
    if all(count == 0 for count in counts):
        counts[random.randrange(frame_count)] = random.randint(1, palette_len)
    if all(count > 0 for count in counts):
        counts[random.randrange(frame_count)] = 0
    return counts


def _place_boxes(height: int, width: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    usable_rows = height - 2
    if usable_rows <= 0:
        return None

    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    occupancy = np.zeros((usable_rows, width), dtype=bool)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)

    for idx in order:
        h, w = sizes[idx]
        if h > usable_rows or w > width:
            return None
        candidates: list[tuple[int, int]] = []
        for r0 in range(usable_rows - h + 1):
            for c0 in range(width - w + 1):
                rr0 = max(0, r0 - PLACEMENT_PADDING)
                rr1 = min(usable_rows, r0 + h + PLACEMENT_PADDING)
                cc0 = max(0, c0 - PLACEMENT_PADDING)
                cc1 = min(width, c0 + w + PLACEMENT_PADDING)
                if not occupancy[rr0:rr1, cc0:cc1].any():
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        rr0 = max(0, r0 - PLACEMENT_PADDING)
        rr1 = min(usable_rows, r0 + h + PLACEMENT_PADDING)
        cc0 = max(0, c0 - PLACEMENT_PADDING)
        cc1 = min(width, c0 + w + PLACEMENT_PADDING)
        occupancy[rr0:rr1, cc0:cc1] = True
        placed[idx] = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))

    return [bbox for bbox in placed if bbox is not None]


def _sample_bboxes(height: int, width: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    for _ in range(PLACEMENT_TRIES):
        bboxes = _place_boxes(height, width, sizes)
        if bboxes is not None:
            return bboxes
    return None


def _palette_row(width: int, bg: int, palette_colors: list[int]) -> np.ndarray:
    row = np.full(width, bg, dtype=int)
    span = 2 * len(palette_colors) - 1
    start = random.randint(0, width - span)
    for i, color in enumerate(palette_colors):
        row[start + 2 * i] = int(color)
    return row


def _sample_isolated_cells(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []

    parity_groups: list[list[tuple[int, int]]] = []
    for rp in (0, 1):
        for cp in (0, 1):
            cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if r % 2 == rp and c % 2 == cp]
            parity_groups.append(cells)

    max_size = max(len(group) for group in parity_groups)
    choices = [group for group in parity_groups if len(group) == max_size]
    slots = random.choice(choices)
    return random.sample(slots, count)


def _expected_output(
    grid: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    counts: list[int],
    palette_colors: list[int],
    bg: int,
    frame_color: int,
) -> np.ndarray:
    out = grid.copy()
    for bbox, count in zip(bboxes, counts):
        r0, c0, r1, c1 = bbox
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        target = frame_color if count == 0 else palette_colors[count - 1]
        _paint_frame(out, bbox, int(target))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg_choices = [color for color in range(COLOR_COUNT) if color != FRAME_COLOR]
    bg = random.choice(bg_choices)

    palette_len = random.randint(PALETTE_LEN_RANGE[0], PALETTE_LEN_RANGE[1])
    available = [color for color in range(COLOR_COUNT) if color not in {bg, FRAME_COLOR}]
    palette_colors = random.sample(available, palette_len)

    frame_count = random.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    counts = _sample_counts(frame_count, palette_len)

    sizes: list[tuple[int, int]] = []
    for count in counts:
        size = _sample_frame_size(count)
        if size is None:
            return None
        sizes.append(size)

    palette_span = 2 * palette_len - 1
    min_width = max(GRID_WIDTH_RANGE[0], palette_span, max(w for _h, w in sizes))
    min_height = max(GRID_HEIGHT_RANGE[0], max(h for h, _w in sizes) + 2)
    area_need = sum((h + 1) * (w + 1) for h, w in sizes)

    width_choices = [w for w in range(min_width, GRID_WIDTH_RANGE[1] + 1)]
    height_choices = [h for h in range(min_height, GRID_HEIGHT_RANGE[1] + 1)]
    random.shuffle(width_choices)
    random.shuffle(height_choices)

    dims: list[tuple[int, int]] = []
    for height in height_choices:
        for width in width_choices:
            if (height - 2) * width >= area_need:
                dims.append((int(height), int(width)))
    if not dims:
        return None

    height, width = random.choice(dims)
    bboxes = _sample_bboxes(height, width, sizes)
    if bboxes is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[-1, :] = _palette_row(width, bg, palette_colors)

    dot_pool = [color for color in range(COLOR_COUNT) if color not in {bg, FRAME_COLOR}]
    for bbox, count in zip(bboxes, counts):
        r0, c0, r1, c1 = bbox
        _paint_frame(grid, bbox, FRAME_COLOR)
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        for dr, dc in _sample_isolated_cells(inner_h, inner_w, count):
            grid[r0 + 1 + dr, c0 + 1 + dc] = random.choice(dot_pool)

    expected = _expected_output(grid, bboxes, counts, palette_colors, bg, FRAME_COLOR)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
