from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SHAPE_COLOR = 5
SEPARATOR_COLOR = 6
PALETTE_POOL = [1, 2, 3, 4, 7, 8, 9]
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PALETTE_BLOCK = 3
PALETTE_GAP_RANGE = (1, 3)
SHAPE_RETRIES = 200
GENERATION_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            holes += 1
            q.append((r, c))
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return holes


def _component_holes(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_holes(mask)


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for c in range(grid.shape[1]):
        column = grid[:, c]
        if int(column[0]) != BG and np.all(column == column[0]):
            candidates.append((c, int(column[0])))
    assert len(candidates) == 1, "expected exactly one full-height separator line"
    return candidates[0]


def _extract_palette(left: np.ndarray) -> list[int]:
    blocks = []
    for coords in _components(left != BG):
        r0, c0, r1, c1 = _bbox(coords)
        color = int(left[r0, c0])
        assert np.all(left[coords[:, 0], coords[:, 1]] == color)
        assert r1 - r0 + 1 == PALETTE_BLOCK
        assert c1 - c0 + 1 == PALETTE_BLOCK
        assert coords.shape[0] == PALETTE_BLOCK * PALETTE_BLOCK
        blocks.append((r0, c0, color))
    blocks.sort()
    assert blocks, "expected at least one palette block"
    return [color for _r, _c, color in blocks]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each right-side shape according to its number of enclosed background holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_col, _sep_color = _find_separator(g)
    palette = _extract_palette(g[:, :sep_col])
    right = g[:, sep_col + 1 :]

    out = g.copy()
    for coords in _components(right != BG):
        values = {int(right[r, c]) for r, c in coords}
        assert len(values) == 1, "each template shape must be monochrome"
        holes = _component_holes(coords)
        assert 0 <= holes < len(palette), "hole count must be covered by the palette"
        color = int(palette[holes])
        out[coords[:, 0], sep_col + 1 + coords[:, 1]] = color

    return out


def _attach_tab(mask: np.ndarray) -> None:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    options = []
    if r0 > 0:
        options.append("top")
    if r1 + 1 < mask.shape[0]:
        options.append("bottom")
    if c0 > 0:
        options.append("left")
    if c1 + 1 < mask.shape[1]:
        options.append("right")
    if not options:
        return

    side = random.choice(options)
    if side in ("top", "bottom"):
        span_start = random.randint(c0, c1)
        span_end = random.randint(span_start, c1)
        if side == "top":
            thickness = random.randint(1, min(2, r0))
            mask[r0 - thickness : r0, span_start : span_end + 1] = True
        else:
            thickness = random.randint(1, min(2, mask.shape[0] - 1 - r1))
            mask[r1 + 1 : r1 + 1 + thickness, span_start : span_end + 1] = True
    else:
        span_start = random.randint(r0, r1)
        span_end = random.randint(span_start, r1)
        if side == "left":
            thickness = random.randint(1, min(2, c0))
            mask[span_start : span_end + 1, c0 - thickness : c0] = True
        else:
            thickness = random.randint(1, min(2, mask.shape[1] - 1 - c1))
            mask[span_start : span_end + 1, c1 + 1 : c1 + 1 + thickness] = True


def _shape_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _make_zero_hole_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        base_h = random.randint(3, 6)
        base_w = random.randint(3, 6)
        canvas = np.zeros((base_h + 4, base_w + 4), dtype=bool)
        canvas[2 : 2 + base_h, 2 : 2 + base_w] = True
        for _ in range(random.randint(1, 4)):
            _attach_tab(canvas)
        shape = _crop(canvas)
        if _shape_connected(shape) and _count_holes(shape) == 0:
            return shape
    return None


def _partition_sizes(count: int) -> list[int]:
    hi = 4 if count <= 2 else 3
    return [random.randint(2, hi) for _ in range(count)]


def _make_partitioned_frame(rows: int, cols: int) -> np.ndarray:
    room_heights = _partition_sizes(rows)
    room_widths = _partition_sizes(cols)
    h = 2 + sum(room_heights) + (rows - 1)
    w = 2 + sum(room_widths) + (cols - 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    cursor = 1 + room_heights[0]
    for room_h in room_heights[1:]:
        mask[cursor, :] = True
        cursor += 1 + room_h

    cursor = 1 + room_widths[0]
    for room_w in room_widths[1:]:
        mask[:, cursor] = True
        cursor += 1 + room_w

    return mask


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs = []
    for r in range(1, n + 1):
        if n % r == 0:
            pairs.append((r, n // r))
    return pairs


def _make_holey_shape(holes: int) -> np.ndarray | None:
    assert holes >= 1
    for _ in range(SHAPE_RETRIES):
        rows, cols = random.choice(_factor_pairs(holes))
        base = _make_partitioned_frame(rows, cols)
        canvas = np.zeros((base.shape[0] + 4, base.shape[1] + 4), dtype=bool)
        canvas[2 : 2 + base.shape[0], 2 : 2 + base.shape[1]] = base
        for _ in range(random.randint(0, 3)):
            _attach_tab(canvas)
        shape = _crop(canvas)
        if _shape_connected(shape) and _count_holes(shape) == holes:
            return shape
    return None


def _make_shape(holes: int) -> np.ndarray | None:
    if holes == 0:
        return _make_zero_hole_shape()
    return _make_holey_shape(holes)


def _split_positive(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _layout_shapes(items: list[tuple[int, np.ndarray]]) -> tuple[int, int, list[tuple[int, np.ndarray, int, int]]]:
    shuffled = items[:]
    random.shuffle(shuffled)
    row_count = random.randint(1, min(3, len(shuffled)))
    row_sizes = _split_positive(len(shuffled), row_count)

    groups = []
    index = 0
    for size in row_sizes:
        group = shuffled[index : index + size]
        index += size
        gap = random.randint(2, 4)
        row_h = max(shape.shape[0] for _holes, shape in group)
        content_w = sum(shape.shape[1] for _holes, shape in group) + gap * (len(group) - 1)
        groups.append((group, gap, row_h, content_w))

    row_gaps = [random.randint(2, 4) for _ in range(row_count - 1)]
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    width = max(content_w for _group, _gap, _row_h, content_w in groups) + random.randint(2, 6)
    height = top_margin + sum(row_h for _group, _gap, row_h, _content_w in groups) + sum(row_gaps) + bottom_margin

    placements: list[tuple[int, np.ndarray, int, int]] = []
    y = top_margin
    for idx, (group, gap, row_h, content_w) in enumerate(groups):
        x = random.randint(1, width - content_w - 1)
        for holes, shape in group:
            dy = random.randint(0, row_h - shape.shape[0])
            placements.append((holes, shape, y + dy, x))
            x += shape.shape[1] + gap
        y += row_h
        if idx < len(row_gaps):
            y += row_gaps[idx]

    return height, width, placements


def _draw_palette(grid: np.ndarray, left_w: int, palette_colors: list[int], gaps: list[int]) -> None:
    palette_h = PALETTE_BLOCK * len(palette_colors) + sum(gaps)
    x0 = random.randint(0, left_w - PALETTE_BLOCK)
    y = random.randint(0, grid.shape[0] - palette_h)
    for idx, color in enumerate(palette_colors):
        grid[y : y + PALETTE_BLOCK, x0 : x0 + PALETTE_BLOCK] = int(color)
        y += PALETTE_BLOCK
        if idx < len(gaps):
            y += gaps[idx]


def _build_expected(
    grid: np.ndarray,
    sep_col: int,
    palette_colors: list[int],
    right_y: int,
    placements: list[tuple[int, np.ndarray, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for holes, shape, r, c in placements:
        color = int(palette_colors[holes])
        rr = right_y + r
        cc = sep_col + 1 + c
        patch = out[rr : rr + shape.shape[0], cc : cc + shape.shape[1]]
        patch[shape] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(3, 5)
    palette_colors = random.sample(PALETTE_POOL, palette_len)
    palette_gaps = [random.randint(*PALETTE_GAP_RANGE) for _ in range(palette_len - 1)]
    palette_h = PALETTE_BLOCK * palette_len + sum(palette_gaps)

    hole_counts = list(range(palette_len))
    for _ in range(random.randint(0, 2)):
        hole_counts.append(random.randint(0, palette_len - 1))

    items: list[tuple[int, np.ndarray]] = []
    for holes in hole_counts:
        shape = _make_shape(holes)
        if shape is None:
            return None
        items.append((holes, shape))

    right_h, right_w, placements = _layout_shapes(items)
    left_w = random.randint(4, 7)
    total_h = max(right_h + random.randint(0, 3), palette_h + random.randint(0, 5))
    total_w = left_w + 1 + right_w
    grid = np.zeros((total_h, total_w), dtype=int)
    sep_col = left_w
    grid[:, sep_col] = SEPARATOR_COLOR

    _draw_palette(grid, left_w, palette_colors, palette_gaps)
    right_y = random.randint(0, total_h - right_h)
    for _holes, shape, r, c in placements:
        rr = right_y + r
        cc = sep_col + 1 + c
        patch = grid[rr : rr + shape.shape[0], cc : cc + shape.shape[1]]
        patch[shape] = SHAPE_COLOR

    expected = _build_expected(grid, sep_col, palette_colors, right_y, placements)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
