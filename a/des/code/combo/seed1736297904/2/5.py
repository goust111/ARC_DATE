from __future__ import annotations

import random
from collections import deque
from itertools import product

import numpy as np


BG = 0
FRAME_COLOR = 8
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)

GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
TARGET_COUNT_RANGE = (2, 4)
SRC_SIZE_BY_TARGETS = {
    2: (2, 5),
    3: (2, 4),
    4: (2, 3),
}
SCALE_BY_TARGETS = {
    2: (2, 5),
    3: (2, 4),
    4: (2, 3),
}
MAX_GENERATION_ATTEMPTS = 400

_RNG = random.Random()


def _background(grid: np.ndarray) -> int:
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
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_components(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == color):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == color):
            continue
        if np.any(sub[1:-1, 1:-1] == color):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def _detect_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    best_color = None
    best_frames: list[tuple[int, int, int, int]] = []
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        frames = _frame_components(grid, color)
        if len(frames) > len(best_frames):
            best_color = color
            best_frames = frames
    assert best_color is not None
    assert len(best_frames) >= 2
    return int(best_color), sorted(best_frames)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the source sprite into each larger frame and recolor it with the local anchor color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    _border_color, frames = _detect_frames(g, bg)

    infos: list[dict[str, object]] = []
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(interior))) - {bg}
        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior": interior,
                "colors": colors,
            }
        )

    source_candidates = [info for info in infos if len(info["colors"]) >= 2]
    assert len(source_candidates) == 1
    source = source_candidates[0]

    source_interior = np.asarray(source["interior"], dtype=int)
    sprite_mask = source_interior != bg
    assert sprite_mask.any()
    assert (~sprite_mask).any()
    src_h, src_w = sprite_mask.shape

    out = g.copy()
    for info in infos:
        if info is source:
            continue
        anchor_colors = sorted(info["colors"])
        assert len(anchor_colors) == 1
        anchor_color = int(anchor_colors[0])
        r0, c0, r1, c1 = info["bbox"]
        target_patch = out[r0 + 1 : r1, c0 + 1 : c1]
        tgt_h, tgt_w = target_patch.shape
        assert tgt_h % src_h == 0 and tgt_w % src_w == 0
        scale_h = tgt_h // src_h
        scale_w = tgt_w // src_w
        assert scale_h == scale_w
        scaled_mask = np.kron(sprite_mask, np.ones((scale_h, scale_h), dtype=bool))
        target_patch[scaled_mask] = anchor_color

    return out


def _random_partition(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[_RNG.randrange(parts)] += 1
    return out


def _sprite_neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_sprite_mask(h: int, w: int) -> np.ndarray | None:
    cell_count = h * w
    min_area = max(3, (cell_count + 2) // 3)
    max_area = min(cell_count - 1, (2 * cell_count + 2) // 3)
    if min_area > max_area:
        return None

    for _ in range(120):
        target_area = _RNG.randint(min_area, max_area)
        start = (_RNG.randrange(h), _RNG.randrange(w))
        mask = np.zeros((h, w), dtype=bool)
        mask[start] = True
        frontier = set(_sprite_neighbors(start[0], start[1], h, w))
        while int(mask.sum()) < target_area and frontier:
            cell = _RNG.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for nbr in _sprite_neighbors(cell[0], cell[1], h, w):
                if not mask[nbr]:
                    frontier.add(nbr)
        if int(mask.sum()) != target_area:
            continue
        if mask.all():
            continue
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0 + 1) < 2 or int(c1 - c0 + 1) < 2:
            continue
        if bool(mask[r0 : r1 + 1, c0 : c1 + 1].all()):
            continue
        return mask
    return None


def _colorize_sprite(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    _RNG.shuffle(cells)
    for cell, color in zip(cells, colors):
        patch[cell] = int(color)
    for cell in cells[len(colors) :]:
        patch[cell] = int(_RNG.choice(colors))
    return patch


def _pack_rectangles(sizes: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(sizes)
    order = list(range(n))
    _RNG.shuffle(order)
    shuffled = [sizes[idx] for idx in order]

    candidates: list[tuple[int, int, tuple[int, ...], list[list[int]], list[int], list[int]]] = []
    for row_count in range(1, min(3, n) + 1):
        for gap in range(1, 4):
            for assignment in product(range(row_count), repeat=n):
                if set(assignment) != set(range(row_count)):
                    continue
                rows: list[list[int]] = [[] for _ in range(row_count)]
                for idx, row_idx in enumerate(assignment):
                    rows[row_idx].append(idx)

                row_heights: list[int] = []
                row_widths: list[int] = []
                for items in rows:
                    width = sum(shuffled[i][1] for i in items) + gap * (len(items) - 1)
                    height = max(shuffled[i][0] for i in items)
                    row_widths.append(width)
                    row_heights.append(height)

                layout_h = sum(row_heights) + gap * (row_count - 1)
                layout_w = max(row_widths)
                if layout_h + 2 > GRID_H_RANGE[1] or layout_w + 2 > GRID_W_RANGE[1]:
                    continue
                candidates.append((row_count, gap, assignment, rows, row_heights, row_widths))

    if not candidates:
        return None

    row_count, gap, _assignment, rows, row_heights, row_widths = _RNG.choice(candidates)
    layout_h = sum(row_heights) + gap * (row_count - 1)
    layout_w = max(row_widths)
    min_h = max(GRID_H_RANGE[0], layout_h + 2)
    min_w = max(GRID_W_RANGE[0], layout_w + 2)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = _RNG.randint(min_h, GRID_H_RANGE[1])
    grid_w = _RNG.randint(min_w, GRID_W_RANGE[1])
    vertical_extra = grid_h - layout_h - 2
    row_offsets = _random_partition(vertical_extra, row_count + 1)

    positions_shuffled: list[tuple[int, int] | None] = [None] * n
    y = 1 + row_offsets[0]
    for row_idx, items in enumerate(rows):
        row_h = row_heights[row_idx]
        row_w = row_widths[row_idx]
        horizontal_extra = grid_w - row_w - 2
        slot_offsets = _random_partition(horizontal_extra, len(items) + 1)
        x = 1 + slot_offsets[0]
        for item_pos, item_idx in enumerate(items):
            rect_h, rect_w = shuffled[item_idx]
            rect_y = y + _RNG.randint(0, row_h - rect_h)
            positions_shuffled[item_idx] = (rect_y, x)
            x += rect_w
            if item_pos != len(items) - 1:
                x += gap + slot_offsets[item_pos + 1]
        y += row_h
        if row_idx != row_count - 1:
            y += gap + row_offsets[row_idx + 1]

    positions: list[tuple[int, int]] = [(-1, -1)] * n
    for shuffled_idx, original_idx in enumerate(order):
        pos = positions_shuffled[shuffled_idx]
        assert pos is not None
        positions[original_idx] = pos
    return grid_h, grid_w, positions


def _draw_frame(grid: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    grid[top, left : left + w] = FRAME_COLOR
    grid[top + h - 1, left : left + w] = FRAME_COLOR
    grid[top : top + h, left] = FRAME_COLOR
    grid[top : top + h, left + w - 1] = FRAME_COLOR


def _scaled_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask, np.ones((scale, scale), dtype=bool))


def _expected_output(
    grid: np.ndarray,
    source_bbox: tuple[int, int, int, int],
    targets: list[tuple[tuple[int, int, int, int], int]],
) -> np.ndarray:
    r0, c0, r1, c1 = source_bbox
    sprite_mask = grid[r0 + 1 : r1, c0 + 1 : c1] != BG
    src_h, src_w = sprite_mask.shape

    out = grid.copy()
    for (tr0, tc0, tr1, tc1), anchor_color in targets:
        patch = out[tr0 + 1 : tr1, tc0 + 1 : tc1]
        scale_h = patch.shape[0] // src_h
        scale_w = patch.shape[1] // src_w
        assert scale_h == scale_w
        patch[_scaled_mask(sprite_mask, scale_h)] = int(anchor_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    target_count = _RNG.randint(*TARGET_COUNT_RANGE)
    src_size_range = SRC_SIZE_BY_TARGETS[target_count]
    scale_range = SCALE_BY_TARGETS[target_count]

    src_h = _RNG.randint(*src_size_range)
    src_w = _RNG.randint(*src_size_range)
    sprite_mask = _random_sprite_mask(src_h, src_w)
    if sprite_mask is None:
        return None

    sprite_area = int(sprite_mask.sum())
    sprite_color_count = min(_RNG.randint(2, 4), sprite_area, len(COLOR_POOL) - target_count)
    if sprite_color_count < 2:
        return None

    palette = list(COLOR_POOL)
    _RNG.shuffle(palette)
    sprite_colors = palette[:sprite_color_count]
    anchor_colors = palette[sprite_color_count : sprite_color_count + target_count]
    if len(anchor_colors) != target_count:
        return None

    scales = [_RNG.randint(*scale_range) for _ in range(target_count)]
    rect_sizes = [(src_h + 2, src_w + 2)] + [(src_h * k + 2, src_w * k + 2) for k in scales]
    max_rect_h = max(h for h, _w in rect_sizes)
    max_rect_w = max(w for _h, w in rect_sizes)
    total_rect_area = sum(h * w for h, w in rect_sizes)
    if max_rect_h + 2 > GRID_H_RANGE[1] or max_rect_w + 2 > GRID_W_RANGE[1]:
        return None
    if total_rect_area > 420:
        return None

    packed = _pack_rectangles(rect_sizes)
    if packed is None:
        return None
    grid_h, grid_w, positions = packed

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    source_top, source_left = positions[0]
    source_bbox = (source_top, source_left, source_top + src_h + 1, source_left + src_w + 1)
    _draw_frame(grid, source_top, source_left, src_h + 2, src_w + 2)
    sprite_patch = _colorize_sprite(sprite_mask, sprite_colors)
    grid[source_top + 1 : source_top + 1 + src_h, source_left + 1 : source_left + 1 + src_w] = sprite_patch

    targets: list[tuple[tuple[int, int, int, int], int]] = []
    for idx, (scale, anchor_color) in enumerate(zip(scales, anchor_colors), start=1):
        target_top, target_left = positions[idx]
        tgt_h = src_h * scale
        tgt_w = src_w * scale
        _draw_frame(grid, target_top, target_left, tgt_h + 2, tgt_w + 2)
        scaled = _scaled_mask(sprite_mask, scale)
        true_cells = [tuple(map(int, cell)) for cell in np.argwhere(scaled)]
        false_cells = [tuple(map(int, cell)) for cell in np.argwhere(~scaled)]
        if not true_cells or not false_cells:
            return None

        anchor_positions = {
            _RNG.choice(true_cells),
            _RNG.choice(false_cells),
        }
        max_anchor_count = min(6, tgt_h * tgt_w)
        target_anchor_count = _RNG.randint(2, max_anchor_count)
        all_cells = [tuple(map(int, cell)) for cell in np.argwhere(np.ones_like(scaled, dtype=bool))]
        remaining = [cell for cell in all_cells if cell not in anchor_positions]
        _RNG.shuffle(remaining)
        for cell in remaining:
            if len(anchor_positions) >= target_anchor_count:
                break
            anchor_positions.add(cell)

        for ar, ac in anchor_positions:
            grid[target_top + 1 + ar, target_left + 1 + ac] = int(anchor_color)

        targets.append(((target_top, target_left, target_top + tgt_h + 1, target_left + tgt_w + 1), int(anchor_color)))

    expected = _expected_output(grid, source_bbox, targets)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
