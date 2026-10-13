from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_legend(
    grid: np.ndarray, bg: int
) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    best: tuple[int, int, list[int], np.ndarray] | None = None

    def check(side: str, square_size: int):
        nonlocal best
        if square_size <= 0 or square_size + 1 > h:
            return
        if side == "top":
            r0 = 0
        else:
            r0 = h - square_size - 1
        band = grid[r0 : r0 + square_size + 1, :]
        bg_sig = tuple([bg] * (square_size + 1))
        entries: list[int] = []
        mask = np.zeros_like(grid, dtype=bool)
        c = 0
        while c < w:
            sig = tuple(int(x) for x in band[:, c])
            c2 = c + 1
            while c2 < w and tuple(int(x) for x in band[:, c2]) == sig:
                c2 += 1
            width = c2 - c
            if sig == bg_sig:
                c = c2
                continue
            if width != square_size:
                return
            square = sig[:-1]
            strip = int(sig[-1])
            if strip == bg or any(v == bg for v in square):
                return
            if len(set(square)) != 1:
                return
            entries.append(strip)
            mask[r0 : r0 + square_size + 1, c:c2] = True
            c = c2
        if len(entries) < 1:
            return
        if np.any((band != bg) & (~mask[r0 : r0 + square_size + 1])):
            return
        score = (len(entries), square_size)
        if best is None or score > (best[0], best[1]):
            best = (len(entries), square_size, entries, mask)

    for square_size in range(1, min(h - 1, w) + 1):
        check("top", square_size)
        check("bottom", square_size)

    assert best is not None, "legend not found"
    return best[2], best[3]


def _count_layers(interior: np.ndarray, bg: int) -> int | None:
    if interior.size == 0 or np.any(interior == bg):
        return None
    row_colors: list[int] = []
    for row in interior:
        vals = np.unique(row)
        if len(vals) != 1:
            return None
        row_colors.append(int(vals[0]))
    layers = 1
    for i in range(1, len(row_colors)):
        if row_colors[i] != row_colors[i - 1]:
            layers += 1
    return layers


def _find_containers(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray
) -> list[tuple[int, tuple[int, int, int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    containers: list[tuple[int, tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or legend_mask[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if visited[nr, nc] or legend_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            if box_h < 3 or box_w < 3:
                continue
            expected = 2 * (box_h + box_w) - 4
            if len(cells) != expected:
                continue
            if any(rr not in (r0, r1) and cc not in (c0, c1) for rr, cc in cells):
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            layers = _count_layers(interior, bg)
            if layers is None:
                continue
            containers.append((layers, (r0, c0, r1, c1)))

    assert containers, "no containers found"
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular container border using the target strip color from the
    Nth legend entry, where N is the number of horizontal layers inside the container.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_targets, legend_mask = _find_legend(g, bg)
    containers = _find_containers(g, bg, legend_mask)

    out = g.copy()
    for layers, (r0, c0, r1, c1) in containers:
        assert 1 <= layers <= len(legend_targets), "container layer count exceeds legend"
        color = int(legend_targets[layers - 1])
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


COLORS = tuple(range(10))
GEN_ATTEMPTS = 400
GRID_H_RANGE = (15, 24)
GRID_W_RANGE = (18, 30)
LEGEND_ENTRY_RANGE = (3, 5)
LEGEND_SIZE_RANGE = (2, 3)
LEGEND_GAP_RANGE = (1, 2)
CONTAINER_COUNT_RANGE = (2, 4)
INTERIOR_W_RANGE = (2, 6)
BAR_HEIGHT_RANGE = (1, 3)


def _rand_non_bg(bg: int) -> int:
    return random.choice([c for c in COLORS if c != bg])


def _layer_row_colors(bg: int, border: int, layer_count: int) -> list[int]:
    choices = [c for c in COLORS if c not in (bg, border)]
    colors: list[int] = []
    for _ in range(layer_count):
        allowed = [c for c in choices if not colors or c != colors[-1]]
        colors.append(random.choice(allowed))
    return colors


def _paint_container(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_w: int,
    heights: list[int],
    border: int,
    bg: int,
) -> None:
    inner_h = sum(heights)
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = border
    grid[bottom, left : right + 1] = border
    grid[top : bottom + 1, left] = border
    grid[top : bottom + 1, right] = border

    row = top + 1
    for color, height in zip(_layer_row_colors(bg, border, len(heights)), heights):
        grid[row : row + height, left + 1 : right] = color
        row += height


def _place_containers(
    grid: np.ndarray,
    bg: int,
    max_layers: int,
    usable_bottom: int,
) -> bool:
    h, w = grid.shape
    count = random.randint(CONTAINER_COUNT_RANGE[0], min(CONTAINER_COUNT_RANGE[1], max_layers))
    layer_counts = random.sample(list(range(1, max_layers + 1)), count)
    random.shuffle(layer_counts)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[usable_bottom + 1 :, :] = True
    specs: list[tuple[int, int, int, list[int], int]] = []
    for layers in sorted(layer_counts, reverse=True):
        inner_w = random.randint(*INTERIOR_W_RANGE)
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(layers)]
        outer_h = sum(heights) + 2
        outer_w = inner_w + 2
        specs.append((layers, outer_h, outer_w, heights, inner_w))

    for _layers, outer_h, outer_w, heights, inner_w in specs:
        placed = False
        top_max = usable_bottom - outer_h + 1
        if top_max < 1 or outer_w > w - 2:
            return False
        for _ in range(120):
            top = random.randint(1, top_max)
            left = random.randint(1, w - outer_w - 1)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            rr0 = max(0, top - 1)
            rr1 = min(h, bottom + 2)
            cc0 = max(0, left - 1)
            cc1 = min(w, right + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            border = _rand_non_bg(bg)
            _paint_container(grid, top, left, inner_w, heights, border, bg)
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _paint_legend(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    entry_count = random.randint(*LEGEND_ENTRY_RANGE)
    square = random.randint(*LEGEND_SIZE_RANGE)
    gap = random.randint(*LEGEND_GAP_RANGE)
    width = entry_count * square + (entry_count - 1) * gap
    if width > w - 2 or square + 3 > h:
        raise ValueError("legend does not fit")

    left = random.randint(1, w - width - 1)
    top = h - square - 1

    square_colors = random.sample([c for c in COLORS if c != bg], entry_count)
    strip_colors = random.sample([c for c in COLORS if c != bg], entry_count)
    if all(a == b for a, b in zip(square_colors, strip_colors)):
        strip_colors = strip_colors[1:] + strip_colors[:1]

    for i in range(entry_count):
        c0 = left + i * (square + gap)
        c1 = c0 + square
        grid[top : top + square, c0:c1] = square_colors[i]
        grid[top + square, c0:c1] = strip_colors[i]
    return top, entry_count


def _validate_generated(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    legend_targets, legend_mask = _find_legend(grid, bg)
    containers = _find_containers(grid, bg, legend_mask)
    if len(legend_targets) < 3 or len(containers) < 2:
        return False
    layer_counts = [layers for layers, _ in containers]
    if max(layer_counts) > len(legend_targets):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        bg = random.choice(COLORS)
        grid = np.full((h, w), bg, dtype=int)
        try:
            legend_top, entry_count = _paint_legend(grid, bg)
        except ValueError:
            continue
        usable_bottom = legend_top - 2
        if usable_bottom < 4:
            continue
        if not _place_containers(grid, bg, entry_count, usable_bottom):
            continue
        if not _validate_generated(grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
