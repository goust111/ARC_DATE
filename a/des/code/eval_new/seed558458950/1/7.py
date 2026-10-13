import random
from collections import Counter, deque
from typing import Dict, List, Tuple

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid.reshape(-1), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_mask(shape: Tuple[int, int], r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _collect_components(
    grid: np.ndarray,
    include: np.ndarray,
    bg: int,
) -> List[Dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros_like(include, dtype=bool)
    comps: List[Dict[str, object]] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for sr in range(h):
        for sc in range(w):
            if not include[sr, sc] or visited[sr, sc] or grid[sr, sc] == bg:
                continue

            color = int(grid[sr, sc])
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells: List[Tuple[int, int]] = []
            rmin, cmin = sr, sc
            rmax, cmax = sr, sc

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r < rmin:
                    rmin = r
                if r > rmax:
                    rmax = r
                if c < cmin:
                    cmin = c
                if c > cmax:
                    cmax = c

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if visited[nr, nc] or not include[nr, nc]:
                        continue
                    if grid[nr, nc] != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            comps.append(
                {
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (rmin, cmin, rmax, cmax),
                    "color": color,
                }
            )

    return comps


def _detect_legend_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    scan_h = min(12, h)
    scan_w = min(22, w)
    scan = grid[:scan_h, :scan_w]

    non_bg_local = scan != bg
    if not np.any(non_bg_local):
        return np.zeros_like(grid, dtype=bool)

    # The legend border is usually gray. If absent, fall back to most frequent local non-bg color.
    cand_colors, cand_counts = np.unique(scan[non_bg_local], return_counts=True)
    if len(cand_colors) == 0:
        return np.zeros_like(grid, dtype=bool)

    if 5 in cand_colors:
        border_color = 5
    else:
        border_color = int(cand_colors[int(np.argmax(cand_counts))])

    border_mask = scan == border_color
    if not np.any(border_mask):
        return np.zeros_like(grid, dtype=bool)

    visited = np.zeros_like(scan, dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    bboxes = []

    for sr in range(scan_h):
        for sc in range(scan_w):
            if not border_mask[sr, sc] or visited[sr, sc]:
                continue

            q = deque([(sr, sc)])
            visited[sr, sc] = True
            rmin, cmin = sr, sc
            rmax, cmax = sr, sc
            comp = 0

            while q:
                r, c = q.popleft()
                comp += 1
                if r < rmin:
                    rmin = r
                if r > rmax:
                    rmax = r
                if c < cmin:
                    cmin = c
                if c > cmax:
                    cmax = c

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= scan_h or nc < 0 or nc >= scan_w:
                        continue
                    if visited[nr, nc] or not border_mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            if comp < 4:
                continue
            if rmax - rmin > 12 or cmax - cmin > 20:
                continue
            if rmin > 4 and cmin > 4:
                continue
            bboxes.append((rmin, cmin, rmax, cmax))

    if not bboxes:
        rows, cols = np.where(non_bg_local)
        if len(rows) == 0:
            return np.zeros_like(grid, dtype=bool)
        r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
        return _bbox_mask(grid.shape, max(0, r0 - 1), min(h, r1 + 2), max(0, c0 - 1), min(w, c1 + 2))

    r0 = min(b[0] for b in bboxes)
    c0 = min(b[1] for b in bboxes)
    r1 = max(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)

    return np.logical_and(
        _bbox_mask(grid.shape, max(0, r0 - 1), min(h, r1 + 2), max(0, c0 - 1), min(w, c1 + 2)),
        grid != bg,
    )


def _apply_texture(
    out: np.ndarray,
    cells: List[Tuple[int, int]],
    pattern: str,
    fg: int,
    bg: int,
    anchor: Tuple[int, int],
) -> None:
    ar, ac = anchor
    for r, c in cells:
        lr = r - ar
        lc = c - ac
        if pattern == "h":
            fill_fg = (lr % 2 == 0)
        elif pattern == "v":
            fill_fg = (lc % 2 == 0)
        else:
            fill_fg = ((lr + lc) % 2 == 0)
        out[r, c] = fg if fill_fg else bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid connected shape interior with cyclic strip textures determined by area ranking.

    Input structure:
    The grid uses the most frequent color as background (bg). A corner legend shows three texture samples
    separated by gray borders: horizontal stripes, vertical stripes, and checkerboard. The remaining non-bg
    pixels are several single-color connected shapes.

    Rule:
    Keep the legend area unchanged. Find all non-bg connected components outside the legend, treat each as one
    shape, and sort by area ascending; ties are broken by smaller top row then smaller left column of the shape
    bounding box. The 1st, 2nd, 3rd, 4th... ranked shapes receive textures in order: horizontal,
    vertical, checkerboard, then repeat.

    Key insight:
    The only semantic branch is the ranking: object area (and top-left tie-break) decides which texture to apply,
    while the texture phase starts from the shape bounding-box top-left pixel.
    """
    out = np.array(grid, copy=True)
    bg = _most_common_color(grid)
    legend_mask = _detect_legend_mask(grid, bg)

    include = ~legend_mask
    shapes = _collect_components(grid, include, bg)
    shapes.sort(
        key=lambda s: (
            s["area"],
            s["bbox"][0],
            s["bbox"][1],
        )
    )

    patterns = ("h", "v", "c")
    for i, shape in enumerate(shapes):
        rmin, cmin, _, _ = shape["bbox"]
        _apply_texture(out, shape["cells"], patterns[i % 3], int(shape["color"]), int(bg), (rmin, cmin))

    return out


def _draw_legend(area: np.ndarray, bg: int, colors: Tuple[int, int, int], border_color: int = 5) -> None:
    h, w = area.shape
    area.fill(bg)

    sample_size = 2
    frame = sample_size + 2
    top = 1
    gap = 1
    left0 = 1

    for i in range(3):
        left = left0 + i * (frame + gap)
        right = left + frame - 1
        bottom = top + frame - 1
        if right >= w or bottom >= h:
            break

        # Gray frame border.
        area[top:bottom + 1, left] = border_color
        area[top:bottom + 1, right] = border_color
        area[top, left:right + 1] = border_color
        area[bottom, left:right + 1] = border_color

        # Texture sample in inner 2x2 area.
        ir = top + 1
        ic = left + 1
        fc = colors[i]
        for dr in range(sample_size):
            for dc in range(sample_size):
                if i == 0:  # horizontal
                    use_fg = (dr % 2 == 0)
                elif i == 1:  # vertical
                    use_fg = (dc % 2 == 0)
                else:  # checkerboard
                    use_fg = ((dr + dc) % 2 == 0)
                area[ir + dr, ic + dc] = fc if use_fg else bg


def _random_connected_shape(
    occupancy: np.ndarray,
    rng: random.Random,
    area: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
) -> List[Tuple[int, int]]:
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    h, w = occupancy.shape

    for _ in range(240):
        r0 = rng.randrange(row_lo, row_hi)
        c0 = rng.randrange(col_lo, col_hi)
        if occupancy[r0, c0]:
            continue

        cells: List[Tuple[int, int]] = [(r0, c0)]
        frontier = [(r0, c0)]
        cell_set = {(r0, c0)}

        while len(cells) < area:
            expand = []
            for fr, fc in frontier:
                for dr, dc in dirs:
                    nr, nc = fr + dr, fc + dc
                    if nr < row_lo or nr >= row_hi or nc < col_lo or nc >= col_hi:
                        continue
                    if occupancy[nr, nc] or (nr, nc) in cell_set:
                        continue
                    expand.append((nr, nc))

            if not expand:
                break

            nr, nc = rng.choice(expand)
            cells.append((nr, nc))
            cell_set.add((nr, nc))
            frontier.append((nr, nc))

        if len(cells) == area:
            return cells

    return []


def _fill_shape(grid: np.ndarray, cells: List[Tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def generate_input() -> np.ndarray:
    rng = random.Random()

    h = rng.randint(16, 30)
    w = rng.randint(20, 30)

    bg = 0
    gray = 5
    grid = np.full((h, w), bg, dtype=np.int32)

    # Small top-left legend.
    legend_h = 7
    legend_w = 16
    legend = np.full((legend_h, legend_w), bg, dtype=np.int32)
    legend_colors = (1, 2, 3)
    _draw_legend(legend, bg=bg, colors=legend_colors, border_color=gray)
    grid[:legend_h, :legend_w] = legend

    occupancy = grid != bg

    # Object placement region: away from legend and boundaries.
    row_lo = legend_h + 1
    row_hi = h - 1
    col_lo = 1
    col_hi = w - 1

    n_shapes = rng.randint(3, 7)
    target_sizes = [rng.randint(6, 30) for _ in range(n_shapes)]

    # Force occasional equal-area tie cases.
    if n_shapes >= 2 and rng.random() < 0.5:
        i, j = rng.sample(range(n_shapes), 2)
        target_sizes[j] = target_sizes[i]

    palette = [4, 6, 7, 8, 9, 10, 11, 12]
    rng.shuffle(palette)
    shape_colors = palette[:n_shapes]

    placed = 0
    for size, color in zip(target_sizes, shape_colors):
        cells = _random_connected_shape(occupancy, rng, size, row_lo, row_hi, col_lo, col_hi)
        attempts = 0
        cur = size
        while not cells and attempts < 4:
            attempts += 1
            cur = max(4, cur - 4)
            cells = _random_connected_shape(occupancy, rng, cur, row_lo, row_hi, col_lo, col_hi)

        if not cells:
            continue

        _fill_shape(grid, cells, color)
        for r, c in cells:
            occupancy[r, c] = True
        placed += 1

    # Maintain expected complexity and guarantee at least 3 shapes in most runs.
    if placed < 3:
        # fallback pass: place smaller shapes with lower constraints.
        for _ in range(3 - placed):
            color = palette[len(legend_colors) + placed]
            cells = _random_connected_shape(occupancy, rng, 5, row_lo, row_hi, col_lo, col_hi)
            if cells:
                _fill_shape(grid, cells, color)
                for r, c in cells:
                    occupancy[r, c] = True

    return grid


if __name__ == "__main__":
    for _ in range(3):
        inp = generate_input()
        print(main(inp).shape)
