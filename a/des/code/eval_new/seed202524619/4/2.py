import numpy as np
from collections import deque
from typing import Dict, List, Tuple


def _infer_background_and_separator(grid: np.ndarray) -> Tuple[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])

    # A divider color should occupy many full rows/columns.
    row_score: Dict[int, int] = {}
    for row in grid:
        color = int(row[0])
        if np.all(row == color):
            row_score[color] = row_score.get(color, 0) + 1

    col_score: Dict[int, int] = {}
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            col_score[color] = col_score.get(color, 0) + 1

    score = row_score.copy()
    for k, v in col_score.items():
        score[k] = score.get(k, 0) + v

    sep_candidates = [(c, s) for c, s in score.items() if c != bg]
    if sep_candidates:
        sep = int(max(sep_candidates, key=lambda x: x[1])[0])
    else:
        # Fallback: choose the most frequent non-background color.
        others = [(int(c), int(s)) for c, s in zip(values, counts) if int(c) != bg]
        sep = int(max(others, key=lambda x: x[1])[0]) if others else bg

    return bg, sep


def _label_regions(grid: np.ndarray, sep: int) -> np.ndarray:
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] == sep or labels[r, c] != 0:
                continue
            label += 1
            dq = deque([(r, c)])
            labels[r, c] = label
            while dq:
                x, y = dq.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and labels[nx, ny] == 0 and grid[nx, ny] != sep:
                        labels[nx, ny] = label
                        dq.append((nx, ny))
    return labels


def _find_arrows(grid: np.ndarray, bg: int, sep: int):
    h, w = grid.shape
    patterns = {
        "up": ((-1, 0), (0, -1), (0, 1), (1, 0)),
        "down": ((1, 0), (0, -1), (0, 1), (-1, 0)),
        "left": ((0, -1), (-1, 0), (1, 0), (0, 1)),
        "right": ((0, 1), (-1, 0), (1, 0), (0, -1)),
    }

    arrows = []
    arrow_mask = np.zeros_like(grid, dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or color == sep:
                continue
            for direction, (tip, side1, side2, back) in patterns.items():
                dr_t, dc_t = tip
                dr_s1, dc_s1 = side1
                dr_s2, dc_s2 = side2
                dr_b, dc_b = back
                p = ((r + dr_t, c + dc_t), (r + dr_s1, c + dc_s1), (r + dr_s2, c + dc_s2), (r + dr_b, c + dc_b))
                if any(not (0 <= x < h and 0 <= y < w) for x, y in p):
                    continue
                if (
                    grid[r + dr_t, c + dc_t] == color
                    and grid[r + dr_s1, c + dc_s1] == color
                    and grid[r + dr_s2, c + dc_s2] == color
                    and grid[r + dr_b, c + dc_b] != color
                ):
                    coords = [(r, c), (r + dr_t, c + dc_t), (r + dr_s1, c + dc_s1), (r + dr_s2, c + dc_s2)]
                    for x, y in coords:
                        arrow_mask[x, y] = True
                    arrows.append(
                        {
                            "center": (r, c),
                            "direction": direction,
                            "color": color,
                            "cells": coords,
                        }
                    )
                    break
    return arrows, arrow_mask


def main(grid: np.ndarray) -> np.ndarray:
    """根据区域中的箭头指向，把相邻区域填成箭头颜色，并清除源箭头。
    Input structure:
    - 背景为最常见颜色；网格被同一颜色的完整行和完整列分割成若干矩形区域。
    - 某些区域内有一个由 4 个同色像素构成的 T 形箭头（中心+三邻域像素），其余为背景色。
    Rule:
    - 在输入中识别每个箭头的颜色与方向（上/下/左/右）。
    - 沿箭头方向越过一条灰色分隔线，定位相邻目标区域。
    - 将目标区域中原本的背景像素，以及目标区域内的箭头像素，统一改为该源箭头颜色。
    - 所有箭头并行生效：基于输入状态一次性重绘，再清除所有源箭头为背景色。
    Key insight:
    - 把箭头解码为方向后，它代表的是“目标区域级指令”而非自身标记，且需同时作用避免级联干扰。"""

    bg, sep = _infer_background_and_separator(grid)
    labels = _label_regions(grid, sep)
    arrows, source_arrow_mask = _find_arrows(grid, bg, sep)

    out = grid.copy()
    direction_delta = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    for arrow in arrows:
        r, c = arrow["center"]
        dr, dc = direction_delta[arrow["direction"]]
        rr, cc = r + dr, c + dc
        crossed = False
        target_label = 0

        while 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
            if grid[rr, cc] == sep:
                crossed = True
                rr += dr
                cc += dc
                continue
            if not crossed:
                rr += dr
                cc += dc
                continue
            target_label = int(labels[rr, cc])
            break

        if target_label <= 0:
            continue

        target_mask = labels == target_label
        recolor_mask = target_mask & ((grid == bg) | source_arrow_mask)
        out[recolor_mask] = arrow["color"]

    # Remove all source arrows after recoloring is completed.
    for arrow in arrows:
        for x, y in arrow["cells"]:
            out[x, y] = bg

    return out


def _interior_point(bounds: Tuple[int, int, int, int], rng: np.random.Generator) -> Tuple[int, int]:
    r0, r1, c0, c1 = bounds
    return int(rng.integers(r0 + 1, r1 - 1)), int(rng.integers(c0 + 1, c1 - 1))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = np.arange(10, dtype=np.int64)

    while True:
        rows = int(rng.integers(2, 6))
        cols = int(rng.integers(2, 6))
        row_sizes = np.array([int(v) for v in rng.integers(3, 8, size=rows)], dtype=int)
        col_sizes = np.array([int(v) for v in rng.integers(3, 8, size=cols)], dtype=int)
        h = int(row_sizes.sum() + (rows - 1))
        w = int(col_sizes.sum() + (cols - 1))
        if h <= 30 and w <= 30:
            break

    bg, sep = rng.choice(palette, size=2, replace=False)
    other_colors = [int(v) for v in palette if int(v) not in (int(bg), int(sep))]

    # Build region bounds: [r0, r1), [c0, c1)
    row_bounds: List[Tuple[int, int]] = []
    cur = 0
    for size in row_sizes:
        row_bounds.append((cur, cur + size))
        cur += size + 1

    col_bounds: List[Tuple[int, int]] = []
    cur = 0
    for size in col_sizes:
        col_bounds.append((cur, cur + size))
        cur += size + 1

    grid = np.full((h, w), int(bg), dtype=np.int64)

    # Full-width/full-height separators.
    for i in range(rows - 1):
        grid[row_bounds[i][1], :] = int(sep)
    for j in range(cols - 1):
        grid[:, col_bounds[j][1]] = int(sep)

    total_regions = rows * cols
    allowed_dirs = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    region_dir_map: Dict[int, str] = {}
    region_with_arrow: List[int] = []

    # Initial random arrow assignment (ensure at least one).
    for idx in range(total_regions):
        if rng.random() < 0.5:
            region_with_arrow.append(idx)

    if not region_with_arrow:
        region_with_arrow.append(int(rng.integers(total_regions)))

    # Enforce branches: some targets also carry arrows.
    for idx in list(region_with_arrow):
        rr, cc = divmod(idx, cols)
        opts = ["up", "down", "left", "right"]
        if rr == 0:
            opts.remove("up")
        if rr == rows - 1:
            opts.remove("down")
        if cc == 0:
            opts.remove("left")
        if cc == cols - 1:
            opts.remove("right")
        if not opts:
            continue
        dir_choice = allowed_dirs[rng.choice(opts)]
        region_dir_map[idx] = [k for k, v in allowed_dirs.items() if v == tuple(dir_choice)][0]

    # Chance to create an arrow in the destination as branch case.
    for idx, direction in list(region_dir_map.items()):
        rr, cc = divmod(idx, cols)
        dr, dc = allowed_dirs[direction]
        tr = rr + dr
        tc = cc + dc
        if 0 <= tr < rows and 0 <= tc < cols:
            dst_idx = tr * cols + tc
            if dst_idx not in region_dir_map and rng.random() < 0.45:
                # Assign a direction in destination region, independent of incoming one.
                dst_opts = ["up", "down", "left", "right"]
                if tr == 0:
                    dst_opts.remove("up")
                if tr == rows - 1:
                    dst_opts.remove("down")
                if tc == 0:
                    dst_opts.remove("left")
                if tc == cols - 1:
                    dst_opts.remove("right")
                if dst_opts:
                    region_dir_map[dst_idx] = str(rng.choice(dst_opts))

    region_colors = {
        idx: int(rng.choice(other_colors)) for idx in region_dir_map.keys()
    }

    for idx, direction in region_dir_map.items():
        rr, cc = divmod(idx, cols)
        r0, r1 = row_bounds[rr]
        c0, c1 = col_bounds[cc]
        r, c = _interior_point((r0, r1, c0, c1), rng)
        color = region_colors[idx]

        if direction == "up":
            coords = [(r, c), (r - 1, c), (r, c - 1), (r, c + 1)]
        elif direction == "down":
            coords = [(r, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        elif direction == "left":
            coords = [(r, c), (r, c - 1), (r - 1, c), (r + 1, c)]
        else:  # right
            coords = [(r, c), (r, c + 1), (r - 1, c), (r + 1, c)]

        for x, y in coords:
            grid[x, y] = int(color)

    return grid
