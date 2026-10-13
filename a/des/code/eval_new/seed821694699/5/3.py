"""ARC solver for puzzle 3.

1) 识别前景的1像素矩形框（仅边框，无内填充），并检测存在且仅存在一侧连续缺口。
2) 在每个框内部提取唯一的非背景连通图案，计算其外推方向与平移距离。
3) 按规则将图案整体平移到框外，并把框内原图案清空为背景色。
4) 输出与输入同尺寸整数颜色网格，允许后处理图案覆盖边框像素。
"""

import random
from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


BG = 0


def _find_connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Find 4-neighbor connected components in a boolean mask."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for i in range(h):
        row_mask = mask[i]
        for j in range(w):
            if not row_mask[j] or visited[i, j]:
                continue
            q = deque([(i, j)])
            visited[i, j] = True
            cells: List[Tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                nr1 = r + 1
                nr2 = r - 1
                nc1 = c + 1
                nc2 = c - 1
                if nr1 < h and mask[nr1, c] and not visited[nr1, c]:
                    visited[nr1, c] = True
                    q.append((nr1, c))
                if nr2 >= 0 and mask[nr2, c] and not visited[nr2, c]:
                    visited[nr2, c] = True
                    q.append((nr2, c))
                if nc1 < w and mask[r, nc1] and not visited[r, nc1]:
                    visited[r, nc1] = True
                    q.append((r, nc1))
                if nc2 >= 0 and mask[r, nc2] and not visited[r, nc2]:
                    visited[r, nc2] = True
                    q.append((r, nc2))
            comps.append(cells)
    return comps


def _max_contiguous_false(bits: Iterable[bool]) -> int:
    """Length of longest contiguous False run in order."""
    max_run = 0
    cur = 0
    for b in bits:
        if b:
            cur = 0
        else:
            cur += 1
            if cur > max_run:
                max_run = cur
    return max_run


def _detect_frames(grid: np.ndarray) -> List[dict]:
    """Detect valid single-color ring frames with exactly one gapped side."""
    h, w = grid.shape
    frames = []
    colors = [c for c in np.unique(grid) if c != BG]
    for color in colors:
        mask = grid == color
        for comp in _find_connected_components(mask):
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            if bh < 3 or bw < 3:
                continue

            cells = set(comp)
            perimeter_count = 0
            invalid = False
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if rr == r0 or rr == r1 or cc == c0 or cc == c1:
                        perimeter_count += 1
                        if (rr, cc) not in cells:
                            continue
                    elif (rr, cc) in cells:
                        invalid = True
                        break
                if invalid:
                    break
            if invalid:
                continue

            if len(cells) > perimeter_count:
                continue

            top = [((r0, cc) in cells) for cc in range(c0, c1 + 1)]
            bottom = [((r1, cc) in cells) for cc in range(c0, c1 + 1)]
            left = [((rr, c0) in cells) for rr in range(r0, r1 + 1)]
            right = [((rr, c1) in cells) for rr in range(r0, r1 + 1)]

            miss_top = top.count(False)
            miss_bottom = bottom.count(False)
            miss_left = left.count(False)
            miss_right = right.count(False)
            cont_top = _max_contiguous_false(top)
            cont_bottom = _max_contiguous_false(bottom)
            cont_left = _max_contiguous_false(left)
            cont_right = _max_contiguous_false(right)

            side_ok = []
            if miss_top > 0 and cont_top == miss_top and miss_top >= 2:
                side_ok.append(("up", 0, -1))
            if miss_bottom > 0 and cont_bottom == miss_bottom and miss_bottom >= 2:
                side_ok.append(("down", 0, 1))
            if miss_left > 0 and cont_left == miss_left and miss_left >= 2:
                side_ok.append(("left", -1, 0))
            if miss_right > 0 and cont_right == miss_right and miss_right >= 2:
                side_ok.append(("right", 1, 0))

            if len(side_ok) != 1:
                continue

            # Other sides should be intact ring.
            if not (
                miss_top == 0 or side_ok[0][0] == "up"
            ):
                continue
            if not (
                miss_bottom == 0 or side_ok[0][0] == "down"
            ):
                continue
            if not (
                miss_left == 0 or side_ok[0][0] == "left"
            ):
                continue
            if not (
                miss_right == 0 or side_ok[0][0] == "right"
            ):
                continue

            frames.append(
                {
                    "color": int(color),
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "gap_side": side_ok[0][0],
                    "cells": cells,
                    "comp": comp,
                }
            )
    return frames


def _extract_pattern(grid: np.ndarray, frame: dict) -> List[Tuple[int, int, int]]:
    """Extract pattern pixels as (r,c,val) from frame interior."""
    r0, r1, c0, c1 = frame["r0"], frame["r1"], frame["c0"], frame["c1"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    h, w = inner.shape

    if h < 1 or w < 1:
        return []

    inner_mask = inner != BG
    if not inner_mask.any():
        return []

    # Identify the connected region(s) inside; choose the largest one.
    comps = _find_connected_components(inner_mask)
    if not comps:
        return []

    comps.sort(key=len, reverse=True)
    comp = comps[0]
    pattern = []
    for rr, cc in comp:
        gr = r0 + 1 + rr
        gc = c0 + 1 + cc
        pattern.append((gr, gc, int(grid[gr, gc])))
    return pattern


def _compute_shift(grid: np.ndarray, frame: dict, pattern: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    """Compute translation vector for a pattern based on gap direction and geometry."""
    if not pattern:
        return (0, 0)

    pr = [p[0] for p in pattern]
    pc = [p[1] for p in pattern]
    p_r0, p_r1 = min(pr), max(pr)
    p_c0, p_c1 = min(pc), max(pc)
    h = p_r1 - p_r0 + 1
    w = p_c1 - p_c0 + 1

    dr = dc = 0
    gap_side = frame["gap_side"]
    r0, r1, c0, c1 = frame["r0"], frame["r1"], frame["c0"], frame["c1"]
    if gap_side == "up":
        dist = p_r0 - r0
        step = h + dist + 1
        dr = -step
    elif gap_side == "down":
        dist = r1 - p_r1
        step = h + dist + 1
        dr = step
    elif gap_side == "left":
        dist = p_c0 - c0
        step = w + dist + 1
        dc = -step
    elif gap_side == "right":
        dist = c1 - p_c1
        step = w + dist + 1
        dc = step
    return dr, dc


def main(grid: np.ndarray) -> np.ndarray:
    """Solve one ARC grid.

    1. Identify each valid ring frame and its gap side.
    2. Extract the enclosed connected pattern inside each frame.
    3. Erase pattern cells in-frame to background.
    4. Translate each pattern using computed offset and write over output.
    """
    arr = np.array(grid, dtype=int).copy()
    h, w = arr.shape
    out = arr.copy()

    frames = _detect_frames(arr)
    movements: List[Tuple[List[Tuple[int, int, int]], int, int]] = []

    for frame in frames:
        pattern = _extract_pattern(arr, frame)
        if not pattern:
            continue
        dr, dc = _compute_shift(arr, frame, pattern)
        if dr == 0 and dc == 0:
            continue
        movements.append((pattern, dr, dc))

        for r, c, _ in pattern:
            out[r, c] = BG

    for pattern, dr, dc in movements:
        for r, c, v in pattern:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = v

    return out


def generate_input() -> np.ndarray:
    """Generate random valid input within size<=30 for this ARC-style task."""
    random.seed()

    H = random.randint(10, 24)
    W = random.randint(10, 24)
    grid = np.zeros((H, W), dtype=int)
    used = set()

    num_frames = random.randint(1, 3)
    colors = list(range(1, 10))
    random.shuffle(colors)

    def overlaps(rect):
        r0, r1, c0, c1 = rect
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if grid[rr, cc] != BG:
                    return True
        return False

    for fidx in range(num_frames):
        if not colors:
            break
        # Build a usable border with at least 5x5.
        for _ in range(200):
            h = random.randint(5, min(15, H))
            w = random.randint(5, min(15, W))
            r0 = random.randint(0, H - h)
            c0 = random.randint(0, W - w)
            r1 = r0 + h - 1
            c1 = c0 + w - 1

            inner_h = h - 2
            inner_w = w - 2
            if inner_h < 3 or inner_w < 3:
                continue

            pattern_h = random.randint(1, inner_h - 2)
            pattern_w = random.randint(1, inner_w - 2)

            top_pad = random.randint(1, inner_h - pattern_h - 1)
            left_pad = random.randint(1, inner_w - pattern_w - 1)

            side = random.choice(["up", "down", "left", "right"])
            if side == "up" and r0 < pattern_h + 1:
                continue
            if side == "down" and r1 > H - pattern_h - 2:
                continue
            if side == "left" and c0 < pattern_w + 1:
                continue
            if side == "right" and c1 > W - pattern_w - 2:
                continue

            if overlaps((r0, r1, c0, c1)):
                continue

            color = colors.pop(0)
            used.add(color)
            # Draw full frame.
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 + 1 : r1, c0] = color
            grid[r0 + 1 : r1, c1] = color

            # Carve contiguous gap on one side.
            if side == "up":
                max_gap = w - 2
                glen = random.randint(2, max(2, max_gap))
                gstart = random.randint(c0, c1 - glen + 1)
                grid[r0, gstart : gstart + glen] = BG
                # ensure gap segment strictly on border
            elif side == "down":
                max_gap = w - 2
                glen = random.randint(2, max(2, max_gap))
                gstart = random.randint(c0, c1 - glen + 1)
                grid[r1, gstart : gstart + glen] = BG
            elif side == "left":
                max_gap = h - 2
                glen = random.randint(2, max(2, max_gap))
                gstart = random.randint(r0, r1 - glen + 1)
                grid[gstart : gstart + glen, c0] = BG
            else:
                max_gap = h - 2
                glen = random.randint(2, max(2, max_gap))
                gstart = random.randint(r0, r1 - glen + 1)
                grid[gstart : gstart + glen, c1] = BG

            # Fill random colorful connected pattern.
            pr0 = r0 + 1 + top_pad
            pc0 = c0 + 1 + left_pad
            # keep at least one-cell margin to all borders by construction
            for rr in range(pr0, pr0 + pattern_h):
                for cc in range(pc0, pc0 + pattern_w):
                    grid[rr, cc] = random.choice(list(range(1, 10)))

            # randomize pattern with a few recolors and possible holes while keeping connectivity
            for _ in range(random.randint(1, 5)):
                rr = random.randint(pr0, pr0 + pattern_h - 1)
                cc = random.randint(pc0, pc0 + pattern_w - 1)
                grid[rr, cc] = random.choice(list(range(1, 10)))
            break

    return grid


if __name__ == "__main__":
    # Small smoke check
    demo = generate_input()
    _ = main(demo)
    print(demo.shape)
