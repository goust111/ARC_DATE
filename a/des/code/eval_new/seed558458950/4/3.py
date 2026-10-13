import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np


def _most_frequent_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator_row(grid: np.ndarray, bg: int) -> Tuple[int, int]:
    h, w = grid.shape
    separator = (-1, -1, -1)  # score, row, color
    for r in range(h):
        row_color = int(grid[r, 0])
        if np.all(grid[r] == row_color):
            top_non_bg = np.count_nonzero(grid[:r] != bg)
            bot_non_bg = np.count_nonzero(grid[r + 1 :] != bg) if r + 1 < h else 0
            if r > 0 and r + 1 < h and top_non_bg > 0 and bot_non_bg > 0:
                score = top_non_bg + bot_non_bg
                if score > separator[0]:
                    separator = (score, r, row_color)
    if separator[0] >= 0:
        return separator[1], separator[2]
    # Fallback: any full row that is not on edges
    for r in range(1, h - 1):
        row_color = int(grid[r, 0])
        if np.all(grid[r] == row_color):
            return r, row_color
    raise ValueError("No separator row found")


def _ordered_path(path_set: set, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    while True:
        r, c = cur
        next_points = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in path_set and nxt != prev:
                next_points.append(nxt)
        if not next_points:
            break
        nxt = next_points[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    return ordered


def _stamp_with_patch(out: np.ndarray, center_r: int, center_c: int, patch: np.ndarray) -> None:
    ph, pw = patch.shape
    start_r = center_r - ph // 2
    start_c = center_c - pw // 2
    dst_r0 = max(0, start_r)
    dst_c0 = max(0, start_c)
    dst_r1 = min(out.shape[0], start_r + ph)
    dst_c1 = min(out.shape[1], start_c + pw)
    if dst_r0 >= dst_r1 or dst_c0 >= dst_c1:
        return
    src_r0 = dst_r0 - start_r
    src_c0 = dst_c0 - start_c
    src_r1 = src_r0 + (dst_r1 - dst_r0)
    src_c1 = src_c0 + (dst_c1 - dst_c0)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = patch[src_r0:src_r1, src_c0:src_c1]


def _tile_positions_along_path(path: List[Tuple[int, int]], patch: np.ndarray) -> List[Tuple[int, int]]:
    if len(path) == 0:
        return []
    if len(path) == 1:
        return [path[0]]

    centers: List[Tuple[int, int]] = [path[0]]
    seen = {path[0]}
    i = 0
    while i < len(path) - 1:
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        dr, dc = r1 - r0, c1 - c0
        j = i + 1
        while j < len(path):
            pr, pc = path[j - 1]
            cr, cc = path[j]
            if cr - pr == dr and cc - pc == dc:
                j += 1
            else:
                break

        run_len = j - i  # number of nodes in this direction run
        stride = patch.shape[1] if dr == 0 else patch.shape[0]
        step = stride
        while i + step < j:
            pos = path[i + step]
            if pos not in seen:
                centers.append(pos)
                seen.add(pos)
            step += stride

        i = j - 1
        if path[i] not in seen:
            centers.append(path[i])
            seen.add(path[i])

    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """用下方按颜色匹配的路径把上方纹理块平铺到路径所在位置并清除原纹理源。

Input structure:
- bg 为最常见颜色，输入分为上、下两部分，由一条完整的灰色水平线隔开。
- 上半区是若干单一主色的矩形纹理块，内部可包含同一非 bg 小纹理。
- 下半区每个颜色对应一条单像素折线路径，路径与上半区相同颜色；端点旁有一枚 yellow 标记箭头。

Rule:
- 找到分隔线并按其上下切分网格。
- 提取上半区每个连通非 bg 区域的外接框作为纹理块，并用该区域出现次数最多的颜色作为其主色。
- 在下半区按主色提取路径，依据有 yellow 标记的端点决定起点（远离箭头的端点），按路径顺序铺设该纹理。
- 铺设时纹理中心对齐起点后，沿路径每段前进：横向以纹理宽度为步长，纵向以纹理高度为步长。
- 最终上半区全部清空为 bg，分隔线保留，路径被盖印内容覆盖。

Key insight:
- 关键在于先按颜色建立源纹理与路径的对齐关系，再利用箭头端点确定起点方向，最后按方向步长进行无重叠拼接。
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g.copy()

    h, w = g.shape
    bg = _most_frequent_color(g)
    sep_row, sep_color = _find_separator_row(g, bg)
    yellow = 9

    out = g.copy()
    if sep_row <= 0 or sep_row >= h - 1:
        return out

    # Detect textured sources in the top region
    top = g[:sep_row, :]
    source_patches: Dict[int, np.ndarray] = {}
    top_mask = top != bg
    visited = np.zeros_like(top_mask, dtype=bool)
    for r in range(top.shape[0]):
        for c in range(top.shape[1]):
            if not top_mask[r, c] or visited[r, c]:
                continue
            q: Deque[Tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            comp: List[Tuple[int, int]] = []
            color_count: Dict[int, int] = {}
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                color = int(top[cr, cc])
                color_count[color] = color_count.get(color, 0) + 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < top.shape[0] and 0 <= nc < w and top_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            main_color = max(color_count.items(), key=lambda kv: kv[1])[0]
            rs = min(v[0] for v in comp)
            re = max(v[0] for v in comp)
            cs = min(v[1] for v in comp)
            ce = max(v[1] for v in comp)
            patch = top[rs : re + 1, cs : ce + 1]
            source_patches[main_color] = patch.copy()

    # Clear top region first
    out[:sep_row, :] = bg

    # Trace each path in the bottom region and stamp the corresponding patch.
    visited = np.zeros_like(g, dtype=bool)
    for r in range(sep_row + 1, h):
        for c in range(w):
            color = int(g[r, c])
            if color in (bg, sep_color, yellow) or visited[r, c] or color not in source_patches:
                continue

            q: Deque[Tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            component: List[Tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                component.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (sep_row < nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            comp_set = set(component)
            endpoints = []
            for p in component:
                pr, pc = p
                degree = 0
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (pr + dr, pc + dc) in comp_set:
                        degree += 1
                if degree == 1:
                    endpoints.append(p)

            if not endpoints:
                continue

            # Select start point by far endpoint from the yellow marker (if available)
            start = None
            arrow_endpoint = None
            for endpoint in endpoints:
                pr, pc = endpoint
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == yellow:
                        arrow_endpoint = endpoint
                        break
                if arrow_endpoint is not None:
                    break

            if len(endpoints) == 1:
                start = endpoints[0]
            elif arrow_endpoint is not None:
                start = endpoints[0] if endpoints[0] != arrow_endpoint else endpoints[1]
            else:
                start = endpoints[0]

            ordered = _ordered_path(comp_set, start)
            if not ordered:
                continue

            patch = source_patches[color]
            centers = _tile_positions_along_path(ordered, patch)
            for rr, cc in centers:
                _stamp_with_patch(out, rr, cc, patch)

    return out


def _random_path_for_color(
    h: int,
    w: int,
    sep_row: int,
    block_h: int,
    block_w: int,
    occupied: np.ndarray,
    color: int,
) -> Tuple[bool, List[Tuple[int, int]], Tuple[int, int]]:
    step_h = block_h
    step_w = block_w
    segment_count = random.choice([1, 2, 2, 3])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for _ in range(160):
        start_r = random.randint(sep_row + 2, h - 3)
        start_c = random.randint(1, w - 2)
        if occupied[start_r, start_c]:
            continue

        path: List[Tuple[int, int]] = [(start_r, start_c)]
        temp_occupied = occupied.copy()
        temp_occupied[start_r, start_c] = True
        prev_dir = None
        ok = True

        for _seg in range(segment_count):
            cur_choices = directions if prev_dir is None else [d for d in directions if d != (-prev_dir[0], -prev_dir[1])]
            chosen = False
            for _ in range(16):
                dr, dc = random.choice(cur_choices)
                stride = step_w if dr == 0 else step_h
                units = random.randint(1, 3) * stride
                candidates = [(start_r, start_c)]
                cur_r, cur_c = path[-1]
                good = True

                for _ in range(units):
                    nr = cur_r + dr
                    nc = cur_c + dc
                    if not (sep_row + 1 < nr < h - 1 and 0 <= nc < w and 0 <= nc < w):
                        good = False
                        break
                    if nr <= sep_row or nc <= 0 or nc >= w - 1:
                        good = False
                        break
                    if temp_occupied[nr, nc]:
                        good = False
                        break
                    candidates.append((nr, nc))
                    temp_occupied[nr, nc] = True
                    cur_r, cur_c = nr, nc

                if good and len(candidates) > 1:
                    path.extend(candidates[1:])
                    prev_dir = (dr, dc)
                    chosen = True
                    break

            if not chosen:
                ok = False
                break

        if not ok or len(path) < 2:
            continue

        # Pick arrow endpoint and ensure marker placement outside path.
        endpoint_order = [0, len(path) - 1] if random.random() < 0.5 else [len(path) - 1, 0]
        arrow_pos = None
        for end_idx in endpoint_order:
            end_r, end_c = path[end_idx]
            nbr_idx = 1 if end_idx == 0 else len(path) - 2
            nr, nc = path[nbr_idx]
            ay = end_r - (nr - end_r)
            ax = end_c - (nc - end_c)
            if ay <= sep_row or ay >= h - 1:
                continue
            if not (0 <= ax < w):
                continue
            if occupied[ay, ax] or temp_occupied[ay, ax]:
                continue
            arrow_pos = (ay, ax)
            break

        if arrow_pos is None:
            continue

        return True, path, arrow_pos

    return False, [], (0, 0)


def _paint_texture(g: np.ndarray, r0: int, c0: int, h: int, w: int, base: int, alt_colors: List[int]) -> None:
    if h < 3 or w < 3:
        return
    if len(alt_colors) == 0:
        return

    inner_r0 = r0 + 1
    inner_r1 = r0 + h - 1
    inner_c0 = c0 + 1
    inner_c1 = c0 + w - 1
    if inner_r0 >= inner_r1 or inner_c0 >= inner_c1:
        return

    mode = random.choice(["none", "stripe_h", "stripe_v", "checker", "dots"])
    if mode == "none":
        return

    if mode == "stripe_h":
        accent = random.choice(alt_colors)
        for rr in range(inner_r0, inner_r1):
            if (rr - r0) % 2 == 0:
                g[rr, inner_c0:inner_c1] = accent
        return

    if mode == "stripe_v":
        accent = random.choice(alt_colors)
        for cc in range(inner_c0, inner_c1):
            if (cc - c0) % 2 == 0:
                g[inner_r0:inner_r1, cc] = accent
        return

    if mode == "checker":
        accent = random.choice(alt_colors)
        for rr in range(inner_r0, inner_r1):
            for cc in range(inner_c0, inner_c1):
                if (rr + cc - r0 - c0) % 2 == 0:
                    g[rr, cc] = accent
        return

    if mode == "dots":
        accent = random.choice(alt_colors)
        for rr in range(inner_r0, inner_r1):
            for cc in range(inner_c0, inner_c1):
                if random.random() < 0.32:
                    g[rr, cc] = accent


def generate_input() -> np.ndarray:
    bg = 0
    sep = 5
    yellow = 9
    palette = [1, 2, 3, 4, 6, 7, 8]

    w = random.randint(16, 30)
    top_h = random.randint(6, 11)
    bottom_h = random.randint(6, 11)
    h = top_h + bottom_h + 1

    g = np.full((h, w), bg, dtype=int)
    sep_row = top_h
    g[sep_row, :] = sep

    top_occ = np.zeros((top_h, w), dtype=bool)
    bottom_occ = np.zeros((h, w), dtype=bool)
    bottom_occ[sep_row, :] = True
    for rr in range(h):
        for cc in range(w):
            if rr != sep_row and g[rr, cc] != bg:
                bottom_occ[rr, cc] = True

    target_pairs = random.randint(1, min(3, len(palette)))
    colors = random.sample(palette, target_pairs)
    placed = 0

    for color in colors:
        success = False
        for _ in range(120):
            bh = random.choice([3, 4, 5, 6])
            bw = random.choice([3, 4, 5, 6])
            if top_h <= bh + 1 or w <= bw + 1:
                continue
            # keep rectangles inside top area with small spacing
            r0 = random.randint(1, top_h - bh - 1)
            c0 = random.randint(1, w - bw - 1)
            r1 = r0 + bh
            c1 = c0 + bw
            pad_r0 = max(0, r0 - 1)
            pad_r1 = min(top_h, r1 + 1)
            pad_c0 = max(0, c0 - 1)
            pad_c1 = min(w, c1 + 1)
            if np.any(top_occ[pad_r0:pad_r1, pad_c0:pad_c1]):
                continue

            # Try to create a path first (dry run), ensure dimensions are compatible.
            path_ok, path_cells, arrow_pos = _random_path_for_color(
                h,
                w,
                sep_row,
                bh,
                bw,
                bottom_occ,
                color,
            )
            if not path_ok:
                continue

            # Place source rectangle
            g[r0:r1, c0:c1] = color
            alt_colors = [c for c in palette if c != color]
            _paint_texture(g, r0, c0, bh, bw, color, alt_colors)
            top_occ[r0:r1, c0:c1] = True
            top_occ[pad_r0:pad_r1, pad_c0:pad_c1] = True

            # Place path and marker
            for rr, cc in path_cells:
                g[rr, cc] = color
                bottom_occ[rr, cc] = True
            ar, ac = arrow_pos
            g[ar, ac] = yellow
            bottom_occ[ar, ac] = True
            placed += 1
            success = True
            break

        if not success:
            continue

    if placed == 0:
        # Guaranteed fallback sample with minimal valid structure.
        if top_h > 7 and w > 10 and h - sep_row > 7:
            color = palette[0]
            r0 = 1
            c0 = 1
            bh, bw = 5, 5
            g[r0:r0 + bh, c0:c0 + bw] = color
            _paint_texture(g, r0, c0, bh, bw, color, [c for c in palette if c != color])

            start_r = sep_row + 3
            start_c = 3
            for cc in range(start_c, start_c + 12):
                g[start_r, cc] = color
            g[start_r, start_c - 1] = yellow

    return g

