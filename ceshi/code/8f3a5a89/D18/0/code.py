# Variant of 8f3a5a89
# Dimension: D18 — 墙壁与边缘的关系 (obstacle)
# Change: 结构墙必须触碰至少一条网格边缘，与网格边缘协同形成可达区域的封闭边界... → 结构墙不需触碰边缘（浮动墙壁）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D18 variant
# a) 具体化与行为追踪
#    - 原题里，真正的蓝墙总是贴着网格边缘；从左下角品红像素出发，边缘 + 蓝墙一起围出可达区。
#    - 变体里，真正的蓝墙改成“浮动闭环”：它完全可以悬空放在网格内部，自己围出一个不可达内腔。
#    - 小例子：若输入中央有一个蓝色空心矩形环，原 solver 会因为它不碰边缘而把它当装饰，洪填会穿过它；
#      变体 solver 则必须识别它本身围住了一块内腔，于是把整圈蓝墙当结构墙，橙色只画在外侧可达一圈，内腔中的装饰墙被擦掉。
# b) 复杂度校准
#    - 若允许任意复杂闭环与任意装饰，少量样例很难归纳；因此生成器固定为 1-3 个“空心矩形环”结构墙。
#    - 这样“闭合 vs 非闭合”在视觉上很稳定：结构墙总是能看出围栏，装饰墙始终只是小块实心斑点或短条。
# c) 隐式假设审查（原 main 中与 D18 相关的失效点）
#    - 原逻辑用 touches_border 判定结构墙；在本变体中这是错误的，因为真正的结构墙通常不接触边缘。
#    - 原逻辑先把所有蓝色都当障碍做一次 reach0，再保留“碰边且靠近 reach0”的分量；浮动闭环会被全部漏掉。
#    - 原逻辑因此会允许洪填穿过闭环，把内腔装饰误判为可达并保留，输出退化成原图加边缘橙框。
#    - 新逻辑改为：逐个蓝色 4 连通分量测试“仅靠该分量自己，是否围出了不与网格边缘连通的区域”；若是，则它是结构墙。
# d) 放置合法性
#    - 结构墙采用 1 像素厚矩形环，且与网格四边至少隔 3 格，保证它真的是“浮动”而非借边围栏。
#    - 不同结构环之间保持至少 1 格缓冲，避免合并成一个分量或互相嵌套。
#    - 隐藏装饰只放在环内部不可达区；可见装饰只放在环外可达区；两者都用小实心块，避免意外形成闭环。
# e) 视觉可观察性
#    - 由于结构环离边缘有明显空隙，输出会出现“边缘橙框 + 内部额外橙环”，这与原题边缘依赖的视觉模式明显不同。
#    - 2-4 个样例里，只要看到蓝色环本身没碰边却仍然阻挡洪填，人类就能归纳出新规则。
# f) 可视化整洁度
#    - 只使用四种颜色且图层关系稳定：蓝色环、背景、角落品红、输出橙边。
#    - 装饰数量少且尺寸小，不与结构环重叠，也不在同一区域堆叠多层信息。
# g) 颜色变化可推断性
#    - 颜色规则不变：蓝是墙、品红是起点、橙是输出边界、青色是背景。
#    - 变化来自几何关系（闭合围栏）而非颜色 id 的隐藏排序。


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _enclosed_mask_by_component(comp_mask: np.ndarray) -> np.ndarray:
    """Return cells enclosed by this wall component alone.

    During this test every non-component cell is treated as open, even if it is also blue.
    This lets hidden decoration walls stay inside a structural loop without breaking closure
    detection for the loop component itself.
    """
    h, w = comp_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not comp_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not comp_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not comp_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return (~comp_mask) & (~outside)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the reachable-side border around floating closed walls found from a corner marker.

    Input structure:
    - bg is the dominant color; blue pixels mix true fences and decorative wall-colored pieces.
    - A unique magenta single pixel marks the start corner, still placed on a grid corner.
    - True structural walls may float entirely inside the canvas; they are the blue 4-connected
      components that close off their own interior pocket.

    Rule:
    - Identify which blue components are self-closed fences by checking whether each one encloses
      at least one non-border-connected cell when considered as the only obstacle.
    - Flood-fill from the marker through background and through non-structural blue decorations;
      structural closed fences block the flood.
    - Erase decorative blue pixels that lie on the unreachable side, then paint orange on every
      reachable background cell on the grid edge, next to a structural wall, or next to an
      unreachable cell.

    Key insight:
    Border contact no longer matters; the decisive feature is whether a blue component forms a
    closed loop that creates its own sealed pocket.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    structural = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if _enclosed_mask_by_component(comp_mask).any():
            structural |= comp_mask

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and not structural[r, c]),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and not structural[r, c] and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any(structural[nr, nc] for nr, nc in neighbors8(r, c) if 0 <= nr < h and 0 <= nc < w):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w):
            out[r, c] = fill_col

    return out


# ---- Palette (task colors) ----
BG = 8
WALL = 1
MARKER = 6
FILL = 7

# ---- Canvas size ----
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

# ---- Structural walls: floating rectangular rings ----
NUM_STRUCTS_RANGE = (1, 3)
STRUCT_RETRIES = 30
FLOAT_GAP = 3
LOOP_MIN_OUTER = 4

# ---- Reachable region gate ----
REACH_RATIO_RANGE = (0.38, 0.87)

# ---- Decorations ----
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

# ---- Outer retry budget ----
GEN_ATTEMPTS = 500


def _ring_masks(r0: int, c0: int, oh: int, ow: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ring = np.zeros((h, w), dtype=bool)
    inner = np.zeros((h, w), dtype=bool)
    r1 = r0 + oh - 1
    c1 = c0 + ow - 1
    ring[r0, c0:c1 + 1] = True
    ring[r1, c0:c1 + 1] = True
    ring[r0:r1 + 1, c0] = True
    ring[r0:r1 + 1, c1] = True
    if oh > 2 and ow > 2:
        inner[r0 + 1:r1, c0 + 1:c1] = True
    return ring, inner


def _place_structural_walls(h: int, w: int, marker_rc: tuple[int, int], rng) -> tuple[np.ndarray, np.ndarray] | None:
    max_oh = h - 2 * FLOAT_GAP
    max_ow = w - 2 * FLOAT_GAP
    if max_oh < LOOP_MIN_OUTER or max_ow < LOOP_MIN_OUTER:
        return None

    structural = np.zeros((h, w), dtype=bool)
    enclosed = np.zeros((h, w), dtype=bool)

    desired = 1 if min(h, w) <= 12 else rng.randint(1, 2)
    placed = 0

    for idx in range(desired):
        placed_this_round = False
        for _ in range(STRUCT_RETRIES):
            if idx == 0:
                base_div = 2 if desired == 1 else 3
                min_oh = min(max_oh, max(LOOP_MIN_OUTER, h // base_div))
                min_ow = min(max_ow, max(LOOP_MIN_OUTER, w // base_div))
            else:
                min_oh = min(max_oh, max(LOOP_MIN_OUTER, h // 4))
                min_ow = min(max_ow, max(LOOP_MIN_OUTER, w // 4))

            oh = rng.randint(min_oh, max_oh)
            ow = rng.randint(min_ow, max_ow)
            r0 = rng.randint(FLOAT_GAP, h - FLOAT_GAP - oh)
            c0 = rng.randint(FLOAT_GAP, w - FLOAT_GAP - ow)
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1

            if (r0 <= marker_rc[0] <= r1) and (c0 <= marker_rc[1] <= c1):
                continue

            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(h, r1 + 2)
            pad_c1 = min(w, c1 + 2)
            if (structural | enclosed)[pad_r0:pad_r1, pad_c0:pad_c1].any():
                continue

            ring, inner = _ring_masks(r0, c0, oh, ow, h, w)
            if not inner.any():
                continue

            structural |= ring
            enclosed |= inner
            placed += 1
            placed_this_round = True
            break

        if not placed_this_round and placed == 0:
            return None
        if not placed_this_round:
            break

    return (structural, enclosed) if placed > 0 else None


def _validate_structural(structural: np.ndarray) -> bool:
    h, w = structural.shape
    for comp_arr in components_mask(structural, connectivity=4):
        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if comp_mask[0, :].any() or comp_mask[-1, :].any() or comp_mask[:, 0].any() or comp_mask[:, -1].any():
            return False
        if not _enclosed_mask_by_component(comp_mask).any():
            return False
    return True


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)
    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)
    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _place_decor(allowed_mask: np.ndarray, count: int, rng) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    placed = _place_structural_walls(h, w, marker_rc, rng)
    if placed is None:
        return None
    structural, enclosed = placed

    if not _validate_structural(structural):
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    out = _paint_output(structural, reach, marker_rc)

    if len(components_mask(structural, connectivity=4)) == 0:
        return None
    # Core D18 visibility gate: at least one structural wall must float away from all edges.
    if np.any(structural[0, :]) or np.any(structural[-1, :]) or np.any(structural[:, 0]) or np.any(structural[:, -1]):
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_hidden = rng.randint(0, n_decor)
    n_visible = n_decor - n_hidden

    visible_allowed = reach & (out == BG)
    hidden_allowed = enclosed.copy()

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    if hidden_placed.all() and hidden_allowed.any():
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    if not np.array_equal(main(inp), out_final):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D18 floating-wall variant")
