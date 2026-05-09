# Variant of 7c66cb00
# Dimension: D9 — 源对象：精灵数量与条的对应基数 (meta)
# Change: 2-3个精灵，2-3个条；每个精灵匹配一个条（通过颜色），一个条可接收0-2个精灵... → 大量精灵（4-6个）匹配少量条（2-3个）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the D9 variant
# ----------------------------------
# a) 具体化与行为追踪
#    - 画面仍是 30x24：底部 2-3 条整行横条，顶部是若干双色或单色小精灵。
#    - 保持原题的颜色规则：精灵外层色匹配某条的 fill，落入该条后外层换成 border，
#      内层若存在则换成该条的 fill；顶部原位置被擦除。
#    - D9 的核心变化不改“按颜色匹配到哪条”，而改“同一条接收很多精灵时怎么摆”：
#      原题是每个模式保留原 x 直接下放；本变体中，若同一条匹配到多个模式，则按它们
#      在顶部的从左到右顺序，压缩到条内部，从左侧开始水平排布，模式之间保留 1 列 fill 间隔。
#    - 小例子：若某条原本接收 3 个宽度分别为 3/2/4 的模式，原逻辑会放在 c=2,10,17；
#      新逻辑会放到 c=1,5,8。差异恰好体现在“多对一”场景，单个模式时仍保留原题行为。
# b) 复杂度校准
#    - 变体要求“很多精灵对应少量条”，因此 generator 固定为 4-6 个顶部视觉精灵、2-3 条。
#    - 为避免视觉上退化成原题，强制至少两个条被使用，且至少一个条接收 2 个以上外层精灵；
#      当精灵数 >= 5 时，要求某条至少接收 3 个外层精灵。
#    - 为避免条内压缩排布溢出，精灵宽度收紧到 2-4，内层块保留但概率降低，并在生成阶段
#      显式检查每条压缩后总宽度是否能放进 22 列内部区域。
# c) 隐式假设审查（针对原 main 的相关代码）
#    - bar 检测逻辑依赖“条仍是整行、左右边框、内部 fill”，该前提在变体中不变。
#    - 顶部模式提取逻辑按“颜色连通块 + bbox”工作；这会把双色精灵的内层块也当成独立模式。
#      原 generator 已经顺着这个实现去渲染 inner sub-sprite，本变体保持这一点，不额外改题。
#    - 原代码的 `place_c = pat["c0"]` 依赖“每个模式独立保留原 x 且不会与同条其他模式冲突”。
#      D9 下这条前提不成立，所以改为：单模式仍保留原 x；多模式时按排序后重新分配 c。
#    - 原代码逐格改色，天然避免“border 色恰好等于某个 inner 色时被二次覆盖”的问题。
#      新实现改为先基于原 patch 建 mask 再赋值，保留这个安全性。
# d) 放置合法性
#    - 顶部精灵只要求彼此在 top area 中至少隔 1 个背景格，不再预留输出时的原 x 位置。
#    - 输出合法性由 generator 双重保证：
#      1. 外层精灵高度不超过目标条高度；
#      2. 若存在 inner 子块，其高度也不超过 inner 目标条高度；
#      3. 每条所有匹配模式在压缩布局时的总宽度 <= 22。
# e) 视觉可观察性
#    - 仅靠颜色数值不可见，因此变体的关键规律必须体现在几何上：多个同色外层精灵会在输出里
#      被“收拢”到同一条的左侧并按顺序排开。
#    - generator 会筛掉“压缩后位置刚好等于原位置”的退化样本，确保新旧 solver 输出不同。
# f) 可视化整洁度
#    - 条与条之间仍隔 1 行背景，顶部精灵也保持 1 格空隙；输出里的多个模式只在同一条内
#      单层水平排布，不叠放、不交叉，避免杂乱。
# g) 颜色变化可推断性
#    - 颜色映射规则完全沿用原题：外层色定位条，落入后外层变 border，内层变 fill。
#      变体只增加“同条多对象的排布规则”，不额外引入新的颜色条件分支。


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (4, 6)
BAR_HEIGHT_RANGE = (4, 7)
MAX_THREE_BAR_TOTAL = 16
MIN_TOP_AREA = 12
SPRITE_H_RANGE = (2, 4)
SPRITE_W_RANGE = (2, 4)
INNER_COLOR_PROB = 0.30
GEN_RETRIES = 600


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    current_bar: dict | None = None
    h, w = g.shape
    for r in range(h):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        fill_color, count = Counter(non_bg).most_common(1)[0]
        if int(count) <= int(w * 0.7):
            if current_bar is not None:
                bars.append(current_bar)
                current_bar = None
            continue

        border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
        if current_bar is not None and int(current_bar["fill"]) == int(fill_color):
            current_bar["r_max"] = int(r)
        else:
            if current_bar is not None:
                bars.append(current_bar)
            current_bar = {
                "fill": int(fill_color),
                "border": int(border),
                "r_min": int(r),
                "r_max": int(r),
                "h": 1,
            }
    if current_bar is not None:
        bars.append(current_bar)
    for bar in bars:
        bar["h"] = int(bar["r_max"] - bar["r_min"] + 1)
    return bars


def _find_patterns(g: np.ndarray, bg: int, pattern_area_end: int) -> list[dict]:
    patterns: list[dict] = []
    top = g[:pattern_area_end, :]
    for col in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(col)
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = map(int, bbox_coords(coords))
            bbox = crop_bbox(g, (r0, c0, r1, c1))

            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue

            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_color = next((int(c) for c in colors_in_bbox if int(c) != int(outer_color)), None)
            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": inner_color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )
    return patterns


def _ordered_matching_patterns(patterns: list[dict], bar_fill: int) -> list[dict]:
    return sorted(
        [p for p in patterns if int(p["outer_color"]) == int(bar_fill)],
        key=lambda p: (int(p["c0"]), int(p["r0"]), int(p["w"]), int(p["h"])),
    )


def _arranged_columns(matching: list[dict], width: int) -> list[tuple[dict, int]]:
    if len(matching) <= 1:
        return [(pat, int(pat["c0"])) for pat in matching]

    placements: list[tuple[dict, int]] = []
    next_c = 1
    for pat in matching:
        if next_c + int(pat["w"]) - 1 > width - 2:
            raise AssertionError("Crowded bar layout exceeds the bar interior width")
        placements.append((pat, next_c))
        next_c += int(pat["w"]) + 1
    return placements


def _recolor_pattern(pat: dict, bar: dict) -> np.ndarray:
    src = np.array(pat["content"], dtype=int)
    transformed = src.copy()
    outer_mask = src == int(pat["outer_color"])
    transformed[outer_mask] = int(bar["border"])
    if pat["inner_color"] is not None:
        transformed[src == int(pat["inner_color"])] = int(bar["fill"])
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)

    bars = _find_bars(g, bg)
    assert bars, "No bars found — grid must contain at least one horizontal bar"

    pattern_area_end = min(int(bar["r_min"]) for bar in bars)
    patterns = _find_patterns(g, bg, pattern_area_end)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for bar in bars:
        matching = _ordered_matching_patterns(patterns, int(bar["fill"]))
        for pat, place_c in _arranged_columns(matching, g.shape[1]):
            transformed = _recolor_pattern(pat, bar)
            place_r = int(bar["r_max"] - pat["h"] + 1)
            paste(out, transformed, place_r, int(place_c), transparent=int(bg))

    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = r0 - 1
    return grid, bars


def _sample_outer_counts(n_bars: int, n_sprites: int) -> list[int]:
    while True:
        counts = [0] * n_bars
        for idx in random.sample(range(n_bars), k=2):
            counts[idx] += 1
        for _ in range(n_sprites - 2):
            counts[random.randrange(n_bars)] += 1
        if max(counts) < 2:
            continue
        if n_sprites >= 5 and max(counts) < 3:
            continue
        return counts


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    outer_fill = int(bars[outer_idx]["fill"])
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"])))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_idx = random.choice([i for i in range(len(bars)) if i != outer_idx])
        max_ih = min(h - 2, int(bars[inner_idx]["h"]))
        valid_inner = [
            (ih, iw)
            for ih in range(1, max_ih + 1)
            for iw in range(1, w - 1)
            if ih * iw * 2 < h * w
        ]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_idx]["fill"])
            inner_bbox = (ir0, ic0, ih, iw)
        else:
            inner_idx = None

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": inner_idx,
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_top_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    ordered = sorted(sprites, key=lambda s: (-int(s["h"] * s["w"]), -int(s["w"]), -int(s["h"])))
    for sp in ordered:
        h, w = int(sp["h"]), int(sp["w"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, top_h - h + 1):
            for c0 in range(1, GRID_W - w):
                if _rect_separated(rects, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return placed


def _collect_patterns_from_placed(placed: list[dict], bars: list[dict]) -> list[dict]:
    patterns: list[dict] = []
    for sp in placed:
        outer_fill = int(bars[int(sp["outer_idx"])]["fill"])
        inner_color = None
        if sp["inner_idx"] is not None:
            inner_color = int(bars[int(sp["inner_idx"])]["fill"])
        patterns.append(
            {
                "outer_color": outer_fill,
                "inner_color": inner_color,
                "r0": int(sp["r0"]),
                "c0": int(sp["c0"]),
                "h": int(sp["h"]),
                "w": int(sp["w"]),
                "content": np.array(sp["patch"], dtype=int).copy(),
            }
        )

        if sp["inner_idx"] is not None:
            ir0, ic0, ih, iw = sp["inner_bbox"]
            inner_patch = np.array(sp["patch"], dtype=int)[ir0 : ir0 + ih, ic0 : ic0 + iw].copy()
            patterns.append(
                {
                    "outer_color": int(bars[int(sp["inner_idx"])]["fill"]),
                    "inner_color": None,
                    "r0": int(sp["r0"] + ir0),
                    "c0": int(sp["c0"] + ic0),
                    "h": int(ih),
                    "w": int(iw),
                    "content": inner_patch,
                }
            )
    return patterns


def _compact_layout_fits(patterns: list[dict], bars: list[dict]) -> bool:
    interior_w = GRID_W - 2
    for bar in bars:
        matching = _ordered_matching_patterns(patterns, int(bar["fill"]))
        if len(matching) <= 1:
            continue
        needed = sum(int(p["w"]) for p in matching) + len(matching) - 1
        if needed > interior_w:
            return False
    return True


def _variant_visible(patterns: list[dict], bars: list[dict]) -> bool:
    crowded = False
    changed = False
    for bar in bars:
        matching = _ordered_matching_patterns(patterns, int(bar["fill"]))
        if len(matching) <= 1:
            continue
        crowded = True
        next_c = 1
        for pat in matching:
            if int(pat["c0"]) != next_c:
                changed = True
            next_c += int(pat["w"]) + 1
    return crowded and changed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    inp = base.copy()
    for sp in placed:
        paste(inp, np.array(sp["patch"], dtype=int), int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

    patterns = _collect_patterns_from_placed(placed, bars)
    out = base.copy()
    for bar in bars:
        matching = _ordered_matching_patterns(patterns, int(bar["fill"]))
        for pat, place_c in _arranged_columns(matching, GRID_W):
            transformed = _recolor_pattern(pat, bar)
            place_r = int(bar["r_max"] - pat["h"] + 1)
            paste(out, transformed, place_r, int(place_c), transparent=int(bg))
    return inp, out, patterns


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    outer_counts = _sample_outer_counts(n_bars, n_sprites)
    outer_indices = [idx for idx, count in enumerate(outer_counts) for _ in range(count)]
    random.shuffle(outer_indices)
    sprites = [_make_sprite(idx, bars) for idx in outer_indices]

    placed = _place_top_sprites(sprites, top_h)
    if placed is None:
        return None

    patterns = _collect_patterns_from_placed(placed, bars)
    if not _compact_layout_fits(patterns, bars):
        return None
    if not _variant_visible(patterns, bars):
        return None

    inp, out, _ = _render(base, placed, bars, bg)
    if get_bg(inp) != bg:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
