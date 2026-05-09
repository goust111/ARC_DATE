from __future__ import annotations

# Variant of 7c66cb00
# Dimension: D5 — 目标对象：条的排列与间距 (dst)
# Change: 条从底部向上堆叠，条间恰好间隔1行背景色，最底部条紧贴网格底边...
#         → 条间间隔改为 2-4 行不等，并且“某条下方的间隔”编码该条内精灵离底的上移量。

import os
import random
import sys
from collections import Counter

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste

"""
design_analysis:
  a_specific_behavior:
    old_rule:
      - 横条自下而上堆叠，相邻条之间固定 1 行背景色。
      - 每个精灵按外层颜色匹配某条后，始终贴到该条底边。
    new_rule:
      - 横条之间改为 2/3/4 行背景色。
      - 某条正下方的空隙大小，决定该条内匹配精灵离底上移多少行：
        gap_below 2 -> lift 0
        gap_below 3 -> lift 1
        gap_below 4 -> lift 2
      - 最底部条没有“下方条间距”，因此保持 lift 0。
    concrete_trace:
      - 例：上条高度 5，下一条在它下方留 4 行空白。原题会把 3x4 精灵放到
        place_r = r_max - 3 + 1；变体改为 place_r = r_max - 2 - 3 + 1。
      - 视觉上，输出中该精灵下方会多出 2 行条的填充色边距，且顶部原精灵仍被擦除。
  b_complexity:
    - 只引入 3 个 lift 档位，复杂度有限但肉眼可见。
    - 生成器强制至少一个非底部条的 gap >= 3，并且所有条都有精灵参与，
      避免退化成“所有精灵仍然贴底”的原题外观。
  c_assumption_audit:
    - 原 bar 检测按“整行大部分为同一非背景色”识别横条，不依赖 gap=1，保留。
    - pattern_area_end = 第一条顶部之前的区域，仍成立。
    - 唯一失效的旧假设是 `place_r = bar["r_max"] - h + 1` 无条件底对齐；
      变体必须改为减去该条的 lift。
    - 内层颜色子组件独立停靠到其对应条，这条副规则保留，但也必须使用目标条自己的 lift。
  d_legality:
    - 生成器先采样 bar heights 与 gaps，确保 sum(heights)+sum(gaps)+top_area <= 30。
    - 每条 bar 的 `capacity = height - lift`，所有被使用条都要求 capacity >= 2，
      这样外层精灵能在条内完整落下。
    - 横向占用继续按 target bar 记录，避免同一条中的多个停靠对象 x 重叠。
  e_visual_observability:
    - 人类在输入里能直接看到不同条之间留出的 2-4 行空白。
    - 在输出里，对应条内的彩色块会悬浮在不同高度；空隙越大，悬浮越高。
    - 颜色交换规则保持原样，因此“高度变化来自间距而不是颜色”可被分离观察。
  f_cleanliness:
    - 顶部精灵仍位于独立区域，条区仍是大块水平带，避免多层叠加造成杂乱。
    - 内层子块只在自己的条内产生一个小矩形，不与其他目标发生横向冲突。
  g_color_inference:
    - 颜色总数与原题同量级，颜色变化只沿用原题的“外层->边框、内层->填充”。
    - 新增规则完全由可见距离驱动，不依赖颜色编号或其他不可见属性。
  new_main_assumptions_checked:
    - solver 假设 bars 按从上到下顺序可被扫描到，并且相邻条之间至少 1 行背景。
      generator 构造 2-4 行，因此满足。
    - solver 假设某条的目标对象高度 <= `bar.height - bar.lift`。
      generator 用 capacity 约束 sprite 尺寸与 inner patch 高度，因此满足。
    - solver 假设每条 fill 色唯一，以便颜色匹配唯一。
      generator 继续为每条条带使用互异 fill。
"""


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
GAP_RANGE = (2, 4)
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 400


def _gap_to_lift(gap: int) -> int:
    """Map visible inter-bar gap size to an in-bar upward lift."""
    return max(0, int(gap) - 2)


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect horizontal bars from top to bottom and annotate their gap-coded lift."""
    H, W = grid.shape
    bars: list[dict] = []
    current_bar: dict | None = None

    for r in range(H):
        row = grid[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(W * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }
            elif current_bar:
                bars.append(current_bar)
                current_bar = None
        elif current_bar:
            bars.append(current_bar)
            current_bar = None

    if current_bar:
        bars.append(current_bar)

    for i, bar in enumerate(bars):
        if i + 1 < len(bars):
            gap_below = int(bars[i + 1]["r_min"] - bar["r_max"] - 1)
        else:
            gap_below = int(H - bar["r_max"] - 1)
        bar["h"] = int(bar["r_max"] - bar["r_min"] + 1)
        bar["gap_below"] = gap_below
        bar["lift"] = _gap_to_lift(gap_below)
        bar["capacity"] = int(bar["h"] - bar["lift"])
    return bars


def _extract_patterns(grid: np.ndarray, bg: int, pattern_area_end: int) -> list[dict]:
    """Extract top-area sprite bboxes the same way as the original solver."""
    patterns: list[dict] = []
    top = grid[:pattern_area_end, :]
    for col in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(col)
        if not bool(mask.any()):
            continue
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = map(int, bbox_coords(coords))
            bbox = crop_bbox(grid, (r0, c0, r1, c1))
            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue
            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_colors = [int(c) for c in colors_in_bbox if int(c) != int(outer_color)]
            patterns.append(
                {
                    "outer_color": outer_color,
                    "inner_color": inner_colors[0] if inner_colors else None,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )
    return patterns


def _recolor_pattern(pattern: dict, bar: dict) -> np.ndarray:
    transformed = np.array(pattern["content"], dtype=int).copy()
    outer = int(pattern["outer_color"])
    inner = pattern["inner_color"]
    transformed[transformed == outer] = int(bar["border"])
    if inner is not None:
        transformed[transformed == int(inner)] = int(bar["fill"])
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched horizontal bars with a color swap.

    Variant rule:
    - Keep the original color matching and recoloring.
    - Replace unconditional bottom-alignment with gap-coded lift:
      the gap below a bar (2/3/4 rows) maps to lift 0/1/2 for objects docked in that bar.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    if not bars:
        return out

    pattern_area_end = int(min(int(b["r_min"]) for b in bars))
    patterns = _extract_patterns(g, bg, pattern_area_end)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for bar in bars:
        for pat in patterns:
            if int(pat["outer_color"]) != int(bar["fill"]):
                continue
            transformed = _recolor_pattern(pat, bar)
            place_r = int(bar["r_max"] - bar["lift"] - pat["h"] + 1)
            if place_r < int(bar["r_min"]):
                continue
            paste(out, transformed, place_r, int(pat["c0"]), transparent=int(bg))

    return out


def _sample_layout(n_bars: int) -> tuple[list[int], list[int]]:
    """Sample bar heights and inter-bar gaps while leaving a sprite area on top."""
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        gaps = [random.randint(*GAP_RANGE) for _ in range(n_bars - 1)]
        if gaps and max(gaps) < 3:
            continue
        total = sum(heights) + sum(gaps)
        if GRID_H - total < MIN_TOP_AREA:
            continue
        lifts = [_gap_to_lift(g) for g in gaps] + [0]
        capacities = [h - lift for h, lift in zip(heights, lifts)]
        if min(capacities) < SPRITE_H_RANGE[0]:
            continue
        return heights, gaps


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int], gaps: list[int]) -> tuple[np.ndarray, list[dict]]:
    """Render bars bottom-up, with variable gaps encoding each upper bar's lift."""
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict | None] = [None] * len(heights)
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
            "h": h,
            "r_min": int(r0),
            "r_max": int(r1),
        }
        if i > 0:
            r = r0 - int(gaps[i - 1])
    out_bars = [bar for bar in bars if bar is not None]
    for i, bar in enumerate(out_bars):
        if i + 1 < len(out_bars):
            gap_below = int(out_bars[i + 1]["r_min"] - bar["r_max"] - 1)
        else:
            gap_below = int(GRID_H - bar["r_max"] - 1)
        bar["gap_below"] = gap_below
        bar["lift"] = _gap_to_lift(gap_below)
        bar["capacity"] = int(bar["h"] - bar["lift"])
    return grid, out_bars


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """Create an outer rectangle with an optional smaller inner-color region."""
    outer_fill = int(bars[outer_idx]["fill"])
    outer_capacity = int(bars[outer_idx]["capacity"])
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], outer_capacity))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        random.shuffle(inner_choices)
        for candidate in inner_choices:
            inner_capacity = int(bars[candidate]["capacity"])
            valid_inner = [
                (ih, iw)
                for ih in range(1, min(h - 2, inner_capacity) + 1)
                for iw in range(1, w - 1)
                if (w - iw) % 2 == 0 and ih * iw * 2 < h * w
            ]
            if not valid_inner:
                continue
            inner_idx = candidate
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_idx]["fill"])
            inner_bbox = (ir0, ic0, ih, iw)
            break

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": None if inner_idx is None else int(inner_idx),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[start : start + length].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    """Find top-area placements while reserving x-ranges needed in each destination bar."""
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx = int(sp["outer_idx"])
        inner_idx = sp["inner_idx"]
        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            if inner_idx is not None:
                _, ic0, _, iw = sp["inner_bbox"]
                if not _interval_free(bar_occ[int(inner_idx)], c0 + int(ic0), int(iw)):
                    continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0 : c0 + w] = True
        if inner_idx is not None:
            _, ic0, _, iw = sp["inner_bbox"]
            bar_occ[int(inner_idx)][c0 + int(ic0) : c0 + int(ic0) + int(iw)] = True
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Render the input grid and the expected lifted output grid."""
    inp, out = base.copy(), base.copy()
    for sp in placed:
        patch = sp["patch"].copy()
        outer_bar = bars[int(sp["outer_idx"])]
        h, c0 = int(sp["h"]), int(sp["c0"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=int(bg))

        transformed = patch.copy()
        transformed[transformed == int(outer_bar["fill"])] = int(outer_bar["border"])
        if sp["inner_idx"] is not None:
            inner_bar = bars[int(sp["inner_idx"])]
            transformed[transformed == int(inner_bar["fill"])] = int(outer_bar["fill"])
        paste(out, transformed, int(outer_bar["r_max"] - outer_bar["lift"] - h + 1), c0, transparent=int(bg))

        if sp["inner_idx"] is not None:
            _, ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_idx"])]
            inner_patch = np.full((int(ih), int(iw)), int(inner_bar["border"]), dtype=int)
            paste(out, inner_patch, int(inner_bar["r_max"] - inner_bar["lift"] - ih + 1), c0 + int(ic0))

    return inp, out


def _sample_outer_targets(n_bars: int, n_sprites: int) -> list[int]:
    """Ensure every bar participates so every gap-coded lift is visible in the output."""
    slots = list(range(n_bars))
    extra = n_sprites - n_bars
    if extra > 0:
        pool = [i for i in range(n_bars)]
        slots.extend(random.sample(pool, extra))
    random.shuffle(slots)
    return slots


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = 3 if n_bars == 3 else random.randint(max(SPRITE_COUNT_RANGE[0], n_bars), SPRITE_COUNT_RANGE[1])

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights, gaps = _sample_layout(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights, gaps)
    top_h = int(bars[0]["r_min"])
    if top_h < MIN_TOP_AREA:
        return None
    if not any(int(bar["lift"]) > 0 for bar in bars[:-1]):
        return None

    outer_targets = _sample_outer_targets(n_bars, n_sprites)
    sprites = [_make_sprite(idx, bars) for idx in outer_targets]
    sprites.sort(key=lambda s: (s["inner_idx"] is None, -int(s["w"]), -int(s["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)
    if get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
