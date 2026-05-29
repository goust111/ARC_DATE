# Combo variant of 7c66cb00 (index 9)
# combo_spec: D5/1+D7/3+D17/0
# Name: 间隔编码信息与边框标记指示镜像翻转

from __future__ import annotations

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
      - 顶部双色精灵按外层颜色匹配到底部某条横条。
      - 输出时精灵保持原始 x 列，贴住目标条底边，并做颜色交换。
    fused_rule:
      - 每条横条现在带有同一个二值指令的两个可见编码：
        1) 该条正下方空白行数：1 行表示保持原 x，2 行表示做全局水平镜像放置。
        2) 条边框上的单个 marker 像素：右边框表示保持原 x，左边框表示镜像放置。
      - 两个信号必须一致；它们不改变颜色匹配，只共同决定放置列：
        no_flip -> place_c = c0
        flip    -> place_c = W - c0 - w
      - 颜色交换仍沿用原题：外层色 -> 目标条边框色，内层色 -> 目标条填充色。
      - 顶部原精灵仍被擦除。
    concrete_trace:
      - 例：W=24，某个宽 5 的精灵输入左列 c0=2，匹配到 bar A。
      - 原题输出仍放在列 2。
      - 融合后若 bar A 下方间隔为 2 且 marker 在左边框，则输出放到
        24 - 2 - 5 = 17；若间隔为 1 且 marker 在右边框，则仍放在列 2。
      - 因此差异集中在“是否启用镜像后的 x 坐标”，不是引入另一套颜色规则。
  b_complexity:
    - 指令只有两档：保留 x / 镜像 x，复杂度低但视觉明显。
    - 每个样例强制同时出现 flip=True 和 flip=False 两类条，避免所有条同态。
    - 每条 bar 都分配一个完整外层精灵，使每个 gap/marker 都真的参与输出。
    - 为了让每条 bar 都拥有自己的 gap 编码，最底部条也保留 1 或 2 行底部空白，
      把“该条正下方空白”统一成可读信号；否则底部条没有间隔可读，冗余编码不完整。
  c_assumption_audit:
    - 原代码用 row[0]/row[-1] 直接当 border。现在单个边框像素可能是 marker，
      所以必须改为统计整条左右边列的众数来恢复真实 border。
    - 原代码假设条间 gap 固定为 1，因此 gap 不携带信息；这里该假设失效，
      solver 需要为每条 bar 读取 gap_below 并解码成 flip。
    - 原代码 `place_c = pat["c0"]` 假设输出横向位置永远保持不变；这里改为
      根据条指令在 `c0` 与 `W-c0-w` 之间切换。
    - 单独的 D17 常会把“翻转”理解为翻转图案朝向，但本融合规格明确由 D7 提供
      的是“镜像后的 x 坐标”。因此这里不再左右翻转 patch 本身，只切换放置列。
    - 原代码按颜色逐个连通域抽 bbox，因此双色精灵会额外生成一个“内层单色子组件”。
      这个副规则继续保留；子组件也按自己匹配到的 bar 读取该 bar 的 flip 指令。
    - 顺序式整数组替换在颜色相撞时不安全，所以 recolor 必须基于原 patch 的掩码。
  d_legality:
    - 生成器先决定每条 bar 的 flip，再把 gap_below 固定为 1 或 2，与 marker 同步。
    - 采样 bar 高度时保证 `sum(heights) + sum(gap_below) + top_area <= 30`。
    - 顶部精灵仍只放在 cols 1..W-2 内，所以输出中的原位与镜像位都不会碰到边框 marker。
    - 顶部放置检查 top-area 互不接触；输出放置检查每个目标 bar 的最终 x 区间互不重叠，
      包括双色精灵额外诱发的内层单色子组件。
  e_visual_observability:
    - 输入里可以直接看到：有些条下方空 1 行、有些空 2 行；同时 marker 在左或右边框。
    - 输出里可以直接看到：同样是颜色匹配后的下落，有些精灵保持左右位置，有些被送到
      网格的镜像列。由于两套信号完全一致，人类能把“左 marker + 2 行 gap”归纳成同一指令。
    - 生成器要求至少一个 flip bar 的精灵真正满足 `mirror_c0 != c0`，避免退化成看不见的镜像。
  f_cleanliness:
    - marker 只占边框上的单个像素，不会进入条内停靠区域。
    - 变化只叠加在原有“两层结构”上：顶部精灵区 + 底部条区；没有新增复杂覆盖层。
    - 每条只放一个完整外层精灵，保证 signal 与 effect 的对应关系清楚。
  g_color_inference:
    - 颜色逻辑仍是原题那套结构性映射：外层颜色选条，输出改成条的边框/填充色。
    - 新规则完全由可见空间线索驱动：gap 行数与 marker 所在边，没有依赖颜色编号。
  h_new_main_assumptions_checked:
    - solver 新增假设：每条 bar 左右边框中恰有一个像素使用 marker 色。generator 保证。
    - solver 新增假设：每条 bar 正下方空白恰为 1 或 2 行。generator 保证。
    - solver 新增假设：gap 信号与 marker 信号一致。generator 保证，并在构造后用 main 校验。
    - solver 仍假设每条 fill 色唯一，才能把外层颜色唯一匹配到目标 bar。generator 保证。
"""


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 500


def _mirror_left(c0: int, w: int, grid_w: int) -> int:
    """Mirror a bbox's left edge across the full grid width."""
    return int(grid_w - c0 - w)


def _gap_to_flip(gap: int) -> bool:
    """Decode a bar's visible gap-below signal into the mirror-x instruction."""
    assert int(gap) in (1, 2), "Gap code must be 1 or 2 rows"
    return int(gap) == 2


def _choose_flips(n_bars: int) -> list[bool]:
    """Sample per-bar instructions, forcing both branches to appear."""
    while True:
        flips = [bool(random.getrandbits(1)) for _ in range(n_bars)]
        if any(flips) and not all(flips):
            return flips


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect bars, recover true border colors, and decode both redundant signals."""
    h, w = grid.shape
    bars: list[dict] = []
    current_bar: dict | None = None

    for r in range(h):
        row = grid[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar:
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
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

    assert bars, "No bars found"

    for i, bar in enumerate(bars):
        r0 = int(bar["r_min"])
        r1 = int(bar["r_max"])
        left_vals = [int(grid[r, 0]) for r in range(r0, r1 + 1)]
        right_vals = [int(grid[r, w - 1]) for r in range(r0, r1 + 1)]
        border = int(Counter(left_vals + right_vals).most_common(1)[0][0])

        left_marker_rows = [r for r in range(r0, r1 + 1) if int(grid[r, 0]) != int(border)]
        right_marker_rows = [r for r in range(r0, r1 + 1) if int(grid[r, w - 1]) != int(border)]
        marker_count = len(left_marker_rows) + len(right_marker_rows)
        assert marker_count == 1, "Each bar must have exactly one border marker pixel"

        marker_flip = bool(left_marker_rows)
        marker_side = "left" if marker_flip else "right"
        marker_r = int(left_marker_rows[0] if marker_flip else right_marker_rows[0])
        marker_c = 0 if marker_flip else w - 1

        if i + 1 < len(bars):
            gap_below = int(bars[i + 1]["r_min"] - bar["r_max"] - 1)
        else:
            gap_below = int(h - bar["r_max"] - 1)
        gap_flip = _gap_to_flip(gap_below)

        signals_match = bool(marker_flip == gap_flip)
        assert signals_match, "Gap code and marker side must agree"

        bar["h"] = int(r1 - r0 + 1)
        bar["border"] = int(border)
        bar["gap_below"] = int(gap_below)
        bar["gap_flip"] = bool(gap_flip)
        bar["marker_flip"] = bool(marker_flip)
        bar["flip"] = bool(marker_flip)
        bar["signals_match"] = bool(signals_match)
        bar["marker_side"] = marker_side
        bar["marker_pos"] = (marker_r, marker_c)

    return bars


def _extract_patterns(grid: np.ndarray, bg: int, pattern_area_end: int) -> list[dict]:
    """Extract top-area pattern bboxes using the original per-color component logic."""
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
                    "outer_color": int(outer_color),
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
    """Apply the original color swap using masks from the unmodified patch."""
    original = np.array(pattern["content"], dtype=int)
    transformed = original.copy()
    transformed[original == int(pattern["outer_color"])] = int(bar["border"])
    if pattern["inner_color"] is not None:
        transformed[original == int(pattern["inner_color"])] = int(bar["fill"])
    return transformed


def _solve(grid: np.ndarray, *, honor_flip: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    pattern_area_end = int(min(int(b["r_min"]) for b in bars))
    patterns = _extract_patterns(g, bg, pattern_area_end)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for bar in bars:
        for pat in patterns:
            if int(pat["outer_color"]) != int(bar["fill"]):
                continue
            transformed = _recolor_pattern(pat, bar)
            place_r = int(bar["r_max"] - pat["h"] + 1)
            place_c = int(pat["c0"])
            if honor_flip and bool(bar["flip"]):
                place_c = _mirror_left(int(pat["c0"]), int(pat["w"]), w)
            if place_c < 0 or place_c + int(pat["w"]) > w:
                continue
            paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched horizontal bars with a color swap.

    Input structure:
    - Background is the most common color.
    - The top region contains 2-3 small sprites made from bar fill colors.
    - The bottom region contains 2-3 full-width bars whose edges use a border color.
    - Each bar also exposes the same binary instruction twice:
      its gap below is 1 or 2 rows, and one marker pixel sits on the right or left border.

    Rule:
    - Match each sprite to the bar whose fill color equals the sprite's outer color.
    - Decode the bar instruction redundantly:
      1-row gap + right marker means keep the original x-position;
      2-row gap + left marker means place the sprite at the globally mirrored x-position.
    - Recolor the matched sprite: outer color becomes the bar border, inner color becomes the bar fill.
    - Keep bottom alignment inside the target bar and erase the original top sprite.

    Key insight:
    - The new information is not in the colors. The bar itself tells you whether to use the
      original docking column or the mirrored one, and it tells you twice with consistent signals:
      spacing below the bar and marker side on the border.
    """
    return _solve(grid, honor_flip=True)


def _sample_bar_heights(n_bars: int) -> list[int]:
    """Sample bar heights that fit together with per-bar 1/2-row gap codes."""
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        return heights


def _build_bars(
    bg: int,
    fills: list[int],
    borders: list[int],
    heights: list[int],
    flips: list[bool],
    marker_color: int,
) -> tuple[np.ndarray, list[dict]]:
    """Render bars bottom-up, with gap and marker both encoding the same flip bit."""
    gap_below = [1 + int(flip) for flip in flips]
    total = sum(int(h) for h in heights) + sum(gap_below)
    top_h = int(GRID_H - total)
    if top_h < MIN_TOP_AREA:
        raise ValueError("layout leaves too little top area")

    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict | None] = [None] * len(heights)

    r = int(GRID_H - gap_below[-1])
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])

        marker_r = random.randint(r0, r1)
        marker_c = 0 if bool(flips[i]) else GRID_W - 1
        grid[marker_r, marker_c] = int(marker_color)

        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": h,
            "r_min": int(r0),
            "r_max": int(r1),
            "flip": bool(flips[i]),
            "gap_below": int(gap_below[i]),
            "marker_pos": (int(marker_r), int(marker_c)),
        }

        if i > 0:
            r = int(r0 - gap_below[i - 1])

    return grid, [bar for bar in bars if bar is not None]


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    """Create a mostly solid outer patch with an optional inner-color rectangle."""
    outer_fill = int(bars[outer_idx]["fill"])
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"])))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        random.shuffle(inner_choices)
        for candidate in inner_choices:
            valid_inner = [
                (ih, iw, ir0, ic0)
                for ih in range(1, h - 1)
                for iw in range(1, w - 1)
                if ih * iw * 2 < h * w
                for ir0 in range(1, h - ih)
                for ic0 in range(1, w - iw)
            ]
            if not valid_inner:
                continue
            inner_idx = int(candidate)
            ih, iw, ir0, ic0 = random.choice(valid_inner)
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_idx]["fill"])
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))
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
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _output_left(abs_c0: int, width: int, flip: bool) -> int:
    return _mirror_left(int(abs_c0), int(width), GRID_W) if bool(flip) else int(abs_c0)


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    """Place sprites in the top area while reserving their final output intervals."""
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx = int(sp["outer_idx"])
        outer_flip = bool(bars[outer_idx]["flip"])
        candidates = []

        for c0 in range(1, GRID_W - w):
            out_c0 = _output_left(c0, w, outer_flip)
            if out_c0 < 1 or out_c0 + w > GRID_W - 1:
                continue
            if outer_flip and out_c0 == c0:
                continue
            if not _interval_free(bar_occ[outer_idx], out_c0, w):
                continue

            inner_out_c0 = None
            if sp["inner_idx"] is not None:
                inner_idx = int(sp["inner_idx"])
                _, ic0, _, iw = sp["inner_bbox"]
                inner_abs_c0 = int(c0 + ic0)
                inner_out_c0 = _output_left(inner_abs_c0, int(iw), bool(bars[inner_idx]["flip"]))
                if inner_out_c0 < 1 or inner_out_c0 + int(iw) > GRID_W - 1:
                    continue
                if not _interval_free(bar_occ[inner_idx], inner_out_c0, int(iw)):
                    continue

            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0), int(out_c0), inner_out_c0))

        if not candidates:
            return None

        r0, c0, out_c0, inner_out_c0 = random.choice(candidates)
        bar_occ[outer_idx][out_c0 : out_c0 + w] = True
        if sp["inner_idx"] is not None and inner_out_c0 is not None:
            _, _ic0, _ih, iw = sp["inner_bbox"]
            bar_occ[int(sp["inner_idx"])][inner_out_c0 : inner_out_c0 + int(iw)] = True

        placed.append(
            {
                **sp,
                "r0": int(r0),
                "c0": int(c0),
                "out_c0": int(out_c0),
                "inner_out_c0": None if inner_out_c0 is None else int(inner_out_c0),
            }
        )
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    if not any(bool(bars[int(sp["outer_idx"])]["flip"]) and int(sp["out_c0"]) != int(sp["c0"]) for sp in placed):
        return None
    return placed


def _recolor_full_sprite(sp: dict, bars: list[dict]) -> np.ndarray:
    patch = np.array(sp["patch"], dtype=int)
    outer_bar = bars[int(sp["outer_idx"])]
    out = patch.copy()
    out[patch == int(outer_bar["fill"])] = int(outer_bar["border"])
    if sp["inner_idx"] is not None:
        inner_bar = bars[int(sp["inner_idx"])]
        out[patch == int(inner_bar["fill"])] = int(outer_bar["fill"])
    return out


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Render input and the expected fused output."""
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        h = int(sp["h"])

        paste(inp, patch, int(sp["r0"]), int(sp["c0"]), transparent=int(bg))

        full_patch = _recolor_full_sprite(sp, bars)
        paste(out, full_patch, int(outer_bar["r_max"]) - h + 1, int(sp["out_c0"]), transparent=int(bg))

        if sp["inner_idx"] is not None and sp["inner_out_c0"] is not None:
            _ir0, _ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_idx"])]
            inner_patch = np.full((int(ih), int(iw)), int(inner_bar["border"]), dtype=int)
            paste(out, inner_patch, int(inner_bar["r_max"]) - int(ih) + 1, int(sp["inner_out_c0"]))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = n_bars

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]
    marker_color = int(palette[1 + 2 * n_bars])

    flips = _choose_flips(n_bars)
    heights = _sample_bar_heights(n_bars)

    try:
        base, bars = _build_bars(bg, fills, borders, heights, flips, marker_color)
    except ValueError:
        return None

    top_h = int(bars[0]["r_min"])
    if top_h < MIN_TOP_AREA:
        return None

    sprites = [_make_sprite(i, bars) for i in range(n_sprites)]
    if all(sp["inner_idx"] is None for sp in sprites) and n_bars >= 2:
        sprites[0] = _make_sprite(0, bars)
    sprites.sort(key=lambda s: (s["inner_idx"] is None, -int(s["w"]), -int(s["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if get_bg(inp) != int(bg):
        return None
    decoded_bars = _find_bars(inp, int(bg))
    if not all(bool(bar["signals_match"]) for bar in decoded_bars):
        return None
    if not any(bool(bar["flip"]) for bar in decoded_bars):
        return None
    if not any(not bool(bar["flip"]) for bar in decoded_bars):
        return None
    if np.array_equal(_solve(inp, honor_flip=False), out):
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
