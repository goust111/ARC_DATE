# Variant of 7c66cb00
# Dimension: D6 — 变换规则：颜色交换逻辑 (transform)
# Change: 精灵放入条后执行颜色交换：外层色变为目标条的边框色，内层色变为目标条的填充色... → 仅外层色变换，内层色消失

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for the D6 variant
# ----------------------------------
# source_task baseline:
# - 58f5dbd5 的核心做法不是“重画一个新颜色图案”，而是把 motif 对应位置改成 bg，
#   形成真实可见的挖空效果。这里直接借用这个复杂度：精灵被放进条内后，
#   只保留外层轮廓，内层区域不再重着色，而是被刻成 bg 洞。
#
# a) 具体化与行为追踪
# - 输入仍是：顶部若干双色小精灵；底部 2-3 条全宽横条。
# - 每个精灵的主色(outer)等于某条的填充色(fill)。精灵可带一个较小的内层色块(inner)。
# - 原题输出：整块精灵下放到底部，outer -> border，inner -> fill，因此落条后的图案完全不透底。
# - 变体输出：outer 仍变成目标条的 border；但 inner 区域改成 bg，并且以不透明方式贴到条上，
#   于是条内部被真正挖掉，出现可见的背景色洞。
# - 小例子：
#   输入顶部 `333 / 323 / 333`，其中 3 匹配某条 fill，条边框色为 4。
#   原题下放得到 `444 / 434 / 444`。
#   本变体下放时先做 `444 / 404 / 444`，再不透明贴到条上，最终仍是 `444 / 404 / 444`，
#   中心成为可见 bg 洞。若 inner 不是单点而是小矩形，洞的可视感会更明显。
#   之所以这样具体化，是因为 ARC 像素网格没有 alpha；若仅做“透明贴回”，在纯色条上会与原题像素级相同。
#
# b) 复杂度校准
# - 只改 D6，不改 bars 的匹配机制、x 对齐、底对齐、擦除顶区这些主结构。
# - 但原 solver 会按颜色拆顶区组件，导致 inner 颜色块还会被当成独立子精灵再下放一次。
#   对“inner 消失”变体这是错误副作用，因此必须把“顶区识别单位”从单色组件改成多色整精灵。
# - 生成器强制每个样本至少有一个双色精灵，避免样本长期退化成纯色精灵，从而让 D6 不可见。
#
# c) 隐式假设审查（与 main() 直接相关）
# - 原 `for col in top_colors -> label_mask(mask)` 依赖旧前提：按颜色拆组件仍能得到合法输出，
#   因为 inner 颜色在旧任务里还能独立落条。变体中这个前提失效，会制造“不该存在的额外图案”。
# - 原 `elif v == inner_color: transformed = bar["fill"]` 依赖旧前提：inner 应保留为实体像素。
#   变体中该行必须改成“inner -> bg”，并以不透明 paste 才能在最终网格中留下可见洞。
# - 原 generator 的 `bar_occ[inner_idx]` 占位与 `_render()` 中 inner 独立下放完全绑定；
#   变体删除该输出分支后，这个占位约束也应删除，否则只是无谓降低生成成功率。
#
# d) 放置合法性
# - 顶区精灵仍要求彼此有至少 1 格间隔，避免整精灵检测时误并成一块。
# - 同一 outer bar 内的精灵仍禁止水平重叠，否则底对齐后会互相覆盖。
# - 精灵仍限制在 cols 1..W-2 内，确保不会覆盖条的竖边框。
# - inner 不再单独落到自己的 bar，所以不需要再为 inner 颜色预留 bar 空间。
#
# e) 视觉可观察性
# - 为了让“洞”一眼可见，inner 只在 h>=3, w>=3 的精灵里生成，并限制面积至少 2。
# - 每个样本至少 1 个双色精灵；其洞在输出中必须落到某条内部，因此能看到“边框色轮廓 + 背景色洞”。
#
# f) 可视化整洁度
# - 仍保持横条全宽、条间单行背景分隔、精灵顶部稀疏摆放，避免多层覆盖造成阅读混乱。
# - 不再产生“inner 子精灵另落一条”的额外图案，输出比原始 artifact 更干净，也更贴近题意。
#
# g) 颜色变化可推断性
# - 颜色种类依然只来自 bar fill / border / bg 三类角色；没有引入数值相关排序。
# - 规则从样例里能读成：outer 跟原题一样变成边框色；inner 不变成新色，而是被挖空为 bg。


def _find_bars(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Detect full-width horizontal bars stacked in the lower area."""
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    current_bar: dict[str, int] | None = None

    for r in range(h):
        row = grid[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
                left = int(row[0])
                right = int(row[-1])
                border = left if left != int(fill_color) else right
                if current_bar and int(current_bar["fill"]) == int(fill_color):
                    current_bar["r_max"] = int(r)
                else:
                    if current_bar is not None:
                        bars.append(current_bar)
                    current_bar = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }
            elif current_bar is not None:
                bars.append(current_bar)
                current_bar = None
        elif current_bar is not None:
            bars.append(current_bar)
            current_bar = None

    if current_bar is not None:
        bars.append(current_bar)
    return bars


def _extract_patterns(top: np.ndarray, bg: int) -> list[dict[str, int | np.ndarray | None]]:
    """
    Extract full multicolor sprites from the top area.

    Using multicolor components is the key D6 adaptation:
    inner regions must stay attached to their host sprite instead of becoming
    standalone matchable components.
    """
    patterns: list[dict[str, int | np.ndarray | None]] = []
    comps = find_components(
        top,
        bg=int(bg),
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        bbox = crop_bbox(top, (r0, c0, r1, c1))
        colors = Counter(int(v) for v in bbox.flatten() if int(v) != int(bg))
        if not colors:
            continue
        outer_color = int(colors.most_common(1)[0][0])
        inner_colors = [int(c) for c in colors if int(c) != int(outer_color)]
        inner_color = inner_colors[0] if inner_colors else None
        patterns.append(
            {
                "outer_color": int(outer_color),
                "inner_color": int(inner_color) if inner_color is not None else None,
                "r0": int(r0),
                "c0": int(c0),
                "h": int(bbox.shape[0]),
                "w": int(bbox.shape[1]),
                "content": bbox,
            }
        )
    return patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched horizontal bars.

    Variant rule:
    - Match each full sprite to the bar whose fill color equals the sprite's outer color.
    - Place the sprite at the same x-position, bottom-aligned in that bar.
    - Recolor outer pixels to the bar border color.
    - Remove inner pixels by turning them into visible bg holes.
    - Erase the original top-area sprites.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found — grid must contain at least one horizontal bar"
    pattern_area_end = int(min(int(bar["r_min"]) for bar in bars))

    top = g[:pattern_area_end, :]
    patterns = _extract_patterns(top, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for bar in bars:
        matching = [pat for pat in patterns if int(pat["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            content = np.array(pat["content"], dtype=int)
            transformed = np.full_like(content, int(bg))
            transformed[content == int(pat["outer_color"])] = int(bar["border"])

            place_r = int(bar["r_max"] - int(pat["h"]) + 1)
            place_c = int(pat["c0"])
            paste(out, transformed, place_r, place_c)

    return out


# --------------- generator ---------------

GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.55
GEN_RETRIES = 300


def _sample_bar_heights(n_bars: int) -> list[int]:
    """Sample bar heights that leave enough room for the sprite top area."""
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(
    bg: int, fills: list[int], borders: list[int], heights: list[int]
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Render horizontal bars bottom-up and return the base grid plus bar metadata."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict[str, int]] = [None] * len(heights)  # type: ignore[list-item]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
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
        r = int(r0 - 1)
    return grid, bars


def _sample_inner_bbox(h: int, w: int) -> tuple[int, int, int, int] | None:
    """
    Sample a visible inner region.

    Constraints:
    - Stays strictly inside the outer rectangle.
    - Centered horizontally to keep the sprite shape readable.
    - Area is a visible minority but at least 2 cells, so the D6 hole is observable.
    """
    valid_inner = [
        (ih, iw)
        for ih in range(1, h - 1)
        for iw in range(1, w - 1)
        if (w - iw) % 2 == 0 and 2 <= ih * iw * 1 <= (h * w) // 2
    ]
    if not valid_inner:
        return None
    ih, iw = random.choice(valid_inner)
    ir0 = random.randint(1, h - ih - 1)
    ic0 = (w - iw) // 2
    return int(ir0), int(ic0), int(ih), int(iw)


def _make_sprite(outer_idx: int, bars: list[dict[str, int]], *, require_inner: bool = False) -> dict:
    """Create a sprite with an outer bar-fill color and an optional inner region."""
    outer_fill = int(bars[outer_idx]["fill"])

    min_h = max(SPRITE_H_RANGE[0], 3 if require_inner else SPRITE_H_RANGE[0])
    min_w = max(SPRITE_W_RANGE[0], 3 if require_inner else SPRITE_W_RANGE[0])
    max_h = min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"]))
    if min_h > max_h:
        min_h = SPRITE_H_RANGE[0]
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, SPRITE_W_RANGE[1])
    patch = np.full((h, w), int(outer_fill), dtype=int)

    inner_idx = None
    inner_bbox = None
    want_inner = require_inner or (
        len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB
    )
    if want_inner and len(bars) >= 2 and h >= 3 and w >= 3:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        inner_idx = random.choice(inner_choices)
        inner_bbox = _sample_inner_bbox(h, w)
        if inner_bbox is not None:
            ir0, ic0, ih, iw = inner_bbox
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_idx]["fill"])
        else:
            inner_idx = None

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": int(inner_idx) if inner_idx is not None else None,
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    """Check that occ[start:start+length] is all False."""
    return not occ[start : start + length].any()


def _rect_separated(
    rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int
) -> bool:
    """Check that a new rect has at least 1-cell gap from all existing rects."""
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(
    sprites: list[dict], bars: list[dict[str, int]], top_h: int
) -> list[dict] | None:
    """Place sprites in the top area while preventing overlap within each destination bar."""
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_idx = int(sp["outer_idx"])

        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_idx], c0, w):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_idx][c0 : c0 + w] = True
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(
    base: np.ndarray, placed: list[dict], bars: list[dict[str, int]], bg: int
) -> tuple[np.ndarray, np.ndarray]:
    """Render input sprites in the top area and output sprites docked into their bars."""
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        outer_bar = bars[int(sp["outer_idx"])]
        h, c0 = int(sp["h"]), int(sp["c0"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=int(bg))

        transformed = np.full_like(patch, int(bg))
        transformed[patch == int(outer_bar["fill"])] = int(outer_bar["border"])
        paste(out, transformed, int(outer_bar["r_max"] - h + 1), c0)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt. Returns an input grid or None on failure."""
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

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)
    sprites = [_make_sprite(outer_slots[i], bars) for i in range(n_sprites)]

    if not any(sp["inner_idx"] is not None for sp in sprites):
        force_i = random.randrange(n_sprites)
        forced = _make_sprite(int(outer_slots[force_i]), bars, require_inner=True)
        if forced["inner_idx"] is None:
            return None
        sprites[force_i] = forced

    sprites.sort(key=lambda s: (s["inner_idx"] is None, -int(s["w"]), -int(s["h"])))
    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a valid input instance for the D6 variant."""
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
