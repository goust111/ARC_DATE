from __future__ import annotations

# Variant of 7c66cb00
# Dimension: D7 — 变换规则：精灵放置对齐方式 (transform)
# Change: 精灵在目标条中底部对齐（bottom-aligned），保持原始x位置
#   -> 精灵仍然底部对齐，但放置列改为相对整张网格做水平镜像：
#      如果输入 bbox 左列为 c0、宽为 w、网格宽为 W，则输出左列为 W - c0 - w

# Design analysis for the D7 variant
# - Concrete behavior:
#   输入仍由顶部双色精灵 + 底部横条组成。外层颜色继续匹配某个条的填充色，
#   输出时精灵仍被擦除并沉到对应条里，颜色交换规则不变。唯一核心变化是：
#   输出中的横向放置位置不再等于输入 c0，而是整张网格上的左右镜像位置。
#   例：W=24、某个宽 5 的精灵在输入列 c0=2，则旧规则输出仍在 2；新规则输出在 24-2-5=17。
#   这会把左侧精灵放到右侧、右侧精灵放到左侧，且宽度不同的对象镜像位置也不同。
# - Flow impact:
#   solver 里只需要改“条内落点的列坐标”计算；颜色匹配、bbox 提取、底对齐、擦除逻辑保持原样。
#   generator 里必须同步改 output-first 的渲染和合法性检查，否则原本在条中不重叠的对象，
#   镜像后可能在同一条里碰撞。
# - Complexity calibration:
#   若频繁生成位于中心对称轴附近的精灵，则镜像前后可能位置相同或变化很小，视觉上不明显。
#   因此生成器要求至少一个精灵满足 mirror_c0 != c0，并限制整体样本中多数精灵远离中心。
#   仍保持 2-3 条、2-3 个精灵，避免引入额外匹配维度。
# - Hidden assumptions audit for main():
#   1. bars 检测依赖横条仍是“左右边框 + 中间填充”的全宽结构；变体不改这一点，因此成立。
#   2. pattern 提取依赖顶部区域的非背景组件 bbox 可完整包住双色精灵；镜像只影响输出位置，因此成立。
#   3. 原代码 `place_c = pat["c0"]` 依赖“输出保留原 x”；该前提在变体中失效，必须替换。
#   4. paste 默认允许覆盖已有颜色，所以若同条多个精灵镜像后重叠，后放置对象会破坏前者。
#      生成器因此必须在镜像后坐标上做无重叠约束。
# - Placement legality:
#   顶部输入精灵仍需互不接触、在 top area 内；输出精灵需在其目标条内底对齐且镜像后不越界。
#   因为 mirror_c0 = W-c0-w，只要输入保证 1 <= c0 <= W-1-w，就自动有 0 <= mirror_c0 <= W-w-1。
#   为避免和边框大面积粘连导致视觉混乱，生成器还要求镜像后至少保留在网格内，并对同条做区间避让。
# - Visual observability:
#   人类在 2-3 组样例中应看到：顶部左侧精灵落到对应条的右侧，右侧精灵落到左侧，而颜色交换不变。
#   为此生成器强制至少一个精灵发生明显横向位移，并尽量避免所有精灵都落在镜像不变点附近。
# - Clean visualization:
#   不增加新标记，只保留原本的顶区与条区两层结构。镜像后的条内对象仍然分散放置，避免覆盖叠加。
# - Color-change inference:
#   颜色规则完全沿用原题，只改空间位置。颜色种类数量与原题一致，因此不会额外增加归纳负担。
# - New assumptions introduced by the variant:
#   solver 现在假设“镜像目标列由整个网格宽度决定，而非条宽”。generator 通过固定条为全宽结构，
#   保证这一全局镜像定义在所有样本里一致、可见、可推断。

from collections import Counter
from pathlib import Path
import random
import sys
import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


def _mirror_left(c0: int, w: int, grid_w: int) -> int:
    """Horizontal mirror of a bbox's left column in a width-`grid_w` grid."""
    return int(grid_w - c0 - w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched horizontal bars with a color swap.

    Variant rule:
    - Match each sprite to the bar whose fill color equals the sprite's outer color.
    - Place the sprite bottom-aligned in that bar.
    - Horizontal placement is mirrored across the full grid width:
      output left column = W - input_left - sprite_width.
    - Recolor: sprite outer -> bar border, sprite inner -> bar fill.
    - Erase the original top sprites.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    out = g.copy()
    bg = int(Counter(g.flatten()).most_common(1)[0][0])

    bars = []
    current_bar = None
    for r in range(H):
        row = g[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            c = Counter(non_bg)
            fill_color, count = c.most_common(1)[0]
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
            else:
                if current_bar:
                    bars.append(current_bar)
                    current_bar = None
        else:
            if current_bar:
                bars.append(current_bar)
                current_bar = None
    if current_bar:
        bars.append(current_bar)

    assert bars, "No bars found — grid must contain at least one horizontal bar"

    pattern_area_end = int(min(int(b["r_min"]) for b in bars))
    patterns = []
    top = g[:pattern_area_end, :]
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
            bbox = crop_bbox(g, (r0, c0, r1, c1))

            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue
            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_colors = [int(c) for c in colors_in_bbox.keys() if int(c) != int(outer_color)]
            inner_color = int(inner_colors[0]) if inner_colors else None

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

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for bar in bars:
        matching = [p for p in patterns if int(p["outer_color"]) == int(bar["fill"])]
        for pat in matching:
            transformed = np.array(pat["content"], dtype=int).copy()
            for rr in range(int(pat["h"])):
                for cc in range(int(pat["w"])):
                    v = int(transformed[int(rr), int(cc)])
                    if v == int(pat["outer_color"]):
                        transformed[int(rr), int(cc)] = int(bar["border"])
                    elif pat["inner_color"] is not None and v == int(pat["inner_color"]):
                        transformed[int(rr), int(cc)] = int(bar["fill"])

            place_r = int(bar["r_max"] - pat["h"] + 1)
            place_c = _mirror_left(int(pat["c0"]), int(pat["w"]), W)
            paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 9)
INNER_COLOR_PROB = 0.45
GEN_RETRIES = 300


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
    bars = [None] * len(heights)
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = heights[i]
        r0, r1 = r - h, r - 1
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = fills[i]
        grid[r0 : r1 + 1, 0] = borders[i]
        grid[r0 : r1 + 1, GRID_W - 1] = borders[i]
        bars[i] = {"fill": fills[i], "border": borders[i], "h": h, "r_min": r0, "r_max": r1}
        r = r0 - 1
    return grid, bars


def _make_sprite(outer_idx: int, bars: list[dict]) -> dict:
    outer_fill = bars[outer_idx]["fill"]
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], bars[outer_idx]["h"]))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        inner_choices = [i for i in range(len(bars)) if i != outer_idx]
        inner_idx = random.choice(inner_choices)
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if (w - iw) % 2 == 0 and ih * iw * 2 < h * w
        ]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = bars[inner_idx]["fill"]
            inner_bbox = (ir0, ic0, ih, iw)
        else:
            inner_idx = None

    return {
        "outer_idx": outer_idx,
        "inner_idx": inner_idx,
        "patch": patch,
        "h": h,
        "w": w,
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


def _mirror_output_ok(sp: dict, c0: int, bar_occ: list[np.ndarray]) -> bool:
    mirror_c0 = _mirror_left(c0, sp["w"], GRID_W)
    if mirror_c0 < 0 or mirror_c0 + sp["w"] > GRID_W:
        return False
    if not _interval_free(bar_occ[sp["outer_idx"]], mirror_c0, sp["w"]):
        return False
    if sp["inner_idx"] is not None:
        _, ic0, _, iw = sp["inner_bbox"]
        inner_c0 = _mirror_left(c0 + ic0, iw, GRID_W)
        if inner_c0 < 0 or inner_c0 + iw > GRID_W:
            return False
        if not _interval_free(bar_occ[sp["inner_idx"]], inner_c0, iw):
            return False
    return True


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed = []

    moved_candidates = 0
    for sp in sprites:
        h, w = sp["h"], sp["w"]
        candidates = []
        for c0 in range(1, GRID_W - w):
            if not _mirror_output_ok(sp, c0, bar_occ):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        non_fixed = [(r0, c0) for (r0, c0) in candidates if _mirror_left(c0, w, GRID_W) != c0]
        if non_fixed:
            chosen_pool = non_fixed if moved_candidates == 0 or random.random() < 0.8 else candidates
            moved_candidates += 1
        else:
            chosen_pool = candidates

        r0, c0 = random.choice(chosen_pool)
        mirror_c0 = _mirror_left(c0, w, GRID_W)
        bar_occ[sp["outer_idx"]][mirror_c0 : mirror_c0 + w] = True
        inner_mirror_c0 = None
        if sp["inner_idx"] is not None:
            _, ic0, _, iw = sp["inner_bbox"]
            inner_mirror_c0 = _mirror_left(c0 + ic0, iw, GRID_W)
            bar_occ[sp["inner_idx"]][inner_mirror_c0 : inner_mirror_c0 + iw] = True

        placed.append({**sp, "r0": r0, "c0": c0, "mirror_c0": mirror_c0, "inner_mirror_c0": inner_mirror_c0})
        top_rects.append((r0, c0, r0 + h - 1, c0 + w - 1))

    if not any(sp["mirror_c0"] != sp["c0"] for sp in placed):
        return None
    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()

    for sp in placed:
        patch = sp["patch"].copy()
        outer_bar = bars[sp["outer_idx"]]
        h = sp["h"]
        c0 = sp["c0"]

        paste(inp, patch, sp["r0"], c0, transparent=bg)

        transformed = patch.copy()
        transformed[transformed == outer_bar["fill"]] = outer_bar["border"]
        if sp["inner_idx"] is not None:
            inner_bar = bars[sp["inner_idx"]]
            transformed[transformed == inner_bar["fill"]] = outer_bar["fill"]
        paste(out, transformed, outer_bar["r_max"] - h + 1, sp["mirror_c0"], transparent=bg)

        if sp["inner_idx"] is not None:
            _, _ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[sp["inner_idx"]]
            inner_patch = np.full((ih, iw), inner_bar["border"], dtype=int)
            paste(out, inner_patch, inner_bar["r_max"] - ih + 1, sp["inner_mirror_c0"])

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    fills = palette[1 : 1 + n_bars]
    borders = palette[1 + n_bars : 1 + 2 * n_bars]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = bars[0]["r_min"]

    outer_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(outer_slots)
    sprites = [_make_sprite(outer_slots[i], bars) for i in range(n_sprites)]
    sprites.sort(key=lambda s: (-s["w"], -s["h"], s["inner_idx"] is None))

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
