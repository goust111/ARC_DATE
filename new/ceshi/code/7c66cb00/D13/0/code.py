# Variant of 7c66cb00
# Dimension: D13 — 条的颜色约束 (meta)
# Change: 每个条有唯一的填充色和边框色，填充色≠边框色，所有填充色互不相同，边框色可与其他条的填充色相同... → 填充色可重复（需额外消歧）
"""
design_analysis:
  concrete_rule:
    - 底部仍是 2-3 条横条，每条结构不变：左右边框色，中间填充色。
    - 变体只改 D13：允许两条 bar 共享同一个 fill_color，但它们的 border_color 仍不同。
    - 当顶部某个 sprite（或 solver 额外检测到的单色子组件）的 outer_color 对应到多个 bar 时，
      不再“一色一条”唯一匹配，而是选择同 fill 候选中更高的 bar。
    - 为避免“更高”与“更靠上”在少量样例里混淆，generator 会让共享 fill 的两条 bar 有时上面更高、
      有时下面更高；因此人类必须观察高度，而不是只看上下顺序。
  behavior_trace:
    - 原规则：若两个 bar 都是红填充，红色 sprite 会被粘到两个 bar，或生成器根本不敢生成这种输入。
    - 新规则：先收集所有红填充 bar，再比较高度，只把 sprite 放入更高的那个红 bar，并做同样的颜色交换。
    - 差异点只发生在“按 outer_color 选目标条”这一步，其余清除顶部、底对齐、颜色交换保持不变。
  complexity_calibration:
    - 只引入一个离散消歧维度：高度最大者胜出；不再叠加距离、排序、旋转等额外自由度。
    - 每个样例都强制出现至少一组重复 fill，且至少一个 sprite 使用这个重复 fill，避免退化回原题。
    - 仍保留 2-3 条 bar、1-2 色 sprite、内层子块独立下落这些原任务线索，视觉上像原题的“近邻变体”。
  implicit_assumptions_in_original_main:
    - 原 `matching = [p for p in patterns if p["outer_color"] == bar["fill"]]` 假设 fill 唯一；D13 下会把同一 pattern
      贴进多个 bar，必须改成“pattern 先选唯一目标 bar，再 paste 一次”。
    - 原生成器按 `outer_idx` 记录占位，默认“选择哪个 fill 就落到哪个 bar”；D13 下同 fill 可能共用同一个目标，
      因此输出占位必须按“解析后的目标 bar”统计，否则不同 sprite 会在输出里互相覆盖。
    - 颜色替换若用顺序式数组赋值，会在 `target_border == inner_color` 时误改刚替换出的外层像素；
      这里改为基于原 patch 的掩码替换，避免这个原题里“通常无害”的覆盖问题。
  placement_legality:
    - sprite 高度按其解析后的目标 bar 高度约束。
    - inner 子块若存在，也要求其高度不超过自己解析后的目标 bar 高度。
    - 顶部 sprite 之间保持至少 1 格间隔；输出阶段按目标 bar 的水平占位做冲突检查。
  visual_observability:
    - 共享 fill 的 bar 颜色完全相同于人眼可见的“匹配线索”，唯一额外差异就是 bar 高度。
    - 输出里只有更高的同色 bar 接收到该 sprite，较矮的同色 bar 保持空白，因此消歧依据能从 input/output 对照看出。
  visual_cleanliness:
    - 仍使用横向整条 bar 和顶部分离 sprite；不引入额外标记。
    - 输出重叠由 generator 的目标 bar 占位约束避免，保证每个落入条内的图案可独立辨认。
  generator_guarantees_for_new_solver:
    - 每次生成都存在且仅存在一组重复 fill。
    - 该重复 fill 的两个 bar 高度不同。
    - 至少一个 sprite 的 outer_color 等于该重复 fill。
    - solver 的“同 fill 取更高 bar”分支因此每次都会被实际使用。
"""

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


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
GEN_RETRIES = 500


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Find full-width horizontal bars, keeping repeated fills as separate bars."""
    h, w = grid.shape
    bars: list[dict] = []
    current: dict | None = None

    for r in range(h):
        row = grid[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
                current = None
            continue

        counts = Counter(non_bg)
        fill_color, count = counts.most_common(1)[0]
        if int(count) <= int(w * 0.7):
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
                current = None
            continue

        border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
        if current is not None and int(current["fill"]) == int(fill_color) and int(current["border"]) == int(border):
            current["r_max"] = int(r)
        else:
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
            current = {
                "fill": int(fill_color),
                "border": int(border),
                "r_min": int(r),
                "r_max": int(r),
            }

    if current is not None:
        current["h"] = int(current["r_max"] - current["r_min"] + 1)
        bars.append(current)

    for idx, bar in enumerate(bars):
        bar["idx"] = int(idx)
        bar["h"] = int(bar["r_max"] - bar["r_min"] + 1)
    return bars


def _select_target_bar(bars: list[dict], fill_color: int) -> dict:
    """
    Disambiguation rule for repeated fills:
    choose the tallest bar among those sharing the sprite's outer color.
    Ties fall back to the uppermost bar for determinism.
    """
    candidates = [bar for bar in bars if int(bar["fill"]) == int(fill_color)]
    assert candidates, f"no bar uses fill color {fill_color}"
    return max(candidates, key=lambda bar: (int(bar["h"]), -int(bar["r_min"])))


def _extract_patterns(grid: np.ndarray, top_end: int, bg: int) -> list[dict]:
    """Reproduce the original solver's per-color component extraction behavior."""
    patterns: list[dict] = []
    top = grid[:top_end, :]
    colors = sorted(set(map(int, top.flatten())) - {int(bg)})
    for col in colors:
        mask = top == int(col)
        if not bool(mask.any()):
            continue
        labels, n_regions = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n_regions) + 1):
            coords = np.argwhere(labels == int(region_id))
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
            inner_color = int(inner_colors[0]) if inner_colors else None
            patterns.append(
                {
                    "outer_color": int(outer_color),
                    "inner_color": inner_color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": np.array(bbox, dtype=int),
                }
            )
    patterns.sort(key=lambda pat: (int(pat["r0"]), int(pat["c0"]), int(pat["h"]), int(pat["w"])))
    return patterns


def _transform_patch(content: np.ndarray, outer_color: int, inner_color: int | None, target_bar: dict) -> np.ndarray:
    """Swap sprite colors using masks from the original content to avoid overwrite artifacts."""
    src = np.array(content, dtype=int)
    transformed = src.copy()
    outer_mask = src == int(outer_color)
    transformed[outer_mask] = int(target_bar["border"])
    if inner_color is not None:
        inner_mask = src == int(inner_color)
        transformed[inner_mask] = int(target_bar["fill"])
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = min(int(bar["r_min"]) for bar in bars)
    patterns = _extract_patterns(g, pattern_area_end, bg)

    out[:pattern_area_end, :] = np.where(out[:pattern_area_end, :] != int(bg), int(bg), out[:pattern_area_end, :])

    for pat in patterns:
        target_bar = _select_target_bar(bars, int(pat["outer_color"]))
        transformed = _transform_patch(pat["content"], int(pat["outer_color"]), pat["inner_color"], target_bar)
        place_r = int(target_bar["r_max"] - pat["h"] + 1)
        paste(out, transformed, place_r, int(pat["c0"]), transparent=int(bg))

    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _choose_fill_layout(n_bars: int, bg: int) -> tuple[list[int], int]:
    palette = [c for c in range(10) if c != int(bg)]
    random.shuffle(palette)
    duplicate_fill = int(palette[0])
    if n_bars == 2:
        return [duplicate_fill, duplicate_fill], duplicate_fill

    unique_fill = int(palette[1])
    duplicate_positions = random.sample(range(3), 2)
    fills = [unique_fill] * 3
    for pos in duplicate_positions:
        fills[pos] = duplicate_fill
    return fills, duplicate_fill


def _choose_borders(bg: int, fills: list[int]) -> list[int]:
    palette = [c for c in range(10) if c != int(bg)]
    random.shuffle(palette)
    borders: list[int] = []
    used: set[int] = set()
    for fill in fills:
        for color in palette:
            if int(color) == int(fill):
                continue
            if int(color) in used:
                continue
            borders.append(int(color))
            used.add(int(color))
            break
    return borders


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[assignment]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0, r1 = int(r - h), int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {
            "idx": int(i),
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = int(r0 - 1)
    return grid, bars


def _target_index_by_fill(bars: list[dict]) -> dict[int, int]:
    targets: dict[int, int] = {}
    for fill in sorted({int(bar["fill"]) for bar in bars}):
        targets[int(fill)] = int(_select_target_bar(bars, int(fill))["idx"])
    return targets


def _make_sprite(outer_fill: int, bars: list[dict], target_by_fill: dict[int, int]) -> dict:
    outer_target = bars[int(target_by_fill[int(outer_fill)])]
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(outer_target["h"])))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), int(outer_fill), dtype=int)

    inner_fill = None
    inner_bbox = None
    distinct_fills = sorted({int(bar["fill"]) for bar in bars})
    other_fills = [fill for fill in distinct_fills if int(fill) != int(outer_fill)]
    if other_fills and h >= 3 and w >= 3 and random.random() < INNER_COLOR_PROB:
        random.shuffle(other_fills)
        for candidate_fill in other_fills:
            inner_target = bars[int(target_by_fill[int(candidate_fill)])]
            valid_inner = [
                (ih, iw)
                for ih in range(1, h - 1)
                for iw in range(1, w - 1)
                if (w - iw) % 2 == 0
                and ih * iw * 2 < h * w
                and ih <= int(inner_target["h"])
            ]
            if not valid_inner:
                continue
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = (w - iw) // 2
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(candidate_fill)
            inner_fill = int(candidate_fill)
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))
            break

    return {
        "outer_fill": int(outer_fill),
        "outer_target_idx": int(target_by_fill[int(outer_fill)]),
        "inner_fill": inner_fill,
        "inner_target_idx": None if inner_fill is None else int(target_by_fill[int(inner_fill)]),
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[int(start) : int(start + length)].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1, c1 = int(r0 + h - 1), int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], n_bars: int, top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in range(n_bars)]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        outer_target_idx = int(sp["outer_target_idx"])
        candidates: list[tuple[int, int]] = []

        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[outer_target_idx], c0, w):
                continue
            if sp["inner_fill"] is not None:
                _, ic0, _, iw = sp["inner_bbox"]
                inner_target_idx = int(sp["inner_target_idx"])
                if not _interval_free(bar_occ[inner_target_idx], c0 + int(ic0), int(iw)):
                    continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[outer_target_idx][c0 : c0 + w] = True
        if sp["inner_fill"] is not None:
            _, ic0, _, iw = sp["inner_bbox"]
            inner_target_idx = int(sp["inner_target_idx"])
            bar_occ[inner_target_idx][c0 + int(ic0) : c0 + int(ic0) + int(iw)] = True

        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp, out = base.copy(), base.copy()
    for sp in placed:
        patch = np.array(sp["patch"], dtype=int)
        h, c0 = int(sp["h"]), int(sp["c0"])

        paste(inp, patch, int(sp["r0"]), c0, transparent=int(bg))

        outer_bar = bars[int(sp["outer_target_idx"])]
        transformed = _transform_patch(patch, int(sp["outer_fill"]), sp["inner_fill"], outer_bar)
        paste(out, transformed, int(outer_bar["r_max"] - h + 1), c0, transparent=int(bg))

        if sp["inner_fill"] is not None:
            _, ic0, ih, iw = sp["inner_bbox"]
            inner_bar = bars[int(sp["inner_target_idx"])]
            inner_patch = np.full((int(ih), int(iw)), int(inner_bar["border"]), dtype=int)
            paste(out, inner_patch, int(inner_bar["r_max"] - ih + 1), int(c0 + ic0))

    return inp, out


def _sample_sprite_fills(n_sprites: int, fills: list[int], duplicate_fill: int) -> list[int]:
    distinct_fills = sorted(set(map(int, fills)))
    slots: list[int] = []
    for fill in distinct_fills:
        slots.extend([int(fill), int(fill)])

    outer_fills = [int(duplicate_fill)]
    slots.remove(int(duplicate_fill))
    random.shuffle(slots)
    outer_fills.extend(int(fill) for fill in slots[: n_sprites - 1])
    random.shuffle(outer_fills)
    return outer_fills


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    bg = random.randint(0, 9)
    fills, duplicate_fill = _choose_fill_layout(n_bars, bg)
    borders = _choose_borders(bg, fills)

    for _ in range(50):
        heights = _sample_bar_heights(n_bars)
        duplicate_heights = [int(heights[i]) for i, fill in enumerate(fills) if int(fill) == int(duplicate_fill)]
        if len(set(duplicate_heights)) == len(duplicate_heights):
            break
    else:
        return None

    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])
    target_by_fill = _target_index_by_fill(bars)

    effective_target_count = len(set(target_by_fill.values()))
    if effective_target_count == 1:
        n_sprites = 2
    else:
        n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    outer_fills = _sample_sprite_fills(n_sprites, fills, duplicate_fill)
    sprites = [_make_sprite(fill, bars, target_by_fill) for fill in outer_fills]
    sprites.sort(key=lambda sp: (sp["inner_fill"] is None, -int(sp["w"]), -int(sp["h"])))

    placed = _place_sprites(sprites, n_bars, top_h)
    if placed is None:
        return None

    if not any(len([bar for bar in bars if int(bar["fill"]) == int(sp["outer_fill"])]) > 1 for sp in placed):
        return None

    inp, out = _render(base, placed, bars, bg)
    if int(get_bg(inp)) != int(bg):
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
