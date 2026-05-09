# Combo variant of 7c66cb00 (index 5)
# combo_spec: D1/2+D2/2+D13/0
# Name: 双颜色联合匹配与填充色重复消歧
"""
design_analysis:
  concrete_rule:
    - 底部仍是 2-3 条横向整条 bar，左右 1 列是 border，中央是 fill。
    - 这次真正的融合规则不是“保留原始按 fill 匹配，再额外用高度消歧”，而是直接把重复 fill 的消歧写进联合键：
      sprite 的 (outer, inner) 必须匹配 bar 的 (fill, border)。
    - D1 在组合里不再单独变成 “edge -> border” 的另一套匹配，而是要求 outer 必须从 sprite 的最外圈 edge 读取，
      并验证该 edge_color == bar.fill。这样 outer 的来源是空间轮廓，不是 bbox 多数色。
    - 因此生成的 sprite 采用 2 色 framed rectangle：1-cell 外框颜色 = 目标 bar.fill，内部颜色 = 目标 bar.border。
      输出时执行交换：edge/outer -> bar.border，interior/inner -> bar.fill。
  behavior_trace:
    - 小例子：若有两条 bar 都是 fill=7，但 border 分别是 2 和 5。
    - 顶部两个 sprite 可分别是 edge=7, inner=2 与 edge=7, inner=5。
    - 原题做法只看 outer/fill=7，会把两个 sprite 都视为“同一类红框”，无法唯一解释为什么它们分去不同 bar。
    - 融合后先取 sprite 的外圈颜色 7，再读内部颜色 2 或 5，只能匹配到 (7,2) 或 (7,5) 对应的 bar。
    - 输出中这两个 sprite 都会保持原 x 位置下落，但其边框与内部颜色互换，形成可见的 swap。
  complexity_calibration:
    - 仍限制为 2-3 bars、2-3 sprites、矩形框状两色图案，避免视觉负担失控。
    - 每个样例都强制出现一组重复 fill，并且至少两个 sprite 分别落向这组重复 fill 的不同 bars。
      这样联合匹配不是偶发分支，而是每次都真的被使用。
    - border 颜色与 fill 集合分离，能让人眼稳定看到 “输入 edge 用 fill 色，输入内层用 border 色，输出交换回来”。
  hidden_assumptions_review:
    - 原 main 按颜色逐色找连通块，会把 framed sprite 的外框和内块拆开；组合后必须改成 top 区域整体 non-bg 连通块提取。
    - 原 main 用 bbox 多数色当 outer_color；组合后 outer 必须来自 outer contour，否则无法体现 D1 的轮廓读取要求。
    - 原 main 默认 fill 唯一，`matching = [p for p in patterns if p["outer_color"] == bar["fill"]]` 在重复 fill 下会把同一个 sprite 贴到多条 bar。
      这里必须改成“sprite 先用 (edge_color, inner_color) 解析唯一目标，再 paste 一次”。
    - 原顺序替色在 `fill == 某个别的目标色` 时可能串改；这里统一用源 patch 掩码做交换。
    - D13 单变体里的“更高 bar 优先”如果保留，会和联合颜色键形成两套同时存在的消歧依据，样例将不可学；因此明确舍弃。
  placement_legality:
    - sprite 高度限制在目标 bar 高度内，宽度限制在 cols 1..W-2 内。
    - 顶部 sprite 之间至少隔 1 格背景，避免 solver 把多个 sprite 误合并。
    - 输出阶段按目标 bar 维护水平占位，避免同一 bar 内底部贴图互相覆盖。
  visual_observability:
    - 解题者能直接看到：有两条甚至更多 bar 拥有相同 fill，但顶部 sprite 的 outer ring 也相同；真正区分它们的是 inner 颜色。
    - 输出里这两个重复 fill 的 bars 会分别收到不同 inner 颜色对应的 sprite，说明不能只看 fill。
    - 若只看 fill，fill-only 基线会把重复 fill sprite 复制到多条 bar；本组合输出明显不同。
  visual_cleanliness:
    - 只保留 “顶部少量 framed sprites + 底部 bars + 条内落位副本” 三层结构，不再引入原题那种按颜色子组件二次下落。
    - sprite 形状统一、颜色数少、区域分明，重复 fill 的差别通过内部色体现，不靠不可见排序。
  color_inference:
    - 输入中 sprite 外圈总是某条 bar 的 fill，内部总是同一条 bar 的 border。
    - 输出严格做空间层交换：edge -> border，interior -> fill。
    - 颜色变化有条结构和原位下落两条辅助线索，因此 2-4 个样例足以归纳。
  generator_guarantees_for_new_main:
    - 每个样例至少存在一组重复 fill bars。
    - 至少两个 sprite 命中这组重复 fill 的不同 bars。
    - 每个 sprite 都是统一 edge 色 + 统一 interior 色的两色连通矩形框。
    - 每个 sprite 的联合键 (edge_color, inner_color) 在所有 bars 中唯一对应一个目标。
    - `fill-only` 基线与组合 solver 的输出必定不同，从而避免退化回原始单色匹配行为。
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
SPRITE_H_RANGE = (3, 5)
SPRITE_W_RANGE = (3, 9)
GEN_RETRIES = 600


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect bottom full-width bars, keeping repeated fills as separate objects."""
    h, w = grid.shape
    bars: list[dict] = []
    current: dict | None = None

    for r in range(h):
        row = grid[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
                current = None
            continue

        fill_color, count = Counter(non_bg).most_common(1)[0]
        if int(count) <= int(w * 0.7):
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
                current = None
            continue

        border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
        if (
            current is not None
            and int(current["fill"]) == int(fill_color)
            and int(current["border"]) == int(border)
        ):
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


def _outer_edge_mask(patch: np.ndarray, bg: int) -> np.ndarray:
    """
    Return the first outer contour ring of the foreground in `patch`.

    Only pixels touching externally reachable background count as edge pixels.
    Internal holes would not create extra edge layers.
    """
    fg = patch != int(bg)
    h, w = fg.shape
    padded_fg = np.pad(fg, 1, constant_values=False)
    external = np.zeros_like(padded_fg, dtype=bool)
    stack = [(0, 0)]
    external[0, 0] = True

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not external[nr, nc] and not padded_fg[nr, nc]:
                external[nr, nc] = True
                stack.append((nr, nc))

    edge = np.zeros_like(fg, dtype=bool)
    for r, c in np.argwhere(fg):
        pr = int(r) + 1
        pc = int(c) + 1
        if (
            external[pr - 1, pc]
            or external[pr + 1, pc]
            or external[pr, pc - 1]
            or external[pr, pc + 1]
        ):
            edge[int(r), int(c)] = True
    return edge


def _extract_sprites(grid: np.ndarray, bg: int, top_end: int) -> list[dict]:
    """Recover whole top sprites and read their edge/interior colors."""
    top = grid[:top_end, :]
    labels, n = label_mask(top != int(bg), connectivity=4)
    sprites: list[dict] = []

    for sprite_id in range(1, int(n) + 1):
        coords = np.argwhere(labels == int(sprite_id))
        if coords.size == 0:
            continue

        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        content = crop_bbox(grid, (r0, c0, r1, c1))
        edge_mask = _outer_edge_mask(content, bg)
        inner_mask = (content != int(bg)) & (~edge_mask)

        edge_colors = Counter(map(int, content[edge_mask].flatten()))
        edge_colors.pop(int(bg), None)
        inner_colors = Counter(map(int, content[inner_mask].flatten()))
        inner_colors.pop(int(bg), None)
        if not edge_colors or not inner_colors:
            continue

        sprites.append(
            {
                "edge_color": int(edge_colors.most_common(1)[0][0]),
                "inner_color": int(inner_colors.most_common(1)[0][0]),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(content.shape[0]),
                "w": int(content.shape[1]),
                "content": np.array(content, dtype=int),
                "edge_mask": edge_mask,
            }
        )

    sprites.sort(key=lambda sprite: (int(sprite["r0"]), int(sprite["c0"])))
    return sprites


def _resolve_target_bar(bars: list[dict], sprite: dict) -> dict:
    """
    Jointly match sprite(edge, inner) to bar(fill, border).

    The edge color is the decisive outer-layer observation:
    it must equal the bar fill. Repeated fills are then disambiguated by the
    sprite's interior color matching the bar border.
    """
    candidates = [bar for bar in bars if int(bar["fill"]) == int(sprite["edge_color"])]
    matches = [bar for bar in candidates if int(bar["border"]) == int(sprite["inner_color"])]
    assert len(matches) == 1, (
        f"joint match failed for sprite pair {(sprite['edge_color'], sprite['inner_color'])}: "
        f"{len(matches)} matches"
    )
    return matches[0]


def _transform_sprite(content: np.ndarray, edge_mask: np.ndarray, bg: int, target_bar: dict) -> np.ndarray:
    """Swap contour/interior colors using source masks to avoid overwrite artifacts."""
    src = np.array(content, dtype=int)
    transformed = np.full_like(src, int(bg))
    fg_mask = src != int(bg)
    transformed[fg_mask] = int(target_bar["fill"])
    transformed[edge_mask] = int(target_bar["border"])
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock framed sprites into jointly matched bars and swap their two color layers.

    Input structure:
    - The background is the most common color.
    - The bottom region contains 2-3 full-width horizontal bars; different bars may share the same fill color, but each (fill, border) pair is unique.
    - The top region contains separated two-color sprites whose outer contour is a bar fill color and whose interior is that same bar's border color.

    Rule:
    - Recover each whole sprite as one non-background connected component, read its first outer contour ring, and use that edge color as the sprite's outer color.
    - Match the sprite only to the bar whose (fill, border) pair equals (sprite edge color, sprite inner color).
    - Erase the original top sprite, then paste it into the matched bar at the same x-position and bottom alignment, recoloring contour -> bar border and interior -> bar fill.

    Key insight:
    Repeated bar fills break the old one-color lookup. The contour still tells you which fill family the sprite belongs to, but only the ordered two-color pair, validated on the true outer ring, uniquely identifies the destination bar.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    top_end = min(int(bar["r_min"]) for bar in bars)
    sprites = _extract_sprites(g, bg, top_end)

    top = out[:top_end, :]
    top[top != int(bg)] = int(bg)

    for sprite in sprites:
        target_bar = _resolve_target_bar(bars, sprite)
        transformed = _transform_sprite(sprite["content"], sprite["edge_mask"], bg, target_bar)
        place_r = int(target_bar["r_max"] - sprite["h"] + 1)
        paste(out, transformed, place_r, int(sprite["c0"]), transparent=int(bg))

    return out


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _choose_fill_layout(n_bars: int, bg: int) -> tuple[list[int], list[int]]:
    """Return fills with exactly one repeated-fill group and its indices."""
    palette = [color for color in range(10) if int(color) != int(bg)]
    random.shuffle(palette)
    duplicate_fill = int(palette[0])

    if n_bars == 2:
        return [duplicate_fill, duplicate_fill], [0, 1]

    unique_fill = int(palette[1])
    duplicate_positions = sorted(random.sample(range(3), 2))
    fills = [int(unique_fill)] * 3
    for pos in duplicate_positions:
        fills[int(pos)] = int(duplicate_fill)
    return fills, duplicate_positions


def _choose_borders(bg: int, fills: list[int]) -> list[int]:
    """Choose unique borders disjoint from the fill set for visual clarity."""
    fill_set = set(map(int, fills))
    palette = [color for color in range(10) if int(color) != int(bg) and int(color) not in fill_set]
    random.shuffle(palette)
    assert len(palette) >= len(fills), "Not enough colors to choose disjoint borders"
    return [int(palette[i]) for i in range(len(fills))]


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[list-item]
    r = GRID_H

    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0 = int(r - h)
        r1 = int(r - 1)
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


def _make_sprite(target_idx: int, bars: list[dict]) -> dict:
    """Create a 2-color framed rectangle whose colors encode one target bar pair."""
    target_bar = bars[int(target_idx)]
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(target_bar["h"])))
    w = random.randint(*SPRITE_W_RANGE)
    patch = np.full((h, w), int(target_bar["fill"]), dtype=int)
    patch[1 : h - 1, 1 : w - 1] = int(target_bar["border"])
    edge_mask = _outer_edge_mask(patch, bg=-1)
    return {
        "target_idx": int(target_idx),
        "h": int(h),
        "w": int(w),
        "patch": patch,
        "edge_mask": edge_mask,
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[int(start) : int(start + length)].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _sample_target_indices(n_bars: int, n_sprites: int, duplicate_positions: list[int]) -> list[int]:
    """
    Force the repeated-fill group to be exercised by at least two sprites,
    one for each duplicated bar.
    """
    counts = [0] * n_bars
    targets = [int(duplicate_positions[0]), int(duplicate_positions[1])]
    counts[int(duplicate_positions[0])] += 1
    counts[int(duplicate_positions[1])] += 1

    while len(targets) < n_sprites:
        choices = [idx for idx in range(n_bars) if int(counts[idx]) < 2]
        target = int(random.choice(choices))
        counts[target] += 1
        targets.append(target)

    random.shuffle(targets)
    return targets


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    """Place top sprites so their future docked copies do not overlap inside each target bar."""
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sprite in sprites:
        h = int(sprite["h"])
        w = int(sprite["w"])
        target_idx = int(sprite["target_idx"])
        candidates: list[tuple[int, int]] = []

        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[target_idx], c0, w):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[target_idx][c0 : c0 + w] = True
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Render the input grid and its expected transformed output."""
    inp = base.copy()
    out = base.copy()

    for sprite in placed:
        patch = np.array(sprite["patch"], dtype=int)
        target_bar = bars[int(sprite["target_idx"])]
        paste(inp, patch, int(sprite["r0"]), int(sprite["c0"]), transparent=int(bg))

        transformed = _transform_sprite(patch, sprite["edge_mask"], bg, target_bar)
        place_r = int(target_bar["r_max"] - sprite["h"] + 1)
        paste(out, transformed, place_r, int(sprite["c0"]), transparent=int(bg))

    return inp, out


def _fill_only_baseline(grid: np.ndarray) -> np.ndarray:
    """
    Comparison baseline: read whole sprites correctly, but match only by fill/edge color.

    Repeated fills therefore duplicate a sprite into every bar with the same fill.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    bars = _find_bars(g, bg)
    top_end = min(int(bar["r_min"]) for bar in bars)
    sprites = _extract_sprites(g, bg, top_end)

    out[:top_end, :] = np.where(out[:top_end, :] != int(bg), int(bg), out[:top_end, :])
    for sprite in sprites:
        candidates = [bar for bar in bars if int(bar["fill"]) == int(sprite["edge_color"])]
        for bar in candidates:
            transformed = _transform_sprite(sprite["content"], sprite["edge_mask"], bg, bar)
            place_r = int(bar["r_max"] - sprite["h"] + 1)
            paste(out, transformed, place_r, int(sprite["c0"]), transparent=int(bg))

    return out


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a generated sample really exercises the combo behavior."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    bars = _find_bars(g, bg)
    top_end = min(int(bar["r_min"]) for bar in bars)
    sprites = _extract_sprites(g, bg, top_end)
    repeated_fills = {int(fill) for fill, n in Counter(int(bar["fill"]) for bar in bars).items() if int(n) > 1}

    ambiguous_fill_only_sprites = 0
    distinct_repeated_fill_targets: set[int] = set()
    joint_match_sprites = 0

    for sprite in sprites:
        fill_candidates = [bar for bar in bars if int(bar["fill"]) == int(sprite["edge_color"])]
        if len(fill_candidates) > 1:
            ambiguous_fill_only_sprites += 1

        joint_candidates = [
            bar
            for bar in fill_candidates
            if int(bar["border"]) == int(sprite["inner_color"])
        ]
        if len(joint_candidates) == 1:
            joint_match_sprites += 1
            target = joint_candidates[0]
            if int(target["fill"]) in repeated_fills:
                distinct_repeated_fill_targets.add(int(target["idx"]))

    combo_out = main(g)
    fill_only_out = _fill_only_baseline(g)
    return {
        "repeated_fill_groups": int(len(repeated_fills)),
        "sprite_count": int(len(sprites)),
        "ambiguous_fill_only_sprites": int(ambiguous_fill_only_sprites),
        "joint_match_sprites": int(joint_match_sprites),
        "distinct_repeated_fill_targets": int(len(distinct_repeated_fill_targets)),
        "fill_only_gap": int(np.count_nonzero(combo_out != fill_only_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)
    bg = random.randint(0, 9)

    fills, duplicate_positions = _choose_fill_layout(n_bars, bg)
    borders = _choose_borders(bg, fills)
    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    target_indices = _sample_target_indices(n_bars, n_sprites, duplicate_positions)
    sprites = [_make_sprite(target_idx, bars) for target_idx in target_indices]
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), -int(sprite["h"]), int(sprite["target_idx"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)
    if int(get_bg(inp)) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = fusion_metrics(inp)
    if metrics["repeated_fill_groups"] != 1:
        return None
    if metrics["joint_match_sprites"] != metrics["sprite_count"]:
        return None
    if metrics["ambiguous_fill_only_sprites"] < 2:
        return None
    if metrics["distinct_repeated_fill_targets"] < 2:
        return None
    if metrics["fill_only_gap"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
