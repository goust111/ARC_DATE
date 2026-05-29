# Variant of a251c730
# Dimension: D1 — 匹配条件：锚点颜色匹配 (match_cond)
# Change: 精灵的锚点颜色（最少出现颜色）== 目标矩形中单像素标记的颜色，一对一匹配... → 连接器形状+颜色匹配

from __future__ import annotations

from collections import Counter
import random
import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D1 variant
# ---------------------------------------------------------------------------
# yaml:
#   concretization:
#     - 输入仍然是壁纸上两个带单色边框的矩形；左/上其中一个是 source，内部放多色精灵；
#       另一个是 destination，内部不再是单像素标记，而是 2-3 像素的同色小连接器组件。
#     - source 精灵仍是双色对象，但“稀有色”从 1 个锚点像素改为 2-3 像素连接器；
#       连接器连在主体边缘，方向可见，例如横向 2 格、竖向 2 格、L 形 3 格。
#     - 输出仍然只保留 destination 矩形，并把对应精灵平移到其中；这次对齐的是
#       “精灵连接器的包围盒左上角”与“目标连接器的包围盒左上角”，因此同形连接器会完全重合。
#   behavior_trace:
#     - 原题在 destination 中扫描每个非背景单像素，读出颜色后直接找到同色模板，再把模板的单锚点
#       对齐到该像素位置。
#     - 现在 destination 中每个标记本身就是一个小组件。solver 必须先提取该组件的颜色与归一化形状，
#       再在 source 中找到“稀有色连接器颜色相同且连接器形状相同”的精灵模板。
#     - 例子：若 source 中有 A=(红色横线连接器) 与 B=(红色竖线连接器)，destination 中出现红色竖线，
#       原规则只看颜色会在 A/B 之间歧义；新规则会选 B。
#   complexity_calibration:
#     - 为了让 D1 在每个样例里都真实参与推理，这里固定 3 个类型：
#       1) T0 与 T1 同色异形；2) T0 与 T2 同形异色。
#       这样颜色单独看不够，形状单独看也不够，必须使用二元键 `(connector_color, connector_shape)`。
#     - 目标内固定放 3 个连接器，source 内放 3-5 个精灵副本，仍保持原题的“从 source 找模板，再去
#       destination 印刻”的主体结构，不引入额外变换。
#   implicit_assumption_audit:
#     - 原 `main()` 把 destination 识别为“最大组件尺寸 == 1”的矩形；变体后应改成
#       “所有组件尺寸都在 2..3 且 source 组件明显更大”。
#     - 原模板键是 `anchor_color`，并且默认锚点只有一个坐标；变体后模板键必须变为
#       `(rare_color, normalized_connector_shape)`，锚点坐标也改为连接器包围盒左上角。
#     - 原 destination 扫描用 `np.argwhere(inner != bg)` 逐像素处理；变体后若继续逐像素，会把一个
#       连接器拆成多个标记并重复印刻，所以必须改成按 4 连通组件处理。
#   placement_legality:
#     - generator 先在 destination 内部按完整精灵补丁放置输出，再只把各补丁中的连接器像素抄回输入，
#       这样天然保证每个连接器都对应一个合法盖章位置。
#     - 连接器组件之间使用补丁级 padding，避免 destination 输入里的多个连接器接触并误并成一个组件。
#     - source 放置仍使用非重叠 + 边距约束，保证 solver 的组件分割稳定。
#   observability:
#     - 同一张图里同时出现“同色异形”和“同形异色”，人类从少量样例就能看到：destination 里的小连接器
#       不是普通彩点，而是决定匹配的钥匙。
#     - 没有基于颜色 id 的排序规则参与任务定义；稀有色唯一性由生成器直接保证，而不是靠颜色编号打破平局。
#   visual_cleanliness:
#     - 连接器只占 2-3 格，source 精灵 bbox 控制在 3x4，destination 组件数量固定为 3，视觉上清楚。
#     - 目标输入里只保留连接器，不叠加其他提示；输出里是完整精灵，前后对比明确。
#   new_solver_assumptions:
#     - 每个 source 精灵的稀有色唯一且形成单个 4 连通连接器组件。
#     - destination 中每个连接器组件为单色、尺寸 2 或 3，并且 `(颜色, 形状)` 在 source 模板中唯一可匹配。
#     - generator 通过固定三类型组合与颜色分配保证这些假设成立。


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Normalize coords to a bbox-anchored tuple shape."""
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via connector color+shape matches.

    Input structure:
    - Two bordered rectangles sit on the wallpaper: source holds full multi-cell sprites,
      destination holds only tiny 2-3 pixel connector components.
    - In each source sprite, the least frequent color forms a connected connector shape.

    Rule:
    - For every destination connector, find the source sprite whose connector has the same
      color and the same normalized orientation/shape.
    - Translate that sprite so the sprite connector overlaps the destination connector.
    - Output is the destination rectangle with its interior cleared and the matched sprites stamped in.

    Key insight: The rare-color region is no longer a single anchor cell; it is a small keyed connector,
    so both its color and its geometry are needed to identify and place the right template.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
        comps = []
        for coords in components_mask(inner != int(ibg), connectivity=4):
            if coords.size == 0:
                continue
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
            mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
            patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
            patch[~mask] = 0
            vals = [int(inner[int(r), int(c)]) for r, c in coords]
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(Counter(vals)),
                    "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                    "mask": mask,
                    "patch": patch,
                }
            )
        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if (
                        int(g[int(r0), int(c1)]) != int(bc)
                        or int(g[int(r1), int(c0)]) != int(bc)
                        or int(g[int(r1), int(c1)]) != int(bc)
                    ):
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == int(bc)).all()):
                        continue
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        sizes = [int(c["size"]) for c in comps]
        max_sz = max(sizes, default=0)
        min_sz = min(sizes, default=0)
        if max_sz > 3 and src is None:
            src = (rect, bc, ibg2, comps)
        if len(comps) >= 1 and 2 <= min_sz <= max_sz <= 3 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[tuple[int, tuple[tuple[int, int], ...]], tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if len(cols) < 2:
            continue
        min_count = min(cols.values())
        rare_colors = [col for col, count in cols.items() if int(count) == int(min_count)]
        assert len(rare_colors) == 1
        connector_col = int(rare_colors[0])
        patch = comp["patch"]
        mask = comp["mask"]
        connector_coords = np.argwhere((patch == int(connector_col)) & mask)
        assert connector_coords.size > 0
        ar, ac, _ar1, _ac1 = bbox_coords(connector_coords)
        key = (int(connector_col), _normalized_shape(connector_coords))
        if key not in templates or int(mask.sum()) > int(templates[key][1].sum()):
            templates[key] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    for coords in components_mask(inner != int(dst_bg), connectivity=4):
        vals = {int(inner[int(r), int(c)]) for r, c in coords}
        assert len(vals) == 1
        connector_col = int(next(iter(vals)))
        mr0, mc0, _mr1, _mc1 = bbox_coords(coords)
        key = (int(connector_col), _normalized_shape(coords))
        if key not in templates:
            continue
        patch, mask, (ar, ac) = templates[key]
        top = int(1 + mr0 - ar)
        left = int(1 + mc0 - ac)
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (output-first) ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (11, 19)
SRC_W_RANGE = (10, 18)
DST_H_RANGE = (11, 19)
DST_W_RANGE = (11, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_COUNT = 3
MARKER_COUNT = 3
SOURCE_SPRITE_RANGE = (3, 5)
DEST_STAMP_PADDING = 1
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400

SHARED_SHAPE_VARIANTS = [
    ("CC..", ".BB.", "BBB."),
    (".CC.", "BBB.", ".B.."),
    ("CC..", "BBB.", "..B."),
]
OTHER_SHAPE_VARIANTS = [
    ("C...", "CBB.", ".BB."),
    ("C...", "CCB.", ".BBB"),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, connector: int, bg: int) -> np.ndarray:
    """Render a sprite template with B=body, C=connector, .=bg."""
    char_map = {"B": body, "C": connector}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _connector_cells(shape_rows: tuple[str, ...]) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, row in enumerate(shape_rows) for c, ch in enumerate(row) if ch == "C"]


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (src_r, src_c), (dst_r, dst_c)


def _assign_palette() -> tuple[list[int], int, int, int, int, int, int, list[int]] | None:
    wallpaper_n = random.choice(list(WALLPAPER_COLORS))
    wallpaper = random.sample(list(range(10)), wallpaper_n)
    remaining = [c for c in range(10) if c not in wallpaper]
    random.shuffle(remaining)

    if len(remaining) < 4:
        return None
    src_border = int(remaining.pop())
    connector_a = int(remaining.pop())
    connector_b = int(remaining.pop())
    if random.random() < 0.5 and remaining:
        dst_border = int(remaining.pop())
    else:
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = int(random.choice(border_choices))

    src_bg = int(random.choice(wallpaper))
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = int(random.choice(dst_bg_choices))

    body_pool = [c for c in wallpaper if c not in {src_bg, dst_bg, dst_border, connector_a, connector_b}]
    body_pool.extend([c for c in remaining if c not in {connector_a, connector_b}])
    seen = set()
    ordered_body_pool = []
    for col in body_pool:
        if col not in seen:
            ordered_body_pool.append(int(col))
            seen.add(int(col))
    if len(ordered_body_pool) < 3:
        return None
    body_colors = random.sample(ordered_body_pool, 3)
    return wallpaper, src_border, dst_border, src_bg, dst_bg, connector_a, connector_b, body_colors


def _place_markers(
    sprite_types: list[dict], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[dict]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_components: list[dict] = []
    for type_id in random.sample(list(range(len(sprite_types))), len(sprite_types)):
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["connector"], dst_bg)
        try:
            r0, c0 = random_free_location(inner, patch, transparent=dst_bg, padding=DEST_STAMP_PADDING)
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=dst_bg)
        cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in sprite["connector_cells"]]
        marker_components.append({"type_id": int(type_id), "cells": cells})
    return inner, marker_components


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["connector"], src_bg)
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    palette_result = _assign_palette()
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, src_bg, dst_bg, connector_a, connector_b, body_colors = palette_result

    shared_a, shared_b = random.sample(SHARED_SHAPE_VARIANTS, 2)
    other = random.choice(OTHER_SHAPE_VARIANTS)
    shape_rows_list = [shared_a, other, shared_b]
    connector_colors = [connector_a, connector_a, connector_b]

    sprite_types = []
    for idx, (shape_rows, body, connector) in enumerate(zip(shape_rows_list, body_colors, connector_colors)):
        connector_cells = _connector_cells(shape_rows)
        body_count = sum(ch == "B" for row in shape_rows for ch in row)
        connector_count = len(connector_cells)
        if body_count <= connector_count:
            return None
        sprite_types.append(
            {
                "type_id": int(idx),
                "shape_rows": shape_rows,
                "body": int(body),
                "connector": int(connector),
                "connector_cells": connector_cells,
                "connector_shape": _normalized_shape(np.array(connector_cells, dtype=int)),
            }
        )

    n_source = random.randint(*SOURCE_SPRITE_RANGE)
    source_type_ids = list(range(SPRITE_TYPE_COUNT))
    while len(source_type_ids) < n_source:
        source_type_ids.append(random.randrange(SPRITE_TYPE_COUNT))
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_components = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for marker in marker_components:
        connector_color = int(sprite_types[int(marker["type_id"])]["connector"])
        for rr, cc in marker["cells"]:
            dest_rect[1 + int(rr), 1 + int(cc)] = connector_color

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D1 variant after {MAX_GEN_ATTEMPTS} attempts")
