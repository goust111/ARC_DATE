# Combo variant of a251c730 (index 7)
# combo_spec: D1/4+D2/0
# Name: 连接器匹配+三色精灵

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# yaml:
#   concretization:
#     - 保留原题的全局框架：30x30 壁纸上有两个带单色边框的矩形，source 中放完整精灵，
#       destination 中放“提示标记”，输出仍是裁出的 destination 矩形。
#     - 融合后 source 精灵固定为三色对象：body=4 像素、decor=2 像素、anchor=1 像素。
#       其中 anchor 是唯一最少颜色，decor 是中频颜色，body 是最多颜色。
#     - 与单独 D1 不同，这里的 destination 连接器不是单色，而是直接取 source 精灵中
#       “所有非 body 像素”的 3 格小组件：1 个 anchor 色 + 2 个 decor 色。也就是说，
#       连接器本身同时编码了颜色分工和形状。
#     - 盖章时不再按单点锚定；而是把 source 精灵连接器 bbox 左上角与 destination
#       连接器 bbox 左上角对齐，因此整个 3 格连接器会完全重合。
#   behavior_trace:
#     - 原题流程：source 中找“最少颜色单像素锚点” -> destination 中找同色单像素 ->
#       把整块精灵按该单点平移过去。
#     - 融合后流程：source 中先识别三色精灵的 body / decor / anchor 频次角色；
#       然后提取“非 body 区域”作为 3 格双色连接器，并记录它的彩色归一化形状。
#       destination 中每个提示也按 4 连通组件读取为一个 3 格双色连接器，再做模板匹配。
#     - 具体例子：
#       T0 = L 形连接器，(anchor=红, decor=黄)
#       T1 = 横线连接器，(anchor=红, decor=黄)
#       T2 = L 形连接器，(anchor=红, decor=蓝)
#       只看 anchor 颜色时三者全冲突；只看 anchor+几何形状时 T0/T2 仍冲突；
#       必须使用“彩色连接器形状”这个复合键，才能唯一定位正确精灵。
#   complexity_calibration:
#     - 生成器固定 3 个类型，并强制它们满足两类对照：
#       1) T0/T1 同 anchor 色、同 decor 色、不同连接器形状；
#       2) T0/T2 同 anchor 色、同连接器形状、不同 decor 色。
#       因此颜色单独看不够，几何单独看不够，anchor+几何也不够。
#     - destination 固定放 3 个连接器，每个类型各出现一次；source 放 3-5 个精灵，
#       每种至少一个副本。这样每个关键自由度在单张样例内都有重复支撑，不靠跨样例猜测。
#   implicit_assumption_audit:
#     - 原 main() 默认 destination 组件大小为 1，这里失效；必须改为按 3 格连接器组件解析。
#     - 原 main() 默认“最少颜色的坐标就是锚点坐标”，这里失效；现在要先识别三色频次，
#       再把整个非 body 组件当作连接器处理。
#     - 原 main() 默认模板键是单个颜色值，这里失效；新键必须是连接器的彩色归一化形状。
#     - 原题允许把“所有非锚点像素”视为同一主体色，这里也失效；decor 色必须被保留，
#       否则 T0/T2 会被错误合并。
#   placement_legality:
#     - generator 先在 destination 内用完整精灵补丁生成 output，再仅把对应的 3 格连接器
#       抄回输入，天然保证每个连接器都对应一个合法、完整、不会越界的盖章位置。
#     - source 与 destination 都使用补丁级随机放置；source 设 padding=1，destination
#       也按完整精灵 footprint 放置并留出 padding，避免输入里的连接器彼此接触后误并成一块。
#     - 由于连接器 bbox 对齐，source 中连接器必须是单个 4 连通组件；shape 库和着色库都按此约束构造。
#   observability:
#     - 人类能直接看到两个新特征：
#       1) source/output 精灵是三色而不是双色；
#       2) destination 不是单点标记，而是 3 格双色连接器。
#     - 同一输入中还会同时出现“同色异形”和“同形异色”，因此解题者能从局部对比看出：
#       真正的提示不是某个颜色 id，而是连接器的颜色分工加几何布局。
#   visual_cleanliness:
#     - 连接器限制为 3 格，精灵 bbox 限制为 3x3，destination 固定 3 个标记，
#       视觉上紧凑而不拥挤。
#     - destination 输入只保留连接器，不叠加其他噪声；output 才出现完整精灵，
#       输入输出差异清晰。
#   inference_safeguard:
#     - 任务不依赖黑色=0 或颜色编号排序。anchor / decor / body 的角色完全由可见频次决定。
#     - 实现后的量化指标应满足：source 类型数在 full-key 下为 3，在 anchor-only 下为 1，
#       在 anchor+geometry 下为 2；若不是这样，就说明生成退化回更简单任务。
#   new_solver_assumptions:
#     - 每个 source 精灵恰有 3 种颜色，且频次严格为 1 < 2 < 4。
#     - 每个 source 精灵的非 body 像素恰为一个 3 格 4 连通连接器。
#     - destination 每个组件都恰为一个 3 格双色连接器，其彩色归一化形状在 source 中唯一匹配。
#     - generator 通过固定类型关系和颜色分配保证这些假设成立。


def _normalized_colored_shape(coords: np.ndarray, color_fn) -> tuple[tuple[int, int, int], ...]:
    """Normalize a colored component to bbox-local (dr, dc, color) tuples."""
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(
        sorted((int(r - r0), int(c - c0), int(color_fn(int(r), int(c)))) for r, c in coords)
    )


def _classify_source_component(comp: dict) -> dict | None:
    """Return sprite roles and connector signature for a tri-color source component."""
    colors = {int(col): int(cnt) for col, cnt in comp["colors"].items()}
    if len(colors) != 3:
        return None
    ordered = sorted(((int(cnt), int(col)) for col, cnt in colors.items()), key=lambda x: x[0])
    counts = [cnt for cnt, _col in ordered]
    if counts != [1, 2, 4]:
        return None
    anchor_col = int(ordered[0][1])
    decor_col = int(ordered[1][1])
    body_col = int(ordered[2][1])
    patch = comp["patch"]
    mask = comp["mask"]
    connector_coords = np.argwhere(mask & (patch != int(body_col)))
    if int(connector_coords.shape[0]) != 3:
        return None
    connector_mask = np.zeros_like(mask, dtype=bool)
    connector_mask[connector_coords[:, 0], connector_coords[:, 1]] = True
    if len(components_mask(connector_mask, connectivity=4)) != 1:
        return None
    connector_colors = Counter(int(patch[int(r), int(c)]) for r, c in connector_coords)
    if connector_colors != Counter({int(anchor_col): 1, int(decor_col): 2}):
        return None
    cr0, cc0, _cr1, _cc1 = bbox_coords(connector_coords)
    signature = _normalized_colored_shape(
        connector_coords,
        lambda r, c: int(patch[int(r), int(c)]),
    )
    return {
        "anchor": int(anchor_col),
        "decor": int(decor_col),
        "body": int(body_col),
        "connector_top_left": (int(cr0), int(cc0)),
        "connector_signature": signature,
    }


def _classify_marker_component(comp: dict) -> dict | None:
    """Return connector signature for a destination marker component."""
    colors = {int(col): int(cnt) for col, cnt in comp["colors"].items()}
    if len(colors) != 2:
        return None
    ordered = sorted(int(cnt) for cnt in colors.values())
    if ordered != [1, 2]:
        return None
    if int(comp["size"]) != 3:
        return None
    patch = comp["patch"]
    mask = comp["mask"]
    coords = np.argwhere(mask)
    if len(components_mask(mask, connectivity=4)) != 1:
        return None
    mr0, mc0, _mr1, _mc1 = map(int, comp["bbox"])
    signature = _normalized_colored_shape(
        coords,
        lambda r, c: int(patch[int(r), int(c)]),
    )
    return {
        "connector_top_left": (int(mr0), int(mc0)),
        "connector_signature": signature,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by matching tri-color sprites to colored connector markers.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: source contains full three-color sprites,
      while destination contains only tiny 3-cell connector components.
    - In each source sprite, the single anchor-colored cell and the two decoration-colored cells form
      one connected non-body connector.

    Rule:
    - Parse each source sprite, identify body / decoration / anchor by visible frequencies, and extract
      the connector made of all non-body cells.
    - For every destination connector, find the source sprite whose connector has the same colored shape.
    - Stamp that whole sprite into the destination frame with connector bbox top-lefts aligned.

    Key insight: The matching cue is not a lone anchor pixel anymore; it is the full colored connector,
    so the solver must preserve both tri-color role structure and connector geometry.
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
        comps: list[dict] = []
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

    assert rects, "no candidate rectangles found"
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        if not comps:
            continue
        src_like = [_classify_source_component(comp) for comp in comps]
        dst_like = [_classify_marker_component(comp) for comp in comps]
        if src is None and all(info is not None for info in src_like):
            src = (rect, bc, ibg2, comps, src_like)
        if dst is None and all(info is not None for info in dst_like):
            dst = (rect, bc, ibg2, comps, dst_like)

    assert src is not None and dst is not None, "must find both source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps, src_infos = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps, dst_infos = dst

    templates: dict[tuple[tuple[int, int, int], ...], tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp, info in zip(src_comps, src_infos):
        assert info is not None
        key = info["connector_signature"]
        patch = comp["patch"]
        mask = comp["mask"]
        ar, ac = info["connector_top_left"]
        if key not in templates:
            templates[key] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    for info in dst_infos:
        assert info is not None
        key = info["connector_signature"]
        if key not in templates:
            continue
        patch, mask, (ar, ac) = templates[key]
        mr0, mc0 = info["connector_top_left"]
        top = int(1 + mr0 - ar)
        left = int(1 + mc0 - ac)
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (output-first) ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = 5
GAP_RANGE = (1, 4)

SRC_H_RANGE = (12, 19)
SRC_W_RANGE = (12, 18)
DST_H_RANGE = (11, 19)
DST_W_RANGE = (11, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_COUNT = 3
MARKER_COUNT = 3
SOURCE_SPRITE_RANGE = (3, 5)
DEST_STAMP_PADDING = 1
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400

L_FAMILY = [
    ("AC.", "CB.", "BBB"),
    ("AC.", "CBB", ".BB"),
    ("ACB", "CBB", ".B."),
]
LINE_FAMILY = [
    ("ACC", ".B.", "BBB"),
    ("ACC", "BB.", ".BB"),
    ("ACC", "BBB", "..B"),
]


def _shape_info(shape_rows: tuple[str, ...]) -> dict:
    counts: Counter = Counter()
    connector_cells: list[tuple[int, int, str]] = []
    for r, row in enumerate(shape_rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            counts[ch] += 1
            if ch in {"A", "C"}:
                connector_cells.append((int(r), int(c), ch))
    assert counts["A"] == 1
    assert counts["C"] == 2
    assert counts["B"] == 4
    coords = np.array([(r, c) for r, c, _ch in connector_cells], dtype=int)
    connector_shape = _normalized_colored_shape(coords, lambda r, c: 1 if shape_rows[r][c] == "A" else 2)
    return {
        "connector_cells": connector_cells,
        "connector_shape": connector_shape,
        "area": int(sum(counts.values())),
        "h": int(len(shape_rows)),
        "w": int(len(shape_rows[0])),
    }


for _shape in L_FAMILY + LINE_FAMILY:
    _info = _shape_info(_shape)
    assert _info["h"] == 3 and _info["w"] == 3


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, decor: int, bg: int) -> np.ndarray:
    char_map = {"A": int(anchor), "B": int(body), "C": int(decor)}
    rows = [[char_map.get(ch, int(bg)) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _render_connector(shape_rows: tuple[str, ...], anchor: int, decor: int, bg: int) -> np.ndarray:
    char_map = {"A": int(anchor), "C": int(decor)}
    rows = [[char_map.get(ch, int(bg)) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), int(bg), dtype=int)
    rect[0, :] = rect[-1, :] = int(border)
    rect[:, 0] = rect[:, -1] = int(border)
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
    return (int(src_r), int(src_c)), (int(dst_r), int(dst_c))


def _assign_palette() -> dict | None:
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, WALLPAPER_COLORS)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    if len(remaining) < 5:
        return None

    src_bg, dst_bg, body_a, body_b = random.sample(wallpaper, 4)
    src_border = int(remaining.pop())
    dst_border = int(remaining.pop())
    anchor = int(remaining.pop())
    decor_a = int(remaining.pop())
    decor_b = int(remaining.pop())
    body_pattern = random.choice(
        [
            [int(body_a), int(body_b), int(body_a)],
            [int(body_a), int(body_b), int(body_b)],
        ]
    )
    return {
        "wallpaper": [int(c) for c in wallpaper],
        "src_bg": int(src_bg),
        "dst_bg": int(dst_bg),
        "src_border": int(src_border),
        "dst_border": int(dst_border),
        "anchor": int(anchor),
        "decor_a": int(decor_a),
        "decor_b": int(decor_b),
        "body_pattern": body_pattern,
    }


def _validate_sprite_types(sprite_types: list[dict]) -> bool:
    """Reject accidental simplifications that collapse the fusion cue."""
    if len(sprite_types) != SPRITE_TYPE_COUNT:
        return False
    full_keys = {tuple(sprite["connector_proto"]) for sprite in sprite_types}
    if len(full_keys) != SPRITE_TYPE_COUNT:
        return False
    anchor_only = {int(sprite["anchor"]) for sprite in sprite_types}
    if len(anchor_only) != 1:
        return False
    anchor_geom = {
        (
            int(sprite["anchor"]),
            tuple((int(r), int(c)) for r, c, _col in sprite["connector_proto"]),
        )
        for sprite in sprite_types
    }
    if len(anchor_geom) != 2:
        return False
    anchor_color = {(int(sprite["anchor"]), int(sprite["decor"])) for sprite in sprite_types}
    return len(anchor_color) == 2


def _build_sprite_types(palette: dict) -> list[dict]:
    shape0, shape2 = random.sample(L_FAMILY, 2)
    shape1 = random.choice(LINE_FAMILY)
    shapes = [shape0, shape1, shape2]
    decors = [int(palette["decor_a"]), int(palette["decor_a"]), int(palette["decor_b"])]
    bodies = [int(c) for c in palette["body_pattern"]]

    sprite_types: list[dict] = []
    for type_id, (shape_rows, decor, body) in enumerate(zip(shapes, decors, bodies)):
        info = _shape_info(shape_rows)
        sprite_types.append(
            {
                "type_id": int(type_id),
                "shape_rows": shape_rows,
                "body": int(body),
                "anchor": int(palette["anchor"]),
                "decor": int(decor),
                "connector_cells": list(info["connector_cells"]),
                "connector_proto": tuple(
                    sorted(
                        (
                            int(r),
                            int(c),
                            int(palette["anchor"]) if ch == "A" else int(decor),
                        )
                        for r, c, ch in info["connector_cells"]
                    )
                ),
            }
        )
    return sprite_types


def _place_markers(sprite_types: list[dict], dst_h: int, dst_w: int, dst_bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    out_inner = np.full((dst_h - 2, dst_w - 2), int(dst_bg), dtype=int)
    marker_inner = np.full((dst_h - 2, dst_w - 2), int(dst_bg), dtype=int)
    order = random.sample(list(range(len(sprite_types))), len(sprite_types))
    for type_id in order:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(
            sprite["shape_rows"],
            sprite["body"],
            sprite["anchor"],
            sprite["decor"],
            int(dst_bg),
        )
        connector_patch = _render_connector(
            sprite["shape_rows"],
            sprite["anchor"],
            sprite["decor"],
            int(dst_bg),
        )
        try:
            r0, c0 = random_free_location(out_inner, patch, transparent=int(dst_bg), padding=DEST_STAMP_PADDING)
        except ValueError:
            return None
        paste(out_inner, patch, int(r0), int(c0), transparent=int(dst_bg))
        paste(marker_inner, connector_patch, int(r0), int(c0), transparent=int(dst_bg))
    return out_inner, marker_inner


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), int(src_bg), dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(
            sprite["shape_rows"],
            sprite["body"],
            sprite["anchor"],
            sprite["decor"],
            int(src_bg),
        )
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=int(src_bg),
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, int(r0), int(c0), transparent=int(src_bg))
    return inner


def _gen_attempt() -> np.ndarray | None:
    palette = _assign_palette()
    if palette is None:
        return None
    sprite_types = _build_sprite_types(palette)
    if not _validate_sprite_types(sprite_types):
        return None

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, dst_h, dst_w, int(palette["dst_bg"]))
    if marker_result is None:
        return None
    out_inner, marker_inner = marker_result

    output = _build_frame(dst_h, dst_w, int(palette["dst_border"]), int(palette["dst_bg"]))
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    n_source = random.randint(*SOURCE_SPRITE_RANGE)
    source_type_ids = list(range(SPRITE_TYPE_COUNT))
    while len(source_type_ids) < n_source:
        source_type_ids.append(random.randrange(SPRITE_TYPE_COUNT))
    random.shuffle(source_type_ids)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, int(palette["src_bg"]))
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, int(palette["src_border"]), int(palette["src_bg"]))
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, int(palette["dst_border"]), int(palette["dst_bg"]))
    paste(dest_rect[1:-1, 1:-1], marker_inner, 0, 0)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), palette["wallpaper"])
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo variant after {MAX_GEN_ATTEMPTS} attempts")
