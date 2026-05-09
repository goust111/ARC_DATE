# Combo variant of a251c730 (index 8)
# combo_spec: D1/4+D3/3
# Name: 连接器匹配+方向标记

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
# yaml:
#   concretization_and_trace:
#     - 输入仍然是壁纸上的两个带边框矩形。source 里放完整双色精灵；destination 不再放单像素，
#       而是“连接器组件 + 1 个方向指示像素”的复合标记。
#     - source 中每个精灵的稀有色不是单点，而是 2-3 格 4 连通连接器；它的颜色和归一化形状共同定义类型。
#     - destination 中先看去掉方向像素后的同色连接器，就能知道该拿哪一种 source 精灵；
#       再看方向像素位于连接器包围盒的哪一侧，决定把该精灵旋转成哪一个朝向。
#     - 具体例子：
#       1) 原题：红色单点 -> 取红锚点精灵 -> 单点对齐后直接印刻。
#       2) 本题：红色横向 2 格连接器 + 其右侧的边框色指示点 ->
#          先在 source 里找“红色 + 横向 2 格连接器”的精灵 ->
#          再按“右侧指示”把精灵顺时针旋转 90 度 ->
#          最后把旋转后精灵的“连接器包围盒左上角”对齐到目标连接器包围盒左上角。
#       3) 与原题差异：匹配不再只靠颜色，放置也不再只靠单锚点，而是由连接器几何和方向像素协同决定。
#   complexity_calibration:
#     - 为保证 D1 真正参与，每个样例固定 3 类精灵：
#       1) T0 与 T1 同连接器颜色但形状不同；
#       2) T0 与 T2 同连接器形状但颜色不同。
#       因此只看颜色或只看形状都会歧义，必须同时使用二元键 `(connector_color, connector_shape)`。
#     - 为保证 D3 真正参与，destination 固定放 3 个标记，并强制至少出现 2 种方向，
#       且至少 1 个标记不是 canonical 朝向。这样输出里一定出现可见旋转。
#     - 方向自由度只保留四个正交朝向，上/右/下/左分别对应 0/90/180/270 度旋转，复杂度有限且稳定。
#   implicit_assumption_audit:
#     - 原 `main()` 默认 destination 组件都是单点；这里必须改为按 4 连通组件解析，并从每个组件中拆出
#       “单个指示像素 + 若干连接器像素”，否则会重复处理。
#     - 原模板键只有 `anchor_color`；这里必须升级为 `(稀有色连接器颜色, 连接器归一化形状)`。
#     - 原题把单锚点坐标作为平移基准；这里单点不存在，因此改为使用“连接器包围盒左上角”作为平移基准。
#     - 原题可直接印 canonical 模板；这里必须先根据方向像素旋转模板，再做平移。
#     - 新增假设：destination 每个标记组件恰有 1 个边框色指示像素，其余像素同色且构成合法连接器；
#       generator 会直接构造并验证这些条件。
#   placement_legality:
#     - generator 采用 output-first：先把旋转后的完整精灵合法放进 destination 输出，再反推出输入标记。
#     - 对每个输出精灵，记录其旋转后连接器的包围盒左上角；输入里把 canonical 连接器放在同一锚点，
#       并在连接器包围盒某一侧补一个边框色方向像素。
#     - 放置时同时检查：
#       1) 输出精灵补丁不越界且互不重叠；
#       2) 输入标记的连接器与方向像素都在界内；
#       3) 不同标记之间保持 4-不接触，避免组件误合并。
#   observability_and_cleanliness:
#     - 人类从样例中能看到：destination 里不是普通彩点，而是小连接器；同色不同连接器会得到不同精灵，
#       同连接器但方向点换边，输出朝向也会跟着变。
#     - 指示像素统一使用 destination 边框色，避免再引入一类无结构新颜色；source/output 仍保持双色精灵，
#       视觉层次清楚。
#     - 不使用颜色 id 排序、黑色特判等人类不可见规则；所有推理都基于可见的颜色、形状、方位关系。
#   anti_degeneracy_checks:
#     - 生成器固定满足 `same_color_diff_shape == True`、`same_shape_diff_color == True`。
#     - 生成器固定满足 `len(unique_directions) >= 2` 且 `non_up_markers >= 1`。
#     - 因而按原题“只看颜色 + 不旋转”求解将系统性失败，不会退化。

GRID_SIZE = 30
WALLPAPER_COLORS = 4
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
MARKER_PLACE_RETRIES = 60
MAX_GEN_ATTEMPTS = 500

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_ROT_K = {
    (-1, 0): 0,
    (0, 1): 3,
    (1, 0): 2,
    (0, -1): 1,
}

SPRITE_SPECS = [
    {
        "name": "same_color_same_shape_a",
        "shape_rows": ("CC..", ".BB.", "BBB."),
        "connector_color_slot": "a",
    },
    {
        "name": "same_color_diff_shape",
        "shape_rows": ("C...", "CCB.", ".BBB"),
        "connector_color_slot": "a",
    },
    {
        "name": "same_shape_diff_color",
        "shape_rows": ("CC..", "BBB.", "..B."),
        "connector_color_slot": "b",
    },
]


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _interior_bg(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(grid: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
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


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], int, int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
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
                    ibg = _interior_bg(g, (int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    return rects[:6]


def _connector_info_from_source(comp: dict) -> tuple[int, np.ndarray, tuple[tuple[int, int], ...], tuple[int, int]] | None:
    cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
    if len(cols) < 2:
        return None
    min_ct = min(cols.values())
    rare_cols = [int(col) for col, ct in cols.items() if int(ct) == int(min_ct)]
    if len(rare_cols) != 1:
        return None
    connector_col = int(rare_cols[0])
    patch = np.array(comp["patch"], dtype=int)
    mask = np.array(comp["mask"], dtype=bool)
    connector_coords = np.argwhere(mask & (patch == int(connector_col)))
    if connector_coords.size == 0:
        return None
    ar, ac, _ar1, _ac1 = bbox_coords(connector_coords)
    return (
        int(connector_col),
        connector_coords,
        _normalized_shape(connector_coords),
        (int(ar), int(ac)),
    )


def _parse_marker_component(
    coords: np.ndarray,
    inner: np.ndarray,
    pointer_color: int,
) -> tuple[int, tuple[tuple[int, int], ...], tuple[int, int], tuple[int, int]] | None:
    cells = [(int(r), int(c), int(inner[int(r), int(c)])) for r, c in coords]
    pointer_cells = [(int(r), int(c)) for r, c, col in cells if int(col) == int(pointer_color)]
    if len(pointer_cells) != 1:
        return None
    pr, pc = pointer_cells[0]
    connector_coords = np.array([(int(r), int(c)) for r, c, col in cells if int(col) != int(pointer_color)], dtype=int)
    if connector_coords.shape[0] not in (2, 3):
        return None
    connector_colors = {int(inner[int(r), int(c)]) for r, c in connector_coords}
    if len(connector_colors) != 1:
        return None
    connector_col = int(next(iter(connector_colors)))
    r0, c0, r1, c1 = bbox_coords(connector_coords)
    if int(pr) < int(r0):
        direction = (-1, 0)
    elif int(pr) > int(r1):
        direction = (1, 0)
    elif int(pc) < int(c0):
        direction = (0, -1)
    elif int(pc) > int(c1):
        direction = (0, 1)
    else:
        return None
    if abs(int(pr) - min(max(int(pr), int(r0)), int(r1))) + abs(int(pc) - min(max(int(pc), int(c0)), int(c1))) != 1:
        return None
    return (
        int(connector_col),
        _normalized_shape(connector_coords),
        (int(r0), int(c0)),
        tuple(direction),
    )


def _transform_template(
    patch: np.ndarray,
    mask: np.ndarray,
    connector_color: int,
    direction: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    k = int(DIR_TO_ROT_K[tuple(direction)])
    tpatch = np.rot90(np.array(patch, dtype=int), k)
    tmask = np.rot90(np.array(mask, dtype=bool), k)
    connector_coords = np.argwhere(tmask & (tpatch == int(connector_color)))
    if connector_coords.size == 0:
        raise ValueError("rotated template lost its connector")
    ar, ac, _ar1, _ac1 = bbox_coords(connector_coords)
    return tpatch.copy(), tmask.copy(), (int(ar), int(ac))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via connector-matched directional markers.

    Input structure:
    - Two bordered rectangles sit on a striped wallpaper: source holds full two-color sprites,
      while destination holds tiny connector markers plus one border-colored direction pixel each.
    - In every source sprite, the least-frequent color forms a 2-3 cell connector component whose
      color and canonical shape identify the sprite type.

    Rule:
    - For each destination marker, ignore the direction pixel temporarily and match the remaining
      connector by both color and normalized shape to a source sprite template.
    - Read the side of the direction pixel to choose the sprite rotation, then align the rotated
      sprite's connector-bbox top-left corner to the marker connector-bbox top-left corner.
    - Return only the destination rectangle, cleared to its background before the transformed
      sprites are stamped in.

    Key insight: The destination marker carries two fused signals at once: the connector itself says
    which sprite to copy, and the extra side pixel says which quarter-turn of that sprite to print.
    """
    g = np.array(grid, dtype=int)

    rects = _find_rectangles(g)
    assert rects

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        sizes = [int(comp["size"]) for comp in comps]
        if not sizes:
            continue
        if max(sizes) >= 5 and src is None:
            src = (rect, bc, ibg2, comps)
        if 3 <= min(sizes) <= max(sizes) <= 4 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find source sprites and destination combo markers"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[tuple[int, tuple[tuple[int, int], ...]], tuple[np.ndarray, np.ndarray]] = {}
    for comp in src_comps:
        info = _connector_info_from_source(comp)
        if info is None:
            continue
        connector_col, _coords, shape, _anchor = info
        key = (int(connector_col), shape)
        patch = np.array(comp["patch"], dtype=int)
        mask = np.array(comp["mask"], dtype=bool)
        if key not in templates or int(mask.sum()) > int(templates[key][1].sum()):
            templates[key] = (patch.copy(), mask.copy())

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    for coords in components_mask(inner != int(dst_bg), connectivity=4):
        parsed = _parse_marker_component(coords, inner, int(dst_bc))
        if parsed is None:
            continue
        connector_col, shape, (mr0, mc0), direction = parsed
        key = (int(connector_col), shape)
        if key not in templates or tuple(direction) not in DIR_TO_ROT_K:
            continue
        patch, mask = templates[key]
        tpatch, tmask, (ar, ac) = _transform_template(patch, mask, int(connector_col), tuple(direction))
        top = int(1 + mr0 - ar)
        left = int(1 + mc0 - ac)
        paste(out, tpatch, int(top), int(left), mask=tmask, overwrite=True, clip=True)

    return out


def _render_sprite(shape_rows: tuple[str, ...], body: int, connector: int, bg: int) -> np.ndarray:
    rows = [[connector if ch == "C" else body if ch == "B" else bg for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _connector_cells(shape_rows: tuple[str, ...]) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, row in enumerate(shape_rows) for c, ch in enumerate(row) if ch == "C"]


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    stripe = np.array(palette * ((GRID_SIZE + len(palette) - 1) // len(palette)), dtype=int)[:GRID_SIZE]
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
    return (int(src_r), int(src_c)), (int(dst_r), int(dst_c))


def _assign_palette() -> tuple[list[int], int, int, int, int, int, int, list[int]] | None:
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, WALLPAPER_COLORS)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)
    if len(remaining) < 4:
        return None

    src_border = int(remaining.pop())
    dst_border = int(remaining.pop())
    connector_a = int(remaining.pop())
    connector_b = int(remaining.pop())

    src_bg = int(random.choice(wallpaper))
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = int(random.choice(dst_bg_choices))

    body_pool = [c for c in wallpaper if c not in {src_bg, dst_bg, src_border, dst_border, connector_a, connector_b}]
    body_pool.extend([c for c in remaining if c not in {src_border, dst_border, connector_a, connector_b}])
    ordered_pool: list[int] = []
    seen: set[int] = set()
    for col in body_pool:
        if int(col) not in seen:
            ordered_pool.append(int(col))
            seen.add(int(col))
    if len(ordered_pool) < SPRITE_TYPE_COUNT:
        return None
    body_colors = random.sample(ordered_pool, SPRITE_TYPE_COUNT)
    return wallpaper, src_border, dst_border, src_bg, dst_bg, connector_a, connector_b, body_colors


def _build_sprite_types(
    src_bg: int,
    dst_bg: int,
    connector_a: int,
    connector_b: int,
    body_colors: list[int],
) -> list[dict] | None:
    sprite_types: list[dict] = []
    connector_colors = {"a": int(connector_a), "b": int(connector_b)}
    for idx, (spec, body) in enumerate(zip(SPRITE_SPECS, body_colors)):
        connector = int(connector_colors[str(spec["connector_color_slot"])])
        shape_rows = tuple(spec["shape_rows"])
        connector_cells = _connector_cells(shape_rows)
        body_count = sum(ch == "B" for row in shape_rows for ch in row)
        if body_count <= len(connector_cells):
            return None
        src_patch = _render_sprite(shape_rows, int(body), int(connector), int(src_bg))
        dst_patch = _render_sprite(shape_rows, int(body), int(connector), int(dst_bg))
        sprite_types.append(
            {
                "type_id": int(idx),
                "name": str(spec["name"]),
                "shape_rows": shape_rows,
                "body": int(body),
                "connector": int(connector),
                "connector_cells": [(int(r), int(c)) for r, c in connector_cells],
                "connector_shape": _normalized_shape(np.array(connector_cells, dtype=int)),
                "src_patch": src_patch,
                "dst_patch": dst_patch,
            }
        )
    return sprite_types


def _choose_source_type_ids() -> list[int]:
    n_source = random.randint(*SOURCE_SPRITE_RANGE)
    type_ids = list(range(SPRITE_TYPE_COUNT))
    while len(type_ids) < n_source:
        type_ids.append(random.randrange(SPRITE_TYPE_COUNT))
    random.shuffle(type_ids)
    return [int(t) for t in type_ids]


def _choose_marker_directions() -> list[tuple[int, int]]:
    distinct_n = random.randint(2, min(len(DIRECTIONS), MARKER_COUNT))
    chosen = random.sample(DIRECTIONS, distinct_n)
    while len(chosen) < MARKER_COUNT:
        chosen.append(random.choice(chosen))
    random.shuffle(chosen)
    return [tuple(direction) for direction in chosen[:MARKER_COUNT]]


def _pointer_cell_for_direction(
    connector_cells: list[tuple[int, int]],
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    coords = np.array(connector_cells, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    if tuple(direction) == (-1, 0):
        candidates = sorted([(int(r0 - 1), int(c)) for r, c in connector_cells if int(r) == int(r0)], key=lambda x: (x[1], x[0]))
    elif tuple(direction) == (0, 1):
        candidates = sorted([(int(r), int(c1 + 1)) for r, c in connector_cells if int(c) == int(c1)], key=lambda x: (x[0], x[1]))
    elif tuple(direction) == (1, 0):
        candidates = sorted([(int(r1 + 1), int(c)) for r, c in connector_cells if int(r) == int(r1)], key=lambda x: (x[1], x[0]))
    elif tuple(direction) == (0, -1):
        candidates = sorted([(int(r), int(c0 - 1)) for r, c in connector_cells if int(c) == int(c0)], key=lambda x: (x[0], x[1]))
    else:
        return None
    connector_set = {(int(r), int(c)) for r, c in connector_cells}
    for cell in candidates:
        if cell not in connector_set:
            return (int(cell[0]), int(cell[1]))
    return None


def _marker_cells_are_clear(
    occupied: list[tuple[int, int]],
    new_cells: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    H, W = map(int, shape)
    if len(set(new_cells)) != len(new_cells):
        return False
    for r, c in new_cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        for pr, pc in occupied:
            if abs(int(r) - int(pr)) + abs(int(c) - int(pc)) <= 1:
                return False
    return True


def _place_markers_and_output(
    sprite_types: list[dict],
    dst_h: int,
    dst_w: int,
    dst_bg: int,
    dst_border: int,
) -> tuple[np.ndarray, list[dict]] | None:
    out_inner = np.full((dst_h - 2, dst_w - 2), int(dst_bg), dtype=int)
    markers: list[dict] = []
    occupied_marker_cells: list[tuple[int, int]] = []
    directions = _choose_marker_directions()

    for type_id, direction in zip(range(SPRITE_TYPE_COUNT), directions):
        sprite = sprite_types[int(type_id)]
        tpatch, tmask, anchor_rc = _transform_template(
            np.array(sprite["dst_patch"], dtype=int),
            np.array(sprite["dst_patch"] != int(dst_bg), dtype=bool),
            int(sprite["connector"]),
            tuple(direction),
        )
        canonical_pointer = _pointer_cell_for_direction(list(sprite["connector_cells"]), tuple(direction))
        if canonical_pointer is None:
            return None

        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(
                    out_inner,
                    tpatch,
                    transparent=dst_bg,
                    padding=DEST_STAMP_PADDING,
                )
            except ValueError:
                return None

            marker_r0 = int(r0 + anchor_rc[0])
            marker_c0 = int(c0 + anchor_rc[1])
            marker_cells = [(int(marker_r0 + rr), int(marker_c0 + cc)) for rr, cc in sprite["connector_cells"]]
            pointer_cell = (int(marker_r0 + canonical_pointer[0]), int(marker_c0 + canonical_pointer[1]))
            all_marker_cells = marker_cells + [pointer_cell]
            if not _marker_cells_are_clear(occupied_marker_cells, all_marker_cells, out_inner.shape):
                continue

            paste(out_inner, tpatch, int(r0), int(c0), transparent=dst_bg)
            markers.append(
                {
                    "type_id": int(type_id),
                    "direction": tuple(direction),
                    "marker_r0": int(marker_r0),
                    "marker_c0": int(marker_c0),
                    "marker_cells": marker_cells,
                    "pointer_cell": pointer_cell,
                }
            )
            occupied_marker_cells.extend(all_marker_cells)
            placed = True
            break
        if not placed:
            return None

    return out_inner, markers


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), int(src_bg), dtype=int)
    for type_id in source_type_ids:
        patch = sprite_types[int(type_id)]["src_patch"]
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
        paste(inner, patch, int(r0), int(c0), transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    palette_result = _assign_palette()
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, src_bg, dst_bg, connector_a, connector_b, body_colors = palette_result

    sprite_types = _build_sprite_types(src_bg, dst_bg, connector_a, connector_b, body_colors)
    if sprite_types is None:
        return None

    source_type_ids = _choose_source_type_ids()

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers_and_output(sprite_types, dst_h, dst_w, dst_bg, dst_border)
    if marker_result is None:
        return None
    out_inner, markers = marker_result

    directions = {tuple(marker["direction"]) for marker in markers}
    if len(directions) < 2 or all(tuple(marker["direction"]) == (-1, 0) for marker in markers):
        return None

    output = _build_frame(dst_h, dst_w, int(dst_border), int(dst_bg))
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, int(src_border), int(src_bg))
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, int(dst_border), int(dst_bg))
    for marker in markers:
        connector_color = int(sprite_types[int(marker["type_id"])]["connector"])
        for rr, cc in marker["marker_cells"]:
            dest_rect[1 + int(rr), 1 + int(cc)] = int(connector_color)
        pr, pc = marker["pointer_cell"]
        dest_rect[1 + int(pr), 1 + int(pc)] = int(dst_border)

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
    raise RuntimeError(f"failed to generate a251c730 combo D1/4+D3/3 after {MAX_GEN_ATTEMPTS} attempts")
