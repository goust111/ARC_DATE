# Variant of a251c730
# Dimension: D12 — 颜色约束体系 (meta)
# Change: 锚点颜色不在壁纸调色板中、不等于源背景色和目标背景色；边框颜色不等于内部背景色；总共7-9种不同颜色... → 颜色对应表编码

# Design analysis
# a) 具体化与行为追踪
#    - 仍然保留两块大矩形：源矩形里放多色精灵，目标矩形里只放单像素标记。
#    - 新增一个独立图例，直接贴在壁纸空白区上。图例固定为 2x4 的色块阵列：
#      顶行两对色块表示两个源精灵的锚点颜色，底行对应两对色块表示目标标记颜色。
#      每个映射对占两列，做成 [A A; X X] 这种小条块，避免 2x2 图例过小不易观察。
#    - 原题流程：目标区某个单点若是颜色 A，就找源区锚点颜色也为 A 的精灵并对齐印刻。
#    - 变体流程：目标区某个单点若是颜色 X，先去图例中找到底行颜色 X 所在那一对列，
#      读出同列顶行颜色 A，再去源区找锚点颜色为 A 的精灵，然后对齐印刻。
#    - 差异发生在“标记颜色 -> 精灵模板”的匹配环节。原题直接匹配；这里多一个“读图例解码”的步骤。
#
# b) 复杂度校准
#    - 精灵类型固定为 2 种，而不是偶尔退化成 1 种。这样图例至少包含两个映射对，
#      解题者能从少量样例里看到“上排颜色来自源精灵、下排颜色来自目标标记”的重复结构。
#    - 图例对的左右顺序会打乱，避免把规则误读成“按颜色大小排序后对应”。
#    - 目标区标记数量为 3-4 个，保证两种映射都会反复出现。
#
# c) 隐式假设审查
#    - 原 solver 用“标记颜色 == 锚点颜色”直接查模板；这个前提在变体中失效，必须先解码图例。
#    - 原 solver 构造 patch 后把非 mask 区域填成 0，并在读颜色时排除了颜色 0；
#      这等价于假设黑色不会是合法锚点。变体里不再做这个假设，所有颜色 0-9 都可能合法使用，
#      因此模板颜色统计只看组件真实颜色，定位锚点时依赖 mask 而不是排除颜色 0。
#    - 原 solver 用 (count, color_id) 选锚点；这里避免用颜色 id 打破平局，生成器直接保证锚点是唯一最少色。
#
# d) 放置合法性
#    - 目标标记仍然先用完整精灵试贴，只有能完整落在目标内框时才记录其锚点位置，所以输出印刻不会越界。
#    - 图例放在两个矩形之外，并与矩形至少隔 1 格，避免视觉混叠，也避免 solver 在矩形搜索时把图例误判成框。
#    - 图例只使用非壁纸色，因此矩形外所有“非壁纸残留”都来自图例，solver 可稳定定位。
#
# e) 视觉可观察性
#    - 变体的关键信号是：目标中的标记色在源矩形里找不到，但会完整地出现在图例底行；
#      源精灵锚点色则出现在图例顶行。多组样例里重复出现这一结构，人类可肉眼推断解码规则。
#    - 不使用颜色 id 排序、数值加减等不可见规则，图例中的同列上下关系就是唯一映射线索。
#
# f) 可视化整洁度
#    - 图例是单独的一小块 2x4 色块，不与矩形叠加；目标区仍只保留单点标记，整体布局和原题一样清楚。
#    - 精灵保持非对称形状，印刻结果容易和输入中的单点标记区分。
#
# g) 颜色变化可推断性
#    - 总颜色数固定为 8：3 个壁纸色 + 1 个源框色 + 2 个锚点色 + 2 个编码色。
#      目标框色和精灵主体色都复用壁纸色，不额外增加颜色负担。
#    - 两个编码色都与锚点色不同；若把同一输入交给原题 solver，目标区只会留下空框，
#      因而不会退化成原规则。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def _interior_bg(g: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != ibg, connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        patch[~mask] = 0

        vals = [int(inner[int(r), int(c)]) for r, c in coords]
        cols = Counter(vals)
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": {int(k): int(v) for k, v in cols.items()},
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
            }
        )

    comps.sort(key=lambda d: -int(d["size"]))
    return ibg, comps


def _find_rectangles(g: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], int, int]]:
    H, W = g.shape
    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(H):
        for c0 in range(W):
            bc = int(g[r0, c0])
            for r1 in range(r0 + min_h - 1, H):
                for c1 in range(c0 + min_w - 1, W):
                    if int(g[r0, c1]) != bc or int(g[r1, c0]) != bc or int(g[r1, c1]) != bc:
                        continue
                    if not bool((g[r0, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r1, c0 : c1 + 1] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c0] == bc).all()):
                        continue
                    if not bool((g[r0 : r1 + 1, c1] == bc).all()):
                        continue
                    ibg = _interior_bg(g, (r0, c0, r1, c1))
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))
    assert rects
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    return rects[:8]


def _extract_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items()}
        if len(cols) < 2:
            continue
        min_count = min(cols.values())
        anchor_candidates = [int(col) for col, count in cols.items() if int(count) == int(min_count)]
        if len(anchor_candidates) != 1:
            continue
        anchor_col = int(anchor_candidates[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere(mask & (patch == anchor_col))
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if anchor_col not in templates or int(mask.sum()) > int(templates[anchor_col][1].sum()):
            templates[anchor_col] = (patch.copy(), mask.copy(), (ar, ac))
    return templates


def _decode_legend(g: np.ndarray, src_rect: tuple[int, int, int, int], dst_rect: tuple[int, int, int, int]) -> dict[int, int]:
    outside = np.ones(g.shape, dtype=bool)
    for r0, c0, r1, c1 in (src_rect, dst_rect):
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    wallpaper = [int(col) for col, _count in Counter(map(int, g[outside].flatten())).most_common(3)]
    non_wallpaper = outside & ~np.isin(g, wallpaper)

    candidates: list[tuple[int, dict[int, int]]] = []
    for coords in components_mask(non_wallpaper, connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        if int(coords.shape[0]) != h * w:
            continue
        if h != 2 or w < 2 or w % 2 != 0:
            continue
        patch = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
        mapping: dict[int, int] = {}
        ok = True
        for c in range(0, w, 2):
            top = [int(v) for v in patch[0, c : c + 2]]
            bottom = [int(v) for v in patch[1, c : c + 2]]
            if len(set(top)) != 1 or len(set(bottom)) != 1:
                ok = False
                break
            anchor_col = int(top[0])
            code_col = int(bottom[0])
            if anchor_col == code_col or code_col in mapping:
                ok = False
                break
            mapping[code_col] = anchor_col
        if ok and mapping:
            candidates.append((len(mapping), mapping))

    assert candidates, "must find the external color legend"
    candidates.sort(reverse=True, key=lambda item: int(item[0]))
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by decoding marker colors through an external legend.

    Input structure:
    - Two bordered rectangles remain: the source holds multi-cell sprites, the destination holds singleton markers.
    - A separate 2-row color legend on the wallpaper pairs source anchor colors with different destination marker colors.

    Rule:
    - Read each legend column-pair to convert a destination marker color back into the matching source anchor color.
    - Select the source sprite whose rare anchor pixel has that decoded color, then stamp it so the anchor lands on the marker.
    - Return the cropped destination rectangle with a cleared interior plus the stamped sprites.

    Key insight:
    - Marker colors are symbolic codes now; the real lookup key lives in the legend, not directly in the destination rectangle.
    """
    g = np.array(grid, dtype=int)

    rects = _find_rectangles(g)

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"

    (sr, sc, sr1, sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates = _extract_templates(src_comps)
    legend = _decode_legend(g, (sr, sc, sr1, sc1), (dr, dc, dr1, dc1))

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        code_col = int(g[mr, mc])
        if code_col not in legend:
            continue
        anchor_col = int(legend[code_col])
        if anchor_col not in templates:
            continue
        patch, mask, (ar, ac) = templates[anchor_col]
        top = int((mr - dr) - int(ar))
        left = int((mc - dc) - int(ac))
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator ---------------

GRID_SIZE = 30
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

N_TYPES = 2
MARKER_COUNT_RANGE = (3, 4)
SOURCE_SPRITE_RANGE = (3, 5)
MARKER_PLACE_RETRIES = 30
SOURCE_SPRITE_PADDING = 1

LEGEND_PAIR_W = 2
LEGEND_H = 2
LEGEND_PADDING = 1

MAX_GEN_ATTEMPTS = 400

SPRITE_SHAPES = [
    (("AB.", "BBB"), (0, 0)),
    (("ABB", "BBB"), (0, 0)),
    ((".A.", "BBB", ".B."), (0, 1)),
    (("AB.", "BBB", ".BB"), (0, 0)),
    (("ABB", "BBB", "..B"), (0, 0)),
    ((".AB", "BBB", "BB."), (0, 1)),
    (("B.B", "BBB", ".A."), (2, 1)),
    ((".B.", "BAB", ".B."), (1, 1)),
    (("BBB", "BAB"), (1, 1)),
    (("AB.", "BBB", ".B."), (0, 0)),
    (("BAB", "BBB"), (0, 1)),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": anchor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    stripe = np.array(palette * ((GRID_SIZE + len(palette) - 1) // len(palette)), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = border
    rect[-1, :] = border
    rect[:, 0] = border
    rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


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


def _assign_palette() -> tuple[list[int], int, int, int, int, list[int], list[int]]:
    chosen = random.sample(list(range(10)), 8)
    wallpaper = chosen[:3]
    src_border = int(chosen[3])
    anchors = [int(chosen[4]), int(chosen[5])]
    codes = [int(chosen[6]), int(chosen[7])]
    random.shuffle(wallpaper)
    random.shuffle(anchors)
    random.shuffle(codes)
    body = int(wallpaper[1])
    dst_border = int(wallpaper[2])
    src_bg = int(wallpaper[0])
    dst_bg = int(wallpaper[0])
    return wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, codes


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            mr = int(r0 + sprite["anchor_rc"][0])
            mc = int(c0 + sprite["anchor_rc"][1])
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((mr, mc, type_id))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], src_bg)
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


def _build_legend_patch(anchor_colors: list[int], code_colors: list[int]) -> np.ndarray:
    pairs = list(zip(anchor_colors, code_colors))
    random.shuffle(pairs)
    cols: list[list[int]] = []
    for anchor, code in pairs:
        cols.extend([[int(anchor), int(code)]] * LEGEND_PAIR_W)
    return np.array(cols, dtype=int).T


def _place_legend(
    grid: np.ndarray,
    legend: np.ndarray,
    src_pos: tuple[int, int],
    src_h: int,
    src_w: int,
    dst_pos: tuple[int, int],
    dst_h: int,
    dst_w: int,
) -> bool:
    occ = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for (r, c), h, w in ((src_pos, src_h, src_w), (dst_pos, dst_h, dst_w)):
        r0 = max(0, int(r) - LEGEND_PADDING)
        c0 = max(0, int(c) - LEGEND_PADDING)
        r1 = min(GRID_SIZE, int(r) + int(h) + LEGEND_PADDING)
        c1 = min(GRID_SIZE, int(c) + int(w) + LEGEND_PADDING)
        occ[r0:r1, c0:c1] = 1
    probe = np.ones(legend.shape, dtype=int)
    try:
        lr, lc = random_free_location(occ, probe, transparent=0, border=1)
    except ValueError:
        return False
    paste(grid, legend, lr, lc)
    return True


def _gen_attempt() -> np.ndarray | None:
    wallpaper, src_border, dst_border, src_bg, dst_bg, anchors, codes = _assign_palette()

    sprite_specs = random.sample(SPRITE_SHAPES, N_TYPES)
    sprite_types = []
    for (shape_rows, anchor_rc), anchor, code in zip(sprite_specs, anchors, codes):
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": anchor_rc,
                "body": int(wallpaper[1]),
                "anchor": int(anchor),
                "code": int(code),
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
            }
        )

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, N_TYPES)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(3, N_TYPES), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, N_TYPES)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, type_id in marker_points:
        dest_rect[1 + mr, 1 + mc] = int(sprite_types[type_id]["code"])

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    legend = _build_legend_patch(anchors, codes)
    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])
    if not _place_legend(grid, legend, src_pos, src_h, src_w, dst_pos, dst_h, dst_w):
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D12 input after {MAX_GEN_ATTEMPTS} attempts")
