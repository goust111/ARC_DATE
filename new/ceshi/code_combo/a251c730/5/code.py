# Combo variant of a251c730 (index 5)
# combo_spec: D1/1+D12/2
# Name: 主体色匹配+颜色映射表
#
# Design analysis
# a) 具体化与行为追踪
#    - 仍然保留两块带边框的大矩形：源矩形里放 2 类双颜色精灵，目标矩形里只放若干单像素标记。
#    - 额外在壁纸空白区放一个独立图例，形状固定为 2x4 色块阵列；每种类型占两列，形成 [B B; X X]。
#      这里 B 是该精灵的主体色，X 是目标标记里实际出现的编码色。
#    - 具体追踪一类精灵：若源里某个精灵主体色是 7、锚点色是 2，而图例给出 [7 7; 4 4]，
#      那么目标区看到单点 4 时，不能像原题那样直接找锚点色 4，也不能像 D1 单变体那样直接找主体色 4。
#      正确流程是先经图例把 4 解码成主体色 7，再去源区选主体色为 7 的精灵，最后仍让锚点色 2 落到那个单点位置。
#    - 可见差异有两层：目标标记色既不等于锚点色，也不等于主体色；输出落到该位置的却仍是锚点像素。
#
# b) 复杂度校准
#    - 精灵类型固定为 2 种，避免 1 种时图例和主体匹配都退化为“看见什么都行”。
#    - 目标标记数量固定在 3-4，且两类都至少出现一次；源里也保证两类都至少出现一次。
#      这样同一输入内会重复出现“编码色 -> 主体色 -> 精灵”的链条，人眼能积累证据。
#    - 图例只给少量离散映射，不引入过多自由度；左右顺序会打乱，避免误读成按位置固定对应。
#
# c) 隐式假设审查
#    - 原 solver 把“目标标记颜色”直接当模板 key；融合后这个前提失效，必须先解码图例。
#    - D1 单变体把模板 key 从锚点改为主体；融合后这一步仍成立，但主体色只在解码后才得到。
#    - 原 solver 用最少色直接选模板并同时定位锚点；融合后最少色只负责定位，最多色才是模板 key。
#    - 原代码曾过滤掉颜色 0；这里不再依赖该假设，颜色统计全部只看组件真实像素，非 mask 区域填 0 不参与统计。
#    - 图例解码器隐含假设“同一编码只映射一个主体、同一主体只对应一种类型”；生成器显式保证主体色唯一、编码色唯一。
#
# d) 放置合法性
#    - 目标区仍采用 output-first：先把完整精灵试贴进目标内框，再记录其锚点坐标并改写成编码单点，
#      因而主 solver 回贴时不会越界。
#    - 源区精灵继续使用带 padding 的随机放置，保证彼此和边界至少隔 1 格。
#    - 图例单独放在两个矩形外，并对两个矩形外扩 1 格后做避让，避免混叠，也避免靠太近产生视觉歧义。
#
# e) 视觉可观察性
#    - 解题者能直接看到：目标标记色在源精灵里找不到，但会出现在图例底行；图例顶行颜色则恰好出现在源精灵主体区域。
#    - 输出里落在标记位置上的却不是图例里的任何编码色，而是源精灵的锚点色，因此“主体只负责选模板，锚点只负责落点”是可见的。
#    - 不使用颜色编号大小规律；唯一线索就是图例上下同行成对关系与源精灵里的多数色。
#
# f) 可视化整洁度
#    - 图例尺寸小且独立，不叠到矩形上；目标区仍只有孤立单点，输入分区清晰。
#    - 精灵形状保持原题那批非对称模板，回贴后和单像素标记差异明显，不会看成简单颜色替换。
#
# g) 变化的可推断性
#    - 颜色变化被压缩成 2 对可见映射，不会出现大量无提示的重编码。
#    - 生成器保证编码色既不等于主体色也不等于锚点色，所以若忽略图例或误把图例顶行当锚点表，都会得到错误结果。
#    - 对 main() 的新假设，生成器逐项保证：
#      1. 每个源组件恰有唯一主体色和唯一锚点色。
#      2. 图例是矩形外唯一的非壁纸连通块，并满足 2 行、偶数列、每两列一组纯色。
#      3. 每个目标编码都能经图例解到唯一主体色，且该主体色在源模板表里存在。

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

    for coords in components_mask(inner != int(ibg), connectivity=4):
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
    return int(ibg), comps


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


def _dominant_and_anchor(comp: dict) -> tuple[int, int] | None:
    items = [(int(col), int(cnt)) for col, cnt in comp["colors"].items()]
    if len(items) < 2:
        return None
    max_count = max(cnt for _, cnt in items)
    min_count = min(cnt for _, cnt in items)
    if max_count == min_count:
        return None
    body_colors = [col for col, cnt in items if cnt == max_count]
    anchor_colors = [col for col, cnt in items if cnt == min_count]
    if len(body_colors) != 1 or len(anchor_colors) != 1:
        return None
    return int(body_colors[0]), int(anchor_colors[0])


def _extract_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        key_colors = _dominant_and_anchor(comp)
        if key_colors is None:
            continue
        body_col, anchor_col = key_colors
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere(mask & (patch == int(anchor_col)))
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if body_col not in templates or int(mask.sum()) > int(templates[body_col][1].sum()):
            templates[body_col] = (patch.copy(), mask.copy(), (ar, ac))
    return templates


def _decode_legend(
    g: np.ndarray, src_rect: tuple[int, int, int, int], dst_rect: tuple[int, int, int, int]
) -> dict[int, int]:
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
        if h != 2 or w < 4 or w % 2 != 0:
            continue

        patch = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
        mapping: dict[int, int] = {}
        used_body: set[int] = set()
        ok = True
        for c in range(0, w, 2):
            top = [int(v) for v in patch[0, c : c + 2]]
            bottom = [int(v) for v in patch[1, c : c + 2]]
            if len(set(top)) != 1 or len(set(bottom)) != 1:
                ok = False
                break
            body_col = int(top[0])
            code_col = int(bottom[0])
            if body_col == code_col or code_col in mapping or body_col in used_body:
                ok = False
                break
            mapping[code_col] = body_col
            used_body.add(body_col)
        if ok and len(mapping) >= 2:
            candidates.append((len(mapping), mapping))

    assert candidates, "must find the external code-to-body legend"
    candidates.sort(reverse=True, key=lambda item: int(item[0]))
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by decoding each marker into a body-color lookup key.

    Input structure:
    - Two bordered rectangles still sit on a striped wallpaper: the source contains multi-cell two-color sprites, while the destination contains isolated singleton markers.
    - A separate two-row legend on the wallpaper pairs each sprite's dominant body color with a different marker code color.

    Rule:
    - Read the legend first to convert every destination marker color into the matching source body color.
    - Use that decoded body color to choose the source sprite, then place the sprite so its rare anchor pixel lands on the marker cell.
    - Return only the destination rectangle, cleared to its border/background before the decoded stamping.

    Key insight: The marker color is only a code; the legend reveals the real template key, and that key is the sprite's majority color rather than its anchor.
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
        body_col = int(legend[code_col])
        if body_col not in templates:
            continue
        patch, mask, (ar, ac) = templates[body_col]
        top = int((mr - dr) - int(ar))
        left = int((mc - dc) - int(ac))
        paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


GRID_SIZE = 30
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

WALLPAPER_N = 3
N_TYPES = 2
MARKER_COUNT_RANGE = (3, 4)
SOURCE_SPRITE_RANGE = (3, 5)
MARKER_PLACE_RETRIES = 30
SOURCE_SPRITE_PADDING = 1

LEGEND_PAIR_W = 2
LEGEND_PADDING = 1

MAX_GEN_ATTEMPTS = 500

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


def _assign_palette() -> tuple[list[int], int, int, int, int, list[int], list[int], list[int]]:
    chosen = random.sample(list(range(10)), 10)
    wallpaper = [int(v) for v in chosen[:WALLPAPER_N]]
    src_border = int(chosen[WALLPAPER_N])
    bodies = [int(chosen[WALLPAPER_N + 1]), int(chosen[WALLPAPER_N + 2])]
    anchors = [int(chosen[WALLPAPER_N + 3]), int(chosen[WALLPAPER_N + 4])]
    codes = [int(chosen[WALLPAPER_N + 5]), int(chosen[WALLPAPER_N + 6])]
    random.shuffle(wallpaper)
    random.shuffle(bodies)
    random.shuffle(anchors)
    random.shuffle(codes)
    src_bg = int(wallpaper[0])
    dst_bg = int(wallpaper[0])
    dst_border = int(wallpaper[1])
    return wallpaper, src_border, dst_border, src_bg, dst_bg, bodies, anchors, codes


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


def _build_legend_patch(body_colors: list[int], code_colors: list[int]) -> np.ndarray:
    pairs = list(zip(body_colors, code_colors))
    random.shuffle(pairs)
    cols: list[list[int]] = []
    for body, code in pairs:
        cols.extend([[int(body), int(code)]] * LEGEND_PAIR_W)
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
    wallpaper, src_border, dst_border, src_bg, dst_bg, bodies, anchors, codes = _assign_palette()

    sprite_specs = random.sample(SPRITE_SHAPES, N_TYPES)
    sprite_types = []
    for (shape_rows, anchor_rc), body, anchor, code in zip(sprite_specs, bodies, anchors, codes):
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": anchor_rc,
                "body": int(body),
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

    legend = _build_legend_patch(bodies, codes)
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
    raise RuntimeError(f"failed to generate a251c730 combo input after {MAX_GEN_ATTEMPTS} attempts")
