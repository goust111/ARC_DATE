# Variant of a251c730
# Dimension: D7 — 源-目标关系：基数映射 (meta)
# Change: 每种精灵类型在目标中恰好对应一个标记（一对一映射），源中可有同类型的多个副本但只有一个被使用... → 多对一映射（叠加）

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis:
# variant_spec:
#   core_change:
#     original_d7: "一个锚点颜色 -> 一个模板 -> 一个目标标记"
#     new_d7: "一个锚点颜色 -> 一组模板 -> 一个目标标记；整组都要在同一标记上叠印"
#   preserved_structure:
#     - 30x30 条纹壁纸背景
#     - 两个不重叠的有边框矩形：源矩形和目标矩形
#     - 源区仍是多像素双色精灵；目标区仍只放单像素标记
#     - 锚点仍由精灵中的唯一稀有色给出，并负责对齐
#   concrete_visualization:
#     - 每个锚点颜色在源区对应 2 个不同精灵：主体色不同，像素数也不同
#     - 目标区每种锚点颜色只出现 1 个单像素标记
#     - 输出时，同组 2 个精灵都平移到该标记上，并按“面积从大到小”印刻
#     - 因为后印刻的是更小的精灵，它会在中心覆盖前一个精灵，外围仍保留大精灵的颜色，叠加效果肉眼可见
# behavior_trace:
#   example:
#     - 源区里某个红色锚点组有两个精灵：7 像素的黄体精灵、5 像素的灰体精灵
#     - 目标区只有 1 个红色单像素标记
#     - 原任务会只选一个模板（通常代码里是选更大的那个），输出只会得到黄体精灵
#     - 新任务会先印黄体精灵，再在同一锚点位置印灰体精灵，最终得到“黄外框 + 灰内核”的复合图像
#   difference:
#     - 差别不在矩形检测，也不在锚点平移，而在“anchor color -> template”的基数从单个变成一组，并且组内需要有可见的覆盖顺序
# complexity_calibration:
#   - 每个样例固定 2 个锚点组，每组固定 2 个精灵；既能重复支撑规则，又不至于过乱
#   - 组内精灵尺寸强制不同，避免排序歧义；主体色强制不同，保证覆盖后能看见变化
#   - 目标区只有 2 个标记，复合图块彼此留空，不会混成一团
# implicit_assumption_audit:
#   - 原 `templates[anchor] = single_patch` 依赖“一种锚点色只需一个模板”；变体下必须改成 `anchor -> list[patch]`
#   - 原 `if anchor not in templates or bigger: keep one` 会丢弃同锚点的其他精灵；变体下这是错误的
#   - 原对目标标记的循环一次只触发一次印刻；变体下每个标记要触发同组多次印刻
#   - 新 solver 额外假设“同组精灵大小互不相同”，这样只按面积即可排序；generator 保证这一点
# placement_legality:
#   - 输出先构造：每组先在局部坐标里合成复合补丁，再把复合补丁作为一个对象放入目标内区，保证不越界且组间不相撞
#   - 输入目标区只回填组锚点单像素，不回填复合图案，因此仍能被 solver 识别为“目标矩形”
#   - 源区单独放置 4 个精灵，保留至少 1 像素间距，避免组件黏连
# observability_and_cleanliness:
#   - 排序规则只用可见属性“面积”，不用颜色 id
#   - 覆盖后的结果与“只取最大模板”的旧逻辑稳定不同，因为较小精灵的主体色会覆盖大精灵中心区域
#   - 组间不叠加，边框/壁纸/标记/精灵四类元素在视觉上分区清楚


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp grouped source sprites into a destination frame via shared-color singleton markers.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: the source holds multi-cell sprites, the destination holds singleton markers.
    - Several different source sprites may share the same rare anchor color, while differing in body color and footprint size.

    Rule:
    - For each destination marker, gather every source sprite whose anchor color matches the marker color.
    - Align all of those sprites to the same marker and stamp them in descending size order so later, smaller sprites overwrite earlier ones.
    - Return the destination rectangle alone, with its interior reset before the layered stamping.

    Key insight:
    - The least-frequent color still identifies the alignment anchor, but now it indexes an ordered stack of sprites rather than a single template.
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
            mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
            mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
            patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
            patch[~mask] = 0
            vals = [int(inner[int(r), int(c)]) for r, c in coords]
            cols = Counter(vals)
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(cols),
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
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if ibg == bc:
                        continue
                    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
                    rects.append((area, (int(r0), int(c0), int(r1), int(c1)), bc, ibg))

    assert rects

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    def component_template(comp: dict) -> tuple[int, np.ndarray, np.ndarray, tuple[int, int]]:
        cols = {k: int(v) for k, v in comp["colors"].items() if int(k) != 0}
        assert cols
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        assert anchors.size > 0
        ar, ac = map(int, anchors[0])
        return anchor_col, patch.copy(), mask.copy(), (int(ar), int(ac))

    # Group every source component by anchor color instead of keeping only one representative.
    templates: dict[int, list[tuple[np.ndarray, np.ndarray, tuple[int, int], int]]] = defaultdict(list)
    for comp in src_comps:
        anchor_col, patch, mask, anchor_rc = component_template(comp)
        templates[anchor_col].append((patch, mask, anchor_rc, int(mask.sum())))

    for anchor_col, items in templates.items():
        # Generator guarantees distinct sizes within a group, so visible size order is unambiguous.
        items.sort(key=lambda item: -int(item[3]))
        templates[anchor_col] = items

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
        mcol = int(g[mr, mc])
        for patch, mask, (ar, ac), _size in templates.get(int(mcol), []):
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

N_GROUPS = 2
GROUP_SIZE = 2
MARKER_PLACE_RETRIES = 30
SOURCE_SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400

SPRITE_SHAPES = [
    {"shape_rows": ("AB.", "BBB"), "anchor_rc": (0, 0), "size": 5},
    {"shape_rows": ("ABB", "BBB"), "anchor_rc": (0, 0), "size": 6},
    {"shape_rows": (".A.", "BBB", ".B."), "anchor_rc": (0, 1), "size": 5},
    {"shape_rows": ("AB.", "BBB", ".BB"), "anchor_rc": (0, 0), "size": 7},
    {"shape_rows": ("ABB", "BBB", "..B"), "anchor_rc": (0, 0), "size": 7},
    {"shape_rows": (".AB", "BBB", "BB."), "anchor_rc": (0, 1), "size": 7},
    {"shape_rows": ("B.B", "BBB", ".A."), "anchor_rc": (2, 1), "size": 6},
    {"shape_rows": (".B.", "BAB", ".B."), "anchor_rc": (1, 1), "size": 5},
    {"shape_rows": ("BBB", "BAB"), "anchor_rc": (1, 1), "size": 6},
    {"shape_rows": ("AB.", "BBB", ".B."), "anchor_rc": (0, 0), "size": 6},
    {"shape_rows": ("BAB", "BBB"), "anchor_rc": (0, 1), "size": 6},
]

SIZE_PAIRS = ((7, 5), (7, 6), (6, 5))


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": anchor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


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


def _assign_palette() -> dict | None:
    wallpaper_n = random.choice((4, 5))
    wallpaper = random.sample(list(range(10)), wallpaper_n)
    remaining = [c for c in range(10) if c not in wallpaper]
    random.shuffle(remaining)

    if len(remaining) < 1 + N_GROUPS:
        return None
    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(N_GROUPS)]

    src_bg, dst_bg = random.sample(wallpaper, 2)
    if remaining and random.random() < 0.5:
        dst_border = remaining.pop()
    else:
        border_choices = [c for c in wallpaper if c != dst_bg]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)

    body_pool = [c for c in wallpaper if c not in {src_bg, dst_bg}]
    if len(body_pool) < 2:
        return None

    return {
        "wallpaper": wallpaper,
        "src_border": src_border,
        "dst_border": dst_border,
        "src_bg": src_bg,
        "dst_bg": dst_bg,
        "anchors": anchors,
        "body_pool": body_pool,
    }


def _choose_sprite_groups(anchors: list[int], body_pool: list[int]) -> list[dict] | None:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for spec in SPRITE_SHAPES:
        by_size[int(spec["size"])].append(spec)

    used_ids: set[int] = set()
    groups = []
    for anchor in anchors:
        size_pair = random.choice(SIZE_PAIRS)
        body_a, body_b = random.sample(body_pool, 2)
        sprites = []
        for body, size in zip((body_a, body_b), size_pair):
            candidates = [spec for spec in by_size[int(size)] if id(spec) not in used_ids]
            if not candidates:
                return None
            spec = random.choice(candidates)
            used_ids.add(id(spec))
            sprites.append(
                {
                    "shape_rows": spec["shape_rows"],
                    "anchor_rc": tuple(spec["anchor_rc"]),
                    "body": int(body),
                    "anchor": int(anchor),
                    "size": int(spec["size"]),
                }
            )
        groups.append({"anchor": int(anchor), "sprites": sprites})
    return groups


def _compose_group_patch(group: dict, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    rendered = []
    min_r = 0
    min_c = 0
    max_r = 0
    max_c = 0
    for sprite in group["sprites"]:
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], bg)
        ar, ac = map(int, sprite["anchor_rc"])
        h, w = patch.shape
        min_r = min(min_r, -ar)
        min_c = min(min_c, -ac)
        max_r = max(max_r, h - ar - 1)
        max_c = max(max_c, w - ac - 1)
        rendered.append((sprite, patch))

    anchor_r = -min_r
    anchor_c = -min_c
    out_h = max_r - min_r + 1
    out_w = max_c - min_c + 1
    composite = np.full((out_h, out_w), bg, dtype=int)

    for sprite, patch in sorted(rendered, key=lambda item: -int(item[0]["size"])):
        ar, ac = map(int, sprite["anchor_rc"])
        top = anchor_r - ar
        left = anchor_c - ac
        paste(composite, patch, int(top), int(left), transparent=bg, overwrite=True)

    return composite, (int(anchor_r), int(anchor_c))


def _place_groups_in_dest(groups: list[dict], dst_h: int, dst_w: int, dst_bg: int) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    shuffled = groups[:]
    random.shuffle(shuffled)

    for group in shuffled:
        composite, anchor_rc = _compose_group_patch(group, dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, composite, transparent=dst_bg, padding=1)
            except ValueError:
                return None
            mr = int(r0 + anchor_rc[0])
            mc = int(c0 + anchor_rc[1])
            if any(abs(mr - pr) + abs(mc - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, composite, int(r0), int(c0), transparent=dst_bg)
            marker_points.append((mr, mc, int(group["anchor"])))
            placed = True
            break
        if not placed:
            return None

    return inner, marker_points


def _place_sources(groups: list[dict], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    sprites = [sprite for group in groups for sprite in group["sprites"]]
    random.shuffle(sprites)
    for sprite in sprites:
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
        paste(inner, patch, int(r0), int(c0), transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    palette = _assign_palette()
    if palette is None:
        return None

    groups = _choose_sprite_groups(palette["anchors"], palette["body_pool"])
    if groups is None:
        return None

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_groups_in_dest(groups, dst_h, dst_w, palette["dst_bg"])
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, palette["dst_border"], palette["dst_bg"])
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(groups, src_h, src_w, palette["src_bg"])
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, palette["src_border"], palette["src_bg"])
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, palette["dst_border"], palette["dst_bg"])
    for mr, mc, anchor in marker_points:
        dest_rect[1 + mr, 1 + mc] = int(anchor)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), palette["wallpaper"])
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
    raise RuntimeError(f"failed to generate a251c730 D7 variant after {MAX_GEN_ATTEMPTS} attempts")
