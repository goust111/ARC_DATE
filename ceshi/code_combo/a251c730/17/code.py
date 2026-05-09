# Combo variant of a251c730 (index 17)
# combo_spec: D2/3+D14/0
# Name: 大稀疏精灵+形状唯一

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# ===============
# design:
#   source_task: a251c730
#   concretization:
#     - 仍然是两块带边框矩形盖在条纹壁纸上: 源框里放精灵, 目标框里只放单像素标记.
#     - 与原题不同, 源框精灵不再是 2x3 / 3x3 紧凑块, 而是 5x5 或 6x6 外接框中的稀疏斜向骨架,
#       总像素仍只有 5-7 个, 空洞和长对角间距都属于形状的一部分.
#     - 与 D14/0 一样, 每个源精灵形状都唯一; 颜色只把精灵分成若干组, 同色组内靠阅读顺序配对.
#     - 由于 5-7 像素不可能在 4 连通下同时撑满 5x5/6x6 bbox, solver 必须把“源精灵是一个组件”的
#       识别改成 8 连通, 生成器也必须保证不同精灵之间至少隔开一个 8 邻域空隙.
#   behavior_trace:
#     - 原逻辑: 颜色 -> 单一模板. 若两个红锚点对应两个不同精灵, solver 只会保留其中一个模板,
#       所有红色标记都会盖成同一个形状.
#     - 融合后: 先在源框里按锚点颜色分组, 再按锚点阅读顺序形成模板序列.
#       目标框中同色单像素也按阅读顺序排序, 第 k 个标记接收该颜色的第 k 个唯一稀疏形状.
#     - 额外差异: 这些模板现在是大 bbox 的稀疏斜向结构, 输出里必须保留大片空洞,
#       所以“盖章”时的 mask 不能退化成紧凑实心块.
#   complexity_calibration:
#     - 每题固定 3-4 个源精灵, 但锚点颜色只有 2-3 种且严格少于精灵数, 保证至少一组颜色复用.
#     - 所有形状从稀疏模板库无放回抽取, 保证无重复类型; 同时所有 body 颜色统一, 让辨识主要依赖几何结构.
#     - 目标标记数量等于源精灵数量, 避免未使用模板, 也避免融合规则退化成“每色一个模板”的原题行为.
#   implicit_assumption_audit:
#     - 原 main 的 `templates[color] = single_template` 假设失效, 必须改成 `color -> ordered list`.
#     - 原 main 用 4 连通提取组件; 对本变体的稀疏对角骨架会被错误拆成多个单像素, 因此必须改成 8 连通.
#     - 原 main 只要找到最小频次颜色就可当锚点; 这里仍成立, 但生成器必须保证该最小频次颜色恰为单像素.
#     - 目标配对不能用颜色值大小、面积大小等不可视因素; 只能用人眼可见的阅读顺序.
#   legality:
#     - 源框放置使用 `random_free_location(..., padding=1, padding_connectivity=8)`, 保证不同稀疏骨架不粘连.
#     - 目标输出也是在空白 inner canvas 上先真实盖印, 再反推单像素标记, 因此不会越界或产生未定义重叠.
#     - 同色组内要求目标锚点阅读顺序严格递增, 使颜色组序列配对有唯一解.
#   observability:
#     - 输入里同色单像素会重复出现, 但输出里这些同色位置出现的是不同的大稀疏形状, 人眼能直接看出
#       “颜色分组 + 组内按顺序回放唯一形状”的规律.
#     - 大 bbox + 极低像素密度使得空洞保留非常显眼, 与原题的小紧凑块有明确视觉差异.
#   cleanliness:
#     - 源框和目标输出都对 8 邻域留空, 防止稀疏骨架互相粘成难分的团块.
#     - 目标输入仍只有单像素标记, 与输出的完整印章层次分明, 不会把线索和答案混在同一层.
#   color_inference:
#     - 颜色种类控制在 2-3 个锚点色, 颜色只负责分组而不负责组内唯一识别, 这样少量样例也能推断.
#     - 因为 body 颜色统一, 解题者会自然转向比较形状差异而不是依赖多色模板记忆.


def _interior_bg(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(
    grid: np.ndarray, rect: tuple[int, int, int, int], *, connectivity: int = 8
) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != int(ibg), connectivity=connectivity):
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


def _find_rectangles(grid: np.ndarray) -> tuple[tuple, tuple]:
    g = np.array(grid, dtype=int)
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
                    rects.append((area, (r0, c0, r1, c1), bc, ibg))

    assert rects
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect, connectivity=8)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"
    return src, dst


def _anchor_info(comp: dict) -> tuple[int, tuple[int, int]] | None:
    cols = {int(k): int(v) for k, v in comp["colors"].items()}
    if not cols:
        return None
    min_count = min(cols.values())
    anchor_candidates = [color for color, count in cols.items() if int(count) == int(min_count)]
    if len(anchor_candidates) != 1:
        return None
    anchor_col = int(anchor_candidates[0])
    anchors = np.argwhere((comp["patch"] == int(anchor_col)) & comp["mask"])
    if anchors.shape[0] != 1:
        return None
    ar, ac = map(int, anchors[0])
    return int(anchor_col), (int(ar), int(ac))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replay large sparse source sprites into a destination frame by color-grouped marker order.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: the source box contains several large sparse multi-pixel sprites, and the destination box contains only singleton anchor-color markers.
    - Every source sprite uses exactly two colors, with a one-pixel anchor color, but multiple different sparse shapes may share that anchor color.

    Rule:
    - Extract each source sprite as an 8-connected sparse stencil, group those stencils by anchor color, and sort each color group by source-anchor reading order.
    - Group destination singleton markers by color, sort those markers in reading order, and place the kth sparse source stencil of a color onto the kth marker of the same color with anchor alignment.
    - Output only the destination rectangle, reset to its background before stamping the matched sparse sprites.

    Key insight: Color only chooses the group; the actual template comes from the ordered family of unique large sparse shapes inside that color group.
    """
    g = np.array(grid, dtype=int)
    src, dst = _find_rectangles(g)

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, list[tuple[tuple[int, int], np.ndarray, np.ndarray, tuple[int, int]]]] = defaultdict(list)
    for comp in src_comps:
        anchor = _anchor_info(comp)
        if anchor is None:
            continue
        anchor_col, (ar, ac) = anchor
        rr0, cc0, _rr1, _cc1 = comp["bbox"]
        anchor_in_src = (int(rr0 + ar), int(cc0 + ac))
        templates[anchor_col].append((anchor_in_src, comp["patch"].copy(), comp["mask"].copy(), (ar, ac)))

    for color in list(templates):
        templates[color].sort(key=lambda item: item[0])

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    markers_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    inner = g[dr + 1 : dr1, dc + 1 : dc1]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        markers_by_color[int(inner[int(rr), int(cc)])].append((int(rr), int(cc)))

    for color in list(markers_by_color):
        markers_by_color[color].sort()

    for color, markers in markers_by_color.items():
        color_templates = templates.get(int(color), [])
        for (_src_anchor, patch, mask, (ar, ac)), (rr, cc) in zip(color_templates, markers):
            top = int(rr + 1 - ar)
            left = int(cc + 1 - ac)
            paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator ---------------

GRID_SIZE = 30
WALLPAPER_COLORS = 4
GAP_RANGE = (1, 4)

SRC_H_RANGE = (12, 19)
SRC_W_RANGE = (10, 18)
DST_H_RANGE = (11, 19)
DST_W_RANGE = (10, 20)
DST_W_SIDE_MAX = 20

UNIQUE_SPRITE_COUNT_RANGE = (3, 4)
ANCHOR_COLOR_COUNT_CHOICES = (2, 3)
MARKER_PLACE_ATTEMPTS = 40
SOURCE_SPRITE_PADDING = 1
OUTPUT_SPRITE_PADDING = 1
MARKER_MIN_MANHATTAN = 3
FOUR_SPRITE_SIDE_MIN = (14, 15, 13, 13)
FOUR_SPRITE_STACKED_MIN = (15, 14, 11, 15)
BRANCH_SPECS = (
    (3, 2, "side_by_side"),
    (3, 2, "stacked"),
    (4, 2, "side_by_side"),
    (4, 2, "stacked"),
    (4, 3, "side_by_side"),
    (4, 3, "stacked"),
)

MAX_GEN_ATTEMPTS = 600

SPARSE_SHAPES = [
    {"shape": (5, 5), "cells": ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4))},
    {"shape": (5, 5), "cells": ((0, 4), (1, 3), (2, 2), (3, 1), (4, 0))},
    {"shape": (5, 5), "cells": ((0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 4))},
    {"shape": (5, 5), "cells": ((0, 4), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0))},
    {"shape": (5, 5), "cells": ((0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (4, 4))},
    {"shape": (5, 5), "cells": ((0, 4), (1, 3), (2, 3), (3, 2), (4, 1), (4, 0))},
    {"shape": (6, 6), "cells": ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))},
    {"shape": (6, 6), "cells": ((0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0))},
    {"shape": (6, 6), "cells": ((0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 5))},
    {"shape": (6, 6), "cells": ((0, 5), (0, 4), (1, 3), (2, 2), (3, 1), (4, 0), (5, 0))},
    {"shape": (6, 6), "cells": ((0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (5, 5))},
    {"shape": (6, 6), "cells": ((0, 5), (1, 4), (2, 4), (3, 3), (4, 2), (5, 1), (5, 0))},
    {"shape": (6, 6), "cells": ((0, 1), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (5, 5))},
    {"shape": (6, 6), "cells": ((0, 4), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1), (5, 0))},
    {"shape": (6, 6), "cells": ((0, 0), (0, 1), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5))},
    {"shape": (6, 6), "cells": ((0, 5), (0, 4), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0))},
]


def _render_sprite(sprite: dict, bg: int) -> np.ndarray:
    patch = np.full((int(sprite["h"]), int(sprite["w"])), int(bg), dtype=int)
    ar, ac = sprite["anchor_rc"]
    patch[int(ar), int(ac)] = int(sprite["anchor"])
    for rr, cc in sprite["body_coords"]:
        patch[int(rr), int(cc)] = int(sprite["body"])
    return patch


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


def _choose_counts(n_total: int, n_bins: int) -> list[int]:
    counts = [1] * n_bins
    for _ in range(n_total - n_bins):
        counts[random.randrange(n_bins)] += 1
    return counts


def _choose_dimensions(layout: str, n_sprites: int) -> tuple[int, int, int, int]:
    src_h_min, src_h_max = SRC_H_RANGE
    src_w_min, src_w_max = SRC_W_RANGE
    dst_h_min, dst_h_max = DST_H_RANGE
    dst_w_min, dst_w_max = DST_W_RANGE
    if n_sprites == 4:
        if layout == "side_by_side":
            min_src_h, min_src_w, min_dst_h, min_dst_w = FOUR_SPRITE_SIDE_MIN
        else:
            min_src_h, min_src_w, min_dst_h, min_dst_w = FOUR_SPRITE_STACKED_MIN
        src_h_min = max(src_h_min, min_src_h)
        src_w_min = max(src_w_min, min_src_w)
        dst_h_min = max(dst_h_min, min_dst_h)
        dst_w_min = max(dst_w_min, min_dst_w)

    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(src_w_min, src_w_max)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < dst_w_min:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(dst_w_min, dst_w_max)
        src_h = random.randint(src_h_min, src_h_max)
        dst_h = random.randint(dst_h_min, dst_h_max)
    else:
        src_h = random.randint(src_h_min, src_h_max)
        dst_h_max = min(dst_h_max, GRID_SIZE - gap - src_h)
        if dst_h_max < dst_h_min:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(dst_h_min, dst_h_max)
        src_w = random.randint(src_w_min, src_w_max)
        dst_w = random.randint(dst_w_min, dst_w_max)
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


def _assign_palette(n_anchor_colors: int) -> tuple[list[int], int, int, int, list[int]]:
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, WALLPAPER_COLORS)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    body = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_anchor_colors)]
    dst_border = random.choice(wallpaper)
    return wallpaper, src_border, dst_border, body, anchors


def _place_sources(
    sprite_instances: list[dict], src_h: int, src_w: int, src_bg: int
) -> tuple[np.ndarray, list[dict]] | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    placed_entries: list[dict] = []
    for sprite in sprite_instances:
        patch = _render_sprite(sprite, src_bg)
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=1,
                padding=SOURCE_SPRITE_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=src_bg)
        ar, ac = sprite["anchor_rc"]
        placed_entries.append(
            {
                **sprite,
                "src_anchor": (int(r0 + ar), int(c0 + ac)),
            }
        )
    return inner, placed_entries


def _valid_output_locations(
    canvas: np.ndarray,
    patch: np.ndarray,
    anchor_rc: tuple[int, int],
    bg: int,
    existing_markers: list[tuple[int, int]],
    last_anchor: tuple[int, int] | None,
) -> list[tuple[int, int, int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    ar, ac = anchor_rc
    mask_coords = np.argwhere(patch != int(bg))
    existing_pixels = [tuple(map(int, rc)) for rc in np.argwhere(canvas != int(bg))]
    candidates: list[tuple[int, int, int, int]] = []

    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in mask_coords]
            if any(int(canvas[rr, cc]) != int(bg) for rr, cc in coords):
                continue
            if any(
                max(abs(rr - er), abs(cc - ec)) <= OUTPUT_SPRITE_PADDING
                for rr, cc in coords
                for er, ec in existing_pixels
            ):
                continue
            mr = int(r0 + ar)
            mc = int(c0 + ac)
            if any(abs(mr - pr) + abs(mc - pc) < MARKER_MIN_MANHATTAN for pr, pc in existing_markers):
                continue
            if last_anchor is not None and (mr, mc) <= last_anchor:
                continue
            candidates.append((int(r0), int(c0), mr, mc))

    return candidates


def _place_markers_and_output(
    placed_entries: list[dict], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for entry in placed_entries:
        grouped[int(entry["anchor"])].append(entry)
    for color in list(grouped):
        grouped[color].sort(key=lambda item: item["src_anchor"])

    color_order = list(grouped)
    for _ in range(MARKER_PLACE_ATTEMPTS):
        random.shuffle(color_order)
        inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
        marker_points: list[tuple[int, int, int]] = []
        success = True
        for color in color_order:
            last_anchor: tuple[int, int] | None = None
            for entry in grouped[color]:
                patch = _render_sprite(entry, dst_bg)
                candidates = _valid_output_locations(
                    inner,
                    patch,
                    entry["anchor_rc"],
                    dst_bg,
                    [(r, c) for r, c, _ in marker_points],
                    last_anchor,
                )
                if not candidates:
                    success = False
                    break
                r0, c0, mr, mc = random.choice(candidates)
                paste(inner, patch, r0, c0, transparent=dst_bg)
                marker_points.append((int(mr), int(mc), int(color)))
                last_anchor = (int(mr), int(mc))
            if not success:
                break
        if success:
            return inner, marker_points
    return None


def _gen_attempt(
    n_sprites: int | None = None, n_anchor_colors: int | None = None, layout: str | None = None
) -> np.ndarray | None:
    if n_sprites is None:
        n_sprites = random.randint(*UNIQUE_SPRITE_COUNT_RANGE)
    max_anchor_colors = min(ANCHOR_COLOR_COUNT_CHOICES[1], n_sprites - 1)
    if n_anchor_colors is None:
        n_anchor_colors = random.randint(ANCHOR_COLOR_COUNT_CHOICES[0], max_anchor_colors)
    if not (ANCHOR_COLOR_COUNT_CHOICES[0] <= n_anchor_colors <= max_anchor_colors):
        return None

    wallpaper, src_border, dst_border, body, anchor_palette = _assign_palette(n_anchor_colors)
    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    shape_specs = random.sample(SPARSE_SHAPES, n_sprites)
    anchor_counts = _choose_counts(n_sprites, n_anchor_colors)
    sprite_anchor_colors = [anchor_palette[i] for i, count in enumerate(anchor_counts) for _ in range(count)]
    random.shuffle(sprite_anchor_colors)

    sprite_instances = []
    for shape_spec, anchor in zip(shape_specs, sprite_anchor_colors):
        cells = [tuple(map(int, cell)) for cell in shape_spec["cells"]]
        sprite_instances.append(
            {
                "h": int(shape_spec["shape"][0]),
                "w": int(shape_spec["shape"][1]),
                "anchor_rc": tuple(map(int, cells[0])),
                "body_coords": [tuple(map(int, cell)) for cell in cells[1:]],
                "body": int(body),
                "anchor": int(anchor),
            }
        )

    if layout is None:
        layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout, n_sprites)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    source_result = _place_sources(sprite_instances, src_h, src_w, src_bg)
    if source_result is None:
        return None
    source_inner, placed_entries = source_result

    marker_result = _place_markers_and_output(placed_entries, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for mr, mc, color in marker_points:
        dest_rect[1 + mr, 1 + mc] = int(color)

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
    branch_specs = list(BRANCH_SPECS)
    random.shuffle(branch_specs)
    attempts_per_branch = max(1, MAX_GEN_ATTEMPTS // len(branch_specs))

    for n_sprites, n_anchor_colors, layout in branch_specs:
        for _ in range(attempts_per_branch):
            grid = _gen_attempt(n_sprites=n_sprites, n_anchor_colors=n_anchor_colors, layout=layout)
            if grid is not None:
                return grid

    for _ in range(MAX_GEN_ATTEMPTS - attempts_per_branch * len(branch_specs)):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo a251c730 input after %d attempts" % MAX_GEN_ATTEMPTS)
