# Variant of a251c730
# Dimension: D8 — 精灵锚点定义 (meta)
# Change: 锚点是精灵中出现次数最少的颜色的像素（单一像素），作为匹配键和定位参考点... → 锚点为特定角落像素
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     old_rule: source sprite 的稀有色单像素既是匹配键，也是贴到目标框时的对齐点
#     new_rule: source sprite 的锚点固定为其外接矩形左上角像素；marker 颜色也改为这个左上角像素颜色
#   concrete_pixel_behavior:
#     source_rect:
#       - 仍然放若干多色 sprite，sprite 仍然只有两种前景色
#       - 但这里的单像素异色只是一种内部点缀，不再承担“锚点”职责
#       - 为了让左上角规则可见，所有可选 sprite 都保证 bbox 左上角是实体像素，且该像素使用主体色
#     dest_rect:
#       - 输入中只保留若干单像素 marker
#       - marker 的颜色等于对应 sprite 的 bbox 左上角像素颜色
#       - 输出中把整个 sprite 贴到目标框，使 sprite 的左上角覆盖 marker 位置
#   behavior_trace:
#     small_example:
#       - 若源 sprite 为
#           B B B
#           B A B
#         旧逻辑会把 A 当锚点，marker 必须画成 A，输出时让 A 对齐 marker。
#       - 新逻辑改成把左上角那个 B 当锚点，marker 画成 B，输出时让整块 sprite 的左上角对齐 marker。
#       - 差异是：匹配键从“罕见颜色”变成“固定角落颜色”；定位参考也从内部单点变成 bbox 角点。
#   complexity_calibration:
#     - 仍保留 1-2 种 sprite type、2-3 个 marker、2-4 个 source sprite，避免任务被简化成单对象平移。
#     - 每个 type 在源框和目标框中都会重复出现，给人类足够样本支撑。
#     - sprite 仍然是 2x3 或 3x3 级别的小图案，复杂度接近原题。
#   implicit_assumption_audit_of_original_main:
#     - `anchor_col = min(cols.items(), key=lambda kv: (count, color))[0]`
#       依赖“某个颜色必然是稀有单像素且可见地区分 template”。新变体下这个前提失效，必须改成读取 bbox 左上角像素。
#     - `anchors = np.argwhere((patch == anchor_col) & mask)` 与 `(ar, ac)=anchors[0]`
#       依赖“锚点位置由颜色反查得到”。新变体下锚点位置固定为 `(0, 0)`，不应该再从颜色搜索。
#     - `templates` 用 anchor color 作为 key
#       原本 marker 颜色就是稀有色；现在 marker 颜色必须改成左上角像素颜色，生成器和 solver 都要同步。
#     - 目标框遍历 marker 的逻辑本身仍成立，因为 marker 仍然是 singleton pixels。
#   placement_legality:
#     - 生成器放置输出 sprite 时，仍然先把整块 patch 摆进目标框内部，再反推输入 marker，因此一定不会越界。
#     - source sprite 仍使用带 padding 的随机摆放，避免源框中相互接触导致组件合并。
#     - 目标 marker 记录的是 patch 左上角位置，因此贴回时天然合法。
#   visual_observability:
#     - 只选“左上角是主体色 B”的 sprite；内部仍保留单像素 A。
#     - 因此输入中的 marker 会匹配到 source sprite 的角落主体色，而不是那个醒目的单像素异色。
#     - 从 2-3 组样例里，人类可以看到：marker 总在输出 sprite 的左上角，不在异色单点上。
#   cleanliness:
#     - 目标框输入只含 singleton markers，输出只含不重叠 sprite，层次与原题一致，不增加额外噪声。
#   color_inference:
#     - 不做任何按颜色 id 的排序或变换。
#     - 颜色只承担“同色匹配 template”的可见角色；关键规则由角落位置结构提供，而不是数值大小。
#   new_main_assumptions_and_generator_guarantees:
#     - solver 假设 source sprite 的 bbox 左上角一定是实体像素；generator 只采样满足此条件的 shape。
#     - solver 假设不同 sprite type 的左上角颜色可区分；generator 为不同 type 分配不同主体色。
#     - solver 假设目标 marker 颜色来自这些左上角颜色；generator 用输出 patch[0,0] 回填 marker。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by copying source sprites to corner-colored markers.

    Input structure:
    - The scene still contains two bordered rectangles on top of a striped wallpaper.
    - The source rectangle holds multicolor sprites; the destination rectangle holds singleton marker pixels.
    - Each sprite's template key and placement anchor is the pixel at the top-left corner of its tight bounding box.

    Rule:
    - Read each source sprite as a bounded patch and use its top-left pixel color as the matching key.
    - For every destination marker of that color, paste the matching sprite so the sprite's top-left corner lands on the marker.
    - Return the destination rectangle with its interior cleared first, then restamped with the matched sprites.

    Key insight: the decisive anchor is a fixed geometric corner of the sprite, not the rare singleton accent color inside it.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
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
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and comps and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for comp in src_comps:
        patch = comp["patch"]
        mask = comp["mask"]
        if not bool(mask[0, 0]):
            continue
        corner_col = int(patch[0, 0])
        if int(corner_col) == 0:
            continue
        if int(corner_col) not in templates or int(mask.sum()) > int(templates[int(corner_col)][1].sum()):
            templates[int(corner_col)] = (patch.copy(), mask.copy())

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    for rr, cc in np.argwhere(inner != int(dst_bg)):
        mr = int(dr + 1 + int(rr))
        mc = int(dc + 1 + int(cc))
        mcol = int(g[mr, mc])
        if int(mcol) not in templates:
            continue
        patch, mask = templates[int(mcol)]
        top = int(mr - dr)
        left = int(mc - dc)
        paste(out, patch, int(top), int(left), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
MARKER_PLACE_RETRIES = 25
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 300

# 'B' is the top-left corner/key color. 'A' is only an internal singleton accent.
SPRITE_SHAPES = [
    (("BBB", "BAB"), (1, 1)),
    (("B.B", "BBB", ".A."), (2, 1)),
    (("BBB", ".AB", ".B."), (1, 1)),
    (("BBB", "B.A"), (1, 2)),
    (("BBB", "AB.", "B.."), (1, 0)),
    (("BBB", "B..", "A.B"), (2, 0)),
    (("BBB", ".BA", ".BB"), (1, 2)),
    (("BBB", "B.B", ".A."), (2, 1)),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, accent: int, bg: int) -> np.ndarray:
    """Render a sprite patch where 'B' is the corner/key color and 'A' is a singleton accent."""
    char_map = {"B": body, "A": accent}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    """Create a 30x30 striped wallpaper (vertical or horizontal)."""
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    """Create an h x w rectangle with a 1-pixel border and interior filled with bg."""
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    """Distribute n_total items across n_types bins, each bin >= 1."""
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    """Pick (src_h, src_w, dst_h, dst_w) that fit the 30x30 canvas for the given layout."""
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
    """Compute non-overlapping (row, col) origins for src and dst rectangles."""
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    """Allocate wallpaper, borders, singleton accent colors, and body/key colors."""
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    accent_pool = [c for c in remaining if c != 0]
    if len(accent_pool) < n_types:
        return None
    accents = random.sample(accent_pool, n_types)
    remaining = [c for c in remaining if c not in accents]

    dst_border_extra = extras_needed >= 2 or (extras_needed == 1 and random.random() < 0.5)
    body_extra = extras_needed >= 2 or (extras_needed >= 1 and not dst_border_extra)

    if dst_border_extra:
        if not remaining:
            return None
        dst_border = remaining.pop()
    else:
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)

    body_pool = wallpaper[:]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, accents, body_pool, forced_body


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]] | None:
    """Place output sprites in the dest interior; record each sprite's top-left marker location."""
    inner = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    marker_points: list[tuple[int, int, int]] = []
    for type_id in marker_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["accent"], dst_bg)
        placed = False
        for _ in range(MARKER_PLACE_RETRIES):
            try:
                r0, c0 = random_free_location(inner, patch, transparent=dst_bg)
            except ValueError:
                return None
            if any(abs(r0 - pr) + abs(c0 - pc) <= 1 for pr, pc, _ in marker_points):
                continue
            paste(inner, patch, r0, c0, transparent=dst_bg)
            marker_points.append((int(r0), int(c0), int(type_id)))
            placed = True
            break
        if not placed:
            return None
    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    """Place sprites into the source interior, returning stamped_inner or None."""
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[type_id]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["accent"], src_bg)
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
    """Single attempt to generate a valid input grid."""
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, accents, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    body_choices = [c for c in body_pool if c not in accents and c != src_bg and c != dst_bg and c != 0]
    unique_body_choices = list(dict.fromkeys(body_choices))
    if len(unique_body_choices) < n_types:
        return None
    if forced_body is not None and forced_body in unique_body_choices:
        remaining = [c for c in unique_body_choices if c != forced_body]
        if len(remaining) < n_types - 1:
            return None
        bodies = [forced_body] + random.sample(remaining, n_types - 1)
    else:
        bodies = random.sample(unique_body_choices, n_types)
    random.shuffle(bodies)

    sprite_types = [
        {
            "shape_rows": shape_rows,
            "accent_rc": accent_rc,
            "body": body,
            "accent": accent,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, accent_rc), body, accent in zip(sprite_specs, bodies, accents)
    ]

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
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
        patch = _render_sprite(
            sprite_types[type_id]["shape_rows"],
            sprite_types[type_id]["body"],
            sprite_types[type_id]["accent"],
            dst_bg,
        )
        dest_rect[1 + mr, 1 + mc] = int(patch[0, 0])

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
    raise RuntimeError("failed to generate a251c730 D8 input after %d attempts" % MAX_GEN_ATTEMPTS)
