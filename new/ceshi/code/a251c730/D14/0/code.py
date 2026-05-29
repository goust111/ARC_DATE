# Variant of a251c730
# Dimension: D14 — 精灵类型多样性 (meta)
# Change: 1-2种不同的精灵类型，从11种预定义形状模板中选取，同类型精灵形状完全相同... → 每个精灵形状唯一

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


# Design analysis
# ===============
# design:
#   source_task: none
#   concretization:
#     - 仍然有两个带边框矩形: 源框里放多像素精灵, 目标框里只放单像素标记.
#     - 与原题不同, 源框中的每个精灵都使用不同的形状模板, 不再出现“同类型复制”.
#     - 锚点颜色不再与形状一一对应: 一个颜色可以服务于 2 个不同形状的精灵.
#     - 目标框中的同色单像素标记也会重复出现; 输出时要把源框里该颜色的全部精灵,
#       按锚点阅读顺序一一对齐到目标框中同色标记的阅读顺序上.
#   behavior_trace:
#     - 原逻辑: 若源框里两个精灵锚点同为红色, template[红] 只会保留其中一个模板;
#       目标框里两个红点都会被盖成同一个形状.
#     - 变体逻辑: 源框里两个红锚点组件会形成一个长度为 2 的序列 [shape_A, shape_B];
#       目标框里两个红点排序后分别接收 shape_A 与 shape_B.
#     - 差异点正好落在 solver 的“按颜色建模板表”环节; 其余矩形检测与裁剪规则可保持.
#   complexity_calibration:
#     - 为了让 D14 肉眼可见, 生成器强制 n_sprites > n_anchor_colors, 因而必有共享锚点颜色.
#     - 同时所有精灵模板从 11 个模板中无放回抽取, 保证形状绝不重复.
#     - 目标标记数设为源精灵数, 避免有未使用精灵, 也避免旧逻辑偶然与新逻辑输出一致.
#   implicit_assumption_audit:
#     - 原 main 中 `templates[anchor_color] = ...` 依赖“每种颜色只对应一个形状”.
#       该假设在变体中失效, 必须改成 `templates[color] -> list[template]`.
#     - 原 main 用“更大模板覆盖更小模板”去解决同色重复, 这在变体中会错误抹掉其它形状.
#     - 原 main 遍历目标标记时直接按颜色取模板; 现在必须先按颜色分组并排序后再配对.
#     - 锚点提取仍依赖“每个精灵恰有一个最稀有颜色像素”; 生成器继续保证 anchor 单像素.
#   legality:
#     - 源框放置继续使用 `random_free_location`, 保证精灵彼此不重叠且离边框至少 1 格.
#     - 目标输出的真实盖印在一个空白 inner canvas 上完成, 以此反推单像素标记位置,
#       所以不会出现越界或印章重叠.
#     - 对同色标记额外要求锚点阅读顺序递增, 使 solver 的同色序列配对有确定答案.
#   observability:
#     - 共享颜色 + 不同形状会直接体现在同色输出印章不相同这一点上, 与原题视觉差异明显.
#     - 配对顺序只使用像素可见的阅读顺序, 不使用颜色 id、面积排行等不可见线索.
#   cleanliness:
#     - 输入目标框仍只有零散单像素, 输出才出现完整印章, 层次与原题一致, 不会混成一团.
#     - 标记之间保留至少曼哈顿距离 2, 防止单像素在输入中连成难辨的块.
#   color_inference:
#     - 颜色本身只负责把源精灵和目标标记分成若干组; 组内的对应关系由空间顺序决定.
#     - 锚点颜色种类控制在 2-3 个, 不让颜色映射过多而失去可推断性.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame using color-grouped marker sequences.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: a source box with multi-cell sprites and a destination box with singleton markers.
    - Every sprite still has a one-pixel anchor color, but several differently shaped sprites may share that anchor color.

    Rule:
    - Group source sprites by anchor color, then sort each color group by anchor reading order inside the source box.
    - Group destination singleton markers by color, sort those markers in reading order, and align the kth source sprite of a color to the kth marker of that color.
    - Output is the destination rectangle alone, cleared to its background before stamping the matched sprites.

    Key insight: Color no longer identifies a single template; it identifies an ordered family of unique shapes that must be replayed in sequence.
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
                    ibg = interior_bg((r0, c0, r1, c1))
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
        ibg2, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if max_sz > 1 and src is None:
            src = (rect, bc, ibg2, comps)
        if max_sz == 1 and len(comps) > 0 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source (multi-cell) and destination (singleton) rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    templates: dict[int, list[tuple[tuple[int, int], np.ndarray, np.ndarray, tuple[int, int]]]] = defaultdict(list)
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if not cols:
            continue
        min_count = min(cols.values())
        anchor_candidates = [color for color, count in cols.items() if int(count) == int(min_count)]
        if len(anchor_candidates) != 1:
            continue
        anchor_col = int(anchor_candidates[0])
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == anchor_col) & mask)
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        rr0, cc0, _rr1, _cc1 = comp["bbox"]
        anchor_in_src = (int(rr0 + ar), int(cc0 + ac))
        templates[anchor_col].append((anchor_in_src, patch.copy(), mask.copy(), (ar, ac)))

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
        for (_src_anchor, patch, mask, (ar, ac)), (rr, cc) in zip(templates.get(int(color), []), markers):
            top = int(rr + 1 - ar)
            left = int(cc + 1 - ac)
            paste(out, patch, top, left, mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator ---------------

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

# === CFG: Sprite diversity ===
UNIQUE_SPRITE_COUNT_RANGE = (3, 4)
ANCHOR_COLOR_COUNT_CHOICES = (2, 3)
MARKER_PLACE_ATTEMPTS = 30
SOURCE_SPRITE_PADDING = 1
MARKER_MIN_MANHATTAN = 2

# === CFG: Generation retries ===
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


def _assign_palette(
    n_anchor_colors: int,
) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    palette_options: list[tuple[int, int]] = []
    for wallpaper_n in WALLPAPER_COLORS:
        for used_total_target in TOTAL_COLORS_TARGET:
            extras_needed = used_total_target - (wallpaper_n + 1 + n_anchor_colors)
            if 0 <= extras_needed <= 2:
                palette_options.append((wallpaper_n, extras_needed))
    if not palette_options:
        return None

    wallpaper_n, extras_needed = random.choice(palette_options)
    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_anchor_colors)]

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

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _place_sources(
    sprite_instances: list[dict], src_h: int, src_w: int, src_bg: int
) -> tuple[np.ndarray, list[dict]] | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    placed_entries: list[dict] = []
    for sprite in sprite_instances:
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
        ar, ac = sprite["anchor_rc"]
        placed_entries.append(
            {
                **sprite,
                "patch": patch,
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
    occ = canvas != int(bg)
    mask = patch != int(bg)
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
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
                candidates = _valid_output_locations(
                    inner,
                    entry["patch"],
                    entry["anchor_rc"],
                    dst_bg,
                    [(r, c) for r, c, _ in marker_points],
                    last_anchor,
                )
                if not candidates:
                    success = False
                    break
                r0, c0, mr, mc = random.choice(candidates)
                paste(inner, entry["patch"], r0, c0, transparent=dst_bg)
                marker_points.append((mr, mc, int(color)))
                last_anchor = (mr, mc)
            if not success:
                break
        if success:
            return inner, marker_points
    return None


def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.randint(*UNIQUE_SPRITE_COUNT_RANGE)
    n_anchor_colors = random.randint(ANCHOR_COLOR_COUNT_CHOICES[0], min(ANCHOR_COLOR_COUNT_CHOICES[1], n_sprites - 1))

    palette_result = _assign_palette(n_anchor_colors)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchor_palette, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_sprites)
    anchor_counts = _choose_counts(n_sprites, n_anchor_colors)
    sprite_anchor_colors = [anchor_palette[i] for i, count in enumerate(anchor_counts) for _ in range(count)]
    random.shuffle(sprite_anchor_colors)

    bodies: list[int] = []
    for i in range(n_sprites):
        choices = [c for c in body_pool if c not in sprite_anchor_colors and c != src_bg and c != dst_bg]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_instances = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": body,
            "anchor": anchor,
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, sprite_anchor_colors)
    ]

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
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
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a251c730 D14 input after %d attempts" % MAX_GEN_ATTEMPTS)
