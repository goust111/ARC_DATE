# Variant of a251c730
# Dimension: D1 — 匹配条件：锚点颜色匹配 (match_cond)
# Change: 精灵的锚点颜色（最少出现颜色）== 目标矩形中单像素标记的颜色，一对一匹配... → 边缘颜色匹配

# Design analysis
# variant:
#   concrete_input:
#   - 仍然有两块叠在壁纸上的矩形。源矩形保持原题结构：统一边框、内部放 1-2 类双色精灵，
#     每类精灵都有一个单像素锚点色，且该色在该精灵中恰好只出现一次。
#   - 目标矩形不再在内部放单像素 marker；其内部在输入中完全清空，只在边框上嵌入锚点色。
#   - 对每个精灵类型，左右边框的若干单格给出目标行，顶部和底部边框的若干单格给出目标列。
#     同色的行集合与列集合做笛卡尔积，就得到该精灵需要印刻的全部锚点位置。
#   behavior_trace:
#   - 原逻辑：目标框内部若有红色单点 (r,c)，就把红锚点精灵直接平移到该点。
#   - 新逻辑：若左右边框的红色单格在第 3、6 行，且上下边框的红色单格在第 8 列，
#     则红精灵要分别印刻到 (3,8) 和 (6,8)。
#   - 差异不在精灵本体，而在“目标位置”从单点直接给出，改成由边框行/列线索共同确定。
#   complexity_calibration:
#   - 保持 1-2 种精灵、总目标数 2-3，不引入额外变换。
#   - 为避免同色产生 2x2 以上的爆炸式组合，生成器约束每种颜色只在一个轴上重复：
#     若某色需要 3 个目标，则布局成 3 行 x 1 列或 1 行 x 3 列，而不是 2 行 x 2 列。
#   - 边框提示使用“镜像的单格边框段”：左右同色表示行，上下同色表示列。这样变化肉眼可见，
#     同时不会破坏“矩形边框大体统一”的整体观感。
#   hidden_assumptions_review:
#   - 原 solver 依赖“目标框边框完全统一”来找矩形；变体后目标框边框允许少量例外，因此必须放宽为：
#     四角同色，且边框绝大多数仍是基底色，非基底色只出现在非角边格。
#   - 原 solver 依赖“目标框内部有单像素组件”来识别 dst；变体后 dst 内部为空，所以改为识别
#     “边框上存在对称的同色行标/列标，内部无组件”的矩形。
#   - 原 solver 从目标内点直接取坐标；变体后必须分别读取 left/right 与 top/bottom 的颜色位置，
#     再求同色交点。
#   legality:
#   - 生成器先决定输出中的真实印刻位置，再反推边框行列标记；并保证所有精灵 bbox 完整落在目标内部。
#   - 同色重复目标在共享一条列或一条行时，沿重复轴至少相隔一个 sprite bbox，避免输出重叠。
#   - 不同颜色之间再用占用检测避免重叠。
#   observability:
#   - 输入的目标框内部彻底空白，只有边框上出现锚点色，和原题的“内部彩色单点”差异非常直接。
#   - 同色在两条竖边和两条横边上重复出现，足以让人从少量样例看出“取交点”的定位机制。
#   cleanliness_and_color:
#   - 不使用颜色 id 做可见规则分支；锚点之所以可识别，是因为它在精灵中是唯一稀有色。
#   - 角色颜色总数仍控制在 7-9，body 复用壁纸色，避免画面过杂。
#   new_main_assumptions:
#   - dst 的角仍保留基底边框色，且总边框提示格数不超过 10；generator 通过“总目标数 2-3、总类型数 1-2”
#     保证这个上界成立。
#   - 每个出现在边框提示中的颜色，必须同时拥有至少一条行提示和一条列提示；generator 强制满足。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location


MAX_BORDER_MARKS = 10


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via edge-coded anchor matches.

    Input structure:
    - Two bordered rectangles sit on the wallpaper: the source still contains multi-cell sprites.
    - The destination interior is empty; instead, anchor colors are embedded on the frame edges to mark rows and columns.

    Rule:
    - Recover each sprite template from the source by its singleton anchor color.
    - In the destination, matching colors on the left/right edges specify rows, and matching colors on the top/bottom edges specify columns.
    - Stamp the sprite so its anchor lands at every same-color row/column intersection, then output the cropped destination frame.

    Key insight: The anchor color still identifies the sprite, but now its placement is inferred from border-aligned row/column cues rather than interior markers.
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
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(Counter(vals)),
                    "mask": mask,
                    "patch": patch,
                }
            )
        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

    def border_nonbase_count(rect: tuple[int, int, int, int]) -> tuple[int, int] | None:
        r0, c0, r1, c1 = map(int, rect)
        base = int(g[int(r0), int(c0)])
        if (
            int(g[int(r0), int(c1)]) != int(base)
            or int(g[int(r1), int(c0)]) != int(base)
            or int(g[int(r1), int(c1)]) != int(base)
        ):
            return None
        top = g[int(r0), int(c0) : int(c1) + 1]
        bottom = g[int(r1), int(c0) : int(c1) + 1]
        left = g[int(r0) + 1 : int(r1), int(c0)]
        right = g[int(r0) + 1 : int(r1), int(c1)]
        nonbase = int((top != int(base)).sum() + (bottom != int(base)).sum() + (left != int(base)).sum() + (right != int(base)).sum())
        return int(base), int(nonbase)

    def edge_marks(rect: tuple[int, int, int, int]) -> tuple[int, dict[int, list[int]], dict[int, list[int]]] | None:
        r0, c0, r1, c1 = map(int, rect)
        base = int(g[int(r0), int(c0)])
        row_marks: dict[int, list[int]] = {}
        col_marks: dict[int, list[int]] = {}

        for rr in range(int(r0) + 1, int(r1)):
            left_col = int(g[int(rr), int(c0)])
            right_col = int(g[int(rr), int(c1)])
            if int(left_col) == int(base) and int(right_col) == int(base):
                continue
            if int(left_col) != int(right_col) or int(left_col) == int(base):
                return None
            row_marks.setdefault(int(left_col), []).append(int(rr - r0))

        for cc in range(int(c0) + 1, int(c1)):
            top_col = int(g[int(r0), int(cc)])
            bottom_col = int(g[int(r1), int(cc)])
            if int(top_col) == int(base) and int(bottom_col) == int(base):
                continue
            if int(top_col) != int(bottom_col) or int(top_col) == int(base):
                return None
            col_marks.setdefault(int(top_col), []).append(int(cc - c0))

        if not row_marks or not col_marks:
            return None
        if set(row_marks.keys()) != set(col_marks.keys()):
            return None
        return int(base), row_marks, col_marks

    rects: list[tuple[int, tuple[int, int, int, int]]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    border_info = border_nonbase_count((int(r0), int(c0), int(r1), int(c1)))
                    if border_info is None:
                        continue
                    _base, nonbase = border_info
                    if int(nonbase) > MAX_BORDER_MARKS:
                        continue
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(g[int(r0), int(c0)]):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1))))

    assert rects
    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:14]

    src = None
    dst = None
    for _area, rect in rects:
        border_info = border_nonbase_count(rect)
        if border_info is None:
            continue
        base, nonbase = border_info
        ibg, comps = rect_components(rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if src is None and int(nonbase) == 0 and max_sz > 1:
            src = (rect, base, ibg, comps)
            continue
        marks = edge_marks(rect)
        if dst is None and int(nonbase) > 0 and max_sz == 0 and marks is not None:
            _dst_base, row_marks, col_marks = marks
            dst = (rect, base, ibg, row_marks, col_marks)

    assert src is not None and dst is not None, "must find both source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), _dst_bc, dst_bg, row_marks, col_marks = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = dict(comp["colors"])
        if not cols:
            continue
        anchor_col, anchor_freq = min(cols.items(), key=lambda kv: int(kv[1]))
        if int(anchor_freq) != 1:
            continue
        patch = comp["patch"]
        mask = comp["mask"]
        anchors = np.argwhere((patch == int(anchor_col)) & mask)
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        prev = templates.get(int(anchor_col))
        if prev is None or int(mask.sum()) > int(prev[1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))

    out = g[int(dr) : int(dr1) + 1, int(dc) : int(dc1) + 1].copy()
    out[1:-1, 1:-1] = int(dst_bg)

    for color, rows in row_marks.items():
        if int(color) not in templates:
            continue
        cols = col_marks.get(int(color), [])
        patch, mask, (ar, ac) = templates[int(color)]
        for rr in rows:
            for cc in cols:
                paste(out, patch, int(rr - ar), int(cc - ac), mask=mask, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
TARGET_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
EDGE_TARGET_RETRIES = 80
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 300

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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int]] | None:
    wallpaper_n = 4 if n_types == 2 else random.choice(list(WALLPAPER_COLORS))
    wallpaper = random.sample(list(range(10)), wallpaper_n)
    role_pool = [c for c in range(1, 10) if c not in wallpaper]
    if len(role_pool) < 2 + n_types:
        return None
    random.shuffle(role_pool)
    src_border = role_pool.pop()
    dst_border = role_pool.pop()
    anchors = [role_pool.pop() for _ in range(n_types)]
    return wallpaper, src_border, dst_border, anchors


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if n == 0:
        return []
    if len(candidates) < n:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return sorted(picked)
    return None


def _plan_edge_targets(
    sprite_types: list[dict], target_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, dict[int, list[int]], dict[int, list[int]], list[tuple[int, int, int]]] | None:
    counts = Counter(target_type_ids)
    order = sorted(
        counts.keys(),
        key=lambda tid: (-int(counts[int(tid)]), -(int(sprite_types[int(tid)]["h"]) * int(sprite_types[int(tid)]["w"]))),
    )

    occupancy = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    row_marks: dict[int, list[int]] = {}
    col_marks: dict[int, list[int]] = {}
    target_points: list[tuple[int, int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for type_id in order:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        sh, sw = map(int, patch.shape)
        ar, ac = map(int, sprite["anchor_rc"])
        count = int(counts[int(type_id)])

        valid_rows = list(range(1 + int(ar), dst_h - 1 - (int(sh) - 1 - int(ar))))
        valid_cols = list(range(1 + int(ac), dst_w - 1 - (int(sw) - 1 - int(ac))))
        valid_rows = [r for r in valid_rows if int(r) not in used_rows]
        valid_cols = [c for c in valid_cols if int(c) not in used_cols]
        if not valid_rows or not valid_cols:
            return None

        orientations = ["rows", "cols"] if count > 1 else [random.choice(["rows", "cols"])]
        random.shuffle(orientations)
        placed = False

        for orient in orientations:
            row_n = count if orient == "rows" else 1
            col_n = 1 if orient == "rows" else count
            if len(valid_rows) < row_n or len(valid_cols) < col_n:
                continue

            for _ in range(EDGE_TARGET_RETRIES):
                rows = _pick_spaced(valid_rows[:], row_n, max(1, sh))
                cols = _pick_spaced(valid_cols[:], col_n, max(1, sw))
                if rows is None or cols is None:
                    break
                pts = [(int(r), int(c)) for r in rows for c in cols]
                if len(pts) != count:
                    continue

                trial = occupancy.copy()
                ok = True
                for tr, tc in pts:
                    top = int(tr - 1 - ar)
                    left = int(tc - 1 - ac)
                    try:
                        paste(trial, patch, int(top), int(left), transparent=dst_bg, overwrite=False)
                    except ValueError:
                        ok = False
                        break
                if not ok:
                    continue

                occupancy = trial
                row_marks[int(type_id)] = rows
                col_marks[int(type_id)] = cols
                used_rows.update(map(int, rows))
                used_cols.update(map(int, cols))
                for tr, tc in pts:
                    target_points.append((int(tr), int(tc), int(type_id)))
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    return occupancy, row_marks, col_marks, target_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
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


def _add_edge_marks(
    rect: np.ndarray, sprite_types: list[dict], row_marks: dict[int, list[int]], col_marks: dict[int, list[int]]
) -> None:
    for type_id, rows in row_marks.items():
        color = int(sprite_types[int(type_id)]["anchor"])
        for rr in rows:
            rect[int(rr), 0] = color
            rect[int(rr), -1] = color
    for type_id, cols in col_marks.items():
        color = int(sprite_types[int(type_id)]["anchor"])
        for cc in cols:
            rect[0, int(cc)] = color
            rect[-1, int(cc)] = color


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != src_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    body_choices = [c for c in wallpaper if c not in {src_bg, dst_bg, 0}]
    if len(body_choices) < n_types:
        return None
    random.shuffle(body_choices)
    bodies = body_choices[:n_types]

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    n_targets = random.randint(*TARGET_COUNT_RANGE)
    target_type_ids = [i for i, ct in enumerate(_choose_counts(n_targets, n_types)) for _ in range(ct)]
    random.shuffle(target_type_ids)

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

    planned = _plan_edge_targets(sprite_types, target_type_ids, dst_h, dst_w, dst_bg)
    if planned is None:
        return None
    out_inner, row_marks, col_marks, _target_points = planned

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    _add_edge_marks(output, sprite_types, row_marks, col_marks)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None
    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    _add_edge_marks(dest_rect, sprite_types, row_marks, col_marks)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, int(src_pos[0]), int(src_pos[1]))
    paste(grid, dest_rect, int(dst_pos[0]), int(dst_pos[1]))

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D1 variant after {MAX_GEN_ATTEMPTS} attempts")
