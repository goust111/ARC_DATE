# Combo variant of a251c730 (index 3)
# combo_spec: D1/3+D8/0
# Name: 边缘色段匹配+几何中心锚点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# ---------------
# variant:
#   concrete_input:
#   - 仍然是在 30x30 条纹壁纸上叠两块矩形。源矩形保持“统一边框 + 若干双色精灵”的原题骨架。
#   - 精灵仍有一个单像素稀有色 A，但融合后它只保留为可见装饰；真正的工作锚点改成精灵 bbox 的几何中心格。
#   - 目标矩形的内部在输入中完全清空，不再出现单像素 marker。取而代之的是：四条边框上嵌入与“中心格颜色”
#     相同的短色段。左右边框的镜像色段给出目标行，顶部/底部的镜像色段给出目标列。
#   - 每个边框段都固定为奇数长度 3，因此其中心格可直接读成一个精确的行/列坐标。
#   behavior_trace:
#   - 原题：在目标内部看到某个红色单点，就找“稀有红锚点”的源精灵，并把那颗稀有红像素对齐到该点。
#   - D1 单变体：红色不在内部，而是在左右边框给行、上下边框给列；交点仍对齐“旧稀有锚点”。
#   - 本融合：边框仍负责给位置，但颜色含义和贴合偏移一起换掉。若某类精灵的 bbox 中心是蓝色，
#     左右边框蓝段中心在第 4 行、上下边框蓝段中心在第 7 列，就把该精灵平移到其“几何中心格”落在 (4,7)。
#   complexity_calibration:
#   - 仍只保留 1-2 类精灵、总目标数 2-3，避免把“边框行列匹配”和“中心锚点”淹没在过多自由度里。
#   - 生成器强制每种精灵的中心颜色唯一，且中心位置不等于稀有像素位置；否则会退化成原题或造成匹配歧义。
#   - 同色目标仍限制成 1xk 或 kx1，而不是 2x2 笛卡尔爆炸；这样少量样例就能看出“同色段中心求交点”。
#   hidden_assumptions_review:
#   - 原 solver 依赖目标框“边框完全纯色”；这里必须放宽成“角仍是基底色，非角边框允许少量彩色连续段”。
#   - 原 solver 依赖目标框内部存在单像素组件来识别 dst；这里 dst 内部必须为空，改靠边框段模式识别。
#   - 原模板提取依赖“最少出现颜色唯一且就是锚点”；这里不再使用该前提，而是假设 bbox 几何中心格被占据。
#   - 多项改动叠加后新增的关键假设是：每种边框提示色同时拥有左右一对行段与上下一对列段，且段长为奇数。
#     generator 会显式保证这些条件。
#   legality:
#   - 输出优先：先在 dst 内部规划真实的 stamp 位置，再把这些中心点反推成边框段中心。
#   - 目标中心位置同时满足两类约束：完整 stamp 必须落在 dst 内部；边框段本身也必须完整落在非角边框上。
#   - 多个 stamp 通过透明粘贴占用检测避免重叠；多个边框段通过最小间距约束避免重叠与粘连。
#   observability:
#   - 视觉上和原题差异足够明显：目标输入框内部完全空白，而四边出现成对的彩色短段。
#   - 精灵的工作颜色通常是 body 色而非稀有色，且同一颜色会同时出现在源精灵中心与目标边框段上，
#     人类更容易从 2-3 组样例中看出“看中心色、取段中心、在交点盖章”。
#   cleanliness_and_inference:
#   - 固定 3 格短段比随机长段更整洁，且“取中点”几乎是肉眼直接可读的规则。
#   - 稀有色仍保留在精灵里，让人能看见它已经不再承担定位功能，从而突出 D8 的变化。

GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

SPRITE_TYPE_CHOICES = (1, 2)
SPRITE_TYPE_PICK_WEIGHTS = (1, 2, 2)
TARGET_COUNT_RANGE = (2, 3)
SOURCE_SPRITE_RANGE = (2, 4)
EDGE_TARGET_RETRIES = 100
SOURCE_SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 300

EDGE_SEG_LEN = 3
EDGE_SEG_HALF = EDGE_SEG_LEN // 2
EDGE_SEG_CLEARANCE = EDGE_SEG_LEN + 1
MAX_BORDER_MARKS = 36

# Every chosen shape has:
# - one singleton 'A'
# - an occupied bbox center
# - center position different from the singleton location
SPRITE_SHAPES = [
    ("AB.", "BBB"),
    ("ABB", "BBB"),
    (".A.", "BBB", ".B."),
    ("AB.", "BBB", ".BB"),
    ("ABB", "BBB", "..B"),
    (".AB", "BBB", "BB."),
    ("B.B", "BBB", ".A."),
    ("BBB", "BAB"),
    ("AB.", "BBB", ".B."),
]


def _center_rc_from_shape(h: int, w: int) -> tuple[int, int]:
    return (int((h - 1) // 2), int((w - 1) // 2))


def _center_rc_from_patch(patch: np.ndarray) -> tuple[int, int]:
    h, w = map(int, patch.shape)
    return _center_rc_from_shape(h, w)


def _shape_singleton_rc(shape_rows: tuple[str, ...], ch: str = "A") -> tuple[int, int]:
    hits = [(r, c) for r, row in enumerate(shape_rows) for c, val in enumerate(row) if val == ch]
    if len(hits) != 1:
        raise ValueError(f"shape must contain exactly one {ch!r}")
    return hits[0]


def _shape_center_char(shape_rows: tuple[str, ...]) -> str:
    cr, cc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
    return shape_rows[cr][cc]


def _render_sprite(shape_rows: tuple[str, ...], body: int, rare: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": rare}
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


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
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
    rare_colors = [remaining.pop() for _ in range(n_types)]

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

    return wallpaper, src_border, dst_border, rare_colors, body_pool, forced_body


def _interior_bg(g: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
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
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
            }
        )
    comps.sort(key=lambda d: -int(d["size"]))
    return int(ibg), comps


def _border_nonbase_count(g: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int] | None:
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


def _parse_nonbase_runs(values: np.ndarray, base: int, offset: int = 0) -> dict[int, list[tuple[int, int]]]:
    runs: dict[int, list[tuple[int, int]]] = {}
    idx = 0
    n = int(values.shape[0])
    while idx < n:
        color = int(values[int(idx)])
        if int(color) == int(base):
            idx += 1
            continue
        j = idx
        while j + 1 < n and int(values[int(j) + 1]) == int(color):
            j += 1
        length = int(j - idx + 1)
        if length < EDGE_SEG_LEN or length % 2 == 0:
            return {}
        center = int(offset + idx + length // 2)
        runs.setdefault(int(color), []).append((int(center), int(length)))
        idx = j + 1
    return runs


def _edge_segment_marks(
    g: np.ndarray, rect: tuple[int, int, int, int]
) -> tuple[int, dict[int, list[int]], dict[int, list[int]]] | None:
    r0, c0, r1, c1 = map(int, rect)
    base = int(g[int(r0), int(c0)])

    top_runs = _parse_nonbase_runs(g[int(r0), int(c0) + 1 : int(c1)], int(base), offset=1)
    bottom_runs = _parse_nonbase_runs(g[int(r1), int(c0) + 1 : int(c1)], int(base), offset=1)
    left_runs = _parse_nonbase_runs(g[int(r0) + 1 : int(r1), int(c0)], int(base), offset=1)
    right_runs = _parse_nonbase_runs(g[int(r0) + 1 : int(r1), int(c1)], int(base), offset=1)
    if not top_runs or not bottom_runs or not left_runs or not right_runs:
        return None
    if set(top_runs) != set(bottom_runs) or set(left_runs) != set(right_runs):
        return None
    if set(top_runs) != set(left_runs):
        return None

    row_marks: dict[int, list[int]] = {}
    col_marks: dict[int, list[int]] = {}
    for color in sorted(top_runs):
        top_list = sorted(top_runs[int(color)])
        bottom_list = sorted(bottom_runs[int(color)])
        left_list = sorted(left_runs[int(color)])
        right_list = sorted(right_runs[int(color)])
        if top_list != bottom_list or left_list != right_list:
            return None
        row_marks[int(color)] = [int(center) for center, _length in left_list]
        col_marks[int(color)] = [int(center) for center, _length in top_list]
        if not row_marks[int(color)] or not col_marks[int(color)]:
            return None
    return int(base), row_marks, col_marks


def _find_src_dst(g: np.ndarray) -> tuple[tuple, tuple]:
    H, W = g.shape
    rects: list[tuple[int, tuple[int, int, int, int]]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    border_info = _border_nonbase_count(g, (int(r0), int(c0), int(r1), int(c1)))
                    if border_info is None:
                        continue
                    _base, nonbase = border_info
                    if int(nonbase) > MAX_BORDER_MARKS:
                        continue
                    ibg = _interior_bg(g, (int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(g[int(r0), int(c0)]):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1))))

    assert rects, "must detect candidate rectangles"
    rects.sort(reverse=True, key=lambda x: int(x[0]))

    src = None
    dst = None
    for _area, rect in rects:
        border_info = _border_nonbase_count(g, rect)
        if border_info is None:
            continue
        base, nonbase = border_info
        ibg, comps = _rect_components(g, rect)
        max_sz = max([int(c["size"]) for c in comps], default=0)
        if src is None and int(nonbase) == 0 and max_sz > 1:
            src = (rect, base, ibg, comps)
            continue
        marks = _edge_segment_marks(g, rect)
        if dst is None and int(nonbase) > 0 and max_sz == 0 and marks is not None:
            _dst_base, row_marks, col_marks = marks
            dst = (rect, base, ibg, row_marks, col_marks)

    assert src is not None and dst is not None, "must find source and destination rectangles"
    return src, dst


def _build_center_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        patch = comp["patch"]
        mask = comp["mask"]
        ar, ac = _center_rc_from_patch(patch)
        if not bool(mask[int(ar), int(ac)]):
            continue
        color = int(patch[int(ar), int(ac)])
        if int(color) == 0:
            continue
        prev = templates.get(int(color))
        if prev is None or int(mask.sum()) > int(prev[1].sum()):
            templates[int(color)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))
    return templates


def _build_center_and_rare_templates(
    src_comps: list[dict],
) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]] = {}
    for comp in src_comps:
        patch = comp["patch"]
        mask = comp["mask"]
        center_rc = _center_rc_from_patch(patch)
        cr, cc = center_rc
        if not bool(mask[int(cr), int(cc)]):
            continue
        center_color = int(patch[int(cr), int(cc)])
        cols = dict(comp["colors"])
        rare_color, rare_freq = min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
        if int(rare_freq) != 1:
            continue
        rare_hits = np.argwhere((patch == int(rare_color)) & mask)
        if rare_hits.size == 0:
            continue
        rr, rc = map(int, rare_hits[0])
        prev = templates.get(int(center_color))
        if prev is None or int(mask.sum()) > int(prev[1].sum()):
            templates[int(center_color)] = (patch.copy(), mask.copy(), (int(cr), int(cc)), (int(rr), int(rc)))
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the destination frame by reading border segments and anchoring at sprite centers.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: the source holds multi-cell two-color sprites, while the destination interior is blank.
    - The destination border is mostly uniform, except for mirrored colored edge segments whose colors match the source sprites' geometric-center pixels.

    Rule:
    - Recover each source sprite template using the color at the geometric center of its bounding box.
    - In the destination, matching left/right edge segments specify target rows and matching top/bottom segments specify target columns; use each segment midpoint as the exact row or column.
    - Stamp the sprite so its geometric-center cell lands on every same-color row/column intersection, then return the cropped destination rectangle.

    Key insight: the border now carries the placement cues, and those cues refer to the sprite's bbox-center color and bbox-center offset rather than the old singleton accent pixel.
    """
    g = np.array(grid, dtype=int)
    src, dst = _find_src_dst(g)

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), _dst_bc, dst_bg, row_marks, col_marks = dst

    templates = _build_center_templates(src_comps)

    out = g[int(dr) : int(dr1) + 1, int(dc) : int(dc1) + 1].copy()
    out[1:-1, 1:-1] = int(dst_bg)

    for color, rows in row_marks.items():
        if int(color) not in templates:
            continue
        patch, mask, (ar, ac) = templates[int(color)]
        cols = col_marks.get(int(color), [])
        for rr in rows:
            for cc in cols:
                paste(out, patch, int(rr - ar), int(cc - ac), mask=mask, overwrite=True, clip=True)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    if n == 0:
        return []
    if len(candidates) < n:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for value in candidates:
        if all(abs(int(value) - int(prev)) >= int(min_spacing) for prev in picked):
            picked.append(int(value))
            if len(picked) == n:
                return sorted(picked)
    return None


def _plan_edge_targets(
    sprite_types: list[dict], target_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, dict[int, list[int]], dict[int, list[int]]] | None:
    counts = Counter(target_type_ids)
    order = sorted(
        counts.keys(),
        key=lambda tid: (-int(counts[int(tid)]), -(int(sprite_types[int(tid)]["h"]) * int(sprite_types[int(tid)]["w"]))),
    )

    occupancy = np.full((dst_h - 2, dst_w - 2), dst_bg, dtype=int)
    row_marks: dict[int, list[int]] = {}
    col_marks: dict[int, list[int]] = {}
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for type_id in order:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["rare"], dst_bg)
        sh, sw = map(int, patch.shape)
        ar, ac = map(int, sprite["center_rc"])
        count = int(counts[int(type_id)])

        row_lo = max(1 + int(ar), 1 + EDGE_SEG_HALF)
        row_hi = min(int(dst_h) - 2 - (int(sh) - 1 - int(ar)), int(dst_h) - 2 - EDGE_SEG_HALF)
        col_lo = max(1 + int(ac), 1 + EDGE_SEG_HALF)
        col_hi = min(int(dst_w) - 2 - (int(sw) - 1 - int(ac)), int(dst_w) - 2 - EDGE_SEG_HALF)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        valid_rows = [
            r
            for r in range(int(row_lo), int(row_hi) + 1)
            if all(abs(int(r) - int(prev)) >= EDGE_SEG_CLEARANCE for prev in used_rows)
        ]
        valid_cols = [
            c
            for c in range(int(col_lo), int(col_hi) + 1)
            if all(abs(int(c) - int(prev)) >= EDGE_SEG_CLEARANCE for prev in used_cols)
        ]
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

            row_spacing = max(EDGE_SEG_CLEARANCE, sh) if row_n > 1 else EDGE_SEG_CLEARANCE
            col_spacing = max(EDGE_SEG_CLEARANCE, sw) if col_n > 1 else EDGE_SEG_CLEARANCE

            for _ in range(EDGE_TARGET_RETRIES):
                rows = _pick_spaced(valid_rows[:], row_n, row_spacing)
                cols = _pick_spaced(valid_cols[:], col_n, col_spacing)
                if rows is None or cols is None:
                    break

                pts = [(int(r), int(c)) for r in rows for c in cols]
                if len(pts) != count:
                    continue

                trial = occupancy.copy()
                ok = True
                for rr, cc in pts:
                    top = int(rr - 1 - ar)
                    left = int(cc - 1 - ac)
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
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    return occupancy, row_marks, col_marks


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["rare"], src_bg)
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


def _paint_edge_segment(line: np.ndarray, center: int, color: int) -> None:
    start = int(center - EDGE_SEG_HALF)
    stop = int(center + EDGE_SEG_HALF + 1)
    line[int(start) : int(stop)] = int(color)


def _add_edge_segments(
    rect: np.ndarray, sprite_types: list[dict], row_marks: dict[int, list[int]], col_marks: dict[int, list[int]]
) -> None:
    for type_id, rows in row_marks.items():
        color = int(sprite_types[int(type_id)]["center_color"])
        for rr in rows:
            _paint_edge_segment(rect[:, 0], int(rr), int(color))
            _paint_edge_segment(rect[:, -1], int(rr), int(color))
    for type_id, cols in col_marks.items():
        color = int(sprite_types[int(type_id)]["center_color"])
        for cc in cols:
            _paint_edge_segment(rect[0, :], int(cc), int(color))
            _paint_edge_segment(rect[-1, :], int(cc), int(color))


def _gen_attempt() -> np.ndarray | None:
    # Two-type scenes have tighter packing/color constraints and otherwise appear less often
    # among successful samples, so bias the draw slightly to keep branch coverage healthier.
    n_types = random.choice(list(SPRITE_TYPE_PICK_WEIGHTS))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, rare_colors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    body_choices = [c for c in body_pool if c not in rare_colors and c not in {src_bg, dst_bg, dst_border}]
    body_choices = list(dict.fromkeys(body_choices))
    if len(body_choices) < n_types:
        return None
    if forced_body is not None and forced_body in body_choices:
        if n_types == 1:
            bodies = [int(forced_body)]
        else:
            others = [c for c in body_choices if c != forced_body]
            if len(others) < n_types - 1:
                return None
            bodies = [int(forced_body)] + random.sample(others, n_types - 1)
            random.shuffle(bodies)
    else:
        bodies = random.sample(body_choices, n_types)

    sprite_types: list[dict] = []
    for shape_rows, body, rare in zip(sprite_specs, bodies, rare_colors):
        center_rc = _center_rc_from_shape(len(shape_rows), len(shape_rows[0]))
        rare_rc = _shape_singleton_rc(shape_rows, "A")
        center_char = _shape_center_char(shape_rows)
        if center_char != "B":
            return None
        if center_rc == rare_rc:
            return None
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "center_rc": center_rc,
                "rare_rc": rare_rc,
                "body": int(body),
                "rare": int(rare),
                "center_color": int(body),
                "h": len(shape_rows),
                "w": len(shape_rows[0]),
            }
        )

    if len({int(sprite["center_color"]) for sprite in sprite_types}) != n_types:
        return None

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
    out_inner, row_marks, col_marks = planned

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    _add_edge_segments(output, sprite_types, row_marks, col_marks)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None
    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    _add_edge_segments(dest_rect, sprite_types, row_marks, col_marks)

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
    raise RuntimeError(f"failed to generate a251c730 combo variant after {MAX_GEN_ATTEMPTS} attempts")
