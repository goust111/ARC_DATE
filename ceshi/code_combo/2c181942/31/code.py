# Combo variant of 2c181942 (index 31)
# combo_spec: D1/0+D4/0+D7/0
# Name: 形状匹配+多色精灵+锚点消歧

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
# ---------------------------------------------------------------------------
#
# spec_variant:
#   overview:
#     - 保留原题的“中心空心方块 + 若干散落精灵 + output-first 贴回”骨架。
#     - 同时叠加三个改动：
#         1) D1/0: 不再按颜色配边，而是看 hub 边缘凹口与 sprite 轮廓是否互补。
#         2) D4/0: sprite 不是单色，而是统一外壳色 + 内层装饰色 + 1 个锚点色。
#         3) D7/0: 不再依赖镜像对称筛旋转，而是让锚点色像素决定正确朝向。
#   concrete_geometry:
#     - hub 固定为 6x6 hollow square，外侧每边再长出 2-cell 厚的 dock。
#     - dock 最外沿用长度 4 的凹口模式编码；真实接触带也是长度 4。
#     - 为了让锚点确实参与推理，只做两类模式：
#         horizontal = 1001，供 top/bot 共用
#         vertical   = 0110，供 left/right 共用
#       这样仅靠轮廓只能先分出“横向精灵”或“纵向精灵”，仍无法决定是上还是下、左还是右。
#   behavior_trace:
#     original:
#       - solver 先找 hub，然后按四边颜色分别抓同色单色 sprite。
#       - 旋转时靠“接触轴镜像对称”排除错误朝向。
#     fused:
#       - solver 先找同色 hub core，并把与 core 8 连通的 dock 一起当作 hub。
#       - 读取四边凹口模式；由于 top/bot 共用一类、left/right 共用一类，轮廓只决定轴向。
#       - 对 hub 外每个多色组件尝试 4 个旋转：
#           * 组件占据轮廓必须与某边的 support+notch 结构匹配；
#           * 且唯一的锚点色像素必须落在该边规定的角落：
#               top -> 左上，bot -> 右下，left -> 左下，right -> 右上。
#       - 匹配成功后整只组件平移贴附，内层装饰色和锚点色都保留到输出。
#     tiny_example:
#       - 一个横向 sprite 的外壳在 shape-only 检查下同时能贴 top 与 bot；
#         若锚点在左上角，它去 top；若旋转 180 度后锚点到右下角，它去 bot。
#       - 所以“轮廓负责配轴向，锚点负责配方向”，两步缺一不可。
#   complexity_calibration:
#     - 如果四边都用不同凹口，轮廓本身已唯一决定边，锚点会退化成装饰；这里避免这种退化。
#     - 如果四边都用同一凹口，形状信息又太弱；这里保留横向/纵向两类，让人能从少量样例看出
#       “先看轮廓属于哪一组，再看锚点决定具体哪边”。
#     - sprite 模板只给少量手工档位，保证视觉差异存在，但自由度仍可控，不会乱。
#   implicit_assumption_audit:
#     - 原 `m = (g == col) & (~hub_mask)` 假设“颜色唯一对应一边”，在融合后完全失效。
#     - 原 `spr_arr[~spr_mask] = bg` 基于“用一种颜色就能裁出整只 sprite”，现在会丢失内层色与锚点色。
#     - 原旋转判定里的 `flipud/fliplr` 对称性约束必须删除，因为 D7 明确取消该前提。
#     - 新 solver 的假设改为：
#         * hub core 仍是中心附近的 6x6 hollow square，四边同色；
#         * hub 外每个真实 sprite 是单个 8 连通组件；
#         * 每个真实 sprite 恰有 1 个锚点色像素；
#         * shape-only 匹配会给出一对对向候选边，anchor-aware 匹配会收敛成唯一边。
#     - generator 对应保证这些假设全部成立，并显式拒绝退化样本。
#   placement_legality:
#     - 仍采用 output-first：先在输出中贴好 hub 与定向 sprite，再旋转散开回输入。
#     - hub 位置预留 margin >= dock 厚度 + 最大 sprite 尺寸 + 2，保证贴附与散放都不越界。
#     - 随机散放使用 `random_free_location(..., padding=2)`，避免对象接触并错误并成一块。
#   visual_observability:
#     - 人类在样例里会稳定看到三件事：
#         * hub 与各 sprite 外壳同色，所以不能靠颜色配边；
#         * top/bot 边长得像一组、left/right 长得像一组，说明轮廓先分轴向；
#         * 最终贴附后锚点总落在该边的固定角，说明方向要靠锚点消歧。
#     - 内层装饰色在不同 sprite 上可重复，输出仍保留，能支撑“整只对象被搬运”。
#   cleanliness:
#     - 统一外壳色 + 固定锚点色 + 少量装饰色，颜色角色清晰。
#     - hub 的额外结构只来自 2-cell dock；sprite 只额外增加 1 个锚点像素，画面仍整洁。


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
BG_COLOR = 8
ANCHOR_COLOR = 2
VISIBLE_COLORS = (1, 3, 4, 5, 6, 7, 9)
SIDES = ("top", "bot", "left", "right")
OPPOSITE = {"top": "bot", "bot": "top", "left": "right", "right": "left"}
PATTERNS = {
    "top": (1, 0, 0, 1),
    "bot": (1, 0, 0, 1),
    "left": (0, 1, 1, 0),
    "right": (0, 1, 1, 0),
}
ANCHOR_RULE = {
    "top": "ul",
    "bot": "ur",
    "left": "ll",
    "right": "lr",
}


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


HORIZONTAL_TEMPLATES: tuple[tuple[np.ndarray, tuple[tuple[int, int], ...]], ...] = (
    (
        _rows_to_mask(
            [
                "1100",
                "1111",
                "1111",
                "1001",
            ]
        ),
        ((1, 1), (1, 2)),
    ),
    (
        _rows_to_mask(
            [
                "1110",
                "1111",
                "1111",
                "1001",
            ]
        ),
        ((1, 1), (1, 2)),
    ),
)

VERTICAL_TEMPLATES: tuple[tuple[np.ndarray, tuple[tuple[int, int], ...]], ...] = (
    (
        _rows_to_mask(
            [
                "1010",
                "0111",
                "0011",
                "1110",
            ]
        ),
        ((1, 1), (3, 1)),
    ),
    (
        _rows_to_mask(
            [
                "1010",
                "0111",
                "0111",
                "1110",
            ]
        ),
        ((1, 1), (2, 1)),
    ),
)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _find_core_hub(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    H, W = g.shape
    best = None
    n = HUB_N
    for r in range(2, H - n - 2):
        for c in range(2, W - n - 2):
            if (
                int(g[r, c]) != bg
                or int(g[r, c + n - 1]) != bg
                or int(g[r + n - 1, c]) != bg
                or int(g[r + n - 1, c + n - 1]) != bg
            ):
                continue

            inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
            if not bool((inner == bg).all()):
                continue

            top = g[r, c + 1 : c + n - 1]
            bot = g[r + n - 1, c + 1 : c + n - 1]
            left = g[r + 1 : r + n - 1, c]
            right = g[r + 1 : r + n - 1, c + n - 1]
            if bool((top == bg).any()) or bool((bot == bg).any()) or bool((left == bg).any()) or bool((right == bg).any()):
                continue

            colors = set(map(int, np.concatenate([top, bot, left, right])))
            if len(colors) != 1:
                continue
            color = int(next(iter(colors)))
            if color == bg or color == ANCHOR_COLOR:
                continue

            cr = r + (n - 1) / 2
            cc = c + (n - 1) / 2
            score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
            cand = (float(score), int(r), int(c), int(color))
            if best is None or cand[0] < best[0]:
                best = cand

    assert best is not None
    _, r, c, color = best
    return r, c, HUB_N, color


def _side_span(r: int, c: int, side: str) -> tuple[int, int]:
    if side in ("top", "bot"):
        return c + 1, c + HUB_N - 2
    return r + 1, r + HUB_N - 2


def _dock_outer_coord(r: int, c: int, side: str) -> int:
    if side == "top":
        return r - 2
    if side == "bot":
        return r + HUB_N + 1
    if side == "left":
        return c - 2
    return c + HUB_N + 1


def _read_hub_patterns(g: np.ndarray, r: int, c: int, bg: int) -> dict[str, tuple[int, ...]]:
    patterns: dict[str, tuple[int, ...]] = {}
    for side in SIDES:
        start, end = _side_span(r, c, side)
        outer = _dock_outer_coord(r, c, side)
        vals = []
        if side in ("top", "bot"):
            for cc in range(start, end + 1):
                vals.append(1 if int(g[outer, cc]) == bg else 0)
        else:
            for rr in range(start, end + 1):
                vals.append(1 if int(g[rr, outer]) == bg else 0)
        patterns[side] = tuple(vals)
    return patterns


def _contact_band(mask: np.ndarray, side: str, pattern: tuple[int, ...]) -> int | None:
    h, w = mask.shape
    L = len(pattern)
    hits: list[int] = []

    if side == "top":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[h - 2, :j].any()) or bool(mask[h - 2, j + L :].any()):
                continue
            if bool(mask[h - 1, :j].any()) or bool(mask[h - 1, j + L :].any()):
                continue
            if not bool(mask[h - 2, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[h - 1, j : j + L]) != pattern:
                continue
            hits.append(j)
    elif side == "bot":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[1, :j].any()) or bool(mask[1, j + L :].any()):
                continue
            if bool(mask[0, :j].any()) or bool(mask[0, j + L :].any()):
                continue
            if not bool(mask[1, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[0, j : j + L]) != pattern:
                continue
            hits.append(j)
    elif side == "left":
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, w - 2].any()) or bool(mask[j + L :, w - 2].any()):
                continue
            if bool(mask[:j, w - 1].any()) or bool(mask[j + L :, w - 1].any()):
                continue
            if not bool(mask[j : j + L, w - 2].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, w - 1]) != pattern:
                continue
            hits.append(j)
    else:
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, 1].any()) or bool(mask[j + L :, 1].any()):
                continue
            if bool(mask[:j, 0].any()) or bool(mask[j + L :, 0].any()):
                continue
            if not bool(mask[j : j + L, 1].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, 0]) != pattern:
                continue
            hits.append(j)

    if len(hits) != 1:
        return None
    return hits[0]


def _attachment_shift(mask: np.ndarray, side: str, pattern: tuple[int, ...], hub_r: int, hub_c: int) -> tuple[int, int] | None:
    pos = _contact_band(mask, side, pattern)
    if pos is None:
        return None
    span_start, _span_end = _side_span(hub_r, hub_c, side)
    outer = _dock_outer_coord(hub_r, hub_c, side)
    h, w = mask.shape

    if side == "top":
        return outer - (h - 1), span_start - pos
    if side == "bot":
        return outer, span_start - pos
    if side == "left":
        return span_start - pos, outer - (w - 1)
    return span_start - pos, outer


def _placeable(out: np.ndarray, arr: np.ndarray, mask: np.ndarray, dr: int, dc: int, bg: int) -> bool:
    H, W = out.shape
    for rr, cc in np.argwhere(mask):
        R = int(rr + dr)
        C = int(cc + dc)
        if not (0 <= R < H and 0 <= C < W):
            return False
        val = int(arr[rr, cc])
        if val == bg:
            continue
        if int(out[R, C]) != bg and int(out[R, C]) != val:
            return False
    return True


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int) -> list[tuple[int, np.ndarray, np.ndarray, int]]:
    fg = (g != bg) & (~hub_mask)
    lab, nc = label_mask(fg, connectivity=8)
    out: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == cid
        arr = np.where(comp_mask, g, bg)
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        out.append((int(cid), arr, comp_mask, int(comp_mask.sum())))
    out.sort(key=lambda x: -x[3])
    return out


def _shape_only_matches(mask: np.ndarray) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for krot in range(4):
        rot = np.rot90(mask, krot)
        rot = _crop_to_mask(rot.astype(int), rot)[1]
        for side in SIDES:
            if _contact_band(rot, side, PATTERNS[side]) is not None:
                matches.append((int(krot), side))
    return matches


def _anchor_corner_ok(arr: np.ndarray, mask: np.ndarray, side: str) -> bool:
    pts = np.argwhere(mask & (arr == ANCHOR_COLOR))
    if pts.shape[0] != 1:
        return False
    rr, cc = map(int, pts[0])
    h, w = mask.shape
    want = ANCHOR_RULE[side]
    if want == "ul":
        return rr == 0 and cc == 0
    if want == "ur":
        return rr == 0 and cc == w - 1
    if want == "ll":
        return rr == h - 1 and cc == 0
    return rr == h - 1 and cc == w - 1


def _anchored_matches(arr: np.ndarray, mask: np.ndarray) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for krot in range(4):
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        colors = {int(v) for v in rot_arr[rot_mask]}
        if ANCHOR_COLOR not in colors or len(colors) < 3:
            continue
        for side in SIDES:
            if _contact_band(rot_mask, side, PATTERNS[side]) is None:
                continue
            if not _anchor_corner_ok(rot_arr, rot_mask, side):
                continue
            matches.append((int(krot), side))
    return matches


def _match_component_to_side(
    comp_arr: np.ndarray,
    comp_mask: np.ndarray,
    side: str,
    pattern: tuple[int, ...],
    hub_r: int,
    hub_c: int,
    out: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    hits = []
    for krot in range(4):
        arr = np.rot90(comp_arr, krot)
        mask = np.rot90(comp_mask, krot)
        arr, mask = _crop_to_mask(arr, mask)
        colors = {int(v) for v in arr[mask]}
        if ANCHOR_COLOR not in colors or len(colors) < 3:
            continue
        shift = _attachment_shift(mask, side, pattern, hub_r, hub_c)
        if shift is None:
            continue
        if not _anchor_corner_ok(arr, mask, side):
            continue
        dr, dc = shift
        if not _placeable(out, arr, mask, dr, dc, bg):
            continue
        hits.append((arr, mask, int(dr), int(dc)))
    if len(hits) != 1:
        return None
    return hits[0]


def _orient_from_base(arr: np.ndarray, side: str) -> np.ndarray:
    if side in ("top", "left"):
        return arr.copy()
    return np.rot90(arr, 2)


def _base_anchor_cell(mask: np.ndarray, side: str) -> tuple[int, int]:
    h, w = mask.shape
    if side == "top":
        return 0, 0
    if side == "bot":
        return h - 1, 0
    if side == "left":
        return h - 1, 0
    return 0, 0


def _make_sprite(side: str, outer_color: int, inner_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    templates = HORIZONTAL_TEMPLATES if side in ("top", "bot") else VERTICAL_TEMPLATES
    base_mask, inner_candidates = rng.choice(templates)

    base_arr = np.full(base_mask.shape, -1, dtype=int)
    base_arr[base_mask] = int(outer_color)

    if len(inner_candidates) == 1:
        inner_cells = inner_candidates
    else:
        inner_cells = rng.choice([inner_candidates[:1], inner_candidates])

    for rr, cc in inner_cells:
        base_arr[int(rr), int(cc)] = int(inner_color)

    ar, ac = _base_anchor_cell(base_mask, side)
    base_arr[int(ar), int(ac)] = int(ANCHOR_COLOR)

    arr = _orient_from_base(base_arr, side)
    mask = arr != -1
    arr = np.where(mask, arr, 0).astype(int)
    arr, mask = _crop_to_mask(arr, mask)
    return arr, mask


def _make_distractor(outer_color: int, accent_palette: tuple[int, ...], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(500):
        h = rng.randint(4, 5)
        w = rng.randint(4, 5)
        mask = np.zeros((h, w), dtype=bool)
        k = rng.randint(6, min(10, h * w - 1))
        cells = rng.sample([(r, c) for r in range(h) for c in range(w)], k)
        for rr, cc in cells:
            mask[rr, cc] = True
        mask = _crop_to_mask(mask.astype(int), mask)[1]
        lab, nc = label_mask(mask, connectivity=8)
        if nc != 1:
            continue
        if _shape_only_matches(mask):
            continue

        arr = np.full(mask.shape, 0, dtype=int)
        arr[mask] = int(outer_color)
        filled = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        anchor_pos = rng.choice(filled)
        inner_choices = [pt for pt in filled if pt != anchor_pos]
        if not inner_choices:
            continue
        inner_pos = rng.choice(inner_choices)
        arr[anchor_pos] = int(ANCHOR_COLOR)
        arr[inner_pos] = int(rng.choice(accent_palette))

        if _anchored_matches(arr, mask):
            continue
        return arr, mask
    return None


def _draw_hub(canvas: np.ndarray, hub_r: int, hub_c: int, color: int, bg: int) -> None:
    n = HUB_N

    for cc in range(1, n - 1):
        canvas[hub_r, hub_c + cc] = color
        canvas[hub_r + n - 1, hub_c + cc] = color
    for rr in range(1, n - 1):
        canvas[hub_r + rr, hub_c] = color
        canvas[hub_r + rr, hub_c + n - 1] = color

    span_c0 = hub_c + 1
    span_r0 = hub_r + 1

    canvas[hub_r - 1, span_c0 : span_c0 + EDGE_LEN] = color
    canvas[hub_r + HUB_N, span_c0 : span_c0 + EDGE_LEN] = color
    canvas[span_r0 : span_r0 + EDGE_LEN, hub_c - 1] = color
    canvas[span_r0 : span_r0 + EDGE_LEN, hub_c + HUB_N] = color

    for i, v in enumerate(PATTERNS["top"]):
        if not v:
            canvas[hub_r - 2, span_c0 + i] = color
    for i, v in enumerate(PATTERNS["bot"]):
        if not v:
            canvas[hub_r + HUB_N + 1, span_c0 + i] = color
    for i, v in enumerate(PATTERNS["left"]):
        if not v:
            canvas[span_r0 + i, hub_c - 2] = color
    for i, v in enumerate(PATTERNS["right"]):
        if not v:
            canvas[span_r0 + i, hub_c + HUB_N + 1] = color

    # Ensure the intended holes stay bg.
    for i, v in enumerate(PATTERNS["top"]):
        if v:
            canvas[hub_r - 2, span_c0 + i] = bg
    for i, v in enumerate(PATTERNS["bot"]):
        if v:
            canvas[hub_r + HUB_N + 1, span_c0 + i] = bg
    for i, v in enumerate(PATTERNS["left"]):
        if v:
            canvas[span_r0 + i, hub_c - 2] = bg
    for i, v in enumerate(PATTERNS["right"]):
        if v:
            canvas[span_r0 + i, hub_c + HUB_N + 1] = bg


def _attach_sprite(
    out: np.ndarray,
    arr: np.ndarray,
    mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    shift = _attachment_shift(mask, side, PATTERNS[side], hub_r, hub_c)
    if shift is None:
        return None
    dr, dc = shift
    if not _anchor_corner_ok(arr, mask, side):
        return None
    if not _placeable(out, arr, mask, dr, dc, bg):
        return None
    patch = np.where(mask, arr, bg).astype(int)
    paste(out, patch, dr, dc, transparent=bg)
    return patch, mask, int(dr), int(dc)


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    hub_r, hub_c, _n, _color = _find_core_hub(g, bg)
    lab, _ = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[hub_r, hub_c + 1])
    metrics = {
        "sprite_components": 0,
        "three_color_components": 0,
        "shape_ambiguous": 0,
        "anchored_unique": 0,
    }
    for cid in range(1, int(lab.max()) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        metrics["sprite_components"] += 1
        colors = {int(v) for v in arr[mask]}
        if len(colors) >= 3 and ANCHOR_COLOR in colors:
            metrics["three_color_components"] += 1
        shape_matches = _shape_only_matches(mask)
        anchor_matches = _anchored_matches(arr, mask)
        side_set = {side for _k, side in shape_matches}
        if len(shape_matches) == 2 and side_set in ({"top", "bot"}, {"left", "right"}):
            metrics["shape_ambiguous"] += 1
        if len(anchor_matches) == 1:
            metrics["anchored_unique"] += 1
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub_r, hub_c, hub_n, _hub_color = _find_core_hub(g, bg)
    assert hub_n == HUB_N

    lab, _nc = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[hub_r, hub_c + 1])
    assert hub_id > 0
    hub_mask = lab == hub_id

    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]

    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)
    components = _extract_components(g, hub_mask, bg)

    used: set[int] = set()
    for side in SIDES:
        found = None
        found_id = None
        for cid, arr, mask, _sz in components:
            if cid in used:
                continue
            match = _match_component_to_side(arr, mask, side, hub_patterns[side], hub_r, hub_c, out, bg)
            if match is None:
                continue
            found = match
            found_id = cid
            break
        if found is None:
            continue
        arr, mask, dr, dc = found
        paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=True)
        used.add(int(found_id))

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    outer_color = int(rng.choice(VISIBLE_COLORS))
    accent_palette = tuple(c for c in VISIBLE_COLORS if c != outer_color)

    H = rng.randint(20, 28)
    W = rng.randint(22, 28)

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), K)

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        inner_color = int(rng.choice(accent_palette[: min(3, len(accent_palette))]))
        arr, mask = _make_sprite(side, outer_color, inner_color, rng)

        shape_matches = _shape_only_matches(mask)
        anchor_matches = _anchored_matches(arr, mask)
        if len(shape_matches) != 2:
            return None
        if {s for _k, s in shape_matches} != {side, OPPOSITE[side]}:
            return None
        if anchor_matches != [(0, side)]:
            return None
        sprite_defs[side] = (arr, mask)

    max_sprite_h = max(arr.shape[0] for arr, _mask in sprite_defs.values())
    max_sprite_w = max(arr.shape[1] for arr, _mask in sprite_defs.values())
    margin = DOCK_THICKNESS + max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    _draw_hub(out, hub_r, hub_c, outer_color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        arr, mask = sprite_defs[side]
        placed = _attach_sprite(out, arr, mask, side, hub_r, hub_c, bg)
        if placed is None:
            return None
        attached[side] = placed

    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(inp, hub_r, hub_c, outer_color, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        patch, mask, _sr, _sc = attached[side]
        k = rng.choice([1, 2, 3])
        rot_patch = np.rot90(patch, k)
        rot_mask = np.rot90(mask, k)
        rot_patch, rot_mask = _crop_to_mask(rot_patch, rot_mask)
        scatter = np.where(rot_mask, rot_patch, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, scatter, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, scatter, pr, pc, transparent=bg)
        paste(inp, scatter, pr, pc, transparent=bg)

    if rng.random() < 0.35:
        distractor = _make_distractor(outer_color, accent_palette, rng)
        if distractor is not None:
            dist_arr, dist_mask = distractor
            dist_patch = np.where(dist_mask, dist_arr, bg).astype(int)
            try:
                pr, pc = random_free_location(canvas, dist_patch, transparent=bg, padding=2, padding_connectivity=8)
                paste(canvas, dist_patch, pr, pc, transparent=bg)
                paste(inp, dist_patch, pr, pc, transparent=bg)
            except ValueError:
                pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _variant_metrics(inp)
    if metrics["anchored_unique"] != K:
        return None
    if metrics["shape_ambiguous"] != K:
        return None
    if metrics["three_color_components"] < K:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
