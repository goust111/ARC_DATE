# Combo variant of 2c181942 (index 35)
# combo_spec: D1/0+D6/1+D7/2
# Name: 形状匹配+同色干扰+非对称

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior trace
#    - 原题: 中央 hub 是空心方框，四边不同色；每个散落 sprite 颜色唯一决定去向。
#      solver 先按颜色筛 sprite，再靠镜像对称+边长约束找旋转。
#    - 融合后:
#      1. hub 和所有 sprite 全都改成同一种前景色，颜色不再提供任何配对信息。
#      2. 为了让“形状配对”可见，沿 D1 的做法把 hub 扩成
#         `6x6 hollow core + 四边 2-cell 厚 dock`。
#      3. 每条边的 dock 最外沿用长度 4 的固定凹口模式编码:
#           top=1000, bot=1100, left=1010, right=1110
#         这里 1 表示 bg 凹口，0 表示该位置仍是 hub 颜色。
#      4. 真 sprite 的接触端做成互补“嘴型”:
#         贴 top 时，sprite 最底两行必须是
#           support row = 1111
#           protrusion row = 1000
#         其它三边同理。
#      5. 所有真 sprite 都是完全非对称轮廓；solver 不再检查镜像，
#         而是对每个组件显式尝试 4 个旋转，只接受唯一能和某条边
#         的 dock 模式互补的那一个。
#      6. 同色干扰 sprite 也散落在输入中，但它在 4 个旋转下都无法与
#         任一 dock 模式形成合法互补，因此必须被几何约束排除。
#    - 小例子:
#      如果 top dock 外沿是 `1000`，则只有一个 sprite 旋转后出现
#      “底部一整条 support 带 + 最左一格 protrusion”。其它同色组件要么
#      protrusion 位置不对，要么 support 带不完整，要么多个旋转都不唯一。
#
# b) Complexity calibration
#    - 若只统一颜色而不改 hub 外形，四边在视觉上无差别，任务退化。
#      因此保留 D1 的 dock 结构是必要的。
#    - 凹口模式只用 4 个固定档位，且彼此不同、也不是彼此 reverse，
#      这样 2-4 个训练样例足以让人总结“凹口↔凸起互补”。
#    - 真 sprite 仍限制在小尺寸单色 8 连通图形，避免自由度过高。
#      非对称性来自主体轮廓，而不是额外标记，因此视觉上清爽。
#
# c) Implicit assumption audit of the original main()
#    - 原 `m = (g == col) & (~hub_mask)` 假设颜色唯一映射到 side；现在失效。
#      新 solver 必须把 hub 之外的所有同色组件都当作候选，逐个做形状匹配。
#    - 原“挑最大同色组件”假设被 D6 打破；同色干扰可以比真 sprite 更大。
#      新 solver 先做几何合法性验证，再在合法候选中选择。
#    - 原镜像对称检查被 D7 打破；新 solver 改为显式枚举 4 个旋转，并要求
#      对某条 side 恰有 1 个旋转成立。
#    - 原 hub 只是一圈直边；为了让 D1 可观察，现在 hub 包含 dock。
#      这要求 `_find_hub` 只识别 6x6 hollow core，而不是整块连通域外框。
#
# d) Placement legality
#    - 生成器仍是 output-first：先摆好 hub 和 attached sprites 得到目标图，
#      再把真 sprite 旋转打散，把干扰 sprite 另外放开。
#    - hub 周围 margin 同时覆盖 dock 厚度与最大 sprite 外接框，保证不越界。
#    - `random_free_location(..., padding=2, padding_connectivity=8)` 保证 hub、
#      真 sprite、干扰 sprite 彼此不接触，不会被 solver 合并成一个组件。
#
# e) Visual observability / inferability
#    - 解题者能直接看到: 所有对象同色；hub 四边有不同凹口；散落 sprite 的
#      嘴型不同且轮廓不对称；其中还有 1 个同色但嘴型不对的干扰块。
#    - 输出里只有几何互补的那些 sprite 会回到对应 dock，颜色不会提供捷径。
#
# f) Cleanliness
#    - 图面中只有一种前景色，复杂度集中在局部轮廓。
#    - dock 只加两层，不做更多叠层；sprite 间距固定，避免局部挤在一起。
#
# g) New solver assumptions checked against generator
#    - 假设 1: hub 的 6x6 hollow core 在整图中唯一且接近中心。
#      generator 把 hub 放在中心裕量区并保持 sprites 远离它。
#    - 假设 2: 真 sprite 对全局 `(rotation, side)` 匹配集合恰好只有 1 个元素。
#      generator 通过 `_count_valid_matches(mask) == [(0, side)]` 强制保证。
#    - 假设 3: 干扰组件对所有 side 都无合法匹配。
#      generator 只接受 `_count_valid_matches(mask) == []` 的同色干扰。
#    - 假设 4: 每个真实 side 最多只有一个未使用组件能合法贴附。
#      generator 只生成每 side 一个真 sprite，并用指标检查 `real_candidate_count == K`。


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
BG_COLOR = 8
SIDES = ("top", "bot", "left", "right")
PATTERNS = {
    "top": (1, 0, 0, 0),
    "bot": (1, 1, 0, 0),
    "left": (1, 0, 1, 0),
    "right": (1, 1, 1, 0),
}


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.where(mask, 1, 0).astype(int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


def _find_core_hub(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    H, W = g.shape
    n = HUB_N
    best = None
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
            if (
                bool((top == bg).any())
                or bool((bot == bg).any())
                or bool((left == bg).any())
                or bool((right == bg).any())
            ):
                continue

            colors = set(map(int, np.concatenate([top, bot, left, right])))
            if len(colors) != 1:
                continue
            color = int(next(iter(colors)))
            if color == bg:
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
    out: dict[str, tuple[int, ...]] = {}
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
        out[side] = tuple(vals)
    return out


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
    span_start, _ = _side_span(hub_r, hub_c, side)
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


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    variants = [
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        mask.T,
        np.fliplr(mask.T),
    ]
    return all(not np.array_equal(mask, other) for other in variants)


def _count_valid_matches(mask: np.ndarray) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for krot in range(4):
        rot = _crop_mask(np.rot90(mask, krot))
        for side, pattern in PATTERNS.items():
            if _contact_band(rot, side, pattern) is not None:
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
        shift = _attachment_shift(mask, side, pattern, hub_r, hub_c)
        if shift is None:
            continue
        dr, dc = shift
        if not _placeable(out, arr, mask, dr, dc, bg):
            continue
        hits.append((arr, mask, int(dr), int(dc)))
    if len(hits) != 1:
        return None
    return hits[0]


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


def _attach_sprite(
    out: np.ndarray,
    mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    arr = np.where(mask, color, bg).astype(int)
    shift = _attachment_shift(mask, side, PATTERNS[side], hub_r, hub_c)
    if shift is None:
        return None
    dr, dc = shift
    if not _placeable(out, arr, mask, dr, dc, bg):
        return None
    paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=False)
    return arr, mask, int(dr), int(dc)


def _is_forbidden_contact_cell(
    side: str,
    rr: int,
    cc: int,
    band_start: int,
    pattern: tuple[int, ...],
    h: int,
    w: int,
) -> bool:
    L = len(pattern)
    if side == "top":
        if rr == h - 2:
            return not (band_start <= cc < band_start + L)
        if rr == h - 1:
            return not (band_start <= cc < band_start + L and pattern[cc - band_start] == 1)
    elif side == "bot":
        if rr == 1:
            return not (band_start <= cc < band_start + L)
        if rr == 0:
            return not (band_start <= cc < band_start + L and pattern[cc - band_start] == 1)
    elif side == "left":
        if cc == w - 2:
            return not (band_start <= rr < band_start + L)
        if cc == w - 1:
            return not (band_start <= rr < band_start + L and pattern[rr - band_start] == 1)
    else:
        if cc == 1:
            return not (band_start <= rr < band_start + L)
        if cc == 0:
            return not (band_start <= rr < band_start + L and pattern[rr - band_start] == 1)
    return False


def _candidate_weight(side: str, rr: int, cc: int, h: int, w: int, band_start: int) -> int:
    if side == "top":
        return 1 + (h - 1 - rr) * 3 + abs(cc - band_start)
    if side == "bot":
        return 1 + rr * 3 + abs(cc - band_start)
    if side == "left":
        return 1 + (w - 1 - cc) * 3 + abs(rr - band_start)
    return 1 + cc * 3 + abs(rr - band_start)


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    L = len(pattern)
    for _ in range(1200):
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        if side in ("top", "bot") and w < L:
            w = L
        if side in ("left", "right") and h < L:
            h = L

        mask = np.zeros((h, w), dtype=bool)
        band_start = rng.randint(0, (w - L) if side in ("top", "bot") else (h - L))

        if side == "top":
            mask[h - 2, band_start : band_start + L] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[h - 1, band_start + i] = True
        elif side == "bot":
            mask[1, band_start : band_start + L] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[0, band_start + i] = True
        elif side == "left":
            mask[band_start : band_start + L, w - 2] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[band_start + i, w - 1] = True
        else:
            mask[band_start : band_start + L, 1] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[band_start + i, 0] = True

        target_pixels = rng.randint(7, min(14, h * w - 1))
        while int(mask.sum()) < target_pixels:
            frontier = []
            for rr, cc in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if bool(mask[nr, nc]):
                            continue
                        if _is_forbidden_contact_cell(side, nr, nc, band_start, pattern, h, w):
                            continue
                        frontier.append((nr, nc))
            if not frontier:
                break
            unique = list({pt for pt in frontier})
            weights = [_candidate_weight(side, nr, nc, h, w, band_start) for nr, nc in unique]
            nr, nc = rng.choices(unique, weights=weights, k=1)[0]
            mask[nr, nc] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        area = int(mask.sum())
        if area < 7 or area > 14:
            continue

        _lab, ncomp = label_mask(mask, connectivity=8)
        if int(ncomp) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_matches(mask) != [(0, side)]:
            continue
        return mask
    return None


def _grow_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    sr = rng.randrange(h)
    sc = rng.randrange(w)
    mask[sr, sc] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = rng.choice(list(frontier))
        mask[nr, nc] = True

    return _crop_mask(mask)


def _make_same_color_distractor_mask(min_area: int, rng: random.Random) -> np.ndarray | None:
    lo = max(9, min_area)
    hi = 16
    for _ in range(900):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        area = rng.randint(lo, min(hi, h * w - 1))
        mask = _grow_connected_mask(h, w, area, rng)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < lo:
            continue
        _lab, ncomp = label_mask(mask, connectivity=8)
        if int(ncomp) != 1:
            continue
        if _count_valid_matches(mask):
            continue
        return mask
    return None


def _component_records(g: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int, np.ndarray], list[tuple[int, np.ndarray, np.ndarray, int]]]:
    hub_r, hub_c, hub_n, _hub_color = _find_core_hub(g, bg)
    lab, _ = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[hub_r, hub_c + 1])
    assert hub_id > 0
    hub_mask = lab == hub_id

    comps = []
    for cid in range(1, int(lab.max()) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        comps.append((int(cid), arr, mask, int(mask.sum())))
    comps.sort(key=lambda item: -item[3])
    return (hub_r, hub_c, hub_n, hub_id, hub_mask), comps


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    non_bg_colors = sorted(set(map(int, g[g != bg].flatten())))
    (hub_r, hub_c, _hub_n, _hub_id, _hub_mask), comps = _component_records(g, bg)
    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    real_candidates = 0
    distractors = 0
    ambiguous = 0
    asymmetric_reals = 0
    for _cid, _arr, mask, _area in comps:
        matches = _count_valid_matches(mask)
        if len(matches) == 1:
            real_candidates += 1
            if _is_fully_asymmetric(mask):
                asymmetric_reals += 1
        elif len(matches) == 0:
            distractors += 1
        else:
            ambiguous += 1

    return {
        "non_bg_colors": int(len(non_bg_colors)),
        "component_count": int(len(comps)),
        "real_candidate_count": int(real_candidates),
        "distractor_count": int(distractors),
        "ambiguous_component_count": int(ambiguous),
        "asymmetric_real_count": int(asymmetric_reals),
        "hub_pattern_ones": int(sum(sum(v) for v in hub_patterns.values())),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    (hub_r, hub_c, hub_n, _hub_id, hub_mask), components = _component_records(g, bg)
    assert hub_n == HUB_N

    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]

    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    used: set[int] = set()
    for side in SIDES:
        pattern = hub_patterns[side]
        candidates = []
        for cid, arr, mask, area in components:
            if cid in used:
                continue
            match = _match_component_to_side(arr, mask, side, pattern, hub_r, hub_c, out, bg)
            if match is None:
                continue
            placed_arr, placed_mask, dr, dc = match
            candidates.append((int(area), int(cid), placed_arr, placed_mask, int(dr), int(dc)))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        _area, cid, arr, mask, dr, dc = candidates[0]
        paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=True)
        used.add(int(cid))

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    color = rng.choice([c for c in range(10) if c != bg])

    H = rng.randint(22, 30)
    W = rng.randint(22, 30)

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(PATTERNS[side], side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = DOCK_THICKNESS + max_extent + 3

    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    _draw_hub(out, hub_r, hub_c, color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_masks[side], side, hub_r, hub_c, color, bg)
        if placed is None:
            return None
        attached[side] = placed

    largest_real = max(int(mask.sum()) for mask in sprite_masks.values())
    distractor_mask = _make_same_color_distractor_mask(largest_real + 1, rng)
    if distractor_mask is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(inp, hub_r, hub_c, color, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        rot_k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, rot_k)
        rot_mask = np.rot90(mask, rot_k)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, color, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    distractor_mask = np.rot90(distractor_mask, rng.choice([0, 1, 2, 3]))
    distractor_arr = np.where(distractor_mask, color, bg).astype(int)
    try:
        pr, pc = random_free_location(canvas, distractor_arr, transparent=bg, padding=2, padding_connectivity=8)
    except ValueError:
        return None
    paste(canvas, distractor_arr, pr, pc, transparent=bg)
    paste(inp, distractor_arr, pr, pc, transparent=bg)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = variant_metrics(inp)
    if metrics["non_bg_colors"] != 1:
        return None
    if metrics["real_candidate_count"] != K:
        return None
    if metrics["asymmetric_real_count"] != K:
        return None
    if metrics["distractor_count"] < 1:
        return None
    if metrics["ambiguous_component_count"] != 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 600 attempts")
