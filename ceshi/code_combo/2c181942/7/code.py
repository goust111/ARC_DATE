# Combo variant of 2c181942 (index 7)
# combo_spec: D1/0+D6/1
# Name: 形状匹配+同色干扰精灵

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
# ---------------------------------------------------------------------------
#
# a) 具体化与行为追踪
#    - 原题：中心是 4 色 hollow square；每个散落精灵靠颜色直接配对到某一边，
#      然后再靠“接触边长度 + 镜像对称”找旋转。
#    - D1/0 改动后：颜色线索彻底取消。hub 与所有 sprite 全都同色，因此如果 hub
#      仍保持原来的直边，四边在视觉上不可区分，规则无法观察。所以这里沿用单变体
#      里的最小适配：保留一个 6x6 hollow core，并在四边外侧增加 2-cell 厚的 dock；
#      dock 最外沿挖出 4 格凹口模式，sprite 的接触带做互补凸起。
#    - D6/1 融合后：输入中稳定再放入 1 个同色干扰精灵。它与真实精灵颜色完全一致，
#      但在任何旋转、任何边上都无法形成“support 带全满 + protrusion 带等于该边凹口”
#      的合法接触，所以 solver 必须逐组件验证几何互补，不能靠颜色，也不能靠取最大连通块。
#    - 小例子：若 top 边外沿凹口模式是 1000（1 表示该位置是 bg 缺口），那么合法 top
#      sprite 在贴附朝向时，倒数第二行必须是连续 4 格支撑带，最底行只在最左 1 格有凸起。
#      一个同色干扰若最底行是 0110 或支撑带断开，就会被排除。
#
# b) 复杂度校准
#    - 只使用 4 个固定模式：top=1000, bot=1100, left=1010, right=1110。
#      它们彼此不同，且没有任何一对互为 reverse，降低“旋转后误配别边”的风险。
#    - core 固定为 6x6，使接触带长度固定为 4；这样轮廓编码足够可见，又不会把自由度放得过大。
#    - 干扰精灵固定为同色且面积通常比真实精灵更大，确保“最大连通块优先”这类旧假设会失效，
#      从视觉上也更容易看出：散落同色物体里并不是每个都能插回 hub。
#
# c) 隐式假设审查
#    - 原 main 中基于 `m = (g == col) & (~hub_mask)` 的按颜色分桶彻底失效，因为整图只有 1 种前景色。
#      新 solver 必须先识别 hub 组件，再把其余所有 8 连通组件都当作候选 sprite。
#    - 原 main 假设 hub 四边颜色互异；现在改为“四边同色但 dock 外沿模式不同”。因此 hub 检测仍可使用
#      空心 core 的几何，但不能再把边色当 side id。
#    - 原 main 借助镜像对称和接触边长度判定合法旋转；融合后改成更强的轮廓判定：
#      每个组件在某一 side 上必须恰有 1 个旋转产生合法接触带。
#    - 原 generator 的“新颜色干扰”也失效；现在干扰必须和真实精灵同色，且被 `_count_valid_matches(mask)==[]`
#      这一可见几何条件排除。
#
# d) 放置合法性
#    - 仍采用 output-first：先构造 hub + attached sprites 的目标图，再把真实精灵旋转后散开。
#    - hub 周围 margin 同时覆盖 dock 厚度与最大真实 sprite 外接框，保证 attached output 不越界。
#    - 输入散布使用 `random_free_location(..., padding=2, padding_connectivity=8)`，保证 hub、真实精灵、
#      干扰精灵始终分离成独立 8 连通组件。
#    - attached sprite 允许与 hub 已画出的同色 dock 重叠，但不允许压到异色或越界位置。
#
# e) 视觉可观察性
#    - 每个样例都只有 1 种前景色，解题者立刻能看到“颜色不再提供任何信息”。
#    - hub 四边外沿各有不同凹口，散落精灵上各有不同嘴型；输出中只有嘴型互补者被贴回。
#    - 同时稳定存在 1 个同色但不合法的干扰精灵，解题者会看到“不是所有同色物体都会被回收”。
#
# f) 可视化整洁度
#    - hub 结构只由 core + dock 两层组成，不叠加额外标记。
#    - 所有散落组件之间保持 Chebyshev 间距 >= 3，避免同色对象视觉上粘连。
#    - 真实 sprite 的 body 噪声只加在远离接触带的一侧，避免破坏轮廓识别。
#
# g) 变化的可推断性
#    - 颜色变化被彻底压缩到“只有 1 种前景色”，不存在需要额外归纳的颜色映射。
#    - 可推断信息全部落在可见几何上：hub 边的凹口模式 ↔ sprite 接触带的凸起模式。
#      由于 4 种模式在每个样例中重复出现，2-4 组样例足够支持人类归纳。
#
# h) 新 solver 假设与 generator 保证
#    - 假设 1：hub 是唯一一个包含 6x6 hollow core 且最接近中心的同色组件。
#      generator 固定只生成这一种 hub，并让其它组件远离中心。
#    - 假设 2：每个真实 sprite 对全部 (rotation, side) 组合恰好只有 1 个合法命中；
#      generator 在生成时用 `_count_valid_matches(mask) == [(0, side)]` 强制这一点。
#    - 假设 3：干扰精灵对全部 (rotation, side) 组合没有合法命中；
#      generator 用 `_count_valid_matches(mask) == []` 强制这一点。
#    - 假设 4：每个 side 最多只有 1 个未使用组件能合法匹配；
#      generator 的唯一模式约束和独立组件散布共同保证这一点。


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
BG_COLOR = 8
SIDES = ("top", "bot", "left", "right")
PATTERNS: dict[str, tuple[int, ...]] = {
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
    best = None
    n = HUB_N
    for r in range(DOCK_THICKNESS, H - n - DOCK_THICKNESS):
        for c in range(DOCK_THICKNESS, W - n - DOCK_THICKNESS):
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
    patterns: dict[str, tuple[int, ...]] = {}
    for side in SIDES:
        start, end = _side_span(r, c, side)
        outer = _dock_outer_coord(r, c, side)
        vals: list[int] = []
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


def _count_valid_matches(mask: np.ndarray) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for krot in range(4):
        rot = _crop_mask(np.rot90(mask, krot))
        for side, pattern in PATTERNS.items():
            if _contact_band(rot, side, pattern) is not None:
                matches.append((int(krot), side))
    return matches


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
    hits: list[tuple[np.ndarray, np.ndarray, int, int]] = []
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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub_r, hub_c, hub_n, _hub_color = _find_core_hub(g, bg)
    assert hub_n == HUB_N

    lab, _nc = label_mask(g != bg, connectivity=8)
    core_seed = (hub_r, hub_c + 1)
    hub_id = int(lab[core_seed])
    assert hub_id > 0
    hub_mask = lab == hub_id

    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]
    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    components: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for cid in range(1, int(lab.max()) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        components.append((int(cid), arr, mask, int(mask.sum())))

    used: set[int] = set()
    for side in SIDES:
        pattern = hub_patterns[side]
        candidates: list[tuple[int, int, np.ndarray, np.ndarray, int, int]] = []
        for cid, arr, mask, area in components:
            if cid in used:
                continue
            match = _match_component_to_side(arr, mask, side, pattern, hub_r, hub_c, out, bg)
            if match is None:
                continue
            marr, mmask, dr, dc = match
            candidates.append((area, cid, marr, mmask, dr, dc))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        _area, cid, arr, mask, dr, dc = candidates[0]
        paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=True)
        used.add(int(cid))

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _add_body_noise(mask: np.ndarray, side: str, band_start: int, rng: random.Random) -> None:
    h, w = mask.shape

    if side == "top":
        if h >= 3:
            row = h - 3
            start = max(0, min(w - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, w - start))
            mask[row, start : start + length] = True
        for rr in range(0, max(0, h - 3)):
            for cc in range(w):
                if rng.random() < 0.28:
                    mask[rr, cc] = True
    elif side == "bot":
        if h >= 3:
            row = 2
            start = max(0, min(w - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, w - start))
            mask[row, start : start + length] = True
        for rr in range(3, h):
            for cc in range(w):
                if rng.random() < 0.28:
                    mask[rr, cc] = True
    elif side == "left":
        if w >= 3:
            col = w - 3
            start = max(0, min(h - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, h - start))
            mask[start : start + length, col] = True
        for rr in range(h):
            for cc in range(0, max(0, w - 3)):
                if rng.random() < 0.28:
                    mask[rr, cc] = True
    else:
        if w >= 3:
            col = 2
            start = max(0, min(h - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, h - start))
            mask[start : start + length, col] = True
        for rr in range(h):
            for cc in range(3, w):
                if rng.random() < 0.28:
                    mask[rr, cc] = True


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    L = len(pattern)
    for _ in range(600):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        if side in ("top", "bot") and w < L:
            w = L
        if side in ("left", "right") and h < L:
            h = L

        mask = np.zeros((h, w), dtype=bool)
        band_start = rng.randint(0, w - L) if side in ("top", "bot") else rng.randint(0, h - L)

        if side == "top":
            mask[h - 2, band_start : band_start + L] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[h - 1, band_start + i] = True
        elif side == "bot":
            mask[1, band_start : band_start + L] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[0, band_start + i] = True
        elif side == "left":
            mask[band_start : band_start + L, w - 2] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[band_start + i, w - 1] = True
        else:
            mask[band_start : band_start + L, 1] = True
            for i, v in enumerate(pattern):
                if v:
                    mask[band_start + i, 0] = True

        _add_body_noise(mask, side, band_start, rng)

        if not bool(mask.any()):
            continue
        mask = _crop_mask(mask)
        area = int(mask.sum())
        if area < 6 or area > 14:
            continue

        _lab, nc = label_mask(mask, connectivity=8)
        if nc != 1:
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

    while int(mask.sum()) < int(area):
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


def _make_large_invalid_mask(min_area: int, rng: random.Random) -> np.ndarray | None:
    lo = max(int(min_area), 15)
    for _ in range(800):
        h = rng.randint(5, 6)
        w = rng.randint(5, 6)
        max_area = min(h * w - 1, 22)
        if lo > max_area:
            continue
        area = rng.randint(lo, max_area)
        mask = _grow_connected_mask(h, w, area, rng)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < lo:
            continue
        _lab, nc = label_mask(mask, connectivity=8)
        if nc != 1:
            continue
        if _count_valid_matches(mask):
            continue
        return mask
    return None


def _draw_hub(canvas: np.ndarray, hub_r: int, hub_c: int, color: int) -> None:
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


def _fusion_metrics(inp: np.ndarray) -> dict[str, object]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    hub_r, hub_c, _n, _color = _find_core_hub(g, bg)
    lab, nc = label_mask(g != bg, connectivity=8)
    hub_id = int(lab[hub_r, hub_c + 1])

    non_bg_colors = sorted(set(map(int, g[g != bg].flatten())))
    valid_components = 0
    invalid_components = 0
    ambiguous_components = 0
    valid_sides: list[str] = []
    largest_valid_area = 0
    largest_invalid_area = 0

    for cid in range(1, int(nc) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        mask = _crop_mask(mask)
        matches = _count_valid_matches(mask)
        area = int(mask.sum())
        if len(matches) == 1:
            valid_components += 1
            valid_sides.append(matches[0][1])
            largest_valid_area = max(largest_valid_area, area)
        elif len(matches) == 0:
            invalid_components += 1
            largest_invalid_area = max(largest_invalid_area, area)
        else:
            ambiguous_components += 1

    return {
        "non_bg_colors": len(non_bg_colors),
        "valid_components": int(valid_components),
        "invalid_components": int(invalid_components),
        "ambiguous_components": int(ambiguous_components),
        "valid_sides": tuple(sorted(valid_sides)),
        "largest_valid_area": int(largest_valid_area),
        "largest_invalid_area": int(largest_invalid_area),
    }


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
    margin = DOCK_THICKNESS + max_extent + 2

    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    _draw_hub(out, hub_r, hub_c, color)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_masks[side], side, hub_r, hub_c, color, bg)
        if placed is None:
            return None
        attached[side] = placed

    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(inp, hub_r, hub_c, color)
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, k)
        rot_mask = np.rot90(mask, k)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, rot_arr, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    max_real_area = max(int(mask.sum()) for mask in sprite_masks.values())
    dist_mask = _make_large_invalid_mask(max_real_area + 1, rng)
    if dist_mask is None:
        return None
    dist_mask = np.rot90(dist_mask, rng.choice([0, 1, 2, 3]))
    dist_arr = np.where(dist_mask, color, bg).astype(int)
    try:
        pr, pc = random_free_location(canvas, dist_arr, transparent=bg, padding=2, padding_connectivity=8)
    except ValueError:
        return None
    paste(canvas, dist_arr, pr, pc, transparent=bg)
    paste(inp, dist_arr, pr, pc, transparent=bg)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _fusion_metrics(inp)
    if metrics["non_bg_colors"] != 1:
        return None
    if metrics["valid_components"] != K:
        return None
    if metrics["invalid_components"] != 1:
        return None
    if metrics["ambiguous_components"] != 0:
        return None
    if metrics["valid_sides"] != tuple(sorted(sprite_sides)):
        return None
    if int(metrics["largest_invalid_area"]) <= int(metrics["largest_valid_area"]):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
