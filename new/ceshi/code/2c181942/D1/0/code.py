# Variant of 2c181942
# Dimension: D1 — 匹配条件：颜色配对机制 (match_cond)
# Change: s.head.color == d.edge.color — 精灵颜色与hub边缘颜色相同即配对，颜色唯一决定精灵应贴附... → 形状轮廓匹配替代颜色匹配

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for this D1 variant
# ---------------------------------------------------------------------------
#
# spec_variant:
#   preserved_core:
#     - 仍然有一个接近中心的空心方块 core，内部和四角都是 bg。
#     - 仍然是 output-first：先构造 hub+attached sprites 的目标图，再把精灵旋转散开到输入图。
#     - 仍然允许 2-4 个真实精灵，0-1 个干扰精灵，背景仍取 8。
#   d1_change:
#     - 原题按颜色把精灵直接分配给四边；这个前提在变体中被彻底移除。
#     - hub 与所有 sprites 全都使用同一种前景色，因此“看颜色找对应边”完全失效。
#     - 为了让形状匹配在视觉上可观察，core 外侧新增 2-cell 厚的同色 dock。
#     - dock 的最外沿挖出长度为 4 的不同凹口模式；sprite 的接触带在同一位置做凸起去补这些凹口。
#   concrete_geometry:
#     - core 固定为 6x6 hollow square。之所以从原来的常用 N=4 调到 N=6，是因为 edge_len=4 才足够承载可见的轮廓编码。
#     - 四边 dock 厚度固定为 2：
#         top: rows r-2, r-1 and cols c+1..c+4
#         bot: rows r+6, r+7 and cols c+1..c+4
#         left: cols c-2, c-1 and rows r+1..r+4
#         right: cols c+6, c+7 and rows r+1..r+4
#     - 外沿的 4-bit 凹口模式使用少量固定档位，避免自由度过高：
#         top   = 1000
#         bot   = 1100
#         left  = 1010
#         right = 1110
#       这些模式互不相同，且没有任何一个是另一个的 reverse，因此旋转后不会意外匹配别边。
#   behavior_trace:
#     original:
#       - solver 先找 hub，再按 top/bot/left/right 的边颜色从整张图中筛出同色精灵。
#       - 旋转精灵直到接触边长度与 hub 边长度一致，再贴上去。
#     variant:
#       - solver 仍先找中心 core，但随后不再按颜色分组，因为整图只有 1 种前景色。
#       - solver 把包含 core 的 8-连通组件当作 hub，其余组件全视为 sprite candidates。
#       - 对 hub 的四边读取 dock 外沿的凹口模式。
#       - 对每个候选 sprite 尝试 4 个旋转；若某一朝向在某一边出现“support 带全满 + protrusion 带恰等于该边凹口模式”，则视为匹配。
#       - 然后按固定几何关系把该 sprite 平移到该边。
#     tiny_example:
#       - 若 top dock 外沿是 1000，则 top sprite 的最底行必须只有最左 1 个凸起；其上方一行则必须是连续 4 格支撑带。
#       - 原题在这里是“看 top 的颜色”；变体变成“看 top 的凹口位置并找几何互补的嘴型”。
#   complexity_calibration:
#     - 如果只把颜色统一而 hub 保持直边，任务会退化：四边没有任何可见差异，solver 和人都无法匹配。
#     - 因此允许对 hub 增加 dock，是 D1 可观察化所必需的最小适配。
#     - 轮廓模式只用 4 个固定档位，并且每个样例都同时展示 4 条边的 4 种模式，人类可在 2-4 组样例里看出“凹口 ↔ 凸起”的规则。
#   implicit_assumption_audit:
#     - 原 main 中 `m = (g == col) & (~hub_mask)` 依赖“颜色唯一对应一边”；现在失效，必须改为“非 hub 的所有组件逐个尝试匹配”。
#     - 原 main 中 `_find_hub()` 要求四边各自纯色；变体仍保留这一点，但四边颜色不再彼此不同。
#     - 原 main 中通过镜像对称筛旋转；变体中不再依赖这个前提，而改为读取接触带模式，因此 generator 也不必强造镜像精灵。
#     - 原 generator 中干扰物用新颜色即可忽略；现在颜色无用，所以干扰物必须和真精灵同色，但保证没有任何合法接触带模式。
#   placement_legality:
#     - hub 放在带足够 margin 的中心区域，margin 同时覆盖 dock 厚度和最大 sprite 尺寸。
#     - attached sprite 只允许与 hub 的 bg 凹口重叠；若压到已有前景色则立即判非法。
#     - scattered 输入使用 `random_free_location(..., padding=2)`，保持与 hub 和其它组件的可辨识间距。
#   visual_observability:
#     - 每个 input 中都能直接看到：一个同色 hub，四边外沿有不同凹口；散落同色 sprite 上有不同凸起嘴型。
#     - output 中只有嘴型吻合的 sprite 会被移回对应边，颜色完全不提供帮助。
#   cleanliness:
#     - 所有前景元素只使用一种颜色，视觉复杂度主要来自轮廓，因此采用固定 dock 厚度和固定长度 4 的接触带，避免图面过乱。
#     - 不在同一区域叠加多重标记；hub 的可见结构只由 core + dock 组成。


HUB_N = 6
EDGE_LEN = HUB_N - 2
DOCK_THICKNESS = 2
BG_COLOR = 8
PATTERNS = {
    "top": (1, 0, 0, 0),
    "bot": (1, 1, 0, 0),
    "left": (1, 0, 1, 0),
    "right": (1, 1, 1, 0),
}
SIDES = ("top", "bot", "left", "right")


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
        vals = []
        if side in ("top", "bot"):
            for cc in range(start, end + 1):
                vals.append(1 if int(g[outer, cc]) == bg else 0)
        else:
            for rr in range(start, end + 1):
                vals.append(1 if int(g[rr, outer]) == bg else 0)
        patterns[side] = tuple(vals)
    return patterns


def _component_arrays(g: np.ndarray, bg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    lab, nc = label_mask(g != bg, connectivity=8)
    comps: list[tuple[np.ndarray, np.ndarray]] = []
    for cid in range(1, nc + 1):
        mask = lab == cid
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        comps.append((arr, mask))
    return comps


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
        rot = np.rot90(mask, krot)
        rot = _crop_to_mask(rot.astype(int), rot)[1]
        for side, pattern in PATTERNS.items():
            if _contact_band(rot, side, pattern) is not None:
                matches.append((krot, side))
    return matches


def _placeable(
    out: np.ndarray,
    arr: np.ndarray,
    mask: np.ndarray,
    dr: int,
    dc: int,
    bg: int,
) -> bool:
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
        hits.append((arr, mask, dr, dc))
    if len(hits) != 1:
        return None
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub_r, hub_c, hub_n, hub_color = _find_core_hub(g, bg)
    assert hub_n == HUB_N

    lab, _nc = label_mask(g != bg, connectivity=8)
    core_seed = (hub_r, hub_c + 1)
    hub_id = int(lab[core_seed])
    assert hub_id > 0
    hub_mask = lab == hub_id

    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]

    hub_patterns = _read_hub_patterns(g, hub_r, hub_c, bg)

    components = []
    for cid in range(1, int(lab.max()) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        arr = np.where(mask, g, bg)
        arr, mask = _crop_to_mask(arr, mask)
        components.append((cid, arr, mask, int(mask.sum())))
    components.sort(key=lambda x: -x[3])

    used: set[int] = set()
    for side in SIDES:
        pattern = hub_patterns[side]
        found = None
        found_id = None
        for cid, arr, mask, _sz in components:
            if cid in used:
                continue
            match = _match_component_to_side(arr, mask, side, pattern, hub_r, hub_c, out, bg)
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


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _add_body_noise(mask: np.ndarray, side: str, band_start: int, rng: random.Random) -> None:
    h, w = mask.shape
    L = EDGE_LEN

    if side == "top":
        if h >= 3:
            row = h - 3
            start = max(0, min(w - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, w - start))
            mask[row, start : start + length] = True
        for rr in range(0, max(0, h - 3)):
            for cc in range(w):
                if rng.random() < 0.3:
                    mask[rr, cc] = True
    elif side == "bot":
        if h >= 3:
            row = 2
            start = max(0, min(w - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, w - start))
            mask[row, start : start + length] = True
        for rr in range(3, h):
            for cc in range(w):
                if rng.random() < 0.3:
                    mask[rr, cc] = True
    elif side == "left":
        if w >= 3:
            col = w - 3
            start = max(0, min(h - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, h - start))
            mask[start : start + length, col] = True
        for rr in range(h):
            for cc in range(0, max(0, w - 3)):
                if rng.random() < 0.3:
                    mask[rr, cc] = True
    else:
        if w >= 3:
            col = 2
            start = max(0, min(h - 1, band_start + rng.randint(-1, 1)))
            length = rng.randint(1, min(3, h - start))
            mask[start : start + length, col] = True
        for rr in range(h):
            for cc in range(3, w):
                if rng.random() < 0.3:
                    mask[rr, cc] = True

    if side in ("top", "bot"):
        for cc in range(w):
            col_sum = int(mask[:, cc].sum())
            if col_sum and col_sum == h:
                cut_r = 0 if side == "top" else h - 1
                if not ((side == "top" and cut_r >= h - 2) or (side == "bot" and cut_r <= 1)):
                    mask[cut_r, cc] = False
    else:
        for rr in range(h):
            row_sum = int(mask[rr, :].sum())
            if row_sum and row_sum == w:
                cut_c = 0 if side == "left" else w - 1
                if not ((side == "left" and cut_c >= w - 2) or (side == "right" and cut_c <= 1)):
                    mask[rr, cut_c] = False


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    L = len(pattern)
    for _ in range(500):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        if side in ("top", "bot") and w < L:
            w = L
        if side in ("left", "right") and h < L:
            h = L
        mask = np.zeros((h, w), dtype=bool)

        if side in ("top", "bot"):
            band_start = rng.randint(0, w - L)
        else:
            band_start = rng.randint(0, h - L)

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
        mask = _crop_to_mask(mask.astype(int), mask)[1]
        if int(mask.sum()) < 6 or int(mask.sum()) > 14:
            continue

        lab, nc = label_mask(mask, connectivity=8)
        if nc != 1:
            continue

        matches = _count_valid_matches(mask)
        if matches != [(0, side)]:
            continue
        return mask
    return None


def _make_distractor_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(500):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        mask = np.zeros((h, w), dtype=bool)
        k = rng.randint(4, min(8, h * w))
        cells = rng.sample([(r, c) for r in range(h) for c in range(w)], k)
        for rr, cc in cells:
            mask[rr, cc] = True
        mask = _crop_to_mask(mask.astype(int), mask)[1]
        if int(mask.sum()) < 4:
            continue
        lab, nc = label_mask(mask, connectivity=8)
        if nc != 1:
            continue
        if _count_valid_matches(mask):
            continue
        return mask
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
    return arr, mask, dr, dc


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    non_bg_colors = sorted(set(map(int, g[g != bg].flatten())))
    hub_r, hub_c, _n, _color = _find_core_hub(g, bg)
    pats = _read_hub_patterns(g, hub_r, hub_c, bg)
    protrusions = sum(sum(v) for v in pats.values())
    lab, nc = label_mask(g != bg, connectivity=8)
    core_seed = (hub_r, hub_c + 1)
    hub_id = int(lab[core_seed])
    sprite_components = sum(1 for cid in range(1, nc + 1) if cid != hub_id)
    return {
        "non_bg_colors": len(non_bg_colors),
        "hub_protrusions": int(protrusions),
        "sprite_components": int(sprite_components),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    color = rng.choice([c for c in range(10) if c != bg])

    H = rng.randint(20, 28)
    W = rng.randint(22, 28)

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
    _draw_hub(out, hub_r, hub_c, color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_masks[side], side, hub_r, hub_c, color, bg)
        if placed is None:
            return None
        attached[side] = placed

    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(inp, hub_r, hub_c, color, bg)
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

    if rng.random() < 0.35:
        dist_mask = _make_distractor_mask(rng)
        if dist_mask is not None:
            dist_arr = np.where(dist_mask, color, bg).astype(int)
            try:
                pr, pc = random_free_location(canvas, dist_arr, transparent=bg, padding=2, padding_connectivity=8)
                paste(canvas, dist_arr, pr, pc, transparent=bg)
                paste(inp, dist_arr, pr, pc, transparent=bg)
            except ValueError:
                pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _variant_metrics(inp)
    if metrics["non_bg_colors"] != 1:
        return None
    if metrics["hub_protrusions"] < 7:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
