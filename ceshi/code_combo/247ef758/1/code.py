# Combo variant of 247ef758 (index 1)
# combo_spec: D1/1+D2/0
# Name: 颜色尺寸双匹配+多色精灵
#
# Design analysis
# ===============
# fused_rule:
#   - 保留原布局: 左侧精灵条 | 单列分隔线 | 右侧整高矩形框。
#   - 左侧每个精灵是一个双色 8 连通组件; 其中主色是严格多数色, 辅色只在印刻时保留。
#   - 框架边缘标记不再是单格, 而是镜像连续同色段:
#     左/右边缘竖段编码目标行, 段长 = 精灵外接矩形高度;
#     上/下边缘横段编码目标列, 段长 = 精灵外接矩形宽度。
#   - 匹配键不是单独颜色, 而是:
#       行方向: (主色, bbox 高度)
#       列方向: (主色, bbox 宽度)
#     命中后取段的 floor-center 作为目标坐标, 再对所有行中心 × 列中心做笛卡尔积印刻。
#
# behavior_trace:
#   - 例: 某个双色精灵 bbox = 3x2, 其中红色 5 像素, 黄色 2 像素, 所以主色=红。
#   - 原题会按“红色行标 × 红色列标”复制, 并且印刻成纯红。
#   - 本融合版必须先把整个双色组件提出来, 数主色=红, 然后只接受:
#       左/右边缘长度 3 的红色竖段;
#       上/下边缘长度 2 的红色横段。
#   - 如果框上另有长度 4 的红色竖段或长度 3 的红色横段, 颜色虽对, 但尺寸不对, 必须忽略。
#   - 命中后印刻的是完整红+黄图案, 所以输出里会出现边框上没有直接标出的黄色辅色。
#
# complexity_calibration:
#   - bbox 高宽限制在 2..4, 让“段长 = 尺寸”肉眼可读, 同时边框仍有足够留白。
#   - 每题 2-3 个 mover, 0-1 个 distractor, 保持和原题同级复杂度。
#   - 每个样例强制至少一个 mover 拥有“同主色但错长度”的行段和列段, 保证只看颜色会出错。
#   - 所有精灵共享一个辅色, 但主色互不相同, 这样人类能稳定看出“多数色决定去向, 辅色随图案复制”。
#
# implicit_assumptions_review:
#   - 原 solver 按颜色分连通块, 在双色精灵下会把一个精灵拆成两个假对象; 这里必须对左条全部非背景像素统一做 8 连通标记。
#   - 原 solver 逐格扫描边框并把每个同色格都当 marker; 这里必须扫描镜像最大同色段并取段中心。
#   - 原 solver 以单个颜色作为唯一 key; 融合后必须按主色 + bbox 尺寸联合筛选。
#   - 原 solver 印刻时对所有像素写同一颜色; 融合后必须保留精灵内部每个像素原来的双色值。
#   - 原 generator 可以把精灵上下贴着堆; 融合后若相邻会并成一个更大的 8 连通组件, 所以必须至少留一空行。
#
# placement_legality:
#   - 行段/列段用占位数组放置, 段之间至少留 1 格 frame_base, 避免视觉合并成更长段。
#   - 有效段中心只在精灵按 bbox 居中后能完整落入框内的位置采样。
#   - 左条精灵按顺序堆叠, 相邻精灵至少隔 1 个背景行, 且都完全落在 divider 左侧。
#
# visual_observability:
#   - 人类能同时看到两个新信号:
#       1. 输出把双色图案整体复制进框内, 说明匹配色不是“全部颜色”, 而是某个主导颜色。
#       2. 同主色但不同段长的边框段并不会都生效, 说明还要比较尺寸。
#   - 由于 generator 强制存在错长度同色段, 若只按颜色规则解释, 会预测出额外复制位置, 与样例不符。
#
# cleanliness_and_inferability:
#   - 框内保持纯背景, 没有额外噪声; 只有复制结果进入框内, 视觉层次清晰。
#   - 颜色种类受控: 背景/分隔线/框底色/共享辅色/若干主色。不存在依赖颜色编号大小的隐藏规则。
#   - 变化规律在 2-3 组样例中可重复观察: “先找双色精灵的多数色, 再用多数色 + bbox 尺寸匹配镜像段长度。”
#
# new_solver_assumptions:
#   - 左条中的每个精灵都是独立 8 连通组件, 且恰有一个严格多数主色。
#   - 框上所有需要识别的 marker 都表现为镜像最大同色段, 相邻段不会接触。
#   - generate_input() 对这些假设逐项构造并用求解器/退化基线差异做验证。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _segment_extents(length: int) -> tuple[int, int]:
    upper = int((int(length) - 1) // 2)
    lower = int(length - 1 - upper)
    return upper, lower


def _segment_bounds(center: int, length: int) -> tuple[int, int]:
    upper, lower = _segment_extents(int(length))
    return int(center - upper), int(center + lower)


def _dominant_color(vals: list[int]) -> int:
    counts = Counter(map(int, vals))
    return sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[0][0]


def _find_divider_mask(g: np.ndarray, bg: int, fc0: int) -> np.ndarray:
    H, W = g.shape
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True
    return divider_mask


def _extract_sprites(g: np.ndarray, bg: int, fc0: int, divider_mask: np.ndarray) -> list[dict]:
    left_mask = np.arange(g.shape[1])[None, :] < int(fc0)
    sprite_mask = (g != int(bg)) & left_mask & (~divider_mask)
    slab, sn = label_mask(sprite_mask, connectivity=8)

    sprites: list[dict] = []
    for sid in range(1, int(sn) + 1):
        pts = np.argwhere(slab == int(sid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        comp = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
        patch = np.where(comp, g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], int(bg))
        colors = [int(v) for v in patch[comp]]
        if not colors:
            continue
        sprites.append(
            {
                "main_color": int(_dominant_color(colors)),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "pattern": patch.astype(int),
            }
        )
    return sprites


def _scan_mirrored_row_segments(
    g: np.ndarray,
    fc0: int,
    fc1: int,
    ir0: int,
    ir1: int,
    allowed_colors: set[int],
) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    r = int(ir0)
    while r <= int(ir1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl in allowed_colors:
            r0 = int(r)
            while r + 1 <= int(ir1) and int(g[int(r + 1), int(fc0)]) == cl and int(g[int(r + 1), int(fc1)]) == cl:
                r += 1
            r1 = int(r)
            segments.append((int(cl), int(r1 - r0 + 1), int((r0 + r1) // 2)))
        r += 1
    return segments


def _scan_mirrored_col_segments(
    g: np.ndarray,
    fr0: int,
    fr1: int,
    ic0: int,
    ic1: int,
    allowed_colors: set[int],
) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    c = int(ic0)
    while c <= int(ic1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct in allowed_colors:
            c0 = int(c)
            while c + 1 <= int(ic1) and int(g[int(fr0), int(c + 1)]) == ct and int(g[int(fr1), int(c + 1)]) == ct:
                c += 1
            c1 = int(c)
            segments.append((int(ct), int(c1 - c0 + 1), int((c0 + c1) // 2)))
        c += 1
    return segments


def _solve(grid: np.ndarray, *, require_size: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = _find_divider_mask(g, int(bg), int(fc0))
    sprites = _extract_sprites(g, int(bg), int(fc0), divider_mask)
    move_colors = {int(s["main_color"]) for s in sprites}

    row_segments = _scan_mirrored_row_segments(g, int(fc0), int(fc1), int(ir0), int(ir1), move_colors)
    col_segments = _scan_mirrored_col_segments(g, int(fr0), int(fr1), int(ic0), int(ic1), move_colors)

    movers: list[dict] = []
    for s in sprites:
        rows = sorted(
            {
                int(center)
                for color, length, center in row_segments
                if int(color) == int(s["main_color"]) and (not require_size or int(length) == int(s["height"]))
            }
        )
        cols = sorted(
            {
                int(center)
                for color, length, center in col_segments
                if int(color) == int(s["main_color"]) and (not require_size or int(length) == int(s["width"]))
            }
        )
        if rows and cols:
            item = dict(s)
            item["targets"] = [(int(r), int(c)) for r in rows for c in cols]
            movers.append(item)

    out = g.copy()
    for s in movers:
        pat = np.array(s["pattern"], dtype=int)
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(pat != int(bg)):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        pat = np.array(s["pattern"], dtype=int)
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(rr), int(cc2), int(pat[int(rr), int(cc2)])) for rr, cc2 in np.argwhere(pat != int(bg))]

        for tr, tc in s["targets"]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for rr0, cc0, color in pix:
                rr = int(s["r0"]) + int(rr0) + int(dr)
                cc3 = int(s["c0"]) + int(cc0) + int(dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Extract each left-strip sprite as one two-color 8-connected component.
    - Use its dominant color as the matching color.
    - Match mirrored edge segments by both dominant color and bbox dimension.
    - Stamp the full two-color sprite at the Cartesian product of matching row/col segment centers.
    """
    return _solve(grid, require_size=True)


def _make_shape_mask(max_h: int, max_w: int) -> np.ndarray | None:
    max_h = min(int(max_h), 4)
    max_w = min(int(max_w), 4)
    if int(max_h) < 2 or int(max_w) < 2:
        return None

    sh = random.choice([v for v in [2, 3, 3, 4] if v <= int(max_h)])
    sw = random.choice([v for v in [2, 3, 3, 4] if v <= int(max_w)])

    mask = np.zeros((int(sh), int(sw)), dtype=bool)
    pivot_r = random.randrange(int(sh))
    pivot_c = random.randrange(int(sw))
    mask[int(pivot_r), :] = True
    mask[:, int(pivot_c)] = True

    extras: list[tuple[int, int]] = []
    for r in range(int(sh)):
        for c in range(int(sw)):
            if bool(mask[int(r), int(c)]):
                continue
            if any(
                0 <= int(r + dr) < int(sh) and 0 <= int(c + dc) < int(sw) and bool(mask[int(r + dr), int(c + dc)])
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            ):
                extras.append((int(r), int(c)))
    random.shuffle(extras)
    for r, c in extras[: random.randint(0, min(2, len(extras)))]:
        mask[int(r), int(c)] = True
    return mask


def _sample_connected_subset(coords: list[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    start = random.choice(coords)
    chosen = {start}
    frontier = [start]
    while len(chosen) < int(k):
        base = random.choice(frontier)
        nbrs = [
            (nr, nc)
            for nr in range(base[0] - 1, base[0] + 2)
            for nc in range(base[1] - 1, base[1] + 2)
            if (nr, nc) in coords and (nr, nc) not in chosen and not (nr == base[0] and nc == base[1])
        ]
        if nbrs:
            nxt = random.choice(nbrs)
            chosen.add(nxt)
            frontier.append(nxt)
            continue
        frontier.append(random.choice(list(chosen)))
        if len(frontier) > len(coords) * 4:
            break
    if len(chosen) < int(k):
        remaining = [p for p in coords if p not in chosen]
        random.shuffle(remaining)
        for p in remaining:
            if any(max(abs(int(p[0]) - int(q[0])), abs(int(p[1]) - int(q[1]))) <= 1 for q in chosen):
                chosen.add(p)
                if len(chosen) >= int(k):
                    break
    return chosen


def _make_multicolor_sprite(primary: int, accent: int, max_h: int, max_w: int) -> np.ndarray | None:
    mask = _make_shape_mask(int(max_h), int(max_w))
    if mask is None:
        return None
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    area = len(coords)
    if area < 3:
        return None

    max_accent = min(3, (int(area) - 1) // 2)
    if max_accent < 1:
        return None
    min_accent = 2 if int(area) >= 5 and max_accent >= 2 else 1
    n_accent = random.randint(int(min_accent), int(max_accent))
    accent_cells = _sample_connected_subset(coords, int(n_accent))

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        sprite[int(r), int(c)] = int(accent if (r, c) in accent_cells else primary)
    return sprite


def _can_place_interval(start: int, end: int, occupied: np.ndarray, lo: int, hi: int) -> bool:
    if int(start) < int(lo) or int(end) > int(hi):
        return False
    pad_start = max(int(lo), int(start - 1))
    pad_end = min(int(hi), int(end + 1))
    for pos in range(int(pad_start), int(pad_end) + 1):
        if bool(occupied[int(pos - lo)]):
            return False
    return True


def _occupy_interval(start: int, end: int, occupied: np.ndarray, lo: int) -> None:
    occupied[int(start - lo) : int(end - lo) + 1] = True


def _place_interval_center(
    length: int,
    occupied: np.ndarray,
    lo: int,
    hi: int,
    *,
    center_lo: int | None = None,
    center_hi: int | None = None,
) -> int | None:
    upper, lower = _segment_extents(int(length))
    low = int(lo + upper if center_lo is None else max(int(center_lo), int(lo + upper)))
    high = int(hi - lower if center_hi is None else min(int(center_hi), int(hi - lower)))
    if low > high:
        return None
    candidates = list(range(int(low), int(high) + 1))
    random.shuffle(candidates)
    for center in candidates:
        start, end = _segment_bounds(int(center), int(length))
        if _can_place_interval(int(start), int(end), occupied, int(lo), int(hi)):
            _occupy_interval(int(start), int(end), occupied, int(lo))
            return int(center)
    return None


def _sample_gaps(total_extra: int, n_slots: int) -> list[int]:
    gaps = [0] * int(n_slots)
    for _ in range(int(total_extra)):
        gaps[random.randrange(int(n_slots))] += 1
    return gaps


def _variant_secondary_footprint(inp: np.ndarray, out: np.ndarray, accent_color: int) -> int:
    return int(np.count_nonzero((out == int(accent_color)) & (inp != int(accent_color))))


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None

    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 1])
    n_total = int(n_movers + n_distractors)

    idx = 0
    divider_color = int(all_colors[idx])
    idx += 1
    frame_base = int(all_colors[idx])
    idx += 1
    accent_color = int(all_colors[idx])
    idx += 1
    primary_colors = [int(v) for v in all_colors[idx : idx + n_total]]
    mover_colors = primary_colors[: int(n_movers)]
    distractor_colors = primary_colors[int(n_movers) :]

    max_sprite_h = min(4, max(2, (int(H) - max(0, int(n_total) - 1)) // max(1, int(n_total))))
    max_sprite_w = min(4, int(sprite_area_w))
    if int(max_sprite_h) < 2 or int(max_sprite_w) < 2:
        return None

    mover_sprites: list[np.ndarray] = []
    distractor_sprites: list[np.ndarray] = []
    sprite_dims: dict[int, tuple[int, int]] = {}

    for mc in mover_colors:
        sp = _make_multicolor_sprite(int(mc), int(accent_color), int(max_sprite_h), int(max_sprite_w))
        if sp is None:
            return None
        mover_sprites.append(sp)
        sprite_dims[int(mc)] = (int(sp.shape[0]), int(sp.shape[1]))

    for dc in distractor_colors:
        sp = _make_multicolor_sprite(int(dc), int(accent_color), int(max_sprite_h), int(max_sprite_w))
        if sp is None:
            return None
        distractor_sprites.append(sp)

    all_item_sprites = mover_sprites + distractor_sprites
    min_needed_h = sum(int(sp.shape[0]) for sp in all_item_sprites) + max(0, int(n_total) - 1)
    if int(min_needed_h) > int(H):
        return None

    row_valid_counts = {int(mc): 1 for mc in mover_colors}
    col_valid_counts = {int(mc): 1 for mc in mover_colors}
    if random.random() < 0.4:
        row_valid_counts[int(random.choice(mover_colors))] += 1
    if random.random() < 0.4:
        col_valid_counts[int(random.choice(mover_colors))] += 1

    row_centers: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}
    row_lengths: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}
    col_centers: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}
    col_lengths: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}

    row_occupied = np.zeros(int(interior_h), dtype=bool)
    col_occupied = np.zeros(int(interior_w), dtype=bool)

    row_specs: list[tuple[int, int]] = []
    for mc in mover_colors:
        sh, _sw = sprite_dims[int(mc)]
        for _ in range(int(row_valid_counts[int(mc)])):
            row_specs.append((int(mc), int(sh)))
    row_specs.sort(key=lambda x: (-int(x[1]), random.random()))
    for mc, sh in row_specs:
        center = _place_interval_center(int(sh), row_occupied, int(ir0), int(ir1))
        if center is None:
            return None
        row_centers[int(mc)].append(int(center))
        row_lengths[int(mc)].append(int(sh))

    col_specs: list[tuple[int, int]] = []
    for mc in mover_colors:
        _sh, sw = sprite_dims[int(mc)]
        for _ in range(int(col_valid_counts[int(mc)])):
            col_specs.append((int(mc), int(sw)))
    col_specs.sort(key=lambda x: (-int(x[1]), random.random()))
    for mc, sw in col_specs:
        center = _place_interval_center(int(sw), col_occupied, int(ic0), int(ic1))
        if center is None:
            return None
        col_centers[int(mc)].append(int(center))
        col_lengths[int(mc)].append(int(sw))

    ambiguous_color = int(random.choice(mover_colors))
    good_h, good_w = sprite_dims[int(ambiguous_color)]
    bad_row_lengths = [int(v) for v in [2, 3, 4] if int(v) <= int(interior_h) and int(v) != int(good_h)]
    bad_col_lengths = [int(v) for v in [2, 3, 4] if int(v) <= int(interior_w) and int(v) != int(good_w)]
    if not bad_row_lengths or not bad_col_lengths:
        return None

    bad_row_len = int(random.choice(bad_row_lengths))
    bad_col_len = int(random.choice(bad_col_lengths))
    bad_row_center = _place_interval_center(int(bad_row_len), row_occupied, int(ir0), int(ir1))
    bad_col_center = _place_interval_center(int(bad_col_len), col_occupied, int(ic0), int(ic1))
    if bad_row_center is None or bad_col_center is None:
        return None
    row_centers[int(ambiguous_color)].append(int(bad_row_center))
    row_lengths[int(ambiguous_color)].append(int(bad_row_len))
    col_centers[int(ambiguous_color)].append(int(bad_col_center))
    col_lengths[int(ambiguous_color)].append(int(bad_col_len))

    inp = np.full((H, W), int(bg), dtype=int)
    inp[:, int(div_col)] = int(divider_color)
    inp[0, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[H - 1, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[1 : H - 1, int(fc0)] = int(frame_base)
    inp[1 : H - 1, int(fc1)] = int(frame_base)

    for mc in mover_colors:
        for center, length in zip(row_centers[int(mc)], row_lengths[int(mc)]):
            r0, r1 = _segment_bounds(int(center), int(length))
            inp[int(r0) : int(r1) + 1, int(fc0)] = int(mc)
            inp[int(r0) : int(r1) + 1, int(fc1)] = int(mc)
        for center, length in zip(col_centers[int(mc)], col_lengths[int(mc)]):
            c0, c1 = _segment_bounds(int(center), int(length))
            inp[0, int(c0) : int(c1) + 1] = int(mc)
            inp[H - 1, int(c0) : int(c1) + 1] = int(mc)

    all_items: list[tuple[int, np.ndarray, bool]] = []
    for mc, sp in zip(mover_colors, mover_sprites):
        all_items.append((int(mc), sp, False))
    for dc, sp in zip(distractor_colors, distractor_sprites):
        all_items.append((int(dc), sp, True))

    extra_rows = int(H - min_needed_h)
    gap_extras = _sample_gaps(int(extra_rows), len(all_items) + 1)
    cur_row = int(gap_extras[0])
    for idx_item, (_main, sp, _is_dist) in enumerate(all_items):
        sh, sw = sp.shape
        if int(sw) > int(sprite_area_w) or int(cur_row + sh) > int(H):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        region = inp[int(cur_row) : int(cur_row + sh), int(c_off) : int(c_off + sw)]
        inp[int(cur_row) : int(cur_row + sh), int(c_off) : int(c_off + sw)] = np.where(sp != int(bg), sp, region)
        cur_row += int(sh)
        if idx_item < len(all_items) - 1:
            cur_row += 1 + int(gap_extras[idx_item + 1])

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        fused_out = main(inp)
        color_only_out = _solve(inp, require_size=False)
    except Exception:
        return None

    if np.array_equal(inp, fused_out):
        return None
    if np.array_equal(fused_out, color_only_out):
        return None
    if _variant_secondary_footprint(inp, fused_out, int(accent_color)) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
