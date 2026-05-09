# Combo variant of 247ef758 (index 13)
# combo_spec: D10/2+D14/0
# Name: 条纹框架+伪同色标记干扰
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    原题:
#      - 框边大多是单一基色。
#      - 某行若左右边缘同为某个 sprite 颜色, 这一行就是该颜色的 row mark。
#      - 某列若上下边缘同为某个 sprite 颜色, 这一列就是 col mark。
#    D14/0 单变体:
#      - 框边不再有“单一基色”, 而是两种 mover 颜色交替条纹。
#      - 真标记不是“特别颜色”, 而是“打断条纹规律的异常点”。
#      - 例如纵边本应是 A,B,A,B,A,B..., 第 5 行却出现 A,A,A, 这个中间位置就是 A 的标记。
#    本融合版再叠加 D10/2:
#      - 仍然使用双 mover 条纹框架。
#      - 但除了真实 row/col 标记外, 还额外制造“单侧异常”:
#          左边缘某一行被改成 A, 右边缘保持正常条纹;
#          或上边缘某一列被改成 B, 下边缘保持正常条纹。
#      - 这些单侧异常与真实标记使用完全相同的颜色集合, 肉眼上也同样是“条纹断点”,
#        所以解题者必须先找异常候选, 再验证异常是否在对边同位置成对出现。
#    一个具体小例子:
#      - 左右边期望条纹(仅看 interior rows)为 B,A,B,A,B,A。
#      - 第 2 行左右都改成 B: 这是 B 的真 row mark。
#      - 第 5 行只有左边改成 B, 右边仍是 A: 这是 B 的伪 row mark。
#      - 原题逻辑会把“同色即标记”推广到所有条纹位置, 完全失效。
#      - 仅有 D14/0 的逻辑会找到所有异常候选, 但若不检查双侧同步, 会把第 5 行误当真标记。
#      - 融合后正确逻辑:
#          1. 恢复左右/上下的最佳二色交替相位。
#          2. 分别找左、右、上、下各自偏离条纹的位置。
#          3. 只有同一行在左右两侧都异常且观测色相同, 才是真 row mark。
#             列标记同理。
#
# b) 复杂度校准
#    - 保持恰好 2 个 mover。这样条纹模式简单、稳定、重复出现, 人类可从少量样例恢复规律。
#    - 每个 mover 仍有 1-2 个真 row mark 和 1-2 个真 col mark, 保留原题的笛卡尔积印刻效果。
#    - 伪标记数量保持少量: 每个样例至少 1 个伪 row 异常和 1 个伪 col 异常, 最多各 2 个。
#      这样能清楚体现“需要双侧验证”, 又不会把边框弄得过于嘈杂。
#    - distractor sprite 仍最多 1 个, 避免左侧区域杂乱。
#
# c) 隐式假设审查
#    - largest_enclosed_component_bbox(...):
#        依赖“框仍围出一块不触边的大背景区域”。本设计保持闭合矩形框, 成立。
#    - divider 检测:
#        依赖“左侧存在一整列近乎满高的统一颜色”。分隔列保持原样, 成立。
#    - 左侧 sprite 提取:
#        依赖“只在 frame 左侧列中找连通块”。即使 mover 颜色也出现在框边, 该提取仍安全。
#    - 原 solver 中:
#          if cl == cr and cl in move_colors
#      这个条件在融合后彻底失效:
#        1. 普通条纹位置左右本来就相等且属于 mover 颜色。
#        2. 单侧伪异常必须先以“各边分别对期望条纹的偏离”被检出,
#           再做左右/上下对应位置的交集验证。
#    - 新 solver 新增假设:
#        1. 框边只使用 2 种非背景颜色, 即两种 mover 颜色。
#        2. 真标记始终在对边两侧同步出现, 且观测色相同。
#        3. 伪标记只出现在单侧, 不会恰好在对边同位置也被改坏。
#        4. 异常是单点替换, 条纹总体相位仍能通过最小失配恢复。
#      generator 会显式构造并保证这些条件。
#
# d) 放置合法性
#    - 真 mark 的行列选择继续按 sprite bbox 约束筛选, 保证每个印刻中心都能容纳该 sprite。
#    - 不同 mover 不共享真 row/col mark, 避免目标中心重合导致规则歧义。
#    - 伪 mark 只放在未占用的边缘 interior 位置, 不影响真 mark 的一一对应。
#    - 所有边缘改写都只发生在框 perimeter, 不与左侧 sprite 或框内部重叠。
#
# e) 视觉可观察性
#    - 人类先能看到“整个框边几乎是规则 ABAB 条纹”。
#    - 再能看到少量局部断点; 其中有些在左右/上下同时出现, 有些只在单侧出现。
#    - 通过 2-4 组样例可观察到: 只有成对断点决定 interior 复制位置, 单侧断点不起作用。
#    - 规则依赖可见的对称/异常结构, 不依赖颜色编号或隐藏排序。
#
# f) 可视化整洁度
#    - 伪标记只加在边缘, 不进入 interior, 不会和最终印刻对象混在一起。
#    - 真/伪标记都只是单像素断点, 数量少, 条纹主体仍然清晰。
#    - 左侧 sprite 数量限制为 2 movers + 最多 1 distractor, 视觉层次稳定。
#
# g) 变化的可推断性
#    - 颜色种类受控: frame 只用 2 个 mover 色; divider 1 色; 可选 distractor 1 色。
#    - 颜色不发生隐藏映射。异常像素显示的就是对应 mover 色。
#    - “什么颜色会被复制到哪里”仍由左侧 sprite 直接给出, 推断负担主要落在
#      “哪些异常是真的 mark”这一结构判断上, 与融合目标一致。
#
# 非退化度量
#   - 定义:
#       true_rows / true_cols = 经过双侧验证后得到的真标记数量
#       naive_rows / naive_cols = 只要任一单侧异常就计为候选的数量
#       pseudo_rows = naive_rows - true_rows
#       pseudo_cols = naive_cols - true_cols
#   - 生成器强制 pseudo_rows >= 1 且 pseudo_cols >= 1。
#   - 如果这两个差值经常为 0, 说明样本退化回仅有条纹异常而无“单侧伪标记”的版本。

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
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


def _best_alternating_expectation(seqs: list[list[int]], colors: list[int]) -> list[int]:
    """Return the lower-error 2-color alternation shared by one or more equal-length sequences."""
    assert len(colors) == 2
    length = len(seqs[0])
    best_phase = 0
    best_err = None
    for phase in (0, 1):
        err = 0
        for seq in seqs:
            assert len(seq) == length
            for i, v in enumerate(seq):
                if int(v) != int(colors[(i + phase) % 2]):
                    err += 1
        if best_err is None or err < best_err:
            best_err = err
            best_phase = phase
    return [int(colors[(i + best_phase) % 2]) for i in range(length)]


def _collect_sprites(g: np.ndarray, bg: int, fc0: int) -> list[dict[str, object]]:
    """Extract per-color 8-connected sprites from the left strip, excluding the divider column."""
    H, W = g.shape
    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)
    sprites: list[dict[str, object]] = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            comp = slab[int(r0): int(r1) + 1, int(c0): int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": comp,
                }
            )
    return sprites


def _detect_marks(
    g: np.ndarray,
    move_colors: set[int],
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[str, int]]:
    """
    Detect true marks by:
    1. recovering the best alternating stripe expectation;
    2. detecting per-side anomalies;
    3. keeping only anomalies mirrored on both opposite edges with the same observed color.
    """
    frame_colors = sorted(
        set(map(int, g[int(fr0), int(fc0): int(fc1) + 1]))
        | set(map(int, g[int(fr1), int(fc0): int(fc1) + 1]))
        | set(map(int, g[int(fr0): int(fr1) + 1, int(fc0)]))
        | set(map(int, g[int(fr0): int(fr1) + 1, int(fc1)]))
    )
    assert len(frame_colors) == 2

    left_seq = [int(g[int(r), int(fc0)]) for r in range(int(ir0), int(ir1) + 1)]
    right_seq = [int(g[int(r), int(fc1)]) for r in range(int(ir0), int(ir1) + 1)]
    top_seq = [int(g[int(fr0), int(c)]) for c in range(int(ic0), int(ic1) + 1)]
    bottom_seq = [int(g[int(fr1), int(c)]) for c in range(int(ic0), int(ic1) + 1)]

    expected_rows = _best_alternating_expectation([left_seq, right_seq], frame_colors)
    expected_cols = _best_alternating_expectation([top_seq, bottom_seq], frame_colors)

    left_row_anom: dict[int, int] = {}
    right_row_anom: dict[int, int] = {}
    top_col_anom: dict[int, int] = {}
    bottom_col_anom: dict[int, int] = {}

    for offset, r in enumerate(range(int(ir0), int(ir1) + 1)):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl != int(expected_rows[int(offset)]) and cl in move_colors:
            left_row_anom[int(r)] = int(cl)
        if cr != int(expected_rows[int(offset)]) and cr in move_colors:
            right_row_anom[int(r)] = int(cr)

    for offset, c in enumerate(range(int(ic0), int(ic1) + 1)):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct != int(expected_cols[int(offset)]) and ct in move_colors:
            top_col_anom[int(c)] = int(ct)
        if cb != int(expected_cols[int(offset)]) and cb in move_colors:
            bottom_col_anom[int(c)] = int(cb)

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    true_row_count = 0
    true_col_count = 0

    for r in sorted(set(left_row_anom) & set(right_row_anom)):
        col = int(left_row_anom[int(r)])
        if col == int(right_row_anom[int(r)]) and col in move_colors:
            row_marks[int(col)].append(int(r))
            true_row_count += 1

    for c in sorted(set(top_col_anom) & set(bottom_col_anom)):
        col = int(top_col_anom[int(c)])
        if col == int(bottom_col_anom[int(c)]) and col in move_colors:
            col_marks[int(col)].append(int(c))
            true_col_count += 1

    metrics = {
        "true_rows": int(true_row_count),
        "true_cols": int(true_col_count),
        "naive_rows": int(len(set(left_row_anom) | set(right_row_anom))),
        "naive_cols": int(len(set(top_col_anom) | set(bottom_col_anom))),
    }
    metrics["pseudo_rows"] = int(metrics["naive_rows"] - metrics["true_rows"])
    metrics["pseudo_cols"] = int(metrics["naive_cols"] - metrics["true_cols"])
    return row_marks, col_marks, metrics


def _scene_and_targets(grid: np.ndarray) -> tuple[np.ndarray, int, tuple[int, int, int, int], list[dict[str, object]], dict[int, list[tuple[int, int]]], dict[str, int]]:
    """Shared parsing for solver and validation helpers."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    sprites = _collect_sprites(g, bg, fc0)
    move_colors = {int(s["color"]) for s in sprites}

    row_marks, col_marks, metrics = _detect_marks(
        g, move_colors, ir0, ic0, ir1, ic1, fr0, fc0, fr1, fc1
    )

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(set(row_marks) & set(col_marks)):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        if pts:
            targets[int(col)] = sorted(pts)

    return g, bg, (ir0, ic0, ir1, ic1), sprites, targets, metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at validated stripe-break targets.

    Fusion rule:
    - The frame perimeter is an alternating 2-color stripe built from the two mover colors.
    - True marks are positions where the stripe is broken on both opposite edges with the same color.
    - Single-edge breaks are pseudo marks and must be ignored.
    - Each mover is stamped at the Cartesian product of its validated row/col marks.
    """
    g, bg, inner_bbox, sprites, targets, _metrics = _scene_and_targets(grid)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Public helper for non-degeneracy checks in tests."""
    _g, _bg, _bbox, _sprites, _targets, metrics = _scene_and_targets(grid)
    return dict(metrics)


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _base_stripe_color(colors: list[int], pos: int, phase: int) -> int:
    return int(colors[(int(pos) + int(phase)) % 2])


def _choose_pseudo_positions(
    candidates: list[int],
    occupied: set[int],
    forced_count: int,
    max_count: int,
) -> list[int] | None:
    usable = [v for v in candidates if v not in occupied]
    if len(usable) < forced_count:
        return None
    n = random.randint(forced_count, min(max_count, len(usable)))
    random.shuffle(usable)
    return sorted(usable[:n])


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
    n_movers = 2
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx: idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx: idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        sp = _make_sprite(mc, 5, sprite_area_w)
        if sp is None:
            return None
        sprites.append(sp)

    row_phase = random.randint(0, 1)
    col_phase = random.randint(0, 1)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = ir0 + sh // 2
        max_row = ir1 - (sh - 1 - sh // 2)
        min_col = ic0 + sw // 2
        max_col = ic1 - (sw - 1 - sw // 2)
        if min_row > max_row or min_col > max_col:
            return None

        avail_rows = [
            r
            for r in range(min_row, max_row + 1)
            if r not in occupied_rows and _base_stripe_color(mover_colors, r, row_phase) != mc
        ]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), sh)
        if not row_marks:
            return None

        avail_cols = [
            c
            for c in range(min_col, max_col + 1)
            if c not in occupied_cols and _base_stripe_color(mover_colors, c - fc0, col_phase) != mc
        ]
        col_marks = _pick_spaced(avail_cols, random.randint(1, 2), sw)
        if not col_marks:
            return None

        mover_targets[int(mc)] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(map(int, row_marks))
        occupied_cols.update(map(int, col_marks))

    row_candidates = [r for r in range(ir0, ir1 + 1)]
    col_candidates = [c for c in range(ic0, ic1 + 1)]

    pseudo_rows = _choose_pseudo_positions(row_candidates, occupied_rows, forced_count=1, max_count=2)
    pseudo_cols = _choose_pseudo_positions(col_candidates, occupied_cols, forced_count=1, max_count=2)
    if pseudo_rows is None or pseudo_cols is None:
        return None

    pseudo_row_sides = {int(r): random.choice(["L", "R"]) for r in pseudo_rows}
    pseudo_col_sides = {int(c): random.choice(["T", "B"]) for c in pseudo_cols}

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.array([_base_stripe_color(mover_colors, i, col_phase) for i in range(frame_w)], dtype=int)
    bottom_edge = top_edge.copy()
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
            bottom_edge[int(cc - fc0)] = int(mc)
    for cc in pseudo_cols:
        idx_edge = int(cc - fc0)
        expected = int(top_edge[idx_edge])
        pseudo_color = int(mover_colors[0] if mover_colors[1] == expected else mover_colors[1])
        if pseudo_col_sides[int(cc)] == "T":
            top_edge[idx_edge] = int(pseudo_color)
        else:
            bottom_edge[idx_edge] = int(pseudo_color)
    inp[0, fc0: fc0 + frame_w] = top_edge
    inp[H - 1, fc0: fc0 + frame_w] = bottom_edge

    left_edge = np.array([_base_stripe_color(mover_colors, r, row_phase) for r in range(H)], dtype=int)
    right_edge = left_edge.copy()
    for mc, (rmarks, _cmarks) in mover_targets.items():
        for rr in rmarks:
            left_edge[int(rr)] = int(mc)
            right_edge[int(rr)] = int(mc)
    for rr in pseudo_rows:
        expected = int(left_edge[int(rr)])
        pseudo_color = int(mover_colors[0] if mover_colors[1] == expected else mover_colors[1])
        if pseudo_row_sides[int(rr)] == "L":
            left_edge[int(rr)] = int(pseudo_color)
        else:
            right_edge[int(rr)] = int(pseudo_color)
    for r in range(H):
        inp[int(r), fc0] = int(left_edge[int(r)])
        inp[int(r), fc1] = int(right_edge[int(r)])

    cur_row = 0
    all_items: list[np.ndarray] = list(sprites)
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append(ds)

    for sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = sp[dr, dc]
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
        metrics = fusion_metrics(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    if int(metrics["pseudo_rows"]) < 1 or int(metrics["pseudo_cols"]) < 1:
        return None
    if int(metrics["true_rows"]) < 2 or int(metrics["true_cols"]) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
