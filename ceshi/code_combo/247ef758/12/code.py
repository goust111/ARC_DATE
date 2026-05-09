# Combo variant of 247ef758 (index 12)
# combo_spec: D12/0+D13/0
# Name: 左上对齐+环绕溢出
#
# Design analysis
# a) 具体化与行为追踪
# - 原题仍是三段布局: 左侧精灵条带 | 整列分隔线 | 右侧矩形框。颜色匹配关系不变: 左右框边同色给出目标行，
#   上下框边同色给出目标列，笛卡尔积交点就是该颜色精灵的复制目标。
# - 原题在每个 `(tr, tc)` 上执行“中心对齐 + 越界裁剪”。例如某个 3x4 精灵若目标点在 `(8, 12)`，
#   原题会把精灵外接框中心对齐到该点，所以像素相对目标既有负偏移也有正偏移；落出 interior 的像素直接丢弃。
# - 本融合变体改成“左上角对齐 + 环绕”。同一个 3x4 精灵在 `(8, 12)` 时，左上角直接放在 `(8, 12)`，
#   所有像素偏移都变成非负，因此精灵整体朝右下展开；若某些像素跑到 bottom/right 外，则不裁剪，而是按
#   interior 的高宽取模，从 top/left 重新进入。
# - 具体小例子: interior 行列为 `1..8`，某 3x3 L 形精灵像素相对左上角为
#   `{(0,0),(1,0),(2,0),(2,1),(2,2)}`，目标点取 `(7,7)`。
#   原题(中心对齐+裁剪): 会围绕 `(7,7)` 展开，超出边界的像素直接消失。
#   融合后(左上对齐+环绕): 原始坐标变成 `(7,7),(8,7),(9,7),(9,8),(9,9)`，其中 row=9 和 col=9 越界，
#   会回卷成 top/left 侧，落到 `(1,7),(1,8),(1,1)` 等位置。视觉上会看到目标交点像是“精灵左上角”，
#   同时碎片从上边和左边冒出来。
#
# b) 复杂度校准
# - 颜色语义完全沿用原题，不新增颜色映射，避免配对歧义。
# - 仍保留 2-3 个 mover 与 0-1 个 distractor，且每个 mover 仍有 1-2 个行标记和 1-2 个列标记；
#   这样同一规则会在一个样例里重复多次，人类可从重复中确认“交点复制”仍成立。
# - 生成器强制至少一个 mover 触发左上锚点下的 bottom/right 溢出，并要求融合输出同时不同于
#   “仅左上对齐+裁剪”以及“仅中心对齐+环绕”，避免退化成单一维度变体。
#
# c) 隐式假设审查
# - 原 solver 中 `cr/cc = bbox center`、`dr/dc = target - center` 依赖“中心对齐”前提；融合后该前提失效，
#   必须改成“目标点就是 bbox 左上角”。
# - 原 solver 末尾 `if ir0 <= rr <= ir1 and ic0 <= cc <= ic1` 依赖“越界像素应丢弃”前提；融合后该前提失效，
#   必须改成相对 interior 原点的模运算。
# - 两项叠加后会产生新的方向性: 因为左上锚点下所有像素偏移都非负，wrap 不再来自 top/left 溢出，
#   而主要来自 bottom/right 溢出并重返 top/left。生成器必须用靠下/靠右的标记去显式触发这一点。
# - 框 interior 识别、divider 识别、左侧单色 8 连通精灵提取、按输入垂直位置决定覆盖顺序，这些假设在融合后仍成立。
#
# d) 放置合法性
# - 输入阶段左侧精灵仍只放在分隔线左边，互不重叠、不越界。
# - 行标记、列标记仍按颜色镜像放在框对边，且不同 mover 不共享同一标记行/列，保持目标归属清晰。
# - 由于输出允许环绕，生成器不再要求 target 能完整容纳精灵；但它仍约束候选标记组合，使大多数 mover 不发生
#   过多 wrap，从而减少输出内部的混乱叠层。
#
# e) 视觉可观察性
# - 左上锚点使标记交点总落在精灵的左上角，而不是中心；只要外接框不为 1x1，这个偏移就可见。
# - 环绕会把右下越界像素重新画到左侧和上侧，形成原题没有的“跨边界续接”证据。
# - 生成器要求样本至少有一个 wrapped target，且融合输出相对两个单变体基线都要有可见差异，保证 2-4 个样例内
#   能同时观察到“锚点改变”和“环绕出现”。
#
# f) 可视化整洁度
# - 除一个强制触发融合行为的 mover 外，其余 mover 优先选择不 wrap 的标记组合，降低 interior 内部的碎片密度。
# - 同一 mover 的多行/多列标记仍按自身高宽做最小间隔筛选，避免副本彼此严重重叠。
# - 覆盖顺序继续保持“输入中更低者先印、更高者后盖”，解释方式与原题一致，不额外引入排序线索。
#
# g) 变化的可推断性
# - 颜色不发生重映射；颜色只负责“哪个精灵对应哪组边框标记”。变化完全体现在空间规则上，因此人类只需推断
#   “交点复制”的锚点和越界行为，不需要额外学习颜色变换表。
#
# h) main() 的新隐式假设与 generator 对应保证
# - `main()` 新假设 target 可以直接作为精灵左上角，且任何越界都应在 interior 上取模；`generate_input()`
#   通过显式筛选 wrap / non-wrap 标记组合来保证这些情况都可解析。
# - `main()` 仍假设 interior 是最大的不接触外边界背景连通块，divider 是左侧近乎整列单色非背景列，
#   左侧精灵是按颜色可分离的 8 连通块；生成器继续构造满足这些前提的输入。

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import random

import numpy as np

from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component that does not touch the border."""
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
        if bool(
            (rs == 0).any()
            or (rs == int(H - 1)).any()
            or (cs == 0).any()
            or (cs == int(W - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _sprite_offsets(mask: np.ndarray, *, anchor_mode: str) -> list[tuple[int, int]]:
    if anchor_mode == "center":
        h, w = mask.shape
        anchor_r = int((h - 1) // 2)
        anchor_c = int((w - 1) // 2)
    elif anchor_mode == "topleft":
        anchor_r = 0
        anchor_c = 0
    else:
        raise ValueError(f"unknown anchor_mode: {anchor_mode}")

    return [
        (int(rr - anchor_r), int(cc - anchor_c))
        for rr, cc in np.argwhere(mask)
    ]


def _stamped_cells(
    offsets: list[tuple[int, int]],
    target: tuple[int, int],
    interior: tuple[int, int, int, int],
    *,
    wrap: bool,
) -> list[tuple[int, int]]:
    ir0, ic0, ir1, ic1 = map(int, interior)
    inner_h = int(ir1 - ir0 + 1)
    inner_w = int(ic1 - ic0 + 1)
    tr, tc = map(int, target)
    cells: list[tuple[int, int]] = []

    for dr, dc in offsets:
        raw_r = int(tr + dr)
        raw_c = int(tc + dc)
        if wrap:
            rr = int(ir0 + ((raw_r - ir0) % inner_h))
            cc = int(ic0 + ((raw_c - ic0) % inner_w))
            cells.append((rr, cc))
        elif int(ir0) <= raw_r <= int(ir1) and int(ic0) <= raw_c <= int(ic1):
            cells.append((raw_r, raw_c))

    return cells


def _extract_scene(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    sprites = []
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
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                }
            )

    move_colors = {int(s["color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    return g, bg, (ir0, ic0, ir1, ic1), sprites, targets


def _solve(grid: np.ndarray, *, anchor_mode: str, wrap: bool) -> np.ndarray:
    g, bg, interior, sprites, targets = _extract_scene(grid)
    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        offsets = _sprite_offsets(s["mask"], anchor_mode=anchor_mode)
        for target in targets[int(col)]:
            for rr, cc in _stamped_cells(offsets, target, interior, wrap=wrap):
                out[int(rr), int(cc)] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    - read the same marker-defined row/column intersections as the original task
    - erase movers from the left strip
    - stamp each mover with its bbox top-left corner anchored at each target
    - pixels that leave the frame interior wrap around toroidally
    - higher sprites still cover lower ones
    """
    return _solve(grid, anchor_mode="topleft", wrap=True)


def _mark_options(candidates: list[int], min_spacing: int) -> list[list[int]]:
    opts: list[list[int]] = []
    vals = sorted(set(map(int, candidates)))
    for n in (1, 2):
        if len(vals) < n:
            continue
        for combo in combinations(vals, n):
            if all(
                abs(int(combo[j]) - int(combo[i])) >= int(min_spacing)
                for i in range(n)
                for j in range(i + 1, n)
            ):
                opts.append([int(v) for v in combo])
    random.shuffle(opts)
    return opts


def _wrap_stats_for_targets(
    offsets: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    interior: tuple[int, int, int, int],
) -> dict[str, int]:
    ir0, ic0, ir1, ic1 = map(int, interior)
    wrap_pixels = 0
    wrap_targets = 0
    vertical_wrap_pixels = 0
    horizontal_wrap_pixels = 0
    both_axis_targets = 0

    for tr, tc in targets:
        target_wrap = False
        target_vertical = False
        target_horizontal = False
        for dr, dc in offsets:
            raw_r = int(tr + dr)
            raw_c = int(tc + dc)
            over_r = raw_r > int(ir1)
            over_c = raw_c > int(ic1)
            if over_r or over_c:
                wrap_pixels += 1
                target_wrap = True
            if over_r:
                vertical_wrap_pixels += 1
                target_vertical = True
            if over_c:
                horizontal_wrap_pixels += 1
                target_horizontal = True
        if target_wrap:
            wrap_targets += 1
        if target_vertical and target_horizontal:
            both_axis_targets += 1

    return {
        "wrap_pixels": int(wrap_pixels),
        "wrap_targets": int(wrap_targets),
        "vertical_wrap_pixels": int(vertical_wrap_pixels),
        "horizontal_wrap_pixels": int(horizontal_wrap_pixels),
        "both_axis_targets": int(both_axis_targets),
    }


def analyze_variant(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the generated sample really exercises the fusion:
    - wrapping must occur under top-left anchoring
    - the combo output must differ from each single-dimension baseline
    """
    g, _bg, interior, sprites, targets = _extract_scene(grid)
    totals = {
        "wrap_pixels": 0,
        "wrap_targets": 0,
        "vertical_wrap_pixels": 0,
        "horizontal_wrap_pixels": 0,
        "both_axis_targets": 0,
    }
    for s in sprites:
        col = int(s["color"])
        if col not in targets:
            continue
        stats = _wrap_stats_for_targets(
            _sprite_offsets(s["mask"], anchor_mode="topleft"),
            targets[col],
            interior,
        )
        for key, value in stats.items():
            totals[key] += int(value)

    combo = _solve(g, anchor_mode="topleft", wrap=True)
    center_clipped = _solve(g, anchor_mode="center", wrap=False)
    topleft_clipped = _solve(g, anchor_mode="topleft", wrap=False)
    center_wrapped = _solve(g, anchor_mode="center", wrap=True)
    totals["diff_vs_original"] = int(np.count_nonzero(combo != center_clipped))
    totals["diff_vs_topleft_clipped"] = int(np.count_nonzero(combo != topleft_clipped))
    totals["diff_vs_center_wrapped"] = int(np.count_nonzero(combo != center_wrapped))
    return totals


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
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
            [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
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


def _choose_target_pair(
    sp: np.ndarray,
    interior: tuple[int, int, int, int],
    avail_rows: list[int],
    avail_cols: list[int],
    *,
    force_wrap: bool,
) -> tuple[list[int], list[int]] | None:
    sh, sw = map(int, sp.shape)
    offsets = _sprite_offsets(sp != 0, anchor_mode="topleft")
    row_options = _mark_options(avail_rows, max(1, sh))
    col_options = _mark_options(avail_cols, max(1, sw))
    if not row_options or not col_options:
        return None

    nonwrap_pairs: list[tuple[list[int], list[int], dict[str, int]]] = []
    wrap_pairs: list[tuple[list[int], list[int], dict[str, int]]] = []
    strong_wrap_pairs: list[tuple[list[int], list[int], dict[str, int]]] = []
    for rows in row_options:
        for cols in col_options:
            targets = [(int(r), int(c)) for r in rows for c in cols]
            stats = _wrap_stats_for_targets(offsets, targets, interior)
            rec = (list(rows), list(cols), stats)
            if int(stats["wrap_targets"]) > 0:
                wrap_pairs.append(rec)
                if int(stats["both_axis_targets"]) > 0 or (
                    int(stats["vertical_wrap_pixels"]) > 0 and int(stats["horizontal_wrap_pixels"]) > 0
                ):
                    strong_wrap_pairs.append(rec)
            else:
                nonwrap_pairs.append(rec)

    if force_wrap:
        pool = strong_wrap_pairs or wrap_pairs
    else:
        if nonwrap_pairs and random.random() < 0.75:
            pool = nonwrap_pairs
        else:
            pool = wrap_pairs or nonwrap_pairs
    if not pool:
        return None

    rows, cols, _stats = random.choice(pool)
    return sorted(rows), sorted(cols)


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
    interior = (ir0, ic0, ir1, ic1)

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, min(5, H), sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    if not any(sp.shape[0] > 1 and sp.shape[1] > 1 for sp in sprites):
        return None

    forced_indices = [i for i, sp in enumerate(sprites) if sp.shape[0] > 1 or sp.shape[1] > 1]
    forced_idx = random.choice(forced_indices)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    for i, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        avail_rows = [r for r in range(ir0, ir1 + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(ic0, ic1 + 1) if c not in occupied_cols]
        choice = _choose_target_pair(
            sp,
            interior,
            avail_rows,
            avail_cols,
            force_wrap=(i == forced_idx),
        )
        if choice is None:
            return None
        rows, cols = choice
        mover_targets[int(mc)] = (rows, cols)
        occupied_rows.update(rows)
        occupied_cols.update(cols)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = top_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc, (rmarks, _) in mover_targets.items():
            if r in rmarks:
                val = mc
                break
        inp[r, fc0] = val
        inp[r, fc1] = val

    cur_row = 0
    all_items: list[tuple[int, np.ndarray]] = [(mc, sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds))

    for color, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(color)
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        combo_output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, combo_output):
        return None

    metrics = analyze_variant(inp)
    if int(metrics["wrap_targets"]) <= 0:
        return None
    if int(metrics["vertical_wrap_pixels"]) + int(metrics["horizontal_wrap_pixels"]) < 2:
        return None
    if int(metrics["diff_vs_topleft_clipped"]) < 2:
        return None
    if int(metrics["diff_vs_center_wrapped"]) < 2:
        return None
    if int(metrics["diff_vs_original"]) < 3:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
