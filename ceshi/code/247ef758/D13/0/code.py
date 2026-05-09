# Variant of 247ef758
# Dimension: D13 — 像素裁剪行为 (meta)
# Change: 精灵印刻时超出框架内部边界的像素被裁剪（不绘制）... → 超出部分环绕到对面

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D13 variant
# -----------------------------------
# spec_variant:
#   unchanged_layout:
#     - 输入仍是: 左侧精灵条带 | 整列分隔线 | 右侧整高矩形框
#     - 行标记依旧读自左右框边相同颜色；列标记依旧读自上下框边相同颜色
#     - mover 仍从左侧擦除，并按“越靠下越先印，越靠上越后印”的顺序覆盖
#   changed_behavior:
#     - 原题在印刻阶段对落到 interior 外的像素直接丢弃
#     - 本变体把 interior 当作环面: 行越上界/下界都按 interior 高度取模，列越左界/右界都按 interior 宽度取模
#     - 因此一个目标可能在框的一侧“断开”，并在对侧继续出现
#   concrete_trace:
#     - 设 interior 行是 1..8，某个 3x3 十字精灵以第 1 行为中心印刻
#     - 原题: 精灵顶部那一行会掉到 row=0，被裁剪掉，只剩 interior 内部的下半部分
#     - 变体: 这几个 row=0 的像素改为回到 row=8；如果列也越界，同理回到另一侧
#     - 差异发生在 solver 的最后 stamping 环节；输入构造和标记读取规则本身不变
#   complexity_calibration:
#     - 若继续使用原生成器的“安全中心区标记”，新旧规则会完全等价，变体不可观察
#     - 因此至少一个 mover 必须拿到靠边 target，使其至少一个像素越界
#     - 同时仍保留 2-3 个 mover、1-2 行标记、1-2 列标记，避免规则退化成只有单次环绕的特例
#   implicit_assumptions_review:
#     - frame interior 的识别仍依赖“最大且不碰外边界的 bg 连通块”；D13 不改变这一点
#     - divider 检测仍依赖“左侧存在近乎整列、单色、非 bg 的分隔列”；D13 不改变这一点
#     - 原 solver 最后一句 `if ir0 <= rr <= ir1 and ic0 <= cc <= ic1` 依赖旧前提“越界像素应丢弃”
#     - 变体下这个前提失效，必须替换成相对 interior 原点的模运算；否则会错误回退成原题
#     - 覆盖顺序仍安全，因为 wrap 只改变坐标映射，不改变 mover 的前后关系
#   placement_legality:
#     - 输入里左侧精灵仍不重叠、不越界
#     - 框边标记仍一一镜像，且不同 mover 不共享同一行/列标记，避免目标归属歧义
#     - 生成器允许 target 靠近边缘，但这只影响输出印刻，不会让输入对象越界
#   observability_and_cleanliness:
#     - 通过拒绝采样要求 wrapped_output != clipped_output，确保样本不是“看起来还是原题”的退化情况
#     - 额外要求至少两个输出像素与裁剪版不同，避免只有单像素差异过于隐蔽
#     - 非强制 wrap 的 mover 优先留在非 wrap 目标，减少 interior 内部过度叠层，保证肉眼可读
#   new_main_assumptions_and_generator_guarantees:
#     - 新 main 假设 mover 的 target 至少来自可解析的边框颜色配对；生成器保持原有镜像标记构造
#     - 新 main 假设 interior 的高宽都 >= 1；原题规格的 8-13 已保证
#     - 新 main 不再要求 target 能容纳完整精灵；生成器正是利用这点制造边缘环绕


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
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


def _sprite_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    """Occupied sprite coordinates relative to the original bbox-center anchor."""
    h, w = mask.shape
    center_r = int((h - 1) // 2)
    center_c = int((w - 1) // 2)
    return [
        (int(rr - center_r), int(cc - center_c))
        for rr, cc in np.argwhere(mask)
    ]


def _stamped_cells(
    offsets: list[tuple[int, int]],
    target: tuple[int, int],
    interior: tuple[int, int, int, int],
    *,
    wrap: bool,
) -> list[tuple[int, int]]:
    """Cells filled when a sprite with the given offsets is stamped at target."""
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
                    "offsets": _sprite_offsets(m),
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


def _solve(grid: np.ndarray, *, wrap: bool) -> np.ndarray:
    """
    Shared solver core.

    wrap=False reproduces the original clipped stamping behavior.
    wrap=True applies the D13 toroidal interior behavior.
    """
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
        for target in targets[int(col)]:
            for rr, cc in _stamped_cells(s["offsets"], target, interior, wrap=wrap):
                out[int(rr), int(cc)] = int(col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    read marker-defined target intersections exactly as in the original task,
    but stamp each mover on the frame interior with toroidal wraparound.
    """
    return _solve(grid, wrap=True)


def _spaced_mark_options(candidates: list[int], min_spacing: int) -> list[list[int]]:
    """All 1- or 2-mark choices with the requested spacing."""
    opts: list[list[int]] = []
    vals = sorted(set(map(int, candidates)))
    for n in (1, 2):
        if len(vals) < n:
            continue
        for combo in combinations(vals, n):
            if all(abs(int(combo[j]) - int(combo[i])) >= int(min_spacing) for i in range(n) for j in range(i + 1, n)):
                opts.append([int(v) for v in combo])
    random.shuffle(opts)
    return opts


def _wrap_stats_for_targets(
    offsets: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    interior: tuple[int, int, int, int],
) -> tuple[int, int]:
    """Return (wrapped_pixel_events, wrapped_targets)."""
    ir0, ic0, ir1, ic1 = map(int, interior)
    wrap_pixels = 0
    wrap_targets = 0
    for tr, tc in targets:
        target_wrap = False
        for dr, dc in offsets:
            raw_r = int(tr + dr)
            raw_c = int(tc + dc)
            if not (int(ir0) <= raw_r <= int(ir1) and int(ic0) <= raw_c <= int(ic1)):
                wrap_pixels += 1
                target_wrap = True
        if target_wrap:
            wrap_targets += 1
    return int(wrap_pixels), int(wrap_targets)


def analyze_variant(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether this input actually exercises the D13 behavior.

    Returns counts that are useful for post-generation validation.
    """
    g, _bg, interior, sprites, targets = _extract_scene(grid)
    wrap_pixels = 0
    wrap_targets = 0
    for s in sprites:
        col = int(s["color"])
        if col not in targets:
            continue
        wp, wt = _wrap_stats_for_targets(s["offsets"], targets[col], interior)
        wrap_pixels += int(wp)
        wrap_targets += int(wt)
    clipped = _solve(g, wrap=False)
    wrapped = _solve(g, wrap=True)
    return {
        "wrap_pixels": int(wrap_pixels),
        "wrap_targets": int(wrap_targets),
        "diff_vs_clipped": int(np.count_nonzero(clipped != wrapped)),
    }


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

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    forced_wrap_color = random.choice(mover_colors)

    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        offsets = _sprite_offsets(sp != bg)
        avail_rows = [r for r in range(ir0, ir1 + 1) if r not in occupied_rows]
        avail_cols = [c for c in range(ic0, ic1 + 1) if c not in occupied_cols]
        row_options = _spaced_mark_options(avail_rows, max(1, sh))
        col_options = _spaced_mark_options(avail_cols, max(1, sw))
        if not row_options or not col_options:
            return None

        nonwrap_pairs: list[tuple[list[int], list[int], int, int]] = []
        wrap_pairs: list[tuple[list[int], list[int], int, int]] = []
        for rows in row_options:
            for cols in col_options:
                targets = [(int(r), int(c)) for r in rows for c in cols]
                wrap_pixels, wrap_targets = _wrap_stats_for_targets(offsets, targets, interior)
                rec = (list(rows), list(cols), int(wrap_pixels), int(wrap_targets))
                if wrap_targets > 0:
                    wrap_pairs.append(rec)
                else:
                    nonwrap_pairs.append(rec)

        if mc == forced_wrap_color:
            pool = wrap_pairs
        else:
            if nonwrap_pairs and random.random() < 0.7:
                pool = nonwrap_pairs
            else:
                pool = wrap_pairs if wrap_pairs else nonwrap_pairs
        if not pool:
            return None

        rows, cols, _wp, _wt = random.choice(pool)
        mover_targets[mc] = (sorted(rows), sorted(cols))
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
    all_items: list[tuple[int, np.ndarray]] = [
        (mc, sp) for mc, sp in zip(mover_colors, sprites)
    ]
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
        wrapped_output = _solve(inp, wrap=True)
        clipped_output = _solve(inp, wrap=False)
    except Exception:
        return None

    if np.array_equal(inp, wrapped_output):
        return None
    if np.array_equal(wrapped_output, clipped_output):
        return None

    metrics = analyze_variant(inp)
    if int(metrics["wrap_targets"]) <= 0:
        return None
    if int(metrics["diff_vs_clipped"]) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
