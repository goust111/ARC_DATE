# Variant of 247ef758
# Dimension: D3 — 目标位置：框架与标记结构 (dst)
# Change: 框架右侧为矩形框（单色边框+内部背景），边缘标记位于边框像素上，目标为行标记×列标记的笛卡尔积中心点，精灵印刻限制在框架内部
#   → 框架边缘非对称标记：左右边仍共同决定目标行，但上边和下边分别独立提供目标列，不再要求上下同色同列镜像。
#
# Design analysis
# ---
# variant_spec:
#   layout:
#     unchanged:
#       - 左侧精灵条
#       - 单列整高分隔符
#       - 右侧整高矩形框，内部为 bg
#     changed:
#       - 左右边缘仍是镜像行标记：同一颜色同时出现在左/右边时，该行是该颜色的目标行
#       - 上下边缘不再镜像：顶部列标记和底部列标记分别独立生效
#       - 某颜色的目标列 = 该颜色在上边出现的列 ∪ 该颜色在下边出现的列
#   concrete_example:
#     old:
#       - 若红色只在 top 的 c=7 出现、bottom 的 c=7 不是红色，则原题不会把 c=7 视为红色目标列
#     new:
#       - 同样输入下，红色 c=7 仍然生效；若红色还在 bottom 的 c=10 出现，则红色列集合是 {7, 10}
#       - 再与左右边给出的红色目标行做笛卡尔积，得到所有印刻中心
#   complexity_calibration:
#     - 只放开列维度，行维度继续保持左右同色，避免两维都独立后匹配过于松散
#     - 所有 mover 列位置在全局唯一，不让两个颜色共享同一列，减少覆盖歧义
#     - 每个样本强制同时出现 top-only 列和 bottom-only 列，保证视觉上能看出“不再要求上下同列同色”
#   implicit_assumption_audit:
#     - 原 solver 的 `ct == cb` 依赖“上下边镜像”；变体下已失效，必须改成分别收集 top/bottom
#     - 原 generator 的 `inp[H - 1] = top_edge` 会直接退化回原题，必须改成独立构造 `bottom_edge`
#     - 原 row 解析 `cl == cr` 仍然合法，因为本变体没有改变左右边规则
#     - 擦除原精灵、按输入纵向顺序覆盖、框内裁剪，这些与 D3 无关，保持不变
#   legality:
#     - 目标行仍按 sprite 高度留出上下余量；目标列按 sprite 宽度留出左右余量
#     - 每个 mover 至少有一个 row mark，且至少在 top/bottom 之一有一个列标记
#     - 顶边和底边的列位置全局不复用，所以不会出现多颜色共享同一目标列
#   observability:
#     - 解题者能直接看到：某些颜色只在 top 标、某些只在 bottom 标，但输出仍在这些列出现复制
#     - 若用原规则求解，这些 top-only / bottom-only 列都会丢失，因此新旧行为有明确差异
#   cleanliness:
#     - 未使用的边框像素全部保持 frame_base，框内只出现印刻结果，不额外引入噪声对象
#   new_main_assumptions_checked_against_generator:
#     - 框内部仍是最大且不接触外边界的 bg 连通块
#     - 分隔列仍是左侧唯一近乎整列同色的非 bg 列
#     - 左右边的 mover 行标记依然成对出现
#     - 每个 mover 至少有一个独立列标记；generator 明确保证

from __future__ import annotations

from collections import defaultdict
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Variant rule:
    - Rows are still determined by matching left/right edge colors.
    - Columns are determined independently by top and bottom edges: any appearance of a sprite
      color on either horizontal edge contributes a target column for that color.
    - Erase each moved sprite from the left and stamp it centered at every row×column target.
      Lower input sprites are stamped first, so upper ones cover them.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert not (inner_bbox is None)
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
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

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
        if ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))
        if cb != int(bg) and cb in move_colors:
            col_marks[int(cb)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        cols = sorted(set(col_marks[int(col)]))
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in cols}
        targets[int(col)] = sorted(pts)

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


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    candidates = list(candidates)
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked) if picked else None


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


def _assign_column_marks(
    mover_colors: list[int],
    sprites: list[np.ndarray],
    ic0: int,
    ic1: int,
) -> dict[int, tuple[list[int], list[int]]] | None:
    """
    Assign non-mirrored column markers.

    Returns:
      {color: (top_cols, bottom_cols)}

    Guarantees at the sample level:
    - at least one mover is top-only
    - at least one mover is bottom-only
    - top/bottom column locations are globally unique across all movers
    """
    order = mover_colors[:]
    random.shuffle(order)
    roles: dict[int, str] = {}
    roles[order[0]] = "top"
    roles[order[1]] = "bottom"
    if len(order) >= 3:
        roles[order[2]] = random.choice(["both", "top", "bottom"])

    by_color_sprite = {int(mc): sp for mc, sp in zip(mover_colors, sprites)}
    occupied_cols: set[int] = set()
    result: dict[int, tuple[list[int], list[int]]] = {}

    for mc in mover_colors:
        sp = by_color_sprite[int(mc)]
        sw = int(sp.shape[1])
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_col > max_col:
            return None

        role = roles[int(mc)]
        if role == "top":
            top_n = random.choice([1, 1, 2])
            bottom_n = 0
        elif role == "bottom":
            top_n = 0
            bottom_n = random.choice([1, 1, 2])
        else:
            top_n = 1
            bottom_n = 1

        total_n = int(top_n + bottom_n)
        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        picked = _pick_spaced(avail_cols, total_n, max(1, sw))
        if not picked or len(picked) != total_n:
            return None

        picked = list(picked)
        random.shuffle(picked)
        top_cols = sorted(picked[:top_n])
        bottom_cols = sorted(picked[top_n:])
        occupied_cols.update(top_cols)
        occupied_cols.update(bottom_cols)
        result[int(mc)] = (top_cols, bottom_cols)

    if not any(len(top) > 0 and len(bottom) == 0 for top, bottom in result.values()):
        return None
    if not any(len(bottom) > 0 and len(top) == 0 for top, bottom in result.values()):
        return None

    return result


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
        s = _make_sprite(mc, 5, sprite_area_w)
        if s is None:
            return None
        sprites.append(s)

    mover_rows: dict[int, list[int]] = {}
    occupied_rows: set[int] = set()
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = sp.shape
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_row > max_row or min_col > max_col:
            return None
        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        row_marks = _pick_spaced(avail_rows, random.randint(1, 2), max(1, sh))
        if not row_marks:
            return None
        mover_rows[int(mc)] = sorted(row_marks)
        occupied_rows.update(row_marks)

    mover_cols = _assign_column_marks(mover_colors, sprites, ic0, ic1)
    if mover_cols is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    top_edge = np.full(frame_w, frame_base, dtype=int)
    bottom_edge = np.full(frame_w, frame_base, dtype=int)
    for mc in mover_colors:
        top_cols, bottom_cols = mover_cols[int(mc)]
        for cc in top_cols:
            top_edge[int(cc - fc0)] = int(mc)
        for cc in bottom_cols:
            bottom_edge[int(cc - fc0)] = int(mc)
    inp[0, fc0 : fc0 + frame_w] = top_edge
    inp[H - 1, fc0 : fc0 + frame_w] = bottom_edge

    for r in range(1, H - 1):
        val = frame_base
        for mc in mover_colors:
            if r in mover_rows[int(mc)]:
                val = int(mc)
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

    for mc, sp in all_items:
        sh, sw = sp.shape
        if sw > sprite_area_w or cur_row + sh > H:
            return None
        c_off = random.randint(0, sprite_area_w - sw)
        for dr in range(sh):
            for dc in range(sw):
                if sp[dr, dc] != bg:
                    inp[cur_row + dr, c_off + dc] = int(sp[dr, dc])
        cur_row += sh

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    asym_cols = 0
    for c in range(int(ic0), int(ic1) + 1):
        top_v = int(inp[0, c])
        bottom_v = int(inp[H - 1, c])
        if (top_v in mover_colors or bottom_v in mover_colors) and top_v != bottom_v:
            asym_cols += 1
    if asym_cols == 0:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
