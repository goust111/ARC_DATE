# Variant of 247ef758
# Dimension: D3 — 目标位置：框架与标记结构 (dst)
# Change: 框架右侧为矩形框（单色边框+内部背景），边缘标记位于边框像素上，目标为行标记×列标记的笛卡尔积中心点，精灵印刻限制在框架内部
#         → 目标为边缘标记的投影交点
#
# Design analysis
# ===============
# variant_spec:
#   what_changes:
#     - 原题在左右边框同一行放同色标记、在上下边框同一列放同色标记；目标点就是这些行列的直接笛卡尔积。
#     - 本变体不再让标记直接给出行/列。每个 mover 颜色现在在边框上编码两组 45 度投影线：
#       1. "/" 线：顶部一个同色标记 + 左边一个同色标记，且二者距左上内角的偏移相同。
#       2. "\" 线：顶部一个同色标记 + 右边一个同色标记，且二者距右上内角的偏移相同。
#     - 真正的目标点是 "/" 与 "\" 两组投影线的交点。一个颜色可有 1-2 条 "/" 线和 1-2 条 "\" 线，
#       所以仍保留“一个精灵复制到多个点”的原题骨架，只是目标集合从“行×列”变成“投影线×投影线”。
#   concrete_pixel_geometry:
#     - frame 仍然是右侧整高矩形框，内部是 bg，左侧仍是精灵条带 + 单列 divider。
#     - "/" 线只靠 top+left 两个端点定义；其离散线满足 r+c = 常数。
#     - "\" 线只靠 top+right 两个端点定义；其离散线满足 r-c = 常数。
#     - 交点公式（内部坐标）：
#         tr = ir0 + (slash_top - back_top) // 2
#         tc = (slash_top + back_top) // 2
#       因此 generator 强制同色两组顶部标记同奇偶，且 slash_top > back_top。
#   behavior_trace:
#     - 原题局部例子：left/right 在第 4 行放红，top/bottom 在第 9 列放红 -> 目标直接是 (4, 9)。
#     - 本变体局部例子：top 第 11 列 + left 第 7 行定义一条 "/"；top 第 7 列 + right 第 5 行定义一条 "\"。
#       solver 先恢复两条投影线，再算交点：
#         tr = ir0 + (11 - 7) / 2
#         tc = (11 + 7) / 2
#       得到内部某一点；这个点既不在标记行，也不在标记列，必须经过“连线求交”。
#   complexity_calibration:
#     - 仍保留 2-3 个 mover、0-1 个 distractor、底部优先覆盖顶部。
#     - 为避免配对歧义，不使用“同色两侧任意排序配对”，而是让每条投影线由角距离唯一决定：
#       top 与 left/right 的偏移一致才构成一条合法线。这样同色有 2 条线时也不会歧义。
#     - 保持 mover 颜色 distinct，且各色的 top/left/right 标记位置互不复用，减少视觉混杂。
#   implicit_assumption_audit:
#     - 原 solver 的 row_marks / col_marks 依赖“左右同一行、上下同一列”这个旧前提；变体中这一前提失效，
#       所以整段直接读取行列的逻辑必须替换。
#     - 原 solver 擦除 mover、按输入垂直顺序覆盖、裁剪到 interior，这些前提与 D3 无关，保持不变。
#     - 新 solver 额外假设：同色 "/" 与 "\" 线由 top+side 的角距离唯一识别，且所有目标交点都落在 interior。
#       generator 显式保证这些条件。
#   placement_legality:
#     - generator 先生成 sprite，再根据 sprite bbox 可居中的合法区域筛选投影线。
#     - 对每个 mover，要求其所有交点都能完整落入 interior；同时 top/left/right 标记不复用，目标中心不重叠。
#     - divider、frame、sprites 仍然互不越界；bg 仍保持为全局多数色。
#   visual_observability:
#     - 观察样例时，人能看到：顶部与左右侧出现同色端点，但它们不在同一行/列；输出中的精灵落在这些斜线交叉处。
#     - bottom 边不再提供目标信息，因此不会被误读为原题的“上下同列”。
#   cleanliness:
#     - 只启用 top/left/right 三边标记，避免四边都堆同色端点造成混乱。
#     - target centers across colors 默认不重叠，减少覆盖造成的判读负担；覆盖规则仍保留为潜在但不主导的现象。
#   color_inference:
#     - 颜色仅用于“哪一个精灵对应哪一组投影线”，不承担隐藏排序或数值语义。
# assumptions_guaranteed_by_generator:
#   - 每个 mover 至少有 1 条 "/" 线和 1 条 "\" 线。
#   - 同一颜色的 slash_top 与 back_top 列满足 slash_top > back_top 且同奇偶，因此交点坐标为整数并位于上半区域。
#   - 对应 left/right 支撑标记按角距离唯一配对；不会出现同一 top 标记同时承担两种角色。
#   - 所有目标点足够远离 interior 边缘，使精灵居中印刻后无需依赖裁剪才能“看见”规则。

from __future__ import annotations

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


def _projection_targets_from_top_cols(
    slash_cols: list[int],
    back_cols: list[int],
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
) -> list[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for sc in slash_cols:
        for bc in back_cols:
            if sc <= bc or ((sc - bc) & 1):
                continue
            tr = int(ir0 + (sc - bc) // 2)
            tc = int((sc + bc) // 2)
            if int(ir0) <= tr <= int(ir1) and int(ic0) <= tc <= int(ic1):
                pts.add((tr, tc))
    return sorted(pts)


def _projection_targets_for_color(
    g: np.ndarray,
    color: int,
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fc0: int,
    fc1: int,
) -> list[tuple[int, int]]:
    top_cols = [int(c) for c in range(int(ic0), int(ic1) + 1) if int(g[int(fr0), int(c)]) == int(color)]
    left_rows = {int(r) for r in range(int(ir0), int(ir1) + 1) if int(g[int(r), int(fc0)]) == int(color)}
    right_rows = {int(r) for r in range(int(ir0), int(ir1) + 1) if int(g[int(r), int(fc1)]) == int(color)}

    slash_cols: list[int] = []
    for c in top_cols:
        left_r = int(ir0 + (c - int(ic0)))
        if int(ir0) <= left_r <= int(ir1) and left_r in left_rows:
            slash_cols.append(int(c))

    back_cols: list[int] = []
    for c in top_cols:
        right_r = int(ir0 + (int(ic1) - c))
        if int(ir0) <= right_r <= int(ir1) and right_r in right_rows:
            back_cols.append(int(c))

    return _projection_targets_from_top_cols(slash_cols, back_cols, ir0=ir0, ic0=ic0, ir1=ir1, ic1=ic1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into the right frame.

    Variant rule:
    - A mover color is indicated by top+left slash markers and top+right backslash markers.
    - Targets are intersections of those projection lines, not direct row/column marker coordinates.
    - Movers are erased from the left strip and stamped centered at every target.
    - Lower sprites stamp first; higher sprites stamp last and cover them.
    """
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
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in sorted(move_colors):
        pts = _projection_targets_for_color(
            g,
            int(col),
            ir0=ir0,
            ic0=ic0,
            ir1=ir1,
            ic1=ic1,
            fr0=fr0,
            fc0=fc0,
            fc1=fc1,
        )
        if pts:
            targets[int(col)] = pts

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


def _fit_bounds_for_sprite(sprite: np.ndarray, *, ir0: int, ic0: int, ir1: int, ic1: int) -> tuple[int, int, int, int]:
    sh, sw = sprite.shape
    top_margin = sh // 2
    bot_margin = sh - 1 - top_margin
    left_margin = sw // 2
    right_margin = sw - 1 - left_margin
    return (
        int(ir0 + top_margin),
        int(ir1 - bot_margin),
        int(ic0 + left_margin),
        int(ic1 - right_margin),
    )


def _choose_projection_sets(
    sprite: np.ndarray,
    *,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    used_top: set[int],
    used_left: set[int],
    used_right: set[int],
    used_targets: set[tuple[int, int]],
) -> tuple[list[int], list[int], list[tuple[int, int]]] | None:
    fit_r0, fit_r1, fit_c0, fit_c1 = _fit_bounds_for_sprite(sprite, ir0=ir0, ic0=ic0, ir1=ir1, ic1=ic1)
    if fit_r0 > fit_r1 or fit_c0 > fit_c1:
        return None

    max_slash = min(int(ic1), int(ic0 + (ir1 - ir0)))
    min_back = max(int(ic0), int(ic1 - (ir1 - ir0)))
    base_slash = [c for c in range(int(ic0), int(max_slash) + 1) if c not in used_top]
    base_back = [c for c in range(int(min_back), int(ic1) + 1) if c not in used_top]
    if not base_slash or not base_back:
        return None

    count_options = [(2, 2), (1, 2), (2, 1), (1, 1)]
    random.shuffle(count_options)

    for n_slash, n_back in count_options:
        for _ in range(200):
            parity = random.randint(0, 1)
            slash_pool = [c for c in base_slash if (c & 1) == parity]
            back_pool = [c for c in base_back if (c & 1) == parity]
            if len(slash_pool) < n_slash or len(back_pool) < n_back:
                continue

            chosen_back = sorted(random.sample(back_pool, n_back))
            higher_slash_pool = [c for c in slash_pool if c > max(chosen_back)]
            if len(higher_slash_pool) < n_slash:
                continue
            chosen_slash = sorted(random.sample(higher_slash_pool, n_slash))

            left_rows = [int(ir0 + (c - int(ic0))) for c in chosen_slash]
            right_rows = [int(ir0 + (int(ic1) - c)) for c in chosen_back]
            if any(r in used_left for r in left_rows) or any(r in used_right for r in right_rows):
                continue

            targets = _projection_targets_from_top_cols(
                chosen_slash, chosen_back, ir0=ir0, ic0=ic0, ir1=ir1, ic1=ic1
            )
            if len(targets) != int(n_slash * n_back):
                continue
            if any((tr, tc) in used_targets for tr, tc in targets):
                continue
            if any(not (fit_r0 <= tr <= fit_r1 and fit_c0 <= tc <= fit_c1) for tr, tc in targets):
                continue

            return chosen_slash, chosen_back, targets
    return None


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
    fr0, fr1 = 0, H - 1

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

    used_top: set[int] = set()
    used_left: set[int] = set()
    used_right: set[int] = set()
    used_targets: set[tuple[int, int]] = set()
    mover_markers: dict[int, tuple[list[int], list[int], list[tuple[int, int]]]] = {}
    for mc, sp in zip(mover_colors, sprites):
        picked = _choose_projection_sets(
            sp,
            ir0=ir0,
            ic0=ic0,
            ir1=ir1,
            ic1=ic1,
            used_top=used_top,
            used_left=used_left,
            used_right=used_right,
            used_targets=used_targets,
        )
        if picked is None:
            return None
        slash_cols, back_cols, targets = picked
        mover_markers[int(mc)] = (slash_cols, back_cols, targets)
        used_top.update(slash_cols)
        used_top.update(back_cols)
        used_left.update(int(ir0 + (c - int(ic0))) for c in slash_cols)
        used_right.update(int(ir0 + (int(ic1) - c)) for c in back_cols)
        used_targets.update(targets)

    inp = np.full((H, W), bg, dtype=int)
    inp[:, div_col] = divider_color

    inp[fr0, fc0 : fc1 + 1] = frame_base
    inp[fr1, fc0 : fc1 + 1] = frame_base
    inp[1 : H - 1, fc0] = frame_base
    inp[1 : H - 1, fc1] = frame_base

    for mc, (slash_cols, back_cols, _) in mover_markers.items():
        for c in slash_cols:
            inp[fr0, int(c)] = int(mc)
            inp[int(ir0 + (int(c) - int(ic0))), fc0] = int(mc)
        for c in back_cols:
            inp[fr0, int(c)] = int(mc)
            inp[int(ir0 + (int(ic1) - int(c))), fc1] = int(mc)

    cur_row = 0
    all_items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(dc_color, min(5, H - cur_row), sprite_area_w)
        if ds is None:
            return None
        all_items.append((dc_color, ds, True))

    for mc, sp, _ in all_items:
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
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
