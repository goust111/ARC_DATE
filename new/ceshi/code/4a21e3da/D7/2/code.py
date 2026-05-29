# Variant of 4a21e3da
# Dimension: D7 — 多步处理序列 (multi_step)
# Change: 两步：先用射线切割精灵，再将片段移到角落... → 先变换再切割
#
# Design analysis for this variant
# --------------------------------
# a) 具体化与行为追踪
# - 变体仍然只有 3 种颜色：bg、fg、ctrl，ctrl 仍然只出现在边界上，仍然定义
#   一条竖切线和/或一条横切线。
# - 关键变化是：输入里的整块 fg 精灵不会直接被切。solver 必须先把这整块精灵在其
#   紧致 bbox 内顺时针旋转 90 度，然后才按原题规则发射 ctrl 射线、把被切开的区域
#   滑到角落、让不接壤 ctrl 边的区域消失。
# - 为了让旋转是可逆且无需额外标记，本实现把“旋转”具体化为：
#   对 fg 的 tight bbox 做 90 度旋转，并把旋转后的 patch 仍然放回该 bbox 的左上角。
#   这会交换高宽，所以输入通常更高更窄，而“待切割的中间态”更矮更宽。
# - 小例子：若输入 fg 的 tight bbox 是 9x6，先旋成 6x9，再看 ctrl 定义的 split line。
#   原任务会直接在 9x6 的当前图形上切；新任务会先得到 6x9 的中间图形，再在这一步上切。
#   因此输出中的 split-line 残留 fg、射线终点、四角碎片形状，都会对应“旋转后”的
#   图形，而不是输入里肉眼直接看到的朝向。
#
# b) 复杂度校准
# - 只固定一种变换：顺时针 90 度。这样规则档位少，样例里可重复支持。
# - 为了避免旋转不可见，generator 强制拒绝旋转前后完全相同的精灵，并额外拒绝
#   legacy solver 与 variant solver 输出完全相同的样本。
# - 保留原题 1 或 2 个 ctrl 的分支，因此仍然有“2 块/4 象限”和“有消失象限/无消失”
#   的结构复杂度，不会退化成单一模板。
#
# c) 隐式假设审查（与 main 直接相关）
# - 原 main 中所有用 `g == fg` 的地方，都隐含“输入里看到的 fg 就是待切对象”。
#   在本变体里这不再成立，因此必须先生成 `work_pts = rotate(fg_pts)`，之后所有
#   split-line、ray、region、corner-slide 都只能基于 `work_pts`。
# - 原 main 中 ctrl 颜色取自边界非 bg 颜色，这个前提仍成立，因为旋转不碰 ctrl。
# - 原 main 中 `c_split/r_split` 由 ctrl 的位置唯一确定，这个前提仍成立。
# - 原 main 中“fg on split line stays in place”仍成立，但这里的 split-line fg 指的是
#   旋转后的中间态像素，不是输入原像素。solver 必须显式改为保留 `work_pts` 上的
#   split-line 像素，否则会把错误的像素留在原地。
#
# d) 放置合法性
# - generator 先构造“旋转后、待切割”的宽形精灵，再把整块精灵逆时针旋回输入。
# - 为了保证旋回后的输入仍离边界至少 4 格，放置 transformed sprite 时使用
#   `max(sprite_h, sprite_w)` 作为安全包络。
# - 生成时继续保留原题对 kept piece / vanish piece 的不重叠、不过 split line、
#   角落滑移后不越界的约束。
#
# e) 视觉可观察性
# - transformed sprite 维持原题的“宽于高”分布（约 5-8 x 8-10），因此输入经逆旋后
#   通常变成“高于宽”的形状；输出中的角落碎片又恢复成按宽形中间态切出的样子。
# - generator 拒绝 `legacy_main(input) == main(input)` 的样本，保证“先旋转”这一步
#   真的影响输出，而不是表面改了规则、实际上仍可按原题理解。
#
# f) 可视化整洁度
# - 仍然只有一块 fg 连通体和 1-2 个边界 ctrl 像素，不引入新颜色或新标记层。
# - 旋转发生在输入精灵内部，不增加额外叠画；输出仍保持原题的“射线 + split-line fg
#   + 角落碎片”三层结构。
#
# g) 颜色变化可推断性
# - 不引入颜色映射；颜色角色与原题完全一致。变化只来自步骤顺序。
#
# New main() assumptions and generator guarantees
# -----------------------------------------------
# - main() 假设：所有 fg 像素属于一个整体精灵；其 tight bbox 旋转后不会碰到边界 ctrl。
#   generator 通过 margin>=4 和安全包络保证这一点。
# - main() 假设：ctrl 仍然是边界上唯一的非 bg 颜色。generator 保持不变。
# - main() 假设：旋转后的精灵在 ctrl 定义的 split line 上仍有足够结构可切。generator
#   是按 transformed sprite 直接构造 split-line 像素和各 region 的，因此保证成立。
from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _bbox_of_points(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return int(min(rs)), int(max(rs)), int(min(cs)), int(max(cs))


def _rotate_points_clockwise_tl(pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not pts:
        return set()
    r0, r1, c0, c1 = _bbox_of_points(pts)
    h = int(r1 - r0 + 1)
    out: set[tuple[int, int]] = set()
    for r, c in pts:
        lr = int(r - r0)
        lc = int(c - c0)
        out.add((int(r0 + lc), int(c0 + (h - 1 - lr))))
    return out


def _rotate_points_counterclockwise_tl(pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if not pts:
        return set()
    r0, r1, c0, c1 = _bbox_of_points(pts)
    w = int(c1 - c0 + 1)
    out: set[tuple[int, int]] = set()
    for r, c in pts:
        lr = int(r - r0)
        lc = int(c - c0)
        out.add((int(r0 + (w - 1 - lc)), int(c0 + lr)))
    return out


def _infer_colors_and_controls(
    g: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, int]], set[str], int | None, int | None]:
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            E.add("top")
            c_split = int(c)
        elif r == H - 1:
            E.add("bottom")
            c_split = int(c)
        elif c == 0:
            E.add("left")
            r_split = int(r)
        elif c == W - 1:
            E.add("right")
            r_split = int(r)
    return int(bg), int(fg), int(ctrl), ctrl_pts, E, c_split, r_split


def _render_cut_and_slide(
    sprite_pts: set[tuple[int, int]],
    H: int,
    W: int,
    bg: int,
    fg: int,
    ctrl: int,
    ctrl_pts: list[tuple[int, int]],
    E: set[str],
    c_split: int | None,
    r_split: int | None,
) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)

    sprite_set = set(sprite_pts)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [rr for rr, cc in sprite_set if cc == c]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if (rr, c) not in sprite_set:
                        out[rr, c] = int(ctrl)
        elif r == H - 1:
            rows = [rr for rr, cc in sprite_set if cc == c]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if (rr, c) not in sprite_set:
                        out[rr, c] = int(ctrl)
        elif c == 0:
            cols = [cc for rr, cc in sprite_set if rr == r]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if (r, cc) not in sprite_set:
                        out[r, cc] = int(ctrl)
        elif c == W - 1:
            cols = [cc for rr, cc in sprite_set if rr == r]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if (r, cc) not in sprite_set:
                        out[r, cc] = int(ctrl)

    for r, c in ctrl_pts:
        out[r, c] = int(ctrl)

    split_line_pts = {
        (r, c)
        for r, c in sprite_set
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split)
    }
    for r, c in split_line_pts:
        out[r, c] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in sprite_set:
        if (r, c) in split_line_pts:
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    def touches_edges(hs: str, vs: str) -> set[str]:
        te: set[str] = set()
        if hs in ("T", "A"):
            te.add("top")
        if hs in ("B", "A"):
            te.add("bottom")
        if vs in ("L", "A"):
            te.add("left")
        if vs in ("R", "A"):
            te.add("right")
        return te

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
        if r_split is not None:
            dh = hs
        elif ("top" in E) and ("bottom" not in E):
            dh = "T"
        elif ("bottom" in E) and ("top" not in E):
            dh = "B"
        else:
            dh = "T"

        if c_split is not None:
            dv = vs
        elif ("left" in E) and ("right" not in E):
            dv = "L"
        elif ("right" in E) and ("left" not in E):
            dv = "R"
        else:
            dv = "L"
        return str(dh), str(dv)

    for (hs, vs), pts in regions.items():
        if not (touches_edges(hs, vs) & E):
            continue

        dh, dv = pick_dest(hs, vs)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
        dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(fg)

    return out


def _solve_variant(grid: np.ndarray, *, rotate_first: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts, E, c_split, r_split = _infer_colors_and_controls(g)
    sprite_pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    if rotate_first:
        sprite_pts = _rotate_points_clockwise_tl(sprite_pts)
    return _render_cut_and_slide(sprite_pts, H, W, bg, fg, ctrl, ctrl_pts, E, c_split, r_split)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, rotate_first=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    1. Rotate the entire fg sprite 90 degrees clockwise inside its tight bbox.
    2. Use the original border-ray cut + corner-slide rule on that rotated sprite.
    """
    return _solve_variant(grid, rotate_first=True)


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= max_h
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= max_w
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _mask_is_contiguous(H: int, W: int, pts: set[tuple[int, int]]) -> bool:
    mask = np.zeros((H, W), dtype=int)
    for r, c in pts:
        mask[int(r), int(c)] = 1
    return bool(is_contiguous(mask))


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    E = set(ctrl_edges)
    has_v = has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, int(c_split)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else W - 1))
            has_h = True

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
        if r_split is not None:
            dh = hs
        elif "top" in E and "bottom" not in E:
            dh = "T"
        elif "bottom" in E and "top" not in E:
            dh = "B"
        else:
            dh = "T"
        if c_split is not None:
            dv = vs
        elif "left" in E and "right" not in E:
            dv = "L"
        elif "right" in E and "left" not in E:
            dv = "R"
        else:
            dv = "L"
        return str(dh), str(dv)

    def touches_E(hs: str, vs: str) -> bool:
        te = set()
        if hs in ("T", "A"):
            te.add("top")
        if hs in ("B", "A"):
            te.add("bottom")
        if vs in ("L", "A"):
            te.add("left")
        if vs in ("R", "A"):
            te.add("right")
        return bool(te & E)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [r for r in all_regions if touches_E(*r)]
    vanish = [r for r in all_regions if r not in kept]

    # This bbox describes the transformed sprite, i.e. the shape after the hidden
    # rotation and before the original cut-and-slide rule.
    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)
    safe_side = int(max(sprite_h, sprite_w))
    max_r0 = H - 5 - safe_side + 1
    max_c0 = W - 5 - safe_side + 1
    if max_r0 < 4 or max_c0 < 4:
        return None

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split - int(frac * sprite_w))
        sprite_c0 = max(4, min(sprite_c0, max_c0))
        if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
            return None
    else:
        sprite_c0 = random.randint(4, max_c0)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
        sprite_r0 = max(4, min(sprite_r0, max_r0))
        if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
            return None
    else:
        sprite_r0 = random.randint(4, max_r0)

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split + 1)
        r1 = (int(r_split - 1) if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split + 1)
        c1 = (int(c_split - 1) if vs == "L" else sprite_c1) if has_v else sprite_c1
        return int(r0), int(r1), int(c0), int(c1)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)

    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    n_kept = len(kept)
    n_vanish = len(vanish)
    budget_pieces = target_total - n_split
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= vanish_budget
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for i in range(n_kept):
        r0, r1, c0, c1 = sub_bbox(*kept[i])
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if i == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - i)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - i - 1) * 3))
        piece_budgets.append(int(px))
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    transformed_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = pick_dest(hs, vs)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[idx]

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1

        if has_h:
            inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        else:
            split_rows = sorted(r for r, c in split_pixels if c == c_split)
            if split_rows:
                mid = (split_rows[0] + split_rows[-1]) // 2
                inp_r0 = max(r0, min(mid - ph // 2, r1 - ph + 1))
            else:
                inp_r0 = r0

        if has_v:
            inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        else:
            split_cols = sorted(c for r, c in split_pixels if r == r_split)
            if split_cols:
                mid = (split_cols[0] + split_cols[-1]) // 2
                inp_c0 = max(c0, min(mid - pw // 2, c1 - pw + 1))
            else:
                inp_c0 = c0

        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None

        transformed_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((int(r + dr), int(c + dc)) for r, c in inp_pts)
        for r, c in out_pts:
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        output_pieces[reg] = out_pts

    all_transformed_fg = set()
    for pts in transformed_pieces.values():
        all_transformed_fg |= set(pts)
    all_transformed_fg |= split_pixels
    if not all_transformed_fg:
        return None

    fg_arr = np.array(list(all_transformed_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None
    if aw <= ah:
        return None
    if not _mask_is_contiguous(H, W, all_transformed_fg):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh, vw = vr1 - vr0 + 1, vc1 - vc0 + 1
            if vh < 1 or vw < 1:
                return None
            max_px = min(vanish_budget, vh * vw)
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph = max(r for r, _ in piece) + 1
            ppw = max(c for _, c in piece) + 1
            vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
            vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (r, c) in all_transformed_fg:
                    return None
            vanish_fg = vpts

    combined_rot = all_transformed_fg | vanish_fg
    if not _mask_is_contiguous(H, W, combined_rot):
        return None

    rot_arr = np.array(list(combined_rot))
    rot_h = int(rot_arr[:, 0].max() - rot_arr[:, 0].min() + 1)
    rot_w = int(rot_arr[:, 1].max() - rot_arr[:, 1].min() + 1)
    if rot_h < 5 or rot_w < 8 or rot_w <= rot_h:
        return None

    combined_in = _rotate_points_counterclockwise_tl(combined_rot)
    if combined_in == combined_rot:
        return None
    if not _mask_is_contiguous(H, W, combined_in):
        return None

    in_arr = np.array(list(combined_in))
    in_h = int(in_arr[:, 0].max() - in_arr[:, 0].min() + 1)
    in_w = int(in_arr[:, 1].max() - in_arr[:, 1].min() + 1)
    if in_h <= in_w:
        return None

    for r, c in combined_in:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
    if any(pt in combined_in for pt in ctrl_pts):
        return None

    density = len(combined_in) / (in_h * in_w)
    if density < 0.38 or density > 0.62:
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined_in:
        inp[r, c] = int(fg)
    for cr, cc in ctrl_pts:
        inp[cr, cc] = int(ctrl_color)

    out = _render_cut_and_slide(combined_rot, H, W, bg, fg, ctrl_color, ctrl_pts, E, c_split, r_split)
    variant_out = main(inp)
    if not np.array_equal(variant_out, out):
        return None

    legacy_out = _legacy_main(inp)
    diff = int(np.count_nonzero(legacy_out != out))
    if diff < 6:
        return None
    if np.array_equal(legacy_out, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
