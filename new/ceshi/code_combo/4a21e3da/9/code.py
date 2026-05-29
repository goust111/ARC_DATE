# Combo variant of 4a21e3da (index 9)
# combo_spec: D5/2+D7/2
# Name: 片段旋转+先变换再切割
#
# Design analysis for the fused variant
# ------------------------------------
# a) 具体化与行为追踪
# - 原题：直接在输入里看到的 fg 精灵上按 ctrl 射线切割；保留区域平移到角落，不保留的区域消失。
# - 本融合版：
#   1. 先把整块 fg 精灵在其 tight bbox 内顺时针旋转 90 度，得到“中间态精灵”。
#   2. 再用 ctrl 定义的竖/横 split line 去切这个中间态精灵，split line 上的 fg 保留原位。
#   3. 每个保留下来的区域不是直接平移，而是先按“目标角与活动控制边的方位关系”做
#      90 CCW / 90 CW / 180 旋转，再把旋转后的 bbox 贴到目标角。
# - 具体例子：ctrl 在 top 和 left，切成 TL/TR/BL/BR 四象限。原题会把 TL/TR/BL 三块
#   原样滑到左上/右上/左下，BR 消失；本题先把整个精灵旋成另一个朝向，再按该朝向切，
#   之后 TL->左上会 180 度旋转，TR->右上会 90 CW，BL->左下会 90 CCW。于是输出的
#   split-line 残留、射线终点、角落碎片轮廓都对应“先整体旋过的精灵”，而角落碎片的
#   局部朝向又额外发生变化。
#
# b) 复杂度校准
# - 整体变换只固定一种：顺时针 90 度；片段旋转只允许 {90 CW, 90 CCW, 180}，自由度有限。
# - generator 仍会稳定产出一批 2-control case，因此样例集中会反复出现“一个 180 度片段 +
#   两个 90 度片段”的局部旋转模式。
# - generator 拒绝：整体旋转不可见、片段旋转不可见、combo 输出与 D7-only / D5-only /
#   原题输出过近或完全相同的样本，避免退化成单维变体。
#
# c) 隐式假设审查（围绕 main）
# - 原题所有 `g == fg` 的使用都默认“输入里看到的 fg 就是待切对象”；本题不成立，必须先
#   构造 `work_pts = rotate_clockwise(fg_pts)`，后续 split-line、ray、regions 都只看 work_pts。
# - 原题“split-line fg stays in place”依赖的是当前被切对象；本题里必须保留 work_pts 上落在
#   split line 的像素，而不是输入原像素。
# - 原题片段落角使用绝对平移；本题不成立，因为局部旋转会改变 bbox。solver 必须先把片段裁到
#   本地 bbox、旋转 mask，再将旋转后的 bbox 贴到角落。
# - `_rotation_k` 默认：保留区域的目标角至少共享一条活动控制边；这是原角落选择逻辑保证的，
#   generator 维持该逻辑。
# - 输出覆盖安全性默认角落碎片互不冲突；generator 显式拒绝跨越 split line 的旋转片段。
#
# d) 放置合法性
# - generator 先构造“整体旋转后、待切割”的中间态精灵，再整体逆时针旋回输入。
# - 使用 `safe_side = max(sprite_h, sprite_w)` 预留旋回包络，保证逆旋后的输入仍离边界 >= 4。
# - 保留片段在中间态里仍各自待在唯一 side/quadrant；输出中的旋转片段必须完全留在目标 split
#   一侧；消失片段只出现在输入与中间态中，不出现在输出。
#
# e) 视觉可观察性
# - 中间态维持原题宽形 bbox（约 5-8 x 8-10），逆旋后的输入通常更高更窄；这让“先整体旋转”
#   能从少量样例里被肉眼注意到。
# - 保留片段强制对其要求的 krot 非旋转不变，因此角落碎片形状会明显改变，而不是只换位置。
# - 生成时要求 combo 输出同时区别于 D7-only 和 D5-only 输出，因此两条变化链都必须起作用。
#
# f) 可视化整洁度
# - 仍只有 3 种颜色：bg/fg/ctrl，没有额外标记层。
# - 输出仍是三层结构：ctrl 射线、split-line fg、角落碎片。局部旋转只改变碎片轮廓，不引入叠色。
#
# g) 变化的可推断性
# - 无颜色映射。解题者只需从形状与位置推断“先整体旋转，再切，再按目标角局部旋转”。
#
# New main() assumptions and generator guarantees
# -----------------------------------------------
# - 假设输入 fg 是一块单连通精灵；generator 对逆旋后的输入做连通性检查。
# - 假设 ctrl 是边界上唯一非 bg 颜色；generator 保持 fg 距边界至少 4 格。
# - 假设整体旋转后的中间态在 split line 上仍有 2-4 个 fg；generator 直接在中间态上构造它们。
# - 假设局部旋转后的角落碎片不会越过 split line；generator 在输出侧逐片验证。

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _bbox_of_points(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
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


def _touches_edges(hs: str, vs: str) -> set[str]:
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


def _pick_dest(hs: str, vs: str, E: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
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


def _rotation_k(dh: str, dv: str, E: set[str]) -> int:
    v_match = (dh == "T" and "top" in E) or (dh == "B" and "bottom" in E)
    h_match = (dv == "L" and "left" in E) or (dv == "R" and "right" in E)
    count = int(v_match) + int(h_match)
    assert count in (1, 2)
    if count == 2:
        return 2
    if v_match:
        if dh == "T":
            return 1 if dv == "L" else 3
        return 3 if dv == "L" else 1
    if dv == "L":
        return 3 if dh == "T" else 1
    return 1 if dh == "T" else 3


def _rotate_local_mask(pts: list[tuple[int, int]] | set[tuple[int, int]], krot: int) -> np.ndarray:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, cmin = int(min(rs)), int(min(cs))
    h = int(max(rs) - rmin + 1)
    w = int(max(cs) - cmin + 1)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[int(r - rmin), int(c - cmin)] = True
    return np.rot90(mask, int(krot))


def _place_rotated_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    krot: int,
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    rot_mask = _rotate_local_mask(pts, int(krot))
    ph, pw = map(int, rot_mask.shape)
    base_r = 0 if dh == "T" else int(H - ph)
    base_c = 0 if dv == "L" else int(W - pw)
    return {(int(base_r + r), int(base_c + c)) for r, c in np.argwhere(rot_mask)}


def _place_translated_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int(-rmin) if dh == "T" else int((H - 1) - rmax)
    dc = int(-cmin) if dv == "L" else int((W - 1) - cmax)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


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


def _render_cut_transform(
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
    *,
    rotate_pieces: bool,
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
        (int(r), int(c))
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

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        if rotate_pieces:
            krot = _rotation_k(dh, dv, E)
            out_pts = _place_rotated_to_corner(pts, int(krot), dh, dv, int(H), int(W))
        else:
            out_pts = _place_translated_to_corner(pts, dh, dv, int(H), int(W))
        for rr, cc in out_pts:
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _solve_variant(grid: np.ndarray, *, rotate_first: bool, rotate_pieces: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, ctrl, ctrl_pts, E, c_split, r_split = _infer_colors_and_controls(g)
    sprite_pts = {(int(r), int(c)) for r, c in np.argwhere(g == int(fg))}
    if rotate_first:
        sprite_pts = _rotate_points_clockwise_tl(sprite_pts)
    return _render_cut_transform(
        sprite_pts,
        int(H),
        int(W),
        int(bg),
        int(fg),
        int(ctrl),
        ctrl_pts,
        E,
        c_split,
        r_split,
        rotate_pieces=rotate_pieces,
    )


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, rotate_first=False, rotate_pieces=False)


def _d5_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, rotate_first=False, rotate_pieces=True)


def _d7_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(grid, rotate_first=True, rotate_pieces=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Rotate the whole fg sprite 90 degrees clockwise in its tight bbox.
    2. Cut that rotated sprite by the border-control rays.
    3. Rotate each kept piece by a corner-dependent quarter/half turn, then place it in the corner.
    """
    return _solve_variant(grid, rotate_first=True, rotate_pieces=True)


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
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _mask_from_piece(piece: frozenset[tuple[int, int]]) -> np.ndarray:
    ph = int(max(r for r, _ in piece) + 1)
    pw = int(max(c for _, c in piece) + 1)
    mask = np.zeros((ph, pw), dtype=bool)
    for r, c in piece:
        mask[int(r), int(c)] = True
    return mask


def _mask_is_contiguous(H: int, W: int, pts: set[tuple[int, int]]) -> bool:
    mask = np.zeros((H, W), dtype=int)
    for r, c in pts:
        mask[int(r), int(c)] = 1
    return bool(is_contiguous(mask))


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    combo = main(inp)
    d7_only = _d7_only_main(inp)
    d5_only = _d5_only_main(inp)
    legacy = _legacy_main(inp)
    return {
        "diff_vs_d7_only": int(np.count_nonzero(combo != d7_only)),
        "diff_vs_d5_only": int(np.count_nonzero(combo != d5_only)),
        "diff_vs_legacy": int(np.count_nonzero(combo != legacy)),
    }


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

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)
    safe_side = int(max(sprite_h, sprite_w))
    max_r0 = int(H - 5 - safe_side + 1)
    max_c0 = int(W - 5 - safe_side + 1)
    if max_r0 < 4 or max_c0 < 4:
        return None

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split - int(frac * sprite_w))
        sprite_c0 = max(4, min(int(sprite_c0), int(max_c0)))
        if int(c_split) <= int(sprite_c0) or int(c_split) >= int(sprite_c0 + sprite_w - 1):
            return None
    else:
        sprite_c0 = random.randint(4, int(max_c0))

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
        sprite_r0 = max(4, min(int(sprite_r0), int(max_r0)))
        if int(r_split) <= int(sprite_r0) or int(r_split) >= int(sprite_r0 + sprite_h - 1):
            return None
    else:
        sprite_r0 = random.randint(4, int(max_r0))

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs in ("T", "A") else int(r_split) + 1)
        r1 = int((int(r_split) - 1 if hs == "T" else int(sprite_r1)) if has_h else int(sprite_r1))
        c0 = int(sprite_c0 if vs in ("L", "A") else int(c_split) + 1)
        c1 = int((int(c_split) - 1 if vs == "L" else int(sprite_c1)) if has_v else int(sprite_c1))
        return int(r0), int(r1), int(c0), int(c1)

    def output_caps(dh: str, dv: str) -> tuple[int, int]:
        out_h_cap = int(H if not has_h else (int(r_split) if dh == "T" else H - int(r_split) - 1))
        out_w_cap = int(W if not has_v else (int(c_split) if dv == "L" else W - int(c_split) - 1))
        return int(out_h_cap), int(out_w_cap)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        rows = [
            int(r)
            for r in range(int(sprite_r0), int(sprite_r1) + 1)
            if r_split is None or int(r) != int(r_split)
        ]
        n = random.randint(2, 4)
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        cols = [
            int(c)
            for c in range(int(sprite_c0), int(sprite_c1) + 1)
            if c_split is None or int(c) != int(c_split)
        ]
        n = random.randint(2, 4)
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = int(len(split_pixels))
    area = int(sprite_h * sprite_w)
    target_total = int(random.uniform(0.39, 0.60) * area)

    n_kept = int(len(kept))
    budget_pieces = int(target_total - n_split)
    if budget_pieces < n_kept * 3:
        return None

    if vanish:
        max_vanish = int(max(3, min(7, budget_pieces // 3)))
        if max_vanish < 3:
            return None
        vanish_budget = random.randint(3, int(max_vanish))
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for i, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if i == len(kept) - 1:
            px = int(remaining)
        else:
            avg = int(remaining // (len(kept) - i))
            lo_px = max(3, int(avg - 2))
            hi_px = min(int(max_area), int(avg + 2))
            if lo_px > hi_px:
                return None
            px = random.randint(int(lo_px), int(hi_px))
        px = max(3, min(int(px), int(max_area), int(remaining - (len(kept) - i - 1) * 3)))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    transformed_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        krot = _rotation_k(dh, dv, E)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])
        out_h_cap, out_w_cap = output_caps(dh, dv)

        accepted = False
        for _piece_try in range(100):
            piece = _grow_piece(sub_h, sub_w, target_px)
            if len(piece) < 3:
                continue
            mask = _mask_from_piece(piece)
            ph, pw = map(int, mask.shape)
            if ph < 2 or pw < 2:
                continue
            if bool(np.array_equal(mask, np.rot90(mask, int(krot)))):
                continue

            rot_mask = np.rot90(mask, int(krot))
            rot_h, rot_w = map(int, rot_mask.shape)
            if rot_h > int(out_h_cap) or rot_w > int(out_w_cap):
                continue

            if has_h:
                inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
            else:
                split_rows = sorted(r for r, c in split_pixels if c == int(c_split))
                if split_rows:
                    mid = int((split_rows[0] + split_rows[-1]) // 2)
                    inp_r0 = max(int(r0), min(int(mid - ph // 2), int(r1 - ph + 1)))
                else:
                    inp_r0 = int(r0)

            if has_v:
                inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
            else:
                split_cols = sorted(c for r, c in split_pixels if r == int(r_split))
                if split_cols:
                    mid = int((split_cols[0] + split_cols[-1]) // 2)
                    inp_c0 = max(int(c0), min(int(mid - pw // 2), int(c1 - pw + 1)))
                else:
                    inp_c0 = int(c0)

            inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

            ok = True
            for r, c in inp_pts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    ok = False
                    break
                if has_v and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                    ok = False
                    break
                if has_h and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                    ok = False
                    break
            if not ok:
                continue

            out_pts = _place_rotated_to_corner(inp_pts, int(krot), dh, dv, int(H), int(W))
            for r, c in out_pts:
                if has_v and ((dv == "L" and c >= int(c_split)) or (dv == "R" and c <= int(c_split))):
                    ok = False
                    break
                if has_h and ((dh == "T" and r >= int(r_split)) or (dh == "B" and r <= int(r_split))):
                    ok = False
                    break
            if not ok:
                continue

            translated_pts = _place_translated_to_corner(inp_pts, dh, dv, int(H), int(W))
            if out_pts == translated_pts:
                continue

            transformed_pieces[reg] = inp_pts
            accepted = True
            break

        if not accepted:
            return None

    all_transformed_fg: set[tuple[int, int]] = set(split_pixels)
    for pts in transformed_pieces.values():
        if all_transformed_fg & set(pts):
            return None
        all_transformed_fg |= set(pts)
    if not all_transformed_fg:
        return None

    fg_arr = np.array(list(all_transformed_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10 or aw <= ah:
        return None
    if not _mask_is_contiguous(H, W, all_transformed_fg):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh = int(vr1 - vr0 + 1)
            vw = int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = min(int(vanish_budget), int(vh * vw))
            if max_px < 3:
                return None

            accepted = False
            for _piece_try in range(80):
                piece = _grow_piece(vh, vw, max_px)
                if len(piece) < 3:
                    continue
                pph = int(max(r for r, _ in piece) + 1)
                ppw = int(max(c for _, c in piece) + 1)
                vr_off = int(vr1 - pph + 1) if vhs == "T" else int(vr0)
                vc_off = int(vc1 - ppw + 1) if vvs == "L" else int(vc0)
                vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
                ok = True
                for r, c in vpts:
                    if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                        ok = False
                        break
                    if (r, c) in all_transformed_fg:
                        ok = False
                        break
                if not ok:
                    continue
                vanish_fg = vpts
                accepted = True
                break
            if not accepted:
                return None

    combined_rot = set(all_transformed_fg) | set(vanish_fg)
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

    density = len(combined_in) / float(in_h * in_w)
    if density < 0.38 or density > 0.62:
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined_in:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl_color)

    out = main(inp)
    combo_ref = _solve_variant(inp, rotate_first=True, rotate_pieces=True)
    if not bool(np.array_equal(out, combo_ref)):
        return None

    d7_only = _d7_only_main(inp)
    d5_only = _d5_only_main(inp)
    legacy = _legacy_main(inp)

    diff_vs_d7 = int(np.count_nonzero(out != d7_only))
    diff_vs_d5 = int(np.count_nonzero(out != d5_only))
    diff_vs_legacy = int(np.count_nonzero(out != legacy))

    if diff_vs_d7 < 6 or diff_vs_d5 < 6 or diff_vs_legacy < 8:
        return None
    if bool(np.array_equal(out, d7_only)) or bool(np.array_equal(out, d5_only)) or bool(np.array_equal(out, legacy)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
