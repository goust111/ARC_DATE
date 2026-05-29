# Variant of 4a21e3da
# Dimension: D10 — 片段目标位置 (meta)
# Change: 片段外接矩形紧贴对应角落放置（flush with corner）... → 片段保持与分割线的距离

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
Design analysis for the D10 variant
===================================

source_task:
  referenced_task: none

variant_spec:
  core_change:
    old: 每个保留片段被平移到目标角落，bbox 直接贴住目标边/角。
    new: 若某一轴存在分割线，则该轴不再贴边；片段在输出中到目标边的留白，
      等于该片段输入 bbox 的“外侧边”到分割线的距离。没有分割线的轴保持原来的贴边逻辑。
  precise_distance_rule:
    - 左侧片段: output.cmin = c_split - input.cmin - 1
    - 右侧片段: output 右边距 = input.cmax - c_split - 1
    - 上侧片段: output.rmin = r_split - input.rmin - 1
    - 下侧片段: output 下边距 = input.rmax - r_split - 1
    - 这里保留的是 bbox 外侧边到分割线的距离，而不是靠近分割线那一侧的距离。
      这是因为输入精灵要保持 4-连通，保留片段通常必须贴着分割线；若保留“近侧距离”，
      绝大多数样例会退化成原题的贴角结果。
  small_trace:
    setup:
      grid_width: 18
      vertical_split: c_split = 8
      kept_left_piece_bbox: cmin = 4, cmax = 7
    old_solver:
      output.cmin = 0
      meaning: 左片段直接贴左边框。
    new_solver:
      output.cmin = 8 - 4 - 1 = 3
      meaning: 左片段左边保留 3 格空白；这个 3 正好是输入里片段外侧边到分割线的距离。
    difference:
      - 原题输出是“冲到角落”。
      - 变体输出是“朝角落移动，但停在一个与分割线等距的位置”。

complexity_calibration:
  - 保留原题的 1 或 2 个控制像素、切线规则、消失规则、split-line 像素保留规则。
  - 生成器继续偏向 2-control，因为这能同时展示横向和纵向的等距平移。
  - 为避免样例退化成与原题几乎无差别，凡是存在竖直分割线的保留片段，bbox 宽度至少为 2；
    凡是存在水平分割线的保留片段，bbox 高度至少为 2。这样输出边距至少为 1，可肉眼观察。

hidden_assumptions_audit_of_original_main:
  - border color detection:
      依赖“边界上唯一的非背景色就是 ctrl”。变体未改颜色体系，这一前提仍成立。
  - split-line inference:
      依赖“至多一条竖切线和一条横切线”。变体未改控制像素布局，这一前提仍成立。
  - ray drawing:
      依赖“ray 只在背景格上着色，遇到 fg 断续穿过直到最远 fg”。D10 不改这一机制。
  - region grouping:
      依赖“去掉 split-line fg 后，其余 fg 可按上下/左右区域分类”。变体未改切分拓扑。
  - old movement code:
      原代码的 dr/dc 直接把 bbox 吸到边界：
        dr = 0-rmin or H-1-rmax
        dc = 0-cmin or W-1-cmax
      这显式依赖旧题的 flush-with-corner 目标；在 D10 下已不成立，必须改成 distance-preserving 公式。

placement_legality_requirements:
  - 输入精灵仍必须单连通、离边界至少 4 格、split-line 穿过其内部。
  - 保留片段在输入中仍尽量贴近分割线，以便和 split-line fg 连通。
  - 输出片段平移后仍必须保持在自己原来的 split 侧，不得跨线，不得与其他片段重叠。
  - 生成器用 main(inp) == out 反验，并额外限制 kept piece 在相关轴上至少 2 格厚，确保变体可见。

visual_observability:
  - 人类可直接比较“片段在输入里从分割线向外伸出去多少格”与“输出里该片段离目标边保留多少格”。
  - 2-control 样例中，同一张图会出现 3 个保留片段，能重复支撑“边距来自各自片段相对 split 的跨度”。
  - 不使用任何颜色排序、数值大小等不可见信息。

new_solver_assumptions_and_generator_guarantees:
  - 若某轴有 split，则该轴的保留片段必须确实位于 split 的对应一侧。生成器通过区域 bbox 与放置校验保证。
  - 若想让新规则显著不同于原题，则相关轴上的片段 bbox 不能退化为 1 格厚。生成器显式约束最小宽/高。
  - 若某轴没有 split，则 solver 在该轴仍用旧的贴边规则；生成器不会依赖不存在的 split 去编码新信息。
"""


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


def _axis_shift(minv: int, maxv: int, size: int, split: int | None, dest: str) -> int:
    if split is None:
        return int(-minv) if dest in ("T", "L") else int((size - 1) - maxv)

    if dest in ("T", "L"):
        target_min = int(split - minv - 1)
        return int(target_min - minv)

    target_max = int((size - 1) - (maxv - split - 1))
    return int(target_max - maxv)


def _variant_output_points(
    pts: list[tuple[int, int]] | set[tuple[int, int]] | frozenset[tuple[int, int]],
    dh: str,
    dv: str,
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = _axis_shift(rmin, rmax, H, r_split, dh)
    dc = _axis_shift(cmin, cmax, W, c_split, dv)
    return frozenset((int(r + dr), int(c + dc)) for r, c in pts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and slide pieces outward.

    D10 variant:
    - On axes that have a split line, kept pieces do not end flush with the target corner.
    - Instead, the output margin to the target border equals the input bbox's outer-side distance to the split line.
    - On axes without a split line, keep the original flush-to-edge behavior.
    """
    g = np.array(grid, dtype=int)
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
        if int(r) == 0:
            E.add("top")
            c_split = int(c)
        elif int(r) == int(H - 1):
            E.add("bottom")
            c_split = int(c)
        elif int(c) == 0:
            E.add("left")
            r_split = int(r)
        elif int(c) == int(W - 1):
            E.add("right")
            r_split = int(r)

    out = np.full((H, W), int(bg), dtype=int)

    for r, c in ctrl_pts:
        if int(r) == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, int(stop) + 1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(r) == int(H - 1):
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(int(H - 1), int(stop) - 1, -1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(c) == 0:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, int(stop) + 1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)
        elif int(c) == int(W - 1):
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(int(W - 1), int(stop) - 1, -1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if int(r) < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if int(c) < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue

        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        for rr, cc in _variant_output_points(pts, dh, dv, H, W, r_split, c_split):
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int, min_h: int = 1, min_w: int = 1) -> frozenset[tuple[int, int]] | None:
    """Grow a connected piece with bounded bbox and optional minimum bbox size."""
    attempts = 40
    for _ in range(attempts):
        cells = {(0, 0)}
        for _step in range(target_px - 1):
            neighbors = set()
            for r, c in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
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
        norm = frozenset((r - min(rs), c - min(cs)) for r, c in cells)
        ph = int(max(r for r, _ in norm) + 1)
        pw = int(max(c for _, c in norm) + 1)
        if len(norm) >= target_px and ph >= min_h and pw >= min_w:
            return norm
    return None


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
            ctrl_pts.append((0 if edge == "top" else H - 1, c_split))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((r_split, 0 if edge == "left" else W - 1))
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

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = c_split - int(frac * sprite_w)
        sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
        if sprite_c0 > W - 5 - sprite_w + 1:
            return None
        if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
            return None
    else:
        hi = W - 5 - sprite_w + 1
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = r_split - int(frac * sprite_h)
        sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
        if sprite_r0 > H - 5 - sprite_h + 1:
            return None
        if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
            return None
    else:
        hi = H - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

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
            split_pixels.add((r, int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), c))

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
    for idx, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if idx == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - idx)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - idx - 1) * 3))
        piece_budgets.append(int(px))
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[idx]

        min_h = 2 if has_h else 1
        min_w = 2 if has_v else 1
        piece = _grow_piece(sub_h, sub_w, target_px, min_h=min_h, min_w=min_w)
        if piece is None or len(piece) < 3:
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

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        inp_h = int(max(inp_rs) - min(inp_rs) + 1)
        inp_w = int(max(inp_cs) - min(inp_cs) + 1)

        if has_h and inp_h < 2:
            return None
        if has_v and inp_w < 2:
            return None

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None

        input_pieces[reg] = inp_pts
        out_pts = _variant_output_points(inp_pts, dh, dv, H, W, r_split, c_split)

        out_rs = [r for r, _ in out_pts]
        out_cs = [c for _, c in out_pts]
        for r, c in out_pts:
            if not (0 <= r < H and 0 <= c < W):
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        if has_v:
            if vs == "L" and min(out_cs) < 1:
                return None
            if vs == "R" and max(out_cs) > W - 2:
                return None
        if has_h:
            if hs == "T" and min(out_rs) < 1:
                return None
            if hs == "B" and max(out_rs) > H - 2:
                return None
        output_pieces[reg] = out_pts

    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    all_inp_fg |= split_pixels
    if not all_inp_fg:
        return None

    fg_arr = np.array(list(all_inp_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
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
            if piece is None or len(piece) < 3:
                return None
            pph = max(r for r, _ in piece) + 1
            ppw = max(c for _, c in piece) + 1
            vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
            vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (r, c) in all_inp_fg:
                    return None
            vanish_fg |= vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[r, c] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = fg
    for r, c in split_pixels:
        out[int(r), int(c)] = fg

    for cr, cc in ctrl_pts:
        if cr == 0:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(max(fg_rows) + 1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cr == H - 1:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(H - 1, min(fg_rows) - 1, -1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cc == 0:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(max(fg_cols) + 1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color
        elif cc == W - 1:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(W - 1, min(fg_cols) - 1, -1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = ctrl_color

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
