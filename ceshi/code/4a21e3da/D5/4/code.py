# Variant of 4a21e3da
# Dimension: D5 — 切割后片段的变换规则 (transform)
# Change: 与控制边接壤的区域滑动到对应角落；不接壤的区域消失（erase）... → 消失区域移至对侧角落
#
# Design analysis:
# source_task: none
# variant:
#   concrete_behavior:
#     - 输入仍然是一个居中的单色连通精灵，外加 2 个边界控制像素；一个控制像素给出竖切线，另一个给出横切线。
#     - 射线仍然只在背景上着色，并在切线上的 fg 像素处间歇穿过；切线上的 fg 保持原位。
#     - 原题在 2 条切线时会把精灵分成 4 个象限：其中 3 个象限与控制边接壤，会滑到对应角落；剩下 1 个对角象限不接壤，原题会删除它。
#     - 本变体把这个“原本删除”的对角象限保留下来，并移到“没有控制像素的那一对边”组成的对侧角落。
#     - 由于 2 切线场景下该象限本来就位于那个对侧象限，所以它的目标角落等价于“按自身象限去对应角落”；区别只在于原题删掉它，本变体保留它。
#   behavior_trace:
#     - 例：控制边为 top + left，切线把精灵切成 TL/TR/BL/BR 四块。
#     - 原题：TL->左上，TR->右上，BL->左下，BR 删除。
#     - 变体：TL->左上，TR->右上，BL->左下，BR->右下。
#     - 因此真正变化的流程环节只有 stage 2 的“是否保留该区域”判定；射线绘制、切线 fg 保持、角落对齐方式都不变。
#   complexity_calibration:
#     - 如果 generator 允许大量单控制像素样例，那么输出会与原题完全一致，D5 变化不明显。
#     - 为了让人类在 2-4 个样例里看出新规则，generator 强制使用 2 个互相垂直的控制像素，并保证非接壤象限至少有 3 个 fg 像素。
#     - 这样每个样例都会出现“原题应消失、变体会保留”的角落块，视觉差异稳定可见。
#   implicit_assumptions_review:
#     - `ctrl`/`fg` 的识别方式不变：边界上唯一的非背景色仍然是控制色。
#     - 切线识别不变：仍然最多一条竖切线和一条横切线。
#     - 旧代码中真正依赖“旧 D5=删除”的关键行是
#       `if not (touches_edges(hs, vs) & E): continue`
#       这一行默认非接壤区域不应出现在输出。变体下这个前提失效，必须移除。
#     - `pick_dest()` 在双切线下本来就返回区域自身的象限角落，因此无需更改角落映射，只需更改“是否保留”。
#     - 生成器里原先的 `vanish` 区域只生成输入、不生成输出，这同样依赖旧 D5，现改为既生成输入也生成输出。
#   placement_legality:
#     - 输入构造继续把各块压向切线侧，使它们与切线 fg 有机会连成一个中心精灵。
#     - 输出构造继续把每块按其 bbox 冲到目标角落；各角落彼此分离，不会互相重叠。
#     - 每块都保持在自己那一侧的半平面中，因此不会覆盖切线 fg，也不会改变射线终点。
#   visual_observability:
#     - 所有训练样例都含 2 条切线和 1 个“对角块”，输出会出现 4 个角落块而不是 3 个。
#     - 该变化只依赖空间位置，不依赖颜色数值或隐藏排序；从少量样例可直接肉眼归纳。
#   cleanliness:
#     - 仍然只使用 bg/fg/ctrl 三色。
#     - 角落块、中心切线 fg、控制射线三类元素位置分离，避免叠层混乱。
#   new_solver_assumptions_to_support:
#     - generator 必须始终生成双控制像素，且非接壤象限非空，否则 D5 不可见。
#     - generator 必须保证该对角块在输入中与主体连通，并在输出角落合法落位；下面的构造和 `main(inp)==out` 验证共同保证这一点。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, get_bg, is_contiguous


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


def _pick_dest(hs: str, vs: str, r_split: int | None, c_split: int | None, edges: set[str]) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif ("top" in edges) and ("bottom" not in edges):
        dh = "T"
    elif ("bottom" in edges) and ("top" not in edges):
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif ("left" in edges) and ("right" not in edges):
        dv = "L"
    elif ("right" in edges) and ("left" not in edges):
        dv = "R"
    else:
        dv = "L"
    return str(dh), str(dv)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and slide every cut piece to a corner.

    Variant D5:
    - Regions touching a control edge still slide to their same-side corner.
    - Regions that would have vanished in the base task are also kept, and move to the
      opposite corner determined by the missing control edges.
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

    edges: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            edges.add("top")
            c_split = c
        elif r == H - 1:
            edges.add("bottom")
            c_split = c
        elif c == 0:
            edges.add("left")
            r_split = r
        elif c == W - 1:
            edges.add("right")
            r_split = r

    out = np.full((H, W), bg, dtype=int)

    # Stage 1: draw rays only on background cells.
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    split_fg = set()
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            out[r, c] = fg
            split_fg.add((r, c))

    # Stage 2: collect fg pixels by region relative to the split lines.
    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if (r, c) in split_fg:
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    # Old task deleted regions that failed `_touches_edges(...) & edges`.
    # Variant D5 keeps every region; the destination corner already distinguishes
    # same-side pieces from opposite-side pieces.
    for (hs, vs), pts in regions.items():
        dh, dv = _pick_dest(hs, vs, r_split, c_split, edges)
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
                out[rr, cc] = fg

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    """Grow a connected piece up to max_h x max_w targeting target_px pixels."""
    cells = {(0, 0)}
    for _ in range(target_px - 1):
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
    return frozenset((r - min(rs), c - min(cs)) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    # Force two perpendicular controls so the D5 difference is always visible.
    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    edges = set(ctrl_edges)
    c_split = random.randint(int(W * 0.4), int(W * 0.6))
    r_split = random.randint(int(H * 0.4), int(H * 0.6))
    ctrl_pts = [
        (0 if "top" in edges else H - 1, c_split),
        (r_split, 0 if "left" in edges else W - 1),
    ]

    def touches_control(hs: str, vs: str) -> bool:
        return bool(_touches_edges(hs, vs) & edges)

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    touching_regions = [reg for reg in all_regions if touches_control(*reg)]
    opposite_regions = [reg for reg in all_regions if reg not in touching_regions]
    if len(touching_regions) != 3 or len(opposite_regions) != 1:
        return None
    opposite_reg = opposite_regions[0]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    frac = random.uniform(0.4, 0.6)
    sprite_c0 = c_split - int(frac * sprite_w)
    sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
    if sprite_c0 > W - 5 - sprite_w + 1:
        return None
    if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
        return None

    frac = random.uniform(0.4, 0.6)
    sprite_r0 = r_split - int(frac * sprite_h)
    sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
    if sprite_r0 > H - 5 - sprite_h + 1:
        return None
    if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
        return None

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs == "T" else r_split + 1
        r1 = r_split - 1 if hs == "T" else sprite_r1
        c0 = sprite_c0 if vs == "L" else c_split + 1
        c1 = c_split - 1 if vs == "L" else sprite_c1
        return r0, r1, c0, c1

    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None
        if (r1 - r0 + 1) * (c1 - c0 + 1) < 3:
            return None

    split_pixels: set[tuple[int, int]] = set()
    v_rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
    h_cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
    if len(v_rows) < 2 or len(h_cols) < 2:
        return None

    v_count = random.randint(2, min(4, len(v_rows)))
    h_count = random.randint(2, min(4, len(h_cols)))
    for r in random.sample(v_rows, v_count):
        split_pixels.add((r, c_split))
    for c in random.sample(h_cols, h_count):
        split_pixels.add((r_split, c))

    n_split = len(split_pixels)
    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)
    budget_pieces = target_total - n_split
    if budget_pieces < 12:
        return None

    opposite_r0, opposite_r1, opposite_c0, opposite_c1 = sub_bbox(*opposite_reg)
    opposite_area = (opposite_r1 - opposite_r0 + 1) * (opposite_c1 - opposite_c0 + 1)
    opposite_budget_hi = min(7, opposite_area, max(4, budget_pieces // 3))
    if opposite_budget_hi < 3:
        return None
    opposite_budget = random.randint(3, opposite_budget_hi)
    budget_touching = budget_pieces - opposite_budget
    if budget_touching < len(touching_regions) * 3:
        return None

    piece_budgets: dict[tuple[str, str], int] = {}
    remaining = budget_touching
    for idx, reg in enumerate(touching_regions):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if idx == len(touching_regions) - 1:
            px = remaining
        else:
            avg = remaining // (len(touching_regions) - idx)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (len(touching_regions) - idx - 1) * 3))
        piece_budgets[reg] = px
        remaining -= px

    if remaining != 0 or any(piece_budgets[reg] < 3 for reg in touching_regions):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for reg in touching_regions:
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, r_split, c_split, edges)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[reg]

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1
        inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        inp_pts = frozenset((r + inp_r0, c + inp_c0) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None

        input_pieces[reg] = inp_pts
        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in inp_pts)

        for r, c in out_pts:
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None
        output_pieces[reg] = out_pts

    opp_hs, opp_vs = opposite_reg
    opp_r0, opp_r1, opp_c0, opp_c1 = sub_bbox(opp_hs, opp_vs)
    opp_h = opp_r1 - opp_r0 + 1
    opp_w = opp_c1 - opp_c0 + 1
    opposite_piece = _grow_piece(opp_h, opp_w, opposite_budget)
    if len(opposite_piece) < 3:
        return None

    opp_ph = max(r for r, _ in opposite_piece) + 1
    opp_pw = max(c for _, c in opposite_piece) + 1
    opp_inp_r0 = (opp_r1 - opp_ph + 1) if opp_hs == "T" else opp_r0
    opp_inp_c0 = (opp_c1 - opp_pw + 1) if opp_vs == "L" else opp_c0
    opposite_input = frozenset((r + opp_inp_r0, c + opp_inp_c0) for r, c in opposite_piece)

    all_inp_fg = set(split_pixels)
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)

    for r, c in opposite_input:
        if r < 4 or r > H - 5 or c < 4 or c > W - 5:
            return None
        if (opp_vs == "L" and c >= c_split) or (opp_vs == "R" and c <= c_split):
            return None
        if (opp_hs == "T" and r >= r_split) or (opp_hs == "B" and r <= r_split):
            return None
        if (r, c) in all_inp_fg:
            return None
    input_pieces[opposite_reg] = opposite_input
    all_inp_fg |= set(opposite_input)

    dh, dv = _pick_dest(opp_hs, opp_vs, r_split, c_split, edges)
    opp_rs = [r for r, _ in opposite_input]
    opp_cs = [c for _, c in opposite_input]
    dr = -min(opp_rs) if dh == "T" else (H - 1 - max(opp_rs))
    dc = -min(opp_cs) if dv == "L" else (W - 1 - max(opp_cs))
    opposite_output = frozenset((r + dr, c + dc) for r, c in opposite_input)
    for r, c in opposite_output:
        if (opp_vs == "L" and c >= c_split) or (opp_vs == "R" and c <= c_split):
            return None
        if (opp_hs == "T" and r >= r_split) or (opp_hs == "B" and r <= r_split):
            return None
    output_pieces[opposite_reg] = opposite_output

    fg_arr = np.array(list(all_inp_fg))
    if fg_arr.size == 0:
        return None
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in all_inp_fg:
        inp[r, c] = fg
    for cr, cc in ctrl_pts:
        inp[cr, cc] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[r, c] = fg
    for r, c in split_pixels:
        out[r, c] = fg

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
        out[cr, cc] = ctrl_color

    final_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    final_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    actual_density = len(all_inp_fg) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
