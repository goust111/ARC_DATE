# Combo variant of a6f40cea (index 28)
# combo_spec: D1/4+D2/3+D8/2
# Name: 外角Z形拐点变色
#
# Design analysis
# ----------------
# variant_analysis:
#   concretization_and_trace:
#     - 保留原题的大场景：背景网格中有一个 1px 封闭矩形框，框外有若干接触 ring 的开放矩形组件，输出仍然只裁剪框内。
#     - 这里不再把 L 型外部组件映射为单拐 L，而是映射为三段式 Z/S 折线。组件仍然只接触一条 top/bottom 边和一条 left/right 边；
#       但框内路径改为 `H-V-H`：先沿 top/bottom 边走到“门柱列”，再垂直到另一触点的行，最后水平走到 left/right 触点。
#     - “门柱列”不是由端点本身决定，而是由外部开放矩形的外角水平伸出深度投影得到。
#       例如 tl 组件若外角在框左侧再伸出 2 格，则框内 gate 列就是 interior 的第 2 列；于是
#       top 触点 `(0,6)` 与 left 触点 `(3,0)` 不再连成 old-L `(0,6)->(3,6)->(3,0)`，
#       而会变成 new-Z `(0,6)->(0,2)->(3,2)->(3,0)`。
#     - 颜色也同步变化：原题双色是整条路径交替；这里融合后使用三段三色，且只在两个结构性折点处切色。
#       同一个外部组件的 path 顺序颜色是 `A | B | C`，输出内部三段也按同一顺序 `A | B | C` 着色。
#   complexity_calibration:
#     - 只使用 2-3 个主组件，并把它们固定为同一类“外部 L / 内部三段 Z”映射，避免样例中混入太多旧题语义。
#     - 外角投影深度只使用 1-3 三档，可从外部离框有多远直接读出，又能在 2-4 个样例里形成重复支撑。
#     - 每个主组件都强制使用 3 个互异颜色，保证“两个折点切色”是显眼的，而不是偶发退化为单色或双色。
#   implicit_assumptions_review:
#     - 原 solver 假设 same-color same-side 触点应聚合成整条 stripe；这里不成立，因为同一组件的两个端点仍可能映射到一条
#       top/bottom + left/right 相邻边组合，但内部不再是单肘 L，而是由外角深度控制的三段式路径。
#     - 原 `build_manhattan`/`inside_corner` 假设单个非直线组件只需要一个 elbow；这里需要两个 switch corner，
#       且路径结构由组件 bbox 的侧向伸出深度决定。
#     - 原双色逻辑依赖端点奇偶相位；这里完全失效，因为颜色切换只在两个指定折点发生，与路径长度奇偶无关。
#     - 为了让 solver 可逆，生成器保证每个主组件都是 simple path、恰有两个 ring 端点、颜色沿路径恰好分成 3 个连续色段。
#   placement_legality:
#     - 外部组件仍通过 `_can_place` / `_place_component` 检查不越界、不重叠、不与已有组件 4 邻接。
#     - 生成器先采样外角深度，再采样 top/bottom 触点列与 left/right 触点行，显式保证 gate 列严格落在两端之间，
#       所以内部 `H-V-H` 三段都为正长度，不会退化。
#     - 组件依旧分配到 `tl/tr/bl/br` 四个局部角域，配合 `used_inner` 检查，避免框内路径互相覆盖。
#   visual_observability:
#     - 解题者在输入里可以直接看到：外部开放矩形的 far corner 离框越远，输出里的中间竖段越往内移。
#     - 同时，输入外部路径本身已经呈现 `A | B | C` 三段色带；输出中的颜色切换也恰好发生在两个几何折点处，
#       因而“几何决定形状、折点决定换色”的对应关系肉眼可见。
#     - 整个规则只依赖可见几何和可见颜色分段，不依赖颜色 id 排序、连通块编号等不可视因素。
#   visual_cleanliness:
#     - 框内路径被约束在四个角域附近，不会在中心区域密集缠绕。
#     - 每个主组件只有一个中间竖段和三个大色块，比细碎交替颜色更易辨认；小干扰块仍远离 ring，不会误导成主组件。
#   color_change_inference:
#     - 每条主路径严格 3 色且三段颜色互异；第 1/3 色直接由两个 ring 端点给出，中间第 2 色由外部 detour 主体给出。
#     - 颜色切换点永远与两个几何折点重合，而不是与步数奇偶或绝对位置绑定，因此人类可从少量例子稳定归纳。
#   post_change_solver_assumptions:
#     - solver 假设每个 ring-touching 主组件是 simple path，且 degree-1 端点中有一个位于 top/bottom ring、另一个位于 left/right ring；
#       生成器按单一路径构造并只使用 L 型外部组件，保证这一点。
#     - solver 假设组件 bbox 在 left 或 right 方向一定比 ring 再多伸出至少 1 格，从而能解码 gate 列；
#       生成器的外角深度范围固定为 1-3，保证该投影存在。
#     - solver 假设沿组件路径压缩后的颜色 run 数恰好为 3；生成器只用三色连续涂装主组件，并用 oracle 校验 `main(input)==expected`。

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _walk_simple_path(
    cells: list[tuple[int, int]],
    preferred_start: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    cell_set = {tuple(map(int, p)) for p in cells}
    deg: dict[tuple[int, int], int] = {}
    for r, c in cell_set:
        deg[(r, c)] = sum((nr, nc) in cell_set for nr, nc in neighbors4(r, c))
    endpoints = [p for p, d in deg.items() if d == 1]
    assert len(endpoints) == 2
    if preferred_start in endpoints:
        start = preferred_start
    else:
        start = endpoints[0]

    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [q for q in neighbors4(*cur) if q in cell_set and q != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        path.append(cur)
    assert len(path) == len(cell_set)
    return path


def _compress_runs(colors: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    if not colors:
        return runs
    start = 0
    cur = int(colors[0])
    for idx, col in enumerate(colors[1:], start=1):
        col = int(col)
        if col != cur:
            runs.append((cur, int(start), int(idx)))
            cur = col
            start = idx
    runs.append((cur, int(start), int(len(colors))))
    return runs


def _paint_three_runs(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, int, int],
    switch_points: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    assert len(colors) == 3
    a, b, c = map(int, colors)
    p1, p2 = switch_points
    assert p1 in path and p2 in path
    i1 = int(path.index(p1))
    i2 = int(path.index(p2))
    assert i1 < i2
    for idx, (rr, cc) in enumerate(path):
        if idx < i1:
            grid[int(rr), int(cc)] = a
        elif idx < i2:
            grid[int(rr), int(cc)] = b
        else:
            grid[int(rr), int(cc)] = c


def _build_inner_z_path(
    tb_pt: tuple[int, int],
    lr_pt: tuple[int, int],
    gate_c: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    rt, ct = map(int, tb_pt)
    rl, cl = map(int, lr_pt)
    corner1 = (int(rt), int(gate_c))
    corner2 = (int(rl), int(gate_c))
    path = _join(
        _hpath(int(rt), int(ct), int(gate_c)),
        _vpath(int(rt), int(rl), int(gate_c)),
        _hpath(int(rl), int(gate_c), int(cl)),
    )
    return path, corner1, corner2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover cropped interior zigzag paths from corner-reaching traces outside the frame.

    Input structure:
    - A uniform background contains one 1px closed rectangular frame with a uniform fill.
    - Outside the frame are colored simple-path components that touch the ring at one top/bottom
      location and one left/right location, plus optional distractor blobs away from the ring.

    Rule:
    - Output only the frame interior, initialized with the fill color.
    - For each ring-touching path component, project its top/bottom touch and left/right touch
      inward to the cropped boundary.
    - Decode a gate column from how far the component's outer corner reaches left or right of the
      frame, then draw a three-segment `H-V-H` Z/S path through that gate column.
    - Paint the three path segments with the component's three ordered color runs, switching
      exactly at the two geometric bends.

    Key insight:
    The outside component carries two visible clues at once: its sideward protrusion fixes where
    the interior zigzag turns inward, and its three contiguous color runs determine the three
    output segment colors in order.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    comp_ring_actual: dict[int, list[tuple[int, int]]] = defaultdict(list)
    comp_ring_mapped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        comp_ring_actual[cid].append((int(rr), int(cc)))
        comp_ring_mapped[cid].append(map_to_out(int(rr), int(cc)))

    comp_cells: dict[int, list[tuple[int, int]]] = {}
    comp_bbox: dict[int, tuple[int, int, int, int]] = {}
    for cid in range(1, int(cn) + 1):
        pts = [tuple(map(int, p)) for p in np.argwhere(clab == int(cid))]
        if not pts:
            continue
        comp_cells[cid] = pts
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        comp_bbox[cid] = (min(rs), min(cs), max(rs), max(cs))

    for cid, ring_pts in comp_ring_actual.items():
        mapped_pts = list(dict.fromkeys(comp_ring_mapped[int(cid)]))
        if len(mapped_pts) != 2:
            continue

        tb_pts = [p for p in mapped_pts if p[0] in (0, outH - 1)]
        lr_pts = [p for p in mapped_pts if p[1] in (0, outW - 1)]
        if len(tb_pts) != 1 or len(lr_pts) != 1:
            continue
        tb_pt = tuple(map(int, tb_pts[0]))
        lr_pt = tuple(map(int, lr_pts[0]))

        actual_tb = None
        for rr, cc in ring_pts:
            if rr in (r0 - 1, r1 + 1):
                actual_tb = (int(rr), int(cc))
                break
        assert actual_tb is not None

        ordered = _walk_simple_path(comp_cells[int(cid)], preferred_start=actual_tb)
        ordered_cols = [int(g[r, c]) for r, c in ordered]
        runs = _compress_runs(ordered_cols)
        assert len(runs) == 3
        run_colors = (int(runs[0][0]), int(runs[1][0]), int(runs[2][0]))

        cr0, cc0, cr1, cc1 = comp_bbox[int(cid)]
        if lr_pt[1] == 0:
            gate_c = int((c0 - 1) - cc0)
        else:
            gate_c = int((outW - 1) - (cc1 - (c1 + 1)))
        gate_c = max(1, min(outW - 2, int(gate_c)))

        path, corner1, corner2 = _build_inner_z_path(tb_pt, lr_pt, gate_c)
        _paint_three_runs(out, path, run_colors, (corner1, corner2))

    return out


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_DEPTH_RANGE = (1, 3)
_N_COMPONENTS_RANGE = (2, 3)
_L_COMPONENT_TRIES = 50
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 500


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != bg:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, int, int] | tuple[int, ...],
    bg: int,
    switch_points: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    if len(colors) == 1:
        rs, cs = zip(*path)
        grid[rs, cs] = colors[0]
    else:
        assert len(colors) == 3 and switch_points is not None
        _paint_three_runs(grid, path, (int(colors[0]), int(colors[1]), int(colors[2])), switch_points)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _choose_endpoint_params(kind: str, H: int, W: int, gate_c: int) -> tuple[int, int]:
    row_span = max(2, min(H - 2, H // 3 + 1))
    col_span = max(2, min(W - 2, W // 3 + 1))

    if kind == "tl":
        r = random.randint(1, min(H - 2, row_span))
        c = random.randint(gate_c + 1, min(W - 2, gate_c + col_span))
        return int(r), int(c)
    if kind == "tr":
        r = random.randint(1, min(H - 2, row_span))
        c = random.randint(max(1, gate_c - col_span), gate_c - 1)
        return int(r), int(c)
    if kind == "bl":
        r = random.randint(max(1, H - 1 - row_span), H - 2)
        c = random.randint(gate_c + 1, min(W - 2, gate_c + col_span))
        return int(r), int(c)
    r = random.randint(max(1, H - 1 - row_span), H - 2)
    c = random.randint(max(1, gate_c - col_span), gate_c - 1)
    return int(r), int(c)


def _z_component_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    depth: int,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int]],
]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    gate_c = int(depth) if is_left else int(W - 1 - depth)
    row_mid, col_tb = _choose_endpoint_params(kind, H, W, gate_c)

    tb_pt = (0 if is_top else H - 1, int(col_tb))
    lr_pt = (int(row_mid), 0 if is_left else W - 1)
    inner, inner_switch1, inner_switch2 = _build_inner_z_path(tb_pt, lr_pt, gate_c)

    ring_tb = (r0 - 1 if is_top else r1 + 1, c0 + 1 + int(col_tb))
    ring_lr = (r0 + 1 + int(row_mid), c0 - 1 if is_left else c1 + 1)
    outer_r = (r0 - 1 - int(depth)) if is_top else (r1 + 1 + int(depth))
    outer_c = (c0 - 1 - int(depth)) if is_left else (c1 + 1 + int(depth))
    outer = _join(
        _vpath(ring_tb[0], outer_r, ring_tb[1]),
        _hpath(outer_r, ring_tb[1], outer_c),
        _vpath(outer_r, ring_lr[0], outer_c),
        _hpath(ring_lr[0], outer_c, ring_lr[1]),
    )
    outer_switch1 = (int(outer_r), int(ring_tb[1]))
    outer_switch2 = (int(ring_lr[0]), int(outer_c))
    return inner, outer, (inner_switch1, inner_switch2), (outer_switch1, outer_switch2)


def _sample_depths(n_components: int) -> list[int]:
    lo, hi = _OUTER_DEPTH_RANGE
    choices = list(range(lo, hi + 1))
    if n_components <= len(choices):
        return random.sample(choices, k=n_components)
    depths = choices[:]
    while len(depths) < n_components:
        depths.append(random.choice(choices))
    random.shuffle(depths)
    return depths


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=n_components)
    depths = _sample_depths(n_components)

    for kind, depth in zip(corners, depths):
        chosen = None
        for _ in range(_L_COMPONENT_TRIES):
            if kind[1] == "l" and depth >= W - 1:
                continue
            if kind[1] == "r" and depth >= W - 1:
                continue
            inner, outer, inner_switches, outer_switches = _z_component_specs(kind, H, W, r0, c0, r1, c1, depth)
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                continue
            if len(set(outer)) != len(outer):
                continue
            if not _can_place(grid, outer, bg, comp_mask):
                continue
            chosen = (inner, outer, inner_switches, outer_switches)
            break
        if chosen is None:
            return None

        inner, outer, inner_switches, outer_switches = chosen
        colors = tuple(map(int, random.sample(palette, 3)))
        if not _place_component(grid, comp_mask, outer, colors, bg, switch_points=outer_switches):
            return None
        _paint_three_runs(expected, inner, colors, inner_switches)
        irs, ics = zip(*inner)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), random.choice(palette))

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo-28 input")
