# Variant of 7ed72f31
# Dimension: D1 — 反射变换类型 (transform)
# Change: 根据红色标记几何形状选择镜像方式：红线→轴反射，单红像素→点反射... → 条件性反射与旋转

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, label_mask


"""
design_analysis:
  variant_core:
    kept_structure:
      - 仍然使用红色(2)作为唯一标记色。
      - 仍然保留三类局部布局：水平红线、垂直红线、孤立单红点。
      - 输入中保留原始 payload 与红标记；输出是在空白背景上补出变换后的副本。
    changed_dimension_D1:
      - 原题由红标记几何形状决定“做哪一种反射”。
      - 本变体改为由 payload 颜色决定“反射 vs 90 度旋转”。
      - 红标记几何形状不再决定大类，而只提供几何参考系：
        - 水平红线：反射时仍对该行做轴反射；旋转时绕红线中心做四分之一转。
        - 垂直红线：反射时仍对该列做轴反射；旋转时绕红线中心做四分之一转。
        - 孤立红点：反射时做点反射；旋转时绕红点做 90 度顺时针旋转。
    color_to_transform_mapping:
      reflect_colors: [3, 4]
      rotate_colors: [5, 6]
      rationale:
        - 颜色种类固定且少，便于从少量样例归纳。
        - 生成器会强制同一几何类型同时出现“反射色”和“旋转色”，避免退化回原题。

  a_concretization_and_trace:
    visual_form:
      - 每个 group 依然是“红标记 + 单色 payload”的紧凑局部图案。
      - 线标记统一使用奇数长度 5，这样旋转时有离散中心像素，避免半格旋转。
      - payload 不是任意随机块，而是若干不对称小 polyomino，确保反射与旋转肉眼可区分。
    traced_example:
      original_logic:
        - 若某 group 是水平红线，则无论 payload 是什么颜色，solver 都把 payload 关于该红行镜像到另一侧。
      new_logic:
        - 若同样的水平红线 group 颜色是 3/4，则仍关于红行镜像。
        - 若颜色是 5/6，则不镜像，而是绕红线中心做 90 度旋转：
          - payload 在红线上方时顺时针转到右侧。
          - payload 在红线下方时逆时针转到右侧。
      difference:
        - 相同红线几何在不同颜色下产生不同变换，解题者必须先归纳颜色类别，再应用几何规则。

  b_complexity_calibration:
    avoided_over_simplification:
      - 不是“所有线都反射、所有点都旋转”，否则颜色控制变量失效。
      - 强制至少出现一组水平反射色和一组水平旋转色，确保同形异变换。
      - 反射色与旋转色各至少复现一次（行/列复用同一主色），为颜色映射提供重复支撑。
    avoided_over_complexity:
      - 只用两档大类：reflect / rotate。
      - 旋转方向并不再额外依赖颜色，只依赖局部相对位置，避免自由度过多。

  c_implicit_assumption_audit_of_original_main:
    original_dependencies_and_adaptations:
      - 原代码把“len(reds)==1 / 共行 / 共列”直接解释成点反射 / 行反射 / 列反射。
        - 变体中这条前提不成立，因此改成：
          - 先读 payload 颜色判定 reflect 或 rotate，
          - 再用红标记几何决定具体轴、中心与旋转参考。
      - 原代码对 marker-containing components 扫全图并按 bbox+颜色吸附额外像素。
        - 这依赖“同色额外像素被一起镜像也无害”的旧题特性。
        - 变体中旋转会更敏感，误吸附会制造远处意外旋转，所以改为仅变换该 component 自身 payload。
      - 原代码对 isolated red pixel 统一做点反射。
        - 现在必须依据相邻 component 的颜色决定点反射还是 90 度旋转。
      - 原代码默认红线长度可任意。
        - 旋转需要稳定离散中心，因此生成器统一使用长度 5；solver 仍按中心像素工作。

  d_placement_legality:
    generator_rules:
      - 先在局部坐标系中构造完整 output group，再删去变换副本得到 input。
      - 全部 group 放置时使用 output 的占位做 8 邻域隔离，避免不同 group 在输入或输出中误连通。
      - 每个候选 group 都先检查：
        - 变换后副本不与原 payload 或红标记重叠；
        - bbox 能放入 18..28 的画布；
        - 孤立红点没有 4 邻接的非背景像素。

  e_visual_observability:
    human_inference_support:
      - 每张图至少同时出现：
        - 水平反射色 group
        - 水平旋转色 group
        - 垂直反射色 group
        - 垂直旋转色 group
      - 因此可以直接在同一几何模板上比较“颜色不同 → 变换不同”。
      - payload 模板强制不具 90 度旋转对称，也尽量避免镜面对称，保证输出差异显眼。

  f_visual_cleanliness:
    cleanliness_choices:
      - 不再生成原题里那种可能被 bbox 误卷入的 free sprites，避免视觉噪声。
      - group 之间留至少 1 格背景缓冲，并按 8 邻域隔离，防止颜色重复时看起来粘连。

  g_color_change_inferability:
    constraints:
      - 只使用四种 payload 颜色，其中两种永远反射、两种永远旋转。
      - 背景只从 {0,1,7,8,9} 中选，不与 payload 颜色冲突。
      - 主反射色与主旋转色在一张图里会重复出现，避免“颜色只出现一次”导致不可归纳。

  h_new_main_assumptions_and_generator_guarantees:
    assumptions:
      - marker-containing component 的 payload 是单色。
      - 水平/垂直线 group 的 payload 完全位于线的一侧。
      - 孤立红点只 8 邻接恰好一个单色 component。
      - 旋转线 group 使用奇数长度红线，从而中心像素明确。
    guaranteed_by_generator:
      - 所有模板都是单色、单侧、非对称的。
      - 线长固定 5。
      - 孤立点模板只从对角线邻接开始扩展，保持“孤立”定义。
"""


MARK = 2
REFLECT_COLORS = {3, 4}
ROTATE_COLORS = {5, 6}
PAYLOAD_COLORS = tuple(sorted(REFLECT_COLORS | ROTATE_COLORS))
BG_CHOICES = (0, 1, 7, 8, 9)
LINE_HALF = 2

ROW_REFLECT_TEMPLATES = (
    ((-1, -2), (-1, -1), (-2, -1), (-3, -1)),
    ((-1, 1), (-2, 1), (-2, 2), (-3, 2)),
    ((-1, -2), (-2, -2), (-2, -1), (-2, 0), (-3, 0)),
    ((-1, 2), (-2, 2), (-2, 1), (-2, 0), (-3, 0)),
)

ROW_ROTATE_TEMPLATES = (
    ((-1, -2), (-1, -1), (-2, -1), (-3, -1)),
    ((-1, 1), (-2, 1), (-2, 2), (-3, 2)),
    ((-1, -2), (-2, -2), (-3, -2), (-3, -1)),
    ((-1, 2), (-2, 2), (-3, 2), (-3, 1)),
)

ISOLATED_TEMPLATES = (
    ((1, 1), (1, 2), (2, 2)),
    ((1, 1), (2, 1), (2, 2), (3, 2)),
    ((1, 1), (2, 2), (3, 2), (3, 3)),
    ((1, 1), (2, 1), (3, 2)),
)


def _transform_kind(color: int) -> str:
    return "reflect" if int(color) in REFLECT_COLORS else "rotate"


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _reflect_point(pivot: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    return int(2 * pr - r), int(2 * pc - c)


def _reflect_row(axis_r: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(axis_r) - int(r)), int(c)


def _reflect_col(axis_c: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(axis_c) - int(c))


def _rotate_cw(pivot: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    dr = int(r - pr)
    dc = int(c - pc)
    return int(pr + dc), int(pc - dr)


def _rotate_ccw(pivot: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, pivot)
    dr = int(r - pr)
    dc = int(c - pc)
    return int(pr - dc), int(pc + dr)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = map(int, g.shape)
    out = g.copy()

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    def paint(src_pixels: list[tuple[int, int, int]], mapper) -> None:
        for r, c, v in src_pixels:
            rr, cc = mapper(int(r), int(c))
            if _in_bounds((h, w), int(rr), int(cc)) and int(g[rr, cc]) == int(bg):
                out[int(rr), int(cc)] = int(v)

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload_pixels = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != int(MARK)]
        if not reds or not payload_pixels:
            continue

        payload_colors = {int(v) for _r, _c, v in payload_pixels}
        if len(payload_colors) != 1:
            continue
        color = int(next(iter(payload_colors)))
        mode = _transform_kind(int(color))

        red_rows = {int(r) for r, _c in reds}
        red_cols = {int(c) for _r, c in reds}

        if len(reds) == 1:
            pivot = reds[0]
            if mode == "reflect":
                paint(payload_pixels, lambda r, c, p=pivot: _reflect_point(p, r, c))
            else:
                paint(payload_pixels, lambda r, c, p=pivot: _rotate_cw(p, r, c))
            continue

        if len(red_rows) == 1:
            axis_r = int(next(iter(red_rows)))
            center_c = int(sorted(int(c) for _r, c in reds)[len(reds) // 2])
            pivot = (int(axis_r), int(center_c))
            top = max(int(r) for r, _c, _v in payload_pixels) < int(axis_r)
            bottom = min(int(r) for r, _c, _v in payload_pixels) > int(axis_r)
            if mode == "reflect":
                paint(payload_pixels, lambda r, c, rr=axis_r: _reflect_row(rr, r, c))
            elif top:
                paint(payload_pixels, lambda r, c, p=pivot: _rotate_cw(p, r, c))
            elif bottom:
                paint(payload_pixels, lambda r, c, p=pivot: _rotate_ccw(p, r, c))
            continue

        if len(red_cols) == 1:
            axis_c = int(next(iter(red_cols)))
            center_r = int(sorted(int(r) for r, _c in reds)[len(reds) // 2])
            pivot = (int(center_r), int(axis_c))
            left = max(int(c) for _r, c, _v in payload_pixels) < int(axis_c)
            right = min(int(c) for _r, c, _v in payload_pixels) > int(axis_c)
            if mode == "reflect":
                paint(payload_pixels, lambda r, c, cc=axis_c: _reflect_col(cc, r, c))
            elif left:
                paint(payload_pixels, lambda r, c, p=pivot: _rotate_ccw(p, r, c))
            elif right:
                paint(payload_pixels, lambda r, c, p=pivot: _rotate_cw(p, r, c))

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        neigh = []
        neigh_colors: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = int(pr + dr)
                cc = int(pc + dc)
                if not _in_bounds((h, w), rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh.append((int(rr), int(cc), int(v)))
                neigh_colors.add(int(v))
        if not neigh_colors:
            continue

        for color in sorted(neigh_colors):
            labs, _ = label_mask(g == int(color), connectivity=8)
            touched = {
                int(labs[int(rr), int(cc)])
                for rr, cc, vv in neigh
                if int(vv) == int(color) and int(labs[int(rr), int(cc)]) != 0
            }
            mode = _transform_kind(int(color))
            for lbl in touched:
                pixels = [(int(r), int(c), int(color)) for r, c in np.argwhere(labs == int(lbl))]
                if mode == "reflect":
                    paint(pixels, lambda r, c, p=(int(pr), int(pc)): _reflect_point(p, r, c))
                else:
                    paint(pixels, lambda r, c, p=(int(pr), int(pc)): _rotate_cw(p, r, c))

    return out


def _row_coords(coords: tuple[tuple[int, int], ...], below: bool) -> list[tuple[int, int]]:
    if not below:
        return [(int(r), int(c)) for r, c in coords]
    return [(-int(r), int(c)) for r, c in coords]


def _col_coords(coords: tuple[tuple[int, int], ...], right: bool) -> list[tuple[int, int]]:
    left = [(int(c), int(r)) for r, c in coords]
    if not right:
        return left
    return [(int(r), -int(c)) for r, c in left]


def _isolated_coords(coords: tuple[tuple[int, int], ...], quadrant: str) -> list[tuple[int, int]]:
    out = []
    for r, c in coords:
        rr, cc = int(r), int(c)
        if quadrant == "se":
            out.append((rr, cc))
        elif quadrant == "sw":
            out.append((rr, -cc))
        elif quadrant == "ne":
            out.append((-rr, cc))
        else:
            out.append((-rr, -cc))
    return out


def _build_row_group(color: int, mode: str) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], dict[str, object]]:
    base = ROW_ROTATE_TEMPLATES if mode == "rotate" else ROW_REFLECT_TEMPLATES
    payload = _row_coords(base[int(np.random.randint(len(base)))], below=bool(np.random.randint(2)))
    reds = [(0, dc) for dc in range(-LINE_HALF, LINE_HALF + 1)]
    if mode == "reflect":
        copy = [(-int(r), int(c)) for r, c in payload]
    else:
        if max(int(r) for r, _c in payload) < 0:
            copy = [(int(c), -int(r)) for r, c in payload]
        else:
            copy = [(-int(c), int(r)) for r, c in payload]

    assert set(payload).isdisjoint(set(reds))
    assert set(copy).isdisjoint(set(reds))
    assert set(copy).isdisjoint(set(payload))

    inp = [(int(r), int(c), int(color)) for r, c in payload] + [(0, int(c), MARK) for _r, c in [(0, dc) for dc in range(-LINE_HALF, LINE_HALF + 1)]]
    out = inp + [(int(r), int(c), int(color)) for r, c in copy]
    meta = {"geometry": "row", "mode": mode, "color": int(color), "copy_pixels": len(copy)}
    return inp, out, meta


def _build_col_group(color: int, mode: str) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], dict[str, object]]:
    base = ROW_ROTATE_TEMPLATES if mode == "rotate" else ROW_REFLECT_TEMPLATES
    payload = _col_coords(base[int(np.random.randint(len(base)))], right=bool(np.random.randint(2)))
    reds = [(dr, 0) for dr in range(-LINE_HALF, LINE_HALF + 1)]
    if mode == "reflect":
        copy = [(int(r), -int(c)) for r, c in payload]
    else:
        if max(int(c) for _r, c in payload) < 0:
            copy = [(-int(c), int(r)) for r, c in payload]
        else:
            copy = [(int(c), -int(r)) for r, c in payload]

    assert set(payload).isdisjoint(set(reds))
    assert set(copy).isdisjoint(set(reds))
    assert set(copy).isdisjoint(set(payload))

    inp = [(int(r), int(c), int(color)) for r, c in payload] + [(int(r), 0, MARK) for r in range(-LINE_HALF, LINE_HALF + 1)]
    out = inp + [(int(r), int(c), int(color)) for r, c in copy]
    meta = {"geometry": "col", "mode": mode, "color": int(color), "copy_pixels": len(copy)}
    return inp, out, meta


def _build_isolated_group(color: int, mode: str) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], dict[str, object]]:
    quad = ("se", "sw", "ne", "nw")[int(np.random.randint(4))]
    payload = _isolated_coords(ISOLATED_TEMPLATES[int(np.random.randint(len(ISOLATED_TEMPLATES)))], quad)
    if mode == "reflect":
        copy = [(-int(r), -int(c)) for r, c in payload]
    else:
        copy = [(int(c), -int(r)) for r, c in payload]

    assert (0, 0) not in set(payload)
    assert set(copy).isdisjoint(set(payload))
    assert (0, 0) not in set(copy)
    for r, c in payload:
        assert not ((abs(int(r)) + abs(int(c)) == 1))

    inp = [(0, 0, MARK)] + [(int(r), int(c), int(color)) for r, c in payload]
    out = inp + [(int(r), int(c), int(color)) for r, c in copy]
    meta = {"geometry": "isolated", "mode": mode, "color": int(color), "copy_pixels": len(copy)}
    return inp, out, meta


def _translated(
    cells: list[tuple[int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int]]:
    return [(int(r + dr), int(c + dc), int(v)) for r, c, v in cells]


def _cell_bounds(cells: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _c, _v in cells]
    cs = [int(c) for _r, c, _v in cells]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _can_place(cells: list[tuple[int, int, int]], h: int, w: int, occupied: set[tuple[int, int]]) -> bool:
    coords = {(int(r), int(c)) for r, c, _v in cells}
    if len(coords) != len(cells):
        return False
    for r, c in coords:
        if not _in_bounds((h, w), int(r), int(c)):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (int(r + dr), int(c + dc)) in occupied:
                    return False
    return True


def _place_group(
    inp_cells: list[tuple[int, int, int]],
    out_cells: list[tuple[int, int, int]],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    r0, c0, r1, c1 = _cell_bounds(out_cells)
    patch_h = int(r1 - r0 + 1)
    patch_w = int(c1 - c0 + 1)
    if patch_h > h or patch_w > w:
        raise ValueError("group too large for grid")

    for _ in range(200):
        top = int(np.random.randint(0, h - patch_h + 1))
        left = int(np.random.randint(0, w - patch_w + 1))
        dr = int(top - r0)
        dc = int(left - c0)
        moved_out = _translated(out_cells, dr, dc)
        if not _can_place(moved_out, int(h), int(w), occupied):
            continue
        moved_inp = _translated(inp_cells, dr, dc)
        return moved_inp, moved_out
    raise ValueError("failed to place group")


def _paint(grid: np.ndarray, cells: list[tuple[int, int, int]]) -> None:
    for r, c, v in cells:
        grid[int(r), int(c)] = int(v)


def generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    for _attempt in range(200):
        h = int(np.random.randint(18, 29))
        w = int(np.random.randint(18, 29))
        bg = int(BG_CHOICES[int(np.random.randint(len(BG_CHOICES)))])
        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full((h, w), int(bg), dtype=int)
        occupied: set[tuple[int, int]] = set()

        reflect_primary = int((3, 4)[int(np.random.randint(2))])
        rotate_primary = int((5, 6)[int(np.random.randint(2))])
        reflect_secondary = 4 if reflect_primary == 3 else 3
        rotate_secondary = 6 if rotate_primary == 5 else 5

        specs: list[tuple[str, int]] = [
            ("row", reflect_primary),
            ("row", rotate_primary),
            ("col", reflect_primary),
            ("col", rotate_primary),
        ]

        extras_pool = [
            ("isolated", reflect_secondary),
            ("isolated", rotate_secondary),
            ("row", reflect_secondary),
            ("col", rotate_secondary),
        ]
        np.random.shuffle(extras_pool)
        extra_count = int(np.random.randint(0, 3))
        specs.extend(extras_pool[:extra_count])
        np.random.shuffle(specs)

        meta_groups = []
        ok = True
        try:
            for geom, color in specs:
                mode = _transform_kind(int(color))
                if geom == "row":
                    local_inp, local_out, meta = _build_row_group(int(color), mode)
                elif geom == "col":
                    local_inp, local_out, meta = _build_col_group(int(color), mode)
                else:
                    local_inp, local_out, meta = _build_isolated_group(int(color), mode)

                moved_inp, moved_out = _place_group(local_inp, local_out, int(h), int(w), occupied)
                _paint(inp, moved_inp)
                _paint(out, moved_out)
                occupied |= {(int(r), int(c)) for r, c, _v in moved_out}
                meta_groups.append(meta)
        except ValueError:
            ok = False

        if not ok:
            continue

        if not np.array_equal(main(inp), out):
            continue

        colors_present = {int(m["color"]) for m in meta_groups}
        modes_present = {str(m["mode"]) for m in meta_groups}
        row_modes = {str(m["mode"]) for m in meta_groups if m["geometry"] == "row"}
        if not ({reflect_primary, rotate_primary} <= colors_present):
            continue
        if modes_present != {"reflect", "rotate"}:
            continue
        if row_modes != {"reflect", "rotate"}:
            continue

        meta = {
            "bg": int(bg),
            "groups": meta_groups,
            "grid_shape": (int(h), int(w)),
            "group_count": len(meta_groups),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid case")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
