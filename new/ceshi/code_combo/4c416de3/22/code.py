# Combo variant of 4c416de3 (index 22)
# combo_spec: D5/0+D6/1+D7/2
# Name: 擦除+裁剪+单框架

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


# Design analysis for the combo variant
# - a) Concrete behavior:
#   现在整张输入里只有一个大 0 空心方框，source 和 target 都落在这一个方框的四角。输入仍沿用原题的
#   “source=完整图案、target=仅 deepest diagonal marker”编码，但输出不再保留 source，也不再返回整张图。
#   一个具体例子：若 source 在 TL，canonical offsets 为 {(-1,0),(0,-1),(0,0),(1,1)}，target 在 TR，
#   则原题会输出整张网格且同时保留 TL 与补出的 TR 图案；本融合题会先把 TL 的四个 source 单元恢复成底图，
#   再只在 TR 画出图案，最后对这对相邻 source-target 角的局部 footprint 取最小包围盒，所以输出变成只含顶部边带的裁片。
# - b) Complexity calibration:
#   为了避免“单框+裁剪”退化成整框，同时让 D5 的 source 擦除在最终输出里仍可见，generator 固定为
#   “单个 source + 单个相邻 target”。这样裁剪后总是一条边带，source 角会以“被清空的原位置”出现在该边带里，
#   target 则是补出的完整图案。图案模板、角朝向与“target 用自己的 marker 颜色”规则保持不变，保证迁移关系可见。
# - c) Hidden assumptions from the original solver:
#   1. 原 solver 假设 `out = g.copy()` 后只需给 target 补画；D5 打破了这一点，所以必须显式擦除 source。
#   2. 原 solver 默认输出尺寸等于输入尺寸；D6 打破了这一点，所以最终结果必须从完整重建图上再裁一刀。
#   3. 原 solver 对多框布局没有“哪些框需要保留”的选择问题；D7 把 source/target 合并到同一框后，裁剪语义
#      不能再依赖“保留 target frame、删除 source frame”，必须改成“按 source-target 交互侧边的局部包围盒裁片”。
#   4. 擦除 source 时仍不能写死成单一颜色：只有 canonical (0,0) 位置恢复成 0，其余 pattern offsets 恢复 bg。
# - d) Placement legality:
#   generator 继续做 output-first 校验，但现在检查的是两个状态：
#   input 状态中 `source full + target markers` 不能互相重叠；output 状态中所有 target full patterns 不能重叠。
#   另外还要求每个可用 corner pattern 只落在允许的底图上，即 (0,0) 对应 0-frame，其他 offsets 对应 bg。
# - e) Visual observability:
#   人类能直接看到：完整 source 图案在输出中消失，marker 变成完整图案，而且整张大图被裁成只围住 source-target
#   相邻边带的一小块。裁片里一端是被清空的 source 角，另一端是补出的 target 角，这能在少量样例中稳定建立
#   “移动到邻角并只保留局部边带”的直觉，不需要依赖颜色编号或隐式排序。
# - f) Visual cleanliness:
#   单框避免了多框并列时的全局杂讯，但也更容易发生角图案互撞；因此 frame 尺寸放大到 8-12，且 generator 拒绝
#   任何 input/output 状态下的彩色单元重叠。裁剪只围住真正参与的 source-target 边带，因此输出比整图更干净。
# - g) Color inference:
#   颜色逻辑仍旧简单：source 颜色只用于输入中的待搬运图案，target 使用自己的 marker 颜色。每个样例只涉及少量角点，
#   且颜色始终绑定在角点位置上，没有额外颜色映射。
# - New solver assumptions reviewed against generator:
#   solver 新增了“裁剪包围盒由参与搬运的 source/target footprint 共同定义”这一假设；generator 通过只选
#   一个与 source 相邻的 target，保证该包围盒是一条局部边带，既不会回到整框，也不会把 source 完全裁掉。
#   solver 还假设所有 source 的 canonical pattern 一致、所有 target marker 都位于最深对角 offset；generator
#   延续原题构造明确保证这些前提。


CORNER_KINDS = ("TL", "TR", "BL", "BR")
CORNER_CYCLE = ("TL", "TR", "BR", "BL")

EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}
OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

_NEIGH_R = 3
_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-oriented) offset; self-inverse."""
    if kind == "TL":
        return (dr, dc)
    if kind == "TR":
        return (dr, -dc)
    if kind == "BL":
        return (-dr, dc)
    if kind == "BR":
        return (-dr, -dc)
    raise ValueError(f"unknown corner kind: {kind}")


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=12):
    cnt = 0
    saw_zero = False
    for k in range(1, max_step + 1):
        rr, cc = r + k * dr, c + k * dc
        if not (0 <= rr < H and 0 <= cc < W):
            break
        v = int(g[rr, cc])
        if v == bg:
            break
        cnt += 1
        if v == 0:
            saw_zero = True
    return cnt, saw_zero


def _find_corners(g, bg):
    H, W = g.shape
    corners = []
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) == bg:
                continue
            for kind in CORNER_KINDS:
                (d1r, d1c), (d2r, d2c) = EDGE_DIRS[kind]
                r1, c1 = r + d1r, c + d1c
                r2, c2 = r + d2r, c + d2c
                if not (0 <= r1 < H and 0 <= c1 < W and 0 <= r2 < H and 0 <= c2 < W):
                    continue
                if int(g[r1, c1]) == bg or int(g[r2, c2]) == bg:
                    continue
                cnt1, sz1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W)
                cnt2, sz2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W)
                if not (cnt1 >= 3 and sz1 and cnt2 >= 3 and sz2):
                    continue
                ok = True
                for odr, odc in OUT_DIRS[kind]:
                    rr, cc = r + odr, c + odc
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == 0:
                        ok = False
                        break
                if not ok:
                    continue
                corners.append((r, c, kind))
                break
    return corners


def _colored_around(g, r, c, kind, bg):
    H, W = g.shape
    items = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _base_value_for_offset(off, bg):
    return 0 if off == (0, 0) else bg


def _pattern_cells(r, c, kind, pattern):
    cells = []
    for off in pattern:
        adr, adc = _rot(*off, kind)
        cells.append(((r + adr, c + adc), off))
    return cells


def _bbox_of_cells(cells):
    rows = [rr for rr, _ in cells]
    cols = [cc for _, cc in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_to_cells(grid, cells):
    r0, c0, r1, c1 = _bbox_of_cells(cells)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _corner_coord(frame, kind):
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _marker_cell(r, c, kind, k_max):
    mdr, mdc = _rot(k_max, k_max, kind)
    return r + mdr, c + mdc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a corner glyph within one large hollow frame, erase the source, then crop to the destination region.

    Input structure:
    - The dominant color is the background; exactly one large hollow 0-frame supplies all four candidate corners.
    - At least one source corner already carries the full one-color glyph, identifiable by outward painted cells.
    - Each participating target corner keeps only the deepest inward-diagonal marker pixel in its own output color;
      non-participating corners may remain plain frame corners.

    Rule:
    - Recover the shared canonical corner glyph from the source corner(s).
    - Erase the glyph from every source corner by restoring 0 at the corner cell and bg elsewhere.
    - Paint the glyph at each marked target corner in that marker's color, then crop the result to the minimal
      rectangle covering the participating source/target corner footprints.

    Key insight:
    - The structural cue for source vs target is unchanged from the base task, so the real fusion happens after
      reconstruction: source pixels are removed rather than kept, and the returned output is not the full frame
      but the local side-strip induced by the source-target interaction.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    def _is_outward(off):
        return off[0] < 0 or off[1] < 0

    sources = []
    for c_key, items in per_corner.items():
        outward = [(o, v) for (o, v) in items if _is_outward(o)]
        if len(outward) >= 2:
            colors = {v for (_, v) in outward}
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner required"

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        return frozenset(o for (o, v) in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, (
            f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"
        )

    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal marker offset"
    marker_canon = (max(diag_ks), max(diag_ks))

    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mr, mc = _marker_cell(r, c, kind, marker_canon[0])
        if not (0 <= mr < H and 0 <= mc < W):
            continue
        v = int(g[mr, mc])
        if v == bg or v == 0:
            continue
        tcol = v
        for off, color in per_corner[c_key]:
            if off in canon:
                assert color == tcol, (
                    f"target ({r},{c},{kind}) has conflicting color {color} at "
                    f"canonical offset {off}; expected {tcol}"
                )
        targets.append((r, c, kind, tcol))
    assert targets, "no target corners detected"

    out = g.copy()
    for r, c, kind in sources:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            out[rr, cc] = _base_value_for_offset((cdr, cdc), bg)

    crop_cells = []
    for r, c, kind in sources:
        crop_cells.extend([coord for coord, _ in _pattern_cells(r, c, kind, canon)])
    for r, c, kind, tcol in targets:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(out[rr, cc])
            assert v == bg or v == 0 or v == tcol, (
                f"paint conflict at ({rr},{cc}) for target ({r},{c},{kind}) "
                f"color {tcol}: existing color {v}"
            )
            out[rr, cc] = tcol
            crop_cells.append((rr, cc))

    return _crop_to_cells(out, crop_cells)


# --------------- generator (output-first moved-and-cropped single-frame variant) ---------------

H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 2
FRAME_SIZE_RANGE = (8, 12)
MAX_ATTEMPTS = 300

PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 0)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)


def _choose_pattern():
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid, r, c, size):
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _assign_corner_colors(frame, bg):
    palette = [c for c in range(1, 10) if c != bg]
    frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _sample_frame(H, W):
    hi = min(FRAME_SIZE_RANGE[1], H - 2 * BORDER, W - 2 * BORDER)
    lo = FRAME_SIZE_RANGE[0]
    if hi < lo:
        return None
    size = random.randint(lo, hi)
    r = random.randint(BORDER, H - size - BORDER)
    c = random.randint(BORDER, W - size - BORDER)
    return {"r": r, "c": c, "size": size}


def _paint_pattern(grid, r, c, kind, pattern, color):
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _pattern_matches_base(base, r, c, kind, pattern, bg):
    H, W = base.shape
    for (rr, cc), off in _pattern_cells(r, c, kind, pattern):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if int(base[rr, cc]) != _base_value_for_offset(off, bg):
            return False
    return True


def _state_cells(frame, pattern, k_max, source_kind, target_kinds, *,
                 include_source_full, include_targets_full):
    occupied = set()
    for kind in CORNER_KINDS:
        r, c = _corner_coord(frame, kind)
        is_source = kind == source_kind
        is_target = kind in target_kinds
        if is_source and include_source_full:
            for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                if (rr, cc) in occupied:
                    return None
                occupied.add((rr, cc))
        elif is_target and include_targets_full:
            for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                if (rr, cc) in occupied:
                    return None
                occupied.add((rr, cc))
        elif is_target:
            marker = _marker_cell(r, c, kind, k_max)
            if marker in occupied:
                return None
            occupied.add(marker)
    return occupied


def _interaction_crop_bbox(frame, pattern, source_kind, target_kinds):
    cells = []
    sr, sc = _corner_coord(frame, source_kind)
    cells.extend([coord for coord, _ in _pattern_cells(sr, sc, source_kind, pattern)])
    for kind in target_kinds:
        r, c = _corner_coord(frame, kind)
        cells.extend([coord for coord, _ in _pattern_cells(r, c, kind, pattern)])
    assert cells
    return _bbox_of_cells(cells)


def _pick_target_kinds(source_kind):
    idx = CORNER_CYCLE.index(source_kind)
    adjacent = [CORNER_CYCLE[(idx + 1) % 4], CORNER_CYCLE[(idx - 1) % 4]]
    return {random.choice(adjacent)}


def _layout_is_legal(base, frame, pattern, k_max, source_kind, target_kinds, bg):
    for kind in {source_kind} | set(target_kinds):
        r, c = _corner_coord(frame, kind)
        if not _pattern_matches_base(base, r, c, kind, pattern, bg):
            return False

    output_cells = _state_cells(
        frame,
        pattern,
        k_max,
        source_kind,
        target_kinds,
        include_source_full=False,
        include_targets_full=True,
    )
    if output_cells is None:
        return False

    input_cells = _state_cells(
        frame,
        pattern,
        k_max,
        source_kind,
        target_kinds,
        include_source_full=True,
        include_targets_full=False,
    )
    if input_cells is None:
        return False

    crop_bbox = _interaction_crop_bbox(frame, pattern, source_kind, target_kinds)
    sr, sc = _corner_coord(frame, source_kind)
    if not (crop_bbox[0] <= sr <= crop_bbox[2] and crop_bbox[1] <= sc <= crop_bbox[3]):
        return False

    full_bbox = (frame["r"] - 1, frame["c"] - 1, frame["r"] + frame["size"], frame["c"] + frame["size"])
    if crop_bbox == full_bbox:
        return False
    return True


def _render_full_output(base, frame, pattern, target_kinds):
    out = base.copy()
    for kind in target_kinds:
        r, c = _corner_coord(frame, kind)
        _paint_pattern(out, r, c, kind, pattern, frame["corner_colors"][kind])
    return out


def _crop_output(full_output, frame, pattern, target_kinds):
    cells = []
    source_kind = frame["source_kind"]
    sr, sc = _corner_coord(frame, source_kind)
    cells.extend([coord for coord, _ in _pattern_cells(sr, sc, source_kind, pattern)])
    for kind in target_kinds:
        r, c = _corner_coord(frame, kind)
        cells.extend([coord for coord, _ in _pattern_cells(r, c, kind, pattern)])
    return _crop_to_cells(full_output, cells)


def _derive_input(base, frame, pattern, k_max, source_kind, target_kinds):
    inp = base.copy()
    sr, sc = _corner_coord(frame, source_kind)
    _paint_pattern(inp, sr, sc, source_kind, pattern, frame["corner_colors"][source_kind])
    for kind in target_kinds:
        tr, tc = _corner_coord(frame, kind)
        mr, mc = _marker_cell(tr, tc, kind, k_max)
        inp[mr, mc] = frame["corner_colors"][kind]
    return inp


def _gen_attempt():
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frame = _sample_frame(H, W)
    if frame is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frame, bg)
    source_kind = random.choice(CORNER_KINDS)
    target_kinds = _pick_target_kinds(source_kind)
    frame["source_kind"] = source_kind

    if not _layout_is_legal(base, frame, pattern, k_max, source_kind, target_kinds, bg):
        return None

    full_output = _render_full_output(base, frame, pattern, target_kinds)
    output = _crop_output(full_output, frame, pattern, target_kinds)
    inp = _derive_input(base, frame, pattern, k_max, source_kind, target_kinds)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
