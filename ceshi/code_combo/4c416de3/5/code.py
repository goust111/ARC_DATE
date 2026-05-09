# Combo variant of 4c416de3 (index 5)
# combo_spec: D5/0+D6/1
# Name: 擦除源+裁剪目标框架

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


# Design analysis for the combo variant
# - a) Concrete behavior:
#   该融合版把原题的“复制到所有 target”改成“从 source 搬走，再只输出 target 框”。一个具体例子：
#   若 canonical glyph 为 {(-1,0),(0,-1),(0,0),(1,1),(2,2)}，输入里 source 框只有一个 TL source，
#   target 框四个角各有一个 deepest marker。原题会在整张大网格上保留 source，并把图案补到 target；
#   D5/0 会把 source 擦回底图；D6/1 再把结果裁到 target 框及其角外尾巴的最小包围盒。因此最终 output
#   不再含 source 框，且尺寸明显小于 input。
# - b) Complexity calibration:
#   为了让“裁哪个框”肉眼可推断，generator 收紧为恰好两个 frame：一个 source frame，一个 target frame。
#   target frame 的四个角都带 marker，source frame 只有一个完整 source。这样样例中始终存在重复支撑：
#   解题者能稳定看到“左/上那个完整图案来自 source，右/下那个框的四个 marker 被补全，最终只剩后者”。
# - c) Hidden assumptions from the original solver:
#   1. 原 solver 默认输出仍与输入同尺寸；这里失效，所以 `main()` 不能直接返回整张 `out`，必须在完成
#      擦除与补画后，识别唯一 target frame 并裁剪其区域。
#   2. 原 solver 默认 source 保留；这里失效，所以必须恢复 source offsets 的底图值。
#   3. 原 solver 允许多个 target frames 同时存在；这里若仍如此，crop 目标会歧义。因此新 solver 显式要求
#      所有被识别出的 target 都属于同一个 frame，generator 也按此构造。
#   4. 原 solver 从 source 邻域提取 canonical pattern。若 source 太多且彼此过近，半径 3 的扫描会把别的
#      彩色像素吸进来，所以 generator 固定为单 source corner，避免新的混色歧义。
# - d) Placement legality:
#   generator 只放两个 6-8 的方框，并检查：source 完整图案与 target 完整图案都只能落在允许的底图
#   上（corner offset (0,0) 落在 0，其余 offsets 落在 bg）；target 四角的完整图案彼此不能重叠；
#   input 中 source 图案与 target markers 也不能重叠。
# - e) Visual observability:
#   2-3 个样例足以看出规律：输入总有一个完整图案所在的 source frame 和一个只带 marker 的 target frame；
#   输出中 source frame 整体消失，只剩 target frame 的裁剪局部。这是纯视觉可见的“搬运+裁剪”，不依赖
#   颜色 id 排序或任何不可见元数据。
# - f) Visual cleanliness:
#   只保留一个 target frame，避免多框 union crop 带来的大块空白；同时维持原题的小型角点 glyph 模板，
#   不引入额外装饰或叠压对象。
# - g) Color inference:
#   颜色规则与原题一致：target 颜色来自各自 marker，source 颜色只用于输入中的模板展示。颜色数保持少量，
#   且每个 marker 在输出中直接扩张成同色 glyph，人类容易从样例归纳。
# - New solver assumptions reviewed against generator:
#   solver 新增两个关键假设：一是“所有 target 属于唯一 frame”，二是“source/target 的模板单元不会越界
#   或互撞”。generator 通过固定单 target frame、单 source corner，以及 `_layout_is_legal` 的占位检查
#   明确保证这些前提成立。


CORNER_KINDS = ("TL", "TR", "BL", "BR")

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


def _edge_metric(g, r, c, dr, dc, bg, H, W, max_step=10):
    """Walk from (r,c) along (dr,dc) until bg/boundary. Return (non_bg_count, saw_zero)."""
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
                cnt1, sz1 = _edge_metric(g, r, c, d1r, d1c, bg, H, W, max_step=max(H, W))
                cnt2, sz2 = _edge_metric(g, r, c, d2r, d2c, bg, H, W, max_step=max(H, W))
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


def _frame_bbox_from_corner(g, r, c, kind, bg):
    H, W = g.shape
    (d1r, d1c), (d2r, d2c) = EDGE_DIRS[kind]
    cnt1, _ = _edge_metric(g, r, c, d1r, d1c, bg, H, W, max_step=max(H, W))
    cnt2, _ = _edge_metric(g, r, c, d2r, d2c, bg, H, W, max_step=max(H, W))
    assert cnt1 == cnt2, f"expected square frame at {(r, c, kind)}, got {cnt1} vs {cnt2}"
    if kind == "TL":
        return (r, c, r + cnt1, c + cnt2)
    if kind == "TR":
        return (r, c - cnt2, r + cnt1, c)
    if kind == "BL":
        return (r - cnt1, c, r, c + cnt2)
    if kind == "BR":
        return (r - cnt1, c - cnt2, r, c)
    raise ValueError(f"unknown corner kind: {kind}")


def _bbox_union(b0, b1):
    return (
        min(b0[0], b1[0]),
        min(b0[1], b1[1]),
        max(b0[2], b1[2]),
        max(b0[3], b1[3]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a corner glyph from the source frame onto the marked target frame, then crop to that target.

    Input structure:
    - The dominant color is the background; hollow 0-frames provide candidate corners.
    - One source corner already shows the full one-color glyph, including outward tail cells and
      inward-diagonal cells.
    - A different frame supplies the destinations: its participating corners keep only the deepest
      inward-diagonal marker, whose color is the final glyph color for that corner.

    Rule:
    - Recover the shared canonical glyph from the source corner.
    - Erase the source glyph back to the underlying bg/0 scaffold.
    - Paint the rotated glyph at every marked target corner, then return only the bounding region
      of the completed target frame.

    Key insight:
    - Sources are still the only corners with outward colored cells, so pattern extraction is the
      same as before. The new work is downstream: all detected targets must belong to one frame,
      and the final answer is that frame's completed local region rather than the full canvas.
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
    assert sources, "at least one source corner (pre-drawn outward pattern) required"

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
    assert diag_ks, "pattern has no inward-diagonal cell to anchor markers"
    marker_canon = (max(diag_ks), max(diag_ks))

    src_set = set(sources)
    targets = []
    target_frame_box = None
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        mdr, mdc = _rot(*marker_canon, kind)
        mr, mc = r + mdr, c + mdc
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
        box = _frame_bbox_from_corner(g, r, c, kind, bg)
        target_frame_box = box if target_frame_box is None else _bbox_union(target_frame_box, box)
        targets.append((r, c, kind, tcol, box))

    assert targets, "no target corners detected (every corner appears inert or source)"
    unique_boxes = {box for (_, _, _, _, box) in targets}
    assert len(unique_boxes) == 1, f"expected one target frame, found {len(unique_boxes)}"
    target_frame_box = next(iter(unique_boxes))
    for r, c, kind in sources:
        assert _frame_bbox_from_corner(g, r, c, kind, bg) != target_frame_box, (
            "source and target must belong to different frames"
        )

    out = g.copy()
    for r, c, kind in sources:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            out[rr, cc] = _base_value_for_offset((cdr, cdc), bg)

    crop_box = target_frame_box
    for r, c, kind, tcol, box in targets:
        assert box == target_frame_box
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
            crop_box = _bbox_union(crop_box, (rr, cc, rr, cc))

    r0, c0, r1, c1 = crop_box
    return out[r0 : r1 + 1, c0 : c1 + 1]


# --------------- generator (output-first movement + target crop (Codex)) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)

# --- structural counts ---
TWO_FRAME_HORIZONTAL_PROB = 0.5

# --- generation retries ---
MAX_ATTEMPTS = 300
_GEN_CALL_COUNT = 0

# --- canonical (TL-oriented) pattern templates ---
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (0, 0)],
    [(-1, -1), (-1, 0), (0, -1), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern(call_id: int):
    # Cycle templates across successive examples so 50-run audits do not depend on RNG luck
    # for branch coverage.
    pattern = list(PATTERN_TEMPLATES[call_id % len(PATTERN_TEMPLATES)])
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame: dict, kind: str):
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _sample_frame_pair(span: int):
    lo, hi = FRAME_SIZE_RANGE
    pair_limit = span - (2 * BORDER + FRAME_GAP)
    max_first = min(hi, pair_limit - lo)
    if max_first < lo:
        return None
    s1 = random.randint(lo, max_first)
    s2 = random.randint(lo, min(hi, pair_limit - s1))
    slack = span - (2 * BORDER + s1 + FRAME_GAP + s2)
    before, between, _after = _split_total(slack, 3)
    o1 = BORDER + before
    o2 = o1 + s1 + FRAME_GAP + between
    return s1, s2, o1, o2


def _sample_two_frames(H: int, W: int, horizontal: bool | None = None):
    if horizontal is None:
        horizontal = random.random() < TWO_FRAME_HORIZONTAL_PROB
    axis = _sample_frame_pair(W if horizontal else H)
    if axis is None:
        return None
    s1, s2, o1, o2 = axis
    if horizontal:
        row = random.randint(BORDER, H - max(s1, s2) - BORDER)
        return [{"r": row, "c": o1, "size": s1}, {"r": row, "c": o2, "size": s2}]
    col = random.randint(BORDER, W - max(s1, s2) - BORDER)
    return [{"r": o1, "c": col, "size": s1}, {"r": o2, "c": col, "size": s2}]


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str, pattern, color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pattern_cells(r: int, c: int, kind: str, pattern):
    cells = []
    for off in pattern:
        adr, adc = _rot(*off, kind)
        cells.append(((r + adr, c + adc), off))
    return cells


def _marker_cell(r: int, c: int, kind: str, k_max: int):
    mdr, mdc = _rot(k_max, k_max, kind)
    return r + mdr, c + mdc


def _pattern_matches_base(base: np.ndarray, r: int, c: int, kind: str, pattern, bg: int) -> bool:
    H, W = base.shape
    for (rr, cc), off in _pattern_cells(r, c, kind, pattern):
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if int(base[rr, cc]) != _base_value_for_offset(off, bg):
            return False
    return True


def _occupied_cells_for_role(
    frames: list[dict],
    pattern,
    k_max: int,
    source_index: int,
    source_kind: str,
    *,
    include_full_target: bool,
):
    occupied = set()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            if fi == source_index and kind == source_kind:
                for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                    if (rr, cc) in occupied:
                        return None
                    occupied.add((rr, cc))
            elif fi != source_index:
                if include_full_target:
                    for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                        if (rr, cc) in occupied:
                            return None
                        occupied.add((rr, cc))
                else:
                    marker = _marker_cell(r, c, kind, k_max)
                    if marker in occupied:
                        return None
                    occupied.add(marker)
    return occupied


def _layout_is_legal(
    base: np.ndarray,
    frames: list[dict],
    pattern,
    k_max: int,
    source_index: int,
    source_kind: str,
    bg: int,
) -> bool:
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if fi == source_index and kind != source_kind:
                continue
            r, c = _corner_coord(frame, kind)
            if not _pattern_matches_base(base, r, c, kind, pattern, bg):
                return False

    output_cells = _occupied_cells_for_role(
        frames,
        pattern,
        k_max,
        source_index,
        source_kind,
        include_full_target=True,
    )
    if output_cells is None:
        return False

    input_cells = _occupied_cells_for_role(
        frames,
        pattern,
        k_max,
        source_index,
        source_kind,
        include_full_target=False,
    )
    if input_cells is None:
        return False

    return True


def _frame_bbox(frame: dict):
    r0, c0 = frame["r"], frame["c"]
    s1 = frame["size"] - 1
    return (r0, c0, r0 + s1, c0 + s1)


def _crop_bbox_for_target_frame(frame: dict, pattern):
    bbox = _frame_bbox(frame)
    for kind in CORNER_KINDS:
        r, c = _corner_coord(frame, kind)
        for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
            bbox = _bbox_union(bbox, (rr, cc, rr, cc))
    return bbox


def _crop_to_bbox(grid: np.ndarray, bbox):
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _render_full_output(base: np.ndarray, frames: list[dict], pattern, source_index: int) -> np.ndarray:
    out = base.copy()
    target_frame = frames[1 - source_index]
    for kind in CORNER_KINDS:
        r, c = _corner_coord(target_frame, kind)
        _paint_pattern(out, r, c, kind, pattern, target_frame["corner_colors"][kind])
    return out


def _derive_input(
    full_output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    pattern,
    k_max: int,
    source_index: int,
    source_kind: str,
) -> np.ndarray:
    inp = full_output.copy()

    target_frame = frames[1 - source_index]
    for kind in CORNER_KINDS:
        r, c = _corner_coord(target_frame, kind)
        marker = _marker_cell(r, c, kind, k_max)
        for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
            if (rr, cc) != marker:
                inp[rr, cc] = base[rr, cc]

    source_frame = frames[source_index]
    sr, sc = _corner_coord(source_frame, source_kind)
    _paint_pattern(inp, sr, sc, source_kind, pattern, source_frame["corner_colors"][source_kind])
    return inp


def _gen_attempt(call_id: int):
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern(call_id)

    horizontal = (call_id % 2) == 0
    frames = _sample_two_frames(H, W, horizontal=horizontal)
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    source_index = (call_id // 2) % 2
    source_kind = CORNER_KINDS[(call_id // 4) % len(CORNER_KINDS)]

    if not _layout_is_legal(base, frames, pattern, k_max, source_index, source_kind, bg):
        return None

    full_output = _render_full_output(base, frames, pattern, source_index)
    inp = _derive_input(full_output, base, frames, pattern, k_max, source_index, source_kind)
    crop_bbox = _crop_bbox_for_target_frame(frames[1 - source_index], pattern)
    output = _crop_to_bbox(full_output, crop_bbox)

    if output.shape == inp.shape:
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_id = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(call_id)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
