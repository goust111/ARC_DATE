# Variant of 4c416de3
# Dimension: D5 — 轨迹：源角点保留策略 (trajectory)
# Change: 源角点的完整图案在输出中保留不变（keep source）... → 擦除源图案

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


# Design analysis for the D5 variant
# - a) Concrete behavior:
#   输入仍然包含 2-3 个空心 0 方框。至少一个角点是 source：该角点周围已经画好了完整图案，
#   图案包含角外的尾巴和角内的对角线像素。其余角点是 target：只保留最深的对角线 marker。
#   原题 output 会把 source 保留，再把图案复制到 target；本变体 output 要把 source 还原成底图，
#   只在 target 角点留下图案。因此 solver 的核心差异只发生在“最终落笔”阶段：原来只新增 target，
#   现在要先擦掉 source，再画 target。一个具体例子：若 TL source 的 canonical offsets 为
#   {(-1,0),(0,-1),(0,0),(1,1)}，原题 output 保留这 4 个源色像素；本题 output 则把前三个恢复为 bg，
#   把 (0,0) 恢复为 0，只在 target 角点出现对应颜色的旋转图案。
# - b) Complexity calibration:
#   不改变图案模板、角点朝向、颜色赋值和 frame 数量，避免把任务变成全新题型；同时 source 的消失
#   在视觉上非常明显，因为完整角外尾巴会被清空。source 数量继续允许 1 或 2，保证样例中既有重复
#   支撑，又不会出现过多子情况。
# - c) Hidden assumptions from the original solver:
#   1. 原 solver 依赖“source 在 output 中仍存在”，所以 `out = g.copy()` 后只对 target 补画。
#      这里不再成立，必须显式把 source 的 canonical pattern offsets 擦回底图。
#   2. 原 solver 从 source 提取 canonical pattern 的逻辑仍然成立，因为 source 仍然只在输入中出现，
#      且仍以“具有 >=2 个 outward 彩色像素”区分。
#   3. 擦除时不能盲目写成统一颜色：offset (0,0) 对应 frame corner，应恢复成 0；其他模板 offset
#      都位于 frame 外部或内部空腔，应恢复成 bg。generator 继续保证这一模板不变量。
# - d) Placement legality:
#   generator 额外检查每个 corner pattern 是否只落在其允许的底图上：corner cell 只能覆盖 0，
#   其他 pattern cell 只能覆盖 bg。这样可以排除图案撞到其他 frame 边或互相重叠的布局。
#   同时分别检查 output(target full patterns) 与 input(source full patterns + target markers)
#   的着色单元互不重合，保证 solver 不会遇到歧义覆盖。
# - e) Visual observability:
#   解题者在 2-3 组样例里能直接看到“某个完整图案消失，其他 marker 变成完整图案”，这是一个可见的
#   move 语义，而不是依赖颜色 id、排序或隐藏计数。source 的 outward 尾巴尤其利于辨认“被搬走了”。
# - f) Visual cleanliness:
#   保持原始方框排布和少量 pattern 模板，但拒绝任何 pattern-to-pattern 或 pattern-to-frame 的碰撞，
#   避免局部区域出现杂乱叠压。
# - g) Color inference:
#   颜色规则不变：target 使用自己的 marker 颜色，source 颜色只用于输入中的被搬运图案。颜色数不多，
#   并且每个颜色始终绑定到其所在角点，不引入额外映射。
# - New solver assumptions reviewed against generator:
#   solver 现在假设 canonical pattern 里至多只有 corner offset (0,0) 会落在 0-frame 上，其余 offsets
#   的底色都是 bg；generator 的 `_pattern_matches_base` 明确保证了这一点。solver 还假设 source/target
#   在输入中的彩色像素不重叠，generator 通过 `_state_cells` 的唯一性检查保证这一点。


CORNER_KINDS = ("TL", "TR", "BL", "BR")

# Inward-along-frame-edge directions for each corner kind.
EDGE_DIRS = {
    "TL": ((+1, 0), (0, +1)),
    "TR": ((+1, 0), (0, -1)),
    "BL": ((-1, 0), (0, +1)),
    "BR": ((-1, 0), (0, -1)),
}
# Outward-from-corner directions; cells here must not be 0 (guards mid-edge false positives).
OUT_DIRS = {
    "TL": ((-1, 0), (0, -1)),
    "TR": ((-1, 0), (0, +1)),
    "BL": ((+1, 0), (0, -1)),
    "BR": ((+1, 0), (0, +1)),
}

# Radius for scanning a corner's neighborhood. Must exceed max canonical |dr|,|dc| a pattern can have.
_NEIGH_R = 3


def _rot(dr, dc, kind):
    """Map actual offset <-> canonical (TL-orientation) offset; self-inverse."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a pre-drawn corner glyph away from every source corner onto all marked destination corners.

    Input structure:
    - The dominant color is the background; hollow 0-frames define candidate corners.
    - One or more source corners already carry the full one-color glyph, including outward cells
      outside the frame and inward-diagonal cells inside it.
    - Every other participating corner shows only a single colored marker on the deepest inward
      diagonal location; that marker color is the glyph color for that destination.

    Rule:
    - Recover the shared canonical glyph from any source corner.
    - Erase that glyph from every source corner by restoring the underlying bg/0 scaffold.
    - Paint the same glyph, rotated to each destination corner, in that destination's marker color.

    Key insight:
    - The input still separates sources from destinations by structure: only sources have outward
      painted cells. Once the canonical offsets are known, this D5 variant differs from the original
      task only in the final write-back step: the source offsets are reset to their base colors
      instead of being left in place.
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
        targets.append((r, c, kind, tcol))

    assert targets, "no target corners detected (every corner appears to be a source)"

    out = g.copy()
    for r, c, kind in sources:
        for cdr, cdc in canon:
            adr, adc = _rot(cdr, cdc, kind)
            rr, cc = r + adr, c + adc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            out[rr, cc] = _base_value_for_offset((cdr, cdc), bg)

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

    return out


# --------------- generator (output-first pattern movement (Codex)) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (5, 8)
SOURCE_MIN_SIZE = 6

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 300

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


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    s1 = frame["size"] - 1
    return frame["r"] + dr * s1, frame["c"] + dc * s1


def _sample_frame_pair(span: int) -> tuple[int, int, int, int] | None:
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


def _sample_two_frames(H: int, W: int) -> list[dict] | None:
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


def _sample_three_frames(H: int, W: int) -> list[dict] | None:
    lo, hi = FRAME_SIZE_RANGE
    missing = random.choice(CORNER_KINDS)
    present = tuple(k for k in CORNER_KINDS if k != missing)

    def _bands(sizes: dict[str, int]) -> tuple[int, int, int, int]:
        top = max((sizes[s] for s in present if s[0] == "T"), default=0)
        bot = max((sizes[s] for s in present if s[0] == "B"), default=0)
        left = max((sizes[s] for s in present if s[1] == "L"), default=0)
        right = max((sizes[s] for s in present if s[1] == "R"), default=0)
        return top, bot, left, right

    feasible = []
    n_sizes = hi - lo + 1
    for combo in np.ndindex(*(n_sizes for _ in present)):
        sizes = {slot: lo + i for slot, i in zip(present, combo)}
        top, bot, left, right = _bands(sizes)
        if top + FRAME_GAP + bot > H - 2 * BORDER:
            continue
        if left + FRAME_GAP + right > W - 2 * BORDER:
            continue
        feasible.append(sizes)
    if not feasible:
        return None

    sizes = random.choice(feasible)
    top, bot, left, right = _bands(sizes)
    top_extra, mid_y, _ = _split_total(H - (2 * BORDER + top + FRAME_GAP + bot), 3)
    left_extra, mid_x, _ = _split_total(W - (2 * BORDER + left + FRAME_GAP + right), 3)
    top_row = BORDER + top_extra
    bot_row = top_row + top + FRAME_GAP + mid_y
    left_col = BORDER + left_extra
    right_col = left_col + left + FRAME_GAP + mid_x
    coords = {
        "TL": (top_row, left_col),
        "TR": (top_row, right_col),
        "BL": (bot_row, left_col),
        "BR": (bot_row, right_col),
    }
    return [{"r": coords[s][0], "c": coords[s][1], "size": sizes[s]} for s in present]


def _paint_pattern(grid: np.ndarray, r: int, c: int, kind: str, pattern, color: int) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    eligible = [
        (fi, kind)
        for fi, f in enumerate(frames)
        if f["size"] >= SOURCE_MIN_SIZE
        for kind in CORNER_KINDS
    ]
    if not eligible:
        return None
    n = random.randint(1, min(N_SOURCES_MAX, len(eligible)))
    return set(random.sample(eligible, n))


def _pattern_cells(r: int, c: int, kind: str, pattern) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cells = []
    for off in pattern:
        adr, adc = _rot(*off, kind)
        cells.append(((r + adr, c + adc), off))
    return cells


def _marker_cell(r: int, c: int, kind: str, k_max: int) -> tuple[int, int]:
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


def _state_cells(frames: list[dict], pattern, k_max: int, sources: set[tuple[int, str]], *,
                 include_sources_full: bool, include_targets_full: bool) -> set[tuple[int, int]] | None:
    occupied = set()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            is_source = (fi, kind) in sources
            if is_source and include_sources_full:
                for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                    if (rr, cc) in occupied:
                        return None
                    occupied.add((rr, cc))
            elif (not is_source) and include_targets_full:
                for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                    if (rr, cc) in occupied:
                        return None
                    occupied.add((rr, cc))
            elif not is_source:
                marker = _marker_cell(r, c, kind, k_max)
                if marker in occupied:
                    return None
                occupied.add(marker)
    return occupied


def _layout_is_legal(base: np.ndarray, frames: list[dict], pattern, k_max: int,
                     sources: set[tuple[int, str]], bg: int) -> bool:
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            if not _pattern_matches_base(base, r, c, kind, pattern, bg):
                return False

    output_cells = _state_cells(
        frames,
        pattern,
        k_max,
        sources,
        include_sources_full=False,
        include_targets_full=True,
    )
    if output_cells is None:
        return False

    input_cells = _state_cells(
        frames,
        pattern,
        k_max,
        sources,
        include_sources_full=True,
        include_targets_full=False,
    )
    if input_cells is None:
        return False

    removed = 0
    for fi, kind in sources:
        r, c = _corner_coord(frames[fi], kind)
        for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
            if (rr, cc) not in output_cells:
                removed += 1
    return removed > 0


def _render_output(base: np.ndarray, frames: list[dict], pattern, sources: set[tuple[int, str]]) -> np.ndarray:
    out = base.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, frame["corner_colors"][kind])
    return out


def _derive_input(output: np.ndarray, base: np.ndarray, frames: list[dict], pattern, k_max: int,
                  sources: set[tuple[int, str]]) -> np.ndarray:
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            if (fi, kind) in sources:
                _paint_pattern(inp, r, c, kind, pattern, frame["corner_colors"][kind])
                continue
            marker = _marker_cell(r, c, kind, k_max)
            for (rr, cc), _ in _pattern_cells(r, c, kind, pattern):
                if (rr, cc) != marker:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = (
        _sample_two_frames(H, W)
        if random.random() < N_FRAMES_2_PROB
        else _sample_three_frames(H, W)
    )
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None
    if not _layout_is_legal(base, frames, pattern, k_max, sources, bg):
        return None

    output = _render_output(base, frames, pattern, sources)
    inp = _derive_input(output, base, frames, pattern, k_max, sources)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
