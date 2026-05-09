# Variant of 4c416de3
# Dimension: D3 — 目标对象：标记像素定义 (dst)
# Change: 目标角点仅有一个彩色标记像素，位于图案最深对角偏移(k_max, k_max)处，其颜色决定输出颜色... → 多个标记像素形成部分图案
#
# Design analysis:
# variant_spec:
#   concretization:
#     source_corner:
#       - 仍然显示完整图案，且图案颜色单一。
#       - 仍然必须带有 2+ 个 outward 像素，这样 solver 可以稳定地区分 source 与 target。
#     target_corner:
#       - 不再保留唯一的“最深对角 marker”。
#       - 改为保留 2-3 个同色 clue 像素，它们来自完整图案的 non-outward 部分。
#       - 至少一部分 target 会故意删掉最深对角像素，只留下例如 {(0,0),(1,1)} 这样的部分图案。
#   behavior_trace:
#     original:
#       - solver 从 source 提取 canon。
#       - solver 只读取每个非 source 角点在 (k_max,k_max) 的单像素颜色，然后整块补全。
#     variant:
#       - solver 仍从 source 提取 canon。
#       - 但 target 的颜色和“这是一个待补全角点”这一事实，都要从 2-3 个同色 clue 像素整体判断。
#       - 例：若 TL-canon 为 {(-1,0),(0,-1),(0,0),(1,1),(2,2)}，一个 BR target 输入里可只保留旋转后的 {(0,0),(1,1)}；
#         旧逻辑因为看不到 (2,2) 会漏掉该 target，新逻辑会据此补全完整图案。
#   complexity_calibration:
#     - 为了让“多 clue 且可缺 deepest clue”真正可见，pattern 模板统一要求至少 3 个 non-outward 像素。
#     - 这样 target 可以保留 2 像素且省略 deepest，也可以保留 3 像素，视觉上能看出不是单 marker 任务。
#     - 不引入额外颜色映射；target 仍使用自身 clue 的颜色，避免把难度转移到颜色规则。
#   implicit_assumptions_review:
#     - 原 main 依赖“每个 target 一定在最深对角处有颜色”，此假设在变体中失效，必须删除。
#     - 原 main 通过 marker 单点读色；现在必须检查 visible clue 是否都位于 canon 内、是否同色、是否数量为 2-3。
#     - 原 generator 的模板里有些 pattern 只有 1 个 non-outward 像素；这会让“多 clue”无法实现，因此模板需收紧。
#     - 原 source 检测逻辑依赖“target 没有 outward 颜色”才安全，因此 target clue 只能取自 non-outward offsets。
#   placement_legality:
#     - 继续使用原有 18-24 画布、2-3 个 0 框、>=4 的 frame gap，避免半径 3 的局部扫描彼此污染。
#     - pattern 仍限制在 |dr|,|dc|<=2，且 outward 只到 -1，因此原有边距足够，不会越界。
#   visual_observability:
#     - 人类可从 source 看到完整图案，从 target 看到同一角点方位上的 2-3 像素残片，直接推断“把残片补完整”。
#     - 至少一个 target 缺 deepest clue，可防止样例退化回“看单 marker 填充”。
#   visual_cleanliness:
#     - clue 仍绑定在各自框角附近，不跨对象、不重叠。
#     - 颜色只承担“该 target 的输出颜色”这一职责，没有隐藏排序或数值编码。

import random
from collections import Counter
from itertools import combinations

import numpy as np

from arc_utils import draw_line
from arc_utils import get_bg


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
    raise ValueError(f"unknown corner kind {kind}")


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


def _is_outward(off):
    return off[0] < 0 or off[1] < 0


def _clue_offsets(pattern):
    """Target clues may only live on non-outward pattern cells."""
    return tuple(sorted(off for off in pattern if not _is_outward(off)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each partial target-corner clue into the full corner pattern.

    Input structure:
    - bg is the most frequent color, with 0-valued hollow frames whose corners may
      be either plain 0 or overwritten by colored pattern cells.
    - At least one source corner already shows the full rotated pattern in a single
      color, including 2+ outward cells.
    - Every other active corner keeps only 2-3 same-colored clue cells from the
      pattern's non-outward portion; the deepest diagonal cell may be absent.

    Rule:
    - Recover the shared canonical (TL-oriented) pattern from any source corner.
    - For each non-source corner whose visible colored cells form a same-colored
      subset of that canonical pattern, paint the entire rotated pattern there
      using the clue color.

    Key insight:
    - The source tells you the whole shape; targets only tell you which corner
      orientation to use and what color to paint. Because target clues stay on the
      non-outward part of the pattern, they remain visibly incomplete and do not
      masquerade as sources.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    corners = _find_corners(g, bg)
    assert corners, "no frame corner detected"

    per_corner = {c: _colored_around(g, *c, bg) for c in corners}

    sources = []
    for c_key, items in per_corner.items():
        outward = [(o, v) for (o, v) in items if _is_outward(o)]
        if len(outward) >= 2:
            colors = {v for (_, v) in outward}
            assert len(colors) == 1, f"source {c_key} has mixed outward colors: {colors}"
            sources.append(c_key)
    assert sources, "at least one source corner (full outward pattern) required"

    def _extract(c_key):
        items = per_corner[c_key]
        pat_col = Counter(v for (_, v) in items).most_common(1)[0][0]
        return frozenset(o for (o, v) in items if v == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for s in sources[1:]:
        other, _ = _extract(s)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror ({dc},{dr})"

    diag_ks = [dr for (dr, dc) in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward-diagonal support"

    clue_zone = frozenset(_clue_offsets(canon))
    assert len(clue_zone) >= 2, "pattern has too few non-outward cells for partial clues"

    src_set = set(sources)
    targets = []
    for c_key in corners:
        if c_key in src_set:
            continue
        r, c, kind = c_key
        items = per_corner[c_key]
        visible = [(off, color) for (off, color) in items if off in clue_zone]
        if not visible:
            continue

        colors = {color for (_, color) in visible}
        assert len(colors) == 1, f"target ({r},{c},{kind}) has mixed clue colors: {colors}"
        assert 2 <= len(visible) <= 3, (
            f"target ({r},{c},{kind}) should keep 2-3 clue cells, got {len(visible)}"
        )
        targets.append((r, c, kind, next(iter(colors))))

    assert targets, "no target corners detected"

    out = g.copy()
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


# --------------- generator (output-first pattern replication (Codex)) ---------------

# --- canvas & layout ---
H_RANGE = (18, 24)
W_RANGE = (18, 24)
BORDER = 1
FRAME_GAP = 4
FRAME_SIZE_RANGE = (6, 8)
SOURCE_MIN_SIZE = 6

# --- structural counts ---
N_FRAMES_2_PROB = 0.5
TWO_FRAME_HORIZONTAL_PROB = 0.5
N_SOURCES_MAX = 2

# --- generation retries ---
MAX_ATTEMPTS = 300

# Canon patterns are restricted so every target can keep 2-3 non-outward clue cells,
# and at least one clue subset can omit the deepest diagonal cell.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
    [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)

_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}


def _split_total(total: int, parts: int) -> list[int]:
    """Partition `total` non-negative units into `parts` bins uniformly at random."""
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out, prev = [], -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    """Return (pattern cells, k_max = deepest inward-diagonal offset)."""
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    """Draw a size x size hollow frame of 0 with TL at (r, c)."""
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
    """Pick two frame sizes and their origins along one axis of length `span`."""
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
    """Place two frames either side-by-side (shared row) or stacked (shared col)."""
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
    """Place three frames in three of the four corner slots (TL/TR/BL/BR), one missing."""
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


def _paint_pattern(
    grid: np.ndarray, r: int, c: int, kind: str, pattern: list[tuple[int, int]], color: int
) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _assign_corner_colors(frames: list[dict], bg: int) -> None:
    """Give every corner of every frame an independently sampled non-bg color."""
    for frame in frames:
        palette = [c for c in range(1, 10) if c != bg]
        frame["corner_colors"] = dict(zip(CORNER_KINDS, random.sample(palette, 4)))


def _pick_sources(frames: list[dict]) -> set[tuple[int, str]] | None:
    """Choose 1..N_SOURCES_MAX source corners from frames larger than the minimum size."""
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


def _target_clue_candidates(pattern: list[tuple[int, int]], k_max: int) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    clueable = _clue_offsets(pattern)
    assert len(clueable) >= 3, "need at least three clueable cells to omit deepest clues"
    deepest = (k_max, k_max)

    all_sets = []
    for size in range(2, min(3, len(clueable)) + 1):
        for subset in combinations(clueable, size):
            all_sets.append(tuple(subset))
    with_deepest = [subset for subset in all_sets if deepest in subset]
    without_deepest = [subset for subset in all_sets if deepest not in subset]

    assert with_deepest and without_deepest, "candidate clue subsets must branch on deepest visibility"
    return all_sets, with_deepest, without_deepest


def _assign_target_clues(
    frames: list[dict],
    pattern: list[tuple[int, int]],
    k_max: int,
    sources: set[tuple[int, str]],
) -> dict[tuple[int, str], tuple[tuple[int, int], ...]]:
    all_sets, _with_deepest, without_deepest = _target_clue_candidates(pattern, k_max)
    triple_sets = [subset for subset in all_sets if len(subset) == 3]

    targets = [
        (fi, kind)
        for fi, _frame in enumerate(frames)
        for kind in CORNER_KINDS
        if (fi, kind) not in sources
    ]
    assert targets, "need at least one target"

    force_omit = random.choice(targets)
    force_triple = None
    if triple_sets and len(targets) >= 2:
        others = [key for key in targets if key != force_omit]
        force_triple = random.choice(others)

    keep = {}
    for key in targets:
        if key == force_omit:
            keep[key] = random.choice(without_deepest)
        elif key == force_triple:
            keep[key] = random.choice(triple_sets)
        else:
            keep[key] = random.choice(all_sets)
    return keep


def _render_output(base: np.ndarray, frames: list[dict], pattern: list[tuple[int, int]]) -> np.ndarray:
    """Paint the canonical pattern at every corner using that corner's color."""
    out = base.copy()
    for frame in frames:
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            _paint_pattern(out, r, c, kind, pattern, frame["corner_colors"][kind])
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    pattern: list[tuple[int, int]],
    sources: set[tuple[int, str]],
    target_clues: dict[tuple[int, str], tuple[tuple[int, int], ...]],
) -> np.ndarray:
    """Erase every non-source corner back to a 2-3 pixel partial clue."""
    inp = output.copy()
    for fi, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (fi, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            keep = set(target_clues[(fi, kind)])
            for dr, dc in pattern:
                rr, cc = _rot(dr, dc, kind)
                rr, cc = r + rr, c + cc
                if (dr, dc) not in keep:
                    inp[rr, cc] = base[rr, cc]
    return inp


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    frames = _sample_two_frames(H, W) if random.random() < N_FRAMES_2_PROB else _sample_three_frames(H, W)
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    _assign_corner_colors(frames, bg)
    sources = _pick_sources(frames)
    if sources is None:
        return None
    target_clues = _assign_target_clues(frames, pattern, k_max, sources)

    output = _render_output(base, frames, pattern)
    inp = _derive_input(output, base, frames, pattern, sources, target_clues)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
