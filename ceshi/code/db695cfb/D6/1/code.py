# Variant of db695cfb
# Dimension: D6 — 源对象类型 (src)
# Change: 一对蓝色像素——两个孤立的蓝色像素定义对角线的起点和终点... → 蓝色线段（多像素）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# design:
#   source_task: none
#   core_change:
#     - 原题把每条待补全的对角线端点直接画成两个孤立蓝点。
#     - 本变体把每个端点替换成一段 2-3 格、水平或垂直的蓝色短棒。
#     - 真正参与连线的不是整段短棒，而是其左上端（northwest-most cell）。
#   concrete_behavior:
#     - 输入里可见的蓝源对象是若干条 2-3 格直线段，方向只允许水平或垂直。
#     - 每条短棒的左上端像素充当原题中的“端点像素”。
#     - 若两个锚点位于同一条 45 度对角线上，则输出中补全这条蓝色对角线。
#     - 对角线经过的品红像素仍然发射垂直对角方向的品红射线，直到边界。
#   behavior_trace:
#     - 原逻辑示例：蓝点 (2,2) 与 (6,6) 直接连成反斜杠，对角线上若有品红点则向另一对角方向发射。
#     - 新逻辑示例：把这两个蓝点分别扩成短棒 [(2,2),(2,3)] 与 [(6,6),(7,6)]，真正使用的仍是各自左上端
#       (2,2) 和 (6,6)；因此输出对角线与原例同向，但输入视觉上不再是两个单点，而是两段短棒。
#     - 差异在于 solver 不能再把“所有蓝像素”都当端点；必须先识别短棒组件，再提取锚点。
#   complexity_calibration:
#     - 短棒长度限制为 2-3，方向限制为水平/垂直，避免锚点规则过多自由度。
#     - 锚点统一取左上端，能在 2-4 个样例里稳定重复，足以让人类归纳。
#     - 仍保留 1-4 条对角线与 0-2 个线上品红触发器，保持与原题相近的组合复杂度。
#   implicit_assumption_audit:
#     - 原 solver 依赖“同色像素本身就是端点”，因此按颜色把所有蓝像素分组后直接按对角线连接。
#     - 该前提在变体中失效：短棒内部额外蓝像素若不先过滤，会被误判为额外端点并生成错误连线。
#     - 新 solver 必须先做 4 邻域连通分量，再只接受长度 2 或 3 的共线分量作为源对象。
#     - 新 solver 还假设：有效蓝源对象互不 4 邻接，否则会合并成更大分量；generator 会显式保证。
#   placement_legality:
#     - 生成器先选定锚点对角线，再给两端锚点附加向右或向下延伸的短棒。
#     - 这样锚点始终是短棒左上端，也避免短棒主体落到待绘制对角线内部。
#     - 所有短棒像素、触发器像素都不得落在已有输出射线或其他输入对象上。
#   visual_observability:
#     - 变化直接可见：输入中的蓝对象从单点变成了短棒，而输出对角线只从短棒的一端出发。
#     - 水平/垂直两种短棒、2/3 两种长度都会出现，支持人类确认“看左上端”而非“看中心”或“看整段”。
#   cleanliness:
#     - 保留原题颜色语义：蓝负责连线，品红负责发射。
#     - 降低无关蓝色干扰物数量，避免大量蓝单点与蓝短棒混杂。


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _connected_components(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        components.append(comp)
    return components


def _is_source_bar(comp: list[tuple[int, int]]) -> bool:
    if len(comp) not in (2, 3):
        return False
    rows = sorted(r for r, _ in comp)
    cols = sorted(c for _, c in comp)
    if len(set(rows)) == 1:
        return cols == list(range(cols[0], cols[0] + len(comp)))
    if len(set(cols)) == 1:
        return rows == list(range(rows[0], rows[0] + len(comp)))
    return False


def _extract_bar_anchors(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if col != bg:
                by_color.setdefault(col, []).append((r, c))

    source_anchors: dict[int, list[tuple[int, int]]] = {}
    best = None  # (dist, color)
    for col, pts in by_color.items():
        anchors = [min(comp) for comp in _connected_components(pts) if _is_source_bar(comp)]
        if not anchors:
            continue
        source_anchors[col] = anchors
        if len(anchors) < 2:
            continue
        for i in range(len(anchors)):
            r0, c0 = anchors[i]
            for j in range(i + 1, len(anchors)):
                r1, c1 = anchors[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned pair of blue bars must exist"
    _, line_color = best
    return line_color, source_anchors[line_color]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the anchor tips of matching blue bars, then emit perpendicular rays from foreign-color pixels on those diagonals.

    Input structure:
    - bg is the dominant color. The source objects are short blue bars (2-3 cells, horizontal or vertical), and each bar's top-left tip acts as the hidden diagonal anchor. Magenta trigger pixels may sit on the anchor-to-anchor diagonal.

    Rule:
    - Identify each valid blue bar, take its northwest-most cell, and connect diagonal-aligned anchor pairs by drawing the full blue diagonal.
    - Whenever that diagonal passes through a non-blue colored pixel, extend rays of that pixel's color along the perpendicular diagonal directions out to the grid edge.

    Key insight:
    - The extra blue cells are decoration that redefine the source object type; only the bar's top-left tip behaves like the old endpoint pixel, while magenta pixels still act as beam splitters on the completed diagonal.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    bg = get_bg(grid)

    line_color, anchors = _extract_bar_anchors(grid, bg)

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in anchors:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)
_SOURCE_BAR_LEN_RANGE = (2, 3)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _compute_ray_cells(triggers, perp_dirs, H, W):
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _bar_cells(anchor: tuple[int, int], orient: str, length: int) -> list[tuple[int, int]]:
    r, c = anchor
    if orient == "h":
        return [(r, c + k) for k in range(length)]
    return [(r + k, c) for k in range(length)]


def _bar_options(anchor: tuple[int, int], H: int, W: int) -> list[tuple[str, int, list[tuple[int, int]]]]:
    r, c = anchor
    options = []
    for length in range(_SOURCE_BAR_LEN_RANGE[0], _SOURCE_BAR_LEN_RANGE[1] + 1):
        if c + length - 1 < W:
            options.append(("h", length, _bar_cells(anchor, "h", length)))
        if r + length - 1 < H:
            options.append(("v", length, _bar_cells(anchor, "v", length)))
    random.shuffle(options)
    return options


def _build_expected(inp, bg, segments):
    H, W = inp.shape
    expected = inp.copy()
    for _, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(slope_type, seg_len, H, W, occupied, drawn):
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        path_interior = set(path[1:-1])
        if any(rc in occupied or rc in drawn for rc in path_interior):
            continue

        anchor0, anchor1 = path[0], path[-1]
        options0 = _bar_options(anchor0, H, W)
        options1 = _bar_options(anchor1, H, W)
        for _, _, bar0 in options0:
            for _, _, bar1 in options1:
                source_cells = set(bar0) | set(bar1)
                if len(source_cells) != len(bar0) + len(bar1):
                    continue
                if any(rc in occupied or rc in drawn for rc in source_cells):
                    continue
                if any(rc in path_interior for rc in source_cells - {anchor0, anchor1}):
                    continue
                return path, dc, [bar0, bar1]
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    drawn = set()
    segments = []
    anchors = []
    all_blue_cells = set()

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc, bars = result
        source_cells = set().union(*map(set, bars))

        path_interior = path[1:-1]
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 and path_interior else []
        perp_dirs = [(1, -dc), (-1, dc)]

        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
        if any(rc in occupied or rc in source_cells for rc in ray_cells):
            continue

        segments.append((bars, trigger_positions, path, perp_dirs))
        anchors.extend([path[0], path[-1]])
        all_blue_cells.update(source_cells)
        occupied.update(source_cells)
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - set(trigger_positions))
        drawn.update(ray_cells)

    if not segments:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for bars, trigger_positions, _, _ in segments:
        for bar in bars:
            for r, c in bar:
                inp[r, c] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(abs(r - ar) == abs(c - ac) and r != ar for ar, ac in anchors):
                continue
            if any((nr, nc) in all_blue_cells for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_blue_cells.add((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
