# Variant of db695cfb
# Dimension: D7 — 交互触发对象 (obstacle)
# Change: 品红像素——位于主对角线路径上的品红像素触发垂直对角延伸... → 多色触发像素（不同颜色不同行为）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D7 variant
# ----------------------------------
# variant_spec:
#   preserved_core:
#     - 蓝色端点仍然定义 45 度对角线段；输出里要把该对角线补全成连续蓝线。
#     - 只有落在蓝色连线路径上的触发像素才会产生额外绘制；离开路径的同色像素保持为静态干扰项。
#     - 所有新增笔画仍然只填充背景，不覆盖任何已有非背景像素。
#   new_trigger_mapping:
#     magenta(6):
#       geometry: 沿与蓝色主对角线垂直的另一组对角方向，向两侧延伸到边界
#       visible_effect: 原题中的 X 形半对角射线
#     red(2):
#       geometry: 沿上下左右四个正交方向延伸到边界
#       visible_effect: 十字形正交射线
#   concrete_trace:
#     original_step:
#       - 蓝色端点 A、B 之间的路径上，只要碰到非蓝像素，就统一记录为“垂直对角触发器”。
#       - 最终所有触发器都按相同的两条斜向射线扩展。
#     variant_step:
#       - 同样先找到 A、B 并补全蓝线。
#       - 若路径上是品红，则记录为“垂直对角”规则；若路径上是红色，则记录为“水平+垂直”规则。
#       - 因而同一条蓝色对角线上可以同时长出斜向射线和十字射线。
#     difference:
#       - 原逻辑把“是否在路径上”作为唯一触发条件。
#       - 变体把“在路径上 + 触发颜色类别”共同作为触发条件。
#   complexity_calibration:
#     - 只使用两种触发色，避免颜色类别过多导致样例不足以支撑映射归纳。
#     - 生成器强制每个样本至少有一个品红触发器和一个红色触发器都落在真实连线路径上，保证 D7 变化肉眼可见。
#     - 额外触发器数量仍然很少，避免把画面挤成难以阅读的射线网。
#   implicit_assumption_review:
#     - 原 main() 在选择 line_color 时会扫描所有颜色；在新变体里，红/品红自己也可能形成同色对角对，因此必须排除“已知触发色”再选线色。
#     - 原 draw_segment() 默认所有非线色像素都共享同一射线规则；现在必须按像素颜色分别分配方向集合。
#     - 原 emit 阶段把 trigger color 直接当成输出色，这一点仍然成立，因此不需要改动“射线颜色等于触发器颜色”这一假设。
#   placement_legality:
#     - 段路径不能与已有输入像素或既有输出专用笔画重叠。
#     - 任一触发器发出的射线不能穿过已有输入像素，也不能与已有输出专用笔画重叠，保持 solver 的“只填背景”前提。
#     - 第一条成功放置的线段强制携带两种触发色，且都位于内部格点，保证 solver 一定能看到两类规则。
#   visual_observability:
#     - 品红保留斜向扩展，红色使用上下左右正交扩展，二者几何差异明显，不依赖颜色编号做隐藏排序。
#     - 离线触发色干扰项仍可保留少量，用来提示“只有落在蓝线上的同色像素才生效”。
#   visual_cleanliness:
#     - 生成器拒绝射线与射线、射线与待补蓝线的重叠，从而避免输出里出现难以归属的交叠中心。
#     - 干扰项数量维持很低，避免多色散点掩盖主结构。
#   generator_contract:
#     - main() 依赖的新增假设是：“至少有一个真实路径品红触发器和一个真实路径红色触发器；所有已知触发色都不是 line_color 候选。”
#     - generate_input() 通过首条混合触发线段、背景排除 1/2/6、以及无碰撞放置来满足这些假设。


_DIAG_TRIGGER_COLOR = 6
_ORTH_TRIGGER_COLOR = 2
_TRIGGER_COLORS = (_DIAG_TRIGGER_COLOR, _ORTH_TRIGGER_COLOR)


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _trigger_dirs(color: int, dr: int, dc: int) -> list[tuple[int, int]]:
    if color == _DIAG_TRIGGER_COLOR:
        return [(dr, -dc), (-dr, dc)]
    if color == _ORTH_TRIGGER_COLOR:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the diagonal blue endpoints, then extend color-specific rays from trigger pixels lying on those diagonals.

    Input structure:
    - Background is the most common color. One non-trigger color forms endpoint pairs on shared diagonals, while path pixels in magenta or red may sit strictly on those diagonal routes.

    Rule:
    - Fill each qualifying diagonal segment between the endpoint pair.
    - Whenever the filled path passes through a magenta pixel, emit the perpendicular diagonal rays in magenta; whenever it passes through a red pixel, emit orthogonal horizontal and vertical rays in red.

    Key insight:
    - The trigger color now chooses the ray geometry: magenta keeps the old rotated-diagonal behavior, while red acts like a cross-shaped splitter.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    # Trigger colors can also repeat on diagonals, so exclude them from line-color selection.
    line_candidates = {col: pts for col, pts in by_color.items() if col not in _TRIGGER_COLORS}

    best = None  # (dist, color)
    for col, pts in line_candidates.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                dirs = _trigger_dirs(v, dr, dc)
                if dirs:
                    key = (r, c, v)
                    triggers.setdefault(key, set()).update(dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

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
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_EXTRA_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _compute_ray_cells(
    trigger_specs: list[tuple[int, int, int]],
    dc: int,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc, color in trigger_specs:
        for ddr, ddc in _trigger_dirs(color, 1, dc):
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(
    inp: np.ndarray,
    bg: int,
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int, int]], list[tuple[int, int]], int]],
) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_specs, path, dc in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc, color in trigger_specs:
            for ddr, ddc in _trigger_dirs(color, 1, dc):
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = color
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int] | None:
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
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _choose_trigger_specs(
    path_interior: list[tuple[int, int]],
    require_mixed: bool,
) -> list[tuple[int, int, int]] | None:
    if require_mixed:
        if len(path_interior) < 2:
            return None
        n_triggers = random.randint(2, min(3, len(path_interior)))
        positions = random.sample(path_interior, n_triggers)
        colors = [_DIAG_TRIGGER_COLOR, _ORTH_TRIGGER_COLOR]
        while len(colors) < n_triggers:
            colors.append(random.choice(_TRIGGER_COLORS))
        random.shuffle(colors)
        return [(r, c, color) for (r, c), color in zip(positions, colors)]

    max_triggers = min(_EXTRA_TRIGGERS_PER_SEG[1], len(path_interior))
    if max_triggers == 0:
        return []
    n_triggers = random.randint(_EXTRA_TRIGGERS_PER_SEG[0], max_triggers)
    if n_triggers == 0:
        return []
    positions = random.sample(path_interior, n_triggers)
    return [(r, c, random.choice(_TRIGGER_COLORS)) for r, c in positions]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, *_TRIGGER_COLORS)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments: list[tuple[tuple[int, int], tuple[int, int], list[tuple[int, int, int]], list[tuple[int, int]], int]] = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result
        path_set = set(path)
        path_interior = path[1:-1]
        require_mixed = len(segments) == 0
        trigger_specs = _choose_trigger_specs(path_interior, require_mixed)
        if trigger_specs is None:
            continue

        ray_cells = _compute_ray_cells(trigger_specs, dc, H, W)
        if any(rc in occupied or rc in drawn or rc in path_set for rc in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_specs, path, dc))
        trigger_positions = {(r, c) for r, c, _ in trigger_specs}
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(set(path_interior) - trigger_positions)
        drawn.update(ray_cells)

    if not segments:
        return None

    on_path_colors = {color for _, _, trigger_specs, _, _ in segments for _, _, color in trigger_specs}
    if on_path_colors != set(_TRIGGER_COLORS):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_specs, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for r, c, color in trigger_specs:
            inp[r, c] = color

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            color = random.choice(_TRIGGER_COLORS)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = color
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
