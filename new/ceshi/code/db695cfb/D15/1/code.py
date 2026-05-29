# Variant of db695cfb
# Dimension: D15 — 背景与前景关系 (meta)
# Change: 在黑色背景上绘制线条，所有非像素区域保持黑色... → 仅在特定背景色区域绘制

from collections import Counter
import random

import numpy as np


"""
design_analysis:
  variant_summary:
    original_task: 使用同色端点确定对角线段，在线段遇到触发色像素时向垂直对角方向发射射线；所有填充都落在单一背景色上。
    d15_change: 背景改为两色交替的竖向条纹，绘制动作只允许落在其中一种背景色条纹上；另一种背景色即使位于几何路径上也必须保留原样。
    retained_parts:
      - 端点颜色与触发颜色的角色不变
      - 最长同色对角线对决定 line_color 的机制不变
      - 触发器仍然在连接对角线上把方向旋转 90 度并发出双向对角射线

  concrete_form_and_trace:
    background_layout:
      - 整张网格按列形成 A/B/A/B/... 两色竖条纹
      - 每个可作用的蓝色端点、蓝色干扰点、品红触发器都只放在同一种条纹颜色上
    visible_effect:
      - 对角线和射线沿几何上连续的 45 度路径延伸
      - 但真正上色时只填活动条纹，所以输出呈现隔列虚线
    example_trace:
      setup:
        - 条纹: 列颜色交替为 grey/orange/grey/orange/...
        - 活动条纹: grey
        - 蓝端点位于 (1,2) 与 (5,6)，品红触发器位于中间的 (3,4)
      original_behavior:
        - 原题会把整条对角线路径 (1,2)(2,3)(3,4)(4,5)(5,6) 都补成蓝色
        - 触发器会从 (3,4) 向两条垂直对角线逐格连续画到边界
      variant_behavior:
        - 新规则只在 grey 条纹列上落笔，于是蓝线只填 (1,2)(3,4)(5,6)，(2,3)(4,5) 作为 orange 条纹被跳过
        - 射线仍沿几何路径传播，但相邻第一格若是 orange 条纹会保留原背景，只在第二格、第四格等 grey 条纹位置留下品红
      difference:
        - 原题的连线是实线；变体是同一路径上的规则性断续线
        - 解题者必须同时识别“哪条对角线要连”和“哪一类背景格允许绘制”

  complexity_calibration:
    why_not_checkerboard:
      - 标准棋盘格在对角步进时不会改变格色，路径上看不到“被跳过的另一色格子”，D15 变化会退化成不可见
      - 竖向双色条纹依旧是“背景双色交替”，但对角移动每一步都会切换背景色，能稳定显现跳格效果
    controlled_freedom:
      - 只保留两种背景色、两种前景色、两类条纹角色
      - 活动条纹可在每题随机是左起偶列或奇列，背景实际颜色也随机，避免把规则退化成固定列奇偶或固定颜色 id

  implicit_assumption_audit:
    original_solver_lines:
      - "bg = get_bg(grid)": 假设背景只有一种颜色；变体中失效，必须改为识别两个主背景色
      - "if v == bg: out[r, c] = line_color": 假设路径上的任意背景格都可填；变体中必须改为只填 active_bg
      - "elif v != line_color: trigger = ...": 在双背景下不能把另一种背景色误判为 trigger，所以条件需改为“非任一背景色且非线色”
      - ray loop 中的 "if out[r, c] == bg": 同样依赖单背景；现在只能在 active_bg 上着色，其余背景格直接穿过
    new_solver_assumptions:
      - 输入的两种最大频数颜色确实是条纹背景色
      - 每列主体上只属于一种背景色，因此可从列内多数票恢复该列的条纹颜色
      - 真正起作用的 line_color 像素都位于同一种活动条纹上
    generator_guarantees:
      - 背景颜色占绝对多数，确保“两种最大频数颜色=背景色”
      - 所有真实端点、触发器和干扰像素都放在活动条纹列
      - 线段长度只取偶数，使两端点落在同种活动条纹上，且路径中一定包含被跳过的非活动格

  placement_legality:
    segments:
      - 对角线段长度限制为 4-16 的偶数，仍接近原题但保证端点同条纹
      - 触发器只放在线段内部且与端点同条纹的位置，因此它们自身也能作为“可绘制条纹”提示
    collision_policy:
      - 任何会真正被画出的蓝色路径格或品红射线格都不得与已有输入像素或已登记的输出格重叠
      - 纯几何上经过但不会上色的非活动条纹格允许与别的“不可绘制路径”共用，从而保持实现简单但视觉仍整洁
    solver_consistency:
      - 生成器按 output-first 构造期望输出，并用 main(inp) 回验

  visual_observability:
    cues_for_humans:
      - 输入中所有蓝/品红功能像素都对齐在同一种底色条纹上
      - 输出新增像素始终只出现在这一种条纹颜色中，另一种条纹沿着本应延伸的几何路径保持空白
    sample_diversity:
      - 背景两色随机、活动条纹随机、斜率可正可负、触发器数量可变
      - 即便只看 2-3 组样例，也能观察到“路径成立但隔条纹绘制”的稳定规律

  cleanliness_and_color_inference:
    cleanliness:
      - 通过限制重叠与保持较低对象数，输出仍能清楚辨认蓝线与品红射线
      - 条纹背景比棋盘格更不拥挤，因为跳过的空格形成整齐节奏
    color_inference:
      - 不依赖颜色 id 排序或映射
      - 哪一种背景色可绘制，完全由输入中功能像素落在哪类条纹上来决定
"""


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _top_two_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, grid.flatten()))
    top = counts.most_common(2)
    assert len(top) == 2, "grid must contain two background colors"
    return int(top[0][0]), int(top[1][0])


def _column_bg_colors(grid: np.ndarray, bg_colors: set[int]) -> list[int]:
    h, w = grid.shape
    col_bgs: list[int] = []
    for c in range(w):
        counts = {bg: 0 for bg in bg_colors}
        for r in range(h):
            v = int(grid[r, c])
            if v in bg_colors:
                counts[v] += 1
        col_bgs.append(max(counts, key=counts.get))
    return col_bgs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the diagonal endpoint pair, but only paint on the designated stripe background, and emit stripe-constrained perpendicular rays from trigger pixels.

    Input structure:
    - The background is made of two alternating vertical stripe colors. Blue candidate endpoints and magenta trigger pixels sit on one of those stripe colors, while the other stripe color is inert.

    Rule:
    - Find the same-color diagonal pair with the longest span and connect the extremes of every matching diagonal group of that color.
    - When the connection hits a differently colored pixel, emit the usual perpendicular diagonal rays, but both segment fills and ray fills may only land on cells belonging to the active stripe background color.

    Key insight:
    - The geometric paths are unchanged from the original task, but painting is filtered by the hidden-under-the-pixels stripe type indicated by the functional input pixels, turning continuous diagonals into regularly skipped dashed traces.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg0, bg1 = _top_two_colors(grid)
    bg_colors = {bg0, bg1}

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) not in bg_colors]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (dist, color)
    for col, pts in by_color.items():
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

    col_bg = _column_bg_colors(grid, bg_colors)
    active_bg = Counter(col_bg[c] for _, c in pts).most_common(1)[0][0]

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
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == active_bg:
                out[r, c] = line_color
            elif v not in bg_colors and v != line_color:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

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
                if int(out[r, c]) == active_bg:
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
_SEG_LEN_RANGE = (4, 16)      # even lengths only; keeps endpoints on the same active stripe
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _bg_at_col(c: int, stripe_colors: tuple[int, int], stripe_phase: int) -> int:
    return stripe_colors[(c + stripe_phase) % 2]


def _make_striped_background(H: int, W: int, stripe_colors: tuple[int, int], stripe_phase: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for c in range(W):
        grid[:, c] = _bg_at_col(c, stripe_colors, stripe_phase)
    return grid


def _is_active_col(c: int, active_bg: int, stripe_colors: tuple[int, int], stripe_phase: int) -> bool:
    return _bg_at_col(c, stripe_colors, stripe_phase) == active_bg


def _paintable_path_cells(path: list[tuple[int, int]], active_bg: int, stripe_colors: tuple[int, int], stripe_phase: int) -> list[tuple[int, int]]:
    return [(r, c) for r, c in path if _is_active_col(c, active_bg, stripe_colors, stripe_phase)]


def _compute_ray_cells(
    triggers: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    H: int,
    W: int,
    active_bg: int,
    stripe_colors: tuple[int, int],
    stripe_phase: int,
) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                if _is_active_col(c, active_bg, stripe_colors, stripe_phase):
                    cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, active_bg: int, stripe_colors: tuple[int, int], stripe_phase: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == active_bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == active_bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    active_bg: int,
    stripe_colors: tuple[int, int],
    stripe_phase: int,
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
):
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

        if not _is_active_col(c0, active_bg, stripe_colors, stripe_phase):
            continue

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        paintable = _paintable_path_cells(path, active_bg, stripe_colors, stripe_phase)
        if any(rc in occupied or rc in drawn for rc in paintable):
            continue
        return path, dc
    return None


def _valid_trigger_positions(path: list[tuple[int, int]], active_bg: int, stripe_colors: tuple[int, int], stripe_phase: int) -> list[tuple[int, int]]:
    candidates = []
    for r, c in path[1:-1]:
        if _is_active_col(c, active_bg, stripe_colors, stripe_phase):
            candidates.append((r, c))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)

    bg_choices = [c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)]
    stripe_colors = tuple(random.sample(bg_choices, 2))
    stripe_phase = random.randint(0, 1)
    active_bg = random.choice(stripe_colors)

    even_lengths = [n for n in range(_SEG_LEN_RANGE[0], min(min(H - 1, W - 1), _SEG_LEN_RANGE[1]) + 1) if n % 2 == 0]
    if not even_lengths:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.choice(even_lengths)

        result = _try_place_segment(
            slope_type,
            seg_len,
            H,
            W,
            active_bg,
            stripe_colors,
            stripe_phase,
            occupied,
            drawn,
        )
        if result is None:
            continue
        path, dc = result

        trigger_candidates = _valid_trigger_positions(path, active_bg, stripe_colors, stripe_phase)
        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(trigger_candidates)))
        trigger_positions = random.sample(trigger_candidates, n_triggers) if n_triggers > 0 else []

        perp_dirs = [(1, -dc), (-1, dc)]
        ray_cells = _compute_ray_cells(
            trigger_positions,
            perp_dirs,
            H,
            W,
            active_bg,
            stripe_colors,
            stripe_phase,
        )

        paintable_path = set(_paintable_path_cells(path, active_bg, stripe_colors, stripe_phase))
        output_cells = (paintable_path - {path[0], path[-1]} - set(trigger_positions)) | ray_cells
        if any(rc in occupied for rc in ray_cells):
            continue
        if any(rc in drawn for rc in output_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
        occupied.update([ep0, ep1])
        occupied.update(trigger_positions)
        drawn.update(output_cells)

    if not segments:
        return None

    inp = _make_striped_background(H, W, stripe_colors, stripe_phase)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            active_cols = [c for c in range(W) if _is_active_col(c, active_bg, stripe_colors, stripe_phase)]
            c = random.choice(active_cols)
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
            r = random.randint(0, H - 1)
            active_cols = [c for c in range(W) if _is_active_col(c, active_bg, stripe_colors, stripe_phase)]
            c = random.choice(active_cols)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, active_bg, stripe_colors, stripe_phase, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
