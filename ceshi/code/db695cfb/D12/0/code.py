# Variant of db695cfb
# Dimension: D12 — 对角线方向约束 (meta)
# Change: 对角线方向由两个蓝色像素的相对位置唯一确定（右上-左下或右下-左上）... → 双方向对角线（X形）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D12 variant
# ----------------------------------
# spec:
#   variant_behavior:
#     - 蓝色端点仍然只给出一条 45 度对角线的两个端点。
#     - 输出不再只补全这条直接对角线；还要在同一个包围方框里补上另一条垂直对角线。
#     - 两条等长对角线共同构成局部 X 形。第二条线是该方框的另一条对角线，因此总能几何上经过同一个中点。
#     - 品红像素只要落在这两条线中的任意一条上，就会沿“另一种对角方向”向两端发射射线直到边界。
#   concrete_trace:
#     - 原题小例子：蓝点在 (1,1) 与 (5,5)，品红在 (3,3)。输出只会补出 \ 对角线，并从 (3,3) 向 / 方向发射。
#     - 变体同例：除了补出 (1,1)->(5,5) 这条 \ 线，还要补出同一 5x5 方框里的 / 线：(1,5)->(5,1)。
#     - 若输入额外有品红在 (2,4)，原题不会理它；变体会因为它位于新增的 / 线上而沿 \ 方向发射。
#   complexity_calibration:
#     - 第二条线若做到整格边界，会让蓝线和品红射线的角色混淆；因此只画“同方框、同长度”的另一条对角线。
#     - 生成器要求至少出现一次仅位于新增对角线上的触发点，这样样例不会退化成“原题 + 多一条蓝线但没人用它”。
#     - 触发点避免放在两条线重合的中心格（偶数长度时会出现），避免四向同时发射造成视觉噪声。
#   implicit_assumptions_review:
#     - 原 main() 的 draw_segment 默认“一条主线只对应一组垂直方向”，这个前提在变体中失效，必须显式分别绘制 direct/cross 两条线。
#     - 原生成器的 drawn 只覆盖主对角线和射线；变体里若不把 cross 线也算进 drawn，后续对象会碰撞到新增蓝线。
#     - 原任务只在 direct 线上选 trigger；变体里若还这样做，solver 虽改了也难以从样例观察出“任一条线都能触发”。
#     - line_color 的识别仍沿用“最长同色对角配对”，因此生成器必须回避让品红形成更强的对角候选；最后用 oracle 反查兜底。
#   placement_legality:
#     - 每个 X 形对象占用一个轴对齐正方形的两条对角线；两条线、端点、trigger、射线都不得和已有 occupied/drawn 重叠。
#     - 蓝端点只放在 direct 线两端；cross 线端点保持为空背景，由输出再补出来，避免输入提示过多。
#     - 品红 distractor 只能落在完全无关的位置，不能踩到任一 X 线或射线路径。
#   observability_and_cleanliness:
#     - 观察者能直接看到：同一对蓝点输出成 X，而不是单线；并且某些输入里的品红虽然不在蓝点连线那条线上，输出仍会发射。
#     - 通过限制每段 direct/cross trigger 各至多一个，并拒绝路径/射线重叠，保持图面整洁。
#   new_solver_assumptions:
#     - solver 假设真正的蓝色对象仍由同色对角端点定义；generator 用“未配对蓝点不与任何蓝点成对角”来保证这一点。
#     - solver 会对同一条 direct 对角线的极端点构造一个 X；generator 因此不能让三个蓝点共线，否则会被合并成更长对象。


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _cross_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the other diagonal of the square spanned by the direct diagonal p->q."""
    r0, c0 = p
    r1, c1 = q
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "endpoints must form a 45-degree diagonal"

    r_lo, r_hi = sorted((r0, r1))
    c_lo, c_hi = sorted((c0, c1))
    if dr * dc > 0:
        return [(r_lo + k, c_hi - k) for k in range(abs(dr) + 1)]
    return [(r_lo + k, c_lo + k) for k in range(abs(dr) + 1)]


def _pick_line_color(by_color: dict[int, list[tuple[int, int]]]) -> int:
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
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand
    assert best is not None, "at least one diagonal-aligned same-color pair must exist"
    return best[1]


def _segment_dirs(p: tuple[int, int], q: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dr = 1 if q[0] > p[0] else -1
    dc = 1 if q[1] > p[1] else -1
    direct_dirs = [(dr, dc), (-dr, -dc)]
    cross_dirs = [(dr, -dc), (-dr, dc)]
    return direct_dirs, cross_dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each blue endpoint pair into an X, then emit diagonal rays from non-blue pixels touched by either branch.

    Input structure:
    - The background is the most common color. One non-background color marks diagonal endpoint pairs, while magenta trigger pixels may lie on the direct diagonal or on the hidden crossing diagonal of the same square.

    Rule:
    - For every same-color diagonal endpoint pair, fill both diagonals of its bounding square: the direct endpoint-to-endpoint diagonal and the perpendicular diagonal through the same midpoint.
    - Whenever either diagonal passes through a differently-colored pixel, extend that pixel's color along the other diagonal orientation in both directions until the grid edge.

    Key insight:
    - The blue pair now defines a whole X-shaped scaffold, so trigger pixels can activate from either branch; the branch they sit on determines which diagonal direction their rays follow.
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

    line_color = _pick_line_color(by_color)
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant (backslash)
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant (slash)
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_path(path: list[tuple[int, int]], ray_dirs: list[tuple[int, int]]) -> None:
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                triggers.setdefault((r, c, v), set()).update(ray_dirs)

    def draw_x_from_pair(p: tuple[int, int], q: tuple[int, int]) -> None:
        direct_path = _diag_path(p, q)
        cross_path = _cross_path(p, q)
        direct_dirs, cross_dirs = _segment_dirs(p, q)
        draw_path(direct_path, cross_dirs)
        draw_path(cross_path, direct_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_x_from_pair(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_x_from_pair(g[0], g[-1])

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
_H_RANGE = (8, 25)                  # grid height range
_W_RANGE = (9, 21)                  # grid width range
_LINE_COLOR = 1                     # color for diagonal segment endpoints/fills
_TRIGGER_COLOR = 6                  # color for trigger pixels and diagonal rays
_SEGMENT_COUNT = (1, 4)             # number of X-shape seeds
_SEG_LEN_RANGE = (3, 16)            # diagonal length in steps between blue endpoints
_DIRECT_TRIGGERS_PER_SEG = (0, 1)   # triggers placed on the direct blue-to-blue diagonal
_CROSS_TRIGGERS_PER_SEG = (0, 1)    # triggers placed on the crossing hidden diagonal

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)      # blue distractor pixels that do not form a pair
_OFF_DIAG_TRIGGERS = (0, 4)         # magenta distractors off every active diagonal

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _compute_ray_cells(trigger_dir_map, H, W):
    cells = set()
    for (tr, tc), dirs in trigger_dir_map.items():
        for ddr, ddc in dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp, bg, segments):
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path_cells"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for (tr, tc), dirs in seg["trigger_dir_map"].items():
            for ddr, ddc in dirs:
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
            p = (r0, c0)
            q = (r0 + seg_len, c0 + seg_len)
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            p = (r0, c0)
            q = (r0 + seg_len, c0 - seg_len)

        direct_path = _diag_path(p, q)
        cross_path = _cross_path(p, q)
        path_cells = set(direct_path) | set(cross_path)
        if any(rc in occupied or rc in drawn for rc in path_cells):
            continue
        return p, q, direct_path, cross_path
    return None


def _exclusive_interior(path, other_path):
    other = set(other_path)
    return [cell for cell in path[1:-1] if cell not in other]


def _choose_triggers(direct_path, cross_path, require_cross):
    direct_candidates = _exclusive_interior(direct_path, cross_path)
    cross_candidates = _exclusive_interior(cross_path, direct_path)

    direct_count_max = min(_DIRECT_TRIGGERS_PER_SEG[1], len(direct_candidates))
    cross_count_max = min(_CROSS_TRIGGERS_PER_SEG[1], len(cross_candidates))

    if require_cross and cross_count_max == 0:
        return None

    direct_count = random.randint(_DIRECT_TRIGGERS_PER_SEG[0], direct_count_max) if direct_count_max >= _DIRECT_TRIGGERS_PER_SEG[0] else 0
    cross_min = 1 if require_cross else _CROSS_TRIGGERS_PER_SEG[0]
    if cross_count_max < cross_min:
        return None
    cross_count = random.randint(cross_min, cross_count_max) if cross_count_max >= cross_min else 0

    direct_triggers = random.sample(direct_candidates, direct_count) if direct_count else []
    cross_triggers = random.sample(cross_candidates, cross_count) if cross_count else []
    return direct_triggers, cross_triggers


def _line_color_selected(inp, bg):
    non_bg = [(r, c, int(inp[r, c])) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if int(inp[r, c]) != bg]
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))
    return _pick_line_color(by_color)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()   # input pixels
    drawn = set()      # output-only cells already reserved
    segments = []
    has_cross_trigger = False

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        p, q, direct_path, cross_path = result

        trigger_choice = _choose_triggers(direct_path, cross_path, require_cross=not has_cross_trigger)
        if trigger_choice is None:
            continue
        direct_triggers, cross_triggers = trigger_choice

        direct_dirs, cross_dirs = _segment_dirs(p, q)
        trigger_dir_map: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for pos in direct_triggers:
            trigger_dir_map.setdefault(pos, set()).update(cross_dirs)
        for pos in cross_triggers:
            trigger_dir_map.setdefault(pos, set()).update(direct_dirs)

        path_cells = set(direct_path) | set(cross_path)
        trigger_cells = set(trigger_dir_map)
        ray_cells = _compute_ray_cells(trigger_dir_map, H, W)

        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue
        if any(rc in path_cells - trigger_cells for rc in ray_cells):
            continue

        segments.append(
            {
                "endpoints": (p, q),
                "path_cells": path_cells,
                "trigger_dir_map": trigger_dir_map,
            }
        )

        occupied.update([p, q])
        occupied.update(trigger_cells)
        drawn.update(path_cells - {p, q} - trigger_cells)
        drawn.update(ray_cells)
        has_cross_trigger = has_cross_trigger or bool(cross_triggers)

    if not segments or not has_cross_trigger:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        p, q = seg["endpoints"]
        inp[p] = _LINE_COLOR
        inp[q] = _LINE_COLOR
        for tp in seg["trigger_dir_map"]:
            inp[tp] = _TRIGGER_COLOR

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
            if (r, c) in occupied or (r, c) in drawn:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        if _line_color_selected(inp, bg) != _LINE_COLOR:
            return None
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
