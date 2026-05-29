# Variant of db695cfb
# Dimension: D12 — 对角线方向约束 (meta)
# Change: 对角线方向由两个蓝色像素的相对位置唯一确定（右上-左下或右下-左上）... → 对角线方向由第三个标记决定
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: 两个蓝色像素本身就在一条 45 度对角线上，因此连接方向天然唯一。
#     new: 两个蓝色像素只给出一个正方形的左上/右下角；真正要画哪条对角线由第三个“指示像素”决定。
#   object_layout:
#     actionable_object:
#       - 两个蓝色像素(1)位于同一个正方形的左上角和右下角。
#       - 一个红色指示像素(2)位于该正方形中心格的上下左右四邻之一。
#       - 若红色像素在中心的上/下方，则选择 "\" 对角线。
#       - 若红色像素在中心的左/右方，则选择 "/" 对角线。
#       - 0-2 个品红像素(6)位于被选中的那条对角线上，它们继续像原题一样发出垂直对角射线。
#     distractors:
#       - 少量额外蓝点，但不与任何蓝点形成候选正方形角对。
#       - 少量额外品红点，但不在任何对象正方形内部。
#   behavior_trace:
#     example:
#       - 蓝点在 (1,1) 和 (5,5)，它们只定义了一个边长 4 的正方形。
#       - 原题逻辑会直接画 "\": (1,1)(2,2)(3,3)(4,4)(5,5)。
#       - 新变体若红点在中心左侧 (3,2)，则必须改画 "/": (1,5)(2,4)(3,3)(4,2)(5,1)。
#       - 若品红点在 (2,4)，它位于 "/" 上，因此向 "\" 方向两侧发射品红射线。
#     difference:
#       - 原题中蓝点本身决定了方向；这里蓝点只决定“哪一个正方形”，方向完全由第三标记编码。
#   complexity_calibration:
#     - 指示规则只分两档：上下= "\"，左右= "/"。分支数低，样例中容易重复支撑。
#     - 保留原题的品红射线机制，使变体既明显可见，又不把任务改成全新题。
#     - 只允许偶数边长正方形，这样中心是整数格，红色指示像素可以稳定地放在中心四邻。
#   implicit_assumption_audit:
#     original_solver_lines_that_break:
#       - “按蓝点分组到两种对角线，再连接每组极端点” 依赖蓝点本身就在目标对角线上；新题不成立。
#       - “触发点一定落在蓝点连线之上” 依赖唯一方向；新题要改成先读红色指示，再找对应对角线上的品红点。
#       - “最长同色对角对就是完整线段” 在新题中只能用来识别蓝色候选角对，不能直接决定输出线方向。
#     new_solver_assumptions:
#       - 真正的蓝色对象一定是左上/右下的偶数跨度角对。
#       - 每个真实对象在正方形中心四邻中恰有一个红色指示像素。
#       - 对象之间不重叠，正方形内部不会混入其他对象的输入像素。
#       - 真实触发点只出现在被选中的对角线上，颜色不是背景/蓝色/红色。
#   placement_legality:
#     - 生成器保留 output-first 流程，但对象层面改为“正方形 + 被选中对角线”。
#     - 每个正方形整块区域都视为保留区，避免对象互穿造成识别混乱。
#     - 射线不能撞到任何输入像素，也不能穿过其他对象的保留区。
#   visual_observability:
#     - 同样的蓝色角对，在不同样例里会因为红点在“上下”或“左右”而出现不同对角线，肉眼可见。
#     - "/" 情况下画出的蓝线甚至不经过蓝角点本身，因此方向变化不会退化成原题。
#   visual_cleanliness:
#     - 对象正方形区域彼此分离，红色指示像素固定在正方形中心附近，视觉归属清楚。
#     - 品红杂点不进入对象正方形，避免与红色指示或有效触发混淆。
#   color_inference:
#     - 颜色功能由空间角色支撑：蓝=角点/线，红=方向指示，品红=触发与射线。
#     - 不做任何基于颜色 id 数值大小的排序或映射。

import random

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use blue corner pairs to identify squares, then let a third marker choose which diagonal is drawn before emitting perpendicular rays.

    Input structure:
    - bg is the dominant color. A real object contains two blue pixels at the top-left and bottom-right corners of an even-sized square, one red selector near that square's center, and optional magenta trigger pixels lying on the selector-chosen diagonal.

    Rule:
    - For each blue corner pair that has a matching red selector, draw the square's chosen diagonal in blue: selector above/below center means "\", selector left/right means "/".
    - Whenever that chosen diagonal passes through a non-blue, non-red pixel, treat it as a trigger and extend magenta rays along the perpendicular diagonal direction until the grid edge.

    Key insight:
    - The blue pixels no longer encode the drawing direction by themselves; they only define a square. The red pixel is the instruction token that disambiguates which of the two diagonals inside that square is active.
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

    def diag_path_from_square(tl: tuple[int, int], br: tuple[int, int], slope: str) -> list[tuple[int, int]]:
        r0, c0 = tl
        r1, c1 = br
        steps = r1 - r0
        if slope == "backslash":
            return [(r0 + k, c0 + k) for k in range(steps + 1)]
        return [(r0 + k, c1 - k) for k in range(steps + 1)]

    def selector_neighbors(center: tuple[int, int]) -> list[tuple[tuple[int, int], str]]:
        cr, cc = center
        out_neighbors = []
        for pos in ((cr - 1, cc), (cr + 1, cc)):
            if 0 <= pos[0] < h and 0 <= pos[1] < w:
                out_neighbors.append((pos, "backslash"))
        for pos in ((cr, cc - 1), (cr, cc + 1)):
            if 0 <= pos[0] < h and 0 <= pos[1] < w:
                out_neighbors.append((pos, "slash"))
        return out_neighbors

    def candidate_pairs(col: int) -> list[dict]:
        pts = sorted(by_color.get(col, []))
        pairs = []
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr < 4 or dr != dc or dr % 2 != 0:
                    continue
                center = ((r0 + r1) // 2, (c0 + c1) // 2)
                nearby = []
                for pos, slope in selector_neighbors(center):
                    v = int(grid[pos])
                    if v != bg and v != col:
                        nearby.append((pos, v, slope))
                if nearby:
                    pairs.append(
                        {
                            "tl": (r0, c0),
                            "br": (r1, c1),
                            "dist": dr,
                            "center": center,
                            "nearby": nearby,
                        }
                    )
        return pairs

    best = None  # (candidate_count, max_dist, color)
    candidate_cache: dict[int, list[dict]] = {}
    for col in by_color:
        pairs = candidate_pairs(col)
        candidate_cache[col] = pairs
        if not pairs:
            continue
        score = (len(pairs), max(pair["dist"] for pair in pairs), col)
        if best is None or score > best:
            best = score

    assert best is not None, "at least one selector-supported blue square must exist"

    _, _, line_color = best
    line_candidates = candidate_cache[line_color]

    selector_counts: dict[int, int] = {}
    for cand in line_candidates:
        for _, col, _ in cand["nearby"]:
            selector_counts[col] = selector_counts.get(col, 0) + 1

    selector_color = max(selector_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    segments = []
    used_endpoints = set()
    for cand in sorted(line_candidates, key=lambda item: item["dist"], reverse=True):
        tl = cand["tl"]
        br = cand["br"]
        if tl in used_endpoints or br in used_endpoints:
            continue

        center = cand["center"]
        slope = None
        for pos, candidate_slope in selector_neighbors(center):
            if int(grid[pos]) == selector_color:
                slope = candidate_slope
                break
        if slope is None:
            continue

        path = diag_path_from_square(tl, br, slope)
        segments.append((tl, br, slope, path))
        used_endpoints.add(tl)
        used_endpoints.add(br)

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    for tl, br, slope, path in segments:
        dc = 1 if slope == "backslash" else -1
        perp_dirs = [(1, -dc), (-1, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color and v != selector_color:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

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
_SELECTOR_COLOR = 2
_SEGMENT_COUNT = (1, 4)
_SQUARE_DIAG_RANGE = (4, 16)   # even corner-to-corner diagonal length
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _selector_choices(center: tuple[int, int], slope: str, H: int, W: int) -> list[tuple[int, int]]:
    cr, cc = center
    if slope == "backslash":
        choices = [(cr - 1, cc), (cr + 1, cc)]
    else:
        choices = [(cr, cc - 1), (cr, cc + 1)]
    return [(r, c) for r, c in choices if 0 <= r < H and 0 <= c < W]


def _diag_path_from_square(tl: tuple[int, int], br: tuple[int, int], slope: str) -> list[tuple[int, int]]:
    r0, c0 = tl
    r1, c1 = br
    steps = r1 - r0
    if slope == "backslash":
        return [(r0 + k, c0 + k) for k in range(steps + 1)]
    return [(r0 + k, c1 - k) for k in range(steps + 1)]


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


def _build_expected(inp, bg, segments):
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in seg["trigger_positions"]:
            for ddr, ddc in seg["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _try_place_segment(H, W, occupied, drawn, reserved):
    even_lengths = [d for d in range(_SQUARE_DIAG_RANGE[0], min(_SQUARE_DIAG_RANGE[1], min(H - 1, W - 1)) + 1) if d % 2 == 0]
    if not even_lengths:
        return None

    for _ in range(_SEGMENT_PLACE_RETRIES):
        dist = random.choice(even_lengths)
        r0 = random.randint(0, H - 1 - dist)
        c0 = random.randint(0, W - 1 - dist)
        tl = (r0, c0)
        br = (r0 + dist, c0 + dist)
        bbox_cells = {(r, c) for r in range(r0, r0 + dist + 1) for c in range(c0, c0 + dist + 1)}
        if bbox_cells & reserved:
            continue

        slope = random.choice(["backslash", "slash"])
        center = ((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
        selector_options = _selector_choices(center, slope, H, W)
        if not selector_options:
            continue
        selector = random.choice(selector_options)

        path = _diag_path_from_square(tl, br, slope)
        path_interior = path[1:-1]
        if selector in occupied or selector in drawn:
            continue

        n_triggers = random.randint(_TRIGGERS_PER_SEG[0], min(_TRIGGERS_PER_SEG[1], len(path_interior)))
        trigger_positions = random.sample(path_interior, n_triggers) if n_triggers > 0 else []

        path_cells = set(path)
        input_cells = {tl, br, selector, *trigger_positions}
        output_only = path_cells - {tl, br} - set(trigger_positions)
        if input_cells & occupied:
            continue
        if (input_cells | output_only) & drawn:
            continue

        dc = 1 if slope == "backslash" else -1
        perp_dirs = [(1, -dc), (-1, dc)]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
        if ray_cells & occupied:
            continue
        if ray_cells & drawn:
            continue
        if ray_cells & reserved:
            continue

        return {
            "tl": tl,
            "br": br,
            "selector": selector,
            "slope": slope,
            "path": path,
            "trigger_positions": trigger_positions,
            "perp_dirs": perp_dirs,
            "bbox_cells": bbox_cells,
            "drawn_cells": output_only | ray_cells,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR, _SELECTOR_COLOR)])

    occupied = set()   # all non-bg input pixels
    drawn = set()      # output-only pixels
    reserved = set()   # full square regions to keep objects visually separate
    segments = []

    target_segments = random.randint(*_SEGMENT_COUNT)
    for _ in range(target_segments):
        seg = _try_place_segment(H, W, occupied, drawn, reserved)
        if seg is None:
            continue
        segments.append(seg)
        occupied.update({seg["tl"], seg["br"], seg["selector"], *seg["trigger_positions"]})
        drawn.update(seg["drawn_cells"])
        reserved.update(seg["bbox_cells"])

    if not segments:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        inp[seg["tl"]] = _LINE_COLOR
        inp[seg["br"]] = _LINE_COLOR
        inp[seg["selector"]] = _SELECTOR_COLOR
        for tp in seg["trigger_positions"]:
            inp[tp] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn or (r, c) in reserved:
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
            if (r, c) in occupied or (r, c) in drawn or (r, c) in reserved:
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
