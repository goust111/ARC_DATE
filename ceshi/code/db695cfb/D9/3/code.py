# Variant of db695cfb
# Dimension: D9 — 多步处理顺序 (multi_step)
# Change: 两步处理——先画蓝色对角线，再从对角线上的品红像素延伸垂直对角线... → 三步处理：连接-延伸-变换
#
# Design analysis:
# spec_variant:
#   preserved_core:
#     - 背景仍是单一主色，且不用 1/2/6。
#     - 蓝色 1 仍表示需要被连成 45 度主对角线的端点。
#     - 品红 6 仍只在“恰好落在主对角线上的位置”才会触发垂直对角射线。
#   new_step_d9:
#     - 在蓝线与品红射线全部画完之后，再检查所有被两种对角斜率同时经过的格子。
#     - 这些交叉格统一改成红色 2；不区分它原先是蓝、品红还是背景。
#   concrete_visual_effect:
#     - 原任务中：触发点会保留品红，主对角线交点若存在会保留蓝。
#     - 本变体中：触发点因同时属于主线与垂直射线，会在第三步变成红色；
#       两条不同斜率的线后续新形成的交点也会变成红色。
#   behavior_trace:
#     - 小例子：两条蓝色主对角线先在中心连成一个 X；其中一条线上还有一个品红触发点。
#     - 原逻辑输出：X 为蓝，触发点发出两条品红射线，中心仍是蓝，触发点仍是品红。
#     - 变体输出：先得到同样的蓝线和品红射线，再把中心交点与触发点都改成红色。
#     - 差异关键：第三步依赖“前两步完成后的整张线稿”，不是只看输入像素。
#   complexity_calibration:
#     - 交叉判定只依赖两种有限的斜率类别（\ 和 /），没有额外颜色映射表，复杂度可控。
#     - 生成器强制包含一组相交的主对角线，因此几乎每个样例都至少有一个“非触发点交叉”。
#       这样人类不会把规则误读成“所有触发点变红”。
#   implicit_assumption_audit:
#     - 原 solver 在画线阶段只靠颜色写入，不记录“哪种斜率经过了该格”；
#       这在原题无害，但变体需要第三步检测交叉，所以必须额外记录每格被哪些斜率访问。
#     - 原 solver 的“遇到非背景即视为触发器”仍成立，但第三步会覆盖这些触发器的最终颜色。
#     - 原 generator 默认避免所有输出线段相交；这里必须放宽该前提，否则第三步除了触发点外不可见。
#   placement_legality:
#     - 生成器固定先放一组必相交的蓝色对角线，只在中心共享一个输出格，输入端点互不重叠。
#     - 品红触发点只放在线段内部且避开中心；其射线不得穿过任何输入像素，避免把端点或干扰点误改色。
#     - 干扰点只能放在所有规划线稿之外，保证 solver 不会把它们当成真实结构的一部分。
#   observability_and_cleanliness:
#     - 红色 2 只在“完成线稿后的交叉格”出现，结构线索直接可见，不依赖颜色编号的隐式排序。
#     - 输出层级稳定为：蓝主线 / 品红射线 / 红交叉标记，视觉上可分辨且不会叠出大片杂色。

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_INTERSECTION_COLOR = 2
_TRIGGER_COLOR = 6

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_CORE_SEG = (1, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_LAYOUT_RETRIES = 120
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 300


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _slope_from_step(dr: int, dc: int) -> int:
    return 1 if dr == dc else -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the blue diagonals, extend magenta diagonals from on-line triggers, then recolor every completed-line crossing.

    Input structure:
    - The background is the majority color. Blue pixels mark endpoint pairs on 45-degree diagonals, and some magenta pixels lie exactly on those diagonals while other magenta pixels may be unrelated distractors.

    Rule:
    - First connect each blue endpoint pair by filling its diagonal.
    - Next, whenever a completed blue diagonal passes through a magenta pixel, emit magenta rays along the perpendicular diagonal direction until the grid edge.
    - Finally, after all blue and magenta strokes exist, recolor every cell touched by both diagonal orientations to red.

    Key insight:
    - The final color change depends on processing order: crossings are detected from the fully built line-work, so both original trigger points and later-created intersections receive the same red marker.
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

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    slope_marks: dict[tuple[int, int], set[int]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def mark(rc: tuple[int, int], slope: int) -> None:
        slope_marks.setdefault(rc, set()).add(slope)

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        slope = _slope_from_step(dr, dc)
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            mark((r, c), slope)
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
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
            ray_slope = _slope_from_step(ddr, ddc)
            mark((r0, c0), ray_slope)
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                mark((r, c), ray_slope)
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    for (r, c), slopes in slope_marks.items():
        if len(slopes) >= 2:
            out[r, c] = _INTERSECTION_COLOR

    return out


def _mark_path_slope(slope_marks: dict[tuple[int, int], set[int]], path: list[tuple[int, int]], slope: int) -> None:
    for rc in path:
        slope_marks.setdefault(rc, set()).add(slope)


def _compute_ray_paths(trigger: tuple[int, int], perp_dirs: list[tuple[int, int]], H: int, W: int) -> list[list[tuple[int, int]]]:
    paths = []
    for ddr, ddc in perp_dirs:
        path = []
        r, c = trigger[0] + ddr, trigger[1] + ddc
        while 0 <= r < H and 0 <= c < W:
            path.append((r, c))
            r += ddr
            c += ddc
        paths.append(path)
    return paths


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    slope_marks: dict[tuple[int, int], set[int]] = {}

    for seg in segments:
        _mark_path_slope(slope_marks, seg["path"], seg["slope"])
        for r, c in seg["path"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR

    for seg in segments:
        for tr, tc in seg["triggers"]:
            for ddr, ddc in seg["perp_dirs"]:
                ray_slope = _slope_from_step(ddr, ddc)
                slope_marks.setdefault((tr, tc), set()).add(ray_slope)
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    slope_marks.setdefault((r, c), set()).add(ray_slope)
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc

    for (r, c), slopes in slope_marks.items():
        if len(slopes) >= 2:
            expected[r, c] = _INTERSECTION_COLOR

    return expected


def _stroke_cells_by_slope(segments: list[dict], H: int, W: int) -> dict[tuple[int, int], set[int]]:
    slope_marks: dict[tuple[int, int], set[int]] = {}
    for seg in segments:
        _mark_path_slope(slope_marks, seg["path"], seg["slope"])
        for tr, tc in seg["triggers"]:
            for ddr, ddc in seg["perp_dirs"]:
                ray_slope = _slope_from_step(ddr, ddc)
                slope_marks.setdefault((tr, tc), set()).add(ray_slope)
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    slope_marks.setdefault((r, c), set()).add(ray_slope)
                    r += ddr
                    c += ddc
    return slope_marks


def _feasible_half_lengths(max_a: int, max_b: int) -> list[tuple[int, int]]:
    pairs = []
    for a in range(1, max_a + 1):
        for b in range(1, max_b + 1):
            total = a + b
            if _SEG_LEN_RANGE[0] <= total <= _SEG_LEN_RANGE[1]:
                pairs.append((a, b))
    return pairs


def _segment_from_center(center: tuple[int, int], slope: int, up: int, down: int) -> dict:
    cr, cc = center
    if slope == 1:
        p = (cr - up, cc - up)
        q = (cr + down, cc + down)
    else:
        p = (cr - up, cc + up)
        q = (cr + down, cc - down)
    path = _diag_path(p, q)
    dr = 1 if q[0] > p[0] else -1
    dc = 1 if q[1] > p[1] else -1
    perp_dirs = [(dr, -dc), (-dr, dc)]
    return {"ep0": p, "ep1": q, "path": path, "slope": slope, "perp_dirs": perp_dirs, "triggers": []}


def _try_build_core_segments(H: int, W: int) -> list[dict] | None:
    for _ in range(_LAYOUT_RETRIES):
        cr = random.randint(1, H - 2)
        cc = random.randint(1, W - 2)
        center = (cr, cc)

        back_pairs = _feasible_half_lengths(min(cr, cc), min(H - 1 - cr, W - 1 - cc))
        slash_pairs = _feasible_half_lengths(min(cr, W - 1 - cc), min(H - 1 - cr, cc))
        if not back_pairs or not slash_pairs:
            continue

        back = _segment_from_center(center, 1, *random.choice(back_pairs))
        slash = _segment_from_center(center, -1, *random.choice(slash_pairs))

        back_candidates = [rc for rc in back["path"][1:-1] if rc != center]
        slash_candidates = [rc for rc in slash["path"][1:-1] if rc != center]
        if not back_candidates or not slash_candidates:
            continue

        n_back = random.randint(_TRIGGERS_PER_CORE_SEG[0], min(_TRIGGERS_PER_CORE_SEG[1], len(back_candidates)))
        n_slash = random.randint(_TRIGGERS_PER_CORE_SEG[0], min(_TRIGGERS_PER_CORE_SEG[1], len(slash_candidates)))
        back["triggers"] = random.sample(back_candidates, n_back)
        slash["triggers"] = random.sample(slash_candidates, n_slash)
        segments = [back, slash]

        input_pixels = {back["ep0"], back["ep1"], slash["ep0"], slash["ep1"], *back["triggers"], *slash["triggers"]}
        ray_hits_input = False
        for seg in segments:
            for trigger in seg["triggers"]:
                for path in _compute_ray_paths(trigger, seg["perp_dirs"], H, W):
                    if any(rc in input_pixels for rc in path):
                        ray_hits_input = True
                        break
                if ray_hits_input:
                    break
            if ray_hits_input:
                break
        if ray_hits_input:
            continue

        slope_marks = _stroke_cells_by_slope(segments, H, W)
        transformed = [rc for rc, slopes in slope_marks.items() if len(slopes) >= 2]
        non_trigger_transformed = [rc for rc in transformed if rc not in back["triggers"] and rc not in slash["triggers"]]
        if not non_trigger_transformed:
            continue

        return segments
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _INTERSECTION_COLOR, _TRIGGER_COLOR)])

    segments = _try_build_core_segments(H, W)
    if segments is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        inp[seg["ep0"]] = _LINE_COLOR
        inp[seg["ep1"]] = _LINE_COLOR
        for tp in seg["triggers"]:
            inp[tp] = _TRIGGER_COLOR

    reserved = set(_stroke_cells_by_slope(segments, H, W))
    occupied = set(zip(*np.where(inp != bg)))

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
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
            if (r, c) in occupied or (r, c) in reserved:
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
