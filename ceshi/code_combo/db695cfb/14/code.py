# Combo variant of db695cfb (index 14)
# combo_spec: D2/1+D7/0+D9/3
# Name: 多色触发+正交延伸+三步处理
#
# Design analysis:
# combo_design:
#   concrete_behavior_trace:
#     - 原题只有一种真实触发色：蓝色端点先连成 45 度主对角线，主线经过的品红像素再沿垂直对角方向发射两条斜向射线。
#     - 融合后保留“蓝色主对角线先补全”的骨架，但把“在路径上遇到什么颜色”提升为第二层规则：
#       品红 6 仍发射垂直对角射线，红色 2 改为发射上下左右四向正交射线。
#     - 第三步不再停留在“画完就结束”，而是回看整张线稿；只要某格被至少两种方向家族同时经过
#       （两种对角斜率之一 + 另一斜率，或对角与横/竖，或横与竖），该格统一改成黄色 4。
#     - 具体小例子：若 \ 主线经过品红触发点，/ 主线经过红触发点，两条蓝主线在中心相交。
#       原题输出只有蓝线和品红斜射线；融合后还会多出红色十字分支，并把两个触发点和中心等交叉格统一涂成黄色。
#   complexity_calibration:
#     - 颜色映射只保留两类真实触发器：品红=斜向垂直对角，红色=正交十字。类别少，2-4 个样例足以支撑归纳。
#     - 第三步使用单一特殊色 4，避免与红色触发器混淆；否则人类难以区分“红色是触发源”还是“红色是交叉结果”。
#     - 正交射线覆盖面积大，因此生成器不追求很多线段，而是固定生成两条相交主线和两个真实触发器，降低冲突并保持画面可读。
#   implicit_assumption_audit:
#     - 原 solver 允许“最长同色对角对”在所有非背景颜色中竞争；融合后红/品红会大量重复，必须排除触发色和交叉色，
#       否则 line_color 可能误判成触发色。
#     - 原 solver 把所有触发器都映射到同一种方向集合；融合后必须按触发颜色分发不同方向。
#     - 原 D9 单变体只需记录两种对角斜率；融合后还要新增横向与纵向两个方向家族，否则无法识别红色十字与其交叉点。
#     - 原题默认输出专用笔画互不相交；融合后第三步恰恰依赖交叉，因此 generator 要主动构造至少一个非触发交叉点。
#   placement_legality:
#     - 生成器先固定一组共享中心的 \ 与 / 主对角线，输入只保留四个端点和两个触发器，中心是输出阶段才出现的交叉格。
#     - 两个真实触发器都放在各自主线内部、避开中心和边界，保证两种分支都能向两个方向明显延伸。
#     - 任意射线都不得穿过任何输入像素（端点、另一个触发器、干扰点），否则 solver 的“只填背景”会留下难解释的断裂。
#     - 干扰蓝点与离线触发点只能放在所有规划笔画之外，确保它们保持静止，不会被第三步意外改色。
#   observability:
#     - 每个样例都强制同时出现一个真实品红触发器和一个真实红色触发器，所以解题者一定能看到“颜色决定分支几何”。
#     - 每个样例都强制存在主线中心交叉点，且该点不是输入触发器；这使第三步可被看成“所有交叉都变黄”，
#       而不是“只有触发点变黄”。
#   cleanliness:
#     - 真实触发器固定为两个，避免红色正交十字把整图刷满。
#     - 交叉变换只改色、不扩形，输出层级稳定为：蓝骨架 / 品红斜枝 / 红色十字枝 / 黄色交叉标记。
#   generator_contract:
#     - main() 的新假设是：至少存在一组蓝色对角端点、至少一个路径品红触发器、至少一个路径红色触发器、
#       且至少一个非触发交叉点。generate_input() 通过“相交双主线 + 双真实触发器 + 预留所有笔画格”来保证这些前提。

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_ORTH_TRIGGER_COLOR = 2
_INTERSECTION_COLOR = 4
_DIAG_TRIGGER_COLOR = 6
_TRIGGER_COLORS = (_ORTH_TRIGGER_COLOR, _DIAG_TRIGGER_COLOR)
_NON_LINE_COLORS = {_ORTH_TRIGGER_COLOR, _INTERSECTION_COLOR, _DIAG_TRIGGER_COLOR}

_MARK_DIAG_POS = 1
_MARK_DIAG_NEG = -1
_MARK_H = 2
_MARK_V = 3

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEG_LEN_RANGE = (3, 16)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_PATH_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_LAYOUT_RETRIES = 180
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _diag_mark(dr: int, dc: int) -> int:
    return _MARK_DIAG_POS if dr == dc else _MARK_DIAG_NEG


def _trigger_ray_specs(color: int, dr: int, dc: int) -> list[tuple[int, int, int]]:
    if color == _DIAG_TRIGGER_COLOR:
        return [
            (dr, -dc, _diag_mark(dr, -dc)),
            (-dr, dc, _diag_mark(-dr, dc)),
        ]
    if color == _ORTH_TRIGGER_COLOR:
        return [
            (-1, 0, _MARK_V),
            (1, 0, _MARK_V),
            (0, -1, _MARK_H),
            (0, 1, _MARK_H),
        ]
    return []


def _iter_ray_cells(
    start: tuple[int, int],
    step: tuple[int, int],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    r, c = start[0] + step[0], start[1] + step[1]
    cells: list[tuple[int, int]] = []
    while 0 <= r < H and 0 <= c < W:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the diagonal blue connectors, branch by trigger color, then recolor every resulting crossing.

    Input structure:
    - The background is the majority color. Blue pixels define endpoint pairs on 45-degree diagonals, while red and magenta pixels may sit on those diagonal interiors as color-coded triggers; off-path triggers can also appear as inert distractors.

    Rule:
    - First fill each qualifying blue diagonal segment between its extreme endpoints.
    - Next, every on-path magenta pixel emits perpendicular diagonal rays in magenta, and every on-path red pixel emits horizontal and vertical rays in red, always extending to the boundary while only painting background cells.
    - Finally, after all blue, magenta, and red strokes have been established, recolor every cell touched by at least two distinct direction families to yellow.

    Key insight:
    - The puzzle now requires two independent inferences at once: trigger color chooses the branch geometry, and the final yellow marks are not input objects but late-stage intersection labels computed from the fully constructed line network.
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

    line_candidates = {col: pts for col, pts in by_color.items() if col not in _NON_LINE_COLORS}

    best = None  # (distance, color)
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
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    mark_map: dict[tuple[int, int], set[int]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def mark(cell: tuple[int, int], family: int) -> None:
        mark_map.setdefault(cell, set()).add(family)

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        main_family = _diag_mark(dr, dc)
        for r, c in path:
            mark((r, c), main_family)
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                ray_specs = _trigger_ray_specs(v, dr, dc)
                if ray_specs:
                    triggers.setdefault((r, c, v), set()).update(ray_specs)

    for group in groups_a.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for group in groups_b.values():
        if len(group) >= 2:
            group = sorted(group)
            draw_segment(group[0], group[-1])

    for (r0, c0, tcol), ray_specs in triggers.items():
        for ddr, ddc, family in ray_specs:
            mark((r0, c0), family)
            for r, c in _iter_ray_cells((r0, c0), (ddr, ddc), h, w):
                mark((r, c), family)
                if int(out[r, c]) == bg:
                    out[r, c] = tcol

    for (r, c), families in mark_map.items():
        if len(families) >= 2:
            out[r, c] = _INTERSECTION_COLOR

    return out


def _mark_path(mark_map: dict[tuple[int, int], set[int]], path: list[tuple[int, int]], family: int) -> None:
    for cell in path:
        mark_map.setdefault(cell, set()).add(family)


def _segment_from_center(center: tuple[int, int], family: int, up: int, down: int) -> dict:
    cr, cc = center
    if family == _MARK_DIAG_POS:
        ep0 = (cr - up, cc - up)
        ep1 = (cr + down, cc + down)
        dc = 1
    else:
        ep0 = (cr - up, cc + up)
        ep1 = (cr + down, cc - down)
        dc = -1
    return {
        "ep0": ep0,
        "ep1": ep1,
        "path": _diag_path(ep0, ep1),
        "family": family,
        "dr": 1,
        "dc": dc,
        "triggers": [],
    }


def _feasible_half_lengths(max_up: int, max_down: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for up in range(1, max_up + 1):
        for down in range(1, max_down + 1):
            total = up + down
            if _SEG_LEN_RANGE[0] <= total <= _SEG_LEN_RANGE[1]:
                pairs.append((up, down))
    return pairs


def _core_trigger_candidates(
    path: list[tuple[int, int]],
    center: tuple[int, int],
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    candidates = []
    for r, c in path[1:-1]:
        if (r, c) == center:
            continue
        if 0 < r < H - 1 and 0 < c < W - 1:
            candidates.append((r, c))
    return candidates


def _stroke_marks(segments: list[dict], H: int, W: int) -> dict[tuple[int, int], set[int]]:
    mark_map: dict[tuple[int, int], set[int]] = {}
    for seg in segments:
        _mark_path(mark_map, seg["path"], seg["family"])
    for seg in segments:
        for tr, tc, color in seg["triggers"]:
            for ddr, ddc, family in _trigger_ray_specs(color, seg["dr"], seg["dc"]):
                mark_map.setdefault((tr, tc), set()).add(family)
                for cell in _iter_ray_cells((tr, tc), (ddr, ddc), H, W):
                    mark_map.setdefault(cell, set()).add(family)
    return mark_map


def _input_pixels(segments: list[dict]) -> set[tuple[int, int]]:
    pixels: set[tuple[int, int]] = set()
    for seg in segments:
        pixels.add(seg["ep0"])
        pixels.add(seg["ep1"])
        for r, c, _ in seg["triggers"]:
            pixels.add((r, c))
    return pixels


def _ray_hits_input(segments: list[dict], H: int, W: int) -> bool:
    inputs = _input_pixels(segments)
    for seg in segments:
        for tr, tc, color in seg["triggers"]:
            for ddr, ddc, _ in _trigger_ray_specs(color, seg["dr"], seg["dc"]):
                for cell in _iter_ray_cells((tr, tc), (ddr, ddc), H, W):
                    if cell in inputs:
                        return True
    return False


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    mark_map: dict[tuple[int, int], set[int]] = {}

    for seg in segments:
        _mark_path(mark_map, seg["path"], seg["family"])
        for r, c in seg["path"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR

    for seg in segments:
        for tr, tc, color in seg["triggers"]:
            for ddr, ddc, family in _trigger_ray_specs(color, seg["dr"], seg["dc"]):
                mark_map.setdefault((tr, tc), set()).add(family)
                for r, c in _iter_ray_cells((tr, tc), (ddr, ddc), H, W):
                    mark_map.setdefault((r, c), set()).add(family)
                    if expected[r, c] == bg:
                        expected[r, c] = color

    for (r, c), families in mark_map.items():
        if len(families) >= 2:
            expected[r, c] = _INTERSECTION_COLOR

    return expected


def _try_build_core_segments(H: int, W: int) -> list[dict] | None:
    if H < 5 or W < 5:
        return None

    for _ in range(_LAYOUT_RETRIES):
        cr = random.randint(1, H - 2)
        cc = random.randint(1, W - 2)
        center = (cr, cc)

        back_pairs = _feasible_half_lengths(min(cr, cc), min(H - 1 - cr, W - 1 - cc))
        slash_pairs = _feasible_half_lengths(min(cr, W - 1 - cc), min(H - 1 - cr, cc))
        if not back_pairs or not slash_pairs:
            continue

        back = _segment_from_center(center, _MARK_DIAG_POS, *random.choice(back_pairs))
        slash = _segment_from_center(center, _MARK_DIAG_NEG, *random.choice(slash_pairs))

        back_candidates = _core_trigger_candidates(back["path"], center, H, W)
        slash_candidates = _core_trigger_candidates(slash["path"], center, H, W)
        if not back_candidates or not slash_candidates:
            continue

        colors = [_DIAG_TRIGGER_COLOR, _ORTH_TRIGGER_COLOR]
        random.shuffle(colors)
        back["triggers"] = [(*random.choice(back_candidates), colors[0])]
        slash["triggers"] = [(*random.choice(slash_candidates), colors[1])]
        segments = [back, slash]

        if _ray_hits_input(segments, H, W):
            continue

        mark_map = _stroke_marks(segments, H, W)
        trigger_cells = {(r, c) for seg in segments for r, c, _ in seg["triggers"]}
        non_trigger_intersections = [cell for cell, families in mark_map.items() if len(families) >= 2 and cell not in trigger_cells]
        if center not in non_trigger_intersections:
            continue

        red_trigger = next((seg["triggers"][0] for seg in segments if seg["triggers"][0][2] == _ORTH_TRIGGER_COLOR), None)
        mag_trigger = next((seg["triggers"][0] for seg in segments if seg["triggers"][0][2] == _DIAG_TRIGGER_COLOR), None)
        assert red_trigger is not None and mag_trigger is not None

        red_cells = set()
        for ddr, ddc, _ in _trigger_ray_specs(_ORTH_TRIGGER_COLOR, 1, 1):
            red_cells.update(_iter_ray_cells((red_trigger[0], red_trigger[1]), (ddr, ddc), H, W))
        mag_cells = set()
        owner = next(seg for seg in segments if seg["triggers"][0][2] == _DIAG_TRIGGER_COLOR)
        for ddr, ddc, _ in _trigger_ray_specs(_DIAG_TRIGGER_COLOR, owner["dr"], owner["dc"]):
            mag_cells.update(_iter_ray_cells((mag_trigger[0], mag_trigger[1]), (ddr, ddc), H, W))
        if not red_cells or not mag_cells:
            continue

        return segments

    return None


def _shares_diagonal(cell: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    r, c = cell
    return any(abs(r - rr) == abs(c - cc) and r != rr for rr, cc in others)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([color for color in range(10) if color not in (_LINE_COLOR, _ORTH_TRIGGER_COLOR, _INTERSECTION_COLOR, _DIAG_TRIGGER_COLOR)])

    segments = _try_build_core_segments(H, W)
    if segments is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        inp[seg["ep0"]] = _LINE_COLOR
        inp[seg["ep1"]] = _LINE_COLOR
        for r, c, color in seg["triggers"]:
            inp[r, c] = color

    reserved = set(_stroke_marks(segments, H, W))
    occupied = set(zip(*np.where(inp != bg)))

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            cell = (random.randint(0, H - 1), random.randint(0, W - 1))
            if cell in occupied or cell in reserved:
                continue
            if _shares_diagonal(cell, all_line_positions):
                continue
            inp[cell] = _LINE_COLOR
            occupied.add(cell)
            all_line_positions.append(cell)
            break

    for _ in range(random.randint(*_OFF_PATH_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            cell = (random.randint(0, H - 1), random.randint(0, W - 1))
            if cell in occupied or cell in reserved:
                continue
            inp[cell] = random.choice(_TRIGGER_COLORS)
            occupied.add(cell)
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
