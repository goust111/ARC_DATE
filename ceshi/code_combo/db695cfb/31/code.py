# Combo variant of db695cfb (index 31)
# combo_spec: D4/2+D5/3
# Name: 渐变主路径+渐变延伸线

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the fused variant
# a) 具体化与行为追踪:
#    原题把一对蓝色端点之间整段填成纯蓝, 并把对角线上遇到的品红触发点向垂直对角方向发射纯品红射线。
#    这里改成共享中间色的双向渐变:
#    - 主路径: 远离触发点的补线仍是 1, 但紧邻品红触发点的补线改成 8, 因而沿线会出现 `1, 8, 6, 8, 1`。
#    - 延伸线: 从触发点往外的第一格是 8, 第二格及更远是 1, 因而沿射线会出现 `6, 8, 1, 1, ...`。
#    ARC 调色板里没有真正的“深蓝”, 因此用唯一额外的蓝系颜色 8 作为近触发点的共享过渡色。
# b) 复杂度校准:
#    只引入一个辅助色 8, 并且它在两种轨迹中承担同一语义: “离品红一格”。这样既明显区别于原题,
#    又不会变成过多颜色的记忆题。每个样例都至少有一个触发点, 保证这条规律必定出现。
# c) 隐式假设审查:
#    - 原 solver 依赖“输入里只有端点蓝和触发品红”。新规则仍保持输入不出现 8, 因此找线色的逻辑不变。
#    - 原 solver 默认输出只在 bg 上补画。这里仍不重涂输入像素; 端点保持 1, 触发点保持 6。
#    - 如果不同段或射线重叠, 渐变优先级会变得含糊。生成器因此禁止路径/射线之间的输出重叠。
# d) 放置合法性:
#    触发点必须离端点至少 2 格, 才能在主路径两侧都看见 8; 同时触发点到四个射线方向的边界距离都至少 2,
#    才能稳定出现 `6 -> 8 -> 1`。生成器在候选触发点阶段直接强制这些约束。
# e) 可观察性:
#    解题者会在每个训练样例里反复看到同一种局部结构: 主路径靠近品红会变成 8, 射线紧邻品红也是 8,
#    两边共用同一个过渡色。这样只需 2-4 个样例就足以归纳规则。
# f) 可视化整洁度:
#    限制为每段恰好 1 个触发点, 并减少段数与干扰点数量, 避免多个 8 色团块互相粘连。
# g) 变化可推断性:
#    颜色映射只有一个关键关系: “与触发点对角相邻的一格都变成 8”。更远的补线统一回到 1。

# === CFG: colors ===
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_BLEND_COLOR = 8

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (5, 16)
_TRIGGERS_PER_SEG = (1, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 100
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _choose_line_color(non_bg: list[tuple[int, int, int]]) -> int:
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
                cand = (abs(dr), col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"
    return best[1]


def _segment_specs(points: list[tuple[int, int]]) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in points:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    segments: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            p, q = g[0], g[-1]
            path = _diag_path(p, q)
            dr = 1 if q[0] > p[0] else -1
            dc = 1 if q[1] > p[1] else -1
            segments.append((path, [(dr, -dc), (-dr, dc)]))

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            p, q = g[0], g[-1]
            path = _diag_path(p, q)
            dr = 1 if q[0] > p[0] else -1
            dc = 1 if q[1] > p[1] else -1
            segments.append((path, [(dr, -dc), (-dr, dc)]))

    return segments


def _apply_gradient_path(
    out: np.ndarray,
    source: np.ndarray,
    bg: int,
    path: list[tuple[int, int]],
    trigger_indices: list[int],
) -> None:
    trigger_index_set = set(trigger_indices)
    for idx, (r, c) in enumerate(path):
        if idx in trigger_index_set:
            continue
        if int(source[r, c]) != bg:
            continue
        if any(abs(idx - t_idx) == 1 for t_idx in trigger_indices):
            out[r, c] = _BLEND_COLOR
        else:
            out[r, c] = _LINE_COLOR


def _emit_gradient_rays(
    out: np.ndarray,
    bg: int,
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]],
) -> None:
    h, w = out.shape
    for (r0, c0, _tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            step = 1
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _BLEND_COLOR if step == 1 else _LINE_COLOR
                r += ddr
                c += ddc
                step += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the blue endpoint pairs across their diagonals, tint the cells next to magenta crossings with a shared bridge color, and let the perpendicular rays fade from magenta into blue.

    Input structure:
    - The background is the most common color and is never 1, 6, or 8. Blue endpoint pixels form one or more diagonal pairs. Magenta pixels that lie on those diagonals are the active crossings; off-diagonal magenta pixels are inert distractors.

    Rule:
    - For each diagonal defined by matching blue endpoints, fill the segment between the endpoints. Standard filled cells are blue, but any newly filled cell immediately adjacent to an on-segment magenta trigger is recolored to 8.
    - Each on-segment magenta trigger emits two perpendicular diagonal rays. Along each ray, the first new cell is 8 and every later new cell is blue, while the trigger itself stays magenta.

    Key insight:
    - Color now encodes distance from a trigger, not object identity alone: 8 means “one diagonal step away from an active magenta trigger,” whether that step lies on the main connecting path or on a perpendicular ray.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    line_color = _choose_line_color(non_bg)
    line_points = [(r, c) for r, c, col in non_bg if col == line_color]

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    for path, perp_dirs in _segment_specs(line_points):
        trigger_indices: list[int] = []
        for idx, (r, c) in enumerate(path):
            v = int(grid[r, c])
            if v != bg and v != line_color:
                trigger_indices.append(idx)
                triggers.setdefault((r, c, v), set()).update(perp_dirs)

        _apply_gradient_path(out, grid, bg, path, trigger_indices)

    _emit_gradient_rays(out, bg, triggers)
    return out


def _ray_arm_length(r: int, c: int, ddr: int, ddc: int, H: int, W: int) -> int:
    steps = 0
    r += ddr
    c += ddc
    while 0 <= r < H and 0 <= c < W:
        steps += 1
        r += ddr
        c += ddc
    return steps


def _compute_ray_cells(
    trigger_positions: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells = set()
    for tr, tc in trigger_positions:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    expected = inp.copy()
    for seg in segments:
        path = seg["path"]
        trigger_indices = seg["trigger_indices"]
        _apply_gradient_path(expected, inp, bg, path, trigger_indices)

        trigger_map = {
            (r, c, _TRIGGER_COLOR): set(seg["perp_dirs"])
            for (r, c) in seg["trigger_positions"]
        }
        _emit_gradient_rays(expected, bg, trigger_map)
    return expected


def _random_path(seg_len: int, H: int, W: int) -> tuple[list[tuple[int, int]], int] | None:
    slope_type = random.choice(["backslash", "slash"])
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
    return path, dc


def _pick_trigger_for_path(
    path: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    H: int,
    W: int,
) -> tuple[int, tuple[int, int]] | None:
    candidates: list[tuple[int, tuple[int, int]]] = []
    for idx in range(2, len(path) - 2):
        r, c = path[idx]
        arm_lengths = [_ray_arm_length(r, c, ddr, ddc, H, W) for ddr, ddc in perp_dirs]
        if min(arm_lengths) >= 2:
            candidates.append((idx, (r, c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _try_place_segment(H: int, W: int, occupied: set[tuple[int, int]], drawn: set[tuple[int, int]]) -> dict | None:
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    for _ in range(_SEGMENT_PLACE_RETRIES):
        seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
        random_path = _random_path(seg_len, H, W)
        if random_path is None:
            return None
        path, dc = random_path

        if any(rc in occupied or rc in drawn for rc in path):
            continue

        perp_dirs = [(1, -dc), (-1, dc)]
        picked = _pick_trigger_for_path(path, perp_dirs, H, W)
        if picked is None:
            continue

        trigger_idx, trigger_pos = picked
        trigger_positions = [trigger_pos]
        ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
        if any(rc in occupied or rc in drawn for rc in ray_cells):
            continue

        path_fill_cells = set(path[1:-1]) - set(trigger_positions)
        if any(rc in occupied or rc in drawn for rc in path_fill_cells):
            continue

        return {
            "ep0": path[0],
            "ep1": path[-1],
            "path": path,
            "trigger_indices": [trigger_idx],
            "trigger_positions": trigger_positions,
            "perp_dirs": perp_dirs,
            "ray_cells": ray_cells,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR, _BLEND_COLOR)])

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    segments: list[dict] = []

    for _ in range(random.randint(*_SEGMENT_COUNT)):
        seg = _try_place_segment(H, W, occupied, drawn)
        if seg is None:
            continue

        segments.append(seg)
        occupied.update([seg["ep0"], seg["ep1"]])
        occupied.update(seg["trigger_positions"])
        drawn.update(set(seg["path"][1:-1]) - set(seg["trigger_positions"]))
        drawn.update(seg["ray_cells"])

    if not segments:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        inp[seg["ep0"]] = _LINE_COLOR
        inp[seg["ep1"]] = _LINE_COLOR
        for tp in seg["trigger_positions"]:
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
