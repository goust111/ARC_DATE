# Variant of db695cfb
# Dimension: D17 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——所有行为由对象颜色和位置直接决定 → 蓝色像素间距编码延伸长度
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - 原题里，落在蓝色对角线上的品红触发点会沿垂直对角线一直射到边界。
#     - 本变体里，蓝色端点间的对角距离 d 额外编码了射线长度上限：
#       每个方向最多延伸 floor(d / 2) 步，若先碰到边界则提前停止。
#   concrete_trace:
#     - 例：两个蓝端点相距 8 个对角步，原题会把触发点的品红线一直画到边界。
#     - 现在同一触发点只在每个垂直对角方向上画 4 格；若该方向边界还更远，
#       剩余区域保持背景色，因此肉眼能看出“没有画满边界”。
#   where_the_pipeline_changes:
#     - 输入构造仍然是“蓝端点 + 落在对角线上的品红点”。
#     - solver 在记录触发点时，除了记住垂直方向，还必须记住该段蓝线携带的长度参数。
#     - generator 必须保证至少有一条射线本来可以继续到边界，但被 floor(d / 2) 截断，
#       否则样例会退化成原题。
#   complexity_calibration:
#     - 只引入一个数值关系：ray_len = floor(blue_distance / 2)。
#     - 为了让规律可观察，每个样例放 2-3 条线段，且要求至少出现 2 个不同的 ray_len。
#     - 每条线段保留 1-2 个触发点，避免自由度过多导致视觉混乱。
#   assumption_audit_for_main:
#     - “最长的同色对角点对对应蓝线颜色”这一前提仍保留，因此仍可用原来的 line_color 识别逻辑。
#     - 原题里触发点只需要存方向；变体后若仍只存方向，发射阶段就会错误地画到边界。
#     - 原题的 while 循环默认无界延伸；变体后必须把每个方向绑定到段长导出的上限。
#     - 若一个触发点理论上被多条线段命中，solver 需要按方向分别存上限；本生成器避免这种歧义，
#       但 solver 仍以 max-cap 的方式安全处理。
#   placement_legality:
#     - 生成器禁止蓝线段共享同一条完整对角线，否则 solver 的“连接极端点”策略会把它们并段。
#     - 生成器把蓝线内部、有限品红射线、以及“本应在原题里继续延伸但本变体故意留白”的区域
#       一并加入禁放集合，避免后续对象破坏 D17 的可见性。
#   visual_observability:
#     - 至少一条触发射线被明确截断。
#     - 同一输入里尽量出现两个不同的蓝距/射线长度档位，使解题者能把“蓝距更长 -> 品红延伸更长”
#       直接从像素长度对比中看出来。
#   cleanliness:
#     - 所有线段、射线和留白证据区互不重叠，减少叠色和遮挡。
#   color_reasoning:
#     - 颜色角色不变：蓝色仍是连接线，品红仍是触发/延伸色；变化只发生在长度参数上，
#       不引入新的颜色映射负担。

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue diagonal endpoint pairs, then emit capped perpendicular magenta rays from trigger pixels on those diagonals.

    Input structure:
    - bg = most common color. One non-background color forms diagonal endpoint pairs, and differently-colored trigger pixels can lie strictly between those endpoints.

    Rule:
    - Fill each qualifying diagonal segment between the same-colored endpoints.
    - If the segment passes through a trigger pixel, shoot perpendicular diagonal rays in the trigger's color, but cap each ray at floor(endpoint distance / 2) cells instead of drawing all the way to the border.

    Key insight:
    - The blue endpoint spacing is a visible numeric code: longer blue gaps produce longer magenta extensions, while shorter blue gaps truncate them sooner along the perpendicular diagonal.
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

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

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

    # Each trigger remembers a per-direction cap derived from the hosting blue segment length.
    triggers: dict[tuple[int, int, int], dict[tuple[int, int], int]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant (slope +1)
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant (slope -1)
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        ray_limit = abs(q[0] - p[0]) // 2
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                key = (r, c, v)
                caps = triggers.setdefault(key, {})
                for ray_dir in perp_dirs:
                    caps[ray_dir] = max(caps.get(ray_dir, 0), ray_limit)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0, tcol), caps in triggers.items():
        for (ddr, ddc), limit in caps.items():
            r, c = r0 + ddr, c0 + ddc
            steps = 0
            while 0 <= r < h and 0 <= c < w and steps < limit:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc
                steps += 1

    return out


# --------------- generator (D17 variant) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (2, 3)
_SEG_LEN_RANGE = (4, 14)
_TRIGGERS_PER_SEG = (1, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 500


def _compute_ray_cells(
    triggers: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    ray_limit: int,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    """Cells painted by the capped perpendicular rays."""
    cells = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            steps = 0
            while 0 <= r < H and 0 <= c < W and steps < ray_limit:
                cells.add((r, c))
                r += ddr
                c += ddc
                steps += 1
    return cells


def _max_steps_to_boundary(r0: int, c0: int, ddr: int, ddc: int, H: int, W: int) -> int:
    steps = 0
    r, c = r0 + ddr, c0 + ddc
    while 0 <= r < H and 0 <= c < W:
        steps += 1
        r += ddr
        c += ddc
    return steps


def _cells_beyond_limit(r0: int, c0: int, ddr: int, ddc: int, limit: int, H: int, W: int) -> list[tuple[int, int]]:
    cells = []
    r, c = r0 + ddr, c0 + ddc
    steps = 0
    while 0 <= r < H and 0 <= c < W:
        if steps >= limit:
            cells.append((r, c))
        r += ddr
        c += ddc
        steps += 1
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in seg["trigger_positions"]:
            for ddr, ddc in seg["perp_dirs"]:
                r, c = tr + ddr, tc + ddc
                steps = 0
                while 0 <= r < H and 0 <= c < W and steps < seg["ray_limit"]:
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
                    steps += 1
    return expected


def _try_place_segment(
    slope_type: str,
    seg_len: int,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    used_backslash: set[int],
    used_slash: set[int],
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            path = [(r0 + k, c0 + k) for k in range(seg_len + 1)]
            diag_key = path[0][0] - path[0][1]
            if diag_key in used_backslash:
                continue
            dc = 1
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            path = [(r0 + k, c0 - k) for k in range(seg_len + 1)]
            diag_key = path[0][0] + path[0][1]
            if diag_key in used_slash:
                continue
            dc = -1

        if any(rc in occupied or rc in blocked for rc in path):
            continue
        return path, dc
    return None


def _choose_triggers(
    path_interior: list[tuple[int, int]],
    perp_dirs: list[tuple[int, int]],
    ray_limit: int,
    H: int,
    W: int,
    require_truncated: bool,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    qualifying = []
    evidence_reserve: set[tuple[int, int]] = set()
    for tr, tc in path_interior:
        reserve = set()
        for ddr, ddc in perp_dirs:
            if _max_steps_to_boundary(tr, tc, ddr, ddc, H, W) > ray_limit:
                reserve.update(_cells_beyond_limit(tr, tc, ddr, ddc, ray_limit, H, W))
        if reserve:
            qualifying.append((tr, tc, reserve))

    if require_truncated and not qualifying:
        return None

    max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
    min_triggers = min(_TRIGGERS_PER_SEG[0], max_triggers)
    if max_triggers <= 0:
        return None
    n_triggers = random.randint(min_triggers, max_triggers)

    chosen: list[tuple[int, int]] = []
    available = list(path_interior)
    if require_truncated:
        tr, tc, reserve = random.choice(qualifying)
        chosen.append((tr, tc))
        evidence_reserve.update(reserve)
        available = [p for p in available if p != (tr, tc)]

    need_more = max(0, n_triggers - len(chosen))
    if need_more > len(available):
        return None
    if need_more:
        chosen.extend(random.sample(available, need_more))
    return chosen, evidence_reserve


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    n_segments = random.randint(*_SEGMENT_COUNT)
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    used_backslash: set[int] = set()
    used_slash: set[int] = set()
    segments: list[dict] = []

    for seg_idx in range(n_segments):
        placed = False
        must_show_variant = seg_idx == n_segments - 1 and not any(seg["evidence_reserve"] for seg in segments)
        preferred_limits = {seg["ray_limit"] for seg in segments}
        for _ in range(_SEGMENT_PLACE_RETRIES):
            seg_len = random.randint(_SEG_LEN_RANGE[0], max_seg_len)
            ray_limit = seg_len // 2
            if preferred_limits and len(preferred_limits) == len(segments) and ray_limit in preferred_limits and random.random() < 0.7:
                continue
            slope_type = random.choice(["backslash", "slash"])
            result = _try_place_segment(
                slope_type,
                seg_len,
                H,
                W,
                occupied,
                blocked,
                used_backslash,
                used_slash,
            )
            if result is None:
                continue
            path, dc = result
            perp_dirs = [(1, -dc), (-1, dc)]
            path_interior = path[1:-1]
            trigger_result = _choose_triggers(path_interior, perp_dirs, ray_limit, H, W, must_show_variant)
            if trigger_result is None:
                continue
            trigger_positions, evidence_reserve = trigger_result
            ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, ray_limit, H, W)
            if any(rc in occupied or rc in blocked for rc in ray_cells):
                continue
            if any(rc in occupied or rc in blocked or rc in ray_cells or rc in path for rc in evidence_reserve):
                continue

            ep0, ep1 = path[0], path[-1]
            segments.append(
                {
                    "ep0": ep0,
                    "ep1": ep1,
                    "trigger_positions": trigger_positions,
                    "path": path,
                    "perp_dirs": perp_dirs,
                    "ray_limit": ray_limit,
                    "evidence_reserve": evidence_reserve,
                }
            )
            occupied.update([ep0, ep1])
            occupied.update(trigger_positions)
            blocked.update(set(path[1:-1]) - set(trigger_positions))
            blocked.update(ray_cells)
            blocked.update(evidence_reserve)
            if slope_type == "backslash":
                used_backslash.add(path[0][0] - path[0][1])
            else:
                used_slash.add(path[0][0] + path[0][1])
            placed = True
            break
        if not placed:
            return None

    if not segments:
        return None

    if not any(seg["evidence_reserve"] for seg in segments):
        return None

    if len({seg["ray_limit"] for seg in segments}) < 2:
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
            if (r, c) in occupied or (r, c) in blocked:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    all_trigger_positions = list(zip(*np.where(inp == _TRIGGER_COLOR)))
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in blocked:
                continue
            if any(abs(r - tr) == abs(c - tc) and r != tr for tr, tc in all_trigger_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            all_trigger_positions.append((r, c))
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
