# Combo variant of db695cfb (index 11)
# combo_spec: D12/0+D14/0
# Name: X形双对角线+交叉特殊颜色

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# fused_spec:
#   concretization_and_trace:
#     - 保留原题的对象词汇：蓝色端点对定义对象，品红像素是触发器，背景色不是 1/2/6。
#     - 现在每对蓝点不只补全“端点直连”的那条 45 度对角线，还要补全同一正方形里的另一条对角线，
#       因而每个对象在输出里变成一个局部 X 形。
#     - 例：蓝点在 (1,1) 和 (5,5) 时，原题只画 "\"；融合后还要画 "/"=(1,5)->(5,1)。
#       若品红在 (2,4)，原题无事发生；融合后它位于新增 "/" 支线上，会沿 "\" 方向向两端发射。
#     - 再加入交叉着色：若某个偶数长度 X 的两条蓝线在中心格重合，该中心格输出为特殊色 2；
#       若任何品红射线击中任意蓝色结构格，该格也输出为 2，而不是保留蓝色。
#   complexity_calibration:
#     - 只允许一个固定特殊色 2，避免把“哪种交叉映射到哪种颜色”变成不可归纳的多分支规则。
#     - 生成器强制出现三类证据：至少一个 cross-only trigger、至少一个蓝蓝交叉、至少一个蓝线-品红射线交叉。
#       这样样例不会退化成“只是 D12”或“只是 D14”。
#     - 额外段默认与已有输出隔离，只把核心交叉放在一个受控局部，避免整图过度混杂。
#   implicit_assumption_review:
#     - 原 main() 默认“每个蓝端点对只定义一条线”；融合后失效，必须分别收集 direct/cross 两条路径。
#     - 原 main() 直接把蓝线写入 out，再让射线只填背景，隐含假设“蓝线-射线交叉保持蓝色”；融合后失效，
#       需要先得到蓝色结构集合/覆盖次数，再单独渲染特殊交叉色。
#     - 现在蓝蓝交叉也有语义，因此蓝色结构不能只记录集合，还要记录每格被多少条蓝线覆盖。
#     - solver 仍沿用“按同色最长对角配对选 line_color、每条对角键连接极端点”的假设；
#       generator 因此保证同一坡向的真实蓝端点不共用同一 diagonal key，未配对蓝点也不与其它蓝点成对角。
#   placement_legality:
#     - 每个对象的合法占位是一个轴对齐正方形的两条对角线；direct/cross 路径、触发器、射线都必须在网格内。
#     - 强制样例先放一个“主 X”：它长度取偶数，天然提供一个蓝蓝交叉中心；再放一个被主 X 的 direct trigger
#       射线击中的目标段，确保蓝线-射线交叉存在且不落在端点/trigger 上。
#     - cross-only trigger 选在主 X 的新增支线上，并要求它的射线避开受控碰撞区，从而既能证明 D12 行为，
#       又不会把局部图案搞得不可读。
#   observability_and_cleanliness:
#     - 人类应能从少量样例看到：同一对蓝点输出成 X；有些品红明明不在蓝点直连线上却也会发射；
#       特殊色 2 总出现在“线与线”或“线与射线”重合处，而不是随机换色。
#     - 为了让这一点可见，生成器限制额外 trigger 数量，并拒绝大面积重叠、ray-ray 纠缠和 trigger 落在交叉中心。
#   new_solver_assumptions:
#     - solver 允许一格被多条蓝线结构性覆盖，但假设真正的输入 trigger 不会正好占据需要被重涂为特殊色的交叉格；
#       generator 显式规避这种输入。
#     - solver 依赖“触发格沿其所在支线的垂直对角方向发射”，因此 generator 只在 direct/cross 的专属内部格放 trigger，
#       避免同一格同时属于两支线导致四向发射歧义。


_LINE_COLOR = 1
_CROSS_COLOR = 2
_TRIGGER_COLOR = 6

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (2, 4)
_SEG_LEN_RANGE = (3, 16)
_EXTRA_DIRECT_TRIGGERS = (0, 1)
_EXTRA_CROSS_TRIGGERS = (0, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_PAIR_RETRIES = 220
_SEGMENT_PLACE_RETRIES = 120
_EXTRA_SEGMENT_RETRIES = 160
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 400


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _cross_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
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


def _exclusive_interior(path: list[tuple[int, int]], other_path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    other = set(other_path)
    return [cell for cell in path[1:-1] if cell not in other]


def _direct_slope_type(p: tuple[int, int], q: tuple[int, int]) -> str:
    return "backslash" if (q[0] - p[0]) * (q[1] - p[1]) > 0 else "slash"


def _direct_key(p: tuple[int, int], q: tuple[int, int]) -> int:
    return p[0] - p[1] if _direct_slope_type(p, q) == "backslash" else p[0] + p[1]


def _make_segment(
    p: tuple[int, int],
    q: tuple[int, int],
    direct_triggers: list[tuple[int, int]],
    cross_triggers: list[tuple[int, int]],
) -> dict:
    direct_path = _diag_path(p, q)
    cross_path = _cross_path(p, q)
    direct_dirs, cross_dirs = _segment_dirs(p, q)

    trigger_dir_map: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for pos in direct_triggers:
        trigger_dir_map.setdefault(pos, set()).update(cross_dirs)
    for pos in cross_triggers:
        trigger_dir_map.setdefault(pos, set()).update(direct_dirs)

    return {
        "endpoints": (p, q),
        "direct_path": direct_path,
        "cross_path": cross_path,
        "path_cells": set(direct_path) | set(cross_path),
        "trigger_dir_map": trigger_dir_map,
        "direct_key": _direct_key(p, q),
        "slope_type": _direct_slope_type(p, q),
    }


def _build_structure_from_segments(
    segments: list[dict],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int, int], set[tuple[int, int]]]]:
    blue_counts: dict[tuple[int, int], int] = {}
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    for seg in segments:
        for path in (seg["direct_path"], seg["cross_path"]):
            for cell in path:
                blue_counts[cell] = blue_counts.get(cell, 0) + 1
        for pos, dirs in seg["trigger_dir_map"].items():
            triggers.setdefault((pos[0], pos[1], _TRIGGER_COLOR), set()).update(dirs)

    return blue_counts, triggers


def _render_variant(
    grid: np.ndarray,
    bg: int,
    line_color: int,
    blue_counts: dict[tuple[int, int], int],
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]],
) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    trigger_positions = {(r, c) for r, c, _ in triggers}
    blue_cells = set(blue_counts)

    for (r, c), count in blue_counts.items():
        if (r, c) in trigger_positions:
            continue
        if count >= 2:
            out[r, c] = _CROSS_COLOR
        elif int(out[r, c]) == bg:
            out[r, c] = line_color

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in blue_cells and (r, c) not in trigger_positions:
                    out[r, c] = _CROSS_COLOR
                elif int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


def _collect_structure_from_grid(
    grid: np.ndarray, bg: int, line_color: int
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int, int], set[tuple[int, int]]]]:
    h, w = grid.shape
    pts = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == line_color]

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    blue_counts: dict[tuple[int, int], int] = {}
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def collect_path(path: list[tuple[int, int]], ray_dirs: list[tuple[int, int]]) -> None:
        for r, c in path:
            blue_counts[(r, c)] = blue_counts.get((r, c), 0) + 1
            v = int(grid[r, c])
            if v != bg and v != line_color:
                triggers.setdefault((r, c, v), set()).update(ray_dirs)

    def collect_x_from_pair(p: tuple[int, int], q: tuple[int, int]) -> None:
        direct_path = _diag_path(p, q)
        cross_path = _cross_path(p, q)
        direct_dirs, cross_dirs = _segment_dirs(p, q)
        collect_path(direct_path, cross_dirs)
        collect_path(cross_path, direct_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            collect_x_from_pair(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            collect_x_from_pair(g[0], g[-1])

    return blue_counts, triggers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each diagonal blue pair into a local X and mark every structural overlap with a special color.

    Input structure:
    - The background is the most frequent color. One non-background color forms paired blue endpoints on 45-degree diagonals, and magenta trigger pixels may lie on either the visible endpoint-to-endpoint branch or the hidden crossing branch of the same square.

    Rule:
    - For every diagonal blue endpoint pair, complete both diagonals of the spanned square, creating an X-shaped blue scaffold.
    - Any non-blue pixel touched by either branch emits rays along the perpendicular diagonal orientation to the grid edge.
    - Cells covered by two blue branches, and cells where a magenta ray hits any blue-structure cell, are recolored with a dedicated special color.

    Key insight:
    - The blue pair no longer defines a single segment; it defines a two-branch structure. The output must therefore reason about branch membership first, then apply a shared “intersection becomes special” rule to both blue-blue and blue-ray collisions.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    line_color = _pick_line_color(by_color)
    blue_counts, triggers = _collect_structure_from_grid(grid, bg, line_color)
    return _render_variant(grid, bg, line_color, blue_counts, triggers)


def _compute_ray_cells(
    trigger_dir_map: dict[tuple[int, int], set[tuple[int, int]]], H: int, W: int
) -> set[tuple[int, int]]:
    cells = set()
    for (tr, tc), dirs in trigger_dir_map.items():
        for ddr, ddc in dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_input(H: int, W: int, bg: int, segments: list[dict]) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        p, q = seg["endpoints"]
        inp[p] = _LINE_COLOR
        inp[q] = _LINE_COLOR
        for tp in seg["trigger_dir_map"]:
            inp[tp] = _TRIGGER_COLOR
    return inp


def _render_from_segments(inp: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    blue_counts, triggers = _build_structure_from_segments(segments)
    return _render_variant(inp, bg, _LINE_COLOR, blue_counts, triggers)


def _current_blocked_cells(H: int, W: int, bg: int, segments: list[dict]) -> set[tuple[int, int]]:
    inp = _build_input(H, W, bg, segments)
    out = _render_from_segments(inp, bg, segments)
    return set(map(tuple, np.argwhere(out != bg)))


def _line_color_selected(inp: np.ndarray, bg: int) -> int:
    non_bg = [
        (r, c, int(inp[r, c]))
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if int(inp[r, c]) != bg
    ]
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))
    return _pick_line_color(by_color)


def _path_in_direction(
    start: tuple[int, int], step: tuple[int, int], H: int, W: int
) -> list[tuple[int, int]]:
    r, c = start[0] + step[0], start[1] + step[1]
    cells = []
    while 0 <= r < H and 0 <= c < W:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _place_segment_anywhere(
    H: int,
    W: int,
    max_seg_len: int,
    forbidden: set[tuple[int, int]],
    used_backslash: set[int],
    used_slash: set[int],
    require_even: bool = False,
) -> dict | None:
    lengths = list(range(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]) + 1))
    if require_even:
        lengths = [length for length in lengths if length % 2 == 0]
    if not lengths:
        return None

    for _ in range(_SEGMENT_PLACE_RETRIES):
        seg_len = random.choice(lengths)
        slope_type = random.choice(["backslash", "slash"])

        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                continue
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            p = (r0, c0)
            q = (r0 + seg_len, c0 + seg_len)
            key = p[0] - p[1]
            if key in used_backslash:
                continue
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                continue
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            p = (r0, c0)
            q = (r0 + seg_len, c0 - seg_len)
            key = p[0] + p[1]
            if key in used_slash:
                continue

        seg = _make_segment(p, q, [], [])
        if seg["path_cells"] & forbidden:
            continue
        return seg

    return None


def _place_segment_through_cell(
    x: tuple[int, int],
    slope_type: str,
    H: int,
    W: int,
    max_seg_len: int,
    forbidden: set[tuple[int, int]],
    used_backslash: set[int],
    used_slash: set[int],
) -> dict | None:
    step = (1, 1) if slope_type == "backslash" else (1, -1)

    neg_max = 0
    r, c = x
    while 0 <= r - step[0] < H and 0 <= c - step[1] < W:
        r -= step[0]
        c -= step[1]
        neg_max += 1

    pos_max = 0
    r, c = x
    while 0 <= r + step[0] < H and 0 <= c + step[1] < W:
        r += step[0]
        c += step[1]
        pos_max += 1

    choices = [(neg, pos) for neg in range(1, neg_max + 1) for pos in range(1, pos_max + 1)]
    random.shuffle(choices)

    max_len = min(max_seg_len, _SEG_LEN_RANGE[1])
    for neg, pos in choices:
        seg_len = neg + pos
        if not (_SEG_LEN_RANGE[0] <= seg_len <= max_len):
            continue
        if seg_len % 2 == 0 and neg == pos:
            continue

        p = (x[0] - neg * step[0], x[1] - neg * step[1])
        q = (x[0] + pos * step[0], x[1] + pos * step[1])
        key = p[0] - p[1] if slope_type == "backslash" else p[0] + p[1]
        if slope_type == "backslash" and key in used_backslash:
            continue
        if slope_type == "slash" and key in used_slash:
            continue

        seg = _make_segment(p, q, [], [])
        overlap = seg["path_cells"] & forbidden
        if overlap != {x}:
            continue
        return seg

    return None


def _try_place_mandatory_segments(H: int, W: int, max_seg_len: int) -> list[dict] | None:
    if max_seg_len < max(_SEG_LEN_RANGE[0], 4):
        return None

    for _ in range(_PAIR_RETRIES):
        source = _place_segment_anywhere(H, W, max_seg_len, set(), set(), set(), require_even=True)
        if source is None:
            continue

        p, q = source["endpoints"]
        direct_dirs, cross_dirs = _segment_dirs(p, q)
        direct_candidates = _exclusive_interior(source["direct_path"], source["cross_path"])
        cross_candidates = _exclusive_interior(source["cross_path"], source["direct_path"])
        if not direct_candidates or not cross_candidates:
            continue

        random.shuffle(direct_candidates)
        random.shuffle(cross_candidates)
        source_path_cells = source["path_cells"]
        source_slope = source["slope_type"]
        target_slope = source_slope

        for direct_trigger in direct_candidates:
            dir_choices = cross_dirs[:]
            random.shuffle(dir_choices)
            for ray_dir in dir_choices:
                ray_branch = _path_in_direction(direct_trigger, ray_dir, H, W)
                if len(ray_branch) < 2:
                    continue
                collision_cells = ray_branch[1:]
                random.shuffle(collision_cells)
                for x in collision_cells:
                    used_backslash = {source["direct_key"]} if source_slope == "backslash" else set()
                    used_slash = {source["direct_key"]} if source_slope == "slash" else set()
                    source_direct_only = _make_segment(p, q, [direct_trigger], [])
                    source_direct_output = _current_blocked_cells(H, W, 0, [source_direct_only])
                    target = _place_segment_through_cell(
                        x,
                        target_slope,
                        H,
                        W,
                        max_seg_len,
                        source_direct_output | source_path_cells,
                        used_backslash,
                        used_slash,
                    )
                    if target is None:
                        continue

                    random.shuffle(cross_candidates)
                    for cross_trigger in cross_candidates:
                        source_full = _make_segment(p, q, [direct_trigger], [cross_trigger])
                        cross_only_rays = _compute_ray_cells({cross_trigger: set(direct_dirs)}, H, W)
                        if cross_only_rays & target["path_cells"]:
                            continue
                        if cross_only_rays & (set(ray_branch) - {direct_trigger}):
                            continue

                        segments = [source_full, target]
                        inp = _build_input(H, W, 0, segments)
                        out = _render_from_segments(inp, 0, segments)
                        special_cells = set(map(tuple, np.argwhere(out == _CROSS_COLOR)))
                        if x not in special_cells:
                            continue
                        if not any(cell in source_path_cells and cell not in source_full["trigger_dir_map"] for cell in special_cells):
                            continue
                        return segments

    return None


def _extract_used_keys(segments: list[dict]) -> tuple[set[int], set[int]]:
    used_backslash = set()
    used_slash = set()
    for seg in segments:
        if seg["slope_type"] == "backslash":
            used_backslash.add(seg["direct_key"])
        else:
            used_slash.add(seg["direct_key"])
    return used_backslash, used_slash


def _try_add_extra_segment(
    segments: list[dict], H: int, W: int, bg: int, max_seg_len: int
) -> list[dict] | None:
    blocked = _current_blocked_cells(H, W, bg, segments)
    used_backslash, used_slash = _extract_used_keys(segments)
    base = _place_segment_anywhere(H, W, max_seg_len, blocked, used_backslash, used_slash)
    if base is None:
        return None

    direct_candidates = _exclusive_interior(base["direct_path"], base["cross_path"])
    cross_candidates = _exclusive_interior(base["cross_path"], base["direct_path"])

    direct_max = min(_EXTRA_DIRECT_TRIGGERS[1], len(direct_candidates))
    cross_max = min(_EXTRA_CROSS_TRIGGERS[1], len(cross_candidates))
    direct_count = random.randint(_EXTRA_DIRECT_TRIGGERS[0], direct_max) if direct_max >= _EXTRA_DIRECT_TRIGGERS[0] else 0
    cross_count = random.randint(_EXTRA_CROSS_TRIGGERS[0], cross_max) if cross_max >= _EXTRA_CROSS_TRIGGERS[0] else 0

    direct_triggers = random.sample(direct_candidates, direct_count) if direct_count else []
    cross_triggers = random.sample(cross_candidates, cross_count) if cross_count else []
    seg = _make_segment(base["endpoints"][0], base["endpoints"][1], direct_triggers, cross_triggers)

    ray_cells = _compute_ray_cells(seg["trigger_dir_map"], H, W)
    if ray_cells & blocked:
        return None
    if ray_cells & (seg["path_cells"] - set(seg["trigger_dir_map"])):
        return None

    return segments + [seg]


def _count_cross_only_triggers(segments: list[dict]) -> int:
    total = 0
    for seg in segments:
        cross_only = set(_exclusive_interior(seg["cross_path"], seg["direct_path"]))
        total += sum(1 for pos in seg["trigger_dir_map"] if pos in cross_only)
    return total


def _count_blue_blue_specials(
    segments: list[dict], out: np.ndarray
) -> int:
    blue_counts, triggers = _build_structure_from_segments(segments)
    trigger_positions = {(r, c) for r, c, _ in triggers}
    total = 0
    for cell, count in blue_counts.items():
        if count >= 2 and cell not in trigger_positions and int(out[cell]) == _CROSS_COLOR:
            total += 1
    return total


def _count_blue_ray_specials(
    segments: list[dict], H: int, W: int, out: np.ndarray
) -> int:
    blue_counts, triggers = _build_structure_from_segments(segments)
    blue_cells = set(blue_counts)
    trigger_positions = {(r, c) for r, c, _ in triggers}
    collided = set()
    for (r0, c0, _), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < H and 0 <= c < W:
                if (r, c) in blue_cells and (r, c) not in trigger_positions and int(out[r, c]) == _CROSS_COLOR:
                    collided.add((r, c))
                r += ddr
                c += ddc
    return len(collided)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR, _CROSS_COLOR)])

    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    segments = _try_place_mandatory_segments(H, W, max_seg_len)
    if segments is None:
        return None

    target_segments = random.randint(*_SEGMENT_COUNT)
    for _ in range(_EXTRA_SEGMENT_RETRIES):
        if len(segments) >= target_segments:
            break
        updated = _try_add_extra_segment(segments, H, W, bg, max_seg_len)
        if updated is not None:
            segments = updated

    inp = _build_input(H, W, bg, segments)
    expected = _render_from_segments(inp, bg, segments)

    if _count_cross_only_triggers(segments) < 1:
        return None
    if _count_blue_blue_specials(segments, expected) < 1:
        return None
    if _count_blue_ray_specials(segments, H, W, expected) < 1:
        return None

    occupied = set(map(tuple, np.argwhere(inp != bg)))
    blocked = set(map(tuple, np.argwhere(expected != bg)))

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

    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in blocked:
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

    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
