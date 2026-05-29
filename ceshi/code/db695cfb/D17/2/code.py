# Variant of db695cfb
# Dimension: D17 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——所有行为由对象颜色和位置直接决定... → 品红像素的邻域编码延伸属性
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题骨架：蓝色像素仍然成对落在 45 度对角线上，输出仍然要把端点之间的整段蓝色对角线补齐。
#    - 只有落在这条蓝色对角线上的品红像素才是“触发器”。新变化是：它们不再自动向垂直对角线的两个方向都发射，而是读取自己上下相邻格中的微指令。
#    - 微指令约定：
#      * 北侧(上方一格)出现红色 2 -> 发射“向上”的那条垂直对角射线。
#      * 南侧(下方一格)出现绿色 3 -> 发射“向下”的那条垂直对角射线。
#      * 两个标记都在 -> 与原题一样双向发射；只出现一个 -> 只发单向射线。
#    - 具体小例子：蓝色端点在 (1,1) 与 (5,5)，品红触发器在 (3,3)。
#      原题：无论邻域如何，都沿 NE 与 SW 两个方向发射。
#      变体：若 (2,3)=红色而 (4,3) 不是绿色，则只沿“向上”的垂直对角方向发射，即只出现 NE 射线。
#      差异集中在“触发器收集 -> 射线方向决定”这一环节。
# b) 复杂度校准
#    - 只引入两种新指令颜色（红/绿），避免为了兼容四个绝对方向再额外引入更多颜色，降低样例归纳难度。
#    - 对角线仍然允许两种斜率；“向上/向下”总是相对于该段蓝色对角线的垂直方向来解释，因此规则稳定、分支有限。
#    - 生成器保证每题至少有一个单向触发器，否则会退化回原题的双向规则。
# c) 隐式假设审查（针对原 main 中与变体相关的关键写法）
#    - 原 `elif v != line_color:` 把落在蓝线上的任何非蓝色像素都当触发器；现在这不成立，因为红/绿只是说明符，不应发射。故改成只接受品红 6。
#    - 原逻辑不需要读邻域；现在必须读取触发器上下邻居，并且这些邻居要从原输入 `grid` 读取，不能从会被逐步填蓝的 `out` 读取。
#    - 原逻辑默认每个触发器一定有两条垂直方向；现在方向集合可能是 0/1/2 条。生成器会保证真实触发器至少有 1 条，solver 也容忍 0 条（即不发射）。
#    - 原逻辑的“只填背景不覆盖非背景”仍然保留，因此生成器必须继续保证输入像素不落在任何应被填充的蓝线/射线路径上。
# d) 放置合法性
#    - 触发器只放在线段内部；因为线段按行递增放置，内部点一定不在第一行或最后一行，所以红/绿微指令的北/南邻格天然在界内。
#    - 红/绿指令像素必须避开：已有输入像素、已有输出填充区域、同段其他触发器的射线路径。
#    - 射线路径也必须避开所有输入像素；这样输出只会往背景中补色，不会出现覆盖。
# e) 视觉可观察性
#    - 人类可以直接看到：有些品红点上方有红点、下方有绿点，而输出中的品红射线也会相应少一侧或双侧。
#    - 生成器要求每题至少一个单向触发器，从而保证新机制在少量样例中可见，不会长期退化成原题。
# f) 可视化整洁度
#    - 指令像素只占触发器的正上/正下两个格，局部结构紧凑，不会在同一区域叠太多符号。
#    - 生成器拒绝任何会让指令像素落到蓝线填充区、已有射线或其他输入像素上的摆放。
# g) 颜色变化可推断性
#    - 新颜色只有两种，并且语义由位置固定绑定：红只会出现在触发器正上方，绿只会出现在正下方。
#    - 输出颜色仍只使用蓝色补线和品红射线，不引入新的输出颜色映射。
# 9) 新隐式假设审查
#    - solver 假设“最长同色对角对”仍然是蓝色端点决定的主线颜色；生成器通过 oracle 校验维持这一点。
#    - solver 假设触发器的红/绿指令只出现在其上下邻位；生成器只按这一方式造题，不制造别的等价编码。
#    - solver 按行排序端点后提取一条 `ddr < 0` 的“向上”垂直射线和一条 `ddr > 0` 的“向下”垂直射线；生成器也始终以同一方向约定放置指令。

import random

import numpy as np

from arc_utils import get_bg


_LINE_COLOR = 1
_UP_MARKER_COLOR = 2
_DOWN_MARKER_COLOR = 3
_TRIGGER_COLOR = 6


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _decode_trigger_dirs(
    grid: np.ndarray,
    r: int,
    c: int,
    perp_dirs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, _ = grid.shape
    up_dir = next((d for d in perp_dirs if d[0] < 0), None)
    down_dir = next((d for d in perp_dirs if d[0] > 0), None)
    dirs: list[tuple[int, int]] = []
    if up_dir is not None and r > 0 and int(grid[r - 1, c]) == _UP_MARKER_COLOR:
        dirs.append(up_dir)
    if down_dir is not None and r + 1 < h and int(grid[r + 1, c]) == _DOWN_MARKER_COLOR:
        dirs.append(down_dir)
    return dirs


def _dirs_from_flags(
    perp_dirs: list[tuple[int, int]],
    emit_up: bool,
    emit_down: bool,
) -> list[tuple[int, int]]:
    up_dir = next(d for d in perp_dirs if d[0] < 0)
    down_dir = next(d for d in perp_dirs if d[0] > 0)
    dirs: list[tuple[int, int]] = []
    if emit_up:
        dirs.append(up_dir)
    if emit_down:
        dirs.append(down_dir)
    return dirs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the longest-color diagonal segments, then let on-segment magenta pixels read local instruction markers to choose which perpendicular rays to emit.

    Input structure:
    - The background is the most frequent color. Blue endpoint pixels identify one or more 45-degree diagonals to complete. Magenta pixels may lie on those diagonals, and a red pixel directly above or a green pixel directly below such a magenta pixel acts as a tiny direction code.

    Rule:
    - Connect each qualifying blue endpoint pair by filling its diagonal.
    - Whenever the filled diagonal passes through a magenta trigger, inspect that trigger's north/south neighbors in the original input: red above means emit only the upward perpendicular diagonal ray, green below means emit only the downward perpendicular diagonal ray, and having both markers means emit both rays.

    Key insight:
    - The magenta pixels are no longer self-sufficient beam splitters; their local red/green neighborhood encodes which subset of the two perpendicular branches should appear.
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

    triggers: dict[tuple[int, int], set[tuple[int, int]]] = {}

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
            if int(out[r, c]) == bg:
                out[r, c] = line_color
            if int(grid[r, c]) == _TRIGGER_COLOR:
                dirs = _decode_trigger_dirs(grid, r, c, perp_dirs)
                if dirs:
                    triggers.setdefault((r, c), set()).update(dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for (r0, c0), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = _TRIGGER_COLOR
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_SEGMENT_COUNT = (1, 4)
_SEG_LEN_RANGE = (3, 16)
_TRIGGERS_PER_SEG = (0, 2)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_DIAG_TRIGGERS = (0, 3)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 50
_GEN_RETRIES = 300


def _diag_aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _conflicts_same_color_diagonal(pos: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    return any(_diag_aligned(pos, other) for other in others)


def _marker_specs(tr: int, tc: int, emit_up: bool, emit_down: bool) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    if emit_up:
        specs.append((tr - 1, tc, _UP_MARKER_COLOR))
    if emit_down:
        specs.append((tr + 1, tc, _DOWN_MARKER_COLOR))
    return specs


def _compute_ray_cells_for_dirs(
    start: tuple[int, int],
    dirs: list[tuple[int, int]],
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    cells = set()
    tr, tc = start
    for ddr, ddc in dirs:
        r, c = tr + ddr, tc + ddc
        while 0 <= r < H and 0 <= c < W:
            cells.add((r, c))
            r += ddr
            c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_specs, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc, emit_up, emit_down in trigger_specs:
            dirs = _dirs_from_flags(perp_dirs, emit_up, emit_down)
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


def _pick_trigger_specs(path, dc, H, W, occupied, drawn, marker_positions):
    path_interior = path[1:-1]
    max_triggers = min(_TRIGGERS_PER_SEG[1], len(path_interior))
    empty_markers = {_UP_MARKER_COLOR: set(), _DOWN_MARKER_COLOR: set()}
    if max_triggers == 0:
        return [], empty_markers

    target = random.randint(_TRIGGERS_PER_SEG[0], max_triggers)
    if target == 0:
        return [], empty_markers

    perp_dirs = [(1, -dc), (-1, dc)]
    trigger_specs = []
    local_input = {path[0], path[-1]}
    local_drawn = set()
    local_marker_positions = empty_markers
    candidates = path_interior[:]
    random.shuffle(candidates)

    for tr, tc in candidates:
        if len(trigger_specs) >= target:
            break
        if (tr, tc) in local_drawn:
            continue

        patterns = [(True, False), (False, True), (True, True)]
        random.shuffle(patterns)
        for emit_up, emit_down in patterns:
            marker_specs = _marker_specs(tr, tc, emit_up, emit_down)
            marker_cells = {(mr, mc) for mr, mc, _ in marker_specs}
            if any(not (0 <= mr < H and 0 <= mc < W) for mr, mc, _ in marker_specs):
                continue
            if any(cell in occupied or cell in drawn or cell in local_input or cell in local_drawn for cell in marker_cells):
                continue

            diag_bad = False
            for mr, mc, col in marker_specs:
                existing = marker_positions[col] | local_marker_positions[col]
                local_same = {
                    (x[0], x[1])
                    for x in marker_specs
                    if x[2] == col and (x[0], x[1]) != (mr, mc)
                }
                if _conflicts_same_color_diagonal((mr, mc), existing | local_same):
                    diag_bad = True
                    break
            if diag_bad:
                continue

            dirs = _dirs_from_flags(perp_dirs, emit_up, emit_down)
            ray_cells = _compute_ray_cells_for_dirs((tr, tc), dirs, H, W)
            blocked = occupied | local_input | marker_cells | local_drawn
            if any(cell in blocked for cell in ray_cells):
                continue

            trigger_specs.append((tr, tc, emit_up, emit_down))
            local_input.add((tr, tc))
            for mr, mc, col in marker_specs:
                local_input.add((mr, mc))
                local_marker_positions[col].add((mr, mc))
            local_drawn.update(ray_cells)
            break

    return trigger_specs, local_marker_positions


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _UP_MARKER_COLOR, _DOWN_MARKER_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied = set()
    drawn = set()
    marker_positions = {_UP_MARKER_COLOR: set(), _DOWN_MARKER_COLOR: set()}
    segments = []

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))

        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result

        trigger_specs, new_marker_positions = _pick_trigger_specs(path, dc, H, W, occupied, drawn, marker_positions)
        perp_dirs = [(1, -dc), (-1, dc)]

        new_input_cells = {path[0], path[-1]}
        ray_cells = set()
        for tr, tc, emit_up, emit_down in trigger_specs:
            new_input_cells.add((tr, tc))
            for mr, mc, _ in _marker_specs(tr, tc, emit_up, emit_down):
                new_input_cells.add((mr, mc))
            dirs = _dirs_from_flags(perp_dirs, emit_up, emit_down)
            ray_cells.update(_compute_ray_cells_for_dirs((tr, tc), dirs, H, W))

        if any(cell in occupied for cell in ray_cells):
            continue

        ep0, ep1 = path[0], path[-1]
        segments.append((ep0, ep1, trigger_specs, path, perp_dirs))
        occupied.update(new_input_cells)
        drawn.update(set(path[1:-1]) - {(tr, tc) for tr, tc, _, _ in trigger_specs})
        drawn.update(ray_cells)
        for col, cells in new_marker_positions.items():
            marker_positions[col].update(cells)

    real_trigger_count = sum(len(trigger_specs) for _, _, trigger_specs, _, _ in segments)
    single_direction_count = sum(
        1
        for _, _, trigger_specs, _, _ in segments
        for _, _, emit_up, emit_down in trigger_specs
        if emit_up ^ emit_down
    )
    if not segments or real_trigger_count == 0 or single_direction_count == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_specs, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tr, tc, emit_up, emit_down in trigger_specs:
            inp[tr, tc] = _TRIGGER_COLOR
            if emit_up:
                inp[tr - 1, tc] = _UP_MARKER_COLOR
            if emit_down:
                inp[tr + 1, tc] = _DOWN_MARKER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(_diag_aligned((r, c), (lr, lc)) for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    all_magenta_positions = list(zip(*np.where(inp == _TRIGGER_COLOR)))
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn:
                continue
            if any(_diag_aligned((r, c), pos) for pos in all_magenta_positions):
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            all_magenta_positions.append((r, c))
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
