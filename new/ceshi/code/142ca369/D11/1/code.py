# Variant of 142ca369
# Dimension: D11 — 箭头数量与密度 (meta)
# Change: 1-14个箭头，每个独立发射射线，箭头间无交互... → 成对箭头（相向发射）
#
# Design analysis
# ---
# variant_spec:
#   idea:
#     core_change: |
#       每种箭头颜色恰好对应一对箭头；同色两箭头沿同一条折线型对角路径相向发射，
#       两条射线按原题相同的“对角前进 + 侧触墙反射 + 继承墙色”规则同步推进。
#       当两条射线在同一步进入同一格时，它们在该格相互抵消：两条射线同时停止，交点保持背景色。
#     preserved_from_original:
#       - 箭头仍是单色 3 像素 L 形，方向仍由缺角决定。
#       - 墙仍是单色 1 像素或 3 像素轴对齐线段。
#       - 侧触墙时仍翻转一个轴，并把当前格改涂为墙色，后续继续继承该颜色。
#     deliberate_encoding: |
#       “谁和谁配对”如果不编码出来，人类很难从 2-3 个样例里确定。这里采用最小改动：
#       同色恰好两枚箭头即为一对。这样视觉上容易看出“同色两端互相朝向”。
#   behavior_trace:
#     original_local_flow: |
#       原题对每个箭头单独 while-loop 追踪；遇到别的箭头只会停止，不会把别的箭头视作交互对象。
#     new_local_flow: |
#       先分别按原题规则求出一对箭头各自的逐步射线路径，再按时间步并行合成：
#       若第 t 步两条路径到达不同格，则两格都保留；若第 t 步到达同一格，则该格作为交点保留一次，
#       两条射线在此合并并停止。
#     concrete_example: |
#       若左上箭头朝右下、右下同色箭头朝左上，且中间无墙：
#       原题会分别向外延伸直至出界；新题会分别向内延伸，并在中点相遇时同时消失，
#       输出是两段朝中间逼近但中点留空的对角线，而不是两条朝边界发散的射线。
#   complexity_calibration:
#     why_not_more_complex: |
#       不引入跨颜色配对、速度差、优先级覆盖等额外自由度；关键新信息只有“成对 + 同步相遇”。
#       反射和颜色继承仍沿用原题，以保留足够结构复杂度。
#     why_not_too_simple: |
#       生成器不会退化成“总是两箭头直连无墙”。每对箭头允许 0-2 次镜像侧反射，
#       所以样例中既能看到相向合并，也能看到与原题一致的墙色继承现象。
#   implicit_assumption_audit:
#     arrow_detection: |
#       原代码按颜色找 3 像素 L 形组件；这在变体中仍成立，因为每个箭头形状未变。
#     old_independence_bug: |
#       原代码的 `for start_col, pts in arrows:` 隐含“每条射线可独立落笔”的前提；
#       在变体中如果仍顺序写出，会把先处理的一条射线直接画过交点，破坏同步相遇语义。
#       因此必须改成“先单独追踪，再按成对时间步合成”。
#     wall_color_overwrite: |
#       原代码只在输出格仍为背景时涂色；这在独立射线下无害。
#       变体中交点是双方共享的“待进入格”，不能简单让第二条射线因为该格非背景而失效；
#       需要显式识别“同一步同格”的碰撞，并保持该格为空。
#     pairing_assumption: |
#       新 solver 假设“同色箭头数为 2 时构成一对”；生成器会强制满足。
#       若出现非成对颜色，solver 会退回原题式单箭头追踪，不崩溃。
#   placement_legality:
#     geometric_guarantees: |
#       生成器先构造以交点为中心的中心对称折线，再从折线两端反推箭头位置。
#       这样天然保证两半路径长度相等、同步到达交点，并且两箭头方向相向。
#     collision_rules: |
#       任意一对的箭头像素、墙像素、输出射线像素都不得与已有对象重叠；
#       镜像出来的两半路径只允许在交点 1 格重合。
#     wall_rules: |
#       每次转折只允许翻转一个轴，因此只使用原题的 side-touch 墙；
#       不生成“正前方撞墙”的 corner bounce，避免和配对同步语义缠绕。
#   visual_observability:
#     human_readability: |
#       输入里可以直接看到“同色两箭头位于同一折线路径两端”；输出则把这条路径从两端补向中间，
#       但在碰撞格留下一个清晰空洞，与原题“朝外延伸到边界”有明显不同。
#     branch_support: |
#       样例中会混合：
#       - 1-3 对箭头
#       - 0 / 1 / 2 次镜像反射
#       - 有些对只保留箭头色，有些对在中途继承墙色
#       这些重复支撑足以让人看出规律。
#   cleanliness_and_color_inference:
#     cleanliness: |
#       折线路径限制为不自交；不同 pair 的占用区域完全分离；墙只贴在各自转折点一侧，
#       避免多层叠加难以分辨。
#     color_logic: |
#       颜色变化仍只有一种来源：墙色继承。因为两侧墙成镜像且同色，同步推进时两条射线始终共享同一“当前色”，
#       所以“为什么只留下中心空洞”也容易归纳：新效果只改变交点占用，不引入新的任意颜色映射。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]):
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None

    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _build_arrow_pts(start: tuple[int, int], fire_dir: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = fire_dir
    mr, mc = sr - dr, sc - dc
    pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
    return [(int(r), int(c)) for r, c in pts]


def _detect_arrows(g: np.ndarray, bg: int):
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    arrows_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    arrow_cells: set[tuple[int, int]] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrows_by_color.setdefault(int(col), []).append(pts)
                arrow_cells |= set(pts)
    return arrows, arrows_by_color, arrow_cells


def _trace_ray_steps(
    g: np.ndarray,
    start_col: int,
    pts: list[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    wall_cells: set[tuple[int, int]],
    bg: int,
) -> list[tuple[tuple[int, int], int]]:
    """Trace one ray exactly as the original solver would, but record per-step cells/colors."""
    info = _arrow_dir_and_start(pts)
    if info is None:
        return []

    H, W = g.shape
    (dr, dc), cur = info
    cur_col = int(start_col)
    seen = set()
    steps: list[tuple[tuple[int, int], int]] = []

    while True:
        state = (cur, int(dr), int(dc), int(cur_col))
        if state in seen:
            break
        seen.add(state)

        nr = int(cur[0] + dr)
        nc = int(cur[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            break

        if (nr, nc) in wall_cells:
            wcol = int(g[nr, nc])
            if steps and int(g[int(cur[0]), int(cur[1])]) == int(bg):
                steps[-1] = (steps[-1][0], int(wcol))
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            continue

        if (nr, nc) in arrow_cells:
            break

        cur = (int(nr), int(nc))
        paint_col = int(cur_col)

        if int(dr) == 1 and (nr + 1, nc) in wall_cells:
            wcol = int(g[nr + 1, nc])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)
        elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
            wcol = int(g[nr - 1, nc])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)

        if int(dc) == 1 and (nr, nc + 1) in wall_cells:
            wcol = int(g[nr, nc + 1])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)
        elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
            wcol = int(g[nr, nc - 1])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)

        steps.append((cur, int(paint_col)))

    return steps


def _paint_steps(out: np.ndarray, steps: list[tuple[tuple[int, int], int]], bg: int) -> None:
    for (r, c), col in steps:
        if int(out[r, c]) == int(bg):
            out[r, c] = int(col)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    arrows, arrows_by_color, arrow_cells = _detect_arrows(g, bg)
    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()
    paired_ids: set[int] = set()

    for col, group in arrows_by_color.items():
        if len(group) != 2:
            continue

        pts_a, pts_b = group
        steps_a = _trace_ray_steps(g, int(col), pts_a, arrow_cells, wall_cells, bg)
        steps_b = _trace_ray_steps(g, int(col), pts_b, arrow_cells, wall_cells, bg)

        meet_idx = None
        for i, (step_a, step_b) in enumerate(zip(steps_a, steps_b)):
            if step_a[0] == step_b[0]:
                meet_idx = i
                break

        if meet_idx is None:
            _paint_steps(out, steps_a, bg)
            _paint_steps(out, steps_b, bg)
        else:
            _paint_steps(out, steps_a[:meet_idx], bg)
            _paint_steps(out, steps_b[:meet_idx], bg)

        paired_ids.add(id(pts_a))
        paired_ids.add(id(pts_b))

    for col, pts in arrows:
        if id(pts) in paired_ids:
            continue
        _paint_steps(out, _trace_ray_steps(g, int(col), pts, arrow_cells, wall_cells, bg), bg)

    return out


def _reflect(center: tuple[int, int], cell: tuple[int, int]) -> tuple[int, int]:
    cr, cc = center
    r, c = cell
    return int(2 * cr - r), int(2 * cc - c)


def _dirs_for_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1]) for i in range(len(cells) - 1)]


def _turns_for_half(cells: list[tuple[int, int]]):
    dirs = _dirs_for_cells(cells)
    turns: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for i in range(1, len(cells) - 1):
        prev_dir = dirs[i - 1]
        next_dir = dirs[i]
        if prev_dir == next_dir:
            continue
        if prev_dir[0] != next_dir[0] and prev_dir[1] != next_dir[1]:
            return None
        flip_axis = "dr" if prev_dir[0] != next_dir[0] else "dc"
        turns.append((cells[i], prev_dir, flip_axis))
    return turns


def _wall_shape(
    turn_cell: tuple[int, int],
    incoming_dir: tuple[int, int],
    flip_axis: str,
    size: int,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    r, c = turn_cell
    dr, dc = incoming_dir
    if flip_axis == "dr":
        wr, wc = r + dr, c
        if not (0 <= wr < H and 0 <= wc < W):
            return None
        if size == 1:
            return [(int(wr), int(wc))]
        cells = [(int(wr), int(wc + off)) for off in (-1, 0, 1)]
        return cells if all(0 <= cc < W for _, cc in cells) else None

    wr, wc = r, c + dc
    if not (0 <= wr < H and 0 <= wc < W):
        return None
    if size == 1:
        return [(int(wr), int(wc))]
    cells = [(int(wr + off), int(wc)) for off in (-1, 0, 1)]
    return cells if all(0 <= rr < H for rr, _ in cells) else None


def _colorize_half(
    half_cells: list[tuple[int, int]],
    pair_color: int,
    turn_colors: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], int]]:
    cur_col = int(pair_color)
    records: list[tuple[tuple[int, int], int]] = []
    for cell in half_cells:
        paint_col = int(turn_colors.get(cell, cur_col))
        records.append((cell, int(paint_col)))
        if cell in turn_colors:
            cur_col = int(turn_colors[cell])
    return records


def _gen_pair(
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    pair_color: int,
    palette: list[int],
):
    for _ in range(180):
        center = (_rng.randint(4, H - 5), _rng.randint(4, W - 5))
        n_turns = _rng.choices([0, 1, 2], weights=[3, 4, 2])[0]
        seg_lengths = [_rng.randint(2, 4) for _ in range(n_turns + 1)]
        cur_dir = _rng.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        current = center
        outward_cells: list[tuple[int, int]] = []
        used_half: set[tuple[int, int]] = {center}
        outward_ok = True
        for seg_idx, seg_len in enumerate(seg_lengths):
            for _step in range(seg_len):
                current = (int(current[0] + cur_dir[0]), int(current[1] + cur_dir[1]))
                if not (0 <= current[0] < H and 0 <= current[1] < W):
                    outward_ok = False
                    break
                if current in used_half or current in occupied:
                    outward_ok = False
                    break
                outward_cells.append(current)
                used_half.add(current)
            if not outward_ok:
                break
            if seg_idx < n_turns:
                if _rng.random() < 0.5:
                    cur_dir = (-cur_dir[0], cur_dir[1])
                else:
                    cur_dir = (cur_dir[0], -cur_dir[1])
        if not outward_ok or len(outward_cells) < 3:
            continue

        half_a = list(reversed(outward_cells)) + [center]
        half_b = [_reflect(center, cell) for cell in half_a[:-1]] + [center]
        all_ray_cells = set(half_a) | set(half_b)
        if len(all_ray_cells) != len(half_a) + len(half_b) - 1:
            continue
        if (all_ray_cells - {center}) & occupied:
            continue

        dirs_a = _dirs_for_cells(half_a)
        dirs_b = _dirs_for_cells(half_b)
        if not dirs_a or not dirs_b:
            continue
        start_a = (int(half_a[0][0] - dirs_a[0][0]), int(half_a[0][1] - dirs_a[0][1]))
        start_b = (int(half_b[0][0] - dirs_b[0][0]), int(half_b[0][1] - dirs_b[0][1]))
        if not (0 <= start_a[0] < H and 0 <= start_a[1] < W and 0 <= start_b[0] < H and 0 <= start_b[1] < W):
            continue

        arrow_a = _build_arrow_pts(start_a, dirs_a[0])
        arrow_b = _build_arrow_pts(start_b, dirs_b[0])
        arrow_cells = set(arrow_a) | set(arrow_b)
        if len(arrow_cells) != 6:
            continue
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_cells):
            continue
        if arrow_cells & occupied:
            continue
        if arrow_cells & all_ray_cells:
            continue

        turns_a = _turns_for_half(half_a)
        turns_b = _turns_for_half(half_b)
        if turns_a is None or turns_b is None or len(turns_a) != len(turns_b):
            continue

        turn_colors_a: dict[tuple[int, int], int] = {}
        turn_colors_b: dict[tuple[int, int], int] = {}
        wall_segments: list[tuple[list[tuple[int, int]], int]] = []
        reserved = set(all_ray_cells) | set(arrow_cells) | set(occupied)
        walls_ok = True

        for idx, (ta, tb) in enumerate(zip(turns_a, turns_b)):
            turn_a, in_dir_a, flip_a = ta
            turn_b, in_dir_b, flip_b = tb
            if flip_a != flip_b:
                walls_ok = False
                break

            size_options = [3, 1] if _rng.random() < 0.6 else [1, 3]
            seg_a = seg_b = None
            for size in size_options:
                cand_a = _wall_shape(turn_a, in_dir_a, flip_a, size, H, W)
                cand_b = _wall_shape(turn_b, in_dir_b, flip_b, size, H, W)
                if cand_a is None or cand_b is None:
                    continue
                cells = set(cand_a) | set(cand_b)
                if len(cells) != len(cand_a) + len(cand_b):
                    continue
                if cells & reserved:
                    continue
                seg_a, seg_b = cand_a, cand_b
                break
            if seg_a is None or seg_b is None:
                walls_ok = False
                break

            wall_color_choices = [c for c in palette if c != pair_color] or list(palette)
            wall_color = int(_rng.choice(wall_color_choices))
            turn_colors_a[turn_a] = wall_color
            turn_colors_b[turn_b] = wall_color
            wall_segments.append((seg_a, wall_color))
            wall_segments.append((seg_b, wall_color))
            reserved |= set(seg_a) | set(seg_b)

        if not walls_ok:
            continue

        records_a = _colorize_half(half_a, pair_color, turn_colors_a)
        records_b = _colorize_half(half_b, pair_color, turn_colors_b)
        if records_a[-1][1] != records_b[-1][1]:
            continue

        pair_cells = set(arrow_cells) | set(all_ray_cells)
        for wall_cells, _col in wall_segments:
            pair_cells |= set(wall_cells)

        return {
            "arrow_color": int(pair_color),
            "arrows": [arrow_a, arrow_b],
            "records": [records_a, records_b],
            "walls": wall_segments,
            "occupied": pair_cells,
            "turn_count": len(turns_a),
        }

    return None


def _apply_pair_to_grids(
    grid: np.ndarray,
    out: np.ndarray,
    pair_data: dict,
) -> None:
    color = int(pair_data["arrow_color"])
    for arrow in pair_data["arrows"]:
        for r, c in arrow:
            grid[r, c] = color
            out[r, c] = color

    for wall_cells, wall_color in pair_data["walls"]:
        for r, c in wall_cells:
            grid[r, c] = int(wall_color)
            out[r, c] = int(wall_color)

    for records in pair_data["records"]:
        # The final cell is the collision cell; annihilation leaves it blank.
        for (r, c), paint_col in records[:-1]:
            out[r, c] = int(paint_col)


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_pairs = _rng.randint(1, 3)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    pair_colors = palette[:n_pairs]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    total_turns = 0
    for pair_color in pair_colors:
        pair_data = _gen_pair(H, W, occupied, int(pair_color), palette)
        if pair_data is None:
            return None
        occupied |= set(pair_data["occupied"])
        total_turns += int(pair_data["turn_count"])
        _apply_pair_to_grids(grid, out, pair_data)

    if total_turns == 0 and n_pairs == 1:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
