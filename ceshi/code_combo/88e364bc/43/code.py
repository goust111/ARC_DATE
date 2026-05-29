# Combo variant of 88e364bc (index 43)
# combo_spec: D9/0+D15/1
# Name: 仅基数方向+先水平后垂直

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste


# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 原题里，每个边缘图例只编码一个方向，黄色像素在封闭区域内一次性沿该方向滑到远端墙壁。
#    - 这次融合后，如果字面上把 D9/0 理解成“只剩上下左右四个总方向”，再把 D15/1 理解成
#      “先水平后垂直”，那么每个区域仍只会有一个轴向分量，第二阶段必然退化为空操作；D15 的
#      控制变量对输出完全不起作用。
#    - 为了避免这种退化，同时保持“只使用基数方向原语”的约束，新的 5x5 图例改成同时携带两条
#      轴向指令：
#         1. 一个左右红指示像素编码水平阶段是向左还是向右；
#         2. 一个上下红指示像素编码垂直阶段是向上还是向下。
#      整个过程从头到尾都只发生左右/上下滑动，没有任何对角一步。
#    - 小例子：区域内部是一个粗 L 形通道，黄色在左下角。
#         原始单步思路若按“先竖后横”会先上到顶端，再沿顶边滑到右端；
#         本变体固定“先横后竖”，它会先在底边向右挪到竖通道下方，再沿该列向上堆到顶端。
#      对于非矩形区域，两种次序的终点不同，因此中间态真的决定最终输出。
#
# b) 复杂度校准
#    - 颜色角色不变：0 背景，1 图例中心，2 红指示，4 黄色，墙色从 {3,5,6,7,8,9} 取。
#    - 图例仍然是边缘实心矩形，仍然由“填充色 C 绑定同色围墙”这条主线连接到区域，不新增颜色映射。
#    - 方向自由度被压到两个二元选择：左右二选一、上下二选一。这样每个样例里常能重复看到
#      同样的 5x5 图例语法，人类容易归纳。
#    - 为了让 D15/1 肉眼可见，生成器强制至少一个区域满足：
#         1. 水平阶段和垂直阶段都真正移动了黄色；
#         2. “先横后竖” 与 “先竖后横” 在同一输入上输出不同。
#
# c) 隐式假设审查
#    - 原 `_legend_directions()` 依赖“两个红点的质心只编码一个向量”。融合后这不成立：
#      两个红点分别承担横向和纵向语义，所以 solver 必须先把红点分成“横指示”和“纵指示”。
#    - 原 `_slide_points()` 中的对角线 corner-cutting 分支不再需要，因为新规则从不执行对角一步。
#    - 原 `main()` 默认一个区域只需要一次滑动。融合后不成立，同一区域必须串行执行两个阶段，
#      且第二阶段读取第一阶段产出的中间位置。
#    - 原按区域独立处理的假设仍然成立，因为围墙结构未变，不同区域之间仍不互相渗透。
#
# d) 放置合法性
#    - 图例放在四角，保留 1 格缓冲，防止与围墙组件相连。
#    - 区域先生成 interior，再取其 4 邻外环作为围墙；只要 interior 与其它对象留出缓冲，
#      就不会重叠、越界，也能保证形成封闭 passable 区域。
#    - 特殊区域使用预制的非矩形通道模板；放置前会显式检查墙环、缓冲区和已有对象是否冲突。
#
# e) 视觉可观察性
#    - 图例的两个红点一横一竖，非常直观地提示“这不是一个单向箭头，而是两阶段正交指令”。
#    - 生成器强制至少一个样例具有次序敏感性；因此从 2-4 对样例中，人类能看到黄色并不是简单地
#      去某个角，而是沿固定先后次序穿过不规则通道。
#
# f) 可视化整洁度
#    - 网格里仍只有三类主对象：角落图例、彩色围墙、黄色像素。没有再引入内部标记或辅助符号。
#    - 区域模板控制在少量、较大的连通形状，避免碎片化噪声。
#
# g) 变化可推断性
#    - 没有新增颜色转换规则；变化只体现在位置。
#    - 控制变量只有两项：水平指示方向、垂直指示方向，以及固定处理顺序“横后竖”。
#      这比随机混用多种顺序或多种编码更容易从少样例归纳。
#
# h) main() 的新假设与 generator 保证
#    - 假设每个合法图例都恰好有 4 个中心色像素和 2 个红指示像素，且两枚红点分别在“横向主导”
#      和“纵向主导”的位置上。generator 用固定 5x5 图例模板保证这一点。
#    - 假设区域内黄色都位于背景可通行格上。generator 只从 region interior 中取输入黄点。
#    - 假设至少一个区域真的是两阶段问题而不是单阶段退化。generator 用次序敏感模板和验算过滤保证。


GRID_H, GRID_W = 20, 20

CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1
GEN_RETRIES = 400
PLACEMENT_RETRIES = 80
MAX_TOTAL_REGIONS = 3

# (horizontal, vertical) where horizontal in {-1,+1}, vertical in {-1,+1}
PHASE_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _parse_shape(rows: list[str]) -> tuple[set[tuple[int, int]], int, int]:
    h = len(rows)
    w = len(rows[0])
    cells = {
        (r, c)
        for r, row in enumerate(rows)
        for c, ch in enumerate(row)
        if ch == "#"
    }
    return cells, h, w


SPECIAL_SHAPES = [
    _parse_shape(
        [
            "######",
            "##....",
            "##....",
            "##....",
            "##....",
        ]
    ),
    _parse_shape(
        [
            "######",
            "###...",
            "##....",
            "##....",
            "##....",
        ]
    ),
    _parse_shape(
        [
            "#####.",
            "##....",
            "##....",
            "####..",
            "##....",
        ]
    ),
]

ORDINARY_SHAPES = [
    _parse_shape(
        [
            "#####",
            "#####",
            "#####",
            "#####",
        ]
    ),
    _parse_shape(
        [
            "######",
            "######",
            "######",
        ]
    ),
    _parse_shape(
        [
            ".####.",
            "######",
            "######",
            ".####.",
        ]
    ),
]


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _legend_5x5(fill: int, hdc: int, vdr: int) -> np.ndarray:
    """
    5x5 fused legend.

    - four blue center pixels form a small cross around the middle
    - one red pixel on the same row encodes left/right
    - one red pixel on the same column encodes up/down
    """
    g = np.full((5, 5), fill, dtype=int)
    for r, c in [(2, 1), (1, 2), (2, 3), (3, 2)]:
        g[r, c] = 1
    g[2, 0 if hdc < 0 else 4] = 2
    g[0 if vdr < 0 else 4, 2] = 2
    return g


def _legend_phase_dirs(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Decode border legends into (horizontal_dir, vertical_dir).

    The 5x5 legend always has:
    - dominant fill color C = wall color
    - 4 center pixels
    - 2 red indicator pixels, one horizontally dominant and one vertically dominant
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) != 2:
            continue
        others.sort()
        indicator_col = others[0][1]
        center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) != 4 or len(indicator) != 2:
            continue

        cc = center.mean(axis=0)
        horiz = []
        vert = []
        for r, c in indicator:
            dr = float(r - cc[0])
            dc = float(c - cc[1])
            if abs(dc) > abs(dr):
                horiz.append(dc)
            elif abs(dr) > abs(dc):
                vert.append(dr)
        if len(horiz) != 1 or len(vert) != 1:
            continue

        hdc = 1 if horiz[0] > 0 else -1
        vdr = 1 if vert[0] > 0 else -1
        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (hdc, vdr))

    return {k: v for k, (_, v) in best.items()}


def _slide_axis(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> tuple[list[tuple[int, int]], int, int]:
    if not points or (dr == 0 and dc == 0):
        return list(points), 0, 0

    points_sorted = sorted(
        points,
        key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]),
        reverse=True,
    )
    occupied: set[tuple[int, int]] = set()
    out_pts: list[tuple[int, int]] = []
    moved_count = 0
    total_steps = 0
    for r0, c0 in points_sorted:
        r, c = r0, c0
        steps = 0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
            steps += 1
        if steps > 0:
            moved_count += 1
            total_steps += steps
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts, moved_count, total_steps


def _apply_region_order(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    hdc: int,
    vdr: int,
    order: tuple[str, str],
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    cur = list(points)
    stats = {
        "h_moved": 0,
        "v_moved": 0,
        "h_steps": 0,
        "v_steps": 0,
    }
    for stage in order:
        if stage == "h":
            cur, moved, steps = _slide_axis(region_set, cur, 0, hdc)
            stats["h_moved"] += moved
            stats["h_steps"] += steps
        else:
            cur, moved, steps = _slide_axis(region_set, cur, vdr, 0)
            stats["v_moved"] += moved
            stats["v_steps"] += steps
    return cur, stats


def _solve(
    grid: np.ndarray,
    *,
    order: tuple[str, str] = ("h", "v"),
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_phase_dirs(g, bg)
    assert dirs, "legend_phase_dirs must find at least one border legend"

    passable = (g == bg) | (g == mover)
    out = g.copy()
    stats = {
        "regions_seen": 0,
        "regions_with_move": 0,
        "regions_with_both_stage_moves": 0,
        "h_steps": 0,
        "v_steps": 0,
    }

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_counts: dict[int, int] = {}
        pset = {(int(r), int(c)) for r, c in pts}
        for r, c in pts:
            for nr, nc in neighbors4(int(r), int(c)):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        hdc, vdr = dirs[border_col]
        new_pts, region_stats = _apply_region_order(pset, movers, hdc, vdr, order)

        for r, c in movers:
            out[r, c] = bg
        for r, c in new_pts:
            out[r, c] = mover

        stats["regions_seen"] += 1
        stats["h_steps"] += region_stats["h_steps"]
        stats["v_steps"] += region_stats["v_steps"]
        if region_stats["h_steps"] > 0 or region_stats["v_steps"] > 0:
            stats["regions_with_move"] += 1
        if region_stats["h_steps"] > 0 and region_stats["v_steps"] > 0:
            stats["regions_with_both_stage_moves"] += 1

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside each enclosed region using a two-stage orthogonal instruction from the border legend.

    Input structure:
    - Background is the most common color; yellow pixels live on passable interior cells bounded by colored walls.
    - Each controlling legend is a 5x5 filled rectangle touching the outer border; its fill color matches one wall color.
    - Inside that legend, four blue center pixels mark the reference point, one red pixel points left/right, and another red pixel points up/down.

    Rule:
    - For each enclosed region, find the majority wall color around that region and match it to the legend with the same fill color.
    - First slide every yellow in that region horizontally according to the legend's left/right red marker, stacking against the far side.
    - Then, from those intermediate positions, slide the same yellows vertically according to the legend's up/down red marker.

    Key insight:
    - The legend no longer means one atomic arrow. It encodes two cardinal sub-moves, and the fixed processing order is part of the rule: horizontal first, vertical second.
    """
    return _solve(grid, order=("h", "v"), return_stats=False)


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _dilate_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _transform_shape(
    cells: set[tuple[int, int]],
    sh: int,
    sw: int,
    hdc: int,
    vdr: int,
) -> tuple[set[tuple[int, int]], int, int]:
    flip_r = vdr > 0
    flip_c = hdc < 0
    out = set()
    for r, c in cells:
        nr = sh - 1 - r if flip_r else r
        nc = sw - 1 - c if flip_c else c
        out.add((nr, nc))
    return out, sh, sw


def _translate(cells: set[tuple[int, int]], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r + r0, c + c0) for r, c in cells}


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    phase_dirs = rng.sample(PHASE_DIRS, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends: list[dict[str, int]] = []
    for corner, color, (hdc, vdr) in zip(corners, wall_colors, phase_dirs):
        patch = _legend_5x5(color, hdc, vdr)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo = max(0, r0 - LEGEND_BUFFER)
        r_hi = min(h, r0 + ph + LEGEND_BUFFER)
        c_lo = max(0, c0 - LEGEND_BUFFER)
        c_hi = min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append({"color": color, "hdc": hdc, "vdr": vdr})
    return legend_occupied, legends


def _try_place_region(
    rng: random.Random,
    occupied: np.ndarray,
    base_cells: set[tuple[int, int]],
    sh: int,
    sw: int,
    color: int,
    hdc: int,
    vdr: int,
    special: bool,
) -> dict | None:
    h, w = occupied.shape
    shape, sh, sw = _transform_shape(base_cells, sh, sw, hdc, vdr)

    r_choices = list(range(1, h - sh - 1))
    c_choices = list(range(1, w - sw - 1))
    rng.shuffle(r_choices)
    rng.shuffle(c_choices)
    attempts = 0
    for r0 in r_choices:
        for c0 in c_choices:
            attempts += 1
            if attempts > PLACEMENT_RETRIES:
                return None
            interior = _translate(shape, r0, c0)
            walls = _blob_wall_cells(interior, h, w)
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in walls):
                continue
            reserve = _dilate_cells(interior | walls, h, w)
            if any(occupied[r, c] for r, c in reserve):
                continue
            for r, c in reserve:
                occupied[r, c] = True
            return {
                "color": color,
                "hdc": hdc,
                "vdr": vdr,
                "interior": interior,
                "walls": walls,
                "special": special,
            }
    return None


def _region_singletons(
    region_set: set[tuple[int, int]],
    hdc: int,
    vdr: int,
    *,
    require_both: bool,
    require_order_diff: bool,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cands = []
    for cell in sorted(region_set):
        hv_out, hv_stats = _apply_region_order(region_set, [cell], hdc, vdr, ("h", "v"))
        vh_out, _ = _apply_region_order(region_set, [cell], hdc, vdr, ("v", "h"))
        if hv_out[0] == cell:
            continue
        if require_both and not (hv_stats["h_steps"] > 0 and hv_stats["v_steps"] > 0):
            continue
        if require_order_diff and hv_out == vh_out:
            continue
        cands.append((cell, hv_out[0]))
    return cands


def _pick_region_yellows(
    rng: random.Random,
    region: dict,
) -> list[tuple[int, int]] | None:
    interior = region["interior"]
    hdc = region["hdc"]
    vdr = region["vdr"]

    if region["special"]:
        special = _region_singletons(
            interior,
            hdc,
            vdr,
            require_both=True,
            require_order_diff=True,
        )
        if not special:
            return None
        start, _ = rng.choice(special)
        return [start]

    generic = _region_singletons(
        interior,
        hdc,
        vdr,
        require_both=False,
        require_order_diff=False,
    )
    if not generic:
        return None
    start, _ = rng.choice(generic)
    picks = [start]

    if rng.random() < 0.45:
        others = [cell for cell, _ in generic if cell != start]
        rng.shuffle(others)
        for cand in others:
            out_pts, stats = _apply_region_order(interior, [start, cand], hdc, vdr, ("h", "v"))
            if len(set(out_pts)) != 2:
                continue
            if stats["h_steps"] + stats["v_steps"] <= 1:
                continue
            picks.append(cand)
            break
    return picks


def _oracle_check(input_grid: np.ndarray) -> bool:
    try:
        out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != 4) & (input_grid != 0)
    if not np.array_equal(out[struct_mask], input_grid[struct_mask]):
        return False
    if int((out == 4).sum()) != int((input_grid == 4).sum()):
        return False
    return True


def _fusion_metrics(input_grid: np.ndarray) -> dict[str, int | bool]:
    hv_out, hv_stats = _solve(input_grid, order=("h", "v"), return_stats=True)
    vh_out, _ = _solve(input_grid, order=("v", "h"), return_stats=True)
    return {
        "order_diff": not np.array_equal(hv_out, vh_out),
        "regions_with_both_stage_moves": hv_stats["regions_with_both_stage_moves"],
        "h_steps": hv_stats["h_steps"],
        "v_steps": hv_stats["v_steps"],
        "yellow_count": int((input_grid == 4).sum()),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    occupied = legend_occupied.copy()

    region_specs: list[dict[str, int | bool]] = []
    total_regions_target = rng.choice([2, 3])
    first_color = rng.randrange(len(legends))
    special_assigned = False
    while len(region_specs) < total_regions_target:
        lg = legends[first_color if len(region_specs) == 0 else rng.randrange(len(legends))]
        special = not special_assigned
        region_specs.append(
            {
                "color": lg["color"],
                "hdc": lg["hdc"],
                "vdr": lg["vdr"],
                "special": special,
            }
        )
        if special:
            special_assigned = True

    regions = []
    for spec in region_specs:
        base_cells, sh, sw = rng.choice(SPECIAL_SHAPES if spec["special"] else ORDINARY_SHAPES)
        reg = _try_place_region(
            rng,
            occupied,
            base_cells,
            sh,
            sw,
            int(spec["color"]),
            int(spec["hdc"]),
            int(spec["vdr"]),
            bool(spec["special"]),
        )
        if reg is None:
            return None
        regions.append(reg)

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    input_grid = grid.copy()
    total_yellows = 0
    for reg in regions:
        yellows = _pick_region_yellows(rng, reg)
        if not yellows:
            return None
        total_yellows += len(yellows)
        for r, c in yellows:
            input_grid[r, c] = 4

    if total_yellows < 2:
        return None
    if not _oracle_check(input_grid):
        return None

    metrics = _fusion_metrics(input_grid)
    if metrics["regions_with_both_stage_moves"] < 1:
        return None
    if not metrics["order_diff"]:
        return None
    if metrics["h_steps"] <= 0 or metrics["v_steps"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
