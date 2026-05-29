# Combo variant of 271d71e2 (index 25)
# combo_spec: D4/0+D6/2+D11/0
# Name: 无边框颜色配对不等长目标

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 原题对象是「黑框容器 + 同色同长 head/dest 线」。solver 通过 5/7 内部向外扩一圈找黑框，
#   再在黑框外找 maroon 线，并按“同色同形”找到 dest。
# - 融合后有三处同时变化：
#   1. D4/0: 黑框消失，容器本体直接是一个由 5/7 组成的实心矩形。
#   2. D6/2: 每个容器的 head/dest 使用自己的可见颜色，靠颜色配对。
#   3. D11/0: dest 与 head 不等长，但保持同朝向、同颜色、同中心对齐。
# - 一个具体小例子（head 在上侧）：
#   输入里某容器是 4x6 的 5/7 矩形；其上方紧贴一条长度 6 的蓝线作为 head；
#   再往上隔若干背景行，有一条长度 8 的蓝线作为 dest，和 head 共享同一个列中心。
#   这个容器向上移动 `min(灰格数, head行与dest行的距离)` 步；移动的同时，最靠上侧的灰格
#   依照原题顺序逐个变橙。与原题相比，差异不是局部替换，而是完整的推理链变化：
#   容器识别不再靠黑框，配对不再靠全局固定颜色，dest 识别不再靠同长同形。
#
# b) 复杂度校准
# - 为避免无框容器退化成难以辨认的单点，生成器将容器尺寸限制在至少 2x2。
# - marker 颜色来自小而稳定的集合 {1,2,3,4,8,9}，每个容器在单个样例内使用唯一颜色。
#   这样颜色配对明确，但不会引入“同色多目标”的额外歧义。
# - dest 长度始终与 head 长度不同，且保持同奇偶性（用 ±2/±4 调整），这样可以做到严格中心对齐，
#   同时肉眼能直接看到“目标并不是靠同长匹配”。
#
# c) 隐式假设审查
# - 原 solver 中失效的假设：
#   - 5/7 连通块外必须有一圈黑框：融合后不存在，必须删掉。
#   - head 颜色可以靠“最常见邻接色”粗略锁定：多颜色后不够稳，必须改为逐侧验证。
#   - dest 必须与 head 同形、同 top-left 对齐：D11/0 后不成立，必须改为“同色、同朝向、
#     同中心、在前方”的规则。
# - 新 solver 的每个关键分支都对应一个由生成器保证的显式前提：
#   - 容器是完整实心矩形。
#   - head 是紧贴容器某一整条边的单层实线。
#   - dest 是同色同朝向的单层实线，长度与 head 不同，且中心对齐。
#   - 每个颜色在一个样例中只服务一个容器，因此不会出现跨容器同色抢配对。
#
# d) 放置合法性
# - 生成器仍按 assembly 放置，每个 assembly 包含 dest + gap + head + container。
# - 对于不等长 dest，assembly 在正交方向上的跨度取 `max(container_edge_len, dest_len)`，
#   再把 container 和 dest 各自居中放入该跨度，保证中心对齐且不越界。
# - 容器移动最多 `gap + 1` 步，因此输出中的 moved container/head 仍被限制在自己的 assembly 内。
# - assemblies 之间额外保留至少 1 格背景缓冲，保证输入和输出都不会互相接触。
#
# e) 视觉可观察性
# - 人类从 2-4 对样例可以同时看出三件事：黑框不见了；head 要和同色 dest 配对；dest 长短不同。
# - 中心对齐让“不等长但仍然配对”这件事在像素层面是可见的，不需要隐藏规则或数值编码。
#
# f) 可视化整洁度
# - 每个 assembly 内只有一块 5/7 矩形和两条同色细线，不叠加额外标记。
# - 颜色数受控、对象间有背景缓冲，因此不会出现一团混在一起的视觉噪声。
#
# g) 变化的可推断性
# - D4/0 在所有样例中恒成立：容器永远无黑框。
# - D6/2 在所有样例中恒成立：至少两个不同 marker 颜色，且容器按颜色找目标。
# - D11/0 在所有样例中恒成立：所有 dest 长度都与 head 长度不同。
# - 这些变化都是结构性的、直接可见的，不依赖人类不可见的颜色编号大小等信息。
#
# h) main() 的新隐式假设与 generate_input() 的保证
# - `_detect_containers()` 假设每个容器都是单个 4 连通、无孔洞的 5/7 实心矩形；
#   生成器直接按整块矩形写入，并用 assembly 间隔避免两个容器拼接成更大连通块。
# - `_pick_head_and_dest()` 假设只有一侧会出现“紧贴容器整边的单色 head 线”；
#   生成器每个容器只放一条 head，且其它三侧保持背景。
# - `_find_dest_for_head()` 假设正确 dest 与 head 同色、同朝向、中心对齐、长度不同；
#   生成器用唯一颜色 + 居中摆放 + `dest_len != head_len` 显式构造这一点。
# - 平移/充能逻辑假设 head 到 dest 的运动路径上没有别的非背景干扰；
#   生成器把每个对象组封装在独立 assembly 中，并在 assembly 外留背景缓冲，
#   因而输入与输出都不会发生跨组接触或覆盖。


BG = 6
GREY = 5
ORANGE = 7
MARKER_COLORS = [1, 2, 3, 4, 8, 9]

TOP = 0
BOTTOM = 1
LEFT = 2
RIGHT = 3

SIDE_TO_DIR = {
    TOP: (-1, 0),
    BOTTOM: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


def _fill_order(height: int, width: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    if head_side == RIGHT:
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == LEFT:
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == BOTTOM:
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _side_line(box: tuple[int, int, int, int], side: int, offset: int) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    d = int(offset)
    if side == TOP:
        rr = r0 - d
        if rr < 0:
            return None
        return [(rr, c) for c in range(c0, c1 + 1)]
    if side == BOTTOM:
        rr = r1 + d
        return [(rr, c) for c in range(c0, c1 + 1)]
    if side == LEFT:
        cc = c0 - d
        if cc < 0:
            return None
        return [(r, cc) for r in range(r0, r1 + 1)]
    cc = c1 + d
    return [(r, cc) for r in range(r0, r1 + 1)]


def _line_info(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> dict | None:
    pts = list(coords)
    if not pts:
        return None
    arr = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h == 1 and len(pts) == w:
        return {
            "axis": "h",
            "tl": (int(r0), int(c0)),
            "line_coord": int(r0),
            "center2": int(2 * c0 + w - 1),
            "length": int(w),
            "pts": {(int(r), int(c)) for r, c in pts},
        }
    if w == 1 and len(pts) == h:
        return {
            "axis": "v",
            "tl": (int(r0), int(c0)),
            "line_coord": int(c0),
            "center2": int(2 * r0 + h - 1),
            "length": int(h),
            "pts": {(int(r), int(c)) for r, c in pts},
        }
    return None


def _line_in_bounds(shape: tuple[int, int], line: list[tuple[int, int]] | None) -> bool:
    if line is None:
        return False
    H, W = shape
    return all(0 <= int(r) < H and 0 <= int(c) < W for r, c in line)


def _runs_in_row(grid: np.ndarray, row: int, color: int) -> list[dict]:
    W = grid.shape[1]
    runs: list[dict] = []
    c = 0
    while c < W:
        if int(grid[row, c]) != int(color):
            c += 1
            continue
        c0 = c
        while c < W and int(grid[row, c]) == int(color):
            c += 1
        pts = [(int(row), int(cc)) for cc in range(c0, c)]
        info = _line_info(pts)
        if info is not None:
            runs.append(info)
    return runs


def _runs_in_col(grid: np.ndarray, col: int, color: int) -> list[dict]:
    H = grid.shape[0]
    runs: list[dict] = []
    r = 0
    while r < H:
        if int(grid[r, col]) != int(color):
            r += 1
            continue
        r0 = r
        while r < H and int(grid[r, col]) == int(color):
            r += 1
        pts = [(int(rr), int(col)) for rr in range(r0, r)]
        info = _line_info(pts)
        if info is not None:
            runs.append(info)
    return runs


def _detect_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    mask = (grid == GREY) | (grid == ORANGE)
    for pts in components_mask(mask, connectivity=4):
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if pts.shape[0] != patch.size:
            continue
        if not bool(np.isin(patch, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (int(r0), int(c0), int(r1), int(c1))})
    return containers


def _find_dest_for_head(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    side: int,
    color: int,
) -> tuple[dict | None, dict | None, int | None]:
    head_line = _side_line(box, side, 1)
    if not _line_in_bounds(grid.shape, head_line):
        return None, None, None
    head_info = _line_info(head_line)
    if head_info is None:
        return None, None, None

    H, W = grid.shape
    dr, dc = SIDE_TO_DIR[side]
    head_coord = int(head_info["line_coord"])
    for step in range(1, max(H, W) + 1):
        coord = int(head_coord + (dr + dc) * step)
        if head_info["axis"] == "h":
            if not (0 <= coord < H):
                break
            runs = _runs_in_row(grid, coord, color)
        else:
            if not (0 <= coord < W):
                break
            runs = _runs_in_col(grid, coord, color)
        for run in runs:
            if int(run["center2"]) != int(head_info["center2"]):
                continue
            if int(run["length"]) == int(head_info["length"]):
                continue
            return head_info, run, step
    return head_info, None, None


def _pick_head_and_dest(grid: np.ndarray, bg: int, box: tuple[int, int, int, int]) -> dict | None:
    candidates = []
    for side in (TOP, BOTTOM, LEFT, RIGHT):
        line = _side_line(box, side, 1)
        if not _line_in_bounds(grid.shape, line):
            continue
        vals = [int(grid[r, c]) for r, c in line]
        uniq = set(vals)
        if len(uniq) != 1:
            continue
        color = vals[0]
        if color in (int(bg), GREY, ORANGE):
            continue
        head_info, dest_info, distance = _find_dest_for_head(grid, box, side, int(color))
        if head_info is None or dest_info is None or distance is None:
            continue
        candidates.append(
            {
                "side": int(side),
                "color": int(color),
                "dir": SIDE_TO_DIR[side],
                "head": head_info,
                "dest": dest_info,
                "distance": int(distance),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["distance"])
    return candidates[0]


def _inspect_case(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    cases = []
    for cont in _detect_containers(g):
        box = cont["box"]
        match = _pick_head_and_dest(g, bg, box)
        if match is None:
            continue
        r0, c0, r1, c1 = box
        cases.append(
            {
                "box": box,
                "side": int(match["side"]),
                "color": int(match["color"]),
                "head_len": int(match["head"]["length"]),
                "dest_len": int(match["dest"]["length"]),
                "distance": int(match["distance"]),
                "grey": int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == GREY)),
            }
        )
    return cases


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move borderless grey/orange containers toward color-matched, center-aligned, unequal-length targets.

    Input structure:
    - Background is 6.
    - Each container is a solid GREY/ORANGE rectangle with no black border.
    - A head line of some visible color is flush against one side of the container.
    - The matching target is a same-colored line further along that axis, but with different length.
      Head and target are center-aligned on the orthogonal axis.

    Rule:
    - Steps = min(head-to-target axis distance, grey cell count).
    - Container + head translate by `steps` toward the target.
    - Grey cells turn orange from the head-facing side inward, one per step.
    """
    g = np.array(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    containers = _detect_containers(g)
    assert containers

    out = g.copy()
    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        match = _pick_head_and_dest(g, bg, cont["box"])
        if match is None:
            continue

        side = int(match["side"])
        color = int(match["color"])
        head_pts = set(match["head"]["pts"])
        distance = int(match["distance"])
        dr, dc = match["dir"]

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        grey_local = [(rr, cc) for rr in range(patch.shape[0]) for cc in range(patch.shape[1]) if int(patch[rr, cc]) == GREY]
        steps = min(int(distance), int(len(grey_local)))
        tr = int(dr * steps)
        tc = int(dc * steps)

        to_fill = []
        for rr, cc in _fill_order(patch.shape[0], patch.shape[1], side):
            if int(patch[rr, cc]) == GREY:
                to_fill.append((rr, cc))
                if len(to_fill) == steps:
                    break

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for rr, cc in head_pts:
            if int(out[rr, cc]) == int(color):
                out[rr, cc] = int(bg)

        moved_patch = patch.copy()
        for rr, cc in to_fill:
            moved_patch[rr, cc] = ORANGE
        paste(out, moved_patch, int(r0 + tr), int(c0 + tc), clip=True)

        for rr, cc in head_pts:
            nr = int(rr + tr)
            nc = int(cc + tc)
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = int(color)

    return out


def _choose_dest_len(head_len: int) -> int:
    choices = []
    for delta in (2, 4):
        if head_len - delta >= 1:
            choices.append(head_len - delta)
        if head_len + delta <= 12:
            choices.append(head_len + delta)
    assert choices
    return int(random.choice(choices))


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    marker_colors = random.sample(MARKER_COLORS, n_containers)

    specs = []
    for idx in range(n_containers):
        is_decoy = idx >= n_movers
        cont_h = random.randint(2, 8)
        cont_w = random.randint(2, 6)
        area = cont_h * cont_w
        head_side = random.randint(0, 3)
        head_len = cont_w if head_side in (TOP, BOTTOM) else cont_h
        dest_len = _choose_dest_len(head_len)

        if is_decoy:
            grey = 0
            initial_orange = area
            gap = random.randint(0, 6)
            steps = 0
        else:
            grey = random.randint(1, area)
            initial_orange = area - grey
            gap = random.randint(0, 10)
            steps = min(grey, gap + 1)

        specs.append(
            {
                "cont_h": cont_h,
                "cont_w": cont_w,
                "head_side": head_side,
                "head_len": head_len,
                "dest_len": dest_len,
                "gap": gap,
                "steps": steps,
                "grey": grey,
                "initial_orange": initial_orange,
                "is_decoy": is_decoy,
                "marker_color": int(marker_colors[idx]),
            }
        )

    assemblies = []
    for spec in specs:
        cont_h = spec["cont_h"]
        cont_w = spec["cont_w"]
        head_side = spec["head_side"]
        gap = spec["gap"]
        dest_len = spec["dest_len"]

        if head_side in (TOP, BOTTOM):
            span_w = max(cont_w, dest_len)
            cont_c = (span_w - cont_w) // 2
            dest_c = (span_w - dest_len) // 2
            if head_side == TOP:
                total_h = 1 + gap + 1 + cont_h
                total_w = span_w
                cont_r = gap + 2
                dest_r = 0
            else:
                total_h = cont_h + 1 + gap + 1
                total_w = span_w
                cont_r = 0
                dest_r = cont_h + 1 + gap
            assemblies.append(
                {
                    **spec,
                    "total_h": total_h,
                    "total_w": total_w,
                    "cont_r": cont_r,
                    "cont_c": cont_c,
                    "dest_r": dest_r,
                    "dest_c": dest_c,
                }
            )
        else:
            span_h = max(cont_h, dest_len)
            cont_r = (span_h - cont_h) // 2
            dest_r = (span_h - dest_len) // 2
            if head_side == LEFT:
                total_h = span_h
                total_w = 1 + gap + 1 + cont_w
                cont_c = gap + 2
                dest_c = 0
            else:
                total_h = span_h
                total_w = cont_w + 1 + gap + 1
                cont_c = 0
                dest_c = cont_w + 1 + gap
            assemblies.append(
                {
                    **spec,
                    "total_h": total_h,
                    "total_w": total_w,
                    "cont_r": cont_r,
                    "cont_c": cont_c,
                    "dest_r": dest_r,
                    "dest_c": dest_c,
                }
            )

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area**0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for asm in assemblies:
        th, tw = asm["total_h"], asm["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(300):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + th, c0 : c0 + tw] = True
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    for asm, (pr, pc) in zip(assemblies, placements):
        cont_h = asm["cont_h"]
        cont_w = asm["cont_w"]
        head_side = asm["head_side"]
        gap = asm["gap"]
        steps = asm["steps"]
        initial_orange = asm["initial_orange"]
        marker_color = int(asm["marker_color"])
        dest_len = asm["dest_len"]
        dr, dc = SIDE_TO_DIR[head_side]

        cr = pr + asm["cont_r"]
        cc = pc + asm["cont_c"]
        dest_r = pr + asm["dest_r"]
        dest_c = pc + asm["dest_c"]

        fill_order = _fill_order(cont_h, cont_w, head_side)
        container = np.full((cont_h, cont_w), GREY, dtype=int)
        for idx in range(initial_orange):
            rr, cc0 = fill_order[idx]
            container[rr, cc0] = ORANGE
        paste(grid, container, cr, cc)
        paste(expected, container, cr, cc)

        if head_side == TOP:
            head_coords = [(cr - 1, c) for c in range(cc, cc + cont_w)]
            dest_coords = [(dest_r, c) for c in range(dest_c, dest_c + dest_len)]
        elif head_side == BOTTOM:
            head_coords = [(cr + cont_h, c) for c in range(cc, cc + cont_w)]
            dest_coords = [(dest_r, c) for c in range(dest_c, dest_c + dest_len)]
        elif head_side == LEFT:
            head_coords = [(r, cc - 1) for r in range(cr, cr + cont_h)]
            dest_coords = [(r, dest_c) for r in range(dest_r, dest_r + dest_len)]
        else:
            head_coords = [(r, cc + cont_w) for r in range(cr, cr + cont_h)]
            dest_coords = [(r, dest_c) for r in range(dest_r, dest_r + dest_len)]

        for r, c in head_coords + dest_coords:
            grid[r, c] = marker_color
            expected[r, c] = marker_color

        if steps == 0:
            continue

        expected[cr : cr + cont_h, cc : cc + cont_w] = BG
        for r, c in head_coords:
            expected[r, c] = BG

        nr = cr + dr * steps
        nc = cc + dc * steps

        moved = container.copy()
        for idx in range(initial_orange, initial_orange + steps):
            rr, cc0 = fill_order[idx]
            moved[rr, cc0] = ORANGE
        paste(expected, moved, nr, nc)

        moved_head = [(r + dr * steps, c + dc * steps) for r, c in head_coords]
        for r, c in moved_head:
            expected[r, c] = marker_color

    cases = _inspect_case(grid)
    if len(cases) != n_containers:
        return None
    if len({case["color"] for case in cases}) < 2:
        return None
    if any(case["head_len"] == case["dest_len"] for case in cases):
        return None

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
