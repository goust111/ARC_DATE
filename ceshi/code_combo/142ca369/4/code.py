# Combo variant of 142ca369 (index 4)
# combo_spec: D5/0+D18/2
# Name: 单像素源+边缘位置编码方向

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 原题源对象是 3 像素 L 形，缺角编码对角方向，射线从直角像素出发。
#    - 融合后源对象改成边框上的单个彩色像素；solver 不再寻找 2x2 的 L 形，
#      而是寻找“位于最外圈、面积恰好为 1 的单色连通块”。
#    - 方向由边缘位置编码：
#        * 四角: 唯一的 inward diagonal。
#        * 每条边的几何中点单元: orthogonal inward。
#        * 其他非角落边框像素: 向网格内部、并朝最近一侧角落倾斜的 diagonal。
#      例如顶边像素：
#        (0,0)->(1,1), 左半边非中点 -> (1,1), 中点 -> (1,0), 右半边非中点 -> (1,-1), (0,W-1)->(1,-1)。
#    - 小例子：
#        原题中 {(3,4),(4,3),(4,4)} 缺角 (3,3)，从 (4,4) 发射 (1,1)。
#        融合后若改成顶边单像素 (0,4) 且它是顶边中点，则直接从 (0,4) 往 (1,0) 竖直下射；
#        若改成顶边左半的 (0,2)，则往 (1,1) 斜射。差异集中在“如何识别源”和“如何解码方向”。
#
# b) 复杂度校准
#    - 只用四角会太简单，也不利于观察“位置编码”；只用整条边按左右半边二分又会退化为 D5 单变体。
#    - 因此每条边保留 3 类可见档位：左/上倾斜、中点正交、右/下倾斜。
#      人类在 2-4 个样例里就能看到重复：角落发对角，中点发正交，边缘其他位置发倾斜对角。
#    - 正交源只出现在边中点 1-2 个格子上，类别少、可学，但不会淹没原题的射线反射风格。
#
# c) 隐式假设审查
#    - 原 `is_arrow_pts` / `arrow_dir_and_start` 假设源面积为 3 且 bbox 为 2x2；融合后完全失效，删除。
#    - 原 `arrow_cells` 扣除源对象、其余非背景当墙 的分类思想仍可保留，但前提变为：
#        * 源必须是单像素边框连通块；
#        * 墙不能触边，且不能与同色源 4 连通，否则会破坏“单像素源”识别。
#    - 原反射循环只依赖当前方向 `(dr, dc)` 和墙位置。把方向扩展到 `{-1,0,1}^2 \\ {(0,0)}`
#      后仍成立：对角线保留原语义；正交射线只会触发单轴翻转，不再出现“双轴中的另一轴”。
#    - 生成器因此必须保证：
#        * 正交源不安排反弹，否则反向后会沿原路重叠；
#        * 墙始终在内部，避免被 solver 当作边框源误识别。
#
# d) 放置合法性
#    - `occupied` 同时跟踪源、射线路径和墙，禁止重叠。
#    - 墙仅允许放在内部单元；反弹墙离射线路径恰好一个单位，匹配 solver 的侧碰检查。
#    - 候选输入一律经 `main(input) == planned_output` 的 oracle 校验后才接受。
#
# e) 视觉可观察性
#    - 单像素源是唯一贴边的非背景对象，墙全部缩在内部，肉眼很容易先分出“谁是源谁是墙”。
#    - 样例会混入中点正交源和非中点斜向源，使“边缘位置编码方向”在少量样例中即可归纳。
#
# f) 可视化整洁度
#    - 墙不触边，避免与源混杂。
#    - 正交源禁止反弹，防止长回折把同一直线涂成噪声。
#    - 每条射线要求至少 3 个新涂色单元，避免边框处只有一个短刺导致不显眼。
#
# g) 变化可推断性
#    - 颜色规则完全继承原题：源色是初始射线色，碰墙后继承墙色。
#    - 新规则不引入任意颜色映射，推断焦点只落在“源形状 + 边缘位置 -> 方向”。


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H) - 1) or int(c) in (0, int(W) - 1)


def _is_corner(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H) - 1) and int(c) in (0, int(W) - 1)


def _mid_indices(length: int) -> set[int]:
    return {int((length - 1) // 2), int(length // 2)}


def _control_dir(H: int, W: int, r: int, c: int) -> tuple[int, int] | None:
    r = int(r)
    c = int(c)
    H = int(H)
    W = int(W)
    if not _is_border(r, c, H, W):
        return None

    if (r, c) == (0, 0):
        return 1, 1
    if (r, c) == (0, W - 1):
        return 1, -1
    if (r, c) == (H - 1, 0):
        return -1, 1
    if (r, c) == (H - 1, W - 1):
        return -1, -1

    mid_cols = _mid_indices(W)
    mid_rows = _mid_indices(H)

    if r == 0:
        if c in mid_cols:
            return 1, 0
        return 1, 1 if c < min(mid_cols) else -1
    if r == H - 1:
        if c in mid_cols:
            return -1, 0
        return -1, 1 if c < min(mid_cols) else -1
    if c == 0:
        if r in mid_rows:
            return 0, 1
        return (1 if r < min(mid_rows) else -1), 1
    if r in mid_rows:
        return 0, -1
    return (1 if r < min(mid_rows) else -1), -1


def _find_control_points(g: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int]]]:
    H, W = g.shape
    controls: list[tuple[int, tuple[int, int]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if len(pts) == 1 and _is_border(pts[0][0], pts[0][1], H, W):
                controls.append((int(col), pts[0]))
    return controls


def _max_steps(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    row_steps = float("inf") if int(dr) == 0 else ((H - 1 - r) if int(dr) == 1 else r)
    col_steps = float("inf") if int(dc) == 0 else ((W - 1 - c) if int(dc) == 1 else c)
    return int(min(row_steps, col_steps))


def _source_kind(H: int, W: int, pos: tuple[int, int]) -> str:
    r, c = pos
    if _is_corner(r, c, H, W):
        return "corner"
    dr, dc = _control_dir(H, W, r, c)
    if dr == 0 or dc == 0:
        return "mid"
    return "edge-diag"


def _available_sources(H: int, W: int, occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(W):
        if (0, c) not in occupied:
            pts.append((0, c))
        if (H - 1, c) not in occupied:
            pts.append((H - 1, c))
    for r in range(1, H - 1):
        if (r, 0) not in occupied:
            pts.append((r, 0))
        if (r, W - 1) not in occupied:
            pts.append((r, W - 1))
    return pts


def _pick_source(
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    force_mid: bool,
) -> tuple[int, int] | None:
    candidates = _available_sources(H, W, occupied)
    if not candidates:
        return None

    mids = [p for p in candidates if _source_kind(H, W, p) == "mid"]
    edge_diags = [p for p in candidates if _source_kind(H, W, p) == "edge-diag"]
    corners = [p for p in candidates if _source_kind(H, W, p) == "corner"]

    if force_mid and mids:
        return _rng.choice(mids)

    roll = _rng.random()
    if mids and roll < 0.35:
        return _rng.choice(mids)
    if edge_diags and roll < 0.9:
        return _rng.choice(edge_diags)
    pool = corners or edge_diags or mids
    if not pool:
        return None
    return _rng.choice(pool)


def _adjacent_source_colors(
    wall_pixels: list[tuple[int, int]],
    source_colors: dict[tuple[int, int], int],
) -> set[int]:
    forbidden: set[int] = set()
    for r, c in wall_pixels:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (int(rr), int(cc)) in source_colors:
                forbidden.add(int(source_colors[(int(rr), int(cc))]))
    return forbidden


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace rays from border control pixels, reflecting off colored walls.

    Variant rule:
    - Every source is a single colored pixel on the outer border.
    - Corner sources fire inward diagonally.
    - Border midpoint sources fire orthogonally inward.
    - Other border sources fire diagonally inward, tilted toward the nearest corner side.
    - Rays keep the original color-inheritance and reflection mechanics.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    controls = _find_control_points(g, bg)
    control_cells = {pos for _col, pos in controls}
    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in control_cells
    }

    out = g.copy()

    for start_col, cur in controls:
        decoded = _control_dir(H, W, cur[0], cur[1])
        if decoded is None:
            continue
        dr, dc = int(decoded[0]), int(decoded[1])
        cur_col = int(start_col)
        seen = set()

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
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in control_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) != 0 and (nr + int(dr), nc) in wall_cells:
                wcol = int(g[nr + int(dr), nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) != 0 and (nr, nc + int(dc)) in wall_cells:
                wcol = int(g[nr, nc + int(dc)])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the combo variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_controls = _rng.randint(2, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    source_colors = palette[:n_controls]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    committed_sources: dict[tuple[int, int], int] = {}

    ensure_mid = _rng.random() < 0.75

    for ai in range(n_controls):
        color = int(source_colors[ai])
        placed = False

        for _ in range(120):
            src = _pick_source(H, W, occupied, force_mid=(ensure_mid and ai == 0))
            if src is None:
                break
            sr, sc = int(src[0]), int(src[1])
            dr, dc = _control_dir(H, W, sr, sc)
            if dr is None or dc is None:
                continue
            dr = int(dr)
            dc = int(dc)

            max0 = _max_steps(sr, sc, dr, dc, H, W)
            if max0 < 3:
                continue

            is_orth = (dr == 0) ^ (dc == 0)
            n_bounces = 0 if is_orth else _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc, cur_col = int(dr), int(dc), int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            local_occ: set[tuple[int, int]] = {(int(sr), int(sc))}
            local_sources = dict(committed_sources)
            local_sources[(int(sr), int(sc))] = int(color)
            ok = True
            bounces_left = int(n_bounces)

            for seg_idx in range(n_bounces + 1):
                max_steps = _max_steps(cur_r, cur_c, cur_dr, cur_dc, H, W)
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        bounces_left = 0
                        seg_len = max_steps
                    else:
                        seg_len = _rng.randint(3, max_steps - 1)
                else:
                    seg_len = max_steps

                for _step in range(seg_len):
                    nr, nc = int(cur_r + cur_dr), int(cur_c + cur_dc)
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied or (nr, nc) in local_occ:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    local_occ.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                if seg_idx >= n_bounces or bounces_left <= 0:
                    continue
                bounces_left -= 1

                options: list[tuple[list[tuple[int, int]], str]] = []
                if int(cur_dr) != 0:
                    wr, wc = int(cur_r + cur_dr), int(cur_c)
                    one = [(int(wr), int(wc))]
                    if 0 <= wr < H and not _is_border(one[0][0], one[0][1], H, W):
                        options.append((one, "dr"))
                    w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                    if (
                        0 <= wr < H
                        and all(0 <= c < W for _r, c in w3)
                        and all(not _is_border(r, c, H, W) for r, c in w3)
                    ):
                        options.append((w3, "dr"))
                if int(cur_dc) != 0:
                    wr2, wc2 = int(cur_r), int(cur_c + cur_dc)
                    one = [(int(wr2), int(wc2))]
                    if 0 <= wc2 < W and not _is_border(one[0][0], one[0][1], H, W):
                        options.append((one, "dc"))
                    w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                    if (
                        0 <= wc2 < W
                        and all(0 <= r < H for r, _c in w3v)
                        and all(not _is_border(r, c, H, W) for r, c in w3v)
                    ):
                        options.append((w3v, "dc"))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda item: len(item[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip in options:
                    wpxs_list = [(int(r), int(c)) for r, c in wpxs]
                    if any((int(r), int(c)) in occupied or (int(r), int(c)) in local_occ for r, c in wpxs_list):
                        continue
                    forbidden = _adjacent_source_colors(wpxs_list, local_sources)
                    color_choices = [p for p in palette if int(p) not in forbidden]
                    if not color_choices:
                        continue
                    chosen_wall = (wpxs_list, str(flip), int(_rng.choice(color_choices)))
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip, wcol = chosen_wall
                walls.append((wpxs, int(wcol)))
                local_occ.update((int(r), int(c)) for r, c in wpxs)
                lr, lc, _old = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = int(wcol)
                if flip == "dr":
                    cur_dr = -int(cur_dr)
                else:
                    cur_dc = -int(cur_dc)

            if not ok or len(ray_cells) < 3:
                continue

            new_cells = {(int(sr), int(sc))}
            new_cells.update((int(r), int(c)) for r, c, _col in ray_cells)
            for wpxs, _wcol in walls:
                new_cells.update((int(r), int(c)) for r, c in wpxs)

            expected_count = 1 + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            grid[int(sr), int(sc)] = out[int(sr), int(sc)] = int(color)
            committed_sources[(int(sr), int(sc))] = int(color)
            for r, c, col in ray_cells:
                out[int(r), int(c)] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[int(r), int(c)] = out[int(r), int(c)] = int(wcol)
            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if ensure_mid:
        has_mid = any(_source_kind(H, W, pos) == "mid" for pos in committed_sources)
        if not has_mid:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the combo variant."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
