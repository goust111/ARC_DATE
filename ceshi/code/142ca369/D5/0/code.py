# Variant of 142ca369
# Dimension: D5 — 射线源对象形状 (src)
# Change: L形三角箭头——2x2外接矩形中缺少一个角的3像素L形，缺失角的方向决定射线发射的对角线方向... → 单像素控制点+方向编码

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the D5 variant
# ----------------------------------
# concrete_form:
#   - 原题的 3 像素 L 形箭头全部删除，改成“贴着外边框的单像素控制点”。
#   - 控制点必须是单色 4 连通块，面积恰好为 1，且位于最外一圈边框上。
#   - 方向不再由缺失角显式给出，而由控制点在边框上的位置编码：
#       * 四个角仍然发射唯一的 inward diagonal：
#           (0,0)->(1,1), (0,W-1)->(1,-1), (H-1,0)->(-1,1), (H-1,W-1)->(-1,-1)
#       * 非角落顶边/底边：所在列落在左半边或右半边，决定 dc 的符号
#       * 非角落左边/右边：所在行落在上半边或下半边，决定 dr 的符号
#     这样仍然只发射对角线射线，保留原题的反射机制，不引入新的水平/垂直射线维度。
#   - 墙仍然是单像素或 3 像素轴对齐线段；射线仍然对角线前进、侧碰翻单轴、正碰翻双轴、碰撞后继承墙颜色。
#
# behavior_trace:
#   - 原题：若一个 L 形位于 {(3,4),(4,3),(4,4)}，缺失角是 (3,3)，则从直角点 (4,4) 往 (1,1) 发射。
#   - 变体：若把它改成顶边上的单像素控制点 (0,2)，且该点位于顶边左半边，则方向解码为 (1,1)。
#     射线从 (0,2) 的下一步 (1,3) 开始进入网格；之后的反射、染色、终止条件与原题一致。
#   - 差异集中在“怎么从输入里识别源、怎么确定起点与方向”这一步，后续物理过程保持不变。
#
# complexity_calibration:
#   - 如果只允许四个角做源，规则会过于接近“角落发射器”，抽象性不够，也会减少布局自由度。
#   - 因此允许整条边上的控制点，只用“边 + 半边位置”编码方向；同一方向会在不同边上重复出现，
#     人类可从 2-4 个样例归纳“位置编码”，同时不会引入太多离散档位。
#   - 生成器偏向非角落控制点，避免退化成“看起来只是把箭头挪到角上”。
#
# implicit_assumptions_review:
#   - 原 solver 中 `is_arrow_pts` 假设源对象面积为 3 且 bbox 为 2x2；该假设失效，必须删除。
#   - 原 `arrow_dir_and_start` 假设可以由缺失角恢复方向、由 2 邻接点恢复起点；现在改为：
#       * 起点 = 控制点自身
#       * 方向 = `_control_dir(...)`
#   - 原 `arrow_cells` 用于从非背景中扣除“源对象”，剩余当墙。现在这个分类仍成立，但前提改为：
#       * 所有源都是边框上的单像素单色连通块
#       * 非源的非背景都当墙
#   - 这意味着生成器必须保证墙不会占据边框、不会与同色控制点 4 连通，否则 solver 会把源识别错。
#
# placement_legality:
#   - 控制点必须放在边框，且下一步沿解码方向进入网格。
#   - 墙禁止触边，这样边框非背景像素全部都可被解释为控制点候选。
#   - 生成器继续使用 output-first：先规划每条射线与墙，再回填输入，并用 solver 校验。
#
# observability_and_cleanliness:
#   - 样例中会混合顶边/底边/左边/右边的控制点，且大多数不在角落，让“位置编码”肉眼可见。
#   - 墙保持与原题相同的稀疏线段风格，控制点是唯一贴边单像素对象，视觉上与墙清楚区分。
#
# color_inference:
#   - 颜色规则不改：射线初始颜色来自控制点，反射后继承墙颜色。
#   - 颜色种类仍限制在 ARC 1-9 palette 内；控制点颜色唯一，墙颜色可重复，结构线索来自几何交互而非任意映射。


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H) - 1) or int(c) in (0, int(W) - 1)


def _is_corner(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H) - 1) and int(c) in (0, int(W) - 1)


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
    if r == 0:
        return 1, 1 if c < (W / 2.0) else -1
    if r == H - 1:
        return -1, 1 if c < (W / 2.0) else -1
    if c == 0:
        return (1 if r < (H / 2.0) else -1), 1
    return (1 if r < (H / 2.0) else -1), -1


def _find_control_points(g: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int]]]:
    H, W = g.shape
    controls: list[tuple[int, tuple[int, int]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if len(pts) == 1 and _is_border(pts[0][0], pts[0][1], H, W):
                controls.append((int(col), pts[0]))
    return controls


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from border control pixels, reflecting off colored walls.

    Input structure:
    - bg grid with 1-pixel control points on the outer border and interior colored wall pixels/segments.

    Rule:
    - Each control point fires one diagonal ray; the direction is decoded from which edge/half contains the pixel.
    - Rays paint bg cells in their current color.
    - A head-on wall hit reverses both axes; a side-touch wall hit flips the touched axis only.
    - On any bounce the ray adopts the wall's color and recolors the contact cell.
    - Ray stops when it exits the grid or reaches another control point.
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

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _candidate_sources(H: int, W: int, dr: int, dc: int) -> list[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(W):
        if _control_dir(H, W, 0, c) == (int(dr), int(dc)):
            pts.add((0, int(c)))
        if _control_dir(H, W, H - 1, c) == (int(dr), int(dc)):
            pts.add((H - 1, int(c)))
    for r in range(H):
        if _control_dir(H, W, r, 0) == (int(dr), int(dc)):
            pts.add((int(r), 0))
        if _control_dir(H, W, r, W - 1) == (int(dr), int(dc)):
            pts.add((int(r), W - 1))
    return sorted(pts)


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


def _pick_source(candidates: list[tuple[int, int]], H: int, W: int) -> tuple[int, int]:
    non_corner = [p for p in candidates if not _is_corner(p[0], p[1], H, W)]
    corner = [p for p in candidates if p not in non_corner]
    if non_corner and (_rng.random() < 0.8 or not corner):
        return _rng.choice(non_corner)
    return _rng.choice(corner if corner else candidates)


def _gen_attempt() -> np.ndarray | None:
    """
    Single attempt to generate a valid input for the D5 control-pixel variant.

    Output-first, same as the original task:
    - choose control point positions implied by desired diagonal directions
    - plan each ray with 0-4 bounces
    - place side-touch walls causing those bounces
    - reject layouts that violate the new source assumptions or the inherited reflection rules
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_controls = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    source_colors = palette[:n_controls]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    committed_sources: dict[tuple[int, int], int] = {}

    for ai in range(n_controls):
        color = int(source_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(90):
            candidates = [p for p in _candidate_sources(H, W, dr, dc) if p not in occupied]
            if not candidates:
                break
            sr, sc = _pick_source(candidates, H, W)

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc, cur_col = int(dr), int(dc), int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            local_occ: set[tuple[int, int]] = {(int(sr), int(sc))}
            ok = True
            bounces_left = int(n_bounces)
            local_sources = dict(committed_sources)
            local_sources[(int(sr), int(sc))] = int(color)

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        bounces_left = 0
                        seg_len = max_steps
                    else:
                        seg_len = _rng.randint(3, max(3, max_steps - 1))
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

                wr, wc = int(cur_r + cur_dr), int(cur_c)
                if 0 <= wr < H:
                    one = [(int(wr), int(wc))]
                    if not _is_border(one[0][0], one[0][1], H, W):
                        options.append((one, "dr"))
                    w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _r, c in w3) and all(not _is_border(r, c, H, W) for r, c in w3):
                        options.append((w3, "dr"))

                wr2, wc2 = int(cur_r), int(cur_c + cur_dc)
                if 0 <= wc2 < W:
                    one = [(int(wr2), int(wc2))]
                    if not _is_border(one[0][0], one[0][1], H, W):
                        options.append((one, "dc"))
                    w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _c in w3v) and all(not _is_border(r, c, H, W) for r, c in w3v):
                        options.append((w3v, "dc"))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda o: len(o[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip in options:
                    if any((int(r), int(c)) in occupied or (int(r), int(c)) in local_occ for r, c in wpxs):
                        continue
                    wpxs_list = [(int(r), int(c)) for r, c in wpxs]
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
                local_occ.update(wpxs)

                if ray_cells:
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

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for task 142ca369, D5 control-pixel variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
