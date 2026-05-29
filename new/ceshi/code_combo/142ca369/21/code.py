# Combo variant of 142ca369 (index 21)
# combo_spec: D5/0+D6/0
# Name: 单像素射线源+目标像素终止

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) concrete_form_and_behavior_trace:
#   - 原题源对象是 3 像素 L 形箭头；这里改成“边框上的单像素控制点”。
#   - 方向编码沿用 D5/0：角落控制点朝唯一的 inward diagonal 发射；非角落边框点用“所在边 + 左/右半边
#     或上/下半边”解码 (dr,dc)。
#   - 终止条件沿用 D6/0：网格内部放置单像素目标点，射线若下一步正好走进目标点，就立刻停下，不再继续
#     画到边界。
#   - 具体例子：若控制点在顶边左半区 (0,3)，则方向为 (1,1)。原题会一直往右下走并在反射后最终出界；
#     现在它仍按同样的反射规则传播，但最后一段若路径为 ...(6,9)->(7,10)->(8,11)，且 (8,11) 是目标，
#     那么输出只会保留到该目标为止，不会继续画 (9,12) 等后续位置。
#
# b) complexity_calibration:
#   - 仅把箭头缩成单像素还不够，因为内部单像素墙会和目标像素视觉冲突，solver 也无法区分。
#   - 因此融合版强制“墙只用 3 像素轴对齐线段”，让“边框 singleton = source，内部 singleton = target”
#     成为稳定、可观察的分类规则。
#   - 为避免样例退化成只看角落，生成器偏向非角落控制点；为避免目标规则太隐蔽，样例通常含 2-6 条射线、
#     多个重复目标色的内点。
#
# c) implicit_assumptions_review:
#   - 原 `is_arrow_pts` / `arrow_dir_and_start` 都失效，因为源不再是 3 像素 L 形。
#   - 原 `wall_cells = all non-bg non-arrow` 也失效，因为内部 singleton 现在应解释为目标，而不是墙。
#   - 原 solver 默认允许单像素墙；融合后必须撤销这一前提，否则 target/wall 歧义会破坏可解性。
#   - 新 solver 依赖的前提变为：
#       * 所有源都是边框上的 1 像素连通块；
#       * 所有目标都是内部的 1 像素连通块；
#       * 其他非背景连通块都是墙；
#       * 墙不会触边，也不会与同色控制点 4 连通。
#
# d) placement_legality:
#   - 控制点必须放在边框，且解码后的下一步必须进入网格。
#   - 墙只能是 3 像素水平/竖直线段，且全部在内部，不能与已有源、目标、射线路径重叠。
#   - 目标必须位于最终对角路径上、非边框、与其它目标不 4 邻接，以保证保持 singleton。
#
# e) observability:
#   - 人类从样例里能看到：边框上的孤立彩点会发射对角线；内部有若干重复颜色的小目标点；路径恰好停在
#     目标上而不是飞出画面。
#   - 方向编码不依赖颜色值，而依赖可见位置；目标也不依赖隐式排序，而依赖可见的“内部 singleton”形态。
#
# f) visual_cleanliness:
#   - 控制点、墙段、目标点三类对象在形状和位置上天然分层：边框 singleton、内部 3 像素线段、内部 singleton。
#   - 生成器禁止重叠和过近合并，避免对象混成一个大块导致分类失真。
#
# g) inferable_color_changes:
#   - 颜色继承规则不变：射线起始色来自 source，撞墙后继承墙色；目标色则保持为专用标记色，不参与映射。
#   - 目标色在同一幅图里重复出现，给出“这些内点都是终止标记”的结构线索；墙色从非目标色中选，避免把
#     “目标色”误当成普通反射墙色。


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


def _classify_objects(
    g: np.ndarray,
    bg: int,
) -> tuple[list[tuple[int, tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    H, W = g.shape
    controls: list[tuple[int, tuple[int, int]]] = []
    target_cells: set[tuple[int, int]] = set()
    wall_cells: set[tuple[int, int]] = set()

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if len(pts) == 1:
                pos = pts[0]
                if _is_border(pos[0], pos[1], H, W):
                    controls.append((int(col), pos))
                else:
                    target_cells.add(pos)
            else:
                wall_cells.update(pts)

    controls.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    control_cells = {pos for _col, pos in controls}
    return controls, control_cells, target_cells, wall_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from border control pixels, reflecting off 3-pixel walls,
    and stop when a ray enters an interior singleton target pixel.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    controls, control_cells, target_cells, wall_cells = _classify_objects(g, bg)
    out = g.copy()

    for start_col, cur in controls:
        decoded = _control_dir(H, W, cur[0], cur[1])
        if decoded is None:
            continue
        dr, dc = int(decoded[0]), int(decoded[1])
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in target_cells:
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


def _pick_source(candidates: list[tuple[int, int]], H: int, W: int) -> tuple[int, int]:
    non_corner = [p for p in candidates if not _is_corner(p[0], p[1], H, W)]
    if non_corner and (_rng.random() < 0.85):
        return _rng.choice(non_corner)
    return _rng.choice(candidates)


def _orth_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _adjacent_input_colors(
    cells: list[tuple[int, int]],
    grid: np.ndarray,
    local_input_colors: dict[tuple[int, int], int],
) -> set[int]:
    forbidden: set[int] = set()
    H, W = grid.shape
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            pos = (int(rr), int(cc))
            if pos in local_input_colors:
                forbidden.add(int(local_input_colors[pos]))
            elif 0 <= int(rr) < H and 0 <= int(cc) < W and int(grid[int(rr), int(cc)]) != 0:
                forbidden.add(int(grid[int(rr), int(cc)]))
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the D5/0 + D6/0 combo.

    The generator enforces the classification assumptions used by `main()`:
    - border singleton -> source
    - interior singleton -> target
    - everything else -> wall
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_controls = _rng.randint(2, 6)

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    source_colors = palette[:n_controls]
    target_color = int(palette[n_controls])
    wall_palette = [int(c) for c in palette if int(c) != target_color]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    target_cells: set[tuple[int, int]] = set()

    for ai in range(n_controls):
        color = int(source_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[1, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(160):
            candidates = [p for p in _candidate_sources(H, W, dr, dc) if p not in occupied]
            if not candidates:
                break
            sr, sc = _pick_source(candidates, H, W)

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            local_occ: set[tuple[int, int]] = {(int(sr), int(sc))}
            local_input_colors = {(int(sr), int(sc)): int(color)}
            ok = True

            for _seg in range(n_bounces):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if max_steps < 4:
                    ok = False
                    break
                seg_len = _rng.randint(3, max(3, max_steps - 1))

                for _step in range(seg_len):
                    nr, nc = int(cur_r + cur_dr), int(cur_c + cur_dc)
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in local_occ:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    local_occ.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                options: list[tuple[list[tuple[int, int]], str]] = []

                wr, wc = int(cur_r + cur_dr), int(cur_c)
                w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                if (
                    0 <= wr < H
                    and all(0 <= c < W for _r, c in w3)
                    and all(not _is_border(r, c, H, W) for r, c in w3)
                ):
                    options.append((w3, "dr"))

                wr2, wc2 = int(cur_r), int(cur_c + cur_dc)
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
                chosen_wall = None
                for wpxs, flip in options:
                    if any((int(r), int(c)) in occupied or (int(r), int(c)) in local_occ for r, c in wpxs):
                        continue
                    forbidden = _adjacent_input_colors(wpxs, grid, local_input_colors)
                    color_choices = [p for p in wall_palette if int(p) not in forbidden]
                    if not color_choices:
                        continue
                    chosen_wall = ([(int(r), int(c)) for r, c in wpxs], str(flip), int(_rng.choice(color_choices)))
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip, wcol = chosen_wall
                walls.append((wpxs, int(wcol)))
                local_occ.update(wpxs)
                for r, c in wpxs:
                    local_input_colors[(int(r), int(c))] = int(wcol)

                lr, lc, _old = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = int(wcol)
                if flip == "dr":
                    cur_dr = -int(cur_dr)
                else:
                    cur_dc = -int(cur_dc)

            if not ok:
                continue

            max_steps = min(
                (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                (W - 1 - cur_c) if cur_dc == 1 else cur_c,
            )
            if max_steps < 3:
                continue

            target_steps = _rng.randint(2, max_steps - 1)
            target: tuple[int, int] | None = None
            for step_idx in range(1, target_steps + 1):
                nr, nc = int(cur_r + cur_dr), int(cur_c + cur_dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (nr, nc) in occupied or (nr, nc) in local_occ:
                    ok = False
                    break
                cur_r, cur_c = int(nr), int(nc)
                if step_idx == target_steps:
                    target = (int(nr), int(nc))
                else:
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    local_occ.add((int(nr), int(nc)))
            if not ok or target is None:
                continue

            if _is_border(target[0], target[1], H, W):
                continue
            if any(_orth_adjacent(target, other) for other in target_cells):
                continue
            if len(ray_cells) < 3:
                continue

            new_cells = {(int(sr), int(sc))} | ray_set | {target}
            for wpxs, _wcol in walls:
                new_cells.update((int(r), int(c)) for r, c in wpxs)
            expected_count = 1 + len(ray_set) + 1 + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or (new_cells & occupied):
                continue

            grid[int(sr), int(sc)] = out[int(sr), int(sc)] = int(color)
            for r, c, col in ray_cells:
                out[int(r), int(c)] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[int(r), int(c)] = out[int(r), int(c)] = int(wcol)
            grid[target] = out[target] = int(target_color)

            occupied |= new_cells
            target_cells.add(target)
            placed = True
            break

        if not placed:
            return None

    if len(target_cells) != n_controls:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for combo variant D5/0+D6/0."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
