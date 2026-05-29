# Variant of 142ca369
# Dimension: D16 — 颜色使用策略 (meta)
# Change: 2-7种非背景色，箭头和墙壁各自单色，射线颜色随反射继承墙壁色... → 箭头颜色编码反射规则
#
# Design analysis
# a) 具体化与行为追踪
# - 保留原题的像素对象: 3 像素 L 形箭头、1/3 像素直墙、45 度对角线射线、撞墙后继承墙色。
# - 只改 side-touch 反射的“哪一轴会翻转”:
#   blue=1 的箭头只响应竖直墙, 只翻转 dc;
#   red=2 的箭头只响应水平墙, 只翻转 dr.
# - 对角线正撞墙的 corner reflection 仍保留原题逻辑, 统一反转两轴; 生成器不主动依赖它。
# - 一个关键局部例子:
#   原题: 射线走到 (r,c) 后, 若 (r+dr,c) 与 (r,c+dc) 都是墙, 就会同时翻 dr 和 dc.
#   变体: 若箭头是 red, 只看 (r+dr,c), 忽略 (r,c+dc); 若箭头是 blue, 则反过来。
#   因此同样的“双邻墙”视觉结构, 不同箭头颜色会产生不同折返方向和不同继承颜色。
#
# b) 复杂度校准
# - 只使用两种箭头色 {1,2}, 映射到两类规则 {vertical-only, horizontal-only}。
# - 这是刻意收缩自由度: 2 个颜色类别足以形成可学习映射, 且在 2-4 个样例里可重复支撑。
# - 墙颜色仍来自独立调色板 {3..9} 的一个子集, 保留颜色继承这一原题机制, 但避免与箭头色混淆。
#
# c) 原 solver 中与变体相关的隐式假设审查
# - 原代码把任意 side-touch 墙都当作有效反射源:
#     if (nr+dr,nc) in wall_cells: dr = -dr
#     if (nr,nc+dc) in wall_cells: dc = -dc
#   这依赖“所有箭头共享同一反射规则”。变体下该前提失效, 必须先由箭头颜色选定允许翻转的轴。
# - 原代码在有效反射时把当前射线色改成墙色, 并把反射接触格重涂为墙色。
#   这一点在变体中仍成立, 但只能对“该箭头允许的那类反射”执行; 对被忽略的邻墙不能继承颜色。
# - 原代码逐色找 3 像素 L 形连通块作为箭头。变体仍可用, 但生成器必须避免墙使用箭头颜色,
#   否则会让“颜色既是对象颜色又是规则参数”的解析变得含混。
#
# d) 放置合法性
# - 生成器沿用 output-first: 先规划每个箭头的真实路径, 再放箭头和墙, 最后用新 solver 校验。
# - 因为 red 只翻 dr、blue 只翻 dc, 每条轨迹只在一个轴上来回摆动, 另一个轴单调前进。
#   这使得未来路径不会回踩已放的双邻墙, 合法性更易保证。
# - 每个墙段都与所属 bounce cell 邻接, 且不与箭头、射线或先前墙段重叠。
#
# e) 视觉可观察性
# - 每个样例至少同时包含一支 red 箭头和一支 blue 箭头。
# - 每支箭头至少有 1 次 bounce, 且第一次 bounce 强制放置“双邻墙”:
#   即 bounce cell 的 row-side 与 col-side 各放一段墙, 但只有一段会真的驱动反射。
# - 这样解题者能直接比较:
#   红箭头遇到同样的双邻墙时沿“水平反射”规则折返,
#   蓝箭头则沿“垂直反射”规则折返。
#
# f) 可视化整洁度
# - 墙色与箭头色分离, 箭头只用两色, 视觉角色清楚。
# - 双邻墙只贴在 bounce cell 周围, 不在同一区域叠太多层对象。
# - 箭头数量控制在 2-6, 网格仍为 18-20, 防止拥挤和歧义。
#
# g) 颜色变化可推断性
# - 箭头色只承担“选规则”这一种语义, 墙色只承担“被继承的轨迹颜色”这一种语义。
# - 因为墙色来自独立集合, 看到输出时可以把“方向变化”归因于箭头色, 把“射线换色”归因于墙色。
#
# h) 参考任务说明
# - 提示中引用了 e87109e9 作为 symbol_instructions 灵感来源。
# - 本工作区存在 e87109e9 的 JSON 样例, 但不存在可读取的 solver `.py`, 所以无法直接复用其代码实现;
#   这里只吸收它“颜色本身携带转向/反射语义”的思想。
#
# i) main() 的新假设审查
# - main() 现在假设“真正的箭头颜色只会是 1 或 2”。生成器通过固定箭头调色板保证这一点。
# - main() 假设墙不会与箭头同色从而造成角色歧义。生成器通过墙色只取 {3..9} 保证这一点。
# - main() 假设至少存在可识别的 3 像素 L 形箭头, 且双邻墙不会压到未来轨迹上。生成器在构造阶段显式检查。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

ARROW_RULES = {
    1: "dc",  # blue arrow: only vertical-wall side touches matter
    2: "dr",  # red arrow: only horizontal-wall side touches matter
}
ARROW_COLORS = tuple(sorted(ARROW_RULES))
WALL_COLOR_POOL = [3, 4, 5, 6, 7, 8, 9]


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


def _rule_axis_from_color(color: int) -> str:
    return ARROW_RULES.get(int(color), "dr")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows.

    Variant rule:
    - blue arrows (1) only react to side-touch walls on the column axis, flipping dc.
    - red arrows (2) only react to side-touch walls on the row axis, flipping dr.
    - A valid side-touch bounce still recolors the contact ray cell with the wall color.
    - Corner hits into a wall cell keep the original task's two-axis reflection.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        allowed_axis = _rule_axis_from_color(start_col)
        (dr, dc), cur = info
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

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if allowed_axis == "dr":
                wall_pos = (nr + int(dr), nc)
                if wall_pos in wall_cells:
                    wcol = int(g[wall_pos[0], wall_pos[1]])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dr = -int(dr)
            else:
                wall_pos = (nr, nc + int(dc))
                if wall_pos in wall_cells:
                    wcol = int(g[wall_pos[0], wall_pos[1]])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dc = -int(dc)

    return out


def _wall_options(H: int, W: int, r: int, c: int, dr: int, dc: int, axis: str) -> list[list[tuple[int, int]]]:
    if axis == "dr":
        wr, wc = r + dr, c
        if not (0 <= wr < H):
            return []
        opts = [[(wr, wc)]]
        if 0 <= wc - 1 and wc + 1 < W:
            opts.append([(wr, wc - 1), (wr, wc), (wr, wc + 1)])
        return opts

    wr, wc = r, c + dc
    if not (0 <= wc < W):
        return []
    opts = [[(wr, wc)]]
    if 0 <= wr - 1 and wr + 1 < H:
        opts.append([(wr - 1, wc), (wr, wc), (wr + 1, wc)])
    return opts


def _choose_wall_segment(
    H: int,
    W: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    axis: str,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    options = _wall_options(H, W, r, c, dr, dc, axis)
    if not options:
        return None
    _rng.shuffle(options)
    if _rng.random() < 0.6:
        options.sort(key=len, reverse=True)
    else:
        options.sort(key=len)
    for wpxs in options:
        if not any((rr, cc) in forbidden for rr, cc in wpxs):
            return wpxs
    return None


def _pick_wall_color(palette: list[int], avoid: int | None = None) -> int:
    candidates = [c for c in palette if c != avoid]
    if candidates:
        return int(_rng.choice(candidates))
    return int(_rng.choice(palette))


def _mono_steps_left(H: int, W: int, r: int, c: int, dr: int, dc: int, rule_axis: str) -> int:
    if rule_axis == "dr":
        return int((W - 1 - c) if dc == 1 else c)
    return int((H - 1 - r) if dr == 1 else r)


def _build_arrow(
    H: int,
    W: int,
    color: int,
    wall_palette: list[int],
    occupied: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    rule_axis = _rule_axis_from_color(color)
    other_axis = "dc" if rule_axis == "dr" else "dr"
    dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
    n_bounces = _rng.choices([1, 2, 3], weights=[2, 2, 1])[0]

    for _ in range(100):
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

        mr, mc = sr - dr, sc - dc
        arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = int(color)

        ray_cells: list[tuple[int, int, int]] = []
        ray_set: set[tuple[int, int]] = set()
        walls: list[tuple[list[tuple[int, int]], int]] = []
        wall_set: set[tuple[int, int]] = set()
        dual_bounces = 0
        ok = True

        for seg_idx in range(n_bounces + 1):
            max_steps = min(
                (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                (W - 1 - cur_c) if cur_dc == 1 else cur_c,
            )
            if seg_idx < n_bounces:
                future_bounces = n_bounces - seg_idx - 1
                mono_left = _mono_steps_left(H, W, cur_r, cur_c, cur_dr, cur_dc, rule_axis)
                reserve = 2 * future_bounces + 1
                seg_hi = min(max_steps - 1, mono_left - reserve)
                if seg_hi < 2:
                    ok = False
                    break
                seg_len = _rng.randint(2, seg_hi)
            else:
                seg_len = max_steps
                if seg_len < 1:
                    ok = False
                    break

            for _step in range(seg_len):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (nr, nc) in occupied or (nr, nc) in arrow_set or (nr, nc) in ray_set or (nr, nc) in wall_set:
                    ok = False
                    break
                ray_cells.append((nr, nc, cur_col))
                ray_set.add((nr, nc))
                cur_r, cur_c = nr, nc
            if not ok:
                break

            if seg_idx >= n_bounces:
                continue

            forbidden = occupied | arrow_set | ray_set | wall_set
            active_wall = _choose_wall_segment(H, W, cur_r, cur_c, cur_dr, cur_dc, rule_axis, forbidden)
            if active_wall is None:
                ok = False
                break
            active_col = _pick_wall_color(wall_palette, avoid=cur_col)
            walls.append((active_wall, active_col))
            wall_set.update(active_wall)

            need_dual = seg_idx == 0 or (_rng.random() < 0.5)
            if need_dual:
                forbidden = occupied | arrow_set | ray_set | wall_set
                decoy_wall = _choose_wall_segment(H, W, cur_r, cur_c, cur_dr, cur_dc, other_axis, forbidden)
                if decoy_wall is None:
                    ok = False
                    break
                decoy_col = _pick_wall_color(wall_palette, avoid=active_col)
                walls.append((decoy_wall, decoy_col))
                wall_set.update(decoy_wall)
                dual_bounces += 1

            lr, lc, _old = ray_cells[-1]
            ray_cells[-1] = (lr, lc, active_col)
            cur_col = active_col
            if rule_axis == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or dual_bounces < 1 or len(ray_cells) < 3:
            continue

        grid_patch = np.zeros((H, W), dtype=int)
        out_patch = np.zeros((H, W), dtype=int)
        for r, c in arrow_pts:
            grid_patch[r, c] = int(color)
            out_patch[r, c] = int(color)
        for r, c, col in ray_cells:
            out_patch[r, c] = int(col)
        for wpxs, wcol in walls:
            for r, c in wpxs:
                if grid_patch[r, c] != 0 or out_patch[r, c] != 0:
                    ok = False
                    break
                grid_patch[r, c] = int(wcol)
                out_patch[r, c] = int(wcol)
            if not ok:
                break
        if not ok:
            continue

        new_cells = arrow_set | ray_set | wall_set
        expected = len(arrow_set) + len(ray_set) + sum(len(w) for w, _ in walls)
        if len(new_cells) != expected or new_cells & occupied:
            continue

        stats = {
            "bounces": int(n_bounces),
            "dual_bounces": int(dual_bounces),
            "rule_axis_dr": int(rule_axis == "dr"),
            "rule_axis_dc": int(rule_axis == "dc"),
        }
        return grid_patch, out_patch, stats

    return None


def _gen_attempt(return_stats: bool = False):
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 6)
    arrow_colors = [1, 2] + [_rng.choice(ARROW_COLORS) for _ in range(n_arrows - 2)]
    _rng.shuffle(arrow_colors)

    n_wall_colors = _rng.randint(2, 5)
    wall_palette = _rng.sample(WALL_COLOR_POOL, n_wall_colors)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    total_dual_bounces = 0

    for color in arrow_colors:
        built = _build_arrow(H, W, int(color), wall_palette, occupied)
        if built is None:
            return None
        grid_patch, out_patch, stats = built
        patch_cells = set(map(tuple, np.argwhere(grid_patch != 0))) | set(map(tuple, np.argwhere(out_patch != 0)))
        if patch_cells & occupied:
            return None
        grid[grid_patch != 0] = grid_patch[grid_patch != 0]
        out[out_patch != 0] = out_patch[out_patch != 0]
        occupied |= patch_cells
        total_dual_bounces += int(stats["dual_bounces"])

    if total_dual_bounces < 2:
        return None

    solved = main(grid)
    if not np.array_equal(solved, out):
        return None

    if return_stats:
        stats = {
            "shape": (int(H), int(W)),
            "arrows": int(n_arrows),
            "dual_bounces": int(total_dual_bounces),
            "wall_colors": int(n_wall_colors),
        }
        return grid, stats
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt(return_stats=False)
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
