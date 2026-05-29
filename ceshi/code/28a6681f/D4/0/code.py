# Variant of 28a6681f
# Dimension: D4 — 水像素的颜色 (trajectory)
# Change: 所有水像素统一为蓝色（color 1），流动过程中颜色不变... → 水在流经不同颜色结构时变色

from __future__ import annotations

import random

import numpy as np


# Design analysis for this D4 variant
# a) 具体化与行为追踪:
#    原题里输入水都用蓝色 1 表示, 求解时也一直把所有水当成同一种像素。这里保留同样的重力/斜滑/横滑几何,
#    但把“水粒子颜色”从常量改成状态量。每个粒子初始为 1, 每次移动到新格后, 如果它正贴着唯一颜色的盆壁,
#    就继承该壁色; 若左右都不贴壁, 但正踩在唯一颜色的支撑面上, 就继承该支撑色。一个具体小例子:
#    原逻辑中, 右边蓝水沿着灰色立壁下滑进入红色盆地时, 最终仍全部是蓝色。新逻辑中, 它先在灰壁旁下落而变灰,
#    溢流越过分隔壁后又在红壁/红台阶旁继续移动, 最终一部分保持灰色, 一部分改成红色, 形成条纹。
# b) 复杂度校准:
#    如果只让水接触一种结构色, 输出会退化成“蓝水换成某个单色”, D4 不明显。生成器因此只保留那些:
#    1) 输出中至少出现两种水色; 2) 新输出和原单色 solver 输出不同 的样本。颜色池仍限制在 2-3 种结构色,
#    避免颜色种类过多导致归纳困难。
# c) 原 main() 的隐式假设审查:
#    - `water = 1` 被同时当作“输入里的水标签”和“模拟中的唯一水状态”; 变体后前者仍成立, 后者失效。
#    - `g == water` 用来枚举所有水像素; 变体后输出水可能等于墙色, 不能再靠颜色反查动态水。
#    - `g[r, c] = water` 在回填时默认不会覆盖出错; 变体后这会把正确的染色结果强行改回 1。
#    - `check_output` 通过 `g == water` 找终态水, 也会漏掉已染色水。解决办法是把静态结构与动态水粒子分开存。
# d) 放置合法性:
#    生成器只从原题官方样例的输入模板出发, 做水平镜像和结构重着色, 因而保留了原题已经验证过的合法几何:
#    不越界, 不重叠, 输入水全部有支撑, 且每个水像素不会同时左右两侧都是背景。
# e) 视觉可观察性:
#    新样本仍是“蓝水 + 彩色盆地”, 但输出里盆地水会出现至少两种结构色。因为镜像和重着色都保持了 basin/
#    overflow 结构, 解题者能看到“哪一段路径贴着哪种墙, 最后哪批水就变成哪种颜色”。
# f) 可视化整洁度:
#    不额外叠加标记。结构仍是底部盆地和可选上方导流件, 变化只落在终态水色上, 不会让画面杂乱。
# g) 颜色变化可推断性:
#    只使用 2-3 个结构色, 且同一种结构色会在同一网格里多次出现, 给“接触该色结构 → 水变该色”提供重复证据。
# h) 新 main() 的输入前提:
#    - 输入里动态水仍全部是 1, 结构色不使用 1。generate_input() 通过重映射官方模板保证这一点。
#    - 染色只来自静态结构, 不是来自其他水粒子。模板生成不会预置已染色水, 从而避免输入阶段的语义歧义。

BG = 0
INPUT_WATER = 1


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _validate_input(g: np.ndarray) -> None:
    H, W = g.shape
    for r, c in np.argwhere(g == int(INPUT_WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, INPUT_WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _apply_contact_color(
    r: int,
    c: int,
    cur_color: int,
    solid: np.ndarray,
    *,
    colorize: bool,
) -> tuple[int, bool]:
    if not colorize:
        return int(cur_color), False

    H, W = solid.shape
    side_colors: set[int] = set()
    if c - 1 >= 0 and int(solid[r, c - 1]) != int(BG):
        side_colors.add(int(solid[r, c - 1]))
    if c + 1 < W and int(solid[r, c + 1]) != int(BG):
        side_colors.add(int(solid[r, c + 1]))

    if len(side_colors) == 1:
        new_color = next(iter(side_colors))
    elif len(side_colors) >= 2:
        return int(cur_color), False
    elif r + 1 < H and int(solid[r + 1, c]) != int(BG):
        new_color = int(solid[r + 1, c])
    else:
        return int(cur_color), False

    return int(new_color), bool(int(new_color) != int(cur_color))


def _simulate(
    grid: np.ndarray,
    *,
    check_input: bool,
    check_output: bool,
    colorize: bool,
    return_particles: bool = False,
):
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        _validate_input(g)

    solid = np.where((g != int(BG)) & (g != int(INPUT_WATER)), g, int(BG))
    particles = [
        {"r": int(r), "c": int(c), "color": int(INPUT_WATER), "dye_events": 0}
        for r, c in np.argwhere(g == int(INPUT_WATER))
    ]
    occupancy = {(p["r"], p["c"]): idx for idx, p in enumerate(particles)}

    def is_empty(r: int, c: int) -> bool:
        return (
            0 <= int(r) < H
            and 0 <= int(c) < W
            and int(solid[int(r), int(c)]) == int(BG)
            and (int(r), int(c)) not in occupancy
        )

    def settle(idx: int) -> tuple[int, int, int, int]:
        rr = int(particles[idx]["r"])
        cc = int(particles[idx]["c"])
        cur_color = int(particles[idx]["color"])
        dye_events = 0

        for _ in range(H * W):
            if rr + 1 < H and is_empty(rr + 1, cc):
                rr += 1
                cur_color, changed = _apply_contact_color(rr, cc, cur_color, solid, colorize=colorize)
                dye_events += int(changed)
                continue

            opts = []
            if rr + 1 < H and cc - 1 >= 0 and is_empty(rr, cc - 1) and is_empty(rr + 1, cc - 1):
                opts.append((int(rr + 1), int(cc - 1)))
            if rr + 1 < H and cc + 1 < W and is_empty(rr, cc + 1) and is_empty(rr + 1, cc + 1):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                cur_color, changed = _apply_contact_color(rr, cc, cur_color, solid, colorize=colorize)
                dye_events += int(changed)
                continue

            if rr + 1 < H and not is_empty(rr + 1, cc):
                L = int(cc)
                while L - 1 >= 0 and is_empty(rr, L - 1):
                    L -= 1
                R = int(cc)
                while R + 1 < W and is_empty(rr, R + 1):
                    R += 1

                drops = [int(x) for x in range(int(L), int(R) + 1) if is_empty(rr + 1, int(x))]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert is_empty(rr, nx)
                cc = nx
                cur_color, changed = _apply_contact_color(rr, cc, cur_color, solid, colorize=colorize)
                dye_events += int(changed)
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")

        return int(rr), int(cc), int(cur_color), int(dye_events)

    for _pass in range(H * W):
        moved = False
        order = sorted(range(len(particles)), key=lambda idx: (particles[idx]["r"], particles[idx]["c"]), reverse=True)
        for idx in order:
            src = (int(particles[idx]["r"]), int(particles[idx]["c"]))
            if occupancy.get(src) != idx:
                continue
            del occupancy[src]
            rr, cc, cur_color, dye_events = settle(idx)
            moved = bool(moved or (rr, cc) != src)
            particles[idx]["r"] = int(rr)
            particles[idx]["c"] = int(cc)
            particles[idx]["color"] = int(cur_color)
            particles[idx]["dye_events"] += int(dye_events)
            assert (rr, cc) not in occupancy, "water particles overlapped"
            occupancy[(rr, cc)] = idx
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        def blocked(r: int, c: int) -> bool:
            return int(solid[r, c]) != int(BG) or (int(r), int(c)) in occupancy

        for p in particles:
            rr = int(p["r"])
            cc = int(p["c"])
            assert cc == 0 or blocked(rr, cc - 1)
            assert rr == H - 1 or blocked(rr + 1, cc)
            assert cc == W - 1 or blocked(rr, cc + 1)

    out = solid.copy()
    for p in particles:
        out[int(p["r"]), int(p["c"])] = int(p["color"])

    if return_particles:
        return out, particles
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate gravity-driven water that can be dyed by the basin structures it touches.

    Input structure:
    - bg = 0. Input water starts as color 1, while all other nonzero colors are fixed walls, steps, or guides.
    - Basin structures stay bottom-anchored as in the original task, and water sources are already supported.
    - Structure colors matter twice: they block motion and they act as dye sources when water runs against them.

    Rule:
    - Water keeps the original fall / diagonal slip / horizontal drift behavior from 28a6681f.
    - After each unit move, a water pixel checks the static structure it is touching: a unique side-wall color has
      priority, otherwise a unique supporting color below can dye it.
    - Water color persists after that contact, so later overflow can carry earlier dye into a neighboring basin until
      another contacted wall recolors it.

    Key insight: The geometry of the motion is unchanged; the new difficulty is that each moving water pixel now has a
    path-dependent color state, so the final basin stripes encode where that pixel most recently rubbed against the
    colored structure.
    """
    return _simulate(
        grid,
        check_input=bool(check_input),
        check_output=bool(check_output),
        colorize=True,
        return_particles=False,
    )


def _main_plain(grid: np.ndarray) -> np.ndarray:
    return _simulate(grid, check_input=True, check_output=True, colorize=False, return_particles=False)


_BASE_INPUTS = [
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 3, 0, 3, 1, 1],
            [0, 0, 0, 0, 0, 3, 0, 3, 1, 1],
            [0, 0, 0, 0, 2, 3, 0, 3, 1, 1],
            [0, 0, 0, 0, 2, 3, 0, 3, 1, 1],
            [0, 2, 2, 2, 2, 3, 3, 3, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 3, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 3, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 3, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 3, 0, 0, 3, 1, 1, 1, 1],
            [0, 0, 3, 0, 0, 3, 5, 1, 1, 5],
            [0, 0, 3, 1, 1, 3, 5, 1, 1, 5],
            [0, 0, 3, 1, 1, 3, 5, 1, 1, 5],
            [0, 0, 3, 3, 3, 3, 5, 5, 5, 5],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 9, 1, 1, 0, 0, 0, 0, 9, 9],
            [0, 9, 9, 1, 0, 0, 0, 9, 9, 0],
            [0, 0, 9, 9, 0, 0, 0, 9, 0, 0],
            [0, 0, 0, 9, 0, 0, 9, 9, 0, 0],
            [0, 0, 0, 0, 0, 9, 9, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
            [2, 0, 0, 4, 0, 4, 0, 0, 0, 0],
            [2, 0, 2, 4, 0, 4, 0, 0, 0, 0],
            [2, 2, 2, 4, 4, 4, 0, 0, 0, 0],
        ],
        dtype=int,
    ),
]


def _remap_template(base: np.ndarray) -> np.ndarray:
    g = np.array(base, dtype=int).copy()
    if random.random() < 0.5:
        g = np.fliplr(g)

    solid_colors = sorted(set(map(int, g.flatten())) - {BG, INPUT_WATER})
    palette = random.sample(list(range(2, 10)), len(solid_colors))
    mapping = {old: new for old, new in zip(solid_colors, palette)}

    out = g.copy()
    for old, new in mapping.items():
        out[g == int(old)] = int(new)
    return out


def _variant_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    variant_out, particles = _simulate(
        grid,
        check_input=True,
        check_output=True,
        colorize=True,
        return_particles=True,
    )
    plain_out = _main_plain(grid)

    water_colors = {int(p["color"]) for p in particles}
    dyed_particles = sum(1 for p in particles if int(p["color"]) != int(INPUT_WATER))
    dye_events = sum(int(p["dye_events"]) for p in particles)
    return {
        "distinct_water_colors": int(len(water_colors)),
        "dyed_particles": int(dyed_particles),
        "dye_events": int(dye_events),
        "differs_from_plain": bool(not np.array_equal(variant_out, plain_out)),
    }


def generate_input() -> np.ndarray:
    """
    Generate a valid 10x10 input for the D4 variant.

    Strategy:
    - Start from original-task legal input templates so the basin geometry and support constraints stay valid.
    - Randomly mirror and recolor the structures.
    - Keep only cases where the new dye rule is actually visible: the colored-water output must differ from the
      original monochrome-water output, and the final water must exhibit at least two colors.
    """
    for _ in range(300):
        base = random.choice(_BASE_INPUTS)
        candidate = _remap_template(base)
        metrics = _variant_metrics(candidate)
        if (
            metrics["differs_from_plain"]
            and int(metrics["distinct_water_colors"]) >= 2
            and int(metrics["dyed_particles"]) >= 3
            and int(metrics["dye_events"]) >= 4
        ):
            return candidate
    raise RuntimeError("Failed to generate a visibly dyed input after 300 attempts")
