# Combo variant of 28a6681f (index 5)
# combo_spec: D4/0+D16/0
# Name: 接触变色+壁面摩擦力

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# a) 具体化与行为追踪:
#    原题里水只有位置状态: 它会下落、斜滑、水平找缺口, 但永远保持颜色 1, 也不会因为支撑面的颜色不同而停住。
#    这个融合版把“接触颜色”和“支撑摩擦”绑定到同一组壁色上。这里固定 color 2 = 粗糙壁, color 3 = 光滑壁。
#    一个典型轨迹是: 左上角一排源水同时压在 3-3-2-2-2 这样的露台上。原题中这整排水都会继续向缺口滑动并坠入右侧
#    低盆地。融合后, 压在 3 上的水会先被染成 3, 然后继续水平滑向缺口; 压在 2 上的水会因为高摩擦被卡在原地,
#    同时被染成 2。于是输出里会同时出现“停在高处的 2 色水”和“流入低盆地的 3 色水”, 两种颜色既编码接触史,
#    也编码运动分支。
# b) 复杂度校准:
#    颜色角色只保留两个档位, 避免“每种颜色一个特殊物理”导致样例不足以归纳。每个模板都会重复出现多格粗糙露台与
#    多格光滑露台, 让解题者能从 2-4 个例子里同时看到“2 会阻流且把水染成 2”“3 允许滑流且把水染成 3”。
#    生成器还要求组合输出相对无摩擦染色版在水位置上明显不同, 防止退化成纯染色题。
# c) 隐式假设审查:
#    - 原 solver 把 `1` 既当输入标签又当唯一动态水色; 组合后输出水可能是 2 或 3, 所以必须把静态结构和动态粒子分离。
#    - 原 solver 用 `g == 1` 枚举所有水; 组合后这样会漏掉已染色水, 因此改为显式粒子列表 + 占据表。
#    - 原 solver 假设“只要下方有支撑就一定能继续水平搜索缺口”; 融合后只有光滑支撑能滑, 粗糙支撑会立即截停。
#    - 原 solver 的输出检查假设最终水左右和下方都不是背景。粗糙露台会合法地产生“旁边仍有空位但因摩擦停住”的水,
#      因此改为把每个终态粒子移除后局部重放, 检查它的终点与终色是否不再变化。
#    - 单独做 D4 时“仅在移动后染色”还说得通, 但组合后会出现输入源水一开始就压在粗糙露台上的情况。如果不在当前
#      接触位置也触发染色, 那些被摩擦立即截停的水会错误地保持蓝色, 与规格冲突。因此这里把染色定义为“每轮结算时
#      读取当前接触的唯一结构色”, 不要求先移动一步。
# d) 放置合法性:
#    模板全部是 10x10, 结构只放在底部三行, 源水是贴边的两层水平条, 不会越界或与结构重叠。上层源水宽度不大于
#    下层宽度, 所以每个输入水像素都由下方的水或结构支撑。露台和缺口布局只保留一个最近可达落点, 维持原题
#    “nearest reachable drop 唯一”的物理前提。
# e) 视觉可观察性:
#    看输入输出时, 人类能直接看到两类证据重复出现: 一类水停在高处粗糙台阶上并变成 2; 另一类水沿 3 色露台继续流
#    动, 最终落入右侧或左侧低盆地并保持/变成 3。因为镜像也会采样, 规则不能靠“总往右”记忆, 必须靠颜色角色归纳。
# f) 可视化整洁度:
#    整个画面只保留底部露台结构和上方源水, 没有额外标记层。结构色只有 2 和 3, 输出水也只会落在这两色上, 画面仍然
#    很干净: 高位残留水和低位流入水在几何上分区明显, 不会出现多种符号互相遮挡。
# g) 变化的可推断性:
#    颜色变化不是任意映射, 而是与接触位置绑定。粗糙色和光滑色都在同一张图里多次重复, 且两者分别对应不同的运动
#    结果, 所以解题者既能从颜色条纹看出“水接触哪种壁就变哪种色”, 也能从停/流分化看出“这两种壁的摩擦不同”。
# h) 融合行为量化:
#    除了基本成功率测试外, 生成器会测量四类指标: 粗糙阻流次数、光滑水平滑移次数、染成 2/3 的事件数、以及组合输出
#    与“无摩擦仅染色版”的水位置差异。只有这些核心量都非零, 才接受样本。
# i) 新 solver 假设与 generator 保证:
#    - 颜色 2 与 3 的功能映射是固定的, 不是按数值大小动态推断; generator 只使用这两种结构色。
#    - 摩擦只在“当前像素已经有支撑, 并尝试进行原题的水平搜索”时生效; generator 因此把源水直接放在活动露台上。
#    - 染色只读取静态结构, 不读取其他水粒子; generator 不会预置已染色输入水, 避免语义混淆。

BG = 0
INPUT_WATER = 1
ROUGH = 2
SMOOTH = 3

_TEMPLATE_SPECS = (
    {
        "bottom_rows": ("3322200000", "0022200300", "0022203330"),
        "source_widths": (5, 5),
    },
    {
        "bottom_rows": ("3322200000", "0022203000", "0022203330"),
        "source_widths": (4, 5),
    },
    {
        "bottom_rows": ("3322200000", "0022200000", "0022203330"),
        "source_widths": (5, 5),
    },
    {
        "bottom_rows": ("3322200000", "0022203030", "0022203330"),
        "source_widths": (4, 4),
    },
    {
        "bottom_rows": ("3322000300", "0022200300", "0022230330"),
        "source_widths": (4, 4),
    },
)


def _grid_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _build_candidate(spec: dict[str, object], mirror: bool) -> np.ndarray:
    rows = ["0000000000"] * 7 + list(spec["bottom_rows"])
    g = _grid_from_rows(rows)
    top_width, bottom_width = spec["source_widths"]
    if bottom_width:
        g[6, : int(bottom_width)] = INPUT_WATER
    if top_width:
        g[5, : int(top_width)] = INPUT_WATER
    if mirror:
        g = np.fliplr(g).copy()
    return g


def _validate_input(g: np.ndarray) -> None:
    H, W = g.shape
    assert (H, W) == (10, 10), "grid must be 10x10"
    assert np.any(g == ROUGH), "input must contain rough structure color 2"
    assert np.any(g == SMOOTH), "input must contain smooth structure color 3"

    for r, c in np.argwhere(g == int(INPUT_WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"


def _apply_contact_color(r: int, c: int, cur_color: int, solid: np.ndarray) -> tuple[int, int | None]:
    H, W = solid.shape
    side_colors: set[int] = set()
    if c - 1 >= 0 and int(solid[r, c - 1]) != int(BG):
        side_colors.add(int(solid[r, c - 1]))
    if c + 1 < W and int(solid[r, c + 1]) != int(BG):
        side_colors.add(int(solid[r, c + 1]))

    if len(side_colors) == 1:
        new_color = next(iter(side_colors))
    elif len(side_colors) >= 2:
        return int(cur_color), None
    elif r + 1 < H and int(solid[r + 1, c]) != int(BG):
        new_color = int(solid[r + 1, c])
    else:
        return int(cur_color), None

    if int(new_color) == int(cur_color):
        return int(cur_color), None
    return int(new_color), int(new_color)


def _particle_mask(particles: list[dict[str, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for p in particles:
        mask[int(p["r"]), int(p["c"])] = True
    return mask


def _simulate(
    grid: np.ndarray,
    *,
    friction_enabled: bool,
    colorize: bool,
    check_input: bool,
    check_output: bool,
    collect_metrics: bool,
) -> tuple[np.ndarray, list[dict[str, int]], dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        _validate_input(g)

    solid = np.where((g != int(BG)) & (g != int(INPUT_WATER)), g, int(BG))
    particles = [
        {"r": int(r), "c": int(c), "color": int(INPUT_WATER)}
        for r, c in np.argwhere(g == int(INPUT_WATER))
    ]
    occupancy = {(p["r"], p["c"]): idx for idx, p in enumerate(particles)}
    metrics = {
        "rough_blocks": 0,
        "smooth_slides": 0,
        "dye_events": 0,
        "dye_to_rough": 0,
        "dye_to_smooth": 0,
    }

    def is_empty(r: int, c: int) -> bool:
        return (
            0 <= int(r) < H
            and 0 <= int(c) < W
            and int(solid[int(r), int(c)]) == int(BG)
            and (int(r), int(c)) not in occupancy
        )

    def settle_particle(
        start_r: int,
        start_c: int,
        start_color: int,
        *,
        record_metrics: bool,
    ) -> tuple[int, int, int]:
        rr = int(start_r)
        cc = int(start_c)
        cur_color = int(start_color)

        for _ in range(H * W):
            if colorize:
                new_color, dye_target = _apply_contact_color(rr, cc, cur_color, solid)
                if dye_target is not None:
                    cur_color = int(new_color)
                    if record_metrics and collect_metrics:
                        metrics["dye_events"] += 1
                        if int(dye_target) == int(ROUGH):
                            metrics["dye_to_rough"] += 1
                        elif int(dye_target) == int(SMOOTH):
                            metrics["dye_to_smooth"] += 1

            if rr + 1 < H and is_empty(rr + 1, cc):
                rr += 1
                continue

            opts = []
            if rr + 1 < H and cc - 1 >= 0 and is_empty(rr, cc - 1) and is_empty(rr + 1, cc - 1):
                opts.append((int(rr + 1), int(cc - 1)))
            if rr + 1 < H and cc + 1 < W and is_empty(rr, cc + 1) and is_empty(rr + 1, cc + 1):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and not is_empty(rr + 1, cc):
                left = int(cc)
                while left - 1 >= 0 and is_empty(rr, left - 1):
                    left -= 1
                right = int(cc)
                while right + 1 < W and is_empty(rr, right + 1):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if is_empty(rr + 1, int(x))]
                support_color = int(solid[rr + 1, cc]) if int(solid[rr + 1, cc]) != int(BG) else int(INPUT_WATER)
                if friction_enabled and support_color == int(ROUGH):
                    if record_metrics and collect_metrics and drops:
                        metrics["rough_blocks"] += 1
                    break
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = int(nearest[0])
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert is_empty(rr, nx)
                if record_metrics and collect_metrics:
                    metrics["smooth_slides"] += 1
                cc = nx
                continue

            break
        else:
            raise AssertionError("water particle did not settle")

        return int(rr), int(cc), int(cur_color)

    for _pass in range(H * W):
        changed = False
        order = sorted(range(len(particles)), key=lambda idx: (particles[idx]["r"], particles[idx]["c"]), reverse=True)
        for idx in order:
            src = (int(particles[idx]["r"]), int(particles[idx]["c"]))
            if occupancy.get(src) != idx:
                continue
            old_color = int(particles[idx]["color"])
            del occupancy[src]
            rr, cc, cur_color = settle_particle(
                int(src[0]),
                int(src[1]),
                int(old_color),
                record_metrics=True,
            )
            particles[idx]["r"] = int(rr)
            particles[idx]["c"] = int(cc)
            particles[idx]["color"] = int(cur_color)
            assert (rr, cc) not in occupancy, "water particles overlapped"
            occupancy[(rr, cc)] = idx
            if (rr, cc) != src or int(cur_color) != int(old_color):
                changed = True
        if not changed:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for idx, p in enumerate(particles):
            src = (int(p["r"]), int(p["c"]))
            assert occupancy.get(src) == idx
            del occupancy[src]
            rr, cc, cur_color = settle_particle(
                int(src[0]),
                int(src[1]),
                int(p["color"]),
                record_metrics=False,
            )
            occupancy[src] = idx
            assert (rr, cc, cur_color) == (int(src[0]), int(src[1]), int(p["color"])), "output water is not locally stable"

    metrics["distinct_final_colors"] = int(len({int(p["color"]) for p in particles}))
    metrics["dyed_particles"] = int(sum(1 for p in particles if int(p["color"]) != int(INPUT_WATER)))
    metrics["final_rough"] = int(sum(1 for p in particles if int(p["color"]) == int(ROUGH)))
    metrics["final_smooth"] = int(sum(1 for p in particles if int(p["color"]) == int(SMOOTH)))
    metrics["top_water"] = int(sum(1 for p in particles if int(p["r"]) < 7))
    metrics["bottom_water"] = int(sum(1 for p in particles if int(p["r"]) >= 7))

    out = solid.copy()
    for p in particles:
        out[int(p["r"]), int(p["c"])] = int(p["color"])
    return out, particles, metrics


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    combo_out, combo_particles, combo = _simulate(
        grid,
        friction_enabled=True,
        colorize=True,
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    friction_out, friction_particles, _ = _simulate(
        grid,
        friction_enabled=True,
        colorize=False,
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )
    dye_out, dye_particles, _ = _simulate(
        grid,
        friction_enabled=False,
        colorize=True,
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )
    plain_out, plain_particles, _ = _simulate(
        grid,
        friction_enabled=False,
        colorize=False,
        check_input=True,
        check_output=True,
        collect_metrics=False,
    )

    combo_mask = _particle_mask(combo_particles, combo_out.shape)
    friction_mask = _particle_mask(friction_particles, combo_out.shape)
    dye_mask = _particle_mask(dye_particles, combo_out.shape)
    plain_mask = _particle_mask(plain_particles, combo_out.shape)

    result = dict(combo)
    result["pos_diff_vs_dye_only"] = int(np.count_nonzero(combo_mask != dye_mask))
    result["pos_diff_vs_plain"] = int(np.count_nonzero(combo_mask != plain_mask))
    result["pos_diff_vs_friction_only"] = int(np.count_nonzero(combo_mask != friction_mask))
    result["cell_diff_vs_friction_only"] = int(np.count_nonzero(combo_out != friction_out))
    result["cell_diff_vs_plain"] = int(np.count_nonzero(combo_out != plain_out))
    result["cell_diff_vs_dye_only"] = int(np.count_nonzero(combo_out != dye_out))
    return result


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate gravity-driven water in bottom-anchored basins where wall colors now control both dyeing and friction.

    Input structure:
    - Background is 0 and every input water pixel starts as color 1.
    - Color 2 marks rough basin or terrace cells, while color 3 marks smooth basin or terrace cells.
    - The colored structure is static; water begins as a supported source resting on those terraces.

    Rule:
    - Water still follows the original local geometry: fall straight down through empty cells, slip diagonally through a unique open shoulder, and otherwise try to slide horizontally toward the nearest reachable drop.
    - At each settling step, a water pixel inspects the static structure it is touching: a unique side-wall color takes priority, otherwise the supporting color below dyes the water to that structure color.
    - Horizontal sliding now depends on the support color: rough support (2) cancels that slide immediately, while smooth support (3) allows the original nearest-drop search to continue.
    - Repeat full passes until neither positions nor contact-driven colors change.

    Key insight: The same wall color must be read in two ways at once: it predicts whether water can keep spilling sideways, and it also predicts the color the water will acquire when it touches that wall.
    """
    out, _particles, _metrics = _simulate(
        grid,
        friction_enabled=True,
        colorize=True,
        check_input=bool(check_input),
        check_output=bool(check_output),
        collect_metrics=False,
    )
    return out


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng
    for _ in range(64):
        spec = rng.choice(_TEMPLATE_SPECS)
        mirror = bool(rng.randrange(2))
        candidate = _build_candidate(spec, mirror)
        try:
            stats = variant_metrics(candidate)
        except AssertionError:
            continue
        if stats["rough_blocks"] < 1:
            continue
        if stats["smooth_slides"] < 1:
            continue
        if stats["dye_to_rough"] < 2:
            continue
        if stats["dye_to_smooth"] < 2:
            continue
        if stats["distinct_final_colors"] < 2:
            continue
        if stats["final_rough"] < 2:
            continue
        if stats["final_smooth"] < 2:
            continue
        if stats["top_water"] < 2:
            continue
        if stats["bottom_water"] < 2:
            continue
        if stats["pos_diff_vs_dye_only"] < 2:
            continue
        if stats["cell_diff_vs_friction_only"] < 4:
            continue
        if stats["cell_diff_vs_plain"] < 6:
            continue
        return candidate
    raise AssertionError("failed to generate a non-degenerate combo sample")
