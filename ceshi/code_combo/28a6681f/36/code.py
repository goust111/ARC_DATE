# Combo variant of 28a6681f (index 36)
# combo_spec: D3/2+D12/0+D16/0
# Name: 水压滑动+蒸发+壁面摩擦

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) 具体化与行为追踪:
#    - 保留 10x10、bg=0、water=1，并固定 3 个有意义的固体颜色:
#      2=高摩擦粗糙壁, 3=下方盆地地板, 4=低摩擦导轨/支撑。
#    - 融合后的移动对象不再是“独立单像素”, 而是“垂直水柱包”:
#      若某列当前有 h 个连续水像素, 则它最多只能做 h-1 次非垂直步。
#      每做一次对角或水平步, 水柱就蒸发掉 1 个像素, 因而剩余高度减 1。
#    - 本生成器固定造两个主动水柱和一小块稳定盆地残水:
#      左侧水柱沿导轨向中央缺口寻找落点, 但会先踩到颜色 2 的粗糙带而提前停住;
#      右侧水柱先从上方导柱对角落到导轨, 再因压力不足停在缺口前。
#    - 与原题对照:
#      原题的单像素既不蒸发也不会受颜色摩擦影响, 因而总会继续向最近落点滑;
#      这里则要同时跟踪“还有几格压力预算”“是否踩到粗糙支撑”“已经损失了多少水”。
# b) 复杂度校准:
#    - 颜色语义只有三类并在每个样例里重复出现, 不依赖颜色编号排序。
#    - 只保留两条主动路径: 一条专门触发粗糙截停, 一条专门触发压力不足;
#      这样三种机制都可见, 但自由度不会多到难以归纳。
#    - 源水柱高度只从很小的档位集合里采样, 保证样例间差异明显且可比较。
# c) 隐式假设审查:
#    - 原 solver 的“总水量守恒”假设失效; 这里每次非垂直移动都会减少水量。
#    - 原 solver 的“逐像素独立 settle”也失效; 水平可走多远依赖整根水柱高度,
#      因而必须把同一列中连续的水像素打包处理。
#    - 原 solver 末尾用“左右都非背景”近似稳定性, 这里不再成立:
#      水可以因为压力耗尽或粗糙截停而停在侧面仍有空隙的位置, 所以改用重放求解器的固定点检查。
#    - 最近落点仍需唯一; 生成器把两条活动路径都做成单缺口, 避免横向搜索歧义。
#    - 非垂直移动要求至少还剩 2 个水像素; 生成器只使用足够高的主动水柱。
# d) 放置合法性:
#    - 两个主动水柱分别贴左右边界, 输入时天然不会出现“左右两边同时是背景”。
#    - 右侧导柱、中央粗糙带和底部平滑盆地都显式写死在 10x10 内, 不会越界或重叠。
#    - 盆地残水放成相邻的两格, 既能保持“有底部残留水”的视觉线索, 又满足输入支撑约束。
# e) 视觉可观察性:
#    - 颜色 2 在中央粗糙带重复出现, 颜色 4 在两条导轨和导柱重复出现, 颜色角色容易从少量样例归纳。
#    - 左侧水柱总是朝缺口方向移动, 但在触碰粗糙带后提前停住并少蒸发几格;
#      右侧水柱则因为高度不同而停在缺口前的不同位置。
#    - 即使某个样例中主动水未真正落进盆地, 下方仍保留稳定盆地残水, 让解题者看得到“目标盆地”。
# f) 可视化整洁度:
#    - 上层主动轨道、中层粗糙脊和底层盆地分居不同区域, 没有颜色或对象重叠遮挡。
#    - 两条主动路径分处左右, 不会在同一行直接互相堵塞。
# g) 变化可推断性:
#    - 颜色映射固定: 2 永远表示高摩擦, 3 永远表示盆地平滑地板, 4 永远表示可滑导轨/导柱。
#      变化来自几何和水柱高度, 而不是不可见的颜色排序。
# 9) 新 solver 假设与 generator 保证:
#    - solver 假设活动路径上的最近落点唯一, generator 用单缺口保证这一点。
#    - solver 假设活动水以垂直连续列出现, generator 只构造边界贴附的垂直水柱和稳定残水。
#    - solver 假设放置剩余水柱时上方留有足够背景空间, generator 只使用不会顶到天花板的高度档位。

BG = 0
WATER = 1
ROUGH = 2
SMOOTH = 3
GUIDE = 4

_PROFILE_CHOICES = (
    (5, 3),
    (5, 4),
    (6, 3),
    (6, 4),
)


def _grid_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _build_candidate(left_height: int, right_height: int, mirror: bool) -> np.ndarray:
    assert left_height in (5, 6)
    assert right_height in (3, 4)

    g = _grid_from_rows(
        [
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000004",
            "4442204444",
            "0022201100",
            "0022233333",
        ]
    )

    for r in range(7 - int(left_height), 7):
        g[r, 0] = WATER
    for r in range(6 - int(right_height), 6):
        g[r, 9] = WATER

    if mirror:
        g = np.fliplr(g).copy()
    return g


def _column_height(g: np.ndarray, r: int, c: int) -> int:
    height = 0
    rr = int(r)
    while rr >= 0 and int(g[rr, c]) == WATER:
        height += 1
        rr -= 1
    return int(height)


def _remove_column(g: np.ndarray, r: int, c: int) -> None:
    rr = int(r)
    while rr >= 0 and int(g[rr, c]) == WATER:
        g[rr, c] = BG
        rr -= 1


def _place_column(g: np.ndarray, r: int, c: int, height: int) -> None:
    for offset in range(int(height)):
        rr = int(r - offset)
        assert rr >= 0 and int(g[rr, c]) == BG, "not enough headroom to place settled water"
        g[rr, c] = WATER


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    for r, c in np.argwhere(g == WATER):
        rr = int(r)
        cc = int(c)
        assert rr == H - 1 or int(g[rr + 1, cc]) != BG, "legacy input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == BG
        assert not (left_bg and right_bg), "legacy input water cannot have bg on both horizontal sides"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == BG:
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == BG
                and int(g[rr + 1, cc - 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == BG
                and int(g[rr + 1, cc + 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous legacy diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != BG:
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == BG:
                    right += 1
                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == BG]
                if not drops:
                    break
                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous legacy nearest drop columns: {nearest}"
                best = int(nearest[0])
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == BG, "legacy slide crossed a blocker"
                cc = nx
                continue

            break
        else:
            raise AssertionError("legacy water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == WATER)], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != WATER:
                continue
            g[r, c] = BG
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = WATER
        if not moved:
            break
    else:
        raise AssertionError("legacy simulation did not converge")

    return g


def _simulate(
    grid: np.ndarray,
    *,
    friction_enabled: bool,
    pressure_enabled: bool,
    check_input: bool,
    check_output: bool,
    collect_metrics: bool = False,
) -> tuple[np.ndarray, dict[str, int]] | np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    metrics = (
        {
            "diag_moves": 0,
            "horizontal_moves": 0,
            "rough_stops": 0,
            "pressure_caps": 0,
            "evaporation_loss": 0,
            "fully_evaporated_packets": 0,
        }
        if collect_metrics
        else None
    )

    if check_input:
        assert g.shape == (10, 10), "grid must be 10x10"
        assert np.any(g == ROUGH), "input must contain rough supports"
        assert np.any(g == GUIDE), "input must contain guide supports"
        assert np.any(g == SMOOTH), "input must contain a lower smooth basin"
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != BG, "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == BG
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def bump(key: str, amount: int = 1) -> None:
        if metrics is not None:
            metrics[key] += int(amount)

    def settle_packet(r: int, c: int, height: int) -> tuple[int, int, int] | None:
        rr = int(r)
        cc = int(c)
        mass = int(height)

        for _ in range(H * W * 4):
            fell = False
            while rr + 1 < H and int(g[rr + 1, cc]) == BG:
                rr += 1
                fell = True
            if fell:
                continue

            opts: list[tuple[int, int]] = []
            if mass >= 2:
                if (
                    rr + 1 < H
                    and cc - 1 >= 0
                    and int(g[rr, cc - 1]) == BG
                    and int(g[rr + 1, cc - 1]) == BG
                ):
                    opts.append((int(rr + 1), int(cc - 1)))
                if (
                    rr + 1 < H
                    and cc + 1 < W
                    and int(g[rr, cc + 1]) == BG
                    and int(g[rr + 1, cc + 1]) == BG
                ):
                    opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                mass -= 1
                bump("diag_moves")
                bump("evaporation_loss")
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != BG and (mass >= 2 or not pressure_enabled):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == BG:
                    right += 1
                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == BG]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = int(nearest[0])

                if friction_enabled and int(g[rr + 1, cc]) == ROUGH:
                    bump("rough_stops")
                    break

                step = -1 if best < int(cc) else 1
                if pressure_enabled:
                    if min_dist > int(mass - 1):
                        bump("pressure_caps")
                    max_steps = min(int(min_dist), int(mass - 1))
                else:
                    max_steps = int(min_dist)
                if max_steps == 0:
                    break

                for _slide in range(int(max_steps)):
                    nx = int(cc + step)
                    assert int(g[rr, nx]) == BG, "horizontal slide crossed a blocker"
                    cc = nx
                    mass -= 1
                    bump("horizontal_moves")
                    bump("evaporation_loss")
                    if mass == 0:
                        bump("fully_evaporated_packets")
                        return None
                    if friction_enabled and int(g[rr + 1, cc]) == ROUGH:
                        bump("rough_stops")
                        return int(rr), int(cc), int(mass)
                    if int(g[rr + 1, cc]) == BG:
                        break
                continue

            break
        else:
            raise AssertionError("water packet did not settle")

        return int(rr), int(cc), int(mass)

    for _pass in range(H * W):
        moved = False
        bottoms = sorted(
            [(int(r), int(c)) for r, c in np.argwhere(g == WATER) if int(r) == H - 1 or int(g[int(r) + 1, int(c)]) != WATER],
            reverse=True,
        )
        for r, c in bottoms:
            if int(g[r, c]) != WATER:
                continue
            height = _column_height(g, int(r), int(c))
            _remove_column(g, int(r), int(c))
            settled = settle_packet(int(r), int(c), int(height))
            if settled is None:
                moved = True
                continue
            rr, cc, mass = settled
            _place_column(g, int(rr), int(cc), int(mass))
            moved = bool(moved or ((int(rr), int(cc), int(mass)) != (int(r), int(c), int(height))))
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if metrics is not None:
        metrics["input_water"] = int(np.count_nonzero(np.array(grid, dtype=int) == WATER))
        metrics["output_water"] = int(np.count_nonzero(g == WATER))
        assert metrics["evaporation_loss"] == metrics["input_water"] - metrics["output_water"], "evaporation accounting mismatch"

    if check_output:
        stabilized = _simulate(
            g,
            friction_enabled=friction_enabled,
            pressure_enabled=pressure_enabled,
            check_input=False,
            check_output=False,
            collect_metrics=False,
        )
        assert np.array_equal(stabilized, g), "output is not a fixed point of the fused solver"

    if metrics is None:
        return g
    return g, metrics


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    out_full, metrics = _simulate(
        grid,
        friction_enabled=True,
        pressure_enabled=True,
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    out_no_friction = _simulate(
        grid,
        friction_enabled=False,
        pressure_enabled=True,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    out_no_pressure = _simulate(
        grid,
        friction_enabled=True,
        pressure_enabled=False,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    out_legacy = _legacy_main(grid)

    result = dict(metrics)
    result["diff_no_friction"] = int(np.count_nonzero(out_full != out_no_friction))
    result["diff_no_pressure"] = int(np.count_nonzero(out_full != out_no_pressure))
    result["diff_legacy"] = int(np.count_nonzero(out_full != out_legacy))
    result["upper_water"] = int(np.count_nonzero(out_full[:8] == WATER))
    result["basin_water"] = int(np.count_nonzero(out_full[8:] == WATER))
    return result


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate supported water columns losing mass as they descend across guide ledges, rough ridges, and a lower basin.

    Input structure:
    - Background is 0 and water is color 1.
    - Color 2 marks high-friction supports, color 3 marks the lower basin floor, and color 4 marks low-friction guide ledges or posts.
    - Water appears as supported vertical source columns near the side edges plus a small already-stable carryover inside the lower basin.

    Rule:
    - A moving water object is treated as a whole vertical column: it may fall straight down for free, but every diagonal or horizontal step evaporates one pixel from that column.
    - When the column is supported, it still aims toward the unique nearest drop in the current background run, but it can spend at most its current column height minus one horizontal steps.
    - If the column ever comes to rest on a rough support cell, horizontal sliding stops immediately; smooth and guide supports allow the slide to continue.
    - Re-run settling passes until every remaining water column is a fixed point of these fused rules.

    Key insight: The final state is determined by three linked visible quantities at once: column height sets the slide budget, rough color truncates that budget early, and every spent step becomes evaporation, so path length and surviving water amount must be inferred together.
    """
    return _simulate(
        grid,
        friction_enabled=True,
        pressure_enabled=True,
        check_input=check_input,
        check_output=check_output,
        collect_metrics=False,
    )


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng
    for _ in range(64):
        left_height, right_height = rng.choice(_PROFILE_CHOICES)
        mirror = bool(rng.randrange(2))
        candidate = _build_candidate(int(left_height), int(right_height), mirror)
        try:
            stats = fusion_metrics(candidate)
        except AssertionError:
            continue
        if stats["rough_stops"] < 1:
            continue
        if stats["pressure_caps"] < 1:
            continue
        if stats["diag_moves"] < 1:
            continue
        if stats["horizontal_moves"] < 5:
            continue
        if stats["evaporation_loss"] < 5:
            continue
        if stats["diff_no_friction"] < 2:
            continue
        if stats["diff_no_pressure"] < 1:
            continue
        if stats["diff_legacy"] < 6:
            continue
        if stats["basin_water"] < 2:
            continue
        return candidate
    raise AssertionError("failed to generate a non-degenerate pressure+evaporation+friction sample")
