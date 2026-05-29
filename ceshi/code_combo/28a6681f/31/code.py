# Combo variant of 28a6681f (index 31)
# combo_spec: D4/0+D12/2+D20/0
# Name: 接触变色+壁面侵蚀+壁面湿润标记

from __future__ import annotations

import numpy as np


# Design analysis for the fused variant
# a) 具体化与行为追踪:
#    - 参考原题 spec, 仍使用 10x10 网格、底部相邻盆地、局部重力水流。这里把“水”从单一颜色像素改成
#      “带颜色状态的粒子”, 同时把“墙体”从输出不变量改成可记录历史的结构层。
#    - 一个具体小例子: 左盆较浅, 右盆较深, 中间隔墙是单格厚竖墙。输入里一堆蓝水(1)稳定压在左盆隔墙旁。
#      原题: 水只会在静态几何里重排, 最终仍是全蓝, 隔墙保持原色。
#      融合后:
#      1) 左盆水先贴着左盆底板/外壁/隔墙, 因而分别染成这些墙色;
#      2) 当某些隔墙格左侧是水、右侧是空腔时, 这些隔墙格被侵蚀成背景;
#      3) 后续部分已染色水穿过破口流入右盆, 又可能被右盆壁重新染色;
#      4) 所有曾被水正交接触、但没有被侵蚀掉的墙, 在最终输出里统一改成湿润色。
#      差异不再只是“水到了哪里”, 而是同时要追踪水的路径颜色、哪些墙被碰湿、哪些墙被磨穿。
# b) 复杂度校准:
#    - 为了让三种变化能在 2-4 个 ARC 样例中被看懂, 生成器只保留两个相邻盆地和一面薄内墙, 不再引入
#      额外 guide 形态。这样视觉重心集中在“染色/侵蚀/湿润”三件事上。
#    - 干墙颜色限制为 2/3/4, 湿润标记只用一个输出色 5。水最终只会落在少量结构色上, 颜色语义重复出现,
#      不会变成杂乱的任意映射。
#    - 仅接受同时满足“至少两种染色水 + 至少一段侵蚀缺口 + 至少若干湿墙 + 至少若干未接触干墙”的样本,
#      避免退化成任一单变体。
# c) 隐式假设审查:
#    - 原 `g == water` 默认“颜色1就是全部动态水”; 融合后失效, 因为输出水可能等于 2/3/4。这里改成
#      `solid` 结构层 + `particles` 粒子表 + `occupancy` 占位表。
#    - 原 `g[r, c] = water` 默认回填不会丢信息; 融合后如果仍这样写会抹掉水的染色结果, 因而必须把颜色存到
#      粒子状态里, 最终再一次性渲染输出。
#    - 原 `check_output` 通过颜色找水并验证三向封闭; 侵蚀后该假设也不够强, 因为水可能在破墙后合法地一侧临空。
#      这里改成 D12 风格的“拿走任一水粒子后重新 settle, 必须回到原位”的局部稳定性检查。
#    - D20 单变体把湿墙定义为“最终贴水的墙”; 融合后若仍这么做, 早先接触但后来远离的墙会丢失历史信息。
#      因而湿墙必须记录整段模拟的接触历史, 再减去最终被侵蚀掉的墙。
# d) 放置合法性:
#    - 生成器固定外墙在左右边界、底板锚在底行, 薄隔墙位于中部列 4 或 5, 不越界也不重叠。
#    - 输入水始终填在源盆内部并自下而上堆叠, 每个水像素都有支撑, 且不会左右两侧同时是背景。
#    - 侵蚀只允许发生在非底行、非边界列的结构格上, 防止把盆底或外墙打穿到无意义的状态。
# e) 视觉可观察性:
#    - 输出里同时能看到四类像素: 染色水(2/3/4)、湿墙(5)、侵蚀后空出的背景(0)、未接触保留的干墙(2/3/4)。
#      其中“同一种干墙色既能出现在干墙里, 也能出现在水里”正是 D4 规则的线索。
#    - 样本要求左右盆都留有水或至少发生跨盆转移, 这样人眼能看出“先压墙 -> 破墙 -> 溢流 -> 再染色”的连锁过程。
# f) 可视化整洁度:
#    - 湿润标记只落在墙体原位置上, 不与水叠加; 侵蚀格直接变背景; 水仍只占空腔格。三类变化互斥, 版面清楚。
#    - 上部不再加入多余导流件, 让视线集中在盆地、缺口和颜色条纹上。
# g) 颜色变化可推断性:
#    - 水的颜色只可能来自可见的邻接墙色, 不依赖颜色编号大小、排序或隐藏表。
#    - 湿墙总是统一改成 5, 不是按原墙色分别映射, 因此“被碰过的墙 -> 5”在少量样例中就能稳定归纳。
# h) 退化检验指标:
#    - 生成器对每个候选样本统计: `distinct_dyed_colors`, `dyed_particles`, `wet_count`, `eroded_count`,
#      `untouched_count`, `target_water`, 并要求融合输出同时不同于“原题输出”和“只有侵蚀但没有染色/湿墙”的输出。
#    - 这保证关键控制变量真的起作用, 而不是靠一个本来就会发生的原题水流样本蒙混过关。
# i) 新 solver 假设与 generator 对齐:
#    - solver 假设湿润色 5 只会出现在输出, 输入中不会出现; generator 保证这一点。
#    - solver 假设需要侵蚀的主要对象是一格厚的内隔墙, generator 总是按该结构生成。
#    - solver 的染色规则优先读取左右侧唯一墙色, 否则退回到下方支撑色; generator 让同一盆内部重复出现这些接触,
#      从而把该规则做成可见的重复证据。

BG = 0
INPUT_WATER = 1
WET = 5
LEFT_COLOR = 2
MID_COLOR = 3
RIGHT_COLOR = 4
GRID_SIZE = 10


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


def _validate_input(grid: np.ndarray) -> None:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    assert not np.any(g == int(WET)), "wet color is output-only"

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


def _contact_positions(r: int, c: int, solid: np.ndarray) -> set[tuple[int, int]]:
    H, W = solid.shape
    contacts: set[tuple[int, int]] = set()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < H and 0 <= nc < W and int(solid[nr, nc]) != int(BG):
            contacts.add((nr, nc))
    return contacts


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
    allow_erosion: bool,
    colorize: bool,
    mark_wet: bool,
    check_input: bool,
    check_output: bool,
    return_state: bool = False,
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
    occupancy = {(int(p["r"]), int(p["c"])): idx for idx, p in enumerate(particles)}
    touched_walls: set[tuple[int, int]] = set()
    eroded_walls: set[tuple[int, int]] = set()
    track_history = bool(colorize or mark_wet or allow_erosion or return_state)

    def is_empty(r: int, c: int) -> bool:
        return (
            0 <= int(r) < H
            and 0 <= int(c) < W
            and int(solid[int(r), int(c)]) == int(BG)
            and (int(r), int(c)) not in occupancy
        )

    def register_contact(r: int, c: int, cur_color: int) -> tuple[int, bool]:
        if track_history:
            touched_walls.update(_contact_positions(int(r), int(c), solid))
        return _apply_contact_color(int(r), int(c), int(cur_color), solid, colorize=bool(colorize))

    def settle_particle(start_r: int, start_c: int, start_color: int) -> tuple[int, int, int, int]:
        rr = int(start_r)
        cc = int(start_c)
        cur_color = int(start_color)
        dye_events = 0

        cur_color, changed = register_contact(rr, cc, cur_color)
        dye_events += int(changed)

        for _ in range(H * W):
            if rr + 1 < H and is_empty(rr + 1, cc):
                rr += 1
                cur_color, changed = register_contact(rr, cc, cur_color)
                dye_events += int(changed)
                continue

            opts: list[tuple[int, int]] = []
            if rr + 1 < H and cc - 1 >= 0 and is_empty(rr, cc - 1) and is_empty(rr + 1, cc - 1):
                opts.append((int(rr + 1), int(cc - 1)))
            if rr + 1 < H and cc + 1 < W and is_empty(rr, cc + 1) and is_empty(rr + 1, cc + 1):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                cur_color, changed = register_contact(rr, cc, cur_color)
                dye_events += int(changed)
                continue

            if rr + 1 < H and not is_empty(rr + 1, cc):
                left = int(cc)
                while left - 1 >= 0 and is_empty(rr, left - 1):
                    left -= 1
                right = int(cc)
                while right + 1 < W and is_empty(rr, right + 1):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if is_empty(rr + 1, int(x))]
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
                cur_color, changed = register_contact(rr, cc, cur_color)
                dye_events += int(changed)
                continue

            break
        else:
            raise AssertionError("water particle did not settle")

        return int(rr), int(cc), int(cur_color), int(dye_events)

    def erode_cells() -> list[tuple[int, int]]:
        if not allow_erosion:
            return []
        cells: list[tuple[int, int]] = []
        for r in range(H - 1):
            for c in range(1, W - 1):
                if int(solid[r, c]) == int(BG):
                    continue
                left_is_water = (int(r), int(c - 1)) in occupancy
                right_is_water = (int(r), int(c + 1)) in occupancy
                left_is_empty = int(solid[r, c - 1]) == int(BG) and (int(r), int(c - 1)) not in occupancy
                right_is_empty = int(solid[r, c + 1]) == int(BG) and (int(r), int(c + 1)) not in occupancy
                if (left_is_water and right_is_empty) or (right_is_water and left_is_empty):
                    cells.append((int(r), int(c)))
        return cells

    def settle_trial(start_r: int, start_c: int) -> tuple[int, int]:
        rr = int(start_r)
        cc = int(start_c)
        for _ in range(H * W):
            if rr + 1 < H and is_empty(rr + 1, cc):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
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
                continue

            break
        else:
            raise AssertionError("water particle trial did not settle")

        return int(rr), int(cc)

    for _pass in range(4 * H * W):
        moved = False
        order = sorted(range(len(particles)), key=lambda idx: (particles[idx]["r"], particles[idx]["c"]), reverse=True)
        for idx in order:
            src = (int(particles[idx]["r"]), int(particles[idx]["c"]))
            if occupancy.get(src) != idx:
                continue
            del occupancy[src]
            rr, cc, cur_color, dye_events = settle_particle(
                int(src[0]),
                int(src[1]),
                int(particles[idx]["color"]),
            )
            moved = bool(moved or (int(rr), int(cc)) != src)
            particles[idx]["r"] = int(rr)
            particles[idx]["c"] = int(cc)
            particles[idx]["color"] = int(cur_color)
            particles[idx]["dye_events"] += int(dye_events)
            assert (int(rr), int(cc)) not in occupancy, "water particles overlapped"
            occupancy[(int(rr), int(cc))] = idx

        eroded = erode_cells()
        if eroded:
            for r, c in eroded:
                eroded_walls.add((int(r), int(c)))
                solid[int(r), int(c)] = int(BG)

        if not moved and not eroded:
            break
    else:
        raise AssertionError("water simulation did not converge")

    wet_walls = {
        (int(r), int(c))
        for r, c in touched_walls
        if (int(r), int(c)) not in eroded_walls and int(solid[int(r), int(c)]) != int(BG)
    }

    if check_output:
        for idx, p in enumerate(particles):
            pos = (int(p["r"]), int(p["c"]))
            assert occupancy.get(pos) == idx
            del occupancy[pos]
            rr1, cc1 = settle_trial(int(p["r"]), int(p["c"]))
            occupancy[pos] = idx
            assert (int(rr1), int(cc1)) == pos, "output water must already be locally stable"

        if mark_wet:
            assert eroded_walls.issubset(touched_walls), "eroded walls must have been touched"

    out = solid.copy()
    if mark_wet:
        for r, c in wet_walls:
            out[int(r), int(c)] = int(WET)
    for p in particles:
        out[int(p["r"]), int(p["c"])] = int(p["color"])

    if check_output and mark_wet:
        out_wet = {(int(r), int(c)) for r, c in np.argwhere(out == int(WET))}
        assert out_wet == wet_walls, "wet walls must equal touched-but-not-eroded walls"

    if return_state:
        return {
            "grid": out,
            "solid": solid.copy(),
            "particles": [dict(p) for p in particles],
            "touched_walls": set(touched_walls),
            "eroded_walls": set(eroded_walls),
            "wet_walls": set(wet_walls),
        }
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-filling water whose wall contacts leave three coupled traces: the water is dyed, thin dividers can be eroded away, and surviving contacted walls are marked as wet.

    Input structure:
    - Background is 0 and all input water starts as color 1.
    - Other nonzero colors are solid basin pixels: bottom-anchored floors, outer walls, and a thin interior divider.
    - Water begins in supported source positions pressed against one basin wall, so it can first pool, then potentially breach into a neighbor.

    Rule:
    - Water keeps the original local motion from 28a6681f: fall straight down through empty cells, otherwise take the unique open diagonal, otherwise slide horizontally toward the nearest reachable drop.
    - Whenever a water particle occupies a cell next to a unique side-wall color, it inherits that wall color; if there is no unique side wall, a unique supporting color below can dye it instead.
    - After each full settling sweep, any non-edge wall cell with water on one horizontal side and empty space on the other is eroded into background, opening a new path.
    - Across the whole simulation, every wall cell ever touched by water is remembered; in the final output, touched walls that survived erosion are recolored to wet color 5.

    Key insight: The solver must track motion history, not just the final geometry. A pixel of water may carry an earlier wall color into a later basin, while the wall it touched may either remain as a wet marker or disappear entirely if that same contact caused erosion.
    """
    return _simulate(
        grid,
        allow_erosion=True,
        colorize=True,
        mark_wet=True,
        check_input=bool(check_input),
        check_output=bool(check_output),
        return_state=False,
    )


def _main_plain(grid: np.ndarray) -> np.ndarray:
    return _simulate(
        grid,
        allow_erosion=False,
        colorize=False,
        mark_wet=False,
        check_input=True,
        check_output=True,
        return_state=False,
    )


def _main_erosion_only(grid: np.ndarray) -> np.ndarray:
    return _simulate(
        grid,
        allow_erosion=True,
        colorize=False,
        mark_wet=False,
        check_input=True,
        check_output=True,
        return_state=False,
    )


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int | str]]:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    partition_col = int(rng.choice([4, 5]))
    source_side = str(rng.choice(["L", "R"]))
    shelf_row = int(rng.choice([7, 8]))
    partition_top = int(rng.integers(2, min(5, shelf_row - 1)))
    water_top = int(rng.integers(partition_top, shelf_row - 1))

    if source_side == "L":
        source_color = LEFT_COLOR
        target_color = RIGHT_COLOR
        source_cols = list(range(1, partition_col))
        target_cols = list(range(partition_col + 1, 9))
        source_outer_col = 0
        target_outer_col = 9
        source_fill_order = list(range(partition_col - 1, 0, -1))
    else:
        source_color = RIGHT_COLOR
        target_color = LEFT_COLOR
        source_cols = list(range(partition_col + 1, 9))
        target_cols = list(range(1, partition_col))
        source_outer_col = 9
        target_outer_col = 0
        source_fill_order = list(range(partition_col + 1, 9))

    source_wall_top = int(rng.integers(1, water_top + 1))
    target_wall_top = int(rng.integers(2, 7))

    g[9, :partition_col] = int(LEFT_COLOR)
    g[9, partition_col] = int(MID_COLOR)
    g[9, partition_col + 1 :] = int(RIGHT_COLOR)
    g[partition_top:9, partition_col] = int(MID_COLOR)
    g[source_wall_top:10, source_outer_col] = int(source_color)
    g[target_wall_top:10, target_outer_col] = int(target_color)
    g[shelf_row, source_cols] = int(source_color)

    rows_touching_wall = int(shelf_row - water_top)
    top_count = int(rng.integers(1, len(source_cols) + 1))
    water_count = rows_touching_wall * len(source_cols) - (len(source_cols) - top_count)

    remaining = int(water_count)
    for r in range(shelf_row - 1, -1, -1):
        if remaining <= 0:
            break
        take = min(len(source_cols), remaining)
        for c in source_fill_order[:take]:
            g[int(r), int(c)] = int(INPUT_WATER)
        remaining -= take
    assert remaining == 0

    return g, {
        "partition_col": int(partition_col),
        "source_side": source_side,
        "source_color": int(source_color),
        "target_color": int(target_color),
        "target_cols_min": int(min(target_cols)),
        "target_cols_max": int(max(target_cols)),
    }


def _source_target_counts(
    particles: list[dict[str, int]],
    partition_col: int,
    source_side: str,
) -> tuple[int, int]:
    left_count = sum(1 for p in particles if int(p["c"]) < int(partition_col))
    right_count = sum(1 for p in particles if int(p["c"]) > int(partition_col))
    if source_side == "L":
        return int(left_count), int(right_count)
    return int(right_count), int(left_count)


def _variant_metrics(grid: np.ndarray, meta: dict[str, int | str]) -> dict[str, int | bool]:
    state = _simulate(
        grid,
        allow_erosion=True,
        colorize=True,
        mark_wet=True,
        check_input=True,
        check_output=True,
        return_state=True,
    )
    particles = state["particles"]
    solid = state["solid"]
    touched = state["touched_walls"]
    eroded = state["eroded_walls"]
    wet = state["wet_walls"]
    out = state["grid"]

    plain_out = _main_plain(grid)
    erosion_only_out = _main_erosion_only(grid)
    source_water, target_water = _source_target_counts(
        particles,
        int(meta["partition_col"]),
        str(meta["source_side"]),
    )

    dyed_colors = {int(p["color"]) for p in particles if int(p["color"]) != int(INPUT_WATER)}
    dry_untouched = 0
    for r in range(solid.shape[0]):
        for c in range(solid.shape[1]):
            if int(solid[r, c]) != int(BG) and (int(r), int(c)) not in touched:
                dry_untouched += 1

    return {
        "distinct_dyed_colors": int(len(dyed_colors)),
        "dyed_particles": int(sum(1 for p in particles if int(p["color"]) != int(INPUT_WATER))),
        "dye_events": int(sum(int(p["dye_events"]) for p in particles)),
        "wet_count": int(len(wet)),
        "eroded_count": int(len(eroded)),
        "untouched_count": int(dry_untouched),
        "source_water": int(source_water),
        "target_water": int(target_water),
        "differs_from_plain": bool(not np.array_equal(out, plain_out)),
        "differs_from_erosion_only": bool(not np.array_equal(out, erosion_only_out)),
    }


def generate_input(rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Generate a valid 10x10 input for the fused variant.

    Strategy:
    - Use a safe two-basin layout with one thin interior divider, following the original task's bottom-anchored basin motif.
    - Pre-stack source water against that divider so the simulation first creates visible wall contact, then breaches the divider, then recolors both water and surviving walls.
    - Reject any sample unless all four visual states are present in the output: dyed water, wet surviving walls, eroded wall gaps, and untouched dry walls.
    """
    rng = rng or np.random.default_rng()

    for _ in range(500):
        candidate, meta = _build_case(rng)
        try:
            metrics = _variant_metrics(candidate, meta)
        except Exception:
            continue

        if int(metrics["eroded_count"]) < 2:
            continue
        if int(metrics["wet_count"]) < 5:
            continue
        if int(metrics["untouched_count"]) < 4:
            continue
        if int(metrics["dyed_particles"]) < 5:
            continue
        if int(metrics["dye_events"]) < 6:
            continue
        if int(metrics["distinct_dyed_colors"]) < 2:
            continue
        if int(metrics["source_water"]) < 2:
            continue
        if int(metrics["target_water"]) < 2:
            continue
        if not bool(metrics["differs_from_plain"]):
            continue
        if not bool(metrics["differs_from_erosion_only"]):
            continue

        return candidate

    raise RuntimeError("generate_input: failed to construct a visible fused sample")
