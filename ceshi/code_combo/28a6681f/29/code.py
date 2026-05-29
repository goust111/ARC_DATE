# Combo variant of 28a6681f (index 29)
# combo_spec: D4/1+D11/1+D16/0
# Name: 多色水+分批处理+壁面摩擦

from __future__ import annotations

import random

import numpy as np

BG = 0
BLUE = 1
RED = 2
ROUGH = 3
SMOOTH = 4
GUIDE = 5

WATER_COLORS = (BLUE, RED)
WATER_ORDER = (BLUE, RED)
GRID_SHAPE = (10, 10)


# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 仍然保留 10x10、底部锚定盆地、垂直下落 / 唯一对角落 / 最近坠口水平滑动这套原始物理。
# - 融合后的关键改动是两个：水不再一起动，而是按颜色分批；同时颜色 3 的壁面变成高摩擦壁，
#   它只会取消“已经被支撑住之后的水平滑动”，不会影响下落和对角落。
# - 生成器固定使用两种可动水：蓝水 1 先处理，红水 2 后处理。为了让 D11 的批次效应肉眼可见，
#   我们故意把蓝水放在更高的上层发射台，把红水放在更低的下层发射台。
# - 原始逻辑里，较低的红水会先因行序优势接近坠口；融合逻辑里，solver 会先把整批蓝水结算完，
#   让蓝水先占据粗糙台阶和右侧盆地，再把红水放进去。于是红水虽然起点更低，却更晚遇到通道。
# - 具体小例子：蓝水在第 4 行导流台上，红水在第 6 行粗糙前台上。原始无批次逻辑会先让红水争抢
#   左侧坠口；融合逻辑先让蓝水落到颜色 3 的粗糙肩部，其中一格会在可继续外溢时被直接钉住，
#   这格蓝水随后就变成红水的额外障碍。红水再轮到时，会比“无摩擦”或“同时处理”都更容易留在
#   起始行，形成“先来的蓝水改变后来的红水路径”的联合作用。
#
# b) 复杂度校准
# - 只保留 2 种水色和 3 种静态颜色，避免把任务变成“太多颜色角色同时学习”。
# - 静态颜色职责固定且重复出现：3 永远粗糙，4 永远平滑盆壁/盆底，5 永远上方导流台。
# - 几何结构只使用一个很窄的主通道和一个右侧接收盆地，再通过镜像和蓝水高度/宽度的小变化产生样例，
#   这样规则变化明显，但自由度仍然受控。
#
# c) 原 main() 中依赖旧属性的代码审查
# - `water = 1`：旧版只认识一种水；融合版必须同时识别 1 和 2，并保留各自颜色。
# - `np.argwhere(g == water)`：旧版全体一起移动；融合版需要同时支持“全部一起”与“按颜色分批”两种
#   调试/对照模式，因为生成器要拿它和两个消融版本比较。
# - `g[rr, cc] = water`：旧版落位总写回 1；融合版必须写回像素自身颜色。
# - 原输入检查默认“看见的整图合法即可”；融合版额外需要“每一批颜色单独拿出来也合法”，因为在蓝批次
#   运行时，红水会被暂时移除，反之亦然。否则会出现后批次源区在前批次移走后悬空的隐式冲突。
# - 原输出检查假设稳定水左右和下方都不再接触背景；粗糙壁引入后，这个前提失效，因为水可以合法地停在
#   仍有侧向空隙的粗糙台阶上。融合版改用“整套动力学重放后保持不变”的不动点检查。
#
# d) 放置合法性
# - 生成器先画底部 3 行盆地，再画第 5 行上导流台，最后放蓝/红源区，不会发生覆盖冲突。
# - 右侧盆地始终锚定底边，左侧前台始终留有唯一坠口；因此最近坠口规则保持单值，不会产生水平歧义。
# - 所有模板都可整体镜像，镜像后支撑关系、边界合法性和唯一坠口性质保持不变。
#
# e) 视觉可观察性
# - 蓝水被故意放得比红水更高，但输出中蓝水仍先出现在更深的槽位和粗糙肩部，这直接暴露“按颜色分批，
#   且蓝先红后”的隐藏顺序，而不是普通的高度优先。
# - 颜色 3 的粗糙肩部会在输出里留下至少一个“蓝水压在粗糙壁上”的停滞像素；同时红水相比无摩擦版会
#   更多地留在下层源行，肉眼能看出“粗糙壁 + 先到蓝水”共同阻塞了后续红水。
#
# f) 可视化整洁度
# - 整个场景只保留一个上层导流台、一个下层发射带、一个粗糙肩部和一个右侧接收盆地，避免多通道叠加。
# - 蓝水、红水、粗糙壁和平滑壁各自分区明确；输出虽然有相互作用，但不会出现多层标记重叠到难以辨认。
#
# g) 变化的可推断性
# - 颜色语义固定不变：1=先批次蓝水，2=后批次红水，3=粗糙壁，4=平滑盆壁/盆底，5=导流台。
# - 样例里的随机性只来自镜像、蓝水宽度和蓝水层数，不来自颜色 id 排序或不可见打分。
# - 因此解题者能从少量样例归纳出：不是“低的水先动”，而是“蓝先结算；粗糙壁会把先到的蓝水钉住，
#   再间接改变红水的可达路径”。
#
# 9) 新 solver 假设与 generator 保证
# - solver 假设每个颜色批次单独加入场景时就已经满足原题那种“有支撑、不会左右两侧同时邻接背景”的
#   源区合法性；生成器用固定导流台和连续源块保证这一点。
# - solver 假设粗糙停滞只发生在“本步本来存在可继续水平外溢”的情形；没有可达坠口时仍按普通静止处理。
# - solver 假设最终状态是当前动力学的不动点；生成器在返回前会用 `variant_metrics()` 重放三套物理
#   （融合版、无批次版、无摩擦版），同时保证融合版自身重放不再变化。


_BASE_TEMPLATE = {
    "guide_width": 4,
    "red_width": 4,
    "bottom_rows": (
        "5533000000",
        "0033304000",
        "0033344440",
    ),
}

_SOURCE_SPECS = (
    {"blue_width": 3, "blue_height": 1},
    {"blue_width": 3, "blue_height": 2},
    {"blue_width": 4, "blue_height": 1},
    {"blue_width": 4, "blue_height": 2},
)


def _water_mask(arr: np.ndarray) -> np.ndarray:
    return np.isin(arr, WATER_COLORS)


def _water_points(arr: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(_water_mask(arr))]


def _color_points(arr: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(arr == int(color))]


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
            r0, c0 = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r0 + dr), int(c0 + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _assert_supported(arr: np.ndarray, cells: list[tuple[int, int]], *, label: str) -> None:
    h, w = arr.shape
    for r, c in cells:
        assert r == h - 1 or int(arr[r + 1, c]) != BG, f"{label} water must be supported"
        left_bg = c > 0 and int(arr[r, c - 1]) == BG
        right_bg = c + 1 < w and int(arr[r, c + 1]) == BG
        assert not (left_bg and right_bg), f"{label} water cannot have bg on both horizontal sides"


def _check_input_structure(g: np.ndarray) -> None:
    assert g.shape == GRID_SHAPE, "grid must stay 10x10"
    values = set(map(int, np.unique(g)))
    allowed = {BG, BLUE, RED, ROUGH, SMOOTH, GUIDE}
    assert values <= allowed, f"unexpected colors: {sorted(values - allowed)}"
    assert np.any(_water_mask(g)), "expected at least one water pixel"
    assert np.any(g == int(ROUGH)), "expected rough wall color"
    assert np.any(g == int(SMOOTH)), "expected smooth wall color"

    _assert_supported(g, _water_points(g), label="input")

    base = np.array(g, dtype=int).copy()
    for color in WATER_COLORS:
        base[base == int(color)] = BG
    for color in WATER_ORDER:
        cells = _color_points(g, color)
        if not cells:
            continue
        probe = np.array(base, dtype=int).copy()
        for r, c in cells:
            assert int(probe[r, c]) == BG, "water source overlaps structure"
            probe[r, c] = int(color)
        _assert_supported(probe, cells, label=f"batch-{color}")

    h, w = g.shape
    bottom_colors = set(map(int, g[h - 1]))
    guide_candidates = sorted(set(map(int, g[:7].flatten())) - {BG} - set(WATER_COLORS) - bottom_colors)
    for color in guide_candidates:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must stay 4-connected"


def _settle(
    g: np.ndarray,
    r: int,
    c: int,
    *,
    color: int,
    friction_enabled: bool,
    metrics: dict[str, int] | None,
    record_metrics: bool,
) -> tuple[int, int]:
    h, w = g.shape
    rr = int(r)
    cc = int(c)
    for _ in range(h * w):
        if rr + 1 < h and int(g[rr + 1, cc]) == BG:
            rr += 1
            continue

        opts: list[tuple[int, int]] = []
        if (
            rr + 1 < h
            and cc - 1 >= 0
            and int(g[rr, cc - 1]) == BG
            and int(g[rr + 1, cc - 1]) == BG
        ):
            opts.append((int(rr + 1), int(cc - 1)))
        if (
            rr + 1 < h
            and cc + 1 < w
            and int(g[rr, cc + 1]) == BG
            and int(g[rr + 1, cc + 1]) == BG
        ):
            opts.append((int(rr + 1), int(cc + 1)))
        if opts:
            assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
            rr, cc = opts[0]
            continue

        if rr + 1 < h and int(g[rr + 1, cc]) != BG:
            left = int(cc)
            while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                left -= 1
            right = int(cc)
            while right + 1 < w and int(g[rr, right + 1]) == BG:
                right += 1

            drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == BG]
            support_color = int(g[rr + 1, cc])
            if friction_enabled and support_color == int(ROUGH) and drops:
                if record_metrics and metrics is not None:
                    key = "blue_rough_blocks" if int(color) == int(BLUE) else "red_rough_blocks"
                    metrics[key] += 1
                break
            if not drops:
                break

            min_dist = min(abs(int(x - cc)) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            best = nearest[0]
            assert best != int(cc)
            step = -1 if best < int(cc) else 1
            nx = int(cc + step)
            assert int(g[rr, nx]) == BG
            if record_metrics and metrics is not None:
                metrics["smooth_slides"] += 1
            cc = nx
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")
    return int(rr), int(cc)


def _simulate(
    grid: np.ndarray,
    *,
    batch_mode: bool,
    friction_enabled: bool,
    check_input: bool,
    check_output: bool,
    collect_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g0 = np.array(grid, dtype=int).copy()
    if check_input:
        _check_input_structure(g0)

    metrics = None
    if collect_metrics:
        metrics = {
            "blue_rough_blocks": 0,
            "red_rough_blocks": 0,
            "smooth_slides": 0,
        }

    input_counts = {color: int(np.count_nonzero(g0 == int(color))) for color in WATER_COLORS}
    h, w = g0.shape

    if batch_mode:
        starts = {color: _color_points(g0, color) for color in WATER_ORDER}
        g = np.array(g0, dtype=int).copy()
        for color in WATER_COLORS:
            g[g == int(color)] = BG

        for color in WATER_ORDER:
            cells = starts[color]
            if not cells:
                continue
            for r, c in cells:
                assert int(g[r, c]) == BG, "water source overlaps settled structure"
                g[r, c] = int(color)

            for _pass in range(h * w):
                moved = False
                pts = sorted(_color_points(g, color), reverse=True)
                for r, c in pts:
                    if int(g[r, c]) != int(color):
                        continue
                    g[r, c] = BG
                    rr, cc = _settle(
                        g,
                        int(r),
                        int(c),
                        color=int(color),
                        friction_enabled=friction_enabled,
                        metrics=metrics,
                        record_metrics=True,
                    )
                    moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
                    assert int(g[rr, cc]) == BG
                    g[rr, cc] = int(color)
                if not moved:
                    break
            else:
                raise AssertionError("batched water simulation did not converge")
    else:
        g = np.array(g0, dtype=int).copy()
        for _pass in range(h * w):
            moved = False
            pts = sorted(
                [(int(r), int(c), int(g[int(r), int(c)])) for r, c in np.argwhere(_water_mask(g))],
                reverse=True,
            )
            for r, c, color in pts:
                if int(g[r, c]) != int(color):
                    continue
                g[r, c] = BG
                rr, cc = _settle(
                    g,
                    int(r),
                    int(c),
                    color=int(color),
                    friction_enabled=friction_enabled,
                    metrics=metrics,
                    record_metrics=True,
                )
                moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
                assert int(g[rr, cc]) == BG
                g[rr, cc] = int(color)
            if not moved:
                break
        else:
            raise AssertionError("simultaneous water simulation did not converge")

    for color in WATER_COLORS:
        assert int(np.count_nonzero(g == int(color))) == int(input_counts[color]), "water count must be conserved"

    if check_output:
        replay = _simulate(
            g,
            batch_mode=batch_mode,
            friction_enabled=friction_enabled,
            check_input=False,
            check_output=False,
            collect_metrics=False,
        )
        assert isinstance(replay, np.ndarray)
        assert np.array_equal(replay, g), "output is not a fixed point of the configured dynamics"

    if metrics is None:
        return g
    return g, metrics


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    combo, combo_metrics = _simulate(
        grid,
        batch_mode=True,
        friction_enabled=True,
        check_input=True,
        check_output=True,
        collect_metrics=True,
    )
    no_batch = _simulate(
        grid,
        batch_mode=False,
        friction_enabled=True,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    no_friction = _simulate(
        grid,
        batch_mode=True,
        friction_enabled=False,
        check_input=True,
        check_output=False,
        collect_metrics=False,
    )
    assert isinstance(combo, np.ndarray)
    assert isinstance(combo_metrics, dict)
    assert isinstance(no_batch, np.ndarray)
    assert isinstance(no_friction, np.ndarray)

    blue_on_rough = sum(
        1
        for r, c in np.argwhere(combo == int(BLUE))
        if int(r) + 1 < combo.shape[0] and int(combo[int(r) + 1, int(c)]) == int(ROUGH)
    )
    red_source_combo = int(np.count_nonzero(combo[6] == int(RED)))
    red_source_no_batch = int(np.count_nonzero(no_batch[6] == int(RED)))
    red_source_no_friction = int(np.count_nonzero(no_friction[6] == int(RED)))

    return {
        "diff_batch": int(np.count_nonzero(combo != no_batch)),
        "diff_friction": int(np.count_nonzero(combo != no_friction)),
        "blue_on_rough": int(blue_on_rough),
        "red_source_combo": int(red_source_combo),
        "red_retention_gain": int(
            (red_source_combo - red_source_no_batch) + (red_source_combo - red_source_no_friction)
        ),
        "blue_moved": int(np.count_nonzero((grid == int(BLUE)) & (combo != int(BLUE)))),
        "red_moved": int(np.count_nonzero((grid == int(RED)) & (combo != int(RED)))),
        "blue_rough_blocks": int(combo_metrics["blue_rough_blocks"]),
        "red_rough_blocks": int(combo_metrics["red_rough_blocks"]),
        "smooth_slides": int(combo_metrics["smooth_slides"]),
    }


def _build_candidate(spec: dict[str, int], *, mirror: bool) -> np.ndarray:
    g = np.zeros(GRID_SHAPE, dtype=int)
    for offset, row in enumerate(_BASE_TEMPLATE["bottom_rows"], start=7):
        g[offset] = np.array([int(ch) for ch in row], dtype=int)

    guide_width = int(_BASE_TEMPLATE["guide_width"])
    red_width = int(_BASE_TEMPLATE["red_width"])
    blue_width = int(spec["blue_width"])
    blue_height = int(spec["blue_height"])

    g[5, :guide_width] = int(GUIDE)
    g[6, :red_width] = int(RED)
    g[5 - blue_height : 5, :blue_width] = int(BLUE)

    if mirror:
        g = np.fliplr(g).copy()
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate a two-color basin puzzle where blue water is poured first and wall color controls lateral friction.

    Input structure:
    - Background is 0, first-batch blue water is 1, and second-batch red water is 2.
    - Color 3 is a rough wall that cancels horizontal spill, color 4 is a smooth basin wall/floor, and color 5 is a solid guide shelf.
    - Water appears as supported source strips above a bottom-anchored basin chain; in the generated instances blue often starts higher than red.

    Rule:
    - Remove later-batch colors until their turn, settle every blue pixel to completion, then insert the red batch and settle it on the updated scene.
    - Within each batch, a pixel falls straight down through background, otherwise takes the unique open diagonal, otherwise slides one step along its supported row toward the nearest reachable drop.
    - If the current support tile is rough color 3, that horizontal search is cancelled immediately; water supported by smooth walls, guide shelves, or earlier water may still slide.
    - Preserve all colors and repeat passes until the active batch no longer changes.

    Key insight: The result is controlled by two coupled priorities at once: blue is resolved before red regardless of height, and rough ledges freeze some early blue in the spill path, so red later sees a different obstacle field than blue did.
    """
    out = _simulate(
        grid,
        batch_mode=True,
        friction_enabled=True,
        check_input=check_input,
        check_output=check_output,
        collect_metrics=False,
    )
    assert isinstance(out, np.ndarray)
    return out


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng
    specs = list(_SOURCE_SPECS)
    rng.shuffle(specs)

    for spec in specs:
        mirror_order = [False, True]
        rng.shuffle(mirror_order)
        for mirror in mirror_order:
            candidate = _build_candidate(spec, mirror=mirror)
            try:
                stats = variant_metrics(candidate)
            except AssertionError:
                continue
            if stats["diff_batch"] < 3:
                continue
            if stats["diff_friction"] < 6:
                continue
            if stats["blue_on_rough"] < 1:
                continue
            if stats["red_retention_gain"] < 4:
                continue
            if stats["blue_moved"] < 2:
                continue
            if stats["red_source_combo"] < 3:
                continue
            if stats["blue_rough_blocks"] < 1:
                continue
            return candidate
    raise AssertionError("failed to generate a non-degenerate D4/1+D11/1+D16/0 sample")
