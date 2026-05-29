# Combo variant of 28a6681f (index 50)
# combo_spec: D8/1+D17/0
# Name: 禁止导轨+注水量编码
#
# Design analysis:
# a) 具体化与行为追踪
# - 公开画面仍保留原题的 10x10、底部并排盆地、侧边蓝色蓄水块这些核心元素，但彻底移除上方斜导轨。
#   上半区不再出现斜坡或漏斗，只剩一个贴边的受支撑水仓，以及若干从顶边垂下的 8 色编码条。
# - 每个盆地都有一个与之竖直对齐的顶边编码列；该列中 8 色像素的个数，就是该盆地在输出里应接收的
#   蓝色水像素数量。求解时不再追踪“这股水会沿哪条路流到哪儿”，而是先读数字，再做数量分配。
# - 具体小例子：
#   * 原题：左侧一大团水会先掉进左盆，再因溢流关系决定是否继续流向右盆。
#   * 本变体：仍然是左侧一大团水，但顶边若给出两列编码分别为 1 和 4，则输出必须让左盆只有 1 格水、
#     右盆有 4 格水，即使这与原本的局部重力直觉相反。
#   * 因而差异非常具体：原题靠路径决定分配；本变体靠边缘计数决定分配。
#
# b) 复杂度校准
# - 编码只用一种特殊颜色 8，而且只用“数像素个数”这一种读法，避免多套图例或颜色映射把任务复杂度
#   推高到不可读。
# - 盆地数量固定在 2-3 个，编码值固定在 1-5 的少量档位，水仓总量不超过 10。这样每个样例中会出现
#   重复的可比模式，解题者能从 2-4 组样例归纳出“顶边短条 = 目标水量”。
# - 为了避免退化成“只是读水位高度”，盆地宽度在 1 和 2 之间变化；因此同样是 4 个编码像素，不同盆地
#   可能表现成不同高度但相同总水量，强化“数量”而非“液面”语义。
#
# c) 隐式假设审查
# - 原 solver 的 guide 检查会把上半区一些非常规固体当成导轨候选；融合后顶边的 8 色编码条不是导轨，
#   因而原有 guide 识别逻辑必须整体移除。
# - 原 `settle()` 和主循环依赖“输出由局部物理唯一决定”。D17/0 直接破坏了这个前提：同一输入几何下，
#   不同编码会得到不同终态，所以不能再沿用原来的逐像素流动规则。
# - 原 `check_output` 假设每个水像素都满足局部静力学稳定性；本变体里这是次要条件，真正的约束是
#   “每个盆地的水量必须等于编码值”。因此输出校验改成计数校验。
# - 新的关键前提是：编码列与盆地一一对应、输入所有水都来自单侧水仓、盆地内部输入时为空。
#   这些假设都由 `generate_input()` 显式构造保证。
#
# d) 放置合法性
# - 底部盆地用若干从第 5 行延伸到底边的竖直墙列表示，彼此不重叠、不越界，源水棚也只占最左或最右两列。
# - 顶边编码列只画在盆地内部列上，并且只占 0-4 行；因此不会与盆地、源水棚或水仓重叠。
# - 先画固定结构与编码，再按总编码量放置水仓前缀块，最后用 `main()` 和对比指标验证，确保样本合法。
#
# e) 视觉可观察性
# - 融合效果在输入图上直接可见：没有任何导轨，取而代之的是顶边竖条编码；输出图上则看到水被分散到多个
#   盆地，且分配量匹配这些竖条，而不是匹配最近路径。
# - 编码列与盆地做竖直对齐，因此“哪个编码对应哪个盆地”不依赖颜色大小或隐藏顺序，肉眼即可配对。
# - 生成器强制选择与原始重力输出明显不同的编码向量，避免虽然加了标记但终态仍和原题一样的退化样本。
#
# f) 可视化整洁度
# - 画面按功能分成三层：顶边编码、侧边水仓、下半区盆地。不同区域互不覆盖，阅读路径清晰。
# - 颜色语义保持极少：0 背景，1 水，8 编码，另外 2-3 个普通结构色。不会出现多种标记挤在同一区域。
#
# g) 变化的可推断性
# - 这里没有不可见排序规则，也没有依赖颜色编号的映射。唯一需要解码的是“8 色条长度”，并且它和输出中的
#   每个盆地水量直接对应。
# - 盆地内部的填充顺序固定为自底向上、同行从左到右，这个规则在样例中会反复出现，足以支撑少量部分行案例。
#
# h) 退化检查指标
# - `gravity_gap`: 本融合输出与原始重力 solver 在同一输入上的像素差异数。要求至少为 3。
# - `count_gap`: 编码向量与原始重力落稳后的各盆地水量向量不能完全一致。
# - `nonlocal_quota`: 至少有一个离源更远的盆地得到的编码量大于更近的盆地，确保“编码主导分配”肉眼可见。
#
# i) main() 新增假设与 generator 保证
# - `main()` 假设每个盆地内部都有唯一编码列，且该列上的 8 色像素从顶边连续开始；generator 严格这样画。
# - `main()` 假设输入水只在单侧水仓里形成无孔前缀块；generator 用统一的前缀填充顺序放置水仓。
# - `main()` 假设总水量等于所有编码值之和，且每个编码值不超过 5；generator 只从满足这些约束的候选中采样。

from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
WATER = 1
MARKER = 8
H = 10
W = 10
WALL_TOP = 5
CODE_ROWS = 5
SOURCE_COLS = 2
SOURCE_HEIGHT = 5
SOLID_POOL = (2, 3, 4, 5, 6, 7, 9)


def _source_columns(source_side: str) -> list[int]:
    return [0, 1] if source_side == "left" else [8, 9]


def _source_cell_order(source_side: str) -> list[tuple[int, int]]:
    cols = [0, 1] if source_side == "left" else [9, 8]
    cells: list[tuple[int, int]] = []
    for r in range(SOURCE_HEIGHT - 1, -1, -1):
        for c in cols:
            cells.append((r, c))
    return cells


def _mirror_cols(cols: list[int]) -> list[int]:
    return sorted(W - 1 - int(c) for c in cols)


def _layout_options(source_side: str) -> list[list[int]]:
    base = [
        [2, 5, 8],
        [2, 4, 7],
        [2, 4, 6, 8],
    ]
    if source_side == "left":
        return [list(x) for x in base]
    return [_mirror_cols(list(x)) for x in base]


def _find_wall_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(W) if int(grid[H - 1, c]) not in (BG, WATER)]


def _parse_basins(grid: np.ndarray) -> list[dict[str, int]]:
    wall_cols = _find_wall_columns(grid)
    assert 3 <= len(wall_cols) <= 4, f"expected 2-3 basins, got wall columns {wall_cols}"
    basins: list[dict[str, int]] = []
    for left, right in zip(wall_cols, wall_cols[1:]):
        width = int(right - left - 1)
        assert width >= 1, "each basin needs at least one interior column"
        code_col = int(left + 1)
        assert code_col < right
        basins.append(
            {
                "left": int(left),
                "right": int(right),
                "width": width,
                "code_col": code_col,
                "capacity": width * (H - WALL_TOP),
            }
        )
    assert 2 <= len(basins) <= 3
    return basins


def _read_quota(grid: np.ndarray, basin: dict[str, int]) -> int:
    c = int(basin["code_col"])
    quota = 0
    seen_gap = False
    for r in range(CODE_ROWS):
        value = int(grid[r, c])
        if value == int(MARKER):
            assert not seen_gap, "marker column must be a top-anchored prefix"
            quota += 1
        else:
            seen_gap = True
    return quota


def _fill_basin(g: np.ndarray, basin: dict[str, int], amount: int) -> None:
    assert 0 <= amount <= int(basin["capacity"])
    placed = 0
    for r in range(H - 1, WALL_TOP - 1, -1):
        for c in range(int(basin["left"]) + 1, int(basin["right"])):
            if placed >= amount:
                return
            assert int(g[r, c]) == int(BG)
            g[r, c] = int(WATER)
            placed += 1
    assert placed == amount


def _count_basin_water(grid: np.ndarray, basins: list[dict[str, int]]) -> list[int]:
    counts: list[int] = []
    for basin in basins:
        sub = grid[WALL_TOP:H, int(basin["left"]) + 1 : int(basin["right"])]
        counts.append(int(np.count_nonzero(sub == int(WATER))))
    return counts


def _detect_source_side(grid: np.ndarray) -> str:
    left_count = int(sum(int(grid[r, c] == WATER) for r, c in _source_cell_order("left")))
    right_count = int(sum(int(grid[r, c] == WATER) for r, c in _source_cell_order("right")))
    if left_count > 0 and right_count == 0:
        return "left"
    if right_count > 0 and left_count == 0:
        return "right"
    raise AssertionError("expected exactly one non-empty side reservoir")


def _strip_all_water(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    g[g == int(WATER)] = int(BG)
    return g


def _render_by_quota(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[dict[str, int]]]:
    basins = _parse_basins(grid)
    quotas = [_read_quota(grid, basin) for basin in basins]
    total_quota = int(sum(quotas))
    total_water = int(np.count_nonzero(grid == int(WATER)))
    assert total_water == total_quota, "source water count must match encoded basin quotas"
    base = _strip_all_water(grid)
    out = np.array(base, dtype=int).copy()
    for basin, quota in zip(basins, quotas):
        _fill_basin(out, basin, int(quota))
    return out, quotas, basins


def _legacy_gravity_main(grid: np.ndarray, check_output: bool = True) -> np.ndarray:
    """
    Original local physics, kept only for generator-side degeneration checks.
    """
    g = np.array(grid, dtype=int).copy()

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                    right += 1
                drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break
                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                cc += -1 if best < int(cc) else 1
                continue

            break
        else:
            raise AssertionError("legacy water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("legacy water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Repack a side reservoir into bottom basins by reading explicit edge-encoded pour amounts.

    Input structure:
    - Background is 0 and water is color 1.
    - The lower half contains 2-3 rectangular bottom basins made from vertical wall columns.
    - There are no upper guide rails or sloped funnels; the only upper structures are a small side support shelf and top-edge marker columns.
    - Each basin owns one top-edge marker column made of color 8 pixels, aligned above that basin; the marker count is that basin's target water amount.
    - All input water starts in a single supported left or right side reservoir rather than already inside the basins.

    Rule:
    - Detect the basins from the bottom wall columns, then read each basin's quota by counting the contiguous 8-colored pixels hanging from the top edge in its marker column.
    - Count the total blue source water and verify it matches the sum of those quotas.
    - Remove the source water from the reservoir and refill the basins in place, one basin at a time, using each decoded quota.
    - Inside a basin, water occupies cells from the bottom upward, and within each row it fills from left to right.

    Key insight: The fused variant removes guide-driven path reasoning entirely; the decisive signal is now the visible edge encoding, so the task becomes quota decoding plus basin-wise count placement rather than local gravity simulation.
    """
    g = np.array(grid, dtype=int).copy()
    basins = _parse_basins(g)
    quotas = [_read_quota(g, basin) for basin in basins]

    if check_input:
        source_side = _detect_source_side(g)
        source_cells = _source_cell_order(source_side)
        source_count = int(sum(int(g[r, c] == int(WATER)) for r, c in source_cells))
        assert 1 <= source_count <= len(source_cells)
        source_mask = [int(g[r, c] == int(WATER)) for r, c in source_cells]
        assert all(source_mask[idx] == 1 for idx in range(source_count)), "source reservoir must be a filled prefix"
        assert all(source_mask[idx] == 0 for idx in range(source_count, len(source_mask))), "source reservoir may not contain holes"
        assert int(sum(quotas)) == source_count, "encoded water totals must match the side reservoir"
        for basin in basins:
            sub = g[WALL_TOP:H, int(basin["left"]) + 1 : int(basin["right"])]
            assert not np.any(sub == int(MARKER)), "markers must stay in the edge code area"
            assert not np.any(sub == int(WATER)), "input basin interiors must start empty"
        top_solids = {(int(r), int(c)) for r, c in np.argwhere((g[:WALL_TOP] != int(BG)) & (g[:WALL_TOP] != int(WATER)))}
        allowed_markers = {(int(r), int(basin["code_col"])) for basin in basins for r in range(_read_quota(g, basin))}
        assert top_solids <= allowed_markers, "upper half may only contain marker pixels, not guide rails"

    out, quotas, basins = _render_by_quota(g)

    if check_output:
        assert _count_basin_water(out, basins) == [int(x) for x in quotas]
        for r, c in np.argwhere(out == int(WATER)):
            assert r >= WALL_TOP, "all output water must lie in the basin band"
            assert any(int(basin["left"]) < int(c) < int(basin["right"]) for basin in basins), "water must stay inside some basin"
        assert not np.any(out[:WALL_TOP, :] == int(WATER)), "source reservoir must drain completely"

    return out


def _draw_structure(
    rng: random.Random,
    source_side: str,
    wall_cols: list[int],
) -> tuple[np.ndarray, list[dict[str, int]]]:
    grid = np.zeros((H, W), dtype=int)
    palette = rng.sample(list(SOLID_POOL), 3)
    wall_color = int(palette[0])
    shelf_color = int(palette[1])
    alt_wall_color = int(palette[2])

    for idx, c in enumerate(wall_cols):
        color = wall_color if idx % 2 == 0 else alt_wall_color
        grid[WALL_TOP:H, int(c)] = color

    grid[WALL_TOP, _source_columns(source_side)] = shelf_color
    basins = _parse_basins(grid)
    return grid, basins


def _place_source_block(grid: np.ndarray, count: int, source_side: str) -> None:
    cells = _source_cell_order(source_side)
    assert 1 <= count <= len(cells)
    for r, c in cells[:count]:
        assert int(grid[r, c]) == int(BG)
        grid[r, c] = int(WATER)


def _add_markers(grid: np.ndarray, basins: list[dict[str, int]], quotas: list[int]) -> None:
    for basin, quota in zip(basins, quotas):
        c = int(basin["code_col"])
        for r in range(int(quota)):
            assert int(grid[r, c]) == int(BG)
            grid[r, c] = int(MARKER)


def _quota_candidates(basin_count: int, source_side: str) -> list[tuple[int, ...]]:
    candidates: list[tuple[int, ...]] = []
    for quotas in itertools.product(range(1, CODE_ROWS + 1), repeat=basin_count):
        total = int(sum(quotas))
        if total > SOURCE_COLS * SOURCE_HEIGHT:
            continue
        if total < basin_count + 2:
            continue
        if len(set(quotas)) < 2:
            continue
        ordered = list(quotas) if source_side == "left" else list(reversed(quotas))
        nonlocal_quota = any(int(ordered[i + 1]) > int(ordered[i]) for i in range(len(ordered) - 1))
        if nonlocal_quota:
            candidates.append(tuple(int(x) for x in quotas))
    assert candidates, "need some nonlocal quota patterns"
    return candidates


def _sample_case(rng: random.Random) -> tuple[np.ndarray, dict[str, object]]:
    source_side = rng.choice(["left", "right"])
    wall_cols = list(rng.choice(_layout_options(source_side)))
    grid, basins = _draw_structure(rng, source_side, wall_cols)
    quotas = list(rng.choice(_quota_candidates(len(basins), source_side)))
    _add_markers(grid, basins, quotas)
    _place_source_block(grid, int(sum(quotas)), source_side)

    out = main(grid)
    legacy = _legacy_gravity_main(grid, check_output=False)
    gravity_gap = int(np.count_nonzero(out != legacy))
    legacy_counts = _count_basin_water(legacy, basins)
    count_gap = int(sum(abs(int(a) - int(b)) for a, b in zip(quotas, legacy_counts)))
    ordered = quotas if source_side == "left" else list(reversed(quotas))
    nonlocal_quota = any(int(ordered[i + 1]) > int(ordered[i]) for i in range(len(ordered) - 1))
    moved_into_multiple = int(sum(int(x > 0) for x in quotas))
    metrics = {
        "source_side": source_side,
        "quotas": tuple(int(x) for x in quotas),
        "gravity_gap": gravity_gap,
        "count_gap": count_gap,
        "legacy_counts": tuple(int(x) for x in legacy_counts),
        "nonlocal_quota": bool(nonlocal_quota),
        "active_basins": moved_into_multiple,
    }
    assert gravity_gap >= 3, f"degenerate gravity gap {gravity_gap}"
    assert count_gap >= 2, f"degenerate basin-count gap {count_gap}"
    assert moved_into_multiple >= 2, "at least two basins should receive water"
    return grid, metrics


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        try:
            grid, _metrics = _sample_case(rng)
            return grid
        except AssertionError:
            continue
    raise AssertionError("failed to generate a non-degenerate encoded-amount sample")
