# Combo variant of 7ed72f31 (index 23)
# combo_spec: D2/2+D8/3+D11/0
# Name: 全局轴分阶段多精灵反射

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import get_bg, label_mask


"""
Design analysis for the combo variant
=====================================

a_concretization_and_trace:
  original_task:
    - 每个局部红标记自己定义轴/中心，并在原始输入上独立执行镜像。
    - 不同组彼此没有共享轴，也没有前后级联。
  combo_rule:
    - 唯一一条整行/整列红线是全局反射轴，所有精灵共享它。
    - 对于水平全局轴：先处理轴上方、再处理轴下方。
      对于竖直全局轴：先处理轴左侧、再处理轴右侧。
    - 一个阶段只反射“与该侧贴轴带相连”的 4 连通彩色精灵。
      贴轴带指轴旁边那一行/列：上侧是 axis-1，下侧是 axis+1；
      左侧是 axis-1，右侧是 axis+1。
    - 第二阶段读取第一阶段后的网格，因此第一阶段制造出的新像素
      可以把原本未贴轴的远侧精灵接到贴轴带上，从而激活它们。
  concrete_small_trace:
    - 例：第 8 行整行全红。
    - 上侧有一个蓝色 seed，含像素 (7,4),(6,5),(7,5),(7,6)，它贴着 axis-1。
      下侧有一个同色 receiver，含像素 (10,5),(11,5),(11,6)，它起初不贴轴。
    - 阶段 1（上侧）把 seed 镜到下侧，得到 (9,4),(10,5),(9,5),(9,6)。
      这些新像素与下侧 receiver 连成一个更大的蓝色连通块，并且触到 axis+1。
    - 阶段 2（下侧）反射这个放大后的下侧连通块，于是除了把阶段 1 的副本反回
      原位以外，还会把 receiver 映到上侧新增 (6,5),(5,5),(5,6) 中的高处像素。
    - 与“上下两侧都直接在原图上各做一次镜像”的并行旧逻辑相比，
      新逻辑会额外生成这些由 receiver 贡献的上侧像素。

b_complexity_calibration:
  - 只保留一条全局轴，避免重新引入局部红标记语义。
  - 阶段顺序固定为 上->下 / 左->右；否则人类难以从 2-4 个样例归纳顺序。
  - 每个样例至少放入 3 个共享轴精灵：
    1. 一个触发级联的第一阶段 seed
    2. 一个被激活的第二阶段 receiver
    3. 至少一个与级联无关但仍共享全局轴的独立精灵
  - 这样既能看出“多精灵共享一条轴”，也能看出“第二阶段读取第一阶段结果”。

c_implicit_assumption_audit_on_main:
  - 原代码假设红标记和 payload 在同一连通块内；这里不成立。
    新 solver 先全局检测唯一整行/整列红线。
  - 原代码按局部红几何区分 row/col/point；这里不再有 point，也没有局部轴。
  - 原代码所有组都读取原始输入；这里第二阶段必须读取第一阶段结果。
  - 新 solver 的关键新假设：
    1. 恰好一条完整红线存在。
    2. payload 都不是背景色也不是红色。
    3. 第一阶段镜像后的像素若与第二阶段侧原精灵接触，则颜色一致，不会冲突。
    4. 不存在不同颜色争夺同一目标格的情况。
    generator 会显式保证这些条件。

d_placement_legality:
  - 生成器只放一条完整红轴，禁止第二条完整红行/列。
  - 所有原始精灵与未来镜像结果都在边界内。
  - 组件之间用槽位分开，避免无意接触；只有预期的 seed->receiver 位置会接触。
  - 通过 solver 回放和指标筛选，拒绝任何退化或碰撞样本。

e_visual_observability:
  - 输入里整条红线很醒目，人类会自然把它当作全局镜轴。
  - 多个不同颜色精灵都围绕同一条轴活动，支持 D11 的“共享轴”归纳。
  - 某个远侧精灵初看并未贴轴，却在输出里也被镜回另一侧；
    只有“先做一侧，再用其结果激活另一侧”才能解释，支持 D8。

f_visual_cleanliness:
  - 不引入额外噪声或颜色映射。
  - 每个精灵都是小型单色 4 连通块，槽位之间保留空白。
  - 级联只发生在一个紧凑局部，其他精灵保持独立，图面可读。

g_color_inference:
  - 颜色不变，只做空间镜像。
  - 唯一特殊颜色仍是红色(2)，其语义仅为全局轴。
  - 只让一个共享颜色承担跨阶段级联，其余颜色只展示“共享轴但不级联”，
    这样既有对照，又不会把规则复杂度推得过高。
"""


MARK = 2
PALETTE = tuple(range(10))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _detect_global_axis(grid: np.ndarray) -> tuple[str, int]:
    g = np.array(grid, dtype=int)
    rows = [int(r) for r in range(g.shape[0]) if np.all(g[int(r), :] == int(MARK))]
    cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, int(c)] == int(MARK))]
    if len(rows) + len(cols) != 1:
        raise ValueError("expected exactly one full red separator line")
    if rows:
        return "row", int(rows[0])
    return "col", int(cols[0])


def _reflect(axis: tuple[str, int], r: int, c: int) -> tuple[int, int]:
    kind, pos = axis
    if kind == "row":
        return int(2 * int(pos) - int(r)), int(c)
    return int(r), int(2 * int(pos) - int(c))


def _side_name(axis: tuple[str, int], sign: int) -> str:
    if axis[0] == "row":
        return "top" if int(sign) < 0 else "bottom"
    return "left" if int(sign) < 0 else "right"


def _payload_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    mask = (grid != int(bg)) & (grid != int(MARK))
    lab, n = label_mask(mask, connectivity=4)
    comps: list[list[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if coords:
            comps.append(coords)
    return comps


def _touches_activation_lane(coords: Iterable[tuple[int, int]], axis: tuple[str, int], sign: int) -> bool:
    pos = int(axis[1])
    if axis[0] == "row":
        lane = int(pos + sign)
        return any(int(r) == int(lane) for r, _c in coords)
    lane = int(pos + sign)
    return any(int(c) == int(lane) for _r, c in coords)


def _on_side(coords: Iterable[tuple[int, int]], axis: tuple[str, int], sign: int) -> bool:
    pos = int(axis[1])
    if axis[0] == "row":
        vals = [int(r) for r, _c in coords]
    else:
        vals = [int(c) for _r, c in coords]
    if int(sign) < 0:
        return all(int(v) < int(pos) for v in vals)
    return all(int(v) > int(pos) for v in vals)


def _eligible_components(base: np.ndarray, axis: tuple[str, int], side_sign: int, bg: int) -> list[list[tuple[int, int]]]:
    comps = []
    for coords in _payload_components(base, bg):
        if not _on_side(coords, axis, int(side_sign)):
            continue
        if _touches_activation_lane(coords, axis, int(side_sign)):
            comps.append(coords)
    return comps


def _apply_stage(base: np.ndarray, axis: tuple[str, int], side_sign: int, bg: int) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    writes: dict[tuple[int, int], int] = {}
    for coords in _eligible_components(base, axis, int(side_sign), bg):
        for r, c in coords:
            v = int(base[int(r), int(c)])
            rr, cc = _reflect(axis, int(r), int(c))
            if not _in_bounds(base.shape, rr, cc):
                continue
            writes[(int(rr), int(cc))] = int(v)
    for (rr, cc), v in writes.items():
        out[int(rr), int(cc)] = int(v)
    return out


def _trace_execution(grid: np.ndarray) -> dict[str, np.ndarray | tuple[str, int]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    axis = _detect_global_axis(g)
    after_first = _apply_stage(g, axis, -1, bg)
    after_second = _apply_stage(after_first, axis, 1, bg)
    return {
        "input": g,
        "axis": axis,
        "after_first": after_first,
        "after_second": after_second,
    }


def _parallel_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    axis = _detect_global_axis(g)
    first = _apply_stage(g, axis, -1, bg)
    second = _apply_stage(g, axis, 1, bg)
    out = g.copy()
    out[first != g] = first[first != g]
    out[second != g] = second[second != g]
    return out


def _delta_coords(before: np.ndarray, after: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(np.array(before, dtype=int) != np.array(after, dtype=int))}


def variant_metrics(grid: np.ndarray) -> dict[str, int | str]:
    trace = _trace_execution(grid)
    g = np.array(trace["input"], dtype=int)
    axis = trace["axis"]
    after_first = np.array(trace["after_first"], dtype=int)
    after_second = np.array(trace["after_second"], dtype=int)
    bg = int(get_bg(g))

    second_on_raw = _apply_stage(g, axis, 1, bg)
    parallel = _parallel_baseline(g)

    return {
        "axis_kind": str(axis[0]),
        "first_side": _side_name(axis, -1),
        "second_side": _side_name(axis, 1),
        "first_stage_groups": int(len(_eligible_components(g, axis, -1, bg))),
        "second_stage_groups_raw": int(len(_eligible_components(g, axis, 1, bg))),
        "second_stage_groups_after_first": int(len(_eligible_components(after_first, axis, 1, bg))),
        "first_added": int(len(_delta_coords(g, after_first))),
        "second_added": int(len(_delta_coords(after_first, after_second))),
        "second_cascade_pixels": int(len(_delta_coords(after_first, after_second) - _delta_coords(g, second_on_raw))),
        "seq_parallel_delta": int(np.count_nonzero(after_second != parallel)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Global-axis staged shared-mirroring variant.

    1. Detect the unique full red row/column.
    2. Reflect every payload component touching the first-side activation lane.
    3. Recompute on the updated grid and reflect every payload component touching
       the opposite-side activation lane.

    For a row axis the order is top then bottom.
    For a column axis the order is left then right.
    """
    return np.array(_trace_execution(np.array(grid, dtype=int))["after_second"], dtype=int)


def _flip_lr(coords: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    max_c = max(int(c) for _r, c in pts)
    return [(int(r), int(max_c - c)) for r, c in pts]


def _transpose(coords: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(int(c), int(r)) for r, c in coords]


_CHAIN_TOP = [(-1, 0), (-2, 1), (-1, 1), (-1, 2)]
_CHAIN_BOTTOM = [(2, 1), (3, 1), (3, 2)]
_TOP_EXTRA = [(-1, 0), (-2, 0), (-1, 1)]
_BOTTOM_ACTIVE = [(1, 0), (1, 1), (2, 0)]


def _row_pattern_set(flip_lr: bool) -> dict[str, list[tuple[int, int]]]:
    def prep(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out = list(coords)
        if flip_lr:
            min_c = min(c for _r, c in out)
            shifted = [(int(r), int(c - min_c)) for r, c in out]
            flipped = _flip_lr(shifted)
            max_c = max(c for _r, c in shifted)
            return [(int(r), int(c + min_c)) for r, c in flipped if 0 <= int(c) <= int(max_c)]
        return list(out)

    return {
        "chain_top": prep(_CHAIN_TOP),
        "chain_bottom": prep(_CHAIN_BOTTOM),
        "top_extra": prep(_TOP_EXTRA),
        "bottom_active": prep(_BOTTOM_ACTIVE),
    }


def _oriented_pattern_set(axis_kind: str, flip_minor: bool) -> dict[str, list[tuple[int, int]]]:
    pats = _row_pattern_set(bool(flip_minor))
    if axis_kind == "row":
        return pats
    return {name: _transpose(coords) for name, coords in pats.items()}


def _stamp_component(
    grid: np.ndarray,
    axis: tuple[str, int],
    color: int,
    rel_coords: Iterable[tuple[int, int]],
    start_minor: int,
) -> None:
    pos = int(axis[1])
    for dr, dc in rel_coords:
        if axis[0] == "row":
            r = int(pos + dr)
            c = int(start_minor + dc)
        else:
            r = int(start_minor + dr)
            c = int(pos + dc)
        if not _in_bounds(grid.shape, r, c):
            raise ValueError("component placement escaped grid")
        grid[int(r), int(c)] = int(color)


def _minor_extent(axis_kind: str, shape: tuple[int, int]) -> int:
    return int(shape[1] if axis_kind == "row" else shape[0])


def _choose_slots(rng: random.Random, axis_kind: str, shape: tuple[int, int]) -> tuple[int, int, int]:
    extent = _minor_extent(axis_kind, shape)
    if extent < 18:
        raise ValueError("grid too small for slot layout")
    a_lo, a_hi = 1, max(1, extent // 4)
    b_lo, b_hi = max(5, extent // 3), min(extent - 3, max(5, extent // 3 + 2))
    c_lo, c_hi = max(10, 2 * extent // 3), min(extent - 3, max(10, 2 * extent // 3 + 2))
    a = rng.randint(int(a_lo), int(a_hi))
    b = rng.randint(int(b_lo), int(b_hi))
    c = rng.randint(int(c_lo), int(c_hi))
    if not (a + 4 <= b and b + 4 <= c):
        raise ValueError("slot spacing collapsed")
    return int(a), int(b), int(c)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    axis_kind = rng.choice(["row", "col"])
    if axis_kind == "row":
        h = rng.randint(18, 26)
        w = rng.randint(18, 26)
        axis_pos = rng.randint(5, h - 6)
    else:
        h = rng.randint(18, 26)
        w = rng.randint(18, 26)
        axis_pos = rng.randint(5, w - 6)

    bg = int(rng.choice([c for c in PALETTE if c != int(MARK)]))
    colors = [c for c in PALETTE if c not in (int(bg), int(MARK))]
    rng.shuffle(colors)
    chain_color, top_color, bottom_color = map(int, colors[:3])

    grid = np.full((int(h), int(w)), int(bg), dtype=int)
    axis = (axis_kind, int(axis_pos))
    if axis_kind == "row":
        grid[int(axis_pos), :] = int(MARK)
    else:
        grid[:, int(axis_pos)] = int(MARK)

    pats = _oriented_pattern_set(axis_kind, flip_minor=bool(rng.randint(0, 1)))
    chain_slot, top_slot, bottom_slot = _choose_slots(rng, axis_kind, grid.shape)

    _stamp_component(grid, axis, int(chain_color), pats["chain_top"], int(chain_slot))
    _stamp_component(grid, axis, int(chain_color), pats["chain_bottom"], int(chain_slot))
    _stamp_component(grid, axis, int(top_color), pats["top_extra"], int(top_slot))
    _stamp_component(grid, axis, int(bottom_color), pats["bottom_active"], int(bottom_slot))

    out = main(grid)
    meta = {
        "axis_kind": axis_kind,
        "axis_pos": int(axis_pos),
        "bg": int(bg),
        "chain_color": int(chain_color),
        "top_color": int(top_color),
        "bottom_color": int(bottom_color),
    }
    return grid, out, meta


def generate_input() -> np.ndarray:
    """
    Input-first generator for the combo variant.

    It builds a compact staged-cascade layout around one global red axis, then
    rejects any sample whose measured behavior collapses to the parallel baseline.
    """
    for _ in range(300):
        rng = random.Random()
        try:
            inp, out, _meta = _build_case(rng)
            metrics = variant_metrics(inp)
        except ValueError:
            continue
        if (
            np.array_equal(main(inp), out)
            and int(metrics["first_stage_groups"]) >= 2
            and int(metrics["second_stage_groups_raw"]) >= 1
            and int(metrics["second_stage_groups_after_first"]) >= 2
            and int(metrics["second_cascade_pixels"]) >= 2
            and int(metrics["seq_parallel_delta"]) >= 2
        ):
            return inp
    raise RuntimeError("failed to generate a non-degenerate combo variant sample")
