# Variant of 9aaea919
# Dimension: D8 — 多步处理与顺序 (multi_step)
# Change: 所有红色车道的精灵同时转移到最近的绿色车道，按精灵数叠加堆叠... → 多轮迭代转移

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D8 variant
# ==================================
# a) 具体化与行为追踪
#    - 保留原任务的 30x30 车道布局、3x5 十字精灵、底部细条标记、灰化规则。
#    - 为了让“逐步重算最近目标”真正可见，做最小必要适配：绿色标记条从 1 条改成 2 条。
#    - 红色车道不再一次性把本车道全部精灵送走，而是按“轮次”处理：
#      每一轮从左到右扫描所有红色车道，每条仍有剩余精灵的红色车道只转移 1 个精灵。
#    - “最近绿色车道”定义为：从当前红色标记条的中心，到每条绿色车道“下一空槽位中心”的
#      曼哈顿距离最小者。绿色车道每接收 1 个精灵，下一空槽位会上移 4 行，因此后续最近关系可能改变。
#    - 小例子：若红车道在中间、左右各有一条绿车道，左绿当前下一空槽更低，则前两次可能都流向左边；
#      但左边堆高后，它的下一空槽抬升，第三次就可能改流向右边。原任务的“一次性整车道转移”不会出现这种分流。
#
# b) 复杂度校准
#    - 只增加“2 条绿色车道 + 逐步重算”这一个关键自由度，不改精灵形状、颜色语义、车道宽度和堆叠周期。
#    - 绿色车道固定为 2 条，避免目标类别太多；红色车道维持 1-2 条，方便 2-4 个样例中看出顺序性。
#    - 生成器刻意保留至少一个会发生“同一红车道分流到两条绿车道”的案例，确保该维度不是表面改动。
#
# c) 原 solver 的隐式假设审查
#    - 原代码 `best = min(green_state.values(), key=lambda st: abs(st["center"] - rc))`
#      依赖“目标只由水平中心决定，且处理期间不会变化”。变体中此假设失效，因为目标取决于当前下一空槽位。
#    - 原代码按红车道逐条整批放置，默认“本车道所有新精灵的目标在本车道处理期间不变”。变体中失效。
#    - 原 `_find_marker_bars()` 的排序对原任务无关紧要；变体需要严格从左到右，因此必须显式按列排序。
#    - 原代码先整体灰化再整体放置在原任务中无害；变体中必须有“一个个接收并更新目标状态”的循环。
#
# d) 放置合法性
#    - 仍使用固定 4 行周期与底部对齐的槽位体系；下一空槽位总在该离散晶格上搜索。
#    - 生成器限制每条绿车道最终总数不超过 7，保证不会越界；solver 也会在放置前检查槽位为空。
#    - 绿色车道初始至少有 1 个精灵，这样“下一空槽位”的垂直位置在输入中可见，也便于人类推断。
#
# e) 视觉可观察性
#    - 排序和目标选择只依赖可见几何关系：左右位置、当前绿堆高度、底部标记条，不依赖颜色 id。
#    - 生成器拒绝“顺序版输出与非顺序版输出完全相同”的样本，避免退化回原行为。
#
# f) 可视化整洁度
#    - 仍是 4 条竖直车道，元素分层简单：车道精灵、底部标记条、背景。
#    - 新增的只是第二条绿色标记条，不引入额外装饰或重叠对象，输出仍然整洁。
#
# g) 颜色变化可推断性
#    - 颜色规则保持原样：红车道原地灰化，新增精灵使用目标绿车道原本的精灵颜色。
#    - 颜色本身不参与排序；它只继续承担“来源/目标/灰化结果”的语义角色。
#
# h) 生成器的量化目标
#    - 核心指标：是否存在至少一条红车道，其转移日志中同时命中两条不同绿色车道（lane split）。
#    - 辅助指标：顺序版最终绿色计数向量必须不同于“初始状态下一次性分配”的基线计数向量。
#
# i) main() 的新假设与 generator 保证
#    - 假设：绿色车道都已有可识别精灵、槽位周期一致、精灵形状跨车道一致。
#      生成器保证：每条车道初始 1-3 个同形单色精灵，周期固定为 4。
#    - 假设：标记条位于各车道底部且与精灵分离 1 行。
#      生成器保证：标记条固定在末行，底部精灵底行止于第 27 行。
#    - 假设：按列排序即可得到“从左到右”的处理顺序。
#      生成器保证：车道列区间固定且互不重叠。


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are thin 4-connected components of the given marker color.
    """
    bars = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            bars.append((0 if len(pts) == 5 else 1, int(r0), int(c0), pts))
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in bars]


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    m = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        cols_in = {int(g[r, c]) for r, c in pts}
        if len(cols_in) == 1:
            out.append(pts)
    return out


def _lane_sprites(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> list[list[tuple[int, int]]]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return []

    non_marker = [pts for pts in comps if len(pts) != len(marker_pts)]
    if not non_marker:
        return []

    sprite_sz = max(len(pts) for pts in non_marker)
    sprites = [pts for pts in non_marker if len(pts) == sprite_sz]
    sprites.sort(key=lambda pts: bbox_coords(np.array(pts))[:2])
    return sprites


def _lane_period(sprites: list[list[tuple[int, int]]]) -> int:
    tops = sorted({min(r for r, _ in pts) for pts in sprites})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0]))
    return int(r1 - r0 + 2)


def _next_open_top(state: dict, h: int) -> int | None:
    sprite_h = int(state["sprite_h"])
    period = int(state["period"])
    bottom0 = int(state["bottom0"])
    occupied = state["occupied"]

    k = 0
    while True:
        new_top = bottom0 - k * period
        k += 1
        if new_top < 0:
            return None
        if new_top + sprite_h > h:
            continue
        if new_top in occupied:
            continue
        return int(new_top)


def _can_place(out: np.ndarray, bg: int, top: int, offsets: list[tuple[int, int]]) -> bool:
    h, w = out.shape
    for dr, c in offsets:
        rr = int(top + dr)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(out[rr, cc]) != int(bg):
            return False
    return True


def _choose_green_lane(red_info: dict, green_infos: list[dict], out: np.ndarray, bg: int) -> tuple[dict, int, list[tuple[int, int]]]:
    candidates = []
    src_center = float(red_info["center"])
    src_row = int(red_info["anchor_row"])
    src_ref_r0 = int(red_info["src_ref_r0"])
    src_ref_pts = red_info["src_ref_pts"]

    for st in green_infos:
        top = _next_open_top(st, out.shape[0])
        if top is None:
            continue
        dc_shift = int(round(float(st["center"]) - src_center))
        offsets = [(int(r - src_ref_r0), int(c + dc_shift)) for r, c in src_ref_pts]
        if not _can_place(out, bg, top, offsets):
            continue

        dst_center_row = top + int(st["sprite_h"]) // 2
        dx = abs(src_center - float(st["center"]))
        dist = abs(src_row - dst_center_row) + dx
        candidates.append(((dist, dx, float(st["center"])), st, int(top), offsets))

    assert candidates, "expected at least one placeable green target"
    candidates.sort(key=lambda t: t[0])
    _key, st, top, offsets = candidates[0]
    return st, top, offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red-lane sprites into green lanes one token at a time, updating the nearest target after each move.

    Input structure:
    - bg is the dominant color. Thin red (2) and green (3) bottom bars mark source and destination lanes.
    - Each lane holds repeated identical monochrome sprites on the same vertical 4-row lattice as the original task.
    - This variant uses two green-marked lanes so the current stack heights can influence later routing decisions.

    Rule:
    - Iterate in rounds over the red lanes from left to right; each active red lane transfers exactly one sprite per visit.
    - The chosen destination is the green lane whose next empty stack slot is currently nearest to the red lane's marker center.
    - Every transferred source sprite is greyed in place, the new sprite is recolored to that destination lane's color, and all marker bars disappear at the end.

    Key insight: The solver must simulate the process step by step because each newly added green sprite raises that lane's next empty slot, which can redirect later transfers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"
    assert len(green_markers) >= 2, "expected at least two green lanes in the D8 variant"

    green_infos = []
    for gm in green_markers:
        sprites = _lane_sprites(g, gm, bg)
        assert sprites, "expected each green lane to contain visible sprites"

        tops = sorted({min(r for r, _ in pts) for pts in sprites})
        dst_color = int(g[sprites[0][0][0], sprites[0][0][1]])
        r0, c0, r1, c1 = bbox_coords(np.array(gm))
        green_infos.append(
            {
                "marker": gm,
                "center": (c0 + c1) / 2.0,
                "dst_color": dst_color,
                "period": _lane_period(sprites),
                "occupied": set(int(t) for t in tops),
                "bottom0": int(max(tops)),
                "sprite_h": int(max(r for r, _ in sprites[0]) - min(r for r, _ in sprites[0]) + 1),
                "marker_row": int(r0),
            }
        )
    green_infos.sort(key=lambda st: float(st["center"]))

    red_infos = []
    for rm in red_markers:
        sprites = _lane_sprites(g, rm, bg)
        if not sprites:
            continue

        sprites_top = sorted(sprites, key=lambda pts: bbox_coords(np.array(pts))[0])
        sprites_bottom = sorted(sprites, key=lambda pts: bbox_coords(np.array(pts))[0], reverse=True)
        mr0, mc0, mr1, mc1 = bbox_coords(np.array(rm))
        ref_pts = sprites_top[0]
        src_ref_r0 = min(r for r, _ in ref_pts)
        red_infos.append(
            {
                "marker": rm,
                "center": (mc0 + mc1) / 2.0,
                "anchor_row": int((mr0 + mr1) // 2),
                "queue": [list(pts) for pts in sprites_bottom],
                "src_ref_pts": list(ref_pts),
                "src_ref_r0": int(src_ref_r0),
            }
        )
    red_infos.sort(key=lambda st: float(st["center"]))
    assert red_infos, "expected at least one red lane with sprites"

    out = g.copy()

    while any(info["queue"] for info in red_infos):
        progressed = False
        for info in red_infos:
            if not info["queue"]:
                continue

            src_pts = info["queue"].pop(0)
            for r, c in src_pts:
                out[int(r), int(c)] = 5

            best, top, offsets = _choose_green_lane(info, green_infos, out, bg)
            for dr, c in offsets:
                out[int(top + dr), int(c)] = int(best["dst_color"])
            best["occupied"].add(int(top))
            progressed = True

        assert progressed, "transfer loop stalled unexpectedly"

    for pts in red_markers + green_markers:
        for r, c in pts:
            out[int(r), int(c)] = int(bg)

    return out


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 500
_GREEN_PAIRS = [(0, 2), (1, 3), (0, 3)]


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _lane_center(lane_idx: int) -> int:
    return int(_LANE_STARTS[lane_idx] + 2)


def _next_slot_top_from_count(count: int) -> int:
    return int(_BOTTOM_TOP - _PERIOD * count)


def _distance_to_next_slot(src_lane: int, dst_lane: int, dst_count: int) -> tuple[int, int, int]:
    src_row = _MARKER_ROW
    src_col = _lane_center(src_lane)
    dst_top = _next_slot_top_from_count(dst_count)
    dst_row = dst_top + 1
    dst_col = _lane_center(dst_lane)
    dx = abs(src_col - dst_col)
    dist = abs(src_row - dst_row) + dx
    return int(dist), int(dx), int(dst_col)


def _sequential_transfer_log(
    counts: list[int], reds: tuple[int, ...], greens: tuple[int, ...]
) -> tuple[dict[int, int], list[tuple[int, int]]]:
    green_counts = {int(g): int(counts[g]) for g in greens}
    remaining = {int(r): int(counts[r]) for r in reds}
    log: list[tuple[int, int]] = []

    while any(remaining[r] > 0 for r in reds):
        progressed = False
        for r in sorted(reds):
            if remaining[r] <= 0:
                continue

            cands = sorted(
                (
                    _distance_to_next_slot(int(r), int(g), int(green_counts[int(g)])),
                    int(g),
                )
                for g in greens
            )
            best_g = int(cands[0][1])
            green_counts[best_g] += 1
            if green_counts[best_g] > _MAX_LANE_TOTAL:
                raise ValueError("green lane overflow in sequential simulation")
            remaining[int(r)] -= 1
            log.append((int(r), best_g))
            progressed = True

        if not progressed:
            raise ValueError("sequential transfer simulation stalled")

    return green_counts, log


def _bulk_transfer_counts(
    counts: list[int], reds: tuple[int, ...], greens: tuple[int, ...]
) -> dict[int, int]:
    green_counts = {int(g): int(counts[g]) for g in greens}
    frozen_counts = {int(g): int(counts[g]) for g in greens}
    for r in sorted(reds):
        cands = sorted((_distance_to_next_slot(int(r), int(g), int(frozen_counts[int(g)])), int(g)) for g in greens)
        best_g = int(cands[0][1])
        green_counts[best_g] += int(counts[int(r)])
    return green_counts


def _lane_split_metric(log: list[tuple[int, int]], reds: tuple[int, ...]) -> int:
    split = 0
    for r in reds:
        touched = {g for rr, g in log if int(rr) == int(r)}
        if len(touched) >= 2:
            split += 1
    return int(split)


def _build_expected_output(
    bg: int,
    lane_colors: list[int],
    input_counts: list[int],
    reds: tuple[int, ...],
    greens: tuple[int, ...],
    final_green_counts: dict[int, int],
) -> np.ndarray:
    out = np.full((_H, _W), int(bg), dtype=int)
    red_set = set(int(r) for r in reds)
    green_set = set(int(g) for g in greens)

    for lane_idx in range(_NUM_LANES):
        if lane_idx in red_set:
            color = _GREY
            count = int(input_counts[lane_idx])
        elif lane_idx in green_set:
            color = int(lane_colors[lane_idx])
            count = int(final_green_counts[lane_idx])
        else:
            color = int(lane_colors[lane_idx])
            count = int(input_counts[lane_idx])
        _paint_lane(out, lane_idx, color, count, int(bg))

    return out


def _feasible_roles(counts: list[int]) -> list[tuple[tuple[int, ...], tuple[int, ...], dict[int, int], list[tuple[int, int]]]]:
    feasible = []
    for greens in _GREEN_PAIRS:
        others = [i for i in range(_NUM_LANES) if i not in greens]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total_red = sum(int(counts[i]) for i in reds)
                if total_red < 3:
                    continue
                if max(int(counts[i]) for i in reds) < 2:
                    continue

                try:
                    seq_counts, log = _sequential_transfer_log(counts, tuple(reds), tuple(greens))
                except ValueError:
                    continue

                bulk_counts = _bulk_transfer_counts(counts, tuple(reds), tuple(greens))
                if any(int(seq_counts[g]) > _MAX_LANE_TOTAL for g in greens):
                    continue
                if tuple(int(seq_counts[g]) for g in greens) == tuple(int(bulk_counts[g]) for g in greens):
                    continue
                if _lane_split_metric(log, tuple(reds)) <= 0:
                    continue

                feasible.append((tuple(reds), tuple(greens), seq_counts, log))
    return feasible


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    if not feasible:
        return None

    reds, greens, seq_counts, _log = feasible[int(rng.integers(len(feasible)))]
    expected = _build_expected_output(bg, lane_colors, input_counts, reds, greens, seq_counts)

    inp = np.full((_H, _W), int(bg), dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, int(lane_colors[lane_idx]), int(input_counts[lane_idx]), int(bg))

    for lane_idx in reds:
        c0 = _LANE_STARTS[int(lane_idx)]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2
    for lane_idx in greens:
        c0 = _LANE_STARTS[int(lane_idx)]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 3

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
