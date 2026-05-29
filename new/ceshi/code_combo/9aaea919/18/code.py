# Combo variant of 9aaea919 (index 18)
# combo_spec: D8/0+D11/2
# Name: 多轮迭代+差值转移

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
#    - 保留原题的 30x30 画布、4 条固定竖直车道、3x5 十字精灵、底部 1x5 标记条和灰化语义。
#    - 为了让 D8 的“多轮重算目标”可见，绿色标记条固定为 2 条；其余 2 条就是红色标记条。
#    - 融合后的一次访问不再固定转移 1 个，也不再整条红车道清空。流程是：
#      每轮按从左到右扫描红车道；对当前红车道，先看哪条绿色车道的“下一空槽”最近，
#      再根据当前 `红剩余数 - 该绿当前数` 的差值决定本次要转多少，取 `ceil(diff/2)`，
#      让两边计数尽量向平衡靠近。
#    - 具体例子：若左红=5、右红=4、左绿=1、右绿=2，第一轮左红先看最近绿槽，若命中左绿，
#      则 diff=4，本次转 2，左红变 3、左绿变 3；随后右红再处理，也会基于更新后的绿堆高度重新选目标。
#      到第二轮时，左红可能已不再适合继续给左绿，而会改投右绿，这正是“多轮 + 重新评估”的核心。
#    - 与原题相比，差异不只在“分几步做”，也在“并非所有红精灵都移动”。输出里会同时出现：
#      部分红车道顶部变灰、底部仍保留原色，以及两条绿车道按轮次累积的新堆叠。
#
# b) 复杂度校准
#    - 关键自由度只保留两个：逐轮重算目标、以及按源/目标差值决定批量大小。
#    - 车道数仍是 4，避免目标组合爆炸；固定为 2 红 2 绿，使顺序性和分流性有足够支撑。
#    - 精灵计数上限从原题的 3 提到 5，这个扩展是刻意的：否则 `ceil(diff/2)` 几乎总是 1，
#      差值批量规则在像素层面不够显著。即便如此，最终绿色总数仍限制在 7 以内，不会越界。
#
# c) 原 solver 的隐式假设审查
#    - 原代码假设“每条红车道的全部精灵都会转移并灰化”。这里失效，因为只转顶部一部分。
#    - 原代码按红车道整批处理，且目标只按水平中心选一次。这里失效，因为目标依赖当前下一空槽，
#      而下一空槽会随每次接收而改变。
#    - D8 单变体里“每访问只转 1 个”的假设在融合后也失效；现在每次访问是差值驱动的可变批量。
#    - D11 单变体里“两个红车道互相比差”的假设在融合后被改写为“当前红车道与当前目标绿车道比差”。
#    - 因此 solver 必须显式维护：红车道剩余数、绿车道当前数、绿车道已占用槽位、以及左到右轮次顺序。
#
# d) 放置合法性
#    - 所有精灵仍落在原来的 period=4 竖直晶格上；新增精灵只会进入绿色车道的下一空槽。
#    - 生成器约束每条绿车道初始至少 1 个精灵，便于识别车道颜色与槽位周期。
#    - 生成器同时约束两条绿车道的最终总数都不超过 7；solver 放置前也逐个检查槽位为空且不越界。
#
# e) 视觉可观察性
#    - 规则只依赖人眼可见因素：左右位置、绿堆当前高度、以及各车道精灵数量。
#    - 为了防退化，生成器拒绝“看起来像普通 D8 顺序版”的样本，要求至少一条红车道在日志中命中两条不同绿车道。
#    - 还要求至少一次批量大小 >= 2，确保“差值决定转移量”不是口头存在而像素上不可见。
#
# f) 可视化整洁度
#    - 仍只有车道精灵和底部标记条两层对象；没有额外覆盖件。
#    - 部分灰化只发生在红车道顶部若干个完整精灵上，保留了底部对齐，输出不会显得凌乱。
#
# g) 颜色变化可推断性
#    - 颜色语义保持结构化：红标记只表示“待处理源车道”，绿标记表示“可接收目标车道”，灰色仍表示已转走的源精灵。
#    - 新增精灵继续使用目标绿车道原本的颜色，不引入任意颜色映射。
#
# h) 生成器的量化目标
#    - `moved_total < total_red`：证明不是“所有红精灵全转走”的原题 / D8 退化。
#    - `split_sources >= 1`：证明至少一条红车道因多轮重算而分流到两条绿车道。
#    - `max_batch >= 2`：证明差值规则确实控制了单次访问的转移量。
#
# i) main() 的新假设与 generator 保证
#    - 假设：恰有 2 条红车道和 2 条绿车道，且两条绿车道都已有可识别精灵。
#      生成器保证：4 条车道全部被标记为红或绿，绿色初始计数至少为 1。
#    - 假设：车道中精灵同形、单色、底部对齐、周期一致。
#      生成器保证：所有车道都用同一个 3x5 十字 patch，在相同 period=4 晶格上堆叠。
#    - 假设：如果某条红车道仍能转移，则至少有一条目标绿车道拥有合法空槽。
#      生成器保证：筛掉会导致目标溢出的配置，并用 oracle 校验 `main(inp) == expected`。


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Return thin marker bars of `marker_color`, ordered from left to right."""
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
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
    out = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _lane_sprites(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> list[list[tuple[int, int]]]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    sprite_comps = []
    for pts in comps:
        pts_wo_marker = [p for p in pts if p not in marker_set]
        if pts_wo_marker:
            sprite_comps.append(pts_wo_marker)
    if not sprite_comps:
        return []

    sprite_sz = max(len(pts) for pts in sprite_comps)
    sprites = [pts for pts in sprite_comps if len(pts) == sprite_sz]
    sprites.sort(key=lambda pts: bbox_coords(np.array(pts))[:2])
    return sprites


def _lane_period(sprites: list[list[tuple[int, int]]]) -> int:
    tops = sorted(min(r for r, _ in pts) for pts in sprites)
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0]))
    return int(r1 - r0 + 2)


def _next_open_top(state: dict[str, object], h: int) -> int | None:
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


def _transfer_batch_size(src_count: int, dst_count: int) -> int:
    diff = int(src_count) - int(dst_count)
    if diff <= 0:
        return 0
    return int((diff + 1) // 2)


def _choose_green_lane(
    red_info: dict[str, object], green_infos: list[dict[str, object]], out: np.ndarray, bg: int
) -> tuple[dict[str, object], list[tuple[int, int]], int] | None:
    remaining = len(red_info["queue"])
    candidates = []
    src_center = float(red_info["center"])
    src_row = int(red_info["anchor_row"])
    src_ref_r0 = int(red_info["src_ref_r0"])
    src_ref_pts = red_info["src_ref_pts"]

    for st in green_infos:
        batch = _transfer_batch_size(remaining, int(st["count"]))
        if batch <= 0:
            continue

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
        candidates.append(((dist, dx, float(st["center"])), st, offsets, batch))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    _key, st, offsets, batch = candidates[0]
    return st, offsets, int(batch)


def _place_batch(
    out: np.ndarray, bg: int, red_info: dict[str, object], green_info: dict[str, object], batch: int
) -> None:
    src_ref_r0 = int(red_info["src_ref_r0"])
    src_ref_pts = red_info["src_ref_pts"]
    dc_shift = int(round(float(green_info["center"]) - float(red_info["center"])))
    offsets = [(int(r - src_ref_r0), int(c + dc_shift)) for r, c in src_ref_pts]

    for _ in range(batch):
        top = _next_open_top(green_info, out.shape[0])
        assert top is not None, "expected an open slot in a green lane"
        assert _can_place(out, bg, top, offsets), "expected the green slot to be placeable"
        for dr, c in offsets:
            out[int(top + dr), int(c)] = int(green_info["dst_color"])
        green_info["occupied"].add(int(top))
        green_info["count"] = int(green_info["count"]) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red-lane sprites toward balance with green lanes by iterating left-to-right and recomputing targets after each visit.

    Input structure:
    - The background is the most common color. Two thin red (2) bars and two thin green (3) bars sit under the four lanes.
    - Every lane contains repeated identical monochrome plus sprites on a shared vertical period-4 lattice.
    - Green lanes already show their current stack heights; those heights define the next empty slot used for distance comparisons.

    Rule:
    - Process red lanes in rounds from left to right. On each visit, a red lane first chooses the currently nearest green lane, where
      nearness is measured to that green lane's next open stack slot.
    - Let `s` be the red lane's remaining colored sprite count and `t` the chosen green lane's current sprite count. Transfer
      `ceil((s - t) / 2)` sprites if `s > t`; otherwise transfer nothing from that lane in that state.
    - The transferred sprites are the topmost remaining sprites in the red lane: they turn grey in place, and equally many new
      sprites are stacked into the chosen green lane using the green lane's own color. After the iterative process stops, all markers vanish.

    Key insight: The solver must simulate mutable lane counts, not just detect source and destination once. Each visit can both
    change how many sprites a red lane still has and raise a green lane's next slot, so later visits may switch targets and move a different batch size.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert len(red_markers) == 2 and len(green_markers) == 2, "expected exactly two red and two green markers"

    green_infos = []
    for gm in green_markers:
        sprites = _lane_sprites(g, gm, bg)
        assert sprites, "expected each green lane to contain visible sprites"

        tops = sorted(min(r for r, _ in pts) for pts in sprites)
        r0, c0, _r1, c1 = bbox_coords(np.array(gm))
        green_infos.append(
            {
                "marker": gm,
                "center": (c0 + c1) / 2.0,
                "dst_color": int(g[sprites[0][0][0], sprites[0][0][1]]),
                "period": _lane_period(sprites),
                "occupied": set(int(t) for t in tops),
                "bottom0": int(max(tops)),
                "sprite_h": int(max(r for r, _ in sprites[0]) - min(r for r, _ in sprites[0]) + 1),
                "count": int(len(sprites)),
                "marker_row": int(r0),
            }
        )
    green_infos.sort(key=lambda st: float(st["center"]))

    red_infos = []
    for rm in red_markers:
        sprites = _lane_sprites(g, rm, bg)
        assert sprites, "expected each red lane to contain visible sprites"

        mr0, mc0, _mr1, mc1 = bbox_coords(np.array(rm))
        ref_pts = sprites[0]
        red_infos.append(
            {
                "marker": rm,
                "center": (mc0 + mc1) / 2.0,
                "anchor_row": int(mr0),
                "queue": [list(pts) for pts in sprites],
                "src_ref_pts": list(ref_pts),
                "src_ref_r0": int(min(r for r, _ in ref_pts)),
            }
        )
    red_infos.sort(key=lambda st: float(st["center"]))

    out = g.copy()

    while True:
        progressed = False
        for red_info in red_infos:
            if not red_info["queue"]:
                continue

            choice = _choose_green_lane(red_info, green_infos, out, bg)
            if choice is None:
                continue

            green_info, _offsets, batch = choice
            batch = min(int(batch), len(red_info["queue"]))
            if batch <= 0:
                continue

            moved = [red_info["queue"].pop(0) for _ in range(batch)]
            for pts in moved:
                for r, c in pts:
                    out[int(r), int(c)] = 5

            _place_batch(out, bg, red_info, green_info, batch)
            progressed = True

        if not progressed:
            break

    for pts in red_markers + green_markers:
        for r, c in pts:
            out[int(r), int(c)] = int(bg)

    return out


# --------------- generator ---------------

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
_MAX_SPRITES = 5
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 1200
_GREEN_PAIRS = [(0, 2), (1, 3), (0, 3), (0, 1), (1, 2), (2, 3)]


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
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


def _simulate_combo_counts(
    counts: list[int], reds: tuple[int, int], greens: tuple[int, int]
) -> dict[str, object]:
    red_counts = {int(r): int(counts[r]) for r in reds}
    green_counts = {int(g): int(counts[g]) for g in greens}
    log: list[dict[str, int]] = []
    round_idx = 0

    while True:
        round_idx += 1
        progressed = False
        for r in sorted(int(x) for x in reds):
            src_count = int(red_counts[r])
            candidates = []
            for g in sorted(int(x) for x in greens):
                dst_count = int(green_counts[g])
                if src_count <= dst_count:
                    continue
                if dst_count >= _MAX_LANE_TOTAL:
                    continue
                next_top = _next_slot_top_from_count(dst_count)
                if next_top < 0:
                    continue
                candidates.append((_distance_to_next_slot(r, g, dst_count), g))
            if not candidates:
                continue

            candidates.sort(key=lambda t: t[0])
            dst = int(candidates[0][1])
            dst_before = int(green_counts[dst])
            batch = min(_transfer_batch_size(src_count, dst_before), src_count, _MAX_LANE_TOTAL - dst_before)
            if batch <= 0:
                continue

            red_counts[r] = src_count - batch
            green_counts[dst] = dst_before + batch
            log.append(
                {
                    "round": int(round_idx),
                    "src": int(r),
                    "dst": int(dst),
                    "batch": int(batch),
                    "src_before": int(src_count),
                    "dst_before": int(dst_before),
                }
            )
            progressed = True

        if not progressed:
            break

    moved_per_red = {int(r): 0 for r in reds}
    touched = {int(r): set() for r in reds}
    max_batch = 0
    for evt in log:
        src = int(evt["src"])
        dst = int(evt["dst"])
        batch = int(evt["batch"])
        moved_per_red[src] += batch
        touched[src].add(dst)
        max_batch = max(max_batch, batch)

    split_sources = sum(1 for r in reds if len(touched[int(r)]) >= 2)
    return {
        "red_counts": red_counts,
        "green_counts": green_counts,
        "moved_per_red": moved_per_red,
        "log": log,
        "rounds": int(max((evt["round"] for evt in log), default=0)),
        "moved_total": int(sum(moved_per_red.values())),
        "split_sources": int(split_sources),
        "max_batch": int(max_batch),
    }


def _build_expected_output(
    bg: int,
    lane_colors: list[int],
    input_counts: list[int],
    reds: tuple[int, int],
    greens: tuple[int, int],
    sim: dict[str, object],
) -> np.ndarray:
    out = np.full((_H, _W), int(bg), dtype=int)
    red_set = set(int(r) for r in reds)
    green_set = set(int(g) for g in greens)
    moved_per_red = sim["moved_per_red"]
    green_counts = sim["green_counts"]

    for lane_idx in range(_NUM_LANES):
        if lane_idx in green_set:
            _paint_lane(out, lane_idx, int(lane_colors[lane_idx]), int(green_counts[lane_idx]), int(bg))
        else:
            _paint_lane(out, lane_idx, int(lane_colors[lane_idx]), int(input_counts[lane_idx]), int(bg))
            if lane_idx in red_set:
                grey_n = int(moved_per_red[lane_idx])
                if grey_n > 0:
                    _paint_tops(out, lane_idx, _GREY, _lane_tops(int(input_counts[lane_idx]))[:grey_n], int(bg))

    return out


def _feasible_roles(
    counts: list[int],
) -> list[tuple[tuple[int, int], tuple[int, int], dict[str, object]]]:
    feasible = []
    for greens in _GREEN_PAIRS:
        reds = tuple(i for i in range(_NUM_LANES) if i not in greens)
        sim = _simulate_combo_counts(counts, tuple(reds), tuple(greens))

        if any(int(sim["green_counts"][g]) > _MAX_LANE_TOTAL for g in greens):
            continue
        if int(sim["moved_total"]) <= 0:
            continue
        if int(sim["moved_total"]) >= sum(int(counts[r]) for r in reds):
            continue
        if int(sim["split_sources"]) < 1:
            continue
        if int(sim["max_batch"]) < 2:
            continue
        if len(sim["log"]) < 3:
            continue

        feasible.append((tuple(reds), tuple(greens), sim))
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

    reds, greens, sim = feasible[int(rng.integers(len(feasible)))]
    expected = _build_expected_output(bg, lane_colors, input_counts, reds, greens, sim)

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
