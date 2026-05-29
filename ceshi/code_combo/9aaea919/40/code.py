# Combo variant of 9aaea919 (index 40)
# combo_spec: D2/2+D3/0+D4/0+D10/0
# Name: 多形状+顶部堆叠+保持源色+可变周期

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# a) Concrete behavior and trace:
#    原题里四条车道都用同一种 3x5 十字精灵、同一个 period=4，并且绿色目标车道从底部往上吸收
#    红色源车道的数量，吸收后还会改成目标车道颜色。融合后同时改变四件事：
#    - 四条车道固定分配四种不同形状：plus / tee / ell / square；
#    - 每条车道各自使用 period 4 或 5；
#    - 目标车道补入新精灵时改为扫描“最高的空槽位优先”；
#    - 补入精灵保留源车道颜色，不再改成目标车道颜色。
#    具体例子：若 lane A 是红标记、颜色 6、tee 形、period 4、计数 2；lane C 是绿标记、颜色 8、square 形、
#    period 5、计数 2，则原题会把 A 的两个 tee 先变成灰，再在 C 的 period-4 底部上方补两个“颜色 8 的 tee”。
#    本融合版则把 A 变灰后，在 C 的 period-5 全部合法槽位里，从最上面的空槽开始补两个“颜色 6 的 tee”。
#    因此输出的目标车道同时混合了目标原生方块和源车道 tee，颜色也同时保留 8 与 6。
# b) Complexity calibration:
#    形状自由度只允许四种固定 3x5 模板，周期自由度只允许 {4, 5}，这样差异明显但不会爆炸。
#    目标车道仍然只有一个，红标记车道仍然只允许 1 或 2 条，所以“谁转移到哪”依旧容易看懂；
#    复杂度主要集中在四个可见线索上：形状保留、源色保留、顶部优先、目标周期继承。
# c) Hidden assumptions audit:
#    原 solver 的三个关键前提都被破坏了：
#    - `max(len(component))` 不再能识别精灵，因为四种形状面积分别是 7/9/11；
#    - `best["dst_color"]` 不再是新精灵颜色来源，颜色现在来自当前 source lane；
#    - `bottom0 - k * period` 不再是正确的放置顺序，因为现在必须重建目标车道全部槽位并从顶部往下扫。
#    因此新 main() 改成：
#    - 在每条车道内按“归一化形状签名”分组，选重复出现的签名作为该车道精灵类型；
#    - 目标车道只用自己的可见 top rows 推断 period，并恢复完整槽位序列；
#    - 每个 source lane 拿自己的模板偏移和自己的颜色，按左到右批次写入目标空槽。
# d) Placement legality:
#    所有形状都限制在同一个 3x5 lane slot 内，底部 marker 仍在 row 29，精灵最高只占到 row 27。
#    生成器固定 bottom sprite top row=25，并按 period 4 或 5 构造合法槽位，因此同车道精灵不会重叠。
#    另外强制 `green_count + sum(red_counts) < lane_capacity(green_period)`，保证顶部堆叠与原题底部堆叠
#    在同一输入上确实会给出不同位置，而不是把整条目标车道填满后看起来一样。
# e) Visual observability:
#    人类可直接看到四类线索：
#    - 红/绿条仍只负责 source / destination 身份；
#    - 目标输出中同时出现原目标形状与源形状，说明 shape 来自 source；
#    - 目标输出中同时出现原目标颜色与源颜色，说明 color 也来自 source；
#    - 目标车道的新增位置贴在最上方空槽，而不是贴着原有底部块向上长。
#    为了让 D10 也可见，生成器要求所有被标记车道里至少出现两种 period，并让每条标记车道至少有 2 个输入精灵。
# f) Visual neatness:
#    仍保持原题四条规整竖车道和底部 1x5 marker bar，不额外引入覆盖层或辅助符号。
#    输出目标车道虽然会混合 2-3 种颜色和 2-3 种形状，但每个精灵占据独立槽位，形状之间不相交，仍然清晰。
# g) Change inferability:
#    颜色变化不依赖任何不可见属性，完全由“哪条红标记车道失去颜色、哪种颜色在绿车道重现”来支撑。
#    顶部堆叠也不依赖颜色编号或隐式排序，只依赖可见空槽几何位置。样本里若按原题逻辑执行，
#    目标新增 top rows 会落在另一组槽位上，因此解题者能通过 2-4 个样例排除原题规则。
# h) Generator strategy:
#    采用 input-first。直接采样四条车道的 shape / color / period / count 与红绿角色，再用 main() 做 oracle。
#    之后额外检查形状混合、颜色混合、top-down 与 bottom-up 结果不同、marked periods 同时含 4/5，
#    以避免“能跑通但退化成原题外观”的输入。
# i) New solver assumptions and generator guarantees:
#    - main() 假设每条被标记车道内部只有一种重复形状签名；生成器固定每车道只画一种形状。
#    - main() 假设目标车道至少能从可见 sprite tops 中推断出唯一 period；生成器让 green lane 输入至少 2 个 sprite。
#    - main() 假设 source 批次处理顺序是按 marker 从左到右；生成器按同样顺序理解多源目标的顶端分层。
#    - main() 假设目标槽位足够容纳所有转移；生成器通过容量约束和 oracle 校验保证这一点。


# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical lattice ---
_BOTTOM_TOP = 25
_PERIOD_CHOICES = (4, 5)

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_LANE_COLOR_POOL = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 2
_MAX_SPRITES = 3
_GEN_RETRIES = 400
_GEN_SCENARIOS: tuple[tuple[int, int], ...] = ((1, 4), (2, 4), (1, 5), (2, 4), (1, 5), (2, 4))
_GEN_CALL_COUNT = 0

# --- CFG: shape library ---
_SHAPE_MASKS: dict[str, tuple[str, str, str]] = {
    "plus": (".XXX.", "XXXXX", ".XXX."),
    "tee": ("XXXXX", "..X..", "..X.."),
    "ell": ("X....", "X....", "XXXXX"),
    "square": (".XXX.", ".XXX.", ".XXX."),
}
_SHAPE_KEYS = tuple(_SHAPE_MASKS)


def _lane_capacity(period: int) -> int:
    return (_BOTTOM_TOP // period) + 1


def _all_slot_rows(period: int) -> list[int]:
    return list(range(_BOTTOM_TOP % period, _BOTTOM_TOP + 1, period))


def _lane_tops(period: int, count: int) -> list[int]:
    return [_BOTTOM_TOP - period * k for k in range(count - 1, -1, -1)]


def _sprite_patch(shape_key: str, color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    for r, row in enumerate(_SHAPE_MASKS[shape_key]):
        for c, ch in enumerate(row):
            if ch == "X":
                patch[r, c] = color
    return patch


def _paint_lane_at_tops(
    grid: np.ndarray, lane_idx: int, shape_key: str, color: int, tops: list[int], bg: int
) -> None:
    patch = _sprite_patch(shape_key, color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, r0, c0, transparent=bg)


def _paint_lane(
    grid: np.ndarray, lane_idx: int, shape_key: str, color: int, count: int, bg: int, period: int
) -> None:
    _paint_lane_at_tops(grid, lane_idx, shape_key, color, _lane_tops(period, count), bg)


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the given marker color."""
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -r1, hh * ww, len(pts), pts))
    cands.sort()
    return [t[4] for t in cands]


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


def _component_signature(pts: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.array(pts, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
    return (int(r1 - r0 + 1), int(c1 - c0 + 1), norm)


def _sorted_sprites(groups: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    return sorted(groups, key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))


def _extract_lane_sprites(
    g: np.ndarray, cols: range, marker_pts: list[tuple[int, int]], bg: int
) -> list[list[tuple[int, int]]]:
    marker_set = set(marker_pts)
    grouped: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[list[tuple[int, int]]]] = {}
    for pts in _single_color_components_in_cols(g, cols, bg):
        kept = [p for p in pts if p not in marker_set]
        if not kept:
            continue
        sig = _component_signature(kept)
        grouped.setdefault(sig, []).append(kept)
    if not grouped:
        return []
    best_group = max(grouped.values(), key=lambda group: (len(group), len(group[0])))
    return _sorted_sprites(best_group)


def _lane_component_infos(
    g: np.ndarray, lane_idx: int, bg: int
) -> list[tuple[int, tuple[int, int, tuple[tuple[int, int], ...]], list[tuple[int, int]]]]:
    c0 = _LANE_STARTS[lane_idx]
    cols = range(c0, c0 + _SPRITE_W)
    infos = []
    for pts in _single_color_components_in_cols(g, cols, bg):
        color = int(g[pts[0][0], pts[0][1]])
        infos.append((color, _component_signature(pts), pts))
    return infos


def _infer_period(tops: list[int], sprite_h: int) -> int:
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b > a]
        if diffs:
            return int(np.gcd.reduce(diffs))

    candidates = [p for p in _PERIOD_CHOICES if all(((_BOTTOM_TOP - t) % p) == 0 for t in tops)]
    if len(candidates) == 1:
        return int(candidates[0])
    return int(sprite_h + 1)


def _lane_slot_rows(occupied_tops: list[int], period: int, sprite_h: int, h: int) -> list[int]:
    top0 = min(occupied_tops)
    while top0 - period >= 0:
        top0 -= period
    rows = []
    r = top0
    while r + sprite_h <= h:
        rows.append(int(r))
        r += period
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer source-lane sprites into the green lane while mixing shape, period, color, and fill-order changes.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) bars on the last row still mark source and destination lanes.
    - Each 5-column lane contains repeated monochrome 3x5 sprites, but the lane shapes differ, and each lane may follow period 4 or period 5 instead of sharing one global cadence.

    Rule:
    - Every sprite in a red-marked lane turns grey (5). The same number of copies is inserted into the nearest green-marked lane.
    - Inserted copies keep the source lane's exact shape and original color, but they must occupy the destination lane's own lattice and fill the highest empty slots first, proceeding downward.
    - If multiple red lanes feed the same target, they are processed left-to-right; afterwards all marker bars are erased.

    Key insight:
    - The destination lane now supplies only the spatial rhythm. Shape and color must travel from each source lane, while the placement order is determined by reconstructing the target lane's full slot ladder and scanning it from top to bottom.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = sorted(
        _find_marker_bars(g, 2), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1]
    )
    green_markers = sorted(
        _find_marker_bars(g, 3), key=lambda pts: bbox_coords(np.array(pts, dtype=int))[1]
    )
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_states = []
    for gm in green_markers:
        _r0, dst_c0, _r1, dst_c1 = bbox_coords(np.array(gm, dtype=int))
        dst_sprites = _extract_lane_sprites(g, range(dst_c0, dst_c1 + 1), gm, bg)
        if not dst_sprites:
            continue

        tops = sorted(min(r for r, _ in pts) for pts in dst_sprites)
        sprite_h = max(r for pts in dst_sprites for r, _ in pts) - min(r for pts in dst_sprites for r, _ in pts) + 1
        period = _infer_period(tops, sprite_h)
        green_states.append(
            {
                "center": (dst_c0 + dst_c1) / 2.0,
                "c0": int(dst_c0),
                "period": int(period),
                "occupied": set(int(t) for t in tops),
                "slots": _lane_slot_rows(tops, period, sprite_h, h),
            }
        )

    assert green_states, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        _r0, src_c0, _r1, src_c1 = bbox_coords(np.array(rm, dtype=int))
        src_sprites = _extract_lane_sprites(g, range(src_c0, src_c1 + 1), rm, bg)
        if not src_sprites:
            continue

        src_color = int(g[src_sprites[0][0][0], src_sprites[0][0][1]])
        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = _GREY

        src_center = (src_c0 + src_c1) / 2.0
        best = min(green_states, key=lambda st: abs(st["center"] - src_center))

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        src_offsets = sorted((int(r - src_ref_r0), int(c - src_c0)) for r, c in src_ref_pts)

        placed = 0
        need = len(src_sprites)
        for new_top in best["slots"]:
            if placed >= need:
                break
            if new_top in best["occupied"]:
                continue

            cells = [(new_top + dr, best["c0"] + dc) for dr, dc in src_offsets]
            ok = True
            for rr, cc in cells:
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if int(out[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for rr, cc in cells:
                out[rr, cc] = src_color
            best["occupied"].add(int(new_top))
            placed += 1

        assert placed == need, "failed to place all transferred sprites"

    for marker in red_markers + green_markers:
        for r, c in marker:
            out[r, c] = bg

    return out


def _topdown_added_tops(existing_count: int, gain: int, period: int) -> list[int]:
    occupied = set(_lane_tops(period, existing_count))
    empties = [r for r in _all_slot_rows(period) if r not in occupied]
    return empties[:gain]


def _bottomup_added_tops(existing_count: int, gain: int, period: int) -> list[int]:
    occupied = set(_lane_tops(period, existing_count))
    out = []
    for r in reversed(_all_slot_rows(period)):
        if r in occupied:
            continue
        out.append(int(r))
        if len(out) == gain:
            break
    return sorted(out)


def _feasible_roles(counts: list[int], periods: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        cap = _lane_capacity(periods[green])
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                total = counts[green] + sum(counts[i] for i in reds)
                marked_periods = {periods[green], *(periods[i] for i in reds)}
                if total < cap and len(marked_periods) == 2:
                    grouped[red_count].append((tuple(sorted(reds)), green))
    return grouped


def _behavior_metrics(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    reds: tuple[int, ...],
    green: int,
    lane_shapes: list[str],
    lane_colors: list[int],
    counts: list[int],
    periods: list[int],
) -> dict[str, int]:
    bg = get_bg(inp)
    gain = sum(counts[i] for i in reds)
    top_added = _topdown_added_tops(counts[green], gain, periods[green])
    bottom_added = _bottomup_added_tops(counts[green], gain, periods[green])

    target_infos = _lane_component_infos(out, green, bg)
    target_shapes = {sig for _color, sig, _pts in target_infos}
    target_colors = {color for color, _sig, _pts in target_infos}
    out_target_tops = sorted(min(r for r, _ in pts) for _color, _sig, pts in target_infos)
    input_target_tops = _lane_tops(periods[green], counts[green])
    added_tops_from_output = sorted(set(out_target_tops) - set(input_target_tops))

    grey_ok = 1
    for lane_idx in reds:
        infos = _lane_component_infos(out, lane_idx, bg)
        if {color for color, _sig, _pts in infos} != {_GREY}:
            grey_ok = 0
            break

    return {
        "shape_mix": len(target_shapes),
        "color_mix": len(target_colors),
        "topdown_differs_from_bottomup": int(sorted(top_added) != sorted(bottom_added)),
        "topdown_realized": int(added_tops_from_output == sorted(top_added)),
        "marked_period_diversity": len({periods[green], *(periods[i] for i in reds)}),
        "grey_sources_ok": grey_ok,
        "expected_shape_mix": 1 + len(reds),
        "expected_color_mix": 1 + len(reds),
        "target_total": len(target_infos),
        "expected_target_total": counts[green] + gain,
        "source_shape_variety": len({lane_shapes[green], *(lane_shapes[i] for i in reds)}),
        "source_color_variety": len({lane_colors[green], *(lane_colors[i] for i in reds)}),
    }


def _gen_attempt(preferred_red_count: int, preferred_green_period: int) -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    lane_colors = rng.choice(
        [c for c in _LANE_COLOR_POOL if c != bg], size=_NUM_LANES, replace=False
    ).tolist()
    lane_shapes = rng.permutation(_SHAPE_KEYS).tolist()
    green = int(rng.integers(_NUM_LANES))
    others = [i for i in range(_NUM_LANES) if i != green]
    reds = tuple(sorted(int(x) for x in rng.choice(others, size=preferred_red_count, replace=False)))

    periods = rng.choice(_PERIOD_CHOICES, size=_NUM_LANES, replace=True).tolist()
    periods[green] = preferred_green_period
    contrast_period = 5 if preferred_green_period == 4 else 4
    varied_red = reds[int(rng.integers(len(reds)))]
    for lane_idx in reds:
        periods[lane_idx] = contrast_period if lane_idx == varied_red else int(rng.choice(_PERIOD_CHOICES))
    if len({periods[green], *(periods[i] for i in reds)}) != 2:
        return None

    counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    if preferred_red_count == 2:
        counts[green] = 2
        for lane_idx in reds:
            counts[lane_idx] = 2
    else:
        if preferred_green_period == 4:
            green_count, red_count = ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.integers(4))]
        else:
            green_count, red_count = ((2, 2), (2, 3), (3, 2))[int(rng.integers(3))]
        counts[green] = green_count
        counts[reds[0]] = red_count

    inp = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, lane_shapes[lane_idx], lane_colors[lane_idx], counts[lane_idx], bg, periods[lane_idx])

    green_c0 = _LANE_STARTS[green]
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3
    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    try:
        out = main(inp)
    except Exception:
        return None

    metrics = _behavior_metrics(
        inp,
        out,
        reds=reds,
        green=green,
        lane_shapes=lane_shapes,
        lane_colors=lane_colors,
        counts=counts,
        periods=periods,
    )
    if metrics["shape_mix"] != metrics["expected_shape_mix"]:
        return None
    if metrics["color_mix"] != metrics["expected_color_mix"]:
        return None
    if metrics["topdown_differs_from_bottomup"] != 1 or metrics["topdown_realized"] != 1:
        return None
    if metrics["marked_period_diversity"] != 2:
        return None
    if metrics["grey_sources_ok"] != 1:
        return None
    if metrics["target_total"] != metrics["expected_target_total"]:
        return None
    if metrics["source_shape_variety"] != metrics["expected_shape_mix"]:
        return None
    if metrics["source_color_variety"] != metrics["expected_color_mix"]:
        return None

    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred_red_count, preferred_green_period = _GEN_SCENARIOS[
        _GEN_CALL_COUNT % len(_GEN_SCENARIOS)
    ]
    _GEN_CALL_COUNT += 1
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(preferred_red_count, preferred_green_period)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
