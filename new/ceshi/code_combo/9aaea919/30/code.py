# Combo variant of 9aaea919 (index 30)
# combo_spec: D1/4+D8/2+D12/2
# Name: 连接线+链式转移+1:1配对

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# a) Concrete behavior:
#    原题直接用红/绿条告诉 solver 哪条车道是 source / target，并且所有 source 一步并行汇入绿色车道。
#    本融合版改成三段单链：
#    - 恰好 1 条 source 车道 A、1 条 bridge 车道 B、1 条 final target 车道 C、1 条 idle 车道。
#    - 底两行不再有 marker bar，而是出现一条控制色连接链：
#      A 只有细尾巴；B 同时有箭头头部且导线从其两侧穿过；C 只有终点箭头头部。
#    - 例如 A/B/C 三条连续车道的输入计数若为 2/1/2，则输出变成：
#      A 原位 2 个精灵变灰；
#      第一步 A -> B，B 临时扩成 3 个本色精灵；
#      第二步 B -> C，B 的 3 个精灵全变灰，C 新增 3 个本色精灵，所以 C 变成 5 个。
#    与原题相比，变化不再只是“source 变灰 + target 增长”，而是“bridge 先长大再变灰”，
#    且 source/bridge/target 的身份全靠连接链的局部几何形状推断。
# b) Complexity calibration:
#    D12/2 要求严格 1:1，因此这里只保留 1 个 source 和 1 个 final target，避免多源配对歧义。
#    为了保留 D8/2 的链式复杂度，再额外固定 1 个 bridge 和 1 个 idle 车道，使样例仍明显区别于原题。
#    连接链只允许占据三条连续车道中的三种节点形态（tail-only / pass-through head / terminal head），
#    自由度足够体现变体，但不会多到让 2-4 个样例难以归纳。
# c) Hidden assumptions audit:
#    原 solver 的以下前提被破坏：
#    - “颜色 2/3 的最后一行条带就是角色标签”失效，lane 颜色现在允许取 2 或 3。
#    - “只需要给绿色车道建状态”失效，因为 bridge 也会在中途扩张并在第二步充当 source。
#    - “每个 source 直接找最终 target”失效，必须按 A->B->C 的顺序执行。
#    因此新 main() 改为：
#    - 使用固定 4 车道布局常量解析车道，不再从 marker bar 反推列范围。
#    - 使用控制色连通块在 row28/29 的节点形状识别 source / bridge / target。
#    - 先把 source 拷贝进 bridge，再把放大的 bridge 全量拷贝进 target。
# d) Placement legality:
#    精灵最高只占到 row27，连接链只使用 row28-29，天然不重叠。
#    三条活跃车道始终是相邻三车道，bridge 在中间，因此 row29 的导线是一条连续水平线，不会回折。
#    生成器强制满足：
#    - bridge_count + source_count <= 7，保证第一跳后 bridge 仍能完整堆叠；
#    - target_count + bridge_count + source_count <= 7，保证第二跳后 target 也能完整堆叠。
# e) Visual observability:
#    人类可以直接看见三类可观察线索：
#    - 尾巴节点对应 source；
#    - 中间带箭头且导线继续穿过的节点对应 bridge；
#    - 终点箭头对应 final target。
#    对照输出时，还能看到 bridge 输出比输入多一段灰色堆叠，而 target 获得的是“bridge 总量”而非仅 source 总量。
# f) Visual neatness:
#    控制结构全部压在底两行，且只使用一种控制色；四条车道横向间隔固定，线段永不交叉。
#    idle 车道完全不接线也不变色，能形成清晰对照，避免整张图所有车道都参与变化而显得拥挤。
# g) Change inferability:
#    颜色变化只依赖可见结构角色：A/B 输出都变灰，C 保持自身车道颜色并向上增殖。
#    连接色不参与排序；source/bridge/target 的推断只依赖节点形状和线是否继续延伸。
# h) New solver assumptions and generator guarantees:
#    - main() 假设恰有 1 个控制色连通块，且其中恰有 1 个 tail-only 节点、1 个 pass-through head、1 个 terminal head。
#    - main() 假设 bridge 位于 source 与 target 之间，row29 导线是连续的。
#    - main() 假设每条车道至少有 1 个标准 3x5 十字精灵，且 period-4 栈位规律成立。
#    - generate_input() 通过固定模式、计数上界和 oracle `main(inp) == out` 保证这些前提。


# --- CFG shared by solver and generator ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4
_CONNECTOR = 8
_GREY = 5
_NODE_ROW = _BOTTOM_TOP + _SPRITE_H
_WIRE_ROW = _NODE_ROW + 1

# --- generator palette / limits ---
_BG_CHOICES = [0, 1, 4, 6, 7, 9]
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300

# --- generator role patterns ---
_PATTERNS = [
    {"source": 0, "bridge": 1, "target": 2, "idle": 3},
    {"source": 2, "bridge": 1, "target": 0, "idle": 3},
    {"source": 1, "bridge": 2, "target": 3, "idle": 0},
    {"source": 3, "bridge": 2, "target": 1, "idle": 0},
]
_PATTERN_CURSOR = 0


def _lane_center(lane_idx: int) -> int:
    return _LANE_STARTS[lane_idx] + 2


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


def _erase_lane_tops(grid: np.ndarray, lane_idx: int, total_count: int, erase_count: int, bg: int) -> None:
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(total_count)[:erase_count]:
        grid[r0 : r0 + _SPRITE_H, c0 : c0 + _SPRITE_W] = bg


def _lane_sprite_components(g: np.ndarray, lane_idx: int, bg: int) -> list[list[tuple[int, int]]]:
    c0 = _LANE_STARTS[lane_idx]
    mask = np.zeros(g.shape, dtype=bool)
    mask[:, c0 : c0 + _SPRITE_W] = True
    sprite_mask = (g != bg) & (g != _CONNECTOR) & mask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(sprite_mask, connectivity=4)]
    if not comps:
        return []
    sprite_sz = max(len(pts) for pts in comps)
    return [pts for pts in comps if len(pts) == sprite_sz]


def _lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict:
    sprites = _lane_sprite_components(g, lane_idx, bg)
    assert sprites, f"expected sprites in lane {lane_idx}"

    sprites_sorted = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        sprites_sorted.append((rmin, cmin, pts))
    sprites_sorted.sort()

    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
    else:
        r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2], dtype=int))
        period = int(r1 - r0 + 2)

    c0 = _LANE_STARTS[lane_idx]
    ref_top, _ref_cmin, ref_pts = sprites_sorted[0]
    template_offsets = sorted((int(r - ref_top), int(c - c0)) for r, c in ref_pts)
    sprite_h = max(dr for dr, _dc in template_offsets) + 1
    color = int(g[ref_pts[0][0], ref_pts[0][1]])

    return {
        "lane_idx": lane_idx,
        "c0": c0,
        "center": float(_lane_center(lane_idx)),
        "color": color,
        "period": period,
        "bottom0": int(max(tops)),
        "occupied": set(int(t) for t in tops),
        "template_offsets": template_offsets,
        "sprite_h": sprite_h,
        "sprites": [pts for _rmin, _cmin, pts in sprites_sorted],
    }


def _grey_lane(out: np.ndarray, lane: dict) -> None:
    for top in sorted(lane["occupied"]):
        for dr, dc in lane["template_offsets"]:
            out[top + dr, lane["c0"] + dc] = _GREY


def _place_into_lane(
    out: np.ndarray,
    lane: dict,
    n_new: int,
    src_offsets: list[tuple[int, int]],
    dst_color: int,
    bg: int,
) -> None:
    h, w = out.shape
    placed = 0
    k = 1
    while placed < n_new:
        new_top = lane["bottom0"] - k * lane["period"]
        k += 1
        if new_top < 0:
            break
        if new_top + lane["sprite_h"] > h or new_top in lane["occupied"]:
            continue

        cells = [(new_top + dr, lane["c0"] + dc) for dr, dc in src_offsets]
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
            out[rr, cc] = int(dst_color)
        lane["occupied"].add(int(new_top))
        placed += 1

    if placed != n_new:
        raise AssertionError("failed to place all transferred sprites")


def _connector_roles(g: np.ndarray) -> tuple[int, int, int, set[tuple[int, int]]]:
    ctrl_comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(g == _CONNECTOR, connectivity=4)]
    assert len(ctrl_comps) == 1, "expected exactly one connector component"
    comp = ctrl_comps[0]

    source = None
    bridge = None
    target = None
    for lane_idx in range(_NUM_LANES):
        cc = _lane_center(lane_idx)
        has_center_node = (_NODE_ROW, cc) in comp and (_WIRE_ROW, cc) in comp
        has_head = {(_NODE_ROW, cc - 1), (_NODE_ROW, cc), (_NODE_ROW, cc + 1)}.issubset(comp)
        left_wire = (_WIRE_ROW, cc - 1) in comp
        right_wire = (_WIRE_ROW, cc + 1) in comp
        if not has_center_node:
            continue
        if not has_head and (left_wire ^ right_wire):
            assert source is None, "expected one tail-only source node"
            source = lane_idx
        elif has_head and left_wire and right_wire:
            assert bridge is None, "expected one pass-through bridge node"
            bridge = lane_idx
        elif has_head and (left_wire ^ right_wire):
            assert target is None, "expected one terminal target node"
            target = lane_idx

    assert source is not None and bridge is not None and target is not None, "failed to decode chain roles"
    sc = _lane_center(source)
    bc = _lane_center(bridge)
    tc = _lane_center(target)
    assert min(sc, tc) < bc < max(sc, tc), "bridge must lie between source and target"
    for c in range(min(sc, tc), max(sc, tc) + 1):
        assert (_WIRE_ROW, c) in comp, "expected a continuous bottom wire"
    return source, bridge, target, comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relay the source lane through a connector-marked bridge lane before filling the final target lane.

    Input structure:
    - The grid is a fixed 30x30 canvas with four 5-column lanes of evenly spaced monochrome plus-sprites on a period-4 stack.
    - No red/green marker bars remain. Instead, one control-color chain occupies the bottom two rows: a tail-only start node, a pass-through arrow node in the middle, and a terminal arrow node at the end.

    Rule:
    - The tail-only lane is the source A, the middle pass-through arrow lane is the bridge B, and the terminal arrow lane is the final target C.
    - First copy all of A's sprites into B using B's color and turn A grey. Then treat B's entire enlarged stack as the next source: turn B grey and copy that full count into C using C's color, stacked upward on the regular lattice.
    - The connector chain is erased in the output, while the idle fourth lane stays unchanged.

    Key insight:
    - The puzzle is no longer a one-hop marker lookup. You must read the directed connector geometry to recover the ordered chain A->B->C, then preserve counts across two sequential transfers so the final target gains source_count + bridge_count rather than only source_count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert (h, w) == (_H, _W), "expected the fixed 30x30 canvas"
    bg = get_bg(g)

    source_idx, bridge_idx, target_idx, connector_pts = _connector_roles(g)
    lane_states = {lane_idx: _lane_state(g, lane_idx, bg) for lane_idx in range(_NUM_LANES)}

    out = g.copy()

    source = lane_states[source_idx]
    bridge = lane_states[bridge_idx]
    target = lane_states[target_idx]

    _grey_lane(out, source)
    _place_into_lane(out, bridge, len(source["sprites"]), source["template_offsets"], bridge["color"], bg)

    _grey_lane(out, bridge)
    _place_into_lane(out, target, len(bridge["occupied"]), bridge["template_offsets"], target["color"], bg)

    for r, c in connector_pts:
        out[r, c] = bg

    return out


def _draw_chain_connector(grid: np.ndarray, source: int, bridge: int, target: int, bg: int) -> bool:
    sc = _lane_center(source)
    bc = _lane_center(bridge)
    tc = _lane_center(target)
    cells = {
        (_NODE_ROW, sc),
        (_WIRE_ROW, sc),
        (_NODE_ROW, bc - 1),
        (_NODE_ROW, bc),
        (_NODE_ROW, bc + 1),
        (_WIRE_ROW, bc),
        (_NODE_ROW, tc - 1),
        (_NODE_ROW, tc),
        (_NODE_ROW, tc + 1),
        (_WIRE_ROW, tc),
    }
    for c in range(min(sc, tc), max(sc, tc) + 1):
        cells.add((_WIRE_ROW, c))
    for r, c in cells:
        if not (0 <= r < _H and 0 <= c < _W):
            return False
        if int(grid[r, c]) != bg:
            return False
    for r, c in cells:
        grid[r, c] = _CONNECTOR
    return True


def _next_pattern() -> dict:
    global _PATTERN_CURSOR
    pattern = _PATTERNS[_PATTERN_CURSOR % len(_PATTERNS)]
    _PATTERN_CURSOR += 1
    return pattern


def _sample_counts(rng: np.random.Generator, pattern: dict) -> list[int]:
    while True:
        counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        source_count = counts[pattern["source"]]
        bridge_total = counts[pattern["bridge"]] + source_count
        target_total = counts[pattern["target"]] + bridge_total
        if bridge_total <= _MAX_LANE_TOTAL and target_total <= _MAX_LANE_TOTAL:
            return counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY, _CONNECTOR)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    pattern = _next_pattern()
    source = int(pattern["source"])
    bridge = int(pattern["bridge"])
    target = int(pattern["target"])

    input_counts = _sample_counts(rng, pattern)
    source_count = input_counts[source]
    bridge_total = input_counts[bridge] + source_count
    target_total = input_counts[target] + bridge_total

    output_counts = input_counts[:]
    output_counts[bridge] = bridge_total
    output_counts[target] = target_total

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx in (source, bridge):
            color = _GREY
        else:
            color = lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    _paint_lane(inp, source, lane_colors[source], input_counts[source], bg)

    _erase_lane_tops(inp, bridge, bridge_total, source_count, bg)
    _paint_lane(inp, bridge, lane_colors[bridge], input_counts[bridge], bg)

    _erase_lane_tops(inp, target, target_total, bridge_total, bg)

    if not _draw_chain_connector(inp, source, bridge, target, bg):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
