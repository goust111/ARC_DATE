# Variant of 9aaea919
# Dimension: D1 — 匹配条件机制 (match_cond)
# Change: marker_pixels — 红色(2)标记条标识源车道，绿色(3)标记条标识目标车道，通过标记颜色直接指定源和目标... → 连接线匹配

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis
# a) Concrete behavior:
#    原题在第 29 行放 1x5 红/绿条来直接告诉 solver 哪些车道是 source / target。
#    这里取消条形标记，改为在最底部精灵下方画一条控制色连接结构：
#    - 第 29 行是水平导线。
#    - 每个 source 车道在其中心列下方有一根 1 像素宽尾巴：row28/29 的竖线。
#    - target 车道在其中心列下方有一个 3 像素宽箭头头部：row28 上连续 3 个像素，row29 中心列与导线相连。
#    因此输入里人眼能看到“1-2 个尾巴汇入一个箭头头部”，输出仍然是 source 变灰、target 自底向上补齐新精灵、控制结构消失。
#    具体追踪：原逻辑在“识别 source/target 车道”这一步读取红/绿条；新逻辑在同一步读取连接结构的尾巴/箭头。
#    后续灰化、计数守恒、向 target 车道补齐的规则保持不变。
# b) Complexity calibration:
#    仍然只保留 4 条固定车道、1 个 target、1-2 个 source，避免匹配自由度过大。
#    但新控制结构不再是颜色标签，而是带方向的几何结构，视觉差异明显，且 2-source 情况会出现分叉连接，足以支撑归纳。
# c) Hidden assumptions audit:
#    原 solver 依赖：
#    - 有红条和绿条，且它们本身给出车道列范围。
#    - 可以按 marker 颜色直接把 source / target 分开。
#    这些前提在变体里都失效，所以 main() 必须改成：
#    - 依赖固定车道布局常量来读取 4 条车道。
#    - 依赖控制色连通块的局部形状来区分 source 尾巴和 target 箭头。
#    同时避免了“看到颜色 2/3 就当 marker”的旧隐式假设；lane 颜色现在允许是 2 或 3。
# d) Placement legality:
#    连接结构只占用 row28-29，精灵只占到 row27，二者不会重叠。
#    4 条车道中心列固定为 5/11/17/23；target 箭头头部宽 3，始终落在该车道 5 列槽内部。
#    两个 source 时，共享 row29 导线并在 row28 分出两个尾巴，仍然是单个 4 连通控制块。
# e) Visual observability:
#    人类能直接看到：哪几条车道的底部接出“尾巴”，哪条车道下方有“箭头头部”，因此能推断转移方向是 tail -> arrowhead。
#    这比原题的最后一行彩条更显式地体现了 D1 变体。
# f) Visual neatness:
#    控制结构被压在网格最底两行，不与车道内部精灵混杂。
#    同一局面里只存在一个 target 箭头和 1-2 个 source 尾巴，画面整洁，不会出现重叠难辨的多层标记。
# g) Color inference:
#    控制色固定为 8，仅承担“连接结构”角色；lane 颜色从其余颜色中选取，并允许出现 2/3。
#    这样颜色变化规则仍由车道位置与连接方向支撑，而不是靠颜色 id 本身做不可见排序。
# h) New solver assumptions and generator guarantees:
#    - main() 假设恰有一个控制色连通块，且其中恰有一个 target 箭头、1-2 个 source 尾巴。
#    - generate_input() 只构造满足该形状约束的样本，并用 oracle `main(inp) == out` 做最终校验。


# --- CFG shared by solver and generator ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_SPRITE_H, _SPRITE_W = 3, 5
_BOTTOM_TOP = 25
_PERIOD = 4
_CONNECTOR = 8
_GREY = 5
_ARROW_ROW = _BOTTOM_TOP + _SPRITE_H
_SHAFT_ROW = _ARROW_ROW + 1

# --- generator palette / limits ---
_BG_CHOICES = [0, 1, 4, 6, 7, 9]
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _lane_center(lane_idx: int) -> int:
    return _LANE_STARTS[lane_idx] + 2


def _lane_sprite_components(g: np.ndarray, lane_idx: int, bg: int) -> list[list[tuple[int, int]]]:
    c0 = _LANE_STARTS[lane_idx]
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, c0 : c0 + _SPRITE_W] = True
    sprite_mask = (g != bg) & (g != _CONNECTOR) & colmask
    spriteish = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(sprite_mask, connectivity=4)]
    if not spriteish:
        return []
    sprite_sz = max(len(pts) for pts in spriteish)
    return [pts for pts in spriteish if len(pts) == sprite_sz]


def _sorted_sprites(sprites: list[list[tuple[int, int]]]) -> list[tuple[int, int, list[tuple[int, int]]]]:
    out = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        out.append((rmin, cmin, pts))
    out.sort()
    return out


def _lane_state(g: np.ndarray, lane_idx: int, bg: int) -> dict:
    sprites = _lane_sprite_components(g, lane_idx, bg)
    assert sprites, f"expected sprites in lane {lane_idx}"
    sprites_sorted = _sorted_sprites(sprites)
    dst_color = int(g[sprites_sorted[0][2][0][0], sprites_sorted[0][2][0][1]])
    tops = sorted({rmin for rmin, _cmin, _pts in sprites_sorted})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else _PERIOD
    else:
        r0, c0, r1, c1 = bbox_coords(np.array(sprites_sorted[0][2], dtype=int))
        period = int(r1 - r0 + 2)
    return {
        "lane_idx": lane_idx,
        "center": float(_lane_center(lane_idx)),
        "color": dst_color,
        "period": period,
        "occupied": set(tops),
        "bottom0": max(tops),
        "sprites": sprites,
        "sprites_sorted": sprites_sorted,
    }


def _connector_roles(g: np.ndarray) -> tuple[list[int], int, set[tuple[int, int]]]:
    ctrl_comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(g == _CONNECTOR, connectivity=4)]
    assert ctrl_comps, "expected a connector structure"

    matches = []
    for comp in ctrl_comps:
        sources = []
        target = None
        for lane_idx in range(_NUM_LANES):
            cc = _lane_center(lane_idx)
            if (_ARROW_ROW, cc) not in comp:
                continue
            head = {(_ARROW_ROW, cc - 1), (_ARROW_ROW, cc), (_ARROW_ROW, cc + 1)}
            if head.issubset(comp):
                assert target is None, "expected a single target arrowhead"
                target = lane_idx
            else:
                sources.append(lane_idx)
        if target is not None and sources:
            matches.append((sorted(sources), target, comp))

    assert len(matches) == 1, "expected exactly one valid connector component"
    sources, target, comp = matches[0]
    assert 1 <= len(sources) <= 2, "expected 1-2 source lanes"
    return sources, target, comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer sprites from connector-tailed source lanes into the connector-headed target lane.

    Input structure:
    - bg = most common color. Four fixed vertical lanes contain repeated monochrome sprites on a period-4 lattice.
    - A third-color connector sits in the bottom two rows: source lanes show thin tails; the destination lane shows a 3-pixel arrowhead.

    Rule:
    - Every tailed source lane donates all of its sprites: the source sprites turn grey (5).
    - The arrow-headed destination lane receives the same number of new sprites, stacked upward into empty lattice slots and recolored to that lane's color.
    - The connector structure disappears after the transfer.

    Key insight: The connector geometry, not red/green marker colors, identifies who sends and who receives; counts are conserved exactly as in the original lane-transfer task.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert (h, w) == (_H, _W), "expected the fixed 30x30 canvas"
    bg = get_bg(g)

    sources, target_idx, connector_pts = _connector_roles(g)
    lane_states = {lane_idx: _lane_state(g, lane_idx, bg) for lane_idx in range(_NUM_LANES)}
    assert target_idx not in sources, "target lane cannot also be a source"

    out = g.copy()
    target_state = lane_states[target_idx]

    for src_idx in sources:
        for pts in lane_states[src_idx]["sprites"]:
            for r, c in pts:
                out[r, c] = _GREY

    for src_idx in sources:
        src_state = lane_states[src_idx]
        src_ref_r0, _src_ref_c0, src_ref_pts = src_state["sprites_sorted"][0]
        dc_shift = int(round(target_state["center"] - src_state["center"]))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        n_new = len(src_state["sprites"])
        placed = 0
        k = 0
        while placed < n_new:
            new_top = target_state["bottom0"] - k * target_state["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in target_state["occupied"]:
                continue
            ok = True
            for dr, c in src_offsets:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < w):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            if not ok:
                continue
            for dr, c in src_offsets:
                out[new_top + dr, c] = target_state["color"]
            target_state["occupied"].add(new_top)
            placed += 1

    for r, c in connector_pts:
        out[r, c] = bg

    return out


# --------------- generator ---------------


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


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for target in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != target]
        for src_count in (1, 2):
            for sources in combinations(others, src_count):
                if counts[target] + sum(counts[i] for i in sources) <= _MAX_LANE_TOTAL:
                    grouped[src_count].append((sources, target))
    return grouped


def _draw_connector(grid: np.ndarray, sources: tuple[int, ...], target: int, bg: int) -> bool:
    tc = _lane_center(target)
    cells = {(_ARROW_ROW, tc - 1), (_ARROW_ROW, tc), (_ARROW_ROW, tc + 1), (_SHAFT_ROW, tc)}
    centers = [tc]
    for src_idx in sources:
        sc = _lane_center(src_idx)
        centers.append(sc)
        cells.add((_ARROW_ROW, sc))
        cells.add((_SHAFT_ROW, sc))
    for c in range(min(centers), max(centers) + 1):
        cells.add((_SHAFT_ROW, c))
    for r, c in cells:
        if not (0 <= r < _H and 0 <= c < _W):
            return False
        if int(grid[r, c]) != bg:
            return False
    for r, c in cells:
        grid[r, c] = _CONNECTOR
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY, _CONNECTOR)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_src_counts = [n for n in (1, 2) if feasible[n]]
    if not feasible_src_counts:
        return None

    src_count = int(rng.choice(feasible_src_counts))
    sources, target = feasible[src_count][int(rng.integers(len(feasible[src_count])))]
    source_set = set(sources)
    gain = sum(input_counts[i] for i in sources)

    output_counts = input_counts[:]
    output_counts[target] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in source_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in sources:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    target_c0 = _LANE_STARTS[target]
    for r0 in _lane_tops(output_counts[target])[:gain]:
        inp[r0 : r0 + _SPRITE_H, target_c0 : target_c0 + _SPRITE_W] = bg

    if not _draw_connector(inp, sources, target, bg):
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
