# Combo variant of 64efde09 (index 9)
# combo_spec: D3/2+D5/1
# Name: 双重长度约束射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis
# ===============
# a) concretization_and_trace:
#    - 保留原题主干：多个 2xN / Nx2 的致密矩形精灵共享同一个模板，单像素标记仍然给活动边端口命名，
#      输出仍把同一个模板端口传播到所有同构精灵。
#    - 融合后的唯一实质变化是“每条复制射线的长度”不再只看一种因素，而是同时受两种可见量控制：
#      1) 模板长边长度 N；2) 标记与其宿主端口之间的背景间隙 gap。最终长度 = min(N, gap)。
#    - 小例子：若某个左朝向端口在列 10，模板长边 N=8：
#      * 标记在列 6，则 gap=3，原题会一直涂到边界，D3-only 会涂 3 格，D5-only 会涂 8 格，
#        融合版取 min(8, 3)=3，只涂列 9,8,7。
#      * 标记在列 -1 对应边界外不合法；若等价地放在同一射线上距端口 10 格处，则 gap=10，
#        D3-only 会涂 10 格，D5-only 会涂 8 格，融合版取 min(8, 10)=8。
#      同一题里只要同时出现 gap<N 和 gap>N，两条约束谁先起效就能被直接对比出来。
# b) complexity_calibration:
#    - 如果仍把所有标记固定在边界，很多样例会退化成“所有 gap 都远大于 N”，视觉上只剩 D5。
#      因此生成器允许标记位于射线中途，并强制每题至少一条短 gap 和至少一条长 gap。
#    - 3 个标记中固定采用“一短两长”的 regime：一条颜色真的被 gap 截短，两条颜色被 N 截短，
#      这样 N 规则有重复支撑，同时仍保留一个显眼的非 N 长度对照。
# c) implicit_assumption_audit:
#    - 原 `_first_marker_on_ray` 只返回颜色，默认“长度不重要”；融合后必须同时返回 gap。
#    - 原绘制循环 `while bg until blocked` 依赖旧停止条件；融合后必须改为精确写入 min(gap, N) 步。
#    - 新 solver 额外假设：同一个模板端口在所有可见宿主上测得的 gap 必须一致，否则同色端口无法共享长度；
#      generator 通过每个模板端口只放一个第一可见标记并最终调用 solver 自检来保证。
#    - 新 solver 还假设所有需要涂的前缀格在输入中都是背景；generator 在候选阶段就逐端口检查。
# d) placement_legality:
#    - 精灵仍保持 dense rectangle、互不重叠，并带 2 格 padding。
#    - 每个精灵朝外方向至少保留 N+3 格可用空间，这样既能放置 gap>N 的长标记，也保证 N 长射线可完整落下。
#    - 标记至少离端口 2 格，且必须是 8 连通下的孤立单像素；不同颜色的输出前缀格互不重叠。
# e) visual_observability:
#    - 解题者在单个样例里就能看到两种 regime：某些颜色的输出长度明显短于 N，并且正好等于标记前空格数；
#      另一些颜色虽然标记更远，但输出统一止于 N。
#    - 这保证融合结果同时不同于 D3-only（长 gap 会被截到 N）和 D5-only（短 gap 会短于 N）。
# f) visual_cleanliness:
#    - 生成器拒绝任何会让不同颜色射线重叠、穿过精灵、或让标记贴邻其它非背景像素的布局。
#    - 标记保留在线性走廊上，输出只产生细直线段，不会把大片边界或对象表面涂乱。
# g) color_inferability:
#    - 颜色语义不变：标记颜色就是对应端口复制出的颜色。
#    - 新信息只编码在“标记离端口有多远”与“模板长边有多长”这两个肉眼可见量里，不依赖颜色编号排序。


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"

    if h > w:
        sides = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        sides = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]

    scored = []
    for direction, coords in sides:
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique"
    return scored[0][1], scored[0][2]


def _first_marker_on_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[int, int] | None:
    H, W = g.shape
    gap = 0
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            marker_color = markers.get((r, c))
            if marker_color is None:
                return None
            return marker_color, gap
        gap += 1
        r += dr
        c += dc
    return None


def _template_length(sub: np.ndarray) -> int:
    return int(max(sub.shape))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate shared port labels across all transformed sprites, but let each ray stop at the earlier of two visible length limits.

    Input structure:
    - The dominant color is background.
    - Several dense N-by-2 or 2-by-N rectangles are dihedral variants of one rectangular template.
    - Isolated single-pixel markers sit on outward rays from some active-edge cells, and their distance from the sprite is meaningful.

    Rule:
    - On each sprite, the active edge is the long side with the stronger same-color majority.
    - The first marker seen on an outward active-edge ray identifies a shared template port and also determines a gap length: the number of background cells before that marker.
    - For every transformed copy of that port, emit the marker color for exactly min(N, gap) cells, where N is the template's long dimension.

    Key insight:
    - The input encodes two independent length cues at once: template size and marker distance.
    - A solver must match ports by template symmetry, measure both cues, and notice that the actual painted ray stops when the earlier constraint wins.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)

    sprites_raw: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) >= 2, "expected multiple transformed rectangular sprites"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"

    ray_len = _template_length(sprites_raw[0]["sub"])
    assert all(_template_length(sprite["sub"]) == ray_len for sprite in sprites_raw), "all sprites must share one template length"

    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites = []
    for sprite in sprites_raw:
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids, _name = matches[0]
        direction, active = _active_side(sprite["sub"])
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active})

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    color_to_gap: dict[int, int] = {}
    template_id_to_gap: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_info = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_info is None:
                continue
            marker_color, gap = marker_info
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            if marker_color in color_to_gap:
                assert color_to_gap[marker_color] == gap
            if template_id in template_id_to_gap:
                assert template_id_to_gap[template_id] == gap
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color
            color_to_gap[marker_color] = gap
            template_id_to_gap[template_id] = gap

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one active-edge cell"

    out = g.copy()
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the sprite active edge"

            emit_len = min(ray_len, template_id_to_gap[template_id])
            r = r0 + lr
            c = c0 + lc
            for _ in range(emit_len):
                r += dr
                c += dc
                assert 0 <= r < H and 0 <= c < W, "combined ray length must stay inside the grid"
                assert int(out[r, c]) in (bg, int(g[r, c]), color), "combined fixed-length rays must not conflict"
                out[r, c] = color

    return out


# --------------- generator ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_MARKER_GAP_MIN = 2
_LONG_GAP_MARGIN = 2
_PLACEMENT_RETRIES = 140
_MARKER_PLACEMENT_RETRIES = 160
_GEN_RETRIES = 500


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    right_majority = random.randint(max(2, N // num_colors), max_right_majority)
    right_dominant = random.choice(template_colors)
    others = [c for c in template_colors if c != right_dominant]
    if not others:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(template.flat)) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    result = []
    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        seen.append(sub)
        result.append((sub, ids_arr, direction, sorted(active_set)))
    return result


def _min_outward_run(ray_len: int) -> int:
    return int(ray_len + _LONG_GAP_MARGIN + 1)


def _random_sprite_position(
    H: int, W: int, h: int, w: int, dr: int, dc: int, min_run: int
) -> tuple[int, int] | None:
    if dr == 0 and dc == -1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = min_run, W - w
    elif dr == 0 and dc == 1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = 0, W - w - min_run
    elif dr == -1 and dc == 0:
        r_lo, r_hi = min_run, H - h
        c_lo, c_hi = 0, W - w
    else:
        r_lo, r_hi = 0, H - h - min_run
        c_lo, c_hi = 0, W - w

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _place_sprites(H: int, W: int, variants, ray_len: int) -> list[tuple] | None:
    placements: list[tuple] = []
    min_run = _min_outward_run(ray_len)
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            pos = _random_sprite_position(H, W, h, w, direction[0], direction[1], min_run)
            if pos is None:
                return None
            r0, c0 = pos
            occ = _occupied_set(placements)
            if any(
                (r0 + r, c0 + c) in occ
                for r in range(-_PLACEMENT_PAD, h + _PLACEMENT_PAD)
                for c in range(-_PLACEMENT_PAD, w + _PLACEMENT_PAD)
            ):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _outward_bg_count(H: int, W: int, r0: int, c0: int, sub: np.ndarray, direction: tuple[int, int]) -> int:
    h, w = sub.shape
    dr, dc = direction
    if dr == 0 and dc == -1:
        return c0
    if dr == 0 and dc == 1:
        return W - (c0 + w)
    if dr == -1 and dc == 0:
        return r0
    return H - (r0 + h)


def _marker_position(port_r: int, port_c: int, dr: int, dc: int, gap: int) -> tuple[int, int]:
    return port_r + dr * (gap + 1), port_c + dc * (gap + 1)


def _ray_cells_from_port(port_r: int, port_c: int, dr: int, dc: int, steps: int) -> list[tuple[int, int]]:
    return [(port_r + dr * step, port_c + dc * step) for step in range(1, steps + 1)]


def _ray_prefix_clear(grid: np.ndarray, port_r: int, port_c: int, dr: int, dc: int, gap: int) -> bool:
    H, W = grid.shape
    r, c = port_r, port_c
    for _ in range(gap):
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != _BG:
            return False
    r += dr
    c += dc
    return 0 <= r < H and 0 <= c < W and int(grid[r, c]) == _BG


def _isolated_singleton(grid: np.ndarray, r: int, c: int) -> bool:
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W) or int(grid[r, c]) != _BG:
        return False
    for nr in range(max(0, r - 1), min(H, r + 2)):
        for nc in range(max(0, c - 1), min(W, c + 2)):
            if (nr, nc) == (r, c):
                continue
            if int(grid[nr, nc]) != _BG:
                return False
    return True


def _tid_port(sprite_placement: tuple, tid: int) -> tuple[int, int, int, int] | None:
    r0, c0, _sub, ids_arr, direction, active_coords = sprite_placement
    locs = np.argwhere(ids_arr == tid)
    if locs.shape[0] != 1:
        return None
    lr, lc = map(int, locs[0])
    if (lr, lc) not in set(active_coords):
        return None
    return r0 + lr, c0 + lc, direction[0], direction[1]


def _analyze_tid_gap(
    grid: np.ndarray,
    placements: list[tuple],
    tid: int,
    gap: int,
    ray_len: int,
) -> dict | None:
    H, W = grid.shape
    emit_len = min(gap, ray_len)
    emit_cells: set[tuple[int, int]] = set()
    hosts = []
    for idx, sprite in enumerate(placements):
        port = _tid_port(sprite, tid)
        if port is None:
            return None
        port_r, port_c, dr, dc = port
        for rr, cc in _ray_cells_from_port(port_r, port_c, dr, dc, emit_len):
            if not (0 <= rr < H and 0 <= cc < W):
                return None
            if int(grid[rr, cc]) != _BG:
                return None
            if (rr, cc) in emit_cells:
                return None
            emit_cells.add((rr, cc))

        if not _ray_prefix_clear(grid, port_r, port_c, dr, dc, gap):
            continue
        marker_r, marker_c = _marker_position(port_r, port_c, dr, dc, gap)
        if not _isolated_singleton(grid, marker_r, marker_c):
            continue
        hosts.append({"sprite_index": idx, "marker": (marker_r, marker_c)})

    if not hosts:
        return None
    return {"tid": int(tid), "gap": int(gap), "emit_len": int(emit_len), "emit_cells": emit_cells, "hosts": hosts}


def _sample_gap_plan(ray_len: int, min_run: int) -> list[int] | None:
    short_choices = list(range(_MARKER_GAP_MIN, ray_len))
    long_choices = list(range(ray_len + 1, min_run))
    if not short_choices or len(long_choices) < 2:
        return None
    short_gap = random.choice(short_choices)
    long_gaps = random.sample(long_choices, 2)
    plan = [short_gap, long_gaps[0], long_gaps[1]]
    random.shuffle(plan)
    return plan


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    ray_len = int(ids_base.shape[0])
    active_template_ids = [int(ids_base[r, 0]) for r in range(ray_len)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False

    available_colors = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available_colors) < _NUM_MARKERS:
        return False

    min_run = min(
        _outward_bg_count(grid.shape[0], grid.shape[1], r0, c0, sub, direction)
        for r0, c0, sub, _ids, direction, _active in sprite_placements
    )
    if min_run < _min_outward_run(ray_len):
        return False

    for _ in range(_MARKER_PLACEMENT_RETRIES):
        gap_plan = _sample_gap_plan(ray_len, min_run)
        if gap_plan is None:
            return False

        tids = random.sample(active_template_ids, _NUM_MARKERS)
        infos = []
        ok = True
        for tid, gap in zip(tids, gap_plan):
            info = _analyze_tid_gap(grid, sprite_placements, tid, gap, ray_len)
            if info is None:
                ok = False
                break
            infos.append(info)
        if not ok:
            continue

        union_emit_cells: set[tuple[int, int]] = set()
        for info in infos:
            if union_emit_cells & info["emit_cells"]:
                ok = False
                break
            union_emit_cells |= info["emit_cells"]
        if not ok:
            continue

        temp = grid.copy()
        placements: list[tuple[int, tuple[int, int], int]] = []
        marker_placeholder = (_BG + 1) % 10
        for info in sorted(infos, key=lambda item: len(item["hosts"])):
            host_options = info["hosts"][:]
            random.shuffle(host_options)
            placed = False
            for host in host_options:
                marker_r, marker_c = host["marker"]
                if (marker_r, marker_c) in union_emit_cells:
                    continue
                if not _isolated_singleton(temp, marker_r, marker_c):
                    continue
                temp[marker_r, marker_c] = marker_placeholder
                placements.append((info["tid"], (marker_r, marker_c), info["gap"]))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        colors = random.sample(available_colors, _NUM_MARKERS)
        final_grid = grid.copy()
        for color, (_tid, (r, c), _gap) in zip(colors, placements):
            final_grid[r, c] = int(color)

        try:
            main(final_grid)
        except Exception:
            continue

        grid[:, :] = final_grid
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template, ids_base)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(unique_variants) < num_sprites:
        return None
    chosen = random.sample(unique_variants, num_sprites)

    placements = _place_sprites(H, W, chosen, N)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, ids_base, template_colors):
        return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
