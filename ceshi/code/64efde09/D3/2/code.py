# Variant of 64efde09
# Dimension: D3 — 轨迹样式：射线的涂色模式 (trajectory)
# Change: keep_source — 射线以标记像素的颜色连续涂色，从端口到停止点全部填充... → 渐变长度轨迹

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis
# ===============
# design_analysis:
#   variant_goal:
#     - 保留原题的“模板端口由单像素标记命名，再把同一个模板端口传播到所有同构精灵”这一主干。
#     - 只把 D3 从“射线一直涂到被阻挡/边界”为止，改成“每种颜色在所有精灵上都只涂固定长度”。
#     - 该固定长度由输入里可见的几何量给出：某个标记像素到它所标记的活动边端口之间，沿射线方向隔着多少个背景格。
#   concretization_and_trace:
#     - 输入仍然由多个 2xN / Nx2 的致密矩形精灵和 3 个孤立标记组成；精灵仍是同一个模板的二面体变换。
#     - 标记仍落在某个活动边端口的外射线上，但不再强制全都贴在边界；这样同一题里可以出现不同的可见长度。
#     - 小例子：若某个左朝向精灵的端口在列 8，标记在同一行列 5，那么端口与标记之间有 2 个背景格（列 7、6）。
#       原逻辑会把所有对应端口向左一直涂到边界或障碍；新逻辑只在每个对应端口外侧涂 2 格，列 5 的标记本身只负责“测长+定色”，不是“整条线的终点复制模板”。
#   complexity_calibration:
#     - 若仍把所有标记都固定在边界，那么同一来源精灵上的三个长度会完全相同，D3 的变化不够显眼。
#     - 因此生成器允许标记位于射线中途，但仍限制它们在同一条正交射线上、且是该方向遇到的第一个非背景像素。
#     - 同时强制 3 个标记使用 3 个不同长度，避免退化成“只是画短一点但长度都一样”。
#   implicit_assumption_audit:
#     - 原 main 中 `_first_marker_on_ray` 只返回颜色，隐含前提是“长度不重要”；变体后必须同时返回端口到标记前的背景格数。
#     - 原 main 末尾 `while ...` 默认可一直涂到阻挡处；变体后这会错误地产生旧行为，必须改为“精确迭代 gap 次”。
#     - 原 generator 把标记放在边界，隐含“长度 = 到边界的剩余空间”。变体后若继续这样做且还想让三种颜色长度不同，会迫使标记来自不同精灵并频繁截断别的精灵，所以必须放宽为“边界或边界内”。
#     - 原 solver 允许不同精灵的同色端口只共享颜色映射；变体后同一个模板端口还必须共享同一个 gap，因此新增 `template_id -> gap` 一致性断言。
#   placement_legality:
#     - 每个精灵外侧都保留至少 5 格可用射线空间，这样 3 个标记可以稳定地采用 3 个不同长度（2、3、4 或更大）。
#     - 标记至少与精灵隔 2 格，因此它们保持 8 连通下的单像素连通块。
#     - 生成器在真正接受样本前调用新 solver 验证，保证：模板识别无歧义、每个标记都能被唯一端口看到、按新长度规则绘制时不会越界也不会冲突。
#   visual_observability:
#     - 解题者能直接看到：某个标记离端口近，则所有对应射线都短；离端口远，则所有对应射线都长。
#     - 三个标记长度在单个样例内就不同，避免只能跨样例比较。
#     - 长度完全来自可见的几何距离，不依赖颜色编号或任何不可见排序。
#   cleanliness_and_color_inference:
#     - 模板颜色仍只负责识别活动边，不参与长度编码；标记颜色只负责“哪一个端口”，长度靠空间距离给出。
#     - 输出里每种颜色仍是一条直的、单像素宽的正交线段，没有额外叠层，视觉上比原题更整洁：线段通常停在空白处，不会把整片边界都染满。


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer colored port labels between matching rectangular sprites, while preserving each label's measured ray length.

    Input structure:
    - The background is the dominant color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are rotations/reflections of one shared template.
    - Isolated single-pixel markers lie on outward rays from some active-edge cells, and the blank gap before each marker is meaningful.

    Rule:
    - For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
    - The first marker seen along an active-edge ray labels that template port with both a color and a gap length: the number of background cells between the port and the marker.
    - On every matching sprite, paint that color outward from the corresponding port for exactly the same number of cells, not all the way to the boundary.

    Key insight:
    - Marker positions serve two roles at once: their colors identify shared template ports, and their distances encode how long every copied ray must be after the template is rotated or reflected.
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

            r = r0 + lr + dr
            c = c0 + lc + dc
            for _ in range(template_id_to_gap[template_id]):
                assert 0 <= r < H and 0 <= c < W, "preserved ray length must stay inside the grid"
                assert int(g[r, c]) == bg, "preserved ray cells must be empty in the input"
                assert int(out[r, c]) in (bg, color), "preserved rays must not conflict"
                out[r, c] = color
                r += dr
                c += dc

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
_MIN_OUTWARD_RUN = 5
_MARKER_GAP_MIN = 2
_PLACEMENT_RETRIES = 120
_MARKER_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 400


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


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int] | None:
    if dr == 0 and dc == -1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = _MIN_OUTWARD_RUN, W - w
    elif dr == 0 and dc == 1:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = 0, W - w - _MIN_OUTWARD_RUN
    elif dr == -1 and dc == 0:
        r_lo, r_hi = _MIN_OUTWARD_RUN, H - h
        c_lo, c_hi = 0, W - w
    else:
        r_lo, r_hi = 0, H - h - _MIN_OUTWARD_RUN
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


def _place_sprites(H: int, W: int, variants) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            pos = _random_sprite_position(H, W, h, w, *direction)
            if pos is None:
                return None
            r0, c0 = pos
            occ = _occupied_set(placements)
            if any(
                (r0 + r, c0 + c) in occ
                for r in range(-pad, h + pad)
                for c in range(-pad, w + pad)
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


def _ray_prefix_clear(
    grid: np.ndarray,
    port_r: int,
    port_c: int,
    dr: int,
    dc: int,
    gap: int,
) -> bool:
    H, W = grid.shape
    r, c = port_r, port_c
    for _ in range(gap):
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != _BG:
            return False
    return True


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


def _replication_prefixes_clear(grid: np.ndarray, sprite_placements: list[tuple], tid: int, gap: int) -> bool:
    for sprite in sprite_placements:
        port = _tid_port(sprite, tid)
        if port is None:
            return False
        port_r, port_c, dr, dc = port
        if not _ray_prefix_clear(grid, port_r, port_c, dr, dc, gap):
            return False
    return True


def _place_markers(
    grid: np.ndarray, sprite_placements: list[tuple], ids_base: np.ndarray, template_colors: list[int]
) -> bool:
    H, W = grid.shape
    active_template_ids = [int(ids_base[r, 0]) for r in range(ids_base.shape[0])]
    if len(active_template_ids) < _NUM_MARKERS:
        return False

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False

    min_run = min(_outward_bg_count(H, W, r0, c0, sub, direction) for r0, c0, sub, _ids, direction, _active in sprite_placements)
    gap_choices = list(range(_MARKER_GAP_MIN, min_run))
    if len(gap_choices) < _NUM_MARKERS:
        return False

    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)
    marker_colors = random.sample(available, _NUM_MARKERS)
    marker_gaps = random.sample(gap_choices, _NUM_MARKERS)
    plans = list(zip(marker_colors, labeled_ids, marker_gaps))

    for _ in range(_MARKER_PLACEMENT_RETRIES):
        temp = grid.copy()
        placed_specs: list[tuple[int, int]] = []
        ok = True
        random.shuffle(plans)

        for color, tid, gap in plans:
            sprite_order = list(range(len(sprite_placements)))
            random.shuffle(sprite_order)
            placed = False
            for si in sprite_order:
                port = _tid_port(sprite_placements[si], tid)
                if port is None:
                    continue
                port_r, port_c, dr, dc = port
                if not _ray_prefix_clear(temp, port_r, port_c, dr, dc, gap):
                    continue
                marker_r, marker_c = _marker_position(port_r, port_c, dr, dc, gap)
                if not _isolated_singleton(temp, marker_r, marker_c):
                    continue

                temp[marker_r, marker_c] = color
                if not _replication_prefixes_clear(temp, sprite_placements, tid, gap):
                    temp[marker_r, marker_c] = _BG
                    continue
                if not all(_replication_prefixes_clear(temp, sprite_placements, old_tid, old_gap) for old_tid, old_gap in placed_specs):
                    temp[marker_r, marker_c] = _BG
                    continue
                placed_specs.append((tid, gap))
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        try:
            main(temp)
        except Exception:
            continue

        grid[:, :] = temp
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

    placements = _place_sprites(H, W, chosen)
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
