# Combo variant of 64efde09 (index 12)
# combo_spec: D7/0+D17/0
# Name: 紧凑裁剪+双精灵极简版
#
# Design analysis
# 1) 具体化与行为追踪
#    - 原题输入会放 3-4 个同构细长矩形 sprite, marker 可能分散在多个 sprite 面向的边界上;
#      输出保留整张画布, 只是在所有对应端口外侧补上彩色射线.
#    - 本融合版把场景压缩成恰好 2 个 sprite: 一个是“参考 sprite”, 3 个 marker 全挂在它的
#      同一条激活长边所面向的边界上; 另一个 sprite 没有 marker, 只负责接收同构端口的射线.
#    - 例子: 若参考 sprite 与目标 sprite 都朝右, marker 标记了模板左列中的 3 个端口, 则
#      参考 sprite 的 3 条射线会在右边界 marker 前一格停下, 目标 sprite 的 3 条射线则直接延伸
#      到右边界. 最终输出不是完整大画布, 而是把这 6 条新射线的最小包围框裁出来.
#    - 因此 D17 改动发生在“对象数量与 marker 分布”, D7 改动发生在“最终返回范围”; 两者
#      叠加后, 解题者既要认出 marker 只定义共享模板端口, 也要认出输出边界只由射线决定.
#
# 2) 复杂度校准
#    - 若把两个 sprite 做成互相垂直, 3 条水平射线和 3 条竖直射线会产生大量交叉点, 很容易
#      发生多色冲突, 还会把 generator 的重试空间浪费在无效摆放上.
#    - 因此这里改成“恰好 2 个、同朝向但不同翻转”的 pair: 仍然需要通过模板 id 做位置对应,
#      不是简单复制同一行/列; 但所有射线彼此平行, 冲突风险显著下降.
#    - marker 仍保留 3 个, 颜色唯一且会在两个 sprite 上重复出现, 这样输出裁剪框里能看到
#      多条同色平行线, 规则足够明显, 不会因过度简化而退化成单条示意线.
#
# 3) 隐式假设审查
#    - 原 solver 依赖“每个 sprite 都能唯一匹配到同一模板的某个二面体变体”. 这里只有 2 个
#      sprite, 这个假设仍然成立, 但 generator 必须避开会导致多重匹配的对称模板, 所以保留
#      solver 自检作为最终过滤.
#    - 原逻辑只要求 `len(sprites_raw) >= 2`; 融合后应为恰好 2 个, 否则 D17 行为没有真正发生.
#    - 原逻辑允许 marker 分布在多个 sprite 上; 融合后 main() 新增“marker host 恰好 1 个”的
#      假设. generator 会把 3 个 marker 全放在参考 sprite 上来保证这一点.
#    - 原题 `out = g.copy()` 可以直接返回; 融合后必须额外记录 `ray_cells`, 因为裁剪框只能由
#      新画出的射线决定, 不能把 sprite 主体或边界 marker 算进 bbox.
#
# 4) 放置合法性
#    - 两个 sprite 都必须是不重叠的稠密矩形, 并保持最少 2 格净空, 同时激活边朝向其面向的边界.
#    - 因为两者同朝向, generator 还要额外保证它们在射线正交方向上的区间分离: 水平射线场景中
#      分离行区间, 垂直射线场景中分离列区间. 这样不同颜色的平行射线不会撞在同一格.
#    - marker 与参考 sprite 之间必须全是背景, 且 marker 不能占用目标 sprite 将要打到的同一
#      条边界位置; 区间分离约束正好同时保证这点.
#
# 5) 视觉可观察性
#    - 人类从少量样例中应能看到: 输入永远只有两个同构 sprite, 但只有其中一个直接带 marker;
#      输出永远比输入画布小, 而且大小取决于射线的联合范围.
#    - 由于目标 sprite 没有 marker, 它发出的射线会比参考 sprite 更长, 这会稳定制造“同色线段
#      在两处重复, 但裁剪框不追随 marker”这一视觉线索.
#
# 6) 可视化整洁度
#    - 所有射线都平行, 不会形成杂乱交叉.
#    - 两个 sprite 使用同一模板但不同翻转, 既能看出它们是同构对象, 又不会在像素级上完全一致.
#    - generator 还要求裁剪后高宽都至少比输入缩短 2 格, 避免输出看起来像“几乎没裁”的噪声变化.
#
# 7) 变化的可推断性
#    - 颜色规则不变: marker 的颜色就是共享模板端口的颜色标签, 会同时出现在两个 sprite 的对应
#      射线上. 颜色种类固定为 3 个, 每色重复 2 次, 推断线索充足.
#    - 裁剪规则也可通过重复样例归纳: 边界 marker 明明在输入最外圈, 却经常不出现在输出中,
#      说明输出边界不是“所有非背景像素的 bbox”, 而是“射线区域的 bbox”.
#
# 8) 融合行为的量化验收
#    - 生成器不会只检验 `main(input)` 能跑通, 还会统计融合指标:
#      `sprite_count == 2`, `marker_host_count == 1`, `crop_h <= H-2`, `crop_w <= W-2`,
#      且两个 sprite 的像素阵列不完全相同.
#    - 若这些指标不满足, 样本就会被拒绝, 以防退化成原题或弱化成“只有裁剪没有双精灵极简结构”.
#
# 9) 修改后 main() 的新假设审查
#    - 新假设 A: 所有 marker 都由同一个 sprite 提供. `_solve_full()` 会显式收集 host 并断言.
#    - 新假设 B: 至少存在可见射线像素. generator 通过边界净空保证这一点.
#    - 新假设 C: 裁剪框的高宽都应明显缩短. 这不是 solver 的必要前提, 但 generator 的验收会
#      强制保证, 从而让 D7 行为稳定可见.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


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
) -> int | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return markers.get((r, c))
        r += dr
        c += dc
    return None


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, list[dict], dict[tuple[int, int], int]]:
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

    assert len(sprites_raw) == 2, "expected exactly two transformed sprites"
    assert len(markers) == 3, "expected exactly three isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"
    return g, bg, sprites_raw, markers


def _match_sprites(sprites_raw: list[dict]) -> list[dict]:
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
    return sprites


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], dict]:
    g, bg, sprites_raw, markers = _extract_scene(grid)
    sprites = _match_sprites(sprites_raw)

    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    marker_hosts: set[int] = set()
    for sprite_index, sprite in enumerate(sprites):
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            marker_hosts.add(sprite_index)
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one active-edge cell"
    assert len(marker_hosts) == 1, "expected one marked reference sprite and one unlabeled peer"

    out = g.copy()
    H, W = g.shape
    ray_cells: set[tuple[int, int]] = set()
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
            while 0 <= r < H and 0 <= c < W:
                if int(g[r, c]) != bg:
                    break
                assert int(out[r, c]) in (bg, color), "control rays must not conflict"
                out[r, c] = color
                ray_cells.add((r, c))
                r += dr
                c += dc

    assert ray_cells, "expected at least one visible ray cell"
    details = {"sprites": sprites, "markers": markers, "marker_hosts": tuple(sorted(marker_hosts))}
    return out, ray_cells, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode three marker-labeled template ports from a single reference sprite, propagate them to its only peer, and return only the ray-defined window.

    Input structure:
    - The background is the most frequent color on a large canvas.
    - Exactly two dense 2-by-N or N-by-2 rectangles are dihedral transforms of one common thin template.
    - Three isolated boundary markers all lie on outward sight-lines from the active edge of just one of the two sprites.

    Rule:
    - On each sprite, find the active long edge by choosing the side whose cells have the strictly stronger same-color majority.
    - The first marker seen along an outward ray from the marked sprite labels that shared template port with the marker's color.
    - Emit each learned color from the corresponding active-edge port of both sprites through background cells until blocked or out of bounds, then crop the full painted result to the smallest rectangle containing the emitted ray cells.

    Key insight:
    - The markers identify template-relative ports, not fixed coordinates; the second sprite inherits those ports through its own transform, and the visible output boundary must also be inferred because only the union of new rays determines the crop.
    """
    out, ray_cells, _details = _solve_full(grid)
    return crop_bbox(out, _bbox_from_cells(ray_cells))


# --------------- generator (att1) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT = 2
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_INNER_MARGIN = 2
_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 500
_MIN_REDUCTION = 2
_MIN_RAY_CELLS = 12
_DIRECTION_CYCLE = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_NEXT_DIRECTION_INDEX = 0


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _design_template(n: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(n, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(n, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    right_majority = random.randint(max(2, n // num_colors), max_right_majority)
    right_dominant = random.choice(template_colors)
    if len(template_colors) < 2:
        return None
    right_col = _build_column(n, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None

    ids_base = np.arange(n * 2, dtype=int).reshape(n, 2)
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
        result.append((sub, ids_arr, direction, list(active_set)))
    return result


def _same_direction_pairs(
    variants: list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]
) -> list[tuple[tuple, tuple]]:
    pairs = []
    for i, left in enumerate(variants):
        for right in variants[i + 1 :]:
            if left[2] != right[2]:
                continue
            if left[0].shape != right[0].shape:
                continue
            if np.array_equal(left[0], right[0]):
                continue
            pairs.append((left, right))
    return pairs


def _choose_pair_for_direction(
    pair_candidates: list[tuple[tuple, tuple]], target_direction: tuple[int, int]
) -> tuple[tuple, tuple] | None:
    directed = [pair for pair in pair_candidates if pair[0][2] == target_direction]
    if not directed:
        return None
    chosen = random.choice(directed)
    if random.random() < 0.5:
        chosen = (chosen[1], chosen[0])
    return chosen


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    row_min = _INNER_MARGIN
    row_max = H - h - _INNER_MARGIN
    col_min = _INNER_MARGIN
    col_max = W - w - _INNER_MARGIN

    if dr == 0 and dc == -1:
        col_min = max(col_min, _PLACEMENT_CLEARANCE)
    elif dr == 0 and dc == 1:
        col_max = min(col_max, W - w - _PLACEMENT_CLEARANCE - 1)
    elif dr == -1 and dc == 0:
        row_min = max(row_min, _PLACEMENT_CLEARANCE)
    else:
        row_max = min(row_max, H - h - _PLACEMENT_CLEARANCE - 1)

    assert row_min <= row_max and col_min <= col_max, "no legal position"
    return random.randint(row_min, row_max), random.randint(col_min, col_max)


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int, pad: int) -> bool:
    return not (a1 + pad < b0 or b1 + pad < a0)


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _orthogonal_intervals_conflict(existing: list[tuple], candidate: tuple[int, int, np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]) -> bool:
    r0, c0, sub, _ids_arr, direction, _active_coords = candidate
    h, w = sub.shape
    dr, dc = direction
    for er0, ec0, esub, _eids, _edir, _eactive in existing:
        eh, ew = esub.shape
        if dc != 0:
            if _ranges_overlap(r0, r0 + h - 1, er0, er0 + eh - 1, _PLACEMENT_PAD):
                return True
        else:
            if _ranges_overlap(c0, c0 + w - 1, ec0, ec0 + ew - 1, _PLACEMENT_PAD):
                return True
    return False


def _place_pair(H: int, W: int, chosen: tuple[tuple, tuple]) -> list[tuple] | None:
    placements: list[tuple] = []
    occ_pad = _PLACEMENT_PAD
    for variant in chosen:
        sub, ids_arr, direction, active_coords = variant
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, direction[0], direction[1])
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-occ_pad, h + occ_pad) for c in range(-occ_pad, w + occ_pad)):
                continue
            candidate = (r0, c0, sub, ids_arr, direction, active_coords)
            if _orthogonal_intervals_conflict(placements, candidate):
                continue
            placements.append(candidate)
            placed = True
            break
        if not placed:
            return None
    return placements


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _reference_template_ids(ids_base: np.ndarray) -> list[int]:
    n = ids_base.shape[0]
    preferred_rows = list(range(1, n - 1))
    usable_rows = preferred_rows if len(preferred_rows) >= _NUM_MARKERS else list(range(n))
    return [int(ids_base[r, 0]) for r in usable_rows]


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    reference = sprite_placements[0]
    r0, c0, _sub, ids_arr, direction, active_coords = reference
    dr, dc = direction

    active_template_ids = _reference_template_ids(ids_base)
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid in zip(marker_colors, labeled_ids):
        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = map(int, locs[0])
        if (lr, lc) not in active_coords:
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)

        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if int(grid[cr, cc]) != _BG:
                return False
            cr += dr
            cc += dc

        if int(grid[marker_r, marker_c]) != _BG:
            return False
        grid[marker_r, marker_c] = color
    return True


def _fusion_metrics(grid: np.ndarray) -> dict:
    out, ray_cells, details = _solve_full(grid)
    H, W = grid.shape
    r0, c0, r1, c1 = _bbox_from_cells(ray_cells)
    crop_h = r1 - r0 + 1
    crop_w = c1 - c0 + 1
    sprites = details["sprites"]
    return {
        "sprite_count": len(sprites),
        "marker_host_count": len(details["marker_hosts"]),
        "ray_cell_count": len(ray_cells),
        "crop_h": crop_h,
        "crop_w": crop_w,
        "input_h": H,
        "input_w": W,
        "reduction_h": H - crop_h,
        "reduction_w": W - crop_w,
        "directions": tuple(sprite["direction"] for sprite in sprites),
        "same_direction": sprites[0]["direction"] == sprites[1]["direction"],
        "distinct_pixels": not np.array_equal(sprites[0]["sub"], sprites[1]["sub"]),
        "output": out,
    }


def _is_good_fusion_instance(grid: np.ndarray) -> bool:
    metrics = _fusion_metrics(grid)
    if metrics["sprite_count"] != _SPRITE_COUNT:
        return False
    if metrics["marker_host_count"] != 1:
        return False
    if not metrics["same_direction"]:
        return False
    if not metrics["distinct_pixels"]:
        return False
    if metrics["reduction_h"] < _MIN_REDUCTION or metrics["reduction_w"] < _MIN_REDUCTION:
        return False
    if metrics["ray_cell_count"] < _MIN_RAY_CELLS:
        return False
    return True


def _gen_attempt(target_direction: tuple[int, int]) -> np.ndarray | None:
    n = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(n)
    if design is None:
        return None
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template, ids_base)
    pair_candidates = _same_direction_pairs(unique_variants)
    if not pair_candidates:
        return None

    chosen = _choose_pair_for_direction(pair_candidates, target_direction)
    if chosen is None:
        return None

    placements = _place_pair(H, W, chosen)
    if placements is None or len(placements) != _SPRITE_COUNT:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, ids_base, template_colors):
        return None

    try:
        if not _is_good_fusion_instance(grid):
            return None
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_DIRECTION_INDEX
    target_direction = _DIRECTION_CYCLE[_NEXT_DIRECTION_INDEX % len(_DIRECTION_CYCLE)]
    _NEXT_DIRECTION_INDEX += 1
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt(target_direction)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
