# Variant of 64efde09
# Dimension: D7 — 网格输出形式 (grid_shape)
# Change: same_size — 输出网格与输入网格尺寸相同... → 紧凑裁剪到射线区域
#
# Design analysis
# 1) 具体化与行为追踪
#    - 输入保持原题外观: 一个较大的背景画布上放置 3-4 个同构的细长矩形 sprite,
#      再在某些 sprite 的激活长边外侧边界上放 3 个单像素 marker.
#    - 原题输出: 在整幅画布上把所有匹配 sprite 的对应端口向外发射彩色射线, 返回完整画布.
#    - 本变体输出: 仍然先在完整画布上得到同样的射线结果, 但最后只保留“所有新画出的射线像素”
#      的最小外接矩形. 这个矩形通常不会包含边界 marker, 也可能把 sprite 主体裁掉.
#    - 小例子: 若某个右向 sprite 在第 8/11/14 行向右发射 3 条射线, 原题输出会保留整张 27x29
#      画布; 本变体只保留覆盖这些射线行列范围的局部窗口. 差异发生在“最终返回结果”的环节,
#      而不是在“识别哪个端口对应哪种颜色”的环节.
#
# 2) 复杂度校准
#    - 不能把裁剪写成“按所有非背景像素 tight crop”, 否则边界 marker 会把 bbox 拉回整幅
#      画布, D7 变化在很多样例里不可见.
#    - 也不能让 generator 长期只生成单条射线, 否则输出常退化为一维细条, 人类不易分辨这是
#      “由所有射线决定的裁剪窗口”. 因此保留 3 个被标记端口, 让每个 sprite 都发出多条平行射线.
#    - 同时保留 3-4 个 sprite 和原有二面体匹配, 避免任务被简化成“看一个对象直接平移”.
#
# 3) 原 solver 中与 D7 相关的隐式前提审查
#    - 组件分割、active edge 判定、marker 反查 template id 的代码都只依赖输入结构,
#      与输出尺寸无关, 可以原样保留.
#    - `out = g.copy()` 在原题里既是中间画布也是最终答案; 在本变体里它只能作为中间结果,
#      不能直接返回.
#    - 原循环里“把 bg 改成 color”这一行为刚好给了我们射线集合: 只有真正新涂上的像素才应
#      参与最终裁剪. 如果把 sprite 本体或 marker 也算进 bbox, 新规则就被破坏.
#    - 因此要显式记录每个被射线着色的坐标; 这在原题中是可省略的, 在本变体中则是必要状态.
#
# 4) 放置合法性
#    - generator 仍需保证 sprite 稠密矩形、不重叠、朝向边界、marker 与对应端口之间视线清空.
#    - 另外必须保证裁剪不退化: 至少留出一个外边界方向不被任何射线触及, 且 sprite 本体也与
#      该边界保持 2 格以上距离, 这样输出 bbox 会稳定小于输入.
#    - 仍沿用 solver 自检, 拒绝任何射线冲突或 orientation 匹配歧义.
#
# 5) 视觉可观察性
#    - 解题者从 2-4 对样例里应能看到: 输入总是大画布, 输出总是更小的局部窗口; 窗口边界随着
#      射线最远延伸位置变化, 而不是随着 marker 边界位置变化.
#    - generator 因此避免“射线刚好触满四边”的样例, 并要求裁剪后至少有一个维度缩短 2 格以上.
#
# 6) 可视化整洁度
#    - 保留原始射线互不冲突的约束, 避免多色覆盖同一格.
#    - quiet-side 约束让裁剪窗口与原始画布边界之间留下整洁留白, 输出不会看上去像随机截断.
#
# 7) 颜色变化可推断性
#    - 颜色逻辑不变: marker 颜色直接成为所有同构端口射线的颜色.
#    - 颜色数量仍控制在少量离散值内, 且每种 marker 颜色会在多个 sprite 上重复出现, 便于人眼
#      建立“端口对应颜色”的关系.
#
# 8) 参考任务说明
#    - 提示将 e87109e9 作为 tight-crop 灵感来源.
#    - 当前工作区只有 `/ARC-data/.../e87109e9.json`, 没有可读取的 solver `.py`,
#      因此这一步按“若不存在则跳过”处理, 这里直接实现标准的“按射线 bbox 紧裁剪”.
#
# 9) 修改后 main() 的新假设审查
#    - 新假设 A: 至少存在 1 个被画出的射线像素. generator 通过边界与 sprite 之间的最小净空
#      保证这一点.
#    - 新假设 B: 射线 bbox 不应等于整幅画布. generator 通过 quiet-side 方向过滤和最终验收保证.
#    - 新假设 C: 裁剪依据必须只看新射线像素. 代码通过单独的 `ray_cells` 集合保证, generator
#      不需要为 marker 是否留在框内做任何额外假设.

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
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
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
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one active-edge cell"

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
    return out, ray_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the shared template ports exactly as in the base task, then return only the ray-spanned output window.

    Input structure:
    - The background is the most frequent color on a large canvas.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral transforms of one common sprite template.
    - Three isolated boundary markers lie on outward sight-lines from active-edge cells of some sprites.

    Rule:
    - For each sprite, locate the active long edge by finding the side whose cells have the strictly stronger same-color majority.
    - A marker first seen along the outward ray from an active-edge cell labels that shared template port with the marker's color.
    - Emit that color from the corresponding port of every transformed sprite through background cells until blocked or out of bounds, then crop the full result to the smallest rectangle containing all emitted ray cells.

    Key insight:
    - The markers still define template-relative ports rather than absolute positions; the only D7 change is that the final output boundary is also inferred, because it is determined solely by the union of visible rays.
    """
    out, ray_cells = _solve_full(grid)
    return crop_bbox(out, _bbox_from_cells(ray_cells))


# --------------- generator (att1) ---------------

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
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 400
_CROP_MARGIN = 2
_MIN_REDUCTION = 2

_QUIET_SIDE_TO_DIR = {
    "left": (0, -1),
    "right": (0, 1),
    "top": (-1, 0),
    "bottom": (1, 0),
}


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
    others = [c for c in template_colors if c != right_dominant]
    if not others:
        return None
    right_col = _build_column(n, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(template.flat)) < num_colors:
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


def _filter_variants_for_quiet_side(
    variants: list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]],
    quiet_side: str,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    banned_direction = _QUIET_SIDE_TO_DIR[quiet_side]
    return [variant for variant in variants if variant[2] != banned_direction]


def _random_sprite_position(
    H: int,
    W: int,
    h: int,
    w: int,
    dr: int,
    dc: int,
    quiet_side: str,
) -> tuple[int, int]:
    row_min = 0
    row_max = H - h
    col_min = 0
    col_max = W - w

    if dr == 0 and dc == -1:
        col_min = max(col_min, _PLACEMENT_CLEARANCE)
    elif dr == 0 and dc == 1:
        col_max = min(col_max, W - w - _PLACEMENT_CLEARANCE - 1)
    elif dr == -1 and dc == 0:
        row_min = max(row_min, _PLACEMENT_CLEARANCE)
    else:
        row_max = min(row_max, H - h - _PLACEMENT_CLEARANCE - 1)

    if quiet_side == "left":
        col_min = max(col_min, _CROP_MARGIN)
    elif quiet_side == "right":
        col_max = min(col_max, W - w - _CROP_MARGIN)
    elif quiet_side == "top":
        row_min = max(row_min, _CROP_MARGIN)
    else:
        row_max = min(row_max, H - h - _CROP_MARGIN)

    assert row_min <= row_max and col_min <= col_max, "no legal position under crop constraints"
    return random.randint(row_min, row_max), random.randint(col_min, col_max)


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _place_sprites(H: int, W: int, variants, quiet_side: str) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, direction[0], direction[1], quiet_side)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
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


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    n = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, 0]) for r in range(n)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    for color, tid in zip(marker_colors, labeled_ids):
        si = random.randint(0, num_sprites - 1)
        r0, c0, _sub, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = int(locs[0, 0]), int(locs[0, 1])
        if (lr, lc) not in active_coords:
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)

        cr, cc = r0 + lr + dr, c0 + lc + dc
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if grid[cr, cc] != _BG:
                return False
            cr += dr
            cc += dc

        if grid[marker_r, marker_c] != _BG:
            return False
        grid[marker_r, marker_c] = color
    return True


def _sprite_direction_count(placements: list[tuple]) -> int:
    return len({direction for *_prefix, direction, _active in placements})


def _is_non_degenerate_crop(grid: np.ndarray) -> bool:
    full_out, ray_cells = _solve_full(grid)
    del full_out
    r0, c0, r1, c1 = _bbox_from_cells(ray_cells)
    H, W = grid.shape
    crop_h = r1 - r0 + 1
    crop_w = c1 - c0 + 1
    if crop_h == H and crop_w == W:
        return False
    if (H - crop_h) < _MIN_REDUCTION and (W - crop_w) < _MIN_REDUCTION:
        return False
    if r0 == 0 and c0 == 0 and r1 == H - 1 and c1 == W - 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(n)
    if design is None:
        return None
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template, ids_base)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    quiet_side = random.choice(list(_QUIET_SIDE_TO_DIR))
    usable_variants = _filter_variants_for_quiet_side(unique_variants, quiet_side)
    if len(usable_variants) < num_sprites:
        return None

    chosen = random.sample(usable_variants, num_sprites)
    placements = _place_sprites(H, W, chosen, quiet_side)
    if placements is None:
        return None
    if _sprite_direction_count(placements) < 2:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, ids_base, template_colors):
        return None

    try:
        if not _is_non_degenerate_crop(grid):
            return None
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
