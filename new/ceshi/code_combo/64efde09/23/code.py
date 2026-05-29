# Combo variant of 64efde09 (index 23)
# combo_spec: D1/1+D12/1
# Name: 反转majority+最少颜色种类活跃边

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg

# Design analysis:
# preserved_core:
#   - 背景色仍是全局最多色，sprite 仍是同一个 N×2 模板的二面体变换，marker 仍是边界上的单像素。
#   - marker 先给模板边端口贴标签，再把这些标签色从所有同构 sprite 的对应端口向外发射。
# fused_rule:
#   - D1/1 把“majority 最高”反过来，变成“majority 最低”。
#   - D12/1 把 active edge 判据改成“颜色种类最少”。
#   - 融合后不能把两条规则各做各的；这里把 active edge 定义为按 (majority_peak, distinct_count) 升序选最小长边：
#       1. 先比较最大同色计数，较低者优先。
#       2. 若 majority 打平，再比较颜色种类数，较少者优先。
#     这样既覆盖“两个指标一致”的情形，也覆盖“冲突时 majority 优先”和“majority 打平时由颜色种类决胜”的情形。
# concrete_trace:
#   - 冲突例: 左列计数 [A,A,A,B,C,C,C]，peak=3, distinct=3；右列 [D,D,D,D,E,E,E]，peak=4, distinct=2。
#     D1/1 会选左列，D12/1 会选右列；融合规则仍选左列，因为 peak 更低优先。
#   - 决胜例: 左列 [A,A,A,A,B,B,B]，peak=4, distinct=2；右列 [C,C,C,C,D,D,E]，peak=4, distinct=3。
#     majority 无法区分，融合规则继续看 distinct，因此选左列。
#   - 原题则看“peak 更高”，所以上述两类模板都不会退化回原题。
# complexity_calibration:
#   - 生成器只产出三种受控模式: agree / conflict / tie_break。
#   - agree: active 边同时拥有更低 peak 与更少 distinct。
#   - conflict: active 边拥有更低 peak，但另一边 distinct 更少，用来暴露优先级。
#   - tie_break: 两边 peak 相同，由 distinct 更少的一边获胜，用来证明第二维真的参与判定。
# implicit_assumptions_review:
#   - 原 `_active_side()` 断言“majority 唯一更高”；融合后改成断言完整键 `(peak, distinct)` 唯一最小。
#   - `main()` 中 marker 收集与射线绘制都只依赖 active 边坐标和模板 id，对新排序键本身没有额外耦合。
#   - `assert (lr, lc) in sprite["active"]` 仍要求 marker 标记的模板 id 必须落在新规则 active 边上，因此 generator 必须按 active_col_idx 选 marker。
#   - orientation 匹配仍假设每个 sprite 恰好命中一种二面体朝向；生成器继续拒绝任何二面体重复模板。
# placement_legality_and_cleanliness:
#   - sprite 之间保留 padding，active 边继续朝向边界，marker 与 sprite 之间保持纯背景视线。
#   - 仍用 solver 端到端验收，阻止不同颜色射线冲突或 marker 被其他对象遮挡。
# visual_observability:
#   - 排序完全依赖肉眼可见的边统计: 同色块多少、边上出现几种颜色；不使用颜色编号大小。
#   - 冲突样本里会故意出现“更杂乱的一边”和“更少颜色的一边”不是同一边；决胜样本里会出现 peak 打平但 distinct 不同。
#   - 这样在少量样例中可以看出：不是原题的“更整齐边”，也不是单纯“颜色种类最少边”，而是有优先级的双判据。
# behavior_metric_plan:
#   - 量化三类模式的样本比例，并检查 combo 选边是否会同时出现 conflict 与 tie_break。
#   - 另与原题方向比较；agree/conflict 模式中原题一定选对边的 opposite，tie_break 模式中原题无法用 strict majority 决出 active。


def _dihedral_variants_with_ids(
    template: np.ndarray, ids: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    return _dihedral_variants_with_ids(template, ids)


def _long_sides(sub: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"
    if h > w:
        return [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    return [
        ((-1, 0), [(0, c) for c in range(w)]),
        ((1, 0), [(h - 1, c) for c in range(w)]),
    ]


def _side_peak(vals: list[int] | np.ndarray) -> int:
    return max(Counter(map(int, vals)).values())


def _side_key(vals: list[int] | np.ndarray) -> tuple[int, int]:
    vals = list(map(int, vals))
    return _side_peak(vals), len(set(vals))


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    scored = []
    for direction, coords in _long_sides(sub):
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((_side_key(vals), direction, set(coords)))

    scored.sort(key=lambda item: item[0])
    assert scored[0][0] < scored[1][0], "active side must be unique under (peak, distinct) priority"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate boundary marker labels from the long edge chosen by a priority blend of low majority and low color diversity.

    Input structure:
    - The background is the most frequent color in the grid.
    - Several dense N-by-2 or 2-by-N rectangles are dihedral transforms of one shared non-square template.
    - Isolated boundary markers lie on outward rays from some sprite long edges; only the edge that wins the fused active-side test should interpret them.

    Rule:
    - For each sprite, inspect its two long sides and rank them by `(maximum same-color count, number of distinct colors)`, smaller first.
    - The first non-background marker seen on an outward ray from a winning-edge cell labels that template port with the marker color.
    - For every transformed sprite, emit each learned port color outward from the corresponding winning-edge cell until a non-background blocker or the grid boundary stops the ray.

    Key insight:
    - Marker positions encode ports of a shared template rather than fixed coordinates; after the active edge is chosen by the fused priority rule, those template-relative ports follow every rotation or reflection.
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
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_PLACEMENT_CLEARANCE = 2
_PLACEMENT_RETRIES = 100
_GEN_RETRIES = 600
_MODE_CYCLE = ("agree", "conflict", "tie_break")
_mode_cycle_index = 0


def _all_dihedral_unique(template: np.ndarray) -> bool:
    seen: list[np.ndarray] = []
    for _name, sub, _ids in _dihedral_variants(template):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev in seen):
            return False
        seen.append(sub.copy())
    return True


def _peak_bounds(n: int, distinct: int) -> tuple[int, int]:
    low = (n + distinct - 1) // distinct
    high = n - distinct + 1
    return low, high


def _pick_column_palettes(
    template_colors: list[int], active_distinct: int, inactive_distinct: int
) -> tuple[list[int], list[int]] | None:
    num_colors = len(template_colors)
    shared_count = active_distinct + inactive_distinct - num_colors
    if shared_count < 1 or shared_count > min(active_distinct, inactive_distinct):
        return None

    shared = random.sample(template_colors, shared_count)
    remaining = [c for c in template_colors if c not in shared]
    random.shuffle(remaining)

    active_extra_count = active_distinct - shared_count
    inactive_extra_count = inactive_distinct - shared_count
    if active_extra_count + inactive_extra_count != len(remaining):
        return None

    active_palette = shared + remaining[:active_extra_count]
    inactive_palette = shared + remaining[active_extra_count:]
    random.shuffle(active_palette)
    random.shuffle(inactive_palette)
    return active_palette, inactive_palette


def _build_side_with_profile(
    n: int,
    palette: list[int],
    peak: int,
) -> list[int] | None:
    distinct = len(palette)
    low, high = _peak_bounds(n, distinct)
    if peak < low or peak > high:
        return None

    dominant = random.choice(palette)
    counts = {color: 0 for color in palette}
    counts[dominant] = peak
    remaining = n - peak

    for color in palette:
        if color == dominant:
            continue
        counts[color] += 1
        remaining -= 1
    if remaining < 0:
        return None

    fillers = [color for color in palette if color != dominant]
    while remaining > 0:
        options = [color for color in fillers if counts[color] < peak]
        if not options:
            return None
        pick = random.choice(options)
        counts[pick] += 1
        remaining -= 1

    vals: list[int] = []
    for color, count in counts.items():
        vals.extend([color] * count)
    random.shuffle(vals)
    return vals


def _expected_direction_for_column(active_col_idx: int) -> tuple[int, int]:
    return (0, -1) if active_col_idx == 0 else (0, 1)


def _column_stats(template: np.ndarray, col_idx: int) -> tuple[int, int]:
    vals = list(map(int, template[:, col_idx]))
    return _side_key(vals)


def _realizes_mode(template: np.ndarray, active_col_idx: int, mode: str) -> bool:
    active_stats = _column_stats(template, active_col_idx)
    other_stats = _column_stats(template, 1 - active_col_idx)
    if mode == "agree":
        return active_stats[0] < other_stats[0] and active_stats[1] < other_stats[1]
    if mode == "conflict":
        return active_stats[0] < other_stats[0] and active_stats[1] > other_stats[1]
    if mode == "tie_break":
        return active_stats[0] == other_stats[0] and active_stats[1] < other_stats[1]
    return False


def _next_mode() -> str:
    global _mode_cycle_index
    mode = _MODE_CYCLE[_mode_cycle_index % len(_MODE_CYCLE)]
    _mode_cycle_index += 1
    return mode


def _design_template(N: int, mode: str) -> tuple[np.ndarray, np.ndarray, list[int], int, str] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)
    active_col_idx = random.randint(0, 1)

    if mode == "conflict":
        active_distinct, inactive_distinct = 3, 2
    else:
        active_distinct, inactive_distinct = 2, 3

    palettes = _pick_column_palettes(template_colors, active_distinct, inactive_distinct)
    if palettes is None:
        return None
    active_palette, inactive_palette = palettes

    active_low, active_high = _peak_bounds(N, active_distinct)
    inactive_low, inactive_high = _peak_bounds(N, inactive_distinct)

    if mode == "agree":
        peak_active_max = min(active_high, inactive_high - 1)
        if active_low > peak_active_max:
            return None
        active_peak = random.randint(active_low, peak_active_max)
        inactive_peak_low = max(inactive_low, active_peak + 1)
        if inactive_peak_low > inactive_high:
            return None
        inactive_peak = random.randint(inactive_peak_low, inactive_high)
    elif mode == "conflict":
        peak_active_max = min(active_high, inactive_high - 1)
        if active_low > peak_active_max:
            return None
        active_peak = random.randint(active_low, peak_active_max)
        inactive_peak_low = max(inactive_low, active_peak + 1)
        if inactive_peak_low > inactive_high:
            return None
        inactive_peak = random.randint(inactive_peak_low, inactive_high)
    else:
        tie_low = max(active_low, inactive_low)
        tie_high = min(active_high, inactive_high)
        if tie_low > tie_high:
            return None
        active_peak = inactive_peak = random.randint(tie_low, tie_high)

    active_col = _build_side_with_profile(N, active_palette, active_peak)
    inactive_col = _build_side_with_profile(N, inactive_palette, inactive_peak)
    if active_col is None or inactive_col is None:
        return None

    columns = [active_col, inactive_col] if active_col_idx == 0 else [inactive_col, active_col]
    template = np.column_stack(columns).astype(int)
    if len(set(map(int, template.flat))) != num_colors:
        return None
    if not _all_dihedral_unique(template):
        return None

    new_direction, _ = _active_side(template)
    if new_direction != _expected_direction_for_column(active_col_idx):
        return None
    if not _realizes_mode(template, active_col_idx, mode):
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors, active_col_idx, mode


def _deduplicated_variants(
    template: np.ndarray, ids_base: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    raw = _dihedral_variants_with_ids(template, ids_base)

    seen: list[np.ndarray] = []
    result = []
    for _name, sub, ids_arr in raw:
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        seen.append(sub.copy())
        result.append((sub, ids_arr, direction, list(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad - 1)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad - 1), random.randint(0, W - w)


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
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
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
    active_col_idx: int,
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, active_col_idx]) for r in range(N)]
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
        lr, lc = map(int, locs[0])
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


def _gen_attempt(mode: str) -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N, mode)
    if design is None:
        return None
    template, ids_base, template_colors, active_col_idx, _mode = design

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

    if not _place_markers(grid, placements, ids_base, template_colors, active_col_idx):
        return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = _next_mode()
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt(mode)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
