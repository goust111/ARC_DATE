# Combo variant of c7f57c3e (index 16)
# combo_spec: D1/2+D13/2
# Name: 按空间位置分组 + 边框分隔结构

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪:
#    - 原题: 每个精灵先独立识别成 proto_A 或 proto_B, 然后直接切到另一种原型。
#    - 融合后: 网格里出现一条横线和一条竖线, 把画布切成四个象限, 每个象限恰好一个精灵。
#      例如左列上下两个区域分别放 A、B, 则输出中上格变成 B、下格变成 A; 如果右列上下都放 A,
#      则这一整列在输出里保持 A。也就是说, 控制变量不再是“这个精灵当前是什么颜色模式”,
#      而是“它位于分隔线哪一侧, 它同列另一侧的伙伴是什么原型”。
# b) 复杂度校准:
#    - 仍然只保留 2 种原型、4 个精灵、1 个放大尺度, 自由度受控。
#    - 每个样例强制一列为 mixed pair、另一列为 same-type pair, 这样新规则会稳定地产生
#      “两格变化、两格不变”的重复证据, 不会退化成原题的“全都翻转”。
# c) 隐式假设审查:
#    - 原 solver 假设所有非背景连通块都参与交换; 现在分隔线是额外连通块, 必须先识别并剔除。
#    - 原 solver 的 `1 - best_idx` 依赖逐精灵切换; 现在替换目标来自同列对侧精灵的 proto_idx。
#    - 行列分组不再依赖颜色或排序缝隙推断, 而是依赖显式横线/竖线, 所以 generator 必须保证
#      每个象限只有一个精灵, 且精灵与分隔线之间留空隙。
# d) 放置合法性:
#    - 每个象限预留的是“输入 patch 与输出 patch 按锚点对齐后的并集框”, 因此无论交换后变成哪种
#      原型, 都不会越界或跨过分隔线。
#    - 四个象限彼此独立, 分隔线厚度固定为 1, 精灵不接触边框和分隔线, 因而无重叠风险。
# e) 视觉可观察性:
#    - 分隔线是肉眼可见的物理分组线索, 上下交换发生在同一列内部, 不是按颜色值做隐藏排序。
#    - 观察者从 2-3 个样例可看到: 同列上下是配对单位, 而另一列若上下原型相同则整列不变。
# f) 可视化整洁度:
#    - 只使用一个十字分隔结构, 不叠加额外标记。每个象限单独容纳一个精灵, 版面规则且清晰。
# g) 变化的可推断性:
#    - 颜色变化仍来自两种重复出现的原型, 不是任意颜色映射。空间分区与是否变化同步出现,
#      因此人类可以从“位置组 + 同列配对”归纳规则。
# h) main() 的新假设与 generator 保证:
#    - main() 假设存在同色的全宽横线与全高竖线; generator 固定绘制一个十字分隔。
#    - main() 假设四个象限各一个精灵, 且最小尺度精灵里能恢复两种原型; generator 固定四象限布局,
#      并让 mixed pair 保持 scale=1, 把唯一的大精灵放在 same-type 列中。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Bounding box of color-1 pixels; color 1 is guaranteed to exist."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _match_proto_index(
    grid: np.ndarray,
    arr: np.ndarray,
    bbox: tuple[int, int, int, int],
    protos: list[np.ndarray],
    scale: int,
    bg: int,
) -> int:
    """Return the prototype index whose scaled patch best matches this sprite."""
    h, w = grid.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        top = int(r0 + ar0 - pr0 * scale)
        left = int(c0 + ac0 - pc0 * scale)

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue

        sub_g = grid[rr0:rr1, cc0:cc1]
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _place_scaled_proto(
    out: np.ndarray,
    proto: np.ndarray,
    scale: int,
    anchor_rc: tuple[int, int],
    bg: int,
) -> None:
    """Paste a scaled prototype so its anchor top-left lands on `anchor_rc`."""
    sp = _scale_nn(proto, scale)
    pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
    top = int(anchor_rc[0] - pr0 * scale)
    left = int(anchor_rc[1] - pc0 * scale)
    paste(out, sp, top, left, transparent=bg)


def _detect_separator_cross(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    """Detect the non-bg cross that partitions the four sprite regions."""
    h, w = grid.shape
    row_candidates = [r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    col_candidates = [c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])]
    assert row_candidates and col_candidates, "missing separator cross"

    colors = set(int(grid[r, 0]) for r in row_candidates) & set(int(grid[0, c]) for c in col_candidates)
    assert colors, "separator row/col colors must agree"

    best = None
    for color in colors:
        rows = [r for r in row_candidates if int(grid[r, 0]) == color]
        cols = [c for c in col_candidates if int(grid[0, c]) == color]
        score = len(rows) + len(cols)
        if best is None or score > best[0]:
            best = (score, color, rows, cols)

    assert best is not None
    _score, sep_color, row_lines, col_lines = best
    return int(sep_color), sorted(row_lines), sorted(col_lines)


def _aligned_union_bbox(output_patch: np.ndarray, input_patch: np.ndarray) -> tuple[int, int, int, int]:
    """Union box for two anchor-aligned patches; returns (dr, dc, union_h, union_w)."""
    oa = _pick_anchor_bbox(output_patch, -1)
    ia = _pick_anchor_bbox(input_patch, -1)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]

    top = min(0, dr)
    left = min(0, dc)
    bottom = max(output_patch.shape[0], dr + input_patch.shape[0])
    right = max(output_patch.shape[1], dc + input_patch.shape[1])
    return int(dr), int(dc), int(bottom - top), int(right - left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange prototype identity across the separator-defined upper/lower regions of each column.

    Input structure:
    - bg is the most common color. A single non-bg cross (one horizontal line and one vertical
      line of the same color) partitions the canvas into four clean regions.
    - Each region contains one disconnected sprite, and the four sprites are scaled copies of
      exactly two prototype patterns that all contain color 1 as an anchor subshape.

    Rule:
    - Ignore the separator cross when extracting sprites, recover the two prototypes from the
      smallest sprites, and infer every sprite's scale and current prototype identity.
    - Group sprites by which side of the cross they occupy, yielding four quadrant slots.
    - Within each column, swap prototype identity between the upper sprite and the lower sprite,
      while keeping each slot's own scale, anchor position, and the separator cross unchanged.

    Key insight:
    - The cross is not decoration: it is the visible statement of the grouping rule. A sprite
      does not flip just because of its own colors; it copies the prototype used in the region
      directly above or below it across the same vertical partition.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep_color, row_lines, col_lines = _detect_separator_cross(g, bg)

    sprite_mask = g != bg
    sprite_mask[row_lines, :] = False
    sprite_mask[:, col_lines] = False
    comps = components_mask(sprite_mask, connectivity=8)
    assert len(comps) == 4, "grid must contain exactly 4 sprite components outside the separators"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest sprites must expose both prototype patterns"

    protos = list(uniq.values())[:2]
    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anc_h = anc0[2] - anc0[0] + 1
    base_anc_w = anc0[3] - anc0[1] + 1
    assert (anc1[2] - anc1[0] + 1, anc1[3] - anc1[1] + 1) == (base_anc_h, base_anc_w)

    top_limit = min(row_lines)
    bottom_limit = max(row_lines)
    left_limit = min(col_lines)
    right_limit = max(col_lines)

    quadrants: dict[tuple[int, int], dict[str, object]] = {}
    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        ar0, ac0, ar1, ac1 = _pick_anchor_bbox(s["arr"], bg)
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1
        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0

        if center_r < top_limit:
            row_group = 0
        else:
            assert center_r > bottom_limit, "sprite may not overlap the horizontal separator"
            row_group = 1
        if center_c < left_limit:
            col_group = 0
        else:
            assert center_c > right_limit, "sprite may not overlap the vertical separator"
            col_group = 1

        key = (row_group, col_group)
        assert key not in quadrants, "each quadrant must contain exactly one sprite"
        s["scale"] = scale
        s["anchor_rc"] = (int(r0 + ar0), int(c0 + ac0))
        s["proto_idx"] = _match_proto_index(g, s["arr"], s["bbox"], protos, scale, bg)
        quadrants[key] = s

    assert set(quadrants) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    out = np.full_like(g, bg)
    out[row_lines, :] = sep_color
    out[:, col_lines] = sep_color
    for col in range(2):
        top_sprite = quadrants[(0, col)]
        bottom_sprite = quadrants[(1, col)]
        _place_scaled_proto(out, protos[int(bottom_sprite["proto_idx"])], int(top_sprite["scale"]), top_sprite["anchor_rc"], bg)
        _place_scaled_proto(out, protos[int(top_sprite["proto_idx"])], int(bottom_sprite["scale"]), bottom_sprite["anchor_rc"], bg)

    return out


# --------------- generator ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 28
PLACEMENT_BORDER = 1
REGION_MARGIN = 1
REGION_EXTRA_MAX = 2
MAX_RETRIES = 300

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
BIG_SCALES = [2]

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _make_labels() -> np.ndarray:
    h, w = random.choice(PROTO_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3

    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    weights_by_role = {
        1: ANCHOR_ROW_WEIGHTS,
        2: VARIABLE_ROW_WEIGHTS,
        3: VARIABLE_ROW_WEIGHTS,
    }
    masks = []
    for role in roles:
        kinds, wts = zip(*weights_by_role[role])
        chosen = random.choices(kinds, weights=wts)[0]
        masks.append(_row_mask(chosen, w))

    if not any(mask[0] or mask[-1] for mask in masks):
        idx = random.randrange(h)
        if roles[idx] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[idx] = _row_mask("outer", w)
        else:
            masks[idx] = _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks, strict=True)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(bg: int, sep_color: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, sep_color, 1)]
    a_cols = random.sample(pool, 2)
    remaining = [c for c in pool if c not in a_cols]
    if random.random() < COLOR_SHARE_PROB:
        b_cols = [random.choice(a_cols), random.choice(remaining)]
        random.shuffle(b_cols)
    else:
        b_cols = random.sample(remaining, 2)

    def _colorize(lab: np.ndarray, cols: list[int]) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = cols[0]
        proto[lab == 3] = cols[1]
        return proto

    return _colorize(labels, a_cols), _colorize(np.flipud(labels), b_cols)


def _gen_attempt() -> np.ndarray | None:
    bg, sep_color = random.sample(BG_CHOICES, 2)
    proto_a, proto_b = _make_prototypes(bg, sep_color)
    protos = [proto_a, proto_b]

    same_col = random.randrange(2)
    same_type = random.randrange(2)
    diff_top_type = random.randrange(2)
    diff_bottom_type = 1 - diff_top_type

    input_types: dict[tuple[int, int], int] = {}
    for col in range(2):
        if col == same_col:
            input_types[(0, col)] = same_type
            input_types[(1, col)] = same_type
        else:
            input_types[(0, col)] = diff_top_type
            input_types[(1, col)] = diff_bottom_type

    big_row = random.randrange(2)
    big_scale = random.choice(BIG_SCALES)
    scales = {(row, col): 1 for row in range(2) for col in range(2)}
    scales[(big_row, same_col)] = big_scale

    slot_data = {}
    for row in range(2):
        for col in range(2):
            in_type = input_types[(row, col)]
            out_type = input_types[(1 - row, col)]
            scale = scales[(row, col)]
            output_patch = _scale_nn(protos[out_type], scale)
            input_patch = _scale_nn(protos[in_type], scale)
            dr, dc, uh, uw = _aligned_union_bbox(output_patch, input_patch)
            slot_data[(row, col)] = {
                "input_patch": input_patch,
                "output_patch": output_patch,
                "dr": dr,
                "dc": dc,
                "uh": uh,
                "uw": uw,
            }

    row_heights = []
    for row in range(2):
        need = max(int(slot_data[(row, col)]["uh"]) for col in range(2))
        row_heights.append(need + 2 * REGION_MARGIN + random.randint(0, REGION_EXTRA_MAX))
    col_widths = []
    for col in range(2):
        need = max(int(slot_data[(row, col)]["uw"]) for row in range(2))
        col_widths.append(need + 2 * REGION_MARGIN + random.randint(0, REGION_EXTRA_MAX))

    base_h = 2 * PLACEMENT_BORDER + row_heights[0] + 1 + row_heights[1]
    base_w = 2 * PLACEMENT_BORDER + col_widths[0] + 1 + col_widths[1]
    min_n = max(GRID_SIZE_MIN, base_h, base_w)
    if min_n > GRID_SIZE_MAX:
        return None

    n = random.randint(min_n, GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)

    extra_h = n - base_h
    extra_w = n - base_w
    top_extra = random.randint(0, extra_h)
    left_extra = random.randint(0, extra_w)

    row_start0 = PLACEMENT_BORDER + top_extra
    sep_row = row_start0 + row_heights[0]
    row_starts = [row_start0, sep_row + 1]

    col_start0 = PLACEMENT_BORDER + left_extra
    sep_col = col_start0 + col_widths[0]
    col_starts = [col_start0, sep_col + 1]

    out[sep_row, :] = sep_color
    out[:, sep_col] = sep_color
    inp[sep_row, :] = sep_color
    inp[:, sep_col] = sep_color

    for row in range(2):
        for col in range(2):
            info = slot_data[(row, col)]
            uh = int(info["uh"])
            uw = int(info["uw"])
            band_h = row_heights[row]
            band_w = col_widths[col]
            r_slack = band_h - uh - 2 * REGION_MARGIN
            c_slack = band_w - uw - 2 * REGION_MARGIN
            assert r_slack >= 0 and c_slack >= 0

            ur = row_starts[row] + REGION_MARGIN + random.randint(0, r_slack)
            uc = col_starts[col] + REGION_MARGIN + random.randint(0, c_slack)

            out_r = ur + max(0, -int(info["dr"]))
            out_c = uc + max(0, -int(info["dc"]))
            in_r = out_r + int(info["dr"])
            in_c = out_c + int(info["dc"])

            paste(out, info["output_patch"], out_r, out_c, transparent=bg)
            paste(inp, info["input_patch"], in_r, in_c, transparent=bg)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
