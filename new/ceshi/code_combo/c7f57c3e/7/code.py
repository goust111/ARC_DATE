# Combo variant of c7f57c3e (index 7)
# combo_spec: D3/0+D14/1
# Name: 最小化重叠（远距移动）+ 交换路径可视化

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ===============
# fused_rule_summary:
#   - 保留原题的核心识别链：背景色 + 两类原型 + color-1 锚框 + 按整数尺度识别并切换到另一类原型。
#   - D3/0 把“交换谁”从隐式原地切换改成显式配对：同尺度对象中不再按最近/最大重叠思路配对，而是选最远的一对。
#   - D14/1 再把这个配对关系可视化：输出中除切换后的精灵外，还画出两端之间的传递轨迹。
# concrete_behavior:
#   input:
#     - 一个 scale=1 的主组，恰好 4 个精灵，左到右顺序固定为 “同类、同类、异类、异类”。
#     - 一个 scale=2 的辅组，恰好 2 个精灵，用来保留原题的“尺度识别”线索。
#     - 两类原型都来自同一标签模板的 flipUD 版本，anchor bbox 完全相同，因此切换后仍可稳定对齐。
#   output:
#     - 所有精灵都在原位切换到另一类原型。
#     - 四精灵主组中，最左对象与最右异类对象配对，内侧两者也配对；即 A A B B 变成远距反向配对。
#     - 外侧长配对画在组上方，内侧配对画在组下方；每条轨迹分成两段颜色，分别使用两端输入精灵的传输色。
# small_example_trace:
#   - 原逻辑若看到同尺度 A A B B，四个位置最终仍只是各自“切换类型”，无法看出哪一个 B 对应哪一个 A。
#   - 融合后，同样先原位切换，但额外出现两条路径：长路径连接最外侧 A/B，短路径连接内侧 A/B。
#   - 若改回“最近配对”，路径会变成 A1-B1 与 A2-B2；因此 D3 的控制变量会直接改变输出轨迹布局。
# complexity_calibration:
#   - 只保留一个真正需要 D3 判定的四精灵主组，避免出现多个自由配对组导致样例过乱。
#   - 主组里同尺度、同类型各重复两次，足以让人从 2-3 组样例看出“不是就近，而是远距反向配对”。
#   - 辅组只有一对同尺度 A/B，用于支撑“尺度照旧保留、轨迹照旧存在”，但不额外引入新的配对歧义。
# implicit_assumptions_review:
#   - 原 solver 的“切换即完成任务”不再成立；现在切换后还必须按配对关系补画轨迹。
#   - 原 generator 的“3-4 个精灵且自由分布”也不再够用；D3 需要至少一个同尺度 2-vs-2 组才能非退化。
#   - 轨迹颜色若不唯一，会让 D14 变得不可观察；因此模板保证只有 role-2 与 anchor 发生 4 邻接。
#   - 轨迹若与其他对象交叠会让输出难读；因此 generator 为每个 patch 预留整块矩形保留区。
# legality_visibility_and_cleanliness:
#   - 主组轨迹固定使用上/下两条独立车道，避免两条远近不同的线互相覆盖。
#   - 辅组轨迹只允许水平或竖直直连，generator 直接构造出空白走廊。
#   - reserve mask 用整块矩形而非仅非背景像素，确保 patch 之间不会在空白走廊里彼此穿插。
# inferability_check:
#   - 人类看到样例时，能先观察到“所有精灵都换型但尺寸不变”，再观察到“真正的配对关系由长短路径决定”。
#   - 颜色变化不是隐藏映射：路径半段颜色就来自路径两端对象靠近 anchor 的传输色，几何关系可直接支撑归纳。


# --- CFG: canvas ---
GRID_SIZE_MIN = 24
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_PATCHES = 3

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
SECONDARY_SHARE_PROB = 0.40
LABEL_TEMPLATES = [
    np.array(
        [
            [0, 2, 2, 0],
            [2, 1, 1, 0],
            [0, 1, 1, 2],
            [3, 0, 0, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 2, 0, 0],
            [0, 1, 1, 2],
            [2, 1, 1, 0],
            [3, 0, 0, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 2, 2, 0],
            [2, 1, 1, 2],
            [0, 1, 1, 0],
            [3, 0, 0, 3],
        ],
        dtype=int,
    ),
]

# --- CFG: quartet patch (the D3-active group) ---
QUARTET_OUTER_GAPS = [1, 2]
QUARTET_CENTER_GAPS = [4, 5]

# --- CFG: support pair patch ---
BIG_SCALE = 2
PAIR_GAPS = [4, 5, 6]
PAIR_ORIENTATIONS = ["h", "v"]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 pixels; every valid sprite contains them."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _transfer_color(arr: np.ndarray, bg: int) -> int:
    """Return the unique non-bg, non-1 color touching the anchor in 4-neighborhood."""
    h, w = arr.shape
    colors: set[int] = set()
    for r, c in np.argwhere(arr == 1):
        rr = int(r)
        cc = int(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                v = int(arr[nr, nc])
                if v not in (bg, 1):
                    colors.add(v)
    assert len(colors) == 1, "expected exactly one anchor-adjacent transfer color"
    return next(iter(colors))


def _infer_scale(
    arr: np.ndarray,
    bg: int,
    base_anc_h: int,
    base_anc_w: int,
) -> tuple[int, tuple[int, int, int, int]]:
    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, ar1, ac1 = anc
    inst_anc_h = int(ar1 - ar0 + 1)
    inst_anc_w = int(ac1 - ac0 + 1)
    assert inst_anc_h % max(1, base_anc_h) == 0
    assert inst_anc_w % max(1, base_anc_w) == 0
    scale_h = max(1, inst_anc_h // max(1, base_anc_h))
    scale_w = max(1, inst_anc_w // max(1, base_anc_w))
    assert scale_h == scale_w, "anchor scale must be isotropic"
    return int(scale_h), anc


def _match_proto(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    arr: np.ndarray,
    anc: tuple[int, int, int, int],
    scale: int,
    protos: list[np.ndarray],
    bg: int,
) -> tuple[int, int, int]:
    """Return (best_proto_idx, top, left) by overlap scoring."""
    h, w = grid.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = anc

    best_idx = 0
    best_score = -1
    best_top = 0
    best_left = 0
    for idx, proto in enumerate(protos):
        scaled = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        pr0 *= scale
        pc0 *= scale
        top = int((r0 + ar0) - pr0)
        left = int((c0 + ac0) - pc0)

        ph, pw = scaled.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue

        sub_g = grid[rr0:rr1, cc0:cc1]
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = int(idx)
            best_top = int(top)
            best_left = int(left)

    return best_idx, best_top, best_left


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(grid != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain sprite components"
    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    return sprites


def _recover_prototypes(sprites: list[dict], bg: int) -> tuple[list[np.ndarray], int, int]:
    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototypes"
    protos = [uniq[key] for key in list(uniq.keys())[:2]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    assert anc0 == anc1, "generator guarantees identical anchor bboxes across the two prototypes"
    base_anc_h = int(anc0[2] - anc0[0] + 1)
    base_anc_w = int(anc0[3] - anc0[1] + 1)
    return protos, base_anc_h, base_anc_w


def _classify_sprites(grid: np.ndarray, sprites: list[dict], protos: list[np.ndarray], bg: int) -> list[dict]:
    anc0 = _pick_anchor_bbox(protos[0], bg)
    base_anc_h = int(anc0[2] - anc0[0] + 1)
    base_anc_w = int(anc0[3] - anc0[1] + 1)

    classified = []
    for s in sprites:
        scale, anc = _infer_scale(s["arr"], bg, base_anc_h, base_anc_w)
        best_idx, top, left = _match_proto(grid, s["bbox"], s["arr"], anc, scale, protos, bg)
        r0, c0, _r1, _c1 = s["bbox"]
        ar0, ac0, ar1, ac1 = anc
        classified.append(
            {
                "pts": s["pts"],
                "bbox": s["bbox"],
                "arr": s["arr"],
                "type": int(best_idx),
                "scale": int(scale),
                "anchor_global": (
                    int(r0 + ar0),
                    int(c0 + ac0),
                    int(r0 + ar1),
                    int(c0 + ac1),
                ),
                "place_top": int(top),
                "place_left": int(left),
            }
        )
    return classified


def _draw_direct_trail(out: np.ndarray, a: dict, b: dict, protos: list[np.ndarray], bg: int) -> None:
    """Draw a straight trail between one same-scale A/B pair."""
    if a["bbox"][1] <= b["bbox"][1]:
        left, right = a, b
    else:
        left, right = b, a
    if a["bbox"][0] <= b["bbox"][0]:
        top_obj, bot_obj = a, b
    else:
        top_obj, bot_obj = b, a

    la0, la1, la2, la3 = left["anchor_global"]
    ra0, _ra1, ra2, _ra3 = right["anchor_global"]
    ta0, ta1, ta2, ta3 = top_obj["anchor_global"]
    _ba0, ba1, _ba2, ba3 = bot_obj["anchor_global"]

    if la0 == ra0 and la2 == ra2 and left["bbox"][3] < right["bbox"][1]:
        c0 = int(left["bbox"][3] + 1)
        c1 = int(right["bbox"][1] - 1)
        assert c0 <= c1, "horizontal pair must have a positive corridor"
        split = (c0 + c1) // 2
        left_color = _transfer_color(protos[left["type"]], bg)
        right_color = _transfer_color(protos[right["type"]], bg)
        out[la0 : la2 + 1, c0 : split + 1] = int(left_color)
        if split + 1 <= c1:
            out[la0 : la2 + 1, split + 1 : c1 + 1] = int(right_color)
        return

    if ta1 == ba1 and ta3 == ba3 and top_obj["bbox"][2] < bot_obj["bbox"][0]:
        r0 = int(top_obj["bbox"][2] + 1)
        r1 = int(bot_obj["bbox"][0] - 1)
        assert r0 <= r1, "vertical pair must have a positive corridor"
        split = (r0 + r1) // 2
        top_color = _transfer_color(protos[top_obj["type"]], bg)
        bot_color = _transfer_color(protos[bot_obj["type"]], bg)
        out[r0 : split + 1, ta1 : ta3 + 1] = int(top_color)
        if split + 1 <= r1:
            out[split + 1 : r1 + 1, ta1 : ta3 + 1] = int(bot_color)
        return

    assert False, "pair anchors must align horizontally or vertically"


def _draw_u_trail(
    out: np.ndarray,
    left: dict,
    right: dict,
    protos: list[np.ndarray],
    bg: int,
    lane: str,
    group_r0: int,
    group_r1: int,
) -> None:
    """Draw one U-shaped trail in the dedicated top or bottom lane."""
    assert left["bbox"][1] < right["bbox"][1]
    ar0, ac0, ar1, ac1 = left["anchor_global"]
    _br0, bc0, _br1, bc1 = right["anchor_global"]
    thickness = int(ar1 - ar0 + 1)
    left_color = _transfer_color(protos[left["type"]], bg)
    right_color = _transfer_color(protos[right["type"]], bg)

    if lane == "top":
        lane_r0 = int(group_r0 - thickness)
        lane_r1 = int(group_r0 - 1)
        assert lane_r0 >= 0
        out[lane_r0:group_r0, ac0 : ac1 + 1] = int(left_color)
        out[lane_r0:group_r0, bc0 : bc1 + 1] = int(right_color)
    else:
        lane_r0 = int(group_r1 + 1)
        lane_r1 = int(group_r1 + thickness)
        assert lane_r1 < out.shape[0]
        out[group_r1 + 1 : lane_r1 + 1, ac0 : ac1 + 1] = int(left_color)
        out[group_r1 + 1 : lane_r1 + 1, bc0 : bc1 + 1] = int(right_color)

    c0 = int(ac0)
    c1 = int(bc1)
    split = (c0 + c1) // 2
    out[lane_r0 : lane_r1 + 1, c0 : split + 1] = int(left_color)
    if split + 1 <= c1:
        out[lane_r0 : lane_r1 + 1, split + 1 : c1 + 1] = int(right_color)


def _draw_quartet_trails(out: np.ndarray, group: list[dict], protos: list[np.ndarray], bg: int, farthest: bool) -> None:
    """Draw the two trails for the D3-active quartet."""
    items = sorted(group, key=lambda rec: (rec["bbox"][1], rec["bbox"][0]))
    left_types = {int(items[0]["type"]), int(items[1]["type"])}
    right_types = {int(items[2]["type"]), int(items[3]["type"])}
    assert len(left_types) == 1 and len(right_types) == 1 and left_types != right_types

    row_spans = {(rec["anchor_global"][0], rec["anchor_global"][2]) for rec in items}
    assert len(row_spans) == 1, "quartet anchors must align in one horizontal band"

    group_r0 = min(int(rec["bbox"][0]) for rec in items)
    group_r1 = max(int(rec["bbox"][2]) for rec in items)

    if farthest:
        pairs = [(items[0], items[3], "top"), (items[1], items[2], "bottom")]
    else:
        pairs = [(items[0], items[2], "top"), (items[1], items[3], "bottom")]

    for left, right, lane in pairs:
        assert left["bbox"][1] < right["bbox"][1]
        _draw_u_trail(out, left, right, protos, bg, lane, group_r0, group_r1)


def _render_solution(grid: np.ndarray, farthest: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sprites = _extract_sprites(g, bg)
    protos, _base_anc_h, _base_anc_w = _recover_prototypes(sprites, bg)
    classified = _classify_sprites(g, sprites, protos, bg)

    out = g.copy()
    for s in sprites:
        for rr, cc in s["pts"]:
            out[int(rr), int(cc)] = int(bg)

    for rec in classified:
        repl = _scale_nn(protos[1 - int(rec["type"])], int(rec["scale"]))
        paste(out, repl, int(rec["place_top"]), int(rec["place_left"]), transparent=bg)

    by_scale: dict[int, list[dict]] = {}
    for rec in classified:
        by_scale.setdefault(int(rec["scale"]), []).append(rec)

    for scale, group in by_scale.items():
        if len(group) == 2:
            types = {int(group[0]["type"]), int(group[1]["type"])}
            assert types == {0, 1}, f"scale {scale} pair must contain one sprite of each prototype"
            _draw_direct_trail(out, group[0], group[1], protos, bg)
            continue

        if len(group) == 4:
            type_counts = {0: 0, 1: 0}
            for rec in group:
                type_counts[int(rec["type"])] += 1
            assert set(type_counts.values()) == {2}, f"scale {scale} quartet must contain 2 sprites of each prototype"
            _draw_quartet_trails(out, group, protos, bg, farthest=farthest)
            continue

        assert False, f"unsupported same-scale group size: {len(group)}"

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, then visualize the farthest same-scale exchanges.

    Input structure:
    - The most common color is the background. Exactly two prototype families appear, and every sprite is an integer-scaled copy of one of them.
    - One scale forms a four-sprite horizontal group with order “same type, same type, other type, other type”; another larger scale contributes a single A/B pair.

    Rule:
    - Recover the two prototypes from the smallest sprites, classify every instance, and replace each one with the opposite prototype at the same scale and anchor alignment.
    - For the four-sprite group, pair each sprite with the farthest compatible opposite-type sprite and draw a two-color trail for each pair; for the single A/B pair, draw the direct corridor trail between them.

    Key insight:
    - The anchor bbox still solves recognition, scaling, and in-place replacement, but the new visual layer comes from grouping sprites by scale: once the horizontal quartet is identified, D3/0 is the reverse-order pairing and D14/1 is just rendering those long exchanges as explicit colored paths.
    """
    return _render_solution(grid, farthest=True)


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = random.choice(LABEL_TEMPLATES)
    pool = [c for c in range(10) if c not in (bg, 1)]

    transfer_a, transfer_b = random.sample(pool, 2)
    remaining = [c for c in pool if c not in (transfer_a, transfer_b)]
    other_a = random.choice(remaining)
    remaining_b = [c for c in remaining if c != other_a]
    if random.random() < SECONDARY_SHARE_PROB:
        other_b = other_a
    else:
        other_b = random.choice(remaining_b)

    def _colorize(lab: np.ndarray, transfer: int, other: int) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = int(transfer)
        proto[lab == 3] = int(other)
        return proto

    proto_a = _colorize(labels, transfer_a, other_a)
    proto_b = _colorize(np.flipud(labels), transfer_b, other_b)
    assert _pick_anchor_bbox(proto_a, bg) == _pick_anchor_bbox(proto_b, bg)
    assert _transfer_color(proto_a, bg) != _transfer_color(proto_b, bg)
    return proto_a, proto_b


def _quartet_patch(protos: list[np.ndarray], bg: int, left_type: int) -> dict:
    """Build the scale-1 quartet that makes D3 non-degenerate."""
    scale = 1
    right_type = 1 - left_type
    types = [left_type, left_type, right_type, right_type]
    sprites = [_scale_nn(protos[t], scale) for t in types]
    anc = [_pick_anchor_bbox(arr, bg) for arr in sprites]
    thickness = int(anc[0][2] - anc[0][0] + 1)
    assert all(a == anc[0] for a in anc)

    gap01 = random.choice(QUARTET_OUTER_GAPS)
    gap12 = random.choice(QUARTET_CENTER_GAPS)
    gap23 = random.choice(QUARTET_OUTER_GAPS)

    anchor_top = thickness + max(int(a[0]) for a in anc)
    tops = [int(anchor_top - a[0]) for a in anc]
    lefts = [0]
    for i, gap in enumerate((gap01, gap12, gap23)):
        lefts.append(int(lefts[-1] + sprites[i].shape[1] + gap))

    group_r0 = min(tops)
    group_r1 = max(int(top + arr.shape[0] - 1) for top, arr in zip(tops, sprites, strict=False))
    patch_h = int(group_r1 + 1 + thickness)
    patch_w = int(lefts[-1] + sprites[-1].shape[1])

    inp = np.full((patch_h, patch_w), bg, dtype=int)
    out = np.full((patch_h, patch_w), bg, dtype=int)
    records = []
    for idx, arr in enumerate(sprites):
        top = int(tops[idx])
        left = int(lefts[idx])
        paste(inp, arr, top, left, transparent=bg)
        paste(out, _scale_nn(protos[1 - types[idx]], scale), top, left, transparent=bg)
        ar0, ac0, ar1, ac1 = anc[idx]
        records.append(
            {
                "bbox": (top, left, top + arr.shape[0] - 1, left + arr.shape[1] - 1),
                "type": int(types[idx]),
                "scale": 1,
                "anchor_global": (top + ar0, left + ac0, top + ar1, left + ac1),
            }
        )

    _draw_quartet_trails(out, records, protos, bg, farthest=True)
    reserve = np.ones((patch_h, patch_w), dtype=int)
    return {"input": inp, "output": out, "reserve": reserve}


def _support_pair_patch(
    protos: list[np.ndarray],
    bg: int,
    scale: int,
    orientation: str,
    first_type: int,
    gap: int,
) -> dict:
    """Build the extra larger-scale A/B pair with a direct trail."""
    second_type = 1 - first_type
    left_or_top = _scale_nn(protos[first_type], scale)
    right_or_bottom = _scale_nn(protos[second_type], scale)
    anc_a = _pick_anchor_bbox(left_or_top, bg)
    anc_b = _pick_anchor_bbox(right_or_bottom, bg)
    assert anc_a == anc_b

    if orientation == "h":
        top_a = 0
        top_b = 0
        left_a = 0
        left_b = int(left_or_top.shape[1] + gap)
        patch_h = int(max(left_or_top.shape[0], right_or_bottom.shape[0]))
        patch_w = int(left_b + right_or_bottom.shape[1])
    else:
        top_a = 0
        top_b = int(left_or_top.shape[0] + gap)
        left_a = 0
        left_b = 0
        patch_h = int(top_b + right_or_bottom.shape[0])
        patch_w = int(max(left_or_top.shape[1], right_or_bottom.shape[1]))

    inp = np.full((patch_h, patch_w), bg, dtype=int)
    out = np.full((patch_h, patch_w), bg, dtype=int)
    paste(inp, left_or_top, top_a, left_a, transparent=bg)
    paste(inp, right_or_bottom, top_b, left_b, transparent=bg)
    paste(out, _scale_nn(protos[second_type], scale), top_a, left_a, transparent=bg)
    paste(out, _scale_nn(protos[first_type], scale), top_b, left_b, transparent=bg)

    records = [
        {
            "bbox": (
                top_a,
                left_a,
                top_a + left_or_top.shape[0] - 1,
                left_a + left_or_top.shape[1] - 1,
            ),
            "type": int(first_type),
            "scale": int(scale),
            "anchor_global": (
                top_a + anc_a[0],
                left_a + anc_a[1],
                top_a + anc_a[2],
                left_a + anc_a[3],
            ),
        },
        {
            "bbox": (
                top_b,
                left_b,
                top_b + right_or_bottom.shape[0] - 1,
                left_b + right_or_bottom.shape[1] - 1,
            ),
            "type": int(second_type),
            "scale": int(scale),
            "anchor_global": (
                top_b + anc_b[0],
                left_b + anc_b[1],
                top_b + anc_b[2],
                left_b + anc_b[3],
            ),
        },
    ]
    _draw_direct_trail(out, records[0], records[1], protos, bg)
    reserve = np.ones((patch_h, patch_w), dtype=int)
    return {"input": inp, "output": out, "reserve": reserve}


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    patches = [
        _quartet_patch(protos, bg, left_type=random.randrange(2)),
        _support_pair_patch(
            protos,
            bg,
            scale=BIG_SCALE,
            orientation=random.choice(PAIR_ORIENTATIONS),
            first_type=random.randrange(2),
            gap=random.choice(PAIR_GAPS),
        ),
    ]
    patches.sort(key=lambda item: item["reserve"].shape[0] * item["reserve"].shape[1], reverse=True)

    max_dim = max(max(item["reserve"].shape) for item in patches)
    n_min = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_PATCHES)
    if n_min > GRID_SIZE_MAX:
        return None
    n = random.randint(n_min, GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for patch in patches:
        try:
            pr, pc = random_free_location(
                reserve,
                patch["reserve"],
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, patch["reserve"], pr, pc, transparent=bg)
        paste(inp, patch["input"], pr, pc, transparent=bg)
        paste(out, patch["output"], pr, pc, transparent=bg)

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
