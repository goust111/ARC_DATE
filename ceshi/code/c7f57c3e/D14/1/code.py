# Variant of c7f57c3e
# Dimension: D14 — 交换后的轨迹/残留 (trajectory)
# Change: 无轨迹，颜色模式直接替换，原始颜色完全消失... → 交换路径可视化

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ===============
# variant_spec:
#   preserved_core:
#     - 输入仍然只包含两类多色精灵；输出仍然先把每个精灵切换成另一类原型。
#     - 缩放仍然由 color-1 锚框决定；替换时仍按锚框对齐。
#   d14_change:
#     - 原题没有残留层；变体在每一对“同尺度、异原型”的精灵之间额外画一条轨迹带。
#     - 轨迹只画在两者之间原本的空白走廊中，不覆盖精灵本体。
#     - 轨迹分成两半：左/上半段用左/上输入精灵的“传输色”，右/下半段用另一端输入精灵的“传输色”。
#   concrete_behavior:
#     input:
#       - 恰好 4 个精灵。
#       - 只有两个尺度：1 和 2。
#       - 每个尺度恰好一对：一个 proto_A、一个 proto_B。
#       - 每对都被生成成同一条水平或竖直走廊上的一左一右/一上一下对象。
#     output:
#       - 四个精灵都在原地切换成另一类原型。
#       - 每对之间新增一条连续色带，厚度等于该尺度下 anchor bbox 的厚度。
#   small_trace_example:
#     - 旧逻辑：scale=1 的 A、B 只在原位互换，二者之间保持背景色空白。
#     - 新逻辑：互换仍在原位发生，但 A 与 B 之间的空白走廊被填成两段颜色，
#       表示 A 的传输色向 B 传递、B 的传输色向 A 传递。
#     - 差异集中在“替换之后追加一层路径绘制”。
#   complexity_calibration:
#     - 如果仍允许 3-4 个精灵且尺度分布自由，哪两个对象互为“交换对”会变得含糊。
#     - 因此把实例组织收紧成两个尺度、每个尺度一对，这是为 D14 服务的最小适配。
#     - 轨迹颜色不使用隐藏排序，而是使用“唯一与 anchor 4 邻接的变量色”，人眼可直接观察。
#   implicit_assumptions_review:
#     - 原 solver 假设“清空旧精灵后直接贴新精灵”已经完成全部任务；变体不成立，必须再画 trail。
#     - 原 generator 允许同尺度出现不成对实例；变体不成立，solver 将无法无歧义配对。
#     - 原型里若两种变量色都贴着 anchor，trail 颜色将不唯一；generator 现在保证只有 role-2 接触 anchor。
#     - 画线若穿过其他对象会破坏结果；generator 现在保留整块 pair patch（含走廊）避免重叠。
#   legality_and_visibility:
#     - 每对对象在 patch 内先按锚框对齐到同一水平带或竖直带，再留出 4-7 格空白走廊。
#     - reserve mask 使用输出 patch 的非背景并额外矩形占位，因此不同 pair 的对象和轨迹都不会互撞。
#     - 传输色在两个原型中强制不同，保证轨迹不是“看不出来的同色线”。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of color-1 pixels; all valid sprites contain them."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _transfer_color(arr: np.ndarray, bg: int) -> int:
    """Return the unique non-bg, non-1 color that touches the anchor in 4-neighborhood."""
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


def _infer_scale(arr: np.ndarray, bg: int, base_anc_h: int, base_anc_w: int) -> tuple[int, tuple[int, int, int, int]]:
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
    """Return (best_proto_idx, top, left) for an instance by overlap scoring."""
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


def _draw_pair_trail(out: np.ndarray, a: dict, b: dict, protos: list[np.ndarray], bg: int) -> None:
    """Draw the bidirectional transfer band between the two same-scale paired sprites."""
    if a["bbox"][1] <= b["bbox"][1]:
        left, right = a, b
    else:
        left, right = b, a
    if a["bbox"][0] <= b["bbox"][0]:
        top_obj, bot_obj = a, b
    else:
        top_obj, bot_obj = b, a

    la0, la1, la2, la3 = left["anchor_global"]
    ra0, ra1, ra2, ra3 = right["anchor_global"]
    ta0, ta1, ta2, ta3 = top_obj["anchor_global"]
    ba0, ba1, ba2, ba3 = bot_obj["anchor_global"]

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, then draw a transfer trail between paired sprites.

    Input structure:
    - bg is the most common color. Exactly two prototype families appear, each scaled by nearest-neighbor.
    - There are four sprite components: for each of two scales, one proto_A instance and one proto_B instance.

    Rule:
    - Recover the two prototypes from the smallest sprites, classify every instance, and replace it with the other prototype at the same scale and anchor alignment.
    - Pair the two sprites that share a scale, then fill the empty corridor between them with a two-color trail that shows the exchanged transfer colors.

    Key insight:
    - The same color-1 anchor box that stabilizes scale/alignment also gives a stable corridor band for the new trail; once same-scale A/B pairs are recovered, D14 is just “original swap + draw the transfer band between the paired anchors”.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain sprite components"

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
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototypes"
    protos = [uniq[key] for key in list(uniq.keys())[:2]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    assert (anc0[2] - anc0[0], anc0[3] - anc0[1]) == (anc1[2] - anc1[0], anc1[3] - anc1[1])
    base_anc_h = int(anc0[2] - anc0[0] + 1)
    base_anc_w = int(anc0[3] - anc0[1] + 1)

    out = g.copy()
    for s in sprites:
        for rr, cc in s["pts"]:
            out[int(rr), int(cc)] = int(bg)

    classified = []
    for s in sprites:
        scale, anc = _infer_scale(s["arr"], bg, base_anc_h, base_anc_w)
        best_idx, _top, _left = _match_proto(g, s["bbox"], s["arr"], anc, scale, protos, bg)
        repl_idx = 1 - best_idx

        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(p, bg)
        pr0 *= scale
        pc0 *= scale
        r0, c0, _r1, _c1 = s["bbox"]
        ar0, ac0, ar1, ac1 = anc
        top = int((r0 + ar0) - pr0)
        left = int((c0 + ac0) - pc0)
        paste(out, sp, top, left, transparent=bg)

        classified.append(
            {
                "bbox": s["bbox"],
                "type": int(best_idx),
                "scale": int(scale),
                "anchor_global": (
                    int(r0 + ar0),
                    int(c0 + ac0),
                    int(r0 + ar1),
                    int(c0 + ac1),
                ),
            }
        )

    by_scale: dict[int, list[dict]] = {}
    for rec in classified:
        by_scale.setdefault(int(rec["scale"]), []).append(rec)

    for scale, group in by_scale.items():
        assert len(group) == 2, f"scale {scale} must contain exactly one A/B pair"
        types = {int(group[0]["type"]), int(group[1]["type"])}
        assert types == {0, 1}, f"scale {scale} pair must contain one sprite of each prototype"
        _draw_pair_trail(out, group[0], group[1], protos, bg)

    return out


# --------------- generator ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_PAIRS = 4

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
SECONDARY_SHARE_PROB = 0.45
LABEL_TEMPLATES = [
    np.array(
        [
            [0, 2, 0],
            [1, 1, 1],
            [0, 2, 0],
            [3, 3, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 2, 2, 0],
            [1, 1, 1, 1],
            [0, 2, 2, 0],
            [3, 0, 0, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 2, 2, 0],
            [1, 1, 1, 1],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [3, 0, 0, 3],
        ],
        dtype=int,
    ),
]

# --- CFG: pairs ---
SCALES = [1, 2]
PAIR_GAP_BY_SCALE = {1: [4, 5, 6, 7], 2: [4, 5, 6]}
ORIENTATIONS = ["h", "v"]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


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
    assert _transfer_color(proto_a, bg) != _transfer_color(proto_b, bg)
    return proto_a, proto_b


def _trail_split(lo: int, hi: int) -> tuple[int, int]:
    split = (lo + hi) // 2
    return int(split), int(split + 1)


def _pair_patch(
    protos: list[np.ndarray],
    bg: int,
    scale: int,
    orientation: str,
    first_type: int,
    gap: int,
) -> dict:
    second_type = 1 - first_type

    inp_a = _scale(protos[first_type], scale)
    inp_b = _scale(protos[second_type], scale)
    out_a = _scale(protos[second_type], scale)
    out_b = _scale(protos[first_type], scale)

    anc_a = _anchor_bbox(inp_a)
    anc_b = _anchor_bbox(inp_b)
    anc_out_a = _anchor_bbox(out_a)
    anc_out_b = _anchor_bbox(out_b)
    ta = _transfer_color(protos[first_type], bg)
    tb = _transfer_color(protos[second_type], bg)

    ha, wa = inp_a.shape
    hb, wb = inp_b.shape

    if orientation == "h":
        in_top_a = max(0, anc_b[0] - anc_a[0])
        in_top_b = max(0, anc_a[0] - anc_b[0])
        in_left_a = 0
        in_left_b = wa + gap

        anchor_a = (in_top_a + anc_a[0], in_left_a + anc_a[1])
        anchor_b = (in_top_b + anc_b[0], in_left_b + anc_b[1])
        out_top_a = anchor_a[0] - anc_out_a[0]
        out_left_a = anchor_a[1] - anc_out_a[1]
        out_top_b = anchor_b[0] - anc_out_b[0]
        out_left_b = anchor_b[1] - anc_out_b[1]

        r_min = min(in_top_a, in_top_b, out_top_a, out_top_b)
        c_min = min(in_left_a, in_left_b, out_left_a, out_left_b)
        shift_r = -r_min
        shift_c = -c_min

        placements = [
            (inp_a, in_top_a + shift_r, in_left_a + shift_c),
            (inp_b, in_top_b + shift_r, in_left_b + shift_c),
            (out_a, out_top_a + shift_r, out_left_a + shift_c),
            (out_b, out_top_b + shift_r, out_left_b + shift_c),
        ]
        patch_h = max(top + arr.shape[0] for arr, top, _left in placements)
        patch_w = max(left + arr.shape[1] for arr, _top, left in placements)
        inp = np.full((patch_h, patch_w), bg, dtype=int)
        out = np.full((patch_h, patch_w), bg, dtype=int)

        paste(inp, inp_a, in_top_a + shift_r, in_left_a + shift_c, transparent=bg)
        paste(inp, inp_b, in_top_b + shift_r, in_left_b + shift_c, transparent=bg)
        paste(out, out_a, out_top_a + shift_r, out_left_a + shift_c, transparent=bg)
        paste(out, out_b, out_top_b + shift_r, out_left_b + shift_c, transparent=bg)

        band_r0 = int(anchor_a[0] + shift_r)
        band_r1 = int(anchor_a[0] + shift_r + (anc_a[2] - anc_a[0]))
        c0 = int(in_left_a + shift_c + wa)
        c1 = int(in_left_b + shift_c - 1)
        left_end, right_start = _trail_split(c0, c1)
        out[band_r0 : band_r1 + 1, c0 : left_end + 1] = int(ta)
        if right_start <= c1:
            out[band_r0 : band_r1 + 1, right_start : c1 + 1] = int(tb)
    else:
        in_left_a = max(0, anc_b[1] - anc_a[1])
        in_left_b = max(0, anc_a[1] - anc_b[1])
        in_top_a = 0
        in_top_b = ha + gap

        anchor_a = (in_top_a + anc_a[0], in_left_a + anc_a[1])
        anchor_b = (in_top_b + anc_b[0], in_left_b + anc_b[1])
        out_top_a = anchor_a[0] - anc_out_a[0]
        out_left_a = anchor_a[1] - anc_out_a[1]
        out_top_b = anchor_b[0] - anc_out_b[0]
        out_left_b = anchor_b[1] - anc_out_b[1]

        r_min = min(in_top_a, in_top_b, out_top_a, out_top_b)
        c_min = min(in_left_a, in_left_b, out_left_a, out_left_b)
        shift_r = -r_min
        shift_c = -c_min

        placements = [
            (inp_a, in_top_a + shift_r, in_left_a + shift_c),
            (inp_b, in_top_b + shift_r, in_left_b + shift_c),
            (out_a, out_top_a + shift_r, out_left_a + shift_c),
            (out_b, out_top_b + shift_r, out_left_b + shift_c),
        ]
        patch_h = max(top + arr.shape[0] for arr, top, _left in placements)
        patch_w = max(left + arr.shape[1] for arr, _top, left in placements)
        inp = np.full((patch_h, patch_w), bg, dtype=int)
        out = np.full((patch_h, patch_w), bg, dtype=int)

        paste(inp, inp_a, in_top_a + shift_r, in_left_a + shift_c, transparent=bg)
        paste(inp, inp_b, in_top_b + shift_r, in_left_b + shift_c, transparent=bg)
        paste(out, out_a, out_top_a + shift_r, out_left_a + shift_c, transparent=bg)
        paste(out, out_b, out_top_b + shift_r, out_left_b + shift_c, transparent=bg)

        band_c0 = int(anchor_a[1] + shift_c)
        band_c1 = int(anchor_a[1] + shift_c + (anc_a[3] - anc_a[1]))
        r0 = int(in_top_a + shift_r + ha)
        r1 = int(in_top_b + shift_r - 1)
        top_end, bot_start = _trail_split(r0, r1)
        out[r0 : top_end + 1, band_c0 : band_c1 + 1] = int(ta)
        if bot_start <= r1:
            out[bot_start : r1 + 1, band_c0 : band_c1 + 1] = int(tb)

    reserve = np.where(out != bg, 1, bg)
    return {"input": inp, "output": out, "reserve": reserve}


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    pairs = []
    for scale in SCALES:
        gap = random.choice(PAIR_GAP_BY_SCALE[scale])
        orientation = random.choice(ORIENTATIONS)
        first_type = random.randrange(2)
        pairs.append(_pair_patch(protos, bg, scale, orientation, first_type, gap))

    pairs.sort(key=lambda item: item["reserve"].shape[0] * item["reserve"].shape[1], reverse=True)
    max_dim = max(max(item["reserve"].shape) for item in pairs)
    n_min = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_PAIRS)
    if n_min > GRID_SIZE_MAX:
        return None
    n = random.randint(max(n_min, GRID_SIZE_MAX - 2), GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for pair in pairs:
        marker = pair["reserve"]
        try:
            pr, pc = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, pr, pc, transparent=bg)
        paste(inp, pair["input"], pr, pc, transparent=bg)
        paste(out, pair["output"], pr, pc, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D14 input")
