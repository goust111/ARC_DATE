# Variant of c7f57c3e
# Dimension: D5 — 目标位置/锚定方式 (dst)
# Change: 交换后精灵保持在与配对精灵最大重叠的位置，本质上近似原位交换... → 重力下落到底部

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# yaml-like notes:
#   concrete_behavior:
#     input:
#       - 仍然只有两类多色 sprite prototype；每个实例都是其中之一的整数倍放大副本。
#       - color 1 仍然充当内部锚区，用来识别 scale 并确定“交换后的起始位置”。
#     output:
#       - 每个 sprite 先切换到另一类 prototype，scale 不变。
#       - 不再停在原位，而是沿固定列带竖直下落，直到撞到网格底边或先落稳的 sprite。
#       - 因此同列带中的多个 sprite 会在底部堆叠，视觉上能直接看出“交换 + 重力”两步。
#     trace_example:
#       - 原题: 两个同列的小/大 sprite 交换颜色模式后，仍锚定在原 bbox 附近。
#       - 本变体: 先按同样的锚点规则得到交换后的 patch，再让较低者先落到底，再让较高者落到它上面。
#       - 差异: 锚点只决定横向和初始纵向，最终目标位置由重力和碰撞决定。
#   complexity_calibration:
#     - 只改 D5，不改 prototype 识别、scale 推断、prototype 切换关系。
#     - generator 至少构造一个含 2 个 sprite 的竖直 lane，避免退化成“所有对象各自落到底、互不作用”。
#     - 保留 3-4 个对象、仅 1 个大尺度对象，复杂度仍接近原题；不会引入额外颜色规则或多种物理模式。
#   hidden_assumptions_review:
#     - 原 main 在逐个 sprite 替换后立即 paste；这依赖“最终位置与其他 sprite 无关”。
#       变体中该前提失效，所以必须先识别全部 replacement，再统一做重力结算。
#     - 原 main 的 prototype 恢复依赖:
#       1. 输入中最小 bbox 面积的 sprite 中恰好包含两个 scale-1 prototype；
#       2. color 1 锚区在每个实例中都存在，且按整数倍放大；
#       3. 输入 sprite 之间 8-邻域断开。
#       这些前提在变体里继续成立，generator 会显式保证。
#     - 新 main 额外依赖:
#       1. 交换后的起始 patch 在网格内；
#       2. 同一 lane 中起始 patch 彼此不重叠；
#       3. 用“较低起点先结算”的顺序可模拟稳定堆叠。
#       generator 通过固定 lane、留出纵向间距和边距来保证。
#   placement_legality:
#     - 输入仍要求对象不重叠、不越界、且 bbox 之间至少留空，避免 components 合并。
#     - 输出 lane 之间保留水平空隙，避免不同堆叠互相接触。
#   visual_observability:
#     - prototype 保留原题的不对称内部结构与换色，因此交换可见。
#     - 至少一个 lane 出现真实堆叠，因此“重力”可从少量样例直接观察。
#   cleanliness_and_color_inference:
#     - 不使用颜色 id 排序；重力只依赖几何位置。
#     - 颜色变化仍完全由“切换到另一 prototype”决定，和原题同一套可见线索一致。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick a stable anchor bbox inside the sprite array (color-1 pixels; always present)."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _extract_sprites(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    return sprites


def _infer_prototypes(sprites: list[dict]) -> tuple[list[np.ndarray], int, int]:
    areas = [
        int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1))
        for s in sprites
    ]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype patterns"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]
    anc0 = _pick_anchor_bbox(protos[0], int(get_bg(protos[0])))
    base_anc_h = int(anc0[2] - anc0[0] + 1)
    base_anc_w = int(anc0[3] - anc0[1] + 1)
    return protos, base_anc_h, base_anc_w


def _classify_sprite(
    g: np.ndarray,
    bg: int,
    protos: list[np.ndarray],
    base_anc_h: int,
    base_anc_w: int,
    sprite: dict,
) -> dict:
    h, w = g.shape
    r0, c0, _r1, _c1 = sprite["bbox"]
    arr = sprite["arr"]
    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, ar1, ac1 = anc
    inst_anc_h = int(ar1 - ar0 + 1)
    inst_anc_w = int(ac1 - ac0 + 1)

    scale_h = max(1, inst_anc_h // max(1, base_anc_h))
    scale_w = max(1, inst_anc_w // max(1, base_anc_w))
    scale = int(max(scale_h, scale_w))

    best_idx = 0
    best_score = -1
    for idx, p in enumerate(protos):
        sp = _scale_nn(p, scale)
        panc = _pick_anchor_bbox(p, bg)
        pr0, pc0 = int(panc[0] * scale), int(panc[1] * scale)
        top = int((r0 + ar0) - pr0)
        left = int((c0 + ac0) - pc0)

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        sub_g = g[rr0:rr1, cc0:cc1]
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = idx

    repl_idx = 1 - best_idx
    repl_proto = protos[repl_idx]
    repl_patch = _scale_nn(repl_proto, scale)
    panc = _pick_anchor_bbox(repl_proto, bg)
    pr0, pc0 = int(panc[0] * scale), int(panc[1] * scale)
    start_top = int((r0 + ar0) - pr0)
    start_left = int((c0 + ac0) - pc0)

    return {
        "kind": int(best_idx),
        "scale": scale,
        "start_top": start_top,
        "start_left": start_left,
        "patch": repl_patch,
    }


def _prepare_replacements(g: np.ndarray, bg: int) -> list[dict]:
    sprites = _extract_sprites(g, bg)
    protos, base_anc_h, base_anc_w = _infer_prototypes(sprites)
    return [_classify_sprite(g, bg, protos, base_anc_h, base_anc_w, s) for s in sprites]


def _can_place(board: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    H, W = board.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > H or left + pw > W:
        return False
    sub = board[top : top + ph, left : left + pw]
    return not bool(((patch != bg) & (sub != bg)).any())


def _render_without_gravity(shape: tuple[int, int], bg: int, replacements: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for item in replacements:
        top = int(item["start_top"])
        left = int(item["start_left"])
        patch = item["patch"]
        assert _can_place(out, patch, top, left, bg), "replacement patch must fit before gravity"
        paste(out, patch, top, left, transparent=bg)
    return out


def _settle_replacements(
    shape: tuple[int, int],
    bg: int,
    replacements: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    H, _W = shape
    board = np.full(shape, bg, dtype=int)
    landed = []

    order = sorted(
        replacements,
        key=lambda item: (
            int(item["start_top"]) + int(item["patch"].shape[0]),
            int(item["start_top"]),
            -int(item["start_left"]),
        ),
        reverse=True,
    )

    for item in order:
        patch = item["patch"]
        ph, _pw = patch.shape
        top = int(item["start_top"])
        left = int(item["start_left"])
        assert _can_place(board, patch, top, left, bg), "transformed patches must start non-overlapping"

        while top + ph < H and _can_place(board, patch, top + 1, left, bg):
            top += 1

        paste(board, patch, top, left, transparent=bg)
        landed.append(
            {
                "start_top": int(item["start_top"]),
                "start_left": left,
                "final_top": top,
                "patch": patch,
            }
        )

    return board, landed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, then let the swapped sprite fall straight down.

    Input structure:
    - bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
    - Each connected sprite instance is a scaled copy of one prototype and contains a color-1 anchor region.

    Rule:
    - Recover the two prototype patterns from the smallest sprites, then identify each instance's prototype and integer scale.
    - Replace each instance with the other prototype at the same scale, aligned by the color-1 anchor bbox just as in the original task.
    - After swapping, drop every replacement vertically until it hits the bottom edge or an already settled replacement.

    Key insight: The anchor still determines which swapped patch belongs to each sprite and where that patch starts, but the final destination is no longer "stay near the old sprite" — gravity reorders the vertical layout into stable bottom-up stacks.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    replacements = _prepare_replacements(g, bg)
    out, _landed = _settle_replacements(g.shape, bg, replacements)
    return out


# --------------- generator (d5 gravity variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 8
INPUT_BORDER = 1

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (3, 4), (4, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.65

# --- CFG: row-mask weights (anchor role vs variable role) ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: gravity layout ---
LANE_GAP = 3
INTRA_LANE_ROW_GAP = (2, 4)
MIN_FALL_DISTANCE = 3
INPUT_PADDING = 2

# --- CFG: generation retries ---
MAX_RETRIES = 400

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _make_labels() -> np.ndarray:
    h, w = random.choice(PROTO_SHAPES)

    counts = {1: 1, 2: 1, 3: 1}
    if h > 3:
        counts[random.choice([1, 2, 3])] += h - 3
    roles = []
    for role in random.sample([1, 2, 3], 3):
        roles.extend([role] * counts[role])

    weights_by_role = {1: ANCHOR_ROW_WEIGHTS, 2: VARIABLE_ROW_WEIGHTS, 3: VARIABLE_ROW_WEIGHTS}
    masks = []
    for role in roles:
        kinds, wts = zip(*weights_by_role[role])
        masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

    if not any(mask[0] or mask[-1] for mask in masks):
        i = random.randrange(h)
        masks[i] = _row_mask("outer", w) if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB else _row_mask("full", w)

    if all(all(mask) for mask in masks):
        masks[random.randrange(h)] = _row_mask("center", w)

    labels = np.zeros((h, w), dtype=int)
    for r, (role, mask) in enumerate(zip(roles, masks)):
        for c, bit in enumerate(mask):
            if bit:
                labels[r, c] = role
    return labels


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
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


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int, padding: int) -> bool:
    return not (a1 + padding < b0 or b1 + padding < a0)


def _bboxes_separated(bboxes: list[tuple[int, int, int, int]], padding: int) -> bool:
    for i, (r0, c0, r1, c1) in enumerate(bboxes):
        for rr0, cc0, rr1, cc1 in bboxes[i + 1 :]:
            rows_hit = _intervals_overlap(r0, r1, rr0, rr1, padding)
            cols_hit = _intervals_overlap(c0, c1, cc0, cc1, padding)
            if rows_hit and cols_hit:
                return False
    return True


def _lane_sizes(count: int) -> list[int]:
    return [2, count - 2]


def _distribute_lane_lefts(n: int, widths: list[int]) -> list[int] | None:
    total = sum(widths) + LANE_GAP * (len(widths) - 1) + 2 * INPUT_BORDER
    if total > n:
        return None
    slack = n - total
    extras = np.random.multinomial(slack, np.ones(len(widths) + 1) / (len(widths) + 1))
    lefts = []
    c = INPUT_BORDER + int(extras[0])
    for i, width in enumerate(widths):
        lefts.append(c)
        c += int(width)
        if i < len(widths) - 1:
            c += LANE_GAP + int(extras[i + 1])
    return lefts


def _build_instances(protos: list[np.ndarray], bg: int) -> list[dict]:
    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    specs = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        specs.append((random.randrange(2), 1))

    instances = []
    for input_type, scale in specs:
        input_patch = _scale(protos[input_type], scale)
        output_patch = _scale(protos[1 - input_type], scale)
        in_anchor = _anchor_bbox(input_patch)
        out_anchor = _anchor_bbox(output_patch)
        instances.append(
            {
                "input_type": int(input_type),
                "scale": int(scale),
                "input_patch": input_patch,
                "output_patch": output_patch,
                "dr": int(in_anchor[0] - out_anchor[0]),
                "dc": int(in_anchor[1] - out_anchor[1]),
            }
        )
    return instances


def _assign_lanes(instances: list[dict]) -> list[list[int]]:
    order = list(range(len(instances)))
    random.shuffle(order)
    order.sort(key=lambda idx: instances[idx]["scale"], reverse=True)
    sizes = _lane_sizes(len(instances))
    lanes = []
    cursor = 0
    for size in sizes:
        lanes.append(order[cursor : cursor + size])
        cursor += size
    return lanes


def _choose_swap_tops(lane: list[int], instances: list[dict], n: int) -> dict[int, int] | None:
    ordered = lane[:]
    random.shuffle(ordered)
    ordered.sort(key=lambda idx: instances[idx]["scale"], reverse=True)

    gaps = [random.randint(*INTRA_LANE_ROW_GAP) for _ in range(max(0, len(ordered) - 1))]
    required_above = 0
    for idx, gap in zip(ordered[1:], gaps, strict=False):
        required_above += int(instances[idx]["output_patch"].shape[0]) + gap

    bottom_idx = ordered[0]
    bottom_h = int(instances[bottom_idx]["output_patch"].shape[0])
    min_bottom_top = INPUT_BORDER + required_above
    max_bottom_top = n - INPUT_BORDER - bottom_h - MIN_FALL_DISTANCE
    if min_bottom_top > max_bottom_top:
        return None

    tops = {}
    cur_top = random.randint(min_bottom_top, max_bottom_top)
    tops[bottom_idx] = cur_top
    for idx, gap in zip(ordered[1:], gaps, strict=False):
        cur_top = cur_top - gap - int(instances[idx]["output_patch"].shape[0])
        tops[idx] = cur_top
    return tops


def _layout_instances(instances: list[dict], n: int) -> tuple[list[dict], list[dict]] | None:
    lanes = _assign_lanes(instances)
    lane_widths = [
        max(max(int(instances[idx]["input_patch"].shape[1]), int(instances[idx]["output_patch"].shape[1])) for idx in lane)
        for lane in lanes
    ]
    lane_lefts = _distribute_lane_lefts(n, lane_widths)
    if lane_lefts is None:
        return None

    placed_inputs = []
    replacements = []

    for lane, lane_left in zip(lanes, lane_lefts, strict=False):
        tops = _choose_swap_tops(lane, instances, n)
        if tops is None:
            return None
        for idx in lane:
            inst = instances[idx]
            input_patch = inst["input_patch"]
            output_patch = inst["output_patch"]
            swap_top = int(tops[idx])
            swap_left = int(lane_left)
            input_r = int(swap_top - inst["dr"])
            input_c = int(swap_left - inst["dc"])

            ih, iw = input_patch.shape
            if (
                input_r < INPUT_BORDER
                or input_c < INPUT_BORDER
                or input_r + ih > n - INPUT_BORDER
                or input_c + iw > n - INPUT_BORDER
            ):
                return None

            placed_inputs.append(
                {
                    "idx": idx,
                    "top": input_r,
                    "left": input_c,
                    "patch": input_patch,
                    "bbox": (input_r, input_c, input_r + ih - 1, input_c + iw - 1),
                }
            )
            replacements.append(
                {
                    "idx": idx,
                    "start_top": swap_top,
                    "start_left": swap_left,
                    "patch": output_patch,
                }
            )

    if not _bboxes_separated([item["bbox"] for item in placed_inputs], INPUT_PADDING):
        return None

    return placed_inputs, replacements


def _count_moving_replacements(landed: list[dict]) -> int:
    return sum(int(item["final_top"] > item["start_top"]) for item in landed)


def _count_supported_replacements(landed: list[dict], bg: int, shape: tuple[int, int]) -> int:
    H, W = shape
    board = np.full(shape, bg, dtype=int)
    supported = 0
    for item in sorted(landed, key=lambda d: int(d["final_top"]), reverse=True):
        patch = item["patch"]
        top = int(item["final_top"])
        left = int(item["start_left"])
        ph, pw = patch.shape
        if top + ph < H:
            below = board[top + ph, left : left + pw]
            if np.any((patch[-1] != bg) & (below != bg)):
                supported += 1
        paste(board, patch, top, left, transparent=bg)
    return supported


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos = list(_make_prototypes(bg))
    instances = _build_instances(protos, bg)

    max_dim = max(
        max(max(inst["input_patch"].shape), max(inst["output_patch"].shape))
        for inst in instances
    )
    lane_width_floor = sum(
        max(int(inst["input_patch"].shape[1]), int(inst["output_patch"].shape[1]))
        for inst in sorted(instances, key=lambda d: d["scale"], reverse=True)[:2]
    )
    min_n = max(
        GRID_SIZE_MIN,
        max_dim + GRID_MARGIN_FOR_SPRITES,
        lane_width_floor + LANE_GAP + 2 * INPUT_BORDER,
    )
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    layout = _layout_instances(instances, n)
    if layout is None:
        return None
    placed_inputs, replacements = layout

    inp = np.full((n, n), bg, dtype=int)
    for item in placed_inputs:
        paste(inp, item["patch"], item["top"], item["left"], transparent=bg)

    if len(components_mask(inp != bg, connectivity=8)) != len(instances):
        return None

    expected_out, landed = _settle_replacements(inp.shape, bg, replacements)
    no_gravity = _render_without_gravity(inp.shape, bg, replacements)
    if np.array_equal(expected_out, no_gravity):
        return None
    if _count_moving_replacements(landed) < 2:
        return None
    if _count_supported_replacements(landed, bg, inp.shape) < 1:
        return None

    try:
        if np.array_equal(main(inp), expected_out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D5 gravity input")
