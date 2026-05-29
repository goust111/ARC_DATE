# Combo variant of c7f57c3e (index 3)
# combo_spec: D3/1+D5/2
# Name: 完全位置交换 + 重力下落

from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# ===============
# a) 具体化与行为追踪:
#    - 输入固定为四个 sprite, 组成两行局部 A/B 近邻对:
#      上排是 A_small 靠近 B_big, 下排是 A_big 靠近 B_small.
#    - 原题会对每个 sprite 原地切到另一 prototype, 然后基本留在原位置。
#    - 本融合版先按近邻恢复两对 A/B 配对, 再做“跨对完全位置交换”:
#      A_small 不再留在左上原位, 而是生成一个 B_small patch 到其右侧 B_big 的锚点位置;
#      B_big 则生成一个 A_big patch 到左上锚点位置。下排同理。
#    - 这样得到的四个“交换后起始 patch”再统一做竖直重力结算, 最终在左右两列底部堆叠。
#      差异是可见的两步流水线: 先横向/跨尺度互换落点, 再纵向下落形成堆积。
# b) 复杂度校准:
#    - 若让配对发生在同尺度 A/B 之间, 则“换位置 + 换 prototype”容易退化得过于接近原题。
#      因此固定成两个跨尺度配对: A1<->Bk 与 Ak<->B1, 其中 k in {2, 3}。
#    - 四个对象足以给出两次重复证据: 两个局部近邻对支撑配对规则, 两个列堆叠支撑重力规则。
#    - 不再引入额外颜色语义或多种物理模式, 复杂度集中在“先配对换位, 再重力”的主链路上。
# c) 隐式假设审查:
#    - 原 solver 假设每个 sprite 可独立原地重绘; 本变体不成立, 因为起始位置取决于配对对象的位置。
#      所以必须先完成: prototype 恢复 -> 全体分类 -> A/B 配对 -> 生成全部交换后 patch -> 统一重力结算。
#    - 原 solver 只需知道“这个 sprite 属于哪类、尺度多少”; 本变体还依赖 A/B 数量平衡且近邻配对无歧义。
#    - 重力又新增了“交换后起始 patch 彼此不重叠、且较低者先落稳”的要求; generator 必须显式保证。
# d) 放置合法性:
#    - generator 对四个锚点使用规则化的 2x2 布局, 但每个槽位都按“输入 patch + 交换后会落到这里的 patch”
#      的并集包围盒预留空间, 防止跨尺度互换后越界或相撞。
#    - 输入阶段与“交换后起始阶段”都要求四个 bbox 断开; 输出阶段则只允许在重力下形成列内堆叠。
# e) 视觉可观察性:
#    - 人眼可以直接看到: 输入是两行 A/B 邻对; 输出先把大/小对象横向送到对侧槽位, 再在左右两列向下堆叠。
#    - 规则完全由可见几何关系决定: prototype 结构、scale、相对近邻、以及向下掉落; 不依赖颜色编号大小。
# f) 可视化整洁度:
#    - 输入保持 2x2 清晰分区, 对象之间留空, 不会因重叠或贴边导致组件混淆。
#    - 输出只形成两列竖直堆叠, 不让多列互相接触, 视觉上仍然干净可读。
# g) 变化可推断性:
#    - 颜色变化仍只来自“切到另一 prototype”, 与原题一致, 但现在与跨尺度换位同时出现。
#    - 由于两列各出现一次大/小堆叠, 解题者能从少量样例观察到“对象先去对侧槽位, 再整体受重力”的一致模式。
# h) 新 main() 假设与 generator 对应保证:
#    - 最小面积 sprite 中必须同时出现两个 scale-1 prototype: generator 固定放置 A1 与 B1。
#    - 所有实例都是整数倍缩放且 color-1 anchor bbox 同比例缩放: generator 仅用 NN scaling。
#    - A/B 数量必须平衡且最佳匹配是按行配对: generator 固定两行各一对 A/B。
#    - 交换后起始 patch 必须在网格内且互不重叠: generator 用槽位并集包围盒和行列间距保证。
#    - 重力排序按较低起点先结算才能得到稳定堆叠: generator 让每列两个起始 patch 共用水平带且起始上下分离。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 pixels inside a sprite patch."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(grid != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))
    return sprites


def _infer_prototypes(sprites: list[dict], bg: int) -> tuple[list[np.ndarray], tuple[int, int]]:
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
    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_hw0 = (anc0[2] - anc0[0] + 1, anc0[3] - anc0[1] + 1)
    base_hw1 = (anc1[2] - anc1[0] + 1, anc1[3] - anc1[1] + 1)
    assert base_hw0 == base_hw1, "prototypes must share the same anchor bbox size"
    return protos, base_hw0


def _infer_scale(arr: np.ndarray, base_anchor_hw: tuple[int, int], bg: int) -> int:
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    inst_h = ar1 - ar0 + 1
    inst_w = ac1 - ac0 + 1
    base_h, base_w = base_anchor_hw
    assert inst_h % base_h == 0 and inst_w % base_w == 0, "anchor bbox must scale integrally"
    scale_h = inst_h // base_h
    scale_w = inst_w // base_w
    assert scale_h == scale_w and scale_h >= 1, "sprite scale must be a positive integer"
    return int(scale_h)


def _match_prototype(
    grid: np.ndarray,
    sprite: dict,
    protos: list[np.ndarray],
    scale: int,
    bg: int,
) -> int:
    """Score which prototype matches the observed sprite when anchor-aligned."""
    h, w = grid.shape
    r0, c0, _r1, _c1 = sprite["bbox"]
    arr = sprite["arr"]
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + sp.shape[0])
        cc1 = min(w, left + sp.shape[1])
        if rr0 >= rr1 or cc0 >= cc1:
            continue

        sub_g = grid[rr0:rr1, cc0:cc1]
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        mask = sub_p != bg
        score = int(np.sum(sub_g[mask] == sub_p[mask])) if np.any(mask) else 0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _anchor_pos_from_bbox(bbox: tuple[int, int, int, int], arr: np.ndarray, bg: int) -> tuple[int, int]:
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)
    return int(r0 + ar0), int(c0 + ac0)


def _top_left_from_anchor(patch: np.ndarray, anchor_pos: tuple[int, int], bg: int) -> tuple[int, int]:
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(patch, bg)
    return int(anchor_pos[0] - ar0), int(anchor_pos[1] - ac0)


def _build_patch(protos: list[np.ndarray], type_idx: int, scale: int) -> np.ndarray:
    return _scale_nn(protos[type_idx], scale)


def _classify_sprites(grid: np.ndarray, bg: int) -> tuple[list[np.ndarray], list[dict]]:
    sprites = _extract_sprites(grid, bg)
    protos, base_anchor_hw = _infer_prototypes(sprites, bg)

    for sprite in sprites:
        scale = _infer_scale(sprite["arr"], base_anchor_hw, bg)
        type_idx = _match_prototype(grid, sprite, protos, scale, bg)
        sprite["scale"] = scale
        sprite["type_idx"] = type_idx
        sprite["anchor_pos"] = _anchor_pos_from_bbox(sprite["bbox"], sprite["arr"], bg)
    return protos, sprites


def _best_opposite_pairing(group_a: list[dict], group_b: list[dict]) -> list[tuple[dict, dict]]:
    """Minimum-total-distance bipartite pairing between A and B sprites."""
    assert len(group_a) == len(group_b) and len(group_a) > 0
    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(group_b))):
        score = 0
        for ia, ib in enumerate(perm):
            ra, ca = group_a[ia]["anchor_pos"]
            rb, cb = group_b[ib]["anchor_pos"]
            score += abs(ra - rb) + abs(ca - cb)
        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return [(group_a[i], group_b[j]) for i, j in enumerate(best_perm)]


def _prepare_inplace_replacements(grid: np.ndarray, bg: int) -> list[dict]:
    protos, sprites = _classify_sprites(grid, bg)
    replacements = []
    for sprite in sprites:
        repl_patch = _build_patch(protos, 1 - int(sprite["type_idx"]), int(sprite["scale"]))
        start_top, start_left = _top_left_from_anchor(repl_patch, sprite["anchor_pos"], bg)
        replacements.append(
            {
                "patch": repl_patch,
                "start_top": start_top,
                "start_left": start_left,
                "src_anchor": sprite["anchor_pos"],
                "dst_anchor": sprite["anchor_pos"],
            }
        )
    return replacements


def _prepare_combo_replacements(grid: np.ndarray, bg: int) -> list[dict]:
    protos, sprites = _classify_sprites(grid, bg)

    by_type: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        by_type[int(sprite["type_idx"])].append(sprite)

    group_a = sorted(by_type.get(0, []), key=lambda s: s["anchor_pos"])
    group_b = sorted(by_type.get(1, []), key=lambda s: s["anchor_pos"])
    assert len(group_a) == len(group_b) and len(group_a) > 0, "grid must contain balanced A/B sprites"

    replacements = []
    for sa, sb in _best_opposite_pairing(group_a, group_b):
        patch_a = _build_patch(protos, 1 - int(sa["type_idx"]), int(sa["scale"]))
        patch_b = _build_patch(protos, 1 - int(sb["type_idx"]), int(sb["scale"]))
        top_a, left_a = _top_left_from_anchor(patch_a, sb["anchor_pos"], bg)
        top_b, left_b = _top_left_from_anchor(patch_b, sa["anchor_pos"], bg)
        replacements.append(
            {
                "patch": patch_a,
                "start_top": top_a,
                "start_left": left_a,
                "src_anchor": sa["anchor_pos"],
                "dst_anchor": sb["anchor_pos"],
            }
        )
        replacements.append(
            {
                "patch": patch_b,
                "start_top": top_b,
                "start_left": left_b,
                "src_anchor": sb["anchor_pos"],
                "dst_anchor": sa["anchor_pos"],
            }
        )
    return replacements


def _can_place(board: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = board.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
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
    h, _w = shape
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

        while top + ph < h and _can_place(board, patch, top + 1, left, bg):
            top += 1

        paste(board, patch, top, left, transparent=bg)
        landed.append(
            {
                "start_top": int(item["start_top"]),
                "start_left": left,
                "final_top": top,
                "patch": patch,
                "src_anchor": item["src_anchor"],
                "dst_anchor": item["dst_anchor"],
            }
        )

    return board, landed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each nearby opposite-pattern sprite, send its swapped prototype to the partner position, then let the result fall.

    Input structure:
    - The background is the most frequent color, and exactly two multicolor sprite prototypes appear as disconnected components.
    - The board contains four sprite instances total: two prototype-A copies and two prototype-B copies, with one small and one large scale for each family.

    Rule:
    - Recover the two scale-1 prototypes, classify every sprite by prototype and integer scale, and match A sprites with B sprites by nearest-anchor bipartite pairing.
    - For each matched pair, keep each sprite's own scale, switch it to the opposite prototype, and place that swapped patch at the partner's anchor position.
    - After all pairwise exchanges are prepared, drop every swapped patch straight downward until it hits the bottom edge or a previously settled patch.

    Key insight:
    - The task is not an in-place recoloring problem anymore: the color-1 anchor is used twice, first to recover which opposite-pattern sprites belong together and where their exchanged start positions are, and then as the fixed horizontal guide for the gravity phase that stacks the exchanged patches.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    replacements = _prepare_combo_replacements(g, bg)
    out, _landed = _settle_replacements(g.shape, bg, replacements)
    return out


# --------------- generator (combo variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
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
BIG_SCALES = [2, 3]

# --- CFG: layout ---
COLUMN_GAP = (2, 5)
ROW_GAP = (2, 4)
INPUT_PADDING = 2
MIN_FALL_DISTANCE = 3

# --- CFG: generation retries ---
MAX_RETRIES = 600

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
        if roles[i] == 1 and random.random() < OUTER_FIXUP_PROB:
            masks[i] = _row_mask("outer", w)
        else:
            masks[i] = _row_mask("full", w)

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


def _extents_around_anchor(patch: np.ndarray) -> tuple[int, int, int, int]:
    ar0, ac0, _ar1, _ac1 = _anchor_bbox(patch)
    h, w = patch.shape
    return int(ar0), int(ac0), int(h - ar0 - 1), int(w - ac0 - 1)


def _union_extents(patches: list[np.ndarray]) -> tuple[int, int, int, int]:
    ups, lefts, downs, rights = zip(*[_extents_around_anchor(p) for p in patches])
    return int(max(ups)), int(max(lefts)), int(max(downs)), int(max(rights))


def _bbox_from_anchor(patch: np.ndarray, anchor_pos: tuple[int, int]) -> tuple[int, int, int, int]:
    ar0, ac0, _ar1, _ac1 = _anchor_bbox(patch)
    top = int(anchor_pos[0] - ar0)
    left = int(anchor_pos[1] - ac0)
    return top, left, top + int(patch.shape[0]) - 1, left + int(patch.shape[1]) - 1


def _paste_at_anchor(dst: np.ndarray, patch: np.ndarray, anchor_pos: tuple[int, int], bg: int) -> None:
    ar0, ac0, _ar1, _ac1 = _anchor_bbox(patch)
    paste(dst, patch, anchor_pos[0] - ar0, anchor_pos[1] - ac0, transparent=bg)


def _build_instances(protos: list[np.ndarray], big_scale: int) -> list[dict]:
    specs = [
        {"pair": 0, "slot": 0, "input_type": 0, "scale": 1},
        {"pair": 0, "slot": 1, "input_type": 1, "scale": big_scale},
        {"pair": 1, "slot": 2, "input_type": 0, "scale": big_scale},
        {"pair": 1, "slot": 3, "input_type": 1, "scale": 1},
    ]

    instances = []
    for spec in specs:
        input_patch = _scale(protos[spec["input_type"]], spec["scale"])
        swapped_patch = _scale(protos[1 - spec["input_type"]], spec["scale"])
        instances.append(
            {
                "pair": spec["pair"],
                "slot": spec["slot"],
                "input_type": spec["input_type"],
                "scale": spec["scale"],
                "input_patch": input_patch,
                "swapped_patch": swapped_patch,
            }
        )
    return instances


def _slot_candidates(instances: list[dict]) -> dict[int, list[np.ndarray]]:
    by_slot = {
        0: [instances[0]["input_patch"], instances[1]["swapped_patch"]],
        1: [instances[1]["input_patch"], instances[0]["swapped_patch"]],
        2: [instances[2]["input_patch"], instances[3]["swapped_patch"]],
        3: [instances[3]["input_patch"], instances[2]["swapped_patch"]],
    }
    return by_slot


def _layout_anchors(instances: list[dict], n: int) -> dict[int, tuple[int, int]] | None:
    slot_cands = _slot_candidates(instances)

    left_slots = slot_cands[0] + slot_cands[2]
    right_slots = slot_cands[1] + slot_cands[3]
    row_slots = slot_cands[0] + slot_cands[1] + slot_cands[2] + slot_cands[3]

    left_up, left_left, left_down, left_right = _union_extents(left_slots)
    right_up, right_left, right_down, right_right = _union_extents(right_slots)
    row_up, _row_left, row_down, _row_right = _union_extents(row_slots)

    left_block_w = left_left + left_right + 1
    right_block_w = right_left + right_right + 1
    row_block_h = row_up + row_down + 1

    base_col_gap = random.randint(*COLUMN_GAP)
    base_row_gap = random.randint(*ROW_GAP)

    need_w = INPUT_BORDER + left_block_w + base_col_gap + right_block_w + INPUT_BORDER
    need_h = INPUT_BORDER + row_block_h + base_row_gap + row_block_h + MIN_FALL_DISTANCE + INPUT_BORDER
    if need_w > n or need_h > n:
        return None

    slack_w = n - need_w
    slack_h = n - need_h

    extra_left = random.randint(0, slack_w)
    extra_between_w = random.randint(0, slack_w - extra_left)
    extra_top = random.randint(0, slack_h)
    extra_between_h = random.randint(0, slack_h - extra_top)

    left_block_left = INPUT_BORDER + extra_left
    right_block_left = left_block_left + left_block_w + base_col_gap + extra_between_w
    top_block_top = INPUT_BORDER + extra_top
    bottom_block_top = top_block_top + row_block_h + base_row_gap + extra_between_h

    anchor_c_left = left_block_left + left_left
    anchor_c_right = right_block_left + right_left
    anchor_r_top = top_block_top + row_up
    anchor_r_bottom = bottom_block_top + row_up

    return {
        0: (anchor_r_top, anchor_c_left),
        1: (anchor_r_top, anchor_c_right),
        2: (anchor_r_bottom, anchor_c_left),
        3: (anchor_r_bottom, anchor_c_right),
    }


def _prepare_layout(instances: list[dict], anchors: dict[int, tuple[int, int]]) -> tuple[list[dict], list[dict]]:
    placed_inputs = []
    replacements = []

    partner_slot = {0: 1, 1: 0, 2: 3, 3: 2}
    for inst in instances:
        slot = int(inst["slot"])
        input_anchor = anchors[slot]
        partner_anchor = anchors[partner_slot[slot]]

        input_bbox = _bbox_from_anchor(inst["input_patch"], input_anchor)
        placed_inputs.append(
            {
                "slot": slot,
                "top": input_bbox[0],
                "left": input_bbox[1],
                "patch": inst["input_patch"],
                "bbox": input_bbox,
            }
        )

        swap_bbox = _bbox_from_anchor(inst["swapped_patch"], partner_anchor)
        replacements.append(
            {
                "slot": slot,
                "pair": int(inst["pair"]),
                "start_top": swap_bbox[0],
                "start_left": swap_bbox[1],
                "patch": inst["swapped_patch"],
                "src_anchor": input_anchor,
                "dst_anchor": partner_anchor,
                "bbox": swap_bbox,
            }
        )

    return placed_inputs, replacements


def _count_moving_replacements(landed: list[dict]) -> int:
    return sum(int(item["final_top"] > item["start_top"]) for item in landed)


def _count_supported_replacements(landed: list[dict], bg: int, shape: tuple[int, int]) -> int:
    h, w = shape
    board = np.full(shape, bg, dtype=int)
    supported = 0
    for item in sorted(landed, key=lambda d: int(d["final_top"]), reverse=True):
        patch = item["patch"]
        top = int(item["final_top"])
        left = int(item["start_left"])
        ph, pw = patch.shape
        if top + ph < h:
            below = board[top + ph, left : left + pw]
            if np.any((patch[-1] != bg) & (below != bg)):
                supported += 1
        paste(board, patch, top, left, transparent=bg)
    return supported


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos = list(_make_prototypes(bg))
    big_scale = random.choice(BIG_SCALES)
    instances = _build_instances(protos, big_scale)

    all_patches = [inst["input_patch"] for inst in instances] + [inst["swapped_patch"] for inst in instances]
    max_dim = max(max(p.shape) for p in all_patches)
    min_n = max(GRID_SIZE_MIN, max_dim * 2 + INPUT_BORDER * 2 + 2)
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    anchors = _layout_anchors(instances, n)
    if anchors is None:
        return None

    placed_inputs, replacements = _prepare_layout(instances, anchors)
    if not _bboxes_separated([item["bbox"] for item in placed_inputs], INPUT_PADDING):
        return None
    if not _bboxes_separated([item["bbox"] for item in replacements], 0):
        return None

    inp = np.full((n, n), bg, dtype=int)
    for item in placed_inputs:
        paste(inp, item["patch"], item["top"], item["left"], transparent=bg)

    if len(components_mask(inp != bg, connectivity=8)) != 4:
        return None

    combo_out, landed = _settle_replacements(inp.shape, bg, replacements)
    combo_start = _render_without_gravity(inp.shape, bg, replacements)

    inplace_replacements = _prepare_inplace_replacements(inp, bg)
    inplace_out, _ = _settle_replacements(inp.shape, bg, inplace_replacements)

    if np.array_equal(combo_out, combo_start):
        return None
    if np.array_equal(combo_out, inplace_out):
        return None
    if _count_moving_replacements(landed) < 2:
        return None
    if _count_supported_replacements(landed, bg, inp.shape) < 2:
        return None

    try:
        if np.array_equal(main(inp), combo_out):
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
