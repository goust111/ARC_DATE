# Combo variant of c7f57c3e (index 23)
# combo_spec: D3/1+D5/2+D14/1
# Name: 完全位置交换 + 重力下落 + 路径可视化

from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# ===============
# fused_variant:
#   concrete_behavior:
#     - 输入固定为 4 个 sprite，分别是 A_small、A_big、B_small、B_big。
#     - 它们被组织成上下两条水平走廊；每条走廊里恰有一个 A 和一个 B，且二者尺度不同。
#     - 两条走廊共用同一对左右落柱，所以“先交换位置，再下落”后，每一列都会形成 2 个 sprite 的堆叠。
#     - 输出步骤是：
#       1. 先按局部近邻把 A 与 B 配对；
#       2. 每对内部把整只 sprite 交换到对方原来的锚点位置；
#       3. 所有已交换 sprite 一起做竖直重力结算；
#       4. 在每对原始位置之间补上一条双色水平路径，路径留在高处，而 sprite 已下落到底部附近。
#   trace_example:
#     - 例如顶层走廊里是 A1 在左、B3 在右，底层走廊里是 B1 在左、A3 在右。
#     - 原题会让四个位置各自“原地换型”；本融合变体则让 A1/B3 互换位置、B1/A3 互换位置，
#       然后两列对象继续向下掉落，最后只剩两条悬空路径标记原始交换关系。
#     - 差异不是单一步骤，而是“邻近配对 -> 整体换位 -> 全局重力 -> 残留路径”的链式组合。
#   complexity_calibration:
#     - 若允许任意数量 sprite，D3 的配对与 D14 的路径都容易歧义；因此收紧为恰好 4 个对象、2 对局部簇。
#     - 若配对双方同尺度，则“位置交换”在视觉上容易退化为原地换型的近似效果；因此每对都强制跨尺度。
#     - 若每对使用独立落柱，则 D5 只表现为各自落到底，堆叠感不明显；因此两对共用左右两列，让重力可见。
#   hidden_assumptions_review:
#     - 原 main 假设可逐个 sprite 原地替换；融合后失效，因为位置交换与重力都依赖全局对象集，必须先整体识别、再统一结算。
#     - 原 D14 参考实现依赖“配对对象之间存在空走廊”；这里生成器显式保证每对是水平分离且中间留空。
#     - 新 solver 额外依赖：
#       1. 最小 bbox 面积的两个 sprite 恰好是 scale=1 的 A/B 原型；
#       2. A/B 数量平衡，可做最小总距离匹配；
#       3. 每对原始锚点同一行，方便画水平路径；
#       4. 交换后的起始位置彼此不重叠，且重力落点合法。
#     - 这些前提都由 generator 构造保证，并在生成阶段做 oracle 校验。
#   placement_legality_and_observability:
#     - 输入阶段检查 4 个 bbox 彼此断开、全部在边界内，且每对之间有正走廊宽度。
#     - 输出阶段检查：至少 2 个 sprite 实际下落；至少 1 列形成支撑堆叠；路径不会被最终落稳 sprite 覆盖。
#     - 路径颜色来自各端 sprite 中唯一贴着 color-1 锚区的“传输色”，不依赖颜色 id 大小或排序。
#   cleanliness:
#     - 路径只画在原始水平走廊里，最终 sprite 全部在更低处，因此“路径层”和“落稳对象层”视觉上分离。
#     - 只使用两条路径、两列落柱，避免多重叠加导致画面混乱。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 pixels; all valid sprites contain them."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _anchor_top_left_global(bbox: tuple[int, int, int, int], arr: np.ndarray, bg: int) -> tuple[int, int]:
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)
    return int(r0 + ar0), int(c0 + ac0)


def _infer_scale(arr: np.ndarray, base_anchor_hw: tuple[int, int], bg: int) -> tuple[int, tuple[int, int, int, int]]:
    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, ar1, ac1 = anc
    inst_h = int(ar1 - ar0 + 1)
    inst_w = int(ac1 - ac0 + 1)
    base_h, base_w = base_anchor_hw
    assert inst_h % base_h == 0 and inst_w % base_w == 0, "anchor bbox must scale integrally"
    scale_h = inst_h // base_h
    scale_w = inst_w // base_w
    assert scale_h == scale_w and scale_h >= 1, "sprite scale must be a positive integer"
    return int(scale_h), anc


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
        top = int((r0 + ar0) - pr0)
        left = int((c0 + ac0) - pc0)

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
    return int(best_idx)


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


def _can_place(board: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = board.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    sub = board[top : top + ph, left : left + pw]
    return not bool(((patch != bg) & (sub != bg)).any())


def _anchor_bbox_global(
    bbox: tuple[int, int, int, int],
    arr: np.ndarray,
    bg: int,
) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, ar1, ac1 = _pick_anchor_bbox(arr, bg)
    return int(r0 + ar0), int(c0 + ac0), int(r0 + ar1), int(c0 + ac1)


def _swap_items_from_pairs(pairs: list[tuple[dict, dict]], bg: int) -> tuple[list[dict], list[dict]]:
    """Build gravity items and trail metadata from opposite-type pairs."""
    items = []
    trails = []
    for sa, sb in pairs:
        start_anchor_a = sb["anchor_pos"]
        start_anchor_b = sa["anchor_pos"]
        a_ar0, a_ac0, _a_ar1, _a_ac1 = _pick_anchor_bbox(sa["arr"], bg)
        b_ar0, b_ac0, _b_ar1, _b_ac1 = _pick_anchor_bbox(sb["arr"], bg)
        items.append(
            {
                "patch": sa["arr"],
                "start_top": int(start_anchor_a[0] - a_ar0),
                "start_left": int(start_anchor_a[1] - a_ac0),
            }
        )
        items.append(
            {
                "patch": sb["arr"],
                "start_top": int(start_anchor_b[0] - b_ar0),
                "start_left": int(start_anchor_b[1] - b_ac0),
            }
        )

        trails.append(
            {
                "a": sa,
                "b": sb,
                "color_a": _transfer_color(sa["arr"], bg),
                "color_b": _transfer_color(sb["arr"], bg),
                "anchor_box_a": _anchor_bbox_global(sa["bbox"], sa["arr"], bg),
                "anchor_box_b": _anchor_bbox_global(sb["bbox"], sb["arr"], bg),
            }
        )
    return items, trails


def _settle_items(shape: tuple[int, int], bg: int, items: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Drop swapped sprites straight down until they collide or hit the floor."""
    h, _w = shape
    board = np.full(shape, bg, dtype=int)
    landed = []

    order = sorted(
        items,
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
        assert _can_place(board, patch, top, left, bg), "swapped sprites must start non-overlapping"

        while top + ph < h and _can_place(board, patch, top + 1, left, bg):
            top += 1

        paste(board, patch, top, left, transparent=bg)
        landed.append(
            {
                "start_top": int(item["start_top"]),
                "start_left": int(item["start_left"]),
                "final_top": int(top),
                "final_left": int(left),
                "patch": patch,
            }
        )
    return board, landed


def _draw_horizontal_trail(out: np.ndarray, trail: dict) -> None:
    """Draw a split-color horizontal trail between the pair's original positions."""
    a = trail["a"]
    b = trail["b"]
    if a["bbox"][1] <= b["bbox"][1]:
        left, right = a, b
        left_color = int(trail["color_a"])
        right_color = int(trail["color_b"])
        left_anchor_box = trail["anchor_box_a"]
        right_anchor_box = trail["anchor_box_b"]
    else:
        left, right = b, a
        left_color = int(trail["color_b"])
        right_color = int(trail["color_a"])
        left_anchor_box = trail["anchor_box_b"]
        right_anchor_box = trail["anchor_box_a"]

    assert left_anchor_box[0] == right_anchor_box[0], "paired anchors must share a row"
    assert left["bbox"][3] < right["bbox"][1], "paired sprites must leave a horizontal corridor"

    band_r0 = int(left_anchor_box[0])
    band_r1 = int(band_r0 + min(
        left_anchor_box[2] - left_anchor_box[0],
        right_anchor_box[2] - right_anchor_box[0],
    ))
    c0 = int(left["bbox"][3] + 1)
    c1 = int(right["bbox"][1] - 1)
    assert c0 <= c1, "trail corridor must have positive width"
    split = (c0 + c1) // 2
    out[band_r0 : band_r1 + 1, c0 : split + 1] = left_color
    if split + 1 <= c1:
        out[band_r0 : band_r1 + 1, split + 1 : c1 + 1] = right_color


def _extract_sprites(g: np.ndarray, bg: int) -> tuple[list[dict], list[np.ndarray], tuple[int, int]]:
    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))

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

    protos = [uniq[key] for key in list(uniq.keys())[:2]]
    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    base_anchor_hw0 = (anc0[2] - anc0[0] + 1, anc0[3] - anc0[1] + 1)
    base_anchor_hw1 = (anc1[2] - anc1[0] + 1, anc1[3] - anc1[1] + 1)
    assert base_anchor_hw0 == base_anchor_hw1, "prototypes must share the same anchor bbox size"
    return sprites, protos, base_anchor_hw0


def _classify_sprites(g: np.ndarray, bg: int) -> tuple[list[dict], list[tuple[dict, dict]]]:
    sprites, protos, base_anchor_hw = _extract_sprites(g, bg)
    by_type: dict[int, list[dict]] = defaultdict(list)

    for sprite in sprites:
        scale, anc = _infer_scale(sprite["arr"], base_anchor_hw, bg)
        type_idx = _match_prototype(g, sprite, protos, scale, bg)
        sprite["scale"] = int(scale)
        sprite["type_idx"] = int(type_idx)
        sprite["anchor_pos"] = _anchor_top_left_global(sprite["bbox"], sprite["arr"], bg)
        sprite["anchor_bbox"] = anc
        by_type[type_idx].append(sprite)

    group_a = sorted(by_type.get(0, []), key=lambda s: s["anchor_pos"])
    group_b = sorted(by_type.get(1, []), key=lambda s: s["anchor_pos"])
    assert len(group_a) == len(group_b) and len(group_a) > 0, "grid must contain balanced A/B sprites"
    pairs = _best_opposite_pairing(group_a, group_b)
    return sprites, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange whole paired sprites, let the exchanged sprites fall, then draw the swap paths they left behind.

    Input structure:
    - bg is the most common color. Exactly two multicolor prototype families appear, each as nearest-neighbor scaled copies.
    - The board contains balanced A/B sprites, and every valid instance has a color-1 anchor region used for scale recovery and geometric alignment.

    Rule:
    - Recover the two prototypes from the smallest sprites, classify every instance by prototype and scale, and pair A sprites with nearby B sprites by minimum total anchor distance.
    - For each pair, move the entire observed sprite to the partner's anchor position rather than recoloring it in place.
    - After all position exchanges are defined, drop every moved sprite straight down until it hits the floor or a previously settled sprite, then draw a split-color trail across the pair's original corridor.

    Key insight:
    - The stable combinational object is not a recolored patch but the whole observed sprite plus its anchor. The anchor recovers prototype identity for pairing, defines the exchanged destination, and also pins the original horizontal corridor where the lingering trail must be drawn after gravity has pulled the moved sprites away.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    _sprites, pairs = _classify_sprites(g, bg)
    items, trails = _swap_items_from_pairs(pairs, bg)
    out, _landed = _settle_items(g.shape, bg, items)
    for trail in trails:
        _draw_horizontal_trail(out, trail)
    return out


# --------------- generator ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 28

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
]

# --- CFG: instances / layout ---
BIG_SCALES = [2]
BORDER = 1
INPUT_PADDING = 2
MIN_FALL_DISTANCE = 4

# --- CFG: generation retries ---
MAX_RETRIES = 500


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


def _place_from_anchor(patch: np.ndarray, anchor_pos: tuple[int, int]) -> tuple[int, int, int, int]:
    ar0, ac0, _ar1, _ac1 = _anchor_bbox(patch)
    top = int(anchor_pos[0] - ar0)
    left = int(anchor_pos[1] - ac0)
    return top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1


def _in_bounds_bbox(bbox: tuple[int, int, int, int], n: int) -> bool:
    r0, c0, r1, c1 = bbox
    return BORDER <= r0 and BORDER <= c0 and r1 < n - BORDER and c1 < n - BORDER


def _render_input(shape: tuple[int, int], bg: int, placed: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for item in placed:
        paste(out, item["patch"], item["top"], item["left"], transparent=bg)
    return out


def _render_swapped_without_gravity(shape: tuple[int, int], bg: int, items: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for item in items:
        assert _can_place(out, item["patch"], int(item["start_top"]), int(item["start_left"]), bg)
        paste(out, item["patch"], int(item["start_top"]), int(item["start_left"]), transparent=bg)
    return out


def _count_moving_replacements(landed: list[dict]) -> int:
    return sum(int(item["final_top"] > item["start_top"]) for item in landed)


def _count_supported_replacements(landed: list[dict], bg: int, shape: tuple[int, int]) -> int:
    h, w = shape
    board = np.full(shape, bg, dtype=int)
    supported = 0
    for item in sorted(landed, key=lambda d: int(d["final_top"]), reverse=True):
        patch = item["patch"]
        top = int(item["final_top"])
        left = int(item["final_left"])
        ph, pw = patch.shape
        if top + ph < h:
            below = board[top + ph, left : left + pw]
            if np.any((patch[-1] != bg) & (below != bg)):
                supported += 1
        paste(board, patch, top, left, transparent=bg)
    return supported


def _trail_has_clear_final_corridor(landed_board: np.ndarray, trail: dict, bg: int) -> bool:
    a = trail["a"]
    b = trail["b"]
    if a["bbox"][1] <= b["bbox"][1]:
        left, right = a, b
        left_anchor_box = trail["anchor_box_a"]
        right_anchor_box = trail["anchor_box_b"]
    else:
        left, right = b, a
        left_anchor_box = trail["anchor_box_b"]
        right_anchor_box = trail["anchor_box_a"]
    if left_anchor_box[0] != right_anchor_box[0]:
        return False
    if left["bbox"][3] >= right["bbox"][1]:
        return False
    band_r0 = int(left_anchor_box[0])
    band_r1 = int(band_r0 + min(
        left_anchor_box[2] - left_anchor_box[0],
        right_anchor_box[2] - right_anchor_box[0],
    ))
    c0 = int(left["bbox"][3] + 1)
    c1 = int(right["bbox"][1] - 1)
    if c0 > c1:
        return False
    return bool(np.all(landed_board[band_r0 : band_r1 + 1, c0 : c1 + 1] == bg))


def _pair_layout_records(
    protos: list[np.ndarray],
    big_scale: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    pair_defs = [
        ((_scale(protos[0], 1), 0, 1), (_scale(protos[1], big_scale), 1, big_scale)),
        ((_scale(protos[1], 1), 1, 1), (_scale(protos[0], big_scale), 0, big_scale)),
    ]
    records = []
    for left_def, right_def in pair_defs:
        if random.random() < 0.5:
            left_def, right_def = right_def, left_def
        records.append((left_def[0], right_def[0]))
    if random.random() < 0.5:
        records.reverse()
    return records


def _pair_vertical_extents(pair: tuple[np.ndarray, np.ndarray]) -> tuple[int, int]:
    left_patch, right_patch = pair
    left_anc = _anchor_bbox(left_patch)
    right_anc = _anchor_bbox(right_patch)
    up = max(int(left_anc[0]), int(right_anc[0]))
    down = max(int(left_patch.shape[0] - left_anc[0]), int(right_patch.shape[0] - right_anc[0]))
    return up, down


def _pair_required_anchor_delta(pair: tuple[np.ndarray, np.ndarray], corridor: int) -> int:
    left_patch, right_patch = pair
    left_anc = _anchor_bbox(left_patch)
    right_anc = _anchor_bbox(right_patch)
    return int(left_patch.shape[1] - left_anc[1] + right_anc[1] + corridor)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos = list(_make_prototypes(bg))
    big_scale = random.choice(BIG_SCALES)
    pair_patches = _pair_layout_records(protos, big_scale)

    all_patches = [patch for pair in pair_patches for patch in pair]
    max_dim = max(max(patch.shape) for patch in all_patches)
    max_anchor_up = max(_anchor_bbox(patch)[0] for patch in all_patches)
    max_anchor_left = max(_anchor_bbox(patch)[1] for patch in all_patches)
    max_anchor_right = max(patch.shape[1] - _anchor_bbox(patch)[1] for patch in all_patches)
    pair_stack_left = sum(pair[1].shape[0] for pair in pair_patches)
    pair_stack_right = sum(pair[0].shape[0] for pair in pair_patches)
    _top_up, top_down = _pair_vertical_extents(pair_patches[0])
    bot_up, bot_down = _pair_vertical_extents(pair_patches[1])
    required_row_delta = top_down + INPUT_PADDING + bot_up + random.randint(1, 3)
    required_col_delta = max(_pair_required_anchor_delta(pair, corridor=4) for pair in pair_patches)

    min_n = max(
        GRID_SIZE_MIN,
        max_dim + 12,
        max(pair_stack_left, pair_stack_right) + 12,
        BORDER + max_anchor_left + required_col_delta + max_anchor_right + 1,
        2 * BORDER + max_anchor_up + required_row_delta + bot_down + 1,
    )
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    extra_col_slack = max(0, n - (BORDER + max_anchor_left + required_col_delta + max_anchor_right + 1))
    left_anchor_col = BORDER + max_anchor_left + random.randint(0, max(0, extra_col_slack // 3))
    right_anchor_col = left_anchor_col + required_col_delta + random.randint(0, min(2, extra_col_slack))
    if right_anchor_col > n - BORDER - max_anchor_right:
        return None

    min_top_anchor = BORDER + max_anchor_up
    max_top_anchor = min(min_top_anchor + 3, n - BORDER - required_row_delta - bot_down)
    if min_top_anchor > max_top_anchor:
        return None
    top_anchor_row = random.randint(min_top_anchor, max_top_anchor)
    bottom_anchor_row = top_anchor_row + required_row_delta

    pair_anchors = [
        ((top_anchor_row, left_anchor_col), (top_anchor_row, right_anchor_col)),
        ((bottom_anchor_row, left_anchor_col), (bottom_anchor_row, right_anchor_col)),
    ]

    placed_inputs = []
    bboxes = []
    for pair_idx, ((left_patch, right_patch), (left_anchor, right_anchor)) in enumerate(
        zip(pair_patches, pair_anchors, strict=True)
    ):
        for side, patch, anchor in (("L", left_patch, left_anchor), ("R", right_patch, right_anchor)):
            top, left, r1, c1 = _place_from_anchor(patch, anchor)
            bbox = (top, left, r1, c1)
            if not _in_bounds_bbox(bbox, n):
                return None
            placed_inputs.append(
                {
                    "pair_idx": pair_idx,
                    "side": side,
                    "patch": patch,
                    "top": int(top),
                    "left": int(left),
                    "bbox": bbox,
                }
            )
            bboxes.append(bbox)

    if not _bboxes_separated(bboxes, INPUT_PADDING):
        return None

    for pair_idx in range(2):
        left_box = [item for item in placed_inputs if item["pair_idx"] == pair_idx and item["side"] == "L"][0]["bbox"]
        right_box = [item for item in placed_inputs if item["pair_idx"] == pair_idx and item["side"] == "R"][0]["bbox"]
        if left_box[3] >= right_box[1]:
            return None

    inp = _render_input((n, n), bg, placed_inputs)
    if len(components_mask(inp != bg, connectivity=8)) != 4:
        return None

    try:
        _sprites, pairs = _classify_sprites(inp, bg)
        items, trails = _swap_items_from_pairs(pairs, bg)
        swapped_only = _render_swapped_without_gravity(inp.shape, bg, items)
        landed_board, landed = _settle_items(inp.shape, bg, items)
        if _count_moving_replacements(landed) < 2:
            return None
        if _count_supported_replacements(landed, bg, inp.shape) < 1:
            return None
        if np.array_equal(swapped_only, landed_board):
            return None
        if not all(_trail_has_clear_final_corridor(landed_board, trail, bg) for trail in trails):
            return None

        expected_out = landed_board.copy()
        for trail in trails:
            _draw_horizontal_trail(expected_out, trail)
        if np.array_equal(expected_out, swapped_only):
            return None
        if np.array_equal(expected_out, inp):
            return None
        if max(int(item["final_top"]) - int(item["start_top"]) for item in landed) < MIN_FALL_DISTANCE:
            return None
        if not np.array_equal(main(inp), expected_out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
