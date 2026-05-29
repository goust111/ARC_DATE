# Variant of c7f57c3e
# Dimension: D3 — 位置重叠约束 (match_cond)
# Change: 配对后最大化精灵的位置重叠（maximize position overlap），交换后精灵尽量保持原位
#         -> 完全位置交换：跨尺度的近邻异型精灵成对后，整只精灵连同其颜色模式一起互换位置。
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original: |
#       每个精灵单独切换到另一种 prototype，并用 color-1 锚点在原地对齐，所以空间上基本不动。
#     variant: |
#       先恢复局部的 A/B 近邻配对，再让配对双方交换锚点位置。
#       输出里搬动的是整只精灵，而不是“原地替换成另一种图样”。
#   concrete_visual_behavior:
#     input:
#       - 网格中恰有四个 sprite：A_small、A_big、B_small、B_big。
#       - 它们被排成两个局部小簇；每个簇里恰有一个 A 和一个 B，但两者尺度不同。
#       - 配对依据不是颜色 id，也不是隐藏标签，而是可见的空间近邻关系。
#     output:
#       - 每个局部 A/B 对内部做整只精灵的位置互换。
#       - 因为成对对象是跨尺度的，交换后每个落点上的“类型”和“尺寸”都会一起变化。
#       - 精灵自身的颜色模式不变；变化发生在“哪只精灵移动到哪里”。
#   behavior_trace:
#     example: |
#       输入若有一个簇是 A1 靠近 B3，另一个簇是 A3 靠近 B1，
#       原题会把四个位置原地改成相反 prototype，尺寸保持不变。
#       本变体会让 A1 与附近的 B3 互换位置，A3 与附近的 B1 互换位置；
#       因为每对内部大小不同，输出会出现“大图样跑到小图样原位、小图样跑到大图样原位”的现象。
#   complexity_calibration:
#     - 若让“同尺度 A/B”互换位置，像素级结果会和原题的“原地换型”完全一致，变体退化。
#     - 因此配对必须跨尺度；本实现固定为两个局部簇：A1<->B_big 与 A_big<->B1。
#     - 为避免配对歧义，簇内距离始终比跨簇距离小，solver 用最小总锚点距离即可恢复正确匹配。
#   hidden_assumption_audit:
#     - 原 solver 通过逐个 sprite 原地重绘，默认 sprite 之间互不依赖；本变体不成立，必须先完成全体分类，再整体清空，再重放到配对位置。
#     - 原 solver 按尺度独立处理即可；本变体不能按尺度分组，否则跨尺度配对会丢失。
#     - 原 generator 只保证当前类型能放下；本变体要求同一槽位在输入/输出两侧都可能承载一对跨尺度对象中的任意一个，
#       所以每个槽位必须预留“该对两只精灵”的 anchor-aligned 并集包围盒。
#   legality_and_cleanliness:
#     - 每个槽位用 pair-specific anchor-aligned union bbox 预留空间，保证这一对中的任一成员放进去都不越界、不撞到别的 sprite。
#     - 对象之间仍保留较大的 pair padding 和 border>=1，视觉上清楚分离。
#   observability:
#     - 两个局部簇重复出现，解题者可以直接看到“近邻异型对象整只互换位置”。
#     - 规则不依赖颜色 id 的大小、排序或任何不可见元信息。

import itertools
import random
from collections import defaultdict

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of color-1 pixels inside a sprite array."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels and should always have them"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _anchor_top_left_global(bbox: tuple[int, int, int, int], arr: np.ndarray, bg: int) -> tuple[int, int]:
    """Global anchor top-left location for a placed sprite."""
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)
    return int(r0 + ar0), int(c0 + ac0)


def _infer_scale(arr: np.ndarray, base_anchor_hw: tuple[int, int], bg: int) -> int:
    """Infer the integer scale from the anchor bbox dimensions."""
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


def _place_by_anchor(out: np.ndarray, arr: np.ndarray, anchor_pos: tuple[int, int], bg: int) -> None:
    """Place a sprite so that its anchor top-left lands on anchor_pos."""
    ar0, ac0, _ar1, _ac1 = _pick_anchor_bbox(arr, bg)
    top = int(anchor_pos[0] - ar0)
    left = int(anchor_pos[1] - ac0)
    paste(out, arr, top, left, transparent=bg)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap paired sprites by moving each whole sprite to its nearby opposite prototype's position.

    Input structure:
    - bg is the most common color. Two distinct multicolor sprite prototypes appear at two integer scales.
    - The board contains four sprites total: two proto_A instances and two proto_B instances, arranged as two local A/B pairs.

    Rule:
    - Recover the two base prototypes from the smallest sprites, then classify every placed sprite by prototype and scale.
    - Pair proto_A sprites with proto_B sprites by minimum total anchor-distance matching, then exchange the matched anchor locations.

    Key insight:
    - This variant no longer redraws sprites in place. The stable object identity is the whole observed sprite array, and the stable location descriptor is the color-1 anchor; solving the task means recovering which opposite-pattern sprites are local partners and then swapping those anchor positions.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
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
    base_anchor_hw0 = (anc0[2] - anc0[0] + 1, anc0[3] - anc0[1] + 1)
    base_anchor_hw1 = (anc1[2] - anc1[0] + 1, anc1[3] - anc1[1] + 1)
    assert base_anchor_hw0 == base_anchor_hw1, "prototypes must share the same anchor bbox size"

    by_type: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        scale = _infer_scale(sprite["arr"], base_anchor_hw0, bg)
        type_idx = _match_prototype(g, sprite, protos, scale, bg)
        sprite["scale"] = scale
        sprite["type_idx"] = type_idx
        sprite["anchor_pos"] = _anchor_top_left_global(sprite["bbox"], sprite["arr"], bg)
        by_type[type_idx].append(sprite)

    out = g.copy()
    for sprite in sprites:
        for rr, cc in sprite["pts"]:
            out[rr, cc] = bg

    group_a = sorted(by_type.get(0, []), key=lambda s: s["anchor_pos"])
    group_b = sorted(by_type.get(1, []), key=lambda s: s["anchor_pos"])
    assert len(group_a) == len(group_b) and len(group_a) > 0, "grid must contain balanced A/B sprites"
    for sa, sb in _best_opposite_pairing(group_a, group_b):
        _place_by_anchor(out, sa["arr"], sb["anchor_pos"], bg)
        _place_by_anchor(out, sb["arr"], sa["anchor_pos"], bg)

    return out


# --------------- generator (o5x variant) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

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
PAIR_GAPS = [1, 2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 4
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

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
    """Build an (h, w) label grid with roles 1 (anchor), 2, 3 and bg holes (0)."""
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
        entries = weights_by_role[role]
        kinds, wts = zip(*entries)
        chosen_kind = random.choices(kinds, weights=wts)[0]
        masks.append(_row_mask(chosen_kind, w))

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
    """Create proto_A and proto_B (flipud + recolored) from a shared label grid."""
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


def _anchor_union_layout(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    """Reserve box around a shared anchor top-left that can host either patch."""
    a0r, a0c, _a1r, _a1c = _anchor_bbox(patch_a)
    b0r, b0c, _b1r, _b1c = _anchor_bbox(patch_b)
    up = max(a0r, b0r)
    left = max(a0c, b0c)
    down = max(patch_a.shape[0] - a0r, patch_b.shape[0] - b0r)
    right = max(patch_a.shape[1] - a0c, patch_b.shape[1] - b0c)
    return up, left, up + down, left + right


def _paste_at_anchor(dst: np.ndarray, patch: np.ndarray, anchor_pos: tuple[int, int], bg: int) -> None:
    """Paste a patch so its anchor top-left sits at anchor_pos."""
    ar0, ac0, _ar1, _ac1 = _anchor_bbox(patch)
    paste(dst, patch, anchor_pos[0] - ar0, anchor_pos[1] - ac0, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid grid; returns None on placement failure or oracle mismatch."""
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    big_scale = random.choice(BIG_SCALES)
    pair_patches = [
        (_scale(proto_a, 1), _scale(proto_b, big_scale)),
        (_scale(proto_a, big_scale), _scale(proto_b, 1)),
    ]
    random.shuffle(pair_patches)

    max_dim = max(max(p.shape) for pair in pair_patches for p in pair)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for patch_a, patch_b in pair_patches:
        up, left, slot_h, slot_w = _anchor_union_layout(patch_a, patch_b)
        gap = random.choice(PAIR_GAPS)
        if random.random() < 0.5:
            offset = (0, slot_w + gap)
            pair_marker = np.full((slot_h, slot_w * 2 + gap), 1, dtype=int)
        else:
            offset = (slot_h + gap, 0)
            pair_marker = np.full((slot_h * 2 + gap, slot_w), 1, dtype=int)

        try:
            mr, mc = random_free_location(
                reserve,
                pair_marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, pair_marker, mr, mc)
        anchor1 = (mr + up, mc + left)
        anchor2 = (mr + offset[0] + up, mc + offset[1] + left)

        if random.random() < 0.5:
            input_order = [patch_a, patch_b]
        else:
            input_order = [patch_b, patch_a]
        output_order = input_order[::-1]

        for anchor_pos, patch in zip([anchor1, anchor2], input_order, strict=True):
            _paste_at_anchor(inp, patch, anchor_pos, bg)
        for anchor_pos, patch in zip([anchor1, anchor2], output_order, strict=True):
            _paste_at_anchor(out, patch, anchor_pos, bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e input")
