# Variant of c7f57c3e
# Dimension: D2 — 配对匹配策略 (match_cond)
# Change: 跨组配对时要求尺度相同（s.scale == d.scale）且位置重叠最大化... → 按洞的数量匹配

import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    get_bg,
    label_mask,
    paste,
    random_free_location,
)


# Design analysis for the D2 variant
# ----------------------------------
# a) 具体化与行为追踪
#    - 原题只有 2 个原型：每个精灵只要先识别出自己属于哪一组，再切换到另一组的同尺度原型即可。
#    - 现在把“跨组可替换对象”扩成 2 个洞数类 × 每类 2 个原型：
#      hole=1 时有一对跨组原型，hole=2 时再有一对跨组原型。每对内部仍靠 color-1 锚点做对齐。
#    - 小例子：若输入里有一个放大 3 倍的 hole=2 精灵，旧逻辑只会看“它是大精灵”和“与哪个原型重叠更像”；
#      新逻辑先数出它内部有 2 个背景洞，只允许在 hole=2 的那对跨组原型里匹配，再换成其中另一个。
#      差异点是：尺度只决定放大倍数，不再决定跨组配对身份。
# b) 复杂度校准
#    - 只使用两档洞数 {1, 2}，避免自由度过多，也避免人类无法从 2-4 个样例中归纳规则。
#    - 每个样例固定出现 4 个 scale=1 的“字典精灵”（四个原型各 1 个）和 1 个大精灵，
#      这样洞数配对规则有重复支撑，且不会退化成只看颜色或只看尺度。
# c) 隐式假设审查（相对原 main 的相关代码）
#    - 原代码假设最小面积精灵恰好给出 2 个原型；这里必须改成恰好 4 个原型。
#    - 原代码用“全体原型里哪个重叠得分最高”识别当前精灵；这里若不先按洞数筛候选，
#      会把 D2 变体又退化回旧题的形状/颜色匹配，因此改为“先数洞，再只在同洞数的 2 个候选里判别”。
#    - 原代码默认“替换目标 = 1 - best_idx”；这里不成立，必须按 hole_count 分组后取同组另一个原型。
# d) 放置合法性
#    - 生成器仍采用 output-first：先决定输出精灵，再放置与其配对的输入精灵。
#    - 输入/输出精灵使用同一个锚点对齐规则；放置时先计算两者锚点对齐后的并集 bbox，再一次性保留空间，
#      因此不会越界，也不会因交换后位姿不同而互相碰撞。
# e) 视觉可观察性
#    - 两个颜色组使用固定但彼此不同的两种变量色；hole=1 与 hole=2 通过内部空洞数量直接可见。
#    - 四个 scale=1 精灵本身就把“同洞数跨组互换”展示成一对一对应，大精灵则展示该规则对尺度不敏感。
# f) 可视化整洁度
#    - 所有原型共享 5x5 bbox、2x1 锚点、相同总体轮廓，内部只在空洞数上变化；视觉上整齐且易比较。
#    - 精灵间保持至少 2 格间距，避免锚点对齐导致替换后互相覆盖。
# g) 颜色变化可推断性
#    - 颜色变化只体现在“换到另一组的配色”，而是否换到哪一个目标由洞数决定；颜色本身不参与排序。
# 9) 新 main() 的额外假设与 generator 保证
#    - 假设最小面积精灵能完整给出 4 个 scale=1 原型：generator 固定放入四个基础精灵。
#    - 假设每个洞数类恰好有 2 个原型：generator 固定使用两档洞数，并对每档建立一对 flipUD 原型。
#    - 假设缩放不改变洞的连通个数：使用整数最近邻放大，4 连通洞数保持不变。


HOLE_CONNECTIVITY = 4


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


def _count_enclosed_holes(arr: np.ndarray, bg: int) -> int:
    """Count 4-connected bg regions fully enclosed inside the sprite bbox."""
    hole_mask = arr == bg
    lab, n = label_mask(hole_mask, connectivity=HOLE_CONNECTIVITY)
    count = 0
    h, w = arr.shape
    for k in range(1, n + 1):
        pts = np.argwhere(lab == k)
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 0] == h - 1) or np.any(pts[:, 1] == 0) or np.any(pts[:, 1] == w - 1):
            continue
        count += 1
    return int(count)


def _aligned_overlap_score(
    grid: np.ndarray,
    sprite: np.ndarray,
    proto: np.ndarray,
    bbox: tuple[int, int, int, int],
    anchor: tuple[int, int, int, int],
    bg: int,
    scale: int,
) -> int:
    """Score how well `proto` matches `sprite` when anchor-aligned at the inferred scale."""
    h, w = grid.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = anchor
    scaled = _scale_nn(proto, scale)
    pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
    pr0 *= scale
    pc0 *= scale
    top = (r0 + ar0) - pr0
    left = (c0 + ac0) - pc0

    ph, pw = scaled.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(h, top + ph)
    cc1 = min(w, left + pw)
    if rr0 >= rr1 or cc0 >= cc1:
        return -1

    sub_g = grid[rr0:rr1, cc0:cc1]
    sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
    m = sub_p != bg
    if not np.any(m):
        return -1
    score = int(np.sum(sub_g[m] == sub_p[m]))
    if sub_p.shape == sprite.shape and np.array_equal(sub_p, sprite):
        score += int(sprite.size * 10)
    return score


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype that has the same hole count.

    Input structure:
    - bg = most common color. Four scale-1 prototype sprites appear: two cross-group pairs, where each pair shares one enclosed-hole count.
    - Larger sprites are nearest-neighbor scaled copies of those prototypes, and every sprite contains color-1 anchor pixels plus bg holes.

    Rule:
    - Recover the four base prototypes from the smallest sprite instances.
    - Count the number of enclosed bg-hole components inside each sprite.
    - For each sprite, only compare against the two prototypes with the same hole count, identify which one it is, then replace it with the other prototype from that hole-count pair at the same scale.

    Key insight:
    - The decisive pairing key is the number of enclosed holes. Scale is still preserved for drawing, but it no longer decides which opposite-group prototype a sprite should swap to.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 5, "grid must contain the 4 base prototypes plus at least one extra sprite"

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
    assert len(uniq) == 4, "smallest-area sprites must yield exactly 4 base prototypes"

    proto_keys = list(uniq.keys())
    protos = [uniq[k] for k in proto_keys]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    a0r0, a0c0, a0r1, a0c1 = anc0
    base_anc_h = a0r1 - a0r0 + 1
    base_anc_w = a0c1 - a0c0 + 1
    for p in protos[1:]:
        pr0, pc0, pr1, pc1 = _pick_anchor_bbox(p, bg)
        assert pr1 - pr0 + 1 == base_anc_h and pc1 - pc0 + 1 == base_anc_w, "all prototypes must share the same anchor bbox size"

    hole_groups: dict[int, list[int]] = {}
    for idx, proto in enumerate(protos):
        holes = _count_enclosed_holes(proto, bg)
        hole_groups.setdefault(int(holes), []).append(int(idx))
    assert len(hole_groups) == 2 and all(len(v) == 2 for v in hole_groups.values()), "prototypes must form two hole-count pairs"

    out = g.copy()

    for s in sprites:
        r0, c0, _r1, _c1 = s["bbox"]
        arr = s["arr"]
        anc = _pick_anchor_bbox(arr, bg)
        ar0, ac0, ar1, ac1 = anc
        inst_anc_h = ar1 - ar0 + 1
        inst_anc_w = ac1 - ac0 + 1

        scale_h = max(1, inst_anc_h // max(1, base_anc_h))
        scale_w = max(1, inst_anc_w // max(1, base_anc_w))
        scale = int(max(scale_h, scale_w))

        sprite_holes = _count_enclosed_holes(arr, bg)
        candidate_idxs = hole_groups.get(int(sprite_holes))
        assert candidate_idxs is not None and len(candidate_idxs) == 2, "every sprite hole count must map to exactly one prototype pair"

        best_idx = int(candidate_idxs[0])
        best_score = -1
        for idx in candidate_idxs:
            score = _aligned_overlap_score(g, arr, protos[idx], s["bbox"], anc, bg, scale)
            if score > best_score:
                best_score = score
                best_idx = int(idx)

        repl_idx = int(candidate_idxs[0] if best_idx == candidate_idxs[1] else candidate_idxs[1])

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        proto = protos[repl_idx]
        scaled = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = scaled.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (D2 hole-pairing) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 24
GRID_SIZE_MAX = 34
GRID_MARGIN_FOR_SPRITES = 8

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
HOLE_COUNTS = [1, 2]

# --- CFG: sprite instances ---
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400

# Shared 5x5 structural templates. Labels: 0=bg hole, 1=anchor, 2/3=variable color regions.
_LABEL_TEMPLATES = {
    1: np.array(
        [
            [2, 1, 1, 3, 3],
            [2, 0, 2, 3, 3],
            [2, 2, 2, 3, 3],
            [2, 2, 2, 2, 3],
            [2, 2, 2, 2, 3],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [2, 1, 1, 3, 3],
            [2, 0, 2, 0, 3],
            [2, 2, 2, 3, 3],
            [2, 2, 2, 2, 3],
            [2, 2, 2, 2, 3],
        ],
        dtype=int,
    ),
}


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _colorize(labels: np.ndarray, bg: int, cols: tuple[int, int]) -> np.ndarray:
    proto = np.full(labels.shape, bg, dtype=int)
    proto[labels == 1] = 1
    proto[labels == 2] = cols[0]
    proto[labels == 3] = cols[1]
    return proto


def _make_prototypes(bg: int) -> tuple[list[np.ndarray], dict[int, int]]:
    pool = [c for c in range(10) if c not in (bg, 1)]
    cols = random.sample(pool, 4)
    a_cols = (cols[0], cols[1])
    b_cols = (cols[2], cols[3])

    protos: list[np.ndarray] = []
    pair_map: dict[int, int] = {}
    for hole_count in HOLE_COUNTS:
        labels = _LABEL_TEMPLATES[hole_count]
        idx_a = len(protos)
        protos.append(_colorize(labels, bg, a_cols))
        idx_b = len(protos)
        protos.append(_colorize(np.flipud(labels), bg, b_cols))
        pair_map[idx_a] = idx_b
        pair_map[idx_b] = idx_a

    assert len({tuple(p.ravel().tolist()) for p in protos}) == 4, "all base prototypes must be distinct"
    assert sorted(_count_enclosed_holes(p, bg) for p in protos) == [1, 1, 2, 2]
    return protos, pair_map


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h = max(patch_a.shape[0], patch_b.shape[0])
    w = max(patch_a.shape[1], patch_b.shape[1])
    return dr, dc, h + abs(dr), w + abs(dc)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos, pair_map = _make_prototypes(bg)

    instances = [(idx, 1) for idx in range(4)]
    instances.append((random.randrange(4), random.choice(BIG_SCALES)))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale in instances:
        output_patch = _scale(protos[out_type], scale)
        input_patch = _scale(protos[pair_map[out_type]], scale)
        dr, dc, uh, uw = _union_bbox(output_patch, input_patch)
        marker = np.full((uh, uw), 1, dtype=int)
        try:
            ur, uc = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        out_r = ur + max(0, -dr)
        out_c = uc + max(0, -dc)
        in_r, in_c = out_r + dr, out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D2-hole variant input")
