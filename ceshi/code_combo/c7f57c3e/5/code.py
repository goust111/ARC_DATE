# Combo variant of c7f57c3e (index 5)
# combo_spec: D6/1+D10/0
# Name: 颜色通道轮换 + 三组三向交换
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题里每个精灵只属于两类 prototype 之一，输出时整块切换到“另一类”。
#    - 这个融合版把“类”改成 3 个颜色相位组：同一结构模板上有 3 个变量通道角色，
#      三组分别是 (c0,c1,c2)、(c1,c2,c0)、(c2,c0,c1) 这三种相位。
#    - 同时保留原题的结构干扰项：同一相位既可能以 canonical 方向出现，也可能以上下翻转版出现；
#      所以输入里看到的是“3 个相位组 x 2 个朝向”的可见精灵集合，但真正分组只按颜色相位分成 3 组。
#    - 一个具体例子：canonical 小原型三条变量带自上而下是 teal/red/green，则三组分别是
#      A=(teal,red,green), B=(red,green,teal), C=(green,teal,red)。
#      若输入中有一个 flipped 的 C 组大精灵，原题会尝试换成另一几何族；这里不会改几何，
#      而是先把它规范化回 canonical，认出它是 C，再推进一相位到 A，最后翻回原朝向。
#    - 因此融合后的真实动作不是“任意重着色”也不是“二类互换”，而是
#      “三相位分组识别 + 沿相位环 A->B->C->A 推进一步”，这一步恰好等价于通道轮换
#      teal->red, red->green, green->teal。
# b) 复杂度校准
#    - 只保留 3 个变量颜色和 3 个相位，避免颜色自由度过高；每题固定出现三个 1x 相位样本，
#      让观察者总能看到完整三环，而不是靠单个大精灵猜规则。
#    - 朝向只保留 canonical / flipud 两档，既保留原题“结构不是完全同一张图”的味道，
#      又不引入过多几何自由度。
#    - 生成器固定为 4-5 个精灵、且只有一个放大对象，避免画面过散，也避免样本太少看不出三向交换。
# c) 隐式假设审查
#    - 原 `len(uniq) == 2` 失效：最小精灵里不再是 2 类，而是 canonical 后应恢复 3 个相位原型。
#    - 原 `repl_idx = 1 - best_idx` 失效：类别集合现在有方向，需要由可见的颜色三元组恢复有向环。
#    - 原 solver 直接按彩色数组匹配会把上下翻转版当成不同类；融合后必须先按 occupancy 做纵向规范化，
#      否则“翻转朝向”和“相位分组”会混在一起。
#    - 原贴回逻辑依赖 anchor 对齐。这里几何不变，所以不需要重新定位；但 scale 仍由 color-1 anchor bbox
#      推回，生成器也必须保证所有相位共用同尺寸 anchor。
# d) 放置合法性
#    - 输出优先生成：先决定输出精灵的相位、朝向、尺度，再把输入设成前一相位但同朝向同尺度。
#    - 因为输入输出几何完全一致，放置时只需为一个 bbox 预留空间；`random_free_location` 的 border/padding
#      约束就足以保证不重叠、不越界。
# e) 视觉可观察性
#    - 人类能直接看到三种重复出现的三色条带顺序，例如 (青,红,绿)、(红,绿,青)、(绿,青,红)。
#    - 输入到输出时，每个精灵都沿同一方向前进一步；同时画面里常混有上下翻转版，迫使解题者注意
#      “真正稳定的是规范化后的相位顺序，而不是某个绝对行号上的颜色”。
#    - 环顺序完全由像素中的上下角色顺序给出，不依赖颜色编号大小规律。
# f) 可视化整洁度
#    - 每个对象仍是单连通小精灵，带少量背景孔洞，对象间 padding>=2，避免视觉粘连。
#    - 全题只有 bg、anchor 色 1、以及 3 个循环色，没有额外标记层，输出变化集中在对象内部，画面干净。
# g) 变化的可推断性
#    - 三个循环色在每个样例中都会重复出现于多个对象，并且附着在固定的三条语义带上，
#      所以颜色轮换有足够结构线索。
#    - 生成器拒绝少于 3 个 canonical 相位的场景，避免退化成只看见局部重着色、看不出三组环关系。
# 9) 新 solver 的隐式假设审查
#    - `main()` 假设：最小面积组件经纵向规范化后恰好给出 3 个 distinct 相位原型；生成器固定放入
#      3 个 scale=1 基座相位并回验该条件。
#    - `main()` 假设：每个原型恰有 3 个非 bg 非 1 颜色，且它们在 canonical 方向上有稳定的上中下次序；
#      `_make_labels()` 用固定 role block 顺序保证这一点。
#    - `main()` 假设：occupancy 对 flipud 非对称，否则规范化方向不唯一；生成器明确排除纵向对称模板。
#    - `main()` 假设：所有实例都是这些 canonical 原型的整数倍缩放或其 flipud；生成器只产出这些情况。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve()
for _parent in ROOT.parents:
    solver_dir = _parent / "solvers"
    if (solver_dir / "arc_utils.py").exists():
        if str(solver_dir) not in sys.path:
            sys.path.append(str(solver_dir))
        break

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Return the bbox of color-1 pixels inside a sprite."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _occupancy_key(arr: np.ndarray, bg: int) -> tuple[int, ...]:
    return tuple((arr != bg).astype(int).ravel().tolist())


def _canonicalize_sprite(arr: np.ndarray, bg: int) -> tuple[np.ndarray, bool]:
    """Normalize away optional vertical flip using only the visible occupancy mask."""
    flipped = np.flipud(arr)
    key = _occupancy_key(arr, bg)
    fkey = _occupancy_key(flipped, bg)
    if fkey < key:
        return flipped.copy(), True
    return arr.copy(), False


def _phase_signature(arr: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Read the three rotating colors in canonical top-to-bottom order."""
    colors = [int(c) for c in np.unique(arr) if int(c) not in (bg, 1)]
    assert len(colors) == 3, "prototype must expose exactly three rotating colors"
    stats = []
    for color in colors:
        rs, _ = np.where(arr == color)
        assert rs.size > 0
        stats.append((int(rs.min()), float(rs.mean()), color))
    stats.sort()
    return int(stats[0][2]), int(stats[1][2]), int(stats[2][2])


def _recover_cycle_order(base_protos: list[np.ndarray], bg: int) -> list[np.ndarray]:
    """Recover the directed 3-cycle of canonical phase prototypes."""
    sigs = [_phase_signature(proto, bg) for proto in base_protos]
    sig_to_idx: dict[tuple[int, int, int], int] = {}
    for idx, sig in enumerate(sigs):
        assert sig not in sig_to_idx, "phase signatures must be distinct"
        sig_to_idx[sig] = idx
    assert len(sig_to_idx) == 3, "expected exactly three canonical phase signatures"

    next_idx: dict[int, int] = {}
    for idx, sig in enumerate(sigs):
        nxt_sig = (sig[1], sig[2], sig[0])
        assert nxt_sig in sig_to_idx, "phase signatures must form a 3-cycle by left rotation"
        next_idx[idx] = sig_to_idx[nxt_sig]

    order = [0]
    while len(order) < 3:
        nxt = next_idx[order[-1]]
        assert nxt not in order, "cycle must visit three distinct prototypes"
        order.append(nxt)
    assert next_idx[order[-1]] == order[0], "cycle must close after three steps"
    return [base_protos[i] for i in order]


def _classify_phase(arr: np.ndarray, protos: list[np.ndarray], scale: int) -> int:
    """Return which canonical phase prototype matches this canonicalized sprite."""
    exact = []
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        if sp.shape == arr.shape and np.array_equal(sp, arr):
            exact.append(idx)
    if len(exact) == 1:
        return exact[0]

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        sp = _scale_nn(proto, scale)
        if sp.shape != arr.shape:
            continue
        score = int(np.sum(arr == sp))
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance every sprite one step around a 3-phase color cycle while preserving its scale and visible orientation.

    Input structure:
    - The background is the most common color. The grid contains 4-5 disconnected sprites built from one small shape template or its vertical flip, at integer scales.
    - Every sprite contains a fixed anchor color 1 plus exactly three rotating colors. After canonicalizing away the optional vertical flip, the smallest sprites reveal three distinct color-phase groups.

    Rule:
    - Extract the smallest sprites, canonicalize them by occupancy-mask vertical normalization, and recover the directed phase ring from their top-to-bottom color triples.
    - For each sprite, infer its integer scale from the color-1 anchor bbox, canonicalize its orientation, classify which phase prototype it matches, and replace it with the next phase A->B->C->A.
    - Keep the sprite in the same bbox and restore its original visible orientation after the phase advance.

    Key insight:
    - The three groups are not arbitrary palettes: they are the three phase states of one shared channel rotation. A sprite may be flipped, but that flip is only a nuisance variable; once the shape is normalized, the output is just the next cyclic permutation of the same three semantic color bands.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 3, "grid must contain at least 3 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        carr, was_flipped = _canonicalize_sprite(arr, bg)
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
                "canonical": carr,
                "was_flipped": was_flipped,
                "anchor_bbox": _pick_anchor_bbox(arr, bg),
            }
        )

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]

    canonical: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["canonical"].ravel().tolist())
        canonical[key] = s["canonical"]
    assert len(canonical) == 3, "smallest-area sprites must expose exactly 3 canonical phase prototypes"

    protos = _recover_cycle_order(list(canonical.values()), bg)

    base_anchor = _pick_anchor_bbox(protos[0], bg)
    base_anc_h = base_anchor[2] - base_anchor[0] + 1
    base_anc_w = base_anchor[3] - base_anchor[1] + 1

    out = g.copy()

    for s in sprites:
        anc = s["anchor_bbox"]
        inst_anc_h = anc[2] - anc[0] + 1
        inst_anc_w = anc[3] - anc[1] + 1
        assert inst_anc_h % base_anc_h == 0 and inst_anc_w % base_anc_w == 0
        scale_h = inst_anc_h // base_anc_h
        scale_w = inst_anc_w // base_anc_w
        assert scale_h == scale_w and scale_h >= 1
        scale = int(scale_h)

        phase_idx = _classify_phase(s["canonical"], protos, scale)
        repl = _scale_nn(protos[(phase_idx + 1) % 3], scale)
        if s["was_flipped"]:
            repl = np.flipud(repl)
        assert repl.shape == s["arr"].shape

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        r0, c0, r1, c1 = s["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = repl != bg
        patch[mask] = repl[mask]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
PROTO_SHAPES = [(4, 3), (4, 4), (5, 4)]
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
ROLE_ORDER = [2, 1, 3, 4]

# --- CFG: row-mask weights ---
ANCHOR_ROW_WEIGHTS = [("center", 0.45), ("outer", 0.35), ("full", 0.20)]
VARIABLE_ROW_WEIGHTS = [("center", 0.68), ("full", 0.32)]
OUTER_FIXUP_PROB = 0.7

# --- CFG: sprite instances ---
SPRITE_COUNTS = [4, 5]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400
MAX_LABEL_RETRIES = 200

_ROW_MASKS = {
    "center": {3: [0, 1, 0], 4: [0, 1, 1, 0]},
    "outer": {3: [1, 0, 1], 4: [1, 0, 0, 1]},
    "full": {3: [1, 1, 1], 4: [1, 1, 1, 1]},
}


def _row_mask(kind: str, w: int) -> list[int]:
    return list(_ROW_MASKS[kind][w])


def _make_labels() -> np.ndarray:
    """Build one canonical label grid with vertically ordered roles 2,1,3,4."""
    for _ in range(MAX_LABEL_RETRIES):
        h, w = random.choice(PROTO_SHAPES)
        counts = {1: 1, 2: 1, 3: 1, 4: 1}
        if h > 4:
            counts[random.choice(ROLE_ORDER)] += h - 4

        weights_by_role = {
            1: ANCHOR_ROW_WEIGHTS,
            2: VARIABLE_ROW_WEIGHTS,
            3: VARIABLE_ROW_WEIGHTS,
            4: VARIABLE_ROW_WEIGHTS,
        }
        roles: list[int] = []
        masks: list[list[int]] = []
        for role in ROLE_ORDER:
            kinds, wts = zip(*weights_by_role[role])
            for _ in range(counts[role]):
                roles.append(role)
                masks.append(_row_mask(random.choices(kinds, weights=wts)[0], w))

        if not any(mask[0] or mask[-1] for mask in masks):
            idx = random.randrange(h)
            if roles[idx] == 1 and random.random() < OUTER_FIXUP_PROB:
                masks[idx] = _row_mask("outer", w)
            else:
                masks[idx] = _row_mask("full", w)

        if all(all(mask) for mask in masks):
            candidates = [i for i, role in enumerate(roles) if role != 1]
            masks[random.choice(candidates)] = _row_mask("center", w)

        labels = np.zeros((h, w), dtype=int)
        for r, (role, mask) in enumerate(zip(roles, masks, strict=False)):
            for c, bit in enumerate(mask):
                if bit:
                    labels[r, c] = role

        occ = labels != 0
        if np.array_equal(occ, np.flipud(occ)):
            continue
        if len(components_mask(occ, connectivity=8)) != 1:
            continue
        if not all(np.any(labels == role) for role in ROLE_ORDER):
            continue
        return labels

    raise RuntimeError("failed to build a connected non-symmetric label grid")


def _colorize_phase(labels: np.ndarray, bg: int, cycle_colors: list[int], phase: int) -> np.ndarray:
    """Colorize the canonical label grid as one phase of the 3-cycle."""
    proto = np.full(labels.shape, bg, dtype=int)
    proto[labels == 1] = 1
    proto[labels == 2] = cycle_colors[phase % 3]
    proto[labels == 3] = cycle_colors[(phase + 1) % 3]
    proto[labels == 4] = cycle_colors[(phase + 2) % 3]
    return proto


def _make_prototypes(bg: int) -> list[np.ndarray]:
    labels = _make_labels()
    pool = [c for c in range(10) if c not in (bg, 1)]
    cycle_colors = random.sample(pool, 3)
    protos = [_colorize_phase(labels, bg, cycle_colors, phase) for phase in range(3)]
    sigs = {_phase_signature(proto, bg) for proto in protos}
    assert len(sigs) == 3
    return protos


def _build_instances() -> list[tuple[int, bool, int]]:
    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, bool(random.getrandbits(1)), 1), (1, bool(random.getrandbits(1)), 1), (2, bool(random.getrandbits(1)), 1)]
    instances.append((random.randrange(3), bool(random.getrandbits(1)), big_scale))
    if count == 5:
        instances.append((random.randrange(3), bool(random.getrandbits(1)), 1))

    if len({flipped for _phase, flipped, _scale in instances}) == 1:
        phase, _flipped, scale = instances[0]
        instances[0] = (phase, not instances[0][1], scale)

    random.shuffle(instances)
    instances.sort(key=lambda item: item[2], reverse=True)
    return instances


def _visible_patch(proto: np.ndarray, flipped: bool, scale: int) -> np.ndarray:
    patch = _scale_nn(proto, scale)
    if flipped:
        patch = np.flipud(patch)
    return patch


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    protos = _make_prototypes(bg)
    instances = _build_instances()

    max_dim = max(max(protos[phase].shape) * scale for phase, _flipped, scale in instances)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_phase, flipped, scale in instances:
        output_patch = _visible_patch(protos[out_phase], flipped, scale)
        input_patch = _visible_patch(protos[(out_phase - 1) % 3], flipped, scale)
        marker = np.full(output_patch.shape, 1, dtype=int)
        try:
            r, c = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, r, c)
        paste(out, output_patch, r, c, transparent=bg)
        paste(inp, input_patch, r, c, transparent=bg)

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
