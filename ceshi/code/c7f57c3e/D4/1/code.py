# Variant of c7f57c3e
# Dimension: D4 — 源对象特征 (src)
# Change: 多色精灵（polychrome sprites），每个精灵使用不同的颜色组合构成可识别的颜色模式... → 含空心区域的框架精灵
#
# Design analysis
# ---------------
# variant_spec:
#   preserved_core:
#     - 仍然只有 2 个原型，网格中的每个对象都是其中之一的整数倍缩放副本。
#     - 仍然使用最小对象恢复两个原型，并把每个对象替换成“另一类原型”。
#     - 仍然按 color-1 锚区对齐，使替换前后尽可能重叠。
#   d4_change:
#     - 原任务对象是多色小精灵；这里改成“单连通的空心框架精灵”。
#     - 外边框的大部分像素使用 frame color，框架内部保留大量 bg 空洞。
#     - color-1 不再只是普通花纹，而是从框架边缘伸向内部的锚 spine。
#     - 内部再放一个 fill region，并让它通过锚 spine 接到框架，保证整只精灵仍是一个连通域。
#     - 两个原型交换时，frame color 与 fill color 成对一起交换。
#   concrete_example:
#     - 5x5 原型 A 可写成：
#         2 1 1 1 2
#         2 0 1 0 2
#         2 0 3 0 2
#         2 0 0 0 2
#         2 2 2 2 2
#       其中 2 是框架色，1 是锚 spine，3 是内部填充色，0 是背景孔洞。
#     - 原型 B 由同一角色模板上下翻转得到，因此锚 spine 落到下侧，fill 也随之翻转；
#       再给 B 重新分配一组 frame/fill 颜色。输入中的每个对象被替换为另一原型，但尺度不变。
#   behavior_trace:
#     - 原逻辑在某个对象上：通过 color-1 bbox 推出缩放倍数；再看它更像 proto_A 还是 proto_B；
#       然后放上另一类原型。
#     - 新逻辑完全沿用这个流程，但“更像哪一类”的依据不再是随意的多色花纹，而是
#       框架轮廓 + 锚 spine + 内部填充三者共同决定的空心框架样式。
#     - 差异在于：若仍使用原题那种“内部可分离的彩色像素”写法，连通域会被拆开；
#       因此生成器必须强制 fill 通过 spine 接回框架。
#   complexity_calibration:
#     - 仍只保留 2 种原型、1 个大尺度对象、3-4 个实例，避免规则分支过多。
#     - 使用 3 种小模板（5x5 / 5x6 / 6x5），既能明显看出“框架+空洞”，又不会过于自由导致匹配歧义。
#     - 颜色仍只围绕 anchor/frame/fill 三种角色展开，视觉规律可从 2-3 组样例归纳。
#   implicit_assumption_review:
#     - components_mask(g != bg) 假设每个 sprite 是单连通的；因此模板里 fill 必须接回框架。
#     - _pick_anchor_bbox 假设 color-1 必然存在；因此每个模板都固定保留 anchor spine。
#     - scale 推断依赖两个原型的锚 bbox 尺寸一致；因此 proto_B 只做 flipud，不改锚宽高。
#     - “最小 bbox 的对象给出两个原型”要求至少有 1 个 A/1 个 B 处于 scale=1；生成器显式保证。
#     - 放置替换对象时可能越过原 bbox；因此生成器仍用 anchor 对齐后的 union bbox 预留空间。
#   placement_legality:
#     - 先构造期望输出，再把每个对象换成另一原型得到输入；放置时按输入/输出 union bbox 占位。
#     - reserve 画布只记录 union bbox，因此对象之间不会重叠，也不会在替换后越界。
#   visual_observability:
#     - 规则不依赖颜色 id 排序，只依赖肉眼可见的角色：哪种框架样式、哪组 frame/fill 颜色。
#     - 上下翻转后的锚 spine 明显从框架上侧换到下侧，空心区域也保留，所以和原题的普通小精灵不同。
#   visual_cleanliness:
#     - 每个对象外轮廓完整、内部孔洞大、对象间距 >= 2，避免多个彩色区域糊在一起。
#   color_inference:
#     - 每个原型只有 frame color 与 fill color 两个变量色；人类可以从“边框色 + 内部填充色”成对切换
#       推断规则，不会出现大量无提示颜色映射。

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of the color-1 anchor spine (always present in this variant)."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "frame sprite has no color-1 anchor spine"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each hollow-frame sprite to the opposite frame/fill prototype without changing scale.

    Input structure:
    - bg is the most common color. Every non-bg component is a connected hollow-frame sprite with a visible internal cavity.
    - Each sprite is an integer-scaled copy of one of two prototype frames; the prototypes differ by their anchor-spine orientation and by the paired frame/fill colors.

    Rule:
    - Recover the two prototype frames from the smallest sprite instances.
    - For each sprite, infer its scale from the color-1 anchor spine and decide which prototype it matches best.
    - Replace it with the other prototype at the same scale, aligned by the anchor spine so the swapped frame lands in the same place.

    Key insight: the stable feature is the color-1 spine that links the frame edge to the interior fill; once that anchor box is located, the alternate frame/fill combination can be overlaid in the same position while preserving the hollow cavity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 frame sprites"

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
    assert len(uniq) == 2, "smallest sprites must yield exactly 2 distinct frame prototypes"

    proto_keys = list(uniq.keys())[:2]
    protos = [uniq[proto_keys[0]], uniq[proto_keys[1]]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    a0r0, a0c0, a0r1, a0c1 = anc0
    base_anc_h = a0r1 - a0r0 + 1
    base_anc_w = a0c1 - a0c0 + 1

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

        best_idx = 0
        best_score = -1
        for idx, p in enumerate(protos):
            sp = _scale_nn(p, scale)
            panc = _pick_anchor_bbox(p, bg)
            pr0, pc0, _pr1, _pc1 = panc
            pr0 *= scale
            pc0 *= scale
            top = (r0 + ar0) - pr0
            left = (c0 + ac0) - pc0

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

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        panc = _pick_anchor_bbox(p, bg)
        pr0, pc0, _pr1, _pc1 = panc
        pr0 *= scale
        pc0 *= scale
        top = (r0 + ar0) - pr0
        left = (c0 + ac0) - pc0

        ph, pw = sp.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        sub_p = sp[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        out_slice = out[rr0:rr1, cc0:cc1]
        out_slice[m] = sub_p[m]
        out[rr0:rr1, cc0:cc1] = out_slice

    return out


# --------------- generator (output-first hollow-frame swap) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 21
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLOR_SHARE_PROB = 0.45

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300

# Role templates:
#   0 = bg cavity
#   1 = anchor spine
#   2 = frame color
#   3 = interior fill color
FRAME_TEMPLATES = (
    np.array(
        [
            [2, 1, 1, 1, 2],
            [2, 0, 1, 0, 2],
            [2, 0, 3, 0, 2],
            [2, 0, 0, 0, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 1, 1, 1, 1, 2],
            [2, 0, 0, 1, 0, 2],
            [2, 0, 3, 3, 0, 2],
            [2, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [2, 1, 1, 1, 2],
            [2, 0, 1, 0, 2],
            [2, 0, 3, 3, 2],
            [2, 0, 0, 0, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=int,
    ),
)


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _colorize_frame(labels: np.ndarray, bg: int, frame_col: int, fill_col: int) -> np.ndarray:
    proto = np.full(labels.shape, bg, dtype=int)
    proto[labels == 1] = 1
    proto[labels == 2] = frame_col
    proto[labels == 3] = fill_col
    return proto


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = random.choice(FRAME_TEMPLATES)
    pool = [c for c in range(10) if c not in (bg, 1)]

    a_frame, a_fill = random.sample(pool, 2)
    remaining = [c for c in pool if c not in (a_frame, a_fill)]
    if random.random() < COLOR_SHARE_PROB:
        fresh = random.choice(remaining)
        if random.random() < 0.5:
            b_frame, b_fill = a_frame, fresh
        else:
            b_frame, b_fill = fresh, a_fill
    else:
        b_frame, b_fill = random.sample(remaining, 2)

    proto_a = _colorize_frame(labels, bg, a_frame, a_fill)
    proto_b = _colorize_frame(np.flipud(labels), bg, b_frame, b_fill)
    return proto_a, proto_b


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    oa = _anchor_bbox(patch_a)
    ia = _anchor_bbox(patch_b)
    dr = oa[0] - ia[0]
    dc = oa[1] - ia[1]
    h, w = patch_a.shape
    return dr, dc, h + abs(dr), w + abs(dc)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    count = random.choice(SPRITE_COUNTS)
    big_scale = random.choice(BIG_SCALES)
    instances = [(0, 1), (1, 1), (random.randrange(2), big_scale)]
    if count == 4:
        instances.append((random.randrange(2), 1))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    max_dim = max(max(protos[typ].shape) * scale for typ, scale in instances)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES)
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    out = np.full((n, n), bg, dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for out_type, scale in instances:
        output_patch = _scale(protos[out_type], scale)
        input_patch = _scale(protos[1 - out_type], scale)
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
        in_r = out_r + dr
        in_c = out_c + dc

        paste(reserve, marker, ur, uc)
        paste(out, output_patch, out_r, out_c, transparent=bg)
        paste(inp, input_patch, in_r, in_c, transparent=bg)

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
    raise RuntimeError("failed to generate a valid c7f57c3e D4 input")
