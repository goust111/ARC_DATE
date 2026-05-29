# Variant of 247ef758
# Dimension: D4 — 变换：印刻与擦除行为 (transform)
# Change: 擦除左侧原始精灵，在目标位置居中印刻精灵图案，像素超出框架内部时裁剪（clipping）... → 印刻时旋转对齐

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D4 variant
# ----------------------------------
# variant_spec:
#   original_backbone:
#     - 输入仍然是 "左侧精灵条带 | 单色分隔列 | 右侧矩形框"。
#     - mover 颜色仍通过 "左右边同色行标 + 上下边同色列标" 得到目标点笛卡尔积。
#     - mover 仍从左侧擦除，distractor 仍保持不动，覆盖顺序仍是下方精灵先画、上方精灵后画。
#   concrete_change:
#     - 原题在每个目标点只做平移复制。
#     - 本变体在每个目标点先判断四个参与边标中哪一侧最近：
#       top -> 原朝向(k=0), left -> 逆时针 90°(k=1), bottom -> 180°(k=2), right -> 270°(k=3)。
#     - 最近侧比较的是目标点到四个同色边标的曼哈顿距离；由于这些边标与目标点同行或同列，
#       实际就是比较到 top/left/bottom/right 四条框边的距离。生成器会避免平局。
#   behavior_trace:
#     - 例：某 mover 有 row_marks=[2, 8], col_marks=[7]。
#     - 原题：两个目标 (2,7),(8,7) 都印刻同一朝向的精灵。
#     - 变体：若 (2,7) 更靠近上边，则用 0°；若 (8,7) 更靠近下边，则用 180°。
#       同一输入精灵会在两个目标点出现不同朝向，这就是 D4 的核心可见差异。
#   complexity_calibration:
#     - 不增加新的对象类型，不改颜色匹配规则，只增加 4 档旋转，复杂度与参考任务 2c181942 同级但更直接。
#     - 生成器至少强制一个 mover 拥有两个且方向不同的目标，避免大量样本退化成“只旋转 0°”。
#   implicit_assumptions_review:
#     - 原 solver 中 `pix + (dr,dc)` 假设所有副本与源精灵同 bbox/同朝向；变体下失效，必须改成“先旋转 mask，再以旋转后 bbox 中心对齐目标点”。
#     - 原生成器用 sprite 自身 h/w 约束 marks，默认不需要考虑旋转后的尺寸；变体下要按 `max(h,w)` 给目标中心留边距，
#       否则某些 90° 旋转会出界。这里仍允许 solver 裁剪，但生成器以完整放下为主，保持画面整洁。
#     - 最近边若出现平局，人类难以从少量样例推断 tie-break；因此 solver 有兜底优先级，但生成器主动规避平局。
#   placement_legality:
#     - mover 的 row/col marks 仍然在颜色之间互不复用，避免两个颜色共享同一目标中心。
#     - 同一个 mover 的多个 row 或 col mark 之间要求至少 `max(h,w)` 间隔，减少同色副本互相压住导致朝向看不清。
#     - 左侧精灵使用小型不对称模板，堆叠放入条带，不与 divider 重叠。
#   observability:
#     - 单色但明显不对称的 L/折线模板保证 90° 旋转肉眼可辨。
#     - 至少一个 mover 会出现两个不同旋转的副本；跨样本随机切换 vertical/horizontal 特例，使 top/bottom 和 left/right 两类旋转都能出现。
#   visual_cleanliness:
#     - 大部分目标会完整落在框内；虽然 solver 支持 clipping，但生成器不依赖大面积裁剪制造差异，避免输出显得杂乱。
#   color_inference:
#     - 颜色职责不变：颜色只负责“哪个精灵对应哪组边标”，旋转完全由空间位置决定。
#     - 不使用颜色值大小、黑色特权等不可见因素做排序或变换。
#   new_solver_assumptions_to_guarantee:
#     - mover 精灵必须是 8 连通单色组件。
#     - 对每个 mover，至少一个 row mark 和一个 col mark 都存在。
#     - 用于触发旋转差异的 special mover 不能在其目标点上发生最近边平局。


def largest_enclosed_component_bbox(
    mask: np.ndarray, *, connectivity: int = 4
) -> tuple[int, int, int, int] | None:
    """
    Local helper (only used by this solver): bbox of largest connected component in `mask`
    that does NOT touch the grid border.
    """
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool(
            (rs == 0).any()
            or (rs == int(H - 1)).any()
            or (cs == 0).any()
            or (cs == int(W - 1)).any()
        ):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))

    return best_bbox


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _nearest_side_and_rotation(
    tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> tuple[str, int]:
    """
    Choose the nearest same-color edge marker direction for a target.

    Priority order is only a safety fallback for accidental ties; generator avoids ties.
    Rotation k follows np.rot90 semantics:
    top=0, left=1, bottom=2, right=3.
    """
    dists = [
        (int(tr - fr0), "top", 0),
        (int(tc - fc0), "left", 1),
        (int(fr1 - tr), "bottom", 2),
        (int(fc1 - tc), "right", 3),
    ]
    dists.sort(key=lambda item: (int(item[0]), int(item[2])))
    _dist, side, krot = dists[0]
    return str(side), int(krot)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move colored sprites from the left strip into a framed area at marker-defined targets.

    Variant rule:
    - Targets are still the Cartesian product of matching row/column edge markers.
    - Before stamping at each target, rotate the sprite according to the nearest participating
      edge marker direction: top=0°, left=90° CCW, bottom=180°, right=270°.
    - Stamping remains centered and clipped to the frame interior; lower input sprites are
      painted first, upper sprites cover them.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert not (inner_bbox is None)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                }
            )

    move_colors = {int(s["color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        base_mask = np.array(s["mask"], dtype=bool)
        for tr, tc in targets[int(col)]:
            _side, krot = _nearest_side_and_rotation(int(tr), int(tc), int(fr0), int(fc0), int(fr1), int(fc1))
            rot_mask = _crop_bool(np.rot90(base_mask, int(krot)))
            anchor_r = int((rot_mask.shape[0] - 1) // 2)
            anchor_c = int((rot_mask.shape[1] - 1) // 2)

            for rr, cc in np.argwhere(rot_mask):
                gr = int(tr) + int(rr) - int(anchor_r)
                gc = int(tc) + int(cc) - int(anchor_c)
                if int(ir0) <= gr <= int(ir1) and int(ic0) <= gc <= int(ic1):
                    out[gr, gc] = int(col)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    pool = list(candidates)
    random.shuffle(pool)
    picked: list[int] = []
    for v in pool:
        if all(abs(int(v) - int(p)) >= int(min_spacing) for p in picked):
            picked.append(int(v))
            if len(picked) == n:
                return picked
    return picked if picked else None


def _rotation_signatures(mask: np.ndarray) -> set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    sigs: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()
    for krot in range(4):
        rot = _crop_bool(np.rot90(mask.astype(bool), int(krot)))
        pts = tuple(tuple(map(int, p)) for p in np.argwhere(rot))
        sigs.add(((int(rot.shape[0]), int(rot.shape[1])), pts))
    return sigs


def _make_asymmetric_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """
    Build a single-color sprite whose four rotations are visually distinguishable.
    """
    templates = [
        np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    ]

    candidates: list[np.ndarray] = []
    for tpl in templates:
        if len(_rotation_signatures(tpl.astype(bool))) != 4:
            continue
        for krot in range(4):
            rot = _crop_bool(np.rot90(tpl.astype(bool), int(krot)))
            if int(rot.shape[0]) <= int(max_h) and int(rot.shape[1]) <= int(max_w):
                arr = np.zeros(rot.shape, dtype=int)
                arr[rot] = int(color)
                candidates.append(arr)
    if not candidates:
        return None
    return np.array(random.choice(candidates), dtype=int)


def _valid_centers(lo: int, hi: int, span: int) -> list[int]:
    """
    Centers that allow any rotation with max extent `span` to stay fully inside [lo, hi].
    """
    anchor = int((int(span) - 1) // 2)
    tail = int(span) - 1 - int(anchor)
    start = int(lo) + int(anchor)
    end = int(hi) - int(tail)
    if start > end:
        return []
    return list(range(int(start), int(end) + 1))


def _nearest_side_code(
    tr: int, tc: int, fr0: int, fc0: int, fr1: int, fc1: int
) -> int | None:
    dists = {
        0: int(tr - fr0),   # top
        1: int(tc - fc0),   # left
        2: int(fr1 - tr),   # bottom
        3: int(fc1 - tc),   # right
    }
    best = min(dists.values())
    best_codes = [int(code) for code, dist in dists.items() if int(dist) == int(best)]
    if len(best_codes) != 1:
        return None
    return int(best_codes[0])


def _pick_special_marks(
    mode: str,
    row_candidates: list[int],
    col_candidates: list[int],
    span: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> tuple[list[int], list[int]] | None:
    rows = list(row_candidates)
    cols = list(col_candidates)
    random.shuffle(rows)
    random.shuffle(cols)

    if mode == "vertical":
        for c in cols:
            top_rows = [r for r in rows if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 0]
            bot_rows = [r for r in rows if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 2]
            random.shuffle(top_rows)
            random.shuffle(bot_rows)
            for rt in top_rows:
                for rb in bot_rows:
                    if abs(int(rt) - int(rb)) >= int(span):
                        return sorted([int(rt), int(rb)]), [int(c)]
        return None

    if mode == "horizontal":
        for r in rows:
            left_cols = [c for c in cols if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 1]
            right_cols = [c for c in cols if _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1)) == 3]
            random.shuffle(left_cols)
            random.shuffle(right_cols)
            for cl in left_cols:
                for cr in right_cols:
                    if abs(int(cl) - int(cr)) >= int(span):
                        return [int(r)], sorted([int(cl), int(cr)])
        return None

    return None


def _pick_generic_marks(
    row_candidates: list[int],
    col_candidates: list[int],
    span: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
    options: list[tuple[int, int]],
) -> tuple[list[int], list[int]] | None:
    for row_n, col_n in options:
        for _ in range(60):
            rows = _pick_spaced(list(row_candidates), int(row_n), int(span))
            cols = _pick_spaced(list(col_candidates), int(col_n), int(span))
            if not rows or not cols or len(rows) != int(row_n) or len(cols) != int(col_n):
                continue
            sides = [
                _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1))
                for r in rows
                for c in cols
            ]
            if any(side is None for side in sides):
                continue
            return sorted(map(int, rows)), sorted(map(int, cols))
    return None


def _assign_targets(
    sprites: list[np.ndarray],
    mover_colors: list[int],
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    fr0: int,
    fc0: int,
    fr1: int,
    fc1: int,
) -> dict[int, tuple[list[int], list[int]]] | None:
    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()

    special_mode = random.choice(["vertical", "horizontal"])

    for idx, (mc, sp) in enumerate(zip(mover_colors, sprites)):
        span = int(max(sp.shape))
        fit_rows = [r for r in _valid_centers(int(ir0), int(ir1), int(span)) if r not in occupied_rows]
        fit_cols = [c for c in _valid_centers(int(ic0), int(ic1), int(span)) if c not in occupied_cols]
        if not fit_rows or not fit_cols:
            return None

        if idx == 0:
            picked = _pick_special_marks(
                str(special_mode),
                fit_rows,
                fit_cols,
                int(span),
                int(fr0),
                int(fc0),
                int(fr1),
                int(fc1),
            )
            if picked is None:
                alt_mode = "horizontal" if special_mode == "vertical" else "vertical"
                picked = _pick_special_marks(
                    str(alt_mode),
                    fit_rows,
                    fit_cols,
                    int(span),
                    int(fr0),
                    int(fc0),
                    int(fr1),
                    int(fc1),
                )
            if picked is None:
                return None
        else:
            weighted_options = [(1, 1), (1, 1), (2, 1), (1, 2)]
            random.shuffle(weighted_options)
            picked = _pick_generic_marks(
                fit_rows,
                fit_cols,
                int(span),
                int(fr0),
                int(fc0),
                int(fr1),
                int(fc1),
                weighted_options,
            )
            if picked is None:
                return None

        rows, cols = picked
        mover_targets[int(mc)] = (list(map(int, rows)), list(map(int, cols)))
        occupied_rows.update(map(int, rows))
        occupied_cols.update(map(int, cols))

    special_color = int(mover_colors[0])
    s_rows, s_cols = mover_targets[int(special_color)]
    special_rotations = {
        _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1))
        for r in s_rows
        for c in s_cols
    }
    if None in special_rotations or len(special_rotations) < 2:
        return None

    return mover_targets


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None
    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1
    fr0, fr1 = 0, H - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_asymmetric_sprite(int(mc), 5, int(sprite_area_w))
        if s is None:
            return None
        sprites.append(s)

    mover_targets = _assign_targets(
        sprites,
        mover_colors,
        int(ir0),
        int(ic0),
        int(ir1),
        int(ic1),
        int(fr0),
        int(fc0),
        int(fr1),
        int(fc1),
    )
    if mover_targets is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)

    top_edge = np.full(frame_w, int(frame_base), dtype=int)
    for mc, (_rmarks, cmarks) in mover_targets.items():
        for cc in cmarks:
            top_edge[int(cc) - int(fc0)] = int(mc)
    inp[0, int(fc0) : int(fc0 + frame_w)] = top_edge
    inp[H - 1, int(fc0) : int(fc0 + frame_w)] = top_edge

    for r in range(1, H - 1):
        val = int(frame_base)
        for mc, (rmarks, _cmarks) in mover_targets.items():
            if int(r) in set(map(int, rmarks)):
                val = int(mc)
                break
        inp[int(r), int(fc0)] = int(val)
        inp[int(r), int(fc1)] = int(val)

    all_items: list[tuple[int, np.ndarray]] = [(int(mc), sp) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_asymmetric_sprite(int(dc_color), 4, int(sprite_area_w))
        if ds is None:
            return None
        all_items.append((int(dc_color), ds))

    cur_row = 0
    for color, sp in all_items:
        sh, sw = map(int, sp.shape)
        if sw > int(sprite_area_w) or cur_row + sh > H:
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(color)
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None

    move_rotations = set()
    for mc, (rows, cols) in mover_targets.items():
        for r in rows:
            for c in cols:
                code = _nearest_side_code(int(r), int(c), int(fr0), int(fc0), int(fr1), int(fc1))
                if code is None:
                    return None
                move_rotations.add(int(code))
    if len(move_rotations) < 2:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
