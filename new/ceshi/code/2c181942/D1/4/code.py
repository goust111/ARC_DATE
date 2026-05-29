# Variant of 2c181942
# Dimension: D1 — 匹配条件：颜色配对机制 (match_cond)
# Change: s.head.color == d.edge.color — 精灵颜色与hub边缘颜色相同即配对，颜色唯一决定精灵应贴附... → 边缘颜色渐变顺序匹配
"""
design_analysis:
  variant_core:
    original_rule: |
      hub 四条边各是单色，散落 sprite 也整体单色；solver 先按颜色把 sprite 分给某条边，
      再只靠几何约束决定旋转与贴附位置。
    new_rule: |
      hub 四条边改成长度 3 的颜色序列；sprite 改成多色条纹对象。solver 不再按单色筛对象，
      而是先提取所有非背景连通块，再尝试旋转，并检查“接触边上的有序颜色序列”是否与 hub 对应边完全一致。
  concretization_and_trace:
    hub_shape: |
      仍是中心空心方块，但固定使用 N=5，使每条边有 3 个可见颜色格：
      . a b c .
      l . . . r
      m . . . s
      n . . . t
      . x y z .
      角和内部仍是背景色。
    sprite_shape: |
      sprite 的 mask 仍遵守原题的几何约束：附着朝向下，接触边必须正好有 3 个连续格，
      并在接触边法线方向上镜像对称。区别在于 sprite 不再单色，而是沿“接触边方向”分成 3 条彩色条纹。
      对 left/right 附着时按行上到下着色；对 top/bot 附着时按列左到右着色。
    local_trace: |
      原逻辑：若 hub top 边颜色是 2，则只看网格里颜色 2 的最大连通块，再找它哪个旋转能让接触边长度正确。
      新逻辑：若 hub top 边序列是 [2,4,7]，solver 不能再看单色 2，而要对每个外部连通块都试旋转；
      某个旋转若满足 bottom 接触边 3 连格，且这 3 格从左到右正好是 [2,4,7]，才说明它对应 top 边。
      差异在于“候选筛选”从单值颜色变成了有方向的短序列匹配。
  complexity_calibration:
    choices: |
      为了让 D1 肉眼可见，N 从原 generator 的 4 提到 5，使边长从 2 增到 3；长度 2 的序列太弱，
      很容易看起来像“两个颜色碰巧不同”，不利于人类从 2-4 个样例归纳。
      同时不把 N 拉得太大，避免 hub 和 sprite 都变得臃肿、放置失败率升高。
    ambiguity_control: |
      四条边序列使用共享颜色池的循环窗口，例如 [a,b,c] / [b,c,d] / [c,d,a] / [d,a,b]。
      这样单个颜色会在多条边重复出现，旧规则失效；但完整序列仍唯一，且不与其它边的逆序相同。
  implicit_assumption_audit:
    original_dependencies:
      - "_find_hub 要求四条边各自是单色；新变体中这条前提失效，因此必须改成只要求边全是非背景，并记录整个序列。"
      - "原 solver 用 (g == edge_color) 提取 sprite；新变体 sprite 是多色对象，这种按单色取 mask 的写法会把一个 sprite 拆碎。"
      - "原 generator 把 sprite 全部填成 side_color；新变体里如果继续这样做，就会退化回原题。"
      - "原 distractor 只需颜色不在 hub 边颜色中；新变体里要避免 distractor 的接触边序列等于任一 hub 边序列。"
    new_safe_patterns: |
      新 solver 改为：先全局提取 hub 外全部非背景 8 连通块，再对每个块的各个旋转检查几何合法性和接触边序列。
      这样不会依赖“对象单色”这个旧前提。
  placement_legality:
    hub: |
      hub 仍放在近中心区域，margin 至少大于最大 sprite 外接尺寸 + 2，保证四周有足够空间贴附和散落。
    output_first: |
      先构造目标输出：hub + 已贴附的多色 sprite；确认不越界不重叠后，再把 sprite 拆下来旋转并散落到输入。
    scattered_input: |
      散落仍用 random_free_location，padding=2、8 邻域，避免 sprite 之间或 sprite 与 hub 彼此粘连，
      从而保证 solver 的连通块提取与生成意图一致。
  visual_observability:
    rationale: |
      输入和输出中都能直接看到 hub 边不是单色，而是有方向的三色条纹；sprite 也明显是三色条纹。
      因为多个 side 共享颜色，观察者不能只靠“这个 sprite 含有红色”来匹配，而必须看完整顺序。
  visual_cleanliness:
    layout: |
      仍沿用原题的“中心 hub + 四周散落 object”版式，没有叠层标记或额外文本。对象数维持 2-4 个，
      可选 distractor 最多 1 个，避免画面过杂。
  color_inference:
    structure: |
      不使用颜色编号做任何不可见排序。序列顺序完全由像素在边上的空间顺序定义：
      top/bot 按左到右，left/right 按上到下。人类可以直接看见这一点。
  solver_assumptions_after_change:
    guaranteed_by_generator:
      - "hub 固定 N=5，且四条边都是长度 3 的非背景序列，每条边至少有 2 种颜色，实际生成时是 3 种。"
      - "每个有效 sprite 是一个多色 8 连通块，且只有一个旋转会同时满足原题几何约束和目标 side 的序列约束。"
      - "不同 hub 边序列唯一，且不等于其它边序列的逆序，因此不会因方向翻转产生错误匹配。"
"""

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


SIDE_ORDER = ("top", "right", "bot", "left")
N = 5
EDGE_LEN = N - 2
CANONICAL_LEFT_MASKS = [
    np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
]


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.where(mask, 1, 0).astype(int)
    cropped, cropped_mask = _crop_to_mask(arr, mask.astype(bool))
    return cropped_mask


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _edge_signature(arr: np.ndarray, mask: np.ndarray, side: str) -> tuple[int, ...] | None:
    seg = _edge_segment(mask, side)
    if seg is None:
        return None
    if side in ("top", "bot"):
        _typ, rr, cols = seg
        if not _is_contig(cols):
            return None
        vals = tuple(int(arr[int(rr), int(cc)]) for cc in cols)
    else:
        _typ, cc, rows = seg
        if not _is_contig(rows):
            return None
        vals = tuple(int(arr[int(rr), int(cc)]) for rr in rows)
    return vals


def _mask_matches_attachment(mask: np.ndarray, need_edge: str, target: int) -> bool:
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return False
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return False
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return False
    if need_edge in ("left", "right"):
        _typ, _cc, rows = seg
        return len(rows) == int(target) and _is_contig(rows)
    _typ, _rr, cols = seg
    return len(cols) == int(target) and _is_contig(cols)


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
    for n in range(N, N + 1):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                if (
                    int(g[int(r), int(c)]) != int(bg)
                    or int(g[int(r), int(c + n - 1)]) != int(bg)
                    or int(g[int(r + n - 1), int(c)]) != int(bg)
                    or int(g[int(r + n - 1), int(c + n - 1)]) != int(bg)
                ):
                    continue

                inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                if not bool((inner == int(bg)).all()):
                    continue

                top = tuple(int(x) for x in g[int(r), int(c + 1) : int(c + n - 1)])
                bot = tuple(int(x) for x in g[int(r + n - 1), int(c + 1) : int(c + n - 1)])
                left = tuple(int(x) for x in g[int(r + 1) : int(r + n - 1), int(c)])
                right = tuple(int(x) for x in g[int(r + 1) : int(r + n - 1), int(c + n - 1)])
                side_seqs = {"top": top, "bot": bot, "left": left, "right": right}

                if any(int(bg) in seq for seq in side_seqs.values()):
                    continue
                if any(len(set(seq)) < 2 for seq in side_seqs.values()):
                    continue
                seqs = list(side_seqs.values())
                if len(set(seqs)) != 4:
                    continue

                cr = float(r + (n - 1) / 2.0)
                cc = float(c + (n - 1) / 2.0)
                score = abs(float(cr - (H - 1) / 2.0)) + abs(float(cc - (W - 1) / 2.0))
                cand = (float(score), (int(r), int(c), int(n)), side_seqs)
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach striped multi-color sprites to the hub side whose ordered edge sequence they match.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert not (hub is None)

    _score, (r, c, n), hub_sides = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    comp_mask = (g != int(bg)) & (~hub_mask)
    labc, nc = label_mask(comp_mask, connectivity=8)
    components = []
    for cid in range(1, int(nc) + 1):
        mask = labc == int(cid)
        if not bool(mask.any()):
            continue
        arr = g.copy()
        arr[~mask] = int(bg)
        arr, mask = _crop_to_mask(arr, mask)
        components.append((arr, mask))

    segs = {
        "top": ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1)))),
        "bot": ("top", int(r + n), list(range(int(c + 1), int(c + n - 1)))),
        "left": ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1)))),
        "right": ("left", int(c + n), list(range(int(r + 1), int(r + n - 1)))),
    }

    used_components: set[int] = set()
    for side in ("top", "bot", "left", "right"):
        target_seq = tuple(int(x) for x in hub_sides[side])
        need_edge, fixed, span = segs[side]
        found = None

        for comp_idx, (spr_arr, spr_mask) in enumerate(components):
            if comp_idx in used_components:
                continue
            for krot in range(4):
                a = np.rot90(spr_arr, int(krot))
                msk = np.rot90(spr_mask, int(krot))
                a, msk = _crop_to_mask(a, msk)

                if not _mask_matches_attachment(msk, need_edge, int(n - 2)):
                    continue
                if _edge_signature(a, msk, need_edge) != target_seq:
                    continue

                seg = _edge_segment(msk, need_edge)
                assert seg is not None
                if need_edge in ("left", "right"):
                    _typ, edge_c, rows = seg
                    dr = int(span[0] - rows[0])
                    dc = int(fixed - edge_c)
                else:
                    _typ, edge_r, cols = seg
                    dr = int(fixed - edge_r)
                    dc = int(span[0] - cols[0])

                placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
                if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
                    continue

                ok = True
                for R, C in placed:
                    val = int(a[int(R - dr), int(C - dc)])
                    if val == int(bg):
                        continue
                    if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
                        ok = False
                        break
                if not ok:
                    continue

                found = (comp_idx, a, msk, int(dr), int(dc))
                break
            if found is not None:
                break

        if found is None:
            continue

        comp_idx, a, msk, dr, dc = found
        used_components.add(int(comp_idx))
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


def _transform_left_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    if side == "bot":
        return np.rot90(mask, 1)
    raise ValueError(side)


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    if int(edge_len) != EDGE_LEN:
        return None
    candidates = []
    for base in CANONICAL_LEFT_MASKS:
        mask = _crop_mask(base)
        valid = 0
        for k in range(4):
            rot = _crop_mask(np.rot90(mask, int(k)))
            if _mask_matches_attachment(rot, "right", int(edge_len)):
                valid += 1
        if valid == 1:
            candidates.append(mask)
    if not candidates:
        return None
    picked = rng.choice(candidates)
    return _transform_left_mask(picked, side)


def _colorize_attached_mask(mask: np.ndarray, side: str, seq: tuple[int, ...], bg: int) -> np.ndarray:
    arr = np.full(mask.shape, int(bg), dtype=int)
    if side in ("left", "right"):
        active_rows = [int(rr) for rr in range(mask.shape[0]) if bool(mask[int(rr)].any())]
        if len(active_rows) != len(seq):
            raise ValueError("row stripe count mismatch")
        for idx, rr in enumerate(active_rows):
            arr[int(rr), mask[int(rr)]] = int(seq[int(idx)])
    else:
        active_cols = [int(cc) for cc in range(mask.shape[1]) if bool(mask[:, int(cc)].any())]
        if len(active_cols) != len(seq):
            raise ValueError("col stripe count mismatch")
        for idx, cc in enumerate(active_cols):
            arr[mask[:, int(cc)], int(cc)] = int(seq[int(idx)])
    return arr


def _build_side_sequences(rng: random.Random, bg: int) -> dict[str, tuple[int, int, int]]:
    palette = rng.sample([c for c in range(10) if c != int(bg)], 4)
    a, b, c, d = map(int, palette)
    return {
        "top": (a, b, c),
        "right": (b, c, d),
        "bot": (c, d, a),
        "left": (d, a, b),
    }


def _place_hub(out: np.ndarray, hub_r: int, hub_c: int, side_seqs: dict[str, tuple[int, int, int]], bg: int) -> None:
    out[int(hub_r) : int(hub_r + N), int(hub_c) : int(hub_c + N)] = int(bg)
    for idx, col in enumerate(side_seqs["top"], start=1):
        out[int(hub_r), int(hub_c + idx)] = int(col)
    for idx, col in enumerate(side_seqs["bot"], start=1):
        out[int(hub_r + N - 1), int(hub_c + idx)] = int(col)
    for idx, col in enumerate(side_seqs["left"], start=1):
        out[int(hub_r + idx), int(hub_c)] = int(col)
    for idx, col in enumerate(side_seqs["right"], start=1):
        out[int(hub_r + idx), int(hub_c + N - 1)] = int(col)


def _make_distractor_sequence(
    rng: random.Random,
    bg: int,
    forbidden: set[tuple[int, ...]],
) -> tuple[int, int, int] | None:
    colors = [c for c in range(10) if c != int(bg)]
    rng.shuffle(colors)
    for seq in itertools.permutations(colors[:5], EDGE_LEN):
        cand = tuple(int(x) for x in seq)
        if cand in forbidden or cand[::-1] in forbidden:
            continue
        return cand
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    side_seqs = _build_side_sequences(rng, bg)
    sides = ["top", "bot", "left", "right"]
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks: dict[str, np.ndarray] = {}
    sprite_arrays: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(EDGE_LEN, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask
        sprite_arrays[side] = _colorize_attached_mask(mask, side, side_seqs[side], bg)

    max_sprite_h = max(arr.shape[0] for arr in sprite_arrays.values())
    max_sprite_w = max(arr.shape[1] for arr in sprite_arrays.values())
    max_extent = max(max_sprite_h, max_sprite_w)
    margin = max_extent + 2

    hub_r_min = margin
    hub_r_max = H - N - margin
    hub_c_min = margin
    hub_c_max = W - N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    _place_hub(out, hub_r, hub_c, side_seqs, bg)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        arr = sprite_arrays[side]
        mask = sprite_masks[side]
        sh, sw = mask.shape

        if side == "top":
            filled = np.where(mask[-1])[0]
            sc = int(hub_c + 1 - filled[0])
            sr = int(hub_r - sh)
        elif side == "bot":
            filled = np.where(mask[0])[0]
            sc = int(hub_c + 1 - filled[0])
            sr = int(hub_r + N)
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = int(hub_r + 1 - filled[0])
            sc = int(hub_c - sw)
        else:
            filled = np.where(mask[:, 0])[0]
            sr = int(hub_r + 1 - filled[0])
            sc = int(hub_c + N)

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if mask[int(rr), int(cc)] and int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None

        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), int(bg), dtype=int)
    inp[int(hub_r) : int(hub_r + N), int(hub_c) : int(hub_c + N)] = out[int(hub_r) : int(hub_r + N), int(hub_c) : int(hub_c + N)]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(k))
        rotated_mask = np.rot90(mask, int(k))
        cropped_arr, cropped_mask = _crop_to_mask(rotated_arr, rotated_mask)
        patch = np.where(cropped_mask, cropped_arr, int(bg)).astype(int)

        try:
            pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    if rng.random() < 0.3:
        forbidden = set(side_seqs.values())
        distractor_seq = _make_distractor_sequence(rng, bg, forbidden)
        if distractor_seq is not None:
            dist_side = rng.choice(sides)
            dist_mask = _make_sprite_mask(EDGE_LEN, dist_side, rng)
            if dist_mask is not None:
                dist_arr = _colorize_attached_mask(dist_mask, dist_side, distractor_seq, bg)
                k = rng.choice([0, 1, 2, 3])
                dist_arr = np.rot90(dist_arr, int(k))
                dist_mask = np.rot90(dist_mask, int(k))
                dist_arr, dist_mask = _crop_to_mask(dist_arr, dist_mask)
                patch = np.where(dist_mask, dist_arr, int(bg)).astype(int)
                try:
                    pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
                    paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
                    paste(inp, patch, int(pr), int(pc), transparent=int(bg))
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
