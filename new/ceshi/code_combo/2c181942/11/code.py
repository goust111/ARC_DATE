# Combo variant of 2c181942 (index 11)
# combo_spec: D4/0+D7/0
# Name: 多色精灵+锚点消歧
#
# Design analysis
# ----------------
# fused_rule:
#   overview:
#     - 保留原题骨架：中心空心方块 hub 不变，四边仍是四种不同颜色；输出仍是把散落
#       sprite 旋转后贴回对应边。
#     - D4/0 改动：sprite 不再是单色，而是双色。外层色 = 某条 hub 边颜色；内层色 =
#       一个非 hub 边色。输出贴附时保留整只双色对象，不能只搬运外层。
#     - D7/0 改动：去掉原题用来筛旋转的镜像对称约束；改为读 inner 锚点像素的位置。
#       每个 sprite 只有 1 个 inner 像素，它既是装饰色，也是锚点。
#   concrete_visualization:
#     - hub 仍是 N=4 的空心方块，角与内部为 bg=8，四边四种不同非背景色。
#     - sprite 仍是 8 连通、小尺寸（3x4 或 4x4 bbox、像素数 6-9）的非对称块。
#     - 在正确贴附朝向下：
#         * 接触边上恰有 N-2=2 个连续外层色像素；
#         * sprite 恰有 1 个 inner 像素，位于“近角标”固定位置：
#             left  -> (0,1)
#             right -> (h-1,w-2)
#             top   -> (1,w-1)
#             bot   -> (h-2,0)
#       这里故意用“近角标”而不是 bbox 真角点，这样 180 度错误朝向时锚点不会落到
#       接触边上；否则外层色检查本身就会把错误旋转排掉，D7 会退化。
#   behavior_trace:
#     - 原题：
#         1) 对某条 hub 边颜色 col，直接取 grid==col 的最大 8 连通块。
#         2) 旋转候选时要求 mask 沿接触轴镜像对称。
#         3) 找到合法朝向后贴回 hub。
#     - 融合后：
#         1) 先对 hub 外全部前景做 8 连通分块，拿到整只双色组件。
#         2) 外层色与 hub 边颜色一致，决定“配到哪一边”。
#         3) 对该组件尝试四个旋转；忽略锚点时会有两个旋转都能让接触边满足
#            “连续 2 格外层色”，即 0/180 二义性。
#         4) 再检查 inner 锚点是否落在该边规定的位置，只剩 1 个旋转。
#         5) 贴附时保留 inner 像素。
#     - 小例子：
#         * 以左边 sprite 为例，某模板在 as-is 与 rot180 两个朝向下，其右边界都能给出
#           连续 2 格外层色，所以只看外壳会有两个合法旋转。
#         * 但只有 as-is 时 inner 像素在 (0,1)；rot180 时会跑到 (h-1,w-2)。
#           因此“外层色配边 + 锚点定朝向”必须联用。
#   complexity_calibration:
#     - 如果 inner 锚点放到接触边上，错误旋转会被“接触边必须是外层色”直接剔除，
#       D7 退化；所以这里把锚点放到近角标位置。
#     - 如果模板完全自由生成，容易出现：
#         * 只有 1 个合法旋转，锚点不起作用；
#         * 90 度也合法，导致过多歧义；
#         * inner 像素跑到接触边，破坏外层匹配。
#       因此改用少量手工模板，并在 generator 中显式验证：
#         shape-only 合法旋转数 = 2，anchor-aware 合法旋转数 = 1。
#     - inner 色允许在不同 sprite 上重复，重复样例能支撑“inner 不负责配边”。
#   hidden_assumptions_audit:
#     - 原 `m = (g == col) & (~hub_mask)` 假设“整只 sprite 只有一种颜色”，融合后失效。
#     - 原 `spr_arr[~spr_mask] = bg` 默认“按外层色裁出的就是完整对象”，现在会丢掉
#       inner 锚点像素。
#     - 原 `np.flipud/np.fliplr` 对称性筛旋转的前提被 D7 明确取消。
#     - 新 solver 的输入假设改为：
#         * 每个真实 sprite 是单个 8 连通前景组件；
#         * 每个真实 sprite 恰有 2 种颜色，且恰有 1 个 inner 像素；
#         * inner 色不会复用 hub 边色；
#         * 对目标 side，shape-only 有且只有 2 个合法旋转，anchor-aware 有且只有 1 个。
#       generator 会逐项构造并验证这些前提。
#   placement_legality:
#     - 仍使用 output-first：先在输出里贴好 hub 和定向 sprite，再随机旋转散放回输入。
#     - hub 四周预留 >= 最大 sprite 尺寸 + 2 的 margin；散放时使用
#       random_free_location(..., padding=2)，避免组件互相接触并成一块。
#     - 所有 attached/scattered 放置都检查不越界、不与已有对象冲突。
#   visual_observability:
#     - 人类能从少量样例看到：
#         * 贴回去的是整只双色对象，而不是只保留外壳；
#         * 外壳色与 hub 边色一一对应，inner 色在不同边可重复，说明 outer 才负责匹配；
#         * 同类模板在不同样例中锚点落在固定“近角标”位置，说明旋转由锚点决定。
#   cleanliness:
#     - 每只 sprite 只额外增加 1 个 inner 锚点像素，不叠加额外标记层，画面足够干净。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


SIDES = ("top", "bot", "left", "right")
CONTACT_EDGE = {"top": "bot", "bot": "top", "left": "right", "right": "left"}


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


# Canonical templates are defined in the attached "left" orientation:
# contact edge = right side, and the inner anchor sits at (0,1).
_CANONICAL_TEMPLATES: tuple[np.ndarray, ...] = (
    _rows_to_mask(
        [
            "1111",
            "1001",
            "0100",
        ]
    ),
    _rows_to_mask(
        [
            "1111",
            "1101",
            "0100",
        ]
    ),
    _rows_to_mask(
        [
            "1110",
            "1011",
            "0001",
        ]
    ),
    _rows_to_mask(
        [
            "1111",
            "1001",
            "0100",
            "0100",
        ]
    ),
)

_BASE_ANCHOR = (0, 1)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
    for n in range(3, 7):
        for r in range(0, H - n + 1):
            for c in range(0, W - n + 1):
                if (
                    int(g[r, c]) != bg
                    or int(g[r, c + n - 1]) != bg
                    or int(g[r + n - 1, c]) != bg
                    or int(g[r + n - 1, c + n - 1]) != bg
                ):
                    continue

                inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                if not bool((inner == bg).all()):
                    continue

                top = g[r, c + 1 : c + n - 1]
                bot = g[r + n - 1, c + 1 : c + n - 1]
                left = g[r + 1 : r + n - 1, c]
                right = g[r + 1 : r + n - 1, c + n - 1]
                if bool((top == bg).any()) or bool((bot == bg).any()) or bool((left == bg).any()) or bool((right == bg).any()):
                    continue
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue

                cr = float(r + (n - 1) / 2)
                cc = float(c + (n - 1) / 2)
                score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
                cand = (
                    float(score),
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or cand[0] < best[0]:
                    best = cand
    return best


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    fg = (g != bg) & (~hub_mask)
    lab, nc = label_mask(fg, connectivity=8)
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == cid
        arr = np.where(comp_mask, g, bg)
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        out.append((int(cid), arr, comp_mask))
    return out


def _anchor_target(side: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side == "left":
        return 0, 1
    if side == "right":
        return h - 1, w - 2
    if side == "top":
        return 1, w - 1
    return h - 2, 0


def _anchor_ok(arr: np.ndarray, mask: np.ndarray, side: str, outer_color: int) -> bool:
    pts = np.argwhere(mask & (arr != int(outer_color)))
    if pts.shape[0] != 1:
        return False
    want = _anchor_target(side, mask.shape)
    rr, cc = map(int, pts[0])
    return (rr, cc) == want


def _shape_match_only(arr: np.ndarray, mask: np.ndarray, need_edge: str, target_len: int, outer_color: int):
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None

    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target_len) or not _is_contig(rows):
            return None
        if any(int(arr[rr, edge_c]) != int(outer_color) for rr in rows):
            return None
    else:
        _typ, edge_r, cols = seg
        if len(cols) != int(target_len) or not _is_contig(cols):
            return None
        if any(int(arr[edge_r, cc]) != int(outer_color) for cc in cols):
            return None
    return seg


def _attachment_info(
    arr: np.ndarray,
    mask: np.ndarray,
    side: str,
    need_edge: str,
    target_len: int,
    outer_color: int,
):
    colors = {int(v) for v in arr[mask]}
    if len(colors) != 2 or int(outer_color) not in colors:
        return None
    if not _anchor_ok(arr, mask, side, outer_color):
        return None
    return _shape_match_only(arr, mask, need_edge, target_len, outer_color)


def _placeable_rotation(
    arr: np.ndarray,
    mask: np.ndarray,
    side: str,
    need_edge: str,
    fixed: int,
    span: list[int],
    target_len: int,
    outer_color: int,
    out: np.ndarray,
    bg: int,
):
    H, W = out.shape
    for krot in range(4):
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)

        seg = _attachment_info(rot_arr, rot_mask, side, need_edge, target_len, outer_color)
        if seg is None:
            continue

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(rot_mask)}
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in placed):
            continue

        ok = True
        for rr, cc in placed:
            val = int(rot_arr[rr - dr, cc - dc])
            if val == int(bg):
                continue
            if int(out[rr, cc]) != int(bg) and int(out[rr, cc]) != val:
                ok = False
                break
        if ok:
            return rot_arr, rot_mask, int(dr), int(dc)
    return None


def _valid_rotation_stats(arr: np.ndarray, mask: np.ndarray, side: str, outer_color: int, target_len: int) -> tuple[list[int], list[int]]:
    need_edge = CONTACT_EDGE[side]
    shape_hits: list[int] = []
    anchor_hits: list[int] = []
    for krot in range(4):
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _shape_match_only(rot_arr, rot_mask, need_edge, target_len, outer_color) is not None:
            shape_hits.append(int(krot))
        if _attachment_info(rot_arr, rot_mask, side, need_edge, target_len, outer_color) is not None:
            anchor_hits.append(int(krot))
    return shape_hits, anchor_hits


def _orient_from_left(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.rot90(arr, 2)
    if side == "top":
        return np.rot90(arr, -1)
    return np.rot90(arr, 1)


def _make_composite_sprite(side: str, outer_color: int, inner_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    base_mask = rng.choice(_CANONICAL_TEMPLATES).copy()
    base_arr = np.full(base_mask.shape, -1, dtype=int)
    base_arr[base_mask] = int(outer_color)
    base_arr[int(_BASE_ANCHOR[0]), int(_BASE_ANCHOR[1])] = int(inner_color)

    arr = _orient_from_left(base_arr, side)
    mask = arr != -1
    arr = np.where(mask, arr, 0).astype(int)
    arr, mask = _crop_to_mask(arr, mask)
    return arr, mask


def _attachment_position(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + n), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + n)


def _variant_metrics(inp: np.ndarray) -> dict[str, object]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros(g.shape, dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    side_colors = {"top": int(top_col), "bot": int(bot_col), "left": int(left_col), "right": int(right_col)}
    edge_color_set = set(side_colors.values())
    components = _extract_components(g, hub_mask, bg)

    shape_counts: list[int] = []
    anchor_counts: list[int] = []
    metrics = {
        "real_sprites": 0,
        "two_color_real": 0,
        "shape_ambiguous": 0,
        "anchor_unique": 0,
        "shape_hit_counts": shape_counts,
        "anchor_hit_counts": anchor_counts,
    }

    for _cid, arr, mask in components:
        colors = {int(v) for v in arr[mask]}
        outer_hits = [side for side, col in side_colors.items() if int(col) in colors]
        if len(outer_hits) != 1:
            continue

        side = outer_hits[0]
        outer_color = int(side_colors[side])
        metrics["real_sprites"] += 1
        if len(colors) == 2 and int(outer_color) in colors:
            metrics["two_color_real"] += 1

        shape_hits, anchor_hits = _valid_rotation_stats(arr, mask, side, outer_color, int(n - 2))
        shape_counts.append(len(shape_hits))
        anchor_counts.append(len(anchor_hits))
        if len(shape_hits) == 2:
            metrics["shape_ambiguous"] += 1
        if len(anchor_hits) == 1:
            metrics["anchor_unique"] += 1

    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    out = np.full_like(g, int(bg))
    out[r : r + n, c : c + n] = g[r : r + n, c : c + n]

    target = int(n - 2)
    components = _extract_components(g, hub_mask, bg)
    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    used_ids: set[int] = set()
    for side, (outer_color, (need_edge, fixed, span)) in segs.items():
        best = None
        best_score = None
        for cid, arr, mask in components:
            if cid in used_ids:
                continue
            vals = {int(v) for v in arr[mask]}
            if int(outer_color) not in vals:
                continue

            placed = _placeable_rotation(arr, mask, side, need_edge, fixed, span, target, outer_color, out, bg)
            if placed is None:
                continue

            score = (int((arr[mask] == int(outer_color)).sum()), int(mask.sum()))
            if best is None or score > best_score:
                best = (cid, placed)
                best_score = score

        if best is None:
            continue

        cid, (arr, mask, dr, dc) = best
        used_ids.add(int(cid))
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    edge_colors = [int(top_col), int(bot_col), int(left_col), int(right_col)]
    side_colors = {"top": int(top_col), "bot": int(bot_col), "left": int(left_col), "right": int(right_col)}
    remaining = [int(c) for c in all_colors if int(c) not in edge_colors]
    if not remaining:
        return None

    sides = ["top", "bot", "left", "right"]
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        outer_color = int(side_colors[side])
        inner_color = int(rng.choice(remaining))
        arr, mask = _make_composite_sprite(side, outer_color, inner_color, rng)

        shape_hits, anchor_hits = _valid_rotation_stats(arr, mask, side, outer_color, edge_len)
        if shape_hits != [0, 2]:
            return None
        if anchor_hits != [0]:
            return None
        sprite_defs[side] = (arr, mask)

    max_sprite_h = max(arr.shape[0] for arr, _mask in sprite_defs.values())
    max_sprite_w = max(arr.shape[1] for arr, _mask in sprite_defs.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + n - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + n - 1] = int(right_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        arr, mask = sprite_defs[side]
        sr, sc = _attachment_position(mask, side, hub_r, hub_c, n)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if mask[rr, cc] and int(out[sr + rr, sc + cc]) != int(bg):
                    return None

        patch = np.where(mask, arr, bg).astype(int)
        paste(out, patch, sr, sc, transparent=bg)
        attached_sprites[side] = (patch, mask, sr, sc)

    inp = np.full((H, W), int(bg), dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    for side in sprite_sides:
        patch, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rot_patch = np.rot90(patch, k)
        rot_mask = np.rot90(mask, k)
        rot_patch, rot_mask = _crop_to_mask(rot_patch, rot_mask)
        scatter = np.where(rot_mask, rot_patch, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, scatter, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, scatter, pr, pc, transparent=bg)
        paste(inp, scatter, pr, pc, transparent=bg)

    if rng.random() < 0.3 and len(remaining) >= 2:
        d_outer = int(rng.choice(remaining))
        d_inner_choices = [c for c in remaining if int(c) != int(d_outer)]
        if d_inner_choices:
            d_inner = int(rng.choice(d_inner_choices))
            d_side = rng.choice(sides)
            dist_arr, dist_mask = _make_composite_sprite(d_side, d_outer, d_inner, rng)
            dist_patch = np.where(dist_mask, dist_arr, bg).astype(int)
            try:
                pr, pc = random_free_location(canvas, dist_patch, transparent=bg, padding=2, padding_connectivity=8)
                paste(canvas, dist_patch, pr, pc, transparent=bg)
                paste(inp, dist_patch, pr, pc, transparent=bg)
            except ValueError:
                pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _variant_metrics(inp)
    if int(metrics["real_sprites"]) != int(K):
        return None
    if int(metrics["two_color_real"]) != int(K):
        return None
    if int(metrics["shape_ambiguous"]) != int(K):
        return None
    if int(metrics["anchor_unique"]) != int(K):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
