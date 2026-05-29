# Variant of 2c181942
# Dimension: D4 — 源对象：精灵的形态与属性 (src)
# Change: 单色实心精灵，8连通，沿接触轴对称，像素数4-12，bbox高2-4宽2-6，分散放置且远离hub... → 多色复合精灵
#
# Design analysis
# spec:
#   variant_goal:
#     core_change: |
#       每个可匹配精灵不再是单色块，而是双色复合对象：
#       外层壳色 = 某条 hub 边的颜色；内层装饰色 = 另一个非 hub 边色。
#       输出贴附后，内层装饰色必须保留，所以解题者不能再把“整只精灵的颜色”
#       当作匹配依据，而要识别“只有外层壳色负责匹配”。
#   concrete_visualization:
#     hub:
#       unchanged: N=4 的空心方框，四边四种不同颜色，角和内部仍为 bg=8。
#     sprites:
#       shape: |
#         仍为 8 连通、沿接触轴对称、接触边长度恰好为 2 的小精灵。
#         但精灵内部有 1-2 个像素改成另一种颜色，形成可见装饰。
#       outer_layer: |
#         外层颜色保持 8 连通，并且有效接触边上的两个像素都必须是外层色。
#       inner_layer: |
#         内层色只出现在精灵内部纵向条带里，不参与 hub 匹配；输出中保留。
#       count: 2-4 个可匹配精灵，外加 0-1 个双色干扰物。
#   behavior_trace:
#     original_flow: |
#       原 solver 对每条 hub 边颜色 col，直接取 grid==col 的最大 8 连通块，
#       这等价于“精灵整体单色”。
#     new_flow: |
#       现在必须先对 hub 外全部前景做 8 连通分块，拿到整只双色对象；
#       再判断该对象是否含有目标外层色 col，以及在某个旋转下该色是否恰好
#       出现在合法接触边上。找到后旋转/平移整只对象，连同内层色一起粘到 hub。
#     concrete_example: |
#       例如左侧 hub 边颜色为红色，散落精灵是一只“红壳 + 黄芯”的 4x4 对称块。
#       原逻辑只会抓到红色像素并丢失黄色内芯；新逻辑先抓整只组件，再检查
#       其右接触边两格是否都是红色，最后把整只红黄组件贴到 hub 左边。
#   complexity_calibration:
#     rationale: |
#       双色精灵若完全自由生成，容易出现外层断裂、旋转歧义、或内层色恰好与别的
#       hub 边色重合导致匹配混乱。这里改用少量手工模板：
#       1) 保证外层色始终 8 连通；
#       2) 左边界长度与右接触边长度不同，避免 180 度旋转仍合法；
#       3) 内层色复用 2-3 种非 hub 边色，视觉上能重复支撑“内层不负责匹配”。
#   hidden_assumptions_audit:
#     old_assumptions_broken:
#       - "`m = (g == col) & (~hub_mask)` 假设 col 就是整只精灵的全部像素。现在失效。"
#       - "`spr_arr[~spr_mask] = bg` 假设用 col 的连通块裁下来的就是完整对象。现在会丢内层。"
#       - "冲突检测里按单色比较原本无害；现在必须粘贴整只多色数组。"
#     new_assumptions:
#       - "每个可匹配精灵是单个 8 连通前景组件。"
#       - "hub 边色不会作为其他精灵的内层色重复出现。"
#       - "真正的接触边两格都属于外层色，而不是内层色。"
#     generator_guarantees:
#       - "所有可匹配/干扰精灵都按上述模板生成并以 padding>=2 散开。"
#       - "内层色从非 hub 边色池中选取，避免把内层误当匹配色。"
#   placement_legality:
#     output_first: |
#       先在输出里把 hub 和已定向精灵贴好，确认不越界、不互相重叠；
#       再把精灵旋转到非贴附朝向，随机散放回输入。
#     spacing: |
#       hub 周围预留 >= max_extent + 2 的边距；散放时使用 random_free_location
#       且 padding=2，保证各对象不会接触或粘连成一个组件。
#   visual_observability:
#     why_human_can_infer: |
#       多个样例里都会看到：贴到 hub 上的是“双色整只对象”，而不是只有与边同色
#       的外壳；同时外壳色与 hub 边色一一对应，内层色在不同边上可重复出现，
#       因而人类能推断“匹配看外层，保留内层”。
#   cleanliness:
#     constraints: |
#       模板数量少、尺寸小、颜色种类受控；不在 hub 区域叠加额外标记，避免画面杂乱。

from __future__ import annotations

import random

import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_CANONICAL_TEMPLATES: tuple[tuple[np.ndarray, tuple[tuple[int, int], ...]], ...] = (
    (
        _rows_to_mask(
            [
                "1000",
                "1111",
                "1111",
                "1000",
            ]
        ),
        ((1, 1), (2, 1)),
    ),
    (
        _rows_to_mask(
            [
                "1100",
                "1111",
                "1111",
                "1100",
            ]
        ),
        ((1, 1), (2, 1)),
    ),
    (
        _rows_to_mask(
            [
                "10000",
                "11111",
                "11111",
                "10000",
            ]
        ),
        ((1, 1), (2, 1)),
    ),
)


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


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int):
    fg = (g != bg) & (~hub_mask)
    lab, nc = label_mask(fg, connectivity=8)
    out = []
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == cid
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = g[comp_mask]
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        out.append((int(cid), arr, comp_mask))
    return out


def _attachment_info(arr: np.ndarray, mask: np.ndarray, need_edge: str, target_len: int, outer_color: int):
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return None
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return None

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

    colors = {int(v) for v in arr[mask]}
    if len(colors) < 2:
        return None
    return seg


def _placeable_rotation(
    arr: np.ndarray,
    mask: np.ndarray,
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
        seg = _attachment_info(rot_arr, rot_mask, need_edge, target_len, outer_color)
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


def _orient_from_left(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, -1)
    return np.rot90(arr, 1)


def _make_composite_sprite(side: str, outer_color: int, inner_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    template_mask, base_inner = rng.choice(_CANONICAL_TEMPLATES)
    template_mask = template_mask.copy()

    inner_options = [
        (base_inner[0],),
        (base_inner[1],),
        base_inner,
    ]
    inner_cells = rng.choice(inner_options)

    left_arr = np.full(template_mask.shape, int(outer_color), dtype=int)
    left_arr[~template_mask] = -1
    for rr, cc in inner_cells:
        left_arr[int(rr), int(cc)] = int(inner_color)

    left_arr = _orient_from_left(left_arr, side)
    left_mask = left_arr != -1
    left_arr = np.where(left_mask, left_arr, 0).astype(int)
    left_arr, left_mask = _crop_to_mask(left_arr, left_mask)
    return left_arr, left_mask


def _valid_rotation_count(arr: np.ndarray, mask: np.ndarray, need_edge: str, target_len: int, outer_color: int) -> int:
    count = 0
    for krot in range(4):
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _attachment_info(rot_arr, rot_mask, need_edge, target_len, outer_color) is not None:
            count += 1
    return count


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
    for _side, (outer_color, (need_edge, fixed, span)) in segs.items():
        best = None
        best_score = None
        for cid, arr, mask in components:
            if cid in used_ids:
                continue
            vals = {int(v) for v in arr[mask]}
            if int(outer_color) not in vals or len(vals) < 2:
                continue

            placed = _placeable_rotation(arr, mask, need_edge, fixed, span, target, outer_color, out, bg)
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
    edge_colors = [top_col, bot_col, left_col, right_col]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}
    remaining = [c for c in all_colors if c not in edge_colors]
    if len(remaining) < 2:
        return None

    accent_palette = remaining[: min(3, len(remaining))]
    sides = ["top", "bot", "left", "right"]
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        outer_color = int(side_colors[side])
        inner_color = int(rng.choice(accent_palette))
        if inner_color == outer_color:
            return None
        arr, mask = _make_composite_sprite(side, outer_color, inner_color, rng)

        need_edge = {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]
        if _valid_rotation_count(arr, mask, need_edge, edge_len, outer_color) != 1:
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

    if rng.random() < 0.3 and len(remaining) >= 3:
        distractor_outer_choices = [c for c in remaining if c not in accent_palette]
        if distractor_outer_choices:
            d_outer = int(rng.choice(distractor_outer_choices))
            d_inner = int(rng.choice(accent_palette))
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
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
