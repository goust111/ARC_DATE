# Combo variant of 2c181942 (index 20)
# combo_spec: D3/0+D4/0
# Name: 八变换+多色复合精灵
#
# Design analysis
# ---------------
# fused_variant:
#   core_change: |
#     原题按“单色 + 4个旋转”处理。这里改成“双色复合精灵 + 8个二面体变换”：
#     每个精灵有外层匹配色和内层装饰色；先用外层色与 hub 边配对，再在
#     去重后的 8 种变换（4 旋转 + 镜像后的 4 旋转）中寻找唯一一个朝向：
#       - 外壳轮廓仍满足原题的接触轴对称；
#       - 接触边长度/颜色仍合法；
#       - 内层装饰落在该边对应的规范槽位。
#     最后整只贴到 hub，内层色保留。
#   concrete_trace: |
#     例：hub 左边是红色，输入里有一只“红壳+黄芯”的散落组件。原 solver 会用
#     `(g == red)` 抓红色连通块，只能得到壳，黄色芯会丢失；并且只试 4 个旋转。
#     新 solver 先把 hub 外全部前景按 8 连通切成整只组件，找到“包含红壳”的那只，
#     再枚举其 8 个唯一变换。只有一个变换同时满足：
#       1. 外壳轮廓沿接触轴对称；
#       2. 右边界正好有 2 个连续接触像素；
#       3. 这 2 个接触像素都是红色外层；
#       4. 黄色内芯回到该朝向规定的上侧槽位。
#     然后把整只红黄组件贴到 hub 左边，黄色芯保留在输出里。
#   complexity_calibration: |
#     若内层颜色完全自由，镜像后可能也合法，8 变换会产生歧义；若把整只双色
#     也做成完全对称，则镜像又会退化成纯旋转等价。这里采用折中设计：外壳保持
#     原题对称性，内层装饰故意偏到规范槽位。这样镜像是否发生会直接体现在装饰
#     的落点上，同时不会引入太多自由度。模板保持小尺寸、颜色数受控、内层颜色
#     在不同精灵间复用，便于人类在 2-4 个样例中看出：
#     “外层负责配边，内层跟着整只对象一起变换，但贴附后会回到固定槽位”。
#   hidden_assumptions_audit: |
#     原 `m = (g == col) & (~hub_mask)` 假设精灵整体单色；融合后失效，因为这样会把
#     内层色切掉。原 `for krot in range(4)` 假设无需镜像；融合后失效，因为输入里会
#     强制出现至少一只需要翻转才能回到贴附朝向的精灵。原代码默认“首个合法旋转即可”，
#     在 8 变换空间里也不够稳健，所以现在对每个候选组件都统计唯一合法变换，只有
#     恰好 1 个唯一朝向合法时才接收该组件。
#   main_assumptions_and_generator_guarantees: |
#     main() 的新假设：
#       - 每只匹配精灵是单个 8 连通前景组件；
#       - hub 边色不会被拿来当其他组件的内层色或干扰物外层色；
#       - 贴附朝向下，外壳轮廓沿接触轴对称；
#       - 对目标边而言，去重后的二面体变换里恰有 1 个合法朝向。
#     generate_input() 的保障：
#       - 所有匹配/干扰物都由同一小模板族生成，保持 8 连通且双色可见；
#       - 内层色只从非 hub 色的小调色板中取，避免误匹配；
#       - output-first 放置保证贴附后不越界、不互相重叠；
#       - 散放时 `padding=2`，保证组件彼此和 hub 分离；
#       - 每题至少一只匹配精灵以“非纯旋转等价”的散落姿态出现，避免退化为 D4-only。
#   visual_observability_and_cleanliness: |
#     人眼能直接看到三点：一是散落精灵与贴附精灵都是双色整只对象，而不是只剩外壳；
#     二是至少有一只输入精灵的朝向不能靠单纯旋转解释，必须考虑镜像；
#     三是装饰色在贴附后会回到固定槽位，说明变换作用于整只对象。模板尺寸小且对象之间
#     留白充足，画面不会因为多色和 8 变换而显得杂乱。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


BG = 8
HUB_N = 4
EDGE_LEN = HUB_N - 2
SIDES = ("top", "bot", "left", "right")


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


# Canonical frame = sprite attached to the LEFT side of the hub:
# - contact edge is the rightmost column
# - the shell mask is flipud-symmetric
# - the single accent pixel sits in a canonical off-axis slot
# - not fliplr-symmetric, so the opposite contact side is distinguishable
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


def _variant_key(arr: np.ndarray, mask: np.ndarray) -> tuple[tuple[int, int], bytes, bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes(), mask.tobytes()


def _unique_dihedral_variants(
    arr: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, bool, int]]:
    variants: list[tuple[np.ndarray, np.ndarray, bool, int]] = []
    seen: set[tuple[tuple[int, int], bytes, bytes]] = set()
    for flipped in (False, True):
        base_arr = np.fliplr(arr) if flipped else arr
        base_mask = np.fliplr(mask) if flipped else mask
        for krot in range(4):
            cand_arr = np.rot90(base_arr, int(krot))
            cand_mask = np.rot90(base_mask, int(krot))
            cand_arr, cand_mask = _crop_to_mask(cand_arr, cand_mask)
            key = _variant_key(cand_arr, cand_mask)
            if key in seen:
                continue
            seen.add(key)
            variants.append((cand_arr, cand_mask, bool(flipped), int(krot)))
    return variants


def _rotation_keys(arr: np.ndarray, mask: np.ndarray) -> set[tuple[tuple[int, int], bytes, bytes]]:
    keys: set[tuple[tuple[int, int], bytes, bytes]] = set()
    for krot in range(4):
        cand_arr = np.rot90(arr, int(krot))
        cand_mask = np.rot90(mask, int(krot))
        cand_arr, cand_mask = _crop_to_mask(cand_arr, cand_mask)
        keys.add(_variant_key(cand_arr, cand_mask))
    return keys


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
    for n in range(3, 7):
        for r in range(0, H - n + 1):
            for c in range(0, W - n + 1):
                if (
                    int(g[r, c]) != int(bg)
                    or int(g[r, c + n - 1]) != int(bg)
                    or int(g[r + n - 1, c]) != int(bg)
                    or int(g[r + n - 1, c + n - 1]) != int(bg)
                ):
                    continue
                inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[r, c + 1 : c + n - 1]
                bot = g[r + n - 1, c + 1 : c + n - 1]
                left = g[r + 1 : r + n - 1, c]
                right = g[r + 1 : r + n - 1, c + n - 1]
                if (
                    bool((top == int(bg)).any())
                    or bool((bot == int(bg)).any())
                    or bool((left == int(bg)).any())
                    or bool((right == int(bg)).any())
                ):
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
    fg = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(fg, connectivity=8)
    out = []
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = g[comp_mask]
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        out.append((int(cid), arr, comp_mask))
    return out


def _attachment_info(
    arr: np.ndarray, mask: np.ndarray, need_edge: str, target_len: int, outer_color: int
):
    # Symmetry stays on the shell silhouette, while the inner accent acts as a
    # visible canonical-slot cue that breaks the D4 mirror ambiguity.
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
        if any(int(arr[int(rr), int(edge_c)]) != int(outer_color) for rr in rows):
            return None
    else:
        _typ, edge_r, cols = seg
        if len(cols) != int(target_len) or not _is_contig(cols):
            return None
        if any(int(arr[int(edge_r), int(cc)]) != int(outer_color) for cc in cols):
            return None

    colors = {int(v) for v in arr[mask]}
    if len(colors) < 2:
        return None
    if not _accent_position_ok(arr, mask, need_edge, outer_color):
        return None
    return seg


def _accent_position_ok(arr: np.ndarray, mask: np.ndarray, need_edge: str, outer_color: int) -> bool:
    pts = np.argwhere(mask & (arr != int(outer_color)))
    if pts.shape[0] != 1:
        return False
    rr, cc = map(int, pts[0])
    h, w = map(int, arr.shape)
    if need_edge in ("left", "right"):
        return int(rr) < int(h // 2)
    if need_edge == "bot":
        return int(cc) >= int(w // 2)
    return int(cc) < int(w // 2)


def _placeable_transform(
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
    valid = []
    for cand_arr, cand_mask, _flipped, _krot in _unique_dihedral_variants(arr, mask):
        seg = _attachment_info(cand_arr, cand_mask, need_edge, target_len, outer_color)
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

        placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(cand_mask)}
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in placed):
            continue

        ok = True
        for rr, cc in placed:
            val = int(cand_arr[int(rr - dr), int(cc - dc)])
            if val == int(bg):
                continue
            if int(out[int(rr), int(cc)]) != int(bg) and int(out[int(rr), int(cc)]) != val:
                ok = False
                break
        if ok:
            valid.append((cand_arr, cand_mask, int(dr), int(dc)))

    if len(valid) != 1:
        return None
    return valid[0]


def _contact_side_for_hub_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _orient_from_left(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, -1)
    return np.rot90(arr, 1)


def _make_composite_sprite(
    side: str, outer_color: int, inner_color: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    template_mask, base_inner = rng.choice(_CANONICAL_TEMPLATES)

    left_arr = np.full(template_mask.shape, int(outer_color), dtype=int)
    left_arr[~template_mask] = -1
    rr, cc = base_inner[0]
    left_arr[int(rr), int(cc)] = int(inner_color)

    arr = _orient_from_left(left_arr, side)
    mask = arr != -1
    arr = np.where(mask, arr, 0).astype(int)
    arr, mask = _crop_to_mask(arr, mask)
    return arr, mask


def _valid_transform_count(
    arr: np.ndarray, mask: np.ndarray, need_edge: str, target_len: int, outer_color: int
) -> int:
    count = 0
    for cand_arr, cand_mask, _flipped, _krot in _unique_dihedral_variants(arr, mask):
        if _attachment_info(cand_arr, cand_mask, need_edge, target_len, outer_color) is not None:
            count += 1
    return int(count)


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


def _non_rotation_variant_choices(
    arr: np.ndarray, mask: np.ndarray
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    attached_key = _variant_key(arr, mask)
    rot_keys = _rotation_keys(arr, mask)
    all_choices: list[tuple[np.ndarray, np.ndarray]] = []
    mirrored_only: list[tuple[np.ndarray, np.ndarray]] = []
    for cand_arr, cand_mask, _flipped, _krot in _unique_dihedral_variants(arr, mask):
        key = _variant_key(cand_arr, cand_mask)
        if key == attached_key:
            continue
        choice = (cand_arr, cand_mask)
        all_choices.append(choice)
        if key not in rot_keys:
            mirrored_only.append(choice)
    return all_choices, mirrored_only


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

            placed = _placeable_transform(
                arr, mask, need_edge, fixed, span, target, outer_color, out, bg
            )
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
    bg = BG
    n = HUB_N
    edge_len = EDGE_LEN

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != int(bg)]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    edge_colors = [top_col, bot_col, left_col, right_col]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    remaining = [c for c in all_colors if c not in edge_colors]
    if len(remaining) < 3:
        return None
    accent_palette = remaining[:2]
    distractor_outer_pool = [c for c in remaining if c not in accent_palette]
    if not distractor_outer_pool:
        return None

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), K)

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    scatter_choices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    mirrored_choices: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for side in sprite_sides:
        outer_color = int(side_colors[side])
        inner_color = int(rng.choice(accent_palette))
        arr, mask = _make_composite_sprite(side, outer_color, inner_color, rng)

        need_edge = _contact_side_for_hub_side(side)
        if _valid_transform_count(arr, mask, need_edge, edge_len, outer_color) != 1:
            return None

        all_variants, mirrored_only = _non_rotation_variant_choices(arr, mask)
        if not all_variants:
            return None

        sprite_defs[side] = (arr, mask)
        scatter_choices[side] = all_variants
        mirrored_choices[side] = mirrored_only

    force_reflect_sides = [side for side in sprite_sides if mirrored_choices[side]]
    if not force_reflect_sides:
        return None
    forced_side = rng.choice(force_reflect_sides)

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
        if side == forced_side:
            cand_arr, cand_mask = rng.choice(mirrored_choices[side])
        else:
            cand_arr, cand_mask = rng.choice(scatter_choices[side])
        scatter = np.where(cand_mask, cand_arr, bg).astype(int)
        try:
            pr, pc = random_free_location(
                canvas, scatter, transparent=bg, padding=2, padding_connectivity=8
            )
        except ValueError:
            return None
        paste(canvas, scatter, pr, pc, transparent=bg)
        paste(inp, scatter, pr, pc, transparent=bg)

    if rng.random() < 0.3 and distractor_outer_pool:
        d_outer = int(rng.choice(distractor_outer_pool))
        d_inner_choices = [c for c in accent_palette if c != int(d_outer)]
        if d_inner_choices:
            d_inner = int(rng.choice(d_inner_choices))
            d_side = rng.choice(list(SIDES))
            dist_arr, dist_mask = _make_composite_sprite(d_side, d_outer, d_inner, rng)
            dist_variants, _ = _non_rotation_variant_choices(dist_arr, dist_mask)
            if dist_variants:
                cand_arr, cand_mask = rng.choice(dist_variants)
                dist_patch = np.where(cand_mask, cand_arr, bg).astype(int)
                try:
                    pr, pc = random_free_location(
                        canvas, dist_patch, transparent=bg, padding=2, padding_connectivity=8
                    )
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
