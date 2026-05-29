# Variant of c7f57c3e
# Dimension: D12 — 颜色模式的定义粒度 (meta)
# Change: 颜色模式按精灵使用的颜色组合整体定义（如 teal+green 为一组模式）... → 按主色/辅色层级定义模式

"""
Design analysis for the D12 variant.

a) 具体化与行为追踪
   - 仍然只有两类小 prototype，结构关系仍是上下翻转；每个精灵仍由 bg、锚点色 1、以及两个非 1 颜色组成。
   - 新变化在于这两个非 1 颜色不再被当成“一个无序颜色组合”，而是按像素数分成主色/辅色：
     主色区域面积严格更大，辅色区域面积更小。
   - 我把 family A 设计成“主色块在上、辅色块在下”的结构，family B 是其上下翻转，因此恰好变成
     “辅色块在上、主色块在下”。这样 family 身份可由几何位置稳定判定，而不是靠颜色 id。
   - 具体追踪一个 4x4 例子：
     输入 A: 上部大块是红色，底部小块是绿色。
     原题输出 B: 整个精灵切换成 B 的完整配色，例如上部小块紫色、下部大块黄色。
     变体输出 B: 只切换主色，保留辅色，因此变成 B 的结构，但下部大块改成 B-family 的主色，
     顶部小块仍保留输入 A 的辅色。反方向的 B→A 则只切换辅色、保留主色。
   - 结果是：两个 family 仍然交换结构，但颜色变化变成“角色级拼接”，而不是整套换色。

b) 复杂度校准
   - 为避免歧义，每个样例仍只保留两个 structural family，不引入更多类别。
   - 为了让主/辅色可见且稳定，generator 强制主色像素数 > 辅色像素数，并使用 4 个互不相同的非背景颜色。
   - 这样既不会退化为原题的整组换色，也不会因为颜色共享导致“到底保留了哪个角色”难以看出。

c) 原 solver 的隐式假设审查
   - 原代码用最小精灵的整张数组作为 prototype key，这依赖“类别 = 完整颜色图样”。
     变体里这不再充分，因为规则依赖颜色在 sprite 内的主/辅角色，而不是颜色对本身。
   - 原代码按 exact-color overlap 给大精灵分类，这也依赖输入和 prototype 的整图配色一致。
     现在必须先把每个 sprite 规范化成 role map: 0/bg, 1/anchor, 2/dominant, 3/secondary。
   - 原代码的 repl_idx = 1 - best_idx 仍然成立，因为结构 family 依旧只有两个且互为翻转。
     真正变化的是 replacement patch 的着色方式：不能再直接拿“另一个 prototype 的整图颜色”，
     而要按角色分别决定哪一层保留、哪一层导入。
   - 原代码默认 prototype 顺序无关紧要；这里不行，因为 A/B 的角色拼接方向不对称。
     因此 solver 要按“主色块在辅色块上方”的可见几何关系给两个 prototype 排序。

d) 放置合法性
   - 仍用原题的 anchor 对齐方式来计算输入 patch 与输出 patch 的联合占位框。
   - generator 放置的是联合 bbox marker，而不是只按输入或输出单独占位，因此两者都不会越界，
     也不会与其他 sprite 的输入/输出位置重叠。
   - family A/B 共享同一 bbox 尺寸，且 anchor bbox 尺寸相同，所以缩放与对齐逻辑保持合法。

e) 视觉可观察性
   - 规则的可观察核心是：结构照旧翻转，但只有一层颜色发生“跨 family 迁移”，另一层留在原 sprite。
   - family A 和 B 的“主色在上 / 主色在下”是直接可见的，因此“哪一类保留主色、哪一类保留辅色”
     可以从少量样例中被肉眼归纳，而不需要读颜色编号。
   - 不使用颜色 id 排序，不根据黑色/0 做判断；只根据像素数量和上下位置这两个可见属性。

f) 可视化整洁度
   - prototype 仍是单个连通精灵，内部只有少量 bg 洞。
   - 对象之间保持 padding>=2，避免不同 sprite 视觉粘连。
   - 没有额外标记层，也没有多个系统叠加在同一区域，输出保持和原题同样整洁。

g) 颜色变化可推断性
   - 每题只涉及 4 个非背景颜色，分属 A 主、A 辅、B 主、B 辅四个角色。
   - 输出统一体现“一个角色来自 A、另一个角色来自 B”的混合，这给出足够强的结构线索：
     同一结构翻过去后，大块颜色和小块颜色不是一起变，而是分角色来源。

9) 新 solver 的隐式假设审查
   - 假设 1：每个 sprite 恰好有两个非 bg 非 1 颜色，且主色计数严格大于辅色计数。
     generator 通过固定模板保证这一点。
   - 假设 2：两个最小 sprite 足以恢复两个 family，且一个主色块在上、另一个主色块在下。
     generator 强制至少各放一个 1x prototype，并让 B = flipud(A)。
   - 假设 3：同一题内所有实例都来自这两个 family 的整数倍缩放。
     generator 保持原题的 scale 规则，只放 1x/2x/3x 的最近邻拷贝。
"""

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    """Pick the bbox of color-1 anchor pixels; they are guaranteed to exist."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _extract_role_map(arr: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    """Normalize a sprite to roles: 0=bg, 1=anchor, 2=dominant color, 3=secondary color."""
    vals, counts = np.unique(arr[(arr != bg) & (arr != 1)], return_counts=True)
    assert len(vals) == 2, "each sprite must have exactly two non-bg non-anchor colors"
    order = np.argsort(-counts)
    dom_color = int(vals[order[0]])
    sec_color = int(vals[order[1]])
    dom_count = int(counts[order[0]])
    sec_count = int(counts[order[1]])
    assert dom_count > sec_count, "dominant color must have strictly larger area than secondary"

    role_map = np.zeros_like(arr, dtype=int)
    role_map[arr == 1] = 1
    role_map[arr == dom_color] = 2
    role_map[arr == sec_color] = 3
    return role_map, dom_color, sec_color


def _colorize_role_map(role_map: np.ndarray, bg: int, dominant: int, secondary: int) -> np.ndarray:
    out = np.full(role_map.shape, bg, dtype=int)
    out[role_map == 1] = 1
    out[role_map == 2] = dominant
    out[role_map == 3] = secondary
    return out


def _dominant_above_secondary(role_map: np.ndarray) -> bool:
    dom_rows = np.where(role_map == 2)[0]
    sec_rows = np.where(role_map == 3)[0]
    assert dom_rows.size > 0 and sec_rows.size > 0
    return float(dom_rows.mean()) < float(sec_rows.mean())


def _order_families(proto_infos: list[dict]) -> list[dict]:
    """family A has its dominant region above the secondary region; family B is the opposite."""
    assert len(proto_infos) == 2
    flags = [_dominant_above_secondary(info["role_map"]) for info in proto_infos]
    assert flags.count(True) == 1, "exactly one prototype must have dominant-above-secondary ordering"
    a_idx = flags.index(True)
    b_idx = 1 - a_idx
    return [proto_infos[a_idx], proto_infos[b_idx]]


def _scaled_overlap_score(sprite_roles: np.ndarray, proto_roles: np.ndarray, scale: int) -> int:
    scaled = _scale_nn(proto_roles, scale)
    if scaled.shape != sprite_roles.shape:
        return -1
    return int(np.sum(scaled == sprite_roles))


def _place_patch_aligned(
    out: np.ndarray,
    patch: np.ndarray,
    bbox: tuple[int, int, int, int],
    inst_anchor: tuple[int, int, int, int],
    proto_anchor: tuple[int, int, int, int],
    scale: int,
    bg: int,
) -> None:
    h, w = out.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = inst_anchor
    pr0, pc0, _pr1, _pc1 = proto_anchor
    top = (r0 + ar0) - pr0 * scale
    left = (c0 + ac0) - pc0 * scale

    ph, pw = patch.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(h, top + ph)
    cc1 = min(w, left + pw)
    sub_p = patch[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
    m = sub_p != bg
    out_slice = out[rr0:rr1, cc0:cc1]
    out_slice[m] = sub_p[m]
    out[rr0:rr1, cc0:cc1] = out_slice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite structure, but only exchange one color role.

    Input structure:
    - The background is the most common color. Two prototype families appear as scaled sprites.
    - Every sprite contains color-1 anchor pixels plus exactly two other colors with unequal areas, so each sprite has a dominant color and a secondary color.
    - The two smallest distinct sprites reveal the two family role-maps; one family has its dominant block above its secondary block, and the other is its vertical counterpart.

    Rule:
    - Recover the two prototype families, normalize them into role maps, and classify every sprite by structure and scale.
    - Family A sprites change to family B structure while keeping their own secondary color and importing family B's dominant color.
    - Family B sprites change to family A structure while keeping their own dominant color and importing family A's secondary color.
    - Preserve each sprite's scale and anchor alignment during the replacement.

    Key insight:
    - The relevant pattern is no longer the unordered pair of colors used by a sprite.
    - Instead, colors must be understood by role inside the shape: which color is the large dominant layer and which is the small secondary layer.
    - After that normalization, the transformation is a structural flip plus a partial role-wise color transplant.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        role_map, dom_color, sec_color = _extract_role_map(arr, bg)
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
                "role_map": role_map,
                "dom_color": dom_color,
                "sec_color": sec_color,
            }
        )

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base_candidates = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], dict] = {}
    for s in base_candidates:
        key = tuple(s["role_map"].ravel().tolist())
        if key not in uniq:
            uniq[key] = {
                "role_map": s["role_map"],
                "dom_color": s["dom_color"],
                "sec_color": s["sec_color"],
                "anchor_bbox": _pick_anchor_bbox(s["arr"], bg),
            }
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototype families"

    families = _order_families(list(uniq.values()))
    base_anchor = families[0]["anchor_bbox"]
    base_anc_h = base_anchor[2] - base_anchor[0] + 1
    base_anc_w = base_anchor[3] - base_anchor[1] + 1
    assert families[1]["anchor_bbox"][2] - families[1]["anchor_bbox"][0] + 1 == base_anc_h
    assert families[1]["anchor_bbox"][3] - families[1]["anchor_bbox"][1] + 1 == base_anc_w

    out = g.copy()

    for s in sprites:
        bbox = s["bbox"]
        arr = s["arr"]
        role_map = s["role_map"]
        inst_anchor = _pick_anchor_bbox(arr, bg)
        inst_anc_h = inst_anchor[2] - inst_anchor[0] + 1
        inst_anc_w = inst_anchor[3] - inst_anchor[1] + 1
        assert inst_anc_h % base_anc_h == 0 and inst_anc_w % base_anc_w == 0
        scale_h = inst_anc_h // base_anc_h
        scale_w = inst_anc_w // base_anc_w
        assert scale_h == scale_w and scale_h >= 1
        scale = int(scale_h)

        best_idx = 0
        best_score = -1
        for idx, fam in enumerate(families):
            score = _scaled_overlap_score(role_map, fam["role_map"], scale)
            if score > best_score:
                best_score = score
                best_idx = idx

        target_idx = 1 - best_idx
        target_family = families[target_idx]

        if best_idx == 0:
            out_dom = int(target_family["dom_color"])
            out_sec = int(s["sec_color"])
        else:
            out_dom = int(s["dom_color"])
            out_sec = int(families[0]["sec_color"])

        replacement = _colorize_role_map(target_family["role_map"], bg, out_dom, out_sec)
        replacement = _scale_nn(replacement, scale)

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        _place_patch_aligned(
            out,
            replacement,
            bbox,
            inst_anchor,
            target_family["anchor_bbox"],
            scale,
            bg,
        )

    return out


# --------------- generator (o5x) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
ROLE_TEMPLATES = [
    np.array(
        [
            [0, 1, 1, 0],
            [2, 1, 1, 2],
            [2, 2, 2, 2],
            [0, 3, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [2, 1, 2],
            [2, 2, 2],
            [0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 0],
            [2, 2, 2, 2],
            [0, 3, 3, 0],
        ],
        dtype=int,
    ),
]

# --- CFG: sprite instances ---
SPRITE_COUNTS = [3, 4]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 300


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Create family A (dominant above secondary) and family B (its vertical flip)."""
    labels_a = random.choice(ROLE_TEMPLATES)
    labels_b = np.flipud(labels_a)

    pool = [c for c in range(10) if c not in (bg, 1)]
    a_dom, a_sec, b_dom, b_sec = random.sample(pool, 4)

    def _paint(labels: np.ndarray, dom: int, sec: int) -> np.ndarray:
        proto = np.full(labels.shape, bg, dtype=int)
        proto[labels == 1] = 1
        proto[labels == 2] = dom
        proto[labels == 3] = sec
        return proto

    return _paint(labels_a, a_dom, a_sec), _paint(labels_b, b_dom, b_sec)


def _hybrid_output_for_input(proto_a: np.ndarray, proto_b: np.ndarray, bg: int, input_type: int) -> np.ndarray:
    """family A -> family B while keeping secondary; family B -> family A while keeping dominant."""
    roles_a, _a_dom, a_sec = _extract_role_map(proto_a, bg)
    roles_b, b_dom, _b_sec = _extract_role_map(proto_b, bg)
    if input_type == 0:
        return _colorize_role_map(roles_b, bg, b_dom, a_sec)
    return _colorize_role_map(roles_a, bg, b_dom, a_sec)


def _union_bbox(patch_a: np.ndarray, patch_b: np.ndarray) -> tuple[int, int, int, int]:
    """Compute (dr, dc, union_h, union_w) for anchor-aligned placement of two patches."""
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
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for input_type, scale in instances:
        input_patch = _scale_nn(protos[input_type], scale)
        output_patch = _scale_nn(_hybrid_output_for_input(proto_a, proto_b, bg, input_type), scale)
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
        paste(inp, input_patch, in_r, in_c, transparent=bg)
        paste(out, output_patch, out_r, out_c, transparent=bg)

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
