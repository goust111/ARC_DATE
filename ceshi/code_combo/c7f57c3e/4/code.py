# Combo variant of c7f57c3e (index 4)
# combo_spec: D4/1+D12/2
# Name: 空心框架精灵 + 按主/辅色层级交换
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 每个对象不再是普通多色小精灵，而是单连通的空心框架：外圈是 frame，内部大部分是 bg 空洞，
#      一条 color-1 spine 从框架边缘伸进内部，内部还有一小块 fill，并通过 spine 与框架连通。
#    - 这里把 D12 的主/辅色层级直接绑定到 D4 的几何层：frame 像素数严格更多，所以是主色；fill 更小，所以是辅色。
#    - family A 使用“spine 从上往下伸”的模板；family B 是其上下翻转，因此 spine 从下往上伸。
#    - 对一个 A 对象：原题会整套换成 B 的 frame+fill 颜色；融合后只换主色，所以仍翻到 B 结构，
#      但保留原来的 fill，改用 B-family 的 frame。
#    - 对一个 B 对象：仍翻到 A 结构，但这次只换辅色，保留自己的 frame，fill 改成 A-family 的 fill。
#    - 因此输出不是整套换色，而是“结构切换 + frame/fill 分层拼接”。
# b) 复杂度校准
#    - 仍只保留两个 family、3-4 个实例、且只有一个放大对象，避免组合规则过散。
#    - 使用少量固定框架模板，保证人类能稳定识别“空心框架 + 上/下 spine”这两个结构族。
#    - 四个 family 颜色全部互异，避免颜色共享把“到底保留了 frame 还是 fill”变得不明显。
# c) 隐式假设审查
#    - 原题按整张彩色数组区分类别；这里不再成立，所以 solver 先把 sprite 归一化成 role map:
#      0=bg, 1=anchor spine, 2=frame(主色), 3=fill(辅色)。
#    - 原题按 exact-color overlap 分类；这里改成按 role-map overlap 分类，否则混合换色后的 family 无法稳定识别。
#    - components_mask(g != bg) 要求每个对象单连通；因此 fill 必须通过 spine 接回框架，不能孤立漂在空洞里。
#    - 缩放仍依赖两个 family 的 anchor bbox 尺寸一致；所以 B 只能由 A 的标签模板做 flipud，不改 spine 尺寸。
#    - family 顺序现在不能任意，因为“A->B 保留 fill / B->A 保留 frame”是有方向的；
#      这里用 spine 相对 fill 在上还是在下这个可见关系给 family 排序。
# d) 放置合法性
#    - generator 继续用 output-first：先确定想要的输出补丁，再反推输入补丁。
#    - 占位时使用 anchor 对齐后的输入/输出联合 bbox，因此替换前后都不会越界，也不会与其他对象重叠。
#    - 网格尺寸下界随最大缩放对象调整，确保 5x6 模板放大到 3 倍后仍能留出边距。
# e) 视觉可观察性
#    - 人类能直接看到两件事：框架是空心的；spine 有时在上、有时在下。
#    - 颜色变化也有明确结构线索：A-family 输出保留小的 interior fill，B-family 输出保留大的 outer frame。
#    - 规则不依赖颜色编号大小，不使用“黑色=0”的排序，只依赖面积层级和几何位置。
# f) 可视化整洁度
#    - 框架内部保留大空洞，对象之间 padding>=2，避免不同实例粘在一起。
#    - 只有 frame、spine、fill 三层可见信息，没有额外标记，输出仍保持干净。
# g) 变化的可推断性
#    - 每题只有四个非背景角色颜色：A-frame、A-fill、B-frame、B-fill。
#    - 输出统一混成“B-frame + A-fill”，但结构仍在 A/B 两族之间切换；这样样例里既能看出颜色层级，
#      又不会退化为无提示的大规模颜色重映射。
# 9) 新 solver 的隐式假设审查
#    - 每个 sprite 恰有两个非 bg 非 1 颜色，且 frame 像素数严格大于 fill；generator 用固定模板保证。
#    - 最小面积对象中必须同时出现两个 family；generator 强制至少放一个 A/1x 和一个 B/1x。
#    - 所有实例都来自这两个 role-template 的整数倍缩放；generator 只产生 1x/2x/3x 的最近邻副本。

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
    """Return the bbox of the color-1 spine; every valid sprite contains one."""
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "frame sprite has no color-1 spine"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _extract_role_map(arr: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    """Normalize to 0=bg, 1=anchor, 2=frame(main), 3=fill(secondary)."""
    vals, counts = np.unique(arr[(arr != bg) & (arr != 1)], return_counts=True)
    assert len(vals) == 2, "each sprite must have exactly two non-bg non-anchor colors"
    order = np.argsort(-counts)
    frame_color = int(vals[order[0]])
    fill_color = int(vals[order[1]])
    frame_count = int(counts[order[0]])
    fill_count = int(counts[order[1]])
    assert frame_count > fill_count, "frame/main color must occupy more cells than fill/secondary color"

    role_map = np.zeros_like(arr, dtype=int)
    role_map[arr == 1] = 1
    role_map[arr == frame_color] = 2
    role_map[arr == fill_color] = 3
    return role_map, frame_color, fill_color


def _colorize_role_map(role_map: np.ndarray, bg: int, frame_color: int, fill_color: int) -> np.ndarray:
    out = np.full(role_map.shape, bg, dtype=int)
    out[role_map == 1] = 1
    out[role_map == 2] = frame_color
    out[role_map == 3] = fill_color
    return out


def _anchor_above_fill(role_map: np.ndarray) -> bool:
    anchor_rows = np.where(role_map == 1)[0]
    fill_rows = np.where(role_map == 3)[0]
    assert anchor_rows.size > 0 and fill_rows.size > 0
    return float(anchor_rows.mean()) < float(fill_rows.mean())


def _order_families(proto_infos: list[dict]) -> list[dict]:
    """Family A has its spine above the fill; family B is the vertically flipped counterpart."""
    assert len(proto_infos) == 2
    flags = [_anchor_above_fill(info["role_map"]) for info in proto_infos]
    assert flags.count(True) == 1, "exactly one prototype must have spine-above-fill ordering"
    a_idx = flags.index(True)
    return [proto_infos[a_idx], proto_infos[1 - a_idx]]


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
    Swap each hollow-frame sprite to the opposite spine orientation while only exchanging one color layer.

    Input structure:
    - The background is the most common color. Each non-bg component is a connected hollow-frame sprite with a visible cavity.
    - Every sprite contains color-1 spine pixels plus exactly two other colors: a larger frame color around the border and a smaller interior fill color.
    - The two smallest distinct sprites reveal the two frame families: one has the spine above the fill, and the other is its vertical counterpart with the spine below.

    Rule:
    - Recover the two family role-maps from the smallest sprites, normalize every sprite into frame/fill roles, and infer its integer scale from the spine bbox.
    - A spine-above sprite changes to the spine-below structure while keeping its own fill color and importing the other family's frame color.
    - A spine-below sprite changes to the spine-above structure while keeping its own frame color and importing the other family's fill color.
    - Preserve scale and anchor alignment when placing the replacement.

    Key insight:
    - The hollow frame naturally separates the two non-anchor colors into outer frame and inner fill, so the relevant pattern is role-based rather than the unordered color pair.
    - Once the sprite is normalized to frame/fill roles and its spine orientation is known, the output is just the opposite structure with one role preserved and the other transplanted.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least 2 frame sprites"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        role_map, frame_color, fill_color = _extract_role_map(arr, bg)
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
                "role_map": role_map,
                "frame_color": frame_color,
                "fill_color": fill_color,
                "anchor_bbox": _pick_anchor_bbox(arr, bg),
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
                "frame_color": s["frame_color"],
                "fill_color": s["fill_color"],
                "anchor_bbox": s["anchor_bbox"],
            }
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct frame families"

    families = _order_families(list(uniq.values()))
    base_anchor = families[0]["anchor_bbox"]
    base_anc_h = base_anchor[2] - base_anchor[0] + 1
    base_anc_w = base_anchor[3] - base_anchor[1] + 1
    other_anchor = families[1]["anchor_bbox"]
    assert other_anchor[2] - other_anchor[0] + 1 == base_anc_h
    assert other_anchor[3] - other_anchor[1] + 1 == base_anc_w

    out = g.copy()

    for s in sprites:
        role_map = s["role_map"]
        inst_anchor = s["anchor_bbox"]
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
            out_frame = int(target_family["frame_color"])
            out_fill = int(s["fill_color"])
        else:
            out_frame = int(s["frame_color"])
            out_fill = int(families[0]["fill_color"])

        replacement = _colorize_role_map(target_family["role_map"], bg, out_frame, out_fill)
        replacement = _scale_nn(replacement, scale)

        for rr, cc in s["pts"]:
            out[rr, cc] = bg

        _place_patch_aligned(
            out,
            replacement,
            s["bbox"],
            inst_anchor,
            target_family["anchor_bbox"],
            scale,
            bg,
        )

    return out


# --------------- generator (output-first hollow-frame role swap) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 21
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]

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
#   2 = frame color (main)
#   3 = interior fill color (secondary)
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


def _colorize_frame(labels: np.ndarray, bg: int, frame_color: int, fill_color: int) -> np.ndarray:
    proto = np.full(labels.shape, bg, dtype=int)
    proto[labels == 1] = 1
    proto[labels == 2] = frame_color
    proto[labels == 3] = fill_color
    return proto


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels_a = random.choice(FRAME_TEMPLATES)
    labels_b = np.flipud(labels_a)

    pool = [c for c in range(10) if c not in (bg, 1)]
    a_frame, a_fill, b_frame, b_fill = random.sample(pool, 4)

    proto_a = _colorize_frame(labels_a, bg, a_frame, a_fill)
    proto_b = _colorize_frame(labels_b, bg, b_frame, b_fill)
    return proto_a, proto_b


def _hybrid_output_for_input(proto_a: np.ndarray, proto_b: np.ndarray, bg: int, input_type: int) -> np.ndarray:
    """A->B keeps fill; B->A keeps frame."""
    roles_a, _a_frame, a_fill = _extract_role_map(proto_a, bg)
    roles_b, b_frame, _b_fill = _extract_role_map(proto_b, bg)
    if input_type == 0:
        return _colorize_role_map(roles_b, bg, b_frame, a_fill)
    return _colorize_role_map(roles_a, bg, b_frame, a_fill)


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
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
