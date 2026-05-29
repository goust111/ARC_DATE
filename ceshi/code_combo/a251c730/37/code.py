# Combo variant of a251c730 (index 37)
# combo_spec: D3/3+D4/0+D15/1
# Name: 方向标记+旋转+标记形状编码变换

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# ---------------
# fused_variant:
#   a) concretization_and_trace:
#     - 原题目标区只有“同色单点 marker”；这里每个 marker 变成一个 4 连通小组件：
#       “单色符号 + 1 个边框色方向像素”。
#     - 颜色仍然只负责匹配哪一类源精灵。
#     - 符号形状负责基础变换：
#       - 水平线 -> identity
#       - 竖线 -> rot90
#       - L 形 -> flip_h
#     - marker 的定位点固定为“符号彩色部分 bbox 的左上角像素”。
#     - 再叠加两层旋转：
#       - 目标定位点所在象限给出一层 quarter-turn（UL/UR/LR/LL -> 0/1/2/3）
#       - 旁侧方向像素再给出一层 quarter-turn（上/右/下/左 -> 0/1/2/3）
#     - 行为追踪例子：
#       - 原题：红色单点 -> 找红锚点精灵 -> 原样平移，使锚点落在该点。
#       - 本题：红色 L 形 + 右侧方向像素，且定位点位于右下象限
#         -> 先对红精灵做 flip_h
#         -> 再按“右下象限 + 右侧方向 = 2 + 1 = 3”做 270 度顺时针旋转
#         -> 最后把变换后的锚点对齐到符号定位点。
#   b) complexity_calibration:
#     - 形状只保留 3 档，方向只保留 4 个轴向，象限只保留 4 档；自由度有上界。
#     - 每个样例固定放 3 个 marker，并强制 3 种形状都出现一次，保证 D15 可见。
#     - 3 个 marker 使用 3 个不同象限、3 个不同方向，保证 D3 和 D4 都可见。
#     - 精灵库限制为对旋转/翻转都明显不对称的形状，避免组合后视觉退化。
#   c) hidden_assumption_audit:
#     - 原版 main 依赖“目标组件都是单点”，这里失效；必须按连通组件解析 marker。
#     - 原版把 marker 像素本身当对齐点；这里改成“彩色符号 bbox 左上角”作为定位点。
#     - 原版只做平移；这里必须同步变换 patch、mask、anchor 坐标。
#     - 原版目标区可用 `max_sz == 1` 识别；这里目标组件大小为 3 或 4，需要新的识别准则。
#     - 原版没有方向/象限耦合；这里 solver 假设总旋转 = 象限旋转 + 方向旋转，generator 同步保证。
#   d) placement_legality:
#     - 生成器仍然 output-first：先把最终变换后的精灵放到输出，再反推 marker。
#     - 只有当“精灵完整落入目标内”“marker 组件完整落入目标内”“marker 之间不接触”时才接受。
#     - 方向像素可能位于符号上方或左侧，因此 marker patch 会自动归一化并记录定位点偏移。
#   e) visual_observability:
#     - 同色重复 marker 会在不同形状/方向/象限下产出不同姿态，规则可直接从样例对比中看出。
#     - 三个控制维度都来自可见几何：颜色、符号形状、方向像素位置、目标中的空间位置。
#     - 不依赖颜色 id 大小、放置顺序或任何人类不可见的排序。
#   f) cleanliness:
#     - 输入目标区只放小型指令符号，不叠加整只精灵，避免与输出层混淆。
#     - 方向像素统一使用目标边框色，因此“符号主体”和“方向提示”视觉角色稳定。
#   g) inferability_guard:
#     - 样例中三种形状总会同时出现；方向和象限各自也至少出现 3 个不同状态。
#     - 这避免 generator 退化到只体现颜色匹配、或只体现其中一层旋转。
#   h) new_solver_assumptions_checked_against_generator:
#     - 每个源精灵都有唯一最稀有锚点色，且每种目标颜色都在源区至少出现一次。
#     - 每个目标 marker 恰有一个方向像素，且其余像素同色并来自固定 3 个符号之一。
#     - 定位点永远是符号彩色部分 bbox 左上角；generator 使用同一定位定义。
#     - 目标定位点不落在中心分界线上，因此象限判定不会歧义。


DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_QTURNS = {
    (-1, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (0, -1): 3,
}

MARKER_SHAPES: dict[str, tuple[str, ...]] = {
    "identity": ("MM",),
    "rot90": ("M", "M"),
    "flip_h": ("M.", "MM"),
}
TRANSFORM_OPS = ("identity", "rot90", "flip_h")


def _normalize_mask(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return frozenset()
    r0, c0 = coords.min(axis=0)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


MARKER_SIGNATURE_TO_OP = {
    frozenset((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "M"): op
    for op, rows in MARKER_SHAPES.items()
}
MARKER_COMPONENT_SIZES = {len(sig) + 1 for sig in MARKER_SIGNATURE_TO_OP}


def _render_shape(rows: tuple[str, ...], fill: int, bg: int, key: str) -> np.ndarray:
    return np.array([[fill if ch == key else bg for ch in row] for row in rows], dtype=int)


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": anchor}
    return np.array([[char_map.get(ch, bg) for ch in row] for row in shape_rows], dtype=int)


def _transform_array(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "identity":
        return np.array(arr, copy=True)
    if op == "rot90":
        return np.rot90(arr, -1).copy()
    if op == "flip_h":
        return np.fliplr(arr).copy()
    raise ValueError(f"unknown transform op: {op}")


def _rotate_array_cw(arr: np.ndarray, quarter_turns_cw: int) -> np.ndarray:
    return np.rot90(arr, (-int(quarter_turns_cw)) % 4).copy()


def _apply_transform(
    patch: np.ndarray,
    mask: np.ndarray,
    anchor_rc: tuple[int, int],
    op: str,
    quarter_turns_cw: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    patch_t = _transform_array(patch, op)
    mask_t = _transform_array(mask.astype(int), op).astype(bool)
    anchor_mask = np.zeros(mask.shape, dtype=int)
    anchor_mask[int(anchor_rc[0]), int(anchor_rc[1])] = 1
    anchor_t = _transform_array(anchor_mask, op)

    patch_r = _rotate_array_cw(patch_t, quarter_turns_cw)
    mask_r = _rotate_array_cw(mask_t.astype(int), quarter_turns_cw).astype(bool)
    anchor_r = _rotate_array_cw(anchor_t, quarter_turns_cw)
    pts = np.argwhere(anchor_r == 1)
    assert pts.shape[0] == 1
    ar, ac = map(int, pts[0])
    return patch_r, mask_r, (int(ar), int(ac))


def _interior_bg(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(grid: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != int(ibg), connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
        patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
        patch[~mask] = 0
        vals = [int(inner[int(r), int(c)]) for r, c in coords]
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": dict(Counter(vals)),
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
            }
        )

    comps.sort(key=lambda d: -int(d["size"]))
    return int(ibg), comps


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], int, int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6

    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if (
                        int(g[int(r0), int(c1)]) != int(bc)
                        or int(g[int(r1), int(c0)]) != int(bc)
                        or int(g[int(r1), int(c1)]) != int(bc)
                    ):
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == int(bc)).all()):
                        continue
                    ibg = _interior_bg(g, (int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    return rects[:6]


def _anchor_color_from_component(comp: dict) -> int | None:
    counts = {int(col): int(ct) for col, ct in comp["colors"].items() if int(col) != 0}
    if not counts:
        return None
    min_ct = min(counts.values())
    anchor_cols = [int(col) for col, ct in counts.items() if int(ct) == int(min_ct)]
    if len(anchor_cols) != 1:
        return None
    return int(anchor_cols[0])


def _quarter_turns_cw_for_locator(rr: int, cc: int, inner_h: int, inner_w: int) -> int:
    top_half = int(2 * int(rr) + 1) < int(inner_h)
    left_half = int(2 * int(cc) + 1) < int(inner_w)
    if top_half and left_half:
        return 0
    if top_half and not left_half:
        return 1
    if not top_half and not left_half:
        return 2
    return 3


def _quadrant_ranges(inner_h: int, inner_w: int, quadrant: int) -> tuple[list[int], list[int]]:
    top_rows = list(range(0, int(inner_h // 2)))
    bot_rows = list(range(int((inner_h + 1) // 2), int(inner_h)))
    left_cols = list(range(0, int(inner_w // 2)))
    right_cols = list(range(int((inner_w + 1) // 2), int(inner_w)))
    if quadrant == 0:
        return top_rows, left_cols
    if quadrant == 1:
        return top_rows, right_cols
    if quadrant == 2:
        return bot_rows, right_cols
    return bot_rows, left_cols


def _pointer_rel_for_op(op: str, direction: tuple[int, int]) -> tuple[int, int]:
    if op == "identity":
        mapping = {
            (-1, 0): (-1, 0),
            (0, 1): (0, 2),
            (1, 0): (1, 0),
            (0, -1): (0, -1),
        }
    elif op == "rot90":
        mapping = {
            (-1, 0): (-1, 0),
            (0, 1): (0, 1),
            (1, 0): (2, 0),
            (0, -1): (0, -1),
        }
    elif op == "flip_h":
        mapping = {
            (-1, 0): (-1, 0),
            (0, 1): (0, 1),
            (1, 0): (2, 0),
            (0, -1): (0, -1),
        }
    else:
        raise ValueError(f"unknown marker op: {op}")
    return tuple(mapping[tuple(direction)])


def _render_marker_component(
    op: str,
    color: int,
    pointer_direction: tuple[int, int],
    pointer_color: int,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    symbol_rows = MARKER_SHAPES[op]
    symbol_coords = [(int(r), int(c)) for r, row in enumerate(symbol_rows) for c, ch in enumerate(row) if ch == "M"]
    pr, pc = _pointer_rel_for_op(op, tuple(pointer_direction))
    all_coords = symbol_coords + [(int(pr), int(pc))]
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_r = max(r for r, _ in all_coords)
    max_c = max(c for _, c in all_coords)
    patch = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)
    for r, c in symbol_coords:
        patch[int(r - min_r), int(c - min_c)] = int(color)
    patch[int(pr - min_r), int(pc - min_c)] = int(pointer_color)
    locator = (int(-min_r), int(-min_c))
    return patch, locator


def _decode_marker_component(comp: dict) -> tuple[int, str, tuple[int, int], tuple[int, int]] | None:
    counts = {int(col): int(ct) for col, ct in comp["colors"].items() if int(col) != 0}
    if len(counts) != 2:
        return None
    sorted_counts = sorted(counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))
    pointer_col, pointer_ct = sorted_counts[0]
    anchor_col, anchor_ct = sorted_counts[1]
    if int(pointer_ct) != 1 or int(anchor_ct) not in (2, 3):
        return None

    patch = np.array(comp["patch"], dtype=int)
    mask = np.array(comp["mask"], dtype=bool)
    anchor_mask = mask & (patch == int(anchor_col))
    pointer_pts = np.argwhere(mask & (patch == int(pointer_col)))
    if pointer_pts.shape[0] != 1:
        return None
    signature = _normalize_mask(anchor_mask)
    op = MARKER_SIGNATURE_TO_OP.get(signature)
    if op is None:
        return None
    anchor_pts = np.argwhere(anchor_mask)
    rr0, cc0 = anchor_pts.min(axis=0)
    pr, pc = map(int, pointer_pts[0])
    pointer_rel = (int(pr - rr0), int(pc - cc0))
    direction = None
    for cand in DIRECTIONS:
        if pointer_rel == _pointer_rel_for_op(op, tuple(cand)):
            direction = tuple(cand)
            break
    if direction is None:
        return None
    bbox_r0, bbox_c0, _bbox_r1, _bbox_c1 = map(int, comp["bbox"])
    locator = (int(bbox_r0 + rr0), int(bbox_c0 + cc0))
    return int(anchor_col), str(op), tuple(direction), tuple(locator)


def _extract_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor_col = _anchor_color_from_component(comp)
        if anchor_col is None:
            continue
        patch = np.array(comp["patch"], dtype=int)
        mask = np.array(comp["mask"], dtype=bool)
        anchors = np.argwhere(mask & (patch == int(anchor_col)))
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via color-matched symbol markers with directional and quadrant rotation cues.

    Input structure:
    - Two bordered rectangles sit on striped wallpaper: the source contains multi-cell two-color sprites, while the destination contains tiny connected instruction symbols.
    - Each destination marker has a colored symbol whose hue matches one sprite anchor, plus one adjacent border-colored pointer pixel.

    Rule:
    - Use the marker color to choose the source sprite, decode the symbol shape into a base transform, then add two quarter-turn rotations: one from the marker's quadrant inside the destination interior and one from the pointer direction.
    - Apply the combined transform to the matched sprite and align the transformed anchor to the symbol's locator cell.
    - Clear the destination interior first, then output only that bordered destination rectangle after all transformed stamps are printed.

    Key insight: The destination cue is now a three-part instruction: color selects the template, symbol shape selects the transform family, and the marker's position plus pointer side jointly determine the final rotation.
    """
    g = np.array(grid, dtype=int)

    rects = _find_rectangles(g)
    assert rects

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        if not comps:
            continue
        sizes = [int(comp["size"]) for comp in comps]
        is_marker_rect = (
            len(comps) >= 2
            and all(int(sz) in MARKER_COMPONENT_SIZES for sz in sizes)
            and all(_decode_marker_component(comp) is not None for comp in comps)
        )
        if max(sizes) >= 5 and src is None:
            src = (rect, bc, ibg2, comps)
        if is_marker_rect and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source sprites and encoded destination markers"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, dst_comps = dst
    templates = _extract_templates(src_comps)

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner_h = int(dr1 - dr - 1)
    inner_w = int(dc1 - dc - 1)
    marker_records: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
    for comp in dst_comps:
        decoded = _decode_marker_component(comp)
        if decoded is not None:
            marker_records.append(decoded)

    for anchor_col, op, direction, locator in sorted(marker_records, key=lambda x: (x[3][0], x[3][1], x[0])):
        if int(anchor_col) not in templates:
            continue
        patch, mask, anchor_rc = templates[int(anchor_col)]
        quadrant_turns = _quarter_turns_cw_for_locator(int(locator[0]), int(locator[1]), int(inner_h), int(inner_w))
        direction_turns = int(DIR_TO_QTURNS[tuple(direction)])
        total_turns = (int(quadrant_turns) + int(direction_turns)) % 4
        patch_t, mask_t, (ar, ac) = _apply_transform(patch, mask, anchor_rc, op, total_turns)
        top = int(1 + int(locator[0]) - int(ar))
        left = int(1 + int(locator[1]) - int(ac))
        paste(out, patch_t, int(top), int(left), mask=mask_t, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT = 3
SOURCE_SPRITE_RANGE = (2, 4)
SOURCE_SPRITE_PADDING = 1
MARKER_PADDING = 1
MARKER_LAYOUT_RETRIES = 80

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 500

SPRITE_SHAPES = [
    (("A..", "BBB", ".B."), (0, 0)),
    (("..A", "BBB", "B.."), (0, 2)),
    ((".A.", "BBB", "..B"), (0, 1)),
    (("A.B", "BBB", "..."), (0, 0)),
    (("AB.", "B..", "BBB"), (0, 0)),
    ((".AB", "..B", "BBB"), (0, 1)),
    (("A..", "BBB", "BB."), (0, 0)),
    (("..A", "BBB", ".BB"), (0, 2)),
]


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (src_r, src_c), (dst_r, dst_c)


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(1, 10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]

    dst_border_extra = extras_needed >= 2 or (extras_needed == 1 and random.random() < 0.5)
    body_extra = extras_needed >= 2 or (extras_needed >= 1 and not dst_border_extra)

    if dst_border_extra:
        if not remaining:
            return None
        dst_border = remaining.pop()
    else:
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)

    body_pool = wallpaper[:]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _variant_signature(
    patch: np.ndarray, mask: np.ndarray, anchor_rc: tuple[int, int], op: str, quarter_turns_cw: int
) -> tuple:
    patch_t, mask_t, anchor_t = _apply_transform(patch, mask, anchor_rc, op, quarter_turns_cw)
    return (
        patch_t.shape,
        tuple(map(tuple, patch_t.tolist())),
        tuple(map(tuple, mask_t.astype(int).tolist())),
        tuple(anchor_t),
    )


def _shape_is_combo_visible(shape_rows: tuple[str, ...], anchor_rc: tuple[int, int]) -> bool:
    patch = _render_sprite(shape_rows, 1, 2, 0)
    mask = patch != 0
    variants = {
        _variant_signature(patch, mask, anchor_rc, op, q)
        for op in TRANSFORM_OPS
        for q in range(4)
    }
    return len(variants) >= 6


ASYM_SPRITE_SHAPES = [spec for spec in SPRITE_SHAPES if _shape_is_combo_visible(*spec)]


def _occupied_overlap(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = patch.shape
    return bool(((canvas[int(top) : int(top) + h, int(left) : int(left) + w] != int(bg)) & (patch != int(bg))).any())


def _bbox_has_padding(canvas: np.ndarray, top: int, left: int, h: int, w: int, bg: int, padding: int) -> bool:
    rr0 = max(0, int(top) - int(padding))
    cc0 = max(0, int(left) - int(padding))
    rr1 = min(canvas.shape[0], int(top) + int(h) + int(padding))
    cc1 = min(canvas.shape[1], int(left) + int(w) + int(padding))
    return not bool((canvas[rr0:rr1, cc0:cc1] != int(bg)).any())


def _place_markers(
    sprite_types: list[dict], marker_specs: list[dict], dst_h: int, dst_w: int, dst_bg: int, dst_border: int
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    inner_h = int(dst_h - 2)
    inner_w = int(dst_w - 2)
    out_inner = np.full((inner_h, inner_w), int(dst_bg), dtype=int)
    marker_inner = np.full((inner_h, inner_w), int(dst_bg), dtype=int)

    for _ in range(MARKER_LAYOUT_RETRIES):
        out_inner[:, :] = int(dst_bg)
        marker_inner[:, :] = int(dst_bg)
        placed: list[dict] = []
        order = marker_specs[:]
        random.shuffle(order)
        placed_all = True

        for spec in order:
            sprite = sprite_types[int(spec["type_id"])]
            quadrant = int(spec["quadrant"])
            direction = tuple(spec["direction"])
            op = str(spec["op"])
            total_turns = (int(quadrant) + int(DIR_TO_QTURNS[tuple(direction)])) % 4
            patch_t, mask_t, anchor_t = _apply_transform(
                sprite["dst_patch"],
                sprite["dst_mask"],
                sprite["anchor_rc"],
                op,
                total_turns,
            )
            marker_patch, marker_loc = _render_marker_component(op, int(sprite["anchor"]), direction, int(dst_border), int(dst_bg))
            ph, pw = patch_t.shape
            mh, mw = marker_patch.shape
            rows, cols = _quadrant_ranges(inner_h, inner_w, quadrant)
            if not rows or not cols:
                placed_all = False
                break

            candidates: list[tuple[int, int, int, int]] = []
            for top in range(0, inner_h - ph + 1):
                for left in range(0, inner_w - pw + 1):
                    if _occupied_overlap(out_inner, patch_t, top, left, int(dst_bg)):
                        continue
                    locator_r = int(top + int(anchor_t[0]))
                    locator_c = int(left + int(anchor_t[1]))
                    if locator_r not in rows or locator_c not in cols:
                        continue
                    marker_top = int(locator_r - int(marker_loc[0]))
                    marker_left = int(locator_c - int(marker_loc[1]))
                    if marker_top < 0 or marker_left < 0 or marker_top + mh > inner_h or marker_left + mw > inner_w:
                        continue
                    if _occupied_overlap(marker_inner, marker_patch, marker_top, marker_left, int(dst_bg)):
                        continue
                    if not _bbox_has_padding(marker_inner, marker_top, marker_left, mh, mw, int(dst_bg), MARKER_PADDING):
                        continue
                    candidates.append((int(top), int(left), int(marker_top), int(marker_left)))

            if not candidates:
                placed_all = False
                break

            top, left, marker_top, marker_left = random.choice(candidates)
            paste(out_inner, patch_t, int(top), int(left), mask=mask_t, overwrite=True)
            paste(marker_inner, marker_patch, int(marker_top), int(marker_left), transparent=int(dst_bg))
            placed.append(
                {
                    "type_id": int(spec["type_id"]),
                    "op": op,
                    "direction": tuple(direction),
                    "quadrant": int(quadrant),
                    "marker_top_left": (int(marker_top), int(marker_left)),
                }
            )

        if placed_all:
            return out_inner.copy(), marker_inner.copy(), placed

    return None


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        try:
            r0, c0 = random_free_location(
                inner,
                sprite["src_patch"],
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, sprite["src_patch"], int(r0), int(c0), transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    if len(ASYM_SPRITE_SHAPES) < n_types:
        return None
    sprite_specs = random.sample(ASYM_SPRITE_SHAPES, n_types)

    body_choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg and c != dst_border]
    if len(set(body_choices)) < n_types:
        return None
    if forced_body is not None and forced_body not in body_choices:
        return None

    bodies: list[int] = []
    remaining_bodies = body_choices[:]
    random.shuffle(remaining_bodies)
    if forced_body is not None:
        bodies.append(int(forced_body))
        remaining_bodies = [c for c in remaining_bodies if c != forced_body]
    while len(bodies) < n_types:
        pick = remaining_bodies.pop()
        if pick not in bodies:
            bodies.append(int(pick))

    sprite_types = []
    for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors):
        src_patch = _render_sprite(shape_rows, int(body), int(anchor), int(src_bg))
        dst_patch = _render_sprite(shape_rows, int(body), int(anchor), int(dst_bg))
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": tuple(anchor_rc),
                "body": int(body),
                "anchor": int(anchor),
                "src_patch": src_patch,
                "dst_patch": dst_patch,
                "dst_mask": dst_patch != int(dst_bg),
            }
        )

    source_total = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(source_total, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    marker_type_ids = [i for i, ct in enumerate(_choose_counts(MARKER_COUNT, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)
    marker_ops = list(TRANSFORM_OPS)
    random.shuffle(marker_ops)
    marker_directions = random.sample(DIRECTIONS, MARKER_COUNT)
    marker_quadrants = random.sample([0, 1, 2, 3], MARKER_COUNT)
    marker_specs = [
        {
            "type_id": int(type_id),
            "op": str(op),
            "direction": tuple(direction),
            "quadrant": int(quadrant),
        }
        for type_id, op, direction, quadrant in zip(marker_type_ids, marker_ops, marker_directions, marker_quadrants)
    ]
    random.shuffle(marker_specs)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_specs, dst_h, dst_w, dst_bg, dst_border)
    if marker_result is None:
        return None
    out_inner, marker_inner, _placed_specs = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(dest_rect[1:-1, 1:-1], marker_inner, 0, 0)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo-37 input after {MAX_GEN_ATTEMPTS} attempts")
