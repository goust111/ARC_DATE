# Combo variant of edb79dae (index 7)
# combo_spec: D5/0+D6/1
# Name: 旋转放置+多色图案重着色

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete fused behavior trace
#    - 原题链路:
#      1. 框内每个纯色 NxN 占位块, 通过颜色匹配到框外同色 sprite.
#      2. 原题把 sprite 原朝向直接贴回占位块.
#      3. 再通过小图例把整块 sprite 统一重着色成单一目标色.
#    - 本融合链路:
#      1. 颜色仍先选中 sprite 形状.
#      2. 根据占位块在框内的象限旋转 sprite:
#         左上 0, 右上 90, 右下 180, 左下 270 (`np.rot90` 次数).
#      3. 图例不再给单色, 而是给一个 N x N 多色 tile. 该 tile 与 sprite 一样先按同一个象限旋转,
#         再由旋转后的 sprite 掩膜裁切.
#      4. 因此“水平条纹 + 右上象限”会变成“竖条纹 sprite”; 不是简单的“旋转形状 + 固定方向图案”.
#    - 差异核心:
#      D5 改变了几何朝向, D6 改变了每个像素如何着色; 融合后两者共享同一旋转变量, 图案方向与形状朝向耦合.
# b) Complexity calibration
#    - 如果 sprite 90 度旋转后不变, D5 退化不可见; 所以生成器要求每个 sprite 的四个 90 度旋转都不同.
#    - 如果图案 90 度旋转后不变, D6 在融合里会退化成“固定图案也看不出来”; 所以生成器优先使用方向敏感图案
#      (横纹 / 竖纹 / 两种斜向带), 并要求至少一个重复颜色跨不同奇偶旋转象限出现.
#    - 如果每个颜色只出现一次, 人类难以看出“同一 sprite 因位置不同而旋转”; 所以生成器强制至少一个源色在
#      不同象限重复出现.
# c) Hidden assumption audit
#    - 原题的图例解析依赖“两色重叠小 glyph”; 融合后不成立, 因为 legend 现在是完整的多色矩形块.
#      solver 改为查找框外的多色连通矩形: bbox 为 N x (N+1), 最左列统一为源色, 右侧 N x N 为图案 tile.
#    - 原题把 sprite 记为相对坐标后直接平移; 融合后不成立, 必须显式旋转 mask.
#    - 原题 repaint loop 给每个 sprite 像素写一个标量颜色; 融合后必须在旋转后的 tile 中索引同一相对坐标.
#    - 象限逻辑要求占位块中心不能落在框中心横线/竖线; 生成器因此只接受所有槽位都严格位于开象限内的布局.
#    - 90 度旋转要求 bbox 为正方形; 原 spec 已经给出 NxN 占位块与 sprite, solver 仍显式断言.
# d) Placement legality
#    - 框内仍是规则 NxN 槽位网格, 槽位之间保留 1 格背景, 所以任何旋转后的 sprite 仍只占原槽位 bbox.
#    - 框外 legend 与 sprite 仍用 reserve + padding 放置, 保证不重叠、不接触, 避免连通组件误合并.
#    - legend 是不透明矩形 patch, sprite 是带背景透明的 NxN patch, 与 solver 的检测方式一致.
# e) Visual observability
#    - 颜色决定“哪种形状”, 框内位置决定“朝向”, legend 决定“多色图案”, 且图案方向也跟着朝向变.
#    - 为了让解题者在 2-4 个样例内看出规则, 生成器保证:
#      1. 四个象限都出现;
#      2. 至少一个颜色跨不同象限重复;
#      3. 至少一个放置里, 若错误地把图案视为固定方向, 输出会与真实输出不同.
# f) Visual cleanliness
#    - 规则层次仍然清楚: 框内只有占位网格, 框外只有 sprite 模板与矩形 legend.
#    - 没有额外叠标或辅助记号; 图案信息全部放在 legend 右侧 N x N 样本中.
# g) Inference / generator guarantees
#    - 每个 legend tile 在 sprite 占据的像素上至少显出两种颜色.
#    - 至少一个源色的重复放置覆盖不同旋转, 且该色图案是方向敏感的.
#    - `_variant_metrics` 会拒绝退化样本: 若“旋转 shape”或“旋转 pattern”任一在输出上不可见, 样本作废.


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color.
    Return (frame_color, (r0,c0,r1,c1)).
    """
    candidates = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((col, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one rectangular frame component, found {len(candidates)}"
    frame_col, bbox = candidates[0]
    return int(frame_col), bbox


def _bbox_h(b: tuple[int, int, int, int]) -> int:
    return int(b[2] - b[0] + 1)


def _bbox_w(b: tuple[int, int, int, int]) -> int:
    return int(b[3] - b[1] + 1)


def _rot_signature(arr: np.ndarray) -> tuple[bytes, ...]:
    return tuple(np.rot90(arr, k).tobytes() for k in range(4))


def _rotation_k_for_bbox(bbox: tuple[int, int, int, int], frame_h: int, frame_w: int) -> int:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    mid_r = (frame_h - 1) / 2.0
    mid_c = (frame_w - 1) / 2.0
    assert center_r != mid_r and center_c != mid_c, "placeholder center lies on a frame midline"
    if center_r < mid_r and center_c < mid_c:
        return 0
    if center_r < mid_r and center_c > mid_c:
        return 1
    if center_r > mid_r and center_c > mid_c:
        return 2
    return 3


def _extract_legend_patterns(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    masked = np.full_like(g, bg)
    masked[outside] = g[outside]
    comps = find_components(masked, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    pattern_map: dict[int, np.ndarray] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = _bbox_h((r0, c0, r1, c1))
        w = _bbox_w((r0, c0, r1, c1))
        if w != h + 1:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if comp["area"] != sub.size:
            continue
        if bg in sub or frame_col in sub:
            continue

        src_col = int(sub[0, 0])
        if src_col not in placeholder_colors:
            continue
        if not np.all(sub[:, 0] == src_col):
            continue

        tile = sub[:, 1:]
        if tile.shape != phw[src_col]:
            continue
        tile_colors = set(map(int, np.unique(tile)))
        if src_col in tile_colors:
            continue
        if len(tile_colors) < 2:
            continue

        assert src_col not in pattern_map, f"duplicate pattern legend for placeholder color {src_col}"
        pattern_map[src_col] = tile.copy()
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert set(pattern_map) == placeholder_colors, (
        f"legend mappings {sorted(pattern_map)} do not match placeholders {sorted(placeholder_colors)}"
    )
    return pattern_map, legend_mask


def _extract_task_parts(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    np.ndarray,
    list[tuple[int, tuple[int, int, int, int]]],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    bg = get_bg(g)
    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]
        assert phw[col][0] == phw[col][1], f"expected square placeholder for color {col}, got {phw[col]}"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    pattern_map, legend_mask = _extract_legend_patterns(g, outside, bg, frame_col, placeholder_colors, phw)

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            if all(bool(legend_mask[r, c]) for r, c in pts_tuples):
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                }
            )
            cid += 1

    sprite_masks: dict[int, np.ndarray] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [c for c in all_candidates if _bbox_h(c["bbox"]) == want_h and _bbox_w(c["bbox"]) == want_w]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        assert sh == sw, f"rotated placement requires square sprites, got {(sh, sw)} for color {src_col}"

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        mask = np.zeros((sh, sw), dtype=bool)
        for r, c in pts_set:
            mask[r - sr0, c - sc0] = True
        sprite_masks[src_col] = mask

    return bg, frame_col, crop, placeholders, pattern_map, sprite_masks


def _patterned_sprite(mask: np.ndarray, tile: np.ndarray, bg: int) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[np.asarray(mask, dtype=bool)] = np.asarray(tile, dtype=int)[np.asarray(mask, dtype=bool)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the framed output by combining color-matched sprites, quadrant-based rotation, and rotating multicolor legend tiles.

    Input structure:
    - A gray rectangular frame encloses solid-color square placeholder blocks arranged on a regular grid.
    - Outside the frame sit one monochrome sprite template per placeholder color and one opaque rectangular legend per color; each legend has a source-color tag column and an NxN multicolor pattern sample.

    Rule:
    - Detect the frame, all placeholder blocks inside it, the matching-color sprite masks outside it, and the pattern tile stored in each legend.
    - For every placeholder block, choose the matching sprite by color, then rotate both the sprite mask and that color's legend tile by the block's quadrant: upper-left 0 deg, upper-right 90 deg, lower-right 180 deg, lower-left 270 deg.
    - Stamp the rotated sprite into the placeholder bbox, coloring each occupied cell from the correspondingly rotated legend tile, while keeping the frame border and background gaps unchanged.

    Key insight: The legend gives a whole color pattern rather than one target color, and the same quadrant rule governs both the sprite's geometry and the pattern's orientation, so stripes or diagonal bands turn with the sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg, _frame_col, crop, placeholders, pattern_map, sprite_masks = _extract_task_parts(g)
    H, W = crop.shape

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, bbox in placeholders:
        pr0, pc0, pr1, pc1 = bbox
        mask = sprite_masks[src_col]
        tile = pattern_map[src_col]
        sh, sw = mask.shape
        assert tile.shape == (sh, sw), f"legend tile shape does not match sprite for color {src_col}"
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )

        krot = _rotation_k_for_bbox(bbox, H, W)
        rot_mask = np.rot90(mask, krot)
        rot_tile = np.rot90(tile, krot)
        paste(out, _patterned_sprite(rot_mask, rot_tile, bg), pr0, pc0, transparent=bg)

    return out


# --------------- generator ---------------

FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 5)
SPRITE_FILL_MIN = 0.5
SPRITE_FILL_MAX = 0.9
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300
LAYOUT_RETRIES = 80
MASK_RETRIES = 200
PATTERN_RETRIES = 60
PATTERN_NAMES = ("horizontal", "vertical", "diag_down", "diag_up")


def _legend(src: int, pattern: np.ndarray) -> np.ndarray:
    src_col = np.full((pattern.shape[0], 1), int(src), dtype=int)
    return np.hstack([src_col, np.asarray(pattern, dtype=int)])


def _has_four_distinct_rotations(mask: np.ndarray) -> bool:
    return len(set(_rot_signature(mask))) == 4


def _sprite_mask(n: int) -> np.ndarray:
    for _ in range(MASK_RETRIES):
        lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
        hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
        target = random.randint(lo, hi)

        mask = np.zeros((n, n), dtype=bool)
        c = random.randrange(n)
        for r in range(n):
            c = max(0, min(n - 1, c + random.choice([-1, 0, 1])))
            mask[r, c] = True
        mask[random.randrange(n), :] = True

        if random.random() < 0.5:
            mask = mask.T.copy()
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        while int(mask.sum()) < target:
            dilated = np.zeros_like(mask)
            dilated[1:, :] |= mask[:-1, :]
            dilated[:-1, :] |= mask[1:, :]
            dilated[:, 1:] |= mask[:, :-1]
            dilated[:, :-1] |= mask[:, 1:]
            frontier = list(zip(*np.where(dilated & ~mask)))
            r, c = random.choice(frontier)
            mask[r, c] = True

        if _has_four_distinct_rotations(mask):
            return mask
    raise RuntimeError(f"failed to build an asymmetric sprite mask for n={n}")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[np.asarray(mask, dtype=bool)] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    total = random.randint(*TOTAL_BLOCKS_RANGE)
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                slots = rows * cols
                empty = slots - total
                if base_fh <= MAX_FRAME_DIM and base_fw <= MAX_FRAME_DIM and 0 <= empty <= MAX_EMPTY_SLOTS:
                    options.append((n, rows, cols, empty, total))
    return random.choice(options) if options else None


def _slot_bbox(slot: tuple[int, int], n: int, top_margin: int, left_margin: int) -> tuple[int, int, int, int]:
    rr, cc = slot
    r0 = 1 + top_margin + rr * (n + 1)
    c0 = 1 + left_margin + cc * (n + 1)
    return (r0, c0, r0 + n - 1, c0 + n - 1)


def _slot_rotations(
    rows: int, cols: int, n: int, frame_h: int, frame_w: int, top_margin: int, left_margin: int
) -> dict[tuple[int, int], int] | None:
    slot_rot = {}
    for rr in range(rows):
        for cc in range(cols):
            bbox = _slot_bbox((rr, cc), n, top_margin, left_margin)
            try:
                slot_rot[(rr, cc)] = _rotation_k_for_bbox(bbox, frame_h, frame_w)
            except AssertionError:
                return None
    if set(slot_rot.values()) != {0, 1, 2, 3}:
        return None
    return slot_rot


def _choose_frame_layout(
    n: int, rows: int, cols: int
) -> tuple[int, int, int, int, dict[tuple[int, int], int]] | None:
    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    max_fh = min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK))
    max_fw = min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK))
    for _ in range(LAYOUT_RETRIES):
        frame_h = random.randint(min_fh, max_fh)
        frame_w = random.randint(min_fw, max_fw)
        y_margin = frame_h - 2 - ch
        x_margin = frame_w - 2 - cw
        top_margin = random.randint(1, y_margin - 1)
        left_margin = random.randint(1, x_margin - 1)
        slot_rot = _slot_rotations(rows, cols, n, frame_h, frame_w, top_margin, left_margin)
        if slot_rot is not None:
            return frame_h, frame_w, top_margin, left_margin, slot_rot
    return None


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int, slot_rot: dict[tuple[int, int], int]
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) != rows:
            continue
        if len({c for _, c in filled}) != cols:
            continue
        if {slot_rot[s] for s in filled} != {0, 1, 2, 3}:
            continue
        valid.append(empty)
    if not valid:
        return None
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _assign_cell_colors(
    filled_slots: list[tuple[int, int]], src_cols: list[int], slot_rot: dict[tuple[int, int], int]
) -> tuple[dict[tuple[int, int], int], int] | None:
    odd_pairs = [
        (a, b)
        for i, a in enumerate(filled_slots)
        for b in filled_slots[i + 1 :]
        if slot_rot[a] != slot_rot[b] and abs(slot_rot[a] - slot_rot[b]) % 2 == 1
    ]
    any_pairs = [
        (a, b)
        for i, a in enumerate(filled_slots)
        for b in filled_slots[i + 1 :]
        if slot_rot[a] != slot_rot[b]
    ]
    pairs = odd_pairs or any_pairs
    if not pairs:
        return None

    anchor_a, anchor_b = random.choice(pairs)
    anchor_col = random.choice(src_cols)
    cell_colors = {anchor_a: anchor_col, anchor_b: anchor_col}

    remaining_slots = [s for s in filled_slots if s not in (anchor_a, anchor_b)]
    random.shuffle(remaining_slots)
    other_cols = [c for c in src_cols if c != anchor_col]
    if len(remaining_slots) < len(other_cols):
        return None

    for col, slot in zip(other_cols, remaining_slots):
        cell_colors[slot] = col

    for slot in remaining_slots:
        if slot not in cell_colors:
            cell_colors[slot] = random.choice(src_cols)

    assert set(cell_colors) == set(filled_slots)
    assert set(cell_colors.values()) == set(src_cols)
    repeated_quads = {slot_rot[s] for s, col in cell_colors.items() if col == anchor_col}
    assert len(repeated_quads) >= 2
    return cell_colors, anchor_col


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), int(bg), dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _build_pattern(name: str, n: int, a: int, b: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    if name == "horizontal":
        selector = rr % 2
    elif name == "vertical":
        selector = cc % 2
    elif name == "diag_down":
        selector = (rr - cc) % 3
    else:
        selector = (rr + cc) % 3
    return np.where(selector == 0, int(a), int(b)).astype(int)


def _pattern_rotation_effect(mask: np.ndarray, tile: np.ndarray, rotations: set[int], bg: int) -> int:
    changed = 0
    for k in rotations:
        if k == 0:
            continue
        rot_mask = np.rot90(mask, k)
        actual = _patterned_sprite(rot_mask, np.rot90(tile, k), bg)
        fixed = _patterned_sprite(rot_mask, tile, bg)
        if not np.array_equal(actual, fixed):
            changed += 1
    return changed


def _make_pattern_tile(
    mask: np.ndarray,
    src: int,
    usable: list[int],
    *,
    prefer_directional: bool,
    relevant_rotations: set[int],
    bg: int,
) -> tuple[np.ndarray, str] | None:
    n = int(mask.shape[0])
    color_pool = [c for c in usable if c != src]
    if len(color_pool) < 2:
        return None

    for _ in range(PATTERN_RETRIES):
        a, b = random.sample(color_pool, 2)
        names = list(PATTERN_NAMES)
        if prefer_directional:
            names = ["horizontal", "vertical", "diag_down", "diag_up"]
        random.shuffle(names)
        for name in names:
            tile = _build_pattern(name, n, a, b)
            if len(set(map(int, np.unique(tile[mask])))) < 2:
                continue
            if len(set(_rot_signature(tile))) == 1:
                continue
            if prefer_directional and _pattern_rotation_effect(mask, tile, relevant_rotations, bg) <= 0:
                continue
            return tile, name
    return None


def _variant_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg, _frame_col, crop, placeholders, pattern_map, sprite_masks = _extract_task_parts(g)
    H, W = crop.shape
    rotation_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    shape_rotation_effect = 0
    pattern_rotation_effect = 0
    color_quadrants: dict[int, set[int]] = {}
    color_pattern_following: dict[int, bool] = {}

    for src_col, bbox in placeholders:
        krot = _rotation_k_for_bbox(bbox, H, W)
        rotation_hist[krot] += 1
        color_quadrants.setdefault(src_col, set()).add(krot)

        mask = sprite_masks[src_col]
        tile = pattern_map[src_col]
        if krot != 0 and not np.array_equal(np.rot90(mask, krot), mask):
            shape_rotation_effect += 1
        if krot != 0:
            actual = _patterned_sprite(np.rot90(mask, krot), np.rot90(tile, krot), bg)
            fixed = _patterned_sprite(np.rot90(mask, krot), tile, bg)
            if not np.array_equal(actual, fixed):
                pattern_rotation_effect += 1
                color_pattern_following[src_col] = True

    sources_with_pattern_following = sum(
        1
        for src_col, rots in color_quadrants.items()
        if len(rots) >= 2 and color_pattern_following.get(src_col, False)
    )
    return {
        "rotation_hist": rotation_hist,
        "shape_rotation_effect": shape_rotation_effect,
        "pattern_rotation_effect": pattern_rotation_effect,
        "sources_with_pattern_following": sources_with_pattern_following,
        "total_placeholders": len(placeholders),
    }


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, total = struct

    layout = _choose_frame_layout(n, rows, cols)
    if layout is None:
        return None
    frame_h, frame_w, top_margin, left_margin, slot_rot = layout

    empty_pick = _choose_empty_slots(rows, cols, empty_count, slot_rot)
    if empty_pick is None:
        return None
    _, filled_slots = empty_pick

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k_hi = min(NUM_COLORS_RANGE[1], total - 1)
    if k_hi < NUM_COLORS_RANGE[0]:
        return None
    k = random.randint(NUM_COLORS_RANGE[0], k_hi)
    src_cols = random.sample(usable, k)
    masks = {src: _sprite_mask(n) for src in src_cols}

    assigned = _assign_cell_colors(filled_slots, src_cols, slot_rot)
    if assigned is None:
        return None
    cell_colors, anchor_col = assigned

    anchor_rotations = {slot_rot[s] for s, col in cell_colors.items() if col == anchor_col}
    patterns: dict[int, np.ndarray] = {}
    for src in src_cols:
        built = _make_pattern_tile(
            masks[src],
            src,
            usable,
            prefer_directional=(src == anchor_col),
            relevant_rotations=anchor_rotations if src == anchor_col else {1},
            bg=bg,
        )
        if built is None:
            return None
        tile, _name = built
        patterns[src] = tile

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for slot, src in cell_colors.items():
        r0, c0, _r1, _c1 = _slot_bbox(slot, n, top_margin, left_margin)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        krot = slot_rot[slot]
        rot_mask = np.rot90(masks[src], krot)
        rot_tile = np.rot90(patterns[src], krot)
        paste(output_frame, _patterned_sprite(rot_mask, rot_tile, bg), r0, c0, transparent=bg)

    ext_area = k * (n * n + n * (n + 1) + 12)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= frame_h and w >= frame_w and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None

    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    fr0 = random.choice([0, h - frame_h])
    fc0 = random.choice([0, w - frame_w])
    if h - frame_h > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fr0 = random.randrange(0, h - frame_h + 1)
    if w - frame_w > 2 and random.random() < FRAME_POS_JITTER_PROB:
        fc0 = random.randrange(0, w - frame_w + 1)
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, np.full((frame_h, frame_w), FRAME_COL, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, patterns[src]), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
            return None
        metrics = _variant_metrics(grid)
        if set(k for k, v in metrics["rotation_hist"].items() if v > 0) != {0, 1, 2, 3}:
            return None
        if metrics["shape_rotation_effect"] <= 0:
            return None
        if metrics["pattern_rotation_effect"] <= 0:
            return None
        if metrics["sources_with_pattern_following"] <= 0:
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")
