# Combo variant of edb79dae (index 8)
# combo_spec: D5/3+D15/2
# Name: 轮廓放置+XOR重叠

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D5/3 + D15/2 combo variant
# ---------------------------------------------------
# variant:
#   concrete_trace:
#     原题在框内使用与精灵同 bbox 的实心占位块，因此把外部精灵按 bbox 对齐贴回去时，
#     不会发生真正的放置冲突。这个融合版把框内对象改成单像素锚点，外部精灵先做 4 邻域
#     空心化，再以锚点为中心放入框内。举例：若两个 5x5 精灵锚点在同一行、横向相距 3，
#     它们的轮廓 bbox 会重叠 2 列；原题式写回会保留后贴上的像素，这里改成按覆盖次数做
#     XOR，所以双方都覆盖的交叉轮廓格会被擦回背景，只剩两侧未重合的边线。
#   complexity_guardrails:
#     - 若锚点间距 >= 精灵宽度，则 XOR 永远不触发，D15/2 退化；因此生成器固定让同一行
#       的水平步长小于精灵宽度，并要求至少存在一对相邻锚点。
#     - 若同时允许横向和纵向都重叠，容易出现三重覆盖，XOR 的颜色归属就不再直观；因此
#       只制造横向两两重叠，纵向步长严格大于精灵高度。
#     - 若精灵空心化后和原形完全相同，D5/3 不可观察；因此每个模板都强制含有至少一个会
#       被挖空的内部像素。
#   solver_assumption_audit:
#     - `_find_frame` 只依赖唯一矩形边框，这一点不变。
#     - 原 solver 通过占位块 bbox 大小去锁定外部精灵；这里不再成立，所以改成：
#       框内只把每个 1x1 组件当作锚点，外部则把同色、非图例的所有组件合并成该颜色唯一精灵。
#     - 原 solver 默认“写一个像素就是最终颜色”，这在 XOR 下失效；现在必须先累计覆盖次数，
#       再把奇偶结果回写到输出。
#     - 新的中心对齐规则要求精灵 bbox 为奇数尺寸，生成器固定使用奇数尺寸并把锚点留在安全边距内。
#   generator_legality:
#     - 框内锚点放在规则行列上，横向步长小于精灵宽度以制造重叠，纵向步长大于精灵高度以避免
#       三重冲突；同时约束每行每列都有锚点，并至少有一对横向相邻锚点。
#     - 框外仍沿用 reserve + padding 的无重叠放置策略，图例和精灵模板互不接触、也不碰框。
#     - 生成器仍是 output-first：先用空心 + XOR 规则合成目标框输出，再生成对应输入，并用
#       `main()` 做 oracle 校验。
#   visual_observability:
#     - 输入里框内是彩色单像素锚点，输出里这些位置“长出”同色模板的轮廓，且相邻轮廓交叉处消失，
#       肉眼能同时看到“空心化”和“重叠消除”。
#     - 颜色重着色仍由框外双色图例给出；为减少歧义，生成器让目标色彼此不同且不复用源色。
#   cleanliness:
#     - 重叠只发生在框内输出层，框外模板与图例保持清晰提示作用。
#     - 纵向不重叠、横向只两两重叠，使画面保留规律网格感，不会混成一团噪声。
#   new_solver_assumptions:
#     - 框内每个非背景、非框色组件都是单像素锚点。
#     - 每种占位色在框外恰有一份非图例精灵，允许由多个 4 连通分量组成，但它们共同落在同一 bbox。
#     - 任一输出像素最多被两个轮廓覆盖；生成器通过单轴重叠布局保证这一点。


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


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the 4-neighbor outer shell of a binary sprite mask."""
    mask = np.asarray(mask, dtype=bool)
    interior = np.zeros_like(mask, dtype=bool)
    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        interior[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    return mask & ~interior


def _collect_outside_components(
    g: np.ndarray, outside: np.ndarray, bg: int, frame_col: int
) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    cid = 0
    for col in [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(pts_tuples),
                }
            )
            cid += 1
    return comps


def _extract_color_map_and_legends(
    g: np.ndarray,
    comps: list[dict[str, object]],
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
) -> tuple[dict[int, int], set[int]]:
    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue
            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = int(ci["col"]) if ci["col"] in placeholder_colors else int(cj["col"])
                tgt = int(cj["col"]) if src == ci["col"] else int(ci["col"])
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = int(ci["col"]), int(cj["col"])
                else:
                    src, tgt = int(cj["col"]), int(ci["col"])

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = tgt
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )
    return color_map, legend_ids


def _extract_sprite_outlines(
    comps: list[dict[str, object]], placeholder_colors: set[int], legend_ids: set[int]
) -> dict[int, np.ndarray]:
    sprite_masks: dict[int, np.ndarray] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        pts = []
        for comp in all_candidates:
            pts.extend(comp["pts"])
        pts_arr = np.array(pts, dtype=int)
        sr0, sc0, sr1, sc1 = bbox_coords(pts_arr)
        sprite_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for r, c in pts:
            sprite_mask[r - sr0, c - sc0] = True
        assert sprite_mask.shape[0] % 2 == 1 and sprite_mask.shape[1] % 2 == 1, (
            f"sprite for color {src_col} must have odd bbox dimensions, got {sprite_mask.shape}"
        )
        outline = _outline_mask(sprite_mask)
        assert outline.any(), f"sprite for color {src_col} lost all pixels after hollowing"
        sprite_masks[src_col] = outline
    return sprite_masks


def _place_outline_xor(
    cover_count: np.ndarray,
    cover_color: np.ndarray,
    outline: np.ndarray,
    tgt_col: int,
    anchor_r: int,
    anchor_c: int,
) -> None:
    sh, sw = outline.shape
    off_r = anchor_r - sh // 2
    off_c = anchor_c - sw // 2
    H, W = cover_count.shape
    assert 1 <= off_r and off_r + sh <= H - 1, "sprite placement escapes frame interior vertically"
    assert 1 <= off_c and off_c + sw <= W - 1, "sprite placement escapes frame interior horizontally"
    for dr, dc in np.argwhere(outline):
        rr = off_r + int(dr)
        cc = off_c + int(dc)
        cover_count[rr, cc] += 1
        assert cover_count[rr, cc] <= 2, "more than two outline placements overlap at one pixel"
        if cover_count[rr, cc] == 1:
            cover_color[rr, cc] = int(tgt_col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow recolored sprite outlines from interior anchor pixels, then cancel outline overlaps by XOR.

    Input structure:
    - The scene has a background, one rectangular gray frame, and several single-pixel color anchors inside that frame.
    - Outside the frame there is one sprite template for each anchor color, plus one compact two-color legend glyph that maps each source color to its final output color.

    Rule:
    - Detect the frame, the anchor pixels inside it, the same-color sprite template outside it, and the legend-based recoloring map.
    - Hollow each sprite template by keeping only its 4-neighbor outer contour.
    - Center that contour on every matching anchor position inside the frame, recolor it via the legend, and combine all placements with XOR coverage logic: cells covered twice return to background, cells covered once stay visible.
    - Preserve the frame border and leave untouched interior cells as background.

    Key insight:
    - The outer objects still split into two roles: sprite color chooses the shape, legend chooses the output color.
    - The fused behavior is in the placement step: anchors expand into hollow contours, and nearby contours do not overwrite one another; their intersections vanish because overlap is treated as a binary XOR.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    H, W = crop.shape
    assert H >= 3 and W >= 3, f"frame too small for interior: {(H, W)}"

    inner_mask = np.zeros((H, W), dtype=bool)
    inner_mask[1:-1, 1:-1] = True

    placeholders: list[tuple[int, int, int]] = []
    placeholder_colors: set[int] = set()
    for col in sorted(set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            area = len(pts)
            assert area == 1 and r0 == r1 and c0 == c1, f"expected 1x1 anchor for color {col}, got bbox {(r0, c0, r1, c1)}"
            placeholders.append((int(col), r0, c0))
            placeholder_colors.add(int(col))
    assert placeholders, "no anchor pixels found inside frame"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _collect_outside_components(g, outside, bg, frame_col)
    color_map, legend_ids = _extract_color_map_and_legends(g, comps, bg, frame_col, placeholder_colors)
    sprite_outlines = _extract_sprite_outlines(comps, placeholder_colors, legend_ids)

    out = crop.copy()
    out[1:-1, 1:-1] = bg
    cover_count = np.zeros_like(out, dtype=int)
    cover_color = np.full_like(out, bg, dtype=int)

    for src_col, anchor_r, anchor_c in sorted(placeholders, key=lambda item: (item[1], item[2], item[0])):
        assert src_col in sprite_outlines and src_col in color_map, (
            f"placeholder color {src_col} missing sprite or color mapping"
        )
        _place_outline_xor(
            cover_count,
            cover_color,
            sprite_outlines[src_col],
            color_map[src_col],
            anchor_r,
            anchor_c,
        )

    overlap_mask = cover_count == 2
    out[overlap_mask] = bg
    single_mask = cover_count == 1
    out[single_mask] = cover_color[single_mask]
    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_SIZE_CHOICES = (5,)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_ANCHORS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)
SPRITE_FILL_MIN = 0.56
SPRITE_FILL_MAX = 0.84
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 3
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 400
MASK_RETRIES = 100
MIN_INTERIOR_CELLS = 1
H_STRIDE = 3
V_STRIDE = 7


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    """Pixels that disappear when only the 4-neighbor outline is kept."""
    mask = np.asarray(mask, dtype=bool)
    interior = np.zeros_like(mask, dtype=bool)
    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        interior[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    return interior


def _sprite_mask(n: int) -> np.ndarray:
    """
    Generate an odd-sized mask that visibly changes under hollowing and guarantees side overlap.

    Every mask contains the center row, so horizontally adjacent placements always have at least
    one outline row in common. A seeded plus ensures a removable interior pixel.
    """
    lo = max(11, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)

    for _ in range(MASK_RETRIES):
        mask = np.zeros((n, n), dtype=bool)
        mid = n // 2
        mask[mid, :] = True
        mask[:, mid] = True

        target = random.randint(lo, hi)
        while int(mask.sum()) < target:
            dilated = np.zeros_like(mask)
            dilated[1:, :] |= mask[:-1, :]
            dilated[:-1, :] |= mask[1:, :]
            dilated[:, 1:] |= mask[:, :-1]
            dilated[:, :-1] |= mask[:, 1:]
            frontier = list(zip(*np.where(dilated & ~mask)))
            if not frontier:
                break
            r, c = random.choice(frontier)
            mask[r, c] = True

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        if int(_interior_mask(mask).sum()) >= MIN_INTERIOR_CELLS:
            return mask

    raise RuntimeError(f"failed to build hollowable sprite of size {n}")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _occupied_extent(n: int, rows: int, cols: int) -> tuple[int, int]:
    """Occupied bbox size of centered placements on the anchor lattice."""
    return (n + (rows - 1) * V_STRIDE, n + (cols - 1) * H_STRIDE)


def _choose_structure() -> tuple[int, int, int, int] | None:
    """Pick (n, rows, cols, total) satisfying frame-size and count constraints."""
    options = []
    for n in SPRITE_SIZE_CHOICES:
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                occupied_h, occupied_w = _occupied_extent(n, rows, cols)
                min_fh = max(MIN_FRAME_H, occupied_h + 2)
                min_fw = max(MIN_FRAME_W, occupied_w + 2)
                if min_fh > MAX_FRAME_DIM or min_fw > MAX_FRAME_DIM:
                    continue
                slots = rows * cols
                for empty_count in range(MAX_EMPTY_SLOTS + 1):
                    total = slots - empty_count
                    if TOTAL_ANCHORS_RANGE[0] <= total <= TOTAL_ANCHORS_RANGE[1]:
                        options.append((n, rows, cols, total))
    return random.choice(options) if options else None


def _choose_filled_slots(rows: int, cols: int, total: int) -> list[tuple[int, int]]:
    """
    Choose filled anchor slots.

    Requirements:
    - every row and column receives at least one anchor,
    - at least one pair of horizontally adjacent anchors exists, so XOR is visible.
    """
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    empty_count = rows * cols - total
    valid: list[list[tuple[int, int]]] = []
    for empty_combo in itertools.combinations(slots, empty_count):
        empty = set(empty_combo)
        filled = [slot for slot in slots if slot not in empty]
        if len({r for r, _ in filled}) != rows or len({c for _, c in filled}) != cols:
            continue
        has_adjacent_pair = any((r, c + 1) in filled for r, c in filled)
        if has_adjacent_pair:
            valid.append(filled)
    chosen = random.choice(valid)
    random.shuffle(chosen)
    return chosen


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _compose_output_frame(
    frame_h: int,
    frame_w: int,
    bg: int,
    placements: list[tuple[int, int, int]],
    outline_masks: dict[int, np.ndarray],
    color_to_tgt: dict[int, int],
) -> tuple[np.ndarray, int]:
    """Compose the expected frame output and count XOR-cancelled pixels."""
    out = _make_bordered_frame(frame_h, frame_w, bg)
    counts = np.zeros_like(out, dtype=int)
    colors = np.full_like(out, bg, dtype=int)

    for src, anchor_r, anchor_c in placements:
        outline = outline_masks[src]
        sh, sw = outline.shape
        off_r = anchor_r - sh // 2
        off_c = anchor_c - sw // 2
        assert 1 <= off_r and off_r + sh <= frame_h - 1
        assert 1 <= off_c and off_c + sw <= frame_w - 1
        for dr, dc in np.argwhere(outline):
            rr = off_r + int(dr)
            cc = off_c + int(dc)
            counts[rr, cc] += 1
            assert counts[rr, cc] <= 2
            if counts[rr, cc] == 1:
                colors[rr, cc] = color_to_tgt[src]
            else:
                colors[rr, cc] = bg

    out[counts == 1] = colors[counts == 1]
    return out, int((counts == 2).sum())


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]
    k = random.randint(*NUM_COLORS_RANGE)
    if k > total:
        return None

    src_cols = random.sample(usable, k)
    tgt_pool = [c for c in usable if c not in src_cols]
    if len(tgt_pool) < k:
        return None
    tgt_cols = random.sample(tgt_pool, k)
    color_to_tgt = dict(zip(src_cols, tgt_cols))

    masks = {src: _sprite_mask(n) for src in src_cols}
    outline_masks = {src: _outline_mask(mask) for src, mask in masks.items()}
    if any(not outline.any() for outline in outline_masks.values()):
        return None

    occupied_h, occupied_w = _occupied_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, occupied_h + 2)
    min_fw = max(MIN_FRAME_W, occupied_w + 2)
    max_fh = min(MAX_FRAME_DIM, occupied_h + 2 + FRAME_MARGIN_SLACK)
    max_fw = min(MAX_FRAME_DIM, occupied_w + 2 + FRAME_MARGIN_SLACK)
    if min_fh > max_fh or min_fw > max_fw:
        return None
    frame_h = random.randint(min_fh, max_fh)
    frame_w = random.randint(min_fw, max_fw)
    y_slack = frame_h - 2 - occupied_h
    x_slack = frame_w - 2 - occupied_w
    top_margin = random.randint(0, y_slack)
    left_margin = random.randint(0, x_slack)

    filled_slots = _choose_filled_slots(rows, cols, total)
    slot_colors: dict[tuple[int, int], int] = {}
    for slot, src in zip(filled_slots[:k], src_cols):
        slot_colors[slot] = src
    for slot in filled_slots[k:]:
        slot_colors[slot] = random.choice(src_cols)

    radius = n // 2
    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    placements: list[tuple[int, int, int]] = []
    for (rr, cc), src in sorted(slot_colors.items()):
        anchor_r = 1 + top_margin + radius + rr * V_STRIDE
        anchor_c = 1 + left_margin + radius + cc * H_STRIDE
        input_frame[anchor_r, anchor_c] = src
        placements.append((src, anchor_r, anchor_c))

    output_frame, overlap_cells = _compose_output_frame(
        frame_h,
        frame_w,
        bg,
        placements,
        outline_masks,
        color_to_tgt,
    )
    if overlap_cells <= 0:
        return None

    outline_removed = sum(int(masks[src].sum() - outline_masks[src].sum()) for src in src_cols)
    if outline_removed <= 0:
        return None

    ext_area = k * (12 + n * n + 12)
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
    for src, tgt in zip(src_cols, tgt_cols):
        patches.append((_colored_sprite(masks[src], src, bg), True))
        patches.append((_legend(src, tgt), False))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].size, reverse=True)
    for patch, transparent in patches:
        if not _place_patch(grid, reserve, patch, bg, transparent=transparent):
            return None

    try:
        if not np.array_equal(main(grid), output_frame):
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
