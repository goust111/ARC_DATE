# Variant of edb79dae
# Dimension: D13 — 图例条目与精灵形状的覆盖关系 (meta)
# Change: 图例中的每种形状模式对应一种输出颜色，所有放置的精灵都能在图例中找到匹配的形状条目... → 图例条目不完整（部分精灵无匹配图例）

"""
design_analysis:
  variant_goal:
    keep:
      - 单个灰色矩形框
      - 框内规则网格中的纯色占位块
      - 框外同色精灵模板
      - 框外 3x4 双色图例块
    change:
      - 图例不再覆盖所有占位色
      - 只有出现在图例中的占位色会被重着色
      - 没有图例的占位色仍然放置对应精灵形状，但输出颜色保持为原占位色
  concrete_behavior:
    example:
      - 框内占位色有 {2,3,6}，框外精灵模板也有 {2,3,6}
      - 图例只有 2->4 和 6->7，没有颜色 3 的图例
      - 原题输出会要求 {2,3,6} 全部被重着色；本变体里颜色 3 的所有放置精灵都保持 3
    affected_stage:
      - 原 solver 在解析完图例后要求 color_map 覆盖全部 placeholder_colors
      - 本变体要把该环节改为“图例可缺失”，并在最终着色时对缺失项使用默认值 src_col
  complexity_calibration:
    choices:
      - 每个样例仍保留 2-5 种占位色，图例覆盖其中真子集
      - 强制至少 1 种有图例、至少 1 种无图例，避免退化回原题
      - 强制至少 1 种无图例颜色在框内重复出现，保证“保留原色”可被肉眼观察
      - 精灵遮罩在同一样例内保持互异，降低“缺图例但形状又重复”带来的歧义
  implicit_assumption_review:
    original_main_lines:
      - legend parsing itself still works: 现有图例块仍是同样的 3x4 双色块
      - "assert set(color_map) == placeholder_colors" no longer holds and must be removed
      - final stamping previously asserted each placeholder color existed in color_map; now recolor defaults to source color
      - sprite extraction via non-legend same-color components remains valid as long as每种占位色始终有一个框外精灵模板
    generator_dependencies:
      - 原 generator 默认给每个 src 都放 legend；现在必须只给严格子集放 legend
      - 原输出总用 color_to_tgt[src]；现在要用 color_to_tgt.get(src, src)
      - 为了让变体可见，target colors 选择为非 src 颜色，避免“已重着色”和“保留原色”在颜色上混淆
  placement_legality:
    guarantees:
      - frame 与所有外部 patch 仍由 random_free_location + padding 放置，彼此不重叠不越界
      - 每个占位色始终对应一个框外同 bbox 的精灵模板
      - 图例只放一部分，不会影响未放图例颜色的精灵模板识别
  visual_observability:
    cues:
      - 样例中同时出现两类结果：部分形状被改成图例颜色，部分形状保持输入占位色
      - 缺失图例不是隐藏元数据，而是肉眼可见的“框外没有该颜色对应的 3x4 图例块”
      - 未映射颜色至少重复一次，使默认规则不是单次偶然现象
  cleanliness:
    notes:
      - 图例仍保持固定小块、与精灵模板分离，避免区域叠加
      - 输出只显示框区域，视觉负担与原题一致
  color_inference:
    policy:
      - 不用颜色 id 做排序或逻辑分支
      - 映射颜色从非占位色池中选取，增强“有图例才变色、没图例不变色”的可推断性
  new_solver_assumptions:
    assumptions:
      - 至少存在一个框
      - 每种占位色都有一个框外同尺寸精灵模板
      - 图例块若存在，仍满足原题的紧凑双组件结构
      - 同一样例中精灵模板按占位色唯一匹配；generator 通过唯一 bbox + padding 维持这一点
"""

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the framed layout with sprite templates, recoloring only the colors that appear in the legend.

    Input structure:
    - The dominant background surrounds one rectangular gray frame whose interior contains solid-colored placeholder blocks.
    - Outside the frame are same-color sprite templates for every placeholder color, plus small two-color legend blocks for only some of those colors.

    Rule:
    - Detect the frame, collect the placeholder blocks, and recover the matching same-color sprite template outside the frame.
    - Stamp each template into every placeholder position; if that placeholder color has a legend entry, recolor the stamped sprite to the legend target color, otherwise leave it in its original color.

    Key insight: The legend is optional rather than exhaustive. A matching legend block overrides the sprite color, but the absence of a legend entry is itself meaningful and means “keep the source color”.
    """
    g = np.asarray(grid, dtype=int)
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

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
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
            if bg in sub_colors:
                continue
            if len(sub_colors) != 2:
                continue
            cset = set(sub_colors)
            if ci["col"] not in cset or cj["col"] not in cset:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = int(tgt)
            legend_ids.add(ci["id"])
            legend_ids.add(cj["id"])

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1) == want_h and (c["bbox"][3] - c["bbox"][1] + 1) == want_w
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords, f"placeholder color {src_col} missing sprite"
        tgt_col = int(color_map.get(src_col, src_col))
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
        off_r = pr0
        off_c = pc0
        for dr, dc in coords:
            rr, cc = off_r + dr, off_c + dc
            out[rr, cc] = tgt_col

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
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
MASK_RETRIES = 80


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
    """Generate a random NxN binary mask with fill ratio in [SPRITE_FILL_MIN, SPRITE_FILL_MAX]."""
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
    return mask


def _unique_sprite_masks(src_cols: list[int], n: int) -> dict[int, np.ndarray] | None:
    """Generate pairwise-distinct masks within one sample to keep legend coverage visually legible."""
    masks: dict[int, np.ndarray] = {}
    seen: set[tuple[int, ...]] = set()
    for src in src_cols:
        for _ in range(MASK_RETRIES):
            mask = _sprite_mask(n)
            key = tuple(int(v) for v in mask.flatten())
            if key not in seen:
                seen.add(key)
                masks[src] = mask
                break
        else:
            return None
    return masks


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    """Render a binary mask as a colored sprite on bg."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _content_extent(n: int, grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Content area needed for grid_rows x grid_cols of NxN blocks with 1-cell gaps."""
    return (grid_rows * n + grid_rows - 1, grid_cols * n + grid_cols - 1)


def _choose_structure() -> tuple[int, int, int, int, int] | None:
    """Pick (n, rows, cols, empty_count, total) satisfying frame size constraints."""
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


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    """Choose empty grid slots while keeping every row and column occupied."""
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    """Create an h x w frame filled with bg and a 1-cell FRAME_COL border."""
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
    """Find a free location for patch and paste it onto grid/reserve. Returns success."""
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return False
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return True


def _choose_legend_subset(src_cols: list[int]) -> tuple[set[int], set[int]]:
    """Pick a non-empty proper subset of source colors that will receive legend entries."""
    mapped_count = random.randint(1, len(src_cols) - 1)
    mapped = set(random.sample(src_cols, mapped_count))
    unmapped = set(src_cols) - mapped
    assert mapped and unmapped
    return mapped, unmapped


def _assign_cell_colors(
    filled_slots: list[tuple[int, int]],
    src_cols: list[int],
    mapped_srcs: set[int],
    unmapped_srcs: set[int],
) -> dict[tuple[int, int], int]:
    """
    Assign placeholder colors so every source color appears at least once and at least one unmapped color repeats.
    """
    counts = {src: 1 for src in src_cols}
    remaining = len(filled_slots) - len(src_cols)
    assert remaining >= 1, "need at least one repeated slot to exhibit missing-legend behavior"

    anchor_unmapped = random.choice(sorted(unmapped_srcs))
    counts[anchor_unmapped] += 1
    remaining -= 1

    if remaining > 0:
        anchor_mapped = random.choice(sorted(mapped_srcs))
        counts[anchor_mapped] += 1
        remaining -= 1

    while remaining > 0:
        counts[random.choice(src_cols)] += 1
        remaining -= 1

    palette = []
    for src in src_cols:
        palette.extend([src] * counts[src])
    random.shuffle(palette)
    assert len(palette) == len(filled_slots)
    return dict(zip(filled_slots, palette))


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    max_k = min(NUM_COLORS_RANGE[1], total - 1)
    min_k = NUM_COLORS_RANGE[0]
    if max_k < min_k:
        return None
    k = random.randint(min_k, max_k)
    src_cols = random.sample(usable, k)

    masks = _unique_sprite_masks(src_cols, n)
    if masks is None:
        return None

    mapped_srcs, unmapped_srcs = _choose_legend_subset(src_cols)
    target_pool = [c for c in usable if c not in src_cols]
    color_to_tgt = {
        src: random.choice(target_pool if target_pool else [c for c in usable if c != src])
        for src in mapped_srcs
    }

    ch, cw = _content_extent(n, rows, cols)
    min_fh = max(MIN_FRAME_H, ch + 4)
    min_fw = max(MIN_FRAME_W, cw + 4)
    frame_h = random.randint(min_fh, min(MAX_FRAME_DIM, max(min_fh, ch + 2 + FRAME_MARGIN_SLACK)))
    frame_w = random.randint(min_fw, min(MAX_FRAME_DIM, max(min_fw, cw + 2 + FRAME_MARGIN_SLACK)))
    y_margin = frame_h - 2 - ch
    x_margin = frame_w - 2 - cw
    top_margin = random.randint(1, y_margin - 1)
    left_margin = random.randint(1, x_margin - 1)

    _, filled_slots = _choose_empty_slots(rows, cols, empty_count)
    cell_colors = _assign_cell_colors(filled_slots, src_cols, mapped_srcs, unmapped_srcs)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        out_col = color_to_tgt.get(src, src)
        paste(output_frame, _colored_sprite(masks[src], out_col, bg), r0, c0, transparent=bg)

    ext_area = k * (12 + n * n + 10)
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
    for src in mapped_srcs:
        patches.append((_legend(src, color_to_tgt[src]), False))
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

    assert mapped_srcs and unmapped_srcs
    assert any(src in unmapped_srcs for src in cell_colors.values())
    assert sum(1 for src in cell_colors.values() if src in unmapped_srcs) >= 2
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae D13 input")
