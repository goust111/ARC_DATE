# Variant of edb79dae
# Dimension: D5 — 第一步变换：精灵放置方式 (transform)
# Change: 用精灵的完整形状替换框架内的单色目标像素（replace d with s.shape），精灵图案居中覆盖目标位置... → 仅放置精灵轮廓（空心化）

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D5 hollow-placement variant
# ---------------------------------------------------
# variant:
#   core_change:
#     原题把外部同色精灵的全部前景像素搬进框内；这里改成先对该精灵做一层“空心化”，
#     只保留其 4 邻域外轮廓。任何同时拥有上/下/左/右四个同色邻居的像素都会被挖空。
#   concrete_behavior:
#     输入仍然包含：灰框、框内实心占位块、框外原始精灵模板、框外颜色图例。
#     输出仍然只保留框区域，但框内不再出现实心模板，而是出现同形 bbox 的空心边框。
#     例：3x3 实心十字 `010/111/010` 原题会整块保留 5 个像素；现在中心像素被删除，
#     变成 `010/101/010`。
#   flow_impact:
#     受影响的只有“把外部模板贴回框内”这一步。找框、找占位色、解析图例、按颜色匹配模板、
#     最后统一重着色，这些高层关系都保持不变。
#   complexity_guardrails:
#     如果某个模板没有内部像素，那么空心化后会和原题输出完全相同，D5 就不可观察。
#     因此生成器额外保证每个模板都至少有 1 个会被挖空的内部像素，并让同一种颜色在框内
#     通常出现多次，从而在一张图里重复展示“实心变空心”的规律。
#   solver_assumption_audit:
#     - `_find_frame` 只依赖唯一矩形框，变体不改这一点。
#     - 占位块收集逻辑只依赖框内仍是实心单色 NxN 方块，仍成立。
#     - 图例检测依赖框外存在小型双色无背景重叠区域，变体不改图例外观，仍成立。
#     - 原 solver 最关键的旧假设是：`sprite_coords` 可直接写回输出。这里改成：
#       先把 coords 还原成掩码，再取轮廓 coords，最后写回。否则会退回原题行为。
#     - 写回时仍按占位 bbox 左上角对齐；因为生成器继续保证模板 bbox 与占位 bbox 同尺寸，
#       不需要额外平移规则。
#   generator_legality:
#     - 框内仍使用规则网格和 1 像素背景间距，所以空心模板之间不会互相接触或覆盖。
#     - 框外仍用 reserve 网格加 padding 放置图例与模板，保证不重叠、不越界。
#     - 生成器采用 output-first：先用空心模板构造期望输出，再回填输入占位块，并用 `main`
#       做 oracle 校验。
#   visual_observability:
#     - 不使用颜色 id 的排序语义来决定行为；颜色只做匹配和重着色。
#     - 模板尺寸仍限制在 3-5，但每个模板强制含有内部像素，所以 2-4 个样例里能直接看见
#       “实心模板被挖成轮廓”的稳定规律。
#   cleanliness:
#     空心化只发生在框内输出区域，图例和外部模板保持原貌，因此画面层次仍然清楚：
#     框外负责提示，框内负责展示变换结果。
#   new_solver_assumptions:
#     - 每个外部模板在其 bbox 内至少有一个非轮廓像素，否则虽然 solver 仍正确，但样例会退化。
#       `generate_input()` 通过 `_sprite_mask()` 的约束保证这一点。
#     - 轮廓提取使用 4 邻域定义，生成器与 solver 共用同一函数，避免语义漂移。


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


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the 4-neighborhood outer shell of a binary sprite mask."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place recolored sprite outlines inside the frame using the placeholders and the external legend.

    Input structure:
    - The scene has a background, one rectangular gray frame, and several solid-color placeholder blocks inside it.
    - Outside the frame sit the source-color sprite templates plus compact two-color legend glyphs that indicate recoloring.

    Rule:
    - Detect the frame, the placeholder colors inside it, and the matching same-color sprite templates outside it.
    - Use the legend to map each placeholder color to its final output color.
    - Before stamping a template into the frame, hollow it out: keep only its outer contour pixels and turn interior pixels back to background.
    - Paste that outline at every placeholder position, preserving the frame border and leaving untouched interior background cells as background.

    Key insight:
    - Placeholder color still chooses which external template to use, and the legend still chooses the output color.
    - The new part is local to the stamping step: the copied object is not the full sprite any more, but its 4-neighbor outline.
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

    def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)

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
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

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

        sprite_mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for r, c in pts_set:
            sprite_mask[r - sr0, c - sc0] = True
        outline = _outline_mask(sprite_mask)
        assert outline.any(), f"sprite for color {src_col} lost all pixels after hollowing"
        coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(outline)))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_coords and src_col in color_map, f"placeholder color {src_col} missing sprite or color mapping"
        tgt_col = int(color_map[src_col])
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
SPRITE_FILL_MIN = 0.6
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
MIN_INTERIOR_CELLS = 1


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    """Pixels that will be removed by the hollowing rule."""
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
    Generate an NxN mask that visibly changes under hollowing.

    The seed includes a plus around an interior cell so at least one pixel is removable.
    Frontier growth keeps the shape organic while preserving the guaranteed interior.
    """
    lo = max(5, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)
    target = random.randint(lo, hi)

    for _ in range(MASK_RETRIES):
        mask = np.zeros((n, n), dtype=bool)

        cr = random.randrange(1, n - 1)
        cc = random.randrange(1, n - 1)
        mask[cr, cc] = True
        mask[cr - 1, cc] = True
        mask[cr + 1, cc] = True
        mask[cr, cc - 1] = True
        mask[cr, cc + 1] = True

        if random.random() < 0.5:
            mask[random.randrange(n), :] = True
        else:
            mask[:, random.randrange(n)] = True

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
            mask = mask.T.copy()
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        if int(_interior_mask(mask).sum()) >= MIN_INTERIOR_CELLS:
            return mask

    raise RuntimeError(f"failed to build hollowable sprite of size {n}")


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
    """Choose which grid slots are empty, ensuring every row and column has at least one filled slot."""
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


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}
    hollow_masks = {src: _outline_mask(mask) for src, mask in masks.items()}

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
    cell_colors = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        paste(output_frame, _colored_sprite(hollow_masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

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
    raise RuntimeError("failed to generate valid edb79dae input")
