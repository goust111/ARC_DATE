# Variant of edb79dae
# Dimension: D2 — 第二步匹配条件：形状-颜色图例查找 (match_cond)
# Change: 放置后的精灵形状与左上角图例中的形状模式匹配（s2.shape == d2.shape），确定重着色的目标颜色... → 颜色-颜色映射替代形状查找

# Design analysis
# ----------------
# variant_spec:
#   core_change:
#     original_d2: 外部图例本身是需要解析的提示对象；第二步重着色依赖额外的图例判读逻辑。
#     new_d2: 图例改成显式颜色映射表。每行左半是输入色，右半是输出色；放入框内的精灵不再和图例做形状对应，只按原始占位色查表重着色。
#   concretization:
#     frame_region:
#       - 仍然只有一个灰色矩形框。
#       - 框内仍然是规则网格中的纯色 NxN 占位块，颜色表示“该位置要放哪种外部精灵”。
#     outside_sprites:
#       - 每种占位色仍然对应一个外部同色精灵模板，bbox 与占位块尺寸一致。
#       - solver 第一步不变：按颜色找到对应精灵，按占位块位置贴回框内。
#     new_legend:
#       - 左上角放一个 Kx4 的无背景颜色表，每行是 [src, src, tgt, tgt]。
#       - 视觉上就是“左列/右列”配色对照表，比原来的重叠双色块更直接。
#       - K 等于框内占位色种类数；每个 src 恰好出现一行。
#   behavior_trace:
#     example:
#       - 假设框内有蓝色占位块，外部有蓝色 L 形精灵。
#       - 原流程：先把蓝色 L 形贴进框内，再从外部图例中推断“蓝色最终该变成什么色”。
#       - 新流程：先把蓝色 L 形贴进框内，然后直接在左上角颜色表中找到蓝色所在行，例如 [1,1,7,7]，于是把该 L 形重着色为 7。
#       - 差异：第二步的判据从“读图例对象的内部模式”变成“按原始颜色查显式表格”。
#   complexity_calibration:
#     - 保留 2-5 种颜色、6-10 个占位块、3-5 尺寸精灵，避免任务退化成单次查色。
#     - 颜色表整块放在左上角，所有样例都可直接观察到“左色 -> 右色”的结构线索。
#     - 目标色要求两两不同，避免表格右半在竖向粘连，保证行对应清晰且 solver 无歧义。
#   implicit_assumptions_review:
#     - _find_frame 仍假设全图只有一个矩形框；生成器不会再制造别的矩形边框对象。
#     - placeholder 提取仍假设框内同色占位块 bbox 尺寸一致；生成器继续保证这一点。
#     - 旧版通过 bbox overlap 找 legend；该前提已失效，所以 main() 必须改为读取左上角表格。
#     - sprite 提取仍依赖“每个占位色在框外有且仅有一个对应 bbox 大小的模板”；生成器继续保证。
#     - 新增假设：左上角最靠前的框外非背景对象就是颜色表，而且颜色表宽 4、高 K。生成器通过固定放置左上表并给它留背景间隔来保证。
#   legality_and_cleanliness:
#     - 颜色表固定贴在画布左上角。
#     - 框和外部精灵都通过 reserve 版面约束随机放置，彼此不重叠且至少隔 1 格背景。
#     - 颜色表不用灰框，避免与主框混淆；也不用背景镂空，避免和散放精灵混在一起。
#   observability:
#     - 人类看到 2-4 个样例时，可以先识别“框内纯色块 -> 外部同色模板”，再看到左上角每行显式给出颜色映射。
#     - 输出中的颜色变化不依赖颜色 id 排序，也不依赖不可见的组件面积比较。

import sys
import itertools
import random

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region using placeholder positions and a color lookup table.

    Input structure:
    - A bg canvas contains one gray rectangular frame whose interior holds solid-color NxN placeholders.
    - Outside the frame are same-color sprite templates and, in the upper-left corner, a compact color table with one row per placeholder color.

    Rule:
    - Find the frame and the placeholder blocks inside it.
    - For each placeholder color, copy the matching-color sprite template into each placeholder position.
    - Read the upper-left table row [src, src, tgt, tgt] to map each original placeholder color to its final output color.
    - Recolor every pasted sprite with the mapped target color.

    Key insight: The sprite still comes from the placeholder color, but recoloring no longer depends on a secondary shape clue; the upper-left legend is now a direct color-to-color dictionary.
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

    outside_fg = np.argwhere(outside & (g != bg) & (g != frame_col))
    assert outside_fg.size > 0, "no external legend/sprites found"
    lr0, lc0 = min((int(r), int(c)) for r, c in outside_fg)
    legend_h = len(placeholder_colors)
    legend_w = 4
    assert lr0 + legend_h <= g.shape[0] and lc0 + legend_w <= g.shape[1], "legend table would exceed canvas bounds"
    legend = g[lr0 : lr0 + legend_h, lc0 : lc0 + legend_w]
    assert legend.shape == (legend_h, legend_w), f"unexpected legend shape {legend.shape}"
    assert np.all(legend[:, 0] == legend[:, 1]), "legend left half must be uniform per row"
    assert np.all(legend[:, 2] == legend[:, 3]), "legend right half must be uniform per row"
    assert not np.any((legend == bg) | (legend == frame_col)), "legend cannot contain bg or frame color"

    src_rows = [int(x) for x in legend[:, 0]]
    tgt_rows = [int(x) for x in legend[:, 2]]
    assert set(src_rows) == placeholder_colors, (
        f"legend source colors {sorted(set(src_rows))} do not match placeholders {sorted(placeholder_colors)}"
    )
    assert len(set(src_rows)) == legend_h, "duplicate source rows in legend"
    assert len(set(tgt_rows)) == legend_h, "target colors must stay row-distinct in legend"
    color_map = {src: tgt for src, tgt in zip(src_rows, tgt_rows)}

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
                }
            )
            cid += 1

    legend_bbox = (lr0, lc0, lr0 + legend_h - 1, lc0 + legend_w - 1)

    def _inside_bbox(b: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = b
        or0, oc0, or1, oc1 = outer
        return or0 <= r0 and r1 <= or1 and oc0 <= c0 and c1 <= oc1

    legend_ids = {comp["id"] for comp in comps if _inside_bbox(comp["bbox"], legend_bbox)}

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
        assert src_col in sprite_coords and src_col in color_map, f"placeholder color {src_col} missing sprite or color mapping"
        tgt_col = int(color_map[src_col])
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        assert pr0 + sh <= H and pc0 + sw <= W, f"sprite for color {src_col} does not fit at placeholder"
        for dr, dc in coords:
            rr, cc = pr0 + dr, pc0 + dc
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
PATCH_PADDING = 1
GEN_RETRIES = 300


def _legend_table(pairs: list[tuple[int, int]]) -> np.ndarray:
    """Build a Kx4 color lookup table: [src, src, tgt, tgt] per row."""
    return np.array([[src, src, tgt, tgt] for src, tgt in pairs], dtype=int)


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
    """Choose empty grid slots while keeping every row and column represented."""
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
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int,
    *, transparent: bool,
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


def _choose_target_colors(src_cols: list[int], usable: list[int]) -> list[int]:
    """Pick distinct target colors with no identity row, allowing cross-maps among source colors."""
    for _ in range(200):
        tgt_cols = random.sample(usable, len(src_cols))
        if all(src != tgt for src, tgt in zip(src_cols, tgt_cols)):
            return tgt_cols
    raise RuntimeError("failed to sample distinct target colors")


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    tgt_cols = _choose_target_colors(src_cols, usable)
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

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
        paste(output_frame, _colored_sprite(masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

    legend_pairs = list(zip(src_cols, tgt_cols))
    random.shuffle(legend_pairs)
    legend = _legend_table(legend_pairs)
    legend_h, legend_w = legend.shape

    ext_area = legend_h * legend_w + k * (n * n + 12) + 20
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= max(frame_h, legend_h)
        and w >= max(frame_w, legend_w)
        and (((h - frame_h) >= legend_h + PATCH_PADDING) or ((w - frame_w) >= legend_w + PATCH_PADDING))
        and h * w - frame_h * frame_w >= ext_area
    ]
    if not possible_dims:
        return None
    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    paste(grid, legend, 0, 0)
    paste(reserve, np.full((legend_h, legend_w), FRAME_COL, dtype=int), 0, 0)

    frame_blocker = np.full((frame_h, frame_w), FRAME_COL, dtype=int)
    try:
        fr0, fc0 = random_free_location(reserve, frame_blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return None
    paste(grid, input_frame, fr0, fc0)
    paste(reserve, frame_blocker, fr0, fc0)

    patches = [(_colored_sprite(masks[src], src, bg), True) for src in src_cols]
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
