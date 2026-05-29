# Variant of edb79dae
# Dimension: D7 — 符号编码：图例的结构与位置 (symbol_instructions)
# Change: 左上角图例由多个条目组成，每个条目展示一个形状模式及其对应的输出颜色，形成形状到颜色的查找表
#   → 图例改为由多个 2x2 颜色矩阵组成的紧凑图例块；每个小矩阵按位置编码，左列=输入色，右列=输出色。
#
# design_analysis:
#   a) concretization_and_trace:
#      - 保留原任务的主骨架：灰色矩形框、框内规则网格中的纯色 placeholder、框外每种颜色一个 sprite 模板。
#      - 只替换“如何告知最终重着色”的机制。原题在框外放若干 3x4 两色图例，用“谁面积更大/谁是 placeholder 色”
#        来推断 src->tgt；变体改成一个由若干 2x2 小矩阵组成的图例块，每个条目直接按空间位置读：
#        [[src, tgt],
#         [src, tgt]]
#      - 小例子：若框内有颜色 2 的 placeholder，框外 sprite 也是颜色 2；原题要在某个 3x4 图例里判断 2 和 7
#        谁是“源色”，再得出 2->7。变体里直接在某个 2x2 条目读到左列是 2、右列是 7，于是同样得出 2->7。
#        差异是：原题依赖形状/面积，变体依赖 2x2 内部的列位置。
#   b) complexity_calibration:
#      - 若只用单个 2x2 图例，就无法编码 2-5 个 placeholder 颜色，所以保持“每种颜色一个图例条目”，只是条目形态变了。
#      - 条目仍然重复出现 2-5 次，能给人类足够样例支撑；同时每条只含两种颜色，视觉上比原 3x4 图例更规整。
#   c) implicit_assumptions_review:
#      - 原 solver 中 `_bbox_overlap` + “双色紧凑区域” + “面积更大者为 src” 的逻辑完全依赖旧图例，不再成立，必须删除。
#      - 原 sprite 选择逻辑假设“同色外部组件里除了 sprite 就只有 legend 组件”。变体里也仍有同色 legend 组件，
#        所以 solver 仍需显式排除 legend cells，否则合并碎片时会把 2x1 图例竖条误当成 sprite 附属部分。
#      - 原任务隐含 placeholder 尺寸至少 3x3，因此 2x2 图例不会和 sprite 盒尺寸冲突；生成器继续保持 N=3..5。
#   d) placement_legality:
#      - 图例条目被打包成一个多色 patch，并作为不透明 bbox 一次性放置；这样其内部 bg 间隔也会被 reserve 保护，
#        后续 sprite 不会塞进图例空隙，避免出现伪造的 2x2 窗口。
#      - 各条目之间保留 1 格 bg 间隔，确保扫描 2x2 窗口时每个条目只会被识别一次，不会跨条目串联。
#   e) visual_observability:
#      - 新规则在肉眼上明显：旧题是奇特的 3x4 两色块，变体是整齐的 2x2 色块，并且“左列/右列”语义可直接比较多条样例归纳。
#      - 不使用颜色 id 排序来决定规则；图例块内部的行列位置就是唯一线索。
#   f) visual_cleanliness:
#      - frame 与外部对象仍分离；图例块集中成一个小矩形区域，sprite 仍各自散开放置，职责清楚，不需要多层覆盖。
#   g) color_change_inferability:
#      - 输出颜色不是凭空变化，而是每个 placeholder 色都在图例块里有对应 2x2 条目。
#      - target 颜色可重复，但映射始终由“同一条目中的右列”给出，归纳线索稳定。

import sys
import itertools
import random

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

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


def _legend_tile(src: int, tgt: int) -> np.ndarray:
    """2x2 position-coded legend entry: left column = src, right column = tgt."""
    return np.array([[src, tgt], [src, tgt]], dtype=int)


def _legend_bank(entries: list[tuple[int, int]], bg: int) -> np.ndarray:
    """Pack 2x2 legend entries into a compact patch with 1-cell bg gaps."""
    cols = max(1, int(np.ceil(np.sqrt(len(entries)))))
    rows = int(np.ceil(len(entries) / cols))
    h = rows * 2 + max(0, rows - 1)
    w = cols * 2 + max(0, cols - 1)
    patch = np.full((h, w), bg, dtype=int)
    for idx, (src, tgt) in enumerate(entries):
        rr = idx // cols
        cc = idx % cols
        r0 = rr * 3
        c0 = cc * 3
        patch[r0 : r0 + 2, c0 : c0 + 2] = _legend_tile(src, tgt)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored sprites into a framed region using placeholder positions and a position-coded legend.

    Input structure:
    - A bg canvas contains one rectangular single-color frame with solid-color placeholder blocks inside.
    - Outside the frame there is one sprite template per placeholder color, plus a compact legend made of 2x2 color tiles.
    - In each legend tile, the left column repeats the placeholder color and the right column repeats the output color.

    Rule:
    - Find the frame and the placeholder blocks inside it.
    - Read every 2x2 legend tile by position: left column = source color, right column = target color.
    - For each placeholder color, locate the matching-color sprite outside the frame and stamp it into every placeholder slot.
    - Recolor each stamped sprite with the target color from that color's legend tile.

    Key insight: The recoloring key is no longer encoded by a special two-color shape; it is encoded spatially inside each 2x2 matrix, so the legend must be read column-wise.
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

    # Decode 2x2 legend tiles directly from their internal column layout.
    color_map: dict[int, int] = {}
    legend_mask = np.zeros_like(g, dtype=bool)
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1] - 1):
            if not np.all(outside[r : r + 2, c : c + 2]):
                continue
            sub = g[r : r + 2, c : c + 2]
            if np.any(sub == bg) or np.any(sub == frame_col):
                continue
            src = int(sub[0, 0])
            tgt = int(sub[0, 1])
            if src == tgt:
                continue
            if int(sub[1, 0]) != src or int(sub[1, 1]) != tgt:
                continue
            if src not in placeholder_colors:
                continue
            assert not np.any(legend_mask[r : r + 2, c : c + 2]), "overlapping legend tiles detected"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[src] = tgt
            legend_mask[r : r + 2, c : c + 2] = True
    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )

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

    legend_ids: set[int] = set()
    for comp in comps:
        inside_flags = [bool(legend_mask[r, c]) for r, c in comp["pts"]]
        if any(inside_flags):
            assert all(inside_flags), "legend component must be fully contained in a legend tile"
            legend_ids.add(comp["id"])

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
        for comp in all_candidates:
            if comp["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(comp["pts"])

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

    entries = list(color_to_tgt.items())
    random.shuffle(entries)
    legend_patch = _legend_bank(entries, bg)

    ext_area = sum(n * n for _ in src_cols) + legend_patch.shape[0] * legend_patch.shape[1] + 10 * k
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

    patches: list[tuple[np.ndarray, bool]] = [(legend_patch, False)]
    for src in src_cols:
        patches.append((_colored_sprite(masks[src], src, bg), True))
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
