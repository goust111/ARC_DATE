# Combo variant of edb79dae (index 16)
# combo_spec: D3/1+D5/3
# Name: 空心精灵+轮廓放置

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D3/1 + D5/3 combo
# ------------------------------------------
# variant:
#   concrete_trace:
#     原题里，框外同色精灵通常是实心/普通单色模板，框内输出直接复制整块形状再按图例重着色。
#     融合后，框外模板先升级为“空心精灵”：
#       例如 5x5 `11111/11111/11011/11111/11111`，中心 0 被同色像素包围，是一个内部洞。
#     若只做 D3/1，框内会得到同样的带洞形状；若只做 D5/3，实心精灵会被削成单层外轮廓。
#     这里两者叠加时，轮廓定义按像素连通区域边界处理：任何拥有上/下/左/右四个同色邻居
#     的像素都被挖空。于是上例会变成
#       `11111/10101/11011/10101/11111`
#     输出同时保留外边界和围绕内部洞的内边界，这个“双轮廓”正是融合后的核心可观察行为。
#   complexity_guardrails:
#     - 若空心精灵只有 1 像素厚的环，轮廓化后可能与原空心精灵完全相同，D5 会退化。
#     - 若精灵没有内部洞，则会退化成单独的 D5/3。
#     因此生成器强制每个模板同时满足：
#       1. bbox 内存在至少一个封闭背景洞；
#       2. 轮廓化会删除至少若干个像素；
#       3. 轮廓结果中存在紧邻洞的像素，确保内边界真实出现。
#   solver_assumption_audit:
#     - `_find_frame` 只依赖唯一矩形框，融合不改这点。
#     - 框内占位块仍是实心单色 NxN 方块，所以 placeholder 收集逻辑可复用。
#     - 图例仍是紧凑双色无背景重叠块；空心精灵因自身 bbox 中含背景洞，不会被误判成图例。
#     - 原 solver 默认“把外部模板的前景坐标原样写回”。融合后必须先重建模板掩码，
#       再取 4 邻域轮廓；此轮廓规则天然会同时保留外边界和内洞边界。
#     - 外部精灵仍按 bbox 与占位块同尺寸匹配，所以生成器必须保证模板四周都有前景，
#       不能因挖洞导致 bbox 缩小。
#   generator_legality:
#     - 继续使用 output-first：先用“空心模板的轮廓化结果”生成目标输出，再回填输入占位块。
#     - 框内仍是规则网格且块间至少 1 格背景，双轮廓模板互不接触。
#     - 框外继续通过 reserve 网格和 padding 放置精灵与图例，保证不重叠、不越界。
#   visual_observability:
#     - 颜色只用于同色匹配和图例重着色，不依赖颜色数值大小。
#     - 精灵尺寸固定在足够显示洞与双轮廓的范围，且同色通常在框内重复出现，
#       解题者可在 2-4 个样例里稳定看到“带洞模板 -> 双边界轮廓”的规律。
#   cleanliness:
#     框外负责提供原始带洞模板和颜色图例；框内只展示变换结果。层次没有叠在同一区域，
#     因此双轮廓结构肉眼可分辨，不会和图例/占位块混淆。
#   new_solver_assumptions:
#     - 每个外部模板必须至少有一个封闭洞，且轮廓化后仍保留非空像素。
#     - 轮廓定义采用 4 邻域“非内点保留”，生成器与 solver 复用同一实现，避免语义漂移。


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """
    Find a rectangular border component:
      - all bbox-border cells are this color,
      - bbox-interior is not entirely this color,
      - bbox-interior contains at least one non-bg, non-frame pixel.
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
            inner_vals = set(map(int, np.unique(sub[1:-1, 1:-1])))
            if not any(v not in (bg, col) for v in inner_vals):
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
    """Keep every boundary pixel of a binary shape under 4-neighbor interior removal."""
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


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    """Return enclosed background cells inside a sprite mask."""
    mask = np.asarray(mask, dtype=bool)
    holes = np.zeros_like(mask, dtype=bool)
    for pts in components_mask(~mask, connectivity=4):
        rows = pts[:, 0]
        cols = pts[:, 1]
        if np.all((rows > 0) & (rows < mask.shape[0] - 1) & (cols > 0) & (cols < mask.shape[1] - 1)):
            holes[rows, cols] = True
    return holes


def _sprite_metrics(mask: np.ndarray) -> dict[str, int]:
    """Quantify whether the combo behavior is actually visible on a sprite."""
    mask = np.asarray(mask, dtype=bool)
    outline = _outline_mask(mask)
    holes = _hole_mask(mask)
    inner_boundary = 0
    for r, c in np.argwhere(outline):
        if (
            (r > 0 and holes[r - 1, c])
            or (r + 1 < holes.shape[0] and holes[r + 1, c])
            or (c > 0 and holes[r, c - 1])
            or (c + 1 < holes.shape[1] and holes[r, c + 1])
        ):
            inner_boundary += 1
    return {
        "holes": int(holes.any()),
        "removed": int(mask.sum() - outline.sum()),
        "inner_boundary": int(inner_boundary),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor and place outlined hollow sprites into the framed workspace.

    Input structure:
    - The canvas has one rectangular gray frame whose interior contains solid-color placeholder blocks.
    - Outside the frame, each placeholder color has a same-color sprite template; these templates are hollow shapes with enclosed background holes.
    - The outside area also contains compact two-color legend glyphs that map placeholder colors to their final output colors.

    Rule:
    - Detect the frame, collect placeholder blocks inside it, and recover the matching same-color hollow sprite for each placeholder color.
    - Use the legend glyphs to determine the recoloring target for each placeholder color.
    - Before stamping a sprite into the frame, convert the sprite to its 4-neighbor contour: keep every foreground pixel on the boundary of the connected region and erase only strict interior pixels.
    - Paste that contour at every placeholder position and recolor it with the legend-selected output color, while preserving the frame border and background.

    Key insight:
    - Placeholder color still selects the template and the legend still selects the final color.
    - The fused step is that the source template already has a hole, and contour extraction is applied to the whole foreground region, so the output keeps both the outer boundary and the boundary around the hole.
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
        assert outline.any(), f"sprite for color {src_col} lost all pixels after outlining"
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

# Keep sprites large enough that both the enclosed hole and the contour-thinning are visible.
FRAME_COL = 5
SPRITE_N_RANGE = (5, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 9)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 300
MASK_RETRIES = 100
MIN_REMOVED_PIXELS = 2
MIN_INNER_BOUNDARY_PIXELS = 4


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


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
    options = []
    for n in range(*SPRITE_N_RANGE):
        for rows in range(*GRID_ROWS_RANGE):
            for cols in range(*GRID_COLS_RANGE):
                ch, cw = _content_extent(n, rows, cols)
                base_fh = ch + 4
                base_fw = cw + 4
                slots = rows * cols
                if base_fh > MAX_FRAME_DIM or base_fw > MAX_FRAME_DIM:
                    continue
                for total in range(TOTAL_BLOCKS_RANGE[0], TOTAL_BLOCKS_RANGE[1] + 1):
                    empty = slots - total
                    if 0 <= empty <= MAX_EMPTY_SLOTS:
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


def _sprite_mask(n: int) -> np.ndarray:
    """
    Generate a hollow NxN sprite whose contour keeps both outer and inner boundaries.

    Construction: start from a filled square so the bbox stays exactly NxN, then carve one
    or two strictly interior rectangles. A candidate is accepted only if it has an enclosed
    hole, contour extraction removes interior mass, and the contour contains pixels adjacent
    to the hole.
    """
    assert n >= 5, "combo generator expects sprites large enough to show a visible hole"
    for _ in range(MASK_RETRIES):
        mask = np.ones((n, n), dtype=bool)
        hole_count = random.choice([1, 1, 2])
        carved = np.zeros((n, n), dtype=bool)
        for _hole_idx in range(hole_count):
            for _ in range(20):
                hh = random.choice([1, 1, 2])
                ww = random.choice([1, 1, 2])
                r0 = random.randint(1, n - hh - 1)
                c0 = random.randint(1, n - ww - 1)
                rect = np.zeros((n, n), dtype=bool)
                rect[r0 : r0 + hh, c0 : c0 + ww] = True
                # Keep holes distinct enough that their boundaries remain readable.
                grown = rect.copy()
                grown[1:, :] |= rect[:-1, :]
                grown[:-1, :] |= rect[1:, :]
                grown[:, 1:] |= rect[:, :-1]
                grown[:, :-1] |= rect[:, 1:]
                if np.any(carved & grown):
                    continue
                carved |= rect
                mask[rect] = False
                break

        metrics = _sprite_metrics(mask)
        if (
            metrics["holes"] >= 1
            and metrics["removed"] >= MIN_REMOVED_PIXELS
            and metrics["inner_boundary"] >= MIN_INNER_BOUNDARY_PIXELS
        ):
            return mask
    raise RuntimeError(f"failed to build visible hollow+outline sprite of size {n}")


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
    outline_masks = {src: _outline_mask(mask) for src, mask in masks.items()}

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
    cell_colors: dict[tuple[int, int], int] = {}
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
        paste(output_frame, _colored_sprite(outline_masks[src], color_to_tgt[src], bg), r0, c0, transparent=bg)

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
