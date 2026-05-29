# Variant of edb79dae
# Dimension: D5 — 第一步变换：精灵放置方式 (transform)
# Change: 用精灵的完整形状替换框架内的单色目标像素（replace d with s.shape），精灵图案居中覆盖目标位置... → 旋转后放置

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D5 variant
# ----------------------------------
# spec:
#   core_objects:
#     - 原题的对象集合保持不变：一个灰色矩形框、框内规则网格上的纯色占位块、框外同色精灵模板、框外图例块。
#     - 仍然先通过占位色选中精灵形状，再通过图例把该占位色重着色成目标色。
#   changed_step:
#     - 唯一核心改动落在“把精灵贴回框内”的这一步。
#     - 原题是把 NxN 精灵模板按原始朝向覆盖到同色占位块。
#     - 变体改为：根据占位块中心落在框内哪一个象限，先对模板做 0/90/180/270 度旋转，再贴入，再做图例重着色。
#     - 象限规则固定为：左上=0，右上=1，右下=2，左下=3（这里数字是 `np.rot90` 的逆时针次数）。
#   concrete_trace:
#     - 例：某个红色占位块原本匹配一个 4x4 蓝底红形状模板。如果它在框内右上象限：
#       原题：直接贴未旋转模板，再按图例改成目标色。
#       变体：先把同一个 4x4 模板旋转 90 度，再贴到该红色占位块位置，最后整块改成目标色。
#     - 因为同一个源色会在不同槽位重复出现，所以同一个模板会在不同位置呈现不同朝向。
#   complexity_calibration:
#     - 仅靠“旋转”还不够，如果模板本身 90/180 度对称，输出和原题几乎看不出差异。
#     - 因此生成器必须额外筛掉旋转对称模板，要求四个 90 度旋转结果彼此不同。
#     - 为了让人能从少量样例归纳出规则，生成器还必须保证：
#       1. 四个象限在样例中都出现占位块；
#       2. 至少有一个源色在两个不同象限重复出现，从而能看见“同形异向”。
#   hidden_assumptions_review:
#     - 原题默认“精灵不旋转”，所以只需要记录坐标并直接平移；这里不再成立，必须在贴入前显式旋转。
#     - 原题允许只验证占位块尺寸等于模板尺寸；这里还额外隐含“模板 bbox 必须是方形”，否则 90 度旋转后尺寸会变。
#       生成器保留 N x N 设计，solver 也显式断言 `h == w`。
#     - 原题把占位块 top-left 与模板 top-left 对齐。变体仍保持这个对齐方式，只在对齐前先在同一个 NxN bbox 内旋转。
#     - 象限判定依赖“占位块中心不正好落在框中心横线/竖线”。原题不关心这一点，变体必须关心。
#       生成器因此只接受所有槽位中心都严格位于四个开象限中的 frame 布局。
#   legality:
#     - 框内仍是规则的 NxN 槽位网格，槽位之间保留 1 格背景，不会互相覆盖。
#     - 旋转不改变 bbox 尺寸，因此贴入后仍然严格落在原占位块区域内，不越界、不碰框边。
#     - 框外模板与图例的放置仍沿用 reserve + padding 的无重叠策略，因此 solver 的组件识别前提保持成立。
#   visual_observability:
#     - 旋转规则只依赖可见的空间位置，不依赖颜色 id、枚举顺序等不可见因素。
#     - 四个象限都要出现，且至少一个源色跨象限重复，这样人能从 2-3 个样例中看出：
#       “颜色决定哪种形状，图例决定最终颜色，位置决定朝向”。
#   cleanliness:
#     - 不新增额外标记，不把多个规则叠加在同一区域；视觉层级仍是“框内目标区 + 框外模板/图例”两层，保持原题整洁度。
#   color_inference:
#     - 颜色映射部分不改，仍由显式图例给出，因此新增的 D5 变化不会破坏颜色规则的可推断性。
#   generator_guarantees_for_new_main:
#     - 每个模板都是 N x N，且四个旋转彼此不同；
#     - 所有已填槽位都能被唯一归入四象限之一；
#     - 已填槽位覆盖四个象限；
#     - 至少一个源色在两个不同象限出现；
#     - 外部模板和图例仍满足原 solver 的组件匹配假设。


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


def _rotation_k_for_bbox(
    bbox: tuple[int, int, int, int], frame_h: int, frame_w: int
) -> int:
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


def _extract_task_parts(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    np.ndarray,
    list[tuple[int, tuple[int, int, int, int]]],
    dict[int, int],
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
            if bg in sub_colors or len(sub_colors) != 2:
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

    sprite_masks: dict[int, np.ndarray] = {}
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

    return bg, frame_col, crop, placeholders, color_map, sprite_masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the framed output by combining color-matched sprites, quadrant-based rotation, and a recoloring legend.

    Input structure:
    - A gray rectangular frame encloses solid-color placeholder blocks on a regular grid.
    - Outside the frame sit one sprite template per placeholder color plus compact two-color legend glyphs that map each placeholder color to its final output color.

    Rule:
    - Detect the frame, all placeholder blocks inside it, the matching-color sprite templates outside it, and the legend-based color mapping.
    - For each placeholder block, rotate its matched sprite according to the block's quadrant inside the frame: upper-left 0 deg, upper-right 90 deg, lower-right 180 deg, lower-left 270 deg.
    - Stamp the rotated sprite into the placeholder's NxN region and recolor it to the legend target color, while preserving the frame border and background gaps.

    Key insight: Color still chooses the sprite shape and the legend still chooses the final color, but the block's position inside the frame now controls the sprite's orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg, _frame_col, crop, placeholders, color_map, sprite_masks = _extract_task_parts(g)
    H, W = crop.shape

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, bbox in placeholders:
        pr0, pc0, pr1, pc1 = bbox
        mask = sprite_masks[src_col]
        sh, sw = mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )

        krot = _rotation_k_for_bbox(bbox, H, W)
        rot_mask = np.rot90(mask, krot)
        tgt_col = int(color_map[src_col])
        sprite = np.full((sh, sw), bg, dtype=int)
        sprite[rot_mask] = tgt_col
        paste(out, sprite, pr0, pc0, transparent=bg)

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
LAYOUT_RETRIES = 80
MASK_RETRIES = 200


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _rot_signature(mask: np.ndarray) -> tuple[bytes, ...]:
    return tuple(np.rot90(mask, k).tobytes() for k in range(4))


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
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
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


def _slot_bbox(
    slot: tuple[int, int], n: int, top_margin: int, left_margin: int
) -> tuple[int, int, int, int]:
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
) -> dict[tuple[int, int], int] | None:
    pairs = [
        (a, b)
        for i, a in enumerate(filled_slots)
        for b in filled_slots[i + 1 :]
        if slot_rot[a] != slot_rot[b]
    ]
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

    unused_slots = [s for s in remaining_slots if s not in cell_colors]
    for slot in unused_slots:
        cell_colors[slot] = random.choice(src_cols)

    assert set(cell_colors) == set(filled_slots)
    assert set(cell_colors.values()) == set(src_cols)
    repeated_quads = {slot_rot[s] for s, col in cell_colors.items() if col == anchor_col}
    assert len(repeated_quads) >= 2
    return cell_colors


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
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


def _variant_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    _bg, _frame_col, crop, placeholders, _color_map, sprite_masks = _extract_task_parts(g)
    H, W = crop.shape
    rotation_hist = {0: 0, 1: 0, 2: 0, 3: 0}
    changed = 0
    for src_col, bbox in placeholders:
        krot = _rotation_k_for_bbox(bbox, H, W)
        rotation_hist[krot] += 1
        if krot != 0 and not np.array_equal(np.rot90(sprite_masks[src_col], krot), sprite_masks[src_col]):
            changed += 1
    return {
        "rotation_hist": rotation_hist,
        "changed_vs_unrotated": changed,
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
    tgt_cols = [random.choice([c for c in usable if c != src]) for src in src_cols]
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n) for src in src_cols}

    cell_colors = _assign_cell_colors(filled_slots, src_cols, slot_rot)
    if cell_colors is None:
        return None

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for slot, src in cell_colors.items():
        r0, c0, _r1, _c1 = _slot_bbox(slot, n, top_margin, left_margin)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        rot_mask = np.rot90(masks[src], slot_rot[slot])
        paste(output_frame, _colored_sprite(rot_mask, color_to_tgt[src], bg), r0, c0, transparent=bg)

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
        metrics = _variant_metrics(grid)
        if metrics["changed_vs_unrotated"] <= 0:
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
