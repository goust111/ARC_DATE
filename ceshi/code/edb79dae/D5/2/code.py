# Variant of edb79dae
# Dimension: D5 — 第一步变换：精灵放置方式 (transform)
# Change: 用精灵的完整形状替换框架内的单色目标像素（replace d with s.shape），精灵图案居中覆盖目标位置... → 镜像翻转后放置
#
# Design analysis
# ---------------
# variant_spec:
#   concrete_behavior:
#     - 输入仍然有一个灰框、框内规则网格状的纯色占位块、框外同色精灵模板，以及框外两色图例。
#     - 占位块不再总是直接替换成原精灵；先根据占位块 bbox 中心相对“整个框”的中心位置决定朝向：
#       左上=原样，右上=水平翻转，左下=垂直翻转，右下=同时水平+垂直翻转。
#     - “目标像素”在这里具体化为占位块整体，因为一个 NxN 占位块只对应一次精灵放置，不能对块内每个像素分别选不同翻转。
#     - 放置完成后仍按图例重着色，输出仍只保留框区域。
#   behavior_trace:
#     - 原逻辑示例：某个 4x4 蓝色占位块位于框右半区，外部蓝色模板是一个向左伸出的不对称形状。原题会把模板原样贴进框中。
#     - 新逻辑：同一个占位块若中心位于框右半区，则先水平翻转该模板，再贴入同一 4x4 bbox；若还位于下半区，则再做垂直翻转。
#     - 差异：同一种颜色在不同位置会出现同形状的镜像版本，而不是所有位置都复制同一朝向。
#   complexity_calibration:
#     - 为了让翻转肉眼可见，生成器要求精灵同时不是左右对称、也不是上下对称；否则翻转会退化成原样。
#     - 为了让人能归纳规则，生成器强制四种朝向分支都出现在每个样例中，并让至少一个颜色跨 3 个以上分支重复出现。
#     - 仍保留原题的颜色图例与“按颜色选精灵”机制，不增加新的隐藏键值。
#   implicit_assumptions_review:
#     - 原 solver 里“placeholder bbox 尺寸 == sprite bbox 尺寸”的前提仍然需要，因为镜像不会改变 bbox 尺寸。
#     - 原 solver 里“按 bbox 左上角直接贴 coords”依赖了朝向不变；这里必须先把 coords 还原为 mask，再做 flip，否则右侧/下侧位置仍会输出原朝向。
#     - 原 generator 里只要精灵 bbox 是 NxN 即可；新规则还额外依赖精灵对 flip 不是不变，因此要筛掉水平/垂直对称 mask。
#     - 颜色图例识别逻辑与 D5 无关，保留原约束：图例区域紧凑、无背景、每个 placeholder color 恰有一个映射。
#   placement_legality:
#     - 框内占位块仍是规则网格加 1 格空隙，镜像只在各自 NxN bbox 内发生，不会越界，也不会跨到别的占位块。
#     - 框外精灵与图例仍通过 `random_free_location` 分离放置，保留背景间距，避免和框重叠。
#     - 生成器在选空槽位时要求每一行、每一列、每一种翻转分支都至少保留一个填充块，保证 solver 可观测且无歧义。
#   visual_observability:
#     - 人类可从同色精灵在左右两侧的朝向对比，和在上下两侧的上下颠倒，对应推出“相对框中心决定镜像轴”的规则。
#     - 不使用颜色 id 排序或不可见元数据；控制变量只有位置分区与图例映射。
#   cleanliness:
#     - 视觉层次仍是：框内占位块 / 框外模板 / 框外图例 三类对象，彼此分离。
#     - 新增变化只发生在输出里的精灵朝向，不引入额外标记，避免画面变脏。

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


def _slot_flips(frame_h: int, frame_w: int, r0: int, c0: int, r1: int, c1: int) -> tuple[bool, bool]:
    """
    Decide vertical/horizontal flips from a placeholder bbox center relative to the frame center.
    Equality means "stay on the non-flipped side" so center-line cases remain deterministic.
    """
    frame_ctr_r = (frame_h - 1) / 2.0
    frame_ctr_c = (frame_w - 1) / 2.0
    box_ctr_r = (r0 + r1) / 2.0
    box_ctr_c = (c0 + c1) / 2.0
    flip_v = box_ctr_r > frame_ctr_r
    flip_h = box_ctr_c > frame_ctr_c
    return flip_v, flip_h


def _mask_from_points(
    pts_set: set[tuple[int, int]], sr0: int, sc0: int, height: int, width: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r, c in pts_set:
        mask[r - sr0, c - sc0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the framed region with recolored sprite templates whose orientation depends on position.

    Input structure:
    - A single-color rectangular frame encloses solid colored placeholder blocks on a bg interior.
    - Outside the frame sit one sprite template per placeholder color and compact two-color legend glyphs.

    Rule:
    - Detect the frame, the placeholder colors inside it, and the matching same-color sprite templates outside it.
    - For each placeholder block, use its bbox center relative to the frame center to choose reflection axes:
      right side means horizontal flip, bottom side means vertical flip.
    - Stamp that oriented sprite into the placeholder bbox, then recolor it using the legend mapping for that placeholder color.

    Key insight: Color still chooses which sprite and which final color to use, but spatial side-of-center now chooses the sprite's mirrored orientation.
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

        pts_set = set(base["pts"])
        for c in all_candidates:
            if c["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = c["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(c["pts"])

        sprite_masks[src_col] = _mask_from_points(pts_set, sr0, sc0, sr1 - sr0 + 1, sc1 - sc0 + 1)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_masks and src_col in color_map, (
            f"placeholder color {src_col} missing sprite or color mapping"
        )
        tgt_col = int(color_map[src_col])
        mask = sprite_masks[src_col]
        sh, sw = mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        flip_v, flip_h = _slot_flips(H, W, pr0, pc0, pr1, pc1)
        oriented = mask
        if flip_h:
            oriented = np.fliplr(oriented)
        if flip_v:
            oriented = np.flipud(oriented)
        out[pr0 : pr1 + 1, pc0 : pc1 + 1][oriented] = tgt_col

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
FOCUS_SIGS = ((False, False), (False, True), (True, False))


def _legend(src: int, tgt: int) -> np.ndarray:
    """Build a 3x4 legend block mapping src color to tgt color."""
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _raw_sprite_mask(n: int) -> np.ndarray:
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


def _sprite_mask(n: int) -> np.ndarray:
    """Generate an asymmetric NxN mask so both horizontal and vertical flips stay visible."""
    for _ in range(120):
        mask = _raw_sprite_mask(n)
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError(f"failed to build asymmetric {n}x{n} sprite")


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


def _slot_bbox(n: int, top_margin: int, left_margin: int, rr: int, cc: int) -> tuple[int, int, int, int]:
    r0 = 1 + top_margin + rr * (n + 1)
    c0 = 1 + left_margin + cc * (n + 1)
    return r0, c0, r0 + n - 1, c0 + n - 1


def _all_slot_signatures(
    frame_h: int, frame_w: int, n: int, rows: int, cols: int, top_margin: int, left_margin: int
) -> dict[tuple[int, int], tuple[bool, bool]]:
    sigs = {}
    for rr in range(rows):
        for cc in range(cols):
            sigs[(rr, cc)] = _slot_flips(frame_h, frame_w, *_slot_bbox(n, top_margin, left_margin, rr, cc))
    return sigs


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int, slot_sigs: dict[tuple[int, int], tuple[bool, bool]]
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    """
    Choose empty slots while keeping every row, every column, and every reflection branch represented.

    Returns (empty_slots, shuffled filled_slots).
    """
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    need_sigs = set(slot_sigs.values())
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) != rows:
            continue
        if len({c for _, c in filled}) != cols:
            continue
        if {slot_sigs[s] for s in filled} != need_sigs:
            continue
        valid.append(empty)
    if not valid:
        return None
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
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, bg: int, *, transparent: bool
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


def _oriented_mask(mask: np.ndarray, flip_v: bool, flip_h: bool) -> np.ndarray:
    out = mask
    if flip_h:
        out = np.fliplr(out)
    if flip_v:
        out = np.flipud(out)
    return out


def _assign_cell_colors(
    filled_slots: list[tuple[int, int]],
    slot_sigs: dict[tuple[int, int], tuple[bool, bool]],
    src_cols: list[int],
) -> dict[tuple[int, int], int] | None:
    by_sig: dict[tuple[bool, bool], list[tuple[int, int]]] = {}
    for slot in filled_slots:
        by_sig.setdefault(slot_sigs[slot], []).append(slot)
    if any(not by_sig.get(sig) for sig in FOCUS_SIGS):
        return None

    for slots in by_sig.values():
        random.shuffle(slots)

    focus = src_cols[0]
    cell_colors: dict[tuple[int, int], int] = {}
    used: set[tuple[int, int]] = set()

    for sig in FOCUS_SIGS:
        slot = next(s for s in by_sig[sig] if s not in used)
        cell_colors[slot] = focus
        used.add(slot)

    leftovers = [slot for slot in filled_slots if slot not in used]
    random.shuffle(leftovers)
    if len(leftovers) < len(src_cols) - 1:
        return None

    for col in src_cols[1:]:
        slot = leftovers.pop()
        cell_colors[slot] = col

    extra_focus_sig = (True, True)
    if leftovers and by_sig.get(extra_focus_sig):
        candidates = [slot for slot in by_sig[extra_focus_sig] if slot not in used and slot in leftovers]
        if candidates and random.random() < 0.75:
            slot = candidates[0]
            cell_colors[slot] = focus
            leftovers.remove(slot)

    for slot in leftovers:
        if random.random() < 0.45:
            cell_colors[slot] = focus
        else:
            cell_colors[slot] = random.choice(src_cols)

    return cell_colors


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, total = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    k_max = min(NUM_COLORS_RANGE[1], len(usable), total - 2)
    if k_max < NUM_COLORS_RANGE[0]:
        return None
    k = random.randint(NUM_COLORS_RANGE[0], k_max)
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

    slot_sigs = None
    top_margin = left_margin = None
    for _ in range(40):
        cand_top = random.randint(1, y_margin - 1)
        cand_left = random.randint(1, x_margin - 1)
        cand_sigs = _all_slot_signatures(frame_h, frame_w, n, rows, cols, cand_top, cand_left)
        if set(cand_sigs.values()) == {(False, False), (False, True), (True, False), (True, True)}:
            top_margin = cand_top
            left_margin = cand_left
            slot_sigs = cand_sigs
            break
    if slot_sigs is None or top_margin is None or left_margin is None:
        return None

    chosen_slots = _choose_empty_slots(rows, cols, empty_count, slot_sigs)
    if chosen_slots is None:
        return None
    _, filled_slots = chosen_slots
    cell_colors = _assign_cell_colors(filled_slots, slot_sigs, src_cols)
    if cell_colors is None:
        return None

    input_frame = _make_bordered_frame(frame_h, frame_w, bg)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg)
    for (rr, cc), src in cell_colors.items():
        r0, c0, r1, c1 = _slot_bbox(n, top_margin, left_margin, rr, cc)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        flip_v, flip_h = slot_sigs[(rr, cc)]
        out_mask = _oriented_mask(masks[src], flip_v, flip_h)
        paste(output_frame, _colored_sprite(out_mask, color_to_tgt[src], bg), r0, c0, transparent=bg)

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
    for src, tgt in zip(src_cols, tgt_cols, strict=False):
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
