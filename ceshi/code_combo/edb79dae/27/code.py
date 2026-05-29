# Combo variant of edb79dae (index 27)
# combo_spec: D7/3+D2/0
# Name: 2x2颜色矩阵图例+面积匹配
#
# design_analysis:
#   a) concretization_and_trace:
#      - 保留原任务第一段骨架：框内仍是规则网格中的纯色 placeholder；框外仍有每种 placeholder 色对应的一份 sprite 模板。
#      - 原题第二段是“看提到该源色的 legend，再读出目标色”；这里改成纯面积驱动，legend 不再提源色。
#      - 融合后的 legend 由若干横向短条组成。每个短条都是同一个 2x2 颜色矩阵
#        [[code, tgt],
#         [code, tgt]]
#        沿水平方向重复若干次形成的。
#      - 具体例子：若某个蓝色 placeholder 对应的外部 sprite 只有 6 个前景像素，那么它要匹配“重复 6 次 2x2 矩阵”的那条 legend。
#        在这条 legend 里，左列总是共享计数色，右列总是目标色；因此先由 2x2 的列位置读出“哪一列是计数、哪一列是输出”，
#        再数重复次数 = 6，最后把蓝色 sprite 重着色成该条右列显示的颜色。
#      - 与原题的根本差异是：原题第二步是“源色 -> legend 条目”；这里是“源色 -> sprite 面积 -> legend 条目”，
#        而 legend 本身还必须按 2x2 矩阵的左右列语义来读。
#   b) complexity_calibration:
#      - 若允许太多颜色和太大面积，2x2 条纹图例会占据过多画布；因此把颜色数控制在 2-3，面积控制为 5-7，既能看出计数差异，又不至于拥挤。
#      - 所有 sprite 面积都强制互异，否则面积匹配会退化为歧义。
#      - 所有 target 色互异且与 placeholder 色、共享计数色都分离，避免出现“看颜色对就能猜”的捷径。
#   c) implicit_assumptions_review:
#      - 原 solver 用“双色紧凑重叠块 + 哪个颜色面积更大”去解 legend，这在融合后完全失效，必须改成识别 2 行高、双色、不含背景、
#        且按列交替的横向矩阵条。
#      - 原 sprite 恢复逻辑默认至少有一个同色 4 连通分量跨满整块 bbox；融合后 sprite 面积被压低到 5-7，这个前提不再可靠。
#        因此这里改成：排除 legend 后，把某个源色在 frame 外的全部剩余像素并起来，整体求 sprite bbox。
#      - 这又引入一个新前提：frame 外某个 placeholder 色只能出现在那一个 sprite 里，不能混入 legend。generator 会显式保证。
#      - legend 解析还依赖“所有 legend 条目共享同一个 code 色”；generator 也会显式保证，并让该颜色不属于 placeholder/target。
#   d) placement_legality:
#      - frame、legend、sprite 都继续用 reserve 网格做一次性放置，保证不重叠、不越界，且彼此至少隔 1 格背景。
#      - 每条 legend 作为不透明矩形 patch 放置，因此其内部不会被别的 sprite 塞入，避免伪造交替条纹。
#   e) visual_observability:
#      - 输入里能肉眼看到两种稳定规律：一是所有 legend 都由相同 2x2 矩阵重复组成；二是不同 legend 的重复次数不同。
#      - 输出里能看到：不同源色 placeholder 最终不按源色本身决定目标色，而是按 sprite 的像素多少落到不同短条。
#      - 这要求解题者同时归纳“左右列语义”和“重复次数=面积”两个线索，融合特征明显。
#   f) visual_cleanliness:
#      - frame 内依旧只有 placeholder 网格；frame 外只有 sprite 和成组条纹图例，两类对象的纹理差异很大，不会混淆。
#      - sprite 保留背景孔洞，而 legend 是完全不透明的 2 行短条，视觉职责清楚。
#   g) color_change_inferability:
#      - 颜色变化不是任意的：每个 target 色都直接出现在某条 2x2 矩阵的右列。
#      - 决定选哪条的结构线索也明确：先看外部 sprite 的前景像素数，再找重复次数相同的 legend。
#
# solver_assumption_audit:
#   - `_find_frame` 仍要求唯一矩形框；generator 保持单框。
#   - placeholder 提取仍要求同色 placeholder 共享同一 bbox 尺寸；generator 保持全局同一个 N。
#   - legend 识别要求每条 legend:
#       1. 在 frame 外；
#       2. 是高 2、宽为偶数的纯双色不透明矩形；
#       3. 两行完全相同；
#       4. 偶数列全是共享 code 色，奇数列全是该条 target 色。
#     generator 用 `_legend_strip()` 精确生成这种形态。
#   - sprite 提取要求某个 placeholder 色在 frame 外且不属于 legend 的所有像素只对应一份 sprite；
#     generator 令 legend 完全不使用 placeholder 色，并且每种 placeholder 色只放一个 sprite。
#   - 面积匹配要求不同 sprite 面积互异，并且 legend 的重复次数集合与这些面积完全一致；
#     generator 通过唯一采样面积并在 oracle 检查里验证。

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 3)
SPRITE_AREA_VALUES = (5, 6, 7)
MIN_FRAME_H = 13
MIN_FRAME_W = 18
MAX_FRAME_DIM = 21
FRAME_MARGIN_SLACK = 4
CANVAS_H_RANGE = (22, 31)
CANVAS_W_RANGE = (23, 31)
FRAME_POS_JITTER_PROB = 0.25
PATCH_PADDING = 1
GEN_RETRIES = 400


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


def _legend_strip(area: int, tgt: int, code_col: int) -> np.ndarray:
    """Repeat the 2x2 matrix [[code,tgt],[code,tgt]] exactly `area` times."""
    patch = np.empty((2, area * 2), dtype=int)
    patch[:, 0::2] = int(code_col)
    patch[:, 1::2] = int(tgt)
    return patch


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


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    """Render a binary mask as a colored sprite on bg."""
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _random_bbox_cover_points(n: int) -> set[tuple[int, int]]:
    """Pick points that guarantee the union bbox spans the whole NxN sprite box."""
    pts = {
        (0, random.randrange(n)),
        (n - 1, random.randrange(n)),
        (random.randrange(n), 0),
        (random.randrange(n), n - 1),
    }
    return pts


def _sprite_mask(n: int, target: int) -> np.ndarray:
    """Generate an NxN mask with exactly `target` pixels and full NxN union bbox."""
    assert 4 <= target < n * n, f"target area {target} incompatible with sprite box {n}"
    pts = _random_bbox_cover_points(n)
    while len(pts) > target:
        pts = _random_bbox_cover_points(n)

    while len(pts) < target:
        frontier = set()
        for r, c in pts:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and (rr, cc) not in pts:
                    frontier.add((rr, cc))
        empties = [(r, c) for r in range(n) for c in range(n) if (r, c) not in pts]
        if frontier and random.random() < 0.75:
            pts.add(random.choice(sorted(frontier)))
        else:
            pts.add(random.choice(empties))

    mask = np.zeros((n, n), dtype=bool)
    for r, c in pts:
        mask[r, c] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _collect_outside_components(
    g: np.ndarray, outside: np.ndarray, bg: int, frame_col: int
) -> list[dict[str, object]]:
    """Collect 4-connected components from all non-bg, non-frame cells outside the frame."""
    comps = []
    cid = 0
    dense_outside = outside & (g != bg) & (g != frame_col)
    for pts in components_mask(dense_outside, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        pts_tuples = [(int(r), int(c)) for r, c in pts]
        colors_here = sorted(set(int(g[r, c]) for r, c in pts_tuples))
        comps.append(
            {
                "id": cid,
                "bbox": (r0, c0, r1, c1),
                "pts": pts_tuples,
                "colors": colors_here,
            }
        )
        cid += 1
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the frame's solid placeholders with external sprites, then recolor them by an area-coded 2x2 legend.

    Input structure:
    - One gray rectangular frame sits on a background canvas; inside it are solid NxN placeholder blocks on a regular grid.
    - Outside the frame there is one sprite template per placeholder color.
    - Outside the frame there are also several opaque legend strips. Each strip is built by repeating the same 2x2 color matrix
      horizontally: left column = shared counting color, right column = that strip's output color.

    Rule:
    - Find the frame, the placeholder blocks, and the same-color sprite template for each placeholder color.
    - Recover each sprite shape and count its foreground pixels.
    - Read each legend strip as a run of identical 2x2 matrices; its repetition count is the numeric code, and the right column
      gives the output color.
    - Match a sprite to the legend strip whose repetition count equals that sprite's area, then stamp the sprite into every
      matching placeholder position recolored to that strip's output color.

    Key insight: The legend is no longer keyed by source color or by sprite shape. You must first decode the 2x2 matrix's column
    roles, then use the number of repeated matrices as an area code, and only then read off the final color.
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

    comps = _collect_outside_components(g, outside, bg, frame_col)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_candidates = []

    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != 2 or w < 2 or w % 2 != 0:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg) or np.any(sub == frame_col):
            continue
        colors_here = sorted(set(map(int, np.unique(sub))))
        if len(colors_here) != 2:
            continue
        if not np.array_equal(sub[0], sub[1]):
            continue
        left_cols = list(map(int, sub[0, 0::2]))
        right_cols = list(map(int, sub[0, 1::2]))
        if len(set(left_cols)) != 1 or len(set(right_cols)) != 1:
            continue
        if left_cols[0] == right_cols[0]:
            continue
        legend_candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "left": int(left_cols[0]),
                "right": int(right_cols[0]),
                "area": w // 2,
            }
        )
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert len(legend_candidates) == len(placeholder_colors), (
        f"expected {len(placeholder_colors)} legend strips, found {len(legend_candidates)}"
    )
    common_left = set.intersection(*({cand["left"], cand["right"]} for cand in legend_candidates))
    assert len(common_left) == 1, f"expected one shared legend code color, found {sorted(common_left)}"
    code_col = int(next(iter(common_left)))
    assert code_col not in placeholder_colors, "legend code color must not also be a placeholder color"

    area_to_tgt: dict[int, int] = {}
    for cand in legend_candidates:
        left = int(cand["left"])
        right = int(cand["right"])
        assert left == code_col, "legend strips must expose the shared code color in the left column"
        tgt = right
        area = int(cand["area"])
        assert area not in area_to_tgt, f"duplicate legend area {area}"
        area_to_tgt[area] = tgt

    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    sprite_areas: dict[int, int] = {}
    for src_col in sorted(placeholder_colors):
        pts = np.argwhere((g == src_col) & outside & ~legend_mask)
        assert len(pts) > 0, f"no sprite pixels found for color {src_col}"
        sr0, sc0, sr1, sc1 = bbox_coords(pts)
        sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
        assert (sh, sw) == phw[src_col], f"sprite bbox size mismatch for color {src_col}: {(sh, sw)} vs {phw[src_col]}"
        coords = tuple(sorted((int(r - sr0), int(c - sc0)) for r, c in pts))
        sprite_coords[src_col] = (coords, sh, sw)
        sprite_areas[src_col] = len(coords)

    assert len(set(sprite_areas.values())) == len(sprite_areas), "sprite areas must be unique"
    assert set(area_to_tgt) == set(sprite_areas.values()), (
        f"legend areas {sorted(area_to_tgt)} do not match sprite areas {sorted(set(sprite_areas.values()))}"
    )

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        coords, sh, sw = sprite_coords[src_col]
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        sprite_area = sprite_areas[src_col]
        assert sprite_area in area_to_tgt, f"no legend entry for sprite area {sprite_area}"
        tgt_col = int(area_to_tgt[sprite_area])
        for dr, dc in coords:
            out[pr0 + dr, pc0 + dc] = tgt_col

    return out


def _gen_attempt() -> np.ndarray | None:
    struct = _choose_structure()
    if struct is None:
        return None
    n, rows, cols, empty_count, _ = struct

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    area_values = [a for a in SPRITE_AREA_VALUES if a < n * n]
    if len(area_values) < NUM_COLORS_RANGE[0]:
        return None

    max_k = min(NUM_COLORS_RANGE[1], len(area_values), (len(usable) - 1) // 2)
    if max_k < NUM_COLORS_RANGE[0]:
        return None
    k = random.randint(NUM_COLORS_RANGE[0], max_k)

    areas = random.sample(area_values, k)
    code_col = random.choice(usable)
    src_pool = [c for c in usable if c != code_col]
    src_cols = random.sample(src_pool, k)
    tgt_pool = [c for c in usable if c not in set(src_cols) | {code_col}]
    if len(tgt_pool) < k:
        return None
    tgt_cols = random.sample(tgt_pool, k)

    color_to_area = dict(zip(src_cols, areas))
    color_to_tgt = dict(zip(src_cols, tgt_cols))
    masks = {src: _sprite_mask(n, color_to_area[src]) for src in src_cols}

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

    legend_patches = [_legend_strip(color_to_area[src], color_to_tgt[src], code_col) for src in src_cols]
    sprite_patches = [_colored_sprite(masks[src], src, bg) for src in src_cols]

    ext_area = sum(p.shape[0] * p.shape[1] for p in legend_patches + sprite_patches) + 12 * k
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

    patches: list[tuple[np.ndarray, bool]] = [(patch, False) for patch in legend_patches]
    patches += [(patch, True) for patch in sprite_patches]
    random.shuffle(patches)
    patches.sort(key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
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
