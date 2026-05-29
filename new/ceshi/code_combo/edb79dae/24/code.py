# Combo variant of edb79dae (index 24)
# combo_spec: D3/0+D6/0+D13/1
# Name: 多色精灵+部分重着色+不完整图例

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


"""
design_analysis:
  fused_rule:
    original_pipeline:
      - 框内每个纯色占位块按颜色找到框外同色单色 sprite
      - 把 sprite 复制进框内后，若图例存在则整块统一改成目标色
      - 原题默认每个占位色一定有图例
    fused_pipeline:
      - 框外 sprite 变成双色 patch，主色面积更大、次色面积更小
      - 占位块仍然只用主色匹配 sprite；复制时保留整个双色图案
      - 只有有图例的主色才会被重着色，而且只改 sprite 中等于主色的像素
      - 没有图例的颜色完全不重着色，主色和次色都保留
    concrete_trace:
      - 例 1：占位色 2 对应框外 sprite {2,8}，其中 2 占 7 格、8 占 3 格，且有图例 2->4
      - 原题会输出整只 sprite 统一变成 4
      - 融合后输出为：原来是 2 的格子变成 4，原来是 8 的格子保持 8
      - 例 2：占位色 3 对应框外 sprite {3,9}，但框外没有 3 的图例
      - 原题不允许这种情况；融合后输出直接保留 {3,9} 不变
      - 差异由三种信息共同决定：sprite 主色、图例是否存在、被改的只是主色像素而非整只 sprite
  complexity_calibration:
    - 每个样例强制同时存在 mapped 与 unmapped 两类占位色，否则 D13 分支不可见
    - 每只 sprite 恰好两种颜色，且主色严格占优，避免“哪个颜色用于匹配”产生歧义
    - 同一样例内不同占位色使用互异 mask，方便对照“同一形状在有/无图例时的视觉差异”
    - 目标色不等于主色，也不等于该 sprite 的次色，避免“已重着色”和“未重着色”混淆
    - 至少有一个 mapped 颜色和一个 unmapped 颜色各自重复出现一次，保证两种分支都不是单例偶然现象
  implicit_assumption_audit:
    - 原 solver 用单色连通分量按颜色收集 sprite；融合后这会把双色 sprite 拆碎，因此必须先排除 legend，
      再在整体非背景掩码上按 4 连通提取完整双色 patch
    - 原 solver 断言 color_map 覆盖全部 placeholder_colors；D13/1 使这条断言失效，最终着色必须改成
      `if src in color_map: only recolor primary pixels else keep patch unchanged`
    - 原 solver 最终逐坐标写入单一 tgt 色；D3/0 + D6/0 使这一前提失效，必须粘贴完整 patch 并只替换主色值
    - legend 检测仍沿用“紧凑、无背景、双色、bbox 相互重叠”的思路；生成器保证双色 sprite 的 bbox 中保留背景孔洞，
      因而不会伪装成 legend
  placement_legality:
    - frame 先放入画布并在 reserve 中占满整个 bbox
    - 框外 sprite 和 legend 继续用 `random_free_location(..., padding=1)` 放置，彼此不重叠、不越界
    - reserve 以完整矩形 bbox 占位，所以即使 sprite 自身有背景孔洞，也不会与其他对象穿插
    - 每个占位色都保证恰有一个同尺寸双色 sprite；legend 只放严格子集，不影响 sprite 唯一匹配
  observability_and_cleanliness:
    - 输入里能直接看到：框外有双色 sprite，但只有部分主色拥有图例块
    - 输出里能直接对比：mapped sprite 变成“目标色主色 + 原次色”，unmapped sprite 保持“原主色 + 原次色”
    - 框内仍只显示单个 framed region，框外对象之间有背景间隔，视觉角色清晰
    - 规则不依赖颜色编号大小、扫描顺序等不可视因素；唯一用到的排序是面积主次，而面积差在像素上可见
  solver_assumptions_enforced_by_generator:
    - 每个双色 sprite 都是一个完整的 4 连通 patch，且主色唯一占优
    - 每个占位色在框内至少出现一次；至少一个 mapped 色和一个 unmapped 色各自出现至少两次
    - legend 若存在，仍是原题 3x4 固定双色块
    - 每个目标样例都通过 `main(input) == oracle_output` 验证后才返回
"""


def _find_frame(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
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


def _extract_placeholders(
    crop: np.ndarray, bg: int, frame_col: int
) -> tuple[list[tuple[int, tuple[int, int, int, int]]], dict[int, tuple[int, int]], set[int]]:
    h, w = crop.shape
    inner_mask = np.zeros((h, w), dtype=bool)
    inner_mask[1:-1, 1:-1] = True
    placeholder_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
    assert placeholder_colors, "no placeholder colors found inside frame"

    placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
    for col in sorted(placeholder_colors):
        for pts in components_mask((crop == col) & inner_mask, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            assert np.all(crop[r0 : r1 + 1, c0 : c1 + 1] == col), f"placeholder color {col} is not a solid block"
            placeholders.append((col, (r0, c0, r1, c1)))

    phw: dict[int, tuple[int, int]] = {}
    for col in placeholder_colors:
        dims = [(b[2] - b[0] + 1, b[3] - b[1] + 1) for c, b in placeholders if c == col]
        assert dims, f"no placeholder component found for color {col}"
        unique_dims = sorted(set(dims))
        assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
        phw[col] = unique_dims[0]
    return placeholders, phw, placeholder_colors


def _collect_monochrome_components(
    g: np.ndarray, outside: np.ndarray, bg: int, frame_col: int
) -> list[dict]:
    comps = []
    cid = 0
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
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
    return comps


def _detect_legends(
    g: np.ndarray,
    comps: list[dict],
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
) -> tuple[dict[int, int], np.ndarray]:
    color_map: dict[int, int] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

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
            legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    return color_map, legend_mask


def _extract_sprite_patches(
    g: np.ndarray,
    outside: np.ndarray,
    legend_mask: np.ndarray,
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    sprite_mask = outside & ~legend_mask & (g != bg) & (g != frame_col)
    sprite_candidates = []
    for pts in components_mask(sprite_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        colors = [int(c) for c in np.unique(patch) if int(c) not in (bg, frame_col)]
        assert len(colors) == 2, f"expected two colors in each sprite, found {colors}"
        counts = {col: int(np.sum(patch == col)) for col in colors}
        ordered = sorted(colors, key=lambda col: counts[col], reverse=True)
        primary, secondary = ordered[0], ordered[1]
        assert counts[primary] > counts[secondary], f"sprite colors must have a unique dominant color: {counts}"
        sprite_candidates.append(
            {
                "primary": int(primary),
                "secondary": int(secondary),
                "dims": (r1 - r0 + 1, c1 - c0 + 1),
                "patch": patch,
            }
        )

    sprite_patches: dict[int, np.ndarray] = {}
    for src_col in sorted(placeholder_colors):
        want_h, want_w = phw[src_col]
        matched = [
            c for c in sprite_candidates if c["primary"] == src_col and c["dims"] == (want_h, want_w)
        ]
        assert len(matched) == 1, (
            f"expected one two-color sprite for placeholder color {src_col}, found {len(matched)}"
        )
        sprite_patches[src_col] = matched[0]["patch"]

    assert set(sprite_patches) == placeholder_colors, "not all placeholder colors received a sprite"
    return sprite_patches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy two-color sprites into the frame, but recolor only their dominant-color pixels when a legend entry exists.

    Input structure:
    - One rectangular gray frame contains solid-color placeholder blocks laid out on the background.
    - Outside the frame there is one connected two-color sprite for every placeholder color; the larger-area color is
      the sprite's primary color and matches the placeholder.
    - Also outside the frame are compact two-color legend glyphs for only some placeholder colors, so the legend set is
      intentionally incomplete.

    Rule:
    - Read the placeholder colors inside the frame and recover the optional legend mapping from source color to target
      color.
    - For each placeholder color, find the outside two-color sprite whose dominant color matches that placeholder, then
      paste the full sprite patch into every placeholder position.
    - If the placeholder color has a legend entry, recolor only the pasted pixels that were in the sprite's primary
      color to the legend target; if no legend exists, leave both colors of that sprite unchanged.

    Key insight: Three cues must be combined: dominant color selects which multicolor sprite belongs to a placeholder,
    the minority accent color is never recolored, and missing legend entries are meaningful because they tell you to
    keep the sprite's primary color instead of changing it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    placeholders, phw, placeholder_colors = _extract_placeholders(crop, bg, frame_col)

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _collect_monochrome_components(g, outside, bg, frame_col)
    color_map, legend_mask = _detect_legends(g, comps, bg, frame_col, placeholder_colors)
    sprite_patches = _extract_sprite_patches(g, outside, legend_mask, bg, frame_col, placeholder_colors, phw)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        patch = sprite_patches[src_col].copy()
        if src_col in color_map:
            # Mapped branch: the legend overrides only the dominant source-color cells.
            patch[patch == src_col] = int(color_map[src_col])
        # Unmapped branch: both colors remain unchanged because the legend is incomplete.
        sh, sw = patch.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match sprite for color {src_col}"
        )
        paste(out, patch, pr0, pc0, transparent=bg)

    return out


# --------------- generator (o5x) ---------------

# === CFG: tunables ===
FRAME_COL = 5
SPRITE_N_RANGE = (3, 6)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (3, 5)
TOTAL_BLOCKS_RANGE = (6, 10)
MAX_EMPTY_SLOTS = 2
NUM_COLORS_RANGE = (2, 4)
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
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _sprite_mask(n: int) -> np.ndarray:
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


def _secondary_mask(mask: np.ndarray) -> np.ndarray:
    filled = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    area = len(filled)
    assert area >= 5, f"sprite mask too small for two colors: {area}"

    max_secondary = max(2, min((area - 1) // 2, int(np.floor(0.4 * area))))
    target = random.randint(2, max_secondary)
    chosen = {random.choice(filled)}

    while len(chosen) < target:
        frontier = []
        for r, c in chosen:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and (rr, cc) not in chosen:
                    frontier.append((rr, cc))
        if not frontier:
            frontier = [cell for cell in filled if cell not in chosen]
        chosen.add(random.choice(frontier))

    out = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        out[r, c] = True
    return out


def _multicolor_sprite(mask: np.ndarray, primary: int, accent: int, bg: int) -> np.ndarray:
    """Render a two-color sprite whose primary color remains the unique dominant color."""
    secondary = _secondary_mask(mask)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(primary)
    out[secondary] = int(accent)
    assert int(np.sum(out == primary)) > int(np.sum(out == accent)), "primary color must dominate"
    return out


def _recolor_primary(patch: np.ndarray, primary: int, target: int) -> np.ndarray:
    """Recolor only the primary-color pixels of a two-color sprite patch."""
    out = patch.copy()
    out[out == primary] = int(target)
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


def _choose_empty_slots(
    rows: int, cols: int, empty_count: int
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
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


def _choose_legend_subset(src_cols: list[int]) -> tuple[set[int], set[int]]:
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
    counts = {src: 1 for src in src_cols}
    remaining = len(filled_slots) - len(src_cols)
    assert remaining >= 2, "need at least two repeated slots to exhibit mapped and unmapped behaviors"

    anchor_mapped = random.choice(sorted(mapped_srcs))
    anchor_unmapped = random.choice(sorted(unmapped_srcs))
    counts[anchor_mapped] += 1
    counts[anchor_unmapped] += 1
    remaining -= 2

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

    max_k = min(NUM_COLORS_RANGE[1], total - 2, len(usable) - 1)
    min_k = NUM_COLORS_RANGE[0]
    if max_k < min_k:
        return None
    k = random.randint(min_k, max_k)
    src_cols = random.sample(usable, k)

    masks = _unique_sprite_masks(src_cols, n)
    if masks is None:
        return None

    accent_pool = [c for c in usable if c not in src_cols]
    if len(accent_pool) < 2:
        return None
    accent_cols = {src: random.choice(accent_pool) for src in src_cols}

    mapped_srcs, unmapped_srcs = _choose_legend_subset(src_cols)
    color_to_tgt = {}
    for src in mapped_srcs:
        choices = [c for c in accent_pool if c != accent_cols[src]]
        if not choices:
            return None
        color_to_tgt[src] = random.choice(choices)

    sprites = {
        src: _multicolor_sprite(masks[src], src, accent_cols[src], bg)
        for src in src_cols
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
        rendered = sprites[src] if src not in color_to_tgt else _recolor_primary(sprites[src], src, color_to_tgt[src])
        paste(output_frame, rendered, r0, c0, transparent=bg)

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
        patches.append((sprites[src], True))
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

    counts = {src: sum(1 for color in cell_colors.values() if color == src) for src in src_cols}
    assert mapped_srcs and unmapped_srcs
    assert any(counts[src] >= 2 for src in mapped_srcs)
    assert any(counts[src] >= 2 for src in unmapped_srcs)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")
