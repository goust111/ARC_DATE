# Combo variant of edb79dae (index 25)
# combo_spec: D11/3+D6/1
# Name: 多色框架编码+多色图案重着色

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, paste, random_free_location


# Design analysis:
# a) 具体化与行为追踪
#   原题流程是：
#     1. 框内实心占位块按颜色索引框外同色精灵。
#     2. 再读一个 source->target legend，把所有放入框内的精灵统一改成单色。
#   融合后改成：
#     1. 框外同色精灵仍然提供“形状”。
#     2. 每个 source 颜色在框外不再只有一个 legend，而是同时给出两种 multicolor legend：
#        - 条纹 legend
#        - 棋盘格 legend
#        二者都用同一个 source 标签列开头，右侧都是 N x N 的多色图案样本。
#     3. 框架颜色不再只是装饰，而是一个全局控制码：
#        - 蓝框(1): 精灵先顺时针旋转 90 度，再选“条纹 legend”进行重着色。
#        - 绿框(3): 精灵不旋转，改用“棋盘格 legend”进行重着色。
#   一个 4x4 的具体追踪例子：
#     - 原题：框内颜色 2 -> 取框外颜色 2 的精灵 M -> legend 说 2->7 -> 输出是单色 7 的 M。
#     - 融合蓝框：框内颜色 2 -> 取精灵 M -> 旋转成 rot90(M) -> 在颜色 2 的两个候选 legend 里，
#       选条纹 tile T_stripe -> 输出是被 rot90(M) 裁剪后的 T_stripe。
#     - 融合绿框：框内颜色 2 -> 取精灵 M -> 不旋转 -> 选棋盘格 tile T_checker -> 输出是被 M 裁剪后的 T_checker。
#   差异点不再是“纯重着色”或“纯框色控制”单独发生，而是框色同时决定几何变换和应读取哪类多色图案。
# b) 复杂度校准
#   只保留两个 frame-rule 档位，避免 ARC 训练样例过少时映射不可判定。
#   图案家族也只保留两类：vertical stripes 与 checkerboard；视觉差异稳定且不依赖颜色编号语义。
#   每个 source 同时出现两种候选 legend，防止规则退化成“只看唯一 legend 就够了”。
# c) 隐式假设审查
#   原 solver 的这些前提被打破了：
#     - legend 是两个颜色重叠得到的小双色 glyph。
#     - legend 只给一个目标色，写回时可以按像素直接填同一个标量。
#     - frame color 不参与真正决策。
#   新 solver 因此显式改成：
#     - 用 monochromatic=False 识别完整不透明 multicolor legend 矩形。
#     - 要求每个 source 对应两张 legend，且能按 tile 内容分类为条纹 / 棋盘格。
#     - 根据 frame color 先选 family，再选对应 legend tile，再裁剪到 sprite mask。
#   仍保留并由 generator 保证的前提：
#     - 每个 source 颜色在框外恰有一个同色 sprite bbox，尺寸与占位块一致。
#     - 占位块仍为方形 NxN，因此蓝框旋转后 bbox 不变。
# d) 放置合法性
#   generator 仍用 reserve 网格和 1-cell padding 放置外部对象，确保 sprite/legend/frame 互不重叠、不断开、不越界。
#   外部对象不复用任一 frame color，避免 frame/sprite/legend 解析歧义。
# e) 视觉可观察性
#   蓝框输出必须同时表现“旋转 + 条纹”；绿框必须表现“无旋转 + 棋盘格”。
#   为此 generator 拒绝：
#     - 对 90 度旋转不变的 sprite mask；
#     - 在蓝框规则下只会落到同一列奇偶类的 mask；
#     - 在绿框规则下只会落到同一棋盘奇偶类的 mask。
#   这样多色效果和 frame-coded 差异都能在少量样例中直接看到。
# f) 可视化整洁度
#   仍保持“一个框 + 框外模板与图例”的原布局，只是每个 source 多出一个候选 legend。
#   不在框内叠加额外标记；图案都在原 sprite bbox 内完成，因此输出区域仍然清晰。
# g) 变化的可推断性
#   颜色变化不靠隐式编号，而靠结构线索：
#     - 相同 source 的两个候选 legend 都可见；
#     - 框色稳定对应条纹或棋盘格；
#     - 同色占位块会重复出现，使同一 family 的图案在同一张图中多次重现。
# 9) main() 的新隐式假设与 generator 对应保证
#   - `_extract_legends` 假设 legend 是 N x (N+1) 的不透明矩形，首列全是 source 色。
#     generator 直接构造此矩形。
#   - `_classify_pattern_tile` 假设 tile 属于受限 family 集合。
#     generator 只产生 vertical/checker 两类 tile。
#   - 最终写回循环假设 frame 选择的 family 对每个 source 都存在。
#     generator 为每个 source 固定生成这两张 legend，solver 再按 frame color 选其中一张。


FRAME_RULES = {
    1: {"turns": 1, "family": "vertical"},  # blue frame
    3: {"turns": 0, "family": "checker"},   # green frame
}
FRAME_COLORS = tuple(sorted(FRAME_RULES))
PATTERN_FAMILIES = tuple(sorted({str(rule["family"]) for rule in FRAME_RULES.values()}))
RESERVE_MARK = 10

# === CFG: tunables ===
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
GEN_RETRIES = 400
MASK_RETRIES = 250


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


def _bbox_h(b: tuple[int, int, int, int]) -> int:
    return int(b[2] - b[0] + 1)


def _bbox_w(b: tuple[int, int, int, int]) -> int:
    return int(b[3] - b[1] + 1)


def _frame_turns(frame_col: int) -> int:
    assert frame_col in FRAME_RULES, f"unsupported frame color {frame_col}"
    return int(FRAME_RULES[frame_col]["turns"])


def _frame_family(frame_col: int) -> str:
    assert frame_col in FRAME_RULES, f"unsupported frame color {frame_col}"
    return str(FRAME_RULES[frame_col]["family"])


def _apply_frame_rule(mask: np.ndarray, frame_col: int) -> np.ndarray:
    turns = _frame_turns(frame_col)
    return np.rot90(mask, -turns).copy() if turns else mask.copy()


def _build_pattern_tile(shape: tuple[int, int], family: str, color_a: int, color_b: int) -> np.ndarray:
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    if family == "vertical":
        selector = cc % 2
    elif family == "checker":
        selector = (rr + cc) % 2
    else:
        raise AssertionError(f"unsupported family {family}")
    return np.where(selector == 0, int(color_a), int(color_b)).astype(int)


def _classify_pattern_tile(tile: np.ndarray) -> str | None:
    if tile.ndim != 2 or min(tile.shape) < 2:
        return None
    colors = [int(c) for c in np.unique(tile)]
    if len(colors) != 2:
        return None

    a = int(tile[0, 0])
    b = int(tile[0, 1])
    if a == b:
        return None

    vertical = _build_pattern_tile(tile.shape, "vertical", a, b)
    if np.array_equal(tile, vertical):
        return "vertical"

    checker = _build_pattern_tile(tile.shape, "checker", a, b)
    if np.array_equal(tile, checker):
        return "checker"

    return None


def _extract_legends(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    masked = np.full_like(g, bg)
    masked[outside] = g[outside]
    comps = find_components(masked, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    legend_tiles: dict[int, dict[str, np.ndarray]] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = _bbox_h((r0, c0, r1, c1))
        w = _bbox_w((r0, c0, r1, c1))
        if w != h + 1:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if comp["area"] != sub.size:
            continue
        if bg in sub or frame_col in sub:
            continue

        src_col = int(sub[0, 0])
        if src_col not in placeholder_colors:
            continue
        if phw[src_col] != (h, h):
            continue
        if not np.all(sub[:, 0] == src_col):
            continue

        tile = sub[:, 1:]
        if tile.shape != phw[src_col]:
            continue
        if src_col in set(map(int, np.unique(tile))):
            continue

        family = _classify_pattern_tile(tile)
        if family is None:
            continue

        legend_tiles.setdefault(src_col, {})
        assert family not in legend_tiles[src_col], f"duplicate {family} legend for source color {src_col}"
        legend_tiles[src_col][family] = tile.copy()
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert set(legend_tiles) == placeholder_colors, (
        f"legend sources {sorted(legend_tiles)} do not match placeholders {sorted(placeholder_colors)}"
    )
    for src_col in sorted(placeholder_colors):
        got = set(legend_tiles[src_col])
        assert got == set(PATTERN_FAMILIES), f"source color {src_col} has legend families {sorted(got)}"
    return legend_tiles, legend_mask


def _extract_sprite_masks(
    g: np.ndarray,
    outside: np.ndarray,
    legend_mask: np.ndarray,
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, np.ndarray]:
    comps = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            if all(bool(legend_mask[r, c]) for r, c in pts_tuples):
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (r0, c0, r1, c1),
                }
            )
            cid += 1

    sprite_masks: dict[int, np.ndarray] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in all_candidates
            if _bbox_h(c["bbox"]) == want_h and _bbox_w(c["bbox"]) == want_w
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

        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        for r, c in pts_set:
            mask[r - sr0, c - sc0] = True
        sprite_masks[src_col] = mask

    return sprite_masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp frame-selected patterned sprites into a framed region using outside sprite templates and paired multicolor legends.

    Input structure:
    - bg background, one rectangular non-gray frame, and solid square placeholder blocks inside that frame.
    - Outside the frame: one monochrome sprite template per placeholder color, plus two opaque N x (N+1) legends for each source color.
      Each legend starts with a uniform source-color column and its right N x N tile is either a striped or checkerboard pattern sample.

    Rule:
    - Find the frame and the placeholder blocks inside it.
    - Use placeholder color to pick the matching outside sprite shape.
    - Read both candidate legends for that source color, then let the frame color decide which legend family applies:
      blue frames select the striped legend and rotate the sprite 90 degrees clockwise; green frames select the checker legend and keep the sprite orientation.
    - Stamp the transformed sprite into every matching placeholder box, clipping the chosen legend tile by the sprite mask.

    Key insight:
    - The multicolor legends alone are not enough, because each source color deliberately has more than one candidate pattern.
      The frame color is the control code that tells you which pattern family to use and whether the sprite shape should be rotated first.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frame_col, (fr0, fc0, fr1, fc1) = _find_frame(g, bg)

    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = crop.shape
    assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

    inner_mask = np.zeros((h, w), dtype=bool)
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
        assert phw[col][0] == phw[col][1], f"expected square placeholder for color {col}, got {phw[col]}"

    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    legend_tiles, legend_mask = _extract_legends(g, outside, bg, frame_col, placeholder_colors, phw)
    sprite_masks = _extract_sprite_masks(g, outside, legend_mask, bg, frame_col, placeholder_colors, phw)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    family = _frame_family(frame_col)
    for src_col, (pr0, pc0, pr1, pc1) in placeholders:
        assert src_col in sprite_masks, f"missing sprite for color {src_col}"
        assert src_col in legend_tiles and family in legend_tiles[src_col], (
            f"missing {family} legend for color {src_col}"
        )

        mask = _apply_frame_rule(sprite_masks[src_col], frame_col)
        tile = legend_tiles[src_col][family]
        sh, sw = mask.shape
        assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
            f"placeholder size does not match transformed sprite for color {src_col}"
        )
        assert tile.shape == (sh, sw), f"legend tile shape does not match sprite for color {src_col}"

        for dr, dc in zip(*np.where(mask)):
            out[pr0 + dr, pc0 + dc] = int(tile[dr, dc])

    return out


def _legend(src: int, tile: np.ndarray) -> np.ndarray:
    src_col = np.full((tile.shape[0], 1), int(src), dtype=int)
    return np.hstack([src_col, np.asarray(tile, dtype=int)])


def _sprite_mask_is_usable(mask: np.ndarray) -> bool:
    rotated = np.rot90(mask, -1)
    if np.array_equal(mask, rotated):
        return False

    rr, cc = np.where(mask)
    if len(rr) == 0 or len(set(((r + c) % 2) for r, c in zip(rr, cc))) < 2:
        return False

    rr2, cc2 = np.where(rotated)
    if len(rr2) == 0 or len(set(int(c % 2) for c in cc2)) < 2:
        return False

    return True


def _sprite_mask(n: int) -> np.ndarray:
    lo = max((n * n + 1) // 2, int(np.ceil(SPRITE_FILL_MIN * n * n)))
    hi = min(int(np.floor(SPRITE_FILL_MAX * n * n)), n * n - 1)

    for _ in range(MASK_RETRIES):
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

        if _sprite_mask_is_usable(mask):
            return mask

    raise RuntimeError(f"failed to generate usable {n}x{n} sprite mask")


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _patterned_sprite(mask: np.ndarray, tile: np.ndarray, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = np.asarray(tile, dtype=int)[mask]
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


def _make_bordered_frame(h: int, w: int, bg: int, frame_col: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = frame_col
    frame[:, 0] = frame[:, -1] = frame_col
    return frame


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> bool:
    blocker = np.full(patch.shape, RESERVE_MARK, dtype=int)
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

    frame_col = random.choice(FRAME_COLORS)
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])
    usable = [c for c in range(10) if c not in ({bg} | set(FRAME_COLORS))]

    k = random.randint(*NUM_COLORS_RANGE)
    src_cols = random.sample(usable, k)
    masks = {src: _sprite_mask(n) for src in src_cols}

    legend_tiles: dict[int, dict[str, np.ndarray]] = {}
    for src in src_cols:
        pool = [c for c in usable if c != src]
        if len(pool) < 2:
            return None
        color_a, color_b = random.sample(pool, 2)
        legend_tiles[src] = {
            family: _build_pattern_tile((n, n), family, color_a, color_b) for family in PATTERN_FAMILIES
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
    cell_colors: dict[tuple[int, int], int] = {}
    for slot, col in zip(filled_slots[:k], src_cols):
        cell_colors[slot] = col
    for slot in filled_slots[k:]:
        cell_colors[slot] = random.choice(src_cols)

    input_frame = _make_bordered_frame(frame_h, frame_w, bg, frame_col)
    output_frame = _make_bordered_frame(frame_h, frame_w, bg, frame_col)
    active_family = _frame_family(frame_col)
    for (rr, cc), src in cell_colors.items():
        r0 = 1 + top_margin + rr * (n + 1)
        c0 = 1 + left_margin + cc * (n + 1)
        paste(input_frame, np.full((n, n), src, dtype=int), r0, c0)
        transformed = _apply_frame_rule(masks[src], frame_col)
        patterned = _patterned_sprite(transformed, legend_tiles[src][active_family], bg)
        paste(output_frame, patterned, r0, c0, transparent=bg)

    ext_area = k * (n * n + 2 * n * (n + 1) + 16)
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
    paste(reserve, np.full((frame_h, frame_w), RESERVE_MARK, dtype=int), fr0, fc0)

    patches: list[tuple[np.ndarray, bool]] = []
    for src in src_cols:
        patches.append((_colored_sprite(masks[src], src, bg), True))
        for family in PATTERN_FAMILIES:
            patches.append((_legend(src, legend_tiles[src][family]), False))
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
