# Variant of b10624e5
# Dimension: D3 — 目标对象：其他象限的红色锚块定位 (dst)
# Change: 其他象限中红色锚块的相邻位置作为目标，图案根据锚块尺寸和象限关系进行缩放和反射后放置... → 图案平铺填满目标象限空余区域

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D3 tiling variant
# a) Concrete behavior:
#    - 原题在每个非 TL 象限只放置一次缩放+镜像后的图案，位置由该象限的红锚块决定。
#    - 变体中，先求出这一次对齐后的“基准砖块”位置，再以砖块高宽为步长，在该目标象限内做二维周期平铺。
#    - 平铺对象仍然是缩放+镜像后的整块图案，但重复砖块中的红像素全部跳过，只保留输入里本来就存在的那个红锚块。
#      否则会在目标象限中制造额外红方块，破坏“每象限一个锚块”的可读性并引入无关歧义。
#    - 小例子：若某目标象限的缩放图案是 4x6，锚块对齐后基准砖块左上角在 (r,c)，
#      原题只会画一次 [r:r+4, c:c+6]；本变体则会继续画 (r±4k, c±6m) 的所有同尺寸砖块，
#      并把落在象限边界外的部分裁掉。
# b) Complexity calibration:
#    - 保留原题的四象限、锚块尺寸编码缩放比例、象限位置编码镜像轴，不引入额外对象类型。
#    - 仅增加“重复放置直到铺满象限”的目标区域行为，变化显著但规则仍单一。
#    - 源图案继续要求每行每列都有至少一个非背景像素，这样平铺后肉眼能看出周期结构，不会退化成零散噪声。
# c) Original-code assumptions that break under the variant:
#    - 原 `for` 循环里按像素把 `patch_s` 只写一次，默认“目标对象=一次贴靠后的局部区域”。
#      变体下这个前提失效，必须改成遍历整个平铺晶格。
#    - 原代码允许复制图案中的红像素，因为只贴一次时红区恰好与目标锚块重合；
#      变体若直接复用会在平铺副本里产生额外红锚块，所以必须显式跳过所有重复砖块里的红像素。
#    - 原 generator 的 `_choose_anchor` 偏向“整块 motif 尽量完整放进象限”；变体不再需要这个约束，
#      因为裁剪是规则的一部分，锚块只需自身合法即可。
# d) Placement legality:
#    - 每个象限仍然只放一个红色正方形锚块，且全部落在各自 13x13 象限内。
#    - TL 源图案完整放在 TL 象限内；非 TL 输出通过象限内裁剪平铺，不跨越分隔线。
#    - 生成器先构造完整输出，再清空非 TL 象限仅保留红锚块，最后用 solver 回验。
# e) Visual observability:
#    - 与原题相比，非 TL 象限会出现明显的重复纹理，而不是单一一次性贴靠图案。
#    - 仍然保留左右/上下镜像和按锚块尺寸缩放的线索；解题者需要同时发现“先变换，再整区平铺”。
# f) Visual cleanliness:
#    - 通过跳过平铺副本中的红色，仅保留每象限一个锚块，避免同一区域出现多个红色定位块造成混乱。
#    - 平铺限制在各自象限边界内，分隔十字保持清晰。
# g) Color inferability:
#    - 不引入新的颜色映射；颜色完全继承 TL 图案本身，只做缩放、镜像和重复。
#    - 规则基于可见几何关系，不依赖颜色 id 的排序或隐藏属性。
# h) New solver assumptions and generator guarantees:
#    - solver 假设非 TL 象限的输入只有红锚块，没有其它非背景像素；generator 显式清空后再放回锚块。
#    - solver 假设 TL 图案与红锚块属于同一个 4 连通分量；generator 只产生满足该条件的源图案。
#    - solver 假设缩放后的图案仍含红像素且砖块尺寸为正；源锚块固定 2x2，目标锚块大小取 1..4，generator 保证成立。


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider: (mid_row, mid_col)."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize supporting non-integer ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, source_size: int, target_size: int) -> np.ndarray:
    """Reflect and scale the source patch for one target quadrant."""
    p = patch
    if qr == 1:
        p = np.fliplr(p)
    if qb == 1:
        p = np.flipud(p)
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    patch_s = _resize_nn(p, new_h, new_w)
    roundtrip = _resize_nn(patch_s, p.shape[0], p.shape[1])
    assert np.array_equal(roundtrip, p), "scaling must preserve all motif information"
    return patch_s


def _tile_quadrant(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    q_r0: int,
    q_r1: int,
    q_c0: int,
    q_c1: int,
    bg: int,
) -> None:
    """
    Tile one transformed/scaled patch across a quadrant.

    The tile lattice is anchored by aligning the patch's red bbox to the target red anchor.
    All repeated red pixels are skipped; the existing anchor already supplies the only red block.
    """
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    tile_h, tile_w = patch_s.shape
    src_rr, src_cc = np.where((patch_s != bg) & (patch_s != RED))
    vals = patch_s[src_rr, src_cc]
    if src_rr.size == 0:
        return

    start_r = place_r0
    while start_r + tile_h - 1 >= q_r0:
        start_r -= tile_h
    start_r += tile_h

    start_c = place_c0
    while start_c + tile_w - 1 >= q_c0:
        start_c -= tile_w
    start_c += tile_w

    for tile_r0 in range(start_r, q_r1 + 1, tile_h):
        for tile_c0 in range(start_c, q_c1 + 1, tile_w):
            tgt_rr = src_rr + tile_r0
            tgt_cc = src_cc + tile_c0
            valid = (
                (tgt_rr >= q_r0)
                & (tgt_rr <= q_r1)
                & (tgt_cc >= q_c0)
                & (tgt_cc <= q_c1)
            )
            if np.any(valid):
                out[tgt_rr[valid], tgt_cc[valid]] = vals[valid]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a motif across quadrants, then tile the transformed motif through each target quadrant.

    Input structure:
    - Grid is split into four 13x13 quadrants by one full non-background row and one full non-background column.
    - Every quadrant contains exactly one red square anchor block, and the TL anchor is the source-size reference.
    - Only the TL quadrant contains the full colored motif; the other quadrants contain just their red anchors.

    Rule:
    - Extract the TL connected motif containing the TL red anchor.
    - For each non-TL quadrant, reflect the motif by quadrant side, then scale it by the ratio between target-anchor size and source-anchor size.
    - Align the scaled motif to the target anchor, and use that aligned placement as the phase origin of a rectangular tiling that fills the whole target quadrant.
    - Repeated red pixels are not copied; only non-background, non-red motif pixels are tiled, with clipping at quadrant boundaries.

    Key insight:
    - Red anchor size still determines scale and quadrant position still determines reflection, but the destination is no longer a single local attachment point:
      the aligned transformed motif becomes a tile that repeats across the entire target quadrant.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    _, _, source_size = anchors[(0, 0)]
    assert source_size > 0, "top-left red anchor must have positive size"

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    quads = {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, g.shape[1] - 1),
        (1, 0): (mid_r + 1, g.shape[0] - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, g.shape[0] - 1, mid_c + 1, g.shape[1] - 1),
    }

    for (qb, qr), (tr0, tc0, target_size) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_s = _scaled_patch(patch, qb, qr, source_size, target_size)
        q_r0, q_r1, q_c0, q_c1 = quads[(qb, qr)]
        _tile_quadrant(out, patch_s, tr0, tc0, q_r0, q_r1, q_c0, q_c1, bg)

    return out


# --------------- generator ---------------

BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]

TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it has n_fill cells."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        filled.add(cell)
        frontier.discard(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
    """Assign colors while ensuring every row and col of the bbox has visible motif content."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p not in red_cells]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    """
    Build the TL motif patch.

    Block mode guarantees exact down/up-scaling for odd target sizes; freeform mode is kept when allowed
    to preserve some of the original task's visual variety.
    """
    s = SOURCE_ANCHOR_SIZE
    use_block = needs_block or random.random() < BLOCK_PROB
    if use_block:
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        ar, ac = random.randrange(lh), random.randrange(lw)
        filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
        logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
        if logical is None:
            return None
        patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - s), random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _choose_anchor(qb: int, qr: int, size: int) -> tuple[int, int]:
    """Pick any anchor position whose red square fits inside the target quadrant."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    return (
        random.randint(q_r0, q_r1 - size + 1),
        random.randint(q_c0, q_c1 - size + 1),
    )


def _count_extra_tiled_pixels(
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    q_r0: int,
    q_r1: int,
    q_c0: int,
    q_c1: int,
) -> int:
    """
    Count visible non-red tiled pixels that lie outside the single base placement.

    This is used only by the generator to reject near-degenerate layouts that visually resemble the original task.
    """
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    tile_h, tile_w = patch_s.shape
    src_rr, src_cc = np.where((patch_s != BG) & (patch_s != RED))
    if src_rr.size == 0:
        return 0

    base_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    base_rr = src_rr + place_r0
    base_cc = src_cc + place_c0
    base_valid = (
        (base_rr >= q_r0)
        & (base_rr <= q_r1)
        & (base_cc >= q_c0)
        & (base_cc <= q_c1)
    )
    base_mask[base_rr[base_valid], base_cc[base_valid]] = True

    visible = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    start_r = place_r0
    while start_r + tile_h - 1 >= q_r0:
        start_r -= tile_h
    start_r += tile_h
    start_c = place_c0
    while start_c + tile_w - 1 >= q_c0:
        start_c -= tile_w
    start_c += tile_w

    for tile_r0 in range(start_r, q_r1 + 1, tile_h):
        for tile_c0 in range(start_c, q_c1 + 1, tile_w):
            tgt_rr = src_rr + tile_r0
            tgt_cc = src_cc + tile_c0
            valid = (
                (tgt_rr >= q_r0)
                & (tgt_rr <= q_r1)
                & (tgt_cc >= q_c0)
                & (tgt_cc <= q_c1)
            )
            visible[tgt_rr[valid], tgt_cc[valid]] = True

    return int(np.count_nonzero(visible & ~base_mask))


def _gen_attempt() -> np.ndarray | None:
    """Build a full tiled output, derive the input, and oracle-check with main()."""
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    needs_block = any(size % SOURCE_ANCHOR_SIZE != 0 for key, size in target_sizes.items() if key != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], SOURCE_ANCHOR_SIZE, size)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key in ((0, 1), (1, 0), (1, 1)):
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key])

    # Reject layouts where tiling is visually too close to a single placement.
    extra_pixels = []
    for key, patch_s in scaled.items():
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        extra_pixels.append(_count_extra_tiled_pixels(patch_s, *anchors[key], q_r0, q_r1, q_c0, q_c1))
    if max(extra_pixels) < 6 or sum(x > 0 for x in extra_pixels) < 2:
        return None

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), *anchors[key])
    for key, patch_s in scaled.items():
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        _tile_quadrant(out, patch_s, *anchors[key], q_r0, q_r1, q_c0, q_c1, BG)

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        paste(inp, np.full((size, size), RED, dtype=int), *anchors[key])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D3 tiling input")
