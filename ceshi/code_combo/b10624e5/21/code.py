# Combo variant of b10624e5 (index 21)
# combo_spec: D3/2+D13/1
# Name: 图案环绕锚块四周 + 目标锚块居中对齐

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# combo_spec:
#   source_task: b10624e5
#   fused_change:
#     - D13/1: TL 图案不再是“和红锚块连成一个整体”的旧式邻接图案，而是四个分离的侧片，
#       分别位于红锚块的上/下/左/右四侧，中间保留背景留距。
#     - D3/2: 目标象限不再把整块模板按红色 bbox 左上角对齐过去；相反，四个侧片都以目标锚块中心
#       为参考独立放置，并沿各自对应的边居中。
#   concrete_layout:
#     - grid 仍为 27x27，divider 在 row=13 / col=13。
#     - 四个象限各有一个红色实心方块锚块；TL 固定 2x2，其余目标尺寸限制在 2..4。
#     - TL 模板固定外框 6x6：红锚块在中间 2x2，四条边各有一条 4 连通侧片，且每个侧片只接触模板的一条外边。
#     - 每条侧片的长度可不同，且允许在本边内偏移；这样 TL 中能直接看到“环绕但不一定居中”的源布局。
#   behavior_trace:
#     old_solver_path:
#       - 取 TL 含红组件或整块 bbox，并整体做镜像 + 缩放，再让缩放后的红区和目标红锚块重合。
#       - 对本融合变体，这会把四边之间的空白也一起缩放，保留 TL 里的横向/纵向偏移。
#     fused_solver_path:
#       - 先取 TL 全部非背景 bbox，再把其中非红像素按连通块拆成 top/bottom/left/right 四个侧片。
#       - 每个侧片单独做：
#         1. 按象限关系决定落到目标锚块的哪一侧；
#         2. 仅沿自身 bbox 做 NN 缩放；
#         3. 以目标红锚块中心为基准，在该侧“居中”放置。
#       - 因而输出会把 TL 中偏左/偏右、偏上/偏下的条片重新居中，明显不同于旧逻辑。
#     tiny_example:
#       - 若 TL 顶边侧片是 row0 上的 2 格序列 [8,3]，且偏在左半边；
#       - 旧逻辑把整块 6x6 一起缩放到 9x9 时，这 2 格仍相对整块左偏；
#       - 新逻辑先把 [8,3] 作为 1x2 小条缩放到 2x3，再把它居中贴到目标锚块上方；
#       - 如果目标象限在右侧，条内顺序还会做 LR 反射，变成 [3,8]。
#   complexity_calibration:
#     - 保留 4 个侧片，且每片内部允许 2-4 长度的彩色序列，视觉上足够区分四边与镜像方向。
#     - 目标尺寸只用 2..4，避免 1x1 时“环绕 + 留距 + 居中”塌缩得过于微弱。
#     - 生成器强制拒绝“融合输出和旧式整块缩放输出完全相同”的样本，避免退化。
#   hidden_assumptions_review:
#     - 原 main 依赖“TL 图案和红锚块属于同一 4 连通组件”；本变体不成立，必须改成取 TL 全部非背景 bbox。
#     - 原 main 依赖“整体模板缩放后，对齐红色 bbox 就等于正确定位”；本变体不成立，因为四个侧片需要独立居中。
#     - 原 roundtrip 缩放断言也不再适用：独立条片的偏移不会被保留，变化本身就是规则的一部分。
#     - 新 main 的新假设是：四个侧片各自只接触源模板的一条外边，且一边恰有一个侧片；
#       生成器通过固定 6x6 框架和无角接触来保证这一点。
#   placement_legality:
#     - 源模板外框固定 6x6；在尺寸 2,3,4 下，居中环绕后的总外框最多 12x12，能完整落在 13x13 象限内。
#     - generator 在选目标锚块位置时，按“侧片 + 红锚块”的整体 footprint 计算可行范围，避免越界与重叠。
#   observability:
#     - 输入 TL 直接展示“四边环绕，但条片不一定在本边居中”的源关系。
#     - 输出非 TL 象限中，四条边会围绕目标锚块中心等距展开；TR 会交换左右边并反转横条顺序，
#       BL 会交换上下边并反转竖条顺序，BR 两者同时发生。
#   cleanliness:
#     - 每个非 TL 象限输入只保留红锚块；输出再添四个离散侧片，没有额外覆盖层或排序噪声。
#   color_inference:
#     - 不做颜色映射，颜色只承载“这是哪条边上的什么顺序”这一可见结构信息。


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
TARGET_SIZE_OPTIONS = (2, 3, 4)
SOURCE_PATCH_SIZE = 6
SIDE_GAP = 1
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400
SIDES = ("top", "bottom", "left", "right")


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
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
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _center_start(anchor0: int, anchor_size: int, span: int) -> int:
    return int(anchor0 + (anchor_size - span) // 2)


def _scaled_gap(target_size: int) -> int:
    ratio = target_size / SOURCE_ANCHOR_SIZE
    return max(1, int(round(SIDE_GAP * ratio)))


def _horizontal_mirror(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return side


def _vertical_mirror(side: str) -> str:
    if side == "top":
        return "bottom"
    if side == "bottom":
        return "top"
    return side


def _extract_side_pieces(patch: np.ndarray, bg: int) -> dict[str, np.ndarray]:
    mask = (patch != bg) & (patch != RED)
    comps = components_mask(mask, connectivity=4)
    h, w = patch.shape
    pieces: dict[str, np.ndarray] = {}
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        touches = []
        if r0 == 0:
            touches.append("top")
        if r1 == h - 1:
            touches.append("bottom")
        if c0 == 0:
            touches.append("left")
        if c1 == w - 1:
            touches.append("right")
        assert len(touches) == 1, "each source side piece must touch exactly one outer edge"
        side = touches[0]
        assert side not in pieces, "there must be exactly one side piece per edge"
        pieces[side] = patch[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert set(pieces) == set(SIDES), "source motif must provide top/bottom/left/right side pieces"
    return pieces


def _transform_side_piece(
    piece: np.ndarray, side: str, qb: int, qr: int, target_size: int
) -> tuple[str, np.ndarray]:
    out_side = side
    out_piece = piece

    if side in {"top", "bottom"} and qr == 1:
        out_piece = np.fliplr(out_piece)
    if side in {"left", "right"} and qb == 1:
        out_piece = np.flipud(out_piece)

    if qr == 1:
        out_side = _horizontal_mirror(out_side)
    if qb == 1:
        out_side = _vertical_mirror(out_side)

    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(out_piece.shape[0] * ratio)))
    new_w = max(1, int(round(out_piece.shape[1] * ratio)))
    return out_side, _resize_nn(out_piece, new_h, new_w)


def _relative_piece_start(side: str, piece_shape: tuple[int, int], target_size: int) -> tuple[int, int]:
    h, w = map(int, piece_shape)
    gap = _scaled_gap(target_size)
    if side == "top":
        return -gap - h, _center_start(0, target_size, w)
    if side == "bottom":
        return target_size + gap, _center_start(0, target_size, w)
    if side == "left":
        return _center_start(0, target_size, h), -gap - w
    if side == "right":
        return _center_start(0, target_size, h), target_size + gap
    raise ValueError(f"unknown side: {side}")


def _absolute_piece_start(
    side: str, piece_shape: tuple[int, int], anchor_r: int, anchor_c: int, target_size: int
) -> tuple[int, int]:
    dr, dc = _relative_piece_start(side, piece_shape, target_size)
    return int(anchor_r + dr), int(anchor_c + dc)


def _stamp_piece(
    out: np.ndarray, piece: np.ndarray, place_r0: int, place_c0: int, bg: int, mid_r: int, mid_c: int
) -> None:
    h, w = out.shape
    rr_p, cc_p = np.where(piece != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < h and 0 <= cc_t < w):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(piece[rr0, cc0])


def _anchors_by_quadrant(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"
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
    assert set(anchors) == set(QUADS), "there must be one red anchor in every quadrant"
    return anchors


def _source_patch_from_tl(tl: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(tl != bg)
    assert pts.size > 0, "top-left quadrant must contain the source template"
    r0, c0, r1, c1 = bbox_coords(pts)
    return tl[r0 : r1 + 1, c0 : c1 + 1].copy()


def _legacy_whole_patch_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _anchors_by_quadrant(g, mid_r, mid_c)

    tl = g[0:mid_r, 0:mid_c]
    patch = _source_patch_from_tl(tl, bg)
    sr0, sc0, sL = anchors[(0, 0)]
    patch_red = np.where(patch == RED)
    assert patch_red[0].size > 0, "source patch must contain red anchor pixels"
    pr_red0 = int(patch_red[0].min())
    pc_red0 = int(patch_red[1].min())
    assert sL == SOURCE_ANCHOR_SIZE
    assert patch.shape == (SOURCE_PATCH_SIZE, SOURCE_PATCH_SIZE)
    assert pr_red0 == sr0 - np.argwhere(tl != bg)[:, 0].min()
    assert pc_red0 == sc0 - np.argwhere(tl != bg)[:, 1].min()

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)
        ratio = tL / sL
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)

        rr_red, cc_red = np.where(patch_s == RED)
        place_r0 = tr0 - int(rr_red.min())
        place_c0 = tc0 - int(cc_red.min())
        _stamp_piece(out, patch_s, place_r0, place_c0, bg, mid_r, mid_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer four surround side-pieces from the top-left anchor into the other quadrants using centered placement.

    Input structure:
    - The grid is split into four quadrants by one full divider row and one full divider column.
    - Every quadrant contains exactly one red square anchor block.
    - The top-left quadrant contains the only complete source motif: four disconnected colored side-pieces around the red anchor, one touching each outer edge of the source bbox.

    Rule:
    - Take the full non-background bbox from the top-left quadrant, then split its non-red pixels into the top/bottom/left/right side-pieces.
    - For each destination quadrant, reflect each side-piece according to the quadrant relation, scale that side-piece by the target/source anchor-size ratio, and map it to the corresponding destination side.
    - Place every transformed side-piece relative to the target anchor center, centered along that edge, with a scaled background gap preserved between the side-piece and the anchor.

    Key insight: The transferable object is not one monolithic template anymore; it is a four-part surround layout whose pieces are transformed independently and re-centered around each target anchor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _anchors_by_quadrant(g, mid_r, mid_c)

    sr0, sc0, sL = anchors[(0, 0)]
    assert sL == SOURCE_ANCHOR_SIZE, "top-left red anchor must be 2x2 in this combo"

    tl = g[0:mid_r, 0:mid_c]
    patch = _source_patch_from_tl(tl, bg)
    assert patch.shape == (SOURCE_PATCH_SIZE, SOURCE_PATCH_SIZE), "source surround template must be 6x6"

    rr_red, cc_red = np.where(patch == RED)
    assert rr_red.size > 0, "source template must contain a red anchor"
    assert int(rr_red.min()) == 2 and int(cc_red.min()) == 2, "source anchor must sit in the center of the 6x6 frame"
    assert int(rr_red.max()) == 3 and int(cc_red.max()) == 3

    side_pieces = _extract_side_pieces(patch, bg)
    assert sr0 >= 0 and sc0 >= 0  # kept only to make the source-anchor dependency explicit

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        for side, piece in side_pieces.items():
            out_side, piece_s = _transform_side_piece(piece, side, qb, qr, tL)
            place_r0, place_c0 = _absolute_piece_start(out_side, piece_s.shape, tr0, tc0, tL)
            _stamp_piece(out, piece_s, place_r0, place_c0, bg, mid_r, mid_c)

    return out


def _target_sizes() -> dict[tuple[int, int], int]:
    while True:
        vals = [random.choice(TARGET_SIZE_OPTIONS) for _ in range(3)]
        if len(set(vals)) >= 2:
            return {
                (0, 0): SOURCE_ANCHOR_SIZE,
                (0, 1): vals[0],
                (1, 0): vals[1],
                (1, 1): vals[2],
            }


def _rand_seq(length: int, palette: list[int]) -> list[int]:
    while True:
        seq = [random.choice(palette) for _ in range(length)]
        if len(set(seq)) >= 2 and seq != list(reversed(seq)):
            return seq


def _paint_side(patch: np.ndarray, side: str, start: int, colors: list[int]) -> None:
    if side == "top":
        patch[0, start : start + len(colors)] = colors
        return
    if side == "bottom":
        patch[-1, start : start + len(colors)] = colors
        return
    if side == "left":
        patch[start : start + len(colors), 0] = colors
        return
    if side == "right":
        patch[start : start + len(colors), -1] = colors
        return
    raise ValueError(f"unknown side: {side}")


def _build_source_patch() -> np.ndarray:
    patch = np.full((SOURCE_PATCH_SIZE, SOURCE_PATCH_SIZE), BG, dtype=int)
    patch[2:4, 2:4] = RED
    palette = random.sample(COLOR_POOL, 4)

    while True:
        specs = {}
        for side in SIDES:
            length = random.randint(2, 4)
            start = random.randint(1, 5 - length)
            specs[side] = (start, _rand_seq(length, palette))
        legacy_like = all(
            (side in {"top", "bottom"} and specs[side][0] == 1 and len(specs[side][1]) == 4)
            or (side in {"left", "right"} and specs[side][0] == 1 and len(specs[side][1]) == 4)
            for side in SIDES
        )
        if not legacy_like:
            break

    for side, (start, colors) in specs.items():
        _paint_side(patch, side, start, colors)

    pieces = _extract_side_pieces(patch, BG)
    assert set(pieces) == set(SIDES)
    return patch


def _transformed_side_pieces(
    side_pieces: dict[str, np.ndarray], qb: int, qr: int, target_size: int
) -> list[tuple[str, np.ndarray]]:
    return [_transform_side_piece(piece, side, qb, qr, target_size) for side, piece in side_pieces.items()]


def _footprint_bounds(side_pieces: dict[str, np.ndarray], qb: int, qr: int, target_size: int) -> tuple[int, int, int, int]:
    min_r, max_r = 0, target_size - 1
    min_c, max_c = 0, target_size - 1
    for side, piece_s in _transformed_side_pieces(side_pieces, qb, qr, target_size):
        dr, dc = _relative_piece_start(side, piece_s.shape, target_size)
        min_r = min(min_r, dr)
        max_r = max(max_r, dr + piece_s.shape[0] - 1)
        min_c = min(min_c, dc)
        max_c = max(max_c, dc + piece_s.shape[1] - 1)
    return int(min_r), int(max_r), int(min_c), int(max_c)


def _choose_anchor(side_pieces: dict[str, np.ndarray], qb: int, qr: int, target_size: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    min_r, max_r, min_c, max_c = _footprint_bounds(side_pieces, qb, qr, target_size)
    r_lo = q_r0 - min_r
    r_hi = q_r1 - max_r
    c_lo = q_c0 - min_c
    c_hi = q_c1 - max_c
    assert r_lo <= r_hi and c_lo <= c_hi, "transformed surround footprint must fit inside the quadrant"
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _gen_attempt() -> np.ndarray | None:
    target_sizes = _target_sizes()
    patch = _build_source_patch()
    side_pieces = _extract_side_pieces(patch, BG)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - SOURCE_PATCH_SIZE)
    tl_c0 = random.randint(0, MID - SOURCE_PATCH_SIZE)
    anchors[(0, 0)] = (tl_r0 + 2, tl_c0 + 2)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        anchors[key] = _choose_anchor(side_pieces, key[0], key[1], size)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
        for side, piece in side_pieces.items():
            out_side, piece_s = _transform_side_piece(piece, side, key[0], key[1], size)
            pr0, pc0 = _absolute_piece_start(out_side, piece_s.shape, ar, ac, size)
            _stamp_piece(out, piece_s, pr0, pc0, BG, MID, MID)

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_whole_patch_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
