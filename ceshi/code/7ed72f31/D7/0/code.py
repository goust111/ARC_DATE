# Variant of 7ed72f31
# Dimension: D7 — 网格输出形态 (grid_shape)
# Change: 输出与输入同尺寸（18-28的正方形网格），精灵原地变换... → 紧凑裁剪输出

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, bbox, crop_bbox, get_bg, label_mask


"""
design_analysis:
  variant_goal:
    keep:
      - 红色标记的语义不变：嵌入式红线做轴反射，孤立红点做点反射。
      - 输入仍是 18-28 的方形大画布，背景占主导，多个小精灵分散放置。
      - 反射后的彩色像素覆盖到背景上，原像素和红标记保留。
    change:
      - 原题输出直接返回整张画布。
      - 本变体先按原题完成整张画布上的反射，再把输出裁成“包含所有非背景像素”的最小矩形。
      - 因此解题者除了推断反射规则，还必须识别输出边界来自前景总包围盒。

  concrete_behavior_and_trace:
    input_picture:
      - 大方形网格中，若干单色 payload 围绕红标记分布，周围留有明显背景边框。
      - 例：一个横向红线精灵放在画布中部，payload 在红线下方；solver 会把它镜像到红线上方。
      - 再例：一个孤立红点旁边有一个 8 邻接的小单色块；solver 会把该色块绕红点做点反射。
    original_flow:
      - `out = g.copy()` 保留整张输入大小。
      - 逐个组件把镜像结果画回 `out`。
      - 返回完整 `out`，即使四周大部分仍是背景。
    variant_flow:
      - 前两步完全相同，先得到完整 `out_full`。
      - 之后求 `out_full != bg` 的总 bbox，只返回该 bbox 内的裁剪结果。
    tiny_example:
      - 若完整输出的非背景最外层坐标范围是 rows 3..11, cols 5..14，
        原题返回 18x18，大部分是背景；
        变体返回 9x10，仅保留 rows 3..11 / cols 5..14。
      - 差异只发生在“最终输出格式”，反射内容本身不变。

  complexity_calibration:
    - 这次只改 D7，不引入新的颜色语义或新的变换类型，避免偏离原题。
    - 为了让 D7 肉眼可见，生成器强制完整输出与外边界至少相隔 1 格背景，
      使裁剪后尺寸一定小于输入尺寸，而不是偶尔恰好等大。
    - 不额外增加太多自由度；反射仍只有原题的三类：横轴、竖轴、点反射。

  implicit_assumption_audit:
    main_related_lines:
      - `bg = get_bg(g)`: 仍依赖“输入背景是众数”。变体不改输入分布，此前提成立。
      - `out = g.copy()`: 原题假设输出先在原地画。变体仍需要这个完整画布作为中间态，
        只是最终不直接返回它。
      - `inb(...)`: 原题只在输入画布内落点。变体仍以完整画布为运算域，因此边界判断不变。
      - marker-containing 组件的 bbox / eligible 逻辑：依赖组件在输入上可分离、payload 与镜像区不重叠。
        变体未改变此结构，但生成器必须继续保证。
      - isolated marker 分支用邻接像素找颜色，再用该颜色的 8 连通分量整体反射。
        这依赖“红点邻域里能唯一指向应被反射的分量”，生成器仍要保证红点只贴近目标分量。
      - 新增的 tight crop 依赖“输出至少有一个非背景像素”。该任务天然成立，
        生成器也始终放置至少 3 个 marker 对应的前景。
    new_risks:
      - 如果完整输出碰到外边界，裁剪可能在某一维上与原尺寸相同，导致 D7 弱化。
        生成器通过 outer margin 消除这个退化。

  placement_legality:
    - 仍采用 output-first：先构造完整输出精灵，再擦掉镜像半边得到输入。
    - 精灵放置必须满足：
      - 各精灵 footprint 互不重叠；
      - 考虑 1 格 dilation padding 后仍不相撞，避免组件意外合并；
      - 所有精灵都在距画布边界至少 1 格的位置，确保 tight crop 真正发生；
      - 由此得到的输入仍满足 solver 的组件识别假设。

  visual_observability:
    - 人类看 2-3 对样例时，会先看到“输入很大、输出内容没变但边框被裁掉”。
    - 因为反射后的形状仍保留原题可见线索，新增规则只是在输出边界上增加一步，
      不会被隐藏在颜色或计数等不可见属性里。
    - 生成器不允许完整输出贴边，因此“输出尺寸缩小”会稳定出现。

  visual_cleanliness:
    - 继续使用稀疏布局、1 格 padding、单色 payload，避免多精灵粘连。
    - 输出裁剪只去掉纯背景外框，不会在对象内部引入新的遮挡或混叠。

  color_inference:
    - 本变体不引入颜色映射。颜色仍只用于区分 sprite，自身不参与排序或裁剪。
"""


MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28
OUTER_MARGIN = 1

# Marker counts
TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX = 3, 7
MAX_EMBEDDED, MAX_ISOLATED = 4, 4

# Embedded payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 6
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 11
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.50, 0.80

# Isolated payload
ISOLATED_PX_MIN, ISOLATED_PX_MAX = 1, 8
ISOLATED_BBOX_MAX = 4

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 40
GEN_RETRIES = 300

_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid != int(bg)
    bb = bbox(mask)
    if bb is None:
        return grid.copy()
    return crop_bbox(grid, bb)


def _solve_full(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    out = g.copy()

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < H and 0 <= int(c) < W

    def refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
        pr, pc = map(int, p)
        return int(2 * pr - r), int(2 * pc - c)

    def refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
        return int(2 * int(r0) - int(r)), int(c)

    def refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
        return int(r), int(2 * int(c0) - int(c))

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        if not reds:
            continue

        rs = {int(r) for r, _ in reds}
        cs = {int(c) for _, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rs) == 1:
            axis = ("row", int(next(iter(rs))))
        elif len(cs) == 1:
            axis = ("col", int(next(iter(cs))))
        else:
            continue

        r0, c0, r1, c1 = map(int, comp["bbox"])
        payload = {int(v) for v in vals if int(v) != int(MARK)}
        if not payload:
            continue

        if axis[0] == "row":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(c0) <= int(c) <= int(c1)

        elif axis[0] == "col":

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(r0) <= int(r) <= int(r1)

        else:
            rr0 = max(0, int(r0 - 1))
            rr1 = min(int(H - 1), int(r1 + 1))
            cc0 = max(0, int(c0 - 1))
            cc1 = min(int(W - 1), int(c1 + 1))

            def eligible(r: int, c: int, v: int) -> bool:
                return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

        for r in range(int(H)):
            for c in range(int(W)):
                v = int(g[int(r), int(c)])
                if v == int(bg) or v == int(MARK):
                    continue
                if not eligible(int(r), int(c), int(v)):
                    continue

                if axis[0] == "row":
                    rr, cc = refl_row(int(axis[1]), int(r), int(c))
                elif axis[0] == "col":
                    rr, cc = refl_col(int(axis[1]), int(r), int(c))
                else:
                    rr, cc = refl_point(axis[1], int(r), int(c))

                if inb(int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(v)

    marker_pts = []
    for comp in comps:
        if len(comp["coords"]) == 1 and int(comp["vals"][0]) == int(MARK):
            marker_pts.append(comp["coords"][0])

    for pr, pc in marker_pts:
        pr = int(pr)
        pc = int(pc)
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        for v in neigh_colors:
            labv, _nv = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = refl_point((int(pr), int(pc)), int(r), int(c))
                    if inb(rr, cc):
                        out[int(rr), int(cc)] = int(v)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror colored sprites using red markers as in the original task, then tight-crop.

    The reflection logic is unchanged from 7ed72f31.
    The only new rule is that the final output is cropped to the minimal rectangle
    containing all non-background pixels.
    """
    full = _solve_full(grid)
    bg = int(get_bg(np.array(grid, dtype=int)))
    return _tight_crop_non_bg(full, bg)


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _build_embedded_sprite(rng: random.Random) -> dict | None:
    ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
    max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
    min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
    if min_px > max_px:
        return None
    pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
    if pmask is None:
        return None

    is_row = rng.choice([True, False])
    side_flip = rng.random() < 0.5

    if is_row:
        h, w = 2 * ph + 1, pw
        payload = np.zeros((h, w), dtype=bool)
        mirror = np.zeros((h, w), dtype=bool)
        red_pixels = [(ph, c) for c in range(pw)]
        orig_slice, mirr_slice = (np.s_[:ph], np.s_[ph + 1 :]) if not side_flip else (np.s_[ph + 1 :], np.s_[:ph])
        payload[orig_slice] = pmask
        mirror[mirr_slice] = pmask[::-1]
    else:
        h, w = pw, 2 * ph + 1
        payload = np.zeros((h, w), dtype=bool)
        mirror = np.zeros((h, w), dtype=bool)
        red_pixels = [(r, ph) for r in range(pw)]
        orig_slice, mirr_slice = (
            (np.s_[:, :ph], np.s_[:, ph + 1 :]) if not side_flip else (np.s_[:, ph + 1 :], np.s_[:, :ph])
        )
        payload[orig_slice] = pmask.T
        mirror[mirr_slice] = pmask.T[:, ::-1]

    return {"payload": payload, "mirror": mirror, "red_pixels": red_pixels}


def _is_4connected_to(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            return True
    return False


def _build_isolated_sprite(rng: random.Random) -> dict | None:
    n_px = rng.randint(ISOLATED_PX_MIN, ISOLATED_PX_MAX)
    pmask = None
    for _ in range(MASK_RETRIES):
        ph = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        pw = rng.randint(1, min(ISOLATED_BBOX_MAX, max(1, n_px)))
        if ph * pw < n_px or n_px < max(ph, pw):
            continue
        pmask = np.ones((ph, pw), dtype=bool) if n_px == ph * pw else _random_connected_mask(ph, pw, n_px, rng=rng)
        if pmask is not None:
            break
    if pmask is None:
        return None

    pcoords = list(zip(*np.where(pmask)))
    rng.shuffle(pcoords)

    for pr, pc in pcoords:
        dirs = list(_DIRS8)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            red_r, red_c = pr + dr, pc + dc
            refl = [(2 * red_r - r, 2 * red_c - c) for r, c in pcoords]
            all_pts = pcoords + refl + [(red_r, red_c)]
            all_rs = [r for r, _ in all_pts]
            all_cs = [c for _, c in all_pts]
            minr, minc = min(all_rs), min(all_cs)
            h = max(all_rs) - minr + 1
            w = max(all_cs) - minc + 1

            payload = np.zeros((h, w), dtype=bool)
            mirror = np.zeros((h, w), dtype=bool)
            for r, c in pcoords:
                payload[r - minr, c - minc] = True
            for r, c in refl:
                mirror[r - minr, c - minc] = True
            red_pos = (red_r - minr, red_c - minc)

            if payload[red_pos] or mirror[red_pos]:
                continue
            if (payload & mirror).any():
                continue
            if _is_4connected_to(payload | mirror, *red_pos):
                continue
            return {"payload": payload, "mirror": mirror, "red_pixels": [red_pos]}
    return None


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
    outer_margin: int = OUTER_MARGIN,
):
    h_canvas, w_canvas = canvas.shape
    h_patch, w_patch = patch_mask.shape
    if h_patch > h_canvas - 2 * outer_margin or w_patch > w_canvas - 2 * outer_margin:
        return None

    occ = canvas != bg
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)

    cands = []
    for r0 in range(outer_margin, h_canvas - h_patch - outer_margin + 1):
        for c0 in range(outer_margin, w_canvas - w_patch - outer_margin + 1):
            if not (occ[r0 : r0 + h_patch, c0 : c0 + w_patch] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = spr["color"]
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = MARK


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = spr["r0"], spr["c0"]
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = bg


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    total_markers = rng.randint(TOTAL_MARKERS_MIN, TOTAL_MARKERS_MAX)
    n_embedded = rng.randint(max(1, total_markers - MAX_ISOLATED), min(MAX_EMBEDDED, total_markers))
    n_isolated = total_markers - n_embedded

    n_colors = rng.randint(PALETTE_COLORS_MIN, min(PALETTE_COLORS_MAX, len(palette), total_markers))
    chosen_colors = rng.sample(palette, n_colors)

    sprites: list[dict] = []
    for builder, count in [(_build_embedded_sprite, n_embedded), (_build_isolated_sprite, n_isolated)]:
        for _ in range(count):
            s = builder(rng)
            if s is None:
                return None
            sprites.append(s)

    assignment = list(chosen_colors) + [rng.choice(chosen_colors) for _ in range(len(sprites) - n_colors)]
    rng.shuffle(assignment)
    for spr, color in zip(sprites, assignment):
        spr["color"] = color

    full_output = np.full((n, n), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        for r, c in spr["red_pixels"]:
            footprint[r, c] = True
        loc = _random_free_location(full_output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = r0, c0
        _stamp_sprite(full_output, spr, r0, c0)

    bb = bbox(full_output != bg)
    if bb is None:
        return None
    r0, c0, r1, c1 = bb
    if r0 < OUTER_MARGIN or c0 < OUTER_MARGIN or r1 > n - 1 - OUTER_MARGIN or c1 > n - 1 - OUTER_MARGIN:
        return None

    inp = full_output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == bg).sum() / (n * n) < BG_COVERAGE_MIN:
        return None

    expected = _tight_crop_non_bg(full_output, bg)
    try:
        produced = main(inp)
    except Exception:
        return None
    if np.array_equal(expected, full_output):
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
