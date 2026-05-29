# Variant of 247ef758
# Dimension: D1 — 匹配条件：边缘颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架左/右边缘行标记颜色 且 == 上/下边缘列标记颜色（双对边镜像匹配），只有两对对边同时存在该颜色时... → 颜色+尺寸双重匹配
#
# Design analysis
# spec_variant:
#   concrete_visual:
#     - 保留原布局: 左侧精灵条 | 单列分隔线 | 右侧整高矩形框。
#     - 原题的“单个彩色边框格子”改成“连续同色段”:
#       左/右边框用竖直同色段编码行目标，段长 = 精灵外接矩形高度；
#       上/下边框用水平同色段编码列目标，段长 = 精灵外接矩形宽度。
#     - 目标点不是段上的每个像素，而是该段按 floor-center 取到的中心格；
#       这个中心定义与原题印刻时使用的 `(r0+r1)//2`, `(c0+c1)//2` 完全一致，
#       所以偶数长度也有稳定的整数中心。
#   behavior_trace:
#     - 例: 某个 4 色精灵 bbox 为 3x2。
#     - 原题逻辑会在边框上找“左右同为 4 的行”和“上下同为 4 的列”，然后做笛卡尔积。
#     - 变体里，左右边如果出现一段长度 3 的 4 色竖条，则其中心行才是有效行标；
#       上下边如果出现一段长度 2 的 4 色横条，则其中心列才是有效列标。
#     - 如果同色还出现长度 4 的额外竖条或长度 3 的额外横条，颜色仍匹配，但尺寸不匹配，必须忽略。
#     - 若仍沿用原题“逐格采样颜色”的写法，会把一整段里的每个格子都误判成目标，明显错误。
#   complexity_calibration:
#     - 尺寸档位控制在 2..4，既能提供可见的数值推理，又不会让边框挤满难以阅读。
#     - 每个样例强制至少有一个 mover 拥有“同色但错长度”的镜像边框段，保证颜色本身不足以决定目标。
#     - mover 仍为 2-3 个，distractor 仍为 0-1 个，避免自由度爆炸。
#   implicit_assumptions_review:
#     - 原 solver 的 row_marks/col_marks 逐格读取，隐含“标记是单个格子”；变体下必须改为扫描最大镜像同色段。
#     - 原 solver 仅以颜色为 key；变体下同色会有多种长度段，所以必须按 `(color, length)` 索引，再按精灵自身 bbox 高/宽取目标。
#     - 原 generator 只需保证某色在对边各出现一次即可；变体下还必须保证段与段之间不接触，否则会在视觉上合并成更长的段。
#   placement_legality:
#     - 所有边框段都在边框内部区间内放置，并留至少 1 个 base-color 间隔，防止合并。
#     - 有效段的中心范围与精灵实际可印刻范围一致，因此 solver 找到的目标一定能把对应精灵完整放进框内。
#   visual_observability:
#     - 人类可直接看到“精灵高/宽”和“边框同色段长度”相对应，再取段中心做笛卡尔积。
#     - 由于加入同色错长度段，解题者能从样例中看到“不是所有同色段都算”，规则差异肉眼可见。
#   cleanliness:
#     - 只在边框上叠加单层段，不引入额外装饰或内部噪声；框内仍为空白背景，输出时只叠印精灵。

from __future__ import annotations

from collections import defaultdict
import random
import numpy as np
from arc_utils import get_bg, label_mask


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """BBox of the largest connected component in `mask` that does not touch the border."""
    H, W = mask.shape
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    lab, n = label_mask(mask.astype(bool), connectivity=int(connectivity))
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if bool((rs == 0).any() or (rs == int(H - 1)).any() or (cs == 0).any() or (cs == int(W - 1)).any()):
            continue
        area = int(coords.shape[0])
        if area > best_area:
            best_area = area
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    return best_bbox


def _segment_extents(length: int) -> tuple[int, int]:
    upper = int((int(length) - 1) // 2)
    lower = int(length - 1 - upper)
    return upper, lower


def _segment_bounds(center: int, length: int) -> tuple[int, int]:
    upper, lower = _segment_extents(int(length))
    return int(center - upper), int(center + lower)


def _scan_mirrored_row_segments(
    g: np.ndarray,
    fc0: int,
    fc1: int,
    ir0: int,
    ir1: int,
    allowed_colors: set[int],
) -> dict[tuple[int, int], list[int]]:
    """
    Scan maximal mirrored vertical segments on the left/right frame edges.
    Keyed by (color, segment_length); values are segment centers.
    """
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    r = int(ir0)
    while r <= int(ir1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl in allowed_colors:
            r0 = int(r)
            while r + 1 <= int(ir1) and int(g[int(r + 1), int(fc0)]) == cl and int(g[int(r + 1), int(fc1)]) == cl:
                r += 1
            r1 = int(r)
            length = int(r1 - r0 + 1)
            center = int((r0 + r1) // 2)
            marks[(int(cl), int(length))].append(int(center))
        r += 1
    return marks


def _scan_mirrored_col_segments(
    g: np.ndarray,
    fr0: int,
    fr1: int,
    ic0: int,
    ic1: int,
    allowed_colors: set[int],
) -> dict[tuple[int, int], list[int]]:
    """
    Scan maximal mirrored horizontal segments on the top/bottom frame edges.
    Keyed by (color, segment_length); values are segment centers.
    """
    marks: dict[tuple[int, int], list[int]] = defaultdict(list)
    c = int(ic0)
    while c <= int(ic1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct in allowed_colors:
            c0 = int(c)
            while c + 1 <= int(ic1) and int(g[int(fr0), int(c + 1)]) == ct and int(g[int(fr1), int(c + 1)]) == ct:
                c += 1
            c1 = int(c)
            length = int(c1 - c0 + 1)
            center = int((c0 + c1) // 2)
            marks[(int(ct), int(length))].append(int(center))
        c += 1
    return marks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move left-strip sprites into the framed area.

    Variant rule:
    - A sprite matches a mirrored left/right edge segment only if both color and segment length
      match the sprite's bbox height.
    - A sprite matches a mirrored top/bottom edge segment only if both color and segment length
      match the sprite's bbox width.
    - Each qualifying segment contributes its floor-center as a target row/col; targets are the
      Cartesian product of matching row centers and col centers.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert not (inner_bbox is None)
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    divider_mask = np.zeros((H, W), dtype=bool)
    for c in range(0, max(0, int(fc0))):
        colv = g[:, int(c)]
        non = colv[colv != int(bg)]
        if int(len(non)) >= int(H - 1) and len(set(map(int, non))) == 1:
            divider_mask[:, int(c)] = True

    left_cols = np.arange(W)[None, :] < int(fc0)

    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & left_cols & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append(
                {
                    "color": int(col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "mask": m,
                    "height": int(m.shape[0]),
                    "width": int(m.shape[1]),
                }
            )

    move_colors = {int(s["color"]) for s in sprites}

    row_marks = _scan_mirrored_row_segments(g, int(fc0), int(fc1), int(ir0), int(ir1), move_colors)
    col_marks = _scan_mirrored_col_segments(g, int(fr0), int(fr1), int(ic0), int(ic1), move_colors)

    movers = []
    for s in sprites:
        rows = sorted(set(row_marks.get((int(s["color"]), int(s["height"])), [])))
        cols = sorted(set(col_marks.get((int(s["color"]), int(s["width"])), [])))
        if rows and cols:
            s["targets"] = [(int(r), int(c)) for r in rows for c in cols]
            movers.append(s)

    out = g.copy()

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in s["targets"]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out

def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """
    Generate a single-color 8-connected sprite cropped to its bounding box.
    The bbox is intentionally kept in the 2..4 range to make segment lengths readable.
    """
    if max_h < 2 or max_w < 2:
        return None
    sh = min(random.choice([2, 3, 3]), max_h)
    sw = min(random.choice([2, 3, 3, 4]), max_w)
    if sh < 2 or sw < 2:
        return None

    mask = np.zeros((sh, sw), dtype=bool)
    pivot_r = random.randrange(sh)
    pivot_c = random.randrange(sw)
    mask[int(pivot_r), :] = True
    mask[:, int(pivot_c)] = True

    extras: list[tuple[int, int]] = []
    for r in range(sh):
        for c in range(sw):
            if bool(mask[int(r), int(c)]):
                continue
            if any(
                0 <= int(r + dr) < sh and 0 <= int(c + dc) < sw and bool(mask[int(r + dr), int(c + dc)])
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            ):
                extras.append((int(r), int(c)))
    random.shuffle(extras)
    for r, c in extras[: random.randint(0, min(2, len(extras)))]:
        mask[int(r), int(c)] = True

    sprite = np.zeros((sh, sw), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _can_place_interval(start: int, end: int, occupied: np.ndarray, lo: int, hi: int) -> bool:
    if int(start) < int(lo) or int(end) > int(hi):
        return False
    pad_start = max(int(lo), int(start - 1))
    pad_end = min(int(hi), int(end + 1))
    for pos in range(int(pad_start), int(pad_end) + 1):
        if bool(occupied[int(pos - lo)]):
            return False
    return True


def _occupy_interval(start: int, end: int, occupied: np.ndarray, lo: int) -> None:
    occupied[int(start - lo) : int(end - lo) + 1] = True


def _place_interval_center(
    length: int,
    occupied: np.ndarray,
    lo: int,
    hi: int,
    *,
    center_lo: int | None = None,
    center_hi: int | None = None,
) -> int | None:
    upper, lower = _segment_extents(int(length))
    low = int(lo + upper if center_lo is None else max(center_lo, lo + upper))
    high = int(hi - lower if center_hi is None else min(center_hi, hi - lower))
    if low > high:
        return None
    candidates = list(range(int(low), int(high) + 1))
    random.shuffle(candidates)
    for center in candidates:
        start, end = _segment_bounds(int(center), int(length))
        if _can_place_interval(int(start), int(end), occupied, int(lo), int(hi)):
            _occupy_interval(int(start), int(end), occupied, int(lo))
            return int(center)
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    interior_h = random.randint(8, 13)
    interior_w = random.randint(8, 13)
    H = interior_h + 2
    if not 10 <= H <= 15:
        return None

    sprite_area_w = random.randint(3, 5)
    frame_w = interior_w + 2
    W = sprite_area_w + 1 + frame_w
    if not 14 <= W <= 20:
        return None

    div_col = sprite_area_w
    fc0, fc1 = sprite_area_w + 1, W - 1
    ir0, ic0 = 1, fc0 + 1
    ir1, ic1 = H - 2, fc1 - 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = random.choice([2, 2, 3])
    n_distractors = random.choice([0, 0, 0, 1])
    total_items = int(n_movers + n_distractors)
    max_sprite_h = min(3, max(2, H // max(1, total_items)))

    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprites: list[np.ndarray] = []
    sprite_dims: dict[int, tuple[int, int]] = {}
    for mc in mover_colors:
        s = _make_sprite(int(mc), int(max_sprite_h), int(sprite_area_w))
        if s is None:
            return None
        sprites.append(s)
        sprite_dims[int(mc)] = (int(s.shape[0]), int(s.shape[1]))

    if int(sum(int(sp.shape[0]) for sp in sprites)) > int(H):
        return None

    row_valid_counts = {int(mc): 1 for mc in mover_colors}
    col_valid_counts = {int(mc): 1 for mc in mover_colors}
    if random.random() < 0.35:
        row_valid_counts[int(random.choice(mover_colors))] += 1
    if random.random() < 0.35:
        col_valid_counts[int(random.choice(mover_colors))] += 1
    row_centers: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}
    col_centers: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}
    row_lengths: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}
    col_lengths: dict[int, list[int]] = {int(mc): [] for mc in mover_colors}

    row_occupied = np.zeros(int(interior_h), dtype=bool)
    col_occupied = np.zeros(int(interior_w), dtype=bool)

    row_specs = []
    for mc in mover_colors:
        sh, _sw = sprite_dims[int(mc)]
        for _ in range(int(row_valid_counts[int(mc)])):
            row_specs.append((int(mc), int(sh)))
    row_specs.sort(key=lambda x: (-int(x[1]), random.random()))
    for mc, sh in row_specs:
        center = _place_interval_center(int(sh), row_occupied, int(ir0), int(ir1))
        if center is None:
            return None
        row_centers[int(mc)].append(int(center))
        row_lengths[int(mc)].append(int(sh))

    col_specs = []
    for mc in mover_colors:
        _sh, sw = sprite_dims[int(mc)]
        for _ in range(int(col_valid_counts[int(mc)])):
            col_specs.append((int(mc), int(sw)))
    col_specs.sort(key=lambda x: (-int(x[1]), random.random()))
    for mc, sw in col_specs:
        center = _place_interval_center(int(sw), col_occupied, int(ic0), int(ic1))
        if center is None:
            return None
        col_centers[int(mc)].append(int(center))
        col_lengths[int(mc)].append(int(sw))

    ambiguous_color = int(random.choice(mover_colors))
    bad_row_lengths = [v for v in [2, 3, 4] if v <= int(interior_h) and v != int(sprite_dims[ambiguous_color][0])]
    bad_col_lengths = [v for v in [2, 3, 4] if v <= int(interior_w) and v != int(sprite_dims[ambiguous_color][1])]
    if not bad_row_lengths or not bad_col_lengths:
        return None

    bad_row_len = int(random.choice(bad_row_lengths))
    bad_col_len = int(random.choice(bad_col_lengths))
    bad_row_center = _place_interval_center(int(bad_row_len), row_occupied, int(ir0), int(ir1))
    bad_col_center = _place_interval_center(int(bad_col_len), col_occupied, int(ic0), int(ic1))
    if bad_row_center is None or bad_col_center is None:
        return None
    row_centers[ambiguous_color].append(int(bad_row_center))
    row_lengths[ambiguous_color].append(int(bad_row_len))
    col_centers[ambiguous_color].append(int(bad_col_center))
    col_lengths[ambiguous_color].append(int(bad_col_len))

    inp = np.full((H, W), bg, dtype=int)
    inp[:, int(div_col)] = int(divider_color)
    inp[0, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[H - 1, int(fc0) : int(fc1) + 1] = int(frame_base)
    inp[1 : H - 1, int(fc0)] = int(frame_base)
    inp[1 : H - 1, int(fc1)] = int(frame_base)

    for mc in mover_colors:
        for center, length in zip(row_centers[int(mc)], row_lengths[int(mc)]):
            r0, r1 = _segment_bounds(int(center), int(length))
            inp[int(r0) : int(r1) + 1, int(fc0)] = int(mc)
            inp[int(r0) : int(r1) + 1, int(fc1)] = int(mc)
        for center, length in zip(col_centers[int(mc)], col_lengths[int(mc)]):
            c0, c1 = _segment_bounds(int(center), int(length))
            inp[0, int(c0) : int(c1) + 1] = int(mc)
            inp[H - 1, int(c0) : int(c1) + 1] = int(mc)

    all_items: list[tuple[int, np.ndarray, bool]] = [(int(mc), sp, False) for mc, sp in zip(mover_colors, sprites)]
    remaining_h = int(H - sum(int(sp.shape[0]) for sp in sprites))
    for dc_color in distractor_colors:
        max_h_here = min(4, max(2, remaining_h))
        ds = _make_sprite(int(dc_color), int(max_h_here), int(sprite_area_w))
        if ds is None:
            return None
        all_items.append((int(dc_color), ds, True))
        remaining_h -= int(ds.shape[0])
    if sum(int(sp.shape[0]) for _col, sp, _is_dist in all_items) > int(H):
        return None

    cur_row = 0
    for _col, sp, _is_dist in all_items:
        sh, sw = sp.shape
        if cur_row + int(sh) > int(H) or int(sw) > int(sprite_area_w):
            return None
        c_off = random.randint(0, int(sprite_area_w - sw))
        for dr in range(int(sh)):
            for dc in range(int(sw)):
                if int(sp[int(dr), int(dc)]) != int(bg):
                    inp[int(cur_row + dr), int(c_off + dc)] = int(sp[int(dr), int(dc)])
        cur_row += int(sh)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != int(bg):
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
