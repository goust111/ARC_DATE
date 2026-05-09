# Variant of 247ef758
# Dimension: D8 — 分隔线结构 (meta)
# Change: 单列均匀非背景色分隔线，贯穿整个网格高度，将左侧精灵区与右侧框架区分开
#   → 分隔线为水平方向（上下分布）

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D8 variant
# a) Concrete behavior:
#    原题是 `left sprites | vertical divider | right frame`。这里改成
#    `top sprites / horizontal divider / bottom frame`。框架仍然是一个矩形，
#    仍然靠“左右边同色 => 行标记、上下边同色 => 列标记”来决定印刻中心。
#    小例子：原题里某个精灵颜色 4 若在框左右边的第 8 行都有标记、上下边的第 11 列
#    都有标记，则会被印到右侧框内部的 (8,11)。变体里完全同理，只是精灵来源区不再
#    是框左边，而是框上边；solver 因此要把“在 fc0 左侧找精灵 / 找 divider 列”改成
#    “在 fr0 上方找精灵 / 找 divider 行”。
# b) Complexity calibration:
#    只改 D8，不改任务主规则。为避免退化成“看起来还是左右题”，这里让框架横向铺满
#    下半区，精灵在顶部横向排布。由于总高只有 10-15，框内部高度会比原题更紧，因此
#    row marker 数量保持 1-2，但生成时允许在行空间不足时退回到 1；列方向更宽，仍保留
#    1-2 个列标记。这样既保留笛卡尔积复制，也避免因为高度太小导致冲突。
# c) Hidden-assumption review:
#    - largest_enclosed_component_bbox 依赖“框内部背景区是最大且不接触外边界的背景块”。
#      变体中这仍成立，因为顶部精灵区和外部背景都接触边界，只有下方框内部是封闭的。
#    - 原 solver 的 divider 检测按列扫描，并默认 sprite 区在 fc0 左边；这在变体中失效，
#      必须改为按行扫描，并把 sprite 搜索区域改成 fr0 上方。
#    - 原 generator 把 movers top-to-bottom 堆在窄竖条里；变体若照搬会超出总高度约束，
#      因此改成在顶部横向打包，但仍给每个精灵一个 0-1 行的垂直偏移，让 bottom-to-top
#      覆盖顺序仍然由可见位置决定。
# d) Placement legality:
#    - 顶部 strip 高度固定为 3-4，精灵高度不超过 strip。
#    - 下方 frame 宽度等于整张图宽度，内部列范围固定是 1..W-2，所有中心点都按 sprite
#      bbox 反推可放置范围，保证印刻后不会整体越界。
#    - 同色/异色 marker 都不共享行或列，避免目标中心歧义。
# e) Visual observability:
#    人类可以直接看到：精灵来源区从“左边竖条”变成“上边横条”，分隔线从竖线变成横线，
#    但框边缘配色和框内复制规则保持一致，因此新旧差异非常显眼且单一。
# f) Visual cleanliness:
#    顶部只放精灵，不叠加框元素；下方只放框。divider 是一整行单色，区域边界清楚。
# g) Color inference:
#    颜色仍然只是“精灵颜色 = 对应 marker 颜色 = 印刻颜色”，没有新增颜色映射。
#
# New solver assumptions introduced by this variant and guaranteed by the generator:
# - 恰有一个横向 divider 行位于 frame 之上，且几乎整行都为同一非背景色。
# - 所有可移动精灵都位于 frame 上方；frame 本身横向铺满底部，没有左右额外边框外扩区。
# - 顶部精灵的垂直位置允许 main() 的 bottom-to-top 排序有意义。


def largest_enclosed_component_bbox(mask: np.ndarray, *, connectivity: int = 4) -> tuple[int, int, int, int] | None:
    """
    Local helper: bbox of the largest connected component in `mask`
    that does NOT touch the grid border.
    """
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    D8 variant:
    - sprites are in a top strip
    - a uniform divider row separates them from the lower frame
    - row/col markers on the frame edges still define stamp centers
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    inner_bbox = largest_enclosed_component_bbox(g == int(bg), connectivity=4)
    assert inner_bbox is not None
    ir0, ic0, ir1, ic1 = map(int, inner_bbox)
    fr0, fc0, fr1, fc1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)

    # In the variant the divider is horizontal, so scan rows above the frame.
    divider_mask = np.zeros((H, W), dtype=bool)
    for r in range(0, max(0, int(fr0))):
        rowv = g[int(r), :]
        non = rowv[rowv != int(bg)]
        if int(len(non)) >= int(W - 1) and len(set(map(int, non))) == 1:
            divider_mask[int(r), :] = True

    top_rows = np.arange(H)[:, None] < int(fr0)

    sprites = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        mask = (g == int(col)) & top_rows & (~divider_mask)
        if not bool(mask.any()):
            continue
        slab, sn = label_mask(mask, connectivity=8)
        for sid in range(1, int(sn) + 1):
            pts = np.argwhere(slab == int(sid))
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            m = slab[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(sid)
            sprites.append({"color": int(col), "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1), "mask": m})

    move_colors = {int(s["color"]) for s in sprites}

    row_marks: dict[int, list[int]] = defaultdict(list)
    col_marks: dict[int, list[int]] = defaultdict(list)
    for r in range(int(ir0), int(ir1) + 1):
        cl = int(g[int(r), int(fc0)])
        cr = int(g[int(r), int(fc1)])
        if cl == cr and cl != int(bg) and cl in move_colors:
            row_marks[int(cl)].append(int(r))
    for c in range(int(ic0), int(ic1) + 1):
        ct = int(g[int(fr0), int(c)])
        cb = int(g[int(fr1), int(c)])
        if ct == cb and ct != int(bg) and ct in move_colors:
            col_marks[int(ct)].append(int(c))

    targets: dict[int, list[tuple[int, int]]] = {}
    for col in set(row_marks.keys()) & set(col_marks.keys()):
        pts = {(int(r), int(c)) for r in row_marks[int(col)] for c in col_marks[int(col)]}
        targets[int(col)] = sorted(pts)

    out = g.copy()
    movers = [s for s in sprites if int(s["color"]) in targets]

    for s in movers:
        r0, c0 = int(s["r0"]), int(s["c0"])
        for rr, cc in np.argwhere(s["mask"]):
            out[int(r0) + int(rr), int(c0) + int(cc)] = int(bg)

    # Same rule as the source task: lower sprites stamp first, higher ones cover later.
    movers.sort(key=lambda s: (-((int(s["r0"]) + int(s["r1"])) / 2.0), int(s["c0"])))

    for s in movers:
        col = int(s["color"])
        cr = (int(s["r0"]) + int(s["r1"])) // 2
        cc = (int(s["c0"]) + int(s["c1"])) // 2
        pix = [(int(s["r0"]) + int(rr), int(s["c0"]) + int(cc2)) for rr, cc2 in np.argwhere(s["mask"])]

        for tr, tc in targets[int(col)]:
            dr = int(tr - cr)
            dc = int(tc - cc)
            for r, c in pix:
                rr = int(r + dr)
                cc3 = int(c + dc)
                if int(ir0) <= rr <= int(ir1) and int(ic0) <= cc3 <= int(ic1):
                    out[rr, cc3] = int(col)

    return out


def _pick_spaced(candidates: list[int], n: int, min_spacing: int) -> list[int] | None:
    """Pick up to n values from candidates with at least min_spacing between any two."""
    if not candidates:
        return None
    if n == 1:
        return [random.choice(candidates)]
    random.shuffle(candidates)
    picked: list[int] = []
    for v in candidates:
        if all(abs(v - p) >= min_spacing for p in picked):
            picked.append(v)
            if len(picked) == n:
                return picked
    return picked if picked else None


def _make_sprite(color: int, max_h: int, max_w: int) -> np.ndarray | None:
    """Generate a single-color 8-connected sprite cropped to its bounding box."""
    sh = min(random.choice([3, 3, 3, 4, 5]), max_h)
    sw = min(random.choice([3, 3, 3, 4, 5]), max_w)
    if sh < 1 or sw < 1:
        return None
    n_pixels = random.randint(3, min(9, sh * sw))
    pixels = {(sh // 2, sw // 2)}
    r, c = sh // 2, sw // 2
    for _ in range(n_pixels * 30):
        if len(pixels) >= n_pixels:
            break
        dr, dc = random.choice(
            [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        nr, nc = r + dr, c + dc
        if 0 <= nr < sh and 0 <= nc < sw:
            pixels.add((nr, nc))
            r, c = nr, nc
    rows = [p[0] for p in pixels]
    cols = [p[1] for p in pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sprite = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for pr, pc in pixels:
        sprite[pr - min_r, pc - min_c] = color
    return sprite


def _pick_marks(avail: list[int], preferred_n: int, min_spacing: int) -> list[int] | None:
    marks = _pick_spaced(list(avail), preferred_n, min_spacing)
    if marks or preferred_n == 1:
        return marks
    return _pick_spaced(list(avail), 1, min_spacing)


def _pack_top_items(items: list[tuple[int, np.ndarray, bool]], strip_h: int, width: int) -> list[tuple[int, np.ndarray, bool, int, int]] | None:
    """
    Pack sprites left-to-right in the top strip.
    Each gets a small vertical offset so the solver's bottom-to-top ordering remains visible.
    """
    widths = [int(sp.shape[1]) for _, sp, _ in items]
    if sum(widths) > width:
        return None

    placed: list[tuple[int, np.ndarray, bool, int, int]] = []
    cursor = 0
    for idx, (col, sp, is_distractor) in enumerate(items):
        sh, sw = map(int, sp.shape)
        if sh > strip_h:
            return None
        remain_width = sum(widths[idx + 1 :])
        max_start = int(width - sw - remain_width)
        if cursor > max_start:
            return None
        c_off = random.randint(cursor, max_start)
        r_off = random.randint(0, strip_h - sh)
        placed.append((int(col), sp, bool(is_distractor), int(r_off), int(c_off)))
        cursor = int(c_off + sw + random.choice([0, 1, 1]))
    return placed


def _gen_attempt() -> np.ndarray | None:
    bg = 0

    W = random.randint(14, 20)
    sprite_area_h = random.randint(3, 4)
    interior_h = random.randint(5, 9)
    H = sprite_area_h + 1 + interior_h + 2
    if not 10 <= H <= 15:
        return None

    div_row = sprite_area_h
    fr0, fr1 = div_row + 1, H - 1
    fc0, fc1 = 0, W - 1
    ir0, ic0 = fr0 + 1, 1
    ir1, ic1 = H - 2, W - 2
    interior_w = ic1 - ic0 + 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    n_movers = 2 if interior_h <= 6 else random.randint(2, 3)
    n_distractors = random.choice([0, 0, 1])
    idx = 0
    divider_color = all_colors[idx]
    idx += 1
    frame_base = all_colors[idx]
    idx += 1
    mover_colors = all_colors[idx : idx + n_movers]
    idx += n_movers
    distractor_colors = all_colors[idx : idx + n_distractors]

    sprite_max_w = 4 if W <= 16 else 5

    sprites: list[np.ndarray] = []
    for mc in mover_colors:
        s = _make_sprite(mc, sprite_area_h, sprite_max_w)
        if s is None:
            return None
        sprites.append(s)

    mover_targets: dict[int, tuple[list[int], list[int]]] = {}
    occupied_rows: set[int] = set()
    occupied_cols: set[int] = set()
    for mc, sp in zip(mover_colors, sprites):
        sh, sw = map(int, sp.shape)
        min_row = int(ir0 + sh // 2)
        max_row = int(ir1 - (sh - 1 - sh // 2))
        min_col = int(ic0 + sw // 2)
        max_col = int(ic1 - (sw - 1 - sw // 2))
        if min_row > max_row or min_col > max_col:
            return None

        avail_rows = [r for r in range(min_row, max_row + 1) if r not in occupied_rows]
        desired_rows = 2 if random.random() < 0.35 else 1
        row_marks = _pick_marks(avail_rows, desired_rows, max(1, sh))
        if not row_marks:
            return None

        avail_cols = [c for c in range(min_col, max_col + 1) if c not in occupied_cols]
        desired_cols = 2 if random.random() < 0.6 else 1
        col_marks = _pick_marks(avail_cols, desired_cols, max(1, sw))
        if not col_marks:
            return None

        mover_targets[mc] = (sorted(row_marks), sorted(col_marks))
        occupied_rows.update(row_marks)
        occupied_cols.update(col_marks)

    inp = np.full((H, W), bg, dtype=int)
    inp[int(div_row), :] = int(divider_color)

    edge_row = np.full(W, frame_base, dtype=int)
    for mc, (_, cmarks) in mover_targets.items():
        for cc in cmarks:
            edge_row[int(cc)] = int(mc)
    inp[int(fr0), :] = edge_row
    inp[int(fr1), :] = edge_row

    for r in range(int(ir0), int(ir1) + 1):
        val = int(frame_base)
        for mc, (rmarks, _) in mover_targets.items():
            if int(r) in rmarks:
                val = int(mc)
                break
        inp[int(r), int(fc0)] = val
        inp[int(r), int(fc1)] = val

    items: list[tuple[int, np.ndarray, bool]] = [(mc, sp, False) for mc, sp in zip(mover_colors, sprites)]
    for dc_color in distractor_colors:
        ds = _make_sprite(int(dc_color), sprite_area_h, sprite_max_w)
        if ds is None:
            return None
        items.append((int(dc_color), ds, True))

    placed = _pack_top_items(items, sprite_area_h, W)
    if placed is None:
        return None

    for col, sp, _is_distractor, r_off, c_off in placed:
        sh, sw = map(int, sp.shape)
        for dr in range(sh):
            for dc in range(sw):
                if int(sp[dr, dc]) != bg:
                    inp[int(r_off + dr), int(c_off + dc)] = int(col)

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg:
        return None

    try:
        output = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
